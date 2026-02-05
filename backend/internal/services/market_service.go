package services

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/shopspring/decimal"
)

type OrderBook struct {
	Symbol    string  `json:"symbol"`
	Bids      []Order `json:"bids"`
	Asks      []Order `json:"asks"`
	Timestamp int64   `json:"timestamp"`
}

type Order struct {
	Price  decimal.Decimal `json:"price"`
	Amount decimal.Decimal `json:"amount"`
}

type MarketService struct {
	redisClient *redis.Client
	ctx         context.Context
	mu          sync.RWMutex
	movers      map[string]StockPrice
	candleCache map[string]map[string][]Candle // symbol -> timeframe -> candles
}

type Candle struct {
	Timestamp int64           `json:"timestamp"`
	Open      decimal.Decimal `json:"open"`
	High      decimal.Decimal `json:"high"`
	Low       decimal.Decimal `json:"low"`
	Close     decimal.Decimal `json:"close"`
}

type StockPrice struct {
	Symbol        string          `json:"symbol"`
	Name          string          `json:"name"`
	Price         decimal.Decimal `json:"price"`
	ChangePercent decimal.Decimal `json:"change_percent"`
	Timestamp     time.Time       `json:"timestamp"`
}

func NewMarketService(redisClient *redis.Client) *MarketService {
	return &MarketService{
		redisClient: redisClient,
		ctx:         context.Background(),
		movers:      make(map[string]StockPrice),
		candleCache: make(map[string]map[string][]Candle),
	}
}

// GetPrice retrieves the latest price for a symbol from Redis
func (s *MarketService) GetPrice(symbol string) (*StockPrice, error) {
	key := fmt.Sprintf("market:price:%s", symbol)
	val, err := s.redisClient.Get(s.ctx, key).Result()
	if err == redis.Nil {
		return nil, fmt.Errorf("price not found for symbol: %s", symbol)
	} else if err != nil {
		return nil, err
	}

	var price StockPrice
	if err := json.Unmarshal([]byte(val), &price); err != nil {
		return nil, err
	}

	return &price, nil
}

// UpdatePrice sets the latest price for a symbol in Redis
func (s *MarketService) UpdatePrice(symbol string, name string, price decimal.Decimal, changePercent decimal.Decimal) error {
	stockPrice := StockPrice{
		Symbol:        symbol,
		Name:          name,
		Price:         price,
		ChangePercent: changePercent,
		Timestamp:     time.Now(),
	}

	val, err := json.Marshal(stockPrice)
	if err != nil {
		return err
	}

	// Update in-memory cache
	s.mu.Lock()
	s.movers[symbol] = stockPrice
	s.mu.Unlock()

	key := fmt.Sprintf("market:price:%s", symbol)
	pipe := s.redisClient.Pipeline()
	pipe.Set(s.ctx, key, val, 24*time.Hour)

	// Add to Sorted Set for Movers (Score = ChangePercent)
	changeFloat, _ := changePercent.Float64()
	pipe.ZAdd(s.ctx, "market:movers", redis.Z{
		Score:  changeFloat,
		Member: symbol,
	})

	_, err = pipe.Exec(s.ctx)
	return err
}

// GetOrderBook retrieves the current order book from Redis
func (s *MarketService) GetOrderBook(symbol string) (*OrderBook, error) {
	key := fmt.Sprintf("market:orderbook:%s", symbol)
	val, err := s.redisClient.Get(context.Background(), key).Result()
	if err != nil {
		return nil, err
	}

	var book OrderBook
	// Assuming we store JSON in Redis for simplicity
	// In production, might use Sorted Sets for bids/asks
	if err := json.Unmarshal([]byte(val), &book); err != nil {
		return nil, err
	}
	return &book, nil
}

// UpdateOrderBook updates the order book in Redis
func (s *MarketService) UpdateOrderBook(symbol string, book *OrderBook) error {
	key := fmt.Sprintf("market:orderbook:%s", symbol)
	data, err := json.Marshal(book)
	if err != nil {
		return err
	}
	return s.redisClient.Set(context.Background(), key, data, 0).Err()
}

// GetTopMovers and GetWorstMovers logic is in market_movers.go (assumed based on previous reads)
// But I need to check if I overwrote them? No, previous `SearchReplace` only modified `getPricesForSymbols`.
// Wait, `market_movers.go` exists. I should not duplicate logic if it's there.
// But this file `market_service.go` seems to be the main one.

// StartMarketSimulator simulates real-time price updates for demo purposes
// In production, this would be replaced by a real feed handler (e.g., WebSocket to Polygon.io/Alpaca)
func (s *MarketService) StartMarketSimulator() {
	ticker := time.NewTicker(1 * time.Second)

	type StockInfo struct {
		Name  string
		Price decimal.Decimal
	}

	// Initial prices
	stocks := map[string]StockInfo{
		"AAPL":  {Name: "Apple Inc.", Price: decimal.NewFromFloat(150.00)},
		"GOOGL": {Name: "Alphabet Inc.", Price: decimal.NewFromFloat(2800.00)},
		"TSLA":  {Name: "Tesla Inc.", Price: decimal.NewFromFloat(900.00)},
		"MSFT":  {Name: "Microsoft Corp.", Price: decimal.NewFromFloat(300.00)},
		"AMZN":  {Name: "Amazon.com Inc.", Price: decimal.NewFromFloat(3400.00)},
		// NASDAQ Gainers
		"MARA": {Name: "Marathon Digital", Price: decimal.NewFromFloat(25.00)},
		"RIVN": {Name: "Rivian Automotive", Price: decimal.NewFromFloat(15.00)},
		"DKNG": {Name: "DraftKings Inc.", Price: decimal.NewFromFloat(40.00)},
		"LCID": {Name: "Lucid Group", Price: decimal.NewFromFloat(4.00)},
		"PLUG": {Name: "Plug Power", Price: decimal.NewFromFloat(3.50)},
		"SOFI": {Name: "SoFi Technologies", Price: decimal.NewFromFloat(8.00)},
		// NASDAQ Losers
		"PTON": {Name: "Peloton Interactive", Price: decimal.NewFromFloat(5.00)},
		"COIN": {Name: "Coinbase Global", Price: decimal.NewFromFloat(150.00)},
		"ZM":   {Name: "Zoom Video", Price: decimal.NewFromFloat(70.00)},
		"ROKU": {Name: "Roku Inc.", Price: decimal.NewFromFloat(65.00)},
		"DOCU": {Name: "DocuSign", Price: decimal.NewFromFloat(55.00)},
		"SNOW": {Name: "Snowflake Inc.", Price: decimal.NewFromFloat(160.00)},
	}

	// Track initial prices to calculate change
	initialPrices := make(map[string]decimal.Decimal)
	for k, v := range stocks {
		initialPrices[k] = v.Price
	}

	go func() {
		for range ticker.C {
			for symbol, info := range stocks {
				// Random walk: -0.5% to +0.5% change
				changePct := (rand.Float64() - 0.5) / 100.0
				change := info.Price.Mul(decimal.NewFromFloat(changePct))
				newPrice := info.Price.Add(change)

				// Update local map
				stocks[symbol] = StockInfo{Name: info.Name, Price: newPrice}

				// Calculate Total Change Percent from Initial
				totalChangePct := newPrice.Sub(initialPrices[symbol]).Div(initialPrices[symbol]).Mul(decimal.NewFromFloat(100))

				if err := s.UpdatePrice(symbol, info.Name, newPrice, totalChangePct); err != nil {
					log.Printf("Error updating price for %s: %v", symbol, err)
				}
			}
		}
	}()
}

// GetCandles generates simulated candlestick data
func (s *MarketService) GetCandles(symbol string, timeframe string) ([]Candle, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Initialize nested map if not exists
	if _, ok := s.candleCache[symbol]; !ok {
		s.candleCache[symbol] = make(map[string][]Candle)
	}

	// 1. Determine interval and count
	var interval time.Duration
	var count int

	switch timeframe {
	case "1m":
		interval = time.Minute
		count = 60
	case "5m":
		interval = 5 * time.Minute
		count = 60
	case "15m":
		interval = 15 * time.Minute
		count = 60
	case "30m":
		interval = 30 * time.Minute
		count = 48
	case "1h":
		interval = time.Hour
		count = 24
	case "12h":
		interval = 12 * time.Hour
		count = 30
	case "24h":
		interval = 24 * time.Hour
		count = 30
	case "3d":
		interval = 72 * time.Hour // 3 days
		count = 30
	case "1w":
		interval = 7 * 24 * time.Hour
		count = 52
	case "1mo":
		interval = 30 * 24 * time.Hour
		count = 24
	case "3mo":
		interval = 90 * 24 * time.Hour
		count = 12
	case "6mo":
		interval = 7 * 24 * time.Hour // Weekly candles for 6 months
		count = 26
	case "1y":
		interval = 7 * 24 * time.Hour // Weekly candles for 1 year
		count = 52
	case "ytd":
		interval = 7 * 24 * time.Hour // Weekly candles for YTD
		// Calculate weeks since start of year
		ytd := time.Now().Sub(time.Date(time.Now().Year(), 1, 1, 0, 0, 0, 0, time.UTC))
		count = int(ytd.Hours() / (24 * 7))
		if count < 1 {
			count = 1
		}
	default:
		interval = 5 * time.Minute
		count = 60
	}

	// 2. Check cache
	cached, exists := s.candleCache[symbol][timeframe]
	if !exists || len(cached) == 0 {
		// Generate initial history
		// Get current price to anchor the simulation
		currentPrice, err := s.GetPrice(symbol)
		if err != nil {
			// If not found, use a default
			price := decimal.NewFromFloat(150.00)
			currentPrice = &StockPrice{Price: price}
		}

		candles := make([]Candle, count)
		now := time.Now().Truncate(interval)
		price := currentPrice.Price

		for i := count - 1; i >= 0; i-- {
			volatility := 0.002
			if interval > time.Hour {
				volatility = 0.01
			}

			// Deterministic seed based on timestamp for historical stability
			// But for initial generation, random is fine as long as we cache it.
			// However, to be consistent if server restarts, we could seed.
			// For now, random walk backwards.

			changePct := (rand.Float64() - 0.5) * 2 * volatility
			open := price.Div(decimal.NewFromFloat(1 + changePct))
			high := decimal.Max(open, price).Mul(decimal.NewFromFloat(1 + rand.Float64()*volatility))
			low := decimal.Min(open, price).Mul(decimal.NewFromFloat(1 - rand.Float64()*volatility))

			candles[i] = Candle{
				Timestamp: now.UnixMilli(),
				Open:      open,
				High:      high,
				Low:       low,
				Close:     price,
			}

			price = open
			now = now.Add(-interval)
		}

		s.candleCache[symbol][timeframe] = candles
		return candles, nil
	}

	// 3. Update cached candles
	// Check if we need to start a new candle or update the latest one
	lastCandle := &cached[len(cached)-1]
	now := time.Now()
	currentIntervalStart := now.Truncate(interval).UnixMilli()

	if currentIntervalStart > lastCandle.Timestamp {
		// Finalize old candle (nothing to do really, it's already in cache)
		// Start new candle
		newOpen := lastCandle.Close
		newCandle := Candle{
			Timestamp: currentIntervalStart,
			Open:      newOpen,
			High:      newOpen,
			Low:       newOpen,
			Close:     newOpen,
		}

		// Shift array: remove oldest, add newest
		// Or if YTD, just append?
		// For fixed count arrays:
		if timeframe != "ytd" {
			cached = append(cached[1:], newCandle)
		} else {
			cached = append(cached, newCandle)
		}
	} else {
		// Update current candle with live simulation
		volatility := 0.0005 // Smaller volatility for live updates
		changePct := (rand.Float64() - 0.5) * 2 * volatility
		newClose := lastCandle.Close.Mul(decimal.NewFromFloat(1 + changePct))

		// Update High/Low
		if newClose.GreaterThan(lastCandle.High) {
			lastCandle.High = newClose
		}
		if newClose.LessThan(lastCandle.Low) {
			lastCandle.Low = newClose
		}
		lastCandle.Close = newClose
	}

	s.candleCache[symbol][timeframe] = cached

	// Return a copy to avoid race conditions if caller modifies it (though we return by value slice header, elements are structs)
	result := make([]Candle, len(cached))
	copy(result, cached)

	return result, nil
}
