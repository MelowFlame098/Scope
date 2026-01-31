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
				totalChange := newPrice.Sub(initialPrices[symbol])
				totalChangePct := totalChange.Div(initialPrices[symbol]).Mul(decimal.NewFromInt(100))

				// Push to Redis
				if err := s.UpdatePrice(symbol, info.Name, newPrice, totalChangePct); err != nil {
					log.Printf("Error updating price for %s: %v", symbol, err)
				}

				// Simulate Order Book Update
				spread := newPrice.Mul(decimal.NewFromFloat(0.001)) // 0.1% spread
				bids := []Order{
					{Price: newPrice.Sub(spread), Amount: decimal.NewFromInt(int64(rand.Intn(100) + 1))},
					{Price: newPrice.Sub(spread.Mul(decimal.NewFromFloat(2))), Amount: decimal.NewFromInt(int64(rand.Intn(100) + 1))},
				}
				asks := []Order{
					{Price: newPrice.Add(spread), Amount: decimal.NewFromInt(int64(rand.Intn(100) + 1))},
					{Price: newPrice.Add(spread.Mul(decimal.NewFromFloat(2))), Amount: decimal.NewFromInt(int64(rand.Intn(100) + 1))},
				}
				book := &OrderBook{
					Symbol:    symbol,
					Bids:      bids,
					Asks:      asks,
					Timestamp: time.Now().Unix(),
				}
				s.UpdateOrderBook(symbol, book)
			}
			// log.Println("Market Simulator: Prices updated")
		}
	}()
}
