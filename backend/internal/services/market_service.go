package services

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
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
}

type StockPrice struct {
	Symbol    string          `json:"symbol"`
	Price     decimal.Decimal `json:"price"`
	Timestamp time.Time       `json:"timestamp"`
}

func NewMarketService(redisClient *redis.Client) *MarketService {
	return &MarketService{
		redisClient: redisClient,
		ctx:         context.Background(),
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
func (s *MarketService) UpdatePrice(symbol string, price decimal.Decimal) error {
	stockPrice := StockPrice{
		Symbol:    symbol,
		Price:     price,
		Timestamp: time.Now(),
	}

	val, err := json.Marshal(stockPrice)
	if err != nil {
		return err
	}

	key := fmt.Sprintf("market:price:%s", symbol)
	// Cache for 24 hours, though usually updates are frequent
	return s.redisClient.Set(s.ctx, key, val, 24*time.Hour).Err()
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

	// Initial prices
	stocks := map[string]decimal.Decimal{
		"AAPL":  decimal.NewFromFloat(150.00),
		"GOOGL": decimal.NewFromFloat(2800.00),
		"TSLA":  decimal.NewFromFloat(900.00),
		"MSFT":  decimal.NewFromFloat(300.00),
		"AMZN":  decimal.NewFromFloat(3400.00),
	}

	go func() {
		for range ticker.C {
			for symbol, price := range stocks {
				// Random walk: -0.5% to +0.5% change
				changePct := (rand.Float64() - 0.5) / 100.0
				change := price.Mul(decimal.NewFromFloat(changePct))
				newPrice := price.Add(change)

				// Update local map
				stocks[symbol] = newPrice

				// Push to Redis
				if err := s.UpdatePrice(symbol, newPrice); err != nil {
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
