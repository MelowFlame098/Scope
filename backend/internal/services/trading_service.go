package services

import (
	"context"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"

	"github.com/shopspring/decimal"
	"go.mongodb.org/mongo-driver/bson/primitive"
)

type TradeOrder struct {
	ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Symbol    string             `bson:"symbol" json:"symbol"`
	Side      string             `bson:"side" json:"side"` // BUY or SELL
	Quantity  int                `bson:"quantity" json:"quantity"`
	Price     decimal.Decimal    `bson:"price" json:"price"`
	Status    string             `bson:"status" json:"status"` // FILLED, FAILED
	CreatedAt time.Time          `bson:"created_at" json:"created_at"`
}

type TradingService struct {
	screenerService *ScreenerService
	marketService   *MarketService
}

func NewTradingService(screener *ScreenerService, market *MarketService) *TradingService {
	return &TradingService{
		screenerService: screener,
		marketService:   market,
	}
}

func (s *TradingService) RunAutomatedStrategy(ctx context.Context) error {
	log.Println("Running Automated Trading Strategy based on Screener Signals...")

	// 1. Get Top Gainers (Simulated strategy filter)
	results, err := s.screenerService.GetScreenerResults(ctx, "", 20)
	if err != nil {
		return fmt.Errorf("failed to get screener results: %w", err)
	}

	for _, stock := range results {
		// Parse Change %
		changeStr := strings.TrimSuffix(stock.Change, "%")
		change, err := strconv.ParseFloat(changeStr, 64)
		if err != nil {
			continue // Skip if parse error
		}

		// STRATEGY: Momentum Buy
		// If stock is up > 3% today, Buy 10 shares
		if change > 3.0 {
			// Get current real-time price
			priceData, err := s.marketService.GetPrice(stock.Ticker)
			if err != nil {
				log.Printf("Failed to get price for %s: %v", stock.Ticker, err)
				continue
			}

			// Execute Order
			err = s.ExecuteOrder(ctx, stock.Ticker, "BUY", 10, priceData.Price)
			if err != nil {
				log.Printf("Order execution failed: %v", err)
			}
		}
	}
	return nil
}

func (s *TradingService) ExecuteOrder(ctx context.Context, symbol, side string, quantity int, price decimal.Decimal) error {
	// In a real system, this would:
	// 1. Validate user balance (omitted for system trade)
	// 2. Create Order record in DB
	// 3. Update Portfolio

	order := TradeOrder{
		ID:        primitive.NewObjectID(),
		Symbol:    symbol,
		Side:      side,
		Quantity:  quantity,
		Price:     price,
		Status:    "FILLED",
		CreatedAt: time.Now(),
	}

	// For now, just log it as "Connected to Execution Service"
	log.Printf("[ORDER EXECUTION] %s: %s %d shares of %s at $%s | Status: %s",
		time.Now().Format(time.RFC3339), side, quantity, symbol, price.String(), order.Status)

	return nil
}
