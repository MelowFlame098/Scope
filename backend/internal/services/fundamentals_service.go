package services

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type FundamentalsRecord struct {
	ID                primitive.ObjectID     `bson:"_id,omitempty" json:"id"`
	Ticker            string                 `bson:"ticker" json:"ticker"`
	Period            time.Time              `bson:"period" json:"period"`
	Timeframe         string                 `bson:"timeframe" json:"timeframe"`
	Revenue           float64                `bson:"revenue" json:"revenue"`
	NetIncome         float64                `bson:"net_income" json:"net_income"`
	TotalAssets       float64                `bson:"total_assets" json:"total_assets"`
	TotalLiabilities  float64                `bson:"total_liabilities" json:"total_liabilities"`
	OperatingCashflow float64                `bson:"operating_cashflow" json:"operating_cashflow"`
	Metrics           map[string]interface{} `bson:"metrics,omitempty" json:"metrics,omitempty"` // New field for extended metrics

	// --- NEW: Core Fundamentals Tables (YFinance) ---
	FinancialsAnnual      []map[string]interface{} `bson:"financials_annual,omitempty" json:"financials_annual,omitempty"`
	BalanceSheetAnnual    []map[string]interface{} `bson:"balance_sheet_annual,omitempty" json:"balance_sheet_annual,omitempty"`
	CashflowAnnual        []map[string]interface{} `bson:"cashflow_annual,omitempty" json:"cashflow_annual,omitempty"`
	FinancialsQuarterly   []map[string]interface{} `bson:"financials_quarterly,omitempty" json:"financials_quarterly,omitempty"`
	BalanceSheetQuarterly []map[string]interface{} `bson:"balance_sheet_quarterly,omitempty" json:"balance_sheet_quarterly,omitempty"`
	CashflowQuarterly     []map[string]interface{} `bson:"cashflow_quarterly,omitempty" json:"cashflow_quarterly,omitempty"`
	MajorHolders          interface{}              `bson:"major_holders,omitempty" json:"major_holders,omitempty"` // map or list
	InstitutionalHolders  []map[string]interface{} `bson:"institutional_holders,omitempty" json:"institutional_holders,omitempty"`

	FetchedAt         time.Time              `bson:"fetched_at" json:"fetched_at"`
}

type FundamentalsService struct {
	collection *mongo.Collection
}

func NewFundamentalsService(db *mongo.Database) *FundamentalsService {
	return &FundamentalsService{
		collection: db.Collection("fundamentals"),
	}
}

func (s *FundamentalsService) GetLatestByTimeframe(ctx context.Context, ticker, timeframe string) (*FundamentalsRecord, error) {
	filter := bson.M{
		"ticker":    ticker,
		"timeframe": timeframe,
	}

	opts := options.FindOne().SetSort(bson.D{{Key: "period", Value: -1}})

	var rec FundamentalsRecord
	err := s.collection.FindOne(ctx, filter, opts).Decode(&rec)
	if err == mongo.ErrNoDocuments {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	return &rec, nil
}

