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

