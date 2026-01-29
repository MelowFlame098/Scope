package services

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type ScreenerResult struct {
	ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Ticker    string             `bson:"Ticker" json:"ticker"`
	Company   string             `bson:"Company" json:"company"`
	Sector    string             `bson:"Sector" json:"sector"`
	Industry  string             `bson:"Industry" json:"industry"`
	Country   string             `bson:"Country" json:"country"`
	MarketCap string             `bson:"Market Cap" json:"market_cap"`
	PE        string             `bson:"P/E" json:"pe"`
	Price     string             `bson:"Price" json:"price"`
	Change    string             `bson:"Change" json:"change"`
	Volume    string             `bson:"Volume" json:"volume"`
	Strategy  string             `bson:"strategy" json:"strategy"`
	FetchedAt time.Time          `bson:"fetched_at" json:"fetched_at"`
}

type ScreenerService struct {
	collection *mongo.Collection
}

func NewScreenerService(db *mongo.Database) *ScreenerService {
	return &ScreenerService{
		collection: db.Collection("screener_results"),
	}
}

func (s *ScreenerService) GetScreenerResults(ctx context.Context, strategy string, limit int64) ([]ScreenerResult, error) {
	filter := bson.M{}
	if strategy != "" {
		filter["strategy"] = strategy
	}
	
	// Sort by fetched_at desc to get latest batch
	opts := options.Find().SetSort(bson.D{{Key: "fetched_at", Value: -1}}).SetLimit(limit)
	
	cursor, err := s.collection.Find(ctx, filter, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var results []ScreenerResult
	if err := cursor.All(ctx, &results); err != nil {
		return nil, err
	}
	return results, nil
}
