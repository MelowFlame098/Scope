package services

import (
	"context"
	"math"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type SectorPerformance struct {
	ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Name      string             `bson:"Name" json:"name"` // Sector Name
	Change    float64            `bson:"Change" json:"change"`
	Volume    float64            `bson:"Volume" json:"volume"`
	Stocks    string             `bson:"Stocks" json:"stocks"`
	MarketCap float64            `bson:"Market Cap" json:"market_cap"`
	PE        float64            `bson:"P/E" json:"pe"`
	FetchedAt time.Time          `bson:"fetched_at" json:"fetched_at"`
}

type SectorService struct {
	collection *mongo.Collection
}

func NewSectorService(db *mongo.Database) *SectorService {
	return &SectorService{
		collection: db.Collection("sector_performance"),
	}
}

func (s *SectorService) GetSectorPerformance(ctx context.Context) ([]SectorPerformance, error) {
	// Sort by Name or Change? Let's just return all
	opts := options.Find().SetSort(bson.D{{Key: "Name", Value: 1}})

	cursor, err := s.collection.Find(ctx, bson.M{}, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	results := []SectorPerformance{}
	if err := cursor.All(ctx, &results); err != nil {
		return nil, err
	}

	// If nil (no results), return empty slice instead of nil to avoid null issues in frontend
	if results == nil {
		results = []SectorPerformance{}
	}

	// Sanitize NaNs and Infs
	for i := range results {
		if math.IsNaN(results[i].Change) || math.IsInf(results[i].Change, 0) {
			results[i].Change = 0
		}
		if math.IsNaN(results[i].Volume) || math.IsInf(results[i].Volume, 0) {
			results[i].Volume = 0
		}
		if math.IsNaN(results[i].MarketCap) || math.IsInf(results[i].MarketCap, 0) {
			results[i].MarketCap = 0
		}
		if math.IsNaN(results[i].PE) || math.IsInf(results[i].PE, 0) {
			results[i].PE = 0
		}
	}

	return results, nil
}
