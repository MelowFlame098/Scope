package services

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type SectorPerformance struct {
	ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Name      string             `bson:"Name" json:"name"` // Sector Name
	Change    string             `bson:"Change" json:"change"`
	Volume    string             `bson:"Volume" json:"volume"`
	Stocks    string             `bson:"Stocks" json:"stocks"`
	MarketCap string             `bson:"Market Cap" json:"market_cap"`
	PE        string             `bson:"P/E" json:"pe"`
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

	var results []SectorPerformance
	if err := cursor.All(ctx, &results); err != nil {
		return nil, err
	}
	return results, nil
}
