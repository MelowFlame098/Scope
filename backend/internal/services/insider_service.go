package services

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type InsiderTrade struct {
	ID           primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Ticker       string             `bson:"Ticker" json:"ticker"`
	Owner        string             `bson:"Owner" json:"owner"`
	Relationship string             `bson:"Relationship" json:"relationship"`
	Date         string             `bson:"Date" json:"date"`
	Transaction  string             `bson:"Transaction" json:"transaction"`
	Cost         string             `bson:"Cost" json:"cost"`
	Shares       string             `bson:"#Shares" json:"shares"`
	Value        string             `bson:"Value ($)" json:"value"`
	TotalShares  string             `bson:"#Shares Total" json:"total_shares"`
	SECForm4     string             `bson:"SEC Form 4" json:"sec_form_4"`
	FetchedAt    time.Time          `bson:"fetched_at" json:"fetched_at"`
}

type InsiderService struct {
	collection *mongo.Collection
}

func NewInsiderService(db *mongo.Database) *InsiderService {
	return &InsiderService{
		collection: db.Collection("insider_trades"),
	}
}

func (s *InsiderService) GetInsiderTrades(ctx context.Context, limit int64) ([]InsiderTrade, error) {
	// Sort by fetched_at desc (or maybe Date if parsed)
	opts := options.Find().SetSort(bson.D{{Key: "fetched_at", Value: -1}}).SetLimit(limit)
	
	cursor, err := s.collection.Find(ctx, bson.M{}, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var trades []InsiderTrade
	if err := cursor.All(ctx, &trades); err != nil {
		return nil, err
	}
	return trades, nil
}
