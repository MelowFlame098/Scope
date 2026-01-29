package services

import (
	"context"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

type NewsArticle struct {
	ID        primitive.ObjectID `bson:"_id,omitempty" json:"id"`
	Title     string             `bson:"title" json:"title"`
	Content   string             `bson:"content" json:"content"`
	Source    string             `bson:"source" json:"source"`
	URL       string             `bson:"url" json:"url"`
	Timestamp time.Time          `bson:"timestamp" json:"timestamp"`
	Sentiment float64            `bson:"sentiment" json:"sentiment"`
	Tags      []string           `bson:"tags" json:"tags"`
}

type NewsService struct {
	collection *mongo.Collection
}

func NewNewsService(db *mongo.Database) *NewsService {
	return &NewsService{
		collection: db.Collection("news"),
	}
}

func (s *NewsService) CreateArticle(ctx context.Context, article *NewsArticle) error {
	article.Timestamp = time.Now()
	_, err := s.collection.InsertOne(ctx, article)
	return err
}

func (s *NewsService) GetLatestNews(ctx context.Context, limit int64) ([]NewsArticle, error) {
	opts := options.Find().SetSort(bson.D{{Key: "timestamp", Value: -1}}).SetLimit(limit)
	cursor, err := s.collection.Find(ctx, bson.M{}, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var articles []NewsArticle
	if err := cursor.All(ctx, &articles); err != nil {
		return nil, err
	}
	return articles, nil
}

func (s *NewsService) GetNewsByTag(ctx context.Context, tag string, limit int64) ([]NewsArticle, error) {
	filter := bson.M{"tags": tag}
	opts := options.Find().SetSort(bson.D{{Key: "timestamp", Value: -1}}).SetLimit(limit)
	
	cursor, err := s.collection.Find(ctx, filter, opts)
	if err != nil {
		return nil, err
	}
	defer cursor.Close(ctx)

	var articles []NewsArticle
	if err := cursor.All(ctx, &articles); err != nil {
		return nil, err
	}
	return articles, nil
}
