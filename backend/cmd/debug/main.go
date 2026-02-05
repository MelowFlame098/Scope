package main

import (
	"context"
	"fmt"
	"log"
	"scope-backend/internal/services"
	"time"

	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

func main() {
	// Connect to MongoDB
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	clientOptions := options.Client().ApplyURI("mongodb://user:password@localhost:27017")
	client, err := mongo.Connect(ctx, clientOptions)
	if err != nil {
		log.Fatal(err)
	}

	err = client.Ping(ctx, nil)
	if err != nil {
		log.Fatal("Could not connect to MongoDB:", err)
	}
	fmt.Println("Connected to MongoDB")

	db := client.Database("scope_db")

	// Test Sector Service
	sectorService := services.NewSectorService(db)
	sectors, err := sectorService.GetSectorPerformance(ctx)
	if err != nil {
		fmt.Printf("SectorService Error: %v\n", err)
	} else {
		fmt.Printf("SectorService Success: %d records\n", len(sectors))
	}

	// Test Insider Service
	insiderService := services.NewInsiderService(db)
	trades, err := insiderService.GetInsiderTrades(ctx, 50)
	if err != nil {
		fmt.Printf("InsiderService Error: %v\n", err)
	} else {
		fmt.Printf("InsiderService Success: %d records\n", len(trades))
	}
}
