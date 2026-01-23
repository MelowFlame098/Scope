package main

import (
	"log"
	"scope-backend/internal/config"
	"scope-backend/internal/database"
	"scope-backend/internal/repository"
	"scope-backend/internal/server"
	"scope-backend/internal/services"
	"scope-backend/internal/worker"
)

func main() {
	// Load Configuration
	cfg, err := config.LoadConfig()
	if err != nil {
		log.Fatalf("Failed to load config: %v", err)
	}

	// Initialize Database (PostgreSQL)
	db, err := database.ConnectDB(cfg.Database)
	if err != nil {
		log.Printf("Warning: Failed to connect to database: %v", err)
	}

	// Initialize MongoDB
	mongoDB, err := database.ConnectMongo(cfg.Mongo)
	if err != nil {
		log.Printf("Warning: Failed to connect to MongoDB: %v", err)
	}

	// Initialize Redis
	rdb := database.ConnectRedis()

	// Initialize Repositories and Services
	var authService *services.AuthService
	if db != nil {
		userRepo := repository.NewUserRepository(db)
		authService = services.NewAuthService(userRepo)
	}

	marketService := services.NewMarketService(rdb)
	marketService.StartMarketSimulator()

	var newsService *services.NewsService
	if mongoDB != nil {
		newsService = services.NewNewsService(mongoDB)
	}

	// Initialize Worker Pool
	// In a real production app, you might run the worker server in a separate process
	// or use a flag to decide whether to run as server or worker or both.
	// Here we run both for simplicity.
	
	// 1. Start Worker Server (Consumer)
	go worker.StartWorkerServer(rdb)

	// 2. Initialize Task Distributor (Producer)
	taskDistributor := worker.NewTaskDistributor(rdb)

	// Initialize HTTP Server
	server := server.NewServer(cfg, db, mongoDB, rdb, taskDistributor, authService, marketService, newsService)
	if err := server.Run(); err != nil {
		log.Fatalf("Failed to run server: %v", err)
	}
}
