package main

import (
	"log"
	"scope-backend/internal/config"
	"scope-backend/internal/database"
	"scope-backend/internal/repository"
	"scope-backend/internal/server"
	"scope-backend/internal/services"
	"scope-backend/internal/worker"
	"time"
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
	rdb := database.ConnectRedis(cfg.Redis)

	// Initialize Repositories and Services
	var authService *services.AuthService
	if db != nil {
		userRepo := repository.NewUserRepository(db)
		authService = services.NewAuthService(userRepo)
	}

	marketService := services.NewMarketService(rdb)
	marketService.StartMarketSimulator()

	var newsService *services.NewsService
	var screenerService *services.ScreenerService
	var insiderService *services.InsiderService
	var sectorService *services.SectorService
	var fundamentalsService *services.FundamentalsService

	if mongoDB != nil {
		newsService = services.NewNewsService(mongoDB)
		screenerService = services.NewScreenerService(mongoDB)
		insiderService = services.NewInsiderService(mongoDB)
		sectorService = services.NewSectorService(mongoDB)
		fundamentalsService = services.NewFundamentalsService(mongoDB)
	}

	tradingService := services.NewTradingService(screenerService, marketService)

	// Initialize Worker Pool
	// In a real production app, you might run the worker server in a separate process
	// or use a flag to decide whether to run as server or worker or both.
	// Here we run both for simplicity.

	// 1. Start Worker Server (Consumer)
	go worker.StartWorkerServer(rdb, tradingService)

	// 2. Initialize Task Distributor (Producer)
	taskDistributor := worker.NewTaskDistributor(rdb)

	// Schedule Automated Trading Task (Every 1 minute)
	go func() {
		// Wait a bit for server to start
		time.Sleep(10 * time.Second)
		log.Println("Starting Automated Trading Scheduler...")

		// Initial run
		if err := taskDistributor.DistributeTaskAutomatedTrading(); err != nil {
			log.Printf("Failed to schedule initial automated trading task: %v", err)
		}

		ticker := time.NewTicker(1 * time.Minute)
		defer ticker.Stop()
		for range ticker.C {
			if err := taskDistributor.DistributeTaskAutomatedTrading(); err != nil {
				log.Printf("Failed to schedule automated trading task: %v", err)
			}
		}
	}()

	// Initialize HTTP Server
	server := server.NewServer(cfg, db, mongoDB, rdb, taskDistributor, authService, marketService, newsService, screenerService, insiderService, sectorService, fundamentalsService)
	if err := server.Run(":" + cfg.Server.Port); err != nil {
		log.Fatalf("Failed to run server: %v", err)
	}
}
