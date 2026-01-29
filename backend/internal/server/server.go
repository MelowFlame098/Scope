package server

import (
	"scope-backend/internal/config"
	"scope-backend/internal/services"
	"scope-backend/internal/worker"

	"strconv"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"go.mongodb.org/mongo-driver/mongo"
	"gorm.io/gorm"
)

type Server struct {
	cfg             *config.Config
	db              *gorm.DB
	mongoDB         *mongo.Database
	router          *gin.Engine
	authService     *services.AuthService
	taskDistributor *worker.TaskDistributor
	marketService   *services.MarketService
	newsService     *services.NewsService
	screenerService *services.ScreenerService
	insiderService  *services.InsiderService
	sectorService   *services.SectorService
	redisClient     *redis.Client
}

func NewServer(cfg *config.Config, db *gorm.DB, mongoDB *mongo.Database, redisClient *redis.Client, taskDistributor *worker.TaskDistributor, authService *services.AuthService, marketService *services.MarketService, newsService *services.NewsService, screenerService *services.ScreenerService, insiderService *services.InsiderService, sectorService *services.SectorService) *Server {
	s := &Server{
		cfg:             cfg,
		db:              db,
		mongoDB:         mongoDB,
		redisClient:     redisClient,
		taskDistributor: taskDistributor,
		authService:     authService,
		marketService:   marketService,
		newsService:     newsService,
		screenerService: screenerService,
		insiderService:  insiderService,
		sectorService:   sectorService,
		router:          gin.Default(),
	}

	s.SetupRoutes()
	return s
}

func (s *Server) SetupRoutes() {
	v1 := s.router.Group("/api/v1")
	{
		auth := v1.Group("/auth")
		{
			auth.POST("/register", s.handleRegister)
			auth.POST("/login", s.handleLogin)
		}

		market := v1.Group("/market")
		{
			market.GET("/price/:symbol", s.handleGetPrice)
			market.GET("/orderbook/:symbol", s.handleGetOrderBook)
		}

		news := v1.Group("/news")
		{
			news.GET("/latest", s.handleGetLatestNews)
			news.GET("/tags/:tag", s.handleGetNewsByTag)
		}

		screener := v1.Group("/screener")
		{
			screener.GET("/", s.handleGetScreenerResults)
		}

		insider := v1.Group("/insider")
		{
			insider.GET("/", s.handleGetInsiderTrades)
		}

		sector := v1.Group("/sector")
		{
			sector.GET("/", s.handleGetSectorPerformance)
		}
	}
}

func (s *Server) handleGetPrice(c *gin.Context) {
	symbol := c.Param("symbol")
	if s.marketService == nil {
		c.JSON(503, gin.H{"error": "Market service unavailable"})
		return
	}

	price, err := s.marketService.GetPrice(symbol)
	if err != nil {
		c.JSON(404, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, price)
}

func (s *Server) handleGetOrderBook(c *gin.Context) {
	symbol := c.Param("symbol")
	if s.marketService == nil {
		c.JSON(503, gin.H{"error": "Market service unavailable"})
		return
	}

	book, err := s.marketService.GetOrderBook(symbol)
	if err != nil {
		c.JSON(404, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, book)
}

func (s *Server) handleGetLatestNews(c *gin.Context) {
	if s.newsService == nil {
		c.JSON(503, gin.H{"error": "News service unavailable"})
		return
	}

	limitStr := c.DefaultQuery("limit", "20")
	limit, err := strconv.ParseInt(limitStr, 10, 64)
	if err != nil {
		limit = 20
	}

	news, err := s.newsService.GetLatestNews(c.Request.Context(), limit)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, news)
}

func (s *Server) handleGetNewsByTag(c *gin.Context) {
	tag := c.Param("tag")
	if s.newsService == nil {
		c.JSON(503, gin.H{"error": "News service unavailable"})
		return
	}

	news, err := s.newsService.GetNewsByTag(c.Request.Context(), tag, 20)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, news)
}

func (s *Server) handleRegister(c *gin.Context) {
	var req struct {
		Email    string `json:"email" binding:"required,email"`
		Password string `json:"password" binding:"required,min=8"`
		FullName string `json:"full_name" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if s.authService == nil {
		c.JSON(503, gin.H{"error": "Database unavailable"})
		return
	}

	user, err := s.authService.Register(req.Email, req.Password, req.FullName)
	if err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	c.JSON(201, gin.H{"id": user.ID, "email": user.Email})
}

func (s *Server) handleLogin(c *gin.Context) {
	var req struct {
		Email    string `json:"email" binding:"required,email"`
		Password string `json:"password" binding:"required"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(400, gin.H{"error": err.Error()})
		return
	}

	if s.authService == nil {
		c.JSON(503, gin.H{"error": "Database unavailable"})
		return
	}

	token, err := s.authService.Login(req.Email, req.Password)
	if err != nil {
		c.JSON(401, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, gin.H{"token": token})
}

func (s *Server) handleGetScreenerResults(c *gin.Context) {
	if s.screenerService == nil {
		c.JSON(503, gin.H{"error": "Screener service unavailable"})
		return
	}

	strategy := c.Query("strategy")
	limitStr := c.DefaultQuery("limit", "50")
	limit, err := strconv.ParseInt(limitStr, 10, 64)
	if err != nil {
		limit = 50
	}

	results, err := s.screenerService.GetScreenerResults(c.Request.Context(), strategy, limit)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, results)
}

func (s *Server) handleGetInsiderTrades(c *gin.Context) {
	if s.insiderService == nil {
		c.JSON(503, gin.H{"error": "Insider service unavailable"})
		return
	}

	limitStr := c.DefaultQuery("limit", "50")
	limit, err := strconv.ParseInt(limitStr, 10, 64)
	if err != nil {
		limit = 50
	}

	trades, err := s.insiderService.GetInsiderTrades(c.Request.Context(), limit)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, trades)
}

func (s *Server) handleGetSectorPerformance(c *gin.Context) {
	if s.sectorService == nil {
		c.JSON(503, gin.H{"error": "Sector service unavailable"})
		return
	}

	performance, err := s.sectorService.GetSectorPerformance(c.Request.Context())
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, performance)
}

func (s *Server) Run() error {
	return s.router.Run(":" + s.cfg.Server.Port)
}
