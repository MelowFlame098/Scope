package server

import (
	"fmt"
	"log"
	"scope-backend/internal/config"
	"scope-backend/internal/services"
	"scope-backend/internal/worker"

	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
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
	fundService     *services.FundamentalsService
	redisClient     *redis.Client
}

func NewServer(cfg *config.Config, db *gorm.DB, mongoDB *mongo.Database, redisClient *redis.Client, taskDistributor *worker.TaskDistributor, authService *services.AuthService, marketService *services.MarketService, newsService *services.NewsService, screenerService *services.ScreenerService, insiderService *services.InsiderService, sectorService *services.SectorService, fundService *services.FundamentalsService) *Server {
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
		fundService:     fundService,
		router:          gin.Default(),
	}

	// Add CORS middleware
	s.router.Use(func(c *gin.Context) {
		c.Writer.Header().Set("Access-Control-Allow-Origin", "*")
		c.Writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
		c.Writer.Header().Set("Access-Control-Allow-Headers", "Content-Type, Content-Length, Accept-Encoding, X-CSRF-Token, Authorization")

		if c.Request.Method == "OPTIONS" {
			c.AbortWithStatus(204)
			return
		}

		c.Next()
	})

	s.SetupRoutes()
	return s
}

func (s *Server) Run(addr string) error {
	return s.router.Run(addr)
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
			market.GET("/movers", s.handleGetMovers)
			market.GET("/candles/:symbol", s.handleGetCandles)
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

		fundamentals := v1.Group("/fundamentals")
		{
			fundamentals.GET("/:symbol", s.handleGetFundamentals)
		}

		monitor := v1.Group("/monitor")
		{
			monitor.GET("/feed", s.handleGetMonitorFeed)
		}
	}
}

func (s *Server) handleGetMonitorFeed(c *gin.Context) {
	now := time.Now().UTC()
	stream := c.DefaultQuery("stream", "news")
	limitStr := c.DefaultQuery("limit", "20")
	limit, err := strconv.ParseInt(limitStr, 10, 64)
	if err != nil || limit <= 0 || limit > 5000 {
		limit = 20
	}
	afterStr := c.Query("after")
	var after time.Time
	if afterStr != "" {
		if t, err := time.Parse(time.RFC3339, afterStr); err == nil {
			after = t.UTC()
		}
	}

	compact := func(s string) string {
		return strings.Join(strings.Fields(strings.ReplaceAll(s, "\n", " ")), " ")
	}
	truncate := func(s string, max int) string {
		s = compact(s)
		if max <= 0 {
			return ""
		}
		r := []rune(s)
		if len(r) <= max {
			return s
		}
		return string(r[:max]) + "…"
	}

	if s.newsService != nil && stream == "news" {
		baseLimit := limit * 15
		if baseLimit > 5000 {
			baseLimit = 5000
		}
		if baseLimit < limit {
			baseLimit = limit
		}

		var articles []services.NewsArticle
		var err error
		if !after.IsZero() {
			articles, err = s.newsService.GetLatestNewsAfter(c.Request.Context(), baseLimit, after)
		} else {
			articles, err = s.newsService.GetLatestNews(c.Request.Context(), baseLimit)
		}
		if err == nil && len(articles) > 0 {
			isBroadCategory := func(s string) bool {
				switch strings.ToLower(s) {
				case "technology", "finance", "crypto", "macro", "geopolitics", "natural resources", "general":
					return true
				default:
					return false
				}
			}

			categoryFor := func(a services.NewsArticle) string {
				if strings.EqualFold(a.Source, "Finviz Aggregated") {
					for _, t := range a.Tags {
						if isBroadCategory(t) && !strings.EqualFold(t, "general") {
							return t
						}
					}
					return "Markets"
				}
				for _, t := range a.Tags {
					if isBroadCategory(t) {
						return t
					}
				}
				if len(a.Tags) > 0 {
					return a.Tags[0]
				}
				return ""
			}

			capFinviz := limit / 3
			if capFinviz < 10 {
				capFinviz = 10
			}
			if capFinviz > 120 {
				capFinviz = 120
			}
			capOther := limit / 4
			if capOther < 6 {
				capOther = 6
			}
			if capOther > 60 {
				capOther = 60
			}

			items := make([]gin.H, 0, limit)
			seen := make(map[string]struct{}, len(articles))
			counts := make(map[string]int, 16)
			appendItem := func(a services.NewsArticle) {
				id := a.ID.Hex()
				if _, ok := seen[id]; ok {
					return
				}
				seen[id] = struct{}{}
				items = append(items, gin.H{
					"id":           id,
					"kind":         "news",
					"title":        truncate(a.Title, 140),
					"summary":      truncate(a.Content, 280),
					"source":       a.Source,
					"url":          a.URL,
					"region":       "Global",
					"category":     categoryFor(a),
					"published_at": a.Timestamp.UTC().Format(time.RFC3339),
					"relevance":    a.Relevance,
				})
			}

			for _, a := range articles {
				src := a.Source
				cap := capOther
				if strings.EqualFold(src, "Finviz Aggregated") {
					cap = capFinviz
				}
				if counts[src] >= int(cap) {
					continue
				}
				counts[src]++
				appendItem(a)
				if int64(len(items)) >= limit {
					break
				}
			}
			if int64(len(items)) < limit {
				for _, a := range articles {
					appendItem(a)
					if int64(len(items)) >= limit {
						break
					}
				}
			}

			c.JSON(200, gin.H{
				"generated_at": now.Format(time.RFC3339),
				"items":        items,
			})
			return
		}
	}

	if s.mongoDB != nil && stream != "news" {
		filter := bson.M{}
		switch stream {
		case "events":
			filter["kind"] = bson.M{"$in": []string{"earthquake", "disaster", "gdelt", "cyber", "aviation"}}
		case "cyber":
			filter["kind"] = bson.M{"$in": []string{"cyber"}}
		case "markets":
			filter["kind"] = bson.M{"$in": []string{"crypto"}}
		case "watchlist":
			// no kind filter; return all OSINT event types
		default:
			filter["kind"] = stream
		}
		if !after.IsZero() {
			filter["published_at"] = bson.M{"$gte": after}
		}

		opts := options.Find().SetSort(bson.D{{Key: "published_at", Value: -1}}).SetLimit(limit)
		cursor, err := s.mongoDB.Collection("osint_events").Find(c.Request.Context(), filter, opts)
		if err == nil {
			var docs []bson.M
			if err := cursor.All(c.Request.Context(), &docs); err == nil && len(docs) > 0 {
				getString := func(m bson.M, key string) string {
					v, ok := m[key]
					if !ok || v == nil {
						return ""
					}
					switch t := v.(type) {
					case string:
						return t
					default:
						return fmt.Sprint(t)
					}
				}
				getTime := func(m bson.M, key string) string {
					v, ok := m[key]
					if !ok || v == nil {
						return ""
					}
					switch t := v.(type) {
					case time.Time:
						return t.UTC().Format(time.RFC3339)
					case primitive.DateTime:
						return t.Time().UTC().Format(time.RFC3339)
					default:
						return ""
					}
				}

				getFloat := func(v any) (float64, bool) {
					switch t := v.(type) {
					case float64:
						return t, true
					case float32:
						return float64(t), true
					case int64:
						return float64(t), true
					case int32:
						return float64(t), true
					case int:
						return float64(t), true
					default:
						return 0, false
					}
				}

				getNestedMetric := func(m bson.M, key string) (float64, bool) {
					v, ok := m["metrics"]
					if !ok || v == nil {
						return 0, false
					}
					switch mt := v.(type) {
					case primitive.M:
						return getFloat(mt[key])
					default:
						return 0, false
					}
				}

				getGeo := func(m bson.M) (float64, float64, bool) {
					v, ok := m["geo"]
					if !ok || v == nil {
						return 0, 0, false
					}
					var gm primitive.M
					switch t := v.(type) {
					case primitive.M:
						gm = t
					default:
						return 0, 0, false
					}
					coordsRaw, ok := gm["coordinates"]
					if !ok || coordsRaw == nil {
						return 0, 0, false
					}
					switch coords := coordsRaw.(type) {
					case primitive.A:
						if len(coords) != 2 {
							return 0, 0, false
						}
						lon, ok1 := getFloat(coords[0])
						lat, ok2 := getFloat(coords[1])
						if !ok1 || !ok2 {
							return 0, 0, false
						}
						return lon, lat, true
					case []interface{}:
						if len(coords) != 2 {
							return 0, 0, false
						}
						lon, ok1 := getFloat(coords[0])
						lat, ok2 := getFloat(coords[1])
						if !ok1 || !ok2 {
							return 0, 0, false
						}
						return lon, lat, true
					default:
						return 0, 0, false
					}
				}

				items := make([]gin.H, 0, len(docs))
				for _, d := range docs {
					id := getString(d, "source_id")
					if id == "" {
						if oid, ok := d["_id"].(primitive.ObjectID); ok {
							id = oid.Hex()
						}
					}

					kind := getString(d, "kind")
					severity := 0.0
					switch kind {
					case "earthquake":
						if v, ok := getNestedMetric(d, "magnitude"); ok {
							severity = v
						}
					case "crypto":
						if v, ok := getNestedMetric(d, "usd_24h_change"); ok {
							if v < 0 {
								severity = -v
							} else {
								severity = v
							}
						}
					case "aviation":
						if v, ok := getNestedMetric(d, "velocity"); ok {
							if v < 0 {
								v = 0
							}
							if v > 1000 {
								v = 1000
							}
							severity = v / 150.0
						} else {
							severity = 3.0
						}
					case "cyber":
						severity = 6.0
					case "disaster":
						severity = 5.0
					case "gdelt":
						severity = 4.0
					}

					var geo any = nil
					if lon, lat, ok := getGeo(d); ok {
						geo = gin.H{"lon": lon, "lat": lat}
					}

					items = append(items, gin.H{
						"id":           id,
						"kind":         kind,
						"title":        truncate(getString(d, "title"), 140),
						"summary":      truncate(getString(d, "summary"), 280),
						"source":       getString(d, "source"),
						"url":          getString(d, "url"),
						"region":       getString(d, "region"),
						"category":     getString(d, "category"),
						"published_at": getTime(d, "published_at"),
						"severity":     severity,
						"geo":          geo,
					})
				}

				c.JSON(200, gin.H{
					"generated_at": now.Format(time.RFC3339),
					"items":        items,
				})
				return
			}
		}
	}

	c.JSON(200, gin.H{
		"generated_at": now.Format(time.RFC3339),
		"items": []gin.H{
			{
				"id":           "demo-001",
				"kind":         "system",
				"title":        "No data yet",
				"summary":      "Scope Monitor is wired to your backend. Next: populate MongoDB (news fetch) and this feed will switch to real items automatically.",
				"source":       "Scope",
				"url":          "",
				"region":       "Global",
				"category":     "System",
				"published_at": now.Add(-5 * time.Minute).Format(time.RFC3339),
			},
		},
	})
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

func (s *Server) handleGetMovers(c *gin.Context) {
	if s.marketService == nil {
		c.JSON(503, gin.H{"error": "Market service unavailable"})
		return
	}

	limitStr := c.DefaultQuery("limit", "6")
	limit, _ := strconv.ParseInt(limitStr, 10, 64)

	top, err := s.marketService.GetTopMovers(limit)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	worst, err := s.marketService.GetWorstMovers(limit)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, gin.H{
		"top":   top,
		"worst": worst,
	})
}

func (s *Server) handleGetCandles(c *gin.Context) {
	symbol := c.Param("symbol")
	timeframe := c.DefaultQuery("timeframe", "5m")

	if s.marketService == nil {
		c.JSON(503, gin.H{"error": "Market service unavailable"})
		return
	}

	candles, err := s.marketService.GetCandles(symbol, timeframe)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, candles)
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
		log.Printf("Insider Error: %v", err)
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

	results, err := s.sectorService.GetSectorPerformance(c.Request.Context())
	if err != nil {
		log.Printf("Sector Error: %v", err)
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}

	c.JSON(200, results)
}

func (s *Server) handleGetFundamentals(c *gin.Context) {
	symbol := c.Param("symbol")
	if s.fundService == nil {
		c.JSON(503, gin.H{"error": "Fundamentals service unavailable"})
		return
	}

	tf := c.DefaultQuery("timeframe", "current")

	rec, err := s.fundService.GetLatestByTimeframe(c.Request.Context(), symbol, tf)
	if err != nil {
		c.JSON(500, gin.H{"error": err.Error()})
		return
	}
	if rec == nil {
		c.JSON(404, gin.H{"error": "no fundamentals found"})
		return
	}

	c.JSON(200, rec)
}
