package services

import (
	"encoding/json"
	"fmt"
	"sort"

	"github.com/redis/go-redis/v9"
)

// Add GetTopMovers and GetWorstMovers to MarketService

func (s *MarketService) GetTopMovers(limit int64) ([]StockPrice, error) {
	// Get top scores from ZSET (descending)
	symbols, err := s.redisClient.ZRevRange(s.ctx, "market:movers", 0, limit-1).Result()
	if err != nil {
		// Fallback to in-memory cache if Redis fails
		return s.getMoversFromCache(limit, true)
	}

	return s.getPricesForSymbols(symbols)
}

func (s *MarketService) GetWorstMovers(limit int64) ([]StockPrice, error) {
	// Get bottom scores from ZSET (ascending)
	symbols, err := s.redisClient.ZRange(s.ctx, "market:movers", 0, limit-1).Result()
	if err != nil {
		// Fallback to in-memory cache if Redis fails
		return s.getMoversFromCache(limit, false)
	}

	return s.getPricesForSymbols(symbols)
}

func (s *MarketService) getMoversFromCache(limit int64, descending bool) ([]StockPrice, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	movers := make([]StockPrice, 0, len(s.movers))
	for _, p := range s.movers {
		movers = append(movers, p)
	}

	sort.Slice(movers, func(i, j int) bool {
		if descending {
			return movers[i].ChangePercent.GreaterThan(movers[j].ChangePercent)
		}
		return movers[i].ChangePercent.LessThan(movers[j].ChangePercent)
	})

	if int64(len(movers)) > limit {
		movers = movers[:limit]
	}
	return movers, nil
}

func (s *MarketService) getPricesForSymbols(symbols []string) ([]StockPrice, error) {
	var prices []StockPrice

	// Pipeline the GET requests for efficiency
	pipe := s.redisClient.Pipeline()
	cmds := make([]*redis.StringCmd, len(symbols))

	for i, sym := range symbols {
		key := fmt.Sprintf("market:price:%s", sym)
		cmds[i] = pipe.Get(s.ctx, key)
	}

	_, err := pipe.Exec(s.ctx)
	if err != nil && err != redis.Nil {
		return nil, err
	}

	for _, cmd := range cmds {
		val, err := cmd.Result()
		if err == nil {
			var price StockPrice
			if err := json.Unmarshal([]byte(val), &price); err == nil {
				prices = append(prices, price)
			}
		}
	}

	return prices, nil
}
