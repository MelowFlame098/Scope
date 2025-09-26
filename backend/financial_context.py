"""Financial Context Engine for FinScope - Phase 6 Implementation

Provides comprehensive financial context and data aggregation for LLM explanations.
Integrates market data, news, sentiment, and portfolio information.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import aiohttp
import json

logger = logging.getLogger(__name__)

class ContextType(str, Enum):
    """Types of financial context"""
    MARKET_DATA = "market_data"
    NEWS = "news"
    SENTIMENT = "sentiment"
    PORTFOLIO = "portfolio"
    SECTOR = "sector"
    ECONOMIC = "economic"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SOCIAL = "social"
    OPTIONS = "options"

class DataSource(str, Enum):
    """Available data sources"""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    NEWS_API = "news_api"
    REDDIT = "reddit"
    TWITTER = "twitter"
    SEC_FILINGS = "sec_filings"
    FRED = "fred"
    INTERNAL = "internal"

@dataclass
class ContextItem:
    """Individual context item"""
    type: ContextType
    source: DataSource
    data: Dict[str, Any]
    timestamp: datetime
    relevance_score: float
    metadata: Dict[str, Any] = None

class MarketContext(BaseModel):
    """Market context information"""
    symbol: str
    current_price: float
    price_change: float
    price_change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    beta: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None
    avg_volume: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class NewsItem(BaseModel):
    """News item with sentiment"""
    title: str
    summary: str
    url: str
    source: str
    published_at: datetime
    sentiment: str  # positive, negative, neutral
    sentiment_score: float
    relevance_score: float
    symbols_mentioned: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)

class SentimentData(BaseModel):
    """Aggregated sentiment data"""
    symbol: str
    overall_sentiment: str
    sentiment_score: float
    news_sentiment: float
    social_sentiment: float
    analyst_sentiment: float
    sentiment_trend: str  # improving, declining, stable
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class EconomicIndicator(BaseModel):
    """Economic indicator data"""
    indicator: str
    value: float
    previous_value: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    release_date: datetime
    next_release: Optional[datetime] = None
    importance: str  # high, medium, low
    impact: str  # positive, negative, neutral

class PortfolioContext(BaseModel):
    """User portfolio context"""
    user_id: str
    total_value: float
    day_change: float
    day_change_percent: float
    positions: List[Dict[str, Any]]
    sector_allocation: Dict[str, float]
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class FinancialContextEngine:
    """Advanced financial context aggregation engine"""
    
    def __init__(self):
        self.data_sources = {
            DataSource.YAHOO_FINANCE: self._get_yahoo_data,
            DataSource.NEWS_API: self._get_news_data,
            DataSource.REDDIT: self._get_reddit_data,
            DataSource.INTERNAL: self._get_internal_data
        }
        
        # Cache for context data
        self.context_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Configuration
        self.max_news_items = 20
        self.max_social_items = 50
        self.relevance_threshold = 0.3
    
    async def get_comprehensive_context(
        self,
        symbol: str,
        context_types: List[ContextType] = None,
        timeframe: str = "1d",
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get comprehensive financial context for a symbol"""
        if context_types is None:
            context_types = [
                ContextType.MARKET_DATA,
                ContextType.NEWS,
                ContextType.SENTIMENT,
                ContextType.TECHNICAL
            ]
        
        context = {}
        
        try:
            # Gather context data in parallel
            tasks = []
            
            if ContextType.MARKET_DATA in context_types:
                tasks.append(self._get_market_context(symbol))
            
            if ContextType.NEWS in context_types:
                tasks.append(self._get_news_context(symbol))
            
            if ContextType.SENTIMENT in context_types:
                tasks.append(self._get_sentiment_context(symbol))
            
            if ContextType.TECHNICAL in context_types:
                tasks.append(self._get_technical_context(symbol, timeframe))
            
            if ContextType.SECTOR in context_types:
                tasks.append(self._get_sector_context(symbol))
            
            if ContextType.ECONOMIC in context_types:
                tasks.append(self._get_economic_context())
            
            if ContextType.PORTFOLIO in context_types and user_id:
                tasks.append(self._get_portfolio_context(user_id))
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            result_index = 0
            if ContextType.MARKET_DATA in context_types:
                if not isinstance(results[result_index], Exception):
                    context["market_data"] = results[result_index]
                result_index += 1
            
            if ContextType.NEWS in context_types:
                if not isinstance(results[result_index], Exception):
                    context["news"] = results[result_index]
                result_index += 1
            
            if ContextType.SENTIMENT in context_types:
                if not isinstance(results[result_index], Exception):
                    context["sentiment"] = results[result_index]
                result_index += 1
            
            if ContextType.TECHNICAL in context_types:
                if not isinstance(results[result_index], Exception):
                    context["technical"] = results[result_index]
                result_index += 1
            
            if ContextType.SECTOR in context_types:
                if not isinstance(results[result_index], Exception):
                    context["sector"] = results[result_index]
                result_index += 1
            
            if ContextType.ECONOMIC in context_types:
                if not isinstance(results[result_index], Exception):
                    context["economic"] = results[result_index]
                result_index += 1
            
            if ContextType.PORTFOLIO in context_types and user_id:
                if not isinstance(results[result_index], Exception):
                    context["portfolio"] = results[result_index]
                result_index += 1
            
            # Add metadata
            context["metadata"] = {
                "symbol": symbol,
                "timeframe": timeframe,
                "generated_at": datetime.utcnow().isoformat(),
                "context_types": context_types
            }
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting comprehensive context: {e}")
            return {"error": str(e)}
    
    async def _get_market_context(self, symbol: str) -> MarketContext:
        """Get current market data context"""
        cache_key = f"market_{symbol}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.context_cache[cache_key]["data"]
        
        try:
            # Mock market data - replace with real API calls
            market_data = MarketContext(
                symbol=symbol,
                current_price=150.25,
                price_change=2.50,
                price_change_percent=1.69,
                volume=1250000,
                market_cap=2500000000,
                pe_ratio=25.4,
                beta=1.2,
                day_high=152.30,
                day_low=148.90,
                week_52_high=180.00,
                week_52_low=120.00,
                avg_volume=1100000
            )
            
            # Cache the result
            self._cache_data(cache_key, market_data)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market context for {symbol}: {e}")
            raise
    
    async def _get_news_context(self, symbol: str, hours_back: int = 24) -> List[NewsItem]:
        """Get relevant news context"""
        cache_key = f"news_{symbol}_{hours_back}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.context_cache[cache_key]["data"]
        
        try:
            # Mock news data - replace with real news API
            news_items = [
                NewsItem(
                    title=f"{symbol} Reports Strong Q4 Earnings",
                    summary="Company beats analyst expectations with strong revenue growth",
                    url="https://example.com/news1",
                    source="Financial Times",
                    published_at=datetime.utcnow() - timedelta(hours=2),
                    sentiment="positive",
                    sentiment_score=0.8,
                    relevance_score=0.9,
                    symbols_mentioned=[symbol],
                    categories=["earnings", "financial"]
                ),
                NewsItem(
                    title=f"Analyst Upgrades {symbol} Price Target",
                    summary="Major investment bank raises price target citing strong fundamentals",
                    url="https://example.com/news2",
                    source="Bloomberg",
                    published_at=datetime.utcnow() - timedelta(hours=6),
                    sentiment="positive",
                    sentiment_score=0.7,
                    relevance_score=0.8,
                    symbols_mentioned=[symbol],
                    categories=["analyst", "upgrade"]
                ),
                NewsItem(
                    title="Market Volatility Affects Tech Stocks",
                    summary="Broader market concerns impact technology sector performance",
                    url="https://example.com/news3",
                    source="Reuters",
                    published_at=datetime.utcnow() - timedelta(hours=12),
                    sentiment="negative",
                    sentiment_score=-0.4,
                    relevance_score=0.6,
                    symbols_mentioned=[symbol, "TECH"],
                    categories=["market", "sector"]
                )
            ]
            
            # Filter by relevance
            relevant_news = [
                item for item in news_items 
                if item.relevance_score >= self.relevance_threshold
            ]
            
            # Sort by relevance and recency
            relevant_news.sort(
                key=lambda x: (x.relevance_score, x.published_at),
                reverse=True
            )
            
            # Limit results
            relevant_news = relevant_news[:self.max_news_items]
            
            # Cache the result
            self._cache_data(cache_key, relevant_news)
            
            return relevant_news
            
        except Exception as e:
            logger.error(f"Error getting news context for {symbol}: {e}")
            return []
    
    async def _get_sentiment_context(self, symbol: str) -> SentimentData:
        """Get aggregated sentiment data"""
        cache_key = f"sentiment_{symbol}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.context_cache[cache_key]["data"]
        
        try:
            # Get news sentiment
            news_items = await self._get_news_context(symbol)
            news_sentiment = np.mean([item.sentiment_score for item in news_items]) if news_items else 0.0
            
            # Mock social sentiment - replace with real social media analysis
            social_sentiment = 0.3
            
            # Mock analyst sentiment - replace with real analyst data
            analyst_sentiment = 0.6
            
            # Calculate overall sentiment
            overall_score = (news_sentiment * 0.4 + social_sentiment * 0.3 + analyst_sentiment * 0.3)
            
            # Determine sentiment label
            if overall_score > 0.2:
                overall_sentiment = "positive"
            elif overall_score < -0.2:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            # Determine trend (mock)
            sentiment_trend = "stable"
            if overall_score > 0.5:
                sentiment_trend = "improving"
            elif overall_score < -0.5:
                sentiment_trend = "declining"
            
            sentiment_data = SentimentData(
                symbol=symbol,
                overall_sentiment=overall_sentiment,
                sentiment_score=overall_score,
                news_sentiment=news_sentiment,
                social_sentiment=social_sentiment,
                analyst_sentiment=analyst_sentiment,
                sentiment_trend=sentiment_trend,
                confidence=0.75
            )
            
            # Cache the result
            self._cache_data(cache_key, sentiment_data)
            
            return sentiment_data
            
        except Exception as e:
            logger.error(f"Error getting sentiment context for {symbol}: {e}")
            raise
    
    async def _get_technical_context(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get technical analysis context"""
        cache_key = f"technical_{symbol}_{timeframe}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.context_cache[cache_key]["data"]
        
        try:
            # Mock technical indicators - replace with real technical analysis
            technical_data = {
                "rsi": {
                    "value": 65.4,
                    "signal": "neutral",
                    "description": "RSI above 60 indicates bullish momentum"
                },
                "macd": {
                    "macd": 1.25,
                    "signal": 1.10,
                    "histogram": 0.15,
                    "signal": "bullish",
                    "description": "MACD above signal line indicates upward momentum"
                },
                "moving_averages": {
                    "sma_20": 148.50,
                    "sma_50": 145.20,
                    "sma_200": 140.80,
                    "signal": "bullish",
                    "description": "Price above all major moving averages"
                },
                "bollinger_bands": {
                    "upper": 155.00,
                    "middle": 150.00,
                    "lower": 145.00,
                    "position": "upper_half",
                    "signal": "neutral",
                    "description": "Price in upper half of Bollinger Bands"
                },
                "support_resistance": {
                    "support_levels": [145.00, 142.50, 140.00],
                    "resistance_levels": [155.00, 158.00, 162.00],
                    "nearest_support": 145.00,
                    "nearest_resistance": 155.00
                },
                "volume_analysis": {
                    "avg_volume": 1100000,
                    "current_volume": 1250000,
                    "volume_ratio": 1.14,
                    "signal": "above_average",
                    "description": "Volume 14% above average indicates increased interest"
                }
            }
            
            # Cache the result
            self._cache_data(cache_key, technical_data)
            
            return technical_data
            
        except Exception as e:
            logger.error(f"Error getting technical context for {symbol}: {e}")
            return {}
    
    async def _get_sector_context(self, symbol: str) -> Dict[str, Any]:
        """Get sector-specific context"""
        cache_key = f"sector_{symbol}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.context_cache[cache_key]["data"]
        
        try:
            # Mock sector data - replace with real sector analysis
            sector_data = {
                "sector": "Technology",
                "industry": "Software",
                "sector_performance": {
                    "day_change": 1.2,
                    "week_change": 3.5,
                    "month_change": -2.1,
                    "ytd_change": 15.8
                },
                "peer_comparison": {
                    "peers": ["MSFT", "GOOGL", "AMZN"],
                    "relative_performance": "outperforming",
                    "percentile_rank": 75
                },
                "sector_trends": [
                    "AI adoption driving growth",
                    "Cloud migration accelerating",
                    "Regulatory concerns in focus"
                ],
                "key_metrics": {
                    "avg_pe_ratio": 28.5,
                    "avg_revenue_growth": 12.3,
                    "avg_profit_margin": 22.1
                }
            }
            
            # Cache the result
            self._cache_data(cache_key, sector_data)
            
            return sector_data
            
        except Exception as e:
            logger.error(f"Error getting sector context for {symbol}: {e}")
            return {}
    
    async def _get_economic_context(self) -> List[EconomicIndicator]:
        """Get relevant economic indicators"""
        cache_key = "economic_indicators"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.context_cache[cache_key]["data"]
        
        try:
            # Mock economic indicators - replace with real economic data
            indicators = [
                EconomicIndicator(
                    indicator="Federal Funds Rate",
                    value=5.25,
                    previous_value=5.00,
                    change=0.25,
                    change_percent=5.0,
                    release_date=datetime.utcnow() - timedelta(days=7),
                    next_release=datetime.utcnow() + timedelta(days=35),
                    importance="high",
                    impact="negative"
                ),
                EconomicIndicator(
                    indicator="Unemployment Rate",
                    value=3.8,
                    previous_value=3.9,
                    change=-0.1,
                    change_percent=-2.6,
                    release_date=datetime.utcnow() - timedelta(days=3),
                    next_release=datetime.utcnow() + timedelta(days=28),
                    importance="high",
                    impact="positive"
                ),
                EconomicIndicator(
                    indicator="CPI Inflation",
                    value=3.2,
                    previous_value=3.4,
                    change=-0.2,
                    change_percent=-5.9,
                    release_date=datetime.utcnow() - timedelta(days=10),
                    next_release=datetime.utcnow() + timedelta(days=20),
                    importance="high",
                    impact="positive"
                )
            ]
            
            # Cache the result
            self._cache_data(cache_key, indicators)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error getting economic context: {e}")
            return []
    
    async def _get_portfolio_context(self, user_id: str) -> PortfolioContext:
        """Get user portfolio context"""
        cache_key = f"portfolio_{user_id}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.context_cache[cache_key]["data"]
        
        try:
            # Mock portfolio data - replace with real portfolio service
            portfolio_data = PortfolioContext(
                user_id=user_id,
                total_value=125000.00,
                day_change=1250.00,
                day_change_percent=1.01,
                positions=[
                    {"symbol": "AAPL", "shares": 100, "value": 15025.00, "weight": 0.12},
                    {"symbol": "MSFT", "shares": 50, "value": 18750.00, "weight": 0.15},
                    {"symbol": "GOOGL", "shares": 25, "value": 12500.00, "weight": 0.10}
                ],
                sector_allocation={
                    "Technology": 0.45,
                    "Healthcare": 0.20,
                    "Financial": 0.15,
                    "Consumer": 0.10,
                    "Energy": 0.05,
                    "Other": 0.05
                },
                risk_metrics={
                    "beta": 1.15,
                    "volatility": 18.5,
                    "sharpe_ratio": 1.25,
                    "max_drawdown": -12.3,
                    "var_95": -2850.00
                },
                performance_metrics={
                    "ytd_return": 12.5,
                    "one_year_return": 18.3,
                    "three_year_return": 45.2,
                    "alpha": 2.1,
                    "tracking_error": 4.2
                }
            )
            
            # Cache the result
            self._cache_data(cache_key, portfolio_data)
            
            return portfolio_data
            
        except Exception as e:
            logger.error(f"Error getting portfolio context for {user_id}: {e}")
            raise
    
    async def get_relevant_news(
        self,
        symbol: str,
        hours_back: int = 24,
        min_relevance: float = 0.5
    ) -> List[NewsItem]:
        """Get relevant news for a symbol"""
        news_items = await self._get_news_context(symbol, hours_back)
        return [
            item for item in news_items 
            if item.relevance_score >= min_relevance
        ]
    
    async def get_user_portfolio_context(self, user_id: str) -> PortfolioContext:
        """Get user portfolio context"""
        return await self._get_portfolio_context(user_id)
    
    async def get_market_summary(self, symbols: List[str]) -> Dict[str, MarketContext]:
        """Get market summary for multiple symbols"""
        tasks = [self._get_market_context(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        summary = {}
        for i, symbol in enumerate(symbols):
            if not isinstance(results[i], Exception):
                summary[symbol] = results[i]
        
        return summary
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.context_cache:
            return False
        
        cached_time = self.context_cache[cache_key]["timestamp"]
        return (datetime.utcnow() - cached_time).total_seconds() < self.cache_ttl
    
    def _cache_data(self, cache_key: str, data: Any) -> None:
        """Cache data with timestamp"""
        self.context_cache[cache_key] = {
            "data": data,
            "timestamp": datetime.utcnow()
        }
    
    async def _get_yahoo_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from Yahoo Finance API"""
        # Mock implementation - replace with real Yahoo Finance API
        return {"source": "yahoo", "symbol": symbol, "data": "mock_data"}
    
    async def _get_news_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from News API"""
        # Mock implementation - replace with real News API
        return {"source": "news_api", "symbol": symbol, "data": "mock_news"}
    
    async def _get_reddit_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from Reddit API"""
        # Mock implementation - replace with real Reddit API
        return {"source": "reddit", "symbol": symbol, "data": "mock_reddit"}
    
    async def _get_internal_data(self, symbol: str) -> Dict[str, Any]:
        """Get data from internal sources"""
        # Mock implementation - replace with internal data sources
        return {"source": "internal", "symbol": symbol, "data": "mock_internal"}
    
    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.context_cache.clear()
        logger.info("Context cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_items = len(self.context_cache)
        valid_items = sum(1 for key in self.context_cache.keys() if self._is_cache_valid(key))
        
        return {
            "total_items": total_items,
            "valid_items": valid_items,
            "expired_items": total_items - valid_items,
            "cache_hit_rate": valid_items / total_items if total_items > 0 else 0,
            "ttl_seconds": self.cache_ttl
        }

# Global context engine instance
financial_context_engine = FinancialContextEngine()