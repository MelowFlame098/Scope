import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MarketSentiment:
    symbol: str
    sentiment_score: float  # -1 to 1 (bearish to bullish)
    confidence: float  # 0 to 1
    source: str
    timestamp: datetime
    summary: str

@dataclass
class TradingSignal:
    symbol: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0 to 1
    source: str
    reasoning: str
    timestamp: datetime
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

class MarketScraperService:
    """Service for scraping market data, sentiment, and trading signals from various sources."""
    
    def __init__(self):
        self.session = None
        self.sentiment_cache = {}
        self.signals_cache = {}
        self.cache_duration = timedelta(minutes=15)
        
        # Headers to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape_reddit_sentiment(self, symbol: str) -> List[MarketSentiment]:
        """Scrape sentiment from Reddit trading communities."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(headers=self.headers)
            
            # Check cache first
            cache_key = f"reddit_{symbol}"
            if self._is_cached(cache_key):
                return self.sentiment_cache[cache_key]['data']
            
            sentiments = []
            
            # Search multiple subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'SecurityAnalysis']
            
            for subreddit in subreddits:
                try:
                    url = f"https://www.reddit.com/r/{subreddit}/search.json?q={symbol}&sort=hot&limit=10"
                    
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for post in data.get('data', {}).get('children', []):
                                post_data = post.get('data', {})
                                title = post_data.get('title', '')
                                selftext = post_data.get('selftext', '')
                                score = post_data.get('score', 0)
                                
                                # Simple sentiment analysis based on keywords
                                sentiment_score = self._analyze_text_sentiment(title + ' ' + selftext)
                                confidence = min(abs(sentiment_score) + 0.3, 1.0)
                                
                                sentiment = MarketSentiment(
                                    symbol=symbol,
                                    sentiment_score=sentiment_score,
                                    confidence=confidence,
                                    source=f"Reddit r/{subreddit}",
                                    timestamp=datetime.now(),
                                    summary=title[:200]
                                )
                                sentiments.append(sentiment)
                                
                except Exception as e:
                    logger.warning(f"Failed to scrape r/{subreddit}: {str(e)}")
                    continue
            
            # Cache results
            self._cache_data(cache_key, sentiments)
            
            logger.info(f"Scraped {len(sentiments)} sentiment data points from Reddit for {symbol}")
            return sentiments
            
        except Exception as e:
            logger.error(f"Reddit sentiment scraping failed for {symbol}: {str(e)}")
            return []
    
    async def scrape_finviz_data(self, symbol: str) -> Dict[str, Any]:
        """Scrape technical and fundamental data from Finviz."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession(headers=self.headers)
            
            cache_key = f"finviz_{symbol}"
            if self._is_cached(cache_key):
                return self.sentiment_cache[cache_key]['data']
            
            url = f"https://finviz.com/quote.ashx?t={symbol}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Extract key metrics
                    data = {
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'metrics': {},
                        'analyst_rating': None,
                        'news_sentiment': None
                    }
                    
                    # Extract fundamental data
                    tables = soup.find_all('table', class_='snapshot-table2')
                    for table in tables:
                        rows = table.find_all('tr')
                        for row in rows:
                            cells = row.find_all('td')
                            if len(cells) >= 2:
                                key = cells[0].get_text(strip=True)
                                value = cells[1].get_text(strip=True)
                                data['metrics'][key] = value
                    
                    # Extract analyst recommendations
                    analyst_section = soup.find('td', string=re.compile('Analyst Recom'))
                    if analyst_section:
                        rating_cell = analyst_section.find_next_sibling('td')
                        if rating_cell:
                            data['analyst_rating'] = rating_cell.get_text(strip=True)
                    
                    # Cache results
                    self._cache_data(cache_key, data)
                    
                    logger.info(f"Scraped Finviz data for {symbol}")
                    return data
                    
        except Exception as e:
            logger.error(f"Finviz scraping failed for {symbol}: {str(e)}")
            return {}
    
    async def scrape_trading_signals(self, symbol: str) -> List[TradingSignal]:
        """Scrape trading signals from various sources."""
        try:
            cache_key = f"signals_{symbol}"
            if self._is_cached(cache_key):
                return self.signals_cache[cache_key]['data']
            
            signals = []
            
            # Combine data from multiple sources to generate signals
            reddit_sentiment = await self.scrape_reddit_sentiment(symbol)
            finviz_data = await self.scrape_finviz_data(symbol)
            
            # Generate signal based on sentiment analysis
            if reddit_sentiment:
                avg_sentiment = sum(s.sentiment_score for s in reddit_sentiment) / len(reddit_sentiment)
                avg_confidence = sum(s.confidence for s in reddit_sentiment) / len(reddit_sentiment)
                
                if avg_sentiment > 0.3:
                    signal_type = 'buy'
                elif avg_sentiment < -0.3:
                    signal_type = 'sell'
                else:
                    signal_type = 'hold'
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=min(abs(avg_sentiment), 1.0),
                    source='Aggregated Sentiment Analysis',
                    reasoning=f"Based on {len(reddit_sentiment)} sentiment data points with average score {avg_sentiment:.2f}",
                    timestamp=datetime.now()
                )
                signals.append(signal)
            
            # Generate signal based on analyst recommendations
            if finviz_data.get('analyst_rating'):
                rating = finviz_data['analyst_rating'].lower()
                if 'buy' in rating or 'strong buy' in rating:
                    signal_type = 'buy'
                    strength = 0.8 if 'strong' in rating else 0.6
                elif 'sell' in rating:
                    signal_type = 'sell'
                    strength = 0.8 if 'strong' in rating else 0.6
                else:
                    signal_type = 'hold'
                    strength = 0.5
                
                signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=strength,
                    source='Analyst Recommendations',
                    reasoning=f"Analyst rating: {finviz_data['analyst_rating']}",
                    timestamp=datetime.now()
                )
                signals.append(signal)
            
            # Cache results
            self._cache_data(cache_key, signals, cache_type='signals')
            
            logger.info(f"Generated {len(signals)} trading signals for {symbol}")
            return signals
            
        except Exception as e:
            logger.error(f"Trading signal generation failed for {symbol}: {str(e)}")
            return []
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Simple keyword-based sentiment analysis."""
        text = text.lower()
        
        # Positive keywords
        positive_words = [
            'buy', 'bull', 'bullish', 'moon', 'rocket', 'pump', 'up', 'rise', 'gain',
            'profit', 'strong', 'good', 'great', 'excellent', 'positive', 'growth',
            'breakout', 'rally', 'surge', 'momentum'
        ]
        
        # Negative keywords
        negative_words = [
            'sell', 'bear', 'bearish', 'crash', 'dump', 'down', 'fall', 'loss',
            'weak', 'bad', 'terrible', 'negative', 'decline', 'drop', 'plunge',
            'breakdown', 'correction', 'recession'
        ]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Calculate sentiment score
        sentiment = (positive_count - negative_count) / max(total_words / 10, 1)
        return max(-1.0, min(1.0, sentiment))
    
    def _is_cached(self, key: str) -> bool:
        """Check if data is cached and still valid."""
        if key in self.sentiment_cache:
            cache_time = self.sentiment_cache[key]['timestamp']
            return datetime.now() - cache_time < self.cache_duration
        return False
    
    def _cache_data(self, key: str, data: Any, cache_type: str = 'sentiment'):
        """Cache data with timestamp."""
        cache = self.sentiment_cache if cache_type == 'sentiment' else self.signals_cache
        cache[key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    async def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market analysis for a symbol."""
        try:
            # Gather all data concurrently
            sentiment_task = self.scrape_reddit_sentiment(symbol)
            finviz_task = self.scrape_finviz_data(symbol)
            signals_task = self.scrape_trading_signals(symbol)
            
            sentiment_data, finviz_data, signals_data = await asyncio.gather(
                sentiment_task, finviz_task, signals_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(sentiment_data, Exception):
                sentiment_data = []
            if isinstance(finviz_data, Exception):
                finviz_data = {}
            if isinstance(signals_data, Exception):
                signals_data = []
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'sentiment_analysis': {
                    'data_points': len(sentiment_data),
                    'average_sentiment': sum(s.sentiment_score for s in sentiment_data) / len(sentiment_data) if sentiment_data else 0,
                    'confidence': sum(s.confidence for s in sentiment_data) / len(sentiment_data) if sentiment_data else 0,
                    'details': [{
                        'source': s.source,
                        'sentiment': s.sentiment_score,
                        'summary': s.summary
                    } for s in sentiment_data]
                },
                'fundamental_data': finviz_data,
                'trading_signals': [{
                    'type': s.signal_type,
                    'strength': s.strength,
                    'source': s.source,
                    'reasoning': s.reasoning
                } for s in signals_data],
                'overall_recommendation': self._generate_overall_recommendation(sentiment_data, signals_data)
            }
            
            logger.info(f"Comprehensive analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed for {symbol}: {str(e)}")
            return {}
    
    def _generate_overall_recommendation(self, sentiment_data: List[MarketSentiment], signals_data: List[TradingSignal]) -> Dict[str, Any]:
        """Generate overall recommendation based on all data sources."""
        try:
            # Calculate weighted scores
            sentiment_score = 0
            signal_score = 0
            
            if sentiment_data:
                sentiment_score = sum(s.sentiment_score * s.confidence for s in sentiment_data) / len(sentiment_data)
            
            if signals_data:
                buy_signals = [s for s in signals_data if s.signal_type == 'buy']
                sell_signals = [s for s in signals_data if s.signal_type == 'sell']
                
                buy_strength = sum(s.strength for s in buy_signals)
                sell_strength = sum(s.strength for s in sell_signals)
                
                signal_score = (buy_strength - sell_strength) / max(len(signals_data), 1)
            
            # Combine scores
            overall_score = (sentiment_score + signal_score) / 2
            
            if overall_score > 0.3:
                recommendation = 'BUY'
            elif overall_score < -0.3:
                recommendation = 'SELL'
            else:
                recommendation = 'HOLD'
            
            return {
                'recommendation': recommendation,
                'confidence': abs(overall_score),
                'sentiment_score': sentiment_score,
                'signal_score': signal_score,
                'overall_score': overall_score
            }
            
        except Exception as e:
            logger.error(f"Overall recommendation generation failed: {str(e)}")
            return {'recommendation': 'HOLD', 'confidence': 0, 'overall_score': 0}

# Global instance
market_scraper_service = MarketScraperService()