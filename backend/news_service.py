import asyncio
import aiohttp
import feedparser
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import re
import os
from dotenv import load_dotenv
from textblob import TextBlob

from schemas import NewsResponse

load_dotenv()

class NewsService:
    def __init__(self):
        # API keys
        self.newsapi_key = os.getenv('NEWSAPI_KEY')
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        
        # RSS feeds for financial news
        self.rss_feeds = {
            'general': [
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://www.reuters.com/business/finance/rss',
                'https://www.cnbc.com/id/100003114/device/rss/rss.html',
                'https://feeds.marketwatch.com/marketwatch/topstories/',
            ],
            'crypto': [
                'https://cointelegraph.com/rss',
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://decrypt.co/feed',
            ],
            'stocks': [
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://www.marketwatch.com/rss/topstories',
            ]
        }
        
        # Cache for news articles
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache
        
        # Keywords for categorization
        self.category_keywords = {
            'crypto': ['bitcoin', 'ethereum', 'cryptocurrency', 'blockchain', 'defi', 'nft', 'altcoin', 'crypto'],
            'stocks': ['stock', 'equity', 'shares', 'earnings', 'dividend', 'ipo', 'nasdaq', 'nyse', 'sp500'],
            'forex': ['forex', 'currency', 'exchange rate', 'dollar', 'euro', 'yen', 'pound', 'fx'],
            'commodities': ['gold', 'silver', 'oil', 'crude', 'commodity', 'wheat', 'corn', 'copper']
        }
    
    async def get_news(self, category: Optional[str] = None, limit: int = 20) -> List[NewsResponse]:
        """Get news articles by category."""
        try:
            # Check cache first
            cache_key = f"news_{category}_{limit}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            news_articles = []
            
            # Fetch from multiple sources
            if self.newsapi_key:
                newsapi_articles = await self._fetch_from_newsapi(category, limit // 2)
                news_articles.extend(newsapi_articles)
            
            # Fetch from RSS feeds
            rss_articles = await self._fetch_from_rss(category, limit // 2)
            news_articles.extend(rss_articles)
            
            # Fetch from Finnhub if available
            if self.finnhub_key:
                finnhub_articles = await self._fetch_from_finnhub(category, limit // 4)
                news_articles.extend(finnhub_articles)
            
            # Remove duplicates and sort by date
            unique_articles = self._remove_duplicates(news_articles)
            sorted_articles = sorted(unique_articles, key=lambda x: x.published_at or datetime.now(), reverse=True)
            
            # Limit results
            result = sorted_articles[:limit]
            
            # Cache the result
            self.cache[cache_key] = {
                'data': result,
                'timestamp': datetime.now()
            }
            
            return result
            
        except Exception as e:
            print(f"Error fetching news: {e}")
            return self._get_mock_news(category, limit)
    
    async def _fetch_from_newsapi(self, category: Optional[str], limit: int) -> List[NewsResponse]:
        """Fetch news from NewsAPI."""
        articles = []
        
        try:
            # Map categories to NewsAPI queries
            query_map = {
                'crypto': 'cryptocurrency OR bitcoin OR ethereum OR blockchain',
                'stocks': 'stock market OR earnings OR NYSE OR NASDAQ',
                'forex': 'forex OR currency OR exchange rate',
                'general': 'finance OR economy OR market'
            }
            
            query = query_map.get(category, query_map['general'])
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': limit,
                'apiKey': self.newsapi_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for article in data.get('articles', []):
                            if article.get('title') and article.get('url'):
                                # Analyze sentiment
                                sentiment_score, sentiment_label = self._analyze_sentiment(
                                    article.get('title', '') + ' ' + (article.get('description') or '')
                                )
                                
                                # Categorize article
                                article_category = self._categorize_article(
                                    article.get('title', '') + ' ' + (article.get('description') or '')
                                )
                                
                                news_article = NewsResponse(
                                    id=self._generate_id(article['url']),
                                    title=article['title'],
                                    content=article.get('content'),
                                    summary=article.get('description'),
                                    url=article['url'],
                                    source=article.get('source', {}).get('name', 'Unknown'),
                                    author=article.get('author'),
                                    category=article_category,
                                    tags=self._extract_tags(article.get('title', '') + ' ' + (article.get('description') or '')),
                                    related_symbols=self._extract_symbols(article.get('title', '') + ' ' + (article.get('description') or '')),
                                    sentiment_score=sentiment_score,
                                    sentiment_label=sentiment_label,
                                    image_url=article.get('urlToImage'),
                                    published_at=self._parse_datetime(article.get('publishedAt')),
                                    views=0,
                                    created_at=datetime.now()
                                )
                                articles.append(news_article)
                                
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
        
        return articles
    
    async def _fetch_from_rss(self, category: Optional[str], limit: int) -> List[NewsResponse]:
        """Fetch news from RSS feeds."""
        articles = []
        
        try:
            feeds = self.rss_feeds.get(category, self.rss_feeds['general'])
            
            for feed_url in feeds:
                try:
                    # Parse RSS feed
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:limit // len(feeds)]:
                        if hasattr(entry, 'title') and hasattr(entry, 'link'):
                            # Get full content if available
                            content = getattr(entry, 'content', [{}])
                            content_text = content[0].get('value', '') if content else ''
                            
                            if not content_text:
                                content_text = getattr(entry, 'summary', '')
                            
                            # Analyze sentiment
                            text_for_analysis = entry.title + ' ' + content_text
                            sentiment_score, sentiment_label = self._analyze_sentiment(text_for_analysis)
                            
                            # Categorize article
                            article_category = self._categorize_article(text_for_analysis)
                            
                            news_article = NewsResponse(
                                id=self._generate_id(entry.link),
                                title=entry.title,
                                content=content_text,
                                summary=getattr(entry, 'summary', '')[:500],
                                url=entry.link,
                                source=feed.feed.get('title', 'RSS Feed'),
                                author=getattr(entry, 'author', None),
                                category=article_category,
                                tags=self._extract_tags(text_for_analysis),
                                related_symbols=self._extract_symbols(text_for_analysis),
                                sentiment_score=sentiment_score,
                                sentiment_label=sentiment_label,
                                published_at=self._parse_datetime(getattr(entry, 'published', None)),
                                views=0,
                                created_at=datetime.now()
                            )
                            articles.append(news_article)
                            
                except Exception as e:
                    print(f"Error parsing RSS feed {feed_url}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error fetching from RSS: {e}")
        
        return articles
    
    async def _fetch_from_finnhub(self, category: Optional[str], limit: int) -> List[NewsResponse]:
        """Fetch news from Finnhub API."""
        articles = []
        
        try:
            url = "https://finnhub.io/api/v1/news"
            params = {
                'category': 'general',
                'token': self.finnhub_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data[:limit]:
                            if item.get('headline') and item.get('url'):
                                # Analyze sentiment
                                sentiment_score, sentiment_label = self._analyze_sentiment(
                                    item.get('headline', '') + ' ' + (item.get('summary') or '')
                                )
                                
                                # Categorize article
                                article_category = self._categorize_article(
                                    item.get('headline', '') + ' ' + (item.get('summary') or '')
                                )
                                
                                news_article = NewsResponse(
                                    id=self._generate_id(item['url']),
                                    title=item['headline'],
                                    summary=item.get('summary'),
                                    url=item['url'],
                                    source=item.get('source', 'Finnhub'),
                                    category=article_category,
                                    tags=self._extract_tags(item.get('headline', '') + ' ' + (item.get('summary') or '')),
                                    related_symbols=item.get('related', '').split(',') if item.get('related') else [],
                                    sentiment_score=sentiment_score,
                                    sentiment_label=sentiment_label,
                                    image_url=item.get('image'),
                                    published_at=datetime.fromtimestamp(item.get('datetime', 0)),
                                    views=0,
                                    created_at=datetime.now()
                                )
                                articles.append(news_article)
                                
        except Exception as e:
            print(f"Error fetching from Finnhub: {e}")
        
        return articles
    
    async def get_article(self, article_id: str) -> Optional[NewsResponse]:
        """Get a specific news article by ID."""
        try:
            # In a real implementation, you would fetch from database
            # For now, search through cached articles
            for cache_key, cache_data in self.cache.items():
                if cache_key.startswith('news_'):
                    for article in cache_data['data']:
                        if article.id == article_id:
                            # Increment view count
                            article.views += 1
                            return article
            
            return None
            
        except Exception as e:
            print(f"Error fetching article {article_id}: {e}")
            return None
    
    def _analyze_sentiment(self, text: str) -> tuple[float, str]:
        """Analyze sentiment of text using TextBlob."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            
            if polarity > 0.1:
                label = 'positive'
            elif polarity < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return polarity, label
            
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return 0.0, 'neutral'
    
    def _categorize_article(self, text: str) -> str:
        """Categorize article based on keywords."""
        text_lower = text.lower()
        
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'general'
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract relevant tags from text."""
        tags = []
        text_lower = text.lower()
        
        # Common financial terms
        financial_terms = [
            'earnings', 'revenue', 'profit', 'loss', 'ipo', 'merger', 'acquisition',
            'dividend', 'buyback', 'guidance', 'forecast', 'outlook', 'rally',
            'crash', 'volatility', 'bull', 'bear', 'correction', 'recession'
        ]
        
        for term in financial_terms:
            if term in text_lower:
                tags.append(term)
        
        # Add category-specific tags
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in text_lower and keyword not in tags:
                    tags.append(keyword)
        
        return tags[:5]  # Limit to 5 tags
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols and crypto symbols from text."""
        symbols = []
        
        # Common stock symbols pattern
        stock_pattern = r'\b[A-Z]{1,5}\b'
        potential_symbols = re.findall(stock_pattern, text)
        
        # Known symbols
        known_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE', 'SOL'
        ]
        
        for symbol in potential_symbols:
            if symbol in known_symbols:
                symbols.append(symbol)
        
        # Crypto mentions
        crypto_mentions = {
            'bitcoin': 'BTC',
            'ethereum': 'ETH',
            'cardano': 'ADA',
            'polkadot': 'DOT',
            'chainlink': 'LINK'
        }
        
        text_lower = text.lower()
        for mention, symbol in crypto_mentions.items():
            if mention in text_lower and symbol not in symbols:
                symbols.append(symbol)
        
        return symbols[:3]  # Limit to 3 symbols
    
    def _generate_id(self, url: str) -> str:
        """Generate a unique ID for an article based on URL."""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()[:12]
    
    def _parse_datetime(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string to datetime object."""
        if not date_str:
            return None
        
        try:
            # Try different datetime formats
            formats = [
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%a, %d %b %Y %H:%M:%S %Z',
                '%a, %d %b %Y %H:%M:%S %z'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            
            # If all formats fail, return current time
            return datetime.now()
            
        except Exception as e:
            print(f"Error parsing datetime {date_str}: {e}")
            return datetime.now()
    
    def _remove_duplicates(self, articles: List[NewsResponse]) -> List[NewsResponse]:
        """Remove duplicate articles based on title similarity."""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            # Normalize title for comparison
            normalized_title = re.sub(r'[^a-zA-Z0-9\s]', '', article.title.lower())
            
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).total_seconds() < self.cache_ttl
    
    def _get_mock_news(self, category: Optional[str], limit: int) -> List[NewsResponse]:
        """Return mock news data when APIs fail."""
        mock_articles = [
            NewsResponse(
                id="mock1",
                title="Bitcoin Surges Past $45,000 as Institutional Adoption Grows",
                content="Bitcoin has reached a new milestone, surging past $45,000 as institutional adoption continues to grow. Major corporations and investment firms are increasingly adding Bitcoin to their portfolios...",
                summary="Bitcoin reaches $45,000 amid growing institutional adoption and positive market sentiment.",
                url="https://example.com/bitcoin-45k",
                source="CryptoNews",
                author="John Crypto",
                category="crypto",
                tags=["bitcoin", "institutional", "adoption"],
                related_symbols=["BTC"],
                sentiment_score=0.7,
                sentiment_label="positive",
                published_at=datetime.now() - timedelta(hours=2),
                views=1250,
                created_at=datetime.now()
            ),
            NewsResponse(
                id="mock2",
                title="Apple Reports Strong Q4 Earnings, Beats Expectations",
                content="Apple Inc. reported strong fourth-quarter earnings that exceeded Wall Street expectations. The tech giant's revenue grew 8% year-over-year...",
                summary="Apple beats Q4 earnings expectations with 8% revenue growth.",
                url="https://example.com/apple-earnings",
                source="Financial Times",
                author="Jane Finance",
                category="stocks",
                tags=["earnings", "apple", "revenue"],
                related_symbols=["AAPL"],
                sentiment_score=0.5,
                sentiment_label="positive",
                published_at=datetime.now() - timedelta(hours=4),
                views=890,
                created_at=datetime.now()
            ),
            NewsResponse(
                id="mock3",
                title="Federal Reserve Signals Potential Rate Cuts in 2024",
                content="The Federal Reserve has signaled potential interest rate cuts in 2024, citing concerns about economic growth and inflation trends...",
                summary="Fed signals possible rate cuts in 2024 amid economic concerns.",
                url="https://example.com/fed-rates",
                source="Reuters",
                author="Economic Reporter",
                category="general",
                tags=["fed", "interest-rates", "economy"],
                related_symbols=[],
                sentiment_score=-0.2,
                sentiment_label="neutral",
                published_at=datetime.now() - timedelta(hours=6),
                views=2100,
                created_at=datetime.now()
            )
        ]
        
        if category:
            mock_articles = [article for article in mock_articles if article.category == category]
        
        return mock_articles[:limit]