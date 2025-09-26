import asyncio
import aiohttp
import feedparser
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import hashlib
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import aioredis
from sentiment_engine import get_sentiment_engine, NewsAnalysis, SentimentResult
import os
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsSource(str, Enum):
    NEWSAPI = "newsapi"
    RSS_FEED = "rss_feed"
    FINNHUB = "finnhub"
    ALPHA_VANTAGE = "alpha_vantage"
    REDDIT = "reddit"
    TWITTER = "twitter"
    CUSTOM = "custom"

class ContentType(str, Enum):
    ARTICLE = "article"
    BLOG_POST = "blog_post"
    PRESS_RELEASE = "press_release"
    SOCIAL_MEDIA = "social_media"
    RESEARCH_REPORT = "research_report"

@dataclass
class NewsArticle:
    id: str
    title: str
    content: str
    summary: str
    url: str
    source: str
    source_type: NewsSource
    content_type: ContentType
    published_at: datetime
    author: Optional[str]
    category: str
    tags: List[str]
    symbols: List[str]
    sentiment_analysis: Optional[NewsAnalysis]
    image_url: Optional[str]
    reading_time: int  # minutes
    engagement_score: float
    credibility_score: float
    language: str
    region: str

@dataclass
class NewsFilter:
    categories: Optional[List[str]] = None
    symbols: Optional[List[str]] = None
    sentiment: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_credibility: Optional[float] = None
    languages: Optional[List[str]] = None
    content_types: Optional[List[ContentType]] = None

class EnhancedNewsService:
    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url
        self.redis_client = None
        self.session = None
        self.sentiment_engine = None
        
        # Cache settings
        self.cache_ttl = {
            'articles': 1800,  # 30 minutes
            'feeds': 3600,     # 1 hour
            'analysis': 7200,  # 2 hours
        }
        
        # API configurations
        self.api_keys = {
            'newsapi': os.getenv('NEWSAPI_KEY'),
            'finnhub': os.getenv('FINNHUB_API_KEY'),
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
        }
        
        # RSS feeds configuration
        self.rss_feeds = {
            'financial': [
                'https://feeds.bloomberg.com/markets/news.rss',
                'https://www.reuters.com/business/finance/rss',
                'https://www.cnbc.com/id/100003114/device/rss/rss.html',
                'https://feeds.marketwatch.com/marketwatch/realtimeheadlines/',
                'https://www.ft.com/rss/home/us',
            ],
            'crypto': [
                'https://cointelegraph.com/rss',
                'https://decrypt.co/feed',
                'https://www.coindesk.com/arc/outboundfeeds/rss/',
                'https://bitcoinmagazine.com/.rss/full/',
            ],
            'technology': [
                'https://techcrunch.com/feed/',
                'https://www.theverge.com/rss/index.xml',
                'https://feeds.arstechnica.com/arstechnica/index',
            ],
            'economics': [
                'https://www.economist.com/finance-and-economics/rss.xml',
                'https://www.federalreserve.gov/feeds/press_all.xml',
            ]
        }
        
        # Content extraction patterns
        self.content_selectors = {
            'bloomberg.com': {'content': '.body-content', 'author': '.author'},
            'reuters.com': {'content': '.StandardArticleBody_body', 'author': '.BylineBar_byline'},
            'cnbc.com': {'content': '.ArticleBody-articleBody', 'author': '.BylineBar_byline'},
            'marketwatch.com': {'content': '.article__content', 'author': '.author'},
            'coindesk.com': {'content': '.at-content', 'author': '.at-author'},
            'cointelegraph.com': {'content': '.post-content', 'author': '.post-meta__author'},
        }
        
        # Source credibility scores
        self.source_credibility = {
            'bloomberg.com': 0.95,
            'reuters.com': 0.95,
            'wsj.com': 0.95,
            'ft.com': 0.90,
            'cnbc.com': 0.85,
            'marketwatch.com': 0.80,
            'coindesk.com': 0.85,
            'cointelegraph.com': 0.75,
            'techcrunch.com': 0.80,
            'default': 0.60
        }
        
        # Thread pool for CPU-intensive tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Rate limiting
        self.rate_limits = {
            'newsapi': {'calls': 100, 'period': 3600},  # 100 calls per hour
            'finnhub': {'calls': 60, 'period': 60},     # 60 calls per minute
        }
        
        self.rate_limit_counters = {}
    
    async def initialize(self):
        """Initialize the service."""
        try:
            # Initialize Redis connection
            if self.redis_url:
                self.redis_client = await aioredis.from_url(self.redis_url)
            
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={'User-Agent': 'FinScope News Aggregator 1.0'}
            )
            
            # Initialize sentiment engine
            self.sentiment_engine = await get_sentiment_engine()
            
            logger.info("Enhanced news service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing enhanced news service: {e}")
            raise
    
    async def fetch_latest_news(
        self,
        categories: List[str] = None,
        limit: int = 50,
        include_analysis: bool = True
    ) -> List[NewsArticle]:
        """Fetch latest news from all sources."""
        try:
            # Check cache first
            cache_key = f"latest_news:{hash(str(categories))}:{limit}"
            if self.redis_client:
                cached_result = await self.redis_client.get(cache_key)
                if cached_result:
                    articles_data = json.loads(cached_result)
                    return [self._dict_to_article(article) for article in articles_data]
            
            # Fetch from multiple sources concurrently
            tasks = []
            
            # RSS feeds
            if not categories or 'financial' in categories:
                tasks.append(self._fetch_rss_articles('financial'))
            if not categories or 'crypto' in categories:
                tasks.append(self._fetch_rss_articles('crypto'))
            if not categories or 'technology' in categories:
                tasks.append(self._fetch_rss_articles('technology'))
            
            # API sources
            if self.api_keys['newsapi']:
                tasks.append(self._fetch_newsapi_articles(categories))
            if self.api_keys['finnhub']:
                tasks.append(self._fetch_finnhub_news())
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine and deduplicate articles
            all_articles = []
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error fetching news: {result}")
            
            # Remove duplicates and sort by date
            unique_articles = self._deduplicate_articles(all_articles)
            unique_articles.sort(key=lambda x: x.published_at, reverse=True)
            
            # Limit results
            limited_articles = unique_articles[:limit]
            
            # Add sentiment analysis if requested
            if include_analysis:
                limited_articles = await self._add_sentiment_analysis(limited_articles)
            
            # Cache the results
            if self.redis_client:
                articles_data = [self._article_to_dict(article) for article in limited_articles]
                await self.redis_client.setex(
                    cache_key,
                    self.cache_ttl['articles'],
                    json.dumps(articles_data, default=str)
                )
            
            return limited_articles
            
        except Exception as e:
            logger.error(f"Error fetching latest news: {e}")
            return []
    
    async def search_news(
        self,
        query: str,
        filters: NewsFilter = None,
        limit: int = 20
    ) -> List[NewsArticle]:
        """Search news articles with advanced filtering."""
        try:
            # Fetch articles from various sources
            articles = await self.fetch_latest_news(limit=limit * 2)
            
            # Apply text search
            if query:
                query_lower = query.lower()
                articles = [
                    article for article in articles
                    if (query_lower in article.title.lower() or
                        query_lower in article.content.lower() or
                        any(query_lower in tag.lower() for tag in article.tags) or
                        any(query_lower in symbol.lower() for symbol in article.symbols))
                ]
            
            # Apply filters
            if filters:
                articles = self._apply_filters(articles, filters)
            
            return articles[:limit]
            
        except Exception as e:
            logger.error(f"Error searching news: {e}")
            return []
    
    async def get_trending_topics(self, hours: int = 24) -> Dict[str, Any]:
        """Get trending topics and keywords."""
        try:
            # Fetch recent articles
            since = datetime.now() - timedelta(hours=hours)
            articles = await self.fetch_latest_news(limit=200)
            
            # Filter by time
            recent_articles = [
                article for article in articles
                if article.published_at >= since
            ]
            
            # Analyze trends
            trends = await self._analyze_trends(recent_articles)
            
            return trends
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return {}
    
    async def get_sentiment_overview(self, categories: List[str] = None) -> Dict[str, Any]:
        """Get overall sentiment analysis for news."""
        try:
            # Fetch recent articles with sentiment analysis
            articles = await self.fetch_latest_news(
                categories=categories,
                limit=100,
                include_analysis=True
            )
            
            # Filter articles with sentiment analysis
            analyzed_articles = [
                article for article in articles
                if article.sentiment_analysis is not None
            ]
            
            if not analyzed_articles:
                return {'error': 'No articles with sentiment analysis found'}
            
            # Calculate overall sentiment metrics
            sentiment_overview = self._calculate_sentiment_overview(analyzed_articles)
            
            return sentiment_overview
            
        except Exception as e:
            logger.error(f"Error getting sentiment overview: {e}")
            return {'error': str(e)}
    
    async def _fetch_rss_articles(self, category: str) -> List[NewsArticle]:
        """Fetch articles from RSS feeds."""
        articles = []
        feeds = self.rss_feeds.get(category, [])
        
        for feed_url in feeds:
            try:
                # Fetch RSS feed
                async with self.session.get(feed_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parse RSS feed
                        loop = asyncio.get_event_loop()
                        feed = await loop.run_in_executor(
                            self.executor,
                            feedparser.parse,
                            content
                        )
                        
                        # Process entries
                        for entry in feed.entries[:10]:  # Limit per feed
                            article = await self._process_rss_entry(entry, category)
                            if article:
                                articles.append(article)
                
            except Exception as e:
                logger.error(f"Error fetching RSS feed {feed_url}: {e}")
                continue
        
        return articles
    
    async def _process_rss_entry(self, entry: Any, category: str) -> Optional[NewsArticle]:
        """Process a single RSS entry."""
        try:
            # Extract basic information
            title = getattr(entry, 'title', '')
            url = getattr(entry, 'link', '')
            summary = getattr(entry, 'summary', '')
            
            if not title or not url:
                return None
            
            # Parse publication date
            published_at = datetime.now()
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6])
            
            # Extract full content
            content = await self._extract_full_content(url, summary)
            
            # Generate article ID
            article_id = hashlib.md5(f"{title}{url}".encode()).hexdigest()
            
            # Determine source
            source = urlparse(url).netloc
            
            # Calculate reading time
            reading_time = max(1, len(content.split()) // 200)  # ~200 WPM
            
            # Get credibility score
            credibility_score = self.source_credibility.get(
                source,
                self.source_credibility['default']
            )
            
            return NewsArticle(
                id=article_id,
                title=title,
                content=content,
                summary=summary,
                url=url,
                source=source,
                source_type=NewsSource.RSS_FEED,
                content_type=ContentType.ARTICLE,
                published_at=published_at,
                author=getattr(entry, 'author', None),
                category=category,
                tags=[],
                symbols=[],
                sentiment_analysis=None,
                image_url=None,
                reading_time=reading_time,
                engagement_score=0.5,
                credibility_score=credibility_score,
                language='en',
                region='global'
            )
            
        except Exception as e:
            logger.error(f"Error processing RSS entry: {e}")
            return None
    
    async def _extract_full_content(self, url: str, fallback: str) -> str:
        """Extract full article content from URL."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return fallback
                
                html = await response.text()
                
                # Parse HTML
                loop = asyncio.get_event_loop()
                soup = await loop.run_in_executor(
                    self.executor,
                    BeautifulSoup,
                    html,
                    'html.parser'
                )
                
                # Try site-specific selectors
                domain = urlparse(url).netloc
                selectors = self.content_selectors.get(domain, {})
                
                content = None
                if 'content' in selectors:
                    content_elem = soup.select_one(selectors['content'])
                    if content_elem:
                        content = content_elem.get_text(strip=True)
                
                # Fallback to common selectors
                if not content:
                    for selector in ['article', '.article-content', '.post-content', '.entry-content']:
                        elem = soup.select_one(selector)
                        if elem:
                            content = elem.get_text(strip=True)
                            break
                
                # Clean up content
                if content:
                    content = re.sub(r'\s+', ' ', content)
                    content = content.strip()
                    
                    # Limit content length
                    if len(content) > 5000:
                        content = content[:5000] + "..."
                    
                    return content
                
                return fallback
                
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return fallback
    
    async def _fetch_newsapi_articles(self, categories: List[str] = None) -> List[NewsArticle]:
        """Fetch articles from NewsAPI."""
        if not self.api_keys['newsapi']:
            return []
        
        try:
            # Check rate limit
            if not await self._check_rate_limit('newsapi'):
                logger.warning("NewsAPI rate limit exceeded")
                return []
            
            articles = []
            
            # Define search queries for different categories
            queries = {
                'financial': 'stocks OR finance OR market OR trading',
                'crypto': 'bitcoin OR cryptocurrency OR blockchain',
                'technology': 'technology OR tech OR startup'
            }
            
            target_categories = categories or list(queries.keys())
            
            for category in target_categories:
                if category in queries:
                    url = "https://newsapi.org/v2/everything"
                    params = {
                        'q': queries[category],
                        'language': 'en',
                        'sortBy': 'publishedAt',
                        'pageSize': 20,
                        'apiKey': self.api_keys['newsapi']
                    }
                    
                    async with self.session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for item in data.get('articles', []):
                                article = await self._process_newsapi_article(item, category)
                                if article:
                                    articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI articles: {e}")
            return []
    
    async def _process_newsapi_article(self, item: Dict, category: str) -> Optional[NewsArticle]:
        """Process a NewsAPI article."""
        try:
            title = item.get('title', '')
            url = item.get('url', '')
            
            if not title or not url:
                return None
            
            # Generate article ID
            article_id = hashlib.md5(f"{title}{url}".encode()).hexdigest()
            
            # Parse publication date
            published_at = datetime.now()
            if item.get('publishedAt'):
                published_at = datetime.fromisoformat(
                    item['publishedAt'].replace('Z', '+00:00')
                )
            
            # Get source info
            source_info = item.get('source', {})
            source = source_info.get('name', urlparse(url).netloc)
            
            # Get content
            content = item.get('content', item.get('description', ''))
            
            # Calculate reading time
            reading_time = max(1, len(content.split()) // 200)
            
            # Get credibility score
            domain = urlparse(url).netloc
            credibility_score = self.source_credibility.get(
                domain,
                self.source_credibility['default']
            )
            
            return NewsArticle(
                id=article_id,
                title=title,
                content=content,
                summary=item.get('description', ''),
                url=url,
                source=source,
                source_type=NewsSource.NEWSAPI,
                content_type=ContentType.ARTICLE,
                published_at=published_at,
                author=item.get('author'),
                category=category,
                tags=[],
                symbols=[],
                sentiment_analysis=None,
                image_url=item.get('urlToImage'),
                reading_time=reading_time,
                engagement_score=0.5,
                credibility_score=credibility_score,
                language='en',
                region='global'
            )
            
        except Exception as e:
            logger.error(f"Error processing NewsAPI article: {e}")
            return None
    
    async def _fetch_finnhub_news(self) -> List[NewsArticle]:
        """Fetch news from Finnhub API."""
        if not self.api_keys['finnhub']:
            return []
        
        try:
            # Check rate limit
            if not await self._check_rate_limit('finnhub'):
                logger.warning("Finnhub rate limit exceeded")
                return []
            
            url = "https://finnhub.io/api/v1/news"
            params = {
                'category': 'general',
                'token': self.api_keys['finnhub']
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    articles = []
                    for item in data:
                        article = await self._process_finnhub_article(item)
                        if article:
                            articles.append(article)
                    
                    return articles
                
                return []
                
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")
            return []
    
    async def _process_finnhub_article(self, item: Dict) -> Optional[NewsArticle]:
        """Process a Finnhub news article."""
        try:
            title = item.get('headline', '')
            url = item.get('url', '')
            
            if not title or not url:
                return None
            
            # Generate article ID
            article_id = hashlib.md5(f"{title}{url}".encode()).hexdigest()
            
            # Parse publication date
            published_at = datetime.now()
            if item.get('datetime'):
                published_at = datetime.fromtimestamp(item['datetime'])
            
            # Get content
            content = item.get('summary', '')
            
            # Calculate reading time
            reading_time = max(1, len(content.split()) // 200)
            
            # Get credibility score
            domain = urlparse(url).netloc
            credibility_score = self.source_credibility.get(
                domain,
                self.source_credibility['default']
            )
            
            return NewsArticle(
                id=article_id,
                title=title,
                content=content,
                summary=content,
                url=url,
                source=item.get('source', domain),
                source_type=NewsSource.FINNHUB,
                content_type=ContentType.ARTICLE,
                published_at=published_at,
                author=None,
                category='financial',
                tags=[],
                symbols=item.get('related', '').split(',') if item.get('related') else [],
                sentiment_analysis=None,
                image_url=item.get('image'),
                reading_time=reading_time,
                engagement_score=0.5,
                credibility_score=credibility_score,
                language='en',
                region='global'
            )
            
        except Exception as e:
            logger.error(f"Error processing Finnhub article: {e}")
            return None
    
    async def _add_sentiment_analysis(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Add sentiment analysis to articles."""
        if not self.sentiment_engine:
            return articles
        
        # Process articles in batches to avoid overwhelming the sentiment engine
        batch_size = 10
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            # Process batch concurrently
            tasks = [
                self.sentiment_engine.analyze_news_article(
                    article.title,
                    article.content,
                    article.url
                )
                for article in batch
            ]
            
            try:
                analyses = await asyncio.gather(*tasks, return_exceptions=True)
                
                for j, analysis in enumerate(analyses):
                    if isinstance(analysis, NewsAnalysis):
                        batch[j].sentiment_analysis = analysis
                        # Update tags and symbols from analysis
                        batch[j].tags.extend(analysis.tags)
                        batch[j].symbols.extend(analysis.related_symbols)
                        # Remove duplicates
                        batch[j].tags = list(set(batch[j].tags))
                        batch[j].symbols = list(set(batch[j].symbols))
                    
            except Exception as e:
                logger.error(f"Error adding sentiment analysis to batch: {e}")
        
        return articles
    
    def _deduplicate_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on title similarity."""
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            # Create a normalized title for comparison
            normalized_title = re.sub(r'[^a-zA-Z0-9\s]', '', article.title.lower())
            normalized_title = ' '.join(normalized_title.split())
            
            if normalized_title not in seen_titles:
                seen_titles.add(normalized_title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _apply_filters(self, articles: List[NewsArticle], filters: NewsFilter) -> List[NewsArticle]:
        """Apply filters to articles."""
        filtered_articles = articles
        
        if filters.categories:
            filtered_articles = [
                article for article in filtered_articles
                if article.category in filters.categories
            ]
        
        if filters.symbols:
            filtered_articles = [
                article for article in filtered_articles
                if any(symbol in article.symbols for symbol in filters.symbols)
            ]
        
        if filters.sentiment and filtered_articles:
            filtered_articles = [
                article for article in filtered_articles
                if (article.sentiment_analysis and
                    article.sentiment_analysis.sentiment.label.value in filters.sentiment)
            ]
        
        if filters.sources:
            filtered_articles = [
                article for article in filtered_articles
                if article.source in filters.sources
            ]
        
        if filters.date_from:
            filtered_articles = [
                article for article in filtered_articles
                if article.published_at >= filters.date_from
            ]
        
        if filters.date_to:
            filtered_articles = [
                article for article in filtered_articles
                if article.published_at <= filters.date_to
            ]
        
        if filters.min_credibility:
            filtered_articles = [
                article for article in filtered_articles
                if article.credibility_score >= filters.min_credibility
            ]
        
        if filters.languages:
            filtered_articles = [
                article for article in filtered_articles
                if article.language in filters.languages
            ]
        
        if filters.content_types:
            filtered_articles = [
                article for article in filtered_articles
                if article.content_type in filters.content_types
            ]
        
        return filtered_articles
    
    async def _analyze_trends(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Analyze trending topics from articles."""
        try:
            # Extract all tags and symbols
            all_tags = []
            all_symbols = []
            sentiment_scores = []
            
            for article in articles:
                all_tags.extend(article.tags)
                all_symbols.extend(article.symbols)
                
                if article.sentiment_analysis:
                    sentiment_scores.append(article.sentiment_analysis.sentiment.score)
            
            # Count occurrences
            from collections import Counter
            tag_counts = Counter(all_tags)
            symbol_counts = Counter(all_symbols)
            
            # Calculate average sentiment
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
            
            return {
                'trending_tags': dict(tag_counts.most_common(10)),
                'trending_symbols': dict(symbol_counts.most_common(10)),
                'average_sentiment': avg_sentiment,
                'total_articles': len(articles),
                'sentiment_distribution': self._calculate_sentiment_distribution(sentiment_scores)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {}
    
    def _calculate_sentiment_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate sentiment distribution."""
        distribution = {'very_positive': 0, 'positive': 0, 'neutral': 0, 'negative': 0, 'very_negative': 0}
        
        for score in scores:
            if score > 0.6:
                distribution['very_positive'] += 1
            elif score > 0.2:
                distribution['positive'] += 1
            elif score > -0.2:
                distribution['neutral'] += 1
            elif score > -0.6:
                distribution['negative'] += 1
            else:
                distribution['very_negative'] += 1
        
        return distribution
    
    def _calculate_sentiment_overview(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Calculate overall sentiment metrics."""
        sentiment_scores = []
        market_sentiments = []
        impact_scores = []
        
        for article in articles:
            if article.sentiment_analysis:
                sentiment_scores.append(article.sentiment_analysis.sentiment.score)
                market_sentiments.append(article.sentiment_analysis.sentiment.market_sentiment.value)
                impact_scores.append(article.sentiment_analysis.impact_score)
        
        if not sentiment_scores:
            return {'error': 'No sentiment data available'}
        
        # Calculate metrics
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        avg_impact = sum(impact_scores) / len(impact_scores)
        
        # Count market sentiments
        from collections import Counter
        market_sentiment_counts = Counter(market_sentiments)
        
        return {
            'average_sentiment': avg_sentiment,
            'average_impact': avg_impact,
            'sentiment_distribution': self._calculate_sentiment_distribution(sentiment_scores),
            'market_sentiment_distribution': dict(market_sentiment_counts),
            'total_analyzed_articles': len(sentiment_scores)
        }
    
    async def _check_rate_limit(self, service: str) -> bool:
        """Check if we can make a request to the service."""
        if service not in self.rate_limits:
            return True
        
        current_time = datetime.now()
        rate_limit = self.rate_limits[service]
        
        # Initialize counter if not exists
        if service not in self.rate_limit_counters:
            self.rate_limit_counters[service] = {'count': 0, 'reset_time': current_time}
        
        counter = self.rate_limit_counters[service]
        
        # Reset counter if period has passed
        if current_time >= counter['reset_time'] + timedelta(seconds=rate_limit['period']):
            counter['count'] = 0
            counter['reset_time'] = current_time
        
        # Check if we can make a request
        if counter['count'] >= rate_limit['calls']:
            return False
        
        # Increment counter
        counter['count'] += 1
        return True
    
    def _article_to_dict(self, article: NewsArticle) -> Dict[str, Any]:
        """Convert article to dictionary for caching."""
        article_dict = asdict(article)
        # Convert datetime to string
        article_dict['published_at'] = article.published_at.isoformat()
        # Convert sentiment analysis to dict if present
        if article.sentiment_analysis:
            article_dict['sentiment_analysis'] = asdict(article.sentiment_analysis)
        return article_dict
    
    def _dict_to_article(self, article_dict: Dict[str, Any]) -> NewsArticle:
        """Convert dictionary back to article object."""
        # Convert string back to datetime
        article_dict['published_at'] = datetime.fromisoformat(article_dict['published_at'])
        
        # Convert sentiment analysis back to object if present
        if article_dict.get('sentiment_analysis'):
            sentiment_dict = article_dict['sentiment_analysis']
            # This is a simplified conversion - in practice, you'd want to properly reconstruct the objects
            article_dict['sentiment_analysis'] = None  # For now, skip reconstruction
        
        return NewsArticle(**article_dict)
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.session:
            await self.session.close()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.executor:
            self.executor.shutdown(wait=True)

# Global instance
enhanced_news_service = None

async def get_enhanced_news_service() -> EnhancedNewsService:
    """Get or create enhanced news service instance."""
    global enhanced_news_service
    if enhanced_news_service is None:
        enhanced_news_service = EnhancedNewsService()
        await enhanced_news_service.initialize()
    return enhanced_news_service