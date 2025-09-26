import asyncio
import aiohttp
import feedparser
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Data class for news articles"""
    title: str
    content: str
    summary: str
    url: str
    source: str
    published_at: datetime
    author: Optional[str] = None
    tags: List[str] = None
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    symbols: List[str] = None

class NewsScraperService:
    """Advanced news scraper for financial news from multiple sources"""
    
    def __init__(self):
        self.session = None
        self.rate_limits = {
            'yahoo': {'requests_per_minute': 60, 'last_request': 0, 'request_count': 0},
            'reuters': {'requests_per_minute': 30, 'last_request': 0, 'request_count': 0},
            'bloomberg': {'requests_per_minute': 20, 'last_request': 0, 'request_count': 0},
            'marketwatch': {'requests_per_minute': 40, 'last_request': 0, 'request_count': 0},
            'cnbc': {'requests_per_minute': 50, 'last_request': 0, 'request_count': 0}
        }
        
        # RSS Feed URLs
        self.rss_feeds = {
            'yahoo_finance': 'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'reuters_business': 'https://www.reuters.com/business/finance/rss',
            'marketwatch': 'https://feeds.marketwatch.com/marketwatch/topstories/',
            'cnbc_finance': 'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'seeking_alpha': 'https://seekingalpha.com/feed.xml'
        }
        
        # Headers for web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _check_rate_limit(self, source: str) -> bool:
        """Check if we can make a request to the source"""
        current_time = time.time()
        rate_limit = self.rate_limits.get(source, {'requests_per_minute': 30, 'last_request': 0, 'request_count': 0})
        
        # Reset counter if a minute has passed
        if current_time - rate_limit['last_request'] >= 60:
            rate_limit['request_count'] = 0
            rate_limit['last_request'] = current_time
        
        # Check if we can make another request
        if rate_limit['request_count'] >= rate_limit['requests_per_minute']:
            return False
        
        rate_limit['request_count'] += 1
        return True

    async def _fetch_rss_feed(self, url: str, source: str) -> List[Dict]:
        """Fetch and parse RSS feed"""
        try:
            if not self._check_rate_limit(source):
                logger.warning(f"Rate limit exceeded for {source}")
                return []

            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    feed = feedparser.parse(content)
                    
                    articles = []
                    for entry in feed.entries[:20]:  # Limit to 20 articles per feed
                        try:
                            published_at = datetime.now()
                            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                                published_at = datetime(*entry.published_parsed[:6])
                            elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                                published_at = datetime(*entry.updated_parsed[:6])
                            
                            article_data = {
                                'title': entry.get('title', ''),
                                'url': entry.get('link', ''),
                                'summary': entry.get('summary', ''),
                                'published_at': published_at,
                                'source': source,
                                'author': entry.get('author', None)
                            }
                            articles.append(article_data)
                        except Exception as e:
                            logger.error(f"Error parsing RSS entry from {source}: {e}")
                            continue
                    
                    return articles
                else:
                    logger.error(f"Failed to fetch RSS feed from {source}: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error fetching RSS feed from {source}: {e}")
            return []

    async def _scrape_article_content(self, url: str, source: str) -> Optional[str]:
        """Scrape full article content from URL"""
        try:
            if not self._check_rate_limit(source):
                return None

            async with self.session.get(url) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove unwanted elements
                    for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                        element.decompose()
                    
                    # Try different content selectors based on source
                    content_selectors = {
                        'yahoo': ['div.caas-body', 'div.canvas-body', 'div.article-body'],
                        'reuters': ['div.StandardArticleBody_body', 'div.ArticleBodyWrapper'],
                        'marketwatch': ['div.article__body', 'div.entry-content'],
                        'cnbc': ['div.ArticleBody-articleBody', 'div.InlineArticleBody'],
                        'default': ['article', 'div.content', 'div.article-content', 'div.post-content']
                    }
                    
                    selectors = content_selectors.get(source, content_selectors['default'])
                    
                    content = ""
                    for selector in selectors:
                        elements = soup.select(selector)
                        if elements:
                            content = ' '.join([elem.get_text(strip=True) for elem in elements])
                            break
                    
                    # Fallback: get all paragraph text
                    if not content:
                        paragraphs = soup.find_all('p')
                        content = ' '.join([p.get_text(strip=True) for p in paragraphs])
                    
                    # Clean up content
                    content = re.sub(r'\s+', ' ', content).strip()
                    return content[:5000]  # Limit content length
                    
        except Exception as e:
            logger.error(f"Error scraping article content from {url}: {e}")
            return None

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        # Common patterns for stock symbols
        symbol_patterns = [
            r'\b[A-Z]{1,5}\b',  # 1-5 uppercase letters
            r'\$[A-Z]{1,5}\b',  # Dollar sign followed by letters
            r'\b[A-Z]{1,5}:[A-Z]{1,5}\b'  # Exchange:Symbol format
        ]
        
        symbols = set()
        for pattern in symbol_patterns:
            matches = re.findall(pattern, text)
            symbols.update(matches)
        
        # Filter out common words that aren't symbols
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'WHAT', 'UP', 'OUT', 'IF', 'ABOUT', 'WHO', 'GET', 'WHICH', 'GO', 'ME', 'WHEN', 'MAKE', 'CAN', 'LIKE', 'TIME', 'NO', 'JUST', 'HIM', 'KNOW', 'TAKE', 'PEOPLE', 'INTO', 'YEAR', 'YOUR', 'GOOD', 'SOME', 'COULD', 'THEM', 'SEE', 'OTHER', 'THAN', 'THEN', 'NOW', 'LOOK', 'ONLY', 'COME', 'ITS', 'OVER', 'THINK', 'ALSO', 'BACK', 'AFTER', 'USE', 'TWO', 'HOW', 'OUR', 'WORK', 'FIRST', 'WELL', 'WAY', 'EVEN', 'NEW', 'WANT', 'BECAUSE', 'ANY', 'THESE', 'GIVE', 'DAY', 'MOST', 'US'}
        
        filtered_symbols = [s for s in symbols if s not in common_words and len(s) <= 5]
        return list(filtered_symbols)

    async def scrape_financial_news(self, symbols: List[str] = None, hours_back: int = 24) -> List[NewsArticle]:
        """Scrape financial news from multiple sources"""
        all_articles = []
        
        try:
            # Fetch from RSS feeds
            rss_tasks = []
            for source, url in self.rss_feeds.items():
                rss_tasks.append(self._fetch_rss_feed(url, source))
            
            rss_results = await asyncio.gather(*rss_tasks, return_exceptions=True)
            
            # Process RSS results
            for i, result in enumerate(rss_results):
                if isinstance(result, Exception):
                    logger.error(f"RSS feed error: {result}")
                    continue
                
                source = list(self.rss_feeds.keys())[i]
                for article_data in result:
                    # Skip old articles
                    if article_data['published_at'] < datetime.now() - timedelta(hours=hours_back):
                        continue
                    
                    # Scrape full content
                    content = await self._scrape_article_content(article_data['url'], source)
                    if not content:
                        content = article_data['summary']
                    
                    # Extract symbols
                    text_for_analysis = f"{article_data['title']} {content}"
                    extracted_symbols = self._extract_symbols(text_for_analysis)
                    
                    # Filter by symbols if provided
                    if symbols and not any(symbol in extracted_symbols for symbol in symbols):
                        continue
                    
                    article = NewsArticle(
                        title=article_data['title'],
                        content=content,
                        summary=article_data['summary'],
                        url=article_data['url'],
                        source=source,
                        published_at=article_data['published_at'],
                        author=article_data.get('author'),
                        symbols=extracted_symbols
                    )
                    
                    all_articles.append(article)
            
            # Sort by publication date (newest first)
            all_articles.sort(key=lambda x: x.published_at, reverse=True)
            
            logger.info(f"Scraped {len(all_articles)} financial news articles")
            return all_articles[:100]  # Limit to 100 most recent articles
            
        except Exception as e:
            logger.error(f"Error in scrape_financial_news: {e}")
            return []

    async def search_news_by_symbol(self, symbol: str, limit: int = 20) -> List[NewsArticle]:
        """Search for news articles related to a specific symbol"""
        try:
            # Yahoo Finance news search
            search_url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}&lang=en-US&region=US&quotesCount=0&newsCount={limit}"
            
            if not self._check_rate_limit('yahoo'):
                logger.warning("Rate limit exceeded for Yahoo Finance")
                return []

            async with self.session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = []
                    
                    if 'news' in data:
                        for item in data['news']:
                            try:
                                published_at = datetime.fromtimestamp(item.get('providerPublishTime', time.time()))
                                
                                article = NewsArticle(
                                    title=item.get('title', ''),
                                    content=item.get('summary', ''),
                                    summary=item.get('summary', ''),
                                    url=item.get('link', ''),
                                    source='yahoo_finance',
                                    published_at=published_at,
                                    symbols=[symbol]
                                )
                                articles.append(article)
                            except Exception as e:
                                logger.error(f"Error parsing Yahoo Finance news item: {e}")
                                continue
                    
                    return articles
                else:
                    logger.error(f"Failed to search Yahoo Finance news: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching news for symbol {symbol}: {e}")
            return []

# Example usage
async def main():
    """Example usage of the NewsScraperService"""
    async with NewsScraperService() as scraper:
        # Scrape general financial news
        articles = await scraper.scrape_financial_news(hours_back=12)
        print(f"Found {len(articles)} articles")
        
        # Search for specific symbol
        symbol_articles = await scraper.search_news_by_symbol('AAPL')
        print(f"Found {len(symbol_articles)} articles for AAPL")

if __name__ == "__main__":
    asyncio.run(main())