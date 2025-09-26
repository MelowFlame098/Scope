import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import re
import time
from urllib.parse import quote_plus
import praw
from textblob import TextBlob
import tweepy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SocialPost:
    """Data class for social media posts"""
    id: str
    title: str
    content: str
    author: str
    platform: str
    url: str
    created_at: datetime
    score: Optional[int] = None
    upvotes: Optional[int] = None
    downvotes: Optional[int] = None
    comments_count: Optional[int] = None
    sentiment_score: Optional[float] = None
    symbols: List[str] = None
    subreddit: Optional[str] = None
    hashtags: List[str] = None

class SocialScraperService:
    """Advanced social media scraper for financial sentiment analysis"""
    
    def __init__(self, reddit_config: Dict = None, twitter_config: Dict = None):
        self.reddit_config = reddit_config or {}
        self.twitter_config = twitter_config or {}
        self.session = None
        
        # Initialize Reddit client if config provided
        self.reddit_client = None
        if self.reddit_config:
            try:
                self.reddit_client = praw.Reddit(
                    client_id=self.reddit_config.get('client_id'),
                    client_secret=self.reddit_config.get('client_secret'),
                    user_agent=self.reddit_config.get('user_agent', 'FinScope/1.0')
                )
            except Exception as e:
                logger.error(f"Failed to initialize Reddit client: {e}")
        
        # Initialize Twitter client if config provided
        self.twitter_client = None
        if self.twitter_config:
            try:
                auth = tweepy.OAuthHandler(
                    self.twitter_config.get('consumer_key'),
                    self.twitter_config.get('consumer_secret')
                )
                auth.set_access_token(
                    self.twitter_config.get('access_token'),
                    self.twitter_config.get('access_token_secret')
                )
                self.twitter_client = tweepy.API(auth, wait_on_rate_limit=True)
            except Exception as e:
                logger.error(f"Failed to initialize Twitter client: {e}")
        
        # Financial subreddits to monitor
        self.financial_subreddits = [
            'wallstreetbets',
            'investing',
            'stocks',
            'SecurityAnalysis',
            'ValueInvesting',
            'financialindependence',
            'StockMarket',
            'pennystocks',
            'options',
            'cryptocurrency',
            'Bitcoin',
            'ethereum',
            'CryptoCurrency'
        ]
        
        # Rate limiting
        self.rate_limits = {
            'reddit': {'requests_per_minute': 60, 'last_request': 0, 'request_count': 0},
            'twitter': {'requests_per_minute': 300, 'last_request': 0, 'request_count': 0}
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    def _check_rate_limit(self, platform: str) -> bool:
        """Check if we can make a request to the platform"""
        current_time = time.time()
        rate_limit = self.rate_limits.get(platform, {'requests_per_minute': 60, 'last_request': 0, 'request_count': 0})
        
        # Reset counter if a minute has passed
        if current_time - rate_limit['last_request'] >= 60:
            rate_limit['request_count'] = 0
            rate_limit['last_request'] = current_time
        
        # Check if we can make another request
        if rate_limit['request_count'] >= rate_limit['requests_per_minute']:
            return False
        
        rate_limit['request_count'] += 1
        return True

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text"""
        # Patterns for stock symbols in social media
        symbol_patterns = [
            r'\$[A-Z]{1,5}\b',  # $AAPL format
            r'\b[A-Z]{1,5}\b(?=\s|$)',  # Standalone uppercase letters
            r'#[A-Z]{1,5}\b'  # #AAPL hashtag format
        ]
        
        symbols = set()
        for pattern in symbol_patterns:
            matches = re.findall(pattern, text.upper())
            # Clean up matches
            cleaned_matches = [match.replace('$', '').replace('#', '') for match in matches]
            symbols.update(cleaned_matches)
        
        # Filter out common words
        common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'WHAT', 'UP', 'OUT', 'IF', 'ABOUT', 'WHO', 'GET', 'WHICH', 'GO', 'ME', 'WHEN', 'MAKE', 'LIKE', 'TIME', 'NO', 'JUST', 'HIM', 'KNOW', 'TAKE', 'PEOPLE', 'INTO', 'YEAR', 'YOUR', 'GOOD', 'SOME', 'COULD', 'THEM', 'SEE', 'OTHER', 'THAN', 'THEN', 'NOW', 'LOOK', 'ONLY', 'COME', 'ITS', 'OVER', 'THINK', 'ALSO', 'BACK', 'AFTER', 'USE', 'TWO', 'HOW', 'WORK', 'FIRST', 'WELL', 'WAY', 'EVEN', 'NEW', 'WANT', 'BECAUSE', 'ANY', 'THESE', 'GIVE', 'DAY', 'MOST', 'US', 'TO', 'IS', 'IT', 'ON', 'BE', 'AT', 'BY', 'THIS', 'HAVE', 'FROM', 'OR', 'AS', 'HE', 'AN', 'WILL', 'MY', 'WOULD', 'THERE', 'BEEN', 'MAY', 'SO', 'WE', 'DO', 'SAY', 'SHE', 'EACH', 'WHICH', 'THEIR', 'HAS', 'HIS'}
        
        filtered_symbols = [s for s in symbols if s not in common_words and len(s) <= 5 and len(s) >= 1]
        return list(filtered_symbols)

    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract hashtags from text"""
        hashtag_pattern = r'#\w+'
        hashtags = re.findall(hashtag_pattern, text)
        return [tag.lower() for tag in hashtags]

    def _calculate_sentiment(self, text: str) -> float:
        """Calculate sentiment score using TextBlob"""
        try:
            blob = TextBlob(text)
            # Return polarity score (-1 to 1)
            return blob.sentiment.polarity
        except Exception as e:
            logger.error(f"Error calculating sentiment: {e}")
            return 0.0

    async def scrape_reddit_posts(self, subreddits: List[str] = None, limit: int = 100, time_filter: str = 'day') -> List[SocialPost]:
        """Scrape Reddit posts from financial subreddits"""
        if not self.reddit_client:
            logger.warning("Reddit client not initialized")
            return []

        subreddits_to_scrape = subreddits or self.financial_subreddits
        posts = []

        try:
            for subreddit_name in subreddits_to_scrape:
                if not self._check_rate_limit('reddit'):
                    logger.warning("Reddit rate limit exceeded")
                    break

                try:
                    subreddit = self.reddit_client.subreddit(subreddit_name)
                    
                    # Get hot posts
                    for submission in subreddit.hot(limit=limit // len(subreddits_to_scrape)):
                        try:
                            # Skip if too old
                            post_time = datetime.fromtimestamp(submission.created_utc)
                            if time_filter == 'day' and post_time < datetime.now() - timedelta(days=1):
                                continue
                            elif time_filter == 'week' and post_time < datetime.now() - timedelta(weeks=1):
                                continue

                            # Extract content
                            content = f"{submission.title} {submission.selftext}"
                            symbols = self._extract_symbols(content)
                            hashtags = self._extract_hashtags(content)
                            sentiment = self._calculate_sentiment(content)

                            post = SocialPost(
                                id=submission.id,
                                title=submission.title,
                                content=submission.selftext,
                                author=str(submission.author) if submission.author else 'deleted',
                                platform='reddit',
                                url=f"https://reddit.com{submission.permalink}",
                                created_at=post_time,
                                score=submission.score,
                                upvotes=submission.ups,
                                downvotes=submission.downs,
                                comments_count=submission.num_comments,
                                sentiment_score=sentiment,
                                symbols=symbols,
                                subreddit=subreddit_name,
                                hashtags=hashtags
                            )
                            posts.append(post)

                        except Exception as e:
                            logger.error(f"Error processing Reddit submission: {e}")
                            continue

                except Exception as e:
                    logger.error(f"Error accessing subreddit {subreddit_name}: {e}")
                    continue

            logger.info(f"Scraped {len(posts)} Reddit posts")
            return posts

        except Exception as e:
            logger.error(f"Error scraping Reddit: {e}")
            return []

    async def scrape_reddit_comments(self, post_ids: List[str], limit: int = 50) -> List[SocialPost]:
        """Scrape Reddit comments for specific posts"""
        if not self.reddit_client:
            logger.warning("Reddit client not initialized")
            return []

        comments = []

        try:
            for post_id in post_ids:
                if not self._check_rate_limit('reddit'):
                    break

                try:
                    submission = self.reddit_client.submission(id=post_id)
                    submission.comments.replace_more(limit=0)

                    for comment in submission.comments.list()[:limit]:
                        try:
                            if hasattr(comment, 'body') and comment.body != '[deleted]':
                                symbols = self._extract_symbols(comment.body)
                                hashtags = self._extract_hashtags(comment.body)
                                sentiment = self._calculate_sentiment(comment.body)

                                comment_post = SocialPost(
                                    id=comment.id,
                                    title=f"Comment on: {submission.title}",
                                    content=comment.body,
                                    author=str(comment.author) if comment.author else 'deleted',
                                    platform='reddit_comment',
                                    url=f"https://reddit.com{comment.permalink}",
                                    created_at=datetime.fromtimestamp(comment.created_utc),
                                    score=comment.score,
                                    sentiment_score=sentiment,
                                    symbols=symbols,
                                    hashtags=hashtags
                                )
                                comments.append(comment_post)

                        except Exception as e:
                            logger.error(f"Error processing Reddit comment: {e}")
                            continue

                except Exception as e:
                    logger.error(f"Error accessing Reddit post {post_id}: {e}")
                    continue

            logger.info(f"Scraped {len(comments)} Reddit comments")
            return comments

        except Exception as e:
            logger.error(f"Error scraping Reddit comments: {e}")
            return []

    async def search_twitter_posts(self, query: str, limit: int = 100) -> List[SocialPost]:
        """Search Twitter posts for financial content"""
        if not self.twitter_client:
            logger.warning("Twitter client not initialized")
            return []

        posts = []

        try:
            if not self._check_rate_limit('twitter'):
                logger.warning("Twitter rate limit exceeded")
                return []

            # Search tweets
            tweets = tweepy.Cursor(
                self.twitter_client.search_tweets,
                q=query,
                lang='en',
                result_type='recent',
                tweet_mode='extended'
            ).items(limit)

            for tweet in tweets:
                try:
                    symbols = self._extract_symbols(tweet.full_text)
                    hashtags = self._extract_hashtags(tweet.full_text)
                    sentiment = self._calculate_sentiment(tweet.full_text)

                    post = SocialPost(
                        id=str(tweet.id),
                        title=tweet.full_text[:100] + '...' if len(tweet.full_text) > 100 else tweet.full_text,
                        content=tweet.full_text,
                        author=tweet.user.screen_name,
                        platform='twitter',
                        url=f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}",
                        created_at=tweet.created_at,
                        score=tweet.favorite_count + tweet.retweet_count,
                        upvotes=tweet.favorite_count,
                        comments_count=tweet.reply_count if hasattr(tweet, 'reply_count') else 0,
                        sentiment_score=sentiment,
                        symbols=symbols,
                        hashtags=hashtags
                    )
                    posts.append(post)

                except Exception as e:
                    logger.error(f"Error processing tweet: {e}")
                    continue

            logger.info(f"Scraped {len(posts)} Twitter posts")
            return posts

        except Exception as e:
            logger.error(f"Error scraping Twitter: {e}")
            return []

    async def get_symbol_sentiment(self, symbol: str, hours_back: int = 24) -> Dict[str, Any]:
        """Get sentiment analysis for a specific symbol"""
        try:
            # Search Reddit
            reddit_posts = await self.scrape_reddit_posts(limit=50)
            symbol_reddit_posts = [post for post in reddit_posts if symbol in post.symbols]

            # Search Twitter
            twitter_query = f"${symbol} OR #{symbol} OR {symbol}"
            twitter_posts = await self.search_twitter_posts(twitter_query, limit=50)

            # Combine all posts
            all_posts = symbol_reddit_posts + twitter_posts

            # Filter by time
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            recent_posts = [post for post in all_posts if post.created_at >= cutoff_time]

            if not recent_posts:
                return {
                    'symbol': symbol,
                    'sentiment_score': 0.0,
                    'post_count': 0,
                    'platform_breakdown': {},
                    'time_range': f'{hours_back} hours'
                }

            # Calculate aggregate sentiment
            sentiments = [post.sentiment_score for post in recent_posts if post.sentiment_score is not None]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

            # Platform breakdown
            platform_breakdown = {}
            for post in recent_posts:
                platform = post.platform
                if platform not in platform_breakdown:
                    platform_breakdown[platform] = {'count': 0, 'avg_sentiment': 0.0, 'sentiments': []}
                
                platform_breakdown[platform]['count'] += 1
                if post.sentiment_score is not None:
                    platform_breakdown[platform]['sentiments'].append(post.sentiment_score)

            # Calculate platform averages
            for platform, data in platform_breakdown.items():
                if data['sentiments']:
                    data['avg_sentiment'] = sum(data['sentiments']) / len(data['sentiments'])

            return {
                'symbol': symbol,
                'sentiment_score': avg_sentiment,
                'post_count': len(recent_posts),
                'platform_breakdown': platform_breakdown,
                'time_range': f'{hours_back} hours',
                'posts': recent_posts[:20]  # Return top 20 posts
            }

        except Exception as e:
            logger.error(f"Error getting sentiment for symbol {symbol}: {e}")
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'post_count': 0,
                'platform_breakdown': {},
                'time_range': f'{hours_back} hours',
                'error': str(e)
            }

# Example usage
async def main():
    """Example usage of the SocialScraperService"""
    # Configure with your API keys
    reddit_config = {
        'client_id': 'your_reddit_client_id',
        'client_secret': 'your_reddit_client_secret',
        'user_agent': 'FinScope/1.0'
    }
    
    twitter_config = {
        'consumer_key': 'your_twitter_consumer_key',
        'consumer_secret': 'your_twitter_consumer_secret',
        'access_token': 'your_twitter_access_token',
        'access_token_secret': 'your_twitter_access_token_secret'
    }

    async with SocialScraperService(reddit_config, twitter_config) as scraper:
        # Scrape Reddit posts
        reddit_posts = await scraper.scrape_reddit_posts(limit=50)
        print(f"Found {len(reddit_posts)} Reddit posts")
        
        # Get sentiment for specific symbol
        sentiment_data = await scraper.get_symbol_sentiment('AAPL')
        print(f"AAPL sentiment: {sentiment_data}")

if __name__ == "__main__":
    asyncio.run(main())