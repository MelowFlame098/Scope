import asyncio
import aiohttp
from typing import Dict, List, Optional, Any
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import os
import logging

from ..repositories.asset import asset_repository
from ..repositories.news import news_repository
from ..models import Asset, NewsArticle
from ..schemas import AssetCreate, AssetUpdate, NewsArticleCreate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataService:
    """
    Service for fetching and managing external market data
    """
    
    def __init__(self):
        self.asset_repo = asset_repository
        self.news_repo = news_repository
        
        # API Keys from environment
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.coingecko_api_key = os.getenv("COINGECKO_API_KEY")
        self.news_api_key = os.getenv("NEWS_API_KEY")
        
        # API URLs
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        self.news_api_url = "https://newsapi.org/v2"
    
    async def fetch_crypto_data(self, db: Session, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Fetch cryptocurrency data from CoinGecko
        
        Args:
            db: Database session
            symbols: Optional list of symbols to fetch
            
        Returns:
            Dictionary with fetch results
        """
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch top cryptocurrencies
                url = f"{self.coingecko_url}/coins/markets"
                params = {
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 100,
                    "page": 1,
                    "sparkline": False,
                    "price_change_percentage": "24h"
                }
                
                if self.coingecko_api_key:
                    params["x_cg_demo_api_key"] = self.coingecko_api_key
                
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return await self._process_crypto_data(db, data)
                    else:
                        logger.error(f"CoinGecko API error: {response.status}")
                        return {"success": False, "error": f"API error: {response.status}"}
        
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return {"success": False, "error": str(e)}
    
    async def fetch_stock_data(self, db: Session, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch stock data from Alpha Vantage
        
        Args:
            db: Database session
            symbols: List of stock symbols
            
        Returns:
            Dictionary with fetch results
        """
        if not self.alpha_vantage_key:
            return {"success": False, "error": "Alpha Vantage API key not configured"}
        
        results = {"success": True, "updated": 0, "errors": []}
        
        try:
            async with aiohttp.ClientSession() as session:
                for symbol in symbols:
                    try:
                        # Fetch quote data
                        params = {
                            "function": "GLOBAL_QUOTE",
                            "symbol": symbol,
                            "apikey": self.alpha_vantage_key
                        }
                        
                        async with session.get(self.alpha_vantage_url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                await self._process_stock_data(db, symbol, data)
                                results["updated"] += 1
                            else:
                                results["errors"].append(f"{symbol}: API error {response.status}")
                        
                        # Rate limiting - Alpha Vantage allows 5 calls per minute for free tier
                        await asyncio.sleep(12)  # 12 seconds between calls
                        
                    except Exception as e:
                        results["errors"].append(f"{symbol}: {str(e)}")
                        logger.error(f"Error fetching data for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error in stock data fetch: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    async def fetch_news_data(self, db: Session, query: str = "cryptocurrency OR bitcoin OR ethereum OR stock market") -> Dict[str, Any]:
        """
        Fetch financial news from News API
        
        Args:
            db: Database session
            query: Search query for news
            
        Returns:
            Dictionary with fetch results
        """
        if not self.news_api_key:
            return {"success": False, "error": "News API key not configured"}
        
        try:
            async with aiohttp.ClientSession() as session:
                # Fetch everything endpoint for financial news
                params = {
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": 50,
                    "apiKey": self.news_api_key
                }
                
                async with session.get(f"{self.news_api_url}/everything", params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return await self._process_news_data(db, data)
                    else:
                        logger.error(f"News API error: {response.status}")
                        return {"success": False, "error": f"API error: {response.status}"}
        
        except Exception as e:
            logger.error(f"Error fetching news data: {e}")
            return {"success": False, "error": str(e)}
    
    async def _process_crypto_data(self, db: Session, data: List[Dict]) -> Dict[str, Any]:
        """
        Process and store cryptocurrency data
        
        Args:
            db: Database session
            data: Raw API data
            
        Returns:
            Processing results
        """
        results = {"success": True, "updated": 0, "created": 0, "errors": []}
        
        for coin_data in data:
            try:
                symbol = coin_data.get("symbol", "").upper()
                name = coin_data.get("name", "")
                
                if not symbol or not name:
                    continue
                
                # Check if asset exists
                existing_asset = self.asset_repo.get_by_symbol(db, symbol=symbol)
                
                price_data = {
                    "current_price": coin_data.get("current_price"),
                    "price_change_24h": coin_data.get("price_change_24h"),
                    "price_change_percentage_24h": coin_data.get("price_change_percentage_24h"),
                    "market_cap": coin_data.get("market_cap"),
                    "volume_24h": coin_data.get("total_volume")
                }
                
                if existing_asset:
                    # Update existing asset
                    self.asset_repo.update_price_data(db, symbol=symbol, price_data=price_data)
                    results["updated"] += 1
                else:
                    # Create new asset
                    asset_data = {
                        "symbol": symbol,
                        "name": name,
                        "category": "crypto",
                        "current_price": price_data["current_price"],
                        "price_change_24h": price_data["price_change_24h"],
                        "price_change_percentage_24h": price_data["price_change_percentage_24h"],
                        "market_cap": price_data["market_cap"],
                        "volume_24h": price_data["volume_24h"],
                        "logo_url": coin_data.get("image"),
                        "last_price_update": datetime.utcnow()
                    }
                    
                    self.asset_repo.create(db, obj_in=asset_data)
                    results["created"] += 1
            
            except Exception as e:
                results["errors"].append(f"{symbol}: {str(e)}")
                logger.error(f"Error processing crypto data for {symbol}: {e}")
        
        return results
    
    async def _process_stock_data(self, db: Session, symbol: str, data: Dict) -> None:
        """
        Process and store stock data
        
        Args:
            db: Database session
            symbol: Stock symbol
            data: Raw API data
        """
        try:
            quote_data = data.get("Global Quote", {})
            
            if not quote_data:
                logger.warning(f"No quote data for {symbol}")
                return
            
            current_price = float(quote_data.get("05. price", 0))
            change = float(quote_data.get("09. change", 0))
            change_percent = quote_data.get("10. change percent", "0%")
            
            # Parse percentage
            change_percentage = float(change_percent.replace("%", ""))
            
            price_data = {
                "current_price": current_price,
                "price_change_24h": change,
                "price_change_percentage_24h": change_percentage,
                "volume_24h": float(quote_data.get("06. volume", 0))
            }
            
            # Check if asset exists
            existing_asset = self.asset_repo.get_by_symbol(db, symbol=symbol)
            
            if existing_asset:
                self.asset_repo.update_price_data(db, symbol=symbol, price_data=price_data)
            else:
                # Create new stock asset
                asset_data = {
                    "symbol": symbol,
                    "name": symbol,  # Would need additional API call for company name
                    "category": "stock",
                    "current_price": current_price,
                    "price_change_24h": change,
                    "price_change_percentage_24h": change_percentage,
                    "volume_24h": float(quote_data.get("06. volume", 0)),
                    "last_price_update": datetime.utcnow()
                }
                
                self.asset_repo.create(db, obj_in=asset_data)
        
        except Exception as e:
            logger.error(f"Error processing stock data for {symbol}: {e}")
            raise
    
    async def _process_news_data(self, db: Session, data: Dict) -> Dict[str, Any]:
        """
        Process and store news data
        
        Args:
            db: Database session
            data: Raw API data
            
        Returns:
            Processing results
        """
        results = {"success": True, "created": 0, "skipped": 0, "errors": []}
        
        articles = data.get("articles", [])
        
        for article_data in articles:
            try:
                url = article_data.get("url")
                
                if not url:
                    continue
                
                # Check if article already exists
                existing_article = self.news_repo.get_by_url(db, url=url)
                
                if existing_article:
                    results["skipped"] += 1
                    continue
                
                # Determine category based on content
                title = article_data.get("title", "").lower()
                content = article_data.get("description", "").lower()
                
                category = "general"
                related_symbols = []
                
                if any(word in title + content for word in ["bitcoin", "btc", "ethereum", "eth", "crypto"]):
                    category = "crypto"
                    if "bitcoin" in title + content or "btc" in title + content:
                        related_symbols.append("BTC")
                    if "ethereum" in title + content or "eth" in title + content:
                        related_symbols.append("ETH")
                elif any(word in title + content for word in ["stock", "nasdaq", "dow", "s&p"]):
                    category = "stocks"
                
                # Parse published date
                published_at = None
                if article_data.get("publishedAt"):
                    try:
                        published_at = datetime.fromisoformat(article_data["publishedAt"].replace("Z", "+00:00"))
                    except:
                        published_at = datetime.utcnow()
                
                news_data = {
                    "title": article_data.get("title", ""),
                    "content": article_data.get("description", ""),
                    "url": url,
                    "source": article_data.get("source", {}).get("name", "Unknown"),
                    "author": article_data.get("author"),
                    "category": category,
                    "related_symbols": related_symbols,
                    "image_url": article_data.get("urlToImage"),
                    "published_at": published_at
                }
                
                self.news_repo.create(db, obj_in=news_data)
                results["created"] += 1
            
            except Exception as e:
                results["errors"].append(f"Article processing error: {str(e)}")
                logger.error(f"Error processing news article: {e}")
        
        return results
    
    def get_stale_assets(self, db: Session, hours: int = 1) -> List[Asset]:
        """
        Get assets with stale price data
        
        Args:
            db: Database session
            hours: Hours to consider data stale
            
        Returns:
            List of assets with stale data
        """
        return self.asset_repo.get_stale_prices(db, hours=hours)
    
    async def refresh_asset_data(self, db: Session, asset_types: List[str] = ["crypto", "stock"]) -> Dict[str, Any]:
        """
        Refresh data for all assets of specified types
        
        Args:
            db: Database session
            asset_types: List of asset types to refresh
            
        Returns:
            Refresh results
        """
        results = {"success": True, "crypto": {}, "stock": {}, "news": {}}
        
        try:
            # Refresh crypto data
            if "crypto" in asset_types:
                results["crypto"] = await self.fetch_crypto_data(db)
            
            # Refresh stock data for existing stocks
            if "stock" in asset_types:
                stock_assets = self.asset_repo.get_by_category(db, category="stock", limit=50)
                stock_symbols = [asset.symbol for asset in stock_assets]
                
                if stock_symbols:
                    results["stock"] = await self.fetch_stock_data(db, stock_symbols)
            
            # Refresh news data
            results["news"] = await self.fetch_news_data(db)
        
        except Exception as e:
            logger.error(f"Error in refresh_asset_data: {e}")
            results["success"] = False
            results["error"] = str(e)
        
        return results

# Create service instance
data_service = DataService()