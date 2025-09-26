import asyncio
import aiohttp
import aioredis
import yfinance as yf
import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv
import logging
from functools import wraps
import time
from dataclasses import dataclass

from schemas import AssetResponse, MarketDataPoint, ChartData, MarketOverview, MarketDataResponse

load_dotenv()
logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limiting configuration for different API providers."""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int

class EnhancedMarketDataService:
    """Enhanced Market Data Service with Redis caching, rate limiting, and improved API integrations."""
    
    def __init__(self):
        # API keys
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.coingecko_key = os.getenv('COINGECKO_API_KEY')
        self.fmp_key = os.getenv('FINANCIAL_MODELING_PREP_API_KEY')
        
        # Redis configuration
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = None
        
        # Cache configuration
        self.cache_ttl = {
            'assets': 300,  # 5 minutes
            'prices': 60,   # 1 minute
            'chart_data': 900,  # 15 minutes
            'market_overview': 180,  # 3 minutes
            'news': 300,    # 5 minutes
        }
        
        # Rate limiting configuration
        self.rate_limits = {
            'alpha_vantage': RateLimitConfig(5, 500, 500),
            'finnhub': RateLimitConfig(60, 3600, 86400),
            'coingecko': RateLimitConfig(10, 1000, 10000),
            'yahoo_finance': RateLimitConfig(100, 6000, 144000),
            'fmp': RateLimitConfig(250, 15000, 360000)
        }
        
        # API endpoints
        self.api_endpoints = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'finnhub': 'https://finnhub.io/api/v1',
            'coingecko': 'https://api.coingecko.com/api/v3',
            'fmp': 'https://financialmodelingprep.com/api/v3',
            'yahoo_finance': 'https://query1.finance.yahoo.com/v8/finance/chart'
        }
        
        # Redis will be initialized when needed
        self.redis_initialized = False
        
        # CCXT exchange for crypto data
        self.binance_exchange = None
        self._init_ccxt()
    
    async def _init_redis(self):
        """Initialize Redis connection."""
        if self.redis_initialized:
            return
            
        try:
            self.redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            self.redis_client = None
        finally:
            self.redis_initialized = True
    
    def _init_ccxt(self):
        """Initialize CCXT exchange for crypto data."""
        try:
            self.binance_exchange = ccxt.binance({
                'apiKey': os.getenv('BINANCE_API_KEY'),
                'secret': os.getenv('BINANCE_SECRET'),
                'sandbox': os.getenv('BINANCE_SANDBOX', 'false').lower() == 'true',
                'enableRateLimit': True,
            })
        except Exception as e:
            logger.warning(f"CCXT initialization failed: {e}")
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from Redis cache."""
        if not self.redis_initialized:
            await self._init_redis()
            
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    async def _set_cache(self, key: str, data: Any, ttl: int):
        """Set data in Redis cache."""
        if not self.redis_initialized:
            await self._init_redis()
            
        if not self.redis_client:
            return
        
        try:
            await self.redis_client.setex(key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    async def _check_rate_limit(self, provider: str) -> bool:
        """Check if API call is within rate limits."""
        if not self.redis_initialized:
            await self._init_redis()
            
        if not self.redis_client:
            return True
        
        current_time = int(time.time())
        minute_key = f"rate_limit:{provider}:minute:{current_time // 60}"
        hour_key = f"rate_limit:{provider}:hour:{current_time // 3600}"
        day_key = f"rate_limit:{provider}:day:{current_time // 86400}"
        
        try:
            # Check current counts
            minute_count = await self.redis_client.get(minute_key) or 0
            hour_count = await self.redis_client.get(hour_key) or 0
            day_count = await self.redis_client.get(day_key) or 0
            
            limits = self.rate_limits.get(provider)
            if not limits:
                return True
            
            # Check if within limits
            if (int(minute_count) >= limits.requests_per_minute or
                int(hour_count) >= limits.requests_per_hour or
                int(day_count) >= limits.requests_per_day):
                return False
            
            # Increment counters
            await self.redis_client.incr(minute_key)
            await self.redis_client.expire(minute_key, 60)
            await self.redis_client.incr(hour_key)
            await self.redis_client.expire(hour_key, 3600)
            await self.redis_client.incr(day_key)
            await self.redis_client.expire(day_key, 86400)
            
            return True
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True
    
    async def get_assets(self, category: Optional[str] = None, limit: int = 50, force_refresh: bool = False) -> List[AssetResponse]:
        """Get assets with enhanced caching and multiple data sources."""
        cache_key = f"assets:{category}:{limit}"
        
        # Try cache first
        if not force_refresh:
            cached_data = await self._get_from_cache(cache_key)
            if cached_data:
                return [AssetResponse(**asset) for asset in cached_data]
        
        assets = []
        
        try:
            if category == 'crypto' or category is None:
                crypto_assets = await self._get_enhanced_crypto_assets(limit // 3 if category is None else limit)
                assets.extend(crypto_assets)
            
            if category == 'stock' or category is None:
                stock_assets = await self._get_enhanced_stock_assets(limit // 3 if category is None else limit)
                assets.extend(stock_assets)
            
            if category == 'forex' or category is None:
                forex_assets = await self._get_enhanced_forex_assets(limit // 3 if category is None else limit)
                assets.extend(forex_assets)
            
            # Cache the results
            asset_dicts = [asset.dict() for asset in assets]
            await self._set_cache(cache_key, asset_dicts, self.cache_ttl['assets'])
            
        except Exception as e:
            logger.error(f"Error fetching assets: {e}")
            # Return mock data if all APIs fail
            return self._get_mock_assets(category, limit)
        
        return assets[:limit] if limit else assets
    
    async def _get_enhanced_crypto_assets(self, limit: int) -> List[AssetResponse]:
        """Get cryptocurrency assets from multiple sources."""
        assets = []
        
        # Try CoinGecko first
        if await self._check_rate_limit('coingecko'):
            try:
                url = f"{self.api_endpoints['coingecko']}/coins/markets"
                params = {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': limit,
                    'page': 1,
                    'sparkline': False,
                    'price_change_percentage': '1h,24h,7d'
                }
                
                if self.coingecko_key:
                    params['x_cg_demo_api_key'] = self.coingecko_key
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for coin in data:
                                asset = AssetResponse(
                                    id=coin['id'],
                                    symbol=coin['symbol'].upper(),
                                    name=coin['name'],
                                    category='crypto',
                                    current_price=coin['current_price'],
                                    price_change_24h=coin['price_change_24h'],
                                    price_change_percentage_24h=coin['price_change_percentage_24h'],
                                    market_cap=coin['market_cap'],
                                    volume_24h=coin['total_volume'],
                                    logo_url=coin['image'],
                                    last_price_update=datetime.now(),
                                    additional_data={
                                        'price_change_1h': coin.get('price_change_percentage_1h_in_currency'),
                                        'price_change_7d': coin.get('price_change_percentage_7d_in_currency'),
                                        'circulating_supply': coin.get('circulating_supply'),
                                        'total_supply': coin.get('total_supply'),
                                        'max_supply': coin.get('max_supply')
                                    }
                                )
                                assets.append(asset)
                            
                            return assets
            except Exception as e:
                logger.error(f"CoinGecko API error: {e}")
        
        # Fallback to CCXT if CoinGecko fails
        if self.binance_exchange and len(assets) == 0:
            try:
                markets = self.binance_exchange.load_markets()
                usdt_pairs = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')][:limit]
                
                for symbol in usdt_pairs:
                    ticker = self.binance_exchange.fetch_ticker(symbol)
                    base_currency = symbol.split('/')[0]
                    
                    asset = AssetResponse(
                        id=base_currency.lower(),
                        symbol=base_currency,
                        name=base_currency,
                        category='crypto',
                        current_price=ticker['last'],
                        price_change_24h=ticker['change'],
                        price_change_percentage_24h=ticker['percentage'],
                        volume_24h=ticker['quoteVolume'],
                        last_price_update=datetime.now()
                    )
                    assets.append(asset)
                    
            except Exception as e:
                logger.error(f"CCXT error: {e}")
        
        return assets
    
    async def _get_enhanced_stock_assets(self, limit: int) -> List[AssetResponse]:
        """Get stock assets from multiple sources."""
        assets = []
        
        # Popular stocks to fetch
        popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'DIS', 'V', 'MA', 'JPM',
            'BAC', 'WMT', 'PG', 'JNJ', 'UNH', 'HD', 'VZ', 'T', 'PFE', 'KO', 'PEP'
        ]
        
        # Try Financial Modeling Prep first
        if self.fmp_key and await self._check_rate_limit('fmp'):
            try:
                symbols_str = ','.join(popular_stocks[:limit])
                url = f"{self.api_endpoints['fmp']}/quote/{symbols_str}"
                params = {'apikey': self.fmp_key}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for stock in data:
                                asset = AssetResponse(
                                    id=stock['symbol'],
                                    symbol=stock['symbol'],
                                    name=stock['name'],
                                    category='stock',
                                    exchange=stock.get('exchange', 'NASDAQ'),
                                    current_price=stock['price'],
                                    price_change_24h=stock['change'],
                                    price_change_percentage_24h=stock['changesPercentage'],
                                    market_cap=stock.get('marketCap'),
                                    volume_24h=stock.get('volume'),
                                    last_price_update=datetime.now(),
                                    additional_data={
                                        'pe_ratio': stock.get('pe'),
                                        'eps': stock.get('eps'),
                                        'day_low': stock.get('dayLow'),
                                        'day_high': stock.get('dayHigh'),
                                        'year_low': stock.get('yearLow'),
                                        'year_high': stock.get('yearHigh')
                                    }
                                )
                                assets.append(asset)
                            
                            return assets
            except Exception as e:
                logger.error(f"FMP API error: {e}")
        
        # Fallback to Yahoo Finance
        if len(assets) == 0:
            try:
                for symbol in popular_stocks[:limit]:
                    if await self._check_rate_limit('yahoo_finance'):
                        ticker = yf.Ticker(symbol)
                        info = ticker.info
                        hist = ticker.history(period='2d')
                        
                        if not hist.empty and len(hist) >= 2:
                            current_price = hist['Close'].iloc[-1]
                            prev_price = hist['Close'].iloc[-2]
                            price_change = current_price - prev_price
                            price_change_pct = (price_change / prev_price) * 100
                            
                            asset = AssetResponse(
                                id=symbol,
                                symbol=symbol,
                                name=info.get('longName', symbol),
                                category='stock',
                                exchange=info.get('exchange', 'NASDAQ'),
                                current_price=float(current_price),
                                price_change_24h=float(price_change),
                                price_change_percentage_24h=float(price_change_pct),
                                market_cap=info.get('marketCap'),
                                volume_24h=float(hist['Volume'].iloc[-1]) if 'Volume' in hist else None,
                                description=info.get('longBusinessSummary'),
                                website=info.get('website'),
                                last_price_update=datetime.now()
                            )
                            assets.append(asset)
            except Exception as e:
                logger.error(f"Yahoo Finance error: {e}")
        
        return assets
    
    async def _get_enhanced_forex_assets(self, limit: int) -> List[AssetResponse]:
        """Get forex pairs from multiple sources."""
        assets = []
        major_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X',
            'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X'
        ]
        
        # Try Alpha Vantage first
        if self.alpha_vantage_key and await self._check_rate_limit('alpha_vantage'):
            try:
                for pair in major_pairs[:limit]:
                    # Convert Yahoo format to Alpha Vantage format
                    from_currency = pair[:3]
                    to_currency = pair[3:6]
                    
                    url = self.api_endpoints['alpha_vantage']
                    params = {
                        'function': 'CURRENCY_EXCHANGE_RATE',
                        'from_currency': from_currency,
                        'to_currency': to_currency,
                        'apikey': self.alpha_vantage_key
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                rate_data = data.get('Realtime Currency Exchange Rate', {})
                                
                                if rate_data:
                                    current_rate = float(rate_data.get('5. Exchange Rate', 0))
                                    
                                    asset = AssetResponse(
                                        id=pair,
                                        symbol=pair,
                                        name=f"{from_currency}/{to_currency}",
                                        category='forex',
                                        current_price=current_rate,
                                        last_price_update=datetime.now(),
                                        additional_data={
                                            'bid_price': rate_data.get('8. Bid Price'),
                                            'ask_price': rate_data.get('9. Ask Price'),
                                            'last_refreshed': rate_data.get('6. Last Refreshed')
                                        }
                                    )
                                    assets.append(asset)
                    
                    # Rate limiting delay
                    await asyncio.sleep(0.2)
                    
            except Exception as e:
                logger.error(f"Alpha Vantage forex error: {e}")
        
        # Fallback to Yahoo Finance
        if len(assets) == 0:
            try:
                for pair in major_pairs[:limit]:
                    if await self._check_rate_limit('yahoo_finance'):
                        ticker = yf.Ticker(pair)
                        hist = ticker.history(period='2d')
                        
                        if not hist.empty and len(hist) >= 2:
                            current_price = hist['Close'].iloc[-1]
                            prev_price = hist['Close'].iloc[-2]
                            price_change = current_price - prev_price
                            price_change_pct = (price_change / prev_price) * 100
                            
                            asset = AssetResponse(
                                id=pair,
                                symbol=pair,
                                name=pair.replace('=X', ''),
                                category='forex',
                                current_price=float(current_price),
                                price_change_24h=float(price_change),
                                price_change_percentage_24h=float(price_change_pct),
                                last_price_update=datetime.now()
                            )
                            assets.append(asset)
            except Exception as e:
                logger.error(f"Yahoo Finance forex error: {e}")
        
        return assets
    
    async def get_real_time_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get real-time prices for multiple symbols with caching."""
        cache_key = f"prices:{'_'.join(sorted(symbols))}"
        
        # Try cache first
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        prices = {}
        
        # Group symbols by category for efficient API calls
        crypto_symbols = [s for s in symbols if self._is_crypto_symbol(s)]
        stock_symbols = [s for s in symbols if self._is_stock_symbol(s)]
        forex_symbols = [s for s in symbols if self._is_forex_symbol(s)]
        
        # Fetch crypto prices
        if crypto_symbols:
            crypto_prices = await self._get_real_time_crypto_prices(crypto_symbols)
            prices.update(crypto_prices)
        
        # Fetch stock prices
        if stock_symbols:
            stock_prices = await self._get_real_time_stock_prices(stock_symbols)
            prices.update(stock_prices)
        
        # Fetch forex prices
        if forex_symbols:
            forex_prices = await self._get_real_time_forex_prices(forex_symbols)
            prices.update(forex_prices)
        
        # Cache the results
        await self._set_cache(cache_key, prices, self.cache_ttl['prices'])
        
        return prices
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Check if symbol is a cryptocurrency."""
        crypto_symbols = ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'XRP', 'LTC', 'BCH', 'BNB', 'SOL']
        return symbol.upper() in crypto_symbols
    
    def _is_stock_symbol(self, symbol: str) -> bool:
        """Check if symbol is a stock."""
        return not (symbol.endswith('=X') or self._is_crypto_symbol(symbol))
    
    def _is_forex_symbol(self, symbol: str) -> bool:
        """Check if symbol is a forex pair."""
        return symbol.endswith('=X')
    
    async def _get_real_time_crypto_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get real-time crypto prices."""
        prices = {}
        
        if await self._check_rate_limit('coingecko'):
            try:
                # Convert symbols to CoinGecko IDs
                symbol_map = {
                    'BTC': 'bitcoin', 'ETH': 'ethereum', 'ADA': 'cardano',
                    'DOT': 'polkadot', 'LINK': 'chainlink', 'XRP': 'ripple',
                    'LTC': 'litecoin', 'BCH': 'bitcoin-cash', 'BNB': 'binancecoin',
                    'SOL': 'solana'
                }
                
                ids = [symbol_map.get(symbol.upper()) for symbol in symbols if symbol.upper() in symbol_map]
                ids = [id for id in ids if id]  # Remove None values
                
                if ids:
                    url = f"{self.api_endpoints['coingecko']}/simple/price"
                    params = {
                        'ids': ','.join(ids),
                        'vs_currencies': 'usd',
                        'include_24hr_change': 'true',
                        'include_24hr_vol': 'true'
                    }
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, params=params) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                for coin_id, price_data in data.items():
                                    # Find the symbol for this coin_id
                                    symbol = next((k for k, v in symbol_map.items() if v == coin_id), coin_id)
                                    
                                    prices[symbol] = {
                                        'price': price_data['usd'],
                                        'change_24h': price_data.get('usd_24h_change', 0),
                                        'volume_24h': price_data.get('usd_24h_vol', 0),
                                        'timestamp': datetime.now().isoformat()
                                    }
            except Exception as e:
                logger.error(f"Real-time crypto prices error: {e}")
        
        return prices
    
    async def _get_real_time_stock_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get real-time stock prices."""
        prices = {}
        
        # Try FMP first
        if self.fmp_key and await self._check_rate_limit('fmp'):
            try:
                symbols_str = ','.join(symbols)
                url = f"{self.api_endpoints['fmp']}/quote/{symbols_str}"
                params = {'apikey': self.fmp_key}
                
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            for stock in data:
                                prices[stock['symbol']] = {
                                    'price': stock['price'],
                                    'change_24h': stock['change'],
                                    'change_percentage_24h': stock['changesPercentage'],
                                    'volume_24h': stock.get('volume', 0),
                                    'timestamp': datetime.now().isoformat()
                                }
            except Exception as e:
                logger.error(f"Real-time stock prices error: {e}")
        
        return prices
    
    async def _get_real_time_forex_prices(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get real-time forex prices."""
        prices = {}
        
        # Use Yahoo Finance for forex
        try:
            for symbol in symbols:
                if await self._check_rate_limit('yahoo_finance'):
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1d', interval='1m')
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        
                        prices[symbol] = {
                            'price': float(current_price),
                            'timestamp': datetime.now().isoformat()
                        }
        except Exception as e:
            logger.error(f"Real-time forex prices error: {e}")
        
        return prices
    
    def _get_mock_assets(self, category: Optional[str], limit: int) -> List[AssetResponse]:
        """Return mock assets when APIs are unavailable."""
        mock_assets = [
            AssetResponse(
                id="bitcoin",
                symbol="BTC",
                name="Bitcoin",
                category="crypto",
                current_price=45000.0,
                price_change_24h=1200.0,
                price_change_percentage_24h=2.74,
                market_cap=850000000000,
                volume_24h=25000000000,
                last_price_update=datetime.now()
            ),
            AssetResponse(
                id="AAPL",
                symbol="AAPL",
                name="Apple Inc.",
                category="stock",
                exchange="NASDAQ",
                current_price=175.50,
                price_change_24h=2.30,
                price_change_percentage_24h=1.33,
                market_cap=2800000000000,
                volume_24h=85000000,
                last_price_update=datetime.now()
            ),
            AssetResponse(
                id="EURUSD=X",
                symbol="EURUSD=X",
                name="EUR/USD",
                category="forex",
                current_price=1.0850,
                price_change_24h=0.0025,
                price_change_percentage_24h=0.23,
                last_price_update=datetime.now()
            )
        ]
        
        if category:
            mock_assets = [asset for asset in mock_assets if asset.category == category]
        
        return mock_assets[:limit]
    
    async def cleanup(self):
        """Cleanup resources."""
        if self.redis_client:
            await self.redis_client.close()
        
        if self.binance_exchange:
            await self.binance_exchange.close()

# Create global instance
enhanced_market_data_service = EnhancedMarketDataService()