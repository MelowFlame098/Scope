import asyncio
import aiohttp
import yfinance as yf
import ccxt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json
import os
from dotenv import load_dotenv

from schemas import AssetResponse, MarketDataPoint, ChartData, MarketOverview, MarketDataResponse

load_dotenv()

class MarketDataService:
    def __init__(self):
        self.crypto_exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_API_KEY'),
            'secret': os.getenv('BINANCE_SECRET'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # API keys for various data sources
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.polygon_key = os.getenv('POLYGON_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        
        # Cache for market data
        self.cache = {}
        self.cache_ttl = 60  # 1 minute cache
        
    async def get_assets(self, category: Optional[str] = None, limit: int = 50) -> List[AssetResponse]:
        """Get list of assets by category."""
        assets = []
        
        try:
            if category == 'crypto' or category is None:
                crypto_assets = await self._get_crypto_assets(limit)
                assets.extend(crypto_assets)
            
            if category == 'stocks' or category is None:
                stock_assets = await self._get_stock_assets(limit)
                assets.extend(stock_assets)
            
            if category == 'forex' or category is None:
                forex_assets = await self._get_forex_assets(limit)
                assets.extend(forex_assets)
                
        except Exception as e:
            print(f"Error fetching assets: {e}")
            # Return mock data if API fails
            return self._get_mock_assets(category, limit)
        
        return assets[:limit] if limit else assets
    
    async def _get_crypto_assets(self, limit: int) -> List[AssetResponse]:
        """Get cryptocurrency assets."""
        try:
            # Get top cryptocurrencies by market cap
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': 'market_cap_desc',
                'per_page': limit,
                'page': 1,
                'sparkline': False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        assets = []
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
                                last_price_update=datetime.now()
                            )
                            assets.append(asset)
                        
                        return assets
        except Exception as e:
            print(f"Error fetching crypto assets: {e}")
        
        return []
    
    async def _get_stock_assets(self, limit: int) -> List[AssetResponse]:
        """Get stock assets."""
        # Popular stocks to fetch
        popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'DIS', 'V', 'MA', 'JPM',
            'BAC', 'WMT', 'PG', 'JNJ', 'UNH', 'HD', 'VZ', 'T', 'PFE', 'KO', 'PEP'
        ]
        
        assets = []
        try:
            for symbol in popular_stocks[:limit]:
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
            print(f"Error fetching stock assets: {e}")
        
        return assets
    
    async def _get_forex_assets(self, limit: int) -> List[AssetResponse]:
        """Get forex pairs."""
        major_pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X',
            'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X'
        ]
        
        assets = []
        try:
            for pair in major_pairs[:limit]:
                ticker = yf.Ticker(pair)
                hist = ticker.history(period='2d')
                
                if not hist.empty and len(hist) >= 2:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2]
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100
                    
                    clean_symbol = pair.replace('=X', '')
                    asset = AssetResponse(
                        id=clean_symbol,
                        symbol=clean_symbol,
                        name=f"{clean_symbol[:3]}/{clean_symbol[3:]} Exchange Rate",
                        category='forex',
                        current_price=float(current_price),
                        price_change_24h=float(price_change),
                        price_change_percentage_24h=float(price_change_pct),
                        volume_24h=float(hist['Volume'].iloc[-1]) if 'Volume' in hist else None,
                        last_price_update=datetime.now()
                    )
                    assets.append(asset)
        except Exception as e:
            print(f"Error fetching forex assets: {e}")
        
        return assets
    
    async def get_asset_details(self, symbol: str) -> Optional[AssetResponse]:
        """Get detailed information for a specific asset."""
        try:
            # Try to determine asset type and fetch accordingly
            if symbol in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK', 'UNI', 'AAVE']:
                return await self._get_crypto_details(symbol)
            elif 'USD' in symbol and len(symbol) == 6:
                return await self._get_forex_details(symbol)
            else:
                return await self._get_stock_details(symbol)
        except Exception as e:
            print(f"Error fetching asset details for {symbol}: {e}")
            return None
    
    async def _get_crypto_details(self, symbol: str) -> Optional[AssetResponse]:
        """Get cryptocurrency details."""
        try:
            url = f"https://api.coingecko.com/api/v3/coins/{symbol.lower()}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        market_data = data.get('market_data', {})
                        
                        return AssetResponse(
                            id=data['id'],
                            symbol=data['symbol'].upper(),
                            name=data['name'],
                            category='crypto',
                            current_price=market_data.get('current_price', {}).get('usd'),
                            price_change_24h=market_data.get('price_change_24h'),
                            price_change_percentage_24h=market_data.get('price_change_percentage_24h'),
                            market_cap=market_data.get('market_cap', {}).get('usd'),
                            volume_24h=market_data.get('total_volume', {}).get('usd'),
                            description=data.get('description', {}).get('en'),
                            website=data.get('links', {}).get('homepage', [None])[0],
                            logo_url=data.get('image', {}).get('large'),
                            last_price_update=datetime.now()
                        )
        except Exception as e:
            print(f"Error fetching crypto details: {e}")
        
        return None
    
    async def _get_stock_details(self, symbol: str) -> Optional[AssetResponse]:
        """Get stock details."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period='2d')
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
                
                return AssetResponse(
                    id=symbol,
                    symbol=symbol,
                    name=info.get('longName', symbol),
                    category='stock',
                    exchange=info.get('exchange'),
                    current_price=float(current_price),
                    price_change_24h=float(price_change),
                    price_change_percentage_24h=float(price_change_pct),
                    market_cap=info.get('marketCap'),
                    volume_24h=float(hist['Volume'].iloc[-1]) if 'Volume' in hist else None,
                    description=info.get('longBusinessSummary'),
                    website=info.get('website'),
                    last_price_update=datetime.now()
                )
        except Exception as e:
            print(f"Error fetching stock details: {e}")
        
        return None
    
    async def _get_forex_details(self, symbol: str) -> Optional[AssetResponse]:
        """Get forex pair details."""
        try:
            yahoo_symbol = f"{symbol}=X"
            ticker = yf.Ticker(yahoo_symbol)
            hist = ticker.history(period='2d')
            
            if not hist.empty:
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[-2] if len(hist) >= 2 else current_price
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
                
                return AssetResponse(
                    id=symbol,
                    symbol=symbol,
                    name=f"{symbol[:3]}/{symbol[3:]} Exchange Rate",
                    category='forex',
                    current_price=float(current_price),
                    price_change_24h=float(price_change),
                    price_change_percentage_24h=float(price_change_pct),
                    volume_24h=float(hist['Volume'].iloc[-1]) if 'Volume' in hist else None,
                    last_price_update=datetime.now()
                )
        except Exception as e:
            print(f"Error fetching forex details: {e}")
        
        return None
    
    async def get_chart_data(self, symbol: str, timeframe: str = "1d", period: str = "1y") -> ChartData:
        """Get chart data for an asset."""
        try:
            # Map timeframes
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk', '1M': '1mo'
            }
            
            yf_interval = interval_map.get(timeframe, '1d')
            
            # Determine if it's crypto, stock, or forex
            if symbol in ['BTC', 'ETH', 'ADA', 'DOT', 'LINK']:
                # For crypto, use different symbol format
                yf_symbol = f"{symbol}-USD"
            elif 'USD' in symbol and len(symbol) == 6:
                # Forex pair
                yf_symbol = f"{symbol}=X"
            else:
                # Stock
                yf_symbol = symbol
            
            ticker = yf.Ticker(yf_symbol)
            hist = ticker.history(period=period, interval=yf_interval)
            
            if hist.empty:
                return ChartData(symbol=symbol, timeframe=timeframe, data=[])
            
            # Convert to MarketDataPoint objects
            data_points = []
            for index, row in hist.iterrows():
                point = MarketDataPoint(
                    timestamp=index.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']) if 'Volume' in row else 0
                )
                data_points.append(point)
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(hist)
            
            return ChartData(
                symbol=symbol,
                timeframe=timeframe,
                data=data_points,
                indicators=indicators
            )
            
        except Exception as e:
            print(f"Error fetching chart data for {symbol}: {e}")
            return ChartData(symbol=symbol, timeframe=timeframe, data=[])
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """Calculate technical indicators."""
        indicators = {}
        
        try:
            # Simple Moving Averages
            indicators['sma_20'] = df['Close'].rolling(window=20).mean().fillna(0).tolist()
            indicators['sma_50'] = df['Close'].rolling(window=50).mean().fillna(0).tolist()
            
            # Exponential Moving Averages
            indicators['ema_12'] = df['Close'].ewm(span=12).mean().fillna(0).tolist()
            indicators['ema_26'] = df['Close'].ewm(span=26).mean().fillna(0).tolist()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            indicators['rsi'] = rsi.fillna(50).tolist()
            
            # MACD
            ema_12 = df['Close'].ewm(span=12).mean()
            ema_26 = df['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            indicators['macd'] = macd.fillna(0).tolist()
            indicators['macd_signal'] = signal.fillna(0).tolist()
            
            # Bollinger Bands
            sma_20 = df['Close'].rolling(window=20).mean()
            std_20 = df['Close'].rolling(window=20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).fillna(0).tolist()
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).fillna(0).tolist()
            indicators['bb_middle'] = sma_20.fillna(0).tolist()
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        
        return indicators
    
    async def get_market_overview(self) -> MarketDataResponse:
        """Get market overview data."""
        try:
            # Get overall market data
            overview = await self._get_market_overview_data()
            
            # Get top gainers, losers, and trending
            assets = await self.get_assets(limit=100)
            
            # Sort by price change percentage
            sorted_by_change = sorted(assets, key=lambda x: x.price_change_percentage_24h or 0, reverse=True)
            
            top_gainers = sorted_by_change[:10]
            top_losers = sorted_by_change[-10:]
            
            # Trending based on volume and price change
            trending = sorted(assets, key=lambda x: (x.volume_24h or 0) * abs(x.price_change_percentage_24h or 0), reverse=True)[:10]
            
            return MarketDataResponse(
                overview=overview,
                top_gainers=top_gainers,
                top_losers=top_losers,
                trending=trending,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error fetching market overview: {e}")
            return self._get_mock_market_overview()
    
    async def _get_market_overview_data(self) -> MarketOverview:
        """Get overall market statistics."""
        try:
            url = "https://api.coingecko.com/api/v3/global"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        global_data = data.get('data', {})
                        
                        return MarketOverview(
                            total_market_cap=global_data.get('total_market_cap', {}).get('usd', 0),
                            total_volume_24h=global_data.get('total_volume', {}).get('usd', 0),
                            btc_dominance=global_data.get('market_cap_percentage', {}).get('btc', 0),
                            active_cryptocurrencies=global_data.get('active_cryptocurrencies', 0),
                            market_cap_change_24h=global_data.get('market_cap_change_percentage_24h_usd', 0)
                        )
        except Exception as e:
            print(f"Error fetching market overview data: {e}")
        
        # Return default values if API fails
        return MarketOverview(
            total_market_cap=2500000000000,  # $2.5T
            total_volume_24h=100000000000,   # $100B
            btc_dominance=45.0,
            active_cryptocurrencies=10000,
            market_cap_change_24h=2.5
        )
    
    async def get_real_time_data(self) -> Dict[str, Any]:
        """Get real-time market data for WebSocket updates."""
        try:
            # Get a few key assets for real-time updates
            key_symbols = ['BTC', 'ETH', 'AAPL', 'TSLA', 'EURUSD']
            real_time_data = {}
            
            for symbol in key_symbols:
                asset = await self.get_asset_details(symbol)
                if asset:
                    real_time_data[symbol] = {
                        'price': asset.current_price,
                        'change': asset.price_change_24h,
                        'change_percent': asset.price_change_percentage_24h,
                        'volume': asset.volume_24h,
                        'timestamp': asset.last_price_update.isoformat() if asset.last_price_update else None
                    }
            
            return real_time_data
            
        except Exception as e:
            print(f"Error fetching real-time data: {e}")
            return {}
    
    def _get_mock_assets(self, category: Optional[str], limit: int) -> List[AssetResponse]:
        """Return mock asset data when APIs fail."""
        mock_assets = [
            AssetResponse(
                id="bitcoin",
                symbol="BTC",
                name="Bitcoin",
                category="crypto",
                current_price=45000.0,
                price_change_24h=1200.0,
                price_change_percentage_24h=2.75,
                market_cap=880000000000,
                volume_24h=25000000000,
                last_price_update=datetime.now()
            ),
            AssetResponse(
                id="ethereum",
                symbol="ETH",
                name="Ethereum",
                category="crypto",
                current_price=2800.0,
                price_change_24h=85.0,
                price_change_percentage_24h=3.13,
                market_cap=340000000000,
                volume_24h=15000000000,
                last_price_update=datetime.now()
            ),
            AssetResponse(
                id="AAPL",
                symbol="AAPL",
                name="Apple Inc.",
                category="stock",
                current_price=175.50,
                price_change_24h=2.25,
                price_change_percentage_24h=1.30,
                market_cap=2800000000000,
                volume_24h=85000000,
                last_price_update=datetime.now()
            )
        ]
        
        if category:
            mock_assets = [asset for asset in mock_assets if asset.category == category]
        
        return mock_assets[:limit]
    
    def _get_mock_market_overview(self) -> MarketDataResponse:
        """Return mock market overview data."""
        overview = MarketOverview(
            total_market_cap=2500000000000,
            total_volume_24h=100000000000,
            btc_dominance=45.0,
            active_cryptocurrencies=10000,
            market_cap_change_24h=2.5
        )
        
        mock_assets = self._get_mock_assets(None, 10)
        
        return MarketDataResponse(
            overview=overview,
            top_gainers=mock_assets[:3],
            top_losers=mock_assets[3:6],
            trending=mock_assets[6:9],
            timestamp=datetime.now()
        )