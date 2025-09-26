import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass
import re
import time
from urllib.parse import quote_plus
import yfinance as yf
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Data class for market data"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    timestamp: datetime = None

@dataclass
class OHLCV:
    """Data class for OHLCV candlestick data"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None

@dataclass
class OptionData:
    """Data class for options data"""
    symbol: str
    strike: float
    expiration: datetime
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last_price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

class MarketDataScraperService:
    """Advanced market data scraper for real-time financial data"""
    
    def __init__(self):
        self.session = None
        self.rate_limits = {
            'yahoo': {'requests_per_minute': 120, 'last_request': 0, 'request_count': 0},
            'alpha_vantage': {'requests_per_minute': 5, 'last_request': 0, 'request_count': 0},
            'finnhub': {'requests_per_minute': 60, 'last_request': 0, 'request_count': 0},
            'polygon': {'requests_per_minute': 100, 'last_request': 0, 'request_count': 0}
        }
        
        # API endpoints
        self.endpoints = {
            'yahoo_quote': 'https://query1.finance.yahoo.com/v8/finance/chart/{symbol}',
            'yahoo_options': 'https://query1.finance.yahoo.com/v7/finance/options/{symbol}',
            'yahoo_fundamentals': 'https://query1.finance.yahoo.com/v10/finance/quoteSummary/{symbol}',
            'finnhub_quote': 'https://finnhub.io/api/v1/quote',
            'polygon_quote': 'https://api.polygon.io/v2/last/nbbo/{symbol}',
            'alpha_vantage_quote': 'https://www.alphavantage.co/query'
        }
        
        # Headers for requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
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
        rate_limit = self.rate_limits.get(source, {'requests_per_minute': 60, 'last_request': 0, 'request_count': 0})
        
        # Reset counter if a minute has passed
        if current_time - rate_limit['last_request'] >= 60:
            rate_limit['request_count'] = 0
            rate_limit['last_request'] = current_time
        
        # Check if we can make another request
        if rate_limit['request_count'] >= rate_limit['requests_per_minute']:
            return False
        
        rate_limit['request_count'] += 1
        return True

    async def get_real_time_quote(self, symbol: str) -> Optional[MarketData]:
        """Get real-time quote for a symbol"""
        try:
            if not self._check_rate_limit('yahoo'):
                logger.warning("Yahoo Finance rate limit exceeded")
                return None

            url = self.endpoints['yahoo_quote'].format(symbol=symbol)
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'chart' in data and data['chart']['result']:
                        result = data['chart']['result'][0]
                        meta = result['meta']
                        
                        current_price = meta.get('regularMarketPrice', 0)
                        previous_close = meta.get('previousClose', current_price)
                        change = current_price - previous_close
                        change_percent = (change / previous_close * 100) if previous_close != 0 else 0
                        
                        market_data = MarketData(
                            symbol=symbol,
                            price=current_price,
                            change=change,
                            change_percent=change_percent,
                            volume=meta.get('regularMarketVolume', 0),
                            market_cap=meta.get('marketCap'),
                            fifty_two_week_high=meta.get('fiftyTwoWeekHigh'),
                            fifty_two_week_low=meta.get('fiftyTwoWeekLow'),
                            timestamp=datetime.now()
                        )
                        
                        return market_data
                else:
                    logger.error(f"Failed to fetch quote for {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error fetching real-time quote for {symbol}: {e}")
            return None

    async def get_historical_data(self, symbol: str, period: str = '1y', interval: str = '1d') -> List[OHLCV]:
        """Get historical OHLCV data for a symbol"""
        try:
            # Use yfinance for historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            ohlcv_data = []
            for timestamp, row in hist.iterrows():
                ohlcv = OHLCV(
                    symbol=symbol,
                    timestamp=timestamp.to_pydatetime(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    adj_close=float(row['Adj Close']) if 'Adj Close' in row else None
                )
                ohlcv_data.append(ohlcv)
            
            logger.info(f"Retrieved {len(ohlcv_data)} historical data points for {symbol}")
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []

    async def get_options_chain(self, symbol: str, expiration_date: str = None) -> List[OptionData]:
        """Get options chain for a symbol"""
        try:
            if not self._check_rate_limit('yahoo'):
                logger.warning("Yahoo Finance rate limit exceeded")
                return []

            url = self.endpoints['yahoo_options'].format(symbol=symbol)
            if expiration_date:
                # Convert date to timestamp if needed
                url += f"?date={expiration_date}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    options_data = []
                    if 'optionChain' in data and data['optionChain']['result']:
                        result = data['optionChain']['result'][0]
                        
                        for option_type in ['calls', 'puts']:
                            if option_type in result['options'][0]:
                                for option in result['options'][0][option_type]:
                                    try:
                                        option_data = OptionData(
                                            symbol=symbol,
                                            strike=float(option.get('strike', 0)),
                                            expiration=datetime.fromtimestamp(option.get('expiration', 0)),
                                            option_type='call' if option_type == 'calls' else 'put',
                                            bid=float(option.get('bid', 0)),
                                            ask=float(option.get('ask', 0)),
                                            last_price=float(option.get('lastPrice', 0)),
                                            volume=int(option.get('volume', 0)),
                                            open_interest=int(option.get('openInterest', 0)),
                                            implied_volatility=float(option.get('impliedVolatility', 0))
                                        )
                                        options_data.append(option_data)
                                    except Exception as e:
                                        logger.error(f"Error parsing option data: {e}")
                                        continue
                    
                    logger.info(f"Retrieved {len(options_data)} options for {symbol}")
                    return options_data
                else:
                    logger.error(f"Failed to fetch options for {symbol}: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error fetching options chain for {symbol}: {e}")
            return []

    async def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for a symbol"""
        try:
            if not self._check_rate_limit('yahoo'):
                logger.warning("Yahoo Finance rate limit exceeded")
                return {}

            url = self.endpoints['yahoo_fundamentals'].format(symbol=symbol)
            params = {
                'modules': 'defaultKeyStatistics,financialData,summaryDetail,assetProfile'
            }
            
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'quoteSummary' in data and data['quoteSummary']['result']:
                        result = data['quoteSummary']['result'][0]
                        
                        fundamentals = {}
                        
                        # Extract key statistics
                        if 'defaultKeyStatistics' in result:
                            stats = result['defaultKeyStatistics']
                            fundamentals.update({
                                'pe_ratio': stats.get('trailingPE', {}).get('raw'),
                                'forward_pe': stats.get('forwardPE', {}).get('raw'),
                                'peg_ratio': stats.get('pegRatio', {}).get('raw'),
                                'price_to_book': stats.get('priceToBook', {}).get('raw'),
                                'enterprise_value': stats.get('enterpriseValue', {}).get('raw'),
                                'profit_margins': stats.get('profitMargins', {}).get('raw'),
                                'beta': stats.get('beta', {}).get('raw')
                            })
                        
                        # Extract financial data
                        if 'financialData' in result:
                            financial = result['financialData']
                            fundamentals.update({
                                'total_cash': financial.get('totalCash', {}).get('raw'),
                                'total_debt': financial.get('totalDebt', {}).get('raw'),
                                'revenue_growth': financial.get('revenueGrowth', {}).get('raw'),
                                'earnings_growth': financial.get('earningsGrowth', {}).get('raw'),
                                'gross_margins': financial.get('grossMargins', {}).get('raw'),
                                'operating_margins': financial.get('operatingMargins', {}).get('raw'),
                                'return_on_equity': financial.get('returnOnEquity', {}).get('raw')
                            })
                        
                        # Extract summary detail
                        if 'summaryDetail' in result:
                            summary = result['summaryDetail']
                            fundamentals.update({
                                'dividend_yield': summary.get('dividendYield', {}).get('raw'),
                                'dividend_rate': summary.get('dividendRate', {}).get('raw'),
                                'payout_ratio': summary.get('payoutRatio', {}).get('raw'),
                                'market_cap': summary.get('marketCap', {}).get('raw'),
                                'fifty_two_week_high': summary.get('fiftyTwoWeekHigh', {}).get('raw'),
                                'fifty_two_week_low': summary.get('fiftyTwoWeekLow', {}).get('raw')
                            })
                        
                        return fundamentals
                else:
                    logger.error(f"Failed to fetch fundamentals for {symbol}: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {}

    async def get_market_movers(self, market: str = 'US') -> Dict[str, List[MarketData]]:
        """Get market movers (gainers, losers, most active)"""
        try:
            movers = {
                'gainers': [],
                'losers': [],
                'most_active': []
            }
            
            # Yahoo Finance screener URLs
            screener_urls = {
                'gainers': 'https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds=day_gainers',
                'losers': 'https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds=day_losers',
                'most_active': 'https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?formatted=true&lang=en-US&region=US&scrIds=most_actives'
            }
            
            for category, url in screener_urls.items():
                if not self._check_rate_limit('yahoo'):
                    break
                
                try:
                    async with self.session.get(url) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            if 'finance' in data and 'result' in data['finance']:
                                results = data['finance']['result'][0]['quotes']
                                
                                for quote in results[:20]:  # Top 20
                                    try:
                                        market_data = MarketData(
                                            symbol=quote.get('symbol', ''),
                                            price=quote.get('regularMarketPrice', 0),
                                            change=quote.get('regularMarketChange', 0),
                                            change_percent=quote.get('regularMarketChangePercent', 0),
                                            volume=quote.get('regularMarketVolume', 0),
                                            market_cap=quote.get('marketCap'),
                                            timestamp=datetime.now()
                                        )
                                        movers[category].append(market_data)
                                    except Exception as e:
                                        logger.error(f"Error parsing mover data: {e}")
                                        continue
                        else:
                            logger.error(f"Failed to fetch {category}: {response.status}")
                            
                except Exception as e:
                    logger.error(f"Error fetching {category}: {e}")
                    continue
            
            return movers
            
        except Exception as e:
            logger.error(f"Error fetching market movers: {e}")
            return {'gainers': [], 'losers': [], 'most_active': []}

    async def get_crypto_data(self, symbols: List[str] = None) -> List[MarketData]:
        """Get cryptocurrency data"""
        try:
            if not symbols:
                symbols = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD']
            
            crypto_data = []
            
            for symbol in symbols:
                if not self._check_rate_limit('yahoo'):
                    break
                
                market_data = await self.get_real_time_quote(symbol)
                if market_data:
                    crypto_data.append(market_data)
            
            return crypto_data
            
        except Exception as e:
            logger.error(f"Error fetching crypto data: {e}")
            return []

    async def get_forex_data(self, pairs: List[str] = None) -> List[MarketData]:
        """Get forex data"""
        try:
            if not pairs:
                pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X']
            
            forex_data = []
            
            for pair in pairs:
                if not self._check_rate_limit('yahoo'):
                    break
                
                market_data = await self.get_real_time_quote(pair)
                if market_data:
                    forex_data.append(market_data)
            
            return forex_data
            
        except Exception as e:
            logger.error(f"Error fetching forex data: {e}")
            return []

    async def search_symbols(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for symbols matching query"""
        try:
            if not self._check_rate_limit('yahoo'):
                logger.warning("Yahoo Finance rate limit exceeded")
                return []

            url = f"https://query1.finance.yahoo.com/v1/finance/search?q={quote_plus(query)}&lang=en-US&region=US&quotesCount={limit}&newsCount=0"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    results = []
                    if 'quotes' in data:
                        for quote in data['quotes']:
                            result = {
                                'symbol': quote.get('symbol', ''),
                                'name': quote.get('longname', quote.get('shortname', '')),
                                'type': quote.get('quoteType', ''),
                                'exchange': quote.get('exchange', ''),
                                'market': quote.get('market', '')
                            }
                            results.append(result)
                    
                    return results
                else:
                    logger.error(f"Failed to search symbols: {response.status}")
                    return []
                    
        except Exception as e:
            logger.error(f"Error searching symbols: {e}")
            return []

# Example usage
async def main():
    """Example usage of the MarketDataScraperService"""
    async with MarketDataScraperService() as scraper:
        # Get real-time quote
        quote = await scraper.get_real_time_quote('AAPL')
        print(f"AAPL Quote: {quote}")
        
        # Get historical data
        historical = await scraper.get_historical_data('AAPL', period='1mo', interval='1d')
        print(f"Historical data points: {len(historical)}")
        
        # Get market movers
        movers = await scraper.get_market_movers()
        print(f"Top gainers: {len(movers['gainers'])}")
        
        # Search symbols
        search_results = await scraper.search_symbols('Apple')
        print(f"Search results: {search_results}")

if __name__ == "__main__":
    asyncio.run(main())