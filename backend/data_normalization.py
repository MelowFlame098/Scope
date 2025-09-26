import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from decimal import Decimal, ROUND_HALF_UP
import re
from schemas import AssetResponse, MarketDataPoint, ChartData

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Supported data sources."""
    YAHOO_FINANCE = "yahoo_finance"
    ALPHA_VANTAGE = "alpha_vantage"
    FINNHUB = "finnhub"
    COINGECKO = "coingecko"
    BINANCE = "binance"
    FMP = "financial_modeling_prep"
    CCXT = "ccxt"
    POLYGON = "polygon"
    IEX_CLOUD = "iex_cloud"

class AssetCategory(Enum):
    """Asset categories."""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    ETF = "etf"
    BOND = "bond"
    OPTION = "option"
    FUTURE = "future"

@dataclass
class NormalizedAsset:
    """Normalized asset data structure."""
    # Core identifiers
    symbol: str
    name: str
    category: AssetCategory
    source: DataSource
    
    # Price data
    current_price: Optional[float] = None
    price_change_24h: Optional[float] = None
    price_change_percentage_24h: Optional[float] = None
    
    # Volume and market data
    volume_24h: Optional[float] = None
    market_cap: Optional[float] = None
    
    # Additional identifiers
    exchange: Optional[str] = None
    currency: str = "USD"
    
    # Metadata
    description: Optional[str] = None
    website: Optional[str] = None
    logo_url: Optional[str] = None
    
    # Timestamps
    last_updated: Optional[datetime] = None
    data_timestamp: Optional[datetime] = None
    
    # Extended data
    extended_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_asset_response(self) -> AssetResponse:
        """Convert to AssetResponse schema."""
        return AssetResponse(
            id=self.symbol.lower(),
            symbol=self.symbol,
            name=self.name,
            category=self.category.value,
            exchange=self.exchange,
            current_price=self.current_price,
            price_change_24h=self.price_change_24h,
            price_change_percentage_24h=self.price_change_percentage_24h,
            market_cap=self.market_cap,
            volume_24h=self.volume_24h,
            description=self.description,
            website=self.website,
            logo_url=self.logo_url,
            last_price_update=self.last_updated or datetime.now(),
            additional_data=self.extended_data
        )

@dataclass
class NormalizedPriceData:
    """Normalized price data structure."""
    symbol: str
    timestamp: datetime
    open_price: Optional[float] = None
    high_price: Optional[float] = None
    low_price: Optional[float] = None
    close_price: Optional[float] = None
    volume: Optional[float] = None
    source: Optional[DataSource] = None
    
    def to_market_data_point(self) -> MarketDataPoint:
        """Convert to MarketDataPoint schema."""
        return MarketDataPoint(
            timestamp=self.timestamp,
            open=self.open_price,
            high=self.high_price,
            low=self.low_price,
            close=self.close_price,
            volume=self.volume
        )

class DataNormalizationService:
    """Service for normalizing data from different sources into standardized formats."""
    
    def __init__(self):
        # Symbol mapping for different sources
        self.symbol_mappings = {
            DataSource.COINGECKO: {
                'bitcoin': 'BTC',
                'ethereum': 'ETH',
                'cardano': 'ADA',
                'polkadot': 'DOT',
                'chainlink': 'LINK',
                'ripple': 'XRP',
                'litecoin': 'LTC',
                'bitcoin-cash': 'BCH',
                'binancecoin': 'BNB',
                'solana': 'SOL'
            },
            DataSource.YAHOO_FINANCE: {
                'EURUSD=X': 'EUR/USD',
                'GBPUSD=X': 'GBP/USD',
                'USDJPY=X': 'USD/JPY',
                'USDCHF=X': 'USD/CHF',
                'AUDUSD=X': 'AUD/USD',
                'USDCAD=X': 'USD/CAD',
                'NZDUSD=X': 'NZD/USD'
            }
        }
        
        # Currency mappings
        self.currency_mappings = {
            'usd': 'USD',
            'eur': 'EUR',
            'gbp': 'GBP',
            'jpy': 'JPY',
            'btc': 'BTC',
            'eth': 'ETH'
        }
        
        # Exchange mappings
        self.exchange_mappings = {
            'nasdaq': 'NASDAQ',
            'nyse': 'NYSE',
            'binance': 'Binance',
            'coinbase': 'Coinbase Pro',
            'kraken': 'Kraken'
        }
    
    def normalize_asset_data(self, raw_data: Dict[str, Any], source: DataSource) -> Optional[NormalizedAsset]:
        """Normalize asset data from any source."""
        try:
            if source == DataSource.COINGECKO:
                return self._normalize_coingecko_asset(raw_data)
            elif source == DataSource.YAHOO_FINANCE:
                return self._normalize_yahoo_asset(raw_data)
            elif source == DataSource.ALPHA_VANTAGE:
                return self._normalize_alpha_vantage_asset(raw_data)
            elif source == DataSource.FINNHUB:
                return self._normalize_finnhub_asset(raw_data)
            elif source == DataSource.FMP:
                return self._normalize_fmp_asset(raw_data)
            elif source == DataSource.BINANCE:
                return self._normalize_binance_asset(raw_data)
            else:
                logger.warning(f"Unsupported data source: {source}")
                return None
        except Exception as e:
            logger.error(f"Error normalizing asset data from {source}: {e}")
            return None
    
    def _normalize_coingecko_asset(self, data: Dict[str, Any]) -> NormalizedAsset:
        """Normalize CoinGecko asset data."""
        symbol = data.get('symbol', '').upper()
        if data.get('id') in self.symbol_mappings[DataSource.COINGECKO]:
            symbol = self.symbol_mappings[DataSource.COINGECKO][data['id']]
        
        return NormalizedAsset(
            symbol=symbol,
            name=data.get('name', ''),
            category=AssetCategory.CRYPTO,
            source=DataSource.COINGECKO,
            current_price=self._safe_float(data.get('current_price')),
            price_change_24h=self._safe_float(data.get('price_change_24h')),
            price_change_percentage_24h=self._safe_float(data.get('price_change_percentage_24h')),
            volume_24h=self._safe_float(data.get('total_volume')),
            market_cap=self._safe_float(data.get('market_cap')),
            logo_url=data.get('image'),
            last_updated=datetime.now(),
            extended_data={
                'coingecko_id': data.get('id'),
                'market_cap_rank': data.get('market_cap_rank'),
                'circulating_supply': self._safe_float(data.get('circulating_supply')),
                'total_supply': self._safe_float(data.get('total_supply')),
                'max_supply': self._safe_float(data.get('max_supply')),
                'ath': self._safe_float(data.get('ath')),
                'atl': self._safe_float(data.get('atl')),
                'price_change_1h': self._safe_float(data.get('price_change_percentage_1h_in_currency')),
                'price_change_7d': self._safe_float(data.get('price_change_percentage_7d_in_currency')),
                'price_change_30d': self._safe_float(data.get('price_change_percentage_30d_in_currency'))
            }
        )
    
    def _normalize_yahoo_asset(self, data: Dict[str, Any]) -> NormalizedAsset:
        """Normalize Yahoo Finance asset data."""
        symbol = data.get('symbol', '')
        
        # Determine category
        category = AssetCategory.STOCK
        if symbol.endswith('=X'):
            category = AssetCategory.FOREX
        elif symbol.endswith('-USD') or any(crypto in symbol for crypto in ['BTC', 'ETH', 'ADA']):
            category = AssetCategory.CRYPTO
        
        # Get display name
        display_name = symbol
        if symbol in self.symbol_mappings[DataSource.YAHOO_FINANCE]:
            display_name = self.symbol_mappings[DataSource.YAHOO_FINANCE][symbol]
        
        return NormalizedAsset(
            symbol=symbol,
            name=data.get('longName', display_name),
            category=category,
            source=DataSource.YAHOO_FINANCE,
            current_price=self._safe_float(data.get('regularMarketPrice')),
            price_change_24h=self._safe_float(data.get('regularMarketChange')),
            price_change_percentage_24h=self._safe_float(data.get('regularMarketChangePercent')),
            volume_24h=self._safe_float(data.get('regularMarketVolume')),
            market_cap=self._safe_float(data.get('marketCap')),
            exchange=self._normalize_exchange(data.get('exchange')),
            currency=self._normalize_currency(data.get('currency')),
            description=data.get('longBusinessSummary'),
            website=data.get('website'),
            last_updated=datetime.now(),
            extended_data={
                'sector': data.get('sector'),
                'industry': data.get('industry'),
                'pe_ratio': self._safe_float(data.get('trailingPE')),
                'forward_pe': self._safe_float(data.get('forwardPE')),
                'peg_ratio': self._safe_float(data.get('pegRatio')),
                'price_to_book': self._safe_float(data.get('priceToBook')),
                'eps': self._safe_float(data.get('trailingEps')),
                'dividend_yield': self._safe_float(data.get('dividendYield')),
                'day_high': self._safe_float(data.get('dayHigh')),
                'day_low': self._safe_float(data.get('dayLow')),
                'year_high': self._safe_float(data.get('fiftyTwoWeekHigh')),
                'year_low': self._safe_float(data.get('fiftyTwoWeekLow')),
                'beta': self._safe_float(data.get('beta'))
            }
        )
    
    def _normalize_alpha_vantage_asset(self, data: Dict[str, Any]) -> NormalizedAsset:
        """Normalize Alpha Vantage asset data."""
        # Alpha Vantage has different response formats
        if 'Realtime Currency Exchange Rate' in data:
            # Forex data
            rate_data = data['Realtime Currency Exchange Rate']
            from_currency = rate_data.get('1. From_Currency Code', '')
            to_currency = rate_data.get('2. To_Currency Code', '')
            symbol = f"{from_currency}{to_currency}=X"
            
            return NormalizedAsset(
                symbol=symbol,
                name=f"{from_currency}/{to_currency}",
                category=AssetCategory.FOREX,
                source=DataSource.ALPHA_VANTAGE,
                current_price=self._safe_float(rate_data.get('5. Exchange Rate')),
                last_updated=self._parse_datetime(rate_data.get('6. Last Refreshed')),
                extended_data={
                    'bid_price': self._safe_float(rate_data.get('8. Bid Price')),
                    'ask_price': self._safe_float(rate_data.get('9. Ask Price'))
                }
            )
        
        elif 'Global Quote' in data:
            # Stock data
            quote_data = data['Global Quote']
            symbol = quote_data.get('01. symbol', '')
            
            return NormalizedAsset(
                symbol=symbol,
                name=symbol,
                category=AssetCategory.STOCK,
                source=DataSource.ALPHA_VANTAGE,
                current_price=self._safe_float(quote_data.get('05. price')),
                price_change_24h=self._safe_float(quote_data.get('09. change')),
                price_change_percentage_24h=self._safe_float(
                    quote_data.get('10. change percent', '').replace('%', '')
                ),
                volume_24h=self._safe_float(quote_data.get('06. volume')),
                last_updated=self._parse_datetime(quote_data.get('07. latest trading day')),
                extended_data={
                    'open': self._safe_float(quote_data.get('02. open')),
                    'high': self._safe_float(quote_data.get('03. high')),
                    'low': self._safe_float(quote_data.get('04. low')),
                    'previous_close': self._safe_float(quote_data.get('08. previous close'))
                }
            )
        
        return None
    
    def _normalize_finnhub_asset(self, data: Dict[str, Any]) -> NormalizedAsset:
        """Normalize Finnhub asset data."""
        symbol = data.get('symbol', '')
        
        return NormalizedAsset(
            symbol=symbol,
            name=data.get('description', symbol),
            category=AssetCategory.STOCK,
            source=DataSource.FINNHUB,
            current_price=self._safe_float(data.get('c')),  # Current price
            price_change_24h=self._safe_float(data.get('d')),  # Change
            price_change_percentage_24h=self._safe_float(data.get('dp')),  # Percent change
            last_updated=datetime.now(),
            extended_data={
                'open': self._safe_float(data.get('o')),
                'high': self._safe_float(data.get('h')),
                'low': self._safe_float(data.get('l')),
                'previous_close': self._safe_float(data.get('pc'))
            }
        )
    
    def _normalize_fmp_asset(self, data: Dict[str, Any]) -> NormalizedAsset:
        """Normalize Financial Modeling Prep asset data."""
        symbol = data.get('symbol', '')
        
        return NormalizedAsset(
            symbol=symbol,
            name=data.get('name', symbol),
            category=AssetCategory.STOCK,
            source=DataSource.FMP,
            exchange=self._normalize_exchange(data.get('exchange')),
            current_price=self._safe_float(data.get('price')),
            price_change_24h=self._safe_float(data.get('change')),
            price_change_percentage_24h=self._safe_float(data.get('changesPercentage')),
            volume_24h=self._safe_float(data.get('volume')),
            market_cap=self._safe_float(data.get('marketCap')),
            last_updated=datetime.now(),
            extended_data={
                'day_low': self._safe_float(data.get('dayLow')),
                'day_high': self._safe_float(data.get('dayHigh')),
                'year_low': self._safe_float(data.get('yearLow')),
                'year_high': self._safe_float(data.get('yearHigh')),
                'pe_ratio': self._safe_float(data.get('pe')),
                'eps': self._safe_float(data.get('eps')),
                'earnings_announcement': data.get('earningsAnnouncement'),
                'shares_outstanding': self._safe_float(data.get('sharesOutstanding')),
                'avg_volume': self._safe_float(data.get('avgVolume'))
            }
        )
    
    def _normalize_binance_asset(self, data: Dict[str, Any]) -> NormalizedAsset:
        """Normalize Binance asset data."""
        symbol = data.get('symbol', '')
        
        # Extract base currency from symbol (e.g., BTCUSDT -> BTC)
        base_currency = symbol.replace('USDT', '').replace('BUSD', '').replace('BTC', '')
        if not base_currency:
            base_currency = symbol[:3]  # Fallback
        
        return NormalizedAsset(
            symbol=base_currency,
            name=base_currency,
            category=AssetCategory.CRYPTO,
            source=DataSource.BINANCE,
            exchange='Binance',
            current_price=self._safe_float(data.get('price') or data.get('last')),
            price_change_24h=self._safe_float(data.get('change')),
            price_change_percentage_24h=self._safe_float(data.get('percentage')),
            volume_24h=self._safe_float(data.get('quoteVolume')),
            last_updated=datetime.now(),
            extended_data={
                'base_volume': self._safe_float(data.get('baseVolume')),
                'high': self._safe_float(data.get('high')),
                'low': self._safe_float(data.get('low')),
                'open': self._safe_float(data.get('open')),
                'close': self._safe_float(data.get('close')),
                'bid': self._safe_float(data.get('bid')),
                'ask': self._safe_float(data.get('ask')),
                'vwap': self._safe_float(data.get('vwap'))
            }
        )
    
    def normalize_price_history(self, raw_data: List[Dict[str, Any]], 
                              source: DataSource, symbol: str) -> List[NormalizedPriceData]:
        """Normalize historical price data from any source."""
        try:
            if source == DataSource.YAHOO_FINANCE:
                return self._normalize_yahoo_price_history(raw_data, symbol)
            elif source == DataSource.ALPHA_VANTAGE:
                return self._normalize_alpha_vantage_price_history(raw_data, symbol)
            elif source == DataSource.COINGECKO:
                return self._normalize_coingecko_price_history(raw_data, symbol)
            elif source == DataSource.BINANCE:
                return self._normalize_binance_price_history(raw_data, symbol)
            else:
                logger.warning(f"Unsupported price history source: {source}")
                return []
        except Exception as e:
            logger.error(f"Error normalizing price history from {source}: {e}")
            return []
    
    def _normalize_yahoo_price_history(self, data: List[Dict[str, Any]], symbol: str) -> List[NormalizedPriceData]:
        """Normalize Yahoo Finance price history."""
        normalized_data = []
        
        for item in data:
            normalized_data.append(NormalizedPriceData(
                symbol=symbol,
                timestamp=self._parse_datetime(item.get('date')),
                open_price=self._safe_float(item.get('open')),
                high_price=self._safe_float(item.get('high')),
                low_price=self._safe_float(item.get('low')),
                close_price=self._safe_float(item.get('close')),
                volume=self._safe_float(item.get('volume')),
                source=DataSource.YAHOO_FINANCE
            ))
        
        return normalized_data
    
    def _normalize_alpha_vantage_price_history(self, data: Dict[str, Any], symbol: str) -> List[NormalizedPriceData]:
        """Normalize Alpha Vantage price history."""
        normalized_data = []
        
        # Alpha Vantage returns time series data
        time_series_key = None
        for key in data.keys():
            if 'Time Series' in key or 'Daily' in key:
                time_series_key = key
                break
        
        if not time_series_key:
            return normalized_data
        
        time_series = data[time_series_key]
        
        for date_str, price_data in time_series.items():
            normalized_data.append(NormalizedPriceData(
                symbol=symbol,
                timestamp=self._parse_datetime(date_str),
                open_price=self._safe_float(price_data.get('1. open')),
                high_price=self._safe_float(price_data.get('2. high')),
                low_price=self._safe_float(price_data.get('3. low')),
                close_price=self._safe_float(price_data.get('4. close')),
                volume=self._safe_float(price_data.get('5. volume')),
                source=DataSource.ALPHA_VANTAGE
            ))
        
        return sorted(normalized_data, key=lambda x: x.timestamp)
    
    def _normalize_coingecko_price_history(self, data: List[List], symbol: str) -> List[NormalizedPriceData]:
        """Normalize CoinGecko price history."""
        normalized_data = []
        
        # CoinGecko returns [[timestamp, price], ...]
        for item in data:
            if len(item) >= 2:
                timestamp = datetime.fromtimestamp(item[0] / 1000)  # Convert from milliseconds
                price = self._safe_float(item[1])
                
                normalized_data.append(NormalizedPriceData(
                    symbol=symbol,
                    timestamp=timestamp,
                    close_price=price,
                    source=DataSource.COINGECKO
                ))
        
        return normalized_data
    
    def _normalize_binance_price_history(self, data: List[List], symbol: str) -> List[NormalizedPriceData]:
        """Normalize Binance OHLCV data."""
        normalized_data = []
        
        # Binance returns [[timestamp, open, high, low, close, volume, ...], ...]
        for item in data:
            if len(item) >= 6:
                normalized_data.append(NormalizedPriceData(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(item[0] / 1000),
                    open_price=self._safe_float(item[1]),
                    high_price=self._safe_float(item[2]),
                    low_price=self._safe_float(item[3]),
                    close_price=self._safe_float(item[4]),
                    volume=self._safe_float(item[5]),
                    source=DataSource.BINANCE
                ))
        
        return normalized_data
    
    def aggregate_data_from_sources(self, data_by_source: Dict[DataSource, List[NormalizedAsset]]) -> List[NormalizedAsset]:
        """Aggregate and deduplicate data from multiple sources."""
        symbol_data = {}
        
        # Priority order for data sources (higher index = higher priority)
        source_priority = {
            DataSource.YAHOO_FINANCE: 1,
            DataSource.ALPHA_VANTAGE: 2,
            DataSource.FINNHUB: 3,
            DataSource.COINGECKO: 4,
            DataSource.FMP: 5,
            DataSource.BINANCE: 6
        }
        
        for source, assets in data_by_source.items():
            for asset in assets:
                symbol = asset.symbol
                
                if symbol not in symbol_data:
                    symbol_data[symbol] = asset
                else:
                    # Use higher priority source or merge data
                    existing_priority = source_priority.get(symbol_data[symbol].source, 0)
                    new_priority = source_priority.get(source, 0)
                    
                    if new_priority > existing_priority:
                        # Keep extended data from both sources
                        merged_extended_data = {**symbol_data[symbol].extended_data, **asset.extended_data}
                        asset.extended_data = merged_extended_data
                        symbol_data[symbol] = asset
                    else:
                        # Merge extended data into existing asset
                        symbol_data[symbol].extended_data.update(asset.extended_data)
        
        return list(symbol_data.values())
    
    def validate_and_clean_data(self, assets: List[NormalizedAsset]) -> List[NormalizedAsset]:
        """Validate and clean normalized asset data."""
        cleaned_assets = []
        
        for asset in assets:
            # Skip assets with invalid data
            if not asset.symbol or not asset.name:
                continue
            
            # Clean and validate prices
            if asset.current_price is not None:
                if asset.current_price <= 0:
                    asset.current_price = None
                else:
                    asset.current_price = self._round_price(asset.current_price)
            
            # Clean percentage changes
            if asset.price_change_percentage_24h is not None:
                if abs(asset.price_change_percentage_24h) > 1000:  # Sanity check
                    asset.price_change_percentage_24h = None
                else:
                    asset.price_change_percentage_24h = round(asset.price_change_percentage_24h, 2)
            
            # Clean volume and market cap
            if asset.volume_24h is not None and asset.volume_24h < 0:
                asset.volume_24h = None
            
            if asset.market_cap is not None and asset.market_cap < 0:
                asset.market_cap = None
            
            # Normalize symbol case
            asset.symbol = asset.symbol.upper()
            
            # Clean extended data
            asset.extended_data = self._clean_extended_data(asset.extended_data)
            
            cleaned_assets.append(asset)
        
        return cleaned_assets
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float."""
        if value is None or value == '':
            return None
        
        try:
            if isinstance(value, str):
                # Remove common non-numeric characters
                value = value.replace(',', '').replace('%', '').replace('$', '')
            
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _round_price(self, price: float) -> float:
        """Round price to appropriate decimal places."""
        if price >= 1000:
            return round(price, 2)
        elif price >= 1:
            return round(price, 4)
        else:
            return round(price, 8)
    
    def _parse_datetime(self, date_str: Any) -> Optional[datetime]:
        """Parse datetime from various string formats."""
        if not date_str:
            return None
        
        if isinstance(date_str, datetime):
            return date_str
        
        try:
            # Common datetime formats
            formats = [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%m/%d/%Y',
                '%d/%m/%Y'
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(str(date_str), fmt)
                except ValueError:
                    continue
            
            # Try pandas parsing as fallback
            return pd.to_datetime(date_str)
            
        except Exception:
            return None
    
    def _normalize_currency(self, currency: str) -> str:
        """Normalize currency code."""
        if not currency:
            return 'USD'
        
        currency_lower = currency.lower()
        return self.currency_mappings.get(currency_lower, currency.upper())
    
    def _normalize_exchange(self, exchange: str) -> Optional[str]:
        """Normalize exchange name."""
        if not exchange:
            return None
        
        exchange_lower = exchange.lower()
        return self.exchange_mappings.get(exchange_lower, exchange)
    
    def _clean_extended_data(self, extended_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate extended data."""
        cleaned_data = {}
        
        for key, value in extended_data.items():
            if value is not None and value != '':
                # Convert numeric strings to numbers
                if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                    try:
                        cleaned_data[key] = float(value)
                    except ValueError:
                        cleaned_data[key] = value
                else:
                    cleaned_data[key] = value
        
        return cleaned_data
    
    def create_chart_data(self, price_history: List[NormalizedPriceData], 
                         interval: str = '1d') -> ChartData:
        """Create chart data from normalized price history."""
        if not price_history:
            return ChartData(symbol='', interval=interval, data=[])
        
        # Sort by timestamp
        sorted_history = sorted(price_history, key=lambda x: x.timestamp)
        
        # Convert to MarketDataPoint list
        data_points = [item.to_market_data_point() for item in sorted_history]
        
        return ChartData(
            symbol=sorted_history[0].symbol,
            interval=interval,
            data=data_points
        )
    
    def calculate_technical_indicators(self, price_history: List[NormalizedPriceData]) -> Dict[str, Any]:
        """Calculate basic technical indicators from price history."""
        if len(price_history) < 20:
            return {}
        
        # Convert to pandas DataFrame
        df = pd.DataFrame([
            {
                'timestamp': item.timestamp,
                'open': item.open_price,
                'high': item.high_price,
                'low': item.low_price,
                'close': item.close_price,
                'volume': item.volume
            }
            for item in price_history
        ])
        
        df = df.dropna(subset=['close'])
        
        if len(df) < 20:
            return {}
        
        indicators = {}
        
        try:
            # Simple Moving Averages
            indicators['sma_20'] = df['close'].rolling(window=20).mean().iloc[-1]
            indicators['sma_50'] = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None
            
            # Exponential Moving Average
            indicators['ema_12'] = df['close'].ewm(span=12).mean().iloc[-1]
            indicators['ema_26'] = df['close'].ewm(span=26).mean().iloc[-1]
            
            # MACD
            ema_12 = df['close'].ewm(span=12).mean()
            ema_26 = df['close'].ewm(span=26).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['macd'] = macd_line.iloc[-1]
            indicators['macd_signal'] = signal_line.iloc[-1]
            indicators['macd_histogram'] = (macd_line - signal_line).iloc[-1]
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
            
            # Bollinger Bands
            sma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()
            indicators['bb_upper'] = (sma_20 + (std_20 * 2)).iloc[-1]
            indicators['bb_lower'] = (sma_20 - (std_20 * 2)).iloc[-1]
            indicators['bb_middle'] = sma_20.iloc[-1]
            
            # Volume indicators
            if 'volume' in df.columns and not df['volume'].isna().all():
                indicators['volume_sma_20'] = df['volume'].rolling(window=20).mean().iloc[-1]
                indicators['volume_ratio'] = df['volume'].iloc[-1] / indicators['volume_sma_20'] if indicators['volume_sma_20'] > 0 else None
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        # Round all float values
        for key, value in indicators.items():
            if isinstance(value, float) and not np.isnan(value):
                indicators[key] = round(value, 6)
        
        return indicators

# Create global instance
data_normalization_service = DataNormalizationService()