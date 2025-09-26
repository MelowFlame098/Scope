"""Comprehensive Technical Indicators Engine for Multi-Asset Trading

This module implements advanced technical indicators for:
- Crypto: S2F, Metcalfe's Law, NVT/NVM, On-chain metrics, ML models
- Stocks: DCF, DDM, CAPM, Fama-French, ARIMA, GARCH, ML models
- Forex: PPP, IRP, UIP, Balance of Payments, Monetary models
- Futures: Cost-of-Carry, Convenience Yield, Samuelson Effect
- Indexes: Macroeconomic factors, APT, Term structure models
- Cross-Asset: Advanced ML, RL, and sentiment analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, XGBRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Import modularized comprehensive indicators
from indicators.crypto.crypto_comprehensive import CryptoComprehensiveIndicators
from indicators.stock.stock_comprehensive import StockComprehensiveIndicators
from indicators.forex.forex_comprehensive import ForexComprehensiveIndicators
from indicators.futures.futures_comprehensive import FuturesComprehensiveIndicators
from indicators.cross_asset.cross_asset_comprehensive import CrossAssetComprehensiveIndicators

logger = logging.getLogger(__name__)

class AssetType(str, Enum):
    """Asset type classification"""
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    FUTURES = "futures"
    INDEX = "index"
    CROSS_ASSET = "cross_asset"

class IndicatorCategory(str, Enum):
    """Indicator category classification"""
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MACHINE_LEARNING = "machine_learning"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ECONOMETRIC = "econometric"
    ON_CHAIN = "on_chain"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    TREND = "trend"
    VOLUME = "volume"

@dataclass
class IndicatorResult:
    """Result of indicator calculation"""
    name: str
    values: Union[pd.Series, pd.DataFrame]
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory

class ComprehensiveIndicators:
    """Comprehensive technical indicators calculation engine"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.cache = {}
        
        # Initialize modularized indicator classes
        self.crypto_indicators = CryptoComprehensiveIndicators()
        self.stock_indicators = StockComprehensiveIndicators()
        self.forex_indicators = ForexComprehensiveIndicators()
        self.futures_indicators = FuturesComprehensiveIndicators()
        self.cross_asset_indicators = CrossAssetComprehensiveIndicators()
        

    

    
    def _empty_result(self, name: str, asset_type: AssetType) -> IndicatorResult:
        """Return empty result for error cases"""
        return IndicatorResult(
            name=name,
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.TECHNICAL
        )
    
    async def calculate_indicator(self, indicator_name: str, data: pd.DataFrame, 
                                 asset_type: AssetType, **kwargs) -> IndicatorResult:
        """Calculate specific indicator based on name and asset type"""
        
        # Delegate to appropriate modularized class
        if asset_type == AssetType.CRYPTO:
            return await self.crypto_indicators.calculate_indicator(indicator_name, data, **kwargs)
        elif asset_type == AssetType.STOCK:
            return await self.stock_indicators.calculate_indicator(indicator_name, data, **kwargs)
        elif asset_type == AssetType.FOREX:
            return await self.forex_indicators.calculate_indicator(indicator_name, data, **kwargs)
        elif asset_type == AssetType.FUTURES:
            return await self.futures_indicators.calculate_indicator(indicator_name, data, **kwargs)
        elif asset_type == AssetType.CROSS_ASSET:
            return await self.cross_asset_indicators.calculate_indicator(indicator_name, data, **kwargs)
        
        return self._empty_result(f"Unknown asset type: {asset_type}", asset_type)