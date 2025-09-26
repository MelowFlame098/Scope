"""Advanced Feature Engineering for Stock Analysis

This module provides comprehensive feature engineering capabilities including:
- Technical indicators and patterns
- Fundamental ratio calculations
- Market microstructure features
- Time series decomposition
- Cross-asset correlations
- Alternative data integration
- Feature selection and dimensionality reduction

Author: Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
import json
warnings.filterwarnings('ignore')

# Try to import additional libraries for advanced features
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.decomposition import PCA, FastICA
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    from scipy.signal import find_peaks, savgol_filter
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

class FeatureCategory(Enum):
    """Categories of features"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    SENTIMENT = "sentiment"
    MACRO_ECONOMIC = "macro_economic"
    CROSS_ASSET = "cross_asset"
    ALTERNATIVE = "alternative"
    DERIVED = "derived"

class ScalingMethod(Enum):
    """Feature scaling methods"""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"

class FeatureSelectionMethod(Enum):
    """Feature selection methods"""
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    F_REGRESSION = "f_regression"
    RANDOM_FOREST = "random_forest"
    PCA = "pca"
    VARIANCE_THRESHOLD = "variance_threshold"

@dataclass
class FeatureMetadata:
    """Metadata for individual features"""
    name: str
    category: FeatureCategory
    description: str
    data_type: str
    importance_score: float
    correlation_with_target: float
    missing_ratio: float
    creation_timestamp: datetime
    computation_time: float
    dependencies: List[str]
    parameters: Dict[str, Any]

@dataclass
class FeatureSet:
    """Complete feature set with metadata"""
    features: pd.DataFrame
    metadata: Dict[str, FeatureMetadata]
    target_column: Optional[str]
    feature_categories: Dict[FeatureCategory, List[str]]
    scaling_info: Dict[str, Any]
    selection_info: Dict[str, Any]
    creation_timestamp: datetime
    total_features: int
    selected_features: int
    data_quality_score: float

class TechnicalIndicatorEngine:
    """Engine for calculating technical indicators"""
    
    def __init__(self):
        self.indicators = {}
    
    def calculate_moving_averages(self, prices: np.ndarray, windows: List[int] = [5, 10, 20, 50, 200]) -> Dict[str, np.ndarray]:
        """Calculate various moving averages"""
        mas = {}
        
        for window in windows:
            if len(prices) >= window:
                ma = pd.Series(prices).rolling(window=window).mean().values
                mas[f'ma_{window}'] = ma
                
                # Moving average ratios
                if window > 5:
                    ma_5 = pd.Series(prices).rolling(window=5).mean().values
                    ratio = ma_5 / (ma + 1e-8)
                    mas[f'ma_ratio_{window}'] = ratio
        
        return mas
    
    def calculate_momentum_indicators(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Calculate momentum-based indicators"""
        indicators = {}
        
        # Rate of Change (ROC)
        for period in [1, 5, 10, 20]:
            if len(prices) > period:
                roc = (prices[period:] - prices[:-period]) / (prices[:-period] + 1e-8)
                roc = np.concatenate([np.full(period, np.nan), roc])
                indicators[f'roc_{period}'] = roc
        
        # Relative Strength Index (RSI)
        if len(prices) >= 14:
            rsi = self._calculate_rsi(prices, 14)
            indicators['rsi_14'] = rsi
        
        # Stochastic Oscillator
        if len(prices) >= 14:
            stoch_k, stoch_d = self._calculate_stochastic(prices, prices, prices, 14, 3)
            indicators['stoch_k'] = stoch_k
            indicators['stoch_d'] = stoch_d
        
        # MACD
        if len(prices) >= 26:
            macd, macd_signal, macd_hist = self._calculate_macd(prices)
            indicators['macd'] = macd
            indicators['macd_signal'] = macd_signal
            indicators['macd_histogram'] = macd_hist
        
        # Williams %R
        if len(prices) >= 14:
            williams_r = self._calculate_williams_r(prices, prices, prices, 14)
            indicators['williams_r'] = williams_r
        
        return indicators
    
    def calculate_volatility_indicators(self, prices: np.ndarray, volumes: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Calculate volatility-based indicators"""
        indicators = {}
        
        # Historical Volatility
        for window in [10, 20, 30]:
            if len(prices) >= window:
                returns = np.diff(np.log(prices + 1e-8))
                vol = pd.Series(returns).rolling(window=window).std().values
                vol = np.concatenate([np.full(1, np.nan), vol])
                indicators[f'volatility_{window}'] = vol
        
        # Bollinger Bands
        if len(prices) >= 20:
            bb_upper, bb_middle, bb_lower, bb_width, bb_position = self._calculate_bollinger_bands(prices, 20, 2)
            indicators['bb_upper'] = bb_upper
            indicators['bb_middle'] = bb_middle
            indicators['bb_lower'] = bb_lower
            indicators['bb_width'] = bb_width
            indicators['bb_position'] = bb_position
        
        # Average True Range (ATR)
        if len(prices) >= 14:
            atr = self._calculate_atr(prices, prices, prices, 14)
            indicators['atr'] = atr
        
        return indicators
    
    def calculate_volume_indicators(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate volume-based indicators"""
        if volumes is None:
            return {}
        
        indicators = {}
        
        # Volume Moving Average
        for window in [10, 20, 50]:
            if len(volumes) >= window:
                vol_ma = pd.Series(volumes).rolling(window=window).mean().values
                indicators[f'volume_ma_{window}'] = vol_ma
                
                # Volume ratio
                vol_ratio = volumes / (vol_ma + 1e-8)
                indicators[f'volume_ratio_{window}'] = vol_ratio
        
        # On-Balance Volume (OBV)
        obv = self._calculate_obv(prices, volumes)
        indicators['obv'] = obv
        
        # Volume Price Trend (VPT)
        vpt = self._calculate_vpt(prices, volumes)
        indicators['vpt'] = vpt
        
        # Accumulation/Distribution Line
        ad_line = self._calculate_ad_line(prices, prices, prices, volumes)
        indicators['ad_line'] = ad_line
        
        return indicators
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(window=period).mean().values
        avg_losses = pd.Series(losses).rolling(window=period).mean().values
        
        rs = avg_gains / (avg_losses + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([np.full(1, np.nan), rsi])
    
    def _calculate_stochastic(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                            k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator"""
        lowest_low = pd.Series(low).rolling(window=k_period).min().values
        highest_high = pd.Series(high).rolling(window=k_period).max().values
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-8)
        d_percent = pd.Series(k_percent).rolling(window=d_period).mean().values
        
        return k_percent, d_percent
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate MACD"""
        ema_fast = pd.Series(prices).ewm(span=fast).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow).mean().values
        
        macd = ema_fast - ema_slow
        macd_signal = pd.Series(macd).ewm(span=signal).mean().values
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def _calculate_williams_r(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Williams %R"""
        highest_high = pd.Series(high).rolling(window=period).max().values
        lowest_low = pd.Series(low).rolling(window=period).min().values
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)
        
        return williams_r
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(window=period).mean().values
        std = pd.Series(prices).rolling(window=period).std().values
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        # Band width and position
        band_width = (upper_band - lower_band) / (sma + 1e-8)
        band_position = (prices - lower_band) / (upper_band - lower_band + 1e-8)
        
        return upper_band, sma, lower_band, band_width, band_position
    
    def _calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(window=period).mean().values
        
        return atr
    
    def _calculate_obv(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate On-Balance Volume"""
        price_changes = np.diff(prices)
        obv = np.zeros(len(prices))
        
        for i in range(1, len(prices)):
            if price_changes[i-1] > 0:
                obv[i] = obv[i-1] + volumes[i]
            elif price_changes[i-1] < 0:
                obv[i] = obv[i-1] - volumes[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def _calculate_vpt(self, prices: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate Volume Price Trend"""
        price_changes = np.diff(prices)
        price_changes = np.concatenate([np.array([0]), price_changes])
        
        vpt = np.zeros(len(prices))
        for i in range(1, len(prices)):
            vpt[i] = vpt[i-1] + volumes[i] * (price_changes[i] / (prices[i-1] + 1e-8))
        
        return vpt
    
    def _calculate_ad_line(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, volumes: np.ndarray) -> np.ndarray:
        """Calculate Accumulation/Distribution Line"""
        clv = ((close - low) - (high - close)) / (high - low + 1e-8)
        ad_line = np.cumsum(clv * volumes)
        
        return ad_line

class FundamentalFeatureEngine:
    """Engine for calculating fundamental analysis features"""
    
    def __init__(self):
        self.ratios = {}
    
    def calculate_valuation_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate valuation ratios"""
        ratios = {}
        
        # Price ratios
        if 'market_cap' in financial_data and 'net_income' in financial_data:
            ratios['pe_ratio'] = financial_data['market_cap'] / (financial_data['net_income'] + 1e-8)
        
        if 'market_cap' in financial_data and 'book_value' in financial_data:
            ratios['pb_ratio'] = financial_data['market_cap'] / (financial_data['book_value'] + 1e-8)
        
        if 'market_cap' in financial_data and 'revenue' in financial_data:
            ratios['ps_ratio'] = financial_data['market_cap'] / (financial_data['revenue'] + 1e-8)
        
        if 'enterprise_value' in financial_data and 'ebitda' in financial_data:
            ratios['ev_ebitda'] = financial_data['enterprise_value'] / (financial_data['ebitda'] + 1e-8)
        
        # Yield ratios
        if 'dividend_per_share' in financial_data and 'price_per_share' in financial_data:
            ratios['dividend_yield'] = financial_data['dividend_per_share'] / (financial_data['price_per_share'] + 1e-8)
        
        if 'earnings_per_share' in financial_data and 'price_per_share' in financial_data:
            ratios['earnings_yield'] = financial_data['earnings_per_share'] / (financial_data['price_per_share'] + 1e-8)
        
        return ratios
    
    def calculate_profitability_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate profitability ratios"""
        ratios = {}
        
        # Margin ratios
        if 'net_income' in financial_data and 'revenue' in financial_data:
            ratios['net_margin'] = financial_data['net_income'] / (financial_data['revenue'] + 1e-8)
        
        if 'gross_profit' in financial_data and 'revenue' in financial_data:
            ratios['gross_margin'] = financial_data['gross_profit'] / (financial_data['revenue'] + 1e-8)
        
        if 'operating_income' in financial_data and 'revenue' in financial_data:
            ratios['operating_margin'] = financial_data['operating_income'] / (financial_data['revenue'] + 1e-8)
        
        # Return ratios
        if 'net_income' in financial_data and 'total_assets' in financial_data:
            ratios['roa'] = financial_data['net_income'] / (financial_data['total_assets'] + 1e-8)
        
        if 'net_income' in financial_data and 'shareholders_equity' in financial_data:
            ratios['roe'] = financial_data['net_income'] / (financial_data['shareholders_equity'] + 1e-8)
        
        if 'net_income' in financial_data and 'invested_capital' in financial_data:
            ratios['roic'] = financial_data['net_income'] / (financial_data['invested_capital'] + 1e-8)
        
        return ratios
    
    def calculate_liquidity_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        ratios = {}
        
        if 'current_assets' in financial_data and 'current_liabilities' in financial_data:
            ratios['current_ratio'] = financial_data['current_assets'] / (financial_data['current_liabilities'] + 1e-8)
        
        if 'quick_assets' in financial_data and 'current_liabilities' in financial_data:
            ratios['quick_ratio'] = financial_data['quick_assets'] / (financial_data['current_liabilities'] + 1e-8)
        
        if 'cash' in financial_data and 'current_liabilities' in financial_data:
            ratios['cash_ratio'] = financial_data['cash'] / (financial_data['current_liabilities'] + 1e-8)
        
        return ratios
    
    def calculate_leverage_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate leverage ratios"""
        ratios = {}
        
        if 'total_debt' in financial_data and 'shareholders_equity' in financial_data:
            ratios['debt_to_equity'] = financial_data['total_debt'] / (financial_data['shareholders_equity'] + 1e-8)
        
        if 'total_debt' in financial_data and 'total_assets' in financial_data:
            ratios['debt_to_assets'] = financial_data['total_debt'] / (financial_data['total_assets'] + 1e-8)
        
        if 'ebitda' in financial_data and 'interest_expense' in financial_data:
            ratios['interest_coverage'] = financial_data['ebitda'] / (financial_data['interest_expense'] + 1e-8)
        
        return ratios
    
    def calculate_efficiency_ratios(self, financial_data: Dict[str, float]) -> Dict[str, float]:
        """Calculate efficiency ratios"""
        ratios = {}
        
        if 'revenue' in financial_data and 'total_assets' in financial_data:
            ratios['asset_turnover'] = financial_data['revenue'] / (financial_data['total_assets'] + 1e-8)
        
        if 'revenue' in financial_data and 'inventory' in financial_data:
            ratios['inventory_turnover'] = financial_data['revenue'] / (financial_data['inventory'] + 1e-8)
        
        if 'revenue' in financial_data and 'accounts_receivable' in financial_data:
            ratios['receivables_turnover'] = financial_data['revenue'] / (financial_data['accounts_receivable'] + 1e-8)
        
        return ratios

class MarketMicrostructureEngine:
    """Engine for market microstructure features"""
    
    def __init__(self):
        self.features = {}
    
    def calculate_spread_features(self, bid_prices: np.ndarray, ask_prices: np.ndarray, 
                                mid_prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate bid-ask spread features"""
        features = {}
        
        # Absolute spread
        spread = ask_prices - bid_prices
        features['bid_ask_spread'] = spread
        
        # Relative spread
        relative_spread = spread / (mid_prices + 1e-8)
        features['relative_spread'] = relative_spread
        
        # Spread volatility
        for window in [10, 20, 50]:
            if len(spread) >= window:
                spread_vol = pd.Series(spread).rolling(window=window).std().values
                features[f'spread_volatility_{window}'] = spread_vol
        
        return features
    
    def calculate_order_flow_features(self, buy_volume: np.ndarray, sell_volume: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate order flow imbalance features"""
        features = {}
        
        # Order flow imbalance
        total_volume = buy_volume + sell_volume
        ofi = (buy_volume - sell_volume) / (total_volume + 1e-8)
        features['order_flow_imbalance'] = ofi
        
        # Cumulative order flow
        features['cumulative_ofi'] = np.cumsum(ofi)
        
        # Order flow momentum
        for window in [5, 10, 20]:
            if len(ofi) >= window:
                ofi_momentum = pd.Series(ofi).rolling(window=window).mean().values
                features[f'ofi_momentum_{window}'] = ofi_momentum
        
        return features
    
    def calculate_price_impact_features(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate price impact features"""
        features = {}
        
        # Price changes
        price_changes = np.diff(prices)
        price_changes = np.concatenate([np.array([0]), price_changes])
        
        # Volume-weighted price impact
        vwap_impact = price_changes / (volumes + 1e-8)
        features['volume_weighted_impact'] = vwap_impact
        
        # Temporary vs permanent impact (simplified)
        for window in [5, 10]:
            if len(price_changes) >= window:
                temp_impact = pd.Series(price_changes).rolling(window=window).mean().values
                features[f'temporary_impact_{window}'] = temp_impact
        
        return features

class AdvancedFeatureEngineer:
    """Main feature engineering class"""
    
    def __init__(self):
        self.technical_engine = TechnicalIndicatorEngine()
        self.fundamental_engine = FundamentalFeatureEngine()
        self.microstructure_engine = MarketMicrostructureEngine()
        
        self.scalers = {}
        self.feature_selectors = {}
        self.feature_metadata = {}
    
    def engineer_features(self, stock_data: Dict[str, Any], market_data: Dict[str, Any] = None) -> FeatureSet:
        """Main feature engineering pipeline"""
        start_time = datetime.now()
        
        # Initialize feature dataframe
        features_dict = {}
        metadata_dict = {}
        
        # Extract basic data
        prices = np.array(stock_data.get('historical_prices', []))
        volumes = np.array(stock_data.get('volume', []))
        
        if len(prices) == 0:
            return self._create_empty_feature_set()
        
        # Technical features
        tech_features = self._engineer_technical_features(prices, volumes)
        features_dict.update(tech_features)
        
        # Fundamental features
        fund_features = self._engineer_fundamental_features(stock_data)
        features_dict.update(fund_features)
        
        # Market microstructure features (if available)
        if 'bid_prices' in stock_data and 'ask_prices' in stock_data:
            micro_features = self._engineer_microstructure_features(stock_data)
            features_dict.update(micro_features)
        
        # Cross-asset features (if market data available)
        if market_data:
            cross_features = self._engineer_cross_asset_features(stock_data, market_data)
            features_dict.update(cross_features)
        
        # Time-based features
        time_features = self._engineer_time_features(len(prices))
        features_dict.update(time_features)
        
        # Derived features
        derived_features = self._engineer_derived_features(features_dict)
        features_dict.update(derived_features)
        
        # Create DataFrame
        max_length = max(len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in features_dict.values())
        
        # Align all features to same length
        aligned_features = {}
        for name, values in features_dict.items():
            if isinstance(values, (list, np.ndarray)):
                if len(values) == max_length:
                    aligned_features[name] = values
                elif len(values) < max_length:
                    # Pad with NaN
                    padded = np.full(max_length, np.nan)
                    padded[-len(values):] = values
                    aligned_features[name] = padded
                else:
                    # Truncate
                    aligned_features[name] = values[-max_length:]
            else:
                # Scalar value - repeat for all rows
                aligned_features[name] = np.full(max_length, values)
        
        features_df = pd.DataFrame(aligned_features)
        
        # Create metadata
        for feature_name in features_df.columns:
            metadata_dict[feature_name] = self._create_feature_metadata(
                feature_name, features_df[feature_name]
            )
        
        # Calculate feature categories
        feature_categories = self._categorize_features(list(features_df.columns))
        
        # Calculate data quality score
        data_quality_score = self._calculate_data_quality_score(features_df)
        
        computation_time = (datetime.now() - start_time).total_seconds()
        
        return FeatureSet(
            features=features_df,
            metadata=metadata_dict,
            target_column=None,
            feature_categories=feature_categories,
            scaling_info={},
            selection_info={},
            creation_timestamp=datetime.now(),
            total_features=len(features_df.columns),
            selected_features=len(features_df.columns),
            data_quality_score=data_quality_score
        )
    
    def _engineer_technical_features(self, prices: np.ndarray, volumes: np.ndarray) -> Dict[str, np.ndarray]:
        """Engineer technical analysis features"""
        features = {}
        
        # Moving averages
        ma_features = self.technical_engine.calculate_moving_averages(prices)
        features.update(ma_features)
        
        # Momentum indicators
        momentum_features = self.technical_engine.calculate_momentum_indicators(prices, volumes)
        features.update(momentum_features)
        
        # Volatility indicators
        volatility_features = self.technical_engine.calculate_volatility_indicators(prices, volumes)
        features.update(volatility_features)
        
        # Volume indicators
        if len(volumes) > 0:
            volume_features = self.technical_engine.calculate_volume_indicators(prices, volumes)
            features.update(volume_features)
        
        # Price patterns
        pattern_features = self._detect_price_patterns(prices)
        features.update(pattern_features)
        
        return features
    
    def _engineer_fundamental_features(self, stock_data: Dict[str, Any]) -> Dict[str, float]:
        """Engineer fundamental analysis features"""
        features = {}
        
        # Extract financial data
        financial_data = {
            'market_cap': stock_data.get('market_cap', 0),
            'revenue': stock_data.get('revenue', 0),
            'net_income': stock_data.get('net_income', 0),
            'total_assets': stock_data.get('total_assets', 0),
            'shareholders_equity': stock_data.get('shareholders_equity', 0),
            'total_debt': stock_data.get('total_debt', 0),
            'free_cash_flow': stock_data.get('free_cash_flow', 0),
            'book_value': stock_data.get('book_value', 0),
            'ebitda': stock_data.get('ebitda', 0)
        }
        
        # Calculate ratio features
        valuation_ratios = self.fundamental_engine.calculate_valuation_ratios(financial_data)
        features.update(valuation_ratios)
        
        profitability_ratios = self.fundamental_engine.calculate_profitability_ratios(financial_data)
        features.update(profitability_ratios)
        
        liquidity_ratios = self.fundamental_engine.calculate_liquidity_ratios(financial_data)
        features.update(liquidity_ratios)
        
        leverage_ratios = self.fundamental_engine.calculate_leverage_ratios(financial_data)
        features.update(leverage_ratios)
        
        efficiency_ratios = self.fundamental_engine.calculate_efficiency_ratios(financial_data)
        features.update(efficiency_ratios)
        
        # Growth features
        growth_features = self._calculate_growth_features(stock_data)
        features.update(growth_features)
        
        return features
    
    def _engineer_microstructure_features(self, stock_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Engineer market microstructure features"""
        features = {}
        
        bid_prices = np.array(stock_data.get('bid_prices', []))
        ask_prices = np.array(stock_data.get('ask_prices', []))
        
        if len(bid_prices) > 0 and len(ask_prices) > 0:
            mid_prices = (bid_prices + ask_prices) / 2
            
            # Spread features
            spread_features = self.microstructure_engine.calculate_spread_features(
                bid_prices, ask_prices, mid_prices
            )
            features.update(spread_features)
            
            # Order flow features (if available)
            if 'buy_volume' in stock_data and 'sell_volume' in stock_data:
                buy_volume = np.array(stock_data['buy_volume'])
                sell_volume = np.array(stock_data['sell_volume'])
                
                flow_features = self.microstructure_engine.calculate_order_flow_features(
                    buy_volume, sell_volume
                )
                features.update(flow_features)
        
        return features
    
    def _engineer_cross_asset_features(self, stock_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, float]:
        """Engineer cross-asset correlation features"""
        features = {}
        
        # Market beta (simplified)
        stock_prices = np.array(stock_data.get('historical_prices', []))
        market_prices = np.array(market_data.get('market_index_prices', []))
        
        if len(stock_prices) > 1 and len(market_prices) > 1:
            min_length = min(len(stock_prices), len(market_prices))
            stock_returns = np.diff(stock_prices[-min_length:])
            market_returns = np.diff(market_prices[-min_length:])
            
            if len(stock_returns) > 0 and len(market_returns) > 0:
                # Beta calculation
                covariance = np.cov(stock_returns, market_returns)[0, 1]
                market_variance = np.var(market_returns)
                beta = covariance / (market_variance + 1e-8)
                features['market_beta'] = beta
                
                # Correlation
                correlation = np.corrcoef(stock_returns, market_returns)[0, 1]
                features['market_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        # Sector relative performance
        if 'sector_index_prices' in market_data:
            sector_prices = np.array(market_data['sector_index_prices'])
            if len(stock_prices) > 1 and len(sector_prices) > 1:
                min_length = min(len(stock_prices), len(sector_prices))
                stock_returns = np.diff(stock_prices[-min_length:])
                sector_returns = np.diff(sector_prices[-min_length:])
                
                if len(stock_returns) > 0 and len(sector_returns) > 0:
                    relative_performance = np.mean(stock_returns) - np.mean(sector_returns)
                    features['sector_relative_performance'] = relative_performance
        
        return features
    
    def _engineer_time_features(self, data_length: int) -> Dict[str, np.ndarray]:
        """Engineer time-based features"""
        features = {}
        
        # Time trend
        time_trend = np.arange(data_length) / data_length
        features['time_trend'] = time_trend
        
        # Cyclical features (assuming daily data)
        if data_length > 7:
            # Weekly cycle
            weekly_cycle = np.sin(2 * np.pi * np.arange(data_length) / 7)
            features['weekly_cycle'] = weekly_cycle
        
        if data_length > 30:
            # Monthly cycle
            monthly_cycle = np.sin(2 * np.pi * np.arange(data_length) / 30)
            features['monthly_cycle'] = monthly_cycle
        
        if data_length > 252:
            # Yearly cycle
            yearly_cycle = np.sin(2 * np.pi * np.arange(data_length) / 252)
            features['yearly_cycle'] = yearly_cycle
        
        return features
    
    def _engineer_derived_features(self, features_dict: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Engineer derived features from existing features"""
        derived = {}
        
        # Feature interactions (simplified)
        feature_names = list(features_dict.keys())
        
        # Select a few key features for interactions
        key_features = [name for name in feature_names if any(key in name.lower() 
                       for key in ['ma_', 'rsi', 'volatility', 'volume'])]
        
        # Create some interaction features
        for i, feat1 in enumerate(key_features[:5]):  # Limit to avoid explosion
            for feat2 in key_features[i+1:6]:
                if feat1 in features_dict and feat2 in features_dict:
                    val1 = features_dict[feat1]
                    val2 = features_dict[feat2]
                    
                    if isinstance(val1, (list, np.ndarray)) and isinstance(val2, (list, np.ndarray)):
                        if len(val1) == len(val2):
                            # Ratio feature
                            ratio = np.array(val1) / (np.array(val2) + 1e-8)
                            derived[f'{feat1}_{feat2}_ratio'] = ratio
        
        return derived
    
    def _detect_price_patterns(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """Detect price patterns"""
        patterns = {}
        
        if len(prices) < 10:
            return patterns
        
        # Support and resistance levels
        if SCIPY_AVAILABLE:
            # Find peaks and troughs
            peaks, _ = find_peaks(prices, distance=5)
            troughs, _ = find_peaks(-prices, distance=5)
            
            # Support/resistance strength
            support_strength = np.zeros(len(prices))
            resistance_strength = np.zeros(len(prices))
            
            for peak in peaks:
                if peak < len(resistance_strength):
                    resistance_strength[peak] = 1.0
            
            for trough in troughs:
                if trough < len(support_strength):
                    support_strength[trough] = 1.0
            
            patterns['resistance_strength'] = resistance_strength
            patterns['support_strength'] = support_strength
        
        # Price channels
        if len(prices) >= 20:
            # Simple channel detection using rolling max/min
            window = 20
            rolling_max = pd.Series(prices).rolling(window=window).max().values
            rolling_min = pd.Series(prices).rolling(window=window).min().values
            
            channel_position = (prices - rolling_min) / (rolling_max - rolling_min + 1e-8)
            patterns['channel_position'] = channel_position
        
        return patterns
    
    def _calculate_growth_features(self, stock_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate growth-related features"""
        features = {}
        
        # Revenue growth
        if 'revenue_growth' in stock_data:
            features['revenue_growth'] = stock_data['revenue_growth']
        
        # Earnings growth
        if 'earnings_growth' in stock_data:
            features['earnings_growth'] = stock_data['earnings_growth']
        
        # Historical price growth
        prices = stock_data.get('historical_prices', [])
        if len(prices) >= 252:  # At least 1 year of data
            price_growth_1y = (prices[-1] - prices[-252]) / (prices[-252] + 1e-8)
            features['price_growth_1y'] = price_growth_1y
        
        if len(prices) >= 63:  # At least 3 months of data
            price_growth_3m = (prices[-1] - prices[-63]) / (prices[-63] + 1e-8)
            features['price_growth_3m'] = price_growth_3m
        
        return features
    
    def _create_feature_metadata(self, feature_name: str, feature_values: pd.Series) -> FeatureMetadata:
        """Create metadata for a feature"""
        # Determine category
        category = self._determine_feature_category(feature_name)
        
        # Calculate statistics
        missing_ratio = feature_values.isna().sum() / len(feature_values)
        
        return FeatureMetadata(
            name=feature_name,
            category=category,
            description=f"Auto-generated feature: {feature_name}",
            data_type=str(feature_values.dtype),
            importance_score=0.0,  # To be calculated later
            correlation_with_target=0.0,  # To be calculated later
            missing_ratio=missing_ratio,
            creation_timestamp=datetime.now(),
            computation_time=0.0,
            dependencies=[],
            parameters={}
        )
    
    def _determine_feature_category(self, feature_name: str) -> FeatureCategory:
        """Determine feature category based on name"""
        name_lower = feature_name.lower()
        
        if any(tech in name_lower for tech in ['ma_', 'rsi', 'macd', 'bb_', 'atr', 'stoch', 'williams']):
            return FeatureCategory.TECHNICAL
        elif any(fund in name_lower for fund in ['ratio', 'margin', 'roe', 'roa', 'debt', 'growth']):
            return FeatureCategory.FUNDAMENTAL
        elif any(micro in name_lower for micro in ['spread', 'ofi', 'impact', 'flow']):
            return FeatureCategory.MARKET_MICROSTRUCTURE
        elif any(cross in name_lower for cross in ['beta', 'correlation', 'relative']):
            return FeatureCategory.CROSS_ASSET
        elif any(time in name_lower for time in ['trend', 'cycle']):
            return FeatureCategory.DERIVED
        else:
            return FeatureCategory.DERIVED
    
    def _categorize_features(self, feature_names: List[str]) -> Dict[FeatureCategory, List[str]]:
        """Categorize all features"""
        categories = {category: [] for category in FeatureCategory}
        
        for name in feature_names:
            category = self._determine_feature_category(name)
            categories[category].append(name)
        
        return categories
    
    def _calculate_data_quality_score(self, features_df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        # Simple quality score based on missing values and variance
        missing_ratio = features_df.isna().sum().sum() / (features_df.shape[0] * features_df.shape[1])
        
        # Variance score (features with zero variance are problematic)
        numeric_features = features_df.select_dtypes(include=[np.number])
        zero_variance_ratio = (numeric_features.var() == 0).sum() / len(numeric_features.columns) if len(numeric_features.columns) > 0 else 0
        
        quality_score = 1.0 - missing_ratio - zero_variance_ratio
        return max(0.0, min(1.0, quality_score))
    
    def _create_empty_feature_set(self) -> FeatureSet:
        """Create empty feature set when no data available"""
        return FeatureSet(
            features=pd.DataFrame(),
            metadata={},
            target_column=None,
            feature_categories={category: [] for category in FeatureCategory},
            scaling_info={},
            selection_info={},
            creation_timestamp=datetime.now(),
            total_features=0,
            selected_features=0,
            data_quality_score=0.0
        )
    
    def scale_features(self, feature_set: FeatureSet, method: ScalingMethod = ScalingMethod.STANDARD, 
                      fit_scaler: bool = True) -> FeatureSet:
        """Scale features using specified method"""
        if not SKLEARN_AVAILABLE or feature_set.features.empty:
            return feature_set
        
        numeric_features = feature_set.features.select_dtypes(include=[np.number])
        
        if method == ScalingMethod.STANDARD:
            scaler = StandardScaler()
        elif method == ScalingMethod.MINMAX:
            scaler = MinMaxScaler()
        elif method == ScalingMethod.ROBUST:
            scaler = RobustScaler()
        else:
            return feature_set  # No scaling
        
        if fit_scaler:
            scaled_features = scaler.fit_transform(numeric_features.fillna(0))
            self.scalers[method.value] = scaler
        else:
            if method.value in self.scalers:
                scaled_features = self.scalers[method.value].transform(numeric_features.fillna(0))
            else:
                return feature_set  # No fitted scaler available
        
        # Update feature set
        scaled_df = feature_set.features.copy()
        scaled_df[numeric_features.columns] = scaled_features
        
        feature_set.features = scaled_df
        feature_set.scaling_info = {'method': method.value, 'fitted': fit_scaler}
        
        return feature_set
    
    def select_features(self, feature_set: FeatureSet, target: np.ndarray, 
                       method: FeatureSelectionMethod = FeatureSelectionMethod.F_REGRESSION,
                       n_features: int = 50) -> FeatureSet:
        """Select most important features"""
        if not SKLEARN_AVAILABLE or feature_set.features.empty or len(target) == 0:
            return feature_set
        
        numeric_features = feature_set.features.select_dtypes(include=[np.number]).fillna(0)
        
        if len(numeric_features.columns) <= n_features:
            return feature_set  # Already have fewer features than requested
        
        # Align target with features
        min_length = min(len(numeric_features), len(target))
        X = numeric_features.iloc[-min_length:].values
        y = target[-min_length:]
        
        try:
            if method == FeatureSelectionMethod.F_REGRESSION:
                selector = SelectKBest(score_func=f_regression, k=min(n_features, X.shape[1]))
            elif method == FeatureSelectionMethod.MUTUAL_INFO:
                selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, X.shape[1]))
            else:
                return feature_set  # Unsupported method
            
            X_selected = selector.fit_transform(X, y)
            selected_features = numeric_features.columns[selector.get_support()].tolist()
            
            # Update feature set
            selected_df = feature_set.features[selected_features].copy()
            feature_set.features = selected_df
            feature_set.selected_features = len(selected_features)
            feature_set.selection_info = {'method': method.value, 'n_selected': len(selected_features)}
            
            # Update metadata with importance scores
            if hasattr(selector, 'scores_'):
                for i, feature_name in enumerate(selected_features):
                    if feature_name in feature_set.metadata:
                        feature_set.metadata[feature_name].importance_score = selector.scores_[selector.get_support()][i]
            
            self.feature_selectors[method.value] = selector
            
        except Exception as e:
            print(f"Feature selection failed: {e}")
        
        return feature_set

# Example usage
if __name__ == "__main__":
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Sample stock data
    sample_stock_data = {
        'symbol': 'AAPL',
        'historical_prices': [140 + i + np.random.normal(0, 2) for i in range(100)],
        'volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(100)],
        'market_cap': 2500000000000,
        'revenue': 365000000000,
        'net_income': 95000000000,
        'total_assets': 350000000000,
        'shareholders_equity': 65000000000,
        'total_debt': 120000000000,
        'free_cash_flow': 80000000000,
        'revenue_growth': 0.08,
        'earnings_growth': 0.12
    }
    
    sample_market_data = {
        'market_index_prices': [4000 + i * 2 + np.random.normal(0, 10) for i in range(100)],
        'sector_index_prices': [1500 + i + np.random.normal(0, 5) for i in range(100)]
    }
    
    # Engineer features
    feature_set = feature_engineer.engineer_features(sample_stock_data, sample_market_data)
    
    print("=== Advanced Feature Engineering Results ===")
    print(f"Total Features: {feature_set.total_features}")
    print(f"Data Quality Score: {feature_set.data_quality_score:.3f}")
    print(f"Feature Categories:")
    for category, features in feature_set.feature_categories.items():
        if features:
            print(f"  {category.value}: {len(features)} features")
    
    print("\n=== Sample Features ===")
    print(feature_set.features.head())
    
    print("\n=== Feature Statistics ===")
    print(feature_set.features.describe())