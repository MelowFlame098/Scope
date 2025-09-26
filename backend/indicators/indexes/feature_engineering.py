"""Advanced Feature Engineering for Indexes Analysis

This module provides comprehensive feature engineering capabilities for indexes data,
including technical indicators, statistical features, and advanced transformations.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import math
from scipy import stats
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA, FastICA
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Using basic feature engineering.")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using custom technical indicators.")

class FeatureType(Enum):
    """Types of features available"""
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    FUNDAMENTAL = "fundamental"
    MACROECONOMIC = "macroeconomic"
    SENTIMENT = "sentiment"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    TREND = "trend"
    CYCLICAL = "cyclical"

@dataclass
class FeatureSet:
    """Container for engineered features"""
    features: Dict[str, float]
    feature_names: List[str]
    feature_types: Dict[str, FeatureType]
    feature_importance: Optional[Dict[str, float]] = None
    timestamp: datetime = None

class TechnicalIndicators:
    """Technical indicators calculator"""
    
    @staticmethod
    def sma(prices: np.ndarray, window: int) -> float:
        """Simple Moving Average"""
        if len(prices) < window:
            return np.mean(prices)
        return np.mean(prices[-window:])
    
    @staticmethod
    def ema(prices: np.ndarray, window: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < window:
            return np.mean(prices)
        
        alpha = 2.0 / (window + 1)
        ema_val = prices[0]
        
        for price in prices[1:]:
            ema_val = alpha * price + (1 - alpha) * ema_val
        
        return ema_val
    
    @staticmethod
    def rsi(prices: np.ndarray, window: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < window + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-window:])
        avg_loss = np.mean(losses[-window:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, window: int = 20, num_std: float = 2.0) -> Tuple[float, float, float]:
        """Bollinger Bands (upper, middle, lower)"""
        if len(prices) < window:
            mean_price = np.mean(prices)
            std_price = np.std(prices)
        else:
            mean_price = np.mean(prices[-window:])
            std_price = np.std(prices[-window:])
        
        upper_band = mean_price + (num_std * std_price)
        lower_band = mean_price - (num_std * std_price)
        
        return upper_band, mean_price, lower_band
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """MACD (MACD line, Signal line, Histogram)"""
        if len(prices) < slow:
            return 0.0, 0.0, 0.0
        
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        macd_line = ema_fast - ema_slow
        
        # Simplified signal line calculation
        signal_line = macd_line * 0.8  # Approximation
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> Tuple[float, float]:
        """Stochastic Oscillator (%K, %D)"""
        if len(close) < window:
            return 50.0, 50.0
        
        highest_high = np.max(high[-window:])
        lowest_low = np.min(low[-window:])
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
        
        # Simplified %D calculation
        d_percent = k_percent * 0.9  # Approximation
        
        return k_percent, d_percent
    
    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int = 14) -> float:
        """Average True Range"""
        if len(close) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < window:
            return np.mean(true_ranges)
        
        return np.mean(true_ranges[-window:])

class StatisticalFeatures:
    """Statistical feature calculator"""
    
    @staticmethod
    def calculate_moments(data: np.ndarray) -> Dict[str, float]:
        """Calculate statistical moments"""
        if len(data) == 0:
            return {'mean': 0, 'std': 0, 'skewness': 0, 'kurtosis': 0}
        
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'skewness': stats.skew(data),
            'kurtosis': stats.kurtosis(data)
        }
    
    @staticmethod
    def calculate_percentiles(data: np.ndarray) -> Dict[str, float]:
        """Calculate percentiles"""
        if len(data) == 0:
            return {f'p{p}': 0 for p in [5, 25, 50, 75, 95]}
        
        percentiles = [5, 25, 50, 75, 95]
        return {f'p{p}': np.percentile(data, p) for p in percentiles}
    
    @staticmethod
    def calculate_autocorrelation(data: np.ndarray, max_lag: int = 10) -> Dict[str, float]:
        """Calculate autocorrelation at different lags"""
        if len(data) < max_lag + 1:
            return {f'autocorr_lag_{i}': 0 for i in range(1, max_lag + 1)}
        
        autocorrs = {}
        for lag in range(1, max_lag + 1):
            if len(data) > lag:
                corr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
                autocorrs[f'autocorr_lag_{lag}'] = corr if not np.isnan(corr) else 0
            else:
                autocorrs[f'autocorr_lag_{lag}'] = 0
        
        return autocorrs
    
    @staticmethod
    def calculate_rolling_statistics(data: np.ndarray, windows: List[int]) -> Dict[str, float]:
        """Calculate rolling statistics for different windows"""
        rolling_stats = {}
        
        for window in windows:
            if len(data) >= window:
                window_data = data[-window:]
                rolling_stats[f'rolling_mean_{window}'] = np.mean(window_data)
                rolling_stats[f'rolling_std_{window}'] = np.std(window_data)
                rolling_stats[f'rolling_min_{window}'] = np.min(window_data)
                rolling_stats[f'rolling_max_{window}'] = np.max(window_data)
            else:
                rolling_stats[f'rolling_mean_{window}'] = np.mean(data) if len(data) > 0 else 0
                rolling_stats[f'rolling_std_{window}'] = np.std(data) if len(data) > 0 else 0
                rolling_stats[f'rolling_min_{window}'] = np.min(data) if len(data) > 0 else 0
                rolling_stats[f'rolling_max_{window}'] = np.max(data) if len(data) > 0 else 0
        
        return rolling_stats

class VolatilityFeatures:
    """Volatility-based feature calculator"""
    
    @staticmethod
    def realized_volatility(returns: np.ndarray, window: int = 20) -> float:
        """Calculate realized volatility"""
        if len(returns) < window:
            return np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        return np.std(returns[-window:]) * np.sqrt(252)
    
    @staticmethod
    def garch_volatility(returns: np.ndarray) -> float:
        """Simplified GARCH volatility estimate"""
        if len(returns) < 10:
            return np.std(returns) if len(returns) > 0 else 0
        
        # Simplified GARCH(1,1) approximation
        alpha = 0.1
        beta = 0.85
        omega = 0.05
        
        variance = np.var(returns)
        for ret in returns[-10:]:
            variance = omega + alpha * ret**2 + beta * variance
        
        return np.sqrt(variance)
    
    @staticmethod
    def volatility_clustering(returns: np.ndarray, window: int = 20) -> float:
        """Measure volatility clustering"""
        if len(returns) < window * 2:
            return 0.0
        
        # Calculate rolling volatilities
        rolling_vols = []
        for i in range(window, len(returns)):
            vol = np.std(returns[i-window:i])
            rolling_vols.append(vol)
        
        if len(rolling_vols) < 2:
            return 0.0
        
        # Measure autocorrelation in volatilities
        return np.corrcoef(rolling_vols[:-1], rolling_vols[1:])[0, 1] if len(rolling_vols) > 1 else 0

class CyclicalFeatures:
    """Cyclical and seasonal feature calculator"""
    
    @staticmethod
    def fourier_features(data: np.ndarray, n_components: int = 5) -> Dict[str, float]:
        """Extract Fourier transform features"""
        if len(data) < n_components * 2:
            return {f'fourier_{i}': 0 for i in range(n_components)}
        
        # Apply FFT
        fft_vals = fft(data)
        freqs = fftfreq(len(data))
        
        # Get dominant frequencies
        magnitudes = np.abs(fft_vals)
        dominant_indices = np.argsort(magnitudes)[-n_components:]
        
        fourier_features = {}
        for i, idx in enumerate(dominant_indices):
            fourier_features[f'fourier_magnitude_{i}'] = magnitudes[idx]
            fourier_features[f'fourier_frequency_{i}'] = freqs[idx]
        
        return fourier_features
    
    @staticmethod
    def seasonal_decomposition(data: np.ndarray, period: int = 252) -> Dict[str, float]:
        """Simple seasonal decomposition"""
        if len(data) < period:
            return {'trend': np.mean(data) if len(data) > 0 else 0, 'seasonal': 0, 'residual': 0}
        
        # Simple trend (moving average)
        trend = np.mean(data)
        
        # Simple seasonal component
        seasonal_pattern = []
        for i in range(min(period, len(data))):
            seasonal_indices = list(range(i, len(data), period))
            if seasonal_indices:
                seasonal_pattern.append(np.mean([data[j] for j in seasonal_indices]))
        
        seasonal = np.std(seasonal_pattern) if seasonal_pattern else 0
        
        # Residual
        residual = np.std(data - trend)
        
        return {'trend': trend, 'seasonal': seasonal, 'residual': residual}

class AdvancedFeatureEngineering:
    """Advanced feature engineering for indexes data"""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
        self.statistical_features = StatisticalFeatures()
        self.volatility_features = VolatilityFeatures()
        self.cyclical_features = CyclicalFeatures()
        
        # Initialize scalers if available
        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.robust_scaler = RobustScaler()
            self.minmax_scaler = MinMaxScaler()
        
        self.feature_cache = {}
    
    def engineer_features(self, indexes_data: Dict[str, Any], 
                         macro_data: Dict[str, Any],
                         feature_types: List[FeatureType] = None) -> FeatureSet:
        """Engineer comprehensive feature set"""
        if feature_types is None:
            feature_types = list(FeatureType)
        
        features = {}
        feature_names = []
        feature_type_map = {}
        
        # Extract price data
        prices = np.array(indexes_data.get('historical_levels', []))
        if len(prices) == 0:
            prices = np.array([indexes_data.get('current_level', 100)])
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1] if len(prices) > 1 else np.array([0])
        
        # Technical features
        if FeatureType.TECHNICAL in feature_types:
            tech_features = self._extract_technical_features(prices, returns)
            features.update(tech_features)
            for name in tech_features.keys():
                feature_names.append(name)
                feature_type_map[name] = FeatureType.TECHNICAL
        
        # Statistical features
        if FeatureType.STATISTICAL in feature_types:
            stat_features = self._extract_statistical_features(prices, returns)
            features.update(stat_features)
            for name in stat_features.keys():
                feature_names.append(name)
                feature_type_map[name] = FeatureType.STATISTICAL
        
        # Fundamental features
        if FeatureType.FUNDAMENTAL in feature_types:
            fund_features = self._extract_fundamental_features(indexes_data)
            features.update(fund_features)
            for name in fund_features.keys():
                feature_names.append(name)
                feature_type_map[name] = FeatureType.FUNDAMENTAL
        
        # Macroeconomic features
        if FeatureType.MACROECONOMIC in feature_types:
            macro_features = self._extract_macroeconomic_features(macro_data)
            features.update(macro_features)
            for name in macro_features.keys():
                feature_names.append(name)
                feature_type_map[name] = FeatureType.MACROECONOMIC
        
        # Volatility features
        if FeatureType.VOLATILITY in feature_types:
            vol_features = self._extract_volatility_features(returns)
            features.update(vol_features)
            for name in vol_features.keys():
                feature_names.append(name)
                feature_type_map[name] = FeatureType.VOLATILITY
        
        # Momentum features
        if FeatureType.MOMENTUM in feature_types:
            mom_features = self._extract_momentum_features(prices, returns)
            features.update(mom_features)
            for name in mom_features.keys():
                feature_names.append(name)
                feature_type_map[name] = FeatureType.MOMENTUM
        
        # Trend features
        if FeatureType.TREND in feature_types:
            trend_features = self._extract_trend_features(prices)
            features.update(trend_features)
            for name in trend_features.keys():
                feature_names.append(name)
                feature_type_map[name] = FeatureType.TREND
        
        # Cyclical features
        if FeatureType.CYCLICAL in feature_types:
            cyclical_features = self._extract_cyclical_features(prices)
            features.update(cyclical_features)
            for name in cyclical_features.keys():
                feature_names.append(name)
                feature_type_map[name] = FeatureType.CYCLICAL
        
        return FeatureSet(
            features=features,
            feature_names=feature_names,
            feature_types=feature_type_map,
            timestamp=datetime.now()
        )
    
    def _extract_technical_features(self, prices: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Extract technical indicator features"""
        features = {}
        
        if len(prices) == 0:
            return features
        
        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'sma_{window}'] = self.technical_indicators.sma(prices, window)
            features[f'ema_{window}'] = self.technical_indicators.ema(prices, window)
        
        # Price ratios
        current_price = prices[-1]
        for window in [5, 10, 20]:
            sma = self.technical_indicators.sma(prices, window)
            if sma != 0:
                features[f'price_to_sma_{window}'] = current_price / sma
            else:
                features[f'price_to_sma_{window}'] = 1.0
        
        # RSI
        features['rsi'] = self.technical_indicators.rsi(prices)
        
        # Bollinger Bands
        if len(prices) >= 20:
            upper, middle, lower = self.technical_indicators.bollinger_bands(prices)
            features['bb_upper'] = upper
            features['bb_middle'] = middle
            features['bb_lower'] = lower
            features['bb_width'] = (upper - lower) / middle if middle != 0 else 0
            features['bb_position'] = (current_price - lower) / (upper - lower) if upper != lower else 0.5
        
        # MACD
        if len(prices) >= 26:
            macd, signal, histogram = self.technical_indicators.macd(prices)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_histogram'] = histogram
        
        return features
    
    def _extract_statistical_features(self, prices: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Extract statistical features"""
        features = {}
        
        # Price moments
        price_moments = self.statistical_features.calculate_moments(prices)
        for key, value in price_moments.items():
            features[f'price_{key}'] = value
        
        # Return moments
        if len(returns) > 0:
            return_moments = self.statistical_features.calculate_moments(returns)
            for key, value in return_moments.items():
                features[f'return_{key}'] = value
        
        # Percentiles
        price_percentiles = self.statistical_features.calculate_percentiles(prices)
        features.update({f'price_{k}': v for k, v in price_percentiles.items()})
        
        # Rolling statistics
        rolling_stats = self.statistical_features.calculate_rolling_statistics(prices, [5, 10, 20])
        features.update(rolling_stats)
        
        # Autocorrelation
        if len(returns) > 10:
            autocorrs = self.statistical_features.calculate_autocorrelation(returns, 5)
            features.update(autocorrs)
        
        return features
    
    def _extract_fundamental_features(self, indexes_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract fundamental features"""
        features = {}
        
        # Basic fundamental metrics
        features['pe_ratio'] = indexes_data.get('pe_ratio', 20.0)
        features['pb_ratio'] = indexes_data.get('pb_ratio', 3.0)
        features['dividend_yield'] = indexes_data.get('dividend_yield', 0.02)
        features['market_cap_log'] = math.log(indexes_data.get('market_cap', 1e9))
        features['beta'] = indexes_data.get('beta', 1.0)
        features['volatility'] = indexes_data.get('volatility', 0.2)
        features['volume_log'] = math.log(indexes_data.get('volume', 1e6))
        features['constituent_count'] = indexes_data.get('constituent_count', 100)
        
        # Sector weights (top 5)
        sector_weights = indexes_data.get('sector_weights', {})
        sorted_sectors = sorted(sector_weights.items(), key=lambda x: x[1], reverse=True)
        for i, (sector, weight) in enumerate(sorted_sectors[:5]):
            features[f'sector_weight_{i+1}'] = weight
        
        # Fill missing sector weights
        for i in range(len(sorted_sectors), 5):
            features[f'sector_weight_{i+1}'] = 0.0
        
        # Derived metrics
        if features['pe_ratio'] != 0:
            features['earnings_yield'] = 1.0 / features['pe_ratio']
        else:
            features['earnings_yield'] = 0.0
        
        if features['pb_ratio'] != 0:
            features['book_yield'] = 1.0 / features['pb_ratio']
        else:
            features['book_yield'] = 0.0
        
        return features
    
    def _extract_macroeconomic_features(self, macro_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract macroeconomic features"""
        features = {}
        
        # Basic macro indicators
        features['gdp_growth'] = macro_data.get('gdp_growth', 2.0)
        features['inflation_rate'] = macro_data.get('inflation_rate', 3.0)
        features['interest_rates'] = macro_data.get('interest_rates', 5.0)
        features['unemployment_rate'] = macro_data.get('unemployment_rate', 4.0)
        features['industrial_production'] = macro_data.get('industrial_production', 1.0)
        features['consumer_confidence'] = macro_data.get('consumer_confidence', 100.0)
        features['oil_prices'] = macro_data.get('oil_prices', 80.0)
        features['exchange_rates'] = macro_data.get('exchange_rates', 1.0)
        features['vix_index'] = macro_data.get('vix_index', 20.0)
        
        # Derived macro features
        features['real_interest_rate'] = features['interest_rates'] - features['inflation_rate']
        features['yield_curve_slope'] = features['interest_rates'] - 2.0  # Simplified
        features['economic_momentum'] = (features['gdp_growth'] + features['industrial_production']) / 2
        features['inflation_pressure'] = features['inflation_rate'] - 2.0  # Target inflation
        features['labor_market_strength'] = 10.0 - features['unemployment_rate']  # Inverted unemployment
        
        # Risk indicators
        features['macro_risk_score'] = (
            (features['vix_index'] - 20) / 20 +
            (features['inflation_rate'] - 2) / 2 +
            (features['unemployment_rate'] - 4) / 4
        ) / 3
        
        return features
    
    def _extract_volatility_features(self, returns: np.ndarray) -> Dict[str, float]:
        """Extract volatility-based features"""
        features = {}
        
        if len(returns) == 0:
            return features
        
        # Realized volatility
        for window in [5, 10, 20, 60]:
            features[f'realized_vol_{window}'] = self.volatility_features.realized_volatility(returns, window)
        
        # GARCH volatility
        features['garch_volatility'] = self.volatility_features.garch_volatility(returns)
        
        # Volatility clustering
        features['vol_clustering'] = self.volatility_features.volatility_clustering(returns)
        
        # Volatility ratios
        vol_5 = self.volatility_features.realized_volatility(returns, 5)
        vol_20 = self.volatility_features.realized_volatility(returns, 20)
        if vol_20 != 0:
            features['vol_ratio_5_20'] = vol_5 / vol_20
        else:
            features['vol_ratio_5_20'] = 1.0
        
        # Volatility percentiles
        if len(returns) >= 20:
            rolling_vols = []
            for i in range(10, len(returns)):
                vol = np.std(returns[i-10:i]) * np.sqrt(252)
                rolling_vols.append(vol)
            
            if rolling_vols:
                current_vol = rolling_vols[-1]
                features['vol_percentile'] = stats.percentileofscore(rolling_vols, current_vol) / 100
            else:
                features['vol_percentile'] = 0.5
        else:
            features['vol_percentile'] = 0.5
        
        return features
    
    def _extract_momentum_features(self, prices: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Extract momentum features"""
        features = {}
        
        if len(prices) == 0:
            return features
        
        current_price = prices[-1]
        
        # Price momentum (returns over different periods)
        for window in [1, 5, 10, 20, 60]:
            if len(prices) > window:
                past_price = prices[-window-1]
                if past_price != 0:
                    features[f'momentum_{window}d'] = (current_price - past_price) / past_price
                else:
                    features[f'momentum_{window}d'] = 0.0
            else:
                features[f'momentum_{window}d'] = 0.0
        
        # Acceleration (second derivative)
        if len(returns) >= 2:
            features['acceleration'] = returns[-1] - returns[-2]
        else:
            features['acceleration'] = 0.0
        
        # Momentum strength
        if len(returns) >= 10:
            positive_returns = np.sum(returns[-10:] > 0)
            features['momentum_strength'] = positive_returns / 10
        else:
            features['momentum_strength'] = 0.5
        
        # Rate of change
        for window in [5, 10, 20]:
            if len(prices) > window:
                roc = (current_price - prices[-window-1]) / prices[-window-1] * 100
                features[f'roc_{window}'] = roc
            else:
                features[f'roc_{window}'] = 0.0
        
        return features
    
    def _extract_trend_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Extract trend features"""
        features = {}
        
        if len(prices) < 10:
            return features
        
        # Linear trend
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
        
        features['trend_slope'] = slope
        features['trend_r_squared'] = r_value ** 2
        features['trend_p_value'] = p_value
        
        # Trend strength
        features['trend_strength'] = abs(slope) * features['trend_r_squared']
        
        # Support and resistance levels
        peaks, _ = find_peaks(prices, height=np.percentile(prices, 80))
        troughs, _ = find_peaks(-prices, height=-np.percentile(prices, 20))
        
        features['num_peaks'] = len(peaks)
        features['num_troughs'] = len(troughs)
        
        if len(peaks) > 0:
            features['resistance_level'] = np.mean(prices[peaks])
        else:
            features['resistance_level'] = np.max(prices)
        
        if len(troughs) > 0:
            features['support_level'] = np.mean(prices[troughs])
        else:
            features['support_level'] = np.min(prices)
        
        # Distance from support/resistance
        current_price = prices[-1]
        features['distance_from_resistance'] = (features['resistance_level'] - current_price) / current_price
        features['distance_from_support'] = (current_price - features['support_level']) / current_price
        
        return features
    
    def _extract_cyclical_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Extract cyclical features"""
        features = {}
        
        if len(prices) < 20:
            return features
        
        # Fourier features
        fourier_features = self.cyclical_features.fourier_features(prices, 3)
        features.update(fourier_features)
        
        # Seasonal decomposition
        seasonal_features = self.cyclical_features.seasonal_decomposition(prices)
        features.update(seasonal_features)
        
        # Cycle detection (simplified)
        detrended = prices - np.mean(prices)
        zero_crossings = np.where(np.diff(np.signbit(detrended)))[0]
        
        if len(zero_crossings) > 1:
            cycle_lengths = np.diff(zero_crossings)
            features['avg_cycle_length'] = np.mean(cycle_lengths)
            features['cycle_regularity'] = 1.0 / (1.0 + np.std(cycle_lengths))
        else:
            features['avg_cycle_length'] = len(prices)
            features['cycle_regularity'] = 0.0
        
        return features
    
    def select_features(self, feature_set: FeatureSet, method: str = 'correlation', 
                       n_features: int = 50) -> FeatureSet:
        """Select most important features"""
        if not SKLEARN_AVAILABLE or len(feature_set.features) <= n_features:
            return feature_set
        
        # Convert features to array
        feature_values = np.array(list(feature_set.features.values())).reshape(1, -1)
        feature_names = list(feature_set.features.keys())
        
        # For single sample, we can't use statistical methods
        # Instead, use domain knowledge to select features
        important_patterns = [
            'momentum', 'volatility', 'trend', 'rsi', 'macd', 'bb_', 
            'pe_ratio', 'pb_ratio', 'gdp_growth', 'inflation_rate',
            'vix_index', 'realized_vol', 'sma_', 'ema_'
        ]
        
        selected_features = {}
        selected_names = []
        selected_types = {}
        
        # First, select features matching important patterns
        for name in feature_names:
            if any(pattern in name for pattern in important_patterns):
                selected_features[name] = feature_set.features[name]
                selected_names.append(name)
                selected_types[name] = feature_set.feature_types.get(name, FeatureType.STATISTICAL)
                
                if len(selected_features) >= n_features:
                    break
        
        # Fill remaining slots with other features
        for name in feature_names:
            if name not in selected_features:
                selected_features[name] = feature_set.features[name]
                selected_names.append(name)
                selected_types[name] = feature_set.feature_types.get(name, FeatureType.STATISTICAL)
                
                if len(selected_features) >= n_features:
                    break
        
        return FeatureSet(
            features=selected_features,
            feature_names=selected_names,
            feature_types=selected_types,
            timestamp=feature_set.timestamp
        )
    
    def normalize_features(self, feature_set: FeatureSet, method: str = 'standard') -> FeatureSet:
        """Normalize features"""
        if not SKLEARN_AVAILABLE:
            return feature_set
        
        # Convert to array
        feature_values = np.array(list(feature_set.features.values())).reshape(1, -1)
        
        # Select scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return feature_set
        
        # For single sample, we can't fit the scaler properly
        # Instead, apply simple normalization
        normalized_values = feature_values.flatten()
        
        # Z-score normalization with robust statistics
        median_val = np.median(normalized_values)
        mad_val = np.median(np.abs(normalized_values - median_val))
        
        if mad_val != 0:
            normalized_values = (normalized_values - median_val) / (1.4826 * mad_val)
        
        # Create normalized feature set
        normalized_features = {}
        for i, name in enumerate(feature_set.feature_names):
            normalized_features[name] = normalized_values[i]
        
        return FeatureSet(
            features=normalized_features,
            feature_names=feature_set.feature_names,
            feature_types=feature_set.feature_types,
            timestamp=feature_set.timestamp
        )

# Example usage
if __name__ == "__main__":
    # Create sample data
    indexes_data = {
        'symbol': 'SPY',
        'name': 'SPDR S&P 500 ETF',
        'current_level': 450.0,
        'historical_levels': [440.0, 445.0, 448.0, 450.0, 452.0, 449.0, 451.0],
        'dividend_yield': 0.015,
        'pe_ratio': 22.5,
        'pb_ratio': 3.2,
        'market_cap': 400000000000,
        'volatility': 0.18,
        'beta': 1.0,
        'sector_weights': {'Technology': 0.28, 'Healthcare': 0.13, 'Financials': 0.11},
        'constituent_count': 500,
        'volume': 50000000
    }
    
    macro_data = {
        'gdp_growth': 2.1,
        'inflation_rate': 3.2,
        'interest_rates': 5.25,
        'unemployment_rate': 3.7,
        'industrial_production': 1.8,
        'consumer_confidence': 102.5,
        'oil_prices': 85.0,
        'exchange_rates': 1.08,
        'vix_index': 18.5
    }
    
    # Initialize feature engineering
    feature_engineer = AdvancedFeatureEngineering()
    
    # Engineer features
    feature_set = feature_engineer.engineer_features(indexes_data, macro_data)
    
    print(f"\n=== Feature Engineering Results ===")
    print(f"Total features: {len(feature_set.features)}")
    print(f"Feature types: {set(feature_set.feature_types.values())}")
    
    # Show sample features
    print(f"\n=== Sample Features ===")
    for i, (name, value) in enumerate(list(feature_set.features.items())[:10]):
        feature_type = feature_set.feature_types.get(name, 'unknown')
        print(f"{name}: {value:.4f} ({feature_type.value})")
    
    # Select important features
    selected_features = feature_engineer.select_features(feature_set, n_features=20)
    print(f"\n=== Selected Features ({len(selected_features.features)}) ===")
    for name, value in selected_features.features.items():
        print(f"{name}: {value:.4f}")