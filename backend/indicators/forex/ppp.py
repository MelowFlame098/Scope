"""Purchasing Power Parity (PPP) Analysis

Purchasing Power Parity is a fundamental economic theory that suggests exchange rates
should adjust to equalize the price of identical goods and services in different countries.
This indicator helps identify currency over/undervaluation based on relative price levels.

Author: Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
import warnings

# Advanced econometric and ML imports
try:
    import xgboost as xgb
    from statsmodels.tsa.stattools import adfuller, kpss, coint
    from statsmodels.stats.diagnostic import het_arch
    from arch import arch_model
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import joblib
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class AssetType(Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    INDEX = "index"
    CROSS_ASSET = "cross_asset"


class IndicatorCategory(Enum):
    FUNDAMENTAL = "fundamental"
    ECONOMIC = "economic"
    VALUATION = "valuation"


@dataclass
class RegimeAnalysis:
    """PPP regime switching analysis results."""
    current_regime: int
    regime_probabilities: np.ndarray
    regime_description: str
    transition_matrix: np.ndarray
    regime_persistence: Dict[str, float]
    expected_duration: Dict[str, float]

@dataclass
class VolatilityAnalysis:
    """PPP volatility clustering analysis."""
    current_volatility: float
    volatility_regime: str
    garch_forecast: np.ndarray
    arch_test_pvalue: float
    volatility_clustering: bool
    conditional_volatility: pd.Series

@dataclass
class MachineLearningPrediction:
    """ML-based PPP predictions."""
    predictions: Dict[str, np.ndarray]
    feature_importance: Dict[str, np.ndarray]
    model_scores: Dict[str, float]
    ensemble_prediction: np.ndarray
    prediction_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]
    cross_validation_scores: Dict[str, np.ndarray]

@dataclass
class EconometricAnalysis:
    """Advanced econometric analysis."""
    unit_root_tests: Dict[str, Dict[str, float]]
    cointegration_tests: Dict[str, Dict[str, float]]
    structural_breaks: List[int]
    half_life_estimates: Dict[str, float]
    error_correction_model: Dict[str, Any]

@dataclass
class RiskMetrics:
    """Risk assessment metrics."""
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    skewness: float
    kurtosis: float

@dataclass
class PPPResult:
    """Result of PPP calculation"""
    name: str
    ppp_rate: float
    current_rate: float
    ppp_deviation: float
    real_exchange_rate: float
    big_mac_ppp: float
    big_mac_deviation: float
    reversion_signal: str
    fair_value_range: Tuple[float, float]
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]
    # Enhanced analysis components
    regime_analysis: Optional[RegimeAnalysis] = None
    volatility_analysis: Optional[VolatilityAnalysis] = None
    ml_predictions: Optional[MachineLearningPrediction] = None
    econometric_analysis: Optional[EconometricAnalysis] = None
    risk_metrics: Optional[RiskMetrics] = None
    model_diagnostics: Optional[Dict[str, Any]] = None
    backtesting_results: Optional[Dict[str, Any]] = None
    sensitivity_analysis: Optional[Dict[str, Any]] = None


class PPPIndicator:
    """Purchasing Power Parity Calculator with Advanced Analysis"""
    
    def __init__(self, base_country: str = "US", quote_country: str = "EU",
                 reversion_window: int = 252, confidence_bands: float = 2.0,
                 enable_ml: bool = True, enable_regime_switching: bool = True,
                 enable_volatility_modeling: bool = True):
        """
        Initialize PPP calculator
        
        Args:
            base_country: Base currency country code (default: "US")
            quote_country: Quote currency country code (default: "EU")
            reversion_window: Window for mean reversion analysis (default: 252 days)
            confidence_bands: Standard deviations for confidence bands (default: 2.0)
            enable_ml: Enable machine learning predictions (default: True)
            enable_regime_switching: Enable regime switching analysis (default: True)
            enable_volatility_modeling: Enable volatility modeling (default: True)
        """
        self.base_country = base_country
        self.quote_country = quote_country
        self.reversion_window = reversion_window
        self.confidence_bands = confidence_bands
        self.enable_ml = enable_ml
        self.enable_regime_switching = enable_regime_switching
        self.enable_volatility_modeling = enable_volatility_modeling
        self.logger = logging.getLogger(__name__)
        
        # Advanced analysis thresholds
        self.thresholds = {
            'volatility_threshold': 0.02,
            'regime_confidence': 0.7,
            'cointegration_pvalue': 0.05,
            'unit_root_pvalue': 0.05
        }
        
        # Initialize ML models and scalers
        if self.enable_ml:
            self.ml_models = self._initialize_ml_models()
            self.scaler = StandardScaler()
        
        # Country-specific data (simplified - in practice would use real economic data)
        self.country_data = self._initialize_country_data()
    
    def calculate(self, data: pd.DataFrame, inflation_data: Optional[Dict] = None,
                 price_data: Optional[Dict] = None, custom_base: Optional[str] = None,
                 custom_quote: Optional[str] = None, asset_type: AssetType = AssetType.FOREX) -> PPPResult:
        """
        Calculate PPP analysis for given exchange rate data
        
        Args:
            data: Exchange rate data DataFrame with 'close' column
            inflation_data: Dictionary containing inflation rates for both countries
            price_data: Dictionary containing price level data
            custom_base: Override base country
            custom_quote: Override quote country
            asset_type: Type of asset being analyzed
            
        Returns:
            PPPResult containing PPP valuation analysis
        """
        try:
            # Use custom countries if provided
            base_country = custom_base or self.base_country
            quote_country = custom_quote or self.quote_country
            
            # Prepare economic data
            economic_data = self._prepare_economic_data(inflation_data, price_data, base_country, quote_country)
            
            # Calculate absolute PPP (price level based)
            absolute_ppp = self._calculate_absolute_ppp(data, economic_data)
            
            # Calculate relative PPP (inflation differential based)
            relative_ppp = self._calculate_relative_ppp(data, economic_data)
            
            # Calculate Big Mac PPP (simplified basket approach)
            big_mac_ppp, big_mac_deviation = self._calculate_big_mac_ppp(data, base_country, quote_country)
            
            # Real exchange rate analysis
            real_exchange_rate = self._calculate_real_exchange_rate(data, economic_data)
            
            # PPP deviation and reversion analysis
            ppp_deviation, reversion_signal, fair_value_range = self._analyze_ppp_deviation(
                data, relative_ppp, economic_data
            )
            
            # Generate trading signals
            signals = self._generate_signals(ppp_deviation, big_mac_deviation, reversion_signal, 
                                           real_exchange_rate, economic_data)
            
            # Create comprehensive time series
            values_df = self._create_time_series(data, relative_ppp, absolute_ppp, 
                                               real_exchange_rate, economic_data, 
                                               big_mac_ppp, base_country, quote_country)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(economic_data, len(data))
            
            current_rate = data['close'].iloc[-1]
            current_ppp_rate = relative_ppp.iloc[-1]
            current_deviation = ppp_deviation.iloc[-1]
            
            # Advanced analysis components
            regime_analysis = None
            volatility_analysis = None
            ml_predictions = None
            econometric_analysis = None
            risk_metrics = None
            model_diagnostics = None
            backtesting_results = None
            sensitivity_analysis = None
            
            # Perform advanced analysis if enabled
            if self.enable_regime_switching:
                regime_analysis = self._perform_regime_analysis(ppp_deviation)
            
            if self.enable_volatility_modeling:
                volatility_analysis = self._perform_volatility_analysis(ppp_deviation)
            
            if self.enable_ml and len(data) > 50:
                ml_predictions = self._perform_ml_predictions(data, economic_data, ppp_deviation)
            
            # Always perform econometric analysis
            econometric_analysis = self._perform_econometric_analysis(data, relative_ppp, ppp_deviation)
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(ppp_deviation)
            
            # Model diagnostics
            model_diagnostics = self._run_model_diagnostics(data, ppp_deviation)
            
            # Backtesting
            if len(data) > 100:
                backtesting_results = self._perform_backtesting(data, relative_ppp)
            
            # Sensitivity analysis
            sensitivity_analysis = self._perform_sensitivity_analysis(data, economic_data)
            
            return PPPResult(
                name="Purchasing Power Parity",
                ppp_rate=current_ppp_rate,
                current_rate=current_rate,
                ppp_deviation=current_deviation,
                real_exchange_rate=real_exchange_rate.iloc[-1],
                big_mac_ppp=big_mac_ppp,
                big_mac_deviation=big_mac_deviation.iloc[-1],
                reversion_signal=reversion_signal,
                fair_value_range=fair_value_range,
                values=values_df,
                metadata={
                    'base_country': base_country,
                    'quote_country': quote_country,
                    'base_inflation': economic_data['base_inflation'],
                    'quote_inflation': economic_data['quote_inflation'],
                    'inflation_differential': economic_data['inflation_differential'],
                    'price_level_ratio': economic_data.get('price_level_ratio', 1.0),
                    'productivity_differential': economic_data.get('productivity_differential', 0.0),
                    'balassa_samuelson_effect': self._calculate_balassa_samuelson(economic_data),
                    'ppp_half_life': self._estimate_ppp_half_life(ppp_deviation),
                    'volatility_analysis': self._analyze_ppp_volatility(ppp_deviation),
                    'regime_analysis': self._analyze_ppp_regimes(values_df),
                    'cointegration_test': self._test_ppp_cointegration(data['close'], relative_ppp),
                    'interpretation': self._get_interpretation(current_deviation, reversion_signal, big_mac_deviation.iloc[-1])
                },
                confidence=confidence,
                timestamp=datetime.now(),
                asset_type=asset_type,
                category=IndicatorCategory.FUNDAMENTAL,
                signals=signals,
                regime_analysis=regime_analysis,
                volatility_analysis=volatility_analysis,
                ml_predictions=ml_predictions,
                econometric_analysis=econometric_analysis,
                risk_metrics=risk_metrics,
                model_diagnostics=model_diagnostics,
                backtesting_results=backtesting_results,
                sensitivity_analysis=sensitivity_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating PPP: {e}")
            return self._empty_result(asset_type)
    
    def _initialize_country_data(self) -> Dict[str, Dict[str, Any]]:
        """Initialize country-specific economic data"""
        return {
            'US': {
                'inflation_rate': 0.025,
                'price_level': 100,
                'productivity_growth': 0.015,
                'big_mac_price': 5.50,
                'gdp_per_capita': 65000
            },
            'EU': {
                'inflation_rate': 0.020,
                'price_level': 95,
                'productivity_growth': 0.012,
                'big_mac_price': 4.50,
                'gdp_per_capita': 45000
            },
            'JP': {
                'inflation_rate': 0.005,
                'price_level': 85,
                'productivity_growth': 0.008,
                'big_mac_price': 390,  # In Yen
                'gdp_per_capita': 40000
            },
            'GB': {
                'inflation_rate': 0.030,
                'price_level': 105,
                'productivity_growth': 0.010,
                'big_mac_price': 4.20,
                'gdp_per_capita': 42000
            },
            'CA': {
                'inflation_rate': 0.022,
                'price_level': 92,
                'productivity_growth': 0.013,
                'big_mac_price': 6.80,  # In CAD
                'gdp_per_capita': 48000
            },
            'AU': {
                'inflation_rate': 0.028,
                'price_level': 98,
                'productivity_growth': 0.014,
                'big_mac_price': 6.40,  # In AUD
                'gdp_per_capita': 55000
            }
        }
    
    def _prepare_economic_data(self, inflation_data: Optional[Dict], price_data: Optional[Dict],
                              base_country: str, quote_country: str) -> Dict[str, Any]:
        """Prepare economic data for PPP calculations"""
        # Use provided data or defaults
        if inflation_data:
            base_inflation = inflation_data.get('base_inflation', self.country_data.get(base_country, {}).get('inflation_rate', 0.025))
            quote_inflation = inflation_data.get('quote_inflation', self.country_data.get(quote_country, {}).get('inflation_rate', 0.020))
        else:
            base_inflation = self.country_data.get(base_country, {}).get('inflation_rate', 0.025)
            quote_inflation = self.country_data.get(quote_country, {}).get('inflation_rate', 0.020)
        
        # Price level data
        if price_data:
            base_price_level = price_data.get('base_price_level', 100)
            quote_price_level = price_data.get('quote_price_level', 95)
        else:
            base_price_level = self.country_data.get(base_country, {}).get('price_level', 100)
            quote_price_level = self.country_data.get(quote_country, {}).get('price_level', 95)
        
        # Additional economic indicators
        base_productivity = self.country_data.get(base_country, {}).get('productivity_growth', 0.015)
        quote_productivity = self.country_data.get(quote_country, {}).get('productivity_growth', 0.012)
        
        return {
            'base_inflation': base_inflation,
            'quote_inflation': quote_inflation,
            'inflation_differential': base_inflation - quote_inflation,
            'base_price_level': base_price_level,
            'quote_price_level': quote_price_level,
            'price_level_ratio': base_price_level / quote_price_level,
            'base_productivity': base_productivity,
            'quote_productivity': quote_productivity,
            'productivity_differential': base_productivity - quote_productivity
        }
    
    def _calculate_absolute_ppp(self, data: pd.DataFrame, economic_data: Dict[str, Any]) -> pd.Series:
        """Calculate absolute PPP based on price levels"""
        # Absolute PPP: S = P_base / P_quote
        price_level_ratio = economic_data['price_level_ratio']
        
        # Adjust for initial exchange rate level
        initial_rate = data['close'].iloc[0]
        normalization_factor = initial_rate / price_level_ratio
        
        # Create constant absolute PPP series
        absolute_ppp = pd.Series(price_level_ratio * normalization_factor, index=data.index)
        
        return absolute_ppp
    
    def _calculate_relative_ppp(self, data: pd.DataFrame, economic_data: Dict[str, Any]) -> pd.Series:
        """Calculate relative PPP based on inflation differentials"""
        base_inflation = economic_data['base_inflation']
        quote_inflation = economic_data['quote_inflation']
        
        # Initial exchange rate
        initial_rate = data['close'].iloc[0]
        
        # Time periods in years
        time_periods = np.arange(len(data)) / 252
        
        # Relative PPP: S(t) = S(0) * (1 + π_quote)^t / (1 + π_base)^t
        relative_ppp = initial_rate * ((1 + quote_inflation) / (1 + base_inflation)) ** time_periods
        
        return pd.Series(relative_ppp, index=data.index)
    
    def _calculate_big_mac_ppp(self, data: pd.DataFrame, base_country: str, 
                              quote_country: str) -> Tuple[float, pd.Series]:
        """Calculate Big Mac PPP (simplified basket approach)"""
        base_big_mac = self.country_data.get(base_country, {}).get('big_mac_price', 5.50)
        quote_big_mac = self.country_data.get(quote_country, {}).get('big_mac_price', 4.50)
        
        # Big Mac PPP rate
        big_mac_ppp_rate = base_big_mac / quote_big_mac
        
        # Deviation from Big Mac PPP
        big_mac_deviation = (data['close'] - big_mac_ppp_rate) / big_mac_ppp_rate * 100
        
        return big_mac_ppp_rate, big_mac_deviation
    
    def _calculate_real_exchange_rate(self, data: pd.DataFrame, economic_data: Dict[str, Any]) -> pd.Series:
        """Calculate real exchange rate"""
        # Real exchange rate adjusts nominal rate for price level differences
        price_level_ratio = economic_data['price_level_ratio']
        
        # RER = Nominal Rate * (P_quote / P_base)
        real_exchange_rate = data['close'] / price_level_ratio
        
        return real_exchange_rate
    
    def _analyze_ppp_deviation(self, data: pd.DataFrame, ppp_rate: pd.Series, 
                              economic_data: Dict[str, Any]) -> Tuple[pd.Series, str, Tuple[float, float]]:
        """Analyze PPP deviation and mean reversion"""
        # PPP deviation percentage
        ppp_deviation = (data['close'] - ppp_rate) / ppp_rate * 100
        
        # Rolling statistics for mean reversion
        rolling_mean = ppp_deviation.rolling(self.reversion_window).mean()
        rolling_std = ppp_deviation.rolling(self.reversion_window).std()
        
        # Z-score for current deviation
        current_zscore = (ppp_deviation.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]
        
        # Reversion signal
        if current_zscore > self.confidence_bands:
            reversion_signal = "STRONG_OVERVALUED"
        elif current_zscore > 1.0:
            reversion_signal = "OVERVALUED"
        elif current_zscore < -self.confidence_bands:
            reversion_signal = "STRONG_UNDERVALUED"
        elif current_zscore < -1.0:
            reversion_signal = "UNDERVALUED"
        else:
            reversion_signal = "FAIR_VALUE"
        
        # Fair value range (confidence bands)
        current_ppp = ppp_rate.iloc[-1]
        current_std = rolling_std.iloc[-1]
        
        lower_bound = current_ppp * (1 - current_std / 100)
        upper_bound = current_ppp * (1 + current_std / 100)
        fair_value_range = (lower_bound, upper_bound)
        
        return ppp_deviation, reversion_signal, fair_value_range
    
    def _calculate_balassa_samuelson(self, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Balassa-Samuelson effect"""
        productivity_diff = economic_data['productivity_differential']
        
        # Balassa-Samuelson suggests that countries with higher productivity growth
        # should experience real exchange rate appreciation
        bs_effect = productivity_diff * 0.5  # Simplified coefficient
        
        return {
            'productivity_differential': productivity_diff,
            'bs_adjustment': bs_effect,
            'interpretation': 'Positive values suggest real appreciation pressure'
        }
    
    def _estimate_ppp_half_life(self, ppp_deviation: pd.Series) -> Dict[str, Any]:
        """Estimate PPP mean reversion half-life"""
        # Simple AR(1) model to estimate persistence
        try:
            deviation_clean = ppp_deviation.dropna()
            if len(deviation_clean) < 50:
                return {'half_life': None, 'persistence': None}
            
            # Lag the series
            lagged_deviation = deviation_clean.shift(1).dropna()
            current_deviation = deviation_clean[1:]
            
            # Estimate AR(1) coefficient
            correlation = np.corrcoef(lagged_deviation, current_deviation)[0, 1]
            
            # Half-life calculation: ln(0.5) / ln(ρ)
            if 0 < correlation < 1:
                half_life = np.log(0.5) / np.log(correlation)
                half_life_years = half_life / 252  # Convert to years
            else:
                half_life_years = None
            
            return {
                'half_life_days': half_life if 0 < correlation < 1 else None,
                'half_life_years': half_life_years,
                'persistence_coefficient': correlation,
                'mean_reversion_strength': 1 - correlation if correlation > 0 else None
            }
            
        except Exception as e:
            self.logger.warning(f"Could not estimate PPP half-life: {e}")
            return {'half_life': None, 'persistence': None}
    
    def _analyze_ppp_volatility(self, ppp_deviation: pd.Series) -> Dict[str, Any]:
        """Analyze PPP deviation volatility patterns"""
        # Rolling volatility
        rolling_vol = ppp_deviation.rolling(63).std()  # Quarterly volatility
        
        # Volatility regimes
        vol_median = rolling_vol.median()
        high_vol_threshold = vol_median * 1.5
        low_vol_threshold = vol_median * 0.5
        
        current_vol = rolling_vol.iloc[-1]
        
        if current_vol > high_vol_threshold:
            vol_regime = "HIGH_VOLATILITY"
        elif current_vol < low_vol_threshold:
            vol_regime = "LOW_VOLATILITY"
        else:
            vol_regime = "NORMAL_VOLATILITY"
        
        return {
            'current_volatility': current_vol,
            'volatility_regime': vol_regime,
            'volatility_percentile': stats.percentileofscore(rolling_vol.dropna(), current_vol),
            'volatility_trend': 'INCREASING' if rolling_vol.iloc[-21:].mean() > rolling_vol.iloc[-63:-21].mean() else 'DECREASING'
        }
    
    def _analyze_ppp_regimes(self, values_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze different PPP regimes"""
        if 'ppp_deviation' not in values_df.columns:
            return {'regimes': 'INSUFFICIENT_DATA'}
        
        ppp_dev = values_df['ppp_deviation'].dropna()
        
        # Define regimes based on deviation levels
        overvalued = ppp_dev > 10  # More than 10% overvalued
        undervalued = ppp_dev < -10  # More than 10% undervalued
        fair_value = (ppp_dev >= -10) & (ppp_dev <= 10)
        
        regime_stats = {
            'overvalued_periods': overvalued.sum(),
            'undervalued_periods': undervalued.sum(),
            'fair_value_periods': fair_value.sum(),
            'overvalued_pct': overvalued.mean() * 100,
            'undervalued_pct': undervalued.mean() * 100,
            'fair_value_pct': fair_value.mean() * 100
        }
        
        # Current regime
        current_dev = ppp_dev.iloc[-1]
        if current_dev > 10:
            current_regime = "OVERVALUED"
        elif current_dev < -10:
            current_regime = "UNDERVALUED"
        else:
            current_regime = "FAIR_VALUE"
        
        regime_stats['current_regime'] = current_regime
        
        return regime_stats
    
    def _test_ppp_cointegration(self, exchange_rate: pd.Series, ppp_rate: pd.Series) -> Dict[str, Any]:
        """Test for cointegration between exchange rate and PPP rate"""
        try:
            # Simple cointegration test using correlation and stationarity
            correlation = exchange_rate.corr(ppp_rate)
            
            # Test residuals for stationarity (simplified)
            residuals = exchange_rate - ppp_rate
            residuals_normalized = (residuals - residuals.mean()) / residuals.std()
            
            # Simple stationarity test (check if residuals revert to mean)
            mean_reversion = abs(residuals_normalized.iloc[-1]) < 2.0
            
            return {
                'correlation': correlation,
                'cointegration_strength': 'STRONG' if correlation > 0.8 else 'MODERATE' if correlation > 0.5 else 'WEAK',
                'mean_reversion_test': mean_reversion,
                'residuals_current': residuals_normalized.iloc[-1]
            }
            
        except Exception as e:
            self.logger.warning(f"Cointegration test failed: {e}")
            return {'cointegration_strength': 'UNKNOWN'}
    
    def _generate_signals(self, ppp_deviation: pd.Series, big_mac_deviation: pd.Series,
                         reversion_signal: str, real_exchange_rate: pd.Series,
                         economic_data: Dict[str, Any]) -> List[str]:
        """Generate trading signals based on PPP analysis"""
        signals = []
        
        # Primary PPP signals
        current_deviation = ppp_deviation.iloc[-1]
        
        if current_deviation > 15:
            signals.append("STRONG_SELL")
        elif current_deviation > 5:
            signals.append("SELL")
        elif current_deviation < -15:
            signals.append("STRONG_BUY")
        elif current_deviation < -5:
            signals.append("BUY")
        else:
            signals.append("HOLD")
        
        # Reversion-based signals
        signals.append(f"PPP_{reversion_signal}")
        
        # Big Mac signals
        big_mac_current = big_mac_deviation.iloc[-1]
        if abs(big_mac_current) > 20:
            signals.append("BIG_MAC_EXTREME")
        elif abs(big_mac_current) > 10:
            signals.append("BIG_MAC_SIGNIFICANT")
        
        # Inflation differential signals
        inflation_diff = economic_data['inflation_differential']
        if inflation_diff > 0.02:  # 2% differential
            signals.append("HIGH_INFLATION_DIFFERENTIAL")
        elif inflation_diff < -0.02:
            signals.append("NEGATIVE_INFLATION_DIFFERENTIAL")
        
        # Real exchange rate signals
        rer_trend = real_exchange_rate.pct_change(21).iloc[-1]  # 21-day change
        if rer_trend > 0.05:
            signals.append("REAL_APPRECIATION")
        elif rer_trend < -0.05:
            signals.append("REAL_DEPRECIATION")
        
        # Combined signals
        if current_deviation > 10 and big_mac_current > 10:
            signals.append("MULTI_MODEL_OVERVALUED")
        elif current_deviation < -10 and big_mac_current < -10:
            signals.append("MULTI_MODEL_UNDERVALUED")
        
        return signals
    
    def _create_time_series(self, data: pd.DataFrame, relative_ppp: pd.Series, 
                           absolute_ppp: pd.Series, real_exchange_rate: pd.Series,
                           economic_data: Dict[str, Any], big_mac_ppp: float,
                           base_country: str, quote_country: str) -> pd.DataFrame:
        """Create comprehensive time series DataFrame"""
        # PPP deviations
        relative_deviation = (data['close'] - relative_ppp) / relative_ppp * 100
        absolute_deviation = (data['close'] - absolute_ppp) / absolute_ppp * 100
        big_mac_deviation = (data['close'] - big_mac_ppp) / big_mac_ppp * 100
        
        # Rolling statistics
        rolling_window = min(63, len(data) // 4)  # Quarterly or available data
        
        rolling_mean_rel = relative_deviation.rolling(rolling_window).mean()
        rolling_std_rel = relative_deviation.rolling(rolling_window).std()
        rolling_zscore = (relative_deviation - rolling_mean_rel) / rolling_std_rel
        
        # Volatility measures
        exchange_rate_vol = data['close'].pct_change().rolling(21).std() * np.sqrt(252) * 100
        ppp_deviation_vol = relative_deviation.rolling(21).std()
        
        # Trend indicators
        ppp_trend = relative_ppp.pct_change(21) * 100  # 21-day PPP change
        exchange_trend = data['close'].pct_change(21) * 100  # 21-day exchange rate change
        
        # Fair value bands
        upper_band = relative_ppp * (1 + rolling_std_rel / 100)
        lower_band = relative_ppp * (1 - rolling_std_rel / 100)
        
        result_df = pd.DataFrame({
            'exchange_rate': data['close'],
            'relative_ppp': relative_ppp,
            'absolute_ppp': absolute_ppp,
            'big_mac_ppp': big_mac_ppp,
            'real_exchange_rate': real_exchange_rate,
            'relative_deviation': relative_deviation,
            'absolute_deviation': absolute_deviation,
            'big_mac_deviation': big_mac_deviation,
            'ppp_zscore': rolling_zscore,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'exchange_rate_volatility': exchange_rate_vol,
            'ppp_deviation_volatility': ppp_deviation_vol,
            'ppp_trend': ppp_trend,
            'exchange_trend': exchange_trend,
            'valuation_zone': self._classify_valuation_zone(relative_deviation)
        }, index=data.index)
        
        return result_df
    
    def _classify_valuation_zone(self, deviation: pd.Series) -> pd.Series:
        """Classify valuation zones based on PPP deviation"""
        def classify_value(dev):
            if pd.isna(dev):
                return "UNKNOWN"
            elif dev > 20:
                return "EXTREMELY_OVERVALUED"
            elif dev > 10:
                return "OVERVALUED"
            elif dev > 5:
                return "SLIGHTLY_OVERVALUED"
            elif dev > -5:
                return "FAIR_VALUE"
            elif dev > -10:
                return "SLIGHTLY_UNDERVALUED"
            elif dev > -20:
                return "UNDERVALUED"
            else:
                return "EXTREMELY_UNDERVALUED"
        
        return deviation.apply(classify_value)
    
    def _calculate_confidence(self, economic_data: Dict[str, Any], data_length: int) -> float:
        """Calculate confidence score based on data quality and economic fundamentals"""
        confidence = 0.4  # Base confidence
        
        # Adjust based on data length
        if data_length >= 1260:  # 5 years
            confidence += 0.2
        elif data_length >= 252:  # 1 year
            confidence += 0.1
        
        # Adjust based on inflation differential reliability
        inflation_diff = abs(economic_data['inflation_differential'])
        if inflation_diff > 0.01:  # Significant differential
            confidence += 0.1
        
        # Adjust based on economic data availability
        if 'productivity_differential' in economic_data:
            confidence += 0.1
        
        # Adjust based on price level data quality
        if economic_data.get('price_level_ratio', 1.0) != 1.0:
            confidence += 0.1
        
        return min(0.9, confidence)
    
    def _get_interpretation(self, ppp_deviation: float, reversion_signal: str, 
                          big_mac_deviation: float) -> str:
        """Get interpretation of PPP results"""
        # Main valuation assessment
        if abs(ppp_deviation) > 15:
            valuation = "significantly mispriced"
        elif abs(ppp_deviation) > 5:
            valuation = "moderately mispriced"
        else:
            valuation = "fairly valued"
        
        # Direction
        if ppp_deviation > 0:
            direction = "overvalued"
        else:
            direction = "undervalued"
        
        # Big Mac confirmation
        big_mac_confirm = ""
        if (ppp_deviation > 0 and big_mac_deviation > 0) or (ppp_deviation < 0 and big_mac_deviation < 0):
            big_mac_confirm = " (confirmed by Big Mac Index)"
        elif abs(big_mac_deviation) > abs(ppp_deviation):
            big_mac_confirm = " (Big Mac Index suggests stronger mispricing)"
        
        # Reversion expectation
        if "STRONG" in reversion_signal:
            reversion = "Strong mean reversion expected"
        elif reversion_signal != "FAIR_VALUE":
            reversion = "Moderate mean reversion expected"
        else:
            reversion = "Currency appears fairly valued"
        
        return f"Currency appears {valuation} and {direction}{big_mac_confirm}. {reversion}."
    
    def _empty_result(self, asset_type: AssetType) -> PPPResult:
        """Return empty result for error cases"""
        return PPPResult(
            name="Purchasing Power Parity",
            ppp_rate=0.0,
            current_rate=0.0,
            ppp_deviation=0.0,
            real_exchange_rate=0.0,
            big_mac_ppp=0.0,
            big_mac_deviation=0.0,
            reversion_signal="ERROR",
            fair_value_range=(0.0, 0.0),
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.FUNDAMENTAL,
            signals=["ERROR"]
        )
    
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models."""
        models = {
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'ridge': Ridge(alpha=1.0)
        }
        
        if ADVANCED_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
        
        return models
    
    def _perform_regime_analysis(self, ppp_deviation: pd.Series) -> RegimeAnalysis:
        """Perform regime switching analysis using Hidden Markov Models."""
        try:
            if not HMM_AVAILABLE or len(ppp_deviation) < 50:
                return self._empty_regime_analysis()
            
            # Prepare data for HMM
            deviation_values = ppp_deviation.dropna().values.reshape(-1, 1)
            
            # Fit HMM with 2 regimes (undervalued/overvalued)
            model = hmm.GaussianHMM(n_components=2, covariance_type="full", random_state=42)
            model.fit(deviation_values)
            
            # Predict current regime
            hidden_states = model.predict(deviation_values)
            current_regime = hidden_states[-1]
            
            # Get regime probabilities
            regime_probs = model.predict_proba(deviation_values)
            current_probs = regime_probs[-1]
            
            # Analyze regime characteristics
            regime_means = model.means_.flatten()
            regime_vars = model.covars_.flatten()
            
            # Determine regime descriptions
            if regime_means[0] < regime_means[1]:
                regime_descriptions = {0: "Undervalued Regime", 1: "Overvalued Regime"}
            else:
                regime_descriptions = {0: "Overvalued Regime", 1: "Undervalued Regime"}
            
            # Calculate regime persistence
            regime_persistence = {}
            expected_duration = {}
            
            for i in range(2):
                persistence = model.transmat_[i, i]
                regime_persistence[f'regime_{i}'] = persistence
                expected_duration[f'regime_{i}'] = 1 / (1 - persistence) if persistence < 1 else float('inf')
            
            return RegimeAnalysis(
                current_regime=current_regime,
                regime_probabilities=current_probs,
                regime_description=regime_descriptions[current_regime],
                transition_matrix=model.transmat_,
                regime_persistence=regime_persistence,
                expected_duration=expected_duration
            )
            
        except Exception as e:
            return self._empty_regime_analysis()
    
    def _perform_volatility_analysis(self, ppp_deviation: pd.Series) -> VolatilityAnalysis:
        """Perform volatility clustering analysis using GARCH models."""
        try:
            if not ADVANCED_AVAILABLE or len(ppp_deviation) < 50:
                return self._empty_volatility_analysis()
            
            # Calculate returns from PPP deviation
            deviation_returns = ppp_deviation.pct_change().dropna()
            
            if len(deviation_returns) < 30:
                return self._empty_volatility_analysis()
            
            # Fit GARCH(1,1) model
            garch_model = arch_model(deviation_returns * 100, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            
            # Get conditional volatility
            conditional_vol = garch_fit.conditional_volatility / 100
            current_volatility = conditional_vol.iloc[-1]
            
            # ARCH test for volatility clustering
            arch_test = het_arch(deviation_returns.dropna())
            arch_pvalue = arch_test[1]
            volatility_clustering = arch_pvalue < 0.05
            
            # Volatility forecast
            garch_forecast = garch_fit.forecast(horizon=5).variance.iloc[-1].values / 10000
            
            # Determine volatility regime
            if current_volatility > self.thresholds['volatility_threshold']:
                volatility_regime = "High Volatility"
            elif current_volatility > self.thresholds['volatility_threshold'] * 0.5:
                volatility_regime = "Medium Volatility"
            else:
                volatility_regime = "Low Volatility"
            
            return VolatilityAnalysis(
                current_volatility=current_volatility,
                volatility_regime=volatility_regime,
                garch_forecast=garch_forecast,
                arch_test_pvalue=arch_pvalue,
                volatility_clustering=volatility_clustering,
                conditional_volatility=conditional_vol
            )
            
        except Exception as e:
            return self._empty_volatility_analysis()
    
    def _perform_ml_predictions(self, data: pd.DataFrame, economic_data: Dict[str, Any], 
                               ppp_deviation: pd.Series) -> MachineLearningPrediction:
        """Perform machine learning predictions."""
        try:
            # Prepare features
            features = self._prepare_ml_features(data, economic_data, ppp_deviation)
            
            if features is None or len(features) < 30:
                return self._empty_ml_predictions()
            
            # Prepare target (future PPP deviation)
            target = ppp_deviation.shift(-5).dropna()  # Predict 5 periods ahead
            
            # Align features and target
            min_len = min(len(features), len(target))
            features = features.iloc[:min_len]
            target = target.iloc[:min_len]
            
            if len(features) < 20:
                return self._empty_ml_predictions()
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            predictions = {}
            feature_importance = {}
            model_scores = {}
            cross_val_scores = {}
            
            for name, model in self.ml_models.items():
                try:
                    # Cross-validation
                    cv_scores = cross_val_score(model, features_scaled, target, cv=tscv, scoring='neg_mean_squared_error')
                    cross_val_scores[name] = -cv_scores
                    
                    # Fit model on full data
                    model.fit(features_scaled, target)
                    
                    # Make predictions
                    pred = model.predict(features_scaled[-10:])  # Last 10 predictions
                    predictions[name] = pred
                    
                    # Feature importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        feature_importance[name] = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        feature_importance[name] = np.abs(model.coef_)
                    
                    # Model score
                    model_scores[name] = model.score(features_scaled, target)
                    
                except Exception as e:
                    continue
            
            # Ensemble prediction (simple average)
            if predictions:
                ensemble_pred = np.mean(list(predictions.values()), axis=0)
            else:
                ensemble_pred = np.array([])
            
            # Prediction intervals (simplified)
            prediction_intervals = {}
            for name, pred in predictions.items():
                std_error = np.std(pred)
                lower = pred - 1.96 * std_error
                upper = pred + 1.96 * std_error
                prediction_intervals[name] = (lower, upper)
            
            return MachineLearningPrediction(
                predictions=predictions,
                feature_importance=feature_importance,
                model_scores=model_scores,
                ensemble_prediction=ensemble_pred,
                prediction_intervals=prediction_intervals,
                cross_validation_scores=cross_val_scores
            )
            
        except Exception as e:
            return self._empty_ml_predictions()
    
    def _prepare_ml_features(self, data: pd.DataFrame, economic_data: Dict[str, Any], 
                            ppp_deviation: pd.Series) -> Optional[pd.DataFrame]:
        """Prepare features for machine learning models."""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price-based features
            features['close'] = data['close']
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['rsi'] = self._calculate_rsi(data['close'])
            
            # PPP-based features
            features['ppp_deviation'] = ppp_deviation
            features['ppp_deviation_ma'] = ppp_deviation.rolling(20).mean()
            features['ppp_deviation_std'] = ppp_deviation.rolling(20).std()
            
            # Economic features
            features['inflation_diff'] = economic_data['inflation_differential']
            features['productivity_diff'] = economic_data['productivity_differential']
            
            # Technical features
            features['ma_20'] = data['close'].rolling(20).mean()
            features['ma_50'] = data['close'].rolling(50).mean()
            features['price_to_ma20'] = data['close'] / features['ma_20']
            
            # Lag features
            for lag in [1, 5, 10]:
                features[f'ppp_deviation_lag_{lag}'] = ppp_deviation.shift(lag)
                features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            
            return features.dropna()
            
        except Exception as e:
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _perform_econometric_analysis(self, data: pd.DataFrame, relative_ppp: pd.Series, 
                                     ppp_deviation: pd.Series) -> EconometricAnalysis:
        """Perform advanced econometric analysis."""
        try:
            if not ADVANCED_AVAILABLE:
                return EconometricAnalysis({}, {}, [], {}, {})
            
            # Unit root tests
            unit_root_tests = {}
            for name, series in [('exchange_rate', data['close']), ('ppp_rate', relative_ppp), 
                               ('ppp_deviation', ppp_deviation)]:
                try:
                    adf_stat, adf_pvalue = adfuller(series.dropna())[:2]
                    kpss_stat, kpss_pvalue = kpss(series.dropna())[:2]
                    unit_root_tests[name] = {
                        'adf_statistic': adf_stat,
                        'adf_pvalue': adf_pvalue,
                        'kpss_statistic': kpss_stat,
                        'kpss_pvalue': kpss_pvalue
                    }
                except:
                    unit_root_tests[name] = {'error': 'test_failed'}
            
            # Cointegration tests
            cointegration_tests = {}
            try:
                coint_stat, coint_pvalue, _ = coint(data['close'].dropna(), relative_ppp.dropna())
                cointegration_tests['engle_granger'] = {
                    'statistic': coint_stat,
                    'pvalue': coint_pvalue,
                    'cointegrated': coint_pvalue < self.thresholds['cointegration_pvalue']
                }
            except:
                cointegration_tests['engle_granger'] = {'error': 'test_failed'}
            
            # Structural breaks (simplified)
            structural_breaks = self._detect_structural_breaks(ppp_deviation)
            
            # Half-life estimates
            half_life_estimates = {
                'ppp_deviation': self._estimate_half_life(ppp_deviation)
            }
            
            # Error correction model (simplified)
            error_correction_model = self._estimate_error_correction_model(data['close'], relative_ppp)
            
            return EconometricAnalysis(
                unit_root_tests=unit_root_tests,
                cointegration_tests=cointegration_tests,
                structural_breaks=structural_breaks,
                half_life_estimates=half_life_estimates,
                error_correction_model=error_correction_model
            )
            
        except Exception as e:
            return EconometricAnalysis({}, {}, [], {}, {})
    
    def _detect_structural_breaks(self, series: pd.Series) -> List[int]:
        """Detect structural breaks in time series."""
        try:
            # Simple method: detect large changes in rolling mean
            rolling_mean = series.rolling(20).mean()
            changes = rolling_mean.diff().abs()
            threshold = changes.quantile(0.95)
            breaks = changes[changes > threshold].index.tolist()
            return [series.index.get_loc(idx) for idx in breaks[:5]]  # Max 5 breaks
        except:
            return []
    
    def _estimate_half_life(self, series: pd.Series) -> float:
        """Estimate half-life of mean reversion."""
        try:
            if not ADVANCED_AVAILABLE:
                return 0.0
            
            # AR(1) regression: y_t = α + β*y_{t-1} + ε_t
            y = series.dropna()
            y_lag = y.shift(1).dropna()
            y = y[1:]  # Align with lagged series
            
            if len(y) < 10:
                return 0.0
            
            # Simple linear regression
            X = np.column_stack([np.ones(len(y_lag)), y_lag])
            beta = np.linalg.lstsq(X, y, rcond=None)[0][1]
            
            if beta >= 1 or beta <= 0:
                return float('inf')
            
            half_life = -np.log(2) / np.log(beta)
            return half_life
            
        except:
            return 0.0
    
    def _estimate_error_correction_model(self, exchange_rate: pd.Series, ppp_rate: pd.Series) -> Dict[str, Any]:
        """Estimate error correction model."""
        try:
            # Simplified ECM
            er_diff = exchange_rate.diff().dropna()
            ppp_diff = ppp_rate.diff().dropna()
            error_term = (exchange_rate - ppp_rate).shift(1).dropna()
            
            # Align series
            min_len = min(len(er_diff), len(ppp_diff), len(error_term))
            er_diff = er_diff.iloc[:min_len]
            ppp_diff = ppp_diff.iloc[:min_len]
            error_term = error_term.iloc[:min_len]
            
            if len(er_diff) < 10:
                return {'error': 'insufficient_data'}
            
            # Simple regression: Δer_t = α + β*Δppp_t + γ*ECT_{t-1} + ε_t
            X = np.column_stack([np.ones(len(er_diff)), ppp_diff, error_term])
            coeffs = np.linalg.lstsq(X, er_diff, rcond=None)[0]
            
            return {
                'alpha': coeffs[0],
                'beta': coeffs[1],
                'gamma': coeffs[2],  # Error correction coefficient
                'adjustment_speed': -coeffs[2] if coeffs[2] < 0 else 0
            }
            
        except:
            return {'error': 'estimation_failed'}
    
    def _calculate_risk_metrics(self, ppp_deviation: pd.Series) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        try:
            deviation_values = ppp_deviation.dropna()
            
            if len(deviation_values) < 10:
                return self._empty_risk_metrics()
            
            # VaR and CVaR (95% confidence)
            var_95 = np.percentile(deviation_values, 5)
            cvar_95 = deviation_values[deviation_values <= var_95].mean()
            
            # Maximum drawdown
            cumulative = (1 + deviation_values / 100).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Sharpe and Sortino ratios
            returns = deviation_values.pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0:
                    sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252)
                else:
                    sortino_ratio = 0.0
            else:
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
            
            # Skewness and kurtosis
            skewness = stats.skew(deviation_values)
            kurtosis = stats.kurtosis(deviation_values)
            
            return RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                skewness=skewness,
                kurtosis=kurtosis
            )
            
        except Exception as e:
            return self._empty_risk_metrics()
    
    def _run_model_diagnostics(self, data: pd.DataFrame, ppp_deviation: pd.Series) -> Dict[str, Any]:
        """Run comprehensive model diagnostics."""
        try:
            diagnostics = {}
            
            # Data quality checks
            diagnostics['data_quality'] = {
                'missing_data_pct': data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100,
                'outliers_count': len(data[(np.abs(stats.zscore(data.select_dtypes(include=[np.number]))) > 3).any(axis=1)]),
                'data_length': len(data)
            }
            
            # Stationarity tests
            diagnostics['stationarity'] = {}
            if ADVANCED_AVAILABLE:
                for name, series in [('exchange_rate', data['close']), ('ppp_deviation', ppp_deviation)]:
                    try:
                        adf_stat, adf_pvalue = adfuller(series.dropna())[:2]
                        diagnostics['stationarity'][name] = {
                            'adf_pvalue': adf_pvalue,
                            'is_stationary': adf_pvalue < self.thresholds['unit_root_pvalue']
                        }
                    except:
                        diagnostics['stationarity'][name] = {'error': 'test_failed'}
            else:
                diagnostics['stationarity'] = {'error': 'statsmodels_not_available'}
            
            return diagnostics
            
        except Exception as e:
            return {'error': str(e)}
    
    def _perform_backtesting(self, data: pd.DataFrame, ppp_rate: pd.Series) -> Dict[str, Any]:
        """Perform backtesting of PPP model."""
        try:
            # Simple backtesting: predict next period exchange rate using PPP
            predictions = []
            actuals = []
            
            # Use last 50% of data for backtesting
            split_point = len(data) // 2
            
            for i in range(split_point, len(data) - 1):
                # Predict using PPP rate
                pred = ppp_rate.iloc[i]
                actual = data['close'].iloc[i + 1]
                
                predictions.append(pred)
                actuals.append(actual)
            
            if len(predictions) < 10:
                return {'error': 'insufficient_data'}
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            # Calculate metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            
            # Direction accuracy
            pred_direction = np.diff(predictions) > 0
            actual_direction = np.diff(actuals) > 0
            direction_accuracy = np.mean(pred_direction == actual_direction)
            
            return {
                'mse': mse,
                'mae': mae,
                'direction_accuracy': direction_accuracy,
                'predictions_count': len(predictions)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _perform_sensitivity_analysis(self, data: pd.DataFrame, economic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform sensitivity analysis on key parameters."""
        try:
            sensitivity_results = {}
            
            # Test sensitivity to inflation differential changes
            base_inflation_diff = economic_data['inflation_differential']
            
            for shock in [-0.02, -0.01, 0.01, 0.02]:  # ±2%, ±1% shocks
                shocked_data = economic_data.copy()
                shocked_data['inflation_differential'] = base_inflation_diff + shock
                
                # Recalculate relative PPP with shocked parameters
                initial_rate = data['close'].iloc[0]
                time_periods = np.arange(len(data)) / 252
                
                base_inflation = shocked_data['base_inflation']
                quote_inflation = shocked_data['quote_inflation']
                
                shocked_ppp = initial_rate * ((1 + quote_inflation) / (1 + base_inflation)) ** time_periods
                shocked_deviation = (data['close'] - shocked_ppp) / shocked_ppp * 100
                
                # Calculate impact on final deviation
                base_final_deviation = (data['close'].iloc[-1] - initial_rate * ((1 + economic_data['quote_inflation']) / (1 + economic_data['base_inflation'])) ** (len(data) / 252)) / (initial_rate * ((1 + economic_data['quote_inflation']) / (1 + economic_data['base_inflation'])) ** (len(data) / 252)) * 100
                shocked_final_deviation = shocked_deviation.iloc[-1]
                
                impact = shocked_final_deviation - base_final_deviation
                
                sensitivity_results[f'inflation_shock_{shock:+.3f}'] = {
                    'shock_size': shock * 100,  # Convert to percentage
                    'deviation_impact': impact
                }
            
            return sensitivity_results
            
        except Exception as e:
            return {'error': str(e)}
    
    # Empty result methods
    def _empty_regime_analysis(self) -> RegimeAnalysis:
        return RegimeAnalysis(
            current_regime=0,
            regime_probabilities=np.array([1.0, 0.0]),
            regime_description="Unknown Regime",
            transition_matrix=np.eye(2),
            regime_persistence={'regime_0': 0.0, 'regime_1': 0.0},
            expected_duration={'regime_0': 0.0, 'regime_1': 0.0}
        )
    
    def _empty_volatility_analysis(self) -> VolatilityAnalysis:
        return VolatilityAnalysis(
            current_volatility=0.0,
            volatility_regime="Unknown",
            garch_forecast=np.array([]),
            arch_test_pvalue=1.0,
            volatility_clustering=False,
            conditional_volatility=pd.Series()
        )
    
    def _empty_ml_predictions(self) -> MachineLearningPrediction:
        return MachineLearningPrediction(
            predictions={},
            feature_importance={},
            model_scores={},
            ensemble_prediction=np.array([]),
            prediction_intervals={},
            cross_validation_scores={}
        )
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        return RiskMetrics(
            var_95=0.0,
            cvar_95=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            skewness=0.0,
            kurtosis=0.0
        )
    
    def get_chart_data(self, result: PPPResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'ppp_analysis',
            'name': 'Purchasing Power Parity',
            'data': {
                'exchange_rate': result.values['exchange_rate'].tolist() if 'exchange_rate' in result.values.columns else [],
                'relative_ppp': result.values['relative_ppp'].tolist() if 'relative_ppp' in result.values.columns else [],
                'ppp_deviation': result.values['relative_deviation'].tolist() if 'relative_deviation' in result.values.columns else [],
                'upper_band': result.values['upper_band'].tolist() if 'upper_band' in result.values.columns else [],
                'lower_band': result.values['lower_band'].tolist() if 'lower_band' in result.values.columns else []
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'ppp_metrics': {
                'current_rate': result.current_rate,
                'ppp_rate': result.ppp_rate,
                'ppp_deviation': result.ppp_deviation,
                'big_mac_ppp': result.big_mac_ppp,
                'big_mac_deviation': result.big_mac_deviation,
                'reversion_signal': result.reversion_signal,
                'fair_value_range': result.fair_value_range
            },
            'series': [
                {
                    'name': 'Exchange Rate',
                    'data': result.values['exchange_rate'].tolist() if 'exchange_rate' in result.values.columns else [],
                    'color': '#2196F3',
                    'type': 'line',
                    'lineWidth': 2
                },
                {
                    'name': 'PPP Fair Value',
                    'data': result.values['relative_ppp'].tolist() if 'relative_ppp' in result.values.columns else [],
                    'color': '#4CAF50',
                    'type': 'line',
                    'lineWidth': 2,
                    'dashStyle': 'Dash'
                },
                {
                    'name': 'Upper Band',
                    'data': result.values['upper_band'].tolist() if 'upper_band' in result.values.columns else [],
                    'color': '#FF9800',
                    'type': 'line',
                    'lineWidth': 1,
                    'dashStyle': 'Dot'
                },
                {
                    'name': 'Lower Band',
                    'data': result.values['lower_band'].tolist() if 'lower_band' in result.values.columns else [],
                    'color': '#FF9800',
                    'type': 'line',
                    'lineWidth': 1,
                    'dashStyle': 'Dot'
                },
                {
                    'name': 'PPP Deviation %',
                    'data': result.values['relative_deviation'].tolist() if 'relative_deviation' in result.values.columns else [],
                    'color': '#9C27B0',
                    'type': 'line',
                    'lineWidth': 1,
                    'yAxis': 1
                }
            ]
        }


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate sample EUR/USD exchange rate data
    np.random.seed(42)
    returns = np.random.randn(252) * 0.01
    prices = 1.10 * (1 + returns).cumprod()  # Starting at 1.10 EUR/USD
    
    sample_data = pd.DataFrame({'close': prices}, index=dates)
    
    # Sample inflation data
    sample_inflation = {
        'base_inflation': 0.025,  # US 2.5%
        'quote_inflation': 0.020  # EU 2.0%
    }
    
    # Calculate PPP
    ppp_calculator = PPPIndicator(base_country="US", quote_country="EU")
    result = ppp_calculator.calculate(sample_data, sample_inflation, asset_type=AssetType.FOREX)
    
    print(f"PPP Analysis (EUR/USD):")
    print(f"Current Rate: {result.current_rate:.4f}")
    print(f"PPP Fair Value: {result.ppp_rate:.4f}")
    print(f"PPP Deviation: {result.ppp_deviation:.2f}%")
    print(f"Big Mac PPP: {result.big_mac_ppp:.4f}")
    print(f"Big Mac Deviation: {result.big_mac_deviation:.2f}%")
    print(f"Reversion Signal: {result.reversion_signal}")
    print(f"Fair Value Range: {result.fair_value_range[0]:.4f} - {result.fair_value_range[1]:.4f}")
    print(f"Signals: {', '.join(result.signals)}")