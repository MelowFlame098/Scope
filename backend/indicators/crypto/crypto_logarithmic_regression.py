"""Crypto Logarithmic Regression Model

Implements logarithmic regression analysis for cryptocurrency price prediction and trend analysis.
This model assumes that cryptocurrency prices follow a logarithmic growth pattern over time,
which is useful for long-term trend analysis and identifying potential support/resistance levels.

Author: Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize, curve_fit
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import warnings
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
    TECHNICAL = "technical"
    STATISTICAL = "statistical"
    ON_CHAIN = "on_chain"


@dataclass
class CryptoLogRegressionResult:
    """Result of crypto logarithmic regression analysis"""
    name: str
    current_price: float
    predicted_price: float
    trend_line_value: float
    price_deviation: float
    deviation_percentage: float
    regression_r2: float
    trend_direction: str
    support_level: float
    resistance_level: float
    price_channel: Dict[str, float]
    volatility_bands: Dict[str, float]
    growth_rate: float
    halving_cycle_analysis: Dict[str, Any]
    market_cycle_phase: str
    forecast_scenarios: Dict[str, Dict[str, float]]
    statistical_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class CryptoLogRegressionIndicator:
    """Crypto Logarithmic Regression Calculator with Advanced Analytics"""
    
    def __init__(self, asset: str = "BTC"):
        """
        Initialize crypto logarithmic regression calculator
        
        Args:
            asset: Cryptocurrency asset symbol (default: "BTC")
        """
        self.asset = asset.upper()
        self.logger = logging.getLogger(__name__)
        
        # Asset-specific parameters
        self.asset_params = self._get_asset_parameters()
        
        # Regression parameters
        self.min_data_points = 100
        self.confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
        
        # Halving cycle parameters (for Bitcoin-like assets)
        self.halving_cycles = self._get_halving_cycles()
    
    def calculate(self, data: pd.DataFrame, custom_asset: Optional[str] = None,
                 asset_type: AssetType = AssetType.CRYPTO,
                 regression_type: str = "log_linear",
                 forecast_days: int = 365) -> CryptoLogRegressionResult:
        """
        Calculate crypto logarithmic regression analysis
        
        Args:
            data: Price data DataFrame with 'close' column and datetime index
            custom_asset: Override asset symbol
            asset_type: Type of asset being analyzed
            regression_type: Type of regression ('log_linear', 'polynomial', 'power_law')
            forecast_days: Number of days to forecast
            
        Returns:
            CryptoLogRegressionResult containing comprehensive regression analysis
        """
        try:
            # Use custom asset if provided
            asset = custom_asset or self.asset
            params = self.asset_params.get(asset, self.asset_params['BTC'])
            
            # Validate data
            if len(data) < self.min_data_points:
                self.logger.warning(f"Insufficient data points: {len(data)} < {self.min_data_points}")
                return self._empty_result(asset_type)
            
            # Prepare data for regression
            regression_data = self._prepare_regression_data(data, params)
            
            # Perform logarithmic regression
            regression_results = self._perform_regression(regression_data, regression_type)
            
            # Calculate current metrics
            current_price = data['close'].iloc[-1]
            predicted_price = regression_results['predicted_values'].iloc[-1]
            trend_line_value = regression_results['trend_line'].iloc[-1]
            
            # Price deviation analysis
            price_deviation = current_price - trend_line_value
            deviation_percentage = (price_deviation / trend_line_value) * 100
            
            # Support and resistance levels
            support_resistance = self._calculate_support_resistance(
                regression_results, regression_data
            )
            
            # Price channel analysis
            price_channel = self._calculate_price_channel(regression_results)
            
            # Volatility bands
            volatility_bands = self._calculate_volatility_bands(
                regression_results, regression_data
            )
            
            # Growth rate analysis
            growth_rate = self._calculate_growth_rate(regression_results, params)
            
            # Halving cycle analysis (for applicable assets)
            halving_analysis = self._analyze_halving_cycles(
                data, regression_results, asset, params
            )
            
            # Market cycle phase
            market_phase = self._determine_market_cycle_phase(
                current_price, trend_line_value, deviation_percentage, halving_analysis
            )
            
            # Forecast scenarios
            forecast_scenarios = self._generate_forecast_scenarios(
                regression_results, forecast_days, params
            )
            
            # Statistical metrics
            statistical_metrics = self._calculate_statistical_metrics(
                regression_results, regression_data
            )
            
            # Confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                regression_results, regression_data
            )
            
            # Generate signals
            signals = self._generate_comprehensive_signals(
                deviation_percentage, market_phase, growth_rate,
                halving_analysis, statistical_metrics
            )
            
            # Create time series data
            values_df = self._create_time_series(
                data, regression_results, support_resistance,
                price_channel, volatility_bands
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                len(data), regression_results['r2_score'], asset
            )
            
            return CryptoLogRegressionResult(
                name="Crypto Logarithmic Regression Analysis",
                current_price=current_price,
                predicted_price=predicted_price,
                trend_line_value=trend_line_value,
                price_deviation=price_deviation,
                deviation_percentage=deviation_percentage,
                regression_r2=regression_results['r2_score'],
                trend_direction=self._determine_trend_direction(growth_rate),
                support_level=support_resistance['support'],
                resistance_level=support_resistance['resistance'],
                price_channel=price_channel,
                volatility_bands=volatility_bands,
                growth_rate=growth_rate,
                halving_cycle_analysis=halving_analysis,
                market_cycle_phase=market_phase,
                forecast_scenarios=forecast_scenarios,
                statistical_metrics=statistical_metrics,
                confidence_intervals=confidence_intervals,
                values=values_df,
                metadata={
                    'asset': asset,
                    'asset_parameters': params,
                    'regression_type': regression_type,
                    'data_points': len(data),
                    'regression_equation': regression_results['equation'],
                    'model_diagnostics': self._perform_model_diagnostics(regression_results, regression_data),
                    'trend_analysis': self._analyze_trend_characteristics(regression_results),
                    'cycle_analysis': self._analyze_price_cycles(data, regression_results),
                    'volatility_analysis': self._analyze_volatility_patterns(data, regression_results),
                    'interpretation': self._get_interpretation(
                        deviation_percentage, market_phase, growth_rate, regression_results['r2_score']
                    )
                },
                confidence=confidence,
                timestamp=datetime.now(),
                asset_type=asset_type,
                category=IndicatorCategory.STATISTICAL,
                signals=signals
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating crypto logarithmic regression: {e}")
            return self._empty_result(asset_type)
    
    def _get_asset_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get asset-specific parameters"""
        return {
            'BTC': {
                'genesis_date': datetime(2009, 1, 3),
                'halving_interval': 210000,  # blocks
                'block_time': 10,  # minutes
                'max_supply': 21000000,
                'volatility_factor': 1.0,
                'growth_expectation': 0.5,  # Annual log growth rate
                'cycle_length': 1460,  # ~4 years in days
                'maturity_factor': 0.8
            },
            'ETH': {
                'genesis_date': datetime(2015, 7, 30),
                'halving_interval': None,  # No fixed halving
                'block_time': 12,  # seconds (converted to minutes: 0.2)
                'max_supply': None,  # No fixed supply cap
                'volatility_factor': 1.2,
                'growth_expectation': 0.6,
                'cycle_length': 1095,  # ~3 years
                'maturity_factor': 0.7
            },
            'LTC': {
                'genesis_date': datetime(2011, 10, 7),
                'halving_interval': 840000,
                'block_time': 2.5,
                'max_supply': 84000000,
                'volatility_factor': 1.1,
                'growth_expectation': 0.4,
                'cycle_length': 1460,
                'maturity_factor': 0.75
            },
            'ADA': {
                'genesis_date': datetime(2017, 9, 29),
                'halving_interval': None,
                'block_time': 20,  # seconds
                'max_supply': 45000000000,
                'volatility_factor': 1.3,
                'growth_expectation': 0.7,
                'cycle_length': 1095,
                'maturity_factor': 0.6
            },
            'DOT': {
                'genesis_date': datetime(2020, 5, 26),
                'halving_interval': None,
                'block_time': 6,  # seconds
                'max_supply': None,
                'volatility_factor': 1.4,
                'growth_expectation': 0.8,
                'cycle_length': 730,  # ~2 years (newer asset)
                'maturity_factor': 0.5
            }
        }
    
    def _get_halving_cycles(self) -> Dict[str, List[datetime]]:
        """Get historical and projected halving dates"""
        return {
            'BTC': [
                datetime(2012, 11, 28),  # First halving
                datetime(2016, 7, 9),    # Second halving
                datetime(2020, 5, 11),   # Third halving
                datetime(2024, 4, 20),   # Fourth halving (estimated)
                datetime(2028, 4, 15),   # Fifth halving (estimated)
                datetime(2032, 4, 10)    # Sixth halving (estimated)
            ],
            'LTC': [
                datetime(2015, 8, 25),   # First halving
                datetime(2019, 8, 5),    # Second halving
                datetime(2023, 8, 2),    # Third halving
                datetime(2027, 8, 1),    # Fourth halving (estimated)
                datetime(2031, 8, 1)     # Fifth halving (estimated)
            ]
        }
    
    def _prepare_regression_data(self, data: pd.DataFrame, 
                                params: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Prepare data for logarithmic regression"""
        # Calculate days since genesis
        genesis_date = params['genesis_date']
        days_since_genesis = (data.index - genesis_date).days
        
        # Handle negative days (data before genesis)
        days_since_genesis = pd.Series(days_since_genesis, index=data.index)
        days_since_genesis = days_since_genesis.clip(lower=1)  # Minimum 1 day
        
        # Log transformations
        log_price = np.log(data['close'])
        log_days = np.log(days_since_genesis)
        
        # Additional time features
        years_since_genesis = days_since_genesis / 365.25
        
        # Cyclical features (for halving cycles)
        if params['halving_interval'] is not None:
            # Approximate blocks based on days and block time
            approx_blocks = days_since_genesis * 24 * 60 / params['block_time']
            halving_cycle_position = (approx_blocks % params['halving_interval']) / params['halving_interval']
        else:
            halving_cycle_position = pd.Series(0, index=data.index)
        
        return {
            'price': data['close'],
            'log_price': log_price,
            'days_since_genesis': days_since_genesis,
            'log_days': log_days,
            'years_since_genesis': years_since_genesis,
            'halving_cycle_position': halving_cycle_position,
            'volume': data.get('volume', pd.Series(dtype=float))
        }
    
    def _perform_regression(self, regression_data: Dict[str, pd.Series],
                          regression_type: str) -> Dict[str, Any]:
        """Perform logarithmic regression analysis"""
        log_price = regression_data['log_price']
        log_days = regression_data['log_days']
        days = regression_data['days_since_genesis']
        
        # Remove any infinite or NaN values
        mask = ~(log_price.isna() | log_days.isna() | np.isinf(log_price) | np.isinf(log_days))
        clean_log_price = log_price[mask]
        clean_log_days = log_days[mask]
        clean_days = days[mask]
        
        if len(clean_log_price) < 10:
            raise ValueError("Insufficient clean data for regression")
        
        # Perform regression based on type
        if regression_type == "log_linear":
            # Simple log-linear regression: log(price) = a + b * log(days)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                clean_log_days, clean_log_price
            )
            
            # Generate predictions
            predicted_log_price = intercept + slope * log_days
            predicted_price = np.exp(predicted_log_price)
            
            equation = f"log(price) = {intercept:.4f} + {slope:.4f} * log(days)"
            
        elif regression_type == "polynomial":
            # Polynomial regression in log space
            coeffs = np.polyfit(clean_log_days, clean_log_price, 2)
            predicted_log_price = np.polyval(coeffs, log_days)
            predicted_price = np.exp(predicted_log_price)
            
            r_value = np.corrcoef(clean_log_price, np.polyval(coeffs, clean_log_days))[0, 1]
            slope = coeffs[1]  # Linear coefficient
            intercept = coeffs[2]  # Constant term
            
            equation = f"log(price) = {coeffs[0]:.6f} * log(days)² + {coeffs[1]:.4f} * log(days) + {coeffs[2]:.4f}"
            
        elif regression_type == "power_law":
            # Power law regression: price = a * days^b
            # Equivalent to: log(price) = log(a) + b * log(days)
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                clean_log_days, clean_log_price
            )
            
            predicted_log_price = intercept + slope * log_days
            predicted_price = np.exp(predicted_log_price)
            
            equation = f"price = {np.exp(intercept):.4f} * days^{slope:.4f}"
            
        else:
            raise ValueError(f"Unknown regression type: {regression_type}")
        
        # Calculate residuals and statistics
        residuals = log_price - predicted_log_price
        r2_score = r_value ** 2
        
        # Calculate trend line (smoothed prediction)
        trend_line = predicted_price.rolling(30, center=True).mean().fillna(predicted_price)
        
        return {
            'predicted_values': predicted_price,
            'predicted_log_values': predicted_log_price,
            'trend_line': trend_line,
            'residuals': residuals,
            'r2_score': r2_score,
            'slope': slope,
            'intercept': intercept,
            'equation': equation,
            'regression_type': regression_type
        }
    
    def _calculate_support_resistance(self, regression_results: Dict[str, Any],
                                     regression_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate dynamic support and resistance levels"""
        try:
            predicted_price = regression_results['predicted_values']
            residuals = regression_results['residuals']
            
            # Calculate standard deviation of residuals
            residual_std = residuals.std()
            
            # Support and resistance as trend line ± standard deviations
            current_trend = regression_results['trend_line'].iloc[-1]
            
            # Convert log-space standard deviations to price space
            support_level = current_trend * np.exp(-2 * residual_std)  # 2σ below
            resistance_level = current_trend * np.exp(2 * residual_std)  # 2σ above
            
            # Additional levels
            strong_support = current_trend * np.exp(-3 * residual_std)  # 3σ below
            strong_resistance = current_trend * np.exp(3 * residual_std)  # 3σ above
            
            return {
                'support': support_level,
                'resistance': resistance_level,
                'strong_support': strong_support,
                'strong_resistance': strong_resistance,
                'trend_line': current_trend
            }
            
        except Exception as e:
            self.logger.warning(f"Support/resistance calculation failed: {e}")
            current_price = regression_data['price'].iloc[-1]
            return {
                'support': current_price * 0.8,
                'resistance': current_price * 1.2,
                'strong_support': current_price * 0.6,
                'strong_resistance': current_price * 1.5,
                'trend_line': current_price
            }
    
    def _calculate_price_channel(self, regression_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate price channel boundaries"""
        try:
            trend_line = regression_results['trend_line']
            residuals = regression_results['residuals']
            
            # Calculate channel width based on historical volatility
            residual_std = residuals.std()
            
            current_trend = trend_line.iloc[-1]
            
            # Channel boundaries
            upper_channel = current_trend * np.exp(residual_std)
            lower_channel = current_trend * np.exp(-residual_std)
            
            # Extended channels
            upper_extended = current_trend * np.exp(2 * residual_std)
            lower_extended = current_trend * np.exp(-2 * residual_std)
            
            return {
                'upper_channel': upper_channel,
                'lower_channel': lower_channel,
                'upper_extended': upper_extended,
                'lower_extended': lower_extended,
                'channel_width': (upper_channel - lower_channel) / current_trend * 100
            }
            
        except Exception as e:
            self.logger.warning(f"Price channel calculation failed: {e}")
            return {
                'upper_channel': 0,
                'lower_channel': 0,
                'upper_extended': 0,
                'lower_extended': 0,
                'channel_width': 0
            }
    
    def _calculate_volatility_bands(self, regression_results: Dict[str, Any],
                                   regression_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate volatility-based bands around trend line"""
        try:
            trend_line = regression_results['trend_line']
            price = regression_data['price']
            
            # Calculate rolling volatility
            returns = price.pct_change()
            rolling_vol = returns.rolling(30).std() * np.sqrt(365)  # Annualized
            current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0.5
            
            current_trend = trend_line.iloc[-1]
            
            # Volatility bands (similar to Bollinger Bands but around trend line)
            vol_multiplier = 2.0
            upper_vol_band = current_trend * (1 + vol_multiplier * current_vol / np.sqrt(365))
            lower_vol_band = current_trend * (1 - vol_multiplier * current_vol / np.sqrt(365))
            
            return {
                'upper_volatility_band': upper_vol_band,
                'lower_volatility_band': lower_vol_band,
                'current_volatility': current_vol,
                'volatility_percentile': self._calculate_volatility_percentile(rolling_vol)
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility bands calculation failed: {e}")
            return {
                'upper_volatility_band': 0,
                'lower_volatility_band': 0,
                'current_volatility': 0,
                'volatility_percentile': 50
            }
    
    def _calculate_volatility_percentile(self, rolling_vol: pd.Series) -> float:
        """Calculate current volatility percentile"""
        try:
            if len(rolling_vol.dropna()) < 30:
                return 50.0
            
            current_vol = rolling_vol.iloc[-1]
            clean_vol = rolling_vol.dropna()
            percentile = (clean_vol < current_vol).mean() * 100
            return percentile
            
        except Exception as e:
            return 50.0
    
    def _calculate_growth_rate(self, regression_results: Dict[str, Any],
                              params: Dict[str, Any]) -> float:
        """Calculate annualized growth rate from regression"""
        try:
            slope = regression_results['slope']
            
            # Convert log-space slope to annualized growth rate
            # slope is d(log(price))/d(log(days))
            # For daily growth: (1 + daily_growth)^365 = exp(slope * log(365))
            
            # Approximate annualized growth rate
            annual_growth_rate = slope * np.log(365) * params.get('maturity_factor', 1.0)
            
            return annual_growth_rate
            
        except Exception as e:
            self.logger.warning(f"Growth rate calculation failed: {e}")
            return 0.0
    
    def _analyze_halving_cycles(self, data: pd.DataFrame, regression_results: Dict[str, Any],
                               asset: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze halving cycle effects (for applicable assets)"""
        try:
            if asset not in self.halving_cycles:
                return {'applicable': False, 'reason': 'No halving cycles for this asset'}
            
            halving_dates = self.halving_cycles[asset]
            current_date = data.index[-1]
            
            # Find current halving cycle
            current_cycle = None
            next_halving = None
            days_since_last_halving = None
            days_to_next_halving = None
            
            for i, halving_date in enumerate(halving_dates):
                if current_date >= halving_date:
                    current_cycle = i + 1
                    days_since_last_halving = (current_date - halving_date).days
                    if i + 1 < len(halving_dates):
                        next_halving = halving_dates[i + 1]
                        days_to_next_halving = (next_halving - current_date).days
                else:
                    if next_halving is None:
                        next_halving = halving_date
                        days_to_next_halving = (halving_date - current_date).days
                    break
            
            # Analyze price performance around halvings
            halving_performance = self._analyze_halving_performance(data, halving_dates)
            
            # Current cycle position
            if days_since_last_halving is not None and days_to_next_halving is not None:
                cycle_length = days_since_last_halving + days_to_next_halving
                cycle_position = days_since_last_halving / cycle_length
            else:
                cycle_position = 0.5
            
            return {
                'applicable': True,
                'current_cycle': current_cycle,
                'days_since_last_halving': days_since_last_halving,
                'days_to_next_halving': days_to_next_halving,
                'next_halving_date': next_halving,
                'cycle_position': cycle_position,
                'halving_performance': halving_performance,
                'cycle_phase': self._determine_halving_cycle_phase(cycle_position)
            }
            
        except Exception as e:
            self.logger.warning(f"Halving cycle analysis failed: {e}")
            return {'applicable': False, 'reason': f'Analysis error: {e}'}
    
    def _analyze_halving_performance(self, data: pd.DataFrame, 
                                    halving_dates: List[datetime]) -> Dict[str, Any]:
        """Analyze price performance around halving events"""
        try:
            performance_data = []
            
            for halving_date in halving_dates:
                if halving_date > data.index[-1]:
                    continue  # Future halving
                
                # Get price data around halving
                pre_halving_date = halving_date - timedelta(days=365)
                post_halving_date = halving_date + timedelta(days=365)
                
                # Find closest available dates
                pre_price = None
                halving_price = None
                post_price = None
                
                for date in data.index:
                    if abs((date - pre_halving_date).days) < 30 and pre_price is None:
                        pre_price = data.loc[date, 'close']
                    if abs((date - halving_date).days) < 30 and halving_price is None:
                        halving_price = data.loc[date, 'close']
                    if abs((date - post_halving_date).days) < 30 and post_price is None:
                        post_price = data.loc[date, 'close']
                
                if all([pre_price, halving_price, post_price]):
                    pre_halving_return = (halving_price - pre_price) / pre_price * 100
                    post_halving_return = (post_price - halving_price) / halving_price * 100
                    
                    performance_data.append({
                        'halving_date': halving_date,
                        'pre_halving_return': pre_halving_return,
                        'post_halving_return': post_halving_return,
                        'total_return': (post_price - pre_price) / pre_price * 100
                    })
            
            if performance_data:
                avg_pre_return = np.mean([p['pre_halving_return'] for p in performance_data])
                avg_post_return = np.mean([p['post_halving_return'] for p in performance_data])
                avg_total_return = np.mean([p['total_return'] for p in performance_data])
                
                return {
                    'historical_halvings': len(performance_data),
                    'average_pre_halving_return': avg_pre_return,
                    'average_post_halving_return': avg_post_return,
                    'average_total_return': avg_total_return,
                    'performance_data': performance_data
                }
            else:
                return {'historical_halvings': 0, 'insufficient_data': True}
                
        except Exception as e:
            return {'error': str(e)}
    
    def _determine_halving_cycle_phase(self, cycle_position: float) -> str:
        """Determine current phase in halving cycle"""
        if cycle_position < 0.25:
            return "POST_HALVING_ACCUMULATION"
        elif cycle_position < 0.5:
            return "EARLY_BULL_MARKET"
        elif cycle_position < 0.75:
            return "LATE_BULL_MARKET"
        else:
            return "PRE_HALVING_CONSOLIDATION"
    
    def _determine_market_cycle_phase(self, current_price: float, trend_line_value: float,
                                     deviation_percentage: float, 
                                     halving_analysis: Dict[str, Any]) -> str:
        """Determine current market cycle phase"""
        try:
            # Price relative to trend
            if deviation_percentage > 100:
                price_phase = "EXTREME_OVERVALUATION"
            elif deviation_percentage > 50:
                price_phase = "OVERVALUATION"
            elif deviation_percentage > 0:
                price_phase = "ABOVE_TREND"
            elif deviation_percentage > -25:
                price_phase = "BELOW_TREND"
            elif deviation_percentage > -50:
                price_phase = "UNDERVALUATION"
            else:
                price_phase = "EXTREME_UNDERVALUATION"
            
            # Combine with halving cycle if applicable
            if halving_analysis.get('applicable', False):
                halving_phase = halving_analysis.get('cycle_phase', 'UNKNOWN')
                
                # Combine phases
                if "ACCUMULATION" in halving_phase and "UNDERVALUATION" in price_phase:
                    return "DEEP_ACCUMULATION_PHASE"
                elif "BULL_MARKET" in halving_phase and "OVERVALUATION" in price_phase:
                    return "EUPHORIA_PHASE"
                elif "CONSOLIDATION" in halving_phase:
                    return "DISTRIBUTION_PHASE"
                else:
                    return f"{halving_phase}_{price_phase}"
            else:
                return price_phase
                
        except Exception as e:
            return "UNKNOWN_PHASE"
    
    def _generate_forecast_scenarios(self, regression_results: Dict[str, Any],
                                    forecast_days: int, params: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Generate multiple forecast scenarios"""
        try:
            current_trend = regression_results['trend_line'].iloc[-1]
            growth_rate = regression_results['slope']
            residual_std = regression_results['residuals'].std()
            
            # Time horizons
            horizons = [30, 90, 180, 365, forecast_days] if forecast_days not in [30, 90, 180, 365] else [30, 90, 180, 365]
            
            scenarios = {}
            
            for days in horizons:
                # Base forecast (trend continuation)
                base_forecast = current_trend * np.exp(growth_rate * np.log(1 + days/365))
                
                # Optimistic scenario (trend + 1σ)
                optimistic = base_forecast * np.exp(residual_std)
                
                # Pessimistic scenario (trend - 1σ)
                pessimistic = base_forecast * np.exp(-residual_std)
                
                # Conservative scenario (reduced growth)
                conservative_growth = growth_rate * 0.5
                conservative = current_trend * np.exp(conservative_growth * np.log(1 + days/365))
                
                # Aggressive scenario (enhanced growth)
                aggressive_growth = growth_rate * 1.5
                aggressive = current_trend * np.exp(aggressive_growth * np.log(1 + days/365))
                
                scenarios[f"{days}_days"] = {
                    'base_case': base_forecast,
                    'optimistic': optimistic,
                    'pessimistic': pessimistic,
                    'conservative': conservative,
                    'aggressive': aggressive
                }
            
            return scenarios
            
        except Exception as e:
            self.logger.warning(f"Forecast scenario generation failed: {e}")
            return {}
    
    def _calculate_statistical_metrics(self, regression_results: Dict[str, Any],
                                      regression_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """Calculate comprehensive statistical metrics"""
        try:
            predicted_price = regression_results['predicted_values']
            actual_price = regression_data['price']
            residuals = regression_results['residuals']
            
            # Align data
            mask = ~(predicted_price.isna() | actual_price.isna())
            pred_clean = predicted_price[mask]
            actual_clean = actual_price[mask]
            
            if len(pred_clean) < 10:
                return {'error': 'Insufficient data for metrics'}
            
            # Statistical metrics
            r2 = r2_score(np.log(actual_clean), np.log(pred_clean))
            mse = mean_squared_error(np.log(actual_clean), np.log(pred_clean))
            mae = mean_absolute_error(np.log(actual_clean), np.log(pred_clean))
            
            # Residual analysis
            residual_mean = residuals.mean()
            residual_std = residuals.std()
            residual_skewness = residuals.skew()
            residual_kurtosis = residuals.kurtosis()
            
            # Durbin-Watson test for autocorrelation
            dw_statistic = self._durbin_watson_test(residuals)
            
            # Information criteria
            n = len(actual_clean)
            k = 2  # Number of parameters
            aic = n * np.log(mse) + 2 * k
            bic = n * np.log(mse) + k * np.log(n)
            
            return {
                'r_squared': r2,
                'adjusted_r_squared': 1 - (1 - r2) * (n - 1) / (n - k - 1),
                'mean_squared_error': mse,
                'mean_absolute_error': mae,
                'residual_mean': residual_mean,
                'residual_std': residual_std,
                'residual_skewness': residual_skewness,
                'residual_kurtosis': residual_kurtosis,
                'durbin_watson': dw_statistic,
                'aic': aic,
                'bic': bic,
                'sample_size': n
            }
            
        except Exception as e:
            self.logger.warning(f"Statistical metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _durbin_watson_test(self, residuals: pd.Series) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation"""
        try:
            clean_residuals = residuals.dropna()
            if len(clean_residuals) < 3:
                return 2.0  # No autocorrelation
            
            diff_residuals = clean_residuals.diff().dropna()
            dw = (diff_residuals ** 2).sum() / (clean_residuals ** 2).sum()
            return dw
            
        except Exception as e:
            return 2.0
    
    def _calculate_confidence_intervals(self, regression_results: Dict[str, Any],
                                       regression_data: Dict[str, pd.Series]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for predictions"""
        try:
            residuals = regression_results['residuals']
            residual_std = residuals.std()
            current_prediction = regression_results['predicted_values'].iloc[-1]
            
            confidence_intervals = {}
            
            for confidence_level in self.confidence_levels:
                # Calculate z-score for confidence level
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
                
                # Calculate interval in log space
                log_prediction = np.log(current_prediction)
                margin_of_error = z_score * residual_std
                
                lower_log = log_prediction - margin_of_error
                upper_log = log_prediction + margin_of_error
                
                # Convert back to price space
                lower_bound = np.exp(lower_log)
                upper_bound = np.exp(upper_log)
                
                confidence_intervals[f"{int(confidence_level*100)}%"] = (lower_bound, upper_bound)
            
            return confidence_intervals
            
        except Exception as e:
            self.logger.warning(f"Confidence interval calculation failed: {e}")
            return {}
    
    def _generate_comprehensive_signals(self, deviation_percentage: float, market_phase: str,
                                       growth_rate: float, halving_analysis: Dict[str, Any],
                                       statistical_metrics: Dict[str, float]) -> List[str]:
        """Generate comprehensive trading and investment signals"""
        signals = []
        
        # Price deviation signals
        if deviation_percentage > 100:
            signals.append("EXTREME_OVERVALUATION_SELL")
        elif deviation_percentage > 50:
            signals.append("OVERVALUATION_CAUTION")
        elif deviation_percentage > 25:
            signals.append("ABOVE_TREND_NEUTRAL")
        elif deviation_percentage > -10:
            signals.append("NEAR_TREND_HOLD")
        elif deviation_percentage > -30:
            signals.append("BELOW_TREND_ACCUMULATE")
        elif deviation_percentage > -50:
            signals.append("UNDERVALUATION_BUY")
        else:
            signals.append("EXTREME_UNDERVALUATION_STRONG_BUY")
        
        # Market phase signals
        if "ACCUMULATION" in market_phase:
            signals.append("ACCUMULATION_PHASE_BUY")
        elif "EUPHORIA" in market_phase:
            signals.append("EUPHORIA_PHASE_SELL")
        elif "DISTRIBUTION" in market_phase:
            signals.append("DISTRIBUTION_PHASE_CAUTION")
        
        # Growth rate signals
        if growth_rate > 1.0:
            signals.append("HIGH_GROWTH_MOMENTUM")
        elif growth_rate > 0.5:
            signals.append("MODERATE_GROWTH")
        elif growth_rate > 0:
            signals.append("POSITIVE_GROWTH")
        else:
            signals.append("DECLINING_GROWTH")
        
        # Halving cycle signals
        if halving_analysis.get('applicable', False):
            cycle_phase = halving_analysis.get('cycle_phase', '')
            if "ACCUMULATION" in cycle_phase:
                signals.append("HALVING_ACCUMULATION_BUY")
            elif "BULL_MARKET" in cycle_phase:
                signals.append("HALVING_BULL_MARKET")
            elif "CONSOLIDATION" in cycle_phase:
                signals.append("PRE_HALVING_CONSOLIDATION")
        
        # Model quality signals
        r2 = statistical_metrics.get('r_squared', 0)
        if r2 > 0.9:
            signals.append("HIGH_MODEL_CONFIDENCE")
        elif r2 > 0.7:
            signals.append("MODERATE_MODEL_CONFIDENCE")
        else:
            signals.append("LOW_MODEL_CONFIDENCE")
        
        # Trend strength signals
        if abs(deviation_percentage) < 10 and r2 > 0.8:
            signals.append("STRONG_TREND_FOLLOWING")
        elif abs(deviation_percentage) > 50:
            signals.append("TREND_DEVIATION_WARNING")
        
        return signals
    
    def _create_time_series(self, data: pd.DataFrame, regression_results: Dict[str, Any],
                           support_resistance: Dict[str, float], price_channel: Dict[str, float],
                           volatility_bands: Dict[str, float]) -> pd.DataFrame:
        """Create comprehensive time series DataFrame"""
        result_df = pd.DataFrame({
            'price': data['close'],
            'predicted_price': regression_results['predicted_values'],
            'trend_line': regression_results['trend_line'],
            'residuals': regression_results['residuals']
        }, index=data.index)
        
        # Add support and resistance levels (broadcast to all dates)
        for key, value in support_resistance.items():
            result_df[f'sr_{key}'] = value
        
        # Add price channel (broadcast to all dates)
        for key, value in price_channel.items():
            if isinstance(value, (int, float)):
                result_df[f'channel_{key}'] = value
        
        # Add volatility bands (broadcast to all dates)
        for key, value in volatility_bands.items():
            if isinstance(value, (int, float)):
                result_df[f'vol_{key}'] = value
        
        # Add derived metrics
        result_df['price_deviation'] = result_df['price'] - result_df['trend_line']
        result_df['deviation_percentage'] = (result_df['price_deviation'] / result_df['trend_line']) * 100
        
        # Add moving averages of key metrics
        result_df['trend_ma30'] = result_df['trend_line'].rolling(30).mean()
        result_df['deviation_ma30'] = result_df['deviation_percentage'].rolling(30).mean()
        
        # Add percentile rankings
        result_df['price_percentile'] = result_df['price'].rolling(365).apply(
            lambda x: (x < x.iloc[-1]).mean() * 100 if len(x) > 10 else 50
        )
        result_df['deviation_percentile'] = result_df['deviation_percentage'].rolling(365).apply(
            lambda x: (x < x.iloc[-1]).mean() * 100 if len(x) > 10 else 50
        )
        
        return result_df
    
    def _perform_model_diagnostics(self, regression_results: Dict[str, Any],
                                  regression_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Perform comprehensive model diagnostics"""
        try:
            residuals = regression_results['residuals']
            
            # Normality test (Shapiro-Wilk)
            if len(residuals.dropna()) > 3:
                shapiro_stat, shapiro_p = stats.shapiro(residuals.dropna()[:5000])  # Limit for performance
            else:
                shapiro_stat, shapiro_p = 0, 1
            
            # Heteroscedasticity test (simplified)
            predicted = regression_results['predicted_log_values']
            if len(residuals) > 10 and len(predicted) > 10:
                # Correlation between absolute residuals and predictions
                hetero_corr = np.corrcoef(np.abs(residuals.dropna()), predicted.dropna())[0, 1]
            else:
                hetero_corr = 0
            
            # Autocorrelation test
            dw_stat = self._durbin_watson_test(residuals)
            
            # Outlier detection
            residual_std = residuals.std()
            outliers = (np.abs(residuals) > 3 * residual_std).sum()
            outlier_percentage = outliers / len(residuals) * 100
            
            return {
                'normality_test': {
                    'shapiro_statistic': shapiro_stat,
                    'shapiro_p_value': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                },
                'heteroscedasticity': {
                    'correlation': hetero_corr,
                    'is_homoscedastic': abs(hetero_corr) < 0.3
                },
                'autocorrelation': {
                    'durbin_watson': dw_stat,
                    'has_autocorrelation': dw_stat < 1.5 or dw_stat > 2.5
                },
                'outliers': {
                    'count': outliers,
                    'percentage': outlier_percentage,
                    'acceptable': outlier_percentage < 5
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Model diagnostics failed: {e}")
            return {'error': str(e)}
    
    def _analyze_trend_characteristics(self, regression_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trend characteristics"""
        try:
            slope = regression_results['slope']
            r2 = regression_results['r2_score']
            trend_line = regression_results['trend_line']
            
            # Trend strength
            if r2 > 0.9:
                trend_strength = "VERY_STRONG"
            elif r2 > 0.7:
                trend_strength = "STRONG"
            elif r2 > 0.5:
                trend_strength = "MODERATE"
            else:
                trend_strength = "WEAK"
            
            # Trend direction
            if slope > 0.5:
                trend_direction = "STRONG_UPTREND"
            elif slope > 0.2:
                trend_direction = "MODERATE_UPTREND"
            elif slope > 0:
                trend_direction = "WEAK_UPTREND"
            elif slope > -0.2:
                trend_direction = "SIDEWAYS"
            else:
                trend_direction = "DOWNTREND"
            
            # Trend consistency
            trend_changes = trend_line.pct_change().abs()
            trend_volatility = trend_changes.std()
            
            if trend_volatility < 0.01:
                trend_consistency = "VERY_CONSISTENT"
            elif trend_volatility < 0.02:
                trend_consistency = "CONSISTENT"
            elif trend_volatility < 0.05:
                trend_consistency = "MODERATELY_CONSISTENT"
            else:
                trend_consistency = "INCONSISTENT"
            
            return {
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'trend_consistency': trend_consistency,
                'slope_value': slope,
                'r_squared': r2,
                'trend_volatility': trend_volatility
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_price_cycles(self, data: pd.DataFrame, 
                             regression_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cyclical patterns in price relative to trend"""
        try:
            price = data['close']
            trend = regression_results['trend_line']
            
            # Calculate detrended price (price / trend)
            detrended = price / trend
            
            # Find cycles using autocorrelation
            if len(detrended) > 200:
                # Calculate autocorrelation for different lags
                lags = [30, 60, 90, 180, 365]  # Monthly, bi-monthly, quarterly, semi-annual, annual
                autocorrelations = {}
                
                for lag in lags:
                    if len(detrended) > lag:
                        autocorr = detrended.autocorr(lag)
                        autocorrelations[f"{lag}_days"] = autocorr
                
                # Identify dominant cycle
                if autocorrelations:
                    dominant_cycle = max(autocorrelations.items(), key=lambda x: abs(x[1]))
                else:
                    dominant_cycle = ("none", 0)
                
                # Cycle strength
                max_autocorr = max([abs(v) for v in autocorrelations.values()]) if autocorrelations else 0
                
                if max_autocorr > 0.5:
                    cycle_strength = "STRONG"
                elif max_autocorr > 0.3:
                    cycle_strength = "MODERATE"
                elif max_autocorr > 0.1:
                    cycle_strength = "WEAK"
                else:
                    cycle_strength = "NONE"
                
                return {
                    'autocorrelations': autocorrelations,
                    'dominant_cycle': dominant_cycle[0],
                    'dominant_cycle_strength': dominant_cycle[1],
                    'overall_cycle_strength': cycle_strength,
                    'detrended_volatility': detrended.std()
                }
            else:
                return {'insufficient_data': True}
                
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_volatility_patterns(self, data: pd.DataFrame,
                                    regression_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volatility patterns relative to trend"""
        try:
            price = data['close']
            trend = regression_results['trend_line']
            
            # Calculate returns
            returns = price.pct_change().dropna()
            
            # Rolling volatility
            rolling_vol_30 = returns.rolling(30).std() * np.sqrt(365)
            rolling_vol_90 = returns.rolling(90).std() * np.sqrt(365)
            
            # Volatility relative to trend level
            trend_normalized_vol = rolling_vol_30 / (trend / trend.mean())
            
            # Volatility regimes
            vol_percentiles = rolling_vol_30.rolling(365).apply(
                lambda x: (x < x.iloc[-1]).mean() * 100 if len(x) > 30 else 50
            )
            
            current_vol_percentile = vol_percentiles.iloc[-1] if not vol_percentiles.empty else 50
            
            if current_vol_percentile > 90:
                vol_regime = "EXTREMELY_HIGH"
            elif current_vol_percentile > 75:
                vol_regime = "HIGH"
            elif current_vol_percentile > 25:
                vol_regime = "NORMAL"
            elif current_vol_percentile > 10:
                vol_regime = "LOW"
            else:
                vol_regime = "EXTREMELY_LOW"
            
            return {
                'current_volatility': rolling_vol_30.iloc[-1] if not rolling_vol_30.empty else 0,
                'volatility_percentile': current_vol_percentile,
                'volatility_regime': vol_regime,
                'volatility_trend': self._calculate_volatility_trend(rolling_vol_30),
                'trend_adjusted_volatility': trend_normalized_vol.iloc[-1] if not trend_normalized_vol.empty else 0
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_volatility_trend(self, volatility_series: pd.Series) -> str:
        """Calculate volatility trend direction"""
        try:
            if len(volatility_series) < 60:
                return "INSUFFICIENT_DATA"
            
            recent_vol = volatility_series.tail(30).mean()
            past_vol = volatility_series.head(30).mean()
            
            vol_change = (recent_vol - past_vol) / past_vol * 100
            
            if vol_change > 20:
                return "INCREASING_RAPIDLY"
            elif vol_change > 5:
                return "INCREASING"
            elif vol_change > -5:
                return "STABLE"
            elif vol_change > -20:
                return "DECREASING"
            else:
                return "DECREASING_RAPIDLY"
                
        except Exception as e:
            return "UNKNOWN"
    
    def _determine_trend_direction(self, growth_rate: float) -> str:
        """Determine trend direction from growth rate"""
        if growth_rate > 0.8:
            return "STRONG_BULLISH"
        elif growth_rate > 0.4:
            return "BULLISH"
        elif growth_rate > 0.1:
            return "MODERATELY_BULLISH"
        elif growth_rate > -0.1:
            return "NEUTRAL"
        elif growth_rate > -0.4:
            return "MODERATELY_BEARISH"
        elif growth_rate > -0.8:
            return "BEARISH"
        else:
            return "STRONG_BEARISH"
    
    def _calculate_confidence(self, data_length: int, r2_score: float, asset: str) -> float:
        """Calculate overall confidence in the analysis"""
        try:
            # Base confidence from data length
            if data_length >= 1000:
                length_confidence = 1.0
            elif data_length >= 500:
                length_confidence = 0.9
            elif data_length >= 200:
                length_confidence = 0.7
            elif data_length >= 100:
                length_confidence = 0.5
            else:
                length_confidence = 0.3
            
            # R² confidence
            r2_confidence = min(1.0, r2_score)
            
            # Asset maturity confidence
            asset_confidence = self.asset_params.get(asset, {}).get('maturity_factor', 0.5)
            
            # Combined confidence
            overall_confidence = (length_confidence * 0.4 + 
                                r2_confidence * 0.4 + 
                                asset_confidence * 0.2)
            
            return min(1.0, max(0.1, overall_confidence))
            
        except Exception as e:
            return 0.5
    
    def _get_interpretation(self, deviation_percentage: float, market_phase: str,
                           growth_rate: float, r2_score: float) -> str:
        """Get human-readable interpretation of the analysis"""
        try:
            interpretation = []
            
            # Price deviation interpretation
            if deviation_percentage > 50:
                interpretation.append(f"Price is {deviation_percentage:.1f}% above the long-term logarithmic trend, suggesting potential overvaluation.")
            elif deviation_percentage < -30:
                interpretation.append(f"Price is {abs(deviation_percentage):.1f}% below the long-term trend, indicating potential undervaluation.")
            else:
                interpretation.append(f"Price is {deviation_percentage:.1f}% relative to trend, within normal deviation range.")
            
            # Growth rate interpretation
            annual_growth = growth_rate * 100
            if annual_growth > 50:
                interpretation.append(f"The model suggests an aggressive annual growth rate of {annual_growth:.1f}%.")
            elif annual_growth > 20:
                interpretation.append(f"The trend indicates a strong annual growth rate of {annual_growth:.1f}%.")
            elif annual_growth > 0:
                interpretation.append(f"The long-term trend shows moderate growth of {annual_growth:.1f}% annually.")
            else:
                interpretation.append(f"The trend indicates declining growth of {annual_growth:.1f}% annually.")
            
            # Model quality interpretation
            if r2_score > 0.8:
                interpretation.append(f"The logarithmic regression model has high explanatory power (R² = {r2_score:.3f}).")
            elif r2_score > 0.6:
                interpretation.append(f"The model shows moderate fit to historical data (R² = {r2_score:.3f}).")
            else:
                interpretation.append(f"The model has limited explanatory power (R² = {r2_score:.3f}), suggesting high price volatility.")
            
            # Market phase interpretation
            if "ACCUMULATION" in market_phase:
                interpretation.append("Current market phase suggests an accumulation period, potentially favorable for long-term investors.")
            elif "EUPHORIA" in market_phase or "OVERVALUATION" in market_phase:
                interpretation.append("Market appears to be in a euphoric phase, warranting caution for new investments.")
            elif "DISTRIBUTION" in market_phase:
                interpretation.append("Market may be entering a distribution phase, suggesting potential price consolidation.")
            
            return " ".join(interpretation)
            
        except Exception as e:
            return f"Analysis interpretation unavailable due to error: {e}"
    
    def _empty_result(self, asset_type: AssetType) -> CryptoLogRegressionResult:
        """Return empty result for error cases"""
        return CryptoLogRegressionResult(
            name="Crypto Logarithmic Regression Analysis",
            current_price=0.0,
            predicted_price=0.0,
            trend_line_value=0.0,
            price_deviation=0.0,
            deviation_percentage=0.0,
            regression_r2=0.0,
            trend_direction="UNKNOWN",
            support_level=0.0,
            resistance_level=0.0,
            price_channel={},
            volatility_bands={},
            growth_rate=0.0,
            halving_cycle_analysis={'applicable': False},
            market_cycle_phase="UNKNOWN",
            forecast_scenarios={},
            statistical_metrics={},
            confidence_intervals={},
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.STATISTICAL,
            signals=["ERROR"]
        )
    
    def get_chart_data(self, result: CryptoLogRegressionResult) -> Dict[str, Any]:
        """Prepare chart data for visualization"""
        try:
            if result.values.empty:
                return {'error': 'No data available for charting'}
            
            # Prepare main price and trend data
            chart_data = {
                'data': [
                    {
                        'x': idx.isoformat(),
                        'price': row['price'],
                        'predicted_price': row['predicted_price'],
                        'trend_line': row['trend_line'],
                        'deviation_percentage': row['deviation_percentage'],
                        'support': row.get('sr_support', 0),
                        'resistance': row.get('sr_resistance', 0),
                        'upper_channel': row.get('channel_upper_channel', 0),
                        'lower_channel': row.get('channel_lower_channel', 0)
                    }
                    for idx, row in result.values.iterrows()
                ],
                'metrics': {
                    'current_price': result.current_price,
                    'predicted_price': result.predicted_price,
                    'trend_value': result.trend_line_value,
                    'deviation_percentage': result.deviation_percentage,
                    'regression_r2': result.regression_r2,
                    'growth_rate': result.growth_rate,
                    'market_phase': result.market_cycle_phase,
                    'confidence': result.confidence
                },
                'forecast_scenarios': result.forecast_scenarios,
                'halving_analysis': result.halving_cycle_analysis,
                'statistical_metrics': result.statistical_metrics
            }
            
            # Chart configuration
            chart_config = {
                'type': 'line',
                'title': f'{result.metadata.get("asset", "Crypto")} Logarithmic Regression Analysis',
                'yAxis': [
                    {
                        'id': 'price',
                        'title': 'Price (USD)',
                        'type': 'logarithmic',
                        'position': 'left',
                        'gridLines': True
                    },
                    {
                        'id': 'deviation',
                        'title': 'Deviation %',
                        'type': 'linear',
                        'position': 'right',
                        'gridLines': False
                    }
                ],
                'series': [
                    {
                        'name': 'Actual Price',
                        'data': 'price',
                        'yAxis': 'price',
                        'color': '#2E86C1',
                        'lineWidth': 2,
                        'type': 'line'
                    },
                    {
                        'name': 'Logarithmic Trend',
                        'data': 'trend_line',
                        'yAxis': 'price',
                        'color': '#E74C3C',
                        'lineWidth': 3,
                        'type': 'line',
                        'dashStyle': 'solid'
                    },
                    {
                        'name': 'Predicted Price',
                        'data': 'predicted_price',
                        'yAxis': 'price',
                        'color': '#F39C12',
                        'lineWidth': 1,
                        'type': 'line',
                        'dashStyle': 'dash'
                    },
                    {
                        'name': 'Support Level',
                        'data': 'support',
                        'yAxis': 'price',
                        'color': '#27AE60',
                        'lineWidth': 1,
                        'type': 'line',
                        'dashStyle': 'dot'
                    },
                    {
                        'name': 'Resistance Level',
                        'data': 'resistance',
                        'yAxis': 'price',
                        'color': '#E67E22',
                        'lineWidth': 1,
                        'type': 'line',
                        'dashStyle': 'dot'
                    },
                    {
                        'name': 'Upper Channel',
                        'data': 'upper_channel',
                        'yAxis': 'price',
                        'color': '#9B59B6',
                        'lineWidth': 1,
                        'type': 'line',
                        'dashStyle': 'shortdot',
                        'opacity': 0.6
                    },
                    {
                        'name': 'Lower Channel',
                        'data': 'lower_channel',
                        'yAxis': 'price',
                        'color': '#9B59B6',
                        'lineWidth': 1,
                        'type': 'line',
                        'dashStyle': 'shortdot',
                        'opacity': 0.6
                    },
                    {
                        'name': 'Price Deviation %',
                        'data': 'deviation_percentage',
                        'yAxis': 'deviation',
                        'color': '#34495E',
                        'lineWidth': 1,
                        'type': 'area',
                        'fillOpacity': 0.3
                    }
                ],
                'plotOptions': {
                    'line': {
                        'marker': {
                            'enabled': False,
                            'radius': 2
                        },
                        'animation': True
                    },
                    'area': {
                        'marker': {
                            'enabled': False
                        },
                        'animation': True
                    }
                },
                'tooltip': {
                    'shared': True,
                    'crosshairs': True,
                    'formatter': '''
                        function() {
                            var s = '<b>' + Highcharts.dateFormat('%Y-%m-%d', this.x) + '</b>';
                            this.points.forEach(function(point) {
                                if (point.series.name.includes('Price') || point.series.name.includes('Level') || point.series.name.includes('Channel')) {
                                    s += '<br/>' + point.series.name + ': $' + Highcharts.numberFormat(point.y, 2);
                                } else {
                                    s += '<br/>' + point.series.name + ': ' + Highcharts.numberFormat(point.y, 2) + '%';
                                }
                            });
                            return s;
                        }
                    '''
                },
                'legend': {
                    'enabled': True,
                    'layout': 'horizontal',
                    'align': 'center',
                    'verticalAlign': 'bottom'
                },
                'xAxis': {
                    'type': 'datetime',
                    'title': 'Date',
                    'gridLines': True
                }
            }
            
            return {
                'chart_data': chart_data,
                'chart_config': chart_config,
                'summary': {
                    'title': 'Crypto Logarithmic Regression Summary',
                    'current_price': f"${result.current_price:,.2f}",
                    'trend_value': f"${result.trend_line_value:,.2f}",
                    'deviation': f"{result.deviation_percentage:+.1f}%",
                    'growth_rate': f"{result.growth_rate*100:.1f}% annually",
                    'model_fit': f"R² = {result.regression_r2:.3f}",
                    'market_phase': result.market_cycle_phase,
                    'confidence': f"{result.confidence*100:.0f}%",
                    'signals': result.signals[:3]  # Top 3 signals
                }
            }
            
        except Exception as e:
            logger.error(f"Error preparing chart data: {e}")
            return {'error': f'Chart data preparation failed: {e}'}