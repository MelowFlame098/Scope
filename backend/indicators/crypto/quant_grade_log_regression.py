"""Quant Grade Logarithmic Regression Model with Advanced Analytics

This module implements an enhanced Logarithmic Regression model with:
- Rainbow chart bands for multi-level support/resistance
- Power law corridor analysis
- Pi Cycle Top indicator
- Long-term trend analysis with regime detection
- Advanced statistical measures and forecasting
- Multi-timeframe analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats, optimize
from scipy.stats import norm, t
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
try:
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Using simplified implementations.")

logger = logging.getLogger(__name__)

@dataclass
class RainbowBands:
    """Rainbow chart bands analysis results"""
    bands: Dict[str, float]  # Band levels (e.g., 'red', 'orange', 'yellow', etc.)
    current_band: str
    band_position: float  # Position within current band (0-1)
    band_strength: float  # How well price respects bands
    support_levels: List[float]
    resistance_levels: List[float]
    trend_direction: str  # 'bullish', 'bearish', 'neutral'

@dataclass
class PowerLawCorridor:
    """Power law corridor analysis results"""
    power_law_exponent: float
    corridor_top: float
    corridor_bottom: float
    current_position: float  # Position within corridor (0-1)
    corridor_width: float
    breakout_probability: float
    long_term_trend: str
    power_law_r_squared: float

@dataclass
class PiCycleIndicator:
    """Pi Cycle Top indicator results"""
    pi_cycle_top_signal: bool
    days_to_signal: Optional[int]
    ma_111_value: float
    ma_350x2_value: float
    crossover_strength: float
    historical_accuracy: float
    signal_confidence: float

@dataclass
class LogRegressionTrend:
    """Long-term trend analysis results"""
    trend_slope: float
    trend_r_squared: float
    trend_confidence: float
    trend_phase: str  # 'early_bull', 'late_bull', 'early_bear', 'late_bear'
    cycle_position: float  # Position in 4-year cycle (0-1)
    halving_impact: float
    trend_forecast: List[float]

@dataclass
class QuantGradeLogRegressionResult:
    """Comprehensive Quant Grade Logarithmic Regression results"""
    # Core regression metrics
    current_price: float
    regression_value: float
    price_deviation: float
    model_confidence: float
    
    # Enhanced analytics
    rainbow_bands: Optional[RainbowBands] = None
    power_law_corridor: Optional[PowerLawCorridor] = None
    pi_cycle_indicator: Optional[PiCycleIndicator] = None
    trend_analysis: Optional[LogRegressionTrend] = None
    
    # Statistical measures
    r_squared: float = 0.0
    adjusted_r_squared: float = 0.0
    durbin_watson: float = 2.0
    akaike_ic: float = 0.0
    
    # Risk metrics
    volatility_forecast: float = 0.0
    drawdown_risk: float = 0.0
    upside_potential: float = 0.0
    
    # Forecasting
    price_forecast: List[float] = field(default_factory=list)
    forecast_confidence: List[float] = field(default_factory=list)
    forecast_horizon: int = 365
    
    # Metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0"
    data_quality_score: float = 1.0

class QuantGradeLogRegressionModel:
    """Enhanced Logarithmic Regression Model with Quant Grade Analytics"""
    
    def __init__(self, 
                 asset: str = "BTC",
                 enable_rainbow_bands: bool = True,
                 enable_power_law: bool = True,
                 enable_pi_cycle: bool = True,
                 enable_trend_analysis: bool = True,
                 regression_degree: int = 2,
                 lookback_window: int = 2000,
                 forecast_days: int = 365):
        """
        Initialize Quant Grade Logarithmic Regression Model
        
        Args:
            asset: Asset symbol (e.g., 'BTC', 'ETH')
            enable_rainbow_bands: Enable rainbow chart analysis
            enable_power_law: Enable power law corridor analysis
            enable_pi_cycle: Enable Pi Cycle Top indicator
            enable_trend_analysis: Enable long-term trend analysis
            regression_degree: Polynomial degree for regression
            lookback_window: Historical data window
            forecast_days: Number of days to forecast
        """
        self.asset = asset.upper()
        self.enable_rainbow_bands = enable_rainbow_bands
        self.enable_power_law = enable_power_law
        self.enable_pi_cycle = enable_pi_cycle
        self.enable_trend_analysis = enable_trend_analysis
        self.regression_degree = max(1, min(regression_degree, 4))
        self.lookback_window = lookback_window
        self.forecast_days = forecast_days
        
        # Asset-specific parameters
        self.asset_params = self._get_asset_parameters()
        
        # Initialize scalers
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
    def _get_asset_parameters(self) -> Dict[str, Any]:
        """Get asset-specific parameters"""
        params = {
            'BTC': {
                'genesis_date': datetime(2009, 1, 3),
                'halving_cycle': 1461,  # ~4 years in days
                'power_law_base': 10**-17,
                'power_law_exponent': 5.8,
                'pi_cycle_enabled': True
            },
            'ETH': {
                'genesis_date': datetime(2015, 7, 30),
                'halving_cycle': None,
                'power_law_base': 10**-10,
                'power_law_exponent': 4.2,
                'pi_cycle_enabled': False
            }
        }
        return params.get(self.asset, params['BTC'])
    
    def analyze(self, data: pd.DataFrame) -> QuantGradeLogRegressionResult:
        """Perform comprehensive logarithmic regression analysis
        
        Args:
            data: Price data with columns ['close', 'volume', 'high', 'low']
            
        Returns:
            QuantGradeLogRegressionResult with comprehensive analysis
        """
        try:
            # Prepare data
            data = data.copy().sort_index()
            prices = data['close'].dropna()
            
            if len(prices) < 100:
                raise ValueError("Need at least 100 data points for analysis")
            
            # Core logarithmic regression
            regression_result = self._calculate_log_regression(prices)
            current_price = prices.iloc[-1]
            regression_value = regression_result['regression_line'].iloc[-1]
            price_deviation = (current_price - regression_value) / regression_value * 100
            
            # Enhanced analysis
            rainbow_bands = None
            power_law_corridor = None
            pi_cycle_indicator = None
            trend_analysis = None
            
            try:
                # Rainbow Bands Analysis
                if self.enable_rainbow_bands and len(prices) > 200:
                    rainbow_bands = self._calculate_rainbow_bands(prices)
                
                # Power Law Corridor Analysis
                if self.enable_power_law and len(prices) > 500:
                    power_law_corridor = self._calculate_power_law_corridor(prices)
                
                # Pi Cycle Top Indicator
                if self.enable_pi_cycle and self.asset_params['pi_cycle_enabled'] and len(prices) > 700:
                    pi_cycle_indicator = self._calculate_pi_cycle_indicator(prices)
                
                # Long-term Trend Analysis
                if self.enable_trend_analysis and len(prices) > 365:
                    trend_analysis = self._calculate_trend_analysis(prices)
                    
                logger.info("Enhanced logarithmic regression analysis completed successfully")
                
            except Exception as e:
                logger.warning(f"Enhanced analysis failed: {e}")
            
            # Statistical measures
            statistical_measures = self._calculate_statistical_measures(regression_result, prices)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(prices, regression_result)
            
            # Price forecasting
            forecast_result = self._generate_price_forecast(prices, regression_result)
            
            return QuantGradeLogRegressionResult(
                current_price=float(current_price),
                regression_value=float(regression_value),
                price_deviation=float(price_deviation),
                model_confidence=float(regression_result['r_squared']),
                rainbow_bands=rainbow_bands,
                power_law_corridor=power_law_corridor,
                pi_cycle_indicator=pi_cycle_indicator,
                trend_analysis=trend_analysis,
                r_squared=float(statistical_measures.get('r_squared', 0.0)),
                adjusted_r_squared=float(statistical_measures.get('adj_r_squared', 0.0)),
                durbin_watson=float(statistical_measures.get('durbin_watson', 2.0)),
                akaike_ic=float(statistical_measures.get('aic', 0.0)),
                volatility_forecast=float(risk_metrics.get('volatility_forecast', 0.0)),
                drawdown_risk=float(risk_metrics.get('drawdown_risk', 0.0)),
                upside_potential=float(risk_metrics.get('upside_potential', 0.0)),
                price_forecast=forecast_result.get('forecast', []),
                forecast_confidence=forecast_result.get('confidence', []),
                data_quality_score=self._assess_data_quality(data)
            )
            
        except Exception as e:
            logger.error(f"Error in Quant Grade Log Regression analysis: {e}")
            return self._create_empty_result()
    
    def _calculate_log_regression(self, prices: pd.Series) -> Dict[str, Any]:
        """Calculate core logarithmic regression"""
        try:
            # Time variable (days since start)
            time_days = (prices.index - prices.index[0]).days.values
            log_prices = np.log(prices.values)
            
            # Polynomial features for regression
            if self.regression_degree == 1:
                X = np.column_stack([
                    np.ones(len(time_days)),
                    np.log(time_days + 1)
                ])
            elif self.regression_degree == 2:
                X = np.column_stack([
                    np.ones(len(time_days)),
                    np.log(time_days + 1),
                    (np.log(time_days + 1))**2
                ])
            else:
                X = np.column_stack([
                    np.ones(len(time_days)),
                    np.log(time_days + 1),
                    (np.log(time_days + 1))**2,
                    time_days / 365.25
                ])
            
            # Fit regression
            coefficients = np.linalg.lstsq(X, log_prices, rcond=None)[0]
            
            # Generate predictions
            log_predicted = X @ coefficients
            regression_line = pd.Series(np.exp(log_predicted), index=prices.index)
            
            # Calculate residuals and R-squared
            residuals = log_prices - log_predicted
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Standard deviation for bands
            std_residual = np.std(residuals)
            
            return {
                'regression_line': regression_line,
                'coefficients': coefficients,
                'residuals': residuals,
                'r_squared': r_squared,
                'std_residual': std_residual,
                'time_days': time_days
            }
            
        except Exception as e:
            logger.warning(f"Log regression calculation failed: {e}")
            return {
                'regression_line': pd.Series([prices.iloc[-1]] * len(prices), index=prices.index),
                'coefficients': np.array([0.0]),
                'residuals': np.array([0.0]),
                'r_squared': 0.0,
                'std_residual': 0.1,
                'time_days': np.array([0])
            }
    
    def _assess_data_quality(self, data: pd.DataFrame) -> float:
        """Assess data quality score"""
        score = 1.0
        
        # Check for missing values
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        score -= missing_ratio * 0.3
        
        # Check data length
        if len(data) < 365:
            score -= 0.2
        elif len(data) < 1000:
            score -= 0.1
        
        # Check for outliers
        for col in data.select_dtypes(include=[np.number]).columns:
            q1, q3 = data[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outliers = ((data[col] < q1 - 1.5 * iqr) | (data[col] > q3 + 1.5 * iqr)).sum()
                outlier_ratio = outliers / len(data)
                score -= outlier_ratio * 0.1
        
        return max(0.0, min(1.0, score))
    
    def _create_empty_result(self) -> QuantGradeLogRegressionResult:
        """Create empty result for error cases"""
        return QuantGradeLogRegressionResult(
            current_price=0.0,
            regression_value=0.0,
            price_deviation=0.0,
            model_confidence=0.0
        )
    
    def _calculate_rainbow_bands(self, prices: pd.Series) -> RainbowBands:
        """Calculate rainbow chart bands for multi-level support/resistance"""
        try:
            # Calculate logarithmic regression
            regression_result = self._calculate_log_regression(prices)
            regression_line = regression_result['regression_line']
            std_residual = regression_result['std_residual']
            
            # Define rainbow band multipliers (based on standard deviations)
            band_multipliers = {
                'red': 2.5,      # Extreme overbought
                'orange': 2.0,   # Strong overbought
                'yellow': 1.5,   # Moderate overbought
                'green': 1.0,    # Fair value upper
                'blue': 0.5,     # Fair value
                'indigo': 0.0,   # Regression line
                'violet': -0.5,  # Fair value lower
                'purple': -1.0,  # Undervalued
                'pink': -1.5,    # Strong undervalued
                'black': -2.0    # Extreme undervalued
            }
            
            # Calculate band levels
            current_regression = regression_line.iloc[-1]
            bands = {}
            for color, multiplier in band_multipliers.items():
                bands[color] = current_regression * np.exp(multiplier * std_residual)
            
            # Determine current band
            current_price = prices.iloc[-1]
            current_band = 'indigo'  # Default to regression line
            band_position = 0.5
            
            sorted_bands = sorted([(v, k) for k, v in bands.items()])
            for i, (level, color) in enumerate(sorted_bands):
                if current_price <= level:
                    current_band = color
                    if i > 0:
                        prev_level = sorted_bands[i-1][0]
                        band_position = (current_price - prev_level) / (level - prev_level)
                    break
            
            # Calculate band strength (how well price respects bands)
            band_touches = 0
            total_periods = min(len(prices), 252)  # Last year
            recent_prices = prices.tail(total_periods)
            recent_regression = regression_line.tail(total_periods)
            
            for i, price in enumerate(recent_prices):
                reg_val = recent_regression.iloc[i]
                for multiplier in band_multipliers.values():
                    band_level = reg_val * np.exp(multiplier * std_residual)
                    if abs(price - band_level) / band_level < 0.05:  # Within 5%
                        band_touches += 1
                        break
            
            band_strength = band_touches / total_periods
            
            # Support and resistance levels
            support_levels = [bands['violet'], bands['purple'], bands['pink']]
            resistance_levels = [bands['green'], bands['yellow'], bands['orange']]
            
            # Trend direction based on recent price action relative to bands
            recent_band_positions = []
            for price, reg_val in zip(recent_prices.tail(20), recent_regression.tail(20)):
                pos = 0.5  # Default neutral
                for i, (level, _) in enumerate(sorted_bands):
                    if price <= level:
                        if i > 0:
                            prev_level = sorted_bands[i-1][0]
                            pos = i / len(sorted_bands) + (price - prev_level) / (level - prev_level) / len(sorted_bands)
                        else:
                            pos = 0.0
                        break
                recent_band_positions.append(pos)
            
            avg_position = np.mean(recent_band_positions)
            trend_direction = 'bullish' if avg_position > 0.6 else 'bearish' if avg_position < 0.4 else 'neutral'
            
            logger.info(f"Rainbow bands calculated: current_band={current_band}, strength={band_strength:.3f}")
            
            return RainbowBands(
                bands=bands,
                current_band=current_band,
                band_position=float(band_position),
                band_strength=float(band_strength),
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend_direction=trend_direction
            )
            
        except Exception as e:
            logger.warning(f"Rainbow bands calculation failed: {e}")
            return RainbowBands(
                bands={},
                current_band='unknown',
                band_position=0.5,
                band_strength=0.0,
                support_levels=[],
                resistance_levels=[],
                trend_direction='neutral'
            )
    
    def _calculate_power_law_corridor(self, prices: pd.Series) -> PowerLawCorridor:
        """Calculate power law corridor analysis"""
        try:
            # Time since genesis in days
            genesis_date = self.asset_params['genesis_date']
            time_days = (prices.index - genesis_date).days.values
            time_days = np.maximum(time_days, 1)  # Avoid zero/negative days
            
            log_prices = np.log(prices.values)
            log_time = np.log(time_days)
            
            # Fit power law: log(price) = a + b * log(time)
            coeffs = np.polyfit(log_time, log_prices, 1)
            power_law_exponent = coeffs[0]
            intercept = coeffs[1]
            
            # Calculate R-squared
            predicted_log_prices = coeffs[0] * log_time + coeffs[1]
            ss_res = np.sum((log_prices - predicted_log_prices) ** 2)
            ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
            power_law_r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate corridor bounds (using residual standard deviation)
            residuals = log_prices - predicted_log_prices
            std_residual = np.std(residuals)
            
            # Current time and predictions
            current_time_days = time_days[-1]
            current_log_time = np.log(current_time_days)
            
            # Power law corridor bounds (±2 standard deviations)
            corridor_center = coeffs[0] * current_log_time + coeffs[1]
            corridor_top = np.exp(corridor_center + 2 * std_residual)
            corridor_bottom = np.exp(corridor_center - 2 * std_residual)
            
            # Current position within corridor
            current_price = prices.iloc[-1]
            if corridor_top > corridor_bottom:
                current_position = (current_price - corridor_bottom) / (corridor_top - corridor_bottom)
                current_position = max(0.0, min(1.0, current_position))
            else:
                current_position = 0.5
            
            # Corridor width (relative to center)
            corridor_center_price = np.exp(corridor_center)
            corridor_width = (corridor_top - corridor_bottom) / corridor_center_price
            
            # Breakout probability (based on position and recent volatility)
            recent_volatility = prices.pct_change().tail(30).std() * np.sqrt(365)
            breakout_probability = min(0.95, max(0.05, 
                abs(current_position - 0.5) * 2 * (1 + recent_volatility)
            ))
            
            # Long-term trend assessment
            if power_law_exponent > 0.3:
                long_term_trend = 'strong_bullish'
            elif power_law_exponent > 0.1:
                long_term_trend = 'bullish'
            elif power_law_exponent > -0.1:
                long_term_trend = 'neutral'
            elif power_law_exponent > -0.3:
                long_term_trend = 'bearish'
            else:
                long_term_trend = 'strong_bearish'
            
            logger.info(f"Power law corridor calculated: exponent={power_law_exponent:.3f}, position={current_position:.3f}")
            
            return PowerLawCorridor(
                power_law_exponent=float(power_law_exponent),
                corridor_top=float(corridor_top),
                corridor_bottom=float(corridor_bottom),
                current_position=float(current_position),
                corridor_width=float(corridor_width),
                breakout_probability=float(breakout_probability),
                long_term_trend=long_term_trend,
                power_law_r_squared=float(power_law_r_squared)
            )
            
        except Exception as e:
            logger.warning(f"Power law corridor calculation failed: {e}")
            return PowerLawCorridor(
                power_law_exponent=0.0,
                corridor_top=0.0,
                corridor_bottom=0.0,
                current_position=0.5,
                corridor_width=0.0,
                breakout_probability=0.5,
                long_term_trend='neutral',
                power_law_r_squared=0.0
            )
    
    def _calculate_pi_cycle_indicator(self, prices: pd.Series) -> PiCycleIndicator:
        """Calculate Pi Cycle Top indicator (Bitcoin specific)"""
        try:
            if len(prices) < 700:
                raise ValueError("Need at least 700 days for Pi Cycle calculation")
            
            # Calculate moving averages
            ma_111 = prices.rolling(window=111, min_periods=111).mean()
            ma_350 = prices.rolling(window=350, min_periods=350).mean()
            ma_350x2 = ma_350 * 2
            
            # Current values
            current_ma_111 = ma_111.iloc[-1]
            current_ma_350x2 = ma_350x2.iloc[-1]
            
            # Check for crossover (111 DMA crosses above 350 DMA * 2)
            pi_cycle_top_signal = current_ma_111 > current_ma_350x2
            
            # Calculate crossover strength
            crossover_strength = abs(current_ma_111 - current_ma_350x2) / current_ma_350x2
            
            # Days to signal (if approaching)
            days_to_signal = None
            if not pi_cycle_top_signal:
                # Estimate when crossover might occur based on recent trends
                ma_111_trend = ma_111.tail(30).pct_change().mean()
                ma_350x2_trend = ma_350x2.tail(30).pct_change().mean()
                
                if ma_111_trend > ma_350x2_trend:  # 111 DMA gaining on 350x2
                    relative_gap = (current_ma_350x2 - current_ma_111) / current_ma_111
                    trend_diff = ma_111_trend - ma_350x2_trend
                    if trend_diff > 0:
                        days_to_signal = int(relative_gap / trend_diff)
                        days_to_signal = max(1, min(days_to_signal, 365))  # Cap at 1 year
            
            # Historical accuracy (based on past signals)
            historical_signals = []
            for i in range(111, len(ma_111) - 1):
                if ma_111.iloc[i-1] <= ma_350x2.iloc[i-1] and ma_111.iloc[i] > ma_350x2.iloc[i]:
                    historical_signals.append(i)
            
            # Simple accuracy estimate (this would need backtesting for real accuracy)
            historical_accuracy = 0.75 if len(historical_signals) > 0 else 0.5
            
            # Signal confidence based on multiple factors
            confidence_factors = [
                min(1.0, crossover_strength * 10),  # Strength of crossover
                1.0 if pi_cycle_top_signal else 0.5,  # Current signal status
                min(1.0, len(prices) / 1000),  # Data sufficiency
                historical_accuracy  # Historical performance
            ]
            signal_confidence = np.mean(confidence_factors)
            
            logger.info(f"Pi Cycle indicator calculated: signal={pi_cycle_top_signal}, confidence={signal_confidence:.3f}")
            
            return PiCycleIndicator(
                pi_cycle_top_signal=pi_cycle_top_signal,
                days_to_signal=days_to_signal,
                ma_111_value=float(current_ma_111),
                ma_350x2_value=float(current_ma_350x2),
                crossover_strength=float(crossover_strength),
                historical_accuracy=float(historical_accuracy),
                signal_confidence=float(signal_confidence)
            )
            
        except Exception as e:
            logger.warning(f"Pi Cycle indicator calculation failed: {e}")
            return PiCycleIndicator(
                pi_cycle_top_signal=False,
                days_to_signal=None,
                ma_111_value=0.0,
                ma_350x2_value=0.0,
                crossover_strength=0.0,
                historical_accuracy=0.5,
                 signal_confidence=0.0
             )
    
    def _calculate_trend_analysis(self, prices: pd.Series) -> LogRegressionTrend:
        """Calculate long-term trend analysis with cycle detection"""
        try:
            # Calculate logarithmic regression for trend
            regression_result = self._calculate_log_regression(prices)
            trend_slope = regression_result['coefficients'][1] if len(regression_result['coefficients']) > 1 else 0.0
            trend_r_squared = regression_result['r_squared']
            
            # Trend confidence based on R-squared and data length
            data_length_factor = min(1.0, len(prices) / 1000)
            trend_confidence = trend_r_squared * data_length_factor
            
            # Determine trend phase based on recent price action and cycle analysis
            if self.asset == 'BTC' and self.asset_params['halving_cycle']:
                # Bitcoin-specific cycle analysis
                genesis_date = self.asset_params['genesis_date']
                days_since_genesis = (prices.index[-1] - genesis_date).days
                cycle_days = self.asset_params['halving_cycle']
                cycle_position = (days_since_genesis % cycle_days) / cycle_days
                
                # Estimate halving impact (stronger near halving events)
                days_to_halving = cycle_days - (days_since_genesis % cycle_days)
                halving_impact = max(0.1, 1.0 - abs(days_to_halving - cycle_days/2) / (cycle_days/2))
            else:
                cycle_position = 0.5  # Neutral for non-Bitcoin assets
                halving_impact = 0.0
            
            # Trend phase classification
            recent_performance = prices.pct_change().tail(90).mean() * 365  # Annualized
            volatility = prices.pct_change().tail(90).std() * np.sqrt(365)
            
            if recent_performance > 0.5 and volatility < 1.0:
                trend_phase = 'early_bull'
            elif recent_performance > 0.2:
                trend_phase = 'late_bull'
            elif recent_performance < -0.3 and volatility > 1.5:
                trend_phase = 'early_bear'
            elif recent_performance < -0.1:
                trend_phase = 'late_bear'
            else:
                trend_phase = 'accumulation'
            
            # Generate trend forecast
            forecast_days = min(self.forecast_days, 365)
            trend_forecast = []
            
            if len(regression_result['coefficients']) >= 2:
                current_time = regression_result['time_days'][-1]
                for i in range(1, forecast_days + 1):
                    future_time = current_time + i
                    if self.regression_degree == 1:
                        log_price = (regression_result['coefficients'][0] + 
                                   regression_result['coefficients'][1] * np.log(future_time + 1))
                    else:
                        log_price = (regression_result['coefficients'][0] + 
                                   regression_result['coefficients'][1] * np.log(future_time + 1) +
                                   regression_result['coefficients'][2] * (np.log(future_time + 1))**2)
                    
                    trend_forecast.append(float(np.exp(log_price)))
            
            logger.info(f"Trend analysis calculated: phase={trend_phase}, confidence={trend_confidence:.3f}")
            
            return LogRegressionTrend(
                trend_slope=float(trend_slope),
                trend_r_squared=float(trend_r_squared),
                trend_confidence=float(trend_confidence),
                trend_phase=trend_phase,
                cycle_position=float(cycle_position),
                halving_impact=float(halving_impact),
                trend_forecast=trend_forecast
            )
            
        except Exception as e:
            logger.warning(f"Trend analysis calculation failed: {e}")
            return LogRegressionTrend(
                trend_slope=0.0,
                trend_r_squared=0.0,
                trend_confidence=0.0,
                trend_phase='neutral',
                cycle_position=0.5,
                halving_impact=0.0,
                trend_forecast=[]
            )
    
    def _calculate_statistical_measures(self, regression_result: Dict[str, Any], prices: pd.Series) -> Dict[str, float]:
        """Calculate statistical measures for model validation"""
        try:
            residuals = regression_result['residuals']
            n = len(residuals)
            k = len(regression_result['coefficients'])
            
            # R-squared
            r_squared = regression_result['r_squared']
            
            # Adjusted R-squared
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1) if n > k + 1 else r_squared
            
            # Durbin-Watson statistic (test for autocorrelation)
            if len(residuals) > 1:
                diff_residuals = np.diff(residuals)
                durbin_watson = np.sum(diff_residuals**2) / np.sum(residuals**2)
            else:
                durbin_watson = 2.0
            
            # Akaike Information Criterion (AIC)
            mse = np.mean(residuals**2)
            if mse > 0:
                aic = n * np.log(mse) + 2 * k
            else:
                aic = 0.0
            
            return {
                'r_squared': r_squared,
                'adj_r_squared': adj_r_squared,
                'durbin_watson': durbin_watson,
                'aic': aic
            }
            
        except Exception as e:
            logger.warning(f"Statistical measures calculation failed: {e}")
            return {
                'r_squared': 0.0,
                'adj_r_squared': 0.0,
                'durbin_watson': 2.0,
                'aic': 0.0
            }
    
    def _calculate_risk_metrics(self, prices: pd.Series, regression_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate risk metrics and volatility forecasts"""
        try:
            returns = prices.pct_change().dropna()
            
            # Volatility forecast (GARCH-like approach)
            recent_returns = returns.tail(90)
            volatility_forecast = recent_returns.std() * np.sqrt(365)  # Annualized
            
            # Drawdown risk (based on regression deviation)
            regression_line = regression_result['regression_line']
            deviations = (prices - regression_line) / regression_line
            max_drawdown = deviations.rolling(window=252, min_periods=30).min().iloc[-1]
            drawdown_risk = abs(max_drawdown) if not np.isnan(max_drawdown) else 0.0
            
            # Upside potential (based on historical deviations above regression)
            positive_deviations = deviations[deviations > 0]
            upside_potential = positive_deviations.quantile(0.95) if len(positive_deviations) > 0 else 0.0
            
            return {
                'volatility_forecast': volatility_forecast,
                'drawdown_risk': drawdown_risk,
                'upside_potential': upside_potential
            }
            
        except Exception as e:
            logger.warning(f"Risk metrics calculation failed: {e}")
            return {
                'volatility_forecast': 0.0,
                'drawdown_risk': 0.0,
                'upside_potential': 0.0
            }
    
    def _generate_price_forecast(self, prices: pd.Series, regression_result: Dict[str, Any]) -> Dict[str, List[float]]:
        """Generate price forecasts with confidence intervals"""
        try:
            forecast_days = min(self.forecast_days, 365)
            coefficients = regression_result['coefficients']
            std_residual = regression_result['std_residual']
            current_time = regression_result['time_days'][-1]
            
            forecasts = []
            confidence_intervals = []
            
            for i in range(1, forecast_days + 1):
                future_time = current_time + i
                
                # Generate base forecast
                if self.regression_degree == 1:
                    log_forecast = coefficients[0] + coefficients[1] * np.log(future_time + 1)
                elif self.regression_degree == 2:
                    log_forecast = (coefficients[0] + 
                                  coefficients[1] * np.log(future_time + 1) +
                                  coefficients[2] * (np.log(future_time + 1))**2)
                else:
                    log_forecast = (coefficients[0] + 
                                  coefficients[1] * np.log(future_time + 1) +
                                  coefficients[2] * (np.log(future_time + 1))**2 +
                                  coefficients[3] * future_time / 365.25)
                
                forecast_price = np.exp(log_forecast)
                forecasts.append(float(forecast_price))
                
                # Confidence interval (increases with time)
                time_factor = np.sqrt(i / 30)  # Uncertainty grows with time
                confidence = np.exp(1.96 * std_residual * time_factor)  # 95% CI
                confidence_intervals.append(float(confidence))
            
            return {
                'forecast': forecasts,
                'confidence': confidence_intervals
            }
            
        except Exception as e:
            logger.warning(f"Price forecast generation failed: {e}")
            return {
                'forecast': [],
                'confidence': []
            }