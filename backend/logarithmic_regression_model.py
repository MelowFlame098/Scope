import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class CryptoIndicatorResult:
    """Result container for crypto indicator calculations"""
    indicator_name: str
    value: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    signal: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1 signal strength

class LogarithmicRegressionModel:
    """Logarithmic regression model for long-term price analysis"""
    
    def calculate_log_regression(self, 
                                prices: List[float],
                                timestamps: List[datetime],
                                degree: int = 2) -> CryptoIndicatorResult:
        """Calculate logarithmic regression bands"""
        try:
            if len(prices) != len(timestamps) or len(prices) < 10:
                raise ValueError("Need at least 10 price points with matching timestamps")
                
            # Convert to numpy arrays
            prices_array = np.array(prices)
            log_prices = np.log(prices_array)
            
            # Create time series (days since first timestamp)
            time_series = np.array([(ts - timestamps[0]).days for ts in timestamps])
            
            # Polynomial regression on log prices
            coefficients = np.polyfit(time_series, log_prices, degree)
            poly_func = np.poly1d(coefficients)
            
            # Calculate regression line
            regression_line = np.exp(poly_func(time_series))
            
            # Calculate standard deviation for bands
            residuals = log_prices - poly_func(time_series)
            std_dev = np.std(residuals)
            
            # Create bands
            upper_band = np.exp(poly_func(time_series) + 2 * std_dev)
            lower_band = np.exp(poly_func(time_series) - 2 * std_dev)
            
            # Current position relative to bands
            current_price = prices[-1]
            current_regression = regression_line[-1]
            current_upper = upper_band[-1]
            current_lower = lower_band[-1]
            
            # Calculate position ratio
            band_width = current_upper - current_lower
            position_in_band = (current_price - current_lower) / band_width
            
            # Generate signals
            if position_in_band < 0.2:  # Near lower band
                signal = 'buy'
                strength = 1.0 - position_in_band / 0.2
            elif position_in_band > 0.8:  # Near upper band
                signal = 'sell'
                strength = (position_in_band - 0.8) / 0.2
            else:
                signal = 'hold'
                strength = 0.5
                
            # R-squared for confidence
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((log_prices - np.mean(log_prices)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            confidence = min(r_squared, 0.95)
            
            return CryptoIndicatorResult(
                indicator_name='Logarithmic Regression',
                value=position_in_band,
                confidence=confidence,
                timestamp=datetime.now(),
                metadata={
                    'current_price': current_price,
                    'regression_value': current_regression,
                    'upper_band': current_upper,
                    'lower_band': current_lower,
                    'r_squared': r_squared,
                    'degree': degree,
                    'coefficients': coefficients.tolist()
                },
                signal=signal,
                strength=strength
            )
            
        except Exception as e:
            logger.error(f"Error calculating logarithmic regression: {e}")
            return self._error_result('Logarithmic Regression', str(e))
    
    def _error_result(self, indicator_name: str, error_msg: str) -> CryptoIndicatorResult:
        return CryptoIndicatorResult(
            indicator_name=indicator_name,
            value=0.0,
            confidence=0.0,
            timestamp=datetime.now(),
            metadata={'error': error_msg},
            signal='hold',
            strength=0.0
        )