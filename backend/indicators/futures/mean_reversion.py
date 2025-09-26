"""Mean Reversion Analysis for Futures Trading

This module provides mean reversion analysis for futures trading,
including Bollinger Bands, Z-scores, stationarity tests, and half-life calculations.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
from scipy import stats

# Conditional imports
try:
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("Statsmodels not available. Using simplified stationarity tests.")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Using custom implementations.")

@dataclass
class FuturesData:
    """Data structure for futures market data"""
    prices: List[float]
    returns: List[float]
    volume: List[float]
    open_interest: List[float]
    timestamps: List[datetime]
    high: List[float]
    low: List[float]
    open: List[float]
    close: List[float]
    contract_symbol: str
    underlying_asset: str

@dataclass
class MeanReversionResult:
    """Results from mean reversion analysis"""
    mean_reversion_scores: List[float]
    mean_reversion_signals: List[str]
    z_scores: List[float]
    bollinger_upper: List[float]
    bollinger_middle: List[float]
    bollinger_lower: List[float]
    adf_statistic: float
    adf_pvalue: float
    half_life: float
    reversion_probability: List[float]

class MeanReversionAnalyzer:
    """Mean reversion analysis for futures trading"""
    
    def __init__(self):
        pass
    
    def analyze(self, futures_data: FuturesData) -> MeanReversionResult:
        """Perform comprehensive mean reversion analysis"""
        
        try:
            # Calculate Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
                futures_data.close
            )
            
            # Calculate Z-scores
            z_scores = self._calculate_z_scores(futures_data.close)
            
            # Perform ADF test for stationarity
            adf_stat, adf_pvalue = self._perform_adf_test(futures_data.returns)
            
            # Calculate half-life of mean reversion
            half_life = self._calculate_half_life(futures_data.returns)
            
            # Calculate mean reversion scores
            mean_reversion_scores = self._calculate_mean_reversion_scores(
                z_scores, bb_upper, bb_middle, bb_lower, futures_data.close
            )
            
            # Generate trading signals
            mean_reversion_signals = self._generate_mean_reversion_signals(
                mean_reversion_scores, z_scores
            )
            
            # Calculate reversion probability
            reversion_probability = self._calculate_reversion_probability(
                z_scores, half_life
            )
            
            return MeanReversionResult(
                mean_reversion_scores=mean_reversion_scores,
                mean_reversion_signals=mean_reversion_signals,
                z_scores=z_scores,
                bollinger_upper=bb_upper,
                bollinger_middle=bb_middle,
                bollinger_lower=bb_lower,
                adf_statistic=adf_stat,
                adf_pvalue=adf_pvalue,
                half_life=half_life,
                reversion_probability=reversion_probability
            )
            
        except Exception as e:
            print(f"Error in mean reversion analysis: {e}")
            # Return default results
            n = len(futures_data.close)
            return MeanReversionResult(
                mean_reversion_scores=[0.0] * n,
                mean_reversion_signals=["HOLD"] * n,
                z_scores=[0.0] * n,
                bollinger_upper=futures_data.close.copy(),
                bollinger_middle=futures_data.close.copy(),
                bollinger_lower=futures_data.close.copy(),
                adf_statistic=0.0,
                adf_pvalue=0.5,
                half_life=20.0,
                reversion_probability=[0.5] * n
            )
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, 
                                 std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands"""
        
        if TALIB_AVAILABLE:
            try:
                upper, middle, lower = talib.BBANDS(np.array(prices), 
                                                   timeperiod=period, 
                                                   nbdevup=std_dev, 
                                                   nbdevdn=std_dev)
                return upper.tolist(), middle.tolist(), lower.tolist()
            except Exception:
                pass
        
        # Fallback implementation
        if len(prices) < period:
            return prices.copy(), prices.copy(), prices.copy()
        
        upper_band = []
        middle_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if i < period - 1:
                upper_band.append(prices[i])
                middle_band.append(prices[i])
                lower_band.append(prices[i])
            else:
                period_prices = prices[i-period+1:i+1]
                sma = np.mean(period_prices)
                std = np.std(period_prices)
                
                upper_band.append(sma + std_dev * std)
                middle_band.append(sma)
                lower_band.append(sma - std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    def _calculate_z_scores(self, prices: List[float], window: int = 20) -> List[float]:
        """Calculate rolling Z-scores"""
        
        if len(prices) < window:
            return [0.0] * len(prices)
        
        z_scores = []
        
        for i in range(len(prices)):
            if i < window - 1:
                z_scores.append(0.0)
            else:
                window_prices = prices[i-window+1:i+1]
                mean_price = np.mean(window_prices)
                std_price = np.std(window_prices)
                
                if std_price == 0:
                    z_scores.append(0.0)
                else:
                    z_score = (prices[i] - mean_price) / std_price
                    z_scores.append(z_score)
        
        return z_scores
    
    def _perform_adf_test(self, returns: List[float]) -> Tuple[float, float]:
        """Perform Augmented Dickey-Fuller test for stationarity"""
        
        if STATSMODELS_AVAILABLE and len(returns) > 10:
            try:
                # Remove any NaN or infinite values
                clean_returns = [r for r in returns if np.isfinite(r)]
                
                if len(clean_returns) > 10:
                    result = adfuller(clean_returns)
                    return result[0], result[1]
            except Exception as e:
                print(f"ADF test failed: {e}")
        
        # Fallback: simple variance-based stationarity test
        if len(returns) < 20:
            return 0.0, 0.5
        
        # Split returns into two halves and compare variances
        mid = len(returns) // 2
        first_half_var = np.var(returns[:mid])
        second_half_var = np.var(returns[mid:])
        
        # If variances are similar, assume more stationary
        var_ratio = min(first_half_var, second_half_var) / max(first_half_var, second_half_var)
        
        # Convert to pseudo ADF statistic and p-value
        adf_stat = -2.0 * var_ratio  # More negative = more stationary
        p_value = 1 - var_ratio  # Lower p-value = more stationary
        
        return adf_stat, p_value
    
    def _calculate_half_life(self, returns: List[float]) -> float:
        """Calculate half-life of mean reversion"""
        
        if len(returns) < 10:
            return 20.0  # Default half-life
        
        try:
            # Remove any NaN or infinite values
            clean_returns = [r for r in returns if np.isfinite(r)]
            
            if len(clean_returns) < 10:
                return 20.0
            
            # Calculate lagged returns for regression
            lagged_returns = clean_returns[:-1]
            current_returns = clean_returns[1:]
            
            if len(lagged_returns) == 0:
                return 20.0
            
            # Perform linear regression: r_t = alpha + beta * r_{t-1} + epsilon
            # Half-life = -ln(2) / ln(beta) if beta < 1
            
            # Simple linear regression
            x = np.array(lagged_returns)
            y = np.array(current_returns)
            
            if len(x) == 0 or np.var(x) == 0:
                return 20.0
            
            # Calculate beta (slope)
            beta = np.cov(x, y)[0, 1] / np.var(x)
            
            # Calculate half-life
            if 0 < beta < 1:
                half_life = -np.log(2) / np.log(beta)
                # Cap half-life at reasonable values
                return min(max(half_life, 1.0), 100.0)
            else:
                return 20.0  # Default if no mean reversion detected
                
        except Exception as e:
            print(f"Half-life calculation failed: {e}")
            return 20.0
    
    def _calculate_mean_reversion_scores(self, z_scores: List[float],
                                       bb_upper: List[float],
                                       bb_middle: List[float],
                                       bb_lower: List[float],
                                       prices: List[float]) -> List[float]:
        """Calculate mean reversion scores"""
        
        scores = []
        
        for i in range(len(z_scores)):
            # Z-score component (higher absolute value = stronger reversion signal)
            z_component = -np.tanh(z_scores[i])  # Negative because we expect reversion
            
            # Bollinger Band component
            if prices[i] > bb_upper[i]:
                bb_component = -0.5  # Expect reversion down
            elif prices[i] < bb_lower[i]:
                bb_component = 0.5   # Expect reversion up
            else:
                # Within bands - proportional to distance from middle
                band_width = bb_upper[i] - bb_lower[i]
                if band_width > 0:
                    distance_from_middle = (prices[i] - bb_middle[i]) / band_width
                    bb_component = -distance_from_middle * 0.3
                else:
                    bb_component = 0.0
            
            # Combine components
            combined_score = 0.7 * z_component + 0.3 * bb_component
            scores.append(np.clip(combined_score, -1, 1))
        
        return scores
    
    def _generate_mean_reversion_signals(self, mean_reversion_scores: List[float],
                                       z_scores: List[float]) -> List[str]:
        """Generate trading signals based on mean reversion analysis"""
        
        signals = []
        
        for i, (score, z_score) in enumerate(zip(mean_reversion_scores, z_scores)):
            # Strong signals based on Z-score extremes
            if z_score > 2.0:  # Very overbought
                signals.append("STRONG_SELL")
            elif z_score > 1.5:  # Overbought
                signals.append("SELL")
            elif z_score < -2.0:  # Very oversold
                signals.append("STRONG_BUY")
            elif z_score < -1.5:  # Oversold
                signals.append("BUY")
            else:
                # Use mean reversion score for moderate signals
                if score > 0.3:
                    signals.append("BUY")
                elif score < -0.3:
                    signals.append("SELL")
                else:
                    signals.append("HOLD")
        
        return signals
    
    def _calculate_reversion_probability(self, z_scores: List[float], 
                                       half_life: float) -> List[float]:
        """Calculate probability of mean reversion"""
        
        probabilities = []
        
        for z_score in z_scores:
            # Base probability on Z-score magnitude and half-life
            # Higher absolute Z-score = higher reversion probability
            # Shorter half-life = higher reversion probability
            
            z_magnitude = abs(z_score)
            
            # Z-score contribution (sigmoid function)
            z_prob = 2 / (1 + np.exp(-z_magnitude)) - 1
            
            # Half-life contribution (shorter half-life = higher probability)
            half_life_factor = 1 / (1 + half_life / 10)
            
            # Combine factors
            combined_prob = 0.5 + 0.3 * z_prob + 0.2 * half_life_factor
            
            probabilities.append(np.clip(combined_prob, 0, 1))
        
        return probabilities

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    import random
    from datetime import timedelta
    
    n_periods = 100
    base_price = 100.0
    
    # Generate mean-reverting price series
    prices = [base_price]
    mean_price = base_price
    
    for _ in range(n_periods - 1):
        # Mean reversion with some noise
        reversion_force = 0.1 * (mean_price - prices[-1]) / mean_price
        noise = random.uniform(-0.02, 0.02)
        change = reversion_force + noise
        prices.append(prices[-1] * (1 + change))
    
    # Generate OHLC data
    high = [p * (1 + abs(random.uniform(0, 0.01))) for p in prices]
    low = [p * (1 - abs(random.uniform(0, 0.01))) for p in prices]
    open_prices = [prices[0]] + prices[:-1]
    
    # Generate other data
    volume = [random.uniform(1000, 10000) for _ in range(n_periods)]
    open_interest = [random.uniform(5000, 15000) for _ in range(n_periods)]
    returns = [(prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0 
              for i in range(n_periods)]
    
    timestamps = [datetime.now() + timedelta(days=i) for i in range(n_periods)]
    
    # Create FuturesData
    futures_data = FuturesData(
        prices=prices,
        returns=returns,
        volume=volume,
        open_interest=open_interest,
        timestamps=timestamps,
        high=high,
        low=low,
        open=open_prices,
        close=prices,
        contract_symbol="TEST_2024_03",
        underlying_asset="Test Asset"
    )
    
    # Test mean reversion analysis
    analyzer = MeanReversionAnalyzer()
    results = analyzer.analyze(futures_data)
    
    print(f"Mean Reversion Analysis Results:")
    print(f"ADF Statistic: {results.adf_statistic:.4f}")
    print(f"ADF P-value: {results.adf_pvalue:.4f}")
    print(f"Half-life: {results.half_life:.2f} periods")
    print(f"Current Z-score: {results.z_scores[-1]:.3f}")
    print(f"Current Mean Reversion Score: {results.mean_reversion_scores[-1]:.3f}")
    print(f"Current Signal: {results.mean_reversion_signals[-1]}")
    print(f"Reversion Probability: {results.reversion_probability[-1]:.1%}")