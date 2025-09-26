"""Technical Indicators for Cross-Asset Analysis

This module implements technical indicators calculation for multiple assets including:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Ichimoku Cloud
- Bollinger Bands
- Stochastic Oscillator

Author: Assistant
Date: 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class CrossAssetData:
    """Cross-asset data structure"""
    asset_prices: Dict[str, List[float]]  # Asset symbol -> price series
    asset_returns: Dict[str, List[float]]  # Asset symbol -> return series
    timestamps: List[datetime]
    volume: Dict[str, List[float]]  # Asset symbol -> volume series
    market_data: Dict[str, Any]  # Additional market data
    news_sentiment: Optional[List[float]] = None  # Sentiment scores
    macro_indicators: Optional[Dict[str, List[float]]] = None  # Economic indicators


@dataclass
class TechnicalIndicators:
    """Technical indicators results"""
    rsi: Dict[str, List[float]]
    macd: Dict[str, Dict[str, List[float]]]  # MACD line, signal, histogram
    ichimoku: Dict[str, Dict[str, List[float]]]  # Tenkan, Kijun, Senkou A/B, Chikou
    bollinger_bands: Dict[str, Dict[str, List[float]]]  # Upper, Middle, Lower
    stochastic: Dict[str, Dict[str, List[float]]]  # %K, %D


class TechnicalIndicatorCalculator:
    """Calculate technical indicators for multiple assets"""
    
    def __init__(self):
        pass
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI"""
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = []
        avg_losses = []
        rsi_values = []
        
        # Initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        # Pad with initial values
        return [50.0] * (period + 1) + rsi_values
    
    def calculate_macd(self, prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, List[float]]:
        """Calculate MACD"""
        if len(prices) < slow:
            return {
                'macd': [0.0] * len(prices),
                'signal': [0.0] * len(prices),
                'histogram': [0.0] * len(prices)
            }
        
        prices_array = np.array(prices)
        
        # Calculate EMAs
        ema_fast = self._calculate_ema(prices_array, fast)
        ema_slow = self._calculate_ema(prices_array, slow)
        
        # MACD line
        macd_line = ema_fast - ema_slow
        
        # Signal line
        signal_line = self._calculate_ema(macd_line, signal)
        
        # Histogram
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line.tolist(),
            'signal': signal_line.tolist(),
            'histogram': histogram.tolist()
        }
    
    def calculate_ichimoku(self, prices: List[float], high: List[float], low: List[float]) -> Dict[str, List[float]]:
        """Calculate Ichimoku Cloud"""
        if len(prices) < 52:
            return {
                'tenkan_sen': prices.copy(),
                'kijun_sen': prices.copy(),
                'senkou_span_a': prices.copy(),
                'senkou_span_b': prices.copy(),
                'chikou_span': prices.copy()
            }
        
        high_array = np.array(high)
        low_array = np.array(low)
        
        # Tenkan-sen (9-period)
        tenkan_sen = []
        for i in range(len(prices)):
            if i < 8:
                tenkan_sen.append(prices[i])
            else:
                period_high = np.max(high_array[i-8:i+1])
                period_low = np.min(low_array[i-8:i+1])
                tenkan_sen.append((period_high + period_low) / 2)
        
        # Kijun-sen (26-period)
        kijun_sen = []
        for i in range(len(prices)):
            if i < 25:
                kijun_sen.append(prices[i])
            else:
                period_high = np.max(high_array[i-25:i+1])
                period_low = np.min(low_array[i-25:i+1])
                kijun_sen.append((period_high + period_low) / 2)
        
        # Senkou Span A
        senkou_span_a = [(t + k) / 2 for t, k in zip(tenkan_sen, kijun_sen)]
        
        # Senkou Span B (52-period)
        senkou_span_b = []
        for i in range(len(prices)):
            if i < 51:
                senkou_span_b.append(prices[i])
            else:
                period_high = np.max(high_array[i-51:i+1])
                period_low = np.min(low_array[i-51:i+1])
                senkou_span_b.append((period_high + period_low) / 2)
        
        # Chikou Span (lagged close)
        chikou_span = [0] * 26 + prices[:-26]
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def calculate_all_indicators(self, data: CrossAssetData) -> TechnicalIndicators:
        """Calculate all technical indicators for all assets"""
        rsi_results = {}
        macd_results = {}
        ichimoku_results = {}
        bollinger_results = {}
        stochastic_results = {}
        
        for asset, prices in data.asset_prices.items():
            # RSI
            rsi_results[asset] = self.calculate_rsi(prices)
            
            # MACD
            macd_results[asset] = self.calculate_macd(prices)
            
            # Ichimoku (using prices as high/low approximation)
            high_prices = [p * 1.01 for p in prices]  # Approximate high
            low_prices = [p * 0.99 for p in prices]   # Approximate low
            ichimoku_results[asset] = self.calculate_ichimoku(prices, high_prices, low_prices)
            
            # Bollinger Bands
            bollinger_results[asset] = self._calculate_bollinger_bands(prices)
            
            # Stochastic
            stochastic_results[asset] = self._calculate_stochastic(prices, high_prices, low_prices)
        
        return TechnicalIndicators(
            rsi=rsi_results,
            macd=macd_results,
            ichimoku=ichimoku_results,
            bollinger_bands=bollinger_results,
            stochastic=stochastic_results
        )
    
    def _calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2) -> Dict[str, List[float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return {
                'upper': prices.copy(),
                'middle': prices.copy(),
                'lower': prices.copy()
            }
        
        prices_array = np.array(prices)
        sma = np.convolve(prices_array, np.ones(period)/period, mode='same')
        
        rolling_std = []
        for i in range(len(prices)):
            if i < period - 1:
                rolling_std.append(np.std(prices_array[:i+1]))
            else:
                rolling_std.append(np.std(prices_array[i-period+1:i+1]))
        
        rolling_std = np.array(rolling_std)
        
        upper_band = sma + (std_dev * rolling_std)
        lower_band = sma - (std_dev * rolling_std)
        
        return {
            'upper': upper_band.tolist(),
            'middle': sma.tolist(),
            'lower': lower_band.tolist()
        }
    
    def _calculate_stochastic(self, prices: List[float], high: List[float], low: List[float], k_period: int = 14, d_period: int = 3) -> Dict[str, List[float]]:
        """Calculate Stochastic Oscillator"""
        if len(prices) < k_period:
            return {
                'percent_k': [50.0] * len(prices),
                'percent_d': [50.0] * len(prices)
            }
        
        percent_k = []
        for i in range(len(prices)):
            if i < k_period - 1:
                percent_k.append(50.0)
            else:
                period_high = max(high[i-k_period+1:i+1])
                period_low = min(low[i-k_period+1:i+1])
                
                if period_high == period_low:
                    k_value = 50.0
                else:
                    k_value = ((prices[i] - period_low) / (period_high - period_low)) * 100
                
                percent_k.append(k_value)
        
        # %D is SMA of %K
        percent_d = []
        for i in range(len(percent_k)):
            if i < d_period - 1:
                percent_d.append(np.mean(percent_k[:i+1]))
            else:
                percent_d.append(np.mean(percent_k[i-d_period+1:i+1]))
        
        return {
            'percent_k': percent_k,
            'percent_d': percent_d
        }


# Example usage
if __name__ == "__main__":
    # Generate sample data
    import random
    from datetime import datetime, timedelta
    
    # Sample cross-asset data
    timestamps = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    
    sample_data = CrossAssetData(
        asset_prices={
            'AAPL': [100 + random.gauss(0, 5) for _ in range(100)],
            'GOOGL': [2000 + random.gauss(0, 50) for _ in range(100)],
            'BTC': [50000 + random.gauss(0, 2000) for _ in range(100)]
        },
        asset_returns={
            'AAPL': [random.gauss(0.001, 0.02) for _ in range(100)],
            'GOOGL': [random.gauss(0.001, 0.025) for _ in range(100)],
            'BTC': [random.gauss(0.002, 0.05) for _ in range(100)]
        },
        timestamps=timestamps,
        volume={
            'AAPL': [1000000 + random.randint(-100000, 100000) for _ in range(100)],
            'GOOGL': [500000 + random.randint(-50000, 50000) for _ in range(100)],
            'BTC': [10000 + random.randint(-1000, 1000) for _ in range(100)]
        },
        market_data={}
    )
    
    # Calculate technical indicators
    calculator = TechnicalIndicatorCalculator()
    indicators = calculator.calculate_all_indicators(sample_data)
    
    print("Technical Indicators Analysis Complete!")
    print(f"Assets analyzed: {list(indicators.rsi.keys())}")
    
    for asset in indicators.rsi.keys():
        print(f"\n{asset}:")
        print(f"  Current RSI: {indicators.rsi[asset][-1]:.2f}")
        print(f"  Current MACD: {indicators.macd[asset]['macd'][-1]:.4f}")
        print(f"  Bollinger Position: {((sample_data.asset_prices[asset][-1] - indicators.bollinger_bands[asset]['lower'][-1]) / (indicators.bollinger_bands[asset]['upper'][-1] - indicators.bollinger_bands[asset]['lower'][-1]) * 100):.1f}%")