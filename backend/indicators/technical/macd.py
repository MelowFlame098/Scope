"""Moving Average Convergence Divergence (MACD) Indicator

MACD is a trend-following momentum indicator that shows the relationship between
two moving averages of a security's price. It consists of MACD line, signal line,
and histogram, providing multiple signals for trend analysis.

Author: Assistant
Date: 2024
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class AssetType(Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    INDEX = "index"
    CROSS_ASSET = "cross_asset"


class IndicatorCategory(Enum):
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    TREND = "trend"


@dataclass
class MACDResult:
    """Result of MACD calculation"""
    name: str
    macd_line: List[float]
    signal_line: List[float]
    histogram: List[float]
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class MACDIndicator:
    """Moving Average Convergence Divergence (MACD) Calculator"""
    
    def __init__(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
        """
        Initialize MACD calculator
        
        Args:
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
    
    def calculate(self, prices: List[float], asset_type: AssetType = AssetType.STOCK) -> MACDResult:
        """
        Calculate MACD for given price series
        
        Args:
            prices: List of price values
            asset_type: Type of asset being analyzed
            
        Returns:
            MACDResult containing MACD components and analysis
        """
        if len(prices) < self.slow_period:
            # Not enough data
            macd_line = [0.0] * len(prices)
            signal_line = [0.0] * len(prices)
            histogram = [0.0] * len(prices)
            signals = ["INSUFFICIENT_DATA"]
            confidence = 0.1
        else:
            macd_line, signal_line, histogram = self._calculate_macd(prices)
            signals = self._generate_signals(macd_line, signal_line, histogram)
            confidence = min(0.95, 0.4 + (len(prices) - self.slow_period) * 0.01)
        
        return MACDResult(
            name="MACD",
            macd_line=macd_line,
            signal_line=signal_line,
            histogram=histogram,
            metadata={
                'fast_period': self.fast_period,
                'slow_period': self.slow_period,
                'signal_period': self.signal_period,
                'current_macd': macd_line[-1] if macd_line else 0.0,
                'current_signal': signal_line[-1] if signal_line else 0.0,
                'current_histogram': histogram[-1] if histogram else 0.0,
                'trend': self._determine_trend(macd_line, signal_line),
                'momentum': self._analyze_momentum(histogram),
                'crossover_points': self._find_crossovers(macd_line, signal_line),
                'divergence': self._check_divergence(prices, macd_line)
            },
            confidence=confidence,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.MOMENTUM,
            signals=signals
        )
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average with proper initialization"""
        if len(prices) < period:
            return [prices[0]] * len(prices) if prices else []
        
        alpha = 2.0 / (period + 1)
        ema = []
        
        # Initialize with SMA of first 'period' values
        sma_init = np.mean(prices[:period])
        ema.extend([sma_init] * period)
        
        # Calculate EMA for remaining values
        for i in range(period, len(prices)):
            ema_value = alpha * prices[i] + (1 - alpha) * ema[i-1]
            ema.append(ema_value)
        
        return ema
    
    def _calculate_macd(self, prices: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """Calculate MACD components"""
        # Calculate EMAs
        fast_ema = self._calculate_ema(prices, self.fast_period)
        slow_ema = self._calculate_ema(prices, self.slow_period)
        
        # MACD line = Fast EMA - Slow EMA
        macd_line = [fast - slow for fast, slow in zip(fast_ema, slow_ema)]
        
        # Signal line = EMA of MACD line
        signal_line = self._calculate_ema(macd_line, self.signal_period)
        
        # Histogram = MACD line - Signal line
        histogram = [macd - signal for macd, signal in zip(macd_line, signal_line)]
        
        return macd_line, signal_line, histogram
    
    def _generate_signals(self, macd_line: List[float], signal_line: List[float], histogram: List[float]) -> List[str]:
        """Generate trading signals based on MACD"""
        if len(macd_line) < 2 or len(signal_line) < 2:
            return ["NO_SIGNAL"]
        
        signals = []
        
        # MACD line crossovers
        if len(macd_line) >= 2 and len(signal_line) >= 2:
            if macd_line[-2] <= signal_line[-2] and macd_line[-1] > signal_line[-1]:
                signals.append("BULLISH_CROSSOVER")
            elif macd_line[-2] >= signal_line[-2] and macd_line[-1] < signal_line[-1]:
                signals.append("BEARISH_CROSSOVER")
        
        # Zero line crossovers
        if len(macd_line) >= 2:
            if macd_line[-2] <= 0 < macd_line[-1]:
                signals.append("BULLISH_ZERO_CROSS")
            elif macd_line[-2] >= 0 > macd_line[-1]:
                signals.append("BEARISH_ZERO_CROSS")
        
        # Histogram analysis
        if len(histogram) >= 3:
            # Histogram turning points
            if histogram[-3] < histogram[-2] > histogram[-1] and histogram[-2] > 0:
                signals.append("BEARISH_MOMENTUM_PEAK")
            elif histogram[-3] > histogram[-2] < histogram[-1] and histogram[-2] < 0:
                signals.append("BULLISH_MOMENTUM_TROUGH")
            
            # Histogram trend
            recent_hist_trend = np.polyfit(range(3), histogram[-3:], 1)[0]
            if recent_hist_trend > 0.001:
                signals.append("INCREASING_MOMENTUM")
            elif recent_hist_trend < -0.001:
                signals.append("DECREASING_MOMENTUM")
        
        # Overall position analysis
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        current_histogram = histogram[-1]
        
        if current_macd > current_signal and current_macd > 0:
            signals.append("BULLISH_POSITION")
        elif current_macd < current_signal and current_macd < 0:
            signals.append("BEARISH_POSITION")
        
        return signals if signals else ["NEUTRAL"]
    
    def _determine_trend(self, macd_line: List[float], signal_line: List[float]) -> str:
        """Determine overall trend based on MACD"""
        if len(macd_line) < 5:
            return "INSUFFICIENT_DATA"
        
        current_macd = macd_line[-1]
        current_signal = signal_line[-1]
        
        # Analyze recent MACD trend
        recent_macd = macd_line[-5:]
        macd_trend = np.polyfit(range(len(recent_macd)), recent_macd, 1)[0]
        
        # Determine trend strength and direction
        if current_macd > current_signal:
            if macd_trend > 0.01:
                return "STRONG_BULLISH"
            elif macd_trend > 0:
                return "BULLISH"
            else:
                return "WEAKENING_BULLISH"
        else:
            if macd_trend < -0.01:
                return "STRONG_BEARISH"
            elif macd_trend < 0:
                return "BEARISH"
            else:
                return "WEAKENING_BEARISH"
    
    def _analyze_momentum(self, histogram: List[float]) -> Dict[str, Any]:
        """Analyze momentum based on histogram"""
        if len(histogram) < 5:
            return {'strength': 'UNKNOWN', 'direction': 'UNKNOWN', 'change': 0}
        
        recent_hist = histogram[-5:]
        current_hist = histogram[-1]
        
        # Calculate momentum change
        momentum_change = np.polyfit(range(len(recent_hist)), recent_hist, 1)[0]
        
        # Determine momentum strength
        abs_momentum = abs(current_hist)
        if abs_momentum > 0.5:
            strength = "STRONG"
        elif abs_momentum > 0.2:
            strength = "MODERATE"
        elif abs_momentum > 0.05:
            strength = "WEAK"
        else:
            strength = "MINIMAL"
        
        # Determine direction
        if current_hist > 0.05:
            direction = "BULLISH"
        elif current_hist < -0.05:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        return {
            'strength': strength,
            'direction': direction,
            'change': momentum_change,
            'current_value': current_hist
        }
    
    def _find_crossovers(self, macd_line: List[float], signal_line: List[float]) -> List[Dict[str, Any]]:
        """Find recent crossover points"""
        crossovers = []
        
        if len(macd_line) < 2 or len(signal_line) < 2:
            return crossovers
        
        # Look for crossovers in recent data (last 20 periods)
        start_idx = max(0, len(macd_line) - 20)
        
        for i in range(start_idx + 1, len(macd_line)):
            if (macd_line[i-1] <= signal_line[i-1] and macd_line[i] > signal_line[i]):
                crossovers.append({
                    'type': 'BULLISH',
                    'index': i,
                    'macd_value': macd_line[i],
                    'signal_value': signal_line[i]
                })
            elif (macd_line[i-1] >= signal_line[i-1] and macd_line[i] < signal_line[i]):
                crossovers.append({
                    'type': 'BEARISH',
                    'index': i,
                    'macd_value': macd_line[i],
                    'signal_value': signal_line[i]
                })
        
        return crossovers[-5:]  # Return last 5 crossovers
    
    def _check_divergence(self, prices: List[float], macd_line: List[float]) -> Dict[str, Any]:
        """Check for price-MACD divergence"""
        if len(prices) < 10 or len(macd_line) < 10:
            return {'type': 'NONE', 'strength': 0}
        
        # Get recent data
        recent_prices = prices[-10:]
        recent_macd = macd_line[-10:]
        
        # Calculate trends
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        macd_trend = np.polyfit(range(len(recent_macd)), recent_macd, 1)[0]
        
        # Normalize trends for comparison
        price_trend_norm = price_trend / np.mean(recent_prices)
        macd_trend_norm = macd_trend / (np.mean(np.abs(recent_macd)) + 1e-8)
        
        # Check for divergence
        divergence_threshold = 0.001
        
        if price_trend_norm > divergence_threshold and macd_trend_norm < -divergence_threshold:
            return {
                'type': 'BEARISH',
                'strength': abs(macd_trend_norm),
                'price_trend': price_trend_norm,
                'macd_trend': macd_trend_norm
            }
        elif price_trend_norm < -divergence_threshold and macd_trend_norm > divergence_threshold:
            return {
                'type': 'BULLISH',
                'strength': macd_trend_norm,
                'price_trend': price_trend_norm,
                'macd_trend': macd_trend_norm
            }
        else:
            return {'type': 'NONE', 'strength': 0}
    
    def get_chart_data(self, result: MACDResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'macd',
            'name': 'MACD',
            'data': {
                'macd': result.macd_line,
                'signal': result.signal_line,
                'histogram': result.histogram
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'yAxis': {
                'title': 'MACD',
                'plotLines': [{'value': 0, 'color': '#666', 'width': 1}]
            },
            'series': [
                {
                    'name': 'MACD',
                    'data': result.macd_line,
                    'color': '#2196F3',
                    'type': 'line'
                },
                {
                    'name': 'Signal',
                    'data': result.signal_line,
                    'color': '#FF9800',
                    'type': 'line'
                },
                {
                    'name': 'Histogram',
                    'data': result.histogram,
                    'color': '#4CAF50',
                    'type': 'column'
                }
            ]
        }


# Example usage
if __name__ == "__main__":
    # Sample price data
    sample_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121, 123, 122, 124, 126, 125]
    
    # Calculate MACD
    macd_calculator = MACDIndicator()
    result = macd_calculator.calculate(sample_prices, AssetType.STOCK)
    
    print(f"MACD Analysis:")
    print(f"Current MACD: {result.metadata['current_macd']:.4f}")
    print(f"Current Signal: {result.metadata['current_signal']:.4f}")
    print(f"Current Histogram: {result.metadata['current_histogram']:.4f}")
    print(f"Trend: {result.metadata['trend']}")
    print(f"Momentum: {result.metadata['momentum']}")
    print(f"Signals: {', '.join(result.signals)}")