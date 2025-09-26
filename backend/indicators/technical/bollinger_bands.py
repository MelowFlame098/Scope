"""Bollinger Bands Indicator

Bollinger Bands consist of a middle band (SMA) and two outer bands that are
standard deviations away from the middle band. They help identify overbought
and oversold conditions, volatility, and potential breakout points.

Author: Assistant
Date: 2024
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from scipy import stats


class AssetType(Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    INDEX = "index"
    CROSS_ASSET = "cross_asset"


class IndicatorCategory(Enum):
    TECHNICAL = "technical"
    VOLATILITY = "volatility"
    TREND = "trend"


@dataclass
class BollingerBandsResult:
    """Result of Bollinger Bands calculation"""
    name: str
    upper_band: List[float]
    middle_band: List[float]
    lower_band: List[float]
    bandwidth: List[float]
    percent_b: List[float]
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class BollingerBandsIndicator:
    """Bollinger Bands Calculator with Advanced Analytics"""
    
    def __init__(self, period: int = 20, std_dev: float = 2.0, ma_type: str = 'SMA'):
        """
        Initialize Bollinger Bands calculator
        
        Args:
            period: Period for moving average calculation (default: 20)
            std_dev: Number of standard deviations for bands (default: 2.0)
            ma_type: Type of moving average ('SMA', 'EMA') (default: 'SMA')
        """
        self.period = period
        self.std_dev = std_dev
        self.ma_type = ma_type.upper()
        
        if self.ma_type not in ['SMA', 'EMA']:
            raise ValueError("ma_type must be 'SMA' or 'EMA'")
    
    def calculate(self, prices: List[float], high: Optional[List[float]] = None, 
                 low: Optional[List[float]] = None, asset_type: AssetType = AssetType.STOCK) -> BollingerBandsResult:
        """
        Calculate Bollinger Bands for given price series
        
        Args:
            prices: List of price values (typically close prices)
            high: List of high prices (optional, for enhanced analysis)
            low: List of low prices (optional, for enhanced analysis)
            asset_type: Type of asset being analyzed
            
        Returns:
            BollingerBandsResult containing bands and analysis
        """
        if len(prices) < self.period:
            # Not enough data
            upper_band = prices.copy()
            middle_band = prices.copy()
            lower_band = prices.copy()
            bandwidth = [0.0] * len(prices)
            percent_b = [0.5] * len(prices)
            signals = ["INSUFFICIENT_DATA"]
            confidence = 0.1
        else:
            upper_band, middle_band, lower_band = self._calculate_bands(prices)
            bandwidth = self._calculate_bandwidth(upper_band, middle_band, lower_band)
            percent_b = self._calculate_percent_b(prices, upper_band, lower_band)
            signals = self._generate_signals(prices, upper_band, middle_band, lower_band, 
                                           bandwidth, percent_b, high, low)
            confidence = min(0.95, 0.5 + (len(prices) - self.period) * 0.01)
        
        return BollingerBandsResult(
            name="Bollinger Bands",
            upper_band=upper_band,
            middle_band=middle_band,
            lower_band=lower_band,
            bandwidth=bandwidth,
            percent_b=percent_b,
            metadata={
                'period': self.period,
                'std_dev': self.std_dev,
                'ma_type': self.ma_type,
                'current_price': prices[-1] if prices else 0,
                'current_upper': upper_band[-1] if upper_band else 0,
                'current_middle': middle_band[-1] if middle_band else 0,
                'current_lower': lower_band[-1] if lower_band else 0,
                'current_bandwidth': bandwidth[-1] if bandwidth else 0,
                'current_percent_b': percent_b[-1] if percent_b else 0.5,
                'volatility_regime': self._analyze_volatility_regime(bandwidth),
                'squeeze_analysis': self._detect_squeeze(bandwidth),
                'breakout_potential': self._assess_breakout_potential(prices, bandwidth, percent_b),
                'mean_reversion_signals': self._analyze_mean_reversion(percent_b),
                'trend_analysis': self._analyze_trend(prices, middle_band)
            },
            confidence=confidence,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.VOLATILITY,
            signals=signals
        )
    
    def _calculate_sma(self, prices: List[float]) -> List[float]:
        """Calculate Simple Moving Average"""
        sma = []
        for i in range(len(prices)):
            if i < self.period - 1:
                # Use expanding window for initial values
                sma.append(np.mean(prices[:i+1]))
            else:
                sma.append(np.mean(prices[i-self.period+1:i+1]))
        return sma
    
    def _calculate_ema(self, prices: List[float]) -> List[float]:
        """Calculate Exponential Moving Average"""
        alpha = 2.0 / (self.period + 1)
        ema = [prices[0]]  # Initialize with first price
        
        for i in range(1, len(prices)):
            ema_value = alpha * prices[i] + (1 - alpha) * ema[i-1]
            ema.append(ema_value)
        
        return ema
    
    def _calculate_rolling_std(self, prices: List[float], ma: List[float]) -> List[float]:
        """Calculate rolling standard deviation"""
        std_values = []
        
        for i in range(len(prices)):
            if i < self.period - 1:
                # Use expanding window for initial values
                window_prices = prices[:i+1]
                window_mean = ma[i]
            else:
                window_prices = prices[i-self.period+1:i+1]
                window_mean = ma[i]
            
            # Calculate standard deviation
            variance = np.mean([(p - window_mean) ** 2 for p in window_prices])
            std_values.append(np.sqrt(variance))
        
        return std_values
    
    def _calculate_bands(self, prices: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands"""
        # Calculate moving average
        if self.ma_type == 'SMA':
            middle_band = self._calculate_sma(prices)
        else:  # EMA
            middle_band = self._calculate_ema(prices)
        
        # Calculate rolling standard deviation
        std_values = self._calculate_rolling_std(prices, middle_band)
        
        # Calculate upper and lower bands
        upper_band = [ma + (self.std_dev * std) for ma, std in zip(middle_band, std_values)]
        lower_band = [ma - (self.std_dev * std) for ma, std in zip(middle_band, std_values)]
        
        return upper_band, middle_band, lower_band
    
    def _calculate_bandwidth(self, upper_band: List[float], middle_band: List[float], 
                           lower_band: List[float]) -> List[float]:
        """Calculate Bollinger Band Width (BBW)"""
        bandwidth = []
        for upper, middle, lower in zip(upper_band, middle_band, lower_band):
            if middle != 0:
                bw = (upper - lower) / middle
            else:
                bw = 0
            bandwidth.append(bw)
        return bandwidth
    
    def _calculate_percent_b(self, prices: List[float], upper_band: List[float], 
                           lower_band: List[float]) -> List[float]:
        """Calculate %B (position within bands)"""
        percent_b = []
        for price, upper, lower in zip(prices, upper_band, lower_band):
            if upper != lower:
                pb = (price - lower) / (upper - lower)
            else:
                pb = 0.5
            percent_b.append(pb)
        return percent_b
    
    def _generate_signals(self, prices: List[float], upper_band: List[float], 
                         middle_band: List[float], lower_band: List[float],
                         bandwidth: List[float], percent_b: List[float],
                         high: Optional[List[float]] = None, 
                         low: Optional[List[float]] = None) -> List[str]:
        """Generate trading signals based on Bollinger Bands"""
        if len(prices) < 2:
            return ["NO_SIGNAL"]
        
        signals = []
        current_price = prices[-1]
        current_upper = upper_band[-1]
        current_middle = middle_band[-1]
        current_lower = lower_band[-1]
        current_percent_b = percent_b[-1]
        
        # Band touch/penetration signals
        if current_price >= current_upper:
            signals.append("UPPER_BAND_TOUCH")
        elif current_price <= current_lower:
            signals.append("LOWER_BAND_TOUCH")
        
        # %B signals
        if current_percent_b >= 1.0:
            signals.append("OVERBOUGHT")
        elif current_percent_b <= 0.0:
            signals.append("OVERSOLD")
        elif current_percent_b >= 0.8:
            signals.append("APPROACHING_OVERBOUGHT")
        elif current_percent_b <= 0.2:
            signals.append("APPROACHING_OVERSOLD")
        
        # Middle band crossover
        if len(prices) >= 2:
            if prices[-2] < middle_band[-2] and current_price > current_middle:
                signals.append("BULLISH_MIDDLE_CROSS")
            elif prices[-2] > middle_band[-2] and current_price < current_middle:
                signals.append("BEARISH_MIDDLE_CROSS")
        
        # Squeeze detection
        if len(bandwidth) >= 20:
            recent_bw = bandwidth[-20:]
            current_bw = bandwidth[-1]
            avg_bw = np.mean(recent_bw)
            
            if current_bw < avg_bw * 0.5:
                signals.append("SQUEEZE")
            elif current_bw > avg_bw * 1.5:
                signals.append("EXPANSION")
        
        # Breakout signals
        if len(prices) >= 3 and len(bandwidth) >= 3:
            # Check for breakout after squeeze
            if (bandwidth[-3] < bandwidth[-2] < bandwidth[-1] and 
                abs(current_percent_b - 0.5) > 0.3):
                if current_percent_b > 0.5:
                    signals.append("BULLISH_BREAKOUT")
                else:
                    signals.append("BEARISH_BREAKOUT")
        
        # Walking the bands
        if len(percent_b) >= 5:
            recent_pb = percent_b[-5:]
            if all(pb > 0.8 for pb in recent_pb[-3:]):
                signals.append("WALKING_UPPER_BAND")
            elif all(pb < 0.2 for pb in recent_pb[-3:]):
                signals.append("WALKING_LOWER_BAND")
        
        return signals if signals else ["NEUTRAL"]
    
    def _analyze_volatility_regime(self, bandwidth: List[float]) -> Dict[str, Any]:
        """Analyze current volatility regime"""
        if len(bandwidth) < 20:
            return {'regime': 'UNKNOWN', 'percentile': 50}
        
        recent_bw = bandwidth[-20:]
        current_bw = bandwidth[-1]
        
        # Calculate percentile of current bandwidth
        percentile = stats.percentileofscore(recent_bw, current_bw)
        
        if percentile >= 80:
            regime = "HIGH_VOLATILITY"
        elif percentile >= 60:
            regime = "MODERATE_HIGH_VOLATILITY"
        elif percentile >= 40:
            regime = "NORMAL_VOLATILITY"
        elif percentile >= 20:
            regime = "MODERATE_LOW_VOLATILITY"
        else:
            regime = "LOW_VOLATILITY"
        
        return {
            'regime': regime,
            'percentile': percentile,
            'current_bandwidth': current_bw,
            'average_bandwidth': np.mean(recent_bw)
        }
    
    def _detect_squeeze(self, bandwidth: List[float]) -> Dict[str, Any]:
        """Detect Bollinger Band squeeze conditions"""
        if len(bandwidth) < 20:
            return {'in_squeeze': False, 'duration': 0}
        
        # Calculate squeeze threshold (lowest 20% of recent bandwidth values)
        recent_bw = bandwidth[-50:] if len(bandwidth) >= 50 else bandwidth
        squeeze_threshold = np.percentile(recent_bw, 20)
        
        # Count consecutive periods in squeeze
        squeeze_duration = 0
        for i in range(len(bandwidth) - 1, -1, -1):
            if bandwidth[i] <= squeeze_threshold:
                squeeze_duration += 1
            else:
                break
        
        in_squeeze = bandwidth[-1] <= squeeze_threshold
        
        return {
            'in_squeeze': in_squeeze,
            'duration': squeeze_duration,
            'threshold': squeeze_threshold,
            'current_bandwidth': bandwidth[-1]
        }
    
    def _assess_breakout_potential(self, prices: List[float], bandwidth: List[float], 
                                  percent_b: List[float]) -> Dict[str, Any]:
        """Assess potential for breakout"""
        if len(prices) < 10:
            return {'potential': 'UNKNOWN', 'direction': 'UNKNOWN'}
        
        # Check for squeeze followed by expansion
        squeeze_info = self._detect_squeeze(bandwidth)
        recent_pb = percent_b[-5:] if len(percent_b) >= 5 else percent_b
        
        # Assess breakout potential
        if squeeze_info['in_squeeze'] and squeeze_info['duration'] >= 5:
            potential = "HIGH"
        elif squeeze_info['duration'] >= 3:
            potential = "MODERATE"
        else:
            potential = "LOW"
        
        # Determine likely direction
        avg_pb = np.mean(recent_pb)
        if avg_pb > 0.6:
            direction = "BULLISH"
        elif avg_pb < 0.4:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"
        
        return {
            'potential': potential,
            'direction': direction,
            'squeeze_duration': squeeze_info['duration'],
            'average_percent_b': avg_pb
        }
    
    def _analyze_mean_reversion(self, percent_b: List[float]) -> Dict[str, Any]:
        """Analyze mean reversion opportunities"""
        if len(percent_b) < 5:
            return {'signal': 'NONE', 'strength': 0}
        
        current_pb = percent_b[-1]
        recent_pb = percent_b[-5:]
        
        # Check for extreme %B values with reversal potential
        if current_pb >= 0.95:
            # Check if %B is declining from extreme high
            if len(recent_pb) >= 3 and recent_pb[-1] < recent_pb[-2]:
                return {'signal': 'BEARISH_REVERSION', 'strength': current_pb}
        elif current_pb <= 0.05:
            # Check if %B is rising from extreme low
            if len(recent_pb) >= 3 and recent_pb[-1] > recent_pb[-2]:
                return {'signal': 'BULLISH_REVERSION', 'strength': 1 - current_pb}
        
        return {'signal': 'NONE', 'strength': 0}
    
    def _analyze_trend(self, prices: List[float], middle_band: List[float]) -> Dict[str, Any]:
        """Analyze trend using middle band"""
        if len(prices) < 10 or len(middle_band) < 10:
            return {'direction': 'UNKNOWN', 'strength': 0}
        
        # Calculate trend of middle band
        recent_mb = middle_band[-10:]
        mb_trend = np.polyfit(range(len(recent_mb)), recent_mb, 1)[0]
        
        # Calculate price position relative to middle band
        recent_prices = prices[-10:]
        above_mb = sum(1 for p, mb in zip(recent_prices, recent_mb) if p > mb)
        
        # Determine trend direction and strength
        if mb_trend > 0 and above_mb >= 7:
            return {'direction': 'STRONG_BULLISH', 'strength': mb_trend, 'position_score': above_mb/10}
        elif mb_trend > 0 and above_mb >= 5:
            return {'direction': 'BULLISH', 'strength': mb_trend, 'position_score': above_mb/10}
        elif mb_trend < 0 and above_mb <= 3:
            return {'direction': 'STRONG_BEARISH', 'strength': abs(mb_trend), 'position_score': above_mb/10}
        elif mb_trend < 0 and above_mb <= 5:
            return {'direction': 'BEARISH', 'strength': abs(mb_trend), 'position_score': above_mb/10}
        else:
            return {'direction': 'SIDEWAYS', 'strength': abs(mb_trend), 'position_score': above_mb/10}
    
    def get_chart_data(self, result: BollingerBandsResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'bollinger_bands',
            'name': 'Bollinger Bands',
            'data': {
                'upper': result.upper_band,
                'middle': result.middle_band,
                'lower': result.lower_band,
                'bandwidth': result.bandwidth,
                'percent_b': result.percent_b
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'series': [
                {
                    'name': 'Upper Band',
                    'data': result.upper_band,
                    'color': '#FF5722',
                    'type': 'line',
                    'lineWidth': 1
                },
                {
                    'name': 'Middle Band',
                    'data': result.middle_band,
                    'color': '#2196F3',
                    'type': 'line',
                    'lineWidth': 2
                },
                {
                    'name': 'Lower Band',
                    'data': result.lower_band,
                    'color': '#4CAF50',
                    'type': 'line',
                    'lineWidth': 1
                }
            ],
            'fill': {
                'between': ['upper', 'lower'],
                'color': 'rgba(33, 150, 243, 0.1)'
            }
        }


# Example usage
if __name__ == "__main__":
    # Sample price data with some volatility
    sample_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121, 123, 122, 124, 126, 125, 127, 129, 128]
    
    # Calculate Bollinger Bands
    bb_calculator = BollingerBandsIndicator(period=20, std_dev=2.0)
    result = bb_calculator.calculate(sample_prices, asset_type=AssetType.STOCK)
    
    print(f"Bollinger Bands Analysis:")
    print(f"Current Price: {result.metadata['current_price']:.2f}")
    print(f"Upper Band: {result.metadata['current_upper']:.2f}")
    print(f"Middle Band: {result.metadata['current_middle']:.2f}")
    print(f"Lower Band: {result.metadata['current_lower']:.2f}")
    print(f"Bandwidth: {result.metadata['current_bandwidth']:.4f}")
    print(f"%B: {result.metadata['current_percent_b']:.2f}")
    print(f"Volatility Regime: {result.metadata['volatility_regime']['regime']}")
    print(f"Signals: {', '.join(result.signals)}")