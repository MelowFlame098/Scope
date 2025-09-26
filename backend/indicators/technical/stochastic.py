"""Stochastic Oscillator Indicator

The Stochastic Oscillator is a momentum indicator that compares a security's closing price
to its price range over a specific period. It consists of two lines: %K (fast) and %D (slow).
Values range from 0 to 100, with readings above 80 indicating overbought conditions
and readings below 20 indicating oversold conditions.

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
    OSCILLATOR = "oscillator"


@dataclass
class StochasticResult:
    """Result of Stochastic Oscillator calculation"""
    name: str
    k_percent: List[float]  # %K line (fast stochastic)
    d_percent: List[float]  # %D line (slow stochastic)
    j_percent: List[float]  # %J line (3*%K - 2*%D)
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class StochasticIndicator:
    """Stochastic Oscillator Calculator with Advanced Analysis"""
    
    def __init__(self, k_period: int = 14, d_period: int = 3, smooth_k: int = 3,
                 overbought_level: float = 80, oversold_level: float = 20):
        """
        Initialize Stochastic Oscillator calculator
        
        Args:
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D smoothing (default: 3)
            smooth_k: Period for %K smoothing (default: 3)
            overbought_level: Overbought threshold (default: 80)
            oversold_level: Oversold threshold (default: 20)
        """
        self.k_period = k_period
        self.d_period = d_period
        self.smooth_k = smooth_k
        self.overbought_level = overbought_level
        self.oversold_level = oversold_level
    
    def calculate(self, high: List[float], low: List[float], close: List[float], 
                 asset_type: AssetType = AssetType.STOCK) -> StochasticResult:
        """
        Calculate Stochastic Oscillator for given price series
        
        Args:
            high: List of high prices
            low: List of low prices
            close: List of close prices
            asset_type: Type of asset being analyzed
            
        Returns:
            StochasticResult containing %K, %D, %J and analysis
        """
        if len(high) != len(low) or len(low) != len(close):
            raise ValueError("High, low, and close arrays must have the same length")
        
        if len(close) < self.k_period:
            # Not enough data - return simplified calculation
            k_percent = [50.0] * len(close)  # Neutral level
            d_percent = [50.0] * len(close)
            j_percent = [50.0] * len(close)
            signals = ["INSUFFICIENT_DATA"]
            confidence = 0.1
        else:
            # Calculate raw %K
            raw_k = self._calculate_raw_k(high, low, close)
            
            # Smooth %K if required
            if self.smooth_k > 1:
                k_percent = self._smooth_series(raw_k, self.smooth_k)
            else:
                k_percent = raw_k
            
            # Calculate %D (smoothed %K)
            d_percent = self._smooth_series(k_percent, self.d_period)
            
            # Calculate %J
            j_percent = self._calculate_j_percent(k_percent, d_percent)
            
            signals = self._generate_signals(k_percent, d_percent, j_percent)
            confidence = min(0.95, 0.3 + (len(close) - self.k_period) * 0.02)
        
        return StochasticResult(
            name="Stochastic Oscillator",
            k_percent=k_percent,
            d_percent=d_percent,
            j_percent=j_percent,
            metadata={
                'k_period': self.k_period,
                'd_period': self.d_period,
                'smooth_k': self.smooth_k,
                'overbought_level': self.overbought_level,
                'oversold_level': self.oversold_level,
                'current_k': k_percent[-1] if k_percent else 50,
                'current_d': d_percent[-1] if d_percent else 50,
                'current_j': j_percent[-1] if j_percent else 50,
                'market_condition': self._analyze_market_condition(k_percent, d_percent),
                'momentum_analysis': self._analyze_momentum(k_percent, d_percent, j_percent),
                'divergence_analysis': self._analyze_divergence(close, k_percent),
                'crossover_analysis': self._analyze_crossovers(k_percent, d_percent),
                'volatility_analysis': self._analyze_volatility(k_percent, d_percent),
                'trend_analysis': self._analyze_trend(k_percent, d_percent),
                'support_resistance': self._identify_support_resistance(k_percent, d_percent)
            },
            confidence=confidence,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.MOMENTUM,
            signals=signals
        )
    
    def _calculate_raw_k(self, high: List[float], low: List[float], close: List[float]) -> List[float]:
        """Calculate raw %K values"""
        raw_k = []
        
        for i in range(len(close)):
            if i < self.k_period - 1:
                # Use available data for initial values
                period_high = max(high[:i+1])
                period_low = min(low[:i+1])
            else:
                period_high = max(high[i-self.k_period+1:i+1])
                period_low = min(low[i-self.k_period+1:i+1])
            
            # Calculate %K
            if period_high == period_low:
                k_value = 50.0  # Neutral when no range
            else:
                k_value = ((close[i] - period_low) / (period_high - period_low)) * 100
            
            raw_k.append(max(0, min(100, k_value)))  # Clamp between 0 and 100
        
        return raw_k
    
    def _smooth_series(self, series: List[float], period: int) -> List[float]:
        """Apply simple moving average smoothing to a series"""
        if period <= 1:
            return series.copy()
        
        smoothed = []
        
        for i in range(len(series)):
            if i < period - 1:
                # Use available data for initial values
                smoothed.append(sum(series[:i+1]) / (i+1))
            else:
                smoothed.append(sum(series[i-period+1:i+1]) / period)
        
        return smoothed
    
    def _calculate_j_percent(self, k_percent: List[float], d_percent: List[float]) -> List[float]:
        """Calculate %J line (3*%K - 2*%D)"""
        j_percent = []
        
        for k, d in zip(k_percent, d_percent):
            j_value = 3 * k - 2 * d
            j_percent.append(max(-20, min(120, j_value)))  # Allow some overshoot
        
        return j_percent
    
    def _generate_signals(self, k_percent: List[float], d_percent: List[float], 
                         j_percent: List[float]) -> List[str]:
        """Generate trading signals based on Stochastic Oscillator"""
        if len(k_percent) < 2 or len(d_percent) < 2:
            return ["NO_SIGNAL"]
        
        signals = []
        current_k = k_percent[-1]
        current_d = d_percent[-1]
        current_j = j_percent[-1] if j_percent else current_k
        
        # Overbought/Oversold conditions
        if current_k >= self.overbought_level and current_d >= self.overbought_level:
            signals.append("OVERBOUGHT")
        elif current_k <= self.oversold_level and current_d <= self.oversold_level:
            signals.append("OVERSOLD")
        
        # %K and %D crossovers
        if len(k_percent) >= 2 and len(d_percent) >= 2:
            if (k_percent[-2] <= d_percent[-2] and current_k > current_d):
                if current_k <= self.oversold_level:
                    signals.append("BULLISH_CROSSOVER_OVERSOLD")
                else:
                    signals.append("BULLISH_CROSSOVER")
            elif (k_percent[-2] >= d_percent[-2] and current_k < current_d):
                if current_k >= self.overbought_level:
                    signals.append("BEARISH_CROSSOVER_OVERBOUGHT")
                else:
                    signals.append("BEARISH_CROSSOVER")
        
        # %J extreme readings
        if current_j >= 100:
            signals.append("J_EXTREME_HIGH")
        elif current_j <= 0:
            signals.append("J_EXTREME_LOW")
        
        # Momentum signals
        if len(k_percent) >= 5:
            recent_k = k_percent[-5:]
            k_trend = np.polyfit(range(len(recent_k)), recent_k, 1)[0]
            
            if k_trend > 2:  # Strong upward momentum
                signals.append("STRONG_BULLISH_MOMENTUM")
            elif k_trend < -2:  # Strong downward momentum
                signals.append("STRONG_BEARISH_MOMENTUM")
        
        # Divergence signals (simplified)
        if len(k_percent) >= 10:
            recent_k = k_percent[-10:]
            if self._detect_bullish_divergence(recent_k):
                signals.append("BULLISH_DIVERGENCE")
            elif self._detect_bearish_divergence(recent_k):
                signals.append("BEARISH_DIVERGENCE")
        
        # Double bottom/top patterns in oversold/overbought regions
        if self._detect_double_bottom(k_percent, d_percent):
            signals.append("DOUBLE_BOTTOM_OVERSOLD")
        elif self._detect_double_top(k_percent, d_percent):
            signals.append("DOUBLE_TOP_OVERBOUGHT")
        
        # Failure swings
        if self._detect_failure_swing_bullish(k_percent):
            signals.append("FAILURE_SWING_BULLISH")
        elif self._detect_failure_swing_bearish(k_percent):
            signals.append("FAILURE_SWING_BEARISH")
        
        return signals if signals else ["NEUTRAL"]
    
    def _detect_bullish_divergence(self, k_values: List[float]) -> bool:
        """Detect bullish divergence pattern"""
        if len(k_values) < 6:
            return False
        
        # Look for higher lows in stochastic while price makes lower lows
        # Simplified: check if recent low is higher than previous low
        mid_point = len(k_values) // 2
        first_half_min = min(k_values[:mid_point])
        second_half_min = min(k_values[mid_point:])
        
        return second_half_min > first_half_min and first_half_min < 30
    
    def _detect_bearish_divergence(self, k_values: List[float]) -> bool:
        """Detect bearish divergence pattern"""
        if len(k_values) < 6:
            return False
        
        # Look for lower highs in stochastic while price makes higher highs
        # Simplified: check if recent high is lower than previous high
        mid_point = len(k_values) // 2
        first_half_max = max(k_values[:mid_point])
        second_half_max = max(k_values[mid_point:])
        
        return second_half_max < first_half_max and first_half_max > 70
    
    def _detect_double_bottom(self, k_percent: List[float], d_percent: List[float]) -> bool:
        """Detect double bottom pattern in oversold region"""
        if len(k_percent) < 10:
            return False
        
        recent_k = k_percent[-10:]
        recent_d = d_percent[-10:]
        
        # Look for two lows in oversold region
        lows = [i for i, (k, d) in enumerate(zip(recent_k, recent_d)) 
                if k <= self.oversold_level and d <= self.oversold_level]
        
        if len(lows) >= 2:
            # Check if there's a peak between the lows
            first_low = lows[0]
            last_low = lows[-1]
            
            if last_low - first_low >= 3:  # At least 3 periods apart
                middle_section = recent_k[first_low+1:last_low]
                if middle_section and max(middle_section) > self.oversold_level + 10:
                    return True
        
        return False
    
    def _detect_double_top(self, k_percent: List[float], d_percent: List[float]) -> bool:
        """Detect double top pattern in overbought region"""
        if len(k_percent) < 10:
            return False
        
        recent_k = k_percent[-10:]
        recent_d = d_percent[-10:]
        
        # Look for two highs in overbought region
        highs = [i for i, (k, d) in enumerate(zip(recent_k, recent_d)) 
                 if k >= self.overbought_level and d >= self.overbought_level]
        
        if len(highs) >= 2:
            # Check if there's a valley between the highs
            first_high = highs[0]
            last_high = highs[-1]
            
            if last_high - first_high >= 3:  # At least 3 periods apart
                middle_section = recent_k[first_high+1:last_high]
                if middle_section and min(middle_section) < self.overbought_level - 10:
                    return True
        
        return False
    
    def _detect_failure_swing_bullish(self, k_percent: List[float]) -> bool:
        """Detect bullish failure swing"""
        if len(k_percent) < 8:
            return False
        
        recent_k = k_percent[-8:]
        
        # Look for pattern: low below 20, rally above 20, pullback that stays above previous low
        for i in range(2, len(recent_k) - 2):
            if (recent_k[i-1] <= self.oversold_level and  # Previous low in oversold
                recent_k[i] > self.oversold_level + 5 and   # Rally above oversold
                recent_k[i+1] < recent_k[i] and             # Pullback starts
                recent_k[i+1] > recent_k[i-1]):             # But stays above previous low
                return True
        
        return False
    
    def _detect_failure_swing_bearish(self, k_percent: List[float]) -> bool:
        """Detect bearish failure swing"""
        if len(k_percent) < 8:
            return False
        
        recent_k = k_percent[-8:]
        
        # Look for pattern: high above 80, decline below 80, bounce that stays below previous high
        for i in range(2, len(recent_k) - 2):
            if (recent_k[i-1] >= self.overbought_level and  # Previous high in overbought
                recent_k[i] < self.overbought_level - 5 and  # Decline below overbought
                recent_k[i+1] > recent_k[i] and              # Bounce starts
                recent_k[i+1] < recent_k[i-1]):              # But stays below previous high
                return True
        
        return False
    
    def _analyze_market_condition(self, k_percent: List[float], d_percent: List[float]) -> Dict[str, Any]:
        """Analyze current market condition"""
        if not k_percent or not d_percent:
            return {'condition': 'UNKNOWN', 'strength': 0}
        
        current_k = k_percent[-1]
        current_d = d_percent[-1]
        
        if current_k >= self.overbought_level and current_d >= self.overbought_level:
            condition = "OVERBOUGHT"
            strength = min((current_k + current_d) / 200, 1.0)
        elif current_k <= self.oversold_level and current_d <= self.oversold_level:
            condition = "OVERSOLD"
            strength = min((200 - current_k - current_d) / 200, 1.0)
        elif current_k > 50 and current_d > 50:
            condition = "BULLISH"
            strength = (current_k + current_d - 100) / 100
        elif current_k < 50 and current_d < 50:
            condition = "BEARISH"
            strength = (100 - current_k - current_d) / 100
        else:
            condition = "NEUTRAL"
            strength = 0.5
        
        return {
            'condition': condition,
            'strength': max(0, min(1, strength)),
            'k_level': current_k,
            'd_level': current_d
        }
    
    def _analyze_momentum(self, k_percent: List[float], d_percent: List[float], 
                         j_percent: List[float]) -> Dict[str, Any]:
        """Analyze momentum characteristics"""
        if len(k_percent) < 5:
            return {'momentum': 'UNKNOWN', 'acceleration': 0}
        
        # Calculate recent trends
        recent_k = k_percent[-5:]
        recent_d = d_percent[-5:]
        recent_j = j_percent[-5:] if len(j_percent) >= 5 else recent_k
        
        k_trend = np.polyfit(range(len(recent_k)), recent_k, 1)[0]
        d_trend = np.polyfit(range(len(recent_d)), recent_d, 1)[0]
        j_trend = np.polyfit(range(len(recent_j)), recent_j, 1)[0]
        
        # Analyze momentum
        if k_trend > 3 and d_trend > 1:
            momentum = "STRONG_BULLISH"
        elif k_trend > 1 and d_trend > 0:
            momentum = "BULLISH"
        elif k_trend < -3 and d_trend < -1:
            momentum = "STRONG_BEARISH"
        elif k_trend < -1 and d_trend < 0:
            momentum = "BEARISH"
        else:
            momentum = "NEUTRAL"
        
        # Calculate acceleration (difference between %K and %D trends)
        acceleration = k_trend - d_trend
        
        return {
            'momentum': momentum,
            'acceleration': acceleration,
            'k_trend': k_trend,
            'd_trend': d_trend,
            'j_trend': j_trend
        }
    
    def _analyze_divergence(self, close: List[float], k_percent: List[float]) -> Dict[str, Any]:
        """Analyze price-oscillator divergence"""
        if len(close) < 10 or len(k_percent) < 10:
            return {'divergence_type': 'NONE', 'strength': 0}
        
        # Get recent data
        recent_close = close[-10:]
        recent_k = k_percent[-10:]
        
        # Calculate trends
        price_trend = np.polyfit(range(len(recent_close)), recent_close, 1)[0]
        k_trend = np.polyfit(range(len(recent_k)), recent_k, 1)[0]
        
        # Detect divergence
        if price_trend > 0 and k_trend < -1:  # Price up, stochastic down
            divergence_type = "BEARISH"
            strength = abs(k_trend) / 10
        elif price_trend < 0 and k_trend > 1:  # Price down, stochastic up
            divergence_type = "BULLISH"
            strength = k_trend / 10
        else:
            divergence_type = "NONE"
            strength = 0
        
        return {
            'divergence_type': divergence_type,
            'strength': max(0, min(1, strength)),
            'price_trend': price_trend,
            'oscillator_trend': k_trend
        }
    
    def _analyze_crossovers(self, k_percent: List[float], d_percent: List[float]) -> Dict[str, Any]:
        """Analyze %K and %D crossovers"""
        if len(k_percent) < 5 or len(d_percent) < 5:
            return {'recent_crossovers': [], 'crossover_frequency': 0}
        
        crossovers = []
        
        # Look for crossovers in recent periods
        for i in range(1, min(10, len(k_percent))):
            idx = -i
            prev_idx = -(i+1)
            
            if (k_percent[prev_idx] <= d_percent[prev_idx] and k_percent[idx] > d_percent[idx]):
                crossovers.append({
                    'type': 'BULLISH',
                    'periods_ago': i-1,
                    'k_level': k_percent[idx],
                    'd_level': d_percent[idx]
                })
            elif (k_percent[prev_idx] >= d_percent[prev_idx] and k_percent[idx] < d_percent[idx]):
                crossovers.append({
                    'type': 'BEARISH',
                    'periods_ago': i-1,
                    'k_level': k_percent[idx],
                    'd_level': d_percent[idx]
                })
        
        return {
            'recent_crossovers': crossovers,
            'crossover_frequency': len(crossovers)
        }
    
    def _analyze_volatility(self, k_percent: List[float], d_percent: List[float]) -> Dict[str, Any]:
        """Analyze volatility based on stochastic movement"""
        if len(k_percent) < 10:
            return {'volatility': 'UNKNOWN', 'stability': 0}
        
        recent_k = k_percent[-10:]
        recent_d = d_percent[-10:]
        
        # Calculate standard deviation as volatility measure
        k_volatility = np.std(recent_k)
        d_volatility = np.std(recent_d)
        
        avg_volatility = (k_volatility + d_volatility) / 2
        
        if avg_volatility > 25:
            volatility = "HIGH"
            stability = 0.2
        elif avg_volatility > 15:
            volatility = "MEDIUM"
            stability = 0.5
        else:
            volatility = "LOW"
            stability = 0.8
        
        return {
            'volatility': volatility,
            'stability': stability,
            'k_volatility': k_volatility,
            'd_volatility': d_volatility,
            'avg_volatility': avg_volatility
        }
    
    def _analyze_trend(self, k_percent: List[float], d_percent: List[float]) -> Dict[str, Any]:
        """Analyze trend using stochastic levels"""
        if len(k_percent) < 5:
            return {'trend': 'UNKNOWN', 'consistency': 0}
        
        recent_k = k_percent[-5:]
        recent_d = d_percent[-5:]
        
        # Count periods above/below 50
        k_above_50 = sum(1 for k in recent_k if k > 50)
        d_above_50 = sum(1 for d in recent_d if d > 50)
        
        total_above = k_above_50 + d_above_50
        total_periods = len(recent_k) + len(recent_d)
        
        if total_above >= 8:  # Strong bullish
            trend = "STRONG_BULLISH"
            consistency = total_above / total_periods
        elif total_above >= 6:  # Bullish
            trend = "BULLISH"
            consistency = total_above / total_periods
        elif total_above <= 2:  # Strong bearish
            trend = "STRONG_BEARISH"
            consistency = (total_periods - total_above) / total_periods
        elif total_above <= 4:  # Bearish
            trend = "BEARISH"
            consistency = (total_periods - total_above) / total_periods
        else:
            trend = "NEUTRAL"
            consistency = 0.5
        
        return {
            'trend': trend,
            'consistency': consistency,
            'k_above_50': k_above_50,
            'd_above_50': d_above_50
        }
    
    def _identify_support_resistance(self, k_percent: List[float], d_percent: List[float]) -> Dict[str, Any]:
        """Identify key support and resistance levels in stochastic"""
        if len(k_percent) < 20:
            return {'support_levels': [], 'resistance_levels': []}
        
        recent_k = k_percent[-20:]
        recent_d = d_percent[-20:]
        
        support_levels = []
        resistance_levels = []
        
        # Standard levels
        support_levels.extend([
            {'level': self.oversold_level, 'type': 'OVERSOLD_LINE', 'strength': 'STRONG'},
            {'level': 30, 'type': 'SUPPORT_30', 'strength': 'MEDIUM'},
            {'level': 50, 'type': 'MIDLINE', 'strength': 'MEDIUM'}
        ])
        
        resistance_levels.extend([
            {'level': self.overbought_level, 'type': 'OVERBOUGHT_LINE', 'strength': 'STRONG'},
            {'level': 70, 'type': 'RESISTANCE_70', 'strength': 'MEDIUM'},
            {'level': 50, 'type': 'MIDLINE', 'strength': 'MEDIUM'}
        ])
        
        # Dynamic levels based on recent price action
        k_min = min(recent_k)
        k_max = max(recent_k)
        
        if k_min > self.oversold_level:
            support_levels.append({
                'level': k_min,
                'type': 'RECENT_LOW',
                'strength': 'WEAK'
            })
        
        if k_max < self.overbought_level:
            resistance_levels.append({
                'level': k_max,
                'type': 'RECENT_HIGH',
                'strength': 'WEAK'
            })
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    
    def get_chart_data(self, result: StochasticResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'stochastic',
            'name': 'Stochastic Oscillator',
            'data': {
                'k_percent': result.k_percent,
                'd_percent': result.d_percent,
                'j_percent': result.j_percent
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'levels': {
                'overbought': self.overbought_level,
                'oversold': self.oversold_level,
                'midline': 50
            },
            'series': [
                {
                    'name': '%K',
                    'data': result.k_percent,
                    'color': '#FF6B6B',
                    'type': 'line',
                    'lineWidth': 2
                },
                {
                    'name': '%D',
                    'data': result.d_percent,
                    'color': '#4ECDC4',
                    'type': 'line',
                    'lineWidth': 2
                },
                {
                    'name': '%J',
                    'data': result.j_percent,
                    'color': '#45B7D1',
                    'type': 'line',
                    'lineWidth': 1,
                    'dashStyle': 'Dash'
                }
            ],
            'zones': [
                {
                    'value': self.overbought_level,
                    'color': 'rgba(244, 67, 54, 0.1)',
                    'label': 'Overbought'
                },
                {
                    'value': self.oversold_level,
                    'color': 'rgba(76, 175, 80, 0.1)',
                    'label': 'Oversold'
                }
            ]
        }


# Example usage
if __name__ == "__main__":
    # Sample OHLC data
    sample_high = [102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116, 115, 117, 119, 118, 120, 122, 121, 123, 125, 124, 126, 128, 127, 129, 131, 130]
    sample_low = [98, 100, 99, 101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116, 118, 117, 119, 121, 120, 122, 124, 123, 125, 127, 126]
    sample_close = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121, 123, 122, 124, 126, 125, 127, 129, 128]
    
    # Calculate Stochastic
    stochastic_calculator = StochasticIndicator()
    result = stochastic_calculator.calculate(sample_high, sample_low, sample_close, AssetType.STOCK)
    
    print(f"Stochastic Analysis:")
    print(f"Current %K: {result.metadata['current_k']:.2f}")
    print(f"Current %D: {result.metadata['current_d']:.2f}")
    print(f"Current %J: {result.metadata['current_j']:.2f}")
    print(f"Market Condition: {result.metadata['market_condition']['condition']}")
    print(f"Momentum: {result.metadata['momentum_analysis']['momentum']}")
    print(f"Signals: {', '.join(result.signals)}")