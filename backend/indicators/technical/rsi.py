"""Relative Strength Index (RSI) Indicator

The RSI is a momentum oscillator that measures the speed and change of price movements.
It oscillates between 0 and 100 and is typically used to identify overbought or oversold conditions.

Author: Assistant
Date: 2024
"""

import numpy as np
from typing import List, Dict, Any, Optional
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
class RSIResult:
    """Result of RSI calculation"""
    name: str
    values: List[float]
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class RSIIndicator:
    """Relative Strength Index (RSI) Calculator"""
    
    def __init__(self, period: int = 14, overbought_threshold: float = 70.0, oversold_threshold: float = 30.0):
        """
        Initialize RSI calculator
        
        Args:
            period: Period for RSI calculation (default: 14)
            overbought_threshold: RSI level considered overbought (default: 70)
            oversold_threshold: RSI level considered oversold (default: 30)
        """
        self.period = period
        self.overbought_threshold = overbought_threshold
        self.oversold_threshold = oversold_threshold
    
    def calculate(self, prices: List[float], asset_type: AssetType = AssetType.STOCK) -> RSIResult:
        """
        Calculate RSI for given price series
        
        Args:
            prices: List of price values
            asset_type: Type of asset being analyzed
            
        Returns:
            RSIResult containing RSI values and analysis
        """
        if len(prices) < self.period + 1:
            # Not enough data, return neutral RSI
            rsi_values = [50.0] * len(prices)
            signals = ["INSUFFICIENT_DATA"]
            confidence = 0.1
        else:
            rsi_values = self._calculate_rsi(prices)
            signals = self._generate_signals(rsi_values)
            confidence = min(0.95, 0.5 + (len(prices) - self.period) * 0.01)
        
        current_rsi = rsi_values[-1] if rsi_values else 50.0
        
        return RSIResult(
            name="Relative Strength Index",
            values=rsi_values,
            metadata={
                'period': self.period,
                'current_rsi': current_rsi,
                'overbought_threshold': self.overbought_threshold,
                'oversold_threshold': self.oversold_threshold,
                'trend': self._determine_trend(rsi_values),
                'divergence': self._check_divergence(prices, rsi_values),
                'interpretation': self._interpret_rsi(current_rsi)
            },
            confidence=confidence,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.MOMENTUM,
            signals=signals
        )
    
    def _calculate_rsi(self, prices: List[float]) -> List[float]:
        """Calculate RSI using Wilder's smoothing method"""
        if len(prices) < 2:
            return [50.0] * len(prices)
        
        # Calculate price changes
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        rsi_values = []
        
        # Calculate initial average gain and loss
        if len(gains) >= self.period:
            avg_gain = np.mean(gains[:self.period])
            avg_loss = np.mean(losses[:self.period])
            
            # Calculate RSI for each subsequent period using Wilder's smoothing
            for i in range(self.period, len(deltas)):
                # Wilder's smoothing: (previous_avg * (period-1) + current_value) / period
                avg_gain = (avg_gain * (self.period - 1) + gains[i]) / self.period
                avg_loss = (avg_loss * (self.period - 1) + losses[i]) / self.period
                
                if avg_loss == 0:
                    rsi = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100.0 - (100.0 / (1.0 + rs))
                
                rsi_values.append(rsi)
        
        # Pad with neutral values for initial period
        return [50.0] * (self.period + 1) + rsi_values
    
    def _generate_signals(self, rsi_values: List[float]) -> List[str]:
        """Generate trading signals based on RSI"""
        if not rsi_values:
            return ["NO_SIGNAL"]
        
        current_rsi = rsi_values[-1]
        signals = []
        
        # Overbought/Oversold signals
        if current_rsi >= self.overbought_threshold:
            signals.append("OVERBOUGHT")
        elif current_rsi <= self.oversold_threshold:
            signals.append("OVERSOLD")
        
        # Trend signals
        if len(rsi_values) >= 3:
            recent_trend = np.polyfit(range(3), rsi_values[-3:], 1)[0]
            if recent_trend > 1:
                signals.append("BULLISH_MOMENTUM")
            elif recent_trend < -1:
                signals.append("BEARISH_MOMENTUM")
        
        # Centerline crossover
        if len(rsi_values) >= 2:
            if rsi_values[-2] < 50 < rsi_values[-1]:
                signals.append("BULLISH_CROSSOVER")
            elif rsi_values[-2] > 50 > rsi_values[-1]:
                signals.append("BEARISH_CROSSOVER")
        
        return signals if signals else ["NEUTRAL"]
    
    def _determine_trend(self, rsi_values: List[float]) -> str:
        """Determine overall RSI trend"""
        if len(rsi_values) < 5:
            return "INSUFFICIENT_DATA"
        
        # Linear regression on recent RSI values
        recent_values = rsi_values[-10:] if len(rsi_values) >= 10 else rsi_values
        x = np.arange(len(recent_values))
        slope = np.polyfit(x, recent_values, 1)[0]
        
        if slope > 2:
            return "STRONG_UPTREND"
        elif slope > 0.5:
            return "UPTREND"
        elif slope < -2:
            return "STRONG_DOWNTREND"
        elif slope < -0.5:
            return "DOWNTREND"
        else:
            return "SIDEWAYS"
    
    def _check_divergence(self, prices: List[float], rsi_values: List[float]) -> Dict[str, Any]:
        """Check for bullish/bearish divergence"""
        if len(prices) < 10 or len(rsi_values) < 10:
            return {'type': 'NONE', 'strength': 0}
        
        # Get recent data
        recent_prices = prices[-10:]
        recent_rsi = rsi_values[-10:]
        
        # Calculate trends
        price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
        rsi_trend = np.polyfit(range(len(recent_rsi)), recent_rsi, 1)[0]
        
        # Check for divergence
        if price_trend > 0 and rsi_trend < -0.5:
            return {'type': 'BEARISH', 'strength': abs(rsi_trend)}
        elif price_trend < 0 and rsi_trend > 0.5:
            return {'type': 'BULLISH', 'strength': rsi_trend}
        else:
            return {'type': 'NONE', 'strength': 0}
    
    def _interpret_rsi(self, rsi_value: float) -> str:
        """Provide interpretation of current RSI value"""
        if rsi_value >= 80:
            return "Extremely overbought - strong sell signal"
        elif rsi_value >= self.overbought_threshold:
            return "Overbought - consider selling"
        elif rsi_value >= 60:
            return "Bullish momentum - uptrend likely"
        elif rsi_value >= 40:
            return "Neutral - no clear direction"
        elif rsi_value >= self.oversold_threshold:
            return "Bearish momentum - downtrend likely"
        elif rsi_value >= 20:
            return "Oversold - consider buying"
        else:
            return "Extremely oversold - strong buy signal"
    
    def get_chart_data(self, result: RSIResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'oscillator',
            'name': 'RSI',
            'data': result.values,
            'levels': {
                'overbought': self.overbought_threshold,
                'oversold': self.oversold_threshold,
                'centerline': 50
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'yAxis': {
                'min': 0,
                'max': 100,
                'title': 'RSI'
            }
        }


# Example usage
if __name__ == "__main__":
    # Sample price data
    sample_prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
    
    # Calculate RSI
    rsi_calculator = RSIIndicator(period=14)
    result = rsi_calculator.calculate(sample_prices, AssetType.STOCK)
    
    print(f"RSI Analysis:")
    print(f"Current RSI: {result.metadata['current_rsi']:.2f}")
    print(f"Trend: {result.metadata['trend']}")
    print(f"Signals: {', '.join(result.signals)}")
    print(f"Interpretation: {result.metadata['interpretation']}")