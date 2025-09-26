"""Ichimoku Kinko Hyo (Ichimoku Cloud) Indicator

Ichimoku is a comprehensive technical analysis system that provides information about
support/resistance, trend direction, momentum, and trading signals all in one view.
It consists of five lines: Tenkan-sen, Kijun-sen, Senkou Span A, Senkou Span B, and Chikou Span.

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
    TREND = "trend"
    MOMENTUM = "momentum"
    SUPPORT_RESISTANCE = "support_resistance"


@dataclass
class IchimokuResult:
    """Result of Ichimoku calculation"""
    name: str
    tenkan_sen: List[float]  # Conversion Line
    kijun_sen: List[float]   # Base Line
    senkou_span_a: List[float]  # Leading Span A
    senkou_span_b: List[float]  # Leading Span B
    chikou_span: List[float]    # Lagging Span
    cloud_top: List[float]      # Upper cloud boundary
    cloud_bottom: List[float]   # Lower cloud boundary
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class IchimokuIndicator:
    """Ichimoku Kinko Hyo Calculator with Advanced Analysis"""
    
    def __init__(self, tenkan_period: int = 9, kijun_period: int = 26, 
                 senkou_b_period: int = 52, displacement: int = 26):
        """
        Initialize Ichimoku calculator
        
        Args:
            tenkan_period: Period for Tenkan-sen (Conversion Line) (default: 9)
            kijun_period: Period for Kijun-sen (Base Line) (default: 26)
            senkou_b_period: Period for Senkou Span B (Leading Span B) (default: 52)
            displacement: Forward displacement for Senkou spans (default: 26)
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_b_period = senkou_b_period
        self.displacement = displacement
    
    def calculate(self, high: List[float], low: List[float], close: List[float], 
                 asset_type: AssetType = AssetType.STOCK) -> IchimokuResult:
        """
        Calculate Ichimoku components for given price series
        
        Args:
            high: List of high prices
            low: List of low prices
            close: List of close prices
            asset_type: Type of asset being analyzed
            
        Returns:
            IchimokuResult containing all Ichimoku components and analysis
        """
        if len(high) != len(low) or len(low) != len(close):
            raise ValueError("High, low, and close arrays must have the same length")
        
        if len(close) < self.senkou_b_period:
            # Not enough data - return simplified calculation
            tenkan_sen = close.copy()
            kijun_sen = close.copy()
            senkou_span_a = close.copy()
            senkou_span_b = close.copy()
            chikou_span = close.copy()
            signals = ["INSUFFICIENT_DATA"]
            confidence = 0.1
        else:
            # Calculate all Ichimoku components
            tenkan_sen = self._calculate_tenkan_sen(high, low)
            kijun_sen = self._calculate_kijun_sen(high, low)
            senkou_span_a = self._calculate_senkou_span_a(tenkan_sen, kijun_sen)
            senkou_span_b = self._calculate_senkou_span_b(high, low)
            chikou_span = self._calculate_chikou_span(close)
            
            signals = self._generate_signals(close, tenkan_sen, kijun_sen, 
                                           senkou_span_a, senkou_span_b, chikou_span)
            confidence = min(0.95, 0.3 + (len(close) - self.senkou_b_period) * 0.01)
        
        # Calculate cloud boundaries
        cloud_top, cloud_bottom = self._calculate_cloud_boundaries(senkou_span_a, senkou_span_b)
        
        return IchimokuResult(
            name="Ichimoku Kinko Hyo",
            tenkan_sen=tenkan_sen,
            kijun_sen=kijun_sen,
            senkou_span_a=senkou_span_a,
            senkou_span_b=senkou_span_b,
            chikou_span=chikou_span,
            cloud_top=cloud_top,
            cloud_bottom=cloud_bottom,
            metadata={
                'tenkan_period': self.tenkan_period,
                'kijun_period': self.kijun_period,
                'senkou_b_period': self.senkou_b_period,
                'displacement': self.displacement,
                'current_price': close[-1] if close else 0,
                'current_tenkan': tenkan_sen[-1] if tenkan_sen else 0,
                'current_kijun': kijun_sen[-1] if kijun_sen else 0,
                'current_cloud_top': cloud_top[-1] if cloud_top else 0,
                'current_cloud_bottom': cloud_bottom[-1] if cloud_bottom else 0,
                'price_vs_cloud': self._analyze_price_vs_cloud(close, cloud_top, cloud_bottom),
                'trend_analysis': self._analyze_trend(close, tenkan_sen, kijun_sen, cloud_top, cloud_bottom),
                'momentum_analysis': self._analyze_momentum(tenkan_sen, kijun_sen),
                'cloud_analysis': self._analyze_cloud(senkou_span_a, senkou_span_b),
                'chikou_analysis': self._analyze_chikou_span(close, chikou_span),
                'support_resistance': self._identify_support_resistance(close, kijun_sen, cloud_top, cloud_bottom)
            },
            confidence=confidence,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.TREND,
            signals=signals
        )
    
    def _calculate_tenkan_sen(self, high: List[float], low: List[float]) -> List[float]:
        """Calculate Tenkan-sen (Conversion Line)"""
        tenkan_sen = []
        
        for i in range(len(high)):
            if i < self.tenkan_period - 1:
                # Use available data for initial values
                period_high = max(high[:i+1])
                period_low = min(low[:i+1])
            else:
                period_high = max(high[i-self.tenkan_period+1:i+1])
                period_low = min(low[i-self.tenkan_period+1:i+1])
            
            tenkan_sen.append((period_high + period_low) / 2)
        
        return tenkan_sen
    
    def _calculate_kijun_sen(self, high: List[float], low: List[float]) -> List[float]:
        """Calculate Kijun-sen (Base Line)"""
        kijun_sen = []
        
        for i in range(len(high)):
            if i < self.kijun_period - 1:
                # Use available data for initial values
                period_high = max(high[:i+1])
                period_low = min(low[:i+1])
            else:
                period_high = max(high[i-self.kijun_period+1:i+1])
                period_low = min(low[i-self.kijun_period+1:i+1])
            
            kijun_sen.append((period_high + period_low) / 2)
        
        return kijun_sen
    
    def _calculate_senkou_span_a(self, tenkan_sen: List[float], kijun_sen: List[float]) -> List[float]:
        """Calculate Senkou Span A (Leading Span A)"""
        senkou_span_a = []
        
        for tenkan, kijun in zip(tenkan_sen, kijun_sen):
            senkou_span_a.append((tenkan + kijun) / 2)
        
        # Shift forward by displacement periods
        return [0] * self.displacement + senkou_span_a
    
    def _calculate_senkou_span_b(self, high: List[float], low: List[float]) -> List[float]:
        """Calculate Senkou Span B (Leading Span B)"""
        senkou_span_b = []
        
        for i in range(len(high)):
            if i < self.senkou_b_period - 1:
                # Use available data for initial values
                period_high = max(high[:i+1])
                period_low = min(low[:i+1])
            else:
                period_high = max(high[i-self.senkou_b_period+1:i+1])
                period_low = min(low[i-self.senkou_b_period+1:i+1])
            
            senkou_span_b.append((period_high + period_low) / 2)
        
        # Shift forward by displacement periods
        return [0] * self.displacement + senkou_span_b
    
    def _calculate_chikou_span(self, close: List[float]) -> List[float]:
        """Calculate Chikou Span (Lagging Span)"""
        # Shift close prices backward by displacement periods
        if len(close) <= self.displacement:
            return [0] * len(close)
        
        return close[self.displacement:] + [0] * self.displacement
    
    def _calculate_cloud_boundaries(self, senkou_span_a: List[float], 
                                  senkou_span_b: List[float]) -> Tuple[List[float], List[float]]:
        """Calculate cloud top and bottom boundaries"""
        cloud_top = []
        cloud_bottom = []
        
        for span_a, span_b in zip(senkou_span_a, senkou_span_b):
            cloud_top.append(max(span_a, span_b))
            cloud_bottom.append(min(span_a, span_b))
        
        return cloud_top, cloud_bottom
    
    def _generate_signals(self, close: List[float], tenkan_sen: List[float], 
                         kijun_sen: List[float], senkou_span_a: List[float], 
                         senkou_span_b: List[float], chikou_span: List[float]) -> List[str]:
        """Generate trading signals based on Ichimoku"""
        if len(close) < 2:
            return ["NO_SIGNAL"]
        
        signals = []
        current_price = close[-1]
        
        # Tenkan-Kijun crossover signals
        if len(tenkan_sen) >= 2 and len(kijun_sen) >= 2:
            if (tenkan_sen[-2] <= kijun_sen[-2] and tenkan_sen[-1] > kijun_sen[-1]):
                signals.append("TENKAN_KIJUN_BULLISH_CROSS")
            elif (tenkan_sen[-2] >= kijun_sen[-2] and tenkan_sen[-1] < kijun_sen[-1]):
                signals.append("TENKAN_KIJUN_BEARISH_CROSS")
        
        # Price vs Kijun-sen
        if len(kijun_sen) >= 2:
            if close[-2] <= kijun_sen[-2] and current_price > kijun_sen[-1]:
                signals.append("PRICE_ABOVE_KIJUN")
            elif close[-2] >= kijun_sen[-2] and current_price < kijun_sen[-1]:
                signals.append("PRICE_BELOW_KIJUN")
        
        # Cloud analysis
        cloud_top, cloud_bottom = self._calculate_cloud_boundaries(senkou_span_a, senkou_span_b)
        
        if len(cloud_top) > 0 and len(cloud_bottom) > 0:
            current_cloud_top = cloud_top[-1] if cloud_top[-1] != 0 else current_price
            current_cloud_bottom = cloud_bottom[-1] if cloud_bottom[-1] != 0 else current_price
            
            if current_price > current_cloud_top:
                signals.append("PRICE_ABOVE_CLOUD")
            elif current_price < current_cloud_bottom:
                signals.append("PRICE_BELOW_CLOUD")
            else:
                signals.append("PRICE_IN_CLOUD")
        
        # Cloud color (bullish/bearish)
        if len(senkou_span_a) >= 2 and len(senkou_span_b) >= 2:
            if senkou_span_a[-1] > senkou_span_b[-1]:
                signals.append("BULLISH_CLOUD")
            elif senkou_span_a[-1] < senkou_span_b[-1]:
                signals.append("BEARISH_CLOUD")
        
        # Chikou Span analysis
        if len(chikou_span) >= self.displacement and len(close) >= self.displacement:
            chikou_current = chikou_span[-(self.displacement + 1)] if len(chikou_span) > self.displacement else 0
            price_26_ago = close[-(self.displacement + 1)] if len(close) > self.displacement else current_price
            
            if chikou_current > price_26_ago:
                signals.append("CHIKOU_BULLISH")
            elif chikou_current < price_26_ago:
                signals.append("CHIKOU_BEARISH")
        
        # Strong trend signals (all components aligned)
        if len(signals) >= 3:
            bullish_signals = sum(1 for s in signals if 'BULLISH' in s or 'ABOVE' in s)
            bearish_signals = sum(1 for s in signals if 'BEARISH' in s or 'BELOW' in s)
            
            if bullish_signals >= 3:
                signals.append("STRONG_BULLISH_TREND")
            elif bearish_signals >= 3:
                signals.append("STRONG_BEARISH_TREND")
        
        return signals if signals else ["NEUTRAL"]
    
    def _analyze_price_vs_cloud(self, close: List[float], cloud_top: List[float], 
                               cloud_bottom: List[float]) -> Dict[str, Any]:
        """Analyze price position relative to cloud"""
        if not close or not cloud_top or not cloud_bottom:
            return {'position': 'UNKNOWN', 'distance': 0}
        
        current_price = close[-1]
        current_cloud_top = cloud_top[-1] if cloud_top[-1] != 0 else current_price
        current_cloud_bottom = cloud_bottom[-1] if cloud_bottom[-1] != 0 else current_price
        
        if current_price > current_cloud_top:
            position = "ABOVE_CLOUD"
            distance = (current_price - current_cloud_top) / current_price
        elif current_price < current_cloud_bottom:
            position = "BELOW_CLOUD"
            distance = (current_cloud_bottom - current_price) / current_price
        else:
            position = "IN_CLOUD"
            cloud_thickness = current_cloud_top - current_cloud_bottom
            if cloud_thickness > 0:
                distance = (current_price - current_cloud_bottom) / cloud_thickness
            else:
                distance = 0.5
        
        return {
            'position': position,
            'distance': distance,
            'cloud_top': current_cloud_top,
            'cloud_bottom': current_cloud_bottom,
            'cloud_thickness': current_cloud_top - current_cloud_bottom
        }
    
    def _analyze_trend(self, close: List[float], tenkan_sen: List[float], 
                      kijun_sen: List[float], cloud_top: List[float], 
                      cloud_bottom: List[float]) -> Dict[str, Any]:
        """Analyze overall trend using all Ichimoku components"""
        if len(close) < 10:
            return {'direction': 'UNKNOWN', 'strength': 0}
        
        current_price = close[-1]
        current_tenkan = tenkan_sen[-1] if tenkan_sen else current_price
        current_kijun = kijun_sen[-1] if kijun_sen else current_price
        current_cloud_top = cloud_top[-1] if cloud_top and cloud_top[-1] != 0 else current_price
        current_cloud_bottom = cloud_bottom[-1] if cloud_bottom and cloud_bottom[-1] != 0 else current_price
        
        # Score based on component alignment
        score = 0
        
        # Price vs components
        if current_price > current_tenkan:
            score += 1
        if current_price > current_kijun:
            score += 1
        if current_price > current_cloud_top:
            score += 2
        
        # Tenkan vs Kijun
        if current_tenkan > current_kijun:
            score += 1
        
        # Cloud color
        if len(tenkan_sen) >= 2 and len(kijun_sen) >= 2:
            senkou_a = (tenkan_sen[-1] + kijun_sen[-1]) / 2
            senkou_b = current_cloud_bottom  # Approximation
            if senkou_a > senkou_b:
                score += 1
        
        # Determine trend
        if score >= 5:
            direction = "STRONG_BULLISH"
            strength = score / 6
        elif score >= 3:
            direction = "BULLISH"
            strength = score / 6
        elif score <= 1:
            direction = "STRONG_BEARISH"
            strength = (6 - score) / 6
        elif score <= 3:
            direction = "BEARISH"
            strength = (6 - score) / 6
        else:
            direction = "NEUTRAL"
            strength = 0.5
        
        return {
            'direction': direction,
            'strength': strength,
            'score': score,
            'max_score': 6
        }
    
    def _analyze_momentum(self, tenkan_sen: List[float], kijun_sen: List[float]) -> Dict[str, Any]:
        """Analyze momentum using Tenkan-Kijun relationship"""
        if len(tenkan_sen) < 5 or len(kijun_sen) < 5:
            return {'momentum': 'UNKNOWN', 'divergence': False}
        
        # Calculate recent trends
        recent_tenkan = tenkan_sen[-5:]
        recent_kijun = kijun_sen[-5:]
        
        tenkan_trend = np.polyfit(range(len(recent_tenkan)), recent_tenkan, 1)[0]
        kijun_trend = np.polyfit(range(len(recent_kijun)), recent_kijun, 1)[0]
        
        # Analyze momentum
        if tenkan_trend > 0 and kijun_trend > 0:
            if tenkan_trend > kijun_trend * 1.5:
                momentum = "ACCELERATING_BULLISH"
            else:
                momentum = "BULLISH"
        elif tenkan_trend < 0 and kijun_trend < 0:
            if abs(tenkan_trend) > abs(kijun_trend) * 1.5:
                momentum = "ACCELERATING_BEARISH"
            else:
                momentum = "BEARISH"
        else:
            momentum = "MIXED"
        
        # Check for divergence
        divergence = (tenkan_trend > 0 and kijun_trend < 0) or (tenkan_trend < 0 and kijun_trend > 0)
        
        return {
            'momentum': momentum,
            'divergence': divergence,
            'tenkan_trend': tenkan_trend,
            'kijun_trend': kijun_trend
        }
    
    def _analyze_cloud(self, senkou_span_a: List[float], senkou_span_b: List[float]) -> Dict[str, Any]:
        """Analyze cloud characteristics"""
        if len(senkou_span_a) < 5 or len(senkou_span_b) < 5:
            return {'color': 'UNKNOWN', 'thickness': 0, 'trend': 'UNKNOWN'}
        
        current_span_a = senkou_span_a[-1]
        current_span_b = senkou_span_b[-1]
        
        # Cloud color
        if current_span_a > current_span_b:
            color = "BULLISH"  # Green/Blue cloud
        elif current_span_a < current_span_b:
            color = "BEARISH"  # Red cloud
        else:
            color = "NEUTRAL"
        
        # Cloud thickness (volatility indicator)
        thickness = abs(current_span_a - current_span_b)
        
        # Cloud trend
        recent_span_a = senkou_span_a[-5:]
        recent_span_b = senkou_span_b[-5:]
        
        span_a_trend = np.polyfit(range(len(recent_span_a)), recent_span_a, 1)[0]
        span_b_trend = np.polyfit(range(len(recent_span_b)), recent_span_b, 1)[0]
        
        avg_trend = (span_a_trend + span_b_trend) / 2
        
        if avg_trend > 0:
            trend = "RISING"
        elif avg_trend < 0:
            trend = "FALLING"
        else:
            trend = "FLAT"
        
        return {
            'color': color,
            'thickness': thickness,
            'trend': trend,
            'span_a_trend': span_a_trend,
            'span_b_trend': span_b_trend
        }
    
    def _analyze_chikou_span(self, close: List[float], chikou_span: List[float]) -> Dict[str, Any]:
        """Analyze Chikou Span for confirmation signals"""
        if len(close) < self.displacement + 5 or len(chikou_span) < self.displacement + 5:
            return {'signal': 'UNKNOWN', 'clear_space': False}
        
        # Current Chikou value vs price 26 periods ago
        chikou_current = chikou_span[-(self.displacement + 1)]
        price_26_ago = close[-(self.displacement + 1)]
        
        if chikou_current > price_26_ago:
            signal = "BULLISH"
        elif chikou_current < price_26_ago:
            signal = "BEARISH"
        else:
            signal = "NEUTRAL"
        
        # Check if Chikou has clear space (not overlapping with price)
        clear_space = True
        for i in range(1, min(self.displacement, len(close) - self.displacement)):
            chikou_val = chikou_span[-(self.displacement + 1 - i)]
            price_val = close[-(self.displacement + 1 - i)]
            
            # If Chikou is very close to price, it's not clear space
            if abs(chikou_val - price_val) / price_val < 0.01:  # Within 1%
                clear_space = False
                break
        
        return {
            'signal': signal,
            'clear_space': clear_space,
            'chikou_value': chikou_current,
            'reference_price': price_26_ago
        }
    
    def _identify_support_resistance(self, close: List[float], kijun_sen: List[float], 
                                   cloud_top: List[float], cloud_bottom: List[float]) -> Dict[str, Any]:
        """Identify key support and resistance levels"""
        if not close or not kijun_sen:
            return {'support_levels': [], 'resistance_levels': []}
        
        current_price = close[-1]
        current_kijun = kijun_sen[-1] if kijun_sen else current_price
        
        support_levels = []
        resistance_levels = []
        
        # Kijun-sen as support/resistance
        if current_price > current_kijun:
            support_levels.append({
                'level': current_kijun,
                'type': 'KIJUN_SUPPORT',
                'strength': 'MEDIUM'
            })
        else:
            resistance_levels.append({
                'level': current_kijun,
                'type': 'KIJUN_RESISTANCE',
                'strength': 'MEDIUM'
            })
        
        # Cloud as support/resistance
        if cloud_top and cloud_bottom:
            current_cloud_top = cloud_top[-1] if cloud_top[-1] != 0 else current_price
            current_cloud_bottom = cloud_bottom[-1] if cloud_bottom[-1] != 0 else current_price
            
            if current_price > current_cloud_top:
                support_levels.extend([
                    {
                        'level': current_cloud_top,
                        'type': 'CLOUD_TOP_SUPPORT',
                        'strength': 'STRONG'
                    },
                    {
                        'level': current_cloud_bottom,
                        'type': 'CLOUD_BOTTOM_SUPPORT',
                        'strength': 'STRONG'
                    }
                ])
            elif current_price < current_cloud_bottom:
                resistance_levels.extend([
                    {
                        'level': current_cloud_bottom,
                        'type': 'CLOUD_BOTTOM_RESISTANCE',
                        'strength': 'STRONG'
                    },
                    {
                        'level': current_cloud_top,
                        'type': 'CLOUD_TOP_RESISTANCE',
                        'strength': 'STRONG'
                    }
                ])
        
        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }
    
    def get_chart_data(self, result: IchimokuResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'ichimoku',
            'name': 'Ichimoku Kinko Hyo',
            'data': {
                'tenkan_sen': result.tenkan_sen,
                'kijun_sen': result.kijun_sen,
                'senkou_span_a': result.senkou_span_a,
                'senkou_span_b': result.senkou_span_b,
                'chikou_span': result.chikou_span,
                'cloud_top': result.cloud_top,
                'cloud_bottom': result.cloud_bottom
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'series': [
                {
                    'name': 'Tenkan-sen',
                    'data': result.tenkan_sen,
                    'color': '#FF6B6B',
                    'type': 'line',
                    'lineWidth': 1
                },
                {
                    'name': 'Kijun-sen',
                    'data': result.kijun_sen,
                    'color': '#4ECDC4',
                    'type': 'line',
                    'lineWidth': 2
                },
                {
                    'name': 'Chikou Span',
                    'data': result.chikou_span,
                    'color': '#45B7D1',
                    'type': 'line',
                    'lineWidth': 1,
                    'dashStyle': 'Dash'
                }
            ],
            'cloud': {
                'senkou_span_a': result.senkou_span_a,
                'senkou_span_b': result.senkou_span_b,
                'bullish_color': 'rgba(76, 175, 80, 0.2)',
                'bearish_color': 'rgba(244, 67, 54, 0.2)'
            }
        }


# Example usage
if __name__ == "__main__":
    # Sample OHLC data
    sample_high = [102, 104, 103, 105, 107, 106, 108, 110, 109, 111, 113, 112, 114, 116, 115, 117, 119, 118, 120, 122, 121, 123, 125, 124, 126, 128, 127, 129, 131, 130]
    sample_low = [98, 100, 99, 101, 103, 102, 104, 106, 105, 107, 109, 108, 110, 112, 111, 113, 115, 114, 116, 118, 117, 119, 121, 120, 122, 124, 123, 125, 127, 126]
    sample_close = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115, 117, 116, 118, 120, 119, 121, 123, 122, 124, 126, 125, 127, 129, 128]
    
    # Calculate Ichimoku
    ichimoku_calculator = IchimokuIndicator()
    result = ichimoku_calculator.calculate(sample_high, sample_low, sample_close, AssetType.STOCK)
    
    print(f"Ichimoku Analysis:")
    print(f"Current Price: {result.metadata['current_price']:.2f}")
    print(f"Tenkan-sen: {result.metadata['current_tenkan']:.2f}")
    print(f"Kijun-sen: {result.metadata['current_kijun']:.2f}")
    print(f"Cloud Top: {result.metadata['current_cloud_top']:.2f}")
    print(f"Cloud Bottom: {result.metadata['current_cloud_bottom']:.2f}")
    print(f"Price vs Cloud: {result.metadata['price_vs_cloud']['position']}")
    print(f"Trend: {result.metadata['trend_analysis']['direction']}")
    print(f"Signals: {', '.join(result.signals)}")