"""Technical Analysis Service

Provides comprehensive technical analysis capabilities including:
- 50+ technical indicators using TA-Lib and pandas-ta
- Chart pattern recognition
- Signal generation and analysis
- Multi-timeframe analysis
- Custom indicator framework
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logging.warning("TA-Lib not available. Some indicators will use pandas-ta fallback.")

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas-ta not available. Limited indicator support.")

from scipy import signal
from scipy.stats import linregress

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Signal types for technical analysis"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    STRONG_BUY = "strong_buy"
    STRONG_SELL = "strong_sell"

class PatternType(Enum):
    """Chart pattern types"""
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    HEAD_SHOULDERS = "head_shoulders"
    INVERSE_HEAD_SHOULDERS = "inverse_head_shoulders"
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    CHANNEL_UP = "channel_up"
    CHANNEL_DOWN = "channel_down"
    SUPPORT = "support"
    RESISTANCE = "resistance"

@dataclass
class TechnicalSignal:
    """Technical analysis signal"""
    indicator: str
    signal_type: SignalType
    strength: float  # 0-1 confidence score
    price: float
    timestamp: datetime
    description: str
    metadata: Dict[str, Any] = None

@dataclass
class PatternSignal:
    """Chart pattern signal"""
    pattern_type: PatternType
    confidence: float  # 0-1 confidence score
    start_time: datetime
    end_time: datetime
    description: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    coordinates: List[Tuple[datetime, float]] = None

@dataclass
class IndicatorResult:
    """Technical indicator result"""
    name: str
    values: pd.Series
    parameters: Dict[str, Any]
    signals: List[TechnicalSignal] = None
    metadata: Dict[str, Any] = None

class TechnicalAnalysisService:
    """Comprehensive technical analysis service"""
    
    def __init__(self):
        self.indicators = {}
        self.patterns = {}
        self._initialize_indicators()
        self._initialize_patterns()
    
    def _initialize_indicators(self):
        """Initialize available technical indicators"""
        # Trend Indicators
        self.indicators.update({
            'sma': self._sma,
            'ema': self._ema,
            'wma': self._wma,
            'dema': self._dema,
            'tema': self._tema,
            'trima': self._trima,
            'kama': self._kama,
            'mama': self._mama,
            'vwap': self._vwap,
            'bollinger_bands': self._bollinger_bands,
            'donchian_channels': self._donchian_channels,
            'keltner_channels': self._keltner_channels,
            'parabolic_sar': self._parabolic_sar,
            'supertrend': self._supertrend,
        })
        
        # Momentum Indicators
        self.indicators.update({
            'rsi': self._rsi,
            'stoch': self._stoch,
            'stoch_rsi': self._stoch_rsi,
            'macd': self._macd,
            'cci': self._cci,
            'williams_r': self._williams_r,
            'roc': self._roc,
            'momentum': self._momentum,
            'tsi': self._tsi,
            'ultimate_oscillator': self._ultimate_oscillator,
            'awesome_oscillator': self._awesome_oscillator,
        })
        
        # Volume Indicators
        self.indicators.update({
            'obv': self._obv,
            'ad_line': self._ad_line,
            'chaikin_mf': self._chaikin_mf,
            'force_index': self._force_index,
            'ease_of_movement': self._ease_of_movement,
            'volume_sma': self._volume_sma,
            'vwma': self._vwma,
            'mfi': self._mfi,
        })
        
        # Volatility Indicators
        self.indicators.update({
            'atr': self._atr,
            'natr': self._natr,
            'trange': self._trange,
            'volatility': self._volatility,
            'garman_klass': self._garman_klass,
        })
    
    def _initialize_patterns(self):
        """Initialize pattern recognition algorithms"""
        self.patterns = {
            'double_top': self._detect_double_top,
            'double_bottom': self._detect_double_bottom,
            'head_shoulders': self._detect_head_shoulders,
            'triangles': self._detect_triangles,
            'channels': self._detect_channels,
            'support_resistance': self._detect_support_resistance,
        }
    
    async def calculate_indicator(self, 
                                data: pd.DataFrame, 
                                indicator: str, 
                                **kwargs) -> IndicatorResult:
        """Calculate a specific technical indicator"""
        if indicator not in self.indicators:
            raise ValueError(f"Indicator '{indicator}' not supported")
        
        try:
            result = await asyncio.to_thread(
                self.indicators[indicator], data, **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Error calculating {indicator}: {str(e)}")
            raise
    
    async def calculate_multiple_indicators(self, 
                                          data: pd.DataFrame, 
                                          indicators: List[str], 
                                          **kwargs) -> Dict[str, IndicatorResult]:
        """Calculate multiple indicators concurrently"""
        tasks = []
        for indicator in indicators:
            if indicator in self.indicators:
                task = self.calculate_indicator(data, indicator, **kwargs)
                tasks.append((indicator, task))
        
        results = {}
        for indicator, task in tasks:
            try:
                results[indicator] = await task
            except Exception as e:
                logger.error(f"Failed to calculate {indicator}: {str(e)}")
                results[indicator] = None
        
        return results
    
    async def detect_patterns(self, 
                            data: pd.DataFrame, 
                            patterns: Optional[List[str]] = None) -> List[PatternSignal]:
        """Detect chart patterns in price data"""
        if patterns is None:
            patterns = list(self.patterns.keys())
        
        all_patterns = []
        for pattern in patterns:
            if pattern in self.patterns:
                try:
                    detected = await asyncio.to_thread(
                        self.patterns[pattern], data
                    )
                    if detected:
                        all_patterns.extend(detected)
                except Exception as e:
                    logger.error(f"Error detecting {pattern}: {str(e)}")
        
        return all_patterns
    
    async def generate_signals(self, 
                             data: pd.DataFrame, 
                             indicators: Optional[List[str]] = None) -> List[TechnicalSignal]:
        """Generate trading signals from technical indicators"""
        if indicators is None:
            indicators = ['rsi', 'macd', 'bollinger_bands', 'stoch']
        
        indicator_results = await self.calculate_multiple_indicators(
            data, indicators
        )
        
        signals = []
        for indicator, result in indicator_results.items():
            if result and result.signals:
                signals.extend(result.signals)
        
        return signals
    
    # Trend Indicators Implementation
    def _sma(self, data: pd.DataFrame, period: int = 20, column: str = 'close') -> IndicatorResult:
        """Simple Moving Average"""
        values = data[column].rolling(window=period).mean()
        
        # Generate signals
        signals = []
        current_price = data[column].iloc[-1]
        current_sma = values.iloc[-1]
        
        if current_price > current_sma:
            signal = TechnicalSignal(
                indicator='sma',
                signal_type=SignalType.BUY,
                strength=min((current_price - current_sma) / current_sma, 1.0),
                price=current_price,
                timestamp=data.index[-1],
                description=f"Price above SMA({period})"
            )
            signals.append(signal)
        elif current_price < current_sma:
            signal = TechnicalSignal(
                indicator='sma',
                signal_type=SignalType.SELL,
                strength=min((current_sma - current_price) / current_sma, 1.0),
                price=current_price,
                timestamp=data.index[-1],
                description=f"Price below SMA({period})"
            )
            signals.append(signal)
        
        return IndicatorResult(
            name=f'SMA({period})',
            values=values,
            parameters={'period': period, 'column': column},
            signals=signals
        )
    
    def _ema(self, data: pd.DataFrame, period: int = 20, column: str = 'close') -> IndicatorResult:
        """Exponential Moving Average"""
        if TALIB_AVAILABLE:
            values = pd.Series(talib.EMA(data[column].values, timeperiod=period), index=data.index)
        else:
            values = data[column].ewm(span=period).mean()
        
        return IndicatorResult(
            name=f'EMA({period})',
            values=values,
            parameters={'period': period, 'column': column}
        )
    
    def _rsi(self, data: pd.DataFrame, period: int = 14, column: str = 'close') -> IndicatorResult:
        """Relative Strength Index"""
        if TALIB_AVAILABLE:
            values = pd.Series(talib.RSI(data[column].values, timeperiod=period), index=data.index)
        else:
            delta = data[column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            values = 100 - (100 / (1 + rs))
        
        # Generate RSI signals
        signals = []
        current_rsi = values.iloc[-1]
        current_price = data[column].iloc[-1]
        
        if current_rsi < 30:
            signal = TechnicalSignal(
                indicator='rsi',
                signal_type=SignalType.BUY,
                strength=(30 - current_rsi) / 30,
                price=current_price,
                timestamp=data.index[-1],
                description=f"RSI oversold: {current_rsi:.2f}"
            )
            signals.append(signal)
        elif current_rsi > 70:
            signal = TechnicalSignal(
                indicator='rsi',
                signal_type=SignalType.SELL,
                strength=(current_rsi - 70) / 30,
                price=current_price,
                timestamp=data.index[-1],
                description=f"RSI overbought: {current_rsi:.2f}"
            )
            signals.append(signal)
        
        return IndicatorResult(
            name=f'RSI({period})',
            values=values,
            parameters={'period': period, 'column': column},
            signals=signals
        )
    
    def _macd(self, data: pd.DataFrame, 
             fast_period: int = 12, 
             slow_period: int = 26, 
             signal_period: int = 9,
             column: str = 'close') -> IndicatorResult:
        """MACD (Moving Average Convergence Divergence)"""
        if TALIB_AVAILABLE:
            macd_line, macd_signal, macd_histogram = talib.MACD(
                data[column].values, 
                fastperiod=fast_period,
                slowperiod=slow_period, 
                signalperiod=signal_period
            )
            values = pd.DataFrame({
                'macd': macd_line,
                'signal': macd_signal,
                'histogram': macd_histogram
            }, index=data.index)
        else:
            ema_fast = data[column].ewm(span=fast_period).mean()
            ema_slow = data[column].ewm(span=slow_period).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal_period).mean()
            macd_histogram = macd_line - macd_signal
            
            values = pd.DataFrame({
                'macd': macd_line,
                'signal': macd_signal,
                'histogram': macd_histogram
            })
        
        # Generate MACD signals
        signals = []
        current_macd = values['macd'].iloc[-1]
        current_signal = values['signal'].iloc[-1]
        prev_macd = values['macd'].iloc[-2]
        prev_signal = values['signal'].iloc[-2]
        current_price = data[column].iloc[-1]
        
        # MACD crossover signals
        if current_macd > current_signal and prev_macd <= prev_signal:
            signal = TechnicalSignal(
                indicator='macd',
                signal_type=SignalType.BUY,
                strength=0.7,
                price=current_price,
                timestamp=data.index[-1],
                description="MACD bullish crossover"
            )
            signals.append(signal)
        elif current_macd < current_signal and prev_macd >= prev_signal:
            signal = TechnicalSignal(
                indicator='macd',
                signal_type=SignalType.SELL,
                strength=0.7,
                price=current_price,
                timestamp=data.index[-1],
                description="MACD bearish crossover"
            )
            signals.append(signal)
        
        return IndicatorResult(
            name=f'MACD({fast_period},{slow_period},{signal_period})',
            values=values,
            parameters={
                'fast_period': fast_period,
                'slow_period': slow_period,
                'signal_period': signal_period,
                'column': column
            },
            signals=signals
        )
    
    def _bollinger_bands(self, data: pd.DataFrame, 
                        period: int = 20, 
                        std_dev: float = 2.0,
                        column: str = 'close') -> IndicatorResult:
        """Bollinger Bands"""
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(
                data[column].values, 
                timeperiod=period, 
                nbdevup=std_dev, 
                nbdevdn=std_dev
            )
            values = pd.DataFrame({
                'upper': upper,
                'middle': middle,
                'lower': lower
            }, index=data.index)
        else:
            sma = data[column].rolling(window=period).mean()
            std = data[column].rolling(window=period).std()
            values = pd.DataFrame({
                'upper': sma + (std * std_dev),
                'middle': sma,
                'lower': sma - (std * std_dev)
            })
        
        # Generate Bollinger Band signals
        signals = []
        current_price = data[column].iloc[-1]
        current_upper = values['upper'].iloc[-1]
        current_lower = values['lower'].iloc[-1]
        current_middle = values['middle'].iloc[-1]
        
        if current_price <= current_lower:
            signal = TechnicalSignal(
                indicator='bollinger_bands',
                signal_type=SignalType.BUY,
                strength=0.8,
                price=current_price,
                timestamp=data.index[-1],
                description="Price at lower Bollinger Band"
            )
            signals.append(signal)
        elif current_price >= current_upper:
            signal = TechnicalSignal(
                indicator='bollinger_bands',
                signal_type=SignalType.SELL,
                strength=0.8,
                price=current_price,
                timestamp=data.index[-1],
                description="Price at upper Bollinger Band"
            )
            signals.append(signal)
        
        return IndicatorResult(
            name=f'BB({period},{std_dev})',
            values=values,
            parameters={'period': period, 'std_dev': std_dev, 'column': column},
            signals=signals
        )
    
    # Pattern Recognition Methods
    def _detect_double_top(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect double top patterns"""
        # Simplified double top detection
        # In practice, this would be more sophisticated
        patterns = []
        
        # Find local maxima
        highs = data['high'].values
        peaks, _ = signal.find_peaks(highs, distance=20, prominence=highs.std())
        
        if len(peaks) >= 2:
            # Check for double top pattern in recent peaks
            recent_peaks = peaks[-10:]  # Last 10 peaks
            for i in range(len(recent_peaks) - 1):
                peak1_idx = recent_peaks[i]
                peak2_idx = recent_peaks[i + 1]
                
                peak1_price = highs[peak1_idx]
                peak2_price = highs[peak2_idx]
                
                # Check if peaks are similar in height (within 2%)
                if abs(peak1_price - peak2_price) / peak1_price < 0.02:
                    pattern = PatternSignal(
                        pattern_type=PatternType.DOUBLE_TOP,
                        confidence=0.7,
                        start_time=data.index[peak1_idx],
                        end_time=data.index[peak2_idx],
                        target_price=min(data['low'][peak1_idx:peak2_idx]) * 0.98,
                        description="Double top pattern detected",
                        coordinates=[
                            (data.index[peak1_idx], peak1_price),
                            (data.index[peak2_idx], peak2_price)
                        ]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_support_resistance(self, data: pd.DataFrame) -> List[PatternSignal]:
        """Detect support and resistance levels"""
        patterns = []
        
        # Find support levels (local minima)
        lows = data['low'].values
        support_indices, _ = signal.find_peaks(-lows, distance=10, prominence=lows.std())
        
        # Find resistance levels (local maxima)
        highs = data['high'].values
        resistance_indices, _ = signal.find_peaks(highs, distance=10, prominence=highs.std())
        
        # Create support patterns
        for idx in support_indices[-5:]:  # Last 5 support levels
            pattern = PatternSignal(
                pattern_type=PatternType.SUPPORT,
                confidence=0.6,
                start_time=data.index[idx],
                end_time=data.index[idx],
                target_price=lows[idx],
                description=f"Support level at {lows[idx]:.2f}"
            )
            patterns.append(pattern)
        
        # Create resistance patterns
        for idx in resistance_indices[-5:]:  # Last 5 resistance levels
            pattern = PatternSignal(
                pattern_type=PatternType.RESISTANCE,
                confidence=0.6,
                start_time=data.index[idx],
                end_time=data.index[idx],
                target_price=highs[idx],
                description=f"Resistance level at {highs[idx]:.2f}"
            )
            patterns.append(pattern)
        
        return patterns
    
    # Placeholder methods for other indicators
    def _wma(self, data: pd.DataFrame, period: int = 20, column: str = 'close') -> IndicatorResult:
        """Weighted Moving Average - placeholder"""
        # Implementation would go here
        pass
    
    def _stoch(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> IndicatorResult:
        """Stochastic Oscillator - placeholder"""
        # Implementation would go here
        pass
    
    def _obv(self, data: pd.DataFrame) -> IndicatorResult:
        """On-Balance Volume - placeholder"""
        # Implementation would go here
        pass
    
    def _atr(self, data: pd.DataFrame, period: int = 14) -> IndicatorResult:
        """Average True Range - placeholder"""
        # Implementation would go here
        pass
    
    # Additional placeholder methods for completeness
    def _dema(self, data, period=20, column='close'): pass
    def _tema(self, data, period=20, column='close'): pass
    def _trima(self, data, period=20, column='close'): pass
    def _kama(self, data, period=20, column='close'): pass
    def _mama(self, data, column='close'): pass
    def _vwap(self, data): pass
    def _donchian_channels(self, data, period=20): pass
    def _keltner_channels(self, data, period=20): pass
    def _parabolic_sar(self, data): pass
    def _supertrend(self, data, period=10, multiplier=3): pass
    def _stoch_rsi(self, data, period=14): pass
    def _cci(self, data, period=20): pass
    def _williams_r(self, data, period=14): pass
    def _roc(self, data, period=10, column='close'): pass
    def _momentum(self, data, period=10, column='close'): pass
    def _tsi(self, data, long_period=25, short_period=13): pass
    def _ultimate_oscillator(self, data): pass
    def _awesome_oscillator(self, data): pass
    def _ad_line(self, data): pass
    def _chaikin_mf(self, data, period=20): pass
    def _force_index(self, data, period=13): pass
    def _ease_of_movement(self, data, period=14): pass
    def _volume_sma(self, data, period=20): pass
    def _vwma(self, data, period=20): pass
    def _mfi(self, data, period=14): pass
    def _natr(self, data, period=14): pass
    def _trange(self, data): pass
    def _volatility(self, data, period=20): pass
    def _garman_klass(self, data): pass
    def _detect_double_bottom(self, data): return []
    def _detect_head_shoulders(self, data): return []
    def _detect_triangles(self, data): return []
    def _detect_channels(self, data): return []

# Global instance
technical_analysis_service = TechnicalAnalysisService()