"""Advanced Charting System for FinScope - Phase 7 Implementation

Provides comprehensive charting capabilities including technical indicators,
custom overlays, pattern recognition, and advanced visualization features.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from market_data import MarketDataService
from db_models import ChartTemplate, TechnicalIndicator, ChartAnnotation

logger = logging.getLogger(__name__)

class ChartType(str, Enum):
    """Types of charts"""
    CANDLESTICK = "candlestick"
    LINE = "line"
    AREA = "area"
    VOLUME = "volume"
    HEIKIN_ASHI = "heikin_ashi"
    RENKO = "renko"
    POINT_FIGURE = "point_figure"
    KAGI = "kagi"
    THREE_LINE_BREAK = "three_line_break"

class TimeFrame(str, Enum):
    """Chart timeframes"""
    TICK = "tick"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"

class IndicatorType(str, Enum):
    """Technical indicator types"""
    # Trend Indicators
    SMA = "sma"  # Simple Moving Average
    EMA = "ema"  # Exponential Moving Average
    WMA = "wma"  # Weighted Moving Average
    MACD = "macd"  # Moving Average Convergence Divergence
    ADX = "adx"  # Average Directional Index
    AROON = "aroon"  # Aroon Indicator
    
    # Momentum Indicators
    RSI = "rsi"  # Relative Strength Index
    STOCH = "stoch"  # Stochastic Oscillator
    CCI = "cci"  # Commodity Channel Index
    WILLIAMS_R = "williams_r"  # Williams %R
    ROC = "roc"  # Rate of Change
    
    # Volatility Indicators
    BOLLINGER_BANDS = "bollinger_bands"
    ATR = "atr"  # Average True Range
    KELTNER_CHANNELS = "keltner_channels"
    DONCHIAN_CHANNELS = "donchian_channels"
    
    # Volume Indicators
    OBV = "obv"  # On-Balance Volume
    VOLUME_SMA = "volume_sma"
    VWAP = "vwap"  # Volume Weighted Average Price
    AD_LINE = "ad_line"  # Accumulation/Distribution Line
    
    # Support/Resistance
    PIVOT_POINTS = "pivot_points"
    FIBONACCI_RETRACEMENT = "fibonacci_retracement"
    SUPPORT_RESISTANCE = "support_resistance"

class PatternType(str, Enum):
    """Chart pattern types"""
    # Reversal Patterns
    HEAD_AND_SHOULDERS = "head_and_shoulders"
    INVERSE_HEAD_AND_SHOULDERS = "inverse_head_and_shoulders"
    DOUBLE_TOP = "double_top"
    DOUBLE_BOTTOM = "double_bottom"
    TRIPLE_TOP = "triple_top"
    TRIPLE_BOTTOM = "triple_bottom"
    
    # Continuation Patterns
    TRIANGLE_ASCENDING = "triangle_ascending"
    TRIANGLE_DESCENDING = "triangle_descending"
    TRIANGLE_SYMMETRICAL = "triangle_symmetrical"
    FLAG = "flag"
    PENNANT = "pennant"
    WEDGE_RISING = "wedge_rising"
    WEDGE_FALLING = "wedge_falling"
    
    # Candlestick Patterns
    DOJI = "doji"
    HAMMER = "hammer"
    HANGING_MAN = "hanging_man"
    SHOOTING_STAR = "shooting_star"
    ENGULFING_BULLISH = "engulfing_bullish"
    ENGULFING_BEARISH = "engulfing_bearish"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"

class AnnotationType(str, Enum):
    """Chart annotation types"""
    TREND_LINE = "trend_line"
    HORIZONTAL_LINE = "horizontal_line"
    VERTICAL_LINE = "vertical_line"
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ARROW = "arrow"
    TEXT = "text"
    FIBONACCI = "fibonacci"
    GANN_FAN = "gann_fan"
    ELLIOTT_WAVE = "elliott_wave"

@dataclass
class ChartData:
    """Chart data structure"""
    symbol: str
    timeframe: TimeFrame
    data: pd.DataFrame
    indicators: Dict[str, pd.DataFrame]
    patterns: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    metadata: Dict[str, Any]

@dataclass
class IndicatorConfig:
    """Technical indicator configuration"""
    type: IndicatorType
    parameters: Dict[str, Any]
    display_settings: Dict[str, Any]
    enabled: bool = True

@dataclass
class PatternDetection:
    """Pattern detection result"""
    pattern_type: PatternType
    confidence: float
    start_time: datetime
    end_time: datetime
    key_points: List[Tuple[datetime, float]]
    description: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None

class ChartRequest(BaseModel):
    """Request for chart data"""
    symbol: str
    timeframe: TimeFrame = TimeFrame.DAILY
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    chart_type: ChartType = ChartType.CANDLESTICK
    indicators: List[Dict[str, Any]] = []
    detect_patterns: bool = True
    include_volume: bool = True
    extended_hours: bool = False

class ChartResponse(BaseModel):
    """Response for chart operations"""
    symbol: str
    timeframe: TimeFrame
    chart_type: ChartType
    data_points: int
    price_data: List[Dict[str, Any]]
    volume_data: List[Dict[str, Any]]
    indicators: Dict[str, List[Dict[str, Any]]]
    patterns: List[Dict[str, Any]]
    annotations: List[Dict[str, Any]]
    support_resistance: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class IndicatorRequest(BaseModel):
    """Request for adding technical indicators"""
    symbol: str
    timeframe: TimeFrame
    indicator_type: IndicatorType
    parameters: Dict[str, Any] = {}
    period: int = 14
    source: str = "close"  # open, high, low, close, volume

class PatternScanRequest(BaseModel):
    """Request for pattern scanning"""
    symbols: List[str]
    timeframes: List[TimeFrame] = [TimeFrame.DAILY]
    pattern_types: List[PatternType] = []
    min_confidence: float = 0.7
    lookback_days: int = 30

class AdvancedCharting:
    """Advanced charting and technical analysis system"""
    
    def __init__(self):
        self.market_service = MarketDataService()
        
        # Chart configuration
        self.max_data_points = 5000
        self.default_lookback_days = 365
        self.pattern_confidence_threshold = 0.6
        
        # Technical indicator calculations
        self.indicator_calculators = {
            IndicatorType.SMA: self._calculate_sma,
            IndicatorType.EMA: self._calculate_ema,
            IndicatorType.RSI: self._calculate_rsi,
            IndicatorType.MACD: self._calculate_macd,
            IndicatorType.BOLLINGER_BANDS: self._calculate_bollinger_bands,
            IndicatorType.STOCH: self._calculate_stochastic,
            IndicatorType.ATR: self._calculate_atr,
            IndicatorType.VWAP: self._calculate_vwap,
            IndicatorType.OBV: self._calculate_obv,
            IndicatorType.ADX: self._calculate_adx
        }
        
        # Pattern detection algorithms
        self.pattern_detectors = {
            PatternType.DOJI: self._detect_doji,
            PatternType.HAMMER: self._detect_hammer,
            PatternType.SHOOTING_STAR: self._detect_shooting_star,
            PatternType.ENGULFING_BULLISH: self._detect_bullish_engulfing,
            PatternType.ENGULFING_BEARISH: self._detect_bearish_engulfing,
            PatternType.DOUBLE_TOP: self._detect_double_top,
            PatternType.DOUBLE_BOTTOM: self._detect_double_bottom,
            PatternType.HEAD_AND_SHOULDERS: self._detect_head_and_shoulders
        }
        
        # Chart templates cache
        self._template_cache = {}
        self._indicator_cache = {}
    
    async def get_chart_data(
        self,
        request: ChartRequest,
        db: Session
    ) -> ChartResponse:
        """Get comprehensive chart data with indicators and patterns"""
        try:
            # Get market data
            end_date = request.end_date or datetime.utcnow()
            start_date = request.start_date or (
                end_date - timedelta(days=self.default_lookback_days)
            )
            
            # Fetch price data
            price_data = await self.market_service.get_historical_data(
                request.symbol,
                start_date,
                end_date,
                request.timeframe.value
            )
            
            if price_data.empty:
                raise ValueError(f"No data available for {request.symbol}")
            
            # Limit data points
            if len(price_data) > self.max_data_points:
                price_data = price_data.tail(self.max_data_points)
            
            # Calculate technical indicators
            indicators = {}
            for indicator_config in request.indicators:
                indicator_type = IndicatorType(indicator_config["type"])
                parameters = indicator_config.get("parameters", {})
                
                if indicator_type in self.indicator_calculators:
                    indicator_data = await self.indicator_calculators[indicator_type](
                        price_data, parameters
                    )
                    indicators[indicator_type.value] = indicator_data
            
            # Detect patterns if requested
            patterns = []
            if request.detect_patterns:
                patterns = await self._detect_chart_patterns(
                    price_data, request.symbol
                )
            
            # Calculate support and resistance levels
            support_resistance = await self._calculate_support_resistance(
                price_data
            )
            
            # Format response data
            price_data_formatted = self._format_price_data(
                price_data, request.chart_type
            )
            
            volume_data_formatted = []
            if request.include_volume and 'volume' in price_data.columns:
                volume_data_formatted = self._format_volume_data(price_data)
            
            indicators_formatted = {}
            for name, data in indicators.items():
                indicators_formatted[name] = self._format_indicator_data(data)
            
            patterns_formatted = [
                self._format_pattern_data(pattern) for pattern in patterns
            ]
            
            support_resistance_formatted = [
                self._format_support_resistance_data(level)
                for level in support_resistance
            ]
            
            # Create response
            response = ChartResponse(
                symbol=request.symbol,
                timeframe=request.timeframe,
                chart_type=request.chart_type,
                data_points=len(price_data),
                price_data=price_data_formatted,
                volume_data=volume_data_formatted,
                indicators=indicators_formatted,
                patterns=patterns_formatted,
                annotations=[],  # Would load from database
                support_resistance=support_resistance_formatted,
                metadata={
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "data_source": "market_data_service",
                    "last_updated": datetime.utcnow().isoformat()
                }
            )
            
            logger.info(
                f"Chart data generated for {request.symbol} "
                f"({request.timeframe.value}): {len(price_data)} points"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting chart data: {str(e)}")
            raise
    
    async def calculate_indicator(
        self,
        request: IndicatorRequest,
        db: Session
    ) -> Dict[str, Any]:
        """Calculate a specific technical indicator"""
        try:
            # Get market data
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=self.default_lookback_days)
            
            price_data = await self.market_service.get_historical_data(
                request.symbol,
                start_date,
                end_date,
                request.timeframe.value
            )
            
            if price_data.empty:
                raise ValueError(f"No data available for {request.symbol}")
            
            # Calculate indicator
            if request.indicator_type in self.indicator_calculators:
                parameters = {
                    "period": request.period,
                    "source": request.source,
                    **request.parameters
                }
                
                indicator_data = await self.indicator_calculators[request.indicator_type](
                    price_data, parameters
                )
                
                return {
                    "symbol": request.symbol,
                    "indicator_type": request.indicator_type.value,
                    "parameters": parameters,
                    "data": self._format_indicator_data(indicator_data),
                    "calculated_at": datetime.utcnow().isoformat()
                }
            
            else:
                raise ValueError(f"Indicator type {request.indicator_type.value} not supported")
            
        except Exception as e:
            logger.error(f"Error calculating indicator: {str(e)}")
            raise
    
    async def scan_patterns(
        self,
        request: PatternScanRequest,
        db: Session
    ) -> List[Dict[str, Any]]:
        """Scan multiple symbols for chart patterns"""
        try:
            results = []
            
            for symbol in request.symbols:
                for timeframe in request.timeframes:
                    try:
                        # Get market data
                        end_date = datetime.utcnow()
                        start_date = end_date - timedelta(days=request.lookback_days)
                        
                        price_data = await self.market_service.get_historical_data(
                            symbol,
                            start_date,
                            end_date,
                            timeframe.value
                        )
                        
                        if price_data.empty:
                            continue
                        
                        # Detect patterns
                        patterns = await self._detect_chart_patterns(
                            price_data, symbol, request.pattern_types
                        )
                        
                        # Filter by confidence
                        filtered_patterns = [
                            pattern for pattern in patterns
                            if pattern.confidence >= request.min_confidence
                        ]
                        
                        for pattern in filtered_patterns:
                            results.append({
                                "symbol": symbol,
                                "timeframe": timeframe.value,
                                "pattern_type": pattern.pattern_type.value,
                                "confidence": pattern.confidence,
                                "start_time": pattern.start_time.isoformat(),
                                "end_time": pattern.end_time.isoformat(),
                                "description": pattern.description,
                                "target_price": pattern.target_price,
                                "stop_loss": pattern.stop_loss
                            })
                    
                    except Exception as e:
                        logger.error(f"Error scanning {symbol} {timeframe.value}: {str(e)}")
                        continue
            
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
            logger.info(f"Pattern scan completed: {len(results)} patterns found")
            
            return results
            
        except Exception as e:
            logger.error(f"Error scanning patterns: {str(e)}")
            return []
    
    async def _detect_chart_patterns(
        self,
        price_data: pd.DataFrame,
        symbol: str,
        pattern_types: Optional[List[PatternType]] = None
    ) -> List[PatternDetection]:
        """Detect chart patterns in price data"""
        try:
            patterns = []
            
            # Use all pattern types if none specified
            if not pattern_types:
                pattern_types = list(self.pattern_detectors.keys())
            
            for pattern_type in pattern_types:
                if pattern_type in self.pattern_detectors:
                    try:
                        detected_patterns = await self.pattern_detectors[pattern_type](
                            price_data
                        )
                        patterns.extend(detected_patterns)
                    except Exception as e:
                        logger.error(f"Error detecting {pattern_type.value}: {str(e)}")
            
            # Filter by confidence threshold
            filtered_patterns = [
                pattern for pattern in patterns
                if pattern.confidence >= self.pattern_confidence_threshold
            ]
            
            return filtered_patterns
            
        except Exception as e:
            logger.error(f"Error detecting chart patterns: {str(e)}")
            return []
    
    async def _calculate_support_resistance(
        self,
        price_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Calculate support and resistance levels"""
        try:
            levels = []
            
            # Use pivot points method
            highs = price_data['high'].values
            lows = price_data['low'].values
            
            # Find local maxima (resistance)
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    
                    # Check for multiple touches
                    touches = sum(1 for h in highs[max(0, i-10):i+10] 
                                 if abs(h - highs[i]) / highs[i] < 0.01)
                    
                    if touches >= 2:
                        levels.append({
                            "type": "resistance",
                            "price": highs[i],
                            "strength": touches,
                            "timestamp": price_data.index[i].isoformat()
                        })
            
            # Find local minima (support)
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    
                    # Check for multiple touches
                    touches = sum(1 for l in lows[max(0, i-10):i+10] 
                                 if abs(l - lows[i]) / lows[i] < 0.01)
                    
                    if touches >= 2:
                        levels.append({
                            "type": "support",
                            "price": lows[i],
                            "strength": touches,
                            "timestamp": price_data.index[i].isoformat()
                        })
            
            # Sort by strength
            levels.sort(key=lambda x: x["strength"], reverse=True)
            
            return levels[:10]  # Return top 10 levels
            
        except Exception as e:
            logger.error(f"Error calculating support/resistance: {str(e)}")
            return []
    
    # Technical Indicator Calculations
    
    async def _calculate_sma(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Simple Moving Average"""
        period = params.get("period", 20)
        source = params.get("source", "close")
        
        sma = data[source].rolling(window=period).mean()
        
        return pd.DataFrame({
            "sma": sma,
            "timestamp": data.index
        })
    
    async def _calculate_ema(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Exponential Moving Average"""
        period = params.get("period", 20)
        source = params.get("source", "close")
        
        ema = data[source].ewm(span=period).mean()
        
        return pd.DataFrame({
            "ema": ema,
            "timestamp": data.index
        })
    
    async def _calculate_rsi(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Relative Strength Index"""
        period = params.get("period", 14)
        source = params.get("source", "close")
        
        delta = data[source].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return pd.DataFrame({
            "rsi": rsi,
            "timestamp": data.index
        })
    
    async def _calculate_macd(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate MACD"""
        fast_period = params.get("fast_period", 12)
        slow_period = params.get("slow_period", 26)
        signal_period = params.get("signal_period", 9)
        source = params.get("source", "close")
        
        ema_fast = data[source].ewm(span=fast_period).mean()
        ema_slow = data[source].ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
            "timestamp": data.index
        })
    
    async def _calculate_bollinger_bands(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        period = params.get("period", 20)
        std_dev = params.get("std_dev", 2)
        source = params.get("source", "close")
        
        sma = data[source].rolling(window=period).mean()
        std = data[source].rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return pd.DataFrame({
            "upper_band": upper_band,
            "middle_band": sma,
            "lower_band": lower_band,
            "timestamp": data.index
        })
    
    async def _calculate_stochastic(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Stochastic Oscillator"""
        k_period = params.get("k_period", 14)
        d_period = params.get("d_period", 3)
        
        lowest_low = data['low'].rolling(window=k_period).min()
        highest_high = data['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((data['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return pd.DataFrame({
            "k_percent": k_percent,
            "d_percent": d_percent,
            "timestamp": data.index
        })
    
    async def _calculate_atr(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Average True Range"""
        period = params.get("period", 14)
        
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift())
        low_close_prev = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()
        
        return pd.DataFrame({
            "atr": atr,
            "timestamp": data.index
        })
    
    async def _calculate_vwap(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price"""
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        vwap = (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
        
        return pd.DataFrame({
            "vwap": vwap,
            "timestamp": data.index
        })
    
    async def _calculate_obv(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate On-Balance Volume"""
        obv = np.where(data['close'] > data['close'].shift(), data['volume'],
                      np.where(data['close'] < data['close'].shift(), -data['volume'], 0)).cumsum()
        
        return pd.DataFrame({
            "obv": obv,
            "timestamp": data.index
        })
    
    async def _calculate_adx(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """Calculate Average Directional Index"""
        period = params.get("period", 14)
        
        # Calculate True Range
        high_low = data['high'] - data['low']
        high_close_prev = np.abs(data['high'] - data['close'].shift())
        low_close_prev = np.abs(data['low'] - data['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        
        # Calculate Directional Movement
        plus_dm = np.where((data['high'] - data['high'].shift()) > (data['low'].shift() - data['low']),
                          np.maximum(data['high'] - data['high'].shift(), 0), 0)
        minus_dm = np.where((data['low'].shift() - data['low']) > (data['high'] - data['high'].shift()),
                           np.maximum(data['low'].shift() - data['low'], 0), 0)
        
        # Smooth the values
        atr = true_range.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return pd.DataFrame({
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "timestamp": data.index
        })
    
    # Pattern Detection Methods
    
    async def _detect_doji(self, data: pd.DataFrame) -> List[PatternDetection]:
        """Detect Doji candlestick patterns"""
        patterns = []
        
        for i in range(len(data)):
            open_price = data.iloc[i]['open']
            close_price = data.iloc[i]['close']
            high_price = data.iloc[i]['high']
            low_price = data.iloc[i]['low']
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            # Doji: very small body relative to total range
            if total_range > 0 and body_size / total_range < 0.1:
                confidence = 1 - (body_size / total_range) * 10
                
                patterns.append(PatternDetection(
                    pattern_type=PatternType.DOJI,
                    confidence=confidence,
                    start_time=data.index[i],
                    end_time=data.index[i],
                    key_points=[(data.index[i], close_price)],
                    description=f"Doji pattern detected with {confidence:.1%} confidence"
                ))
        
        return patterns
    
    async def _detect_hammer(self, data: pd.DataFrame) -> List[PatternDetection]:
        """Detect Hammer candlestick patterns"""
        patterns = []
        
        for i in range(len(data)):
            open_price = data.iloc[i]['open']
            close_price = data.iloc[i]['close']
            high_price = data.iloc[i]['high']
            low_price = data.iloc[i]['low']
            
            body_size = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)
            total_range = high_price - low_price
            
            # Hammer: small body, long lower shadow, small upper shadow
            if (total_range > 0 and
                body_size / total_range < 0.3 and
                lower_shadow / total_range > 0.6 and
                upper_shadow / total_range < 0.1):
                
                confidence = min(lower_shadow / total_range, 1.0)
                
                patterns.append(PatternDetection(
                    pattern_type=PatternType.HAMMER,
                    confidence=confidence,
                    start_time=data.index[i],
                    end_time=data.index[i],
                    key_points=[(data.index[i], close_price)],
                    description=f"Hammer pattern detected with {confidence:.1%} confidence"
                ))
        
        return patterns
    
    async def _detect_shooting_star(self, data: pd.DataFrame) -> List[PatternDetection]:
        """Detect Shooting Star candlestick patterns"""
        patterns = []
        
        for i in range(len(data)):
            open_price = data.iloc[i]['open']
            close_price = data.iloc[i]['close']
            high_price = data.iloc[i]['high']
            low_price = data.iloc[i]['low']
            
            body_size = abs(close_price - open_price)
            lower_shadow = min(open_price, close_price) - low_price
            upper_shadow = high_price - max(open_price, close_price)
            total_range = high_price - low_price
            
            # Shooting Star: small body, long upper shadow, small lower shadow
            if (total_range > 0 and
                body_size / total_range < 0.3 and
                upper_shadow / total_range > 0.6 and
                lower_shadow / total_range < 0.1):
                
                confidence = min(upper_shadow / total_range, 1.0)
                
                patterns.append(PatternDetection(
                    pattern_type=PatternType.SHOOTING_STAR,
                    confidence=confidence,
                    start_time=data.index[i],
                    end_time=data.index[i],
                    key_points=[(data.index[i], close_price)],
                    description=f"Shooting Star pattern detected with {confidence:.1%} confidence"
                ))
        
        return patterns
    
    async def _detect_bullish_engulfing(self, data: pd.DataFrame) -> List[PatternDetection]:
        """Detect Bullish Engulfing patterns"""
        patterns = []
        
        for i in range(1, len(data)):
            prev_open = data.iloc[i-1]['open']
            prev_close = data.iloc[i-1]['close']
            curr_open = data.iloc[i]['open']
            curr_close = data.iloc[i]['close']
            
            # Previous candle is bearish, current is bullish and engulfs previous
            if (prev_close < prev_open and  # Previous bearish
                curr_close > curr_open and  # Current bullish
                curr_open < prev_close and  # Current opens below previous close
                curr_close > prev_open):    # Current closes above previous open
                
                confidence = min(
                    (curr_close - curr_open) / (prev_open - prev_close),
                    1.0
                )
                
                patterns.append(PatternDetection(
                    pattern_type=PatternType.ENGULFING_BULLISH,
                    confidence=confidence,
                    start_time=data.index[i-1],
                    end_time=data.index[i],
                    key_points=[
                        (data.index[i-1], prev_close),
                        (data.index[i], curr_close)
                    ],
                    description=f"Bullish Engulfing pattern detected with {confidence:.1%} confidence"
                ))
        
        return patterns
    
    async def _detect_bearish_engulfing(self, data: pd.DataFrame) -> List[PatternDetection]:
        """Detect Bearish Engulfing patterns"""
        patterns = []
        
        for i in range(1, len(data)):
            prev_open = data.iloc[i-1]['open']
            prev_close = data.iloc[i-1]['close']
            curr_open = data.iloc[i]['open']
            curr_close = data.iloc[i]['close']
            
            # Previous candle is bullish, current is bearish and engulfs previous
            if (prev_close > prev_open and  # Previous bullish
                curr_close < curr_open and  # Current bearish
                curr_open > prev_close and  # Current opens above previous close
                curr_close < prev_open):    # Current closes below previous open
                
                confidence = min(
                    (curr_open - curr_close) / (prev_close - prev_open),
                    1.0
                )
                
                patterns.append(PatternDetection(
                    pattern_type=PatternType.ENGULFING_BEARISH,
                    confidence=confidence,
                    start_time=data.index[i-1],
                    end_time=data.index[i],
                    key_points=[
                        (data.index[i-1], prev_close),
                        (data.index[i], curr_close)
                    ],
                    description=f"Bearish Engulfing pattern detected with {confidence:.1%} confidence"
                ))
        
        return patterns
    
    async def _detect_double_top(self, data: pd.DataFrame) -> List[PatternDetection]:
        """Detect Double Top patterns"""
        patterns = []
        
        # Simplified double top detection
        highs = data['high'].values
        
        for i in range(20, len(highs) - 20):
            # Find local maxima
            if (highs[i] > highs[i-5:i].max() and
                highs[i] > highs[i+1:i+6].max()):
                
                # Look for another peak within reasonable distance
                for j in range(i+10, min(i+50, len(highs))):
                    if (highs[j] > highs[j-5:j].max() and
                        highs[j] > highs[j+1:j+6].max() and
                        abs(highs[i] - highs[j]) / highs[i] < 0.02):  # Similar height
                        
                        confidence = 1 - abs(highs[i] - highs[j]) / highs[i]
                        
                        patterns.append(PatternDetection(
                            pattern_type=PatternType.DOUBLE_TOP,
                            confidence=confidence,
                            start_time=data.index[i],
                            end_time=data.index[j],
                            key_points=[
                                (data.index[i], highs[i]),
                                (data.index[j], highs[j])
                            ],
                            description=f"Double Top pattern detected with {confidence:.1%} confidence"
                        ))
                        break
        
        return patterns
    
    async def _detect_double_bottom(self, data: pd.DataFrame) -> List[PatternDetection]:
        """Detect Double Bottom patterns"""
        patterns = []
        
        # Simplified double bottom detection
        lows = data['low'].values
        
        for i in range(20, len(lows) - 20):
            # Find local minima
            if (lows[i] < lows[i-5:i].min() and
                lows[i] < lows[i+1:i+6].min()):
                
                # Look for another trough within reasonable distance
                for j in range(i+10, min(i+50, len(lows))):
                    if (lows[j] < lows[j-5:j].min() and
                        lows[j] < lows[j+1:j+6].min() and
                        abs(lows[i] - lows[j]) / lows[i] < 0.02):  # Similar depth
                        
                        confidence = 1 - abs(lows[i] - lows[j]) / lows[i]
                        
                        patterns.append(PatternDetection(
                            pattern_type=PatternType.DOUBLE_BOTTOM,
                            confidence=confidence,
                            start_time=data.index[i],
                            end_time=data.index[j],
                            key_points=[
                                (data.index[i], lows[i]),
                                (data.index[j], lows[j])
                            ],
                            description=f"Double Bottom pattern detected with {confidence:.1%} confidence"
                        ))
                        break
        
        return patterns
    
    async def _detect_head_and_shoulders(self, data: pd.DataFrame) -> List[PatternDetection]:
        """Detect Head and Shoulders patterns"""
        patterns = []
        
        # Simplified head and shoulders detection
        highs = data['high'].values
        
        for i in range(30, len(highs) - 30):
            # Find potential head (highest point)
            if (highs[i] > highs[i-10:i].max() and
                highs[i] > highs[i+1:i+11].max()):
                
                # Look for left shoulder
                left_shoulder_idx = None
                for j in range(i-30, i-10):
                    if (highs[j] > highs[j-5:j].max() and
                        highs[j] > highs[j+1:j+6].max() and
                        highs[j] < highs[i] * 0.95):  # Lower than head
                        left_shoulder_idx = j
                        break
                
                # Look for right shoulder
                right_shoulder_idx = None
                for k in range(i+10, i+30):
                    if (highs[k] > highs[k-5:k].max() and
                        highs[k] > highs[k+1:k+6].max() and
                        highs[k] < highs[i] * 0.95):  # Lower than head
                        right_shoulder_idx = k
                        break
                
                if left_shoulder_idx and right_shoulder_idx:
                    # Check if shoulders are roughly equal
                    shoulder_diff = abs(highs[left_shoulder_idx] - highs[right_shoulder_idx])
                    if shoulder_diff / highs[left_shoulder_idx] < 0.05:
                        
                        confidence = 1 - shoulder_diff / highs[left_shoulder_idx]
                        
                        patterns.append(PatternDetection(
                            pattern_type=PatternType.HEAD_AND_SHOULDERS,
                            confidence=confidence,
                            start_time=data.index[left_shoulder_idx],
                            end_time=data.index[right_shoulder_idx],
                            key_points=[
                                (data.index[left_shoulder_idx], highs[left_shoulder_idx]),
                                (data.index[i], highs[i]),
                                (data.index[right_shoulder_idx], highs[right_shoulder_idx])
                            ],
                            description=f"Head and Shoulders pattern detected with {confidence:.1%} confidence"
                        ))
        
        return patterns
    
    # Data Formatting Methods
    
    def _format_price_data(self, data: pd.DataFrame, chart_type: ChartType) -> List[Dict[str, Any]]:
        """Format price data for chart display"""
        formatted_data = []
        
        for i, row in data.iterrows():
            if chart_type == ChartType.CANDLESTICK:
                formatted_data.append({
                    "timestamp": i.isoformat(),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close'])
                })
            elif chart_type == ChartType.LINE:
                formatted_data.append({
                    "timestamp": i.isoformat(),
                    "value": float(row['close'])
                })
            elif chart_type == ChartType.AREA:
                formatted_data.append({
                    "timestamp": i.isoformat(),
                    "value": float(row['close'])
                })
        
        return formatted_data
    
    def _format_volume_data(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format volume data for chart display"""
        return [
            {
                "timestamp": i.isoformat(),
                "volume": float(row['volume'])
            }
            for i, row in data.iterrows()
            if 'volume' in row and not pd.isna(row['volume'])
        ]
    
    def _format_indicator_data(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Format indicator data for chart display"""
        formatted_data = []
        
        for i, row in data.iterrows():
            point = {"timestamp": i.isoformat()}
            
            for column in data.columns:
                if column != "timestamp" and not pd.isna(row[column]):
                    point[column] = float(row[column])
            
            formatted_data.append(point)
        
        return formatted_data
    
    def _format_pattern_data(self, pattern: PatternDetection) -> Dict[str, Any]:
        """Format pattern data for chart display"""
        return {
            "pattern_type": pattern.pattern_type.value,
            "confidence": pattern.confidence,
            "start_time": pattern.start_time.isoformat(),
            "end_time": pattern.end_time.isoformat(),
            "key_points": [
                {
                    "timestamp": point[0].isoformat(),
                    "price": point[1]
                }
                for point in pattern.key_points
            ],
            "description": pattern.description,
            "target_price": pattern.target_price,
            "stop_loss": pattern.stop_loss
        }
    
    def _format_support_resistance_data(self, level: Dict[str, Any]) -> Dict[str, Any]:
        """Format support/resistance data for chart display"""
        return {
            "type": level["type"],
            "price": level["price"],
            "strength": level["strength"],
            "timestamp": level["timestamp"]
        }

# Global advanced charting instance
advanced_charting = AdvancedCharting()

def get_advanced_charting() -> AdvancedCharting:
    """Get advanced charting instance"""
    return advanced_charting