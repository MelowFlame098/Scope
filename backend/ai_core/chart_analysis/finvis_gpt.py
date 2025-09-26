# FinVis-GPT Component
# Chart Interpreter and Feature Extractor for Financial Charts

import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from datetime import datetime, timedelta
import json
import base64
from PIL import Image
import openai
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class IndicatorType(Enum):
    """Types of technical indicators"""
    SMA = "simple_moving_average"
    EMA = "exponential_moving_average"
    RSI = "relative_strength_index"
    MACD = "moving_average_convergence_divergence"
    BOLLINGER_BANDS = "bollinger_bands"
    STOCHASTIC = "stochastic_oscillator"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    TREND_LINES = "trend_lines"
    FIBONACCI = "fibonacci_retracement"
    CUSTOM = "custom_indicator"

@dataclass
class OHLCData:
    """OHLC candlestick data point"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None
    
@dataclass
class IndicatorData:
    """Technical indicator data"""
    type: IndicatorType
    name: str
    values: List[float]
    timestamps: List[datetime]
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ChartFeatures:
    """Extracted features from financial chart"""
    ohlc_data: List[OHLCData]
    indicators: List[IndicatorData]
    timeframe: str
    symbol: Optional[str]
    chart_period: Tuple[datetime, datetime]
    price_range: Tuple[float, float]
    volume_data: Optional[List[float]]
    support_levels: List[float]
    resistance_levels: List[float]
    trend_lines: List[Dict[str, Any]]
    custom_indicators: List[IndicatorData]
    extraction_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class FinVisGPT:
    """
    FinVis-GPT - Chart Interpreter and Feature Extractor
    
    Advanced AI-powered component for extracting structured data from financial charts.
    Combines computer vision techniques with LLM capabilities to:
    - Extract OHLC candlestick data
    - Identify and extract standard technical indicators
    - Process custom indicators
    - Detect support/resistance levels
    - Extract trend lines and patterns
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # API configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.use_vision_api = self.config.get('use_vision_api', True)
        self.model_name = self.config.get('model_name', 'gpt-4-vision-preview')
        
        # Computer vision parameters
        self.cv_config = {
            'edge_threshold': self.config.get('edge_threshold', 50),
            'contour_min_area': self.config.get('contour_min_area', 100),
            'line_detection_threshold': self.config.get('line_detection_threshold', 100),
            'color_tolerance': self.config.get('color_tolerance', 30)
        }
        
        # Standard indicator configurations
        self.standard_indicators = {
            IndicatorType.SMA: {'periods': [20, 50, 200], 'colors': ['blue', 'orange', 'red']},
            IndicatorType.EMA: {'periods': [12, 26], 'colors': ['green', 'purple']},
            IndicatorType.RSI: {'period': 14, 'overbought': 70, 'oversold': 30},
            IndicatorType.MACD: {'fast': 12, 'slow': 26, 'signal': 9},
            IndicatorType.BOLLINGER_BANDS: {'period': 20, 'std_dev': 2}
        }
        
        # Initialize OpenAI if available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        logger.info("FinVis-GPT initialized with config: %s", self.config)
    
    async def extract_features(self, chart_image: np.ndarray, custom_indicators: Optional[List[str]] = None) -> ChartFeatures:
        """
        Main feature extraction method.
        
        Args:
            chart_image: Preprocessed chart image as numpy array
            custom_indicators: List of custom indicator names to extract
            
        Returns:
            ChartFeatures object with all extracted data
        """
        try:
            logger.info("Starting feature extraction from chart")
            
            # Step 1: Extract OHLC data using computer vision
            ohlc_data = await self._extract_ohlc_data(chart_image)
            
            # Step 2: Extract standard technical indicators
            indicators = await self._extract_standard_indicators(chart_image, ohlc_data)
            
            # Step 3: Extract custom indicators if specified
            custom_indicator_data = []
            if custom_indicators:
                custom_indicator_data = await self._extract_custom_indicators(chart_image, custom_indicators)
            
            # Step 4: Detect support and resistance levels
            support_levels, resistance_levels = await self._detect_support_resistance(chart_image, ohlc_data)
            
            # Step 5: Extract trend lines
            trend_lines = await self._extract_trend_lines(chart_image)
            
            # Step 6: Determine timeframe and period
            timeframe, chart_period = await self._determine_timeframe(chart_image, ohlc_data)
            
            # Step 7: Extract volume data if present
            volume_data = await self._extract_volume_data(chart_image)
            
            # Step 8: Calculate price range
            price_range = self._calculate_price_range(ohlc_data)
            
            # Step 9: Use LLM for additional analysis if available
            llm_analysis = await self._llm_analysis(chart_image, ohlc_data)
            
            # Step 10: Calculate overall extraction confidence
            extraction_confidence = self._calculate_extraction_confidence(
                ohlc_data, indicators, support_levels, resistance_levels
            )
            
            # Compile results
            features = ChartFeatures(
                ohlc_data=ohlc_data,
                indicators=indicators,
                timeframe=timeframe,
                symbol=llm_analysis.get('symbol'),
                chart_period=chart_period,
                price_range=price_range,
                volume_data=volume_data,
                support_levels=support_levels,
                resistance_levels=resistance_levels,
                trend_lines=trend_lines,
                custom_indicators=custom_indicator_data,
                extraction_confidence=extraction_confidence,
                metadata={
                    'extraction_timestamp': datetime.now().isoformat(),
                    'llm_analysis': llm_analysis,
                    'cv_config': self.cv_config
                }
            )
            
            logger.info(f"Feature extraction completed with confidence: {extraction_confidence:.2f}")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._create_empty_features()
    
    async def _extract_ohlc_data(self, chart_image: np.ndarray) -> List[OHLCData]:
        """Extract OHLC candlestick data from chart image"""
        try:
            logger.info("Extracting OHLC data")
            
            # Convert to grayscale for processing
            if len(chart_image.shape) == 3:
                gray = cv2.cvtColor(chart_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = chart_image
            
            # Detect candlestick bodies and wicks
            candlesticks = await self._detect_candlesticks(chart_image, gray)
            
            # Extract price levels from y-axis
            price_levels = await self._extract_price_levels(chart_image)
            
            # Extract time information from x-axis
            time_points = await self._extract_time_points(chart_image)
            
            # Combine candlestick positions with price and time data
            ohlc_data = await self._combine_ohlc_data(candlesticks, price_levels, time_points)
            
            logger.info(f"Extracted {len(ohlc_data)} OHLC data points")
            return ohlc_data
            
        except Exception as e:
            logger.error(f"OHLC extraction failed: {e}")
            return []
    
    async def _detect_candlesticks(self, color_image: np.ndarray, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect individual candlestick elements"""
        try:
            candlesticks = []
            
            # Detect red and green candlestick bodies
            red_bodies = self._detect_colored_rectangles(color_image, 'red')
            green_bodies = self._detect_colored_rectangles(color_image, 'green')
            
            # Detect wicks (thin vertical lines)
            wicks = self._detect_vertical_lines(gray_image)
            
            # Combine bodies and wicks into candlestick structures
            all_bodies = red_bodies + green_bodies
            
            for body in all_bodies:
                # Find associated wicks
                associated_wicks = self._find_associated_wicks(body, wicks)
                
                candlestick = {
                    'body': body,
                    'wicks': associated_wicks,
                    'x_center': body['x'] + body['width'] // 2,
                    'body_top': body['y'],
                    'body_bottom': body['y'] + body['height'],
                    'color': body['color']
                }
                
                # Calculate high and low from wicks
                if associated_wicks:
                    wick_tops = [w['y'] for w in associated_wicks]
                    wick_bottoms = [w['y'] + w['height'] for w in associated_wicks]
                    candlestick['high'] = min(wick_tops + [candlestick['body_top']])
                    candlestick['low'] = max(wick_bottoms + [candlestick['body_bottom']])
                else:
                    candlestick['high'] = candlestick['body_top']
                    candlestick['low'] = candlestick['body_bottom']
                
                candlesticks.append(candlestick)
            
            # Sort by x position
            candlesticks.sort(key=lambda x: x['x_center'])
            
            return candlesticks
            
        except Exception as e:
            logger.error(f"Candlestick detection failed: {e}")
            return []
    
    def _detect_colored_rectangles(self, image: np.ndarray, color: str) -> List[Dict[str, Any]]:
        """Detect rectangular shapes of specific colors"""
        try:
            rectangles = []
            
            # Define color ranges
            color_ranges = {
                'red': ([150, 0, 0], [255, 100, 100]),
                'green': ([0, 150, 0], [100, 255, 100])
            }
            
            if color not in color_ranges:
                return rectangles
            
            lower, upper = color_ranges[color]
            mask = cv2.inRange(image, np.array(lower), np.array(upper))
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.cv_config['contour_min_area']:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter for rectangular shapes
                    if w > 3 and h > 5:  # Minimum candlestick body size
                        rectangles.append({
                            'x': x, 'y': y, 'width': w, 'height': h,
                            'area': area, 'color': color
                        })
            
            return rectangles
            
        except Exception as e:
            logger.error(f"Colored rectangle detection failed: {e}")
            return []
    
    def _detect_vertical_lines(self, gray_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect thin vertical lines (candlestick wicks)"""
        try:
            # Use morphological operations to detect thin vertical lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
            vertical_mask = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
            
            # Find contours of vertical lines
            contours, _ = cv2.findContours(vertical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            lines = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter for thin vertical lines
                if w <= 3 and h >= 5:  # Wick criteria
                    lines.append({
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'x_center': x + w // 2
                    })
            
            return lines
            
        except Exception as e:
            logger.error(f"Vertical line detection failed: {e}")
            return []
    
    def _find_associated_wicks(self, body: Dict[str, Any], wicks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find wicks associated with a candlestick body"""
        associated = []
        body_center = body['x'] + body['width'] // 2
        
        for wick in wicks:
            wick_center = wick['x_center']
            
            # Check if wick is horizontally aligned with body
            if abs(wick_center - body_center) <= 5:  # Tolerance for alignment
                # Check if wick extends beyond body
                wick_top = wick['y']
                wick_bottom = wick['y'] + wick['height']
                body_top = body['y']
                body_bottom = body['y'] + body['height']
                
                if wick_top < body_top or wick_bottom > body_bottom:
                    associated.append(wick)
        
        return associated
    
    async def _extract_price_levels(self, chart_image: np.ndarray) -> Dict[str, float]:
        """Extract price level information from y-axis"""
        try:
            # This would typically involve OCR on the y-axis labels
            # For now, we'll use a simplified approach
            
            height = chart_image.shape[0]
            
            # Estimate price range based on chart area
            # This is a placeholder - real implementation would use OCR
            estimated_high = 100.0  # Would be extracted from chart
            estimated_low = 50.0    # Would be extracted from chart
            
            price_per_pixel = (estimated_high - estimated_low) / height
            
            return {
                'high': estimated_high,
                'low': estimated_low,
                'price_per_pixel': price_per_pixel
            }
            
        except Exception as e:
            logger.error(f"Price level extraction failed: {e}")
            return {'high': 100.0, 'low': 50.0, 'price_per_pixel': 0.1}
    
    async def _extract_time_points(self, chart_image: np.ndarray) -> List[datetime]:
        """Extract time information from x-axis"""
        try:
            # This would typically involve OCR on the x-axis labels
            # For now, we'll generate estimated time points
            
            width = chart_image.shape[1]
            num_points = width // 20  # Estimate one data point per 20 pixels
            
            # Generate time series (placeholder)
            base_time = datetime.now() - timedelta(days=num_points)
            time_points = [base_time + timedelta(days=i) for i in range(num_points)]
            
            return time_points
            
        except Exception as e:
            logger.error(f"Time point extraction failed: {e}")
            return [datetime.now()]
    
    async def _combine_ohlc_data(self, candlesticks: List[Dict], price_levels: Dict, time_points: List[datetime]) -> List[OHLCData]:
        """Combine candlestick positions with price and time data"""
        try:
            ohlc_data = []
            
            for i, candlestick in enumerate(candlesticks):
                if i < len(time_points):
                    # Convert pixel positions to prices
                    price_per_pixel = price_levels['price_per_pixel']
                    chart_height = candlestick.get('chart_height', 400)  # Default height
                    
                    # Calculate OHLC values from pixel positions
                    high_price = price_levels['high'] - (candlestick['high'] * price_per_pixel)
                    low_price = price_levels['high'] - (candlestick['low'] * price_per_pixel)
                    
                    # Determine open/close based on color
                    if candlestick['color'] == 'green':  # Bullish
                        open_price = price_levels['high'] - (candlestick['body_bottom'] * price_per_pixel)
                        close_price = price_levels['high'] - (candlestick['body_top'] * price_per_pixel)
                    else:  # Bearish
                        open_price = price_levels['high'] - (candlestick['body_top'] * price_per_pixel)
                        close_price = price_levels['high'] - (candlestick['body_bottom'] * price_per_pixel)
                    
                    ohlc_point = OHLCData(
                        timestamp=time_points[i],
                        open=open_price,
                        high=high_price,
                        low=low_price,
                        close=close_price
                    )
                    
                    ohlc_data.append(ohlc_point)
            
            return ohlc_data
            
        except Exception as e:
            logger.error(f"OHLC data combination failed: {e}")
            return []
    
    async def _extract_standard_indicators(self, chart_image: np.ndarray, ohlc_data: List[OHLCData]) -> List[IndicatorData]:
        """Extract standard technical indicators from chart"""
        try:
            indicators = []
            
            # Extract moving averages
            sma_indicators = await self._extract_moving_averages(chart_image, ohlc_data, 'SMA')
            indicators.extend(sma_indicators)
            
            # Extract EMA
            ema_indicators = await self._extract_moving_averages(chart_image, ohlc_data, 'EMA')
            indicators.extend(ema_indicators)
            
            # Extract RSI if present
            rsi_indicator = await self._extract_rsi(chart_image)
            if rsi_indicator:
                indicators.append(rsi_indicator)
            
            # Extract MACD if present
            macd_indicator = await self._extract_macd(chart_image)
            if macd_indicator:
                indicators.append(macd_indicator)
            
            # Extract Bollinger Bands
            bb_indicators = await self._extract_bollinger_bands(chart_image, ohlc_data)
            indicators.extend(bb_indicators)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Standard indicator extraction failed: {e}")
            return []
    
    async def _extract_moving_averages(self, chart_image: np.ndarray, ohlc_data: List[OHLCData], ma_type: str) -> List[IndicatorData]:
        """Extract moving average lines from chart"""
        try:
            indicators = []
            
            # Detect colored lines that could be moving averages
            line_colors = ['blue', 'orange', 'red', 'green', 'purple']
            
            for color in line_colors:
                line_points = self._detect_colored_lines(chart_image, color)
                
                if line_points:
                    # Convert line points to price values
                    ma_values = self._convert_line_to_values(line_points, ohlc_data)
                    
                    if ma_values:
                        indicator = IndicatorData(
                            type=IndicatorType.SMA if ma_type == 'SMA' else IndicatorType.EMA,
                            name=f"{ma_type}_{color}",
                            values=ma_values,
                            timestamps=[point.timestamp for point in ohlc_data[:len(ma_values)]],
                            confidence=0.7,  # Placeholder confidence
                            metadata={'color': color, 'detection_method': 'line_detection'}
                        )
                        indicators.append(indicator)
            
            return indicators
            
        except Exception as e:
            logger.error(f"Moving average extraction failed: {e}")
            return []
    
    def _detect_colored_lines(self, image: np.ndarray, color: str) -> List[Tuple[int, int]]:
        """Detect continuous lines of specific colors"""
        try:
            # Define color ranges for line detection
            color_ranges = {
                'blue': ([100, 0, 0], [255, 100, 100]),
                'orange': ([0, 100, 200], [100, 200, 255]),
                'red': ([150, 0, 0], [255, 100, 100]),
                'green': ([0, 150, 0], [100, 255, 100]),
                'purple': ([100, 0, 100], [200, 100, 200])
            }
            
            if color not in color_ranges:
                return []
            
            lower, upper = color_ranges[color]
            mask = cv2.inRange(image, np.array(lower), np.array(upper))
            
            # Find line contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            line_points = []
            for contour in contours:
                # Approximate contour to get line points
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                for point in approx:
                    x, y = point[0]
                    line_points.append((x, y))
            
            # Sort points by x coordinate
            line_points.sort(key=lambda p: p[0])
            
            return line_points
            
        except Exception as e:
            logger.error(f"Colored line detection failed: {e}")
            return []
    
    def _convert_line_to_values(self, line_points: List[Tuple[int, int]], ohlc_data: List[OHLCData]) -> List[float]:
        """Convert line pixel coordinates to price values"""
        try:
            if not line_points or not ohlc_data:
                return []
            
            # Estimate price range from OHLC data
            all_prices = []
            for ohlc in ohlc_data:
                all_prices.extend([ohlc.open, ohlc.high, ohlc.low, ohlc.close])
            
            if not all_prices:
                return []
            
            price_high = max(all_prices)
            price_low = min(all_prices)
            
            # Assume chart height for conversion
            chart_height = 400  # This should be extracted from actual chart
            price_per_pixel = (price_high - price_low) / chart_height
            
            # Convert y coordinates to prices
            values = []
            for x, y in line_points:
                price = price_high - (y * price_per_pixel)
                values.append(price)
            
            return values
            
        except Exception as e:
            logger.error(f"Line to values conversion failed: {e}")
            return []
    
    async def _extract_rsi(self, chart_image: np.ndarray) -> Optional[IndicatorData]:
        """Extract RSI indicator if present in chart"""
        try:
            # Look for RSI panel (typically below main chart)
            # This is a simplified implementation
            
            # RSI would typically be in a separate panel
            # For now, return None as this requires more complex detection
            return None
            
        except Exception as e:
            logger.error(f"RSI extraction failed: {e}")
            return None
    
    async def _extract_macd(self, chart_image: np.ndarray) -> Optional[IndicatorData]:
        """Extract MACD indicator if present in chart"""
        try:
            # Look for MACD panel (typically below main chart)
            # This is a simplified implementation
            
            # MACD would typically be in a separate panel
            # For now, return None as this requires more complex detection
            return None
            
        except Exception as e:
            logger.error(f"MACD extraction failed: {e}")
            return None
    
    async def _extract_bollinger_bands(self, chart_image: np.ndarray, ohlc_data: List[OHLCData]) -> List[IndicatorData]:
        """Extract Bollinger Bands if present"""
        try:
            # Bollinger Bands typically appear as three lines around price
            # This would require sophisticated line detection and grouping
            
            # For now, return empty list
            return []
            
        except Exception as e:
            logger.error(f"Bollinger Bands extraction failed: {e}")
            return []
    
    async def _extract_custom_indicators(self, chart_image: np.ndarray, custom_indicators: List[str]) -> List[IndicatorData]:
        """Extract custom indicators specified by user"""
        try:
            custom_data = []
            
            for indicator_name in custom_indicators:
                # Use LLM to help identify custom indicators
                if self.openai_api_key and self.use_vision_api:
                    indicator_data = await self._llm_extract_custom_indicator(chart_image, indicator_name)
                    if indicator_data:
                        custom_data.append(indicator_data)
            
            return custom_data
            
        except Exception as e:
            logger.error(f"Custom indicator extraction failed: {e}")
            return []
    
    async def _llm_extract_custom_indicator(self, chart_image: np.ndarray, indicator_name: str) -> Optional[IndicatorData]:
        """Use LLM to extract custom indicator"""
        try:
            if not self.openai_api_key:
                return None
            
            # Convert image to base64 for API
            image_base64 = self._image_to_base64(chart_image)
            
            prompt = f"""
            Analyze this financial chart and extract data for the custom indicator: {indicator_name}
            
            Please identify:
            1. The location of the {indicator_name} indicator on the chart
            2. The values of the indicator at different time points
            3. Any parameters or settings visible for this indicator
            
            Return the data in JSON format with timestamps and values.
            """
            
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                        ]
                    }
                ],
                max_tokens=1000
            )
            
            # Parse LLM response and create IndicatorData
            llm_result = response.choices[0].message.content
            
            # This would need proper JSON parsing and validation
            # For now, return a placeholder
            return IndicatorData(
                type=IndicatorType.CUSTOM,
                name=indicator_name,
                values=[],  # Would be populated from LLM response
                timestamps=[],
                confidence=0.5,
                metadata={'extraction_method': 'llm', 'llm_response': llm_result}
            )
            
        except Exception as e:
            logger.error(f"LLM custom indicator extraction failed: {e}")
            return None
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        try:
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image, mode='L')
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Image to base64 conversion failed: {e}")
            return ""
    
    async def _detect_support_resistance(self, chart_image: np.ndarray, ohlc_data: List[OHLCData]) -> Tuple[List[float], List[float]]:
        """Detect support and resistance levels"""
        try:
            if not ohlc_data:
                return [], []
            
            # Extract price points
            highs = [ohlc.high for ohlc in ohlc_data]
            lows = [ohlc.low for ohlc in ohlc_data]
            
            # Find local maxima and minima
            resistance_levels = self._find_local_extrema(highs, 'max')
            support_levels = self._find_local_extrema(lows, 'min')
            
            return support_levels, resistance_levels
            
        except Exception as e:
            logger.error(f"Support/resistance detection failed: {e}")
            return [], []
    
    def _find_local_extrema(self, prices: List[float], extrema_type: str) -> List[float]:
        """Find local maxima or minima in price data"""
        try:
            if len(prices) < 3:
                return []
            
            extrema = []
            
            for i in range(1, len(prices) - 1):
                if extrema_type == 'max':
                    if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                        extrema.append(prices[i])
                else:  # min
                    if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                        extrema.append(prices[i])
            
            # Remove duplicates and sort
            extrema = list(set(extrema))
            extrema.sort()
            
            return extrema
            
        except Exception as e:
            logger.error(f"Local extrema detection failed: {e}")
            return []
    
    async def _extract_trend_lines(self, chart_image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract trend lines from chart"""
        try:
            # Use Hough line detection to find trend lines
            if len(chart_image.shape) == 3:
                gray = cv2.cvtColor(chart_image, cv2.COLOR_RGB2GRAY)
            else:
                gray = chart_image
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Hough line detection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=self.cv_config['line_detection_threshold'],
                                   minLineLength=50, maxLineGap=10)
            
            trend_lines = []
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate line properties
                    length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    angle = np.arctan2(y2-y1, x2-x1) * 180 / np.pi
                    
                    # Filter for trend lines (not too steep, reasonable length)
                    if length > 100 and abs(angle) < 45:
                        trend_lines.append({
                            'start': (x1, y1),
                            'end': (x2, y2),
                            'length': length,
                            'angle': angle,
                            'slope': (y2-y1)/(x2-x1) if x2 != x1 else float('inf')
                        })
            
            return trend_lines
            
        except Exception as e:
            logger.error(f"Trend line extraction failed: {e}")
            return []
    
    async def _determine_timeframe(self, chart_image: np.ndarray, ohlc_data: List[OHLCData]) -> Tuple[str, Tuple[datetime, datetime]]:
        """Determine chart timeframe and period"""
        try:
            if not ohlc_data:
                return "1D", (datetime.now() - timedelta(days=30), datetime.now())
            
            # Calculate time differences between data points
            if len(ohlc_data) > 1:
                time_diffs = []
                for i in range(1, len(ohlc_data)):
                    diff = ohlc_data[i].timestamp - ohlc_data[i-1].timestamp
                    time_diffs.append(diff.total_seconds())
                
                avg_diff = sum(time_diffs) / len(time_diffs)
                
                # Determine timeframe based on average difference
                if avg_diff < 300:  # 5 minutes
                    timeframe = "1m"
                elif avg_diff < 900:  # 15 minutes
                    timeframe = "5m"
                elif avg_diff < 3600:  # 1 hour
                    timeframe = "15m"
                elif avg_diff < 14400:  # 4 hours
                    timeframe = "1h"
                elif avg_diff < 86400:  # 1 day
                    timeframe = "4h"
                else:
                    timeframe = "1D"
            else:
                timeframe = "1D"
            
            # Determine chart period
            start_time = ohlc_data[0].timestamp if ohlc_data else datetime.now() - timedelta(days=30)
            end_time = ohlc_data[-1].timestamp if ohlc_data else datetime.now()
            
            return timeframe, (start_time, end_time)
            
        except Exception as e:
            logger.error(f"Timeframe determination failed: {e}")
            return "1D", (datetime.now() - timedelta(days=30), datetime.now())
    
    async def _extract_volume_data(self, chart_image: np.ndarray) -> Optional[List[float]]:
        """Extract volume data if present in chart"""
        try:
            # Volume is typically shown as bars at the bottom of the chart
            # This would require detecting the volume panel and extracting bar heights
            
            # For now, return None as this requires complex panel detection
            return None
            
        except Exception as e:
            logger.error(f"Volume data extraction failed: {e}")
            return None
    
    def _calculate_price_range(self, ohlc_data: List[OHLCData]) -> Tuple[float, float]:
        """Calculate price range from OHLC data"""
        try:
            if not ohlc_data:
                return (0.0, 100.0)
            
            all_prices = []
            for ohlc in ohlc_data:
                all_prices.extend([ohlc.open, ohlc.high, ohlc.low, ohlc.close])
            
            return (min(all_prices), max(all_prices))
            
        except Exception as e:
            logger.error(f"Price range calculation failed: {e}")
            return (0.0, 100.0)
    
    async def _llm_analysis(self, chart_image: np.ndarray, ohlc_data: List[OHLCData]) -> Dict[str, Any]:
        """Use LLM for additional chart analysis"""
        try:
            if not self.openai_api_key or not self.use_vision_api:
                return {}
            
            # Convert image to base64
            image_base64 = self._image_to_base64(chart_image)
            
            prompt = """
            Analyze this financial chart and provide:
            1. The likely symbol/ticker being displayed
            2. Overall trend direction
            3. Key patterns or formations visible
            4. Any text or labels visible on the chart
            5. Chart quality assessment
            
            Return the analysis in JSON format.
            """
            
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                        ]
                    }
                ],
                max_tokens=500
            )
            
            llm_result = response.choices[0].message.content
            
            # Parse JSON response (with error handling)
            try:
                analysis = json.loads(llm_result)
            except json.JSONDecodeError:
                analysis = {'raw_response': llm_result}
            
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {}
    
    def _calculate_extraction_confidence(self, ohlc_data: List[OHLCData], indicators: List[IndicatorData], 
                                       support_levels: List[float], resistance_levels: List[float]) -> float:
        """Calculate overall extraction confidence score"""
        try:
            confidence_factors = []
            
            # OHLC data confidence
            if ohlc_data:
                ohlc_confidence = min(len(ohlc_data) / 50, 1.0)  # More data points = higher confidence
                confidence_factors.append(ohlc_confidence * 0.4)
            
            # Indicators confidence
            if indicators:
                indicator_confidence = min(len(indicators) / 5, 1.0)
                confidence_factors.append(indicator_confidence * 0.3)
            
            # Support/resistance confidence
            sr_count = len(support_levels) + len(resistance_levels)
            if sr_count > 0:
                sr_confidence = min(sr_count / 10, 1.0)
                confidence_factors.append(sr_confidence * 0.2)
            
            # Base confidence
            confidence_factors.append(0.1)  # Minimum base confidence
            
            return sum(confidence_factors)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _create_empty_features(self) -> ChartFeatures:
        """Create empty features object for error cases"""
        return ChartFeatures(
            ohlc_data=[],
            indicators=[],
            timeframe="1D",
            symbol=None,
            chart_period=(datetime.now() - timedelta(days=30), datetime.now()),
            price_range=(0.0, 100.0),
            volume_data=None,
            support_levels=[],
            resistance_levels=[],
            trend_lines=[],
            custom_indicators=[],
            extraction_confidence=0.0,
            metadata={'error': 'Feature extraction failed'}
        )