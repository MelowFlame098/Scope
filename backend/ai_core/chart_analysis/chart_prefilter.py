# Chart Prefilter Component
# Validates and preprocesses financial chart images

import cv2
import numpy as np
import base64
from PIL import Image, ImageEnhance
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import io
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class ChartType(Enum):
    """Supported chart types"""
    CANDLESTICK = "candlestick"
    LINE = "line"
    BAR = "bar"
    AREA = "area"
    VOLUME = "volume"
    UNKNOWN = "unknown"

class ValidationStatus(Enum):
    """Chart validation status"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    NEEDS_PREPROCESSING = "needs_preprocessing"

@dataclass
class ChartValidationResult:
    """Result of chart validation process"""
    is_valid: bool
    status: ValidationStatus
    chart_type: ChartType
    confidence: float
    errors: List[str]
    warnings: List[str]
    processed_chart: Optional[np.ndarray]
    metadata: Dict[str, any]
    preprocessing_applied: List[str]

class ChartPrefilter:
    """
    Chart Prefilter Component - First stage of the chart analysis pipeline.
    
    Responsibilities:
    - Validate chart image format and quality
    - Detect chart type (candlestick, line, bar, etc.)
    - Apply preprocessing to enhance chart readability
    - Filter out invalid or low-quality charts
    - Extract basic metadata
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Configuration parameters
        self.min_width = self.config.get('min_width', 400)
        self.min_height = self.config.get('min_height', 300)
        self.max_width = self.config.get('max_width', 4000)
        self.max_height = self.config.get('max_height', 3000)
        self.min_confidence = self.config.get('min_confidence', 0.7)
        self.supported_formats = self.config.get('supported_formats', ['PNG', 'JPEG', 'JPG', 'WEBP'])
        
        # Chart detection parameters
        self.chart_patterns = {
            ChartType.CANDLESTICK: {
                'color_variance_threshold': 0.3,
                'vertical_line_density': 0.1,
                'red_green_ratio_min': 0.2
            },
            ChartType.LINE: {
                'continuous_line_threshold': 0.8,
                'line_smoothness': 0.7
            },
            ChartType.BAR: {
                'vertical_bar_density': 0.15,
                'uniform_width_threshold': 0.8
            }
        }
        
        logger.info("ChartPrefilter initialized with config: %s", self.config)
    
    async def validate_chart(self, chart_input: Union[str, bytes, np.ndarray, Image.Image]) -> ChartValidationResult:
        """
        Main validation method for chart inputs.
        
        Args:
            chart_input: Chart image in various formats (base64, bytes, numpy array, PIL Image)
            
        Returns:
            ChartValidationResult with validation status and processed chart
        """
        try:
            # Convert input to numpy array
            chart_array = await self._convert_to_array(chart_input)
            if chart_array is None:
                return self._create_invalid_result("Failed to convert input to image array")
            
            # Basic format validation
            format_result = self._validate_format(chart_array)
            if not format_result['valid']:
                return self._create_invalid_result(format_result['errors'])
            
            # Dimension validation
            dimension_result = self._validate_dimensions(chart_array)
            if not dimension_result['valid']:
                return self._create_invalid_result(dimension_result['errors'])
            
            # Quality assessment
            quality_result = await self._assess_quality(chart_array)
            
            # Chart type detection
            chart_type, confidence = await self._detect_chart_type(chart_array)
            
            # Apply preprocessing if needed
            processed_chart, preprocessing_steps = await self._preprocess_chart(chart_array, quality_result)
            
            # Extract metadata
            metadata = await self._extract_metadata(processed_chart)
            
            # Determine final validation status
            is_valid = (
                confidence >= self.min_confidence and
                quality_result['score'] >= 0.6 and
                chart_type != ChartType.UNKNOWN
            )
            
            status = ValidationStatus.VALID if is_valid else ValidationStatus.INVALID
            if quality_result['score'] < 0.8:
                status = ValidationStatus.WARNING
            
            return ChartValidationResult(
                is_valid=is_valid,
                status=status,
                chart_type=chart_type,
                confidence=confidence,
                errors=format_result.get('errors', []) + dimension_result.get('errors', []),
                warnings=quality_result.get('warnings', []),
                processed_chart=processed_chart,
                metadata=metadata,
                preprocessing_applied=preprocessing_steps
            )
            
        except Exception as e:
            logger.error(f"Chart validation failed: {e}")
            return self._create_invalid_result(f"Validation error: {str(e)}")
    
    async def _convert_to_array(self, chart_input: Union[str, bytes, np.ndarray, Image.Image]) -> Optional[np.ndarray]:
        """Convert various input formats to numpy array"""
        try:
            if isinstance(chart_input, np.ndarray):
                return chart_input
            
            elif isinstance(chart_input, Image.Image):
                return np.array(chart_input)
            
            elif isinstance(chart_input, str):
                # Assume base64 encoded image
                if chart_input.startswith('data:image'):
                    # Remove data URL prefix
                    chart_input = chart_input.split(',')[1]
                
                image_data = base64.b64decode(chart_input)
                image = Image.open(io.BytesIO(image_data))
                return np.array(image)
            
            elif isinstance(chart_input, bytes):
                image = Image.open(io.BytesIO(chart_input))
                return np.array(image)
            
            else:
                logger.error(f"Unsupported input type: {type(chart_input)}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to convert input to array: {e}")
            return None
    
    def _validate_format(self, chart_array: np.ndarray) -> Dict[str, any]:
        """Validate basic image format requirements"""
        errors = []
        
        # Check if array is valid
        if chart_array is None or chart_array.size == 0:
            errors.append("Empty or invalid image array")
        
        # Check dimensions
        if len(chart_array.shape) not in [2, 3]:
            errors.append(f"Invalid image dimensions: {chart_array.shape}")
        
        # Check color channels
        if len(chart_array.shape) == 3 and chart_array.shape[2] not in [1, 3, 4]:
            errors.append(f"Invalid number of color channels: {chart_array.shape[2]}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _validate_dimensions(self, chart_array: np.ndarray) -> Dict[str, any]:
        """Validate image dimensions"""
        errors = []
        height, width = chart_array.shape[:2]
        
        if width < self.min_width:
            errors.append(f"Image width {width} below minimum {self.min_width}")
        
        if height < self.min_height:
            errors.append(f"Image height {height} below minimum {self.min_height}")
        
        if width > self.max_width:
            errors.append(f"Image width {width} exceeds maximum {self.max_width}")
        
        if height > self.max_height:
            errors.append(f"Image height {height} exceeds maximum {self.max_height}")
        
        # Check aspect ratio
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 4.0:
            errors.append(f"Unusual aspect ratio: {aspect_ratio:.2f}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    async def _assess_quality(self, chart_array: np.ndarray) -> Dict[str, any]:
        """Assess image quality metrics"""
        try:
            # Convert to grayscale for analysis
            if len(chart_array.shape) == 3:
                gray = cv2.cvtColor(chart_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = chart_array
            
            # Calculate quality metrics
            metrics = {}
            
            # Sharpness (Laplacian variance)
            metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Contrast (standard deviation)
            metrics['contrast'] = np.std(gray)
            
            # Brightness (mean intensity)
            metrics['brightness'] = np.mean(gray)
            
            # Noise level (high frequency content)
            metrics['noise'] = self._estimate_noise(gray)
            
            # Overall quality score
            quality_score = self._calculate_quality_score(metrics)
            
            warnings = []
            if quality_score < 0.7:
                warnings.append("Low image quality detected")
            if metrics['sharpness'] < 100:
                warnings.append("Image appears blurry")
            if metrics['contrast'] < 30:
                warnings.append("Low contrast image")
            
            return {
                'score': quality_score,
                'metrics': metrics,
                'warnings': warnings
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {'score': 0.5, 'metrics': {}, 'warnings': ['Quality assessment failed']}
    
    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in the image"""
        try:
            # Use high-pass filter to detect noise
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(gray_image, -1, kernel)
            return np.std(filtered)
        except:
            return 0.0
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from metrics"""
        try:
            # Normalize metrics to 0-1 range
            sharpness_score = min(metrics.get('sharpness', 0) / 500, 1.0)
            contrast_score = min(metrics.get('contrast', 0) / 100, 1.0)
            brightness_score = 1.0 - abs(metrics.get('brightness', 128) - 128) / 128
            noise_score = max(0, 1.0 - metrics.get('noise', 0) / 50)
            
            # Weighted average
            weights = {'sharpness': 0.3, 'contrast': 0.3, 'brightness': 0.2, 'noise': 0.2}
            
            quality_score = (
                sharpness_score * weights['sharpness'] +
                contrast_score * weights['contrast'] +
                brightness_score * weights['brightness'] +
                noise_score * weights['noise']
            )
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Quality score calculation failed: {e}")
            return 0.5
    
    async def _detect_chart_type(self, chart_array: np.ndarray) -> Tuple[ChartType, float]:
        """Detect the type of financial chart"""
        try:
            # Convert to RGB if needed
            if len(chart_array.shape) == 3 and chart_array.shape[2] == 4:
                chart_rgb = cv2.cvtColor(chart_array, cv2.COLOR_RGBA2RGB)
            elif len(chart_array.shape) == 3:
                chart_rgb = chart_array
            else:
                chart_rgb = cv2.cvtColor(chart_array, cv2.COLOR_GRAY2RGB)
            
            # Detect candlestick patterns
            candlestick_confidence = self._detect_candlestick_pattern(chart_rgb)
            
            # Detect line chart patterns
            line_confidence = self._detect_line_pattern(chart_rgb)
            
            # Detect bar chart patterns
            bar_confidence = self._detect_bar_pattern(chart_rgb)
            
            # Determine best match
            confidences = {
                ChartType.CANDLESTICK: candlestick_confidence,
                ChartType.LINE: line_confidence,
                ChartType.BAR: bar_confidence
            }
            
            best_type = max(confidences, key=confidences.get)
            best_confidence = confidences[best_type]
            
            if best_confidence < 0.3:
                return ChartType.UNKNOWN, best_confidence
            
            return best_type, best_confidence
            
        except Exception as e:
            logger.error(f"Chart type detection failed: {e}")
            return ChartType.UNKNOWN, 0.0
    
    def _detect_candlestick_pattern(self, chart_rgb: np.ndarray) -> float:
        """Detect candlestick chart patterns"""
        try:
            # Look for red and green colors typical in candlestick charts
            red_mask = self._detect_color_range(chart_rgb, [150, 0, 0], [255, 100, 100])
            green_mask = self._detect_color_range(chart_rgb, [0, 150, 0], [100, 255, 100])
            
            red_ratio = np.sum(red_mask) / chart_rgb.size
            green_ratio = np.sum(green_mask) / chart_rgb.size
            
            # Look for vertical line patterns (wicks)
            gray = cv2.cvtColor(chart_rgb, cv2.COLOR_RGB2GRAY)
            vertical_lines = self._detect_vertical_lines(gray)
            
            # Calculate confidence based on color ratios and line patterns
            color_confidence = min((red_ratio + green_ratio) * 10, 1.0)
            line_confidence = min(vertical_lines / 100, 1.0)
            
            return (color_confidence * 0.6 + line_confidence * 0.4)
            
        except Exception as e:
            logger.error(f"Candlestick detection failed: {e}")
            return 0.0
    
    def _detect_line_pattern(self, chart_rgb: np.ndarray) -> float:
        """Detect line chart patterns"""
        try:
            gray = cv2.cvtColor(chart_rgb, cv2.COLOR_RGB2GRAY)
            
            # Detect continuous lines using edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Look for horizontal continuity in edges
            horizontal_lines = 0
            for row in edges:
                continuous_segments = self._find_continuous_segments(row)
                horizontal_lines += len([seg for seg in continuous_segments if seg > 20])
            
            # Calculate confidence based on line continuity
            line_density = horizontal_lines / edges.shape[0]
            confidence = min(line_density * 2, 1.0)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Line pattern detection failed: {e}")
            return 0.0
    
    def _detect_bar_pattern(self, chart_rgb: np.ndarray) -> float:
        """Detect bar chart patterns"""
        try:
            gray = cv2.cvtColor(chart_rgb, cv2.COLOR_RGB2GRAY)
            
            # Detect vertical rectangular patterns
            vertical_bars = self._detect_vertical_rectangles(gray)
            
            # Calculate confidence based on bar patterns
            confidence = min(vertical_bars / 50, 1.0)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Bar pattern detection failed: {e}")
            return 0.0
    
    def _detect_color_range(self, image: np.ndarray, lower: List[int], upper: List[int]) -> np.ndarray:
        """Detect pixels within a color range"""
        lower_bound = np.array(lower)
        upper_bound = np.array(upper)
        mask = cv2.inRange(image, lower_bound, upper_bound)
        return mask
    
    def _detect_vertical_lines(self, gray_image: np.ndarray) -> int:
        """Count vertical line patterns"""
        try:
            # Use morphological operations to detect vertical lines
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            vertical = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)
            return np.sum(vertical > 0)
        except:
            return 0
    
    def _find_continuous_segments(self, row: np.ndarray) -> List[int]:
        """Find continuous segments in a row"""
        segments = []
        current_length = 0
        
        for pixel in row:
            if pixel > 0:
                current_length += 1
            else:
                if current_length > 0:
                    segments.append(current_length)
                current_length = 0
        
        if current_length > 0:
            segments.append(current_length)
        
        return segments
    
    def _detect_vertical_rectangles(self, gray_image: np.ndarray) -> int:
        """Detect vertical rectangular patterns"""
        try:
            # Find contours
            contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            vertical_rects = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if h > w * 2 and w > 5 and h > 10:  # Vertical rectangle criteria
                    vertical_rects += 1
            
            return vertical_rects
        except:
            return 0
    
    async def _preprocess_chart(self, chart_array: np.ndarray, quality_result: Dict) -> Tuple[np.ndarray, List[str]]:
        """Apply preprocessing to enhance chart quality"""
        processed = chart_array.copy()
        steps_applied = []
        
        try:
            # Enhance contrast if needed
            if quality_result.get('metrics', {}).get('contrast', 100) < 30:
                processed = self._enhance_contrast(processed)
                steps_applied.append('contrast_enhancement')
            
            # Sharpen if blurry
            if quality_result.get('metrics', {}).get('sharpness', 200) < 100:
                processed = self._sharpen_image(processed)
                steps_applied.append('sharpening')
            
            # Denoise if noisy
            if quality_result.get('metrics', {}).get('noise', 0) > 30:
                processed = self._denoise_image(processed)
                steps_applied.append('denoising')
            
            # Normalize brightness
            brightness = quality_result.get('metrics', {}).get('brightness', 128)
            if brightness < 80 or brightness > 180:
                processed = self._normalize_brightness(processed)
                steps_applied.append('brightness_normalization')
            
            return processed, steps_applied
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return chart_array, []
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        try:
            if len(image.shape) == 3:
                # Convert to PIL for enhancement
                pil_image = Image.fromarray(image)
                enhancer = ImageEnhance.Contrast(pil_image)
                enhanced = enhancer.enhance(1.5)
                return np.array(enhanced)
            else:
                # Use CLAHE for grayscale
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
        except:
            return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter"""
        try:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            if len(image.shape) == 3:
                sharpened = cv2.filter2D(image, -1, kernel)
            else:
                sharpened = cv2.filter2D(image, -1, kernel)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        except:
            return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising filter"""
        try:
            if len(image.shape) == 3:
                return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            else:
                return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
        except:
            return image
    
    def _normalize_brightness(self, image: np.ndarray) -> np.ndarray:
        """Normalize image brightness"""
        try:
            if len(image.shape) == 3:
                # Convert to HSV and adjust V channel
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            else:
                return cv2.equalizeHist(image)
        except:
            return image
    
    async def _extract_metadata(self, chart_array: np.ndarray) -> Dict[str, any]:
        """Extract metadata from the processed chart"""
        try:
            height, width = chart_array.shape[:2]
            
            metadata = {
                'dimensions': {'width': width, 'height': height},
                'channels': chart_array.shape[2] if len(chart_array.shape) == 3 else 1,
                'size_bytes': chart_array.nbytes,
                'dtype': str(chart_array.dtype),
                'timestamp': datetime.now().isoformat(),
                'aspect_ratio': width / height
            }
            
            # Add color analysis
            if len(chart_array.shape) == 3:
                metadata['color_analysis'] = {
                    'mean_rgb': np.mean(chart_array, axis=(0, 1)).tolist(),
                    'std_rgb': np.std(chart_array, axis=(0, 1)).tolist()
                }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {'error': str(e)}
    
    def _create_invalid_result(self, errors: Union[str, List[str]]) -> ChartValidationResult:
        """Create an invalid validation result"""
        if isinstance(errors, str):
            errors = [errors]
        
        return ChartValidationResult(
            is_valid=False,
            status=ValidationStatus.INVALID,
            chart_type=ChartType.UNKNOWN,
            confidence=0.0,
            errors=errors,
            warnings=[],
            processed_chart=None,
            metadata={},
            preprocessing_applied=[]
        )