import asyncio
import base64
import io
from typing import Optional, Tuple, Dict, Any
from PIL import Image, ImageGrab
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ScreenCaptureService:
    """Service for capturing and processing screen content for chart analysis."""
    
    def __init__(self):
        self.capture_history = []
        self.max_history = 10
    
    async def capture_screen_region(self, 
                                  x: int, 
                                  y: int, 
                                  width: int, 
                                  height: int) -> Optional[str]:
        """Capture a specific region of the screen and return as base64 encoded image.
        
        Args:
            x: X coordinate of top-left corner
            y: Y coordinate of top-left corner
            width: Width of capture region
            height: Height of capture region
            
        Returns:
            Base64 encoded image string or None if capture fails
        """
        try:
            # Capture the specified screen region
            bbox = (x, y, x + width, y + height)
            screenshot = ImageGrab.grab(bbox=bbox)
            
            # Convert to RGB if necessary
            if screenshot.mode != 'RGB':
                screenshot = screenshot.convert('RGB')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Store in history
            capture_data = {
                'timestamp': datetime.now().isoformat(),
                'region': {'x': x, 'y': y, 'width': width, 'height': height},
                'image_base64': image_base64
            }
            
            self._add_to_history(capture_data)
            
            logger.info(f"Screen capture successful: {width}x{height} at ({x}, {y})")
            return image_base64
            
        except Exception as e:
            logger.error(f"Screen capture failed: {str(e)}")
            return None
    
    async def capture_full_screen(self) -> Optional[str]:
        """Capture the entire screen.
        
        Returns:
            Base64 encoded image string or None if capture fails
        """
        try:
            screenshot = ImageGrab.grab()
            
            # Convert to RGB if necessary
            if screenshot.mode != 'RGB':
                screenshot = screenshot.convert('RGB')
            
            # Save to bytes buffer
            buffer = io.BytesIO()
            screenshot.save(buffer, format='PNG')
            buffer.seek(0)
            
            # Encode to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Store in history
            capture_data = {
                'timestamp': datetime.now().isoformat(),
                'region': 'full_screen',
                'image_base64': image_base64
            }
            
            self._add_to_history(capture_data)
            
            logger.info("Full screen capture successful")
            return image_base64
            
        except Exception as e:
            logger.error(f"Full screen capture failed: {str(e)}")
            return None
    
    async def detect_chart_region(self, image_base64: str) -> Optional[Dict[str, int]]:
        """Detect chart region within a captured image using basic image processing.
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Dictionary with chart region coordinates or None if not found
        """
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to numpy array for processing
            img_array = np.array(image)
            
            # Simple chart detection logic (can be enhanced with ML)
            # Look for rectangular regions with grid patterns or candlestick patterns
            height, width = img_array.shape[:2]
            
            # For now, return center region as potential chart area
            # This should be replaced with actual chart detection algorithm
            chart_region = {
                'x': width // 4,
                'y': height // 4,
                'width': width // 2,
                'height': height // 2
            }
            
            logger.info(f"Chart region detected: {chart_region}")
            return chart_region
            
        except Exception as e:
            logger.error(f"Chart detection failed: {str(e)}")
            return None
    
    async def capture_and_analyze_chart(self, 
                                      region: Optional[Dict[str, int]] = None) -> Optional[Dict[str, Any]]:
        """Capture screen region and prepare for chart analysis.
        
        Args:
            region: Optional region specification, if None captures full screen
            
        Returns:
            Dictionary with capture data and metadata
        """
        try:
            if region:
                image_base64 = await self.capture_screen_region(
                    region['x'], region['y'], region['width'], region['height']
                )
            else:
                image_base64 = await self.capture_full_screen()
            
            if not image_base64:
                return None
            
            # Detect chart region if full screen was captured
            chart_region = None
            if not region:
                chart_region = await self.detect_chart_region(image_base64)
            
            result = {
                'image_base64': image_base64,
                'timestamp': datetime.now().isoformat(),
                'capture_region': region or 'full_screen',
                'detected_chart_region': chart_region,
                'ready_for_analysis': True
            }
            
            logger.info("Screen capture and chart preparation completed")
            return result
            
        except Exception as e:
            logger.error(f"Capture and analysis preparation failed: {str(e)}")
            return None
    
    def _add_to_history(self, capture_data: Dict[str, Any]):
        """Add capture data to history, maintaining max history size."""
        self.capture_history.append(capture_data)
        if len(self.capture_history) > self.max_history:
            self.capture_history.pop(0)
    
    def get_capture_history(self) -> list:
        """Get the capture history."""
        return self.capture_history
    
    def clear_history(self):
        """Clear the capture history."""
        self.capture_history.clear()
        logger.info("Capture history cleared")

# Global instance
screen_capture_service = ScreenCaptureService()