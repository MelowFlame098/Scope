from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime

from ..services.screen_capture import screen_capture_service
from ..services.market_scraper import market_scraper_service
from ..ai_core.chart_analysis import ChartAnalysisPipeline

logger = logging.getLogger(__name__)

# Initialize the chart analysis pipeline
chart_pipeline = ChartAnalysisPipeline()

router = APIRouter(prefix="/api/screen-analysis", tags=["screen-analysis"])

# Request/Response Models
class ScreenCaptureRequest(BaseModel):
    x: int = Field(..., description="X coordinate of capture region")
    y: int = Field(..., description="Y coordinate of capture region")
    width: int = Field(..., description="Width of capture region")
    height: int = Field(..., description="Height of capture region")
    symbol: Optional[str] = Field(None, description="Stock symbol for analysis")
    timeframe: Optional[str] = Field("1D", description="Chart timeframe")
    custom_indicators: Optional[List[str]] = Field(default_factory=list)
    enable_forecasting: bool = Field(True, description="Enable price forecasting")
    enable_execution: bool = Field(True, description="Enable execution signals")

class FullScreenCaptureRequest(BaseModel):
    symbol: Optional[str] = Field(None, description="Stock symbol for analysis")
    timeframe: Optional[str] = Field("1D", description="Chart timeframe")
    custom_indicators: Optional[List[str]] = Field(default_factory=list)
    enable_forecasting: bool = Field(True, description="Enable price forecasting")
    enable_execution: bool = Field(True, description="Enable execution signals")
    auto_detect_chart: bool = Field(True, description="Automatically detect chart region")

class MarketAnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Stock symbol to analyze")
    include_sentiment: bool = Field(True, description="Include sentiment analysis")
    include_signals: bool = Field(True, description="Include trading signals")
    include_fundamentals: bool = Field(True, description="Include fundamental data")

class EnhancedAnalysisRequest(BaseModel):
    capture_region: Optional[Dict[str, int]] = Field(None, description="Screen capture region")
    symbol: str = Field(..., description="Stock symbol for analysis")
    timeframe: Optional[str] = Field("1D", description="Chart timeframe")
    custom_indicators: Optional[List[str]] = Field(default_factory=list)
    enable_forecasting: bool = Field(True, description="Enable price forecasting")
    enable_execution: bool = Field(True, description="Enable execution signals")
    include_market_data: bool = Field(True, description="Include scraped market data")

class ScreenCaptureResponse(BaseModel):
    success: bool
    image_base64: Optional[str] = None
    timestamp: str
    capture_region: Dict[str, Any]
    detected_chart_region: Optional[Dict[str, int]] = None
    error: Optional[str] = None

class MarketAnalysisResponse(BaseModel):
    success: bool
    symbol: str
    timestamp: str
    sentiment_analysis: Optional[Dict[str, Any]] = None
    fundamental_data: Optional[Dict[str, Any]] = None
    trading_signals: Optional[List[Dict[str, Any]]] = None
    overall_recommendation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class EnhancedAnalysisResponse(BaseModel):
    success: bool
    timestamp: str
    chart_analysis: Optional[Dict[str, Any]] = None
    market_analysis: Optional[Dict[str, Any]] = None
    combined_recommendation: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

@router.post("/capture-region", response_model=ScreenCaptureResponse)
async def capture_screen_region(request: ScreenCaptureRequest):
    """Capture a specific region of the screen."""
    try:
        logger.info(f"Capturing screen region: {request.x}, {request.y}, {request.width}x{request.height}")
        
        result = await screen_capture_service.capture_and_analyze_chart({
            'x': request.x,
            'y': request.y,
            'width': request.width,
            'height': request.height
        })
        
        if not result:
            raise HTTPException(status_code=500, detail="Screen capture failed")
        
        return ScreenCaptureResponse(
            success=True,
            image_base64=result['image_base64'],
            timestamp=result['timestamp'],
            capture_region=result['capture_region'],
            detected_chart_region=result.get('detected_chart_region')
        )
        
    except Exception as e:
        logger.error(f"Screen capture failed: {str(e)}")
        return ScreenCaptureResponse(
            success=False,
            timestamp=datetime.now().isoformat(),
            capture_region={},
            error=str(e)
        )

@router.post("/capture-fullscreen", response_model=ScreenCaptureResponse)
async def capture_full_screen(request: FullScreenCaptureRequest):
    """Capture the entire screen and optionally detect chart region."""
    try:
        logger.info("Capturing full screen")
        
        result = await screen_capture_service.capture_and_analyze_chart()
        
        if not result:
            raise HTTPException(status_code=500, detail="Full screen capture failed")
        
        return ScreenCaptureResponse(
            success=True,
            image_base64=result['image_base64'],
            timestamp=result['timestamp'],
            capture_region=result['capture_region'],
            detected_chart_region=result.get('detected_chart_region')
        )
        
    except Exception as e:
        logger.error(f"Full screen capture failed: {str(e)}")
        return ScreenCaptureResponse(
            success=False,
            timestamp=datetime.now().isoformat(),
            capture_region={},
            error=str(e)
        )

@router.post("/analyze-market", response_model=MarketAnalysisResponse)
async def analyze_market_data(request: MarketAnalysisRequest):
    """Analyze market data for a given symbol using web scraping."""
    try:
        logger.info(f"Analyzing market data for {request.symbol}")
        
        async with market_scraper_service as scraper:
            analysis = await scraper.get_comprehensive_analysis(request.symbol)
        
        if not analysis:
            raise HTTPException(status_code=500, detail="Market analysis failed")
        
        return MarketAnalysisResponse(
            success=True,
            symbol=request.symbol,
            timestamp=analysis['timestamp'],
            sentiment_analysis=analysis.get('sentiment_analysis') if request.include_sentiment else None,
            fundamental_data=analysis.get('fundamental_data') if request.include_fundamentals else None,
            trading_signals=analysis.get('trading_signals') if request.include_signals else None,
            overall_recommendation=analysis.get('overall_recommendation')
        )
        
    except Exception as e:
        logger.error(f"Market analysis failed for {request.symbol}: {str(e)}")
        return MarketAnalysisResponse(
            success=False,
            symbol=request.symbol,
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@router.post("/enhanced-analysis", response_model=EnhancedAnalysisResponse)
async def enhanced_chart_analysis(request: EnhancedAnalysisRequest):
    """Perform enhanced analysis combining screen capture, chart analysis, and market data."""
    try:
        logger.info(f"Starting enhanced analysis for {request.symbol}")
        
        # Step 1: Capture screen
        capture_result = await screen_capture_service.capture_and_analyze_chart(request.capture_region)
        
        if not capture_result:
            raise HTTPException(status_code=500, detail="Screen capture failed")
        
        # Step 2: Analyze chart using AI pipeline
        chart_analysis = None
        if capture_result['image_base64']:
            try:
                chart_analysis = await chart_pipeline.analyze_chart(
                    image_base64=capture_result['image_base64'],
                    symbol=request.symbol,
                    timeframe=request.timeframe,
                    custom_indicators=request.custom_indicators,
                    enable_forecasting=request.enable_forecasting,
                    enable_execution=request.enable_execution
                )
            except Exception as e:
                logger.warning(f"Chart analysis failed: {str(e)}")
        
        # Step 3: Gather market data
        market_analysis = None
        if request.include_market_data:
            try:
                async with market_scraper_service as scraper:
                    market_analysis = await scraper.get_comprehensive_analysis(request.symbol)
            except Exception as e:
                logger.warning(f"Market analysis failed: {str(e)}")
        
        # Step 4: Combine recommendations
        combined_recommendation = _combine_recommendations(chart_analysis, market_analysis)
        
        return EnhancedAnalysisResponse(
            success=True,
            timestamp=datetime.now().isoformat(),
            chart_analysis=chart_analysis,
            market_analysis=market_analysis,
            combined_recommendation=combined_recommendation
        )
        
    except Exception as e:
        logger.error(f"Enhanced analysis failed for {request.symbol}: {str(e)}")
        return EnhancedAnalysisResponse(
            success=False,
            timestamp=datetime.now().isoformat(),
            error=str(e)
        )

@router.get("/capture-history")
async def get_capture_history():
    """Get the screen capture history."""
    try:
        history = screen_capture_service.get_capture_history()
        return {
            "success": True,
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Failed to get capture history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/capture-history")
async def clear_capture_history():
    """Clear the screen capture history."""
    try:
        screen_capture_service.clear_history()
        return {"success": True, "message": "Capture history cleared"}
    except Exception as e:
        logger.error(f"Failed to clear capture history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def _combine_recommendations(chart_analysis: Optional[Dict], market_analysis: Optional[Dict]) -> Dict[str, Any]:
    """Combine recommendations from chart analysis and market data."""
    try:
        combined = {
            'recommendation': 'HOLD',
            'confidence': 0.0,
            'reasoning': [],
            'chart_score': 0.0,
            'market_score': 0.0,
            'final_score': 0.0
        }
        
        chart_score = 0.0
        market_score = 0.0
        
        # Extract chart analysis score
        if chart_analysis and chart_analysis.get('success'):
            execution_signals = chart_analysis.get('execution_signals', {})
            if execution_signals.get('primary_signal'):
                signal = execution_signals['primary_signal']
                if signal['action'] == 'BUY':
                    chart_score = signal.get('confidence', 0.5)
                elif signal['action'] == 'SELL':
                    chart_score = -signal.get('confidence', 0.5)
                
                combined['reasoning'].append(f"Chart analysis suggests {signal['action']} with {signal.get('confidence', 0)*100:.1f}% confidence")
        
        # Extract market analysis score
        if market_analysis and market_analysis.get('overall_recommendation'):
            rec = market_analysis['overall_recommendation']
            market_score = rec.get('overall_score', 0.0)
            
            combined['reasoning'].append(f"Market sentiment analysis suggests {rec.get('recommendation', 'HOLD')} with score {market_score:.2f}")
        
        # Combine scores (weighted average)
        chart_weight = 0.6  # Give more weight to chart analysis
        market_weight = 0.4
        
        final_score = (chart_score * chart_weight) + (market_score * market_weight)
        
        # Determine final recommendation
        if final_score > 0.3:
            recommendation = 'BUY'
        elif final_score < -0.3:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'
        
        combined.update({
            'recommendation': recommendation,
            'confidence': abs(final_score),
            'chart_score': chart_score,
            'market_score': market_score,
            'final_score': final_score
        })
        
        return combined
        
    except Exception as e:
        logger.error(f"Failed to combine recommendations: {str(e)}")
        return {
            'recommendation': 'HOLD',
            'confidence': 0.0,
            'reasoning': ['Error combining analysis results'],
            'final_score': 0.0
        }