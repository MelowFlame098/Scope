"""Chart Analysis API Endpoints

This module provides REST API endpoints for the AI-powered financial chart analysis pipeline.
It exposes the chart analysis functionality through HTTP endpoints for frontend integration.
"""

import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import base64
import io
from PIL import Image

# Import the chart analysis pipeline
from ..ai_core.chart_analysis import ChartAnalysisPipeline, PipelineResult
from ..auth.dependencies import get_current_user
from ..models.user import User

logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/api/chart-analysis", tags=["Chart Analysis"])

# Initialize pipeline (singleton)
pipeline = None

def get_pipeline() -> ChartAnalysisPipeline:
    """Get or create chart analysis pipeline instance"""
    global pipeline
    if pipeline is None:
        config = {
            'enable_preprocessing': True,
            'enable_forecasting': True,
            'enable_execution': True,
            'max_processing_time': 300,
            'cache_results': True
        }
        pipeline = ChartAnalysisPipeline(config)
        logger.info("Chart analysis pipeline initialized")
    return pipeline

# Request/Response Models
class ChartAnalysisRequest(BaseModel):
    """Request model for chart analysis"""
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL, TSLA)")
    timeframe: str = Field(default="1D", description="Chart timeframe (1m, 5m, 1h, 1D, 1W)")
    custom_indicators: Optional[List[str]] = Field(default=None, description="Custom indicators to include")
    enable_forecasting: Optional[bool] = Field(default=True, description="Enable time-series forecasting")
    enable_execution: Optional[bool] = Field(default=True, description="Enable execution signal generation")
    market_data: Optional[Dict[str, Any]] = Field(default=None, description="Additional market data context")

class BatchAnalysisRequest(BaseModel):
    """Request model for batch chart analysis"""
    charts: List[Dict[str, Any]] = Field(..., description="List of chart analysis requests")
    max_concurrent: Optional[int] = Field(default=3, description="Maximum concurrent analyses")
    timeout_per_chart: Optional[int] = Field(default=300, description="Timeout per chart in seconds")

class PipelineConfigRequest(BaseModel):
    """Request model for updating pipeline configuration"""
    enable_preprocessing: Optional[bool] = None
    enable_forecasting: Optional[bool] = None
    enable_execution: Optional[bool] = None
    max_processing_time: Optional[int] = None
    cache_results: Optional[bool] = None

class ChartAnalysisResponse(BaseModel):
    """Response model for chart analysis"""
    success: bool
    symbol: str
    timeframe: str
    processing_time: float
    chart_validation: Optional[Dict[str, Any]] = None
    extracted_features: Optional[Dict[str, Any]] = None
    forecasts: Optional[Dict[str, Any]] = None
    strategy_recommendations: Optional[Dict[str, Any]] = None
    execution_signals: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

# API Endpoints

@router.post("/analyze", response_model=ChartAnalysisResponse)
async def analyze_chart(
    chart_file: UploadFile = File(..., description="Chart image file"),
    symbol: str = Form(..., description="Stock symbol"),
    timeframe: str = Form(default="1D", description="Chart timeframe"),
    custom_indicators: Optional[str] = Form(default=None, description="Comma-separated custom indicators"),
    enable_forecasting: bool = Form(default=True, description="Enable forecasting"),
    enable_execution: bool = Form(default=True, description="Enable execution signals"),
    current_user: User = Depends(get_current_user)
):
    """Analyze a financial chart image and generate trading insights"""
    try:
        # Validate file type
        if not chart_file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        image_data = await chart_file.read()
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        # Parse custom indicators
        custom_indicators_list = None
        if custom_indicators:
            custom_indicators_list = [ind.strip() for ind in custom_indicators.split(',') if ind.strip()]
        
        # Get pipeline and analyze
        analysis_pipeline = get_pipeline()
        
        # Update pipeline settings for this request
        temp_config = {
            'settings': {
                'enable_forecasting': enable_forecasting,
                'enable_execution': enable_execution
            }
        }
        await analysis_pipeline.update_config(temp_config)
        
        # Run analysis
        result = await analysis_pipeline.analyze_chart(
            chart_image=image_data,
            symbol=symbol.upper(),
            timeframe=timeframe,
            custom_indicators=custom_indicators_list,
            portfolio_context={'user_id': current_user.id},
            market_data={'current_price': 100, 'volume': 1000, 'volatility': 0.02}  # Mock data
        )
        
        # Convert result to response format
        response_data = {
            'success': result.success,
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'processing_time': result.processing_time or 0.0,
            'error_message': result.error_message,
            'metadata': result.metadata
        }
        
        # Add optional fields if present
        if result.chart_validation:
            response_data['chart_validation'] = {
                'is_valid': result.chart_validation.is_valid,
                'chart_type': result.chart_validation.chart_type.value if result.chart_validation.chart_type else None,
                'quality_score': result.chart_validation.quality_score,
                'issues': result.chart_validation.issues
            }
        
        if result.extracted_features:
            response_data['extracted_features'] = {
                'ohlc_count': len(result.extracted_features.ohlc_data) if result.extracted_features.ohlc_data else 0,
                'indicators_count': len(result.extracted_features.indicators) if result.extracted_features.indicators else 0,
                'custom_indicators_count': len(result.extracted_features.custom_indicators) if result.extracted_features.custom_indicators else 0,
                'confidence': result.extracted_features.confidence
            }
        
        if result.forecasts:
            response_data['forecasts'] = {
                'price_forecast': {
                    'direction': result.forecasts.price_forecast.direction.value if result.forecasts.price_forecast else None,
                    'confidence': result.forecasts.price_forecast.confidence if result.forecasts.price_forecast else 0.0
                } if result.forecasts.price_forecast else None,
                'volatility_forecast': {
                    'level': result.forecasts.volatility_forecast.level.value if result.forecasts.volatility_forecast else None,
                    'confidence': result.forecasts.volatility_forecast.confidence if result.forecasts.volatility_forecast else 0.0
                } if result.forecasts.volatility_forecast else None
            }
        
        if result.strategy_recommendations:
            response_data['strategy_recommendations'] = {
                'primary_recommendation': result.strategy_recommendations.primary_recommendation.value if result.strategy_recommendations.primary_recommendation else None,
                'confidence': result.strategy_recommendations.confidence,
                'strategies_count': len(result.strategy_recommendations.strategy_recommendations) if result.strategy_recommendations.strategy_recommendations else 0
            }
        
        if result.execution_signals:
            response_data['execution_signals'] = [
                {
                    'action': signal.action,
                    'confidence': signal.confidence,
                    'signal_strength': signal.signal_strength.value,
                    'position_size': signal.position_size,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'reasoning': signal.reasoning
                }
                for signal in result.execution_signals
            ]
        
        return ChartAnalysisResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze-base64", response_model=ChartAnalysisResponse)
async def analyze_chart_base64(
    request: ChartAnalysisRequest,
    chart_data: str = Form(..., description="Base64 encoded chart image"),
    current_user: User = Depends(get_current_user)
):
    """Analyze a chart from base64 encoded image data"""
    try:
        # Decode base64 image
        try:
            image_data = base64.b64decode(chart_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")
        
        # Validate image
        try:
            Image.open(io.BytesIO(image_data))
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Get pipeline and analyze
        analysis_pipeline = get_pipeline()
        
        # Run analysis
        result = await analysis_pipeline.analyze_chart(
            chart_image=image_data,
            symbol=request.symbol.upper(),
            timeframe=request.timeframe,
            custom_indicators=request.custom_indicators,
            portfolio_context={'user_id': current_user.id},
            market_data=request.market_data or {'current_price': 100, 'volume': 1000, 'volatility': 0.02}
        )
        
        # Convert and return response (same logic as above)
        response_data = {
            'success': result.success,
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'processing_time': result.processing_time or 0.0,
            'error_message': result.error_message,
            'metadata': result.metadata
        }
        
        return ChartAnalysisResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Base64 chart analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/batch-analyze")
async def batch_analyze_charts(
    request: BatchAnalysisRequest,
    current_user: User = Depends(get_current_user)
):
    """Analyze multiple charts in batch"""
    try:
        analysis_pipeline = get_pipeline()
        
        # Add user context to each chart request
        for chart_data in request.charts:
            chart_data['portfolio_context'] = {'user_id': current_user.id}
        
        # Run batch analysis
        results = await analysis_pipeline.batch_analyze(
            charts=request.charts,
            max_concurrent=request.max_concurrent,
            timeout_per_chart=request.timeout_per_chart
        )
        
        # Convert results to response format
        response_results = []
        for result in results:
            response_data = {
                'success': result.success,
                'symbol': result.symbol,
                'timeframe': result.timeframe,
                'processing_time': result.processing_time or 0.0,
                'error_message': result.error_message
            }
            response_results.append(response_data)
        
        return {
            'success': True,
            'total_charts': len(request.charts),
            'successful_analyses': len([r for r in results if r.success]),
            'failed_analyses': len([r for r in results if not r.success]),
            'results': response_results
        }
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@router.get("/health")
async def get_pipeline_health():
    """Get pipeline health status"""
    try:
        analysis_pipeline = get_pipeline()
        health_status = await analysis_pipeline.get_pipeline_health()
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'overall_status': 'error',
            'error_message': str(e),
            'timestamp': datetime.now().isoformat()
        }

@router.get("/supported-indicators")
async def get_supported_indicators():
    """Get list of supported custom indicators"""
    try:
        analysis_pipeline = get_pipeline()
        indicators = analysis_pipeline.get_supported_indicators()
        return {
            'indicators': indicators,
            'count': len(indicators)
        }
    except Exception as e:
        logger.error(f"Failed to get supported indicators: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve indicators")

@router.get("/metrics")
async def get_pipeline_metrics(
    current_user: User = Depends(get_current_user)
):
    """Get pipeline performance metrics"""
    try:
        analysis_pipeline = get_pipeline()
        metrics = analysis_pipeline.get_pipeline_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Failed to get pipeline metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve metrics")

@router.post("/config")
async def update_pipeline_config(
    request: PipelineConfigRequest,
    current_user: User = Depends(get_current_user)
):
    """Update pipeline configuration (admin only)"""
    try:
        # Check if user has admin privileges (implement based on your auth system)
        # if not current_user.is_admin:
        #     raise HTTPException(status_code=403, detail="Admin access required")
        
        analysis_pipeline = get_pipeline()
        
        # Convert request to config dict
        config_updates = {}
        if request.enable_preprocessing is not None:
            config_updates['enable_preprocessing'] = request.enable_preprocessing
        if request.enable_forecasting is not None:
            config_updates['enable_forecasting'] = request.enable_forecasting
        if request.enable_execution is not None:
            config_updates['enable_execution'] = request.enable_execution
        if request.max_processing_time is not None:
            config_updates['max_processing_time'] = request.max_processing_time
        if request.cache_results is not None:
            config_updates['cache_results'] = request.cache_results
        
        success = await analysis_pipeline.update_config({'settings': config_updates})
        
        return {
            'success': success,
            'updated_config': config_updates,
            'timestamp': datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        raise HTTPException(status_code=500, detail="Configuration update failed")

@router.get("/status")
async def get_pipeline_status():
    """Get current pipeline status and configuration"""
    try:
        analysis_pipeline = get_pipeline()
        return {
            'status': 'operational',
            'pipeline_initialized': pipeline is not None,
            'settings': analysis_pipeline.settings,
            'supported_timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '1D', '1W', '1M'],
            'max_file_size': '10MB',
            'supported_formats': ['PNG', 'JPEG', 'JPG', 'GIF', 'BMP'],
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            'status': 'error',
            'error_message': str(e),
            'timestamp': datetime.now().isoformat()
        }

# Error handlers
@router.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': exc.detail,
            'timestamp': datetime.now().isoformat()
        }
    )

@router.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception in chart analysis API: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            'error': 'Internal server error',
            'timestamp': datetime.now().isoformat()
        }
    )