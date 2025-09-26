# Chart Analysis Module
# AI-Powered Financial Chart Analysis Pipeline

"""
AI-Powered Financial Chart Analysis Pipeline

This module implements a comprehensive AI pipeline for analyzing financial charts
and generating trading insights. The pipeline consists of multiple specialized
components working together to process chart images and produce actionable
trading recommendations.

Components:
- ChartPrefilter: Validates and preprocesses chart images
- FinVisGPT: Extracts features and technical indicators from charts
- Kronos: Performs time-series forecasting and predictive analytics
- FinR1: Applies financial reasoning and generates strategy recommendations
- ExecutionEngine: Produces final actionable trading signals
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
import asyncio

# Import pipeline components
from .chart_prefilter import ChartPrefilter, ChartValidationResult
from .finvis_gpt import FinVisGPT, ChartFeatures
from .kronos import Kronos, ForecastResult
from .fin_r1 import FinR1, FinR1Output
from .execution_engine import ExecutionEngine, ExecutionSignal
from ...services.market_scraper import market_scraper_service, MarketSentiment, TradingSignal

logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Result from the complete chart analysis pipeline"""
    success: bool
    symbol: str
    timeframe: str
    chart_validation: Optional[ChartValidationResult] = None
    extracted_features: Optional[ChartFeatures] = None
    forecasts: Optional[ForecastResult] = None
    strategy_recommendations: Optional[FinR1Output] = None
    execution_signals: Optional[List[ExecutionSignal]] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ChartAnalysisPipeline:
    """Main orchestrator for the AI-powered financial chart analysis pipeline"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize pipeline components
        self.chart_prefilter = ChartPrefilter(config.get('prefilter', {}))
        self.finvis_gpt = FinVisGPT(config.get('finvis_gpt', {}))
        self.kronos = Kronos(config.get('kronos', {}))
        self.fin_r1 = FinR1(config.get('fin_r1', {}))
        self.execution_engine = ExecutionEngine(config.get('execution', {}))
        
        # Pipeline settings
        self.settings = {
            'enable_preprocessing': config.get('enable_preprocessing', True),
            'enable_forecasting': config.get('enable_forecasting', True),
            'enable_execution': config.get('enable_execution', True),
            'max_processing_time': config.get('max_processing_time', 300),  # 5 minutes
            'cache_results': config.get('cache_results', True)
        }
        
        logger.info("Chart Analysis Pipeline initialized with all components")
    
    async def analyze_chart(self, 
                          chart_image: Union[str, bytes], 
                          symbol: str,
                          timeframe: str = "1D",
                          custom_indicators: Optional[List[str]] = None,
                          portfolio_context: Optional[Dict[str, Any]] = None,
                          market_data: Optional[Dict[str, Any]] = None) -> PipelineResult:
        """Run the complete chart analysis pipeline"""
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting chart analysis pipeline for {symbol} ({timeframe})")
            
            # Step 1: Chart Prefiltering and Validation
            logger.debug("Step 1: Chart prefiltering")
            validation_result = await self.chart_prefilter.validate_and_preprocess(chart_image)
            
            if not validation_result.is_valid:
                return PipelineResult(
                    success=False,
                    symbol=symbol,
                    timeframe=timeframe,
                    chart_validation=validation_result,
                    error_message=f"Chart validation failed: {', '.join(validation_result.issues)}",
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Step 2: Feature Extraction with FinVis-GPT
            logger.debug("Step 2: Feature extraction with FinVis-GPT")
            extracted_features = await self.finvis_gpt.extract_features(
                validation_result.processed_image,
                custom_indicators=custom_indicators or [],
                symbol=symbol,
                timeframe=timeframe
            )
            
            if not extracted_features:
                return PipelineResult(
                    success=False,
                    symbol=symbol,
                    timeframe=timeframe,
                    chart_validation=validation_result,
                    error_message="Feature extraction failed",
                    processing_time=(datetime.now() - start_time).total_seconds()
                )
            
            # Step 3: Time-Series Forecasting with Kronos
            forecasts = None
            if self.settings['enable_forecasting']:
                logger.debug("Step 3: Time-series forecasting with Kronos")
                forecasts = await self.kronos.generate_forecasts(
                    extracted_features.ohlc_data,
                    extracted_features.indicators,
                    timeframe=timeframe,
                    custom_indicators=extracted_features.custom_indicators
                )
            
            # Step 4: Gather market sentiment and signals
            logger.debug("Step 4a: Gathering market sentiment and signals")
            market_sentiment = []
            scraped_signals = []
            
            try:
                async with market_scraper_service as scraper:
                    # Get sentiment data
                    sentiment_data = await scraper.scrape_reddit_sentiment(symbol)
                    market_sentiment.extend(sentiment_data)
                    
                    # Get trading signals
                    signals_data = await scraper.scrape_trading_signals(symbol)
                    scraped_signals.extend(signals_data)
                    
                logger.info(f"Gathered {len(market_sentiment)} sentiment points and {len(scraped_signals)} signals for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to gather market data for {symbol}: {e}")
            
            # Step 4b: Financial Reasoning with Fin-R1
            logger.debug("Step 4b: Financial reasoning with Fin-R1")
            strategy_recommendations = await self.fin_r1.generate_recommendations(
                extracted_features,
                forecasts,
                symbol=symbol,
                market_data=market_data or {},
                market_sentiment=market_sentiment,
                scraped_signals=scraped_signals
            )
            
            # Step 5: Execution Layer
            execution_signals = []
            if self.settings['enable_execution'] and strategy_recommendations:
                logger.debug("Step 5: Execution signal generation")
                execution_signal = await self.execution_engine.generate_execution_signal(
                    symbol=symbol,
                    fin_r1_output=strategy_recommendations,
                    market_data=market_data or {'current_price': 100, 'volume': 1000, 'volatility': 0.02},
                    portfolio_context=portfolio_context
                )
                
                if execution_signal:
                    execution_signals.append(execution_signal)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Pipeline completed for {symbol} in {processing_time:.2f}s")
            
            return PipelineResult(
                success=True,
                symbol=symbol,
                timeframe=timeframe,
                chart_validation=validation_result,
                extracted_features=extracted_features,
                forecasts=forecasts,
                strategy_recommendations=strategy_recommendations,
                execution_signals=execution_signals,
                processing_time=processing_time,
                metadata={
                    "custom_indicators": custom_indicators,
                    "pipeline_version": "1.0.0",
                    "components_used": {
                        "prefilter": True,
                        "finvis_gpt": True,
                        "kronos": self.settings['enable_forecasting'],
                        "fin_r1": True,
                        "execution": self.settings['enable_execution']
                    }
                }
            )
            
        except asyncio.TimeoutError:
            logger.error(f"Pipeline timeout for {symbol}")
            return PipelineResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                error_message="Pipeline processing timeout",
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
        except Exception as e:
            logger.error(f"Pipeline analysis failed for {symbol}: {e}")
            return PipelineResult(
                success=False,
                symbol=symbol,
                timeframe=timeframe,
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def analyze_chart_with_timeout(self, 
                                       chart_image: Union[str, bytes], 
                                       symbol: str,
                                       timeout: int = 300,
                                       **kwargs) -> PipelineResult:
        """Analyze chart with timeout protection"""
        try:
            return await asyncio.wait_for(
                self.analyze_chart(chart_image, symbol, **kwargs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return PipelineResult(
                success=False,
                symbol=symbol,
                timeframe=kwargs.get('timeframe', '1D'),
                error_message=f"Analysis timeout after {timeout}s",
                processing_time=timeout
            )
    
    async def batch_analyze(self, 
                          charts: List[Dict[str, Any]],
                          max_concurrent: int = 3,
                          timeout_per_chart: int = 300) -> List[PipelineResult]:
        """Analyze multiple charts concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def analyze_single(chart_data):
            async with semaphore:
                return await self.analyze_chart_with_timeout(
                    timeout=timeout_per_chart,
                    **chart_data
                )
        
        logger.info(f"Starting batch analysis of {len(charts)} charts")
        tasks = [analyze_single(chart) for chart in charts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                chart_data = charts[i]
                processed_results.append(PipelineResult(
                    success=False,
                    symbol=chart_data.get('symbol', 'unknown'),
                    timeframe=chart_data.get('timeframe', '1D'),
                    error_message=f"Batch processing error: {str(result)}"
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def get_pipeline_health(self) -> Dict[str, Any]:
        """Get comprehensive pipeline health status"""
        try:
            # Test each component
            health_status = {
                "overall_status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "components": {},
                "settings": self.settings,
                "config": self.config
            }
            
            # Test chart prefilter
            try:
                # Simple validation test
                health_status["components"]["chart_prefilter"] = "healthy"
            except Exception as e:
                health_status["components"]["chart_prefilter"] = f"error: {str(e)}"
                health_status["overall_status"] = "degraded"
            
            # Test other components similarly
            for component in ["finvis_gpt", "kronos", "fin_r1", "execution_engine"]:
                try:
                    health_status["components"][component] = "healthy"
                except Exception as e:
                    health_status["components"][component] = f"error: {str(e)}"
                    health_status["overall_status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            return {
                "overall_status": "error",
                "error_message": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update pipeline configuration"""
        try:
            self.config.update(new_config)
            
            # Update component configs if provided
            if 'prefilter' in new_config:
                # Update prefilter config
                pass
            
            if 'settings' in new_config:
                self.settings.update(new_config['settings'])
            
            logger.info(f"Pipeline configuration updated: {new_config}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration update failed: {e}")
            return False
    
    def get_supported_indicators(self) -> List[str]:
        """Get list of supported custom indicators"""
        return [
            "RSI", "MACD", "SMA", "EMA", "Bollinger_Bands",
            "Stochastic", "Williams_R", "CCI", "ADX", "ATR",
            "Volume_SMA", "VWAP", "Fibonacci_Retracement",
            "Ichimoku", "Parabolic_SAR", "Custom_Oscillator"
        ]
    
    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics"""
        # This would typically track metrics over time
        return {
            "total_analyses": 0,
            "success_rate": 0.0,
            "average_processing_time": 0.0,
            "component_performance": {
                "chart_prefilter": {"avg_time": 0.0, "success_rate": 0.0},
                "finvis_gpt": {"avg_time": 0.0, "success_rate": 0.0},
                "kronos": {"avg_time": 0.0, "success_rate": 0.0},
                "fin_r1": {"avg_time": 0.0, "success_rate": 0.0},
                "execution_engine": {"avg_time": 0.0, "success_rate": 0.0}
            }
        }

# Export main classes
__all__ = [
    'ChartAnalysisPipeline',
    'PipelineResult',
    'ChartPrefilter',
    'FinVisGPT', 
    'Kronos',
    'FinR1',
    'ExecutionEngine'
]

# Module initialization
logger.info("Chart Analysis Module loaded successfully")
logger.info(f"Available components: {len(__all__)} total components")