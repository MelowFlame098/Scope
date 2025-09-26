# Predictive Analytics Module
# Phase 9: AI-First Platform Implementation

"""
Predictive Analytics Module for FinScope AI-First Platform

This module provides advanced predictive analytics capabilities including:
- Market forecasting using ensemble models
- Risk prediction and scenario analysis
- Portfolio optimization predictions
- Economic indicator forecasting
- Sentiment-based market predictions
- Multi-timeframe trend analysis

Components:
- MarketForecaster: Advanced market prediction engine
- RiskPredictor: Risk assessment and prediction
- TrendAnalyzer: Trend detection and analysis
- ScenarioEngine: Scenario analysis and stress testing
- EconomicForecaster: Economic indicator predictions
- SentimentPredictor: Market sentiment analysis
"""

import logging
from typing import Optional

# Module metadata
__version__ = "1.0.0"
__author__ = "FinScope AI Team"
__description__ = "Advanced predictive analytics for financial markets"

# Configure logging
logger = logging.getLogger(__name__)

# Core components
try:
    from .market_forecaster import MarketForecaster, ForecastType, ForecastResult
    from .risk_predictor import RiskPredictor, RiskPrediction, RiskScenario
    from .trend_analyzer import TrendAnalyzer, TrendDirection, TrendStrength
    from .scenario_engine import ScenarioEngine, ScenarioType, ScenarioResult
    from .economic_forecaster import EconomicForecaster, EconomicIndicator
    from .sentiment_predictor import SentimentPredictor, SentimentSignal
    
    logger.info("Predictive analytics components loaded successfully")
    
except ImportError as e:
    logger.warning(f"Some predictive analytics components not available: {e}")
    
    # Fallback imports
    MarketForecaster = None
    RiskPredictor = None
    TrendAnalyzer = None
    ScenarioEngine = None
    EconomicForecaster = None
    SentimentPredictor = None

# Export main classes
__all__ = [
    'MarketForecaster',
    'RiskPredictor', 
    'TrendAnalyzer',
    'ScenarioEngine',
    'EconomicForecaster',
    'SentimentPredictor',
    'ForecastType',
    'ForecastResult',
    'RiskPrediction',
    'RiskScenario',
    'TrendDirection',
    'TrendStrength',
    'ScenarioType',
    'ScenarioResult',
    'EconomicIndicator',
    'SentimentSignal'
]

def get_version() -> str:
    """Get module version"""
    return __version__

def get_available_components() -> list:
    """Get list of available components"""
    available = []
    
    if MarketForecaster:
        available.append('MarketForecaster')
    if RiskPredictor:
        available.append('RiskPredictor')
    if TrendAnalyzer:
        available.append('TrendAnalyzer')
    if ScenarioEngine:
        available.append('ScenarioEngine')
    if EconomicForecaster:
        available.append('EconomicForecaster')
    if SentimentPredictor:
        available.append('SentimentPredictor')
    
    return available

class PredictiveAnalytics:
    """Main predictive analytics orchestrator"""
    
    def __init__(self):
        self.market_forecaster = MarketForecaster() if MarketForecaster else None
        self.risk_predictor = RiskPredictor() if RiskPredictor else None
        self.trend_analyzer = TrendAnalyzer() if TrendAnalyzer else None
        self.scenario_engine = ScenarioEngine() if ScenarioEngine else None
        self.economic_forecaster = EconomicForecaster() if EconomicForecaster else None
        self.sentiment_predictor = SentimentPredictor() if SentimentPredictor else None
        
        logger.info("Predictive analytics orchestrator initialized")
    
    def is_available(self) -> bool:
        """Check if predictive analytics is available"""
        return any([
            self.market_forecaster,
            self.risk_predictor,
            self.trend_analyzer,
            self.scenario_engine,
            self.economic_forecaster,
            self.sentiment_predictor
        ])
    
    async def get_comprehensive_forecast(self, symbol: str, timeframe: str = '1d') -> dict:
        """Get comprehensive forecast combining all available predictors"""
        try:
            results = {}
            
            # Market forecast
            if self.market_forecaster:
                results['market_forecast'] = await self.market_forecaster.forecast(symbol, timeframe)
            
            # Risk prediction
            if self.risk_predictor:
                results['risk_prediction'] = await self.risk_predictor.predict_risk(symbol)
            
            # Trend analysis
            if self.trend_analyzer:
                results['trend_analysis'] = await self.trend_analyzer.analyze_trends(symbol)
            
            # Scenario analysis
            if self.scenario_engine:
                results['scenario_analysis'] = await self.scenario_engine.run_scenarios(symbol)
            
            # Sentiment prediction
            if self.sentiment_predictor:
                results['sentiment_prediction'] = await self.sentiment_predictor.predict_sentiment(symbol)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting comprehensive forecast: {e}")
            return {}

# Global instance
predictive_analytics = PredictiveAnalytics()