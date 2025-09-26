"""Explanation Generator for FinScope - Phase 6 Implementation

Generates comprehensive financial explanations by combining market data,
technical analysis, and AI/ML model outputs with LLM-powered insights.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

from llm_service import LLMService, ExplanationRequest, ExplanationResponse, LLMProvider
from prompt_engine import PromptEngine, PromptType, ExplanationComplexity
from financial_context import FinancialContextEngine
# from market_data_service import MarketDataService  # Module not found
from technical_analysis import TechnicalAnalysisService

logger = logging.getLogger(__name__)

class ExplanationType(str, Enum):
    """Types of financial explanations"""
    PRICE_MOVEMENT = "price_movement"
    TECHNICAL_ANALYSIS = "technical_analysis"
    VOLUME_ANALYSIS = "volume_analysis"
    MARKET_SENTIMENT = "market_sentiment"
    RISK_ASSESSMENT = "risk_assessment"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    MODEL_PREDICTION = "model_prediction"
    NEWS_IMPACT = "news_impact"
    CORRELATION_ANALYSIS = "correlation_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"

class ExplanationScope(str, Enum):
    """Scope of explanation analysis"""
    SINGLE_ASSET = "single_asset"
    PORTFOLIO = "portfolio"
    SECTOR = "sector"
    MARKET = "market"
    GLOBAL = "global"

@dataclass
class ExplanationContext:
    """Context for explanation generation"""
    symbol: Optional[str] = None
    timeframe: str = "1d"
    period: str = "1mo"
    user_id: Optional[str] = None
    portfolio_data: Optional[Dict[str, Any]] = None
    market_conditions: Optional[Dict[str, Any]] = None
    news_context: Optional[List[Dict[str, Any]]] = None
    technical_indicators: Optional[Dict[str, Any]] = None
    model_outputs: Optional[Dict[str, Any]] = None

class FinancialExplanationRequest(BaseModel):
    """Request for financial explanation generation"""
    explanation_type: ExplanationType
    scope: ExplanationScope = ExplanationScope.SINGLE_ASSET
    symbol: Optional[str] = None
    symbols: Optional[List[str]] = None
    timeframe: str = "1d"
    period: str = "1mo"
    complexity: ExplanationComplexity = ExplanationComplexity.INTERMEDIATE
    user_id: Optional[str] = None
    conversation_id: Optional[str] = None
    include_predictions: bool = True
    include_recommendations: bool = True
    context: Dict[str, Any] = Field(default_factory=dict)

class FinancialExplanationResponse(BaseModel):
    """Response for financial explanation"""
    explanation: str
    summary: str
    key_insights: List[str]
    data_sources: List[str]
    confidence_score: float
    recommendations: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    charts_suggested: List[str] = Field(default_factory=list)
    related_symbols: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ExplanationGenerator:
    """Advanced financial explanation generator"""
    
    def __init__(self):
        self.llm_service = LLMService()
        self.prompt_engine = PromptEngine()
        self.context_engine = FinancialContextEngine()
        # self.market_service = MarketDataService()  # Module not available
        self.technical_service = TechnicalAnalysisService()
        
        # Explanation templates and configurations
        self.explanation_configs = self._initialize_explanation_configs()
    
    def _initialize_explanation_configs(self) -> Dict[ExplanationType, Dict[str, Any]]:
        """Initialize configuration for different explanation types"""
        return {
            ExplanationType.PRICE_MOVEMENT: {
                "required_data": ["price_data", "volume_data"],
                "optional_data": ["news", "market_sentiment", "technical_indicators"],
                "analysis_depth": "comprehensive",
                "include_charts": True
            },
            ExplanationType.TECHNICAL_ANALYSIS: {
                "required_data": ["price_data", "technical_indicators"],
                "optional_data": ["volume_data", "market_context"],
                "analysis_depth": "detailed",
                "include_charts": True
            },
            ExplanationType.VOLUME_ANALYSIS: {
                "required_data": ["volume_data", "price_data"],
                "optional_data": ["market_microstructure"],
                "analysis_depth": "focused",
                "include_charts": True
            },
            ExplanationType.MARKET_SENTIMENT: {
                "required_data": ["sentiment_data"],
                "optional_data": ["news", "social_media", "options_flow"],
                "analysis_depth": "contextual",
                "include_charts": False
            },
            ExplanationType.RISK_ASSESSMENT: {
                "required_data": ["price_data", "volatility_data"],
                "optional_data": ["correlation_data", "portfolio_data"],
                "analysis_depth": "quantitative",
                "include_charts": True
            },
            ExplanationType.MODEL_PREDICTION: {
                "required_data": ["model_output"],
                "optional_data": ["feature_importance", "confidence_intervals"],
                "analysis_depth": "interpretive",
                "include_charts": False
            }
        }
    
    async def generate_explanation(
        self,
        request: FinancialExplanationRequest,
        provider: LLMProvider = LLMProvider.OPENAI
    ) -> FinancialExplanationResponse:
        """Generate comprehensive financial explanation"""
        try:
            # Gather required data
            context = await self._gather_explanation_context(request)
            
            # Generate explanation based on type
            if request.explanation_type == ExplanationType.PRICE_MOVEMENT:
                response = await self._explain_price_movement(request, context, provider)
            elif request.explanation_type == ExplanationType.TECHNICAL_ANALYSIS:
                response = await self._explain_technical_analysis(request, context, provider)
            elif request.explanation_type == ExplanationType.VOLUME_ANALYSIS:
                response = await self._explain_volume_analysis(request, context, provider)
            elif request.explanation_type == ExplanationType.MARKET_SENTIMENT:
                response = await self._explain_market_sentiment(request, context, provider)
            elif request.explanation_type == ExplanationType.RISK_ASSESSMENT:
                response = await self._explain_risk_assessment(request, context, provider)
            elif request.explanation_type == ExplanationType.MODEL_PREDICTION:
                response = await self._explain_model_prediction(request, context, provider)
            else:
                response = await self._generate_generic_explanation(request, context, provider)
            
            # Enhance response with additional insights
            response = await self._enhance_explanation_response(response, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return FinancialExplanationResponse(
                explanation=f"Unable to generate explanation: {str(e)}",
                summary="Error occurred during explanation generation",
                key_insights=[],
                data_sources=[],
                confidence_score=0.0
            )
    
    async def _gather_explanation_context(
        self,
        request: FinancialExplanationRequest
    ) -> ExplanationContext:
        """Gather all necessary context for explanation"""
        context = ExplanationContext(
            symbol=request.symbol,
            timeframe=request.timeframe,
            period=request.period,
            user_id=request.user_id
        )
        
        # Get market data if symbol provided
        if request.symbol:
            try:
                # Get price and volume data
                market_data = await self.market_service.get_historical_data(
                    request.symbol,
                    period=request.period,
                    interval=request.timeframe
                )
                context.market_conditions = market_data
                
                # Get technical indicators
                if request.explanation_type in [
                    ExplanationType.TECHNICAL_ANALYSIS,
                    ExplanationType.PRICE_MOVEMENT
                ]:
                    indicators = await self.technical_service.calculate_indicators(
                        request.symbol,
                        timeframe=request.timeframe
                    )
                    context.technical_indicators = indicators
                
            except Exception as e:
                logger.warning(f"Could not gather market data: {e}")
        
        # Get portfolio data if user provided
        if request.user_id:
            try:
                portfolio_data = await self.context_engine.get_user_portfolio_context(
                    request.user_id
                )
                context.portfolio_data = portfolio_data
            except Exception as e:
                logger.warning(f"Could not gather portfolio data: {e}")
        
        # Get news context
        if request.symbol and request.explanation_type in [
            ExplanationType.PRICE_MOVEMENT,
            ExplanationType.MARKET_SENTIMENT,
            ExplanationType.NEWS_IMPACT
        ]:
            try:
                news_data = await self.context_engine.get_relevant_news(
                    request.symbol,
                    hours_back=24
                )
                context.news_context = news_data
            except Exception as e:
                logger.warning(f"Could not gather news data: {e}")
        
        return context
    
    async def _explain_price_movement(
        self,
        request: FinancialExplanationRequest,
        context: ExplanationContext,
        provider: LLMProvider
    ) -> FinancialExplanationResponse:
        """Generate explanation for price movements"""
        # Analyze price data
        price_analysis = await self._analyze_price_data(context)
        
        # Prepare explanation content
        content = f"""Price Movement Analysis for {request.symbol}:

{price_analysis['summary']}

Key Metrics:
- Current Price: ${price_analysis.get('current_price', 'N/A')}
- Price Change: {price_analysis.get('price_change', 'N/A')}%
- Volume: {price_analysis.get('volume', 'N/A')}
- Volatility: {price_analysis.get('volatility', 'N/A')}%

Technical Signals:
{self._format_technical_signals(context.technical_indicators)}

Market Context:
{self._format_market_context(context.market_conditions)}
"""
        
        if context.news_context:
            content += f"\nRelevant News:\n{self._format_news_context(context.news_context)}"
        
        # Generate LLM explanation
        llm_request = ExplanationRequest(
            content=content,
            context={
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "analysis_type": "price_movement"
            },
            complexity=request.complexity,
            conversation_id=request.conversation_id
        )
        
        llm_response = await self.llm_service.generate_explanation(
            llm_request, provider
        )
        
        return FinancialExplanationResponse(
            explanation=llm_response.explanation,
            summary=price_analysis['summary'],
            key_insights=self._extract_key_insights(llm_response.explanation),
            data_sources=["Market Data", "Technical Analysis", "News"],
            confidence_score=llm_response.confidence,
            recommendations=self._generate_price_recommendations(price_analysis),
            follow_up_questions=llm_response.follow_up_questions,
            charts_suggested=["Price Chart", "Volume Chart", "Technical Indicators"],
            related_symbols=await self._get_related_symbols(request.symbol),
            metadata={
                "price_analysis": price_analysis,
                "technical_indicators": context.technical_indicators
            }
        )
    
    async def _explain_technical_analysis(
        self,
        request: FinancialExplanationRequest,
        context: ExplanationContext,
        provider: LLMProvider
    ) -> FinancialExplanationResponse:
        """Generate explanation for technical analysis"""
        if not context.technical_indicators:
            raise ValueError("Technical indicators required for technical analysis explanation")
        
        # Analyze technical indicators
        ta_analysis = await self._analyze_technical_indicators(context.technical_indicators)
        
        # Generate prompt using prompt engine
        prompt = await self.prompt_engine.generate_technical_analysis_prompt(
            indicators=context.technical_indicators,
            symbol=request.symbol,
            timeframe=request.timeframe,
            complexity=request.complexity
        )
        
        # Generate LLM explanation
        llm_request = ExplanationRequest(
            content=prompt,
            context={
                "symbol": request.symbol,
                "timeframe": request.timeframe,
                "analysis_type": "technical_analysis"
            },
            complexity=request.complexity,
            conversation_id=request.conversation_id
        )
        
        llm_response = await self.llm_service.generate_explanation(
            llm_request, provider
        )
        
        return FinancialExplanationResponse(
            explanation=llm_response.explanation,
            summary=ta_analysis['summary'],
            key_insights=ta_analysis['signals'],
            data_sources=["Technical Analysis", "Price Data"],
            confidence_score=ta_analysis['confidence'],
            recommendations=ta_analysis['recommendations'],
            follow_up_questions=llm_response.follow_up_questions,
            charts_suggested=["Candlestick Chart", "Indicator Overlay", "Signal Chart"],
            metadata={"technical_analysis": ta_analysis}
        )
    
    async def _explain_model_prediction(
        self,
        request: FinancialExplanationRequest,
        context: ExplanationContext,
        provider: LLMProvider
    ) -> FinancialExplanationResponse:
        """Generate explanation for AI/ML model predictions"""
        model_output = request.context.get("model_output")
        if not model_output:
            raise ValueError("Model output required for model prediction explanation")
        
        model_info = request.context.get("model_info", {})
        
        # Generate prompt using prompt engine
        prompt = await self.prompt_engine.generate_model_interpretation_prompt(
            model_output=model_output,
            model_info=model_info,
            complexity=request.complexity
        )
        
        # Generate LLM explanation
        llm_request = ExplanationRequest(
            content=prompt,
            context={
                "model_type": model_info.get("type", "Unknown"),
                "confidence": model_output.get("confidence", 0),
                "prediction_horizon": model_info.get("horizon", "Unknown")
            },
            complexity=request.complexity,
            conversation_id=request.conversation_id
        )
        
        llm_response = await self.llm_service.generate_explanation(
            llm_request, provider
        )
        
        return FinancialExplanationResponse(
            explanation=llm_response.explanation,
            summary=f"Model predicts {model_output.get('prediction', 'N/A')} with {model_output.get('confidence', 0):.1%} confidence",
            key_insights=self._extract_model_insights(model_output),
            data_sources=["AI/ML Model", "Historical Data"],
            confidence_score=model_output.get("confidence", 0.5),
            recommendations=self._generate_model_recommendations(model_output),
            follow_up_questions=llm_response.follow_up_questions,
            metadata={"model_output": model_output, "model_info": model_info}
        )
    
    async def _generate_generic_explanation(
        self,
        request: FinancialExplanationRequest,
        context: ExplanationContext,
        provider: LLMProvider
    ) -> FinancialExplanationResponse:
        """Generate generic financial explanation"""
        content = f"Financial analysis for {request.symbol or 'portfolio'}:\n"
        
        if context.market_conditions:
            content += f"Market Data: {context.market_conditions}\n"
        
        if context.technical_indicators:
            content += f"Technical Analysis: {context.technical_indicators}\n"
        
        if context.portfolio_data:
            content += f"Portfolio Context: {context.portfolio_data}\n"
        
        # Generate LLM explanation
        llm_request = ExplanationRequest(
            content=content,
            context=request.context,
            complexity=request.complexity,
            conversation_id=request.conversation_id
        )
        
        llm_response = await self.llm_service.generate_explanation(
            llm_request, provider
        )
        
        return FinancialExplanationResponse(
            explanation=llm_response.explanation,
            summary="General financial analysis",
            key_insights=self._extract_key_insights(llm_response.explanation),
            data_sources=["Market Data"],
            confidence_score=llm_response.confidence,
            follow_up_questions=llm_response.follow_up_questions
        )
    
    async def _analyze_price_data(self, context: ExplanationContext) -> Dict[str, Any]:
        """Analyze price data for insights"""
        if not context.market_conditions:
            return {"summary": "No price data available"}
        
        # Extract price information (mock implementation)
        return {
            "summary": "Price analysis completed",
            "current_price": 150.25,
            "price_change": 2.5,
            "volume": "1.2M",
            "volatility": 15.3
        }
    
    async def _analyze_technical_indicators(
        self,
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze technical indicators for signals"""
        # Mock analysis - would implement real technical analysis
        return {
            "summary": "Technical analysis shows mixed signals",
            "signals": ["RSI oversold", "MACD bullish crossover", "Support at $145"],
            "confidence": 0.75,
            "recommendations": ["Consider buying on dips", "Watch for volume confirmation"]
        }
    
    def _format_technical_signals(self, indicators: Optional[Dict[str, Any]]) -> str:
        """Format technical indicators for display"""
        if not indicators:
            return "No technical indicators available"
        
        formatted = []
        for indicator, value in indicators.items():
            if isinstance(value, dict):
                formatted.append(f"{indicator}: {value}")
            else:
                formatted.append(f"{indicator}: {value}")
        
        return "\n".join(formatted[:5])  # Limit to top 5
    
    def _format_market_context(self, market_data: Optional[Dict[str, Any]]) -> str:
        """Format market context for display"""
        if not market_data:
            return "No market context available"
        
        return f"Market conditions: {market_data.get('summary', 'Normal trading')}"
    
    def _format_news_context(self, news_data: Optional[List[Dict[str, Any]]]) -> str:
        """Format news context for display"""
        if not news_data:
            return "No relevant news"
        
        formatted = []
        for item in news_data[:3]:  # Top 3 news items
            formatted.append(f"- {item.get('title', 'News item')}: {item.get('sentiment', 'neutral')}")
        
        return "\n".join(formatted)
    
    def _extract_key_insights(self, explanation: str) -> List[str]:
        """Extract key insights from explanation text"""
        # Simple extraction - could use NLP for better results
        sentences = explanation.split(". ")
        insights = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in [
                "important", "significant", "notable", "key", "critical"
            ]):
                insights.append(sentence.strip())
        
        return insights[:3]  # Top 3 insights
    
    def _extract_model_insights(self, model_output: Dict[str, Any]) -> List[str]:
        """Extract insights from model output"""
        insights = []
        
        if "prediction" in model_output:
            insights.append(f"Model predicts: {model_output['prediction']}")
        
        if "confidence" in model_output:
            confidence = model_output["confidence"]
            if confidence > 0.8:
                insights.append("High confidence prediction")
            elif confidence < 0.6:
                insights.append("Low confidence - use caution")
        
        if "feature_importance" in model_output:
            top_feature = max(model_output["feature_importance"], key=lambda x: x["importance"])
            insights.append(f"Key factor: {top_feature['feature']}")
        
        return insights
    
    def _generate_price_recommendations(self, price_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on price analysis"""
        recommendations = []
        
        price_change = price_analysis.get("price_change", 0)
        if price_change > 5:
            recommendations.append("Consider taking profits on strong gains")
        elif price_change < -5:
            recommendations.append("Potential buying opportunity on weakness")
        
        volatility = price_analysis.get("volatility", 0)
        if volatility > 20:
            recommendations.append("High volatility - consider position sizing")
        
        return recommendations
    
    def _generate_model_recommendations(self, model_output: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on model output"""
        recommendations = []
        
        confidence = model_output.get("confidence", 0.5)
        if confidence < 0.6:
            recommendations.append("Low model confidence - seek additional confirmation")
        
        prediction = model_output.get("prediction", "")
        if "bullish" in str(prediction).lower():
            recommendations.append("Model suggests upward price movement")
        elif "bearish" in str(prediction).lower():
            recommendations.append("Model suggests downward price movement")
        
        return recommendations
    
    async def _get_related_symbols(self, symbol: str) -> List[str]:
        """Get symbols related to the given symbol"""
        # Mock implementation - would use real correlation analysis
        sector_symbols = {
            "AAPL": ["MSFT", "GOOGL", "AMZN"],
            "TSLA": ["NIO", "RIVN", "LCID"],
            "SPY": ["QQQ", "IWM", "DIA"]
        }
        
        return sector_symbols.get(symbol, [])
    
    async def _enhance_explanation_response(
        self,
        response: FinancialExplanationResponse,
        context: ExplanationContext
    ) -> FinancialExplanationResponse:
        """Enhance explanation response with additional context"""
        # Add timestamp and metadata
        response.timestamp = datetime.utcnow()
        
        # Add context to metadata
        if context.symbol:
            response.metadata["symbol"] = context.symbol
        if context.timeframe:
            response.metadata["timeframe"] = context.timeframe
        
        return response

# Global explanation generator instance
explanation_generator = ExplanationGenerator()