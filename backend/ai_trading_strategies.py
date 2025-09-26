from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import asyncio
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from database import get_db
from db_models import User, Portfolio, Asset
import logging

logger = logging.getLogger(__name__)

class StrategyRequest(BaseModel):
    strategy_type: str
    parameters: Dict[str, Any]
    assets: List[str]
    risk_level: str = "medium"
    capital_allocation: float = 1.0

class StrategyResponse(BaseModel):
    strategy_id: str
    name: str
    description: str
    status: str
    performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    created_at: datetime
    updated_at: datetime

class SignalRequest(BaseModel):
    strategy_id: str
    asset: str
    signal_type: str  # buy, sell, hold
    confidence: float
    reasoning: str

class MarketPrediction(BaseModel):
    asset: str
    timeframe: str
    prediction: float
    confidence: float
    factors: List[str]
    timestamp: datetime

class TradingSignal(BaseModel):
    signal_id: str
    strategy_id: str
    asset: str
    action: str  # buy, sell, hold
    confidence: float
    price_target: Optional[float]
    stop_loss: Optional[float]
    reasoning: str
    timestamp: datetime

class AIInsight(BaseModel):
    insight_id: str
    type: str  # market_analysis, risk_alert, opportunity
    title: str
    description: str
    impact: str  # high, medium, low
    assets_affected: List[str]
    timestamp: datetime

class PortfolioOptimization(BaseModel):
    optimization_id: str
    current_allocation: Dict[str, float]
    recommended_allocation: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    reasoning: str
    timestamp: datetime

class SentimentAnalysis(BaseModel):
    asset: str
    sentiment_score: float  # -1 to 1
    sentiment_label: str  # bearish, neutral, bullish
    confidence: float
    sources: List[str]
    key_topics: List[str]
    timestamp: datetime

class AITradingStrategiesService:
    def __init__(self):
        self.active_strategies = {}
        self.signals_history = []
        self.predictions_cache = {}
        self.insights_cache = []
        
    async def get_strategies(self, user_id: str) -> List[StrategyResponse]:
        """Get all AI trading strategies for a user"""
        try:
            # Mock strategies for demonstration
            strategies = [
                {
                    "strategy_id": "momentum_001",
                    "name": "Momentum Trading",
                    "description": "AI-powered momentum strategy using technical indicators",
                    "status": "active",
                    "performance": {
                        "total_return": 12.5,
                        "win_rate": 68.2,
                        "avg_trade_duration": 3.2,
                        "max_drawdown": -5.8
                    },
                    "risk_metrics": {
                        "sharpe_ratio": 1.85,
                        "volatility": 15.2,
                        "beta": 1.12,
                        "var_95": -2.3
                    },
                    "created_at": datetime.now() - timedelta(days=30),
                    "updated_at": datetime.now()
                },
                {
                    "strategy_id": "mean_reversion_002",
                    "name": "Mean Reversion",
                    "description": "Statistical arbitrage using mean reversion patterns",
                    "status": "active",
                    "performance": {
                        "total_return": 8.7,
                        "win_rate": 72.1,
                        "avg_trade_duration": 1.8,
                        "max_drawdown": -3.2
                    },
                    "risk_metrics": {
                        "sharpe_ratio": 2.1,
                        "volatility": 8.9,
                        "beta": 0.65,
                        "var_95": -1.8
                    },
                    "created_at": datetime.now() - timedelta(days=45),
                    "updated_at": datetime.now()
                }
            ]
            
            return [StrategyResponse(**strategy) for strategy in strategies]
            
        except Exception as e:
            logger.error(f"Error getting strategies: {e}")
            return []
    
    async def create_strategy(self, user_id: str, request: StrategyRequest) -> StrategyResponse:
        """Create a new AI trading strategy"""
        try:
            strategy_id = f"{request.strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            strategy = {
                "strategy_id": strategy_id,
                "name": request.strategy_type.replace('_', ' ').title(),
                "description": f"AI-powered {request.strategy_type} strategy",
                "status": "initializing",
                "performance": {
                    "total_return": 0.0,
                    "win_rate": 0.0,
                    "avg_trade_duration": 0.0,
                    "max_drawdown": 0.0
                },
                "risk_metrics": {
                    "sharpe_ratio": 0.0,
                    "volatility": 0.0,
                    "beta": 0.0,
                    "var_95": 0.0
                },
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            
            self.active_strategies[strategy_id] = strategy
            
            return StrategyResponse(**strategy)
            
        except Exception as e:
            logger.error(f"Error creating strategy: {e}")
            raise
    
    async def get_market_predictions(self, assets: List[str], timeframe: str = "1d") -> List[MarketPrediction]:
        """Get AI market predictions for specified assets"""
        try:
            predictions = []
            
            for asset in assets:
                # Mock prediction logic
                prediction_value = np.random.uniform(-0.05, 0.05)  # -5% to +5%
                confidence = np.random.uniform(0.6, 0.95)
                
                factors = [
                    "Technical momentum",
                    "Market sentiment",
                    "Volume analysis",
                    "News sentiment"
                ]
                
                prediction = MarketPrediction(
                    asset=asset,
                    timeframe=timeframe,
                    prediction=prediction_value,
                    confidence=confidence,
                    factors=np.random.choice(factors, size=2, replace=False).tolist(),
                    timestamp=datetime.now()
                )
                
                predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting market predictions: {e}")
            return []
    
    async def get_trading_signals(self, strategy_id: str) -> List[TradingSignal]:
        """Get trading signals from a specific strategy"""
        try:
            # Mock signals for demonstration
            signals = [
                TradingSignal(
                    signal_id="signal_001",
                    strategy_id=strategy_id,
                    asset="BTC",
                    action="buy",
                    confidence=0.85,
                    price_target=45000.0,
                    stop_loss=42000.0,
                    reasoning="Strong momentum breakout with high volume",
                    timestamp=datetime.now() - timedelta(minutes=15)
                ),
                TradingSignal(
                    signal_id="signal_002",
                    strategy_id=strategy_id,
                    asset="ETH",
                    action="hold",
                    confidence=0.72,
                    price_target=None,
                    stop_loss=None,
                    reasoning="Consolidation phase, waiting for clear direction",
                    timestamp=datetime.now() - timedelta(minutes=30)
                )
            ]
            
            return signals
            
        except Exception as e:
            logger.error(f"Error getting trading signals: {e}")
            return []
    
    async def get_ai_insights(self, user_id: str) -> List[AIInsight]:
        """Get AI-generated market insights"""
        try:
            insights = [
                AIInsight(
                    insight_id="insight_001",
                    type="market_analysis",
                    title="Crypto Market Momentum Shift",
                    description="AI models detect a potential momentum shift in the crypto market based on technical and sentiment indicators.",
                    impact="high",
                    assets_affected=["BTC", "ETH", "ADA"],
                    timestamp=datetime.now() - timedelta(hours=2)
                ),
                AIInsight(
                    insight_id="insight_002",
                    type="risk_alert",
                    title="Increased Volatility Expected",
                    description="Market volatility models predict increased volatility in the next 24-48 hours.",
                    impact="medium",
                    assets_affected=["SPY", "QQQ", "TSLA"],
                    timestamp=datetime.now() - timedelta(hours=4)
                )
            ]
            
            return insights
            
        except Exception as e:
            logger.error(f"Error getting AI insights: {e}")
            return []
    
    async def optimize_portfolio(self, user_id: str, portfolio_data: Dict[str, Any]) -> PortfolioOptimization:
        """Generate AI-powered portfolio optimization recommendations"""
        try:
            # Mock optimization logic
            current_allocation = portfolio_data.get("allocation", {
                "BTC": 0.4,
                "ETH": 0.3,
                "STOCKS": 0.2,
                "CASH": 0.1
            })
            
            # AI-recommended allocation
            recommended_allocation = {
                "BTC": 0.35,
                "ETH": 0.25,
                "STOCKS": 0.25,
                "BONDS": 0.1,
                "CASH": 0.05
            }
            
            optimization = PortfolioOptimization(
                optimization_id=f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                current_allocation=current_allocation,
                recommended_allocation=recommended_allocation,
                expected_return=0.12,
                expected_risk=0.18,
                sharpe_ratio=1.67,
                reasoning="Rebalancing to reduce concentration risk and improve risk-adjusted returns",
                timestamp=datetime.now()
            )
            
            return optimization
            
        except Exception as e:
            logger.error(f"Error optimizing portfolio: {e}")
            raise
    
    async def analyze_sentiment(self, assets: List[str]) -> List[SentimentAnalysis]:
        """Analyze market sentiment for specified assets"""
        try:
            sentiment_analyses = []
            
            for asset in assets:
                # Mock sentiment analysis
                sentiment_score = np.random.uniform(-0.8, 0.8)
                
                if sentiment_score > 0.3:
                    sentiment_label = "bullish"
                elif sentiment_score < -0.3:
                    sentiment_label = "bearish"
                else:
                    sentiment_label = "neutral"
                
                analysis = SentimentAnalysis(
                    asset=asset,
                    sentiment_score=sentiment_score,
                    sentiment_label=sentiment_label,
                    confidence=np.random.uniform(0.7, 0.95),
                    sources=["Twitter", "Reddit", "News", "Forums"],
                    key_topics=["adoption", "regulation", "technology", "market_trends"],
                    timestamp=datetime.now()
                )
                
                sentiment_analyses.append(analysis)
            
            return sentiment_analyses
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return []
    
    async def start_strategy(self, strategy_id: str) -> bool:
        """Start an AI trading strategy"""
        try:
            if strategy_id in self.active_strategies:
                self.active_strategies[strategy_id]["status"] = "active"
                logger.info(f"Started strategy {strategy_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error starting strategy: {e}")
            return False
    
    async def pause_strategy(self, strategy_id: str) -> bool:
        """Pause an AI trading strategy"""
        try:
            if strategy_id in self.active_strategies:
                self.active_strategies[strategy_id]["status"] = "paused"
                logger.info(f"Paused strategy {strategy_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error pausing strategy: {e}")
            return False
    
    async def stop_strategy(self, strategy_id: str) -> bool:
        """Stop an AI trading strategy"""
        try:
            if strategy_id in self.active_strategies:
                self.active_strategies[strategy_id]["status"] = "stopped"
                logger.info(f"Stopped strategy {strategy_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error stopping strategy: {e}")
            return False