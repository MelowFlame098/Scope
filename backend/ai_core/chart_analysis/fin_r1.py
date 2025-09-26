# Fin-R1 Component
# Financial Reasoning and Strategy Layer for AI-Powered Trading Decisions

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from datetime import datetime, timedelta
import json
import openai
import os
from dotenv import load_dotenv

from .finvis_gpt import ChartFeatures, IndicatorData
from .kronos import KronosOutput, ForecastScenario, VolatilityForecast, TrendForecast, TrendDirection
from ...services.market_scraper import MarketSentiment, TradingSignal

load_dotenv()
logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of trading actions"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    WAIT = "wait"

class StrategyType(Enum):
    """Types of trading strategies"""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    CONTRARIAN = "contrarian"
    SCALPING = "scalping"
    SWING = "swing"
    POSITION = "position"

class RiskLevel(Enum):
    """Risk level classifications"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class MarketRegime(Enum):
    """Market regime classifications"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    TRENDING = "trending"

@dataclass
class TradingSignal:
    """Individual trading signal"""
    action: ActionType
    confidence: float  # 0-1 scale
    strength: float   # 0-1 scale
    timeframe: str
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    signal_source: str
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class StrategyRecommendation:
    """Strategy recommendation with detailed analysis"""
    strategy_type: StrategyType
    primary_action: ActionType
    confidence: float
    risk_level: RiskLevel
    expected_return: float
    max_drawdown: float
    holding_period: timedelta
    entry_conditions: List[str]
    exit_conditions: List[str]
    risk_management: Dict[str, Any]
    supporting_signals: List[TradingSignal]
    conflicting_signals: List[TradingSignal]
    market_context: str
    reasoning: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MarketAnalysis:
    """Comprehensive market analysis"""
    market_regime: MarketRegime
    trend_analysis: Dict[str, Any]
    volatility_analysis: Dict[str, Any]
    momentum_analysis: Dict[str, Any]
    support_resistance_analysis: Dict[str, Any]
    volume_analysis: Dict[str, Any]
    sentiment_indicators: Dict[str, float]
    key_levels: Dict[str, float]
    market_structure: Dict[str, Any]
    risk_factors: List[str]
    opportunities: List[str]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FinR1Output:
    """Complete Fin-R1 reasoning output"""
    primary_recommendation: StrategyRecommendation
    alternative_strategies: List[StrategyRecommendation]
    market_analysis: MarketAnalysis
    trading_signals: List[TradingSignal]
    market_sentiment: List[MarketSentiment]
    scraped_signals: List[TradingSignal]
    risk_assessment: Dict[str, Any]
    position_sizing: Dict[str, float]
    execution_plan: Dict[str, Any]
    monitoring_plan: Dict[str, Any]
    scenario_analysis: Dict[str, Any]
    confidence_score: float
    reasoning_chain: List[str]
    llm_insights: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class FinR1:
    """
    Fin-R1 - Advanced Financial Reasoning and Strategy Layer
    
    Sophisticated AI reasoning system that:
    - Receives structured data from FinVis-GPT and Kronos predictions
    - Applies prompt-based reasoning using standard and custom indicators
    - Generates comprehensive trading strategies and recommendations
    - Provides detailed rationale and confidence assessments
    - Integrates multiple analysis frameworks and risk management
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # LLM configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.model_name = self.config.get('model_name', 'gpt-4')
        self.use_llm_reasoning = self.config.get('use_llm_reasoning', True)
        self.max_tokens = self.config.get('max_tokens', 2000)
        self.temperature = self.config.get('temperature', 0.3)
        
        # Analysis configuration
        self.analysis_config = {
            'timeframes': self.config.get('timeframes', ['1D', '1W', '1M']),
            'risk_tolerance': self.config.get('risk_tolerance', 'medium'),
            'strategy_preference': self.config.get('strategy_preference', 'balanced'),
            'min_confidence_threshold': self.config.get('min_confidence_threshold', 0.6),
            'max_risk_per_trade': self.config.get('max_risk_per_trade', 0.02),
            'position_sizing_method': self.config.get('position_sizing_method', 'kelly')
        }
        
        # Strategy weights and preferences
        self.strategy_weights = {
            StrategyType.MOMENTUM: self.config.get('momentum_weight', 0.2),
            StrategyType.MEAN_REVERSION: self.config.get('mean_reversion_weight', 0.15),
            StrategyType.BREAKOUT: self.config.get('breakout_weight', 0.2),
            StrategyType.TREND_FOLLOWING: self.config.get('trend_following_weight', 0.25),
            StrategyType.CONTRARIAN: self.config.get('contrarian_weight', 0.1),
            StrategyType.SWING: self.config.get('swing_weight', 0.1)
        }
        
        # Risk management parameters
        self.risk_params = {
            'max_portfolio_risk': self.config.get('max_portfolio_risk', 0.1),
            'correlation_threshold': self.config.get('correlation_threshold', 0.7),
            'volatility_adjustment': self.config.get('volatility_adjustment', True),
            'dynamic_position_sizing': self.config.get('dynamic_position_sizing', True)
        }
        
        # Initialize OpenAI if available
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
        
        logger.info("Fin-R1 initialized with config: %s", self.config)
    
    async def generate_recommendations(self, chart_features: ChartFeatures, 
                                     kronos_output: KronosOutput,
                                     custom_indicators: Optional[Dict[str, List[float]]] = None,
                                     market_sentiment: Optional[List[MarketSentiment]] = None,
                                     scraped_signals: Optional[List[TradingSignal]] = None) -> FinR1Output:
        """
        Main reasoning method to generate trading recommendations.
        
        Args:
            chart_features: Extracted features from FinVis-GPT
            kronos_output: Forecasting results from Kronos
            custom_indicators: Additional custom indicator data
            
        Returns:
            FinR1Output with comprehensive recommendations
        """
        try:
            logger.info("Starting Fin-R1 financial reasoning")
            
            # Step 1: Perform comprehensive market analysis
            market_analysis = await self._analyze_market_conditions(chart_features, kronos_output)
            
            # Step 2: Generate trading signals from multiple sources
            trading_signals = await self._generate_trading_signals(chart_features, kronos_output, custom_indicators)
            
            # Step 3: Apply LLM-based reasoning if available
            llm_insights = await self._apply_llm_reasoning(chart_features, kronos_output, market_analysis, market_sentiment, scraped_signals)
            
            # Step 4: Synthesize strategy recommendations
            strategies = await self._synthesize_strategies(market_analysis, trading_signals, llm_insights)
            
            # Step 5: Select primary recommendation
            primary_recommendation = await self._select_primary_strategy(strategies, market_analysis)
            
            # Step 6: Perform risk assessment
            risk_assessment = await self._assess_risks(primary_recommendation, market_analysis, kronos_output)
            
            # Step 7: Calculate position sizing
            position_sizing = await self._calculate_position_sizing(primary_recommendation, risk_assessment)
            
            # Step 8: Create execution plan
            execution_plan = await self._create_execution_plan(primary_recommendation, market_analysis)
            
            # Step 9: Design monitoring plan
            monitoring_plan = await self._create_monitoring_plan(primary_recommendation, market_analysis)
            
            # Step 10: Perform scenario analysis
            scenario_analysis = await self._perform_scenario_analysis(primary_recommendation, kronos_output)
            
            # Step 11: Generate reasoning chain
            reasoning_chain = await self._generate_reasoning_chain(market_analysis, primary_recommendation, llm_insights)
            
            # Step 12: Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(
                primary_recommendation, market_analysis, trading_signals, llm_insights
            )
            
            # Compile results
            output = FinR1Output(
                primary_recommendation=primary_recommendation,
                alternative_strategies=strategies[1:] if len(strategies) > 1 else [],
                market_analysis=market_analysis,
                trading_signals=trading_signals,
                market_sentiment=market_sentiment or [],
                scraped_signals=scraped_signals or [],
                risk_assessment=risk_assessment,
                position_sizing=position_sizing,
                execution_plan=execution_plan,
                monitoring_plan=monitoring_plan,
                scenario_analysis=scenario_analysis,
                confidence_score=confidence_score,
                reasoning_chain=reasoning_chain,
                llm_insights=llm_insights,
                metadata={
                    'analysis_timestamp': datetime.now().isoformat(),
                    'config': self.analysis_config,
                    'strategy_weights': self.strategy_weights,
                    'signals_count': len(trading_signals),
                    'sentiment_count': len(market_sentiment) if market_sentiment else 0,
                    'scraped_signals_count': len(scraped_signals) if scraped_signals else 0
                }
            )
            
            logger.info(f"Fin-R1 reasoning completed with confidence: {confidence_score:.2f}")
            return output
            
        except Exception as e:
            logger.error(f"Fin-R1 reasoning failed: {e}")
            return self._create_empty_output()
    
    async def _analyze_market_conditions(self, chart_features: ChartFeatures, 
                                       kronos_output: KronosOutput) -> MarketAnalysis:
        """Analyze current market conditions"""
        try:
            logger.info("Analyzing market conditions")
            
            # Determine market regime
            market_regime = await self._determine_market_regime(chart_features, kronos_output)
            
            # Analyze trend
            trend_analysis = await self._analyze_trend(chart_features, kronos_output.trend_forecast)
            
            # Analyze volatility
            volatility_analysis = await self._analyze_volatility(kronos_output.volatility_forecast)
            
            # Analyze momentum
            momentum_analysis = await self._analyze_momentum(chart_features)
            
            # Analyze support and resistance
            sr_analysis = await self._analyze_support_resistance(chart_features)
            
            # Analyze volume
            volume_analysis = await self._analyze_volume(chart_features)
            
            # Calculate sentiment indicators
            sentiment_indicators = await self._calculate_sentiment_indicators(chart_features, kronos_output)
            
            # Identify key levels
            key_levels = await self._identify_key_levels(chart_features, kronos_output)
            
            # Analyze market structure
            market_structure = await self._analyze_market_structure(chart_features)
            
            # Identify risk factors and opportunities
            risk_factors, opportunities = await self._identify_risks_and_opportunities(
                chart_features, kronos_output, market_regime
            )
            
            # Calculate analysis confidence
            analysis_confidence = self._calculate_analysis_confidence(
                trend_analysis, volatility_analysis, momentum_analysis
            )
            
            return MarketAnalysis(
                market_regime=market_regime,
                trend_analysis=trend_analysis,
                volatility_analysis=volatility_analysis,
                momentum_analysis=momentum_analysis,
                support_resistance_analysis=sr_analysis,
                volume_analysis=volume_analysis,
                sentiment_indicators=sentiment_indicators,
                key_levels=key_levels,
                market_structure=market_structure,
                risk_factors=risk_factors,
                opportunities=opportunities,
                confidence=analysis_confidence,
                metadata={
                    'analysis_method': 'comprehensive_technical_analysis',
                    'data_quality': chart_features.extraction_confidence
                }
            )
            
        except Exception as e:
            logger.error(f"Market analysis failed: {e}")
            return self._create_empty_market_analysis()
    
    async def _determine_market_regime(self, chart_features: ChartFeatures, 
                                     kronos_output: KronosOutput) -> MarketRegime:
        """Determine current market regime"""
        try:
            # Analyze trend direction and strength
            trend_forecast = kronos_output.trend_forecast
            volatility_forecast = kronos_output.volatility_forecast
            
            # Check trend strength and direction
            if trend_forecast.trend_direction in [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]:
                if trend_forecast.trend_strength > 0.7:
                    return MarketRegime.BULL_MARKET
                elif volatility_forecast.volatility_regime == "high":
                    return MarketRegime.VOLATILE
                else:
                    return MarketRegime.TRENDING
            
            elif trend_forecast.trend_direction in [TrendDirection.STRONG_BEARISH, TrendDirection.BEARISH]:
                if trend_forecast.trend_strength > 0.7:
                    return MarketRegime.BEAR_MARKET
                elif volatility_forecast.volatility_regime == "high":
                    return MarketRegime.VOLATILE
                else:
                    return MarketRegime.TRENDING
            
            else:  # Neutral trend
                if volatility_forecast.volatility_regime == "high":
                    return MarketRegime.VOLATILE
                else:
                    return MarketRegime.SIDEWAYS
            
        except Exception as e:
            logger.error(f"Market regime determination failed: {e}")
            return MarketRegime.SIDEWAYS
    
    async def _analyze_trend(self, chart_features: ChartFeatures, 
                           trend_forecast: TrendForecast) -> Dict[str, Any]:
        """Analyze trend characteristics"""
        try:
            if not chart_features.ohlc_data:
                return {}
            
            # Get recent price data
            prices = [ohlc.close for ohlc in chart_features.ohlc_data]
            
            # Calculate trend metrics
            short_ma = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
            long_ma = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
            current_price = prices[-1]
            
            # Trend slope
            if len(prices) >= 10:
                trend_slope = np.polyfit(range(10), prices[-10:], 1)[0]
            else:
                trend_slope = 0
            
            return {
                'direction': trend_forecast.trend_direction.value,
                'strength': trend_forecast.trend_strength,
                'slope': trend_slope,
                'short_ma': short_ma,
                'long_ma': long_ma,
                'price_vs_short_ma': (current_price - short_ma) / short_ma,
                'price_vs_long_ma': (current_price - long_ma) / long_ma,
                'ma_alignment': 'bullish' if short_ma > long_ma else 'bearish',
                'trend_duration': trend_forecast.trend_duration.days,
                'reversal_probability': trend_forecast.reversal_probability,
                'continuation_probability': trend_forecast.continuation_probability
            }
            
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return {}
    
    async def _analyze_volatility(self, volatility_forecast: VolatilityForecast) -> Dict[str, Any]:
        """Analyze volatility characteristics"""
        try:
            return {
                'regime': volatility_forecast.volatility_regime,
                'current_level': volatility_forecast.predicted_volatility[0] if volatility_forecast.predicted_volatility else 0,
                'percentiles': volatility_forecast.volatility_percentiles,
                'expected_range': volatility_forecast.expected_range,
                'breakout_probability': volatility_forecast.breakout_probability,
                'mean_reversion_probability': volatility_forecast.mean_reversion_probability,
                'volatility_trend': 'increasing' if len(volatility_forecast.predicted_volatility) > 1 and 
                                  volatility_forecast.predicted_volatility[-1] > volatility_forecast.predicted_volatility[0] else 'decreasing'
            }
            
        except Exception as e:
            logger.error(f"Volatility analysis failed: {e}")
            return {}
    
    async def _analyze_momentum(self, chart_features: ChartFeatures) -> Dict[str, Any]:
        """Analyze momentum indicators"""
        try:
            if not chart_features.ohlc_data:
                return {}
            
            prices = [ohlc.close for ohlc in chart_features.ohlc_data]
            
            # Calculate momentum metrics
            momentum_metrics = {}
            
            # Price momentum (rate of change)
            if len(prices) >= 10:
                momentum_10 = (prices[-1] - prices[-10]) / prices[-10]
                momentum_metrics['momentum_10d'] = momentum_10
            
            if len(prices) >= 20:
                momentum_20 = (prices[-1] - prices[-20]) / prices[-20]
                momentum_metrics['momentum_20d'] = momentum_20
            
            # RSI-like momentum
            if len(prices) >= 14:
                price_changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
                gains = [change if change > 0 else 0 for change in price_changes[-14:]]
                losses = [-change if change < 0 else 0 for change in price_changes[-14:]]
                
                avg_gain = np.mean(gains) if gains else 0
                avg_loss = np.mean(losses) if losses else 0
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    momentum_metrics['rsi'] = rsi
                    momentum_metrics['rsi_signal'] = 'overbought' if rsi > 70 else ('oversold' if rsi < 30 else 'neutral')
            
            # Volume-price momentum
            if chart_features.volume_data and len(chart_features.volume_data) >= 10:
                recent_volume = np.mean(chart_features.volume_data[-10:])
                avg_volume = np.mean(chart_features.volume_data)
                momentum_metrics['volume_momentum'] = recent_volume / avg_volume if avg_volume > 0 else 1
            
            return momentum_metrics
            
        except Exception as e:
            logger.error(f"Momentum analysis failed: {e}")
            return {}
    
    async def _analyze_support_resistance(self, chart_features: ChartFeatures) -> Dict[str, Any]:
        """Analyze support and resistance levels"""
        try:
            current_price = chart_features.ohlc_data[-1].close if chart_features.ohlc_data else 0
            
            # Analyze support levels
            support_analysis = {
                'levels': chart_features.support_levels,
                'nearest_support': None,
                'support_strength': 'weak',
                'distance_to_support': None
            }
            
            if chart_features.support_levels:
                # Find nearest support below current price
                supports_below = [s for s in chart_features.support_levels if s < current_price]
                if supports_below:
                    nearest_support = max(supports_below)
                    support_analysis['nearest_support'] = nearest_support
                    support_analysis['distance_to_support'] = (current_price - nearest_support) / current_price
                    
                    # Assess support strength based on distance
                    distance_pct = support_analysis['distance_to_support']
                    if distance_pct < 0.02:
                        support_analysis['support_strength'] = 'very_strong'
                    elif distance_pct < 0.05:
                        support_analysis['support_strength'] = 'strong'
                    elif distance_pct < 0.1:
                        support_analysis['support_strength'] = 'moderate'
            
            # Analyze resistance levels
            resistance_analysis = {
                'levels': chart_features.resistance_levels,
                'nearest_resistance': None,
                'resistance_strength': 'weak',
                'distance_to_resistance': None
            }
            
            if chart_features.resistance_levels:
                # Find nearest resistance above current price
                resistances_above = [r for r in chart_features.resistance_levels if r > current_price]
                if resistances_above:
                    nearest_resistance = min(resistances_above)
                    resistance_analysis['nearest_resistance'] = nearest_resistance
                    resistance_analysis['distance_to_resistance'] = (nearest_resistance - current_price) / current_price
                    
                    # Assess resistance strength
                    distance_pct = resistance_analysis['distance_to_resistance']
                    if distance_pct < 0.02:
                        resistance_analysis['resistance_strength'] = 'very_strong'
                    elif distance_pct < 0.05:
                        resistance_analysis['resistance_strength'] = 'strong'
                    elif distance_pct < 0.1:
                        resistance_analysis['resistance_strength'] = 'moderate'
            
            return {
                'support': support_analysis,
                'resistance': resistance_analysis,
                'current_price': current_price,
                'price_position': self._calculate_price_position(current_price, chart_features)
            }
            
        except Exception as e:
            logger.error(f"Support/resistance analysis failed: {e}")
            return {}
    
    def _calculate_price_position(self, current_price: float, chart_features: ChartFeatures) -> str:
        """Calculate price position relative to range"""
        try:
            if not chart_features.ohlc_data:
                return 'unknown'
            
            # Get price range from recent data
            recent_highs = [ohlc.high for ohlc in chart_features.ohlc_data[-50:]]
            recent_lows = [ohlc.low for ohlc in chart_features.ohlc_data[-50:]]
            
            range_high = max(recent_highs)
            range_low = min(recent_lows)
            
            if range_high == range_low:
                return 'neutral'
            
            position = (current_price - range_low) / (range_high - range_low)
            
            if position > 0.8:
                return 'upper_range'
            elif position > 0.6:
                return 'upper_middle'
            elif position > 0.4:
                return 'middle'
            elif position > 0.2:
                return 'lower_middle'
            else:
                return 'lower_range'
            
        except Exception as e:
            logger.error(f"Price position calculation failed: {e}")
            return 'unknown'
    
    async def _analyze_volume(self, chart_features: ChartFeatures) -> Dict[str, Any]:
        """Analyze volume characteristics"""
        try:
            if not chart_features.volume_data:
                return {'status': 'no_volume_data'}
            
            volume_data = chart_features.volume_data
            
            # Calculate volume metrics
            current_volume = volume_data[-1]
            avg_volume_20 = np.mean(volume_data[-20:]) if len(volume_data) >= 20 else np.mean(volume_data)
            avg_volume_50 = np.mean(volume_data[-50:]) if len(volume_data) >= 50 else np.mean(volume_data)
            
            volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            volume_ratio_50 = current_volume / avg_volume_50 if avg_volume_50 > 0 else 1
            
            # Volume trend
            if len(volume_data) >= 10:
                recent_avg = np.mean(volume_data[-10:])
                previous_avg = np.mean(volume_data[-20:-10]) if len(volume_data) >= 20 else recent_avg
                volume_trend = 'increasing' if recent_avg > previous_avg else 'decreasing'
            else:
                volume_trend = 'stable'
            
            # Volume signal
            if volume_ratio_20 > 2.0:
                volume_signal = 'very_high'
            elif volume_ratio_20 > 1.5:
                volume_signal = 'high'
            elif volume_ratio_20 > 0.8:
                volume_signal = 'normal'
            else:
                volume_signal = 'low'
            
            return {
                'current_volume': current_volume,
                'avg_volume_20': avg_volume_20,
                'avg_volume_50': avg_volume_50,
                'volume_ratio_20': volume_ratio_20,
                'volume_ratio_50': volume_ratio_50,
                'volume_trend': volume_trend,
                'volume_signal': volume_signal,
                'volume_confirmation': volume_ratio_20 > 1.2  # Volume confirms price movement
            }
            
        except Exception as e:
            logger.error(f"Volume analysis failed: {e}")
            return {}
    
    async def _calculate_sentiment_indicators(self, chart_features: ChartFeatures, 
                                            kronos_output: KronosOutput) -> Dict[str, float]:
        """Calculate sentiment indicators"""
        try:
            sentiment = {}
            
            # Trend sentiment
            trend_direction = kronos_output.trend_forecast.trend_direction
            if trend_direction in [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]:
                sentiment['trend_sentiment'] = 0.7 if trend_direction == TrendDirection.BULLISH else 0.9
            elif trend_direction in [TrendDirection.STRONG_BEARISH, TrendDirection.BEARISH]:
                sentiment['trend_sentiment'] = 0.3 if trend_direction == TrendDirection.BEARISH else 0.1
            else:
                sentiment['trend_sentiment'] = 0.5
            
            # Volatility sentiment (high volatility = negative sentiment)
            vol_regime = kronos_output.volatility_forecast.volatility_regime
            if vol_regime == 'low':
                sentiment['volatility_sentiment'] = 0.7
            elif vol_regime == 'medium':
                sentiment['volatility_sentiment'] = 0.5
            else:
                sentiment['volatility_sentiment'] = 0.3
            
            # Price position sentiment
            if chart_features.ohlc_data:
                current_price = chart_features.ohlc_data[-1].close
                price_range = chart_features.price_range
                
                if price_range[1] != price_range[0]:
                    price_position = (current_price - price_range[0]) / (price_range[1] - price_range[0])
                    sentiment['price_position_sentiment'] = price_position
                else:
                    sentiment['price_position_sentiment'] = 0.5
            
            # Overall sentiment (weighted average)
            weights = {'trend_sentiment': 0.5, 'volatility_sentiment': 0.3, 'price_position_sentiment': 0.2}
            overall_sentiment = sum(sentiment[key] * weights[key] for key in sentiment if key in weights)
            sentiment['overall_sentiment'] = overall_sentiment
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Sentiment calculation failed: {e}")
            return {'overall_sentiment': 0.5}
    
    async def _identify_key_levels(self, chart_features: ChartFeatures, 
                                 kronos_output: KronosOutput) -> Dict[str, float]:
        """Identify key price levels"""
        try:
            key_levels = {}
            
            if chart_features.ohlc_data:
                current_price = chart_features.ohlc_data[-1].close
                key_levels['current_price'] = current_price
                
                # Support and resistance levels
                if chart_features.support_levels:
                    key_levels['key_support'] = max([s for s in chart_features.support_levels if s < current_price], default=current_price * 0.95)
                
                if chart_features.resistance_levels:
                    key_levels['key_resistance'] = min([r for r in chart_features.resistance_levels if r > current_price], default=current_price * 1.05)
                
                # Price targets from forecasts
                if kronos_output.price_forecasts:
                    primary_forecast = kronos_output.price_forecasts[0]
                    key_levels['price_target'] = primary_forecast.price_targets[1]  # Middle target
                    key_levels['upside_target'] = primary_forecast.price_targets[2]  # Upper target
                    key_levels['downside_target'] = primary_forecast.price_targets[0]  # Lower target
                
                # Moving averages (if available in indicators)
                for indicator in chart_features.indicators:
                    if 'SMA' in indicator.name and indicator.values:
                        key_levels[f'{indicator.name}_level'] = indicator.values[-1]
            
            return key_levels
            
        except Exception as e:
            logger.error(f"Key levels identification failed: {e}")
            return {}
    
    async def _analyze_market_structure(self, chart_features: ChartFeatures) -> Dict[str, Any]:
        """Analyze market structure"""
        try:
            if not chart_features.ohlc_data:
                return {}
            
            # Analyze higher highs/lower lows pattern
            highs = [ohlc.high for ohlc in chart_features.ohlc_data[-20:]]
            lows = [ohlc.low for ohlc in chart_features.ohlc_data[-20:]]
            
            # Find recent peaks and troughs
            structure = {
                'trend_structure': 'unknown',
                'swing_highs': [],
                'swing_lows': [],
                'structure_strength': 'weak'
            }
            
            # Simple structure analysis
            if len(highs) >= 10:
                recent_high = max(highs[-5:])
                previous_high = max(highs[-10:-5])
                
                recent_low = min(lows[-5:])
                previous_low = min(lows[-10:-5])
                
                if recent_high > previous_high and recent_low > previous_low:
                    structure['trend_structure'] = 'higher_highs_higher_lows'
                    structure['structure_strength'] = 'strong'
                elif recent_high < previous_high and recent_low < previous_low:
                    structure['trend_structure'] = 'lower_highs_lower_lows'
                    structure['structure_strength'] = 'strong'
                else:
                    structure['trend_structure'] = 'mixed'
                    structure['structure_strength'] = 'weak'
            
            return structure
            
        except Exception as e:
            logger.error(f"Market structure analysis failed: {e}")
            return {}
    
    async def _identify_risks_and_opportunities(self, chart_features: ChartFeatures, 
                                              kronos_output: KronosOutput, 
                                              market_regime: MarketRegime) -> Tuple[List[str], List[str]]:
        """Identify risk factors and opportunities"""
        try:
            risk_factors = []
            opportunities = []
            
            # Risk factors based on market regime
            if market_regime == MarketRegime.VOLATILE:
                risk_factors.extend(['High volatility', 'Unpredictable price swings', 'Increased slippage'])
            elif market_regime == MarketRegime.BEAR_MARKET:
                risk_factors.extend(['Downtrend pressure', 'Negative sentiment', 'Support level breaks'])
            
            # Risk factors from volatility
            vol_forecast = kronos_output.volatility_forecast
            if vol_forecast.breakout_probability > 0.7:
                risk_factors.append('High breakout probability - potential for large moves')
            
            # Risk factors from trend
            trend_forecast = kronos_output.trend_forecast
            if trend_forecast.reversal_probability > 0.6:
                risk_factors.append('High trend reversal probability')
            
            # Opportunities based on market conditions
            if market_regime == MarketRegime.BULL_MARKET:
                opportunities.extend(['Uptrend momentum', 'Positive sentiment', 'Breakout potential'])
            elif market_regime == MarketRegime.SIDEWAYS:
                opportunities.extend(['Range trading opportunities', 'Mean reversion plays'])
            
            # Opportunities from support/resistance
            if chart_features.support_levels and chart_features.resistance_levels:
                opportunities.append('Clear support/resistance levels for trading')
            
            # Opportunities from volume
            if chart_features.volume_data:
                recent_volume = np.mean(chart_features.volume_data[-5:])
                avg_volume = np.mean(chart_features.volume_data)
                if recent_volume > avg_volume * 1.5:
                    opportunities.append('Increased volume suggests strong interest')
            
            return risk_factors, opportunities
            
        except Exception as e:
            logger.error(f"Risk/opportunity identification failed: {e}")
            return [], []
    
    def _calculate_analysis_confidence(self, trend_analysis: Dict, 
                                     volatility_analysis: Dict, 
                                     momentum_analysis: Dict) -> float:
        """Calculate confidence in market analysis"""
        try:
            confidence_factors = []
            
            # Trend analysis confidence
            if trend_analysis and 'strength' in trend_analysis:
                confidence_factors.append(trend_analysis['strength'] * 0.4)
            
            # Volatility analysis confidence
            if volatility_analysis and 'regime' in volatility_analysis:
                regime_confidence = 0.8 if volatility_analysis['regime'] != 'unknown' else 0.3
                confidence_factors.append(regime_confidence * 0.3)
            
            # Momentum analysis confidence
            if momentum_analysis and 'rsi' in momentum_analysis:
                confidence_factors.append(0.7 * 0.3)  # RSI available
            elif momentum_analysis:
                confidence_factors.append(0.5 * 0.3)  # Some momentum data
            
            return min(sum(confidence_factors), 1.0) if confidence_factors else 0.5
            
        except Exception as e:
            logger.error(f"Analysis confidence calculation failed: {e}")
            return 0.5
    
    async def _generate_trading_signals(self, chart_features: ChartFeatures, 
                                      kronos_output: KronosOutput,
                                      custom_indicators: Optional[Dict[str, List[float]]]) -> List[TradingSignal]:
        """Generate trading signals from multiple sources"""
        try:
            logger.info("Generating trading signals")
            
            signals = []
            
            # Trend-based signals
            trend_signals = await self._generate_trend_signals(kronos_output.trend_forecast)
            signals.extend(trend_signals)
            
            # Momentum signals
            momentum_signals = await self._generate_momentum_signals(chart_features)
            signals.extend(momentum_signals)
            
            # Support/resistance signals
            sr_signals = await self._generate_support_resistance_signals(chart_features)
            signals.extend(sr_signals)
            
            # Volume signals
            volume_signals = await self._generate_volume_signals(chart_features)
            signals.extend(volume_signals)
            
            # Volatility signals
            vol_signals = await self._generate_volatility_signals(kronos_output.volatility_forecast)
            signals.extend(vol_signals)
            
            # Custom indicator signals
            if custom_indicators:
                custom_signals = await self._generate_custom_indicator_signals(custom_indicators)
                signals.extend(custom_signals)
            
            # Price forecast signals
            forecast_signals = await self._generate_forecast_signals(kronos_output.price_forecasts)
            signals.extend(forecast_signals)
            
            return signals
            
        except Exception as e:
            logger.error(f"Trading signal generation failed: {e}")
            return []
    
    async def _generate_trend_signals(self, trend_forecast: TrendForecast) -> List[TradingSignal]:
        """Generate signals based on trend analysis"""
        try:
            signals = []
            
            # Primary trend signal
            if trend_forecast.trend_direction == TrendDirection.STRONG_BULLISH:
                signal = TradingSignal(
                    action=ActionType.STRONG_BUY,
                    confidence=trend_forecast.trend_strength,
                    strength=0.9,
                    timeframe="1W",
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                    risk_reward_ratio=None,
                    signal_source="trend_analysis",
                    reasoning=f"Strong bullish trend with {trend_forecast.trend_strength:.2f} strength"
                )
                signals.append(signal)
            
            elif trend_forecast.trend_direction == TrendDirection.BULLISH:
                signal = TradingSignal(
                    action=ActionType.BUY,
                    confidence=trend_forecast.trend_strength,
                    strength=0.7,
                    timeframe="1W",
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                    risk_reward_ratio=None,
                    signal_source="trend_analysis",
                    reasoning=f"Bullish trend with {trend_forecast.trend_strength:.2f} strength"
                )
                signals.append(signal)
            
            elif trend_forecast.trend_direction == TrendDirection.BEARISH:
                signal = TradingSignal(
                    action=ActionType.SELL,
                    confidence=trend_forecast.trend_strength,
                    strength=0.7,
                    timeframe="1W",
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                    risk_reward_ratio=None,
                    signal_source="trend_analysis",
                    reasoning=f"Bearish trend with {trend_forecast.trend_strength:.2f} strength"
                )
                signals.append(signal)
            
            elif trend_forecast.trend_direction == TrendDirection.STRONG_BEARISH:
                signal = TradingSignal(
                    action=ActionType.STRONG_SELL,
                    confidence=trend_forecast.trend_strength,
                    strength=0.9,
                    timeframe="1W",
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                    risk_reward_ratio=None,
                    signal_source="trend_analysis",
                    reasoning=f"Strong bearish trend with {trend_forecast.trend_strength:.2f} strength"
                )
                signals.append(signal)
            
            # Trend reversal signal
            if trend_forecast.reversal_probability > 0.7:
                reversal_action = ActionType.SELL if trend_forecast.trend_direction in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH] else ActionType.BUY
                signal = TradingSignal(
                    action=reversal_action,
                    confidence=trend_forecast.reversal_probability,
                    strength=0.6,
                    timeframe="1D",
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                    risk_reward_ratio=None,
                    signal_source="trend_reversal",
                    reasoning=f"High reversal probability: {trend_forecast.reversal_probability:.2f}"
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Trend signal generation failed: {e}")
            return []
    
    async def _generate_momentum_signals(self, chart_features: ChartFeatures) -> List[TradingSignal]:
        """Generate signals based on momentum analysis"""
        try:
            signals = []
            
            if not chart_features.ohlc_data:
                return signals
            
            # Calculate basic momentum
            prices = [ohlc.close for ohlc in chart_features.ohlc_data]
            
            if len(prices) >= 10:
                momentum_10 = (prices[-1] - prices[-10]) / prices[-10]
                
                if momentum_10 > 0.05:  # 5% positive momentum
                    signal = TradingSignal(
                        action=ActionType.BUY,
                        confidence=min(abs(momentum_10) * 10, 1.0),
                        strength=0.6,
                        timeframe="1D",
                        entry_price=None,
                        stop_loss=None,
                        take_profit=None,
                        risk_reward_ratio=None,
                        signal_source="momentum",
                        reasoning=f"Positive 10-day momentum: {momentum_10:.2%}"
                    )
                    signals.append(signal)
                
                elif momentum_10 < -0.05:  # 5% negative momentum
                    signal = TradingSignal(
                        action=ActionType.SELL,
                        confidence=min(abs(momentum_10) * 10, 1.0),
                        strength=0.6,
                        timeframe="1D",
                        entry_price=None,
                        stop_loss=None,
                        take_profit=None,
                        risk_reward_ratio=None,
                        signal_source="momentum",
                        reasoning=f"Negative 10-day momentum: {momentum_10:.2%}"
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Momentum signal generation failed: {e}")
            return []
    
    async def _generate_support_resistance_signals(self, chart_features: ChartFeatures) -> List[TradingSignal]:
        """Generate signals based on support/resistance levels"""
        try:
            signals = []
            
            if not chart_features.ohlc_data:
                return signals
            
            current_price = chart_features.ohlc_data[-1].close
            
            # Support bounce signal
            if chart_features.support_levels:
                nearest_support = max([s for s in chart_features.support_levels if s < current_price], default=None)
                if nearest_support:
                    distance_to_support = (current_price - nearest_support) / current_price
                    
                    if distance_to_support < 0.02:  # Within 2% of support
                        signal = TradingSignal(
                            action=ActionType.BUY,
                            confidence=0.7,
                            strength=0.8,
                            timeframe="1D",
                            entry_price=current_price,
                            stop_loss=nearest_support * 0.98,
                            take_profit=current_price * 1.05,
                            risk_reward_ratio=2.5,
                            signal_source="support_level",
                            reasoning=f"Price near support level at {nearest_support:.2f}"
                        )
                        signals.append(signal)
            
            # Resistance rejection signal
            if chart_features.resistance_levels:
                nearest_resistance = min([r for r in chart_features.resistance_levels if r > current_price], default=None)
                if nearest_resistance:
                    distance_to_resistance = (nearest_resistance - current_price) / current_price
                    
                    if distance_to_resistance < 0.02:  # Within 2% of resistance
                        signal = TradingSignal(
                            action=ActionType.SELL,
                            confidence=0.7,
                            strength=0.8,
                            timeframe="1D",
                            entry_price=current_price,
                            stop_loss=nearest_resistance * 1.02,
                            take_profit=current_price * 0.95,
                            risk_reward_ratio=2.5,
                            signal_source="resistance_level",
                            reasoning=f"Price near resistance level at {nearest_resistance:.2f}"
                        )
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Support/resistance signal generation failed: {e}")
            return []
    
    async def _generate_volume_signals(self, chart_features: ChartFeatures) -> List[TradingSignal]:
        """Generate signals based on volume analysis"""
        try:
            signals = []
            
            if not chart_features.volume_data or len(chart_features.volume_data) < 20:
                return signals
            
            current_volume = chart_features.volume_data[-1]
            avg_volume = np.mean(chart_features.volume_data[-20:])
            
            # High volume signal
            if current_volume > avg_volume * 2:
                # Determine direction based on price movement
                if len(chart_features.ohlc_data) >= 2:
                    price_change = (chart_features.ohlc_data[-1].close - chart_features.ohlc_data[-2].close) / chart_features.ohlc_data[-2].close
                    
                    if price_change > 0:
                        action = ActionType.BUY
                        reasoning = "High volume with positive price movement"
                    else:
                        action = ActionType.SELL
                        reasoning = "High volume with negative price movement"
                    
                    signal = TradingSignal(
                        action=action,
                        confidence=0.6,
                        strength=0.7,
                        timeframe="1D",
                        entry_price=None,
                        stop_loss=None,
                        take_profit=None,
                        risk_reward_ratio=None,
                        signal_source="volume",
                        reasoning=reasoning
                    )
                    signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Volume signal generation failed: {e}")
            return []
    
    async def _generate_volatility_signals(self, volatility_forecast: VolatilityForecast) -> List[TradingSignal]:
        """Generate signals based on volatility analysis"""
        try:
            signals = []
            
            # Breakout signal based on volatility
            if volatility_forecast.breakout_probability > 0.7:
                signal = TradingSignal(
                    action=ActionType.WAIT,  # Wait for direction confirmation
                    confidence=volatility_forecast.breakout_probability,
                    strength=0.8,
                    timeframe="1D",
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                    risk_reward_ratio=None,
                    signal_source="volatility_breakout",
                    reasoning=f"High breakout probability: {volatility_forecast.breakout_probability:.2f}"
                )
                signals.append(signal)
            
            # Mean reversion signal
            if volatility_forecast.mean_reversion_probability > 0.7:
                signal = TradingSignal(
                    action=ActionType.HOLD,  # Hold for mean reversion
                    confidence=volatility_forecast.mean_reversion_probability,
                    strength=0.6,
                    timeframe="1W",
                    entry_price=None,
                    stop_loss=None,
                    take_profit=None,
                    risk_reward_ratio=None,
                    signal_source="volatility_mean_reversion",
                    reasoning=f"High mean reversion probability: {volatility_forecast.mean_reversion_probability:.2f}"
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Volatility signal generation failed: {e}")
            return []
    
    async def _generate_custom_indicator_signals(self, custom_indicators: Dict[str, List[float]]) -> List[TradingSignal]:
        """Generate signals from custom indicators"""
        try:
            signals = []
            
            for indicator_name, values in custom_indicators.items():
                if len(values) >= 10:
                    # Simple trend analysis of custom indicator
                    recent_trend = np.polyfit(range(5), values[-5:], 1)[0]
                    
                    if recent_trend > 0:
                        signal = TradingSignal(
                            action=ActionType.BUY,
                            confidence=0.5,
                            strength=0.5,
                            timeframe="1D",
                            entry_price=None,
                            stop_loss=None,
                            take_profit=None,
                            risk_reward_ratio=None,
                            signal_source=f"custom_{indicator_name}",
                            reasoning=f"Positive trend in custom indicator {indicator_name}"
                        )
                        signals.append(signal)
                    
                    elif recent_trend < 0:
                        signal = TradingSignal(
                            action=ActionType.SELL,
                            confidence=0.5,
                            strength=0.5,
                            timeframe="1D",
                            entry_price=None,
                            stop_loss=None,
                            take_profit=None,
                            risk_reward_ratio=None,
                            signal_source=f"custom_{indicator_name}",
                            reasoning=f"Negative trend in custom indicator {indicator_name}"
                        )
                        signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Custom indicator signal generation failed: {e}")
            return []
    
    async def _generate_forecast_signals(self, price_forecasts: List[ForecastScenario]) -> List[TradingSignal]:
        """Generate signals based on price forecasts"""
        try:
            signals = []
            
            if not price_forecasts:
                return signals
            
            # Use primary forecast
            primary_forecast = price_forecasts[0]
            
            # Determine action based on forecast targets
            lower_target, middle_target, upper_target = primary_forecast.price_targets
            
            # Assume current price is the baseline for comparison
            if middle_target > lower_target * 1.02:  # Forecast suggests upside
                signal = TradingSignal(
                    action=ActionType.BUY,
                    confidence=primary_forecast.probability,
                    strength=0.7,
                    timeframe="1W",
                    entry_price=None,
                    stop_loss=lower_target,
                    take_profit=upper_target,
                    risk_reward_ratio=(upper_target - middle_target) / (middle_target - lower_target) if middle_target != lower_target else None,
                    signal_source="price_forecast",
                    reasoning=f"Price forecast suggests upside to {middle_target:.2f}"
                )
                signals.append(signal)
            
            elif middle_target < upper_target * 0.98:  # Forecast suggests downside
                signal = TradingSignal(
                    action=ActionType.SELL,
                    confidence=primary_forecast.probability,
                    strength=0.7,
                    timeframe="1W",
                    entry_price=None,
                    stop_loss=upper_target,
                    take_profit=lower_target,
                    risk_reward_ratio=(middle_target - lower_target) / (upper_target - middle_target) if upper_target != middle_target else None,
                    signal_source="price_forecast",
                    reasoning=f"Price forecast suggests downside to {middle_target:.2f}"
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Forecast signal generation failed: {e}")
            return []
    
    async def _apply_llm_reasoning(self, chart_features: ChartFeatures, 
                                 kronos_output: KronosOutput, 
                                 market_analysis: MarketAnalysis,
                                 market_sentiment: Optional[List[MarketSentiment]] = None,
                                 scraped_signals: Optional[List[TradingSignal]] = None) -> Dict[str, Any]:
        """Apply LLM-based reasoning for additional insights"""
        try:
            if not self.openai_api_key or not self.use_llm_reasoning:
                return {'status': 'llm_not_available'}
            
            logger.info("Applying LLM reasoning")
            
            # Prepare context for LLM
            context = self._prepare_llm_context(chart_features, kronos_output, market_analysis, market_sentiment, scraped_signals)
            
            # Create reasoning prompt
            prompt = self._create_reasoning_prompt(context)
            
            # Call LLM
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst and trader with deep knowledge of technical analysis, market dynamics, and risk management."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            llm_response = response.choices[0].message.content
            
            # Parse LLM response
            insights = self._parse_llm_response(llm_response)
            
            return insights
            
        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}")
            return {'status': 'llm_reasoning_failed', 'error': str(e)}
    
    def _prepare_llm_context(self, chart_features: ChartFeatures, 
                           kronos_output: KronosOutput, 
                           market_analysis: MarketAnalysis,
                           market_sentiment: Optional[List[MarketSentiment]] = None,
                           scraped_signals: Optional[List[TradingSignal]] = None) -> Dict[str, Any]:
        """Prepare context for LLM reasoning"""
        try:
            context = {
                'market_regime': market_analysis.market_regime.value,
                'trend_direction': kronos_output.trend_forecast.trend_direction.value,
                'trend_strength': kronos_output.trend_forecast.trend_strength,
                'volatility_regime': kronos_output.volatility_forecast.volatility_regime,
                'price_forecasts': [{
                    'scenario': forecast.scenario_name,
                    'probability': forecast.probability,
                    'targets': forecast.price_targets
                } for forecast in kronos_output.price_forecasts[:3]],
                'support_levels': chart_features.support_levels[:5],
                'resistance_levels': chart_features.resistance_levels[:5],
                'indicators': [{
                    'name': ind.name,
                    'type': ind.type.value,
                    'confidence': ind.confidence
                } for ind in chart_features.indicators[:10]],
                'risk_factors': market_analysis.risk_factors[:5],
                'opportunities': market_analysis.opportunities[:5],
                'market_sentiment': [{
                    'symbol': sentiment.symbol,
                    'sentiment_score': sentiment.sentiment_score,
                    'confidence': sentiment.confidence,
                    'source': sentiment.source,
                    'summary': sentiment.summary
                } for sentiment in (market_sentiment or [])[:5]],
                'scraped_signals': [{
                    'signal_type': signal.signal_type,
                    'strength': signal.strength,
                    'source': signal.source,
                    'reasoning': signal.reasoning,
                    'target_price': signal.target_price,
                    'stop_loss': signal.stop_loss
                } for signal in (scraped_signals or [])[:5]]
            }
            
            return context
            
        except Exception as e:
            logger.error(f"LLM context preparation failed: {e}")
            return {}
    
    def _create_reasoning_prompt(self, context: Dict[str, Any]) -> str:
        """Create reasoning prompt for LLM"""
        try:
            prompt = f"""
            Analyze the following financial market data and provide strategic trading recommendations:
            
            MARKET CONTEXT:
            - Market Regime: {context.get('market_regime', 'unknown')}
            - Trend Direction: {context.get('trend_direction', 'unknown')}
            - Trend Strength: {context.get('trend_strength', 0):.2f}
            - Volatility Regime: {context.get('volatility_regime', 'unknown')}
            
            PRICE FORECASTS:
            {json.dumps(context.get('price_forecasts', []), indent=2)}
            
            SUPPORT/RESISTANCE LEVELS:
            - Support: {context.get('support_levels', [])}
            - Resistance: {context.get('resistance_levels', [])}
            
            TECHNICAL INDICATORS:
            {json.dumps(context.get('indicators', []), indent=2)}
            
            RISK FACTORS:
            {context.get('risk_factors', [])}
            
            OPPORTUNITIES:
            {context.get('opportunities', [])}
            
            Please provide:
            1. Primary trading recommendation (BUY/SELL/HOLD)
            2. Confidence level (0-1)
            3. Key reasoning points
            4. Risk assessment
            5. Suggested position sizing
            6. Entry/exit strategy
            
            Format your response as JSON with these fields:
            {{
                "recommendation": "BUY/SELL/HOLD",
                "confidence": 0.0-1.0,
                "reasoning": ["point1", "point2", ...],
                "risk_level": "LOW/MEDIUM/HIGH",
                "position_size": 0.0-1.0,
                "entry_strategy": "description",
                "exit_strategy": "description",
                "key_insights": ["insight1", "insight2", ...]
            }}
            """
            
            return prompt
            
        except Exception as e:
            logger.error(f"Reasoning prompt creation failed: {e}")
            return "Analyze the current market conditions and provide trading recommendations."
    
    def _parse_llm_response(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response into structured insights"""
        try:
            # Try to parse as JSON first
            if '{' in llm_response and '}' in llm_response:
                json_start = llm_response.find('{')
                json_end = llm_response.rfind('}') + 1
                json_str = llm_response[json_start:json_end]
                
                try:
                    parsed_response = json.loads(json_str)
                    return {
                        'status': 'success',
                        'recommendation': parsed_response.get('recommendation', 'HOLD'),
                        'confidence': parsed_response.get('confidence', 0.5),
                        'reasoning': parsed_response.get('reasoning', []),
                        'risk_level': parsed_response.get('risk_level', 'MEDIUM'),
                        'position_size': parsed_response.get('position_size', 0.1),
                        'entry_strategy': parsed_response.get('entry_strategy', ''),
                        'exit_strategy': parsed_response.get('exit_strategy', ''),
                        'key_insights': parsed_response.get('key_insights', []),
                        'raw_response': llm_response
                    }
                except json.JSONDecodeError:
                    pass
            
            # Fallback to text parsing
            return {
                'status': 'text_parsed',
                'raw_response': llm_response,
                'key_insights': [llm_response[:500]]  # First 500 chars as insight
            }
            
        except Exception as e:
            logger.error(f"LLM response parsing failed: {e}")
            return {'status': 'parsing_failed', 'error': str(e)}
    
    async def _synthesize_strategies(self, market_analysis: MarketAnalysis, 
                                   trading_signals: List[TradingSignal], 
                                   llm_insights: Dict[str, Any]) -> List[StrategyRecommendation]:
        """Synthesize multiple strategy recommendations"""
        try:
            logger.info("Synthesizing trading strategies")
            
            strategies = []
            
            # Trend following strategy
            trend_strategy = await self._create_trend_following_strategy(market_analysis, trading_signals)
            if trend_strategy:
                strategies.append(trend_strategy)
            
            # Mean reversion strategy
            mean_reversion_strategy = await self._create_mean_reversion_strategy(market_analysis, trading_signals)
            if mean_reversion_strategy:
                strategies.append(mean_reversion_strategy)
            
            # Breakout strategy
            breakout_strategy = await self._create_breakout_strategy(market_analysis, trading_signals)
            if breakout_strategy:
                strategies.append(breakout_strategy)
            
            # Momentum strategy
            momentum_strategy = await self._create_momentum_strategy(market_analysis, trading_signals)
            if momentum_strategy:
                strategies.append(momentum_strategy)
            
            # LLM-enhanced strategy
            if llm_insights.get('status') == 'success':
                llm_strategy = await self._create_llm_enhanced_strategy(market_analysis, llm_insights)
                if llm_strategy:
                    strategies.append(llm_strategy)
            
            # Sort strategies by confidence and expected return
            strategies.sort(key=lambda s: (s.confidence * s.expected_return), reverse=True)
            
            return strategies[:5]  # Return top 5 strategies
            
        except Exception as e:
            logger.error(f"Strategy synthesis failed: {e}")
            return []
    
    async def _create_trend_following_strategy(self, market_analysis: MarketAnalysis, 
                                             trading_signals: List[TradingSignal]) -> Optional[StrategyRecommendation]:
        """Create trend following strategy"""
        try:
            # Get trend signals
            trend_signals = [s for s in trading_signals if s.signal_source == 'trend_analysis']
            
            if not trend_signals:
                return None
            
            primary_signal = trend_signals[0]
            
            # Determine strategy parameters based on market regime
            if market_analysis.market_regime in [MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET]:
                confidence = 0.8
                expected_return = 0.15
                risk_level = RiskLevel.MEDIUM
            else:
                confidence = 0.6
                expected_return = 0.08
                risk_level = RiskLevel.HIGH
            
            return StrategyRecommendation(
                strategy_type=StrategyType.TREND_FOLLOWING,
                primary_action=primary_signal.action,
                confidence=confidence,
                risk_level=risk_level,
                expected_return=expected_return,
                max_drawdown=0.1,
                holding_period=timedelta(weeks=4),
                entry_conditions=[
                    "Trend confirmation from multiple timeframes",
                    "Volume confirmation",
                    "Momentum alignment"
                ],
                exit_conditions=[
                    "Trend reversal signal",
                    "Stop loss hit",
                    "Take profit target reached"
                ],
                risk_management={
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.15,
                    "position_size_pct": 0.1
                },
                supporting_signals=trend_signals,
                conflicting_signals=[],
                market_context=f"Market regime: {market_analysis.market_regime.value}",
                reasoning="Trend following strategy based on strong directional momentum",
                metadata={"strategy_weight": self.strategy_weights[StrategyType.TREND_FOLLOWING]}
            )
            
        except Exception as e:
            logger.error(f"Trend following strategy creation failed: {e}")
            return None
    
    async def _create_mean_reversion_strategy(self, market_analysis: MarketAnalysis, 
                                            trading_signals: List[TradingSignal]) -> Optional[StrategyRecommendation]:
        """Create mean reversion strategy"""
        try:
            # Mean reversion works best in sideways markets
            if market_analysis.market_regime not in [MarketRegime.SIDEWAYS, MarketRegime.VOLATILE]:
                return None
            
            # Get relevant signals
            sr_signals = [s for s in trading_signals if s.signal_source in ['support_level', 'resistance_level']]
            vol_signals = [s for s in trading_signals if s.signal_source == 'volatility_mean_reversion']
            
            if not (sr_signals or vol_signals):
                return None
            
            primary_signal = sr_signals[0] if sr_signals else vol_signals[0]
            
            return StrategyRecommendation(
                strategy_type=StrategyType.MEAN_REVERSION,
                primary_action=primary_signal.action,
                confidence=0.7,
                risk_level=RiskLevel.MEDIUM,
                expected_return=0.08,
                max_drawdown=0.06,
                holding_period=timedelta(weeks=2),
                entry_conditions=[
                    "Price near support/resistance levels",
                    "Oversold/overbought conditions",
                    "Low volatility environment"
                ],
                exit_conditions=[
                    "Return to mean",
                    "Opposite extreme reached",
                    "Stop loss triggered"
                ],
                risk_management={
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.08,
                    "position_size_pct": 0.08
                },
                supporting_signals=sr_signals + vol_signals,
                conflicting_signals=[],
                market_context="Sideways/volatile market suitable for mean reversion",
                reasoning="Mean reversion strategy targeting price extremes",
                metadata={"strategy_weight": self.strategy_weights[StrategyType.MEAN_REVERSION]}
            )
            
        except Exception as e:
            logger.error(f"Mean reversion strategy creation failed: {e}")
            return None
    
    async def _create_breakout_strategy(self, market_analysis: MarketAnalysis, 
                                      trading_signals: List[TradingSignal]) -> Optional[StrategyRecommendation]:
        """Create breakout strategy"""
        try:
            # Get breakout-related signals
            breakout_signals = [s for s in trading_signals if s.signal_source == 'volatility_breakout']
            volume_signals = [s for s in trading_signals if s.signal_source == 'volume']
            
            if not breakout_signals:
                return None
            
            primary_signal = breakout_signals[0]
            
            # Breakout strategy parameters
            confidence = 0.75 if volume_signals else 0.6
            
            return StrategyRecommendation(
                strategy_type=StrategyType.BREAKOUT,
                primary_action=ActionType.WAIT,  # Wait for direction confirmation
                confidence=confidence,
                risk_level=RiskLevel.HIGH,
                expected_return=0.12,
                max_drawdown=0.08,
                holding_period=timedelta(days=10),
                entry_conditions=[
                    "Breakout from consolidation",
                    "High volume confirmation",
                    "Clear direction established"
                ],
                exit_conditions=[
                    "Failed breakout (return to range)",
                    "Target reached",
                    "Stop loss hit"
                ],
                risk_management={
                    "stop_loss_pct": 0.04,
                    "take_profit_pct": 0.12,
                    "position_size_pct": 0.06
                },
                supporting_signals=breakout_signals + volume_signals,
                conflicting_signals=[],
                market_context="High breakout probability detected",
                reasoning="Breakout strategy targeting range expansion",
                metadata={"strategy_weight": self.strategy_weights[StrategyType.BREAKOUT]}
            )
            
        except Exception as e:
            logger.error(f"Breakout strategy creation failed: {e}")
            return None
    
    async def _create_momentum_strategy(self, market_analysis: MarketAnalysis, 
                                      trading_signals: List[TradingSignal]) -> Optional[StrategyRecommendation]:
        """Create momentum strategy"""
        try:
            # Get momentum signals
            momentum_signals = [s for s in trading_signals if s.signal_source == 'momentum']
            
            if not momentum_signals:
                return None
            
            primary_signal = momentum_signals[0]
            
            return StrategyRecommendation(
                strategy_type=StrategyType.MOMENTUM,
                primary_action=primary_signal.action,
                confidence=0.65,
                risk_level=RiskLevel.MEDIUM,
                expected_return=0.10,
                max_drawdown=0.07,
                holding_period=timedelta(weeks=1),
                entry_conditions=[
                    "Strong momentum confirmation",
                    "Price acceleration",
                    "Volume support"
                ],
                exit_conditions=[
                    "Momentum divergence",
                    "Overbought/oversold levels",
                    "Stop loss triggered"
                ],
                risk_management={
                    "stop_loss_pct": 0.04,
                    "take_profit_pct": 0.10,
                    "position_size_pct": 0.08
                },
                supporting_signals=momentum_signals,
                conflicting_signals=[],
                market_context="Strong momentum detected",
                reasoning="Momentum strategy riding price acceleration",
                metadata={"strategy_weight": self.strategy_weights[StrategyType.MOMENTUM]}
            )
            
        except Exception as e:
            logger.error(f"Momentum strategy creation failed: {e}")
            return None
    
    async def _create_llm_enhanced_strategy(self, market_analysis: MarketAnalysis, 
                                          llm_insights: Dict[str, Any]) -> Optional[StrategyRecommendation]:
        """Create LLM-enhanced strategy"""
        try:
            if llm_insights.get('status') != 'success':
                return None
            
            # Map LLM recommendation to action
            llm_rec = llm_insights.get('recommendation', 'HOLD')
            action_map = {
                'BUY': ActionType.BUY,
                'SELL': ActionType.SELL,
                'HOLD': ActionType.HOLD,
                'STRONG_BUY': ActionType.STRONG_BUY,
                'STRONG_SELL': ActionType.STRONG_SELL
            }
            
            primary_action = action_map.get(llm_rec, ActionType.HOLD)
            
            # Map risk level
            risk_map = {
                'LOW': RiskLevel.LOW,
                'MEDIUM': RiskLevel.MEDIUM,
                'HIGH': RiskLevel.HIGH
            }
            
            risk_level = risk_map.get(llm_insights.get('risk_level', 'MEDIUM'), RiskLevel.MEDIUM)
            
            return StrategyRecommendation(
                strategy_type=StrategyType.SWING,  # Default to swing for LLM strategies
                primary_action=primary_action,
                confidence=llm_insights.get('confidence', 0.6),
                risk_level=risk_level,
                expected_return=0.12,
                max_drawdown=0.08,
                holding_period=timedelta(weeks=2),
                entry_conditions=[
                    "LLM analysis confirmation",
                    llm_insights.get('entry_strategy', 'Market entry')
                ],
                exit_conditions=[
                    "LLM exit signal",
                    llm_insights.get('exit_strategy', 'Market exit')
                ],
                risk_management={
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.12,
                    "position_size_pct": llm_insights.get('position_size', 0.1)
                },
                supporting_signals=[],
                conflicting_signals=[],
                market_context="LLM-enhanced analysis",
                reasoning=f"LLM reasoning: {'; '.join(llm_insights.get('reasoning', []))}",
                metadata={
                    "llm_insights": llm_insights.get('key_insights', []),
                    "strategy_weight": 0.3
                }
            )
            
        except Exception as e:
            logger.error(f"LLM-enhanced strategy creation failed: {e}")
            return None