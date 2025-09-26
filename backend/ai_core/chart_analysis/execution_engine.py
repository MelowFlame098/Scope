"""Execution Engine for AI Financial Chart Analysis Pipeline

This module implements the custom algorithmic execution layer that combines
model outputs and confidence scores to produce final actionable trading signals.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from decimal import Decimal

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_WEAK = "very_weak"
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"

class ExecutionTiming(Enum):
    """Execution timing options"""
    IMMEDIATE = "immediate"
    MARKET_OPEN = "market_open"
    MARKET_CLOSE = "market_close"
    NEXT_SESSION = "next_session"
    CONDITIONAL = "conditional"

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_position_size: float  # Maximum position size as % of portfolio
    stop_loss_pct: float     # Stop loss percentage
    take_profit_pct: float   # Take profit percentage
    max_daily_loss: float    # Maximum daily loss limit
    correlation_limit: float # Maximum correlation with existing positions
    volatility_adjustment: float  # Volatility-based position sizing

@dataclass
class ExecutionSignal:
    """Final actionable trading signal"""
    symbol: str
    action: str  # BUY, SELL, HOLD, CLOSE
    signal_strength: SignalStrength
    confidence: float
    position_size: float  # Recommended position size
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    execution_timing: ExecutionTiming
    risk_parameters: RiskParameters
    reasoning: str
    supporting_strategies: List[str]
    conflicting_signals: List[str]
    market_conditions: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    expiry: Optional[datetime]

class ExecutionEngine:
    """Custom algorithmic execution layer"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Risk management settings
        self.risk_settings = {
            'max_portfolio_risk': 0.02,  # 2% max portfolio risk per trade
            'max_position_size': 0.10,   # 10% max position size
            'correlation_threshold': 0.7, # Maximum correlation threshold
            'volatility_lookback': 20,   # Days for volatility calculation
            'confidence_threshold': 0.6, # Minimum confidence for execution
            'max_concurrent_positions': 5 # Maximum concurrent positions
        }
        
        # Strategy weights for ensemble decision making
        self.strategy_weights = {
            'trend_following': 0.25,
            'mean_reversion': 0.20,
            'breakout': 0.20,
            'momentum': 0.15,
            'llm_enhanced': 0.20
        }
        
        # Market regime adjustments
        self.regime_adjustments = {
            'BULL': {'risk_multiplier': 1.2, 'confidence_boost': 0.1},
            'BEAR': {'risk_multiplier': 0.8, 'confidence_penalty': 0.1},
            'SIDEWAYS': {'risk_multiplier': 0.9, 'confidence_neutral': 0.0},
            'VOLATILE': {'risk_multiplier': 0.7, 'confidence_penalty': 0.15}
        }
    
    async def generate_execution_signal(self, 
                                      symbol: str,
                                      fin_r1_output: Any,  # FinR1Output
                                      market_data: Dict[str, Any],
                                      portfolio_context: Optional[Dict[str, Any]] = None) -> Optional[ExecutionSignal]:
        """Generate final execution signal"""
        try:
            # Validate inputs
            if not self._validate_inputs(fin_r1_output, market_data):
                return None
            
            # Calculate ensemble signal
            ensemble_decision = await self._calculate_ensemble_decision(fin_r1_output)
            
            if not ensemble_decision:
                return None
            
            # Apply risk management
            risk_params = await self._calculate_risk_parameters(
                symbol, ensemble_decision, market_data, portfolio_context
            )
            
            # Determine execution timing
            execution_timing = await self._determine_execution_timing(
                ensemble_decision, market_data
            )
            
            # Calculate position sizing
            position_size = await self._calculate_position_size(
                ensemble_decision, risk_params, market_data
            )
            
            # Generate final signal
            signal = ExecutionSignal(
                symbol=symbol,
                action=ensemble_decision['action'],
                signal_strength=ensemble_decision['strength'],
                confidence=ensemble_decision['confidence'],
                position_size=position_size,
                entry_price=market_data.get('current_price'),
                stop_loss=risk_params.stop_loss_pct,
                take_profit=risk_params.take_profit_pct,
                execution_timing=execution_timing,
                risk_parameters=risk_params,
                reasoning=ensemble_decision['reasoning'],
                supporting_strategies=ensemble_decision['supporting_strategies'],
                conflicting_signals=ensemble_decision['conflicting_signals'],
                market_conditions=market_data,
                metadata=ensemble_decision['metadata'],
                timestamp=datetime.now(),
                expiry=datetime.now() + timedelta(hours=24)
            )
            
            # Final validation
            if await self._validate_signal(signal, portfolio_context):
                return signal
            
            return None
            
        except Exception as e:
            logger.error(f"Execution signal generation failed: {e}")
            return None
    
    async def _validate_inputs(self, fin_r1_output: Any, market_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        try:
            # Check FinR1 output
            if not fin_r1_output or not hasattr(fin_r1_output, 'strategy_recommendations'):
                return False
            
            # Check market data
            required_fields = ['current_price', 'volume', 'volatility']
            if not all(field in market_data for field in required_fields):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    async def _calculate_ensemble_decision(self, fin_r1_output: Any) -> Optional[Dict[str, Any]]:
        """Calculate ensemble decision from multiple strategies"""
        try:
            strategies = fin_r1_output.strategy_recommendations
            
            if not strategies:
                return None
            
            # Calculate weighted scores for each action
            action_scores = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
            total_weight = 0
            supporting_strategies = []
            conflicting_signals = []
            
            for strategy in strategies:
                strategy_type = strategy.strategy_type.value
                weight = self.strategy_weights.get(strategy_type, 0.1)
                confidence = strategy.confidence
                
                # Weight by confidence and strategy weight
                weighted_score = weight * confidence
                
                if strategy.primary_action.value in action_scores:
                    action_scores[strategy.primary_action.value] += weighted_score
                    supporting_strategies.append(strategy_type)
                
                total_weight += weight
            
            # Normalize scores
            if total_weight > 0:
                for action in action_scores:
                    action_scores[action] /= total_weight
            
            # Determine primary action
            primary_action = max(action_scores, key=action_scores.get)
            confidence = action_scores[primary_action]
            
            # Check confidence threshold
            if confidence < self.risk_settings['confidence_threshold']:
                primary_action = 'HOLD'
            
            # Determine signal strength
            strength = self._calculate_signal_strength(confidence)
            
            # Generate reasoning
            reasoning = self._generate_ensemble_reasoning(strategies, primary_action, confidence)
            
            return {
                'action': primary_action,
                'confidence': confidence,
                'strength': strength,
                'supporting_strategies': supporting_strategies,
                'conflicting_signals': conflicting_signals,
                'reasoning': reasoning,
                'metadata': {
                    'action_scores': action_scores,
                    'strategy_count': len(strategies),
                    'total_weight': total_weight
                }
            }
            
        except Exception as e:
            logger.error(f"Ensemble decision calculation failed: {e}")
            return None
    
    def _calculate_signal_strength(self, confidence: float) -> SignalStrength:
        """Calculate signal strength based on confidence"""
        if confidence >= 0.9:
            return SignalStrength.VERY_STRONG
        elif confidence >= 0.8:
            return SignalStrength.STRONG
        elif confidence >= 0.7:
            return SignalStrength.MODERATE
        elif confidence >= 0.6:
            return SignalStrength.WEAK
        else:
            return SignalStrength.VERY_WEAK
    
    def _generate_ensemble_reasoning(self, strategies: List[Any], 
                                   action: str, confidence: float) -> str:
        """Generate reasoning for ensemble decision"""
        try:
            strategy_types = [s.strategy_type.value for s in strategies]
            strategy_count = len(strategies)
            
            reasoning = f"Ensemble decision: {action} with {confidence:.2f} confidence. "
            reasoning += f"Based on {strategy_count} strategies: {', '.join(strategy_types)}. "
            
            # Add specific insights
            high_conf_strategies = [s for s in strategies if s.confidence > 0.8]
            if high_conf_strategies:
                reasoning += f"High confidence from: {', '.join([s.strategy_type.value for s in high_conf_strategies])}. "
            
            return reasoning
            
        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return f"Ensemble decision: {action} with {confidence:.2f} confidence"
    
    async def _calculate_risk_parameters(self, symbol: str, ensemble_decision: Dict[str, Any],
                                       market_data: Dict[str, Any], 
                                       portfolio_context: Optional[Dict[str, Any]]) -> RiskParameters:
        """Calculate risk management parameters"""
        try:
            # Base risk parameters
            base_stop_loss = 0.02  # 2%
            base_take_profit = 0.06  # 6%
            
            # Adjust for volatility
            volatility = market_data.get('volatility', 0.02)
            volatility_multiplier = min(2.0, max(0.5, volatility / 0.02))
            
            # Adjust for confidence
            confidence = ensemble_decision['confidence']
            confidence_multiplier = 0.5 + (confidence * 0.5)
            
            # Calculate final parameters
            stop_loss_pct = base_stop_loss * volatility_multiplier
            take_profit_pct = base_take_profit * confidence_multiplier
            
            # Position sizing based on risk
            max_position = self.risk_settings['max_position_size']
            risk_adjusted_position = max_position * confidence_multiplier
            
            return RiskParameters(
                max_position_size=risk_adjusted_position,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                max_daily_loss=0.05,  # 5% max daily loss
                correlation_limit=self.risk_settings['correlation_threshold'],
                volatility_adjustment=volatility_multiplier
            )
            
        except Exception as e:
            logger.error(f"Risk parameter calculation failed: {e}")
            return RiskParameters(
                max_position_size=0.05,
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                max_daily_loss=0.05,
                correlation_limit=0.7,
                volatility_adjustment=1.0
            )
    
    async def _determine_execution_timing(self, ensemble_decision: Dict[str, Any],
                                        market_data: Dict[str, Any]) -> ExecutionTiming:
        """Determine optimal execution timing"""
        try:
            confidence = ensemble_decision['confidence']
            strength = ensemble_decision['strength']
            
            # High confidence signals execute immediately
            if confidence > 0.8 and strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
                return ExecutionTiming.IMMEDIATE
            
            # Medium confidence waits for market open
            elif confidence > 0.7:
                return ExecutionTiming.MARKET_OPEN
            
            # Lower confidence waits for next session
            else:
                return ExecutionTiming.NEXT_SESSION
                
        except Exception as e:
            logger.error(f"Execution timing determination failed: {e}")
            return ExecutionTiming.NEXT_SESSION
    
    async def _calculate_position_size(self, ensemble_decision: Dict[str, Any],
                                     risk_params: RiskParameters,
                                     market_data: Dict[str, Any]) -> float:
        """Calculate optimal position size"""
        try:
            # Base position size from risk parameters
            base_size = risk_params.max_position_size
            
            # Adjust for confidence
            confidence = ensemble_decision['confidence']
            confidence_adjustment = 0.5 + (confidence * 0.5)
            
            # Adjust for signal strength
            strength = ensemble_decision['strength']
            strength_multipliers = {
                SignalStrength.VERY_WEAK: 0.3,
                SignalStrength.WEAK: 0.5,
                SignalStrength.MODERATE: 0.7,
                SignalStrength.STRONG: 0.9,
                SignalStrength.VERY_STRONG: 1.0
            }
            
            strength_adjustment = strength_multipliers.get(strength, 0.5)
            
            # Calculate final position size
            position_size = base_size * confidence_adjustment * strength_adjustment
            
            # Apply minimum and maximum limits
            position_size = max(0.01, min(position_size, risk_params.max_position_size))
            
            return round(position_size, 4)
            
        except Exception as e:
            logger.error(f"Position size calculation failed: {e}")
            return 0.02  # Default 2% position
    
    async def _validate_signal(self, signal: ExecutionSignal, 
                             portfolio_context: Optional[Dict[str, Any]]) -> bool:
        """Final validation of execution signal"""
        try:
            # Check confidence threshold
            if signal.confidence < self.risk_settings['confidence_threshold']:
                return False
            
            # Check position size limits
            if signal.position_size > self.risk_settings['max_position_size']:
                return False
            
            # Check portfolio context if provided
            if portfolio_context:
                # Check maximum concurrent positions
                current_positions = portfolio_context.get('current_positions', 0)
                if current_positions >= self.risk_settings['max_concurrent_positions']:
                    return False
                
                # Check correlation limits (simplified)
                existing_symbols = portfolio_context.get('symbols', [])
                if signal.symbol in existing_symbols:
                    return False  # Avoid duplicate positions
            
            return True
            
        except Exception as e:
            logger.error(f"Signal validation failed: {e}")
            return False
    
    async def update_risk_settings(self, new_settings: Dict[str, Any]) -> bool:
        """Update risk management settings"""
        try:
            self.risk_settings.update(new_settings)
            logger.info(f"Risk settings updated: {new_settings}")
            return True
            
        except Exception as e:
            logger.error(f"Risk settings update failed: {e}")
            return False
    
    async def get_signal_metrics(self, signals: List[ExecutionSignal]) -> Dict[str, Any]:
        """Calculate metrics for execution signals"""
        try:
            if not signals:
                return {}
            
            # Calculate basic metrics
            total_signals = len(signals)
            buy_signals = len([s for s in signals if s.action == 'BUY'])
            sell_signals = len([s for s in signals if s.action == 'SELL'])
            hold_signals = len([s for s in signals if s.action == 'HOLD'])
            
            # Calculate confidence distribution
            confidences = [s.confidence for s in signals]
            avg_confidence = np.mean(confidences)
            
            # Calculate position size distribution
            position_sizes = [s.position_size for s in signals]
            avg_position_size = np.mean(position_sizes)
            
            return {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'avg_confidence': round(avg_confidence, 3),
                'avg_position_size': round(avg_position_size, 4),
                'confidence_range': [min(confidences), max(confidences)],
                'position_size_range': [min(position_sizes), max(position_sizes)]
            }
            
        except Exception as e:
            logger.error(f"Signal metrics calculation failed: {e}")
            return {}