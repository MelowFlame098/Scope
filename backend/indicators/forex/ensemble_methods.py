import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
from collections import deque
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexMarketRegime(Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING_HIGH_VOL = "ranging_high_vol"
    RANGING_LOW_VOL = "ranging_low_vol"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    CRISIS = "crisis"
    CARRY_TRADE = "carry_trade"
    RISK_OFF = "risk_off"
    INTERVENTION = "intervention"

class EnsembleStrategy(Enum):
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    REGIME_AWARE = "regime_aware"
    DYNAMIC_SELECTION = "dynamic_selection"
    BAYESIAN_MODEL_AVERAGING = "bayesian_model_averaging"
    STACKING = "stacking"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    CORRELATION_ADJUSTED = "correlation_adjusted"
    ADAPTIVE_LEARNING = "adaptive_learning"

@dataclass
class ForexModelPrediction:
    """Individual model prediction for ensemble"""
    model_name: str
    model_type: str
    timestamp: datetime
    currency_pair: str
    
    # Core prediction
    predicted_price: float
    predicted_direction: int  # -1, 0, 1
    confidence: float
    
    # Risk metrics
    predicted_volatility: float = 0.0
    value_at_risk: float = 0.0
    
    # Model-specific data
    model_features: Dict[str, float] = field(default_factory=dict)
    prediction_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Performance tracking
    recent_accuracy: float = 0.5
    recent_sharpe: float = 0.0
    recent_drawdown: float = 0.0
    
    # Regime sensitivity
    regime_performance: Dict[ForexMarketRegime, float] = field(default_factory=dict)
    regime_confidence: Dict[ForexMarketRegime, float] = field(default_factory=dict)

@dataclass
class ForexModelPerformance:
    """Performance tracking for forex models"""
    model_name: str
    
    # Overall performance
    total_predictions: int = 0
    correct_predictions: int = 0
    accuracy: float = 0.5
    
    # Risk-adjusted performance
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Regime-specific performance
    regime_accuracy: Dict[ForexMarketRegime, float] = field(default_factory=dict)
    regime_sharpe: Dict[ForexMarketRegime, float] = field(default_factory=dict)
    regime_predictions: Dict[ForexMarketRegime, int] = field(default_factory=dict)
    
    # Recent performance (rolling window)
    recent_accuracy_30d: float = 0.5
    recent_accuracy_7d: float = 0.5
    recent_sharpe_30d: float = 0.0
    recent_volatility: float = 0.0
    
    # Prediction quality
    average_confidence: float = 0.5
    confidence_accuracy_correlation: float = 0.0
    prediction_bias: float = 0.0  # Tendency to over/under predict
    
    # Stability metrics
    prediction_consistency: float = 1.0
    model_drift: float = 0.0
    
    # Last update
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ForexEnsembleResult:
    """Result from ensemble prediction"""
    timestamp: datetime
    currency_pair: str
    ensemble_strategy: EnsembleStrategy
    market_regime: ForexMarketRegime
    
    # Ensemble predictions
    ensemble_price: float
    ensemble_direction: int
    ensemble_confidence: float
    
    # Individual model contributions
    model_predictions: Dict[str, ForexModelPrediction] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    
    # Consensus analysis
    price_consensus: float = 0.0  # 0-1, higher = more agreement
    direction_consensus: float = 0.0
    confidence_consensus: float = 0.0
    
    # Risk assessment
    ensemble_volatility: float = 0.0
    ensemble_var: float = 0.0
    model_disagreement: float = 0.0
    prediction_uncertainty: float = 0.0
    
    # Regime analysis
    regime_confidence: float = 0.0
    regime_stability: float = 0.0
    regime_transition_probability: Dict[ForexMarketRegime, float] = field(default_factory=dict)
    
    # Trading recommendations
    recommended_action: str = "HOLD"
    position_size: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Quality metrics
    prediction_quality: float = 0.0
    ensemble_stability: float = 0.0
    
    # Supporting information
    key_factors: Dict[str, float] = field(default_factory=dict)
    risk_warnings: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)

class ForexRegimeDetector:
    """Detect market regimes for forex pairs"""
    
    def __init__(self,
                 lookback_period: int = 50,
                 volatility_threshold: float = 0.015,
                 trend_threshold: float = 0.02):
        """
        Initialize regime detector
        
        Args:
            lookback_period: Number of periods to analyze
            volatility_threshold: Threshold for high/low volatility regimes
            trend_threshold: Threshold for trending vs ranging markets
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        
        # Regime history
        self.regime_history: deque = deque(maxlen=100)
        self.regime_transitions = {}
        
        logger.info(f"Initialized ForexRegimeDetector with {lookback_period} period lookback")
    
    def detect_regime(self, prices: List[float], volumes: List[float] = None, 
                     economic_data: Dict[str, float] = None) -> Tuple[ForexMarketRegime, float]:
        """Detect current market regime"""
        try:
            if len(prices) < self.lookback_period:
                return ForexMarketRegime.RANGING_LOW_VOL, 0.5
            
            recent_prices = prices[-self.lookback_period:]
            
            # Calculate key metrics
            returns = np.diff(recent_prices) / recent_prices[:-1]
            volatility = np.std(returns)
            
            # Trend analysis
            price_change = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            trend_strength = abs(price_change)
            
            # Moving averages for trend confirmation
            short_ma = np.mean(recent_prices[-10:])
            long_ma = np.mean(recent_prices[-30:] if len(recent_prices) >= 30 else recent_prices)
            ma_divergence = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
            
            # Range analysis
            price_range = (max(recent_prices) - min(recent_prices)) / np.mean(recent_prices)
            
            # Economic factors
            crisis_indicator = 0.0
            intervention_probability = 0.0
            carry_trade_attractiveness = 0.0
            
            if economic_data:
                crisis_indicator = economic_data.get('vix', 0) / 100.0  # Normalize VIX
                intervention_probability = economic_data.get('intervention_prob', 0.0)
                carry_trade_attractiveness = economic_data.get('interest_rate_diff', 0.0)
            
            # Regime detection logic
            confidence = 0.5
            
            # Crisis regime (highest priority)
            if crisis_indicator > 0.3 or volatility > self.volatility_threshold * 2:
                regime = ForexMarketRegime.CRISIS
                confidence = min(0.9, 0.5 + crisis_indicator + volatility / self.volatility_threshold)
            
            # Intervention regime
            elif intervention_probability > 0.7:
                regime = ForexMarketRegime.INTERVENTION
                confidence = intervention_probability
            
            # Trending regimes
            elif trend_strength > self.trend_threshold and abs(ma_divergence) > 0.01:
                if price_change > 0:
                    regime = ForexMarketRegime.TRENDING_BULL
                else:
                    regime = ForexMarketRegime.TRENDING_BEAR
                confidence = min(0.9, 0.5 + trend_strength / self.trend_threshold)
            
            # Breakout regime
            elif price_range > self.trend_threshold * 1.5 and volatility > self.volatility_threshold:
                regime = ForexMarketRegime.BREAKOUT
                confidence = min(0.8, 0.5 + price_range / self.trend_threshold)
            
            # Carry trade regime
            elif abs(carry_trade_attractiveness) > 0.02 and volatility < self.volatility_threshold:
                regime = ForexMarketRegime.CARRY_TRADE
                confidence = min(0.8, 0.5 + abs(carry_trade_attractiveness) / 0.04)
            
            # Risk-off regime
            elif crisis_indicator > 0.15 and volatility > self.volatility_threshold * 0.8:
                regime = ForexMarketRegime.RISK_OFF
                confidence = min(0.8, 0.4 + crisis_indicator + volatility / self.volatility_threshold)
            
            # Ranging regimes
            elif volatility > self.volatility_threshold:
                regime = ForexMarketRegime.RANGING_HIGH_VOL
                confidence = min(0.7, 0.5 + volatility / self.volatility_threshold)
            else:
                regime = ForexMarketRegime.RANGING_LOW_VOL
                confidence = min(0.7, 0.6 - volatility / self.volatility_threshold)
            
            # Update regime history
            self.regime_history.append((datetime.now(), regime, confidence))
            
            return regime, confidence
            
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return ForexMarketRegime.RANGING_LOW_VOL, 0.5
    
    def get_regime_stability(self, lookback: int = 10) -> float:
        """Calculate regime stability over recent periods"""
        try:
            if len(self.regime_history) < lookback:
                return 0.5
            
            recent_regimes = [r[1] for r in list(self.regime_history)[-lookback:]]
            most_common_regime = max(set(recent_regimes), key=recent_regimes.count)
            stability = recent_regimes.count(most_common_regime) / len(recent_regimes)
            
            return stability
            
        except Exception as e:
            logger.error(f"Error calculating regime stability: {e}")
            return 0.5
    
    def predict_regime_transition(self, current_regime: ForexMarketRegime) -> Dict[ForexMarketRegime, float]:
        """Predict probability of regime transitions"""
        try:
            # Simple transition probabilities (in practice, use Markov chains or ML)
            transition_probs = {
                ForexMarketRegime.TRENDING_BULL: 0.1,
                ForexMarketRegime.TRENDING_BEAR: 0.1,
                ForexMarketRegime.RANGING_HIGH_VOL: 0.2,
                ForexMarketRegime.RANGING_LOW_VOL: 0.3,
                ForexMarketRegime.BREAKOUT: 0.1,
                ForexMarketRegime.REVERSAL: 0.05,
                ForexMarketRegime.CRISIS: 0.02,
                ForexMarketRegime.CARRY_TRADE: 0.08,
                ForexMarketRegime.RISK_OFF: 0.03,
                ForexMarketRegime.INTERVENTION: 0.02
            }
            
            # Adjust based on current regime
            if current_regime == ForexMarketRegime.TRENDING_BULL:
                transition_probs[ForexMarketRegime.TRENDING_BULL] = 0.4
                transition_probs[ForexMarketRegime.REVERSAL] = 0.15
                transition_probs[ForexMarketRegime.RANGING_HIGH_VOL] = 0.25
            elif current_regime == ForexMarketRegime.CRISIS:
                transition_probs[ForexMarketRegime.CRISIS] = 0.5
                transition_probs[ForexMarketRegime.RISK_OFF] = 0.3
                transition_probs[ForexMarketRegime.RANGING_HIGH_VOL] = 0.15
            
            # Normalize probabilities
            total_prob = sum(transition_probs.values())
            transition_probs = {k: v/total_prob for k, v in transition_probs.items()}
            
            return transition_probs
            
        except Exception as e:
            logger.error(f"Error predicting regime transition: {e}")
            return {regime: 1.0/len(ForexMarketRegime) for regime in ForexMarketRegime}

class ForexAdvancedEnsemblePredictor:
    """Advanced ensemble predictor for forex with regime awareness"""
    
    def __init__(self,
                 default_strategy: EnsembleStrategy = EnsembleStrategy.REGIME_AWARE,
                 performance_window: int = 30,
                 min_models: int = 2,
                 max_models: int = 10):
        """
        Initialize advanced ensemble predictor
        
        Args:
            default_strategy: Default ensemble strategy
            performance_window: Window for performance calculation
            min_models: Minimum number of models required
            max_models: Maximum number of models to use
        """
        self.default_strategy = default_strategy
        self.performance_window = performance_window
        self.min_models = min_models
        self.max_models = max_models
        
        # Components
        self.regime_detector = ForexRegimeDetector()
        
        # Model tracking
        self.model_performance: Dict[str, ForexModelPerformance] = {}
        self.prediction_history: deque = deque(maxlen=1000)
        
        # Ensemble strategies
        self.ensemble_strategies = {
            EnsembleStrategy.EQUAL_WEIGHT: self._equal_weight_ensemble,
            EnsembleStrategy.PERFORMANCE_WEIGHTED: self._performance_weighted_ensemble,
            EnsembleStrategy.VOLATILITY_ADJUSTED: self._volatility_adjusted_ensemble,
            EnsembleStrategy.REGIME_AWARE: self._regime_aware_ensemble,
            EnsembleStrategy.DYNAMIC_SELECTION: self._dynamic_selection_ensemble,
            EnsembleStrategy.CONFIDENCE_WEIGHTED: self._confidence_weighted_ensemble,
            EnsembleStrategy.CORRELATION_ADJUSTED: self._correlation_adjusted_ensemble,
            EnsembleStrategy.ADAPTIVE_LEARNING: self._adaptive_learning_ensemble
        }
        
        # Strategy performance tracking
        self.strategy_performance: Dict[EnsembleStrategy, float] = {}
        
        logger.info(f"Initialized ForexAdvancedEnsemblePredictor with {default_strategy.value} strategy")
    
    def update_model_performance(self, model_name: str, actual_price: float, 
                               predicted_price: float, prediction_time: datetime,
                               regime: ForexMarketRegime = None):
        """Update model performance with actual results"""
        try:
            if model_name not in self.model_performance:
                self.model_performance[model_name] = ForexModelPerformance(model_name=model_name)
            
            perf = self.model_performance[model_name]
            
            # Calculate prediction error
            error = abs(actual_price - predicted_price) / actual_price
            is_correct_direction = (actual_price > predicted_price) == (predicted_price > 0)  # Simplified
            
            # Update overall performance
            perf.total_predictions += 1
            if is_correct_direction:
                perf.correct_predictions += 1
            
            perf.accuracy = perf.correct_predictions / perf.total_predictions
            
            # Update regime-specific performance
            if regime:
                if regime not in perf.regime_accuracy:
                    perf.regime_accuracy[regime] = 0.0
                    perf.regime_predictions[regime] = 0
                
                perf.regime_predictions[regime] += 1
                if is_correct_direction:
                    perf.regime_accuracy[regime] = (
                        (perf.regime_accuracy[regime] * (perf.regime_predictions[regime] - 1) + 1.0) /
                        perf.regime_predictions[regime]
                    )
                else:
                    perf.regime_accuracy[regime] = (
                        (perf.regime_accuracy[regime] * (perf.regime_predictions[regime] - 1)) /
                        perf.regime_predictions[regime]
                    )
            
            perf.last_updated = datetime.now()
            
            logger.debug(f"Updated performance for {model_name}: accuracy={perf.accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error updating model performance for {model_name}: {e}")
    
    def _equal_weight_ensemble(self, predictions: List[ForexModelPrediction], 
                              regime: ForexMarketRegime, regime_confidence: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Equal weight ensemble strategy"""
        weights = {pred.model_name: 1.0 / len(predictions) for pred in predictions}
        metadata = {'strategy': 'equal_weight', 'num_models': len(predictions)}
        return weights, metadata
    
    def _performance_weighted_ensemble(self, predictions: List[ForexModelPrediction], 
                                     regime: ForexMarketRegime, regime_confidence: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Performance-weighted ensemble strategy"""
        try:
            weights = {}
            total_performance = 0.0
            
            for pred in predictions:
                if pred.model_name in self.model_performance:
                    perf = self.model_performance[pred.model_name]
                    # Use recent accuracy with regime adjustment
                    base_performance = perf.recent_accuracy_30d
                    
                    # Adjust for regime-specific performance
                    if regime in perf.regime_accuracy and perf.regime_predictions[regime] > 5:
                        regime_adjustment = perf.regime_accuracy[regime] - 0.5  # Center around 0
                        base_performance += 0.3 * regime_adjustment
                    
                    performance = max(0.1, min(0.9, base_performance))  # Clamp between 0.1 and 0.9
                else:
                    performance = 0.5  # Default for new models
                
                weights[pred.model_name] = performance
                total_performance += performance
            
            # Normalize weights
            if total_performance > 0:
                weights = {k: v/total_performance for k, v in weights.items()}
            else:
                weights = {pred.model_name: 1.0 / len(predictions) for pred in predictions}
            
            metadata = {
                'strategy': 'performance_weighted',
                'total_performance': total_performance,
                'regime_adjusted': regime in [p.regime_performance.keys() for p in predictions if p.model_name in self.model_performance]
            }
            
            return weights, metadata
            
        except Exception as e:
            logger.error(f"Error in performance weighted ensemble: {e}")
            return self._equal_weight_ensemble(predictions, regime, regime_confidence)
    
    def _volatility_adjusted_ensemble(self, predictions: List[ForexModelPrediction], 
                                    regime: ForexMarketRegime, regime_confidence: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Volatility-adjusted ensemble strategy"""
        try:
            weights = {}
            total_inv_vol = 0.0
            
            for pred in predictions:
                # Use inverse volatility weighting
                volatility = max(0.001, pred.predicted_volatility)  # Avoid division by zero
                inv_vol = 1.0 / volatility
                
                # Adjust for model confidence
                confidence_adjustment = pred.confidence
                adjusted_weight = inv_vol * confidence_adjustment
                
                weights[pred.model_name] = adjusted_weight
                total_inv_vol += adjusted_weight
            
            # Normalize weights
            if total_inv_vol > 0:
                weights = {k: v/total_inv_vol for k, v in weights.items()}
            else:
                weights = {pred.model_name: 1.0 / len(predictions) for pred in predictions}
            
            metadata = {
                'strategy': 'volatility_adjusted',
                'avg_volatility': np.mean([p.predicted_volatility for p in predictions]),
                'volatility_range': (min([p.predicted_volatility for p in predictions]), 
                                   max([p.predicted_volatility for p in predictions]))
            }
            
            return weights, metadata
            
        except Exception as e:
            logger.error(f"Error in volatility adjusted ensemble: {e}")
            return self._equal_weight_ensemble(predictions, regime, regime_confidence)
    
    def _regime_aware_ensemble(self, predictions: List[ForexModelPrediction], 
                             regime: ForexMarketRegime, regime_confidence: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Regime-aware ensemble strategy"""
        try:
            weights = {}
            total_regime_score = 0.0
            
            for pred in predictions:
                # Base weight from recent performance
                base_weight = 0.5
                if pred.model_name in self.model_performance:
                    perf = self.model_performance[pred.model_name]
                    base_weight = perf.recent_accuracy_30d
                
                # Regime-specific adjustment
                regime_multiplier = 1.0
                if regime in pred.regime_performance:
                    regime_performance = pred.regime_performance[regime]
                    regime_multiplier = 0.5 + regime_performance  # Scale from 0.5 to 1.5
                
                # Confidence in regime detection
                regime_weight = regime_confidence * regime_multiplier + (1 - regime_confidence) * 1.0
                
                # Model's confidence in current regime
                model_regime_confidence = pred.regime_confidence.get(regime, 0.5)
                
                # Final weight calculation
                final_weight = base_weight * regime_weight * model_regime_confidence
                
                weights[pred.model_name] = final_weight
                total_regime_score += final_weight
            
            # Normalize weights
            if total_regime_score > 0:
                weights = {k: v/total_regime_score for k, v in weights.items()}
            else:
                weights = {pred.model_name: 1.0 / len(predictions) for pred in predictions}
            
            metadata = {
                'strategy': 'regime_aware',
                'regime': regime.value,
                'regime_confidence': regime_confidence,
                'regime_adjustment_applied': True
            }
            
            return weights, metadata
            
        except Exception as e:
            logger.error(f"Error in regime aware ensemble: {e}")
            return self._performance_weighted_ensemble(predictions, regime, regime_confidence)
    
    def _dynamic_selection_ensemble(self, predictions: List[ForexModelPrediction], 
                                  regime: ForexMarketRegime, regime_confidence: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Dynamic model selection ensemble strategy"""
        try:
            # Select top performing models for current regime
            model_scores = []
            
            for pred in predictions:
                score = 0.5  # Default score
                
                if pred.model_name in self.model_performance:
                    perf = self.model_performance[pred.model_name]
                    
                    # Base score from recent performance
                    score = perf.recent_accuracy_7d * 0.6 + perf.recent_accuracy_30d * 0.4
                    
                    # Regime-specific boost
                    if regime in perf.regime_accuracy and perf.regime_predictions[regime] > 3:
                        regime_boost = (perf.regime_accuracy[regime] - 0.5) * 0.5
                        score += regime_boost
                    
                    # Confidence boost
                    confidence_boost = (pred.confidence - 0.5) * 0.2
                    score += confidence_boost
                    
                    # Stability penalty
                    stability_penalty = (1.0 - perf.prediction_consistency) * 0.1
                    score -= stability_penalty
                
                model_scores.append((pred.model_name, score))
            
            # Sort by score and select top models
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Select top 60% of models or at least min_models
            num_selected = max(self.min_models, int(len(model_scores) * 0.6))
            num_selected = min(num_selected, self.max_models)
            
            selected_models = model_scores[:num_selected]
            
            # Assign weights based on scores
            weights = {}
            total_score = sum(score for _, score in selected_models)
            
            if total_score > 0:
                for model_name, score in selected_models:
                    weights[model_name] = score / total_score
            else:
                for model_name, _ in selected_models:
                    weights[model_name] = 1.0 / len(selected_models)
            
            # Set zero weights for non-selected models
            for pred in predictions:
                if pred.model_name not in weights:
                    weights[pred.model_name] = 0.0
            
            metadata = {
                'strategy': 'dynamic_selection',
                'models_selected': len(selected_models),
                'selection_threshold': 0.6,
                'top_model': selected_models[0][0] if selected_models else None,
                'top_score': selected_models[0][1] if selected_models else 0.0
            }
            
            return weights, metadata
            
        except Exception as e:
            logger.error(f"Error in dynamic selection ensemble: {e}")
            return self._performance_weighted_ensemble(predictions, regime, regime_confidence)
    
    def _confidence_weighted_ensemble(self, predictions: List[ForexModelPrediction], 
                                    regime: ForexMarketRegime, regime_confidence: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Confidence-weighted ensemble strategy"""
        try:
            weights = {}
            total_confidence = 0.0
            
            for pred in predictions:
                # Use model confidence with performance adjustment
                base_confidence = pred.confidence
                
                # Adjust confidence based on historical accuracy
                if pred.model_name in self.model_performance:
                    perf = self.model_performance[pred.model_name]
                    accuracy_adjustment = (perf.accuracy - 0.5) * 0.3  # Scale adjustment
                    adjusted_confidence = base_confidence + accuracy_adjustment
                else:
                    adjusted_confidence = base_confidence
                
                # Ensure confidence is in valid range
                adjusted_confidence = max(0.1, min(0.95, adjusted_confidence))
                
                weights[pred.model_name] = adjusted_confidence
                total_confidence += adjusted_confidence
            
            # Normalize weights
            if total_confidence > 0:
                weights = {k: v/total_confidence for k, v in weights.items()}
            else:
                weights = {pred.model_name: 1.0 / len(predictions) for pred in predictions}
            
            metadata = {
                'strategy': 'confidence_weighted',
                'avg_confidence': total_confidence / len(predictions),
                'confidence_range': (min([p.confidence for p in predictions]), 
                                   max([p.confidence for p in predictions]))
            }
            
            return weights, metadata
            
        except Exception as e:
            logger.error(f"Error in confidence weighted ensemble: {e}")
            return self._equal_weight_ensemble(predictions, regime, regime_confidence)
    
    def _correlation_adjusted_ensemble(self, predictions: List[ForexModelPrediction], 
                                     regime: ForexMarketRegime, regime_confidence: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Correlation-adjusted ensemble strategy"""
        try:
            # Start with performance weights
            base_weights, _ = self._performance_weighted_ensemble(predictions, regime, regime_confidence)
            
            # Adjust for correlation (simplified - in practice, use historical correlation matrix)
            # Models with similar predictions get reduced weights to avoid over-concentration
            
            prices = [pred.predicted_price for pred in predictions]
            price_mean = np.mean(prices)
            
            adjusted_weights = {}
            for pred in predictions:
                base_weight = base_weights[pred.model_name]
                
                # Penalty for predictions far from consensus
                price_deviation = abs(pred.predicted_price - price_mean) / price_mean if price_mean > 0 else 0
                correlation_penalty = min(0.3, price_deviation * 2)  # Max 30% penalty
                
                adjusted_weight = base_weight * (1 - correlation_penalty)
                adjusted_weights[pred.model_name] = adjusted_weight
            
            # Renormalize
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {k: v/total_weight for k, v in adjusted_weights.items()}
            else:
                adjusted_weights = base_weights
            
            metadata = {
                'strategy': 'correlation_adjusted',
                'price_std': np.std(prices),
                'max_deviation': max([abs(p - price_mean) for p in prices]) / price_mean if price_mean > 0 else 0
            }
            
            return adjusted_weights, metadata
            
        except Exception as e:
            logger.error(f"Error in correlation adjusted ensemble: {e}")
            return self._performance_weighted_ensemble(predictions, regime, regime_confidence)
    
    def _adaptive_learning_ensemble(self, predictions: List[ForexModelPrediction], 
                                  regime: ForexMarketRegime, regime_confidence: float) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """Adaptive learning ensemble strategy"""
        try:
            # Combine multiple strategies with adaptive weights
            strategies_to_combine = [
                EnsembleStrategy.PERFORMANCE_WEIGHTED,
                EnsembleStrategy.REGIME_AWARE,
                EnsembleStrategy.CONFIDENCE_WEIGHTED
            ]
            
            strategy_weights = {
                EnsembleStrategy.PERFORMANCE_WEIGHTED: 0.4,
                EnsembleStrategy.REGIME_AWARE: 0.4,
                EnsembleStrategy.CONFIDENCE_WEIGHTED: 0.2
            }
            
            # Adjust strategy weights based on recent performance
            if hasattr(self, 'strategy_performance') and self.strategy_performance:
                total_perf = sum(self.strategy_performance.values())
                if total_perf > 0:
                    for strategy in strategies_to_combine:
                        if strategy in self.strategy_performance:
                            strategy_weights[strategy] = self.strategy_performance[strategy] / total_perf
            
            # Get weights from each strategy
            combined_weights = {pred.model_name: 0.0 for pred in predictions}
            
            for strategy in strategies_to_combine:
                if strategy in self.ensemble_strategies:
                    strategy_func = self.ensemble_strategies[strategy]
                    weights, _ = strategy_func(predictions, regime, regime_confidence)
                    
                    strategy_weight = strategy_weights[strategy]
                    for model_name, weight in weights.items():
                        combined_weights[model_name] += strategy_weight * weight
            
            metadata = {
                'strategy': 'adaptive_learning',
                'strategy_weights': strategy_weights,
                'strategies_combined': [s.value for s in strategies_to_combine]
            }
            
            return combined_weights, metadata
            
        except Exception as e:
            logger.error(f"Error in adaptive learning ensemble: {e}")
            return self._regime_aware_ensemble(predictions, regime, regime_confidence)
    
    def predict_ensemble(self, predictions: List[ForexModelPrediction], 
                        prices: List[float], currency_pair: str,
                        strategy: EnsembleStrategy = None,
                        economic_data: Dict[str, float] = None) -> ForexEnsembleResult:
        """Generate ensemble prediction"""
        try:
            if len(predictions) < self.min_models:
                logger.warning(f"Insufficient models for ensemble: {len(predictions)} < {self.min_models}")
                return ForexEnsembleResult(
                    timestamp=datetime.now(),
                    currency_pair=currency_pair,
                    ensemble_strategy=strategy or self.default_strategy,
                    market_regime=ForexMarketRegime.RANGING_LOW_VOL,
                    ensemble_price=0.0,
                    ensemble_direction=0,
                    ensemble_confidence=0.0
                )
            
            # Detect market regime
            regime, regime_confidence = self.regime_detector.detect_regime(prices, economic_data=economic_data)
            
            # Select ensemble strategy
            if strategy is None:
                strategy = self.default_strategy
            
            # Get ensemble weights
            if strategy in self.ensemble_strategies:
                model_weights, strategy_metadata = self.ensemble_strategies[strategy](predictions, regime, regime_confidence)
            else:
                logger.warning(f"Unknown strategy {strategy}, using equal weight")
                model_weights, strategy_metadata = self._equal_weight_ensemble(predictions, regime, regime_confidence)
            
            # Calculate ensemble predictions
            ensemble_price = 0.0
            ensemble_confidence = 0.0
            direction_votes = {-1: 0.0, 0: 0.0, 1: 0.0}
            
            total_weight = sum(model_weights.values())
            
            for pred in predictions:
                weight = model_weights.get(pred.model_name, 0.0)
                if weight > 0 and total_weight > 0:
                    normalized_weight = weight / total_weight
                    
                    ensemble_price += normalized_weight * pred.predicted_price
                    ensemble_confidence += normalized_weight * pred.confidence
                    direction_votes[pred.predicted_direction] += normalized_weight
            
            # Determine ensemble direction
            ensemble_direction = max(direction_votes.keys(), key=lambda k: direction_votes[k])
            
            # Calculate consensus metrics
            prices_list = [p.predicted_price for p in predictions]
            directions_list = [p.predicted_direction for p in predictions]
            confidences_list = [p.confidence for p in predictions]
            
            # Price consensus
            price_std = np.std(prices_list)
            price_mean = np.mean(prices_list)
            price_consensus = 1.0 - (price_std / (price_mean + 1e-8)) if price_mean > 0 else 0.0
            price_consensus = max(0.0, min(1.0, price_consensus))
            
            # Direction consensus
            direction_counts = {-1: 0, 0: 0, 1: 0}
            for direction in directions_list:
                direction_counts[direction] += 1
            max_direction_count = max(direction_counts.values())
            direction_consensus = max_direction_count / len(directions_list)
            
            # Confidence consensus
            confidence_consensus = 1.0 - (np.std(confidences_list) / (np.mean(confidences_list) + 1e-8))
            confidence_consensus = max(0.0, min(1.0, confidence_consensus))
            
            # Risk metrics
            ensemble_volatility = np.mean([p.predicted_volatility for p in predictions if p.predicted_volatility > 0])
            if ensemble_volatility == 0:
                ensemble_volatility = 0.01  # Default volatility
            
            ensemble_var = ensemble_price * 0.05 * ensemble_volatility  # 5% VaR
            model_disagreement = 1.0 - price_consensus
            prediction_uncertainty = 1.0 - ensemble_confidence
            
            # Regime analysis
            regime_stability = self.regime_detector.get_regime_stability()
            regime_transitions = self.regime_detector.predict_regime_transition(regime)
            
            # Trading recommendations
            current_price = prices[-1] if prices else ensemble_price
            signal_strength = abs(ensemble_price - current_price) / current_price if current_price > 0 else 0.0
            
            # Decision logic
            if ensemble_confidence > 0.8 and price_consensus > 0.7 and regime_confidence > 0.6:
                if signal_strength > 0.015:
                    recommended_action = "STRONG_BUY" if ensemble_direction == 1 else "STRONG_SELL" if ensemble_direction == -1 else "HOLD"
                    position_size = min(1.0, ensemble_confidence * price_consensus * regime_confidence)
                else:
                    recommended_action = "BUY" if ensemble_direction == 1 else "SELL" if ensemble_direction == -1 else "HOLD"
                    position_size = min(0.7, ensemble_confidence * price_consensus)
            elif ensemble_confidence > 0.6 and price_consensus > 0.5:
                recommended_action = "WEAK_BUY" if ensemble_direction == 1 else "WEAK_SELL" if ensemble_direction == -1 else "HOLD"
                position_size = min(0.4, ensemble_confidence * price_consensus)
            else:
                recommended_action = "HOLD"
                position_size = 0.0
            
            # Stop loss and take profit
            stop_loss = None
            take_profit = None
            
            if recommended_action in ["STRONG_BUY", "BUY", "WEAK_BUY"]:
                stop_loss = ensemble_price * (1 - 2 * ensemble_volatility)
                take_profit = ensemble_price * (1 + 3 * ensemble_volatility)
            elif recommended_action in ["STRONG_SELL", "SELL", "WEAK_SELL"]:
                stop_loss = ensemble_price * (1 + 2 * ensemble_volatility)
                take_profit = ensemble_price * (1 - 3 * ensemble_volatility)
            
            # Quality metrics
            prediction_quality = (ensemble_confidence + price_consensus + regime_confidence) / 3
            ensemble_stability = (price_consensus + confidence_consensus + regime_stability) / 3
            
            # Key factors
            key_factors = {}
            for pred in predictions:
                for factor, importance in pred.model_features.items():
                    weight = model_weights.get(pred.model_name, 0.0)
                    if factor in key_factors:
                        key_factors[factor] += weight * importance
                    else:
                        key_factors[factor] = weight * importance
            
            # Risk warnings and opportunities
            risk_warnings = []
            opportunities = []
            
            if model_disagreement > 0.5:
                risk_warnings.append("High model disagreement")
            if regime_confidence < 0.5:
                risk_warnings.append("Uncertain market regime")
            if ensemble_volatility > 0.02:
                risk_warnings.append("High predicted volatility")
            
            if price_consensus > 0.8:
                opportunities.append("Strong price consensus")
            if regime_confidence > 0.8:
                opportunities.append("Clear market regime")
            if ensemble_confidence > 0.8:
                opportunities.append("High model confidence")
            
            # Create result
            result = ForexEnsembleResult(
                timestamp=datetime.now(),
                currency_pair=currency_pair,
                ensemble_strategy=strategy,
                market_regime=regime,
                ensemble_price=ensemble_price,
                ensemble_direction=ensemble_direction,
                ensemble_confidence=ensemble_confidence,
                model_predictions={pred.model_name: pred for pred in predictions},
                model_weights=model_weights,
                price_consensus=price_consensus,
                direction_consensus=direction_consensus,
                confidence_consensus=confidence_consensus,
                ensemble_volatility=ensemble_volatility,
                ensemble_var=ensemble_var,
                model_disagreement=model_disagreement,
                prediction_uncertainty=prediction_uncertainty,
                regime_confidence=regime_confidence,
                regime_stability=regime_stability,
                regime_transition_probability=regime_transitions,
                recommended_action=recommended_action,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                prediction_quality=prediction_quality,
                ensemble_stability=ensemble_stability,
                key_factors=key_factors,
                risk_warnings=risk_warnings,
                opportunities=opportunities
            )
            
            # Store prediction for performance tracking
            self.prediction_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating ensemble prediction: {e}")
            return ForexEnsembleResult(
                timestamp=datetime.now(),
                currency_pair=currency_pair,
                ensemble_strategy=strategy or self.default_strategy,
                market_regime=ForexMarketRegime.RANGING_LOW_VOL,
                ensemble_price=0.0,
                ensemble_direction=0,
                ensemble_confidence=0.0
            )
    
    def get_ensemble_performance(self) -> Dict[str, Any]:
        """Get ensemble performance metrics"""
        try:
            if not self.prediction_history:
                return {'status': 'no_data'}
            
            recent_predictions = list(self.prediction_history)[-50:]  # Last 50 predictions
            
            # Calculate metrics
            avg_confidence = np.mean([p.ensemble_confidence for p in recent_predictions])
            avg_consensus = np.mean([p.price_consensus for p in recent_predictions])
            avg_quality = np.mean([p.prediction_quality for p in recent_predictions])
            avg_stability = np.mean([p.ensemble_stability for p in recent_predictions])
            
            # Strategy distribution
            strategy_counts = {}
            for pred in recent_predictions:
                strategy = pred.ensemble_strategy.value
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            # Regime distribution
            regime_counts = {}
            for pred in recent_predictions:
                regime = pred.market_regime.value
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            return {
                'status': 'success',
                'total_predictions': len(self.prediction_history),
                'recent_predictions': len(recent_predictions),
                'average_confidence': avg_confidence,
                'average_consensus': avg_consensus,
                'average_quality': avg_quality,
                'average_stability': avg_stability,
                'strategy_distribution': strategy_counts,
                'regime_distribution': regime_counts,
                'models_tracked': len(self.model_performance)
            }
            
        except Exception as e:
            logger.error(f"Error getting ensemble performance: {e}")
            return {'status': 'error', 'message': str(e)}

# Example usage
if __name__ == "__main__":
    print("=== Forex Advanced Ensemble Methods Demo ===")
    
    # Initialize ensemble predictor
    ensemble_predictor = ForexAdvancedEnsemblePredictor(
        default_strategy=EnsembleStrategy.REGIME_AWARE,
        performance_window=30,
        min_models=2,
        max_models=8
    )
    
    print(f"\nInitialized ensemble predictor with {ensemble_predictor.default_strategy.value} strategy")
    
    # Generate sample predictions from different models
    np.random.seed(42)
    
    base_price = 1.1000  # EUR/USD
    sample_prices = []
    
    # Generate price history
    for i in range(100):
        price_change = np.random.normal(0, 0.001)
        base_price *= (1 + price_change)
        sample_prices.append(base_price)
    
    current_price = sample_prices[-1]
    
    # Create sample model predictions
    sample_predictions = [
        ForexModelPrediction(
            model_name="PPP_Model",
            model_type="fundamental",
            timestamp=datetime.now(),
            currency_pair="EURUSD",
            predicted_price=current_price * (1 + np.random.normal(0, 0.005)),
            predicted_direction=np.random.choice([-1, 0, 1]),
            confidence=0.6 + 0.3 * np.random.random(),
            predicted_volatility=0.01 + 0.005 * np.random.random(),
            recent_accuracy=0.65,
            regime_performance={
                ForexMarketRegime.TRENDING_BULL: 0.7,
                ForexMarketRegime.RANGING_LOW_VOL: 0.6
            }
        ),
        ForexModelPrediction(
            model_name="LSTM_Model",
            model_type="machine_learning",
            timestamp=datetime.now(),
            currency_pair="EURUSD",
            predicted_price=current_price * (1 + np.random.normal(0, 0.008)),
            predicted_direction=np.random.choice([-1, 0, 1]),
            confidence=0.75 + 0.2 * np.random.random(),
            predicted_volatility=0.012 + 0.003 * np.random.random(),
            recent_accuracy=0.78,
            regime_performance={
                ForexMarketRegime.TRENDING_BULL: 0.8,
                ForexMarketRegime.RANGING_HIGH_VOL: 0.75
            }
        ),
        ForexModelPrediction(
            model_name="Technical_Model",
            model_type="technical",
            timestamp=datetime.now(),
            currency_pair="EURUSD",
            predicted_price=current_price * (1 + np.random.normal(0, 0.006)),
            predicted_direction=np.random.choice([-1, 0, 1]),
            confidence=0.55 + 0.25 * np.random.random(),
            predicted_volatility=0.008 + 0.004 * np.random.random(),
            recent_accuracy=0.58,
            regime_performance={
                ForexMarketRegime.TRENDING_BULL: 0.6,
                ForexMarketRegime.BREAKOUT: 0.7
            }
        ),
        ForexModelPrediction(
            model_name="Sentiment_Model",
            model_type="sentiment",
            timestamp=datetime.now(),
            currency_pair="EURUSD",
            predicted_price=current_price * (1 + np.random.normal(0, 0.004)),
            predicted_direction=np.random.choice([-1, 0, 1]),
            confidence=0.5 + 0.3 * np.random.random(),
            predicted_volatility=0.015 + 0.005 * np.random.random(),
            recent_accuracy=0.52,
            regime_performance={
                ForexMarketRegime.CRISIS: 0.65,
                ForexMarketRegime.RISK_OFF: 0.7
            }
        )
    ]
    
    print(f"\nGenerated {len(sample_predictions)} model predictions")
    for pred in sample_predictions:
        print(f"  {pred.model_name}: {pred.predicted_price:.5f} (conf: {pred.confidence:.3f})")
    
    # Test different ensemble strategies
    strategies_to_test = [
        EnsembleStrategy.EQUAL_WEIGHT,
        EnsembleStrategy.PERFORMANCE_WEIGHTED,
        EnsembleStrategy.REGIME_AWARE,
        EnsembleStrategy.DYNAMIC_SELECTION,
        EnsembleStrategy.CONFIDENCE_WEIGHTED
    ]
    
    print("\n=== Ensemble Strategy Comparison ===")
    
    economic_data = {
        'vix': 18.5,
        'intervention_prob': 0.1,
        'interest_rate_diff': 0.025
    }
    
    for strategy in strategies_to_test:
        result = ensemble_predictor.predict_ensemble(
            predictions=sample_predictions,
            prices=sample_prices,
            currency_pair="EURUSD",
            strategy=strategy,
            economic_data=economic_data
        )
        
        print(f"\n{strategy.value}:")
        print(f"  Ensemble Price: {result.ensemble_price:.5f}")
        print(f"  Direction: {result.ensemble_direction}")
        print(f"  Confidence: {result.ensemble_confidence:.3f}")
        print(f"  Price Consensus: {result.price_consensus:.3f}")
        print(f"  Market Regime: {result.market_regime.value}")
        print(f"  Regime Confidence: {result.regime_confidence:.3f}")
        print(f"  Recommended Action: {result.recommended_action}")
        print(f"  Position Size: {result.position_size:.3f}")
        print(f"  Prediction Quality: {result.prediction_quality:.3f}")
        
        # Show top model weights
        sorted_weights = sorted(result.model_weights.items(), key=lambda x: x[1], reverse=True)
        print(f"  Top Model Weights:")
        for model, weight in sorted_weights[:3]:
            print(f"    {model}: {weight:.3f}")
    
    # Update model performance with simulated results
    print("\n=== Updating Model Performance ===")
    
    for pred in sample_predictions:
        # Simulate actual price (small random walk from predicted)
        actual_price = pred.predicted_price * (1 + np.random.normal(0, 0.002))
        
        ensemble_predictor.update_model_performance(
            model_name=pred.model_name,
            actual_price=actual_price,
            predicted_price=pred.predicted_price,
            prediction_time=pred.timestamp,
            regime=ForexMarketRegime.TRENDING_BULL
        )
        
        print(f"Updated performance for {pred.model_name}")
    
    # Show performance summary
    print("\n=== Performance Summary ===")
    
    # Generate more predictions for history
    for i in range(10):
        test_result = ensemble_predictor.predict_ensemble(
            predictions=sample_predictions,
            prices=sample_prices[-(50+i*5):],
            currency_pair="EURUSD",
            economic_data=economic_data
        )
    
    performance = ensemble_predictor.get_ensemble_performance()
    
    if performance['status'] == 'success':
        print(f"\nEnsemble Performance:")
        print(f"  Total Predictions: {performance['total_predictions']}")
        print(f"  Average Confidence: {performance['average_confidence']:.3f}")
        print(f"  Average Consensus: {performance['average_consensus']:.3f}")
        print(f"  Average Quality: {performance['average_quality']:.3f}")
        print(f"  Average Stability: {performance['average_stability']:.3f}")
        print(f"  Models Tracked: {performance['models_tracked']}")
        
        print(f"\nStrategy Distribution:")
        for strategy, count in performance['strategy_distribution'].items():
            print(f"  {strategy}: {count}")
        
        print(f"\nRegime Distribution:")
        for regime, count in performance['regime_distribution'].items():
            print(f"  {regime}: {count}")
    
    print("\n=== Forex Advanced Ensemble Methods Demo Complete ===")