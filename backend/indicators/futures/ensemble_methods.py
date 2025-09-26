import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FuturesMarketRegime(Enum):
    CONTANGO = "contango"
    BACKWARDATION = "backwardation"
    NEUTRAL = "neutral"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"

class EnsembleStrategy(Enum):
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    REGIME_AWARE = "regime_aware"
    DYNAMIC_BAYESIAN = "dynamic_bayesian"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    SHARPE_WEIGHTED = "sharpe_weighted"

@dataclass
class FuturesModelPrediction:
    """Individual model prediction for futures"""
    model_name: str
    contract_symbol: str
    prediction_value: float
    confidence: float
    timestamp: datetime
    
    # Futures-specific predictions
    price_target: Optional[float] = None
    direction_signal: Optional[int] = None  # -1, 0, 1
    volatility_forecast: Optional[float] = None
    basis_forecast: Optional[float] = None
    roll_yield_forecast: Optional[float] = None
    
    # Model performance metrics
    recent_accuracy: float = 0.5
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.5
    
    # Risk metrics
    var_95: Optional[float] = None
    expected_shortfall: Optional[float] = None
    
    # Model metadata
    model_type: str = "unknown"
    prediction_horizon: int = 1  # days
    data_quality: float = 1.0

@dataclass
class FuturesModelPerformance:
    """Performance tracking for futures models"""
    model_name: str
    contract_symbol: str
    evaluation_period: int  # days
    
    # Accuracy metrics
    direction_accuracy: float = 0.0
    price_accuracy: float = 0.0  # MAPE
    volatility_accuracy: float = 0.0
    
    # Risk-adjusted performance
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Futures-specific performance
    basis_prediction_accuracy: float = 0.0
    roll_yield_prediction_accuracy: float = 0.0
    term_structure_accuracy: float = 0.0
    
    # Regime-specific performance
    contango_performance: float = 0.0
    backwardation_performance: float = 0.0
    high_vol_performance: float = 0.0
    low_vol_performance: float = 0.0
    
    # Consistency metrics
    performance_stability: float = 0.0
    drawdown_recovery: float = 0.0
    
    # Recent performance
    recent_hit_rate: float = 0.5
    recent_profit_factor: float = 1.0
    
    # Performance history
    daily_returns: List[float] = field(default_factory=list)
    prediction_errors: List[float] = field(default_factory=list)
    confidence_calibration: float = 0.0

@dataclass
class FuturesEnsembleResult:
    """Result from futures ensemble prediction"""
    contract_symbol: str
    ensemble_strategy: EnsembleStrategy
    consensus_prediction: float
    consensus_confidence: float
    timestamp: datetime
    
    # Individual model contributions
    model_predictions: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)
    
    # Ensemble metrics
    prediction_variance: float = 0.0
    model_agreement: float = 0.0
    ensemble_uncertainty: float = 0.0
    
    # Futures-specific ensemble results
    consensus_price_target: Optional[float] = None
    consensus_direction: Optional[int] = None
    consensus_volatility: Optional[float] = None
    basis_consensus: Optional[float] = None
    
    # Market regime analysis
    detected_regime: FuturesMarketRegime = FuturesMarketRegime.NEUTRAL
    regime_confidence: float = 0.0
    regime_stability: float = 0.0
    
    # Risk assessment
    ensemble_var_95: Optional[float] = None
    ensemble_expected_shortfall: Optional[float] = None
    tail_risk_estimate: float = 0.0
    
    # Strategy performance
    expected_sharpe: float = 0.0
    expected_max_drawdown: float = 0.0
    
    # Actionable insights
    trading_signal: str = "NEUTRAL"
    signal_strength: float = 0.0  # 0-1 scale
    recommended_position_size: float = 0.0  # 0-1 scale
    
    # Ensemble diagnostics
    weight_entropy: float = 0.0  # Measure of weight concentration
    prediction_stability: float = 0.0
    model_correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

class FuturesAdvancedEnsemblePredictor:
    """Advanced ensemble prediction system for futures trading"""
    
    def __init__(self,
                 lookback_period: int = 252,
                 min_performance_history: int = 30,
                 regime_detection_window: int = 60,
                 volatility_threshold: float = 0.02,
                 trend_threshold: float = 0.01):
        """
        Initialize the advanced ensemble predictor
        
        Args:
            lookback_period: Days to look back for performance calculation
            min_performance_history: Minimum days of history required
            regime_detection_window: Window for regime detection
            volatility_threshold: Threshold for high/low volatility regimes
            trend_threshold: Threshold for trending/mean-reverting regimes
        """
        self.lookback_period = lookback_period
        self.min_performance_history = min_performance_history
        self.regime_detection_window = regime_detection_window
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        
        # Model performance tracking
        self.model_performance_history: Dict[str, FuturesModelPerformance] = {}
        self.prediction_history: List[Dict[str, Any]] = []
        
        # Regime detection parameters
        self.regime_weights = {
            FuturesMarketRegime.CONTANGO: {},
            FuturesMarketRegime.BACKWARDATION: {},
            FuturesMarketRegime.HIGH_VOLATILITY: {},
            FuturesMarketRegime.LOW_VOLATILITY: {},
            FuturesMarketRegime.TRENDING: {},
            FuturesMarketRegime.MEAN_REVERTING: {},
            FuturesMarketRegime.NEUTRAL: {}
        }
        
        # Ensemble strategies
        self.ensemble_strategies = {
            EnsembleStrategy.EQUAL_WEIGHT: self._equal_weight_ensemble,
            EnsembleStrategy.PERFORMANCE_WEIGHTED: self._performance_weighted_ensemble,
            EnsembleStrategy.VOLATILITY_ADJUSTED: self._volatility_adjusted_ensemble,
            EnsembleStrategy.REGIME_AWARE: self._regime_aware_ensemble,
            EnsembleStrategy.DYNAMIC_BAYESIAN: self._dynamic_bayesian_ensemble,
            EnsembleStrategy.CONFIDENCE_WEIGHTED: self._confidence_weighted_ensemble,
            EnsembleStrategy.SHARPE_WEIGHTED: self._sharpe_weighted_ensemble
        }
    
    def detect_market_regime(self, 
                           prices: List[float], 
                           volumes: List[float],
                           basis_data: Optional[List[float]] = None,
                           returns: Optional[List[float]] = None) -> Tuple[FuturesMarketRegime, float]:
        """Detect current market regime"""
        try:
            if len(prices) < self.regime_detection_window:
                return FuturesMarketRegime.NEUTRAL, 0.5
            
            recent_prices = prices[-self.regime_detection_window:]
            recent_returns = returns[-self.regime_detection_window:] if returns else np.diff(recent_prices) / recent_prices[:-1]
            
            # Volatility regime
            volatility = np.std(recent_returns)
            
            # Trend regime
            trend_slope, _, r_value, _, _ = stats.linregress(range(len(recent_prices)), recent_prices)
            trend_strength = abs(r_value)
            
            # Basis regime (if available)
            basis_regime = FuturesMarketRegime.NEUTRAL
            basis_confidence = 0.5
            
            if basis_data and len(basis_data) >= 10:
                recent_basis = basis_data[-10:]
                avg_basis = np.mean(recent_basis)
                
                if avg_basis > 0.02:
                    basis_regime = FuturesMarketRegime.CONTANGO
                    basis_confidence = min(0.9, abs(avg_basis) * 10)
                elif avg_basis < -0.02:
                    basis_regime = FuturesMarketRegime.BACKWARDATION
                    basis_confidence = min(0.9, abs(avg_basis) * 10)
            
            # Determine primary regime
            regime_scores = {
                FuturesMarketRegime.HIGH_VOLATILITY: volatility / self.volatility_threshold if volatility > self.volatility_threshold else 0,
                FuturesMarketRegime.LOW_VOLATILITY: (self.volatility_threshold / volatility) if volatility < self.volatility_threshold else 0,
                FuturesMarketRegime.TRENDING: trend_strength if abs(trend_slope) > self.trend_threshold else 0,
                FuturesMarketRegime.MEAN_REVERTING: (1 - trend_strength) if abs(trend_slope) < self.trend_threshold else 0,
                basis_regime: basis_confidence if basis_regime != FuturesMarketRegime.NEUTRAL else 0
            }
            
            # Select regime with highest score
            best_regime = max(regime_scores.items(), key=lambda x: x[1])
            
            if best_regime[1] > 0.3:
                return best_regime[0], min(0.95, best_regime[1])
            else:
                return FuturesMarketRegime.NEUTRAL, 0.5
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return FuturesMarketRegime.NEUTRAL, 0.5
    
    def update_model_performance(self, 
                               model_name: str,
                               contract_symbol: str,
                               actual_values: List[float],
                               predicted_values: List[float],
                               prediction_dates: List[datetime]):
        """Update performance tracking for a model"""
        try:
            if len(actual_values) != len(predicted_values) or len(actual_values) < 2:
                return
            
            # Calculate performance metrics
            actual_returns = np.diff(actual_values) / actual_values[:-1]
            predicted_returns = np.diff(predicted_values) / predicted_values[:-1]
            
            # Direction accuracy
            actual_directions = np.sign(actual_returns)
            predicted_directions = np.sign(predicted_returns)
            direction_accuracy = np.mean(actual_directions == predicted_directions)
            
            # Price accuracy (MAPE)
            price_accuracy = 1.0 - np.mean(np.abs((actual_values - predicted_values) / actual_values))
            price_accuracy = max(0, price_accuracy)
            
            # Risk-adjusted metrics
            if np.std(actual_returns) > 0:
                sharpe_ratio = np.mean(actual_returns) / np.std(actual_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Create or update performance record
            performance = FuturesModelPerformance(
                model_name=model_name,
                contract_symbol=contract_symbol,
                evaluation_period=len(actual_values),
                direction_accuracy=direction_accuracy,
                price_accuracy=price_accuracy,
                sharpe_ratio=sharpe_ratio,
                recent_hit_rate=direction_accuracy,
                daily_returns=actual_returns.tolist(),
                prediction_errors=(np.array(actual_values) - np.array(predicted_values)).tolist()
            )
            
            self.model_performance_history[f"{model_name}_{contract_symbol}"] = performance
            
        except Exception as e:
            logger.error(f"Error updating model performance: {e}")
    
    def _equal_weight_ensemble(self, predictions: List[FuturesModelPrediction]) -> Dict[str, float]:
        """Equal weight ensemble strategy"""
        if not predictions:
            return {}
        
        weight = 1.0 / len(predictions)
        return {pred.model_name: weight for pred in predictions}
    
    def _performance_weighted_ensemble(self, predictions: List[FuturesModelPrediction]) -> Dict[str, float]:
        """Performance-based weighting"""
        weights = {}
        total_performance = 0
        
        for pred in predictions:
            performance_key = f"{pred.model_name}_{pred.contract_symbol}"
            if performance_key in self.model_performance_history:
                perf = self.model_performance_history[performance_key]
                # Combine accuracy and risk-adjusted return
                performance_score = (perf.direction_accuracy * 0.5 + 
                                   max(0, perf.sharpe_ratio / 2) * 0.3 + 
                                   perf.price_accuracy * 0.2)
            else:
                performance_score = pred.recent_accuracy
            
            weights[pred.model_name] = max(0.01, performance_score)  # Minimum weight
            total_performance += weights[pred.model_name]
        
        # Normalize weights
        if total_performance > 0:
            weights = {k: v / total_performance for k, v in weights.items()}
        else:
            return self._equal_weight_ensemble(predictions)
        
        return weights
    
    def _volatility_adjusted_ensemble(self, predictions: List[FuturesModelPrediction]) -> Dict[str, float]:
        """Volatility-adjusted weighting"""
        weights = {}
        
        # Calculate inverse volatility weights
        volatilities = []
        for pred in predictions:
            if pred.volatility_forecast and pred.volatility_forecast > 0:
                volatilities.append(pred.volatility_forecast)
            else:
                volatilities.append(0.02)  # Default volatility
        
        # Inverse volatility weighting
        inv_vol_weights = [1.0 / vol for vol in volatilities]
        total_weight = sum(inv_vol_weights)
        
        for i, pred in enumerate(predictions):
            weights[pred.model_name] = inv_vol_weights[i] / total_weight
        
        return weights
    
    def _regime_aware_ensemble(self, 
                             predictions: List[FuturesModelPrediction],
                             current_regime: FuturesMarketRegime = FuturesMarketRegime.NEUTRAL) -> Dict[str, float]:
        """Regime-aware ensemble weighting"""
        weights = {}
        
        # Base weights from performance
        base_weights = self._performance_weighted_ensemble(predictions)
        
        # Regime-specific adjustments
        regime_adjustments = {
            FuturesMarketRegime.CONTANGO: {'technical': 0.8, 'fundamental': 1.2, 'ml': 1.1},
            FuturesMarketRegime.BACKWARDATION: {'technical': 1.1, 'fundamental': 1.3, 'ml': 0.9},
            FuturesMarketRegime.HIGH_VOLATILITY: {'technical': 1.2, 'fundamental': 0.8, 'ml': 1.1},
            FuturesMarketRegime.LOW_VOLATILITY: {'technical': 0.9, 'fundamental': 1.1, 'ml': 1.0},
            FuturesMarketRegime.TRENDING: {'technical': 1.3, 'fundamental': 0.9, 'ml': 1.1},
            FuturesMarketRegime.MEAN_REVERTING: {'technical': 0.8, 'fundamental': 1.2, 'ml': 1.0}
        }
        
        adjustments = regime_adjustments.get(current_regime, {})
        
        for pred in predictions:
            base_weight = base_weights.get(pred.model_name, 1.0 / len(predictions))
            
            # Apply regime adjustment based on model type
            adjustment = 1.0
            for model_type, adj_factor in adjustments.items():
                if model_type.lower() in pred.model_name.lower():
                    adjustment = adj_factor
                    break
            
            weights[pred.model_name] = base_weight * adjustment
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _dynamic_bayesian_ensemble(self, predictions: List[FuturesModelPrediction]) -> Dict[str, float]:
        """Dynamic Bayesian ensemble weighting"""
        weights = {}
        
        # Start with performance-based priors
        prior_weights = self._performance_weighted_ensemble(predictions)
        
        # Update with confidence-based likelihood
        for pred in predictions:
            prior = prior_weights.get(pred.model_name, 1.0 / len(predictions))
            likelihood = pred.confidence
            
            # Bayesian update (simplified)
            posterior = prior * likelihood
            weights[pred.model_name] = posterior
        
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _confidence_weighted_ensemble(self, predictions: List[FuturesModelPrediction]) -> Dict[str, float]:
        """Confidence-based weighting"""
        weights = {}
        total_confidence = sum(pred.confidence for pred in predictions)
        
        if total_confidence > 0:
            for pred in predictions:
                weights[pred.model_name] = pred.confidence / total_confidence
        else:
            return self._equal_weight_ensemble(predictions)
        
        return weights
    
    def _sharpe_weighted_ensemble(self, predictions: List[FuturesModelPrediction]) -> Dict[str, float]:
        """Sharpe ratio weighted ensemble"""
        weights = {}
        sharpe_ratios = []
        
        for pred in predictions:
            if pred.sharpe_ratio > 0:
                sharpe_ratios.append(pred.sharpe_ratio)
            else:
                sharpe_ratios.append(0.1)  # Minimum Sharpe ratio
        
        total_sharpe = sum(sharpe_ratios)
        
        if total_sharpe > 0:
            for i, pred in enumerate(predictions):
                weights[pred.model_name] = sharpe_ratios[i] / total_sharpe
        else:
            return self._equal_weight_ensemble(predictions)
        
        return weights
    
    def predict_ensemble(self,
                        predictions: List[FuturesModelPrediction],
                        strategy: EnsembleStrategy = EnsembleStrategy.REGIME_AWARE,
                        market_data: Optional[Dict[str, List[float]]] = None) -> FuturesEnsembleResult:
        """Generate ensemble prediction using specified strategy"""
        try:
            if not predictions:
                return FuturesEnsembleResult(
                    contract_symbol="UNKNOWN",
                    ensemble_strategy=strategy,
                    consensus_prediction=0.0,
                    consensus_confidence=0.0,
                    timestamp=datetime.now()
                )
            
            contract_symbol = predictions[0].contract_symbol
            
            # Detect market regime if data available
            current_regime = FuturesMarketRegime.NEUTRAL
            regime_confidence = 0.5
            
            if market_data:
                prices = market_data.get('prices', [])
                volumes = market_data.get('volumes', [])
                basis = market_data.get('basis', [])
                returns = market_data.get('returns', [])
                
                current_regime, regime_confidence = self.detect_market_regime(
                    prices, volumes, basis, returns
                )
            
            # Get ensemble weights based on strategy
            if strategy == EnsembleStrategy.REGIME_AWARE:
                weights = self._regime_aware_ensemble(predictions, current_regime)
            else:
                ensemble_func = self.ensemble_strategies.get(strategy, self._equal_weight_ensemble)
                weights = ensemble_func(predictions)
            
            # Calculate consensus prediction
            consensus_prediction = sum(
                pred.prediction_value * weights.get(pred.model_name, 0)
                for pred in predictions
            )
            
            # Calculate consensus confidence
            consensus_confidence = sum(
                pred.confidence * weights.get(pred.model_name, 0)
                for pred in predictions
            )
            
            # Calculate model agreement
            pred_values = [pred.prediction_value for pred in predictions]
            if len(pred_values) > 1:
                model_agreement = 1.0 - (np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8))
                model_agreement = max(0.0, min(1.0, model_agreement))
            else:
                model_agreement = 1.0
            
            # Calculate ensemble uncertainty
            ensemble_uncertainty = np.std(pred_values) if len(pred_values) > 1 else 0.0
            
            # Consensus price target and direction
            price_targets = [pred.price_target for pred in predictions if pred.price_target is not None]
            consensus_price_target = np.mean(price_targets) if price_targets else None
            
            directions = [pred.direction_signal for pred in predictions if pred.direction_signal is not None]
            consensus_direction = int(np.sign(np.mean(directions))) if directions else None
            
            # Generate trading signal
            signal_strength = abs(consensus_prediction - 0.5) * 2  # Convert to 0-1 scale
            
            if consensus_prediction > 0.6:
                trading_signal = "BUY"
            elif consensus_prediction < 0.4:
                trading_signal = "SELL"
            else:
                trading_signal = "NEUTRAL"
            
            # Calculate weight entropy (measure of concentration)
            weight_values = list(weights.values())
            if weight_values:
                weight_entropy = -sum(w * np.log(w + 1e-8) for w in weight_values if w > 0)
            else:
                weight_entropy = 0
            
            # Extract individual predictions and confidences
            model_predictions = {pred.model_name: pred.prediction_value for pred in predictions}
            model_confidences = {pred.model_name: pred.confidence for pred in predictions}
            
            return FuturesEnsembleResult(
                contract_symbol=contract_symbol,
                ensemble_strategy=strategy,
                consensus_prediction=consensus_prediction,
                consensus_confidence=consensus_confidence,
                timestamp=datetime.now(),
                model_predictions=model_predictions,
                model_weights=weights,
                model_confidences=model_confidences,
                prediction_variance=np.var(pred_values) if pred_values else 0.0,
                model_agreement=model_agreement,
                ensemble_uncertainty=ensemble_uncertainty,
                consensus_price_target=consensus_price_target,
                consensus_direction=consensus_direction,
                detected_regime=current_regime,
                regime_confidence=regime_confidence,
                trading_signal=trading_signal,
                signal_strength=signal_strength,
                recommended_position_size=min(1.0, signal_strength * consensus_confidence),
                weight_entropy=weight_entropy
            )
            
        except Exception as e:
            logger.error(f"Error generating ensemble prediction: {e}")
            return FuturesEnsembleResult(
                contract_symbol=predictions[0].contract_symbol if predictions else "UNKNOWN",
                ensemble_strategy=strategy,
                consensus_prediction=0.0,
                consensus_confidence=0.0,
                timestamp=datetime.now()
            )
    
    def get_model_performance_summary(self) -> Dict[str, Dict[str, float]]:
        """Get performance summary for all tracked models"""
        summary = {}
        
        for key, performance in self.model_performance_history.items():
            summary[key] = {
                'direction_accuracy': performance.direction_accuracy,
                'price_accuracy': performance.price_accuracy,
                'sharpe_ratio': performance.sharpe_ratio,
                'recent_hit_rate': performance.recent_hit_rate
            }
        
        return summary

# Example usage
if __name__ == "__main__":
    # Create sample model predictions
    sample_predictions = [
        FuturesModelPrediction(
            model_name="LSTM_Model",
            contract_symbol="CL_2024_06",
            prediction_value=0.65,
            confidence=0.8,
            timestamp=datetime.now(),
            price_target=75.50,
            direction_signal=1,
            recent_accuracy=0.72,
            sharpe_ratio=1.2,
            model_type="ml"
        ),
        FuturesModelPrediction(
            model_name="Technical_Analysis",
            contract_symbol="CL_2024_06",
            prediction_value=0.58,
            confidence=0.7,
            timestamp=datetime.now(),
            price_target=74.80,
            direction_signal=1,
            recent_accuracy=0.68,
            sharpe_ratio=0.9,
            model_type="technical"
        ),
        FuturesModelPrediction(
            model_name="Fundamental_Model",
            contract_symbol="CL_2024_06",
            prediction_value=0.72,
            confidence=0.75,
            timestamp=datetime.now(),
            price_target=76.20,
            direction_signal=1,
            recent_accuracy=0.65,
            sharpe_ratio=0.8,
            model_type="fundamental"
        ),
        FuturesModelPrediction(
            model_name="Term_Structure_Model",
            contract_symbol="CL_2024_06",
            prediction_value=0.45,
            confidence=0.6,
            timestamp=datetime.now(),
            price_target=73.90,
            direction_signal=-1,
            recent_accuracy=0.63,
            sharpe_ratio=0.7,
            model_type="term_structure"
        )
    ]
    
    # Sample market data
    np.random.seed(42)
    n_points = 100
    base_price = 74.0
    prices = base_price + np.cumsum(np.random.normal(0, 0.5, n_points))
    
    market_data = {
        'prices': prices.tolist(),
        'volumes': np.random.lognormal(10, 0.5, n_points).tolist(),
        'basis': np.random.normal(0.02, 0.1, n_points).tolist(),
        'returns': (np.diff(prices, prepend=prices[0]) / prices[0]).tolist()
    }
    
    print("=== Futures Advanced Ensemble Methods Demo ===")
    
    # Initialize ensemble predictor
    ensemble_predictor = FuturesAdvancedEnsemblePredictor()
    
    # Test different ensemble strategies
    strategies = [
        EnsembleStrategy.EQUAL_WEIGHT,
        EnsembleStrategy.PERFORMANCE_WEIGHTED,
        EnsembleStrategy.VOLATILITY_ADJUSTED,
        EnsembleStrategy.REGIME_AWARE,
        EnsembleStrategy.CONFIDENCE_WEIGHTED,
        EnsembleStrategy.SHARPE_WEIGHTED
    ]
    
    print(f"\nTesting {len(strategies)} ensemble strategies...")
    
    for strategy in strategies:
        print(f"\n=== {strategy.value.upper()} STRATEGY ===")
        
        result = ensemble_predictor.predict_ensemble(
            predictions=sample_predictions,
            strategy=strategy,
            market_data=market_data
        )
        
        print(f"Consensus Prediction: {result.consensus_prediction:.3f}")
        print(f"Consensus Confidence: {result.consensus_confidence:.3f}")
        print(f"Trading Signal: {result.trading_signal}")
        print(f"Signal Strength: {result.signal_strength:.3f}")
        print(f"Model Agreement: {result.model_agreement:.3f}")
        print(f"Detected Regime: {result.detected_regime.value}")
        print(f"Recommended Position Size: {result.recommended_position_size:.3f}")
        
        print("Model Weights:")
        for model, weight in result.model_weights.items():
            print(f"  {model}: {weight:.3f}")
    
    # Test regime detection
    print(f"\n=== REGIME DETECTION TEST ===")
    regime, confidence = ensemble_predictor.detect_market_regime(
        prices=market_data['prices'],
        volumes=market_data['volumes'],
        basis_data=market_data['basis'],
        returns=market_data['returns']
    )
    
    print(f"Detected Regime: {regime.value}")
    print(f"Regime Confidence: {confidence:.3f}")
    
    # Performance tracking example
    print(f"\n=== PERFORMANCE TRACKING EXAMPLE ===")
    
    # Simulate some historical performance
    actual_values = [74.0, 74.5, 75.2, 74.8, 75.5, 76.0, 75.7, 76.2]
    predicted_values = [74.2, 74.7, 75.0, 74.9, 75.3, 75.8, 75.9, 76.1]
    dates = [datetime.now() - timedelta(days=i) for i in range(len(actual_values))]
    
    ensemble_predictor.update_model_performance(
        model_name="LSTM_Model",
        contract_symbol="CL_2024_06",
        actual_values=actual_values,
        predicted_values=predicted_values,
        prediction_dates=dates
    )
    
    performance_summary = ensemble_predictor.get_model_performance_summary()
    print("Performance Summary:")
    for model, metrics in performance_summary.items():
        print(f"  {model}:")
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.3f}")
    
    print("\n=== Futures Ensemble Methods Complete ===")