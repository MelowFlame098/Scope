from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
try:
    from scipy import stats
    from scipy.optimize import minimize
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    from sklearn.preprocessing import StandardScaler
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Some ensemble methods will be limited.")

logger = logging.getLogger(__name__)

class CryptoEnsembleStrategy(Enum):
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_WEIGHT = "performance_weight"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    BAYESIAN_AVERAGE = "bayesian_average"
    STACKING = "stacking"
    ADAPTIVE_SELECTION = "adaptive_selection"
    REGIME_AWARE = "regime_aware"
    RISK_PARITY = "risk_parity"
    DYNAMIC_WEIGHT = "dynamic_weight"
    CONFIDENCE_WEIGHTED = "confidence_weighted"

@dataclass
class CryptoModelPrediction:
    """Individual model prediction for crypto"""
    model_name: str
    prediction: float
    confidence: float
    volatility_forecast: float
    signal_strength: float
    category: str  # fundamental, technical, onchain, ml
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class CryptoModelPerformance:
    """Performance metrics for crypto models"""
    model_name: str
    accuracy: float
    mse: float
    mae: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    avg_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float
    information_ratio: float
    tracking_error: float
    beta: float
    alpha: float
    recent_performance: float  # Last 30 days
    regime_performance: Dict[str, float]  # Performance by market regime
    category_performance: Dict[str, float]  # Performance by prediction category

@dataclass
class CryptoEnsembleResult:
    """Advanced ensemble prediction result for crypto"""
    strategy: CryptoEnsembleStrategy
    consensus_prediction: float
    prediction_interval: Tuple[float, float]
    individual_predictions: Dict[str, CryptoModelPrediction]
    model_weights: Dict[str, float]
    confidence_score: float
    volatility_forecast: float
    market_signal: str
    signal_strength: float
    risk_metrics: Dict[str, float]
    regime_analysis: Dict[str, Any]
    ensemble_performance: Dict[str, float]
    attribution_analysis: Dict[str, float]
    timestamp: datetime

class CryptoAdvancedEnsemblePredictor:
    """Advanced ensemble prediction system for cryptocurrency models"""
    
    def __init__(self, 
                 lookback_window: int = 252,
                 min_observations: int = 30,
                 confidence_threshold: float = 0.3,
                 volatility_window: int = 30,
                 regime_detection: bool = True):
        
        self.lookback_window = lookback_window
        self.min_observations = min_observations
        self.confidence_threshold = confidence_threshold
        self.volatility_window = volatility_window
        self.regime_detection = regime_detection
        
        # Performance tracking
        self.model_performance_history = {}
        self.prediction_history = []
        self.actual_history = []
        self.ensemble_history = []
        
        # Market regime detection
        self.current_regime = "normal"
        self.regime_history = []
        
        # Dynamic parameters
        self.adaptive_weights = {}
        self.performance_decay = 0.95  # Exponential decay for performance weighting
        
    def detect_market_regime(self, price_data: pd.Series) -> str:
        """Detect current market regime for crypto"""
        if len(price_data) < 30:
            return "normal"
        
        # Calculate recent volatility and returns
        returns = price_data.pct_change().dropna()
        recent_vol = returns.tail(30).std() * np.sqrt(365)
        recent_return = (price_data.iloc[-1] / price_data.iloc[-30] - 1) * 12  # Annualized
        
        # Crypto-specific regime detection
        if recent_vol > 1.5:  # Very high volatility
            if recent_return > 0.5:
                regime = "crypto_bull_volatile"
            elif recent_return < -0.5:
                regime = "crypto_bear_volatile"
            else:
                regime = "high_volatility"
        elif recent_vol > 0.8:
            if recent_return > 0.2:
                regime = "crypto_bull"
            elif recent_return < -0.2:
                regime = "crypto_bear"
            else:
                regime = "normal_volatile"
        else:
            if recent_return > 0.1:
                regime = "low_vol_bull"
            elif recent_return < -0.1:
                regime = "low_vol_bear"
            else:
                regime = "low_volatility"
        
        self.current_regime = regime
        return regime
    
    def calculate_model_performance(self, 
                                  model_name: str, 
                                  predictions: List[float], 
                                  actuals: List[float],
                                  returns: List[float]) -> CryptoModelPerformance:
        """Calculate comprehensive performance metrics for crypto models"""
        
        if len(predictions) != len(actuals) or len(predictions) < self.min_observations:
            # Return default performance
            return CryptoModelPerformance(
                model_name=model_name,
                accuracy=0.5, mse=1.0, mae=1.0, sharpe_ratio=0.0,
                max_drawdown=0.0, hit_rate=0.5, avg_return=0.0,
                volatility=0.0, calmar_ratio=0.0, sortino_ratio=0.0,
                information_ratio=0.0, tracking_error=0.0, beta=1.0,
                alpha=0.0, recent_performance=0.5,
                regime_performance={}, category_performance={}
            )
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        returns = np.array(returns)
        
        # Basic metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        
        # Direction accuracy
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actuals))
        hit_rate = np.mean(pred_direction == actual_direction) if len(pred_direction) > 0 else 0.5
        
        # Return-based metrics
        if len(returns) > 0:
            avg_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(365)
            
            # Sharpe ratio
            sharpe_ratio = avg_return / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_vol = np.std(downside_returns) * np.sqrt(365) if len(downside_returns) > 0 else volatility
            sortino_ratio = avg_return / downside_vol if downside_vol > 0 else 0
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Calmar ratio
            calmar_ratio = avg_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
        else:
            avg_return = volatility = sharpe_ratio = sortino_ratio = 0
            max_drawdown = calmar_ratio = 0
        
        # Recent performance (last 30 observations)
        recent_predictions = predictions[-30:] if len(predictions) >= 30 else predictions
        recent_actuals = actuals[-30:] if len(actuals) >= 30 else actuals
        recent_performance = 1 - mean_squared_error(recent_actuals, recent_predictions) / np.var(recent_actuals) if np.var(recent_actuals) > 0 else 0.5
        
        return CryptoModelPerformance(
            model_name=model_name,
            accuracy=hit_rate,
            mse=mse,
            mae=mae,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            hit_rate=hit_rate,
            avg_return=avg_return,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio,
            information_ratio=sharpe_ratio,  # Simplified
            tracking_error=volatility,
            beta=1.0,  # Simplified
            alpha=avg_return,  # Simplified
            recent_performance=recent_performance,
            regime_performance={},  # Would need regime-specific data
            category_performance={}  # Would need category-specific data
        )
    
    def equal_weight_ensemble(self, predictions: Dict[str, CryptoModelPrediction]) -> Tuple[float, Dict[str, float]]:
        """Simple equal weight ensemble"""
        if not predictions:
            return 0.0, {}
        
        weights = {name: 1.0/len(predictions) for name in predictions.keys()}
        consensus = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        
        return consensus, weights
    
    def performance_weight_ensemble(self, predictions: Dict[str, CryptoModelPrediction]) -> Tuple[float, Dict[str, float]]:
        """Performance-based weighted ensemble"""
        if not predictions:
            return 0.0, {}
        
        # Use stored performance metrics or default
        performance_scores = {}
        for name in predictions.keys():
            if name in self.model_performance_history:
                perf = self.model_performance_history[name]
                # Combine multiple performance metrics
                score = (perf.hit_rate * 0.4 + 
                        (1 - perf.mse) * 0.3 + 
                        max(0, perf.sharpe_ratio) * 0.2 + 
                        perf.recent_performance * 0.1)
                performance_scores[name] = max(0.01, score)  # Minimum weight
            else:
                performance_scores[name] = 0.5  # Default
        
        # Normalize weights
        total_score = sum(performance_scores.values())
        weights = {name: score/total_score for name, score in performance_scores.items()}
        
        consensus = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        
        return consensus, weights
    
    def volatility_adjusted_ensemble(self, 
                                   predictions: Dict[str, CryptoModelPrediction],
                                   price_data: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Volatility-adjusted ensemble for crypto"""
        if not predictions:
            return 0.0, {}
        
        # Calculate current market volatility
        returns = price_data.pct_change().dropna()
        current_vol = returns.tail(self.volatility_window).std() * np.sqrt(365)
        
        # Adjust weights based on model volatility forecasts and current market conditions
        vol_adjusted_weights = {}
        
        for name, pred in predictions.items():
            # Base weight from confidence
            base_weight = pred.confidence
            
            # Volatility adjustment
            model_vol_forecast = getattr(pred, 'volatility_forecast', current_vol)
            vol_accuracy = 1.0 / (1.0 + abs(model_vol_forecast - current_vol))
            
            # Regime adjustment for crypto
            regime_multiplier = 1.0
            if self.current_regime in ['crypto_bull_volatile', 'crypto_bear_volatile']:
                # In high volatility crypto regimes, favor models with higher signal strength
                regime_multiplier = 1.0 + pred.signal_strength * 0.5
            elif self.current_regime in ['low_volatility', 'low_vol_bull', 'low_vol_bear']:
                # In low volatility, favor consistent models
                regime_multiplier = 1.0 + pred.confidence * 0.3
            
            vol_adjusted_weights[name] = base_weight * vol_accuracy * regime_multiplier
        
        # Normalize weights
        total_weight = sum(vol_adjusted_weights.values())
        weights = {name: w/total_weight for name, w in vol_adjusted_weights.items()}
        
        consensus = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        
        return consensus, weights
    
    def bayesian_ensemble(self, predictions: Dict[str, CryptoModelPrediction]) -> Tuple[float, Dict[str, float]]:
        """Bayesian model averaging for crypto predictions"""
        if not predictions or not SCIPY_AVAILABLE:
            return self.equal_weight_ensemble(predictions)
        
        # Use confidence scores as proxy for model evidence
        log_evidences = {name: np.log(max(0.01, pred.confidence)) for name, pred in predictions.items()}
        
        # Convert to probabilities (Bayesian weights)
        max_log_evidence = max(log_evidences.values())
        evidences = {name: np.exp(log_ev - max_log_evidence) for name, log_ev in log_evidences.items()}
        
        total_evidence = sum(evidences.values())
        weights = {name: ev/total_evidence for name, ev in evidences.items()}
        
        # Bayesian model averaging
        consensus = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        
        return consensus, weights
    
    def adaptive_selection_ensemble(self, 
                                  predictions: Dict[str, CryptoModelPrediction],
                                  price_data: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Adaptive model selection based on recent performance and market conditions"""
        if not predictions:
            return 0.0, {}
        
        # Detect current market regime
        current_regime = self.detect_market_regime(price_data)
        
        # Select top performing models for current regime
        regime_scores = {}
        
        for name, pred in predictions.items():
            base_score = pred.confidence
            
            # Regime-specific adjustments for crypto
            if current_regime in ['crypto_bull', 'crypto_bull_volatile']:
                # In bull markets, favor momentum and growth models
                if pred.category in ['technical', 'ml']:
                    base_score *= 1.3
                elif pred.category == 'fundamental':
                    base_score *= 0.8
            elif current_regime in ['crypto_bear', 'crypto_bear_volatile']:
                # In bear markets, favor value and risk models
                if pred.category in ['fundamental', 'onchain']:
                    base_score *= 1.3
                elif pred.category == 'technical':
                    base_score *= 0.7
            
            # Signal strength adjustment
            base_score *= (1.0 + pred.signal_strength * 0.5)
            
            regime_scores[name] = base_score
        
        # Select top models (adaptive threshold)
        sorted_models = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Dynamic selection: take top 60% of models or minimum 2
        n_select = max(2, int(len(sorted_models) * 0.6))
        selected_models = dict(sorted_models[:n_select])
        
        # Normalize weights among selected models
        total_score = sum(selected_models.values())
        weights = {name: score/total_score if name in selected_models else 0.0 
                  for name in predictions.keys()}
        
        consensus = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        
        return consensus, weights
    
    def risk_parity_ensemble(self, predictions: Dict[str, CryptoModelPrediction]) -> Tuple[float, Dict[str, float]]:
        """Risk parity ensemble for crypto (equal risk contribution)"""
        if not predictions:
            return 0.0, {}
        
        # Use volatility forecasts as risk measures
        risk_measures = {}
        for name, pred in predictions.items():
            # Use volatility forecast or default based on signal strength
            vol_forecast = getattr(pred, 'volatility_forecast', 0.5)
            risk_measures[name] = max(0.01, vol_forecast)
        
        # Risk parity weights (inverse volatility)
        inv_vol_weights = {name: 1.0/risk for name, risk in risk_measures.items()}
        total_inv_vol = sum(inv_vol_weights.values())
        
        weights = {name: w/total_inv_vol for name, w in inv_vol_weights.items()}
        consensus = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        
        return consensus, weights
    
    def dynamic_weight_ensemble(self, 
                              predictions: Dict[str, CryptoModelPrediction],
                              price_data: pd.Series) -> Tuple[float, Dict[str, float]]:
        """Dynamic weighting based on multiple factors"""
        if not predictions:
            return 0.0, {}
        
        # Multiple weighting factors
        weights = {}
        
        for name, pred in predictions.items():
            # Base confidence weight
            confidence_weight = pred.confidence
            
            # Recent performance weight
            perf_weight = 1.0
            if name in self.model_performance_history:
                perf = self.model_performance_history[name]
                perf_weight = perf.recent_performance
            
            # Signal strength weight
            signal_weight = 1.0 + pred.signal_strength * 0.5
            
            # Category diversification weight
            category_weight = 1.0
            category_counts = {}
            for p in predictions.values():
                category_counts[p.category] = category_counts.get(p.category, 0) + 1
            
            # Favor less represented categories for diversification
            if pred.category in category_counts:
                category_weight = 1.0 / np.sqrt(category_counts[pred.category])
            
            # Combine all weights
            combined_weight = confidence_weight * perf_weight * signal_weight * category_weight
            weights[name] = max(0.01, combined_weight)
        
        # Normalize
        total_weight = sum(weights.values())
        weights = {name: w/total_weight for name, w in weights.items()}
        
        consensus = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        
        return consensus, weights
    
    def predict(self, 
                predictions: Dict[str, CryptoModelPrediction],
                price_data: pd.Series,
                strategy: CryptoEnsembleStrategy = CryptoEnsembleStrategy.DYNAMIC_WEIGHT) -> CryptoEnsembleResult:
        """Generate ensemble prediction using specified strategy"""
        
        if not predictions:
            return CryptoEnsembleResult(
                strategy=strategy,
                consensus_prediction=0.0,
                prediction_interval=(0.0, 0.0),
                individual_predictions={},
                model_weights={},
                confidence_score=0.0,
                volatility_forecast=0.0,
                market_signal="HOLD",
                signal_strength=0.0,
                risk_metrics={},
                regime_analysis={},
                ensemble_performance={},
                attribution_analysis={},
                timestamp=datetime.now()
            )
        
        # Select ensemble strategy
        if strategy == CryptoEnsembleStrategy.EQUAL_WEIGHT:
            consensus, weights = self.equal_weight_ensemble(predictions)
        elif strategy == CryptoEnsembleStrategy.PERFORMANCE_WEIGHT:
            consensus, weights = self.performance_weight_ensemble(predictions)
        elif strategy == CryptoEnsembleStrategy.VOLATILITY_ADJUSTED:
            consensus, weights = self.volatility_adjusted_ensemble(predictions, price_data)
        elif strategy == CryptoEnsembleStrategy.BAYESIAN_AVERAGE:
            consensus, weights = self.bayesian_ensemble(predictions)
        elif strategy == CryptoEnsembleStrategy.ADAPTIVE_SELECTION:
            consensus, weights = self.adaptive_selection_ensemble(predictions, price_data)
        elif strategy == CryptoEnsembleStrategy.RISK_PARITY:
            consensus, weights = self.risk_parity_ensemble(predictions)
        elif strategy == CryptoEnsembleStrategy.DYNAMIC_WEIGHT:
            consensus, weights = self.dynamic_weight_ensemble(predictions, price_data)
        else:
            consensus, weights = self.dynamic_weight_ensemble(predictions, price_data)
        
        # Calculate ensemble metrics
        confidence_score = sum(pred.confidence * weights.get(pred.model_name, 0) 
                             for pred in predictions.values())
        
        # Volatility forecast
        volatility_forecast = sum(getattr(pred, 'volatility_forecast', 0.3) * weights.get(pred.model_name, 0)
                                for pred in predictions.values())
        
        # Signal strength
        signal_strength = sum(pred.signal_strength * weights.get(pred.model_name, 0)
                            for pred in predictions.values())
        
        # Market signal
        current_price = price_data.iloc[-1]
        price_change = (consensus - current_price) / current_price
        
        if price_change > 0.1:
            market_signal = "STRONG_BUY"
        elif price_change > 0.03:
            market_signal = "BUY"
        elif price_change < -0.1:
            market_signal = "STRONG_SELL"
        elif price_change < -0.03:
            market_signal = "SELL"
        else:
            market_signal = "HOLD"
        
        # Prediction interval
        pred_values = [pred.prediction for pred in predictions.values()]
        pred_std = np.std(pred_values) if len(pred_values) > 1 else abs(consensus * 0.05)
        prediction_interval = (consensus - 1.96 * pred_std, consensus + 1.96 * pred_std)
        
        # Risk metrics
        risk_metrics = {
            'prediction_uncertainty': pred_std / consensus if consensus != 0 else 0,
            'model_disagreement': np.std(pred_values) / np.mean(pred_values) if np.mean(pred_values) != 0 else 0,
            'weight_concentration': sum(w**2 for w in weights.values()),  # Herfindahl index
            'volatility_forecast': volatility_forecast
        }
        
        # Regime analysis
        current_regime = self.detect_market_regime(price_data)
        regime_analysis = {
            'current_regime': current_regime,
            'regime_confidence': confidence_score,
            'regime_appropriate_models': len([p for p in predictions.values() if p.confidence > 0.5])
        }
        
        # Attribution analysis
        attribution_analysis = {}
        for name, pred in predictions.items():
            contribution = (pred.prediction - consensus) * weights.get(name, 0)
            attribution_analysis[name] = contribution
        
        return CryptoEnsembleResult(
            strategy=strategy,
            consensus_prediction=consensus,
            prediction_interval=prediction_interval,
            individual_predictions=predictions,
            model_weights=weights,
            confidence_score=confidence_score,
            volatility_forecast=volatility_forecast,
            market_signal=market_signal,
            signal_strength=signal_strength,
            risk_metrics=risk_metrics,
            regime_analysis=regime_analysis,
            ensemble_performance={},  # Would be populated with historical data
            attribution_analysis=attribution_analysis,
            timestamp=datetime.now()
        )
    
    def update_performance(self, model_name: str, prediction: float, actual: float, return_value: float):
        """Update model performance tracking"""
        if model_name not in self.model_performance_history:
            self.model_performance_history[model_name] = {
                'predictions': [],
                'actuals': [],
                'returns': []
            }
        
        # Add new observation
        self.model_performance_history[model_name]['predictions'].append(prediction)
        self.model_performance_history[model_name]['actuals'].append(actual)
        self.model_performance_history[model_name]['returns'].append(return_value)
        
        # Keep only recent observations
        for key in ['predictions', 'actuals', 'returns']:
            if len(self.model_performance_history[model_name][key]) > self.lookback_window:
                self.model_performance_history[model_name][key] = \
                    self.model_performance_history[model_name][key][-self.lookback_window:]
        
        # Recalculate performance metrics
        perf_data = self.model_performance_history[model_name]
        if len(perf_data['predictions']) >= self.min_observations:
            performance = self.calculate_model_performance(
                model_name,
                perf_data['predictions'],
                perf_data['actuals'],
                perf_data['returns']
            )
            self.model_performance_history[model_name] = performance

# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate Bitcoin price data
    price_data = pd.Series(
        np.cumsum(np.random.randn(len(dates)) * 0.03) * 1000 + 45000,
        index=dates
    )
    
    # Create sample predictions
    sample_predictions = {
        'lstm_model': CryptoModelPrediction(
            model_name='lstm_model',
            prediction=47500.0,
            confidence=0.75,
            volatility_forecast=0.65,
            signal_strength=0.8,
            category='ml',
            timestamp=datetime.now(),
            metadata={}
        ),
        'stock_to_flow': CryptoModelPrediction(
            model_name='stock_to_flow',
            prediction=52000.0,
            confidence=0.65,
            volatility_forecast=0.45,
            signal_strength=0.6,
            category='fundamental',
            timestamp=datetime.now(),
            metadata={}
        ),
        'mvrv_model': CryptoModelPrediction(
            model_name='mvrv_model',
            prediction=44000.0,
            confidence=0.70,
            volatility_forecast=0.55,
            signal_strength=0.4,
            category='onchain',
            timestamp=datetime.now(),
            metadata={}
        )
    }
    
    # Initialize ensemble predictor
    ensemble = CryptoAdvancedEnsemblePredictor()
    
    # Test different ensemble strategies
    strategies = [
        CryptoEnsembleStrategy.EQUAL_WEIGHT,
        CryptoEnsembleStrategy.PERFORMANCE_WEIGHT,
        CryptoEnsembleStrategy.VOLATILITY_ADJUSTED,
        CryptoEnsembleStrategy.DYNAMIC_WEIGHT
    ]
    
    print("\n=== Crypto Advanced Ensemble Analysis ===")
    
    for strategy in strategies:
        result = ensemble.predict(sample_predictions, price_data, strategy)
        
        print(f"\n--- {strategy.value.upper()} STRATEGY ---")
        print(f"Consensus Prediction: ${result.consensus_prediction:.2f}")
        print(f"Market Signal: {result.market_signal}")
        print(f"Confidence Score: {result.confidence_score:.2f}")
        print(f"Signal Strength: {result.signal_strength:.2f}")
        print(f"Volatility Forecast: {result.volatility_forecast:.2f}")
        
        print("\nModel Weights:")
        for model, weight in result.model_weights.items():
            print(f"  {model}: {weight:.3f}")
        
        print("\nRisk Metrics:")
        for metric, value in result.risk_metrics.items():
            print(f"  {metric}: {value:.3f}")
        
        print(f"\nRegime Analysis: {result.regime_analysis['current_regime']}")
        print(f"Prediction Interval: ${result.prediction_interval[0]:.2f} - ${result.prediction_interval[1]:.2f}")