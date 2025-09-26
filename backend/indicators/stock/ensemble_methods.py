"""Advanced Ensemble Methods for Stock Prediction

This module implements sophisticated ensemble prediction methods including:
- Dynamic model weighting based on performance
- Bayesian model averaging
- Stacking ensemble with meta-learners
- Adaptive ensemble selection
- Multi-horizon prediction ensemble
- Risk-adjusted ensemble optimization

Author: Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import json
warnings.filterwarnings('ignore')

# Try to import sklearn for advanced ensemble methods
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Try to import scipy for statistical methods
try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

class EnsembleStrategy(Enum):
    """Different ensemble strategies"""
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_WEIGHT = "performance_weight"
    BAYESIAN_AVERAGE = "bayesian_average"
    STACKING = "stacking"
    ADAPTIVE_SELECTION = "adaptive_selection"
    RISK_ADJUSTED = "risk_adjusted"
    DYNAMIC_WEIGHT = "dynamic_weight"

class ModelPerformanceMetric(Enum):
    """Performance metrics for model evaluation"""
    MSE = "mse"
    MAE = "mae"
    MAPE = "mape"
    SHARPE_RATIO = "sharpe_ratio"
    INFORMATION_RATIO = "information_ratio"
    ACCURACY = "accuracy"
    DIRECTIONAL_ACCURACY = "directional_accuracy"

@dataclass
class ModelPrediction:
    """Individual model prediction with metadata"""
    model_name: str
    prediction: float
    confidence: float
    signal: str
    risk_level: str
    timestamp: datetime
    features_used: List[str]
    model_version: str
    computation_time: float
    metadata: Dict[str, Any]

@dataclass
class ModelPerformance:
    """Model performance tracking"""
    model_name: str
    mse: float
    mae: float
    mape: float
    accuracy: float
    directional_accuracy: float
    sharpe_ratio: float
    information_ratio: float
    prediction_count: int
    last_updated: datetime
    performance_history: List[float]
    weight_history: List[float]

@dataclass
class EnsembleResult:
    """Advanced ensemble prediction result"""
    consensus_prediction: float
    consensus_confidence: float
    consensus_signal: str
    consensus_risk_level: str
    prediction_interval: Tuple[float, float]
    individual_predictions: Dict[str, ModelPrediction]
    model_weights: Dict[str, float]
    ensemble_strategy: EnsembleStrategy
    prediction_variance: float
    model_agreement: float
    uncertainty_measure: float
    feature_importance: Dict[str, float]
    performance_metrics: Dict[str, ModelPerformance]
    timestamp: datetime
    metadata: Dict[str, Any]

class BaseEnsembleMethod(ABC):
    """Base class for ensemble methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.model_performances = {}
        self.prediction_history = deque(maxlen=1000)
        self.weight_history = deque(maxlen=100)
    
    @abstractmethod
    def calculate_weights(self, predictions: Dict[str, ModelPrediction], 
                         historical_performance: Dict[str, ModelPerformance]) -> Dict[str, float]:
        """Calculate model weights based on strategy"""
        pass
    
    @abstractmethod
    def combine_predictions(self, predictions: Dict[str, ModelPrediction], 
                          weights: Dict[str, float]) -> Tuple[float, float]:
        """Combine predictions using weights"""
        pass
    
    def update_performance(self, model_name: str, actual_value: float, 
                          predicted_value: float, prediction_time: datetime):
        """Update model performance metrics"""
        if model_name not in self.model_performances:
            self.model_performances[model_name] = ModelPerformance(
                model_name=model_name,
                mse=0.0, mae=0.0, mape=0.0, accuracy=0.0,
                directional_accuracy=0.0, sharpe_ratio=0.0,
                information_ratio=0.0, prediction_count=0,
                last_updated=prediction_time,
                performance_history=[], weight_history=[]
            )
        
        perf = self.model_performances[model_name]
        error = actual_value - predicted_value
        abs_error = abs(error)
        
        # Update metrics using exponential moving average
        alpha = 0.1  # Smoothing factor
        perf.mse = alpha * (error ** 2) + (1 - alpha) * perf.mse
        perf.mae = alpha * abs_error + (1 - alpha) * perf.mae
        
        if actual_value != 0:
            mape_error = abs_error / abs(actual_value)
            perf.mape = alpha * mape_error + (1 - alpha) * perf.mape
        
        perf.prediction_count += 1
        perf.last_updated = prediction_time
        perf.performance_history.append(abs_error)
        
        # Keep only last 100 performance records
        if len(perf.performance_history) > 100:
            perf.performance_history = perf.performance_history[-100:]

class EqualWeightEnsemble(BaseEnsembleMethod):
    """Simple equal weight ensemble"""
    
    def __init__(self):
        super().__init__("Equal Weight")
    
    def calculate_weights(self, predictions: Dict[str, ModelPrediction], 
                         historical_performance: Dict[str, ModelPerformance]) -> Dict[str, float]:
        """Equal weights for all models"""
        n_models = len(predictions)
        return {name: 1.0 / n_models for name in predictions.keys()}
    
    def combine_predictions(self, predictions: Dict[str, ModelPrediction], 
                          weights: Dict[str, float]) -> Tuple[float, float]:
        """Simple weighted average"""
        weighted_pred = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        weighted_conf = sum(pred.confidence * weights[name] for name, pred in predictions.items())
        return weighted_pred, weighted_conf

class PerformanceWeightedEnsemble(BaseEnsembleMethod):
    """Performance-based weighted ensemble"""
    
    def __init__(self, metric: ModelPerformanceMetric = ModelPerformanceMetric.MAE):
        super().__init__("Performance Weighted")
        self.metric = metric
    
    def calculate_weights(self, predictions: Dict[str, ModelPrediction], 
                         historical_performance: Dict[str, ModelPerformance]) -> Dict[str, float]:
        """Calculate weights based on historical performance"""
        weights = {}
        
        for name in predictions.keys():
            if name in historical_performance:
                perf = historical_performance[name]
                
                # Get performance metric (lower is better for error metrics)
                if self.metric == ModelPerformanceMetric.MAE:
                    score = 1.0 / (1.0 + perf.mae) if perf.mae > 0 else 1.0
                elif self.metric == ModelPerformanceMetric.MSE:
                    score = 1.0 / (1.0 + perf.mse) if perf.mse > 0 else 1.0
                elif self.metric == ModelPerformanceMetric.MAPE:
                    score = 1.0 / (1.0 + perf.mape) if perf.mape > 0 else 1.0
                elif self.metric == ModelPerformanceMetric.ACCURACY:
                    score = perf.accuracy
                elif self.metric == ModelPerformanceMetric.DIRECTIONAL_ACCURACY:
                    score = perf.directional_accuracy
                else:
                    score = 1.0
                
                weights[name] = max(0.01, score)  # Minimum weight
            else:
                weights[name] = 0.5  # Default weight for new models
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def combine_predictions(self, predictions: Dict[str, ModelPrediction], 
                          weights: Dict[str, float]) -> Tuple[float, float]:
        """Weighted average based on performance"""
        weighted_pred = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        weighted_conf = sum(pred.confidence * weights[name] for name, pred in predictions.items())
        return weighted_pred, weighted_conf

class BayesianEnsemble(BaseEnsembleMethod):
    """Bayesian model averaging ensemble"""
    
    def __init__(self):
        super().__init__("Bayesian Average")
        self.prior_weights = {}
    
    def calculate_weights(self, predictions: Dict[str, ModelPrediction], 
                         historical_performance: Dict[str, ModelPerformance]) -> Dict[str, float]:
        """Calculate Bayesian weights"""
        weights = {}
        
        for name in predictions.keys():
            if name in historical_performance:
                perf = historical_performance[name]
                
                # Bayesian weight based on inverse variance and prior
                if perf.prediction_count > 0 and len(perf.performance_history) > 1:
                    variance = np.var(perf.performance_history)
                    precision = 1.0 / (variance + 1e-6)  # Add small constant to avoid division by zero
                    
                    # Prior weight (can be updated based on model characteristics)
                    prior = self.prior_weights.get(name, 1.0)
                    
                    # Posterior weight
                    weights[name] = precision * prior
                else:
                    weights[name] = 1.0
            else:
                weights[name] = 1.0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def combine_predictions(self, predictions: Dict[str, ModelPrediction], 
                          weights: Dict[str, float]) -> Tuple[float, float]:
        """Bayesian combination with uncertainty quantification"""
        weighted_pred = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        
        # Calculate prediction uncertainty
        pred_variance = sum(weights[name] * (pred.prediction - weighted_pred) ** 2 
                           for name, pred in predictions.items())
        
        # Confidence based on inverse of uncertainty
        confidence = 1.0 / (1.0 + pred_variance)
        
        return weighted_pred, confidence

class StackingEnsemble(BaseEnsembleMethod):
    """Stacking ensemble with meta-learner"""
    
    def __init__(self):
        super().__init__("Stacking")
        self.meta_learner = None
        self.is_fitted = False
        
        if SKLEARN_AVAILABLE:
            self.meta_learner = Ridge(alpha=1.0)
    
    def fit_meta_learner(self, base_predictions: np.ndarray, targets: np.ndarray):
        """Fit the meta-learner on base model predictions"""
        if self.meta_learner is not None and len(base_predictions) > 0:
            try:
                self.meta_learner.fit(base_predictions, targets)
                self.is_fitted = True
            except Exception as e:
                print(f"Error fitting meta-learner: {e}")
                self.is_fitted = False
    
    def calculate_weights(self, predictions: Dict[str, ModelPrediction], 
                         historical_performance: Dict[str, ModelPerformance]) -> Dict[str, float]:
        """Weights determined by meta-learner"""
        if self.is_fitted and self.meta_learner is not None:
            try:
                # Create feature matrix from predictions
                X = np.array([[pred.prediction, pred.confidence] for pred in predictions.values()])
                
                # Get meta-learner coefficients as weights
                if hasattr(self.meta_learner, 'coef_'):
                    coeffs = self.meta_learner.coef_
                    if len(coeffs) == len(predictions):
                        weights = {name: max(0.01, abs(coeff)) for name, coeff in zip(predictions.keys(), coeffs)}
                    else:
                        # Fallback to equal weights
                        weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
                else:
                    weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
            except Exception:
                weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
        else:
            weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def combine_predictions(self, predictions: Dict[str, ModelPrediction], 
                          weights: Dict[str, float]) -> Tuple[float, float]:
        """Combine using meta-learner if available"""
        if self.is_fitted and self.meta_learner is not None:
            try:
                # Create feature matrix
                X = np.array([[pred.prediction, pred.confidence] for pred in predictions.values()]).reshape(1, -1)
                
                # Predict using meta-learner
                meta_prediction = self.meta_learner.predict(X)[0]
                
                # Confidence based on model agreement
                pred_values = [pred.prediction for pred in predictions.values()]
                agreement = 1.0 - (np.std(pred_values) / (np.mean(pred_values) + 1e-6))
                confidence = max(0.1, min(0.9, agreement))
                
                return meta_prediction, confidence
            except Exception:
                pass
        
        # Fallback to weighted average
        weighted_pred = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        weighted_conf = sum(pred.confidence * weights[name] for name, pred in predictions.items())
        return weighted_pred, weighted_conf

class AdaptiveEnsemble(BaseEnsembleMethod):
    """Adaptive ensemble that selects best models dynamically"""
    
    def __init__(self, selection_window: int = 20, min_models: int = 2):
        super().__init__("Adaptive Selection")
        self.selection_window = selection_window
        self.min_models = min_models
        self.selected_models = set()
    
    def calculate_weights(self, predictions: Dict[str, ModelPrediction], 
                         historical_performance: Dict[str, ModelPerformance]) -> Dict[str, float]:
        """Select and weight best performing models"""
        # Rank models by recent performance
        model_scores = {}
        
        for name in predictions.keys():
            if name in historical_performance:
                perf = historical_performance[name]
                
                # Calculate recent performance score
                if len(perf.performance_history) >= self.selection_window:
                    recent_errors = perf.performance_history[-self.selection_window:]
                    score = 1.0 / (1.0 + np.mean(recent_errors))
                else:
                    score = 1.0 / (1.0 + perf.mae)
                
                model_scores[name] = score
            else:
                model_scores[name] = 0.5  # Default score for new models
        
        # Select top models
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        n_select = max(self.min_models, min(len(sorted_models), len(sorted_models) // 2 + 1))
        
        selected = dict(sorted_models[:n_select])
        self.selected_models = set(selected.keys())
        
        # Assign weights only to selected models
        weights = {name: 0.0 for name in predictions.keys()}
        
        if selected:
            total_score = sum(selected.values())
            for name, score in selected.items():
                weights[name] = score / total_score if total_score > 0 else 1.0 / len(selected)
        
        return weights
    
    def combine_predictions(self, predictions: Dict[str, ModelPrediction], 
                          weights: Dict[str, float]) -> Tuple[float, float]:
        """Combine only selected models"""
        selected_predictions = {name: pred for name, pred in predictions.items() 
                              if name in self.selected_models and weights[name] > 0}
        
        if not selected_predictions:
            # Fallback to all models
            selected_predictions = predictions
            weights = {name: 1.0 / len(predictions) for name in predictions.keys()}
        
        weighted_pred = sum(pred.prediction * weights[name] for name, pred in selected_predictions.items())
        weighted_conf = sum(pred.confidence * weights[name] for name, pred in selected_predictions.items())
        
        return weighted_pred, weighted_conf

class RiskAdjustedEnsemble(BaseEnsembleMethod):
    """Risk-adjusted ensemble optimization"""
    
    def __init__(self, risk_aversion: float = 1.0):
        super().__init__("Risk Adjusted")
        self.risk_aversion = risk_aversion
    
    def calculate_weights(self, predictions: Dict[str, ModelPrediction], 
                         historical_performance: Dict[str, ModelPerformance]) -> Dict[str, float]:
        """Optimize weights considering risk-return tradeoff"""
        weights = {}
        
        # Calculate expected returns and risks
        expected_returns = {}
        risks = {}
        
        for name in predictions.keys():
            if name in historical_performance:
                perf = historical_performance[name]
                
                # Expected return (inverse of error)
                expected_returns[name] = 1.0 / (1.0 + perf.mae)
                
                # Risk (variance of predictions)
                if len(perf.performance_history) > 1:
                    risks[name] = np.var(perf.performance_history)
                else:
                    risks[name] = 0.1  # Default risk
            else:
                expected_returns[name] = 0.5
                risks[name] = 0.1
        
        # Risk-adjusted optimization (simplified mean-variance)
        for name in predictions.keys():
            utility = expected_returns[name] - 0.5 * self.risk_aversion * risks[name]
            weights[name] = max(0.01, utility)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def combine_predictions(self, predictions: Dict[str, ModelPrediction], 
                          weights: Dict[str, float]) -> Tuple[float, float]:
        """Risk-adjusted combination"""
        weighted_pred = sum(pred.prediction * weights[name] for name, pred in predictions.items())
        
        # Confidence adjusted by risk
        pred_values = [pred.prediction for pred in predictions.values()]
        prediction_risk = np.std(pred_values) if len(pred_values) > 1 else 0.0
        
        base_confidence = sum(pred.confidence * weights[name] for name, pred in predictions.items())
        risk_adjusted_confidence = base_confidence * (1.0 - min(0.5, prediction_risk / (weighted_pred + 1e-6)))
        
        return weighted_pred, max(0.1, risk_adjusted_confidence)

class AdvancedEnsemblePredictor:
    """Advanced ensemble predictor with multiple strategies"""
    
    def __init__(self):
        self.ensemble_methods = {
            EnsembleStrategy.EQUAL_WEIGHT: EqualWeightEnsemble(),
            EnsembleStrategy.PERFORMANCE_WEIGHT: PerformanceWeightedEnsemble(),
            EnsembleStrategy.BAYESIAN_AVERAGE: BayesianEnsemble(),
            EnsembleStrategy.STACKING: StackingEnsemble(),
            EnsembleStrategy.ADAPTIVE_SELECTION: AdaptiveEnsemble(),
            EnsembleStrategy.RISK_ADJUSTED: RiskAdjustedEnsemble()
        }
        
        self.default_strategy = EnsembleStrategy.PERFORMANCE_WEIGHT
        self.prediction_history = deque(maxlen=1000)
        self.ensemble_performance = {}
    
    def predict_ensemble(self, predictions: Dict[str, ModelPrediction], 
                        strategy: EnsembleStrategy = None,
                        target_confidence: float = 0.7) -> EnsembleResult:
        """Make ensemble prediction using specified strategy"""
        if not predictions:
            return self._create_empty_result()
        
        strategy = strategy or self.default_strategy
        ensemble_method = self.ensemble_methods[strategy]
        
        # Get historical performance
        historical_performance = ensemble_method.model_performances
        
        # Calculate weights
        weights = ensemble_method.calculate_weights(predictions, historical_performance)
        
        # Combine predictions
        consensus_pred, consensus_conf = ensemble_method.combine_predictions(predictions, weights)
        
        # Calculate additional metrics
        pred_values = [pred.prediction for pred in predictions.values()]
        prediction_variance = np.var(pred_values) if len(pred_values) > 1 else 0.0
        
        # Model agreement (inverse of coefficient of variation)
        mean_pred = np.mean(pred_values)
        model_agreement = 1.0 - (np.std(pred_values) / (abs(mean_pred) + 1e-6)) if mean_pred != 0 else 0.0
        model_agreement = max(0.0, min(1.0, model_agreement))
        
        # Uncertainty measure
        uncertainty_measure = prediction_variance / (consensus_pred ** 2 + 1e-6)
        
        # Prediction interval (simple approach)
        std_dev = np.sqrt(prediction_variance)
        prediction_interval = (consensus_pred - 1.96 * std_dev, consensus_pred + 1.96 * std_dev)
        
        # Consensus signal and risk level
        signals = [pred.signal for pred in predictions.values()]
        signal_counts = {'BUY': signals.count('BUY'), 'SELL': signals.count('SELL'), 'HOLD': signals.count('HOLD')}
        consensus_signal = max(signal_counts, key=signal_counts.get)
        
        risk_levels = [pred.risk_level for pred in predictions.values()]
        risk_counts = {'LOW': risk_levels.count('LOW'), 'MEDIUM': risk_levels.count('MEDIUM'), 'HIGH': risk_levels.count('HIGH')}
        consensus_risk_level = max(risk_counts, key=risk_counts.get)
        
        # Feature importance (simplified)
        feature_importance = self._calculate_feature_importance(predictions, weights)
        
        # Create result
        result = EnsembleResult(
            consensus_prediction=consensus_pred,
            consensus_confidence=consensus_conf,
            consensus_signal=consensus_signal,
            consensus_risk_level=consensus_risk_level,
            prediction_interval=prediction_interval,
            individual_predictions=predictions,
            model_weights=weights,
            ensemble_strategy=strategy,
            prediction_variance=prediction_variance,
            model_agreement=model_agreement,
            uncertainty_measure=uncertainty_measure,
            feature_importance=feature_importance,
            performance_metrics=historical_performance,
            timestamp=datetime.now(),
            metadata={
                'n_models': len(predictions),
                'signal_distribution': signal_counts,
                'risk_distribution': risk_counts,
                'target_confidence': target_confidence,
                'achieved_confidence': consensus_conf
            }
        )
        
        # Store prediction
        self.prediction_history.append(result)
        
        return result
    
    def _calculate_feature_importance(self, predictions: Dict[str, ModelPrediction], 
                                    weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate aggregated feature importance"""
        feature_importance = defaultdict(float)
        
        for name, pred in predictions.items():
            weight = weights.get(name, 0.0)
            
            # Simple feature importance based on model type
            if 'fundamental' in name.lower() or 'dcf' in name.lower() or 'ddm' in name.lower():
                feature_importance['fundamental_analysis'] += weight
            elif 'technical' in name.lower() or 'arima' in name.lower():
                feature_importance['technical_analysis'] += weight
            elif 'ml' in name.lower() or 'lstm' in name.lower() or 'transformer' in name.lower():
                feature_importance['machine_learning'] += weight
            else:
                feature_importance['other'] += weight
        
        return dict(feature_importance)
    
    def _create_empty_result(self) -> EnsembleResult:
        """Create empty result when no predictions available"""
        return EnsembleResult(
            consensus_prediction=0.0,
            consensus_confidence=0.0,
            consensus_signal='HOLD',
            consensus_risk_level='HIGH',
            prediction_interval=(0.0, 0.0),
            individual_predictions={},
            model_weights={},
            ensemble_strategy=self.default_strategy,
            prediction_variance=0.0,
            model_agreement=0.0,
            uncertainty_measure=1.0,
            feature_importance={},
            performance_metrics={},
            timestamp=datetime.now(),
            metadata={'error': 'No predictions available'}
        )
    
    def update_model_performance(self, model_name: str, actual_value: float, 
                               predicted_value: float, prediction_time: datetime,
                               strategy: EnsembleStrategy = None):
        """Update performance for specific ensemble method"""
        strategy = strategy or self.default_strategy
        ensemble_method = self.ensemble_methods[strategy]
        ensemble_method.update_performance(model_name, actual_value, predicted_value, prediction_time)
    
    def get_ensemble_performance(self, strategy: EnsembleStrategy = None) -> Dict[str, ModelPerformance]:
        """Get performance metrics for ensemble method"""
        strategy = strategy or self.default_strategy
        return self.ensemble_methods[strategy].model_performances
    
    def optimize_ensemble_strategy(self, validation_data: List[Tuple[Dict[str, ModelPrediction], float]]) -> EnsembleStrategy:
        """Find best ensemble strategy based on validation data"""
        strategy_performance = {}
        
        for strategy in EnsembleStrategy:
            if strategy in self.ensemble_methods:
                total_error = 0.0
                count = 0
                
                for predictions, actual_value in validation_data:
                    try:
                        result = self.predict_ensemble(predictions, strategy)
                        error = abs(result.consensus_prediction - actual_value)
                        total_error += error
                        count += 1
                    except Exception:
                        continue
                
                if count > 0:
                    strategy_performance[strategy] = total_error / count
                else:
                    strategy_performance[strategy] = float('inf')
        
        # Return strategy with lowest error
        if strategy_performance:
            best_strategy = min(strategy_performance, key=strategy_performance.get)
            self.default_strategy = best_strategy
            return best_strategy
        
        return self.default_strategy
    
    def get_prediction_history(self, limit: int = 10) -> List[EnsembleResult]:
        """Get recent ensemble predictions"""
        return list(self.prediction_history)[-limit:] if self.prediction_history else []

# Example usage
if __name__ == "__main__":
    # Initialize ensemble predictor
    ensemble_predictor = AdvancedEnsemblePredictor()
    
    # Sample predictions from different models
    sample_predictions = {
        'dcf': ModelPrediction(
            model_name='dcf',
            prediction=155.0,
            confidence=0.7,
            signal='BUY',
            risk_level='LOW',
            timestamp=datetime.now(),
            features_used=['revenue', 'cash_flow', 'growth_rate'],
            model_version='1.0',
            computation_time=0.1,
            metadata={'fair_value': 155.0}
        ),
        'capm': ModelPrediction(
            model_name='capm',
            prediction=148.0,
            confidence=0.6,
            signal='HOLD',
            risk_level='MEDIUM',
            timestamp=datetime.now(),
            features_used=['beta', 'market_return', 'risk_free_rate'],
            model_version='1.0',
            computation_time=0.05,
            metadata={'expected_return': 0.08}
        ),
        'ml_ensemble': ModelPrediction(
            model_name='ml_ensemble',
            prediction=152.0,
            confidence=0.8,
            signal='BUY',
            risk_level='MEDIUM',
            timestamp=datetime.now(),
            features_used=['price_history', 'volume', 'technical_indicators'],
            model_version='2.0',
            computation_time=0.5,
            metadata={'lstm_pred': 151.0, 'rf_pred': 153.0}
        )
    }
    
    # Test different ensemble strategies
    strategies = [EnsembleStrategy.EQUAL_WEIGHT, EnsembleStrategy.PERFORMANCE_WEIGHT, 
                 EnsembleStrategy.BAYESIAN_AVERAGE, EnsembleStrategy.ADAPTIVE_SELECTION]
    
    print("=== Advanced Ensemble Prediction Results ===")
    
    for strategy in strategies:
        result = ensemble_predictor.predict_ensemble(sample_predictions, strategy)
        
        print(f"\n--- {strategy.value.upper()} ---")
        print(f"Consensus Prediction: ${result.consensus_prediction:.2f}")
        print(f"Consensus Confidence: {result.consensus_confidence:.3f}")
        print(f"Consensus Signal: {result.consensus_signal}")
        print(f"Prediction Interval: ${result.prediction_interval[0]:.2f} - ${result.prediction_interval[1]:.2f}")
        print(f"Model Agreement: {result.model_agreement:.3f}")
        print(f"Uncertainty Measure: {result.uncertainty_measure:.3f}")
        print(f"Model Weights: {result.model_weights}")
        print(f"Feature Importance: {result.feature_importance}")