from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CrossAssetMarketRegime(Enum):
    """Cross-asset market regimes"""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    NORMAL = "normal"
    TRANSITION = "transition"

class CrossAssetEnsembleStrategy(Enum):
    """Cross-asset ensemble strategies"""
    EQUAL_WEIGHT = "equal_weight"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    REGIME_AWARE = "regime_aware"
    CORRELATION_ADJUSTED = "correlation_adjusted"
    DYNAMIC_WEIGHT = "dynamic_weight"
    RISK_PARITY = "risk_parity"
    BAYESIAN_MODEL_AVERAGING = "bayesian_model_averaging"

@dataclass
class CrossAssetModelPrediction:
    """Individual cross-asset model prediction"""
    model_name: str
    prediction: Union[float, np.ndarray]
    confidence: float
    timestamp: datetime
    asset_class: str
    prediction_horizon: str  # '1d', '1w', '1m', etc.
    metadata: Dict[str, Any]
    risk_metrics: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None

@dataclass
class CrossAssetModelPerformance:
    """Cross-asset model performance metrics"""
    model_name: str
    accuracy: float
    sharpe_ratio: float
    max_drawdown: float
    volatility: float
    hit_rate: float
    avg_return: float
    regime_performance: Dict[CrossAssetMarketRegime, float]
    asset_class_performance: Dict[str, float]
    recent_performance: float  # Performance over last N predictions
    stability_score: float  # Consistency of performance

@dataclass
class CrossAssetEnsembleResult:
    """Result from cross-asset ensemble prediction"""
    ensemble_prediction: Union[float, np.ndarray]
    individual_predictions: List[CrossAssetModelPrediction]
    model_weights: Dict[str, float]
    confidence_score: float
    regime: CrossAssetMarketRegime
    strategy_used: CrossAssetEnsembleStrategy
    risk_assessment: Dict[str, float]
    diversification_benefit: float
    prediction_intervals: Dict[str, Tuple[float, float]]
    execution_time: float
    metadata: Dict[str, Any]

class CrossAssetAdvancedEnsemblePredictor:
    """Advanced ensemble predictor for cross-asset models with regime awareness"""
    
    def __init__(self, 
                 lookback_window: int = 252,
                 regime_detection_window: int = 60,
                 min_models_required: int = 3):
        self.lookback_window = lookback_window
        self.regime_detection_window = regime_detection_window
        self.min_models_required = min_models_required
        
        # Model performance tracking
        self.model_performance_history: Dict[str, List[CrossAssetModelPerformance]] = {}
        self.regime_history: List[Tuple[datetime, CrossAssetMarketRegime]] = []
        
        # Regime detection components
        self.regime_detector = None
        self.scaler = StandardScaler()
        
        # Dynamic weight adjustment parameters
        self.weight_decay_factor = 0.95
        self.performance_threshold = 0.6
        self.regime_transition_penalty = 0.1
    
    def detect_market_regime(self, 
                           market_data: Dict[str, pd.DataFrame],
                           economic_indicators: Optional[Dict[str, float]] = None) -> CrossAssetMarketRegime:
        """Detect current cross-asset market regime"""
        try:
            # Calculate regime indicators
            regime_features = self._calculate_regime_features(market_data, economic_indicators)
            
            # Use clustering or rule-based approach for regime detection
            if len(self.regime_history) < self.regime_detection_window:
                return self._rule_based_regime_detection(regime_features)
            else:
                return self._ml_based_regime_detection(regime_features)
        
        except Exception as e:
            print(f"Regime detection error: {e}")
            return CrossAssetMarketRegime.NORMAL
    
    def _calculate_regime_features(self, 
                                 market_data: Dict[str, pd.DataFrame],
                                 economic_indicators: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate features for regime detection"""
        features = {}
        
        try:
            # Cross-asset volatility features
            volatilities = []
            correlations = []
            returns_data = {}
            
            # Calculate returns and volatilities for each asset class
            for asset_class, data in market_data.items():
                if 'close' in data.columns and len(data) > 20:
                    returns = data['close'].pct_change().dropna()
                    returns_data[asset_class] = returns
                    
                    # Volatility features
                    vol_20d = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
                    vol_5d = returns.rolling(5).std().iloc[-1] * np.sqrt(252)
                    volatilities.append(vol_20d)
                    
                    features[f'{asset_class}_volatility'] = vol_20d
                    features[f'{asset_class}_vol_ratio'] = vol_5d / vol_20d if vol_20d > 0 else 1.0
            
            # Cross-asset correlation features
            asset_classes = list(returns_data.keys())
            for i, asset1 in enumerate(asset_classes):
                for j, asset2 in enumerate(asset_classes[i+1:], i+1):
                    if len(returns_data[asset1]) > 20 and len(returns_data[asset2]) > 20:
                        # Align the series
                        common_index = returns_data[asset1].index.intersection(returns_data[asset2].index)
                        if len(common_index) > 20:
                            corr = returns_data[asset1].loc[common_index].corr(
                                returns_data[asset2].loc[common_index]
                            )
                            correlations.append(abs(corr))
                            features[f'{asset1}_{asset2}_correlation'] = corr
            
            # Aggregate features
            if volatilities:
                features['avg_volatility'] = np.mean(volatilities)
                features['max_volatility'] = np.max(volatilities)
                features['vol_dispersion'] = np.std(volatilities)
            
            if correlations:
                features['avg_correlation'] = np.mean(correlations)
                features['max_correlation'] = np.max(correlations)
                features['correlation_dispersion'] = np.std(correlations)
            
            # VIX-like fear index (if available)
            if 'VIX' in market_data and 'close' in market_data['VIX'].columns:
                vix_level = market_data['VIX']['close'].iloc[-1]
                features['fear_index'] = vix_level
                features['fear_percentile'] = self._calculate_percentile(market_data['VIX']['close'], vix_level)
            
            # Economic indicators
            if economic_indicators:
                features.update(economic_indicators)
            
            # Market breadth indicators
            if 'stocks' in market_data:
                features.update(self._calculate_breadth_indicators(market_data['stocks']))
            
        except Exception as e:
            print(f"Feature calculation error: {e}")
        
        return features
    
    def _rule_based_regime_detection(self, features: Dict[str, float]) -> CrossAssetMarketRegime:
        """Rule-based regime detection"""
        try:
            avg_vol = features.get('avg_volatility', 0.2)
            avg_corr = features.get('avg_correlation', 0.5)
            fear_index = features.get('fear_index', 20)
            
            # Crisis regime
            if fear_index > 40 or avg_vol > 0.4:
                return CrossAssetMarketRegime.CRISIS
            
            # High volatility regime
            elif avg_vol > 0.25:
                return CrossAssetMarketRegime.HIGH_VOLATILITY
            
            # Risk-off regime (high correlation, moderate volatility)
            elif avg_corr > 0.7 and avg_vol > 0.15:
                return CrossAssetMarketRegime.RISK_OFF
            
            # Low volatility regime
            elif avg_vol < 0.1:
                return CrossAssetMarketRegime.LOW_VOLATILITY
            
            # Risk-on regime (low correlation, low-moderate volatility)
            elif avg_corr < 0.3 and avg_vol < 0.2:
                return CrossAssetMarketRegime.RISK_ON
            
            else:
                return CrossAssetMarketRegime.NORMAL
        
        except Exception as e:
            print(f"Rule-based regime detection error: {e}")
            return CrossAssetMarketRegime.NORMAL
    
    def _ml_based_regime_detection(self, features: Dict[str, float]) -> CrossAssetMarketRegime:
        """ML-based regime detection using clustering"""
        try:
            # Prepare feature vector
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            
            # Initialize or update regime detector
            if self.regime_detector is None:
                self._initialize_regime_detector()
            
            # Predict regime
            regime_id = self.regime_detector.predict(self.scaler.transform(feature_vector))[0]
            
            # Map cluster ID to regime
            regime_mapping = {
                0: CrossAssetMarketRegime.LOW_VOLATILITY,
                1: CrossAssetMarketRegime.NORMAL,
                2: CrossAssetMarketRegime.HIGH_VOLATILITY,
                3: CrossAssetMarketRegime.RISK_OFF,
                4: CrossAssetMarketRegime.CRISIS
            }
            
            return regime_mapping.get(regime_id, CrossAssetMarketRegime.NORMAL)
        
        except Exception as e:
            print(f"ML-based regime detection error: {e}")
            return self._rule_based_regime_detection(features)
    
    def _initialize_regime_detector(self):
        """Initialize ML regime detector"""
        try:
            # Use historical regime features to train clustering model
            if len(self.regime_history) >= 50:
                # This would typically use historical feature data
                # For now, we'll use a simple KMeans with 5 clusters
                self.regime_detector = KMeans(n_clusters=5, random_state=42, n_init=10)
                
                # In a real implementation, you would fit this on historical data
                # For demonstration, we'll create dummy data
                dummy_features = np.random.randn(100, 10)
                self.scaler.fit(dummy_features)
                self.regime_detector.fit(self.scaler.transform(dummy_features))
        
        except Exception as e:
            print(f"Regime detector initialization error: {e}")
    
    def update_model_performance(self, 
                               model_name: str,
                               performance: CrossAssetModelPerformance):
        """Update model performance history"""
        if model_name not in self.model_performance_history:
            self.model_performance_history[model_name] = []
        
        self.model_performance_history[model_name].append(performance)
        
        # Keep only recent performance history
        if len(self.model_performance_history[model_name]) > self.lookback_window:
            self.model_performance_history[model_name] = \
                self.model_performance_history[model_name][-self.lookback_window:]
    
    def calculate_ensemble_weights(self, 
                                 predictions: List[CrossAssetModelPrediction],
                                 strategy: CrossAssetEnsembleStrategy,
                                 current_regime: CrossAssetMarketRegime) -> Dict[str, float]:
        """Calculate ensemble weights based on strategy"""
        if not predictions:
            return {}
        
        model_names = [pred.model_name for pred in predictions]
        
        if strategy == CrossAssetEnsembleStrategy.EQUAL_WEIGHT:
            return self._equal_weight_strategy(model_names)
        
        elif strategy == CrossAssetEnsembleStrategy.PERFORMANCE_WEIGHTED:
            return self._performance_weighted_strategy(model_names, current_regime)
        
        elif strategy == CrossAssetEnsembleStrategy.VOLATILITY_ADJUSTED:
            return self._volatility_adjusted_strategy(predictions)
        
        elif strategy == CrossAssetEnsembleStrategy.REGIME_AWARE:
            return self._regime_aware_strategy(model_names, current_regime)
        
        elif strategy == CrossAssetEnsembleStrategy.CORRELATION_ADJUSTED:
            return self._correlation_adjusted_strategy(predictions)
        
        elif strategy == CrossAssetEnsembleStrategy.DYNAMIC_WEIGHT:
            return self._dynamic_weight_strategy(predictions, current_regime)
        
        elif strategy == CrossAssetEnsembleStrategy.RISK_PARITY:
            return self._risk_parity_strategy(predictions)
        
        elif strategy == CrossAssetEnsembleStrategy.BAYESIAN_MODEL_AVERAGING:
            return self._bayesian_model_averaging_strategy(predictions, current_regime)
        
        else:
            return self._equal_weight_strategy(model_names)
    
    def _equal_weight_strategy(self, model_names: List[str]) -> Dict[str, float]:
        """Equal weight strategy"""
        weight = 1.0 / len(model_names) if model_names else 0.0
        return {name: weight for name in model_names}
    
    def _performance_weighted_strategy(self, 
                                     model_names: List[str],
                                     current_regime: CrossAssetMarketRegime) -> Dict[str, float]:
        """Performance-weighted strategy"""
        weights = {}
        total_performance = 0.0
        
        for model_name in model_names:
            if model_name in self.model_performance_history:
                recent_performances = self.model_performance_history[model_name][-10:]
                if recent_performances:
                    # Use regime-specific performance if available
                    regime_performance = np.mean([
                        perf.regime_performance.get(current_regime, perf.accuracy)
                        for perf in recent_performances
                    ])
                    performance_score = max(regime_performance, 0.1)  # Minimum weight
                else:
                    performance_score = 0.5  # Default for new models
            else:
                performance_score = 0.5  # Default for unknown models
            
            weights[model_name] = performance_score
            total_performance += performance_score
        
        # Normalize weights
        if total_performance > 0:
            weights = {name: weight / total_performance for name, weight in weights.items()}
        else:
            weights = self._equal_weight_strategy(model_names)
        
        return weights
    
    def _volatility_adjusted_strategy(self, predictions: List[CrossAssetModelPrediction]) -> Dict[str, float]:
        """Volatility-adjusted strategy"""
        weights = {}
        
        for pred in predictions:
            # Use inverse volatility weighting
            volatility = pred.risk_metrics.get('volatility', 0.2) if pred.risk_metrics else 0.2
            inv_vol = 1.0 / max(volatility, 0.01)  # Avoid division by zero
            weights[pred.model_name] = inv_vol
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            weights = self._equal_weight_strategy([pred.model_name for pred in predictions])
        
        return weights
    
    def _regime_aware_strategy(self, 
                             model_names: List[str],
                             current_regime: CrossAssetMarketRegime) -> Dict[str, float]:
        """Regime-aware strategy"""
        weights = {}
        
        # Define regime-specific model preferences
        regime_preferences = {
            CrossAssetMarketRegime.CRISIS: {'momentum': 0.4, 'volatility': 0.3, 'correlation': 0.3},
            CrossAssetMarketRegime.RISK_OFF: {'correlation': 0.4, 'volatility': 0.3, 'momentum': 0.3},
            CrossAssetMarketRegime.RISK_ON: {'momentum': 0.4, 'mean_reversion': 0.3, 'trend': 0.3},
            CrossAssetMarketRegime.HIGH_VOLATILITY: {'volatility': 0.5, 'momentum': 0.3, 'correlation': 0.2},
            CrossAssetMarketRegime.LOW_VOLATILITY: {'mean_reversion': 0.4, 'trend': 0.3, 'momentum': 0.3},
            CrossAssetMarketRegime.NORMAL: {'trend': 0.3, 'momentum': 0.3, 'mean_reversion': 0.2, 'volatility': 0.2}
        }
        
        preferences = regime_preferences.get(current_regime, {})
        
        for model_name in model_names:
            # Assign weights based on model type and regime preferences
            base_weight = 1.0 / len(model_names)
            
            # Adjust weight based on model type (simplified mapping)
            model_type = self._infer_model_type(model_name)
            regime_multiplier = preferences.get(model_type, 1.0)
            
            weights[model_name] = base_weight * regime_multiplier
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            weights = self._equal_weight_strategy(model_names)
        
        return weights
    
    def _correlation_adjusted_strategy(self, predictions: List[CrossAssetModelPrediction]) -> Dict[str, float]:
        """Correlation-adjusted strategy to reduce redundancy"""
        if len(predictions) < 2:
            return self._equal_weight_strategy([pred.model_name for pred in predictions])
        
        # Calculate prediction correlations (simplified)
        pred_values = []
        model_names = []
        
        for pred in predictions:
            if isinstance(pred.prediction, (int, float)):
                pred_values.append(pred.prediction)
                model_names.append(pred.model_name)
        
        if len(pred_values) < 2:
            return self._equal_weight_strategy([pred.model_name for pred in predictions])
        
        # Create correlation matrix (simplified - in practice, use historical predictions)
        correlation_penalty = {}
        for i, name1 in enumerate(model_names):
            penalty = 0.0
            for j, name2 in enumerate(model_names):
                if i != j:
                    # Simplified correlation calculation
                    corr = abs(pred_values[i] - pred_values[j]) / (abs(pred_values[i]) + abs(pred_values[j]) + 1e-8)
                    penalty += (1.0 - corr)  # Higher penalty for similar predictions
            correlation_penalty[name1] = penalty
        
        # Adjust weights inversely to correlation penalty
        weights = {}
        for name in model_names:
            weights[name] = 1.0 / (1.0 + correlation_penalty[name])
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            weights = self._equal_weight_strategy(model_names)
        
        return weights
    
    def _dynamic_weight_strategy(self, 
                               predictions: List[CrossAssetModelPrediction],
                               current_regime: CrossAssetMarketRegime) -> Dict[str, float]:
        """Dynamic weight strategy combining multiple factors"""
        # Combine performance, volatility, and regime awareness
        perf_weights = self._performance_weighted_strategy(
            [pred.model_name for pred in predictions], current_regime
        )
        vol_weights = self._volatility_adjusted_strategy(predictions)
        regime_weights = self._regime_aware_strategy(
            [pred.model_name for pred in predictions], current_regime
        )
        
        # Weighted combination
        combined_weights = {}
        for pred in predictions:
            name = pred.model_name
            combined_weights[name] = (
                0.4 * perf_weights.get(name, 0.0) +
                0.3 * vol_weights.get(name, 0.0) +
                0.3 * regime_weights.get(name, 0.0)
            )
        
        # Normalize
        total_weight = sum(combined_weights.values())
        if total_weight > 0:
            combined_weights = {name: weight / total_weight for name, weight in combined_weights.items()}
        else:
            combined_weights = self._equal_weight_strategy([pred.model_name for pred in predictions])
        
        return combined_weights
    
    def _risk_parity_strategy(self, predictions: List[CrossAssetModelPrediction]) -> Dict[str, float]:
        """Risk parity strategy"""
        weights = {}
        
        for pred in predictions:
            # Use inverse risk weighting
            risk_score = pred.risk_metrics.get('overall_risk', 0.5) if pred.risk_metrics else 0.5
            inv_risk = 1.0 / max(risk_score, 0.01)
            weights[pred.model_name] = inv_risk
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            weights = self._equal_weight_strategy([pred.model_name for pred in predictions])
        
        return weights
    
    def _bayesian_model_averaging_strategy(self, 
                                         predictions: List[CrossAssetModelPrediction],
                                         current_regime: CrossAssetMarketRegime) -> Dict[str, float]:
        """Bayesian Model Averaging strategy"""
        weights = {}
        
        for pred in predictions:
            # Use confidence as proxy for model evidence
            confidence = pred.confidence
            
            # Adjust for regime-specific performance
            model_name = pred.model_name
            if model_name in self.model_performance_history:
                recent_performances = self.model_performance_history[model_name][-5:]
                if recent_performances:
                    regime_performance = np.mean([
                        perf.regime_performance.get(current_regime, perf.accuracy)
                        for perf in recent_performances
                    ])
                    # Combine confidence with historical performance
                    bayesian_weight = confidence * regime_performance
                else:
                    bayesian_weight = confidence
            else:
                bayesian_weight = confidence
            
            weights[pred.model_name] = bayesian_weight
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            weights = self._equal_weight_strategy([pred.model_name for pred in predictions])
        
        return weights
    
    def generate_ensemble_prediction(self, 
                                   predictions: List[CrossAssetModelPrediction],
                                   market_data: Dict[str, pd.DataFrame],
                                   strategy: CrossAssetEnsembleStrategy = CrossAssetEnsembleStrategy.DYNAMIC_WEIGHT,
                                   economic_indicators: Optional[Dict[str, float]] = None) -> CrossAssetEnsembleResult:
        """Generate ensemble prediction with regime awareness"""
        start_time = datetime.now()
        
        if len(predictions) < self.min_models_required:
            raise ValueError(f"Minimum {self.min_models_required} models required for ensemble")
        
        # Detect current market regime
        current_regime = self.detect_market_regime(market_data, economic_indicators)
        
        # Calculate ensemble weights
        model_weights = self.calculate_ensemble_weights(predictions, strategy, current_regime)
        
        # Generate weighted ensemble prediction
        ensemble_pred = self._calculate_weighted_prediction(predictions, model_weights)
        
        # Calculate confidence score
        confidence_score = self._calculate_ensemble_confidence(predictions, model_weights)
        
        # Assess risk
        risk_assessment = self._assess_ensemble_risk(predictions, model_weights)
        
        # Calculate diversification benefit
        diversification_benefit = self._calculate_diversification_benefit(predictions)
        
        # Calculate prediction intervals
        prediction_intervals = self._calculate_prediction_intervals(predictions, model_weights)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Update regime history
        self.regime_history.append((datetime.now(), current_regime))
        if len(self.regime_history) > self.lookback_window:
            self.regime_history = self.regime_history[-self.lookback_window:]
        
        return CrossAssetEnsembleResult(
            ensemble_prediction=ensemble_pred,
            individual_predictions=predictions,
            model_weights=model_weights,
            confidence_score=confidence_score,
            regime=current_regime,
            strategy_used=strategy,
            risk_assessment=risk_assessment,
            diversification_benefit=diversification_benefit,
            prediction_intervals=prediction_intervals,
            execution_time=execution_time,
            metadata={
                'num_models': len(predictions),
                'regime_stability': self._calculate_regime_stability(),
                'weight_concentration': self._calculate_weight_concentration(model_weights)
            }
        )
    
    def _calculate_weighted_prediction(self, 
                                     predictions: List[CrossAssetModelPrediction],
                                     weights: Dict[str, float]) -> Union[float, np.ndarray]:
        """Calculate weighted ensemble prediction"""
        if not predictions:
            return 0.0
        
        # Handle different prediction types
        numeric_predictions = []
        numeric_weights = []
        
        for pred in predictions:
            weight = weights.get(pred.model_name, 0.0)
            if isinstance(pred.prediction, (int, float)) and weight > 0:
                numeric_predictions.append(pred.prediction)
                numeric_weights.append(weight)
        
        if not numeric_predictions:
            return 0.0
        
        # Weighted average
        weighted_sum = sum(pred * weight for pred, weight in zip(numeric_predictions, numeric_weights))
        total_weight = sum(numeric_weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _calculate_ensemble_confidence(self, 
                                     predictions: List[CrossAssetModelPrediction],
                                     weights: Dict[str, float]) -> float:
        """Calculate ensemble confidence score"""
        if not predictions:
            return 0.0
        
        # Weighted average of individual confidences
        weighted_confidence = 0.0
        total_weight = 0.0
        
        for pred in predictions:
            weight = weights.get(pred.model_name, 0.0)
            weighted_confidence += pred.confidence * weight
            total_weight += weight
        
        base_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # Adjust for prediction agreement (consensus bonus)
        consensus_bonus = self._calculate_consensus_bonus(predictions)
        
        # Adjust for regime stability
        regime_stability_bonus = self._calculate_regime_stability() * 0.1
        
        final_confidence = min(1.0, base_confidence + consensus_bonus + regime_stability_bonus)
        return max(0.0, final_confidence)
    
    def _calculate_consensus_bonus(self, predictions: List[CrossAssetModelPrediction]) -> float:
        """Calculate bonus for prediction consensus"""
        if len(predictions) < 2:
            return 0.0
        
        numeric_predictions = [pred.prediction for pred in predictions 
                             if isinstance(pred.prediction, (int, float))]
        
        if len(numeric_predictions) < 2:
            return 0.0
        
        # Calculate coefficient of variation (lower = more consensus)
        mean_pred = np.mean(numeric_predictions)
        std_pred = np.std(numeric_predictions)
        
        if abs(mean_pred) < 1e-8:
            return 0.1 if std_pred < 0.01 else 0.0
        
        cv = std_pred / abs(mean_pred)
        consensus_bonus = max(0.0, 0.2 * (1.0 - min(cv, 1.0)))
        
        return consensus_bonus
    
    def _assess_ensemble_risk(self, 
                            predictions: List[CrossAssetModelPrediction],
                            weights: Dict[str, float]) -> Dict[str, float]:
        """Assess ensemble risk metrics"""
        risk_metrics = {
            'model_risk': 0.5,
            'concentration_risk': 0.5,
            'regime_risk': 0.5,
            'prediction_risk': 0.5,
            'overall_risk': 0.5
        }
        
        try:
            # Model concentration risk
            weight_values = list(weights.values())
            if weight_values:
                max_weight = max(weight_values)
                risk_metrics['concentration_risk'] = max_weight
            
            # Prediction dispersion risk
            numeric_predictions = [pred.prediction for pred in predictions 
                                 if isinstance(pred.prediction, (int, float))]
            if len(numeric_predictions) > 1:
                pred_std = np.std(numeric_predictions)
                pred_mean = np.mean(np.abs(numeric_predictions))
                if pred_mean > 0:
                    risk_metrics['prediction_risk'] = min(1.0, pred_std / pred_mean)
            
            # Regime transition risk
            regime_stability = self._calculate_regime_stability()
            risk_metrics['regime_risk'] = 1.0 - regime_stability
            
            # Overall risk
            risk_metrics['overall_risk'] = np.mean([
                risk_metrics['model_risk'],
                risk_metrics['concentration_risk'],
                risk_metrics['regime_risk'],
                risk_metrics['prediction_risk']
            ])
        
        except Exception as e:
            print(f"Risk assessment error: {e}")
        
        return risk_metrics
    
    def _calculate_diversification_benefit(self, predictions: List[CrossAssetModelPrediction]) -> float:
        """Calculate diversification benefit of ensemble"""
        if len(predictions) < 2:
            return 0.0
        
        try:
            # Simple diversification measure based on prediction spread
            numeric_predictions = [pred.prediction for pred in predictions 
                                 if isinstance(pred.prediction, (int, float))]
            
            if len(numeric_predictions) < 2:
                return 0.0
            
            # Higher spread = better diversification
            pred_range = max(numeric_predictions) - min(numeric_predictions)
            pred_mean = np.mean(np.abs(numeric_predictions))
            
            if pred_mean > 0:
                diversification_ratio = pred_range / pred_mean
                return min(1.0, diversification_ratio / 2.0)  # Normalize
            else:
                return 0.5
        
        except Exception as e:
            print(f"Diversification calculation error: {e}")
            return 0.0
    
    def _calculate_prediction_intervals(self, 
                                      predictions: List[CrossAssetModelPrediction],
                                      weights: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Calculate prediction intervals"""
        intervals = {}
        
        try:
            numeric_predictions = [pred.prediction for pred in predictions 
                                 if isinstance(pred.prediction, (int, float))]
            
            if len(numeric_predictions) < 2:
                return {'95%': (0.0, 0.0), '68%': (0.0, 0.0)}
            
            # Calculate weighted statistics
            weighted_mean = self._calculate_weighted_prediction(predictions, weights)
            
            # Calculate weighted standard deviation
            weighted_var = 0.0
            total_weight = 0.0
            
            for pred in predictions:
                if isinstance(pred.prediction, (int, float)):
                    weight = weights.get(pred.model_name, 0.0)
                    weighted_var += weight * (pred.prediction - weighted_mean) ** 2
                    total_weight += weight
            
            if total_weight > 0:
                weighted_std = np.sqrt(weighted_var / total_weight)
            else:
                weighted_std = np.std(numeric_predictions)
            
            # Calculate intervals
            intervals['68%'] = (
                weighted_mean - weighted_std,
                weighted_mean + weighted_std
            )
            intervals['95%'] = (
                weighted_mean - 1.96 * weighted_std,
                weighted_mean + 1.96 * weighted_std
            )
        
        except Exception as e:
            print(f"Prediction interval calculation error: {e}")
            intervals = {'95%': (0.0, 0.0), '68%': (0.0, 0.0)}
        
        return intervals
    
    def _calculate_regime_stability(self) -> float:
        """Calculate regime stability score"""
        if len(self.regime_history) < 10:
            return 0.5  # Default for insufficient history
        
        # Count regime changes in recent history
        recent_regimes = [regime for _, regime in self.regime_history[-20:]]
        regime_changes = sum(1 for i in range(1, len(recent_regimes)) 
                           if recent_regimes[i] != recent_regimes[i-1])
        
        # Stability = 1 - (change_rate)
        change_rate = regime_changes / (len(recent_regimes) - 1)
        stability = max(0.0, 1.0 - change_rate)
        
        return stability
    
    def _calculate_weight_concentration(self, weights: Dict[str, float]) -> float:
        """Calculate weight concentration (Herfindahl index)"""
        if not weights:
            return 1.0
        
        # Herfindahl-Hirschman Index
        hhi = sum(weight ** 2 for weight in weights.values())
        return hhi
    
    def _infer_model_type(self, model_name: str) -> str:
        """Infer model type from model name (simplified)"""
        name_lower = model_name.lower()
        
        if 'momentum' in name_lower or 'trend' in name_lower:
            return 'momentum'
        elif 'volatility' in name_lower or 'vol' in name_lower:
            return 'volatility'
        elif 'correlation' in name_lower or 'corr' in name_lower:
            return 'correlation'
        elif 'reversion' in name_lower or 'mean' in name_lower:
            return 'mean_reversion'
        else:
            return 'trend'  # Default
    
    def _calculate_percentile(self, series: pd.Series, value: float) -> float:
        """Calculate percentile of value in series"""
        try:
            return (series <= value).mean()
        except:
            return 0.5
    
    def _calculate_breadth_indicators(self, stock_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate market breadth indicators"""
        indicators = {}
        
        try:
            if 'close' in stock_data.columns and len(stock_data) > 20:
                returns = stock_data['close'].pct_change().dropna()
                
                # Advance-decline ratio (simplified)
                positive_days = (returns > 0).sum()
                total_days = len(returns)
                indicators['advance_decline_ratio'] = positive_days / total_days if total_days > 0 else 0.5
                
                # Momentum breadth
                recent_returns = returns.tail(20)
                positive_momentum = (recent_returns > 0).sum()
                indicators['momentum_breadth'] = positive_momentum / len(recent_returns) if len(recent_returns) > 0 else 0.5
        
        except Exception as e:
            print(f"Breadth indicator calculation error: {e}")
        
        return indicators

# Example usage
if __name__ == "__main__":
    # Initialize ensemble predictor
    ensemble_predictor = CrossAssetAdvancedEnsemblePredictor()
    
    # Sample market data
    sample_market_data = {
        'stocks': pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        }),
        'bonds': pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 50,
            'volume': np.random.randint(500000, 5000000, 100)
        }),
        'commodities': pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 150,
            'volume': np.random.randint(100000, 1000000, 100)
        }),
        'VIX': pd.DataFrame({
            'close': np.random.uniform(10, 40, 100)
        })
    }
    
    # Sample predictions
    sample_predictions = [
        CrossAssetModelPrediction(
            model_name="LSTM_Model",
            prediction=0.05,
            confidence=0.8,
            timestamp=datetime.now(),
            asset_class="stocks",
            prediction_horizon="1d",
            metadata={"model_type": "neural_network"},
            risk_metrics={"volatility": 0.15, "overall_risk": 0.4}
        ),
        CrossAssetModelPrediction(
            model_name="Momentum_Model",
            prediction=0.03,
            confidence=0.7,
            timestamp=datetime.now(),
            asset_class="stocks",
            prediction_horizon="1d",
            metadata={"model_type": "momentum"},
            risk_metrics={"volatility": 0.12, "overall_risk": 0.3}
        ),
        CrossAssetModelPrediction(
            model_name="Correlation_Model",
            prediction=0.02,
            confidence=0.6,
            timestamp=datetime.now(),
            asset_class="multi_asset",
            prediction_horizon="1d",
            metadata={"model_type": "correlation"},
            risk_metrics={"volatility": 0.18, "overall_risk": 0.5}
        )
    ]
    
    # Generate ensemble prediction
    result = ensemble_predictor.generate_ensemble_prediction(
        predictions=sample_predictions,
        market_data=sample_market_data,
        strategy=CrossAssetEnsembleStrategy.DYNAMIC_WEIGHT
    )
    
    print(f"Ensemble Prediction: {result.ensemble_prediction:.4f}")
    print(f"Confidence Score: {result.confidence_score:.4f}")
    print(f"Market Regime: {result.regime.value}")
    print(f"Strategy Used: {result.strategy_used.value}")
    print(f"Model Weights: {result.model_weights}")
    print(f"Risk Assessment: {result.risk_assessment}")
    print(f"Diversification Benefit: {result.diversification_benefit:.4f}")
    print(f"Execution Time: {result.execution_time:.4f}s")