from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime

# Import existing cross-asset models
from .comprehensive_analyzer import CrossAssetAnalyzer
from .cross_asset_comprehensive import CrossAssetComprehensiveIndicators
from .ml_models import CrossAssetMLModels
from .nlp_models import CrossAssetNLPModels
from .advanced_ml_models import CrossAssetAdvancedMLModels

class CrossAssetModelCategory(Enum):
    """Categories of cross-asset models"""
    TRADITIONAL = "traditional"
    MACHINE_LEARNING = "machine_learning"
    ADVANCED_ML = "advanced_ml"
    NLP = "nlp"
    COMPREHENSIVE = "comprehensive"

class CrossAssetModelType(Enum):
    """Types of cross-asset models"""
    # Traditional models
    CORRELATION_ANALYSIS = "correlation_analysis"
    VOLATILITY_ANALYSIS = "volatility_analysis"
    FACTOR_ANALYSIS = "factor_analysis"
    REGIME_DETECTION = "regime_detection"
    
    # ML models
    LSTM = "lstm"
    RANDOM_FOREST = "random_forest"
    SVM = "svm"
    GRADIENT_BOOSTING = "gradient_boosting"
    
    # Advanced ML models
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"
    
    # NLP models
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    NEWS_IMPACT = "news_impact"
    TEXT_CLASSIFICATION = "text_classification"
    
    # Comprehensive models
    MULTI_ASSET_ANALYSIS = "multi_asset_analysis"
    INTEGRATED_PREDICTION = "integrated_prediction"

@dataclass
class CrossAssetPrediction:
    """Individual cross-asset model prediction"""
    model_type: CrossAssetModelType
    prediction: Union[float, np.ndarray, Dict[str, Any]]
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any]
    asset_correlations: Optional[Dict[str, float]] = None
    risk_metrics: Optional[Dict[str, float]] = None

@dataclass
class CrossAssetModelStatus:
    """Status of a cross-asset model"""
    model_type: CrossAssetModelType
    is_trained: bool
    last_updated: datetime
    performance_metrics: Dict[str, float]
    data_requirements: List[str]
    computational_cost: str  # 'low', 'medium', 'high'

@dataclass
class CrossAssetUnifiedResult:
    """Unified result from cross-asset analysis"""
    individual_predictions: List[CrossAssetPrediction]
    ensemble_prediction: Optional[CrossAssetPrediction]
    model_weights: Dict[CrossAssetModelType, float]
    consensus_score: float
    risk_assessment: Dict[str, float]
    asset_allocation_suggestions: Dict[str, float]
    market_regime: str
    confidence_intervals: Dict[str, tuple]
    execution_time: float

class CrossAssetUnifiedInterface:
    """Unified interface for all cross-asset models"""
    
    def __init__(self):
        self.models = {}
        self.model_status = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all cross-asset models"""
        try:
            # Traditional models
            self.models[CrossAssetModelCategory.COMPREHENSIVE] = CrossAssetComprehensiveIndicators()
            self.models[CrossAssetModelCategory.TRADITIONAL] = CrossAssetAnalyzer()
            
            # ML models
            self.models[CrossAssetModelCategory.MACHINE_LEARNING] = CrossAssetMLModels()
            
            # Advanced ML models
            self.models[CrossAssetModelCategory.ADVANCED_ML] = CrossAssetAdvancedMLModels()
            
            # NLP models
            self.models[CrossAssetModelCategory.NLP] = CrossAssetNLPModels()
            
            # Initialize model status
            for category in CrossAssetModelCategory:
                self.model_status[category] = CrossAssetModelStatus(
                    model_type=CrossAssetModelType.MULTI_ASSET_ANALYSIS,
                    is_trained=False,
                    last_updated=datetime.now(),
                    performance_metrics={},
                    data_requirements=["price_data", "volume_data", "economic_indicators"],
                    computational_cost="medium"
                )
        except Exception as e:
            print(f"Warning: Some models could not be initialized: {e}")
    
    def predict_single_model(self, 
                           model_type: CrossAssetModelType,
                           data: Dict[str, pd.DataFrame],
                           **kwargs) -> CrossAssetPrediction:
        """Get prediction from a single cross-asset model"""
        try:
            category = self._get_model_category(model_type)
            model = self.models.get(category)
            
            if not model:
                raise ValueError(f"Model category {category} not available")
            
            # Route to appropriate prediction method based on model type
            if model_type == CrossAssetModelType.CORRELATION_ANALYSIS:
                result = self._predict_correlation_analysis(model, data, **kwargs)
            elif model_type == CrossAssetModelType.LSTM:
                result = self._predict_lstm(model, data, **kwargs)
            elif model_type == CrossAssetModelType.TRANSFORMER:
                result = self._predict_transformer(model, data, **kwargs)
            elif model_type == CrossAssetModelType.SENTIMENT_ANALYSIS:
                result = self._predict_sentiment(model, data, **kwargs)
            else:
                result = self._predict_generic(model, data, model_type, **kwargs)
            
            return CrossAssetPrediction(
                model_type=model_type,
                prediction=result.get('prediction', 0.0),
                confidence=result.get('confidence', 0.5),
                timestamp=datetime.now(),
                metadata=result.get('metadata', {}),
                asset_correlations=result.get('correlations'),
                risk_metrics=result.get('risk_metrics')
            )
        
        except Exception as e:
            return CrossAssetPrediction(
                model_type=model_type,
                prediction=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={'error': str(e)}
            )
    
    def predict_category(self, 
                        category: CrossAssetModelCategory,
                        data: Dict[str, pd.DataFrame],
                        **kwargs) -> List[CrossAssetPrediction]:
        """Get predictions from all models in a category"""
        model_types = self._get_models_by_category(category)
        predictions = []
        
        for model_type in model_types:
            prediction = self.predict_single_model(model_type, data, **kwargs)
            predictions.append(prediction)
        
        return predictions
    
    def predict_unified(self, 
                       data: Dict[str, pd.DataFrame],
                       model_types: Optional[List[CrossAssetModelType]] = None,
                       ensemble_method: str = 'weighted_average',
                       **kwargs) -> CrossAssetUnifiedResult:
        """Get unified prediction from multiple cross-asset models"""
        start_time = datetime.now()
        
        # Use all models if none specified
        if model_types is None:
            model_types = list(CrossAssetModelType)
        
        # Get individual predictions
        individual_predictions = []
        for model_type in model_types:
            prediction = self.predict_single_model(model_type, data, **kwargs)
            individual_predictions.append(prediction)
        
        # Calculate model weights based on confidence and historical performance
        model_weights = self._calculate_model_weights(individual_predictions)
        
        # Generate ensemble prediction
        ensemble_prediction = self._generate_ensemble_prediction(
            individual_predictions, model_weights, ensemble_method
        )
        
        # Calculate consensus score
        consensus_score = self._calculate_consensus_score(individual_predictions)
        
        # Assess risk
        risk_assessment = self._assess_cross_asset_risk(individual_predictions, data)
        
        # Generate asset allocation suggestions
        allocation_suggestions = self._generate_allocation_suggestions(
            individual_predictions, risk_assessment
        )
        
        # Detect market regime
        market_regime = self._detect_market_regime(data, individual_predictions)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(individual_predictions)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return CrossAssetUnifiedResult(
            individual_predictions=individual_predictions,
            ensemble_prediction=ensemble_prediction,
            model_weights=model_weights,
            consensus_score=consensus_score,
            risk_assessment=risk_assessment,
            asset_allocation_suggestions=allocation_suggestions,
            market_regime=market_regime,
            confidence_intervals=confidence_intervals,
            execution_time=execution_time
        )
    
    def _get_model_category(self, model_type: CrossAssetModelType) -> CrossAssetModelCategory:
        """Get the category for a model type"""
        category_mapping = {
            CrossAssetModelType.CORRELATION_ANALYSIS: CrossAssetModelCategory.TRADITIONAL,
            CrossAssetModelType.VOLATILITY_ANALYSIS: CrossAssetModelCategory.TRADITIONAL,
            CrossAssetModelType.FACTOR_ANALYSIS: CrossAssetModelCategory.TRADITIONAL,
            CrossAssetModelType.REGIME_DETECTION: CrossAssetModelCategory.TRADITIONAL,
            CrossAssetModelType.LSTM: CrossAssetModelCategory.MACHINE_LEARNING,
            CrossAssetModelType.RANDOM_FOREST: CrossAssetModelCategory.MACHINE_LEARNING,
            CrossAssetModelType.SVM: CrossAssetModelCategory.MACHINE_LEARNING,
            CrossAssetModelType.GRADIENT_BOOSTING: CrossAssetModelCategory.MACHINE_LEARNING,
            CrossAssetModelType.TRANSFORMER: CrossAssetModelCategory.ADVANCED_ML,
            CrossAssetModelType.ENSEMBLE: CrossAssetModelCategory.ADVANCED_ML,
            CrossAssetModelType.DEEP_LEARNING: CrossAssetModelCategory.ADVANCED_ML,
            CrossAssetModelType.SENTIMENT_ANALYSIS: CrossAssetModelCategory.NLP,
            CrossAssetModelType.NEWS_IMPACT: CrossAssetModelCategory.NLP,
            CrossAssetModelType.TEXT_CLASSIFICATION: CrossAssetModelCategory.NLP,
            CrossAssetModelType.MULTI_ASSET_ANALYSIS: CrossAssetModelCategory.COMPREHENSIVE,
            CrossAssetModelType.INTEGRATED_PREDICTION: CrossAssetModelCategory.COMPREHENSIVE,
        }
        return category_mapping.get(model_type, CrossAssetModelCategory.TRADITIONAL)
    
    def _get_models_by_category(self, category: CrossAssetModelCategory) -> List[CrossAssetModelType]:
        """Get all model types in a category"""
        models_by_category = {
            CrossAssetModelCategory.TRADITIONAL: [
                CrossAssetModelType.CORRELATION_ANALYSIS,
                CrossAssetModelType.VOLATILITY_ANALYSIS,
                CrossAssetModelType.FACTOR_ANALYSIS,
                CrossAssetModelType.REGIME_DETECTION
            ],
            CrossAssetModelCategory.MACHINE_LEARNING: [
                CrossAssetModelType.LSTM,
                CrossAssetModelType.RANDOM_FOREST,
                CrossAssetModelType.SVM,
                CrossAssetModelType.GRADIENT_BOOSTING
            ],
            CrossAssetModelCategory.ADVANCED_ML: [
                CrossAssetModelType.TRANSFORMER,
                CrossAssetModelType.ENSEMBLE,
                CrossAssetModelType.DEEP_LEARNING
            ],
            CrossAssetModelCategory.NLP: [
                CrossAssetModelType.SENTIMENT_ANALYSIS,
                CrossAssetModelType.NEWS_IMPACT,
                CrossAssetModelType.TEXT_CLASSIFICATION
            ],
            CrossAssetModelCategory.COMPREHENSIVE: [
                CrossAssetModelType.MULTI_ASSET_ANALYSIS,
                CrossAssetModelType.INTEGRATED_PREDICTION
            ]
        }
        return models_by_category.get(category, [])
    
    def _predict_correlation_analysis(self, model, data: Dict[str, pd.DataFrame], **kwargs):
        """Predict using correlation analysis"""
        try:
            # Implement correlation analysis prediction
            correlations = {}
            for asset1 in data.keys():
                for asset2 in data.keys():
                    if asset1 != asset2 and 'close' in data[asset1].columns and 'close' in data[asset2].columns:
                        corr = data[asset1]['close'].corr(data[asset2]['close'])
                        correlations[f"{asset1}_{asset2}"] = corr
            
            # Generate prediction based on correlation strength
            avg_correlation = np.mean(list(correlations.values()))
            prediction = avg_correlation
            confidence = min(abs(avg_correlation), 1.0)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'correlations': correlations,
                'metadata': {'analysis_type': 'correlation'}
            }
        except Exception as e:
            return {'prediction': 0.0, 'confidence': 0.0, 'metadata': {'error': str(e)}}
    
    def _predict_lstm(self, model, data: Dict[str, pd.DataFrame], **kwargs):
        """Predict using LSTM model"""
        try:
            if hasattr(model, 'predict_lstm'):
                result = model.predict_lstm(data, **kwargs)
                return {
                    'prediction': result.get('prediction', 0.0),
                    'confidence': result.get('confidence', 0.5),
                    'metadata': {'model_type': 'lstm'}
                }
            else:
                return {'prediction': 0.0, 'confidence': 0.0, 'metadata': {'error': 'LSTM method not available'}}
        except Exception as e:
            return {'prediction': 0.0, 'confidence': 0.0, 'metadata': {'error': str(e)}}
    
    def _predict_transformer(self, model, data: Dict[str, pd.DataFrame], **kwargs):
        """Predict using Transformer model"""
        try:
            if hasattr(model, 'predict_transformer'):
                result = model.predict_transformer(data, **kwargs)
                return {
                    'prediction': result.get('prediction', 0.0),
                    'confidence': result.get('confidence', 0.5),
                    'metadata': {'model_type': 'transformer'}
                }
            else:
                return {'prediction': 0.0, 'confidence': 0.0, 'metadata': {'error': 'Transformer method not available'}}
        except Exception as e:
            return {'prediction': 0.0, 'confidence': 0.0, 'metadata': {'error': str(e)}}
    
    def _predict_sentiment(self, model, data: Dict[str, pd.DataFrame], **kwargs):
        """Predict using sentiment analysis"""
        try:
            if hasattr(model, 'analyze_sentiment'):
                result = model.analyze_sentiment(data, **kwargs)
                return {
                    'prediction': result.get('sentiment_score', 0.0),
                    'confidence': result.get('confidence', 0.5),
                    'metadata': {'model_type': 'sentiment'}
                }
            else:
                return {'prediction': 0.0, 'confidence': 0.0, 'metadata': {'error': 'Sentiment method not available'}}
        except Exception as e:
            return {'prediction': 0.0, 'confidence': 0.0, 'metadata': {'error': str(e)}}
    
    def _predict_generic(self, model, data: Dict[str, pd.DataFrame], model_type: CrossAssetModelType, **kwargs):
        """Generic prediction method"""
        try:
            # Try common method names
            method_names = ['predict', 'analyze', 'calculate', 'process']
            
            for method_name in method_names:
                if hasattr(model, method_name):
                    method = getattr(model, method_name)
                    result = method(data, **kwargs)
                    
                    if isinstance(result, dict):
                        return result
                    else:
                        return {
                            'prediction': result if isinstance(result, (int, float)) else 0.0,
                            'confidence': 0.5,
                            'metadata': {'model_type': model_type.value}
                        }
            
            return {'prediction': 0.0, 'confidence': 0.0, 'metadata': {'error': 'No suitable method found'}}
        except Exception as e:
            return {'prediction': 0.0, 'confidence': 0.0, 'metadata': {'error': str(e)}}
    
    def _calculate_model_weights(self, predictions: List[CrossAssetPrediction]) -> Dict[CrossAssetModelType, float]:
        """Calculate weights for ensemble based on confidence and performance"""
        weights = {}
        total_confidence = sum(pred.confidence for pred in predictions)
        
        if total_confidence == 0:
            # Equal weights if no confidence information
            equal_weight = 1.0 / len(predictions) if predictions else 0.0
            for pred in predictions:
                weights[pred.model_type] = equal_weight
        else:
            # Weight by confidence
            for pred in predictions:
                weights[pred.model_type] = pred.confidence / total_confidence
        
        return weights
    
    def _generate_ensemble_prediction(self, 
                                    predictions: List[CrossAssetPrediction],
                                    weights: Dict[CrossAssetModelType, float],
                                    method: str) -> CrossAssetPrediction:
        """Generate ensemble prediction"""
        if not predictions:
            return CrossAssetPrediction(
                model_type=CrossAssetModelType.ENSEMBLE,
                prediction=0.0,
                confidence=0.0,
                timestamp=datetime.now(),
                metadata={'method': method}
            )
        
        if method == 'weighted_average':
            weighted_sum = 0.0
            total_weight = 0.0
            
            for pred in predictions:
                if isinstance(pred.prediction, (int, float)):
                    weight = weights.get(pred.model_type, 0.0)
                    weighted_sum += pred.prediction * weight
                    total_weight += weight
            
            ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
            ensemble_confidence = np.mean([pred.confidence for pred in predictions])
        
        elif method == 'median':
            numeric_predictions = [pred.prediction for pred in predictions 
                                 if isinstance(pred.prediction, (int, float))]
            ensemble_prediction = np.median(numeric_predictions) if numeric_predictions else 0.0
            ensemble_confidence = np.mean([pred.confidence for pred in predictions])
        
        else:  # simple average
            numeric_predictions = [pred.prediction for pred in predictions 
                                 if isinstance(pred.prediction, (int, float))]
            ensemble_prediction = np.mean(numeric_predictions) if numeric_predictions else 0.0
            ensemble_confidence = np.mean([pred.confidence for pred in predictions])
        
        return CrossAssetPrediction(
            model_type=CrossAssetModelType.ENSEMBLE,
            prediction=ensemble_prediction,
            confidence=ensemble_confidence,
            timestamp=datetime.now(),
            metadata={'method': method, 'num_models': len(predictions)}
        )
    
    def _calculate_consensus_score(self, predictions: List[CrossAssetPrediction]) -> float:
        """Calculate consensus score among predictions"""
        if len(predictions) < 2:
            return 1.0
        
        numeric_predictions = [pred.prediction for pred in predictions 
                             if isinstance(pred.prediction, (int, float))]
        
        if len(numeric_predictions) < 2:
            return 0.5
        
        # Calculate standard deviation as inverse of consensus
        std_dev = np.std(numeric_predictions)
        max_possible_std = np.std([min(numeric_predictions), max(numeric_predictions)])
        
        if max_possible_std == 0:
            return 1.0
        
        consensus_score = 1.0 - (std_dev / max_possible_std)
        return max(0.0, min(1.0, consensus_score))
    
    def _assess_cross_asset_risk(self, 
                               predictions: List[CrossAssetPrediction],
                               data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Assess cross-asset risk metrics"""
        risk_metrics = {
            'volatility_risk': 0.5,
            'correlation_risk': 0.5,
            'liquidity_risk': 0.5,
            'concentration_risk': 0.5,
            'overall_risk': 0.5
        }
        
        try:
            # Calculate volatility risk
            volatilities = []
            for asset_data in data.values():
                if 'close' in asset_data.columns and len(asset_data) > 1:
                    returns = asset_data['close'].pct_change().dropna()
                    vol = returns.std() * np.sqrt(252)  # Annualized volatility
                    volatilities.append(vol)
            
            if volatilities:
                avg_volatility = np.mean(volatilities)
                risk_metrics['volatility_risk'] = min(avg_volatility / 0.5, 1.0)  # Normalize
            
            # Calculate correlation risk (high correlation = high risk)
            correlations = []
            asset_names = list(data.keys())
            for i, asset1 in enumerate(asset_names):
                for j, asset2 in enumerate(asset_names[i+1:], i+1):
                    if ('close' in data[asset1].columns and 
                        'close' in data[asset2].columns):
                        corr = abs(data[asset1]['close'].corr(data[asset2]['close']))
                        correlations.append(corr)
            
            if correlations:
                avg_correlation = np.mean(correlations)
                risk_metrics['correlation_risk'] = avg_correlation
            
            # Overall risk as weighted average
            risk_metrics['overall_risk'] = (
                0.4 * risk_metrics['volatility_risk'] +
                0.3 * risk_metrics['correlation_risk'] +
                0.2 * risk_metrics['liquidity_risk'] +
                0.1 * risk_metrics['concentration_risk']
            )
        
        except Exception as e:
            print(f"Risk assessment error: {e}")
        
        return risk_metrics
    
    def _generate_allocation_suggestions(self, 
                                       predictions: List[CrossAssetPrediction],
                                       risk_assessment: Dict[str, float]) -> Dict[str, float]:
        """Generate asset allocation suggestions"""
        # Simple allocation based on prediction confidence and risk
        allocations = {
            'stocks': 0.4,
            'bonds': 0.3,
            'commodities': 0.2,
            'cash': 0.1
        }
        
        try:
            # Adjust based on overall risk
            overall_risk = risk_assessment.get('overall_risk', 0.5)
            
            if overall_risk > 0.7:  # High risk - more conservative
                allocations['cash'] += 0.1
                allocations['bonds'] += 0.1
                allocations['stocks'] -= 0.15
                allocations['commodities'] -= 0.05
            elif overall_risk < 0.3:  # Low risk - more aggressive
                allocations['stocks'] += 0.1
                allocations['commodities'] += 0.05
                allocations['bonds'] -= 0.1
                allocations['cash'] -= 0.05
            
            # Ensure allocations sum to 1 and are non-negative
            total = sum(allocations.values())
            allocations = {k: max(0, v/total) for k, v in allocations.items()}
        
        except Exception as e:
            print(f"Allocation suggestion error: {e}")
        
        return allocations
    
    def _detect_market_regime(self, 
                            data: Dict[str, pd.DataFrame],
                            predictions: List[CrossAssetPrediction]) -> str:
        """Detect current market regime"""
        try:
            # Simple regime detection based on volatility and correlations
            volatilities = []
            for asset_data in data.values():
                if 'close' in asset_data.columns and len(asset_data) > 20:
                    returns = asset_data['close'].pct_change().dropna()
                    vol = returns.rolling(20).std().iloc[-1]
                    volatilities.append(vol)
            
            if volatilities:
                avg_volatility = np.mean(volatilities)
                if avg_volatility > 0.03:
                    return "high_volatility"
                elif avg_volatility < 0.01:
                    return "low_volatility"
                else:
                    return "normal"
            
            return "unknown"
        
        except Exception as e:
            print(f"Regime detection error: {e}")
            return "unknown"
    
    def _calculate_confidence_intervals(self, 
                                      predictions: List[CrossAssetPrediction]) -> Dict[str, tuple]:
        """Calculate confidence intervals for predictions"""
        intervals = {}
        
        try:
            numeric_predictions = [pred.prediction for pred in predictions 
                                 if isinstance(pred.prediction, (int, float))]
            
            if numeric_predictions:
                mean_pred = np.mean(numeric_predictions)
                std_pred = np.std(numeric_predictions)
                
                # 95% confidence interval
                lower_bound = mean_pred - 1.96 * std_pred
                upper_bound = mean_pred + 1.96 * std_pred
                
                intervals['95%'] = (lower_bound, upper_bound)
                
                # 68% confidence interval
                lower_bound_68 = mean_pred - std_pred
                upper_bound_68 = mean_pred + std_pred
                
                intervals['68%'] = (lower_bound_68, upper_bound_68)
        
        except Exception as e:
            print(f"Confidence interval calculation error: {e}")
            intervals['95%'] = (0.0, 0.0)
            intervals['68%'] = (0.0, 0.0)
        
        return intervals
    
    def get_model_status(self, model_type: Optional[CrossAssetModelType] = None) -> Union[CrossAssetModelStatus, Dict[CrossAssetModelType, CrossAssetModelStatus]]:
        """Get status of models"""
        if model_type:
            category = self._get_model_category(model_type)
            return self.model_status.get(category)
        else:
            return self.model_status
    
    def update_model_performance(self, 
                               model_type: CrossAssetModelType,
                               performance_metrics: Dict[str, float]):
        """Update model performance metrics"""
        category = self._get_model_category(model_type)
        if category in self.model_status:
            self.model_status[category].performance_metrics.update(performance_metrics)
            self.model_status[category].last_updated = datetime.now()

# Example usage
if __name__ == "__main__":
    # Initialize unified interface
    unified_interface = CrossAssetUnifiedInterface()
    
    # Sample data
    sample_data = {
        'SPY': pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000000, 10000000, 100)
        }),
        'TLT': pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 50,
            'volume': np.random.randint(500000, 5000000, 100)
        }),
        'GLD': pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 150,
            'volume': np.random.randint(100000, 1000000, 100)
        })
    }
    
    # Get unified prediction
    result = unified_interface.predict_unified(
        data=sample_data,
        model_types=[CrossAssetModelType.CORRELATION_ANALYSIS, CrossAssetModelType.LSTM],
        ensemble_method='weighted_average'
    )
    
    print(f"Ensemble Prediction: {result.ensemble_prediction.prediction}")
    print(f"Consensus Score: {result.consensus_score}")
    print(f"Market Regime: {result.market_regime}")
    print(f"Asset Allocation: {result.asset_allocation_suggestions}")
    print(f"Execution Time: {result.execution_time:.2f}s")