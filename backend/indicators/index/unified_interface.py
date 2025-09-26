"""Unified Index Indicator Interface

This module provides a unified interface for all index indicator models,
allowing seamless integration and consistent API across different model types.

Author: Trading Platform Team
Date: 2024
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime

# Import all model classes
from apt_model import AdvancedArbitragePricingTheory
from term_structure_model import AdvancedTermStructureModel
from macro_factor_model import MacroeconomicFactorModel
from ml_models import AdvancedMLModels, IndexIndicatorType, IndexData, MacroeconomicData, MLResult


class ModelType(Enum):
    """Available model types"""
    APT = "apt"
    TERM_STRUCTURE = "term_structure"
    MACROECONOMIC = "macroeconomic"
    ML_LSTM = "ml_lstm"
    ML_TRANSFORMER = "ml_transformer"
    ML_ENSEMBLE = "ml_ensemble"


@dataclass
class UnifiedResult:
    """Unified result structure for all models"""
    model_type: ModelType
    fair_value: float
    signal: str  # BUY, SELL, HOLD
    confidence: float
    risk_level: str  # LOW, MEDIUM, HIGH
    timestamp: datetime
    additional_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "model_type": self.model_type.value,
            "fair_value": self.fair_value,
            "signal": self.signal,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "timestamp": self.timestamp.isoformat(),
            "additional_metrics": self.additional_metrics
        }


@dataclass
class EnsembleResult:
    """Ensemble result combining multiple models"""
    consensus_fair_value: float
    consensus_signal: str
    consensus_confidence: float
    consensus_risk_level: str
    individual_results: List[UnifiedResult]
    model_weights: Dict[ModelType, float]
    disagreement_score: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ensemble result to dictionary"""
        return {
            "consensus_fair_value": self.consensus_fair_value,
            "consensus_signal": self.consensus_signal,
            "consensus_confidence": self.consensus_confidence,
            "consensus_risk_level": self.consensus_risk_level,
            "individual_results": [result.to_dict() for result in self.individual_results],
            "model_weights": {k.value: v for k, v in self.model_weights.items()},
            "disagreement_score": self.disagreement_score,
            "timestamp": self.timestamp.isoformat()
        }


class UnifiedIndexIndicator:
    """Unified interface for all index indicator models"""
    
    def __init__(self, 
                 enabled_models: Optional[List[ModelType]] = None,
                 model_weights: Optional[Dict[ModelType, float]] = None,
                 confidence_threshold: float = 0.6,
                 ensemble_mode: bool = True):
        """
        Initialize unified index indicator
        
        Args:
            enabled_models: List of models to enable (default: all)
            model_weights: Weights for ensemble (default: equal weights)
            confidence_threshold: Minimum confidence for signals
            ensemble_mode: Whether to use ensemble predictions
        """
        self.enabled_models = enabled_models or list(ModelType)
        self.confidence_threshold = confidence_threshold
        self.ensemble_mode = ensemble_mode
        
        # Set default equal weights
        if model_weights is None:
            weight = 1.0 / len(self.enabled_models)
            self.model_weights = {model: weight for model in self.enabled_models}
        else:
            self.model_weights = model_weights
            # Normalize weights
            total_weight = sum(self.model_weights.values())
            self.model_weights = {k: v / total_weight for k, v in self.model_weights.items()}
        
        # Initialize models
        self.models = self._initialize_models()
        
        # Performance tracking
        self.prediction_history: List[Union[UnifiedResult, EnsembleResult]] = []
        self.performance_metrics = {
            "total_predictions": 0,
            "correct_signals": 0,
            "average_confidence": 0.0,
            "model_performance": {model.value: {"accuracy": 0.0, "count": 0} for model in self.enabled_models}
        }
    
    def _initialize_models(self) -> Dict[ModelType, Any]:
        """Initialize all enabled models"""
        models = {}
        
        for model_type in self.enabled_models:
            try:
                if model_type == ModelType.APT:
                    models[model_type] = AdvancedArbitragePricingTheory()
                elif model_type == ModelType.TERM_STRUCTURE:
                    models[model_type] = AdvancedTermStructureModel()
                elif model_type == ModelType.MACROECONOMIC:
                    models[model_type] = MacroeconomicFactorModel()
                elif model_type in [ModelType.ML_LSTM, ModelType.ML_TRANSFORMER, ModelType.ML_ENSEMBLE]:
                    models[model_type] = AdvancedMLModels()
                    
            except Exception as e:
                print(f"Warning: Failed to initialize {model_type.value} model: {e}")
                continue
        
        return models
    
    def predict(self, 
                index_data: IndexData, 
                macro_data: MacroeconomicData,
                model_types: Optional[List[ModelType]] = None) -> Union[UnifiedResult, EnsembleResult]:
        """
        Generate unified prediction from enabled models
        
        Args:
            index_data: Current index data
            macro_data: Macroeconomic data
            model_types: Specific models to use (optional)
            
        Returns:
            UnifiedResult or EnsembleResult depending on ensemble_mode
        """
        individual_results = []
        models_to_use = model_types or self.enabled_models
        
        # Get predictions from all enabled models
        for model_type in models_to_use:
            if model_type not in self.models:
                continue
                
            try:
                result = self._get_model_prediction(model_type, index_data, macro_data, 
                                                  index_data, macro_data)
                if result:
                    individual_results.append(result)
            except Exception as e:
                print(f"Warning: {model_type.value} model prediction failed: {e}")
                continue
        
        if not individual_results:
            raise ValueError("No models produced valid predictions")
        
        # Return single result or ensemble
        if not self.ensemble_mode or len(individual_results) == 1:
            best_result = max(individual_results, key=lambda x: x.confidence)
            self.prediction_history.append(best_result)
            self._update_performance_metrics(best_result)
            return best_result
        else:
            ensemble_result = self._create_ensemble_prediction(individual_results)
            self.prediction_history.append(ensemble_result)
            self._update_performance_metrics(ensemble_result)
            return ensemble_result
    
    def _get_model_prediction(self, 
                            model_type: ModelType, 
                            index_data: IndexData,
                            macro_data: Optional[MacroeconomicData],
                            ts_index_data: Optional[IndexData],
                            ts_macro_data: Optional[MacroeconomicData]) -> Optional[UnifiedResult]:
        """Get prediction from specific model"""
        model = self.models[model_type]
        timestamp = datetime.now()
        
        try:
            if model_type == ModelType.APT:
                result = model.calculate(index_data)
                return UnifiedResult(
                    model_type=model_type,
                    fair_value=result.fair_value,
                    signal=result.signal,
                    confidence=result.confidence,
                    risk_level=result.risk_level,
                    timestamp=timestamp,
                    additional_metrics={
                        "factor_loadings": result.factor_loadings,
                        "expected_return": result.expected_return
                    }
                )
            
            elif model_type == ModelType.TERM_STRUCTURE:
                if not macro_data:
                    return None
                result = model.calculate(macro_data)
                return UnifiedResult(
                    model_type=model_type,
                    fair_value=result.fair_value,
                    signal=result.signal,
                    confidence=result.confidence,
                    risk_level=result.risk_level,
                    timestamp=timestamp,
                    additional_metrics={
                        "yield_curve_shape": result.yield_curve_shape,
                        "term_premium": result.term_premium
                    }
                )
            
            elif model_type == ModelType.MACROECONOMIC:
                if not macro_data:
                    return None
                result = model.calculate(index_data, macro_data)
                return UnifiedResult(
                    model_type=model_type,
                    fair_value=result.fair_value,
                    signal=result.signal,
                    confidence=result.confidence,
                    risk_level=result.risk_level,
                    timestamp=timestamp,
                    additional_metrics={
                        "economic_regime": result.current_regime,
                        "factor_contributions": result.factor_contributions
                    }
                )
            
            elif model_type == ModelType.ML_LSTM:
                if not (macro_data and index_data):
                    return None
                result = model.lstm_prediction(index_data, macro_data)
                return self._convert_ml_result(result, model_type, timestamp)
            
            elif model_type == ModelType.ML_TRANSFORMER:
                if not (macro_data and index_data):
                    return None
                result = model.transformer_prediction(index_data, macro_data)
                return self._convert_ml_result(result, model_type, timestamp)
            
            elif model_type == ModelType.ML_ENSEMBLE:
                if not (macro_data and index_data):
                    return None
                result = model.ensemble_prediction(index_data, macro_data)
                return self._convert_ml_result(result, model_type, timestamp)
                
        except Exception as e:
            print(f"Error in {model_type.value} prediction: {e}")
            return None
        
        return None
    
    def _convert_ml_result(self, ml_result: MLResult, model_type: ModelType, timestamp: datetime) -> UnifiedResult:
        """Convert ML result to unified result"""
        return UnifiedResult(
            model_type=model_type,
            fair_value=ml_result.predicted_value,
            signal=ml_result.signal,
            confidence=ml_result.confidence,
            risk_level=ml_result.risk_level,
            timestamp=timestamp,
            additional_metrics={
                "feature_importance": getattr(ml_result, 'feature_importance', {})
            }
        )
    
    def _create_ensemble_prediction(self, individual_results: List[UnifiedResult]) -> EnsembleResult:
        """Create ensemble prediction from individual results"""
        # Calculate weighted consensus
        weighted_fair_values = []
        weighted_confidences = []
        signal_votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
        risk_votes = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        
        for result in individual_results:
            weight = self.model_weights.get(result.model_type, 0.0)
            
            weighted_fair_values.append(result.fair_value * weight)
            weighted_confidences.append(result.confidence * weight)
            
            signal_votes[result.signal] += weight
            risk_votes[result.risk_level] += weight
        
        # Consensus values
        consensus_fair_value = sum(weighted_fair_values)
        consensus_confidence = sum(weighted_confidences)
        consensus_signal = max(signal_votes, key=signal_votes.get)
        consensus_risk_level = max(risk_votes, key=risk_votes.get)
        
        # Calculate disagreement score
        fair_value_std = np.std([r.fair_value for r in individual_results])
        fair_value_mean = np.mean([r.fair_value for r in individual_results])
        disagreement_score = fair_value_std / (fair_value_mean + 1e-8)
        
        # Adjust confidence based on disagreement
        consensus_confidence *= (1.0 - min(0.5, disagreement_score))
        
        return EnsembleResult(
            consensus_fair_value=consensus_fair_value,
            consensus_signal=consensus_signal,
            consensus_confidence=consensus_confidence,
            consensus_risk_level=consensus_risk_level,
            individual_results=individual_results,
            model_weights=self.model_weights,
            disagreement_score=disagreement_score,
            timestamp=datetime.now()
        )
    
    def _update_performance_metrics(self, result: Union[UnifiedResult, EnsembleResult]):
        """Update performance tracking metrics"""
        self.performance_metrics["total_predictions"] += 1
        
        if isinstance(result, UnifiedResult):
            # Update individual model performance
            model_perf = self.performance_metrics["model_performance"][result.model_type.value]
            model_perf["count"] += 1
            
            # Update average confidence
            total_preds = self.performance_metrics["total_predictions"]
            current_avg = self.performance_metrics["average_confidence"]
            self.performance_metrics["average_confidence"] = (
                (current_avg * (total_preds - 1) + result.confidence) / total_preds
            )
        
        elif isinstance(result, EnsembleResult):
            # Update ensemble performance
            total_preds = self.performance_metrics["total_predictions"]
            current_avg = self.performance_metrics["average_confidence"]
            self.performance_metrics["average_confidence"] = (
                (current_avg * (total_preds - 1) + result.consensus_confidence) / total_preds
            )
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        return self.performance_metrics.copy()
    
    def get_prediction_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get prediction history"""
        history = self.prediction_history[-limit:] if limit else self.prediction_history
        return [result.to_dict() for result in history]
    
    def update_model_weights(self, new_weights: Dict[ModelType, float]):
        """Update model weights for ensemble"""
        # Normalize weights
        total_weight = sum(new_weights.values())
        self.model_weights = {k: v / total_weight for k, v in new_weights.items()}
    
    def set_confidence_threshold(self, threshold: float):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def enable_models(self, models: List[ModelType]):
        """Enable specific models"""
        self.enabled_models = models
        self.models = self._initialize_models()
        
        # Update weights for enabled models
        weight = 1.0 / len(self.enabled_models)
        self.model_weights = {model: weight for model in self.enabled_models}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        return {
            "enabled_models": [model.value for model in self.enabled_models],
            "model_weights": {k.value: v for k, v in self.model_weights.items()},
            "confidence_threshold": self.confidence_threshold,
            "ensemble_mode": self.ensemble_mode,
            "initialized_models": [model.value for model in self.models.keys()],
            "total_predictions": self.performance_metrics["total_predictions"]
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize unified interface
    unified_indicator = UnifiedIndexIndicator(
        enabled_models=[ModelType.MACROECONOMIC, ModelType.ML_ENSEMBLE],
        ensemble_mode=True
    )
    
    # Create sample data
    index_data = IndexData(
        symbol="SPX",
        name="S&P 500",
        current_level=4500.0,
        historical_levels=[4400, 4450, 4500, 4520, 4500],
        pe_ratio=22.5,
        pb_ratio=3.2,
        dividend_yield=0.015,
        market_cap=45000000000000,
        volatility=0.18,
        beta=1.0,
        sector_weights={"Technology": 0.3, "Healthcare": 0.15, "Finance": 0.12},
        constituent_count=500,
        volume=5000000000
    )
    
    macro_data = MacroeconomicData(
        gdp_growth=2.5,
        inflation_rate=3.2,
        unemployment_rate=3.8,
        interest_rate=5.25,
        money_supply_growth=8.5,
        government_debt_to_gdp=120.0,
        trade_balance=-50.0,
        consumer_confidence=105.0,
        business_confidence=95.0,
        manufacturing_pmi=52.0,
        services_pmi=54.0,
        retail_sales_growth=4.2,
        industrial_production=2.8,
        housing_starts=1.4,
        oil_price=75.0,
        dollar_index=103.0,
        vix=18.5
    )
    
    try:
        # Get prediction
        result = unified_indicator.predict(index_data, macro_data)
        
        print("=== UNIFIED INDEX INDICATOR RESULT ===")
        if isinstance(result, EnsembleResult):
            print(f"Consensus Fair Value: ${result.consensus_fair_value:.2f}")
            print(f"Consensus Signal: {result.consensus_signal}")
            print(f"Consensus Confidence: {result.consensus_confidence:.3f}")
            print(f"Disagreement Score: {result.disagreement_score:.3f}")
            print(f"Individual Models: {len(result.individual_results)}")
        else:
            print(f"Model: {result.model_type.value}")
            print(f"Fair Value: ${result.fair_value:.2f}")
            print(f"Signal: {result.signal}")
            print(f"Confidence: {result.confidence:.3f}")
        
        # Show model status
        status = unified_indicator.get_model_status()
        print(f"\nModel Status: {status}")
        
    except Exception as e:
        print(f"Error: {e}")