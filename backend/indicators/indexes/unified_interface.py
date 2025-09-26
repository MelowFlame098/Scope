"""Unified Interface for Indexes Models

This module provides a unified interface for all indexes prediction models,
integrating traditional financial models with advanced ML capabilities.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import existing models
try:
    from arbitrage_pricing_theory import ArbitragePricingTheory, APTResult
    APT_AVAILABLE = True
except ImportError:
    APT_AVAILABLE = False
    print("APT model not available")

try:
    from capm_analyzer import CAPMAnalyzer
    CAPM_AVAILABLE = True
except ImportError:
    CAPM_AVAILABLE = False
    print("CAPM analyzer not available")

try:
    from macroeconomic_factor_model import MacroeconomicFactorModel
    MACRO_AVAILABLE = True
except ImportError:
    MACRO_AVAILABLE = False
    print("Macroeconomic factor model not available")

try:
    from ml_models import AdvancedMLModels, MLResult, IndexesData, MacroData, ModelType
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("ML models not available")

class PredictionModel(Enum):
    """Available prediction models"""
    APT = "apt"
    CAPM = "capm"
    MACROECONOMIC = "macroeconomic"
    ML_ENSEMBLE = "ml_ensemble"
    ML_LSTM = "ml_lstm"
    ML_TRANSFORMER = "ml_transformer"
    ML_RANDOM_FOREST = "ml_random_forest"
    ML_GRADIENT_BOOSTING = "ml_gradient_boosting"
    ML_XGBOOST = "ml_xgboost"

@dataclass
class UnifiedResult:
    """Unified result structure for all models"""
    model_name: str
    predicted_value: float
    confidence: float
    signal: str  # 'BUY', 'SELL', 'HOLD'
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH'
    timestamp: datetime
    
    # Model-specific results
    apt_result: Optional[Any] = None
    capm_result: Optional[Any] = None
    macro_result: Optional[Any] = None
    ml_result: Optional[MLResult] = None
    
    # Additional metrics
    prediction_interval: Optional[tuple] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_metrics: Optional[Dict[str, float]] = None

@dataclass
class EnsembleResult:
    """Result from ensemble of multiple models"""
    consensus_prediction: float
    consensus_confidence: float
    consensus_signal: str
    consensus_risk_level: str
    model_predictions: Dict[str, UnifiedResult]
    model_weights: Dict[str, float]
    prediction_variance: float
    timestamp: datetime

class UnifiedIndexesInterface:
    """Unified interface for all indexes prediction models"""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.prediction_history = []
        
        # Initialize available models
        self._initialize_models()
        
        # Set default model weights
        self._set_default_weights()
    
    def _initialize_models(self):
        """Initialize all available models"""
        if APT_AVAILABLE:
            self.models['apt'] = ArbitragePricingTheory()
            print("APT model initialized")
        
        if CAPM_AVAILABLE:
            self.models['capm'] = CAPMAnalyzer()
            print("CAPM analyzer initialized")
        
        if MACRO_AVAILABLE:
            self.models['macroeconomic'] = MacroeconomicFactorModel()
            print("Macroeconomic factor model initialized")
        
        if ML_AVAILABLE:
            self.models['ml_ensemble'] = AdvancedMLModels()
            print("ML models initialized")
        
        print(f"Initialized {len(self.models)} models")
    
    def _set_default_weights(self):
        """Set default weights for ensemble prediction"""
        total_models = len(self.models)
        if total_models == 0:
            return
        
        # Equal weights by default, but can be adjusted based on performance
        base_weight = 1.0 / total_models
        
        for model_name in self.models.keys():
            if model_name == 'ml_ensemble':
                self.model_weights[model_name] = base_weight * 1.2  # Slightly higher weight for ML
            else:
                self.model_weights[model_name] = base_weight
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        for model_name in self.model_weights:
            self.model_weights[model_name] /= total_weight
    
    def predict(self, model: PredictionModel, indexes_data: Dict[str, Any], 
                macro_data: Dict[str, Any]) -> UnifiedResult:
        """Make prediction using specified model"""
        model_name = model.value
        
        if model_name not in self.models and not model_name.startswith('ml_'):
            return self._create_empty_result(model_name, "Model not available")
        
        try:
            if model_name == 'apt':
                return self._predict_apt(indexes_data, macro_data)
            elif model_name == 'capm':
                return self._predict_capm(indexes_data, macro_data)
            elif model_name == 'macroeconomic':
                return self._predict_macroeconomic(indexes_data, macro_data)
            elif model_name.startswith('ml_'):
                return self._predict_ml(model_name, indexes_data, macro_data)
            else:
                return self._create_empty_result(model_name, "Unknown model")
        
        except Exception as e:
            print(f"Error in {model_name} prediction: {e}")
            return self._create_empty_result(model_name, str(e))
    
    def _predict_apt(self, indexes_data: Dict[str, Any], macro_data: Dict[str, Any]) -> UnifiedResult:
        """Make prediction using APT model"""
        # Convert data to APT format
        from arbitrage_pricing_theory import IndexData, MacroeconomicFactors
        
        index_data = IndexData(
            prices=indexes_data.get('historical_levels', []),
            returns=self._calculate_returns(indexes_data.get('historical_levels', [])),
            volume=indexes_data.get('volume', []),
            market_cap=indexes_data.get('market_cap', []),
            timestamps=[datetime.now()],
            index_symbol=indexes_data.get('symbol', 'UNKNOWN'),
            constituent_weights=indexes_data.get('constituent_weights'),
            sector_weights=indexes_data.get('sector_weights')
        )
        
        macro_factors = MacroeconomicFactors(
            gdp_growth=[macro_data.get('gdp_growth', 0)],
            inflation_rate=[macro_data.get('inflation_rate', 0)],
            interest_rates=[macro_data.get('interest_rates', 0)],
            unemployment_rate=[macro_data.get('unemployment_rate', 0)],
            industrial_production=[macro_data.get('industrial_production', 0)],
            consumer_confidence=[macro_data.get('consumer_confidence', 0)],
            oil_prices=[macro_data.get('oil_prices', 0)],
            exchange_rates=[macro_data.get('exchange_rates', 0)],
            vix_index=[macro_data.get('vix_index', 0)],
            timestamps=[datetime.now()]
        )
        
        apt_result = self.models['apt'].analyze_apt(index_data, macro_factors)
        
        # Extract prediction from APT result
        predicted_value = apt_result.expected_returns[0] if apt_result.expected_returns else 0.0
        confidence = apt_result.model_fit.get('r_squared', 0.0) if apt_result.model_fit else 0.0
        
        # Generate signal based on expected return
        signal = self._generate_signal(predicted_value, indexes_data.get('current_level', 0))
        
        # Risk assessment
        systematic_risk = apt_result.systematic_risk[0] if apt_result.systematic_risk else 0.0
        risk_level = "HIGH" if systematic_risk > 0.1 else "MEDIUM" if systematic_risk > 0.05 else "LOW"
        
        return UnifiedResult(
            model_name="APT",
            predicted_value=predicted_value,
            confidence=confidence,
            signal=signal,
            risk_level=risk_level,
            timestamp=datetime.now(),
            apt_result=apt_result,
            model_metrics=apt_result.model_fit
        )
    
    def _predict_capm(self, indexes_data: Dict[str, Any], macro_data: Dict[str, Any]) -> UnifiedResult:
        """Make prediction using CAPM model"""
        # Simplified CAPM prediction
        beta = indexes_data.get('beta', 1.0)
        risk_free_rate = macro_data.get('interest_rates', 0.05) / 100
        market_return = 0.08  # Assumed market return
        
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        current_level = indexes_data.get('current_level', 100)
        predicted_value = current_level * (1 + expected_return)
        
        confidence = 0.7  # CAPM typically has moderate confidence
        signal = self._generate_signal(predicted_value, current_level)
        risk_level = "HIGH" if beta > 1.5 else "MEDIUM" if beta > 1.0 else "LOW"
        
        return UnifiedResult(
            model_name="CAPM",
            predicted_value=predicted_value,
            confidence=confidence,
            signal=signal,
            risk_level=risk_level,
            timestamp=datetime.now(),
            model_metrics={'beta': beta, 'expected_return': expected_return}
        )
    
    def _predict_macroeconomic(self, indexes_data: Dict[str, Any], macro_data: Dict[str, Any]) -> UnifiedResult:
        """Make prediction using macroeconomic factor model"""
        # Simplified macroeconomic prediction
        gdp_impact = macro_data.get('gdp_growth', 0) * 0.5
        inflation_impact = -macro_data.get('inflation_rate', 0) * 0.3
        interest_impact = -macro_data.get('interest_rates', 0) * 0.2
        
        total_impact = gdp_impact + inflation_impact + interest_impact
        current_level = indexes_data.get('current_level', 100)
        predicted_value = current_level * (1 + total_impact / 100)
        
        confidence = 0.6  # Moderate confidence for macro models
        signal = self._generate_signal(predicted_value, current_level)
        
        # Risk based on macro volatility
        vix = macro_data.get('vix_index', 20)
        risk_level = "HIGH" if vix > 30 else "MEDIUM" if vix > 20 else "LOW"
        
        return UnifiedResult(
            model_name="Macroeconomic",
            predicted_value=predicted_value,
            confidence=confidence,
            signal=signal,
            risk_level=risk_level,
            timestamp=datetime.now(),
            model_metrics={
                'gdp_impact': gdp_impact,
                'inflation_impact': inflation_impact,
                'interest_impact': interest_impact
            }
        )
    
    def _predict_ml(self, model_name: str, indexes_data: Dict[str, Any], 
                   macro_data: Dict[str, Any]) -> UnifiedResult:
        """Make prediction using ML models"""
        if not ML_AVAILABLE or 'ml_ensemble' not in self.models:
            return self._create_empty_result(model_name, "ML models not available")
        
        # Convert data to ML format
        indexes_ml_data = IndexesData(
            symbol=indexes_data.get('symbol', 'UNKNOWN'),
            name=indexes_data.get('name', 'Unknown Index'),
            current_level=indexes_data.get('current_level', 100.0),
            historical_levels=indexes_data.get('historical_levels', []),
            dividend_yield=indexes_data.get('dividend_yield', 0.02),
            pe_ratio=indexes_data.get('pe_ratio', 20.0),
            pb_ratio=indexes_data.get('pb_ratio', 3.0),
            market_cap=indexes_data.get('market_cap', 1000000000),
            volatility=indexes_data.get('volatility', 0.2),
            beta=indexes_data.get('beta', 1.0),
            sector_weights=indexes_data.get('sector_weights', {}),
            constituent_count=indexes_data.get('constituent_count', 100),
            volume=indexes_data.get('volume', 1000000)
        )
        
        macro_ml_data = MacroData(
            gdp_growth=macro_data.get('gdp_growth', 2.0),
            inflation_rate=macro_data.get('inflation_rate', 3.0),
            interest_rates=macro_data.get('interest_rates', 5.0),
            unemployment_rate=macro_data.get('unemployment_rate', 4.0),
            industrial_production=macro_data.get('industrial_production', 1.0),
            consumer_confidence=macro_data.get('consumer_confidence', 100.0),
            oil_prices=macro_data.get('oil_prices', 80.0),
            exchange_rates=macro_data.get('exchange_rates', 1.0),
            vix_index=macro_data.get('vix_index', 20.0),
            timestamp=datetime.now()
        )
        
        # Make ML prediction
        if model_name == 'ml_ensemble':
            ml_result = self.models['ml_ensemble'].ensemble_prediction(indexes_ml_data, macro_ml_data)
        else:
            # Extract specific model type
            ml_type_map = {
                'ml_lstm': ModelType.LSTM,
                'ml_transformer': ModelType.TRANSFORMER,
                'ml_random_forest': ModelType.RANDOM_FOREST,
                'ml_gradient_boosting': ModelType.GRADIENT_BOOSTING,
                'ml_xgboost': ModelType.XGBOOST
            }
            
            if model_name in ml_type_map:
                ml_result = self.models['ml_ensemble'].predict(ml_type_map[model_name], indexes_ml_data, macro_ml_data)
            else:
                return self._create_empty_result(model_name, "Unknown ML model type")
        
        return UnifiedResult(
            model_name=model_name.upper().replace('_', ' '),
            predicted_value=ml_result.predicted_value,
            confidence=ml_result.confidence,
            signal=ml_result.signal,
            risk_level=ml_result.risk_assessment,
            timestamp=ml_result.timestamp,
            ml_result=ml_result,
            prediction_interval=ml_result.prediction_interval,
            feature_importance=ml_result.feature_importance,
            model_metrics=ml_result.model_metrics
        )
    
    def ensemble_prediction(self, indexes_data: Dict[str, Any], 
                          macro_data: Dict[str, Any]) -> EnsembleResult:
        """Make ensemble prediction using all available models"""
        model_predictions = {}
        valid_predictions = []
        valid_weights = []
        
        # Get predictions from all models
        for model_name in self.models.keys():
            try:
                if model_name == 'apt':
                    result = self._predict_apt(indexes_data, macro_data)
                elif model_name == 'capm':
                    result = self._predict_capm(indexes_data, macro_data)
                elif model_name == 'macroeconomic':
                    result = self._predict_macroeconomic(indexes_data, macro_data)
                elif model_name == 'ml_ensemble':
                    result = self._predict_ml('ml_ensemble', indexes_data, macro_data)
                else:
                    continue
                
                model_predictions[model_name] = result
                
                # Only include valid predictions in ensemble
                if result.predicted_value != 0.0 and result.confidence > 0.1:
                    valid_predictions.append(result.predicted_value)
                    valid_weights.append(self.model_weights.get(model_name, 0.25) * result.confidence)
                
            except Exception as e:
                print(f"Error in {model_name} prediction: {e}")
                continue
        
        if not valid_predictions:
            return EnsembleResult(
                consensus_prediction=0.0,
                consensus_confidence=0.0,
                consensus_signal="HOLD",
                consensus_risk_level="HIGH",
                model_predictions=model_predictions,
                model_weights=self.model_weights,
                prediction_variance=0.0,
                timestamp=datetime.now()
            )
        
        # Calculate weighted ensemble prediction
        valid_weights = np.array(valid_weights)
        if np.sum(valid_weights) > 0:
            valid_weights = valid_weights / np.sum(valid_weights)
            consensus_prediction = np.average(valid_predictions, weights=valid_weights)
        else:
            consensus_prediction = np.mean(valid_predictions)
        
        # Calculate consensus metrics
        prediction_variance = np.var(valid_predictions)
        consensus_confidence = np.mean([r.confidence for r in model_predictions.values() if r.confidence > 0])
        
        # Generate consensus signal
        current_level = indexes_data.get('current_level', 100)
        consensus_signal = self._generate_signal(consensus_prediction, current_level)
        
        # Consensus risk level
        risk_levels = [r.risk_level for r in model_predictions.values()]
        risk_counts = {'LOW': risk_levels.count('LOW'), 
                      'MEDIUM': risk_levels.count('MEDIUM'), 
                      'HIGH': risk_levels.count('HIGH')}
        consensus_risk_level = max(risk_counts, key=risk_counts.get)
        
        return EnsembleResult(
            consensus_prediction=consensus_prediction,
            consensus_confidence=consensus_confidence,
            consensus_signal=consensus_signal,
            consensus_risk_level=consensus_risk_level,
            model_predictions=model_predictions,
            model_weights=self.model_weights,
            prediction_variance=prediction_variance,
            timestamp=datetime.now()
        )
    
    def _generate_signal(self, predicted_value: float, current_value: float) -> str:
        """Generate trading signal based on prediction"""
        if current_value == 0:
            return "HOLD"
        
        change_pct = (predicted_value - current_value) / current_value
        
        if change_pct > 0.02:  # 2% threshold
            return "BUY"
        elif change_pct < -0.02:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate returns from price series"""
        if len(prices) < 2:
            return [0.0]
        
        returns = []
        for i in range(1, len(prices)):
            if prices[i-1] != 0:
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(ret)
            else:
                returns.append(0.0)
        
        return returns
    
    def _create_empty_result(self, model_name: str, error_msg: str = "") -> UnifiedResult:
        """Create empty result for failed predictions"""
        return UnifiedResult(
            model_name=model_name,
            predicted_value=0.0,
            confidence=0.0,
            signal="HOLD",
            risk_level="HIGH",
            timestamp=datetime.now()
        )
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            'total_models': len(self.models),
            'available_models': list(self.models.keys()),
            'model_weights': self.model_weights,
            'last_prediction': len(self.prediction_history)
        }
        
        return status
    
    def update_model_weights(self, new_weights: Dict[str, float]):
        """Update model weights for ensemble prediction"""
        # Validate weights
        total_weight = sum(new_weights.values())
        if total_weight <= 0:
            print("Invalid weights: total weight must be positive")
            return
        
        # Normalize weights
        for model_name, weight in new_weights.items():
            if model_name in self.model_weights:
                self.model_weights[model_name] = weight / total_weight
        
        print("Model weights updated successfully")

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    indexes_data = {
        'symbol': 'SPY',
        'name': 'SPDR S&P 500 ETF',
        'current_level': 450.0,
        'historical_levels': [440.0, 445.0, 448.0, 450.0],
        'dividend_yield': 0.015,
        'pe_ratio': 22.5,
        'pb_ratio': 3.2,
        'market_cap': 400000000000,
        'volatility': 0.18,
        'beta': 1.0,
        'sector_weights': {'Technology': 0.28, 'Healthcare': 0.13, 'Financials': 0.11},
        'constituent_count': 500,
        'volume': 50000000
    }
    
    macro_data = {
        'gdp_growth': 2.1,
        'inflation_rate': 3.2,
        'interest_rates': 5.25,
        'unemployment_rate': 3.7,
        'industrial_production': 1.8,
        'consumer_confidence': 102.5,
        'oil_prices': 85.0,
        'exchange_rates': 1.08,
        'vix_index': 18.5
    }
    
    # Initialize unified interface
    interface = UnifiedIndexesInterface()
    
    print("\n=== Unified Indexes Interface Test ===")
    print(f"Models initialized: {interface.get_model_status()}")
    
    # Test ensemble prediction
    try:
        ensemble_result = interface.ensemble_prediction(indexes_data, macro_data)
        
        print(f"\n=== Ensemble Prediction ===")
        print(f"Consensus Prediction: {ensemble_result.consensus_prediction:.2f}")
        print(f"Consensus Confidence: {ensemble_result.consensus_confidence:.2f}")
        print(f"Consensus Signal: {ensemble_result.consensus_signal}")
        print(f"Consensus Risk Level: {ensemble_result.consensus_risk_level}")
        print(f"Prediction Variance: {ensemble_result.prediction_variance:.4f}")
        
        print(f"\n=== Individual Model Results ===")
        for model_name, result in ensemble_result.model_predictions.items():
            print(f"{model_name}: {result.predicted_value:.2f} (confidence: {result.confidence:.2f}, signal: {result.signal})")
        
    except Exception as e:
        print(f"Error in ensemble prediction: {e}")
    
    # Test individual model predictions
    print(f"\n=== Individual Model Tests ===")
    
    # Test CAPM
    try:
        capm_result = interface.predict(PredictionModel.CAPM, indexes_data, macro_data)
        print(f"CAPM: {capm_result.predicted_value:.2f} (signal: {capm_result.signal})")
    except Exception as e:
        print(f"CAPM error: {e}")
    
    # Test ML ensemble
    try:
        ml_result = interface.predict(PredictionModel.ML_ENSEMBLE, indexes_data, macro_data)
        print(f"ML Ensemble: {ml_result.predicted_value:.2f} (signal: {ml_result.signal})")
    except Exception as e:
        print(f"ML Ensemble error: {e}")
    
    print("\n=== Model Status ===")
    status = interface.get_model_status()
    for key, value in status.items():
        print(f"{key}: {value}")