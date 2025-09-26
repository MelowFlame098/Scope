"""Unified Interface for Stock Analysis Models

This module provides a unified interface that integrates:
- Traditional financial models (DCF, DDM, CAPM, Fama-French)
- Advanced ML models (LSTM, Transformer, Ensemble)
- Technical analysis models (ARIMA, GARCH, VAR)
- Comprehensive evaluation and prediction capabilities

Author: Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import models with fallbacks
try:
    from advanced_ml_models import AdvancedMLModels, StockData, MarketData, EnsembleResult
    ML_MODELS_AVAILABLE = True
except ImportError:
    ML_MODELS_AVAILABLE = False
    # Fallback definitions
    @dataclass
    class StockData:
        symbol: str = ""
        current_price: float = 0.0
        historical_prices: List[float] = None
    
    @dataclass
    class MarketData:
        risk_free_rate: float = 0.02
        market_return: float = 0.10
    
    @dataclass
    class EnsembleResult:
        consensus_prediction: float = 0.0
        consensus_confidence: float = 0.0

try:
    from dcf_model import DCFModel
    DCF_AVAILABLE = True
except ImportError:
    DCF_AVAILABLE = False

try:
    from ddm_model import DDMModel
    DDM_AVAILABLE = True
except ImportError:
    DDM_AVAILABLE = False

try:
    from capm_model import CAPMModel
    CAPM_AVAILABLE = True
except ImportError:
    CAPM_AVAILABLE = False

try:
    from fama_french_model import FamaFrenchModel
    FAMA_FRENCH_AVAILABLE = True
except ImportError:
    FAMA_FRENCH_AVAILABLE = False

try:
    from arima_garch_var import ARIMAGARCHVARModel
    ARIMA_GARCH_VAR_AVAILABLE = True
except ImportError:
    ARIMA_GARCH_VAR_AVAILABLE = False

try:
    from financial_ratios import FinancialRatiosAnalyzer
    FINANCIAL_RATIOS_AVAILABLE = True
except ImportError:
    FINANCIAL_RATIOS_AVAILABLE = False

class ModelCategory(Enum):
    """Categories of stock analysis models"""
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    MACHINE_LEARNING = "machine_learning"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

@dataclass
class UnifiedPrediction:
    """Unified prediction result from all models"""
    consensus_prediction: float
    consensus_confidence: float
    consensus_signal: SignalType
    consensus_risk_level: RiskLevel
    individual_predictions: Dict[str, Dict[str, Any]]
    model_weights: Dict[str, float]
    prediction_variance: float
    fundamental_score: float
    technical_score: float
    ml_score: float
    overall_score: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class ModelStatus:
    """Status of individual models"""
    total_models: int
    available_models: List[str]
    unavailable_models: List[str]
    model_categories: Dict[str, List[str]]
    last_update: datetime

class StockUnifiedInterface:
    """Unified interface for all stock analysis models"""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.model_categories = {
            ModelCategory.FUNDAMENTAL: [],
            ModelCategory.TECHNICAL: [],
            ModelCategory.MACHINE_LEARNING: [],
            ModelCategory.STATISTICAL: [],
            ModelCategory.ENSEMBLE: []
        }
        
        # Initialize available models
        self._initialize_models()
        self._set_default_weights()
        
        self.last_prediction = None
        self.prediction_history = []
    
    def _initialize_models(self):
        """Initialize all available models"""
        # Fundamental models
        if DCF_AVAILABLE:
            self.models['dcf'] = DCFModel()
            self.model_categories[ModelCategory.FUNDAMENTAL].append('dcf')
        
        if DDM_AVAILABLE:
            self.models['ddm'] = DDMModel()
            self.model_categories[ModelCategory.FUNDAMENTAL].append('ddm')
        
        if CAPM_AVAILABLE:
            self.models['capm'] = CAPMModel()
            self.model_categories[ModelCategory.FUNDAMENTAL].append('capm')
        
        if FAMA_FRENCH_AVAILABLE:
            self.models['fama_french'] = FamaFrenchModel()
            self.model_categories[ModelCategory.FUNDAMENTAL].append('fama_french')
        
        if FINANCIAL_RATIOS_AVAILABLE:
            self.models['financial_ratios'] = FinancialRatiosAnalyzer()
            self.model_categories[ModelCategory.FUNDAMENTAL].append('financial_ratios')
        
        # Technical/Statistical models
        if ARIMA_GARCH_VAR_AVAILABLE:
            self.models['arima_garch_var'] = ARIMAGARCHVARModel()
            self.model_categories[ModelCategory.STATISTICAL].append('arima_garch_var')
        
        # Machine Learning models
        if ML_MODELS_AVAILABLE:
            self.models['ml_ensemble'] = AdvancedMLModels()
            self.model_categories[ModelCategory.MACHINE_LEARNING].append('ml_ensemble')
    
    def _set_default_weights(self):
        """Set default weights for model ensemble"""
        total_models = len(self.models)
        if total_models == 0:
            return
        
        # Default equal weighting with slight preference for ML models
        base_weight = 1.0 / total_models
        
        for model_name in self.models.keys():
            if model_name == 'ml_ensemble':
                self.model_weights[model_name] = base_weight * 1.5  # Higher weight for ML
            elif model_name in ['dcf', 'ddm', 'capm']:
                self.model_weights[model_name] = base_weight * 1.2  # Higher weight for fundamental
            else:
                self.model_weights[model_name] = base_weight
        
        # Normalize weights
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
    
    def _create_stock_data(self, symbol: str, current_price: float, 
                          historical_data: Dict[str, Any]) -> StockData:
        """Create StockData object from input parameters"""
        return StockData(
            symbol=symbol,
            name=historical_data.get('name', symbol),
            current_price=current_price,
            historical_prices=historical_data.get('historical_prices', [current_price] * 100),
            volume=historical_data.get('volume', [1000000] * 100),
            market_cap=historical_data.get('market_cap', current_price * 1e9),
            pe_ratio=historical_data.get('pe_ratio', 20.0),
            pb_ratio=historical_data.get('pb_ratio', 3.0),
            dividend_yield=historical_data.get('dividend_yield', 0.02),
            beta=historical_data.get('beta', 1.0),
            sector=historical_data.get('sector', 'Unknown'),
            industry=historical_data.get('industry', 'Unknown'),
            revenue=historical_data.get('revenue', current_price * 1e8),
            net_income=historical_data.get('net_income', current_price * 1e7),
            free_cash_flow=historical_data.get('free_cash_flow', current_price * 8e6),
            debt_to_equity=historical_data.get('debt_to_equity', 1.0),
            roe=historical_data.get('roe', 0.15),
            roa=historical_data.get('roa', 0.08),
            profit_margin=historical_data.get('profit_margin', 0.10),
            revenue_growth=historical_data.get('revenue_growth', 0.05),
            earnings_growth=historical_data.get('earnings_growth', 0.08)
        )
    
    def _create_market_data(self, market_conditions: Dict[str, Any]) -> MarketData:
        """Create MarketData object from input parameters"""
        return MarketData(
            risk_free_rate=market_conditions.get('risk_free_rate', 0.02),
            market_return=market_conditions.get('market_return', 0.10),
            inflation_rate=market_conditions.get('inflation_rate', 0.03),
            gdp_growth=market_conditions.get('gdp_growth', 0.025),
            unemployment_rate=market_conditions.get('unemployment_rate', 0.04),
            vix=market_conditions.get('vix', 20.0),
            dollar_index=market_conditions.get('dollar_index', 100.0),
            oil_price=market_conditions.get('oil_price', 80.0),
            sector_performance=market_conditions.get('sector_performance', {}),
            market_volatility=market_conditions.get('market_volatility', 0.15)
        )
    
    def _predict_fundamental_models(self, stock_data: StockData, market_data: MarketData) -> Dict[str, Dict[str, Any]]:
        """Get predictions from fundamental models"""
        predictions = {}
        
        # DCF Model
        if 'dcf' in self.models:
            try:
                # Simplified DCF prediction
                growth_rate = stock_data.revenue_growth
                discount_rate = market_data.risk_free_rate + stock_data.beta * (market_data.market_return - market_data.risk_free_rate)
                
                # Simple DCF calculation
                fcf = stock_data.free_cash_flow
                terminal_value = fcf * (1 + growth_rate) / (discount_rate - 0.02)  # 2% terminal growth
                fair_value = terminal_value / 1e9  # Simplified per share
                
                price_change = (fair_value - stock_data.current_price) / stock_data.current_price
                
                predictions['dcf'] = {
                    'prediction': fair_value,
                    'confidence': 0.7,
                    'signal': 'BUY' if price_change > 0.05 else 'SELL' if price_change < -0.05 else 'HOLD',
                    'risk_level': 'LOW' if abs(price_change) < 0.1 else 'MEDIUM',
                    'metadata': {'fair_value': fair_value, 'discount_rate': discount_rate}
                }
            except Exception as e:
                predictions['dcf'] = {'prediction': 0.0, 'confidence': 0.0, 'signal': 'HOLD', 'error': str(e)}
        
        # DDM Model
        if 'ddm' in self.models:
            try:
                # Simplified DDM calculation
                dividend = stock_data.current_price * stock_data.dividend_yield
                growth_rate = stock_data.earnings_growth * 0.5  # Conservative dividend growth
                required_return = market_data.risk_free_rate + stock_data.beta * (market_data.market_return - market_data.risk_free_rate)
                
                if required_return > growth_rate:
                    fair_value = dividend * (1 + growth_rate) / (required_return - growth_rate)
                    price_change = (fair_value - stock_data.current_price) / stock_data.current_price
                    
                    predictions['ddm'] = {
                        'prediction': fair_value,
                        'confidence': 0.6,
                        'signal': 'BUY' if price_change > 0.05 else 'SELL' if price_change < -0.05 else 'HOLD',
                        'risk_level': 'LOW' if stock_data.dividend_yield > 0.02 else 'MEDIUM',
                        'metadata': {'fair_value': fair_value, 'required_return': required_return}
                    }
                else:
                    predictions['ddm'] = {'prediction': stock_data.current_price, 'confidence': 0.3, 'signal': 'HOLD', 'risk_level': 'HIGH'}
            except Exception as e:
                predictions['ddm'] = {'prediction': 0.0, 'confidence': 0.0, 'signal': 'HOLD', 'error': str(e)}
        
        # CAPM Model
        if 'capm' in self.models:
            try:
                # CAPM expected return
                expected_return = market_data.risk_free_rate + stock_data.beta * (market_data.market_return - market_data.risk_free_rate)
                expected_price = stock_data.current_price * (1 + expected_return)
                
                predictions['capm'] = {
                    'prediction': expected_price,
                    'confidence': 0.65,
                    'signal': 'BUY' if expected_return > 0.08 else 'SELL' if expected_return < 0.02 else 'HOLD',
                    'risk_level': 'LOW' if stock_data.beta < 1.2 else 'MEDIUM' if stock_data.beta < 1.8 else 'HIGH',
                    'metadata': {'expected_return': expected_return, 'beta': stock_data.beta}
                }
            except Exception as e:
                predictions['capm'] = {'prediction': 0.0, 'confidence': 0.0, 'signal': 'HOLD', 'error': str(e)}
        
        return predictions
    
    def _predict_technical_models(self, stock_data: StockData) -> Dict[str, Dict[str, Any]]:
        """Get predictions from technical/statistical models"""
        predictions = {}
        
        # ARIMA-GARCH-VAR Model
        if 'arima_garch_var' in self.models:
            try:
                # Simplified technical analysis
                prices = stock_data.historical_prices[-30:] if len(stock_data.historical_prices) >= 30 else stock_data.historical_prices
                
                # Moving averages
                short_ma = np.mean(prices[-5:]) if len(prices) >= 5 else stock_data.current_price
                long_ma = np.mean(prices[-20:]) if len(prices) >= 20 else stock_data.current_price
                
                # Trend analysis
                trend = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
                volatility = np.std(prices) / np.mean(prices) if len(prices) > 1 and np.mean(prices) != 0 else 0.1
                
                # Prediction based on trend
                prediction = stock_data.current_price * (1 + trend * 0.5)
                confidence = max(0.1, 0.8 - volatility)
                
                predictions['arima_garch_var'] = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'signal': 'BUY' if trend > 0.02 else 'SELL' if trend < -0.02 else 'HOLD',
                    'risk_level': 'LOW' if volatility < 0.1 else 'MEDIUM' if volatility < 0.2 else 'HIGH',
                    'metadata': {'trend': trend, 'volatility': volatility, 'short_ma': short_ma, 'long_ma': long_ma}
                }
            except Exception as e:
                predictions['arima_garch_var'] = {'prediction': 0.0, 'confidence': 0.0, 'signal': 'HOLD', 'error': str(e)}
        
        return predictions
    
    def _predict_ml_models(self, stock_data: StockData, market_data: MarketData) -> Dict[str, Dict[str, Any]]:
        """Get predictions from ML models"""
        predictions = {}
        
        if 'ml_ensemble' in self.models:
            try:
                ensemble_result = self.models['ml_ensemble'].ensemble_prediction(stock_data, market_data)
                
                predictions['ml_ensemble'] = {
                    'prediction': ensemble_result.consensus_prediction,
                    'confidence': ensemble_result.consensus_confidence,
                    'signal': ensemble_result.consensus_signal,
                    'risk_level': ensemble_result.consensus_risk_level,
                    'metadata': {
                        'individual_predictions': {k: v.prediction for k, v in ensemble_result.individual_predictions.items()},
                        'model_weights': ensemble_result.model_weights,
                        'prediction_variance': ensemble_result.prediction_variance
                    }
                }
            except Exception as e:
                predictions['ml_ensemble'] = {'prediction': 0.0, 'confidence': 0.0, 'signal': 'HOLD', 'error': str(e)}
        
        return predictions
    
    def ensemble_prediction(self, symbol: str, current_price: float, 
                          historical_data: Dict[str, Any], 
                          market_conditions: Dict[str, Any]) -> UnifiedPrediction:
        """Make ensemble prediction using all available models"""
        try:
            # Create data objects
            stock_data = self._create_stock_data(symbol, current_price, historical_data)
            market_data = self._create_market_data(market_conditions)
            
            # Get predictions from all model categories
            fundamental_predictions = self._predict_fundamental_models(stock_data, market_data)
            technical_predictions = self._predict_technical_models(stock_data)
            ml_predictions = self._predict_ml_models(stock_data, market_data)
            
            # Combine all predictions
            all_predictions = {**fundamental_predictions, **technical_predictions, **ml_predictions}
            
            if not all_predictions:
                return self._create_fallback_prediction(symbol, current_price)
            
            # Calculate weighted ensemble prediction
            valid_predictions = {k: v for k, v in all_predictions.items() if v.get('prediction', 0) > 0 and 'error' not in v}
            
            if not valid_predictions:
                return self._create_fallback_prediction(symbol, current_price)
            
            # Weight by confidence and model weights
            weighted_predictions = []
            total_weight = 0
            
            for model_name, pred in valid_predictions.items():
                model_weight = self.model_weights.get(model_name, 1.0 / len(valid_predictions))
                confidence_weight = pred.get('confidence', 0.5)
                combined_weight = model_weight * confidence_weight
                
                weighted_predictions.append(pred['prediction'] * combined_weight)
                total_weight += combined_weight
            
            # Consensus prediction
            consensus_prediction = sum(weighted_predictions) / total_weight if total_weight > 0 else current_price
            
            # Consensus confidence
            confidences = [pred.get('confidence', 0.5) for pred in valid_predictions.values()]
            consensus_confidence = np.mean(confidences)
            
            # Consensus signal
            signals = [pred.get('signal', 'HOLD') for pred in valid_predictions.values()]
            signal_counts = {'BUY': signals.count('BUY'), 'SELL': signals.count('SELL'), 'HOLD': signals.count('HOLD')}
            consensus_signal = SignalType(max(signal_counts, key=signal_counts.get))
            
            # Consensus risk level
            risk_levels = [pred.get('risk_level', 'MEDIUM') for pred in valid_predictions.values()]
            risk_counts = {'LOW': risk_levels.count('LOW'), 'MEDIUM': risk_levels.count('MEDIUM'), 'HIGH': risk_levels.count('HIGH')}
            consensus_risk_level = RiskLevel(max(risk_counts, key=risk_counts.get))
            
            # Calculate category scores
            fundamental_score = np.mean([pred.get('confidence', 0) for name, pred in fundamental_predictions.items() if 'error' not in pred]) if fundamental_predictions else 0.0
            technical_score = np.mean([pred.get('confidence', 0) for name, pred in technical_predictions.items() if 'error' not in pred]) if technical_predictions else 0.0
            ml_score = np.mean([pred.get('confidence', 0) for name, pred in ml_predictions.items() if 'error' not in pred]) if ml_predictions else 0.0
            
            overall_score = (fundamental_score + technical_score + ml_score) / 3
            
            # Prediction variance
            predictions_list = [pred['prediction'] for pred in valid_predictions.values()]
            prediction_variance = np.var(predictions_list) if len(predictions_list) > 1 else 0.0
            
            # Create unified prediction result
            unified_prediction = UnifiedPrediction(
                consensus_prediction=consensus_prediction,
                consensus_confidence=consensus_confidence,
                consensus_signal=consensus_signal,
                consensus_risk_level=consensus_risk_level,
                individual_predictions=all_predictions,
                model_weights=self.model_weights,
                prediction_variance=prediction_variance,
                fundamental_score=fundamental_score,
                technical_score=technical_score,
                ml_score=ml_score,
                overall_score=overall_score,
                timestamp=datetime.now(),
                metadata={
                    'symbol': symbol,
                    'current_price': current_price,
                    'valid_models': len(valid_predictions),
                    'total_models': len(all_predictions),
                    'signal_distribution': signal_counts,
                    'risk_distribution': risk_counts
                }
            )
            
            # Store prediction
            self.last_prediction = unified_prediction
            self.prediction_history.append(unified_prediction)
            
            # Keep only last 100 predictions
            if len(self.prediction_history) > 100:
                self.prediction_history = self.prediction_history[-100:]
            
            return unified_prediction
        
        except Exception as e:
            return self._create_fallback_prediction(symbol, current_price, error=str(e))
    
    def _create_fallback_prediction(self, symbol: str, current_price: float, error: str = None) -> UnifiedPrediction:
        """Create fallback prediction when models fail"""
        return UnifiedPrediction(
            consensus_prediction=current_price,
            consensus_confidence=0.1,
            consensus_signal=SignalType.HOLD,
            consensus_risk_level=RiskLevel.HIGH,
            individual_predictions={},
            model_weights={},
            prediction_variance=0.0,
            fundamental_score=0.0,
            technical_score=0.0,
            ml_score=0.0,
            overall_score=0.0,
            timestamp=datetime.now(),
            metadata={'symbol': symbol, 'current_price': current_price, 'error': error}
        )
    
    def get_model_status(self) -> ModelStatus:
        """Get status of all models"""
        available_models = list(self.models.keys())
        unavailable_models = []
        
        # Check for unavailable models
        if not DCF_AVAILABLE:
            unavailable_models.append('dcf')
        if not DDM_AVAILABLE:
            unavailable_models.append('ddm')
        if not CAPM_AVAILABLE:
            unavailable_models.append('capm')
        if not FAMA_FRENCH_AVAILABLE:
            unavailable_models.append('fama_french')
        if not ARIMA_GARCH_VAR_AVAILABLE:
            unavailable_models.append('arima_garch_var')
        if not ML_MODELS_AVAILABLE:
            unavailable_models.append('ml_ensemble')
        if not FINANCIAL_RATIOS_AVAILABLE:
            unavailable_models.append('financial_ratios')
        
        return ModelStatus(
            total_models=len(available_models),
            available_models=available_models,
            unavailable_models=unavailable_models,
            model_categories={k.value: v for k, v in self.model_categories.items()},
            last_update=datetime.now()
        )
    
    def update_model_weights(self, new_weights: Dict[str, float]):
        """Update model weights for ensemble"""
        # Validate weights
        valid_weights = {k: v for k, v in new_weights.items() if k in self.models and v >= 0}
        
        if valid_weights:
            # Normalize weights
            total_weight = sum(valid_weights.values())
            if total_weight > 0:
                self.model_weights.update({k: v/total_weight for k, v in valid_weights.items()})
    
    def get_prediction_history(self, limit: int = 10) -> List[UnifiedPrediction]:
        """Get recent prediction history"""
        return self.prediction_history[-limit:] if self.prediction_history else []

# Example usage
if __name__ == "__main__":
    # Initialize unified interface
    stock_interface = StockUnifiedInterface()
    
    # Sample data
    symbol = "AAPL"
    current_price = 150.0
    
    historical_data = {
        'name': 'Apple Inc.',
        'historical_prices': [140 + i + np.random.normal(0, 2) for i in range(100)],
        'volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(100)],
        'market_cap': 2500000000000,
        'pe_ratio': 25.0,
        'pb_ratio': 8.0,
        'dividend_yield': 0.005,
        'beta': 1.2,
        'sector': 'Technology',
        'revenue': 365000000000,
        'net_income': 95000000000,
        'free_cash_flow': 80000000000,
        'debt_to_equity': 1.5,
        'roe': 0.25,
        'roa': 0.15,
        'profit_margin': 0.26,
        'revenue_growth': 0.08,
        'earnings_growth': 0.12
    }
    
    market_conditions = {
        'risk_free_rate': 0.02,
        'market_return': 0.10,
        'inflation_rate': 0.03,
        'gdp_growth': 0.025,
        'unemployment_rate': 0.04,
        'vix': 20.0,
        'market_volatility': 0.15
    }
    
    # Make prediction
    prediction = stock_interface.ensemble_prediction(symbol, current_price, historical_data, market_conditions)
    
    print("=== Stock Unified Interface Prediction ===")
    print(f"Symbol: {symbol}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Consensus Prediction: ${prediction.consensus_prediction:.2f}")
    print(f"Consensus Confidence: {prediction.consensus_confidence:.2f}")
    print(f"Consensus Signal: {prediction.consensus_signal.value}")
    print(f"Consensus Risk Level: {prediction.consensus_risk_level.value}")
    print(f"Prediction Variance: {prediction.prediction_variance:.4f}")
    
    print("\n=== Category Scores ===")
    print(f"Fundamental Score: {prediction.fundamental_score:.2f}")
    print(f"Technical Score: {prediction.technical_score:.2f}")
    print(f"ML Score: {prediction.ml_score:.2f}")
    print(f"Overall Score: {prediction.overall_score:.2f}")
    
    print("\n=== Individual Model Results ===")
    for model_name, result in prediction.individual_predictions.items():
        if 'error' not in result:
            print(f"{model_name}: ${result['prediction']:.2f} (confidence: {result['confidence']:.2f}, signal: {result['signal']})")
        else:
            print(f"{model_name}: Error - {result['error']}")
    
    print("\n=== Model Status ===")
    status = stock_interface.get_model_status()
    print(f"Total Models: {status.total_models}")
    print(f"Available Models: {status.available_models}")
    print(f"Model Weights: {prediction.model_weights}")