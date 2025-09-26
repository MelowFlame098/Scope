from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from advanced_ml_models import CryptoAdvancedMLModels, CryptoData, CryptoEnsembleResult
    from crypto_comprehensive import CryptoComprehensiveIndicators, IndicatorResult, AssetType
    from puell_multiple import PuellMultipleModel
    from quant_grade_mvrv import QuantGradeMVRVModel
    from exchange_flow import ExchangeFlowModel
except ImportError as e:
    print(f"Warning: Some crypto models not available: {e}")
    # Create fallback classes
    class CryptoAdvancedMLModels:
        def analyze(self, data): return None
    class CryptoComprehensiveIndicators:
        def stock_to_flow_model(self, data, asset): return None
        def metcalfe_law(self, data): return None
        def crypto_logarithmic_regression(self, data): return None

logger = logging.getLogger(__name__)

class CryptoPredictionCategory(Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"
    MACHINE_LEARNING = "machine_learning"

@dataclass
class CryptoPrediction:
    """Unified crypto prediction result"""
    category: CryptoPredictionCategory
    model_name: str
    prediction: float
    confidence: float
    signal: str
    metadata: Dict[str, Any]
    timestamp: datetime

@dataclass
class CryptoModelStatus:
    """Status of crypto models"""
    total_models: int
    available_models: List[str]
    fitted_models: List[str]
    model_weights: Dict[str, float]
    last_update: datetime
    data_quality: str

@dataclass
class CryptoUnifiedResult:
    """Comprehensive crypto analysis result"""
    consensus_prediction: float
    market_signal: str
    confidence_score: float
    category_scores: Dict[str, float]
    individual_predictions: Dict[str, CryptoPrediction]
    model_status: CryptoModelStatus
    risk_assessment: Dict[str, Any]
    market_regime: str
    timestamp: datetime

class CryptoUnifiedInterface:
    """Unified interface for all crypto prediction models"""
    
    def __init__(self, asset: str = "BTC", enable_ml: bool = True):
        self.asset = asset.upper()
        self.enable_ml = enable_ml
        
        # Initialize model components
        self.ml_models = CryptoAdvancedMLModels() if enable_ml else None
        self.comprehensive_indicators = CryptoComprehensiveIndicators()
        self.puell_model = PuellMultipleModel(asset=asset)
        self.mvrv_model = QuantGradeMVRVModel(asset=asset)
        self.exchange_flow = ExchangeFlowModel(asset=asset)
        
        # Model weights (can be adjusted based on performance)
        self.model_weights = {
            'ml_ensemble': 0.30,
            'stock_to_flow': 0.20,
            'mvrv': 0.15,
            'puell_multiple': 0.10,
            'metcalfe_law': 0.10,
            'log_regression': 0.10,
            'exchange_flow': 0.05
        }
        
        self.fitted_models = set()
        self.last_predictions = {}
        
    def set_model_weights(self, weights: Dict[str, float]):
        """Update model weights for ensemble"""
        total_weight = sum(weights.values())
        self.model_weights = {k: v/total_weight for k, v in weights.items()}
        
    def predict_fundamental(self, crypto_data: CryptoData) -> Dict[str, CryptoPrediction]:
        """Generate fundamental analysis predictions"""
        predictions = {}
        
        try:
            # Stock-to-Flow Model
            data_df = pd.DataFrame({
                'close': crypto_data.price,
                'volume': crypto_data.volume if crypto_data.volume is not None else crypto_data.price * 0
            })
            
            s2f_result = self.comprehensive_indicators.stock_to_flow_model(data_df, self.asset)
            if s2f_result and hasattr(s2f_result, 'values'):
                s2f_prediction = s2f_result.values['predicted_price'].iloc[-1]
                s2f_confidence = s2f_result.confidence
                
                current_price = crypto_data.price.iloc[-1]
                price_change = (s2f_prediction - current_price) / current_price
                
                signal = "BUY" if price_change > 0.1 else "SELL" if price_change < -0.1 else "HOLD"
                
                predictions['stock_to_flow'] = CryptoPrediction(
                    category=CryptoPredictionCategory.FUNDAMENTAL,
                    model_name="Stock-to-Flow",
                    prediction=s2f_prediction,
                    confidence=s2f_confidence,
                    signal=signal,
                    metadata={
                        'price_deviation': price_change * 100,
                        'model_type': 'scarcity_model',
                        'asset_specific': True
                    },
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.warning(f"Stock-to-Flow prediction failed: {e}")
        
        try:
            # Metcalfe's Law
            if crypto_data.active_addresses is not None:
                metcalfe_result = self.comprehensive_indicators.metcalfe_law(data_df)
                if metcalfe_result and hasattr(metcalfe_result, 'values'):
                    metcalfe_prediction = metcalfe_result.values['metcalfe_price'].iloc[-1]
                    
                    predictions['metcalfe_law'] = CryptoPrediction(
                        category=CryptoPredictionCategory.FUNDAMENTAL,
                        model_name="Metcalfe's Law",
                        prediction=metcalfe_prediction,
                        confidence=metcalfe_result.confidence,
                        signal="BUY" if metcalfe_prediction > crypto_data.price.iloc[-1] else "SELL",
                        metadata={'network_effect': True},
                        timestamp=datetime.now()
                    )
        except Exception as e:
            logger.warning(f"Metcalfe's Law prediction failed: {e}")
        
        return predictions
    
    def predict_technical(self, crypto_data: CryptoData) -> Dict[str, CryptoPrediction]:
        """Generate technical analysis predictions"""
        predictions = {}
        
        try:
            # Logarithmic Regression
            data_df = pd.DataFrame({
                'close': crypto_data.price,
                'volume': crypto_data.volume if crypto_data.volume is not None else crypto_data.price * 0
            })
            
            log_reg_result = self.comprehensive_indicators.crypto_logarithmic_regression(data_df)
            if log_reg_result and hasattr(log_reg_result, 'values'):
                log_reg_prediction = log_reg_result.values['log_regression'].iloc[-1]
                band_position = log_reg_result.values['band_position'].iloc[-1]
                
                # Signal based on band position
                if band_position > 0.5:
                    signal = "SELL"  # Overvalued
                elif band_position < -0.5:
                    signal = "BUY"   # Undervalued
                else:
                    signal = "HOLD"
                
                predictions['log_regression'] = CryptoPrediction(
                    category=CryptoPredictionCategory.TECHNICAL,
                    model_name="Logarithmic Regression",
                    prediction=log_reg_prediction,
                    confidence=log_reg_result.confidence,
                    signal=signal,
                    metadata={
                        'band_position': band_position,
                        'r_squared': log_reg_result.metadata.get('r_squared', 0),
                        'trend_analysis': True
                    },
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.warning(f"Logarithmic regression prediction failed: {e}")
        
        return predictions
    
    def predict_onchain(self, crypto_data: CryptoData) -> Dict[str, CryptoPrediction]:
        """Generate on-chain analysis predictions"""
        predictions = {}
        
        try:
            # MVRV Analysis
            if crypto_data.mvrv_ratio is not None:
                current_mvrv = crypto_data.mvrv_ratio.iloc[-1]
                
                # MVRV signal interpretation
                if current_mvrv > 3.0:
                    signal = "STRONG_SELL"  # Historically overvalued
                elif current_mvrv > 2.0:
                    signal = "SELL"
                elif current_mvrv < 0.8:
                    signal = "STRONG_BUY"   # Historically undervalued
                elif current_mvrv < 1.2:
                    signal = "BUY"
                else:
                    signal = "HOLD"
                
                # Simple MVRV-based price prediction
                fair_value = crypto_data.price.iloc[-1] / current_mvrv
                
                predictions['mvrv'] = CryptoPrediction(
                    category=CryptoPredictionCategory.ONCHAIN,
                    model_name="MVRV Ratio",
                    prediction=fair_value,
                    confidence=0.75,
                    signal=signal,
                    metadata={
                        'current_mvrv': current_mvrv,
                        'fair_value_estimate': fair_value,
                        'valuation_metric': True
                    },
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.warning(f"MVRV prediction failed: {e}")
        
        try:
            # Puell Multiple
            if crypto_data.hash_rate is not None:
                # Simplified Puell Multiple calculation
                current_price = crypto_data.price.iloc[-1]
                
                # Estimate mining revenue (simplified)
                daily_issuance = 900 if self.asset == 'BTC' else 1000  # Approximate
                mining_revenue = current_price * daily_issuance
                avg_mining_revenue = crypto_data.price.rolling(365).mean().iloc[-1] * daily_issuance
                
                puell_multiple = mining_revenue / avg_mining_revenue if avg_mining_revenue > 0 else 1.0
                
                # Puell Multiple signals
                if puell_multiple > 4.0:
                    signal = "SELL"  # High miner selling pressure expected
                elif puell_multiple < 0.5:
                    signal = "BUY"   # Low miner selling pressure
                else:
                    signal = "HOLD"
                
                predictions['puell_multiple'] = CryptoPrediction(
                    category=CryptoPredictionCategory.ONCHAIN,
                    model_name="Puell Multiple",
                    prediction=current_price,  # Current price as baseline
                    confidence=0.65,
                    signal=signal,
                    metadata={
                        'puell_multiple': puell_multiple,
                        'mining_profitability': 'high' if puell_multiple > 2 else 'low',
                        'miner_behavior': True
                    },
                    timestamp=datetime.now()
                )
        except Exception as e:
            logger.warning(f"Puell Multiple prediction failed: {e}")
        
        return predictions
    
    def predict_ml(self, crypto_data: CryptoData) -> Dict[str, CryptoPrediction]:
        """Generate ML-based predictions"""
        predictions = {}
        
        if not self.enable_ml or self.ml_models is None:
            return predictions
        
        try:
            ml_result = self.ml_models.analyze(crypto_data)
            if ml_result:
                predictions['ml_ensemble'] = CryptoPrediction(
                    category=CryptoPredictionCategory.MACHINE_LEARNING,
                    model_name="ML Ensemble",
                    prediction=ml_result.consensus_prediction,
                    confidence=ml_result.confidence_score,
                    signal=ml_result.market_signal,
                    metadata={
                        'individual_models': ml_result.individual_predictions,
                        'model_weights': ml_result.model_weights,
                        'prediction_range': ml_result.prediction_range,
                        'ensemble_method': True
                    },
                    timestamp=ml_result.timestamp
                )
        except Exception as e:
            logger.warning(f"ML prediction failed: {e}")
        
        return predictions
    
    def generate_ensemble_prediction(self, all_predictions: Dict[str, CryptoPrediction]) -> Tuple[float, str, float]:
        """Generate ensemble prediction from all models"""
        if not all_predictions:
            return 0.0, "HOLD", 0.0
        
        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0
        confidence_sum = 0.0
        
        buy_signals = 0
        sell_signals = 0
        total_signals = len(all_predictions)
        
        for model_name, prediction in all_predictions.items():
            weight = self.model_weights.get(model_name, 0.1)
            weighted_sum += prediction.prediction * weight * prediction.confidence
            total_weight += weight * prediction.confidence
            confidence_sum += prediction.confidence
            
            # Count signals
            if prediction.signal in ['BUY', 'STRONG_BUY']:
                buy_signals += 1
            elif prediction.signal in ['SELL', 'STRONG_SELL']:
                sell_signals += 1
        
        # Ensemble prediction
        ensemble_prediction = weighted_sum / total_weight if total_weight > 0 else 0.0
        ensemble_confidence = confidence_sum / len(all_predictions)
        
        # Ensemble signal
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        if buy_ratio > 0.6:
            ensemble_signal = "BUY"
        elif sell_ratio > 0.6:
            ensemble_signal = "SELL"
        else:
            ensemble_signal = "HOLD"
        
        return ensemble_prediction, ensemble_signal, ensemble_confidence
    
    def analyze(self, crypto_data: CryptoData) -> CryptoUnifiedResult:
        """Comprehensive crypto analysis using all available models"""
        all_predictions = {}
        
        # Gather predictions from all categories
        fundamental_preds = self.predict_fundamental(crypto_data)
        technical_preds = self.predict_technical(crypto_data)
        onchain_preds = self.predict_onchain(crypto_data)
        ml_preds = self.predict_ml(crypto_data)
        
        all_predictions.update(fundamental_preds)
        all_predictions.update(technical_preds)
        all_predictions.update(onchain_preds)
        all_predictions.update(ml_preds)
        
        # Generate ensemble prediction
        consensus_pred, market_signal, confidence = self.generate_ensemble_prediction(all_predictions)
        
        # Calculate category scores
        category_scores = {
            'fundamental': np.mean([p.confidence for p in fundamental_preds.values()]) if fundamental_preds else 0.0,
            'technical': np.mean([p.confidence for p in technical_preds.values()]) if technical_preds else 0.0,
            'onchain': np.mean([p.confidence for p in onchain_preds.values()]) if onchain_preds else 0.0,
            'ml': np.mean([p.confidence for p in ml_preds.values()]) if ml_preds else 0.0
        }
        
        # Model status
        available_models = list(all_predictions.keys())
        model_status = CryptoModelStatus(
            total_models=len(self.model_weights),
            available_models=available_models,
            fitted_models=list(self.fitted_models),
            model_weights=self.model_weights,
            last_update=datetime.now(),
            data_quality="good" if len(available_models) > 3 else "limited"
        )
        
        # Risk assessment
        current_price = crypto_data.price.iloc[-1]
        price_volatility = crypto_data.price.pct_change().std() * np.sqrt(365)
        
        risk_assessment = {
            'volatility': price_volatility,
            'prediction_spread': max([p.prediction for p in all_predictions.values()]) - min([p.prediction for p in all_predictions.values()]) if all_predictions else 0,
            'model_agreement': len([p for p in all_predictions.values() if p.signal == market_signal]) / len(all_predictions) if all_predictions else 0,
            'data_completeness': len(available_models) / len(self.model_weights)
        }
        
        # Market regime (simplified)
        if price_volatility > 0.8:
            market_regime = "high_volatility"
        elif price_volatility < 0.3:
            market_regime = "low_volatility"
        else:
            market_regime = "normal"
        
        return CryptoUnifiedResult(
            consensus_prediction=consensus_pred,
            market_signal=market_signal,
            confidence_score=confidence,
            category_scores=category_scores,
            individual_predictions=all_predictions,
            model_status=model_status,
            risk_assessment=risk_assessment,
            market_regime=market_regime,
            timestamp=datetime.now()
        )

# Example usage
if __name__ == "__main__":
    # Create sample crypto data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate Bitcoin data
    price_data = pd.Series(
        np.cumsum(np.random.randn(len(dates)) * 0.03) * 1000 + 45000,
        index=dates
    )
    
    volume_data = pd.Series(
        np.random.exponential(2000000, len(dates)),
        index=dates
    )
    
    # Simulate MVRV ratio
    mvrv_data = pd.Series(
        np.random.normal(1.5, 0.5, len(dates)),
        index=dates
    )
    
    crypto_data = CryptoData(
        price=price_data,
        volume=volume_data,
        mvrv_ratio=mvrv_data
    )
    
    # Initialize unified interface
    crypto_interface = CryptoUnifiedInterface(asset="BTC", enable_ml=True)
    
    # Run comprehensive analysis
    result = crypto_interface.analyze(crypto_data)
    
    print("\n=== Crypto Unified Analysis ===")
    print(f"Consensus Prediction: ${result.consensus_prediction:.2f}")
    print(f"Market Signal: {result.market_signal}")
    print(f"Confidence Score: {result.confidence_score:.2f}")
    print(f"Market Regime: {result.market_regime}")
    
    print("\nCategory Scores:")
    for category, score in result.category_scores.items():
        print(f"  {category}: {score:.2f}")
    
    print("\nIndividual Model Results:")
    for model_name, prediction in result.individual_predictions.items():
        print(f"  {model_name}: ${prediction.prediction:.2f} ({prediction.signal})")
    
    print("\nModel Status:")
    print(f"  Total Models: {result.model_status.total_models}")
    print(f"  Available Models: {result.model_status.available_models}")
    print(f"  Data Quality: {result.model_status.data_quality}")
    
    print("\nRisk Assessment:")
    for risk_factor, value in result.risk_assessment.items():
        print(f"  {risk_factor}: {value:.3f}")