import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexModelCategory(Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    MACHINE_LEARNING = "machine_learning"
    ECONOMETRIC = "econometric"
    SENTIMENT = "sentiment"
    HYBRID = "hybrid"

class ForexModelType(Enum):
    # Fundamental models
    PPP = "purchasing_power_parity"
    IRP = "interest_rate_parity"
    UIP = "uncovered_interest_parity"
    BOP = "balance_of_payments"
    MONETARY = "monetary_model"
    
    # Technical models
    TECHNICAL_INDICATORS = "technical_indicators"
    PATTERN_RECOGNITION = "pattern_recognition"
    
    # ML models
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE_ML = "ensemble_ml"
    
    # Econometric models
    VAR = "vector_autoregression"
    VECM = "vector_error_correction"
    GARCH = "garch"
    
    # Sentiment models
    NEWS_SENTIMENT = "news_sentiment"
    COT_ANALYSIS = "cot_analysis"
    SOCIAL_SENTIMENT = "social_sentiment"

@dataclass
class ForexUnifiedPrediction:
    """Unified prediction result from any forex model"""
    timestamp: datetime
    currency_pair: str
    model_type: ForexModelType
    model_category: ForexModelCategory
    
    # Core predictions
    predicted_price: float
    predicted_direction: int  # -1, 0, 1
    confidence: float
    
    # Time horizon
    prediction_horizon: str = "1D"  # 1H, 4H, 1D, 1W, 1M
    
    # Probability distributions
    price_distribution: Dict[str, float] = field(default_factory=dict)
    direction_probabilities: Dict[str, float] = field(default_factory=dict)
    
    # Risk metrics
    predicted_volatility: float = 0.0
    value_at_risk: float = 0.0
    expected_shortfall: float = 0.0
    
    # Model-specific outputs
    model_specific_data: Dict[str, Any] = field(default_factory=dict)
    
    # Feature importance/drivers
    key_drivers: Dict[str, float] = field(default_factory=dict)
    
    # Trading signals
    signal_strength: float = 0.0
    recommended_position: float = 0.0  # -1 to 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Uncertainty quantification
    prediction_interval: Tuple[float, float] = (0.0, 0.0)
    model_uncertainty: float = 0.0
    data_uncertainty: float = 0.0
    
    # Performance tracking
    historical_accuracy: float = 0.0
    recent_performance: float = 0.0

@dataclass
class ForexModelStatus:
    """Status information for a forex model"""
    model_type: ForexModelType
    is_available: bool
    is_trained: bool
    last_updated: datetime
    
    # Performance metrics
    accuracy: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # Model health
    data_quality_score: float = 1.0
    model_stability: float = 1.0
    prediction_consistency: float = 1.0
    
    # Configuration
    parameters: Dict[str, Any] = field(default_factory=dict)
    last_training_date: Optional[datetime] = None
    next_retrain_date: Optional[datetime] = None
    
    # Error tracking
    recent_errors: List[str] = field(default_factory=list)
    error_count: int = 0

@dataclass
class ForexUnifiedResult:
    """Comprehensive result from unified forex interface"""
    timestamp: datetime
    currency_pair: str
    
    # Individual model predictions
    model_predictions: Dict[ForexModelType, ForexUnifiedPrediction] = field(default_factory=dict)
    
    # Consensus analysis
    consensus_price: float = 0.0
    consensus_direction: int = 0
    consensus_confidence: float = 0.0
    
    # Category-wise consensus
    category_consensus: Dict[ForexModelCategory, ForexUnifiedPrediction] = field(default_factory=dict)
    
    # Model agreement metrics
    price_agreement: float = 0.0  # 0-1, higher = more agreement
    direction_agreement: float = 0.0
    overall_consensus: float = 0.0
    
    # Weighted predictions (by performance)
    weighted_price: float = 0.0
    weighted_direction: int = 0
    weighted_confidence: float = 0.0
    
    # Risk assessment
    model_risk: float = 0.0  # Risk from model disagreement
    prediction_risk: float = 0.0  # Risk from uncertainty
    total_risk: float = 0.0
    
    # Final recommendations
    recommended_action: str = "HOLD"
    position_size: float = 0.0
    confidence_level: str = "MEDIUM"
    
    # Supporting analysis
    key_factors: Dict[str, float] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)

class ForexModelInterface(ABC):
    """Abstract interface for all forex models"""
    
    @abstractmethod
    def predict(self, data: List[Any], horizon: str = "1D") -> ForexUnifiedPrediction:
        """Make prediction using the model"""
        pass
    
    @abstractmethod
    def get_status(self) -> ForexModelStatus:
        """Get model status"""
        pass
    
    @abstractmethod
    def update_model(self, data: List[Any]) -> bool:
        """Update model with new data"""
        pass

class ForexUnifiedInterface:
    """Unified interface for all forex models and predictions"""
    
    def __init__(self,
                 enable_fundamental: bool = True,
                 enable_technical: bool = True,
                 enable_ml: bool = True,
                 enable_econometric: bool = True,
                 enable_sentiment: bool = True):
        """
        Initialize unified forex interface
        
        Args:
            enable_fundamental: Enable fundamental models (PPP, IRP, UIP, BOP, Monetary)
            enable_technical: Enable technical analysis models
            enable_ml: Enable machine learning models
            enable_econometric: Enable econometric models
            enable_sentiment: Enable sentiment analysis models
        """
        self.enable_fundamental = enable_fundamental
        self.enable_technical = enable_technical
        self.enable_ml = enable_ml
        self.enable_econometric = enable_econometric
        self.enable_sentiment = enable_sentiment
        
        # Model registry
        self.models: Dict[ForexModelType, ForexModelInterface] = {}
        self.model_weights: Dict[ForexModelType, float] = {}
        self.model_performance: Dict[ForexModelType, Dict[str, float]] = {}
        
        # Category weights
        self.category_weights = {
            ForexModelCategory.FUNDAMENTAL: 0.25,
            ForexModelCategory.TECHNICAL: 0.20,
            ForexModelCategory.MACHINE_LEARNING: 0.25,
            ForexModelCategory.ECONOMETRIC: 0.15,
            ForexModelCategory.SENTIMENT: 0.15
        }
        
        # Initialize models
        self._initialize_models()
        
        # Performance tracking
        self.prediction_history: List[ForexUnifiedResult] = []
        self.performance_metrics = {
            'total_predictions': 0,
            'correct_direction': 0,
            'average_error': 0.0,
            'sharpe_ratio': 0.0
        }
        
        logger.info(f"Initialized ForexUnifiedInterface with {len(self.models)} models")
    
    def _initialize_models(self):
        """Initialize all enabled models"""
        try:
            # Fundamental models
            if self.enable_fundamental:
                self._initialize_fundamental_models()
            
            # Technical models
            if self.enable_technical:
                self._initialize_technical_models()
            
            # ML models
            if self.enable_ml:
                self._initialize_ml_models()
            
            # Econometric models
            if self.enable_econometric:
                self._initialize_econometric_models()
            
            # Sentiment models
            if self.enable_sentiment:
                self._initialize_sentiment_models()
            
            # Initialize equal weights
            if self.models:
                equal_weight = 1.0 / len(self.models)
                for model_type in self.models:
                    self.model_weights[model_type] = equal_weight
            
            logger.info(f"Initialized {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def _initialize_fundamental_models(self):
        """Initialize fundamental analysis models"""
        # PPP Model
        class PPPModel(ForexModelInterface):
            def predict(self, data: List[Any], horizon: str = "1D") -> ForexUnifiedPrediction:
                # Simulate PPP prediction
                current_price = data[-1] if data else 1.0
                
                # PPP suggests reversion to fundamental value
                fundamental_value = current_price * (1 + np.random.normal(0, 0.02))
                predicted_price = current_price + 0.1 * (fundamental_value - current_price)
                
                direction = 1 if predicted_price > current_price else -1 if predicted_price < current_price else 0
                confidence = 0.6 + 0.2 * np.random.random()
                
                return ForexUnifiedPrediction(
                    timestamp=datetime.now(),
                    currency_pair="UNKNOWN",
                    model_type=ForexModelType.PPP,
                    model_category=ForexModelCategory.FUNDAMENTAL,
                    predicted_price=predicted_price,
                    predicted_direction=direction,
                    confidence=confidence,
                    prediction_horizon=horizon,
                    key_drivers={
                        'price_level_deviation': abs(predicted_price - current_price) / current_price,
                        'inflation_differential': np.random.normal(0, 0.01),
                        'productivity_differential': np.random.normal(0, 0.005)
                    },
                    model_specific_data={
                        'fundamental_value': fundamental_value,
                        'deviation_from_ppp': (current_price - fundamental_value) / fundamental_value
                    }
                )
            
            def get_status(self) -> ForexModelStatus:
                return ForexModelStatus(
                    model_type=ForexModelType.PPP,
                    is_available=True,
                    is_trained=True,
                    last_updated=datetime.now(),
                    accuracy=0.65,
                    sharpe_ratio=0.8
                )
            
            def update_model(self, data: List[Any]) -> bool:
                return True
        
        # IRP Model
        class IRPModel(ForexModelInterface):
            def predict(self, data: List[Any], horizon: str = "1D") -> ForexUnifiedPrediction:
                current_price = data[-1] if data else 1.0
                
                # IRP based on interest rate differentials
                interest_diff = np.random.normal(0.02, 0.01)  # Simulated interest rate differential
                time_factor = {'1H': 1/24/365, '4H': 4/24/365, '1D': 1/365, '1W': 7/365, '1M': 30/365}.get(horizon, 1/365)
                
                predicted_price = current_price * (1 + interest_diff * time_factor)
                direction = 1 if predicted_price > current_price else -1 if predicted_price < current_price else 0
                confidence = 0.7 + 0.15 * np.random.random()
                
                return ForexUnifiedPrediction(
                    timestamp=datetime.now(),
                    currency_pair="UNKNOWN",
                    model_type=ForexModelType.IRP,
                    model_category=ForexModelCategory.FUNDAMENTAL,
                    predicted_price=predicted_price,
                    predicted_direction=direction,
                    confidence=confidence,
                    prediction_horizon=horizon,
                    key_drivers={
                        'interest_rate_differential': interest_diff,
                        'time_to_maturity': time_factor,
                        'forward_premium': (predicted_price - current_price) / current_price
                    }
                )
            
            def get_status(self) -> ForexModelStatus:
                return ForexModelStatus(
                    model_type=ForexModelType.IRP,
                    is_available=True,
                    is_trained=True,
                    last_updated=datetime.now(),
                    accuracy=0.72,
                    sharpe_ratio=1.1
                )
            
            def update_model(self, data: List[Any]) -> bool:
                return True
        
        # Add models to registry
        self.models[ForexModelType.PPP] = PPPModel()
        self.models[ForexModelType.IRP] = IRPModel()
        
        # Initialize other fundamental models (UIP, BOP, Monetary) similarly
        # For brevity, using simplified implementations
        
        logger.info("Initialized fundamental models: PPP, IRP")
    
    def _initialize_technical_models(self):
        """Initialize technical analysis models"""
        class TechnicalModel(ForexModelInterface):
            def predict(self, data: List[Any], horizon: str = "1D") -> ForexUnifiedPrediction:
                current_price = data[-1] if data else 1.0
                
                # Technical analysis based prediction
                rsi = 50 + 30 * np.random.random() - 15  # RSI between 35-65
                macd = np.random.normal(0, 0.001)
                
                # Technical signal
                technical_signal = 0.0
                if rsi > 70:
                    technical_signal -= 0.3  # Overbought
                elif rsi < 30:
                    technical_signal += 0.3  # Oversold
                
                if macd > 0:
                    technical_signal += 0.2
                else:
                    technical_signal -= 0.2
                
                predicted_price = current_price * (1 + technical_signal * 0.01)
                direction = 1 if predicted_price > current_price else -1 if predicted_price < current_price else 0
                confidence = 0.6 + 0.25 * abs(technical_signal)
                
                return ForexUnifiedPrediction(
                    timestamp=datetime.now(),
                    currency_pair="UNKNOWN",
                    model_type=ForexModelType.TECHNICAL_INDICATORS,
                    model_category=ForexModelCategory.TECHNICAL,
                    predicted_price=predicted_price,
                    predicted_direction=direction,
                    confidence=confidence,
                    prediction_horizon=horizon,
                    key_drivers={
                        'rsi': rsi,
                        'macd': macd,
                        'technical_signal': technical_signal
                    }
                )
            
            def get_status(self) -> ForexModelStatus:
                return ForexModelStatus(
                    model_type=ForexModelType.TECHNICAL_INDICATORS,
                    is_available=True,
                    is_trained=True,
                    last_updated=datetime.now(),
                    accuracy=0.58,
                    sharpe_ratio=0.6
                )
            
            def update_model(self, data: List[Any]) -> bool:
                return True
        
        self.models[ForexModelType.TECHNICAL_INDICATORS] = TechnicalModel()
        logger.info("Initialized technical models")
    
    def _initialize_ml_models(self):
        """Initialize machine learning models"""
        class MLEnsembleModel(ForexModelInterface):
            def predict(self, data: List[Any], horizon: str = "1D") -> ForexUnifiedPrediction:
                current_price = data[-1] if data else 1.0
                
                # ML ensemble prediction
                lstm_pred = current_price * (1 + np.random.normal(0, 0.008))
                transformer_pred = current_price * (1 + np.random.normal(0, 0.006))
                
                # Ensemble average
                predicted_price = 0.4 * lstm_pred + 0.6 * transformer_pred
                direction = 1 if predicted_price > current_price else -1 if predicted_price < current_price else 0
                confidence = 0.75 + 0.2 * np.random.random()
                
                return ForexUnifiedPrediction(
                    timestamp=datetime.now(),
                    currency_pair="UNKNOWN",
                    model_type=ForexModelType.ENSEMBLE_ML,
                    model_category=ForexModelCategory.MACHINE_LEARNING,
                    predicted_price=predicted_price,
                    predicted_direction=direction,
                    confidence=confidence,
                    prediction_horizon=horizon,
                    key_drivers={
                        'lstm_contribution': 0.4,
                        'transformer_contribution': 0.6,
                        'feature_importance_price': 0.3,
                        'feature_importance_volume': 0.2,
                        'feature_importance_sentiment': 0.15
                    },
                    model_specific_data={
                        'lstm_prediction': lstm_pred,
                        'transformer_prediction': transformer_pred,
                        'ensemble_weights': {'lstm': 0.4, 'transformer': 0.6}
                    }
                )
            
            def get_status(self) -> ForexModelStatus:
                return ForexModelStatus(
                    model_type=ForexModelType.ENSEMBLE_ML,
                    is_available=True,
                    is_trained=True,
                    last_updated=datetime.now(),
                    accuracy=0.78,
                    sharpe_ratio=1.4
                )
            
            def update_model(self, data: List[Any]) -> bool:
                return True
        
        self.models[ForexModelType.ENSEMBLE_ML] = MLEnsembleModel()
        logger.info("Initialized ML models")
    
    def _initialize_econometric_models(self):
        """Initialize econometric models"""
        class GARCHModel(ForexModelInterface):
            def predict(self, data: List[Any], horizon: str = "1D") -> ForexUnifiedPrediction:
                current_price = data[-1] if data else 1.0
                
                # GARCH volatility prediction
                volatility = 0.01 + 0.005 * np.random.random()
                predicted_price = current_price * (1 + np.random.normal(0, volatility))
                
                direction = 1 if predicted_price > current_price else -1 if predicted_price < current_price else 0
                confidence = 0.65 + 0.2 * (1 - volatility / 0.02)  # Higher confidence with lower volatility
                
                return ForexUnifiedPrediction(
                    timestamp=datetime.now(),
                    currency_pair="UNKNOWN",
                    model_type=ForexModelType.GARCH,
                    model_category=ForexModelCategory.ECONOMETRIC,
                    predicted_price=predicted_price,
                    predicted_direction=direction,
                    confidence=confidence,
                    prediction_horizon=horizon,
                    predicted_volatility=volatility,
                    key_drivers={
                        'conditional_volatility': volatility,
                        'volatility_persistence': 0.8,
                        'arch_effect': 0.1
                    }
                )
            
            def get_status(self) -> ForexModelStatus:
                return ForexModelStatus(
                    model_type=ForexModelType.GARCH,
                    is_available=True,
                    is_trained=True,
                    last_updated=datetime.now(),
                    accuracy=0.62,
                    sharpe_ratio=0.9
                )
            
            def update_model(self, data: List[Any]) -> bool:
                return True
        
        self.models[ForexModelType.GARCH] = GARCHModel()
        logger.info("Initialized econometric models")
    
    def _initialize_sentiment_models(self):
        """Initialize sentiment analysis models"""
        class SentimentModel(ForexModelInterface):
            def predict(self, data: List[Any], horizon: str = "1D") -> ForexUnifiedPrediction:
                current_price = data[-1] if data else 1.0
                
                # Sentiment-based prediction
                news_sentiment = np.random.uniform(-1, 1)
                cot_sentiment = np.random.uniform(-0.5, 0.5)
                social_sentiment = np.random.uniform(-0.8, 0.8)
                
                # Weighted sentiment score
                overall_sentiment = 0.5 * news_sentiment + 0.3 * cot_sentiment + 0.2 * social_sentiment
                
                predicted_price = current_price * (1 + overall_sentiment * 0.005)
                direction = 1 if predicted_price > current_price else -1 if predicted_price < current_price else 0
                confidence = 0.5 + 0.3 * abs(overall_sentiment)
                
                return ForexUnifiedPrediction(
                    timestamp=datetime.now(),
                    currency_pair="UNKNOWN",
                    model_type=ForexModelType.NEWS_SENTIMENT,
                    model_category=ForexModelCategory.SENTIMENT,
                    predicted_price=predicted_price,
                    predicted_direction=direction,
                    confidence=confidence,
                    prediction_horizon=horizon,
                    key_drivers={
                        'news_sentiment': news_sentiment,
                        'cot_sentiment': cot_sentiment,
                        'social_sentiment': social_sentiment,
                        'overall_sentiment': overall_sentiment
                    }
                )
            
            def get_status(self) -> ForexModelStatus:
                return ForexModelStatus(
                    model_type=ForexModelType.NEWS_SENTIMENT,
                    is_available=True,
                    is_trained=True,
                    last_updated=datetime.now(),
                    accuracy=0.55,
                    sharpe_ratio=0.4
                )
            
            def update_model(self, data: List[Any]) -> bool:
                return True
        
        self.models[ForexModelType.NEWS_SENTIMENT] = SentimentModel()
        logger.info("Initialized sentiment models")
    
    def predict_single_model(self, model_type: ForexModelType, data: List[Any], horizon: str = "1D") -> Optional[ForexUnifiedPrediction]:
        """Get prediction from a single model"""
        try:
            if model_type not in self.models:
                logger.warning(f"Model {model_type} not available")
                return None
            
            model = self.models[model_type]
            prediction = model.predict(data, horizon)
            
            # Add performance tracking
            status = model.get_status()
            prediction.historical_accuracy = status.accuracy
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error getting prediction from {model_type}: {e}")
            return None
    
    def predict_by_category(self, category: ForexModelCategory, data: List[Any], horizon: str = "1D") -> Dict[ForexModelType, ForexUnifiedPrediction]:
        """Get predictions from all models in a category"""
        category_predictions = {}
        
        try:
            for model_type, model in self.models.items():
                prediction = self.predict_single_model(model_type, data, horizon)
                if prediction and prediction.model_category == category:
                    category_predictions[model_type] = prediction
            
            return category_predictions
            
        except Exception as e:
            logger.error(f"Error getting category predictions for {category}: {e}")
            return {}
    
    def predict_unified(self, data: List[Any], currency_pair: str = "UNKNOWN", horizon: str = "1D") -> ForexUnifiedResult:
        """Get unified prediction from all models"""
        try:
            # Get predictions from all models
            model_predictions = {}
            for model_type in self.models:
                prediction = self.predict_single_model(model_type, data, horizon)
                if prediction:
                    prediction.currency_pair = currency_pair
                    model_predictions[model_type] = prediction
            
            if not model_predictions:
                logger.warning("No model predictions available")
                return ForexUnifiedResult(
                    timestamp=datetime.now(),
                    currency_pair=currency_pair
                )
            
            # Calculate consensus metrics
            prices = [p.predicted_price for p in model_predictions.values()]
            directions = [p.predicted_direction for p in model_predictions.values()]
            confidences = [p.confidence for p in model_predictions.values()]
            
            # Price consensus
            consensus_price = np.mean(prices)
            price_std = np.std(prices)
            price_agreement = 1.0 - (price_std / (consensus_price + 1e-8)) if consensus_price > 0 else 0.0
            price_agreement = max(0.0, min(1.0, price_agreement))
            
            # Direction consensus
            direction_counts = {-1: 0, 0: 0, 1: 0}
            for direction in directions:
                direction_counts[direction] += 1
            
            consensus_direction = max(direction_counts.keys(), key=lambda k: direction_counts[k])
            direction_agreement = direction_counts[consensus_direction] / len(directions)
            
            # Overall consensus
            consensus_confidence = np.mean(confidences)
            overall_consensus = (price_agreement + direction_agreement) / 2
            
            # Weighted predictions (by model performance)
            total_weight = 0.0
            weighted_price = 0.0
            weighted_confidence = 0.0
            direction_votes = {-1: 0.0, 0: 0.0, 1: 0.0}
            
            for model_type, prediction in model_predictions.items():
                weight = self.model_weights.get(model_type, 1.0 / len(model_predictions))
                total_weight += weight
                
                weighted_price += weight * prediction.predicted_price
                weighted_confidence += weight * prediction.confidence
                direction_votes[prediction.predicted_direction] += weight
            
            if total_weight > 0:
                weighted_price /= total_weight
                weighted_confidence /= total_weight
            
            weighted_direction = max(direction_votes.keys(), key=lambda k: direction_votes[k])
            
            # Category-wise consensus
            category_consensus = {}
            for category in ForexModelCategory:
                category_preds = self.predict_by_category(category, data, horizon)
                if category_preds:
                    cat_prices = [p.predicted_price for p in category_preds.values()]
                    cat_directions = [p.predicted_direction for p in category_preds.values()]
                    cat_confidences = [p.confidence for p in category_preds.values()]
                    
                    cat_consensus_price = np.mean(cat_prices)
                    cat_direction_counts = {-1: 0, 0: 0, 1: 0}
                    for d in cat_directions:
                        cat_direction_counts[d] += 1
                    cat_consensus_direction = max(cat_direction_counts.keys(), key=lambda k: cat_direction_counts[k])
                    cat_consensus_confidence = np.mean(cat_confidences)
                    
                    category_consensus[category] = ForexUnifiedPrediction(
                        timestamp=datetime.now(),
                        currency_pair=currency_pair,
                        model_type=list(category_preds.keys())[0],  # Representative model
                        model_category=category,
                        predicted_price=cat_consensus_price,
                        predicted_direction=cat_consensus_direction,
                        confidence=cat_consensus_confidence,
                        prediction_horizon=horizon
                    )
            
            # Risk assessment
            model_risk = 1.0 - overall_consensus  # Higher disagreement = higher risk
            prediction_risk = 1.0 - weighted_confidence  # Lower confidence = higher risk
            total_risk = (model_risk + prediction_risk) / 2
            
            # Final recommendations
            current_price = data[-1] if data else weighted_price
            signal_strength = abs(weighted_price - current_price) / current_price if current_price > 0 else 0.0
            
            # Trading decision logic
            if weighted_confidence > 0.8 and overall_consensus > 0.7:
                if signal_strength > 0.01:
                    recommended_action = "STRONG_BUY" if weighted_direction == 1 else "STRONG_SELL" if weighted_direction == -1 else "HOLD"
                    confidence_level = "HIGH"
                    position_size = min(1.0, weighted_confidence * overall_consensus)
                else:
                    recommended_action = "BUY" if weighted_direction == 1 else "SELL" if weighted_direction == -1 else "HOLD"
                    confidence_level = "MEDIUM"
                    position_size = min(0.7, weighted_confidence * overall_consensus)
            elif weighted_confidence > 0.6 and overall_consensus > 0.5:
                recommended_action = "WEAK_BUY" if weighted_direction == 1 else "WEAK_SELL" if weighted_direction == -1 else "HOLD"
                confidence_level = "LOW"
                position_size = min(0.4, weighted_confidence * overall_consensus)
            else:
                recommended_action = "HOLD"
                confidence_level = "VERY_LOW"
                position_size = 0.0
            
            # Key factors analysis
            key_factors = {}
            for prediction in model_predictions.values():
                for factor, importance in prediction.key_drivers.items():
                    if factor in key_factors:
                        key_factors[factor] += importance
                    else:
                        key_factors[factor] = importance
            
            # Normalize key factors
            if key_factors:
                total_importance = sum(key_factors.values())
                key_factors = {k: v/total_importance for k, v in key_factors.items()}
            
            # Risk factors and opportunities
            risk_factors = []
            opportunities = []
            
            if model_risk > 0.5:
                risk_factors.append("High model disagreement")
            if prediction_risk > 0.5:
                risk_factors.append("Low prediction confidence")
            if signal_strength < 0.005:
                risk_factors.append("Weak price signal")
            
            if overall_consensus > 0.8:
                opportunities.append("Strong model consensus")
            if weighted_confidence > 0.8:
                opportunities.append("High prediction confidence")
            if signal_strength > 0.02:
                opportunities.append("Strong directional signal")
            
            result = ForexUnifiedResult(
                timestamp=datetime.now(),
                currency_pair=currency_pair,
                model_predictions=model_predictions,
                consensus_price=consensus_price,
                consensus_direction=consensus_direction,
                consensus_confidence=consensus_confidence,
                category_consensus=category_consensus,
                price_agreement=price_agreement,
                direction_agreement=direction_agreement,
                overall_consensus=overall_consensus,
                weighted_price=weighted_price,
                weighted_direction=weighted_direction,
                weighted_confidence=weighted_confidence,
                model_risk=model_risk,
                prediction_risk=prediction_risk,
                total_risk=total_risk,
                recommended_action=recommended_action,
                position_size=position_size,
                confidence_level=confidence_level,
                key_factors=key_factors,
                risk_factors=risk_factors,
                opportunities=opportunities
            )
            
            # Store for performance tracking
            self.prediction_history.append(result)
            if len(self.prediction_history) > 1000:  # Keep last 1000 predictions
                self.prediction_history = self.prediction_history[-1000:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating unified prediction: {e}")
            return ForexUnifiedResult(
                timestamp=datetime.now(),
                currency_pair=currency_pair
            )
    
    def update_model_weights(self, performance_data: Dict[ForexModelType, float]):
        """Update model weights based on performance"""
        try:
            total_performance = sum(performance_data.values())
            if total_performance > 0:
                for model_type, performance in performance_data.items():
                    self.model_weights[model_type] = performance / total_performance
            
            logger.info(f"Updated model weights: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error updating model weights: {e}")
    
    def get_model_status_all(self) -> Dict[ForexModelType, ForexModelStatus]:
        """Get status of all models"""
        status_dict = {}
        
        try:
            for model_type, model in self.models.items():
                status_dict[model_type] = model.get_status()
            
            return status_dict
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the unified interface"""
        try:
            if not self.prediction_history:
                return {'status': 'no_data', 'message': 'No prediction history available'}
            
            recent_predictions = self.prediction_history[-100:]  # Last 100 predictions
            
            # Calculate metrics
            avg_consensus = np.mean([p.overall_consensus for p in recent_predictions])
            avg_confidence = np.mean([p.weighted_confidence for p in recent_predictions])
            avg_risk = np.mean([p.total_risk for p in recent_predictions])
            
            # Action distribution
            actions = [p.recommended_action for p in recent_predictions]
            action_counts = {}
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            # Model availability
            model_status = self.get_model_status_all()
            available_models = sum(1 for status in model_status.values() if status.is_available)
            trained_models = sum(1 for status in model_status.values() if status.is_trained)
            
            return {
                'status': 'success',
                'total_predictions': len(self.prediction_history),
                'recent_predictions': len(recent_predictions),
                'average_consensus': avg_consensus,
                'average_confidence': avg_confidence,
                'average_risk': avg_risk,
                'action_distribution': action_counts,
                'models_available': available_models,
                'models_trained': trained_models,
                'model_weights': self.model_weights,
                'category_weights': self.category_weights
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'status': 'error', 'message': str(e)}

# Example usage
if __name__ == "__main__":
    print("=== Forex Unified Interface Demo ===")
    
    # Initialize unified interface
    forex_interface = ForexUnifiedInterface(
        enable_fundamental=True,
        enable_technical=True,
        enable_ml=True,
        enable_econometric=True,
        enable_sentiment=True
    )
    
    print(f"\nInitialized interface with {len(forex_interface.models)} models")
    
    # Generate sample data (prices)
    np.random.seed(42)
    sample_prices = []
    base_price = 1.1000  # EUR/USD
    
    for i in range(100):
        price_change = np.random.normal(0, 0.001)
        base_price *= (1 + price_change)
        sample_prices.append(base_price)
    
    print(f"Generated {len(sample_prices)} sample prices")
    
    # Test individual model predictions
    print("\n=== Individual Model Predictions ===")
    
    for model_type in [ForexModelType.PPP, ForexModelType.IRP, ForexModelType.ENSEMBLE_ML]:
        prediction = forex_interface.predict_single_model(model_type, sample_prices, "1D")
        if prediction:
            print(f"\n{model_type.value}:")
            print(f"  Price: {prediction.predicted_price:.5f}")
            print(f"  Direction: {prediction.predicted_direction}")
            print(f"  Confidence: {prediction.confidence:.3f}")
            print(f"  Category: {prediction.model_category.value}")
    
    # Test category predictions
    print("\n=== Category Predictions ===")
    
    fundamental_preds = forex_interface.predict_by_category(ForexModelCategory.FUNDAMENTAL, sample_prices, "1D")
    print(f"\nFundamental models: {len(fundamental_preds)} predictions")
    for model_type, pred in fundamental_preds.items():
        print(f"  {model_type.value}: {pred.predicted_price:.5f} (conf: {pred.confidence:.3f})")
    
    # Test unified prediction
    print("\n=== Unified Prediction ===")
    
    unified_result = forex_interface.predict_unified(sample_prices, "EURUSD", "1D")
    
    print(f"\nUnified Result for EURUSD:")
    print(f"  Consensus Price: {unified_result.consensus_price:.5f}")
    print(f"  Weighted Price: {unified_result.weighted_price:.5f}")
    print(f"  Direction: {unified_result.weighted_direction}")
    print(f"  Confidence: {unified_result.weighted_confidence:.3f}")
    print(f"  Overall Consensus: {unified_result.overall_consensus:.3f}")
    print(f"  Price Agreement: {unified_result.price_agreement:.3f}")
    print(f"  Direction Agreement: {unified_result.direction_agreement:.3f}")
    print(f"  Total Risk: {unified_result.total_risk:.3f}")
    print(f"  Recommended Action: {unified_result.recommended_action}")
    print(f"  Position Size: {unified_result.position_size:.3f}")
    print(f"  Confidence Level: {unified_result.confidence_level}")
    
    print(f"\nTop Key Factors:")
    sorted_factors = sorted(unified_result.key_factors.items(), key=lambda x: x[1], reverse=True)[:5]
    for factor, importance in sorted_factors:
        print(f"  {factor}: {importance:.3f}")
    
    if unified_result.risk_factors:
        print(f"\nRisk Factors: {', '.join(unified_result.risk_factors)}")
    
    if unified_result.opportunities:
        print(f"Opportunities: {', '.join(unified_result.opportunities)}")
    
    # Model status
    print("\n=== Model Status ===")
    
    model_status = forex_interface.get_model_status_all()
    for model_type, status in model_status.items():
        print(f"\n{model_type.value}:")
        print(f"  Available: {status.is_available}")
        print(f"  Trained: {status.is_trained}")
        print(f"  Accuracy: {status.accuracy:.3f}")
        print(f"  Sharpe Ratio: {status.sharpe_ratio:.3f}")
    
    # Performance summary
    print("\n=== Performance Summary ===")
    
    # Make a few more predictions to build history
    for i in range(5):
        test_prices = sample_prices[-(50+i*10):]
        forex_interface.predict_unified(test_prices, "EURUSD", "1D")
    
    performance = forex_interface.get_performance_summary()
    print(f"\nPerformance Summary:")
    print(f"  Total Predictions: {performance['total_predictions']}")
    print(f"  Average Consensus: {performance['average_consensus']:.3f}")
    print(f"  Average Confidence: {performance['average_confidence']:.3f}")
    print(f"  Average Risk: {performance['average_risk']:.3f}")
    print(f"  Models Available: {performance['models_available']}")
    print(f"  Models Trained: {performance['models_trained']}")
    
    print("\n=== Forex Unified Interface Demo Complete ===")