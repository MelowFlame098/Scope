import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Import existing futures models
try:
    from .advanced_ml_models import FuturesAdvancedMLModels, FuturesData, FuturesEnsembleResult
    from .futures_comprehensive import FuturesComprehensiveIndicators
    from .samuelson_backwardation import SamuelsonBackwardationModel
    from .technical_indicators import FuturesTechnicalIndicators
except ImportError:
    # Fallback for standalone execution
    from advanced_ml_models import FuturesAdvancedMLModels, FuturesData, FuturesEnsembleResult
    
    # Mock classes for missing imports
    class FuturesComprehensiveIndicators:
        def analyze(self, data): return type('Result', (), {'values': pd.DataFrame({'signal': [0.5]})})()
    
    class SamuelsonBackwardationModel:
        def analyze(self, data): return type('Result', (), {'samuelson_effect': 0.5, 'backwardation_signal': 0.5})()
    
    class FuturesTechnicalIndicators:
        def calculate_all_indicators(self, data): return {'rsi': 50, 'macd': 0, 'bollinger_signal': 0}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FuturesModelCategory(Enum):
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    TERM_STRUCTURE = "term_structure"
    MACHINE_LEARNING = "machine_learning"
    ENSEMBLE = "ensemble"

@dataclass
class FuturesPrediction:
    """Unified prediction result for futures models"""
    contract_symbol: str
    model_name: str
    category: FuturesModelCategory
    prediction_value: float
    confidence: float
    signal_strength: float  # -1 to 1, where -1 is strong sell, 1 is strong buy
    timestamp: datetime
    
    # Futures-specific predictions
    price_target: Optional[float] = None
    basis_forecast: Optional[float] = None
    roll_yield_forecast: Optional[float] = None
    term_structure_signal: Optional[str] = None  # 'contango', 'backwardation', 'neutral'
    volatility_forecast: Optional[float] = None
    
    # Risk metrics
    downside_risk: Optional[float] = None
    upside_potential: Optional[float] = None
    max_drawdown_estimate: Optional[float] = None
    
    # Model metadata
    model_version: str = "1.0"
    data_quality_score: float = 1.0
    prediction_horizon: int = 1  # days
    
    # Additional context
    market_regime: Optional[str] = None
    seasonal_factor: Optional[float] = None
    storage_cost_impact: Optional[float] = None

@dataclass
class FuturesModelStatus:
    """Status information for futures models"""
    model_name: str
    category: FuturesModelCategory
    is_available: bool
    last_update: datetime
    performance_score: float  # 0-1 scale
    reliability_score: float  # 0-1 scale
    
    # Model-specific status
    training_status: str = "ready"  # 'training', 'ready', 'error'
    data_freshness: int = 0  # minutes since last data update
    prediction_accuracy: float = 0.0  # recent accuracy score
    
    # Resource usage
    memory_usage: float = 0.0  # MB
    computation_time: float = 0.0  # seconds for last prediction
    
    # Error information
    last_error: Optional[str] = None
    error_count: int = 0

@dataclass
class FuturesUnifiedResult:
    """Comprehensive unified result from all futures models"""
    contract_symbol: str
    analysis_timestamp: datetime
    
    # Consensus predictions
    consensus_signal: float  # -1 to 1
    consensus_price_target: float
    consensus_confidence: float
    
    # Category-wise results
    fundamental_score: float = 0.0
    technical_score: float = 0.0
    term_structure_score: float = 0.0
    ml_score: float = 0.0
    
    # Individual model predictions
    model_predictions: List[FuturesPrediction] = field(default_factory=list)
    
    # Futures-specific analysis
    term_structure_analysis: Dict[str, Any] = field(default_factory=dict)
    basis_analysis: Dict[str, float] = field(default_factory=dict)
    roll_schedule_analysis: Dict[str, Any] = field(default_factory=dict)
    
    # Risk assessment
    overall_risk_score: float = 0.0
    volatility_forecast: float = 0.0
    liquidity_assessment: str = "unknown"
    
    # Market context
    market_regime: str = "neutral"
    seasonal_factors: Dict[str, float] = field(default_factory=dict)
    curve_shape: str = "normal"  # 'contango', 'backwardation', 'normal'
    
    # Model performance
    model_agreement: float = 0.0  # How much models agree
    prediction_uncertainty: float = 0.0
    
    # Actionable insights
    trading_signals: List[str] = field(default_factory=list)
    risk_warnings: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)

class FuturesUnifiedInterface:
    """Unified interface for all futures trading models"""
    
    def __init__(self, 
                 enable_ml_models: bool = True,
                 enable_fundamental_analysis: bool = True,
                 enable_technical_analysis: bool = True,
                 enable_term_structure_analysis: bool = True,
                 model_weights: Optional[Dict[str, float]] = None):
        """
        Initialize the unified futures interface
        
        Args:
            enable_ml_models: Enable machine learning models
            enable_fundamental_analysis: Enable fundamental analysis
            enable_technical_analysis: Enable technical analysis
            enable_term_structure_analysis: Enable term structure analysis
            model_weights: Custom weights for model categories
        """
        self.enable_ml_models = enable_ml_models
        self.enable_fundamental_analysis = enable_fundamental_analysis
        self.enable_technical_analysis = enable_technical_analysis
        self.enable_term_structure_analysis = enable_term_structure_analysis
        
        # Default model weights
        self.model_weights = model_weights or {
            'fundamental': 0.25,
            'technical': 0.25,
            'term_structure': 0.25,
            'machine_learning': 0.25
        }
        
        # Initialize models
        self.models = {}
        self.model_status = {}
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all available models"""
        try:
            # ML Models
            if self.enable_ml_models:
                self.models['ml_suite'] = FuturesAdvancedMLModels()
                self.model_status['ml_suite'] = FuturesModelStatus(
                    model_name='ML Suite',
                    category=FuturesModelCategory.MACHINE_LEARNING,
                    is_available=True,
                    last_update=datetime.now(),
                    performance_score=0.8,
                    reliability_score=0.85
                )
            
            # Fundamental Analysis
            if self.enable_fundamental_analysis:
                self.models['comprehensive'] = FuturesComprehensiveIndicators()
                self.model_status['comprehensive'] = FuturesModelStatus(
                    model_name='Comprehensive Analysis',
                    category=FuturesModelCategory.FUNDAMENTAL,
                    is_available=True,
                    last_update=datetime.now(),
                    performance_score=0.75,
                    reliability_score=0.8
                )
            
            # Term Structure Analysis
            if self.enable_term_structure_analysis:
                self.models['samuelson'] = SamuelsonBackwardationModel()
                self.model_status['samuelson'] = FuturesModelStatus(
                    model_name='Samuelson Backwardation',
                    category=FuturesModelCategory.TERM_STRUCTURE,
                    is_available=True,
                    last_update=datetime.now(),
                    performance_score=0.7,
                    reliability_score=0.75
                )
            
            # Technical Analysis
            if self.enable_technical_analysis:
                self.models['technical'] = FuturesTechnicalIndicators()
                self.model_status['technical'] = FuturesModelStatus(
                    model_name='Technical Indicators',
                    category=FuturesModelCategory.TECHNICAL,
                    is_available=True,
                    last_update=datetime.now(),
                    performance_score=0.65,
                    reliability_score=0.7
                )
            
            logger.info(f"Initialized {len(self.models)} futures models")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    def predict_from_fundamental_models(self, data: FuturesData) -> List[FuturesPrediction]:
        """Get predictions from fundamental analysis models"""
        predictions = []
        
        try:
            if 'comprehensive' in self.models:
                # Convert FuturesData to format expected by comprehensive model
                df = pd.DataFrame({
                    'close': data.close,
                    'volume': data.volume,
                    'open_interest': data.open_interest,
                    'timestamp': data.timestamps
                })
                
                result = self.models['comprehensive'].analyze(df)
                
                # Extract signal from result
                if hasattr(result, 'values') and not result.values.empty:
                    signal_value = result.values.iloc[-1, 0] if len(result.values.columns) > 0 else 0.5
                else:
                    signal_value = 0.5
                
                prediction = FuturesPrediction(
                    contract_symbol=data.contract_symbol,
                    model_name='Comprehensive Analysis',
                    category=FuturesModelCategory.FUNDAMENTAL,
                    prediction_value=signal_value,
                    confidence=0.75,
                    signal_strength=(signal_value - 0.5) * 2,  # Convert to -1 to 1 scale
                    timestamp=datetime.now(),
                    price_target=data.close[-1] * (1 + (signal_value - 0.5) * 0.1) if data.close else None
                )
                
                predictions.append(prediction)
                
        except Exception as e:
            logger.error(f"Error in fundamental analysis: {e}")
        
        return predictions
    
    def predict_from_technical_models(self, data: FuturesData) -> List[FuturesPrediction]:
        """Get predictions from technical analysis models"""
        predictions = []
        
        try:
            if 'technical' in self.models:
                # Convert data format
                df = pd.DataFrame({
                    'close': data.close,
                    'high': data.high,
                    'low': data.low,
                    'volume': data.volume
                })
                
                indicators = self.models['technical'].calculate_all_indicators(df)
                
                # Combine technical indicators into a signal
                rsi = indicators.get('rsi', 50)
                macd = indicators.get('macd', 0)
                bollinger = indicators.get('bollinger_signal', 0)
                
                # Normalize RSI to -1 to 1 scale
                rsi_signal = (rsi - 50) / 50
                
                # Combine signals
                combined_signal = (rsi_signal + np.sign(macd) * 0.5 + bollinger) / 3
                combined_signal = max(-1, min(1, combined_signal))
                
                prediction = FuturesPrediction(
                    contract_symbol=data.contract_symbol,
                    model_name='Technical Indicators',
                    category=FuturesModelCategory.TECHNICAL,
                    prediction_value=(combined_signal + 1) / 2,  # Convert to 0-1 scale
                    confidence=0.7,
                    signal_strength=combined_signal,
                    timestamp=datetime.now(),
                    price_target=data.close[-1] * (1 + combined_signal * 0.05) if data.close else None
                )
                
                predictions.append(prediction)
                
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
        
        return predictions
    
    def predict_from_term_structure_models(self, data: FuturesData) -> List[FuturesPrediction]:
        """Get predictions from term structure models"""
        predictions = []
        
        try:
            if 'samuelson' in self.models:
                # Convert data format
                df = pd.DataFrame({
                    'close': data.close,
                    'timestamp': data.timestamps,
                    'volume': data.volume
                })
                
                result = self.models['samuelson'].analyze(df)
                
                # Extract signals
                samuelson_effect = getattr(result, 'samuelson_effect', 0.5)
                backwardation_signal = getattr(result, 'backwardation_signal', 0.5)
                
                # Combine term structure signals
                combined_signal = (samuelson_effect + backwardation_signal) / 2
                signal_strength = (combined_signal - 0.5) * 2
                
                # Determine term structure regime
                if data.basis and len(data.basis) > 0:
                    avg_basis = np.mean(data.basis[-10:]) if len(data.basis) >= 10 else data.basis[-1]
                    if avg_basis > 0.02:
                        term_structure_signal = 'contango'
                    elif avg_basis < -0.02:
                        term_structure_signal = 'backwardation'
                    else:
                        term_structure_signal = 'neutral'
                else:
                    term_structure_signal = 'neutral'
                
                prediction = FuturesPrediction(
                    contract_symbol=data.contract_symbol,
                    model_name='Samuelson Backwardation',
                    category=FuturesModelCategory.TERM_STRUCTURE,
                    prediction_value=combined_signal,
                    confidence=0.8,
                    signal_strength=signal_strength,
                    timestamp=datetime.now(),
                    term_structure_signal=term_structure_signal,
                    basis_forecast=data.basis[-1] * 1.1 if data.basis else None
                )
                
                predictions.append(prediction)
                
        except Exception as e:
            logger.error(f"Error in term structure analysis: {e}")
        
        return predictions
    
    def predict_from_ml_models(self, data: FuturesData) -> List[FuturesPrediction]:
        """Get predictions from machine learning models"""
        predictions = []
        
        try:
            if 'ml_suite' in self.models:
                # Train models if not already trained
                training_results = self.models['ml_suite'].train_all_models(data)
                
                # Get ensemble prediction
                ensemble_result = self.models['ml_suite'].predict_ensemble(data)
                
                prediction = FuturesPrediction(
                    contract_symbol=data.contract_symbol,
                    model_name='ML Ensemble',
                    category=FuturesModelCategory.MACHINE_LEARNING,
                    prediction_value=ensemble_result.consensus_prediction / data.close[-1] if data.close and data.close[-1] != 0 else 1.0,
                    confidence=ensemble_result.confidence,
                    signal_strength=(ensemble_result.consensus_prediction / data.close[-1] - 1) * 10 if data.close and data.close[-1] != 0 else 0.0,
                    timestamp=datetime.now(),
                    price_target=ensemble_result.consensus_prediction,
                    volatility_forecast=ensemble_result.ensemble_uncertainty
                )
                
                # Normalize signal strength to -1 to 1 range
                prediction.signal_strength = max(-1, min(1, prediction.signal_strength))
                
                predictions.append(prediction)
                
        except Exception as e:
            logger.error(f"Error in ML analysis: {e}")
        
        return predictions
    
    def generate_unified_prediction(self, data: FuturesData) -> FuturesUnifiedResult:
        """Generate comprehensive unified prediction from all models"""
        try:
            # Get predictions from all model categories
            all_predictions = []
            
            if self.enable_fundamental_analysis:
                all_predictions.extend(self.predict_from_fundamental_models(data))
            
            if self.enable_technical_analysis:
                all_predictions.extend(self.predict_from_technical_models(data))
            
            if self.enable_term_structure_analysis:
                all_predictions.extend(self.predict_from_term_structure_models(data))
            
            if self.enable_ml_models:
                all_predictions.extend(self.predict_from_ml_models(data))
            
            if not all_predictions:
                return self._create_default_result(data)
            
            # Calculate category scores
            category_scores = {
                'fundamental': 0.0,
                'technical': 0.0,
                'term_structure': 0.0,
                'ml': 0.0
            }
            
            category_counts = {k: 0 for k in category_scores.keys()}
            
            for pred in all_predictions:
                if pred.category == FuturesModelCategory.FUNDAMENTAL:
                    category_scores['fundamental'] += pred.signal_strength
                    category_counts['fundamental'] += 1
                elif pred.category == FuturesModelCategory.TECHNICAL:
                    category_scores['technical'] += pred.signal_strength
                    category_counts['technical'] += 1
                elif pred.category == FuturesModelCategory.TERM_STRUCTURE:
                    category_scores['term_structure'] += pred.signal_strength
                    category_counts['term_structure'] += 1
                elif pred.category == FuturesModelCategory.MACHINE_LEARNING:
                    category_scores['ml'] += pred.signal_strength
                    category_counts['ml'] += 1
            
            # Average category scores
            for category in category_scores:
                if category_counts[category] > 0:
                    category_scores[category] /= category_counts[category]
            
            # Calculate consensus signal using weighted average
            consensus_signal = (
                category_scores['fundamental'] * self.model_weights.get('fundamental', 0.25) +
                category_scores['technical'] * self.model_weights.get('technical', 0.25) +
                category_scores['term_structure'] * self.model_weights.get('term_structure', 0.25) +
                category_scores['ml'] * self.model_weights.get('machine_learning', 0.25)
            )
            
            # Calculate consensus price target
            price_targets = [p.price_target for p in all_predictions if p.price_target is not None]
            consensus_price_target = np.mean(price_targets) if price_targets else (data.close[-1] if data.close else 100.0)
            
            # Calculate consensus confidence
            confidences = [p.confidence for p in all_predictions]
            consensus_confidence = np.mean(confidences) if confidences else 0.5
            
            # Model agreement calculation
            signal_strengths = [p.signal_strength for p in all_predictions]
            model_agreement = 1.0 - (np.std(signal_strengths) / (np.mean(np.abs(signal_strengths)) + 1e-8)) if signal_strengths else 0.0
            model_agreement = max(0.0, min(1.0, model_agreement))
            
            # Determine market regime
            market_regime = 'neutral'
            if data.basis and len(data.basis) > 0:
                recent_basis = np.mean(data.basis[-5:]) if len(data.basis) >= 5 else data.basis[-1]
                if recent_basis > 0.02:
                    market_regime = 'contango'
                elif recent_basis < -0.02:
                    market_regime = 'backwardation'
            
            # Generate trading signals
            trading_signals = []
            if consensus_signal > 0.3:
                trading_signals.append(f"BUY signal detected (strength: {consensus_signal:.2f})")
            elif consensus_signal < -0.3:
                trading_signals.append(f"SELL signal detected (strength: {consensus_signal:.2f})")
            else:
                trading_signals.append("NEUTRAL - No clear directional signal")
            
            # Risk warnings
            risk_warnings = []
            if model_agreement < 0.5:
                risk_warnings.append("Low model agreement - high uncertainty")
            if consensus_confidence < 0.4:
                risk_warnings.append("Low prediction confidence")
            
            # Opportunities
            opportunities = []
            if market_regime == 'contango' and consensus_signal < 0:
                opportunities.append("Potential roll yield opportunity in contango market")
            elif market_regime == 'backwardation' and consensus_signal > 0:
                opportunities.append("Favorable backwardation structure for long positions")
            
            return FuturesUnifiedResult(
                contract_symbol=data.contract_symbol,
                analysis_timestamp=datetime.now(),
                consensus_signal=consensus_signal,
                consensus_price_target=consensus_price_target,
                consensus_confidence=consensus_confidence,
                fundamental_score=category_scores['fundamental'],
                technical_score=category_scores['technical'],
                term_structure_score=category_scores['term_structure'],
                ml_score=category_scores['ml'],
                model_predictions=all_predictions,
                market_regime=market_regime,
                model_agreement=model_agreement,
                prediction_uncertainty=np.std(signal_strengths) if signal_strengths else 0.0,
                trading_signals=trading_signals,
                risk_warnings=risk_warnings,
                opportunities=opportunities,
                curve_shape=market_regime if market_regime != 'neutral' else 'normal'
            )
            
        except Exception as e:
            logger.error(f"Error generating unified prediction: {e}")
            return self._create_default_result(data)
    
    def _create_default_result(self, data: FuturesData) -> FuturesUnifiedResult:
        """Create default result when analysis fails"""
        return FuturesUnifiedResult(
            contract_symbol=data.contract_symbol,
            analysis_timestamp=datetime.now(),
            consensus_signal=0.0,
            consensus_price_target=data.close[-1] if data.close else 100.0,
            consensus_confidence=0.1,
            trading_signals=["Analysis unavailable - insufficient data or model errors"],
            risk_warnings=["Unable to perform comprehensive analysis"]
        )
    
    def get_model_status(self) -> Dict[str, FuturesModelStatus]:
        """Get status of all models"""
        return self.model_status.copy()
    
    def update_model_weights(self, new_weights: Dict[str, float]):
        """Update model category weights"""
        # Normalize weights to sum to 1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            self.model_weights = {k: v/total_weight for k, v in new_weights.items()}
            logger.info(f"Updated model weights: {self.model_weights}")

# Example usage
if __name__ == "__main__":
    # Create sample futures data
    np.random.seed(42)
    n_points = 500
    
    # Generate synthetic futures data with term structure
    base_price = 75.0
    price_trend = np.cumsum(np.random.normal(0.001, 0.02, n_points))
    prices = base_price * np.exp(price_trend)
    
    sample_data = FuturesData(
        prices=prices.tolist(),
        returns=np.diff(prices, prepend=prices[0]).tolist(),
        volume=np.random.lognormal(10, 0.5, n_points).tolist(),
        open_interest=np.random.lognormal(12, 0.3, n_points).tolist(),
        timestamps=[datetime.now() - timedelta(days=n_points-i) for i in range(n_points)],
        high=(prices * (1 + np.random.uniform(0, 0.015, n_points))).tolist(),
        low=(prices * (1 - np.random.uniform(0, 0.015, n_points))).tolist(),
        open=prices.tolist(),
        close=prices.tolist(),
        contract_symbol='ES_2024_06',
        underlying_asset='S&P 500 E-mini',
        basis=np.random.normal(0.01, 0.3, n_points).tolist(),
        roll_yield=np.random.normal(-0.005, 0.05, n_points).tolist(),
        convenience_yield=np.random.normal(0.01, 0.005, n_points).tolist()
    )
    
    print("=== Futures Unified Interface Demo ===")
    
    # Initialize unified interface
    futures_interface = FuturesUnifiedInterface(
        enable_ml_models=True,
        enable_fundamental_analysis=True,
        enable_technical_analysis=True,
        enable_term_structure_analysis=True
    )
    
    print(f"\nInitialized with {len(futures_interface.models)} models")
    
    # Get model status
    print("\n=== Model Status ===")
    model_status = futures_interface.get_model_status()
    for name, status in model_status.items():
        print(f"{status.model_name}: {status.category.value} - {'✓' if status.is_available else '✗'}")
        print(f"  Performance: {status.performance_score:.2f}, Reliability: {status.reliability_score:.2f}")
    
    # Generate unified prediction
    print("\n=== Generating Unified Prediction ===")
    result = futures_interface.generate_unified_prediction(sample_data)
    
    print(f"\n=== Unified Analysis Results ===")
    print(f"Contract: {result.contract_symbol}")
    print(f"Consensus Signal: {result.consensus_signal:.3f}")
    print(f"Consensus Price Target: ${result.consensus_price_target:.2f}")
    print(f"Consensus Confidence: {result.consensus_confidence:.3f}")
    print(f"Market Regime: {result.market_regime}")
    print(f"Model Agreement: {result.model_agreement:.3f}")
    
    print(f"\n=== Category Scores ===")
    print(f"Fundamental: {result.fundamental_score:.3f}")
    print(f"Technical: {result.technical_score:.3f}")
    print(f"Term Structure: {result.term_structure_score:.3f}")
    print(f"Machine Learning: {result.ml_score:.3f}")
    
    print(f"\n=== Individual Model Predictions ===")
    for pred in result.model_predictions:
        print(f"{pred.model_name} ({pred.category.value}):")
        print(f"  Signal Strength: {pred.signal_strength:.3f}")
        print(f"  Confidence: {pred.confidence:.3f}")
        if pred.price_target:
            print(f"  Price Target: ${pred.price_target:.2f}")
        if pred.term_structure_signal:
            print(f"  Term Structure: {pred.term_structure_signal}")
    
    print(f"\n=== Trading Signals ===")
    for signal in result.trading_signals:
        print(f"• {signal}")
    
    if result.risk_warnings:
        print(f"\n=== Risk Warnings ===")
        for warning in result.risk_warnings:
            print(f"⚠ {warning}")
    
    if result.opportunities:
        print(f"\n=== Opportunities ===")
        for opportunity in result.opportunities:
            print(f"💡 {opportunity}")
    
    print("\n=== Futures Unified Interface Complete ===")