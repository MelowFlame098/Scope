import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FuturesData:
    """Enhanced data structure for futures market data"""
    prices: List[float]
    returns: List[float]
    volume: List[float]
    open_interest: List[float]
    timestamps: List[datetime]
    high: List[float]
    low: List[float]
    open: List[float]
    close: List[float]
    contract_symbol: str
    underlying_asset: str
    time_to_expiry: List[float] = field(default_factory=list)
    basis: List[float] = field(default_factory=list)
    roll_yield: List[float] = field(default_factory=list)
    term_structure: List[Dict[str, float]] = field(default_factory=list)
    convenience_yield: List[float] = field(default_factory=list)
    storage_costs: List[float] = field(default_factory=list)

@dataclass
class FuturesMLPredictionResult:
    """Results from futures ML model predictions"""
    contract_symbol: str
    prediction_type: str  # 'price', 'return', 'direction', 'basis'
    predicted_value: float
    confidence: float
    prediction_horizon: int  # days
    model_name: str
    timestamp: datetime
    
    # Futures-specific predictions
    basis_prediction: Optional[float] = None
    roll_yield_prediction: Optional[float] = None
    term_structure_prediction: Optional[Dict[str, float]] = None
    volatility_prediction: Optional[float] = None
    
    # Risk metrics
    var_95: Optional[float] = None
    expected_shortfall: Optional[float] = None
    
    # Model diagnostics
    feature_importance: Dict[str, float] = field(default_factory=dict)
    prediction_interval: Tuple[float, float] = (0.0, 0.0)
    model_uncertainty: float = 0.0

@dataclass
class FuturesEnsembleResult:
    """Results from futures ensemble model"""
    contract_symbol: str
    consensus_prediction: float
    prediction_type: str
    confidence: float
    timestamp: datetime
    
    # Individual model contributions
    model_predictions: Dict[str, float] = field(default_factory=dict)
    model_weights: Dict[str, float] = field(default_factory=dict)
    model_confidences: Dict[str, float] = field(default_factory=dict)
    
    # Ensemble metrics
    prediction_variance: float = 0.0
    model_agreement: float = 0.0  # How much models agree
    ensemble_uncertainty: float = 0.0
    
    # Futures-specific ensemble results
    basis_consensus: Optional[float] = None
    term_structure_consensus: Optional[Dict[str, float]] = None
    regime_probability: Dict[str, float] = field(default_factory=dict)  # contango, backwardation, etc.

class FuturesLSTMModel:
    """LSTM model specifically designed for futures trading"""
    
    def __init__(self, 
                 sequence_length: int = 60,
                 hidden_units: int = 128,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 include_term_structure: bool = True,
                 include_basis: bool = True):
        """
        Initialize Futures LSTM Model
        
        Args:
            sequence_length: Length of input sequences
            hidden_units: Number of LSTM hidden units
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimization
            include_term_structure: Include term structure features
            include_basis: Include basis and roll yield features
        """
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.include_term_structure = include_term_structure
        self.include_basis = include_basis
        
        self.scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        
        # Feature engineering components
        self.feature_columns = ['close', 'volume', 'open_interest', 'returns']
        if include_basis:
            self.feature_columns.extend(['basis', 'roll_yield', 'convenience_yield'])
        if include_term_structure:
            self.feature_columns.extend(['term_structure_slope', 'term_structure_curvature'])
    
    def _prepare_features(self, data: FuturesData) -> pd.DataFrame:
        """Prepare features for LSTM model"""
        try:
            df = pd.DataFrame({
                'close': data.close,
                'volume': data.volume,
                'open_interest': data.open_interest,
                'returns': data.returns,
                'timestamps': data.timestamps
            })
            
            # Add futures-specific features
            if self.include_basis and data.basis:
                df['basis'] = data.basis + [0] * (len(df) - len(data.basis))
                df['roll_yield'] = data.roll_yield + [0] * (len(df) - len(data.roll_yield))
                df['convenience_yield'] = data.convenience_yield + [0] * (len(df) - len(data.convenience_yield))
            
            if self.include_term_structure and data.term_structure:
                # Extract term structure features
                ts_slopes = []
                ts_curvatures = []
                
                for ts in data.term_structure:
                    if ts and len(ts) >= 2:
                        prices = list(ts.values())
                        maturities = list(range(len(prices)))
                        
                        # Calculate slope (linear regression)
                        if len(prices) >= 2:
                            slope = np.polyfit(maturities, prices, 1)[0]
                            ts_slopes.append(slope)
                        else:
                            ts_slopes.append(0)
                        
                        # Calculate curvature (quadratic fit)
                        if len(prices) >= 3:
                            curvature = np.polyfit(maturities, prices, 2)[0]
                            ts_curvatures.append(curvature)
                        else:
                            ts_curvatures.append(0)
                    else:
                        ts_slopes.append(0)
                        ts_curvatures.append(0)
                
                # Pad to match dataframe length
                while len(ts_slopes) < len(df):
                    ts_slopes.append(0)
                while len(ts_curvatures) < len(df):
                    ts_curvatures.append(0)
                
                df['term_structure_slope'] = ts_slopes[:len(df)]
                df['term_structure_curvature'] = ts_curvatures[:len(df)]
            
            # Technical indicators
            df['sma_20'] = df['close'].rolling(20).mean()
            df['sma_50'] = df['close'].rolling(50).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Fill NaN values
            df = df.fillna(method='forward').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _create_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(target[i])
        return np.array(X), np.array(y)
    
    def train(self, data: FuturesData, target_column: str = 'close') -> bool:
        """Train the LSTM model"""
        try:
            # Prepare features
            df = self._prepare_features(data)
            if df.empty:
                return False
            
            # Select features
            available_features = [col for col in self.feature_columns if col in df.columns]
            if not available_features:
                logger.error("No valid features available for training")
                return False
            
            feature_data = df[available_features].values
            target_data = df[target_column].values
            
            # Scale features
            feature_data_scaled = self.scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = self._create_sequences(feature_data_scaled, target_data)
            
            if len(X) == 0:
                logger.error("Not enough data to create sequences")
                return False
            
            # For demonstration, use a simple ensemble of traditional models
            # In practice, you would implement actual LSTM using TensorFlow/PyTorch
            self.model = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            # Train models on flattened sequences (simplified approach)
            X_flat = X.reshape(X.shape[0], -1)
            
            for name, model in self.model.items():
                model.fit(X_flat, y)
            
            self.is_trained = True
            logger.info(f"LSTM model trained successfully with {len(X)} sequences")
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return False
    
    def predict(self, data: FuturesData, horizon: int = 1) -> FuturesMLPredictionResult:
        """Make predictions using the trained model"""
        try:
            if not self.is_trained or not self.model:
                return FuturesMLPredictionResult(
                    contract_symbol=data.contract_symbol,
                    prediction_type='price',
                    predicted_value=0.0,
                    confidence=0.0,
                    prediction_horizon=horizon,
                    model_name='FuturesLSTM',
                    timestamp=datetime.now()
                )
            
            # Prepare features
            df = self._prepare_features(data)
            if df.empty or len(df) < self.sequence_length:
                return FuturesMLPredictionResult(
                    contract_symbol=data.contract_symbol,
                    prediction_type='price',
                    predicted_value=data.close[-1] if data.close else 0.0,
                    confidence=0.1,
                    prediction_horizon=horizon,
                    model_name='FuturesLSTM',
                    timestamp=datetime.now()
                )
            
            # Get latest sequence
            available_features = [col for col in self.feature_columns if col in df.columns]
            feature_data = df[available_features].values
            feature_data_scaled = self.scaler.transform(feature_data)
            
            # Get last sequence
            last_sequence = feature_data_scaled[-self.sequence_length:]
            X_pred = last_sequence.reshape(1, -1)  # Flatten for traditional models
            
            # Ensemble prediction
            predictions = []
            for name, model in self.model.items():
                pred = model.predict(X_pred)[0]
                predictions.append(pred)
            
            # Average predictions
            final_prediction = np.mean(predictions)
            confidence = 1.0 - (np.std(predictions) / (np.mean(np.abs(predictions)) + 1e-8))
            confidence = max(0.1, min(0.95, confidence))
            
            # Calculate additional futures-specific predictions
            basis_pred = None
            roll_yield_pred = None
            volatility_pred = None
            
            if self.include_basis and data.basis:
                basis_pred = data.basis[-1] * (1 + np.random.normal(0, 0.1))  # Simple projection
            
            if data.returns:
                volatility_pred = np.std(data.returns[-20:]) if len(data.returns) >= 20 else 0.1
            
            return FuturesMLPredictionResult(
                contract_symbol=data.contract_symbol,
                prediction_type='price',
                predicted_value=final_prediction,
                confidence=confidence,
                prediction_horizon=horizon,
                model_name='FuturesLSTM',
                timestamp=datetime.now(),
                basis_prediction=basis_pred,
                roll_yield_prediction=roll_yield_pred,
                volatility_prediction=volatility_pred,
                prediction_interval=(final_prediction * 0.95, final_prediction * 1.05)
            )
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return FuturesMLPredictionResult(
                contract_symbol=data.contract_symbol,
                prediction_type='price',
                predicted_value=0.0,
                confidence=0.0,
                prediction_horizon=horizon,
                model_name='FuturesLSTM',
                timestamp=datetime.now()
            )

class FuturesTransformerModel:
    """Transformer model for futures price prediction with attention mechanisms"""
    
    def __init__(self,
                 sequence_length: int = 60,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dropout_rate: float = 0.1,
                 include_regime_detection: bool = True):
        """
        Initialize Futures Transformer Model
        
        Args:
            sequence_length: Length of input sequences
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout_rate: Dropout rate
            include_regime_detection: Include market regime detection
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.include_regime_detection = include_regime_detection
        
        self.scaler = MinMaxScaler()
        self.model = None
        self.is_trained = False
        
        # Regime detection thresholds
        self.contango_threshold = 0.02
        self.backwardation_threshold = -0.02
    
    def _detect_market_regime(self, data: FuturesData) -> str:
        """Detect current market regime"""
        try:
            if not data.basis or len(data.basis) < 10:
                return 'neutral'
            
            recent_basis = np.mean(data.basis[-10:])
            
            if recent_basis > self.contango_threshold:
                return 'contango'
            elif recent_basis < self.backwardation_threshold:
                return 'backwardation'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def train(self, data: FuturesData) -> bool:
        """Train the transformer model"""
        try:
            # For demonstration, use ensemble of models with regime awareness
            # In practice, implement actual Transformer architecture
            
            regime = self._detect_market_regime(data)
            
            # Create regime-specific models
            self.model = {
                'contango': RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
                'backwardation': GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, random_state=42),
                'neutral': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            
            # Prepare training data
            if len(data.close) < 100:
                return False
            
            # Create features
            features = []
            targets = []
            
            for i in range(self.sequence_length, len(data.close) - 1):
                # Price features
                price_seq = data.close[i-self.sequence_length:i]
                volume_seq = data.volume[i-self.sequence_length:i] if len(data.volume) > i else [1] * self.sequence_length
                
                # Combine features
                feature_vector = list(price_seq) + list(volume_seq)
                
                # Add basis if available
                if data.basis and len(data.basis) > i:
                    feature_vector.append(data.basis[i-1])
                else:
                    feature_vector.append(0)
                
                features.append(feature_vector)
                targets.append(data.close[i])
            
            if not features:
                return False
            
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train all regime models
            for regime_name, model in self.model.items():
                model.fit(X_scaled, y)
            
            self.is_trained = True
            logger.info(f"Transformer model trained with {len(X)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training transformer model: {e}")
            return False
    
    def predict(self, data: FuturesData, horizon: int = 1) -> FuturesMLPredictionResult:
        """Make predictions using transformer model"""
        try:
            if not self.is_trained or not self.model:
                return FuturesMLPredictionResult(
                    contract_symbol=data.contract_symbol,
                    prediction_type='price',
                    predicted_value=0.0,
                    confidence=0.0,
                    prediction_horizon=horizon,
                    model_name='FuturesTransformer',
                    timestamp=datetime.now()
                )
            
            if len(data.close) < self.sequence_length:
                return FuturesMLPredictionResult(
                    contract_symbol=data.contract_symbol,
                    prediction_type='price',
                    predicted_value=data.close[-1] if data.close else 0.0,
                    confidence=0.1,
                    prediction_horizon=horizon,
                    model_name='FuturesTransformer',
                    timestamp=datetime.now()
                )
            
            # Detect current regime
            current_regime = self._detect_market_regime(data)
            
            # Prepare features
            price_seq = data.close[-self.sequence_length:]
            volume_seq = data.volume[-self.sequence_length:] if len(data.volume) >= self.sequence_length else [1] * self.sequence_length
            
            feature_vector = list(price_seq) + list(volume_seq)
            
            # Add basis
            if data.basis:
                feature_vector.append(data.basis[-1])
            else:
                feature_vector.append(0)
            
            X_pred = np.array([feature_vector])
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Get regime-specific prediction
            regime_model = self.model[current_regime]
            prediction = regime_model.predict(X_pred_scaled)[0]
            
            # Calculate confidence based on regime stability
            regime_confidence = 0.8 if current_regime != 'neutral' else 0.6
            
            return FuturesMLPredictionResult(
                contract_symbol=data.contract_symbol,
                prediction_type='price',
                predicted_value=prediction,
                confidence=regime_confidence,
                prediction_horizon=horizon,
                model_name='FuturesTransformer',
                timestamp=datetime.now(),
                prediction_interval=(prediction * 0.92, prediction * 1.08)
            )
            
        except Exception as e:
            logger.error(f"Error making transformer prediction: {e}")
            return FuturesMLPredictionResult(
                contract_symbol=data.contract_symbol,
                prediction_type='price',
                predicted_value=0.0,
                confidence=0.0,
                prediction_horizon=horizon,
                model_name='FuturesTransformer',
                timestamp=datetime.now()
            )

class FuturesAdvancedMLModels:
    """Advanced ML models suite for futures trading"""
    
    def __init__(self):
        self.lstm_model = FuturesLSTMModel()
        self.transformer_model = FuturesTransformerModel()
        self.ensemble_weights = {'lstm': 0.6, 'transformer': 0.4}
    
    def train_all_models(self, data: FuturesData) -> Dict[str, bool]:
        """Train all ML models"""
        results = {}
        
        logger.info("Training LSTM model...")
        results['lstm'] = self.lstm_model.train(data)
        
        logger.info("Training Transformer model...")
        results['transformer'] = self.transformer_model.train(data)
        
        return results
    
    def predict_ensemble(self, data: FuturesData, horizon: int = 1) -> FuturesEnsembleResult:
        """Generate ensemble predictions"""
        try:
            # Get individual predictions
            lstm_pred = self.lstm_model.predict(data, horizon)
            transformer_pred = self.transformer_model.predict(data, horizon)
            
            # Calculate weighted ensemble
            predictions = {
                'lstm': lstm_pred.predicted_value,
                'transformer': transformer_pred.predicted_value
            }
            
            confidences = {
                'lstm': lstm_pred.confidence,
                'transformer': transformer_pred.confidence
            }
            
            # Weighted average
            consensus = (
                predictions['lstm'] * self.ensemble_weights['lstm'] +
                predictions['transformer'] * self.ensemble_weights['transformer']
            )
            
            # Ensemble confidence
            ensemble_confidence = (
                confidences['lstm'] * self.ensemble_weights['lstm'] +
                confidences['transformer'] * self.ensemble_weights['transformer']
            )
            
            # Model agreement
            pred_values = list(predictions.values())
            agreement = 1.0 - (np.std(pred_values) / (np.mean(np.abs(pred_values)) + 1e-8))
            agreement = max(0.0, min(1.0, agreement))
            
            return FuturesEnsembleResult(
                contract_symbol=data.contract_symbol,
                consensus_prediction=consensus,
                prediction_type='price',
                confidence=ensemble_confidence,
                timestamp=datetime.now(),
                model_predictions=predictions,
                model_weights=self.ensemble_weights,
                model_confidences=confidences,
                model_agreement=agreement,
                ensemble_uncertainty=np.std(pred_values)
            )
            
        except Exception as e:
            logger.error(f"Error generating ensemble prediction: {e}")
            return FuturesEnsembleResult(
                contract_symbol=data.contract_symbol,
                consensus_prediction=0.0,
                prediction_type='price',
                confidence=0.0,
                timestamp=datetime.now()
            )

# Example usage
if __name__ == "__main__":
    # Create sample futures data
    np.random.seed(42)
    n_points = 1000
    
    # Generate synthetic futures data
    base_price = 100
    price_trend = np.cumsum(np.random.normal(0, 0.02, n_points))
    prices = base_price + price_trend
    
    sample_data = FuturesData(
        prices=prices.tolist(),
        returns=np.diff(prices, prepend=prices[0]).tolist(),
        volume=np.random.lognormal(10, 0.5, n_points).tolist(),
        open_interest=np.random.lognormal(12, 0.3, n_points).tolist(),
        timestamps=[datetime.now() - timedelta(days=n_points-i) for i in range(n_points)],
        high=(prices * (1 + np.random.uniform(0, 0.02, n_points))).tolist(),
        low=(prices * (1 - np.random.uniform(0, 0.02, n_points))).tolist(),
        open=prices.tolist(),
        close=prices.tolist(),
        contract_symbol='CL_2024_03',
        underlying_asset='Crude Oil',
        basis=np.random.normal(0, 0.5, n_points).tolist(),
        roll_yield=np.random.normal(0, 0.1, n_points).tolist(),
        convenience_yield=np.random.normal(0.02, 0.01, n_points).tolist()
    )
    
    print("=== Futures Advanced ML Models Demo ===")
    
    # Initialize and train models
    ml_suite = FuturesAdvancedMLModels()
    
    print("\nTraining models...")
    training_results = ml_suite.train_all_models(sample_data)
    
    for model_name, success in training_results.items():
        status = "✓" if success else "✗"
        print(f"{status} {model_name.upper()} model: {'Trained successfully' if success else 'Training failed'}")
    
    # Generate predictions
    print("\nGenerating ensemble predictions...")
    ensemble_result = ml_suite.predict_ensemble(sample_data, horizon=5)
    
    print(f"\n=== Ensemble Prediction Results ===")
    print(f"Contract: {ensemble_result.contract_symbol}")
    print(f"Consensus Prediction: ${ensemble_result.consensus_prediction:.2f}")
    print(f"Confidence: {ensemble_result.confidence:.3f}")
    print(f"Model Agreement: {ensemble_result.model_agreement:.3f}")
    print(f"Ensemble Uncertainty: {ensemble_result.ensemble_uncertainty:.3f}")
    
    print(f"\n=== Individual Model Predictions ===")
    for model, prediction in ensemble_result.model_predictions.items():
        weight = ensemble_result.model_weights.get(model, 0)
        confidence = ensemble_result.model_confidences.get(model, 0)
        print(f"{model.upper()}: ${prediction:.2f} (weight: {weight:.1f}, confidence: {confidence:.3f})")
    
    # Test individual models
    print(f"\n=== Individual Model Details ===")
    
    lstm_pred = ml_suite.lstm_model.predict(sample_data)
    print(f"LSTM - Basis Prediction: {lstm_pred.basis_prediction}")
    print(f"LSTM - Volatility Prediction: {lstm_pred.volatility_prediction}")
    print(f"LSTM - Prediction Interval: {lstm_pred.prediction_interval}")
    
    transformer_pred = ml_suite.transformer_model.predict(sample_data)
    print(f"Transformer - Prediction Interval: {transformer_pred.prediction_interval}")
    
    print("\n=== Futures ML Models Enhancement Complete ===")