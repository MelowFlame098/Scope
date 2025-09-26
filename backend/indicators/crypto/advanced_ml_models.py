from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("Warning: ML libraries not available. Install tensorflow and scikit-learn for full functionality.")

logger = logging.getLogger(__name__)

class CryptoModelType(Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    MLP = "mlp"
    ENSEMBLE = "ensemble"

@dataclass
class CryptoData:
    """Crypto-specific data structure"""
    price: pd.Series
    volume: pd.Series
    market_cap: Optional[pd.Series] = None
    hash_rate: Optional[pd.Series] = None
    difficulty: Optional[pd.Series] = None
    active_addresses: Optional[pd.Series] = None
    transaction_count: Optional[pd.Series] = None
    exchange_inflow: Optional[pd.Series] = None
    exchange_outflow: Optional[pd.Series] = None
    fear_greed_index: Optional[pd.Series] = None
    social_sentiment: Optional[pd.Series] = None
    network_value: Optional[pd.Series] = None
    realized_value: Optional[pd.Series] = None
    mvrv_ratio: Optional[pd.Series] = None
    nvt_ratio: Optional[pd.Series] = None
    puell_multiple: Optional[pd.Series] = None
    stock_to_flow: Optional[pd.Series] = None

@dataclass
class CryptoMLPredictionResult:
    """ML prediction result for crypto"""
    model_type: CryptoModelType
    prediction: float
    confidence: float
    prediction_interval: Tuple[float, float]
    feature_importance: Dict[str, float]
    model_metrics: Dict[str, float]
    timestamp: datetime
    horizon_days: int
    market_regime: str
    risk_assessment: str

@dataclass
class CryptoEnsembleResult:
    """Ensemble prediction result for crypto"""
    consensus_prediction: float
    individual_predictions: Dict[str, float]
    model_weights: Dict[str, float]
    confidence_score: float
    prediction_range: Tuple[float, float]
    market_signal: str
    risk_level: str
    timestamp: datetime

class CryptoLSTMModel:
    """LSTM model specifically designed for cryptocurrency prediction"""
    
    def __init__(self, sequence_length: int = 60, features: int = 10):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def _build_model(self) -> Sequential:
        """Build LSTM architecture optimized for crypto data"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.features)),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y = [], []
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i, 0])  # Predict price (first column)
        return np.array(X), np.array(y)
    
    def fit(self, crypto_data: CryptoData) -> Dict[str, float]:
        """Train the LSTM model"""
        if not ML_AVAILABLE:
            raise ImportError("TensorFlow not available")
            
        # Prepare feature matrix
        features = [crypto_data.price]
        if crypto_data.volume is not None:
            features.append(crypto_data.volume)
        if crypto_data.hash_rate is not None:
            features.append(crypto_data.hash_rate)
        if crypto_data.mvrv_ratio is not None:
            features.append(crypto_data.mvrv_ratio)
        if crypto_data.nvt_ratio is not None:
            features.append(crypto_data.nvt_ratio)
        
        # Combine features
        feature_matrix = pd.concat(features, axis=1).fillna(method='ffill').dropna()
        
        # Scale data
        scaled_data = self.scaler.fit_transform(feature_matrix.values)
        
        # Prepare sequences
        X, y = self.prepare_sequences(scaled_data)
        
        if len(X) < 100:
            raise ValueError("Insufficient data for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and train model
        self.model = self._build_model()
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Calculate metrics
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.is_fitted = True
        
        return {
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'final_loss': history.history['loss'][-1]
        }
    
    def predict(self, crypto_data: CryptoData, horizon_days: int = 1) -> CryptoMLPredictionResult:
        """Make prediction using trained LSTM model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare recent data
        features = [crypto_data.price]
        if crypto_data.volume is not None:
            features.append(crypto_data.volume)
        if crypto_data.hash_rate is not None:
            features.append(crypto_data.hash_rate)
        if crypto_data.mvrv_ratio is not None:
            features.append(crypto_data.mvrv_ratio)
        if crypto_data.nvt_ratio is not None:
            features.append(crypto_data.nvt_ratio)
        
        feature_matrix = pd.concat(features, axis=1).fillna(method='ffill')
        recent_data = feature_matrix.tail(self.sequence_length).values
        
        # Scale and reshape
        scaled_recent = self.scaler.transform(recent_data)
        X_pred = scaled_recent.reshape(1, self.sequence_length, -1)
        
        # Make prediction
        prediction_scaled = self.model.predict(X_pred)[0][0]
        
        # Inverse transform prediction
        dummy_array = np.zeros((1, scaled_recent.shape[1]))
        dummy_array[0, 0] = prediction_scaled
        prediction = self.scaler.inverse_transform(dummy_array)[0, 0]
        
        # Calculate confidence and intervals
        current_price = crypto_data.price.iloc[-1]
        volatility = crypto_data.price.pct_change().std() * np.sqrt(252)
        
        confidence = max(0.1, 1.0 - abs(prediction - current_price) / current_price)
        
        prediction_interval = (
            prediction * (1 - 1.96 * volatility / np.sqrt(252)),
            prediction * (1 + 1.96 * volatility / np.sqrt(252))
        )
        
        return CryptoMLPredictionResult(
            model_type=CryptoModelType.LSTM,
            prediction=prediction,
            confidence=confidence,
            prediction_interval=prediction_interval,
            feature_importance={'lstm_features': 1.0},
            model_metrics={'confidence': confidence},
            timestamp=datetime.now(),
            horizon_days=horizon_days,
            market_regime='normal',
            risk_assessment='medium'
        )

class CryptoTransformerModel:
    """Transformer model for crypto prediction with attention mechanism"""
    
    def __init__(self, sequence_length: int = 60, d_model: int = 64, num_heads: int = 8):
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
    
    def _build_transformer(self, input_shape: Tuple[int, int]) -> Model:
        """Build transformer architecture"""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.d_model
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = LayerNormalization()(inputs + attention_output)
        
        # Feed forward
        ffn_output = Dense(self.d_model * 2, activation='relu')(attention_output)
        ffn_output = Dense(self.d_model)(ffn_output)
        
        # Add & Norm
        ffn_output = LayerNormalization()(attention_output + ffn_output)
        
        # Global average pooling and output
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        outputs = Dense(1)(pooled)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, crypto_data: CryptoData) -> Dict[str, float]:
        """Train transformer model"""
        if not ML_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        # Similar to LSTM preparation but for transformer
        features = [crypto_data.price, crypto_data.volume]
        if crypto_data.hash_rate is not None:
            features.append(crypto_data.hash_rate)
        
        feature_matrix = pd.concat(features, axis=1).fillna(method='ffill').dropna()
        scaled_data = self.scaler.fit_transform(feature_matrix.values)
        
        # Prepare sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build and train
        self.model = self._build_transformer((self.sequence_length, X.shape[2]))
        
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        self.is_fitted = True
        
        y_pred = self.model.predict(X_test)
        return {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }

class CryptoAdvancedMLModels:
    """Advanced ML models suite for cryptocurrency analysis"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_fitted = {}
        
    def initialize_models(self):
        """Initialize all ML models"""
        if ML_AVAILABLE:
            self.models = {
                'lstm': CryptoLSTMModel(),
                'transformer': CryptoTransformerModel(),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
            }
        
    def analyze(self, crypto_data: CryptoData) -> CryptoEnsembleResult:
        """Comprehensive crypto analysis using all models"""
        if not self.models:
            self.initialize_models()
        
        predictions = {}
        confidences = {}
        
        # LSTM prediction
        try:
            if 'lstm' in self.models and not self.is_fitted.get('lstm', False):
                self.models['lstm'].fit(crypto_data)
                self.is_fitted['lstm'] = True
            
            if self.is_fitted.get('lstm', False):
                lstm_result = self.models['lstm'].predict(crypto_data)
                predictions['lstm'] = lstm_result.prediction
                confidences['lstm'] = lstm_result.confidence
        except Exception as e:
            logger.warning(f"LSTM prediction failed: {e}")
            predictions['lstm'] = crypto_data.price.iloc[-1]
            confidences['lstm'] = 0.1
        
        # Traditional ML models
        try:
            # Prepare features for traditional models
            features = self._prepare_features(crypto_data)
            
            for model_name in ['random_forest', 'gradient_boosting', 'mlp']:
                if model_name in self.models:
                    if not self.is_fitted.get(model_name, False):
                        # Simple training for demo
                        X = features[:-1].values
                        y = crypto_data.price[1:].values
                        
                        if len(X) > 50:
                            self.models[model_name].fit(X, y)
                            self.is_fitted[model_name] = True
                    
                    if self.is_fitted.get(model_name, False):
                        pred = self.models[model_name].predict([features.iloc[-1].values])[0]
                        predictions[model_name] = pred
                        confidences[model_name] = 0.7
        except Exception as e:
            logger.warning(f"Traditional ML prediction failed: {e}")
        
        # Ensemble prediction
        if predictions:
            weights = self._calculate_weights(confidences)
            consensus = sum(pred * weights.get(model, 0) for model, pred in predictions.items())
            
            # Market signal
            current_price = crypto_data.price.iloc[-1]
            price_change = (consensus - current_price) / current_price
            
            if price_change > 0.05:
                signal = "STRONG_BUY"
            elif price_change > 0.02:
                signal = "BUY"
            elif price_change < -0.05:
                signal = "STRONG_SELL"
            elif price_change < -0.02:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            return CryptoEnsembleResult(
                consensus_prediction=consensus,
                individual_predictions=predictions,
                model_weights=weights,
                confidence_score=np.mean(list(confidences.values())),
                prediction_range=(min(predictions.values()), max(predictions.values())),
                market_signal=signal,
                risk_level="medium",
                timestamp=datetime.now()
            )
        else:
            # Fallback
            current_price = crypto_data.price.iloc[-1]
            return CryptoEnsembleResult(
                consensus_prediction=current_price,
                individual_predictions={'fallback': current_price},
                model_weights={'fallback': 1.0},
                confidence_score=0.1,
                prediction_range=(current_price * 0.95, current_price * 1.05),
                market_signal="HOLD",
                risk_level="high",
                timestamp=datetime.now()
            )
    
    def _prepare_features(self, crypto_data: CryptoData) -> pd.DataFrame:
        """Prepare features for traditional ML models"""
        features = pd.DataFrame(index=crypto_data.price.index)
        
        # Price features
        features['price'] = crypto_data.price
        features['price_sma_20'] = crypto_data.price.rolling(20).mean()
        features['price_std_20'] = crypto_data.price.rolling(20).std()
        features['price_rsi'] = self._calculate_rsi(crypto_data.price)
        
        # Volume features
        if crypto_data.volume is not None:
            features['volume'] = crypto_data.volume
            features['volume_sma_20'] = crypto_data.volume.rolling(20).mean()
        
        # Crypto-specific features
        if crypto_data.mvrv_ratio is not None:
            features['mvrv_ratio'] = crypto_data.mvrv_ratio
        
        if crypto_data.nvt_ratio is not None:
            features['nvt_ratio'] = crypto_data.nvt_ratio
        
        return features.fillna(method='ffill').dropna()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_weights(self, confidences: Dict[str, float]) -> Dict[str, float]:
        """Calculate ensemble weights based on confidence scores"""
        total_confidence = sum(confidences.values())
        if total_confidence == 0:
            return {model: 1.0/len(confidences) for model in confidences}
        return {model: conf/total_confidence for model, conf in confidences.items()}

# Example usage
if __name__ == "__main__":
    # Create sample crypto data
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate Bitcoin-like price data
    price_data = pd.Series(
        np.cumsum(np.random.randn(len(dates)) * 0.02) + 10000,
        index=dates
    )
    
    volume_data = pd.Series(
        np.random.exponential(1000000, len(dates)),
        index=dates
    )
    
    crypto_data = CryptoData(
        price=price_data,
        volume=volume_data
    )
    
    # Initialize and run analysis
    crypto_ml = CryptoAdvancedMLModels()
    result = crypto_ml.analyze(crypto_data)
    
    print("\n=== Crypto Advanced ML Analysis ===")
    print(f"Consensus Prediction: ${result.consensus_prediction:.2f}")
    print(f"Market Signal: {result.market_signal}")
    print(f"Confidence Score: {result.confidence_score:.2f}")
    print(f"Risk Level: {result.risk_level}")
    
    print("\nIndividual Model Predictions:")
    for model, prediction in result.individual_predictions.items():
        weight = result.model_weights.get(model, 0)
        print(f"  {model}: ${prediction:.2f} (weight: {weight:.2f})")
    
    print(f"\nPrediction Range: ${result.prediction_range[0]:.2f} - ${result.prediction_range[1]:.2f}")
    print(f"Analysis Timestamp: {result.timestamp}")