import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
import pickle
import json
from pathlib import Path

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# Deep Learning (if available)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam, RMSprop
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. Deep learning models will be disabled.")

# Time series analysis
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("Statsmodels not available. Time series models will be limited.")

import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Data class for model predictions"""
    model_name: str
    prediction_type: str  # 'price', 'direction', 'volatility', 'return'
    predicted_value: Union[float, int, List[float]]
    confidence: float
    probability_distribution: Optional[Dict[str, float]] = None
    feature_importance: Optional[Dict[str, float]] = None
    model_metrics: Optional[Dict[str, float]] = None
    timestamp: datetime = None

@dataclass
class ModelPerformance:
    """Data class for model performance metrics"""
    model_name: str
    model_type: str
    accuracy: Optional[float] = None
    mse: Optional[float] = None
    r2_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    win_rate: Optional[float] = None
    training_time: Optional[float] = None
    last_updated: datetime = None

class MLTradingModels:
    """Advanced machine learning models for trading predictions"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.performance_metrics = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'regressor': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                'classifier': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            },
            'gradient_boosting': {
                'regressor': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'classifier': None  # Will use XGBoost if available
            },
            'neural_network': {
                'regressor': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
                'classifier': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
            },
            'svm': {
                'regressor': SVR(kernel='rbf', C=1.0),
                'classifier': SVC(kernel='rbf', C=1.0, probability=True)
            },
            'linear': {
                'regressor': Ridge(alpha=1.0),
                'classifier': LogisticRegression(random_state=42, max_iter=1000)
            }
        }

    def prepare_features(self, df: pd.DataFrame, target_column: str = 'future_return') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for machine learning"""
        try:
            # Technical indicators features
            feature_columns = []
            
            # Price-based features
            price_features = ['open', 'high', 'low', 'close', 'volume']
            feature_columns.extend([col for col in price_features if col in df.columns])
            
            # Technical indicators
            technical_features = [
                'sma_20', 'sma_50', 'sma_200', 'ema_12', 'ema_26',
                'rsi', 'macd', 'macd_signal', 'macd_histogram',
                'bb_upper', 'bb_middle', 'bb_lower',
                'stoch_k', 'stoch_d', 'williams_r', 'atr', 'cci', 'mfi',
                'adx', 'aroon_up', 'aroon_down'
            ]
            feature_columns.extend([col for col in technical_features if col in df.columns])
            
            # Advanced features
            advanced_features = [
                'price_momentum_5', 'price_momentum_10', 'price_momentum_20',
                'volume_ratio', 'volatility_10', 'volatility_20', 'volatility_ratio',
                'support_distance', 'resistance_distance', 'trend_strength'
            ]
            feature_columns.extend([col for col in advanced_features if col in df.columns])
            
            # Lagged features
            lag_features = [col for col in df.columns if 'lag_' in col]
            feature_columns.extend(lag_features)
            
            # Rolling statistics
            rolling_features = [col for col in df.columns if any(stat in col for stat in ['_mean_', '_std_', '_max_', '_min_'])]
            feature_columns.extend(rolling_features)
            
            # Time-based features
            time_features = ['hour', 'day_of_week', 'month']
            feature_columns.extend([col for col in time_features if col in df.columns])
            
            # Remove duplicates and ensure target exists
            feature_columns = list(set(feature_columns))
            
            if target_column not in df.columns:
                logger.error(f"Target column '{target_column}' not found in dataframe")
                return pd.DataFrame(), pd.Series()
            
            # Create feature matrix and target vector
            X = df[feature_columns].copy()
            y = df[target_column].copy()
            
            # Remove rows with NaN values
            mask = ~(X.isnull().any(axis=1) | y.isnull())
            X = X[mask]
            y = y[mask]
            
            logger.info(f"Prepared {len(feature_columns)} features with {len(X)} samples")
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), pd.Series()

    def feature_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> Tuple[pd.DataFrame, Any]:
        """Select best features using statistical methods"""
        try:
            if len(X.columns) <= k:
                return X, None
            
            # Use SelectKBest with mutual information
            selector = SelectKBest(score_func=mutual_info_regression, k=k)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[selector.get_support()].tolist()
            X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
            logger.info(f"Selected {len(selected_features)} features from {len(X.columns)}")
            return X_selected_df, selector
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return X, None

    def train_ensemble_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'regressor') -> Dict[str, Any]:
        """Train ensemble of multiple models"""
        try:
            if len(X) < 100:
                logger.warning("Insufficient data for training ensemble model")
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            ensemble_results = {}
            
            for model_name, config in self.model_configs.items():
                if config[model_type] is None:
                    continue
                
                try:
                    model = config[model_type]
                    
                    # Train model
                    start_time = datetime.now()
                    model.fit(X_train_scaled, y_train)
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    # Make predictions
                    y_pred = model.predict(X_test_scaled)
                    
                    # Calculate metrics
                    if model_type == 'regressor':
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        ensemble_results[model_name] = {
                            'model': model,
                            'mse': mse,
                            'r2': r2,
                            'training_time': training_time,
                            'predictions': y_pred
                        }
                        
                        logger.info(f"Trained {model_name}: MSE={mse:.6f}, R2={r2:.4f}")
                        
                    else:  # classifier
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        ensemble_results[model_name] = {
                            'model': model,
                            'accuracy': accuracy,
                            'training_time': training_time,
                            'predictions': y_pred
                        }
                        
                        logger.info(f"Trained {model_name}: Accuracy={accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    continue
            
            if not ensemble_results:
                return {}
            
            # Create ensemble prediction (weighted average)
            if model_type == 'regressor':
                # Weight by R2 score
                weights = {name: max(0, result['r2']) for name, result in ensemble_results.items()}
                total_weight = sum(weights.values())
                
                if total_weight > 0:
                    ensemble_pred = np.zeros_like(y_test)
                    for name, result in ensemble_results.items():
                        weight = weights[name] / total_weight
                        ensemble_pred += weight * result['predictions']
                    
                    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
                    ensemble_r2 = r2_score(y_test, ensemble_pred)
                    
                    logger.info(f"Ensemble model: MSE={ensemble_mse:.6f}, R2={ensemble_r2:.4f}")
                    
                    return {
                        'ensemble_results': ensemble_results,
                        'ensemble_prediction': ensemble_pred,
                        'ensemble_metrics': {'mse': ensemble_mse, 'r2': ensemble_r2},
                        'scaler': scaler,
                        'feature_names': X.columns.tolist(),
                        'weights': weights
                    }
            
            return {
                'ensemble_results': ensemble_results,
                'scaler': scaler,
                'feature_names': X.columns.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {}

    def create_lstm_model(self, input_shape: Tuple[int, int], output_dim: int = 1) -> Optional[Any]:
        """Create LSTM model for time series prediction"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available, cannot create LSTM model")
            return None
        
        try:
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=input_shape),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(output_dim)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            return model
            
        except Exception as e:
            logger.error(f"Error creating LSTM model: {e}")
            return None

    def prepare_lstm_data(self, data: np.ndarray, lookback: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM training"""
        try:
            X, y = [], []
            
            for i in range(lookback, len(data)):
                X.append(data[i-lookback:i])
                y.append(data[i])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing LSTM data: {e}")
            return np.array([]), np.array([])

    def train_lstm_model(self, df: pd.DataFrame, target_column: str = 'close', lookback: int = 60) -> Dict[str, Any]:
        """Train LSTM model for time series prediction"""
        if not TENSORFLOW_AVAILABLE:
            return {}
        
        try:
            # Prepare data
            data = df[target_column].values.reshape(-1, 1)
            
            # Scale data
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Create sequences
            X, y = self.prepare_lstm_data(scaled_data, lookback)
            
            if len(X) < 100:
                logger.warning("Insufficient data for LSTM training")
                return {}
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Create and train model
            model = self.create_lstm_model((lookback, 1))
            if model is None:
                return {}
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Inverse transform predictions
            y_test_orig = scaler.inverse_transform(y_test)
            y_pred_orig = scaler.inverse_transform(y_pred)
            
            # Calculate metrics
            mse = mean_squared_error(y_test_orig, y_pred_orig)
            r2 = r2_score(y_test_orig, y_pred_orig)
            
            logger.info(f"LSTM model trained: MSE={mse:.6f}, R2={r2:.4f}")
            
            return {
                'model': model,
                'scaler': scaler,
                'lookback': lookback,
                'mse': mse,
                'r2': r2,
                'history': history.history,
                'target_column': target_column
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {}

    def predict_with_ensemble(self, X: pd.DataFrame, symbol: str, model_type: str = 'regressor') -> Optional[ModelPrediction]:
        """Make prediction using ensemble model"""
        try:
            if symbol not in self.models or model_type not in self.models[symbol]:
                logger.warning(f"No {model_type} model found for {symbol}")
                return None
            
            model_info = self.models[symbol][model_type]
            scaler = model_info['scaler']
            ensemble_results = model_info['ensemble_results']
            weights = model_info.get('weights', {})
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions with each model
            predictions = {}
            total_weight = 0
            
            for model_name, result in ensemble_results.items():
                try:
                    pred = result['model'].predict(X_scaled)
                    weight = weights.get(model_name, 1.0)
                    predictions[model_name] = {'prediction': pred[0], 'weight': weight}
                    total_weight += weight
                except Exception as e:
                    logger.error(f"Error predicting with {model_name}: {e}")
                    continue
            
            if not predictions:
                return None
            
            # Calculate weighted ensemble prediction
            ensemble_pred = sum(
                pred_info['prediction'] * pred_info['weight'] 
                for pred_info in predictions.values()
            ) / total_weight
            
            # Calculate confidence based on model agreement
            pred_values = [pred_info['prediction'] for pred_info in predictions.values()]
            confidence = 1.0 / (1.0 + np.std(pred_values)) if len(pred_values) > 1 else 0.5
            
            return ModelPrediction(
                model_name='ensemble',
                prediction_type=model_type,
                predicted_value=ensemble_pred,
                confidence=min(0.95, confidence),
                probability_distribution=None,
                feature_importance=None,
                model_metrics=model_info.get('ensemble_metrics', {}),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error making ensemble prediction: {e}")
            return None

    def predict_with_lstm(self, df: pd.DataFrame, symbol: str, steps_ahead: int = 1) -> Optional[ModelPrediction]:
        """Make prediction using LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        try:
            if symbol not in self.models or 'lstm' not in self.models[symbol]:
                logger.warning(f"No LSTM model found for {symbol}")
                return None
            
            model_info = self.models[symbol]['lstm']
            model = model_info['model']
            scaler = model_info['scaler']
            lookback = model_info['lookback']
            target_column = model_info['target_column']
            
            # Get recent data
            recent_data = df[target_column].tail(lookback).values.reshape(-1, 1)
            scaled_data = scaler.transform(recent_data)
            
            # Reshape for LSTM
            X = scaled_data.reshape(1, lookback, 1)
            
            # Make prediction
            predictions = []
            current_input = X.copy()
            
            for _ in range(steps_ahead):
                pred = model.predict(current_input, verbose=0)
                predictions.append(pred[0, 0])
                
                # Update input for next prediction
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0] = pred[0, 0]
            
            # Inverse transform predictions
            pred_array = np.array(predictions).reshape(-1, 1)
            pred_orig = scaler.inverse_transform(pred_array)
            
            # Calculate confidence based on model performance
            confidence = max(0.1, min(0.9, model_info.get('r2', 0.5)))
            
            return ModelPrediction(
                model_name='lstm',
                prediction_type='price',
                predicted_value=pred_orig.flatten().tolist() if steps_ahead > 1 else pred_orig[0, 0],
                confidence=confidence,
                probability_distribution=None,
                feature_importance=None,
                model_metrics={'mse': model_info.get('mse'), 'r2': model_info.get('r2')},
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error making LSTM prediction: {e}")
            return None

    def save_model(self, symbol: str, model_type: str) -> bool:
        """Save trained model to disk"""
        try:
            if symbol not in self.models or model_type not in self.models[symbol]:
                logger.warning(f"No {model_type} model found for {symbol}")
                return False
            
            model_path = self.model_dir / f"{symbol}_{model_type}.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[symbol][model_type], f)
            
            logger.info(f"Saved {model_type} model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False

    def load_model(self, symbol: str, model_type: str) -> bool:
        """Load trained model from disk"""
        try:
            model_path = self.model_dir / f"{symbol}_{model_type}.pkl"
            
            if not model_path.exists():
                logger.warning(f"Model file not found: {model_path}")
                return False
            
            with open(model_path, 'rb') as f:
                model_info = pickle.load(f)
            
            if symbol not in self.models:
                self.models[symbol] = {}
            
            self.models[symbol][model_type] = model_info
            logger.info(f"Loaded {model_type} model for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    async def train_comprehensive_model(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Train comprehensive model suite for a symbol"""
        try:
            results = {}
            
            # Prepare features for regression (price prediction)
            X_reg, y_reg = self.prepare_features(df, 'future_return_1')
            
            if len(X_reg) > 0:
                # Feature selection
                X_reg_selected, feature_selector = self.feature_selection(X_reg, y_reg)
                
                # Train ensemble regressor
                ensemble_reg = self.train_ensemble_model(X_reg_selected, y_reg, 'regressor')
                
                if ensemble_reg:
                    if symbol not in self.models:
                        self.models[symbol] = {}
                    
                    self.models[symbol]['regressor'] = ensemble_reg
                    self.feature_selectors[symbol] = feature_selector
                    results['regressor'] = ensemble_reg['ensemble_metrics']
            
            # Prepare features for classification (direction prediction)
            if 'future_direction' in df.columns:
                X_clf, y_clf = self.prepare_features(df, 'future_direction')
                
                if len(X_clf) > 0:
                    # Feature selection
                    X_clf_selected, _ = self.feature_selection(X_clf, y_clf)
                    
                    # Train ensemble classifier
                    ensemble_clf = self.train_ensemble_model(X_clf_selected, y_clf, 'classifier')
                    
                    if ensemble_clf:
                        if symbol not in self.models:
                            self.models[symbol] = {}
                        
                        self.models[symbol]['classifier'] = ensemble_clf
                        results['classifier'] = {'accuracy': 'calculated_in_ensemble'}
            
            # Train LSTM model
            if TENSORFLOW_AVAILABLE and len(df) > 200:
                lstm_result = self.train_lstm_model(df, 'close')
                
                if lstm_result:
                    if symbol not in self.models:
                        self.models[symbol] = {}
                    
                    self.models[symbol]['lstm'] = lstm_result
                    results['lstm'] = {'mse': lstm_result['mse'], 'r2': lstm_result['r2']}
            
            # Save models
            for model_type in self.models.get(symbol, {}):
                self.save_model(symbol, model_type)
            
            logger.info(f"Trained comprehensive model suite for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error training comprehensive model for {symbol}: {e}")
            return {}

# Example usage
async def main():
    """Example usage of MLTradingModels"""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data with features
    price = 100
    data = []
    
    for i, date in enumerate(dates):
        price += np.random.normal(0, 2)
        volume = np.random.randint(1000000, 10000000)
        
        data.append({
            'timestamp': date,
            'open': price * (1 + np.random.uniform(-0.02, 0.02)),
            'high': price * (1 + np.random.uniform(0, 0.05)),
            'low': price * (1 - np.random.uniform(0, 0.05)),
            'close': price,
            'volume': volume,
            'future_return_1': np.random.normal(0, 0.02),
            'future_direction': np.random.choice([0, 1]),
            'rsi': np.random.uniform(20, 80),
            'macd': np.random.normal(0, 1),
            'sma_20': price * (1 + np.random.uniform(-0.05, 0.05))
        })
    
    df = pd.DataFrame(data)
    
    # Initialize ML models
    ml_models = MLTradingModels()
    
    # Train comprehensive model
    results = await ml_models.train_comprehensive_model(df, 'AAPL')
    
    print(f"Training results: {results}")
    
    # Make predictions
    latest_features = df[['rsi', 'macd', 'sma_20']].tail(1)
    
    reg_pred = ml_models.predict_with_ensemble(latest_features, 'AAPL', 'regressor')
    if reg_pred:
        print(f"Price prediction: {reg_pred.predicted_value:.4f} (confidence: {reg_pred.confidence:.2f})")
    
    lstm_pred = ml_models.predict_with_lstm(df, 'AAPL')
    if lstm_pred:
        print(f"LSTM prediction: {lstm_pred.predicted_value:.4f} (confidence: {lstm_pred.confidence:.2f})")

if __name__ == "__main__":
    asyncio.run(main())