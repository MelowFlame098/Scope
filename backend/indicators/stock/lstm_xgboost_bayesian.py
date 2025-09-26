import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# Deep Learning imports (with fallbacks)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. LSTM and Bayesian NN models will use simplified implementations.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Using alternative implementations.")

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelPrediction:
    """Single model prediction with uncertainty."""
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    timestamp: datetime
    model_name: str
    
@dataclass
class LSTMResults:
    """Results from LSTM model."""
    predictions: np.ndarray
    actual_values: np.ndarray
    train_loss: List[float]
    val_loss: List[float]
    mse: float
    mae: float
    r2: float
    directional_accuracy: float
    model_architecture: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]] = None
    uncertainty_estimates: Optional[np.ndarray] = None
    
@dataclass
class XGBoostResults:
    """Results from XGBoost model."""
    predictions: np.ndarray
    actual_values: np.ndarray
    feature_importance: Dict[str, float]
    mse: float
    mae: float
    r2: float
    directional_accuracy: float
    model_parameters: Dict[str, Any]
    cross_validation_scores: List[float]
    prediction_intervals: Optional[Dict[str, np.ndarray]] = None
    
@dataclass
class BayesianNNResults:
    """Results from Bayesian Neural Network."""
    mean_predictions: np.ndarray
    uncertainty_estimates: np.ndarray
    actual_values: np.ndarray
    epistemic_uncertainty: np.ndarray
    aleatoric_uncertainty: np.ndarray
    mse: float
    mae: float
    r2: float
    calibration_score: float
    model_architecture: Dict[str, Any]
    posterior_samples: Optional[List[np.ndarray]] = None
    
@dataclass
class EnsembleResults:
    """Results from ensemble of models."""
    ensemble_predictions: np.ndarray
    individual_predictions: Dict[str, np.ndarray]
    model_weights: Dict[str, float]
    uncertainty_estimates: np.ndarray
    actual_values: np.ndarray
    mse: float
    mae: float
    r2: float
    directional_accuracy: float
    model_performance: Dict[str, Dict[str, float]]
    
@dataclass
class MLAnalysisResults:
    """Comprehensive ML analysis results."""
    lstm_results: Optional[LSTMResults]
    xgboost_results: Optional[XGBoostResults]
    bayesian_results: Optional[BayesianNNResults]
    ensemble_results: Optional[EnsembleResults]
    best_model: str
    model_comparison: Dict[str, Dict[str, float]]
    feature_analysis: Dict[str, Any]
    insights: Dict[str, Any]

class FeatureEngineer:
    """Feature engineering for time series data."""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def create_features(self, data: pd.DataFrame, price_column: str = 'close',
                       lookback_periods: List[int] = None) -> pd.DataFrame:
        """Create comprehensive features for ML models."""
        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 50]
            
        df = data.copy()
        
        # Price-based features
        df['returns'] = df[price_column].pct_change()
        df['log_returns'] = np.log(df[price_column] / df[price_column].shift(1))
        
        # Moving averages
        for period in lookback_periods:
            df[f'ma_{period}'] = df[price_column].rolling(window=period).mean()
            df[f'ma_ratio_{period}'] = df[price_column] / df[f'ma_{period}']
            
        # Volatility features
        for period in lookback_periods:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            df[f'realized_vol_{period}'] = df['log_returns'].rolling(window=period).std() * np.sqrt(252)
            
        # Technical indicators
        df['rsi'] = self._calculate_rsi(df[price_column])
        df['macd'], df['macd_signal'] = self._calculate_macd(df[price_column])
        df['bb_upper'], df['bb_lower'], df['bb_width'] = self._calculate_bollinger_bands(df[price_column])
        
        # Momentum features
        for period in [1, 3, 5, 10]:
            df[f'momentum_{period}'] = df[price_column] / df[price_column].shift(period) - 1
            
        # Lag features
        for lag in range(1, 6):
            df[f'price_lag_{lag}'] = df[price_column].shift(lag)
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
            
        # Time-based features
        if hasattr(df.index, 'dayofweek'):
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            
        # Volume features (if available)
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            df['price_volume'] = df[price_column] * df['volume']
            
        # High-low features (if available)
        if 'high' in df.columns and 'low' in df.columns:
            df['hl_ratio'] = df['high'] / df['low']
            df['price_position'] = (df[price_column] - df['low']) / (df['high'] - df['low'])
            
        # Remove rows with NaN values
        df = df.dropna()
        
        # Store feature names
        self.feature_names = [col for col in df.columns if col != price_column]
        
        return df
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line
        
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        width = (upper - lower) / ma
        return upper, lower, width
        
    def prepare_sequences(self, data: pd.DataFrame, target_column: str,
                         sequence_length: int = 60, forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training."""
        feature_columns = [col for col in data.columns if col != target_column]
        
        X, y = [], []
        
        for i in range(sequence_length, len(data) - forecast_horizon + 1):
            # Features sequence
            X.append(data[feature_columns].iloc[i-sequence_length:i].values)
            # Target (future price)
            y.append(data[target_column].iloc[i + forecast_horizon - 1])
            
        return np.array(X), np.array(y)
        
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray,
                      scaler_type: str = 'standard') -> Tuple[np.ndarray, np.ndarray]:
        """Scale features for training."""
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
            
        # Reshape for scaling if needed
        original_shape_train = X_train.shape
        original_shape_test = X_test.shape
        
        if len(X_train.shape) == 3:  # LSTM sequences
            X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
            X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        else:
            X_train_reshaped = X_train
            X_test_reshaped = X_test
            
        # Fit and transform
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)
        
        # Reshape back
        if len(original_shape_train) == 3:
            X_train_scaled = X_train_scaled.reshape(original_shape_train)
            X_test_scaled = X_test_scaled.reshape(original_shape_test)
            
        self.scalers[scaler_type] = scaler
        return X_train_scaled, X_test_scaled

class LSTMModel:
    """LSTM model for time series prediction."""
    
    def __init__(self, input_shape: Tuple[int, int], architecture: Dict[str, Any] = None):
        self.input_shape = input_shape
        self.architecture = architecture or {
            'lstm_units': [50, 50],
            'dropout_rate': 0.2,
            'dense_units': [25],
            'activation': 'relu',
            'output_activation': 'linear'
        }
        self.model = None
        self.history = None
        
    def build_model(self) -> None:
        """Build LSTM model architecture."""
        if TF_AVAILABLE:
            self._build_tensorflow_model()
        elif TORCH_AVAILABLE:
            self._build_pytorch_model()
        else:
            raise ImportError("Neither TensorFlow nor PyTorch available for LSTM implementation")
            
    def _build_tensorflow_model(self) -> None:
        """Build model using TensorFlow/Keras."""
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            self.architecture['lstm_units'][0],
            return_sequences=len(self.architecture['lstm_units']) > 1,
            input_shape=self.input_shape
        ))
        model.add(Dropout(self.architecture['dropout_rate']))
        
        # Additional LSTM layers
        for i, units in enumerate(self.architecture['lstm_units'][1:], 1):
            return_sequences = i < len(self.architecture['lstm_units']) - 1
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(self.architecture['dropout_rate']))
            
        # Dense layers
        for units in self.architecture['dense_units']:
            model.add(Dense(units, activation=self.architecture['activation']))
            model.add(Dropout(self.architecture['dropout_rate']))
            
        # Output layer
        model.add(Dense(1, activation=self.architecture['output_activation']))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        
    def _build_pytorch_model(self) -> None:
        """Build model using PyTorch."""
        class LSTMNet(nn.Module):
            def __init__(self, input_size, lstm_units, dense_units, dropout_rate):
                super(LSTMNet, self).__init__()
                
                # LSTM layers
                self.lstm_layers = nn.ModuleList()
                prev_size = input_size
                
                for units in lstm_units:
                    self.lstm_layers.append(nn.LSTM(prev_size, units, batch_first=True))
                    prev_size = units
                    
                # Dense layers
                self.dense_layers = nn.ModuleList()
                for units in dense_units:
                    self.dense_layers.append(nn.Linear(prev_size, units))
                    prev_size = units
                    
                self.output_layer = nn.Linear(prev_size, 1)
                self.dropout = nn.Dropout(dropout_rate)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                # LSTM layers
                for lstm in self.lstm_layers:
                    x, _ = lstm(x)
                    x = self.dropout(x)
                    
                # Take last output
                x = x[:, -1, :]
                
                # Dense layers
                for dense in self.dense_layers:
                    x = self.relu(dense(x))
                    x = self.dropout(x)
                    
                return self.output_layer(x)
                
        self.model = LSTMNet(
            self.input_shape[1],
            self.architecture['lstm_units'],
            self.architecture['dense_units'],
            self.architecture['dropout_rate']
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 32) -> LSTMResults:
        """Train the LSTM model."""
        if TF_AVAILABLE:
            return self._train_tensorflow(X_train, y_train, X_val, y_val, epochs, batch_size)
        elif TORCH_AVAILABLE:
            return self._train_pytorch(X_train, y_train, X_val, y_val, epochs, batch_size)
        else:
            raise ImportError("No deep learning framework available")
            
    def _train_tensorflow(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         epochs: int, batch_size: int) -> LSTMResults:
        """Train using TensorFlow."""
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Predictions
        train_pred = self.model.predict(X_train).flatten()
        val_pred = self.model.predict(X_val).flatten()
        
        # Metrics
        mse = mean_squared_error(y_val, val_pred)
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        # Directional accuracy
        val_direction = np.sign(np.diff(y_val))
        pred_direction = np.sign(np.diff(val_pred))
        directional_accuracy = np.mean(val_direction == pred_direction)
        
        return LSTMResults(
            predictions=val_pred,
            actual_values=y_val,
            train_loss=history.history['loss'],
            val_loss=history.history['val_loss'],
            mse=mse,
            mae=mae,
            r2=r2,
            directional_accuracy=directional_accuracy,
            model_architecture=self.architecture
        )
        
    def _train_pytorch(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      epochs: int, batch_size: int) -> LSTMResults:
        """Train using PyTorch."""
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor).item()
                
            train_losses.append(train_loss / len(train_loader))
            val_losses.append(val_loss)
            
        # Final predictions
        self.model.eval()
        with torch.no_grad():
            val_pred = self.model(X_val_tensor).numpy().flatten()
            
        # Metrics
        mse = mean_squared_error(y_val, val_pred)
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        # Directional accuracy
        val_direction = np.sign(np.diff(y_val))
        pred_direction = np.sign(np.diff(val_pred))
        directional_accuracy = np.mean(val_direction == pred_direction)
        
        return LSTMResults(
            predictions=val_pred,
            actual_values=y_val,
            train_loss=train_losses,
            val_loss=val_losses,
            mse=mse,
            mae=mae,
            r2=r2,
            directional_accuracy=directional_accuracy,
            model_architecture=self.architecture
        )
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if TF_AVAILABLE and hasattr(self.model, 'predict'):
            return self.model.predict(X).flatten()
        elif TORCH_AVAILABLE:
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                return self.model(X_tensor).numpy().flatten()
        else:
            raise RuntimeError("Model not trained or framework not available")

class XGBoostModel:
    """XGBoost model for time series prediction."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = None
        self.feature_names = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             feature_names: List[str] = None) -> XGBoostResults:
        """Train XGBoost model."""
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
        
        # Training parameters
        params = self.params.copy()
        params['eval_metric'] = 'rmse'
        
        # Train model
        evals = [(dtrain, 'train'), (dval, 'eval')]
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=evals,
            early_stopping_rounds=20,
            verbose_eval=False
        )
        
        # Predictions
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        
        # Feature importance
        importance_dict = self.model.get_score(importance_type='weight')
        feature_importance = {name: importance_dict.get(name, 0) for name in self.feature_names}
        
        # Metrics
        mse = mean_squared_error(y_val, val_pred)
        mae = mean_absolute_error(y_val, val_pred)
        r2 = r2_score(y_val, val_pred)
        
        # Directional accuracy
        val_direction = np.sign(np.diff(y_val))
        pred_direction = np.sign(np.diff(val_pred))
        directional_accuracy = np.mean(val_direction == pred_direction)
        
        # Cross-validation
        cv_scores = self._cross_validate(X_train, y_train)
        
        # Prediction intervals using quantile regression
        prediction_intervals = self._calculate_prediction_intervals(X_val, y_val)
        
        return XGBoostResults(
            predictions=val_pred,
            actual_values=y_val,
            feature_importance=feature_importance,
            mse=mse,
            mae=mae,
            r2=r2,
            directional_accuracy=directional_accuracy,
            model_parameters=self.params,
            cross_validation_scores=cv_scores,
            prediction_intervals=prediction_intervals
        )
        
    def _cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> List[float]:
        """Perform time series cross-validation."""
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            # Train model
            dtrain = xgb.DMatrix(X_train_cv, label=y_train_cv)
            dval = xgb.DMatrix(X_val_cv, label=y_val_cv)
            
            model = xgb.train(
                self.params,
                dtrain,
                num_boost_round=self.params.get('n_estimators', 100),
                verbose_eval=False
            )
            
            # Predict and score
            pred = model.predict(dval)
            score = r2_score(y_val_cv, pred)
            scores.append(score)
            
        return scores
        
    def _calculate_prediction_intervals(self, X_val: np.ndarray, y_val: np.ndarray,
                                      quantiles: List[float] = None) -> Dict[str, np.ndarray]:
        """Calculate prediction intervals using quantile regression."""
        if quantiles is None:
            quantiles = [0.05, 0.25, 0.75, 0.95]
            
        intervals = {}
        
        for quantile in quantiles:
            # Train quantile regression model
            params = self.params.copy()
            params['objective'] = f'reg:quantileerror'
            params['quantile_alpha'] = quantile
            
            dtrain = xgb.DMatrix(X_val, label=y_val)
            
            quantile_model = xgb.train(
                params,
                dtrain,
                num_boost_round=50,
                verbose_eval=False
            )
            
            intervals[f'quantile_{quantile}'] = quantile_model.predict(dtrain)
            
        return intervals
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if self.model is None:
            raise RuntimeError("Model not trained")
            
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance."""
        if self.model is None:
            raise RuntimeError("Model not trained")
            
        importance_dict = self.model.get_score(importance_type='weight')
        return {name: importance_dict.get(name, 0) for name in self.feature_names}

class BayesianNeuralNetwork:
    """Bayesian Neural Network for uncertainty quantification."""
    
    def __init__(self, input_dim: int, architecture: Dict[str, Any] = None):
        self.input_dim = input_dim
        self.architecture = architecture or {
            'hidden_units': [50, 25],
            'activation': 'relu',
            'n_samples': 100,
            'prior_std': 1.0
        }
        self.models = []
        self.posterior_samples = []
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             n_epochs: int = 100) -> BayesianNNResults:
        """Train Bayesian Neural Network using Monte Carlo Dropout."""
        # Build ensemble of models with dropout
        n_models = self.architecture['n_samples']
        
        predictions = []
        epistemic_uncertainties = []
        
        for i in range(n_models):
            # Create model with dropout
            model = self._build_dropout_model()
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=n_epochs,
                batch_size=32,
                verbose=0
            )
            
            self.models.append(model)
            
            # MC Dropout predictions
            mc_predictions = []
            for _ in range(50):  # MC samples
                pred = model.predict(X_val, training=True)  # Keep dropout active
                mc_predictions.append(pred.flatten())
                
            mc_predictions = np.array(mc_predictions)
            mean_pred = np.mean(mc_predictions, axis=0)
            epistemic_unc = np.std(mc_predictions, axis=0)
            
            predictions.append(mean_pred)
            epistemic_uncertainties.append(epistemic_unc)
            
        # Ensemble predictions
        predictions = np.array(predictions)
        mean_predictions = np.mean(predictions, axis=0)
        epistemic_uncertainty = np.mean(epistemic_uncertainties, axis=0)
        
        # Aleatoric uncertainty (data noise)
        residuals = y_val - mean_predictions
        aleatoric_uncertainty = np.std(residuals) * np.ones_like(mean_predictions)
        
        # Total uncertainty
        total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)
        
        # Metrics
        mse = mean_squared_error(y_val, mean_predictions)
        mae = mean_absolute_error(y_val, mean_predictions)
        r2 = r2_score(y_val, mean_predictions)
        
        # Calibration score
        calibration_score = self._calculate_calibration_score(
            y_val, mean_predictions, total_uncertainty
        )
        
        return BayesianNNResults(
            mean_predictions=mean_predictions,
            uncertainty_estimates=total_uncertainty,
            actual_values=y_val,
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=aleatoric_uncertainty,
            mse=mse,
            mae=mae,
            r2=r2,
            calibration_score=calibration_score,
            model_architecture=self.architecture,
            posterior_samples=predictions.tolist()
        )
        
    def _build_dropout_model(self) -> Any:
        """Build neural network with dropout for uncertainty estimation."""
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow required for Bayesian NN")
            
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.architecture['hidden_units'][0],
            activation=self.architecture['activation'],
            input_dim=self.input_dim
        ))
        model.add(Dropout(0.3))
        
        # Hidden layers
        for units in self.architecture['hidden_units'][1:]:
            model.add(Dense(units, activation=self.architecture['activation']))
            model.add(Dropout(0.3))
            
        # Output layer
        model.add(Dense(1, activation='linear'))
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def _calculate_calibration_score(self, y_true: np.ndarray, y_pred: np.ndarray,
                                   uncertainty: np.ndarray) -> float:
        """Calculate calibration score for uncertainty estimates."""
        # Check if predictions fall within confidence intervals
        confidence_levels = [0.5, 0.68, 0.95, 0.99]
        calibration_errors = []
        
        for conf_level in confidence_levels:
            z_score = stats.norm.ppf((1 + conf_level) / 2)
            
            # Confidence interval
            lower = y_pred - z_score * uncertainty
            upper = y_pred + z_score * uncertainty
            
            # Fraction of points within interval
            within_interval = np.mean((y_true >= lower) & (y_true <= upper))
            
            # Calibration error
            calibration_error = abs(within_interval - conf_level)
            calibration_errors.append(calibration_error)
            
        return np.mean(calibration_errors)
        
    def predict(self, X: np.ndarray, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with uncertainty estimates."""
        if not self.models:
            raise RuntimeError("Model not trained")
            
        all_predictions = []
        
        for model in self.models:
            mc_predictions = []
            for _ in range(n_samples):
                pred = model.predict(X, training=True)
                mc_predictions.append(pred.flatten())
                
            mc_predictions = np.array(mc_predictions)
            all_predictions.append(mc_predictions)
            
        # Combine all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        mean_pred = np.mean(all_predictions, axis=0)
        uncertainty = np.std(all_predictions, axis=0)
        
        return mean_pred, uncertainty

class EnsembleModel:
    """Ensemble of different ML models."""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scalers = {}
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add model to ensemble."""
        self.models[name] = model
        self.weights[name] = weight
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             feature_names: List[str] = None) -> EnsembleResults:
        """Train ensemble of models."""
        individual_predictions = {}
        model_performance = {}
        
        # Train each model
        for name, model in self.models.items():
            try:
                if isinstance(model, XGBoostModel):
                    results = model.train(X_train, y_train, X_val, y_val, feature_names)
                    individual_predictions[name] = results.predictions
                    model_performance[name] = {
                        'mse': results.mse,
                        'mae': results.mae,
                        'r2': results.r2,
                        'directional_accuracy': results.directional_accuracy
                    }
                elif isinstance(model, LSTMModel):
                    # Reshape data for LSTM if needed
                    if len(X_train.shape) == 2:
                        # Convert to sequences
                        sequence_length = min(60, X_train.shape[0] // 4)
                        X_train_seq, y_train_seq = self._create_sequences(X_train, y_train, sequence_length)
                        X_val_seq, y_val_seq = self._create_sequences(X_val, y_val, sequence_length)
                    else:
                        X_train_seq, y_train_seq = X_train, y_train
                        X_val_seq, y_val_seq = X_val, y_val
                        
                    results = model.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
                    individual_predictions[name] = results.predictions
                    model_performance[name] = {
                        'mse': results.mse,
                        'mae': results.mae,
                        'r2': results.r2,
                        'directional_accuracy': results.directional_accuracy
                    }
                elif isinstance(model, BayesianNeuralNetwork):
                    results = model.train(X_train, y_train, X_val, y_val)
                    individual_predictions[name] = results.mean_predictions
                    model_performance[name] = {
                        'mse': results.mse,
                        'mae': results.mae,
                        'r2': results.r2,
                        'calibration_score': results.calibration_score
                    }
                    
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
                
        # Calculate ensemble weights based on performance
        self._calculate_optimal_weights(model_performance)
        
        # Ensemble predictions
        ensemble_predictions = self._combine_predictions(individual_predictions)
        
        # Ensemble uncertainty
        uncertainty_estimates = self._calculate_ensemble_uncertainty(
            individual_predictions, ensemble_predictions
        )
        
        # Metrics
        mse = mean_squared_error(y_val, ensemble_predictions)
        mae = mean_absolute_error(y_val, ensemble_predictions)
        r2 = r2_score(y_val, ensemble_predictions)
        
        # Directional accuracy
        val_direction = np.sign(np.diff(y_val))
        pred_direction = np.sign(np.diff(ensemble_predictions))
        directional_accuracy = np.mean(val_direction == pred_direction)
        
        return EnsembleResults(
            ensemble_predictions=ensemble_predictions,
            individual_predictions=individual_predictions,
            model_weights=self.weights,
            uncertainty_estimates=uncertainty_estimates,
            actual_values=y_val,
            mse=mse,
            mae=mae,
            r2=r2,
            directional_accuracy=directional_accuracy,
            model_performance=model_performance
        )
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM from tabular data."""
        X_seq, y_seq = [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])
            
        return np.array(X_seq), np.array(y_seq)
        
    def _calculate_optimal_weights(self, model_performance: Dict[str, Dict[str, float]]):
        """Calculate optimal ensemble weights based on performance."""
        # Use inverse MSE as weights
        mse_scores = {name: perf['mse'] for name, perf in model_performance.items()}
        
        if mse_scores:
            # Inverse MSE weights
            inv_mse = {name: 1.0 / (mse + 1e-8) for name, mse in mse_scores.items()}
            total_weight = sum(inv_mse.values())
            
            self.weights = {name: weight / total_weight for name, weight in inv_mse.items()}
        else:
            # Equal weights if no performance data
            n_models = len(self.models)
            self.weights = {name: 1.0 / n_models for name in self.models.keys()}
            
    def _combine_predictions(self, individual_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine individual model predictions using weights."""
        if not individual_predictions:
            raise ValueError("No predictions to combine")
            
        # Ensure all predictions have the same length
        min_length = min(len(pred) for pred in individual_predictions.values())
        
        ensemble_pred = np.zeros(min_length)
        total_weight = 0
        
        for name, predictions in individual_predictions.items():
            weight = self.weights.get(name, 0)
            ensemble_pred += weight * predictions[:min_length]
            total_weight += weight
            
        if total_weight > 0:
            ensemble_pred /= total_weight
            
        return ensemble_pred
        
    def _calculate_ensemble_uncertainty(self, individual_predictions: Dict[str, np.ndarray],
                                      ensemble_predictions: np.ndarray) -> np.ndarray:
        """Calculate ensemble uncertainty."""
        if not individual_predictions:
            return np.zeros_like(ensemble_predictions)
            
        # Model disagreement as uncertainty measure
        predictions_array = np.array(list(individual_predictions.values()))
        uncertainty = np.std(predictions_array, axis=0)
        
        return uncertainty[:len(ensemble_predictions)]
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions with uncertainty."""
        individual_predictions = {}
        
        for name, model in self.models.items():
            try:
                if isinstance(model, (XGBoostModel, LSTMModel)):
                    pred = model.predict(X)
                elif isinstance(model, BayesianNeuralNetwork):
                    pred, _ = model.predict(X)
                else:
                    continue
                    
                individual_predictions[name] = pred
            except Exception as e:
                print(f"Error predicting with {name}: {e}")
                continue
                
        ensemble_predictions = self._combine_predictions(individual_predictions)
        uncertainty = self._calculate_ensemble_uncertainty(individual_predictions, ensemble_predictions)
        
        return ensemble_predictions, uncertainty

class MLStockAnalyzer:
    """Main class for comprehensive ML stock analysis."""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.models = {}
        self.results = {}
        
    def analyze_stock_data(self, data: Union[pd.DataFrame, np.ndarray],
                          price_column: str = 'close',
                          methods: List[str] = None,
                          test_size: float = 0.2) -> MLAnalysisResults:
        """Comprehensive ML analysis of stock data."""
        if methods is None:
            methods = ['xgboost', 'lstm', 'bayesian', 'ensemble']
            
        # Prepare data
        if isinstance(data, np.ndarray):
            # Convert to DataFrame
            data = pd.DataFrame(data, columns=[price_column])
            
        # Feature engineering
        featured_data = self.feature_engineer.create_features(data, price_column)
        
        # Prepare target and features
        target = featured_data[price_column].values
        features = featured_data.drop(columns=[price_column]).values
        feature_names = featured_data.drop(columns=[price_column]).columns.tolist()
        
        # Train-test split (time series aware)
        split_idx = int(len(features) * (1 - test_size))
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = target[:split_idx], target[split_idx:]
        
        # Scale features
        X_train_scaled, X_test_scaled = self.feature_engineer.scale_features(
            X_train, X_test, 'standard'
        )
        
        results = {}
        
        # XGBoost
        if 'xgboost' in methods:
            results['xgboost'] = self._analyze_xgboost(
                X_train, y_train, X_test, y_test, feature_names
            )
            
        # LSTM
        if 'lstm' in methods:
            results['lstm'] = self._analyze_lstm(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
        # Bayesian NN
        if 'bayesian' in methods:
            results['bayesian'] = self._analyze_bayesian_nn(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            
        # Ensemble
        if 'ensemble' in methods:
            results['ensemble'] = self._analyze_ensemble(
                X_train_scaled, y_train, X_test_scaled, y_test, feature_names
            )
            
        # Model comparison
        model_comparison = self._compare_models(results)
        
        # Feature analysis
        feature_analysis = self._analyze_features(results, feature_names)
        
        # Generate insights
        insights = self._generate_insights(results, data, price_column)
        
        return MLAnalysisResults(
            lstm_results=results.get('lstm'),
            xgboost_results=results.get('xgboost'),
            bayesian_results=results.get('bayesian'),
            ensemble_results=results.get('ensemble'),
            best_model=model_comparison['best_model'],
            model_comparison=model_comparison,
            feature_analysis=feature_analysis,
            insights=insights
        )
        
    def _analyze_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        feature_names: List[str]) -> XGBoostResults:
        """Analyze with XGBoost."""
        try:
            model = XGBoostModel()
            results = model.train(X_train, y_train, X_test, y_test, feature_names)
            self.models['xgboost'] = model
            return results
        except Exception as e:
            print(f"XGBoost analysis failed: {e}")
            return None
            
    def _analyze_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray) -> LSTMResults:
        """Analyze with LSTM."""
        try:
            # Create sequences
            sequence_length = min(60, len(X_train) // 4)
            X_train_seq, y_train_seq = self.feature_engineer.prepare_sequences(
                pd.DataFrame(X_train), 'target', sequence_length
            )
            X_test_seq, y_test_seq = self.feature_engineer.prepare_sequences(
                pd.DataFrame(X_test), 'target', sequence_length
            )
            
            # Add target column for sequence preparation
            train_df = pd.DataFrame(X_train)
            train_df['target'] = y_train
            test_df = pd.DataFrame(X_test)
            test_df['target'] = y_test
            
            X_train_seq, y_train_seq = self.feature_engineer.prepare_sequences(
                train_df, 'target', sequence_length
            )
            X_test_seq, y_test_seq = self.feature_engineer.prepare_sequences(
                test_df, 'target', sequence_length
            )
            
            if len(X_train_seq) == 0 or len(X_test_seq) == 0:
                print("Insufficient data for LSTM sequences")
                return None
                
            # Build and train model
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            model = LSTMModel(input_shape)
            model.build_model()
            
            results = model.train(X_train_seq, y_train_seq, X_test_seq, y_test_seq)
            self.models['lstm'] = model
            return results
            
        except Exception as e:
            print(f"LSTM analysis failed: {e}")
            return None
            
    def _analyze_bayesian_nn(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> BayesianNNResults:
        """Analyze with Bayesian Neural Network."""
        try:
            model = BayesianNeuralNetwork(X_train.shape[1])
            results = model.train(X_train, y_train, X_test, y_test)
            self.models['bayesian'] = model
            return results
        except Exception as e:
            print(f"Bayesian NN analysis failed: {e}")
            return None
            
    def _analyze_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_test: np.ndarray, y_test: np.ndarray,
                         feature_names: List[str]) -> EnsembleResults:
        """Analyze with ensemble of models."""
        try:
            ensemble = EnsembleModel()
            
            # Add models to ensemble
            if 'xgboost' in self.models:
                ensemble.add_model('xgboost', self.models['xgboost'])
            else:
                ensemble.add_model('xgboost', XGBoostModel())
                
            if TF_AVAILABLE:
                ensemble.add_model('bayesian', BayesianNeuralNetwork(X_train.shape[1]))
                
            results = ensemble.train(X_train, y_train, X_test, y_test, feature_names)
            self.models['ensemble'] = ensemble
            return results
            
        except Exception as e:
            print(f"Ensemble analysis failed: {e}")
            return None
            
    def _compare_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare model performance."""
        comparison = {
            'performance_metrics': {},
            'rankings': {},
            'best_model': None
        }
        
        # Collect metrics
        for name, result in results.items():
            if result is not None:
                comparison['performance_metrics'][name] = {
                    'mse': result.mse,
                    'mae': result.mae,
                    'r2': result.r2,
                    'directional_accuracy': getattr(result, 'directional_accuracy', 0)
                }
                
        # Rank by R²
        if comparison['performance_metrics']:
            sorted_models = sorted(
                comparison['performance_metrics'].items(),
                key=lambda x: x[1]['r2'],
                reverse=True
            )
            
            comparison['rankings']['by_r2'] = [model[0] for model in sorted_models]
            comparison['best_model'] = sorted_models[0][0]
            
        return comparison
        
    def _analyze_features(self, results: Dict[str, Any], feature_names: List[str]) -> Dict[str, Any]:
        """Analyze feature importance across models."""
        feature_analysis = {
            'importance_by_model': {},
            'average_importance': {},
            'top_features': []
        }
        
        # Collect feature importance from XGBoost
        if 'xgboost' in results and results['xgboost'] is not None:
            feature_analysis['importance_by_model']['xgboost'] = results['xgboost'].feature_importance
            
        # Average importance
        if feature_analysis['importance_by_model']:
            all_features = set()
            for importances in feature_analysis['importance_by_model'].values():
                all_features.update(importances.keys())
                
            for feature in all_features:
                importances = []
                for model_importances in feature_analysis['importance_by_model'].values():
                    importances.append(model_importances.get(feature, 0))
                feature_analysis['average_importance'][feature] = np.mean(importances)
                
            # Top features
            sorted_features = sorted(
                feature_analysis['average_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            feature_analysis['top_features'] = [f[0] for f in sorted_features[:10]]
            
        return feature_analysis
        
    def _generate_insights(self, results: Dict[str, Any], data: pd.DataFrame,
                          price_column: str) -> Dict[str, Any]:
        """Generate insights from ML analysis."""
        insights = {
            'model_performance': {},
            'uncertainty_analysis': {},
            'prediction_confidence': {},
            'feature_insights': {},
            'trading_signals': {}
        }
        
        # Model performance insights
        best_r2 = 0
        best_model = None
        
        for name, result in results.items():
            if result is not None:
                insights['model_performance'][name] = {
                    'accuracy': 'high' if result.r2 > 0.7 else 'medium' if result.r2 > 0.4 else 'low',
                    'reliability': 'high' if result.directional_accuracy > 0.6 else 'medium' if result.directional_accuracy > 0.5 else 'low'
                }
                
                if result.r2 > best_r2:
                    best_r2 = result.r2
                    best_model = name
                    
        # Uncertainty analysis
        if 'bayesian' in results and results['bayesian'] is not None:
            bayesian_result = results['bayesian']
            avg_uncertainty = np.mean(bayesian_result.uncertainty_estimates)
            
            insights['uncertainty_analysis'] = {
                'average_uncertainty': float(avg_uncertainty),
                'uncertainty_trend': 'increasing' if bayesian_result.uncertainty_estimates[-1] > avg_uncertainty else 'decreasing',
                'epistemic_vs_aleatoric': {
                    'epistemic_ratio': float(np.mean(bayesian_result.epistemic_uncertainty) / (avg_uncertainty + 1e-8)),
                    'aleatoric_ratio': float(np.mean(bayesian_result.aleatoric_uncertainty) / (avg_uncertainty + 1e-8))
                }
            }
            
        # Prediction confidence
        if best_model and results[best_model] is not None:
            best_result = results[best_model]
            
            insights['prediction_confidence'] = {
                'overall_confidence': 'high' if best_r2 > 0.7 else 'medium' if best_r2 > 0.4 else 'low',
                'directional_accuracy': float(best_result.directional_accuracy),
                'recommended_horizon': 'short-term' if best_r2 > 0.6 else 'medium-term'
            }
            
        # Feature insights
        if 'xgboost' in results and results['xgboost'] is not None:
            feature_importance = results['xgboost'].feature_importance
            top_feature = max(feature_importance.items(), key=lambda x: x[1])[0]
            
            insights['feature_insights'] = {
                'most_important_feature': top_feature,
                'technical_vs_fundamental': self._categorize_features(feature_importance),
                'feature_stability': 'high'  # Placeholder
            }
            
        # Trading signals
        if best_model and results[best_model] is not None:
            predictions = results[best_model].predictions
            actual = results[best_model].actual_values
            
            if len(predictions) > 1:
                recent_trend = 'bullish' if predictions[-1] > predictions[-2] else 'bearish'
                prediction_strength = abs(predictions[-1] - predictions[-2]) / np.std(predictions)
                
                insights['trading_signals'] = {
                    'short_term_trend': recent_trend,
                    'signal_strength': 'strong' if prediction_strength > 1 else 'weak',
                    'confidence_level': insights['prediction_confidence']['overall_confidence']
                }
                
        return insights
        
    def _categorize_features(self, feature_importance: Dict[str, float]) -> Dict[str, float]:
        """Categorize features into technical vs fundamental."""
        technical_keywords = ['ma_', 'rsi', 'macd', 'bb_', 'volatility_', 'momentum_', 'lag_']
        fundamental_keywords = ['volume', 'price_volume', 'hl_ratio']
        
        technical_importance = 0
        fundamental_importance = 0
        other_importance = 0
        
        for feature, importance in feature_importance.items():
            if any(keyword in feature for keyword in technical_keywords):
                technical_importance += importance
            elif any(keyword in feature for keyword in fundamental_keywords):
                fundamental_importance += importance
            else:
                other_importance += importance
                
        total_importance = technical_importance + fundamental_importance + other_importance
        
        if total_importance > 0:
            return {
                'technical': technical_importance / total_importance,
                'fundamental': fundamental_importance / total_importance,
                'other': other_importance / total_importance
            }
        else:
            return {'technical': 0.33, 'fundamental': 0.33, 'other': 0.34}
            
    def plot_results(self, results: MLAnalysisResults, save_path: str = None) -> None:
        """Plot comprehensive analysis results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('ML Stock Analysis Results', fontsize=16)
        
        # Model comparison
        if results.model_comparison['performance_metrics']:
            models = list(results.model_comparison['performance_metrics'].keys())
            r2_scores = [results.model_comparison['performance_metrics'][m]['r2'] for m in models]
            
            axes[0, 0].bar(models, r2_scores)
            axes[0, 0].set_title('Model R² Comparison')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
        # Feature importance
        if results.feature_analysis['top_features']:
            top_features = results.feature_analysis['top_features'][:10]
            importances = [results.feature_analysis['average_importance'][f] for f in top_features]
            
            axes[0, 1].barh(top_features, importances)
            axes[0, 1].set_title('Top 10 Feature Importance')
            axes[0, 1].set_xlabel('Importance')
            
        # Predictions vs Actual
        if results.xgboost_results:
            axes[0, 2].scatter(results.xgboost_results.actual_values, 
                             results.xgboost_results.predictions, alpha=0.6)
            axes[0, 2].plot([results.xgboost_results.actual_values.min(), 
                           results.xgboost_results.actual_values.max()],
                          [results.xgboost_results.actual_values.min(), 
                           results.xgboost_results.actual_values.max()], 'r--')
            axes[0, 2].set_title('XGBoost: Predictions vs Actual')
            axes[0, 2].set_xlabel('Actual Values')
            axes[0, 2].set_ylabel('Predicted Values')
            
        # LSTM training history
        if results.lstm_results:
            axes[1, 0].plot(results.lstm_results.train_loss, label='Training Loss')
            axes[1, 0].plot(results.lstm_results.val_loss, label='Validation Loss')
            axes[1, 0].set_title('LSTM Training History')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            
        # Uncertainty estimates
        if results.bayesian_results:
            x_range = range(len(results.bayesian_results.uncertainty_estimates))
            axes[1, 1].plot(x_range, results.bayesian_results.mean_predictions, label='Predictions')
            axes[1, 1].fill_between(x_range,
                                  results.bayesian_results.mean_predictions - results.bayesian_results.uncertainty_estimates,
                                  results.bayesian_results.mean_predictions + results.bayesian_results.uncertainty_estimates,
                                  alpha=0.3, label='Uncertainty')
            axes[1, 1].plot(x_range, results.bayesian_results.actual_values, label='Actual', alpha=0.7)
            axes[1, 1].set_title('Bayesian NN: Predictions with Uncertainty')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].legend()
            
        # Ensemble performance
        if results.ensemble_results:
            models = list(results.ensemble_results.model_weights.keys())
            weights = list(results.ensemble_results.model_weights.values())
            
            axes[1, 2].pie(weights, labels=models, autopct='%1.1f%%')
            axes[1, 2].set_title('Ensemble Model Weights')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_trading_signals(self, results: MLAnalysisResults, 
                               confidence_threshold: float = 0.6) -> Dict[str, Any]:
        """Generate trading signals based on ML predictions."""
        signals = {
            'primary_signal': 'hold',
            'confidence': 0.0,
            'supporting_signals': [],
            'risk_assessment': 'medium',
            'recommended_actions': []
        }
        
        if not results.best_model or results.best_model not in ['xgboost', 'lstm', 'bayesian', 'ensemble']:
            return signals
            
        # Get best model results
        if results.best_model == 'xgboost' and results.xgboost_results:
            best_result = results.xgboost_results
        elif results.best_model == 'lstm' and results.lstm_results:
            best_result = results.lstm_results
        elif results.best_model == 'bayesian' and results.bayesian_results:
            best_result = results.bayesian_results
        elif results.best_model == 'ensemble' and results.ensemble_results:
            best_result = results.ensemble_results
        else:
            return signals
            
        # Determine primary signal
        if hasattr(best_result, 'predictions') and len(best_result.predictions) >= 2:
            recent_change = best_result.predictions[-1] - best_result.predictions[-2]
            prediction_strength = abs(recent_change) / np.std(best_result.predictions)
            
            if recent_change > 0 and prediction_strength > 0.5:
                signals['primary_signal'] = 'buy'
            elif recent_change < 0 and prediction_strength > 0.5:
                signals['primary_signal'] = 'sell'
                
        # Calculate confidence
        model_r2 = best_result.r2 if hasattr(best_result, 'r2') else 0
        directional_acc = best_result.directional_accuracy if hasattr(best_result, 'directional_accuracy') else 0.5
        
        signals['confidence'] = (model_r2 + directional_acc) / 2
        
        # Supporting signals
        if signals['confidence'] > confidence_threshold:
            signals['supporting_signals'].append('high_model_accuracy')
            
        if results.bayesian_results:
            avg_uncertainty = np.mean(results.bayesian_results.uncertainty_estimates)
            if avg_uncertainty < np.std(results.bayesian_results.mean_predictions) * 0.5:
                signals['supporting_signals'].append('low_prediction_uncertainty')
                
        # Risk assessment
        if signals['confidence'] > 0.7:
            signals['risk_assessment'] = 'low'
        elif signals['confidence'] > 0.5:
            signals['risk_assessment'] = 'medium'
        else:
            signals['risk_assessment'] = 'high'
            
        # Recommended actions
        if signals['primary_signal'] == 'buy' and signals['confidence'] > confidence_threshold:
            signals['recommended_actions'].append('Consider long position')
            signals['recommended_actions'].append('Set stop-loss based on uncertainty estimates')
        elif signals['primary_signal'] == 'sell' and signals['confidence'] > confidence_threshold:
            signals['recommended_actions'].append('Consider short position or exit long')
            signals['recommended_actions'].append('Monitor for trend reversal')
        else:
            signals['recommended_actions'].append('Hold current position')
            signals['recommended_actions'].append('Wait for clearer signals')
            
        return signals
        
    def backtest_strategy(self, data: pd.DataFrame, results: MLAnalysisResults,
                         initial_capital: float = 10000, 
                         transaction_cost: float = 0.001) -> Dict[str, Any]:
        """Backtest trading strategy based on ML predictions."""
        if not results.best_model:
            return {'error': 'No valid model results for backtesting'}
            
        # Get predictions
        if results.best_model == 'xgboost' and results.xgboost_results:
            predictions = results.xgboost_results.predictions
            actual_values = results.xgboost_results.actual_values
        elif results.best_model == 'ensemble' and results.ensemble_results:
            predictions = results.ensemble_results.ensemble_predictions
            actual_values = results.ensemble_results.actual_values
        else:
            return {'error': 'Best model results not available'}
            
        # Simple strategy: buy when prediction > current, sell when prediction < current
        positions = []
        returns = []
        capital = initial_capital
        position = 0  # 0: no position, 1: long, -1: short
        
        for i in range(1, len(predictions)):
            current_price = actual_values[i-1]
            predicted_price = predictions[i]
            next_price = actual_values[i]
            
            # Generate signal
            if predicted_price > current_price * 1.01 and position <= 0:  # Buy signal
                if position == -1:  # Close short
                    capital *= (current_price / next_price)
                    capital *= (1 - transaction_cost)
                # Open long
                position = 1
                entry_price = next_price
                
            elif predicted_price < current_price * 0.99 and position >= 0:  # Sell signal
                if position == 1:  # Close long
                    capital *= (next_price / entry_price)
                    capital *= (1 - transaction_cost)
                # Open short
                position = -1
                entry_price = next_price
                
            positions.append(position)
            returns.append(capital / initial_capital - 1)
            
        # Calculate performance metrics
        total_return = (capital - initial_capital) / initial_capital
        
        if len(returns) > 1:
            returns_array = np.array(returns)
            volatility = np.std(np.diff(returns_array)) * np.sqrt(252)
            sharpe_ratio = (total_return * 252) / (volatility + 1e-8)
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + np.diff(returns_array, prepend=0))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdown)
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0
            
        return {
            'total_return': total_return,
            'annualized_return': total_return * 252 / len(returns) if returns else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_capital': capital,
            'number_of_trades': len([p for p in positions if p != 0]),
            'win_rate': self._calculate_win_rate(positions, actual_values)
        }
        
    def _calculate_win_rate(self, positions: List[int], prices: np.ndarray) -> float:
        """Calculate win rate of trades."""
        trades = []
        entry_price = None
        entry_position = None
        
        for i, position in enumerate(positions):
            if position != 0 and entry_position is None:
                entry_price = prices[i]
                entry_position = position
            elif position == 0 and entry_position is not None:
                exit_price = prices[i]
                if entry_position == 1:  # Long trade
                    profit = (exit_price - entry_price) / entry_price
                else:  # Short trade
                    profit = (entry_price - exit_price) / entry_price
                trades.append(profit > 0)
                entry_price = None
                entry_position = None
                
        return np.mean(trades) if trades else 0.0

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Generate synthetic stock data
    n_days = len(dates)
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = [100]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
        
    # Add some volume and high/low data
    volumes = np.random.lognormal(10, 0.5, n_days)
    highs = np.array(prices) * (1 + np.random.uniform(0, 0.05, n_days))
    lows = np.array(prices) * (1 - np.random.uniform(0, 0.05, n_days))
    
    sample_data = pd.DataFrame({
        'close': prices,
        'high': highs,
        'low': lows,
        'volume': volumes
    }, index=dates)
    
    # Initialize analyzer
    analyzer = MLStockAnalyzer()
    
    # Run analysis
    print("Running ML Stock Analysis...")
    results = analyzer.analyze_stock_data(
        sample_data, 
        methods=['xgboost', 'bayesian']
    )
    
    # Print results
    print(f"\nBest Model: {results.best_model}")
    print(f"Model Comparison: {results.model_comparison}")
    print(f"Top Features: {results.feature_analysis['top_features'][:5]}")
    print(f"Insights: {results.insights}")
    
    # Generate trading signals
    signals = analyzer.generate_trading_signals(results)
    print(f"\nTrading Signals: {signals}")
    
    # Backtest strategy
    backtest_results = analyzer.backtest_strategy(sample_data, results)
    print(f"\nBacktest Results: {backtest_results}")
    
    print("\nML Stock Analysis completed successfully!")