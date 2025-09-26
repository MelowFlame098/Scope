"""LSTM Model for Stock Price Prediction

This module implements LSTM neural networks for time series prediction
of stock prices with comprehensive feature engineering and uncertainty estimation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Deep Learning imports (with fallbacks)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using simplified LSTM implementation.")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Using alternative implementations.")

# Suppress warnings
warnings.filterwarnings('ignore')

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
            print("Neither TensorFlow nor PyTorch available. Using simplified implementation.")
            self._build_simplified_model()
            
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
        
    def _build_simplified_model(self) -> None:
        """Build simplified model when deep learning frameworks are not available."""
        # Use ensemble of simple models as fallback
        self.model = {
            'linear': LinearRegression(),
            'rf': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 32) -> LSTMResults:
        """Train the LSTM model."""
        if TF_AVAILABLE and hasattr(self.model, 'fit'):
            return self._train_tensorflow(X_train, y_train, X_val, y_val, epochs, batch_size)
        elif TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            return self._train_pytorch(X_train, y_train, X_val, y_val, epochs, batch_size)
        else:
            return self._train_simplified(X_train, y_train, X_val, y_val)
            
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
        
    def _train_simplified(self, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray) -> LSTMResults:
        """Train using simplified models when deep learning frameworks are not available."""
        # Flatten sequences for traditional ML models
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        # Train ensemble of simple models
        predictions = []
        
        for name, model in self.model.items():
            model.fit(X_train_flat, y_train)
            pred = model.predict(X_val_flat)
            predictions.append(pred)
            
        # Ensemble prediction (average)
        val_pred = np.mean(predictions, axis=0)
        
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
            train_loss=[],  # Not available for simplified models
            val_loss=[],
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
        elif TORCH_AVAILABLE and isinstance(self.model, nn.Module):
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X)
                return self.model(X_tensor).numpy().flatten()
        else:
            # Simplified model prediction
            X_flat = X.reshape(X.shape[0], -1)
            predictions = []
            for model in self.model.values():
                pred = model.predict(X_flat)
                predictions.append(pred)
            return np.mean(predictions, axis=0)
    
    def plot_training_history(self, results: LSTMResults):
        """Plot training history."""
        try:
            import matplotlib.pyplot as plt
            
            if results.train_loss and results.val_loss:
                plt.figure(figsize=(12, 4))
                
                plt.subplot(1, 2, 1)
                plt.plot(results.train_loss, label='Training Loss')
                plt.plot(results.val_loss, label='Validation Loss')
                plt.title('Model Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(1, 2, 2)
                plt.scatter(results.actual_values, results.predictions, alpha=0.6)
                plt.plot([results.actual_values.min(), results.actual_values.max()],
                        [results.actual_values.min(), results.actual_values.max()], 'r--')
                plt.title('Predictions vs Actual')
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
            else:
                print("No training history available for plotting")
                
        except ImportError:
            print("Matplotlib not available for plotting")

# Example usage
if __name__ == "__main__":
    # Generate sample stock data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Generate synthetic stock data
    n_days = len(dates)
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = [100]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
        
    # Add volume and high/low data
    volumes = np.random.lognormal(10, 0.5, n_days)
    highs = np.array(prices) * (1 + np.random.uniform(0, 0.05, n_days))
    lows = np.array(prices) * (1 - np.random.uniform(0, 0.05, n_days))
    
    sample_data = pd.DataFrame({
        'close': prices,
        'high': highs,
        'low': lows,
        'volume': volumes
    }, index=dates)
    
    # Feature engineering
    feature_engineer = FeatureEngineer()
    featured_data = feature_engineer.create_features(sample_data, 'close')
    
    print(f"Created {len(feature_engineer.feature_names)} features")
    print(f"Feature names: {feature_engineer.feature_names[:10]}...")  # Show first 10
    
    # Prepare sequences
    X, y = feature_engineer.prepare_sequences(featured_data, 'close', sequence_length=30)
    
    print(f"Sequence shape: {X.shape}, Target shape: {y.shape}")
    
    # Train-test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    X_train_scaled, X_test_scaled = feature_engineer.scale_features(X_train, X_test)
    
    # Build and train LSTM model
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    lstm_model = LSTMModel(input_shape)
    
    try:
        print("Building LSTM model...")
        lstm_model.build_model()
        
        print("Training LSTM model...")
        results = lstm_model.train(X_train_scaled, y_train, X_test_scaled, y_test, epochs=50)
        
        print("\n=== LSTM Results ===")
        print(f"MSE: {results.mse:.6f}")
        print(f"MAE: {results.mae:.6f}")
        print(f"R²: {results.r2:.6f}")
        print(f"Directional Accuracy: {results.directional_accuracy:.6f}")
        
        # Plot results
        lstm_model.plot_training_history(results)
        
        print("\nLSTM model training completed successfully!")
        
    except Exception as e:
        print(f"LSTM model training failed: {e}")
        import traceback
        traceback.print_exc()