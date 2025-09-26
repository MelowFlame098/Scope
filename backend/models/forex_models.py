import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from .base_model import BaseModel, ModelType, AssetCategory
import logging

class PPPModel(BaseModel):
    """
    Purchasing Power Parity (PPP) model for forex prediction.
    Based on the theory that exchange rates should adjust to equalize prices.
    """
    
    def __init__(self):
        super().__init__(
            model_id="forex_ppp",
            name="Purchasing Power Parity",
            category=AssetCategory.FOREX,
            model_type=ModelType.FUNDAMENTAL,
            description="PPP model for long-term exchange rate prediction"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for PPP analysis.
        """
        required_cols = ['timestamp', 'close', 'inflation_domestic', 'inflation_foreign']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for PPP model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate inflation differential
        df['inflation_diff'] = df['inflation_domestic'] - df['inflation_foreign']
        
        # Calculate cumulative inflation differential
        df['cum_inflation_diff'] = df['inflation_diff'].cumsum()
        
        # Calculate real exchange rate
        base_rate = df['close'].iloc[0]
        df['theoretical_rate'] = base_rate * np.exp(df['cum_inflation_diff'] / 100)
        
        # Calculate PPP deviation
        df['ppp_deviation'] = (df['close'] - df['theoretical_rate']) / df['theoretical_rate']
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train PPP model by analyzing historical deviations.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            
            # Calculate mean reversion parameters
            ppp_deviations = prepared_data['ppp_deviation']
            
            # Estimate half-life of mean reversion
            from sklearn.linear_model import LinearRegression
            
            # AR(1) model for mean reversion
            y = ppp_deviations[1:].values
            X = ppp_deviations[:-1].values.reshape(-1, 1)
            
            reg = LinearRegression()
            reg.fit(X, y)
            
            phi = reg.coef_[0]  # AR coefficient
            half_life = -np.log(2) / np.log(abs(phi)) if abs(phi) < 1 else np.inf
            
            # Calculate statistics
            mean_deviation = ppp_deviations.mean()
            std_deviation = ppp_deviations.std()
            
            # Calculate accuracy based on mean reversion strength
            accuracy = min(100, (1 - abs(phi)) * 100) if abs(phi) < 1 else 0
            
            self.parameters = {
                'phi': phi,
                'half_life': half_life,
                'mean_deviation': mean_deviation,
                'std_deviation': std_deviation,
                'base_rate': prepared_data['close'].iloc[0]
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'phi': phi,
                'half_life': half_life,
                'mean_deviation': mean_deviation,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate PPP-based exchange rate predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            
            predictions = []
            signals = []
            
            for _, row in prepared_data.iterrows():
                current_deviation = row['ppp_deviation']
                theoretical_rate = row['theoretical_rate']
                
                # Predict next period deviation using AR(1)
                next_deviation = self.parameters['phi'] * current_deviation
                predicted_rate = theoretical_rate * (1 + next_deviation)
                
                predictions.append(predicted_rate)
                
                # Generate trading signals based on deviation
                if current_deviation < -self.parameters['std_deviation']:
                    signals.append(1)  # Buy - undervalued
                elif current_deviation > self.parameters['std_deviation']:
                    signals.append(-1)  # Sell - overvalued
                else:
                    signals.append(0)  # Hold
            
            return {
                'status': 'success',
                'predictions': predictions,
                'signals': signals,
                'theoretical_rates': prepared_data['theoretical_rate'].tolist(),
                'deviations': prepared_data['ppp_deviation'].tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class ForexLSTMModel(BaseModel):
    """
    LSTM model specifically designed for forex prediction.
    """
    
    def __init__(self):
        super().__init__(
            model_id="forex_lstm",
            name="LSTM for Forex",
            category=AssetCategory.FOREX,
            model_type=ModelType.ML,
            description="LSTM neural network for forex prediction"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare forex data for LSTM training.
        """
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for Forex LSTM model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate forex-specific features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Technical indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Session indicators (forex market sessions)
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 9)).astype(int)
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 17)).astype(int)
        df['ny_session'] = ((df['hour'] >= 13) & (df['hour'] < 22)).astype(int)
        
        # Normalize features
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'sma_20', 'sma_50', 
                       'macd', 'rsi', 'bb_position', 'volatility']
        
        for col in feature_cols:
            if col in df.columns:
                df[f'{col}_norm'] = (df[col] - df[col].mean()) / df[col].std()
        
        return df.dropna()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> tuple:
        """Create sequences for LSTM training."""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 3])  # Close price index
        return np.array(X), np.array(y)
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train Forex LSTM model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            
            # Select normalized features
            feature_cols = [col for col in prepared_data.columns if col.endswith('_norm')] + \
                          ['asian_session', 'london_session', 'ny_session', 'hour', 'day_of_week']
            
            features = prepared_data[feature_cols].values
            
            sequence_length = kwargs.get('sequence_length', 60)
            X, y = self._create_sequences(features, sequence_length)
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Use RandomForest as LSTM substitute for demonstration
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Flatten sequences for sklearn
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            self.model_instance = RandomForestRegressor(
                n_estimators=200,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model_instance.fit(X_train_flat, y_train)
            
            # Evaluate
            y_pred = self.model_instance.predict(X_test_flat)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Calculate directional accuracy
            y_test_direction = np.sign(np.diff(y_test))
            y_pred_direction = np.sign(np.diff(y_pred))
            directional_accuracy = np.mean(y_test_direction == y_pred_direction) * 100
            
            accuracy = max(0, directional_accuracy)
            
            self.parameters = {
                'sequence_length': sequence_length,
                'n_features': len(feature_cols),
                'feature_cols': feature_cols,
                'mse': mse,
                'mae': mae,
                'directional_accuracy': directional_accuracy
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'mse': mse,
                'mae': mae,
                'directional_accuracy': directional_accuracy,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate Forex LSTM predictions.
        """
        if self.model_instance is None:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            
            features = prepared_data[self.parameters['feature_cols']].values
            sequence_length = self.parameters['sequence_length']
            
            if len(features) < sequence_length:
                return {'status': 'error', 'message': 'Insufficient data for prediction'}
            
            # Create sequences
            X, _ = self._create_sequences(features, sequence_length)
            X_flat = X.reshape(X.shape[0], -1)
            
            # Generate predictions
            predictions = self.model_instance.predict(X_flat)
            
            # Generate trading signals
            signals = []
            for i in range(1, len(predictions)):
                price_change = predictions[i] - predictions[i-1]
                if price_change > 0.0001:  # Threshold for forex
                    signals.append(1)  # Buy
                elif price_change < -0.0001:
                    signals.append(-1)  # Sell
                else:
                    signals.append(0)  # Hold
            
            signals = [0] + signals  # Add initial signal
            
            return {
                'status': 'success',
                'predictions': predictions.tolist(),
                'signals': signals,
                'timestamps': prepared_data['timestamp'].iloc[sequence_length:].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        metrics = self.calculate_common_metrics(actual, predicted)
        
        # Add forex-specific metrics
        if len(actual) > 1 and len(predicted) > 1:
            actual_direction = np.sign(np.diff(actual))
            predicted_direction = np.sign(np.diff(predicted))
            directional_accuracy = np.mean(actual_direction == predicted_direction) * 100
            metrics['directional_accuracy'] = directional_accuracy
        
        return metrics

class IRPModel(BaseModel):
    """
    Interest Rate Parity (IRP) model for forex prediction.
    """
    
    def __init__(self):
        super().__init__(
            model_id="forex_irp",
            name="Interest Rate Parity",
            category=AssetCategory.FOREX,
            model_type=ModelType.FUNDAMENTAL,
            description="Interest Rate Parity model for forex prediction"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for IRP analysis.
        """
        required_cols = ['timestamp', 'close', 'interest_rate_domestic', 'interest_rate_foreign']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for IRP model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate interest rate differential
        df['rate_diff'] = df['interest_rate_domestic'] - df['interest_rate_foreign']
        
        # Calculate theoretical forward rate (simplified)
        df['forward_rate'] = df['close'] * (1 + df['interest_rate_domestic'] / 100) / (1 + df['interest_rate_foreign'] / 100)
        
        # Calculate IRP deviation
        df['irp_deviation'] = (df['close'] - df['forward_rate']) / df['forward_rate']
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train IRP model.
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            
            # Analyze relationship between rate differentials and exchange rate changes
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            X = prepared_data['rate_diff'].values.reshape(-1, 1)
            y = prepared_data['close'].pct_change().dropna().values
            
            # Align arrays
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            reg = LinearRegression()
            reg.fit(X, y)
            
            y_pred = reg.predict(X)
            r2 = r2_score(y, y_pred)
            
            # Calculate mean reversion parameters for deviations
            deviations = prepared_data['irp_deviation']
            mean_deviation = deviations.mean()
            std_deviation = deviations.std()
            
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'slope': reg.coef_[0],
                'intercept': reg.intercept_,
                'r2': r2,
                'mean_deviation': mean_deviation,
                'std_deviation': std_deviation
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'slope': reg.coef_[0],
                'r2': r2,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate IRP-based predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            
            predictions = []
            signals = []
            
            for _, row in prepared_data.iterrows():
                rate_diff = row['rate_diff']
                current_rate = row['close']
                irp_deviation = row['irp_deviation']
                
                # Predict exchange rate change based on interest rate differential
                predicted_change = self.parameters['slope'] * rate_diff + self.parameters['intercept']
                predicted_rate = current_rate * (1 + predicted_change)
                
                predictions.append(predicted_rate)
                
                # Generate signals based on IRP deviation
                if irp_deviation < -self.parameters['std_deviation']:
                    signals.append(1)  # Buy - undervalued
                elif irp_deviation > self.parameters['std_deviation']:
                    signals.append(-1)  # Sell - overvalued
                else:
                    signals.append(0)  # Hold
            
            return {
                'status': 'success',
                'predictions': predictions,
                'signals': signals,
                'forward_rates': prepared_data['forward_rate'].tolist(),
                'deviations': prepared_data['irp_deviation'].tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)

class ForexGARCHModel(BaseModel):
    """
    GARCH model for forex volatility prediction.
    """
    
    def __init__(self):
        super().__init__(
            model_id="forex_garch",
            name="GARCH Volatility",
            category=AssetCategory.FOREX,
            model_type=ModelType.STATISTICAL,
            description="GARCH model for forex volatility prediction"
        )
    
    def prepare_data(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Prepare data for GARCH analysis.
        """
        required_cols = ['timestamp', 'close']
        if not self.validate_data(data, required_cols):
            raise ValueError("Invalid data format for GARCH model")
        
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Calculate squared returns for volatility
        df['squared_returns'] = df['returns'] ** 2
        
        return df.dropna()
    
    def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Train GARCH model (simplified implementation).
        """
        self.start_run()
        
        try:
            prepared_data = self.prepare_data(data)
            
            returns = prepared_data['returns'].values
            
            # Simplified GARCH(1,1) estimation using rolling statistics
            # In practice, you would use specialized libraries like arch
            
            # Calculate rolling volatility
            window = kwargs.get('window', 20)
            prepared_data['rolling_vol'] = prepared_data['returns'].rolling(window=window).std()
            
            # Estimate GARCH parameters (simplified)
            squared_returns = prepared_data['squared_returns'].dropna()
            lagged_squared_returns = squared_returns.shift(1).dropna()
            lagged_variance = prepared_data['rolling_vol'].shift(1).dropna() ** 2
            
            # Align series
            min_len = min(len(squared_returns), len(lagged_squared_returns), len(lagged_variance))
            y = squared_returns.iloc[:min_len].values
            X1 = lagged_squared_returns.iloc[:min_len].values
            X2 = lagged_variance.iloc[:min_len].values
            
            # Simple linear regression for GARCH parameters
            from sklearn.linear_model import LinearRegression
            
            X = np.column_stack([X1, X2])
            reg = LinearRegression()
            reg.fit(X, y)
            
            alpha = reg.coef_[0]  # ARCH parameter
            beta = reg.coef_[1]   # GARCH parameter
            omega = reg.intercept_  # Constant
            
            # Calculate model accuracy
            y_pred = reg.predict(X)
            from sklearn.metrics import r2_score
            r2 = r2_score(y, y_pred)
            
            accuracy = max(0, r2 * 100)
            
            self.parameters = {
                'alpha': alpha,
                'beta': beta,
                'omega': omega,
                'window': window,
                'r2': r2
            }
            
            self.complete_run(accuracy)
            
            return {
                'status': 'success',
                'alpha': alpha,
                'beta': beta,
                'omega': omega,
                'r2': r2,
                'accuracy': accuracy
            }
            
        except Exception as e:
            self.error_run(str(e))
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Generate GARCH volatility predictions.
        """
        if not self.parameters:
            return {'status': 'error', 'message': 'Model not trained'}
        
        try:
            prepared_data = self.prepare_data(data)
            
            # Calculate rolling volatility
            window = self.parameters['window']
            prepared_data['rolling_vol'] = prepared_data['returns'].rolling(window=window).std()
            
            volatility_predictions = []
            
            for i in range(1, len(prepared_data)):
                if i >= window:
                    prev_squared_return = prepared_data['squared_returns'].iloc[i-1]
                    prev_variance = prepared_data['rolling_vol'].iloc[i-1] ** 2
                    
                    # GARCH(1,1) prediction
                    predicted_variance = (self.parameters['omega'] + 
                                        self.parameters['alpha'] * prev_squared_return + 
                                        self.parameters['beta'] * prev_variance)
                    
                    predicted_volatility = np.sqrt(max(0, predicted_variance))
                    volatility_predictions.append(predicted_volatility)
                else:
                    volatility_predictions.append(None)
            
            # Add initial None for alignment
            volatility_predictions = [None] + volatility_predictions
            
            return {
                'status': 'success',
                'volatility_predictions': volatility_predictions,
                'actual_volatility': prepared_data['rolling_vol'].tolist(),
                'timestamps': prepared_data['timestamp'].dt.isoformat().tolist()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def evaluate(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        return self.calculate_common_metrics(actual, predicted)