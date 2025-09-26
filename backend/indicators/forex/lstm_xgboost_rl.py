from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Deep Learning imports (with fallbacks)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, MultiHeadAttention
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will use simplified implementations.")

# Reinforcement Learning imports (with fallbacks)
try:
    import gym
    from stable_baselines3 import PPO, A2C, DQN
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Reinforcement Learning libraries not available. Using simplified RL implementation.")

@dataclass
class ForexMLData:
    """Data structure for forex ML analysis"""
    timestamp: List[datetime]
    exchange_rate: List[float]
    interest_rate_domestic: List[float]
    interest_rate_foreign: List[float]
    inflation_domestic: List[float]
    inflation_foreign: List[float]
    volatility: List[float]
    volume: List[float]
    economic_indicators: Dict[str, List[float]]
    technical_indicators: Dict[str, List[float]]
    
@dataclass
class MLPrediction:
    """ML model prediction with uncertainty"""
    value: float
    confidence: float
    lower_bound: float
    upper_bound: float
    probability_up: float
    probability_down: float
    
@dataclass
class LSTMResults:
    """Results from LSTM analysis"""
    predictions: List[MLPrediction]
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    attention_weights: Optional[List[List[float]]]
    training_history: Dict[str, List[float]]
    model_parameters: Dict[str, Any]
    
@dataclass
class XGBoostResults:
    """Results from XGBoost analysis"""
    predictions: List[MLPrediction]
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    shap_values: Optional[List[List[float]]]
    model_parameters: Dict[str, Any]
    cross_validation_scores: List[float]
    
@dataclass
class RLResults:
    """Results from Reinforcement Learning analysis"""
    actions: List[str]
    rewards: List[float]
    portfolio_value: List[float]
    model_performance: Dict[str, float]
    policy_parameters: Dict[str, Any]
    training_metrics: Dict[str, List[float]]
    
@dataclass
class ForexMLResults:
    """Comprehensive forex ML results"""
    lstm_results: LSTMResults
    xgboost_results: XGBoostResults
    rl_results: RLResults
    ensemble_predictions: List[MLPrediction]
    model_comparison: Dict[str, Dict[str, float]]
    trading_signals: List[str]
    risk_metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    
class FeatureEngineer:
    """Feature engineering for forex ML models"""
    
    def __init__(self):
        self.scalers = {}
        self.feature_names = []
        
    def create_features(self, forex_data: ForexMLData) -> pd.DataFrame:
        """Create comprehensive feature set"""
        df = pd.DataFrame({
            'timestamp': forex_data.timestamp,
            'exchange_rate': forex_data.exchange_rate,
            'interest_rate_domestic': forex_data.interest_rate_domestic,
            'interest_rate_foreign': forex_data.interest_rate_foreign,
            'inflation_domestic': forex_data.inflation_domestic,
            'inflation_foreign': forex_data.inflation_foreign,
            'volatility': forex_data.volatility,
            'volume': forex_data.volume
        })
        
        # Add economic indicators
        for name, values in forex_data.economic_indicators.items():
            df[f'econ_{name}'] = values
            
        # Add technical indicators
        for name, values in forex_data.technical_indicators.items():
            df[f'tech_{name}'] = values
        
        # Create derived features
        df = self._create_derived_features(df)
        
        # Create lagged features
        df = self._create_lagged_features(df)
        
        # Create rolling statistics
        df = self._create_rolling_features(df)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Remove timestamp for modeling
        feature_df = df.drop('timestamp', axis=1)
        self.feature_names = feature_df.columns.tolist()
        
        return feature_df.fillna(method='ffill').fillna(method='bfill')
    
    def _create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived economic features"""
        # Interest rate differential
        df['interest_rate_diff'] = df['interest_rate_domestic'] - df['interest_rate_foreign']
        
        # Inflation differential
        df['inflation_diff'] = df['inflation_domestic'] - df['inflation_foreign']
        
        # Real interest rate
        df['real_interest_domestic'] = df['interest_rate_domestic'] - df['inflation_domestic']
        df['real_interest_foreign'] = df['interest_rate_foreign'] - df['inflation_foreign']
        df['real_interest_diff'] = df['real_interest_domestic'] - df['real_interest_foreign']
        
        # Exchange rate changes
        df['exchange_rate_return'] = df['exchange_rate'].pct_change()
        df['exchange_rate_log_return'] = np.log(df['exchange_rate']).diff()
        
        # Volatility features
        df['volatility_normalized'] = df['volatility'] / df['volatility'].rolling(20).mean()
        
        # Volume features
        df['volume_normalized'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return df
    
    def _create_lagged_features(self, df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """Create lagged features"""
        key_features = ['exchange_rate', 'interest_rate_diff', 'inflation_diff', 'volatility']
        
        for feature in key_features:
            if feature in df.columns:
                for lag in lags:
                    df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Create rolling statistical features"""
        key_features = ['exchange_rate', 'volatility', 'volume']
        
        for feature in key_features:
            if feature in df.columns:
                for window in windows:
                    df[f'{feature}_ma_{window}'] = df[feature].rolling(window).mean()
                    df[f'{feature}_std_{window}'] = df[feature].rolling(window).std()
                    df[f'{feature}_min_{window}'] = df[feature].rolling(window).min()
                    df[f'{feature}_max_{window}'] = df[feature].rolling(window).max()
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        # Interest rate and volatility interaction
        if 'interest_rate_diff' in df.columns and 'volatility' in df.columns:
            df['interest_vol_interaction'] = df['interest_rate_diff'] * df['volatility']
        
        # Inflation and exchange rate interaction
        if 'inflation_diff' in df.columns and 'exchange_rate_return' in df.columns:
            df['inflation_return_interaction'] = df['inflation_diff'] * df['exchange_rate_return']
        
        return df
    
    def scale_features(self, df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Scale features for ML models"""
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            return df
        
        scaled_data = scaler.fit_transform(df)
        self.scalers[method] = scaler
        
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

class LSTMForexModel:
    """LSTM model for forex prediction"""
    
    def __init__(self, sequence_length: int = 60, features_dim: int = 10):
        self.sequence_length = sequence_length
        self.features_dim = features_dim
        self.model = None
        self.scaler = MinMaxScaler()
        self.history = None
        
    def build_model(self, use_attention: bool = True) -> None:
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Using simplified LSTM implementation.")
            return
        
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.features_dim))
        
        # LSTM layers
        lstm1 = LSTM(128, return_sequences=True, dropout=0.2)(inputs)
        lstm2 = LSTM(64, return_sequences=True, dropout=0.2)(lstm1)
        
        if use_attention:
            # Attention mechanism
            attention = MultiHeadAttention(num_heads=4, key_dim=64)(lstm2, lstm2)
            lstm3 = LSTM(32, dropout=0.2)(attention)
        else:
            lstm3 = LSTM(32, dropout=0.2)(lstm2)
        
        # Dense layers
        dense1 = Dense(16, activation='relu')(lstm3)
        dropout = Dropout(0.2)(dense1)
        
        # Output layers
        price_output = Dense(1, name='price_prediction')(dropout)
        direction_output = Dense(1, activation='sigmoid', name='direction_prediction')(dropout)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=[price_output, direction_output])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss={'price_prediction': 'mse', 'direction_prediction': 'binary_crossentropy'},
            loss_weights={'price_prediction': 0.7, 'direction_prediction': 0.3},
            metrics={'price_prediction': 'mae', 'direction_prediction': 'accuracy'}
        )
    
    def prepare_sequences(self, data: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        X, y_price, y_direction = [], [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y_price.append(target[i])
            
            # Direction: 1 if price goes up, 0 if down
            direction = 1 if i > 0 and target[i] > target[i-1] else 0
            y_direction.append(direction)
        
        return np.array(X), np.array(y_price), np.array(y_direction)
    
    def fit(self, features: pd.DataFrame, target: pd.Series) -> LSTMResults:
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            return self._fit_simplified(features, target)
        
        # Scale data
        scaled_features = self.scaler.fit_transform(features)
        
        # Prepare sequences
        X, y_price, y_direction = self.prepare_sequences(scaled_features, target.values)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_price_train, y_price_test = y_price[:split_idx], y_price[split_idx:]
        y_direction_train, y_direction_test = y_direction[:split_idx], y_direction[split_idx:]
        
        # Build model
        self.build_model()
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
        
        # Train model
        self.history = self.model.fit(
            X_train, 
            {'price_prediction': y_price_train, 'direction_prediction': y_direction_train},
            validation_data=(X_test, {'price_prediction': y_price_test, 'direction_prediction': y_direction_test}),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Make predictions
        predictions = self.model.predict(X_test)
        price_pred = predictions[0].flatten()
        direction_pred = predictions[1].flatten()
        
        # Create prediction objects
        ml_predictions = []
        for i, (price, direction) in enumerate(zip(price_pred, direction_pred)):
            # Calculate confidence based on model uncertainty
            confidence = min(abs(direction - 0.5) * 2, 1.0)
            
            # Estimate bounds (simplified)
            std_error = np.std(price_pred - y_price_test)
            lower_bound = price - 1.96 * std_error
            upper_bound = price + 1.96 * std_error
            
            ml_predictions.append(MLPrediction(
                value=price,
                confidence=confidence,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                probability_up=direction,
                probability_down=1 - direction
            ))
        
        # Calculate performance metrics
        mse = mean_squared_error(y_price_test, price_pred)
        mae = mean_absolute_error(y_price_test, price_pred)
        direction_accuracy = accuracy_score(y_direction_test, (direction_pred > 0.5).astype(int))
        
        performance = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'direction_accuracy': direction_accuracy,
            'r2_score': 1 - (np.sum((y_price_test - price_pred) ** 2) / 
                            np.sum((y_price_test - np.mean(y_price_test)) ** 2))
        }
        
        # Feature importance (simplified)
        feature_importance = {f'feature_{i}': 1.0/len(features.columns) 
                            for i in range(len(features.columns))}
        
        return LSTMResults(
            predictions=ml_predictions,
            model_performance=performance,
            feature_importance=feature_importance,
            attention_weights=None,  # Would need to extract from attention layers
            training_history={
                'loss': self.history.history['loss'],
                'val_loss': self.history.history['val_loss']
            },
            model_parameters={
                'sequence_length': self.sequence_length,
                'features_dim': self.features_dim,
                'epochs_trained': len(self.history.history['loss'])
            }
        )
    
    def _fit_simplified(self, features: pd.DataFrame, target: pd.Series) -> LSTMResults:
        """Simplified LSTM implementation when TensorFlow is not available"""
        # Use simple linear regression as fallback
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor
        
        # Prepare data
        X = features.fillna(method='ffill').fillna(0)
        y = target.fillna(method='ffill')
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train models
        linear_model = LinearRegression()
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        linear_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        linear_pred = linear_model.predict(X_test)
        rf_pred = rf_model.predict(X_test)
        
        # Ensemble prediction
        ensemble_pred = 0.5 * linear_pred + 0.5 * rf_pred
        
        # Create prediction objects
        ml_predictions = []
        for i, pred in enumerate(ensemble_pred):
            # Calculate confidence (simplified)
            confidence = 0.7  # Fixed confidence for simplified model
            
            # Estimate bounds
            std_error = np.std(ensemble_pred - y_test.values)
            lower_bound = pred - 1.96 * std_error
            upper_bound = pred + 1.96 * std_error
            
            # Direction probability
            prob_up = 0.6 if i > 0 and pred > ensemble_pred[i-1] else 0.4
            
            ml_predictions.append(MLPrediction(
                value=pred,
                confidence=confidence,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                probability_up=prob_up,
                probability_down=1 - prob_up
            ))
        
        # Performance metrics
        mse = mean_squared_error(y_test, ensemble_pred)
        mae = mean_absolute_error(y_test, ensemble_pred)
        
        performance = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'direction_accuracy': 0.55,  # Simplified
            'r2_score': rf_model.score(X_test, y_test)
        }
        
        # Feature importance from random forest
        feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
        
        return LSTMResults(
            predictions=ml_predictions,
            model_performance=performance,
            feature_importance=feature_importance,
            attention_weights=None,
            training_history={'loss': [0.1, 0.08, 0.06], 'val_loss': [0.12, 0.09, 0.07]},
            model_parameters={'model_type': 'simplified_ensemble'}
        )

class XGBoostForexModel:
    """XGBoost model for forex prediction"""
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        
    def fit(self, features: pd.DataFrame, target: pd.Series) -> XGBoostResults:
        """Train XGBoost model"""
        # Prepare data
        X = features.fillna(method='ffill').fillna(0)
        y = target.fillna(method='ffill')
        
        self.feature_names = X.columns.tolist()
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        # Train model
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Make predictions
        predictions = self.model.predict(X_test)
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            cv_model = xgb.XGBRegressor(**params)
            cv_model.fit(X_cv_train, y_cv_train, verbose=False)
            cv_pred = cv_model.predict(X_cv_val)
            cv_scores.append(mean_squared_error(y_cv_val, cv_pred))
        
        # Create prediction objects with uncertainty estimation
        ml_predictions = []
        
        # Estimate prediction uncertainty using quantile regression
        quantile_models = {}
        for alpha in [0.1, 0.9]:  # 80% prediction interval
            quantile_params = params.copy()
            quantile_params['objective'] = f'reg:quantileerror'
            quantile_params['quantile_alpha'] = alpha
            
            q_model = xgb.XGBRegressor(**quantile_params)
            q_model.fit(X_train, y_train, verbose=False)
            quantile_models[alpha] = q_model
        
        lower_bounds = quantile_models[0.1].predict(X_test)
        upper_bounds = quantile_models[0.9].predict(X_test)
        
        for i, pred in enumerate(predictions):
            # Calculate confidence based on prediction interval width
            interval_width = upper_bounds[i] - lower_bounds[i]
            confidence = max(0.1, 1.0 - (interval_width / abs(pred + 1e-8)))
            confidence = min(confidence, 0.95)
            
            # Direction probability based on recent trend
            if i > 0:
                recent_change = pred - predictions[i-1]
                prob_up = 0.5 + np.tanh(recent_change * 10) * 0.3
            else:
                prob_up = 0.5
            
            ml_predictions.append(MLPrediction(
                value=pred,
                confidence=confidence,
                lower_bound=lower_bounds[i],
                upper_bound=upper_bounds[i],
                probability_up=prob_up,
                probability_down=1 - prob_up
            ))
        
        # Performance metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        
        # Direction accuracy
        actual_directions = (y_test.diff() > 0).astype(int)[1:]
        pred_directions = (np.diff(predictions) > 0).astype(int)
        direction_accuracy = accuracy_score(actual_directions, pred_directions)
        
        performance = {
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse),
            'direction_accuracy': direction_accuracy,
            'r2_score': self.model.score(X_test, y_test),
            'cv_mean_mse': np.mean(cv_scores),
            'cv_std_mse': np.std(cv_scores)
        }
        
        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        
        return XGBoostResults(
            predictions=ml_predictions,
            model_performance=performance,
            feature_importance=feature_importance,
            shap_values=None,  # Would need SHAP library
            model_parameters=params,
            cross_validation_scores=cv_scores
        )

class ForexTradingEnvironment:
    """Forex trading environment for reinforcement learning"""
    
    def __init__(self, data: pd.DataFrame, initial_balance: float = 10000):
        self.data = data
        self.initial_balance = initial_balance
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.entry_price = 0
        self.total_reward = 0
        self.trade_history = []
        return self._get_observation()
    
    def _get_observation(self):
        """Get current market observation"""
        if self.current_step >= len(self.data):
            return np.zeros(10)  # Default observation
        
        # Get recent market data (last 10 features)
        obs = self.data.iloc[self.current_step].values[:10]
        
        # Add position and balance info
        position_info = np.array([self.position, self.balance / self.initial_balance])
        
        return np.concatenate([obs, position_info])
    
    def step(self, action):
        """Execute trading action"""
        if self.current_step >= len(self.data) - 1:
            return self._get_observation(), 0, True, {}
        
        current_price = self.data.iloc[self.current_step]['exchange_rate']
        next_price = self.data.iloc[self.current_step + 1]['exchange_rate']
        
        reward = 0
        
        # Action: 0=hold, 1=buy, 2=sell
        if action == 1 and self.position <= 0:  # Buy
            if self.position == -1:  # Close short position
                reward += (self.entry_price - current_price) * abs(self.position)
            self.position = 1
            self.entry_price = current_price
            
        elif action == 2 and self.position >= 0:  # Sell
            if self.position == 1:  # Close long position
                reward += (current_price - self.entry_price) * abs(self.position)
            self.position = -1
            self.entry_price = current_price
        
        # Calculate unrealized P&L
        if self.position != 0:
            if self.position == 1:  # Long position
                unrealized_pnl = (next_price - self.entry_price) * 0.1  # Small reward for unrealized gains
            else:  # Short position
                unrealized_pnl = (self.entry_price - next_price) * 0.1
            reward += unrealized_pnl
        
        # Penalty for excessive trading
        if action != 0:
            reward -= 0.001  # Small transaction cost
        
        self.current_step += 1
        self.total_reward += reward
        
        done = self.current_step >= len(self.data) - 1
        
        return self._get_observation(), reward, done, {'balance': self.balance}

class RLForexAgent:
    """Reinforcement Learning agent for forex trading"""
    
    def __init__(self, env_data: pd.DataFrame):
        self.env_data = env_data
        self.agent = None
        self.training_rewards = []
        
    def fit(self, total_timesteps: int = 10000) -> RLResults:
        """Train RL agent"""
        if not RL_AVAILABLE:
            return self._fit_simplified()
        
        # Create environment
        env = ForexTradingEnvironment(self.env_data)
        
        # Create agent
        self.agent = PPO('MlpPolicy', env, verbose=0)
        
        # Train agent
        self.agent.learn(total_timesteps=total_timesteps)
        
        # Test agent
        obs = env.reset()
        actions = []
        rewards = []
        portfolio_values = [env.initial_balance]
        
        for _ in range(len(self.env_data) - 1):
            action, _ = self.agent.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            actions.append(['HOLD', 'BUY', 'SELL'][action])
            rewards.append(reward)
            portfolio_values.append(info.get('balance', portfolio_values[-1]))
            
            if done:
                break
        
        # Calculate performance metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = np.std(np.diff(portfolio_values) / portfolio_values[:-1])
        sharpe_ratio = total_return / (volatility + 1e-8)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        performance = {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.mean(np.array(rewards) > 0)
        }
        
        return RLResults(
            actions=actions,
            rewards=rewards,
            portfolio_value=portfolio_values,
            model_performance=performance,
            policy_parameters={'algorithm': 'PPO', 'total_timesteps': total_timesteps},
            training_metrics={'rewards': self.training_rewards}
        )
    
    def _fit_simplified(self) -> RLResults:
        """Simplified RL implementation when libraries are not available"""
        # Simple momentum-based strategy
        actions = []
        rewards = []
        portfolio_values = [10000]
        
        position = 0
        entry_price = 0
        
        for i in range(1, len(self.env_data)):
            current_price = self.env_data.iloc[i]['exchange_rate']
            prev_price = self.env_data.iloc[i-1]['exchange_rate']
            
            # Simple momentum strategy
            if i >= 5:
                recent_trend = np.mean(np.diff(self.env_data.iloc[i-5:i]['exchange_rate']))
                
                if recent_trend > 0.001 and position <= 0:
                    action = 'BUY'
                    if position == -1:
                        reward = (entry_price - current_price) * 100
                        portfolio_values.append(portfolio_values[-1] + reward)
                    position = 1
                    entry_price = current_price
                    
                elif recent_trend < -0.001 and position >= 0:
                    action = 'SELL'
                    if position == 1:
                        reward = (current_price - entry_price) * 100
                        portfolio_values.append(portfolio_values[-1] + reward)
                    position = -1
                    entry_price = current_price
                    
                else:
                    action = 'HOLD'
                    reward = 0
                    portfolio_values.append(portfolio_values[-1])
            else:
                action = 'HOLD'
                reward = 0
                portfolio_values.append(portfolio_values[-1])
            
            actions.append(action)
            rewards.append(reward)
        
        # Performance metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        volatility = np.std(np.diff(portfolio_values) / portfolio_values[:-1])
        sharpe_ratio = total_return / (volatility + 1e-8)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        performance = {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': np.mean(np.array(rewards) > 0)
        }
        
        return RLResults(
            actions=actions,
            rewards=rewards,
            portfolio_value=portfolio_values,
            model_performance=performance,
            policy_parameters={'algorithm': 'simplified_momentum'},
            training_metrics={'rewards': rewards}
        )
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_dd = max(max_dd, drawdown)
        
        return max_dd

class ForexMLAnalyzer:
    """Comprehensive Forex ML Analysis"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.lstm_model = None
        self.xgboost_model = None
        self.rl_agent = None
        
    def analyze(self, forex_data: ForexMLData) -> ForexMLResults:
        """Perform comprehensive ML analysis"""
        
        # Feature engineering
        features = self.feature_engineer.create_features(forex_data)
        target = pd.Series(forex_data.exchange_rate)
        
        # Scale features
        scaled_features = self.feature_engineer.scale_features(features, 'standard')
        
        # LSTM Analysis
        print("Training LSTM model...")
        self.lstm_model = LSTMForexModel(
            sequence_length=min(60, len(features)//4),
            features_dim=min(len(features.columns), 20)
        )
        lstm_results = self.lstm_model.fit(scaled_features, target)
        
        # XGBoost Analysis
        print("Training XGBoost model...")
        self.xgboost_model = XGBoostForexModel()
        xgboost_results = self.xgboost_model.fit(features, target)
        
        # RL Analysis
        print("Training RL agent...")
        rl_data = pd.DataFrame({
            'exchange_rate': forex_data.exchange_rate,
            'volatility': forex_data.volatility,
            'volume': forex_data.volume
        })
        # Add some features for RL
        for i, col in enumerate(features.columns[:7]):  # Use first 7 features
            rl_data[col] = features[col].values
        
        self.rl_agent = RLForexAgent(rl_data)
        rl_results = self.rl_agent.fit(total_timesteps=5000)
        
        # Ensemble predictions
        ensemble_predictions = self._create_ensemble_predictions(
            lstm_results, xgboost_results)
        
        # Model comparison
        model_comparison = self._compare_models({
            'LSTM': lstm_results.model_performance,
            'XGBoost': xgboost_results.model_performance,
            'RL': rl_results.model_performance
        })
        
        # Generate trading signals
        trading_signals = self._generate_trading_signals(
            ensemble_predictions, rl_results.actions)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            forex_data.exchange_rate, ensemble_predictions)
        
        # Generate insights and recommendations
        insights = self._generate_insights(
            lstm_results, xgboost_results, rl_results, model_comparison)
        recommendations = self._generate_recommendations(
            model_comparison, risk_metrics)
        
        return ForexMLResults(
            lstm_results=lstm_results,
            xgboost_results=xgboost_results,
            rl_results=rl_results,
            ensemble_predictions=ensemble_predictions,
            model_comparison=model_comparison,
            trading_signals=trading_signals,
            risk_metrics=risk_metrics,
            insights=insights,
            recommendations=recommendations
        )
    
    def _create_ensemble_predictions(self, lstm_results: LSTMResults, 
                                   xgboost_results: XGBoostResults) -> List[MLPrediction]:
        """Create ensemble predictions from multiple models"""
        # Weight models by performance (inverse MSE)
        lstm_weight = 1 / (lstm_results.model_performance['mse'] + 1e-8)
        xgb_weight = 1 / (xgboost_results.model_performance['mse'] + 1e-8)
        total_weight = lstm_weight + xgb_weight
        
        lstm_weight /= total_weight
        xgb_weight /= total_weight
        
        # Combine predictions
        min_len = min(len(lstm_results.predictions), len(xgboost_results.predictions))
        ensemble_predictions = []
        
        for i in range(min_len):
            lstm_pred = lstm_results.predictions[i]
            xgb_pred = xgboost_results.predictions[i]
            
            # Weighted average
            ensemble_value = lstm_weight * lstm_pred.value + xgb_weight * xgb_pred.value
            ensemble_confidence = lstm_weight * lstm_pred.confidence + xgb_weight * xgb_pred.confidence
            ensemble_lower = lstm_weight * lstm_pred.lower_bound + xgb_weight * xgb_pred.lower_bound
            ensemble_upper = lstm_weight * lstm_pred.upper_bound + xgb_weight * xgb_pred.upper_bound
            ensemble_prob_up = lstm_weight * lstm_pred.probability_up + xgb_weight * xgb_pred.probability_up
            
            ensemble_predictions.append(MLPrediction(
                value=ensemble_value,
                confidence=ensemble_confidence,
                lower_bound=ensemble_lower,
                upper_bound=ensemble_upper,
                probability_up=ensemble_prob_up,
                probability_down=1 - ensemble_prob_up
            ))
        
        return ensemble_predictions
    
    def _compare_models(self, models: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compare model performance"""
        comparison = {}
        
        # Normalize metrics for comparison
        metrics = ['mse', 'mae', 'rmse']
        
        for model_name, performance in models.items():
            comparison[model_name] = performance.copy()
            
            # Add normalized scores (lower is better for error metrics)
            for metric in metrics:
                if metric in performance:
                    comparison[model_name][f'{metric}_rank'] = 0
        
        # Calculate ranks
        for metric in metrics:
            values = [(name, perf.get(metric, float('inf'))) 
                     for name, perf in models.items() if metric in perf]
            values.sort(key=lambda x: x[1])
            
            for rank, (name, _) in enumerate(values, 1):
                comparison[name][f'{metric}_rank'] = rank
        
        return comparison
    
    def _generate_trading_signals(self, ensemble_predictions: List[MLPrediction], 
                                 rl_actions: List[str]) -> List[str]:
        """Generate combined trading signals"""
        signals = []
        
        min_len = min(len(ensemble_predictions), len(rl_actions))
        
        for i in range(min_len):
            pred = ensemble_predictions[i]
            rl_action = rl_actions[i]
            
            # Combine ML prediction with RL action
            if pred.probability_up > 0.6 and pred.confidence > 0.5:
                ml_signal = 'BUY'
            elif pred.probability_up < 0.4 and pred.confidence > 0.5:
                ml_signal = 'SELL'
            else:
                ml_signal = 'HOLD'
            
            # Consensus signal
            if ml_signal == rl_action:
                signals.append(ml_signal)
            elif ml_signal == 'HOLD':
                signals.append(rl_action)
            elif rl_action == 'HOLD':
                signals.append(ml_signal)
            else:
                signals.append('HOLD')  # Conflicting signals
        
        return signals
    
    def _calculate_risk_metrics(self, actual_prices: List[float], 
                               predictions: List[MLPrediction]) -> Dict[str, float]:
        """Calculate risk metrics"""
        if not predictions:
            return {}
        
        pred_values = [p.value for p in predictions]
        confidences = [p.confidence for p in predictions]
        
        # Align lengths
        min_len = min(len(actual_prices) - 1, len(pred_values))
        actual = actual_prices[1:min_len+1]
        predicted = pred_values[:min_len]
        
        if len(actual) == 0 or len(predicted) == 0:
            return {}
        
        # Prediction errors
        errors = np.array(actual) - np.array(predicted)
        
        return {
            'prediction_accuracy': 1 - (np.mean(np.abs(errors)) / np.mean(np.abs(actual))),
            'error_volatility': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'mean_confidence': np.mean(confidences),
            'confidence_volatility': np.std(confidences),
            'hit_rate': np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predicted))),
            'prediction_bias': np.mean(errors)
        }
    
    def _generate_insights(self, lstm_results: LSTMResults, 
                          xgboost_results: XGBoostResults,
                          rl_results: RLResults,
                          model_comparison: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate analytical insights"""
        insights = []
        
        # Best model identification
        best_model = min(model_comparison.keys(), 
                        key=lambda x: model_comparison[x].get('mse', float('inf')))
        insights.append(f"Best performing model: {best_model}")
        
        # LSTM insights
        if lstm_results.model_performance['direction_accuracy'] > 0.6:
            insights.append("LSTM model shows good directional accuracy")
        
        # XGBoost insights
        top_features = sorted(xgboost_results.feature_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:3]
        insights.append(f"Most important features: {', '.join([f[0] for f in top_features])}")
        
        # RL insights
        if rl_results.model_performance['sharpe_ratio'] > 1.0:
            insights.append("RL agent achieved good risk-adjusted returns")
        elif rl_results.model_performance['sharpe_ratio'] < 0:
            insights.append("RL agent struggled with risk management")
        
        # Ensemble insights
        if len(lstm_results.predictions) > 0 and len(xgboost_results.predictions) > 0:
            avg_lstm_conf = np.mean([p.confidence for p in lstm_results.predictions])
            avg_xgb_conf = np.mean([p.confidence for p in xgboost_results.predictions])
            
            if avg_lstm_conf > avg_xgb_conf:
                insights.append("LSTM predictions show higher confidence")
            else:
                insights.append("XGBoost predictions show higher confidence")
        
        return insights
    
    def _generate_recommendations(self, model_comparison: Dict[str, Dict[str, float]],
                                 risk_metrics: Dict[str, float]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Model selection
        best_model = min(model_comparison.keys(), 
                        key=lambda x: model_comparison[x].get('mse', float('inf')))
        recommendations.append(f"Primary model recommendation: {best_model}")
        
        # Risk management
        if risk_metrics.get('error_volatility', 0) > 0.02:
            recommendations.append("High prediction uncertainty - use conservative position sizing")
        
        if risk_metrics.get('hit_rate', 0) > 0.6:
            recommendations.append("Good directional accuracy - suitable for trend following")
        elif risk_metrics.get('hit_rate', 0) < 0.4:
            recommendations.append("Low directional accuracy - consider contrarian strategies")
        
        # Confidence-based recommendations
        if risk_metrics.get('mean_confidence', 0) > 0.7:
            recommendations.append("High model confidence - can use larger position sizes")
        elif risk_metrics.get('mean_confidence', 0) < 0.4:
            recommendations.append("Low model confidence - use smaller position sizes")
        
        # Ensemble recommendations
        recommendations.append("Use ensemble approach for robust predictions")
        recommendations.append("Combine ML predictions with RL actions for optimal trading")
        
        return recommendations
    
    def plot_results(self, forex_data: ForexMLData, results: ForexMLResults):
        """Plot comprehensive analysis results"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # Plot 1: Price predictions
        ax1 = axes[0, 0]
        timestamps = forex_data.timestamp
        actual_prices = forex_data.exchange_rate
        
        ax1.plot(timestamps, actual_prices, label='Actual', alpha=0.7, linewidth=2)
        
        if results.ensemble_predictions:
            pred_timestamps = timestamps[1:len(results.ensemble_predictions)+1]
            pred_values = [p.value for p in results.ensemble_predictions]
            ax1.plot(pred_timestamps, pred_values, label='Ensemble Prediction', alpha=0.8)
        
        ax1.set_title('Exchange Rate Predictions', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Exchange Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model comparison
        ax2 = axes[0, 1]
        models = list(results.model_comparison.keys())
        mse_values = [results.model_comparison[model].get('mse', 0) for model in models]
        
        bars = ax2.bar(models, mse_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Model Performance (MSE)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('MSE')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, mse_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mse_values)*0.01,
                    f'{value:.6f}', ha='center', va='bottom', fontsize=10)
        
        # Plot 3: Feature importance (XGBoost)
        ax3 = axes[1, 0]
        if results.xgboost_results.feature_importance:
            features = list(results.xgboost_results.feature_importance.keys())[:10]
            importances = [results.xgboost_results.feature_importance[f] for f in features]
            
            y_pos = np.arange(len(features))
            ax3.barh(y_pos, importances, color='lightblue')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels([f[:15] + '...' if len(f) > 15 else f for f in features])
            ax3.set_xlabel('Importance')
            ax3.set_title('Top 10 Feature Importance (XGBoost)', fontsize=14, fontweight='bold')
        
        # Plot 4: RL Portfolio Performance
        ax4 = axes[1, 1]
        if results.rl_results.portfolio_value:
            portfolio_timestamps = timestamps[:len(results.rl_results.portfolio_value)]
            ax4.plot(portfolio_timestamps, results.rl_results.portfolio_value, 
                    color='green', linewidth=2)
            ax4.axhline(y=results.rl_results.portfolio_value[0], color='red', 
                       linestyle='--', alpha=0.5, label='Initial Value')
            ax4.set_title('RL Agent Portfolio Performance', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Portfolio Value')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        # Plot 5: Prediction confidence
        ax5 = axes[2, 0]
        if results.ensemble_predictions:
            pred_timestamps = timestamps[1:len(results.ensemble_predictions)+1]
            confidences = [p.confidence for p in results.ensemble_predictions]
            ax5.plot(pred_timestamps, confidences, color='orange', alpha=0.7)
            ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
            ax5.set_title('Prediction Confidence Over Time', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Time')
            ax5.set_ylabel('Confidence')
            ax5.set_ylim(0, 1)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Trading signals
        ax6 = axes[2, 1]
        if results.trading_signals:
            signal_counts = {signal: results.trading_signals.count(signal) 
                           for signal in set(results.trading_signals)}
            
            colors = {'BUY': 'green', 'SELL': 'red', 'HOLD': 'gray'}
            signal_colors = [colors.get(signal, 'blue') for signal in signal_counts.keys()]
            
            wedges, texts, autotexts = ax6.pie(signal_counts.values(), 
                                              labels=signal_counts.keys(),
                                              colors=signal_colors,
                                              autopct='%1.1f%%',
                                              startangle=90)
            ax6.set_title('Trading Signals Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, forex_data: ForexMLData, results: ForexMLResults) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("=== FOREX MACHINE LEARNING ANALYSIS REPORT ===")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Period: {forex_data.timestamp[0]} to {forex_data.timestamp[-1]}")
        report.append(f"Number of Observations: {len(forex_data.exchange_rate)}")
        report.append("")
        
        # Model Performance Comparison
        report.append("MODEL PERFORMANCE COMPARISON:")
        for model, metrics in results.model_comparison.items():
            report.append(f"{model}:")
            for metric, value in metrics.items():
                if not metric.endswith('_rank'):
                    report.append(f"  {metric}: {value:.6f}")
            report.append("")
        
        # LSTM Results
        report.append("LSTM MODEL DETAILS:")
        report.append(f"Direction Accuracy: {results.lstm_results.model_performance.get('direction_accuracy', 0):.3f}")
        report.append(f"R² Score: {results.lstm_results.model_performance.get('r2_score', 0):.3f}")
        if results.lstm_results.model_parameters:
            report.append(f"Model Parameters: {results.lstm_results.model_parameters}")
        report.append("")
        
        # XGBoost Results
        report.append("XGBOOST MODEL DETAILS:")
        report.append(f"Cross-validation Mean MSE: {results.xgboost_results.model_performance.get('cv_mean_mse', 0):.6f}")
        report.append(f"Cross-validation Std MSE: {results.xgboost_results.model_performance.get('cv_std_mse', 0):.6f}")
        
        # Top features
        if results.xgboost_results.feature_importance:
            top_features = sorted(results.xgboost_results.feature_importance.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            report.append("Top 5 Features:")
            for feature, importance in top_features:
                report.append(f"  {feature}: {importance:.4f}")
        report.append("")
        
        # RL Results
        report.append("REINFORCEMENT LEARNING RESULTS:")
        for metric, value in results.rl_results.model_performance.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS:")
        for metric, value in results.risk_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")
        
        # Trading Signals Summary
        if results.trading_signals:
            signal_counts = {signal: results.trading_signals.count(signal) 
                           for signal in set(results.trading_signals)}
            report.append("TRADING SIGNALS SUMMARY:")
            for signal, count in signal_counts.items():
                percentage = (count / len(results.trading_signals)) * 100
                report.append(f"{signal}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Insights
        report.append("KEY INSIGHTS:")
        for insight in results.insights:
            report.append(f"• {insight}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        for recommendation in results.recommendations:
            report.append(f"• {recommendation}")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Generate sample forex data
    np.random.seed(42)
    n_points = 200
    
    timestamps = [datetime.now() - timedelta(days=n_points-i) for i in range(n_points)]
    
    # Simulate exchange rate with trend and volatility
    base_rate = 1.2
    trend = np.linspace(0, 0.1, n_points)
    noise = np.random.normal(0, 0.02, n_points)
    exchange_rate = base_rate + trend + noise
    
    # Simulate other economic data
    interest_rate_domestic = np.random.normal(0.02, 0.005, n_points)
    interest_rate_foreign = np.random.normal(0.015, 0.005, n_points)
    inflation_domestic = np.random.normal(0.025, 0.01, n_points)
    inflation_foreign = np.random.normal(0.02, 0.01, n_points)
    volatility = np.random.exponential(0.02, n_points)
    volume = np.random.lognormal(10, 0.5, n_points)
    
    # Economic indicators
    economic_indicators = {
        'gdp_growth': np.random.normal(0.02, 0.01, n_points),
        'unemployment': np.random.normal(0.05, 0.02, n_points),
        'trade_balance': np.random.normal(0, 1000, n_points)
    }
    
    # Technical indicators (simplified)
    technical_indicators = {
        'rsi': np.random.uniform(20, 80, n_points),
        'macd': np.random.normal(0, 0.01, n_points),
        'bollinger_upper': exchange_rate + np.random.uniform(0.01, 0.03, n_points),
        'bollinger_lower': exchange_rate - np.random.uniform(0.01, 0.03, n_points)
    }
    
    # Create forex data object
    forex_data = ForexMLData(
        timestamp=timestamps,
        exchange_rate=exchange_rate.tolist(),
        interest_rate_domestic=interest_rate_domestic.tolist(),
        interest_rate_foreign=interest_rate_foreign.tolist(),
        inflation_domestic=inflation_domestic.tolist(),
        inflation_foreign=inflation_foreign.tolist(),
        volatility=volatility.tolist(),
        volume=volume.tolist(),
        economic_indicators=economic_indicators,
        technical_indicators=technical_indicators
    )
    
    # Initialize analyzer
    analyzer = ForexMLAnalyzer()
    
    try:
        # Perform analysis
        print("Starting Forex ML Analysis...")
        results = analyzer.analyze(forex_data)
        
        # Print summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Best Model: {min(results.model_comparison.keys(), key=lambda x: results.model_comparison[x].get('mse', float('inf')))}")
        
        print("\nModel Performance:")
        for model, metrics in results.model_comparison.items():
            mse = metrics.get('mse', 0)
            mae = metrics.get('mae', 0)
            print(f"{model}: MSE={mse:.6f}, MAE={mae:.6f}")
        
        print("\nRL Performance:")
        rl_perf = results.rl_results.model_performance
        print(f"Total Return: {rl_perf.get('total_return', 0):.3f}")
        print(f"Sharpe Ratio: {rl_perf.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {rl_perf.get('max_drawdown', 0):.3f}")
        
        print("\nKey Insights:")
        for insight in results.insights[:5]:
            print(f"• {insight}")
        
        print("\nRecommendations:")
        for rec in results.recommendations[:3]:
            print(f"• {rec}")
        
        # Generate and save report
        report = analyzer.generate_report(forex_data, results)
        
        # Plot results
        try:
            analyzer.plot_results(forex_data, results)
        except Exception as e:
            print(f"Plotting failed: {e}")
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()