# Kronos Component
# Time-Series Forecasting and Predictive Analytics for Financial Data

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from datetime import datetime, timedelta
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    tf = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None

from .finvis_gpt import OHLCData, IndicatorData, ChartFeatures

logger = logging.getLogger(__name__)

class ForecastType(Enum):
    """Types of forecasts"""
    PRICE = "price_forecast"
    VOLATILITY = "volatility_forecast"
    TREND = "trend_forecast"
    VOLUME = "volume_forecast"
    SUPPORT_RESISTANCE = "support_resistance_forecast"
    CUSTOM_INDICATOR = "custom_indicator_forecast"

class ModelType(Enum):
    """Types of forecasting models"""
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    ENSEMBLE = "ensemble"
    ARIMA = "arima"
    PROPHET = "prophet"

class TrendDirection(Enum):
    """Trend direction classifications"""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

@dataclass
class ForecastScenario:
    """Individual forecast scenario"""
    scenario_name: str
    probability: float
    price_targets: List[float]
    timeframes: List[str]
    confidence_intervals: Dict[str, Tuple[float, float]]
    key_levels: Dict[str, float]
    risk_factors: List[str]
    catalysts: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class VolatilityForecast:
    """Volatility prediction data"""
    predicted_volatility: List[float]
    volatility_regime: str  # low, medium, high
    volatility_percentiles: Dict[str, float]
    expected_range: Tuple[float, float]
    breakout_probability: float
    mean_reversion_probability: float
    timestamps: List[datetime]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrendForecast:
    """Trend prediction data"""
    trend_direction: TrendDirection
    trend_strength: float  # 0-1 scale
    trend_duration: timedelta
    reversal_probability: float
    continuation_probability: float
    key_inflection_points: List[Tuple[datetime, float]]
    trend_channels: Dict[str, Tuple[float, float]]
    momentum_indicators: Dict[str, float]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KronosOutput:
    """Complete Kronos forecasting output"""
    price_forecasts: List[ForecastScenario]
    volatility_forecast: VolatilityForecast
    trend_forecast: TrendForecast
    custom_indicator_forecasts: Dict[str, List[float]]
    model_performance: Dict[str, float]
    feature_importance: Dict[str, float]
    forecast_horizon: timedelta
    forecast_timestamp: datetime
    confidence_score: float
    risk_assessment: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)

class Kronos:
    """
    Kronos - Advanced Time-Series Forecasting Engine
    
    Sophisticated forecasting system that:
    - Consumes extracted OHLC data and technical indicators
    - Incorporates custom indicators as input features
    - Generates multi-horizon price, volatility, and trend forecasts
    - Provides scenario-based predictions with confidence intervals
    - Supports multiple ML/DL models and ensemble methods
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Model configuration
        self.model_config = {
            'sequence_length': self.config.get('sequence_length', 60),
            'forecast_horizon': self.config.get('forecast_horizon', 30),
            'ensemble_models': self.config.get('ensemble_models', ['lstm', 'random_forest', 'gradient_boosting']),
            'validation_split': self.config.get('validation_split', 0.2),
            'epochs': self.config.get('epochs', 100),
            'batch_size': self.config.get('batch_size', 32),
            'learning_rate': self.config.get('learning_rate', 0.001)
        }
        
        # Feature engineering configuration
        self.feature_config = {
            'technical_indicators': self.config.get('technical_indicators', True),
            'price_features': self.config.get('price_features', True),
            'volume_features': self.config.get('volume_features', True),
            'volatility_features': self.config.get('volatility_features', True),
            'custom_features': self.config.get('custom_features', True),
            'lag_features': self.config.get('lag_features', [1, 2, 3, 5, 10]),
            'rolling_windows': self.config.get('rolling_windows', [5, 10, 20, 50])
        }
        
        # Scenario configuration
        self.scenario_config = {
            'num_scenarios': self.config.get('num_scenarios', 3),
            'confidence_levels': self.config.get('confidence_levels', [0.68, 0.95, 0.99]),
            'monte_carlo_simulations': self.config.get('monte_carlo_simulations', 1000)
        }
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        # Initialize data storage
        self.training_data = None
        self.features = None
        self.targets = None
        
        logger.info("Kronos initialized with config: %s", self.config)
    
    async def generate_forecasts(self, chart_features: ChartFeatures, 
                               custom_indicators: Optional[Dict[str, List[float]]] = None) -> KronosOutput:
        """
        Main forecasting method.
        
        Args:
            chart_features: Extracted features from FinVis-GPT
            custom_indicators: Additional custom indicator data
            
        Returns:
            KronosOutput with comprehensive forecasts
        """
        try:
            logger.info("Starting Kronos forecasting pipeline")
            
            # Step 1: Prepare and engineer features
            features_df = await self._prepare_features(chart_features, custom_indicators)
            
            if features_df.empty:
                logger.warning("No features available for forecasting")
                return self._create_empty_output()
            
            # Step 2: Train/update models
            await self._train_models(features_df)
            
            # Step 3: Generate price forecasts with scenarios
            price_forecasts = await self._generate_price_forecasts(features_df)
            
            # Step 4: Generate volatility forecasts
            volatility_forecast = await self._generate_volatility_forecast(features_df)
            
            # Step 5: Generate trend forecasts
            trend_forecast = await self._generate_trend_forecast(features_df)
            
            # Step 6: Forecast custom indicators
            custom_forecasts = await self._forecast_custom_indicators(features_df, custom_indicators)
            
            # Step 7: Assess model performance
            performance_metrics = await self._assess_model_performance()
            
            # Step 8: Calculate feature importance
            feature_importance = await self._calculate_feature_importance()
            
            # Step 9: Perform risk assessment
            risk_assessment = await self._perform_risk_assessment(price_forecasts, volatility_forecast)
            
            # Step 10: Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(
                price_forecasts, volatility_forecast, trend_forecast, performance_metrics
            )
            
            # Compile results
            output = KronosOutput(
                price_forecasts=price_forecasts,
                volatility_forecast=volatility_forecast,
                trend_forecast=trend_forecast,
                custom_indicator_forecasts=custom_forecasts,
                model_performance=performance_metrics,
                feature_importance=feature_importance,
                forecast_horizon=timedelta(days=self.model_config['forecast_horizon']),
                forecast_timestamp=datetime.now(),
                confidence_score=confidence_score,
                risk_assessment=risk_assessment,
                metadata={
                    'model_config': self.model_config,
                    'feature_config': self.feature_config,
                    'data_points_used': len(features_df),
                    'models_used': list(self.models.keys())
                }
            )
            
            logger.info(f"Kronos forecasting completed with confidence: {confidence_score:.2f}")
            return output
            
        except Exception as e:
            logger.error(f"Kronos forecasting failed: {e}")
            return self._create_empty_output()
    
    async def _prepare_features(self, chart_features: ChartFeatures, 
                              custom_indicators: Optional[Dict[str, List[float]]]) -> pd.DataFrame:
        """Prepare and engineer features for forecasting"""
        try:
            logger.info("Preparing features for forecasting")
            
            # Convert OHLC data to DataFrame
            ohlc_df = self._ohlc_to_dataframe(chart_features.ohlc_data)
            
            if ohlc_df.empty:
                return pd.DataFrame()
            
            # Add technical indicators
            features_df = await self._add_technical_indicators(ohlc_df, chart_features.indicators)
            
            # Add price-based features
            features_df = self._add_price_features(features_df)
            
            # Add volatility features
            features_df = self._add_volatility_features(features_df)
            
            # Add volume features if available
            if chart_features.volume_data:
                features_df = self._add_volume_features(features_df, chart_features.volume_data)
            
            # Add custom indicators
            if custom_indicators:
                features_df = self._add_custom_indicators(features_df, custom_indicators)
            
            # Add lag features
            features_df = self._add_lag_features(features_df)
            
            # Add rolling window features
            features_df = self._add_rolling_features(features_df)
            
            # Add time-based features
            features_df = self._add_time_features(features_df)
            
            # Clean and validate features
            features_df = self._clean_features(features_df)
            
            logger.info(f"Prepared {len(features_df.columns)} features from {len(features_df)} data points")
            return features_df
            
        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return pd.DataFrame()
    
    def _ohlc_to_dataframe(self, ohlc_data: List[OHLCData]) -> pd.DataFrame:
        """Convert OHLC data to pandas DataFrame"""
        try:
            if not ohlc_data:
                return pd.DataFrame()
            
            data = []
            for ohlc in ohlc_data:
                data.append({
                    'timestamp': ohlc.timestamp,
                    'open': ohlc.open,
                    'high': ohlc.high,
                    'low': ohlc.low,
                    'close': ohlc.close,
                    'volume': ohlc.volume or 0
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"OHLC to DataFrame conversion failed: {e}")
            return pd.DataFrame()
    
    async def _add_technical_indicators(self, df: pd.DataFrame, indicators: List[IndicatorData]) -> pd.DataFrame:
        """Add technical indicators to features DataFrame"""
        try:
            features_df = df.copy()
            
            for indicator in indicators:
                if len(indicator.values) == len(df):
                    # Align indicator values with DataFrame index
                    indicator_series = pd.Series(indicator.values, index=df.index)
                    features_df[f"{indicator.name}"] = indicator_series
                    
                    # Add indicator metadata as features if available
                    if indicator.parameters:
                        for param_name, param_value in indicator.parameters.items():
                            if isinstance(param_value, (int, float)):
                                features_df[f"{indicator.name}_{param_name}"] = param_value
            
            return features_df
            
        except Exception as e:
            logger.error(f"Technical indicator addition failed: {e}")
            return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        try:
            features_df = df.copy()
            
            # Price changes
            features_df['price_change'] = features_df['close'].pct_change()
            features_df['price_change_abs'] = features_df['price_change'].abs()
            
            # High-Low spread
            features_df['hl_spread'] = (features_df['high'] - features_df['low']) / features_df['close']
            
            # Open-Close spread
            features_df['oc_spread'] = (features_df['close'] - features_df['open']) / features_df['open']
            
            # True Range
            features_df['true_range'] = np.maximum(
                features_df['high'] - features_df['low'],
                np.maximum(
                    abs(features_df['high'] - features_df['close'].shift(1)),
                    abs(features_df['low'] - features_df['close'].shift(1))
                )
            )
            
            # Price position within range
            features_df['price_position'] = (features_df['close'] - features_df['low']) / (features_df['high'] - features_df['low'])
            
            # Gap features
            features_df['gap'] = features_df['open'] - features_df['close'].shift(1)
            features_df['gap_pct'] = features_df['gap'] / features_df['close'].shift(1)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Price feature addition failed: {e}")
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        try:
            features_df = df.copy()
            
            # Calculate returns for volatility
            returns = features_df['close'].pct_change().dropna()
            
            # Rolling volatility (different windows)
            for window in [5, 10, 20, 50]:
                if len(returns) >= window:
                    vol_col = f'volatility_{window}d'
                    features_df[vol_col] = returns.rolling(window=window).std() * np.sqrt(252)
            
            # Parkinson volatility (using high-low)
            features_df['parkinson_vol'] = np.sqrt(
                (1 / (4 * np.log(2))) * np.log(features_df['high'] / features_df['low']) ** 2
            )
            
            # Garman-Klass volatility
            features_df['gk_vol'] = np.sqrt(
                0.5 * np.log(features_df['high'] / features_df['low']) ** 2 -
                (2 * np.log(2) - 1) * np.log(features_df['close'] / features_df['open']) ** 2
            )
            
            # Volatility regime indicators
            if 'volatility_20d' in features_df.columns:
                vol_20d = features_df['volatility_20d']
                vol_median = vol_20d.rolling(window=100).median()
                features_df['vol_regime'] = (vol_20d > vol_median).astype(int)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Volatility feature addition failed: {e}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame, volume_data: List[float]) -> pd.DataFrame:
        """Add volume-based features"""
        try:
            features_df = df.copy()
            
            if len(volume_data) == len(df):
                volume_series = pd.Series(volume_data, index=df.index)
                features_df['volume'] = volume_series
                
                # Volume features
                features_df['volume_change'] = features_df['volume'].pct_change()
                features_df['volume_ma_ratio'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
                
                # Price-Volume features
                features_df['price_volume'] = features_df['close'] * features_df['volume']
                features_df['vwap'] = (features_df['price_volume'].rolling(20).sum() / 
                                     features_df['volume'].rolling(20).sum())
                
                # On-Balance Volume
                price_change = features_df['close'].diff()
                obv = np.where(price_change > 0, features_df['volume'],
                              np.where(price_change < 0, -features_df['volume'], 0))
                features_df['obv'] = pd.Series(obv, index=df.index).cumsum()
            
            return features_df
            
        except Exception as e:
            logger.error(f"Volume feature addition failed: {e}")
            return df
    
    def _add_custom_indicators(self, df: pd.DataFrame, custom_indicators: Dict[str, List[float]]) -> pd.DataFrame:
        """Add custom indicators as features"""
        try:
            features_df = df.copy()
            
            for indicator_name, values in custom_indicators.items():
                if len(values) == len(df):
                    indicator_series = pd.Series(values, index=df.index)
                    features_df[f"custom_{indicator_name}"] = indicator_series
                    
                    # Add derived features from custom indicators
                    features_df[f"custom_{indicator_name}_change"] = indicator_series.pct_change()
                    features_df[f"custom_{indicator_name}_ma5"] = indicator_series.rolling(5).mean()
                    features_df[f"custom_{indicator_name}_std5"] = indicator_series.rolling(5).std()
            
            return features_df
            
        except Exception as e:
            logger.error(f"Custom indicator addition failed: {e}")
            return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged features"""
        try:
            features_df = df.copy()
            
            # Key columns to create lags for
            lag_columns = ['close', 'volume', 'price_change', 'volatility_20d']
            lag_columns = [col for col in lag_columns if col in features_df.columns]
            
            for col in lag_columns:
                for lag in self.feature_config['lag_features']:
                    features_df[f"{col}_lag_{lag}"] = features_df[col].shift(lag)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Lag feature addition failed: {e}")
            return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling window features"""
        try:
            features_df = df.copy()
            
            # Key columns for rolling features
            rolling_columns = ['close', 'volume', 'price_change']
            rolling_columns = [col for col in rolling_columns if col in features_df.columns]
            
            for col in rolling_columns:
                for window in self.feature_config['rolling_windows']:
                    if len(features_df) >= window:
                        features_df[f"{col}_ma_{window}"] = features_df[col].rolling(window).mean()
                        features_df[f"{col}_std_{window}"] = features_df[col].rolling(window).std()
                        features_df[f"{col}_min_{window}"] = features_df[col].rolling(window).min()
                        features_df[f"{col}_max_{window}"] = features_df[col].rolling(window).max()
            
            return features_df
            
        except Exception as e:
            logger.error(f"Rolling feature addition failed: {e}")
            return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            features_df = df.copy()
            
            # Extract time components
            features_df['hour'] = features_df.index.hour
            features_df['day_of_week'] = features_df.index.dayofweek
            features_df['day_of_month'] = features_df.index.day
            features_df['month'] = features_df.index.month
            features_df['quarter'] = features_df.index.quarter
            
            # Cyclical encoding for time features
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            features_df['dow_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['dow_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
            
            return features_df
            
        except Exception as e:
            logger.error(f"Time feature addition failed: {e}")
            return df
    
    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        try:
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill missing values
            df = df.fillna(method='ffill')
            
            # Drop remaining NaN values
            df = df.dropna()
            
            # Remove constant columns
            constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
            df = df.drop(columns=constant_columns)
            
            return df
            
        except Exception as e:
            logger.error(f"Feature cleaning failed: {e}")
            return df
    
    async def _train_models(self, features_df: pd.DataFrame) -> None:
        """Train forecasting models"""
        try:
            logger.info("Training forecasting models")
            
            if len(features_df) < self.model_config['sequence_length'] + 10:
                logger.warning("Insufficient data for model training")
                return
            
            # Prepare training data
            X, y = self._prepare_training_data(features_df)
            
            if X is None or y is None:
                logger.warning("Failed to prepare training data")
                return
            
            # Split data
            split_idx = int(len(X) * (1 - self.model_config['validation_split']))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train ensemble models
            for model_name in self.model_config['ensemble_models']:
                try:
                    if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                        model = await self._train_lstm_model(X_train, y_train, X_val, y_val)
                        self.models[model_name] = model
                    elif model_name == 'random_forest':
                        model = await self._train_random_forest(X_train, y_train)
                        self.models[model_name] = model
                    elif model_name == 'gradient_boosting':
                        model = await self._train_gradient_boosting(X_train, y_train)
                        self.models[model_name] = model
                    
                    logger.info(f"Successfully trained {model_name} model")
                    
                except Exception as e:
                    logger.error(f"Failed to train {model_name} model: {e}")
            
            # Store training data for later use
            self.training_data = features_df
            self.features = X
            self.targets = y
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def _prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare training data for models"""
        try:
            # Use close price as target
            target_col = 'close'
            if target_col not in features_df.columns:
                return None, None
            
            # Prepare features (exclude target and future-looking columns)
            feature_cols = [col for col in features_df.columns 
                          if col not in [target_col, 'open', 'high', 'low']]
            
            X_data = features_df[feature_cols].values
            y_data = features_df[target_col].values
            
            # Create sequences for time series models
            seq_length = self.model_config['sequence_length']
            
            X_sequences = []
            y_sequences = []
            
            for i in range(seq_length, len(X_data)):
                X_sequences.append(X_data[i-seq_length:i])
                y_sequences.append(y_data[i])
            
            X = np.array(X_sequences)
            y = np.array(y_sequences)
            
            # Scale features
            scaler = StandardScaler()
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
            
            # Store scaler
            self.scalers['features'] = scaler
            
            # Scale targets
            target_scaler = StandardScaler()
            y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            self.scalers['target'] = target_scaler
            
            return X, y
            
        except Exception as e:
            logger.error(f"Training data preparation failed: {e}")
            return None, None
    
    async def _train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                              X_val: np.ndarray, y_val: np.ndarray) -> Optional[Any]:
        """Train LSTM model"""
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow not available, skipping LSTM training")
                return None
            
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=self.model_config['learning_rate']),
                         loss='mse', metrics=['mae'])
            
            # Train model
            history = model.fit(
                X_train, y_train,
                batch_size=self.model_config['batch_size'],
                epochs=self.model_config['epochs'],
                validation_data=(X_val, y_val),
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
            
            return model
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            return None
    
    async def _train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray) -> Optional[Any]:
        """Train Random Forest model"""
        try:
            # Flatten sequences for tree-based models
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train_flat, y_train)
            
            return model
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            return None
    
    async def _train_gradient_boosting(self, X_train: np.ndarray, y_train: np.ndarray) -> Optional[Any]:
        """Train Gradient Boosting model"""
        try:
            # Flatten sequences for tree-based models
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            model.fit(X_train_flat, y_train)
            
            return model
            
        except Exception as e:
            logger.error(f"Gradient Boosting training failed: {e}")
            return None
    
    async def _generate_price_forecasts(self, features_df: pd.DataFrame) -> List[ForecastScenario]:
        """Generate price forecast scenarios"""
        try:
            logger.info("Generating price forecasts")
            
            if not self.models:
                logger.warning("No trained models available for forecasting")
                return []
            
            # Get latest features for prediction
            latest_features = self._get_latest_features(features_df)
            
            if latest_features is None:
                return []
            
            # Generate predictions from each model
            model_predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                        pred = model.predict(latest_features.reshape(1, *latest_features.shape), verbose=0)
                        model_predictions[model_name] = pred[0][0]
                    else:  # Tree-based models
                        pred = model.predict(latest_features.reshape(1, -1))
                        model_predictions[model_name] = pred[0]
                except Exception as e:
                    logger.error(f"Prediction failed for {model_name}: {e}")
            
            if not model_predictions:
                return []
            
            # Inverse transform predictions
            if 'target' in self.scalers:
                for model_name in model_predictions:
                    pred_scaled = np.array([[model_predictions[model_name]]])
                    pred_unscaled = self.scalers['target'].inverse_transform(pred_scaled)
                    model_predictions[model_name] = pred_unscaled[0][0]
            
            # Create forecast scenarios
            scenarios = await self._create_forecast_scenarios(model_predictions, features_df)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Price forecast generation failed: {e}")
            return []
    
    def _get_latest_features(self, features_df: pd.DataFrame) -> Optional[np.ndarray]:
        """Get latest features for prediction"""
        try:
            if len(features_df) < self.model_config['sequence_length']:
                return None
            
            # Get feature columns (same as training)
            target_col = 'close'
            feature_cols = [col for col in features_df.columns 
                          if col not in [target_col, 'open', 'high', 'low']]
            
            # Get latest sequence
            latest_data = features_df[feature_cols].iloc[-self.model_config['sequence_length']:].values
            
            # Scale features
            if 'features' in self.scalers:
                latest_data = self.scalers['features'].transform(latest_data)
            
            return latest_data
            
        except Exception as e:
            logger.error(f"Latest features extraction failed: {e}")
            return None
    
    async def _create_forecast_scenarios(self, model_predictions: Dict[str, float], 
                                       features_df: pd.DataFrame) -> List[ForecastScenario]:
        """Create forecast scenarios from model predictions"""
        try:
            scenarios = []
            
            # Get current price
            current_price = features_df['close'].iloc[-1]
            
            # Calculate ensemble prediction
            ensemble_pred = np.mean(list(model_predictions.values()))
            
            # Calculate prediction variance
            pred_variance = np.var(list(model_predictions.values()))
            pred_std = np.sqrt(pred_variance)
            
            # Create scenarios based on confidence intervals
            for i, confidence_level in enumerate(self.scenario_config['confidence_levels']):
                z_score = 1.96 if confidence_level == 0.95 else (2.58 if confidence_level == 0.99 else 1.0)
                
                # Calculate price targets
                lower_bound = ensemble_pred - z_score * pred_std
                upper_bound = ensemble_pred + z_score * pred_std
                
                scenario_name = f"Scenario_{i+1}_{int(confidence_level*100)}%"
                
                scenario = ForecastScenario(
                    scenario_name=scenario_name,
                    probability=confidence_level,
                    price_targets=[lower_bound, ensemble_pred, upper_bound],
                    timeframes=["1D", "1W", "1M"],
                    confidence_intervals={
                        "1D": (lower_bound, upper_bound),
                        "1W": (lower_bound * 0.8, upper_bound * 1.2),
                        "1M": (lower_bound * 0.6, upper_bound * 1.4)
                    },
                    key_levels={
                        "support": current_price * 0.95,
                        "resistance": current_price * 1.05,
                        "target": ensemble_pred
                    },
                    risk_factors=["Market volatility", "Economic events"],
                    catalysts=["Technical breakout", "Volume confirmation"],
                    metadata={
                        "model_predictions": model_predictions,
                        "ensemble_prediction": ensemble_pred,
                        "prediction_variance": pred_variance
                    }
                )
                
                scenarios.append(scenario)
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Forecast scenario creation failed: {e}")
            return []
    
    async def _generate_volatility_forecast(self, features_df: pd.DataFrame) -> VolatilityForecast:
        """Generate volatility forecast"""
        try:
            logger.info("Generating volatility forecast")
            
            # Calculate historical volatility
            returns = features_df['close'].pct_change().dropna()
            
            if len(returns) < 20:
                return self._create_empty_volatility_forecast()
            
            # Current volatility metrics
            current_vol = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            vol_percentiles = {
                "10th": returns.rolling(100).std().quantile(0.1) * np.sqrt(252),
                "25th": returns.rolling(100).std().quantile(0.25) * np.sqrt(252),
                "50th": returns.rolling(100).std().quantile(0.5) * np.sqrt(252),
                "75th": returns.rolling(100).std().quantile(0.75) * np.sqrt(252),
                "90th": returns.rolling(100).std().quantile(0.9) * np.sqrt(252)
            }
            
            # Determine volatility regime
            if current_vol < vol_percentiles["25th"]:
                vol_regime = "low"
            elif current_vol > vol_percentiles["75th"]:
                vol_regime = "high"
            else:
                vol_regime = "medium"
            
            # Forecast future volatility (simple GARCH-like approach)
            forecast_horizon = self.model_config['forecast_horizon']
            predicted_vol = [current_vol] * forecast_horizon
            
            # Add some mean reversion
            long_term_vol = vol_percentiles["50th"]
            for i in range(1, forecast_horizon):
                predicted_vol[i] = predicted_vol[i-1] * 0.95 + long_term_vol * 0.05
            
            # Calculate expected price range
            current_price = features_df['close'].iloc[-1]
            expected_range = (
                current_price * (1 - current_vol / np.sqrt(252)),
                current_price * (1 + current_vol / np.sqrt(252))
            )
            
            # Estimate probabilities
            breakout_prob = min(current_vol / vol_percentiles["75th"], 1.0)
            mean_reversion_prob = 1.0 - breakout_prob
            
            # Generate timestamps
            last_timestamp = features_df.index[-1]
            timestamps = [last_timestamp + timedelta(days=i) for i in range(1, forecast_horizon + 1)]
            
            return VolatilityForecast(
                predicted_volatility=predicted_vol,
                volatility_regime=vol_regime,
                volatility_percentiles=vol_percentiles,
                expected_range=expected_range,
                breakout_probability=breakout_prob,
                mean_reversion_probability=mean_reversion_prob,
                timestamps=timestamps,
                confidence=0.7,
                metadata={
                    "current_volatility": current_vol,
                    "long_term_volatility": long_term_vol,
                    "volatility_method": "rolling_std_with_mean_reversion"
                }
            )
            
        except Exception as e:
            logger.error(f"Volatility forecast generation failed: {e}")
            return self._create_empty_volatility_forecast()
    
    async def _generate_trend_forecast(self, features_df: pd.DataFrame) -> TrendForecast:
        """Generate trend forecast"""
        try:
            logger.info("Generating trend forecast")
            
            if len(features_df) < 50:
                return self._create_empty_trend_forecast()
            
            # Calculate trend indicators
            prices = features_df['close']
            
            # Short-term trend (20 periods)
            short_ma = prices.rolling(20).mean()
            short_trend = (prices.iloc[-1] - short_ma.iloc[-1]) / short_ma.iloc[-1]
            
            # Long-term trend (50 periods)
            long_ma = prices.rolling(50).mean()
            long_trend = (prices.iloc[-1] - long_ma.iloc[-1]) / long_ma.iloc[-1]
            
            # Determine trend direction
            if short_trend > 0.05 and long_trend > 0.02:
                trend_direction = TrendDirection.STRONG_BULLISH
            elif short_trend > 0.02 or long_trend > 0.01:
                trend_direction = TrendDirection.BULLISH
            elif short_trend < -0.05 and long_trend < -0.02:
                trend_direction = TrendDirection.STRONG_BEARISH
            elif short_trend < -0.02 or long_trend < -0.01:
                trend_direction = TrendDirection.BEARISH
            else:
                trend_direction = TrendDirection.NEUTRAL
            
            # Calculate trend strength
            trend_strength = min(abs(short_trend) + abs(long_trend), 1.0)
            
            # Estimate trend duration
            trend_duration = timedelta(days=max(10, int(trend_strength * 30)))
            
            # Calculate reversal probability
            price_position = (prices.iloc[-1] - prices.rolling(100).min().iloc[-1]) / \
                           (prices.rolling(100).max().iloc[-1] - prices.rolling(100).min().iloc[-1])
            
            if trend_direction in [TrendDirection.STRONG_BULLISH, TrendDirection.BULLISH]:
                reversal_prob = price_position  # Higher when price is at top of range
            else:
                reversal_prob = 1 - price_position  # Higher when price is at bottom of range
            
            continuation_prob = 1 - reversal_prob
            
            # Find inflection points (simplified)
            inflection_points = []
            for i in range(len(prices) - 20, len(prices)):
                if i > 10 and i < len(prices) - 10:
                    if (prices.iloc[i] > prices.iloc[i-5:i].max() and 
                        prices.iloc[i] > prices.iloc[i+1:i+6].max()):
                        inflection_points.append((features_df.index[i], prices.iloc[i]))
            
            # Calculate trend channels
            recent_highs = prices.rolling(20).max().iloc[-20:]
            recent_lows = prices.rolling(20).min().iloc[-20:]
            
            trend_channels = {
                "upper_channel": (recent_highs.min(), recent_highs.max()),
                "lower_channel": (recent_lows.min(), recent_lows.max()),
                "current_channel": (recent_lows.iloc[-1], recent_highs.iloc[-1])
            }
            
            # Momentum indicators
            momentum_indicators = {
                "short_momentum": short_trend,
                "long_momentum": long_trend,
                "price_position": price_position,
                "trend_consistency": trend_strength
            }
            
            return TrendForecast(
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                trend_duration=trend_duration,
                reversal_probability=reversal_prob,
                continuation_probability=continuation_prob,
                key_inflection_points=inflection_points,
                trend_channels=trend_channels,
                momentum_indicators=momentum_indicators,
                confidence=0.7,
                metadata={
                    "short_ma": short_ma.iloc[-1],
                    "long_ma": long_ma.iloc[-1],
                    "calculation_method": "moving_average_crossover"
                }
            )
            
        except Exception as e:
            logger.error(f"Trend forecast generation failed: {e}")
            return self._create_empty_trend_forecast()
    
    async def _forecast_custom_indicators(self, features_df: pd.DataFrame, 
                                        custom_indicators: Optional[Dict[str, List[float]]]) -> Dict[str, List[float]]:
        """Forecast custom indicators"""
        try:
            if not custom_indicators:
                return {}
            
            forecasts = {}
            
            for indicator_name, values in custom_indicators.items():
                if len(values) >= 10:
                    # Simple trend extrapolation for custom indicators
                    recent_values = values[-10:]
                    trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
                    
                    # Forecast next 30 periods
                    forecast_periods = 30
                    forecast_values = []
                    
                    for i in range(1, forecast_periods + 1):
                        forecast_value = values[-1] + trend * i
                        forecast_values.append(forecast_value)
                    
                    forecasts[indicator_name] = forecast_values
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Custom indicator forecasting failed: {e}")
            return {}
    
    async def _assess_model_performance(self) -> Dict[str, float]:
        """Assess model performance metrics"""
        try:
            if not self.models or self.features is None or self.targets is None:
                return {}
            
            performance = {}
            
            # Use last 20% of data for validation
            val_split = int(len(self.features) * 0.8)
            X_val = self.features[val_split:]
            y_val = self.targets[val_split:]
            
            for model_name, model in self.models.items():
                try:
                    if model_name == 'lstm' and TENSORFLOW_AVAILABLE:
                        predictions = model.predict(X_val, verbose=0).flatten()
                    else:  # Tree-based models
                        X_val_flat = X_val.reshape(X_val.shape[0], -1)
                        predictions = model.predict(X_val_flat)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_val, predictions)
                    mae = mean_absolute_error(y_val, predictions)
                    
                    performance[f"{model_name}_mse"] = mse
                    performance[f"{model_name}_mae"] = mae
                    performance[f"{model_name}_rmse"] = np.sqrt(mse)
                    
                except Exception as e:
                    logger.error(f"Performance assessment failed for {model_name}: {e}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Model performance assessment failed: {e}")
            return {}
    
    async def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate feature importance"""
        try:
            importance = {}
            
            # Get feature importance from tree-based models
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    # This is a tree-based model
                    importances = model.feature_importances_
                    
                    # Map to feature names (simplified)
                    for i, imp in enumerate(importances):
                        feature_name = f"feature_{i}"
                        importance[f"{model_name}_{feature_name}"] = imp
            
            return importance
            
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    async def _perform_risk_assessment(self, price_forecasts: List[ForecastScenario], 
                                     volatility_forecast: VolatilityForecast) -> Dict[str, Any]:
        """Perform risk assessment"""
        try:
            risk_assessment = {
                "overall_risk": "medium",
                "volatility_risk": volatility_forecast.volatility_regime,
                "trend_risk": "neutral",
                "scenario_risks": [],
                "risk_factors": [
                    "Market volatility",
                    "Economic uncertainty",
                    "Technical levels"
                ],
                "risk_mitigation": [
                    "Diversification",
                    "Position sizing",
                    "Stop losses"
                ]
            }
            
            # Assess scenario-specific risks
            for scenario in price_forecasts:
                scenario_risk = {
                    "scenario": scenario.scenario_name,
                    "probability": scenario.probability,
                    "risk_level": "medium",
                    "key_risks": scenario.risk_factors
                }
                risk_assessment["scenario_risks"].append(scenario_risk)
            
            return risk_assessment
            
        except Exception as e:
            logger.error(f"Risk assessment failed: {e}")
            return {"overall_risk": "unknown"}
    
    def _calculate_overall_confidence(self, price_forecasts: List[ForecastScenario], 
                                    volatility_forecast: VolatilityForecast,
                                    trend_forecast: TrendForecast, 
                                    performance_metrics: Dict[str, float]) -> float:
        """Calculate overall confidence score"""
        try:
            confidence_factors = []
            
            # Model performance confidence
            if performance_metrics:
                avg_mae = np.mean([v for k, v in performance_metrics.items() if 'mae' in k])
                perf_confidence = max(0, 1 - avg_mae)  # Lower MAE = higher confidence
                confidence_factors.append(perf_confidence * 0.3)
            
            # Forecast consistency confidence
            if price_forecasts:
                price_variance = np.var([scenario.price_targets[1] for scenario in price_forecasts])
                consistency_confidence = max(0, 1 - price_variance / 100)  # Normalized
                confidence_factors.append(consistency_confidence * 0.2)
            
            # Volatility confidence
            vol_confidence = volatility_forecast.confidence
            confidence_factors.append(vol_confidence * 0.2)
            
            # Trend confidence
            trend_confidence = trend_forecast.confidence
            confidence_factors.append(trend_confidence * 0.2)
            
            # Base confidence
            confidence_factors.append(0.1)
            
            return min(sum(confidence_factors), 1.0)
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.5
    
    def _create_empty_output(self) -> KronosOutput:
        """Create empty output for error cases"""
        return KronosOutput(
            price_forecasts=[],
            volatility_forecast=self._create_empty_volatility_forecast(),
            trend_forecast=self._create_empty_trend_forecast(),
            custom_indicator_forecasts={},
            model_performance={},
            feature_importance={},
            forecast_horizon=timedelta(days=30),
            forecast_timestamp=datetime.now(),
            confidence_score=0.0,
            risk_assessment={"overall_risk": "unknown"},
            metadata={"error": "Forecasting failed"}
        )
    
    def _create_empty_volatility_forecast(self) -> VolatilityForecast:
        """Create empty volatility forecast"""
        return VolatilityForecast(
            predicted_volatility=[],
            volatility_regime="unknown",
            volatility_percentiles={},
            expected_range=(0.0, 0.0),
            breakout_probability=0.5,
            mean_reversion_probability=0.5,
            timestamps=[],
            confidence=0.0,
            metadata={"error": "Volatility forecasting failed"}
        )
    
    def _create_empty_trend_forecast(self) -> TrendForecast:
        """Create empty trend forecast"""
        return TrendForecast(
            trend_direction=TrendDirection.NEUTRAL,
            trend_strength=0.0,
            trend_duration=timedelta(days=0),
            reversal_probability=0.5,
            continuation_probability=0.5,
            key_inflection_points=[],
            trend_channels={},
            momentum_indicators={},
            confidence=0.0,
            metadata={"error": "Trend forecasting failed"}
        )