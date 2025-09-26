"""Machine Learning Pipeline Service

Provides comprehensive ML capabilities including:
- Feature engineering and selection
- Model training and evaluation
- Hyperparameter optimization
- Cross-validation and backtesting
- Model persistence and versioning
- Ensemble methods
- Deep learning models
"""

import asyncio
import logging
import pickle
import joblib
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    RandomizedSearchCV, TimeSeriesSplit
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    LabelEncoder, OneHotEncoder
)
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, VotingRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    SGDRegressor, BayesianRidge
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

# XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

# Deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available")

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logging.warning("PyTorch not available")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Machine learning model types"""
    LINEAR_REGRESSION = "linear_regression"
    RIDGE = "ridge"
    LASSO = "lasso"
    ELASTIC_NET = "elastic_net"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVM = "svm"
    KNN = "knn"
    NEURAL_NETWORK = "neural_network"
    LSTM = "lstm"
    GRU = "gru"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"

class FeatureType(Enum):
    """Feature engineering types"""
    TECHNICAL_INDICATORS = "technical_indicators"
    PRICE_FEATURES = "price_features"
    VOLUME_FEATURES = "volume_features"
    TIME_FEATURES = "time_features"
    LAG_FEATURES = "lag_features"
    ROLLING_FEATURES = "rolling_features"
    STATISTICAL_FEATURES = "statistical_features"

@dataclass
class ModelConfig:
    """ML model configuration"""
    model_type: ModelType
    parameters: Dict[str, Any]
    feature_columns: List[str]
    target_column: str
    preprocessing: Dict[str, Any] = None
    validation_strategy: str = 'time_series_split'
    test_size: float = 0.2
    random_state: int = 42

@dataclass
class ModelResult:
    """ML model training result"""
    model_id: str
    model_type: ModelType
    model: Any  # Trained model object
    metrics: Dict[str, float]
    feature_importance: Optional[pd.Series] = None
    predictions: Optional[pd.Series] = None
    validation_scores: Optional[Dict[str, List[float]]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    training_time: Optional[float] = None
    metadata: Dict[str, Any] = None

@dataclass
class FeatureSet:
    """Feature engineering result"""
    features: pd.DataFrame
    feature_names: List[str]
    feature_types: Dict[str, FeatureType]
    importance_scores: Optional[pd.Series] = None
    correlation_matrix: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = None

@dataclass
class BacktestResult:
    """Backtesting result"""
    strategy_returns: pd.Series
    benchmark_returns: pd.Series
    metrics: Dict[str, float]
    trades: pd.DataFrame
    equity_curve: pd.Series
    drawdown: pd.Series
    rolling_metrics: pd.DataFrame

class MLPipelineService:
    """Comprehensive machine learning pipeline service"""
    
    def __init__(self, model_storage_path: str = "models"):
        self.model_storage_path = Path(model_storage_path)
        self.model_storage_path.mkdir(exist_ok=True)
        
        self.models = {}
        self.feature_engineers = {}
        self.scalers = {}
        self._initialize_models()
        self._initialize_feature_engineers()
    
    def _initialize_models(self):
        """Initialize available ML models"""
        self.models = {
            ModelType.LINEAR_REGRESSION: LinearRegression,
            ModelType.RIDGE: Ridge,
            ModelType.LASSO: Lasso,
            ModelType.ELASTIC_NET: ElasticNet,
            ModelType.RANDOM_FOREST: RandomForestRegressor,
            ModelType.GRADIENT_BOOSTING: GradientBoostingRegressor,
            ModelType.SVM: SVR,
            ModelType.KNN: KNeighborsRegressor,
            ModelType.NEURAL_NETWORK: MLPRegressor,
        }
        
        if XGBOOST_AVAILABLE:
            self.models[ModelType.XGBOOST] = xgb.XGBRegressor
        
        if LIGHTGBM_AVAILABLE:
            self.models[ModelType.LIGHTGBM] = lgb.LGBMRegressor
    
    def _initialize_feature_engineers(self):
        """Initialize feature engineering functions"""
        self.feature_engineers = {
            FeatureType.TECHNICAL_INDICATORS: self._create_technical_features,
            FeatureType.PRICE_FEATURES: self._create_price_features,
            FeatureType.VOLUME_FEATURES: self._create_volume_features,
            FeatureType.TIME_FEATURES: self._create_time_features,
            FeatureType.LAG_FEATURES: self._create_lag_features,
            FeatureType.ROLLING_FEATURES: self._create_rolling_features,
            FeatureType.STATISTICAL_FEATURES: self._create_statistical_features,
        }
    
    async def engineer_features(self, 
                              data: pd.DataFrame, 
                              feature_types: List[FeatureType],
                              **kwargs) -> FeatureSet:
        """Engineer features from raw data"""
        all_features = data.copy()
        feature_names = list(data.columns)
        feature_type_mapping = {col: FeatureType.PRICE_FEATURES for col in data.columns}
        
        for feature_type in feature_types:
            if feature_type in self.feature_engineers:
                try:
                    new_features = await asyncio.to_thread(
                        self.feature_engineers[feature_type], data, **kwargs
                    )
                    
                    if new_features is not None and not new_features.empty:
                        # Add new features
                        for col in new_features.columns:
                            if col not in all_features.columns:
                                all_features[col] = new_features[col]
                                feature_names.append(col)
                                feature_type_mapping[col] = feature_type
                
                except Exception as e:
                    logger.error(f"Error creating {feature_type} features: {str(e)}")
        
        # Calculate feature importance using correlation with target
        importance_scores = None
        if 'close' in all_features.columns:
            numeric_features = all_features.select_dtypes(include=[np.number])
            correlations = numeric_features.corr()['close'].abs().sort_values(ascending=False)
            importance_scores = correlations.drop('close', errors='ignore')
        
        # Calculate correlation matrix
        correlation_matrix = all_features.select_dtypes(include=[np.number]).corr()
        
        return FeatureSet(
            features=all_features,
            feature_names=feature_names,
            feature_types=feature_type_mapping,
            importance_scores=importance_scores,
            correlation_matrix=correlation_matrix,
            metadata={'original_columns': list(data.columns)}
        )
    
    async def train_model(self, 
                         config: ModelConfig, 
                         data: pd.DataFrame) -> ModelResult:
        """Train a machine learning model"""
        start_time = datetime.now()
        
        # Prepare data
        X = data[config.feature_columns].copy()
        y = data[config.target_column].copy()
        
        # Remove NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Preprocessing
        if config.preprocessing:
            X_clean = await self._preprocess_features(X_clean, config.preprocessing)
        
        # Split data
        if config.validation_strategy == 'time_series_split':
            split_idx = int(len(X_clean) * (1 - config.test_size))
            X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
            y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=config.test_size, 
                random_state=config.random_state
            )
        
        # Initialize and train model
        if config.model_type not in self.models:
            raise ValueError(f"Model type '{config.model_type}' not supported")
        
        model_class = self.models[config.model_type]
        model = model_class(**config.parameters)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
        }
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.Series(
                model.feature_importances_, 
                index=config.feature_columns
            ).sort_values(ascending=False)
        elif hasattr(model, 'coef_'):
            feature_importance = pd.Series(
                np.abs(model.coef_), 
                index=config.feature_columns
            ).sort_values(ascending=False)
        
        # Cross-validation scores
        validation_scores = None
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=TimeSeriesSplit(n_splits=5), 
                scoring='neg_mean_squared_error'
            )
            validation_scores = {
                'cv_mse_mean': -cv_scores.mean(),
                'cv_mse_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist()
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
        
        # Generate model ID
        model_id = f"{config.model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Create predictions series
        predictions = pd.Series(y_pred_test, index=y_test.index)
        
        return ModelResult(
            model_id=model_id,
            model_type=config.model_type,
            model=model,
            metrics=metrics,
            feature_importance=feature_importance,
            predictions=predictions,
            validation_scores=validation_scores,
            hyperparameters=config.parameters,
            training_time=training_time,
            metadata={
                'feature_columns': config.feature_columns,
                'target_column': config.target_column,
                'train_size': len(X_train),
                'test_size': len(X_test)
            }
        )
    
    async def optimize_hyperparameters(self, 
                                     config: ModelConfig, 
                                     data: pd.DataFrame,
                                     param_grid: Dict[str, List],
                                     search_type: str = 'grid') -> ModelResult:
        """Optimize model hyperparameters"""
        # Prepare data
        X = data[config.feature_columns].copy()
        y = data[config.target_column].copy()
        
        # Remove NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Preprocessing
        if config.preprocessing:
            X_clean = await self._preprocess_features(X_clean, config.preprocessing)
        
        # Initialize model
        model_class = self.models[config.model_type]
        base_model = model_class()
        
        # Setup search
        cv = TimeSeriesSplit(n_splits=5)
        
        if search_type == 'grid':
            search = GridSearchCV(
                base_model, param_grid, cv=cv, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
        else:  # random search
            search = RandomizedSearchCV(
                base_model, param_grid, cv=cv,
                scoring='neg_mean_squared_error', n_jobs=-1,
                n_iter=50, random_state=config.random_state
            )
        
        # Perform search
        search.fit(X_clean, y_clean)
        
        # Update config with best parameters
        optimized_config = ModelConfig(
            model_type=config.model_type,
            parameters=search.best_params_,
            feature_columns=config.feature_columns,
            target_column=config.target_column,
            preprocessing=config.preprocessing,
            validation_strategy=config.validation_strategy,
            test_size=config.test_size,
            random_state=config.random_state
        )
        
        # Train final model with best parameters
        result = await self.train_model(optimized_config, data)
        result.metadata['hyperparameter_search'] = {
            'best_score': search.best_score_,
            'best_params': search.best_params_,
            'search_type': search_type
        }
        
        return result
    
    async def create_ensemble(self, 
                            models: List[ModelResult], 
                            method: str = 'voting') -> ModelResult:
        """Create ensemble model from multiple trained models"""
        if method == 'voting':
            # Create voting regressor
            estimators = [(f"model_{i}", model.model) for i, model in enumerate(models)]
            ensemble_model = VotingRegressor(estimators=estimators)
            
            # For simplicity, we'll use the feature columns from the first model
            # In practice, you'd want to ensure all models use the same features
            feature_columns = models[0].metadata['feature_columns']
            
            model_id = f"ensemble_voting_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Combine metrics (simple average)
            combined_metrics = {}
            for metric in models[0].metrics.keys():
                combined_metrics[metric] = np.mean([model.metrics[metric] for model in models])
            
            return ModelResult(
                model_id=model_id,
                model_type=ModelType.ENSEMBLE,
                model=ensemble_model,
                metrics=combined_metrics,
                metadata={
                    'ensemble_method': method,
                    'component_models': [model.model_id for model in models],
                    'feature_columns': feature_columns
                }
            )
        else:
            raise ValueError(f"Ensemble method '{method}' not supported")
    
    async def backtest_strategy(self, 
                              model: ModelResult, 
                              data: pd.DataFrame,
                              strategy_config: Dict[str, Any]) -> BacktestResult:
        """Backtest a trading strategy using ML predictions"""
        # Generate predictions
        X = data[model.metadata['feature_columns']]
        predictions = model.model.predict(X)
        
        # Create signals based on predictions
        signals = self._generate_trading_signals(predictions, strategy_config)
        
        # Calculate returns
        returns = data['close'].pct_change()
        strategy_returns = signals.shift(1) * returns  # Lag signals by 1 period
        
        # Calculate metrics
        total_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + strategy_returns.mean()) ** 252 - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility != 0 else 0
        
        # Calculate drawdown
        equity_curve = (1 + strategy_returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (strategy_returns > 0).mean()
        }
        
        # Create trades DataFrame
        trades = pd.DataFrame({
            'signal': signals,
            'return': strategy_returns,
            'cumulative_return': equity_curve
        })
        
        return BacktestResult(
            strategy_returns=strategy_returns,
            benchmark_returns=returns,
            metrics=metrics,
            trades=trades,
            equity_curve=equity_curve,
            drawdown=drawdown,
            rolling_metrics=pd.DataFrame()  # Placeholder
        )
    
    async def save_model(self, model_result: ModelResult) -> str:
        """Save trained model to disk"""
        model_path = self.model_storage_path / f"{model_result.model_id}.pkl"
        
        # Save model and metadata
        model_data = {
            'model': model_result.model,
            'model_type': model_result.model_type,
            'metrics': model_result.metrics,
            'feature_importance': model_result.feature_importance,
            'hyperparameters': model_result.hyperparameters,
            'metadata': model_result.metadata
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    async def load_model(self, model_id: str) -> ModelResult:
        """Load trained model from disk"""
        model_path = self.model_storage_path / f"{model_id}.pkl"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_id} not found")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        return ModelResult(
            model_id=model_id,
            model_type=model_data['model_type'],
            model=model_data['model'],
            metrics=model_data['metrics'],
            feature_importance=model_data.get('feature_importance'),
            hyperparameters=model_data.get('hyperparameters'),
            metadata=model_data.get('metadata', {})
        )
    
    # Feature Engineering Methods
    def _create_technical_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Create technical indicator features"""
        features = pd.DataFrame(index=data.index)
        
        if 'close' in data.columns:
            # Simple moving averages
            for period in [5, 10, 20, 50]:
                features[f'sma_{period}'] = data['close'].rolling(period).mean()
                features[f'price_sma_{period}_ratio'] = data['close'] / features[f'sma_{period}']
            
            # Exponential moving averages
            for period in [12, 26]:
                features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            
            # MACD
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            features['macd'] = ema_12 - ema_26
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            sma_20 = data['close'].rolling(20).mean()
            std_20 = data['close'].rolling(20).std()
            features['bb_upper'] = sma_20 + (std_20 * 2)
            features['bb_lower'] = sma_20 - (std_20 * 2)
            features['bb_width'] = features['bb_upper'] - features['bb_lower']
            features['bb_position'] = (data['close'] - features['bb_lower']) / features['bb_width']
        
        return features
    
    def _create_price_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Create price-based features"""
        features = pd.DataFrame(index=data.index)
        
        if 'close' in data.columns:
            # Returns
            features['return_1d'] = data['close'].pct_change()
            features['return_5d'] = data['close'].pct_change(5)
            features['return_10d'] = data['close'].pct_change(10)
            
            # Log returns
            features['log_return'] = np.log(data['close'] / data['close'].shift(1))
            
            # Price momentum
            features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
            features['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        
        if all(col in data.columns for col in ['high', 'low', 'close']):
            # True Range
            features['true_range'] = np.maximum(
                data['high'] - data['low'],
                np.maximum(
                    np.abs(data['high'] - data['close'].shift(1)),
                    np.abs(data['low'] - data['close'].shift(1))
                )
            )
            
            # Average True Range
            features['atr'] = features['true_range'].rolling(14).mean()
            
            # High-Low spread
            features['hl_spread'] = (data['high'] - data['low']) / data['close']
        
        return features
    
    def _create_volume_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Create volume-based features"""
        features = pd.DataFrame(index=data.index)
        
        if 'volume' in data.columns:
            # Volume moving averages
            features['volume_sma_10'] = data['volume'].rolling(10).mean()
            features['volume_sma_20'] = data['volume'].rolling(20).mean()
            
            # Volume ratio
            features['volume_ratio'] = data['volume'] / features['volume_sma_20']
            
            # On-Balance Volume (simplified)
            if 'close' in data.columns:
                price_change = data['close'].diff()
                volume_direction = np.where(price_change > 0, data['volume'], 
                                          np.where(price_change < 0, -data['volume'], 0))
                features['obv'] = pd.Series(volume_direction, index=data.index).cumsum()
        
        return features
    
    def _create_time_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Create time-based features"""
        features = pd.DataFrame(index=data.index)
        
        # Extract time components
        features['hour'] = data.index.hour
        features['day_of_week'] = data.index.dayofweek
        features['day_of_month'] = data.index.day
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter
        
        # Cyclical encoding
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features
    
    def _create_lag_features(self, data: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10], **kwargs) -> pd.DataFrame:
        """Create lagged features"""
        features = pd.DataFrame(index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            for lag in lags:
                features[f'{col}_lag_{lag}'] = data[col].shift(lag)
        
        return features
    
    def _create_rolling_features(self, data: pd.DataFrame, windows: List[int] = [5, 10, 20], **kwargs) -> pd.DataFrame:
        """Create rolling window features"""
        features = pd.DataFrame(index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            for window in windows:
                features[f'{col}_mean_{window}'] = data[col].rolling(window).mean()
                features[f'{col}_std_{window}'] = data[col].rolling(window).std()
                features[f'{col}_min_{window}'] = data[col].rolling(window).min()
                features[f'{col}_max_{window}'] = data[col].rolling(window).max()
        
        return features
    
    def _create_statistical_features(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Create statistical features"""
        features = pd.DataFrame(index=data.index)
        
        for col in data.select_dtypes(include=[np.number]).columns:
            # Rolling statistics
            rolling_20 = data[col].rolling(20)
            features[f'{col}_skew'] = rolling_20.skew()
            features[f'{col}_kurt'] = rolling_20.kurt()
            features[f'{col}_quantile_25'] = rolling_20.quantile(0.25)
            features[f'{col}_quantile_75'] = rolling_20.quantile(0.75)
        
        return features
    
    async def _preprocess_features(self, X: pd.DataFrame, preprocessing_config: Dict[str, Any]) -> pd.DataFrame:
        """Preprocess features"""
        X_processed = X.copy()
        
        # Scaling
        if 'scaler' in preprocessing_config:
            scaler_type = preprocessing_config['scaler']
            if scaler_type == 'standard':
                scaler = StandardScaler()
            elif scaler_type == 'minmax':
                scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                scaler = RobustScaler()
            else:
                scaler = None
            
            if scaler:
                X_processed = pd.DataFrame(
                    scaler.fit_transform(X_processed),
                    columns=X_processed.columns,
                    index=X_processed.index
                )
        
        return X_processed
    
    def _generate_trading_signals(self, predictions: np.ndarray, strategy_config: Dict[str, Any]) -> pd.Series:
        """Generate trading signals from predictions"""
        # Simple threshold-based strategy
        threshold = strategy_config.get('threshold', 0.01)
        
        signals = np.where(predictions > threshold, 1,  # Buy signal
                          np.where(predictions < -threshold, -1, 0))  # Sell signal, Hold
        
        return pd.Series(signals)

# Global instance
ml_pipeline_service = MLPipelineService()