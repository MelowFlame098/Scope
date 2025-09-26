# Market Forecaster
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ForecastType(Enum):
    PRICE = "price"
    RETURN = "return"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    DIRECTION = "direction"
    SUPPORT_RESISTANCE = "support_resistance"

class TimeHorizon(Enum):
    INTRADAY = "intraday"  # 1-24 hours
    SHORT_TERM = "short_term"  # 1-7 days
    MEDIUM_TERM = "medium_term"  # 1-4 weeks
    LONG_TERM = "long_term"  # 1-12 months

class ModelType(Enum):
    LINEAR = "linear"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"

@dataclass
class ForecastResult:
    symbol: str
    forecast_type: ForecastType
    time_horizon: TimeHorizon
    predictions: List[float]
    timestamps: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    confidence_score: float
    model_used: str
    feature_importance: Dict[str, float]
    metrics: Dict[str, float]
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class MarketFeatures:
    price_features: Dict[str, float]
    technical_features: Dict[str, float]
    volume_features: Dict[str, float]
    volatility_features: Dict[str, float]
    momentum_features: Dict[str, float]
    external_features: Dict[str, float]

class MarketForecaster:
    """Advanced market forecasting engine using ensemble models"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.models = {}
        self.scalers = {}
        self.feature_extractors = {}
        
        # Model configurations
        self.model_configs = {
            ModelType.LINEAR: {
                'class': Ridge,
                'params': {'alpha': 1.0, 'random_state': 42}
            },
            ModelType.RANDOM_FOREST: {
                'class': RandomForestRegressor,
                'params': {'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}
            },
            ModelType.GRADIENT_BOOSTING: {
                'class': GradientBoostingRegressor,
                'params': {'n_estimators': 100, 'random_state': 42}
            }
        }
        
        # Ensemble weights
        self.ensemble_weights = {
            ModelType.LINEAR: 0.2,
            ModelType.RANDOM_FOREST: 0.3,
            ModelType.GRADIENT_BOOSTING: 0.5
        }
        
        # Feature windows
        self.feature_windows = {
            'short': [5, 10, 20],
            'medium': [50, 100],
            'long': [200, 252]
        }
        
        # Initialize models
        asyncio.create_task(self._initialize_models())
        
        logger.info("Market forecaster initialized")
    
    async def forecast(self, symbol: str, timeframe: str = '1d', 
                     forecast_type: ForecastType = ForecastType.PRICE,
                     time_horizon: TimeHorizon = TimeHorizon.SHORT_TERM,
                     periods: int = 5) -> ForecastResult:
        """Generate market forecast"""
        try:
            # Get historical data
            data = await self._get_market_data(symbol, timeframe)
            
            if data is None or len(data) < 100:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Extract features
            features = await self._extract_features(data)
            
            # Prepare target variable
            target = self._prepare_target(data, forecast_type)
            
            # Generate predictions
            predictions = await self._generate_predictions(
                features, target, forecast_type, time_horizon, periods
            )
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                predictions, features, target
            )
            
            # Generate timestamps
            timestamps = self._generate_timestamps(data, timeframe, periods)
            
            # Calculate metrics
            metrics = await self._calculate_metrics(features, target)
            
            # Get feature importance
            feature_importance = self._get_feature_importance(forecast_type)
            
            return ForecastResult(
                symbol=symbol,
                forecast_type=forecast_type,
                time_horizon=time_horizon,
                predictions=predictions,
                timestamps=timestamps,
                confidence_intervals=confidence_intervals,
                confidence_score=metrics.get('confidence', 0.0),
                model_used='ensemble',
                feature_importance=feature_importance,
                metrics=metrics,
                metadata={
                    'data_points': len(data),
                    'feature_count': len(features.columns) if hasattr(features, 'columns') else 0,
                    'timeframe': timeframe,
                    'periods': periods
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error generating forecast for {symbol}: {e}")
            return self._create_fallback_forecast(symbol, forecast_type, time_horizon, periods)
    
    async def batch_forecast(self, symbols: List[str], 
                           forecast_type: ForecastType = ForecastType.PRICE,
                           time_horizon: TimeHorizon = TimeHorizon.SHORT_TERM) -> Dict[str, ForecastResult]:
        """Generate forecasts for multiple symbols"""
        try:
            tasks = []
            for symbol in symbols:
                task = self.forecast(symbol, forecast_type=forecast_type, time_horizon=time_horizon)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            forecasts = {}
            for symbol, result in zip(symbols, results):
                if isinstance(result, Exception):
                    logger.error(f"Error forecasting {symbol}: {result}")
                    forecasts[symbol] = self._create_fallback_forecast(symbol, forecast_type, time_horizon)
                else:
                    forecasts[symbol] = result
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error in batch forecast: {e}")
            return {}
    
    async def get_forecast_accuracy(self, symbol: str, days_back: int = 30) -> Dict[str, float]:
        """Evaluate forecast accuracy over historical period"""
        try:
            # Get historical data
            data = await self._get_market_data(symbol, '1d', days_back + 50)
            
            if data is None or len(data) < days_back + 20:
                return {'error': 'Insufficient data'}
            
            # Split data for backtesting
            train_data = data[:-days_back]
            test_data = data[-days_back:]
            
            # Extract features and targets
            train_features = await self._extract_features(train_data)
            train_target = self._prepare_target(train_data, ForecastType.PRICE)
            
            test_features = await self._extract_features(test_data)
            test_target = self._prepare_target(test_data, ForecastType.PRICE)
            
            # Train models on historical data
            await self._train_models(train_features, train_target)
            
            # Generate predictions
            predictions = await self._predict_with_ensemble(test_features)
            
            # Calculate accuracy metrics
            mse = mean_squared_error(test_target, predictions)
            mae = mean_absolute_error(test_target, predictions)
            r2 = r2_score(test_target, predictions)
            
            # Calculate directional accuracy
            actual_direction = np.sign(np.diff(test_target))
            predicted_direction = np.sign(np.diff(predictions))
            directional_accuracy = np.mean(actual_direction == predicted_direction)
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'r2_score': float(r2),
                'directional_accuracy': float(directional_accuracy),
                'rmse': float(np.sqrt(mse))
            }
            
        except Exception as e:
            logger.error(f"Error calculating forecast accuracy: {e}")
            return {'error': str(e)}
    
    async def _initialize_models(self):
        """Initialize forecasting models"""
        try:
            for model_type, config in self.model_configs.items():
                model_class = config['class']
                params = config['params']
                
                self.models[model_type] = model_class(**params)
                self.scalers[model_type] = StandardScaler()
            
            logger.info("Forecasting models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def _get_market_data(self, symbol: str, timeframe: str, periods: int = 500) -> Optional[pd.DataFrame]:
        """Get market data for symbol"""
        try:
            # This would typically fetch from a data provider
            # For now, generate synthetic data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='D')
            
            # Generate realistic price data
            np.random.seed(hash(symbol) % 2**32)
            returns = np.random.normal(0.001, 0.02, periods)
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Generate volume data
            volume = np.random.lognormal(15, 0.5, periods)
            
            # Create OHLC data
            high = prices * (1 + np.abs(np.random.normal(0, 0.01, periods)))
            low = prices * (1 - np.abs(np.random.normal(0, 0.01, periods)))
            
            data = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': high,
                'low': low,
                'close': prices,
                'volume': volume
            })
            
            data.set_index('timestamp', inplace=True)
            return data
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    async def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract features from market data"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price features
            features['price'] = data['close']
            features['price_change'] = data['close'].pct_change()
            features['price_change_2d'] = data['close'].pct_change(2)
            features['price_change_5d'] = data['close'].pct_change(5)
            
            # Moving averages
            for window in [5, 10, 20, 50, 100, 200]:
                if len(data) > window:
                    features[f'ma_{window}'] = data['close'].rolling(window).mean()
                    features[f'price_to_ma_{window}'] = data['close'] / features[f'ma_{window}']
            
            # Technical indicators
            features['rsi'] = self._calculate_rsi(data['close'])
            features['macd'], features['macd_signal'] = self._calculate_macd(data['close'])
            features['bb_upper'], features['bb_lower'] = self._calculate_bollinger_bands(data['close'])
            
            # Volume features
            features['volume'] = data['volume']
            features['volume_ma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_ma']
            
            # Volatility features
            features['volatility'] = data['close'].rolling(20).std()
            features['high_low_ratio'] = data['high'] / data['low']
            features['true_range'] = self._calculate_true_range(data)
            features['atr'] = features['true_range'].rolling(14).mean()
            
            # Momentum features
            features['momentum'] = data['close'] / data['close'].shift(10)
            features['roc'] = data['close'].pct_change(10)
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                features[f'price_lag_{lag}'] = data['close'].shift(lag)
                features[f'return_lag_{lag}'] = data['close'].pct_change().shift(lag)
            
            # Time-based features
            features['day_of_week'] = data.index.dayofweek
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            
            # Drop rows with NaN values
            features = features.dropna()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return pd.DataFrame()
    
    def _prepare_target(self, data: pd.DataFrame, forecast_type: ForecastType) -> pd.Series:
        """Prepare target variable based on forecast type"""
        try:
            if forecast_type == ForecastType.PRICE:
                return data['close']
            elif forecast_type == ForecastType.RETURN:
                return data['close'].pct_change()
            elif forecast_type == ForecastType.VOLATILITY:
                return data['close'].rolling(20).std()
            elif forecast_type == ForecastType.VOLUME:
                return data['volume']
            elif forecast_type == ForecastType.DIRECTION:
                return np.sign(data['close'].pct_change())
            else:
                return data['close']
                
        except Exception as e:
            logger.error(f"Error preparing target: {e}")
            return pd.Series()
    
    async def _generate_predictions(self, features: pd.DataFrame, target: pd.Series,
                                  forecast_type: ForecastType, time_horizon: TimeHorizon,
                                  periods: int) -> List[float]:
        """Generate predictions using ensemble models"""
        try:
            # Align features and target
            common_index = features.index.intersection(target.index)
            features = features.loc[common_index]
            target = target.loc[common_index]
            
            if len(features) < 50:
                raise ValueError("Insufficient data for training")
            
            # Train models
            await self._train_models(features, target)
            
            # Generate predictions for future periods
            predictions = []
            current_features = features.iloc[-1:].copy()
            
            for i in range(periods):
                # Predict next value
                pred = await self._predict_with_ensemble(current_features)
                predictions.append(float(pred[0]))
                
                # Update features for next prediction (simplified)
                # In practice, this would involve more sophisticated feature updating
                if forecast_type == ForecastType.PRICE:
                    # Update price-based features
                    current_features.iloc[0, current_features.columns.get_loc('price')] = pred[0]
                    if 'price_change' in current_features.columns:
                        last_price = features.iloc[-1]['price']
                        current_features.iloc[0, current_features.columns.get_loc('price_change')] = (pred[0] - last_price) / last_price
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return [0.0] * periods
    
    async def _train_models(self, features: pd.DataFrame, target: pd.Series):
        """Train ensemble models"""
        try:
            # Prepare data
            X = features.select_dtypes(include=[np.number]).fillna(0)
            y = target.fillna(method='ffill').fillna(0)
            
            # Align X and y
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            if len(X) < 20:
                logger.warning("Insufficient data for training")
                return
            
            # Train each model
            for model_type, model in self.models.items():
                try:
                    # Scale features
                    scaler = self.scalers[model_type]
                    X_scaled = scaler.fit_transform(X)
                    
                    # Train model
                    model.fit(X_scaled, y)
                    
                except Exception as e:
                    logger.error(f"Error training {model_type}: {e}")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
    
    async def _predict_with_ensemble(self, features: pd.DataFrame) -> np.ndarray:
        """Generate predictions using ensemble of models"""
        try:
            X = features.select_dtypes(include=[np.number]).fillna(0)
            predictions = []
            weights = []
            
            for model_type, model in self.models.items():
                try:
                    # Scale features
                    scaler = self.scalers[model_type]
                    X_scaled = scaler.transform(X)
                    
                    # Generate prediction
                    pred = model.predict(X_scaled)
                    predictions.append(pred)
                    weights.append(self.ensemble_weights.get(model_type, 0.1))
                    
                except Exception as e:
                    logger.error(f"Error predicting with {model_type}: {e}")
            
            if not predictions:
                return np.array([0.0])
            
            # Weighted ensemble
            predictions = np.array(predictions)
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize weights
            
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return np.array([0.0])
    
    def _calculate_confidence_intervals(self, predictions: List[float], 
                                      features: pd.DataFrame, target: pd.Series) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for predictions"""
        try:
            # Simple confidence interval based on historical volatility
            if len(target) > 20:
                volatility = target.pct_change().std()
            else:
                volatility = 0.02  # Default 2% volatility
            
            confidence_intervals = []
            for pred in predictions:
                lower = pred * (1 - 1.96 * volatility)
                upper = pred * (1 + 1.96 * volatility)
                confidence_intervals.append((lower, upper))
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {e}")
            return [(pred * 0.95, pred * 1.05) for pred in predictions]
    
    def _generate_timestamps(self, data: pd.DataFrame, timeframe: str, periods: int) -> List[datetime]:
        """Generate timestamps for predictions"""
        try:
            last_timestamp = data.index[-1]
            
            # Determine frequency based on timeframe
            if timeframe == '1d':
                freq = 'D'
            elif timeframe == '1h':
                freq = 'H'
            elif timeframe == '1m':
                freq = 'T'
            else:
                freq = 'D'
            
            # Generate future timestamps
            future_timestamps = pd.date_range(
                start=last_timestamp + pd.Timedelta(days=1),
                periods=periods,
                freq=freq
            )
            
            return future_timestamps.to_list()
            
        except Exception as e:
            logger.error(f"Error generating timestamps: {e}")
            return [datetime.now() + timedelta(days=i+1) for i in range(periods)]
    
    async def _calculate_metrics(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, float]:
        """Calculate model performance metrics"""
        try:
            if len(features) < 20 or len(target) < 20:
                return {'confidence': 0.5}
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            X = features.select_dtypes(include=[np.number]).fillna(0)
            y = target.fillna(method='ffill').fillna(0)
            
            # Align X and y
            common_index = X.index.intersection(y.index)
            X = X.loc[common_index]
            y = y.loc[common_index]
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Train a simple model for validation
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                scaler = StandardScaler()
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                score = r2_score(y_test, y_pred)
                scores.append(score)
            
            avg_score = np.mean(scores)
            confidence = max(0.0, min(1.0, (avg_score + 1) / 2))  # Convert R² to 0-1 range
            
            return {
                'r2_score': float(avg_score),
                'confidence': float(confidence),
                'cv_scores': [float(s) for s in scores]
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'confidence': 0.5}
    
    def _get_feature_importance(self, forecast_type: ForecastType) -> Dict[str, float]:
        """Get feature importance from trained models"""
        try:
            importance_dict = {}
            
            # Get importance from Random Forest model
            rf_model = self.models.get(ModelType.RANDOM_FOREST)
            if rf_model and hasattr(rf_model, 'feature_importances_'):
                # This would need feature names from the last training
                # For now, return generic importance
                importance_dict = {
                    'price_features': 0.3,
                    'technical_indicators': 0.25,
                    'volume_features': 0.2,
                    'momentum_features': 0.15,
                    'volatility_features': 0.1
                }
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _create_fallback_forecast(self, symbol: str, forecast_type: ForecastType, 
                                 time_horizon: TimeHorizon, periods: int = 5) -> ForecastResult:
        """Create fallback forecast when main forecasting fails"""
        try:
            # Generate simple predictions (e.g., random walk)
            base_value = 100.0  # Default base value
            predictions = [base_value * (1 + np.random.normal(0, 0.01)) for _ in range(periods)]
            
            timestamps = [datetime.now() + timedelta(days=i+1) for i in range(periods)]
            confidence_intervals = [(pred * 0.95, pred * 1.05) for pred in predictions]
            
            return ForecastResult(
                symbol=symbol,
                forecast_type=forecast_type,
                time_horizon=time_horizon,
                predictions=predictions,
                timestamps=timestamps,
                confidence_intervals=confidence_intervals,
                confidence_score=0.3,
                model_used='fallback',
                feature_importance={},
                metrics={'confidence': 0.3},
                metadata={'fallback': True},
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error creating fallback forecast: {e}")
            return ForecastResult(
                symbol=symbol,
                forecast_type=forecast_type,
                time_horizon=time_horizon,
                predictions=[100.0] * periods,
                timestamps=[datetime.now()] * periods,
                confidence_intervals=[(95.0, 105.0)] * periods,
                confidence_score=0.1,
                model_used='error_fallback',
                feature_importance={},
                metrics={},
                metadata={'error': str(e)},
                created_at=datetime.now()
            )
    
    # Technical indicator calculations
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index, dtype=float)
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            return macd, macd_signal
        except:
            return pd.Series(index=prices.index, dtype=float), pd.Series(index=prices.index, dtype=float)
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        try:
            rolling_mean = prices.rolling(window=window).mean()
            rolling_std = prices.rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std)
            lower_band = rolling_mean - (rolling_std * num_std)
            return upper_band, lower_band
        except:
            return pd.Series(index=prices.index, dtype=float), pd.Series(index=prices.index, dtype=float)
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        try:
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            return pd.Series(true_range, index=data.index)
        except:
            return pd.Series(index=data.index, dtype=float)