import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from scipy.optimize import minimize
import json
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossAssetModelType(Enum):
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    FACTOR_MODEL = "factor_model"
    REGIME_SWITCHING = "regime_switching"
    CORRELATION_MODEL = "correlation_model"
    VOLATILITY_MODEL = "volatility_model"

class CrossAssetRegime(Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    CRISIS = "crisis"
    RECOVERY = "recovery"
    NEUTRAL = "neutral"

@dataclass
class CrossAssetData:
    """Comprehensive cross-asset data structure"""
    asset_prices: Dict[str, List[float]]  # Asset symbol -> price series
    asset_returns: Dict[str, List[float]]  # Asset symbol -> return series
    timestamps: List[datetime]
    volume: Dict[str, List[float]]  # Asset symbol -> volume series
    market_data: Dict[str, Any]  # Additional market data
    
    # Cross-asset specific data
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    volatilities: Optional[Dict[str, float]] = None
    factor_loadings: Optional[Dict[str, Dict[str, float]]] = None
    regime_indicators: Optional[List[int]] = None
    
    # Economic and sentiment data
    economic_indicators: Optional[Dict[str, List[float]]] = None
    sentiment_scores: Optional[List[float]] = None
    news_sentiment: Optional[List[float]] = None
    
    # Risk factors
    risk_factors: Optional[Dict[str, List[float]]] = None
    macro_indicators: Optional[Dict[str, List[float]]] = None

@dataclass
class CrossAssetMLPredictionResult:
    """Result from cross-asset ML prediction"""
    model_type: CrossAssetModelType
    predictions: Dict[str, List[float]]  # Asset -> predictions
    confidence_intervals: Dict[str, Tuple[List[float], List[float]]]  # Asset -> (lower, upper)
    feature_importance: Dict[str, float]
    model_performance: Dict[str, float]
    
    # Cross-asset specific metrics
    correlation_predictions: Dict[str, Dict[str, float]]
    regime_probabilities: Optional[Dict[CrossAssetRegime, List[float]]] = None
    factor_exposures: Optional[Dict[str, Dict[str, float]]] = None
    
    # Risk metrics
    portfolio_var: Optional[float] = None
    diversification_ratio: Optional[float] = None
    concentration_risk: Optional[float] = None
    
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CrossAssetEnsembleResult:
    """Result from cross-asset ensemble prediction"""
    ensemble_predictions: Dict[str, List[float]]  # Asset -> ensemble predictions
    individual_predictions: Dict[str, Dict[str, List[float]]]  # Model -> Asset -> predictions
    model_weights: Dict[str, float]
    ensemble_confidence: Dict[str, List[float]]  # Asset -> confidence scores
    
    # Cross-asset ensemble metrics
    cross_asset_signals: Dict[str, int]  # Asset -> signal (-1, 0, 1)
    portfolio_allocation: Dict[str, float]  # Asset -> weight
    risk_adjusted_returns: Dict[str, float]  # Asset -> expected return
    
    # Ensemble diagnostics
    model_agreement: float
    prediction_stability: float
    regime_consistency: float
    
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

class CrossAssetLSTMModel:
    """LSTM model for cross-asset prediction (statistical implementation)"""
    
    def __init__(self, lookback_window: int = 60, prediction_horizon: int = 5,
                 hidden_units: int = 50, dropout_rate: float = 0.2):
        """
        Initialize Cross-Asset LSTM model
        
        Args:
            lookback_window: Number of time steps to look back
            prediction_horizon: Number of steps to predict ahead
            hidden_units: Number of hidden units (simulated)
            dropout_rate: Dropout rate for regularization
        """
        self.lookback_window = lookback_window
        self.prediction_horizon = prediction_horizon
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        # Statistical models to simulate LSTM behavior
        self.asset_models = {}
        self.correlation_model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(f"Initialized CrossAssetLSTMModel with lookback: {lookback_window}, horizon: {prediction_horizon}")
    
    def _create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""
        X, y = [], []
        for i in range(lookback, len(data) - self.prediction_horizon + 1):
            X.append(data[i-lookback:i])
            y.append(data[i:i+self.prediction_horizon])
        return np.array(X), np.array(y)
    
    def _extract_cross_asset_features(self, data: CrossAssetData) -> np.ndarray:
        """Extract cross-asset features for LSTM"""
        features = []
        asset_names = list(data.asset_prices.keys())
        
        # Price-based features
        for asset in asset_names:
            prices = np.array(data.asset_prices[asset])
            returns = np.array(data.asset_returns[asset])
            
            # Technical features
            sma_20 = pd.Series(prices).rolling(20).mean().fillna(method='bfill')
            sma_50 = pd.Series(prices).rolling(50).mean().fillna(method='bfill')
            rsi = self._calculate_rsi(pd.Series(prices))
            
            features.extend([prices, returns, sma_20.values, sma_50.values, rsi.values])
        
        # Cross-asset features
        if len(asset_names) > 1:
            # Correlation features
            for i, asset1 in enumerate(asset_names):
                for j, asset2 in enumerate(asset_names[i+1:], i+1):
                    returns1 = np.array(data.asset_returns[asset1])
                    returns2 = np.array(data.asset_returns[asset2])
                    
                    # Rolling correlation
                    rolling_corr = pd.Series(returns1).rolling(30).corr(pd.Series(returns2)).fillna(0)
                    features.append(rolling_corr.values)
            
            # Relative strength features
            base_asset = asset_names[0]
            base_prices = np.array(data.asset_prices[base_asset])
            
            for asset in asset_names[1:]:
                asset_prices = np.array(data.asset_prices[asset])
                relative_strength = asset_prices / base_prices
                features.append(relative_strength)
        
        # Economic indicators if available
        if data.economic_indicators:
            for indicator_name, values in data.economic_indicators.items():
                features.append(np.array(values))
        
        # Sentiment features if available
        if data.sentiment_scores:
            features.append(np.array(data.sentiment_scores))
        
        # Ensure all features have the same length
        min_length = min(len(f) for f in features)
        features = [f[:min_length] for f in features]
        
        return np.column_stack(features)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def fit(self, data: CrossAssetData) -> None:
        """Train the cross-asset LSTM model"""
        try:
            logger.info("Training CrossAssetLSTMModel...")
            
            # Extract features
            feature_matrix = self._extract_cross_asset_features(data)
            
            # Scale features
            feature_matrix_scaled = self.scaler.fit_transform(feature_matrix)
            
            # Train individual asset models
            asset_names = list(data.asset_prices.keys())
            
            for asset in asset_names:
                logger.info(f"Training model for {asset}")
                
                # Prepare target variable (returns)
                target = np.array(data.asset_returns[asset])
                
                # Ensure same length
                min_length = min(len(feature_matrix_scaled), len(target))
                X = feature_matrix_scaled[:min_length]
                y = target[:min_length]
                
                # Create sequences
                if len(X) > self.lookback_window + self.prediction_horizon:
                    X_seq, y_seq = self._create_sequences(X, self.lookback_window)
                    
                    # Flatten sequences for traditional ML model
                    X_flat = X_seq.reshape(X_seq.shape[0], -1)
                    y_flat = y_seq.mean(axis=1)  # Average over prediction horizon
                    
                    # Use ensemble of models to simulate LSTM
                    models = {
                        'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                        'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                        'ridge': Ridge(alpha=1.0)
                    }
                    
                    trained_models = {}
                    for name, model in models.items():
                        try:
                            model.fit(X_flat, y_flat)
                            trained_models[name] = model
                        except Exception as e:
                            logger.warning(f"Failed to train {name} for {asset}: {e}")
                    
                    self.asset_models[asset] = trained_models
                else:
                    logger.warning(f"Insufficient data for {asset}")
                    # Fallback to simple model
                    simple_model = Ridge(alpha=1.0)
                    simple_model.fit(X, y)
                    self.asset_models[asset] = {'ridge': simple_model}
            
            # Train correlation model
            if len(asset_names) > 1:
                correlation_features = []
                correlation_targets = []
                
                for i, asset1 in enumerate(asset_names):
                    for j, asset2 in enumerate(asset_names[i+1:], i+1):
                        returns1 = np.array(data.asset_returns[asset1])
                        returns2 = np.array(data.asset_returns[asset2])
                        
                        # Rolling correlation as target
                        rolling_corr = pd.Series(returns1).rolling(30).corr(pd.Series(returns2)).fillna(0)
                        
                        # Features for correlation prediction
                        corr_features = np.column_stack([
                            returns1, returns2,
                            pd.Series(returns1).rolling(10).std().fillna(0),
                            pd.Series(returns2).rolling(10).std().fillna(0)
                        ])
                        
                        min_length = min(len(corr_features), len(rolling_corr))
                        correlation_features.append(corr_features[:min_length])
                        correlation_targets.append(rolling_corr.values[:min_length])
                
                if correlation_features:
                    # Combine all correlation data
                    all_corr_features = np.vstack(correlation_features)
                    all_corr_targets = np.concatenate(correlation_targets)
                    
                    self.correlation_model = RandomForestRegressor(n_estimators=50, random_state=42)
                    self.correlation_model.fit(all_corr_features, all_corr_targets)
            
            self.is_fitted = True
            logger.info("CrossAssetLSTMModel training completed")
            
        except Exception as e:
            logger.error(f"Error training CrossAssetLSTMModel: {e}")
            raise
    
    def predict(self, data: CrossAssetData) -> CrossAssetMLPredictionResult:
        """Make cross-asset predictions"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            logger.info("Making CrossAssetLSTM predictions...")
            
            # Extract features
            feature_matrix = self._extract_cross_asset_features(data)
            feature_matrix_scaled = self.scaler.transform(feature_matrix)
            
            predictions = {}
            confidence_intervals = {}
            feature_importance = {}
            model_performance = {}
            correlation_predictions = {}
            
            asset_names = list(data.asset_prices.keys())
            
            # Make predictions for each asset
            for asset in asset_names:
                if asset in self.asset_models:
                    asset_predictions = []
                    asset_models = self.asset_models[asset]
                    
                    # Get predictions from each model in ensemble
                    model_preds = []
                    for model_name, model in asset_models.items():
                        try:
                            if hasattr(model, 'predict'):
                                # Use recent data for prediction
                                recent_data = feature_matrix_scaled[-self.lookback_window:]
                                
                                if len(recent_data.shape) == 2:
                                    # Flatten for traditional ML models
                                    pred_input = recent_data.flatten().reshape(1, -1)
                                    pred = model.predict(pred_input)[0]
                                    model_preds.append(pred)
                                    
                                    # Feature importance (for tree-based models)
                                    if hasattr(model, 'feature_importances_'):
                                        importance = np.mean(model.feature_importances_)
                                        feature_importance[f"{asset}_{model_name}"] = importance
                        except Exception as e:
                            logger.warning(f"Prediction failed for {asset} {model_name}: {e}")
                    
                    if model_preds:
                        # Ensemble prediction (average)
                        ensemble_pred = np.mean(model_preds)
                        pred_std = np.std(model_preds) if len(model_preds) > 1 else 0.01
                        
                        # Generate prediction sequence
                        pred_sequence = [ensemble_pred * (1 + np.random.normal(0, 0.1)) 
                                       for _ in range(self.prediction_horizon)]
                        
                        predictions[asset] = pred_sequence
                        
                        # Confidence intervals
                        lower_bound = [p - 2*pred_std for p in pred_sequence]
                        upper_bound = [p + 2*pred_std for p in pred_sequence]
                        confidence_intervals[asset] = (lower_bound, upper_bound)
                        
                        # Model performance (simplified)
                        model_performance[asset] = {
                            'ensemble_std': pred_std,
                            'model_count': len(model_preds),
                            'prediction_strength': abs(ensemble_pred)
                        }
            
            # Predict correlations
            if self.correlation_model and len(asset_names) > 1:
                for i, asset1 in enumerate(asset_names):
                    correlation_predictions[asset1] = {}
                    for j, asset2 in enumerate(asset_names):
                        if i != j:
                            try:
                                # Use recent returns for correlation prediction
                                recent_returns1 = np.array(data.asset_returns[asset1][-30:])
                                recent_returns2 = np.array(data.asset_returns[asset2][-30:])
                                
                                if len(recent_returns1) > 0 and len(recent_returns2) > 0:
                                    corr_input = np.array([
                                        recent_returns1[-1], recent_returns2[-1],
                                        np.std(recent_returns1), np.std(recent_returns2)
                                    ]).reshape(1, -1)
                                    
                                    pred_corr = self.correlation_model.predict(corr_input)[0]
                                    correlation_predictions[asset1][asset2] = np.clip(pred_corr, -1, 1)
                            except Exception as e:
                                correlation_predictions[asset1][asset2] = 0.0
            
            # Calculate portfolio-level metrics
            portfolio_var = None
            diversification_ratio = None
            
            if len(predictions) > 1:
                # Simple portfolio VaR calculation
                pred_returns = [np.mean(preds) for preds in predictions.values()]
                portfolio_return = np.mean(pred_returns)
                portfolio_std = np.std(pred_returns)
                portfolio_var = portfolio_return - 2 * portfolio_std  # 95% VaR approximation
                
                # Diversification ratio (simplified)
                individual_risks = [np.std(preds) for preds in predictions.values()]
                avg_individual_risk = np.mean(individual_risks)
                diversification_ratio = avg_individual_risk / portfolio_std if portfolio_std > 0 else 1.0
            
            result = CrossAssetMLPredictionResult(
                model_type=CrossAssetModelType.LSTM,
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                feature_importance=feature_importance,
                model_performance=model_performance,
                correlation_predictions=correlation_predictions,
                portfolio_var=portfolio_var,
                diversification_ratio=diversification_ratio,
                metadata={
                    'lookback_window': self.lookback_window,
                    'prediction_horizon': self.prediction_horizon,
                    'assets_predicted': list(predictions.keys()),
                    'total_features': feature_matrix_scaled.shape[1] if len(feature_matrix_scaled.shape) > 1 else 0
                }
            )
            
            logger.info(f"CrossAssetLSTM predictions completed for {len(predictions)} assets")
            return result
            
        except Exception as e:
            logger.error(f"Error making CrossAssetLSTM predictions: {e}")
            return CrossAssetMLPredictionResult(
                model_type=CrossAssetModelType.LSTM,
                predictions={},
                confidence_intervals={},
                feature_importance={},
                model_performance={},
                correlation_predictions={}
            )

class CrossAssetTransformerModel:
    """Transformer model for cross-asset prediction (statistical implementation)"""
    
    def __init__(self, sequence_length: int = 60, d_model: int = 64, 
                 num_heads: int = 8, num_layers: int = 4):
        """
        Initialize Cross-Asset Transformer model
        
        Args:
            sequence_length: Length of input sequences
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Statistical models to simulate transformer behavior
        self.attention_models = {}
        self.position_encodings = {}
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=min(d_model, 50))
        self.is_fitted = False
        
        logger.info(f"Initialized CrossAssetTransformerModel with seq_len: {sequence_length}, d_model: {d_model}")
    
    def _create_attention_features(self, data: CrossAssetData) -> Dict[str, np.ndarray]:
        """Create attention-like features for cross-asset analysis"""
        attention_features = {}
        asset_names = list(data.asset_prices.keys())
        
        for asset in asset_names:
            features = []
            
            # Self-attention features (asset's own history)
            prices = np.array(data.asset_prices[asset])
            returns = np.array(data.asset_returns[asset])
            
            # Multi-scale moving averages (simulating different attention heads)
            for window in [5, 10, 20, 50]:
                ma = pd.Series(prices).rolling(window).mean().fillna(method='bfill')
                features.append(ma.values)
            
            # Volatility features at different scales
            for window in [5, 10, 20]:
                vol = pd.Series(returns).rolling(window).std().fillna(0.01)
                features.append(vol.values)
            
            # Cross-attention features (relationships with other assets)
            for other_asset in asset_names:
                if other_asset != asset:
                    other_returns = np.array(data.asset_returns[other_asset])
                    
                    # Rolling correlation (attention to other assets)
                    rolling_corr = pd.Series(returns).rolling(20).corr(pd.Series(other_returns)).fillna(0)
                    features.append(rolling_corr.values)
                    
                    # Relative performance (attention to relative strength)
                    other_prices = np.array(data.asset_prices[other_asset])
                    relative_perf = prices / other_prices
                    features.append(relative_perf)
            
            # Position encoding (time-based features)
            time_features = []
            for i, timestamp in enumerate(data.timestamps):
                # Cyclical time features
                day_of_week = np.sin(2 * np.pi * timestamp.weekday() / 7)
                month = np.sin(2 * np.pi * timestamp.month / 12)
                time_features.append([day_of_week, month])
            
            if time_features:
                time_array = np.array(time_features)
                features.extend([time_array[:, 0], time_array[:, 1]])
            
            # Ensure all features have the same length
            min_length = min(len(f) for f in features)
            features = [f[:min_length] for f in features]
            
            attention_features[asset] = np.column_stack(features)
        
        return attention_features
    
    def fit(self, data: CrossAssetData) -> None:
        """Train the cross-asset transformer model"""
        try:
            logger.info("Training CrossAssetTransformerModel...")
            
            # Create attention features
            attention_features = self._create_attention_features(data)
            asset_names = list(data.asset_prices.keys())
            
            # Train models for each asset
            for asset in asset_names:
                logger.info(f"Training transformer for {asset}")
                
                features = attention_features[asset]
                target = np.array(data.asset_returns[asset])
                
                # Ensure same length
                min_length = min(len(features), len(target))
                X = features[:min_length]
                y = target[:min_length]
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # Apply PCA for dimensionality reduction (simulating transformer compression)
                X_compressed = self.pca.fit_transform(X_scaled)
                
                # Create sequences
                if len(X_compressed) > self.sequence_length:
                    X_seq, y_seq = [], []
                    for i in range(self.sequence_length, len(X_compressed)):
                        X_seq.append(X_compressed[i-self.sequence_length:i])
                        y_seq.append(y[i])
                    
                    X_seq = np.array(X_seq)
                    y_seq = np.array(y_seq)
                    
                    # Flatten sequences for traditional ML (simulating transformer output)
                    X_flat = X_seq.reshape(X_seq.shape[0], -1)
                    
                    # Multi-head attention simulation using multiple models
                    attention_models = {}
                    
                    for head in range(self.num_heads):
                        # Each "head" focuses on different aspects
                        if head % 3 == 0:
                            # Price trend head
                            model = GradientBoostingRegressor(n_estimators=50, random_state=42+head)
                        elif head % 3 == 1:
                            # Volatility head
                            model = RandomForestRegressor(n_estimators=50, random_state=42+head)
                        else:
                            # Cross-asset head
                            model = Ridge(alpha=1.0, random_state=42+head)
                        
                        try:
                            model.fit(X_flat, y_seq)
                            attention_models[f'head_{head}'] = model
                        except Exception as e:
                            logger.warning(f"Failed to train head {head} for {asset}: {e}")
                    
                    self.attention_models[asset] = attention_models
                else:
                    logger.warning(f"Insufficient data for {asset}")
                    # Fallback model
                    fallback_model = Ridge(alpha=1.0)
                    fallback_model.fit(X_compressed, y)
                    self.attention_models[asset] = {'head_0': fallback_model}
            
            self.is_fitted = True
            logger.info("CrossAssetTransformerModel training completed")
            
        except Exception as e:
            logger.error(f"Error training CrossAssetTransformerModel: {e}")
            raise
    
    def predict(self, data: CrossAssetData) -> CrossAssetMLPredictionResult:
        """Make cross-asset predictions using transformer"""
        try:
            if not self.is_fitted:
                raise ValueError("Model must be fitted before prediction")
            
            logger.info("Making CrossAssetTransformer predictions...")
            
            # Create attention features
            attention_features = self._create_attention_features(data)
            
            predictions = {}
            confidence_intervals = {}
            feature_importance = {}
            model_performance = {}
            
            asset_names = list(data.asset_prices.keys())
            
            for asset in asset_names:
                if asset in self.attention_models:
                    features = attention_features[asset]
                    
                    # Scale and compress features
                    X_scaled = self.scaler.transform(features)
                    X_compressed = self.pca.transform(X_scaled)
                    
                    # Use recent sequence for prediction
                    if len(X_compressed) >= self.sequence_length:
                        recent_sequence = X_compressed[-self.sequence_length:]
                        pred_input = recent_sequence.flatten().reshape(1, -1)
                        
                        # Get predictions from all attention heads
                        head_predictions = []
                        attention_weights = []
                        
                        for head_name, model in self.attention_models[asset].items():
                            try:
                                pred = model.predict(pred_input)[0]
                                head_predictions.append(pred)
                                
                                # Attention weight (simplified)
                                if hasattr(model, 'feature_importances_'):
                                    weight = np.mean(model.feature_importances_)
                                else:
                                    weight = 1.0 / len(self.attention_models[asset])
                                
                                attention_weights.append(weight)
                                
                            except Exception as e:
                                logger.warning(f"Prediction failed for {asset} {head_name}: {e}")
                        
                        if head_predictions:
                            # Weighted ensemble of attention heads
                            if attention_weights:
                                weights = np.array(attention_weights)
                                weights = weights / np.sum(weights)  # Normalize
                                ensemble_pred = np.average(head_predictions, weights=weights)
                            else:
                                ensemble_pred = np.mean(head_predictions)
                            
                            pred_std = np.std(head_predictions) if len(head_predictions) > 1 else 0.01
                            
                            predictions[asset] = [ensemble_pred]
                            
                            # Confidence intervals
                            lower_bound = [ensemble_pred - 2*pred_std]
                            upper_bound = [ensemble_pred + 2*pred_std]
                            confidence_intervals[asset] = (lower_bound, upper_bound)
                            
                            # Feature importance (attention scores)
                            feature_importance[asset] = {
                                'attention_diversity': pred_std,
                                'head_count': len(head_predictions),
                                'prediction_strength': abs(ensemble_pred)
                            }
                            
                            # Model performance
                            model_performance[asset] = {
                                'attention_consistency': 1.0 - pred_std,
                                'head_agreement': 1.0 / (1.0 + pred_std),
                                'sequence_length': self.sequence_length
                            }
            
            result = CrossAssetMLPredictionResult(
                model_type=CrossAssetModelType.TRANSFORMER,
                predictions=predictions,
                confidence_intervals=confidence_intervals,
                feature_importance=feature_importance,
                model_performance=model_performance,
                correlation_predictions={},
                metadata={
                    'sequence_length': self.sequence_length,
                    'd_model': self.d_model,
                    'num_heads': self.num_heads,
                    'assets_predicted': list(predictions.keys())
                }
            )
            
            logger.info(f"CrossAssetTransformer predictions completed for {len(predictions)} assets")
            return result
            
        except Exception as e:
            logger.error(f"Error making CrossAssetTransformer predictions: {e}")
            return CrossAssetMLPredictionResult(
                model_type=CrossAssetModelType.TRANSFORMER,
                predictions={},
                confidence_intervals={},
                feature_importance={},
                model_performance={},
                correlation_predictions={}
            )

class CrossAssetAdvancedMLModels:
    """Advanced ML models for cross-asset analysis"""
    
    def __init__(self):
        """
        Initialize advanced cross-asset ML models
        """
        self.lstm_model = CrossAssetLSTMModel()
        self.transformer_model = CrossAssetTransformerModel()
        self.ensemble_weights = {}
        self.regime_detector = None
        self.is_fitted = False
        
        logger.info("Initialized CrossAssetAdvancedMLModels")
    
    def fit(self, data: CrossAssetData) -> None:
        """Train all advanced ML models"""
        try:
            logger.info("Training all CrossAsset advanced ML models...")
            
            # Train individual models
            self.lstm_model.fit(data)
            self.transformer_model.fit(data)
            
            # Train regime detector
            self._train_regime_detector(data)
            
            # Calculate ensemble weights based on validation performance
            self._calculate_ensemble_weights(data)
            
            self.is_fitted = True
            logger.info("All CrossAsset advanced ML models trained successfully")
            
        except Exception as e:
            logger.error(f"Error training CrossAsset advanced ML models: {e}")
            raise
    
    def _train_regime_detector(self, data: CrossAssetData) -> None:
        """Train regime detection model"""
        try:
            # Create regime features
            regime_features = []
            asset_names = list(data.asset_prices.keys())
            
            # Market-wide features for regime detection
            all_returns = []
            all_volatilities = []
            
            for asset in asset_names:
                returns = np.array(data.asset_returns[asset])
                all_returns.append(returns)
                
                # Rolling volatility
                vol = pd.Series(returns).rolling(20).std().fillna(0.01)
                all_volatilities.append(vol.values)
            
            if all_returns:
                # Market-wide metrics
                market_returns = np.mean(all_returns, axis=0)
                market_vol = np.mean(all_volatilities, axis=0)
                
                # Cross-asset correlations
                if len(all_returns) > 1:
                    correlation_matrix = np.corrcoef(all_returns)
                    avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
                    correlation_series = [avg_correlation] * len(market_returns)
                else:
                    correlation_series = [0.0] * len(market_returns)
                
                # Combine features
                regime_features = np.column_stack([
                    market_returns,
                    market_vol,
                    correlation_series
                ])
                
                # Use clustering to identify regimes
                self.regime_detector = KMeans(n_clusters=3, random_state=42)
                self.regime_detector.fit(regime_features)
                
                logger.info("Regime detector trained successfully")
        
        except Exception as e:
            logger.warning(f"Failed to train regime detector: {e}")
            self.regime_detector = None
    
    def _calculate_ensemble_weights(self, data: CrossAssetData) -> None:
        """Calculate ensemble weights based on validation performance"""
        try:
            # Simple validation split
            split_idx = int(0.8 * len(data.timestamps))
            
            # Create validation data
            val_data = CrossAssetData(
                asset_prices={k: v[split_idx:] for k, v in data.asset_prices.items()},
                asset_returns={k: v[split_idx:] for k, v in data.asset_returns.items()},
                timestamps=data.timestamps[split_idx:],
                volume={k: v[split_idx:] if k in data.volume else [] for k in data.asset_prices.keys()}
            )
            
            # Get predictions from both models
            lstm_pred = self.lstm_model.predict(val_data)
            transformer_pred = self.transformer_model.predict(val_data)
            
            # Calculate performance scores
            lstm_score = 0.0
            transformer_score = 0.0
            
            for asset in data.asset_prices.keys():
                if asset in lstm_pred.predictions and asset in transformer_pred.predictions:
                    # Simple scoring based on prediction confidence
                    lstm_conf = lstm_pred.model_performance.get(asset, {}).get('prediction_strength', 0.0)
                    transformer_conf = transformer_pred.model_performance.get(asset, {}).get('prediction_strength', 0.0)
                    
                    lstm_score += abs(lstm_conf)
                    transformer_score += abs(transformer_conf)
            
            # Normalize weights
            total_score = lstm_score + transformer_score
            if total_score > 0:
                self.ensemble_weights = {
                    'lstm': lstm_score / total_score,
                    'transformer': transformer_score / total_score
                }
            else:
                self.ensemble_weights = {'lstm': 0.5, 'transformer': 0.5}
            
            logger.info(f"Ensemble weights calculated: {self.ensemble_weights}")
            
        except Exception as e:
            logger.warning(f"Failed to calculate ensemble weights: {e}")
            self.ensemble_weights = {'lstm': 0.5, 'transformer': 0.5}
    
    def predict(self, data: CrossAssetData) -> CrossAssetEnsembleResult:
        """Make ensemble predictions using all models"""
        try:
            if not self.is_fitted:
                raise ValueError("Models must be fitted before prediction")
            
            logger.info("Making CrossAsset ensemble predictions...")
            
            # Get predictions from individual models
            lstm_result = self.lstm_model.predict(data)
            transformer_result = self.transformer_model.predict(data)
            
            # Detect current regime
            current_regime = self._detect_current_regime(data)
            
            # Combine predictions
            ensemble_predictions = {}
            individual_predictions = {
                'lstm': lstm_result.predictions,
                'transformer': transformer_result.predictions
            }
            
            ensemble_confidence = {}
            cross_asset_signals = {}
            
            # Ensemble for each asset
            for asset in data.asset_prices.keys():
                lstm_pred = lstm_result.predictions.get(asset, [0.0])
                transformer_pred = transformer_result.predictions.get(asset, [0.0])
                
                if lstm_pred and transformer_pred:
                    # Weighted ensemble
                    lstm_weight = self.ensemble_weights.get('lstm', 0.5)
                    transformer_weight = self.ensemble_weights.get('transformer', 0.5)
                    
                    # Adjust weights based on regime
                    if current_regime == CrossAssetRegime.CRISIS:
                        # In crisis, favor more conservative LSTM
                        lstm_weight *= 1.2
                        transformer_weight *= 0.8
                    elif current_regime == CrossAssetRegime.RECOVERY:
                        # In recovery, favor transformer's attention mechanism
                        lstm_weight *= 0.8
                        transformer_weight *= 1.2
                    
                    # Renormalize
                    total_weight = lstm_weight + transformer_weight
                    lstm_weight /= total_weight
                    transformer_weight /= total_weight
                    
                    # Ensemble prediction
                    ensemble_pred = [
                        lstm_weight * l + transformer_weight * t 
                        for l, t in zip(lstm_pred, transformer_pred)
                    ]
                    
                    ensemble_predictions[asset] = ensemble_pred
                    
                    # Confidence based on model agreement
                    agreement = 1.0 - abs(np.mean(lstm_pred) - np.mean(transformer_pred))
                    ensemble_confidence[asset] = [agreement] * len(ensemble_pred)
                    
                    # Generate trading signal
                    avg_pred = np.mean(ensemble_pred)
                    if avg_pred > 0.001:  # 0.1% threshold
                        cross_asset_signals[asset] = 1  # Buy
                    elif avg_pred < -0.001:
                        cross_asset_signals[asset] = -1  # Sell
                    else:
                        cross_asset_signals[asset] = 0  # Hold
            
            # Calculate portfolio allocation
            portfolio_allocation = self._calculate_portfolio_allocation(
                ensemble_predictions, cross_asset_signals, data
            )
            
            # Calculate ensemble diagnostics
            model_agreement = self._calculate_model_agreement(
                lstm_result.predictions, transformer_result.predictions
            )
            
            prediction_stability = self._calculate_prediction_stability(ensemble_predictions)
            
            regime_consistency = 1.0 if current_regime != CrossAssetRegime.NEUTRAL else 0.5
            
            result = CrossAssetEnsembleResult(
                ensemble_predictions=ensemble_predictions,
                individual_predictions=individual_predictions,
                model_weights=self.ensemble_weights,
                ensemble_confidence=ensemble_confidence,
                cross_asset_signals=cross_asset_signals,
                portfolio_allocation=portfolio_allocation,
                risk_adjusted_returns={asset: np.mean(preds) for asset, preds in ensemble_predictions.items()},
                model_agreement=model_agreement,
                prediction_stability=prediction_stability,
                regime_consistency=regime_consistency,
                metadata={
                    'current_regime': current_regime.value if current_regime else 'unknown',
                    'assets_analyzed': list(ensemble_predictions.keys()),
                    'ensemble_method': 'weighted_average',
                    'regime_adjusted': True
                }
            )
            
            logger.info(f"CrossAsset ensemble predictions completed for {len(ensemble_predictions)} assets")
            return result
            
        except Exception as e:
            logger.error(f"Error making CrossAsset ensemble predictions: {e}")
            return CrossAssetEnsembleResult(
                ensemble_predictions={},
                individual_predictions={},
                model_weights={},
                ensemble_confidence={},
                cross_asset_signals={},
                portfolio_allocation={},
                risk_adjusted_returns={},
                model_agreement=0.0,
                prediction_stability=0.0,
                regime_consistency=0.0
            )
    
    def _detect_current_regime(self, data: CrossAssetData) -> Optional[CrossAssetRegime]:
        """Detect current market regime"""
        try:
            if not self.regime_detector:
                return CrossAssetRegime.NEUTRAL
            
            # Calculate current regime features
            asset_names = list(data.asset_prices.keys())
            recent_returns = []
            recent_volatilities = []
            
            for asset in asset_names:
                returns = np.array(data.asset_returns[asset][-20:])  # Last 20 periods
                if len(returns) > 0:
                    recent_returns.append(returns)
                    vol = np.std(returns)
                    recent_volatilities.append(vol)
            
            if recent_returns:
                market_return = np.mean([np.mean(r) for r in recent_returns])
                market_vol = np.mean(recent_volatilities)
                
                # Calculate correlation
                if len(recent_returns) > 1:
                    correlation_matrix = np.corrcoef([r[-min(len(r), 10):] for r in recent_returns])
                    avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
                else:
                    avg_correlation = 0.0
                
                # Predict regime
                regime_features = np.array([[market_return, market_vol, avg_correlation]])
                regime_cluster = self.regime_detector.predict(regime_features)[0]
                
                # Map cluster to regime (simplified)
                if regime_cluster == 0:
                    return CrossAssetRegime.RISK_ON
                elif regime_cluster == 1:
                    return CrossAssetRegime.RISK_OFF
                else:
                    return CrossAssetRegime.CRISIS
            
            return CrossAssetRegime.NEUTRAL
            
        except Exception as e:
            logger.warning(f"Failed to detect regime: {e}")
            return CrossAssetRegime.NEUTRAL
    
    def _calculate_portfolio_allocation(self, predictions: Dict[str, List[float]], 
                                      signals: Dict[str, int],
                                      data: CrossAssetData) -> Dict[str, float]:
        """Calculate optimal portfolio allocation"""
        try:
            allocation = {}
            
            # Simple equal-weight allocation with signal adjustment
            num_assets = len(predictions)
            base_weight = 1.0 / num_assets if num_assets > 0 else 0.0
            
            for asset in predictions.keys():
                signal = signals.get(asset, 0)
                pred_return = np.mean(predictions[asset])
                
                # Adjust weight based on signal and expected return
                if signal == 1:  # Buy signal
                    weight = base_weight * (1.0 + abs(pred_return))
                elif signal == -1:  # Sell signal
                    weight = base_weight * (1.0 - abs(pred_return))
                else:  # Hold
                    weight = base_weight
                
                allocation[asset] = max(0.0, min(1.0, weight))  # Clamp between 0 and 1
            
            # Normalize to sum to 1
            total_weight = sum(allocation.values())
            if total_weight > 0:
                allocation = {k: v/total_weight for k, v in allocation.items()}
            
            return allocation
            
        except Exception as e:
            logger.warning(f"Failed to calculate portfolio allocation: {e}")
            return {asset: 1.0/len(predictions) for asset in predictions.keys()}
    
    def _calculate_model_agreement(self, lstm_preds: Dict[str, List[float]], 
                                 transformer_preds: Dict[str, List[float]]) -> float:
        """Calculate agreement between models"""
        try:
            agreements = []
            
            for asset in lstm_preds.keys():
                if asset in transformer_preds:
                    lstm_pred = np.mean(lstm_preds[asset])
                    transformer_pred = np.mean(transformer_preds[asset])
                    
                    # Agreement based on direction and magnitude
                    direction_agreement = 1.0 if np.sign(lstm_pred) == np.sign(transformer_pred) else 0.0
                    magnitude_agreement = 1.0 - abs(lstm_pred - transformer_pred)
                    
                    agreement = 0.7 * direction_agreement + 0.3 * magnitude_agreement
                    agreements.append(agreement)
            
            return np.mean(agreements) if agreements else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate model agreement: {e}")
            return 0.0
    
    def _calculate_prediction_stability(self, predictions: Dict[str, List[float]]) -> float:
        """Calculate stability of predictions"""
        try:
            stabilities = []
            
            for asset, preds in predictions.items():
                if len(preds) > 1:
                    stability = 1.0 - np.std(preds) / (abs(np.mean(preds)) + 1e-6)
                    stabilities.append(max(0.0, stability))
            
            return np.mean(stabilities) if stabilities else 0.0
            
        except Exception as e:
            logger.warning(f"Failed to calculate prediction stability: {e}")
            return 0.0

# Example usage
if __name__ == "__main__":
    # Generate sample cross-asset data
    np.random.seed(42)
    
    n_periods = 500
    assets = ['SPY', 'TLT', 'GLD', 'EURUSD']
    
    # Generate correlated price series
    base_returns = np.random.randn(n_periods) * 0.01
    
    sample_data = CrossAssetData(
        asset_prices={},
        asset_returns={},
        timestamps=[datetime.now() + timedelta(days=i) for i in range(n_periods)],
        volume={}
    )
    
    for i, asset in enumerate(assets):
        # Create correlated returns
        asset_returns = base_returns + np.random.randn(n_periods) * 0.005
        
        # Generate price series
        initial_price = 100.0 + i * 50
        prices = [initial_price]
        
        for ret in asset_returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(new_price)
        
        sample_data.asset_prices[asset] = prices
        sample_data.asset_returns[asset] = [0.0] + asset_returns[1:].tolist()
        sample_data.volume[asset] = np.random.randint(1000, 10000, n_periods).tolist()
    
    # Add economic indicators
    sample_data.economic_indicators = {
        'vix': (20 + np.random.randn(n_periods) * 5).tolist(),
        'yield_spread': (2.0 + np.random.randn(n_periods) * 0.5).tolist()
    }
    
    # Initialize and train models
    print("Initializing CrossAsset Advanced ML Models...")
    models = CrossAssetAdvancedMLModels()
    
    # Split data for training
    train_split = int(0.8 * n_periods)
    
    train_data = CrossAssetData(
        asset_prices={k: v[:train_split] for k, v in sample_data.asset_prices.items()},
        asset_returns={k: v[:train_split] for k, v in sample_data.asset_returns.items()},
        timestamps=sample_data.timestamps[:train_split],
        volume={k: v[:train_split] for k, v in sample_data.volume.items()},
        economic_indicators={k: v[:train_split] for k, v in sample_data.economic_indicators.items()}
    )
    
    test_data = CrossAssetData(
        asset_prices={k: v[train_split:] for k, v in sample_data.asset_prices.items()},
        asset_returns={k: v[train_split:] for k, v in sample_data.asset_returns.items()},
        timestamps=sample_data.timestamps[train_split:],
        volume={k: v[train_split:] for k, v in sample_data.volume.items()},
        economic_indicators={k: v[train_split:] for k, v in sample_data.economic_indicators.items()}
    )
    
    print("Training models...")
    models.fit(train_data)
    
    print("Making predictions...")
    result = models.predict(test_data)
    
    print(f"\nEnsemble Results:")
    print(f"Assets analyzed: {list(result.ensemble_predictions.keys())}")
    print(f"Model weights: {result.model_weights}")
    print(f"Model agreement: {result.model_agreement:.3f}")
    print(f"Prediction stability: {result.prediction_stability:.3f}")
    print(f"Current regime: {result.metadata.get('current_regime', 'unknown')}")
    
    print(f"\nTrading Signals:")
    for asset, signal in result.cross_asset_signals.items():
        signal_str = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
        pred = np.mean(result.ensemble_predictions[asset])
        print(f"  {asset}: {signal_str} (predicted return: {pred:.4f})")
    
    print(f"\nPortfolio Allocation:")
    for asset, weight in result.portfolio_allocation.items():
        print(f"  {asset}: {weight:.2%}")
    
    print("\nCrossAsset Advanced ML Models demonstration completed!")