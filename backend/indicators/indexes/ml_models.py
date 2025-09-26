"""Advanced Machine Learning Models for Indexes Analysis

This module implements enhanced LSTM and Transformer models with attention mechanisms,
ensemble methods, and advanced feature engineering for indexes prediction.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from datetime import datetime
from enum import Enum
import math

# Try to import ML libraries with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Using simplified ML models.")

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available.")

class ModelType(Enum):
    """Types of ML models available"""
    LSTM = "lstm"
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    XGBOOST = "xgboost"

@dataclass
class IndexesData:
    """Indexes information"""
    symbol: str
    name: str
    current_level: float
    historical_levels: list[float]
    dividend_yield: float
    pe_ratio: float
    pb_ratio: float
    market_cap: float
    volatility: float
    beta: float
    sector_weights: Dict[str, float]
    constituent_count: int
    volume: float

@dataclass
class MacroData:
    """Macroeconomic data"""
    gdp_growth: float
    inflation_rate: float
    interest_rates: float
    unemployment_rate: float
    industrial_production: float
    consumer_confidence: float
    oil_prices: float
    exchange_rates: float
    vix_index: float
    timestamp: datetime

@dataclass
class MLResult:
    """Result from ML model prediction"""
    predicted_value: float
    confidence: float
    feature_importance: Dict[str, float]
    model_metrics: Dict[str, float]
    prediction_interval: Tuple[float, float]
    risk_assessment: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    timestamp: datetime

class LSTMModel(nn.Module):
    """Enhanced LSTM model with attention mechanism"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, 
                 dropout: float = 0.2, attention: bool = True):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        if self.attention:
            self.attention_layer = nn.MultiheadAttention(hidden_size, num_heads=8)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        if self.attention:
            # Apply attention mechanism
            lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, hidden_size)
            attn_out, _ = self.attention_layer(lstm_out, lstm_out, lstm_out)
            lstm_out = attn_out.transpose(0, 1)  # Back to (batch, seq_len, hidden_size)
        
        # Use the last output
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 8, 
                 num_layers: int = 6, dropout: float = 0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, 
                                                   dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.output_layer = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = self.input_projection(x) * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # (seq_len, batch_size, d_model)
        x = self.pos_encoder(x)
        
        output = self.transformer_encoder(x)
        output = output[-1, :, :]  # Take the last time step
        output = self.dropout(output)
        output = self.output_layer(output)
        return output

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class AdvancedMLModels:
    """Advanced ML models for indexes prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_metrics = {}
        
        # Initialize models if libraries are available
        if SKLEARN_AVAILABLE:
            self._initialize_sklearn_models()
        if XGBOOST_AVAILABLE:
            self._initialize_xgboost_models()
        if TORCH_AVAILABLE:
            self._initialize_torch_models()
    
    def _initialize_sklearn_models(self):
        """Initialize scikit-learn models"""
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42
        )
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.models['neural_network'] = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32), max_iter=1000, random_state=42
        )
        
        # Initialize scalers
        for model_name in ['random_forest', 'gradient_boosting', 'neural_network']:
            self.scalers[model_name] = StandardScaler()
    
    def _initialize_xgboost_models(self):
        """Initialize XGBoost models"""
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        self.scalers['xgboost'] = StandardScaler()
    
    def _initialize_torch_models(self):
        """Initialize PyTorch models"""
        # These will be initialized when training with specific input dimensions
        self.models['lstm'] = None
        self.models['transformer'] = None
    
    def prepare_features(self, indexes_data: IndexesData, macro_data: MacroData) -> np.ndarray:
        """Prepare features for ML models"""
        features = []
        
        # Indexes features
        features.extend([
            indexes_data.current_level,
            indexes_data.dividend_yield,
            indexes_data.pe_ratio,
            indexes_data.pb_ratio,
            math.log(indexes_data.market_cap) if indexes_data.market_cap > 0 else 0,
            indexes_data.volatility,
            indexes_data.beta,
            indexes_data.constituent_count,
            math.log(indexes_data.volume) if indexes_data.volume > 0 else 0
        ])
        
        # Technical indicators from historical levels
        if len(indexes_data.historical_levels) >= 20:
            levels = np.array(indexes_data.historical_levels[-20:])
            features.extend([
                np.mean(levels),  # Moving average
                np.std(levels),   # Volatility
                (levels[-1] - levels[0]) / levels[0],  # Return
                np.max(levels) / np.min(levels) - 1,   # Range ratio
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Sector weights (top 5 sectors)
        sector_values = list(indexes_data.sector_weights.values())
        sector_values.sort(reverse=True)
        features.extend(sector_values[:5] + [0] * (5 - len(sector_values[:5])))
        
        # Macroeconomic features
        features.extend([
            macro_data.gdp_growth,
            macro_data.inflation_rate,
            macro_data.interest_rates,
            macro_data.unemployment_rate,
            macro_data.industrial_production,
            macro_data.consumer_confidence,
            macro_data.oil_prices,
            macro_data.exchange_rates,
            macro_data.vix_index
        ])
        
        return np.array(features).reshape(1, -1)
    
    def train_model(self, model_type: ModelType, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train a specific model"""
        model_name = model_type.value
        
        if model_name not in self.models:
            return {'status': 'error', 'message': f'Model {model_name} not available'}
        
        try:
            if model_name in ['lstm', 'transformer'] and TORCH_AVAILABLE:
                return self._train_torch_model(model_name, X, y)
            else:
                return self._train_sklearn_model(model_name, X, y)
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def _train_sklearn_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train scikit-learn or XGBoost model"""
        # Scale features
        X_scaled = self.scalers[model_name].fit_transform(X)
        
        # Train model
        self.models[model_name].fit(X_scaled, y)
        
        # Calculate metrics
        y_pred = self.models[model_name].predict(X_scaled)
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'r2': r2_score(y, y_pred),
            'mae': mean_absolute_error(y, y_pred)
        }
        
        # Feature importance
        if hasattr(self.models[model_name], 'feature_importances_'):
            self.feature_importance[model_name] = self.models[model_name].feature_importances_
        elif hasattr(self.models[model_name], 'coef_'):
            self.feature_importance[model_name] = np.abs(self.models[model_name].coef_)
        
        self.model_metrics[model_name] = metrics
        
        return {'status': 'success', 'metrics': metrics}
    
    def _train_torch_model(self, model_name: str, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train PyTorch model"""
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).unsqueeze(1)
        
        # Initialize model
        input_size = X.shape[-1]
        if model_name == 'lstm':
            self.models[model_name] = LSTMModel(input_size)
        elif model_name == 'transformer':
            self.models[model_name] = TransformerModel(input_size)
        
        model = self.models[model_name]
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        
        # Calculate metrics
        model.eval()
        with torch.no_grad():
            y_pred = model(X_tensor).numpy()
            metrics = {
                'mse': mean_squared_error(y, y_pred.flatten()),
                'r2': r2_score(y, y_pred.flatten()),
                'mae': mean_absolute_error(y, y_pred.flatten())
            }
        
        self.model_metrics[model_name] = metrics
        
        return {'status': 'success', 'metrics': metrics}
    
    def predict(self, model_type: ModelType, indexes_data: IndexesData, 
                macro_data: MacroData) -> MLResult:
        """Make prediction using specified model"""
        model_name = model_type.value
        
        if model_name not in self.models or self.models[model_name] is None:
            return MLResult(
                predicted_value=0.0,
                confidence=0.0,
                feature_importance={},
                model_metrics={},
                prediction_interval=(0.0, 0.0),
                risk_assessment="HIGH",
                signal="HOLD",
                timestamp=datetime.now()
            )
        
        try:
            # Prepare features
            features = self.prepare_features(indexes_data, macro_data)
            
            # Make prediction
            if model_name in ['lstm', 'transformer'] and TORCH_AVAILABLE:
                prediction = self._predict_torch_model(model_name, features)
            else:
                prediction = self._predict_sklearn_model(model_name, features)
            
            # Calculate confidence based on model metrics
            confidence = self._calculate_confidence(model_name, prediction)
            
            # Generate signal
            signal = self._generate_signal(prediction, indexes_data.current_level)
            
            # Risk assessment
            risk_assessment = self._assess_risk(prediction, indexes_data, macro_data)
            
            # Prediction interval
            prediction_interval = self._calculate_prediction_interval(model_name, prediction)
            
            return MLResult(
                predicted_value=float(prediction),
                confidence=confidence,
                feature_importance=self.feature_importance.get(model_name, {}),
                model_metrics=self.model_metrics.get(model_name, {}),
                prediction_interval=prediction_interval,
                risk_assessment=risk_assessment,
                signal=signal,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return MLResult(
                predicted_value=0.0,
                confidence=0.0,
                feature_importance={},
                model_metrics={},
                prediction_interval=(0.0, 0.0),
                risk_assessment="HIGH",
                signal="HOLD",
                timestamp=datetime.now()
            )
    
    def _predict_sklearn_model(self, model_name: str, features: np.ndarray) -> float:
        """Make prediction using scikit-learn model"""
        # Check if scaler is fitted, if not use simple normalization
        try:
            features_scaled = self.scalers[model_name].transform(features)
        except:
            # If scaler is not fitted, use the features as-is or apply simple normalization
            features_scaled = features
        
        prediction = self.models[model_name].predict(features_scaled)
        return prediction[0]
    
    def _predict_torch_model(self, model_name: str, features: np.ndarray) -> float:
        """Make prediction using PyTorch model"""
        model = self.models[model_name]
        model.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            if len(features_tensor.shape) == 2:
                features_tensor = features_tensor.unsqueeze(0)  # Add batch dimension
            prediction = model(features_tensor)
            return prediction.item()
    
    def _calculate_confidence(self, model_name: str, prediction: float) -> float:
        """Calculate prediction confidence based on model metrics"""
        metrics = self.model_metrics.get(model_name, {})
        r2 = metrics.get('r2', 0.0)
        
        # Convert R² to confidence (0-1 scale)
        confidence = max(0.0, min(1.0, r2))
        return confidence
    
    def _generate_signal(self, prediction: float, current_level: float) -> str:
        """Generate trading signal based on prediction"""
        change_pct = (prediction - current_level) / current_level
        
        if change_pct > 0.02:  # 2% threshold
            return "BUY"
        elif change_pct < -0.02:
            return "SELL"
        else:
            return "HOLD"
    
    def _assess_risk(self, prediction: float, indexes_data: IndexesData, 
                    macro_data: MacroData) -> str:
        """Assess risk level of the prediction"""
        risk_factors = 0
        
        # Volatility risk
        if indexes_data.volatility > 0.3:
            risk_factors += 1
        
        # VIX risk
        if macro_data.vix_index > 25:
            risk_factors += 1
        
        # Interest rate risk
        if macro_data.interest_rates > 5.0:
            risk_factors += 1
        
        # Inflation risk
        if macro_data.inflation_rate > 4.0:
            risk_factors += 1
        
        if risk_factors >= 3:
            return "HIGH"
        elif risk_factors >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _calculate_prediction_interval(self, model_name: str, prediction: float) -> Tuple[float, float]:
        """Calculate prediction interval"""
        metrics = self.model_metrics.get(model_name, {})
        mse = metrics.get('mse', 1.0)
        std_error = math.sqrt(mse)
        
        # 95% confidence interval (approximately 2 standard errors)
        lower_bound = prediction - 2 * std_error
        upper_bound = prediction + 2 * std_error
        
        return (lower_bound, upper_bound)
    
    def ensemble_prediction(self, indexes_data: IndexesData, macro_data: MacroData) -> MLResult:
        """Make ensemble prediction using multiple models"""
        predictions = []
        confidences = []
        feature_importances = {}
        
        # Get predictions from all available models
        for model_name in self.models:
            if self.models[model_name] is not None:
                try:
                    model_type = ModelType(model_name)
                    result = self.predict(model_type, indexes_data, macro_data)
                    predictions.append(result.predicted_value)
                    confidences.append(result.confidence)
                    
                    # Aggregate feature importance
                    for feature, importance in result.feature_importance.items():
                        if feature not in feature_importances:
                            feature_importances[feature] = []
                        feature_importances[feature].append(importance)
                except:
                    continue
        
        if not predictions:
            return MLResult(
                predicted_value=0.0,
                confidence=0.0,
                feature_importance={},
                model_metrics={},
                prediction_interval=(0.0, 0.0),
                risk_assessment="HIGH",
                signal="HOLD",
                timestamp=datetime.now()
            )
        
        # Weighted average prediction
        weights = np.array(confidences)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
            ensemble_prediction = np.average(predictions, weights=weights)
        else:
            ensemble_prediction = np.mean(predictions)
        
        # Average confidence
        ensemble_confidence = np.mean(confidences)
        
        # Average feature importance
        avg_feature_importance = {}
        for feature, importances in feature_importances.items():
            avg_feature_importance[feature] = np.mean(importances)
        
        # Generate ensemble signal
        signal = self._generate_signal(ensemble_prediction, indexes_data.current_level)
        
        # Risk assessment
        risk_assessment = self._assess_risk(ensemble_prediction, indexes_data, macro_data)
        
        # Prediction interval based on prediction variance
        prediction_std = np.std(predictions) if len(predictions) > 1 else 0.1
        prediction_interval = (
            ensemble_prediction - 2 * prediction_std,
            ensemble_prediction + 2 * prediction_std
        )
        
        return MLResult(
            predicted_value=float(ensemble_prediction),
            confidence=ensemble_confidence,
            feature_importance=avg_feature_importance,
            model_metrics={'ensemble_models': len(predictions)},
            prediction_interval=prediction_interval,
            risk_assessment=risk_assessment,
            signal=signal,
            timestamp=datetime.now()
        )

# Example usage
if __name__ == "__main__":
    # Create sample data
    indexes_data = IndexesData(
        symbol="SPY",
        name="SPDR S&P 500 ETF",
        current_level=450.0,
        historical_levels=[440.0, 445.0, 448.0, 450.0],
        dividend_yield=0.015,
        pe_ratio=22.5,
        pb_ratio=3.2,
        market_cap=400000000000,
        volatility=0.18,
        beta=1.0,
        sector_weights={"Technology": 0.28, "Healthcare": 0.13, "Financials": 0.11},
        constituent_count=500,
        volume=50000000
    )
    
    macro_data = MacroData(
        gdp_growth=2.1,
        inflation_rate=3.2,
        interest_rates=5.25,
        unemployment_rate=3.7,
        industrial_production=1.8,
        consumer_confidence=102.5,
        oil_prices=85.0,
        exchange_rates=1.08,
        vix_index=18.5,
        timestamp=datetime.now()
    )
    
    # Initialize ML models
    ml_models = AdvancedMLModels()
    
    # Make ensemble prediction
    result = ml_models.ensemble_prediction(indexes_data, macro_data)
    
    print(f"Ensemble Prediction: {result.predicted_value:.2f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Signal: {result.signal}")
    print(f"Risk Assessment: {result.risk_assessment}")