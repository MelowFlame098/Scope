from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from datetime import datetime
from enum import Enum

class CrossAssetIndicatorType(Enum):
    """Cross-asset indicator types"""
    ARIMA = "arima"  # ARIMA Models
    SARIMA = "sarima"  # Seasonal ARIMA
    GARCH = "garch"  # GARCH Models
    EGARCH = "egarch"  # Exponential GARCH
    TGARCH = "tgarch"  # Threshold GARCH
    LSTM = "lstm"  # Long Short-Term Memory
    GRU = "gru"  # Gated Recurrent Unit
    TRANSFORMER = "transformer"  # Transformer Models
    XGBOOST = "xgboost"  # XGBoost
    LIGHTGBM = "lightgbm"  # LightGBM
    SVM = "svm"  # Support Vector Machine
    RSI = "rsi"  # Relative Strength Index
    MACD = "macd"  # Moving Average Convergence Divergence
    ICHIMOKU = "ichimoku"  # Ichimoku Cloud
    BOLLINGER_BANDS = "bollinger_bands"  # Bollinger Bands
    STOCHASTIC = "stochastic"  # Stochastic Oscillator
    PPO = "ppo"  # Proximal Policy Optimization (RL)
    SAC = "sac"  # Soft Actor-Critic (RL)
    DDPG = "ddpg"  # Deep Deterministic Policy Gradient (RL)
    MARKOWITZ_MPT = "markowitz_mpt"  # Modern Portfolio Theory
    MONTE_CARLO = "monte_carlo"  # Monte Carlo Simulation
    FINBERT = "finbert"  # Financial BERT
    CRYPTOBERT = "cryptobert"  # Crypto BERT
    FOREXBERT = "forexbert"  # Forex BERT
    HMM = "hmm"  # Hidden Markov Model
    BAYESIAN_CHANGE_POINT = "bayesian_change_point"  # Bayesian Change Point Detection
    CORRELATION_ANALYSIS = "correlation_analysis"  # Cross-Asset Correlation
    COINTEGRATION = "cointegration"  # Cointegration Analysis
    PAIRS_TRADING = "pairs_trading"  # Pairs Trading Strategy
    REGIME_SWITCHING = "regime_switching"  # Regime Switching Models

@dataclass
class AssetData:
    """Generic asset data structure"""
    symbol: str
    asset_type: str  # "crypto", "stock", "forex", "futures", "index"
    current_price: float
    historical_prices: List[float]
    volume: float
    market_cap: Optional[float] = None
    volatility: float = 0.0
    beta: Optional[float] = None
    correlation_matrix: Optional[Dict[str, float]] = None
    fundamental_data: Optional[Dict[str, Any]] = None

@dataclass
class CrossAssetIndicatorResult:
    """Result of cross-asset indicator calculation"""
    indicator_type: CrossAssetIndicatorType
    value: Union[float, Dict[str, float], List[float]]
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str
    signal: str  # "BUY", "SELL", "HOLD"
    time_horizon: str
    asset_symbols: List[str]

class MachineLearningModels:
    """Machine learning models for cross-asset analysis"""
    
    @staticmethod
    def lstm_prediction(asset_data: AssetData, sequence_length: int = 60, forecast_horizon: int = 1) -> CrossAssetIndicatorResult:
        """LSTM-based price prediction"""
        try:
            prices = np.array(asset_data.historical_prices)
            if len(prices) < sequence_length + 10:
                raise ValueError(f"Insufficient data for LSTM prediction (need {sequence_length + 10} points)")
            
            # Normalize prices
            price_mean = np.mean(prices)
            price_std = np.std(prices)
            normalized_prices = (prices - price_mean) / price_std
            
            # Create sequences for training (simplified approach)
            sequences = []
            targets = []
            
            for i in range(sequence_length, len(normalized_prices)):
                sequences.append(normalized_prices[i-sequence_length:i])
                targets.append(normalized_prices[i])
            
            sequences = np.array(sequences)
            targets = np.array(targets)
            
            # Simplified LSTM simulation (using linear regression as proxy)
            # In practice, this would use TensorFlow/PyTorch
            
            # Feature engineering: extract statistical features from sequences
            features = []
            for seq in sequences:
                feature_vector = [
                    np.mean(seq),           # Mean
                    np.std(seq),            # Standard deviation
                    seq[-1] - seq[0],       # Total change
                    np.max(seq) - np.min(seq),  # Range
                    np.mean(np.diff(seq)),  # Average change
                    seq[-1]                 # Last value
                ]
                features.append(feature_vector)
            
            features = np.array(features)
            
            # Simple linear model (proxy for LSTM)
            if len(features) > 10:
                # Use last 80% for training, 20% for validation
                train_size = int(0.8 * len(features))
                
                X_train = features[:train_size]
                y_train = targets[:train_size]
                X_val = features[train_size:]
                y_val = targets[train_size:]
                
                # Fit linear model (simplified)
                try:
                    # Pseudo-inverse for linear regression
                    weights = np.linalg.pinv(X_train.T @ X_train) @ X_train.T @ y_train
                except:
                    # Fallback to simple average
                    weights = np.ones(features.shape[1]) / features.shape[1]
                
                # Validation error
                val_predictions = X_val @ weights
                mse = np.mean((val_predictions - y_val) ** 2)
                
                # Make prediction for next period
                last_sequence = normalized_prices[-sequence_length:]
                last_features = np.array([
                    np.mean(last_sequence),
                    np.std(last_sequence),
                    last_sequence[-1] - last_sequence[0],
                    np.max(last_sequence) - np.min(last_sequence),
                    np.mean(np.diff(last_sequence)),
                    last_sequence[-1]
                ])
                
                predicted_normalized = last_features @ weights
                predicted_price = predicted_normalized * price_std + price_mean
                
                # Calculate confidence based on validation performance
                confidence = max(0.3, min(0.8, 1 - mse))
                
                # Generate signal
                expected_return = (predicted_price - asset_data.current_price) / asset_data.current_price
                
                if expected_return > 0.02:
                    signal = "BUY"
                elif expected_return < -0.02:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                risk_level = "Low" if abs(expected_return) < 0.05 else "Medium" if abs(expected_return) < 0.1 else "High"
                
                return CrossAssetIndicatorResult(
                    indicator_type=CrossAssetIndicatorType.LSTM,
                    value=predicted_price,
                    confidence=confidence,
                    metadata={
                        "sequence_length": sequence_length,
                        "forecast_horizon": forecast_horizon,
                        "expected_return": expected_return,
                        "validation_mse": mse,
                        "training_samples": len(X_train),
                        "price_normalization": {"mean": price_mean, "std": price_std}
                    },
                    timestamp=datetime.now(),
                    interpretation=f"LSTM prediction: {predicted_price:.2f} ({expected_return:.2%})",
                    risk_level=risk_level,
                    signal=signal,
                    time_horizon="Short-term",
                    asset_symbols=[asset_data.symbol]
                )
            else:
                raise ValueError("Insufficient training data")
                
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.LSTM,
                value=asset_data.current_price,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="LSTM prediction failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset_data.symbol]
            )
    
    @staticmethod
    def xgboost_prediction(asset_data: AssetData, n_estimators: int = 100, max_depth: int = 6) -> CrossAssetIndicatorResult:
        """XGBoost-based prediction"""
        try:
            prices = np.array(asset_data.historical_prices)
            if len(prices) < 50:
                raise ValueError("Insufficient data for XGBoost prediction")
            
            # Feature engineering
            def create_features(prices, window=20):
                features = []
                targets = []
                
                for i in range(window, len(prices) - 1):
                    price_window = prices[i-window:i]
                    
                    # Technical features
                    sma_5 = np.mean(price_window[-5:])
                    sma_10 = np.mean(price_window[-10:])
                    sma_20 = np.mean(price_window)
                    
                    # Volatility features
                    volatility = np.std(price_window)
                    
                    # Momentum features
                    momentum_5 = (price_window[-1] - price_window[-5]) / price_window[-5] if len(price_window) >= 5 else 0
                    momentum_10 = (price_window[-1] - price_window[-10]) / price_window[-10] if len(price_window) >= 10 else 0
                    
                    # RSI-like feature
                    changes = np.diff(price_window)
                    gains = np.where(changes > 0, changes, 0)
                    losses = np.where(changes < 0, -changes, 0)
                    avg_gain = np.mean(gains) if len(gains) > 0 else 0
                    avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
                    rsi_like = 100 - (100 / (1 + avg_gain / avg_loss))
                    
                    # Price position features
                    price_position = (price_window[-1] - np.min(price_window)) / (np.max(price_window) - np.min(price_window)) if np.max(price_window) != np.min(price_window) else 0.5
                    
                    feature_vector = [
                        sma_5 / price_window[-1],      # Normalized SMA 5
                        sma_10 / price_window[-1],     # Normalized SMA 10
                        sma_20 / price_window[-1],     # Normalized SMA 20
                        volatility / price_window[-1], # Normalized volatility
                        momentum_5,                     # 5-day momentum
                        momentum_10,                    # 10-day momentum
                        rsi_like / 100,                # Normalized RSI-like
                        price_position,                 # Price position in range
                        len([x for x in changes[-5:] if x > 0]) / 5,  # Positive days ratio
                        np.mean(changes[-5:]) / price_window[-1]       # Average change ratio
                    ]
                    
                    features.append(feature_vector)
                    
                    # Target: next day return
                    next_return = (prices[i+1] - prices[i]) / prices[i]
                    targets.append(next_return)
                
                return np.array(features), np.array(targets)
            
            # Create training data
            X, y = create_features(prices)
            
            if len(X) < 20:
                raise ValueError("Insufficient training samples")
            
            # Split data
            train_size = int(0.8 * len(X))
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]
            
            # Simplified XGBoost simulation (using ensemble of simple models)
            # In practice, this would use the actual XGBoost library
            
            models = []
            predictions_train = np.zeros(len(y_train))
            
            # Create ensemble of simple models
            for i in range(min(n_estimators, 10)):  # Limit for simulation
                # Random feature subset
                feature_indices = np.random.choice(X_train.shape[1], size=max(1, X_train.shape[1]//2), replace=False)
                X_subset = X_train[:, feature_indices]
                
                # Simple linear model
                try:
                    weights = np.linalg.pinv(X_subset.T @ X_subset) @ X_subset.T @ y_train
                    models.append((feature_indices, weights))
                    
                    # Add predictions
                    pred = X_subset @ weights
                    predictions_train += pred / min(n_estimators, 10)
                except:
                    continue
            
            # Validation
            predictions_val = np.zeros(len(y_val))
            for feature_indices, weights in models:
                X_val_subset = X_val[:, feature_indices]
                pred = X_val_subset @ weights
                predictions_val += pred / len(models)
            
            # Calculate validation metrics
            mse = np.mean((predictions_val - y_val) ** 2)
            mae = np.mean(np.abs(predictions_val - y_val))
            
            # Make prediction for current price
            current_features, _ = create_features(prices[-21:])  # Use last 21 prices to create 1 feature vector
            if len(current_features) > 0:
                current_X = current_features[-1:]
                
                prediction = 0
                for feature_indices, weights in models:
                    X_current_subset = current_X[:, feature_indices]
                    pred = X_current_subset @ weights
                    prediction += pred[0] / len(models)
                
                predicted_return = prediction
                predicted_price = asset_data.current_price * (1 + predicted_return)
                
                # Calculate confidence
                confidence = max(0.3, min(0.8, 1 - mse * 10))  # Scale MSE for confidence
                
                # Generate signal
                if predicted_return > 0.02:
                    signal = "BUY"
                elif predicted_return < -0.02:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                risk_level = "Low" if abs(predicted_return) < 0.05 else "Medium" if abs(predicted_return) < 0.1 else "High"
                
                return CrossAssetIndicatorResult(
                    indicator_type=CrossAssetIndicatorType.XGBOOST,
                    value=predicted_price,
                    confidence=confidence,
                    metadata={
                        "predicted_return": predicted_return,
                        "n_estimators": len(models),
                        "max_depth": max_depth,
                        "validation_mse": mse,
                        "validation_mae": mae,
                        "training_samples": len(X_train),
                        "feature_count": X.shape[1]
                    },
                    timestamp=datetime.now(),
                    interpretation=f"XGBoost prediction: {predicted_price:.2f} ({predicted_return:.2%})",
                    risk_level=risk_level,
                    signal=signal,
                    time_horizon="Short-term",
                    asset_symbols=[asset_data.symbol]
                )
            else:
                raise ValueError("Could not create features for prediction")
                
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.XGBOOST,
                value=asset_data.current_price,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="XGBoost prediction failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset_data.symbol]
            )