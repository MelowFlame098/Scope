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

class TimeSeriesModels:
    """Time series models for cross-asset analysis"""
    
    @staticmethod
    def arima_forecast(asset_data: AssetData, order: Tuple[int, int, int] = (2, 1, 2)) -> CrossAssetIndicatorResult:
        """ARIMA model forecast"""
        try:
            prices = np.array(asset_data.historical_prices)
            if len(prices) < 50:
                raise ValueError("Insufficient data for ARIMA model")
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Simplified ARIMA implementation
            # In practice, this would use statsmodels or similar library
            p, d, q = order
            
            # AR component (simplified)
            if p > 0 and len(returns) > p:
                ar_coeffs = np.polyfit(range(p), returns[-p:], 1)[0]
            else:
                ar_coeffs = 0.0
            
            # MA component (simplified)
            if q > 0 and len(returns) > q:
                ma_coeffs = np.mean(returns[-q:])
            else:
                ma_coeffs = 0.0
            
            # Forecast next period
            forecast_return = ar_coeffs * returns[-1] + ma_coeffs
            forecast_price = asset_data.current_price * np.exp(forecast_return)
            
            # Calculate prediction interval
            residual_std = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
            confidence_interval = 1.96 * residual_std  # 95% CI
            
            # Generate signal
            expected_return = (forecast_price - asset_data.current_price) / asset_data.current_price
            
            if expected_return > 0.02:
                signal = "BUY"
            elif expected_return < -0.02:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            # Model diagnostics
            aic = len(returns) * np.log(residual_std**2) + 2 * (p + q + 1)  # Simplified AIC
            confidence = min(0.8, max(0.3, 1 - abs(expected_return) / 0.1))
            
            risk_level = "Low" if abs(expected_return) < 0.05 else "Medium" if abs(expected_return) < 0.1 else "High"
            
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.ARIMA,
                value=forecast_price,
                confidence=confidence,
                metadata={
                    "forecast_return": expected_return,
                    "confidence_interval": confidence_interval,
                    "ar_coefficient": ar_coeffs,
                    "ma_coefficient": ma_coeffs,
                    "aic": aic,
                    "order": order,
                    "residual_std": residual_std
                },
                timestamp=datetime.now(),
                interpretation=f"ARIMA({p},{d},{q}) forecast: {forecast_price:.2f} ({expected_return:.2%})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short-term",
                asset_symbols=[asset_data.symbol]
            )
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.ARIMA,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="ARIMA forecast failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset_data.symbol]
            )
    
    @staticmethod
    def garch_volatility(asset_data: AssetData, p: int = 1, q: int = 1) -> CrossAssetIndicatorResult:
        """GARCH volatility model"""
        try:
            prices = np.array(asset_data.historical_prices)
            if len(prices) < 100:
                raise ValueError("Insufficient data for GARCH model")
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            squared_returns = returns**2
            
            # Simplified GARCH(1,1) implementation
            # σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
            
            # Initialize parameters (would be estimated via MLE in practice)
            omega = 0.00001  # Long-term variance
            alpha = 0.1      # ARCH coefficient
            beta = 0.85      # GARCH coefficient
            
            # Ensure stationarity condition
            if alpha + beta >= 1:
                alpha = 0.05
                beta = 0.9
            
            # Calculate conditional variances
            conditional_variances = []
            sigma2 = np.var(returns)  # Initial variance
            
            for i in range(len(squared_returns)):
                sigma2 = omega + alpha * squared_returns[i] + beta * sigma2
                conditional_variances.append(sigma2)
            
            # Current volatility (annualized)
            current_volatility = np.sqrt(conditional_variances[-1] * 252)
            
            # Forecast next period volatility
            forecast_variance = omega + alpha * squared_returns[-1] + beta * conditional_variances[-1]
            forecast_volatility = np.sqrt(forecast_variance * 252)
            
            # Volatility regime classification
            long_term_vol = np.sqrt(omega / (1 - alpha - beta) * 252)
            vol_ratio = current_volatility / long_term_vol
            
            if vol_ratio > 1.5:
                regime = "High Volatility"
                signal = "SELL"  # High vol suggests caution
            elif vol_ratio < 0.7:
                regime = "Low Volatility"
                signal = "BUY"   # Low vol suggests opportunity
            else:
                regime = "Normal Volatility"
                signal = "HOLD"
            
            # Model fit quality
            residuals = returns / np.sqrt(conditional_variances)
            ljung_box_stat = np.sum(residuals**2)  # Simplified test statistic
            confidence = min(0.8, max(0.4, 1 - abs(vol_ratio - 1)))
            
            risk_level = "High" if regime == "High Volatility" else "Low" if regime == "Low Volatility" else "Medium"
            
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.GARCH,
                value=current_volatility,
                confidence=confidence,
                metadata={
                    "forecast_volatility": forecast_volatility,
                    "long_term_volatility": long_term_vol,
                    "volatility_ratio": vol_ratio,
                    "regime": regime,
                    "omega": omega,
                    "alpha": alpha,
                    "beta": beta,
                    "ljung_box_stat": ljung_box_stat,
                    "conditional_variances": conditional_variances[-10:]  # Last 10 values
                },
                timestamp=datetime.now(),
                interpretation=f"GARCH volatility: {current_volatility:.1%} ({regime})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short-term",
                asset_symbols=[asset_data.symbol]
            )
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.GARCH,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="GARCH volatility calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset_data.symbol]
            )