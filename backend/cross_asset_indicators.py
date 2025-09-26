from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from enum import Enum

# Import modularized classes
from time_series_models import TimeSeriesModels
from technical_indicators import TechnicalIndicators
from machine_learning_models import MachineLearningModels
from portfolio_models import PortfolioModels

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

# TimeSeriesModels class moved to time_series_models.py

# TechnicalIndicators class moved to technical_indicators.py

# MachineLearningModels class moved to machine_learning_models.py

# PortfolioModels class moved to portfolio_models.py

class CrossAssetIndicatorEngine:
    """Main engine for calculating cross-asset indicators"""
    
    def __init__(self):
        self.time_series = TimeSeriesModels()
        self.technical = TechnicalIndicators()
        self.ml_models = MachineLearningModels()
        self.portfolio = PortfolioModels()
    
    def calculate_all_indicators(self, asset_data: AssetData, 
                               additional_assets: Optional[List[AssetData]] = None) -> Dict[CrossAssetIndicatorType, CrossAssetIndicatorResult]:
        """Calculate all cross-asset indicators for a given asset"""
        results = {}
        
        # Time series models
        results[CrossAssetIndicatorType.ARIMA] = self.time_series.arima_forecast(asset_data)
        results[CrossAssetIndicatorType.GARCH] = self.time_series.garch_volatility(asset_data)
        
        # Technical indicators
        results[CrossAssetIndicatorType.RSI] = self.technical.rsi(asset_data)
        results[CrossAssetIndicatorType.MACD] = self.technical.macd(asset_data)
        results[CrossAssetIndicatorType.ICHIMOKU] = self.technical.ichimoku_cloud(asset_data)
        
        # Machine learning models
        results[CrossAssetIndicatorType.LSTM] = self.ml_models.lstm_prediction(asset_data)
        results[CrossAssetIndicatorType.XGBOOST] = self.ml_models.xgboost_prediction(asset_data)
        
        # Portfolio models (if additional assets provided)
        if additional_assets:
            all_assets = [asset_data] + additional_assets
            results[CrossAssetIndicatorType.MARKOWITZ_MPT] = self.portfolio.markowitz_optimization(all_assets)
        
        # Monte Carlo simulation
        results[CrossAssetIndicatorType.MONTE_CARLO] = self.portfolio.monte_carlo_simulation(asset_data)
        
        # Additional indicators
        results[CrossAssetIndicatorType.BOLLINGER_BANDS] = self._calculate_bollinger_bands(asset_data)
        results[CrossAssetIndicatorType.STOCHASTIC] = self._calculate_stochastic(asset_data)
        
        return results
    
    def _calculate_bollinger_bands(self, asset_data: AssetData, period: int = 20, std_dev: float = 2.0) -> CrossAssetIndicatorResult:
        """Calculate Bollinger Bands"""
        try:
            prices = np.array(asset_data.historical_prices)
            if len(prices) < period:
                raise ValueError(f"Insufficient data for Bollinger Bands (need {period} points)")
            
            # Calculate moving average and standard deviation
            sma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            
            # Calculate bands
            upper_band = sma + (std_dev * std)
            lower_band = sma - (std_dev * std)
            
            current_price = asset_data.current_price
            
            # Calculate position within bands
            band_width = upper_band - lower_band
            position = (current_price - lower_band) / band_width if band_width > 0 else 0.5
            
            # Generate signals
            if current_price > upper_band:
                signal = "SELL"  # Overbought
                interpretation = "Price above upper Bollinger Band (overbought)"
            elif current_price < lower_band:
                signal = "BUY"   # Oversold
                interpretation = "Price below lower Bollinger Band (oversold)"
            else:
                signal = "HOLD"
                interpretation = f"Price within Bollinger Bands ({position:.1%} position)"
            
            # Confidence based on how far outside bands
            if current_price > upper_band or current_price < lower_band:
                distance = min(abs(current_price - upper_band), abs(current_price - lower_band))
                confidence = min(0.8, max(0.5, distance / (band_width * 0.1) + 0.5))
            else:
                confidence = 0.4
            
            risk_level = "Low" if lower_band <= current_price <= upper_band else "Medium"
            
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.BOLLINGER_BANDS,
                value={
                    "sma": sma,
                    "upper_band": upper_band,
                    "lower_band": lower_band,
                    "band_width": band_width,
                    "position": position
                },
                confidence=confidence,
                metadata={
                    "period": period,
                    "std_dev_multiplier": std_dev,
                    "current_price": current_price,
                    "standard_deviation": std
                },
                timestamp=datetime.now(),
                interpretation=interpretation,
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short-term",
                asset_symbols=[asset_data.symbol]
            )
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.BOLLINGER_BANDS,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Bollinger Bands calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset_data.symbol]
            )
    
    def _calculate_stochastic(self, asset_data: AssetData, k_period: int = 14, d_period: int = 3) -> CrossAssetIndicatorResult:
        """Calculate Stochastic Oscillator"""
        try:
            prices = np.array(asset_data.historical_prices)
            if len(prices) < k_period:
                raise ValueError(f"Insufficient data for Stochastic (need {k_period} points)")
            
            # For simplicity, using closing prices as high/low
            # In practice, would use actual high/low data
            highs = prices
            lows = prices
            closes = prices
            
            # Calculate %K
            recent_high = np.max(highs[-k_period:])
            recent_low = np.min(lows[-k_period:])
            current_close = closes[-1]
            
            if recent_high == recent_low:
                k_percent = 50  # Avoid division by zero
            else:
                k_percent = ((current_close - recent_low) / (recent_high - recent_low)) * 100
            
            # Calculate %D (moving average of %K)
            # Simplified - would normally calculate %K for multiple periods
            d_percent = k_percent  # Simplified
            
            # Generate signals
            if k_percent > 80:
                signal = "SELL"  # Overbought
                interpretation = f"Stochastic overbought (%K: {k_percent:.1f})"
            elif k_percent < 20:
                signal = "BUY"   # Oversold
                interpretation = f"Stochastic oversold (%K: {k_percent:.1f})"
            else:
                signal = "HOLD"
                interpretation = f"Stochastic neutral (%K: {k_percent:.1f})"
            
            # Confidence based on extremity
            if k_percent > 90 or k_percent < 10:
                confidence = 0.8
            elif k_percent > 80 or k_percent < 20:
                confidence = 0.6
            else:
                confidence = 0.4
            
            risk_level = "Low" if 20 <= k_percent <= 80 else "Medium"
            
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.STOCHASTIC,
                value={
                    "k_percent": k_percent,
                    "d_percent": d_percent
                },
                confidence=confidence,
                metadata={
                    "k_period": k_period,
                    "d_period": d_period,
                    "recent_high": recent_high,
                    "recent_low": recent_low,
                    "current_close": current_close
                },
                timestamp=datetime.now(),
                interpretation=interpretation,
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short-term",
                asset_symbols=[asset_data.symbol]
            )
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.STOCHASTIC,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Stochastic calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset_data.symbol]
            )