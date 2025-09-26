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

class TechnicalIndicators:
    """Technical indicators for cross-asset analysis"""
    
    @staticmethod
    def rsi(asset_data: AssetData, period: int = 14) -> CrossAssetIndicatorResult:
        """Relative Strength Index"""
        try:
            prices = np.array(asset_data.historical_prices)
            if len(prices) < period + 1:
                raise ValueError(f"Insufficient data for RSI calculation (need {period + 1} points)")
            
            # Calculate price changes
            deltas = np.diff(prices)
            
            # Separate gains and losses
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calculate average gains and losses
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            # Calculate RSI
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Generate signals
            if rsi > 70:
                signal = "SELL"
                interpretation = "Overbought condition"
                risk_level = "Medium"
            elif rsi < 30:
                signal = "BUY"
                interpretation = "Oversold condition"
                risk_level = "Medium"
            else:
                signal = "HOLD"
                interpretation = "Neutral momentum"
                risk_level = "Low"
            
            # Calculate confidence based on how extreme the RSI is
            if rsi > 80 or rsi < 20:
                confidence = 0.8
            elif rsi > 70 or rsi < 30:
                confidence = 0.6
            else:
                confidence = 0.4
            
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.RSI,
                value=rsi,
                confidence=confidence,
                metadata={
                    "period": period,
                    "avg_gain": avg_gain,
                    "avg_loss": avg_loss,
                    "recent_gains": gains[-5:].tolist(),
                    "recent_losses": losses[-5:].tolist()
                },
                timestamp=datetime.now(),
                interpretation=f"RSI({period}): {rsi:.1f} - {interpretation}",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short-term",
                asset_symbols=[asset_data.symbol]
            )
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.RSI,
                value=50.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="RSI calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset_data.symbol]
            )
    
    @staticmethod
    def macd(asset_data: AssetData, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> CrossAssetIndicatorResult:
        """Moving Average Convergence Divergence"""
        try:
            prices = np.array(asset_data.historical_prices)
            if len(prices) < slow_period + signal_period:
                raise ValueError(f"Insufficient data for MACD calculation")
            
            # Calculate exponential moving averages
            def ema(data, period):
                alpha = 2 / (period + 1)
                ema_values = [data[0]]
                for price in data[1:]:
                    ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
                return np.array(ema_values)
            
            fast_ema = ema(prices, fast_period)
            slow_ema = ema(prices, slow_period)
            
            # Calculate MACD line
            macd_line = fast_ema - slow_ema
            
            # Calculate signal line (EMA of MACD)
            signal_line = ema(macd_line, signal_period)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            # Current values
            current_macd = macd_line[-1]
            current_signal = signal_line[-1]
            current_histogram = histogram[-1]
            
            # Generate signals
            if current_macd > current_signal and histogram[-2] < histogram[-1]:
                signal = "BUY"
                interpretation = "Bullish crossover"
                risk_level = "Medium"
            elif current_macd < current_signal and histogram[-2] > histogram[-1]:
                signal = "SELL"
                interpretation = "Bearish crossover"
                risk_level = "Medium"
            else:
                signal = "HOLD"
                interpretation = "No clear signal"
                risk_level = "Low"
            
            # Calculate confidence based on histogram momentum
            histogram_momentum = abs(current_histogram - histogram[-2]) if len(histogram) > 1 else 0
            confidence = min(0.8, max(0.3, histogram_momentum * 100))
            
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.MACD,
                value={
                    "macd": current_macd,
                    "signal": current_signal,
                    "histogram": current_histogram
                },
                confidence=confidence,
                metadata={
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "signal_period": signal_period,
                    "histogram_momentum": histogram_momentum,
                    "recent_histogram": histogram[-5:].tolist()
                },
                timestamp=datetime.now(),
                interpretation=f"MACD: {current_macd:.4f}, Signal: {current_signal:.4f} - {interpretation}",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Medium-term",
                asset_symbols=[asset_data.symbol]
            )
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.MACD,
                value={"macd": 0.0, "signal": 0.0, "histogram": 0.0},
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="MACD calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset_data.symbol]
            )
    
    @staticmethod
    def ichimoku_cloud(asset_data: AssetData) -> CrossAssetIndicatorResult:
        """Ichimoku Cloud analysis"""
        try:
            prices = np.array(asset_data.historical_prices)
            if len(prices) < 52:
                raise ValueError("Insufficient data for Ichimoku Cloud calculation")
            
            # Calculate high and low arrays (simplified - using prices as both)
            highs = prices
            lows = prices
            
            # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
            tenkan_sen = (np.max(highs[-9:]) + np.min(lows[-9:])) / 2
            
            # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
            kijun_sen = (np.max(highs[-26:]) + np.min(lows[-26:])) / 2
            
            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen) / 2
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
            
            # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
            senkou_span_b = (np.max(highs[-52:]) + np.min(lows[-52:])) / 2
            
            # Chikou Span (Lagging Span): Current closing price plotted 26 periods back
            chikou_span = prices[-1]
            
            current_price = asset_data.current_price
            
            # Cloud analysis
            cloud_top = max(senkou_span_a, senkou_span_b)
            cloud_bottom = min(senkou_span_a, senkou_span_b)
            
            # Generate signals
            signals = []
            
            # Price vs Cloud
            if current_price > cloud_top:
                signals.append("Above cloud (Bullish)")
                price_signal = "BUY"
            elif current_price < cloud_bottom:
                signals.append("Below cloud (Bearish)")
                price_signal = "SELL"
            else:
                signals.append("Inside cloud (Neutral)")
                price_signal = "HOLD"
            
            # Tenkan-sen vs Kijun-sen
            if tenkan_sen > kijun_sen:
                signals.append("TK Cross Bullish")
            elif tenkan_sen < kijun_sen:
                signals.append("TK Cross Bearish")
            
            # Cloud color (Senkou Span A vs B)
            if senkou_span_a > senkou_span_b:
                cloud_color = "Green (Bullish)"
            else:
                cloud_color = "Red (Bearish)"
            
            # Overall signal
            bullish_signals = sum(1 for s in signals if "Bullish" in s)
            bearish_signals = sum(1 for s in signals if "Bearish" in s)
            
            if bullish_signals > bearish_signals:
                overall_signal = "BUY"
                risk_level = "Medium"
            elif bearish_signals > bullish_signals:
                overall_signal = "SELL"
                risk_level = "Medium"
            else:
                overall_signal = "HOLD"
                risk_level = "Low"
            
            # Calculate confidence
            signal_strength = abs(bullish_signals - bearish_signals) / len(signals)
            confidence = min(0.8, max(0.4, signal_strength))
            
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.ICHIMOKU,
                value={
                    "tenkan_sen": tenkan_sen,
                    "kijun_sen": kijun_sen,
                    "senkou_span_a": senkou_span_a,
                    "senkou_span_b": senkou_span_b,
                    "chikou_span": chikou_span,
                    "cloud_top": cloud_top,
                    "cloud_bottom": cloud_bottom
                },
                confidence=confidence,
                metadata={
                    "signals": signals,
                    "cloud_color": cloud_color,
                    "price_vs_cloud": price_signal,
                    "signal_strength": signal_strength
                },
                timestamp=datetime.now(),
                interpretation=f"Ichimoku: {', '.join(signals)} | Cloud: {cloud_color}",
                risk_level=risk_level,
                signal=overall_signal,
                time_horizon="Medium-term",
                asset_symbols=[asset_data.symbol]
            )
        except Exception as e:
            return CrossAssetIndicatorResult(
                indicator_type=CrossAssetIndicatorType.ICHIMOKU,
                value={"tenkan_sen": 0.0, "kijun_sen": 0.0, "senkou_span_a": 0.0, "senkou_span_b": 0.0, "chikou_span": 0.0},
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Ichimoku Cloud calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A",
                asset_symbols=[asset_data.symbol]
            )
    
    @staticmethod
    def _calculate_bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands"""
        # Simple moving average
        sma = np.convolve(prices, np.ones(period)/period, mode='valid')
        
        # Standard deviation
        std = np.array([np.std(prices[i:i+period]) for i in range(len(prices) - period + 1)])
        
        # Bollinger Bands
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    @staticmethod
    def _calculate_stochastic(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Stochastic Oscillator"""
        # %K calculation
        k_values = []
        for i in range(k_period - 1, len(closes)):
            highest_high = np.max(highs[i - k_period + 1:i + 1])
            lowest_low = np.min(lows[i - k_period + 1:i + 1])
            
            if highest_high == lowest_low:
                k_value = 50  # Avoid division by zero
            else:
                k_value = ((closes[i] - lowest_low) / (highest_high - lowest_low)) * 100
            
            k_values.append(k_value)
        
        k_values = np.array(k_values)
        
        # %D calculation (SMA of %K)
        d_values = np.convolve(k_values, np.ones(d_period)/d_period, mode='valid')
        
        return k_values, d_values