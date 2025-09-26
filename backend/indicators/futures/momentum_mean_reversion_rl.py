from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import gym
    from stable_baselines3 import PPO, SAC, DDPG
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    RL_LIBS_AVAILABLE = True
except ImportError:
    RL_LIBS_AVAILABLE = False
    print("Reinforcement Learning libraries not available. Using simplified implementations.")

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using custom technical indicators.")

@dataclass
class FuturesData:
    """Structure for futures market data"""
    prices: List[float]
    returns: List[float]
    volume: List[float]
    open_interest: List[float]
    timestamps: List[datetime]
    high: List[float]
    low: List[float]
    open: List[float]
    close: List[float]
    contract_symbol: str
    underlying_asset: str

@dataclass
class MomentumResult:
    """Results from momentum analysis"""
    momentum_scores: List[float]
    momentum_signals: List[str]
    momentum_strength: List[float]
    trend_direction: List[str]
    momentum_divergence: List[bool]
    rsi_values: List[float]
    macd_values: List[float]
    macd_signal: List[float]
    stochastic_k: List[float]
    stochastic_d: List[float]
    williams_r: List[float]
    
@dataclass
class MeanReversionResult:
    """Results from mean reversion analysis"""
    mean_reversion_scores: List[float]
    reversion_signals: List[str]
    bollinger_upper: List[float]
    bollinger_lower: List[float]
    bollinger_middle: List[float]
    z_scores: List[float]
    adf_pvalue: float
    half_life: float
    reversion_probability: List[float]
    oversold_levels: List[bool]
    overbought_levels: List[bool]
    
@dataclass
class RLAgentResult:
    """Results from reinforcement learning agents"""
    agent_type: str  # 'PPO', 'SAC', 'DDPG'
    actions: List[int]  # 0: Hold, 1: Buy, 2: Sell
    action_probabilities: List[List[float]]
    rewards: List[float]
    cumulative_returns: List[float]
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    
@dataclass
class FuturesMomentumMeanReversionResult:
    """Comprehensive futures momentum and mean reversion analysis results"""
    momentum_results: MomentumResult
    mean_reversion_results: MeanReversionResult
    rl_results: Dict[str, RLAgentResult]
    combined_signals: List[str]
    strategy_performance: Dict[str, Dict[str, float]]
    risk_metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    model_confidence: Dict[str, float]

class TechnicalIndicators:
    """Technical indicators calculator"""
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if TALIB_AVAILABLE:
            return talib.RSI(np.array(prices), timeperiod=period).tolist()
        
        # Custom RSI implementation
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(delta, 0) for delta in deltas]
        losses = [max(-delta, 0) for delta in deltas]
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi_values = [50.0]  # First value
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            
            rsi_values.append(rsi)
        
        # Pad beginning with neutral values
        return [50.0] * (len(prices) - len(rsi_values)) + rsi_values
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float]]:
        """Calculate MACD and signal line"""
        if TALIB_AVAILABLE:
            macd_line, signal_line, _ = talib.MACD(np.array(prices), fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return macd_line.tolist(), signal_line.tolist()
        
        # Custom MACD implementation
        if len(prices) < slow:
            return [0.0] * len(prices), [0.0] * len(prices)
        
        # Calculate EMAs
        ema_fast = TechnicalIndicators._ema(prices, fast)
        ema_slow = TechnicalIndicators._ema(prices, slow)
        
        # MACD line
        macd_line = [ema_fast[i] - ema_slow[i] for i in range(len(prices))]
        
        # Signal line (EMA of MACD)
        signal_line = TechnicalIndicators._ema(macd_line, signal)
        
        return macd_line, signal_line
    
    @staticmethod
    def stochastic(high: List[float], low: List[float], close: List[float], 
                  k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
        """Calculate Stochastic Oscillator"""
        if TALIB_AVAILABLE:
            k, d = talib.STOCH(np.array(high), np.array(low), np.array(close), 
                              fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return k.tolist(), d.tolist()
        
        # Custom implementation
        if len(close) < k_period:
            return [50.0] * len(close), [50.0] * len(close)
        
        k_values = []
        for i in range(len(close)):
            if i < k_period - 1:
                k_values.append(50.0)
            else:
                period_high = max(high[i-k_period+1:i+1])
                period_low = min(low[i-k_period+1:i+1])
                
                if period_high == period_low:
                    k = 50.0
                else:
                    k = 100.0 * (close[i] - period_low) / (period_high - period_low)
                k_values.append(k)
        
        # %D is SMA of %K
        d_values = TechnicalIndicators._sma(k_values, d_period)
        
        return k_values, d_values
    
    @staticmethod
    def williams_r(high: List[float], low: List[float], close: List[float], period: int = 14) -> List[float]:
        """Calculate Williams %R"""
        if TALIB_AVAILABLE:
            return talib.WILLR(np.array(high), np.array(low), np.array(close), timeperiod=period).tolist()
        
        # Custom implementation
        if len(close) < period:
            return [-50.0] * len(close)
        
        wr_values = []
        for i in range(len(close)):
            if i < period - 1:
                wr_values.append(-50.0)
            else:
                period_high = max(high[i-period+1:i+1])
                period_low = min(low[i-period+1:i+1])
                
                if period_high == period_low:
                    wr = -50.0
                else:
                    wr = -100.0 * (period_high - close[i]) / (period_high - period_low)
                wr_values.append(wr)
        
        return wr_values
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands"""
        if TALIB_AVAILABLE:
            upper, middle, lower = talib.BBANDS(np.array(prices), timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
            return upper.tolist(), middle.tolist(), lower.tolist()
        
        # Custom implementation
        if len(prices) < period:
            return prices.copy(), prices.copy(), prices.copy()
        
        middle = TechnicalIndicators._sma(prices, period)
        
        upper = []
        lower = []
        
        for i in range(len(prices)):
            if i < period - 1:
                upper.append(prices[i])
                lower.append(prices[i])
            else:
                period_prices = prices[i-period+1:i+1]
                std = np.std(period_prices)
                upper.append(middle[i] + std_dev * std)
                lower.append(middle[i] - std_dev * std)
        
        return upper, middle, lower
    
    @staticmethod
    def _ema(prices: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) == 0:
            return []
        
        alpha = 2.0 / (period + 1)
        ema_values = [prices[0]]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema_values[-1]
            ema_values.append(ema)
        
        return ema_values
    
    @staticmethod
    def _sma(prices: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return prices.copy()
        
        sma_values = []
        for i in range(len(prices)):
            if i < period - 1:
                sma_values.append(prices[i])
            else:
                sma = np.mean(prices[i-period+1:i+1])
                sma_values.append(sma)
        
        return sma_values

class MomentumAnalyzer:
    """Momentum analysis for futures"""
    
    def __init__(self):
        self.model_name = "Momentum Analysis"
        
    def analyze_momentum(self, futures_data: FuturesData) -> MomentumResult:
        """Analyze momentum indicators"""
        
        prices = futures_data.close
        high = futures_data.high
        low = futures_data.low
        volume = futures_data.volume
        
        # Calculate technical indicators
        rsi_values = TechnicalIndicators.rsi(prices)
        macd_values, macd_signal = TechnicalIndicators.macd(prices)
        stochastic_k, stochastic_d = TechnicalIndicators.stochastic(high, low, prices)
        williams_r = TechnicalIndicators.williams_r(high, low, prices)
        
        # Calculate momentum scores
        momentum_scores = self._calculate_momentum_scores(
            rsi_values, macd_values, macd_signal, stochastic_k, williams_r
        )
        
        # Generate momentum signals
        momentum_signals = self._generate_momentum_signals(momentum_scores, rsi_values, macd_values, macd_signal)
        
        # Calculate momentum strength
        momentum_strength = self._calculate_momentum_strength(momentum_scores, volume)
        
        # Determine trend direction
        trend_direction = self._determine_trend_direction(prices, macd_values)
        
        # Detect momentum divergence
        momentum_divergence = self._detect_momentum_divergence(prices, rsi_values)
        
        return MomentumResult(
            momentum_scores=momentum_scores,
            momentum_signals=momentum_signals,
            momentum_strength=momentum_strength,
            trend_direction=trend_direction,
            momentum_divergence=momentum_divergence,
            rsi_values=rsi_values,
            macd_values=macd_values,
            macd_signal=macd_signal,
            stochastic_k=stochastic_k,
            stochastic_d=stochastic_d,
            williams_r=williams_r
        )
    
    def _calculate_momentum_scores(self, rsi: List[float], macd: List[float], 
                                 macd_signal: List[float], stoch_k: List[float], 
                                 williams_r: List[float]) -> List[float]:
        """Calculate composite momentum scores"""
        
        scores = []
        for i in range(len(rsi)):
            # RSI component (0-100 scale, normalize to -1 to 1)
            rsi_score = (rsi[i] - 50) / 50
            
            # MACD component
            macd_score = 1 if macd[i] > macd_signal[i] else -1
            
            # Stochastic component
            stoch_score = (stoch_k[i] - 50) / 50
            
            # Williams %R component (already -100 to 0, normalize to -1 to 1)
            wr_score = (williams_r[i] + 50) / 50
            
            # Weighted average
            composite_score = (0.3 * rsi_score + 0.3 * macd_score + 
                             0.2 * stoch_score + 0.2 * wr_score)
            
            scores.append(np.clip(composite_score, -1, 1))
        
        return scores
    
    def _generate_momentum_signals(self, momentum_scores: List[float], 
                                 rsi: List[float], macd: List[float], 
                                 macd_signal: List[float]) -> List[str]:
        """Generate trading signals based on momentum"""
        
        signals = []
        for i in range(len(momentum_scores)):
            score = momentum_scores[i]
            
            # Strong momentum signals
            if score > 0.6:
                signals.append("STRONG_BUY")
            elif score > 0.3:
                signals.append("BUY")
            elif score < -0.6:
                signals.append("STRONG_SELL")
            elif score < -0.3:
                signals.append("SELL")
            else:
                # Check for specific indicator conditions
                if rsi[i] > 70 and macd[i] < macd_signal[i]:
                    signals.append("SELL")  # Overbought with MACD divergence
                elif rsi[i] < 30 and macd[i] > macd_signal[i]:
                    signals.append("BUY")   # Oversold with MACD confirmation
                else:
                    signals.append("HOLD")
        
        return signals
    
    def _calculate_momentum_strength(self, momentum_scores: List[float], 
                                   volume: List[float]) -> List[float]:
        """Calculate momentum strength considering volume"""
        
        if not volume or len(volume) != len(momentum_scores):
            return [abs(score) for score in momentum_scores]
        
        # Normalize volume
        avg_volume = np.mean(volume)
        volume_factor = [v / avg_volume for v in volume]
        
        # Combine momentum score with volume
        strength = []
        for i in range(len(momentum_scores)):
            base_strength = abs(momentum_scores[i])
            volume_adjusted = base_strength * min(volume_factor[i], 2.0)  # Cap at 2x
            strength.append(min(volume_adjusted, 1.0))  # Cap at 1.0
        
        return strength
    
    def _determine_trend_direction(self, prices: List[float], 
                                 macd: List[float]) -> List[str]:
        """Determine trend direction"""
        
        directions = []
        
        # Calculate moving averages for trend
        short_ma = TechnicalIndicators._sma(prices, 10)
        long_ma = TechnicalIndicators._sma(prices, 30)
        
        for i in range(len(prices)):
            if i < 30:  # Not enough data
                directions.append("NEUTRAL")
                continue
            
            # Price trend
            price_trend = "UP" if short_ma[i] > long_ma[i] else "DOWN"
            
            # MACD trend
            macd_trend = "UP" if macd[i] > 0 else "DOWN"
            
            # Combine trends
            if price_trend == "UP" and macd_trend == "UP":
                directions.append("STRONG_UP")
            elif price_trend == "DOWN" and macd_trend == "DOWN":
                directions.append("STRONG_DOWN")
            elif price_trend == "UP":
                directions.append("UP")
            elif price_trend == "DOWN":
                directions.append("DOWN")
            else:
                directions.append("NEUTRAL")
        
        return directions
    
    def _detect_momentum_divergence(self, prices: List[float], 
                                  rsi: List[float]) -> List[bool]:
        """Detect momentum divergence"""
        
        divergences = [False] * len(prices)
        
        if len(prices) < 20:
            return divergences
        
        # Look for divergences in the last 20 periods
        for i in range(20, len(prices)):
            # Check for bullish divergence (price makes lower low, RSI makes higher low)
            price_window = prices[i-20:i]
            rsi_window = rsi[i-20:i]
            
            price_min_idx = price_window.index(min(price_window))
            rsi_min_idx = rsi_window.index(min(rsi_window))
            
            # Bullish divergence
            if (price_min_idx < len(price_window) - 5 and  # Price low not too recent
                rsi_min_idx < price_min_idx and  # RSI low came before price low
                rsi_window[-1] > rsi_window[rsi_min_idx]):  # RSI is higher now
                divergences[i] = True
            
            # Bearish divergence (price makes higher high, RSI makes lower high)
            price_max_idx = price_window.index(max(price_window))
            rsi_max_idx = rsi_window.index(max(rsi_window))
            
            if (price_max_idx < len(price_window) - 5 and  # Price high not too recent
                rsi_max_idx < price_max_idx and  # RSI high came before price high
                rsi_window[-1] < rsi_window[rsi_max_idx]):  # RSI is lower now
                divergences[i] = True
        
        return divergences

class MeanReversionAnalyzer:
    """Mean reversion analysis for futures"""
    
    def __init__(self):
        self.model_name = "Mean Reversion Analysis"
        
    def analyze_mean_reversion(self, futures_data: FuturesData) -> MeanReversionResult:
        """Analyze mean reversion characteristics"""
        
        prices = futures_data.close
        
        # Calculate Bollinger Bands
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(prices)
        
        # Calculate Z-scores
        z_scores = self._calculate_z_scores(prices)
        
        # Test for stationarity
        adf_pvalue = self._adf_test(prices)
        
        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(prices)
        
        # Calculate mean reversion scores
        mean_reversion_scores = self._calculate_mean_reversion_scores(prices, bb_upper, bb_lower, bb_middle)
        
        # Generate reversion signals
        reversion_signals = self._generate_reversion_signals(mean_reversion_scores, z_scores)
        
        # Calculate reversion probability
        reversion_probability = self._calculate_reversion_probability(z_scores, half_life)
        
        # Identify oversold/overbought levels
        oversold_levels = [z < -2.0 for z in z_scores]
        overbought_levels = [z > 2.0 for z in z_scores]
        
        return MeanReversionResult(
            mean_reversion_scores=mean_reversion_scores,
            reversion_signals=reversion_signals,
            bollinger_upper=bb_upper,
            bollinger_lower=bb_lower,
            bollinger_middle=bb_middle,
            z_scores=z_scores,
            adf_pvalue=adf_pvalue,
            half_life=half_life,
            reversion_probability=reversion_probability,
            oversold_levels=oversold_levels,
            overbought_levels=overbought_levels
        )
    
    def _calculate_z_scores(self, prices: List[float], window: int = 20) -> List[float]:
        """Calculate rolling Z-scores"""
        
        z_scores = []
        for i in range(len(prices)):
            if i < window - 1:
                z_scores.append(0.0)
            else:
                window_prices = prices[i-window+1:i+1]
                mean_price = np.mean(window_prices)
                std_price = np.std(window_prices)
                
                if std_price > 0:
                    z_score = (prices[i] - mean_price) / std_price
                else:
                    z_score = 0.0
                
                z_scores.append(z_score)
        
        return z_scores
    
    def _adf_test(self, prices: List[float]) -> float:
        """Augmented Dickey-Fuller test for stationarity"""
        
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(prices)
            return result[1]  # p-value
        except ImportError:
            # Simple stationarity test
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
            return 0.05 if abs(np.mean(returns)) < 0.001 else 0.5
    
    def _calculate_half_life(self, prices: List[float]) -> float:
        """Calculate half-life of mean reversion"""
        
        if len(prices) < 10:
            return 10.0
        
        # Calculate price differences
        price_diff = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        price_lag = prices[:-1]
        
        # Simple linear regression: diff = alpha + beta * lag + error
        if len(price_diff) > 0 and len(price_lag) > 0:
            correlation = np.corrcoef(price_lag, price_diff)[0, 1]
            
            # Estimate beta (mean reversion coefficient)
            beta = correlation * (np.std(price_diff) / np.std(price_lag))
            
            # Half-life = -ln(2) / ln(1 + beta)
            if beta < 0:
                half_life = -np.log(2) / np.log(1 + beta)
                return max(1.0, min(half_life, 100.0))  # Reasonable bounds
        
        return 10.0  # Default half-life
    
    def _calculate_mean_reversion_scores(self, prices: List[float], 
                                       bb_upper: List[float], bb_lower: List[float], 
                                       bb_middle: List[float]) -> List[float]:
        """Calculate mean reversion scores"""
        
        scores = []
        for i in range(len(prices)):
            price = prices[i]
            upper = bb_upper[i]
            lower = bb_lower[i]
            middle = bb_middle[i]
            
            # Calculate position within Bollinger Bands
            if upper > lower:
                bb_position = (price - lower) / (upper - lower)
            else:
                bb_position = 0.5
            
            # Mean reversion score (higher when price is far from mean)
            if bb_position > 0.8:  # Near upper band
                score = -(bb_position - 0.5) * 2  # Negative for sell signal
            elif bb_position < 0.2:  # Near lower band
                score = -(bb_position - 0.5) * 2  # Positive for buy signal
            else:
                score = 0.0
            
            scores.append(np.clip(score, -1, 1))
        
        return scores
    
    def _generate_reversion_signals(self, mean_reversion_scores: List[float], 
                                  z_scores: List[float]) -> List[str]:
        """Generate mean reversion trading signals"""
        
        signals = []
        for i in range(len(mean_reversion_scores)):
            mr_score = mean_reversion_scores[i]
            z_score = z_scores[i]
            
            # Strong reversion signals
            if mr_score > 0.6 and z_score < -2.0:
                signals.append("STRONG_BUY")  # Oversold, expect reversion up
            elif mr_score < -0.6 and z_score > 2.0:
                signals.append("STRONG_SELL")  # Overbought, expect reversion down
            elif mr_score > 0.3 and z_score < -1.5:
                signals.append("BUY")
            elif mr_score < -0.3 and z_score > 1.5:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        return signals
    
    def _calculate_reversion_probability(self, z_scores: List[float], 
                                       half_life: float) -> List[float]:
        """Calculate probability of mean reversion"""
        
        probabilities = []
        for z_score in z_scores:
            # Higher Z-score magnitude = higher reversion probability
            base_prob = 1 - np.exp(-abs(z_score) / 2)
            
            # Adjust for half-life (shorter half-life = higher probability)
            half_life_factor = 1 / (1 + half_life / 10)
            
            prob = base_prob * half_life_factor
            probabilities.append(min(prob, 0.95))  # Cap at 95%
        
        return probabilities

class SimpleFuturesTradingEnv:
    """Simple futures trading environment for RL"""
    
    def __init__(self, futures_data: FuturesData, lookback_window: int = 10):
        self.futures_data = futures_data
        self.lookback_window = lookback_window
        self.current_step = lookback_window
        self.position = 0  # -1: Short, 0: Neutral, 1: Long
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.trade_history = []
        
    def reset(self):
        """Reset environment"""
        self.current_step = self.lookback_window
        self.position = 0
        self.entry_price = 0.0
        self.total_reward = 0.0
        self.trade_history = []
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state"""
        # Actions: 0=Hold, 1=Buy, 2=Sell
        reward = 0.0
        done = False
        
        current_price = self.futures_data.close[self.current_step]
        
        # Execute action
        if action == 1 and self.position <= 0:  # Buy
            if self.position == -1:  # Close short position
                reward += (self.entry_price - current_price) / self.entry_price
            self.position = 1
            self.entry_price = current_price
            
        elif action == 2 and self.position >= 0:  # Sell
            if self.position == 1:  # Close long position
                reward += (current_price - self.entry_price) / self.entry_price
            self.position = -1
            self.entry_price = current_price
        
        # Calculate holding reward
        if self.position != 0:
            if self.position == 1:  # Long position
                holding_reward = (current_price - self.entry_price) / self.entry_price * 0.01
            else:  # Short position
                holding_reward = (self.entry_price - current_price) / self.entry_price * 0.01
            reward += holding_reward
        
        self.total_reward += reward
        self.current_step += 1
        
        # Check if episode is done
        if self.current_step >= len(self.futures_data.close) - 1:
            done = True
        
        return self._get_observation(), reward, done, {}
    
    def _get_observation(self):
        """Get current observation"""
        if self.current_step < self.lookback_window:
            return np.zeros(self.lookback_window * 4 + 1)
        
        # Price features
        prices = self.futures_data.close[self.current_step-self.lookback_window:self.current_step]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        returns = [0.0] + returns  # Pad first return
        
        # Volume features
        volumes = self.futures_data.volume[self.current_step-self.lookback_window:self.current_step]
        volume_norm = [v / max(volumes) if max(volumes) > 0 else 0 for v in volumes]
        
        # Technical indicators
        rsi = TechnicalIndicators.rsi(prices)[-self.lookback_window:]
        
        # Combine features
        observation = np.array(returns + volume_norm + rsi + [self.position])
        return observation

class SimpleRLAgent:
    """Simple reinforcement learning agent"""
    
    def __init__(self, agent_type: str = "PPO"):
        self.agent_type = agent_type
        self.model = None
        
    def train(self, env: SimpleFuturesTradingEnv, total_timesteps: int = 10000):
        """Train the RL agent"""
        
        if RL_LIBS_AVAILABLE:
            try:
                # Create vectorized environment
                vec_env = DummyVecEnv([lambda: env])
                
                # Initialize model based on type
                if self.agent_type == "PPO":
                    self.model = PPO("MlpPolicy", vec_env, verbose=0)
                elif self.agent_type == "SAC":
                    self.model = SAC("MlpPolicy", vec_env, verbose=0)
                elif self.agent_type == "DDPG":
                    self.model = DDPG("MlpPolicy", vec_env, verbose=0)
                else:
                    self.model = PPO("MlpPolicy", vec_env, verbose=0)
                
                # Train the model
                self.model.learn(total_timesteps=total_timesteps)
                
            except Exception as e:
                print(f"RL training failed: {e}")
                self._train_simple_agent(env)
        else:
            self._train_simple_agent(env)
    
    def _train_simple_agent(self, env: SimpleFuturesTradingEnv):
        """Simple rule-based agent training"""
        # This is a placeholder for when RL libraries are not available
        print(f"Training simple {self.agent_type} agent...")
        
    def predict(self, observation):
        """Predict action given observation"""
        
        if self.model is not None and RL_LIBS_AVAILABLE:
            try:
                action, _ = self.model.predict(observation, deterministic=True)
                return action, [0.33, 0.33, 0.34]  # Dummy probabilities
            except:
                pass
        
        # Simple rule-based prediction
        return self._simple_predict(observation)
    
    def _simple_predict(self, observation):
        """Simple rule-based prediction"""
        
        # Extract features from observation
        obs_len = len(observation)
        lookback = (obs_len - 1) // 4
        
        if lookback > 0:
            returns = observation[:lookback]
            rsi_values = observation[lookback*3:lookback*4]
            
            # Simple momentum strategy
            recent_return = np.mean(returns[-3:]) if len(returns) >= 3 else 0
            current_rsi = rsi_values[-1] if len(rsi_values) > 0 else 50
            
            if recent_return > 0.01 and current_rsi < 70:
                return 1, [0.1, 0.8, 0.1]  # Buy
            elif recent_return < -0.01 and current_rsi > 30:
                return 2, [0.1, 0.1, 0.8]  # Sell
            else:
                return 0, [0.8, 0.1, 0.1]  # Hold
        
        return 0, [0.33, 0.33, 0.34]

class RLAnalyzer:
    """Reinforcement Learning analyzer for futures trading"""
    
    def __init__(self):
        self.model_name = "RL Analysis"
        
    def analyze_rl_agents(self, futures_data: FuturesData) -> Dict[str, RLAgentResult]:
        """Analyze multiple RL agents"""
        
        results = {}
        agent_types = ["PPO", "SAC", "DDPG"]
        
        for agent_type in agent_types:
            print(f"Training {agent_type} agent...")
            result = self._train_and_evaluate_agent(futures_data, agent_type)
            results[agent_type] = result
        
        return results
    
    def _train_and_evaluate_agent(self, futures_data: FuturesData, agent_type: str) -> RLAgentResult:
        """Train and evaluate a single RL agent"""
        
        # Create environment
        env = SimpleFuturesTradingEnv(futures_data)
        
        # Create and train agent
        agent = SimpleRLAgent(agent_type)
        agent.train(env, total_timesteps=5000)  # Reduced for demo
        
        # Evaluate agent
        env.reset()
        actions = []
        action_probabilities = []
        rewards = []
        cumulative_returns = [0.0]
        
        done = False
        while not done:
            obs = env._get_observation()
            action, probs = agent.predict(obs)
            
            actions.append(action)
            action_probabilities.append(probs)
            
            _, reward, done, _ = env.step(action)
            rewards.append(reward)
            cumulative_returns.append(cumulative_returns[-1] + reward)
        
        # Calculate performance metrics
        total_return = cumulative_returns[-1]
        returns_array = np.array(rewards)
        
        # Sharpe ratio
        if len(returns_array) > 1 and np.std(returns_array) > 0:
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        
        # Win rate
        profitable_trades = sum(1 for r in rewards if r > 0)
        total_trades = len([a for a in actions if a != 0])  # Non-hold actions
        win_rate = profitable_trades / max(total_trades, 1)
        
        return RLAgentResult(
            agent_type=agent_type,
            actions=actions,
            action_probabilities=action_probabilities,
            rewards=rewards,
            cumulative_returns=cumulative_returns,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profitable_trades=profitable_trades
        )
    
    def _calculate_max_drawdown(self, cumulative_returns: List[float]) -> float:
        """Calculate maximum drawdown"""
        
        if len(cumulative_returns) < 2:
            return 0.0
        
        peak = cumulative_returns[0]
        max_dd = 0.0
        
        for ret in cumulative_returns[1:]:
            if ret > peak:
                peak = ret
            else:
                drawdown = (peak - ret) / max(abs(peak), 1e-6)
                max_dd = max(max_dd, drawdown)
        
        return max_dd

class FuturesMomentumMeanReversionAnalyzer:
    """Comprehensive futures momentum and mean reversion analyzer"""
    
    def __init__(self):
        self.momentum_analyzer = MomentumAnalyzer()
        self.mean_reversion_analyzer = MeanReversionAnalyzer()
        self.rl_analyzer = RLAnalyzer()
        
    def analyze(self, futures_data: FuturesData) -> FuturesMomentumMeanReversionResult:
        """Perform comprehensive analysis"""
        
        print(f"Analyzing momentum and mean reversion for {futures_data.contract_symbol}...")
        
        # Momentum analysis
        momentum_results = self.momentum_analyzer.analyze_momentum(futures_data)
        
        # Mean reversion analysis
        mean_reversion_results = self.mean_reversion_analyzer.analyze_mean_reversion(futures_data)
        
        # RL analysis
        rl_results = self.rl_analyzer.analyze_rl_agents(futures_data)
        
        # Combine signals
        combined_signals = self._combine_signals(momentum_results, mean_reversion_results, rl_results)
        
        # Calculate strategy performance
        strategy_performance = self._calculate_strategy_performance(
            futures_data, momentum_results, mean_reversion_results, rl_results
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(futures_data, combined_signals)
        
        # Generate insights
        insights = self._generate_insights(
            momentum_results, mean_reversion_results, rl_results, strategy_performance
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            momentum_results, mean_reversion_results, rl_results, risk_metrics
        )
        
        # Calculate model confidence
        model_confidence = self._calculate_model_confidence(
            momentum_results, mean_reversion_results, rl_results
        )
        
        return FuturesMomentumMeanReversionResult(
            momentum_results=momentum_results,
            mean_reversion_results=mean_reversion_results,
            rl_results=rl_results,
            combined_signals=combined_signals,
            strategy_performance=strategy_performance,
            risk_metrics=risk_metrics,
            insights=insights,
            recommendations=recommendations,
            model_confidence=model_confidence
        )
    
    def _combine_signals(self, momentum_results: MomentumResult,
                        mean_reversion_results: MeanReversionResult,
                        rl_results: Dict[str, RLAgentResult]) -> List[str]:
        """Combine signals from different strategies"""
        
        n_periods = len(momentum_results.momentum_signals)
        combined_signals = []
        
        for i in range(n_periods):
            momentum_signal = momentum_results.momentum_signals[i]
            reversion_signal = mean_reversion_results.reversion_signals[i]
            
            # Get RL signals (use PPO as primary)
            rl_signal = "HOLD"
            if "PPO" in rl_results and i < len(rl_results["PPO"].actions):
                action = rl_results["PPO"].actions[i]
                if action == 1:
                    rl_signal = "BUY"
                elif action == 2:
                    rl_signal = "SELL"
            
            # Combine signals with voting
            signals = [momentum_signal, reversion_signal, rl_signal]
            
            # Count votes
            buy_votes = sum(1 for s in signals if "BUY" in s)
            sell_votes = sum(1 for s in signals if "SELL" in s)
            
            # Determine combined signal
            if buy_votes >= 2:
                if "STRONG_BUY" in signals:
                    combined_signals.append("STRONG_BUY")
                else:
                    combined_signals.append("BUY")
            elif sell_votes >= 2:
                if "STRONG_SELL" in signals:
                    combined_signals.append("STRONG_SELL")
                else:
                    combined_signals.append("SELL")
            else:
                combined_signals.append("HOLD")
        
        return combined_signals
    
    def _calculate_strategy_performance(self, futures_data: FuturesData,
                                      momentum_results: MomentumResult,
                                      mean_reversion_results: MeanReversionResult,
                                      rl_results: Dict[str, RLAgentResult]) -> Dict[str, Dict[str, float]]:
        """Calculate performance of different strategies"""
        
        performance = {}
        
        # Momentum strategy performance
        momentum_returns = self._calculate_signal_returns(
            futures_data.returns, momentum_results.momentum_signals
        )
        performance["momentum"] = {
            "total_return": sum(momentum_returns),
            "sharpe_ratio": self._calculate_sharpe_ratio(momentum_returns),
            "max_drawdown": self._calculate_max_drawdown_from_returns(momentum_returns),
            "win_rate": self._calculate_win_rate(momentum_returns)
        }
        
        # Mean reversion strategy performance
        reversion_returns = self._calculate_signal_returns(
            futures_data.returns, mean_reversion_results.reversion_signals
        )
        performance["mean_reversion"] = {
            "total_return": sum(reversion_returns),
            "sharpe_ratio": self._calculate_sharpe_ratio(reversion_returns),
            "max_drawdown": self._calculate_max_drawdown_from_returns(reversion_returns),
            "win_rate": self._calculate_win_rate(reversion_returns)
        }
        
        # RL strategies performance
        for agent_type, rl_result in rl_results.items():
            performance[f"rl_{agent_type.lower()}"] = {
                "total_return": rl_result.cumulative_returns[-1] if rl_result.cumulative_returns else 0,
                "sharpe_ratio": rl_result.sharpe_ratio,
                "max_drawdown": rl_result.max_drawdown,
                "win_rate": rl_result.win_rate
            }
        
        return performance
    
    def _calculate_signal_returns(self, market_returns: List[float], signals: List[str]) -> List[float]:
        """Calculate returns based on trading signals"""
        
        strategy_returns = []
        position = 0  # 0: No position, 1: Long, -1: Short
        
        for i, signal in enumerate(signals):
            if i >= len(market_returns):
                break
                
            market_return = market_returns[i]
            
            # Update position based on signal
            if "BUY" in signal:
                position = 1
            elif "SELL" in signal:
                position = -1
            elif signal == "HOLD" and i == 0:
                position = 0
            
            # Calculate strategy return
            strategy_return = position * market_return
            strategy_returns.append(strategy_return)
        
        return strategy_returns
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return * np.sqrt(252)  # Annualized
    
    def _calculate_max_drawdown_from_returns(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumsum(returns)
        return self._calculate_max_drawdown(cumulative.tolist())
    
    def _calculate_win_rate(self, returns: List[float]) -> float:
        """Calculate win rate"""
        
        if len(returns) == 0:
            return 0.0
        
        winning_periods = sum(1 for r in returns if r > 0)
        return winning_periods / len(returns)
    
    def _calculate_risk_metrics(self, futures_data: FuturesData, 
                              combined_signals: List[str]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        metrics = {}
        
        # Basic market metrics
        returns = futures_data.returns
        if returns:
            metrics["market_volatility"] = np.std(returns)
            metrics["market_skewness"] = stats.skew(returns)
            metrics["market_kurtosis"] = stats.kurtosis(returns)
        
        # Strategy-specific metrics
        strategy_returns = self._calculate_signal_returns(returns, combined_signals)
        if strategy_returns:
            metrics["strategy_volatility"] = np.std(strategy_returns)
            metrics["strategy_skewness"] = stats.skew(strategy_returns)
            metrics["strategy_kurtosis"] = stats.kurtosis(strategy_returns)
            
            # Value at Risk (5%)
            metrics["var_5pct"] = np.percentile(strategy_returns, 5)
            
            # Expected Shortfall (5%)
            var_threshold = metrics["var_5pct"]
            tail_returns = [r for r in strategy_returns if r <= var_threshold]
            metrics["expected_shortfall_5pct"] = np.mean(tail_returns) if tail_returns else 0
        
        # Signal consistency
        signal_changes = sum(1 for i in range(1, len(combined_signals)) 
                           if combined_signals[i] != combined_signals[i-1])
        metrics["signal_stability"] = 1 - (signal_changes / max(len(combined_signals), 1))
        
        return metrics
    
    def _generate_insights(self, momentum_results: MomentumResult,
                         mean_reversion_results: MeanReversionResult,
                         rl_results: Dict[str, RLAgentResult],
                         strategy_performance: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate analytical insights"""
        
        insights = []
        
        # Momentum insights
        avg_momentum = np.mean(momentum_results.momentum_scores)
        if avg_momentum > 0.2:
            insights.append(f"Strong positive momentum detected (avg: {avg_momentum:.2f})")
        elif avg_momentum < -0.2:
            insights.append(f"Strong negative momentum detected (avg: {avg_momentum:.2f})")
        
        # Mean reversion insights
        if mean_reversion_results.adf_pvalue < 0.05:
            insights.append(f"Price series is stationary (ADF p-value: {mean_reversion_results.adf_pvalue:.3f})")
            insights.append(f"Mean reversion half-life: {mean_reversion_results.half_life:.1f} periods")
        
        # Strategy performance insights
        best_strategy = max(strategy_performance.keys(), 
                          key=lambda k: strategy_performance[k]["sharpe_ratio"])
        best_sharpe = strategy_performance[best_strategy]["sharpe_ratio"]
        insights.append(f"Best performing strategy: {best_strategy} (Sharpe: {best_sharpe:.2f})")
        
        # RL insights
        if rl_results:
            best_rl = max(rl_results.keys(), key=lambda k: rl_results[k].sharpe_ratio)
            insights.append(f"Best RL agent: {best_rl} (Win rate: {rl_results[best_rl].win_rate:.1%})")
        
        # Divergence insights
        divergence_count = sum(momentum_results.momentum_divergence)
        if divergence_count > len(momentum_results.momentum_divergence) * 0.1:
            insights.append(f"Frequent momentum divergences detected ({divergence_count} instances)")
        
        return insights
    
    def _generate_recommendations(self, momentum_results: MomentumResult,
                                mean_reversion_results: MeanReversionResult,
                                rl_results: Dict[str, RLAgentResult],
                                risk_metrics: Dict[str, float]) -> List[str]:
        """Generate trading recommendations"""
        
        recommendations = []
        
        # Strategy selection recommendations
        if mean_reversion_results.adf_pvalue < 0.05 and mean_reversion_results.half_life < 20:
            recommendations.append("Market shows strong mean reversion - favor contrarian strategies")
        else:
            recommendations.append("Market shows momentum characteristics - favor trend-following strategies")
        
        # Risk management recommendations
        strategy_vol = risk_metrics.get("strategy_volatility", 0)
        if strategy_vol > 0.03:
            recommendations.append(f"High strategy volatility ({strategy_vol:.1%}) - reduce position sizes")
        
        var_5pct = risk_metrics.get("var_5pct", 0)
        if var_5pct < -0.05:
            recommendations.append(f"High tail risk (5% VaR: {var_5pct:.1%}) - implement strict stop losses")
        
        # Signal stability recommendations
        signal_stability = risk_metrics.get("signal_stability", 1)
        if signal_stability < 0.7:
            recommendations.append("Low signal stability - consider longer-term indicators or signal smoothing")
        
        # RL-specific recommendations
        if rl_results:
            best_rl_agent = max(rl_results.keys(), key=lambda k: rl_results[k].sharpe_ratio)
            recommendations.append(f"Consider using {best_rl_agent} agent for automated trading")
        
        # Technical recommendations
        current_rsi = momentum_results.rsi_values[-1] if momentum_results.rsi_values else 50
        if current_rsi > 70:
            recommendations.append("RSI indicates overbought conditions - consider taking profits")
        elif current_rsi < 30:
            recommendations.append("RSI indicates oversold conditions - consider buying opportunities")
        
        # General recommendations
        recommendations.append("Combine momentum and mean reversion signals for robust trading decisions")
        recommendations.append("Regularly retrain RL models with new market data")
        recommendations.append("Monitor regime changes that may affect strategy effectiveness")
        
        return recommendations
    
    def _calculate_model_confidence(self, momentum_results: MomentumResult,
                                  mean_reversion_results: MeanReversionResult,
                                  rl_results: Dict[str, RLAgentResult]) -> Dict[str, float]:
        """Calculate confidence scores for different models"""
        
        confidence = {}
        
        # Momentum model confidence
        momentum_strength_avg = np.mean(momentum_results.momentum_strength)
        confidence["momentum"] = min(momentum_strength_avg * 2, 1.0)  # Scale to 0-1
        
        # Mean reversion model confidence
        if mean_reversion_results.adf_pvalue < 0.05:
            stationarity_confidence = 1 - mean_reversion_results.adf_pvalue
        else:
            stationarity_confidence = 0.5
        
        half_life_confidence = 1 / (1 + mean_reversion_results.half_life / 20)
        confidence["mean_reversion"] = (stationarity_confidence + half_life_confidence) / 2
        
        # RL model confidence
        for agent_type, rl_result in rl_results.items():
            # Base confidence on win rate and Sharpe ratio
            win_rate_score = rl_result.win_rate
            sharpe_score = max(0, min(rl_result.sharpe_ratio / 2, 1))  # Normalize Sharpe
            confidence[f"rl_{agent_type.lower()}"] = (win_rate_score + sharpe_score) / 2
        
        # Overall confidence
        confidence["overall"] = np.mean(list(confidence.values()))
        
        return confidence
    
    def plot_results(self, futures_data: FuturesData, 
                    results: FuturesMomentumMeanReversionResult):
        """Plot comprehensive analysis results"""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        timestamps = futures_data.timestamps
        
        # Plot 1: Price and Bollinger Bands
        ax1 = axes[0, 0]
        ax1.plot(timestamps, futures_data.close, label='Price', linewidth=2)
        ax1.plot(timestamps, results.mean_reversion_results.bollinger_upper, 
                label='BB Upper', linestyle='--', alpha=0.7)
        ax1.plot(timestamps, results.mean_reversion_results.bollinger_lower, 
                label='BB Lower', linestyle='--', alpha=0.7)
        ax1.fill_between(timestamps, 
                        results.mean_reversion_results.bollinger_upper,
                        results.mean_reversion_results.bollinger_lower,
                        alpha=0.1)
        
        ax1.set_title('Price and Bollinger Bands', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Momentum Indicators
        ax2 = axes[0, 1]
        ax2.plot(timestamps, results.momentum_results.rsi_values, label='RSI', color='purple')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        
        ax2.set_title('RSI Momentum Indicator', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: MACD
        ax3 = axes[1, 0]
        ax3.plot(timestamps, results.momentum_results.macd_values, label='MACD', color='blue')
        ax3.plot(timestamps, results.momentum_results.macd_signal, label='Signal', color='red')
        ax3.bar(timestamps, [m - s for m, s in zip(results.momentum_results.macd_values, 
                                                   results.momentum_results.macd_signal)], 
               label='Histogram', alpha=0.3, color='green')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        ax3.set_title('MACD Indicator', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('MACD')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Z-Scores and Mean Reversion
        ax4 = axes[1, 1]
        ax4.plot(timestamps, results.mean_reversion_results.z_scores, 
                label='Z-Score', color='orange', linewidth=2)
        ax4.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Overbought (+2σ)')
        ax4.axhline(y=-2, color='green', linestyle='--', alpha=0.7, label='Oversold (-2σ)')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.fill_between(timestamps, 2, -2, alpha=0.1, color='gray')
        
        ax4.set_title('Z-Score Mean Reversion', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Z-Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Combined Trading Signals
        ax5 = axes[2, 0]
        signal_colors = {'STRONG_BUY': 'darkgreen', 'BUY': 'green', 'HOLD': 'gray', 
                        'SELL': 'red', 'STRONG_SELL': 'darkred'}
        
        # Create signal plot
        signal_numeric = []
        for signal in results.combined_signals:
            if signal == 'STRONG_BUY':
                signal_numeric.append(2)
            elif signal == 'BUY':
                signal_numeric.append(1)
            elif signal == 'HOLD':
                signal_numeric.append(0)
            elif signal == 'SELL':
                signal_numeric.append(-1)
            elif signal == 'STRONG_SELL':
                signal_numeric.append(-2)
            else:
                signal_numeric.append(0)
        
        ax5.plot(timestamps, signal_numeric, marker='o', markersize=3, linewidth=1)
        ax5.set_ylim(-2.5, 2.5)
        ax5.set_yticks([-2, -1, 0, 1, 2])
        ax5.set_yticklabels(['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy'])
        ax5.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax5.set_title('Combined Trading Signals', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Time')
        ax5.set_ylabel('Signal')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Strategy Performance Comparison
        ax6 = axes[2, 1]
        strategies = list(results.strategy_performance.keys())
        sharpe_ratios = [results.strategy_performance[s]['sharpe_ratio'] for s in strategies]
        
        bars = ax6.bar(strategies, sharpe_ratios, 
                      color=['blue', 'green', 'red', 'purple', 'orange'][:len(strategies)])
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax6.set_title('Strategy Performance (Sharpe Ratio)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Strategy')
        ax6.set_ylabel('Sharpe Ratio')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, futures_data: FuturesData, 
                       results: FuturesMomentumMeanReversionResult) -> str:
        """Generate comprehensive analysis report"""
        
        report = f"""
# FUTURES MOMENTUM & MEAN REVERSION ANALYSIS REPORT
## Contract: {futures_data.contract_symbol} ({futures_data.underlying_asset})
## Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### EXECUTIVE SUMMARY
This report presents a comprehensive analysis of momentum and mean reversion characteristics 
for the {futures_data.contract_symbol} futures contract, incorporating advanced machine learning 
and reinforcement learning techniques.

### MOMENTUM ANALYSIS
**Current Momentum Score:** {results.momentum_results.momentum_scores[-1]:.3f}
**Current RSI:** {results.momentum_results.rsi_values[-1]:.1f}
**Current MACD:** {results.momentum_results.macd_values[-1]:.4f}
**Trend Direction:** {results.momentum_results.trend_direction[-1]}
**Momentum Divergences Detected:** {sum(results.momentum_results.momentum_divergence)}

### MEAN REVERSION ANALYSIS
**Stationarity (ADF p-value):** {results.mean_reversion_results.adf_pvalue:.4f}
**Mean Reversion Half-Life:** {results.mean_reversion_results.half_life:.1f} periods
**Current Z-Score:** {results.mean_reversion_results.z_scores[-1]:.3f}
**Current Mean Reversion Score:** {results.mean_reversion_results.mean_reversion_scores[-1]:.3f}
**Reversion Probability:** {results.mean_reversion_results.reversion_probability[-1]:.1%}

### REINFORCEMENT LEARNING ANALYSIS
"""
        
        for agent_type, rl_result in results.rl_results.items():
            report += f"""
**{agent_type} Agent Performance:**
- Total Return: {rl_result.cumulative_returns[-1]:.2%}
- Sharpe Ratio: {rl_result.sharpe_ratio:.2f}
- Maximum Drawdown: {rl_result.max_drawdown:.2%}
- Win Rate: {rl_result.win_rate:.1%}
- Total Trades: {rl_result.total_trades}
"""
        
        report += f"""

### STRATEGY PERFORMANCE COMPARISON
"""
        
        for strategy, performance in results.strategy_performance.items():
            report += f"""
**{strategy.replace('_', ' ').title()} Strategy:**
- Total Return: {performance['total_return']:.2%}
- Sharpe Ratio: {performance['sharpe_ratio']:.2f}
- Maximum Drawdown: {performance['max_drawdown']:.2%}
- Win Rate: {performance['win_rate']:.1%}
"""
        
        report += f"""

### RISK METRICS
**Market Volatility:** {results.risk_metrics.get('market_volatility', 0):.2%}
**Strategy Volatility:** {results.risk_metrics.get('strategy_volatility', 0):.2%}
**Value at Risk (5%):** {results.risk_metrics.get('var_5pct', 0):.2%}
**Expected Shortfall (5%):** {results.risk_metrics.get('expected_shortfall_5pct', 0):.2%}
**Signal Stability:** {results.risk_metrics.get('signal_stability', 0):.1%}

### KEY INSIGHTS
"""
        
        for i, insight in enumerate(results.insights, 1):
            report += f"{i}. {insight}\n"
        
        report += f"""

### RECOMMENDATIONS
"""
        
        for i, recommendation in enumerate(results.recommendations, 1):
            report += f"{i}. {recommendation}\n"
        
        report += f"""

### MODEL CONFIDENCE SCORES
"""
        
        for model, confidence in results.model_confidence.items():
            report += f"**{model.replace('_', ' ').title()}:** {confidence:.1%}\n"
        
        report += f"""

### CURRENT TRADING SIGNAL
**Combined Signal:** {results.combined_signals[-1]}

### METHODOLOGY
This analysis employs multiple complementary approaches:

1. **Momentum Analysis:** RSI, MACD, Stochastic Oscillator, Williams %R
2. **Mean Reversion Analysis:** Bollinger Bands, Z-scores, ADF stationarity test
3. **Reinforcement Learning:** PPO, SAC, and DDPG agents trained on historical data
4. **Signal Combination:** Voting mechanism across all strategies
5. **Risk Assessment:** VaR, Expected Shortfall, volatility metrics

**Disclaimer:** This analysis is for informational purposes only and should not be 
considered as financial advice. Past performance does not guarantee future results.
"""
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Create sample futures data
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    
    # Generate realistic futures price data
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_periods)  # Daily returns
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    high_prices = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    low_prices = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    open_prices = [prices[0]] + prices[:-1]  # Previous close as next open
    
    # Generate volume and open interest
    base_volume = 10000
    volumes = [base_volume * (1 + np.random.normal(0, 0.3)) for _ in range(len(prices))]
    volumes = [max(1000, v) for v in volumes]  # Ensure positive volume
    
    open_interests = [50000 * (1 + np.random.normal(0, 0.1)) for _ in range(len(prices))]
    open_interests = [max(10000, oi) for oi in open_interests]  # Ensure positive OI
    
    # Create timestamps
    start_date = datetime.now() - timedelta(days=n_periods)
    timestamps = [start_date + timedelta(days=i) for i in range(len(prices))]
    
    # Create FuturesData object
    futures_data = FuturesData(
        prices=prices,
        returns=returns.tolist(),
        volume=volumes,
        open_interest=open_interests,
        timestamps=timestamps,
        high=high_prices,
        low=low_prices,
        open=open_prices,
        close=prices,
        contract_symbol="CL_2024_03",
        underlying_asset="Crude Oil"
    )
    
    # Initialize analyzer
    analyzer = FuturesMomentumMeanReversionAnalyzer()
    
    # Perform analysis
    print("Performing comprehensive futures momentum and mean reversion analysis...")
    results = analyzer.analyze(futures_data)
    
    # Print summary
    print("\n" + "="*80)
    print("FUTURES MOMENTUM & MEAN REVERSION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nContract: {futures_data.contract_symbol}")
    print(f"Underlying Asset: {futures_data.underlying_asset}")
    print(f"Analysis Period: {len(futures_data.prices)} periods")
    
    print(f"\nCurrent Signals:")
    print(f"- Momentum: {results.momentum_results.momentum_signals[-1]}")
    print(f"- Mean Reversion: {results.mean_reversion_results.reversion_signals[-1]}")
    print(f"- Combined: {results.combined_signals[-1]}")
    
    print(f"\nStrategy Performance (Sharpe Ratios):")
    for strategy, performance in results.strategy_performance.items():
        print(f"- {strategy.replace('_', ' ').title()}: {performance['sharpe_ratio']:.2f}")
    
    print(f"\nRL Agent Performance:")
    for agent_type, rl_result in results.rl_results.items():
        print(f"- {agent_type}: Return={rl_result.cumulative_returns[-1]:.2%}, "
              f"Sharpe={rl_result.sharpe_ratio:.2f}, Win Rate={rl_result.win_rate:.1%}")
    
    print(f"\nKey Insights:")
    for i, insight in enumerate(results.insights[:5], 1):
        print(f"{i}. {insight}")
    
    print(f"\nTop Recommendations:")
    for i, recommendation in enumerate(results.recommendations[:3], 1):
        print(f"{i}. {recommendation}")
    
    print(f"\nModel Confidence:")
    for model, confidence in results.model_confidence.items():
        print(f"- {model.replace('_', ' ').title()}: {confidence:.1%}")
    
    # Generate and save report
    report = analyzer.generate_report(futures_data, results)
    
    try:
        with open("futures_momentum_mean_reversion_analysis_report.txt", "w") as f:
            f.write(report)
        print(f"\nDetailed report saved to: futures_momentum_mean_reversion_analysis_report.txt")
    except Exception as e:
        print(f"\nCould not save report: {e}")
    
    # Plot results
    try:
        print("\nGenerating plots...")
        analyzer.plot_results(futures_data, results)
    except Exception as e:
        print(f"Could not generate plots: {e}")
    
    print("\nAnalysis completed successfully!")