"""Technical Indicators for Futures Trading Analysis

This module provides technical indicators and momentum analysis for futures trading,
including RSI, MACD, Stochastic Oscillator, Williams %R, and Bollinger Bands.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings

# Conditional imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not available. Using custom implementations.")

@dataclass
class FuturesData:
    """Data structure for futures market data"""
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

class TechnicalIndicators:
    """Technical indicators with fallback implementations"""
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI with fallback implementation"""
        
        if TALIB_AVAILABLE:
            try:
                return talib.RSI(np.array(prices), timeperiod=period).tolist()
            except Exception:
                pass
        
        # Fallback implementation
        if len(prices) < period + 1:
            return [50.0] * len(prices)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = []
        avg_losses = []
        rsi_values = []
        
        # Initial averages
        initial_gain = np.mean(gains[:period])
        initial_loss = np.mean(losses[:period])
        
        avg_gains.append(initial_gain)
        avg_losses.append(initial_loss)
        
        if initial_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = initial_gain / initial_loss
            rsi_values.append(100 - (100 / (1 + rs)))
        
        # Calculate subsequent RSI values
        for i in range(period, len(deltas)):
            avg_gain = (avg_gains[-1] * (period - 1) + gains[i]) / period
            avg_loss = (avg_losses[-1] * (period - 1) + losses[i]) / period
            
            avg_gains.append(avg_gain)
            avg_losses.append(avg_loss)
            
            if avg_loss == 0:
                rsi_values.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_values.append(100 - (100 / (1 + rs)))
        
        # Pad with initial values
        result = [50.0] * period + rsi_values
        return result[:len(prices)]
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float]]:
        """Calculate MACD with fallback implementation"""
        
        if TALIB_AVAILABLE:
            try:
                macd_line, signal_line, _ = talib.MACD(np.array(prices), 
                                                      fastperiod=fast, 
                                                      slowperiod=slow, 
                                                      signalperiod=signal)
                return macd_line.tolist(), signal_line.tolist()
            except Exception:
                pass
        
        # Fallback implementation
        if len(prices) < slow:
            return [0.0] * len(prices), [0.0] * len(prices)
        
        # Calculate EMAs
        def ema(data, period):
            alpha = 2 / (period + 1)
            ema_values = [data[0]]
            for price in data[1:]:
                ema_values.append(alpha * price + (1 - alpha) * ema_values[-1])
            return ema_values
        
        fast_ema = ema(prices, fast)
        slow_ema = ema(prices, slow)
        
        # Calculate MACD line
        macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
        
        # Calculate signal line
        signal_line = ema(macd_line, signal)
        
        return macd_line, signal_line
    
    @staticmethod
    def stochastic(high: List[float], low: List[float], close: List[float], 
                  k_period: int = 14, d_period: int = 3) -> Tuple[List[float], List[float]]:
        """Calculate Stochastic Oscillator with fallback implementation"""
        
        if TALIB_AVAILABLE:
            try:
                k_values, d_values = talib.STOCH(np.array(high), np.array(low), 
                                               np.array(close), 
                                               fastk_period=k_period, 
                                               slowk_period=d_period, 
                                               slowd_period=d_period)
                return k_values.tolist(), d_values.tolist()
            except Exception:
                pass
        
        # Fallback implementation
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
                    k_values.append(50.0)
                else:
                    k_percent = 100 * (close[i] - period_low) / (period_high - period_low)
                    k_values.append(k_percent)
        
        # Calculate %D (moving average of %K)
        d_values = []
        for i in range(len(k_values)):
            if i < d_period - 1:
                d_values.append(k_values[i])
            else:
                d_values.append(np.mean(k_values[i-d_period+1:i+1]))
        
        return k_values, d_values
    
    @staticmethod
    def williams_r(high: List[float], low: List[float], close: List[float], 
                  period: int = 14) -> List[float]:
        """Calculate Williams %R with fallback implementation"""
        
        if TALIB_AVAILABLE:
            try:
                return talib.WILLR(np.array(high), np.array(low), 
                                 np.array(close), timeperiod=period).tolist()
            except Exception:
                pass
        
        # Fallback implementation
        if len(close) < period:
            return [-50.0] * len(close)
        
        williams_values = []
        
        for i in range(len(close)):
            if i < period - 1:
                williams_values.append(-50.0)
            else:
                period_high = max(high[i-period+1:i+1])
                period_low = min(low[i-period+1:i+1])
                
                if period_high == period_low:
                    williams_values.append(-50.0)
                else:
                    wr = -100 * (period_high - close[i]) / (period_high - period_low)
                    williams_values.append(wr)
        
        return williams_values
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, 
                       std_dev: float = 2.0) -> Tuple[List[float], List[float], List[float]]:
        """Calculate Bollinger Bands with fallback implementation"""
        
        if TALIB_AVAILABLE:
            try:
                upper, middle, lower = talib.BBANDS(np.array(prices), 
                                                   timeperiod=period, 
                                                   nbdevup=std_dev, 
                                                   nbdevdn=std_dev)
                return upper.tolist(), middle.tolist(), lower.tolist()
            except Exception:
                pass
        
        # Fallback implementation
        if len(prices) < period:
            return prices.copy(), prices.copy(), prices.copy()
        
        upper_band = []
        middle_band = []
        lower_band = []
        
        for i in range(len(prices)):
            if i < period - 1:
                upper_band.append(prices[i])
                middle_band.append(prices[i])
                lower_band.append(prices[i])
            else:
                period_prices = prices[i-period+1:i+1]
                sma = np.mean(period_prices)
                std = np.std(period_prices)
                
                upper_band.append(sma + std_dev * std)
                middle_band.append(sma)
                lower_band.append(sma - std_dev * std)
        
        return upper_band, middle_band, lower_band

class MomentumAnalyzer:
    """Momentum analysis for futures trading"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def analyze(self, futures_data: FuturesData) -> MomentumResult:
        """Perform comprehensive momentum analysis"""
        
        try:
            # Calculate technical indicators
            rsi_values = self.indicators.rsi(futures_data.close)
            macd_values, macd_signal = self.indicators.macd(futures_data.close)
            stochastic_k, stochastic_d = self.indicators.stochastic(
                futures_data.high, futures_data.low, futures_data.close
            )
            williams_r = self.indicators.williams_r(
                futures_data.high, futures_data.low, futures_data.close
            )
            
            # Calculate composite momentum scores
            momentum_scores = self._calculate_momentum_scores(
                rsi_values, macd_values, macd_signal, stochastic_k, williams_r
            )
            
            # Generate trading signals
            momentum_signals = self._generate_momentum_signals(momentum_scores)
            
            # Calculate momentum strength
            momentum_strength = self._calculate_momentum_strength(
                momentum_scores, futures_data.volume
            )
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(
                futures_data.close, macd_values
            )
            
            # Detect momentum divergence
            momentum_divergence = self._detect_momentum_divergence(
                futures_data.close, rsi_values
            )
            
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
            
        except Exception as e:
            print(f"Error in momentum analysis: {e}")
            # Return default results
            n = len(futures_data.close)
            return MomentumResult(
                momentum_scores=[0.0] * n,
                momentum_signals=["HOLD"] * n,
                momentum_strength=[0.5] * n,
                trend_direction=["SIDEWAYS"] * n,
                momentum_divergence=[False] * n,
                rsi_values=[50.0] * n,
                macd_values=[0.0] * n,
                macd_signal=[0.0] * n,
                stochastic_k=[50.0] * n,
                stochastic_d=[50.0] * n,
                williams_r=[-50.0] * n
            )
    
    def _calculate_momentum_scores(self, rsi_values: List[float], 
                                 macd_values: List[float], 
                                 macd_signal: List[float],
                                 stochastic_k: List[float], 
                                 williams_r: List[float]) -> List[float]:
        """Calculate composite momentum scores"""
        
        momentum_scores = []
        
        for i in range(len(rsi_values)):
            # RSI component (normalized to -1 to 1)
            rsi_score = (rsi_values[i] - 50) / 50
            
            # MACD component
            macd_score = 1 if macd_values[i] > macd_signal[i] else -1
            
            # Stochastic component (normalized to -1 to 1)
            stoch_score = (stochastic_k[i] - 50) / 50
            
            # Williams %R component (normalized to -1 to 1)
            williams_score = (williams_r[i] + 50) / 50
            
            # Composite score (weighted average)
            composite_score = (
                0.3 * rsi_score + 
                0.3 * macd_score + 
                0.2 * stoch_score + 
                0.2 * williams_score
            )
            
            momentum_scores.append(np.clip(composite_score, -1, 1))
        
        return momentum_scores
    
    def _generate_momentum_signals(self, momentum_scores: List[float]) -> List[str]:
        """Generate trading signals based on momentum scores"""
        
        signals = []
        
        for score in momentum_scores:
            if score > 0.5:
                signals.append("STRONG_BUY")
            elif score > 0.2:
                signals.append("BUY")
            elif score < -0.5:
                signals.append("STRONG_SELL")
            elif score < -0.2:
                signals.append("SELL")
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
        volume_ratios = [v / avg_volume for v in volume]
        
        strength = []
        for i, score in enumerate(momentum_scores):
            # Combine momentum score with volume confirmation
            base_strength = abs(score)
            volume_factor = min(volume_ratios[i], 2.0)  # Cap at 2x average
            
            # Volume confirmation boosts strength
            adjusted_strength = base_strength * (0.7 + 0.3 * volume_factor)
            strength.append(min(adjusted_strength, 1.0))
        
        return strength
    
    def _determine_trend_direction(self, prices: List[float], 
                                 macd_values: List[float]) -> List[str]:
        """Determine trend direction"""
        
        directions = []
        
        for i in range(len(prices)):
            if i < 10:  # Need some history
                directions.append("SIDEWAYS")
                continue
            
            # Price trend (last 10 periods)
            recent_prices = prices[i-9:i+1]
            price_trend = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
            
            # MACD confirmation
            macd_positive = macd_values[i] > 0
            
            if price_trend > 0.02 and macd_positive:
                directions.append("UPTREND")
            elif price_trend < -0.02 and not macd_positive:
                directions.append("DOWNTREND")
            else:
                directions.append("SIDEWAYS")
        
        return directions
    
    def _detect_momentum_divergence(self, prices: List[float], 
                                  rsi_values: List[float]) -> List[bool]:
        """Detect momentum divergence"""
        
        divergences = []
        
        for i in range(len(prices)):
            if i < 20:  # Need sufficient history
                divergences.append(False)
                continue
            
            # Look for divergence in last 10 periods
            recent_prices = prices[i-9:i+1]
            recent_rsi = rsi_values[i-9:i+1]
            
            # Price direction
            price_direction = recent_prices[-1] - recent_prices[0]
            
            # RSI direction
            rsi_direction = recent_rsi[-1] - recent_rsi[0]
            
            # Divergence occurs when price and RSI move in opposite directions
            divergence = (
                (price_direction > 0 and rsi_direction < 0) or
                (price_direction < 0 and rsi_direction > 0)
            )
            
            divergences.append(divergence)
        
        return divergences

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    import random
    from datetime import timedelta
    
    n_periods = 100
    base_price = 100.0
    
    # Generate sample price data
    prices = [base_price]
    for _ in range(n_periods - 1):
        change = random.uniform(-0.02, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    # Generate OHLC data
    high = [p * (1 + abs(random.uniform(0, 0.01))) for p in prices]
    low = [p * (1 - abs(random.uniform(0, 0.01))) for p in prices]
    open_prices = [prices[0]] + prices[:-1]
    
    # Generate other data
    volume = [random.uniform(1000, 10000) for _ in range(n_periods)]
    open_interest = [random.uniform(5000, 15000) for _ in range(n_periods)]
    returns = [(prices[i] - prices[i-1]) / prices[i-1] if i > 0 else 0 
              for i in range(n_periods)]
    
    timestamps = [datetime.now() + timedelta(days=i) for i in range(n_periods)]
    
    # Create FuturesData
    futures_data = FuturesData(
        prices=prices,
        returns=returns,
        volume=volume,
        open_interest=open_interest,
        timestamps=timestamps,
        high=high,
        low=low,
        open=open_prices,
        close=prices,
        contract_symbol="TEST_2024_03",
        underlying_asset="Test Asset"
    )
    
    # Test momentum analysis
    analyzer = MomentumAnalyzer()
    results = analyzer.analyze(futures_data)
    
    print(f"Momentum Analysis Results:")
    print(f"Current RSI: {results.rsi_values[-1]:.2f}")
    print(f"Current Momentum Score: {results.momentum_scores[-1]:.3f}")
    print(f"Current Signal: {results.momentum_signals[-1]}")
    print(f"Current Trend: {results.trend_direction[-1]}")