"""Volatility Regime Switching Model for Index Analysis

This module implements volatility regime detection and switching models
for index analysis, identifying different volatility states and their implications.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from datetime import datetime
from enum import Enum

class IndexIndicatorType(Enum):
    """Index-specific indicator types"""
    VOLATILITY_REGIME = "volatility_regime"

class VolatilityRegime(Enum):
    """Volatility regime types"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"

@dataclass
class IndexData:
    """Index information"""
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
class RegimeAnalysis:
    """Volatility regime analysis results"""
    current_regime: VolatilityRegime
    regime_probability: float
    regime_persistence: float
    expected_duration: int
    transition_probability: Dict[VolatilityRegime, float]

@dataclass
class VolatilityRegimeResult:
    """Result of volatility regime analysis"""
    regime_analysis: RegimeAnalysis
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str
    signal: str
    time_horizon: str

class VolatilityRegimeSwitching:
    """Volatility Regime Switching Model"""
    
    def __init__(self):
        # Volatility thresholds for regime classification
        self.regime_thresholds = {
            VolatilityRegime.LOW: (0.0, 0.12),      # 0-12% annualized
            VolatilityRegime.NORMAL: (0.12, 0.20),   # 12-20% annualized
            VolatilityRegime.HIGH: (0.20, 0.35),     # 20-35% annualized
            VolatilityRegime.EXTREME: (0.35, 1.0)    # 35%+ annualized
        }
        
        # Historical regime transition probabilities (simplified)
        self.transition_matrix = {
            VolatilityRegime.LOW: {
                VolatilityRegime.LOW: 0.85,
                VolatilityRegime.NORMAL: 0.12,
                VolatilityRegime.HIGH: 0.03,
                VolatilityRegime.EXTREME: 0.00
            },
            VolatilityRegime.NORMAL: {
                VolatilityRegime.LOW: 0.15,
                VolatilityRegime.NORMAL: 0.70,
                VolatilityRegime.HIGH: 0.13,
                VolatilityRegime.EXTREME: 0.02
            },
            VolatilityRegime.HIGH: {
                VolatilityRegime.LOW: 0.05,
                VolatilityRegime.NORMAL: 0.25,
                VolatilityRegime.HIGH: 0.60,
                VolatilityRegime.EXTREME: 0.10
            },
            VolatilityRegime.EXTREME: {
                VolatilityRegime.LOW: 0.02,
                VolatilityRegime.NORMAL: 0.08,
                VolatilityRegime.HIGH: 0.40,
                VolatilityRegime.EXTREME: 0.50
            }
        }
        
        # Expected regime durations (in trading days)
        self.expected_durations = {
            VolatilityRegime.LOW: 60,      # ~3 months
            VolatilityRegime.NORMAL: 45,   # ~2 months
            VolatilityRegime.HIGH: 30,     # ~1.5 months
            VolatilityRegime.EXTREME: 15   # ~3 weeks
        }
    
    def calculate(self, index_data: IndexData) -> VolatilityRegimeResult:
        """Analyze volatility regime and generate signals"""
        try:
            # Calculate volatility metrics
            volatility_metrics = self._calculate_volatility_metrics(index_data.historical_levels)
            
            # Identify current regime
            current_regime = self._identify_current_regime(volatility_metrics)
            
            # Calculate regime probabilities
            regime_probabilities = self._calculate_regime_probabilities(volatility_metrics)
            
            # Analyze regime persistence
            regime_persistence = self._analyze_regime_persistence(index_data.historical_levels, current_regime)
            
            # Calculate transition probabilities
            transition_probs = self.transition_matrix.get(current_regime, {})
            
            # Create regime analysis
            regime_analysis = RegimeAnalysis(
                current_regime=current_regime,
                regime_probability=regime_probabilities.get(current_regime, 0.0),
                regime_persistence=regime_persistence,
                expected_duration=self.expected_durations.get(current_regime, 30),
                transition_probability=transition_probs
            )
            
            # Generate signal based on regime
            signal = self._generate_regime_signal(regime_analysis, volatility_metrics)
            
            # Calculate confidence
            confidence = self._calculate_confidence(regime_analysis, volatility_metrics)
            
            # Assess risk level
            risk_level = self._assess_risk_level(regime_analysis)
            
            return VolatilityRegimeResult(
                regime_analysis=regime_analysis,
                confidence=confidence,
                metadata={
                    "volatility_metrics": volatility_metrics,
                    "regime_probabilities": regime_probabilities,
                    "current_volatility": index_data.volatility,
                    "regime_thresholds": self.regime_thresholds
                },
                timestamp=datetime.now(),
                interpretation=f"Volatility regime: {current_regime.value.upper()} (Prob: {regime_probabilities.get(current_regime, 0.0):.2f})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short to Medium-term"
            )
        except Exception as e:
            return VolatilityRegimeResult(
                regime_analysis=RegimeAnalysis(
                    current_regime=VolatilityRegime.NORMAL,
                    regime_probability=0.0,
                    regime_persistence=0.0,
                    expected_duration=30,
                    transition_probability={}
                ),
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Volatility regime analysis failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )
    
    def _calculate_volatility_metrics(self, price_data: List[float]) -> Dict[str, float]:
        """Calculate various volatility metrics"""
        if len(price_data) < 10:
            return {"short_term_vol": 0.0, "long_term_vol": 0.0, "vol_ratio": 1.0, "vol_zscore": 0.0}
        
        # Calculate returns
        returns = []
        for i in range(1, len(price_data)):
            ret = (price_data[i] - price_data[i-1]) / price_data[i-1]
            returns.append(ret)
        
        # Short-term volatility (last 10 periods)
        short_window = min(10, len(returns))
        short_term_returns = returns[-short_window:]
        short_term_vol = np.std(short_term_returns) * np.sqrt(252)  # Annualized
        
        # Long-term volatility (last 30 periods or all available)
        long_window = min(30, len(returns))
        long_term_returns = returns[-long_window:]
        long_term_vol = np.std(long_term_returns) * np.sqrt(252)  # Annualized
        
        # Volatility ratio
        vol_ratio = short_term_vol / long_term_vol if long_term_vol > 0 else 1.0
        
        # Volatility Z-score (how many standard deviations from long-term mean)
        if len(returns) >= 60:
            historical_vols = []
            for i in range(20, len(returns)):
                window_returns = returns[i-20:i]
                window_vol = np.std(window_returns) * np.sqrt(252)
                historical_vols.append(window_vol)
            
            if historical_vols:
                mean_vol = np.mean(historical_vols)
                std_vol = np.std(historical_vols)
                vol_zscore = (short_term_vol - mean_vol) / std_vol if std_vol > 0 else 0.0
            else:
                vol_zscore = 0.0
        else:
            vol_zscore = 0.0
        
        # GARCH-like volatility clustering measure
        volatility_clustering = self._calculate_volatility_clustering(returns)
        
        return {
            "short_term_vol": short_term_vol,
            "long_term_vol": long_term_vol,
            "vol_ratio": vol_ratio,
            "vol_zscore": vol_zscore,
            "volatility_clustering": volatility_clustering,
            "current_returns_std": np.std(short_term_returns) if short_term_returns else 0.0
        }
    
    def _calculate_volatility_clustering(self, returns: List[float]) -> float:
        """Calculate volatility clustering measure"""
        if len(returns) < 10:
            return 0.0
        
        # Calculate squared returns (proxy for volatility)
        squared_returns = [r**2 for r in returns]
        
        # Calculate autocorrelation of squared returns
        if len(squared_returns) >= 10:
            lag1_corr = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
            return lag1_corr if not np.isnan(lag1_corr) else 0.0
        
        return 0.0
    
    def _identify_current_regime(self, volatility_metrics: Dict[str, float]) -> VolatilityRegime:
        """Identify current volatility regime"""
        current_vol = volatility_metrics["short_term_vol"]
        
        for regime, (low_thresh, high_thresh) in self.regime_thresholds.items():
            if low_thresh <= current_vol < high_thresh:
                return regime
        
        return VolatilityRegime.EXTREME  # Default for very high volatility
    
    def _calculate_regime_probabilities(self, volatility_metrics: Dict[str, float]) -> Dict[VolatilityRegime, float]:
        """Calculate probabilities for each regime using Gaussian mixture approach"""
        current_vol = volatility_metrics["short_term_vol"]
        vol_zscore = volatility_metrics["vol_zscore"]
        
        # Simplified probability calculation based on distance from regime centers
        regime_centers = {
            VolatilityRegime.LOW: 0.08,
            VolatilityRegime.NORMAL: 0.16,
            VolatilityRegime.HIGH: 0.27,
            VolatilityRegime.EXTREME: 0.45
        }
        
        probabilities = {}
        total_prob = 0.0
        
        for regime, center in regime_centers.items():
            # Calculate probability based on distance from center
            distance = abs(current_vol - center)
            prob = np.exp(-distance * 10)  # Exponential decay
            probabilities[regime] = prob
            total_prob += prob
        
        # Normalize probabilities
        if total_prob > 0:
            for regime in probabilities:
                probabilities[regime] /= total_prob
        
        return probabilities
    
    def _analyze_regime_persistence(self, price_data: List[float], current_regime: VolatilityRegime) -> float:
        """Analyze how persistent the current regime has been"""
        if len(price_data) < 20:
            return 0.5  # Default moderate persistence
        
        # Calculate rolling volatility regimes
        window_size = 10
        regime_history = []
        
        for i in range(window_size, len(price_data)):
            window_data = price_data[i-window_size:i]
            window_metrics = self._calculate_volatility_metrics(window_data)
            window_regime = self._identify_current_regime(window_metrics)
            regime_history.append(window_regime)
        
        if not regime_history:
            return 0.5
        
        # Calculate persistence as fraction of recent periods in same regime
        recent_periods = min(10, len(regime_history))
        recent_regimes = regime_history[-recent_periods:]
        same_regime_count = sum(1 for regime in recent_regimes if regime == current_regime)
        
        persistence = same_regime_count / recent_periods
        return persistence
    
    def _generate_regime_signal(self, regime_analysis: RegimeAnalysis, volatility_metrics: Dict[str, float]) -> str:
        """Generate trading signal based on volatility regime"""
        current_regime = regime_analysis.current_regime
        vol_zscore = volatility_metrics["vol_zscore"]
        
        # Low volatility regime - generally bullish
        if current_regime == VolatilityRegime.LOW:
            if vol_zscore < -1.0:  # Extremely low volatility
                return "BUY"  # Complacency, but still bullish
            else:
                return "BUY"
        
        # Normal volatility regime - neutral to slightly bullish
        elif current_regime == VolatilityRegime.NORMAL:
            if vol_zscore > 0.5:
                return "HOLD"  # Rising volatility, be cautious
            else:
                return "BUY"
        
        # High volatility regime - bearish to neutral
        elif current_regime == VolatilityRegime.HIGH:
            if vol_zscore > 1.0:
                return "SELL"  # Stress building
            else:
                return "HOLD"  # Wait for clarity
        
        # Extreme volatility regime - contrarian opportunity
        elif current_regime == VolatilityRegime.EXTREME:
            if vol_zscore > 2.0:
                return "BUY"  # Extreme fear, contrarian opportunity
            else:
                return "SELL"  # Still dangerous
        
        return "HOLD"
    
    def _calculate_confidence(self, regime_analysis: RegimeAnalysis, volatility_metrics: Dict[str, float]) -> float:
        """Calculate confidence in regime analysis"""
        # Base confidence from regime probability
        regime_confidence = regime_analysis.regime_probability
        
        # Adjust for regime persistence
        persistence_bonus = regime_analysis.regime_persistence * 0.2
        
        # Adjust for data quality
        vol_ratio = volatility_metrics["vol_ratio"]
        data_quality = 1.0 - min(0.3, abs(vol_ratio - 1.0))  # Penalize extreme ratios
        
        # Combine factors
        total_confidence = (regime_confidence * 0.6) + (persistence_bonus) + (data_quality * 0.2)
        
        return max(0.2, min(0.9, total_confidence))
    
    def _assess_risk_level(self, regime_analysis: RegimeAnalysis) -> str:
        """Assess risk level based on volatility regime"""
        current_regime = regime_analysis.current_regime
        
        if current_regime == VolatilityRegime.EXTREME:
            return "High"
        elif current_regime == VolatilityRegime.HIGH:
            return "Medium"
        elif current_regime == VolatilityRegime.LOW and regime_analysis.regime_persistence > 0.8:
            return "Medium"  # Complacency risk
        else:
            return "Low"
    
    def forecast_regime_transition(self, index_data: IndexData, periods_ahead: int = 5) -> Dict[str, Any]:
        """Forecast regime transitions over specified periods"""
        current_result = self.calculate(index_data)
        current_regime = current_result.regime_analysis.current_regime
        
        # Simulate regime evolution using transition matrix
        regime_probabilities = {regime: 0.0 for regime in VolatilityRegime}
        regime_probabilities[current_regime] = 1.0
        
        forecast = {}
        
        for period in range(1, periods_ahead + 1):
            new_probabilities = {regime: 0.0 for regime in VolatilityRegime}
            
            # Apply transition matrix
            for from_regime, prob in regime_probabilities.items():
                if prob > 0 and from_regime in self.transition_matrix:
                    for to_regime, transition_prob in self.transition_matrix[from_regime].items():
                        new_probabilities[to_regime] += prob * transition_prob
            
            regime_probabilities = new_probabilities
            forecast[f"period_{period}"] = regime_probabilities.copy()
        
        return {
            "forecast": forecast,
            "most_likely_regime": max(regime_probabilities, key=regime_probabilities.get),
            "regime_stability": regime_probabilities[current_regime]
        }
    
    def analyze_regime_impact_on_returns(self, index_data: IndexData) -> Dict[str, float]:
        """Analyze historical returns by volatility regime"""
        if len(index_data.historical_levels) < 30:
            return {}
        
        # Calculate returns and regimes for historical data
        returns = []
        regimes = []
        
        window_size = 10
        for i in range(window_size, len(index_data.historical_levels)):
            # Calculate return
            ret = (index_data.historical_levels[i] - index_data.historical_levels[i-1]) / index_data.historical_levels[i-1]
            returns.append(ret)
            
            # Calculate regime for this period
            window_data = index_data.historical_levels[i-window_size:i]
            window_metrics = self._calculate_volatility_metrics(window_data)
            regime = self._identify_current_regime(window_metrics)
            regimes.append(regime)
        
        # Calculate average returns by regime
        regime_returns = {regime: [] for regime in VolatilityRegime}
        
        for ret, regime in zip(returns, regimes):
            regime_returns[regime].append(ret)
        
        # Calculate statistics
        regime_stats = {}
        for regime, regime_rets in regime_returns.items():
            if regime_rets:
                regime_stats[f"{regime.value}_mean_return"] = np.mean(regime_rets)
                regime_stats[f"{regime.value}_volatility"] = np.std(regime_rets)
                regime_stats[f"{regime.value}_sharpe"] = np.mean(regime_rets) / np.std(regime_rets) if np.std(regime_rets) > 0 else 0.0
                regime_stats[f"{regime.value}_count"] = len(regime_rets)
        
        return regime_stats

# Example usage
if __name__ == "__main__":
    # Sample data with varying volatility
    sample_index = IndexData(
        symbol="SPX",
        name="S&P 500",
        current_level=4200.0,
        historical_levels=[
            4000, 4020, 4010, 4030, 4050, 4040, 4060, 4080, 4070, 4090,
            4100, 4120, 4110, 4130, 4150, 4140, 4160, 4180, 4170, 4190,
            4200, 4180, 4160, 4140, 4120, 4100, 4080, 4100, 4120, 4140,
            4160, 4180, 4200, 4220, 4210, 4190, 4200, 4210, 4200, 4190, 4200
        ],
        dividend_yield=1.8,
        pe_ratio=22.5,
        pb_ratio=3.2,
        market_cap=35000000000000,  # $35T
        volatility=0.18,
        beta=1.0,
        sector_weights={"Technology": 0.28, "Healthcare": 0.13, "Financials": 0.11},
        constituent_count=500,
        volume=1000000000
    )
    
    # Create volatility regime analyzer
    vol_regime = VolatilityRegimeSwitching()
    result = vol_regime.calculate(sample_index)
    
    print(f"Current Regime: {result.regime_analysis.current_regime.value.upper()}")
    print(f"Regime Probability: {result.regime_analysis.regime_probability:.2f}")
    print(f"Regime Persistence: {result.regime_analysis.regime_persistence:.2f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Signal: {result.signal}")
    print(f"Risk Level: {result.risk_level}")
    
    # Forecast regime transitions
    forecast = vol_regime.forecast_regime_transition(sample_index, periods_ahead=5)
    print(f"\nMost Likely Future Regime: {forecast['most_likely_regime'].value.upper()}")
    print(f"Regime Stability: {forecast['regime_stability']:.2f}")