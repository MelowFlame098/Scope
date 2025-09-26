from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from enum import Enum

# Import individual model classes
from indicators.index.macro_factor_model import MacroeconomicFactorModel, MacroeconomicData
from indicators.index.apt_model import ArbitragePricingTheory
from indicators.index.term_structure_model import TermStructureModel
from indicators.index.ml_models import AdvancedMLModels
from indicators.index.elliott_wave_model import ElliottWaveAnalysis
from indicators.index.volatility_regime_model import VolatilityRegimeSwitching

class IndexIndicatorType(Enum):
    """Index-specific indicator types"""
    MACROECONOMIC_FACTORS = "macro_factors"  # Macroeconomic Factor Models
    ARBITRAGE_PRICING_THEORY = "apt"  # Arbitrage Pricing Theory
    CAPM = "capm"  # Capital Asset Pricing Model
    DIVIDEND_DISCOUNT_MODEL = "ddm"  # Dividend Discount Model
    TERM_STRUCTURE = "term_structure"  # Term Structure Models
    ARIMA = "arima"  # ARIMA Models
    SARIMA = "sarima"  # Seasonal ARIMA
    VAR = "var"  # Vector Autoregression
    GARCH = "garch"  # GARCH Models
    EGARCH = "egarch"  # Exponential GARCH
    TGARCH = "tgarch"  # Threshold GARCH
    KALMAN_FILTER = "kalman_filter"  # Kalman Filters
    COINTEGRATION = "cointegration"  # Cointegration Analysis
    VECM = "vecm"  # Vector Error Correction Model
    LSTM = "lstm"  # Long Short-Term Memory
    GRU = "gru"  # Gated Recurrent Unit
    TRANSFORMER = "transformer"  # Transformer Models
    XGBOOST = "xgboost"  # XGBoost
    RANDOM_FOREST = "random_forest"  # Random Forest
    AUTOML = "automl"  # Automated Machine Learning
    SENTIMENT_ANALYSIS = "sentiment"  # Sentiment & News Analysis
    HYBRID_ARIMA_LSTM = "hybrid_arima_lstm"  # Hybrid Models
    ELLIOTT_WAVE = "elliott_wave"  # Elliott Wave Theory
    FIBONACCI = "fibonacci"  # Fibonacci Analysis
    MOMENTUM = "momentum"  # Momentum Strategies
    MEAN_REVERSION = "mean_reversion"  # Mean Reversion
    MARKOWITZ = "markowitz"  # Markowitz Portfolio Theory
    FACTOR_INVESTING = "factor_investing"  # Factor Investing
    VOLATILITY_REGIME = "volatility_regime"  # Volatility Regime Switching

@dataclass
class MacroeconomicData:
    """Macroeconomic indicators"""
    gdp_growth: float
    inflation_rate: float
    unemployment_rate: float
    interest_rate: float
    money_supply_growth: float
    government_debt_to_gdp: float
    trade_balance: float
    consumer_confidence: float
    business_confidence: float
    manufacturing_pmi: float
    services_pmi: float
    retail_sales_growth: float
    industrial_production: float
    housing_starts: float
    oil_price: float
    dollar_index: float
    vix: float

@dataclass
class IndexData:
    """Index information"""
    symbol: str
    name: str
    current_level: float
    historical_levels: List[float]
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
class IndexIndicatorResult:
    """Result of index indicator calculation"""
    indicator_type: IndexIndicatorType
    value: float
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str
    signal: str  # "BUY", "SELL", "HOLD"
    time_horizon: str

# MacroeconomicFactorModel is now imported from indicators.index.macro_factor_model

# ArbitragePricingTheory is now imported from indicators.index.apt_model

# TermStructureModel is now imported from indicators.index.term_structure_model

# AdvancedMLModels is now imported from indicators.index.ml_models

# ElliottWaveAnalysis is now imported from indicators.index.elliott_wave_model

# VolatilityRegimeSwitching is now imported from indicators.index.volatility_regime_model

class IndexIndicatorEngine:
    """Main engine for calculating index indicators"""
    
    def __init__(self):
        self.macro_model = MacroeconomicFactorModel()
        self.apt_model = ArbitragePricingTheory()
        self.term_structure = TermStructureModel()
        self.ml_models = AdvancedMLModels()
        self.elliott_wave = ElliottWaveAnalysis()
        self.volatility_regime = VolatilityRegimeSwitching()
    
    def calculate_all_indicators(self, index_data: IndexData, 
                               macro_data: MacroeconomicData,
                               yield_curve: Dict[str, float],
                               risk_factors: Dict[str, float]) -> Dict[IndexIndicatorType, IndexIndicatorResult]:
        """Calculate all index indicators"""
        results = {}
        
        # Fundamental models
        results[IndexIndicatorType.MACROECONOMIC_FACTORS] = self.macro_model.calculate(index_data, macro_data)
        results[IndexIndicatorType.ARBITRAGE_PRICING_THEORY] = self.apt_model.calculate(index_data, macro_data, risk_factors)
        results[IndexIndicatorType.TERM_STRUCTURE] = self.term_structure.calculate(index_data, yield_curve)
        
        # Machine learning models
        results[IndexIndicatorType.LSTM] = self.ml_models.lstm_prediction(index_data, macro_data)
        results[IndexIndicatorType.TRANSFORMER] = self.ml_models.transformer_prediction(index_data, macro_data)
        
        # Technical analysis
        results[IndexIndicatorType.ELLIOTT_WAVE] = self.elliott_wave.calculate(index_data)
        results[IndexIndicatorType.VOLATILITY_REGIME] = self.volatility_regime.calculate(index_data)
        
        # Additional indicators
        results[IndexIndicatorType.MOMENTUM] = self._calculate_momentum(index_data)
        results[IndexIndicatorType.MEAN_REVERSION] = self._calculate_mean_reversion(index_data)
        results[IndexIndicatorType.MARKOWITZ] = self._calculate_markowitz_efficiency(index_data, macro_data)
        
        return results
    
    def _calculate_momentum(self, index_data: IndexData) -> IndexIndicatorResult:
        """Calculate momentum indicator"""
        try:
            historical_levels = index_data.historical_levels
            if len(historical_levels) < 50:
                raise ValueError("Insufficient data for momentum calculation")
            
            current_level = index_data.current_level
            level_1m = historical_levels[-20] if len(historical_levels) >= 20 else current_level
            level_3m = historical_levels[-60] if len(historical_levels) >= 60 else current_level
            
            momentum_1m = (current_level - level_1m) / level_1m
            momentum_3m = (current_level - level_3m) / level_3m
            
            # Combined momentum score
            momentum_score = momentum_1m * 0.6 + momentum_3m * 0.4
            
            signal = "BUY" if momentum_score > 0.05 else "SELL" if momentum_score < -0.05 else "HOLD"
            confidence = min(0.7, max(0.3, 1 - index_data.volatility))
            risk_level = "Medium"  # Momentum strategies are inherently risky
            
            return IndexIndicatorResult(
                indicator_type=IndexIndicatorType.MOMENTUM,
                value=momentum_score,
                confidence=confidence,
                metadata={
                    "momentum_1m": momentum_1m,
                    "momentum_3m": momentum_3m,
                    "current_level": current_level,
                    "level_1m": level_1m,
                    "level_3m": level_3m
                },
                timestamp=datetime.now(),
                interpretation=f"Momentum score: {momentum_score:.2%}",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Short-term"
            )
        except Exception as e:
            return IndexIndicatorResult(
                indicator_type=IndexIndicatorType.MOMENTUM,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Momentum calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )
    
    def _calculate_mean_reversion(self, index_data: IndexData) -> IndexIndicatorResult:
        """Calculate mean reversion indicator"""
        try:
            historical_levels = index_data.historical_levels
            if len(historical_levels) < 100:
                raise ValueError("Insufficient data for mean reversion calculation")
            
            long_term_mean = np.mean(historical_levels)
            current_level = index_data.current_level
            
            deviation = (current_level - long_term_mean) / long_term_mean
            std_dev = np.std(historical_levels)
            z_score = (current_level - long_term_mean) / std_dev
            
            # Mean reversion signal
            if abs(z_score) > 2:
                signal = "SELL" if z_score > 0 else "BUY"
            elif abs(z_score) > 1:
                signal = "SELL" if z_score > 0 else "BUY"
            else:
                signal = "HOLD"
            
            confidence = min(0.8, max(0.3, abs(z_score) / 3))
            risk_level = "Low" if abs(z_score) > 2 else "Medium" if abs(z_score) > 1 else "High"
            
            return IndexIndicatorResult(
                indicator_type=IndexIndicatorType.MEAN_REVERSION,
                value=long_term_mean,
                confidence=confidence,
                metadata={
                    "current_level": current_level,
                    "long_term_mean": long_term_mean,
                    "deviation": deviation,
                    "z_score": z_score,
                    "std_dev": std_dev
                },
                timestamp=datetime.now(),
                interpretation=f"Mean reversion target: {long_term_mean:.0f} (Z-score: {z_score:.2f})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Long-term"
            )
        except Exception as e:
            return IndexIndicatorResult(
                indicator_type=IndexIndicatorType.MEAN_REVERSION,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Mean reversion calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )
    
    def _calculate_markowitz_efficiency(self, index_data: IndexData, macro_data: MacroeconomicData) -> IndexIndicatorResult:
        """Calculate Markowitz portfolio efficiency"""
        try:
            # Simplified Markowitz analysis for index
            expected_return = index_data.dividend_yield / 100 + 0.05  # Assume 5% capital appreciation
            volatility = index_data.volatility
            risk_free_rate = macro_data.interest_rate / 100
            
            # Sharpe ratio
            sharpe_ratio = (expected_return - risk_free_rate) / volatility
            
            # Efficiency score (simplified)
            efficiency_score = sharpe_ratio / 2  # Normalize
            
            # Compare to theoretical efficient frontier
            theoretical_sharpe = 0.5  # Benchmark Sharpe ratio
            efficiency_gap = sharpe_ratio - theoretical_sharpe
            
            signal = "BUY" if efficiency_gap > 0.1 else "SELL" if efficiency_gap < -0.1 else "HOLD"
            confidence = min(0.8, max(0.4, abs(efficiency_gap) * 2 + 0.4))
            risk_level = "Low" if volatility < 0.15 else "Medium" if volatility < 0.25 else "High"
            
            return IndexIndicatorResult(
                indicator_type=IndexIndicatorType.MARKOWITZ,
                value=efficiency_score,
                confidence=confidence,
                metadata={
                    "expected_return": expected_return,
                    "volatility": volatility,
                    "risk_free_rate": risk_free_rate,
                    "sharpe_ratio": sharpe_ratio,
                    "theoretical_sharpe": theoretical_sharpe,
                    "efficiency_gap": efficiency_gap
                },
                timestamp=datetime.now(),
                interpretation=f"Portfolio efficiency: {efficiency_score:.2f} (Sharpe: {sharpe_ratio:.2f})",
                risk_level=risk_level,
                signal=signal,
                time_horizon="Long-term"
            )
        except Exception as e:
            return IndexIndicatorResult(
                indicator_type=IndexIndicatorType.MARKOWITZ,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Markowitz efficiency calculation failed",
                risk_level="High",
                signal="HOLD",
                time_horizon="N/A"
            )