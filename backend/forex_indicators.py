from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import math
from enum import Enum

# Import modularized forex models
from indicators.forex.balance_of_payments_model import BalanceOfPaymentsModel
from indicators.forex.monetary_model import MonetaryModel
from indicators.forex.advanced_forex_ml import AdvancedForexML

class ForexIndicatorType(Enum):
    """Forex-specific indicator types"""
    PPP = "ppp"  # Purchasing Power Parity
    IRP = "irp"  # Interest Rate Parity
    UIP = "uip"  # Uncovered Interest Parity
    BALANCE_OF_PAYMENTS = "bop"  # Balance of Payments Model
    MONETARY_MODEL = "monetary"  # Monetary Model
    ARIMA = "arima"  # AutoRegressive Integrated Moving Average
    GARCH = "garch"  # Generalized AutoRegressive Conditional Heteroskedasticity
    EGARCH = "egarch"  # Exponential GARCH
    KALMAN_FILTER = "kalman_filter"  # Kalman Filter
    LSTM = "lstm"  # Long Short-Term Memory
    XGBOOST = "xgboost"  # XGBoost
    RL_AGENTS = "rl_agents"  # Reinforcement Learning Agents
    FOREXBERT = "forexbert"  # ForexBERT Sentiment Analysis
    CARRY_TRADE = "carry_trade"  # Carry Trade Strategy
    MOMENTUM = "momentum"  # Momentum Strategy
    MEAN_REVERSION = "mean_reversion"  # Mean Reversion Strategy
    VOLATILITY_BREAKOUT = "volatility_breakout"  # Volatility Breakout
    CORRELATION_ANALYSIS = "correlation"  # Cross-Currency Correlation
    CENTRAL_BANK_POLICY = "cb_policy"  # Central Bank Policy Impact

@dataclass
class EconomicData:
    """Economic indicators for forex analysis"""
    gdp_growth: float
    inflation_rate: float
    unemployment_rate: float
    interest_rate: float
    money_supply_growth: float
    government_debt_to_gdp: float
    current_account_balance: float
    trade_balance: float
    foreign_reserves: float
    political_stability_index: float

@dataclass
class CurrencyPair:
    """Currency pair information"""
    base_currency: str
    quote_currency: str
    current_rate: float
    base_economic_data: EconomicData
    quote_economic_data: EconomicData
    historical_rates: List[float]
    volatility: float

@dataclass
class ForexIndicatorResult:
    """Result of forex indicator calculation"""
    indicator_type: ForexIndicatorType
    value: float
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str
    signal: str  # "BUY", "SELL", "HOLD"

class PurchasingPowerParity:
    """Purchasing Power Parity Model"""
    
    @staticmethod
    def calculate(currency_pair: CurrencyPair) -> ForexIndicatorResult:
        """Calculate PPP fair value"""
        try:
            # Relative PPP calculation
            base_inflation = currency_pair.base_economic_data.inflation_rate
            quote_inflation = currency_pair.quote_economic_data.inflation_rate
            
            # Calculate expected exchange rate change based on inflation differential
            inflation_differential = base_inflation - quote_inflation
            
            # PPP suggests currency should appreciate/depreciate based on inflation differential
            ppp_adjustment = -inflation_differential  # Negative because higher inflation weakens currency
            
            # Calculate fair value
            current_rate = currency_pair.current_rate
            ppp_fair_value = current_rate * (1 + ppp_adjustment)
            
            # Determine over/under valuation
            valuation_gap = (current_rate - ppp_fair_value) / ppp_fair_value
            
            # Generate signal
            if valuation_gap > 0.05:  # Overvalued
                signal = "SELL"
            elif valuation_gap < -0.05:  # Undervalued
                signal = "BUY"
            else:
                signal = "HOLD"
            
            confidence = min(0.8, max(0.4, 1 - abs(valuation_gap) * 2))
            risk_level = "Low" if abs(valuation_gap) < 0.1 else "Medium" if abs(valuation_gap) < 0.2 else "High"
            
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.PPP,
                value=ppp_fair_value,
                confidence=confidence,
                metadata={
                    "current_rate": current_rate,
                    "inflation_differential": inflation_differential,
                    "valuation_gap": valuation_gap,
                    "base_inflation": base_inflation,
                    "quote_inflation": quote_inflation
                },
                timestamp=datetime.now(),
                interpretation=f"PPP fair value: {ppp_fair_value:.4f} (Gap: {valuation_gap:.2%})",
                risk_level=risk_level,
                signal=signal
            )
        except Exception as e:
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.PPP,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="PPP calculation failed",
                risk_level="High",
                signal="HOLD"
            )

class InterestRateParity:
    """Interest Rate Parity Model"""
    
    @staticmethod
    def calculate(currency_pair: CurrencyPair, forward_rate: Optional[float] = None) -> ForexIndicatorResult:
        """Calculate IRP and detect arbitrage opportunities"""
        try:
            base_rate = currency_pair.base_economic_data.interest_rate
            quote_rate = currency_pair.quote_economic_data.interest_rate
            spot_rate = currency_pair.current_rate
            
            # Calculate theoretical forward rate using IRP
            # F = S * (1 + r_quote) / (1 + r_base)
            time_period = 1  # 1 year
            theoretical_forward = spot_rate * ((1 + quote_rate) / (1 + base_rate))
            
            if forward_rate:
                # Check for arbitrage opportunity
                arbitrage_profit = (forward_rate - theoretical_forward) / theoretical_forward
                
                if abs(arbitrage_profit) > 0.01:  # 1% threshold
                    signal = "BUY" if arbitrage_profit > 0 else "SELL"
                else:
                    signal = "HOLD"
                
                confidence = min(0.9, max(0.5, 1 - abs(arbitrage_profit) * 5))
            else:
                # Use theoretical forward as prediction
                forward_premium = (theoretical_forward - spot_rate) / spot_rate
                signal = "BUY" if forward_premium > 0.02 else "SELL" if forward_premium < -0.02 else "HOLD"
                confidence = 0.7
                arbitrage_profit = 0
            
            risk_level = "Low" if abs(base_rate - quote_rate) < 0.02 else "Medium" if abs(base_rate - quote_rate) < 0.05 else "High"
            
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.IRP,
                value=theoretical_forward,
                confidence=confidence,
                metadata={
                    "spot_rate": spot_rate,
                    "base_rate": base_rate,
                    "quote_rate": quote_rate,
                    "rate_differential": base_rate - quote_rate,
                    "arbitrage_profit": arbitrage_profit,
                    "actual_forward": forward_rate
                },
                timestamp=datetime.now(),
                interpretation=f"IRP theoretical forward: {theoretical_forward:.4f}",
                risk_level=risk_level,
                signal=signal
            )
        except Exception as e:
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.IRP,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="IRP calculation failed",
                risk_level="High",
                signal="HOLD"
            )

class UncoveredInterestParity:
    """Uncovered Interest Parity Model"""
    
    @staticmethod
    def calculate(currency_pair: CurrencyPair) -> ForexIndicatorResult:
        """Calculate UIP expected exchange rate change"""
        try:
            base_rate = currency_pair.base_economic_data.interest_rate
            quote_rate = currency_pair.quote_economic_data.interest_rate
            current_rate = currency_pair.current_rate
            
            # UIP suggests that interest rate differential equals expected depreciation
            # E[ΔS] = r_base - r_quote
            expected_change = base_rate - quote_rate
            
            # Calculate expected future spot rate
            expected_future_rate = current_rate * (1 + expected_change)
            
            # Carry trade opportunity
            carry_return = expected_change
            
            # Generate signal based on carry trade potential
            if carry_return > 0.02:  # 2% threshold
                signal = "BUY"  # Buy high-yielding currency
            elif carry_return < -0.02:
                signal = "SELL"  # Sell low-yielding currency
            else:
                signal = "HOLD"
            
            # Adjust confidence based on volatility (UIP often fails in short term)
            volatility_adjustment = min(0.5, currency_pair.volatility)
            confidence = max(0.3, 0.8 - volatility_adjustment)
            
            risk_level = "Low" if currency_pair.volatility < 0.1 else "Medium" if currency_pair.volatility < 0.2 else "High"
            
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.UIP,
                value=expected_future_rate,
                confidence=confidence,
                metadata={
                    "current_rate": current_rate,
                    "base_rate": base_rate,
                    "quote_rate": quote_rate,
                    "expected_change": expected_change,
                    "carry_return": carry_return,
                    "volatility": currency_pair.volatility
                },
                timestamp=datetime.now(),
                interpretation=f"UIP expected rate: {expected_future_rate:.4f} (Change: {expected_change:.2%})",
                risk_level=risk_level,
                signal=signal
            )
        except Exception as e:
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.UIP,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="UIP calculation failed",
                risk_level="High",
                signal="HOLD"
            )

# BalanceOfPaymentsModel is now imported from indicators.forex.balance_of_payments_model

# MonetaryModel class has been extracted to indicators/forex/monetary_model.py
# Import: from indicators.forex.monetary_model import MonetaryModel

# AdvancedForexML class has been extracted to indicators/forex/advanced_forex_ml.py
# Import: from indicators.forex.advanced_forex_ml import AdvancedForexML

class ForexIndicatorEngine:
    """Main engine for calculating forex indicators"""
    
    def __init__(self):
        self.ppp_model = PurchasingPowerParity()
        self.irp_model = InterestRateParity()
        self.uip_model = UncoveredInterestParity()
        self.bop_model = BalanceOfPaymentsModel()
        self.monetary_model = MonetaryModel()
        self.ml_models = AdvancedForexML()
    
    def calculate_all_indicators(self, currency_pair: CurrencyPair, 
                               forward_rate: Optional[float] = None) -> Dict[ForexIndicatorType, ForexIndicatorResult]:
        """Calculate all forex indicators"""
        results = {}
        
        # Fundamental models
        results[ForexIndicatorType.PPP] = self.ppp_model.calculate(currency_pair)
        results[ForexIndicatorType.IRP] = self.irp_model.calculate(currency_pair, forward_rate)
        results[ForexIndicatorType.UIP] = self.uip_model.calculate(currency_pair)
        results[ForexIndicatorType.BALANCE_OF_PAYMENTS] = self.bop_model.calculate(currency_pair)
        results[ForexIndicatorType.MONETARY_MODEL] = self.monetary_model.calculate(currency_pair)
        
        # ML models
        results[ForexIndicatorType.LSTM] = self.ml_models.lstm_prediction(currency_pair)
        results[ForexIndicatorType.RL_AGENTS] = self.ml_models.reinforcement_learning_agent(currency_pair)
        
        # Technical indicators
        results[ForexIndicatorType.CARRY_TRADE] = self._calculate_carry_trade(currency_pair)
        results[ForexIndicatorType.MOMENTUM] = self._calculate_momentum(currency_pair)
        results[ForexIndicatorType.MEAN_REVERSION] = self._calculate_mean_reversion(currency_pair)
        
        return results
    
    def _calculate_carry_trade(self, currency_pair: CurrencyPair) -> ForexIndicatorResult:
        """Calculate carry trade opportunity"""
        try:
            base_rate = currency_pair.base_economic_data.interest_rate
            quote_rate = currency_pair.quote_economic_data.interest_rate
            
            carry_return = base_rate - quote_rate
            
            # Adjust for volatility risk
            risk_adjusted_return = carry_return - (currency_pair.volatility * 2)
            
            signal = "BUY" if risk_adjusted_return > 0.02 else "SELL" if risk_adjusted_return < -0.02 else "HOLD"
            confidence = min(0.8, max(0.3, 1 - currency_pair.volatility * 2))
            risk_level = "Low" if currency_pair.volatility < 0.1 else "Medium" if currency_pair.volatility < 0.2 else "High"
            
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.CARRY_TRADE,
                value=risk_adjusted_return,
                confidence=confidence,
                metadata={
                    "carry_return": carry_return,
                    "volatility_penalty": currency_pair.volatility * 2,
                    "base_rate": base_rate,
                    "quote_rate": quote_rate
                },
                timestamp=datetime.now(),
                interpretation=f"Carry trade return: {risk_adjusted_return:.2%}",
                risk_level=risk_level,
                signal=signal
            )
        except Exception as e:
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.CARRY_TRADE,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Carry trade calculation failed",
                risk_level="High",
                signal="HOLD"
            )
    
    def _calculate_momentum(self, currency_pair: CurrencyPair) -> ForexIndicatorResult:
        """Calculate momentum indicator"""
        try:
            historical_rates = currency_pair.historical_rates
            if len(historical_rates) < 20:
                raise ValueError("Insufficient data for momentum calculation")
            
            # Calculate momentum over different periods
            current_rate = currency_pair.current_rate
            rate_1w = historical_rates[-5] if len(historical_rates) >= 5 else current_rate
            rate_1m = historical_rates[-20] if len(historical_rates) >= 20 else current_rate
            
            momentum_1w = (current_rate - rate_1w) / rate_1w
            momentum_1m = (current_rate - rate_1m) / rate_1m
            
            # Combined momentum score
            momentum_score = momentum_1w * 0.6 + momentum_1m * 0.4
            
            signal = "BUY" if momentum_score > 0.02 else "SELL" if momentum_score < -0.02 else "HOLD"
            confidence = min(0.7, max(0.3, 1 - currency_pair.volatility))
            risk_level = "Medium"  # Momentum strategies are inherently risky
            
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.MOMENTUM,
                value=momentum_score,
                confidence=confidence,
                metadata={
                    "momentum_1w": momentum_1w,
                    "momentum_1m": momentum_1m,
                    "current_rate": current_rate,
                    "rate_1w": rate_1w,
                    "rate_1m": rate_1m
                },
                timestamp=datetime.now(),
                interpretation=f"Momentum score: {momentum_score:.2%}",
                risk_level=risk_level,
                signal=signal
            )
        except Exception as e:
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.MOMENTUM,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Momentum calculation failed",
                risk_level="High",
                signal="HOLD"
            )
    
    def _calculate_mean_reversion(self, currency_pair: CurrencyPair) -> ForexIndicatorResult:
        """Calculate mean reversion indicator"""
        try:
            historical_rates = currency_pair.historical_rates
            if len(historical_rates) < 50:
                raise ValueError("Insufficient data for mean reversion calculation")
            
            # Calculate long-term mean
            long_term_mean = np.mean(historical_rates)
            current_rate = currency_pair.current_rate
            
            # Deviation from mean
            deviation = (current_rate - long_term_mean) / long_term_mean
            
            # Z-score calculation
            std_dev = np.std(historical_rates)
            z_score = (current_rate - long_term_mean) / std_dev
            
            # Mean reversion signal (opposite to current deviation)
            if abs(z_score) > 2:  # Strong deviation
                signal = "SELL" if z_score > 0 else "BUY"
            elif abs(z_score) > 1:  # Moderate deviation
                signal = "SELL" if z_score > 0 else "BUY"
            else:
                signal = "HOLD"
            
            confidence = min(0.8, max(0.3, abs(z_score) / 3))
            risk_level = "Low" if abs(z_score) > 2 else "Medium" if abs(z_score) > 1 else "High"
            
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.MEAN_REVERSION,
                value=long_term_mean,
                confidence=confidence,
                metadata={
                    "current_rate": current_rate,
                    "long_term_mean": long_term_mean,
                    "deviation": deviation,
                    "z_score": z_score,
                    "std_dev": std_dev
                },
                timestamp=datetime.now(),
                interpretation=f"Mean reversion target: {long_term_mean:.4f} (Z-score: {z_score:.2f})",
                risk_level=risk_level,
                signal=signal
            )
        except Exception as e:
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.MEAN_REVERSION,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Mean reversion calculation failed",
                risk_level="High",
                signal="HOLD"
            )