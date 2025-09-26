"""Balance of Payments Model for Forex Analysis

This module implements the Balance of Payments approach to exchange rate determination.
The model analyzes currency strength based on current account, trade balance, and foreign reserves.

Author: Assistant
Date: 2024
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
from enum import Enum

class ForexIndicatorType(Enum):
    """Forex-specific indicator types"""
    BALANCE_OF_PAYMENTS = "bop"  # Balance of Payments Model

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

class BalanceOfPaymentsModel:
    """Balance of Payments Model for Exchange Rate Analysis
    
    The Balance of Payments approach suggests that exchange rates are determined by
    the supply and demand for currencies arising from international transactions.
    Key components:
    - Current Account: Trade balance, income flows, transfers
    - Capital Account: Foreign investment flows
    - Foreign Reserves: Central bank intervention capacity
    """
    
    @staticmethod
    def calculate(currency_pair: CurrencyPair) -> ForexIndicatorResult:
        """Calculate currency strength based on balance of payments
        
        Args:
            currency_pair: Currency pair with economic data
            
        Returns:
            ForexIndicatorResult with BOP analysis
        """
        try:
            base_data = currency_pair.base_economic_data
            quote_data = currency_pair.quote_economic_data
            
            # Current account analysis
            base_ca_strength = BalanceOfPaymentsModel._normalize_account_balance(
                base_data.current_account_balance
            )
            quote_ca_strength = BalanceOfPaymentsModel._normalize_account_balance(
                quote_data.current_account_balance
            )
            
            # Trade balance analysis
            base_trade_strength = BalanceOfPaymentsModel._normalize_account_balance(
                base_data.trade_balance
            )
            quote_trade_strength = BalanceOfPaymentsModel._normalize_account_balance(
                quote_data.trade_balance
            )
            
            # Foreign reserves analysis (normalized)
            base_reserves_strength = min(1, max(-1, base_data.foreign_reserves / 500000))
            quote_reserves_strength = min(1, max(-1, quote_data.foreign_reserves / 500000))
            
            # Political stability factor
            base_stability = base_data.political_stability_index
            quote_stability = quote_data.political_stability_index
            
            # Overall BOP score calculation
            base_bop_score = BalanceOfPaymentsModel._calculate_bop_score(
                base_ca_strength, base_trade_strength, base_reserves_strength, base_stability
            )
            quote_bop_score = BalanceOfPaymentsModel._calculate_bop_score(
                quote_ca_strength, quote_trade_strength, quote_reserves_strength, quote_stability
            )
            
            # Relative strength
            relative_strength = base_bop_score - quote_bop_score
            
            # Calculate expected exchange rate adjustment
            current_rate = currency_pair.current_rate
            bop_adjustment = relative_strength * 0.15  # 15% max adjustment
            bop_fair_value = current_rate * (1 + bop_adjustment)
            
            # Generate trading signal
            signal = BalanceOfPaymentsModel._generate_signal(relative_strength)
            
            # Calculate confidence and risk
            confidence = BalanceOfPaymentsModel._calculate_confidence(
                relative_strength, base_data, quote_data
            )
            risk_level = BalanceOfPaymentsModel._assess_risk_level(
                relative_strength, currency_pair.volatility
            )
            
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.BALANCE_OF_PAYMENTS,
                value=bop_fair_value,
                confidence=confidence,
                metadata={
                    "base_bop_score": base_bop_score,
                    "quote_bop_score": quote_bop_score,
                    "relative_strength": relative_strength,
                    "current_rate": current_rate,
                    "bop_adjustment": bop_adjustment,
                    "base_ca_strength": base_ca_strength,
                    "quote_ca_strength": quote_ca_strength,
                    "base_trade_strength": base_trade_strength,
                    "quote_trade_strength": quote_trade_strength,
                    "base_reserves_strength": base_reserves_strength,
                    "quote_reserves_strength": quote_reserves_strength
                },
                timestamp=datetime.now(),
                interpretation=f"BOP fair value: {bop_fair_value:.4f} (Strength: {relative_strength:.2f})",
                risk_level=risk_level,
                signal=signal
            )
        except Exception as e:
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.BALANCE_OF_PAYMENTS,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="BOP calculation failed",
                risk_level="High",
                signal="HOLD"
            )
    
    @staticmethod
    def _normalize_account_balance(balance: float) -> float:
        """Normalize account balance to -1 to 1 scale"""
        if balance == 0:
            return 0
        return balance / abs(balance) * min(1, abs(balance) / 100000)
    
    @staticmethod
    def _calculate_bop_score(ca_strength: float, trade_strength: float, 
                           reserves_strength: float, stability: float) -> float:
        """Calculate overall Balance of Payments score
        
        Weights:
        - Current Account: 35%
        - Trade Balance: 35% 
        - Foreign Reserves: 20%
        - Political Stability: 10%
        """
        return (
            ca_strength * 0.35 + 
            trade_strength * 0.35 + 
            reserves_strength * 0.20 + 
            (stability - 0.5) * 2 * 0.10  # Convert 0-1 to -1 to 1
        )
    
    @staticmethod
    def _generate_signal(relative_strength: float) -> str:
        """Generate trading signal based on relative BOP strength"""
        if relative_strength > 0.3:
            return "BUY"  # Base currency stronger
        elif relative_strength < -0.3:
            return "SELL"  # Base currency weaker
        else:
            return "HOLD"
    
    @staticmethod
    def _calculate_confidence(relative_strength: float, base_data: EconomicData, 
                            quote_data: EconomicData) -> float:
        """Calculate confidence in BOP analysis"""
        # Base confidence from strength magnitude
        strength_confidence = min(0.8, abs(relative_strength) * 2)
        
        # Adjust for data quality (political stability)
        stability_factor = (base_data.political_stability_index + 
                          quote_data.political_stability_index) / 2
        
        # Adjust for economic volatility
        volatility_factor = 1 - abs(base_data.gdp_growth - quote_data.gdp_growth) * 2
        volatility_factor = max(0.3, min(1.0, volatility_factor))
        
        return max(0.3, min(0.9, strength_confidence * stability_factor * volatility_factor))
    
    @staticmethod
    def _assess_risk_level(relative_strength: float, volatility: float) -> str:
        """Assess risk level based on strength and volatility"""
        if abs(relative_strength) < 0.2 or volatility > 0.25:
            return "High"
        elif abs(relative_strength) < 0.4 or volatility > 0.15:
            return "Medium"
        else:
            return "Low"
    
    @staticmethod
    def analyze_bop_components(currency_pair: CurrencyPair) -> Dict[str, Any]:
        """Detailed analysis of BOP components
        
        Returns:
            Dictionary with detailed BOP component analysis
        """
        base_data = currency_pair.base_economic_data
        quote_data = currency_pair.quote_economic_data
        
        analysis = {
            "current_account_analysis": {
                "base_balance": base_data.current_account_balance,
                "quote_balance": quote_data.current_account_balance,
                "relative_position": "surplus" if base_data.current_account_balance > quote_data.current_account_balance else "deficit",
                "strength_differential": base_data.current_account_balance - quote_data.current_account_balance
            },
            "trade_balance_analysis": {
                "base_balance": base_data.trade_balance,
                "quote_balance": quote_data.trade_balance,
                "relative_position": "surplus" if base_data.trade_balance > quote_data.trade_balance else "deficit",
                "strength_differential": base_data.trade_balance - quote_data.trade_balance
            },
            "reserves_analysis": {
                "base_reserves": base_data.foreign_reserves,
                "quote_reserves": quote_data.foreign_reserves,
                "relative_strength": "stronger" if base_data.foreign_reserves > quote_data.foreign_reserves else "weaker",
                "reserves_ratio": base_data.foreign_reserves / (quote_data.foreign_reserves + 1e-6)
            },
            "stability_analysis": {
                "base_stability": base_data.political_stability_index,
                "quote_stability": quote_data.political_stability_index,
                "relative_stability": "more stable" if base_data.political_stability_index > quote_data.political_stability_index else "less stable",
                "stability_differential": base_data.political_stability_index - quote_data.political_stability_index
            }
        }
        
        return analysis

# Example usage
if __name__ == "__main__":
    # Sample data for testing
    base_econ = EconomicData(
        gdp_growth=0.025,
        inflation_rate=0.02,
        unemployment_rate=0.05,
        interest_rate=0.03,
        money_supply_growth=0.05,
        government_debt_to_gdp=0.8,
        current_account_balance=50000,  # Surplus
        trade_balance=25000,  # Surplus
        foreign_reserves=750000,  # Strong reserves
        political_stability_index=0.85  # Stable
    )
    
    quote_econ = EconomicData(
        gdp_growth=0.02,
        inflation_rate=0.015,
        unemployment_rate=0.04,
        interest_rate=0.025,
        money_supply_growth=0.04,
        government_debt_to_gdp=0.7,
        current_account_balance=-20000,  # Deficit
        trade_balance=-10000,  # Deficit
        foreign_reserves=400000,  # Weaker reserves
        political_stability_index=0.75  # Less stable
    )
    
    currency_pair = CurrencyPair(
        base_currency="USD",
        quote_currency="EUR",
        current_rate=1.1000,
        base_economic_data=base_econ,
        quote_economic_data=quote_econ,
        historical_rates=[1.08, 1.09, 1.095, 1.10, 1.105],
        volatility=0.12
    )
    
    # Test BOP model
    bop_model = BalanceOfPaymentsModel()
    result = bop_model.calculate(currency_pair)
    
    print("Balance of Payments Analysis:")
    print(f"Signal: {result.signal}")
    print(f"Fair Value: {result.value:.4f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Interpretation: {result.interpretation}")
    
    # Detailed component analysis
    components = bop_model.analyze_bop_components(currency_pair)
    print("\nDetailed BOP Component Analysis:")
    for component, analysis in components.items():
        print(f"\n{component.replace('_', ' ').title()}:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")