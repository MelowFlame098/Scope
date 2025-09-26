"""Monetary Model for Forex Analysis

This module implements the Monetary approach to exchange rate determination.
The model analyzes currency strength based on money supply, interest rates, and economic fundamentals.

Author: Assistant
Date: 2024
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
from enum import Enum
import math

class ForexIndicatorType(Enum):
    """Forex-specific indicator types"""
    MONETARY_MODEL = "monetary"  # Monetary Model

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

class MonetaryModel:
    """Monetary Model for Exchange Rate Analysis
    
    The Monetary approach suggests that exchange rates are determined by the relative
    supply and demand for money in different countries. Key principles:
    - Purchasing Power Parity (PPP) as long-run equilibrium
    - Money supply growth affects inflation and exchange rates
    - Interest rate differentials drive short-term movements
    - Economic growth affects money demand
    
    The model uses the flexible-price monetary model:
    S = (M1/M2) * (Y2/Y1) * exp((i2-i1)/100)
    
    Where:
    S = Exchange rate (base/quote)
    M = Money supply
    Y = Real income (GDP)
    i = Interest rate
    """
    
    @staticmethod
    def calculate(currency_pair: CurrencyPair) -> ForexIndicatorResult:
        """Calculate currency strength based on monetary model
        
        Args:
            currency_pair: Currency pair with economic data
            
        Returns:
            ForexIndicatorResult with monetary analysis
        """
        try:
            base_data = currency_pair.base_economic_data
            quote_data = currency_pair.quote_economic_data
            current_rate = currency_pair.current_rate
            
            # Calculate monetary fundamentals
            money_supply_ratio = MonetaryModel._calculate_money_supply_effect(
                base_data.money_supply_growth, quote_data.money_supply_growth
            )
            
            income_ratio = MonetaryModel._calculate_income_effect(
                base_data.gdp_growth, quote_data.gdp_growth
            )
            
            interest_differential = base_data.interest_rate - quote_data.interest_rate
            interest_effect = math.exp(interest_differential)
            
            # Inflation differential effect
            inflation_differential = base_data.inflation_rate - quote_data.inflation_rate
            inflation_effect = math.exp(-inflation_differential)  # Higher inflation weakens currency
            
            # Calculate theoretical exchange rate using monetary model
            monetary_rate = current_rate * money_supply_ratio * income_ratio * interest_effect * inflation_effect
            
            # Calculate relative monetary strength
            monetary_strength = MonetaryModel._calculate_monetary_strength(
                base_data, quote_data
            )
            
            # Adjust for economic stability and risk factors
            stability_adjustment = MonetaryModel._calculate_stability_adjustment(
                base_data, quote_data
            )
            
            # Final adjusted rate
            adjusted_rate = monetary_rate * stability_adjustment
            
            # Generate trading signal
            rate_deviation = (adjusted_rate - current_rate) / current_rate
            signal = MonetaryModel._generate_signal(rate_deviation)
            
            # Calculate confidence and risk
            confidence = MonetaryModel._calculate_confidence(
                rate_deviation, monetary_strength, base_data, quote_data
            )
            risk_level = MonetaryModel._assess_risk_level(
                rate_deviation, currency_pair.volatility, monetary_strength
            )
            
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.MONETARY_MODEL,
                value=adjusted_rate,
                confidence=confidence,
                metadata={
                    "current_rate": current_rate,
                    "monetary_rate": monetary_rate,
                    "money_supply_ratio": money_supply_ratio,
                    "income_ratio": income_ratio,
                    "interest_effect": interest_effect,
                    "inflation_effect": inflation_effect,
                    "interest_differential": interest_differential,
                    "inflation_differential": inflation_differential,
                    "monetary_strength": monetary_strength,
                    "stability_adjustment": stability_adjustment,
                    "rate_deviation": rate_deviation,
                    "base_money_growth": base_data.money_supply_growth,
                    "quote_money_growth": quote_data.money_supply_growth
                },
                timestamp=datetime.now(),
                interpretation=f"Monetary fair value: {adjusted_rate:.4f} (Deviation: {rate_deviation:.2%})",
                risk_level=risk_level,
                signal=signal
            )
        except Exception as e:
            return ForexIndicatorResult(
                indicator_type=ForexIndicatorType.MONETARY_MODEL,
                value=0.0,
                confidence=0.0,
                metadata={"error": str(e)},
                timestamp=datetime.now(),
                interpretation="Monetary model calculation failed",
                risk_level="High",
                signal="HOLD"
            )
    
    @staticmethod
    def _calculate_money_supply_effect(base_growth: float, quote_growth: float) -> float:
        """Calculate the effect of relative money supply growth
        
        Higher money supply growth typically weakens a currency
        """
        if quote_growth == 0:
            quote_growth = 0.001  # Avoid division by zero
        
        # Relative money supply growth effect
        growth_ratio = (1 + base_growth) / (1 + quote_growth)
        
        # Apply logarithmic scaling to moderate extreme values
        if growth_ratio > 1:
            return 1 + math.log(growth_ratio) * 0.5
        else:
            return 1 - math.log(1/growth_ratio) * 0.5
    
    @staticmethod
    def _calculate_income_effect(base_gdp_growth: float, quote_gdp_growth: float) -> float:
        """Calculate the effect of relative income (GDP) growth
        
        Higher GDP growth typically strengthens a currency through increased money demand
        """
        if base_gdp_growth <= 0:
            base_gdp_growth = 0.001
        if quote_gdp_growth <= 0:
            quote_gdp_growth = 0.001
        
        # Income effect - higher growth increases money demand, strengthening currency
        income_ratio = quote_gdp_growth / base_gdp_growth  # Inverted for exchange rate formula
        
        # Apply moderate scaling
        return max(0.5, min(2.0, income_ratio))
    
    @staticmethod
    def _calculate_monetary_strength(base_data: EconomicData, quote_data: EconomicData) -> float:
        """Calculate overall monetary strength score (-1 to 1)
        
        Considers:
        - Money supply growth (lower is better)
        - Interest rates (higher is better for currency strength)
        - Inflation (lower is better)
        - GDP growth (higher is better)
        """
        # Money supply score (lower growth is better)
        money_score = (quote_data.money_supply_growth - base_data.money_supply_growth) * 10
        money_score = max(-1, min(1, money_score))
        
        # Interest rate score (higher rates attract capital)
        interest_score = (base_data.interest_rate - quote_data.interest_rate) * 20
        interest_score = max(-1, min(1, interest_score))
        
        # Inflation score (lower inflation is better)
        inflation_score = (quote_data.inflation_rate - base_data.inflation_rate) * 25
        inflation_score = max(-1, min(1, inflation_score))
        
        # GDP growth score (higher growth is better)
        gdp_score = (base_data.gdp_growth - quote_data.gdp_growth) * 15
        gdp_score = max(-1, min(1, gdp_score))
        
        # Weighted average
        monetary_strength = (
            money_score * 0.25 +
            interest_score * 0.30 +
            inflation_score * 0.25 +
            gdp_score * 0.20
        )
        
        return max(-1, min(1, monetary_strength))
    
    @staticmethod
    def _calculate_stability_adjustment(base_data: EconomicData, quote_data: EconomicData) -> float:
        """Calculate adjustment factor based on economic stability
        
        Considers:
        - Political stability
        - Debt levels
        - Unemployment rates
        """
        # Political stability differential
        stability_diff = base_data.political_stability_index - quote_data.political_stability_index
        stability_factor = 1 + stability_diff * 0.1
        
        # Debt sustainability factor
        debt_diff = quote_data.government_debt_to_gdp - base_data.government_debt_to_gdp
        debt_factor = 1 + debt_diff * 0.05
        
        # Employment factor
        unemployment_diff = quote_data.unemployment_rate - base_data.unemployment_rate
        employment_factor = 1 + unemployment_diff * 0.5
        
        # Combined adjustment (moderate the impact)
        adjustment = stability_factor * debt_factor * employment_factor
        return max(0.8, min(1.2, adjustment))
    
    @staticmethod
    def _generate_signal(rate_deviation: float) -> str:
        """Generate trading signal based on rate deviation from fair value"""
        if rate_deviation > 0.05:  # Current rate 5% below fair value
            return "BUY"  # Currency undervalued
        elif rate_deviation < -0.05:  # Current rate 5% above fair value
            return "SELL"  # Currency overvalued
        else:
            return "HOLD"
    
    @staticmethod
    def _calculate_confidence(rate_deviation: float, monetary_strength: float,
                            base_data: EconomicData, quote_data: EconomicData) -> float:
        """Calculate confidence in monetary model analysis"""
        # Base confidence from deviation magnitude
        deviation_confidence = min(0.8, abs(rate_deviation) * 10)
        
        # Adjust for monetary strength clarity
        strength_confidence = min(0.9, abs(monetary_strength) + 0.3)
        
        # Adjust for data quality (stability and consistency)
        stability_avg = (base_data.political_stability_index + quote_data.political_stability_index) / 2
        
        # Economic consistency (lower volatility in key indicators)
        consistency_factor = 1 - abs(base_data.gdp_growth - quote_data.gdp_growth) * 5
        consistency_factor = max(0.5, min(1.0, consistency_factor))
        
        return max(0.3, min(0.9, deviation_confidence * strength_confidence * stability_avg * consistency_factor))
    
    @staticmethod
    def _assess_risk_level(rate_deviation: float, volatility: float, monetary_strength: float) -> str:
        """Assess risk level based on deviation, volatility, and strength"""
        risk_score = 0
        
        # Deviation risk
        if abs(rate_deviation) < 0.02:
            risk_score += 1
        elif abs(rate_deviation) > 0.10:
            risk_score += 3
        else:
            risk_score += 2
        
        # Volatility risk
        if volatility > 0.20:
            risk_score += 2
        elif volatility > 0.10:
            risk_score += 1
        
        # Strength clarity risk
        if abs(monetary_strength) < 0.2:
            risk_score += 1
        
        if risk_score <= 2:
            return "Low"
        elif risk_score <= 4:
            return "Medium"
        else:
            return "High"
    
    @staticmethod
    def analyze_monetary_components(currency_pair: CurrencyPair) -> Dict[str, Any]:
        """Detailed analysis of monetary model components
        
        Returns:
            Dictionary with detailed monetary component analysis
        """
        base_data = currency_pair.base_economic_data
        quote_data = currency_pair.quote_economic_data
        
        analysis = {
            "money_supply_analysis": {
                "base_growth": base_data.money_supply_growth,
                "quote_growth": quote_data.money_supply_growth,
                "growth_differential": base_data.money_supply_growth - quote_data.money_supply_growth,
                "impact": "weakening" if base_data.money_supply_growth > quote_data.money_supply_growth else "strengthening"
            },
            "interest_rate_analysis": {
                "base_rate": base_data.interest_rate,
                "quote_rate": quote_data.interest_rate,
                "differential": base_data.interest_rate - quote_data.interest_rate,
                "impact": "positive" if base_data.interest_rate > quote_data.interest_rate else "negative"
            },
            "inflation_analysis": {
                "base_inflation": base_data.inflation_rate,
                "quote_inflation": quote_data.inflation_rate,
                "differential": base_data.inflation_rate - quote_data.inflation_rate,
                "impact": "weakening" if base_data.inflation_rate > quote_data.inflation_rate else "strengthening"
            },
            "growth_analysis": {
                "base_gdp_growth": base_data.gdp_growth,
                "quote_gdp_growth": quote_data.gdp_growth,
                "growth_differential": base_data.gdp_growth - quote_data.gdp_growth,
                "impact": "positive" if base_data.gdp_growth > quote_data.gdp_growth else "negative"
            },
            "stability_factors": {
                "base_stability": base_data.political_stability_index,
                "quote_stability": quote_data.political_stability_index,
                "base_debt_ratio": base_data.government_debt_to_gdp,
                "quote_debt_ratio": quote_data.government_debt_to_gdp,
                "base_unemployment": base_data.unemployment_rate,
                "quote_unemployment": quote_data.unemployment_rate
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
        interest_rate=0.035,  # Higher interest rate
        money_supply_growth=0.04,  # Moderate money growth
        government_debt_to_gdp=0.8,
        current_account_balance=50000,
        trade_balance=25000,
        foreign_reserves=750000,
        political_stability_index=0.85
    )
    
    quote_econ = EconomicData(
        gdp_growth=0.02,
        inflation_rate=0.025,  # Higher inflation
        unemployment_rate=0.04,
        interest_rate=0.025,  # Lower interest rate
        money_supply_growth=0.06,  # Higher money growth
        government_debt_to_gdp=0.7,
        current_account_balance=-20000,
        trade_balance=-10000,
        foreign_reserves=400000,
        political_stability_index=0.75
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
    
    # Test Monetary model
    monetary_model = MonetaryModel()
    result = monetary_model.calculate(currency_pair)
    
    print("Monetary Model Analysis:")
    print(f"Signal: {result.signal}")
    print(f"Fair Value: {result.value:.4f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Interpretation: {result.interpretation}")
    
    # Detailed component analysis
    components = monetary_model.analyze_monetary_components(currency_pair)
    print("\nDetailed Monetary Component Analysis:")
    for component, analysis in components.items():
        print(f"\n{component.replace('_', ' ').title()}:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")