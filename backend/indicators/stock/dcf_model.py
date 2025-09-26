"""DCF (Discounted Cash Flow) Model for Stock Valuation

This module implements the DCF valuation model for stocks including:
- WACC (Weighted Average Cost of Capital) calculation
- Future cash flow projections
- Terminal value calculation
- Present value discounting

Author: Assistant
Date: 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StockFundamentals:
    """Stock fundamental data"""
    revenue: float
    net_income: float
    free_cash_flow: float
    total_debt: float
    shareholders_equity: float
    shares_outstanding: float
    dividend_per_share: float
    earnings_per_share: float
    book_value_per_share: float
    revenue_growth_rate: float
    earnings_growth_rate: float
    dividend_growth_rate: float
    beta: float
    market_cap: float

@dataclass
class MarketData:
    """Market and economic data"""
    risk_free_rate: float
    market_return: float
    inflation_rate: float
    gdp_growth: float
    unemployment_rate: float
    sector_performance: Dict[str, float]
    market_volatility: float

@dataclass
class DCFResult:
    """Result of DCF calculation"""
    intrinsic_value_per_share: float
    enterprise_value: float
    equity_value: float
    terminal_value: float
    wacc: float
    projected_cash_flows: List[float]
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str

class DCFModel:
    """Discounted Cash Flow Model"""
    
    def __init__(self):
        self.default_projection_years = 5
        self.default_terminal_growth_rate = 0.025
        self.default_tax_rate = 0.25
        self.default_cost_of_debt = 0.05
    
    def calculate(self, fundamentals: StockFundamentals, market_data: MarketData, 
                 projection_years: int = None, terminal_growth_rate: float = None) -> DCFResult:
        """Calculate DCF valuation"""
        try:
            projection_years = projection_years or self.default_projection_years
            terminal_growth_rate = terminal_growth_rate or self.default_terminal_growth_rate
            
            # Calculate WACC (Weighted Average Cost of Capital)
            wacc = self._calculate_wacc(fundamentals, market_data)
            
            # Project future cash flows
            projected_fcf, present_values = self._project_cash_flows(
                fundamentals, wacc, projection_years
            )
            
            # Calculate terminal value
            terminal_value = self._calculate_terminal_value(
                projected_fcf[-1], terminal_growth_rate, wacc
            )
            terminal_pv = terminal_value / ((1 + wacc) ** projection_years)
            
            # Enterprise value
            enterprise_value = sum(present_values) + terminal_pv
            
            # Equity value
            equity_value = enterprise_value - fundamentals.total_debt
            intrinsic_value_per_share = equity_value / fundamentals.shares_outstanding
            
            # Calculate confidence
            confidence = self._calculate_confidence(market_data, fundamentals)
            
            # Determine risk level
            risk_level = self._determine_risk_level(fundamentals, market_data)
            
            return DCFResult(
                intrinsic_value_per_share=intrinsic_value_per_share,
                enterprise_value=enterprise_value,
                equity_value=equity_value,
                terminal_value=terminal_value,
                wacc=wacc,
                projected_cash_flows=projected_fcf,
                confidence=confidence,
                metadata={
                    "projection_years": projection_years,
                    "terminal_growth_rate": terminal_growth_rate,
                    "present_values": present_values,
                    "terminal_pv": terminal_pv,
                    "debt_adjustment": fundamentals.total_debt
                },
                timestamp=datetime.now(),
                interpretation=f"DCF intrinsic value: ${intrinsic_value_per_share:.2f}",
                risk_level=risk_level
            )
        except Exception as e:
            return self._create_fallback_result(str(e))
    
    def _calculate_wacc(self, fundamentals: StockFundamentals, market_data: MarketData) -> float:
        """Calculate Weighted Average Cost of Capital"""
        # Cost of equity using CAPM
        cost_of_equity = market_data.risk_free_rate + fundamentals.beta * (
            market_data.market_return - market_data.risk_free_rate
        )
        
        # Capital structure ratios
        total_capital = fundamentals.total_debt + fundamentals.shareholders_equity
        debt_ratio = fundamentals.total_debt / total_capital if total_capital > 0 else 0
        equity_ratio = 1 - debt_ratio
        
        # WACC calculation
        wacc = (
            equity_ratio * cost_of_equity + 
            debt_ratio * self.default_cost_of_debt * (1 - self.default_tax_rate)
        )
        
        return wacc
    
    def _project_cash_flows(self, fundamentals: StockFundamentals, wacc: float, 
                           projection_years: int) -> tuple[List[float], List[float]]:
        """Project future cash flows and calculate present values"""
        projected_fcf = []
        present_values = []
        current_fcf = fundamentals.free_cash_flow
        
        for year in range(1, projection_years + 1):
            # Declining growth rate over time
            growth_rate = fundamentals.revenue_growth_rate * (0.9 ** (year - 1))
            
            # Project future FCF
            future_fcf = current_fcf * ((1 + growth_rate) ** year)
            projected_fcf.append(future_fcf)
            
            # Calculate present value
            present_value = future_fcf / ((1 + wacc) ** year)
            present_values.append(present_value)
        
        return projected_fcf, present_values
    
    def _calculate_terminal_value(self, final_year_fcf: float, 
                                 terminal_growth_rate: float, wacc: float) -> float:
        """Calculate terminal value using Gordon Growth Model"""
        terminal_fcf = final_year_fcf * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (wacc - terminal_growth_rate)
        return terminal_value
    
    def _calculate_confidence(self, market_data: MarketData, 
                            fundamentals: StockFundamentals) -> float:
        """Calculate confidence score based on market conditions and fundamentals"""
        base_confidence = 0.7
        
        # Adjust for market volatility
        volatility_adjustment = min(0.25, market_data.market_volatility / 2)
        
        # Adjust for financial stability
        debt_ratio = fundamentals.total_debt / fundamentals.shareholders_equity
        stability_adjustment = min(0.15, debt_ratio / 4) if debt_ratio > 1 else 0
        
        # Adjust for growth consistency
        growth_consistency = 0.1 if fundamentals.revenue_growth_rate > 0 and fundamentals.earnings_growth_rate > 0 else -0.1
        
        confidence = base_confidence - volatility_adjustment - stability_adjustment + growth_consistency
        return min(0.95, max(0.3, confidence))
    
    def _determine_risk_level(self, fundamentals: StockFundamentals, 
                            market_data: MarketData) -> str:
        """Determine risk level based on various factors"""
        risk_factors = 0
        
        # High debt
        if fundamentals.total_debt / fundamentals.shareholders_equity > 1:
            risk_factors += 1
        
        # High market volatility
        if market_data.market_volatility > 0.3:
            risk_factors += 1
        
        # Negative growth
        if fundamentals.revenue_growth_rate < 0 or fundamentals.earnings_growth_rate < 0:
            risk_factors += 1
        
        # High beta
        if fundamentals.beta > 1.5:
            risk_factors += 1
        
        if risk_factors >= 3:
            return "High"
        elif risk_factors >= 1:
            return "Medium"
        else:
            return "Low"
    
    def _create_fallback_result(self, error_message: str) -> DCFResult:
        """Create fallback result when calculation fails"""
        return DCFResult(
            intrinsic_value_per_share=0.0,
            enterprise_value=0.0,
            equity_value=0.0,
            terminal_value=0.0,
            wacc=0.0,
            projected_cash_flows=[],
            confidence=0.0,
            metadata={"error": error_message},
            timestamp=datetime.now(),
            interpretation="DCF calculation failed",
            risk_level="High"
        )
    
    def analyze_sensitivity(self, fundamentals: StockFundamentals, market_data: MarketData,
                          wacc_range: tuple = (-0.02, 0.02), 
                          growth_range: tuple = (-0.01, 0.01)) -> Dict[str, Any]:
        """Perform sensitivity analysis on key DCF parameters"""
        base_result = self.calculate(fundamentals, market_data)
        
        sensitivity_results = {
            "base_value": base_result.intrinsic_value_per_share,
            "wacc_sensitivity": {},
            "growth_sensitivity": {}
        }
        
        # WACC sensitivity
        for wacc_adj in np.linspace(wacc_range[0], wacc_range[1], 5):
            # Temporarily adjust market return to affect WACC
            adjusted_market_data = MarketData(
                risk_free_rate=market_data.risk_free_rate,
                market_return=market_data.market_return + wacc_adj,
                inflation_rate=market_data.inflation_rate,
                gdp_growth=market_data.gdp_growth,
                unemployment_rate=market_data.unemployment_rate,
                sector_performance=market_data.sector_performance,
                market_volatility=market_data.market_volatility
            )
            result = self.calculate(fundamentals, adjusted_market_data)
            sensitivity_results["wacc_sensitivity"][f"{wacc_adj:.2%}"] = result.intrinsic_value_per_share
        
        # Growth sensitivity
        for growth_adj in np.linspace(growth_range[0], growth_range[1], 5):
            adjusted_fundamentals = StockFundamentals(
                revenue=fundamentals.revenue,
                net_income=fundamentals.net_income,
                free_cash_flow=fundamentals.free_cash_flow,
                total_debt=fundamentals.total_debt,
                shareholders_equity=fundamentals.shareholders_equity,
                shares_outstanding=fundamentals.shares_outstanding,
                dividend_per_share=fundamentals.dividend_per_share,
                earnings_per_share=fundamentals.earnings_per_share,
                book_value_per_share=fundamentals.book_value_per_share,
                revenue_growth_rate=fundamentals.revenue_growth_rate + growth_adj,
                earnings_growth_rate=fundamentals.earnings_growth_rate + growth_adj,
                dividend_growth_rate=fundamentals.dividend_growth_rate,
                beta=fundamentals.beta,
                market_cap=fundamentals.market_cap
            )
            result = self.calculate(adjusted_fundamentals, market_data)
            sensitivity_results["growth_sensitivity"][f"{growth_adj:.2%}"] = result.intrinsic_value_per_share
        
        return sensitivity_results

# Example usage
if __name__ == "__main__":
    # Sample data
    fundamentals = StockFundamentals(
        revenue=10000000000,  # $10B
        net_income=1000000000,  # $1B
        free_cash_flow=800000000,  # $800M
        total_debt=2000000000,  # $2B
        shareholders_equity=5000000000,  # $5B
        shares_outstanding=100000000,  # 100M shares
        dividend_per_share=2.0,
        earnings_per_share=10.0,
        book_value_per_share=50.0,
        revenue_growth_rate=0.08,  # 8%
        earnings_growth_rate=0.10,  # 10%
        dividend_growth_rate=0.05,  # 5%
        beta=1.2,
        market_cap=8000000000  # $8B
    )
    
    market_data = MarketData(
        risk_free_rate=0.03,  # 3%
        market_return=0.10,  # 10%
        inflation_rate=0.025,  # 2.5%
        gdp_growth=0.025,  # 2.5%
        unemployment_rate=0.05,  # 5%
        sector_performance={"Technology": 0.15, "Healthcare": 0.08},
        market_volatility=0.20  # 20%
    )
    
    # Calculate DCF
    dcf_model = DCFModel()
    result = dcf_model.calculate(fundamentals, market_data)
    
    print(f"DCF Analysis Results:")
    print(f"Intrinsic Value per Share: ${result.intrinsic_value_per_share:.2f}")
    print(f"Enterprise Value: ${result.enterprise_value:,.0f}")
    print(f"WACC: {result.wacc:.2%}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Interpretation: {result.interpretation}")
    
    # Sensitivity analysis
    sensitivity = dcf_model.analyze_sensitivity(fundamentals, market_data)
    print(f"\nSensitivity Analysis:")
    print(f"Base Value: ${sensitivity['base_value']:.2f}")
    print(f"WACC Sensitivity: {sensitivity['wacc_sensitivity']}")
    print(f"Growth Sensitivity: {sensitivity['growth_sensitivity']}")