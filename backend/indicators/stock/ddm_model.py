"""DDM (Dividend Discount Model) for Stock Valuation

This module implements the Dividend Discount Model for stocks including:
- Gordon Growth Model (constant growth DDM)
- Multi-stage DDM for high growth scenarios
- Required return calculation using CAPM
- Dividend sustainability analysis

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
class DDMResult:
    """Result of DDM calculation"""
    intrinsic_value: float
    required_return: float
    dividend_yield: float
    payout_ratio: float
    dividend_sustainability_score: float
    model_type: str  # "gordon", "multi_stage", "zero_growth"
    projected_dividends: List[float]
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str
    risk_level: str

class DividendDiscountModel:
    """Dividend Discount Model"""
    
    def __init__(self):
        self.high_growth_years = 5
        self.stable_growth_rate = 0.03  # Long-term stable growth rate
        self.min_required_return = 0.05  # Minimum required return
    
    def calculate(self, fundamentals: StockFundamentals, market_data: MarketData) -> DDMResult:
        """Calculate DDM valuation"""
        try:
            # Calculate required return using CAPM
            required_return = self._calculate_required_return(fundamentals, market_data)
            
            # Check dividend sustainability
            sustainability_score = self._calculate_dividend_sustainability(fundamentals)
            
            # Determine appropriate DDM model
            if fundamentals.dividend_per_share <= 0:
                return self._zero_dividend_model(fundamentals, market_data, required_return)
            elif fundamentals.dividend_growth_rate >= required_return:
                return self._multi_stage_model(fundamentals, market_data, required_return)
            else:
                return self._gordon_growth_model(fundamentals, market_data, required_return)
                
        except Exception as e:
            return self._create_fallback_result(str(e))
    
    def _calculate_required_return(self, fundamentals: StockFundamentals, 
                                 market_data: MarketData) -> float:
        """Calculate required return using CAPM"""
        required_return = (
            market_data.risk_free_rate + 
            fundamentals.beta * (market_data.market_return - market_data.risk_free_rate)
        )
        return max(self.min_required_return, required_return)
    
    def _calculate_dividend_sustainability(self, fundamentals: StockFundamentals) -> float:
        """Calculate dividend sustainability score (0-100)"""
        if fundamentals.dividend_per_share <= 0:
            return 0.0
        
        # Payout ratio
        payout_ratio = fundamentals.dividend_per_share / fundamentals.earnings_per_share if fundamentals.earnings_per_share > 0 else 1.0
        
        # Free cash flow coverage
        total_dividends = fundamentals.dividend_per_share * fundamentals.shares_outstanding
        fcf_coverage = fundamentals.free_cash_flow / total_dividends if total_dividends > 0 else 0
        
        # Debt level impact
        debt_to_equity = fundamentals.total_debt / fundamentals.shareholders_equity if fundamentals.shareholders_equity > 0 else 0
        
        # Calculate sustainability score
        score = 100
        
        # Penalize high payout ratios
        if payout_ratio > 0.8:
            score -= 30
        elif payout_ratio > 0.6:
            score -= 15
        
        # Penalize poor FCF coverage
        if fcf_coverage < 1.0:
            score -= 25
        elif fcf_coverage < 1.5:
            score -= 10
        
        # Penalize high debt
        if debt_to_equity > 1.0:
            score -= 20
        elif debt_to_equity > 0.5:
            score -= 10
        
        # Reward earnings growth
        if fundamentals.earnings_growth_rate > 0.1:
            score += 10
        elif fundamentals.earnings_growth_rate < 0:
            score -= 15
        
        return max(0, min(100, score))
    
    def _gordon_growth_model(self, fundamentals: StockFundamentals, 
                           market_data: MarketData, required_return: float) -> DDMResult:
        """Calculate value using Gordon Growth Model"""
        next_dividend = fundamentals.dividend_per_share * (1 + fundamentals.dividend_growth_rate)
        intrinsic_value = next_dividend / (required_return - fundamentals.dividend_growth_rate)
        
        # Calculate metrics
        dividend_yield = fundamentals.dividend_per_share / (fundamentals.market_cap / fundamentals.shares_outstanding)
        payout_ratio = fundamentals.dividend_per_share / fundamentals.earnings_per_share if fundamentals.earnings_per_share > 0 else 0
        sustainability_score = self._calculate_dividend_sustainability(fundamentals)
        
        # Confidence based on dividend history and sustainability
        confidence = self._calculate_confidence(fundamentals, sustainability_score, "gordon")
        
        # Risk level
        risk_level = self._determine_risk_level(fundamentals, sustainability_score)
        
        return DDMResult(
            intrinsic_value=intrinsic_value,
            required_return=required_return,
            dividend_yield=dividend_yield,
            payout_ratio=payout_ratio,
            dividend_sustainability_score=sustainability_score,
            model_type="gordon",
            projected_dividends=[next_dividend],
            confidence=confidence,
            metadata={
                "dividend_growth_rate": fundamentals.dividend_growth_rate,
                "current_dividend": fundamentals.dividend_per_share,
                "next_dividend": next_dividend
            },
            timestamp=datetime.now(),
            interpretation=f"Gordon Growth DDM value: ${intrinsic_value:.2f}",
            risk_level=risk_level
        )
    
    def _multi_stage_model(self, fundamentals: StockFundamentals, 
                         market_data: MarketData, required_return: float) -> DDMResult:
        """Calculate value using Multi-stage DDM"""
        # High growth phase
        pv_dividends = 0
        current_dividend = fundamentals.dividend_per_share
        projected_dividends = []
        
        for year in range(1, self.high_growth_years + 1):
            future_dividend = current_dividend * ((1 + fundamentals.dividend_growth_rate) ** year)
            projected_dividends.append(future_dividend)
            pv_dividends += future_dividend / ((1 + required_return) ** year)
        
        # Terminal value (stable growth phase)
        terminal_dividend = projected_dividends[-1] * (1 + self.stable_growth_rate)
        terminal_value = terminal_dividend / (required_return - self.stable_growth_rate)
        pv_terminal = terminal_value / ((1 + required_return) ** self.high_growth_years)
        
        intrinsic_value = pv_dividends + pv_terminal
        
        # Calculate metrics
        current_price = fundamentals.market_cap / fundamentals.shares_outstanding
        dividend_yield = fundamentals.dividend_per_share / current_price
        payout_ratio = fundamentals.dividend_per_share / fundamentals.earnings_per_share if fundamentals.earnings_per_share > 0 else 0
        sustainability_score = self._calculate_dividend_sustainability(fundamentals)
        
        # Confidence
        confidence = self._calculate_confidence(fundamentals, sustainability_score, "multi_stage")
        
        # Risk level
        risk_level = self._determine_risk_level(fundamentals, sustainability_score)
        
        return DDMResult(
            intrinsic_value=intrinsic_value,
            required_return=required_return,
            dividend_yield=dividend_yield,
            payout_ratio=payout_ratio,
            dividend_sustainability_score=sustainability_score,
            model_type="multi_stage",
            projected_dividends=projected_dividends,
            confidence=confidence,
            metadata={
                "high_growth_years": self.high_growth_years,
                "stable_growth_rate": self.stable_growth_rate,
                "pv_growth_phase": pv_dividends,
                "terminal_value": terminal_value,
                "pv_terminal": pv_terminal
            },
            timestamp=datetime.now(),
            interpretation=f"Multi-stage DDM value: ${intrinsic_value:.2f}",
            risk_level=risk_level
        )
    
    def _zero_dividend_model(self, fundamentals: StockFundamentals, 
                           market_data: MarketData, required_return: float) -> DDMResult:
        """Handle stocks with no dividends"""
        # For non-dividend paying stocks, estimate potential dividend capacity
        potential_payout_ratio = 0.3  # Conservative estimate
        potential_dividend = fundamentals.earnings_per_share * potential_payout_ratio
        
        # Assume they might start paying dividends in the future
        years_to_dividend = 3
        future_eps = fundamentals.earnings_per_share * ((1 + fundamentals.earnings_growth_rate) ** years_to_dividend)
        future_dividend = future_eps * potential_payout_ratio
        
        # Value based on potential future dividends
        if fundamentals.earnings_growth_rate < required_return:
            terminal_value = future_dividend / (required_return - self.stable_growth_rate)
            intrinsic_value = terminal_value / ((1 + required_return) ** years_to_dividend)
        else:
            intrinsic_value = 0  # Cannot value with DDM if no dividends expected
        
        sustainability_score = self._calculate_dividend_sustainability(fundamentals)
        
        return DDMResult(
            intrinsic_value=intrinsic_value,
            required_return=required_return,
            dividend_yield=0.0,
            payout_ratio=0.0,
            dividend_sustainability_score=sustainability_score,
            model_type="zero_growth",
            projected_dividends=[],
            confidence=0.2,  # Low confidence for non-dividend stocks
            metadata={
                "potential_dividend": potential_dividend,
                "years_to_dividend": years_to_dividend,
                "assumption": "Estimated future dividend capacity"
            },
            timestamp=datetime.now(),
            interpretation=f"Zero-dividend DDM estimate: ${intrinsic_value:.2f} (speculative)",
            risk_level="High"
        )
    
    def _calculate_confidence(self, fundamentals: StockFundamentals, 
                            sustainability_score: float, model_type: str) -> float:
        """Calculate confidence score"""
        base_confidence = {
            "gordon": 0.8,
            "multi_stage": 0.7,
            "zero_growth": 0.2
        }.get(model_type, 0.5)
        
        # Adjust for dividend sustainability
        sustainability_adjustment = (sustainability_score - 50) / 200  # -0.25 to +0.25
        
        # Adjust for dividend history (assume positive if current dividend > 0)
        history_adjustment = 0.1 if fundamentals.dividend_per_share > 0 else -0.2
        
        # Adjust for earnings stability
        earnings_adjustment = 0.1 if fundamentals.earnings_growth_rate > 0 else -0.1
        
        confidence = base_confidence + sustainability_adjustment + history_adjustment + earnings_adjustment
        return min(0.95, max(0.1, confidence))
    
    def _determine_risk_level(self, fundamentals: StockFundamentals, 
                            sustainability_score: float) -> str:
        """Determine risk level"""
        risk_factors = 0
        
        # Low sustainability
        if sustainability_score < 40:
            risk_factors += 2
        elif sustainability_score < 60:
            risk_factors += 1
        
        # High payout ratio
        payout_ratio = fundamentals.dividend_per_share / fundamentals.earnings_per_share if fundamentals.earnings_per_share > 0 else 1
        if payout_ratio > 0.8:
            risk_factors += 1
        
        # Negative earnings growth
        if fundamentals.earnings_growth_rate < 0:
            risk_factors += 1
        
        # High debt
        debt_to_equity = fundamentals.total_debt / fundamentals.shareholders_equity if fundamentals.shareholders_equity > 0 else 0
        if debt_to_equity > 1.0:
            risk_factors += 1
        
        if risk_factors >= 3:
            return "High"
        elif risk_factors >= 1:
            return "Medium"
        else:
            return "Low"
    
    def _create_fallback_result(self, error_message: str) -> DDMResult:
        """Create fallback result when calculation fails"""
        return DDMResult(
            intrinsic_value=0.0,
            required_return=0.0,
            dividend_yield=0.0,
            payout_ratio=0.0,
            dividend_sustainability_score=0.0,
            model_type="error",
            projected_dividends=[],
            confidence=0.0,
            metadata={"error": error_message},
            timestamp=datetime.now(),
            interpretation="DDM calculation failed",
            risk_level="High"
        )
    
    def analyze_dividend_policy(self, fundamentals: StockFundamentals) -> Dict[str, Any]:
        """Analyze dividend policy and sustainability"""
        if fundamentals.dividend_per_share <= 0:
            return {
                "policy_type": "No Dividend",
                "sustainability_score": 0,
                "recommendations": ["Consider initiating dividend if cash flow permits"],
                "risk_factors": ["No dividend income for investors"]
            }
        
        payout_ratio = fundamentals.dividend_per_share / fundamentals.earnings_per_share if fundamentals.earnings_per_share > 0 else 1.0
        total_dividends = fundamentals.dividend_per_share * fundamentals.shares_outstanding
        fcf_coverage = fundamentals.free_cash_flow / total_dividends if total_dividends > 0 else 0
        
        sustainability_score = self._calculate_dividend_sustainability(fundamentals)
        
        # Policy classification
        if payout_ratio > 0.8:
            policy_type = "High Payout"
        elif payout_ratio > 0.4:
            policy_type = "Moderate Payout"
        else:
            policy_type = "Conservative Payout"
        
        # Recommendations
        recommendations = []
        risk_factors = []
        
        if payout_ratio > 1.0:
            recommendations.append("Consider reducing dividend to sustainable level")
            risk_factors.append("Dividend exceeds earnings")
        
        if fcf_coverage < 1.0:
            recommendations.append("Improve free cash flow to support dividend")
            risk_factors.append("Insufficient cash flow coverage")
        
        if fundamentals.earnings_growth_rate < 0:
            recommendations.append("Focus on earnings recovery before dividend increases")
            risk_factors.append("Declining earnings trend")
        
        if sustainability_score > 80:
            recommendations.append("Strong dividend policy - consider gradual increases")
        
        return {
            "policy_type": policy_type,
            "payout_ratio": payout_ratio,
            "fcf_coverage": fcf_coverage,
            "sustainability_score": sustainability_score,
            "recommendations": recommendations,
            "risk_factors": risk_factors,
            "dividend_yield": fundamentals.dividend_per_share / (fundamentals.market_cap / fundamentals.shares_outstanding)
        }

# Example usage
if __name__ == "__main__":
    # Sample data for dividend-paying stock
    fundamentals = StockFundamentals(
        revenue=10000000000,  # $10B
        net_income=1000000000,  # $1B
        free_cash_flow=800000000,  # $800M
        total_debt=2000000000,  # $2B
        shareholders_equity=5000000000,  # $5B
        shares_outstanding=100000000,  # 100M shares
        dividend_per_share=4.0,  # $4 dividend
        earnings_per_share=10.0,
        book_value_per_share=50.0,
        revenue_growth_rate=0.05,  # 5%
        earnings_growth_rate=0.08,  # 8%
        dividend_growth_rate=0.06,  # 6%
        beta=1.1,
        market_cap=8000000000  # $8B
    )
    
    market_data = MarketData(
        risk_free_rate=0.03,  # 3%
        market_return=0.10,  # 10%
        inflation_rate=0.025,  # 2.5%
        gdp_growth=0.025,  # 2.5%
        unemployment_rate=0.05,  # 5%
        sector_performance={"Utilities": 0.08, "Consumer Staples": 0.07},
        market_volatility=0.15  # 15%
    )
    
    # Calculate DDM
    ddm_model = DividendDiscountModel()
    result = ddm_model.calculate(fundamentals, market_data)
    
    print(f"DDM Analysis Results:")
    print(f"Intrinsic Value: ${result.intrinsic_value:.2f}")
    print(f"Model Type: {result.model_type}")
    print(f"Required Return: {result.required_return:.2%}")
    print(f"Dividend Yield: {result.dividend_yield:.2%}")
    print(f"Payout Ratio: {result.payout_ratio:.2%}")
    print(f"Sustainability Score: {result.dividend_sustainability_score:.1f}/100")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Risk Level: {result.risk_level}")
    print(f"Interpretation: {result.interpretation}")
    
    # Dividend policy analysis
    policy_analysis = ddm_model.analyze_dividend_policy(fundamentals)
    print(f"\nDividend Policy Analysis:")
    print(f"Policy Type: {policy_analysis['policy_type']}")
    print(f"Payout Ratio: {policy_analysis['payout_ratio']:.2%}")
    print(f"FCF Coverage: {policy_analysis['fcf_coverage']:.1f}x")
    print(f"Recommendations: {policy_analysis['recommendations']}")
    if policy_analysis['risk_factors']:
        print(f"Risk Factors: {policy_analysis['risk_factors']}")