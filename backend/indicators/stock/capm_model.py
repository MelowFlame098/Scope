"""CAPM (Capital Asset Pricing Model) for Stock Analysis

This module implements the Capital Asset Pricing Model for stocks including:
- Expected return calculation using CAPM
- Risk-adjusted return analysis
- Beta analysis and interpretation
- Sharpe ratio approximation
- Security Market Line analysis

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
class CAPMResult:
    """Result of CAPM calculation"""
    expected_return: float
    risk_adjusted_return: float
    sharpe_ratio: float
    beta: float
    alpha: float
    systematic_risk: float
    unsystematic_risk: float
    risk_premium: float
    security_market_line_position: str  # "above", "below", "on"
    risk_classification: str  # "Conservative", "Moderate", "Aggressive"
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str

class CAPMModel:
    """Capital Asset Pricing Model"""
    
    def __init__(self):
        self.min_beta = 0.1
        self.max_beta = 3.0
        self.market_volatility_threshold = 0.20
    
    def calculate(self, fundamentals: StockFundamentals, market_data: MarketData, 
                 actual_return: Optional[float] = None) -> CAPMResult:
        """Calculate CAPM metrics"""
        try:
            # Validate and adjust beta
            beta = self._validate_beta(fundamentals.beta)
            
            # Calculate expected return using CAPM
            expected_return = self._calculate_expected_return(beta, market_data)
            
            # Calculate risk-adjusted return
            risk_adjusted_return = self._calculate_risk_adjusted_return(
                expected_return, market_data.risk_free_rate
            )
            
            # Calculate Sharpe ratio approximation
            sharpe_ratio = self._calculate_sharpe_ratio(
                expected_return, market_data.risk_free_rate, market_data.market_volatility, beta
            )
            
            # Calculate alpha if actual return is provided
            alpha = self._calculate_alpha(actual_return, expected_return) if actual_return else 0.0
            
            # Calculate risk components
            systematic_risk, unsystematic_risk = self._calculate_risk_components(
                beta, market_data.market_volatility
            )
            
            # Calculate risk premium
            risk_premium = expected_return - market_data.risk_free_rate
            
            # Determine SML position
            sml_position = self._determine_sml_position(alpha)
            
            # Risk classification
            risk_classification = self._classify_risk(beta)
            
            # Calculate confidence
            confidence = self._calculate_confidence(beta, market_data)
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                expected_return, beta, alpha, risk_classification
            )
            
            return CAPMResult(
                expected_return=expected_return,
                risk_adjusted_return=risk_adjusted_return,
                sharpe_ratio=sharpe_ratio,
                beta=beta,
                alpha=alpha,
                systematic_risk=systematic_risk,
                unsystematic_risk=unsystematic_risk,
                risk_premium=risk_premium,
                security_market_line_position=sml_position,
                risk_classification=risk_classification,
                confidence=confidence,
                metadata={
                    "risk_free_rate": market_data.risk_free_rate,
                    "market_return": market_data.market_return,
                    "market_premium": market_data.market_return - market_data.risk_free_rate,
                    "actual_return": actual_return,
                    "market_volatility": market_data.market_volatility
                },
                timestamp=datetime.now(),
                interpretation=interpretation
            )
            
        except Exception as e:
            return self._create_fallback_result(str(e))
    
    def _validate_beta(self, beta: float) -> float:
        """Validate and adjust beta value"""
        if beta <= 0 or np.isnan(beta) or np.isinf(beta):
            return 1.0  # Default to market beta
        return max(self.min_beta, min(self.max_beta, beta))
    
    def _calculate_expected_return(self, beta: float, market_data: MarketData) -> float:
        """Calculate expected return using CAPM formula"""
        market_premium = market_data.market_return - market_data.risk_free_rate
        return market_data.risk_free_rate + beta * market_premium
    
    def _calculate_risk_adjusted_return(self, expected_return: float, 
                                      risk_free_rate: float) -> float:
        """Calculate risk-adjusted return (excess return)"""
        return expected_return - risk_free_rate
    
    def _calculate_sharpe_ratio(self, expected_return: float, risk_free_rate: float,
                              market_volatility: float, beta: float) -> float:
        """Calculate approximated Sharpe ratio"""
        excess_return = expected_return - risk_free_rate
        # Approximate stock volatility using beta and market volatility
        stock_volatility = abs(beta) * market_volatility
        
        if stock_volatility == 0:
            return 0.0
        
        return excess_return / stock_volatility
    
    def _calculate_alpha(self, actual_return: float, expected_return: float) -> float:
        """Calculate Jensen's alpha"""
        return actual_return - expected_return
    
    def _calculate_risk_components(self, beta: float, 
                                 market_volatility: float) -> tuple[float, float]:
        """Calculate systematic and unsystematic risk components"""
        # Systematic risk (market-related)
        systematic_risk = (beta ** 2) * (market_volatility ** 2)
        
        # Approximate total risk (simplified)
        total_risk_variance = systematic_risk * 1.5  # Rough approximation
        
        # Unsystematic risk (company-specific)
        unsystematic_risk = max(0, total_risk_variance - systematic_risk)
        
        return np.sqrt(systematic_risk), np.sqrt(unsystematic_risk)
    
    def _determine_sml_position(self, alpha: float) -> str:
        """Determine position relative to Security Market Line"""
        if alpha > 0.02:  # 2% threshold
            return "above"
        elif alpha < -0.02:
            return "below"
        else:
            return "on"
    
    def _classify_risk(self, beta: float) -> str:
        """Classify risk level based on beta"""
        if beta < 0.7:
            return "Conservative"
        elif beta <= 1.3:
            return "Moderate"
        else:
            return "Aggressive"
    
    def _calculate_confidence(self, beta: float, market_data: MarketData) -> float:
        """Calculate confidence in CAPM estimate"""
        base_confidence = 0.7
        
        # Adjust for beta reliability
        if 0.8 <= beta <= 1.2:
            beta_adjustment = 0.1  # Beta close to market
        elif 0.5 <= beta <= 1.5:
            beta_adjustment = 0.05
        else:
            beta_adjustment = -0.1  # Extreme beta values
        
        # Adjust for market volatility
        if market_data.market_volatility > self.market_volatility_threshold:
            volatility_adjustment = -0.1  # High volatility reduces confidence
        else:
            volatility_adjustment = 0.05
        
        # Adjust for market conditions
        market_premium = market_data.market_return - market_data.risk_free_rate
        if market_premium < 0.02:  # Very low market premium
            market_adjustment = -0.1
        else:
            market_adjustment = 0.0
        
        confidence = base_confidence + beta_adjustment + volatility_adjustment + market_adjustment
        return min(0.95, max(0.3, confidence))
    
    def _generate_interpretation(self, expected_return: float, beta: float, 
                               alpha: float, risk_classification: str) -> str:
        """Generate interpretation of CAPM results"""
        interpretation_parts = []
        
        # Expected return interpretation
        interpretation_parts.append(
            f"CAPM expected return: {expected_return:.2%}"
        )
        
        # Beta interpretation
        if beta > 1.0:
            interpretation_parts.append(
                f"High beta ({beta:.2f}) indicates higher volatility than market"
            )
        elif beta < 1.0:
            interpretation_parts.append(
                f"Low beta ({beta:.2f}) indicates lower volatility than market"
            )
        else:
            interpretation_parts.append(
                f"Beta ({beta:.2f}) indicates similar volatility to market"
            )
        
        # Alpha interpretation
        if alpha > 0.02:
            interpretation_parts.append(
                f"Positive alpha ({alpha:.2%}) suggests outperformance"
            )
        elif alpha < -0.02:
            interpretation_parts.append(
                f"Negative alpha ({alpha:.2%}) suggests underperformance"
            )
        
        # Risk classification
        interpretation_parts.append(f"Risk profile: {risk_classification}")
        
        return "; ".join(interpretation_parts)
    
    def _create_fallback_result(self, error_message: str) -> CAPMResult:
        """Create fallback result when calculation fails"""
        return CAPMResult(
            expected_return=0.0,
            risk_adjusted_return=0.0,
            sharpe_ratio=0.0,
            beta=1.0,
            alpha=0.0,
            systematic_risk=0.0,
            unsystematic_risk=0.0,
            risk_premium=0.0,
            security_market_line_position="on",
            risk_classification="Unknown",
            confidence=0.0,
            metadata={"error": error_message},
            timestamp=datetime.now(),
            interpretation="CAPM calculation failed"
        )
    
    def analyze_beta_stability(self, historical_betas: List[float]) -> Dict[str, Any]:
        """Analyze beta stability over time"""
        if not historical_betas or len(historical_betas) < 2:
            return {
                "stability_score": 0,
                "trend": "Unknown",
                "volatility": 0,
                "recommendation": "Insufficient data for beta analysis"
            }
        
        betas = np.array(historical_betas)
        
        # Calculate stability metrics
        beta_mean = np.mean(betas)
        beta_std = np.std(betas)
        beta_cv = beta_std / beta_mean if beta_mean != 0 else float('inf')
        
        # Stability score (0-100)
        stability_score = max(0, 100 - (beta_cv * 100))
        
        # Trend analysis
        if len(betas) >= 3:
            recent_trend = np.polyfit(range(len(betas)), betas, 1)[0]
            if recent_trend > 0.1:
                trend = "Increasing"
            elif recent_trend < -0.1:
                trend = "Decreasing"
            else:
                trend = "Stable"
        else:
            trend = "Unknown"
        
        # Recommendation
        if stability_score > 80:
            recommendation = "Beta is stable - reliable for CAPM"
        elif stability_score > 60:
            recommendation = "Beta is moderately stable - use with caution"
        else:
            recommendation = "Beta is unstable - consider alternative risk measures"
        
        return {
            "stability_score": stability_score,
            "mean_beta": beta_mean,
            "beta_volatility": beta_std,
            "coefficient_of_variation": beta_cv,
            "trend": trend,
            "recommendation": recommendation,
            "current_vs_historical": {
                "current": betas[-1],
                "historical_average": beta_mean,
                "deviation": abs(betas[-1] - beta_mean)
            }
        }
    
    def compare_to_sector(self, stock_beta: float, sector_betas: Dict[str, float]) -> Dict[str, Any]:
        """Compare stock beta to sector averages"""
        if not sector_betas:
            return {"comparison": "No sector data available"}
        
        sector_beta_values = list(sector_betas.values())
        sector_mean = np.mean(sector_beta_values)
        sector_std = np.std(sector_beta_values)
        
        # Calculate percentile
        percentile = (sum(1 for beta in sector_beta_values if beta < stock_beta) / 
                     len(sector_beta_values)) * 100
        
        # Classification relative to sector
        if stock_beta > sector_mean + sector_std:
            sector_classification = "High risk relative to sector"
        elif stock_beta < sector_mean - sector_std:
            sector_classification = "Low risk relative to sector"
        else:
            sector_classification = "Average risk relative to sector"
        
        return {
            "stock_beta": stock_beta,
            "sector_average": sector_mean,
            "sector_std": sector_std,
            "percentile": percentile,
            "classification": sector_classification,
            "sector_comparison": {
                name: {
                    "beta": beta,
                    "relative_risk": "Higher" if beta > stock_beta else "Lower"
                }
                for name, beta in sector_betas.items()
            }
        }
    
    def calculate_portfolio_beta(self, holdings: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Calculate portfolio beta from individual stock holdings
        
        Args:
            holdings: Dict with stock symbols as keys, values are dicts with 'weight' and 'beta'
        """
        if not holdings:
            return {"portfolio_beta": 0, "error": "No holdings provided"}
        
        total_weight = sum(stock['weight'] for stock in holdings.values())
        
        if total_weight == 0:
            return {"portfolio_beta": 0, "error": "Total weight is zero"}
        
        # Calculate weighted average beta
        portfolio_beta = sum(
            stock['weight'] * stock['beta'] for stock in holdings.values()
        ) / total_weight
        
        # Risk contribution analysis
        risk_contributions = {}
        for symbol, stock in holdings.items():
            contribution = (stock['weight'] / total_weight) * stock['beta']
            risk_contributions[symbol] = {
                "weight": stock['weight'] / total_weight,
                "beta": stock['beta'],
                "risk_contribution": contribution,
                "risk_percentage": (contribution / portfolio_beta) * 100 if portfolio_beta != 0 else 0
            }
        
        # Portfolio risk classification
        portfolio_risk_class = self._classify_risk(portfolio_beta)
        
        return {
            "portfolio_beta": portfolio_beta,
            "risk_classification": portfolio_risk_class,
            "total_weight": total_weight,
            "risk_contributions": risk_contributions,
            "diversification_benefit": {
                "individual_avg_beta": np.mean([stock['beta'] for stock in holdings.values()]),
                "portfolio_beta": portfolio_beta,
                "benefit": "Yes" if portfolio_beta < np.mean([stock['beta'] for stock in holdings.values()]) else "No"
            }
        }

# Example usage
if __name__ == "__main__":
    # Sample data
    fundamentals = StockFundamentals(
        revenue=10000000000,
        net_income=1000000000,
        free_cash_flow=800000000,
        total_debt=2000000000,
        shareholders_equity=5000000000,
        shares_outstanding=100000000,
        dividend_per_share=4.0,
        earnings_per_share=10.0,
        book_value_per_share=50.0,
        revenue_growth_rate=0.05,
        earnings_growth_rate=0.08,
        dividend_growth_rate=0.06,
        beta=1.2,  # Higher than market
        market_cap=8000000000
    )
    
    market_data = MarketData(
        risk_free_rate=0.03,  # 3%
        market_return=0.10,  # 10%
        inflation_rate=0.025,
        gdp_growth=0.025,
        unemployment_rate=0.05,
        sector_performance={"Technology": 0.12, "Healthcare": 0.09},
        market_volatility=0.15  # 15%
    )
    
    # Calculate CAPM
    capm_model = CAPMModel()
    result = capm_model.calculate(fundamentals, market_data, actual_return=0.12)
    
    print(f"CAPM Analysis Results:")
    print(f"Expected Return: {result.expected_return:.2%}")
    print(f"Risk-Adjusted Return: {result.risk_adjusted_return:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"Beta: {result.beta:.2f}")
    print(f"Alpha: {result.alpha:.2%}")
    print(f"Risk Premium: {result.risk_premium:.2%}")
    print(f"Risk Classification: {result.risk_classification}")
    print(f"SML Position: {result.security_market_line_position}")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Interpretation: {result.interpretation}")
    
    # Beta stability analysis
    historical_betas = [1.1, 1.15, 1.18, 1.22, 1.2, 1.25, 1.2]
    beta_analysis = capm_model.analyze_beta_stability(historical_betas)
    print(f"\nBeta Stability Analysis:")
    print(f"Stability Score: {beta_analysis['stability_score']:.1f}/100")
    print(f"Trend: {beta_analysis['trend']}")
    print(f"Recommendation: {beta_analysis['recommendation']}")
    
    # Sector comparison
    sector_betas = {
        "AAPL": 1.1,
        "MSFT": 0.9,
        "GOOGL": 1.0,
        "AMZN": 1.3,
        "TSLA": 2.0
    }
    sector_comparison = capm_model.compare_to_sector(1.2, sector_betas)
    print(f"\nSector Comparison:")
    print(f"Stock Beta: {sector_comparison['stock_beta']:.2f}")
    print(f"Sector Average: {sector_comparison['sector_average']:.2f}")
    print(f"Percentile: {sector_comparison['percentile']:.1f}%")
    print(f"Classification: {sector_comparison['classification']}")