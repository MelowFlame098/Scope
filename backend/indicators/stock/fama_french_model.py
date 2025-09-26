"""Fama-French Model for Stock Analysis

This module implements the Fama-French factor models including:
- 3-Factor Model (Market, Size, Value)
- 5-Factor Model (Market, Size, Value, Profitability, Investment)
- Factor loading analysis
- Risk attribution
- Performance attribution

Author: Assistant
Date: 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
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
class FamaFrenchFactors:
    """Fama-French factor data"""
    market_premium: float  # Rm - Rf
    size_premium: float    # SMB (Small Minus Big)
    value_premium: float   # HML (High Minus Low)
    profitability_premium: float  # RMW (Robust Minus Weak)
    investment_premium: float     # CMA (Conservative Minus Aggressive)

@dataclass
class FamaFrenchResult:
    """Result of Fama-French analysis"""
    expected_return_3f: float
    expected_return_5f: float
    alpha_3f: float
    alpha_5f: float
    market_beta: float
    size_loading: float
    value_loading: float
    profitability_loading: float
    investment_loading: float
    r_squared_3f: float
    r_squared_5f: float
    factor_contributions: Dict[str, float]
    risk_attribution: Dict[str, float]
    style_classification: Dict[str, str]
    model_comparison: Dict[str, Any]
    confidence: float
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str

class FamaFrenchModel:
    """Fama-French Factor Model"""
    
    def __init__(self):
        self.min_loading = -2.0
        self.max_loading = 2.0
        self.significance_threshold = 0.3
    
    def calculate(self, fundamentals: StockFundamentals, market_data: MarketData,
                 factors: Optional[FamaFrenchFactors] = None,
                 actual_return: Optional[float] = None) -> FamaFrenchResult:
        """Calculate Fama-French model results"""
        try:
            # Use provided factors or estimate from fundamentals
            if factors is None:
                factors = self._estimate_factors(fundamentals, market_data)
            
            # Calculate factor loadings based on stock characteristics
            loadings = self._calculate_factor_loadings(fundamentals, market_data)
            
            # Calculate expected returns
            expected_return_3f = self._calculate_3factor_return(
                market_data.risk_free_rate, factors, loadings
            )
            expected_return_5f = self._calculate_5factor_return(
                market_data.risk_free_rate, factors, loadings
            )
            
            # Calculate alphas if actual return is provided
            alpha_3f = (actual_return - expected_return_3f) if actual_return else 0.0
            alpha_5f = (actual_return - expected_return_5f) if actual_return else 0.0
            
            # Calculate R-squared (simplified approximation)
            r_squared_3f, r_squared_5f = self._calculate_r_squared(loadings)
            
            # Factor contributions
            factor_contributions = self._calculate_factor_contributions(
                factors, loadings, market_data.risk_free_rate
            )
            
            # Risk attribution
            risk_attribution = self._calculate_risk_attribution(loadings, factors)
            
            # Style classification
            style_classification = self._classify_style(loadings)
            
            # Model comparison
            model_comparison = self._compare_models(
                expected_return_3f, expected_return_5f, alpha_3f, alpha_5f,
                r_squared_3f, r_squared_5f
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(loadings, r_squared_5f)
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                loadings, style_classification, alpha_5f
            )
            
            return FamaFrenchResult(
                expected_return_3f=expected_return_3f,
                expected_return_5f=expected_return_5f,
                alpha_3f=alpha_3f,
                alpha_5f=alpha_5f,
                market_beta=loadings['market'],
                size_loading=loadings['size'],
                value_loading=loadings['value'],
                profitability_loading=loadings['profitability'],
                investment_loading=loadings['investment'],
                r_squared_3f=r_squared_3f,
                r_squared_5f=r_squared_5f,
                factor_contributions=factor_contributions,
                risk_attribution=risk_attribution,
                style_classification=style_classification,
                model_comparison=model_comparison,
                confidence=confidence,
                metadata={
                    "factors_used": {
                        "market_premium": factors.market_premium,
                        "size_premium": factors.size_premium,
                        "value_premium": factors.value_premium,
                        "profitability_premium": factors.profitability_premium,
                        "investment_premium": factors.investment_premium
                    },
                    "actual_return": actual_return,
                    "risk_free_rate": market_data.risk_free_rate
                },
                timestamp=datetime.now(),
                interpretation=interpretation
            )
            
        except Exception as e:
            return self._create_fallback_result(str(e))
    
    def _estimate_factors(self, fundamentals: StockFundamentals, 
                         market_data: MarketData) -> FamaFrenchFactors:
        """Estimate Fama-French factors from available data"""
        # Market premium
        market_premium = market_data.market_return - market_data.risk_free_rate
        
        # Size premium (simplified estimation)
        # Assume small cap premium based on market conditions
        size_premium = 0.02 if market_premium > 0.05 else 0.01
        
        # Value premium (simplified estimation)
        # Based on general market value premium
        value_premium = 0.03 if market_data.inflation_rate < 0.03 else 0.02
        
        # Profitability premium
        profitability_premium = 0.025
        
        # Investment premium
        investment_premium = 0.02
        
        return FamaFrenchFactors(
            market_premium=market_premium,
            size_premium=size_premium,
            value_premium=value_premium,
            profitability_premium=profitability_premium,
            investment_premium=investment_premium
        )
    
    def _calculate_factor_loadings(self, fundamentals: StockFundamentals,
                                 market_data: MarketData) -> Dict[str, float]:
        """Calculate factor loadings based on stock characteristics"""
        # Market beta (from fundamentals)
        market_loading = fundamentals.beta
        
        # Size loading (based on market cap)
        # Smaller companies have higher size loading
        median_market_cap = 10000000000  # $10B as median
        if fundamentals.market_cap < median_market_cap / 2:
            size_loading = 0.8  # Small cap
        elif fundamentals.market_cap < median_market_cap * 2:
            size_loading = 0.2  # Mid cap
        else:
            size_loading = -0.3  # Large cap
        
        # Value loading (based on P/B ratio)
        price_per_share = fundamentals.market_cap / fundamentals.shares_outstanding
        pb_ratio = price_per_share / fundamentals.book_value_per_share
        
        if pb_ratio < 1.0:
            value_loading = 0.6  # Deep value
        elif pb_ratio < 2.0:
            value_loading = 0.3  # Value
        elif pb_ratio < 4.0:
            value_loading = -0.1  # Neutral
        else:
            value_loading = -0.4  # Growth
        
        # Profitability loading (based on ROE)
        roe = fundamentals.net_income / fundamentals.shareholders_equity if fundamentals.shareholders_equity > 0 else 0
        
        if roe > 0.15:
            profitability_loading = 0.5  # High profitability
        elif roe > 0.10:
            profitability_loading = 0.2  # Good profitability
        elif roe > 0.05:
            profitability_loading = -0.1  # Average profitability
        else:
            profitability_loading = -0.4  # Low profitability
        
        # Investment loading (based on growth)
        # High growth companies tend to have negative investment loading
        avg_growth = (fundamentals.revenue_growth_rate + fundamentals.earnings_growth_rate) / 2
        
        if avg_growth > 0.15:
            investment_loading = -0.4  # High growth (aggressive investment)
        elif avg_growth > 0.08:
            investment_loading = -0.1  # Moderate growth
        elif avg_growth > 0.03:
            investment_loading = 0.1   # Conservative growth
        else:
            investment_loading = 0.3   # Low/no growth (conservative)
        
        # Constrain loadings to reasonable ranges
        return {
            'market': max(self.min_loading, min(self.max_loading, market_loading)),
            'size': max(self.min_loading, min(self.max_loading, size_loading)),
            'value': max(self.min_loading, min(self.max_loading, value_loading)),
            'profitability': max(self.min_loading, min(self.max_loading, profitability_loading)),
            'investment': max(self.min_loading, min(self.max_loading, investment_loading))
        }
    
    def _calculate_3factor_return(self, risk_free_rate: float, 
                                factors: FamaFrenchFactors,
                                loadings: Dict[str, float]) -> float:
        """Calculate expected return using 3-factor model"""
        return (
            risk_free_rate +
            loadings['market'] * factors.market_premium +
            loadings['size'] * factors.size_premium +
            loadings['value'] * factors.value_premium
        )
    
    def _calculate_5factor_return(self, risk_free_rate: float,
                                factors: FamaFrenchFactors,
                                loadings: Dict[str, float]) -> float:
        """Calculate expected return using 5-factor model"""
        return (
            risk_free_rate +
            loadings['market'] * factors.market_premium +
            loadings['size'] * factors.size_premium +
            loadings['value'] * factors.value_premium +
            loadings['profitability'] * factors.profitability_premium +
            loadings['investment'] * factors.investment_premium
        )
    
    def _calculate_r_squared(self, loadings: Dict[str, float]) -> Tuple[float, float]:
        """Calculate approximate R-squared for both models"""
        # Simplified R-squared calculation based on factor loadings
        # Higher absolute loadings generally indicate better model fit
        
        # 3-factor R-squared
        factor_3f_strength = (
            abs(loadings['market']) +
            abs(loadings['size']) +
            abs(loadings['value'])
        ) / 3
        r_squared_3f = min(0.9, factor_3f_strength * 0.6)
        
        # 5-factor R-squared (generally higher)
        factor_5f_strength = (
            abs(loadings['market']) +
            abs(loadings['size']) +
            abs(loadings['value']) +
            abs(loadings['profitability']) +
            abs(loadings['investment'])
        ) / 5
        r_squared_5f = min(0.95, factor_5f_strength * 0.7)
        
        return r_squared_3f, r_squared_5f
    
    def _calculate_factor_contributions(self, factors: FamaFrenchFactors,
                                      loadings: Dict[str, float],
                                      risk_free_rate: float) -> Dict[str, float]:
        """Calculate contribution of each factor to expected return"""
        return {
            'risk_free': risk_free_rate,
            'market': loadings['market'] * factors.market_premium,
            'size': loadings['size'] * factors.size_premium,
            'value': loadings['value'] * factors.value_premium,
            'profitability': loadings['profitability'] * factors.profitability_premium,
            'investment': loadings['investment'] * factors.investment_premium
        }
    
    def _calculate_risk_attribution(self, loadings: Dict[str, float],
                                  factors: FamaFrenchFactors) -> Dict[str, float]:
        """Calculate risk attribution to each factor"""
        # Simplified risk attribution based on factor loadings and premiums
        total_risk = sum(abs(loading) * 0.1 for loading in loadings.values())  # Simplified
        
        if total_risk == 0:
            return {factor: 0.0 for factor in loadings.keys()}
        
        return {
            factor: (abs(loading) * 0.1) / total_risk
            for factor, loading in loadings.items()
        }
    
    def _classify_style(self, loadings: Dict[str, float]) -> Dict[str, str]:
        """Classify investment style based on factor loadings"""
        classifications = {}
        
        # Size classification
        if loadings['size'] > self.significance_threshold:
            classifications['size'] = 'Small Cap'
        elif loadings['size'] < -self.significance_threshold:
            classifications['size'] = 'Large Cap'
        else:
            classifications['size'] = 'Mid Cap'
        
        # Value/Growth classification
        if loadings['value'] > self.significance_threshold:
            classifications['style'] = 'Value'
        elif loadings['value'] < -self.significance_threshold:
            classifications['style'] = 'Growth'
        else:
            classifications['style'] = 'Blend'
        
        # Profitability classification
        if loadings['profitability'] > self.significance_threshold:
            classifications['profitability'] = 'High Quality'
        elif loadings['profitability'] < -self.significance_threshold:
            classifications['profitability'] = 'Low Quality'
        else:
            classifications['profitability'] = 'Average Quality'
        
        # Investment classification
        if loadings['investment'] > self.significance_threshold:
            classifications['investment'] = 'Conservative'
        elif loadings['investment'] < -self.significance_threshold:
            classifications['investment'] = 'Aggressive'
        else:
            classifications['investment'] = 'Moderate'
        
        return classifications
    
    def _compare_models(self, return_3f: float, return_5f: float,
                      alpha_3f: float, alpha_5f: float,
                      r_squared_3f: float, r_squared_5f: float) -> Dict[str, Any]:
        """Compare 3-factor vs 5-factor model performance"""
        return {
            'preferred_model': '5-Factor' if r_squared_5f > r_squared_3f else '3-Factor',
            'r_squared_improvement': r_squared_5f - r_squared_3f,
            'alpha_improvement': abs(alpha_5f) < abs(alpha_3f),
            'return_difference': return_5f - return_3f,
            'model_metrics': {
                '3_factor': {
                    'expected_return': return_3f,
                    'alpha': alpha_3f,
                    'r_squared': r_squared_3f
                },
                '5_factor': {
                    'expected_return': return_5f,
                    'alpha': alpha_5f,
                    'r_squared': r_squared_5f
                }
            }
        }
    
    def _calculate_confidence(self, loadings: Dict[str, float], 
                            r_squared: float) -> float:
        """Calculate confidence in the model"""
        base_confidence = 0.6
        
        # Adjust for R-squared
        r_squared_adjustment = r_squared * 0.3
        
        # Adjust for factor significance
        significant_factors = sum(
            1 for loading in loadings.values() 
            if abs(loading) > self.significance_threshold
        )
        significance_adjustment = (significant_factors / len(loadings)) * 0.2
        
        # Adjust for extreme loadings (reduce confidence)
        extreme_loadings = sum(
            1 for loading in loadings.values()
            if abs(loading) > 1.5
        )
        extreme_adjustment = -extreme_loadings * 0.05
        
        confidence = (
            base_confidence + r_squared_adjustment + 
            significance_adjustment + extreme_adjustment
        )
        
        return min(0.95, max(0.3, confidence))
    
    def _generate_interpretation(self, loadings: Dict[str, float],
                               style_classification: Dict[str, str],
                               alpha: float) -> str:
        """Generate interpretation of results"""
        interpretation_parts = []
        
        # Style interpretation
        interpretation_parts.append(
            f"Style: {style_classification.get('size', 'Unknown')} "
            f"{style_classification.get('style', 'Unknown')}"
        )
        
        # Quality interpretation
        if 'profitability' in style_classification:
            interpretation_parts.append(
                f"Quality: {style_classification['profitability']}"
            )
        
        # Alpha interpretation
        if abs(alpha) > 0.02:
            alpha_desc = "outperforming" if alpha > 0 else "underperforming"
            interpretation_parts.append(
                f"Alpha: {alpha:.2%} ({alpha_desc} factors)"
            )
        
        # Dominant factors
        dominant_factors = [
            factor for factor, loading in loadings.items()
            if abs(loading) > self.significance_threshold
        ]
        
        if dominant_factors:
            interpretation_parts.append(
                f"Key factors: {', '.join(dominant_factors)}"
            )
        
        return "; ".join(interpretation_parts)
    
    def _create_fallback_result(self, error_message: str) -> FamaFrenchResult:
        """Create fallback result when calculation fails"""
        return FamaFrenchResult(
            expected_return_3f=0.0,
            expected_return_5f=0.0,
            alpha_3f=0.0,
            alpha_5f=0.0,
            market_beta=1.0,
            size_loading=0.0,
            value_loading=0.0,
            profitability_loading=0.0,
            investment_loading=0.0,
            r_squared_3f=0.0,
            r_squared_5f=0.0,
            factor_contributions={},
            risk_attribution={},
            style_classification={},
            model_comparison={},
            confidence=0.0,
            metadata={"error": error_message},
            timestamp=datetime.now(),
            interpretation="Fama-French calculation failed"
        )
    
    def analyze_factor_exposure(self, loadings: Dict[str, float]) -> Dict[str, Any]:
        """Analyze factor exposure and provide insights"""
        exposure_analysis = {}
        
        for factor, loading in loadings.items():
            if abs(loading) > 1.0:
                exposure_level = "High"
            elif abs(loading) > 0.5:
                exposure_level = "Moderate"
            elif abs(loading) > 0.2:
                exposure_level = "Low"
            else:
                exposure_level = "Minimal"
            
            direction = "Positive" if loading > 0 else "Negative"
            
            exposure_analysis[factor] = {
                "loading": loading,
                "exposure_level": exposure_level,
                "direction": direction,
                "interpretation": self._interpret_factor_loading(factor, loading)
            }
        
        return exposure_analysis
    
    def _interpret_factor_loading(self, factor: str, loading: float) -> str:
        """Interpret individual factor loading"""
        interpretations = {
            'market': {
                'positive': "Higher sensitivity to market movements",
                'negative': "Inverse relationship to market (unusual)"
            },
            'size': {
                'positive': "Small-cap characteristics",
                'negative': "Large-cap characteristics"
            },
            'value': {
                'positive': "Value stock characteristics",
                'negative': "Growth stock characteristics"
            },
            'profitability': {
                'positive': "High profitability/quality characteristics",
                'negative': "Lower profitability characteristics"
            },
            'investment': {
                'positive': "Conservative investment policy",
                'negative': "Aggressive investment/high growth"
            }
        }
        
        direction = 'positive' if loading > 0 else 'negative'
        return interpretations.get(factor, {}).get(direction, "Unknown factor interpretation")

# Example usage
if __name__ == "__main__":
    # Sample data
    fundamentals = StockFundamentals(
        revenue=5000000000,  # $5B (smaller company)
        net_income=500000000,  # $500M
        free_cash_flow=400000000,
        total_debt=1000000000,
        shareholders_equity=2500000000,
        shares_outstanding=50000000,
        dividend_per_share=2.0,
        earnings_per_share=10.0,
        book_value_per_share=50.0,
        revenue_growth_rate=0.12,  # High growth
        earnings_growth_rate=0.15,
        dividend_growth_rate=0.08,
        beta=1.3,  # Higher beta
        market_cap=4000000000  # $4B (mid-cap)
    )
    
    market_data = MarketData(
        risk_free_rate=0.03,
        market_return=0.10,
        inflation_rate=0.025,
        gdp_growth=0.025,
        unemployment_rate=0.05,
        sector_performance={"Technology": 0.12},
        market_volatility=0.15
    )
    
    # Custom factors (optional)
    factors = FamaFrenchFactors(
        market_premium=0.07,  # 7%
        size_premium=0.02,    # 2%
        value_premium=0.03,   # 3%
        profitability_premium=0.025,  # 2.5%
        investment_premium=0.02       # 2%
    )
    
    # Calculate Fama-French
    ff_model = FamaFrenchModel()
    result = ff_model.calculate(fundamentals, market_data, factors, actual_return=0.14)
    
    print(f"Fama-French Analysis Results:")
    print(f"3-Factor Expected Return: {result.expected_return_3f:.2%}")
    print(f"5-Factor Expected Return: {result.expected_return_5f:.2%}")
    print(f"3-Factor Alpha: {result.alpha_3f:.2%}")
    print(f"5-Factor Alpha: {result.alpha_5f:.2%}")
    print(f"\nFactor Loadings:")
    print(f"Market Beta: {result.market_beta:.3f}")
    print(f"Size Loading: {result.size_loading:.3f}")
    print(f"Value Loading: {result.value_loading:.3f}")
    print(f"Profitability Loading: {result.profitability_loading:.3f}")
    print(f"Investment Loading: {result.investment_loading:.3f}")
    print(f"\nModel Fit:")
    print(f"3-Factor R²: {result.r_squared_3f:.3f}")
    print(f"5-Factor R²: {result.r_squared_5f:.3f}")
    print(f"\nStyle Classification:")
    for aspect, classification in result.style_classification.items():
        print(f"{aspect.title()}: {classification}")
    print(f"\nConfidence: {result.confidence:.1%}")
    print(f"Interpretation: {result.interpretation}")
    
    # Factor exposure analysis
    loadings_dict = {
        'market': result.market_beta,
        'size': result.size_loading,
        'value': result.value_loading,
        'profitability': result.profitability_loading,
        'investment': result.investment_loading
    }
    
    exposure_analysis = ff_model.analyze_factor_exposure(loadings_dict)
    print(f"\nFactor Exposure Analysis:")
    for factor, analysis in exposure_analysis.items():
        print(f"{factor.title()}: {analysis['exposure_level']} {analysis['direction']} exposure")
        print(f"  - {analysis['interpretation']}")