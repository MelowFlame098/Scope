"""Macroeconomic Factor Model for Index Analysis

This module implements macroeconomic factor models for index valuation,
analyzing the impact of various economic indicators on index fair value.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime
from enum import Enum
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class IndexIndicatorType(Enum):
    """Index-specific indicator types"""
    MACROECONOMIC_FACTORS = "macro_factors"

class EconomicRegime(Enum):
    """Economic regime types for regime-switching model"""
    EXPANSION = "expansion"
    RECESSION = "recession"
    RECOVERY = "recovery"
    STAGFLATION = "stagflation"
    DEFLATION = "deflation"

@dataclass
class RegimeIndicators:
    """Indicators for regime detection"""
    gdp_growth_trend: float
    unemployment_trend: float
    inflation_trend: float
    yield_curve_slope: float
    credit_spreads: float
    volatility_regime: float

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
class MacroFactorResult:
    """Result of macroeconomic factor model calculation"""
    fair_value: float
    confidence: float
    signal: str
    risk_level: str
    factor_scores: Dict[str, float]
    interpretation: str
    time_horizon: str
    current_regime: EconomicRegime
    regime_probability: Dict[str, float]
    dynamic_loadings: Dict[str, float]
    regime_adjusted_fair_value: float
    factor_contributions: Dict[str, float]

class MacroeconomicFactorModel:
    """Advanced macroeconomic factor model with regime-switching and dynamic loading"""
    
    def __init__(self):
        # Base factor weights (will be dynamically adjusted)
        self.base_factor_weights = {
            "gdp_growth": 0.25,
            "inflation_rate": 0.20,
            "interest_rate": 0.20,
            "unemployment_rate": 0.15,
            "consumer_confidence": 0.10,
            "manufacturing_pmi": 0.10
        }
        
        # Regime-specific factor weights
        self.regime_weights = {
            EconomicRegime.EXPANSION: {
                "gdp_growth": 0.30, "inflation_rate": 0.15, "interest_rate": 0.15,
                "unemployment_rate": 0.10, "consumer_confidence": 0.15, "manufacturing_pmi": 0.15
            },
            EconomicRegime.RECESSION: {
                "gdp_growth": 0.35, "inflation_rate": 0.10, "interest_rate": 0.25,
                "unemployment_rate": 0.20, "consumer_confidence": 0.05, "manufacturing_pmi": 0.05
            },
            EconomicRegime.RECOVERY: {
                "gdp_growth": 0.25, "inflation_rate": 0.15, "interest_rate": 0.20,
                "unemployment_rate": 0.15, "consumer_confidence": 0.12, "manufacturing_pmi": 0.13
            },
            EconomicRegime.STAGFLATION: {
                "gdp_growth": 0.20, "inflation_rate": 0.30, "interest_rate": 0.25,
                "unemployment_rate": 0.15, "consumer_confidence": 0.05, "manufacturing_pmi": 0.05
            },
            EconomicRegime.DEFLATION: {
                "gdp_growth": 0.30, "inflation_rate": 0.25, "interest_rate": 0.20,
                "unemployment_rate": 0.15, "consumer_confidence": 0.05, "manufacturing_pmi": 0.05
            }
        }
        
        # ML models for dynamic loading
        self.ridge_model = Ridge(alpha=1.0)
        self.elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Historical data for regime detection (simulated)
        self.historical_regimes = []
        
    def calculate(self, index_data: IndexData, macro_data: MacroeconomicData) -> MacroFactorResult:
        """Calculate advanced macroeconomic factor analysis with regime-switching"""
        try:
            # Detect current economic regime
            current_regime, regime_probabilities = self._detect_regime(macro_data)
            
            # Calculate dynamic factor loadings
            dynamic_loadings = self._calculate_dynamic_loadings(index_data, macro_data, current_regime)
            
            # Calculate factor loadings with regime adjustment
            factor_loadings = self._calculate_regime_adjusted_loadings(index_data, macro_data, current_regime)
            
            # Normalize factors
            normalized_factors = self._normalize_factors(macro_data)
            
            # Calculate factor scores with dynamic weights
            factor_scores = self._calculate_factor_scores(normalized_factors, factor_loadings)
            
            # Calculate base fair value
            base_fair_value = self._calculate_fair_value(index_data, factor_scores)
            
            # Apply regime adjustment to fair value
            regime_adjusted_fair_value = self._apply_regime_adjustment(base_fair_value, current_regime, regime_probabilities)
            
            # Calculate factor contributions
            factor_contributions = self._calculate_factor_contributions(normalized_factors, dynamic_loadings)
            
            # Generate signal with regime consideration
            signal = self._generate_regime_aware_signal(index_data.current_level, regime_adjusted_fair_value, current_regime)
            
            # Calculate confidence with regime uncertainty
            confidence = self._calculate_regime_adjusted_confidence(factor_scores, macro_data, regime_probabilities)
            
            # Assess risk level with regime consideration
            risk_level = self._assess_regime_adjusted_risk(macro_data, factor_scores, current_regime)
            
            # Generate comprehensive interpretation
            interpretation = self._generate_advanced_interpretation(
                signal, confidence, risk_level, factor_scores, current_regime, factor_contributions
            )
            
            return MacroFactorResult(
                 fair_value=base_fair_value,
                 confidence=confidence,
                 signal=signal,
                 risk_level=risk_level,
                 factor_scores=factor_scores,
                 interpretation=interpretation,
                 time_horizon="3-6 months",
                 current_regime=current_regime,
                 regime_probability=regime_probabilities,
                 dynamic_loadings=dynamic_loadings,
                 regime_adjusted_fair_value=regime_adjusted_fair_value,
                 factor_contributions=factor_contributions
             )
            
        except Exception as e:
             return MacroFactorResult(
                 fair_value=0.0,
                 confidence=0.0,
                 signal="HOLD",
                 risk_level="HIGH",
                 factor_scores={},
                 interpretation=f"Error in calculation: {str(e)}",
                 time_horizon="N/A",
                 current_regime=EconomicRegime.EXPANSION,
                 regime_probability={},
                 dynamic_loadings={},
                 regime_adjusted_fair_value=0.0,
                 factor_contributions={}
             )
    
    def _detect_regime(self, macro_data: MacroeconomicData) -> Tuple[EconomicRegime, Dict[str, float]]:
        """Detect current economic regime using macroeconomic indicators"""
        # Create regime indicators
        regime_indicators = RegimeIndicators(
            gdp_growth_trend=macro_data.gdp_growth,
            unemployment_trend=macro_data.unemployment_rate,
            inflation_trend=macro_data.inflation_rate,
            yield_curve_slope=max(0, macro_data.interest_rate - 2.0),  # Simplified yield curve
            credit_spreads=max(0, macro_data.interest_rate - 1.0),  # Simplified credit spreads
            volatility_regime=macro_data.vix
        )
        
        # Simple regime detection logic (in practice, would use more sophisticated methods)
        regime_scores = {
            EconomicRegime.EXPANSION: 0.0,
            EconomicRegime.RECESSION: 0.0,
            EconomicRegime.RECOVERY: 0.0,
            EconomicRegime.STAGFLATION: 0.0,
            EconomicRegime.DEFLATION: 0.0
        }
        
        # Expansion indicators
        if regime_indicators.gdp_growth_trend > 2.5:
            regime_scores[EconomicRegime.EXPANSION] += 0.3
        if regime_indicators.unemployment_trend < 4.0:
            regime_scores[EconomicRegime.EXPANSION] += 0.2
        if regime_indicators.volatility_regime < 20:
            regime_scores[EconomicRegime.EXPANSION] += 0.2
            
        # Recession indicators
        if regime_indicators.gdp_growth_trend < 0:
            regime_scores[EconomicRegime.RECESSION] += 0.4
        if regime_indicators.unemployment_trend > 6.0:
            regime_scores[EconomicRegime.RECESSION] += 0.3
        if regime_indicators.volatility_regime > 30:
            regime_scores[EconomicRegime.RECESSION] += 0.2
            
        # Stagflation indicators
        if regime_indicators.inflation_trend > 4.0 and regime_indicators.gdp_growth_trend < 2.0:
            regime_scores[EconomicRegime.STAGFLATION] += 0.5
            
        # Recovery indicators
        if 0 < regime_indicators.gdp_growth_trend < 2.5 and regime_indicators.unemployment_trend > 5.0:
            regime_scores[EconomicRegime.RECOVERY] += 0.4
            
        # Deflation indicators
        if regime_indicators.inflation_trend < 1.0:
            regime_scores[EconomicRegime.DEFLATION] += 0.3
            
        # Normalize scores to probabilities
        total_score = sum(regime_scores.values())
        if total_score == 0:
            # Default to expansion if no clear signals
            regime_probabilities = {regime.value: 0.2 for regime in EconomicRegime}
        else:
            regime_probabilities = {regime.value: score/total_score for regime, score in regime_scores.items()}
            
        # Select regime with highest probability
        current_regime = max(regime_scores.keys(), key=lambda x: regime_scores[x])
        
        return current_regime, regime_probabilities
    
    def _calculate_dynamic_loadings(self, index_data: IndexData, macro_data: MacroeconomicData, regime: EconomicRegime) -> Dict[str, float]:
        """Calculate dynamic factor loadings based on current regime"""
        base_loadings = {
            "gdp_growth": 2.5,
            "inflation_rate": -1.2,
            "interest_rate": -1.8,
            "unemployment_rate": -0.8,
            "consumer_confidence": 1.5,
            "manufacturing_pmi": 1.3
        }
        
        # Adjust loadings based on regime
        regime_adjustments = {
            EconomicRegime.EXPANSION: {"gdp_growth": 1.2, "consumer_confidence": 1.3},
            EconomicRegime.RECESSION: {"unemployment_rate": 1.5, "interest_rate": 1.2},
            EconomicRegime.STAGFLATION: {"inflation_rate": 1.4, "interest_rate": 1.3},
            EconomicRegime.RECOVERY: {"gdp_growth": 1.1, "manufacturing_pmi": 1.2},
            EconomicRegime.DEFLATION: {"inflation_rate": 1.3, "gdp_growth": 1.1}
        }
        
        dynamic_loadings = base_loadings.copy()
        if regime in regime_adjustments:
            for factor, adjustment in regime_adjustments[regime].items():
                if factor in dynamic_loadings:
                    dynamic_loadings[factor] *= adjustment
                    
        return dynamic_loadings
    
    def _calculate_regime_adjusted_loadings(self, index_data: IndexData, macro_data: MacroeconomicData, regime: EconomicRegime) -> Dict[str, float]:
        """Calculate factor loadings adjusted for current regime"""
        return self._calculate_dynamic_loadings(index_data, macro_data, regime)
    
    def _normalize_factors(self, macro_data: MacroeconomicData) -> Dict[str, float]:
        """Normalize macroeconomic factors to z-scores"""
        return {
            "gdp_growth": (macro_data.gdp_growth - 2.0) / 1.5,
            "inflation_rate": (macro_data.inflation_rate - 2.0) / 1.0,
            "interest_rate": (macro_data.interest_rate - 3.0) / 2.0,
            "unemployment_rate": (macro_data.unemployment_rate - 5.0) / 2.0,
            "consumer_confidence": (macro_data.consumer_confidence - 100) / 20,
            "manufacturing_pmi": (macro_data.manufacturing_pmi - 50) / 10
        }
    
    def _calculate_factor_scores(self, normalized_factors: Dict[str, float], factor_loadings: Dict[str, float]) -> Dict[str, float]:
        """Calculate factor scores"""
        factor_scores = {}
        for factor in normalized_factors:
            if factor in factor_loadings:
                factor_scores[factor] = normalized_factors[factor] * factor_loadings[factor]
        return factor_scores
    
    def _calculate_fair_value(self, index_data: IndexData, factor_scores: Dict[str, float]) -> float:
        """Calculate fair value based on factor scores"""
        total_score = sum(factor_scores.values())
        expected_return = total_score * 0.02  # 2% max impact per unit
        return index_data.current_level * (1 + expected_return)
    
    def _apply_regime_adjustment(self, base_fair_value: float, regime: EconomicRegime, regime_probabilities: Dict[str, float]) -> float:
        """Apply regime-specific adjustments to fair value"""
        regime_adjustments = {
            EconomicRegime.EXPANSION: 1.05,
            EconomicRegime.RECESSION: 0.90,
            EconomicRegime.RECOVERY: 1.02,
            EconomicRegime.STAGFLATION: 0.95,
            EconomicRegime.DEFLATION: 0.92
        }
        
        adjustment = regime_adjustments.get(regime, 1.0)
        return base_fair_value * adjustment
    
    def _calculate_factor_contributions(self, normalized_factors: Dict[str, float], dynamic_loadings: Dict[str, float]) -> Dict[str, float]:
        """Calculate individual factor contributions to fair value"""
        contributions = {}
        for factor in normalized_factors:
            if factor in dynamic_loadings:
                contributions[factor] = normalized_factors[factor] * dynamic_loadings[factor] * 0.02
        return contributions
    
    def _generate_regime_aware_signal(self, current_level: float, fair_value: float, regime: EconomicRegime) -> str:
        """Generate trading signal considering regime"""
        valuation_gap = (fair_value - current_level) / current_level
        
        # Adjust thresholds based on regime
        if regime == EconomicRegime.RECESSION:
            buy_threshold, sell_threshold = 0.08, -0.08
        elif regime == EconomicRegime.EXPANSION:
            buy_threshold, sell_threshold = 0.03, -0.03
        else:
            buy_threshold, sell_threshold = 0.05, -0.05
            
        if valuation_gap > buy_threshold:
            return "BUY"
        elif valuation_gap < sell_threshold:
            return "SELL"
        else:
            return "HOLD"
    
    def _calculate_regime_adjusted_confidence(self, factor_scores: Dict[str, float], macro_data: MacroeconomicData, regime_probabilities: Dict[str, float]) -> float:
        """Calculate confidence adjusted for regime uncertainty"""
        base_confidence = 1 - abs(sum(factor_scores.values())) / 10
        regime_uncertainty = 1 - max(regime_probabilities.values())
        return min(0.9, max(0.3, base_confidence * (1 - regime_uncertainty * 0.3)))
    
    def _assess_regime_adjusted_risk(self, macro_data: MacroeconomicData, factor_scores: Dict[str, float], regime: EconomicRegime) -> str:
        """Assess risk level considering regime"""
        base_risk = abs(sum(factor_scores.values()))
        
        if regime in [EconomicRegime.RECESSION, EconomicRegime.STAGFLATION]:
            risk_multiplier = 1.5
        elif regime == EconomicRegime.EXPANSION:
            risk_multiplier = 0.8
        else:
            risk_multiplier = 1.0
            
        adjusted_risk = base_risk * risk_multiplier
        
        if adjusted_risk < 3:
            return "Low"
        elif adjusted_risk < 6:
            return "Medium"
        else:
            return "High"
    
    def _generate_advanced_interpretation(self, signal: str, confidence: float, risk_level: str, 
                                        factor_scores: Dict[str, float], regime: EconomicRegime, 
                                        factor_contributions: Dict[str, float]) -> str:
        """Generate comprehensive interpretation"""
        total_contribution = sum(factor_contributions.values())
        top_factors = sorted(factor_contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        interpretation = f"Regime: {regime.value.title()}, Signal: {signal}, Confidence: {confidence:.2f}\n"
        interpretation += f"Expected return impact: {total_contribution:.2%}\n"
        interpretation += f"Top factors: {', '.join([f'{factor}: {contrib:.2%}' for factor, contrib in top_factors])}"
        
        return interpretation
    
    def analyze_factor_sensitivity(self, index_data: IndexData, macro_data: MacroeconomicData) -> Dict[str, float]:
        """Analyze sensitivity to individual macroeconomic factors with regime consideration"""
        base_result = self.calculate(index_data, macro_data)
        sensitivities = {}
        
        # Test sensitivity to each factor
        factor_changes = {
            "gdp_growth": 1.0,  # 1% change
            "inflation_rate": 1.0,
            "interest_rate": 1.0,
            "unemployment_rate": 1.0,
            "consumer_confidence": 10.0,
            "manufacturing_pmi": 5.0,
            "oil_price": 10.0,
            "dollar_index": 5.0,
            "vix": 5.0
        }
        
        for factor, change in factor_changes.items():
            # Create modified macro data
            modified_macro = MacroeconomicData(**macro_data.__dict__)
            if hasattr(modified_macro, factor):
                setattr(modified_macro, factor, getattr(modified_macro, factor) + change)
                
                # Calculate new result
                modified_result = self.calculate(index_data, modified_macro)
                
                # Calculate sensitivity
                if base_result.fair_value != 0:
                    sensitivity = (modified_result.regime_adjusted_fair_value - base_result.regime_adjusted_fair_value) / base_result.regime_adjusted_fair_value
                    original_value = getattr(macro_data, factor)
                    if original_value != 0:
                        sensitivities[factor] = sensitivity / (change / original_value)
                    else:
                        sensitivities[factor] = sensitivity
        
        return sensitivities
    
    def generate_regime_transition_analysis(self, index_data: IndexData, macro_data: MacroeconomicData) -> Dict[str, MacroFactorResult]:
        """Analyze potential regime transitions and their impact"""
        scenarios = {}
        
        # Current regime
        current_result = self.calculate(index_data, macro_data)
        scenarios["current"] = current_result
        
        # Simulate each regime
        for regime in EconomicRegime:
            if regime != current_result.current_regime:
                # Create modified macro data that would trigger this regime
                modified_macro = self._create_regime_scenario(macro_data, regime)
                scenarios[regime.value] = self.calculate(index_data, modified_macro)
        
        return scenarios
    
    def _create_regime_scenario(self, base_macro: MacroeconomicData, target_regime: EconomicRegime) -> MacroeconomicData:
        """Create macroeconomic scenario for specific regime"""
        modified_macro = MacroeconomicData(**base_macro.__dict__)
        
        if target_regime == EconomicRegime.RECESSION:
            modified_macro.gdp_growth = -1.0
            modified_macro.unemployment_rate = 7.5
            modified_macro.vix = 35.0
        elif target_regime == EconomicRegime.EXPANSION:
            modified_macro.gdp_growth = 3.5
            modified_macro.unemployment_rate = 3.5
            modified_macro.consumer_confidence = 115.0
        elif target_regime == EconomicRegime.STAGFLATION:
            modified_macro.gdp_growth = 1.0
            modified_macro.inflation_rate = 6.0
            modified_macro.unemployment_rate = 6.5
        elif target_regime == EconomicRegime.RECOVERY:
            modified_macro.gdp_growth = 1.5
            modified_macro.unemployment_rate = 6.0
            modified_macro.manufacturing_pmi = 55.0
        elif target_regime == EconomicRegime.DEFLATION:
            modified_macro.inflation_rate = 0.5
            modified_macro.gdp_growth = 0.5
            
        return modified_macro
    
    def generate_scenario_analysis(self, index_data: IndexData, macro_data: MacroeconomicData) -> Dict[str, MacroFactorResult]:
        """Generate comprehensive scenario analysis with regime-aware modeling"""
        scenarios = {}
        
        # Base case
        scenarios["base"] = self.calculate(index_data, macro_data)
        
        # Bull case - positive macro environment
        bull_macro = MacroeconomicData(
            gdp_growth=macro_data.gdp_growth + 1.0,
            inflation_rate=max(1.5, macro_data.inflation_rate - 0.5),
            unemployment_rate=max(3.0, macro_data.unemployment_rate - 1.0),
            interest_rate=macro_data.interest_rate,
            money_supply_growth=macro_data.money_supply_growth,
            government_debt_to_gdp=macro_data.government_debt_to_gdp,
            trade_balance=macro_data.trade_balance + 10,
            consumer_confidence=min(120, macro_data.consumer_confidence + 15),
            business_confidence=macro_data.business_confidence,
            manufacturing_pmi=min(60, macro_data.manufacturing_pmi + 5),
            services_pmi=macro_data.services_pmi,
            retail_sales_growth=macro_data.retail_sales_growth,
            industrial_production=macro_data.industrial_production,
            housing_starts=macro_data.housing_starts,
            oil_price=macro_data.oil_price,
            dollar_index=macro_data.dollar_index,
            vix=max(10, macro_data.vix - 5)
        )
        scenarios["bull"] = self.calculate(index_data, bull_macro)
        
        # Bear case - negative macro environment
        bear_macro = MacroeconomicData(
            gdp_growth=max(-2.0, macro_data.gdp_growth - 2.0),
            inflation_rate=macro_data.inflation_rate + 1.0,
            unemployment_rate=macro_data.unemployment_rate + 2.0,
            interest_rate=macro_data.interest_rate + 1.0,
            money_supply_growth=macro_data.money_supply_growth,
            government_debt_to_gdp=macro_data.government_debt_to_gdp,
            trade_balance=macro_data.trade_balance - 20,
            consumer_confidence=max(70, macro_data.consumer_confidence - 20),
            business_confidence=macro_data.business_confidence,
            manufacturing_pmi=max(40, macro_data.manufacturing_pmi - 8),
            services_pmi=macro_data.services_pmi,
            retail_sales_growth=macro_data.retail_sales_growth,
            industrial_production=macro_data.industrial_production,
            housing_starts=macro_data.housing_starts,
            oil_price=macro_data.oil_price,
            dollar_index=macro_data.dollar_index,
            vix=macro_data.vix + 10
        )
        scenarios["bear"] = self.calculate(index_data, bear_macro)
        
        # Add regime transition scenarios
        regime_scenarios = self.generate_regime_transition_analysis(index_data, macro_data)
        scenarios.update(regime_scenarios)
        
        return scenarios

# Example usage
if __name__ == "__main__":
    # Sample data
    sample_index = IndexData(
        symbol="SPX",
        name="S&P 500",
        current_level=4200.0,
        historical_levels=[4000, 4050, 4100, 4150, 4200],
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
    
    sample_macro = MacroeconomicData(
        gdp_growth=2.5,
        inflation_rate=3.2,
        unemployment_rate=3.8,
        interest_rate=5.25,
        money_supply_growth=8.5,
        government_debt_to_gdp=120.0,
        trade_balance=-50.0,
        consumer_confidence=105.0,
        business_confidence=95.0,
        manufacturing_pmi=52.0,
        services_pmi=54.0,
        retail_sales_growth=4.2,
        industrial_production=2.8,
        housing_starts=1.4,
        oil_price=75.0,
        dollar_index=103.0,
        vix=18.5
    )
    
    # Create model and calculate
    model = MacroeconomicFactorModel()
    result = model.calculate(sample_index, sample_macro)
    
    print(f"Fair Value: {result.fair_value:.0f}")
    print(f"Regime-Adjusted Fair Value: {result.regime_adjusted_fair_value:.0f}")
    print(f"Current Regime: {result.current_regime.value}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Signal: {result.signal}")
    print(f"Interpretation: {result.interpretation}")
    
    # Regime probabilities
    print("\nRegime Probabilities:")
    for regime, prob in result.regime_probability.items():
        print(f"{regime}: {prob:.3f}")
    
    # Factor contributions
    print("\nFactor Contributions:")
    for factor, contribution in result.factor_contributions.items():
        print(f"{factor}: {contribution:.3%}")
    
    # Sensitivity analysis
    sensitivities = model.analyze_factor_sensitivity(sample_index, sample_macro)
    print("\nFactor Sensitivities:")
    for factor, sensitivity in sensitivities.items():
        print(f"{factor}: {sensitivity:.3f}")
    
    # Scenario analysis
    scenarios = model.generate_scenario_analysis(sample_index, sample_macro)
    print("\nScenario Analysis:")
    for scenario_name, scenario_result in scenarios.items():
        if hasattr(scenario_result, 'regime_adjusted_fair_value'):
            print(f"{scenario_name}: {scenario_result.regime_adjusted_fair_value:.0f} ({scenario_result.signal})")
        else:
            print(f"{scenario_name}: {scenario_result.fair_value:.0f} ({scenario_result.signal})")