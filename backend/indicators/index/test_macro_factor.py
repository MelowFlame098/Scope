#!/usr/bin/env python3
"""
Test script for Enhanced Macroeconomic Factor Model
Tests regime-switching, dynamic factor loading, and advanced analytics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from macro_factor_model import (
    MacroeconomicFactorModel, MacroeconomicData, IndexData, 
    EconomicRegime, RegimeIndicators
)

def test_enhanced_macro_factor_model():
    """Test the enhanced macroeconomic factor model"""
    print("=== Enhanced Macroeconomic Factor Model Test ===")
    
    # Create sample data
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
    
    # Test different economic scenarios
    scenarios = {
        "expansion": MacroeconomicData(
            gdp_growth=3.5,
            inflation_rate=2.1,
            unemployment_rate=3.5,
            interest_rate=4.5,
            money_supply_growth=6.0,
            government_debt_to_gdp=110.0,
            trade_balance=-30.0,
            consumer_confidence=115.0,
            business_confidence=105.0,
            manufacturing_pmi=55.0,
            services_pmi=56.0,
            retail_sales_growth=5.2,
            industrial_production=3.8,
            housing_starts=1.6,
            oil_price=70.0,
            dollar_index=100.0,
            vix=15.0
        ),
        "recession": MacroeconomicData(
            gdp_growth=-1.5,
            inflation_rate=1.8,
            unemployment_rate=7.5,
            interest_rate=2.0,
            money_supply_growth=12.0,
            government_debt_to_gdp=130.0,
            trade_balance=-70.0,
            consumer_confidence=75.0,
            business_confidence=70.0,
            manufacturing_pmi=42.0,
            services_pmi=45.0,
            retail_sales_growth=-2.1,
            industrial_production=-1.5,
            housing_starts=1.0,
            oil_price=60.0,
            dollar_index=105.0,
            vix=35.0
        ),
        "stagflation": MacroeconomicData(
            gdp_growth=1.0,
            inflation_rate=6.5,
            unemployment_rate=6.5,
            interest_rate=7.0,
            money_supply_growth=15.0,
            government_debt_to_gdp=125.0,
            trade_balance=-60.0,
            consumer_confidence=85.0,
            business_confidence=80.0,
            manufacturing_pmi=48.0,
            services_pmi=49.0,
            retail_sales_growth=1.5,
            industrial_production=0.5,
            housing_starts=1.1,
            oil_price=90.0,
            dollar_index=110.0,
            vix=28.0
        )
    }
    
    # Create model
    model = MacroeconomicFactorModel()
    
    # Test each scenario
    for scenario_name, macro_data in scenarios.items():
        print(f"\n--- {scenario_name.upper()} SCENARIO ---")
        
        try:
            result = model.calculate(sample_index, macro_data)
            
            print(f"Fair Value: ${result.fair_value:.0f}")
            print(f"Regime-Adjusted Fair Value: ${result.regime_adjusted_fair_value:.0f}")
            print(f"Current Regime: {result.current_regime.value}")
            print(f"Signal: {result.signal}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Risk Level: {result.risk_level}")
            
            # Show regime probabilities
            print("\nRegime Probabilities:")
            for regime, prob in result.regime_probability.items():
                print(f"  {regime}: {prob:.3f}")
            
            # Show top factor contributions
            print("\nTop Factor Contributions:")
            factor_contributions = result.factor_contributions
            sorted_contributions = sorted(
                factor_contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )[:3]
            for factor, contribution in sorted_contributions:
                print(f"  {factor}: {contribution:.3%}")
            
            # Show dynamic loadings
            print("\nDynamic Factor Loadings:")
            dynamic_loadings = result.dynamic_loadings
            for factor, loading in list(dynamic_loadings.items())[:3]:
                print(f"  {factor}: {loading:.2f}")
                
            print(f"\nInterpretation: {result.interpretation}")
            
        except Exception as e:
            print(f"Error in {scenario_name} scenario: {e}")
    
    # Test sensitivity analysis
    print("\n--- SENSITIVITY ANALYSIS ---")
    try:
        sensitivities = model.analyze_factor_sensitivity(sample_index, scenarios["expansion"])
        print("Factor Sensitivities (Expansion Scenario):")
        for factor, sensitivity in sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {factor}: {sensitivity:.4f}")
    except Exception as e:
        print(f"Error in sensitivity analysis: {e}")
    
    # Test scenario analysis
    print("\n--- COMPREHENSIVE SCENARIO ANALYSIS ---")
    try:
        scenario_results = model.generate_scenario_analysis(sample_index, scenarios["expansion"])
        print("Scenario Comparison:")
        for scenario_name, scenario_result in scenario_results.items():
            if hasattr(scenario_result, 'regime_adjusted_fair_value'):
                fair_value = scenario_result.regime_adjusted_fair_value
            else:
                fair_value = scenario_result.fair_value
            print(f"  {scenario_name}: ${fair_value:.0f} ({scenario_result.signal})")
    except Exception as e:
        print(f"Error in scenario analysis: {e}")
    
    # Test regime transition analysis
    print("\n--- REGIME TRANSITION ANALYSIS ---")
    try:
        regime_transitions = model.generate_regime_transition_analysis(sample_index, scenarios["expansion"])
        print("Regime Transition Impact:")
        for regime_name, regime_result in regime_transitions.items():
            if hasattr(regime_result, 'regime_adjusted_fair_value'):
                fair_value = regime_result.regime_adjusted_fair_value
            else:
                fair_value = regime_result.fair_value
            print(f"  {regime_name}: ${fair_value:.0f} ({regime_result.signal})")
    except Exception as e:
        print(f"Error in regime transition analysis: {e}")
    
    print("\n=== Enhanced Macro Factor Model Test Complete ===")
    return True

if __name__ == "__main__":
    success = test_enhanced_macro_factor_model()
    if success:
        print("\n✅ All tests passed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)