#!/usr/bin/env python3
"""
Simple test for Term Structure Model
"""

from term_structure_model import AdvancedTermStructureModel, MacroeconomicData, IndexData
from datetime import datetime

def test_basic_functionality():
    """Test basic model functionality"""
    print("Testing Term Structure Model...")
    
    # Create model with minimal features
    model = AdvancedTermStructureModel(
        model_type="nelson_siegel_svensson",
        enable_volatility_surface=False,  # Disable complex features
        enable_pca_analysis=False
    )
    
    # Sample data
    macro_data = MacroeconomicData(
        gdp_growth=2.5,
        inflation_rate=2.0,
        unemployment_rate=4.0,
        interest_rate=4.5,
        money_supply_growth=5.0,
        government_debt_to_gdp=100.0,
        trade_balance=-50.0,
        consumer_confidence=110.0,
        business_confidence=105.0,
        manufacturing_pmi=52.0,
        services_pmi=54.0,
        retail_sales_growth=3.0,
        industrial_production=1.5,
        housing_starts=1.2,
        oil_price=75.0,
        dollar_index=102.0,
        vix=18.0
    )
    
    index_data = IndexData(
        symbol="SPY",
        name="SPDR S&P 500 ETF",
        current_level=450.0,
        historical_levels=[440.0, 445.0, 450.0],
        dividend_yield=1.5,
        pe_ratio=20.0,
        pb_ratio=3.5,
        market_cap=400000000000.0,
        volatility=0.15,
        beta=1.0,
        sector_weights={"Technology": 0.3, "Healthcare": 0.15, "Financials": 0.12},
        constituent_count=500,
        volume=50000000.0
    )
    
    try:
        print("Calling model.calculate()...")
        result = model.calculate(index_data, macro_data)
        
        print(f"Fair Value: {result.fair_value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Signal: {result.signal}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Interpretation: {result.interpretation}")
        
        print("\nTest completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during calculation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        print("\n✅ Term Structure Model test passed!")
    else:
        print("\n❌ Term Structure Model test failed!")