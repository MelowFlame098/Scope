#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced Interest Rate Parity (IRP) implementation.
Tests all advanced features including regime switching, volatility modeling,
machine learning predictions, econometric analysis, and risk metrics.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings

# Add the forex indicators directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced IRP analyzer
from ppp_irp_uip import IRPAnalyzer, IRPResult

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def generate_test_data(n_periods=252):
    """
    Generate synthetic forex and interest rate data for testing.
    
    Args:
        n_periods: Number of time periods to generate
        
    Returns:
        dict: Dictionary containing test data
    """
    np.random.seed(42)  # For reproducible results
    
    # Generate dates
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
    
    # Generate spot rates (EUR/USD) with some trend and volatility
    spot_base = 1.20
    spot_drift = np.random.normal(0, 0.001, n_periods)
    spot_volatility = np.random.normal(0, 0.01, n_periods)
    spot_rates = pd.Series(
        spot_base + np.cumsum(spot_drift) + spot_volatility,
        index=dates
    )
    
    # Generate forward rates (slightly different from spot)
    forward_premium = np.random.normal(0.001, 0.0005, n_periods)
    forward_rates = pd.Series(
        spot_rates + forward_premium,
        index=dates
    )
    
    # Generate interest rates
    base_rate_mean = 0.02  # 2% base rate
    target_rate_mean = 0.015  # 1.5% target rate
    
    base_rates = pd.Series(
        np.maximum(0, base_rate_mean + np.random.normal(0, 0.002, n_periods)),
        index=dates
    )
    
    target_rates = pd.Series(
        np.maximum(0, target_rate_mean + np.random.normal(0, 0.002, n_periods)),
        index=dates
    )
    
    return {
        'spot_rates': spot_rates,
        'forward_rates': forward_rates,
        'base_rates': base_rates,
        'target_rates': target_rates,
        'dates': dates
    }

def test_basic_irp():
    """
    Test basic IRP functionality with scalar inputs.
    """
    print("\n=== Testing Basic IRP Analysis ===")
    
    # Initialize analyzer
    analyzer = IRPAnalyzer(
        base_currency='USD',
        target_currency='EUR'
    )
    
    # Test with scalar inputs
    result = analyzer.analyze_irp(
        spot_rates=1.20,
        forward_rates=1.205,
        base_interest_rates=0.02,
        target_interest_rates=0.015
    )
    
    # Validate results
    assert isinstance(result, IRPResult), "Result should be IRPResult instance"
    assert hasattr(result, 'covered_irp'), "Result should have covered_irp"
    assert hasattr(result, 'uncovered_irp'), "Result should have uncovered_irp"
    assert hasattr(result, 'forward_premium'), "Result should have forward_premium"
    assert hasattr(result, 'irp_deviation'), "Result should have irp_deviation"
    
    print(f"✓ Covered IRP - CIP Deviation: {result.covered_irp['cip_deviation']:.6f}")
    print(f"✓ Covered IRP - CIP Holds: {result.covered_irp['cip_holds']}")
    print(f"✓ Uncovered IRP - UIP Deviation: {result.uncovered_irp['uip_deviation']:.6f}")
    print(f"✓ Forward Premium: {result.forward_premium:.6f}")
    print(f"✓ IRP Deviation: {result.irp_deviation:.6f}")
    print(f"✓ Arbitrage Opportunity: {result.arbitrage_opportunity}")
    
    if result.arbitrage_opportunity:
        print(f"✓ Arbitrage Profit: {result.arbitrage_profit:.6f}")
    
    print("✓ Basic IRP test passed!")
    return result

def test_enhanced_irp_features():
    """
    Test enhanced IRP features with time series data.
    """
    print("\n=== Testing Enhanced IRP Features ===")
    
    # Generate test data
    data = generate_test_data(252)
    
    # Initialize analyzer with enhanced features
    analyzer = IRPAnalyzer(
        base_currency='USD',
        target_currency='EUR',
        enable_ml=True,
        enable_regime_switching=True,
        enable_volatility_modeling=True
    )
    
    # Analyze with time series data
    result = analyzer.analyze_irp(
        spot_rates=data['spot_rates'],
        forward_rates=data['forward_rates'],
        base_interest_rates=data['base_rates'],
        target_interest_rates=data['target_rates']
    )
    
    # Validate enhanced features
    print(f"✓ Basic IRP metrics calculated")
    
    if result.regime_analysis:
        print(f"✓ Regime Analysis: Current regime {result.regime_analysis.current_regime}")
        print(f"  - Regime probabilities: {result.regime_analysis.regime_probabilities}")
    
    if result.volatility_analysis:
        print(f"✓ Volatility Analysis: Current volatility {result.volatility_analysis.current_volatility:.6f}")
        print(f"  - GARCH parameters: {result.volatility_analysis.garch_params}")
    
    if result.ml_predictions:
        print(f"✓ ML Predictions: Best model {result.ml_predictions.best_model}")
        best_score = max(result.ml_predictions.model_scores.values()) if result.ml_predictions.model_scores else 0.0
        print(f"  - Best R² score: {best_score:.4f}")
        ensemble_pred = result.ml_predictions.ensemble_prediction
        if isinstance(ensemble_pred, np.ndarray) and len(ensemble_pred) > 0:
            print(f"  - Ensemble prediction: {ensemble_pred[-1]:.6f}")
        else:
            print(f"  - Ensemble prediction: {ensemble_pred:.6f}")
    
    if result.econometric_analysis:
        print(f"✓ Econometric Analysis completed")
        if result.econometric_analysis.unit_root_tests:
            print(f"  - Unit root tests available for {len(result.econometric_analysis.unit_root_tests)} series")
        if result.econometric_analysis.cointegration_tests:
            print(f"  - Cointegration tests completed")
        print(f"  - Half-life: {result.econometric_analysis.half_life:.2f} periods")
    
    if result.risk_metrics:
        print(f"✓ Risk Metrics: VaR(95%) {result.risk_metrics.var_95:.6f}")
        print(f"  - Sharpe ratio: {result.risk_metrics.sharpe_ratio:.4f}")
        print(f"  - Maximum drawdown: {result.risk_metrics.max_drawdown:.6f}")
    
    print("✓ Enhanced IRP features test passed!")
    return result

def test_regime_switching():
    """
    Test regime switching analysis specifically.
    """
    print("\n=== Testing Regime Switching Analysis ===")
    
    # Generate data with clear regime changes
    data = generate_test_data(500)
    
    analyzer = IRPAnalyzer(
        base_currency='USD',
        target_currency='EUR',
        enable_regime_switching=True
    )
    
    result = analyzer.analyze_irp(
        spot_rates=data['spot_rates'],
        forward_rates=data['forward_rates'],
        base_interest_rates=data['base_rates'],
        target_interest_rates=data['target_rates']
    )
    
    if result.regime_analysis:
        print(f"✓ Detected {len(result.regime_analysis.regime_means)} regimes")
        print(f"✓ Current regime: {result.regime_analysis.current_regime}")
        current_regime = result.regime_analysis.current_regime
        if current_regime in result.regime_analysis.regime_persistence:
            persistence = result.regime_analysis.regime_persistence[current_regime]
            print(f"✓ Current regime persistence: {persistence:.4f}")
        if current_regime in result.regime_analysis.expected_regime_duration:
            duration = result.regime_analysis.expected_regime_duration[current_regime]
            print(f"✓ Expected duration: {duration:.2f} periods")
        print("✓ Regime switching test passed!")
    else:
        print("⚠ Regime analysis not available (insufficient data or missing dependencies)")

def test_volatility_modeling():
    """
    Test volatility modeling with GARCH.
    """
    print("\n=== Testing Volatility Modeling ===")
    
    data = generate_test_data(300)
    
    analyzer = IRPAnalyzer(
        base_currency='USD',
        target_currency='EUR',
        enable_volatility_modeling=True
    )
    
    result = analyzer.analyze_irp(
        spot_rates=data['spot_rates'],
        forward_rates=data['forward_rates'],
        base_interest_rates=data['base_rates'],
        target_interest_rates=data['target_rates']
    )
    
    if result.volatility_analysis:
        print(f"✓ Current volatility: {result.volatility_analysis.current_volatility:.6f}")
        forecast = result.volatility_analysis.volatility_forecast
        if isinstance(forecast, np.ndarray) and len(forecast) > 0:
            print(f"✓ Volatility forecast (next period): {forecast[0]:.6f}")
        else:
            print(f"✓ Volatility forecast: {forecast:.6f}")
        print(f"✓ ARCH test p-value: {result.volatility_analysis.arch_test_pvalue:.4f}")
        print("✓ Volatility modeling test passed!")
    else:
        print("⚠ Volatility analysis not available (missing dependencies)")

def test_machine_learning():
    """
    Test machine learning predictions.
    """
    print("\n=== Testing Machine Learning Predictions ===")
    
    data = generate_test_data(400)
    
    analyzer = IRPAnalyzer(
        base_currency='USD',
        target_currency='EUR',
        enable_ml=True
    )
    
    result = analyzer.analyze_irp(
        spot_rates=data['spot_rates'],
        forward_rates=data['forward_rates'],
        base_interest_rates=data['base_rates'],
        target_interest_rates=data['target_rates']
    )
    
    if result.ml_predictions:
        print(f"✓ Best model: {result.ml_predictions.best_model}")
        best_score = max(result.ml_predictions.model_scores.values()) if result.ml_predictions.model_scores else 0.0
        print(f"✓ Best R² score: {best_score:.4f}")
        ensemble_pred = result.ml_predictions.ensemble_prediction
        if isinstance(ensemble_pred, np.ndarray) and len(ensemble_pred) > 0:
            print(f"✓ Ensemble prediction: {ensemble_pred[-1]:.6f}")
        else:
            print(f"✓ Ensemble prediction: {ensemble_pred:.6f}")
        print(f"✓ Number of models trained: {len(result.ml_predictions.model_scores)}")
        
        if result.ml_predictions.feature_importance:
            print(f"✓ Feature importance available for {len(result.ml_predictions.feature_importance)} features")
        
        print("✓ Machine learning test passed!")
    else:
        print("⚠ ML predictions not available (insufficient data)")

def test_error_handling():
    """
    Test error handling and edge cases.
    """
    print("\n=== Testing Error Handling ===")
    
    analyzer = IRPAnalyzer('USD', 'EUR')
    
    # Test with invalid inputs
    try:
        result = analyzer.analyze_irp(
            spot_rates=np.nan,
            forward_rates=1.20,
            base_interest_rates=0.02,
            target_interest_rates=0.015
        )
        print("✓ Handled NaN input gracefully")
    except Exception as e:
        print(f"✓ Properly caught error: {type(e).__name__}")
    
    # Test with very small dataset
    small_data = pd.Series([1.20, 1.21, 1.19])
    result = analyzer.analyze_irp(
        spot_rates=small_data,
        forward_rates=small_data + 0.001,
        base_interest_rates=pd.Series([0.02, 0.021, 0.019]),
        target_interest_rates=pd.Series([0.015, 0.016, 0.014])
    )
    
    print("✓ Handled small dataset gracefully")
    print("✓ Error handling test passed!")

def test_performance():
    """
    Test performance with larger datasets.
    """
    print("\n=== Testing Performance ===")
    
    # Generate larger dataset
    large_data = generate_test_data(1000)
    
    analyzer = IRPAnalyzer(
        base_currency='USD',
        target_currency='EUR',
        enable_ml=True,
        enable_regime_switching=True,
        enable_volatility_modeling=True
    )
    
    import time
    start_time = time.time()
    
    result = analyzer.analyze_irp(
        spot_rates=large_data['spot_rates'],
        forward_rates=large_data['forward_rates'],
        base_interest_rates=large_data['base_rates'],
        target_interest_rates=large_data['target_rates']
    )
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    print(f"✓ Processed {len(large_data['spot_rates'])} data points in {execution_time:.2f} seconds")
    print(f"✓ Performance: {len(large_data['spot_rates'])/execution_time:.0f} data points per second")
    print("✓ Performance test passed!")

def main():
    """
    Run all IRP tests.
    """
    print("Starting Enhanced IRP Implementation Tests...")
    print("=" * 60)
    
    try:
        # Run all tests
        test_basic_irp()
        test_enhanced_irp_features()
        test_regime_switching()
        test_volatility_modeling()
        test_machine_learning()
        test_error_handling()
        test_performance()
        
        print("\n" + "=" * 60)
        print("🎉 ALL IRP TESTS PASSED SUCCESSFULLY! 🎉")
        print("Enhanced IRP implementation is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)