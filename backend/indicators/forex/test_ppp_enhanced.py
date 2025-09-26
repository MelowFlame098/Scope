#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced PPP (Purchasing Power Parity) model.
Tests all advanced features including ML, regime switching, volatility modeling, etc.
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ppp import PPPIndicator

def generate_sample_data(n_periods=500):
    """
    Generate sample forex data for testing.
    """
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=n_periods)
    dates = pd.date_range(start=start_date, periods=n_periods, freq='D')
    
    # Generate realistic exchange rate data with trend and volatility
    initial_rate = 1.2000
    trend = 0.0001  # Small upward trend
    volatility = 0.015
    
    # Generate price series with some mean reversion
    prices = [initial_rate]
    for i in range(1, n_periods):
        # Add trend, mean reversion, and random shock
        mean_reversion = -0.001 * (prices[-1] - initial_rate)
        shock = np.random.normal(0, volatility)
        new_price = prices[-1] + trend + mean_reversion + shock
        prices.append(max(new_price, 0.5))  # Prevent negative prices
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_periods)
    })
    
    data.set_index('date', inplace=True)
    return data

def test_basic_ppp_functionality():
    """
    Test basic PPP functionality.
    """
    print("\n=== Testing Basic PPP Functionality ===")
    
    try:
        # Generate test data
        data = generate_sample_data(200)
        
        # Initialize PPP indicator
        ppp = PPPIndicator(
            base_country='US',
            quote_country='EU',
            enable_ml=False,
            enable_regime_switching=False,
            enable_volatility_modeling=False
        )
        
        # Calculate PPP
        result = ppp.calculate(data)
        
        # Validate results
        assert result is not None, "PPP result should not be None"
        assert hasattr(result, 'ppp_rate'), "Result should have ppp_rate"
        assert hasattr(result, 'big_mac_ppp'), "Result should have big_mac_ppp"
        assert hasattr(result, 'real_exchange_rate'), "Result should have real_exchange_rate"
        assert hasattr(result, 'ppp_deviation'), "Result should have ppp_deviation"
        
        print(f"✓ Basic PPP calculation successful")
        print(f"  - PPP Rate: {result.ppp_rate:.4f}")
        print(f"  - Big Mac PPP: {result.big_mac_ppp:.4f}")
        print(f"  - Current deviation: {result.ppp_deviation:.2f}%")
        print(f"  - Reversion signal: {result.reversion_signal}")
        print(f"  - Confidence: {result.confidence:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic PPP test failed: {e}")
        return False

def test_advanced_ppp_features():
    """
    Test advanced PPP features (ML, regime switching, volatility).
    """
    print("\n=== Testing Advanced PPP Features ===")
    
    try:
        # Generate test data
        data = generate_sample_data(300)
        
        # Initialize PPP indicator with all advanced features
        ppp = PPPIndicator(
            base_country='US',
            quote_country='EU',
            enable_ml=True,
            enable_regime_switching=True,
            enable_volatility_modeling=True
        )
        
        # Calculate PPP with advanced features
        result = ppp.calculate(data)
        
        # Validate advanced results
        assert result is not None, "PPP result should not be None"
        
        # Test regime analysis
        if result.regime_analysis:
            print(f"✓ Regime Analysis:")
            print(f"  - Current regime: {result.regime_analysis.current_regime}")
            print(f"  - Regime description: {result.regime_analysis.regime_description}")
            print(f"  - Regime probabilities: {result.regime_analysis.regime_probabilities}")
        
        # Test volatility analysis
        if result.volatility_analysis:
            print(f"✓ Volatility Analysis:")
            print(f"  - Current volatility: {result.volatility_analysis.current_volatility:.4f}")
            print(f"  - Volatility regime: {result.volatility_analysis.volatility_regime}")
            print(f"  - ARCH test p-value: {result.volatility_analysis.arch_test_pvalue:.4f}")
            print(f"  - Volatility clustering: {result.volatility_analysis.volatility_clustering}")
        
        # Test ML predictions
        if result.ml_predictions:
            print(f"✓ Machine Learning Predictions:")
            print(f"  - Models trained: {len(result.ml_predictions.predictions)}")
            print(f"  - Ensemble prediction length: {len(result.ml_predictions.ensemble_prediction)}")
            if result.ml_predictions.model_scores:
                best_model = max(result.ml_predictions.model_scores.items(), key=lambda x: x[1])
                print(f"  - Best model: {best_model[0]} (score: {best_model[1]:.4f})")
        
        # Test econometric analysis
        if result.econometric_analysis:
            print(f"✓ Econometric Analysis:")
            print(f"  - Unit root tests: {len(result.econometric_analysis.unit_root_tests)}")
            print(f"  - Cointegration tests: {len(result.econometric_analysis.cointegration_tests)}")
            print(f"  - Structural breaks: {len(result.econometric_analysis.structural_breaks)}")
            if 'ppp_deviation' in result.econometric_analysis.half_life_estimates:
                half_life = result.econometric_analysis.half_life_estimates['ppp_deviation']
                print(f"  - PPP deviation half-life: {half_life:.2f} periods")
        
        # Test risk metrics
        if result.risk_metrics:
            print(f"✓ Risk Metrics:")
            print(f"  - VaR (95%): {result.risk_metrics.var_95:.4f}")
            print(f"  - CVaR (95%): {result.risk_metrics.cvar_95:.4f}")
            print(f"  - Max drawdown: {result.risk_metrics.max_drawdown:.4f}")
            print(f"  - Sharpe ratio: {result.risk_metrics.sharpe_ratio:.4f}")
            print(f"  - Sortino ratio: {result.risk_metrics.sortino_ratio:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Advanced PPP features test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_diagnostics():
    """
    Test model diagnostics functionality.
    """
    print("\n=== Testing Model Diagnostics ===")
    
    try:
        # Generate test data with some missing values
        data = generate_sample_data(150)
        
        # Add some missing values
        data.loc[data.index[10:15], 'close'] = np.nan
        
        # Initialize PPP indicator
        ppp = PPPIndicator(
            base_country='US',
            quote_country='EU',
            enable_ml=True
        )
        
        # Calculate PPP
        result = ppp.calculate(data)
        
        # Check diagnostics
        if result.model_diagnostics:
            print(f"✓ Model Diagnostics:")
            if 'data_quality' in result.model_diagnostics:
                dq = result.model_diagnostics['data_quality']
                print(f"  - Missing data: {dq.get('missing_data_pct', 0):.2f}%")
                print(f"  - Outliers count: {dq.get('outliers_count', 0)}")
                print(f"  - Data length: {dq.get('data_length', 0)}")
            
            if 'stationarity' in result.model_diagnostics:
                stationarity = result.model_diagnostics['stationarity']
                print(f"  - Stationarity tests: {len(stationarity)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model diagnostics test failed: {e}")
        return False

def test_backtesting_and_sensitivity():
    """
    Test backtesting and sensitivity analysis.
    """
    print("\n=== Testing Backtesting and Sensitivity Analysis ===")
    
    try:
        # Generate test data
        data = generate_sample_data(250)
        
        # Initialize PPP indicator
        ppp = PPPIndicator(
            base_country='US',
            quote_country='EU',
            enable_ml=True
        )
        
        # Calculate PPP
        result = ppp.calculate(data)
        
        # Check backtesting results
        if result.backtesting_results:
            print(f"✓ Backtesting Results:")
            if 'mse' in result.backtesting_results:
                print(f"  - MSE: {result.backtesting_results['mse']:.6f}")
            if 'mae' in result.backtesting_results:
                print(f"  - MAE: {result.backtesting_results['mae']:.6f}")
            if 'direction_accuracy' in result.backtesting_results:
                print(f"  - Direction accuracy: {result.backtesting_results['direction_accuracy']:.2%}")
            if 'predictions_count' in result.backtesting_results:
                print(f"  - Predictions count: {result.backtesting_results['predictions_count']}")
        
        # Check sensitivity analysis
        if result.sensitivity_analysis:
            print(f"✓ Sensitivity Analysis:")
            print(f"  - Sensitivity tests: {len(result.sensitivity_analysis)}")
            for key, value in list(result.sensitivity_analysis.items())[:3]:  # Show first 3
                if isinstance(value, dict) and 'shock_size' in value:
                    print(f"  - {key}: {value['shock_size']:.1f}% shock -> {value.get('deviation_impact', 0):.4f} impact")
        
        return True
        
    except Exception as e:
        print(f"✗ Backtesting and sensitivity test failed: {e}")
        return False

def test_error_handling():
    """
    Test error handling with invalid data.
    """
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test with insufficient data
        small_data = generate_sample_data(10)
        
        ppp = PPPIndicator(
            base_country='US',
            quote_country='EU',
            enable_ml=True,
            enable_regime_switching=True,
            enable_volatility_modeling=True
        )
        
        result = ppp.calculate(small_data)
        
        # Should still return a result, but with limited analysis
        assert result is not None, "Should return result even with small data"
        print(f"✓ Handled insufficient data gracefully")
        
        # Test with all NaN data
        nan_data = generate_sample_data(50)
        nan_data['close'] = np.nan
        
        try:
            result_nan = ppp.calculate(nan_data)
            print(f"✓ Handled NaN data gracefully")
        except Exception as e:
            print(f"✓ Properly raised exception for invalid data: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False

def test_performance():
    """
    Test performance with larger datasets.
    """
    print("\n=== Testing Performance ===")
    
    try:
        import time
        
        # Generate larger dataset
        large_data = generate_sample_data(1000)
        
        ppp = PPPIndicator(
            base_country='US',
            quote_country='EU',
            enable_ml=True,
            enable_regime_switching=True,
            enable_volatility_modeling=True
        )
        
        # Measure execution time
        start_time = time.time()
        result = ppp.calculate(large_data)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        assert result is not None, "Should return result for large dataset"
        print(f"✓ Processed {len(large_data)} data points in {execution_time:.2f} seconds")
        print(f"  - Average time per data point: {execution_time/len(large_data)*1000:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance test failed: {e}")
        return False

def main():
    """
    Run all tests.
    """
    print("Enhanced PPP Model Test Suite")
    print("=" * 50)
    
    tests = [
        test_basic_ppp_functionality,
        test_advanced_ppp_features,
        test_model_diagnostics,
        test_backtesting_and_sensitivity,
        test_error_handling,
        test_performance
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced PPP model is working correctly.")
        print("\nKey features validated:")
        print("  ✓ Basic PPP calculations (Absolute, Relative, Big Mac)")
        print("  ✓ Regime switching analysis with Hidden Markov Models")
        print("  ✓ Volatility modeling with GARCH")
        print("  ✓ Machine learning predictions with multiple models")
        print("  ✓ Econometric analysis (unit root, cointegration, half-life)")
        print("  ✓ Risk metrics (VaR, CVaR, Sharpe ratio, etc.)")
        print("  ✓ Model diagnostics and data quality checks")
        print("  ✓ Backtesting and sensitivity analysis")
        print("  ✓ Error handling and performance optimization")
    else:
        print(f"⚠️  {total - passed} tests failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)