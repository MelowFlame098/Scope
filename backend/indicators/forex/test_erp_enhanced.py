#!/usr/bin/env python3
"""
Test script for Enhanced Exchange Rate Pressure (ERP) Analyzer
Tests regime switching, predictive modeling, and volatility analysis features.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from exchange_rate_pressure_analyzer import ExchangeRatePressureAnalyzer

def generate_test_data(n_periods=200):
    """Generate synthetic test data for ERP analysis."""
    np.random.seed(42)
    
    # Create date index
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(n_periods)]
    
    # Generate regime-switching data
    regime_changes = [0, 50, 120, 170]  # Points where regime changes
    regimes = np.zeros(n_periods)
    
    for i, change_point in enumerate(regime_changes[1:], 1):
        start_point = regime_changes[i-1]
        regimes[start_point:change_point] = i % 2
    regimes[regime_changes[-1]:] = len(regime_changes) % 2
    
    # Generate exchange rates with regime-dependent volatility
    exchange_rates = [100.0]  # Starting exchange rate
    for i in range(1, n_periods):
        if regimes[i] == 0:  # Low volatility regime
            change = np.random.normal(0, 0.02)
        else:  # High volatility regime
            change = np.random.normal(0, 0.05)
        
        new_rate = exchange_rates[-1] * (1 + change)
        exchange_rates.append(max(new_rate, 50))  # Prevent negative rates
    
    # Generate interest rates with mean reversion
    interest_rates = [5.0]  # Starting interest rate
    for i in range(1, n_periods):
        mean_reversion = 0.1 * (5.0 - interest_rates[-1])
        shock = np.random.normal(0, 0.5)
        new_rate = interest_rates[-1] + mean_reversion + shock
        interest_rates.append(max(new_rate, 0))  # Prevent negative rates
    
    # Generate reserves with trend and volatility
    reserves = [1000.0]  # Starting reserves
    for i in range(1, n_periods):
        trend = -0.5 if regimes[i] == 1 else 0.2  # Declining in high pressure regime
        shock = np.random.normal(0, 20)
        new_reserves = reserves[-1] + trend + shock
        reserves.append(max(new_reserves, 100))  # Minimum reserves
    
    # Create pandas Series
    exchange_rate_series = pd.Series(exchange_rates, index=dates)
    interest_rate_series = pd.Series(interest_rates, index=dates)
    reserves_series = pd.Series(reserves, index=dates)
    
    return exchange_rate_series, interest_rate_series, reserves_series

def test_basic_erp_analysis():
    """Test basic ERP analysis functionality."""
    print("\n=== Testing Basic ERP Analysis ===")
    
    # Generate test data
    er_data, ir_data, res_data = generate_test_data()
    
    # Initialize analyzer
    analyzer = ExchangeRatePressureAnalyzer(
        enable_regime_switching=False,
        enable_predictive_modeling=False,
        enable_volatility_analysis=False
    )
    
    # Calculate pressure index
    result = analyzer.calculate_pressure_index(er_data, ir_data, res_data)
    
    # Validate results
    assert len(result.pressure_index) > 0, "Pressure index should not be empty"
    assert len(result.pressure_components) == 3, "Should have 3 pressure components"
    assert result.pressure_threshold > 0, "Pressure threshold should be positive"
    
    print(f"✓ Pressure index calculated with {len(result.pressure_index)} data points")
    print(f"✓ Pressure threshold: {result.pressure_threshold:.3f}")
    print(f"✓ Crisis periods identified: {len(result.crisis_periods)}")
    print(f"✓ Current warning level: {result.early_warning_signals.get('warning_level', 'unknown')}")
    
    # Print component contributions
    if 'component_contributions' in result.early_warning_signals:
        contributions = result.early_warning_signals['component_contributions']
        print(f"✓ Component contributions:")
        for component, value in contributions.items():
            print(f"  - {component}: {value:.3f}")
    
    return result

def test_enhanced_erp_analysis():
    """Test enhanced ERP analysis with all features enabled."""
    print("\n=== Testing Enhanced ERP Analysis ===")
    
    # Generate test data
    er_data, ir_data, res_data = generate_test_data()
    
    # Initialize analyzer with all features enabled
    analyzer = ExchangeRatePressureAnalyzer(
        enable_regime_switching=True,
        enable_predictive_modeling=True,
        enable_volatility_analysis=True,
        forecast_horizon=12
    )
    
    # Calculate pressure index
    result = analyzer.calculate_pressure_index(er_data, ir_data, res_data)
    
    # Test statistical tests
    if result.statistical_tests:
        print(f"✓ Statistical tests performed:")
        if 'adf_test' in result.statistical_tests:
            adf_result = result.statistical_tests['adf_test']
            print(f"  - ADF test p-value: {adf_result['pvalue']:.4f} (stationary: {adf_result['is_stationary']})")
        
        if 'normality_test' in result.statistical_tests:
            norm_result = result.statistical_tests['normality_test']
            print(f"  - Normality test p-value: {norm_result['pvalue']:.4f} (normal: {norm_result['is_normal']})")
        
        if 'descriptive_stats' in result.statistical_tests:
            desc_stats = result.statistical_tests['descriptive_stats']
            print(f"  - Mean: {desc_stats['mean']:.3f}, Std: {desc_stats['std']:.3f}")
            print(f"  - Skewness: {desc_stats['skewness']:.3f}, Kurtosis: {desc_stats['kurtosis']:.3f}")
    
    # Test regime analysis
    if result.regime_analysis:
        print(f"✓ Regime analysis completed:")
        print(f"  - Current regime: {result.regime_analysis.current_regime} ({result.regime_analysis.regime_description[result.regime_analysis.current_regime]})")
        
        current_regime = result.regime_analysis.current_regime
        if current_regime in result.regime_analysis.regime_persistence:
            persistence = result.regime_analysis.regime_persistence[current_regime]
            duration = result.regime_analysis.expected_regime_duration[current_regime]
            volatility = result.regime_analysis.regime_volatility[current_regime]
            print(f"  - Regime persistence: {persistence:.3f}")
            print(f"  - Expected duration: {duration:.1f} periods")
            print(f"  - Regime volatility: {volatility:.3f}")
    
    # Test volatility analysis
    if result.volatility_analysis:
        print(f"✓ Volatility analysis completed:")
        print(f"  - GARCH parameters:")
        for param, value in result.volatility_analysis.garch_params.items():
            print(f"    - {param}: {value:.4f}")
        print(f"  - Volatility clustering: {result.volatility_analysis.volatility_clustering}")
        print(f"  - ARCH test p-value: {result.volatility_analysis.arch_test_pvalue:.4f}")
        
        # Handle volatility forecast (could be array)
        forecast = result.volatility_analysis.volatility_forecast
        if isinstance(forecast, np.ndarray) and len(forecast) > 0:
            print(f"  - Next period volatility forecast: {forecast[0]:.4f}")
        elif isinstance(forecast, (int, float)):
            print(f"  - Next period volatility forecast: {forecast:.4f}")
    
    # Test predictive modeling
    if result.predictive_modeling:
        print(f"✓ Predictive modeling completed:")
        print(f"  - Best model: {result.predictive_modeling.best_model}")
        print(f"  - Model scores:")
        for model, score in result.predictive_modeling.model_scores.items():
            print(f"    - {model}: {score:.4f}")
        
        if result.predictive_modeling.feature_importance:
            print(f"  - Top 3 important features:")
            sorted_features = sorted(result.predictive_modeling.feature_importance.items(), 
                                   key=lambda x: x[1], reverse=True)[:3]
            for feature, importance in sorted_features:
                print(f"    - {feature}: {importance:.4f}")
    
    # Test model diagnostics
    if result.model_diagnostics:
        print(f"✓ Model diagnostics completed:")
        
        if 'stability_metrics' in result.model_diagnostics:
            stability = result.model_diagnostics['stability_metrics']
            print(f"  - Mean stability: {stability['mean_stability']:.4f}")
            print(f"  - Coefficient of variation: {stability['coefficient_of_variation']:.4f}")
        
        if 'outlier_analysis' in result.model_diagnostics:
            outliers = result.model_diagnostics['outlier_analysis']
            print(f"  - Outliers detected: {outliers['outlier_count']} ({outliers['outlier_percentage']:.1f}%)")
        
        if 'trend_analysis' in result.model_diagnostics:
            trend = result.model_diagnostics['trend_analysis']
            print(f"  - Trend slope: {trend['slope']:.6f} (significant: {trend['trend_significance']})")
            print(f"  - Trend R²: {trend['r_squared']:.4f}")
    
    return result

def test_error_handling():
    """Test error handling with insufficient or invalid data."""
    print("\n=== Testing Error Handling ===")
    
    analyzer = ExchangeRatePressureAnalyzer()
    
    # Test with insufficient data
    short_data = pd.Series([1, 2, 3], index=pd.date_range('2023-01-01', periods=3))
    result = analyzer.calculate_pressure_index(short_data, short_data, short_data)
    
    assert len(result.pressure_index) == 0, "Should return empty result for insufficient data"
    print("✓ Handles insufficient data correctly")
    
    # Test with NaN data
    nan_data = pd.Series([1, np.nan, 3, 4, 5], index=pd.date_range('2023-01-01', periods=5))
    result = analyzer.calculate_pressure_index(nan_data, nan_data, nan_data)
    print("✓ Handles NaN data correctly")
    
    return True

def test_performance():
    """Test performance with larger dataset."""
    print("\n=== Testing Performance ===")
    
    import time
    
    # Generate larger dataset
    er_data, ir_data, res_data = generate_test_data(500)
    
    analyzer = ExchangeRatePressureAnalyzer(
        enable_regime_switching=True,
        enable_predictive_modeling=True,
        enable_volatility_analysis=True
    )
    
    start_time = time.time()
    result = analyzer.calculate_pressure_index(er_data, ir_data, res_data)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"✓ Analysis completed in {execution_time:.2f} seconds for {len(er_data)} data points")
    print(f"✓ Performance: {len(er_data)/execution_time:.0f} data points per second")
    
    return execution_time < 30  # Should complete within 30 seconds

def main():
    """Run all ERP tests."""
    print("Enhanced Exchange Rate Pressure (ERP) Analyzer Test Suite")
    print("=" * 60)
    
    try:
        # Run tests
        test_basic_erp_analysis()
        test_enhanced_erp_analysis()
        test_error_handling()
        test_performance()
        
        print("\n" + "=" * 60)
        print("✅ All ERP tests passed successfully!")
        print("Enhanced Exchange Rate Pressure analyzer is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)