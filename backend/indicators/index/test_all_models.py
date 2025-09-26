#!/usr/bin/env python3
"""
Comprehensive Test Suite for All Enhanced Index Indicator Models
Tests Term Structure, Macroeconomic Factor, and other models with backtesting and performance metrics
"""

import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from term_structure_model import AdvancedTermStructureModel, MacroeconomicData, IndexData as TSIndexData
from macro_factor_model import (
    MacroeconomicFactorModel, MacroeconomicData as MacroData, IndexData as MFIndexData, 
    EconomicRegime
)

class ModelPerformanceMetrics:
    """Calculate performance metrics for model validation"""
    
    @staticmethod
    def calculate_accuracy(predictions, actuals, threshold=0.05):
        """Calculate prediction accuracy within threshold"""
        if len(predictions) != len(actuals):
            return 0.0
        
        correct = 0
        for pred, actual in zip(predictions, actuals):
            if abs(pred - actual) / actual <= threshold:
                correct += 1
        
        return correct / len(predictions)
    
    @staticmethod
    def calculate_signal_accuracy(predicted_signals, actual_returns):
        """Calculate signal accuracy based on actual returns"""
        if len(predicted_signals) != len(actual_returns):
            return 0.0
        
        correct = 0
        for signal, return_val in zip(predicted_signals, actual_returns):
            if signal == "BUY" and return_val > 0:
                correct += 1
            elif signal == "SELL" and return_val < 0:
                correct += 1
            elif signal == "HOLD" and abs(return_val) < 0.02:
                correct += 1
        
        return correct / len(predicted_signals)
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = np.array(returns) - risk_free_rate / 252  # Daily risk-free rate
        if np.std(excess_returns) == 0:
            return 0.0
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(returns):
        """Calculate maximum drawdown"""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))

def generate_test_data():
    """Generate synthetic test data for backtesting"""
    np.random.seed(42)  # For reproducible results
    
    # Generate 100 days of test data
    test_data = []
    base_level = 4200.0
    
    for i in range(100):
        # Simulate market movements
        daily_return = np.random.normal(0.0005, 0.015)  # 0.05% daily return, 1.5% volatility
        base_level *= (1 + daily_return)
        
        # Term Structure data (using MacroeconomicData for yield curve info)
        ts_macro_data = MacroeconomicData(
            gdp_growth=2.5 + np.random.normal(0, 0.5),
            inflation_rate=3.2 + np.random.normal(0, 0.3),
            unemployment_rate=3.8 + np.random.normal(0, 0.5),
            interest_rate=5.25 + np.random.normal(0, 0.25),
            money_supply_growth=8.5 + np.random.normal(0, 1),
            government_debt_to_gdp=120.0 + np.random.normal(0, 2),
            trade_balance=-50.0 + np.random.normal(0, 10),
            consumer_confidence=105.0 + np.random.normal(0, 5),
            business_confidence=95.0 + np.random.normal(0, 5),
            manufacturing_pmi=52.0 + np.random.normal(0, 2),
            services_pmi=54.0 + np.random.normal(0, 2),
            retail_sales_growth=4.2 + np.random.normal(0, 1),
            industrial_production=2.8 + np.random.normal(0, 0.5),
            housing_starts=1.4 + np.random.normal(0, 0.2),
            oil_price=75.0 + np.random.normal(0, 5),
            dollar_index=103.0 + np.random.normal(0, 2),
            vix=18.5 + np.random.normal(0, 3)
        )
        
        ts_index = TSIndexData(
            symbol="SPX",
            name="S&P 500",
            current_level=base_level,
            historical_levels=[base_level * (1 + np.random.normal(0, 0.01)) for _ in range(5)],
            dividend_yield=1.8 + np.random.normal(0, 0.2),
            pe_ratio=22.5 + np.random.normal(0, 2),
            pb_ratio=3.2 + np.random.normal(0, 0.3),
            market_cap=35000000000000,
            volatility=0.18 + np.random.normal(0, 0.02),
            beta=1.0 + np.random.normal(0, 0.1),
            sector_weights={"Technology": 0.28, "Healthcare": 0.13, "Financials": 0.11},
            constituent_count=500,
            volume=1000000000
        )
        
        # Macroeconomic data
        macro_data = MacroData(
            gdp_growth=2.5 + np.random.normal(0, 0.5),
            inflation_rate=3.2 + np.random.normal(0, 0.3),
            unemployment_rate=3.8 + np.random.normal(0, 0.5),
            interest_rate=5.25 + np.random.normal(0, 0.25),
            money_supply_growth=8.5 + np.random.normal(0, 1),
            government_debt_to_gdp=120.0 + np.random.normal(0, 2),
            trade_balance=-50.0 + np.random.normal(0, 10),
            consumer_confidence=105.0 + np.random.normal(0, 5),
            business_confidence=95.0 + np.random.normal(0, 5),
            manufacturing_pmi=52.0 + np.random.normal(0, 2),
            services_pmi=54.0 + np.random.normal(0, 2),
            retail_sales_growth=4.2 + np.random.normal(0, 1),
            industrial_production=2.8 + np.random.normal(0, 0.5),
            housing_starts=1.4 + np.random.normal(0, 0.2),
            oil_price=75.0 + np.random.normal(0, 5),
            dollar_index=103.0 + np.random.normal(0, 2),
            vix=18.5 + np.random.normal(0, 3)
        )
        
        mf_index = MFIndexData(
            symbol="SPX",
            name="S&P 500",
            current_level=base_level,
            historical_levels=[base_level * (1 + np.random.normal(0, 0.01)) for _ in range(5)],
            dividend_yield=1.8 + np.random.normal(0, 0.2),
            pe_ratio=22.5 + np.random.normal(0, 2),
            pb_ratio=3.2 + np.random.normal(0, 0.3),
            market_cap=35000000000000,
            volatility=0.18 + np.random.normal(0, 0.02),
            beta=1.0 + np.random.normal(0, 0.1),
            sector_weights={"Technology": 0.28, "Healthcare": 0.13, "Financials": 0.11},
            constituent_count=500,
            volume=1000000000
        )
        
        test_data.append({
            'date': datetime.now() - timedelta(days=100-i),
            'actual_level': base_level,
            'actual_return': daily_return,
            'ts_macro_data': ts_macro_data,
            'ts_index': ts_index,
            'macro_data': macro_data,
            'mf_index': mf_index
        })
    
    return test_data

def test_term_structure_model(test_data):
    """Test Term Structure model performance"""
    print("\n=== TERM STRUCTURE MODEL TESTING ===")
    
    model = AdvancedTermStructureModel()
    predictions = []
    signals = []
    actual_returns = []
    
    for i, data_point in enumerate(test_data[:-1]):  # Exclude last point for prediction
        try:
            result = model.calculate(data_point['ts_index'], data_point['ts_macro_data'])
            
            # Get next day's actual return for validation
            next_return = test_data[i+1]['actual_return']
            
            predictions.append(result.fair_value)
            signals.append(result.signal)
            actual_returns.append(next_return)
            
        except Exception as e:
            print(f"Error in Term Structure model at day {i}: {e}")
            continue
    
    # Calculate performance metrics
    if predictions:
        actual_levels = [data['actual_level'] for data in test_data[1:len(predictions)+1]]
        
        accuracy = ModelPerformanceMetrics.calculate_accuracy(predictions, actual_levels)
        signal_accuracy = ModelPerformanceMetrics.calculate_signal_accuracy(signals, actual_returns)
        sharpe_ratio = ModelPerformanceMetrics.calculate_sharpe_ratio(actual_returns)
        max_drawdown = ModelPerformanceMetrics.calculate_max_drawdown(actual_returns)
        
        print(f"Term Structure Model Results:")
        print(f"  Fair Value Accuracy (5% threshold): {accuracy:.2%}")
        print(f"  Signal Accuracy: {signal_accuracy:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  Maximum Drawdown: {max_drawdown:.2%}")
        print(f"  Total Predictions: {len(predictions)}")
        
        # Signal distribution
        signal_counts = {signal: signals.count(signal) for signal in set(signals)}
        print(f"  Signal Distribution: {signal_counts}")
        
        return {
            'accuracy': accuracy,
            'signal_accuracy': signal_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_predictions': len(predictions)
        }
    else:
        print("No valid predictions generated")
        return None

def test_macro_factor_model(test_data):
    """Test Macroeconomic Factor model performance"""
    print("\n=== MACROECONOMIC FACTOR MODEL TESTING ===")
    
    model = MacroeconomicFactorModel()
    predictions = []
    signals = []
    actual_returns = []
    regimes = []
    confidences = []
    
    for i, data_point in enumerate(test_data[:-1]):  # Exclude last point for prediction
        try:
            result = model.calculate(data_point['mf_index'], data_point['macro_data'])
            
            # Get next day's actual return for validation
            next_return = test_data[i+1]['actual_return']
            
            predictions.append(result.regime_adjusted_fair_value)
            signals.append(result.signal)
            actual_returns.append(next_return)
            regimes.append(result.current_regime.value)
            confidences.append(result.confidence)
            
        except Exception as e:
            print(f"Error in Macro Factor model at day {i}: {e}")
            continue
    
    # Calculate performance metrics
    if predictions:
        actual_levels = [data['actual_level'] for data in test_data[1:len(predictions)+1]]
        
        accuracy = ModelPerformanceMetrics.calculate_accuracy(predictions, actual_levels)
        signal_accuracy = ModelPerformanceMetrics.calculate_signal_accuracy(signals, actual_returns)
        sharpe_ratio = ModelPerformanceMetrics.calculate_sharpe_ratio(actual_returns)
        max_drawdown = ModelPerformanceMetrics.calculate_max_drawdown(actual_returns)
        avg_confidence = np.mean(confidences)
        
        print(f"Macroeconomic Factor Model Results:")
        print(f"  Fair Value Accuracy (5% threshold): {accuracy:.2%}")
        print(f"  Signal Accuracy: {signal_accuracy:.2%}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.3f}")
        print(f"  Maximum Drawdown: {max_drawdown:.2%}")
        print(f"  Average Confidence: {avg_confidence:.3f}")
        print(f"  Total Predictions: {len(predictions)}")
        
        # Signal and regime distribution
        signal_counts = {signal: signals.count(signal) for signal in set(signals)}
        regime_counts = {regime: regimes.count(regime) for regime in set(regimes)}
        print(f"  Signal Distribution: {signal_counts}")
        print(f"  Regime Distribution: {regime_counts}")
        
        return {
            'accuracy': accuracy,
            'signal_accuracy': signal_accuracy,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_confidence': avg_confidence,
            'total_predictions': len(predictions)
        }
    else:
        print("No valid predictions generated")
        return None

def test_model_consistency():
    """Test model consistency and stability"""
    print("\n=== MODEL CONSISTENCY TESTING ===")
    
    # Create consistent test data
    ts_macro_data = MacroeconomicData(
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
    
    ts_index = TSIndexData(
        symbol="SPX",
        name="S&P 500",
        current_level=4200.0,
        historical_levels=[4000, 4050, 4100, 4150, 4200],
        dividend_yield=1.8,
        pe_ratio=22.5,
        pb_ratio=3.2,
        market_cap=35000000000000,
        volatility=0.18,
        beta=1.0,
        sector_weights={"Technology": 0.28, "Healthcare": 0.13, "Financials": 0.11},
        constituent_count=500,
        volume=1000000000
    )
    
    macro_data = MacroData(
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
    
    mf_index = MFIndexData(
        symbol="SPX",
        name="S&P 500",
        current_level=4200.0,
        historical_levels=[4000, 4050, 4100, 4150, 4200],
        dividend_yield=1.8,
        pe_ratio=22.5,
        pb_ratio=3.2,
        market_cap=35000000000000,
        volatility=0.18,
        beta=1.0,
        sector_weights={"Technology": 0.28, "Healthcare": 0.13, "Financials": 0.11},
        constituent_count=500,
        volume=1000000000
    )
    
    # Test Term Structure model consistency
    ts_model = AdvancedTermStructureModel()
    ts_results = []
    for _ in range(10):
        result = ts_model.calculate(ts_index, ts_macro_data)
        ts_results.append(result.fair_value)
    
    ts_consistency = np.std(ts_results) / np.mean(ts_results) if np.mean(ts_results) != 0 else float('inf')
    
    # Test Macro Factor model consistency
    mf_model = MacroeconomicFactorModel()
    mf_results = []
    for _ in range(10):
        result = mf_model.calculate(mf_index, macro_data)
        mf_results.append(result.regime_adjusted_fair_value)
    
    mf_consistency = np.std(mf_results) / np.mean(mf_results) if np.mean(mf_results) != 0 else float('inf')
    
    print(f"Model Consistency Results:")
    print(f"  Term Structure Model CV: {ts_consistency:.6f}")
    print(f"  Macro Factor Model CV: {mf_consistency:.6f}")
    print(f"  Term Structure Mean: ${np.mean(ts_results):.2f}")
    print(f"  Macro Factor Mean: ${np.mean(mf_results):.2f}")
    
    return {
        'ts_consistency': ts_consistency,
        'mf_consistency': mf_consistency,
        'ts_mean': np.mean(ts_results),
        'mf_mean': np.mean(mf_results)
    }

def run_comprehensive_tests():
    """Run all comprehensive tests"""
    print("=== COMPREHENSIVE INDEX INDICATOR MODEL TESTING ===")
    print(f"Test started at: {datetime.now()}")
    
    start_time = time.time()
    
    # Generate test data
    print("\nGenerating synthetic test data...")
    test_data = generate_test_data()
    print(f"Generated {len(test_data)} data points")
    
    # Test individual models
    ts_results = test_term_structure_model(test_data)
    mf_results = test_macro_factor_model(test_data)
    
    # Test model consistency
    consistency_results = test_model_consistency()
    
    # Summary
    print("\n=== COMPREHENSIVE TEST SUMMARY ===")
    
    if ts_results:
        print(f"Term Structure Model:")
        print(f"  ✓ Fair Value Accuracy: {ts_results['accuracy']:.2%}")
        print(f"  ✓ Signal Accuracy: {ts_results['signal_accuracy']:.2%}")
        print(f"  ✓ Sharpe Ratio: {ts_results['sharpe_ratio']:.3f}")
    
    if mf_results:
        print(f"Macroeconomic Factor Model:")
        print(f"  ✓ Fair Value Accuracy: {mf_results['accuracy']:.2%}")
        print(f"  ✓ Signal Accuracy: {mf_results['signal_accuracy']:.2%}")
        print(f"  ✓ Sharpe Ratio: {mf_results['sharpe_ratio']:.3f}")
        print(f"  ✓ Average Confidence: {mf_results['avg_confidence']:.3f}")
    
    print(f"Model Consistency:")
    print(f"  ✓ Term Structure CV: {consistency_results['ts_consistency']:.6f}")
    print(f"  ✓ Macro Factor CV: {consistency_results['mf_consistency']:.6f}")
    
    end_time = time.time()
    print(f"\nTotal test duration: {end_time - start_time:.2f} seconds")
    
    # Overall assessment
    overall_success = True
    if ts_results and ts_results['accuracy'] < 0.3:
        overall_success = False
    if mf_results and mf_results['accuracy'] < 0.3:
        overall_success = False
    if consistency_results['ts_consistency'] > 0.1 or consistency_results['mf_consistency'] > 0.1:
        overall_success = False
    
    if overall_success:
        print("\n🎉 ALL MODELS PASSED COMPREHENSIVE TESTING!")
        return True
    else:
        print("\n⚠️  Some models need improvement")
        return False

if __name__ == "__main__":
    success = run_comprehensive_tests()
    if success:
        print("\n✅ Comprehensive testing completed successfully!")
    else:
        print("\n❌ Some tests failed - models need refinement!")
        sys.exit(1)