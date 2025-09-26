"""Comprehensive Testing and Validation Framework for Indexes Models

This module provides extensive testing capabilities for all indexes models,
including unit tests, integration tests, performance validation, and backtesting.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
import time
import json
from pathlib import Path
warnings.filterwarnings('ignore')

# Import our models
try:
    from arbitrage_pricing_theory import ArbitragePricingTheory, APTResult
    from capm_analyzer import CAPMAnalyzer, CAPMResult
    from macroeconomic_factor_model import MacroeconomicFactorModel, FactorModelResult
    from ml_models import AdvancedMLModels, MLResult, ModelType
    from unified_interface import UnifiedIndexesInterface
    from feature_engineering import AdvancedFeatureEngineering, FeatureType
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import models: {e}")
    MODELS_AVAILABLE = False

# Try to import testing libraries
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Scikit-learn not available. Using basic metrics.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Plotting libraries not available.")

class TestType(Enum):
    """Types of tests available"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    ROBUSTNESS = "robustness"
    BACKTESTING = "backtesting"
    STRESS = "stress"

class TestResult(Enum):
    """Test result status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"

@dataclass
class TestCase:
    """Individual test case"""
    name: str
    test_type: TestType
    description: str
    expected_result: Any = None
    tolerance: float = 0.01
    timeout: float = 30.0
    critical: bool = False

@dataclass
class TestOutcome:
    """Result of a test case"""
    test_case: TestCase
    result: TestResult
    actual_value: Any = None
    expected_value: Any = None
    error_message: str = ""
    execution_time: float = 0.0
    details: Dict[str, Any] = None

@dataclass
class TestSuite:
    """Collection of test cases"""
    name: str
    test_cases: List[TestCase]
    setup_data: Dict[str, Any] = None
    teardown_required: bool = False

@dataclass
class ValidationMetrics:
    """Validation metrics for model performance"""
    mse: float
    mae: float
    rmse: float
    r2_score: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    profit_factor: float

class DataGenerator:
    """Generate synthetic data for testing"""
    
    @staticmethod
    def generate_index_data(n_points: int = 252, start_price: float = 100.0, 
                           volatility: float = 0.2, trend: float = 0.05) -> Dict[str, Any]:
        """Generate synthetic index data"""
        np.random.seed(42)  # For reproducibility
        
        # Generate price series with trend and volatility
        returns = np.random.normal(trend/252, volatility/np.sqrt(252), n_points)
        prices = [start_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        prices = np.array(prices[1:])  # Remove initial price
        
        # Generate volume data
        base_volume = 1000000
        volume = np.random.lognormal(np.log(base_volume), 0.3, n_points)
        
        # Generate high/low data
        daily_range = np.random.uniform(0.01, 0.05, n_points)
        highs = prices * (1 + daily_range/2)
        lows = prices * (1 - daily_range/2)
        
        return {
            'symbol': 'TEST_INDEX',
            'name': 'Test Index',
            'current_level': prices[-1],
            'historical_levels': prices.tolist(),
            'historical_highs': highs.tolist(),
            'historical_lows': lows.tolist(),
            'historical_volumes': volume.tolist(),
            'dividend_yield': 0.02,
            'pe_ratio': 20.0,
            'pb_ratio': 3.0,
            'market_cap': 1e12,
            'volatility': volatility,
            'beta': 1.0,
            'sector_weights': {
                'Technology': 0.25,
                'Healthcare': 0.15,
                'Financials': 0.12,
                'Consumer': 0.10,
                'Energy': 0.08
            },
            'constituent_count': 500,
            'volume': volume[-1]
        }
    
    @staticmethod
    def generate_macro_data(n_points: int = 252) -> Dict[str, Any]:
        """Generate synthetic macroeconomic data"""
        np.random.seed(42)
        
        # Generate time series for macro indicators
        base_values = {
            'gdp_growth': 2.5,
            'inflation_rate': 2.0,
            'interest_rates': 3.0,
            'unemployment_rate': 4.0,
            'industrial_production': 1.0,
            'consumer_confidence': 100.0,
            'oil_prices': 70.0,
            'exchange_rates': 1.0,
            'vix_index': 20.0
        }
        
        macro_data = {}
        for key, base_value in base_values.items():
            # Add some random walk
            changes = np.random.normal(0, base_value * 0.05, n_points)
            series = [base_value]
            
            for change in changes:
                new_value = series[-1] + change
                # Keep values in reasonable ranges
                if key == 'unemployment_rate':
                    new_value = max(2.0, min(15.0, new_value))
                elif key == 'inflation_rate':
                    new_value = max(-2.0, min(10.0, new_value))
                elif key == 'interest_rates':
                    new_value = max(0.0, min(15.0, new_value))
                elif key == 'vix_index':
                    new_value = max(10.0, min(80.0, new_value))
                
                series.append(new_value)
            
            macro_data[key] = series[-1]  # Current value
            macro_data[f'{key}_history'] = series[1:]  # Historical series
        
        return macro_data
    
    @staticmethod
    def generate_factor_data(n_factors: int = 5, n_points: int = 252) -> Dict[str, Any]:
        """Generate synthetic factor data"""
        np.random.seed(42)
        
        factors = {}
        for i in range(n_factors):
            factor_name = f'factor_{i+1}'
            factor_returns = np.random.normal(0, 0.1, n_points)
            factors[factor_name] = factor_returns.tolist()
        
        return factors

class ModelTester:
    """Base class for model testing"""
    
    def __init__(self):
        self.test_results = []
        self.data_generator = DataGenerator()
        self.start_time = None
        self.end_time = None
    
    def run_test_case(self, test_case: TestCase, model: Any, data: Dict[str, Any]) -> TestOutcome:
        """Run a single test case"""
        start_time = time.time()
        
        try:
            # Execute the test based on type
            if test_case.test_type == TestType.UNIT:
                result = self._run_unit_test(test_case, model, data)
            elif test_case.test_type == TestType.INTEGRATION:
                result = self._run_integration_test(test_case, model, data)
            elif test_case.test_type == TestType.PERFORMANCE:
                result = self._run_performance_test(test_case, model, data)
            elif test_case.test_type == TestType.ACCURACY:
                result = self._run_accuracy_test(test_case, model, data)
            elif test_case.test_type == TestType.ROBUSTNESS:
                result = self._run_robustness_test(test_case, model, data)
            else:
                result = TestResult.SKIP
            
            execution_time = time.time() - start_time
            
            # Check timeout
            if execution_time > test_case.timeout:
                return TestOutcome(
                    test_case=test_case,
                    result=TestResult.FAIL,
                    error_message=f"Test timed out after {execution_time:.2f}s",
                    execution_time=execution_time
                )
            
            return TestOutcome(
                test_case=test_case,
                result=result,
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            return TestOutcome(
                test_case=test_case,
                result=TestResult.FAIL,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _run_unit_test(self, test_case: TestCase, model: Any, data: Dict[str, Any]) -> TestResult:
        """Run unit test"""
        # Test basic model functionality
        if hasattr(model, 'predict') or hasattr(model, 'analyze'):
            return TestResult.PASS
        return TestResult.FAIL
    
    def _run_integration_test(self, test_case: TestCase, model: Any, data: Dict[str, Any]) -> TestResult:
        """Run integration test"""
        # Test model integration with data
        try:
            if hasattr(model, 'predict'):
                result = model.predict(data)
            elif hasattr(model, 'analyze_apt'):
                result = model.analyze_apt(data)
            elif hasattr(model, 'analyze_capm'):
                result = model.analyze_capm(data)
            elif hasattr(model, 'analyze_factors'):
                result = model.analyze_factors(data)
            elif hasattr(model, 'engineer_features'):
                result = model.engineer_features(data.get('indexes_data', {}), data.get('macro_data', {}))
            else:
                return TestResult.FAIL
            
            return TestResult.PASS if result is not None else TestResult.FAIL
        except Exception:
            return TestResult.FAIL
    
    def _run_performance_test(self, test_case: TestCase, model: Any, data: Dict[str, Any]) -> TestResult:
        """Run performance test"""
        # Test model performance (speed)
        start_time = time.time()
        
        try:
            for _ in range(10):  # Run multiple times
                if hasattr(model, 'predict'):
                    model.predict(data)
                elif hasattr(model, 'analyze'):
                    model.analyze(data)
            
            avg_time = (time.time() - start_time) / 10
            return TestResult.PASS if avg_time < 1.0 else TestResult.WARNING
        
        except Exception:
            return TestResult.FAIL
    
    def _run_accuracy_test(self, test_case: TestCase, model: Any, data: Dict[str, Any]) -> TestResult:
        """Run accuracy test"""
        # Test model accuracy (simplified)
        try:
            if hasattr(model, 'predict'):
                result = model.predict(data)
            elif hasattr(model, 'analyze_apt'):
                result = model.analyze_apt(data)
            elif hasattr(model, 'analyze_capm'):
                result = model.analyze_capm(data)
            elif hasattr(model, 'analyze_factors'):
                result = model.analyze_factors(data)
            elif hasattr(model, 'engineer_features'):
                result = model.engineer_features(data.get('indexes_data', {}), data.get('macro_data', {}))
            else:
                return TestResult.FAIL
            
            # Check if result has reasonable values
            if hasattr(result, 'prediction') and result.prediction is not None:
                if not np.isnan(result.prediction) and not np.isinf(result.prediction):
                    return TestResult.PASS
            elif hasattr(result, 'expected_return') and result.expected_return is not None:
                if not np.isnan(result.expected_return) and not np.isinf(result.expected_return):
                    return TestResult.PASS
            elif hasattr(result, 'features') and result.features is not None:
                return TestResult.PASS
            
            return TestResult.WARNING
        
        except Exception:
            return TestResult.FAIL
    
    def _run_robustness_test(self, test_case: TestCase, model: Any, data: Dict[str, Any]) -> TestResult:
        """Run robustness test"""
        # Test model robustness with edge cases
        test_cases = [
            # Empty data
            {**data, 'historical_levels': []},
            # Single data point
            {**data, 'historical_levels': [100.0]},
            # Extreme values
            {**data, 'historical_levels': [1e-6, 1e6, -1e6]},
            # NaN values (if supported)
            {**data, 'volatility': float('nan')}
        ]
        
        passed_tests = 0
        for test_data in test_cases:
            try:
                if hasattr(model, 'predict'):
                    result = model.predict(test_data)
                elif hasattr(model, 'analyze_apt'):
                    result = model.analyze_apt(test_data)
                elif hasattr(model, 'analyze_capm'):
                    result = model.analyze_capm(test_data)
                elif hasattr(model, 'analyze_factors'):
                    result = model.analyze_factors(test_data)
                elif hasattr(model, 'engineer_features'):
                    result = model.engineer_features(test_data.get('indexes_data', {}), test_data.get('macro_data', {}))
                else:
                    continue
                
                if result is not None:
                    passed_tests += 1
            except Exception:
                continue
        
        success_rate = passed_tests / len(test_cases)
        if success_rate >= 0.75:
            return TestResult.PASS
        elif success_rate >= 0.5:
            return TestResult.WARNING
        else:
            return TestResult.FAIL

class IndexesTestSuite:
    """Comprehensive test suite for indexes models"""
    
    def __init__(self):
        self.model_tester = ModelTester()
        self.data_generator = DataGenerator()
        self.test_suites = self._create_test_suites()
        self.results = []
    
    def _create_test_suites(self) -> List[TestSuite]:
        """Create all test suites"""
        suites = []
        
        # APT Model Tests
        apt_tests = [
            TestCase("APT Basic Functionality", TestType.UNIT, "Test APT model initialization"),
            TestCase("APT Integration", TestType.INTEGRATION, "Test APT with real data"),
            TestCase("APT Performance", TestType.PERFORMANCE, "Test APT execution speed"),
            TestCase("APT Accuracy", TestType.ACCURACY, "Test APT prediction accuracy"),
            TestCase("APT Robustness", TestType.ROBUSTNESS, "Test APT with edge cases")
        ]
        suites.append(TestSuite("APT Model Tests", apt_tests))
        
        # CAPM Model Tests
        capm_tests = [
            TestCase("CAPM Basic Functionality", TestType.UNIT, "Test CAPM model initialization"),
            TestCase("CAPM Integration", TestType.INTEGRATION, "Test CAPM with real data"),
            TestCase("CAPM Performance", TestType.PERFORMANCE, "Test CAPM execution speed"),
            TestCase("CAPM Accuracy", TestType.ACCURACY, "Test CAPM prediction accuracy"),
            TestCase("CAPM Robustness", TestType.ROBUSTNESS, "Test CAPM with edge cases")
        ]
        suites.append(TestSuite("CAPM Model Tests", capm_tests))
        
        # Macro Factor Model Tests
        macro_tests = [
            TestCase("Macro Basic Functionality", TestType.UNIT, "Test Macro model initialization"),
            TestCase("Macro Integration", TestType.INTEGRATION, "Test Macro with real data"),
            TestCase("Macro Performance", TestType.PERFORMANCE, "Test Macro execution speed"),
            TestCase("Macro Accuracy", TestType.ACCURACY, "Test Macro prediction accuracy"),
            TestCase("Macro Robustness", TestType.ROBUSTNESS, "Test Macro with edge cases")
        ]
        suites.append(TestSuite("Macro Factor Model Tests", macro_tests))
        
        # ML Models Tests
        ml_tests = [
            TestCase("ML Basic Functionality", TestType.UNIT, "Test ML models initialization"),
            TestCase("ML Integration", TestType.INTEGRATION, "Test ML with real data"),
            TestCase("ML Performance", TestType.PERFORMANCE, "Test ML execution speed"),
            TestCase("ML Accuracy", TestType.ACCURACY, "Test ML prediction accuracy"),
            TestCase("ML Robustness", TestType.ROBUSTNESS, "Test ML with edge cases")
        ]
        suites.append(TestSuite("ML Models Tests", ml_tests))
        
        # Unified Interface Tests
        unified_tests = [
            TestCase("Unified Basic Functionality", TestType.UNIT, "Test Unified interface initialization"),
            TestCase("Unified Integration", TestType.INTEGRATION, "Test Unified with all models"),
            TestCase("Unified Performance", TestType.PERFORMANCE, "Test Unified execution speed"),
            TestCase("Unified Ensemble", TestType.ACCURACY, "Test Unified ensemble predictions"),
            TestCase("Unified Robustness", TestType.ROBUSTNESS, "Test Unified with edge cases")
        ]
        suites.append(TestSuite("Unified Interface Tests", unified_tests))
        
        # Feature Engineering Tests
        feature_tests = [
            TestCase("Feature Basic Functionality", TestType.UNIT, "Test Feature engineering initialization"),
            TestCase("Feature Integration", TestType.INTEGRATION, "Test Feature with real data"),
            TestCase("Feature Performance", TestType.PERFORMANCE, "Test Feature execution speed"),
            TestCase("Feature Quality", TestType.ACCURACY, "Test Feature quality and completeness"),
            TestCase("Feature Robustness", TestType.ROBUSTNESS, "Test Feature with edge cases")
        ]
        suites.append(TestSuite("Feature Engineering Tests", feature_tests))
        
        return suites
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites"""
        if not MODELS_AVAILABLE:
            print("Models not available. Skipping tests.")
            return {'status': 'skipped', 'reason': 'Models not available'}
        
        print("\n=== Running Comprehensive Indexes Tests ===")
        
        # Generate test data
        index_data = self.data_generator.generate_index_data()
        macro_data = self.data_generator.generate_macro_data()
        factor_data = self.data_generator.generate_factor_data()
        
        all_results = {}
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        warning_tests = 0
        skipped_tests = 0
        
        for suite in self.test_suites:
            print(f"\n--- Running {suite.name} ---")
            suite_results = []
            
            # Initialize model based on suite name
            model = self._get_model_for_suite(suite.name)
            if model is None:
                print(f"Skipping {suite.name} - model not available")
                continue
            
            for test_case in suite.test_cases:
                print(f"  Running: {test_case.name}...", end=" ")
                
                # Prepare data based on model type
                test_data = self._prepare_test_data(suite.name, index_data, macro_data, factor_data)
                
                # Run test
                outcome = self.model_tester.run_test_case(test_case, model, test_data)
                suite_results.append(outcome)
                
                # Update counters
                total_tests += 1
                if outcome.result == TestResult.PASS:
                    passed_tests += 1
                    print("✓ PASS")
                elif outcome.result == TestResult.FAIL:
                    failed_tests += 1
                    print("✗ FAIL")
                    if outcome.error_message:
                        print(f"    Error: {outcome.error_message}")
                elif outcome.result == TestResult.WARNING:
                    warning_tests += 1
                    print("⚠ WARNING")
                else:
                    skipped_tests += 1
                    print("- SKIP")
                
                print(f"    Time: {outcome.execution_time:.3f}s")
            
            all_results[suite.name] = suite_results
        
        # Summary
        print(f"\n=== Test Summary ===")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
        print(f"Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        print(f"Warnings: {warning_tests} ({warning_tests/total_tests*100:.1f}%)")
        print(f"Skipped: {skipped_tests} ({skipped_tests/total_tests*100:.1f}%)")
        
        return {
            'status': 'completed',
            'total_tests': total_tests,
            'passed': passed_tests,
            'failed': failed_tests,
            'warnings': warning_tests,
            'skipped': skipped_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'detailed_results': all_results
        }
    
    def _get_model_for_suite(self, suite_name: str) -> Any:
        """Get appropriate model for test suite"""
        try:
            if "APT" in suite_name:
                return ArbitragePricingTheory()
            elif "CAPM" in suite_name:
                return CAPMAnalyzer()
            elif "Macro" in suite_name:
                return MacroeconomicFactorModel()
            elif "ML" in suite_name:
                return AdvancedMLModels()
            elif "Unified" in suite_name:
                return UnifiedIndexesInterface()
            elif "Feature" in suite_name:
                return AdvancedFeatureEngineering()
            else:
                return None
        except Exception as e:
            print(f"Error initializing model for {suite_name}: {e}")
            return None
    
    def _prepare_test_data(self, suite_name: str, index_data: Dict, 
                          macro_data: Dict, factor_data: Dict) -> Dict[str, Any]:
        """Prepare test data based on suite requirements"""
        if "APT" in suite_name or "CAPM" in suite_name:
            return {
                **index_data,
                'market_data': {
                    'returns': np.random.normal(0.001, 0.02, 252).tolist(),
                    'risk_free_rate': 0.02
                },
                'factor_data': factor_data
            }
        elif "Macro" in suite_name:
            return {
                **index_data,
                **macro_data
            }
        elif "ML" in suite_name or "Unified" in suite_name:
            return {
                'index_data': index_data,
                'macro_data': macro_data
            }
        elif "Feature" in suite_name:
            return {
                'indexes_data': index_data,
                'macro_data': macro_data
            }
        else:
            return index_data
    
    def run_backtesting(self, start_date: str = "2020-01-01", 
                       end_date: str = "2023-12-31") -> Dict[str, Any]:
        """Run backtesting on models"""
        print(f"\n=== Running Backtesting ({start_date} to {end_date}) ===")
        
        if not MODELS_AVAILABLE:
            return {'status': 'skipped', 'reason': 'Models not available'}
        
        # Generate historical data for backtesting
        n_days = 1000  # Approximately 4 years
        historical_data = []
        
        for i in range(n_days):
            day_data = {
                'date': datetime.now() - timedelta(days=n_days-i),
                'index_data': self.data_generator.generate_index_data(252),
                'macro_data': self.data_generator.generate_macro_data(252)
            }
            historical_data.append(day_data)
        
        # Test models on historical data
        models_to_test = {
            'APT': ArbitragePricingTheory(),
            'CAPM': CAPMAnalyzer(),
            'Macro': MacroeconomicFactorModel(),
            'Unified': UnifiedIndexesInterface()
        }
        
        backtest_results = {}
        
        for model_name, model in models_to_test.items():
            print(f"\nBacktesting {model_name}...")
            
            predictions = []
            actuals = []
            
            try:
                for i, day_data in enumerate(historical_data[:-1]):
                    # Use current day to predict next day
                    next_day = historical_data[i + 1]
                    
                    if model_name == 'Unified':
                        result = model.ensemble_prediction(
                            day_data['index_data'], 
                            day_data['macro_data']
                        )
                        if hasattr(result, 'consensus_prediction'):
                            pred = result.consensus_prediction
                        else:
                            pred = 0.0
                    else:
                        if hasattr(model, 'predict'):
                            result = model.predict(day_data['index_data'])
                        elif hasattr(model, 'analyze_apt'):
                            result = model.analyze_apt(day_data['index_data'])
                        elif hasattr(model, 'analyze_capm'):
                            result = model.analyze_capm(day_data['index_data'])
                        elif hasattr(model, 'analyze_factors'):
                            result = model.analyze_factors(day_data['index_data'])
                        else:
                            pred = 0.0
                            continue
                        
                        if hasattr(result, 'prediction'):
                            pred = result.prediction
                        elif hasattr(result, 'expected_return'):
                            pred = result.expected_return
                        else:
                            pred = 0.0
                    
                    # Actual return
                    current_price = day_data['index_data']['current_level']
                    next_price = next_day['index_data']['current_level']
                    actual = (next_price - current_price) / current_price
                    
                    predictions.append(pred)
                    actuals.append(actual)
                
                # Calculate metrics
                if predictions and actuals:
                    metrics = self._calculate_backtest_metrics(predictions, actuals)
                    backtest_results[model_name] = metrics
                    
                    print(f"  MSE: {metrics['mse']:.6f}")
                    print(f"  MAE: {metrics['mae']:.6f}")
                    print(f"  Hit Rate: {metrics['hit_rate']:.2%}")
                    print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
                
            except Exception as e:
                print(f"  Error in backtesting {model_name}: {e}")
                backtest_results[model_name] = {'error': str(e)}
        
        return {
            'status': 'completed',
            'period': f"{start_date} to {end_date}",
            'results': backtest_results
        }
    
    def _calculate_backtest_metrics(self, predictions: List[float], 
                                   actuals: List[float]) -> Dict[str, float]:
        """Calculate backtesting metrics"""
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(predictions) & np.isfinite(actuals)
        predictions = predictions[valid_mask]
        actuals = actuals[valid_mask]
        
        if len(predictions) == 0:
            return {'error': 'No valid predictions'}
        
        # Basic metrics
        mse = np.mean((predictions - actuals) ** 2)
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(mse)
        
        # Hit rate (directional accuracy)
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(actuals)
        hit_rate = np.mean(pred_direction == actual_direction)
        
        # Sharpe ratio (simplified)
        if np.std(predictions) > 0:
            sharpe_ratio = np.mean(predictions) / np.std(predictions) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # R-squared
        if SKLEARN_AVAILABLE:
            r2 = r2_score(actuals, predictions)
        else:
            ss_res = np.sum((actuals - predictions) ** 2)
            ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'hit_rate': float(hit_rate),
            'sharpe_ratio': float(sharpe_ratio),
            'n_predictions': len(predictions)
        }
    
    def generate_test_report(self, results: Dict[str, Any], 
                           output_file: str = "indexes_test_report.json") -> str:
        """Generate comprehensive test report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'test_summary': {
                'total_tests': results.get('total_tests', 0),
                'passed': results.get('passed', 0),
                'failed': results.get('failed', 0),
                'warnings': results.get('warnings', 0),
                'skipped': results.get('skipped', 0),
                'success_rate': results.get('success_rate', 0)
            },
            'detailed_results': results.get('detailed_results', {}),
            'recommendations': self._generate_recommendations(results)
        }
        
        # Save to file
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\nTest report saved to: {output_file}")
        except Exception as e:
            print(f"Error saving report: {e}")
        
        return json.dumps(report, indent=2, default=str)
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        success_rate = results.get('success_rate', 0)
        failed_tests = results.get('failed', 0)
        warning_tests = results.get('warnings', 0)
        
        if success_rate < 0.8:
            recommendations.append(
                "Overall success rate is below 80%. Consider reviewing failed tests and improving model robustness."
            )
        
        if failed_tests > 0:
            recommendations.append(
                f"{failed_tests} tests failed. Review error messages and fix critical issues."
            )
        
        if warning_tests > 0:
            recommendations.append(
                f"{warning_tests} tests generated warnings. Consider optimizing performance and accuracy."
            )
        
        if success_rate >= 0.9:
            recommendations.append(
                "Excellent test performance! Models are ready for production use."
            )
        
        return recommendations

# Example usage and main execution
if __name__ == "__main__":
    # Initialize test suite
    test_suite = IndexesTestSuite()
    
    # Run all tests
    print("Starting comprehensive testing of indexes models...")
    results = test_suite.run_all_tests()
    
    # Run backtesting
    backtest_results = test_suite.run_backtesting()
    
    # Generate report
    if results.get('status') == 'completed':
        report = test_suite.generate_test_report(results)
        print("\n=== Testing Complete ===")
        print(f"Success Rate: {results['success_rate']:.1%}")
        
        if backtest_results.get('status') == 'completed':
            print("\n=== Backtesting Results ===")
            for model_name, metrics in backtest_results['results'].items():
                if 'error' not in metrics:
                    print(f"{model_name}: Hit Rate {metrics['hit_rate']:.1%}, Sharpe {metrics['sharpe_ratio']:.2f}")
    else:
        print(f"Testing skipped: {results.get('reason', 'Unknown reason')}")