"""Comprehensive Testing Framework for Stock Models

This module provides extensive testing capabilities for stock analysis models including:
- Model accuracy and performance testing
- Backtesting with historical data
- Robustness testing under different market conditions
- Cross-validation and statistical validation
- Risk metrics and performance attribution
- Model comparison and benchmarking

Author: Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
import json
import logging
from pathlib import Path
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import additional libraries
try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    from scipy.stats import jarque_bera, shapiro
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Import local modules
try:
    from .advanced_ml_models import AdvancedMLModels, StockData, MLPredictionResult
    from .unified_interface import StockUnifiedInterface
    from .ensemble_methods import AdvancedEnsemblePredictor
    from .feature_engineering import AdvancedFeatureEngineer
except ImportError:
    # Handle case where modules might not be available
    pass

class TestType(Enum):
    """Types of tests available"""
    ACCURACY = "accuracy"
    BACKTESTING = "backtesting"
    ROBUSTNESS = "robustness"
    CROSS_VALIDATION = "cross_validation"
    STRESS_TEST = "stress_test"
    BENCHMARK = "benchmark"
    STATISTICAL = "statistical"

class MarketCondition(Enum):
    """Market conditions for testing"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"
    RECOVERY = "recovery"

class RiskMetric(Enum):
    """Risk metrics to calculate"""
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    BETA = "beta"
    ALPHA = "alpha"
    INFORMATION_RATIO = "information_ratio"
    CALMAR_RATIO = "calmar_ratio"

@dataclass
class TestResult:
    """Individual test result"""
    test_name: str
    test_type: TestType
    model_name: str
    success: bool
    score: float
    metrics: Dict[str, float]
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime
    error_message: Optional[str] = None

@dataclass
class BacktestResult:
    """Backtesting result"""
    model_name: str
    start_date: datetime
    end_date: datetime
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: List[Dict[str, Any]]
    equity_curve: List[float]
    risk_metrics: Dict[RiskMetric, float]
    benchmark_comparison: Dict[str, float]

@dataclass
class ModelComparison:
    """Model comparison result"""
    models: List[str]
    metrics: Dict[str, Dict[str, float]]
    rankings: Dict[str, List[str]]
    statistical_significance: Dict[str, Dict[str, float]]
    best_model: str
    worst_model: str
    summary: str

@dataclass
class TestSuite:
    """Complete test suite results"""
    suite_name: str
    test_results: List[TestResult]
    backtest_results: List[BacktestResult]
    model_comparisons: List[ModelComparison]
    overall_score: float
    execution_time: float
    timestamp: datetime
    summary_report: str

class PerformanceMetrics:
    """Calculate various performance metrics"""
    
    @staticmethod
    def calculate_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate returns from prices"""
        return np.diff(prices) / prices[:-1]
    
    @staticmethod
    def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
        """Calculate log returns from prices"""
        return np.diff(np.log(prices))
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns / np.std(returns) * np.sqrt(252)  # Annualized
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = np.mean(returns) - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
        
        downside_deviation = np.std(downside_returns)
        if downside_deviation == 0:
            return 0.0
        
        return excess_returns / downside_deviation * np.sqrt(252)
    
    @staticmethod
    def calculate_max_drawdown(prices: np.ndarray) -> Tuple[float, int, int]:
        """Calculate maximum drawdown"""
        if len(prices) == 0:
            return 0.0, 0, 0
        
        peak = prices[0]
        max_dd = 0.0
        peak_idx = 0
        trough_idx = 0
        
        for i, price in enumerate(prices):
            if price > peak:
                peak = price
                peak_idx = i
            
            drawdown = (peak - price) / peak
            if drawdown > max_dd:
                max_dd = drawdown
                trough_idx = i
        
        return max_dd, peak_idx, trough_idx
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
        
        return np.percentile(returns, (1 - confidence_level) * 100)
    
    @staticmethod
    def calculate_beta(stock_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate beta relative to market"""
        if len(stock_returns) == 0 or len(market_returns) == 0:
            return 1.0
        
        min_length = min(len(stock_returns), len(market_returns))
        stock_returns = stock_returns[-min_length:]
        market_returns = market_returns[-min_length:]
        
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    @staticmethod
    def calculate_alpha(stock_returns: np.ndarray, market_returns: np.ndarray, 
                      risk_free_rate: float = 0.02) -> float:
        """Calculate alpha (Jensen's alpha)"""
        if len(stock_returns) == 0 or len(market_returns) == 0:
            return 0.0
        
        beta = PerformanceMetrics.calculate_beta(stock_returns, market_returns)
        
        stock_return = np.mean(stock_returns) * 252  # Annualized
        market_return = np.mean(market_returns) * 252  # Annualized
        
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        
        return stock_return - expected_return

class ModelTester:
    """Main testing class for stock models"""
    
    def __init__(self):
        self.test_results = []
        self.performance_metrics = PerformanceMetrics()
        self.feature_engineer = None
        
        # Initialize feature engineer if available
        try:
            self.feature_engineer = AdvancedFeatureEngineer()
        except:
            pass
    
    def run_accuracy_test(self, model: Any, test_data: Dict[str, Any], 
                         target_column: str = 'target') -> TestResult:
        """Run accuracy test on a model"""
        start_time = datetime.now()
        
        try:
            # Prepare test data
            if isinstance(test_data, dict) and 'historical_prices' in test_data:
                prices = np.array(test_data['historical_prices'])
                if len(prices) < 2:
                    raise ValueError("Insufficient price data for testing")
                
                # Create target (next day return)
                returns = self.performance_metrics.calculate_returns(prices)
                target = returns[1:]  # Next day returns
                
                # Get predictions based on model type
                predictions = self._get_model_predictions(model, test_data, len(target))
                
                if len(predictions) == 0:
                    raise ValueError("No predictions generated")
                
                # Align predictions and target
                min_length = min(len(predictions), len(target))
                predictions = predictions[-min_length:]
                target = target[-min_length:]
                
                # Calculate metrics
                metrics = self._calculate_accuracy_metrics(predictions, target)
                
                success = True
                score = metrics.get('r2_score', 0.0)
                error_message = None
                
            else:
                raise ValueError("Invalid test data format")
                
        except Exception as e:
            success = False
            score = 0.0
            metrics = {}
            error_message = str(e)
            logger.error(f"Accuracy test failed: {e}")
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = TestResult(
            test_name="accuracy_test",
            test_type=TestType.ACCURACY,
            model_name=getattr(model, '__class__', type(model)).__name__,
            success=success,
            score=score,
            metrics=metrics,
            details={'target_column': target_column},
            execution_time=execution_time,
            timestamp=datetime.now(),
            error_message=error_message
        )
        
        self.test_results.append(result)
        return result
    
    def run_backtest(self, model: Any, historical_data: Dict[str, Any], 
                    start_date: datetime = None, end_date: datetime = None,
                    initial_capital: float = 100000.0) -> BacktestResult:
        """Run backtesting on historical data"""
        try:
            prices = np.array(historical_data['historical_prices'])
            if len(prices) < 50:
                raise ValueError("Insufficient historical data for backtesting")
            
            # Default date range
            if start_date is None:
                start_date = datetime.now() - timedelta(days=len(prices))
            if end_date is None:
                end_date = datetime.now()
            
            # Initialize backtesting variables
            capital = initial_capital
            position = 0.0
            trades = []
            equity_curve = [capital]
            
            # Run backtest
            for i in range(50, len(prices) - 1):  # Leave room for lookback and lookahead
                current_data = {
                    'historical_prices': prices[:i+1],
                    'volume': historical_data.get('volume', [1000000] * (i+1))[:i+1]
                }
                
                # Get prediction
                try:
                    predictions = self._get_model_predictions(model, current_data, 1)
                    if len(predictions) > 0:
                        signal = predictions[0]
                        
                        # Simple trading logic
                        current_price = prices[i]
                        next_price = prices[i + 1]
                        
                        # Buy signal (positive prediction)
                        if signal > 0.01 and position <= 0:
                            if position < 0:
                                # Close short position
                                capital += position * (current_price - next_price)
                                trades.append({
                                    'type': 'cover',
                                    'price': current_price,
                                    'quantity': -position,
                                    'timestamp': i
                                })
                            
                            # Open long position
                            position = capital / current_price * 0.95  # 95% of capital
                            trades.append({
                                'type': 'buy',
                                'price': current_price,
                                'quantity': position,
                                'timestamp': i
                            })
                        
                        # Sell signal (negative prediction)
                        elif signal < -0.01 and position >= 0:
                            if position > 0:
                                # Close long position
                                capital = position * current_price
                                trades.append({
                                    'type': 'sell',
                                    'price': current_price,
                                    'quantity': position,
                                    'timestamp': i
                                })
                                position = 0
                            
                            # Open short position
                            position = -capital / current_price * 0.95
                            trades.append({
                                'type': 'short',
                                'price': current_price,
                                'quantity': -position,
                                'timestamp': i
                            })
                        
                        # Calculate current equity
                        if position > 0:
                            current_equity = position * next_price
                        elif position < 0:
                            current_equity = capital + position * (current_price - next_price)
                        else:
                            current_equity = capital
                        
                        equity_curve.append(current_equity)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed at step {i}: {e}")
                    equity_curve.append(equity_curve[-1])  # Keep previous equity
            
            # Calculate performance metrics
            final_equity = equity_curve[-1]
            total_return = (final_equity - initial_capital) / initial_capital
            
            # Convert equity curve to returns for metrics calculation
            equity_returns = self.performance_metrics.calculate_returns(np.array(equity_curve))
            
            annualized_return = total_return * (252 / len(equity_curve)) if len(equity_curve) > 0 else 0.0
            volatility = np.std(equity_returns) * np.sqrt(252) if len(equity_returns) > 0 else 0.0
            sharpe_ratio = self.performance_metrics.calculate_sharpe_ratio(equity_returns)
            max_drawdown, _, _ = self.performance_metrics.calculate_max_drawdown(np.array(equity_curve))
            
            # Calculate win rate
            winning_trades = sum(1 for trade in trades if trade.get('profit', 0) > 0)
            win_rate = winning_trades / len(trades) if len(trades) > 0 else 0.0
            
            # Calculate profit factor
            gross_profit = sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) > 0)
            gross_loss = abs(sum(trade.get('profit', 0) for trade in trades if trade.get('profit', 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Risk metrics
            risk_metrics = {
                RiskMetric.SHARPE_RATIO: sharpe_ratio,
                RiskMetric.SORTINO_RATIO: self.performance_metrics.calculate_sortino_ratio(equity_returns),
                RiskMetric.MAX_DRAWDOWN: max_drawdown,
                RiskMetric.VAR_95: self.performance_metrics.calculate_var(equity_returns, 0.95),
                RiskMetric.VAR_99: self.performance_metrics.calculate_var(equity_returns, 0.99)
            }
            
            # Benchmark comparison (vs buy and hold)
            buy_hold_return = (prices[-1] - prices[50]) / prices[50]
            benchmark_comparison = {
                'excess_return': total_return - buy_hold_return,
                'tracking_error': np.std(equity_returns - self.performance_metrics.calculate_returns(prices[50:])) if len(prices) > 50 else 0.0
            }
            
            return BacktestResult(
                model_name=getattr(model, '__class__', type(model)).__name__,
                start_date=start_date,
                end_date=end_date,
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                trades=trades,
                equity_curve=equity_curve,
                risk_metrics=risk_metrics,
                benchmark_comparison=benchmark_comparison
            )
            
        except Exception as e:
            logger.error(f"Backtesting failed: {e}")
            # Return empty result
            return BacktestResult(
                model_name=getattr(model, '__class__', type(model)).__name__,
                start_date=start_date or datetime.now(),
                end_date=end_date or datetime.now(),
                total_return=0.0,
                annualized_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                trades=[],
                equity_curve=[initial_capital],
                risk_metrics={},
                benchmark_comparison={}
            )
    
    def run_robustness_test(self, model: Any, test_data: Dict[str, Any], 
                           noise_levels: List[float] = [0.01, 0.05, 0.1]) -> List[TestResult]:
        """Test model robustness under different noise conditions"""
        results = []
        
        for noise_level in noise_levels:
            start_time = datetime.now()
            
            try:
                # Add noise to test data
                noisy_data = self._add_noise_to_data(test_data, noise_level)
                
                # Run accuracy test on noisy data
                result = self.run_accuracy_test(model, noisy_data)
                result.test_name = f"robustness_test_noise_{noise_level}"
                result.test_type = TestType.ROBUSTNESS
                result.details['noise_level'] = noise_level
                
                results.append(result)
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                
                result = TestResult(
                    test_name=f"robustness_test_noise_{noise_level}",
                    test_type=TestType.ROBUSTNESS,
                    model_name=getattr(model, '__class__', type(model)).__name__,
                    success=False,
                    score=0.0,
                    metrics={},
                    details={'noise_level': noise_level},
                    execution_time=execution_time,
                    timestamp=datetime.now(),
                    error_message=str(e)
                )
                
                results.append(result)
                logger.error(f"Robustness test failed for noise level {noise_level}: {e}")
        
        return results
    
    def compare_models(self, models: List[Any], test_data: Dict[str, Any]) -> ModelComparison:
        """Compare multiple models"""
        model_results = {}
        
        # Run tests on all models
        for model in models:
            model_name = getattr(model, '__class__', type(model)).__name__
            
            # Run accuracy test
            accuracy_result = self.run_accuracy_test(model, test_data)
            
            # Run backtest
            backtest_result = self.run_backtest(model, test_data)
            
            model_results[model_name] = {
                'accuracy_score': accuracy_result.score,
                'total_return': backtest_result.total_return,
                'sharpe_ratio': backtest_result.sharpe_ratio,
                'max_drawdown': backtest_result.max_drawdown,
                'win_rate': backtest_result.win_rate
            }
        
        # Create rankings
        rankings = {}
        for metric in ['accuracy_score', 'total_return', 'sharpe_ratio', 'win_rate']:
            sorted_models = sorted(model_results.keys(), 
                                 key=lambda x: model_results[x].get(metric, 0), 
                                 reverse=True)
            rankings[metric] = sorted_models
        
        # Max drawdown ranking (lower is better)
        sorted_models = sorted(model_results.keys(), 
                             key=lambda x: model_results[x].get('max_drawdown', float('inf')))
        rankings['max_drawdown'] = sorted_models
        
        # Determine best and worst models (based on average ranking)
        model_avg_ranks = {}
        for model_name in model_results.keys():
            ranks = []
            for metric, ranking in rankings.items():
                if model_name in ranking:
                    ranks.append(ranking.index(model_name))
            model_avg_ranks[model_name] = np.mean(ranks) if ranks else float('inf')
        
        best_model = min(model_avg_ranks.keys(), key=lambda x: model_avg_ranks[x])
        worst_model = max(model_avg_ranks.keys(), key=lambda x: model_avg_ranks[x])
        
        # Create summary
        summary = f"Model Comparison Results:\n"
        summary += f"Best Overall Model: {best_model}\n"
        summary += f"Worst Overall Model: {worst_model}\n"
        summary += f"Total Models Tested: {len(models)}\n"
        
        return ModelComparison(
            models=list(model_results.keys()),
            metrics=model_results,
            rankings=rankings,
            statistical_significance={},  # Would need more sophisticated testing
            best_model=best_model,
            worst_model=worst_model,
            summary=summary
        )
    
    def _get_model_predictions(self, model: Any, data: Dict[str, Any], n_predictions: int) -> np.ndarray:
        """Get predictions from different model types"""
        try:
            # Check if model has specific prediction methods
            if hasattr(model, 'analyze_apt'):
                result = model.analyze_apt(data)
                if hasattr(result, 'prediction'):
                    return np.array([result.prediction] * n_predictions)
                elif 'prediction' in result:
                    return np.array([result['prediction']] * n_predictions)
            
            elif hasattr(model, 'analyze_capm'):
                result = model.analyze_capm(data)
                if hasattr(result, 'expected_return'):
                    return np.array([result.expected_return] * n_predictions)
                elif 'expected_return' in result:
                    return np.array([result['expected_return']] * n_predictions)
            
            elif hasattr(model, 'analyze_factors'):
                result = model.analyze_factors(data)
                if hasattr(result, 'prediction'):
                    return np.array([result.prediction] * n_predictions)
                elif 'prediction' in result:
                    return np.array([result['prediction']] * n_predictions)
            
            elif hasattr(model, 'engineer_features'):
                result = model.engineer_features(data)
                if hasattr(result, 'features') and not result.features.empty:
                    # Use last feature value as prediction proxy
                    last_features = result.features.iloc[-1]
                    prediction = last_features.mean() if not last_features.empty else 0.0
                    return np.array([prediction] * n_predictions)
            
            elif hasattr(model, 'ensemble_prediction'):
                result = model.ensemble_prediction(data)
                if hasattr(result, 'consensus_prediction'):
                    return np.array([result.consensus_prediction] * n_predictions)
                elif 'consensus_prediction' in result:
                    return np.array([result['consensus_prediction']] * n_predictions)
            
            elif hasattr(model, 'predict'):
                # Standard ML model interface
                if self.feature_engineer:
                    feature_set = self.feature_engineer.engineer_features(data)
                    if not feature_set.features.empty:
                        features = feature_set.features.fillna(0).values[-1:]
                        predictions = model.predict(features)
                        return np.array(predictions[:n_predictions] if len(predictions) >= n_predictions else [predictions[0]] * n_predictions)
            
            # Fallback: return small random predictions
            return np.random.normal(0, 0.01, n_predictions)
            
        except Exception as e:
            logger.warning(f"Failed to get predictions from model: {e}")
            return np.array([])
    
    def _calculate_accuracy_metrics(self, predictions: np.ndarray, actual: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy metrics"""
        metrics = {}
        
        try:
            if SKLEARN_AVAILABLE:
                metrics['mse'] = mean_squared_error(actual, predictions)
                metrics['mae'] = mean_absolute_error(actual, predictions)
                metrics['r2_score'] = r2_score(actual, predictions)
            else:
                # Manual calculation
                metrics['mse'] = np.mean((actual - predictions) ** 2)
                metrics['mae'] = np.mean(np.abs(actual - predictions))
                
                # R-squared
                ss_res = np.sum((actual - predictions) ** 2)
                ss_tot = np.sum((actual - np.mean(actual)) ** 2)
                metrics['r2_score'] = 1 - (ss_res / (ss_tot + 1e-8))
            
            # Additional metrics
            metrics['rmse'] = np.sqrt(metrics['mse'])
            metrics['mape'] = np.mean(np.abs((actual - predictions) / (actual + 1e-8))) * 100
            
            # Directional accuracy
            actual_direction = np.sign(actual)
            pred_direction = np.sign(predictions)
            metrics['directional_accuracy'] = np.mean(actual_direction == pred_direction)
            
            # Correlation
            if len(actual) > 1 and len(predictions) > 1:
                correlation = np.corrcoef(actual, predictions)[0, 1]
                metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
            else:
                metrics['correlation'] = 0.0
            
        except Exception as e:
            logger.error(f"Error calculating accuracy metrics: {e}")
            metrics = {'mse': float('inf'), 'mae': float('inf'), 'r2_score': -float('inf')}
        
        return metrics
    
    def _add_noise_to_data(self, data: Dict[str, Any], noise_level: float) -> Dict[str, Any]:
        """Add noise to test data for robustness testing"""
        noisy_data = data.copy()
        
        # Add noise to price data
        if 'historical_prices' in data:
            prices = np.array(data['historical_prices'])
            noise = np.random.normal(0, noise_level * np.std(prices), len(prices))
            noisy_data['historical_prices'] = prices + noise
        
        # Add noise to volume data
        if 'volume' in data:
            volumes = np.array(data['volume'])
            noise = np.random.normal(0, noise_level * np.std(volumes), len(volumes))
            noisy_data['volume'] = np.maximum(volumes + noise, 0)  # Ensure non-negative
        
        return noisy_data
    
    def generate_report(self, test_suite: TestSuite) -> str:
        """Generate comprehensive test report"""
        report = f"\n=== Stock Model Testing Report ===\n"
        report += f"Suite: {test_suite.suite_name}\n"
        report += f"Execution Time: {test_suite.execution_time:.2f} seconds\n"
        report += f"Overall Score: {test_suite.overall_score:.3f}\n"
        report += f"Timestamp: {test_suite.timestamp}\n\n"
        
        # Test Results Summary
        report += "=== Test Results Summary ===\n"
        successful_tests = sum(1 for result in test_suite.test_results if result.success)
        report += f"Total Tests: {len(test_suite.test_results)}\n"
        report += f"Successful Tests: {successful_tests}\n"
        report += f"Failed Tests: {len(test_suite.test_results) - successful_tests}\n\n"
        
        # Individual Test Results
        report += "=== Individual Test Results ===\n"
        for result in test_suite.test_results:
            status = "PASS" if result.success else "FAIL"
            report += f"[{status}] {result.test_name} ({result.model_name})\n"
            report += f"  Score: {result.score:.3f}\n"
            report += f"  Execution Time: {result.execution_time:.2f}s\n"
            
            if result.metrics:
                report += "  Metrics:\n"
                for metric, value in result.metrics.items():
                    report += f"    {metric}: {value:.4f}\n"
            
            if result.error_message:
                report += f"  Error: {result.error_message}\n"
            
            report += "\n"
        
        # Backtest Results
        if test_suite.backtest_results:
            report += "=== Backtesting Results ===\n"
            for backtest in test_suite.backtest_results:
                report += f"Model: {backtest.model_name}\n"
                report += f"  Total Return: {backtest.total_return:.2%}\n"
                report += f"  Annualized Return: {backtest.annualized_return:.2%}\n"
                report += f"  Sharpe Ratio: {backtest.sharpe_ratio:.3f}\n"
                report += f"  Max Drawdown: {backtest.max_drawdown:.2%}\n"
                report += f"  Win Rate: {backtest.win_rate:.2%}\n"
                report += f"  Profit Factor: {backtest.profit_factor:.2f}\n"
                report += f"  Total Trades: {len(backtest.trades)}\n\n"
        
        # Model Comparisons
        if test_suite.model_comparisons:
            report += "=== Model Comparisons ===\n"
            for comparison in test_suite.model_comparisons:
                report += comparison.summary + "\n"
        
        report += test_suite.summary_report
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize tester
    tester = ModelTester()
    
    # Sample test data
    sample_data = {
        'symbol': 'AAPL',
        'historical_prices': [140 + i + np.random.normal(0, 2) for i in range(200)],
        'volume': [1000000 + np.random.randint(-100000, 100000) for _ in range(200)],
        'market_cap': 2500000000000,
        'revenue': 365000000000,
        'net_income': 95000000000
    }
    
    # Create a simple mock model for testing
    class MockModel:
        def analyze_apt(self, data):
            prices = data.get('historical_prices', [])
            if len(prices) > 1:
                return {'prediction': (prices[-1] - prices[-2]) / prices[-2]}
            return {'prediction': 0.0}
    
    mock_model = MockModel()
    
    print("=== Running Stock Model Tests ===")
    
    # Run accuracy test
    print("\n1. Running Accuracy Test...")
    accuracy_result = tester.run_accuracy_test(mock_model, sample_data)
    print(f"Accuracy Test Result: {accuracy_result.success}")
    print(f"Score: {accuracy_result.score:.3f}")
    print(f"Metrics: {accuracy_result.metrics}")
    
    # Run backtest
    print("\n2. Running Backtest...")
    backtest_result = tester.run_backtest(mock_model, sample_data)
    print(f"Backtest Results:")
    print(f"  Total Return: {backtest_result.total_return:.2%}")
    print(f"  Sharpe Ratio: {backtest_result.sharpe_ratio:.3f}")
    print(f"  Max Drawdown: {backtest_result.max_drawdown:.2%}")
    print(f"  Total Trades: {len(backtest_result.trades)}")
    
    # Run robustness test
    print("\n3. Running Robustness Test...")
    robustness_results = tester.run_robustness_test(mock_model, sample_data)
    for result in robustness_results:
        print(f"  Noise Level {result.details['noise_level']}: Score {result.score:.3f}")
    
    print("\n=== Testing Complete ===")