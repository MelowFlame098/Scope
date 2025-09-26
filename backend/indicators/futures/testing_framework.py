import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestType(Enum):
    ACCURACY = "accuracy"
    BACKTEST = "backtest"
    ROBUSTNESS = "robustness"
    STRESS = "stress"
    REGIME = "regime"
    CROSS_VALIDATION = "cross_validation"
    WALK_FORWARD = "walk_forward"

class FuturesModelType(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    ML = "ml"
    ENSEMBLE = "ensemble"
    TERM_STRUCTURE = "term_structure"
    STATISTICAL = "statistical"

@dataclass
class FuturesPerformanceMetrics:
    """Comprehensive performance metrics for futures models"""
    model_name: str
    test_period: Tuple[datetime, datetime]
    
    # Accuracy metrics
    direction_accuracy: float = 0.0
    hit_rate: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Price prediction accuracy
    mae: float = 0.0  # Mean Absolute Error
    mse: float = 0.0  # Mean Squared Error
    rmse: float = 0.0  # Root Mean Squared Error
    mape: float = 0.0  # Mean Absolute Percentage Error
    r_squared: float = 0.0
    
    # Risk-adjusted returns
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Drawdown metrics
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    drawdown_duration: int = 0  # days
    recovery_factor: float = 0.0
    
    # Trading metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Futures-specific metrics
    roll_yield_accuracy: float = 0.0
    basis_prediction_accuracy: float = 0.0
    term_structure_accuracy: float = 0.0
    seasonality_capture: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    tail_ratio: float = 0.0
    
    # Consistency metrics
    monthly_win_rate: float = 0.0
    quarterly_consistency: float = 0.0
    annual_consistency: float = 0.0
    
    # Model-specific metrics
    prediction_stability: float = 0.0
    confidence_calibration: float = 0.0
    feature_importance_stability: float = 0.0
    
    # Additional statistics
    skewness: float = 0.0
    kurtosis: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

@dataclass
class FuturesBacktestResult:
    """Results from futures model backtesting"""
    model_name: str
    contract_symbol: str
    backtest_period: Tuple[datetime, datetime]
    
    # Performance metrics
    performance_metrics: FuturesPerformanceMetrics
    
    # Time series data
    equity_curve: List[float] = field(default_factory=list)
    drawdown_curve: List[float] = field(default_factory=list)
    returns_series: List[float] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)  # -1, 0, 1
    
    # Trade analysis
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    monthly_returns: Dict[str, float] = field(default_factory=dict)
    
    # Predictions vs actuals
    predictions: List[float] = field(default_factory=list)
    actual_values: List[float] = field(default_factory=list)
    prediction_dates: List[datetime] = field(default_factory=list)
    
    # Risk analysis
    rolling_sharpe: List[float] = field(default_factory=list)
    rolling_volatility: List[float] = field(default_factory=list)
    
    # Market regime performance
    regime_performance: Dict[str, float] = field(default_factory=dict)
    
    # Statistical tests
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    
    # Benchmark comparison
    benchmark_comparison: Dict[str, float] = field(default_factory=dict)

@dataclass
class FuturesRobustnessResult:
    """Results from robustness testing"""
    model_name: str
    test_type: str
    
    # Robustness scores
    parameter_sensitivity: float = 0.0
    data_quality_sensitivity: float = 0.0
    regime_stability: float = 0.0
    outlier_resistance: float = 0.0
    
    # Stress test results
    stress_test_results: Dict[str, float] = field(default_factory=dict)
    
    # Parameter sensitivity analysis
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    sensitivity_scores: Dict[str, float] = field(default_factory=dict)
    
    # Monte Carlo results
    monte_carlo_results: List[float] = field(default_factory=list)
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Regime analysis
    regime_breakdown: Dict[str, Dict[str, float]] = field(default_factory=dict)

class FuturesModelInterface(ABC):
    """Abstract interface for futures models to be tested"""
    
    @abstractmethod
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction given input data"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name"""
        pass
    
    @abstractmethod
    def get_model_type(self) -> FuturesModelType:
        """Get model type"""
        pass

class FuturesModelTester:
    """Comprehensive testing framework for futures models"""
    
    def __init__(self,
                 risk_free_rate: float = 0.02,
                 transaction_costs: float = 0.001,
                 slippage: float = 0.0005,
                 benchmark_return: float = 0.05):
        """
        Initialize the futures model tester
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            transaction_costs: Transaction costs as percentage
            slippage: Slippage as percentage
            benchmark_return: Benchmark return for comparison
        """
        self.risk_free_rate = risk_free_rate
        self.transaction_costs = transaction_costs
        self.slippage = slippage
        self.benchmark_return = benchmark_return
        
        # Test results storage
        self.test_results: Dict[str, Any] = {}
        
    def test_accuracy(self,
                     model: FuturesModelInterface,
                     test_data: List[Dict[str, Any]],
                     actual_values: List[float],
                     prediction_horizon: int = 1) -> FuturesPerformanceMetrics:
        """Test model prediction accuracy"""
        
        try:
            predictions = []
            confidences = []
            
            # Generate predictions
            for data_point in test_data:
                result = model.predict(data_point)
                predictions.append(result.get('prediction', 0.0))
                confidences.append(result.get('confidence', 0.5))
            
            predictions = np.array(predictions)
            actual_values = np.array(actual_values)
            
            # Calculate accuracy metrics
            mae = mean_absolute_error(actual_values, predictions)
            mse = mean_squared_error(actual_values, predictions)
            rmse = np.sqrt(mse)
            
            # MAPE (handling division by zero)
            mape = np.mean(np.abs((actual_values - predictions) / (actual_values + 1e-8))) * 100
            
            # R-squared
            r_squared = r2_score(actual_values, predictions)
            
            # Direction accuracy
            actual_directions = np.sign(np.diff(actual_values, prepend=actual_values[0]))
            predicted_directions = np.sign(np.diff(predictions, prepend=predictions[0]))
            direction_accuracy = np.mean(actual_directions == predicted_directions)
            
            # Create performance metrics
            metrics = FuturesPerformanceMetrics(
                model_name=model.get_model_name(),
                test_period=(datetime.now() - timedelta(days=len(test_data)), datetime.now()),
                direction_accuracy=direction_accuracy,
                mae=mae,
                mse=mse,
                rmse=rmse,
                mape=mape,
                r_squared=r_squared,
                hit_rate=direction_accuracy,
                total_trades=len(predictions)
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in accuracy testing: {e}")
            return FuturesPerformanceMetrics(
                model_name=model.get_model_name(),
                test_period=(datetime.now(), datetime.now())
            )
    
    def backtest_model(self,
                      model: FuturesModelInterface,
                      historical_data: List[Dict[str, Any]],
                      prices: List[float],
                      dates: List[datetime],
                      contract_symbol: str,
                      initial_capital: float = 100000,
                      position_size: float = 0.1) -> FuturesBacktestResult:
        """Comprehensive backtesting of futures model"""
        
        try:
            # Initialize backtest variables
            capital = initial_capital
            position = 0  # -1 (short), 0 (neutral), 1 (long)
            equity_curve = [capital]
            positions = [0]
            trade_log = []
            returns_series = []
            
            predictions = []
            actual_values = prices.copy()
            
            # Run backtest
            for i in range(1, len(historical_data)):
                try:
                    # Get model prediction
                    result = model.predict(historical_data[i-1])
                    prediction = result.get('prediction', 0.5)
                    confidence = result.get('confidence', 0.5)
                    
                    predictions.append(prediction)
                    
                    # Generate trading signal
                    if prediction > 0.6 and confidence > 0.7:
                        target_position = 1  # Long
                    elif prediction < 0.4 and confidence > 0.7:
                        target_position = -1  # Short
                    else:
                        target_position = 0  # Neutral
                    
                    # Calculate returns
                    price_return = (prices[i] - prices[i-1]) / prices[i-1]
                    
                    # Apply position from previous period
                    strategy_return = position * price_return
                    
                    # Apply transaction costs if position changes
                    if target_position != position:
                        strategy_return -= self.transaction_costs + self.slippage
                        
                        # Log trade
                        if position != 0:  # Closing position
                            trade_log.append({
                                'date': dates[i],
                                'action': 'close',
                                'position': position,
                                'price': prices[i],
                                'return': strategy_return
                            })
                        
                        if target_position != 0:  # Opening position
                            trade_log.append({
                                'date': dates[i],
                                'action': 'open',
                                'position': target_position,
                                'price': prices[i],
                                'confidence': confidence
                            })
                    
                    # Update capital and position
                    capital *= (1 + strategy_return)
                    position = target_position
                    
                    equity_curve.append(capital)
                    positions.append(position)
                    returns_series.append(strategy_return)
                    
                except Exception as e:
                    logger.warning(f"Error in backtest step {i}: {e}")
                    equity_curve.append(equity_curve[-1])
                    positions.append(positions[-1])
                    returns_series.append(0.0)
            
            # Calculate performance metrics
            returns_array = np.array(returns_series)
            
            # Basic performance
            total_return = (capital - initial_capital) / initial_capital
            annualized_return = (capital / initial_capital) ** (252 / len(returns_series)) - 1
            volatility = np.std(returns_array) * np.sqrt(252)
            
            # Risk-adjusted metrics
            sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            
            # Downside deviation for Sortino ratio
            downside_returns = returns_array[returns_array < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Drawdown analysis
            equity_array = np.array(equity_curve)
            running_max = np.maximum.accumulate(equity_array)
            drawdown = (equity_array - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Trading statistics
            winning_trades = sum(1 for trade in trade_log if trade.get('return', 0) > 0)
            losing_trades = sum(1 for trade in trade_log if trade.get('return', 0) < 0)
            total_trades = len([t for t in trade_log if t['action'] == 'close'])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Create performance metrics
            performance_metrics = FuturesPerformanceMetrics(
                model_name=model.get_model_name(),
                test_period=(dates[0], dates[-1]),
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate
            )
            
            # Create backtest result
            backtest_result = FuturesBacktestResult(
                model_name=model.get_model_name(),
                contract_symbol=contract_symbol,
                backtest_period=(dates[0], dates[-1]),
                performance_metrics=performance_metrics,
                equity_curve=equity_curve,
                drawdown_curve=drawdown.tolist(),
                returns_series=returns_series,
                positions=positions,
                trade_log=trade_log,
                predictions=predictions,
                actual_values=actual_values,
                prediction_dates=dates[1:]
            )
            
            return backtest_result
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return FuturesBacktestResult(
                model_name=model.get_model_name(),
                contract_symbol=contract_symbol,
                backtest_period=(dates[0] if dates else datetime.now(), 
                               dates[-1] if dates else datetime.now()),
                performance_metrics=FuturesPerformanceMetrics(
                    model_name=model.get_model_name(),
                    test_period=(datetime.now(), datetime.now())
                )
            )
    
    def test_robustness(self,
                       model: FuturesModelInterface,
                       test_data: List[Dict[str, Any]],
                       actual_values: List[float],
                       noise_levels: List[float] = [0.01, 0.05, 0.1],
                       outlier_ratios: List[float] = [0.01, 0.05, 0.1]) -> FuturesRobustnessResult:
        """Test model robustness to noise and outliers"""
        
        try:
            # Baseline performance
            baseline_metrics = self.test_accuracy(model, test_data, actual_values)
            baseline_accuracy = baseline_metrics.direction_accuracy
            
            robustness_scores = {}
            
            # Test noise sensitivity
            noise_scores = []
            for noise_level in noise_levels:
                # Add noise to test data
                noisy_data = []
                for data_point in test_data:
                    noisy_point = data_point.copy()
                    # Add noise to price data (assuming 'price' key exists)
                    if 'price' in noisy_point:
                        noise = np.random.normal(0, noise_level * noisy_point['price'])
                        noisy_point['price'] += noise
                    noisy_data.append(noisy_point)
                
                # Test with noisy data
                noisy_metrics = self.test_accuracy(model, noisy_data, actual_values)
                noise_score = noisy_metrics.direction_accuracy / baseline_accuracy
                noise_scores.append(noise_score)
            
            robustness_scores['noise_resistance'] = np.mean(noise_scores)
            
            # Test outlier sensitivity
            outlier_scores = []
            for outlier_ratio in outlier_ratios:
                # Add outliers to test data
                outlier_data = test_data.copy()
                n_outliers = int(len(outlier_data) * outlier_ratio)
                outlier_indices = np.random.choice(len(outlier_data), n_outliers, replace=False)
                
                for idx in outlier_indices:
                    if 'price' in outlier_data[idx]:
                        # Create extreme outlier
                        outlier_data[idx]['price'] *= np.random.choice([0.5, 2.0])  # 50% drop or 100% increase
                
                # Test with outlier data
                outlier_metrics = self.test_accuracy(model, outlier_data, actual_values)
                outlier_score = outlier_metrics.direction_accuracy / baseline_accuracy
                outlier_scores.append(outlier_score)
            
            robustness_scores['outlier_resistance'] = np.mean(outlier_scores)
            
            # Monte Carlo robustness test
            monte_carlo_scores = []
            for _ in range(100):
                # Random subsample of data
                sample_size = int(len(test_data) * 0.8)
                sample_indices = np.random.choice(len(test_data), sample_size, replace=False)
                
                sample_data = [test_data[i] for i in sample_indices]
                sample_actuals = [actual_values[i] for i in sample_indices]
                
                sample_metrics = self.test_accuracy(model, sample_data, sample_actuals)
                monte_carlo_scores.append(sample_metrics.direction_accuracy)
            
            # Create robustness result
            robustness_result = FuturesRobustnessResult(
                model_name=model.get_model_name(),
                test_type="robustness",
                parameter_sensitivity=robustness_scores.get('noise_resistance', 0.0),
                outlier_resistance=robustness_scores.get('outlier_resistance', 0.0),
                stress_test_results=robustness_scores,
                monte_carlo_results=monte_carlo_scores,
                confidence_intervals={
                    '95%': (np.percentile(monte_carlo_scores, 2.5), np.percentile(monte_carlo_scores, 97.5))
                }
            )
            
            return robustness_result
            
        except Exception as e:
            logger.error(f"Error in robustness testing: {e}")
            return FuturesRobustnessResult(
                model_name=model.get_model_name(),
                test_type="robustness"
            )
    
    def compare_models(self,
                      models: List[FuturesModelInterface],
                      test_data: List[Dict[str, Any]],
                      actual_values: List[float],
                      prices: List[float],
                      dates: List[datetime],
                      contract_symbol: str) -> Dict[str, Any]:
        """Compare multiple models across various metrics"""
        
        comparison_results = {
            'accuracy_results': {},
            'backtest_results': {},
            'robustness_results': {},
            'rankings': {},
            'statistical_tests': {}
        }
        
        try:
            # Test each model
            for model in models:
                model_name = model.get_model_name()
                
                # Accuracy test
                accuracy_result = self.test_accuracy(model, test_data, actual_values)
                comparison_results['accuracy_results'][model_name] = accuracy_result
                
                # Backtest
                backtest_result = self.backtest_model(model, test_data, prices, dates, contract_symbol)
                comparison_results['backtest_results'][model_name] = backtest_result
                
                # Robustness test
                robustness_result = self.test_robustness(model, test_data, actual_values)
                comparison_results['robustness_results'][model_name] = robustness_result
            
            # Create rankings
            metrics_to_rank = [
                ('direction_accuracy', 'accuracy_results', 'direction_accuracy'),
                ('sharpe_ratio', 'backtest_results', 'performance_metrics.sharpe_ratio'),
                ('max_drawdown', 'backtest_results', 'performance_metrics.max_drawdown'),
                ('total_return', 'backtest_results', 'performance_metrics.total_return'),
                ('robustness', 'robustness_results', 'outlier_resistance')
            ]
            
            for metric_name, result_type, metric_path in metrics_to_rank:
                model_scores = {}
                
                for model_name in comparison_results[result_type].keys():
                    result = comparison_results[result_type][model_name]
                    
                    # Navigate to metric value
                    value = result
                    for path_part in metric_path.split('.'):
                        if hasattr(value, path_part):
                            value = getattr(value, path_part)
                        else:
                            value = 0.0
                            break
                    
                    model_scores[model_name] = value
                
                # Rank models (higher is better, except for max_drawdown)
                reverse = metric_name != 'max_drawdown'
                ranked_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=reverse)
                comparison_results['rankings'][metric_name] = ranked_models
            
            # Statistical significance tests
            if len(models) >= 2:
                model_names = list(comparison_results['backtest_results'].keys())
                
                for i in range(len(model_names)):
                    for j in range(i+1, len(model_names)):
                        model1_name = model_names[i]
                        model2_name = model_names[j]
                        
                        returns1 = comparison_results['backtest_results'][model1_name].returns_series
                        returns2 = comparison_results['backtest_results'][model2_name].returns_series
                        
                        if len(returns1) > 10 and len(returns2) > 10:
                            # T-test for return difference
                            t_stat, p_value = stats.ttest_ind(returns1, returns2)
                            
                            comparison_results['statistical_tests'][f'{model1_name}_vs_{model2_name}'] = {
                                't_statistic': t_stat,
                                'p_value': p_value,
                                'significant': p_value < 0.05
                            }
            
        except Exception as e:
            logger.error(f"Error in model comparison: {e}")
        
        return comparison_results
    
    def generate_report(self,
                       results: Dict[str, Any],
                       output_file: Optional[str] = None) -> str:
        """Generate comprehensive testing report"""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FUTURES MODEL TESTING REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        try:
            # Accuracy Results
            if 'accuracy_results' in results:
                report_lines.append("ACCURACY RESULTS")
                report_lines.append("-" * 40)
                
                for model_name, metrics in results['accuracy_results'].items():
                    report_lines.append(f"\nModel: {model_name}")
                    report_lines.append(f"  Direction Accuracy: {metrics.direction_accuracy:.3f}")
                    report_lines.append(f"  MAE: {metrics.mae:.4f}")
                    report_lines.append(f"  RMSE: {metrics.rmse:.4f}")
                    report_lines.append(f"  MAPE: {metrics.mape:.2f}%")
                    report_lines.append(f"  R-squared: {metrics.r_squared:.3f}")
                
                report_lines.append("")
            
            # Backtest Results
            if 'backtest_results' in results:
                report_lines.append("BACKTEST RESULTS")
                report_lines.append("-" * 40)
                
                for model_name, backtest in results['backtest_results'].items():
                    metrics = backtest.performance_metrics
                    report_lines.append(f"\nModel: {model_name}")
                    report_lines.append(f"  Total Return: {metrics.total_return:.2%}")
                    report_lines.append(f"  Annualized Return: {metrics.annualized_return:.2%}")
                    report_lines.append(f"  Volatility: {metrics.volatility:.2%}")
                    report_lines.append(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
                    report_lines.append(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
                    report_lines.append(f"  Win Rate: {metrics.win_rate:.2%}")
                    report_lines.append(f"  Total Trades: {metrics.total_trades}")
                
                report_lines.append("")
            
            # Rankings
            if 'rankings' in results:
                report_lines.append("MODEL RANKINGS")
                report_lines.append("-" * 40)
                
                for metric_name, rankings in results['rankings'].items():
                    report_lines.append(f"\n{metric_name.replace('_', ' ').title()}:")
                    for i, (model_name, score) in enumerate(rankings, 1):
                        report_lines.append(f"  {i}. {model_name}: {score:.4f}")
                
                report_lines.append("")
            
            # Statistical Tests
            if 'statistical_tests' in results and results['statistical_tests']:
                report_lines.append("STATISTICAL SIGNIFICANCE TESTS")
                report_lines.append("-" * 40)
                
                for test_name, test_result in results['statistical_tests'].items():
                    significance = "Significant" if test_result['significant'] else "Not Significant"
                    report_lines.append(f"\n{test_name}:")
                    report_lines.append(f"  T-statistic: {test_result['t_statistic']:.3f}")
                    report_lines.append(f"  P-value: {test_result['p_value']:.4f}")
                    report_lines.append(f"  Result: {significance}")
                
                report_lines.append("")
            
            # Robustness Results
            if 'robustness_results' in results:
                report_lines.append("ROBUSTNESS RESULTS")
                report_lines.append("-" * 40)
                
                for model_name, robustness in results['robustness_results'].items():
                    report_lines.append(f"\nModel: {model_name}")
                    report_lines.append(f"  Parameter Sensitivity: {robustness.parameter_sensitivity:.3f}")
                    report_lines.append(f"  Outlier Resistance: {robustness.outlier_resistance:.3f}")
                    
                    if robustness.confidence_intervals:
                        ci_95 = robustness.confidence_intervals.get('95%', (0, 0))
                        report_lines.append(f"  95% Confidence Interval: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
                
                report_lines.append("")
            
            report_lines.append("=" * 80)
            report_lines.append("END OF REPORT")
            report_lines.append("=" * 80)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            report_lines.append(f"Error generating report: {e}")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write(report_text)
                logger.info(f"Report saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving report to file: {e}")
        
        return report_text

# Sample model implementations for testing
class SampleTechnicalModel(FuturesModelInterface):
    """Sample technical analysis model for testing"""
    
    def __init__(self, name: str = "Technical_Model"):
        self.name = name
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Simple moving average crossover strategy
        prices = data.get('prices', [100])
        
        if len(prices) >= 20:
            ma_5 = np.mean(prices[-5:])
            ma_20 = np.mean(prices[-20:])
            
            if ma_5 > ma_20:
                prediction = 0.7  # Bullish
            else:
                prediction = 0.3  # Bearish
        else:
            prediction = 0.5  # Neutral
        
        return {
            'prediction': prediction,
            'confidence': 0.6,
            'signal': 'BUY' if prediction > 0.6 else 'SELL' if prediction < 0.4 else 'HOLD'
        }
    
    def get_model_name(self) -> str:
        return self.name
    
    def get_model_type(self) -> FuturesModelType:
        return FuturesModelType.TECHNICAL

class SampleMLModel(FuturesModelInterface):
    """Sample ML model for testing"""
    
    def __init__(self, name: str = "ML_Model"):
        self.name = name
    
    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Simple random forest-like prediction
        prices = data.get('prices', [100])
        volumes = data.get('volumes', [1000])
        
        # Simple feature engineering
        if len(prices) >= 5:
            price_momentum = (prices[-1] - prices[-5]) / prices[-5]
            volume_ratio = volumes[-1] / np.mean(volumes[-5:]) if len(volumes) >= 5 else 1.0
            
            # Simple prediction logic
            prediction = 0.5 + 0.3 * np.tanh(price_momentum * 10) + 0.1 * np.tanh(volume_ratio - 1)
            prediction = max(0.1, min(0.9, prediction))  # Clip to reasonable range
            
            confidence = 0.7 + 0.2 * abs(price_momentum)
            confidence = max(0.5, min(0.95, confidence))
        else:
            prediction = 0.5
            confidence = 0.5
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'signal': 'BUY' if prediction > 0.6 else 'SELL' if prediction < 0.4 else 'HOLD'
        }
    
    def get_model_name(self) -> str:
        return self.name
    
    def get_model_type(self) -> FuturesModelType:
        return FuturesModelType.ML

# Example usage
if __name__ == "__main__":
    print("=== Futures Model Testing Framework Demo ===")
    
    # Generate sample data
    np.random.seed(42)
    n_days = 252  # One year of data
    
    # Generate price series with trend and noise
    base_price = 75.0
    trend = 0.0001  # Small upward trend
    volatility = 0.02
    
    prices = [base_price]
    for i in range(n_days):
        price_change = trend + np.random.normal(0, volatility)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)
    
    # Generate volume data
    volumes = np.random.lognormal(10, 0.5, n_days + 1).tolist()
    
    # Generate dates
    start_date = datetime.now() - timedelta(days=n_days)
    dates = [start_date + timedelta(days=i) for i in range(n_days + 1)]
    
    # Create test data
    test_data = []
    for i in range(len(prices)):
        test_data.append({
            'prices': prices[max(0, i-20):i+1],
            'volumes': volumes[max(0, i-20):i+1],
            'date': dates[i]
        })
    
    # Create sample models
    technical_model = SampleTechnicalModel("Technical_MA_Crossover")
    ml_model = SampleMLModel("Simple_ML_Model")
    
    models = [technical_model, ml_model]
    
    # Initialize tester
    tester = FuturesModelTester()
    
    print(f"\nTesting {len(models)} models on {len(test_data)} data points...")
    
    # Run comprehensive comparison
    comparison_results = tester.compare_models(
        models=models,
        test_data=test_data[:-50],  # Use most data for training/testing
        actual_values=prices[:-50],
        prices=prices,
        dates=dates,
        contract_symbol="CL_2024_06"
    )
    
    # Generate and display report
    report = tester.generate_report(comparison_results)
    print("\n" + report)
    
    # Individual model testing examples
    print("\n=== INDIVIDUAL MODEL TESTING ===")
    
    # Test accuracy
    accuracy_result = tester.test_accuracy(
        model=technical_model,
        test_data=test_data[-50:],  # Last 50 days
        actual_values=prices[-50:]
    )
    
    print(f"\nTechnical Model Accuracy:")
    print(f"  Direction Accuracy: {accuracy_result.direction_accuracy:.3f}")
    print(f"  MAE: {accuracy_result.mae:.4f}")
    print(f"  RMSE: {accuracy_result.rmse:.4f}")
    
    # Test robustness
    robustness_result = tester.test_robustness(
        model=ml_model,
        test_data=test_data[-50:],
        actual_values=prices[-50:]
    )
    
    print(f"\nML Model Robustness:")
    print(f"  Parameter Sensitivity: {robustness_result.parameter_sensitivity:.3f}")
    print(f"  Outlier Resistance: {robustness_result.outlier_resistance:.3f}")
    
    if robustness_result.confidence_intervals:
        ci_95 = robustness_result.confidence_intervals.get('95%', (0, 0))
        print(f"  95% Confidence Interval: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]")
    
    print("\n=== Futures Model Testing Framework Complete ===")