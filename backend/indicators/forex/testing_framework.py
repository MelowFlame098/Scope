import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import json
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexTestType(Enum):
    ACCURACY = "accuracy"
    BACKTEST = "backtest"
    ROBUSTNESS = "robustness"
    STRESS = "stress"
    REGIME = "regime"
    CROSS_VALIDATION = "cross_validation"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"

class ForexMetricType(Enum):
    RETURN_BASED = "return_based"
    RISK_ADJUSTED = "risk_adjusted"
    DIRECTIONAL = "directional"
    STATISTICAL = "statistical"
    ECONOMIC = "economic"

@dataclass
class ForexPerformanceMetrics:
    """Comprehensive performance metrics for forex models"""
    
    # Basic accuracy metrics
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    r_squared: float = 0.0
    
    # Directional accuracy
    directional_accuracy: float = 0.0
    hit_rate: float = 0.0
    
    # Return-based metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    excess_return: float = 0.0
    
    # Risk metrics
    volatility: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    
    # Risk-adjusted metrics
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    information_ratio: float = 0.0
    
    # Trading metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    
    # Statistical significance
    t_statistic: float = 0.0
    p_value: float = 1.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Model-specific metrics
    prediction_stability: float = 0.0
    feature_importance_stability: float = 0.0
    regime_consistency: float = 0.0
    
    # Economic significance
    economic_significance: float = 0.0
    transaction_cost_impact: float = 0.0
    capacity_constraint: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'mse': self.mse,
            'rmse': self.rmse,
            'mae': self.mae,
            'mape': self.mape,
            'r_squared': self.r_squared,
            'directional_accuracy': self.directional_accuracy,
            'hit_rate': self.hit_rate,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'excess_return': self.excess_return,
            'volatility': self.volatility,
            'max_drawdown': self.max_drawdown,
            'var_95': self.var_95,
            'var_99': self.var_99,
            'expected_shortfall': self.expected_shortfall,
            'sharpe_ratio': self.sharpe_ratio,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'information_ratio': self.information_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            't_statistic': self.t_statistic,
            'p_value': self.p_value,
            'confidence_interval': self.confidence_interval,
            'prediction_stability': self.prediction_stability,
            'feature_importance_stability': self.feature_importance_stability,
            'regime_consistency': self.regime_consistency,
            'economic_significance': self.economic_significance,
            'transaction_cost_impact': self.transaction_cost_impact,
            'capacity_constraint': self.capacity_constraint
        }

@dataclass
class ForexBacktestResult:
    """Results from forex model backtesting"""
    model_name: str
    currency_pair: str
    start_date: datetime
    end_date: datetime
    
    # Performance metrics
    metrics: ForexPerformanceMetrics = field(default_factory=ForexPerformanceMetrics)
    
    # Time series data
    equity_curve: List[float] = field(default_factory=list)
    drawdown_series: List[float] = field(default_factory=list)
    returns_series: List[float] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)
    actuals: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    
    # Trade analysis
    trades: List[Dict[str, Any]] = field(default_factory=list)
    trade_statistics: Dict[str, float] = field(default_factory=dict)
    
    # Risk analysis
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    stress_test_results: Dict[str, float] = field(default_factory=dict)
    
    # Model diagnostics
    residual_analysis: Dict[str, float] = field(default_factory=dict)
    stability_metrics: Dict[str, float] = field(default_factory=dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of backtest results"""
        return {
            'model_name': self.model_name,
            'currency_pair': self.currency_pair,
            'period': f"{self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            'total_return': f"{self.metrics.total_return:.2%}",
            'annualized_return': f"{self.metrics.annualized_return:.2%}",
            'volatility': f"{self.metrics.volatility:.2%}",
            'sharpe_ratio': f"{self.metrics.sharpe_ratio:.2f}",
            'max_drawdown': f"{self.metrics.max_drawdown:.2%}",
            'win_rate': f"{self.metrics.win_rate:.2%}",
            'directional_accuracy': f"{self.metrics.directional_accuracy:.2%}",
            'total_trades': len(self.trades),
            'r_squared': f"{self.metrics.r_squared:.3f}"
        }

@dataclass
class ForexRobustnessResult:
    """Results from robustness testing"""
    model_name: str
    test_type: str
    
    # Robustness metrics
    parameter_sensitivity: Dict[str, float] = field(default_factory=dict)
    data_sensitivity: Dict[str, float] = field(default_factory=dict)
    regime_performance: Dict[str, ForexPerformanceMetrics] = field(default_factory=dict)
    
    # Cross-validation results
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Walk-forward analysis
    walk_forward_results: List[ForexPerformanceMetrics] = field(default_factory=list)
    
    # Monte Carlo results
    monte_carlo_results: Dict[str, List[float]] = field(default_factory=dict)
    
    # Stability metrics
    prediction_consistency: float = 0.0
    feature_stability: float = 0.0
    model_degradation: float = 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of robustness results"""
        return {
            'model_name': self.model_name,
            'test_type': self.test_type,
            'cv_mean': f"{self.cv_mean:.3f}",
            'cv_std': f"{self.cv_std:.3f}",
            'prediction_consistency': f"{self.prediction_consistency:.3f}",
            'feature_stability': f"{self.feature_stability:.3f}",
            'model_degradation': f"{self.model_degradation:.3f}",
            'regime_count': len(self.regime_performance)
        }

class ForexModelTester:
    """Comprehensive testing framework for forex models"""
    
    def __init__(self, risk_free_rate: float = 0.02, transaction_cost: float = 0.0001):
        """
        Initialize forex model tester
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
            transaction_cost: Transaction cost per trade (in basis points)
        """
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        
        logger.info(f"Initialized ForexModelTester with risk-free rate: {risk_free_rate:.2%}")
    
    def test_accuracy(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                     currency_pair: str = "EURUSD") -> ForexPerformanceMetrics:
        """Test model accuracy with comprehensive metrics"""
        try:
            logger.info(f"Testing accuracy for {currency_pair}")
            
            # Get predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
            else:
                raise ValueError("Model must have a 'predict' method")
            
            # Ensure predictions and actuals are 1D
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            if y_test.ndim > 1:
                y_test = y_test.flatten()
            
            metrics = ForexPerformanceMetrics()
            
            # Basic statistical metrics
            metrics.mse = mean_squared_error(y_test, predictions)
            metrics.rmse = np.sqrt(metrics.mse)
            metrics.mae = mean_absolute_error(y_test, predictions)
            
            # MAPE (handle division by zero)
            non_zero_mask = y_test != 0
            if np.any(non_zero_mask):
                metrics.mape = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask]))
            
            metrics.r_squared = r2_score(y_test, predictions)
            
            # Directional accuracy
            actual_direction = np.sign(y_test)
            predicted_direction = np.sign(predictions)
            metrics.directional_accuracy = np.mean(actual_direction == predicted_direction)
            metrics.hit_rate = metrics.directional_accuracy
            
            # Statistical significance
            if len(predictions) > 1:
                residuals = y_test - predictions
                metrics.t_statistic, metrics.p_value = stats.ttest_1samp(residuals, 0)
                
                # Confidence interval for mean residual
                confidence_level = 0.95
                degrees_freedom = len(residuals) - 1
                sample_mean = np.mean(residuals)
                sample_standard_error = stats.sem(residuals)
                confidence_interval = stats.t.interval(confidence_level, degrees_freedom, sample_mean, sample_standard_error)
                metrics.confidence_interval = confidence_interval
            
            # Prediction stability (consistency of predictions)
            if len(predictions) > 10:
                # Rolling correlation of predictions with a window
                window_size = min(20, len(predictions) // 2)
                rolling_corrs = []
                
                for i in range(window_size, len(predictions) - window_size):
                    corr = np.corrcoef(predictions[i-window_size:i], predictions[i:i+window_size])[0, 1]
                    if not np.isnan(corr):
                        rolling_corrs.append(corr)
                
                if rolling_corrs:
                    metrics.prediction_stability = np.mean(rolling_corrs)
            
            logger.info(f"Accuracy testing completed. R²: {metrics.r_squared:.3f}, Directional Accuracy: {metrics.directional_accuracy:.2%}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in accuracy testing: {e}")
            return ForexPerformanceMetrics()
    
    def backtest_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      prices: List[float], timestamps: List[datetime],
                      currency_pair: str = "EURUSD",
                      initial_capital: float = 100000.0,
                      position_size: float = 0.1) -> ForexBacktestResult:
        """Comprehensive backtesting of forex model"""
        try:
            logger.info(f"Starting backtest for {currency_pair}")
            
            # Get predictions
            predictions = model.predict(X_test)
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            if y_test.ndim > 1:
                y_test = y_test.flatten()
            
            # Initialize backtest result
            result = ForexBacktestResult(
                model_name=getattr(model, '__class__', type(model)).__name__,
                currency_pair=currency_pair,
                start_date=timestamps[0] if timestamps else datetime.now(),
                end_date=timestamps[-1] if timestamps else datetime.now()
            )
            
            # Initialize trading simulation
            capital = initial_capital
            position = 0.0
            equity_curve = [capital]
            returns_series = []
            trades = []
            
            # Simulate trading
            for i in range(1, len(predictions)):
                if i >= len(prices) or i >= len(y_test):
                    break
                
                current_price = prices[i]
                predicted_return = predictions[i]
                actual_return = y_test[i]
                
                # Trading signal based on prediction
                signal_threshold = 0.0001  # 1 pip threshold
                
                if predicted_return > signal_threshold and position <= 0:
                    # Buy signal
                    if position < 0:
                        # Close short position
                        pnl = -position * (current_price - prices[i-1]) - abs(position) * self.transaction_cost
                        capital += pnl
                        trades.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else datetime.now(),
                            'type': 'close_short',
                            'price': current_price,
                            'size': -position,
                            'pnl': pnl
                        })
                    
                    # Open long position
                    position = position_size * capital / current_price
                    trades.append({
                        'timestamp': timestamps[i] if i < len(timestamps) else datetime.now(),
                        'type': 'buy',
                        'price': current_price,
                        'size': position,
                        'pnl': 0
                    })
                    
                elif predicted_return < -signal_threshold and position >= 0:
                    # Sell signal
                    if position > 0:
                        # Close long position
                        pnl = position * (current_price - prices[i-1]) - position * self.transaction_cost
                        capital += pnl
                        trades.append({
                            'timestamp': timestamps[i] if i < len(timestamps) else datetime.now(),
                            'type': 'close_long',
                            'price': current_price,
                            'size': position,
                            'pnl': pnl
                        })
                    
                    # Open short position
                    position = -position_size * capital / current_price
                    trades.append({
                        'timestamp': timestamps[i] if i < len(timestamps) else datetime.now(),
                        'type': 'sell',
                        'price': current_price,
                        'size': position,
                        'pnl': 0
                    })
                
                # Calculate unrealized PnL
                if position != 0 and i > 0:
                    unrealized_pnl = position * (current_price - prices[i-1])
                    current_equity = capital + unrealized_pnl
                else:
                    current_equity = capital
                
                equity_curve.append(current_equity)
                
                # Calculate period return
                if len(equity_curve) > 1:
                    period_return = (current_equity - equity_curve[-2]) / equity_curve[-2]
                    returns_series.append(period_return)
            
            # Store results
            result.equity_curve = equity_curve
            result.returns_series = returns_series
            result.predictions = predictions.tolist()
            result.actuals = y_test.tolist()
            result.timestamps = timestamps
            result.trades = trades
            
            # Calculate performance metrics
            result.metrics = self._calculate_backtest_metrics(equity_curve, returns_series, trades, initial_capital)
            
            # Calculate additional metrics
            result.metrics.directional_accuracy = np.mean(np.sign(y_test[1:]) == np.sign(predictions[1:]))
            result.metrics.mse = mean_squared_error(y_test, predictions)
            result.metrics.r_squared = r2_score(y_test, predictions)
            
            # Calculate drawdown series
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (np.array(equity_curve) - peak) / peak
            result.drawdown_series = drawdown.tolist()
            
            logger.info(f"Backtest completed. Total Return: {result.metrics.total_return:.2%}, Sharpe: {result.metrics.sharpe_ratio:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in backtesting: {e}")
            return ForexBacktestResult(
                model_name="Unknown",
                currency_pair=currency_pair,
                start_date=datetime.now(),
                end_date=datetime.now()
            )
    
    def test_robustness(self, model: Any, X: np.ndarray, y: np.ndarray,
                       test_types: List[ForexTestType] = None,
                       cv_folds: int = 5) -> ForexRobustnessResult:
        """Test model robustness with various methods"""
        try:
            if test_types is None:
                test_types = [ForexTestType.CROSS_VALIDATION, ForexTestType.WALK_FORWARD]
            
            logger.info(f"Testing robustness with methods: {[t.value for t in test_types]}")
            
            result = ForexRobustnessResult(
                model_name=getattr(model, '__class__', type(model)).__name__,
                test_type="_".join([t.value for t in test_types])
            )
            
            # Cross-validation
            if ForexTestType.CROSS_VALIDATION in test_types:
                tscv = TimeSeriesSplit(n_splits=cv_folds)
                cv_scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    # Train model
                    if hasattr(model, 'fit'):
                        model.fit(X_train, y_train)
                    
                    # Predict and score
                    predictions = model.predict(X_val)
                    if predictions.ndim > 1:
                        predictions = predictions.flatten()
                    if y_val.ndim > 1:
                        y_val = y_val.flatten()
                    
                    score = r2_score(y_val, predictions)
                    cv_scores.append(score)
                
                result.cv_scores = cv_scores
                result.cv_mean = np.mean(cv_scores)
                result.cv_std = np.std(cv_scores)
            
            # Walk-forward analysis
            if ForexTestType.WALK_FORWARD in test_types:
                window_size = len(X) // 10  # 10% of data for each window
                walk_forward_results = []
                
                for i in range(window_size, len(X) - window_size, window_size // 2):
                    train_end = i
                    test_start = i
                    test_end = min(i + window_size, len(X))
                    
                    X_train = X[:train_end]
                    y_train = y[:train_end]
                    X_test = X[test_start:test_end]
                    y_test = y[test_start:test_end]
                    
                    if len(X_train) > 0 and len(X_test) > 0:
                        # Train model
                        if hasattr(model, 'fit'):
                            model.fit(X_train, y_train)
                        
                        # Test accuracy
                        metrics = self.test_accuracy(model, X_test, y_test)
                        walk_forward_results.append(metrics)
                
                result.walk_forward_results = walk_forward_results
            
            # Calculate stability metrics
            if result.walk_forward_results:
                r2_scores = [m.r_squared for m in result.walk_forward_results]
                directional_accuracies = [m.directional_accuracy for m in result.walk_forward_results]
                
                result.prediction_consistency = 1.0 - np.std(r2_scores) if r2_scores else 0.0
                result.feature_stability = 1.0 - np.std(directional_accuracies) if directional_accuracies else 0.0
                
                # Model degradation (trend in performance over time)
                if len(r2_scores) > 2:
                    x = np.arange(len(r2_scores))
                    slope, _, _, _, _ = stats.linregress(x, r2_scores)
                    result.model_degradation = -slope  # Negative slope indicates degradation
            
            logger.info(f"Robustness testing completed. CV Mean: {result.cv_mean:.3f}, Consistency: {result.prediction_consistency:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error in robustness testing: {e}")
            return ForexRobustnessResult(model_name="Unknown", test_type="error")
    
    def _calculate_backtest_metrics(self, equity_curve: List[float], 
                                   returns_series: List[float],
                                   trades: List[Dict[str, Any]],
                                   initial_capital: float) -> ForexPerformanceMetrics:
        """Calculate comprehensive backtest metrics"""
        try:
            metrics = ForexPerformanceMetrics()
            
            if not equity_curve or not returns_series:
                return metrics
            
            # Return metrics
            final_equity = equity_curve[-1]
            metrics.total_return = (final_equity - initial_capital) / initial_capital
            
            # Annualized return (assuming daily data)
            trading_days = len(returns_series)
            if trading_days > 0:
                metrics.annualized_return = (1 + metrics.total_return) ** (252 / trading_days) - 1
            
            # Risk metrics
            returns_array = np.array(returns_series)
            if len(returns_array) > 0:
                metrics.volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
                
                # VaR calculations
                metrics.var_95 = np.percentile(returns_array, 5)
                metrics.var_99 = np.percentile(returns_array, 1)
                metrics.expected_shortfall = np.mean(returns_array[returns_array <= metrics.var_95])
            
            # Drawdown
            peak = np.maximum.accumulate(equity_curve)
            drawdown = (np.array(equity_curve) - peak) / peak
            metrics.max_drawdown = np.min(drawdown)
            
            # Risk-adjusted metrics
            if metrics.volatility > 0:
                excess_return = metrics.annualized_return - self.risk_free_rate
                metrics.sharpe_ratio = excess_return / metrics.volatility
                
                # Sortino ratio (downside deviation)
                negative_returns = returns_array[returns_array < 0]
                if len(negative_returns) > 0:
                    downside_deviation = np.std(negative_returns) * np.sqrt(252)
                    metrics.sortino_ratio = excess_return / downside_deviation
                
                # Calmar ratio
                if metrics.max_drawdown < 0:
                    metrics.calmar_ratio = metrics.annualized_return / abs(metrics.max_drawdown)
            
            # Trading metrics
            if trades:
                profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
                losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
                
                metrics.win_rate = len(profitable_trades) / len(trades) if trades else 0.0
                
                if profitable_trades:
                    metrics.average_win = np.mean([t['pnl'] for t in profitable_trades])
                    metrics.largest_win = max([t['pnl'] for t in profitable_trades])
                
                if losing_trades:
                    metrics.average_loss = np.mean([t['pnl'] for t in losing_trades])
                    metrics.largest_loss = min([t['pnl'] for t in losing_trades])
                
                # Profit factor
                total_profit = sum([t['pnl'] for t in profitable_trades])
                total_loss = abs(sum([t['pnl'] for t in losing_trades]))
                
                if total_loss > 0:
                    metrics.profit_factor = total_profit / total_loss
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating backtest metrics: {e}")
            return ForexPerformanceMetrics()
    
    def generate_report(self, results: List[Union[ForexBacktestResult, ForexRobustnessResult]],
                       output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive testing report"""
        try:
            logger.info("Generating comprehensive testing report")
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {},
                'detailed_results': [],
                'comparative_analysis': {},
                'recommendations': []
            }
            
            # Process results
            backtest_results = [r for r in results if isinstance(r, ForexBacktestResult)]
            robustness_results = [r for r in results if isinstance(r, ForexRobustnessResult)]
            
            # Summary statistics
            if backtest_results:
                returns = [r.metrics.total_return for r in backtest_results]
                sharpe_ratios = [r.metrics.sharpe_ratio for r in backtest_results]
                max_drawdowns = [r.metrics.max_drawdown for r in backtest_results]
                
                report['summary']['backtest_summary'] = {
                    'num_models': len(backtest_results),
                    'avg_return': np.mean(returns),
                    'avg_sharpe': np.mean(sharpe_ratios),
                    'avg_max_drawdown': np.mean(max_drawdowns),
                    'best_model': max(backtest_results, key=lambda x: x.metrics.sharpe_ratio).model_name,
                    'worst_model': min(backtest_results, key=lambda x: x.metrics.sharpe_ratio).model_name
                }
            
            if robustness_results:
                cv_means = [r.cv_mean for r in robustness_results if r.cv_mean != 0]
                consistencies = [r.prediction_consistency for r in robustness_results]
                
                report['summary']['robustness_summary'] = {
                    'num_models': len(robustness_results),
                    'avg_cv_score': np.mean(cv_means) if cv_means else 0.0,
                    'avg_consistency': np.mean(consistencies),
                    'most_robust': max(robustness_results, key=lambda x: x.prediction_consistency).model_name if robustness_results else "N/A"
                }
            
            # Detailed results
            for result in results:
                if isinstance(result, ForexBacktestResult):
                    report['detailed_results'].append({
                        'type': 'backtest',
                        'model_name': result.model_name,
                        'summary': result.get_summary(),
                        'metrics': result.metrics.to_dict()
                    })
                elif isinstance(result, ForexRobustnessResult):
                    report['detailed_results'].append({
                        'type': 'robustness',
                        'model_name': result.model_name,
                        'summary': result.get_summary()
                    })
            
            # Comparative analysis
            if len(backtest_results) > 1:
                # Rank models by different criteria
                report['comparative_analysis']['rankings'] = {
                    'by_return': sorted([(r.model_name, r.metrics.total_return) for r in backtest_results], 
                                      key=lambda x: x[1], reverse=True),
                    'by_sharpe': sorted([(r.model_name, r.metrics.sharpe_ratio) for r in backtest_results], 
                                      key=lambda x: x[1], reverse=True),
                    'by_max_drawdown': sorted([(r.model_name, r.metrics.max_drawdown) for r in backtest_results], 
                                            key=lambda x: x[1], reverse=True),
                    'by_win_rate': sorted([(r.model_name, r.metrics.win_rate) for r in backtest_results], 
                                        key=lambda x: x[1], reverse=True)
                }
            
            # Recommendations
            recommendations = []
            
            if backtest_results:
                best_sharpe = max(backtest_results, key=lambda x: x.metrics.sharpe_ratio)
                if best_sharpe.metrics.sharpe_ratio > 1.0:
                    recommendations.append(f"Model '{best_sharpe.model_name}' shows excellent risk-adjusted returns (Sharpe: {best_sharpe.metrics.sharpe_ratio:.2f})")
                
                high_drawdown_models = [r for r in backtest_results if r.metrics.max_drawdown < -0.2]
                if high_drawdown_models:
                    recommendations.append(f"Consider risk management improvements for models with high drawdown: {[r.model_name for r in high_drawdown_models]}")
            
            if robustness_results:
                unstable_models = [r for r in robustness_results if r.prediction_consistency < 0.5]
                if unstable_models:
                    recommendations.append(f"Models showing instability need further validation: {[r.model_name for r in unstable_models]}")
            
            report['recommendations'] = recommendations
            
            # Save report if output file specified
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                logger.info(f"Report saved to {output_file}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {'error': str(e)}
    
    def plot_results(self, backtest_result: ForexBacktestResult, save_path: str = None):
        """Plot backtest results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Backtest Results: {backtest_result.model_name} - {backtest_result.currency_pair}', fontsize=16)
            
            # Equity curve
            axes[0, 0].plot(backtest_result.equity_curve)
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].set_ylabel('Portfolio Value')
            axes[0, 0].grid(True)
            
            # Drawdown
            axes[0, 1].fill_between(range(len(backtest_result.drawdown_series)), 
                                   backtest_result.drawdown_series, 0, alpha=0.3, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown %')
            axes[0, 1].grid(True)
            
            # Returns distribution
            if backtest_result.returns_series:
                axes[1, 0].hist(backtest_result.returns_series, bins=50, alpha=0.7)
                axes[1, 0].set_title('Returns Distribution')
                axes[1, 0].set_xlabel('Return')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True)
            
            # Predictions vs Actuals
            if backtest_result.predictions and backtest_result.actuals:
                sample_size = min(100, len(backtest_result.predictions))
                sample_pred = backtest_result.predictions[:sample_size]
                sample_actual = backtest_result.actuals[:sample_size]
                
                axes[1, 1].scatter(sample_actual, sample_pred, alpha=0.6)
                axes[1, 1].plot([min(sample_actual), max(sample_actual)], 
                               [min(sample_actual), max(sample_actual)], 'r--')
                axes[1, 1].set_title('Predictions vs Actuals')
                axes[1, 1].set_xlabel('Actual')
                axes[1, 1].set_ylabel('Predicted')
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting results: {e}")

# Example usage
if __name__ == "__main__":
    # Mock model for demonstration
    class MockForexModel:
        def __init__(self):
            self.coef_ = np.random.randn(10)
        
        def fit(self, X, y):
            pass
        
        def predict(self, X):
            return np.dot(X, self.coef_[:X.shape[1]])
    
    # Initialize tester
    tester = ForexModelTester(risk_free_rate=0.02, transaction_cost=0.0001)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples) * 0.01  # Small returns typical for forex
    prices = 1.1000 + np.cumsum(y)  # Price series
    timestamps = [datetime.now() + timedelta(days=i) for i in range(n_samples)]
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    test_prices = prices[split_idx:]
    test_timestamps = timestamps[split_idx:]
    
    # Create and test models
    models = {
        'Model_A': MockForexModel(),
        'Model_B': MockForexModel(),
        'Model_C': MockForexModel()
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTesting {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Test accuracy
        accuracy_metrics = tester.test_accuracy(model, X_test, y_test, "EURUSD")
        print(f"  Accuracy - R²: {accuracy_metrics.r_squared:.3f}, Directional: {accuracy_metrics.directional_accuracy:.2%}")
        
        # Backtest
        backtest_result = tester.backtest_model(
            model, X_test, y_test, test_prices, test_timestamps, "EURUSD"
        )
        results.append(backtest_result)
        print(f"  Backtest - Return: {backtest_result.metrics.total_return:.2%}, Sharpe: {backtest_result.metrics.sharpe_ratio:.2f}")
        
        # Robustness test
        robustness_result = tester.test_robustness(model, X_train, y_train)
        results.append(robustness_result)
        print(f"  Robustness - CV Mean: {robustness_result.cv_mean:.3f}, Consistency: {robustness_result.prediction_consistency:.3f}")
    
    # Generate comprehensive report
    report = tester.generate_report(results, "forex_testing_report.json")
    print(f"\nGenerated report with {len(results)} test results")
    print(f"Best model by Sharpe ratio: {report['summary'].get('backtest_summary', {}).get('best_model', 'N/A')}")
    
    # Plot results for first model
    backtest_results = [r for r in results if isinstance(r, ForexBacktestResult)]
    if backtest_results:
        print(f"\nPlotting results for {backtest_results[0].model_name}...")
        # tester.plot_results(backtest_results[0])  # Uncomment to show plots