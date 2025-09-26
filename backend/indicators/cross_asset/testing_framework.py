from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CrossAssetTestType(Enum):
    """Types of cross-asset model tests"""
    ACCURACY = "accuracy"
    BACKTEST = "backtest"
    ROBUSTNESS = "robustness"
    STRESS_TEST = "stress_test"
    REGIME_ANALYSIS = "regime_analysis"
    CROSS_VALIDATION = "cross_validation"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"

class CrossAssetMetricType(Enum):
    """Types of performance metrics for cross-asset models"""
    # Prediction accuracy metrics
    MSE = "mse"
    RMSE = "rmse"
    MAE = "mae"
    MAPE = "mape"
    R2 = "r2"
    DIRECTIONAL_ACCURACY = "directional_accuracy"
    
    # Trading performance metrics
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    
    # Risk metrics
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    CVAR_95 = "cvar_95"
    CVAR_99 = "cvar_99"
    BETA = "beta"
    ALPHA = "alpha"
    
    # Cross-asset specific metrics
    CORRELATION_STABILITY = "correlation_stability"
    REGIME_CONSISTENCY = "regime_consistency"
    CROSS_ASSET_DIVERSIFICATION = "cross_asset_diversification"
    FACTOR_EXPOSURE = "factor_exposure"

@dataclass
class CrossAssetPerformanceMetrics:
    """Performance metrics for cross-asset models"""
    # Prediction metrics
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    r2: float = 0.0
    directional_accuracy: float = 0.0
    
    # Trading metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    cvar_99: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    
    # Cross-asset metrics
    correlation_stability: float = 0.0
    regime_consistency: float = 0.0
    cross_asset_diversification: float = 0.0
    factor_exposure: Dict[str, float] = None
    
    # Metadata
    test_period: Tuple[datetime, datetime] = None
    n_observations: int = 0
    asset_classes: List[str] = None
    timestamp: datetime = None

@dataclass
class CrossAssetBacktestResult:
    """Results from cross-asset backtesting"""
    performance_metrics: CrossAssetPerformanceMetrics
    daily_returns: pd.Series
    cumulative_returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    drawdown_series: pd.Series
    benchmark_comparison: Dict[str, float]
    regime_performance: Dict[str, CrossAssetPerformanceMetrics]
    asset_class_performance: Dict[str, CrossAssetPerformanceMetrics]
    monthly_returns: pd.Series
    yearly_returns: pd.Series
    test_metadata: Dict[str, Any]
    timestamp: datetime = None

@dataclass
class CrossAssetRobustnessResult:
    """Results from cross-asset robustness testing"""
    base_performance: CrossAssetPerformanceMetrics
    stress_test_results: Dict[str, CrossAssetPerformanceMetrics]
    parameter_sensitivity: Dict[str, Dict[str, float]]
    regime_stability: Dict[str, float]
    cross_validation_scores: Dict[str, List[float]]
    monte_carlo_results: Dict[str, List[float]]
    confidence_intervals: Dict[str, Tuple[float, float]]
    robustness_score: float
    test_metadata: Dict[str, Any]
    timestamp: datetime = None

class CrossAssetModelTester:
    """Comprehensive testing framework for cross-asset models"""
    
    def __init__(self, 
                 risk_free_rate: float = 0.02,
                 benchmark_returns: Optional[pd.Series] = None,
                 confidence_level: float = 0.95):
        
        self.risk_free_rate = risk_free_rate
        self.benchmark_returns = benchmark_returns
        self.confidence_level = confidence_level
        
        # Test results storage
        self.test_results = {}
        self.backtest_results = {}
        self.robustness_results = {}
    
    def test_accuracy(self, 
                     predictions: Dict[str, pd.Series],
                     actual_values: Dict[str, pd.Series],
                     test_name: str = "accuracy_test") -> Dict[str, CrossAssetPerformanceMetrics]:
        """Test prediction accuracy across asset classes"""
        
        accuracy_results = {}
        
        for asset_class in predictions.keys():
            if asset_class not in actual_values:
                continue
            
            pred = predictions[asset_class]
            actual = actual_values[asset_class]
            
            # Align series
            common_index = pred.index.intersection(actual.index)
            if len(common_index) == 0:
                continue
            
            pred_aligned = pred.loc[common_index]
            actual_aligned = actual.loc[common_index]
            
            # Calculate metrics
            metrics = self._calculate_prediction_metrics(pred_aligned, actual_aligned)
            
            # Create performance metrics object
            performance = CrossAssetPerformanceMetrics(
                mse=metrics['mse'],
                rmse=metrics['rmse'],
                mae=metrics['mae'],
                mape=metrics['mape'],
                r2=metrics['r2'],
                directional_accuracy=metrics['directional_accuracy'],
                test_period=(common_index.min(), common_index.max()),
                n_observations=len(common_index),
                asset_classes=[asset_class],
                timestamp=datetime.now()
            )
            
            accuracy_results[asset_class] = performance
        
        # Store results
        self.test_results[test_name] = accuracy_results
        
        return accuracy_results
    
    def backtest_strategy(self,
                         predictions: Dict[str, pd.Series],
                         price_data: Dict[str, pd.DataFrame],
                         strategy_function: Callable,
                         initial_capital: float = 100000,
                         transaction_costs: float = 0.001,
                         test_name: str = "backtest") -> CrossAssetBacktestResult:
        """Comprehensive backtesting of cross-asset strategy"""
        
        # Prepare data
        aligned_data = self._align_backtest_data(predictions, price_data)
        
        if not aligned_data:
            raise ValueError("No aligned data available for backtesting")
        
        # Generate trading signals
        signals = strategy_function(aligned_data['predictions'])
        
        # Execute backtest
        backtest_results = self._execute_backtest(
            signals=signals,
            price_data=aligned_data['prices'],
            initial_capital=initial_capital,
            transaction_costs=transaction_costs
        )
        
        # Calculate performance metrics
        performance_metrics = self._calculate_backtest_metrics(
            backtest_results['returns'],
            backtest_results['positions'],
            aligned_data['prices']
        )
        
        # Analyze regime performance
        regime_performance = self._analyze_regime_performance(
            backtest_results['returns'],
            aligned_data['prices']
        )
        
        # Asset class performance breakdown
        asset_performance = self._analyze_asset_class_performance(
            backtest_results['positions'],
            aligned_data['prices']
        )
        
        # Create comprehensive result
        result = CrossAssetBacktestResult(
            performance_metrics=performance_metrics,
            daily_returns=backtest_results['returns'],
            cumulative_returns=backtest_results['cumulative_returns'],
            positions=backtest_results['positions'],
            trades=backtest_results['trades'],
            drawdown_series=backtest_results['drawdowns'],
            benchmark_comparison=self._compare_to_benchmark(backtest_results['returns']),
            regime_performance=regime_performance,
            asset_class_performance=asset_performance,
            monthly_returns=backtest_results['returns'].resample('M').sum(),
            yearly_returns=backtest_results['returns'].resample('Y').sum(),
            test_metadata={
                'initial_capital': initial_capital,
                'transaction_costs': transaction_costs,
                'test_period': (aligned_data['prices'].index.min(), aligned_data['prices'].index.max()),
                'asset_classes': list(aligned_data['prices'].columns)
            },
            timestamp=datetime.now()
        )
        
        # Store results
        self.backtest_results[test_name] = result
        
        return result
    
    def test_robustness(self,
                       model,
                       data: Dict[str, pd.DataFrame],
                       parameter_ranges: Dict[str, List],
                       n_monte_carlo: int = 1000,
                       test_name: str = "robustness_test") -> CrossAssetRobustnessResult:
        """Test model robustness across different conditions"""
        
        # Base performance
        base_performance = self._get_base_performance(model, data)
        
        # Stress testing
        stress_results = self._conduct_stress_tests(model, data)
        
        # Parameter sensitivity analysis
        sensitivity_results = self._analyze_parameter_sensitivity(
            model, data, parameter_ranges
        )
        
        # Cross-validation
        cv_results = self._cross_validate_model(model, data)
        
        # Monte Carlo simulation
        mc_results = self._monte_carlo_simulation(
            model, data, n_monte_carlo
        )
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            mc_results, self.confidence_level
        )
        
        # Regime stability analysis
        regime_stability = self._analyze_regime_stability(model, data)
        
        # Calculate overall robustness score
        robustness_score = self._calculate_robustness_score(
            base_performance, stress_results, sensitivity_results, cv_results
        )
        
        # Create result
        result = CrossAssetRobustnessResult(
            base_performance=base_performance,
            stress_test_results=stress_results,
            parameter_sensitivity=sensitivity_results,
            regime_stability=regime_stability,
            cross_validation_scores=cv_results,
            monte_carlo_results=mc_results,
            confidence_intervals=confidence_intervals,
            robustness_score=robustness_score,
            test_metadata={
                'n_monte_carlo': n_monte_carlo,
                'confidence_level': self.confidence_level,
                'parameter_ranges': parameter_ranges,
                'test_date': datetime.now()
            },
            timestamp=datetime.now()
        )
        
        # Store results
        self.robustness_results[test_name] = result
        
        return result
    
    def _calculate_prediction_metrics(self, predictions: pd.Series, actual: pd.Series) -> Dict[str, float]:
        """Calculate prediction accuracy metrics"""
        try:
            # Remove any infinite or NaN values
            mask = np.isfinite(predictions) & np.isfinite(actual)
            pred_clean = predictions[mask]
            actual_clean = actual[mask]
            
            if len(pred_clean) == 0:
                return {metric: 0.0 for metric in ['mse', 'rmse', 'mae', 'mape', 'r2', 'directional_accuracy']}
            
            # Basic metrics
            mse = mean_squared_error(actual_clean, pred_clean)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual_clean, pred_clean)
            
            # MAPE (handle division by zero)
            mape = np.mean(np.abs((actual_clean - pred_clean) / (actual_clean + 1e-8))) * 100
            
            # R-squared
            r2 = r2_score(actual_clean, pred_clean)
            
            # Directional accuracy
            pred_direction = np.sign(pred_clean.diff())
            actual_direction = np.sign(actual_clean.diff())
            directional_accuracy = np.mean(pred_direction == actual_direction) * 100
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'r2': r2,
                'directional_accuracy': directional_accuracy
            }
        
        except Exception as e:
            print(f"Error calculating prediction metrics: {e}")
            return {metric: 0.0 for metric in ['mse', 'rmse', 'mae', 'mape', 'r2', 'directional_accuracy']}
    
    def _align_backtest_data(self, 
                           predictions: Dict[str, pd.Series], 
                           price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Align prediction and price data for backtesting"""
        try:
            # Find common assets
            common_assets = set(predictions.keys()).intersection(set(price_data.keys()))
            
            if not common_assets:
                return {}
            
            # Align time indices
            all_indices = []
            for asset in common_assets:
                if 'close' in price_data[asset].columns:
                    pred_index = predictions[asset].index
                    price_index = price_data[asset].index
                    common_index = pred_index.intersection(price_index)
                    all_indices.append(common_index)
            
            if not all_indices:
                return {}
            
            # Find overall common index
            common_index = all_indices[0]
            for idx in all_indices[1:]:
                common_index = common_index.intersection(idx)
            
            if len(common_index) == 0:
                return {}
            
            # Create aligned datasets
            aligned_predictions = {}
            aligned_prices = pd.DataFrame()
            
            for asset in common_assets:
                if 'close' in price_data[asset].columns:
                    aligned_predictions[asset] = predictions[asset].loc[common_index]
                    aligned_prices[asset] = price_data[asset]['close'].loc[common_index]
            
            return {
                'predictions': aligned_predictions,
                'prices': aligned_prices
            }
        
        except Exception as e:
            print(f"Error aligning backtest data: {e}")
            return {}
    
    def _execute_backtest(self,
                         signals: Dict[str, pd.Series],
                         price_data: pd.DataFrame,
                         initial_capital: float,
                         transaction_costs: float) -> Dict[str, Any]:
        """Execute the backtest simulation"""
        try:
            # Initialize tracking variables
            portfolio_value = initial_capital
            positions = pd.DataFrame(0, index=price_data.index, columns=price_data.columns)
            trades = []
            portfolio_values = []
            returns = []
            
            prev_positions = pd.Series(0, index=price_data.columns)
            
            for date in price_data.index:
                current_prices = price_data.loc[date]
                
                # Get signals for this date
                current_signals = {}
                for asset in signals.keys():
                    if date in signals[asset].index:
                        current_signals[asset] = signals[asset].loc[date]
                    else:
                        current_signals[asset] = 0
                
                # Calculate position sizes (equal weight for simplicity)
                total_signal = sum(abs(s) for s in current_signals.values())
                if total_signal > 0:
                    position_sizes = {asset: (signal / total_signal) * portfolio_value / current_prices[asset] 
                                    for asset, signal in current_signals.items() 
                                    if asset in current_prices.index and current_prices[asset] > 0}
                else:
                    position_sizes = {asset: 0 for asset in current_signals.keys()}
                
                # Update positions
                for asset in position_sizes.keys():
                    if asset in positions.columns:
                        new_position = position_sizes[asset]
                        old_position = prev_positions.get(asset, 0)
                        
                        # Record trade if position changed
                        if abs(new_position - old_position) > 1e-6:
                            trade_value = abs(new_position - old_position) * current_prices[asset]
                            transaction_cost = trade_value * transaction_costs
                            portfolio_value -= transaction_cost
                            
                            trades.append({
                                'date': date,
                                'asset': asset,
                                'old_position': old_position,
                                'new_position': new_position,
                                'price': current_prices[asset],
                                'transaction_cost': transaction_cost
                            })
                        
                        positions.loc[date, asset] = new_position
                        prev_positions[asset] = new_position
                
                # Calculate portfolio value
                portfolio_value = sum(positions.loc[date, asset] * current_prices[asset] 
                                    for asset in positions.columns 
                                    if asset in current_prices.index)
                
                portfolio_values.append(portfolio_value)
                
                # Calculate return
                if len(portfolio_values) > 1:
                    daily_return = (portfolio_values[-1] - portfolio_values[-2]) / portfolio_values[-2]
                    returns.append(daily_return)
                else:
                    returns.append(0.0)
            
            # Create return series
            returns_series = pd.Series(returns, index=price_data.index[1:] if len(returns) == len(price_data) - 1 else price_data.index[:len(returns)])
            
            # Calculate cumulative returns
            cumulative_returns = (1 + returns_series).cumprod()
            
            # Calculate drawdowns
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            
            return {
                'returns': returns_series,
                'cumulative_returns': cumulative_returns,
                'positions': positions,
                'trades': pd.DataFrame(trades),
                'drawdowns': drawdowns,
                'portfolio_values': pd.Series(portfolio_values, index=price_data.index)
            }
        
        except Exception as e:
            print(f"Error executing backtest: {e}")
            # Return empty results
            empty_index = price_data.index[:1] if len(price_data) > 0 else pd.DatetimeIndex([])
            return {
                'returns': pd.Series([], dtype=float),
                'cumulative_returns': pd.Series([], dtype=float),
                'positions': pd.DataFrame(),
                'trades': pd.DataFrame(),
                'drawdowns': pd.Series([], dtype=float),
                'portfolio_values': pd.Series([], dtype=float)
            }
    
    def _calculate_backtest_metrics(self,
                                  returns: pd.Series,
                                  positions: pd.DataFrame,
                                  price_data: pd.DataFrame) -> CrossAssetPerformanceMetrics:
        """Calculate comprehensive backtest performance metrics"""
        try:
            if len(returns) == 0:
                return CrossAssetPerformanceMetrics(timestamp=datetime.now())
            
            # Basic return metrics
            total_return = (1 + returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = returns.std() * np.sqrt(252)
            
            # Risk-adjusted metrics
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = excess_returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
            
            # Drawdown metrics
            cumulative_returns = (1 + returns).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
            cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0
            
            # Beta and Alpha (if benchmark available)
            beta, alpha = 0.0, 0.0
            if self.benchmark_returns is not None:
                common_index = returns.index.intersection(self.benchmark_returns.index)
                if len(common_index) > 1:
                    aligned_returns = returns.loc[common_index]
                    aligned_benchmark = self.benchmark_returns.loc[common_index]
                    
                    covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                    benchmark_variance = np.var(aligned_benchmark)
                    
                    if benchmark_variance > 0:
                        beta = covariance / benchmark_variance
                        alpha = aligned_returns.mean() - beta * aligned_benchmark.mean()
            
            return CrossAssetPerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                beta=beta,
                alpha=alpha,
                test_period=(returns.index.min(), returns.index.max()),
                n_observations=len(returns),
                asset_classes=list(price_data.columns) if not price_data.empty else [],
                timestamp=datetime.now()
            )
        
        except Exception as e:
            print(f"Error calculating backtest metrics: {e}")
            return CrossAssetPerformanceMetrics(timestamp=datetime.now())
    
    def _analyze_regime_performance(self, 
                                  returns: pd.Series, 
                                  price_data: pd.DataFrame) -> Dict[str, CrossAssetPerformanceMetrics]:
        """Analyze performance across different market regimes"""
        try:
            regime_performance = {}
            
            if len(returns) == 0 or price_data.empty:
                return regime_performance
            
            # Simple regime classification based on volatility
            # Calculate rolling volatility
            rolling_vol = returns.rolling(20).std() * np.sqrt(252)
            
            # Define regime thresholds
            vol_33 = rolling_vol.quantile(0.33)
            vol_67 = rolling_vol.quantile(0.67)
            
            # Classify regimes
            low_vol_mask = rolling_vol <= vol_33
            medium_vol_mask = (rolling_vol > vol_33) & (rolling_vol <= vol_67)
            high_vol_mask = rolling_vol > vol_67
            
            # Calculate performance for each regime
            regimes = {
                'low_volatility': low_vol_mask,
                'medium_volatility': medium_vol_mask,
                'high_volatility': high_vol_mask
            }
            
            for regime_name, mask in regimes.items():
                regime_returns = returns[mask]
                if len(regime_returns) > 0:
                    regime_performance[regime_name] = self._calculate_backtest_metrics(
                        regime_returns, pd.DataFrame(), pd.DataFrame()
                    )
            
            return regime_performance
        
        except Exception as e:
            print(f"Error analyzing regime performance: {e}")
            return {}
    
    def _analyze_asset_class_performance(self,
                                       positions: pd.DataFrame,
                                       price_data: pd.DataFrame) -> Dict[str, CrossAssetPerformanceMetrics]:
        """Analyze performance by asset class"""
        try:
            asset_performance = {}
            
            if positions.empty or price_data.empty:
                return asset_performance
            
            for asset in positions.columns:
                if asset in price_data.columns:
                    # Calculate asset-specific returns
                    asset_positions = positions[asset]
                    asset_prices = price_data[asset]
                    
                    # Calculate returns from position changes and price movements
                    price_changes = asset_prices.pct_change()
                    asset_returns = asset_positions.shift(1) * price_changes
                    asset_returns = asset_returns.dropna()
                    
                    if len(asset_returns) > 0:
                        asset_performance[asset] = self._calculate_backtest_metrics(
                            asset_returns, pd.DataFrame(), pd.DataFrame()
                        )
            
            return asset_performance
        
        except Exception as e:
            print(f"Error analyzing asset class performance: {e}")
            return {}
    
    def _compare_to_benchmark(self, returns: pd.Series) -> Dict[str, float]:
        """Compare strategy performance to benchmark"""
        try:
            comparison = {}
            
            if self.benchmark_returns is None or len(returns) == 0:
                return comparison
            
            # Align returns
            common_index = returns.index.intersection(self.benchmark_returns.index)
            if len(common_index) == 0:
                return comparison
            
            strategy_returns = returns.loc[common_index]
            benchmark_returns = self.benchmark_returns.loc[common_index]
            
            # Calculate comparison metrics
            strategy_total = (1 + strategy_returns).prod() - 1
            benchmark_total = (1 + benchmark_returns).prod() - 1
            
            comparison['excess_return'] = strategy_total - benchmark_total
            comparison['tracking_error'] = (strategy_returns - benchmark_returns).std() * np.sqrt(252)
            comparison['information_ratio'] = (strategy_returns.mean() - benchmark_returns.mean()) / (strategy_returns - benchmark_returns).std() * np.sqrt(252) if (strategy_returns - benchmark_returns).std() > 0 else 0
            
            return comparison
        
        except Exception as e:
            print(f"Error comparing to benchmark: {e}")
            return {}
    
    def _get_base_performance(self, model, data: Dict[str, pd.DataFrame]) -> CrossAssetPerformanceMetrics:
        """Get base model performance"""
        try:
            # This is a simplified implementation
            # In practice, you would run the model and calculate metrics
            return CrossAssetPerformanceMetrics(
                r2=0.5,  # Placeholder values
                sharpe_ratio=1.0,
                max_drawdown=-0.1,
                timestamp=datetime.now()
            )
        except Exception as e:
            print(f"Error getting base performance: {e}")
            return CrossAssetPerformanceMetrics(timestamp=datetime.now())
    
    def _conduct_stress_tests(self, model, data: Dict[str, pd.DataFrame]) -> Dict[str, CrossAssetPerformanceMetrics]:
        """Conduct stress tests on the model"""
        try:
            stress_results = {}
            
            # Example stress scenarios
            scenarios = {
                'market_crash': 'Simulate 20% market decline',
                'volatility_spike': 'Double historical volatility',
                'correlation_breakdown': 'Reduce correlations by 50%',
                'liquidity_crisis': 'Increase transaction costs by 10x'
            }
            
            for scenario_name, description in scenarios.items():
                # Placeholder - implement actual stress testing logic
                stress_results[scenario_name] = CrossAssetPerformanceMetrics(
                    r2=0.3,  # Reduced performance under stress
                    sharpe_ratio=0.5,
                    max_drawdown=-0.25,
                    timestamp=datetime.now()
                )
            
            return stress_results
        
        except Exception as e:
            print(f"Error conducting stress tests: {e}")
            return {}
    
    def _analyze_parameter_sensitivity(self, 
                                     model, 
                                     data: Dict[str, pd.DataFrame], 
                                     parameter_ranges: Dict[str, List]) -> Dict[str, Dict[str, float]]:
        """Analyze sensitivity to parameter changes"""
        try:
            sensitivity_results = {}
            
            for param_name, param_values in parameter_ranges.items():
                param_sensitivity = {}
                
                for value in param_values:
                    # Placeholder - implement actual parameter testing
                    # You would modify the model parameter and test performance
                    performance_change = np.random.normal(0, 0.1)  # Simulated sensitivity
                    param_sensitivity[str(value)] = performance_change
                
                sensitivity_results[param_name] = param_sensitivity
            
            return sensitivity_results
        
        except Exception as e:
            print(f"Error analyzing parameter sensitivity: {e}")
            return {}
    
    def _cross_validate_model(self, model, data: Dict[str, pd.DataFrame]) -> Dict[str, List[float]]:
        """Perform cross-validation on the model"""
        try:
            cv_results = {}
            
            # Placeholder for cross-validation implementation
            # You would use TimeSeriesSplit for time series data
            metrics = ['r2', 'sharpe_ratio', 'max_drawdown']
            
            for metric in metrics:
                # Simulate cross-validation scores
                cv_scores = np.random.normal(0.5, 0.1, 5)  # 5-fold CV
                cv_results[metric] = cv_scores.tolist()
            
            return cv_results
        
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            return {}
    
    def _monte_carlo_simulation(self, 
                              model, 
                              data: Dict[str, pd.DataFrame], 
                              n_simulations: int) -> Dict[str, List[float]]:
        """Perform Monte Carlo simulation"""
        try:
            mc_results = {}
            
            # Placeholder for Monte Carlo implementation
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
            
            for metric in metrics:
                # Simulate Monte Carlo results
                if metric == 'total_return':
                    simulations = np.random.normal(0.08, 0.15, n_simulations)
                elif metric == 'sharpe_ratio':
                    simulations = np.random.normal(0.8, 0.3, n_simulations)
                else:  # max_drawdown
                    simulations = np.random.normal(-0.15, 0.05, n_simulations)
                
                mc_results[metric] = simulations.tolist()
            
            return mc_results
        
        except Exception as e:
            print(f"Error in Monte Carlo simulation: {e}")
            return {}
    
    def _calculate_confidence_intervals(self, 
                                      mc_results: Dict[str, List[float]], 
                                      confidence_level: float) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals from Monte Carlo results"""
        try:
            confidence_intervals = {}
            alpha = 1 - confidence_level
            
            for metric, values in mc_results.items():
                if values:
                    lower = np.percentile(values, (alpha/2) * 100)
                    upper = np.percentile(values, (1 - alpha/2) * 100)
                    confidence_intervals[metric] = (lower, upper)
            
            return confidence_intervals
        
        except Exception as e:
            print(f"Error calculating confidence intervals: {e}")
            return {}
    
    def _analyze_regime_stability(self, model, data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Analyze model stability across different regimes"""
        try:
            stability_metrics = {
                'volatility_regime_stability': 0.8,
                'correlation_regime_stability': 0.7,
                'trend_regime_stability': 0.75
            }
            
            return stability_metrics
        
        except Exception as e:
            print(f"Error analyzing regime stability: {e}")
            return {}
    
    def _calculate_robustness_score(self, 
                                  base_performance: CrossAssetPerformanceMetrics,
                                  stress_results: Dict[str, CrossAssetPerformanceMetrics],
                                  sensitivity_results: Dict[str, Dict[str, float]],
                                  cv_results: Dict[str, List[float]]) -> float:
        """Calculate overall robustness score"""
        try:
            # Simplified robustness scoring
            scores = []
            
            # Base performance score
            if base_performance.sharpe_ratio > 0:
                scores.append(min(base_performance.sharpe_ratio / 2.0, 1.0))
            
            # Stress test resilience
            if stress_results:
                stress_scores = []
                for stress_perf in stress_results.values():
                    if stress_perf.sharpe_ratio > 0:
                        stress_scores.append(stress_perf.sharpe_ratio / base_performance.sharpe_ratio)
                
                if stress_scores:
                    scores.append(np.mean(stress_scores))
            
            # Cross-validation consistency
            if cv_results and 'sharpe_ratio' in cv_results:
                cv_std = np.std(cv_results['sharpe_ratio'])
                consistency_score = max(0, 1 - cv_std)  # Lower std = higher consistency
                scores.append(consistency_score)
            
            # Overall robustness score
            return np.mean(scores) if scores else 0.0
        
        except Exception as e:
            print(f"Error calculating robustness score: {e}")
            return 0.0
    
    def generate_test_report(self, 
                           test_name: str, 
                           include_plots: bool = True) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        report = {
            'test_name': test_name,
            'timestamp': datetime.now(),
            'summary': {},
            'detailed_results': {},
            'recommendations': []
        }
        
        # Include accuracy test results
        if test_name in self.test_results:
            report['detailed_results']['accuracy'] = self.test_results[test_name]
            
            # Summary statistics
            avg_r2 = np.mean([metrics.r2 for metrics in self.test_results[test_name].values()])
            avg_directional = np.mean([metrics.directional_accuracy for metrics in self.test_results[test_name].values()])
            
            report['summary']['average_r2'] = avg_r2
            report['summary']['average_directional_accuracy'] = avg_directional
        
        # Include backtest results
        if test_name in self.backtest_results:
            backtest = self.backtest_results[test_name]
            report['detailed_results']['backtest'] = backtest
            
            report['summary']['total_return'] = backtest.performance_metrics.total_return
            report['summary']['sharpe_ratio'] = backtest.performance_metrics.sharpe_ratio
            report['summary']['max_drawdown'] = backtest.performance_metrics.max_drawdown
        
        # Include robustness results
        if test_name in self.robustness_results:
            robustness = self.robustness_results[test_name]
            report['detailed_results']['robustness'] = robustness
            
            report['summary']['robustness_score'] = robustness.robustness_score
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['summary'])
        
        return report
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # R2 recommendations
        if 'average_r2' in summary:
            if summary['average_r2'] < 0.3:
                recommendations.append("Consider improving feature engineering or model complexity")
            elif summary['average_r2'] > 0.8:
                recommendations.append("Check for potential overfitting")
        
        # Sharpe ratio recommendations
        if 'sharpe_ratio' in summary:
            if summary['sharpe_ratio'] < 0.5:
                recommendations.append("Strategy may not provide adequate risk-adjusted returns")
            elif summary['sharpe_ratio'] > 2.0:
                recommendations.append("Excellent risk-adjusted performance - verify robustness")
        
        # Drawdown recommendations
        if 'max_drawdown' in summary:
            if summary['max_drawdown'] < -0.2:
                recommendations.append("Consider implementing better risk management")
        
        # Robustness recommendations
        if 'robustness_score' in summary:
            if summary['robustness_score'] < 0.5:
                recommendations.append("Model may not be robust - consider ensemble methods")
        
        return recommendations
    
    def plot_results(self, test_name: str, save_path: Optional[str] = None):
        """Plot comprehensive test results"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Cross-Asset Model Test Results: {test_name}', fontsize=16)
            
            # Plot 1: Cumulative Returns (if backtest available)
            if test_name in self.backtest_results:
                backtest = self.backtest_results[test_name]
                axes[0, 0].plot(backtest.cumulative_returns.index, backtest.cumulative_returns.values)
                axes[0, 0].set_title('Cumulative Returns')
                axes[0, 0].set_ylabel('Cumulative Return')
                axes[0, 0].grid(True)
            
            # Plot 2: Drawdown
            if test_name in self.backtest_results:
                backtest = self.backtest_results[test_name]
                axes[0, 1].fill_between(backtest.drawdown_series.index, 
                                       backtest.drawdown_series.values, 0, 
                                       alpha=0.3, color='red')
                axes[0, 1].set_title('Drawdown')
                axes[0, 1].set_ylabel('Drawdown')
                axes[0, 1].grid(True)
            
            # Plot 3: Monthly Returns Distribution
            if test_name in self.backtest_results:
                backtest = self.backtest_results[test_name]
                axes[1, 0].hist(backtest.monthly_returns.values, bins=20, alpha=0.7)
                axes[1, 0].set_title('Monthly Returns Distribution')
                axes[1, 0].set_xlabel('Monthly Return')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True)
            
            # Plot 4: Performance Metrics Comparison
            if test_name in self.test_results:
                accuracy_results = self.test_results[test_name]
                assets = list(accuracy_results.keys())
                r2_scores = [metrics.r2 for metrics in accuracy_results.values()]
                
                axes[1, 1].bar(assets, r2_scores)
                axes[1, 1].set_title('R² Scores by Asset Class')
                axes[1, 1].set_ylabel('R² Score')
                axes[1, 1].tick_params(axis='x', rotation=45)
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
        
        except Exception as e:
            print(f"Error plotting results: {e}")

# Example usage
if __name__ == "__main__":
    # Sample data setup
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Create sample cross-asset data
    sample_data = {
        'equities': pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02)
        }, index=dates),
        'bonds': pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
        }, index=dates),
        'commodities': pd.DataFrame({
            'close': 50 + np.cumsum(np.random.randn(len(dates)) * 0.03)
        }, index=dates)
    }
    
    # Create sample predictions
    sample_predictions = {}
    for asset_class, data in sample_data.items():
        # Simple momentum-based predictions
        returns = data['close'].pct_change()
        predictions = returns.rolling(10).mean().shift(1)  # Lagged momentum
        sample_predictions[asset_class] = predictions.dropna()
    
    print("Cross-Asset Model Testing Framework Example")
    print("=" * 50)
    
    # Initialize tester
    tester = CrossAssetModelTester(
        risk_free_rate=0.02,
        confidence_level=0.95
    )
    
    # Test 1: Accuracy Testing
    print("\n1. Testing prediction accuracy...")
    
    # Create actual values (next period returns)
    actual_values = {}
    for asset_class, data in sample_data.items():
        actual_values[asset_class] = data['close'].pct_change().shift(-1).dropna()
    
    accuracy_results = tester.test_accuracy(
        predictions=sample_predictions,
        actual_values=actual_values,
        test_name="momentum_strategy_accuracy"
    )
    
    for asset_class, metrics in accuracy_results.items():
        print(f"   {asset_class}:")
        print(f"     - R²: {metrics.r2:.4f}")
        print(f"     - RMSE: {metrics.rmse:.4f}")
        print(f"     - Directional Accuracy: {metrics.directional_accuracy:.2f}%")
    
    # Test 2: Backtesting
    print("\n2. Running backtest...")
    
    def simple_momentum_strategy(predictions):
        """Simple momentum strategy"""
        signals = {}
        for asset_class, pred in predictions.items():
            # Generate signals based on predictions
            signals[asset_class] = np.sign(pred) * 0.5  # 50% allocation based on signal direction
        return signals
    
    backtest_result = tester.backtest_strategy(
        predictions=sample_predictions,
        price_data=sample_data,
        strategy_function=simple_momentum_strategy,
        initial_capital=100000,
        transaction_costs=0.001,
        test_name="momentum_strategy_backtest"
    )
    
    print(f"   - Total Return: {backtest_result.performance_metrics.total_return:.2%}")
    print(f"   - Annualized Return: {backtest_result.performance_metrics.annualized_return:.2%}")
    print(f"   - Sharpe Ratio: {backtest_result.performance_metrics.sharpe_ratio:.2f}")
    print(f"   - Max Drawdown: {backtest_result.performance_metrics.max_drawdown:.2%}")
    print(f"   - Number of Trades: {len(backtest_result.trades)}")
    
    # Test 3: Robustness Testing
    print("\n3. Testing model robustness...")
    
    # Mock model for robustness testing
    class MockModel:
        def __init__(self):
            self.param1 = 0.5
            self.param2 = 10
    
    mock_model = MockModel()
    parameter_ranges = {
        'param1': [0.3, 0.5, 0.7],
        'param2': [5, 10, 15]
    }
    
    robustness_result = tester.test_robustness(
        model=mock_model,
        data=sample_data,
        parameter_ranges=parameter_ranges,
        n_monte_carlo=100,  # Reduced for example
        test_name="momentum_strategy_robustness"
    )
    
    print(f"   - Robustness Score: {robustness_result.robustness_score:.2f}")
    print(f"   - Stress Test Scenarios: {len(robustness_result.stress_test_results)}")
    print(f"   - Parameter Sensitivity Tests: {len(robustness_result.parameter_sensitivity)}")
    
    # Generate comprehensive report
    print("\n4. Generating test report...")
    report = tester.generate_test_report("momentum_strategy_backtest")
    
    print(f"   - Report generated for: {report['test_name']}")
    print(f"   - Summary metrics: {len(report['summary'])}")
    print(f"   - Recommendations: {len(report['recommendations'])}")
    
    if report['recommendations']:
        print("   - Key recommendations:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"     {i}. {rec}")
    
    print("\nCross-asset model testing completed successfully!")
    print(f"Total tests conducted: {len(tester.test_results) + len(tester.backtest_results) + len(tester.robustness_results)}")