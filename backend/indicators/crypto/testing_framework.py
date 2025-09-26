from typing import Optional, Dict, Any, List, Tuple, Union, Callable
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Statistical imports
try:
    from scipy import stats
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy/Scikit-learn not available. Some testing features will be limited.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class CryptoTestType(Enum):
    ACCURACY = "accuracy"
    BACKTEST = "backtest"
    ROBUSTNESS = "robustness"
    STRESS_TEST = "stress_test"
    REGIME_TEST = "regime_test"
    CROSS_VALIDATION = "cross_validation"
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"

class CryptoMarketRegime(Enum):
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRYPTO_WINTER = "crypto_winter"
    ALTCOIN_SEASON = "altcoin_season"

@dataclass
class CryptoPerformanceMetrics:
    """Comprehensive performance metrics for crypto models"""
    # Accuracy metrics
    mse: float
    mae: float
    rmse: float
    mape: float
    r2_score: float
    hit_rate: float
    
    # Return-based metrics
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # Risk metrics
    max_drawdown: float
    var_95: float
    cvar_95: float
    downside_deviation: float
    
    # Trading metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Crypto-specific metrics
    crypto_correlation: float
    regime_consistency: float
    volatility_adjusted_return: float
    
    # Time-based metrics
    best_month: float
    worst_month: float
    consistency_score: float

@dataclass
class CryptoBacktestResult:
    """Backtest results for crypto models"""
    model_name: str
    test_period: Tuple[datetime, datetime]
    performance_metrics: CryptoPerformanceMetrics
    equity_curve: pd.Series
    trades: List[Dict[str, Any]]
    regime_performance: Dict[CryptoMarketRegime, float]
    monthly_returns: pd.Series
    drawdown_periods: List[Dict[str, Any]]
    risk_adjusted_metrics: Dict[str, float]
    benchmark_comparison: Dict[str, float]

@dataclass
class CryptoRobustnessResult:
    """Robustness test results"""
    model_name: str
    base_performance: float
    noise_sensitivity: Dict[str, float]
    parameter_sensitivity: Dict[str, float]
    regime_stability: Dict[CryptoMarketRegime, float]
    stress_test_results: Dict[str, float]
    monte_carlo_stats: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

class CryptoModelTester:
    """Comprehensive testing framework for crypto models"""
    
    def __init__(self, 
                 benchmark_symbol: str = 'BTC',
                 risk_free_rate: float = 0.02,
                 transaction_cost: float = 0.001,
                 min_test_period: int = 252):
        
        self.benchmark_symbol = benchmark_symbol
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.min_test_period = min_test_period
        
        # Test results storage
        self.test_results = {}
        self.comparison_results = {}
        
    def detect_market_regime(self, price_data: pd.Series, window: int = 60) -> pd.Series:
        """Detect market regimes for crypto"""
        returns = price_data.pct_change().dropna()
        
        # Rolling metrics
        rolling_return = returns.rolling(window).mean() * 252  # Annualized
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)  # Annualized
        
        regimes = pd.Series(index=returns.index, dtype='object')
        
        for i in range(len(returns)):
            if i < window:
                regimes.iloc[i] = CryptoMarketRegime.SIDEWAYS
                continue
            
            ret = rolling_return.iloc[i]
            vol = rolling_vol.iloc[i]
            
            # Crypto-specific regime detection
            if vol > 1.0:  # High volatility (>100% annualized)
                if ret > 0.5:  # High positive returns
                    regimes.iloc[i] = CryptoMarketRegime.BULL_MARKET
                elif ret < -0.3:  # Significant negative returns
                    regimes.iloc[i] = CryptoMarketRegime.CRYPTO_WINTER
                else:
                    regimes.iloc[i] = CryptoMarketRegime.HIGH_VOLATILITY
            elif vol < 0.4:  # Low volatility
                regimes.iloc[i] = CryptoMarketRegime.LOW_VOLATILITY
            else:  # Medium volatility
                if ret > 0.2:
                    regimes.iloc[i] = CryptoMarketRegime.BULL_MARKET
                elif ret < -0.2:
                    regimes.iloc[i] = CryptoMarketRegime.BEAR_MARKET
                else:
                    regimes.iloc[i] = CryptoMarketRegime.SIDEWAYS
        
        return regimes
    
    def calculate_performance_metrics(self, 
                                    predictions: np.ndarray,
                                    actuals: np.ndarray,
                                    returns: Optional[np.ndarray] = None,
                                    benchmark_returns: Optional[np.ndarray] = None) -> CryptoPerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        
        # Accuracy metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mse)
        
        # MAPE (handling division by zero)
        mape = np.mean(np.abs((actuals - predictions) / np.where(actuals != 0, actuals, 1))) * 100
        
        r2 = r2_score(actuals, predictions)
        
        # Hit rate (directional accuracy)
        if len(predictions) > 1:
            pred_direction = np.sign(np.diff(predictions))
            actual_direction = np.sign(np.diff(actuals))
            hit_rate = np.mean(pred_direction == actual_direction)
        else:
            hit_rate = 0.5
        
        # Return-based metrics (if returns provided)
        if returns is not None and len(returns) > 0:
            total_return = np.prod(1 + returns) - 1
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            
            # Risk-adjusted metrics
            excess_returns = returns - self.risk_free_rate / 252
            sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean(returns[returns <= var_95])
            
            # Trading metrics (simplified)
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            
            win_rate = len(positive_returns) / len(returns) if len(returns) > 0 else 0
            avg_win = np.mean(positive_returns) if len(positive_returns) > 0 else 0
            avg_loss = np.mean(negative_returns) if len(negative_returns) > 0 else 0
            profit_factor = abs(avg_win * len(positive_returns) / (avg_loss * len(negative_returns))) if avg_loss != 0 and len(negative_returns) > 0 else 0
            
            # Monthly analysis
            if len(returns) >= 30:
                monthly_returns = pd.Series(returns).resample('M').apply(lambda x: np.prod(1 + x) - 1)
                best_month = monthly_returns.max() if len(monthly_returns) > 0 else 0
                worst_month = monthly_returns.min() if len(monthly_returns) > 0 else 0
                consistency_score = len(monthly_returns[monthly_returns > 0]) / len(monthly_returns) if len(monthly_returns) > 0 else 0
            else:
                best_month = worst_month = consistency_score = 0
            
        else:
            # Default values when no returns provided
            total_return = annualized_return = volatility = 0
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
            max_drawdown = var_95 = cvar_95 = downside_deviation = 0
            win_rate = avg_win = avg_loss = profit_factor = 0
            best_month = worst_month = consistency_score = 0
        
        # Crypto-specific metrics
        crypto_correlation = 0.8  # Default high correlation with crypto market
        if benchmark_returns is not None and returns is not None:
            if len(benchmark_returns) == len(returns):
                crypto_correlation = np.corrcoef(returns, benchmark_returns)[0, 1]
        
        regime_consistency = hit_rate  # Simplified
        volatility_adjusted_return = annualized_return / volatility if volatility > 0 else 0
        
        return CryptoPerformanceMetrics(
            mse=mse, mae=mae, rmse=rmse, mape=mape, r2_score=r2,
            hit_rate=hit_rate, total_return=total_return,
            annualized_return=annualized_return, volatility=volatility,
            sharpe_ratio=sharpe_ratio, sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio, max_drawdown=max_drawdown,
            var_95=var_95, cvar_95=cvar_95, downside_deviation=downside_deviation,
            total_trades=len(returns) if returns is not None else 0,
            win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
            profit_factor=profit_factor, crypto_correlation=crypto_correlation,
            regime_consistency=regime_consistency,
            volatility_adjusted_return=volatility_adjusted_return,
            best_month=best_month, worst_month=worst_month,
            consistency_score=consistency_score
        )
    
    def run_accuracy_test(self, 
                         model: Callable,
                         test_data: pd.DataFrame,
                         target_column: str = 'close') -> Dict[str, Any]:
        """Run accuracy testing for crypto model"""
        
        logger.info("Running accuracy test...")
        
        # Generate predictions
        predictions = []
        actuals = []
        
        for i in range(len(test_data)):
            try:
                # Get prediction from model
                pred = model(test_data.iloc[:i+1])
                if isinstance(pred, (list, np.ndarray)):
                    pred = pred[-1] if len(pred) > 0 else 0
                
                predictions.append(float(pred))
                actuals.append(float(test_data[target_column].iloc[i]))
                
            except Exception as e:
                logger.warning(f"Error in prediction {i}: {e}")
                predictions.append(0.0)
                actuals.append(float(test_data[target_column].iloc[i]))
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        metrics = self.calculate_performance_metrics(predictions, actuals)
        
        return {
            'test_type': CryptoTestType.ACCURACY,
            'metrics': metrics,
            'predictions': predictions,
            'actuals': actuals,
            'test_period': (test_data.index[0], test_data.index[-1])
        }
    
    def run_backtest(self, 
                    model: Callable,
                    price_data: pd.DataFrame,
                    initial_capital: float = 100000,
                    position_size: float = 0.1) -> CryptoBacktestResult:
        """Run comprehensive backtest for crypto model"""
        
        logger.info("Running backtest...")
        
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        # Detect market regimes
        regimes = self.detect_market_regime(price_data['close'])
        regime_returns = {regime: [] for regime in CryptoMarketRegime}
        
        for i in range(1, len(price_data)):
            current_price = price_data['close'].iloc[i]
            prev_price = price_data['close'].iloc[i-1]
            
            try:
                # Get model prediction
                prediction = model(price_data.iloc[:i+1])
                if isinstance(prediction, (list, np.ndarray)):
                    prediction = prediction[-1] if len(prediction) > 0 else current_price
                
                # Generate trading signal
                expected_return = (prediction - current_price) / current_price
                
                # Simple trading logic
                if expected_return > 0.02 and position <= 0:  # Buy signal
                    if position < 0:  # Close short position
                        pnl = position * (prev_price - current_price)
                        capital += pnl
                        trades.append({
                            'date': price_data.index[i],
                            'action': 'cover',
                            'price': current_price,
                            'quantity': -position,
                            'pnl': pnl
                        })
                    
                    # Open long position
                    position = (capital * position_size) / current_price
                    capital -= position * current_price * (1 + self.transaction_cost)
                    trades.append({
                        'date': price_data.index[i],
                        'action': 'buy',
                        'price': current_price,
                        'quantity': position,
                        'pnl': 0
                    })
                
                elif expected_return < -0.02 and position >= 0:  # Sell signal
                    if position > 0:  # Close long position
                        pnl = position * (current_price - prev_price)
                        capital += position * current_price * (1 - self.transaction_cost)
                        trades.append({
                            'date': price_data.index[i],
                            'action': 'sell',
                            'price': current_price,
                            'quantity': position,
                            'pnl': pnl
                        })
                        position = 0
            
            except Exception as e:
                logger.warning(f"Error in backtest step {i}: {e}")
            
            # Calculate current equity
            current_equity = capital + position * current_price
            equity_curve.append(current_equity)
            
            # Track regime performance
            if i > 0 and i < len(regimes):
                regime = regimes.iloc[i]
                daily_return = (current_equity - equity_curve[-2]) / equity_curve[-2] if len(equity_curve) > 1 else 0
                regime_returns[regime].append(daily_return)
        
        # Calculate final metrics
        equity_series = pd.Series(equity_curve, index=price_data.index[1:])
        returns = equity_series.pct_change().dropna()
        
        # Benchmark returns (buy and hold)
        benchmark_returns = price_data['close'].pct_change().dropna()
        
        performance_metrics = self.calculate_performance_metrics(
            predictions=np.array([initial_capital] * len(returns)),
            actuals=equity_series.values,
            returns=returns.values,
            benchmark_returns=benchmark_returns.values
        )
        
        # Regime performance summary
        regime_performance = {}
        for regime, regime_rets in regime_returns.items():
            if regime_rets:
                regime_performance[regime] = np.mean(regime_rets) * 252  # Annualized
            else:
                regime_performance[regime] = 0.0
        
        # Monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: np.prod(1 + x) - 1)
        
        # Drawdown analysis
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max
        
        drawdown_periods = []
        in_drawdown = False
        start_date = None
        
        for date, dd in drawdown.items():
            if dd < -0.05 and not in_drawdown:  # Start of significant drawdown
                in_drawdown = True
                start_date = date
            elif dd >= -0.01 and in_drawdown:  # End of drawdown
                in_drawdown = False
                drawdown_periods.append({
                    'start': start_date,
                    'end': date,
                    'duration': (date - start_date).days,
                    'max_drawdown': drawdown[start_date:date].min()
                })
        
        return CryptoBacktestResult(
            model_name=getattr(model, '__name__', 'Unknown'),
            test_period=(price_data.index[0], price_data.index[-1]),
            performance_metrics=performance_metrics,
            equity_curve=equity_series,
            trades=trades,
            regime_performance=regime_performance,
            monthly_returns=monthly_returns,
            drawdown_periods=drawdown_periods,
            risk_adjusted_metrics={
                'sharpe_ratio': performance_metrics.sharpe_ratio,
                'sortino_ratio': performance_metrics.sortino_ratio,
                'calmar_ratio': performance_metrics.calmar_ratio
            },
            benchmark_comparison={
                'excess_return': performance_metrics.annualized_return - benchmark_returns.mean() * 252,
                'tracking_error': np.std(returns - benchmark_returns[:len(returns)]) * np.sqrt(252),
                'information_ratio': (performance_metrics.annualized_return - benchmark_returns.mean() * 252) / 
                                   (np.std(returns - benchmark_returns[:len(returns)]) * np.sqrt(252))
            }
        )
    
    def run_robustness_test(self, 
                           model: Callable,
                           test_data: pd.DataFrame,
                           noise_levels: List[float] = [0.01, 0.05, 0.1],
                           n_simulations: int = 100) -> CryptoRobustnessResult:
        """Run robustness testing with noise and parameter sensitivity"""
        
        logger.info("Running robustness test...")
        
        # Base performance
        base_result = self.run_accuracy_test(model, test_data)
        base_performance = base_result['metrics'].r2_score
        
        # Noise sensitivity testing
        noise_sensitivity = {}
        
        for noise_level in noise_levels:
            performances = []
            
            for _ in range(n_simulations):
                # Add noise to price data
                noisy_data = test_data.copy()
                noise = np.random.normal(0, noise_level, len(test_data))
                noisy_data['close'] *= (1 + noise)
                
                try:
                    result = self.run_accuracy_test(model, noisy_data)
                    performances.append(result['metrics'].r2_score)
                except:
                    performances.append(0.0)
            
            noise_sensitivity[f'noise_{noise_level}'] = np.mean(performances)
        
        # Regime stability (simplified)
        regimes = self.detect_market_regime(test_data['close'])
        regime_stability = {}
        
        for regime in CryptoMarketRegime:
            regime_mask = regimes == regime
            if regime_mask.sum() > 10:  # Enough data points
                regime_data = test_data[regime_mask]
                try:
                    result = self.run_accuracy_test(model, regime_data)
                    regime_stability[regime] = result['metrics'].r2_score
                except:
                    regime_stability[regime] = 0.0
            else:
                regime_stability[regime] = base_performance
        
        # Monte Carlo simulation
        mc_performances = []
        for _ in range(n_simulations):
            # Bootstrap sampling
            sample_indices = np.random.choice(len(test_data), size=len(test_data), replace=True)
            sample_data = test_data.iloc[sample_indices].reset_index(drop=True)
            
            try:
                result = self.run_accuracy_test(model, sample_data)
                mc_performances.append(result['metrics'].r2_score)
            except:
                mc_performances.append(0.0)
        
        monte_carlo_stats = {
            'mean': np.mean(mc_performances),
            'std': np.std(mc_performances),
            'min': np.min(mc_performances),
            'max': np.max(mc_performances),
            'percentile_5': np.percentile(mc_performances, 5),
            'percentile_95': np.percentile(mc_performances, 95)
        }
        
        # Confidence intervals
        confidence_intervals = {
            'performance_95': (np.percentile(mc_performances, 2.5), np.percentile(mc_performances, 97.5)),
            'noise_sensitivity_95': (np.min(list(noise_sensitivity.values())), np.max(list(noise_sensitivity.values())))
        }
        
        return CryptoRobustnessResult(
            model_name=getattr(model, '__name__', 'Unknown'),
            base_performance=base_performance,
            noise_sensitivity=noise_sensitivity,
            parameter_sensitivity={},  # Would require model-specific implementation
            regime_stability=regime_stability,
            stress_test_results={},  # Would require specific stress scenarios
            monte_carlo_stats=monte_carlo_stats,
            confidence_intervals=confidence_intervals
        )
    
    def compare_models(self, 
                      models: Dict[str, Callable],
                      test_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare multiple crypto models"""
        
        logger.info(f"Comparing {len(models)} models...")
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Testing model: {name}")
            
            # Run comprehensive tests
            accuracy_result = self.run_accuracy_test(model, test_data)
            robustness_result = self.run_robustness_test(model, test_data)
            
            results[name] = {
                'accuracy': accuracy_result,
                'robustness': robustness_result,
                'overall_score': self._calculate_overall_score(accuracy_result, robustness_result)
            }
        
        # Rank models
        ranked_models = sorted(results.items(), 
                             key=lambda x: x[1]['overall_score'], 
                             reverse=True)
        
        return {
            'individual_results': results,
            'ranking': ranked_models,
            'best_model': ranked_models[0][0] if ranked_models else None,
            'comparison_summary': self._generate_comparison_summary(results)
        }
    
    def _calculate_overall_score(self, accuracy_result: Dict, robustness_result: CryptoRobustnessResult) -> float:
        """Calculate overall model score"""
        accuracy_score = accuracy_result['metrics'].r2_score * 0.4
        robustness_score = robustness_result.base_performance * 0.3
        stability_score = np.mean(list(robustness_result.regime_stability.values())) * 0.3
        
        return accuracy_score + robustness_score + stability_score
    
    def _generate_comparison_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison summary"""
        summary = {
            'best_accuracy': max(results.items(), key=lambda x: x[1]['accuracy']['metrics'].r2_score),
            'best_robustness': max(results.items(), key=lambda x: x[1]['robustness'].base_performance),
            'most_stable': max(results.items(), key=lambda x: np.mean(list(x[1]['robustness'].regime_stability.values()))),
            'performance_range': {
                'min_r2': min(r['accuracy']['metrics'].r2_score for r in results.values()),
                'max_r2': max(r['accuracy']['metrics'].r2_score for r in results.values())
            }
        }
        
        return summary
    
    def generate_report(self, test_results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        
        report = "\n" + "="*60 + "\n"
        report += "CRYPTO MODEL TESTING REPORT\n"
        report += "="*60 + "\n"
        
        if 'individual_results' in test_results:
            # Model comparison report
            report += f"\nTested {len(test_results['individual_results'])} models\n"
            report += f"Best performing model: {test_results['best_model']}\n\n"
            
            for i, (model_name, _) in enumerate(test_results['ranking']):
                result = test_results['individual_results'][model_name]
                metrics = result['accuracy']['metrics']
                
                report += f"{i+1}. {model_name.upper()}\n"
                report += f"   R² Score: {metrics.r2_score:.4f}\n"
                report += f"   Hit Rate: {metrics.hit_rate:.4f}\n"
                report += f"   RMSE: {metrics.rmse:.4f}\n"
                report += f"   Overall Score: {result['overall_score']:.4f}\n\n"
        
        else:
            # Single model report
            if 'metrics' in test_results:
                metrics = test_results['metrics']
                report += "ACCURACY METRICS:\n"
                report += f"R² Score: {metrics.r2_score:.4f}\n"
                report += f"RMSE: {metrics.rmse:.4f}\n"
                report += f"MAE: {metrics.mae:.4f}\n"
                report += f"Hit Rate: {metrics.hit_rate:.4f}\n"
        
        report += "\n" + "="*60 + "\n"
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Create sample crypto data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate Bitcoin price data with crypto-like volatility
    n_days = len(dates)
    returns = np.random.randn(n_days) * 0.04  # Higher volatility for crypto
    prices = 45000 * np.exp(np.cumsum(returns))
    
    crypto_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_days) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(n_days)) * 0.03),
        'low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.03),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_days)
    }, index=dates)
    
    # Sample crypto models for testing
    def simple_trend_model(data):
        """Simple trend-following model"""
        if len(data) < 10:
            return data['close'].iloc[-1]
        
        sma_short = data['close'].tail(5).mean()
        sma_long = data['close'].tail(20).mean()
        
        if sma_short > sma_long:
            return data['close'].iloc[-1] * 1.02  # Bullish
        else:
            return data['close'].iloc[-1] * 0.98  # Bearish
    
    def mean_reversion_model(data):
        """Simple mean reversion model"""
        if len(data) < 20:
            return data['close'].iloc[-1]
        
        sma = data['close'].tail(20).mean()
        current_price = data['close'].iloc[-1]
        
        # Mean reversion logic
        deviation = (current_price - sma) / sma
        if abs(deviation) > 0.1:
            return sma  # Predict reversion to mean
        else:
            return current_price  # No strong signal
    
    # Initialize tester
    tester = CryptoModelTester()
    
    print("\n=== CRYPTO MODEL TESTING FRAMEWORK ===")
    
    # Test individual model
    print("\n--- SINGLE MODEL TEST ---")
    accuracy_result = tester.run_accuracy_test(simple_trend_model, crypto_data)
    print(f"R² Score: {accuracy_result['metrics'].r2_score:.4f}")
    print(f"Hit Rate: {accuracy_result['metrics'].hit_rate:.4f}")
    print(f"RMSE: {accuracy_result['metrics'].rmse:.2f}")
    
    # Test robustness
    print("\n--- ROBUSTNESS TEST ---")
    robustness_result = tester.run_robustness_test(simple_trend_model, crypto_data)
    print(f"Base Performance: {robustness_result.base_performance:.4f}")
    print("Noise Sensitivity:")
    for noise_level, performance in robustness_result.noise_sensitivity.items():
        print(f"  {noise_level}: {performance:.4f}")
    
    # Compare models
    print("\n--- MODEL COMPARISON ---")
    models = {
        'trend_model': simple_trend_model,
        'mean_reversion': mean_reversion_model
    }
    
    comparison_result = tester.compare_models(models, crypto_data)
    
    print(f"Best Model: {comparison_result['best_model']}")
    print("\nRanking:")
    for i, (name, _) in enumerate(comparison_result['ranking']):
        score = comparison_result['individual_results'][name]['overall_score']
        print(f"  {i+1}. {name}: {score:.4f}")
    
    # Generate report
    report = tester.generate_report(comparison_result)
    print(report)