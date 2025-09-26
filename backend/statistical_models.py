"""Statistical Models Service

Provides advanced statistical analysis and modeling capabilities including:
- Time series analysis (ARIMA, GARCH, VAR)
- Regression models (Linear, Polynomial, Ridge, Lasso)
- Statistical tests and hypothesis testing
- Risk metrics and portfolio optimization
- Monte Carlo simulations
- Correlation and cointegration analysis
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum

# Statistical libraries
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logging.warning("statsmodels not available. Time series analysis will be limited.")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    logging.warning("arch not available. GARCH models will not be available.")

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Statistical model types"""
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    ELASTIC_NET = "elastic_net"
    ARIMA = "arima"
    GARCH = "garch"
    VAR = "var"
    MONTE_CARLO = "monte_carlo"

class RiskMetric(Enum):
    """Risk metric types"""
    VALUE_AT_RISK = "var"
    CONDITIONAL_VAR = "cvar"
    MAXIMUM_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    VOLATILITY = "volatility"
    BETA = "beta"
    ALPHA = "alpha"

@dataclass
class ModelResult:
    """Statistical model result"""
    model_type: ModelType
    parameters: Dict[str, Any]
    fitted_values: pd.Series
    residuals: pd.Series
    metrics: Dict[str, float]
    predictions: Optional[pd.Series] = None
    confidence_intervals: Optional[pd.DataFrame] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class RiskAnalysis:
    """Risk analysis result"""
    metrics: Dict[RiskMetric, float]
    var_breakdown: Dict[str, float]
    stress_test_results: Dict[str, float]
    correlation_matrix: pd.DataFrame
    portfolio_weights: Optional[Dict[str, float]] = None
    optimal_portfolio: Optional[Dict[str, Any]] = None

@dataclass
class StatisticalTest:
    """Statistical test result"""
    test_name: str
    statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_significant: bool
    interpretation: str
    confidence_level: float = 0.05

class StatisticalModelsService:
    """Advanced statistical modeling service"""
    
    def __init__(self):
        self.models = {}
        self.risk_calculators = {}
        self._initialize_models()
        self._initialize_risk_calculators()
    
    def _initialize_models(self):
        """Initialize available statistical models"""
        self.models = {
            ModelType.LINEAR_REGRESSION: self._linear_regression,
            ModelType.POLYNOMIAL_REGRESSION: self._polynomial_regression,
            ModelType.RIDGE_REGRESSION: self._ridge_regression,
            ModelType.LASSO_REGRESSION: self._lasso_regression,
            ModelType.ELASTIC_NET: self._elastic_net,
            ModelType.ARIMA: self._arima_model,
            ModelType.GARCH: self._garch_model,
            ModelType.VAR: self._var_model,
            ModelType.MONTE_CARLO: self._monte_carlo_simulation,
        }
    
    def _initialize_risk_calculators(self):
        """Initialize risk metric calculators"""
        self.risk_calculators = {
            RiskMetric.VALUE_AT_RISK: self._calculate_var,
            RiskMetric.CONDITIONAL_VAR: self._calculate_cvar,
            RiskMetric.MAXIMUM_DRAWDOWN: self._calculate_max_drawdown,
            RiskMetric.SHARPE_RATIO: self._calculate_sharpe_ratio,
            RiskMetric.SORTINO_RATIO: self._calculate_sortino_ratio,
            RiskMetric.CALMAR_RATIO: self._calculate_calmar_ratio,
            RiskMetric.VOLATILITY: self._calculate_volatility,
            RiskMetric.BETA: self._calculate_beta,
            RiskMetric.ALPHA: self._calculate_alpha,
        }
    
    async def fit_model(self, 
                       data: pd.DataFrame, 
                       model_type: ModelType, 
                       target_column: str = 'close',
                       **kwargs) -> ModelResult:
        """Fit a statistical model to the data"""
        if model_type not in self.models:
            raise ValueError(f"Model type '{model_type}' not supported")
        
        try:
            result = await asyncio.to_thread(
                self.models[model_type], data, target_column, **kwargs
            )
            return result
        except Exception as e:
            logger.error(f"Error fitting {model_type}: {str(e)}")
            raise
    
    async def calculate_risk_metrics(self, 
                                   returns: pd.Series, 
                                   benchmark_returns: Optional[pd.Series] = None,
                                   confidence_level: float = 0.05) -> RiskAnalysis:
        """Calculate comprehensive risk metrics"""
        metrics = {}
        
        # Calculate individual risk metrics
        for metric_type in RiskMetric:
            try:
                if metric_type in [RiskMetric.BETA, RiskMetric.ALPHA] and benchmark_returns is None:
                    continue
                
                calculator = self.risk_calculators[metric_type]
                if metric_type in [RiskMetric.BETA, RiskMetric.ALPHA]:
                    value = await asyncio.to_thread(calculator, returns, benchmark_returns)
                else:
                    value = await asyncio.to_thread(calculator, returns, confidence_level)
                
                metrics[metric_type] = value
            except Exception as e:
                logger.error(f"Error calculating {metric_type}: {str(e)}")
                metrics[metric_type] = None
        
        # VaR breakdown by time horizons
        var_breakdown = {
            '1_day': self._calculate_var(returns, 0.05),
            '5_day': self._calculate_var(returns, 0.05) * np.sqrt(5),
            '10_day': self._calculate_var(returns, 0.05) * np.sqrt(10),
            '30_day': self._calculate_var(returns, 0.05) * np.sqrt(30),
        }
        
        # Stress test scenarios
        stress_tests = {
            'market_crash_2008': self._stress_test_scenario(returns, -0.20),
            'covid_crash_2020': self._stress_test_scenario(returns, -0.35),
            'flash_crash': self._stress_test_scenario(returns, -0.10),
            'high_volatility': self._stress_test_scenario(returns, returns.std() * 3),
        }
        
        # Correlation matrix (if multiple assets)
        correlation_matrix = pd.DataFrame()
        if isinstance(returns, pd.DataFrame):
            correlation_matrix = returns.corr()
        
        return RiskAnalysis(
            metrics=metrics,
            var_breakdown=var_breakdown,
            stress_test_results=stress_tests,
            correlation_matrix=correlation_matrix
        )
    
    async def perform_statistical_tests(self, 
                                      data: pd.Series, 
                                      test_type: str = 'stationarity') -> List[StatisticalTest]:
        """Perform statistical hypothesis tests"""
        tests = []
        
        if test_type == 'stationarity' or test_type == 'all':
            # Augmented Dickey-Fuller test
            if STATSMODELS_AVAILABLE:
                adf_result = adfuller(data.dropna())
                test = StatisticalTest(
                    test_name='Augmented Dickey-Fuller',
                    statistic=adf_result[0],
                    p_value=adf_result[1],
                    critical_values=adf_result[4],
                    is_significant=adf_result[1] < 0.05,
                    interpretation='Data is stationary' if adf_result[1] < 0.05 else 'Data is non-stationary'
                )
                tests.append(test)
        
        if test_type == 'normality' or test_type == 'all':
            # Shapiro-Wilk test for normality
            shapiro_stat, shapiro_p = stats.shapiro(data.dropna())
            test = StatisticalTest(
                test_name='Shapiro-Wilk',
                statistic=shapiro_stat,
                p_value=shapiro_p,
                critical_values={},
                is_significant=shapiro_p < 0.05,
                interpretation='Data is not normally distributed' if shapiro_p < 0.05 else 'Data is normally distributed'
            )
            tests.append(test)
            
            # Jarque-Bera test for normality
            jb_stat, jb_p = stats.jarque_bera(data.dropna())
            test = StatisticalTest(
                test_name='Jarque-Bera',
                statistic=jb_stat,
                p_value=jb_p,
                critical_values={},
                is_significant=jb_p < 0.05,
                interpretation='Data is not normally distributed' if jb_p < 0.05 else 'Data is normally distributed'
            )
            tests.append(test)
        
        return tests
    
    async def optimize_portfolio(self, 
                               returns: pd.DataFrame, 
                               method: str = 'mean_variance',
                               constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Optimize portfolio allocation"""
        if method == 'mean_variance':
            return await self._mean_variance_optimization(returns, constraints)
        elif method == 'risk_parity':
            return await self._risk_parity_optimization(returns, constraints)
        elif method == 'minimum_variance':
            return await self._minimum_variance_optimization(returns, constraints)
        else:
            raise ValueError(f"Optimization method '{method}' not supported")
    
    # Model Implementation Methods
    def _linear_regression(self, data: pd.DataFrame, target_column: str, **kwargs) -> ModelResult:
        """Linear regression model"""
        # Prepare features (use all numeric columns except target)
        feature_columns = [col for col in data.select_dtypes(include=[np.number]).columns 
                          if col != target_column]
        
        if not feature_columns:
            # If no other features, use time-based features
            data_copy = data.copy()
            data_copy['time_index'] = range(len(data_copy))
            feature_columns = ['time_index']
            X = data_copy[feature_columns]
        else:
            X = data[feature_columns]
        
        y = data[target_column]
        
        # Remove NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        # Fit model
        model = LinearRegression()
        model.fit(X_clean, y_clean)
        
        # Predictions and residuals
        fitted_values = pd.Series(model.predict(X_clean), index=y_clean.index)
        residuals = y_clean - fitted_values
        
        # Calculate metrics
        r2 = r2_score(y_clean, fitted_values)
        mse = mean_squared_error(y_clean, fitted_values)
        mae = mean_absolute_error(y_clean, fitted_values)
        
        metrics = {
            'r_squared': r2,
            'mse': mse,
            'mae': mae,
            'rmse': np.sqrt(mse)
        }
        
        return ModelResult(
            model_type=ModelType.LINEAR_REGRESSION,
            parameters={'coefficients': model.coef_, 'intercept': model.intercept_},
            fitted_values=fitted_values,
            residuals=residuals,
            metrics=metrics,
            metadata={'features': feature_columns}
        )
    
    def _arima_model(self, data: pd.DataFrame, target_column: str, 
                    order: Tuple[int, int, int] = (1, 1, 1), **kwargs) -> ModelResult:
        """ARIMA time series model"""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ARIMA models")
        
        series = data[target_column].dropna()
        
        # Fit ARIMA model
        model = ARIMA(series, order=order)
        fitted_model = model.fit()
        
        # Get fitted values and residuals
        fitted_values = fitted_model.fittedvalues
        residuals = fitted_model.resid
        
        # Calculate metrics
        aic = fitted_model.aic
        bic = fitted_model.bic
        
        metrics = {
            'aic': aic,
            'bic': bic,
            'log_likelihood': fitted_model.llf
        }
        
        # Generate forecasts
        forecast_steps = kwargs.get('forecast_steps', 10)
        forecast = fitted_model.forecast(steps=forecast_steps)
        
        return ModelResult(
            model_type=ModelType.ARIMA,
            parameters={'order': order, 'coefficients': fitted_model.params.to_dict()},
            fitted_values=fitted_values,
            residuals=residuals,
            metrics=metrics,
            predictions=forecast,
            summary=str(fitted_model.summary())
        )
    
    def _monte_carlo_simulation(self, data: pd.DataFrame, target_column: str, 
                               num_simulations: int = 1000, 
                               time_horizon: int = 252, **kwargs) -> ModelResult:
        """Monte Carlo simulation for price forecasting"""
        returns = data[target_column].pct_change().dropna()
        
        # Calculate parameters
        mean_return = returns.mean()
        std_return = returns.std()
        last_price = data[target_column].iloc[-1]
        
        # Run simulations
        simulations = []
        for _ in range(num_simulations):
            # Generate random returns
            random_returns = np.random.normal(mean_return, std_return, time_horizon)
            
            # Calculate price path
            price_path = [last_price]
            for ret in random_returns:
                price_path.append(price_path[-1] * (1 + ret))
            
            simulations.append(price_path[1:])  # Exclude initial price
        
        # Convert to DataFrame
        simulations_df = pd.DataFrame(simulations).T
        
        # Calculate statistics
        mean_path = simulations_df.mean(axis=1)
        percentile_5 = simulations_df.quantile(0.05, axis=1)
        percentile_95 = simulations_df.quantile(0.95, axis=1)
        
        # Create confidence intervals
        confidence_intervals = pd.DataFrame({
            'lower_5': percentile_5,
            'upper_95': percentile_95
        })
        
        metrics = {
            'final_price_mean': mean_path.iloc[-1],
            'final_price_std': simulations_df.iloc[-1].std(),
            'probability_positive': (simulations_df.iloc[-1] > last_price).mean(),
            'var_5': np.percentile(simulations_df.iloc[-1], 5),
            'var_1': np.percentile(simulations_df.iloc[-1], 1)
        }
        
        return ModelResult(
            model_type=ModelType.MONTE_CARLO,
            parameters={
                'num_simulations': num_simulations,
                'time_horizon': time_horizon,
                'mean_return': mean_return,
                'std_return': std_return
            },
            fitted_values=pd.Series(mean_path),
            residuals=pd.Series([]),  # Not applicable for MC
            metrics=metrics,
            predictions=pd.Series(mean_path),
            confidence_intervals=confidence_intervals,
            metadata={'all_simulations': simulations_df}
        )
    
    # Risk Calculation Methods
    def _calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        return np.percentile(returns.dropna(), confidence_level * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        var = self._calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    def _calculate_max_drawdown(self, returns: pd.Series, confidence_level: float = None) -> float:
        """Calculate Maximum Drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, confidence_level: float = None, 
                               risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate  # Annualized
        volatility = returns.std() * np.sqrt(252)  # Annualized
        return excess_returns / volatility if volatility != 0 else 0
    
    def _calculate_sortino_ratio(self, returns: pd.Series, confidence_level: float = None,
                                risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio"""
        excess_returns = returns.mean() * 252 - risk_free_rate
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252)
        return excess_returns / downside_volatility if downside_volatility != 0 else 0
    
    def _calculate_calmar_ratio(self, returns: pd.Series, confidence_level: float = None) -> float:
        """Calculate Calmar Ratio"""
        annual_return = returns.mean() * 252
        max_drawdown = abs(self._calculate_max_drawdown(returns))
        return annual_return / max_drawdown if max_drawdown != 0 else 0
    
    def _calculate_volatility(self, returns: pd.Series, confidence_level: float = None) -> float:
        """Calculate annualized volatility"""
        return returns.std() * np.sqrt(252)
    
    def _calculate_beta(self, returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate Beta relative to benchmark"""
        covariance = np.cov(returns.dropna(), benchmark_returns.dropna())[0][1]
        benchmark_variance = benchmark_returns.var()
        return covariance / benchmark_variance if benchmark_variance != 0 else 0
    
    def _calculate_alpha(self, returns: pd.Series, benchmark_returns: pd.Series,
                        risk_free_rate: float = 0.02) -> float:
        """Calculate Alpha relative to benchmark"""
        beta = self._calculate_beta(returns, benchmark_returns)
        portfolio_return = returns.mean() * 252
        benchmark_return = benchmark_returns.mean() * 252
        return portfolio_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
    
    def _stress_test_scenario(self, returns: pd.Series, shock: float) -> float:
        """Apply stress test scenario"""
        stressed_returns = returns + shock
        return stressed_returns.sum()
    
    # Portfolio Optimization Methods
    async def _mean_variance_optimization(self, returns: pd.DataFrame, 
                                        constraints: Optional[Dict] = None) -> Dict[str, Any]:
        """Mean-variance portfolio optimization"""
        mean_returns = returns.mean() * 252  # Annualized
        cov_matrix = returns.cov() * 252  # Annualized
        
        num_assets = len(returns.columns)
        
        # Objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility  # Negative for maximization
        
        # Constraints
        constraints_list = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Bounds (0 to 1 for long-only portfolio)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess (equal weights)
        initial_guess = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(objective, initial_guess, method='SLSQP', 
                         bounds=bounds, constraints=constraints_list)
        
        optimal_weights = result.x
        optimal_return = np.sum(mean_returns * optimal_weights)
        optimal_volatility = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        optimal_sharpe = optimal_return / optimal_volatility
        
        return {
            'weights': dict(zip(returns.columns, optimal_weights)),
            'expected_return': optimal_return,
            'volatility': optimal_volatility,
            'sharpe_ratio': optimal_sharpe,
            'optimization_success': result.success
        }
    
    # Placeholder methods for other models
    def _polynomial_regression(self, data, target_column, degree=2, **kwargs): pass
    def _ridge_regression(self, data, target_column, alpha=1.0, **kwargs): pass
    def _lasso_regression(self, data, target_column, alpha=1.0, **kwargs): pass
    def _elastic_net(self, data, target_column, alpha=1.0, l1_ratio=0.5, **kwargs): pass
    def _garch_model(self, data, target_column, **kwargs): pass
    def _var_model(self, data, target_column, **kwargs): pass
    async def _risk_parity_optimization(self, returns, constraints): pass
    async def _minimum_variance_optimization(self, returns, constraints): pass

# Global instance
statistical_models_service = StatisticalModelsService()