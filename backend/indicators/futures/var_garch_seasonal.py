from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced time series libraries
try:
    from arch import arch_model
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.vector_ar.var_model import VAR
    from statsmodels.stats.diagnostic import acorr_ljungbox
    ADVANCED_LIBS_AVAILABLE = True
except ImportError:
    ADVANCED_LIBS_AVAILABLE = False
    print("Advanced time series libraries not available. Using simplified implementations.")

@dataclass
class FuturesTimeSeriesData:
    """Structure for futures time series data"""
    prices: List[float]
    returns: List[float]
    timestamps: List[datetime]
    volume: List[float]
    open_interest: List[float]
    contract_symbol: str
    underlying_asset: str
    
@dataclass
class VARResult:
    """Results from Vector Autoregression analysis"""
    model_order: int
    coefficients: Dict[str, List[float]]
    residuals: List[List[float]]
    forecast: List[List[float]]
    forecast_intervals: List[List[Tuple[float, float]]]
    granger_causality: Dict[str, Dict[str, float]]
    impulse_responses: Dict[str, List[float]]
    variance_decomposition: Dict[str, Dict[str, float]]
    
@dataclass
class GARCHResult:
    """Results from GARCH family models"""
    model_type: str  # 'GARCH', 'EGARCH', 'TGARCH', 'FIGARCH'
    parameters: Dict[str, float]
    conditional_volatility: List[float]
    standardized_residuals: List[float]
    log_likelihood: float
    aic: float
    bic: float
    volatility_forecast: List[float]
    var_estimates: List[float]  # Value at Risk
    
@dataclass
class SeasonalARIMAResult:
    """Results from Seasonal ARIMA analysis"""
    model_order: Tuple[int, int, int]  # (p, d, q)
    seasonal_order: Tuple[int, int, int, int]  # (P, D, Q, s)
    parameters: Dict[str, float]
    fitted_values: List[float]
    residuals: List[float]
    forecast: List[float]
    forecast_intervals: List[Tuple[float, float]]
    seasonal_components: Dict[str, List[float]]
    diagnostics: Dict[str, float]
    
@dataclass
class FuturesTimeSeriesResult:
    """Comprehensive futures time series analysis results"""
    var_results: Optional[VARResult]
    garch_results: Dict[str, GARCHResult]
    seasonal_arima_results: SeasonalARIMAResult
    model_comparison: Dict[str, Dict[str, float]]
    trading_signals: List[str]
    risk_metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    model_performance: Dict[str, float]

class VARAnalyzer:
    """Vector Autoregression analyzer for multivariate time series"""
    
    def __init__(self, max_lags: int = 10):
        self.model_name = "Vector Autoregression"
        self.max_lags = max_lags
        
    def analyze_var(self, data_dict: Dict[str, List[float]], 
                   timestamps: List[datetime]) -> Optional[VARResult]:
        """Analyze multivariate time series using VAR model"""
        
        if not ADVANCED_LIBS_AVAILABLE:
            return self._simple_var_analysis(data_dict, timestamps)
        
        try:
            # Prepare data
            df = pd.DataFrame(data_dict, index=timestamps)
            df = df.dropna()
            
            if len(df) < 50:  # Need sufficient data
                return None
            
            # Fit VAR model
            model = VAR(df)
            
            # Select optimal lag order
            lag_order_results = model.select_order(maxlags=min(self.max_lags, len(df)//4))
            optimal_lags = lag_order_results.aic
            
            # Fit model with optimal lags
            fitted_model = model.fit(optimal_lags)
            
            # Extract results
            coefficients = {}
            for i, var_name in enumerate(df.columns):
                coefficients[var_name] = fitted_model.coefs[:, :, i].flatten().tolist()
            
            residuals = fitted_model.resid.values.tolist()
            
            # Generate forecasts
            forecast_steps = min(10, len(df)//10)
            forecast_result = fitted_model.forecast(df.values[-optimal_lags:], steps=forecast_steps)
            forecast = forecast_result.tolist()
            
            # Calculate forecast intervals (simplified)
            forecast_intervals = []
            for step_forecast in forecast:
                intervals = []
                for val in step_forecast:
                    std_err = np.std(residuals) * np.sqrt(len(step_forecast))
                    intervals.append((val - 1.96 * std_err, val + 1.96 * std_err))
                forecast_intervals.append(intervals)
            
            # Granger causality tests
            granger_causality = {}
            for caused in df.columns:
                granger_causality[caused] = {}
                for causing in df.columns:
                    if caused != causing:
                        try:
                            test_result = fitted_model.test_causality(caused, causing, kind='f')
                            granger_causality[caused][causing] = test_result.pvalue
                        except:
                            granger_causality[caused][causing] = 1.0
            
            # Impulse response functions
            irf = fitted_model.irf(periods=20)
            impulse_responses = {}
            for i, var_name in enumerate(df.columns):
                impulse_responses[var_name] = irf.irfs[:, i, i].tolist()
            
            # Variance decomposition
            fevd = fitted_model.fevd(periods=20)
            variance_decomposition = {}
            for i, var_name in enumerate(df.columns):
                variance_decomposition[var_name] = {}
                for j, other_var in enumerate(df.columns):
                    variance_decomposition[var_name][other_var] = fevd.decomp[-1, i, j]
            
            return VARResult(
                model_order=optimal_lags,
                coefficients=coefficients,
                residuals=residuals,
                forecast=forecast,
                forecast_intervals=forecast_intervals,
                granger_causality=granger_causality,
                impulse_responses=impulse_responses,
                variance_decomposition=variance_decomposition
            )
            
        except Exception as e:
            print(f"VAR analysis failed: {e}")
            return self._simple_var_analysis(data_dict, timestamps)
    
    def _simple_var_analysis(self, data_dict: Dict[str, List[float]], 
                           timestamps: List[datetime]) -> Optional[VARResult]:
        """Simplified VAR analysis when advanced libraries are not available"""
        
        if len(data_dict) < 2:
            return None
        
        # Simple multivariate analysis
        series_names = list(data_dict.keys())
        series_data = [data_dict[name] for name in series_names]
        
        # Ensure all series have the same length
        min_length = min(len(series) for series in series_data)
        series_data = [series[:min_length] for series in series_data]
        
        if min_length < 20:
            return None
        
        # Simple autoregressive coefficients
        coefficients = {}
        residuals = []
        
        for i, (name, series) in enumerate(zip(series_names, series_data)):
            # Simple AR(1) coefficients
            if len(series) > 1:
                x = np.array(series[:-1]).reshape(-1, 1)
                y = np.array(series[1:])
                
                # Simple linear regression
                coef = np.corrcoef(x.flatten(), y)[0, 1] if len(x) > 1 else 0.0
                coefficients[name] = [coef]
                
                # Calculate residuals
                predicted = [series[0]] + [series[j] * coef for j in range(len(series)-1)]
                residuals.append([series[j] - predicted[j] for j in range(len(series))])
        
        # Simple forecast
        forecast_steps = 5
        forecast = []
        for step in range(forecast_steps):
            step_forecast = []
            for i, (name, series) in enumerate(zip(series_names, series_data)):
                if coefficients[name]:
                    last_val = series[-1] if step == 0 else forecast[-1][i]
                    next_val = last_val * coefficients[name][0]
                    step_forecast.append(next_val)
                else:
                    step_forecast.append(series[-1])
            forecast.append(step_forecast)
        
        # Simple forecast intervals
        forecast_intervals = []
        for step_forecast in forecast:
            intervals = []
            for i, val in enumerate(step_forecast):
                std_err = np.std(residuals[i]) if residuals[i] else 1.0
                intervals.append((val - 1.96 * std_err, val + 1.96 * std_err))
            forecast_intervals.append(intervals)
        
        # Simple correlation-based causality
        granger_causality = {}
        for i, name1 in enumerate(series_names):
            granger_causality[name1] = {}
            for j, name2 in enumerate(series_names):
                if i != j:
                    # Use correlation as proxy for causality
                    corr = np.corrcoef(series_data[i][:-1], series_data[j][1:])[0, 1]
                    p_value = 1.0 - abs(corr)  # Simple p-value proxy
                    granger_causality[name1][name2] = p_value
        
        # Simple impulse responses
        impulse_responses = {}
        for name in series_names:
            # Exponential decay response
            coef = coefficients[name][0] if coefficients[name] else 0.5
            impulse_responses[name] = [coef ** i for i in range(20)]
        
        # Simple variance decomposition
        variance_decomposition = {}
        for name1 in series_names:
            variance_decomposition[name1] = {}
            total_var = sum(abs(coefficients[name][0]) if coefficients[name] else 0.1 
                          for name in series_names)
            for name2 in series_names:
                own_var = abs(coefficients[name2][0]) if coefficients[name2] else 0.1
                variance_decomposition[name1][name2] = own_var / total_var if total_var > 0 else 0.5
        
        return VARResult(
            model_order=1,
            coefficients=coefficients,
            residuals=residuals,
            forecast=forecast,
            forecast_intervals=forecast_intervals,
            granger_causality=granger_causality,
            impulse_responses=impulse_responses,
            variance_decomposition=variance_decomposition
        )

class GARCHAnalyzer:
    """GARCH family models analyzer for volatility modeling"""
    
    def __init__(self):
        self.model_name = "GARCH Family Models"
        
    def analyze_garch_models(self, returns: List[float]) -> Dict[str, GARCHResult]:
        """Analyze returns using various GARCH models"""
        
        if not returns or len(returns) < 50:
            return {}
        
        results = {}
        
        # Clean returns data
        returns_clean = [r for r in returns if not np.isnan(r) and not np.isinf(r)]
        if len(returns_clean) < 50:
            return {}
        
        # Convert to percentage returns
        returns_pct = [r * 100 for r in returns_clean]
        
        if ADVANCED_LIBS_AVAILABLE:
            # GARCH(1,1)
            results['GARCH'] = self._fit_garch_model(returns_pct, 'GARCH')
            
            # EGARCH(1,1)
            results['EGARCH'] = self._fit_garch_model(returns_pct, 'EGARCH')
            
            # TGARCH(1,1)
            results['TGARCH'] = self._fit_garch_model(returns_pct, 'TGARCH')
        else:
            # Simple volatility models
            results['Simple_GARCH'] = self._simple_garch_model(returns_pct)
        
        return results
    
    def _fit_garch_model(self, returns: List[float], model_type: str) -> Optional[GARCHResult]:
        """Fit GARCH family model using arch library"""
        
        try:
            returns_series = pd.Series(returns)
            
            # Define model based on type
            if model_type == 'GARCH':
                model = arch_model(returns_series, vol='Garch', p=1, q=1)
            elif model_type == 'EGARCH':
                model = arch_model(returns_series, vol='EGARCH', p=1, q=1)
            elif model_type == 'TGARCH':
                model = arch_model(returns_series, vol='GARCH', p=1, q=1, power=2.0)
            else:
                model = arch_model(returns_series, vol='Garch', p=1, q=1)
            
            # Fit model
            fitted_model = model.fit(disp='off')
            
            # Extract parameters
            parameters = dict(fitted_model.params)
            
            # Get conditional volatility
            conditional_volatility = fitted_model.conditional_volatility.tolist()
            
            # Get standardized residuals
            standardized_residuals = fitted_model.std_resid.tolist()
            
            # Model diagnostics
            log_likelihood = fitted_model.loglikelihood
            aic = fitted_model.aic
            bic = fitted_model.bic
            
            # Forecast volatility
            forecast_horizon = min(10, len(returns)//10)
            forecast_result = fitted_model.forecast(horizon=forecast_horizon)
            volatility_forecast = forecast_result.variance.iloc[-1].tolist()
            
            # Calculate VaR estimates
            var_estimates = []
            for vol in conditional_volatility:
                # 5% VaR assuming normal distribution
                var_5 = -1.645 * vol / 100  # Convert back from percentage
                var_estimates.append(var_5)
            
            return GARCHResult(
                model_type=model_type,
                parameters=parameters,
                conditional_volatility=conditional_volatility,
                standardized_residuals=standardized_residuals,
                log_likelihood=log_likelihood,
                aic=aic,
                bic=bic,
                volatility_forecast=volatility_forecast,
                var_estimates=var_estimates
            )
            
        except Exception as e:
            print(f"GARCH {model_type} model fitting failed: {e}")
            return self._simple_garch_model(returns, model_type)
    
    def _simple_garch_model(self, returns: List[float], model_type: str = 'Simple_GARCH') -> GARCHResult:
        """Simple GARCH-like volatility model"""
        
        # Simple exponentially weighted moving average for volatility
        alpha = 0.1  # Smoothing parameter
        beta = 0.85  # Persistence parameter
        
        conditional_volatility = []
        long_run_var = np.var(returns)
        
        # Initialize
        current_var = long_run_var
        
        for i, ret in enumerate(returns):
            if i == 0:
                conditional_volatility.append(np.sqrt(current_var))
            else:
                # GARCH(1,1) equation: σ²(t) = ω + α*ε²(t-1) + β*σ²(t-1)
                omega = long_run_var * (1 - alpha - beta)
                current_var = omega + alpha * (returns[i-1] ** 2) + beta * current_var
                conditional_volatility.append(np.sqrt(current_var))
        
        # Standardized residuals
        standardized_residuals = []
        for i, ret in enumerate(returns):
            if conditional_volatility[i] > 0:
                std_resid = ret / conditional_volatility[i]
                standardized_residuals.append(std_resid)
            else:
                standardized_residuals.append(0.0)
        
        # Simple parameters
        parameters = {
            'omega': long_run_var * (1 - alpha - beta),
            'alpha': alpha,
            'beta': beta
        }
        
        # Simple diagnostics
        residuals_var = np.var(standardized_residuals)
        log_likelihood = -0.5 * len(returns) * (np.log(2 * np.pi) + np.log(residuals_var) + 1)
        aic = -2 * log_likelihood + 2 * len(parameters)
        bic = -2 * log_likelihood + len(parameters) * np.log(len(returns))
        
        # Simple forecast
        last_return = returns[-1]
        last_var = conditional_volatility[-1] ** 2
        
        volatility_forecast = []
        for h in range(1, 11):
            # Multi-step forecast
            forecast_var = parameters['omega'] / (1 - parameters['beta']) + \
                          (parameters['alpha'] * last_return**2 + parameters['beta'] * last_var - 
                           parameters['omega'] / (1 - parameters['beta'])) * (parameters['beta'] ** (h-1))
            volatility_forecast.append(forecast_var)
        
        # VaR estimates
        var_estimates = [-1.645 * vol / 100 for vol in conditional_volatility]
        
        return GARCHResult(
            model_type=model_type,
            parameters=parameters,
            conditional_volatility=conditional_volatility,
            standardized_residuals=standardized_residuals,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            volatility_forecast=volatility_forecast,
            var_estimates=var_estimates
        )

class SeasonalARIMAAnalyzer:
    """Seasonal ARIMA analyzer for time series with seasonal patterns"""
    
    def __init__(self):
        self.model_name = "Seasonal ARIMA"
        
    def analyze_seasonal_arima(self, prices: List[float], 
                             timestamps: List[datetime]) -> SeasonalARIMAResult:
        """Analyze time series using Seasonal ARIMA"""
        
        if not prices or len(prices) < 50:
            return self._create_empty_result()
        
        if ADVANCED_LIBS_AVAILABLE:
            return self._fit_seasonal_arima(prices, timestamps)
        else:
            return self._simple_seasonal_analysis(prices, timestamps)
    
    def _fit_seasonal_arima(self, prices: List[float], 
                          timestamps: List[datetime]) -> SeasonalARIMAResult:
        """Fit Seasonal ARIMA model using statsmodels"""
        
        try:
            # Create time series
            ts = pd.Series(prices, index=timestamps)
            ts = ts.dropna()
            
            if len(ts) < 50:
                return self._simple_seasonal_analysis(prices, timestamps)
            
            # Determine seasonality
            seasonal_period = self._detect_seasonality(ts)
            
            # Test for stationarity
            d = self._determine_differencing(ts)
            D = self._determine_seasonal_differencing(ts, seasonal_period) if seasonal_period > 1 else 0
            
            # Auto-select ARIMA parameters
            best_aic = float('inf')
            best_model = None
            best_order = None
            best_seasonal_order = None
            
            # Grid search for optimal parameters
            for p in range(3):
                for q in range(3):
                    for P in range(2):
                        for Q in range(2):
                            try:
                                order = (p, d, q)
                                seasonal_order = (P, D, Q, seasonal_period)
                                
                                model = ARIMA(ts, order=order, seasonal_order=seasonal_order)
                                fitted_model = model.fit()
                                
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_model = fitted_model
                                    best_order = order
                                    best_seasonal_order = seasonal_order
                                    
                            except:
                                continue
            
            if best_model is None:
                return self._simple_seasonal_analysis(prices, timestamps)
            
            # Extract results
            parameters = dict(best_model.params)
            fitted_values = best_model.fittedvalues.tolist()
            residuals = best_model.resid.tolist()
            
            # Generate forecasts
            forecast_steps = min(20, len(ts)//5)
            forecast_result = best_model.forecast(steps=forecast_steps)
            forecast = forecast_result.tolist()
            
            # Forecast intervals
            forecast_ci = best_model.get_forecast(steps=forecast_steps).conf_int()
            forecast_intervals = [(row[0], row[1]) for _, row in forecast_ci.iterrows()]
            
            # Seasonal decomposition
            seasonal_components = self._decompose_seasonal(ts, seasonal_period)
            
            # Diagnostics
            diagnostics = {
                'aic': best_model.aic,
                'bic': best_model.bic,
                'log_likelihood': best_model.llf,
                'ljung_box_pvalue': self._ljung_box_test(residuals),
                'jarque_bera_pvalue': self._jarque_bera_test(residuals)
            }
            
            return SeasonalARIMAResult(
                model_order=best_order,
                seasonal_order=best_seasonal_order,
                parameters=parameters,
                fitted_values=fitted_values,
                residuals=residuals,
                forecast=forecast,
                forecast_intervals=forecast_intervals,
                seasonal_components=seasonal_components,
                diagnostics=diagnostics
            )
            
        except Exception as e:
            print(f"Seasonal ARIMA fitting failed: {e}")
            return self._simple_seasonal_analysis(prices, timestamps)
    
    def _simple_seasonal_analysis(self, prices: List[float], 
                                timestamps: List[datetime]) -> SeasonalARIMAResult:
        """Simple seasonal analysis when advanced libraries are not available"""
        
        # Simple moving average and seasonal decomposition
        n = len(prices)
        if n < 20:
            return self._create_empty_result()
        
        # Detect simple seasonality (monthly pattern)
        seasonal_period = 12 if len(timestamps) > 24 else 4
        
        # Simple trend estimation
        window = min(seasonal_period, n//4)
        trend = []
        for i in range(n):
            start_idx = max(0, i - window//2)
            end_idx = min(n, i + window//2 + 1)
            trend.append(np.mean(prices[start_idx:end_idx]))
        
        # Simple seasonal component
        seasonal = []
        detrended = [prices[i] - trend[i] for i in range(n)]
        
        for i in range(n):
            # Average of same seasonal position
            same_season_values = []
            for j in range(i % seasonal_period, n, seasonal_period):
                if j < len(detrended):
                    same_season_values.append(detrended[j])
            seasonal.append(np.mean(same_season_values) if same_season_values else 0)
        
        # Residuals
        residuals = [prices[i] - trend[i] - seasonal[i] for i in range(n)]
        
        # Simple AR(1) model for residuals
        if len(residuals) > 1:
            ar_coef = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        else:
            ar_coef = 0.0
        
        # Fitted values
        fitted_values = [trend[i] + seasonal[i] for i in range(n)]
        
        # Simple forecast
        forecast_steps = min(12, n//4)
        forecast = []
        
        last_trend = trend[-1]
        for h in range(1, forecast_steps + 1):
            # Simple trend extrapolation
            if len(trend) > 1:
                trend_change = trend[-1] - trend[-2]
                forecast_trend = last_trend + h * trend_change
            else:
                forecast_trend = last_trend
            
            # Seasonal component
            seasonal_idx = (n + h - 1) % seasonal_period
            if seasonal_idx < len(seasonal):
                forecast_seasonal = seasonal[seasonal_idx]
            else:
                forecast_seasonal = 0
            
            # AR component
            if h == 1 and residuals:
                ar_component = ar_coef * residuals[-1]
            else:
                ar_component = 0
            
            forecast_value = forecast_trend + forecast_seasonal + ar_component
            forecast.append(forecast_value)
        
        # Simple forecast intervals
        residual_std = np.std(residuals) if residuals else 1.0
        forecast_intervals = []
        for i, f_val in enumerate(forecast):
            margin = 1.96 * residual_std * np.sqrt(i + 1)
            forecast_intervals.append((f_val - margin, f_val + margin))
        
        # Seasonal components
        seasonal_components = {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residuals
        }
        
        # Simple diagnostics
        mse = np.mean([r**2 for r in residuals])
        n_params = 3  # trend, seasonal, ar
        aic = n * np.log(mse) + 2 * n_params
        bic = n * np.log(mse) + n_params * np.log(n)
        
        diagnostics = {
            'aic': aic,
            'bic': bic,
            'log_likelihood': -0.5 * n * (np.log(2 * np.pi * mse) + 1),
            'ljung_box_pvalue': 0.5,  # Placeholder
            'jarque_bera_pvalue': 0.5  # Placeholder
        }
        
        return SeasonalARIMAResult(
            model_order=(1, 0, 0),
            seasonal_order=(1, 0, 0, seasonal_period),
            parameters={'ar_coef': ar_coef, 'seasonal_period': seasonal_period},
            fitted_values=fitted_values,
            residuals=residuals,
            forecast=forecast,
            forecast_intervals=forecast_intervals,
            seasonal_components=seasonal_components,
            diagnostics=diagnostics
        )
    
    def _detect_seasonality(self, ts: pd.Series) -> int:
        """Detect seasonal period in time series"""
        try:
            # Try different seasonal periods
            periods_to_test = [4, 12, 52]  # Quarterly, monthly, weekly
            best_period = 1
            best_score = 0
            
            for period in periods_to_test:
                if len(ts) >= 2 * period:
                    # Calculate autocorrelation at seasonal lag
                    if len(ts) > period:
                        seasonal_corr = ts.autocorr(lag=period)
                        if not np.isnan(seasonal_corr) and abs(seasonal_corr) > best_score:
                            best_score = abs(seasonal_corr)
                            best_period = period
            
            return best_period if best_score > 0.1 else 1
            
        except:
            return 12  # Default to monthly
    
    def _determine_differencing(self, ts: pd.Series) -> int:
        """Determine number of differences needed for stationarity"""
        try:
            # ADF test
            adf_result = adfuller(ts.dropna())
            if adf_result[1] <= 0.05:  # p-value <= 0.05
                return 0  # Already stationary
            
            # Try first difference
            ts_diff = ts.diff().dropna()
            if len(ts_diff) > 10:
                adf_result = adfuller(ts_diff)
                if adf_result[1] <= 0.05:
                    return 1
            
            return 1  # Default to first difference
            
        except:
            return 1
    
    def _determine_seasonal_differencing(self, ts: pd.Series, seasonal_period: int) -> int:
        """Determine seasonal differencing needed"""
        try:
            if seasonal_period <= 1 or len(ts) < 2 * seasonal_period:
                return 0
            
            # Test seasonal stationarity
            seasonal_ts = ts[::seasonal_period]
            if len(seasonal_ts) > 10:
                adf_result = adfuller(seasonal_ts.dropna())
                if adf_result[1] > 0.05:  # Not stationary
                    return 1
            
            return 0
            
        except:
            return 0
    
    def _decompose_seasonal(self, ts: pd.Series, seasonal_period: int) -> Dict[str, List[float]]:
        """Decompose time series into seasonal components"""
        try:
            if seasonal_period > 1 and len(ts) >= 2 * seasonal_period:
                decomposition = seasonal_decompose(ts, model='additive', period=seasonal_period)
                return {
                    'trend': decomposition.trend.fillna(0).tolist(),
                    'seasonal': decomposition.seasonal.fillna(0).tolist(),
                    'residual': decomposition.resid.fillna(0).tolist()
                }
        except:
            pass
        
        # Simple decomposition
        n = len(ts)
        trend = ts.rolling(window=min(seasonal_period, n//4), center=True).mean().fillna(method='bfill').fillna(method='ffill').tolist()
        
        seasonal = [0] * n
        residual = [ts.iloc[i] - trend[i] for i in range(n)]
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def _ljung_box_test(self, residuals: List[float]) -> float:
        """Ljung-Box test for residual autocorrelation"""
        try:
            if ADVANCED_LIBS_AVAILABLE and len(residuals) > 10:
                result = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
                return result['lb_pvalue'].iloc[-1]
        except:
            pass
        return 0.5  # Placeholder
    
    def _jarque_bera_test(self, residuals: List[float]) -> float:
        """Jarque-Bera test for normality"""
        try:
            if len(residuals) > 10:
                statistic, p_value = stats.jarque_bera(residuals)
                return p_value
        except:
            pass
        return 0.5  # Placeholder
    
    def _create_empty_result(self) -> SeasonalARIMAResult:
        """Create empty result when analysis fails"""
        return SeasonalARIMAResult(
            model_order=(0, 0, 0),
            seasonal_order=(0, 0, 0, 1),
            parameters={},
            fitted_values=[],
            residuals=[],
            forecast=[],
            forecast_intervals=[],
            seasonal_components={},
            diagnostics={}
        )

class FuturesTimeSeriesAnalyzer:
    """Comprehensive futures time series analyzer"""
    
    def __init__(self):
        self.var_analyzer = VARAnalyzer()
        self.garch_analyzer = GARCHAnalyzer()
        self.seasonal_arima_analyzer = SeasonalARIMAAnalyzer()
        
    def analyze(self, futures_data: FuturesTimeSeriesData, 
               additional_series: Optional[Dict[str, List[float]]] = None) -> FuturesTimeSeriesResult:
        """Perform comprehensive time series analysis"""
        
        print(f"Analyzing time series for {futures_data.contract_symbol}...")
        
        # VAR analysis (if additional series provided)
        var_results = None
        if additional_series:
            data_dict = {'futures_prices': futures_data.prices}
            data_dict.update(additional_series)
            var_results = self.var_analyzer.analyze_var(data_dict, futures_data.timestamps)
        
        # GARCH analysis
        garch_results = self.garch_analyzer.analyze_garch_models(futures_data.returns)
        
        # Seasonal ARIMA analysis
        seasonal_arima_results = self.seasonal_arima_analyzer.analyze_seasonal_arima(
            futures_data.prices, futures_data.timestamps
        )
        
        # Model comparison
        model_comparison = self._compare_models(garch_results, seasonal_arima_results)
        
        # Generate trading signals
        trading_signals = self._generate_trading_signals(
            var_results, garch_results, seasonal_arima_results
        )
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            futures_data, var_results, garch_results, seasonal_arima_results
        )
        
        # Generate insights
        insights = self._generate_insights(
            var_results, garch_results, seasonal_arima_results, risk_metrics
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            var_results, garch_results, seasonal_arima_results, risk_metrics
        )
        
        # Calculate model performance
        model_performance = self._calculate_model_performance(
            futures_data, garch_results, seasonal_arima_results
        )
        
        return FuturesTimeSeriesResult(
            var_results=var_results,
            garch_results=garch_results,
            seasonal_arima_results=seasonal_arima_results,
            model_comparison=model_comparison,
            trading_signals=trading_signals,
            risk_metrics=risk_metrics,
            insights=insights,
            recommendations=recommendations,
            model_performance=model_performance
        )
    
    def _compare_models(self, garch_results: Dict[str, GARCHResult],
                      seasonal_arima_results: SeasonalARIMAResult) -> Dict[str, Dict[str, float]]:
        """Compare different models"""
        comparison = {}
        
        # GARCH models comparison
        if garch_results:
            garch_comparison = {}
            for model_name, result in garch_results.items():
                garch_comparison[model_name] = {
                    'aic': result.aic,
                    'bic': result.bic,
                    'log_likelihood': result.log_likelihood
                }
            comparison['garch_models'] = garch_comparison
        
        # ARIMA model
        if seasonal_arima_results.diagnostics:
            comparison['seasonal_arima'] = {
                'aic': seasonal_arima_results.diagnostics.get('aic', 0),
                'bic': seasonal_arima_results.diagnostics.get('bic', 0),
                'log_likelihood': seasonal_arima_results.diagnostics.get('log_likelihood', 0)
            }
        
        return comparison
    
    def _generate_trading_signals(self, var_results: Optional[VARResult],
                                garch_results: Dict[str, GARCHResult],
                                seasonal_arima_results: SeasonalARIMAResult) -> List[str]:
        """Generate trading signals based on time series analysis"""
        signals = []
        
        # Determine signal length
        signal_length = 0
        if seasonal_arima_results.fitted_values:
            signal_length = len(seasonal_arima_results.fitted_values)
        elif garch_results:
            first_garch = next(iter(garch_results.values()))
            signal_length = len(first_garch.conditional_volatility)
        
        if signal_length == 0:
            return signals
        
        # Generate signals for each time period
        for i in range(signal_length):
            signal = "HOLD"
            
            # ARIMA-based signals
            if (seasonal_arima_results.fitted_values and 
                seasonal_arima_results.residuals and 
                i < len(seasonal_arima_results.fitted_values) and 
                i < len(seasonal_arima_results.residuals)):
                
                residual = seasonal_arima_results.residuals[i]
                residual_std = np.std(seasonal_arima_results.residuals)
                
                if residual > 2 * residual_std:
                    signal = "SELL"  # Price above model prediction
                elif residual < -2 * residual_std:
                    signal = "BUY"   # Price below model prediction
            
            # GARCH-based volatility signals
            if garch_results:
                best_garch = min(garch_results.values(), key=lambda x: x.aic)
                if i < len(best_garch.conditional_volatility):
                    current_vol = best_garch.conditional_volatility[i]
                    avg_vol = np.mean(best_garch.conditional_volatility)
                    
                    if current_vol > 1.5 * avg_vol:
                        signal = "REDUCE_POSITION"  # High volatility
                    elif current_vol < 0.7 * avg_vol:
                        if signal == "HOLD":
                            signal = "INCREASE_POSITION"  # Low volatility
            
            # VAR-based signals
            if var_results and var_results.forecast:
                # Use VAR forecast for directional signals
                if len(var_results.forecast) > 0 and len(var_results.forecast[0]) > 0:
                    next_price_forecast = var_results.forecast[0][0]  # First variable, first forecast
                    if i > 0:  # Need previous price for comparison
                        # This is a simplified approach
                        if signal == "HOLD":
                            signal = "BUY" if next_price_forecast > 0 else "SELL"
            
            signals.append(signal)
        
        return signals
    
    def _calculate_risk_metrics(self, futures_data: FuturesTimeSeriesData,
                              var_results: Optional[VARResult],
                              garch_results: Dict[str, GARCHResult],
                              seasonal_arima_results: SeasonalARIMAResult) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        metrics = {}
        
        # Basic return metrics
        if futures_data.returns:
            returns = [r for r in futures_data.returns if not np.isnan(r)]
            if returns:
                metrics['return_volatility'] = np.std(returns)
                metrics['return_skewness'] = stats.skew(returns)
                metrics['return_kurtosis'] = stats.kurtosis(returns)
                metrics['max_drawdown'] = self._calculate_max_drawdown(futures_data.prices)
        
        # GARCH-based risk metrics
        if garch_results:
            best_garch = min(garch_results.values(), key=lambda x: x.aic)
            
            if best_garch.conditional_volatility:
                metrics['garch_avg_volatility'] = np.mean(best_garch.conditional_volatility)
                metrics['garch_vol_volatility'] = np.std(best_garch.conditional_volatility)
                
            if best_garch.var_estimates:
                metrics['average_var_5pct'] = np.mean(best_garch.var_estimates)
                metrics['max_var_5pct'] = min(best_garch.var_estimates)  # Most negative
        
        # ARIMA-based metrics
        if seasonal_arima_results.residuals:
            residuals = seasonal_arima_results.residuals
            metrics['arima_residual_volatility'] = np.std(residuals)
            metrics['arima_forecast_accuracy'] = 1.0 / (1.0 + np.mean([abs(r) for r in residuals]))
        
        # VAR-based metrics
        if var_results and var_results.residuals:
            # Multivariate risk metrics
            residual_matrix = np.array(var_results.residuals)
            if residual_matrix.size > 0:
                metrics['var_system_volatility'] = np.mean(np.std(residual_matrix, axis=0))
        
        # Model uncertainty
        model_count = len(garch_results) + (1 if seasonal_arima_results.fitted_values else 0)
        metrics['model_uncertainty'] = 1.0 / max(model_count, 1)
        
        return metrics
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0.0
        
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            else:
                drawdown = (peak - price) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _generate_insights(self, var_results: Optional[VARResult],
                         garch_results: Dict[str, GARCHResult],
                         seasonal_arima_results: SeasonalARIMAResult,
                         risk_metrics: Dict[str, float]) -> List[str]:
        """Generate analytical insights"""
        insights = []
        
        # GARCH insights
        if garch_results:
            best_garch_name = min(garch_results.keys(), key=lambda k: garch_results[k].aic)
            insights.append(f"Best volatility model: {best_garch_name} (AIC: {garch_results[best_garch_name].aic:.2f})")
            
            best_garch = garch_results[best_garch_name]
            if best_garch.conditional_volatility:
                avg_vol = np.mean(best_garch.conditional_volatility)
                current_vol = best_garch.conditional_volatility[-1]
                
                if current_vol > 1.2 * avg_vol:
                    insights.append(f"Current volatility ({current_vol:.2f}%) is {current_vol/avg_vol:.1f}x above average")
                elif current_vol < 0.8 * avg_vol:
                    insights.append(f"Current volatility ({current_vol:.2f}%) is below average - low risk period")
        
        # Seasonal ARIMA insights
        if seasonal_arima_results.seasonal_components:
            seasonal_comp = seasonal_arima_results.seasonal_components.get('seasonal', [])
            if seasonal_comp:
                max_seasonal = max(seasonal_comp)
                min_seasonal = min(seasonal_comp)
                if abs(max_seasonal - min_seasonal) > 0.1:
                    insights.append(f"Strong seasonal pattern detected (range: {min_seasonal:.2f} to {max_seasonal:.2f})")
        
        # VAR insights
        if var_results and var_results.granger_causality:
            significant_causalities = []
            for caused, causers in var_results.granger_causality.items():
                for causer, p_value in causers.items():
                    if p_value < 0.05:
                        significant_causalities.append(f"{causer} → {caused}")
            
            if significant_causalities:
                insights.append(f"Significant Granger causalities: {', '.join(significant_causalities[:3])}")
        
        # Risk insights
        return_vol = risk_metrics.get('return_volatility', 0)
        if return_vol > 0.03:
            insights.append(f"High return volatility ({return_vol:.1%}) indicates elevated risk")
        
        max_dd = risk_metrics.get('max_drawdown', 0)
        if max_dd > 0.2:
            insights.append(f"Significant maximum drawdown ({max_dd:.1%}) observed")
        
        skewness = risk_metrics.get('return_skewness', 0)
        if abs(skewness) > 0.5:
            direction = "negative" if skewness < 0 else "positive"
            insights.append(f"Returns show {direction} skewness ({skewness:.2f}) - asymmetric risk")
        
        kurtosis = risk_metrics.get('return_kurtosis', 0)
        if kurtosis > 3:
            insights.append(f"High kurtosis ({kurtosis:.2f}) indicates fat-tailed return distribution")
        
        return insights
    
    def _generate_recommendations(self, var_results: Optional[VARResult],
                                garch_results: Dict[str, GARCHResult],
                                seasonal_arima_results: SeasonalARIMAResult,
                                risk_metrics: Dict[str, float]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Volatility-based recommendations
        garch_vol = risk_metrics.get('garch_avg_volatility', 0)
        if garch_vol > 25:  # High volatility
            recommendations.append("High volatility environment - reduce position sizes and use wider stops")
            recommendations.append("Consider volatility trading strategies (straddles, strangles)")
        elif garch_vol < 15:  # Low volatility
            recommendations.append("Low volatility environment - opportunity for trend-following strategies")
        
        # Seasonal recommendations
        if seasonal_arima_results.seasonal_components:
            seasonal_comp = seasonal_arima_results.seasonal_components.get('seasonal', [])
            if seasonal_comp and len(seasonal_comp) > 12:
                recommendations.append("Incorporate seasonal patterns into trading calendar")
                recommendations.append("Consider seasonal spread strategies")
        
        # VAR-based recommendations
        if var_results and var_results.granger_causality:
            recommendations.append("Monitor leading indicators identified by VAR analysis")
            recommendations.append("Consider cross-asset hedging based on causality relationships")
        
        # Risk management recommendations
        max_dd = risk_metrics.get('max_drawdown', 0)
        if max_dd > 0.15:
            recommendations.append(f"Historical max drawdown of {max_dd:.1%} - implement strict risk controls")
        
        var_5pct = risk_metrics.get('average_var_5pct', 0)
        if var_5pct < -0.03:
            recommendations.append(f"Average 5% VaR of {var_5pct:.1%} - size positions accordingly")
        
        # Model-based recommendations
        if garch_results:
            recommendations.append("Use GARCH volatility forecasts for dynamic position sizing")
        
        if seasonal_arima_results.forecast:
            recommendations.append("Incorporate ARIMA price forecasts into entry/exit timing")
        
        # General recommendations
        recommendations.append("Monitor model residuals for regime changes")
        recommendations.append("Regularly re-estimate models as new data becomes available")
        recommendations.append("Consider ensemble forecasting combining multiple model predictions")
        
        return recommendations
    
    def _calculate_model_performance(self, futures_data: FuturesTimeSeriesData,
                                   garch_results: Dict[str, GARCHResult],
                                   seasonal_arima_results: SeasonalARIMAResult) -> Dict[str, float]:
        """Calculate model performance metrics"""
        performance = {}
        
        # GARCH model performance
        if garch_results:
            best_garch = min(garch_results.values(), key=lambda x: x.aic)
            performance['best_garch_aic'] = best_garch.aic
            performance['best_garch_bic'] = best_garch.bic
            
            # Volatility prediction accuracy
            if best_garch.conditional_volatility and futures_data.returns:
                realized_vol = [abs(r) for r in futures_data.returns if not np.isnan(r)]
                predicted_vol = best_garch.conditional_volatility[:len(realized_vol)]
                
                if len(realized_vol) == len(predicted_vol) and len(realized_vol) > 0:
                    vol_mse = mean_squared_error(realized_vol, predicted_vol)
                    vol_mae = mean_absolute_error(realized_vol, predicted_vol)
                    performance['garch_volatility_mse'] = vol_mse
                    performance['garch_volatility_mae'] = vol_mae
        
        # ARIMA model performance
        if seasonal_arima_results.fitted_values and seasonal_arima_results.diagnostics:
            performance['arima_aic'] = seasonal_arima_results.diagnostics.get('aic', 0)
            performance['arima_bic'] = seasonal_arima_results.diagnostics.get('bic', 0)
            
            # Price prediction accuracy
            if len(seasonal_arima_results.fitted_values) == len(futures_data.prices):
                price_mse = mean_squared_error(futures_data.prices, seasonal_arima_results.fitted_values)
                price_mae = mean_absolute_error(futures_data.prices, seasonal_arima_results.fitted_values)
                performance['arima_price_mse'] = price_mse
                performance['arima_price_mae'] = price_mae
                
                # R-squared
                ss_res = sum((futures_data.prices[i] - seasonal_arima_results.fitted_values[i])**2 
                           for i in range(len(futures_data.prices)))
                ss_tot = sum((p - np.mean(futures_data.prices))**2 for p in futures_data.prices)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                performance['arima_r_squared'] = r_squared
        
        # Overall model confidence
        confidence_scores = []
        if 'arima_r_squared' in performance:
            confidence_scores.append(max(0, performance['arima_r_squared']))
        if 'garch_volatility_mae' in performance:
            # Convert MAE to confidence score (lower MAE = higher confidence)
            mae_score = max(0, 1 - performance['garch_volatility_mae'])
            confidence_scores.append(mae_score)
        
        performance['overall_confidence'] = np.mean(confidence_scores) if confidence_scores else 0.5
        
        return performance
    
    def plot_results(self, futures_data: FuturesTimeSeriesData, 
                    results: FuturesTimeSeriesResult):
        """Plot comprehensive time series analysis results"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        timestamps = futures_data.timestamps
        
        # Plot 1: Price and ARIMA fit
        ax1 = axes[0, 0]
        ax1.plot(timestamps, futures_data.prices, label='Actual Prices', linewidth=2)
        
        if results.seasonal_arima_results.fitted_values:
            ax1.plot(timestamps[:len(results.seasonal_arima_results.fitted_values)], 
                    results.seasonal_arima_results.fitted_values, 
                    label='ARIMA Fit', linestyle='--', alpha=0.8)
        
        ax1.set_title('Price Series and ARIMA Fit', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: GARCH Conditional Volatility
        ax2 = axes[0, 1]
        if results.garch_results:
            best_garch_name = min(results.garch_results.keys(), 
                                key=lambda k: results.garch_results[k].aic)
            best_garch = results.garch_results[best_garch_name]
            
            if best_garch.conditional_volatility:
                vol_timestamps = timestamps[:len(best_garch.conditional_volatility)]
                ax2.plot(vol_timestamps, best_garch.conditional_volatility, 
                        color='red', linewidth=2, label=f'{best_garch_name} Volatility')
                
                # Add realized volatility if available
                if futures_data.returns:
                    realized_vol = [abs(r) * 100 for r in futures_data.returns if not np.isnan(r)]
                    real_vol_timestamps = timestamps[:len(realized_vol)]
                    ax2.plot(real_vol_timestamps, realized_vol, 
                            color='blue', alpha=0.6, label='Realized Volatility')
        
        ax2.set_title('Conditional Volatility', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Volatility (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Seasonal Decomposition
        ax3 = axes[1, 0]
        if results.seasonal_arima_results.seasonal_components:
            seasonal_comp = results.seasonal_arima_results.seasonal_components
            
            if 'trend' in seasonal_comp and seasonal_comp['trend']:
                trend_timestamps = timestamps[:len(seasonal_comp['trend'])]
                ax3.plot(trend_timestamps, seasonal_comp['trend'], 
                        label='Trend', linewidth=2)
            
            if 'seasonal' in seasonal_comp and seasonal_comp['seasonal']:
                seasonal_timestamps = timestamps[:len(seasonal_comp['seasonal'])]
                ax3.plot(seasonal_timestamps, seasonal_comp['seasonal'], 
                        label='Seasonal', linewidth=2)
        
        ax3.set_title('Seasonal Decomposition', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Component Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Residuals Analysis
        ax4 = axes[1, 1]
        if results.seasonal_arima_results.residuals:
            residuals = results.seasonal_arima_results.residuals
            residual_timestamps = timestamps[:len(residuals)]
            
            ax4.scatter(residual_timestamps, residuals, alpha=0.6, s=20)
            ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            
            # Add confidence bands
            residual_std = np.std(residuals)
            ax4.axhline(y=2*residual_std, color='orange', linestyle=':', alpha=0.7, label='±2σ')
            ax4.axhline(y=-2*residual_std, color='orange', linestyle=':', alpha=0.7)
        
        ax4.set_title('ARIMA Residuals', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Residuals')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Trading Signals
        ax5 = axes[2, 0]
        if results.trading_signals:
            signal_mapping = {'BUY': 1, 'SELL': -1, 'HOLD': 0, 'REDUCE_POSITION': -0.5, 'INCREASE_POSITION': 0.5}
            signal_values = [signal_mapping.get(signal, 0) for signal in results.trading_signals]
            
            colors = ['green' if s > 0 else 'red' if s < 0 else 'gray' for s in signal_values]
            signal_timestamps = timestamps[:len(signal_values)]
            
            ax5.bar(range(len(signal_values)), signal_values, color=colors, alpha=0.7)
            ax5.set_title('Trading Signals', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Time Period')
            ax5.set_ylabel('Signal Strength')
            ax5.set_ylim(-1.2, 1.2)
            ax5.grid(True, alpha=0.3)
        
        # Plot 6: Model Comparison
        ax6 = axes[2, 1]
        if results.model_comparison:
            model_names = []
            aic_values = []
            
            # GARCH models
            if 'garch_models' in results.model_comparison:
                for model_name, metrics in results.model_comparison['garch_models'].items():
                    model_names.append(model_name)
                    aic_values.append(metrics.get('aic', 0))
            
            # ARIMA model
            if 'seasonal_arima' in results.model_comparison:
                model_names.append('SARIMA')
                aic_values.append(results.model_comparison['seasonal_arima'].get('aic', 0))
            
            if model_names and aic_values:
                bars = ax6.bar(model_names, aic_values, color='skyblue', alpha=0.7)
                ax6.set_title('Model Comparison (AIC)', fontsize=14, fontweight='bold')
                ax6.set_xlabel('Model')
                ax6.set_ylabel('AIC Value')
                ax6.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, aic_values):
                    height = bar.get_height()
                    ax6.text(bar.get_x() + bar.get_width()/2., height,
                            f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.suptitle(f'Futures Time Series Analysis: {futures_data.contract_symbol}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.show()
    
    def generate_report(self, futures_data: FuturesTimeSeriesData, 
                       results: FuturesTimeSeriesResult) -> str:
        """Generate comprehensive analysis report"""
        
        report = f"""
# FUTURES TIME SERIES ANALYSIS REPORT
## Contract: {futures_data.contract_symbol} ({futures_data.underlying_asset})
## Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### EXECUTIVE SUMMARY
"""
        
        # Model Performance Summary
        if results.model_performance:
            best_model = "Unknown"
            if 'arima_r_squared' in results.model_performance:
                r_sq = results.model_performance['arima_r_squared']
                report += f"- ARIMA Model R²: {r_sq:.3f}\n"
                if r_sq > 0.7:
                    best_model = "SARIMA (Strong Fit)"
                elif r_sq > 0.5:
                    best_model = "SARIMA (Moderate Fit)"
            
            if 'best_garch_aic' in results.model_performance:
                report += f"- Best GARCH Model AIC: {results.model_performance['best_garch_aic']:.2f}\n"
            
            confidence = results.model_performance.get('overall_confidence', 0.5)
            report += f"- Overall Model Confidence: {confidence:.1%}\n"
        
        # Risk Summary
        if results.risk_metrics:
            report += "\n### RISK METRICS\n"
            
            vol = results.risk_metrics.get('return_volatility', 0)
            report += f"- Return Volatility: {vol:.1%}\n"
            
            max_dd = results.risk_metrics.get('max_drawdown', 0)
            report += f"- Maximum Drawdown: {max_dd:.1%}\n"
            
            var_5 = results.risk_metrics.get('average_var_5pct', 0)
            if var_5 != 0:
                report += f"- Average 5% VaR: {var_5:.1%}\n"
            
            skew = results.risk_metrics.get('return_skewness', 0)
            kurt = results.risk_metrics.get('return_kurtosis', 0)
            report += f"- Return Skewness: {skew:.2f}\n"
            report += f"- Return Kurtosis: {kurt:.2f}\n"
        
        # Model Results
        report += "\n### MODEL ANALYSIS\n"
        
        # GARCH Results
        if results.garch_results:
            report += "\n#### GARCH Volatility Models\n"
            for model_name, garch_result in results.garch_results.items():
                report += f"\n**{model_name} Model:**\n"
                report += f"- AIC: {garch_result.aic:.2f}\n"
                report += f"- BIC: {garch_result.bic:.2f}\n"
                
                if garch_result.conditional_volatility:
                    avg_vol = np.mean(garch_result.conditional_volatility)
                    current_vol = garch_result.conditional_volatility[-1]
                    report += f"- Average Volatility: {avg_vol:.2f}%\n"
                    report += f"- Current Volatility: {current_vol:.2f}%\n"
        
        # ARIMA Results
        if results.seasonal_arima_results.model_order:
            report += "\n#### Seasonal ARIMA Model\n"
            p, d, q = results.seasonal_arima_results.model_order
            P, D, Q, s = results.seasonal_arima_results.seasonal_order
            report += f"- Model Order: ARIMA({p},{d},{q}) × ({P},{D},{Q})[{s}]\n"
            
            if results.seasonal_arima_results.diagnostics:
                diag = results.seasonal_arima_results.diagnostics
                report += f"- AIC: {diag.get('aic', 0):.2f}\n"
                report += f"- BIC: {diag.get('bic', 0):.2f}\n"
                report += f"- Ljung-Box p-value: {diag.get('ljung_box_pvalue', 0):.3f}\n"
        
        # VAR Results
        if results.var_results:
            report += "\n#### Vector Autoregression (VAR)\n"
            report += f"- Model Order: VAR({results.var_results.model_order})\n"
            
            if results.var_results.granger_causality:
                report += "- Significant Granger Causalities:\n"
                for caused, causers in results.var_results.granger_causality.items():
                    for causer, p_value in causers.items():
                        if p_value < 0.05:
                            report += f"  * {causer} → {caused} (p={p_value:.3f})\n"
        
        # Key Insights
        if results.insights:
            report += "\n### KEY INSIGHTS\n"
            for i, insight in enumerate(results.insights, 1):
                report += f"{i}. {insight}\n"
        
        # Recommendations
        if results.recommendations:
            report += "\n### RECOMMENDATIONS\n"
            for i, recommendation in enumerate(results.recommendations, 1):
                report += f"{i}. {recommendation}\n"
        
        # Trading Signals Summary
        if results.trading_signals:
            signal_counts = {}
            for signal in results.trading_signals:
                signal_counts[signal] = signal_counts.get(signal, 0) + 1
            
            report += "\n### TRADING SIGNALS SUMMARY\n"
            total_signals = len(results.trading_signals)
            for signal, count in signal_counts.items():
                percentage = (count / total_signals) * 100
                report += f"- {signal}: {count} ({percentage:.1f}%)\n"
        
        report += "\n### METHODOLOGY\n"
        report += "This analysis employs multiple time series models:\n"
        report += "- **GARCH Family**: Models volatility clustering and heteroskedasticity\n"
        report += "- **Seasonal ARIMA**: Captures trend and seasonal patterns in prices\n"
        report += "- **VAR**: Analyzes multivariate relationships and causality\n"
        report += "\nAll models are estimated using maximum likelihood and compared using information criteria.\n"
        
        report += "\n---\n"
        report += "*Report generated by FinScope Futures Time Series Analyzer*\n"
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Create sample futures data
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    
    # Generate sample timestamps
    start_date = datetime.now() - timedelta(days=n_periods)
    timestamps = [start_date + timedelta(days=i) for i in range(n_periods)]
    
    # Generate sample price data with trend and seasonality
    trend = np.linspace(100, 120, n_periods)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_periods) / 12)  # Monthly seasonality
    noise = np.random.normal(0, 2, n_periods)
    prices = trend + seasonal + noise
    
    # Generate returns
    returns = [0.0] + [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
    
    # Generate volume and open interest
    volume = np.random.lognormal(10, 0.5, n_periods).tolist()
    open_interest = np.random.lognormal(12, 0.3, n_periods).tolist()
    
    # Create futures data object
    futures_data = FuturesTimeSeriesData(
        prices=prices.tolist(),
        returns=returns,
        timestamps=timestamps,
        volume=volume,
        open_interest=open_interest,
        contract_symbol="CL_Z23",
        underlying_asset="Crude Oil"
    )
    
    # Create additional series for VAR analysis
    additional_series = {
        'volume': volume,
        'open_interest': open_interest
    }
    
    # Initialize analyzer
    analyzer = FuturesTimeSeriesAnalyzer()
    
    # Perform analysis
    print("Performing comprehensive time series analysis...")
    results = analyzer.analyze(futures_data, additional_series)
    
    # Print summary
    print("\n" + "="*50)
    print("FUTURES TIME SERIES ANALYSIS SUMMARY")
    print("="*50)
    
    # Model Performance
    if results.model_performance:
        print("\nMODEL PERFORMANCE:")
        for metric, value in results.model_performance.items():
            print(f"  {metric}: {value:.4f}")
    
    # Risk Metrics
    if results.risk_metrics:
        print("\nRISK METRICS:")
        for metric, value in results.risk_metrics.items():
            if 'volatility' in metric.lower() or 'var' in metric.lower():
                print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value:.4f}")
    
    # Key Insights
    if results.insights:
        print("\nKEY INSIGHTS:")
        for i, insight in enumerate(results.insights[:5], 1):
            print(f"  {i}. {insight}")
    
    # Recommendations
    if results.recommendations:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(results.recommendations[:5], 1):
            print(f"  {i}. {rec}")
    
    # Generate and save report
    try:
        report = analyzer.generate_report(futures_data, results)
        print("\n" + "="*50)
        print("FULL REPORT GENERATED")
        print("="*50)
        print(report[:1000] + "..." if len(report) > 1000 else report)
    except Exception as e:
        print(f"Report generation failed: {e}")
    
    # Plot results
    try:
        print("\nGenerating plots...")
        analyzer.plot_results(futures_data, results)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\nAnalysis completed successfully!")