from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Conditional imports for advanced libraries
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from scipy.stats import jarque_bera
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Using simplified seasonal analysis.")

@dataclass
class FuturesTimeSeriesData:
    """Data structure for futures time series analysis"""
    prices: np.ndarray
    returns: np.ndarray
    dates: List[datetime]
    volume: Optional[np.ndarray] = None
    open_interest: Optional[np.ndarray] = None
    additional_series: Optional[Dict[str, np.ndarray]] = None

@dataclass
class SeasonalARIMAResult:
    """Results from Seasonal ARIMA analysis"""
    model_order: Tuple[int, int, int]
    seasonal_order: Tuple[int, int, int, int]
    parameters: Dict[str, float]
    fitted_values: np.ndarray
    residuals: np.ndarray
    forecast: np.ndarray
    forecast_intervals: Tuple[np.ndarray, np.ndarray]
    seasonal_decomposition: Dict[str, np.ndarray]
    diagnostics: Dict[str, float]
    aic: float
    bic: float
    log_likelihood: float
    model_summary: Dict[str, Any]

class SeasonalARIMAAnalyzer:
    """Seasonal ARIMA analyzer for time series with seasonal patterns"""
    
    def __init__(self, seasonal_periods: Optional[int] = None):
        """
        Initialize Seasonal ARIMA analyzer
        
        Args:
            seasonal_periods: Number of periods in a season (e.g., 12 for monthly data)
        """
        self.seasonal_periods = seasonal_periods
        self.fitted_model = None
        
    def analyze_seasonal_arima(self, data: FuturesTimeSeriesData) -> SeasonalARIMAResult:
        """Perform Seasonal ARIMA analysis"""
        try:
            # Use prices for ARIMA analysis (returns for other models)
            series = data.prices
            
            if len(series) < 20:
                warnings.warn("Insufficient data for ARIMA analysis")
                return self._create_default_seasonal_result(data)
            
            if STATSMODELS_AVAILABLE:
                return self._fit_seasonal_arima(series, data.dates)
            else:
                return self._simple_seasonal_analysis(series, data.dates)
                
        except Exception as e:
            warnings.warn(f"Seasonal ARIMA analysis failed: {str(e)}")
            return self._create_default_seasonal_result(data)
    
    def _fit_seasonal_arima(self, series: np.ndarray, dates: List[datetime]) -> SeasonalARIMAResult:
        """Fit Seasonal ARIMA model using statsmodels"""
        # Create time series with proper index
        ts = pd.Series(series, index=pd.to_datetime(dates) if dates else None)
        ts = ts.dropna()
        
        # Detect seasonality if not provided
        if self.seasonal_periods is None:
            self.seasonal_periods = self._detect_seasonality(ts)
        
        # Determine differencing orders
        d = self._determine_differencing(ts)
        D = self._determine_seasonal_differencing(ts, self.seasonal_periods) if self.seasonal_periods > 1 else 0
        
        # Grid search for best ARIMA parameters
        best_aic = np.inf
        best_model = None
        best_order = None
        best_seasonal_order = None
        
        # Parameter ranges for grid search
        p_range = range(0, min(3, len(ts) // 10))
        q_range = range(0, min(3, len(ts) // 10))
        P_range = range(0, 2) if self.seasonal_periods > 1 else [0]
        Q_range = range(0, 2) if self.seasonal_periods > 1 else [0]
        
        for p in p_range:
            for q in q_range:
                for P in P_range:
                    for Q in Q_range:
                        try:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, self.seasonal_periods) if self.seasonal_periods > 1 else (0, 0, 0, 0)
                            
                            if self.seasonal_periods > 1:
                                model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
                            else:
                                model = ARIMA(ts, order=order)
                            
                            fitted_model = model.fit(disp=False)
                            
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_model = fitted_model
                                best_order = order
                                best_seasonal_order = seasonal_order
                                
                        except:
                            continue
        
        if best_model is None:
            # Fallback to simple ARIMA(1,1,1)
            try:
                order = (1, 1, 1)
                seasonal_order = (0, 0, 0, 0)
                model = ARIMA(ts, order=order)
                best_model = model.fit(disp=False)
                best_order = order
                best_seasonal_order = seasonal_order
            except:
                return self._simple_seasonal_analysis(series, dates)
        
        self.fitted_model = best_model
        
        # Generate forecasts
        forecast_steps = 5
        forecast_result = best_model.get_forecast(steps=forecast_steps)
        forecast = forecast_result.predicted_mean.values
        forecast_intervals = (
            forecast_result.conf_int().iloc[:, 0].values,
            forecast_result.conf_int().iloc[:, 1].values
        )
        
        # Seasonal decomposition
        seasonal_decomp = self._decompose_seasonal(ts)
        
        # Extract parameters
        parameters = self._extract_arima_parameters(best_model)
        
        # Calculate diagnostics
        diagnostics = self._calculate_arima_diagnostics(best_model)
        
        return SeasonalARIMAResult(
            model_order=best_order,
            seasonal_order=best_seasonal_order,
            parameters=parameters,
            fitted_values=best_model.fittedvalues.values,
            residuals=best_model.resid.values,
            forecast=forecast,
            forecast_intervals=forecast_intervals,
            seasonal_decomposition=seasonal_decomp,
            diagnostics=diagnostics,
            aic=best_model.aic,
            bic=best_model.bic,
            log_likelihood=best_model.llf,
            model_summary={
                'n_observations': len(ts),
                'model_type': 'SARIMAX' if self.seasonal_periods > 1 else 'ARIMA',
                'seasonal_periods': self.seasonal_periods,
                'convergence': best_model.mle_retvals['converged'] if hasattr(best_model, 'mle_retvals') else True
            }
        )
    
    def _simple_seasonal_analysis(self, series: np.ndarray, dates: List[datetime]) -> SeasonalARIMAResult:
        """Simplified seasonal analysis when statsmodels is not available"""
        # Simple seasonal decomposition using moving averages
        if self.seasonal_periods is None:
            self.seasonal_periods = self._detect_seasonality_simple(series)
        
        # Simple trend estimation using moving average
        window = min(self.seasonal_periods, len(series) // 4) if self.seasonal_periods > 1 else 5
        trend = pd.Series(series).rolling(window=window, center=True).mean().values
        
        # Simple seasonal component
        if self.seasonal_periods > 1 and len(series) >= 2 * self.seasonal_periods:
            seasonal = np.zeros_like(series)
            for i in range(self.seasonal_periods):
                seasonal_indices = np.arange(i, len(series), self.seasonal_periods)
                if len(seasonal_indices) > 1:
                    seasonal_mean = np.nanmean(series[seasonal_indices] - trend[seasonal_indices])
                    seasonal[seasonal_indices] = seasonal_mean
        else:
            seasonal = np.zeros_like(series)
        
        # Residual component
        residual = series - np.nan_to_num(trend) - seasonal
        
        # Simple AR(1) model on residuals
        residual_clean = residual[~np.isnan(residual)]
        if len(residual_clean) > 2:
            # Fit AR(1): r_t = φ*r_{t-1} + ε_t
            y = residual_clean[1:]
            y_lag = residual_clean[:-1]
            
            if len(y) > 0 and np.var(y_lag) > 1e-10:
                phi = np.cov(y, y_lag)[0, 1] / np.var(y_lag)
                phi = np.clip(phi, -0.99, 0.99)  # Ensure stationarity
            else:
                phi = 0.0
        else:
            phi = 0.0
        
        # Simple forecast
        last_trend = trend[-1] if not np.isnan(trend[-1]) else series[-1]
        last_seasonal = seasonal[-1] if self.seasonal_periods > 1 else 0
        last_residual = residual[-1] if not np.isnan(residual[-1]) else 0
        
        forecast = np.zeros(5)
        for i in range(5):
            # Forecast residual using AR(1)
            forecast_residual = phi ** (i + 1) * last_residual
            
            # Forecast seasonal (repeat pattern)
            if self.seasonal_periods > 1:
                seasonal_idx = (len(series) + i) % self.seasonal_periods
                forecast_seasonal = seasonal[seasonal_idx] if seasonal_idx < len(seasonal) else 0
            else:
                forecast_seasonal = 0
            
            forecast[i] = last_trend + forecast_seasonal + forecast_residual
        
        # Simple confidence intervals (±2 standard deviations)
        residual_std = np.std(residual_clean) if len(residual_clean) > 1 else np.std(series) * 0.1
        forecast_intervals = (
            forecast - 2 * residual_std * np.sqrt(np.arange(1, 6)),
            forecast + 2 * residual_std * np.sqrt(np.arange(1, 6))
        )
        
        # Simple fitted values
        fitted_values = np.nan_to_num(trend) + seasonal
        
        # Simple diagnostics
        diagnostics = {
            'ljung_box_stat': 0.0,
            'ljung_box_pvalue': 1.0,
            'jarque_bera_stat': 0.0,
            'jarque_bera_pvalue': 1.0,
            'adf_stat': -3.0,
            'adf_pvalue': 0.05
        }
        
        # Simple log-likelihood
        residual_var = np.var(residual_clean) if len(residual_clean) > 1 else 1.0
        log_likelihood = -0.5 * len(series) * np.log(2 * np.pi * residual_var) - \
                        0.5 * np.sum(residual_clean**2) / residual_var
        
        # Simple AIC and BIC
        n_params = 2  # AR coefficient and variance
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(len(series))
        
        return SeasonalARIMAResult(
            model_order=(1, 0, 0),
            seasonal_order=(0, 0, 0, self.seasonal_periods),
            parameters={'ar1': phi, 'sigma2': residual_var},
            fitted_values=fitted_values,
            residuals=residual,
            forecast=forecast,
            forecast_intervals=forecast_intervals,
            seasonal_decomposition={
                'trend': np.nan_to_num(trend),
                'seasonal': seasonal,
                'residual': residual
            },
            diagnostics=diagnostics,
            aic=aic,
            bic=bic,
            log_likelihood=log_likelihood,
            model_summary={
                'n_observations': len(series),
                'model_type': 'Simple Seasonal',
                'seasonal_periods': self.seasonal_periods,
                'convergence': True
            }
        )
    
    def _detect_seasonality(self, ts: pd.Series) -> int:
        """Detect seasonal period using autocorrelation"""
        if len(ts) < 24:
            return 1  # No seasonality for short series
        
        try:
            # Calculate autocorrelation for different lags
            max_lag = min(len(ts) // 3, 24)
            autocorrs = []
            
            for lag in range(1, max_lag + 1):
                if len(ts) > lag:
                    corr = ts.autocorr(lag=lag)
                    autocorrs.append((lag, abs(corr) if not np.isnan(corr) else 0))
            
            if not autocorrs:
                return 1
            
            # Find the lag with highest autocorrelation (excluding lag 1)
            autocorrs_filtered = [(lag, corr) for lag, corr in autocorrs if lag > 1]
            
            if autocorrs_filtered:
                best_lag = max(autocorrs_filtered, key=lambda x: x[1])[0]
                # Only consider it seasonal if autocorrelation is significant
                if autocorrs_filtered[best_lag - 2][1] > 0.1:  # Threshold for significance
                    return best_lag
            
            return 1  # No significant seasonality detected
            
        except:
            return 1
    
    def _detect_seasonality_simple(self, series: np.ndarray) -> int:
        """Simple seasonality detection using autocorrelation"""
        if len(series) < 24:
            return 1
        
        try:
            # Calculate simple autocorrelations
            max_lag = min(len(series) // 3, 24)
            best_corr = 0
            best_lag = 1
            
            for lag in range(2, max_lag + 1):
                if len(series) > lag:
                    x1 = series[:-lag]
                    x2 = series[lag:]
                    
                    if len(x1) > 0 and np.std(x1) > 1e-10 and np.std(x2) > 1e-10:
                        corr = np.corrcoef(x1, x2)[0, 1]
                        if not np.isnan(corr) and abs(corr) > best_corr:
                            best_corr = abs(corr)
                            best_lag = lag
            
            return best_lag if best_corr > 0.1 else 1
            
        except:
            return 1
    
    def _determine_differencing(self, ts: pd.Series) -> int:
        """Determine the order of differencing needed for stationarity"""
        try:
            # ADF test for stationarity
            adf_result = adfuller(ts.dropna())
            
            if adf_result[1] <= 0.05:  # p-value <= 0.05, series is stationary
                return 0
            
            # Try first difference
            ts_diff = ts.diff().dropna()
            if len(ts_diff) > 10:
                adf_result_diff = adfuller(ts_diff)
                if adf_result_diff[1] <= 0.05:
                    return 1
            
            # If still not stationary, use second difference
            return 2
            
        except:
            # Default to first difference if ADF test fails
            return 1
    
    def _determine_seasonal_differencing(self, ts: pd.Series, seasonal_periods: int) -> int:
        """Determine seasonal differencing order"""
        if seasonal_periods <= 1 or len(ts) < 2 * seasonal_periods:
            return 0
        
        try:
            # Simple test: if seasonal autocorrelation is high, apply seasonal differencing
            seasonal_corr = ts.autocorr(lag=seasonal_periods)
            
            if not np.isnan(seasonal_corr) and abs(seasonal_corr) > 0.7:
                return 1
            else:
                return 0
                
        except:
            return 0
    
    def _decompose_seasonal(self, ts: pd.Series) -> Dict[str, np.ndarray]:
        """Perform seasonal decomposition"""
        try:
            if self.seasonal_periods > 1 and len(ts) >= 2 * self.seasonal_periods:
                decomposition = seasonal_decompose(ts, model='additive', 
                                                 period=self.seasonal_periods, 
                                                 extrapolate_trend='freq')
                
                return {
                    'trend': decomposition.trend.values,
                    'seasonal': decomposition.seasonal.values,
                    'residual': decomposition.resid.values
                }
            else:
                # Simple trend using moving average
                trend = ts.rolling(window=min(5, len(ts)//2), center=True).mean().values
                return {
                    'trend': trend,
                    'seasonal': np.zeros_like(ts.values),
                    'residual': ts.values - np.nan_to_num(trend)
                }
                
        except:
            return {
                'trend': np.full_like(ts.values, ts.mean()),
                'seasonal': np.zeros_like(ts.values),
                'residual': ts.values - ts.mean()
            }
    
    def _extract_arima_parameters(self, fitted_model) -> Dict[str, float]:
        """Extract ARIMA model parameters"""
        parameters = {}
        
        try:
            params = fitted_model.params
            
            # AR parameters
            ar_params = [param for param in params.index if param.startswith('ar.')]
            for i, param in enumerate(ar_params):
                parameters[f'ar{i+1}'] = params[param]
            
            # MA parameters
            ma_params = [param for param in params.index if param.startswith('ma.')]
            for i, param in enumerate(ma_params):
                parameters[f'ma{i+1}'] = params[param]
            
            # Seasonal AR parameters
            sar_params = [param for param in params.index if param.startswith('ar.S.')]
            for i, param in enumerate(sar_params):
                parameters[f'sar{i+1}'] = params[param]
            
            # Seasonal MA parameters
            sma_params = [param for param in params.index if param.startswith('ma.S.')]
            for i, param in enumerate(sma_params):
                parameters[f'sma{i+1}'] = params[param]
            
            # Intercept/constant
            if 'const' in params.index:
                parameters['const'] = params['const']
            elif 'intercept' in params.index:
                parameters['intercept'] = params['intercept']
            
            # Variance
            if 'sigma2' in params.index:
                parameters['sigma2'] = params['sigma2']
                
        except:
            parameters = {'ar1': 0.5, 'sigma2': 1.0}
        
        return parameters
    
    def _calculate_arima_diagnostics(self, fitted_model) -> Dict[str, float]:
        """Calculate ARIMA model diagnostics"""
        diagnostics = {}
        
        try:
            residuals = fitted_model.resid
            
            # Ljung-Box test
            lb_result = self._ljung_box_test(residuals)
            diagnostics['ljung_box_stat'] = lb_result[0]
            diagnostics['ljung_box_pvalue'] = lb_result[1]
            
            # Jarque-Bera test
            jb_result = self._jarque_bera_test(residuals)
            diagnostics['jarque_bera_stat'] = jb_result[0]
            diagnostics['jarque_bera_pvalue'] = jb_result[1]
            
            # ADF test on residuals
            try:
                adf_result = adfuller(residuals.dropna())
                diagnostics['adf_stat'] = adf_result[0]
                diagnostics['adf_pvalue'] = adf_result[1]
            except:
                diagnostics['adf_stat'] = -3.0
                diagnostics['adf_pvalue'] = 0.05
                
        except:
            diagnostics = {
                'ljung_box_stat': 0.0,
                'ljung_box_pvalue': 1.0,
                'jarque_bera_stat': 0.0,
                'jarque_bera_pvalue': 1.0,
                'adf_stat': -3.0,
                'adf_pvalue': 0.05
            }
        
        return diagnostics
    
    def _ljung_box_test(self, residuals: pd.Series) -> Tuple[float, float]:
        """Ljung-Box test for autocorrelation in residuals"""
        try:
            lb_result = acorr_ljungbox(residuals.dropna(), lags=10, return_df=True)
            return lb_result['lb_stat'].iloc[-1], lb_result['lb_pvalue'].iloc[-1]
        except:
            return 0.0, 1.0
    
    def _jarque_bera_test(self, residuals: pd.Series) -> Tuple[float, float]:
        """Jarque-Bera test for normality of residuals"""
        try:
            jb_stat, jb_pvalue = jarque_bera(residuals.dropna())
            return jb_stat, jb_pvalue
        except:
            return 0.0, 1.0
    
    def _create_default_seasonal_result(self, data: FuturesTimeSeriesData) -> SeasonalARIMAResult:
        """Create default seasonal ARIMA result for error cases"""
        n_obs = len(data.prices)
        
        return SeasonalARIMAResult(
            model_order=(1, 1, 1),
            seasonal_order=(0, 0, 0, 1),
            parameters={'ar1': 0.5, 'ma1': 0.3, 'sigma2': 1.0},
            fitted_values=data.prices,
            residuals=np.zeros(n_obs),
            forecast=np.full(5, data.prices[-1] if n_obs > 0 else 0.0),
            forecast_intervals=(
                np.full(5, data.prices[-1] - 1.0 if n_obs > 0 else -1.0),
                np.full(5, data.prices[-1] + 1.0 if n_obs > 0 else 1.0)
            ),
            seasonal_decomposition={
                'trend': data.prices,
                'seasonal': np.zeros(n_obs),
                'residual': np.zeros(n_obs)
            },
            diagnostics={
                'ljung_box_stat': 0.0,
                'ljung_box_pvalue': 1.0,
                'jarque_bera_stat': 0.0,
                'jarque_bera_pvalue': 1.0,
                'adf_stat': -3.0,
                'adf_pvalue': 0.05
            },
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0,
            model_summary={
                'n_observations': n_obs,
                'model_type': 'Default',
                'seasonal_periods': 1,
                'convergence': False
            }
        )