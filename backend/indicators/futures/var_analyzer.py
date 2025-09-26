from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Conditional imports for advanced libraries
try:
    from statsmodels.tsa.api import VAR
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Using simplified VAR analysis.")

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
class VARResult:
    """Results from VAR analysis"""
    model_summary: Dict[str, Any]
    coefficients: Dict[str, np.ndarray]
    residuals: np.ndarray
    forecast: Dict[str, np.ndarray]
    forecast_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]
    granger_causality: Dict[str, Dict[str, float]]
    impulse_response: Dict[str, np.ndarray]
    variance_decomposition: Dict[str, np.ndarray]
    diagnostics: Dict[str, float]
    aic: float
    bic: float
    log_likelihood: float

class VARAnalyzer:
    """Vector Autoregression (VAR) analyzer for multivariate time series"""
    
    def __init__(self, max_lags: int = 10):
        self.max_lags = max_lags
        self.model = None
        self.fitted_model = None
        
    def analyze_var(self, data: FuturesTimeSeriesData, 
                   additional_series: Optional[Dict[str, np.ndarray]] = None) -> VARResult:
        """Perform VAR analysis on multivariate time series"""
        try:
            if not additional_series and not data.additional_series:
                # Cannot perform VAR with single series
                return self._create_default_var_result()
                
            if STATSMODELS_AVAILABLE:
                return self._fit_var_model(data, additional_series)
            else:
                return self._simple_var_analysis(data, additional_series)
                
        except Exception as e:
            warnings.warn(f"VAR analysis failed: {str(e)}")
            return self._create_default_var_result()
    
    def _fit_var_model(self, data: FuturesTimeSeriesData, 
                      additional_series: Optional[Dict[str, np.ndarray]] = None) -> VARResult:
        """Fit VAR model using statsmodels"""
        # Prepare multivariate data
        series_dict = {'returns': data.returns}
        
        if additional_series:
            series_dict.update(additional_series)
        elif data.additional_series:
            series_dict.update(data.additional_series)
            
        # Create DataFrame
        min_length = min(len(series) for series in series_dict.values())
        df_data = {name: series[:min_length] for name, series in series_dict.items()}
        df = pd.DataFrame(df_data)
        
        # Remove any NaN values
        df = df.dropna()
        
        if len(df) < 20:  # Minimum data requirement
            return self._simple_var_analysis(data, additional_series)
        
        # Fit VAR model
        model = VAR(df)
        
        # Select optimal lag order
        lag_order = self._select_lag_order(model)
        
        # Fit model with selected lags
        fitted_model = model.fit(lag_order)
        self.fitted_model = fitted_model
        
        # Extract results
        coefficients = self._extract_coefficients(fitted_model)
        residuals = fitted_model.resid.values
        
        # Generate forecasts
        forecast_steps = 5
        forecast_result = fitted_model.forecast(df.values[-lag_order:], steps=forecast_steps)
        forecast_intervals = self._calculate_forecast_intervals(fitted_model, forecast_steps)
        
        # Granger causality tests
        granger_results = self._perform_granger_causality_tests(df)
        
        # Impulse response functions
        impulse_response = self._calculate_impulse_response(fitted_model)
        
        # Variance decomposition
        variance_decomp = self._calculate_variance_decomposition(fitted_model)
        
        # Model diagnostics
        diagnostics = self._calculate_diagnostics(fitted_model)
        
        return VARResult(
            model_summary={
                'lag_order': lag_order,
                'n_observations': len(df),
                'n_variables': len(df.columns),
                'variable_names': list(df.columns)
            },
            coefficients=coefficients,
            residuals=residuals,
            forecast={
                col: forecast_result[:, i] 
                for i, col in enumerate(df.columns)
            },
            forecast_intervals=forecast_intervals,
            granger_causality=granger_results,
            impulse_response=impulse_response,
            variance_decomposition=variance_decomp,
            diagnostics=diagnostics,
            aic=fitted_model.aic,
            bic=fitted_model.bic,
            log_likelihood=fitted_model.llf
        )
    
    def _select_lag_order(self, model: 'VAR') -> int:
        """Select optimal lag order using information criteria"""
        try:
            lag_order_results = model.select_order(maxlags=min(self.max_lags, 10))
            return lag_order_results.aic
        except:
            return min(2, self.max_lags)
    
    def _extract_coefficients(self, fitted_model) -> Dict[str, np.ndarray]:
        """Extract VAR coefficients"""
        coefficients = {}
        for i, var_name in enumerate(fitted_model.names):
            coefficients[var_name] = fitted_model.coefs[:, :, i]
        return coefficients
    
    def _calculate_forecast_intervals(self, fitted_model, steps: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Calculate forecast confidence intervals (simplified)"""
        # This is a simplified implementation
        # In practice, you'd use the model's forecast error variance
        intervals = {}
        residual_std = np.std(fitted_model.resid.values, axis=0)
        
        for i, var_name in enumerate(fitted_model.names):
            std_err = residual_std[i] * np.sqrt(np.arange(1, steps + 1))
            intervals[var_name] = (-1.96 * std_err, 1.96 * std_err)
            
        return intervals
    
    def _perform_granger_causality_tests(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Perform Granger causality tests between variables"""
        results = {}
        variables = list(df.columns)
        
        for i, var1 in enumerate(variables):
            results[var1] = {}
            for j, var2 in enumerate(variables):
                if i != j:
                    try:
                        # Test if var2 Granger-causes var1
                        test_data = df[[var1, var2]].dropna()
                        if len(test_data) > 20:
                            gc_result = grangercausalitytests(test_data, maxlag=4, verbose=False)
                            # Get p-value from F-test at lag 1
                            p_value = gc_result[1][0]['ssr_ftest'][1]
                            results[var1][var2] = p_value
                        else:
                            results[var1][var2] = 1.0
                    except:
                        results[var1][var2] = 1.0
                        
        return results
    
    def _calculate_impulse_response(self, fitted_model) -> Dict[str, np.ndarray]:
        """Calculate impulse response functions"""
        try:
            irf = fitted_model.irf(periods=10)
            impulse_responses = {}
            
            for i, var_name in enumerate(fitted_model.names):
                impulse_responses[var_name] = irf.irfs[:, :, i]
                
            return impulse_responses
        except:
            # Return empty dict if calculation fails
            return {var: np.zeros((10, len(fitted_model.names))) 
                   for var in fitted_model.names}
    
    def _calculate_variance_decomposition(self, fitted_model) -> Dict[str, np.ndarray]:
        """Calculate forecast error variance decomposition"""
        try:
            fevd = fitted_model.fevd(periods=10)
            variance_decomp = {}
            
            for i, var_name in enumerate(fitted_model.names):
                variance_decomp[var_name] = fevd.decomp[:, :, i]
                
            return variance_decomp
        except:
            # Return uniform decomposition if calculation fails
            n_vars = len(fitted_model.names)
            uniform_decomp = np.ones((10, n_vars)) / n_vars
            return {var: uniform_decomp for var in fitted_model.names}
    
    def _calculate_diagnostics(self, fitted_model) -> Dict[str, float]:
        """Calculate model diagnostics"""
        diagnostics = {}
        
        try:
            # Ljung-Box test for residual autocorrelation
            residuals = fitted_model.resid
            lb_stats = []
            lb_pvalues = []
            
            for col in residuals.columns:
                lb_result = acorr_ljungbox(residuals[col], lags=10, return_df=True)
                lb_stats.append(lb_result['lb_stat'].iloc[-1])
                lb_pvalues.append(lb_result['lb_pvalue'].iloc[-1])
            
            diagnostics['ljung_box_stat'] = np.mean(lb_stats)
            diagnostics['ljung_box_pvalue'] = np.mean(lb_pvalues)
            
        except:
            diagnostics['ljung_box_stat'] = 0.0
            diagnostics['ljung_box_pvalue'] = 1.0
        
        # Additional diagnostics
        diagnostics['determinant'] = fitted_model.detomega
        diagnostics['log_likelihood'] = fitted_model.llf
        
        return diagnostics
    
    def _simple_var_analysis(self, data: FuturesTimeSeriesData, 
                           additional_series: Optional[Dict[str, np.ndarray]] = None) -> VARResult:
        """Simplified VAR analysis when statsmodels is not available"""
        # Prepare data
        series_dict = {'returns': data.returns}
        
        if additional_series:
            series_dict.update(additional_series)
        elif data.additional_series:
            series_dict.update(data.additional_series)
        
        # Simple AR(1) model for each series
        coefficients = {}
        residuals_list = []
        forecasts = {}
        
        for name, series in series_dict.items():
            # Fit AR(1): y_t = c + φ*y_{t-1} + ε_t
            y = series[1:]
            y_lag = series[:-1]
            
            if len(y) > 2:
                # Simple linear regression
                X = np.column_stack([np.ones(len(y_lag)), y_lag])
                try:
                    coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    coefficients[name] = coef.reshape(1, -1)
                    
                    # Calculate residuals
                    y_pred = X @ coef
                    residuals = y - y_pred
                    residuals_list.append(residuals)
                    
                    # Simple forecast
                    last_value = series[-1]
                    forecast_1 = coef[0] + coef[1] * last_value
                    forecasts[name] = np.array([forecast_1])
                    
                except:
                    coefficients[name] = np.array([[0.0, 0.5]])
                    residuals_list.append(np.zeros(len(y)))
                    forecasts[name] = np.array([series[-1]])
            else:
                coefficients[name] = np.array([[0.0, 0.5]])
                residuals_list.append(np.zeros(1))
                forecasts[name] = np.array([series[-1] if len(series) > 0 else 0.0])
        
        # Combine residuals
        min_length = min(len(r) for r in residuals_list) if residuals_list else 1
        combined_residuals = np.column_stack([r[:min_length] for r in residuals_list])
        
        # Simple correlation-based "causality"
        granger_results = {}
        var_names = list(series_dict.keys())
        
        for var1 in var_names:
            granger_results[var1] = {}
            for var2 in var_names:
                if var1 != var2:
                    try:
                        corr = np.corrcoef(series_dict[var1][:-1], series_dict[var2][1:])[0, 1]
                        # Convert correlation to pseudo p-value
                        granger_results[var1][var2] = max(0.01, 1 - abs(corr))
                    except:
                        granger_results[var1][var2] = 1.0
        
        return VARResult(
            model_summary={
                'lag_order': 1,
                'n_observations': min_length,
                'n_variables': len(series_dict),
                'variable_names': var_names
            },
            coefficients=coefficients,
            residuals=combined_residuals,
            forecast=forecasts,
            forecast_intervals={
                name: (np.array([-0.1]), np.array([0.1])) 
                for name in forecasts.keys()
            },
            granger_causality=granger_results,
            impulse_response={
                name: np.zeros((5, len(var_names))) 
                for name in var_names
            },
            variance_decomposition={
                name: np.ones((5, len(var_names))) / len(var_names) 
                for name in var_names
            },
            diagnostics={
                'ljung_box_stat': 0.0,
                'ljung_box_pvalue': 1.0,
                'determinant': 1.0,
                'log_likelihood': 0.0
            },
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0
        )
    
    def _create_default_var_result(self) -> VARResult:
        """Create default VAR result for error cases"""
        return VARResult(
            model_summary={
                'lag_order': 1,
                'n_observations': 0,
                'n_variables': 1,
                'variable_names': ['returns']
            },
            coefficients={'returns': np.array([[0.0, 0.5]])},
            residuals=np.array([[0.0]]),
            forecast={'returns': np.array([0.0])},
            forecast_intervals={'returns': (np.array([-0.1]), np.array([0.1]))},
            granger_causality={'returns': {}},
            impulse_response={'returns': np.zeros((5, 1))},
            variance_decomposition={'returns': np.ones((5, 1))},
            diagnostics={
                'ljung_box_stat': 0.0,
                'ljung_box_pvalue': 1.0,
                'determinant': 1.0,
                'log_likelihood': 0.0
            },
            aic=0.0,
            bic=0.0,
            log_likelihood=0.0
        )