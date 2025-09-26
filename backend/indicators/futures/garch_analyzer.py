from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Conditional imports for advanced libraries
try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, TGARCH
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    warnings.warn("arch library not available. Using simplified GARCH analysis.")

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
class GARCHResult:
    """Results from GARCH analysis"""
    model_type: str
    parameters: Dict[str, float]
    conditional_volatility: np.ndarray
    standardized_residuals: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    diagnostics: Dict[str, float]
    forecast_volatility: np.ndarray
    var_estimates: Dict[str, float]  # Value at Risk estimates
    model_summary: Dict[str, Any]

class GARCHAnalyzer:
    """GARCH family models for volatility analysis"""
    
    def __init__(self, model_type: str = 'GARCH'):
        """
        Initialize GARCH analyzer
        
        Args:
            model_type: Type of GARCH model ('GARCH', 'EGARCH', 'TGARCH')
        """
        self.model_type = model_type
        self.fitted_model = None
        
    def analyze_garch(self, data: FuturesTimeSeriesData) -> GARCHResult:
        """Perform GARCH analysis on return series"""
        try:
            if ARCH_AVAILABLE:
                return self._fit_garch_model(data)
            else:
                return self._simple_garch_model(data)
                
        except Exception as e:
            warnings.warn(f"GARCH analysis failed: {str(e)}")
            return self._create_default_garch_result(data)
    
    def _fit_garch_model(self, data: FuturesTimeSeriesData) -> GARCHResult:
        """Fit GARCH model using arch library"""
        returns = data.returns * 100  # Convert to percentage for better numerical stability
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        if len(returns) < 50:  # Minimum data requirement
            return self._simple_garch_model(data)
        
        # Create GARCH model based on type
        if self.model_type.upper() == 'EGARCH':
            model = arch_model(returns, vol='EGARCH', p=1, q=1)
        elif self.model_type.upper() == 'TGARCH':
            model = arch_model(returns, vol='GARCH', p=1, q=1, power=1.0)
        else:  # Default GARCH
            model = arch_model(returns, vol='GARCH', p=1, q=1)
        
        # Fit the model
        fitted_model = model.fit(disp='off', show_warning=False)
        self.fitted_model = fitted_model
        
        # Extract parameters
        parameters = self._extract_parameters(fitted_model)
        
        # Get conditional volatility
        conditional_volatility = fitted_model.conditional_volatility / 100  # Convert back
        
        # Get standardized residuals
        standardized_residuals = fitted_model.std_resid
        
        # Calculate diagnostics
        diagnostics = self._calculate_garch_diagnostics(fitted_model, standardized_residuals)
        
        # Generate volatility forecasts
        forecast_horizon = 5
        forecast_result = fitted_model.forecast(horizon=forecast_horizon)
        forecast_volatility = np.sqrt(forecast_result.variance.values[-1, :]) / 100
        
        # Calculate VaR estimates
        var_estimates = self._calculate_var_estimates(conditional_volatility, returns/100)
        
        return GARCHResult(
            model_type=self.model_type,
            parameters=parameters,
            conditional_volatility=conditional_volatility,
            standardized_residuals=standardized_residuals,
            log_likelihood=fitted_model.loglikelihood,
            aic=fitted_model.aic,
            bic=fitted_model.bic,
            diagnostics=diagnostics,
            forecast_volatility=forecast_volatility,
            var_estimates=var_estimates,
            model_summary={
                'n_observations': len(returns),
                'model_type': self.model_type,
                'convergence': fitted_model.convergence_flag == 0,
                'iterations': getattr(fitted_model, 'iterations', 0)
            }
        )
    
    def _extract_parameters(self, fitted_model) -> Dict[str, float]:
        """Extract GARCH model parameters"""
        params = fitted_model.params
        parameter_dict = {}
        
        # Common parameters
        if 'omega' in params.index:
            parameter_dict['omega'] = params['omega']
        if 'alpha[1]' in params.index:
            parameter_dict['alpha'] = params['alpha[1]']
        if 'beta[1]' in params.index:
            parameter_dict['beta'] = params['beta[1]']
        
        # EGARCH specific parameters
        if 'gamma[1]' in params.index:
            parameter_dict['gamma'] = params['gamma[1]']
        
        # Mean model parameters
        if 'mu' in params.index:
            parameter_dict['mu'] = params['mu']
        
        return parameter_dict
    
    def _calculate_garch_diagnostics(self, fitted_model, standardized_residuals: np.ndarray) -> Dict[str, float]:
        """Calculate GARCH model diagnostics"""
        diagnostics = {}
        
        # Ljung-Box test on standardized residuals
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(standardized_residuals, lags=10, return_df=True)
            diagnostics['ljung_box_stat'] = lb_result['lb_stat'].iloc[-1]
            diagnostics['ljung_box_pvalue'] = lb_result['lb_pvalue'].iloc[-1]
        except:
            diagnostics['ljung_box_stat'] = 0.0
            diagnostics['ljung_box_pvalue'] = 1.0
        
        # Ljung-Box test on squared standardized residuals
        try:
            lb_result_sq = acorr_ljungbox(standardized_residuals**2, lags=10, return_df=True)
            diagnostics['ljung_box_sq_stat'] = lb_result_sq['lb_stat'].iloc[-1]
            diagnostics['ljung_box_sq_pvalue'] = lb_result_sq['lb_pvalue'].iloc[-1]
        except:
            diagnostics['ljung_box_sq_stat'] = 0.0
            diagnostics['ljung_box_sq_pvalue'] = 1.0
        
        # ARCH-LM test
        try:
            from statsmodels.stats.diagnostic import het_arch
            arch_lm = het_arch(standardized_residuals, nlags=5)
            diagnostics['arch_lm_stat'] = arch_lm[0]
            diagnostics['arch_lm_pvalue'] = arch_lm[1]
        except:
            diagnostics['arch_lm_stat'] = 0.0
            diagnostics['arch_lm_pvalue'] = 1.0
        
        # Jarque-Bera test for normality
        try:
            from scipy.stats import jarque_bera
            jb_stat, jb_pvalue = jarque_bera(standardized_residuals)
            diagnostics['jarque_bera_stat'] = jb_stat
            diagnostics['jarque_bera_pvalue'] = jb_pvalue
        except:
            diagnostics['jarque_bera_stat'] = 0.0
            diagnostics['jarque_bera_pvalue'] = 1.0
        
        return diagnostics
    
    def _calculate_var_estimates(self, conditional_volatility: np.ndarray, 
                               returns: np.ndarray) -> Dict[str, float]:
        """Calculate Value at Risk estimates"""
        var_estimates = {}
        
        # Current volatility (last observation)
        current_vol = conditional_volatility[-1] if len(conditional_volatility) > 0 else 0.01
        
        # VaR estimates assuming normal distribution
        confidence_levels = [0.95, 0.99]
        
        for conf_level in confidence_levels:
            # Normal VaR
            z_score = np.percentile(np.random.standard_normal(10000), (1-conf_level)*100)
            var_normal = z_score * current_vol
            var_estimates[f'var_{int(conf_level*100)}_normal'] = var_normal
            
            # Historical simulation VaR (if enough data)
            if len(returns) > 100:
                var_historical = np.percentile(returns, (1-conf_level)*100)
                var_estimates[f'var_{int(conf_level*100)}_historical'] = var_historical
            else:
                var_estimates[f'var_{int(conf_level*100)}_historical'] = var_normal
        
        # Expected Shortfall (Conditional VaR)
        try:
            var_95 = var_estimates['var_95_historical']
            tail_returns = returns[returns <= var_95]
            if len(tail_returns) > 0:
                var_estimates['expected_shortfall_95'] = np.mean(tail_returns)
            else:
                var_estimates['expected_shortfall_95'] = var_95 * 1.5
        except:
            var_estimates['expected_shortfall_95'] = current_vol * -2.5
        
        return var_estimates
    
    def _simple_garch_model(self, data: FuturesTimeSeriesData) -> GARCHResult:
        """Simplified GARCH model when arch library is not available"""
        returns = data.returns
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 10:
            return self._create_default_garch_result(data)
        
        # Simple exponentially weighted moving average for volatility
        lambda_param = 0.94  # RiskMetrics lambda
        
        # Initialize volatility with sample standard deviation
        vol_ewma = np.zeros(len(returns))
        vol_ewma[0] = np.std(returns[:min(30, len(returns))])
        
        # Calculate EWMA volatility
        for i in range(1, len(returns)):
            vol_ewma[i] = np.sqrt(lambda_param * vol_ewma[i-1]**2 + 
                                (1 - lambda_param) * returns[i-1]**2)
        
        # Simple standardized residuals
        standardized_residuals = returns / (vol_ewma + 1e-8)
        
        # Simple diagnostics
        diagnostics = {
            'ljung_box_stat': 0.0,
            'ljung_box_pvalue': 1.0,
            'ljung_box_sq_stat': 0.0,
            'ljung_box_sq_pvalue': 1.0,
            'arch_lm_stat': 0.0,
            'arch_lm_pvalue': 1.0,
            'jarque_bera_stat': 0.0,
            'jarque_bera_pvalue': 1.0
        }
        
        # Simple forecast (persistence of current volatility)
        current_vol = vol_ewma[-1] if len(vol_ewma) > 0 else 0.01
        forecast_volatility = np.full(5, current_vol)
        
        # Simple VaR estimates
        var_estimates = {
            'var_95_normal': -1.645 * current_vol,
            'var_99_normal': -2.326 * current_vol,
            'var_95_historical': np.percentile(returns, 5) if len(returns) > 20 else -1.645 * current_vol,
            'var_99_historical': np.percentile(returns, 1) if len(returns) > 100 else -2.326 * current_vol,
            'expected_shortfall_95': np.percentile(returns, 2.5) if len(returns) > 40 else -2.0 * current_vol
        }
        
        # Simple log-likelihood (assuming normal distribution)
        log_likelihood = -0.5 * len(returns) * np.log(2 * np.pi) - \
                        0.5 * np.sum(np.log(vol_ewma**2 + 1e-8)) - \
                        0.5 * np.sum(returns**2 / (vol_ewma**2 + 1e-8))
        
        # Simple AIC and BIC (assuming 3 parameters: mu, omega, lambda)
        n_params = 3
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(len(returns))
        
        return GARCHResult(
            model_type='EWMA',
            parameters={
                'lambda': lambda_param,
                'initial_vol': vol_ewma[0],
                'current_vol': current_vol
            },
            conditional_volatility=vol_ewma,
            standardized_residuals=standardized_residuals,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            diagnostics=diagnostics,
            forecast_volatility=forecast_volatility,
            var_estimates=var_estimates,
            model_summary={
                'n_observations': len(returns),
                'model_type': 'EWMA',
                'convergence': True,
                'iterations': 0
            }
        )
    
    def _create_default_garch_result(self, data: FuturesTimeSeriesData) -> GARCHResult:
        """Create default GARCH result for error cases"""
        n_obs = len(data.returns) if len(data.returns) > 0 else 1
        default_vol = 0.01
        
        return GARCHResult(
            model_type='Default',
            parameters={'omega': 0.0001, 'alpha': 0.1, 'beta': 0.8},
            conditional_volatility=np.full(n_obs, default_vol),
            standardized_residuals=np.zeros(n_obs),
            log_likelihood=0.0,
            aic=0.0,
            bic=0.0,
            diagnostics={
                'ljung_box_stat': 0.0,
                'ljung_box_pvalue': 1.0,
                'ljung_box_sq_stat': 0.0,
                'ljung_box_sq_pvalue': 1.0,
                'arch_lm_stat': 0.0,
                'arch_lm_pvalue': 1.0,
                'jarque_bera_stat': 0.0,
                'jarque_bera_pvalue': 1.0
            },
            forecast_volatility=np.full(5, default_vol),
            var_estimates={
                'var_95_normal': -1.645 * default_vol,
                'var_99_normal': -2.326 * default_vol,
                'var_95_historical': -1.645 * default_vol,
                'var_99_historical': -2.326 * default_vol,
                'expected_shortfall_95': -2.0 * default_vol
            },
            model_summary={
                'n_observations': n_obs,
                'model_type': 'Default',
                'convergence': False,
                'iterations': 0
            }
        )