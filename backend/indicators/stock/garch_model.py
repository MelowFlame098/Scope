"""GARCH Volatility Modeling Module

This module provides comprehensive GARCH (Generalized Autoregressive Conditional Heteroskedasticity) 
analysis for modeling volatility in financial time series.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import logging

# Optional imports with fallbacks
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Some statistical tests will use simplified implementations.")

try:
    from arch import arch_model
    from arch.unitroot import ADF
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("Warning: ARCH library not available. Using simplified GARCH implementation.")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

@dataclass
class GARCHResult:
    """GARCH model results"""
    model_type: str
    parameters: Dict[str, float]
    conditional_volatility: np.ndarray
    standardized_residuals: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    volatility_forecast: np.ndarray
    arch_test: Dict[str, float]
    model_summary: str

class GARCHAnalyzer:
    """GARCH Volatility Analysis"""
    
    def __init__(self):
        pass
    
    def fit_garch(self, 
                  returns: np.ndarray, 
                  model_type: str = 'GARCH',
                  p: int = 1, 
                  q: int = 1) -> GARCHResult:
        """Fit GARCH model to return series"""
        
        if ARCH_AVAILABLE:
            try:
                # Remove any NaN or infinite values
                clean_returns = returns[np.isfinite(returns)]
                
                if len(clean_returns) < 50:
                    logger.warning("Insufficient data for GARCH modeling. Using simplified implementation.")
                    return self._simple_garch(returns, model_type, p, q)
                
                # Scale returns to percentage for numerical stability
                scaled_returns = clean_returns * 100
                
                # Fit GARCH model based on type
                if model_type.upper() == 'GARCH':
                    model = arch_model(scaled_returns, vol='Garch', p=p, q=q)
                elif model_type.upper() == 'EGARCH':
                    model = arch_model(scaled_returns, vol='EGARCH', p=p, q=q)
                elif model_type.upper() == 'TGARCH':
                    model = arch_model(scaled_returns, vol='GARCH', p=p, q=q, power=1.0)
                elif model_type.upper() == 'GJR-GARCH':
                    model = arch_model(scaled_returns, vol='GARCH', p=p, o=1, q=q)
                else:
                    model = arch_model(scaled_returns, vol='Garch', p=p, q=q)
                
                # Fit the model
                fitted_model = model.fit(disp='off', show_warning=False)
                
                # Extract results
                conditional_volatility = fitted_model.conditional_volatility.values / 100  # Scale back
                standardized_residuals = fitted_model.std_resid.values
                
                # Extend volatility to match original returns length
                if len(conditional_volatility) < len(returns):
                    # Pad with the first volatility value
                    padding = np.full(len(returns) - len(conditional_volatility), conditional_volatility[0])
                    conditional_volatility = np.concatenate([padding, conditional_volatility])
                
                # Forecast volatility
                forecast_horizon = min(20, len(returns) // 4)
                volatility_forecast = fitted_model.forecast(horizon=forecast_horizon).variance.iloc[-1].values
                volatility_forecast = np.sqrt(volatility_forecast) / 100  # Convert to returns scale
                
                # Parameters
                parameters = dict(fitted_model.params)
                
                # ARCH effect test
                arch_test = self._arch_lm_test(standardized_residuals)
                
                return GARCHResult(
                    model_type=model_type,
                    parameters=parameters,
                    conditional_volatility=conditional_volatility,
                    standardized_residuals=standardized_residuals,
                    log_likelihood=fitted_model.loglikelihood,
                    aic=fitted_model.aic,
                    bic=fitted_model.bic,
                    volatility_forecast=volatility_forecast,
                    arch_test=arch_test,
                    model_summary=str(fitted_model.summary())
                )
                
            except Exception as e:
                logger.warning(f"GARCH fitting failed: {e}. Using simplified implementation.")
        
        # Simplified GARCH implementation
        return self._simple_garch(returns, model_type, p, q)
    
    def _simple_garch(self, returns: np.ndarray, model_type: str, p: int, q: int) -> GARCHResult:
        """Simplified GARCH implementation"""
        
        n = len(returns)
        
        # Initialize volatility with sample standard deviation
        volatility = np.zeros(n)
        volatility[0] = np.std(returns) if n > 1 else 0.02
        
        # Simple GARCH(1,1) parameters
        omega = 0.00001  # Long-term variance
        alpha = 0.1      # ARCH coefficient
        beta = 0.85      # GARCH coefficient
        
        # Generate conditional volatility
        for t in range(1, n):
            if t == 1:
                volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[0]**2)
            else:
                volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2)
        
        # Standardized residuals
        standardized_residuals = returns / volatility
        standardized_residuals[volatility == 0] = 0
        
        # Simple forecast (persistence)
        last_return = returns[-1] if n > 0 else 0
        last_vol = volatility[-1] if n > 0 else 0.02
        
        forecast_vol = []
        current_vol = last_vol
        
        for _ in range(10):
            current_vol = np.sqrt(omega + alpha * last_return**2 + beta * current_vol**2)
            forecast_vol.append(current_vol)
            last_return = 0  # Assume zero mean for future returns
        
        volatility_forecast = np.array(forecast_vol)
        
        # Calculate log-likelihood (simplified)
        log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * volatility**2) + (returns**2 / volatility**2))
        
        # Information criteria
        n_params = 3  # omega, alpha, beta
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + np.log(n) * n_params
        
        # Simple ARCH test
        arch_test = self._simple_arch_test(standardized_residuals)
        
        return GARCHResult(
            model_type=model_type,
            parameters={'omega': omega, 'alpha[1]': alpha, 'beta[1]': beta},
            conditional_volatility=volatility,
            standardized_residuals=standardized_residuals,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            volatility_forecast=volatility_forecast,
            arch_test=arch_test,
            model_summary=f"Simplified {model_type}({p},{q}) model"
        )
    
    def _arch_lm_test(self, residuals: np.ndarray, lags: int = 5) -> Dict[str, float]:
        """ARCH LM test for heteroskedasticity"""
        
        try:
            n = len(residuals)
            if n <= lags + 1:
                return self._simple_arch_test(residuals)
            
            # Squared residuals
            squared_resid = residuals**2
            
            # Create lagged variables
            X = []
            y = []
            
            for t in range(lags, n):
                lag_values = [squared_resid[t-i-1] for i in range(lags)]
                X.append([1.0] + lag_values)  # Include intercept
                y.append(squared_resid[t])
            
            X = np.array(X)
            y = np.array(y)
            
            # Fit regression
            model = LinearRegression()
            model.fit(X, y)
            
            # Calculate R-squared
            y_pred = model.predict(X)
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # LM statistic
            lm_statistic = len(y) * r_squared
            
            # Approximate p-value (chi-squared with lags degrees of freedom)
            if SCIPY_AVAILABLE:
                p_value = 1 - stats.chi2.cdf(lm_statistic, lags)
            else:
                # Rough approximation
                p_value = np.exp(-lm_statistic / (2 * lags)) if lm_statistic > 0 else 0.5
            
            return {
                'lm_statistic': lm_statistic,
                'lm_pvalue': p_value,
                'r_squared': r_squared
            }
            
        except Exception as e:
            logger.warning(f"ARCH LM test failed: {e}")
            return self._simple_arch_test(residuals)
    
    def _simple_arch_test(self, residuals: np.ndarray) -> Dict[str, float]:
        """Simplified ARCH test"""
        
        # Simple variance ratio test
        n = len(residuals)
        if n < 10:
            return {'lm_statistic': 1.0, 'lm_pvalue': 0.5, 'r_squared': 0.1}
        
        mid = n // 2
        var1 = np.var(residuals[:mid]**2)
        var2 = np.var(residuals[mid:]**2)
        
        ratio = var1 / var2 if var2 > 0 else 1.0
        
        # Simple test statistic
        test_stat = abs(np.log(ratio))
        p_value = 0.05 if test_stat > 1.0 else 0.3
        
        return {
            'lm_statistic': test_stat,
            'lm_pvalue': p_value,
            'r_squared': min(test_stat / 10, 0.5)
        }
    
    def compare_garch_models(self, returns: np.ndarray) -> Dict[str, GARCHResult]:
        """Compare different GARCH model specifications"""
        
        models = ['GARCH', 'EGARCH', 'GJR-GARCH']
        results = {}
        
        for model_type in models:
            try:
                result = self.fit_garch(returns, model_type=model_type)
                results[model_type] = result
            except Exception as e:
                logger.warning(f"Failed to fit {model_type}: {e}")
                continue
        
        return results
    
    def plot_volatility(self, result: GARCHResult, returns: np.ndarray, title: str = "GARCH Volatility Analysis"):
        """Plot GARCH volatility results"""
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # Returns and conditional volatility
            time_index = range(len(returns))
            axes[0].plot(time_index, returns, alpha=0.7, label='Returns')
            axes[0].plot(time_index, result.conditional_volatility, color='red', label='Conditional Volatility')
            axes[0].plot(time_index, -result.conditional_volatility, color='red', alpha=0.5)
            axes[0].fill_between(time_index, -result.conditional_volatility, result.conditional_volatility, alpha=0.2, color='red')
            axes[0].set_title(f'{title} - Returns and Volatility')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Standardized residuals
            axes[1].plot(result.standardized_residuals, alpha=0.7)
            axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
            axes[1].axhline(y=2, color='red', linestyle='--', alpha=0.5)
            axes[1].axhline(y=-2, color='red', linestyle='--', alpha=0.5)
            axes[1].set_title('Standardized Residuals')
            axes[1].grid(True, alpha=0.3)
            
            # Volatility forecast
            forecast_index = range(len(returns), len(returns) + len(result.volatility_forecast))
            axes[2].plot(time_index[-50:], result.conditional_volatility[-50:], label='Historical Volatility')
            axes[2].plot(forecast_index, result.volatility_forecast, color='red', label='Volatility Forecast')
            axes[2].set_title('Volatility Forecast')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available. Cannot plot volatility analysis.")
        except Exception as e:
            print(f"Error plotting volatility analysis: {e}")
    
    def get_volatility_insights(self, result: GARCHResult) -> Dict[str, str]:
        """Generate insights from GARCH analysis"""
        
        insights = {}
        
        # Model information
        insights['model'] = f"Model: {result.model_type}, AIC: {result.aic:.2f}, BIC: {result.bic:.2f}"
        
        # Volatility statistics
        avg_vol = np.mean(result.conditional_volatility)
        max_vol = np.max(result.conditional_volatility)
        min_vol = np.min(result.conditional_volatility)
        vol_persistence = result.parameters.get('alpha[1]', 0) + result.parameters.get('beta[1]', 0)
        
        insights['volatility_stats'] = f"Avg: {avg_vol:.4f}, Max: {max_vol:.4f}, Min: {min_vol:.4f}"
        insights['persistence'] = f"Volatility Persistence: {vol_persistence:.3f} ({'High' if vol_persistence > 0.9 else 'Moderate' if vol_persistence > 0.7 else 'Low'})"
        
        # ARCH effects
        arch_pvalue = result.arch_test.get('lm_pvalue', 1.0)
        insights['arch_effects'] = f"ARCH Effects: {'Significant' if arch_pvalue < 0.05 else 'Not Significant'} (p={arch_pvalue:.3f})"
        
        # Forecast insights
        forecast_avg = np.mean(result.volatility_forecast)
        current_vol = result.conditional_volatility[-1] if len(result.conditional_volatility) > 0 else 0
        vol_direction = 'Increasing' if forecast_avg > current_vol else 'Decreasing'
        
        insights['forecast'] = f"Forecast Direction: {vol_direction}, Avg Forecast: {forecast_avg:.4f}"
        
        # Risk assessment
        if avg_vol > 0.03:
            risk_level = "High"
        elif avg_vol > 0.015:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        insights['risk_assessment'] = f"Risk Level: {risk_level} (based on average volatility)"
        
        return insights

# Example usage
if __name__ == "__main__":
    # Generate sample return data with volatility clustering
    np.random.seed(42)
    n_obs = 500
    
    # Simulate GARCH(1,1) process
    returns = np.zeros(n_obs)
    volatility = np.zeros(n_obs)
    volatility[0] = 0.02
    
    # GARCH parameters
    omega = 0.00001
    alpha = 0.1
    beta = 0.85
    
    for t in range(1, n_obs):
        volatility[t] = np.sqrt(omega + alpha * returns[t-1]**2 + beta * volatility[t-1]**2)
        returns[t] = np.random.normal(0, volatility[t])
    
    # Create analyzer
    analyzer = GARCHAnalyzer()
    
    # Fit GARCH model
    result = analyzer.fit_garch(returns, model_type='GARCH')
    
    print("=== GARCH Analysis Results ===")
    print(f"Model Type: {result.model_type}")
    print(f"Log-Likelihood: {result.log_likelihood:.2f}")
    print(f"AIC: {result.aic:.2f}")
    print(f"BIC: {result.bic:.2f}")
    print()
    
    print("=== Model Parameters ===")
    for param, value in result.parameters.items():
        print(f"{param}: {value:.6f}")
    print()
    
    print("=== ARCH Test ===")
    print(f"LM Statistic: {result.arch_test['lm_statistic']:.4f}")
    print(f"P-value: {result.arch_test['lm_pvalue']:.4f}")
    print(f"ARCH Effects: {'Significant' if result.arch_test['lm_pvalue'] < 0.05 else 'Not Significant'}")
    print()
    
    print("=== Volatility Statistics ===")
    print(f"Average Volatility: {np.mean(result.conditional_volatility):.4f}")
    print(f"Maximum Volatility: {np.max(result.conditional_volatility):.4f}")
    print(f"Minimum Volatility: {np.min(result.conditional_volatility):.4f}")
    print(f"Volatility of Volatility: {np.std(result.conditional_volatility):.4f}")
    print()
    
    print("=== Forecast ===")
    print(f"Next 5 volatility forecasts: {result.volatility_forecast[:5]}")
    print()
    
    # Get insights
    insights = analyzer.get_volatility_insights(result)
    print("=== Insights ===")
    for key, insight in insights.items():
        print(f"{key}: {insight}")
    print()
    
    # Compare models
    print("=== Model Comparison ===")
    model_results = analyzer.compare_garch_models(returns)
    for model_name, model_result in model_results.items():
        print(f"{model_name}: AIC = {model_result.aic:.2f}, BIC = {model_result.bic:.2f}")
    
    # Plot results
    analyzer.plot_volatility(result, returns)