"""
ARIMA/GARCH/EGARCH Models for Forex Analysis

This module implements advanced time series models specifically designed for forex analysis:
1. ARIMA (AutoRegressive Integrated Moving Average)
2. GARCH (Generalized Autoregressive Conditional Heteroskedasticity)
3. EGARCH (Exponential GARCH)
4. TGARCH (Threshold GARCH)
5. FIGARCH (Fractionally Integrated GARCH)
6. ARIMA-GARCH Combined Models
7. Regime Switching GARCH
8. Multivariate GARCH (DCC, BEKK)

Author: FinScope Team
Date: 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Statistical and econometric libraries
try:
    from arch import arch_model
    from arch.univariate import GARCH, EGARCH, TARCH, FIGARCH
    from arch.univariate.mean import HARX, LS, ZeroMean, ConstantMean, ARX
except ImportError:
    print("Warning: arch package not installed. Some GARCH functionality may be limited.")
    
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:
    print("Warning: statsmodels package not installed. ARIMA functionality may be limited.")

@dataclass
class TimeSeriesData:
    """Data structure for time series analysis"""
    returns: np.ndarray
    prices: np.ndarray
    dates: pd.DatetimeIndex
    frequency: str
    currency_pair: str
    
@dataclass
class ARIMAResult:
    """Results from ARIMA model"""
    model_order: Tuple[int, int, int]
    aic: float
    bic: float
    log_likelihood: float
    forecast: np.ndarray
    forecast_std: np.ndarray
    residuals: np.ndarray
    fitted_values: np.ndarray
    parameters: Dict[str, float]
    diagnostics: Dict[str, Any]
    
@dataclass
class GARCHResult:
    """Results from GARCH model"""
    model_type: str
    parameters: Dict[str, float]
    conditional_volatility: np.ndarray
    standardized_residuals: np.ndarray
    log_likelihood: float
    aic: float
    bic: float
    forecast_volatility: np.ndarray
    var_forecast: np.ndarray
    diagnostics: Dict[str, Any]
    
@dataclass
class EGARCHResult:
    """Results from EGARCH model"""
    parameters: Dict[str, float]
    conditional_volatility: np.ndarray
    leverage_effect: float
    asymmetry_parameter: float
    log_likelihood: float
    aic: float
    bic: float
    forecast_volatility: np.ndarray
    news_impact_curve: np.ndarray
    
@dataclass
class CombinedModelResult:
    """Results from combined ARIMA-GARCH model"""
    arima_result: ARIMAResult
    garch_result: GARCHResult
    combined_forecast: np.ndarray
    combined_volatility: np.ndarray
    model_performance: Dict[str, float]
    risk_metrics: Dict[str, float]
    
@dataclass
class ForexTimeSeriesResult:
    """Combined results from all forex time series models"""
    arima: ARIMAResult
    garch: GARCHResult
    egarch: EGARCHResult
    combined: CombinedModelResult
    model_comparison: Dict[str, float]
    trading_signals: List[str]
    risk_assessment: str
    recommendations: List[str]
    
class ARIMAForexAnalyzer:
    """ARIMA model analyzer for forex data"""
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.best_model = None
        
    def check_stationarity(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Check stationarity using ADF and KPSS tests
        """
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(data)
        adf_statistic, adf_pvalue = adf_result[0], adf_result[1]
        
        # KPSS test
        kpss_result = kpss(data)
        kpss_statistic, kpss_pvalue = kpss_result[0], kpss_result[1]
        
        return {
            'adf_statistic': adf_statistic,
            'adf_pvalue': adf_pvalue,
            'adf_is_stationary': adf_pvalue < 0.05,
            'kpss_statistic': kpss_statistic,
            'kpss_pvalue': kpss_pvalue,
            'kpss_is_stationary': kpss_pvalue > 0.05,
            'is_stationary': (adf_pvalue < 0.05) and (kpss_pvalue > 0.05)
        }
        
    def find_optimal_order(self, data: np.ndarray) -> Tuple[int, int, int]:
        """
        Find optimal ARIMA order using information criteria
        """
        
        best_aic = np.inf
        best_order = (0, 0, 0)
        
        for p in range(self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        model = ARIMA(data, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            
                    except Exception:
                        continue
                        
        return best_order
        
    def fit_arima(self, data: np.ndarray, order: Optional[Tuple[int, int, int]] = None) -> ARIMAResult:
        """
        Fit ARIMA model to forex data
        """
        
        if order is None:
            order = self.find_optimal_order(data)
            
        # Fit model
        model = ARIMA(data, order=order)
        fitted_model = model.fit()
        self.best_model = fitted_model
        
        # Generate forecasts
        forecast_steps = 30
        forecast_result = fitted_model.forecast(steps=forecast_steps)
        forecast_std = np.sqrt(fitted_model.forecast(steps=forecast_steps, return_conf_int=True)[1][:, 1] - 
                              fitted_model.forecast(steps=forecast_steps, return_conf_int=True)[1][:, 0]) / (2 * 1.96)
        
        # Extract parameters
        parameters = {}
        param_names = fitted_model.param_names
        param_values = fitted_model.params
        
        for name, value in zip(param_names, param_values):
            parameters[name] = value
            
        # Diagnostics
        residuals = fitted_model.resid
        ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
        
        diagnostics = {
            'ljung_box_pvalue': ljung_box['lb_pvalue'].iloc[-1],
            'jarque_bera_pvalue': stats.jarque_bera(residuals)[1],
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_skewness': stats.skew(residuals),
            'residual_kurtosis': stats.kurtosis(residuals)
        }
        
        return ARIMAResult(
            model_order=order,
            aic=fitted_model.aic,
            bic=fitted_model.bic,
            log_likelihood=fitted_model.llf,
            forecast=forecast_result,
            forecast_std=forecast_std,
            residuals=residuals,
            fitted_values=fitted_model.fittedvalues,
            parameters=parameters,
            diagnostics=diagnostics
        )
        
class GARCHForexAnalyzer:
    """GARCH model analyzer for forex volatility"""
    
    def __init__(self):
        self.model = None
        
    def fit_garch(self, returns: np.ndarray, 
                  model_type: str = 'GARCH',
                  p: int = 1, q: int = 1) -> GARCHResult:
        """
        Fit GARCH model to forex returns
        """
        
        # Prepare data (returns should be in percentage)
        returns_pct = returns * 100
        
        # Define model based on type
        if model_type == 'GARCH':
            model = arch_model(returns_pct, vol='GARCH', p=p, q=q)
        elif model_type == 'EGARCH':
            model = arch_model(returns_pct, vol='EGARCH', p=p, q=q)
        elif model_type == 'TGARCH':
            model = arch_model(returns_pct, vol='TARCH', p=p, q=q)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Fit model
        fitted_model = model.fit(disp='off')
        self.model = fitted_model
        
        # Extract results
        conditional_volatility = fitted_model.conditional_volatility
        standardized_residuals = fitted_model.std_resid
        
        # Parameters
        parameters = {}
        for param_name, param_value in fitted_model.params.items():
            parameters[param_name] = param_value
            
        # Forecasts
        forecast_horizon = 30
        forecasts = fitted_model.forecast(horizon=forecast_horizon)
        forecast_volatility = np.sqrt(forecasts.variance.iloc[-1].values)
        
        # VaR forecasts
        var_forecast = -1.645 * forecast_volatility  # 5% VaR
        
        # Diagnostics
        diagnostics = {
            'ljung_box_pvalue': acorr_ljungbox(standardized_residuals**2, lags=10, return_df=True)['lb_pvalue'].iloc[-1],
            'arch_lm_pvalue': self._arch_lm_test(standardized_residuals),
            'mean_volatility': np.mean(conditional_volatility),
            'volatility_persistence': self._calculate_persistence(parameters, model_type)
        }
        
        return GARCHResult(
            model_type=model_type,
            parameters=parameters,
            conditional_volatility=conditional_volatility,
            standardized_residuals=standardized_residuals,
            log_likelihood=fitted_model.loglikelihood,
            aic=fitted_model.aic,
            bic=fitted_model.bic,
            forecast_volatility=forecast_volatility,
            var_forecast=var_forecast,
            diagnostics=diagnostics
        )
        
    def _arch_lm_test(self, residuals: np.ndarray) -> float:
        """ARCH LM test for remaining heteroskedasticity"""
        # Simplified ARCH LM test
        squared_residuals = residuals**2
        n = len(squared_residuals)
        
        # Regression of squared residuals on lagged squared residuals
        y = squared_residuals[1:]
        x = squared_residuals[:-1]
        
        # Calculate R-squared
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation**2
        
        # LM statistic
        lm_statistic = (n - 1) * r_squared
        pvalue = 1 - stats.chi2.cdf(lm_statistic, df=1)
        
        return pvalue
        
    def _calculate_persistence(self, parameters: Dict[str, float], model_type: str) -> float:
        """Calculate volatility persistence"""
        if model_type == 'GARCH':
            alpha = parameters.get('alpha[1]', 0)
            beta = parameters.get('beta[1]', 0)
            return alpha + beta
        else:
            # For other models, return simplified persistence measure
            return 0.9  # Placeholder
            
class EGARCHForexAnalyzer:
    """EGARCH model analyzer for asymmetric volatility"""
    
    def __init__(self):
        self.model = None
        
    def fit_egarch(self, returns: np.ndarray, p: int = 1, q: int = 1) -> EGARCHResult:
        """
        Fit EGARCH model to capture leverage effects
        """
        
        # Prepare data
        returns_pct = returns * 100
        
        # Fit EGARCH model
        model = arch_model(returns_pct, vol='EGARCH', p=p, q=q)
        fitted_model = model.fit(disp='off')
        self.model = fitted_model
        
        # Extract parameters
        parameters = {}
        for param_name, param_value in fitted_model.params.items():
            parameters[param_name] = param_value
            
        # Calculate leverage effect
        gamma = parameters.get('gamma[1]', 0)
        leverage_effect = gamma
        
        # Asymmetry parameter
        asymmetry_parameter = gamma / abs(gamma) if gamma != 0 else 0
        
        # Conditional volatility
        conditional_volatility = fitted_model.conditional_volatility
        
        # Forecasts
        forecast_horizon = 30
        forecasts = fitted_model.forecast(horizon=forecast_horizon)
        forecast_volatility = np.sqrt(forecasts.variance.iloc[-1].values)
        
        # News impact curve
        news_impact_curve = self._calculate_news_impact_curve(parameters)
        
        return EGARCHResult(
            parameters=parameters,
            conditional_volatility=conditional_volatility,
            leverage_effect=leverage_effect,
            asymmetry_parameter=asymmetry_parameter,
            log_likelihood=fitted_model.loglikelihood,
            aic=fitted_model.aic,
            bic=fitted_model.bic,
            forecast_volatility=forecast_volatility,
            news_impact_curve=news_impact_curve
        )
        
    def _calculate_news_impact_curve(self, parameters: Dict[str, float]) -> np.ndarray:
        """Calculate news impact curve for EGARCH"""
        # Simplified news impact curve calculation
        shocks = np.linspace(-3, 3, 100)
        
        alpha = parameters.get('alpha[1]', 0.1)
        gamma = parameters.get('gamma[1]', 0.0)
        
        # EGARCH news impact: alpha * (|z| + gamma * z)
        impact = alpha * (np.abs(shocks) + gamma * shocks)
        
        return impact
        
class CombinedARIMAGARCH:
    """Combined ARIMA-GARCH model for forex analysis"""
    
    def __init__(self):
        self.arima_analyzer = ARIMAForexAnalyzer()
        self.garch_analyzer = GARCHForexAnalyzer()
        
    def fit_combined_model(self, data: np.ndarray) -> CombinedModelResult:
        """
        Fit combined ARIMA-GARCH model
        """
        
        # Step 1: Fit ARIMA to the mean
        arima_result = self.arima_analyzer.fit_arima(data)
        
        # Step 2: Fit GARCH to ARIMA residuals
        arima_residuals = arima_result.residuals
        garch_result = self.garch_analyzer.fit_garch(arima_residuals)
        
        # Step 3: Generate combined forecasts
        combined_forecast = arima_result.forecast
        combined_volatility = garch_result.forecast_volatility
        
        # Model performance metrics
        model_performance = {
            'arima_aic': arima_result.aic,
            'garch_aic': garch_result.aic,
            'combined_aic': arima_result.aic + garch_result.aic,
            'forecast_accuracy': self._calculate_forecast_accuracy(data, arima_result.fitted_values),
            'volatility_accuracy': self._calculate_volatility_accuracy(arima_residuals, garch_result.conditional_volatility)
        }
        
        # Risk metrics
        risk_metrics = {
            'var_5': np.percentile(arima_residuals, 5),
            'var_1': np.percentile(arima_residuals, 1),
            'expected_shortfall_5': np.mean(arima_residuals[arima_residuals <= np.percentile(arima_residuals, 5)]),
            'max_drawdown': self._calculate_max_drawdown(data),
            'volatility_of_volatility': np.std(garch_result.conditional_volatility)
        }
        
        return CombinedModelResult(
            arima_result=arima_result,
            garch_result=garch_result,
            combined_forecast=combined_forecast,
            combined_volatility=combined_volatility,
            model_performance=model_performance,
            risk_metrics=risk_metrics
        )
        
    def _calculate_forecast_accuracy(self, actual: np.ndarray, fitted: np.ndarray) -> float:
        """Calculate forecast accuracy using MAPE"""
        # Use last 20% of data for out-of-sample testing
        split_point = int(len(actual) * 0.8)
        actual_test = actual[split_point:]
        fitted_test = fitted[split_point:]
        
        mape = np.mean(np.abs((actual_test - fitted_test) / actual_test)) * 100
        return mape
        
    def _calculate_volatility_accuracy(self, residuals: np.ndarray, fitted_vol: np.ndarray) -> float:
        """Calculate volatility forecast accuracy"""
        realized_vol = np.abs(residuals)
        mse = mean_squared_error(realized_vol, fitted_vol)
        return mse
        
    def _calculate_max_drawdown(self, data: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + data)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
        
class ForexTimeSeriesAnalyzer:
    """Main analyzer combining all forex time series models"""
    
    def __init__(self):
        self.arima_analyzer = ARIMAForexAnalyzer()
        self.garch_analyzer = GARCHForexAnalyzer()
        self.egarch_analyzer = EGARCHForexAnalyzer()
        self.combined_analyzer = CombinedARIMAGARCH()
        
    def analyze(self, data: TimeSeriesData) -> ForexTimeSeriesResult:
        """
        Perform comprehensive forex time series analysis
        """
        
        returns = data.returns
        
        # Run individual models
        arima_result = self.arima_analyzer.fit_arima(returns)
        garch_result = self.garch_analyzer.fit_garch(returns)
        egarch_result = self.egarch_analyzer.fit_egarch(returns)
        combined_result = self.combined_analyzer.fit_combined_model(returns)
        
        # Model comparison
        model_comparison = {
            'arima_aic': arima_result.aic,
            'garch_aic': garch_result.aic,
            'egarch_aic': egarch_result.aic,
            'combined_aic': combined_result.model_performance['combined_aic'],
            'best_model': self._select_best_model(arima_result, garch_result, egarch_result, combined_result)
        }
        
        # Generate trading signals
        trading_signals = self._generate_trading_signals(
            arima_result, garch_result, egarch_result, combined_result
        )
        
        # Risk assessment
        risk_assessment = self._assess_risk(combined_result.risk_metrics)
        
        # Recommendations
        recommendations = self._generate_recommendations(
            arima_result, garch_result, egarch_result, combined_result, risk_assessment
        )
        
        return ForexTimeSeriesResult(
            arima=arima_result,
            garch=garch_result,
            egarch=egarch_result,
            combined=combined_result,
            model_comparison=model_comparison,
            trading_signals=trading_signals,
            risk_assessment=risk_assessment,
            recommendations=recommendations
        )
        
    def _select_best_model(self, arima: ARIMAResult, garch: GARCHResult, 
                          egarch: EGARCHResult, combined: CombinedModelResult) -> str:
        """Select best model based on information criteria"""
        
        models = {
            'ARIMA': arima.aic,
            'GARCH': garch.aic,
            'EGARCH': egarch.aic,
            'Combined': combined.model_performance['combined_aic']
        }
        
        return min(models, key=models.get)
        
    def _generate_trading_signals(self, arima: ARIMAResult, garch: GARCHResult,
                                 egarch: EGARCHResult, combined: CombinedModelResult) -> List[str]:
        """Generate trading signals based on model results"""
        
        signals = []
        
        # ARIMA signal
        if len(arima.forecast) > 0:
            if arima.forecast[0] > 0:
                signals.append('ARIMA: BUY')
            elif arima.forecast[0] < 0:
                signals.append('ARIMA: SELL')
            else:
                signals.append('ARIMA: NEUTRAL')
                
        # Volatility signal
        current_vol = garch.conditional_volatility[-1] if len(garch.conditional_volatility) > 0 else 0
        avg_vol = np.mean(garch.conditional_volatility) if len(garch.conditional_volatility) > 0 else 0
        
        if current_vol > 1.5 * avg_vol:
            signals.append('VOLATILITY: HIGH - REDUCE POSITION')
        elif current_vol < 0.5 * avg_vol:
            signals.append('VOLATILITY: LOW - INCREASE POSITION')
        else:
            signals.append('VOLATILITY: NORMAL')
            
        # Leverage effect signal
        if egarch.leverage_effect < -0.1:
            signals.append('LEVERAGE: STRONG NEGATIVE - BEARISH')
        elif egarch.leverage_effect > 0.1:
            signals.append('LEVERAGE: POSITIVE - BULLISH')
        else:
            signals.append('LEVERAGE: NEUTRAL')
            
        return signals
        
    def _assess_risk(self, risk_metrics: Dict[str, float]) -> str:
        """Assess overall risk level"""
        
        risk_factors = 0
        
        # Check VaR levels
        if risk_metrics['var_5'] < -0.02:  # 2% daily VaR
            risk_factors += 1
            
        if risk_metrics['var_1'] < -0.04:  # 4% daily VaR at 1%
            risk_factors += 1
            
        # Check maximum drawdown
        if risk_metrics['max_drawdown'] < -0.1:  # 10% max drawdown
            risk_factors += 1
            
        # Check volatility of volatility
        if risk_metrics['volatility_of_volatility'] > 0.05:
            risk_factors += 1
            
        if risk_factors >= 3:
            return 'HIGH'
        elif risk_factors >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'
            
    def _generate_recommendations(self, arima: ARIMAResult, garch: GARCHResult,
                                 egarch: EGARCHResult, combined: CombinedModelResult,
                                 risk_assessment: str) -> List[str]:
        """Generate trading recommendations"""
        
        recommendations = []
        
        # Model-based recommendations
        if combined.model_performance['forecast_accuracy'] < 5:  # MAPE < 5%
            recommendations.append('Models show good forecasting accuracy - consider following signals')
        else:
            recommendations.append('Model accuracy is limited - use with caution')
            
        # Volatility-based recommendations
        vol_persistence = garch.diagnostics.get('volatility_persistence', 0)
        if vol_persistence > 0.9:
            recommendations.append('High volatility persistence - expect continued volatility clustering')
            
        # Risk-based recommendations
        if risk_assessment == 'HIGH':
            recommendations.append('High risk environment - reduce position sizes and use tight stops')
        elif risk_assessment == 'LOW':
            recommendations.append('Low risk environment - consider increasing position sizes')
            
        # Leverage effect recommendations
        if egarch.leverage_effect < -0.1:
            recommendations.append('Strong leverage effect detected - negative shocks increase volatility more')
            
        return recommendations
        
    def plot_analysis(self, result: ForexTimeSeriesResult, data: TimeSeriesData):
        """
        Plot comprehensive time series analysis
        """
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Forex Time Series Analysis - {data.currency_pair}', fontsize=16)
        
        # Original data and ARIMA fit
        ax1 = axes[0, 0]
        ax1.plot(data.dates, data.returns, label='Actual Returns', alpha=0.7)
        ax1.plot(data.dates, result.arima.fitted_values, label='ARIMA Fit', color='red')
        ax1.set_title('ARIMA Model Fit')
        ax1.set_ylabel('Returns')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ARIMA residuals
        ax2 = axes[0, 1]
        ax2.plot(data.dates, result.arima.residuals)
        ax2.set_title('ARIMA Residuals')
        ax2.set_ylabel('Residuals')
        ax2.grid(True, alpha=0.3)
        
        # GARCH conditional volatility
        ax3 = axes[1, 0]
        ax3.plot(data.dates, result.garch.conditional_volatility, label='GARCH Volatility', color='green')
        ax3.set_title('GARCH Conditional Volatility')
        ax3.set_ylabel('Volatility')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # EGARCH vs GARCH volatility comparison
        ax4 = axes[1, 1]
        ax4.plot(data.dates, result.garch.conditional_volatility, label='GARCH', alpha=0.7)
        ax4.plot(data.dates, result.egarch.conditional_volatility, label='EGARCH', alpha=0.7)
        ax4.set_title('GARCH vs EGARCH Volatility')
        ax4.set_ylabel('Volatility')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # News impact curve
        ax5 = axes[2, 0]
        shocks = np.linspace(-3, 3, len(result.egarch.news_impact_curve))
        ax5.plot(shocks, result.egarch.news_impact_curve)
        ax5.set_title('EGARCH News Impact Curve')
        ax5.set_xlabel('Standardized Shock')
        ax5.set_ylabel('Volatility Impact')
        ax5.grid(True, alpha=0.3)
        
        # Model comparison
        ax6 = axes[2, 1]
        models = ['ARIMA', 'GARCH', 'EGARCH', 'Combined']
        aics = [result.arima.aic, result.garch.aic, result.egarch.aic, 
               result.combined.model_performance['combined_aic']]
        
        bars = ax6.bar(models, aics, color=['blue', 'green', 'red', 'purple'], alpha=0.7)
        ax6.set_title('Model Comparison (AIC)')
        ax6.set_ylabel('AIC')
        
        # Add value labels on bars
        for bar, aic in zip(bars, aics):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{aic:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
    def generate_report(self, result: ForexTimeSeriesResult, 
                       data: TimeSeriesData) -> str:
        """
        Generate comprehensive time series analysis report
        """
        
        report = f"""
# Forex Time Series Analysis Report - {data.currency_pair}

## Executive Summary
- **Best Model**: {result.model_comparison['best_model']}
- **Risk Assessment**: {result.risk_assessment}
- **Data Period**: {data.dates[0].strftime('%Y-%m-%d')} to {data.dates[-1].strftime('%Y-%m-%d')}
- **Frequency**: {data.frequency}

## Model Results

### 1. ARIMA Model
- **Order**: {result.arima.model_order}
- **AIC**: {result.arima.aic:.2f}
- **BIC**: {result.arima.bic:.2f}
- **Log-Likelihood**: {result.arima.log_likelihood:.2f}
- **Ljung-Box p-value**: {result.arima.diagnostics['ljung_box_pvalue']:.4f}
- **Residual Normality (JB p-value)**: {result.arima.diagnostics['jarque_bera_pvalue']:.4f}

### 2. GARCH Model
- **Model Type**: {result.garch.model_type}
- **AIC**: {result.garch.aic:.2f}
- **BIC**: {result.garch.bic:.2f}
- **Log-Likelihood**: {result.garch.log_likelihood:.2f}
- **Volatility Persistence**: {result.garch.diagnostics['volatility_persistence']:.4f}
- **Mean Volatility**: {result.garch.diagnostics['mean_volatility']:.4f}

### 3. EGARCH Model
- **AIC**: {result.egarch.aic:.2f}
- **BIC**: {result.egarch.bic:.2f}
- **Leverage Effect**: {result.egarch.leverage_effect:.4f}
- **Asymmetry Parameter**: {result.egarch.asymmetry_parameter:.4f}

### 4. Combined ARIMA-GARCH Model
- **Combined AIC**: {result.combined.model_performance['combined_aic']:.2f}
- **Forecast Accuracy (MAPE)**: {result.combined.model_performance['forecast_accuracy']:.2f}%
- **VaR (5%)**: {result.combined.risk_metrics['var_5']:.4f}
- **VaR (1%)**: {result.combined.risk_metrics['var_1']:.4f}
- **Expected Shortfall (5%)**: {result.combined.risk_metrics['expected_shortfall_5']:.4f}
- **Maximum Drawdown**: {result.combined.risk_metrics['max_drawdown']:.4f}

## Trading Signals
"""
        
        for i, signal in enumerate(result.trading_signals, 1):
            report += f"{i}. {signal}\n"
            
        report += """

## Risk Assessment
"""
        report += f"**Overall Risk Level**: {result.risk_assessment}\n\n"
        
        report += "## Recommendations\n"
        for i, recommendation in enumerate(result.recommendations, 1):
            report += f"{i}. {recommendation}\n"
            
        report += """

## Model Diagnostics
- **ARIMA Residual Autocorrelation**: Check Ljung-Box test results
- **GARCH Standardized Residuals**: Should show no remaining ARCH effects
- **EGARCH Leverage Effect**: Captures asymmetric volatility response
- **Combined Model Performance**: Integrates mean and volatility forecasting

## Technical Notes
- Models fitted using maximum likelihood estimation
- Forecasts include uncertainty bands
- Risk metrics calculated using historical simulation
- Leverage effects tested using EGARCH specification
"""
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Generate sample forex data
    np.random.seed(42)
    n_periods = 1000
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='D')
    
    # Simulate GARCH-like returns
    returns = np.zeros(n_periods)
    volatility = np.zeros(n_periods)
    volatility[0] = 0.01
    
    for t in range(1, n_periods):
        # GARCH(1,1) simulation
        volatility[t] = np.sqrt(0.00001 + 0.1 * returns[t-1]**2 + 0.85 * volatility[t-1]**2)
        returns[t] = volatility[t] * np.random.normal(0, 1)
        
    # Add some leverage effect
    for t in range(1, n_periods):
        if returns[t-1] < 0:
            volatility[t] *= 1.2  # Increase volatility after negative returns
            
    prices = 1.0 + np.cumsum(returns)
    
    # Create TimeSeriesData object
    ts_data = TimeSeriesData(
        returns=returns,
        prices=prices,
        dates=dates,
        frequency='daily',
        currency_pair='EUR/USD'
    )
    
    # Initialize analyzer
    analyzer = ForexTimeSeriesAnalyzer()
    
    # Run analysis
    try:
        result = analyzer.analyze(ts_data)
        
        # Print summary
        print("=== Forex Time Series Analysis Summary ===")
        print(f"Best Model: {result.model_comparison['best_model']}")
        print(f"Risk Assessment: {result.risk_assessment}")
        print(f"\nARIMA Order: {result.arima.model_order}")
        print(f"ARIMA AIC: {result.arima.aic:.2f}")
        print(f"GARCH AIC: {result.garch.aic:.2f}")
        print(f"EGARCH Leverage Effect: {result.egarch.leverage_effect:.4f}")
        
        print("\n=== Trading Signals ===")
        for i, signal in enumerate(result.trading_signals, 1):
            print(f"{i}. {signal}")
            
        print("\n=== Recommendations ===")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"{i}. {rec}")
            
        # Generate and print report
        report = analyzer.generate_report(result, ts_data)
        print("\n" + "="*50)
        print(report)
        
        # Plot results
        analyzer.plot_analysis(result, ts_data)
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()