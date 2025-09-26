from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

# Import the individual analyzers
from .var_analyzer import VARAnalyzer, VARResult, FuturesTimeSeriesData
from .garch_analyzer import GARCHAnalyzer, GARCHResult
from .seasonal_arima_analyzer import SeasonalARIMAAnalyzer, SeasonalARIMAResult

@dataclass
class FuturesTimeSeriesResult:
    """Comprehensive results from futures time series analysis"""
    var_result: Optional[VARResult]
    garch_result: GARCHResult
    seasonal_arima_result: SeasonalARIMAResult
    trading_signals: Dict[str, np.ndarray]
    risk_metrics: Dict[str, float]
    model_comparison: Dict[str, Dict[str, float]]
    insights: List[str]
    recommendations: List[str]
    model_performance: Dict[str, float]
    combined_forecast: Dict[str, np.ndarray]

class FuturesTimeSeriesAnalyzer:
    """Comprehensive futures time series analyzer orchestrating VAR, GARCH, and Seasonal ARIMA"""
    
    def __init__(self, seasonal_periods: Optional[int] = None, garch_model: str = 'GARCH'):
        """
        Initialize the comprehensive analyzer
        
        Args:
            seasonal_periods: Number of periods in a season for ARIMA analysis
            garch_model: Type of GARCH model ('GARCH', 'EGARCH', 'TGARCH')
        """
        self.var_analyzer = VARAnalyzer()
        self.garch_analyzer = GARCHAnalyzer(model_type=garch_model)
        self.seasonal_arima_analyzer = SeasonalARIMAAnalyzer(seasonal_periods=seasonal_periods)
        
    def analyze(self, data: FuturesTimeSeriesData, 
               additional_series: Optional[Dict[str, np.ndarray]] = None) -> FuturesTimeSeriesResult:
        """Perform comprehensive time series analysis"""
        try:
            # Perform individual analyses
            print("Performing VAR analysis...")
            var_result = None
            if additional_series or data.additional_series:
                var_result = self.var_analyzer.analyze_var(data, additional_series)
            
            print("Performing GARCH analysis...")
            garch_result = self.garch_analyzer.analyze_garch(data)
            
            print("Performing Seasonal ARIMA analysis...")
            seasonal_arima_result = self.seasonal_arima_analyzer.analyze_seasonal_arima(data)
            
            # Compare models
            model_comparison = self._compare_models(var_result, garch_result, seasonal_arima_result)
            
            # Generate trading signals
            trading_signals = self._generate_trading_signals(
                data, var_result, garch_result, seasonal_arima_result
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                data, var_result, garch_result, seasonal_arima_result
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(
                var_result, garch_result, seasonal_arima_result, risk_metrics
            )
            recommendations = self._generate_recommendations(
                var_result, garch_result, seasonal_arima_result, risk_metrics
            )
            
            # Calculate model performance
            model_performance = self._calculate_model_performance(
                data, var_result, garch_result, seasonal_arima_result
            )
            
            # Create combined forecasts
            combined_forecast = self._create_combined_forecast(
                var_result, garch_result, seasonal_arima_result
            )
            
            return FuturesTimeSeriesResult(
                var_result=var_result,
                garch_result=garch_result,
                seasonal_arima_result=seasonal_arima_result,
                trading_signals=trading_signals,
                risk_metrics=risk_metrics,
                model_comparison=model_comparison,
                insights=insights,
                recommendations=recommendations,
                model_performance=model_performance,
                combined_forecast=combined_forecast
            )
            
        except Exception as e:
            warnings.warn(f"Comprehensive analysis failed: {str(e)}")
            return self._create_default_result(data)
    
    def _compare_models(self, var_result: Optional[VARResult], 
                       garch_result: GARCHResult, 
                       seasonal_arima_result: SeasonalARIMAResult) -> Dict[str, Dict[str, float]]:
        """Compare different models based on information criteria"""
        comparison = {
            'GARCH': {
                'AIC': garch_result.aic,
                'BIC': garch_result.bic,
                'Log_Likelihood': garch_result.log_likelihood
            },
            'Seasonal_ARIMA': {
                'AIC': seasonal_arima_result.aic,
                'BIC': seasonal_arima_result.bic,
                'Log_Likelihood': seasonal_arima_result.log_likelihood
            }
        }
        
        if var_result:
            comparison['VAR'] = {
                'AIC': var_result.aic,
                'BIC': var_result.bic,
                'Log_Likelihood': var_result.log_likelihood
            }
        
        return comparison
    
    def _generate_trading_signals(self, data: FuturesTimeSeriesData,
                                var_result: Optional[VARResult],
                                garch_result: GARCHResult,
                                seasonal_arima_result: SeasonalARIMAResult) -> Dict[str, np.ndarray]:
        """Generate trading signals based on analysis results"""
        signals = {}
        n_obs = len(data.returns)
        
        # ARIMA-based signals (trend following)
        arima_residuals = seasonal_arima_result.residuals
        if len(arima_residuals) > 0:
            # Signal based on residuals: negative residuals suggest undervaluation
            arima_threshold = np.std(arima_residuals) * 0.5
            arima_signals = np.where(arima_residuals < -arima_threshold, 1,
                                   np.where(arima_residuals > arima_threshold, -1, 0))
            signals['arima_trend'] = arima_signals
        else:
            signals['arima_trend'] = np.zeros(n_obs)
        
        # GARCH-based signals (volatility regime)
        conditional_vol = garch_result.conditional_volatility
        if len(conditional_vol) > 0:
            # High volatility = risk-off, low volatility = risk-on
            vol_threshold_high = np.percentile(conditional_vol, 75)
            vol_threshold_low = np.percentile(conditional_vol, 25)
            
            garch_signals = np.where(conditional_vol > vol_threshold_high, -1,
                                   np.where(conditional_vol < vol_threshold_low, 1, 0))
            signals['garch_volatility'] = garch_signals
        else:
            signals['garch_volatility'] = np.zeros(n_obs)
        
        # VAR-based signals (if available)
        if var_result and len(var_result.forecast) > 0:
            # Use VAR forecast for directional signals
            var_forecast = var_result.forecast.get('returns', np.array([0]))
            if len(var_forecast) > 0:
                # Simple directional signal based on forecast
                var_signal = 1 if var_forecast[0] > 0 else -1 if var_forecast[0] < 0 else 0
                signals['var_forecast'] = np.full(n_obs, var_signal)
            else:
                signals['var_forecast'] = np.zeros(n_obs)
        else:
            signals['var_forecast'] = np.zeros(n_obs)
        
        # Combined signal (ensemble)
        signal_weights = {'arima_trend': 0.4, 'garch_volatility': 0.3, 'var_forecast': 0.3}
        
        combined_signal = np.zeros(n_obs)
        for signal_name, weight in signal_weights.items():
            if signal_name in signals:
                signal_array = signals[signal_name]
                if len(signal_array) == n_obs:
                    combined_signal += weight * signal_array
        
        # Discretize combined signal
        signals['combined'] = np.where(combined_signal > 0.3, 1,
                                     np.where(combined_signal < -0.3, -1, 0))
        
        return signals
    
    def _calculate_risk_metrics(self, data: FuturesTimeSeriesData,
                              var_result: Optional[VARResult],
                              garch_result: GARCHResult,
                              seasonal_arima_result: SeasonalARIMAResult) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        risk_metrics = {}
        returns = data.returns[~np.isnan(data.returns)]
        
        if len(returns) == 0:
            return {'error': 1.0}
        
        # Basic return statistics
        risk_metrics['return_volatility'] = np.std(returns)
        risk_metrics['return_skewness'] = self._calculate_skewness(returns)
        risk_metrics['return_kurtosis'] = self._calculate_kurtosis(returns)
        risk_metrics['max_drawdown'] = self._calculate_max_drawdown(np.cumsum(returns))
        
        # GARCH-based risk metrics
        if len(garch_result.conditional_volatility) > 0:
            risk_metrics['garch_volatility_mean'] = np.mean(garch_result.conditional_volatility)
            risk_metrics['garch_volatility_max'] = np.max(garch_result.conditional_volatility)
            risk_metrics['garch_var_95'] = garch_result.var_estimates.get('var_95_normal', 0.0)
            risk_metrics['garch_var_99'] = garch_result.var_estimates.get('var_99_normal', 0.0)
            risk_metrics['garch_expected_shortfall'] = garch_result.var_estimates.get('expected_shortfall_95', 0.0)
        
        # ARIMA residual analysis
        if len(seasonal_arima_result.residuals) > 0:
            arima_residuals = seasonal_arima_result.residuals[~np.isnan(seasonal_arima_result.residuals)]
            if len(arima_residuals) > 0:
                risk_metrics['arima_residual_volatility'] = np.std(arima_residuals)
                risk_metrics['arima_residual_skewness'] = self._calculate_skewness(arima_residuals)
        
        # VAR system risk (if available)
        if var_result and len(var_result.residuals) > 0:
            var_residuals = var_result.residuals
            if var_residuals.size > 0:
                # System volatility (determinant of covariance matrix)
                if var_residuals.ndim > 1 and var_residuals.shape[1] > 1:
                    cov_matrix = np.cov(var_residuals.T)
                    risk_metrics['var_system_volatility'] = np.sqrt(np.linalg.det(cov_matrix))
                else:
                    risk_metrics['var_system_volatility'] = np.std(var_residuals.flatten())
        
        return risk_metrics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        if len(data) < 3:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis"""
        if len(data) < 4:
            return 0.0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0.0
        
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_max_drawdown(self, cumulative_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(cumulative_values) == 0:
            return 0.0
        
        peak = np.maximum.accumulate(cumulative_values)
        drawdown = (cumulative_values - peak) / (peak + 1e-8)
        return np.min(drawdown)
    
    def _generate_insights(self, var_result: Optional[VARResult],
                         garch_result: GARCHResult,
                         seasonal_arima_result: SeasonalARIMAResult,
                         risk_metrics: Dict[str, float]) -> List[str]:
        """Generate analytical insights"""
        insights = []
        
        # GARCH insights
        if 'garch_volatility_mean' in risk_metrics:
            vol_level = risk_metrics['garch_volatility_mean']
            if vol_level > 0.02:  # 2% daily volatility
                insights.append(f"High volatility regime detected (avg: {vol_level:.3f}). Consider risk management strategies.")
            elif vol_level < 0.01:  # 1% daily volatility
                insights.append(f"Low volatility regime detected (avg: {vol_level:.3f}). Potential for volatility expansion.")
        
        # Seasonal patterns
        if seasonal_arima_result.seasonal_order[3] > 1:
            insights.append(f"Seasonal patterns detected with period {seasonal_arima_result.seasonal_order[3]}. Consider seasonal trading strategies.")
        
        # VAR insights
        if var_result:
            # Check for Granger causality
            significant_causalities = []
            for var1, causalities in var_result.granger_causality.items():
                for var2, p_value in causalities.items():
                    if p_value < 0.05:
                        significant_causalities.append(f"{var2} → {var1}")
            
            if significant_causalities:
                insights.append(f"Significant Granger causalities found: {', '.join(significant_causalities[:3])}")
        
        # Risk insights
        if 'max_drawdown' in risk_metrics and risk_metrics['max_drawdown'] < -0.1:
            insights.append(f"Significant drawdown risk detected ({risk_metrics['max_drawdown']:.2%}). Implement stop-loss strategies.")
        
        if 'return_skewness' in risk_metrics:
            skew = risk_metrics['return_skewness']
            if skew < -0.5:
                insights.append("Negative skewness detected. Higher probability of large negative returns.")
            elif skew > 0.5:
                insights.append("Positive skewness detected. Higher probability of large positive returns.")
        
        return insights
    
    def _generate_recommendations(self, var_result: Optional[VARResult],
                                garch_result: GARCHResult,
                                seasonal_arima_result: SeasonalARIMAResult,
                                risk_metrics: Dict[str, float]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Risk-based recommendations
        if 'max_drawdown' in risk_metrics:
            max_dd = abs(risk_metrics['max_drawdown'])
            if max_dd > 0.15:  # 15% max drawdown
                recommendations.append("High drawdown risk: Implement strict position sizing (max 2% risk per trade)")
            elif max_dd > 0.10:  # 10% max drawdown
                recommendations.append("Moderate drawdown risk: Use 3% risk per trade with trailing stops")
            else:
                recommendations.append("Low drawdown risk: Standard position sizing (5% risk per trade) acceptable")
        
        # Volatility-based recommendations
        if 'garch_volatility_mean' in risk_metrics:
            vol = risk_metrics['garch_volatility_mean']
            if vol > 0.025:  # 2.5% daily volatility
                recommendations.append("High volatility: Use wider stops, reduce position size, consider volatility selling strategies")
            elif vol < 0.01:  # 1% daily volatility
                recommendations.append("Low volatility: Tighten stops, consider volatility buying strategies, increase position size")
        
        # Model-specific recommendations
        if seasonal_arima_result.seasonal_order[3] > 1:
            recommendations.append(f"Seasonal patterns present: Align trading with {seasonal_arima_result.seasonal_order[3]}-period cycles")
        
        if var_result and var_result.granger_causality:
            recommendations.append("Multi-asset relationships detected: Monitor correlated instruments for confirmation signals")
        
        # VaR-based recommendations
        if 'garch_var_95' in risk_metrics:
            var_95 = abs(risk_metrics['garch_var_95'])
            recommendations.append(f"Daily VaR (95%): {var_95:.2%}. Set stop-loss at 1.5x VaR level")
        
        # General recommendations
        recommendations.extend([
            "Monitor model residuals for regime changes",
            "Re-estimate models monthly or after significant market events",
            "Use ensemble forecasting combining all three models",
            "Implement dynamic hedging based on GARCH volatility forecasts"
        ])
        
        return recommendations
    
    def _calculate_model_performance(self, data: FuturesTimeSeriesData,
                                   var_result: Optional[VARResult],
                                   garch_result: GARCHResult,
                                   seasonal_arima_result: SeasonalARIMAResult) -> Dict[str, float]:
        """Calculate model performance metrics"""
        performance = {}
        
        # GARCH performance (volatility prediction accuracy)
        if len(garch_result.conditional_volatility) > 1 and len(data.returns) > 1:
            realized_vol = np.abs(data.returns[1:])  # Proxy for realized volatility
            predicted_vol = garch_result.conditional_volatility[:-1]
            
            min_length = min(len(realized_vol), len(predicted_vol))
            if min_length > 0:
                realized_vol = realized_vol[:min_length]
                predicted_vol = predicted_vol[:min_length]
                
                # Mean Squared Error
                mse = np.mean((realized_vol - predicted_vol) ** 2)
                performance['garch_volatility_mse'] = mse
                
                # Mean Absolute Error
                mae = np.mean(np.abs(realized_vol - predicted_vol))
                performance['garch_volatility_mae'] = mae
        
        # ARIMA performance (price prediction accuracy)
        if len(seasonal_arima_result.fitted_values) > 0 and len(data.prices) > 0:
            actual_prices = data.prices
            fitted_prices = seasonal_arima_result.fitted_values
            
            min_length = min(len(actual_prices), len(fitted_prices))
            if min_length > 0:
                actual_prices = actual_prices[:min_length]
                fitted_prices = fitted_prices[:min_length]
                
                # Mean Squared Error
                mse = np.mean((actual_prices - fitted_prices) ** 2)
                performance['arima_price_mse'] = mse
                
                # Mean Absolute Error
                mae = np.mean(np.abs(actual_prices - fitted_prices))
                performance['arima_price_mae'] = mae
                
                # R-squared
                ss_res = np.sum((actual_prices - fitted_prices) ** 2)
                ss_tot = np.sum((actual_prices - np.mean(actual_prices)) ** 2)
                r_squared = 1 - (ss_res / (ss_tot + 1e-8))
                performance['arima_r_squared'] = r_squared
        
        # Overall model confidence (inverse of AIC, normalized)
        aic_values = []
        if garch_result.aic != 0:
            aic_values.append(garch_result.aic)
        if seasonal_arima_result.aic != 0:
            aic_values.append(seasonal_arima_result.aic)
        if var_result and var_result.aic != 0:
            aic_values.append(var_result.aic)
        
        if aic_values:
            # Normalize AIC values and convert to confidence scores
            min_aic = min(aic_values)
            performance['overall_confidence'] = np.exp(-min_aic / 1000)  # Scaled confidence
        else:
            performance['overall_confidence'] = 0.5
        
        return performance
    
    def _create_combined_forecast(self, var_result: Optional[VARResult],
                                garch_result: GARCHResult,
                                seasonal_arima_result: SeasonalARIMAResult) -> Dict[str, np.ndarray]:
        """Create combined forecasts from all models"""
        combined_forecast = {}
        
        # Price forecast (from ARIMA)
        if len(seasonal_arima_result.forecast) > 0:
            combined_forecast['price'] = seasonal_arima_result.forecast
            combined_forecast['price_lower'] = seasonal_arima_result.forecast_intervals[0]
            combined_forecast['price_upper'] = seasonal_arima_result.forecast_intervals[1]
        
        # Volatility forecast (from GARCH)
        if len(garch_result.forecast_volatility) > 0:
            combined_forecast['volatility'] = garch_result.forecast_volatility
        
        # Multi-asset forecast (from VAR, if available)
        if var_result and var_result.forecast:
            for asset, forecast in var_result.forecast.items():
                combined_forecast[f'var_{asset}'] = forecast
        
        return combined_forecast
    
    def _create_default_result(self, data: FuturesTimeSeriesData) -> FuturesTimeSeriesResult:
        """Create default result for error cases"""
        n_obs = len(data.returns)
        
        # Create default results for each component
        default_garch = self.garch_analyzer._create_default_garch_result(data)
        default_arima = self.seasonal_arima_analyzer._create_default_seasonal_result(data)
        
        return FuturesTimeSeriesResult(
            var_result=None,
            garch_result=default_garch,
            seasonal_arima_result=default_arima,
            trading_signals={'combined': np.zeros(n_obs)},
            risk_metrics={'error': 1.0},
            model_comparison={'error': {'AIC': 0.0}},
            insights=["Analysis failed. Using default values."],
            recommendations=["Re-run analysis with more data or check data quality."],
            model_performance={'overall_confidence': 0.0},
            combined_forecast={'price': np.array([0.0])}
        )
    
    def plot_results(self, data: FuturesTimeSeriesData, result: FuturesTimeSeriesResult, 
                    save_path: Optional[str] = None) -> None:
        """Generate comprehensive visualizations"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Futures Time Series Analysis Results', fontsize=16)
        
        # Plot 1: Price series with ARIMA fit
        ax1 = axes[0, 0]
        ax1.plot(data.prices, label='Actual Prices', alpha=0.7)
        if len(result.seasonal_arima_result.fitted_values) > 0:
            ax1.plot(result.seasonal_arima_result.fitted_values, label='ARIMA Fit', alpha=0.8)
        ax1.set_title('Price Series and ARIMA Fit')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: GARCH conditional volatility
        ax2 = axes[0, 1]
        if len(result.garch_result.conditional_volatility) > 0:
            ax2.plot(result.garch_result.conditional_volatility, label='Conditional Volatility', color='red')
            ax2.axhline(y=np.mean(result.garch_result.conditional_volatility), 
                       color='red', linestyle='--', alpha=0.5, label='Mean Volatility')
        ax2.set_title('GARCH Conditional Volatility')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Seasonal decomposition
        ax3 = axes[1, 0]
        seasonal_data = result.seasonal_arima_result.seasonal_decomposition
        if 'trend' in seasonal_data and len(seasonal_data['trend']) > 0:
            ax3.plot(seasonal_data['trend'], label='Trend', alpha=0.8)
        if 'seasonal' in seasonal_data and len(seasonal_data['seasonal']) > 0:
            ax3.plot(seasonal_data['seasonal'], label='Seasonal', alpha=0.8)
        ax3.set_title('Seasonal Decomposition')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: ARIMA residuals
        ax4 = axes[1, 1]
        if len(result.seasonal_arima_result.residuals) > 0:
            ax4.plot(result.seasonal_arima_result.residuals, alpha=0.7)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.axhline(y=np.std(result.seasonal_arima_result.residuals), 
                       color='red', linestyle='--', alpha=0.5)
            ax4.axhline(y=-np.std(result.seasonal_arima_result.residuals), 
                       color='red', linestyle='--', alpha=0.5)
        ax4.set_title('ARIMA Residuals')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Trading signals
        ax5 = axes[2, 0]
        if 'combined' in result.trading_signals:
            signals = result.trading_signals['combined']
            ax5.plot(signals, label='Combined Signal', linewidth=2)
            ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax5.fill_between(range(len(signals)), signals, 0, alpha=0.3)
        ax5.set_title('Trading Signals')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Model comparison (AIC)
        ax6 = axes[2, 1]
        model_names = []
        aic_values = []
        
        for model, metrics in result.model_comparison.items():
            if 'AIC' in metrics and metrics['AIC'] != 0:
                model_names.append(model)
                aic_values.append(metrics['AIC'])
        
        if model_names and aic_values:
            bars = ax6.bar(model_names, aic_values, alpha=0.7)
            ax6.set_title('Model Comparison (AIC - Lower is Better)')
            ax6.set_ylabel('AIC')
            
            # Color bars based on performance
            if len(aic_values) > 1:
                min_aic_idx = np.argmin(aic_values)
                bars[min_aic_idx].set_color('green')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_report(self, data: FuturesTimeSeriesData, result: FuturesTimeSeriesResult) -> str:
        """Generate a comprehensive analysis report"""
        report = []
        report.append("# Futures Time Series Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append(f"- Analysis Period: {len(data.prices)} observations")
        report.append(f"- Models Used: GARCH, Seasonal ARIMA" + (", VAR" if result.var_result else ""))
        report.append(f"- Overall Model Confidence: {result.model_performance.get('overall_confidence', 0):.2%}")
        report.append("")
        
        # Risk Metrics
        report.append("## Risk Metrics")
        for metric, value in result.risk_metrics.items():
            if metric != 'error':
                if 'volatility' in metric.lower():
                    report.append(f"- {metric.replace('_', ' ').title()}: {value:.4f}")
                elif 'drawdown' in metric.lower() or 'var' in metric.lower():
                    report.append(f"- {metric.replace('_', ' ').title()}: {value:.2%}")
                else:
                    report.append(f"- {metric.replace('_', ' ').title()}: {value:.4f}")
        report.append("")
        
        # Model Analysis
        report.append("## Model Analysis")
        
        # GARCH Analysis
        report.append("### GARCH Model")
        garch = result.garch_result
        report.append(f"- Model Type: {garch.model_type}")
        report.append(f"- AIC: {garch.aic:.2f}")
        report.append(f"- BIC: {garch.bic:.2f}")
        report.append(f"- Current Volatility: {garch.conditional_volatility[-1]:.4f}" if len(garch.conditional_volatility) > 0 else "")
        report.append(f"- VaR (95%): {garch.var_estimates.get('var_95_normal', 0):.2%}")
        report.append("")
        
        # ARIMA Analysis
        report.append("### Seasonal ARIMA Model")
        arima = result.seasonal_arima_result
        report.append(f"- Model Order: ARIMA{arima.model_order}")
        report.append(f"- Seasonal Order: {arima.seasonal_order}")
        report.append(f"- AIC: {arima.aic:.2f}")
        report.append(f"- BIC: {arima.bic:.2f}")
        report.append("")
        
        # VAR Analysis (if available)
        if result.var_result:
            report.append("### VAR Model")
            var = result.var_result
            report.append(f"- Lag Order: {var.model_summary.get('lag_order', 'N/A')}")
            report.append(f"- Variables: {var.model_summary.get('n_variables', 'N/A')}")
            report.append(f"- AIC: {var.aic:.2f}")
            report.append(f"- BIC: {var.bic:.2f}")
            report.append("")
        
        # Key Insights
        report.append("## Key Insights")
        for i, insight in enumerate(result.insights, 1):
            report.append(f"{i}. {insight}")
        report.append("")
        
        # Recommendations
        report.append("## Trading Recommendations")
        for i, recommendation in enumerate(result.recommendations, 1):
            report.append(f"{i}. {recommendation}")
        report.append("")
        
        # Trading Signals Summary
        report.append("## Trading Signals Summary")
        if 'combined' in result.trading_signals:
            signals = result.trading_signals['combined']
            buy_signals = np.sum(signals == 1)
            sell_signals = np.sum(signals == -1)
            neutral_signals = np.sum(signals == 0)
            
            report.append(f"- Buy Signals: {buy_signals} ({buy_signals/len(signals):.1%})")
            report.append(f"- Sell Signals: {sell_signals} ({sell_signals/len(signals):.1%})")
            report.append(f"- Neutral Signals: {neutral_signals} ({neutral_signals/len(signals):.1%})")
        report.append("")
        
        # Methodology
        report.append("## Methodology")
        report.append("This analysis combines three complementary time series models:")
        report.append("1. **GARCH**: Models volatility clustering and provides risk metrics")
        report.append("2. **Seasonal ARIMA**: Captures trend and seasonal patterns in prices")
        report.append("3. **VAR**: Analyzes multivariate relationships (when additional series available)")
        report.append("")
        report.append("Signals are generated using ensemble methods combining all model outputs.")
        report.append("Risk metrics incorporate both unconditional and conditional measures.")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    n_obs = 252  # One year of daily data
    
    # Generate synthetic price data with trend, seasonality, and volatility clustering
    dates = pd.date_range('2023-01-01', periods=n_obs, freq='D')
    
    # Base trend
    trend = np.linspace(100, 120, n_obs)
    
    # Seasonal component
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n_obs) / 50)  # 50-day cycle
    
    # GARCH-like volatility
    returns = np.random.normal(0, 0.01, n_obs)
    volatility = np.zeros(n_obs)
    volatility[0] = 0.01
    
    for i in range(1, n_obs):
        volatility[i] = np.sqrt(0.00001 + 0.05 * returns[i-1]**2 + 0.9 * volatility[i-1]**2)
        returns[i] = np.random.normal(0, volatility[i])
    
    # Construct prices
    prices = trend + seasonal + np.cumsum(returns)
    
    # Additional series for VAR
    additional_series = {
        'market_index': np.cumsum(np.random.normal(0.0005, 0.015, n_obs)),
        'interest_rate': 0.05 + 0.02 * np.sin(2 * np.pi * np.arange(n_obs) / 252) + np.cumsum(np.random.normal(0, 0.001, n_obs))
    }
    
    # Create data structure
    data = FuturesTimeSeriesData(
        prices=prices,
        returns=returns,
        dates=dates.tolist(),
        volume=np.random.lognormal(10, 0.5, n_obs),
        additional_series=additional_series
    )
    
    # Initialize analyzer
    analyzer = FuturesTimeSeriesAnalyzer(seasonal_periods=50, garch_model='GARCH')
    
    # Perform analysis
    print("Starting comprehensive futures time series analysis...")
    result = analyzer.analyze(data)
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS COMPLETE")
    print("="*50)
    print(f"GARCH AIC: {result.garch_result.aic:.2f}")
    print(f"ARIMA AIC: {result.seasonal_arima_result.aic:.2f}")
    if result.var_result:
        print(f"VAR AIC: {result.var_result.aic:.2f}")
    
    print(f"\nOverall Confidence: {result.model_performance.get('overall_confidence', 0):.2%}")
    print(f"Max Drawdown: {result.risk_metrics.get('max_drawdown', 0):.2%}")
    
    # Generate and print report
    report = analyzer.generate_report(data, result)
    print("\n" + report)
    
    # Generate plots
    try:
        analyzer.plot_results(data, result)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\nAnalysis completed successfully!")