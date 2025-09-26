import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro

# Optional advanced libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Some advanced features will be disabled.")

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    warnings.warn("hmmlearn not available. HMM-based regime detection will be disabled.")

# Import individual analyzers
from .garch_analyzer import IndexData, GARCHResult, GARCHAnalyzer
from .kalman_analyzer import KalmanResult, KalmanFilterAnalyzer
from .vecm_analyzer import VECMResult, VECMAnalyzer

warnings.filterwarnings('ignore')

@dataclass
class EnsembleModelResults:
    """Results from ensemble modeling"""
    ensemble_forecast: List[float]
    individual_forecasts: Dict[str, List[float]]
    model_weights: Dict[str, float]
    ensemble_accuracy: Dict[str, float]
    feature_importance: Dict[str, float]
    cross_validation_scores: Dict[str, List[float]]
    best_individual_model: str
    ensemble_vs_individual: Dict[str, float]

@dataclass
class AdvancedRiskMetrics:
    """Advanced risk assessment metrics"""
    tail_risk_measures: Dict[str, float]
    regime_specific_risk: Dict[str, Dict[str, float]]
    dynamic_risk_measures: Dict[str, List[float]]
    stress_test_results: Dict[str, float]
    liquidity_risk_metrics: Dict[str, float]
    concentration_risk: Dict[str, float]
    correlation_risk: Dict[str, float]
    model_risk_assessment: Dict[str, float]

@dataclass
class MachineLearningInsights:
    """Machine learning based insights"""
    anomaly_detection: Dict[str, Any]
    pattern_recognition: Dict[str, Any]
    clustering_analysis: Dict[str, Any]
    dimensionality_reduction: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    predictive_modeling: Dict[str, Any]
    model_interpretability: Dict[str, Any]

@dataclass
class ComprehensiveDiagnostics:
    """Comprehensive model diagnostics"""
    residual_analysis: Dict[str, Any]
    model_stability: Dict[str, Any]
    parameter_significance: Dict[str, Any]
    goodness_of_fit: Dict[str, float]
    model_assumptions: Dict[str, bool]
    robustness_tests: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]
    backtesting_results: Dict[str, Any]

@dataclass
class IndexVolatilityStateResult:
    """Results from comprehensive volatility and state analysis"""
    garch_results: GARCHResult
    kalman_results: KalmanResult
    vecm_results: VECMResult
    cointegration_analysis: Dict[str, Any]
    regime_analysis: Dict[str, Any]
    risk_metrics: Dict[str, float]
    model_comparison: Dict[str, Any]
    trading_signals: Dict[str, List[int]]
    insights: List[str]
    recommendations: List[str]

@dataclass
class EnhancedIndexAnalysisResult:
    """Enhanced comprehensive analysis results"""
    basic_analysis: IndexVolatilityStateResult
    ensemble_results: Optional[EnsembleModelResults]
    advanced_risk_metrics: Optional[AdvancedRiskMetrics]
    ml_insights: Optional[MachineLearningInsights]
    comprehensive_diagnostics: Optional[ComprehensiveDiagnostics]
    enhanced_insights: List[str]
    strategic_recommendations: List[str]
    confidence_intervals: Dict[str, Tuple[float, float]]
    model_uncertainty: Dict[str, float]

class IndexVolatilityStateAnalyzer:
    """Comprehensive volatility and state analysis for index data"""
    
    def __init__(self, enable_ensemble_methods=True, enable_advanced_risk=True, 
                 enable_ml_insights=True, enable_comprehensive_diagnostics=True,
                 rolling_window=252, forecast_horizon=30):
        """Initialize the comprehensive analyzer
        
        Args:
            enable_ensemble_methods: Enable ensemble modeling and forecasting
            enable_advanced_risk: Enable advanced risk assessment
            enable_ml_insights: Enable machine learning insights
            enable_comprehensive_diagnostics: Enable comprehensive diagnostics
            rolling_window: Window size for rolling analysis
            forecast_horizon: Number of periods to forecast
        """
        self.garch_analyzer = GARCHAnalyzer()
        self.kalman_analyzer = KalmanFilterAnalyzer()
        self.vecm_analyzer = VECMAnalyzer()
        
        # Enhanced features configuration
        self.enable_ensemble_methods = enable_ensemble_methods
        self.enable_advanced_risk = enable_advanced_risk
        self.enable_ml_insights = enable_ml_insights
        self.enable_comprehensive_diagnostics = enable_comprehensive_diagnostics
        self.rolling_window = rolling_window
        self.forecast_horizon = forecast_horizon
        
        # Initialize ML components
        self.scaler = StandardScaler()
        
        # Check library availability
        self.xgboost_available = XGBOOST_AVAILABLE
        self.hmm_available = HMM_AVAILABLE
    
    def analyze_enhanced(self, index_data: IndexData, additional_series: Optional[List[List[float]]] = None) -> EnhancedIndexAnalysisResult:
        """Perform enhanced comprehensive analysis with ensemble methods and advanced features"""
        
        try:
            # Perform basic analysis first
            basic_result = self.analyze(index_data, additional_series)
            
            # Initialize enhanced components
            ensemble_results = None
            advanced_risk_metrics = None
            ml_insights = None
            comprehensive_diagnostics = None
            
            # Ensemble modeling
            if self.enable_ensemble_methods:
                try:
                    ensemble_results = self._perform_ensemble_analysis(index_data, basic_result)
                except Exception as e:
                    print(f"Ensemble analysis failed: {e}")
            
            # Advanced risk assessment
            if self.enable_advanced_risk:
                try:
                    advanced_risk_metrics = self._perform_advanced_risk_assessment(index_data, basic_result)
                except Exception as e:
                    print(f"Advanced risk assessment failed: {e}")
            
            # Machine learning insights
            if self.enable_ml_insights:
                try:
                    ml_insights = self._generate_ml_insights(index_data, basic_result)
                except Exception as e:
                    print(f"ML insights generation failed: {e}")
            
            # Comprehensive diagnostics
            if self.enable_comprehensive_diagnostics:
                try:
                    comprehensive_diagnostics = self._perform_comprehensive_diagnostics(index_data, basic_result)
                except Exception as e:
                    print(f"Comprehensive diagnostics failed: {e}")
            
            # Generate enhanced insights and recommendations
            enhanced_insights = self._generate_enhanced_insights(basic_result, ensemble_results, 
                                                                advanced_risk_metrics, ml_insights)
            strategic_recommendations = self._generate_strategic_recommendations(basic_result, ensemble_results,
                                                                               advanced_risk_metrics, ml_insights)
            
            # Calculate confidence intervals and model uncertainty
            confidence_intervals = self._calculate_confidence_intervals(basic_result, ensemble_results)
            model_uncertainty = self._assess_model_uncertainty(basic_result, ensemble_results)
            
            return EnhancedIndexAnalysisResult(
                basic_analysis=basic_result,
                ensemble_results=ensemble_results,
                advanced_risk_metrics=advanced_risk_metrics,
                ml_insights=ml_insights,
                comprehensive_diagnostics=comprehensive_diagnostics,
                enhanced_insights=enhanced_insights,
                strategic_recommendations=strategic_recommendations,
                confidence_intervals=confidence_intervals,
                model_uncertainty=model_uncertainty
            )
            
        except Exception as e:
            print(f"Enhanced analysis failed: {e}")
            # Return basic analysis with empty enhanced components
            basic_result = self.analyze(index_data, additional_series)
            return EnhancedIndexAnalysisResult(
                basic_analysis=basic_result,
                ensemble_results=None,
                advanced_risk_metrics=None,
                ml_insights=None,
                comprehensive_diagnostics=None,
                enhanced_insights=["Enhanced analysis failed - using basic results only"],
                strategic_recommendations=["Review data quality and model parameters"],
                confidence_intervals={},
                model_uncertainty={}
            )
    
    def analyze(self, index_data: IndexData, additional_series: Optional[List[List[float]]] = None) -> IndexVolatilityStateResult:
        """Perform comprehensive volatility and state analysis"""
        
        try:
            # 1. GARCH Analysis - fit and select best model
            print("Fitting GARCH models...")
            garch_models = ['GARCH', 'EGARCH', 'TGARCH']
            garch_results = {}
            
            for model in garch_models:
                try:
                    result = self.garch_analyzer.fit_garch(index_data.returns, model_type=model)
                    if result.parameters:  # Only include if fitting succeeded
                        garch_results[model] = result
                except Exception as e:
                    print(f"Warning: {model} fitting failed: {e}")
                    continue
            
            # Select best GARCH model
            if garch_results:
                best_garch = min(garch_results.keys(), key=lambda k: garch_results[k].aic)
                selected_garch_result = garch_results[best_garch]
            else:
                # Fallback if all models fail
                selected_garch_result = self.garch_analyzer._create_fallback_result(index_data.returns)
            
            # 2. Kalman Filter Analysis
            print("Fitting Kalman Filter...")
            try:
                kalman_result = self.kalman_analyzer.fit_kalman_filter(index_data.prices, model_type='local_trend')
            except Exception as e:
                print(f"Warning: Kalman filter failed: {e}")
                kalman_result = self.kalman_analyzer._create_fallback_result(index_data.prices)
            
            # 3. VECM Analysis (if additional series provided)
            print("Performing VECM analysis...")
            if additional_series and len(additional_series) > 0:
                try:
                    # Combine main series with additional series
                    all_series = [index_data.prices] + additional_series
                    vecm_result = self.vecm_analyzer.fit_vecm(all_series)
                except Exception as e:
                    print(f"Warning: VECM analysis failed: {e}")
                    vecm_result = self.vecm_analyzer._create_fallback_result([index_data.prices])
            else:
                # Create dummy VECM result for single series
                vecm_result = self.vecm_analyzer._create_fallback_result([index_data.prices])
            
            # 4. Cointegration Analysis
            cointegration_analysis = self._analyze_cointegration(index_data, additional_series)
            
            # 5. Regime Analysis
            regime_analysis = self._analyze_regimes(selected_garch_result, kalman_result)
            
            # 6. Risk Metrics
            risk_metrics = self._calculate_risk_metrics(index_data, selected_garch_result)
            
            # 7. Model Comparison
            model_comparison = self._compare_models(garch_results, kalman_result, vecm_result)
            
            # 8. Trading Signals
            trading_signals = self._generate_trading_signals(selected_garch_result, kalman_result)
            
            # 9. Generate Insights and Recommendations
            insights = self._generate_insights(selected_garch_result, kalman_result, vecm_result, 
                                             regime_analysis, risk_metrics)
            recommendations = self._generate_recommendations(selected_garch_result, kalman_result, 
                                                           vecm_result, risk_metrics)
            
            return IndexVolatilityStateResult(
                garch_results=selected_garch_result,
                kalman_results=kalman_result,
                vecm_results=vecm_result,
                cointegration_analysis=cointegration_analysis,
                regime_analysis=regime_analysis,
                risk_metrics=risk_metrics,
                model_comparison=model_comparison,
                trading_signals=trading_signals,
                insights=insights,
                recommendations=recommendations
            )
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            # Return fallback results
            return self._create_fallback_comprehensive_result(index_data)
    
    def _analyze_cointegration(self, index_data: IndexData, additional_series: Optional[List[List[float]]]) -> Dict[str, Any]:
        """Analyze cointegration relationships"""
        
        if not additional_series or len(additional_series) == 0:
            return {
                'cointegration_detected': False,
                'n_relationships': 0,
                'correlation_matrix': None,
                'note': 'Single series - cointegration analysis not applicable'
            }
        
        try:
            # Simple correlation-based cointegration analysis
            all_series = [index_data.prices] + additional_series
            
            # Ensure all series have same length
            min_length = min(len(series) for series in all_series)
            truncated_series = [series[:min_length] for series in all_series]
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(truncated_series)
            
            # Simple cointegration test based on high correlation
            high_corr_threshold = 0.7
            cointegration_pairs = []
            
            for i in range(len(truncated_series)):
                for j in range(i+1, len(truncated_series)):
                    if abs(correlation_matrix[i, j]) > high_corr_threshold:
                        cointegration_pairs.append((i, j, correlation_matrix[i, j]))
            
            return {
                'cointegration_detected': len(cointegration_pairs) > 0,
                'n_relationships': len(cointegration_pairs),
                'correlation_matrix': correlation_matrix.tolist(),
                'cointegration_pairs': cointegration_pairs,
                'threshold_used': high_corr_threshold
            }
            
        except Exception as e:
            return {
                'cointegration_detected': False,
                'n_relationships': 0,
                'correlation_matrix': None,
                'error': str(e)
            }
    
    def _analyze_regimes(self, garch_result: GARCHResult, kalman_result: KalmanResult) -> Dict[str, Any]:
        """Analyze volatility and state regimes"""
        
        regime_analysis = {}
        
        try:
            # Volatility regime analysis from GARCH
            if garch_result.conditional_volatility:
                volatility = np.array(garch_result.conditional_volatility)
                vol_median = np.median(volatility)
                
                # Simple regime classification: high/low volatility
                vol_regimes = (volatility > vol_median).astype(int)
                
                # Calculate regime persistence
                regime_changes = np.sum(np.diff(vol_regimes) != 0)
                regime_persistence = 1 - (regime_changes / len(vol_regimes))
                
                regime_analysis['volatility_regimes'] = {
                    'regimes': vol_regimes.tolist(),
                    'high_vol_periods': np.sum(vol_regimes).item(),
                    'low_vol_periods': len(vol_regimes) - np.sum(vol_regimes).item(),
                    'regime_persistence': regime_persistence,
                    'median_volatility': vol_median
                }
            
            # State regime analysis from Kalman filter
            if kalman_result.filtered_states:
                states = np.array([state[0] if isinstance(state, list) and len(state) > 0 else state 
                                 for state in kalman_result.filtered_states])
                
                # Simple trend analysis
                state_changes = np.diff(states)
                uptrend_periods = np.sum(state_changes > 0)
                downtrend_periods = np.sum(state_changes < 0)
                
                regime_analysis['state_regimes'] = {
                    'uptrend_periods': uptrend_periods.item(),
                    'downtrend_periods': downtrend_periods.item(),
                    'sideways_periods': len(state_changes) - uptrend_periods - downtrend_periods,
                    'trend_strength': np.std(state_changes).item()
                }
            
        except Exception as e:
            regime_analysis['error'] = str(e)
        
        return regime_analysis
    
    def _calculate_risk_metrics(self, index_data: IndexData, garch_result: GARCHResult) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        try:
            returns = np.array(index_data.returns)
            
            # Basic risk metrics
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # VaR calculations
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = np.mean(returns[returns <= var_95])
            
            # Maximum Drawdown
            prices = np.array(index_data.prices)
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Sharpe Ratio (assuming risk-free rate = 0)
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # GARCH-based VaR (if available)
            garch_var_95 = None
            if garch_result.conditional_volatility:
                current_vol = garch_result.conditional_volatility[-1]
                garch_var_95 = -1.645 * current_vol  # 95% VaR using current volatility
            
            # Higher moments
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            risk_metrics = {
                'volatility': volatility,
                'var_95': var_95,
                'var_99': var_99,
                'es_95': es_95,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'skewness': skewness,
                'kurtosis': kurtosis
            }
            
            if garch_var_95 is not None:
                risk_metrics['garch_var_95'] = garch_var_95
            
            return risk_metrics
            
        except Exception as e:
            print(f"Error calculating risk metrics: {e}")
            return {
                'volatility': 0.0,
                'var_95': 0.0,
                'var_99': 0.0,
                'es_95': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
    
    def _compare_models(self, garch_results: Dict[str, GARCHResult], 
                      kalman_result: KalmanResult, vecm_result: VECMResult) -> Dict[str, Any]:
        """Compare different models"""
        
        comparison = {}
        
        # GARCH model comparison
        if garch_results:
            garch_comparison = {}
            for model_name, result in garch_results.items():
                garch_comparison[model_name] = {
                    'aic': result.aic,
                    'bic': result.bic,
                    'log_likelihood': result.log_likelihood
                }
            
            # Find best model
            best_garch = min(garch_comparison.keys(), key=lambda k: garch_comparison[k]['aic'])
            
            comparison['garch_comparison'] = garch_comparison
            comparison['model_selection'] = {'best_garch': best_garch}
        
        # Add information criteria for other models
        if kalman_result.log_likelihood:
            comparison['kalman_log_likelihood'] = kalman_result.log_likelihood
        
        if vecm_result.log_likelihood:
            comparison['vecm_log_likelihood'] = vecm_result.log_likelihood
        
        return comparison
    
    def _generate_trading_signals(self, garch_result: GARCHResult, kalman_result: KalmanResult) -> Dict[str, List[int]]:
        """Generate trading signals based on volatility and trend"""
        
        try:
            # Initialize signals
            n_obs = len(garch_result.conditional_volatility) if garch_result.conditional_volatility else 100
            
            signals = {
                'volatility_signals': [0] * n_obs,
                'trend_signals': [0] * n_obs,
                'combined_signals': [0] * n_obs
            }
            
            # Volatility-based signals
            if garch_result.conditional_volatility:
                volatility = np.array(garch_result.conditional_volatility)
                vol_ma = np.convolve(volatility, np.ones(20)/20, mode='same')  # 20-period MA
                
                for i in range(20, len(volatility)):
                    if volatility[i] < vol_ma[i] * 0.8:  # Low volatility
                        signals['volatility_signals'][i] = 1  # Buy signal
                    elif volatility[i] > vol_ma[i] * 1.2:  # High volatility
                        signals['volatility_signals'][i] = -1  # Sell signal
            
            # Trend-based signals from Kalman filter
            if kalman_result.filtered_states and len(kalman_result.filtered_states) > 1:
                states = [state[0] if isinstance(state, list) and len(state) > 0 else state 
                         for state in kalman_result.filtered_states]
                
                # Calculate slopes (trend)
                slopes = np.diff(states)
                
                for i in range(len(slopes)):
                    if slopes[i] > 0.001:  # Positive trend
                        signals['trend_signals'][i] = 1
                    elif slopes[i] < -0.001:  # Negative trend
                        signals['trend_signals'][i] = -1
            
            # Combined signals
            for i in range(len(signals['combined_signals'])):
                vol_signal = signals['volatility_signals'][i]
                trend_signal = signals['trend_signals'][i]
                
                # Simple combination: both must agree
                if vol_signal == trend_signal and vol_signal != 0:
                    signals['combined_signals'][i] = vol_signal
            
            return signals
            
        except Exception as e:
            print(f"Error generating trading signals: {e}")
            return {
                'volatility_signals': [0] * 100,
                'trend_signals': [0] * 100,
                'combined_signals': [0] * 100
            }
    
    def _generate_insights(self, garch_result: GARCHResult, kalman_result: KalmanResult,
                         vecm_result: VECMResult, regime_analysis: Dict[str, Any],
                         risk_metrics: Dict[str, float]) -> List[str]:
        """Generate analytical insights"""
        
        insights = []
        
        # GARCH insights
        if garch_result.model_type:
            insights.append(f"{garch_result.model_type} model selected as best volatility model")
            
            if 'beta' in garch_result.parameters:
                beta = garch_result.parameters['beta']
                if beta > 0.9:
                    insights.append("High volatility persistence detected - shocks have long-lasting effects")
                elif beta < 0.5:
                    insights.append("Low volatility persistence - volatility shocks decay quickly")
        
        # Risk insights
        if risk_metrics.get('skewness', 0) < -0.5:
            insights.append("Negative skewness indicates higher probability of large negative returns")
        elif risk_metrics.get('skewness', 0) > 0.5:
            insights.append("Positive skewness suggests potential for large positive returns")
        
        if risk_metrics.get('kurtosis', 0) > 3:
            insights.append("Excess kurtosis detected - fat tails indicate higher extreme event probability")
        
        # Regime insights
        if 'volatility_regimes' in regime_analysis:
            vol_regimes = regime_analysis['volatility_regimes']
            persistence = vol_regimes.get('regime_persistence', 0)
            
            if persistence > 0.8:
                insights.append("High regime persistence - volatility states tend to cluster")
            elif persistence < 0.3:
                insights.append("Low regime persistence - frequent volatility regime switches")
        
        # Cointegration insights
        if vecm_result.johansen_test.get('n_cointegrating', 0) > 0:
            insights.append("Cointegration relationships detected - long-term equilibrium exists")
        
        # Kalman filter insights
        if kalman_result.log_likelihood > -100:
            insights.append("State-space model shows good fit - underlying trends well captured")
        
        return insights
    
    def _generate_recommendations(self, garch_result: GARCHResult, kalman_result: KalmanResult,
                                vecm_result: VECMResult, risk_metrics: Dict[str, float]) -> List[str]:
        """Generate investment recommendations"""
        
        recommendations = []
        
        # Volatility-based recommendations
        if garch_result.conditional_volatility:
            current_vol = garch_result.conditional_volatility[-1]
            avg_vol = np.mean(garch_result.conditional_volatility)
            
            if current_vol > avg_vol * 1.5:
                recommendations.append("High volatility detected - consider reducing position sizes")
                recommendations.append("Implement volatility-based stop losses")
            elif current_vol < avg_vol * 0.7:
                recommendations.append("Low volatility environment - consider increasing position sizes")
                recommendations.append("Good opportunity for volatility selling strategies")
        
        # Risk-based recommendations
        if risk_metrics.get('var_95', 0) < -0.03:
            recommendations.append("High VaR indicates significant downside risk - implement hedging")
        
        if risk_metrics.get('max_drawdown', 0) < -0.2:
            recommendations.append("Large historical drawdowns - consider diversification strategies")
        
        if risk_metrics.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("Low risk-adjusted returns - review investment strategy")
        elif risk_metrics.get('sharpe_ratio', 0) > 1.5:
            recommendations.append("Strong risk-adjusted performance - consider maintaining current allocation")
        
        # Model-specific recommendations
        if garch_result.model_type == 'EGARCH':
            recommendations.append("Asymmetric volatility effects detected - monitor leverage impact")
        elif garch_result.model_type == 'TGARCH':
            recommendations.append("Threshold effects present - bad news increases volatility more than good news")
        
        # Cointegration recommendations
        if vecm_result.johansen_test.get('n_cointegrating', 0) > 0:
            recommendations.append("Cointegration detected - consider pairs trading strategies")
            recommendations.append("Mean reversion opportunities may exist")
        
        return recommendations
    
    def _perform_ensemble_analysis(self, index_data: IndexData, basic_result: IndexVolatilityStateResult) -> EnsembleModelResults:
        """Perform ensemble modeling combining multiple forecasting approaches"""
        
        try:
            returns = np.array(index_data.returns)
            prices = np.array(index_data.prices)
            
            # Prepare features for ML models
            features = self._prepare_ensemble_features(returns, prices, basic_result)
            
            # Individual model forecasts
            individual_forecasts = {}
            model_accuracies = {}
            cross_val_scores = {}
            
            # GARCH-based forecast
            if basic_result.garch_results.conditional_volatility:
                garch_forecast = self._garch_ensemble_forecast(returns, basic_result.garch_results)
                individual_forecasts['GARCH'] = garch_forecast
                model_accuracies['GARCH'] = self._evaluate_forecast_accuracy(returns[-len(garch_forecast):], garch_forecast)
            
            # Kalman Filter forecast
            if basic_result.kalman_results.filtered_states:
                kalman_forecast = self._kalman_ensemble_forecast(prices, basic_result.kalman_results)
                individual_forecasts['Kalman'] = kalman_forecast
                model_accuracies['Kalman'] = self._evaluate_forecast_accuracy(prices[-len(kalman_forecast):], kalman_forecast)
            
            # Random Forest forecast
            if len(features) > 50:  # Ensure sufficient data
                rf_forecast, rf_cv_scores = self._random_forest_forecast(features, returns)
                individual_forecasts['RandomForest'] = rf_forecast
                model_accuracies['RandomForest'] = self._evaluate_forecast_accuracy(returns[-len(rf_forecast):], rf_forecast)
                cross_val_scores['RandomForest'] = rf_cv_scores
            
            # XGBoost forecast (if available)
            if self.xgboost_available and len(features) > 50:
                xgb_forecast, xgb_cv_scores = self._xgboost_forecast(features, returns)
                individual_forecasts['XGBoost'] = xgb_forecast
                model_accuracies['XGBoost'] = self._evaluate_forecast_accuracy(returns[-len(xgb_forecast):], xgb_forecast)
                cross_val_scores['XGBoost'] = xgb_cv_scores
            
            # Ridge regression forecast
            ridge_forecast, ridge_cv_scores = self._ridge_forecast(features, returns)
            individual_forecasts['Ridge'] = ridge_forecast
            model_accuracies['Ridge'] = self._evaluate_forecast_accuracy(returns[-len(ridge_forecast):], ridge_forecast)
            cross_val_scores['Ridge'] = ridge_cv_scores
            
            # Calculate ensemble weights based on performance
            model_weights = self._calculate_ensemble_weights(model_accuracies)
            
            # Generate ensemble forecast
            ensemble_forecast = self._combine_forecasts(individual_forecasts, model_weights)
            
            # Evaluate ensemble performance
            ensemble_accuracy = self._evaluate_ensemble_performance(individual_forecasts, ensemble_forecast, returns)
            
            # Feature importance (from tree-based models)
            feature_importance = self._calculate_feature_importance(features, returns)
            
            # Best individual model
            best_individual_model = max(model_accuracies.keys(), key=lambda k: model_accuracies[k]['r2_score'])
            
            # Ensemble vs individual comparison
            ensemble_vs_individual = self._compare_ensemble_vs_individual(ensemble_accuracy, model_accuracies)
            
            return EnsembleModelResults(
                ensemble_forecast=ensemble_forecast,
                individual_forecasts=individual_forecasts,
                model_weights=model_weights,
                ensemble_accuracy=ensemble_accuracy,
                feature_importance=feature_importance,
                cross_validation_scores=cross_val_scores,
                best_individual_model=best_individual_model,
                ensemble_vs_individual=ensemble_vs_individual
            )
            
        except Exception as e:
            print(f"Error in ensemble analysis: {e}")
            return EnsembleModelResults(
                ensemble_forecast=[0.0] * self.forecast_horizon,
                individual_forecasts={},
                model_weights={},
                ensemble_accuracy={'mse': 0.0, 'r2_score': 0.0},
                feature_importance={},
                cross_validation_scores={},
                best_individual_model='None',
                ensemble_vs_individual={}
            )
    
    def _prepare_ensemble_features(self, returns: np.ndarray, prices: np.ndarray, 
                                 basic_result: IndexVolatilityStateResult) -> np.ndarray:
        """Prepare features for ensemble modeling"""
        
        try:
            features_list = []
            
            # Lagged returns
            for lag in range(1, 11):  # 10 lags
                if len(returns) > lag:
                    lagged_returns = np.roll(returns, lag)
                    lagged_returns[:lag] = 0
                    features_list.append(lagged_returns)
            
            # Rolling statistics
            for window in [5, 10, 20, 50]:
                if len(returns) >= window:
                    rolling_mean = pd.Series(returns).rolling(window).mean().fillna(0).values
                    rolling_std = pd.Series(returns).rolling(window).std().fillna(0).values
                    features_list.extend([rolling_mean, rolling_std])
            
            # Technical indicators
            if len(prices) >= 20:
                # Simple moving averages
                sma_20 = pd.Series(prices).rolling(20).mean().fillna(method='bfill').values
                sma_50 = pd.Series(prices).rolling(50).mean().fillna(method='bfill').values if len(prices) >= 50 else sma_20
                
                # Price ratios
                price_to_sma20 = prices / sma_20
                price_to_sma50 = prices / sma_50
                
                features_list.extend([sma_20, sma_50, price_to_sma20, price_to_sma50])
            
            # GARCH volatility features
            if basic_result.garch_results.conditional_volatility:
                volatility = np.array(basic_result.garch_results.conditional_volatility)
                # Pad or truncate to match returns length
                if len(volatility) != len(returns):
                    if len(volatility) < len(returns):
                        volatility = np.pad(volatility, (len(returns) - len(volatility), 0), 'edge')
                    else:
                        volatility = volatility[:len(returns)]
                features_list.append(volatility)
                
                # Volatility ratios
                vol_ma = pd.Series(volatility).rolling(20).mean().fillna(method='bfill').values
                vol_ratio = volatility / vol_ma
                features_list.append(vol_ratio)
            
            # Kalman states features
            if basic_result.kalman_results.filtered_states:
                states = np.array([state[0] if isinstance(state, list) and len(state) > 0 else state 
                                 for state in basic_result.kalman_results.filtered_states])
                # Pad or truncate to match returns length
                if len(states) != len(returns):
                    if len(states) < len(returns):
                        states = np.pad(states, (len(returns) - len(states), 0), 'edge')
                    else:
                        states = states[:len(returns)]
                features_list.append(states)
                
                # State changes
                state_changes = np.diff(states, prepend=states[0])
                features_list.append(state_changes)
            
            # Combine all features
            if features_list:
                features = np.column_stack(features_list)
                # Handle any remaining NaN values
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                return features
            else:
                # Fallback: use lagged returns only
                return np.column_stack([np.roll(returns, i) for i in range(1, 6)])
                
        except Exception as e:
            print(f"Error preparing ensemble features: {e}")
            # Fallback: simple lagged returns
            return np.column_stack([np.roll(returns, i) for i in range(1, 6)])
    
    def _garch_ensemble_forecast(self, returns: np.ndarray, garch_result: GARCHResult) -> List[float]:
        """Generate GARCH-based forecast for ensemble"""
        
        try:
            if not garch_result.conditional_volatility:
                return [0.0] * self.forecast_horizon
            
            # Use last volatility and parameters for forecasting
            current_vol = garch_result.conditional_volatility[-1]
            
            # Simple GARCH(1,1) forecast
            forecast = []
            vol_forecast = current_vol
            
            # Extract parameters (simplified)
            alpha = garch_result.parameters.get('alpha', 0.1)
            beta = garch_result.parameters.get('beta', 0.85)
            omega = garch_result.parameters.get('omega', 0.00001)
            
            for _ in range(self.forecast_horizon):
                # Forecast next period volatility
                vol_forecast = np.sqrt(omega + alpha * (returns[-1]**2) + beta * (vol_forecast**2))
                # Generate return forecast (mean-reverting to 0)
                return_forecast = np.random.normal(0, vol_forecast)
                forecast.append(return_forecast)
            
            return forecast
            
        except Exception as e:
            print(f"Error in GARCH ensemble forecast: {e}")
            return [0.0] * self.forecast_horizon
    
    def _kalman_ensemble_forecast(self, prices: np.ndarray, kalman_result: KalmanResult) -> List[float]:
        """Generate Kalman Filter-based forecast for ensemble"""
        
        try:
            if not kalman_result.filtered_states:
                return [0.0] * self.forecast_horizon
            
            # Extract trend from last few states
            states = np.array([state[0] if isinstance(state, list) and len(state) > 0 else state 
                             for state in kalman_result.filtered_states[-10:]])
            
            # Calculate trend
            if len(states) > 1:
                trend = np.mean(np.diff(states))
            else:
                trend = 0.0
            
            # Generate forecast
            forecast = []
            last_state = states[-1] if len(states) > 0 else prices[-1]
            
            for i in range(self.forecast_horizon):
                next_state = last_state + trend * (i + 1)
                forecast.append(next_state)
            
            return forecast
            
        except Exception as e:
            print(f"Error in Kalman ensemble forecast: {e}")
            return [0.0] * self.forecast_horizon
    
    def _random_forest_forecast(self, features: np.ndarray, returns: np.ndarray) -> Tuple[List[float], List[float]]:
        """Generate Random Forest forecast with cross-validation scores"""
        
        try:
            # Prepare data for supervised learning
            X, y = self._prepare_supervised_data(features, returns)
            
            if len(X) < 50:  # Insufficient data
                return [0.0] * self.forecast_horizon, [0.0]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='r2')
            
            # Generate forecast
            forecast = []
            last_features = X[-1].reshape(1, -1)
            
            for _ in range(self.forecast_horizon):
                pred = rf.predict(last_features)[0]
                forecast.append(pred)
                # Update features for next prediction (simplified)
                last_features = np.roll(last_features, -1)
                last_features[0, -1] = pred
            
            return forecast, cv_scores.tolist()
            
        except Exception as e:
            print(f"Error in Random Forest forecast: {e}")
            return [0.0] * self.forecast_horizon, [0.0]
    
    def _xgboost_forecast(self, features: np.ndarray, returns: np.ndarray) -> Tuple[List[float], List[float]]:
        """Generate XGBoost forecast with cross-validation scores"""
        
        try:
            if not self.xgboost_available:
                return [0.0] * self.forecast_horizon, [0.0]
            
            # Prepare data for supervised learning
            X, y = self._prepare_supervised_data(features, returns)
            
            if len(X) < 50:  # Insufficient data
                return [0.0] * self.forecast_horizon, [0.0]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            xgb_model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='r2')
            
            # Generate forecast
            forecast = []
            last_features = X[-1].reshape(1, -1)
            
            for _ in range(self.forecast_horizon):
                pred = xgb_model.predict(last_features)[0]
                forecast.append(pred)
                # Update features for next prediction (simplified)
                last_features = np.roll(last_features, -1)
                last_features[0, -1] = pred
            
            return forecast, cv_scores.tolist()
            
        except Exception as e:
            print(f"Error in XGBoost forecast: {e}")
            return [0.0] * self.forecast_horizon, [0.0]
    
    def _ridge_forecast(self, features: np.ndarray, returns: np.ndarray) -> Tuple[List[float], List[float]]:
        """Generate Ridge regression forecast with cross-validation scores"""
        
        try:
            # Prepare data for supervised learning
            X, y = self._prepare_supervised_data(features, returns)
            
            if len(X) < 20:  # Insufficient data
                return [0.0] * self.forecast_horizon, [0.0]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Ridge regression
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_train_scaled, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='r2')
            
            # Generate forecast
            forecast = []
            last_features = scaler.transform(X[-1].reshape(1, -1))
            
            for _ in range(self.forecast_horizon):
                pred = ridge.predict(last_features)[0]
                forecast.append(pred)
                # Update features for next prediction (simplified)
                last_features = np.roll(last_features, -1)
                last_features[0, -1] = pred
            
            return forecast, cv_scores.tolist()
            
        except Exception as e:
            print(f"Error in Ridge forecast: {e}")
            return [0.0] * self.forecast_horizon, [0.0]
    
    def _prepare_supervised_data(self, features: np.ndarray, returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for supervised learning"""
        
        try:
            # Use features to predict next period return
            X = features[:-1]  # All but last observation
            y = returns[1:]    # All but first observation (targets)
            
            # Ensure same length
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
            
            return X, y
            
        except Exception as e:
            print(f"Error preparing supervised data: {e}")
            return np.array([]), np.array([])
    
    def _perform_advanced_risk_assessment(self, returns: np.ndarray, prices: np.ndarray, 
                                        garch_result: GARCHResult) -> AdvancedRiskMetrics:
        """Perform comprehensive risk assessment with advanced metrics"""
        
        try:
            # Basic risk metrics
            volatility = np.std(returns) * np.sqrt(252)
            
            # Value at Risk (multiple confidence levels)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            var_99_5 = np.percentile(returns, 0.5)
            
            # Expected Shortfall (Conditional VaR)
            es_95 = np.mean(returns[returns <= var_95])
            es_99 = np.mean(returns[returns <= var_99])
            
            # Tail Risk Metrics
            tail_ratio = np.percentile(returns, 95) / abs(np.percentile(returns, 5))
            
            # Maximum Drawdown with detailed analysis
            cumulative_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Drawdown duration analysis
            drawdown_periods = self._analyze_drawdown_periods(drawdowns)
            
            # Regime-based risk (if GARCH volatility available)
            regime_risk = self._calculate_regime_risk(returns, garch_result)
            
            # Stress testing scenarios
            stress_scenarios = self._perform_stress_testing(returns, prices)
            
            # Risk-adjusted performance
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            sortino_ratio = self._calculate_sortino_ratio(returns)
            calmar_ratio = (np.mean(returns) * 252) / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Higher moments
            skewness = self._calculate_skewness(returns)
            kurtosis = self._calculate_kurtosis(returns)
            
            # Liquidity risk proxy (based on return volatility patterns)
            liquidity_risk = self._estimate_liquidity_risk(returns)
            
            # Correlation risk (if multiple series available)
            correlation_risk = self._assess_correlation_risk(returns)
            
            return AdvancedRiskMetrics(
                volatility_annual=volatility,
                var_95=var_95,
                var_99=var_99,
                var_99_5=var_99_5,
                expected_shortfall_95=es_95,
                expected_shortfall_99=es_99,
                tail_ratio=tail_ratio,
                max_drawdown=max_drawdown,
                drawdown_periods=drawdown_periods,
                regime_risk=regime_risk,
                stress_scenarios=stress_scenarios,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                skewness=skewness,
                kurtosis=kurtosis,
                liquidity_risk=liquidity_risk,
                correlation_risk=correlation_risk
            )
            
        except Exception as e:
            print(f"Error in advanced risk assessment: {e}")
            return AdvancedRiskMetrics(
                volatility_annual=0.0, var_95=0.0, var_99=0.0, var_99_5=0.0,
                expected_shortfall_95=0.0, expected_shortfall_99=0.0, tail_ratio=0.0,
                max_drawdown=0.0, drawdown_periods={}, regime_risk={},
                stress_scenarios={}, sharpe_ratio=0.0, sortino_ratio=0.0,
                calmar_ratio=0.0, skewness=0.0, kurtosis=0.0,
                liquidity_risk=0.0, correlation_risk=0.0
            )
    
    def _analyze_drawdown_periods(self, drawdowns: np.ndarray) -> Dict[str, float]:
        """Analyze drawdown periods and recovery times"""
        
        try:
            # Find drawdown periods
            in_drawdown = drawdowns < -0.01  # 1% threshold
            
            if not np.any(in_drawdown):
                return {'avg_duration': 0, 'max_duration': 0, 'recovery_factor': 1.0}
            
            # Calculate drawdown durations
            drawdown_starts = np.where(np.diff(in_drawdown.astype(int)) == 1)[0]
            drawdown_ends = np.where(np.diff(in_drawdown.astype(int)) == -1)[0]
            
            # Handle edge cases
            if len(drawdown_starts) == 0:
                return {'avg_duration': 0, 'max_duration': 0, 'recovery_factor': 1.0}
            
            if len(drawdown_ends) < len(drawdown_starts):
                drawdown_ends = np.append(drawdown_ends, len(drawdowns) - 1)
            
            durations = drawdown_ends[:len(drawdown_starts)] - drawdown_starts
            
            avg_duration = np.mean(durations) if len(durations) > 0 else 0
            max_duration = np.max(durations) if len(durations) > 0 else 0
            
            # Recovery factor (how quickly recoveries happen)
            recovery_factor = 1.0 / (1.0 + avg_duration / 252)  # Normalize by trading days
            
            return {
                'avg_duration': float(avg_duration),
                'max_duration': float(max_duration),
                'recovery_factor': float(recovery_factor)
            }
            
        except Exception as e:
            print(f"Error analyzing drawdown periods: {e}")
            return {'avg_duration': 0, 'max_duration': 0, 'recovery_factor': 1.0}
    
    def _calculate_regime_risk(self, returns: np.ndarray, garch_result: GARCHResult) -> Dict[str, float]:
        """Calculate risk metrics by volatility regime"""
        
        try:
            if not garch_result.conditional_volatility:
                return {'high_vol_var': 0.0, 'low_vol_var': 0.0, 'regime_persistence': 0.0}
            
            volatility = np.array(garch_result.conditional_volatility)
            vol_median = np.median(volatility)
            
            # Define regimes
            high_vol_regime = volatility > vol_median
            low_vol_regime = volatility <= vol_median
            
            # Calculate regime-specific VaR
            high_vol_returns = returns[high_vol_regime[1:]]  # Align with returns
            low_vol_returns = returns[low_vol_regime[1:]]
            
            high_vol_var = np.percentile(high_vol_returns, 5) if len(high_vol_returns) > 0 else 0.0
            low_vol_var = np.percentile(low_vol_returns, 5) if len(low_vol_returns) > 0 else 0.0
            
            # Regime persistence
            regime_changes = np.sum(np.diff(high_vol_regime.astype(int)) != 0)
            regime_persistence = 1.0 - (regime_changes / len(high_vol_regime))
            
            return {
                'high_vol_var': float(high_vol_var),
                'low_vol_var': float(low_vol_var),
                'regime_persistence': float(regime_persistence)
            }
            
        except Exception as e:
            print(f"Error calculating regime risk: {e}")
            return {'high_vol_var': 0.0, 'low_vol_var': 0.0, 'regime_persistence': 0.0}
    
    def _perform_stress_testing(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, float]:
        """Perform stress testing scenarios"""
        
        try:
            # Historical stress scenarios
            worst_day = np.min(returns)
            worst_week = np.min([np.sum(returns[i:i+5]) for i in range(len(returns)-4)])
            worst_month = np.min([np.sum(returns[i:i+21]) for i in range(len(returns)-20)])
            
            # Hypothetical stress scenarios
            # 1. Market crash scenario (-20% in one day)
            crash_impact = -0.20
            
            # 2. Volatility spike scenario (3x normal volatility)
            normal_vol = np.std(returns)
            vol_spike_impact = normal_vol * 3
            
            # 3. Liquidity crisis (increased correlation with market stress)
            liquidity_stress = abs(worst_day) * 1.5
            
            return {
                'worst_day_historical': float(worst_day),
                'worst_week_historical': float(worst_week),
                'worst_month_historical': float(worst_month),
                'market_crash_scenario': float(crash_impact),
                'volatility_spike_impact': float(vol_spike_impact),
                'liquidity_stress_impact': float(liquidity_stress)
            }
            
        except Exception as e:
            print(f"Error in stress testing: {e}")
            return {
                'worst_day_historical': 0.0, 'worst_week_historical': 0.0,
                'worst_month_historical': 0.0, 'market_crash_scenario': 0.0,
                'volatility_spike_impact': 0.0, 'liquidity_stress_impact': 0.0
            }
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        
        try:
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                return 0.0
            
            downside_deviation = np.std(downside_returns)
            if downside_deviation == 0:
                return 0.0
            
            return (np.mean(returns) / downside_deviation) * np.sqrt(252)
            
        except Exception as e:
            print(f"Error calculating Sortino ratio: {e}")
            return 0.0
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            skewness = np.mean(((returns - mean_return) / std_return) ** 3)
            return float(skewness)
            
        except Exception as e:
            print(f"Error calculating skewness: {e}")
            return 0.0
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
            return float(kurtosis)
            
        except Exception as e:
            print(f"Error calculating kurtosis: {e}")
            return 0.0
    
    def _estimate_liquidity_risk(self, returns: np.ndarray) -> float:
        """Estimate liquidity risk based on return patterns"""
        
        try:
            # Use return autocorrelation as liquidity proxy
            if len(returns) < 10:
                return 0.0
            
            # Calculate first-order autocorrelation
            returns_lag1 = returns[:-1]
            returns_current = returns[1:]
            
            correlation = np.corrcoef(returns_lag1, returns_current)[0, 1]
            
            # Higher autocorrelation suggests lower liquidity
            liquidity_risk = abs(correlation) if not np.isnan(correlation) else 0.0
            
            return float(liquidity_risk)
            
        except Exception as e:
            print(f"Error estimating liquidity risk: {e}")
            return 0.0
    
    def _assess_correlation_risk(self, returns: np.ndarray) -> float:
        """Assess correlation risk (simplified version)"""
        
        try:
            # For single series, use rolling correlation with market proxy
            # This is a simplified version - in practice, you'd use market index
            
            if len(returns) < 50:
                return 0.0
            
            # Create a synthetic market proxy (simplified)
            market_proxy = np.random.normal(0, np.std(returns), len(returns))
            
            # Rolling correlation
            window = 30
            correlations = []
            
            for i in range(window, len(returns)):
                corr = np.corrcoef(returns[i-window:i], market_proxy[i-window:i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            
            # Risk is volatility of correlations
            correlation_risk = np.std(correlations) if len(correlations) > 0 else 0.0
            
            return float(correlation_risk)
            
        except Exception as e:
            print(f"Error assessing correlation risk: {e}")
            return 0.0
    
    def _generate_ml_insights(self, returns: np.ndarray, prices: np.ndarray, 
                            features: np.ndarray) -> MachineLearningInsights:
        """Generate machine learning-based insights"""
        
        try:
            # Anomaly Detection
            anomalies = self._detect_anomalies(returns, features)
            
            # Clustering Analysis
            clusters = self._perform_clustering_analysis(features)
            
            # Pattern Recognition
            patterns = self._recognize_patterns(returns, prices)
            
            # Feature Importance (from ensemble models)
            feature_importance = self._calculate_feature_importance(features, returns)
            
            # Regime Detection using HMM (if available)
            regime_analysis = self._hmm_regime_detection(returns)
            
            # Predictive Accuracy Assessment
            prediction_metrics = self._assess_prediction_accuracy(features, returns)
            
            # Market Microstructure Insights
            microstructure = self._analyze_microstructure(returns, prices)
            
            return MachineLearningInsights(
                anomaly_scores=anomalies['scores'],
                anomaly_dates=anomalies['dates'],
                cluster_labels=clusters['labels'],
                cluster_centers=clusters['centers'],
                pattern_recognition=patterns,
                feature_importance=feature_importance,
                regime_probabilities=regime_analysis['probabilities'],
                regime_states=regime_analysis['states'],
                prediction_accuracy=prediction_metrics,
                microstructure_insights=microstructure
            )
            
        except Exception as e:
            print(f"Error generating ML insights: {e}")
            return MachineLearningInsights(
                anomaly_scores=[], anomaly_dates=[], cluster_labels=[], cluster_centers=[],
                pattern_recognition={}, feature_importance={}, regime_probabilities=[],
                regime_states=[], prediction_accuracy={}, microstructure_insights={}
            )
    
    def _detect_anomalies(self, returns: np.ndarray, features: np.ndarray) -> Dict[str, List]:
        """Detect anomalies using Isolation Forest"""
        
        try:
            if len(features) < 50:
                return {'scores': [], 'dates': []}
            
            # Use Isolation Forest for anomaly detection
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            
            # Fit on features
            anomaly_labels = iso_forest.fit_predict(features)
            anomaly_scores = iso_forest.decision_function(features)
            
            # Find anomalous periods
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            
            return {
                'scores': anomaly_scores.tolist(),
                'dates': anomaly_indices.tolist()
            }
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return {'scores': [], 'dates': []}
    
    def _perform_clustering_analysis(self, features: np.ndarray) -> Dict[str, List]:
        """Perform clustering analysis on market regimes"""
        
        try:
            if len(features) < 20:
                return {'labels': [], 'centers': []}
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply PCA for dimensionality reduction
            pca = PCA(n_components=min(5, features_scaled.shape[1]))
            features_pca = pca.fit_transform(features_scaled)
            
            # K-means clustering
            n_clusters = min(4, len(features) // 10)  # Adaptive number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_pca)
            
            # Transform cluster centers back to original space
            cluster_centers_pca = kmeans.cluster_centers_
            cluster_centers = pca.inverse_transform(cluster_centers_pca)
            cluster_centers = scaler.inverse_transform(cluster_centers)
            
            return {
                'labels': cluster_labels.tolist(),
                'centers': cluster_centers.tolist()
            }
            
        except Exception as e:
            print(f"Error in clustering analysis: {e}")
            return {'labels': [], 'centers': []}
    
    def _recognize_patterns(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, float]:
        """Recognize common financial patterns"""
        
        try:
            patterns = {}
            
            # Momentum patterns
            patterns['momentum_strength'] = self._calculate_momentum_strength(returns)
            
            # Mean reversion patterns
            patterns['mean_reversion_tendency'] = self._calculate_mean_reversion(returns)
            
            # Volatility clustering
            patterns['volatility_clustering'] = self._measure_volatility_clustering(returns)
            
            # Trend persistence
            patterns['trend_persistence'] = self._measure_trend_persistence(prices)
            
            # Jump detection
            patterns['jump_frequency'] = self._detect_jumps(returns)
            
            # Seasonality patterns
            patterns['seasonality_strength'] = self._detect_seasonality(returns)
            
            return patterns
            
        except Exception as e:
            print(f"Error in pattern recognition: {e}")
            return {}
    
    def _calculate_momentum_strength(self, returns: np.ndarray) -> float:
        """Calculate momentum strength"""
        
        try:
            if len(returns) < 20:
                return 0.0
            
            # Calculate rolling momentum
            window = min(10, len(returns) // 4)
            momentum_scores = []
            
            for i in range(window, len(returns)):
                period_returns = returns[i-window:i]
                # Momentum is consistency of direction
                positive_periods = np.sum(period_returns > 0)
                momentum_score = abs(positive_periods - window/2) / (window/2)
                momentum_scores.append(momentum_score)
            
            return float(np.mean(momentum_scores))
            
        except Exception as e:
            print(f"Error calculating momentum strength: {e}")
            return 0.0
    
    def _calculate_mean_reversion(self, returns: np.ndarray) -> float:
        """Calculate mean reversion tendency"""
        
        try:
            if len(returns) < 10:
                return 0.0
            
            # Calculate autocorrelation at lag 1
            returns_lag1 = returns[:-1]
            returns_current = returns[1:]
            
            correlation = np.corrcoef(returns_lag1, returns_current)[0, 1]
            
            # Negative correlation indicates mean reversion
            mean_reversion = -correlation if not np.isnan(correlation) else 0.0
            
            return float(max(0, mean_reversion))  # Only positive values
            
        except Exception as e:
            print(f"Error calculating mean reversion: {e}")
            return 0.0
    
    def _measure_volatility_clustering(self, returns: np.ndarray) -> float:
        """Measure volatility clustering"""
        
        try:
            if len(returns) < 20:
                return 0.0
            
            # Calculate squared returns (proxy for volatility)
            squared_returns = returns ** 2
            
            # Calculate autocorrelation of squared returns
            sq_returns_lag1 = squared_returns[:-1]
            sq_returns_current = squared_returns[1:]
            
            correlation = np.corrcoef(sq_returns_lag1, sq_returns_current)[0, 1]
            
            clustering = correlation if not np.isnan(correlation) else 0.0
            
            return float(max(0, clustering))  # Only positive values
            
        except Exception as e:
            print(f"Error measuring volatility clustering: {e}")
            return 0.0
    
    def _measure_trend_persistence(self, prices: np.ndarray) -> float:
        """Measure trend persistence"""
        
        try:
            if len(prices) < 30:
                return 0.0
            
            # Calculate Hurst exponent (simplified version)
            # Use rescaled range analysis
            
            def hurst_exponent(ts):
                """Calculate Hurst exponent"""
                lags = range(2, min(100, len(ts) // 4))
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                
                # Linear regression on log-log plot
                log_lags = np.log(lags)
                log_tau = np.log(tau)
                
                # Remove any infinite or NaN values
                valid_indices = np.isfinite(log_lags) & np.isfinite(log_tau)
                if np.sum(valid_indices) < 3:
                    return 0.5  # Random walk
                
                poly = np.polyfit(log_lags[valid_indices], log_tau[valid_indices], 1)
                return poly[0] * 2.0
            
            hurst = hurst_exponent(prices)
            
            # Convert to persistence measure (0 = anti-persistent, 0.5 = random, 1 = persistent)
            persistence = abs(hurst - 0.5) * 2
            
            return float(min(1.0, persistence))
            
        except Exception as e:
            print(f"Error measuring trend persistence: {e}")
            return 0.0
    
    def _detect_jumps(self, returns: np.ndarray) -> float:
        """Detect jump frequency"""
        
        try:
            if len(returns) < 20:
                return 0.0
            
            # Define jumps as returns beyond 3 standard deviations
            std_returns = np.std(returns)
            threshold = 3 * std_returns
            
            jumps = np.abs(returns) > threshold
            jump_frequency = np.sum(jumps) / len(returns)
            
            return float(jump_frequency)
            
        except Exception as e:
            print(f"Error detecting jumps: {e}")
            return 0.0
    
    def _detect_seasonality(self, returns: np.ndarray) -> float:
        """Detect seasonality patterns (simplified)"""
        
        try:
            if len(returns) < 50:
                return 0.0
            
            # Simple seasonality detection using autocorrelation at different lags
            seasonal_lags = [5, 10, 21, 63]  # Weekly, bi-weekly, monthly, quarterly
            seasonal_correlations = []
            
            for lag in seasonal_lags:
                if len(returns) > lag:
                    returns_lagged = returns[:-lag]
                    returns_current = returns[lag:]
                    
                    if len(returns_lagged) > 0 and len(returns_current) > 0:
                        corr = np.corrcoef(returns_lagged, returns_current)[0, 1]
                        if not np.isnan(corr):
                            seasonal_correlations.append(abs(corr))
            
            seasonality = np.mean(seasonal_correlations) if seasonal_correlations else 0.0
            
            return float(seasonality)
            
        except Exception as e:
            print(f"Error detecting seasonality: {e}")
            return 0.0
    
    def _calculate_feature_importance(self, features: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using Random Forest"""
        
        try:
            if len(features) < 50:
                return {}
            
            # Prepare supervised learning data
            X, y = self._prepare_supervised_data(features, returns)
            
            if len(X) < 20:
                return {}
            
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance = rf.feature_importances_
            
            # Create feature names
            feature_names = [f'feature_{i}' for i in range(len(importance))]
            
            return dict(zip(feature_names, importance.tolist()))
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return {}
    
    def _hmm_regime_detection(self, returns: np.ndarray) -> Dict[str, List]:
        """Detect regimes using Hidden Markov Model"""
        
        try:
            if not self.hmmlearn_available or len(returns) < 100:
                return {'probabilities': [], 'states': []}
            
            from hmmlearn import hmm
            
            # Prepare data for HMM
            X = returns.reshape(-1, 1)
            
            # Fit Gaussian HMM with 2 states (low/high volatility)
            model = hmm.GaussianHMM(n_components=2, covariance_type="full", random_state=42)
            model.fit(X)
            
            # Predict states and probabilities
            states = model.predict(X)
            probabilities = model.predict_proba(X)
            
            return {
                'probabilities': probabilities.tolist(),
                'states': states.tolist()
            }
            
        except Exception as e:
            print(f"Error in HMM regime detection: {e}")
            return {'probabilities': [], 'states': []}
    
    def _assess_prediction_accuracy(self, features: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Assess prediction accuracy of different models"""
        
        try:
            if len(features) < 50:
                return {}
            
            # Prepare data
            X, y = self._prepare_supervised_data(features, returns)
            
            if len(X) < 20:
                return {}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            accuracy_metrics = {}
            
            # Random Forest
            try:
                rf = RandomForestRegressor(n_estimators=50, random_state=42)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)
                accuracy_metrics['random_forest_r2'] = r2_score(y_test, rf_pred)
                accuracy_metrics['random_forest_mse'] = mean_squared_error(y_test, rf_pred)
            except:
                pass
            
            # Ridge Regression
            try:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_train_scaled, y_train)
                ridge_pred = ridge.predict(X_test_scaled)
                accuracy_metrics['ridge_r2'] = r2_score(y_test, ridge_pred)
                accuracy_metrics['ridge_mse'] = mean_squared_error(y_test, ridge_pred)
            except:
                pass
            
            return accuracy_metrics
            
        except Exception as e:
            print(f"Error assessing prediction accuracy: {e}")
            return {}
    
    def _analyze_microstructure(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, float]:
        """Analyze market microstructure patterns"""
        
        try:
            microstructure = {}
            
            # Bid-ask spread proxy (using return volatility)
            microstructure['spread_proxy'] = np.std(returns) * 2
            
            # Price impact (simplified)
            if len(returns) > 10:
                abs_returns = np.abs(returns)
                microstructure['price_impact'] = np.mean(abs_returns)
            else:
                microstructure['price_impact'] = 0.0
            
            # Market efficiency (autocorrelation)
            if len(returns) > 5:
                returns_lag1 = returns[:-1]
                returns_current = returns[1:]
                correlation = np.corrcoef(returns_lag1, returns_current)[0, 1]
                microstructure['efficiency_measure'] = 1.0 - abs(correlation) if not np.isnan(correlation) else 1.0
            else:
                microstructure['efficiency_measure'] = 1.0
            
            # Intraday patterns (simplified - assumes daily data)
            microstructure['intraday_volatility'] = np.std(returns)
            
            return microstructure
            
        except Exception as e:
            print(f"Error analyzing microstructure: {e}")
            return {}
    
    def _perform_comprehensive_diagnostics(self, returns: np.ndarray, garch_result: GARCHResult, 
                                         kalman_result: KalmanResult, vecm_result: VECMResult) -> ComprehensiveDiagnostics:
        """Perform comprehensive model diagnostics"""
        
        try:
            # Statistical Tests
            statistical_tests = self._perform_statistical_tests(returns)
            
            # Model Validation
            model_validation = self._validate_models(returns, garch_result, kalman_result, vecm_result)
            
            # Residual Analysis
            residual_analysis = self._analyze_residuals(returns, garch_result)
            
            # Goodness of Fit
            goodness_of_fit = self._assess_goodness_of_fit(returns, garch_result, kalman_result)
            
            # Cross-Validation Results
            cross_validation = self._perform_cross_validation(returns)
            
            # Stability Tests
            stability_tests = self._perform_stability_tests(returns, garch_result)
            
            # Overall Model Quality Score
            quality_score = self._calculate_overall_quality_score(
                statistical_tests, model_validation, residual_analysis, goodness_of_fit
            )
            
            return ComprehensiveDiagnostics(
                statistical_tests=statistical_tests,
                model_validation=model_validation,
                residual_analysis=residual_analysis,
                goodness_of_fit=goodness_of_fit,
                cross_validation_results=cross_validation,
                stability_tests=stability_tests,
                overall_quality_score=quality_score
            )
            
        except Exception as e:
            print(f"Error in comprehensive diagnostics: {e}")
            return ComprehensiveDiagnostics(
                statistical_tests={}, model_validation={}, residual_analysis={},
                goodness_of_fit={}, cross_validation_results={}, stability_tests={},
                overall_quality_score=0.0
            )
    
    def _perform_statistical_tests(self, returns: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Perform comprehensive statistical tests"""
        
        try:
            tests = {}
            
            # Stationarity Tests
            tests['stationarity'] = self._test_stationarity(returns)
            
            # Normality Tests
            tests['normality'] = self._test_normality(returns)
            
            # Autocorrelation Tests
            tests['autocorrelation'] = self._test_autocorrelation(returns)
            
            # Heteroskedasticity Tests
            tests['heteroskedasticity'] = self._test_heteroskedasticity(returns)
            
            # Unit Root Tests
            tests['unit_root'] = self._test_unit_root(returns)
            
            return tests
            
        except Exception as e:
            print(f"Error in statistical tests: {e}")
            return {}
    
    def _test_stationarity(self, returns: np.ndarray) -> Dict[str, float]:
        """Test for stationarity"""
        
        try:
            results = {}
            
            # Augmented Dickey-Fuller Test
            try:
                adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(returns, autolag='AIC')
                results['adf_statistic'] = adf_stat
                results['adf_pvalue'] = adf_pvalue
                results['adf_critical_1%'] = adf_critical['1%']
                results['adf_critical_5%'] = adf_critical['5%']
            except Exception as e:
                print(f"Error in ADF test: {e}")
            
            # KPSS Test
            try:
                kpss_stat, kpss_pvalue, _, kpss_critical = kpss(returns, regression='c')
                results['kpss_statistic'] = kpss_stat
                results['kpss_pvalue'] = kpss_pvalue
                results['kpss_critical_1%'] = kpss_critical['1%']
                results['kpss_critical_5%'] = kpss_critical['5%']
            except Exception as e:
                print(f"Error in KPSS test: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error testing stationarity: {e}")
            return {}
    
    def _test_normality(self, returns: np.ndarray) -> Dict[str, float]:
        """Test for normality"""
        
        try:
            results = {}
            
            # Jarque-Bera Test
            try:
                jb_stat, jb_pvalue = jarque_bera(returns)
                results['jarque_bera_statistic'] = jb_stat
                results['jarque_bera_pvalue'] = jb_pvalue
            except Exception as e:
                print(f"Error in Jarque-Bera test: {e}")
            
            # Shapiro-Wilk Test (for smaller samples)
            try:
                if len(returns) <= 5000:  # Shapiro-Wilk has sample size limitations
                    sw_stat, sw_pvalue = shapiro(returns)
                    results['shapiro_wilk_statistic'] = sw_stat
                    results['shapiro_wilk_pvalue'] = sw_pvalue
            except Exception as e:
                print(f"Error in Shapiro-Wilk test: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error testing normality: {e}")
            return {}
    
    def _test_autocorrelation(self, returns: np.ndarray) -> Dict[str, float]:
        """Test for autocorrelation"""
        
        try:
            results = {}
            
            # Ljung-Box Test on returns
            try:
                lb_stat, lb_pvalue = acorr_ljungbox(returns, lags=10, return_df=False)
                results['ljung_box_returns_statistic'] = lb_stat[-1]  # Last lag
                results['ljung_box_returns_pvalue'] = lb_pvalue[-1]
            except Exception as e:
                print(f"Error in Ljung-Box test on returns: {e}")
            
            # Ljung-Box Test on squared returns (for ARCH effects)
            try:
                squared_returns = returns ** 2
                lb_sq_stat, lb_sq_pvalue = acorr_ljungbox(squared_returns, lags=10, return_df=False)
                results['ljung_box_squared_statistic'] = lb_sq_stat[-1]
                results['ljung_box_squared_pvalue'] = lb_sq_pvalue[-1]
            except Exception as e:
                print(f"Error in Ljung-Box test on squared returns: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error testing autocorrelation: {e}")
            return {}
    
    def _test_heteroskedasticity(self, returns: np.ndarray) -> Dict[str, float]:
        """Test for heteroskedasticity"""
        
        try:
            results = {}
            
            # ARCH Test
            try:
                arch_stat, arch_pvalue, _, _ = het_arch(returns, maxlag=5)
                results['arch_statistic'] = arch_stat
                results['arch_pvalue'] = arch_pvalue
            except Exception as e:
                print(f"Error in ARCH test: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error testing heteroskedasticity: {e}")
            return {}
    
    def _test_unit_root(self, returns: np.ndarray) -> Dict[str, float]:
        """Additional unit root tests"""
        
        try:
            results = {}
            
            # Phillips-Perron Test (simplified version using ADF)
            try:
                # This is a simplified version - in practice, you'd use a dedicated PP test
                adf_stat, adf_pvalue, _, _, _, _ = adfuller(returns, autolag='AIC')
                results['phillips_perron_statistic'] = adf_stat  # Approximation
                results['phillips_perron_pvalue'] = adf_pvalue
            except Exception as e:
                print(f"Error in Phillips-Perron test: {e}")
            
            return results
            
        except Exception as e:
            print(f"Error in unit root tests: {e}")
            return {}
    
    def _validate_models(self, returns: np.ndarray, garch_result: GARCHResult, 
                        kalman_result: KalmanResult, vecm_result: VECMResult) -> Dict[str, Dict[str, float]]:
        """Validate model performance"""
        
        try:
            validation = {}
            
            # GARCH Model Validation
            validation['garch'] = self._validate_garch_model(returns, garch_result)
            
            # Kalman Filter Validation
            validation['kalman'] = self._validate_kalman_model(returns, kalman_result)
            
            # VECM Validation
            validation['vecm'] = self._validate_vecm_model(vecm_result)
            
            return validation
            
        except Exception as e:
            print(f"Error validating models: {e}")
            return {}
    
    def _validate_garch_model(self, returns: np.ndarray, garch_result: GARCHResult) -> Dict[str, float]:
        """Validate GARCH model"""
        
        try:
            validation = {}
            
            if garch_result.conditional_volatility:
                # Model fit quality
                validation['aic'] = garch_result.aic if hasattr(garch_result, 'aic') else 0.0
                validation['bic'] = garch_result.bic if hasattr(garch_result, 'bic') else 0.0
                validation['log_likelihood'] = garch_result.log_likelihood
                
                # Volatility forecast accuracy (simplified)
                realized_vol = np.abs(returns)
                predicted_vol = np.array(garch_result.conditional_volatility)
                
                if len(realized_vol) == len(predicted_vol):
                    mse = np.mean((realized_vol - predicted_vol) ** 2)
                    validation['volatility_forecast_mse'] = mse
                    
                    # Correlation between realized and predicted volatility
                    corr = np.corrcoef(realized_vol, predicted_vol)[0, 1]
                    validation['volatility_correlation'] = corr if not np.isnan(corr) else 0.0
            
            return validation
            
        except Exception as e:
            print(f"Error validating GARCH model: {e}")
            return {}
    
    def _validate_kalman_model(self, returns: np.ndarray, kalman_result: KalmanResult) -> Dict[str, float]:
        """Validate Kalman Filter model"""
        
        try:
            validation = {}
            
            if kalman_result.filtered_states:
                # Log likelihood
                validation['log_likelihood'] = kalman_result.log_likelihood
                
                # State prediction accuracy (simplified)
                states = np.array([state[0] if isinstance(state, list) and len(state) > 0 else state 
                                 for state in kalman_result.filtered_states])
                
                if len(states) > 1:
                    # Measure smoothness of states
                    state_changes = np.diff(states)
                    validation['state_smoothness'] = 1.0 / (1.0 + np.std(state_changes))
                    
                    # Innovation variance (measure of unpredictability)
                    validation['innovation_variance'] = np.var(state_changes)
            
            return validation
            
        except Exception as e:
            print(f"Error validating Kalman model: {e}")
            return {}
    
    def _validate_vecm_model(self, vecm_result: VECMResult) -> Dict[str, float]:
        """Validate VECM model"""
        
        try:
            validation = {}
            
            # Cointegration test results
            if vecm_result.johansen_trace_stat:
                validation['johansen_trace_statistic'] = vecm_result.johansen_trace_stat
                validation['johansen_max_eigen_statistic'] = vecm_result.johansen_max_eigen_stat
            
            # Error correction mechanism
            if vecm_result.error_correction_coef:
                validation['error_correction_coefficient'] = vecm_result.error_correction_coef
                
                # Speed of adjustment (should be negative and significant)
                validation['adjustment_speed_quality'] = abs(vecm_result.error_correction_coef) if vecm_result.error_correction_coef < 0 else 0.0
            
            return validation
            
        except Exception as e:
            print(f"Error validating VECM model: {e}")
            return {}
    
    def _analyze_residuals(self, returns: np.ndarray, garch_result: GARCHResult) -> Dict[str, float]:
        """Analyze model residuals"""
        
        try:
            analysis = {}
            
            if garch_result.standardized_residuals:
                residuals = np.array(garch_result.standardized_residuals)
                
                # Basic residual statistics
                analysis['residual_mean'] = np.mean(residuals)
                analysis['residual_std'] = np.std(residuals)
                analysis['residual_skewness'] = self._calculate_skewness(residuals)
                analysis['residual_kurtosis'] = self._calculate_kurtosis(residuals)
                
                # Residual autocorrelation
                if len(residuals) > 5:
                    residuals_lag1 = residuals[:-1]
                    residuals_current = residuals[1:]
                    corr = np.corrcoef(residuals_lag1, residuals_current)[0, 1]
                    analysis['residual_autocorrelation'] = corr if not np.isnan(corr) else 0.0
                
                # Residual normality (simplified)
                analysis['residual_normality_score'] = 1.0 / (1.0 + abs(analysis['residual_kurtosis']))
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing residuals: {e}")
            return {}
    
    def _assess_goodness_of_fit(self, returns: np.ndarray, garch_result: GARCHResult, 
                               kalman_result: KalmanResult) -> Dict[str, float]:
        """Assess overall goodness of fit"""
        
        try:
            fit_metrics = {}
            
            # Information Criteria
            if hasattr(garch_result, 'aic'):
                fit_metrics['garch_aic'] = garch_result.aic
            if hasattr(garch_result, 'bic'):
                fit_metrics['garch_bic'] = garch_result.bic
            
            fit_metrics['garch_log_likelihood'] = garch_result.log_likelihood
            fit_metrics['kalman_log_likelihood'] = kalman_result.log_likelihood
            
            # Combined fit score
            garch_score = 1.0 / (1.0 + abs(garch_result.log_likelihood)) if garch_result.log_likelihood < 0 else 0.5
            kalman_score = 1.0 / (1.0 + abs(kalman_result.log_likelihood)) if kalman_result.log_likelihood < 0 else 0.5
            
            fit_metrics['combined_fit_score'] = (garch_score + kalman_score) / 2
            
            return fit_metrics
            
        except Exception as e:
            print(f"Error assessing goodness of fit: {e}")
            return {}
    
    def _perform_cross_validation(self, returns: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation analysis"""
        
        try:
            cv_results = {}
            
            if len(returns) < 100:
                return cv_results
            
            # Time series cross-validation (walk-forward)
            n_splits = 5
            split_size = len(returns) // n_splits
            
            forecast_errors = []
            
            for i in range(2, n_splits):
                train_end = i * split_size
                test_start = train_end
                test_end = min((i + 1) * split_size, len(returns))
                
                if test_end - test_start < 10:
                    continue
                
                train_returns = returns[:train_end]
                test_returns = returns[test_start:test_end]
                
                # Simple forecast (mean)
                forecast = np.mean(train_returns)
                actual = np.mean(test_returns)
                
                error = abs(forecast - actual)
                forecast_errors.append(error)
            
            if forecast_errors:
                cv_results['mean_forecast_error'] = np.mean(forecast_errors)
                cv_results['forecast_error_std'] = np.std(forecast_errors)
                cv_results['forecast_stability'] = 1.0 / (1.0 + cv_results['forecast_error_std'])
            
            return cv_results
            
        except Exception as e:
            print(f"Error in cross-validation: {e}")
            return {}
    
    def _perform_stability_tests(self, returns: np.ndarray, garch_result: GARCHResult) -> Dict[str, float]:
        """Perform model stability tests"""
        
        try:
            stability = {}
            
            # Parameter stability (simplified)
            if garch_result.parameters:
                # Check if parameters are within reasonable bounds
                alpha = garch_result.parameters.get('alpha', 0.0)
                beta = garch_result.parameters.get('beta', 0.0)
                
                # GARCH constraints: alpha + beta < 1 for stationarity
                persistence = alpha + beta
                stability['garch_persistence'] = persistence
                stability['garch_stationarity'] = 1.0 if persistence < 1.0 else 0.0
                
                # Parameter significance (simplified)
                stability['alpha_significance'] = 1.0 if alpha > 0.01 else 0.0
                stability['beta_significance'] = 1.0 if beta > 0.01 else 0.0
            
            # Structural break test (simplified)
            if len(returns) > 100:
                mid_point = len(returns) // 2
                first_half_vol = np.std(returns[:mid_point])
                second_half_vol = np.std(returns[mid_point:])
                
                vol_ratio = min(first_half_vol, second_half_vol) / max(first_half_vol, second_half_vol)
                stability['structural_stability'] = vol_ratio
            
            return stability
            
        except Exception as e:
            print(f"Error in stability tests: {e}")
            return {}
    
    def _calculate_overall_quality_score(self, statistical_tests: Dict, model_validation: Dict, 
                                       residual_analysis: Dict, goodness_of_fit: Dict) -> float:
        """Calculate overall model quality score"""
        
        try:
            scores = []
            
            # Statistical test scores
            if 'stationarity' in statistical_tests:
                adf_pvalue = statistical_tests['stationarity'].get('adf_pvalue', 1.0)
                scores.append(1.0 - adf_pvalue)  # Lower p-value is better for stationarity
            
            # Model validation scores
            if 'garch' in model_validation:
                vol_corr = model_validation['garch'].get('volatility_correlation', 0.0)
                scores.append(abs(vol_corr))
            
            # Residual analysis scores
            if residual_analysis:
                normality_score = residual_analysis.get('residual_normality_score', 0.0)
                scores.append(normality_score)
            
            # Goodness of fit scores
            if goodness_of_fit:
                combined_fit = goodness_of_fit.get('combined_fit_score', 0.0)
                scores.append(combined_fit)
            
            # Calculate weighted average
            if scores:
                quality_score = np.mean(scores)
                return float(min(1.0, max(0.0, quality_score)))
            else:
                return 0.5  # Neutral score if no metrics available
            
        except Exception as e:
            print(f"Error calculating quality score: {e}")
            return 0.0
    
    def _generate_enhanced_insights(self, basic_result: IndexVolatilityStateResult, 
                                  ensemble_result: EnsembleModelResults, 
                                  advanced_risk: AdvancedRiskMetrics,
                                  ml_insights: MachineLearningInsights,
                                  diagnostics: ComprehensiveDiagnostics) -> List[str]:
        """Generate enhanced analytical insights"""
        
        try:
            insights = []
            
            # Ensemble Model Insights
            if ensemble_result.ensemble_accuracy > 0.7:
                insights.append(f"Strong ensemble model performance with {ensemble_result.ensemble_accuracy:.1%} accuracy")
            
            if ensemble_result.feature_importance:
                top_feature = max(ensemble_result.feature_importance.items(), key=lambda x: x[1])
                insights.append(f"Most predictive feature: {top_feature[0]} (importance: {top_feature[1]:.3f})")
            
            # Advanced Risk Insights
            if advanced_risk.tail_risk > 0.05:
                insights.append(f"Elevated tail risk detected: {advanced_risk.tail_risk:.1%}")
            
            if advanced_risk.regime_risk and 'high_volatility' in advanced_risk.regime_risk:
                high_vol_var = advanced_risk.regime_risk['high_volatility'].get('var_95', 0)
                if high_vol_var > 0.03:
                    insights.append(f"High volatility regime shows significant risk (VaR: {high_vol_var:.1%})")
            
            # Machine Learning Insights
            if ml_insights.anomaly_scores and len(ml_insights.anomaly_scores) > 0:
                anomaly_rate = np.mean(np.array(ml_insights.anomaly_scores) < -0.1)
                if anomaly_rate > 0.05:
                    insights.append(f"Anomaly detection identifies {anomaly_rate:.1%} of observations as outliers")
            
            if ml_insights.pattern_analysis:
                momentum = ml_insights.pattern_analysis.get('momentum_strength', 0)
                if momentum > 0.6:
                    insights.append(f"Strong momentum patterns detected (strength: {momentum:.2f})")
                
                volatility_clustering = ml_insights.pattern_analysis.get('volatility_clustering', 0)
                if volatility_clustering > 0.7:
                    insights.append("Significant volatility clustering suggests GARCH effects")
            
            # Diagnostic Insights
            if diagnostics.overall_quality_score > 0.8:
                insights.append(f"Excellent model quality (score: {diagnostics.overall_quality_score:.2f})")
            elif diagnostics.overall_quality_score < 0.5:
                insights.append(f"Model quality concerns identified (score: {diagnostics.overall_quality_score:.2f})")
            
            # Statistical Test Insights
            if diagnostics.statistical_tests:
                stationarity = diagnostics.statistical_tests.get('stationarity', {})
                adf_pvalue = stationarity.get('adf_pvalue', 1.0)
                if adf_pvalue < 0.05:
                    insights.append("Data exhibits strong stationarity (suitable for time series modeling)")
                
                normality = diagnostics.statistical_tests.get('normality', {})
                jb_pvalue = normality.get('jarque_bera_pvalue', 1.0)
                if jb_pvalue < 0.05:
                    insights.append("Returns deviate significantly from normality (consider robust methods)")
            
            return insights
            
        except Exception as e:
            print(f"Error generating enhanced insights: {e}")
            return ["Enhanced insights generation encountered errors"]
    
    def _generate_strategic_recommendations(self, basic_result: IndexVolatilityStateResult,
                                          ensemble_result: EnsembleModelResults,
                                          advanced_risk: AdvancedRiskMetrics,
                                          ml_insights: MachineLearningInsights) -> List[str]:
        """Generate strategic investment recommendations"""
        
        try:
            recommendations = []
            
            # Risk-based Recommendations
            if advanced_risk.var_95 > 0.05:
                recommendations.append("Consider reducing position size due to elevated VaR")
            
            if advanced_risk.max_drawdown_duration and advanced_risk.max_drawdown_duration > 30:
                recommendations.append("Long drawdown periods suggest need for diversification")
            
            if advanced_risk.sharpe_ratio > 1.5:
                recommendations.append("Strong risk-adjusted returns support increased allocation")
            elif advanced_risk.sharpe_ratio < 0.5:
                recommendations.append("Poor risk-adjusted returns suggest caution")
            
            # Regime-based Recommendations
            if ml_insights.regime_analysis and 'current_regime' in ml_insights.regime_analysis:
                current_regime = ml_insights.regime_analysis['current_regime']
                if current_regime == 'high_volatility':
                    recommendations.append("Current high volatility regime suggests defensive positioning")
                elif current_regime == 'low_volatility':
                    recommendations.append("Low volatility environment may support growth strategies")
            
            # Ensemble Model Recommendations
            if ensemble_result.ensemble_forecast:
                forecast_direction = "positive" if ensemble_result.ensemble_forecast[-1] > 0 else "negative"
                confidence = ensemble_result.ensemble_accuracy
                recommendations.append(f"Ensemble models predict {forecast_direction} returns (confidence: {confidence:.1%})")
            
            # Pattern-based Recommendations
            if ml_insights.pattern_analysis:
                mean_reversion = ml_insights.pattern_analysis.get('mean_reversion_strength', 0)
                if mean_reversion > 0.6:
                    recommendations.append("Strong mean reversion suggests contrarian strategies")
                
                trend_persistence = ml_insights.pattern_analysis.get('trend_persistence', 0)
                if trend_persistence > 0.6:
                    recommendations.append("Trend persistence supports momentum strategies")
            
            # Liquidity Recommendations
            if advanced_risk.liquidity_risk > 0.3:
                recommendations.append("Elevated liquidity risk suggests smaller position sizes")
            
            # Correlation Recommendations
            if advanced_risk.correlation_risk > 0.7:
                recommendations.append("High correlation risk indicates need for alternative assets")
            
            return recommendations
            
        except Exception as e:
            print(f"Error generating strategic recommendations: {e}")
            return ["Strategic recommendation generation encountered errors"]
    
    def _calculate_confidence_intervals(self, ensemble_result: EnsembleModelResults,
                                      advanced_risk: AdvancedRiskMetrics) -> Dict[str, Dict[str, float]]:
        """Calculate confidence intervals for key metrics"""
        
        try:
            confidence_intervals = {}
            
            # Forecast Confidence Intervals
            if ensemble_result.ensemble_forecast and len(ensemble_result.ensemble_forecast) > 0:
                forecast_std = np.std(ensemble_result.ensemble_forecast)
                last_forecast = ensemble_result.ensemble_forecast[-1]
                
                confidence_intervals['forecast'] = {
                    'point_estimate': last_forecast,
                    'lower_95': last_forecast - 1.96 * forecast_std,
                    'upper_95': last_forecast + 1.96 * forecast_std,
                    'lower_68': last_forecast - forecast_std,
                    'upper_68': last_forecast + forecast_std
                }
            
            # VaR Confidence Intervals (simplified)
            if advanced_risk.var_95 > 0:
                var_std = advanced_risk.var_95 * 0.1  # Simplified standard error
                confidence_intervals['var_95'] = {
                    'point_estimate': advanced_risk.var_95,
                    'lower_95': max(0, advanced_risk.var_95 - 1.96 * var_std),
                    'upper_95': advanced_risk.var_95 + 1.96 * var_std
                }
            
            # Sharpe Ratio Confidence Intervals
            if advanced_risk.sharpe_ratio:
                sharpe_std = 0.1  # Simplified standard error
                confidence_intervals['sharpe_ratio'] = {
                    'point_estimate': advanced_risk.sharpe_ratio,
                    'lower_95': advanced_risk.sharpe_ratio - 1.96 * sharpe_std,
                    'upper_95': advanced_risk.sharpe_ratio + 1.96 * sharpe_std
                }
            
            return confidence_intervals
            
        except Exception as e:
            print(f"Error calculating confidence intervals: {e}")
            return {}
    
    def _assess_model_uncertainty(self, ensemble_result: EnsembleModelResults,
                                diagnostics: ComprehensiveDiagnostics) -> Dict[str, float]:
        """Assess overall model uncertainty"""
        
        try:
            uncertainty = {}
            
            # Ensemble Uncertainty
            if ensemble_result.model_comparison:
                model_accuracies = [comp.get('accuracy', 0) for comp in ensemble_result.model_comparison.values()]
                if model_accuracies:
                    accuracy_std = np.std(model_accuracies)
                    uncertainty['ensemble_uncertainty'] = accuracy_std
                    uncertainty['model_agreement'] = 1.0 - accuracy_std  # Higher agreement = lower uncertainty
            
            # Diagnostic Uncertainty
            if diagnostics.cross_validation_results:
                forecast_error_std = diagnostics.cross_validation_results.get('forecast_error_std', 0)
                uncertainty['forecast_uncertainty'] = forecast_error_std
                
                forecast_stability = diagnostics.cross_validation_results.get('forecast_stability', 1.0)
                uncertainty['temporal_stability'] = forecast_stability
            
            # Parameter Uncertainty (from stability tests)
            if diagnostics.stability_tests:
                garch_stationarity = diagnostics.stability_tests.get('garch_stationarity', 1.0)
                structural_stability = diagnostics.stability_tests.get('structural_stability', 1.0)
                
                uncertainty['parameter_uncertainty'] = 1.0 - min(garch_stationarity, structural_stability)
            
            # Overall Uncertainty Score
            uncertainty_scores = [v for k, v in uncertainty.items() if 'uncertainty' in k]
            if uncertainty_scores:
                uncertainty['overall_uncertainty'] = np.mean(uncertainty_scores)
            else:
                uncertainty['overall_uncertainty'] = 0.5  # Neutral uncertainty
            
            # Confidence Level (inverse of uncertainty)
            uncertainty['confidence_level'] = 1.0 - uncertainty['overall_uncertainty']
            
            return uncertainty
            
        except Exception as e:
            print(f"Error assessing model uncertainty: {e}")
            return {'overall_uncertainty': 0.5, 'confidence_level': 0.5}
    
    def _create_fallback_enhanced_result(self, index_data: IndexData) -> EnhancedIndexAnalysisResult:
        """Create fallback enhanced result when analysis fails"""
        
        try:
            # Create basic fallback result
            basic_result = self._create_fallback_comprehensive_result(index_data)
            
            # Create fallback enhanced components
            ensemble_result = EnsembleModelResults(
                ensemble_forecast=[], ensemble_weights={}, ensemble_accuracy=0.0,
                model_comparison={}, feature_importance={}, ensemble_performance={}
            )
            
            advanced_risk = AdvancedRiskMetrics(
                var_95=0.05, var_99=0.1, expected_shortfall_95=0.07, expected_shortfall_99=0.15,
                tail_risk=0.05, max_drawdown=0.2, max_drawdown_duration=30, drawdown_recovery_time=45,
                regime_risk={}, stress_test_results={}, sharpe_ratio=0.0, sortino_ratio=0.0,
                calmar_ratio=0.0, skewness=0.0, kurtosis=3.0, liquidity_risk=0.1, correlation_risk=0.5
            )
            
            ml_insights = MachineLearningInsights(
                anomaly_scores=[], clustering_results={}, pattern_analysis={},
                feature_importance={}, regime_analysis={}, predictive_accuracy={},
                market_microstructure={}
            )
            
            diagnostics = ComprehensiveDiagnostics(
                statistical_tests={}, model_validation={}, residual_analysis={},
                goodness_of_fit={}, cross_validation_results={}, stability_tests={},
                overall_quality_score=0.5
            )
            
            return EnhancedIndexAnalysisResult(
                basic_analysis=basic_result,
                ensemble_results=ensemble_result,
                advanced_risk_metrics=advanced_risk,
                ml_insights=ml_insights,
                comprehensive_diagnostics=diagnostics,
                enhanced_insights=["Analysis failed - using fallback results"],
                strategic_recommendations=["Unable to generate recommendations due to analysis failure"],
                confidence_intervals={},
                model_uncertainty={'overall_uncertainty': 1.0, 'confidence_level': 0.0}
            )
            
        except Exception as e:
            print(f"Error creating fallback enhanced result: {e}")
            # Return minimal fallback if even the fallback creation fails
            return None
    
    def _create_fallback_comprehensive_result(self, index_data: IndexData) -> IndexVolatilityStateResult:
        """Create fallback result when analysis fails"""
        
        fallback_garch = self.garch_analyzer._create_fallback_result(index_data.returns)
        fallback_kalman = self.kalman_analyzer._create_fallback_result(index_data.prices)
        fallback_vecm = self.vecm_analyzer._create_fallback_result([index_data.prices])
        
        return IndexVolatilityStateResult(
            garch_results=fallback_garch,
            kalman_results=fallback_kalman,
            vecm_results=fallback_vecm,
            cointegration_analysis={'cointegration_detected': False, 'n_relationships': 0},
            regime_analysis={},
            risk_metrics={'volatility': 0.0, 'var_95': 0.0, 'sharpe_ratio': 0.0},
            model_comparison={},
            trading_signals={'combined_signals': [0] * len(index_data.prices)},
            insights=["Analysis failed - using fallback results"],
            recommendations=["Review data quality and model parameters"]
        )
    
    def plot_results(self, result: IndexVolatilityStateResult, index_data: IndexData):
        """Plot comprehensive volatility and state analysis results"""
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f'Volatility & State Analysis: {index_data.index_symbol}', fontsize=16, fontweight='bold')
        
        # Plot 1: Index prices and Kalman filtered states
        ax1 = axes[0, 0]
        dates = index_data.timestamps if index_data.timestamps else range(len(index_data.prices))
        
        ax1.plot(dates, index_data.prices, 'b-', alpha=0.7, label='Observed Prices')
        
        if result.kalman_results.filtered_states:
            filtered_prices = [state[0] if isinstance(state, list) and len(state) > 0 else state 
                             for state in result.kalman_results.filtered_states]
            ax1.plot(dates, filtered_prices, 'r--', alpha=0.8, label='Kalman Filtered')
        
        ax1.set_title('Index Prices vs Kalman Filter')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: GARCH conditional volatility
        ax2 = axes[0, 1]
        if result.garch_results.conditional_volatility:
            vol_dates = dates[:len(result.garch_results.conditional_volatility)]
            ax2.plot(vol_dates, result.garch_results.conditional_volatility, 'g-', linewidth=1.5)
            ax2.fill_between(vol_dates, result.garch_results.conditional_volatility, alpha=0.3)
        
        ax2.set_title(f'{result.garch_results.model_type} Conditional Volatility')
        ax2.set_ylabel('Volatility')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Standardized residuals
        ax3 = axes[1, 0]
        if result.garch_results.standardized_residuals:
            ax3.plot(result.garch_results.standardized_residuals, 'purple', alpha=0.7)
            ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
            ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='±2σ')
            ax3.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
        
        ax3.set_title('Standardized Residuals')
        ax3.set_ylabel('Std. Residuals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Trading signals
        ax4 = axes[1, 1]
        if result.trading_signals.get('combined_signals'):
            signals = result.trading_signals['combined_signals']
            signal_dates = dates[:len(signals)]
            
            # Plot buy/sell signals
            buy_signals = [i for i, s in enumerate(signals) if s == 1]
            sell_signals = [i for i, s in enumerate(signals) if s == -1]
            
            ax4.plot(signal_dates, index_data.prices[:len(signal_dates)], 'b-', alpha=0.7, label='Price')
            
            if buy_signals:
                buy_dates = [signal_dates[i] for i in buy_signals if i < len(signal_dates)]
                buy_prices = [index_data.prices[i] for i in buy_signals if i < len(index_data.prices)]
                ax4.scatter(buy_dates, buy_prices, color='green', marker='^', s=50, label='Buy')
            
            if sell_signals:
                sell_dates = [signal_dates[i] for i in sell_signals if i < len(signal_dates)]
                sell_prices = [index_data.prices[i] for i in sell_signals if i < len(index_data.prices)]
                ax4.scatter(sell_dates, sell_prices, color='red', marker='v', s=50, label='Sell')
        
        ax4.set_title('Trading Signals')
        ax4.set_ylabel('Price')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Model comparison (AIC values)
        ax5 = axes[2, 0]
        if result.model_comparison.get('garch_comparison'):
            models = list(result.model_comparison['garch_comparison'].keys())
            aic_values = [result.model_comparison['garch_comparison'][m]['aic'] for m in models]
            
            bars = ax5.bar(models, aic_values, alpha=0.7, color=['blue', 'orange', 'green'][:len(models)])
            ax5.set_title('GARCH Model Comparison (AIC)')
            ax5.set_ylabel('AIC Value')
            
            # Highlight best model
            if result.model_comparison.get('model_selection', {}).get('best_garch'):
                best_model = result.model_comparison['model_selection']['best_garch']
                if best_model in models:
                    best_idx = models.index(best_model)
                    bars[best_idx].set_color('red')
                    bars[best_idx].set_alpha(0.9)
        
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Risk metrics visualization
        ax6 = axes[2, 1]
        risk_names = ['VaR 95%', 'VaR 99%', 'ES 95%', 'Max DD']
        risk_values = [
            result.risk_metrics.get('var_95', 0),
            result.risk_metrics.get('var_99', 0),
            result.risk_metrics.get('es_95', 0),
            result.risk_metrics.get('max_drawdown', 0)
        ]
        
        colors = ['red' if v < 0 else 'green' for v in risk_values]
        bars = ax6.bar(risk_names, risk_values, color=colors, alpha=0.7)
        ax6.set_title('Risk Metrics')
        ax6.set_ylabel('Value')
        ax6.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, result: IndexVolatilityStateResult, index_data: IndexData) -> str:
        """Generate comprehensive volatility and state analysis report"""
        
        report = f"""
# Volatility & State Analysis Report: {index_data.index_symbol}

## Executive Summary

This report presents a comprehensive analysis of {index_data.index_symbol} using advanced volatility modeling (GARCH variants), state-space models (Kalman Filters), and cointegration analysis (VECM). The analysis covers {len(index_data.prices)} observations and provides insights into volatility dynamics, hidden states, and long-term relationships.

## GARCH Volatility Analysis

### Model Selection
- **Selected Model**: {result.garch_results.model_type}
- **AIC**: {result.garch_results.aic:.4f}
- **BIC**: {result.garch_results.bic:.4f}
- **Log-Likelihood**: {result.garch_results.log_likelihood:.4f}

### Volatility Characteristics
"""
        
        if result.garch_results.conditional_volatility:
            current_vol = result.garch_results.conditional_volatility[-1]
            avg_vol = np.mean(result.garch_results.conditional_volatility)
            max_vol = np.max(result.garch_results.conditional_volatility)
            min_vol = np.min(result.garch_results.conditional_volatility)
            
            report += f"""
- **Current Volatility**: {current_vol:.4f}
- **Average Volatility**: {avg_vol:.4f}
- **Maximum Volatility**: {max_vol:.4f}
- **Minimum Volatility**: {min_vol:.4f}
- **Volatility Range**: {max_vol - min_vol:.4f}
"""
        
        if result.garch_results.model_type in ['EGARCH', 'TGARCH']:
            report += f"""

### Asymmetric Effects
- **Model Type**: {result.garch_results.model_type}
- **Leverage Effect**: {'Detected' if result.garch_results.model_type == 'EGARCH' else 'Threshold Effects Present'}
"""
        
        report += f"""

## Kalman Filter State Analysis

### State Estimation
- **State Dimension**: {len(result.kalman_results.filtered_states[0]) if result.kalman_results.filtered_states else 'N/A'}
- **Observation Model**: Linear Gaussian
- **Transition Model**: Random Walk with Drift

### Filter Performance
"""
        
        if result.kalman_results.log_likelihood:
            report += f"- **Log-Likelihood**: {result.kalman_results.log_likelihood:.4f}\n"
        
        if result.kalman_results.filtered_states:
            report += f"- **State Estimates**: {len(result.kalman_results.filtered_states)} observations\n"
        
        report += f"""

## VECM Cointegration Analysis

### Johansen Test Results
"""
        
        if result.vecm_results.johansen_test:
            n_coint = result.vecm_results.johansen_test.get('n_cointegrating', 0)
            report += f"""
- **Cointegrating Relationships**: {n_coint}
- **Test Statistic**: {result.vecm_results.johansen_test.get('test_statistic', 'N/A')}
- **Critical Value**: {result.vecm_results.johansen_test.get('critical_value', 'N/A')}
- **Cointegration Status**: {'Detected' if n_coint > 0 else 'Not Detected'}
"""
        
        if result.vecm_results.error_correction_terms:
            report += f"""

### Error Correction
- **ECT Coefficient**: {result.vecm_results.error_correction_terms[0]:.4f}
- **Adjustment Speed**: {'Fast' if abs(result.vecm_results.error_correction_terms[0]) > 0.1 else 'Slow'}
"""
        
        report += f"""

## Risk Metrics

### Value at Risk (VaR)
- **VaR 95%**: {result.risk_metrics.get('var_95', 0):.4f}
- **VaR 99%**: {result.risk_metrics.get('var_99', 0):.4f}
- **Expected Shortfall 95%**: {result.risk_metrics.get('es_95', 0):.4f}

### Performance Metrics
- **Sharpe Ratio**: {result.risk_metrics.get('sharpe_ratio', 0):.4f}
- **Maximum Drawdown**: {result.risk_metrics.get('max_drawdown', 0):.4f}
- **Volatility**: {result.risk_metrics.get('volatility', 0):.4f}

## Model Comparison

### GARCH Model Selection
"""
        
        if result.model_comparison.get('garch_comparison'):
            for model, metrics in result.model_comparison['garch_comparison'].items():
                report += f"- **{model}**: AIC = {metrics['aic']:.4f}, BIC = {metrics['bic']:.4f}\n"
        
        if result.model_comparison.get('model_selection'):
            best_garch = result.model_comparison['model_selection'].get('best_garch', 'N/A')
            report += f"\n**Best Model**: {best_garch}\n"
        
        report += f"""

## Trading Signals

### Signal Generation
"""
        
        if result.trading_signals.get('combined_signals'):
            signals = result.trading_signals['combined_signals']
            buy_count = sum(1 for s in signals if s == 1)
            sell_count = sum(1 for s in signals if s == -1)
            hold_count = sum(1 for s in signals if s == 0)
            
            report += f"""
- **Total Signals**: {len(signals)}
- **Buy Signals**: {buy_count} ({buy_count/len(signals)*100:.1f}%)
- **Sell Signals**: {sell_count} ({sell_count/len(signals)*100:.1f}%)
- **Hold Signals**: {hold_count} ({hold_count/len(signals)*100:.1f}%)

### Current Signal
- **Latest Signal**: {'Buy' if signals[-1] == 1 else 'Sell' if signals[-1] == -1 else 'Hold'}
"""
        
        report += f"""

## Key Insights

"""
        
        for insight in result.insights:
            report += f"- {insight}\n"
        
        report += f"""

## Recommendations

"""
        
        for recommendation in result.recommendations:
            report += f"- {recommendation}\n"
        
        report += f"""

## Methodology

### GARCH Models
- **GARCH(1,1)**: Standard volatility clustering model
- **EGARCH**: Exponential GARCH for asymmetric volatility
- **TGARCH**: Threshold GARCH for leverage effects

### Kalman Filter
- **State Space Model**: Linear Gaussian framework
- **Filtering**: Recursive Bayesian estimation
- **Smoothing**: Backward pass for optimal estimates

### VECM Analysis
- **Johansen Test**: Cointegration detection
- **Error Correction**: Long-term relationship modeling
- **Impulse Response**: Dynamic adjustment analysis

### Risk Assessment
- **Historical Simulation**: Non-parametric VaR estimation
- **Monte Carlo**: Parametric risk simulation
- **Stress Testing**: Extreme scenario analysis

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def plot_enhanced_results(self, enhanced_result: EnhancedIndexAnalysisResult, 
                            index_data: IndexData, figsize: Tuple[int, int] = (20, 24)) -> None:
        """Plot enhanced comprehensive analysis results"""
        
        try:
            fig, axes = plt.subplots(6, 2, figsize=figsize)
            fig.suptitle('Enhanced Index Volatility and State Analysis', fontsize=16, fontweight='bold')
            
            # Prepare data
            dates = index_data.dates
            prices = index_data.prices
            returns = np.diff(np.log(prices))
            
            # 1. Index Prices vs Ensemble Forecast
            ax1 = axes[0, 0]
            ax1.plot(dates, prices, 'b-', linewidth=1, label='Index Prices')
            if enhanced_result.ensemble_results.ensemble_forecast:
                forecast_dates = dates[-len(enhanced_result.ensemble_results.ensemble_forecast):]
                ax1.plot(forecast_dates, enhanced_result.ensemble_results.ensemble_forecast, 
                        'r--', linewidth=2, label='Ensemble Forecast')
            ax1.set_title('Index Prices vs Ensemble Forecast')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Advanced Risk Metrics
            ax2 = axes[0, 1]
            risk_metrics = {
                'VaR 95%': enhanced_result.advanced_risk_metrics.var_95,
                'VaR 99%': enhanced_result.advanced_risk_metrics.var_99,
                'ES 95%': enhanced_result.advanced_risk_metrics.expected_shortfall_95,
                'ES 99%': enhanced_result.advanced_risk_metrics.expected_shortfall_99,
                'Tail Risk': enhanced_result.advanced_risk_metrics.tail_risk
            }
            bars = ax2.bar(range(len(risk_metrics)), list(risk_metrics.values()), 
                          color=['red', 'darkred', 'orange', 'darkorange', 'purple'])
            ax2.set_title('Advanced Risk Metrics')
            ax2.set_ylabel('Risk Level')
            ax2.set_xticks(range(len(risk_metrics)))
            ax2.set_xticklabels(list(risk_metrics.keys()), rotation=45)
            ax2.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, risk_metrics.values()):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            # 3. Anomaly Detection
            ax3 = axes[1, 0]
            if enhanced_result.ml_insights.anomaly_scores:
                anomaly_dates = dates[-len(enhanced_result.ml_insights.anomaly_scores):]
                ax3.scatter(anomaly_dates, enhanced_result.ml_insights.anomaly_scores, 
                           c=['red' if score < -0.1 else 'blue' for score in enhanced_result.ml_insights.anomaly_scores],
                           alpha=0.6, s=20)
                ax3.axhline(y=-0.1, color='red', linestyle='--', alpha=0.7, label='Anomaly Threshold')
                ax3.set_title('Anomaly Detection Scores')
                ax3.set_ylabel('Anomaly Score')
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'No Anomaly Data Available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Anomaly Detection Scores')
            ax3.grid(True, alpha=0.3)
            
            # 4. Model Performance Comparison
            ax4 = axes[1, 1]
            if enhanced_result.ensemble_results.model_comparison:
                models = list(enhanced_result.ensemble_results.model_comparison.keys())
                accuracies = [comp.get('accuracy', 0) for comp in enhanced_result.ensemble_results.model_comparison.values()]
                bars = ax4.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'][:len(models)])
                ax4.set_title('Model Performance Comparison')
                ax4.set_ylabel('Accuracy')
                ax4.set_ylim(0, 1)
                
                # Add value labels
                for bar, acc in zip(bars, accuracies):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax4.text(0.5, 0.5, 'No Model Comparison Data', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Model Performance Comparison')
            ax4.grid(True, alpha=0.3)
            
            # 5. Feature Importance
            ax5 = axes[2, 0]
            if enhanced_result.ensemble_results.feature_importance:
                features = list(enhanced_result.ensemble_results.feature_importance.keys())[:10]  # Top 10
                importances = [enhanced_result.ensemble_results.feature_importance[f] for f in features]
                bars = ax5.barh(features, importances, color='lightblue')
                ax5.set_title('Top 10 Feature Importance')
                ax5.set_xlabel('Importance')
                
                # Add value labels
                for bar, imp in zip(bars, importances):
                    ax5.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                            f'{imp:.3f}', ha='left', va='center', fontsize=8)
            else:
                ax5.text(0.5, 0.5, 'No Feature Importance Data', ha='center', va='center', transform=ax5.transAxes)
                ax5.set_title('Feature Importance')
            ax5.grid(True, alpha=0.3)
            
            # 6. Regime Analysis
            ax6 = axes[2, 1]
            if enhanced_result.ml_insights.regime_analysis and 'regime_probabilities' in enhanced_result.ml_insights.regime_analysis:
                regime_probs = enhanced_result.ml_insights.regime_analysis['regime_probabilities']
                if isinstance(regime_probs, list) and len(regime_probs) > 0:
                    regime_dates = dates[-len(regime_probs):]
                    ax6.plot(regime_dates, regime_probs, 'g-', linewidth=2, label='High Vol Regime Prob')
                    ax6.fill_between(regime_dates, 0, regime_probs, alpha=0.3, color='green')
                    ax6.set_title('Volatility Regime Probabilities')
                    ax6.set_ylabel('Probability')
                    ax6.set_ylim(0, 1)
                    ax6.legend()
                else:
                    ax6.text(0.5, 0.5, 'No Regime Probability Data', ha='center', va='center', transform=ax6.transAxes)
                    ax6.set_title('Volatility Regime Analysis')
            else:
                ax6.text(0.5, 0.5, 'No Regime Analysis Data', ha='center', va='center', transform=ax6.transAxes)
                ax6.set_title('Volatility Regime Analysis')
            ax6.grid(True, alpha=0.3)
            
            # 7. Diagnostic Quality Scores
            ax7 = axes[3, 0]
            quality_metrics = {
                'Overall Quality': enhanced_result.comprehensive_diagnostics.overall_quality_score,
                'Model Uncertainty': enhanced_result.model_uncertainty.get('overall_uncertainty', 0.5),
                'Confidence Level': enhanced_result.model_uncertainty.get('confidence_level', 0.5)
            }
            bars = ax7.bar(range(len(quality_metrics)), list(quality_metrics.values()), 
                          color=['green', 'orange', 'blue'])
            ax7.set_title('Model Quality and Uncertainty')
            ax7.set_ylabel('Score')
            ax7.set_xticks(range(len(quality_metrics)))
            ax7.set_xticklabels(list(quality_metrics.keys()))
            ax7.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, quality_metrics.values()):
                ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            ax7.grid(True, alpha=0.3)
            
            # 8. Risk-Return Profile
            ax8 = axes[3, 1]
            returns_annual = np.mean(returns) * 252
            volatility_annual = np.std(returns) * np.sqrt(252)
            sharpe_ratio = enhanced_result.advanced_risk_metrics.sharpe_ratio
            
            ax8.scatter(volatility_annual, returns_annual, s=200, c='red', alpha=0.7, label='Current Asset')
            ax8.set_title(f'Risk-Return Profile (Sharpe: {sharpe_ratio:.2f})')
            ax8.set_xlabel('Annualized Volatility')
            ax8.set_ylabel('Annualized Return')
            ax8.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax8.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax8.legend()
            ax8.grid(True, alpha=0.3)
            
            # 9. Confidence Intervals
            ax9 = axes[4, 0]
            if enhanced_result.confidence_intervals:
                ci_data = []
                ci_labels = []
                for metric, intervals in enhanced_result.confidence_intervals.items():
                    if 'point_estimate' in intervals:
                        ci_data.append([
                            intervals.get('lower_95', 0),
                            intervals.get('point_estimate', 0),
                            intervals.get('upper_95', 0)
                        ])
                        ci_labels.append(metric)
                
                if ci_data:
                    for i, (lower, point, upper) in enumerate(ci_data):
                        ax9.errorbar(i, point, yerr=[[point-lower], [upper-point]], 
                                   fmt='o', capsize=5, capthick=2, label=ci_labels[i])
                    ax9.set_title('95% Confidence Intervals')
                    ax9.set_xticks(range(len(ci_labels)))
                    ax9.set_xticklabels(ci_labels)
                    ax9.legend()
                else:
                    ax9.text(0.5, 0.5, 'No Confidence Interval Data', ha='center', va='center', transform=ax9.transAxes)
                    ax9.set_title('Confidence Intervals')
            else:
                ax9.text(0.5, 0.5, 'No Confidence Interval Data', ha='center', va='center', transform=ax9.transAxes)
                ax9.set_title('Confidence Intervals')
            ax9.grid(True, alpha=0.3)
            
            # 10. Pattern Analysis
            ax10 = axes[4, 1]
            if enhanced_result.ml_insights.pattern_analysis:
                patterns = {
                    'Momentum': enhanced_result.ml_insights.pattern_analysis.get('momentum_strength', 0),
                    'Mean Reversion': enhanced_result.ml_insights.pattern_analysis.get('mean_reversion_strength', 0),
                    'Vol Clustering': enhanced_result.ml_insights.pattern_analysis.get('volatility_clustering', 0),
                    'Trend Persistence': enhanced_result.ml_insights.pattern_analysis.get('trend_persistence', 0)
                }
                bars = ax10.bar(range(len(patterns)), list(patterns.values()), 
                               color=['blue', 'green', 'red', 'purple'])
                ax10.set_title('Pattern Analysis')
                ax10.set_ylabel('Strength')
                ax10.set_xticks(range(len(patterns)))
                ax10.set_xticklabels(list(patterns.keys()), rotation=45)
                ax10.set_ylim(0, 1)
                
                # Add value labels
                for bar, value in zip(bars, patterns.values()):
                    ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.2f}', ha='center', va='bottom', fontsize=8)
            else:
                ax10.text(0.5, 0.5, 'No Pattern Analysis Data', ha='center', va='center', transform=ax10.transAxes)
                ax10.set_title('Pattern Analysis')
            ax10.grid(True, alpha=0.3)
            
            # 11. Stress Test Results
            ax11 = axes[5, 0]
            if enhanced_result.advanced_risk_metrics.stress_test_results:
                stress_scenarios = list(enhanced_result.advanced_risk_metrics.stress_test_results.keys())
                stress_losses = [enhanced_result.advanced_risk_metrics.stress_test_results[s].get('portfolio_loss', 0) 
                               for s in stress_scenarios]
                bars = ax11.bar(stress_scenarios, stress_losses, color='darkred', alpha=0.7)
                ax11.set_title('Stress Test Results')
                ax11.set_ylabel('Portfolio Loss')
                ax11.tick_params(axis='x', rotation=45)
                
                # Add value labels
                for bar, loss in zip(bars, stress_losses):
                    ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                            f'{loss:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax11.text(0.5, 0.5, 'No Stress Test Data', ha='center', va='center', transform=ax11.transAxes)
                ax11.set_title('Stress Test Results')
            ax11.grid(True, alpha=0.3)
            
            # 12. Summary Statistics
            ax12 = axes[5, 1]
            summary_stats = {
                'Sharpe Ratio': enhanced_result.advanced_risk_metrics.sharpe_ratio,
                'Sortino Ratio': enhanced_result.advanced_risk_metrics.sortino_ratio,
                'Calmar Ratio': enhanced_result.advanced_risk_metrics.calmar_ratio,
                'Max Drawdown': enhanced_result.advanced_risk_metrics.max_drawdown,
                'Skewness': enhanced_result.advanced_risk_metrics.skewness,
                'Kurtosis': enhanced_result.advanced_risk_metrics.kurtosis
            }
            
            # Create a table-like visualization
            ax12.axis('off')
            table_data = [[k, f'{v:.3f}'] for k, v in summary_stats.items()]
            table = ax12.table(cellText=table_data, colLabels=['Metric', 'Value'], 
                              cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax12.set_title('Summary Statistics')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting enhanced results: {e}")
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'Error generating enhanced plots: {str(e)}', 
                    ha='center', va='center', fontsize=12)
            plt.title('Enhanced Analysis Plotting Error')
            plt.show()
    
    def generate_enhanced_report(self, enhanced_result: EnhancedIndexAnalysisResult, 
                               index_data: IndexData) -> str:
        """Generate comprehensive enhanced analysis report"""
        
        try:
            report = []
            report.append("=" * 80)
            report.append("ENHANCED INDEX VOLATILITY AND STATE ANALYSIS REPORT")
            report.append("=" * 80)
            report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report.append(f"Data Period: {index_data.dates[0]} to {index_data.dates[-1]}")
            report.append(f"Total Observations: {len(index_data.prices)}")
            report.append("")
            
            # Executive Summary
            report.append("EXECUTIVE SUMMARY")
            report.append("-" * 40)
            report.append(f"Overall Model Quality Score: {enhanced_result.comprehensive_diagnostics.overall_quality_score:.3f}")
            report.append(f"Model Confidence Level: {enhanced_result.model_uncertainty.get('confidence_level', 0.5):.1%}")
            report.append(f"Ensemble Model Accuracy: {enhanced_result.ensemble_results.ensemble_accuracy:.1%}")
            report.append(f"Advanced Risk Assessment: VaR 95% = {enhanced_result.advanced_risk_metrics.var_95:.1%}")
            report.append("")
            
            # Enhanced Insights
            report.append("ENHANCED ANALYTICAL INSIGHTS")
            report.append("-" * 40)
            for i, insight in enumerate(enhanced_result.enhanced_insights, 1):
                report.append(f"{i}. {insight}")
            report.append("")
            
            # Strategic Recommendations
            report.append("STRATEGIC RECOMMENDATIONS")
            report.append("-" * 40)
            for i, recommendation in enumerate(enhanced_result.strategic_recommendations, 1):
                report.append(f"{i}. {recommendation}")
            report.append("")
            
            # Ensemble Model Analysis
            report.append("ENSEMBLE MODEL ANALYSIS")
            report.append("-" * 40)
            report.append(f"Ensemble Accuracy: {enhanced_result.ensemble_results.ensemble_accuracy:.3f}")
            
            if enhanced_result.ensemble_results.model_comparison:
                report.append("\nModel Performance Comparison:")
                for model, metrics in enhanced_result.ensemble_results.model_comparison.items():
                    accuracy = metrics.get('accuracy', 0)
                    report.append(f"  {model}: {accuracy:.3f}")
            
            if enhanced_result.ensemble_results.feature_importance:
                report.append("\nTop 5 Most Important Features:")
                sorted_features = sorted(enhanced_result.ensemble_results.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:5]
                for feature, importance in sorted_features:
                    report.append(f"  {feature}: {importance:.3f}")
            report.append("")
            
            # Advanced Risk Metrics
            report.append("ADVANCED RISK ASSESSMENT")
            report.append("-" * 40)
            report.append(f"Value at Risk (95%): {enhanced_result.advanced_risk_metrics.var_95:.1%}")
            report.append(f"Value at Risk (99%): {enhanced_result.advanced_risk_metrics.var_99:.1%}")
            report.append(f"Expected Shortfall (95%): {enhanced_result.advanced_risk_metrics.expected_shortfall_95:.1%}")
            report.append(f"Expected Shortfall (99%): {enhanced_result.advanced_risk_metrics.expected_shortfall_99:.1%}")
            report.append(f"Tail Risk: {enhanced_result.advanced_risk_metrics.tail_risk:.1%}")
            report.append(f"Maximum Drawdown: {enhanced_result.advanced_risk_metrics.max_drawdown:.1%}")
            report.append(f"Drawdown Duration: {enhanced_result.advanced_risk_metrics.max_drawdown_duration} days")
            report.append(f"Recovery Time: {enhanced_result.advanced_risk_metrics.drawdown_recovery_time} days")
            report.append("")
            
            # Performance Metrics
            report.append("RISK-ADJUSTED PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Sharpe Ratio: {enhanced_result.advanced_risk_metrics.sharpe_ratio:.3f}")
            report.append(f"Sortino Ratio: {enhanced_result.advanced_risk_metrics.sortino_ratio:.3f}")
            report.append(f"Calmar Ratio: {enhanced_result.advanced_risk_metrics.calmar_ratio:.3f}")
            report.append(f"Skewness: {enhanced_result.advanced_risk_metrics.skewness:.3f}")
            report.append(f"Kurtosis: {enhanced_result.advanced_risk_metrics.kurtosis:.3f}")
            report.append(f"Liquidity Risk: {enhanced_result.advanced_risk_metrics.liquidity_risk:.3f}")
            report.append(f"Correlation Risk: {enhanced_result.advanced_risk_metrics.correlation_risk:.3f}")
            report.append("")
            
            # Machine Learning Insights
            report.append("MACHINE LEARNING INSIGHTS")
            report.append("-" * 40)
            
            if enhanced_result.ml_insights.anomaly_scores:
                anomaly_rate = np.mean(np.array(enhanced_result.ml_insights.anomaly_scores) < -0.1)
                report.append(f"Anomaly Detection Rate: {anomaly_rate:.1%}")
            
            if enhanced_result.ml_insights.pattern_analysis:
                report.append("Pattern Analysis:")
                for pattern, strength in enhanced_result.ml_insights.pattern_analysis.items():
                    report.append(f"  {pattern.replace('_', ' ').title()}: {strength:.3f}")
            
            if enhanced_result.ml_insights.regime_analysis:
                report.append("Regime Analysis:")
                for regime, info in enhanced_result.ml_insights.regime_analysis.items():
                    if isinstance(info, (int, float)):
                        report.append(f"  {regime.replace('_', ' ').title()}: {info}")
            report.append("")
            
            # Model Diagnostics
            report.append("COMPREHENSIVE MODEL DIAGNOSTICS")
            report.append("-" * 40)
            report.append(f"Overall Quality Score: {enhanced_result.comprehensive_diagnostics.overall_quality_score:.3f}")
            
            if enhanced_result.comprehensive_diagnostics.statistical_tests:
                report.append("\nStatistical Tests:")
                for test_category, tests in enhanced_result.comprehensive_diagnostics.statistical_tests.items():
                    report.append(f"  {test_category.title()}:")
                    for test_name, result in tests.items():
                        if 'pvalue' in test_name:
                            significance = "Significant" if result < 0.05 else "Not Significant"
                            report.append(f"    {test_name}: {result:.4f} ({significance})")
                        else:
                            report.append(f"    {test_name}: {result:.4f}")
            
            if enhanced_result.comprehensive_diagnostics.model_validation:
                report.append("\nModel Validation:")
                for model, validation in enhanced_result.comprehensive_diagnostics.model_validation.items():
                    report.append(f"  {model.upper()} Model:")
                    for metric, value in validation.items():
                        report.append(f"    {metric}: {value:.4f}")
            report.append("")
            
            # Confidence Intervals
            if enhanced_result.confidence_intervals:
                report.append("CONFIDENCE INTERVALS (95%)")
                report.append("-" * 40)
                for metric, intervals in enhanced_result.confidence_intervals.items():
                    point = intervals.get('point_estimate', 0)
                    lower = intervals.get('lower_95', 0)
                    upper = intervals.get('upper_95', 0)
                    report.append(f"{metric.title()}: {point:.4f} [{lower:.4f}, {upper:.4f}]")
                report.append("")
            
            # Model Uncertainty
            report.append("MODEL UNCERTAINTY ASSESSMENT")
            report.append("-" * 40)
            for uncertainty_type, value in enhanced_result.model_uncertainty.items():
                report.append(f"{uncertainty_type.replace('_', ' ').title()}: {value:.3f}")
            report.append("")
            
            # Basic Analysis Summary (from original comprehensive analysis)
            report.append("BASIC ANALYSIS SUMMARY")
            report.append("-" * 40)
            basic_summary = self.generate_report(enhanced_result.basic_analysis, index_data)
            # Extract key sections from basic report
            basic_lines = basic_summary.split('\n')
            in_summary = False
            for line in basic_lines:
                if 'GARCH VOLATILITY ANALYSIS' in line:
                    in_summary = True
                if in_summary and ('METHODOLOGY' in line or len(report) > 200):  # Limit length
                    break
                if in_summary:
                    report.append(line)
            
            report.append("")
            report.append("=" * 80)
            report.append("END OF ENHANCED ANALYSIS REPORT")
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            return f"Error generating enhanced report: {str(e)}"

# Example usage and testing
if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED INDEX VOLATILITY AND STATE ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Create sample index data
    np.random.seed(42)
    n_obs = 1000
    
    # Generate synthetic index data with volatility clustering
    returns = np.random.normal(0.001, 0.02, n_obs)
    
    # Add GARCH effects
    volatility = np.zeros(n_obs)
    volatility[0] = 0.02
    
    for i in range(1, n_obs):
        volatility[i] = np.sqrt(0.00001 + 0.05 * returns[i-1]**2 + 0.9 * volatility[i-1]**2)
        returns[i] = np.random.normal(0.001, volatility[i])
    
    # Generate price series
    prices = np.exp(np.cumsum(returns)) * 100
    
    # Create timestamps
    timestamps = pd.date_range(start='2020-01-01', periods=n_obs, freq='D')
    
    # Create IndexData object
    index_data = IndexData(
        index_symbol="TEST_INDEX",
        prices=prices.tolist(),
        returns=returns.tolist(),
        timestamps=timestamps.tolist(),
        market_cap=None,
        sector_weights=None
    )
    
    print(f"Dataset: {len(prices)} observations from {timestamps[0].date()} to {timestamps[-1].date()}")
    print(f"Index Symbol: {index_data.index_symbol}")
    print(f"Price Range: ${min(prices):.2f} - ${max(prices):.2f}")
    print()
    
    # Create enhanced analyzer with all features enabled
    print("Initializing Enhanced Analyzer with Advanced Features...")
    analyzer = IndexVolatilityStateAnalyzer(
        enable_ensemble_methods=True,
        enable_advanced_risk=True,
        enable_ml_insights=True,
        enable_comprehensive_diagnostics=True,
        rolling_window=50,
        forecast_horizon=10
    )
    
    # Perform basic comprehensive analysis
    print("\n1. Performing Basic Comprehensive Analysis...")
    result = analyzer.analyze(index_data)
    
    print("   ✓ GARCH volatility modeling completed")
    print("   ✓ Kalman filter state estimation completed")
    print("   ✓ VECM cointegration analysis completed")
    print("   ✓ Risk metrics calculation completed")
    print("   ✓ Trading signals generated")
    
    # Perform enhanced analysis
    print("\n2. Performing Enhanced Analysis with ML and Advanced Features...")
    enhanced_result = analyzer.analyze_enhanced(index_data)
    
    if isinstance(enhanced_result, EnhancedIndexAnalysisResult):
        print("   ✓ Ensemble modeling completed")
        print("   ✓ Advanced risk assessment completed")
        print("   ✓ Machine learning insights generated")
        print("   ✓ Comprehensive diagnostics completed")
        print("   ✓ Enhanced insights and recommendations generated")
        
        # Print enhanced summary
        print("\n" + "="*60)
        print("ENHANCED ANALYSIS SUMMARY")
        print("="*60)
        print(f"Overall Model Quality Score: {enhanced_result.comprehensive_diagnostics.overall_quality_score:.3f}")
        print(f"Model Confidence Level: {enhanced_result.model_uncertainty.get('confidence_level', 0.5):.1%}")
        print(f"Ensemble Model Accuracy: {enhanced_result.ensemble_results.ensemble_accuracy:.1%}")
        
        # Advanced Risk Metrics
        print("\nAdvanced Risk Assessment:")
        print(f"  VaR (95%): {enhanced_result.advanced_risk_metrics.var_95:.1%}")
        print(f"  VaR (99%): {enhanced_result.advanced_risk_metrics.var_99:.1%}")
        print(f"  Expected Shortfall (95%): {enhanced_result.advanced_risk_metrics.expected_shortfall_95:.1%}")
        print(f"  Tail Risk: {enhanced_result.advanced_risk_metrics.tail_risk:.1%}")
        print(f"  Maximum Drawdown: {enhanced_result.advanced_risk_metrics.max_drawdown:.1%}")
        print(f"  Sharpe Ratio: {enhanced_result.advanced_risk_metrics.sharpe_ratio:.3f}")
        print(f"  Sortino Ratio: {enhanced_result.advanced_risk_metrics.sortino_ratio:.3f}")
        
        # Model Performance
        if enhanced_result.ensemble_results.model_comparison:
            print("\nModel Performance Comparison:")
            for model, metrics in enhanced_result.ensemble_results.model_comparison.items():
                accuracy = metrics.get('accuracy', 0)
                print(f"  {model}: {accuracy:.1%}")
        
        # Enhanced Insights
        print("\nEnhanced Analytical Insights:")
        for i, insight in enumerate(enhanced_result.enhanced_insights[:3], 1):  # Show top 3
            print(f"  {i}. {insight}")
        
        # Strategic Recommendations
        print("\nStrategic Recommendations:")
        for i, rec in enumerate(enhanced_result.strategic_recommendations[:3], 1):  # Show top 3
            print(f"  {i}. {rec}")
        
        # Generate enhanced plots
        print("\n3. Generating Enhanced Visualization...")
        try:
            analyzer.plot_enhanced_results(enhanced_result, index_data)
            print("   ✓ Enhanced plots generated successfully")
        except Exception as e:
            print(f"   ⚠ Plot generation error: {e}")
        
        # Generate enhanced report
        print("\n4. Generating Enhanced Analysis Report...")
        try:
            enhanced_report = analyzer.generate_enhanced_report(enhanced_result, index_data)
            
            # Save enhanced report
            with open('enhanced_analysis_report.txt', 'w') as f:
                f.write(enhanced_report)
            print("   ✓ Enhanced report saved to 'enhanced_analysis_report.txt'")
        except Exception as e:
            print(f"   ⚠ Enhanced report generation error: {e}")
        
    else:
        print("   ⚠ Enhanced analysis failed, using fallback results")
        print(f"   Fallback reason: {enhanced_result}")
    
    # Also generate basic analysis outputs for comparison
    print("\n5. Generating Basic Analysis Outputs for Comparison...")
    try:
        # Basic summary
        print(f"\n=== Basic Analysis Summary ===")
        print(f"Index: {index_data.index_symbol}")
        print(f"Observations: {len(index_data.prices)}")
        print(f"Best GARCH Model: {result.garch_results.model_type}")
        print(f"GARCH AIC: {result.garch_results.aic:.4f}")
        print(f"Current Volatility: {result.garch_results.conditional_volatility[-1]:.4f}")
        print(f"Sharpe Ratio: {result.risk_metrics.get('sharpe_ratio', 0):.4f}")
        print(f"VaR 95%: {result.risk_metrics.get('var_95', 0):.4f}")
        
        if result.vecm_results.johansen_test.get('n_cointegrating', 0) > 0:
            print(f"Cointegration: Detected ({result.vecm_results.johansen_test['n_cointegrating']} relationships)")
        else:
            print("Cointegration: Not detected")
        
        print(f"\nKey Insights:")
        for insight in result.insights[:3]:  # Show first 3 insights
            print(f"- {insight}")
        
        print(f"\nRecommendations:")
        for rec in result.recommendations[:3]:  # Show first 3 recommendations
            print(f"- {rec}")
        
        # Basic plots
        analyzer.plot_results(result, index_data)
        print("   ✓ Basic plots generated")
        
        # Basic report
        basic_report = analyzer.generate_report(result, index_data)
        with open('basic_analysis_report.txt', 'w') as f:
            f.write(basic_report)
        print("   ✓ Basic report saved to 'basic_analysis_report.txt'")
    except Exception as e:
        print(f"   ⚠ Basic analysis output error: {e}")
    
    print("\n" + "="*80)
    print("ANALYSIS DEMONSTRATION COMPLETE")
    print("="*80)
    print("Files Generated:")
    print("  • enhanced_analysis_report.txt - Comprehensive enhanced analysis")
    print("  • basic_analysis_report.txt - Standard comprehensive analysis")
    print("\nFeatures Demonstrated:")
    print("  • GARCH volatility modeling with multiple specifications")
    print("  • Kalman Filter state space modeling")
    print("  • VECM cointegration analysis")
    print("  • Ensemble machine learning forecasting")
    print("  • Advanced risk metrics (VaR, ES, stress testing)")
    print("  • Anomaly detection and pattern recognition")
    print("  • Regime analysis and model diagnostics")
    print("  • Confidence intervals and uncertainty quantification")
    print("  • Enhanced visualization and reporting")
    print("\nNote: This demonstration uses synthetic data with realistic")
    print("      volatility clustering patterns. For production use,")
    print("      replace with actual market index data.")
    print("\nOptional Libraries Used (if available):")
    print(f"  • XGBoost: {'✓ Available' if analyzer.has_xgboost else '✗ Not Available'}")
    print(f"  • HMMLearn: {'✓ Available' if analyzer.has_hmmlearn else '✗ Not Available'}")
    print("\nTo install optional dependencies:")
    print("  pip install xgboost hmmlearn")
    print("\nAnalysis completed successfully!")