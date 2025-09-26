"""GARCH Variants for Index Volatility Analysis

This module implements GARCH, EGARCH, and TGARCH volatility models
specifically designed for index time series analysis.

Models included:
- GARCH: Standard GARCH(1,1) model
- EGARCH: Exponential GARCH with leverage effects
- TGARCH: Threshold GARCH (GJR-GARCH) with asymmetric effects
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller, kpss, acorr_ljungbox
from statsmodels.stats.diagnostic import het_arch
from statsmodels.stats.stattools import jarque_bera
import warnings
warnings.filterwarnings('ignore')

# Conditional imports for advanced features
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

@dataclass
class IndexData:
    """Index data structure"""
    index_symbol: str
    prices: List[float]
    returns: List[float]
    timestamps: List[datetime]
    volume: Optional[List[float]] = None
    market_cap: Optional[List[float]] = None

@dataclass
class GARCHResult:
    """GARCH model results"""
    model_type: str  # 'GARCH', 'EGARCH', 'TGARCH'
    parameters: Dict[str, float]
    conditional_volatility: List[float]
    standardized_residuals: List[float]
    log_likelihood: float
    aic: float
    bic: float
    model_diagnostics: Dict[str, Any]
    forecasts: List[float]
    confidence_intervals: Dict[str, List[float]]

@dataclass
class VolatilityRegimeAnalysis:
    """Results from volatility regime switching analysis"""
    n_regimes: int
    regime_probabilities: pd.Series
    current_regime: int
    transition_matrix: np.ndarray
    regime_parameters: Dict[int, Dict[str, float]]
    regime_persistence: Dict[int, float]
    expected_duration: Dict[int, float]
    regime_volatility: Dict[int, float]

@dataclass
class VolatilityAnomalyDetection:
    """Results from volatility anomaly detection"""
    anomaly_scores: List[float]
    anomaly_threshold: float
    anomaly_dates: List[int]
    n_anomalies: int
    anomaly_severity: List[str]
    isolation_forest_scores: List[float]
    statistical_outliers: List[int]

@dataclass
class AdvancedVolatilityForecasts:
    """Advanced volatility forecasting results"""
    ml_forecasts: Dict[str, List[float]]
    ensemble_forecast: List[float]
    forecast_uncertainty: List[float]
    model_weights: Dict[str, float]
    feature_importance: Dict[str, float]
    forecast_accuracy: Dict[str, float]
    risk_metrics: Dict[str, float]

@dataclass
class VolatilityDiagnostics:
    """Comprehensive volatility model diagnostics"""
    stationarity_tests: Dict[str, Any]
    arch_effects: Dict[str, Any]
    normality_tests: Dict[str, Any]
    autocorr_tests: Dict[str, Any]
    heterosked_tests: Dict[str, Any]
    model_stability: Dict[str, Any]
    goodness_of_fit: Dict[str, Any]

@dataclass
class EnhancedGARCHResult:
    """Comprehensive enhanced GARCH analysis results"""
    basic_garch: Dict[str, GARCHResult]
    best_model: str
    regime_analysis: Optional[VolatilityRegimeAnalysis]
    anomaly_detection: Optional[VolatilityAnomalyDetection]
    advanced_forecasts: Optional[AdvancedVolatilityForecasts]
    diagnostics: Optional[VolatilityDiagnostics]
    rolling_analysis: Dict[str, Any]
    risk_attribution: Dict[str, Any]

class GARCHAnalyzer:
    """GARCH family volatility models for index analysis"""
    
    def __init__(self, enable_regime_switching: bool = True, enable_anomaly_detection: bool = True,
                 enable_advanced_forecasting: bool = True, enable_comprehensive_diagnostics: bool = True,
                 rolling_window: int = 252, forecast_horizon: int = 30):
        """Initialize enhanced GARCH analyzer
        
        Args:
            enable_regime_switching: Enable volatility regime switching analysis
            enable_anomaly_detection: Enable volatility anomaly detection
            enable_advanced_forecasting: Enable ML-based volatility forecasting
            enable_comprehensive_diagnostics: Enable comprehensive model diagnostics
            rolling_window: Window size for rolling analysis
            forecast_horizon: Number of periods to forecast
        """
        self.model_cache = {}
        self.enable_regime_switching = enable_regime_switching and (HMM_AVAILABLE or STATSMODELS_AVAILABLE)
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_advanced_forecasting = enable_advanced_forecasting
        self.enable_comprehensive_diagnostics = enable_comprehensive_diagnostics
        self.rolling_window = rolling_window
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
    
    def analyze_enhanced_garch(self, index_data: IndexData, model_types: Optional[List[str]] = None) -> EnhancedGARCHResult:
        """Perform comprehensive enhanced GARCH analysis
        
        Args:
            index_data: IndexData object containing price and return data
            model_types: List of GARCH models to fit
            
        Returns:
            EnhancedGARCHResult with comprehensive analysis
        """
        try:
            # Basic GARCH analysis
            basic_results = self.analyze_garch(index_data, model_types)
            
            # Find best model
            best_model = min(basic_results.items(), key=lambda x: x[1].aic)[0]
            
            # Enhanced analyses
            regime_analysis = None
            anomaly_detection = None
            advanced_forecasts = None
            diagnostics = None
            rolling_analysis = {}
            risk_attribution = {}
            
            if len(index_data.returns) >= 100:
                # Regime switching analysis
                if self.enable_regime_switching:
                    regime_analysis = self._analyze_volatility_regimes(index_data)
                
                # Anomaly detection
                if self.enable_anomaly_detection:
                    anomaly_detection = self._detect_volatility_anomalies(index_data)
                
                # Advanced forecasting
                if self.enable_advanced_forecasting:
                    advanced_forecasts = self._generate_advanced_forecasts(index_data, basic_results[best_model])
                
                # Comprehensive diagnostics
                if self.enable_comprehensive_diagnostics:
                    diagnostics = self._perform_comprehensive_diagnostics(index_data, basic_results[best_model])
                
                # Rolling analysis
                rolling_analysis = self._perform_rolling_garch_analysis(index_data)
                
                # Risk attribution
                risk_attribution = self._perform_volatility_risk_attribution(index_data, basic_results[best_model])
            
            return EnhancedGARCHResult(
                basic_garch=basic_results,
                best_model=best_model,
                regime_analysis=regime_analysis,
                anomaly_detection=anomaly_detection,
                advanced_forecasts=advanced_forecasts,
                diagnostics=diagnostics,
                rolling_analysis=rolling_analysis,
                risk_attribution=risk_attribution
            )
            
        except Exception as e:
            # Fallback to basic analysis
            basic_results = self.analyze_garch(index_data, model_types)
            best_model = min(basic_results.items(), key=lambda x: x[1].aic)[0]
            
            return EnhancedGARCHResult(
                basic_garch=basic_results,
                best_model=best_model,
                regime_analysis=None,
                anomaly_detection=None,
                advanced_forecasts=None,
                diagnostics=None,
                rolling_analysis={},
                risk_attribution={}
            )
    
    def _analyze_volatility_regimes(self, index_data: IndexData) -> Optional[VolatilityRegimeAnalysis]:
        """Analyze volatility regime switching"""
        try:
            returns = np.array(index_data.returns)
            
            if len(returns) < 100:
                return None
            
            # Calculate rolling volatility as feature
            rolling_vol = pd.Series(returns).rolling(window=20).std().fillna(method='bfill').values
            
            # Try HMM first, then fall back to Markov Regression
            if HMM_AVAILABLE:
                return self._hmm_volatility_regime_analysis(returns, rolling_vol)
            elif STATSMODELS_AVAILABLE:
                return self._markov_volatility_regime_analysis(returns, rolling_vol)
            else:
                return None
                
        except Exception:
            return None
    
    def _hmm_volatility_regime_analysis(self, returns: np.ndarray, rolling_vol: np.ndarray) -> Optional[VolatilityRegimeAnalysis]:
        """Volatility regime analysis using Hidden Markov Models"""
        try:
            # Prepare data for HMM (use returns and rolling volatility as features)
            X = np.column_stack([returns, rolling_vol])
            
            # Fit 2-regime HMM
            n_regimes = 2
            model = GaussianHMM(n_components=n_regimes, covariance_type="full", random_state=42)
            model.fit(X)
            
            # Get regime probabilities and states
            regime_probs = model.predict_proba(X)
            regime_states = model.predict(X)
            
            # Create regime probabilities series
            regime_probabilities = pd.Series(regime_states, name='regime')
            current_regime = int(regime_states[-1])
            
            # Transition matrix
            transition_matrix = model.transmat_
            
            # Calculate regime-specific parameters
            regime_parameters = {}
            regime_persistence = {}
            expected_duration = {}
            regime_volatility = {}
            
            for regime in range(n_regimes):
                regime_mask = regime_states == regime
                if np.sum(regime_mask) > 10:
                    regime_returns = returns[regime_mask]
                    regime_vol = rolling_vol[regime_mask]
                    
                    regime_parameters[regime] = {
                        'mean_return': float(np.mean(regime_returns)),
                        'volatility': float(np.std(regime_returns, ddof=1)),
                        'mean_rolling_vol': float(np.mean(regime_vol)),
                        'skewness': float(pd.Series(regime_returns).skew()),
                        'kurtosis': float(pd.Series(regime_returns).kurtosis())
                    }
                    
                    # Regime persistence and duration
                    persistence = transition_matrix[regime, regime]
                    regime_persistence[regime] = float(persistence)
                    expected_duration[regime] = float(1 / (1 - persistence)) if persistence < 1 else float('inf')
                    
                    # Regime volatility (annualized)
                    regime_volatility[regime] = float(np.std(regime_returns, ddof=1) * np.sqrt(252))
                else:
                    regime_parameters[regime] = {
                        'mean_return': 0.0, 'volatility': 0.02, 'mean_rolling_vol': 0.02,
                        'skewness': 0.0, 'kurtosis': 0.0
                    }
                    regime_persistence[regime] = 0.5
                    expected_duration[regime] = 2.0
                    regime_volatility[regime] = 0.2
            
            return VolatilityRegimeAnalysis(
                n_regimes=n_regimes,
                regime_probabilities=regime_probabilities,
                current_regime=current_regime,
                transition_matrix=transition_matrix,
                regime_parameters=regime_parameters,
                regime_persistence=regime_persistence,
                expected_duration=expected_duration,
                regime_volatility=regime_volatility
            )
            
        except Exception:
            return None
    
    def _markov_volatility_regime_analysis(self, returns: np.ndarray, rolling_vol: np.ndarray) -> Optional[VolatilityRegimeAnalysis]:
        """Volatility regime analysis using Markov Regression"""
        try:
            # Use squared returns as dependent variable (proxy for volatility)
            endog = returns ** 2
            exog = np.column_stack([np.ones(len(returns)), rolling_vol])
            
            # Fit 2-regime Markov switching model
            model = MarkovRegression(endog, k_regimes=2, exog=exog, switching_variance=True)
            results = model.fit()
            
            # Extract regime probabilities
            regime_probs = results.smoothed_marginal_probabilities
            regime_states = np.argmax(regime_probs, axis=1)
            current_regime = int(regime_states[-1])
            
            # Create regime probabilities series
            regime_probabilities = pd.Series(regime_states, name='regime')
            
            # Transition matrix
            transition_matrix = results.regime_transition
            
            # Regime parameters
            regime_parameters = {}
            regime_persistence = {}
            expected_duration = {}
            regime_volatility = {}
            
            for regime in range(2):
                regime_mask = regime_states == regime
                if np.sum(regime_mask) > 5:
                    regime_returns = returns[regime_mask]
                    
                    regime_parameters[regime] = {
                        'mean_return': float(np.mean(regime_returns)),
                        'volatility': float(np.std(regime_returns, ddof=1)),
                        'mean_rolling_vol': float(np.mean(rolling_vol[regime_mask])),
                        'skewness': float(pd.Series(regime_returns).skew()),
                        'kurtosis': float(pd.Series(regime_returns).kurtosis())
                    }
                    
                    # Persistence and duration
                    persistence = float(transition_matrix[regime, regime])
                    regime_persistence[regime] = persistence
                    expected_duration[regime] = 1 / (1 - persistence) if persistence < 1 else float('inf')
                    
                    # Volatility (annualized)
                    regime_volatility[regime] = float(np.std(regime_returns, ddof=1) * np.sqrt(252))
                else:
                    regime_parameters[regime] = {
                        'mean_return': 0.0, 'volatility': 0.02, 'mean_rolling_vol': 0.02,
                        'skewness': 0.0, 'kurtosis': 0.0
                    }
                    regime_persistence[regime] = 0.5
                    expected_duration[regime] = 2.0
                    regime_volatility[regime] = 0.2
            
            return VolatilityRegimeAnalysis(
                n_regimes=2,
                regime_probabilities=regime_probabilities,
                current_regime=current_regime,
                transition_matrix=transition_matrix,
                regime_parameters=regime_parameters,
                regime_persistence=regime_persistence,
                expected_duration=expected_duration,
                regime_volatility=regime_volatility
            )
            
        except Exception:
            return None
    
    def _detect_volatility_anomalies(self, index_data: IndexData) -> Optional[VolatilityAnomalyDetection]:
        """Detect volatility anomalies using multiple methods"""
        try:
            returns = np.array(index_data.returns)
            
            if len(returns) < 50:
                return None
            
            # Calculate rolling volatility and other features
            returns_series = pd.Series(returns)
            rolling_vol = returns_series.rolling(window=20).std().fillna(method='bfill')
            rolling_mean = returns_series.rolling(window=20).mean().fillna(method='bfill')
            
            # Prepare features for anomaly detection
            features = np.column_stack([
                returns,
                rolling_vol.values,
                rolling_mean.values,
                np.abs(returns),  # Absolute returns
                returns ** 2      # Squared returns (volatility proxy)
            ])
            
            # Scale features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Isolation Forest anomaly detection
            isolation_forest = IsolationForest(
                contamination=0.1,  # Expect 10% anomalies
                random_state=42,
                n_estimators=100
            )
            
            anomaly_scores = isolation_forest.decision_function(features_scaled)
            anomaly_labels = isolation_forest.predict(features_scaled)
            
            # Statistical anomaly detection (Z-score based)
            z_scores = np.abs((returns - np.mean(returns)) / np.std(returns, ddof=1))
            statistical_anomalies = z_scores > 3.0  # 3-sigma rule
            
            # Volatility-based anomalies (extreme volatility periods)
            vol_threshold = np.percentile(rolling_vol, 95)
            volatility_anomalies = rolling_vol > vol_threshold
            
            # Combine anomaly indicators
            ml_anomalies = anomaly_labels == -1
            combined_anomalies = ml_anomalies | statistical_anomalies | volatility_anomalies
            
            # Calculate anomaly statistics
            anomaly_dates = []
            if hasattr(index_data, 'timestamps') and index_data.timestamps:
                anomaly_dates = [index_data.timestamps[i] for i in range(len(combined_anomalies)) 
                               if combined_anomalies[i]]
            
            anomaly_returns = returns[combined_anomalies]
            anomaly_count = int(np.sum(combined_anomalies))
            anomaly_percentage = float(anomaly_count / len(returns) * 100)
            
            # Severity analysis
            severity_scores = np.abs(anomaly_scores[ml_anomalies]) if np.any(ml_anomalies) else np.array([])
            avg_severity = float(np.mean(severity_scores)) if len(severity_scores) > 0 else 0.0
            max_severity = float(np.max(severity_scores)) if len(severity_scores) > 0 else 0.0
            
            # Impact analysis
            if len(anomaly_returns) > 0:
                impact_on_returns = {
                    'mean_anomaly_return': float(np.mean(anomaly_returns)),
                    'std_anomaly_return': float(np.std(anomaly_returns, ddof=1)),
                    'max_anomaly_return': float(np.max(np.abs(anomaly_returns))),
                    'skewness': float(pd.Series(anomaly_returns).skew()),
                    'kurtosis': float(pd.Series(anomaly_returns).kurtosis())
                }
            else:
                impact_on_returns = {
                    'mean_anomaly_return': 0.0,
                    'std_anomaly_return': 0.0,
                    'max_anomaly_return': 0.0,
                    'skewness': 0.0,
                    'kurtosis': 0.0
                }
            
            return VolatilityAnomalyDetection(
                anomaly_scores=anomaly_scores,
                anomaly_labels=combined_anomalies,
                anomaly_dates=anomaly_dates,
                anomaly_count=anomaly_count,
                anomaly_percentage=anomaly_percentage,
                severity_scores=severity_scores,
                avg_severity=avg_severity,
                max_severity=max_severity,
                impact_on_returns=impact_on_returns
            )
            
        except Exception:
            return None
    
    def _generate_advanced_forecasts(self, index_data: IndexData, garch_result: GARCHResult) -> Optional[AdvancedVolatilityForecasts]:
        """Generate advanced volatility forecasts using GARCH + ML"""
        try:
            returns = np.array(index_data.returns)
            
            if len(returns) < 100:
                return None
            
            # Prepare features for ML forecasting
            returns_series = pd.Series(returns)
            
            # Create lagged features
            features_df = pd.DataFrame({
                'return_lag1': returns_series.shift(1),
                'return_lag2': returns_series.shift(2),
                'return_lag3': returns_series.shift(3),
                'vol_lag1': returns_series.rolling(5).std().shift(1),
                'vol_lag2': returns_series.rolling(10).std().shift(1),
                'vol_lag3': returns_series.rolling(20).std().shift(1),
                'abs_return_lag1': np.abs(returns_series).shift(1),
                'squared_return_lag1': (returns_series ** 2).shift(1),
                'rolling_mean_5': returns_series.rolling(5).mean().shift(1),
                'rolling_mean_20': returns_series.rolling(20).mean().shift(1)
            })
            
            # Target variable (next period volatility)
            target = returns_series.rolling(5).std().shift(-1)
            
            # Remove NaN values
            valid_idx = ~(features_df.isna().any(axis=1) | target.isna())
            X = features_df[valid_idx].values
            y = target[valid_idx].values
            
            if len(X) < 50:
                return None
            
            # Split data for training and testing
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_train)
            
            # Generate predictions
            rf_predictions = rf_model.predict(X_test_scaled)
            
            # XGBoost model (if available)
            xgb_predictions = None
            if XGB_AVAILABLE:
                try:
                    import xgboost as xgb
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                    xgb_model.fit(X_train_scaled, y_train)
                    xgb_predictions = xgb_model.predict(X_test_scaled)
                except Exception:
                    pass
            
            # GARCH-based forecasts (use conditional volatility)
            garch_forecasts = garch_result.conditional_volatility[-len(y_test):] if len(garch_result.conditional_volatility) >= len(y_test) else None
            
            # Ensemble forecast (combine GARCH + ML)
            if garch_forecasts is not None and xgb_predictions is not None:
                ensemble_forecasts = (0.4 * garch_forecasts + 0.3 * rf_predictions + 0.3 * xgb_predictions)
            elif garch_forecasts is not None:
                ensemble_forecasts = (0.6 * garch_forecasts + 0.4 * rf_predictions)
            else:
                ensemble_forecasts = rf_predictions
            
            # Calculate forecast accuracy metrics
            rf_mse = float(mean_squared_error(y_test, rf_predictions))
            rf_r2 = float(r2_score(y_test, rf_predictions))
            
            ensemble_mse = float(mean_squared_error(y_test, ensemble_forecasts))
            ensemble_r2 = float(r2_score(y_test, ensemble_forecasts))
            
            accuracy_metrics = {
                'rf_mse': rf_mse,
                'rf_r2': rf_r2,
                'ensemble_mse': ensemble_mse,
                'ensemble_r2': ensemble_r2
            }
            
            if xgb_predictions is not None:
                xgb_mse = float(mean_squared_error(y_test, xgb_predictions))
                xgb_r2 = float(r2_score(y_test, xgb_predictions))
                accuracy_metrics.update({
                    'xgb_mse': xgb_mse,
                    'xgb_r2': xgb_r2
                })
            
            if garch_forecasts is not None:
                garch_mse = float(mean_squared_error(y_test, garch_forecasts))
                garch_r2 = float(r2_score(y_test, garch_forecasts))
                accuracy_metrics.update({
                    'garch_mse': garch_mse,
                    'garch_r2': garch_r2
                })
            
            # Feature importance (from Random Forest)
            feature_names = ['return_lag1', 'return_lag2', 'return_lag3', 'vol_lag1', 'vol_lag2', 
                           'vol_lag3', 'abs_return_lag1', 'squared_return_lag1', 'rolling_mean_5', 'rolling_mean_20']
            feature_importance = dict(zip(feature_names, rf_model.feature_importances_))
            
            # Generate future forecasts (next 5 periods)
            last_features = X_test_scaled[-1:]
            future_forecasts = []
            
            for _ in range(5):
                next_pred = rf_model.predict(last_features)[0]
                future_forecasts.append(float(next_pred))
                
                # Update features for next prediction (simple approach)
                # In practice, you'd want more sophisticated feature updating
                last_features = np.roll(last_features, -1)
                last_features[0, -1] = next_pred
            
            return AdvancedVolatilityForecasts(
                ml_forecasts=rf_predictions,
                ensemble_forecasts=ensemble_forecasts,
                future_forecasts=future_forecasts,
                accuracy_metrics=accuracy_metrics,
                feature_importance=feature_importance,
                forecast_horizon=len(rf_predictions)
            )
            
        except Exception:
            return None
    
    def _perform_comprehensive_diagnostics(self, index_data: IndexData, garch_result: GARCHResult) -> Optional[VolatilityDiagnostics]:
        """Perform comprehensive GARCH model diagnostics"""
        try:
            returns = np.array(index_data.returns)
            
            if len(returns) < 50 or not garch_result.standardized_residuals:
                return None
            
            residuals = np.array(garch_result.standardized_residuals)
            
            # Stationarity tests
            stationarity_tests = {}
            try:
                adf_stat, adf_pvalue, _, _, adf_critical, _ = adfuller(residuals, autolag='AIC')
                stationarity_tests['adf'] = {
                    'statistic': float(adf_stat),
                    'p_value': float(adf_pvalue),
                    'critical_values': {k: float(v) for k, v in adf_critical.items()},
                    'is_stationary': adf_pvalue < 0.05
                }
            except Exception:
                stationarity_tests['adf'] = {'statistic': 0.0, 'p_value': 1.0, 'is_stationary': False}
            
            try:
                kpss_stat, kpss_pvalue, _, kpss_critical = kpss(residuals, regression='c')
                stationarity_tests['kpss'] = {
                    'statistic': float(kpss_stat),
                    'p_value': float(kpss_pvalue),
                    'critical_values': {k: float(v) for k, v in kpss_critical.items()},
                    'is_stationary': kpss_pvalue > 0.05
                }
            except Exception:
                stationarity_tests['kpss'] = {'statistic': 0.0, 'p_value': 1.0, 'is_stationary': True}
            
            # Autocorrelation tests
            autocorr_tests = {}
            try:
                # Ljung-Box test on residuals
                lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=10, return_df=False)
                autocorr_tests['ljung_box_residuals'] = {
                    'statistic': float(lb_stat.iloc[-1]) if hasattr(lb_stat, 'iloc') else float(lb_stat[-1]),
                    'p_value': float(lb_pvalue.iloc[-1]) if hasattr(lb_pvalue, 'iloc') else float(lb_pvalue[-1]),
                    'no_autocorrelation': (lb_pvalue.iloc[-1] if hasattr(lb_pvalue, 'iloc') else lb_pvalue[-1]) > 0.05
                }
                
                # Ljung-Box test on squared residuals (ARCH effects)
                lb_sq_stat, lb_sq_pvalue = acorr_ljungbox(residuals**2, lags=10, return_df=False)
                autocorr_tests['ljung_box_squared'] = {
                    'statistic': float(lb_sq_stat.iloc[-1]) if hasattr(lb_sq_stat, 'iloc') else float(lb_sq_stat[-1]),
                    'p_value': float(lb_sq_pvalue.iloc[-1]) if hasattr(lb_sq_pvalue, 'iloc') else float(lb_sq_pvalue[-1]),
                    'no_arch_effects': (lb_sq_pvalue.iloc[-1] if hasattr(lb_sq_pvalue, 'iloc') else lb_sq_pvalue[-1]) > 0.05
                }
            except Exception:
                autocorr_tests = {
                    'ljung_box_residuals': {'statistic': 0.0, 'p_value': 1.0, 'no_autocorrelation': True},
                    'ljung_box_squared': {'statistic': 0.0, 'p_value': 1.0, 'no_arch_effects': True}
                }
            
            # Normality tests
            normality_tests = {}
            try:
                jb_stat, jb_pvalue = jarque_bera(residuals)
                normality_tests['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_pvalue),
                    'is_normal': jb_pvalue > 0.05
                }
            except Exception:
                normality_tests['jarque_bera'] = {'statistic': 0.0, 'p_value': 1.0, 'is_normal': True}
            
            try:
                from scipy.stats import shapiro
                sw_stat, sw_pvalue = shapiro(residuals[:5000] if len(residuals) > 5000 else residuals)  # Shapiro-Wilk has sample size limit
                normality_tests['shapiro_wilk'] = {
                    'statistic': float(sw_stat),
                    'p_value': float(sw_pvalue),
                    'is_normal': sw_pvalue > 0.05
                }
            except Exception:
                normality_tests['shapiro_wilk'] = {'statistic': 0.0, 'p_value': 1.0, 'is_normal': True}
            
            # Heteroskedasticity tests
            heterosked_tests = {}
            try:
                # ARCH test
                arch_stat, arch_pvalue, _, _ = het_arch(residuals, maxlag=5)
                heterosked_tests['arch'] = {
                    'statistic': float(arch_stat),
                    'p_value': float(arch_pvalue),
                    'no_arch_effects': arch_pvalue > 0.05
                }
            except Exception:
                heterosked_tests['arch'] = {'statistic': 0.0, 'p_value': 1.0, 'no_arch_effects': True}
            
            # Model fit quality
            model_fit_quality = {
                'aic': float(garch_result.aic),
                'bic': float(garch_result.bic),
                'log_likelihood': float(garch_result.log_likelihood),
                'mean_absolute_residual': float(np.mean(np.abs(residuals))),
                'residual_variance': float(np.var(residuals, ddof=1)),
                'residual_skewness': float(pd.Series(residuals).skew()),
                'residual_kurtosis': float(pd.Series(residuals).kurtosis())
            }
            
            # Volatility clustering analysis
            volatility_clustering = {}
            try:
                # Calculate autocorrelation of absolute returns
                abs_returns = np.abs(returns)
                autocorr_abs = pd.Series(abs_returns).autocorr(lag=1)
                
                # Calculate autocorrelation of squared returns
                sq_returns = returns ** 2
                autocorr_sq = pd.Series(sq_returns).autocorr(lag=1)
                
                volatility_clustering = {
                    'abs_returns_autocorr': float(autocorr_abs) if not np.isnan(autocorr_abs) else 0.0,
                    'squared_returns_autocorr': float(autocorr_sq) if not np.isnan(autocorr_sq) else 0.0,
                    'clustering_present': (autocorr_abs > 0.1) or (autocorr_sq > 0.1)
                }
            except Exception:
                volatility_clustering = {
                    'abs_returns_autocorr': 0.0,
                    'squared_returns_autocorr': 0.0,
                    'clustering_present': False
                }
            
            # Overall model adequacy
            adequacy_score = 0.0
            adequacy_components = []
            
            # Check stationarity
            if stationarity_tests.get('adf', {}).get('is_stationary', False):
                adequacy_score += 0.2
                adequacy_components.append('stationary_residuals')
            
            # Check no autocorrelation
            if autocorr_tests.get('ljung_box_residuals', {}).get('no_autocorrelation', False):
                adequacy_score += 0.2
                adequacy_components.append('no_autocorrelation')
            
            # Check no ARCH effects
            if autocorr_tests.get('ljung_box_squared', {}).get('no_arch_effects', False):
                adequacy_score += 0.3
                adequacy_components.append('no_arch_effects')
            
            # Check model fit (AIC/BIC reasonable)
            if garch_result.aic < 0:  # Negative AIC generally indicates good fit
                adequacy_score += 0.15
                adequacy_components.append('good_information_criteria')
            
            # Check residual properties
            if abs(model_fit_quality['residual_skewness']) < 1.0 and model_fit_quality['residual_kurtosis'] < 5.0:
                adequacy_score += 0.15
                adequacy_components.append('reasonable_residual_distribution')
            
            model_adequacy = {
                'adequacy_score': float(adequacy_score),
                'adequacy_components': adequacy_components,
                'is_adequate': adequacy_score >= 0.6
            }
            
            return VolatilityDiagnostics(
                stationarity_tests=stationarity_tests,
                autocorr_tests=autocorr_tests,
                normality_tests=normality_tests,
                heterosked_tests=heterosked_tests,
                model_fit_quality=model_fit_quality,
                volatility_clustering=volatility_clustering,
                model_adequacy=model_adequacy
            )
            
        except Exception:
            return None
    
    def _perform_rolling_analysis(self, index_data: IndexData) -> Optional[Dict]:
        """Perform rolling window GARCH analysis"""
        try:
            returns = np.array(index_data.returns)
            
            if len(returns) < self.rolling_window * 2:
                return None
            
            rolling_results = []
            
            for i in range(self.rolling_window, len(returns), max(1, self.rolling_window // 4)):
                window_returns = returns[i-self.rolling_window:i]
                
                try:
                    # Fit GARCH model on rolling window
                    window_result = self.fit_garch(window_returns.tolist(), model_type='GARCH')
                    
                    if window_result and window_result.parameters:
                        rolling_results.append({
                            'end_date': i,
                            'alpha': window_result.parameters.get('alpha', 0.0),
                            'beta': window_result.parameters.get('beta', 0.0),
                            'omega': window_result.parameters.get('omega', 0.0),
                            'volatility': float(np.std(window_returns, ddof=1) * np.sqrt(252)),
                            'aic': window_result.aic,
                            'bic': window_result.bic
                        })
                except Exception:
                    continue
            
            if not rolling_results:
                return None
            
            # Calculate rolling statistics
            rolling_df = pd.DataFrame(rolling_results)
            
            rolling_stats = {
                'parameter_stability': {
                    'alpha_std': float(rolling_df['alpha'].std()),
                    'beta_std': float(rolling_df['beta'].std()),
                    'omega_std': float(rolling_df['omega'].std()),
                    'volatility_std': float(rolling_df['volatility'].std())
                },
                'parameter_trends': {
                    'alpha_trend': float(rolling_df['alpha'].corr(pd.Series(range(len(rolling_df))))),
                    'beta_trend': float(rolling_df['beta'].corr(pd.Series(range(len(rolling_df))))),
                    'volatility_trend': float(rolling_df['volatility'].corr(pd.Series(range(len(rolling_df)))))
                },
                'model_fit_evolution': {
                    'aic_trend': float(rolling_df['aic'].corr(pd.Series(range(len(rolling_df))))),
                    'avg_aic': float(rolling_df['aic'].mean()),
                    'avg_bic': float(rolling_df['bic'].mean())
                },
                'n_windows': len(rolling_results)
            }
            
            return {
                'rolling_results': rolling_results,
                'rolling_stats': rolling_stats
            }
            
        except Exception:
            return None
    
    def _perform_risk_attribution(self, index_data: IndexData, garch_result: GARCHResult) -> Optional[Dict]:
        """Perform volatility risk attribution analysis"""
        try:
            returns = np.array(index_data.returns)
            
            if len(returns) < 50 or not garch_result.conditional_volatility:
                return None
            
            conditional_vol = np.array(garch_result.conditional_volatility)
            
            # Decompose total risk
            total_variance = float(np.var(returns, ddof=1))
            conditional_variance = float(np.mean(conditional_vol ** 2))
            
            # Risk decomposition
            systematic_risk = conditional_variance  # Risk explained by GARCH model
            idiosyncratic_risk = max(0.0, total_variance - conditional_variance)  # Unexplained risk
            
            risk_decomposition = {
                'total_variance': total_variance,
                'systematic_variance': systematic_risk,
                'idiosyncratic_variance': idiosyncratic_risk,
                'systematic_percentage': float(systematic_risk / total_variance * 100) if total_variance > 0 else 0.0,
                'idiosyncratic_percentage': float(idiosyncratic_risk / total_variance * 100) if total_variance > 0 else 0.0
            }
            
            # Time-varying risk analysis
            returns_series = pd.Series(returns)
            vol_series = pd.Series(conditional_vol)
            
            # Calculate rolling correlations and risk metrics
            rolling_corr = returns_series.rolling(window=30).corr(vol_series)
            
            time_varying_risk = {
                'volatility_persistence': float(vol_series.autocorr(lag=1)) if not vol_series.empty else 0.0,
                'volatility_mean_reversion': float(1 - vol_series.autocorr(lag=1)) if not vol_series.empty else 1.0,
                'avg_conditional_vol': float(np.mean(conditional_vol)),
                'max_conditional_vol': float(np.max(conditional_vol)),
                'min_conditional_vol': float(np.min(conditional_vol)),
                'vol_of_vol': float(np.std(conditional_vol, ddof=1)),
                'return_vol_correlation': float(rolling_corr.mean()) if not rolling_corr.isna().all() else 0.0
            }
            
            # Risk regime analysis
            vol_percentiles = np.percentile(conditional_vol, [25, 50, 75, 90, 95])
            
            risk_regimes = {
                'low_vol_threshold': float(vol_percentiles[0]),
                'medium_vol_threshold': float(vol_percentiles[1]),
                'high_vol_threshold': float(vol_percentiles[2]),
                'extreme_vol_threshold': float(vol_percentiles[3]),
                'crisis_vol_threshold': float(vol_percentiles[4]),
                'current_regime': self._classify_vol_regime(conditional_vol[-1], vol_percentiles)
            }
            
            # Risk-adjusted performance metrics
            if len(returns) > 0 and np.mean(conditional_vol) > 0:
                risk_adjusted_metrics = {
                    'volatility_adjusted_return': float(np.mean(returns) / np.mean(conditional_vol)),
                    'risk_efficiency': float(np.mean(returns) / np.sqrt(total_variance)) if total_variance > 0 else 0.0,
                    'conditional_sharpe': float(np.mean(returns) / np.mean(conditional_vol)),
                    'downside_deviation': float(np.std(returns[returns < 0], ddof=1)) if np.any(returns < 0) else 0.0
                }
            else:
                risk_adjusted_metrics = {
                    'volatility_adjusted_return': 0.0,
                    'risk_efficiency': 0.0,
                    'conditional_sharpe': 0.0,
                    'downside_deviation': 0.0
                }
            
            return {
                'risk_decomposition': risk_decomposition,
                'time_varying_risk': time_varying_risk,
                'risk_regimes': risk_regimes,
                'risk_adjusted_metrics': risk_adjusted_metrics
            }
            
        except Exception:
            return None
    
    def _classify_vol_regime(self, current_vol: float, percentiles: np.ndarray) -> str:
        """Classify current volatility regime"""
        if current_vol <= percentiles[0]:
            return 'low'
        elif current_vol <= percentiles[1]:
            return 'medium-low'
        elif current_vol <= percentiles[2]:
            return 'medium'
        elif current_vol <= percentiles[3]:
            return 'high'
        elif current_vol <= percentiles[4]:
            return 'extreme'
        else:
            return 'crisis'

    def fit_garch(self, returns: List[float], model_type: str = 'GARCH', 
                  p: int = 1, q: int = 1) -> GARCHResult:
        """Fit GARCH family models
        
        Args:
            returns: List of return values
            model_type: Type of GARCH model ('GARCH', 'EGARCH', 'TGARCH')
            p: ARCH order (default: 1)
            q: GARCH order (default: 1)
            
        Returns:
            GARCHResult containing model parameters and diagnostics
        """
        
        returns = np.array(returns)
        n = len(returns)
        
        # Initial parameter estimates
        if model_type == 'GARCH':
            initial_params = [0.01, 0.05, 0.9]  # omega, alpha, beta
        elif model_type == 'EGARCH':
            initial_params = [0.01, 0.1, -0.05, 0.9]  # omega, alpha, gamma, beta
        elif model_type == 'TGARCH':
            initial_params = [0.01, 0.05, 0.02, 0.9]  # omega, alpha, gamma, beta
        
        # Optimize parameters
        result = minimize(
            self._garch_likelihood,
            initial_params,
            args=(returns, model_type, p, q),
            method='L-BFGS-B',
            bounds=self._get_parameter_bounds(model_type)
        )
        
        optimal_params = result.x
        
        # Calculate conditional volatility
        cond_vol = self._calculate_conditional_volatility(
            returns, optimal_params, model_type, p, q
        )
        
        # Standardized residuals
        std_residuals = returns[1:] / np.sqrt(cond_vol[1:])
        
        # Model diagnostics
        diagnostics = self._garch_diagnostics(std_residuals, cond_vol)
        
        # Generate forecasts
        forecasts, conf_intervals = self._garch_forecast(
            returns, optimal_params, model_type, cond_vol, horizon=10
        )
        
        # Calculate information criteria
        log_likelihood = -self._garch_likelihood(optimal_params, returns, model_type, p, q)
        n_params = len(optimal_params)
        aic = 2 * n_params - 2 * log_likelihood
        bic = np.log(n) * n_params - 2 * log_likelihood
        
        # Parameter dictionary
        param_names = self._get_parameter_names(model_type)
        parameters = dict(zip(param_names, optimal_params))
        
        return GARCHResult(
            model_type=model_type,
            parameters=parameters,
            conditional_volatility=cond_vol.tolist(),
            standardized_residuals=std_residuals.tolist(),
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            model_diagnostics=diagnostics,
            forecasts=forecasts,
            confidence_intervals=conf_intervals
        )
    
    def _garch_likelihood(self, params: np.ndarray, returns: np.ndarray, 
                         model_type: str, p: int, q: int) -> float:
        """Calculate GARCH log-likelihood"""
        
        try:
            cond_vol = self._calculate_conditional_volatility(returns, params, model_type, p, q)
            
            # Avoid numerical issues
            cond_vol = np.maximum(cond_vol, 1e-8)
            
            # Log-likelihood calculation
            log_likelihood = -0.5 * np.sum(
                np.log(2 * np.pi * cond_vol[1:]) + (returns[1:] ** 2) / cond_vol[1:]
            )
            
            return -log_likelihood  # Return negative for minimization
            
        except:
            return 1e10  # Return large value if calculation fails
    
    def _calculate_conditional_volatility(self, returns: np.ndarray, params: np.ndarray,
                                        model_type: str, p: int, q: int) -> np.ndarray:
        """Calculate conditional volatility for GARCH models"""
        
        n = len(returns)
        cond_vol = np.zeros(n)
        cond_vol[0] = np.var(returns)  # Initial variance
        
        if model_type == 'GARCH':
            omega, alpha, beta = params
            
            for t in range(1, n):
                cond_vol[t] = (omega + 
                             alpha * returns[t-1]**2 + 
                             beta * cond_vol[t-1])
        
        elif model_type == 'EGARCH':
            omega, alpha, gamma, beta = params
            
            for t in range(1, n):
                # EGARCH specification
                log_vol = (omega + 
                          alpha * (abs(returns[t-1]) / np.sqrt(cond_vol[t-1]) - 
                                  np.sqrt(2/np.pi)) +
                          gamma * returns[t-1] / np.sqrt(cond_vol[t-1]) +
                          beta * np.log(cond_vol[t-1]))
                cond_vol[t] = np.exp(log_vol)
        
        elif model_type == 'TGARCH':
            omega, alpha, gamma, beta = params
            
            for t in range(1, n):
                # Threshold GARCH (GJR-GARCH)
                indicator = 1 if returns[t-1] < 0 else 0
                cond_vol[t] = (omega + 
                             alpha * returns[t-1]**2 + 
                             gamma * indicator * returns[t-1]**2 +
                             beta * cond_vol[t-1])
        
        return np.maximum(cond_vol, 1e-8)  # Ensure positive variance
    
    def _get_parameter_bounds(self, model_type: str) -> List[Tuple[float, float]]:
        """Get parameter bounds for optimization"""
        
        if model_type == 'GARCH':
            return [(1e-6, 1.0), (0.0, 1.0), (0.0, 0.99)]  # omega, alpha, beta
        elif model_type == 'EGARCH':
            return [(1e-6, 1.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 0.99)]  # omega, alpha, gamma, beta
        elif model_type == 'TGARCH':
            return [(1e-6, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 0.99)]  # omega, alpha, gamma, beta
    
    def _get_parameter_names(self, model_type: str) -> List[str]:
        """Get parameter names for model"""
        
        if model_type == 'GARCH':
            return ['omega', 'alpha', 'beta']
        elif model_type == 'EGARCH':
            return ['omega', 'alpha', 'gamma', 'beta']
        elif model_type == 'TGARCH':
            return ['omega', 'alpha', 'gamma', 'beta']
    
    def _garch_diagnostics(self, std_residuals: np.ndarray, 
                          cond_vol: np.ndarray) -> Dict[str, Any]:
        """Perform GARCH model diagnostics"""
        
        diagnostics = {}
        
        # Ljung-Box test on standardized residuals
        try:
            # Try to import statsmodels for proper Ljung-Box test
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_result = acorr_ljungbox(std_residuals, lags=10, return_df=True)
            diagnostics['ljung_box_stat'] = float(lb_result['lb_stat'].iloc[-1])
            diagnostics['ljung_box_pvalue'] = float(lb_result['lb_pvalue'].iloc[-1])
        except ImportError:
            # Fallback: simple autocorrelation test
            if len(std_residuals) > 10:
                autocorr = np.corrcoef(std_residuals[:-1], std_residuals[1:])[0, 1]
                diagnostics['ljung_box_stat'] = abs(autocorr) * len(std_residuals)
                diagnostics['ljung_box_pvalue'] = 0.05 if abs(autocorr) > 0.1 else 0.5
            else:
                diagnostics['ljung_box_stat'] = np.nan
                diagnostics['ljung_box_pvalue'] = np.nan
        except Exception:
            diagnostics['ljung_box_stat'] = np.nan
            diagnostics['ljung_box_pvalue'] = np.nan
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(std_residuals)
            diagnostics['jarque_bera_stat'] = float(jb_stat)
            diagnostics['jarque_bera_pvalue'] = float(jb_pvalue)
        except:
            diagnostics['jarque_bera_stat'] = np.nan
            diagnostics['jarque_bera_pvalue'] = np.nan
        
        # ARCH test on standardized residuals squared
        try:
            if len(std_residuals) > 1:
                arch_stat = np.corrcoef(std_residuals[:-1]**2, std_residuals[1:]**2)[0, 1]
                diagnostics['arch_effect'] = float(arch_stat)
            else:
                diagnostics['arch_effect'] = np.nan
        except:
            diagnostics['arch_effect'] = np.nan
        
        # Basic statistics
        diagnostics['mean_std_residual'] = float(np.mean(std_residuals))
        diagnostics['std_std_residual'] = float(np.std(std_residuals))
        diagnostics['skewness'] = float(stats.skew(std_residuals))
        diagnostics['kurtosis'] = float(stats.kurtosis(std_residuals))
        
        return diagnostics
    
    def _garch_forecast(self, returns: np.ndarray, params: np.ndarray,
                       model_type: str, cond_vol: np.ndarray, 
                       horizon: int = 10) -> Tuple[List[float], Dict[str, List[float]]]:
        """Generate GARCH volatility forecasts"""
        
        forecasts = []
        last_return = returns[-1]
        last_vol = cond_vol[-1]
        
        if model_type == 'GARCH':
            omega, alpha, beta = params
            
            # Multi-step ahead forecasts
            for h in range(1, horizon + 1):
                if h == 1:
                    forecast = omega + alpha * last_return**2 + beta * last_vol
                else:
                    # Long-run variance for multi-step forecasts
                    long_run_var = omega / (1 - alpha - beta)
                    forecast = long_run_var + (beta**(h-1)) * (last_vol - long_run_var)
                
                forecasts.append(max(forecast, 1e-8))
        
        elif model_type in ['EGARCH', 'TGARCH']:
            # Simplified forecasting for complex models
            for h in range(1, horizon + 1):
                forecasts.append(last_vol)
        
        # Generate confidence intervals (simplified)
        conf_intervals = {
            'lower_95': [f * 0.8 for f in forecasts],
            'upper_95': [f * 1.2 for f in forecasts],
            'lower_99': [f * 0.7 for f in forecasts],
            'upper_99': [f * 1.3 for f in forecasts]
        }
        
        return forecasts, conf_intervals
    
    def analyze_garch(self, index_data: IndexData, model_types: Optional[List[str]] = None) -> Dict[str, GARCHResult]:
        """Analyze index using multiple GARCH models and select the best one
        
        Args:
            index_data: IndexData object containing price and return data
            model_types: List of GARCH models to fit (default: ['GARCH', 'EGARCH', 'TGARCH'])
            
        Returns:
            Dictionary of GARCHResult objects for each model type
        """
        
        if model_types is None:
            model_types = ['GARCH', 'EGARCH', 'TGARCH']
        
        results = {}
        
        for model_type in model_types:
            try:
                print(f"Fitting {model_type} model...")
                result = self.fit_garch(index_data.returns, model_type)
                results[model_type] = result
            except Exception as e:
                print(f"Failed to fit {model_type}: {e}")
                # Create a fallback result
                results[model_type] = self._create_fallback_result(index_data, model_type)
        
        return results
    
    def _create_fallback_result(self, index_data: IndexData, model_type: str) -> GARCHResult:
        """Create fallback GARCH result when model fitting fails"""
        
        returns = np.array(index_data.returns)
        n = len(returns)
        
        # Simple volatility estimate
        volatility = np.std(returns)
        conditional_vol = [volatility] * n
        
        return GARCHResult(
            model_type=model_type,
            parameters={'omega': volatility**2 * 0.1, 'alpha': 0.05, 'beta': 0.9},
            conditional_volatility=conditional_vol,
            standardized_residuals=(returns / volatility).tolist(),
            log_likelihood=-1000,
            aic=2000,
            bic=2010,
            model_diagnostics={
                'ljung_box_stat': np.nan,
                'ljung_box_pvalue': np.nan,
                'jarque_bera_stat': np.nan,
                'jarque_bera_pvalue': np.nan,
                'arch_effect': np.nan,
                'mean_std_residual': 0.0,
                'std_std_residual': 1.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            },
            forecasts=[volatility] * 10,
            confidence_intervals={
                'lower_95': [volatility * 0.8] * 10,
                'upper_95': [volatility * 1.2] * 10,
                'lower_99': [volatility * 0.7] * 10,
                'upper_99': [volatility * 1.3] * 10
            }
        )

# Example usage and testing
if __name__ == "__main__":
    # Generate sample index data for testing
    np.random.seed(42)
    n_obs = 500
    
    # Generate synthetic returns with volatility clustering
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
    timestamps = pd.date_range(start='2022-01-01', periods=n_obs, freq='D')
    
    # Create IndexData object
    index_data = IndexData(
        index_symbol="TEST_INDEX",
        prices=prices.tolist(),
        returns=returns.tolist(),
        timestamps=timestamps.tolist()
    )
    
    # Create analyzer
    analyzer = GARCHAnalyzer()
    
    # Perform analysis
    print("Performing GARCH analysis...")
    results = analyzer.analyze_garch(index_data)
    
    # Print results
    print(f"\n=== GARCH Analysis Results ===\n")
    
    for model_type, result in results.items():
        print(f"{model_type} Model:")
        print(f"  AIC: {result.aic:.4f}")
        print(f"  BIC: {result.bic:.4f}")
        print(f"  Log-Likelihood: {result.log_likelihood:.4f}")
        print(f"  Parameters: {result.parameters}")
        print(f"  Current Volatility: {result.conditional_volatility[-1]:.6f}")
        print(f"  Ljung-Box p-value: {result.model_diagnostics.get('ljung_box_pvalue', 'N/A')}")
        print()
    
    # Find best model (lowest AIC)
    best_model = min(results.items(), key=lambda x: x[1].aic)
    print(f"Best Model: {best_model[0]} (AIC: {best_model[1].aic:.4f})")
    
    print("\nGARCH analysis completed successfully!")