import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# Optional advanced dependencies
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class RegimeAnalysisERP:
    """Regime analysis results for Exchange Rate Pressure."""
    current_regime: int
    regime_probabilities: pd.Series
    regime_description: Dict[int, str]
    transition_matrix: np.ndarray
    regime_persistence: Dict[int, float]
    expected_regime_duration: Dict[int, float]
    regime_volatility: Dict[int, float]
    
@dataclass
class PredictiveModelingERP:
    """Predictive modeling results for Exchange Rate Pressure."""
    model_predictions: Dict[str, np.ndarray]
    model_scores: Dict[str, float]
    best_model: str
    feature_importance: Dict[str, float]
    forecast_horizon: int
    prediction_intervals: Dict[str, Tuple[np.ndarray, np.ndarray]]
    ensemble_prediction: np.ndarray
    
@dataclass
class VolatilityAnalysisERP:
    """Volatility analysis results for Exchange Rate Pressure."""
    garch_params: Dict[str, float]
    volatility_forecast: np.ndarray
    volatility_clustering: bool
    arch_test_pvalue: float
    conditional_volatility: pd.Series
    
@dataclass
class ExchangeRatePressureResult:
    """Exchange rate pressure analysis result."""
    pressure_index: pd.Series
    pressure_components: Dict[str, pd.Series]
    crisis_probability: pd.Series
    pressure_threshold: float
    crisis_periods: List[Tuple[datetime, datetime]]
    early_warning_signals: Dict[str, Any]
    regime_analysis: Optional[RegimeAnalysisERP] = None
    predictive_modeling: Optional[PredictiveModelingERP] = None
    volatility_analysis: Optional[VolatilityAnalysisERP] = None
    statistical_tests: Dict[str, Any] = field(default_factory=dict)
    model_diagnostics: Dict[str, Any] = field(default_factory=dict)

class ExchangeRatePressureAnalyzer:
    """Exchange rate pressure analysis implementation."""
    
    def __init__(self, enable_regime_switching: bool = True,
                 enable_predictive_modeling: bool = True,
                 enable_volatility_analysis: bool = True,
                 forecast_horizon: int = 12):
        self.weights = {'exchange_rate': 0.4, 'interest_rate': 0.3, 'reserves': 0.3}
        self.enable_regime_switching = enable_regime_switching and (HMM_AVAILABLE or STATSMODELS_AVAILABLE)
        self.enable_predictive_modeling = enable_predictive_modeling
        self.enable_volatility_analysis = enable_volatility_analysis and ARCH_AVAILABLE
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        
    def calculate_pressure_index(self, exchange_rates: pd.Series,
                                interest_rates: pd.Series,
                                reserves: pd.Series,
                                weights: Dict[str, float] = None) -> ExchangeRatePressureResult:
        """Calculate exchange rate pressure index."""
        try:
            if weights:
                self.weights = weights
                
            # Align data
            data = pd.DataFrame({
                'exchange_rate': exchange_rates,
                'interest_rate': interest_rates,
                'reserves': reserves
            }).dropna()
            
            if len(data) < 10:
                return self._create_empty_pressure_result()
                
            # Calculate pressure components
            components = self._calculate_pressure_components(data)
            
            # Calculate composite pressure index
            pressure_index = self._calculate_composite_index(components)
            
            # Determine pressure threshold
            threshold = self._determine_pressure_threshold(pressure_index)
            
            # Identify crisis periods
            crisis_periods = self._identify_crisis_periods(pressure_index, threshold)
            
            # Calculate crisis probability
            crisis_probability = self._calculate_crisis_probability(pressure_index, threshold)
            
            # Generate early warning signals
            early_warning = self._generate_early_warning_signals(pressure_index, components)
            
            # Advanced analyses
            regime_analysis = None
            predictive_modeling = None
            volatility_analysis = None
            statistical_tests = {}
            model_diagnostics = {}
            
            if len(data) >= 50:  # Minimum data requirement for advanced analyses
                # Statistical tests
                statistical_tests = self._perform_statistical_tests(pressure_index)
                
                # Regime switching analysis
                if self.enable_regime_switching:
                    regime_analysis = self._analyze_regime_switching(pressure_index)
                    
                # Volatility analysis
                if self.enable_volatility_analysis:
                    volatility_analysis = self._analyze_volatility(pressure_index)
                    
                # Predictive modeling
                if self.enable_predictive_modeling:
                    predictive_modeling = self._build_predictive_models(data, pressure_index)
                    
                # Model diagnostics
                model_diagnostics = self._perform_model_diagnostics(pressure_index, components)
            
            return ExchangeRatePressureResult(
                pressure_index=pressure_index,
                pressure_components=components,
                crisis_probability=crisis_probability,
                pressure_threshold=threshold,
                crisis_periods=crisis_periods,
                early_warning_signals=early_warning,
                regime_analysis=regime_analysis,
                predictive_modeling=predictive_modeling,
                volatility_analysis=volatility_analysis,
                statistical_tests=statistical_tests,
                model_diagnostics=model_diagnostics
            )
            
        except Exception as e:
            return self._create_empty_pressure_result()
            
    def _calculate_pressure_components(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate individual pressure components."""
        components = {}
        
        # Exchange rate pressure (depreciation pressure)
        er_returns = data['exchange_rate'].pct_change()
        er_pressure = (er_returns - er_returns.mean()) / er_returns.std()
        components['exchange_rate_pressure'] = er_pressure
        
        # Interest rate pressure (defensive rate increases)
        ir_changes = data['interest_rate'].diff()
        ir_pressure = (ir_changes - ir_changes.mean()) / ir_changes.std()
        components['interest_rate_pressure'] = ir_pressure
        
        # Reserve pressure (reserve losses)
        reserve_changes = data['reserves'].pct_change()
        reserve_pressure = -(reserve_changes - reserve_changes.mean()) / reserve_changes.std()
        components['reserve_pressure'] = reserve_pressure
        
        return components
        
    def _calculate_composite_index(self, components: Dict[str, pd.Series]) -> pd.Series:
        """Calculate composite pressure index."""
        try:
            # Weighted average of components
            pressure_index = (self.weights['exchange_rate'] * components['exchange_rate_pressure'] +
                            self.weights['interest_rate'] * components['interest_rate_pressure'] +
                            self.weights['reserves'] * components['reserve_pressure'])
            
            return pressure_index.fillna(0)
        except:
            return pd.Series(dtype=float)
            
    def _determine_pressure_threshold(self, pressure_index: pd.Series) -> float:
        """Determine pressure threshold for crisis identification."""
        try:
            # Use 90th percentile as threshold
            return pressure_index.quantile(0.9)
        except:
            return 1.5
            
    def _identify_crisis_periods(self, pressure_index: pd.Series, 
                                threshold: float) -> List[Tuple[datetime, datetime]]:
        """Identify crisis periods based on pressure threshold."""
        try:
            crisis_periods = []
            in_crisis = False
            crisis_start = None
            
            for date, pressure in pressure_index.items():
                if pressure > threshold and not in_crisis:
                    # Start of crisis period
                    in_crisis = True
                    crisis_start = date
                elif pressure <= threshold and in_crisis:
                    # End of crisis period
                    in_crisis = False
                    if crisis_start:
                        crisis_periods.append((crisis_start, date))
                        
            # Handle case where crisis continues to end of data
            if in_crisis and crisis_start:
                crisis_periods.append((crisis_start, pressure_index.index[-1]))
                
            return crisis_periods
        except:
            return []
            
    def _calculate_crisis_probability(self, pressure_index: pd.Series,
                                     threshold: float) -> pd.Series:
        """Calculate rolling crisis probability."""
        try:
            # Logistic transformation of pressure index
            normalized_pressure = (pressure_index - pressure_index.mean()) / pressure_index.std()
            crisis_probability = 1 / (1 + np.exp(-normalized_pressure))
            
            return crisis_probability
        except:
            return pd.Series(dtype=float)
            
    def _generate_early_warning_signals(self, pressure_index: pd.Series,
                                       components: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Generate early warning signals."""
        try:
            current_pressure = pressure_index.iloc[-1] if len(pressure_index) > 0 else 0
            recent_trend = pressure_index.tail(6).mean() if len(pressure_index) >= 6 else 0
            
            # Warning levels
            if current_pressure > pressure_index.quantile(0.9):
                warning_level = 'high'
            elif current_pressure > pressure_index.quantile(0.75):
                warning_level = 'moderate'
            else:
                warning_level = 'low'
                
            return {
                'current_pressure': current_pressure,
                'recent_trend': recent_trend,
                'warning_level': warning_level,
                'component_contributions': {
                    'exchange_rate': components['exchange_rate_pressure'].iloc[-1] if len(components['exchange_rate_pressure']) > 0 else 0,
                    'interest_rate': components['interest_rate_pressure'].iloc[-1] if len(components['interest_rate_pressure']) > 0 else 0,
                    'reserves': components['reserve_pressure'].iloc[-1] if len(components['reserve_pressure']) > 0 else 0
                }
            }
        except:
            return {'warning_level': 'unknown'}
            
    def _perform_statistical_tests(self, pressure_index: pd.Series) -> Dict[str, Any]:
        """Perform statistical tests on pressure index."""
        try:
            tests = {}
            
            # Stationarity tests
            if STATSMODELS_AVAILABLE:
                # Augmented Dickey-Fuller test
                adf_result = adfuller(pressure_index.dropna())
                tests['adf_test'] = {
                    'statistic': adf_result[0],
                    'pvalue': adf_result[1],
                    'is_stationary': adf_result[1] < 0.05
                }
                
                # KPSS test
                kpss_result = kpss(pressure_index.dropna())
                tests['kpss_test'] = {
                    'statistic': kpss_result[0],
                    'pvalue': kpss_result[1],
                    'is_stationary': kpss_result[1] > 0.05
                }
                
                # Ljung-Box test for autocorrelation
                lb_result = acorr_ljungbox(pressure_index.dropna(), lags=10, return_df=True)
                tests['ljung_box_test'] = {
                    'pvalue': lb_result['lb_pvalue'].iloc[-1],
                    'has_autocorrelation': lb_result['lb_pvalue'].iloc[-1] < 0.05
                }
            
            # Normality test
            shapiro_stat, shapiro_p = stats.shapiro(pressure_index.dropna())
            tests['normality_test'] = {
                'statistic': shapiro_stat,
                'pvalue': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
            
            # Descriptive statistics
            tests['descriptive_stats'] = {
                'mean': pressure_index.mean(),
                'std': pressure_index.std(),
                'skewness': stats.skew(pressure_index.dropna()),
                'kurtosis': stats.kurtosis(pressure_index.dropna()),
                'jarque_bera_pvalue': stats.jarque_bera(pressure_index.dropna())[1]
            }
            
            return tests
        except Exception as e:
             return {'error': str(e)}
    
    def _analyze_regime_switching(self, pressure_index: pd.Series) -> Optional[RegimeAnalysisERP]:
        """Analyze regime switching in pressure index."""
        try:
            data = pressure_index.dropna()
            if len(data) < 50:
                return None
                
            # Try Hidden Markov Model first
            if HMM_AVAILABLE:
                # Prepare data for HMM
                X = data.values.reshape(-1, 1)
                
                # Fit 2-regime HMM
                model = hmm.GaussianHMM(n_components=2, covariance_type="full", random_state=42)
                model.fit(X)
                
                # Get regime probabilities and states
                regime_probs = model.predict_proba(X)
                regime_states = model.predict(X)
                
                # Create regime probabilities series
                regime_prob_series = pd.Series(
                    regime_probs[:, 1],  # Probability of high-pressure regime
                    index=data.index
                )
                
                current_regime = regime_states[-1]
                
                # Calculate transition matrix
                transition_matrix = model.transmat_
                
                # Calculate regime persistence and expected duration
                regime_persistence = {
                    0: transition_matrix[0, 0],
                    1: transition_matrix[1, 1]
                }
                
                expected_duration = {
                    0: 1 / (1 - transition_matrix[0, 0]) if transition_matrix[0, 0] < 1 else np.inf,
                    1: 1 / (1 - transition_matrix[1, 1]) if transition_matrix[1, 1] < 1 else np.inf
                }
                
                # Calculate regime volatility
                regime_volatility = {}
                for regime in [0, 1]:
                    regime_data = data[regime_states == regime]
                    regime_volatility[regime] = regime_data.std() if len(regime_data) > 1 else 0.0
                
                regime_description = {
                    0: "Low Pressure Regime",
                    1: "High Pressure Regime"
                }
                
            elif STATSMODELS_AVAILABLE:
                # Use Markov Regression as fallback
                try:
                    model = MarkovRegression(
                        data.values,
                        k_regimes=2,
                        trend='c',
                        switching_variance=True
                    )
                    results = model.fit()
                    
                    # Get regime probabilities
                    regime_prob_series = pd.Series(
                        results.smoothed_marginal_probabilities[1],
                        index=data.index
                    )
                    
                    current_regime = int(regime_prob_series.iloc[-1] > 0.5)
                    
                    # Approximate transition matrix
                    transition_matrix = np.array([
                        [results.params['p[0->0]'], 1 - results.params['p[0->0]']],
                        [1 - results.params['p[1->1]'], results.params['p[1->1]']]
                    ])
                    
                    regime_persistence = {
                        0: transition_matrix[0, 0],
                        1: transition_matrix[1, 1]
                    }
                    
                    expected_duration = {
                        0: 1 / (1 - transition_matrix[0, 0]) if transition_matrix[0, 0] < 1 else np.inf,
                        1: 1 / (1 - transition_matrix[1, 1]) if transition_matrix[1, 1] < 1 else np.inf
                    }
                    
                    # Estimate regime volatility
                    regime_volatility = {
                        0: np.sqrt(results.params['sigma2[0]']),
                        1: np.sqrt(results.params['sigma2[1]'])
                    }
                    
                    regime_description = {
                        0: "Low Pressure Regime",
                        1: "High Pressure Regime"
                    }
                    
                except Exception:
                    return None
            else:
                return None
                
            return RegimeAnalysisERP(
                current_regime=current_regime,
                regime_probabilities=regime_prob_series,
                regime_description=regime_description,
                transition_matrix=transition_matrix,
                regime_persistence=regime_persistence,
                expected_regime_duration=expected_duration,
                regime_volatility=regime_volatility
            )
            
        except Exception as e:
             return None
    
    def _analyze_volatility(self, pressure_index: pd.Series) -> Optional[VolatilityAnalysisERP]:
        """Analyze volatility patterns in pressure index."""
        try:
            if not ARCH_AVAILABLE:
                return None
                
            data = pressure_index.dropna()
            if len(data) < 50:
                return None
                
            # Fit GARCH(1,1) model
            model = arch_model(data, vol='Garch', p=1, q=1)
            fitted_model = model.fit(disp='off')
            
            # Extract GARCH parameters
            garch_params = {
                'omega': fitted_model.params['omega'],
                'alpha': fitted_model.params['alpha[1]'],
                'beta': fitted_model.params['beta[1]']
            }
            
            # Generate volatility forecast
            forecast = fitted_model.forecast(horizon=self.forecast_horizon)
            volatility_forecast = np.sqrt(forecast.variance.values[-1, :])
            
            # Get conditional volatility
            conditional_volatility = pd.Series(
                fitted_model.conditional_volatility,
                index=data.index
            )
            
            # Test for ARCH effects
            arch_test = fitted_model.arch_lm_test(lags=5)
            arch_test_pvalue = arch_test.pvalue
            
            # Check for volatility clustering
            volatility_clustering = arch_test_pvalue < 0.05
            
            return VolatilityAnalysisERP(
                garch_params=garch_params,
                volatility_forecast=volatility_forecast,
                volatility_clustering=volatility_clustering,
                arch_test_pvalue=arch_test_pvalue,
                conditional_volatility=conditional_volatility
            )
            
        except Exception as e:
             return None
    
    def _build_predictive_models(self, data: pd.DataFrame, pressure_index: pd.Series) -> Optional[PredictiveModelingERP]:
        """Build predictive models for pressure index forecasting."""
        try:
            # Prepare features and target
            features_df = data.copy()
            
            # Create lagged features
            for lag in [1, 2, 3, 6, 12]:
                features_df[f'pressure_lag_{lag}'] = pressure_index.shift(lag)
                features_df[f'er_lag_{lag}'] = features_df['exchange_rate'].shift(lag)
                features_df[f'ir_lag_{lag}'] = features_df['interest_rate'].shift(lag)
                features_df[f'res_lag_{lag}'] = features_df['reserves'].shift(lag)
            
            # Create moving averages
            for window in [3, 6, 12]:
                features_df[f'pressure_ma_{window}'] = pressure_index.rolling(window).mean()
                features_df[f'er_ma_{window}'] = features_df['exchange_rate'].rolling(window).mean()
            
            # Create volatility features
            features_df['pressure_volatility'] = pressure_index.rolling(12).std()
            features_df['er_volatility'] = features_df['exchange_rate'].pct_change().rolling(12).std()
            
            # Drop rows with NaN values
            features_df = features_df.dropna()
            
            if len(features_df) < 30:
                return None
            
            # Prepare target variable (future pressure index)
            target = pressure_index.shift(-1).dropna()
            
            # Align features and target
            common_index = features_df.index.intersection(target.index)
            X = features_df.loc[common_index]
            y = target.loc[common_index]
            
            if len(X) < 20:
                return None
            
            # Split data for time series validation
            split_point = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
            y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Initialize models
            models = {
                'linear_regression': LinearRegression(),
                'ridge': Ridge(alpha=1.0),
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'svr': SVR(kernel='rbf'),
                'mlp': MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
            }
            
            # Add XGBoost if available
            if XGB_AVAILABLE:
                models['xgboost'] = xgb.XGBRegressor(n_estimators=100, random_state=42)
            
            # Train and evaluate models
            model_predictions = {}
            model_scores = {}
            feature_importance = {}
            
            for name, model in models.items():
                try:
                    # Train model
                    if name in ['linear_regression', 'ridge', 'svr', 'mlp']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    model_predictions[name] = y_pred
                    model_scores[name] = r2_score(y_test, y_pred)
                    
                    # Extract feature importance
                    if hasattr(model, 'feature_importances_'):
                        importance = dict(zip(X.columns, model.feature_importances_))
                        feature_importance[name] = importance
                    elif hasattr(model, 'coef_'):
                        importance = dict(zip(X.columns, np.abs(model.coef_)))
                        feature_importance[name] = importance
                        
                except Exception:
                    continue
            
            if not model_scores:
                return None
            
            # Find best model
            best_model = max(model_scores.keys(), key=lambda k: model_scores[k])
            
            # Create ensemble prediction (simple average)
            ensemble_pred = np.mean(list(model_predictions.values()), axis=0)
            
            # Calculate prediction intervals (using ensemble std)
            pred_std = np.std(list(model_predictions.values()), axis=0)
            prediction_intervals = {
                '95%': (ensemble_pred - 1.96 * pred_std, ensemble_pred + 1.96 * pred_std),
                '80%': (ensemble_pred - 1.28 * pred_std, ensemble_pred + 1.28 * pred_std)
            }
            
            # Aggregate feature importance
            if feature_importance:
                avg_importance = {}
                for feature in X.columns:
                    importances = [imp.get(feature, 0) for imp in feature_importance.values()]
                    avg_importance[feature] = np.mean(importances)
            else:
                avg_importance = {}
            
            return PredictiveModelingERP(
                model_predictions=model_predictions,
                model_scores=model_scores,
                best_model=best_model,
                feature_importance=avg_importance,
                forecast_horizon=len(y_test),
                prediction_intervals=prediction_intervals,
                ensemble_prediction=ensemble_pred
            )
            
        except Exception as e:
             return None
    
    def _perform_model_diagnostics(self, pressure_index: pd.Series, 
                                  components: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Perform comprehensive model diagnostics."""
        try:
            diagnostics = {}
            
            # Component correlation analysis
            component_df = pd.DataFrame(components)
            correlation_matrix = component_df.corr()
            diagnostics['component_correlations'] = correlation_matrix.to_dict()
            
            # Pressure index stability analysis
            rolling_mean = pressure_index.rolling(window=12).mean()
            rolling_std = pressure_index.rolling(window=12).std()
            
            diagnostics['stability_metrics'] = {
                'mean_stability': rolling_mean.std(),
                'volatility_stability': rolling_std.std(),
                'coefficient_of_variation': pressure_index.std() / abs(pressure_index.mean()) if pressure_index.mean() != 0 else np.inf
            }
            
            # Outlier detection
            q1 = pressure_index.quantile(0.25)
            q3 = pressure_index.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = pressure_index[(pressure_index < lower_bound) | (pressure_index > upper_bound)]
            diagnostics['outlier_analysis'] = {
                'outlier_count': len(outliers),
                'outlier_percentage': len(outliers) / len(pressure_index) * 100,
                'outlier_dates': outliers.index.tolist()
            }
            
            # Trend analysis
            from scipy.stats import linregress
            x_values = np.arange(len(pressure_index))
            slope, intercept, r_value, p_value, std_err = linregress(x_values, pressure_index.values)
            
            diagnostics['trend_analysis'] = {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_significance': p_value < 0.05
            }
            
            # Seasonality detection (if enough data)
            if len(pressure_index) >= 24:
                try:
                    # Simple seasonal decomposition
                    monthly_means = pressure_index.groupby(pressure_index.index.month).mean()
                    seasonal_variation = monthly_means.std()
                    
                    diagnostics['seasonality'] = {
                        'seasonal_variation': seasonal_variation,
                        'monthly_patterns': monthly_means.to_dict()
                    }
                except Exception:
                    diagnostics['seasonality'] = {'error': 'Could not detect seasonality'}
            
            # Model performance summary
            diagnostics['performance_summary'] = {
                'data_points': len(pressure_index),
                'missing_values': pressure_index.isna().sum(),
                'data_quality_score': (len(pressure_index) - pressure_index.isna().sum()) / len(pressure_index)
            }
            
            return diagnostics
            
        except Exception as e:
            return {'error': str(e)}
    
    def _create_empty_pressure_result(self) -> ExchangeRatePressureResult:
        """Create empty pressure result for error cases."""
        return ExchangeRatePressureResult(
            pressure_index=pd.Series(dtype=float),
            pressure_components={},
            crisis_probability=pd.Series(dtype=float),
            pressure_threshold=0.0,
            crisis_periods=[],
            early_warning_signals={'warning_level': 'unknown'}
        )