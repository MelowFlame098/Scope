import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import DBSCAN

# Advanced libraries with availability checks
try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from arch import arch_model
    from arch.unitroot import ADF, KPSS
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

try:
    import statsmodels.api as sm
    from statsmodels.tsa.regime_switching import markov_regression
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.stattools import jarque_bera
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class CurrentAccountData:
    """Current account components data."""
    trade_balance: pd.Series  # Exports - Imports
    services_balance: pd.Series  # Services exports - imports
    primary_income: pd.Series  # Investment income, compensation
    secondary_income: pd.Series  # Transfers, remittances
    current_account_balance: pd.Series  # Total current account

@dataclass
class RegimeAnalysisCA:
    """Regime analysis results for Current Account."""
    current_regime: int
    regime_probabilities: np.ndarray
    regime_description: Dict[int, str]
    transition_matrix: np.ndarray
    regime_persistence: Dict[int, float]
    expected_regime_duration: Dict[int, float]
    regime_volatility: Dict[int, float]

@dataclass
class AnomalyDetectionCA:
    """Anomaly detection results for Current Account."""
    anomaly_scores: pd.Series
    anomaly_threshold: float
    detected_anomalies: pd.Series
    anomaly_periods: List[datetime]
    anomaly_severity: Dict[str, float]
    isolation_forest_scores: pd.Series
    statistical_outliers: pd.Series

@dataclass
class PredictiveModelingCA:
    """Predictive modeling results for Current Account."""
    model_scores: Dict[str, float]
    best_model: str
    predictions: np.ndarray
    prediction_intervals: Dict[str, np.ndarray]
    feature_importance: Dict[str, float]
    ensemble_prediction: np.ndarray
    forecast_horizon: int

@dataclass
class VolatilityAnalysisCA:
    """Volatility analysis results for Current Account."""
    garch_params: Dict[str, float]
    volatility_forecast: Union[float, np.ndarray]
    volatility_clustering: bool
    arch_test_pvalue: float
    conditional_volatility: pd.Series
    volatility_regimes: pd.Series

@dataclass
class EconometricAnalysisCA:
    """Econometric analysis results for Current Account."""
    cointegration_tests: Dict[str, Any]
    granger_causality: Dict[str, Any]
    impulse_responses: Dict[str, np.ndarray]
    variance_decomposition: Dict[str, Dict[str, float]]
    structural_breaks: Dict[str, Any]
    unit_root_tests: Dict[str, Any]

class CurrentAccountAnalyzer:
    """Enhanced Current account analysis implementation with advanced econometric features."""
    
    def __init__(self, 
                 enable_regime_switching: bool = True,
                 enable_anomaly_detection: bool = True,
                 enable_predictive_modeling: bool = True,
                 enable_volatility_analysis: bool = True,
                 enable_econometric_analysis: bool = True,
                 forecast_horizon: int = 12):
        self.data = None
        self.gdp_data = None
        self.enable_regime_switching = enable_regime_switching and (HMM_AVAILABLE or STATSMODELS_AVAILABLE)
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_predictive_modeling = enable_predictive_modeling
        self.enable_volatility_analysis = enable_volatility_analysis and ARCH_AVAILABLE
        self.enable_econometric_analysis = enable_econometric_analysis and STATSMODELS_AVAILABLE
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        
    def analyze_current_account(self, ca_data: CurrentAccountData, 
                               gdp_data: pd.Series = None,
                               exchange_rates: pd.Series = None) -> Dict[str, Any]:
        """Analyze current account components and sustainability with advanced features."""
        self.data = ca_data
        self.gdp_data = gdp_data
        
        # Check data availability
        if len(ca_data.current_account_balance.dropna()) < 10:
            return self._create_empty_result()
        
        # Basic statistics
        basic_stats = self._calculate_basic_statistics()
        
        # Trend analysis
        trend_analysis = self._analyze_trends()
        
        # Cyclical patterns
        cyclical_analysis = self._analyze_cyclical_patterns()
        
        # Sustainability analysis
        sustainability = self._analyze_sustainability(gdp_data)
        
        # Basic volatility analysis
        volatility = self._analyze_volatility()
        
        # Correlation analysis
        correlations = self._analyze_correlations(exchange_rates)
        
        # Statistical tests
        statistical_tests = self._perform_statistical_tests()
        
        # Advanced analyses (conditional on availability and settings)
        regime_analysis = None
        if self.enable_regime_switching and len(ca_data.current_account_balance.dropna()) >= 50:
            regime_analysis = self._analyze_regime_switching()
        
        anomaly_detection = None
        if self.enable_anomaly_detection and len(ca_data.current_account_balance.dropna()) >= 30:
            anomaly_detection = self._detect_anomalies()
        
        volatility_analysis = None
        if self.enable_volatility_analysis and len(ca_data.current_account_balance.dropna()) >= 50:
            volatility_analysis = self._analyze_advanced_volatility()
        
        predictive_modeling = None
        if self.enable_predictive_modeling and len(ca_data.current_account_balance.dropna()) >= 50:
            predictive_modeling = self._build_predictive_models(gdp_data, exchange_rates)
        
        econometric_analysis = None
        if self.enable_econometric_analysis and len(ca_data.current_account_balance.dropna()) >= 50:
            econometric_analysis = self._perform_econometric_analysis(gdp_data, exchange_rates)
        
        # Model diagnostics
        model_diagnostics = self._perform_model_diagnostics()
        
        return {
            'basic_statistics': basic_stats,
            'trend_analysis': trend_analysis,
            'cyclical_patterns': cyclical_analysis,
            'sustainability': sustainability,
            'volatility': volatility,
            'correlations': correlations,
            'statistical_tests': statistical_tests,
            'regime_analysis': regime_analysis,
            'anomaly_detection': anomaly_detection,
            'volatility_analysis': volatility_analysis,
            'predictive_modeling': predictive_modeling,
            'econometric_analysis': econometric_analysis,
            'model_diagnostics': model_diagnostics
        }
        
    def _calculate_basic_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistics for current account components."""
        return {
            'trade_balance': self.data.trade_balance.describe().to_dict(),
            'services_balance': self.data.services_balance.describe().to_dict(),
            'primary_income': self.data.primary_income.describe().to_dict(),
            'secondary_income': self.data.secondary_income.describe().to_dict(),
            'current_account_balance': self.data.current_account_balance.describe().to_dict()
        }
        
    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze trends in current account components."""
        return {
            'trade_balance_trend': self._calculate_trend(self.data.trade_balance),
            'services_balance_trend': self._calculate_trend(self.data.services_balance),
            'primary_income_trend': self._calculate_trend(self.data.primary_income),
            'secondary_income_trend': self._calculate_trend(self.data.secondary_income),
            'current_account_trend': self._calculate_trend(self.data.current_account_balance),
            'trend_interpretation': {
                component: 'improving' if trend > 0.1 else 'deteriorating' if trend < -0.1 else 'stable'
                for component, trend in {
                    'trade_balance': self._calculate_trend(self.data.trade_balance),
                    'services_balance': self._calculate_trend(self.data.services_balance),
                    'primary_income': self._calculate_trend(self.data.primary_income),
                    'secondary_income': self._calculate_trend(self.data.secondary_income),
                    'current_account': self._calculate_trend(self.data.current_account_balance)
                }.items()
            }
        }
        
    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate trend using linear regression."""
        try:
            if len(series.dropna()) < 2:
                return 0.0
            
            x = np.arange(len(series))
            y = series.values
            
            # Remove NaN values
            mask = ~np.isnan(y)
            if np.sum(mask) < 2:
                return 0.0
                
            slope, _, _, _, _ = stats.linregress(x[mask], y[mask])
            return slope
        except:
            return 0.0
            
    def _analyze_cyclical_patterns(self) -> Dict[str, Any]:
        """Analyze cyclical patterns in current account."""
        try:
            # Calculate rolling correlations and seasonal patterns
            ca_balance = self.data.current_account_balance.dropna()
            
            if len(ca_balance) < 12:
                return {'seasonal_pattern': 'insufficient_data'}
                
            # Simple seasonal analysis
            monthly_avg = ca_balance.groupby(ca_balance.index.month).mean()
            seasonal_volatility = ca_balance.groupby(ca_balance.index.month).std()
            
            return {
                'seasonal_pattern': monthly_avg.to_dict(),
                'seasonal_volatility': seasonal_volatility.to_dict(),
                'peak_month': monthly_avg.idxmax(),
                'trough_month': monthly_avg.idxmin()
            }
        except:
            return {'seasonal_pattern': 'analysis_failed'}
            
    def _analyze_sustainability(self, gdp_data: pd.Series = None) -> Dict[str, Any]:
        """Analyze current account sustainability."""
        try:
            ca_balance = self.data.current_account_balance
            
            if gdp_data is not None and len(gdp_data) == len(ca_balance):
                # Calculate CA/GDP ratio
                ca_gdp_ratio = (ca_balance / gdp_data) * 100
                
                # Sustainability thresholds (IMF guidelines)
                deficit_threshold = -5.0  # CA deficit > 5% of GDP is concerning
                surplus_threshold = 10.0  # CA surplus > 10% of GDP may indicate imbalances
                
                current_ratio = ca_gdp_ratio.iloc[-1] if len(ca_gdp_ratio) > 0 else 0
                avg_ratio = ca_gdp_ratio.mean()
                
                # Risk assessment
                if current_ratio < deficit_threshold:
                    risk_level = 'high'
                elif current_ratio > surplus_threshold:
                    risk_level = 'moderate'
                else:
                    risk_level = 'low'
                    
                return {
                    'ca_gdp_ratio': {
                        'current': current_ratio,
                        'average': avg_ratio,
                        'trend': self._calculate_trend(ca_gdp_ratio)
                    },
                    'risk_level': risk_level,
                    'sustainability_score': max(0, min(100, 50 + (current_ratio / deficit_threshold) * 25))
                }
            else:
                # Basic sustainability analysis without GDP data
                ca_trend = self._calculate_trend(ca_balance)
                ca_volatility = ca_balance.std()
                
                return {
                    'trend_sustainability': 'improving' if ca_trend > 0 else 'deteriorating',
                    'volatility_level': 'high' if ca_volatility > ca_balance.abs().mean() else 'moderate',
                    'risk_level': 'unknown_without_gdp'
                }
        except:
            return {'sustainability_analysis': 'failed'}
            
    def _analyze_volatility(self) -> Dict[str, Any]:
        """Analyze volatility patterns in current account."""
        ca_balance = self.data.current_account_balance
        ca_returns = ca_balance.pct_change().dropna()
        
        return {
            'volatility': ca_returns.std(),
            'max_drawdown': self._calculate_max_drawdown(ca_balance),
            'volatility_regime': self._classify_volatility_regime(ca_returns),
            'volatility_components': {
                'trade_balance': self.data.trade_balance.pct_change().std(),
                'services_balance': self.data.services_balance.pct_change().std(),
                'primary_income': self.data.primary_income.pct_change().std(),
                'secondary_income': self.data.secondary_income.pct_change().std()
            }
        }
        
    def _calculate_max_drawdown(self, series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = series.cumsum()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max.abs()
            return drawdown.min()
        except:
            return 0.0
            
    def _classify_volatility_regime(self, returns: pd.Series) -> str:
        """Classify volatility regime."""
        try:
            if len(returns) < 10:
                return 'insufficient_data'
                
            volatility = returns.std()
            mean_abs_return = returns.abs().mean()
            
            if volatility > 2 * mean_abs_return:
                return 'high_volatility'
            elif volatility > mean_abs_return:
                return 'moderate_volatility'
            else:
                return 'low_volatility'
        except:
            return 'unknown'
            
    def _analyze_correlations(self, exchange_rates: pd.Series = None) -> Dict[str, Any]:
        """Analyze correlations between current account components."""
        try:
            # Internal correlations
            components = pd.DataFrame({
                'trade_balance': self.data.trade_balance,
                'services_balance': self.data.services_balance,
                'primary_income': self.data.primary_income,
                'secondary_income': self.data.secondary_income,
                'current_account': self.data.current_account_balance
            })
            
            correlation_matrix = components.corr()
            
            result = {
                'internal_correlations': correlation_matrix.to_dict(),
                'strongest_correlation': {
                    'components': None,
                    'value': 0.0
                }
            }
            
            # Find strongest correlation (excluding self-correlations)
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = abs(correlation_matrix.iloc[i, j])
                    if corr_val > result['strongest_correlation']['value']:
                        result['strongest_correlation'] = {
                            'components': (correlation_matrix.columns[i], correlation_matrix.columns[j]),
                            'value': correlation_matrix.iloc[i, j]
                        }
            
            # Exchange rate correlation if available
            if exchange_rates is not None and len(exchange_rates) == len(self.data.current_account_balance):
                fx_corr = self.data.current_account_balance.corr(exchange_rates)
                result['fx_correlation'] = {
                    'value': fx_corr,
                    'interpretation': self._interpret_fx_correlation(fx_corr)
                }
                
            return result
        except:
            return {'correlation_analysis': 'failed'}
            
    def _interpret_fx_correlation(self, correlation: float) -> str:
        """Interpret FX correlation."""
        if abs(correlation) < 0.3:
            return 'weak'
        elif abs(correlation) < 0.7:
            return 'moderate'
        else:
            return 'strong'
            
    def _detect_structural_breaks(self) -> Dict[str, Any]:
        """Detect structural breaks in current account."""
        try:
            ca_balance = self.data.current_account_balance.dropna()
            
            if len(ca_balance) < 20:
                return {'structural_breaks': 'insufficient_data'}
            
            # Simple structural break detection using rolling statistics
            window = min(12, len(ca_balance) // 4)
            rolling_mean = ca_balance.rolling(window=window).mean()
            rolling_std = ca_balance.rolling(window=window).std()
            
            # Detect significant changes in mean
            mean_changes = rolling_mean.diff().abs()
            std_changes = rolling_std.diff().abs()
            
            # Identify potential break points
            mean_threshold = mean_changes.quantile(0.9)
            std_threshold = std_changes.quantile(0.9)
            
            break_points = []
            for i in range(len(mean_changes)):
                if (mean_changes.iloc[i] > mean_threshold or 
                    std_changes.iloc[i] > std_threshold):
                    break_points.append(ca_balance.index[i])
            
            return {
                'potential_breaks': break_points[:5],  # Limit to top 5
                'break_count': len(break_points),
                'structural_stability': 'stable' if len(break_points) < 3 else 'unstable'
            }
        except:
            return {'structural_breaks': 'analysis_failed'}
    
    def _perform_statistical_tests(self) -> Dict[str, Any]:
        """Perform comprehensive statistical tests on current account data."""
        try:
            ca_balance = self.data.current_account_balance.dropna()
            
            if len(ca_balance) < 10:
                return {'error': 'insufficient_data'}
            
            tests = {}
            
            # Stationarity tests
            if STATSMODELS_AVAILABLE:
                try:
                    # ADF test
                    adf_result = adfuller(ca_balance)
                    tests['adf_test'] = {
                        'statistic': adf_result[0],
                        'pvalue': adf_result[1],
                        'critical_values': adf_result[4],
                        'is_stationary': adf_result[1] < 0.05
                    }
                    
                    # KPSS test
                    kpss_result = kpss(ca_balance)
                    tests['kpss_test'] = {
                        'statistic': kpss_result[0],
                        'pvalue': kpss_result[1],
                        'critical_values': kpss_result[3],
                        'is_stationary': kpss_result[1] > 0.05
                    }
                except:
                    pass
            
            # Normality tests
            try:
                shapiro_stat, shapiro_p = stats.shapiro(ca_balance)
                tests['normality_test'] = {
                    'statistic': shapiro_stat,
                    'pvalue': shapiro_p,
                    'is_normal': shapiro_p > 0.05
                }
            except:
                pass
            
            # Autocorrelation test
            if STATSMODELS_AVAILABLE and len(ca_balance) > 20:
                try:
                    ljung_box = acorr_ljungbox(ca_balance, lags=min(10, len(ca_balance)//4), return_df=True)
                    tests['autocorr_test'] = {
                        'ljung_box_stat': ljung_box['lb_stat'].iloc[-1],
                        'ljung_box_pvalue': ljung_box['lb_pvalue'].iloc[-1],
                        'has_autocorr': ljung_box['lb_pvalue'].iloc[-1] < 0.05
                    }
                except:
                    pass
            
            # Descriptive statistics
            tests['descriptive_stats'] = {
                'mean': float(ca_balance.mean()),
                'std': float(ca_balance.std()),
                'skewness': float(stats.skew(ca_balance)),
                'kurtosis': float(stats.kurtosis(ca_balance)),
                'jarque_bera_stat': float(stats.jarque_bera(ca_balance)[0]),
                'jarque_bera_pvalue': float(stats.jarque_bera(ca_balance)[1])
            }
            
            return tests
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_regime_switching(self) -> Optional[RegimeAnalysisCA]:
        """Analyze regime switching in current account balance."""
        try:
            ca_balance = self.data.current_account_balance.dropna()
            
            if len(ca_balance) < 50:
                return None
            
            # Prepare data
            ca_returns = ca_balance.pct_change().dropna()
            data_array = ca_returns.values.reshape(-1, 1)
            
            # Try HMM first
            if HMM_AVAILABLE:
                try:
                    # Fit 2-regime Gaussian HMM
                    model = hmm.GaussianHMM(n_components=2, covariance_type="full", random_state=42)
                    model.fit(data_array)
                    
                    # Get regime probabilities and states
                    regime_probs = model.predict_proba(data_array)
                    regime_states = model.predict(data_array)
                    
                    # Calculate transition matrix
                    transition_matrix = model.transmat_
                    
                    # Calculate regime persistence and expected duration
                    regime_persistence = {}
                    expected_duration = {}
                    regime_volatility = {}
                    
                    for i in range(2):
                        persistence = transition_matrix[i, i]
                        regime_persistence[i] = persistence
                        expected_duration[i] = 1 / (1 - persistence) if persistence < 1 else float('inf')
                        
                        # Calculate regime-specific volatility
                        regime_mask = regime_states == i
                        if np.sum(regime_mask) > 1:
                            regime_volatility[i] = np.std(ca_returns.values[regime_mask])
                        else:
                            regime_volatility[i] = 0.0
                    
                    # Regime descriptions
                    regime_description = {
                        0: "Low Volatility Regime" if regime_volatility.get(0, 0) < regime_volatility.get(1, 0) else "High Volatility Regime",
                        1: "High Volatility Regime" if regime_volatility.get(0, 0) < regime_volatility.get(1, 0) else "Low Volatility Regime"
                    }
                    
                    current_regime = regime_states[-1]
                    
                    return RegimeAnalysisCA(
                        current_regime=current_regime,
                        regime_probabilities=regime_probs,
                        regime_description=regime_description,
                        transition_matrix=transition_matrix,
                        regime_persistence=regime_persistence,
                        expected_regime_duration=expected_duration,
                        regime_volatility=regime_volatility
                    )
                    
                except Exception:
                    pass
            
            # Fallback to statsmodels Markov regression
            if STATSMODELS_AVAILABLE:
                try:
                    # Prepare data for Markov regression
                    y = ca_returns.values
                    
                    # Fit Markov switching model
                    model = markov_regression.MarkovRegression(
                        y, k_regimes=2, trend='c', switching_variance=True
                    )
                    results = model.fit()
                    
                    # Get regime probabilities
                    regime_probs = results.smoothed_marginal_probabilities
                    regime_states = np.argmax(regime_probs, axis=1)
                    
                    # Calculate transition matrix
                    transition_matrix = results.regime_transition
                    
                    # Calculate metrics
                    regime_persistence = {}
                    expected_duration = {}
                    regime_volatility = {}
                    
                    for i in range(2):
                        persistence = transition_matrix[i, i]
                        regime_persistence[i] = persistence
                        expected_duration[i] = 1 / (1 - persistence) if persistence < 1 else float('inf')
                        
                        # Regime-specific volatility
                        regime_mask = regime_states == i
                        if np.sum(regime_mask) > 1:
                            regime_volatility[i] = np.std(y[regime_mask])
                        else:
                            regime_volatility[i] = 0.0
                    
                    regime_description = {
                        0: "Stable Regime" if regime_volatility.get(0, 0) < regime_volatility.get(1, 0) else "Volatile Regime",
                        1: "Volatile Regime" if regime_volatility.get(0, 0) < regime_volatility.get(1, 0) else "Stable Regime"
                    }
                    
                    current_regime = regime_states[-1]
                    
                    return RegimeAnalysisCA(
                        current_regime=current_regime,
                        regime_probabilities=regime_probs,
                        regime_description=regime_description,
                        transition_matrix=transition_matrix,
                        regime_persistence=regime_persistence,
                        expected_regime_duration=expected_duration,
                        regime_volatility=regime_volatility
                    )
                    
                except Exception:
                    pass
            
            return None
            
        except Exception:
            return None
    
    def _detect_anomalies(self) -> Optional[AnomalyDetectionCA]:
        """Detect anomalies in current account balance using multiple methods."""
        try:
            ca_balance = self.data.current_account_balance.dropna()
            
            if len(ca_balance) < 30:
                return None
            
            # Statistical outlier detection (Z-score method)
            z_scores = np.abs(stats.zscore(ca_balance))
            statistical_threshold = 2.5
            statistical_outliers = pd.Series(z_scores > statistical_threshold, index=ca_balance.index)
            
            # Isolation Forest for anomaly detection
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            ca_reshaped = ca_balance.values.reshape(-1, 1)
            isolation_scores = isolation_forest.decision_function(ca_reshaped)
            isolation_outliers = isolation_forest.predict(ca_reshaped) == -1
            
            # Combined anomaly scores
            combined_scores = (z_scores / statistical_threshold + 
                             np.abs(isolation_scores) / np.max(np.abs(isolation_scores))) / 2
            
            # Dynamic threshold based on data distribution
            anomaly_threshold = np.percentile(combined_scores, 90)
            
            # Detected anomalies
            detected_anomalies = pd.Series(combined_scores > anomaly_threshold, index=ca_balance.index)
            anomaly_periods = ca_balance.index[detected_anomalies].tolist()
            
            # Anomaly severity classification
            anomaly_severity = {
                'mild': np.sum((combined_scores > anomaly_threshold) & (combined_scores <= np.percentile(combined_scores, 95))),
                'moderate': np.sum((combined_scores > np.percentile(combined_scores, 95)) & (combined_scores <= np.percentile(combined_scores, 99))),
                'severe': np.sum(combined_scores > np.percentile(combined_scores, 99))
            }
            
            return AnomalyDetectionCA(
                anomaly_scores=pd.Series(combined_scores, index=ca_balance.index),
                anomaly_threshold=anomaly_threshold,
                detected_anomalies=detected_anomalies,
                anomaly_periods=anomaly_periods,
                anomaly_severity=anomaly_severity,
                isolation_forest_scores=pd.Series(isolation_scores, index=ca_balance.index),
                statistical_outliers=statistical_outliers
            )
            
        except Exception:
            return None
    
    def _analyze_advanced_volatility(self) -> Optional[VolatilityAnalysisCA]:
        """Analyze volatility using GARCH models and advanced techniques."""
        try:
            if not ARCH_AVAILABLE:
                return None
            
            ca_balance = self.data.current_account_balance.dropna()
            ca_returns = ca_balance.pct_change().dropna() * 100  # Convert to percentage
            
            if len(ca_returns) < 50:
                return None
            
            # Fit GARCH(1,1) model
            garch_model = arch_model(ca_returns, vol='Garch', p=1, q=1)
            garch_results = garch_model.fit(disp='off')
            
            # Extract GARCH parameters
            garch_params = {
                'omega': float(garch_results.params['omega']),
                'alpha': float(garch_results.params['alpha[1]']),
                'beta': float(garch_results.params['beta[1]'])
            }
            
            # Generate volatility forecast
            volatility_forecast = garch_results.forecast(horizon=self.forecast_horizon)
            forecast_values = volatility_forecast.variance.iloc[-1].values
            
            # Conditional volatility
            conditional_volatility = pd.Series(
                garch_results.conditional_volatility, 
                index=ca_returns.index
            )
            
            # Test for ARCH effects
            arch_test = garch_results.arch_lm_test(lags=5)
            arch_test_pvalue = arch_test.pvalue
            
            # Volatility clustering test
            volatility_clustering = arch_test_pvalue < 0.05
            
            # Volatility regimes (simple threshold-based)
            vol_median = conditional_volatility.median()
            volatility_regimes = pd.Series(
                (conditional_volatility > vol_median).astype(int),
                index=conditional_volatility.index
            )
            
            return VolatilityAnalysisCA(
                garch_params=garch_params,
                volatility_forecast=forecast_values,
                volatility_clustering=volatility_clustering,
                arch_test_pvalue=arch_test_pvalue,
                conditional_volatility=conditional_volatility,
                volatility_regimes=volatility_regimes
            )
            
        except Exception:
            return None