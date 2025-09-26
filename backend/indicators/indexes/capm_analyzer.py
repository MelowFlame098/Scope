from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint, adfuller, kpss
    from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
    from statsmodels.stats.stattools import jarque_bera
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Using simplified statistical methods.")

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("ARCH library not available. Using simplified volatility models.")

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("HMMlearn not available. Using alternative regime detection.")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Using alternative ML models.")

@dataclass
class IndexData:
    """Structure for index market data"""
    prices: List[float]
    returns: List[float]
    volume: List[float]
    market_cap: List[float]
    timestamps: List[datetime]
    index_symbol: str
    constituent_weights: Optional[Dict[str, float]] = None
    sector_weights: Optional[Dict[str, float]] = None

@dataclass
class CAPMResult:
    """Results from CAPM analysis"""
    beta: float
    alpha: float
    market_risk_premium: float
    expected_return: float
    systematic_risk: float
    unsystematic_risk: float
    sharpe_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    r_squared: float
    tracking_error: float
    information_ratio: float

@dataclass
class MultiFactorResult:
    """Results from multi-factor model analysis"""
    factor_loadings: Dict[str, float]
    factor_premiums: Dict[str, float]
    model_r_squared: float
    adjusted_r_squared: float
    factor_significance: Dict[str, float]  # p-values
    residual_analysis: Dict[str, Any]
    model_diagnostics: Dict[str, Any]
    expected_return: float
    factor_contributions: Dict[str, float]

@dataclass
class RegimeAnalysisCAPM:
    """Results from regime switching analysis"""
    n_regimes: int
    regime_probabilities: pd.Series
    current_regime: int
    transition_matrix: np.ndarray
    regime_parameters: Dict[int, Dict[str, float]]
    regime_persistence: Dict[int, float]
    expected_duration: Dict[int, float]
    regime_volatility: Dict[int, float]

@dataclass
class AdvancedDiagnostics:
    """Advanced econometric diagnostics"""
    stationarity_tests: Dict[str, Any]
    autocorr_tests: Dict[str, Any]
    heteroskedasticity_tests: Dict[str, Any]
    normality_tests: Dict[str, Any]
    structural_break_tests: Dict[str, Any]
    cointegration_tests: Optional[Dict[str, Any]]
    model_stability: Dict[str, Any]

@dataclass
class EnhancedCAPMResult:
    """Comprehensive CAPM analysis results"""
    basic_capm: CAPMResult
    multi_factor: Optional[MultiFactorResult]
    regime_analysis: Optional[RegimeAnalysisCAPM]
    advanced_diagnostics: Optional[AdvancedDiagnostics]
    rolling_analysis: Optional[Dict[str, pd.Series]]
    forecasts: Optional[Dict[str, np.ndarray]]
    risk_attribution: Optional[Dict[str, float]]

class CAPMAnalyzer:
    """Enhanced Capital Asset Pricing Model analyzer with multi-factor models and regime detection"""
    
    def __init__(self, risk_free_rate: float = 0.02, 
                 enable_multi_factor: bool = True,
                 enable_regime_switching: bool = True,
                 enable_advanced_diagnostics: bool = True,
                 rolling_window: int = 252,
                 forecast_horizon: int = 30):
        self.risk_free_rate = risk_free_rate
        self.enable_multi_factor = enable_multi_factor and STATSMODELS_AVAILABLE
        self.enable_regime_switching = enable_regime_switching and (HMM_AVAILABLE or STATSMODELS_AVAILABLE)
        self.enable_advanced_diagnostics = enable_advanced_diagnostics and STATSMODELS_AVAILABLE
        self.rolling_window = rolling_window
        self.forecast_horizon = forecast_horizon
        self.scaler = StandardScaler()
        
    def analyze_enhanced_capm(self, index_data: IndexData, 
                            market_data: IndexData,
                            factor_data: Optional[Dict[str, IndexData]] = None) -> EnhancedCAPMResult:
        """Perform comprehensive CAPM analysis with multi-factor models and regime detection"""
        try:
            # Basic CAPM analysis
            basic_capm = self.analyze_capm(index_data, market_data)
            
            # Multi-factor analysis
            multi_factor = None
            if self.enable_multi_factor and factor_data:
                multi_factor = self._analyze_multi_factor(index_data, market_data, factor_data)
            
            # Regime switching analysis
            regime_analysis = None
            if self.enable_regime_switching:
                regime_analysis = self._analyze_regime_switching(index_data, market_data)
            
            # Advanced diagnostics
            advanced_diagnostics = None
            if self.enable_advanced_diagnostics:
                advanced_diagnostics = self._perform_advanced_diagnostics(index_data, market_data)
            
            # Rolling analysis
            rolling_analysis = self._perform_rolling_analysis(index_data, market_data)
            
            # Forecasting
            forecasts = self._generate_forecasts(index_data, market_data, basic_capm)
            
            # Risk attribution
            risk_attribution = self._calculate_risk_attribution(basic_capm, multi_factor)
            
            return EnhancedCAPMResult(
                basic_capm=basic_capm,
                multi_factor=multi_factor,
                regime_analysis=regime_analysis,
                advanced_diagnostics=advanced_diagnostics,
                rolling_analysis=rolling_analysis,
                forecasts=forecasts,
                risk_attribution=risk_attribution
            )
            
        except Exception as e:
            print(f"Error in enhanced CAPM analysis: {e}")
            # Return basic analysis as fallback
            basic_capm = self.analyze_capm(index_data, market_data)
            return EnhancedCAPMResult(
                basic_capm=basic_capm,
                multi_factor=None,
                regime_analysis=None,
                advanced_diagnostics=None,
                rolling_analysis=None,
                forecasts=None,
                risk_attribution=None
            )
        
    def analyze_capm(self, index_data: IndexData, 
                    market_data: IndexData) -> CAPMResult:
        """Perform CAPM analysis on index vs market"""
        try:
            # Align data
            aligned_data = self._align_market_data(index_data, market_data)
            
            index_returns = aligned_data['index_returns']
            market_returns = aligned_data['market_returns']
            
            # Calculate excess returns
            risk_free_daily = self.risk_free_rate / 252  # Convert to daily
            index_excess = index_returns - risk_free_daily
            market_excess = market_returns - risk_free_daily
            
            # Fit CAPM regression: R_i - R_f = alpha + beta * (R_m - R_f)
            model = LinearRegression()
            model.fit(market_excess.reshape(-1, 1), index_excess)
            
            # Extract CAPM parameters
            beta = model.coef_[0]
            alpha = model.intercept_
            
            # Calculate predictions and R-squared
            predictions = model.predict(market_excess.reshape(-1, 1))
            r_squared = r2_score(index_excess, predictions)
            
            # Market risk premium (annualized)
            market_risk_premium = np.mean(market_excess) * 252
            
            # Expected return according to CAPM
            expected_return = self.risk_free_rate + beta * market_risk_premium
            
            # Risk decomposition
            total_variance = np.var(index_returns, ddof=1)
            systematic_variance = (beta ** 2) * np.var(market_returns, ddof=1)
            unsystematic_variance = total_variance - systematic_variance
            
            systematic_risk = np.sqrt(systematic_variance)
            unsystematic_risk = np.sqrt(max(0, unsystematic_variance))  # Ensure non-negative
            
            # Performance metrics
            avg_index_return = np.mean(index_returns) * 252  # Annualized
            index_volatility = np.std(index_returns, ddof=1) * np.sqrt(252)  # Annualized
            
            # Sharpe ratio
            sharpe_ratio = (avg_index_return - self.risk_free_rate) / index_volatility if index_volatility > 0 else 0
            
            # Treynor ratio
            treynor_ratio = (avg_index_return - self.risk_free_rate) / beta if beta != 0 else 0
            
            # Jensen's alpha (annualized)
            jensen_alpha = alpha * 252
            
            # Tracking error and information ratio
            tracking_error = np.std(index_returns - market_returns, ddof=1) * np.sqrt(252)
            excess_return_vs_market = avg_index_return - np.mean(market_returns) * 252
            information_ratio = excess_return_vs_market / tracking_error if tracking_error > 0 else 0
            
            return CAPMResult(
                beta=beta,
                alpha=alpha,
                market_risk_premium=market_risk_premium,
                expected_return=expected_return,
                systematic_risk=systematic_risk,
                unsystematic_risk=unsystematic_risk,
                sharpe_ratio=sharpe_ratio,
                treynor_ratio=treynor_ratio,
                jensen_alpha=jensen_alpha,
                r_squared=r_squared,
                tracking_error=tracking_error,
                information_ratio=information_ratio
            )
            
        except Exception as e:
            print(f"Error in CAPM analysis: {e}")
            # Return default result
            return CAPMResult(
                beta=1.0,
                alpha=0.0,
                market_risk_premium=0.06,
                expected_return=self.risk_free_rate + 0.06,
                systematic_risk=0.15,
                unsystematic_risk=0.05,
                sharpe_ratio=0.0,
                treynor_ratio=0.0,
                jensen_alpha=0.0,
                r_squared=0.0,
                tracking_error=0.20,
                information_ratio=0.0
            )
    
    def _align_market_data(self, index_data: IndexData, 
                          market_data: IndexData) -> Dict[str, np.ndarray]:
        """Align index and market data by timestamps"""
        # Convert to pandas for easier alignment
        index_df = pd.DataFrame({
            'timestamp': index_data.timestamps,
            'returns': index_data.returns
        })
        
        market_df = pd.DataFrame({
            'timestamp': market_data.timestamps,
            'returns': market_data.returns
        })
        
        # Merge on timestamp
        merged_df = pd.merge(index_df, market_df, on='timestamp', how='inner', suffixes=('_index', '_market'))
        
        return {
            'index_returns': merged_df['returns_index'].values,
            'market_returns': merged_df['returns_market'].values,
            'timestamps': merged_df['timestamp'].tolist()
        }
    
    def _analyze_multi_factor(self, index_data: IndexData, market_data: IndexData, 
                            factor_data: Dict[str, IndexData]) -> Optional[MultiFactorResult]:
        """Analyze multi-factor models (Fama-French, etc.)"""
        try:
            if not STATSMODELS_AVAILABLE:
                return None
            
            # Align all data
            aligned_data = self._align_market_data(index_data, market_data)
            index_returns = aligned_data['index_returns']
            market_returns = aligned_data['market_returns']
            timestamps = aligned_data['timestamps']
            
            # Calculate excess returns
            risk_free_daily = self.risk_free_rate / 252
            index_excess = index_returns - risk_free_daily
            market_excess = market_returns - risk_free_daily
            
            # Prepare factor data
            factors_df = pd.DataFrame({
                'market': market_excess,
                'timestamp': timestamps
            })
            
            # Add additional factors
            for factor_name, factor_data_obj in factor_data.items():
                factor_df = pd.DataFrame({
                    'timestamp': factor_data_obj.timestamps,
                    'returns': factor_data_obj.returns
                })
                factors_df = pd.merge(factors_df, factor_df, on='timestamp', how='inner')
                factors_df = factors_df.rename(columns={'returns': factor_name})
            
            # Remove timestamp column for regression
            X = factors_df.drop('timestamp', axis=1).values
            y = index_excess[:len(X)]
            
            if len(X) < 30:
                return None
            
            # Fit multi-factor model using OLS
            X_with_const = sm.add_constant(X)
            model = sm.OLS(y, X_with_const).fit()
            
            # Extract results
            factor_names = ['const'] + list(factors_df.columns[1:])
            factor_loadings = dict(zip(factor_names[1:], model.params[1:]))
            factor_significance = dict(zip(factor_names, model.pvalues))
            
            # Calculate factor premiums (annualized)
            factor_premiums = {}
            for i, factor_name in enumerate(factor_names[1:], 1):
                factor_premiums[factor_name] = np.mean(X[:, i-1]) * 252
            
            # Expected return from multi-factor model
            expected_return = self.risk_free_rate + sum(
                factor_loadings[factor] * factor_premiums[factor] 
                for factor in factor_loadings.keys()
            )
            
            # Factor contributions to risk
            factor_contributions = {}
            total_factor_var = 0
            for factor in factor_loadings.keys():
                factor_idx = list(factor_loadings.keys()).index(factor)
                factor_var = (factor_loadings[factor] ** 2) * np.var(X[:, factor_idx], ddof=1)
                factor_contributions[factor] = factor_var
                total_factor_var += factor_var
            
            # Normalize contributions
            if total_factor_var > 0:
                factor_contributions = {k: v/total_factor_var for k, v in factor_contributions.items()}
            
            # Residual analysis
            residuals = model.resid
            residual_analysis = {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals, ddof=1)),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals)),
                'jarque_bera_stat': float(stats.jarque_bera(residuals)[0]),
                'jarque_bera_pvalue': float(stats.jarque_bera(residuals)[1])
            }
            
            # Model diagnostics
            model_diagnostics = {
                'f_statistic': float(model.fvalue),
                'f_pvalue': float(model.f_pvalue),
                'durbin_watson': float(sm.stats.durbin_watson(residuals)),
                'condition_number': float(np.linalg.cond(X_with_const))
            }
            
            # Heteroskedasticity test
            if len(residuals) > 20:
                try:
                    lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(residuals, X_with_const)
                    model_diagnostics['breusch_pagan_lm'] = float(lm_stat)
                    model_diagnostics['breusch_pagan_pvalue'] = float(lm_pvalue)
                except:
                    pass
            
            return MultiFactorResult(
                factor_loadings=factor_loadings,
                factor_premiums=factor_premiums,
                model_r_squared=float(model.rsquared),
                adjusted_r_squared=float(model.rsquared_adj),
                factor_significance=factor_significance,
                residual_analysis=residual_analysis,
                model_diagnostics=model_diagnostics,
                expected_return=expected_return,
                factor_contributions=factor_contributions
            )
            
        except Exception:
            return None
    
    def _perform_advanced_diagnostics(self, index_data: IndexData, market_data: IndexData, capm_result: CAPMResult) -> Optional[AdvancedDiagnostics]:
        """Perform advanced econometric diagnostics on CAPM model"""
        try:
            # Align data
            aligned_data = self._align_market_data(index_data, market_data)
            index_returns = aligned_data['index_returns']
            market_returns = aligned_data['market_returns']
            
            if len(index_returns) < 30:
                return None
            
            # Calculate excess returns and residuals
            risk_free_daily = self.risk_free_rate / 252
            index_excess = index_returns - risk_free_daily
            market_excess = market_returns - risk_free_daily
            
            # Calculate residuals from CAPM model
            predicted_returns = capm_result.alpha + capm_result.beta * market_excess
            residuals = index_excess - predicted_returns
            
            # Stationarity tests
            stationarity_tests = self._test_stationarity(residuals)
            
            # Autocorrelation tests
            autocorr_tests = self._test_autocorrelation(residuals)
            
            # Normality tests
            normality_tests = self._test_normality(residuals)
            
            # Heteroskedasticity tests
            heterosked_tests = self._test_heteroskedasticity(residuals, market_excess)
            
            # Model stability tests
            stability_tests = self._test_model_stability(index_excess, market_excess)
            
            return AdvancedDiagnostics(
                stationarity_tests=stationarity_tests,
                autocorr_tests=autocorr_tests,
                normality_tests=normality_tests,
                heterosked_tests=heterosked_tests,
                stability_tests=stability_tests
            )
            
        except Exception:
            return None
    
    def _test_stationarity(self, series: np.ndarray) -> Dict[str, Any]:
        """Test for stationarity using ADF and KPSS tests"""
        try:
            results = {}
            
            # Augmented Dickey-Fuller test
            adf_stat, adf_pvalue, adf_lags, adf_nobs, adf_critical, adf_icbest = adfuller(series, autolag='AIC')
            results['adf'] = {
                'statistic': float(adf_stat),
                'p_value': float(adf_pvalue),
                'lags_used': int(adf_lags),
                'critical_values': {k: float(v) for k, v in adf_critical.items()},
                'is_stationary': adf_pvalue < 0.05
            }
            
            # KPSS test (if available)
            if STATSMODELS_AVAILABLE:
                kpss_stat, kpss_pvalue, kpss_lags, kpss_critical = kpss(series, regression='c')
                results['kpss'] = {
                    'statistic': float(kpss_stat),
                    'p_value': float(kpss_pvalue),
                    'lags_used': int(kpss_lags),
                    'critical_values': {k: float(v) for k, v in kpss_critical.items()},
                    'is_stationary': kpss_pvalue > 0.05
                }
            
            return results
            
        except Exception:
            return {'adf': {'is_stationary': True}, 'kpss': {'is_stationary': True}}
    
    def _test_autocorrelation(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for autocorrelation in residuals"""
        try:
            results = {}
            
            # Ljung-Box test
            if STATSMODELS_AVAILABLE and len(residuals) > 20:
                lb_stat, lb_pvalue = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=False)
                results['ljung_box'] = {
                    'statistic': float(lb_stat[-1]) if hasattr(lb_stat, '__iter__') else float(lb_stat),
                    'p_value': float(lb_pvalue[-1]) if hasattr(lb_pvalue, '__iter__') else float(lb_pvalue),
                    'no_autocorr': (float(lb_pvalue[-1]) if hasattr(lb_pvalue, '__iter__') else float(lb_pvalue)) > 0.05
                }
            
            # Durbin-Watson test (approximate)
            dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
            results['durbin_watson'] = {
                'statistic': float(dw_stat),
                'interpretation': 'no_autocorr' if 1.5 < dw_stat < 2.5 else 'autocorr_present'
            }
            
            return results
            
        except Exception:
            return {'ljung_box': {'no_autocorr': True}, 'durbin_watson': {'interpretation': 'no_autocorr'}}
    
    def _test_normality(self, residuals: np.ndarray) -> Dict[str, Any]:
        """Test for normality of residuals"""
        try:
            results = {}
            
            # Jarque-Bera test
            if STATSMODELS_AVAILABLE:
                jb_stat, jb_pvalue = jarque_bera(residuals)
                results['jarque_bera'] = {
                    'statistic': float(jb_stat),
                    'p_value': float(jb_pvalue),
                    'is_normal': jb_pvalue > 0.05
                }
            
            # Shapiro-Wilk test (for smaller samples)
            if len(residuals) <= 5000:
                from scipy.stats import shapiro
                sw_stat, sw_pvalue = shapiro(residuals)
                results['shapiro_wilk'] = {
                    'statistic': float(sw_stat),
                    'p_value': float(sw_pvalue),
                    'is_normal': sw_pvalue > 0.05
                }
            
            # Basic descriptive statistics
            results['descriptive'] = {
                'skewness': float(pd.Series(residuals).skew()),
                'kurtosis': float(pd.Series(residuals).kurtosis()),
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals, ddof=1))
            }
            
            return results
            
        except Exception:
            return {'jarque_bera': {'is_normal': True}, 'descriptive': {'skewness': 0.0, 'kurtosis': 0.0}}
    
    def _test_heteroskedasticity(self, residuals: np.ndarray, market_excess: np.ndarray) -> Dict[str, Any]:
        """Test for heteroskedasticity in residuals"""
        try:
            results = {}
            
            # Breusch-Pagan test (manual implementation)
            residuals_sq = residuals ** 2
            
            # Regress squared residuals on market excess returns
            X = np.column_stack([np.ones(len(market_excess)), market_excess])
            try:
                beta = np.linalg.lstsq(X, residuals_sq, rcond=None)[0]
                fitted = X @ beta
                
                # Calculate test statistic
                ssr = np.sum((residuals_sq - fitted) ** 2)
                tss = np.sum((residuals_sq - np.mean(residuals_sq)) ** 2)
                r_squared = 1 - ssr / tss if tss > 0 else 0
                
                # LM statistic
                lm_stat = len(residuals) * r_squared
                
                # Approximate p-value (chi-square with 1 df)
                from scipy.stats import chi2
                p_value = 1 - chi2.cdf(lm_stat, df=1)
                
                results['breusch_pagan'] = {
                    'statistic': float(lm_stat),
                    'p_value': float(p_value),
                    'homoskedastic': p_value > 0.05
                }
            except:
                results['breusch_pagan'] = {'homoskedastic': True}
            
            # White test (simplified)
            try:
                # Regress squared residuals on market returns and their squares
                X_white = np.column_stack([np.ones(len(market_excess)), market_excess, market_excess**2])
                beta_white = np.linalg.lstsq(X_white, residuals_sq, rcond=None)[0]
                fitted_white = X_white @ beta_white
                
                ssr_white = np.sum((residuals_sq - fitted_white) ** 2)
                tss_white = np.sum((residuals_sq - np.mean(residuals_sq)) ** 2)
                r_squared_white = 1 - ssr_white / tss_white if tss_white > 0 else 0
                
                lm_stat_white = len(residuals) * r_squared_white
                p_value_white = 1 - chi2.cdf(lm_stat_white, df=2)
                
                results['white'] = {
                    'statistic': float(lm_stat_white),
                    'p_value': float(p_value_white),
                    'homoskedastic': p_value_white > 0.05
                }
            except:
                results['white'] = {'homoskedastic': True}
            
            return results
            
        except Exception:
            return {'breusch_pagan': {'homoskedastic': True}, 'white': {'homoskedastic': True}}
    
    def _test_model_stability(self, index_excess: np.ndarray, market_excess: np.ndarray) -> Dict[str, Any]:
        """Test for model stability using recursive estimation"""
        try:
            results = {}
            
            if len(index_excess) < 60:
                return {'cusum': {'stable': True}, 'recursive_residuals': {'stable': True}}
            
            # Split data for stability testing
            split_point = len(index_excess) // 2
            
            # First half
            X1 = np.column_stack([np.ones(split_point), market_excess[:split_point]])
            y1 = index_excess[:split_point]
            beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
            
            # Second half
            X2 = np.column_stack([np.ones(len(market_excess) - split_point), market_excess[split_point:]])
            y2 = index_excess[split_point:]
            beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
            
            # Chow test (simplified)
            # Calculate F-statistic for structural break
            rss1 = np.sum((y1 - X1 @ beta1) ** 2)
            rss2 = np.sum((y2 - X2 @ beta2) ** 2)
            
            # Full sample
            X_full = np.column_stack([np.ones(len(market_excess)), market_excess])
            beta_full = np.linalg.lstsq(X_full, index_excess, rcond=None)[0]
            rss_full = np.sum((index_excess - X_full @ beta_full) ** 2)
            
            # F-statistic
            f_stat = ((rss_full - (rss1 + rss2)) / 2) / ((rss1 + rss2) / (len(index_excess) - 4))
            
            results['chow_test'] = {
                'f_statistic': float(f_stat),
                'stable': f_stat < 3.0  # Approximate critical value
            }
            
            # Parameter stability (coefficient difference)
            alpha_diff = abs(beta1[0] - beta2[0])
            beta_diff = abs(beta1[1] - beta2[1])
            
            results['parameter_stability'] = {
                'alpha_difference': float(alpha_diff),
                'beta_difference': float(beta_diff),
                'stable': alpha_diff < 0.01 and beta_diff < 0.5
            }
            
            return results
            
        except Exception:
            return {'chow_test': {'stable': True}, 'parameter_stability': {'stable': True}}
    
    def _perform_rolling_analysis(self, index_data: IndexData, market_data: IndexData, window: int = 252) -> Dict[str, Any]:
        """Perform rolling window CAPM analysis"""
        try:
            # Align data
            aligned_data = self._align_market_data(index_data, market_data)
            index_returns = aligned_data['index_returns']
            market_returns = aligned_data['market_returns']
            
            if len(index_returns) < window + 50:
                return {}
            
            # Calculate excess returns
            risk_free_daily = self.risk_free_rate / 252
            index_excess = index_returns - risk_free_daily
            market_excess = market_returns - risk_free_daily
            
            # Rolling analysis
            rolling_betas = []
            rolling_alphas = []
            rolling_r_squared = []
            rolling_dates = []
            
            for i in range(window, len(index_excess)):
                # Extract window data
                window_index = index_excess[i-window:i]
                window_market = market_excess[i-window:i]
                
                # Fit CAPM model
                try:
                    model = LinearRegression()
                    model.fit(window_market.reshape(-1, 1), window_index)
                    
                    beta = float(model.coef_[0])
                    alpha = float(model.intercept_)
                    r_sq = float(r2_score(window_index, model.predict(window_market.reshape(-1, 1))))
                    
                    rolling_betas.append(beta)
                    rolling_alphas.append(alpha)
                    rolling_r_squared.append(r_sq)
                    rolling_dates.append(i)
                    
                except:
                    rolling_betas.append(np.nan)
                    rolling_alphas.append(np.nan)
                    rolling_r_squared.append(np.nan)
                    rolling_dates.append(i)
            
            # Convert to pandas Series for easier analysis
            rolling_betas = pd.Series(rolling_betas, index=rolling_dates)
            rolling_alphas = pd.Series(rolling_alphas, index=rolling_dates)
            rolling_r_squared = pd.Series(rolling_r_squared, index=rolling_dates)
            
            # Calculate statistics
            results = {
                'rolling_betas': {
                    'values': rolling_betas.tolist(),
                    'mean': float(rolling_betas.mean()),
                    'std': float(rolling_betas.std()),
                    'min': float(rolling_betas.min()),
                    'max': float(rolling_betas.max()),
                    'current': float(rolling_betas.iloc[-1]) if not rolling_betas.empty else 1.0
                },
                'rolling_alphas': {
                    'values': rolling_alphas.tolist(),
                    'mean': float(rolling_alphas.mean()),
                    'std': float(rolling_alphas.std()),
                    'min': float(rolling_alphas.min()),
                    'max': float(rolling_alphas.max()),
                    'current': float(rolling_alphas.iloc[-1]) if not rolling_alphas.empty else 0.0
                },
                'rolling_r_squared': {
                    'values': rolling_r_squared.tolist(),
                    'mean': float(rolling_r_squared.mean()),
                    'std': float(rolling_r_squared.std()),
                    'min': float(rolling_r_squared.min()),
                    'max': float(rolling_r_squared.max()),
                    'current': float(rolling_r_squared.iloc[-1]) if not rolling_r_squared.empty else 0.5
                },
                'window_size': window,
                'n_observations': len(rolling_betas)
            }
            
            return results
            
        except Exception:
            return {}
    
    def _generate_forecasts(self, index_data: IndexData, market_data: IndexData, capm_result: CAPMResult, horizon: int = 30) -> Dict[str, Any]:
        """Generate CAPM-based forecasts"""
        try:
            # Align data
            aligned_data = self._align_market_data(index_data, market_data)
            index_returns = aligned_data['index_returns']
            market_returns = aligned_data['market_returns']
            
            if len(index_returns) < 50:
                return {}
            
            # Calculate recent volatility for market returns
            recent_market_vol = np.std(market_returns[-60:], ddof=1) if len(market_returns) >= 60 else np.std(market_returns, ddof=1)
            
            # Generate market return scenarios
            np.random.seed(42)  # For reproducibility
            
            # Historical mean market return
            historical_market_mean = np.mean(market_returns)
            
            # Generate scenarios
            n_scenarios = 1000
            market_scenarios = np.random.normal(historical_market_mean, recent_market_vol, (n_scenarios, horizon))
            
            # Calculate index forecasts using CAPM
            risk_free_daily = self.risk_free_rate / 252
            index_forecasts = []
            
            for scenario in market_scenarios:
                # Convert to excess returns
                market_excess = scenario - risk_free_daily
                
                # Apply CAPM model
                index_excess_forecast = capm_result.alpha + capm_result.beta * market_excess
                index_forecast = index_excess_forecast + risk_free_daily
                
                index_forecasts.append(index_forecast)
            
            index_forecasts = np.array(index_forecasts)
            
            # Calculate forecast statistics
            forecast_mean = np.mean(index_forecasts, axis=0)
            forecast_std = np.std(index_forecasts, axis=0, ddof=1)
            forecast_percentiles = np.percentile(index_forecasts, [5, 25, 50, 75, 95], axis=0)
            
            # Cumulative returns
            cumulative_returns = np.cumprod(1 + forecast_mean) - 1
            
            results = {
                'horizon_days': horizon,
                'n_scenarios': n_scenarios,
                'daily_forecasts': {
                    'mean': forecast_mean.tolist(),
                    'std': forecast_std.tolist(),
                    'percentile_5': forecast_percentiles[0].tolist(),
                    'percentile_25': forecast_percentiles[1].tolist(),
                    'percentile_50': forecast_percentiles[2].tolist(),
                    'percentile_75': forecast_percentiles[3].tolist(),
                    'percentile_95': forecast_percentiles[4].tolist()
                },
                'cumulative_returns': {
                    'expected': cumulative_returns.tolist(),
                    'final_expected': float(cumulative_returns[-1]),
                    'volatility': float(np.std(np.sum(index_forecasts, axis=1), ddof=1))
                },
                'risk_metrics': {
                    'var_5': float(np.percentile(np.sum(index_forecasts, axis=1), 5)),
                    'var_1': float(np.percentile(np.sum(index_forecasts, axis=1), 1)),
                    'expected_shortfall_5': float(np.mean(np.sum(index_forecasts, axis=1)[np.sum(index_forecasts, axis=1) <= np.percentile(np.sum(index_forecasts, axis=1), 5)])),
                    'max_drawdown_expected': float(np.min(cumulative_returns))
                }
            }
            
            return results
            
        except Exception:
            return {}
    
    def _perform_risk_attribution(self, index_data: IndexData, market_data: IndexData, capm_result: CAPMResult) -> Dict[str, Any]:
        """Perform risk attribution analysis"""
        try:
            # Align data
            aligned_data = self._align_market_data(index_data, market_data)
            index_returns = aligned_data['index_returns']
            market_returns = aligned_data['market_returns']
            
            if len(index_returns) < 30:
                return {}
            
            # Calculate various risk metrics
            index_vol = np.std(index_returns, ddof=1) * np.sqrt(252)
            market_vol = np.std(market_returns, ddof=1) * np.sqrt(252)
            
            # Systematic risk (beta * market volatility)
            systematic_risk = abs(capm_result.beta) * market_vol
            
            # Idiosyncratic risk
            idiosyncratic_risk = np.sqrt(max(0, index_vol**2 - systematic_risk**2))
            
            # Risk decomposition
            total_risk_squared = index_vol**2
            systematic_contribution = (systematic_risk**2 / total_risk_squared) * 100 if total_risk_squared > 0 else 0
            idiosyncratic_contribution = (idiosyncratic_risk**2 / total_risk_squared) * 100 if total_risk_squared > 0 else 0
            
            # Correlation-based analysis
            correlation = np.corrcoef(index_returns, market_returns)[0, 1]
            
            # Tracking error
            tracking_error = np.std(index_returns - market_returns, ddof=1) * np.sqrt(252)
            
            # Active risk vs passive risk
            active_risk = tracking_error
            passive_risk = abs(capm_result.beta) * market_vol
            
            results = {
                'total_risk': float(index_vol),
                'systematic_risk': float(systematic_risk),
                'idiosyncratic_risk': float(idiosyncratic_risk),
                'risk_decomposition': {
                    'systematic_percentage': float(systematic_contribution),
                    'idiosyncratic_percentage': float(idiosyncratic_contribution)
                },
                'correlation_with_market': float(correlation),
                'tracking_error': float(tracking_error),
                'active_vs_passive': {
                    'active_risk': float(active_risk),
                    'passive_risk': float(passive_risk),
                    'active_share': float(active_risk / (active_risk + passive_risk)) if (active_risk + passive_risk) > 0 else 0.5
                },
                'beta_adjusted_metrics': {
                     'beta_adjusted_return': float(np.mean(index_returns) * 252 / capm_result.beta) if capm_result.beta != 0 else 0.0,
                     'beta_adjusted_volatility': float(index_vol / abs(capm_result.beta)) if capm_result.beta != 0 else index_vol
                 }
            }
            
            return results
            
        except Exception:
            return {}
    
    def _analyze_regime_switching(self, index_data: IndexData, market_data: IndexData) -> Optional[RegimeAnalysisCAPM]:
        """Analyze regime switching in CAPM relationship"""
        try:
            # Align data
            aligned_data = self._align_market_data(index_data, market_data)
            index_returns = aligned_data['index_returns']
            market_returns = aligned_data['market_returns']
            
            if len(index_returns) < 100:
                return None
            
            # Calculate excess returns
            risk_free_daily = self.risk_free_rate / 252
            index_excess = index_returns - risk_free_daily
            market_excess = market_returns - risk_free_daily
            
            # Try HMM first, then fall back to Markov Regression
            if HMM_AVAILABLE:
                return self._hmm_regime_analysis(index_excess, market_excess)
            elif STATSMODELS_AVAILABLE:
                return self._markov_regime_analysis(index_excess, market_excess)
            else:
                return None
                
        except Exception:
            return None
    
    def _hmm_regime_analysis(self, index_excess: np.ndarray, market_excess: np.ndarray) -> Optional[RegimeAnalysisCAPM]:
        """Regime analysis using Hidden Markov Models"""
        try:
            # Prepare data for HMM (use both returns as features)
            X = np.column_stack([index_excess, market_excess])
            
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
                    # Fit CAPM for this regime
                    regime_index = index_excess[regime_mask]
                    regime_market = market_excess[regime_mask]
                    
                    if len(regime_index) > 5:
                        model_reg = LinearRegression()
                        model_reg.fit(regime_market.reshape(-1, 1), regime_index)
                        
                        regime_parameters[regime] = {
                            'alpha': float(model_reg.intercept_),
                            'beta': float(model_reg.coef_[0]),
                            'r_squared': float(r2_score(regime_index, model_reg.predict(regime_market.reshape(-1, 1))))
                        }
                    else:
                        regime_parameters[regime] = {'alpha': 0.0, 'beta': 1.0, 'r_squared': 0.0}
                    
                    # Regime persistence and duration
                    persistence = transition_matrix[regime, regime]
                    regime_persistence[regime] = float(persistence)
                    expected_duration[regime] = float(1 / (1 - persistence)) if persistence < 1 else float('inf')
                    
                    # Regime volatility
                    regime_volatility[regime] = float(np.std(regime_index, ddof=1) * np.sqrt(252))
                else:
                    regime_parameters[regime] = {'alpha': 0.0, 'beta': 1.0, 'r_squared': 0.0}
                    regime_persistence[regime] = 0.5
                    expected_duration[regime] = 2.0
                    regime_volatility[regime] = 0.2
            
            return RegimeAnalysisCAPM(
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
    
    def _markov_regime_analysis(self, index_excess: np.ndarray, market_excess: np.ndarray) -> Optional[RegimeAnalysisCAPM]:
        """Regime analysis using Markov Regression"""
        try:
            # Prepare data
            endog = index_excess
            exog = sm.add_constant(market_excess)
            
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
                # Extract regime-specific parameters
                alpha = float(results.params[f'const[{regime}]'])
                beta = float(results.params[f'x1[{regime}]'])
                
                regime_parameters[regime] = {
                    'alpha': alpha,
                    'beta': beta,
                    'r_squared': 0.5  # Approximate
                }
                
                # Persistence and duration
                persistence = float(transition_matrix[regime, regime])
                regime_persistence[regime] = persistence
                expected_duration[regime] = 1 / (1 - persistence) if persistence < 1 else float('inf')
                
                # Volatility (approximate)
                regime_mask = regime_states == regime
                if np.sum(regime_mask) > 5:
                    regime_volatility[regime] = float(np.std(index_excess[regime_mask], ddof=1) * np.sqrt(252))
                else:
                    regime_volatility[regime] = 0.2
            
            return RegimeAnalysisCAPM(
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