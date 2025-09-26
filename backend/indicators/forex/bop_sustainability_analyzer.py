import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import xgboost as xgb
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import het_arch
    from arch import arch_model
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
from scipy.signal import find_peaks, savgol_filter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import joblib
from concurrent.futures import ThreadPoolExecutor

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
class CapitalAccountData:
    """Capital and financial account data."""
    capital_account: pd.Series  # Capital transfers
    direct_investment: pd.Series  # FDI flows
    portfolio_investment: pd.Series  # Portfolio flows
    other_investment: pd.Series  # Bank flows, trade credits
    reserve_assets: pd.Series  # Central bank reserves
    financial_account_balance: pd.Series  # Total financial account

@dataclass
class RegimeAnalysis:
    """Regime switching analysis results."""
    current_regime: int
    regime_probabilities: np.ndarray
    regime_description: str
    transition_matrix: np.ndarray
    regime_persistence: Dict[str, float]
    expected_duration: Dict[str, float]

@dataclass
class VolatilityAnalysis:
    """Volatility clustering analysis."""
    current_volatility: float
    volatility_regime: str
    garch_forecast: np.ndarray
    arch_test_pvalue: float
    volatility_clustering: bool
    conditional_volatility: pd.Series

@dataclass
class MachineLearningPrediction:
    """ML-based BOP predictions."""
    ca_forecast: np.ndarray
    fa_forecast: np.ndarray
    sustainability_forecast: np.ndarray
    model_accuracy: Dict[str, float]
    feature_importance: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

@dataclass
class EconometricAnalysis:
    """Advanced econometric analysis."""
    cointegration_test: Dict[str, float]
    causality_tests: Dict[str, Dict[str, float]]
    structural_breaks: List[datetime]
    unit_root_tests: Dict[str, Dict[str, float]]
    error_correction_model: Dict[str, Any]

@dataclass
class RiskMetrics:
    """Comprehensive risk assessment."""
    var_95: float
    cvar_95: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    tail_ratio: float
    skewness: float
    kurtosis: float

@dataclass
class SustainabilityResult:
    """Enhanced BOP sustainability analysis result."""
    # Basic Analysis
    current_account_sustainability: Dict[str, Any]
    external_debt_sustainability: Dict[str, Any]
    reserve_adequacy: Dict[str, Any]
    vulnerability_indicators: Dict[str, Any]
    sustainability_score: float
    risk_level: str
    
    # Advanced Analysis
    regime_analysis: RegimeAnalysis
    volatility_analysis: VolatilityAnalysis
    ml_predictions: MachineLearningPrediction
    econometric_analysis: EconometricAnalysis
    risk_metrics: RiskMetrics
    
    # Model Performance
    model_diagnostics: Dict[str, Any]
    backtesting_results: Dict[str, Any]
    sensitivity_analysis: Dict[str, Any]

class BOPSustainabilityAnalyzer:
    """Enhanced BOP sustainability analysis with advanced econometric modeling."""
    
    def __init__(self, enable_ml: bool = True, enable_regime_switching: bool = True,
                 enable_volatility_modeling: bool = True):
        self.enable_ml = enable_ml
        self.enable_regime_switching = enable_regime_switching
        self.enable_volatility_modeling = enable_volatility_modeling
        
        self.thresholds = {
            'ca_deficit_threshold': -5.0,  # % of GDP
            'debt_gdp_threshold': 60.0,    # % of GDP
            'reserve_months_threshold': 3.0,  # months of imports
            'volatility_threshold': 0.15,   # 15% volatility threshold
            'regime_confidence': 0.7        # 70% confidence for regime classification
        }
        
        # Initialize models
        self.ml_models = self._initialize_ml_models()
        self.scalers = {
            'standard': StandardScaler(),
            'robust': RobustScaler()
        }
        
    def analyze_sustainability(self, ca_data: CurrentAccountData,
                              fa_data: CapitalAccountData,
                              gdp_data: pd.Series,
                              external_debt: pd.Series = None,
                              exchange_rate: pd.Series = None,
                              interest_rates: Dict[str, pd.Series] = None) -> SustainabilityResult:
        """Enhanced BOP sustainability analysis with advanced econometric modeling."""
        try:
            # Basic Analysis
            ca_sustainability = self._analyze_ca_sustainability(ca_data, gdp_data)
            debt_sustainability = self._analyze_debt_sustainability(external_debt, gdp_data, ca_data)
            reserve_adequacy = self._analyze_reserve_adequacy(fa_data, ca_data)
            vulnerability = self._calculate_vulnerability_indicators(ca_data, fa_data, gdp_data)
            
            # Overall sustainability score
            sustainability_score = self._calculate_sustainability_score(
                ca_sustainability, debt_sustainability, reserve_adequacy, vulnerability
            )
            risk_level = self._determine_risk_level(sustainability_score)
            
            # Advanced Analysis
            regime_analysis = self._perform_regime_analysis(ca_data, fa_data) if self.enable_regime_switching else self._empty_regime_analysis()
            volatility_analysis = self._perform_volatility_analysis(ca_data, exchange_rate) if self.enable_volatility_modeling else self._empty_volatility_analysis()
            ml_predictions = self._perform_ml_predictions(ca_data, fa_data, gdp_data) if self.enable_ml else self._empty_ml_predictions()
            econometric_analysis = self._perform_econometric_analysis(ca_data, fa_data, gdp_data, exchange_rate, interest_rates)
            risk_metrics = self._calculate_risk_metrics(ca_data, fa_data)
            
            # Model Performance
            model_diagnostics = self._run_model_diagnostics(ca_data, fa_data)
            backtesting_results = self._perform_backtesting(ca_data, fa_data, gdp_data)
            sensitivity_analysis = self._perform_sensitivity_analysis(ca_data, fa_data, gdp_data)
            
            return SustainabilityResult(
                # Basic Analysis
                current_account_sustainability=ca_sustainability,
                external_debt_sustainability=debt_sustainability,
                reserve_adequacy=reserve_adequacy,
                vulnerability_indicators=vulnerability,
                sustainability_score=sustainability_score,
                risk_level=risk_level,
                
                # Advanced Analysis
                regime_analysis=regime_analysis,
                volatility_analysis=volatility_analysis,
                ml_predictions=ml_predictions,
                econometric_analysis=econometric_analysis,
                risk_metrics=risk_metrics,
                
                # Model Performance
                model_diagnostics=model_diagnostics,
                backtesting_results=backtesting_results,
                sensitivity_analysis=sensitivity_analysis
            )
            
        except Exception as e:
            return self._create_empty_sustainability_result()
            
    def _analyze_ca_sustainability(self, ca_data: CurrentAccountData,
                                  gdp_data: pd.Series) -> Dict[str, Any]:
        """Analyze current account sustainability."""
        try:
            ca_balance = ca_data.current_account_balance
            ca_gdp_ratio = (ca_balance / gdp_data) * 100
            
            current_ratio = ca_gdp_ratio.iloc[-1] if len(ca_gdp_ratio) > 0 else 0
            avg_ratio = ca_gdp_ratio.mean()
            trend = self._calculate_trend(ca_gdp_ratio)
            
            # Sustainability assessment
            if current_ratio < self.thresholds['ca_deficit_threshold']:
                sustainability = 'unsustainable'
            elif current_ratio < -2.0:
                sustainability = 'concerning'
            else:
                sustainability = 'sustainable'
                
            return {
                'ca_gdp_ratio': {
                    'current': current_ratio,
                    'average': avg_ratio,
                    'trend': trend
                },
                'sustainability_assessment': sustainability,
                'financing_needs': abs(min(0, current_ratio)) * gdp_data.iloc[-1] / 100 if len(gdp_data) > 0 else 0
            }
        except:
            return {'ca_sustainability': 'analysis_failed'}
            
    def _analyze_debt_sustainability(self, external_debt: pd.Series,
                                    gdp_data: pd.Series,
                                    ca_data: CurrentAccountData) -> Dict[str, Any]:
        """Analyze external debt sustainability."""
        try:
            if external_debt is None or len(external_debt) == 0:
                return {'debt_sustainability': 'no_debt_data'}
                
            debt_gdp_ratio = (external_debt / gdp_data) * 100
            current_debt_ratio = debt_gdp_ratio.iloc[-1] if len(debt_gdp_ratio) > 0 else 0
            debt_trend = self._calculate_trend(debt_gdp_ratio)
            
            # Debt sustainability assessment
            if current_debt_ratio > self.thresholds['debt_gdp_threshold']:
                sustainability = 'high_risk'
            elif current_debt_ratio > 40.0:
                sustainability = 'moderate_risk'
            else:
                sustainability = 'low_risk'
                
            return {
                'debt_gdp_ratio': {
                    'current': current_debt_ratio,
                    'trend': debt_trend
                },
                'sustainability_assessment': sustainability,
                'debt_service_capacity': self._assess_debt_service_capacity(ca_data, external_debt)
            }
        except:
            return {'debt_sustainability': 'analysis_failed'}
            
    def _assess_debt_service_capacity(self, ca_data: CurrentAccountData, 
                                     external_debt: pd.Series) -> Dict[str, Any]:
        """Assess debt service capacity."""
        try:
            # Approximate debt service as percentage of exports
            exports_proxy = ca_data.trade_balance + ca_data.services_balance  # Rough proxy
            debt_service_ratio = (external_debt * 0.05) / exports_proxy.abs()  # Assume 5% service rate
            
            current_ratio = debt_service_ratio.iloc[-1] if len(debt_service_ratio) > 0 else 0
            
            if current_ratio > 0.25:  # 25% threshold
                capacity = 'weak'
            elif current_ratio > 0.15:
                capacity = 'moderate'
            else:
                capacity = 'strong'
                
            return {
                'debt_service_ratio': current_ratio,
                'service_capacity': capacity
            }
        except:
            return {'debt_service_capacity': 'assessment_failed'}
            
    def _analyze_reserve_adequacy(self, fa_data: CapitalAccountData,
                                 ca_data: CurrentAccountData) -> Dict[str, Any]:
        """Analyze reserve adequacy."""
        try:
            reserves = fa_data.reserve_assets.cumsum()  # Cumulative reserves
            imports_proxy = -ca_data.trade_balance  # Rough imports proxy
            
            # Calculate months of import coverage
            monthly_imports = imports_proxy.rolling(12).mean() / 12
            reserve_months = reserves / monthly_imports.abs()
            
            current_months = reserve_months.iloc[-1] if len(reserve_months) > 0 else 0
            
            # Adequacy assessment
            if current_months < self.thresholds['reserve_months_threshold']:
                adequacy = 'inadequate'
            elif current_months < 6.0:
                adequacy = 'adequate'
            else:
                adequacy = 'strong'
                
            return {
                'reserve_months': current_months,
                'adequacy_assessment': adequacy,
                'reserve_trend': self._calculate_trend(reserves),
                'reserve_volatility': fa_data.reserve_assets.std()
            }
        except:
            return {'reserve_adequacy': 'analysis_failed'}
            
    def _calculate_vulnerability_indicators(self, ca_data: CurrentAccountData,
                                          fa_data: CapitalAccountData,
                                          gdp_data: pd.Series) -> Dict[str, Any]:
        """Calculate vulnerability indicators."""
        try:
            indicators = {}
            
            # External financing dependence
            ca_deficit = -ca_data.current_account_balance
            financing_dependence = (ca_deficit / gdp_data) * 100
            indicators['financing_dependence'] = financing_dependence.mean()
            
            # Capital flow volatility
            portfolio_volatility = fa_data.portfolio_investment.pct_change().std()
            indicators['capital_flow_volatility'] = portfolio_volatility
            
            # Hot money ratio (portfolio flows / total flows)
            total_flows = fa_data.financial_account_balance.abs()
            hot_money_ratio = (fa_data.portfolio_investment.abs() / total_flows).mean()
            indicators['hot_money_ratio'] = hot_money_ratio
            
            # Current account persistence
            ca_autocorr = ca_data.current_account_balance.autocorr()
            indicators['ca_persistence'] = ca_autocorr
            
            # Vulnerability score (0-100, higher = more vulnerable)
            vulnerability_score = (
                min(20, max(0, indicators['financing_dependence'])) +  # 0-20
                min(30, indicators['capital_flow_volatility'] * 100) +  # 0-30
                min(25, indicators['hot_money_ratio'] * 100) +  # 0-25
                min(25, (1 - abs(indicators['ca_persistence'])) * 100)  # 0-25
            )
            
            indicators['vulnerability_score'] = vulnerability_score
            
            # Risk classification
            if vulnerability_score > 70:
                risk_class = 'high_vulnerability'
            elif vulnerability_score > 40:
                risk_class = 'moderate_vulnerability'
            else:
                risk_class = 'low_vulnerability'
                
            indicators['vulnerability_classification'] = risk_class
            
            return indicators
        except:
            return {'vulnerability_indicators': 'calculation_failed'}
            
    def _calculate_sustainability_score(self, ca_sustainability: Dict[str, Any],
                                       debt_sustainability: Dict[str, Any],
                                       reserve_adequacy: Dict[str, Any],
                                       vulnerability: Dict[str, Any]) -> float:
        """Calculate overall sustainability score (0-100)."""
        try:
            score = 100.0  # Start with perfect score
            
            # Current account component (0-30 points)
            if 'ca_gdp_ratio' in ca_sustainability:
                ca_ratio = ca_sustainability['ca_gdp_ratio']['current']
                if ca_ratio < -10:
                    score -= 30
                elif ca_ratio < -5:
                    score -= 20
                elif ca_ratio < -2:
                    score -= 10
                    
            # Debt component (0-25 points)
            if 'debt_gdp_ratio' in debt_sustainability:
                debt_ratio = debt_sustainability['debt_gdp_ratio']['current']
                if debt_ratio > 80:
                    score -= 25
                elif debt_ratio > 60:
                    score -= 15
                elif debt_ratio > 40:
                    score -= 8
                    
            # Reserve component (0-20 points)
            if 'reserve_months' in reserve_adequacy:
                reserve_months = reserve_adequacy['reserve_months']
                if reserve_months < 2:
                    score -= 20
                elif reserve_months < 3:
                    score -= 12
                elif reserve_months < 4:
                    score -= 5
                    
            # Vulnerability component (0-25 points)
            if 'vulnerability_score' in vulnerability:
                vuln_score = vulnerability['vulnerability_score']
                score -= (vuln_score / 100) * 25
                
            return max(0, min(100, score))
        except:
            return 50.0  # Default neutral score
            
    def _determine_risk_level(self, sustainability_score: float) -> str:
        """Determine risk level based on sustainability score."""
        if sustainability_score >= 80:
            return 'low_risk'
        elif sustainability_score >= 60:
            return 'moderate_risk'
        elif sustainability_score >= 40:
            return 'high_risk'
        else:
            return 'very_high_risk'
            
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
            
    def _initialize_ml_models(self) -> Dict[str, Any]:
        """Initialize machine learning models."""
        return {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42),
            'svr': SVR(kernel='rbf'),
            'elastic_net': ElasticNet(random_state=42)
        }
    
    def _perform_regime_analysis(self, ca_data: CurrentAccountData, fa_data: CapitalAccountData) -> RegimeAnalysis:
        """Perform regime switching analysis on BOP data."""
        try:
            if not HMM_AVAILABLE:
                return self._empty_regime_analysis()
                
            # Combine CA and FA data for regime analysis
            ca_balance = ca_data.current_account_balance.fillna(0)
            fa_balance = fa_data.financial_account_balance.fillna(0)
            
            if len(ca_balance) < 20:
                return self._empty_regime_analysis()
            
            # Prepare data for regime switching model
            data_matrix = np.column_stack([ca_balance.values, fa_balance.values])
            
            # Fit Hidden Markov Model
            n_regimes = 3  # Surplus, Balanced, Deficit regimes
            model = hmm.GaussianHMM(n_components=n_regimes, covariance_type="full", random_state=42)
            model.fit(data_matrix)
            
            # Get regime predictions
            regime_states = model.predict(data_matrix)
            regime_probs = model.predict_proba(data_matrix)
            
            # Current regime
            current_regime = regime_states[-1]
            current_probs = regime_probs[-1]
            
            # Regime descriptions
            regime_descriptions = {
                0: "BOP Surplus Regime",
                1: "BOP Balanced Regime", 
                2: "BOP Deficit Regime"
            }
            
            # Calculate transition matrix and persistence
            transition_matrix = model.transmat_
            regime_persistence = {f"Regime_{i}": transition_matrix[i, i] for i in range(n_regimes)}
            expected_duration = {f"Regime_{i}": 1 / (1 - transition_matrix[i, i]) if transition_matrix[i, i] < 1 else np.inf for i in range(n_regimes)}
            
            return RegimeAnalysis(
                current_regime=current_regime,
                regime_probabilities=current_probs,
                regime_description=regime_descriptions.get(current_regime, "Unknown Regime"),
                transition_matrix=transition_matrix,
                regime_persistence=regime_persistence,
                expected_duration=expected_duration
            )
            
        except Exception as e:
            return self._empty_regime_analysis()
    
    def _perform_volatility_analysis(self, ca_data: CurrentAccountData, exchange_rate: pd.Series = None) -> VolatilityAnalysis:
        """Perform volatility clustering analysis."""
        try:
            if not STATSMODELS_AVAILABLE:
                return self._empty_volatility_analysis()
                
            # Use CA balance for volatility analysis
            ca_balance = ca_data.current_account_balance.fillna(0)
            
            if len(ca_balance) < 20:
                return self._empty_volatility_analysis()
            
            # Calculate returns
            ca_returns = ca_balance.pct_change().dropna()
            
            if len(ca_returns) < 10:
                return self._empty_volatility_analysis()
            
            # Fit GARCH model
            garch_model = arch_model(ca_returns * 100, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            
            # Get conditional volatility
            conditional_vol = garch_fit.conditional_volatility / 100
            current_volatility = conditional_vol.iloc[-1]
            
            # ARCH test for volatility clustering
            arch_test = het_arch(ca_returns.dropna())
            arch_pvalue = arch_test[1]
            volatility_clustering = arch_pvalue < 0.05
            
            # Volatility forecast
            garch_forecast = garch_fit.forecast(horizon=5).variance.iloc[-1].values / 10000
            
            # Determine volatility regime
            if current_volatility > self.thresholds['volatility_threshold']:
                volatility_regime = "High Volatility"
            elif current_volatility > self.thresholds['volatility_threshold'] * 0.5:
                volatility_regime = "Medium Volatility"
            else:
                volatility_regime = "Low Volatility"
            
            return VolatilityAnalysis(
                current_volatility=current_volatility,
                volatility_regime=volatility_regime,
                garch_forecast=garch_forecast,
                arch_test_pvalue=arch_pvalue,
                volatility_clustering=volatility_clustering,
                conditional_volatility=conditional_vol
            )
            
        except Exception as e:
            return self._empty_volatility_analysis()
    
    def _perform_ml_predictions(self, ca_data: CurrentAccountData, fa_data: CapitalAccountData, gdp_data: pd.Series) -> MachineLearningPrediction:
        """Perform machine learning predictions."""
        try:
            # Prepare features
            features = self._prepare_ml_features(ca_data, fa_data, gdp_data)
            
            if len(features) < 20:
                return self._empty_ml_predictions()
            
            # Prepare targets
            ca_target = ca_data.current_account_balance.shift(-1).dropna()
            fa_target = fa_data.financial_account_balance.shift(-1).dropna()
            
            # Align data
            min_len = min(len(features), len(ca_target), len(fa_target))
            features = features.iloc[:min_len]
            ca_target = ca_target.iloc[:min_len]
            fa_target = fa_target.iloc[:min_len]
            
            # Scale features
            features_scaled = self.scalers['robust'].fit_transform(features)
            
            # Train models and make predictions
            model_accuracy = {}
            ca_predictions = []
            fa_predictions = []
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            for name, model in self.ml_models.items():
                try:
                    # CA prediction
                    ca_scores = cross_val_score(model, features_scaled, ca_target, cv=tscv, scoring='r2')
                    model.fit(features_scaled, ca_target)
                    ca_pred = model.predict(features_scaled[-5:])
                    ca_predictions.append(ca_pred)
                    
                    # FA prediction
                    fa_scores = cross_val_score(model, features_scaled, fa_target, cv=tscv, scoring='r2')
                    model.fit(features_scaled, fa_target)
                    fa_pred = model.predict(features_scaled[-5:])
                    fa_predictions.append(fa_pred)
                    
                    model_accuracy[name] = {
                        'ca_r2': ca_scores.mean(),
                        'fa_r2': fa_scores.mean()
                    }
                    
                except:
                    continue
            
            # Ensemble predictions
            ca_forecast = np.mean(ca_predictions, axis=0) if ca_predictions else np.zeros(5)
            fa_forecast = np.mean(fa_predictions, axis=0) if fa_predictions else np.zeros(5)
            
            # Sustainability forecast (simple heuristic)
            sustainability_forecast = (ca_forecast + fa_forecast) / 2
            
            # Feature importance (using Random Forest)
            feature_importance = {}
            try:
                rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_model.fit(features_scaled, ca_target)
                feature_importance = dict(zip(features.columns, rf_model.feature_importances_))
            except:
                pass
            
            # Confidence intervals (simple approach)
            confidence_intervals = {
                'ca_forecast': (ca_forecast * 0.8, ca_forecast * 1.2),
                'fa_forecast': (fa_forecast * 0.8, fa_forecast * 1.2)
            }
            
            return MachineLearningPrediction(
                ca_forecast=ca_forecast,
                fa_forecast=fa_forecast,
                sustainability_forecast=sustainability_forecast,
                model_accuracy=model_accuracy,
                feature_importance=feature_importance,
                confidence_intervals=confidence_intervals
            )
            
        except Exception as e:
            return self._empty_ml_predictions()
    
    def _prepare_ml_features(self, ca_data: CurrentAccountData, fa_data: CapitalAccountData, gdp_data: pd.Series) -> pd.DataFrame:
        """Prepare features for machine learning."""
        features = pd.DataFrame()
        
        try:
            # Current account features
            features['ca_balance'] = ca_data.current_account_balance
            features['trade_balance'] = ca_data.trade_balance
            features['services_balance'] = ca_data.services_balance
            
            # Financial account features
            features['fa_balance'] = fa_data.financial_account_balance
            features['fdi_flows'] = fa_data.direct_investment
            features['portfolio_flows'] = fa_data.portfolio_investment
            
            # GDP features
            features['gdp'] = gdp_data
            features['ca_gdp_ratio'] = ca_data.current_account_balance / gdp_data
            
            # Technical indicators
            for col in ['ca_balance', 'fa_balance', 'gdp']:
                if col in features.columns:
                    features[f'{col}_ma5'] = features[col].rolling(5).mean()
                    features[f'{col}_ma20'] = features[col].rolling(20).mean()
                    features[f'{col}_std5'] = features[col].rolling(5).std()
            
            return features.dropna()
            
        except Exception as e:
            return pd.DataFrame()
    
    def _perform_econometric_analysis(self, ca_data: CurrentAccountData, fa_data: CapitalAccountData, 
                                    gdp_data: pd.Series, exchange_rate: pd.Series = None, 
                                    interest_rates: Dict[str, pd.Series] = None) -> EconometricAnalysis:
        """Perform advanced econometric analysis."""
        try:
            if not STATSMODELS_AVAILABLE:
                return EconometricAnalysis({}, {}, [], {}, {})
                
            # Unit root tests
            unit_root_tests = {}
            for name, series in [('ca_balance', ca_data.current_account_balance), 
                               ('fa_balance', fa_data.financial_account_balance)]:
                try:
                    adf_stat, adf_pvalue = adfuller(series.dropna())[:2]
                    kpss_stat, kpss_pvalue = kpss(series.dropna())[:2]
                    unit_root_tests[name] = {
                        'adf_statistic': adf_stat,
                        'adf_pvalue': adf_pvalue,
                        'kpss_statistic': kpss_stat,
                        'kpss_pvalue': kpss_pvalue
                    }
                except:
                    unit_root_tests[name] = {'error': 'test_failed'}
            
            # Cointegration test (simplified)
            cointegration_test = {}
            try:
                ca_balance = ca_data.current_account_balance.dropna()
                fa_balance = fa_data.financial_account_balance.dropna()
                
                if len(ca_balance) > 10 and len(fa_balance) > 10:
                    # Simple correlation-based cointegration proxy
                    correlation = ca_balance.corr(fa_balance)
                    cointegration_test = {
                        'correlation': correlation,
                        'cointegrated': abs(correlation) > 0.7
                    }
            except:
                cointegration_test = {'error': 'test_failed'}
            
            # Structural breaks (simplified)
            structural_breaks = []
            try:
                ca_balance = ca_data.current_account_balance.dropna()
                if len(ca_balance) > 20:
                    # Simple approach: find significant changes in rolling mean
                    rolling_mean = ca_balance.rolling(10).mean()
                    changes = rolling_mean.diff().abs()
                    threshold = changes.quantile(0.95)
                    break_points = changes[changes > threshold].index.tolist()
                    structural_breaks = break_points[:5]  # Limit to 5 breaks
            except:
                pass
            
            return EconometricAnalysis(
                cointegration_test=cointegration_test,
                causality_tests={},  # Placeholder
                structural_breaks=structural_breaks,
                unit_root_tests=unit_root_tests,
                error_correction_model={}  # Placeholder
            )
            
        except Exception as e:
            return EconometricAnalysis(
                cointegration_test={},
                causality_tests={},
                structural_breaks=[],
                unit_root_tests={},
                error_correction_model={}
            )
    
    def _calculate_risk_metrics(self, ca_data: CurrentAccountData, fa_data: CapitalAccountData) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        try:
            # Combine CA and FA data
            ca_balance = ca_data.current_account_balance.fillna(0)
            fa_balance = fa_data.financial_account_balance.fillna(0)
            combined_balance = ca_balance + fa_balance
            
            if len(combined_balance) < 10:
                return self._empty_risk_metrics()
            
            # Calculate returns
            returns = combined_balance.pct_change().dropna()
            
            if len(returns) < 5:
                return self._empty_risk_metrics()
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Drawdown analysis
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Risk-adjusted returns
            mean_return = returns.mean()
            volatility = returns.std()
            
            sharpe_ratio = mean_return / volatility if volatility > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_deviation = downside_returns.std() if len(downside_returns) > 0 else volatility
            sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
            
            # Calmar ratio
            calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Tail ratio
            tail_ratio = abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5)) if np.percentile(returns, 5) != 0 else 1
            
            # Higher moments
            skewness = stats.skew(returns)
            kurtosis = stats.kurtosis(returns)
            
            return RiskMetrics(
                var_95=var_95,
                cvar_95=cvar_95,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                tail_ratio=tail_ratio,
                skewness=skewness,
                kurtosis=kurtosis
            )
            
        except Exception as e:
            return self._empty_risk_metrics()
    
    def _run_model_diagnostics(self, ca_data: CurrentAccountData, fa_data: CapitalAccountData) -> Dict[str, Any]:
        """Run comprehensive model diagnostics."""
        try:
            diagnostics = {}
            
            # Data quality checks
            ca_balance = ca_data.current_account_balance
            fa_balance = fa_data.financial_account_balance
            
            diagnostics['data_quality'] = {
                'ca_missing_pct': ca_balance.isna().mean() * 100,
                'fa_missing_pct': fa_balance.isna().mean() * 100,
                'ca_outliers': len(ca_balance[np.abs(stats.zscore(ca_balance.dropna())) > 3]),
                'fa_outliers': len(fa_balance[np.abs(stats.zscore(fa_balance.dropna())) > 3])
            }
            
            # Stationarity tests
            diagnostics['stationarity'] = {}
            if STATSMODELS_AVAILABLE:
                for name, series in [('ca_balance', ca_balance), ('fa_balance', fa_balance)]:
                    try:
                        adf_stat, adf_pvalue = adfuller(series.dropna())[:2]
                        diagnostics['stationarity'][name] = {
                            'adf_pvalue': adf_pvalue,
                            'is_stationary': adf_pvalue < 0.05
                        }
                    except:
                        diagnostics['stationarity'][name] = {'error': 'test_failed'}
            else:
                diagnostics['stationarity'] = {'error': 'statsmodels_not_available'}
            
            return diagnostics
            
        except Exception as e:
            return {'error': 'diagnostics_failed'}
    
    def _perform_backtesting(self, ca_data: CurrentAccountData, fa_data: CapitalAccountData, gdp_data: pd.Series) -> Dict[str, Any]:
        """Perform backtesting of sustainability predictions."""
        try:
            # Simple backtesting approach
            ca_balance = ca_data.current_account_balance.dropna()
            
            if len(ca_balance) < 50:
                return {'error': 'insufficient_data'}
            
            # Split data for backtesting
            split_point = int(len(ca_balance) * 0.8)
            train_data = ca_balance[:split_point]
            test_data = ca_balance[split_point:]
            
            # Simple prediction model (moving average)
            predictions = []
            actuals = []
            
            for i in range(len(test_data)):
                if i == 0:
                    pred = train_data.tail(5).mean()
                else:
                    pred = test_data[:i].tail(5).mean()
                predictions.append(pred)
                actuals.append(test_data.iloc[i])
            
            # Calculate metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            
            return {
                'mse': mse,
                'mae': mae,
                'predictions_count': len(predictions),
                'accuracy_score': 1 - (mae / np.mean(np.abs(actuals))) if np.mean(np.abs(actuals)) > 0 else 0
            }
            
        except Exception as e:
            return {'error': 'backtesting_failed'}
    
    def _perform_sensitivity_analysis(self, ca_data: CurrentAccountData, fa_data: CapitalAccountData, gdp_data: pd.Series) -> Dict[str, Any]:
        """Perform sensitivity analysis of sustainability metrics."""
        try:
            # Base case sustainability score
            base_sustainability = self._calculate_sustainability_score(
                self._analyze_ca_sustainability(ca_data, gdp_data),
                self._analyze_debt_sustainability(None, gdp_data, ca_data),
                self._analyze_reserve_adequacy(fa_data, ca_data),
                self._calculate_vulnerability_indicators(ca_data, fa_data, gdp_data)
            )
            
            sensitivity_results = {}
            
            # Test sensitivity to CA balance changes
            for shock in [-0.2, -0.1, 0.1, 0.2]:  # ±20%, ±10% shocks
                shocked_ca = CurrentAccountData(
                    trade_balance=ca_data.trade_balance * (1 + shock),
                    services_balance=ca_data.services_balance,
                    primary_income=ca_data.primary_income,
                    secondary_income=ca_data.secondary_income,
                    current_account_balance=ca_data.current_account_balance * (1 + shock)
                )
                
                shocked_sustainability = self._calculate_sustainability_score(
                    self._analyze_ca_sustainability(shocked_ca, gdp_data),
                    self._analyze_debt_sustainability(None, gdp_data, shocked_ca),
                    self._analyze_reserve_adequacy(fa_data, shocked_ca),
                    self._calculate_vulnerability_indicators(shocked_ca, fa_data, gdp_data)
                )
                
                sensitivity_results[f'ca_shock_{shock}'] = {
                    'shock_magnitude': shock,
                    'sustainability_change': shocked_sustainability - base_sustainability,
                    'sensitivity': (shocked_sustainability - base_sustainability) / shock if shock != 0 else 0
                }
            
            return sensitivity_results
            
        except Exception as e:
            return {'error': 'sensitivity_analysis_failed'}
    
    # Empty result methods
    def _empty_regime_analysis(self) -> RegimeAnalysis:
        return RegimeAnalysis(
            current_regime=0,
            regime_probabilities=np.array([1.0, 0.0, 0.0]),
            regime_description="Analysis not available",
            transition_matrix=np.eye(3),
            regime_persistence={},
            expected_duration={}
        )
    
    def _empty_volatility_analysis(self) -> VolatilityAnalysis:
        return VolatilityAnalysis(
            current_volatility=0.0,
            volatility_regime="Unknown",
            garch_forecast=np.zeros(5),
            arch_test_pvalue=1.0,
            volatility_clustering=False,
            conditional_volatility=pd.Series()
        )
    
    def _empty_ml_predictions(self) -> MachineLearningPrediction:
        return MachineLearningPrediction(
            ca_forecast=np.zeros(5),
            fa_forecast=np.zeros(5),
            sustainability_forecast=np.zeros(5),
            model_accuracy={},
            feature_importance={},
            confidence_intervals={}
        )
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        return RiskMetrics(
            var_95=0.0,
            cvar_95=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            tail_ratio=1.0,
            skewness=0.0,
            kurtosis=0.0
        )
            
    def _create_empty_sustainability_result(self) -> SustainabilityResult:
        """Create empty sustainability result for error cases."""
        return SustainabilityResult(
            # Basic Analysis
            current_account_sustainability={'analysis': 'failed'},
            external_debt_sustainability={'analysis': 'failed'},
            reserve_adequacy={'analysis': 'failed'},
            vulnerability_indicators={'analysis': 'failed'},
            sustainability_score=50.0,
            risk_level='unknown',
            
            # Advanced Analysis
            regime_analysis=self._empty_regime_analysis(),
            volatility_analysis=self._empty_volatility_analysis(),
            ml_predictions=self._empty_ml_predictions(),
            econometric_analysis=EconometricAnalysis({}, {}, [], {}, {}),
            risk_metrics=self._empty_risk_metrics(),
            
            # Model Performance
            model_diagnostics={'error': 'analysis_failed'},
            backtesting_results={'error': 'analysis_failed'},
            sensitivity_analysis={'error': 'analysis_failed'}
        )