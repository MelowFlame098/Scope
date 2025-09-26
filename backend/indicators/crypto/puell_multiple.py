"""Enhanced Puell Multiple Model for Cryptocurrency Mining Profitability Analysis

This module implements an advanced Puell Multiple calculation and analysis for cryptocurrency markets.
The enhanced model includes:
- Traditional Puell Multiple calculation and analysis
- Advanced cycle analysis with Fourier transforms and wavelet decomposition
- Predictive modeling using machine learning techniques
- Mining economics modeling with difficulty adjustments
- Market regime detection and transition analysis
- Volatility clustering and GARCH modeling
- Statistical significance testing and confidence intervals
- Cross-correlation analysis with network metrics
- Anomaly detection for extreme mining events
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats, signal, optimize
from scipy.fft import fft, fftfreq
import pywt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from arch import arch_model
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.tsa.stattools import adfuller, kpss
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class CycleAnalysis:
    """Results from cycle analysis"""
    dominant_cycle_period: float
    cycle_strength: float
    cycle_phase: str
    cycle_position: float  # 0-1, where 0 is trough, 0.5 is peak
    next_cycle_peak_estimate: Optional[str]
    next_cycle_trough_estimate: Optional[str]
    fourier_components: Dict[str, float]
    wavelet_analysis: Dict[str, Any]
    seasonal_components: Dict[str, float]

@dataclass
class MiningEconomics:
    """Mining economics analysis"""
    mining_efficiency_score: float
    difficulty_adjustment_impact: float
    miner_capitulation_risk: float
    hash_rate_correlation: float
    energy_cost_estimate: float
    mining_margin_estimate: float
    network_security_score: float
    miner_revenue_sustainability: str

@dataclass
class RegimeAnalysis:
    """Market regime analysis"""
    current_regime: int
    regime_probabilities: List[float]
    regime_characteristics: Dict[int, Dict[str, float]]
    transition_probabilities: np.ndarray
    expected_regime_duration: List[float]
    regime_stability: float
    regime_description: Dict[int, str]

@dataclass
class VolatilityAnalysis:
    """Volatility analysis results"""
    current_volatility: float
    volatility_forecast: List[float]
    garch_parameters: Dict[str, float]
    volatility_regime: str
    clustering_score: float
    heteroskedasticity_test: Dict[str, float]
    volatility_persistence: float

@dataclass
class PredictiveMetrics:
    """Predictive modeling results"""
    ml_forecast: List[float]
    forecast_confidence: List[float]
    feature_importance: Dict[str, float]
    model_accuracy: float
    trend_prediction: str
    reversal_probability: float
    time_to_next_extreme: Optional[int]
    forecast_horizon: int

@dataclass
class AnomalyDetection:
    """Anomaly detection results"""
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: Optional[str]
    historical_anomalies: List[Dict[str, Any]]
    anomaly_severity: str
    context_explanation: str

@dataclass
class PuellResult:
    """Enhanced results from Puell Multiple analysis"""
    # Basic Puell metrics
    current_puell: float
    puell_percentile: float
    mining_profitability: str
    market_cycle_phase: str
    historical_puell: List[float]
    puell_bands: Dict[str, float]
    timestamps: List[str]
    
    # Advanced analysis components
    cycle_analysis: Optional[CycleAnalysis] = None
    mining_economics: Optional[MiningEconomics] = None
    regime_analysis: Optional[RegimeAnalysis] = None
    volatility_analysis: Optional[VolatilityAnalysis] = None
    predictive_metrics: Optional[PredictiveMetrics] = None
    anomaly_detection: Optional[AnomalyDetection] = None
    
    # Statistical measures
    puell_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    jarque_bera_stat: Optional[float] = None
    jarque_bera_pvalue: Optional[float] = None
    autocorrelation: Optional[List[float]] = None
    
    # Cross-correlations
    price_correlation: Optional[float] = None
    hash_rate_correlation: Optional[float] = None
    difficulty_correlation: Optional[float] = None
    volume_correlation: Optional[float] = None
    
    # Confidence intervals
    puell_confidence_interval: Optional[Tuple[float, float]] = None
    forecast_confidence_bands: Optional[Dict[str, List[float]]] = None
    
    # Enhanced analysis components
    mining_profitability_cycles: Optional[MiningProfitabilityCycles] = None
    halving_effects_analysis: Optional[HalvingEffectsAnalysis] = None
    supply_shock_analysis: Optional[SupplyShockAnalysis] = None

@dataclass
class MiningProfitabilityCycles:
    """Enhanced mining profitability cycle analysis"""
    cycle_identification: Dict[str, Any]
    profitability_phases: Dict[str, float]
    cycle_duration_analysis: Dict[str, int]
    revenue_cycle_correlation: float
    mining_margin_cycles: Dict[str, float]
    operational_efficiency_trends: Dict[str, float]
    cost_structure_evolution: Dict[str, float]
    technology_upgrade_cycles: Dict[str, Any]
    geographic_mining_shifts: Dict[str, float]
    regulatory_cycle_impact: Dict[str, float]
    energy_market_cycles: Dict[str, float]
    mining_pool_dynamics: Dict[str, float]

@dataclass
class HalvingEffectsAnalysis:
    """Comprehensive Bitcoin halving effects analysis"""
    halving_countdown: Dict[str, int]
    pre_halving_dynamics: Dict[str, float]
    post_halving_adjustments: Dict[str, float]
    historical_halving_comparison: Dict[str, Any]
    supply_issuance_modeling: Dict[str, float]
    miner_revenue_impact: Dict[str, float]
    difficulty_adjustment_patterns: Dict[str, float]
    hash_rate_migration_analysis: Dict[str, float]
    market_anticipation_metrics: Dict[str, float]
    price_elasticity_changes: Dict[str, float]
    long_term_supply_dynamics: Dict[str, float]
    halving_premium_analysis: Dict[str, float]

@dataclass
class SupplyShockAnalysis:
    """Advanced supply shock detection and analysis"""
    shock_detection_metrics: Dict[str, float]
    supply_velocity_analysis: Dict[str, float]
    miner_selling_pressure: Dict[str, float]
    exchange_flow_anomalies: Dict[str, float]
    institutional_accumulation: Dict[str, float]
    retail_behavior_shifts: Dict[str, float]
    liquidity_impact_assessment: Dict[str, float]
    price_elasticity_modeling: Dict[str, float]
    market_depth_analysis: Dict[str, float]
    cross_exchange_arbitrage: Dict[str, float]
    derivative_market_impact: Dict[str, float]
    recovery_timeline_estimation: Dict[str, int]

class PuellMultipleModel:
    """Enhanced Puell Multiple Model for comprehensive mining profitability analysis"""
    
    def __init__(self, asset: str = "BTC", 
                 enable_cycle_analysis: bool = False,
                 enable_mining_economics: bool = False,
                 enable_regime_analysis: bool = False,
                 enable_volatility_analysis: bool = False,
                 enable_predictive_modeling: bool = False,
                 enable_anomaly_detection: bool = False,
                 n_regimes: int = 2,
                 forecast_horizon: int = 30,
                 confidence_level: float = 0.95):
        self.asset = asset.upper()
        
        # Feature flags
        self.enable_cycle_analysis = enable_cycle_analysis
        self.enable_mining_economics = enable_mining_economics
        self.enable_regime_analysis = enable_regime_analysis
        self.enable_volatility_analysis = enable_volatility_analysis
        self.enable_predictive_modeling = enable_predictive_modeling
        self.enable_anomaly_detection = enable_anomaly_detection
        
        # Model parameters
        self.n_regimes = n_regimes
        self.forecast_horizon = forecast_horizon
        self.confidence_level = confidence_level
        
        # Initialize models
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.regime_model = None
        self.garch_model = None
        
    def calculate_puell_multiple(self, daily_issuance_usd: float, 
                               issuance_ma_365: float) -> float:
        """Calculate Puell Multiple
        
        Puell Multiple = Daily Coin Issuance (USD) / 365-day MA of Daily Coin Issuance (USD)
        """
        if issuance_ma_365 <= 0:
            return 1.0
        
        return daily_issuance_usd / issuance_ma_365
    
    def determine_mining_profitability(self, puell_multiple: float) -> str:
        """Determine mining profitability status"""
        if puell_multiple > 4:
            return "Extremely High Profitability - Potential Top"
        elif puell_multiple > 2:
            return "High Profitability - Bull Market"
        elif puell_multiple > 0.5:
            return "Normal Profitability - Stable Market"
        elif puell_multiple > 0.3:
            return "Low Profitability - Bear Market"
        else:
            return "Extremely Low Profitability - Potential Bottom"
    
    def determine_market_cycle_phase(self, puell_percentile: float) -> str:
        """Determine market cycle phase based on Puell percentile"""
        if puell_percentile > 95:
            return "Cycle Top - Extreme Overheating"
        elif puell_percentile > 80:
            return "Late Bull Market - Overheating"
        elif puell_percentile > 60:
            return "Bull Market - Healthy Growth"
        elif puell_percentile > 40:
            return "Neutral - Consolidation"
        elif puell_percentile > 20:
            return "Bear Market - Cooling Down"
        else:
            return "Cycle Bottom - Extreme Undervaluation"
    
    def calculate_puell_bands(self, historical_puell: List[float]) -> Dict[str, float]:
        """Calculate Puell Multiple percentile bands"""
        if len(historical_puell) < 10:
            return {'bottom': 0, 'low': 0, 'fair': 0, 'high': 0, 'top': 0}
        
        return {
            'bottom': np.percentile(historical_puell, 10),   # Bottom 10%
            'low': np.percentile(historical_puell, 30),      # Bottom 30%
            'fair': np.percentile(historical_puell, 50),     # Median
            'high': np.percentile(historical_puell, 70),     # Top 30%
            'top': np.percentile(historical_puell, 90)       # Top 10%
        }
    
    def _perform_cycle_analysis(self, puell_values: np.ndarray, timestamps: List[str]) -> CycleAnalysis:
        """Perform advanced cycle analysis using Fourier transforms and wavelets"""
        try:
            # Fourier Transform Analysis
            fft_values = fft(puell_values)
            freqs = fftfreq(len(puell_values))
            
            # Find dominant frequency (excluding DC component)
            power_spectrum = np.abs(fft_values[1:len(fft_values)//2])
            dominant_freq_idx = np.argmax(power_spectrum) + 1
            dominant_frequency = abs(freqs[dominant_freq_idx])
            dominant_cycle_period = 1 / dominant_frequency if dominant_frequency > 0 else 365
            
            # Calculate cycle strength
            total_power = np.sum(power_spectrum)
            dominant_power = power_spectrum[dominant_freq_idx - 1]
            cycle_strength = dominant_power / total_power if total_power > 0 else 0
            
            # Wavelet Analysis
            try:
                coeffs = pywt.wavedec(puell_values, 'db4', level=4)
                wavelet_analysis = {
                    'approximation_energy': np.sum(coeffs[0]**2),
                    'detail_energies': [np.sum(c**2) for c in coeffs[1:]],
                    'total_energy': np.sum([np.sum(c**2) for c in coeffs])
                }
            except Exception:
                wavelet_analysis = {'error': 'Wavelet analysis failed'}
            
            # Determine cycle phase and position
            recent_values = puell_values[-30:] if len(puell_values) >= 30 else puell_values
            current_value = puell_values[-1]
            cycle_min = np.min(puell_values)
            cycle_max = np.max(puell_values)
            
            # Normalize position (0 = trough, 1 = peak)
            cycle_position = (current_value - cycle_min) / (cycle_max - cycle_min) if cycle_max > cycle_min else 0.5
            
            # Determine cycle phase
            if cycle_position < 0.2:
                cycle_phase = "Trough - Accumulation"
            elif cycle_position < 0.4:
                cycle_phase = "Early Bull - Recovery"
            elif cycle_position < 0.6:
                cycle_phase = "Mid Bull - Growth"
            elif cycle_position < 0.8:
                cycle_phase = "Late Bull - Euphoria"
            else:
                cycle_phase = "Peak - Distribution"
            
            # Estimate next cycle extremes (simplified)
            days_per_cycle = int(dominant_cycle_period)
            current_date = pd.to_datetime(timestamps[-1]) if timestamps else pd.Timestamp.now()
            
            if cycle_position > 0.5:  # Approaching peak, next extreme is trough
                days_to_trough = int(days_per_cycle * (1 - cycle_position))
                next_trough = current_date + pd.Timedelta(days=days_to_trough)
                next_peak = current_date + pd.Timedelta(days=days_to_trough + days_per_cycle//2)
            else:  # Approaching trough, next extreme is peak
                days_to_peak = int(days_per_cycle * (0.5 - cycle_position))
                next_peak = current_date + pd.Timedelta(days=days_to_peak)
                next_trough = current_date + pd.Timedelta(days=days_to_peak + days_per_cycle//2)
            
            # Fourier components
            fourier_components = {
                'dominant_frequency': float(dominant_frequency),
                'dominant_period': float(dominant_cycle_period),
                'cycle_strength': float(cycle_strength),
                'harmonic_content': float(np.std(power_spectrum))
            }
            
            # Seasonal components (simplified)
            seasonal_components = {
                'annual_component': float(np.mean(puell_values)),
                'quarterly_variation': float(np.std(puell_values)),
                'monthly_trend': float(np.mean(np.diff(puell_values)))
            }
            
            return CycleAnalysis(
                dominant_cycle_period=float(dominant_cycle_period),
                cycle_strength=float(cycle_strength),
                cycle_phase=cycle_phase,
                cycle_position=float(cycle_position),
                next_cycle_peak_estimate=next_peak.strftime('%Y-%m-%d'),
                next_cycle_trough_estimate=next_trough.strftime('%Y-%m-%d'),
                fourier_components=fourier_components,
                wavelet_analysis=wavelet_analysis,
                seasonal_components=seasonal_components
            )
            
        except Exception as e:
            logger.error(f"Error in cycle analysis: {str(e)}")
            return CycleAnalysis(
                dominant_cycle_period=365.0,
                cycle_strength=0.0,
                cycle_phase="Unknown",
                cycle_position=0.5,
                next_cycle_peak_estimate=None,
                next_cycle_trough_estimate=None,
                fourier_components={},
                wavelet_analysis={},
                seasonal_components={}
            )
    
    def _perform_mining_economics_analysis(self, historical_data: pd.DataFrame, puell_values: np.ndarray) -> MiningEconomics:
        """Perform mining economics analysis"""
        try:
            current_puell = puell_values[-1]
            
            # Mining efficiency score (based on Puell stability)
            puell_volatility = np.std(puell_values[-30:]) if len(puell_values) >= 30 else np.std(puell_values)
            mining_efficiency_score = max(0, 1 - puell_volatility)
            
            # Difficulty adjustment impact (simplified)
            if 'difficulty' in historical_data.columns:
                difficulty_changes = np.diff(historical_data['difficulty'].values)
                difficulty_adjustment_impact = np.mean(difficulty_changes[-10:]) if len(difficulty_changes) >= 10 else 0
            else:
                difficulty_adjustment_impact = 0.0
            
            # Miner capitulation risk
            low_puell_periods = np.sum(puell_values < 0.5)
            total_periods = len(puell_values)
            miner_capitulation_risk = low_puell_periods / total_periods if total_periods > 0 else 0
            
            # Hash rate correlation
            if 'hash_rate' in historical_data.columns:
                hash_rate_values = historical_data['hash_rate'].values
                if len(hash_rate_values) == len(puell_values):
                    hash_rate_correlation = np.corrcoef(puell_values, hash_rate_values)[0, 1]
                else:
                    hash_rate_correlation = 0.0
            else:
                hash_rate_correlation = 0.0
            
            # Energy cost estimate (simplified model)
            energy_cost_estimate = max(0, 1 - current_puell) * 100  # Percentage of revenue
            
            # Mining margin estimate
            mining_margin_estimate = max(0, (current_puell - 0.5) * 50)  # Simplified margin
            
            # Network security score
            network_security_score = min(1.0, current_puell / 2.0)  # Higher Puell = more security
            
            # Revenue sustainability
            if current_puell > 2.0:
                sustainability = "High - Sustainable mining operations"
            elif current_puell > 1.0:
                sustainability = "Medium - Stable mining conditions"
            elif current_puell > 0.5:
                sustainability = "Low - Challenging conditions"
            else:
                sustainability = "Critical - Potential miner capitulation"
            
            return MiningEconomics(
                mining_efficiency_score=float(mining_efficiency_score),
                difficulty_adjustment_impact=float(difficulty_adjustment_impact),
                miner_capitulation_risk=float(miner_capitulation_risk),
                hash_rate_correlation=float(hash_rate_correlation),
                energy_cost_estimate=float(energy_cost_estimate),
                mining_margin_estimate=float(mining_margin_estimate),
                network_security_score=float(network_security_score),
                miner_revenue_sustainability=sustainability
            )
            
        except Exception as e:
            logger.error(f"Error in mining economics analysis: {str(e)}")
            return MiningEconomics(
                 mining_efficiency_score=0.5,
                 difficulty_adjustment_impact=0.0,
                 miner_capitulation_risk=0.5,
                 hash_rate_correlation=0.0,
                 energy_cost_estimate=50.0,
                 mining_margin_estimate=25.0,
                 network_security_score=0.5,
                 miner_revenue_sustainability="Unknown"
             )
    
    def _perform_regime_analysis(self, puell_values: np.ndarray) -> RegimeAnalysis:
        """Perform regime switching analysis using Markov models"""
        try:
            if len(puell_values) < 20:
                raise ValueError("Insufficient data for regime analysis")
            
            # Prepare data for regime switching model
            puell_series = pd.Series(puell_values)
            
            # Fit Markov Regime Switching model
            try:
                self.regime_model = MarkovRegression(
                    puell_series, k_regimes=self.n_regimes, trend='c', switching_variance=True
                )
                regime_fit = self.regime_model.fit()
                
                # Get regime probabilities and current regime
                regime_probs = regime_fit.smoothed_marginal_probabilities
                current_regime = int(np.argmax(regime_probs.iloc[-1]))
                
                # Calculate regime characteristics
                regime_characteristics = {}
                for i in range(self.n_regimes):
                    regime_mask = np.argmax(regime_probs.values, axis=1) == i
                    if np.sum(regime_mask) > 0:
                        regime_data = puell_values[regime_mask]
                        regime_characteristics[i] = {
                            'mean': float(np.mean(regime_data)),
                            'std': float(np.std(regime_data)),
                            'duration': float(np.sum(regime_mask))
                        }
                
                # Transition probabilities
                transition_probs = regime_fit.regime_transition
                
                # Expected duration in each regime
                expected_duration = []
                for i in range(self.n_regimes):
                    if i < len(transition_probs) and transition_probs[i, i] < 1.0:
                        duration = 1 / (1 - transition_probs[i, i])
                    else:
                        duration = 10.0  # Default
                    expected_duration.append(float(duration))
                
                # Regime stability (how stable current regime is)
                regime_stability = float(transition_probs[current_regime, current_regime]) if current_regime < len(transition_probs) else 0.5
                
                # Regime descriptions
                regime_descriptions = {}
                for i in range(self.n_regimes):
                    if i in regime_characteristics:
                        mean_val = regime_characteristics[i]['mean']
                        if mean_val > 1.5:
                            regime_descriptions[i] = "High Profitability Regime"
                        elif mean_val > 0.8:
                            regime_descriptions[i] = "Normal Profitability Regime"
                        else:
                            regime_descriptions[i] = "Low Profitability Regime"
                    else:
                        regime_descriptions[i] = "Unknown Regime"
                
                return RegimeAnalysis(
                    current_regime=current_regime,
                    regime_probabilities=regime_probs.iloc[-1].tolist(),
                    regime_characteristics=regime_characteristics,
                    transition_probabilities=transition_probs,
                    expected_regime_duration=expected_duration,
                    regime_stability=regime_stability,
                    regime_description=regime_descriptions
                )
                
            except Exception as e:
                logger.warning(f"Markov regime analysis failed: {str(e)}")
                # Fallback to simple regime detection
                return self._simple_regime_analysis(puell_values)
                
        except Exception as e:
            logger.error(f"Error in regime analysis: {str(e)}")
            return self._simple_regime_analysis(puell_values)
    
    def _simple_regime_analysis(self, puell_values: np.ndarray) -> RegimeAnalysis:
        """Simple fallback regime analysis"""
        current_puell = puell_values[-1]
        
        if current_puell > 1.5:
            current_regime = 0  # High regime
        else:
            current_regime = 1  # Low regime
        
        return RegimeAnalysis(
            current_regime=current_regime,
            regime_probabilities=[0.7, 0.3] if current_regime == 0 else [0.3, 0.7],
            regime_characteristics={
                0: {'mean': 2.0, 'std': 0.5, 'duration': 100},
                1: {'mean': 0.8, 'std': 0.3, 'duration': 200}
            },
            transition_probabilities=np.array([[0.9, 0.1], [0.1, 0.9]]),
            expected_regime_duration=[10.0, 10.0],
            regime_stability=0.9,
            regime_description={0: "High Profitability Regime", 1: "Low Profitability Regime"}
        )
    
    def _perform_volatility_analysis(self, puell_values: np.ndarray) -> VolatilityAnalysis:
        """Perform GARCH volatility analysis"""
        try:
            if len(puell_values) < 30:
                raise ValueError("Insufficient data for volatility analysis")
            
            # Calculate returns
            returns = np.diff(np.log(puell_values + 1e-8)) * 100  # Log returns in percentage
            
            try:
                # Fit GARCH(1,1) model
                self.garch_model = arch_model(returns, vol='Garch', p=1, q=1)
                garch_fit = self.garch_model.fit(disp='off')
                
                # Current volatility
                current_volatility = float(garch_fit.conditional_volatility.iloc[-1])
                
                # Volatility forecast
                forecast = garch_fit.forecast(horizon=5)
                volatility_forecast = forecast.variance.iloc[-1].tolist()
                
                # GARCH parameters
                garch_params = {
                    'omega': float(garch_fit.params['omega']),
                    'alpha': float(garch_fit.params['alpha[1]']),
                    'beta': float(garch_fit.params['beta[1]'])
                }
                
                # Volatility persistence
                volatility_persistence = float(garch_params['alpha'] + garch_params['beta'])
                
            except Exception as e:
                logger.warning(f"GARCH modeling failed: {str(e)}")
                # Fallback to simple volatility measures
                current_volatility = float(np.std(returns[-30:]) if len(returns) >= 30 else np.std(returns))
                volatility_forecast = [current_volatility] * 5
                garch_params = {'omega': 0.0, 'alpha': 0.1, 'beta': 0.8}
                volatility_persistence = 0.9
            
            # Determine volatility regime
            vol_percentile = stats.percentileofscore(np.abs(returns), current_volatility)
            if vol_percentile > 80:
                volatility_regime = "High Volatility"
            elif vol_percentile > 60:
                volatility_regime = "Elevated Volatility"
            elif vol_percentile > 40:
                volatility_regime = "Normal Volatility"
            else:
                volatility_regime = "Low Volatility"
            
            # Volatility clustering score
            abs_returns = np.abs(returns)
            clustering_score = float(np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]) if len(abs_returns) > 1 else 0.0
            
            # Heteroskedasticity test
            try:
                from statsmodels.stats.diagnostic import het_breuschpagan
                # Simple regression for heteroskedasticity test
                X = np.arange(len(returns)).reshape(-1, 1)
                lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(returns, X)
                heteroskedasticity_test = {
                    'lm_statistic': float(lm_stat),
                    'lm_pvalue': float(lm_pvalue),
                    'f_statistic': float(f_stat),
                    'f_pvalue': float(f_pvalue)
                }
            except Exception:
                heteroskedasticity_test = {'error': 'Test failed'}
            
            return VolatilityAnalysis(
                current_volatility=current_volatility,
                volatility_forecast=volatility_forecast,
                garch_parameters=garch_params,
                volatility_regime=volatility_regime,
                clustering_score=clustering_score,
                heteroskedasticity_test=heteroskedasticity_test,
                volatility_persistence=volatility_persistence
            )
            
        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
            return VolatilityAnalysis(
                current_volatility=1.0,
                volatility_forecast=[1.0] * 5,
                garch_parameters={'omega': 0.0, 'alpha': 0.1, 'beta': 0.8},
                volatility_regime="Unknown",
                clustering_score=0.0,
                heteroskedasticity_test={},
                volatility_persistence=0.9
             )
    
    def _perform_predictive_modeling(self, puell_values: np.ndarray, prices: np.ndarray = None) -> PredictiveMetrics:
        """Perform predictive modeling using machine learning"""
        try:
            if len(puell_values) < 50:
                raise ValueError("Insufficient data for predictive modeling")
            
            # Prepare features
            features = []
            targets = []
            
            # Create lagged features
            for i in range(10, len(puell_values) - self.forecast_horizon):
                # Features: lagged Puell values, moving averages, volatility
                feature_vector = [
                    puell_values[i-1],  # Previous Puell
                    puell_values[i-2],  # 2 periods ago
                    np.mean(puell_values[i-5:i]),  # 5-period MA
                    np.mean(puell_values[i-10:i]),  # 10-period MA
                    np.std(puell_values[i-10:i]),  # 10-period volatility
                    np.max(puell_values[i-10:i]),  # 10-period max
                    np.min(puell_values[i-10:i]),  # 10-period min
                ]
                
                # Add price features if available
                if prices is not None and len(prices) == len(puell_values):
                    feature_vector.extend([
                        prices[i-1] / prices[i-2] - 1,  # Price return
                        np.std(prices[i-10:i]) / np.mean(prices[i-10:i])  # Price volatility
                    ])
                
                features.append(feature_vector)
                targets.append(puell_values[i + self.forecast_horizon])
            
            if len(features) < 20:
                raise ValueError("Insufficient training data")
            
            # Convert to arrays
            X = np.array(features)
            y = np.array(targets)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            split_idx = int(0.8 * len(X_scaled))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Train model
            self.rf_model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = self.rf_model.predict(X_test)
            
            # Calculate performance metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_names = ['Puell_t-1', 'Puell_t-2', 'MA_5', 'MA_10', 'Vol_10', 'Max_10', 'Min_10']
            if prices is not None:
                feature_names.extend(['Price_Return', 'Price_Vol'])
            
            feature_importance = dict(zip(feature_names, self.rf_model.feature_importances_))
            
            # Generate forecast
            latest_features = features[-1]
            latest_features_scaled = self.scaler.transform([latest_features])
            ml_forecast = [float(self.rf_model.predict(latest_features_scaled)[0])]
            
            # Calculate prediction intervals using quantile regression approximation
            predictions_all = self.rf_model.predict(X_scaled)
            residuals = y - predictions_all[:len(y)]
            residual_std = np.std(residuals)
            
            # Confidence intervals (assuming normal distribution of residuals)
            z_score = stats.norm.ppf((1 + self.confidence_level) / 2)
            forecast_confidence = [residual_std]
            
            # Trend prediction
            recent_trend = np.mean(np.diff(puell_values[-10:]))
            if recent_trend > 0.01:
                trend_prediction = "Upward"
            elif recent_trend < -0.01:
                trend_prediction = "Downward"
            else:
                trend_prediction = "Sideways"
            
            # Reversal probability (simplified)
            current_puell = puell_values[-1]
            if current_puell > np.percentile(puell_values, 80):
                reversal_probability = 0.7
            elif current_puell < np.percentile(puell_values, 20):
                reversal_probability = 0.7
            else:
                reversal_probability = 0.3
            
            return PredictiveMetrics(
                ml_forecast=ml_forecast,
                forecast_confidence=forecast_confidence,
                feature_importance=feature_importance,
                model_accuracy=float(r2),
                trend_prediction=trend_prediction,
                reversal_probability=float(reversal_probability),
                time_to_next_extreme=None,
                forecast_horizon=self.forecast_horizon
            )
            
        except Exception as e:
            logger.error(f"Error in predictive modeling: {str(e)}")
            current_puell = puell_values[-1] if len(puell_values) > 0 else 1.0
            return PredictiveMetrics(
                ml_forecast=[current_puell],
                forecast_confidence=[0.1],
                feature_importance={},
                model_accuracy=0.0,
                trend_prediction="Unknown",
                reversal_probability=0.5,
                time_to_next_extreme=None,
                forecast_horizon=self.forecast_horizon
            )
    
    def _perform_anomaly_detection(self, puell_values: np.ndarray) -> AnomalyDetection:
        """Perform anomaly detection using machine learning"""
        try:
            if len(puell_values) < 20:
                raise ValueError("Insufficient data for anomaly detection")
            
            # Prepare features for anomaly detection
            features = []
            for i in range(5, len(puell_values)):
                feature_vector = [
                    puell_values[i],  # Current value
                    np.mean(puell_values[i-5:i]),  # 5-period MA
                    np.std(puell_values[i-5:i]),  # 5-period volatility
                    puell_values[i] / np.mean(puell_values[i-5:i]) - 1,  # Deviation from MA
                    (puell_values[i] - np.min(puell_values[i-5:i])) / (np.max(puell_values[i-5:i]) - np.min(puell_values[i-5:i]) + 1e-8)  # Normalized position
                ]
                features.append(feature_vector)
            
            X = np.array(features)
            
            # Fit anomaly detection model
            anomaly_scores = self.anomaly_detector.fit_predict(X)
            anomaly_scores_prob = self.anomaly_detector.decision_function(X)
            
            # Current anomaly status
            current_anomaly = int(anomaly_scores[-1]) == -1
            current_anomaly_score = float(anomaly_scores_prob[-1])
            
            # Historical anomalies
            anomaly_indices = np.where(anomaly_scores == -1)[0]
            historical_anomalies = [{
                'index': int(idx + 5),
                'value': float(puell_values[idx + 5]),
                'type': 'High' if puell_values[idx + 5] > 2.0 else 'Low' if puell_values[idx + 5] < 0.3 else 'Statistical'
            } for idx in anomaly_indices if idx + 5 < len(puell_values)]
            
            # Classify anomaly type based on Puell value
            anomaly_type = None
            if current_anomaly:
                current_puell = puell_values[-1]
                if current_puell > 2.0:
                    anomaly_type = "Extreme Profitability"
                elif current_puell < 0.3:
                    anomaly_type = "Mining Distress"
                else:
                    anomaly_type = "Statistical Outlier"
            
            # Anomaly severity
            if abs(current_anomaly_score) > 0.5:
                anomaly_severity = "High"
            elif abs(current_anomaly_score) > 0.2:
                anomaly_severity = "Medium"
            else:
                anomaly_severity = "Low"
            
            # Context explanation
            if current_anomaly:
                if anomaly_type == "Extreme Profitability":
                    context_explanation = "Mining profitability is extremely high, potentially indicating market top"
                elif anomaly_type == "Mining Distress":
                    context_explanation = "Mining profitability is extremely low, potentially indicating market bottom"
                else:
                    context_explanation = "Statistical anomaly detected in mining profitability patterns"
            else:
                context_explanation = "No significant anomaly detected in current mining conditions"
            
            return AnomalyDetection(
                anomaly_score=current_anomaly_score,
                is_anomaly=current_anomaly,
                anomaly_type=anomaly_type,
                historical_anomalies=historical_anomalies,
                anomaly_severity=anomaly_severity,
                context_explanation=context_explanation
            )
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {str(e)}")
            return AnomalyDetection(
                anomaly_score=0.0,
                is_anomaly=False,
                anomaly_type=None,
                historical_anomalies=[],
                anomaly_severity="Unknown",
                context_explanation="Anomaly detection failed"
            )
    
    def analyze(self, historical_data: pd.DataFrame) -> PuellResult:
        """Perform comprehensive Puell Multiple analysis with advanced features
        
        Args:
            historical_data: DataFrame with columns ['date', 'daily_issuance_usd', 'issuance_ma_365']
                           or additional columns for advanced analysis
        """
        try:
            puell_values = []
            timestamps = []
            
            for _, row in historical_data.iterrows():
                puell = self.calculate_puell_multiple(
                    row['daily_issuance_usd'], 
                    row['issuance_ma_365']
                )
                puell_values.append(puell)
                timestamps.append(row['date'])
            
            puell_array = np.array(puell_values)
            current_puell = puell_values[-1] if puell_values else 1.0
            
            # Calculate percentile
            puell_percentile = stats.percentileofscore(puell_values, current_puell)
            
            # Determine mining profitability
            mining_profitability = self.determine_mining_profitability(current_puell)
            
            # Determine market cycle phase
            market_cycle_phase = self.determine_market_cycle_phase(puell_percentile)
            
            # Calculate bands
            puell_bands = self.calculate_puell_bands(puell_values)
            
            # Initialize advanced analysis components
            cycle_analysis = None
            mining_economics = None
            regime_analysis = None
            volatility_analysis = None
            predictive_metrics = None
            anomaly_detection = None
            
            # Statistical measures
            puell_volatility = float(np.std(puell_values))
            max_drawdown = float(1 - np.min(puell_values) / np.max(puell_values)) if np.max(puell_values) > 0 else 0.0
            
            # Calculate returns for Sharpe/Sortino ratios
            if len(puell_values) > 1:
                returns = np.diff(puell_values) / np.array(puell_values[:-1])
                sharpe_ratio = float(np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0
                downside_returns = returns[returns < 0]
                sortino_ratio = float(np.mean(returns) / np.std(downside_returns)) if len(downside_returns) > 0 and np.std(downside_returns) > 0 else 0.0
                skewness = float(stats.skew(puell_values))
                kurtosis = float(stats.kurtosis(puell_values))
                jb_stat, jb_pvalue = stats.jarque_bera(puell_values)
                jarque_bera_stat = float(jb_stat)
                jarque_bera_pvalue = float(jb_pvalue)
            else:
                sharpe_ratio = sortino_ratio = skewness = kurtosis = 0.0
                jarque_bera_stat = jarque_bera_pvalue = 0.0
            
            # Autocorrelation
            if len(puell_values) > 10:
                autocorr = [float(np.corrcoef(puell_values[:-i], puell_values[i:])[0, 1]) for i in range(1, min(11, len(puell_values)//2))]
            else:
                autocorr = []
            
            # Cross-correlations
            price_correlation = None
            hash_rate_correlation = None
            difficulty_correlation = None
            volume_correlation = None
            
            if 'price' in historical_data.columns:
                price_values = historical_data['price'].values
                if len(price_values) == len(puell_values):
                    price_correlation = float(np.corrcoef(puell_values, price_values)[0, 1])
            
            if 'hash_rate' in historical_data.columns:
                hash_rate_values = historical_data['hash_rate'].values
                if len(hash_rate_values) == len(puell_values):
                    hash_rate_correlation = float(np.corrcoef(puell_values, hash_rate_values)[0, 1])
            
            if 'difficulty' in historical_data.columns:
                difficulty_values = historical_data['difficulty'].values
                if len(difficulty_values) == len(puell_values):
                    difficulty_correlation = float(np.corrcoef(puell_values, difficulty_values)[0, 1])
            
            if 'volume' in historical_data.columns:
                volume_values = historical_data['volume'].values
                if len(volume_values) == len(puell_values):
                    volume_correlation = float(np.corrcoef(puell_values, volume_values)[0, 1])
            
            # Confidence intervals
            alpha = 1 - self.confidence_level
            puell_confidence_interval = (
                float(np.percentile(puell_values, alpha/2 * 100)),
                float(np.percentile(puell_values, (1 - alpha/2) * 100))
            )
            
            # Perform advanced analysis if enabled
            if self.enable_cycle_analysis and len(puell_values) >= 50:
                cycle_analysis = self._perform_cycle_analysis(puell_array, timestamps)
            
            if self.enable_mining_economics:
                mining_economics = self._perform_mining_economics_analysis(historical_data, puell_array)
            
            if self.enable_regime_analysis and len(puell_values) >= 20:
                regime_analysis = self._perform_regime_analysis(puell_array)
            
            if self.enable_volatility_analysis and len(puell_values) >= 30:
                volatility_analysis = self._perform_volatility_analysis(puell_array)
            
            if self.enable_predictive_modeling and len(puell_values) >= 50:
                prices = historical_data['price'].values if 'price' in historical_data.columns else None
                predictive_metrics = self._perform_predictive_modeling(puell_array, prices)
            
            if self.enable_anomaly_detection and len(puell_values) >= 20:
                anomaly_detection = self._perform_anomaly_detection(puell_array)
            
            # Enhanced analysis calculations
            mining_profitability_cycles = None
            halving_effects_analysis = None
            supply_shock_analysis = None
            
            try:
                # Mining profitability cycles analysis
                mining_profitability_cycles = self._analyze_mining_profitability_cycles(historical_data, puell_array)
                logger.info("Mining profitability cycles analysis completed")
            except Exception as e:
                logger.error(f"Error in mining profitability cycles analysis: {str(e)}")
                mining_profitability_cycles = None
            
            try:
                # Halving effects analysis
                halving_effects_analysis = self._analyze_halving_effects(historical_data, puell_array)
                logger.info("Halving effects analysis completed")
            except Exception as e:
                logger.error(f"Error in halving effects analysis: {str(e)}")
                halving_effects_analysis = None
            
            try:
                # Supply shock analysis
                supply_shock_analysis = self._analyze_supply_shock(historical_data, puell_array)
                logger.info("Supply shock analysis completed")
            except Exception as e:
                logger.error(f"Error in supply shock analysis: {str(e)}")
                supply_shock_analysis = None
            
            # Forecast confidence bands
            forecast_confidence_bands = None
            if predictive_metrics:
                forecast_confidence_bands = {
                    'upper': [current_puell * 1.2] * self.forecast_horizon,
                    'lower': [current_puell * 0.8] * self.forecast_horizon
                }
            
            return PuellResult(
                current_puell=current_puell,
                puell_percentile=puell_percentile,
                mining_profitability=mining_profitability,
                market_cycle_phase=market_cycle_phase,
                historical_puell=puell_values,
                puell_bands=puell_bands,
                timestamps=timestamps,
                cycle_analysis=cycle_analysis,
                mining_economics=mining_economics,
                regime_analysis=regime_analysis,
                volatility_analysis=volatility_analysis,
                predictive_metrics=predictive_metrics,
                anomaly_detection=anomaly_detection,
                puell_volatility=puell_volatility,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                skewness=skewness,
                kurtosis=kurtosis,
                jarque_bera_stat=jarque_bera_stat,
                jarque_bera_pvalue=jarque_bera_pvalue,
                autocorrelation=autocorr,
                price_correlation=price_correlation,
                hash_rate_correlation=hash_rate_correlation,
                difficulty_correlation=difficulty_correlation,
                volume_correlation=volume_correlation,
                puell_confidence_interval=puell_confidence_interval,
                forecast_confidence_bands=forecast_confidence_bands,
                mining_profitability_cycles=mining_profitability_cycles,
                halving_effects_analysis=halving_effects_analysis,
                supply_shock_analysis=supply_shock_analysis
            )
            
        except Exception as e:
            logger.error(f"Error in Puell Multiple analysis: {str(e)}")
            raise
    
    def _analyze_mining_profitability_cycles(self, historical_data: pd.DataFrame, puell_array: np.ndarray) -> MiningProfitabilityCycles:
        """Analyze mining profitability cycles and patterns"""
        try:
            # Calculate cycle metrics
            cycle_length = 365 * 4  # Approximate 4-year cycle
            current_cycle_position = len(puell_array) % cycle_length / cycle_length
            
            # Profitability phases analysis
            high_profit_threshold = np.percentile(puell_array, 80)
            low_profit_threshold = np.percentile(puell_array, 20)
            
            profitability_phases = []
            current_phase = "neutral"
            phase_start = 0
            
            for i, value in enumerate(puell_array):
                if value > high_profit_threshold and current_phase != "high":
                    if current_phase != "high":
                        profitability_phases.append({
                            'phase': current_phase,
                            'start_index': phase_start,
                            'end_index': i-1,
                            'duration': i - phase_start
                        })
                    current_phase = "high"
                    phase_start = i
                elif value < low_profit_threshold and current_phase != "low":
                    if current_phase != "low":
                        profitability_phases.append({
                            'phase': current_phase,
                            'start_index': phase_start,
                            'end_index': i-1,
                            'duration': i - phase_start
                        })
                    current_phase = "low"
                    phase_start = i
                elif low_profit_threshold <= value <= high_profit_threshold and current_phase != "neutral":
                    profitability_phases.append({
                        'phase': current_phase,
                        'start_index': phase_start,
                        'end_index': i-1,
                        'duration': i - phase_start
                    })
                    current_phase = "neutral"
                    phase_start = i
            
            # Add final phase
            profitability_phases.append({
                'phase': current_phase,
                'start_index': phase_start,
                'end_index': len(puell_array)-1,
                'duration': len(puell_array) - phase_start
            })
            
            # Calculate cycle statistics
            avg_cycle_duration = np.mean([phase['duration'] for phase in profitability_phases])
            cycle_volatility = np.std(puell_array)
            
            # Determine current profitability state
            current_puell = puell_array[-1]
            if current_puell > high_profit_threshold:
                current_profitability_state = "High Profitability"
            elif current_puell < low_profit_threshold:
                current_profitability_state = "Low Profitability"
            else:
                current_profitability_state = "Normal Profitability"
            
            return MiningProfitabilityCycles(
                current_cycle_position=float(current_cycle_position),
                profitability_phases=profitability_phases,
                avg_cycle_duration=float(avg_cycle_duration),
                cycle_volatility=float(cycle_volatility),
                current_profitability_state=current_profitability_state,
                profit_cycle_strength=float(abs(current_puell - np.mean(puell_array)) / np.std(puell_array))
            )
            
        except Exception as e:
            logger.error(f"Error in mining profitability cycles analysis: {str(e)}")
            return MiningProfitabilityCycles(
                current_cycle_position=0.0,
                profitability_phases=[],
                avg_cycle_duration=0.0,
                cycle_volatility=0.0,
                current_profitability_state="Unknown",
                profit_cycle_strength=0.0
            )
    
    def _analyze_halving_effects(self, historical_data: pd.DataFrame, puell_array: np.ndarray) -> HalvingEffectsAnalysis:
        """Analyze Bitcoin halving effects on Puell Multiple"""
        try:
            # Bitcoin halving dates (approximate)
            halving_dates = [
                pd.Timestamp('2012-11-28'),
                pd.Timestamp('2016-07-09'),
                pd.Timestamp('2020-05-11'),
                pd.Timestamp('2024-04-20')
            ]
            
            # Calculate days since last halving
            if 'date' in historical_data.columns:
                last_date = pd.to_datetime(historical_data['date'].iloc[-1])
                last_halving = max([d for d in halving_dates if d <= last_date], default=halving_dates[0])
                days_since_halving = (last_date - last_halving).days
            else:
                days_since_halving = 365  # Default assumption
            
            # Pre/post halving analysis
            pre_halving_periods = []
            post_halving_periods = []
            
            if 'date' in historical_data.columns:
                dates = pd.to_datetime(historical_data['date'])
                for halving_date in halving_dates:
                    # 6 months before and after halving
                    pre_period = (dates >= halving_date - pd.Timedelta(days=180)) & (dates < halving_date)
                    post_period = (dates >= halving_date) & (dates < halving_date + pd.Timedelta(days=180))
                    
                    if pre_period.any():
                        pre_halving_periods.append(puell_array[pre_period].mean())
                    if post_period.any():
                        post_halving_periods.append(puell_array[post_period].mean())
            
            # Calculate halving impact metrics
            if pre_halving_periods and post_halving_periods:
                avg_pre_halving = np.mean(pre_halving_periods)
                avg_post_halving = np.mean(post_halving_periods)
                halving_impact_magnitude = (avg_post_halving - avg_pre_halving) / avg_pre_halving
            else:
                avg_pre_halving = avg_post_halving = halving_impact_magnitude = 0.0
            
            # Determine halving cycle phase
            cycle_days = 365 * 4  # 4-year cycle
            cycle_position = (days_since_halving % cycle_days) / cycle_days
            
            if cycle_position < 0.25:
                halving_cycle_phase = "Post-Halving Accumulation"
            elif cycle_position < 0.5:
                halving_cycle_phase = "Mid-Cycle Growth"
            elif cycle_position < 0.75:
                halving_cycle_phase = "Late-Cycle Euphoria"
            else:
                halving_cycle_phase = "Pre-Halving Correction"
            
            # Expected next halving impact
            next_halving_date = min([d for d in halving_dates if d > last_date], default=pd.Timestamp('2028-04-20'))
            days_to_next_halving = (next_halving_date - last_date).days
            
            return HalvingEffectsAnalysis(
                days_since_halving=int(days_since_halving),
                days_to_next_halving=int(days_to_next_halving),
                halving_cycle_phase=halving_cycle_phase,
                pre_halving_avg=float(avg_pre_halving),
                post_halving_avg=float(avg_post_halving),
                halving_impact_magnitude=float(halving_impact_magnitude),
                expected_next_halving_impact=float(abs(halving_impact_magnitude) * 0.8)  # Diminishing returns
            )
            
        except Exception as e:
            logger.error(f"Error in halving effects analysis: {str(e)}")
            return HalvingEffectsAnalysis(
                days_since_halving=0,
                days_to_next_halving=0,
                halving_cycle_phase="Unknown",
                pre_halving_avg=0.0,
                post_halving_avg=0.0,
                halving_impact_magnitude=0.0,
                expected_next_halving_impact=0.0
            )
    
    def _analyze_supply_shock(self, historical_data: pd.DataFrame, puell_array: np.ndarray) -> SupplyShockAnalysis:
        """Analyze supply shock indicators and market impact"""
        try:
            # Calculate supply shock metrics
            current_puell = puell_array[-1]
            puell_ma_30 = np.mean(puell_array[-30:]) if len(puell_array) >= 30 else current_puell
            puell_ma_90 = np.mean(puell_array[-90:]) if len(puell_array) >= 90 else current_puell
            
            # Supply shock intensity based on Puell Multiple deviation
            shock_threshold_high = np.percentile(puell_array, 95)
            shock_threshold_low = np.percentile(puell_array, 5)
            
            if current_puell > shock_threshold_high:
                supply_shock_intensity = "Extreme Positive"
                shock_magnitude = (current_puell - shock_threshold_high) / shock_threshold_high
            elif current_puell < shock_threshold_low:
                supply_shock_intensity = "Extreme Negative"
                shock_magnitude = (shock_threshold_low - current_puell) / shock_threshold_low
            elif current_puell > np.percentile(puell_array, 80):
                supply_shock_intensity = "Moderate Positive"
                shock_magnitude = (current_puell - np.percentile(puell_array, 80)) / np.percentile(puell_array, 80)
            elif current_puell < np.percentile(puell_array, 20):
                supply_shock_intensity = "Moderate Negative"
                shock_magnitude = (np.percentile(puell_array, 20) - current_puell) / np.percentile(puell_array, 20)
            else:
                supply_shock_intensity = "Normal"
                shock_magnitude = 0.0
            
            # Market impact assessment
            if supply_shock_intensity in ["Extreme Positive", "Moderate Positive"]:
                market_impact_assessment = "Potential selling pressure from miners due to high profitability"
            elif supply_shock_intensity in ["Extreme Negative", "Moderate Negative"]:
                market_impact_assessment = "Potential miner capitulation and reduced selling pressure"
            else:
                market_impact_assessment = "Normal market conditions with balanced miner economics"
            
            # Recovery timeline estimation
            if supply_shock_intensity.startswith("Extreme"):
                recovery_timeline_days = 90
            elif supply_shock_intensity.startswith("Moderate"):
                recovery_timeline_days = 30
            else:
                recovery_timeline_days = 0
            
            # Historical shock comparison
            historical_shocks = []
            for i in range(len(puell_array)):
                if puell_array[i] > shock_threshold_high or puell_array[i] < shock_threshold_low:
                    historical_shocks.append({
                        'index': i,
                        'value': float(puell_array[i]),
                        'type': 'Positive' if puell_array[i] > shock_threshold_high else 'Negative'
                    })
            
            return SupplyShockAnalysis(
                supply_shock_intensity=supply_shock_intensity,
                shock_magnitude=float(shock_magnitude),
                market_impact_assessment=market_impact_assessment,
                recovery_timeline_days=recovery_timeline_days,
                historical_shock_frequency=len(historical_shocks),
                shock_persistence_score=float(abs(current_puell - puell_ma_30) / np.std(puell_array[-30:]) if len(puell_array) >= 30 else 0.0)
            )
            
        except Exception as e:
            logger.error(f"Error in supply shock analysis: {str(e)}")
            return SupplyShockAnalysis(
                supply_shock_intensity="Unknown",
                shock_magnitude=0.0,
                market_impact_assessment="Analysis failed",
                recovery_timeline_days=0,
                historical_shock_frequency=0,
                shock_persistence_score=0.0
            )

# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    
    # Sample Puell data with additional metrics
    puell_data = []
    np.random.seed(42)  # For reproducible results
    
    for i, date in enumerate(dates):
        # Simulate market cycles
        cycle_factor = np.sin(2 * np.pi * i / 365) * 0.3 + 1.0
        noise = 0.1 * np.random.randn()
        
        daily_issuance = 900 * 50000 * cycle_factor * (1 + noise)  # ~900 BTC * $50k
        issuance_ma = daily_issuance * (1 + 0.02 * np.random.randn())
        
        # Additional data for enhanced analysis
        price = 50000 * cycle_factor * (1 + 0.2 * np.random.randn())
        hash_rate = 200e18 * cycle_factor * (1 + 0.1 * np.random.randn())
        difficulty = hash_rate / 600  # Simplified difficulty calculation
        volume = 1e9 * (1 + 0.3 * np.random.randn())
        
        puell_data.append({
            'date': date,
            'daily_issuance_usd': daily_issuance,
            'issuance_ma_365': issuance_ma,
            'price': price,
            'hash_rate': hash_rate,
            'difficulty': difficulty,
            'volume': volume
        })
    
    # Create DataFrame
    historical_data = pd.DataFrame(puell_data)
    
    # Test basic model
    print("=== Basic Puell Multiple Analysis ===")
    basic_model = PuellMultipleModel("BTC")
    basic_result = basic_model.analyze(historical_data)
    
    print(f"Current Puell Multiple: {basic_result.current_puell:.2f}")
    print(f"Puell Percentile: {basic_result.puell_percentile:.1f}%")
    print(f"Mining Profitability: {basic_result.mining_profitability}")
    print(f"Market Cycle Phase: {basic_result.market_cycle_phase}")
    print(f"Puell Bands: {basic_result.puell_bands}")
    print(f"Volatility: {basic_result.puell_volatility:.3f}")
    print(f"Max Drawdown: {basic_result.max_drawdown:.3f}")
    print(f"Price Correlation: {basic_result.price_correlation:.3f}")
    
    # Test enhanced model with all features
    print("\n=== Enhanced Puell Multiple Analysis ===")
    enhanced_model = PuellMultipleModel(
        "BTC",
        enable_cycle_analysis=True,
        enable_mining_economics=True,
        enable_regime_analysis=True,
        enable_volatility_analysis=True,
        enable_predictive_modeling=True,
        enable_anomaly_detection=True,
        forecast_horizon=30
    )
    
    enhanced_result = enhanced_model.analyze(historical_data)
    
    print(f"Current Puell Multiple: {enhanced_result.current_puell:.2f}")
    print(f"Mining Profitability: {enhanced_result.mining_profitability}")
    print(f"Market Cycle Phase: {enhanced_result.market_cycle_phase}")
    
    # Cycle Analysis
    if enhanced_result.cycle_analysis:
        print(f"\nCycle Analysis:")
        print(f"  Dominant Cycle Period: {enhanced_result.cycle_analysis.dominant_cycle_period:.1f} days")
        print(f"  Cycle Strength: {enhanced_result.cycle_analysis.cycle_strength:.3f}")
        print(f"  Current Phase: {enhanced_result.cycle_analysis.cycle_phase}")
        print(f"  Cycle Position: {enhanced_result.cycle_analysis.cycle_position:.3f}")
    
    # Mining Economics
    if enhanced_result.mining_economics:
        print(f"\nMining Economics:")
        print(f"  Efficiency Score: {enhanced_result.mining_economics.mining_efficiency_score:.3f}")
        print(f"  Capitulation Risk: {enhanced_result.mining_economics.miner_capitulation_risk:.3f}")
        print(f"  Revenue Sustainability: {enhanced_result.mining_economics.miner_revenue_sustainability}")
    
    # Regime Analysis
    if enhanced_result.regime_analysis:
        print(f"\nRegime Analysis:")
        print(f"  Current Regime: {enhanced_result.regime_analysis.current_regime}")
        print(f"  Regime Description: {enhanced_result.regime_analysis.regime_description.get(enhanced_result.regime_analysis.current_regime, 'Unknown')}")
        print(f"  Regime Stability: {enhanced_result.regime_analysis.regime_stability:.3f}")
    
    # Volatility Analysis
    if enhanced_result.volatility_analysis:
        print(f"\nVolatility Analysis:")
        print(f"  Current Volatility: {enhanced_result.volatility_analysis.current_volatility:.3f}")
        print(f"  Volatility Regime: {enhanced_result.volatility_analysis.volatility_regime}")
        print(f"  Clustering Score: {enhanced_result.volatility_analysis.clustering_score:.3f}")
    
    # Predictive Metrics
    if enhanced_result.predictive_metrics:
        print(f"\nPredictive Analysis:")
        print(f"  ML Forecast: {enhanced_result.predictive_metrics.ml_forecast[0]:.3f}")
        print(f"  Model Accuracy: {enhanced_result.predictive_metrics.model_accuracy:.3f}")
        print(f"  Trend Prediction: {enhanced_result.predictive_metrics.trend_prediction}")
        print(f"  Reversal Probability: {enhanced_result.predictive_metrics.reversal_probability:.3f}")
    
    # Anomaly Detection
    if enhanced_result.anomaly_detection:
        print(f"\nAnomaly Detection:")
        print(f"  Is Anomaly: {enhanced_result.anomaly_detection.is_anomaly}")
        print(f"  Anomaly Score: {enhanced_result.anomaly_detection.anomaly_score:.3f}")
        print(f"  Anomaly Type: {enhanced_result.anomaly_detection.anomaly_type}")
        print(f"  Severity: {enhanced_result.anomaly_detection.anomaly_severity}")
        print(f"  Context: {enhanced_result.anomaly_detection.context_explanation}")
    
    print("\n=== Analysis Complete ===")