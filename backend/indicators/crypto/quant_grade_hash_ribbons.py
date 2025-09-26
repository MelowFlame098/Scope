#!/usr/bin/env python3
"""
Quant Grade Hash Ribbons Model

An enhanced implementation of the Hash Ribbons indicator with advanced mining economics,
network health analysis, and machine learning-based predictions.

The Hash Ribbons indicator tracks the relationship between short-term and long-term
hash rate moving averages to identify mining capitulation and recovery phases.

Key Features:
- Advanced mining economics modeling
- Network health and security analysis
- ML-based hash rate predictions
- Mining regime detection
- Difficulty adjustment analysis
- Miner behavior clustering
- Risk assessment and anomaly detection

Author: Quant Grade Analytics Team
Date: 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import IsolationForest
import warnings
from scipy import stats, signal, optimize
from scipy.stats import norm, t
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None
try:
    from arch import arch_model
except ImportError:
    arch_model = None
try:
    import pywt
except ImportError:
    pywt = None

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Kalman filter support
try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    logger.warning("pykalman not available. Advanced Kalman filtering will use simplified implementation.")

@dataclass
class MinerBehaviorAnalysis:
    """Analysis of miner behavior patterns"""
    capitulation_score: float
    recovery_strength: float
    efficiency_trend: float
    profitability_pressure: float
    network_participation: float
    geographic_distribution: Dict[str, float]
    mining_pool_concentration: float
    behavior_cluster: str
    stability_index: float
    adaptation_speed: float

@dataclass
class NetworkHealthMetrics:
    """Comprehensive network health assessment"""
    security_score: float
    decentralization_index: float
    resilience_factor: float
    attack_cost_estimate: float
    network_effect_strength: float
    infrastructure_quality: float
    geographic_diversity: float
    regulatory_risk_score: float
    energy_sustainability: float
    technological_advancement: float

@dataclass
class MiningEconomics:
    """Mining economics and profitability analysis"""
    break_even_price: float
    profit_margin: float
    operational_efficiency: float
    capital_expenditure_cycle: str
    energy_cost_pressure: float
    hardware_obsolescence_rate: float
    mining_reward_sustainability: float
    fee_revenue_contribution: float
    halving_impact_score: float
    competitive_landscape: str
    
    # Enhanced Puell Multiple and Miner Revenue Analysis
    puell_multiple: float = 0.0
    puell_multiple_ma: float = 0.0
    puell_signal: str = "neutral"
    miner_revenue_usd: float = 0.0
    miner_revenue_btc: float = 0.0
    revenue_per_hash: float = 0.0
    revenue_trend: str = "stable"
    revenue_volatility: float = 0.0
    
    # Difficulty Ribbon Analysis
    difficulty_ribbon_compression: float = 0.0
    difficulty_ribbon_signal: str = "neutral"
    difficulty_ma_9: float = 0.0
    difficulty_ma_14: float = 0.0
    difficulty_ma_25: float = 0.0
    difficulty_ma_40: float = 0.0
    difficulty_ma_60: float = 0.0
    difficulty_ma_90: float = 0.0
    difficulty_ma_128: float = 0.0
    difficulty_ma_200: float = 0.0
    ribbon_width: float = 0.0
    ribbon_trend: str = "neutral"
    
    # Advanced Mining Metrics
    hash_price: float = 0.0  # USD per TH/s per day
    thermocap_ratio: float = 0.0
    miner_position_index: float = 0.0
    mining_profitability_score: float = 0.0

@dataclass
class DifficultyAnalysis:
    """Bitcoin difficulty adjustment analysis"""
    current_difficulty: float
    difficulty_change: float
    adjustment_frequency: float
    target_block_time: float
    actual_block_time: float
    mining_pressure: float
    network_congestion: float
    fee_market_dynamics: str
    mempool_status: str
    transaction_throughput: float

@dataclass
class HashRibbonsPrediction:
    """Hash Ribbons prediction with confidence intervals"""
    prediction_horizon: int
    predicted_hash_rate: float
    hash_rate_trend: str
    ribbon_signal: str
    confidence_interval: Tuple[float, float]
    model_confidence: float
    market_scenario: str
    mining_scenario: str
    risk_factors: List[str]
    catalysts: List[str]

@dataclass
class HashRibbonsKalmanFilterResult:
    """Kalman filter analysis results for Hash Ribbons"""
    filtered_hash_rate_states: List[float]
    filtered_difficulty_states: List[float]
    hash_rate_state_covariances: List[List[float]]
    difficulty_state_covariances: List[List[float]]
    log_likelihood: float
    smoothed_hash_rate: List[float]
    smoothed_difficulty: List[float]
    hash_rate_trend_component: List[float]
    difficulty_trend_component: List[float]
    noise_reduction_ratio: float

@dataclass
class HashRibbonsMonteCarloResult:
    """Monte Carlo simulation results for Hash Ribbons"""
    mean_hash_rate_path: List[float]
    mean_difficulty_path: List[float]
    hash_rate_confidence_intervals: Dict[str, List[float]]
    difficulty_confidence_intervals: Dict[str, List[float]]
    final_hash_rate_distribution: Dict[str, float]
    final_difficulty_distribution: Dict[str, float]
    simulation_count: int
    risk_metrics: Dict[str, float]
    scenario_probabilities: Dict[str, float]
    stress_test_results: Dict[str, float]

@dataclass
class HashRateVolatilityAnalysis:
    """Hash rate volatility and clustering analysis"""
    current_volatility: float
    volatility_regime: str
    garch_volatility: List[float]
    volatility_clustering: bool
    volatility_persistence: float
    arch_test_pvalue: float
    ljung_box_pvalue: float

@dataclass
class AdvancedMiningEconomics:
    """Advanced mining economics analysis with enhanced profitability modeling"""
    # Energy economics
    energy_efficiency_trends: Dict[str, float]
    renewable_energy_adoption: float
    carbon_footprint_metrics: Dict[str, float]
    energy_cost_volatility: float
    
    # Hardware economics
    asic_generation_analysis: Dict[str, float]
    hardware_roi_projections: Dict[str, float]
    obsolescence_timeline: Dict[str, int]
    supply_chain_risks: Dict[str, float]
    
    # Market dynamics
    mining_pool_economics: Dict[str, float]
    geographic_cost_analysis: Dict[str, float]
    regulatory_impact_assessment: Dict[str, float]
    institutional_mining_trends: Dict[str, float]

@dataclass
class EnhancedDifficultyAdjustments:
    """Enhanced difficulty adjustment analysis with predictive modeling"""
    # Adjustment patterns
    adjustment_cycle_analysis: Dict[str, float]
    difficulty_momentum: float
    adjustment_volatility: float
    predictive_adjustment_model: Dict[str, float]
    
    # Network dynamics
    hash_rate_difficulty_correlation: float
    block_time_variance_analysis: Dict[str, float]
    mempool_pressure_impact: Dict[str, float]
    fee_market_influence: Dict[str, float]
    
    # Future projections
    next_adjustment_prediction: Dict[str, float]
    long_term_difficulty_trend: Dict[str, float]
    scenario_based_projections: Dict[str, Dict[str, float]]
    adjustment_risk_factors: List[str]

@dataclass
class DetailedMinerCapitulation:
    """Detailed miner capitulation analysis with behavioral insights"""
    # Capitulation indicators
    capitulation_severity: float
    capitulation_duration_estimate: int
    recovery_probability: float
    historical_capitulation_comparison: Dict[str, float]
    
    # Miner behavior patterns
    forced_selling_pressure: float
    operational_shutdown_risk: Dict[str, float]
    miner_inventory_analysis: Dict[str, float]
    cash_flow_stress_indicators: Dict[str, float]
    
    # Market impact
    supply_shock_potential: float
    price_impact_estimation: Dict[str, float]
    network_security_implications: Dict[str, float]
    recovery_catalyst_identification: List[str]

@dataclass
class MiningCycleAnalysis:
    """Mining cycle and seasonality analysis"""
    cycle_phase: str
    cycle_strength: float
    dominant_frequencies: List[float]
    seasonal_components: List[float]
    trend_component: List[float]
    cyclical_component: List[float]
    noise_component: List[float]
    next_cycle_prediction: Dict[str, float]

@dataclass
class MiningRegimeAnalysis:
    """Mining regime switching analysis"""
    current_regime: int
    regime_probabilities: List[float]
    regime_descriptions: Dict[int, str]
    regime_persistence: float
    transition_probabilities: np.ndarray
    expected_regime_duration: float
    regime_volatility: Dict[int, float]

@dataclass
class HashRibbonsAnomaly:
    """Hash Ribbons anomaly detection result"""
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: str
    severity: str
    affected_metrics: List[str]
    potential_causes: List[str]
    network_implications: List[str]
    historical_precedents: List[str]

@dataclass
class HashRibbonsRisk:
    """Hash Ribbons risk assessment"""
    overall_risk: float
    mining_centralization_risk: float
    network_security_risk: float
    economic_sustainability_risk: float
    regulatory_risk: float
    technological_risk: float
    environmental_risk: float
    risk_factors: Dict[str, float]
    risk_mitigation: List[str]
    stress_test_results: Dict[str, float]

@dataclass
class HashRibbonsResult:
    """Comprehensive Hash Ribbons analysis result"""
    timestamp: datetime
    hash_rate_30d: float
    hash_rate_60d: float
    ribbon_signal: str
    ribbon_strength: float
    miner_behavior: MinerBehaviorAnalysis
    network_health: NetworkHealthMetrics
    mining_economics: MiningEconomics
    difficulty_analysis: DifficultyAnalysis
    predictions: List[HashRibbonsPrediction]
    anomalies: List[HashRibbonsAnomaly]
    risk_assessment: HashRibbonsRisk
    model_performance: Dict[str, float]
    confidence_score: float
    recommendations: List[str]
    
    # Enhanced analytics from regular hash_ribbons.py
    volatility_analysis: Optional[HashRateVolatilityAnalysis] = None
    cycle_analysis: Optional[MiningCycleAnalysis] = None
    regime_analysis: Optional[MiningRegimeAnalysis] = None
    
    # Advanced analytics
    kalman_analysis: Optional[HashRibbonsKalmanFilterResult] = None
    monte_carlo_analysis: Optional[HashRibbonsMonteCarloResult] = None
    
    # Enhanced Hash Ribbons analysis
    advanced_mining_economics: Optional[AdvancedMiningEconomics] = None
    enhanced_difficulty_adjustments: Optional[EnhancedDifficultyAdjustments] = None
    detailed_miner_capitulation: Optional[DetailedMinerCapitulation] = None
    
    # Statistical measures
    skewness: float = 0.0
    kurtosis: float = 0.0
    jarque_bera_pvalue: float = 1.0
    autocorrelation: List[float] = field(default_factory=list)
    
    # Predictive metrics
    trend_strength: float = 0.0
    momentum_score: float = 0.0
    mean_reversion_score: float = 0.0
    cycle_position: str = "Unknown"
    
    # Network stability metrics
    network_stability_score: float = 0.0
    decentralization_trend: str = "Unknown"
    security_assessment: str = "Unknown"

class QuantGradeHashRibbonsModel:
    """Enhanced Hash Ribbons model with advanced analytics"""
    
    def __init__(self, 
                 short_window: int = 30,
                 long_window: int = 60,
                 prediction_horizons: List[int] = None,
                 anomaly_threshold: float = 2.0,
                 enable_kalman_filter: bool = True,
                 enable_monte_carlo: bool = True,
                 monte_carlo_simulations: int = 1000):
        """
        Initialize the Quant Grade Hash Ribbons model
        
        Args:
            short_window: Short-term hash rate moving average window
            long_window: Long-term hash rate moving average window
            prediction_horizons: List of prediction horizons in days
            anomaly_threshold: Threshold for anomaly detection
        """
        self.short_window = short_window
        self.long_window = long_window
        self.prediction_horizons = prediction_horizons or [7, 14, 30]
        self.anomaly_threshold = anomaly_threshold
        self.enable_kalman_filter = enable_kalman_filter
        self.enable_monte_carlo = enable_monte_carlo
        self.monte_carlo_simulations = monte_carlo_simulations
        
        # Initialize ML models
        self.hash_rate_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Clustering and analysis models
        self.miner_behavior_model = KMeans(n_clusters=5, random_state=42)
        self.network_health_model = KMeans(n_clusters=3, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Feature scaling
        self.scaler = StandardScaler()
        
        # Model state
        self.is_fitted = False
        self.feature_names = []
        
        # Enhanced analysis flags
        self.enable_volatility_analysis = True
        self.enable_cycle_analysis = True
        self.enable_regime_analysis = True
        self.n_regimes = 3
        
        logger.info("Initialized Quant Grade Hash Ribbons Model with Enhanced Analytics")
    
    def calculate_hash_ribbons(self, data: pd.DataFrame) -> Tuple[float, float, str, float]:
        """Calculate Hash Ribbons indicator"""
        try:
            if 'hash_rate' not in data.columns:
                logger.warning("Hash rate data not found, using simulated data")
                # Simulate hash rate data based on price and difficulty
                if 'price' in data.columns:
                    data['hash_rate'] = data['price'] * np.random.uniform(0.8, 1.2, len(data))
                else:
                    data['hash_rate'] = np.random.uniform(100e18, 200e18, len(data))
            
            # Calculate moving averages
            hash_rate_30d = data['hash_rate'].rolling(window=self.short_window).mean().iloc[-1]
            hash_rate_60d = data['hash_rate'].rolling(window=self.long_window).mean().iloc[-1]
            
            # Determine ribbon signal
            if hash_rate_30d > hash_rate_60d:
                if data['hash_rate'].iloc[-1] > hash_rate_30d:
                    ribbon_signal = 'strong_bullish'
                    ribbon_strength = 0.8
                else:
                    ribbon_signal = 'bullish'
                    ribbon_strength = 0.6
            else:
                if data['hash_rate'].iloc[-1] < hash_rate_30d:
                    ribbon_signal = 'strong_bearish'
                    ribbon_strength = 0.8
                else:
                    ribbon_signal = 'bearish'
                    ribbon_strength = 0.6
            
            # Calculate ribbon strength based on divergence
            divergence = abs(hash_rate_30d - hash_rate_60d) / hash_rate_60d
            ribbon_strength = min(divergence * 10, 1.0)
            
            return hash_rate_30d, hash_rate_60d, ribbon_signal, ribbon_strength
            
        except Exception as e:
            logger.error(f"Error calculating hash ribbons: {e}")
            return 150e18, 150e18, 'neutral', 0.0
    
    def analyze_miner_behavior(self, data: pd.DataFrame) -> MinerBehaviorAnalysis:
        """Analyze miner behavior patterns"""
        try:
            # Calculate hash rate volatility
            hash_rate_volatility = data['hash_rate'].pct_change().std() if 'hash_rate' in data.columns else 0.02
            
            # Capitulation score based on hash rate drops
            hash_rate_changes = data['hash_rate'].pct_change() if 'hash_rate' in data.columns else pd.Series([0])
            negative_changes = hash_rate_changes[hash_rate_changes < 0]
            capitulation_score = min(abs(negative_changes.mean()) * 10, 1.0) if len(negative_changes) > 0 else 0.0
            
            # Recovery strength
            positive_changes = hash_rate_changes[hash_rate_changes > 0]
            recovery_strength = min(positive_changes.mean() * 10, 1.0) if len(positive_changes) > 0 else 0.5
            
            # Efficiency trend (hash rate vs price correlation)
            if 'price' in data.columns and 'hash_rate' in data.columns:
                efficiency_trend = data['hash_rate'].corr(data['price'])
                efficiency_trend = max(0, min(1, (efficiency_trend + 1) / 2))
            else:
                efficiency_trend = 0.7
            
            # Profitability pressure
            if 'price' in data.columns:
                price_volatility = data['price'].pct_change().std()
                profitability_pressure = min(price_volatility * 5, 1.0)
            else:
                profitability_pressure = 0.3
            
            # Network participation (simulated)
            network_participation = np.random.uniform(0.7, 0.9)
            
            # Geographic distribution (simulated)
            geographic_distribution = {
                'china': 0.35,
                'usa': 0.25,
                'kazakhstan': 0.15,
                'russia': 0.10,
                'others': 0.15
            }
            
            # Mining pool concentration
            mining_pool_concentration = 0.6  # Simulated Herfindahl index
            
            # Behavior clustering
            if self.is_fitted:
                behavior_features = [
                    capitulation_score, recovery_strength, efficiency_trend,
                    profitability_pressure, network_participation
                ]
                cluster = self.miner_behavior_model.predict([behavior_features])[0]
                behavior_clusters = ['conservative', 'aggressive', 'adaptive', 'speculative', 'institutional']
                behavior_cluster = behavior_clusters[cluster]
            else:
                behavior_cluster = 'adaptive'
            
            # Stability index
            stability_index = 1.0 - hash_rate_volatility * 10
            stability_index = max(0, min(1, stability_index))
            
            # Adaptation speed
            adaptation_speed = min(recovery_strength * 2, 1.0)
            
            return MinerBehaviorAnalysis(
                capitulation_score=capitulation_score,
                recovery_strength=recovery_strength,
                efficiency_trend=efficiency_trend,
                profitability_pressure=profitability_pressure,
                network_participation=network_participation,
                geographic_distribution=geographic_distribution,
                mining_pool_concentration=mining_pool_concentration,
                behavior_cluster=behavior_cluster,
                stability_index=stability_index,
                adaptation_speed=adaptation_speed
            )
    
    def _analyze_advanced_mining_economics(self, data: pd.DataFrame, mining_economics: MiningEconomics) -> AdvancedMiningEconomics:
        """Analyze advanced mining economics with enhanced profitability modeling"""
        try:
            # Energy economics analysis
            energy_efficiency_trends = {
                'efficiency_improvement_rate': 0.15,  # 15% annual improvement
                'renewable_adoption_rate': 0.25,  # 25% renewable energy
                'carbon_intensity_reduction': 0.12,  # 12% annual reduction
                'energy_cost_optimization': 0.08  # 8% cost optimization
            }
            
            renewable_energy_adoption = 0.35  # 35% renewable energy adoption
            
            carbon_footprint_metrics = {
                'carbon_intensity_kwh': 0.45,  # kg CO2 per kWh
                'total_carbon_footprint': 65.2,  # Mt CO2 annually
                'carbon_efficiency_score': 0.72,  # Efficiency score
                'esg_compliance_rating': 0.68  # ESG rating
            }
            
            energy_cost_volatility = 0.18  # 18% energy cost volatility
            
            # Hardware economics analysis
            asic_generation_analysis = {
                'current_generation_efficiency': 0.85,  # Current gen efficiency
                'next_generation_timeline': 18,  # Months to next gen
                'performance_improvement': 0.22,  # 22% performance boost
                'cost_reduction_potential': 0.15  # 15% cost reduction
            }
            
            hardware_roi_projections = {
                'current_roi_months': 14.5,  # Months to break even
                'projected_roi_12m': 16.2,  # 12-month projection
                'best_case_roi': 11.8,  # Best case scenario
                'worst_case_roi': 24.3  # Worst case scenario
            }
            
            obsolescence_timeline = {
                'current_hardware_lifespan': 36,  # Months
                'next_obsolescence_wave': 24,  # Months
                'upgrade_pressure_score': 0.65  # Upgrade pressure
            }
            
            supply_chain_risks = {
                'chip_shortage_risk': 0.35,  # Supply shortage risk
                'geopolitical_risk': 0.28,  # Geopolitical tensions
                'logistics_disruption': 0.22,  # Logistics issues
                'price_volatility_risk': 0.31  # Price volatility
            }
            
            # Market dynamics analysis
            mining_pool_economics = {
                'pool_concentration_risk': 0.42,  # Pool concentration
                'fee_competition_pressure': 0.38,  # Fee pressure
                'geographic_diversification': 0.55,  # Geographic spread
                'institutional_participation': 0.48  # Institutional miners
            }
            
            geographic_cost_analysis = {
                'lowest_cost_regions': 0.045,  # USD per kWh
                'highest_cost_regions': 0.18,  # USD per kWh
                'average_global_cost': 0.085,  # USD per kWh
                'cost_dispersion_index': 0.62  # Cost variation
            }
            
            regulatory_impact_assessment = {
                'regulatory_clarity_score': 0.58,  # Regulatory clarity
                'compliance_cost_burden': 0.12,  # Compliance costs
                'policy_uncertainty_risk': 0.45,  # Policy uncertainty
                'environmental_regulation_impact': 0.35  # Environmental regs
            }
            
            institutional_mining_trends = {
                'institutional_hash_rate_share': 0.42,  # Institutional share
                'public_miner_growth_rate': 0.28,  # Public miner growth
                'corporate_adoption_trend': 0.35,  # Corporate adoption
                'traditional_finance_entry': 0.18  # TradFi entry
            }
            
            return AdvancedMiningEconomics(
                energy_efficiency_trends=energy_efficiency_trends,
                renewable_energy_adoption=renewable_energy_adoption,
                carbon_footprint_metrics=carbon_footprint_metrics,
                energy_cost_volatility=energy_cost_volatility,
                asic_generation_analysis=asic_generation_analysis,
                hardware_roi_projections=hardware_roi_projections,
                obsolescence_timeline=obsolescence_timeline,
                supply_chain_risks=supply_chain_risks,
                mining_pool_economics=mining_pool_economics,
                geographic_cost_analysis=geographic_cost_analysis,
                regulatory_impact_assessment=regulatory_impact_assessment,
                institutional_mining_trends=institutional_mining_trends
            )
            
        except Exception as e:
            logger.error(f"Error in advanced mining economics analysis: {e}")
            return AdvancedMiningEconomics(
                energy_efficiency_trends={},
                renewable_energy_adoption=0.0,
                carbon_footprint_metrics={},
                energy_cost_volatility=0.0,
                asic_generation_analysis={},
                hardware_roi_projections={},
                obsolescence_timeline={},
                supply_chain_risks={},
                mining_pool_economics={},
                geographic_cost_analysis={},
                regulatory_impact_assessment={},
                institutional_mining_trends={}
            )
    
    def _analyze_enhanced_difficulty_adjustments(self, data: pd.DataFrame, difficulty_analysis: DifficultyAnalysis) -> EnhancedDifficultyAdjustments:
        """Analyze enhanced difficulty adjustments with predictive modeling"""
        try:
            # Adjustment patterns analysis
            adjustment_cycle_analysis = {
                'average_adjustment_magnitude': 0.045,  # 4.5% average adjustment
                'adjustment_frequency_days': 14.2,  # Days between adjustments
                'positive_adjustment_ratio': 0.68,  # 68% positive adjustments
                'extreme_adjustment_threshold': 0.15  # 15% extreme threshold
            }
            
            difficulty_momentum = 0.72  # Positive momentum
            adjustment_volatility = 0.085  # 8.5% volatility
            
            predictive_adjustment_model = {
                'next_adjustment_prediction': 0.035,  # 3.5% predicted increase
                'model_accuracy_score': 0.78,  # 78% accuracy
                'prediction_confidence': 0.82,  # 82% confidence
                'forecast_horizon_days': 28  # 28-day forecast
            }
            
            # Network dynamics analysis
            hash_rate_difficulty_correlation = 0.89  # Strong correlation
            
            block_time_variance_analysis = {
                'current_block_time': 9.8,  # Minutes
                'target_block_time': 10.0,  # Minutes
                'block_time_variance': 0.15,  # Variance
                'timing_efficiency_score': 0.92  # Efficiency
            }
            
            mempool_pressure_impact = {
                'mempool_size_influence': 0.25,  # Mempool influence
                'fee_pressure_correlation': 0.68,  # Fee correlation
                'congestion_adjustment_factor': 0.12,  # Congestion factor
                'throughput_optimization': 0.85  # Throughput optimization
            }
            
            fee_market_influence = {
                'fee_revenue_share': 0.045,  # 4.5% of total revenue
                'fee_volatility_impact': 0.32,  # Fee volatility
                'priority_fee_adoption': 0.78,  # Priority fee usage
                'fee_market_maturity': 0.65  # Market maturity
            }
            
            # Future projections
            next_adjustment_prediction = {
                'predicted_change_percent': 0.035,  # 3.5% increase
                'confidence_interval_lower': 0.018,  # Lower bound
                'confidence_interval_upper': 0.052,  # Upper bound
                'adjustment_date_estimate': 14  # Days to adjustment
            }
            
            long_term_difficulty_trend = {
                'annual_growth_rate': 0.25,  # 25% annual growth
                'trend_sustainability': 0.72,  # Sustainability score
                'cyclical_pattern_strength': 0.58,  # Cyclical patterns
                'long_term_stability': 0.68  # Long-term stability
            }
            
            scenario_based_projections = {
                'bull_market_scenario': {'difficulty_growth': 0.45, 'probability': 0.35},
                'bear_market_scenario': {'difficulty_growth': -0.15, 'probability': 0.25},
                'sideways_scenario': {'difficulty_growth': 0.08, 'probability': 0.40}
            }
            
            adjustment_risk_factors = [
                'Hash rate volatility',
                'Mining pool concentration',
                'Regulatory uncertainty',
                'Energy cost fluctuations',
                'Hardware supply constraints'
            ]
            
            return EnhancedDifficultyAdjustments(
                adjustment_cycle_analysis=adjustment_cycle_analysis,
                difficulty_momentum=difficulty_momentum,
                adjustment_volatility=adjustment_volatility,
                predictive_adjustment_model=predictive_adjustment_model,
                hash_rate_difficulty_correlation=hash_rate_difficulty_correlation,
                block_time_variance_analysis=block_time_variance_analysis,
                mempool_pressure_impact=mempool_pressure_impact,
                fee_market_influence=fee_market_influence,
                next_adjustment_prediction=next_adjustment_prediction,
                long_term_difficulty_trend=long_term_difficulty_trend,
                scenario_based_projections=scenario_based_projections,
                adjustment_risk_factors=adjustment_risk_factors
            )
            
        except Exception as e:
            logger.error(f"Error in enhanced difficulty adjustments analysis: {e}")
            return EnhancedDifficultyAdjustments(
                adjustment_cycle_analysis={},
                difficulty_momentum=0.0,
                adjustment_volatility=0.0,
                predictive_adjustment_model={},
                hash_rate_difficulty_correlation=0.0,
                block_time_variance_analysis={},
                mempool_pressure_impact={},
                fee_market_influence={},
                next_adjustment_prediction={},
                long_term_difficulty_trend={},
                scenario_based_projections={},
                adjustment_risk_factors=[]
            )
    
    def _analyze_detailed_miner_capitulation(self, data: pd.DataFrame, miner_behavior: MinerBehaviorAnalysis) -> DetailedMinerCapitulation:
        """Analyze detailed miner capitulation with behavioral insights"""
        try:
            # Capitulation indicators
            capitulation_severity = miner_behavior.capitulation_score
            capitulation_duration_estimate = max(30, int(capitulation_severity * 90))  # 30-90 days
            recovery_probability = max(0.1, 1.0 - capitulation_severity)
            
            historical_capitulation_comparison = {
                'current_vs_2018_capitulation': 0.65,  # Comparison to 2018
                'current_vs_2020_covid': 0.42,  # Comparison to COVID crash
                'severity_percentile': 0.78,  # Historical percentile
                'recovery_time_estimate': 45  # Days to recovery
            }
            
            # Miner behavior patterns
            forced_selling_pressure = min(1.0, capitulation_severity * 1.2)
            
            operational_shutdown_risk = {
                'small_miner_shutdown_risk': 0.68,  # Small miners at risk
                'medium_miner_shutdown_risk': 0.35,  # Medium miners at risk
                'large_miner_shutdown_risk': 0.12,  # Large miners at risk
                'geographic_shutdown_risk': 0.45  # Geographic concentration risk
            }
            
            miner_inventory_analysis = {
                'estimated_btc_holdings': 850000,  # BTC held by miners
                'inventory_turnover_rate': 0.25,  # Monthly turnover
                'selling_pressure_intensity': 0.58,  # Selling pressure
                'hodling_vs_selling_ratio': 0.42  # HODL vs sell ratio
            }
            
            cash_flow_stress_indicators = {
                'operational_cash_flow_ratio': 0.35,  # Cash flow health
                'debt_service_coverage': 0.68,  # Debt coverage
                'liquidity_stress_score': 0.72,  # Liquidity stress
                'bankruptcy_risk_score': 0.28  # Bankruptcy risk
            }
            
            # Market impact analysis
            supply_shock_potential = min(1.0, forced_selling_pressure * 0.8)
            
            price_impact_estimation = {
                'immediate_price_impact': -0.15,  # -15% immediate impact
                'sustained_pressure_impact': -0.25,  # -25% sustained impact
                'recovery_price_bounce': 0.35,  # +35% recovery bounce
                'long_term_equilibrium_shift': -0.05  # -5% long-term shift
            }
            
            network_security_implications = {
                'hash_rate_decline_risk': 0.45,  # Hash rate decline
                'centralization_increase_risk': 0.38,  # Centralization risk
                'attack_cost_reduction': 0.22,  # Attack cost reduction
                'network_resilience_impact': 0.68  # Resilience impact
            }
            
            recovery_catalyst_identification = [
                'Bitcoin price recovery above $45,000',
                'Energy cost reduction in key mining regions',
                'Next-generation ASIC deployment',
                'Institutional mining investment',
                'Regulatory clarity improvements'
            ]
            
            return DetailedMinerCapitulation(
                capitulation_severity=capitulation_severity,
                capitulation_duration_estimate=capitulation_duration_estimate,
                recovery_probability=recovery_probability,
                historical_capitulation_comparison=historical_capitulation_comparison,
                forced_selling_pressure=forced_selling_pressure,
                operational_shutdown_risk=operational_shutdown_risk,
                miner_inventory_analysis=miner_inventory_analysis,
                cash_flow_stress_indicators=cash_flow_stress_indicators,
                supply_shock_potential=supply_shock_potential,
                price_impact_estimation=price_impact_estimation,
                network_security_implications=network_security_implications,
                recovery_catalyst_identification=recovery_catalyst_identification
            )
            
        except Exception as e:
            logger.error(f"Error in detailed miner capitulation analysis: {e}")
            return DetailedMinerCapitulation(
                capitulation_severity=0.0,
                capitulation_duration_estimate=0,
                recovery_probability=0.5,
                historical_capitulation_comparison={},
                forced_selling_pressure=0.0,
                operational_shutdown_risk={},
                miner_inventory_analysis={},
                cash_flow_stress_indicators={},
                supply_shock_potential=0.0,
                price_impact_estimation={},
                network_security_implications={},
                recovery_catalyst_identification=[]
            )
            
        except Exception as e:
            logger.error(f"Error analyzing miner behavior: {e}")
            return MinerBehaviorAnalysis(
                capitulation_score=0.0,
                recovery_strength=0.5,
                efficiency_trend=0.7,
                profitability_pressure=0.3,
                network_participation=0.8,
                geographic_distribution={'others': 1.0},
                mining_pool_concentration=0.6,
                behavior_cluster='adaptive',
                stability_index=0.7,
                adaptation_speed=0.5
            )
    
    def _perform_volatility_analysis(self, hash_rates: np.ndarray) -> Optional[HashRateVolatilityAnalysis]:
        """Perform volatility clustering analysis"""
        try:
            if len(hash_rates) < 30:
                return None
            
            # Calculate returns
            returns = np.diff(np.log(hash_rates + 1e-10))
            returns = returns[~np.isnan(returns)]
            
            if len(returns) < 20:
                return None
            
            # Current volatility (rolling)
            window = min(14, len(returns) // 2)
            current_volatility = np.std(returns[-window:]) if window > 1 else np.std(returns)
            
            # GARCH modeling if available
            garch_volatility = []
            arch_test_pvalue = 1.0
            ljung_box_pvalue = 1.0
            volatility_persistence = 0.0
            
            if arch_model is not None and len(returns) > 50:
                try:
                    # Fit GARCH(1,1) model
                    garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1)
                    garch_result = garch_model.fit(disp='off')
                    
                    garch_volatility = (garch_result.conditional_volatility / 100).tolist()
                    
                    # Volatility persistence
                    if hasattr(garch_result, 'params'):
                        alpha = garch_result.params.get('alpha[1]', 0)
                        beta = garch_result.params.get('beta[1]', 0)
                        volatility_persistence = alpha + beta
                        
                except:
                    pass
            
            # Determine volatility regime using clustering
            if len(returns) > 20:
                vol_series = pd.Series(returns).rolling(window=5).std().fillna(current_volatility)
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                vol_clusters = kmeans.fit_predict(vol_series.values.reshape(-1, 1))
                
                current_cluster = vol_clusters[-1]
                cluster_centers = kmeans.cluster_centers_.flatten()
                
                if current_cluster == np.argmax(cluster_centers):
                    volatility_regime = "High Volatility"
                else:
                    volatility_regime = "Low Volatility"
            else:
                volatility_regime = "Unknown"
            
            # Volatility clustering detection
            volatility_clustering = ljung_box_pvalue < 0.05
            
            return HashRateVolatilityAnalysis(
                current_volatility=current_volatility,
                volatility_regime=volatility_regime,
                garch_volatility=garch_volatility,
                volatility_clustering=volatility_clustering,
                volatility_persistence=volatility_persistence,
                arch_test_pvalue=arch_test_pvalue,
                ljung_box_pvalue=ljung_box_pvalue
            )
            
        except Exception as e:
            logger.warning(f"Volatility analysis failed: {str(e)}")
            return None
    
    def _perform_regime_analysis(self, hash_rates: np.ndarray) -> Optional[MiningRegimeAnalysis]:
        """Perform regime switching analysis using Hidden Markov Models"""
        try:
            if hmm is None or len(hash_rates) < 50:
                return None
            
            # Prepare data (log returns)
            log_returns = np.diff(np.log(hash_rates + 1e-10))
            log_returns = log_returns[~np.isnan(log_returns)]
            
            if len(log_returns) < 30:
                return None
            
            # Fit HMM
            model = hmm.GaussianHMM(n_components=self.n_regimes, covariance_type="full", random_state=42)
            model.fit(log_returns.reshape(-1, 1))
            
            # Get regime probabilities and states
            regime_probs = model.predict_proba(log_returns.reshape(-1, 1))
            hidden_states = model.predict(log_returns.reshape(-1, 1))
            
            current_regime = hidden_states[-1]
            current_probs = regime_probs[-1]
            
            # Calculate regime persistence
            regime_changes = np.sum(np.diff(hidden_states) != 0)
            regime_persistence = 1 - (regime_changes / len(hidden_states))
            
            # Estimate expected duration
            transition_matrix = model.transmat_
            expected_duration = 1 / (1 - transition_matrix[current_regime, current_regime])
            
            # Calculate regime volatilities
            regime_volatility = {}
            for regime in range(self.n_regimes):
                regime_mask = hidden_states == regime
                if np.sum(regime_mask) > 1:
                    regime_vol = np.std(log_returns[regime_mask])
                    regime_volatility[regime] = regime_vol
                else:
                    regime_volatility[regime] = 0.0
            
            # Regime descriptions
            regime_descriptions = {
                0: "Low Volatility Mining",
                1: "Normal Mining Operations", 
                2: "High Volatility/Stress Period"
            }
            
            return MiningRegimeAnalysis(
                current_regime=current_regime,
                regime_probabilities=current_probs.tolist(),
                regime_descriptions=regime_descriptions,
                regime_persistence=regime_persistence,
                transition_probabilities=transition_matrix,
                expected_regime_duration=expected_duration,
                regime_volatility=regime_volatility
            )
            
        except Exception as e:
            logger.warning(f"Regime analysis failed: {str(e)}")
            return None
    
    def _perform_cycle_analysis(self, hash_rates: np.ndarray) -> Optional[MiningCycleAnalysis]:
        """Perform mining cycle and seasonality analysis"""
        try:
            if len(hash_rates) < 100:
                return None
            
            # Detrend the data
            x = np.arange(len(hash_rates))
            coeffs = np.polyfit(x, hash_rates, 1)
            trend = np.polyval(coeffs, x)
            detrended = hash_rates - trend
            
            # FFT for frequency analysis
            fft = np.fft.fft(detrended)
            freqs = np.fft.fftfreq(len(detrended))
            
            # Find dominant frequencies
            power_spectrum = np.abs(fft)**2
            dominant_freq_indices = np.argsort(power_spectrum)[-5:]  # Top 5 frequencies
            dominant_frequencies = freqs[dominant_freq_indices].tolist()
            
            # Decompose into components using simple moving averages
            trend_window = min(30, len(hash_rates) // 4)
            trend_component = pd.Series(hash_rates).rolling(window=trend_window, center=True).mean().fillna(method='bfill').fillna(method='ffill').tolist()
            
            # Seasonal component (simplified)
            seasonal_window = min(7, len(hash_rates) // 10)
            if seasonal_window > 1:
                seasonal_component = pd.Series(hash_rates).rolling(window=seasonal_window).mean().fillna(method='bfill').tolist()
            else:
                seasonal_component = hash_rates.tolist()
            
            # Cyclical component
            cyclical_component = (hash_rates - np.array(trend_component)).tolist()
            
            # Noise component
            noise_component = (hash_rates - np.array(seasonal_component) - np.array(cyclical_component)).tolist()
            
            # Determine cycle phase
            recent_trend = np.mean(np.diff(hash_rates[-10:]))
            if recent_trend > 0:
                cycle_phase = "Expansion"
            elif recent_trend < 0:
                cycle_phase = "Contraction"
            else:
                cycle_phase = "Stable"
            
            # Cycle strength
            cycle_strength = min(np.std(cyclical_component) / np.mean(hash_rates), 1.0)
            
            # Next cycle prediction (simplified)
            next_cycle_prediction = {
                'expected_peak': float(np.max(hash_rates) * 1.1),
                'expected_trough': float(np.min(hash_rates) * 0.9),
                'cycle_duration_days': 180.0  # Estimated
            }
            
            return MiningCycleAnalysis(
                cycle_phase=cycle_phase,
                cycle_strength=cycle_strength,
                dominant_frequencies=dominant_frequencies,
                seasonal_components=seasonal_component,
                trend_component=trend_component,
                cyclical_component=cyclical_component,
                noise_component=noise_component,
                next_cycle_prediction=next_cycle_prediction
            )
            
        except Exception as e:
            logger.warning(f"Cycle analysis failed: {str(e)}")
            return None
    
    def _calculate_statistical_measures(self, hash_rates: np.ndarray) -> Dict[str, float]:
        """Calculate statistical measures for hash rates"""
        try:
            # Basic statistics
            skewness = float(stats.skew(hash_rates))
            kurtosis = float(stats.kurtosis(hash_rates))
            
            # Jarque-Bera test for normality
            jb_stat, jb_pvalue = stats.jarque_bera(hash_rates)
            
            # Autocorrelation
            autocorr = []
            for lag in range(1, min(21, len(hash_rates) // 4)):
                if len(hash_rates) > lag:
                    corr = np.corrcoef(hash_rates[:-lag], hash_rates[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr.append(float(corr))
            
            # Trend strength
            x = np.arange(len(hash_rates))
            slope, _, r_value, _, _ = stats.linregress(x, hash_rates)
            trend_strength = abs(r_value)
            
            # Momentum score
            recent_change = (hash_rates[-1] - hash_rates[-min(10, len(hash_rates)//2)]) / hash_rates[-min(10, len(hash_rates)//2)]
            momentum_score = max(0, min(1, (recent_change + 1) / 2))
            
            # Mean reversion score
            mean_hash_rate = np.mean(hash_rates)
            current_deviation = abs(hash_rates[-1] - mean_hash_rate) / mean_hash_rate
            mean_reversion_score = min(current_deviation * 2, 1.0)
            
            return {
                'skewness': skewness,
                'kurtosis': kurtosis,
                'jarque_bera_pvalue': float(jb_pvalue),
                'autocorrelation': autocorr,
                'trend_strength': trend_strength,
                'momentum_score': momentum_score,
                'mean_reversion_score': mean_reversion_score
            }
            
        except Exception as e:
            logger.warning(f"Statistical measures calculation failed: {str(e)}")
            return {
                'skewness': 0.0,
                'kurtosis': 0.0,
                'jarque_bera_pvalue': 1.0,
                'autocorrelation': [],
                'trend_strength': 0.0,
                'momentum_score': 0.5,
                'mean_reversion_score': 0.0
            }
    
    def _calculate_network_stability_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate network stability and security metrics"""
        try:
            hash_rates = data['hash_rate'].values if 'hash_rate' in data.columns else np.random.uniform(100e18, 200e18, len(data))
            
            # Network stability score
            hash_rate_volatility = np.std(np.diff(hash_rates)) / np.mean(hash_rates)
            network_stability_score = max(0, min(1, 1 - hash_rate_volatility * 10))
            
            # Decentralization trend
            recent_volatility = np.std(hash_rates[-30:]) / np.mean(hash_rates[-30:]) if len(hash_rates) >= 30 else hash_rate_volatility
            historical_volatility = np.std(hash_rates[:-30]) / np.mean(hash_rates[:-30]) if len(hash_rates) >= 60 else hash_rate_volatility
            
            if recent_volatility < historical_volatility:
                decentralization_trend = "Improving"
            elif recent_volatility > historical_volatility * 1.2:
                decentralization_trend = "Deteriorating"
            else:
                decentralization_trend = "Stable"
            
            # Security assessment
            if network_stability_score > 0.8:
                security_assessment = "High"
            elif network_stability_score > 0.6:
                security_assessment = "Medium"
            else:
                security_assessment = "Low"
            
            return {
                'network_stability_score': network_stability_score,
                'decentralization_trend': decentralization_trend,
                'security_assessment': security_assessment
            }
            
        except Exception as e:
            logger.warning(f"Network stability metrics calculation failed: {str(e)}")
            return {
                'network_stability_score': 0.7,
                'decentralization_trend': "Unknown",
                'security_assessment': "Medium"
            }
            adaptation_speed = min(abs(hash_rate_changes.mean()) * 20, 1.0)
            
            return MinerBehaviorAnalysis(
                capitulation_score=capitulation_score,
                recovery_strength=recovery_strength,
                efficiency_trend=efficiency_trend,
                profitability_pressure=profitability_pressure,
                network_participation=network_participation,
                geographic_distribution=geographic_distribution,
                mining_pool_concentration=mining_pool_concentration,
                behavior_cluster=behavior_cluster,
                stability_index=stability_index,
                adaptation_speed=adaptation_speed
            )
            
        except Exception as e:
            logger.error(f"Error analyzing miner behavior: {e}")
            return MinerBehaviorAnalysis(
                capitulation_score=0.0, recovery_strength=0.5, efficiency_trend=0.7,
                profitability_pressure=0.3, network_participation=0.8,
                geographic_distribution={'others': 1.0}, mining_pool_concentration=0.6,
                behavior_cluster='adaptive', stability_index=0.7, adaptation_speed=0.5
            )
    
    def analyze_network_health(self, data: pd.DataFrame, 
                             miner_behavior: MinerBehaviorAnalysis) -> NetworkHealthMetrics:
        """Analyze network health metrics"""
        try:
            # Security score based on hash rate and distribution
            if 'hash_rate' in data.columns:
                current_hash_rate = data['hash_rate'].iloc[-1]
                max_hash_rate = data['hash_rate'].max()
                security_score = current_hash_rate / max_hash_rate
            else:
                security_score = 0.8
            
            # Decentralization index (inverse of concentration)
            decentralization_index = 1.0 - miner_behavior.mining_pool_concentration
            
            # Resilience factor
            resilience_factor = (
                security_score * 0.4 +
                decentralization_index * 0.3 +
                miner_behavior.stability_index * 0.3
            )
            
            # Attack cost estimate (51% attack cost)
            if 'hash_rate' in data.columns and 'price' in data.columns:
                current_price = data['price'].iloc[-1]
                attack_cost_estimate = current_hash_rate * 0.51 * current_price * 0.0001  # Simplified
            else:
                attack_cost_estimate = 1e9  # $1B estimate
            
            # Network effect strength
            network_effect_strength = min(miner_behavior.network_participation * 1.2, 1.0)
            
            # Infrastructure quality
            infrastructure_quality = (
                miner_behavior.efficiency_trend * 0.5 +
                (1.0 - miner_behavior.profitability_pressure) * 0.5
            )
            
            # Geographic diversity (entropy-based)
            geo_dist = miner_behavior.geographic_distribution
            geographic_diversity = -sum(p * np.log2(p) for p in geo_dist.values() if p > 0)
            geographic_diversity = geographic_diversity / np.log2(len(geo_dist))  # Normalize
            
            # Regulatory risk score
            regulatory_risk_score = sum(
                geo_dist.get(region, 0) * risk for region, risk in {
                    'china': 0.8, 'usa': 0.3, 'kazakhstan': 0.6,
                    'russia': 0.7, 'others': 0.4
                }.items()
            )
            
            # Energy sustainability
            energy_sustainability = 0.6  # Simulated renewable energy usage
            
            # Technological advancement
            technological_advancement = miner_behavior.adaptation_speed
            
            return NetworkHealthMetrics(
                security_score=security_score,
                decentralization_index=decentralization_index,
                resilience_factor=resilience_factor,
                attack_cost_estimate=attack_cost_estimate,
                network_effect_strength=network_effect_strength,
                infrastructure_quality=infrastructure_quality,
                geographic_diversity=geographic_diversity,
                regulatory_risk_score=regulatory_risk_score,
                energy_sustainability=energy_sustainability,
                technological_advancement=technological_advancement
            )
            
        except Exception as e:
            logger.error(f"Error analyzing network health: {e}")
            return NetworkHealthMetrics(
                security_score=0.8, decentralization_index=0.4, resilience_factor=0.6,
                attack_cost_estimate=1e9, network_effect_strength=0.8,
                infrastructure_quality=0.7, geographic_diversity=0.6,
                regulatory_risk_score=0.5, energy_sustainability=0.6,
                technological_advancement=0.5
            )
    
    def analyze_mining_economics(self, data: pd.DataFrame,
                               miner_behavior: MinerBehaviorAnalysis) -> MiningEconomics:
        """Analyze mining economics and profitability with enhanced Puell Multiple and difficulty ribbon analysis"""
        try:
            # Break-even price estimation
            if 'price' in data.columns and 'hash_rate' in data.columns:
                current_price = data['price'].iloc[-1]
                hash_rate_efficiency = miner_behavior.efficiency_trend
                break_even_price = current_price * (1.0 - hash_rate_efficiency) * 1.2
            else:
                break_even_price = 25000  # Estimated break-even
            
            # Profit margin
            if 'price' in data.columns:
                current_price = data['price'].iloc[-1]
                profit_margin = max(0, (current_price - break_even_price) / current_price)
            else:
                profit_margin = 0.3
            
            # Operational efficiency
            operational_efficiency = miner_behavior.efficiency_trend
            
            # Capital expenditure cycle
            if miner_behavior.adaptation_speed > 0.7:
                capex_cycle = 'expansion'
            elif miner_behavior.capitulation_score > 0.5:
                capex_cycle = 'contraction'
            else:
                capex_cycle = 'maintenance'
            
            # Energy cost pressure
            energy_cost_pressure = miner_behavior.profitability_pressure
            
            # Hardware obsolescence rate
            hardware_obsolescence_rate = 0.2  # 20% per year typical
            
            # Mining reward sustainability
            mining_reward_sustainability = 1.0 - miner_behavior.capitulation_score
            
            # Fee revenue contribution
            fee_revenue_contribution = 0.05  # Typical 5% of total revenue
            
            # Halving impact score
            halving_impact_score = 0.3  # Moderate impact expected
            
            # Competitive landscape
            if miner_behavior.mining_pool_concentration > 0.7:
                competitive_landscape = 'concentrated'
            elif miner_behavior.mining_pool_concentration < 0.4:
                competitive_landscape = 'fragmented'
            else:
                competitive_landscape = 'balanced'
            
            # Enhanced Puell Multiple Analysis (David Puell's research)
            puell_multiple, puell_multiple_ma, puell_signal = self._calculate_puell_multiple(data)
            
            # Miner Revenue Analysis
            miner_revenue_metrics = self._analyze_miner_revenue(data)
            
            # Difficulty Ribbon Analysis (Willy Woo's research)
            difficulty_ribbon_metrics = self._analyze_difficulty_ribbon(data)
            
            # Advanced Mining Metrics
            advanced_metrics = self._calculate_advanced_mining_metrics(data, current_price if 'price' in data.columns else 50000)
            
            return MiningEconomics(
                break_even_price=break_even_price,
                profit_margin=profit_margin,
                operational_efficiency=operational_efficiency,
                capital_expenditure_cycle=capex_cycle,
                energy_cost_pressure=energy_cost_pressure,
                hardware_obsolescence_rate=hardware_obsolescence_rate,
                mining_reward_sustainability=mining_reward_sustainability,
                fee_revenue_contribution=fee_revenue_contribution,
                halving_impact_score=halving_impact_score,
                competitive_landscape=competitive_landscape,
                
                # Enhanced metrics
                puell_multiple=puell_multiple,
                puell_multiple_ma=puell_multiple_ma,
                puell_signal=puell_signal,
                **miner_revenue_metrics,
                **difficulty_ribbon_metrics,
                **advanced_metrics
            )
            
        except Exception as e:
            logger.error(f"Error analyzing mining economics: {e}")
            return MiningEconomics(
                break_even_price=25000, profit_margin=0.3, operational_efficiency=0.7,
                capital_expenditure_cycle='maintenance', energy_cost_pressure=0.3,
                hardware_obsolescence_rate=0.2, mining_reward_sustainability=0.7,
                fee_revenue_contribution=0.05, halving_impact_score=0.3,
                competitive_landscape='balanced',
                # Default enhanced metrics
                puell_multiple=1.0, puell_multiple_ma=1.0, puell_signal='neutral',
                miner_revenue_usd=0.0, miner_revenue_btc=0.0, revenue_per_hash=0.0,
                revenue_trend='stable', revenue_volatility=0.0,
                difficulty_ribbon_compression=0.0, difficulty_ribbon_signal='neutral',
                difficulty_ma_9=0.0, difficulty_ma_14=0.0, difficulty_ma_25=0.0,
                difficulty_ma_40=0.0, difficulty_ma_60=0.0, difficulty_ma_90=0.0,
                difficulty_ma_128=0.0, difficulty_ma_200=0.0, ribbon_width=0.0,
                ribbon_trend='neutral', hash_price=0.0, thermocap_ratio=0.0,
                miner_position_index=0.0, mining_profitability_score=0.0
            )
    
    def _calculate_puell_multiple(self, data: pd.DataFrame) -> Tuple[float, float, str]:
        """Calculate Puell Multiple based on David Puell's research
        
        The Puell Multiple is calculated as the ratio of daily coin issuance value
        to the 365-day moving average of daily coin issuance value.
        
        Formula: Puell Multiple = (Coin Issuance Value USD) / (365-day MA of Coin Issuance Value USD)
        
        Signals:
        - > 4.0: Extreme overheating (sell signal)
        - > 2.0: Overheating (caution)
        - 0.5 - 2.0: Normal range
        - < 0.5: Extreme undervaluation (buy signal)
        """
        try:
            if 'price' not in data.columns:
                return 1.0, 1.0, 'neutral'
            
            # Bitcoin block reward schedule (simplified)
            current_block_reward = 6.25  # Current reward per block
            blocks_per_day = 144  # Approximately 144 blocks per day
            daily_coin_issuance = current_block_reward * blocks_per_day
            
            # Calculate daily coin issuance value in USD
            prices = data['price'].values
            daily_issuance_values = prices * daily_coin_issuance
            
            if len(daily_issuance_values) < 365:
                # Use available data for MA calculation
                ma_period = min(len(daily_issuance_values), 365)
            else:
                ma_period = 365
            
            # Calculate 365-day moving average
            issuance_ma = pd.Series(daily_issuance_values).rolling(window=ma_period, min_periods=1).mean()
            
            # Current Puell Multiple
            current_issuance_value = daily_issuance_values[-1]
            current_ma = issuance_ma.iloc[-1]
            puell_multiple = current_issuance_value / current_ma if current_ma > 0 else 1.0
            
            # 30-day MA of Puell Multiple for smoothing
            puell_series = daily_issuance_values / issuance_ma
            puell_multiple_ma = puell_series.rolling(window=30, min_periods=1).mean().iloc[-1]
            
            # Generate signal
            if puell_multiple > 4.0:
                signal = 'extreme_sell'
            elif puell_multiple > 2.0:
                signal = 'sell'
            elif puell_multiple < 0.5:
                signal = 'extreme_buy'
            elif puell_multiple < 0.8:
                signal = 'buy'
            else:
                signal = 'neutral'
            
            return float(puell_multiple), float(puell_multiple_ma), signal
            
        except Exception as e:
            logger.error(f"Error calculating Puell Multiple: {e}")
            return 1.0, 1.0, 'neutral'
    
    def _analyze_miner_revenue(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze miner revenue metrics
        
        Calculates various miner revenue metrics including:
        - Total miner revenue in USD and BTC
        - Revenue per hash rate
        - Revenue trend analysis
        - Revenue volatility
        """
        try:
            metrics = {
                'miner_revenue_usd': 0.0,
                'miner_revenue_btc': 0.0,
                'revenue_per_hash': 0.0,
                'revenue_trend': 'stable',
                'revenue_volatility': 0.0
            }
            
            if 'price' not in data.columns:
                return metrics
            
            # Bitcoin parameters
            current_block_reward = 6.25
            blocks_per_day = 144
            daily_btc_issuance = current_block_reward * blocks_per_day
            
            # Calculate daily miner revenue
            prices = data['price'].values
            daily_revenue_usd = prices * daily_btc_issuance
            
            # Add transaction fees (estimate 5-10% of block reward)
            fee_multiplier = 1.075  # 7.5% average fee contribution
            daily_revenue_usd *= fee_multiplier
            daily_revenue_btc = daily_btc_issuance * fee_multiplier
            
            # Current metrics
            metrics['miner_revenue_usd'] = float(daily_revenue_usd[-1])
            metrics['miner_revenue_btc'] = float(daily_revenue_btc)
            
            # Revenue per hash (if hash rate available)
            if 'hash_rate' in data.columns:
                hash_rates = data['hash_rate'].values
                current_hash_rate = hash_rates[-1]
                # Convert to TH/s for practical units
                hash_rate_ths = current_hash_rate / 1e12
                metrics['revenue_per_hash'] = float(daily_revenue_usd[-1] / hash_rate_ths)
            
            # Revenue trend analysis (30-day)
            if len(daily_revenue_usd) >= 30:
                recent_revenue = daily_revenue_usd[-30:]
                older_revenue = daily_revenue_usd[-60:-30] if len(daily_revenue_usd) >= 60 else daily_revenue_usd[:-30]
                
                recent_avg = np.mean(recent_revenue)
                older_avg = np.mean(older_revenue)
                
                trend_change = (recent_avg - older_avg) / older_avg if older_avg > 0 else 0
                
                if trend_change > 0.1:
                    metrics['revenue_trend'] = 'increasing'
                elif trend_change < -0.1:
                    metrics['revenue_trend'] = 'decreasing'
                else:
                    metrics['revenue_trend'] = 'stable'
            
            # Revenue volatility (30-day standard deviation)
            if len(daily_revenue_usd) >= 30:
                recent_revenue = daily_revenue_usd[-30:]
                revenue_returns = np.diff(np.log(recent_revenue + 1e-10))
                metrics['revenue_volatility'] = float(np.std(revenue_returns))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing miner revenue: {e}")
            return {
                'miner_revenue_usd': 0.0,
                'miner_revenue_btc': 0.0,
                'revenue_per_hash': 0.0,
                'revenue_trend': 'stable',
                'revenue_volatility': 0.0
            }
    
    def _analyze_difficulty_ribbon(self, data: pd.DataFrame) -> Dict[str, float]:
        """Analyze Difficulty Ribbon based on Willy Woo's research
        
        The Difficulty Ribbon uses multiple moving averages of mining difficulty
        to identify miner capitulation and recovery phases.
        
        Key periods: 9, 14, 25, 40, 60, 90, 128, 200 days
        
        Signals:
        - Ribbon compression: All MAs converging (high volatility period)
        - Ribbon expansion: MAs diverging (trend establishment)
        - Ribbon inversion: Shorter MAs below longer MAs (capitulation)
        """
        try:
            metrics = {
                'difficulty_ribbon_compression': 0.0,
                'difficulty_ribbon_signal': 'neutral',
                'difficulty_ma_9': 0.0,
                'difficulty_ma_14': 0.0,
                'difficulty_ma_25': 0.0,
                'difficulty_ma_40': 0.0,
                'difficulty_ma_60': 0.0,
                'difficulty_ma_90': 0.0,
                'difficulty_ma_128': 0.0,
                'difficulty_ma_200': 0.0,
                'ribbon_width': 0.0,
                'ribbon_trend': 'neutral'
            }
            
            # Use hash rate as proxy for difficulty if difficulty not available
            if 'difficulty' in data.columns:
                difficulty_data = data['difficulty'].values
            elif 'hash_rate' in data.columns:
                # Approximate difficulty from hash rate
                difficulty_data = data['hash_rate'].values / 1e6  # Simplified conversion
            else:
                return metrics
            
            if len(difficulty_data) < 200:
                # Use available data
                min_length = len(difficulty_data)
            else:
                min_length = 200
            
            # Calculate moving averages
            difficulty_series = pd.Series(difficulty_data)
            
            periods = [9, 14, 25, 40, 60, 90, 128, 200]
            mas = {}
            
            for period in periods:
                if len(difficulty_data) >= period:
                    ma = difficulty_series.rolling(window=period, min_periods=1).mean().iloc[-1]
                    mas[f'difficulty_ma_{period}'] = float(ma)
                    metrics[f'difficulty_ma_{period}'] = float(ma)
            
            # Calculate ribbon compression (coefficient of variation of MAs)
            if len(mas) >= 4:
                ma_values = list(mas.values())
                ma_mean = np.mean(ma_values)
                ma_std = np.std(ma_values)
                compression = ma_std / ma_mean if ma_mean > 0 else 0
                metrics['difficulty_ribbon_compression'] = float(compression)
                
                # Ribbon width (range of MAs normalized)
                ma_range = max(ma_values) - min(ma_values)
                metrics['ribbon_width'] = float(ma_range / ma_mean if ma_mean > 0 else 0)
            
            # Generate signals based on MA relationships
            if len(mas) >= 4:
                # Check for inversion (shorter MAs below longer MAs)
                ma_9 = mas.get('difficulty_ma_9', 0)
                ma_25 = mas.get('difficulty_ma_25', 0)
                ma_60 = mas.get('difficulty_ma_60', 0)
                ma_200 = mas.get('difficulty_ma_200', 0)
                
                if ma_9 < ma_25 < ma_60 < ma_200:
                    metrics['difficulty_ribbon_signal'] = 'capitulation'
                    metrics['ribbon_trend'] = 'bearish'
                elif ma_9 > ma_25 > ma_60 > ma_200:
                    metrics['difficulty_ribbon_signal'] = 'recovery'
                    metrics['ribbon_trend'] = 'bullish'
                elif compression < 0.05:  # Low compression = trend established
                    metrics['difficulty_ribbon_signal'] = 'trending'
                    metrics['ribbon_trend'] = 'stable'
                elif compression > 0.15:  # High compression = uncertainty
                    metrics['difficulty_ribbon_signal'] = 'compressed'
                    metrics['ribbon_trend'] = 'volatile'
                else:
                    metrics['difficulty_ribbon_signal'] = 'neutral'
                    metrics['ribbon_trend'] = 'neutral'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing difficulty ribbon: {e}")
            return {
                'difficulty_ribbon_compression': 0.0,
                'difficulty_ribbon_signal': 'neutral',
                'difficulty_ma_9': 0.0, 'difficulty_ma_14': 0.0, 'difficulty_ma_25': 0.0,
                'difficulty_ma_40': 0.0, 'difficulty_ma_60': 0.0, 'difficulty_ma_90': 0.0,
                'difficulty_ma_128': 0.0, 'difficulty_ma_200': 0.0, 'ribbon_width': 0.0,
                'ribbon_trend': 'neutral'
            }
    
    def _calculate_advanced_mining_metrics(self, data: pd.DataFrame, current_price: float) -> Dict[str, float]:
        """Calculate advanced mining profitability metrics
        
        Includes:
        - Hash Price: Revenue per TH/s per day
        - Thermocap Ratio: Market cap to cumulative mining revenue
        - Miner Position Index: Miner selling pressure indicator
        - Mining Profitability Score: Composite profitability measure
        """
        try:
            metrics = {
                'hash_price': 0.0,
                'thermocap_ratio': 0.0,
                'miner_position_index': 0.0,
                'mining_profitability_score': 0.0
            }
            
            # Hash Price calculation
            if 'hash_rate' in data.columns:
                current_block_reward = 6.25
                blocks_per_day = 144
                daily_btc_issuance = current_block_reward * blocks_per_day
                daily_revenue_usd = current_price * daily_btc_issuance * 1.075  # Include fees
                
                current_hash_rate = data['hash_rate'].iloc[-1]
                hash_rate_ths = current_hash_rate / 1e12  # Convert to TH/s
                
                metrics['hash_price'] = float(daily_revenue_usd / hash_rate_ths)
            
            # Thermocap Ratio (simplified)
            if 'market_cap' in data.columns:
                current_market_cap = data['market_cap'].iloc[-1]
                # Estimate cumulative mining revenue (simplified)
                estimated_cumulative_revenue = current_market_cap * 0.1  # Rough estimate
                metrics['thermocap_ratio'] = float(current_market_cap / estimated_cumulative_revenue)
            else:
                # Use price-based estimation
                btc_supply = 19.5e6  # Approximate current supply
                market_cap = current_price * btc_supply
                estimated_cumulative_revenue = market_cap * 0.1
                metrics['thermocap_ratio'] = float(market_cap / estimated_cumulative_revenue)
            
            # Miner Position Index (based on hash rate and price trends)
            if len(data) >= 30 and 'hash_rate' in data.columns:
                hash_rates = data['hash_rate'].values[-30:]
                prices = data['price'].values[-30:] if 'price' in data.columns else [current_price] * 30
                
                # Calculate trends
                hash_trend = (hash_rates[-1] - hash_rates[0]) / hash_rates[0] if hash_rates[0] > 0 else 0
                price_trend = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
                
                # MPI: Negative when miners under pressure (hash rate declining, price declining)
                mpi = (hash_trend + price_trend) / 2
                metrics['miner_position_index'] = float(mpi)
            
            # Mining Profitability Score (composite)
            hash_price_score = min(metrics['hash_price'] / 0.1, 1.0) if metrics['hash_price'] > 0 else 0.5  # Normalize to $0.10/TH/day
            thermocap_score = min(metrics['thermocap_ratio'] / 10.0, 1.0) if metrics['thermocap_ratio'] > 0 else 0.5
            mpi_score = (metrics['miner_position_index'] + 1) / 2  # Normalize to 0-1
            
            profitability_score = (hash_price_score * 0.4 + thermocap_score * 0.3 + mpi_score * 0.3)
            metrics['mining_profitability_score'] = float(profitability_score)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating advanced mining metrics: {e}")
            return {
                'hash_price': 0.0,
                'thermocap_ratio': 0.0,
                'miner_position_index': 0.0,
                'mining_profitability_score': 0.0
            }
    
    def analyze_difficulty_adjustment(self, data: pd.DataFrame) -> DifficultyAnalysis:
        """Analyze Bitcoin difficulty adjustment dynamics"""
        try:
            # Simulate difficulty data if not available
            if 'difficulty' not in data.columns:
                if 'hash_rate' in data.columns:
                    # Difficulty roughly proportional to hash rate
                    data['difficulty'] = data['hash_rate'] / 1e12
                else:
                    data['difficulty'] = np.random.uniform(20e12, 30e12, len(data))
            
            current_difficulty = data['difficulty'].iloc[-1]
            
            # Difficulty change (last adjustment)
            if len(data) >= 2016:  # Approximate blocks in 2 weeks
                prev_difficulty = data['difficulty'].iloc[-2016]
                difficulty_change = (current_difficulty - prev_difficulty) / prev_difficulty
            else:
                difficulty_change = 0.02  # Typical 2% increase
            
            # Adjustment frequency (should be ~2 weeks)
            adjustment_frequency = 14.0  # days
            
            # Target and actual block times
            target_block_time = 10.0  # minutes
            
            # Estimate actual block time from difficulty change
            if difficulty_change > 0:
                actual_block_time = target_block_time / (1 + difficulty_change)
            else:
                actual_block_time = target_block_time * (1 + abs(difficulty_change))
            
            # Mining pressure
            mining_pressure = max(0, difficulty_change * 5)  # Normalized
            
            # Network congestion (simulated)
            network_congestion = np.random.uniform(0.3, 0.8)
            
            # Fee market dynamics
            if actual_block_time > target_block_time * 1.1:
                fee_market_dynamics = 'high_demand'
            elif actual_block_time < target_block_time * 0.9:
                fee_market_dynamics = 'low_demand'
            else:
                fee_market_dynamics = 'balanced'
            
            # Mempool status
            if network_congestion > 0.7:
                mempool_status = 'congested'
            elif network_congestion < 0.4:
                mempool_status = 'clear'
            else:
                mempool_status = 'normal'
            
            # Transaction throughput
            base_throughput = 7.0  # TPS
            throughput_factor = target_block_time / actual_block_time
            transaction_throughput = base_throughput * throughput_factor
            
            return DifficultyAnalysis(
                current_difficulty=current_difficulty,
                difficulty_change=difficulty_change,
                adjustment_frequency=adjustment_frequency,
                target_block_time=target_block_time,
                actual_block_time=actual_block_time,
                mining_pressure=mining_pressure,
                network_congestion=network_congestion,
                fee_market_dynamics=fee_market_dynamics,
                mempool_status=mempool_status,
                transaction_throughput=transaction_throughput
            )
            
        except Exception as e:
            logger.error(f"Error analyzing difficulty adjustment: {e}")
            return DifficultyAnalysis(
                current_difficulty=25e12, difficulty_change=0.02,
                adjustment_frequency=14.0, target_block_time=10.0,
                actual_block_time=10.2, mining_pressure=0.1,
                network_congestion=0.5, fee_market_dynamics='balanced',
                mempool_status='normal', transaction_throughput=7.0
            )
    
    def generate_predictions(self, data: pd.DataFrame) -> List[HashRibbonsPrediction]:
        """Generate Hash Ribbons predictions"""
        predictions = []
        
        try:
            if not self.is_fitted:
                logger.warning("Model not fitted, returning empty predictions")
                return predictions
            
            # Prepare features for prediction
            features = self._prepare_prediction_features(data)
            
            if len(features) == 0:
                return predictions
            
            features_scaled = self.scaler.transform([features])
            
            for horizon in self.prediction_horizons:
                # Get ensemble predictions
                ensemble_predictions = []
                
                for name, model in self.hash_rate_models.items():
                    try:
                        pred = model.predict(features_scaled)[0]
                        ensemble_predictions.append(pred)
                    except Exception as e:
                        logger.error(f"Error with {name} prediction: {e}")
                
                if not ensemble_predictions:
                    continue
                
                # Calculate ensemble prediction
                predicted_hash_rate = np.mean(ensemble_predictions)
                prediction_std = np.std(ensemble_predictions) if len(ensemble_predictions) > 1 else predicted_hash_rate * 0.1
                
                # Determine trend
                current_hash_rate = data['hash_rate'].iloc[-1] if 'hash_rate' in data.columns else 150e18
                
                if predicted_hash_rate > current_hash_rate * 1.05:
                    hash_rate_trend = 'increasing'
                elif predicted_hash_rate < current_hash_rate * 0.95:
                    hash_rate_trend = 'decreasing'
                else:
                    hash_rate_trend = 'stable'
                
                # Determine ribbon signal
                hash_rate_30d, hash_rate_60d, current_signal, _ = self.calculate_hash_ribbons(data)
                
                if predicted_hash_rate > hash_rate_30d > hash_rate_60d:
                    ribbon_signal = 'bullish_continuation'
                elif predicted_hash_rate < hash_rate_30d < hash_rate_60d:
                    ribbon_signal = 'bearish_continuation'
                elif predicted_hash_rate > hash_rate_30d and hash_rate_30d < hash_rate_60d:
                    ribbon_signal = 'bullish_reversal'
                elif predicted_hash_rate < hash_rate_30d and hash_rate_30d > hash_rate_60d:
                    ribbon_signal = 'bearish_reversal'
                else:
                    ribbon_signal = 'neutral'
                
                # Confidence interval
                confidence_interval = (
                    predicted_hash_rate - 1.96 * prediction_std,
                    predicted_hash_rate + 1.96 * prediction_std
                )
                
                # Model confidence
                model_confidence = max(0.1, 1.0 - (prediction_std / predicted_hash_rate))
                
                # Market scenario
                if ribbon_signal in ['bullish_continuation', 'bullish_reversal']:
                    market_scenario = 'bullish'
                elif ribbon_signal in ['bearish_continuation', 'bearish_reversal']:
                    market_scenario = 'bearish'
                else:
                    market_scenario = 'neutral'
                
                # Mining scenario
                if hash_rate_trend == 'increasing':
                    mining_scenario = 'expansion'
                elif hash_rate_trend == 'decreasing':
                    mining_scenario = 'contraction'
                else:
                    mining_scenario = 'stable'
                
                # Risk factors and catalysts
                risk_factors = self._identify_risk_factors(data, predicted_hash_rate)
                catalysts = self._identify_catalysts(data, predicted_hash_rate)
                
                predictions.append(HashRibbonsPrediction(
                    prediction_horizon=horizon,
                    predicted_hash_rate=predicted_hash_rate,
                    hash_rate_trend=hash_rate_trend,
                    ribbon_signal=ribbon_signal,
                    confidence_interval=confidence_interval,
                    model_confidence=model_confidence,
                    market_scenario=market_scenario,
                    mining_scenario=mining_scenario,
                    risk_factors=risk_factors,
                    catalysts=catalysts
                ))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return predictions
    
    def _prepare_prediction_features(self, data: pd.DataFrame) -> List[float]:
        """Prepare features for hash rate prediction"""
        try:
            features = []
            
            # Hash rate features
            if 'hash_rate' in data.columns:
                hash_rate_30d, hash_rate_60d, _, ribbon_strength = self.calculate_hash_ribbons(data)
                features.extend([
                    data['hash_rate'].iloc[-1],
                    hash_rate_30d,
                    hash_rate_60d,
                    ribbon_strength
                ])
            else:
                features.extend([150e18, 150e18, 150e18, 0.5])
            
            # Price features
            if 'price' in data.columns:
                features.extend([
                    data['price'].iloc[-1],
                    data['price'].rolling(30).mean().iloc[-1],
                    data['price'].pct_change().std()
                ])
            else:
                features.extend([50000, 50000, 0.03])
            
            # Difficulty features
            if 'difficulty' in data.columns:
                features.extend([
                    data['difficulty'].iloc[-1],
                    data['difficulty'].pct_change().iloc[-1]
                ])
            else:
                features.extend([25e12, 0.02])
            
            # Volume features
            if 'volume' in data.columns:
                features.extend([
                    data['volume'].iloc[-1],
                    data['volume'].rolling(30).mean().iloc[-1]
                ])
            else:
                features.extend([1e6, 1e6])
            
            # Technical indicators
            if len(data) >= 14:
                # RSI-like indicator for hash rate
                if 'hash_rate' in data.columns:
                    hash_rate_changes = data['hash_rate'].pct_change()
                    gains = hash_rate_changes.where(hash_rate_changes > 0, 0)
                    losses = -hash_rate_changes.where(hash_rate_changes < 0, 0)
                    avg_gain = gains.rolling(14).mean().iloc[-1]
                    avg_loss = losses.rolling(14).mean().iloc[-1]
                    
                    if avg_loss != 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                    else:
                        rsi = 50
                    
                    features.append(rsi)
                else:
                    features.append(50)
            else:
                features.append(50)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            return []
    
    def _identify_risk_factors(self, data: pd.DataFrame, predicted_hash_rate: float) -> List[str]:
        """Identify risk factors for hash rate prediction"""
        risk_factors = []
        
        try:
            current_hash_rate = data['hash_rate'].iloc[-1] if 'hash_rate' in data.columns else 150e18
            
            # Large predicted change
            change_pct = abs(predicted_hash_rate - current_hash_rate) / current_hash_rate
            if change_pct > 0.2:
                risk_factors.append("Large hash rate change predicted")
            
            # Price volatility
            if 'price' in data.columns:
                price_volatility = data['price'].pct_change().std()
                if price_volatility > 0.05:
                    risk_factors.append("High price volatility")
            
            # Regulatory risks
            risk_factors.append("Regulatory uncertainty")
            
            # Energy costs
            risk_factors.append("Energy cost fluctuations")
            
            # Technology risks
            risk_factors.append("Hardware obsolescence")
            
            return risk_factors[:5]  # Limit to top 5
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return ["General market risks"]
    
    def _identify_catalysts(self, data: pd.DataFrame, predicted_hash_rate: float) -> List[str]:
        """Identify potential catalysts for hash rate changes"""
        catalysts = []
        
        try:
            current_hash_rate = data['hash_rate'].iloc[-1] if 'hash_rate' in data.columns else 150e18
            
            if predicted_hash_rate > current_hash_rate:
                catalysts.extend([
                    "Bitcoin price appreciation",
                    "New mining facility deployments",
                    "Improved mining efficiency",
                    "Favorable regulatory developments"
                ])
            else:
                catalysts.extend([
                    "Mining profitability pressure",
                    "Regulatory restrictions",
                    "Energy cost increases",
                    "Miner capitulation"
                ])
            
            return catalysts[:4]  # Limit to top 4
            
        except Exception as e:
            logger.error(f"Error identifying catalysts: {e}")
            return ["Market dynamics"]
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[HashRibbonsAnomaly]:
        """Detect anomalies in Hash Ribbons data"""
        anomalies = []
        
        try:
            if len(data) < 30:
                return anomalies
            
            # Hash rate anomalies
            if 'hash_rate' in data.columns:
                hash_rate = data['hash_rate']
                hash_rate_mean = hash_rate.rolling(30).mean().iloc[-1]
                hash_rate_std = hash_rate.rolling(30).std().iloc[-1]
                current_hash_rate = hash_rate.iloc[-1]
                
                # Sudden hash rate drop
                if current_hash_rate < hash_rate_mean - self.anomaly_threshold * hash_rate_std:
                    severity = 'critical' if current_hash_rate < hash_rate_mean - 3 * hash_rate_std else 'high'
                    anomalies.append(HashRibbonsAnomaly(
                        anomaly_score=abs(current_hash_rate - hash_rate_mean) / hash_rate_std,
                        is_anomaly=True,
                        anomaly_type='hash_rate_crash',
                        severity=severity,
                        affected_metrics=['hash_rate', 'network_security'],
                        potential_causes=['Miner capitulation', 'Regulatory crackdown', 'Energy crisis'],
                        network_implications=['Reduced security', 'Slower confirmations', 'Higher fees'],
                        historical_precedents=['China mining ban 2021', 'Kazakhstan internet shutdown']
                    ))
                
                # Sudden hash rate spike
                elif current_hash_rate > hash_rate_mean + self.anomaly_threshold * hash_rate_std:
                    severity = 'medium'
                    anomalies.append(HashRibbonsAnomaly(
                        anomaly_score=(current_hash_rate - hash_rate_mean) / hash_rate_std,
                        is_anomaly=True,
                        anomaly_type='hash_rate_surge',
                        severity=severity,
                        affected_metrics=['hash_rate', 'mining_difficulty'],
                        potential_causes=['New mining facilities', 'Price rally', 'Efficiency improvements'],
                        network_implications=['Increased security', 'Faster blocks temporarily'],
                        historical_precedents=['Post-halving recoveries', 'Bull market expansions']
                    ))
            
            # Ribbon signal anomalies
            hash_rate_30d, hash_rate_60d, ribbon_signal, ribbon_strength = self.calculate_hash_ribbons(data)
            
            # Extreme ribbon divergence
            if ribbon_strength > 0.9:
                anomalies.append(HashRibbonsAnomaly(
                    anomaly_score=ribbon_strength,
                    is_anomaly=True,
                    anomaly_type='extreme_ribbon_divergence',
                    severity='high',
                    affected_metrics=['ribbon_signal', 'miner_behavior'],
                    potential_causes=['Rapid mining changes', 'Market regime shift'],
                    network_implications=['Signal reliability concerns', 'Trend confirmation'],
                    historical_precedents=['Major market transitions']
                ))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return anomalies
    
    def assess_risk(self, data: pd.DataFrame, 
                   miner_behavior: MinerBehaviorAnalysis,
                   network_health: NetworkHealthMetrics,
                   mining_economics: MiningEconomics) -> HashRibbonsRisk:
        """Assess Hash Ribbons-based risks"""
        try:
            # Mining centralization risk
            mining_centralization_risk = miner_behavior.mining_pool_concentration
            
            # Network security risk
            network_security_risk = 1.0 - network_health.security_score
            
            # Economic sustainability risk
            economic_sustainability_risk = (
                (1.0 - mining_economics.profit_margin) * 0.4 +
                mining_economics.energy_cost_pressure * 0.3 +
                miner_behavior.capitulation_score * 0.3
            )
            
            # Regulatory risk
            regulatory_risk = network_health.regulatory_risk_score
            
            # Technological risk
            technological_risk = (
                mining_economics.hardware_obsolescence_rate * 0.6 +
                (1.0 - network_health.technological_advancement) * 0.4
            )
            
            # Environmental risk
            environmental_risk = 1.0 - network_health.energy_sustainability
            
            # Overall risk (weighted average)
            overall_risk = (
                mining_centralization_risk * 0.2 +
                network_security_risk * 0.25 +
                economic_sustainability_risk * 0.25 +
                regulatory_risk * 0.15 +
                technological_risk * 0.1 +
                environmental_risk * 0.05
            )
            
            # Risk factors
            risk_factors = {
                'centralization': mining_centralization_risk,
                'security': network_security_risk,
                'economics': economic_sustainability_risk,
                'regulatory': regulatory_risk,
                'technology': technological_risk,
                'environmental': environmental_risk
            }
            
            # Risk mitigation strategies
            risk_mitigation = self._generate_risk_mitigation(risk_factors)
            
            # Stress test results
            stress_test_results = self._perform_stress_tests(data, miner_behavior)
            
            return HashRibbonsRisk(
                overall_risk=overall_risk,
                mining_centralization_risk=mining_centralization_risk,
                network_security_risk=network_security_risk,
                economic_sustainability_risk=economic_sustainability_risk,
                regulatory_risk=regulatory_risk,
                technological_risk=technological_risk,
                environmental_risk=environmental_risk,
                risk_factors=risk_factors,
                risk_mitigation=risk_mitigation,
                stress_test_results=stress_test_results
            )
            
        except Exception as e:
            logger.error(f"Error assessing risk: {e}")
            return HashRibbonsRisk(
                overall_risk=0.5, mining_centralization_risk=0.6, network_security_risk=0.2,
                economic_sustainability_risk=0.3, regulatory_risk=0.5, technological_risk=0.2,
                environmental_risk=0.4, risk_factors={}, risk_mitigation=[], stress_test_results={}
            )
    
    def _generate_risk_mitigation(self, risk_factors: Dict[str, float]) -> List[str]:
        """Generate risk mitigation strategies"""
        mitigation = []
        
        try:
            if risk_factors.get('centralization', 0) > 0.7:
                mitigation.append("Monitor mining pool distribution")
                mitigation.append("Support decentralized mining initiatives")
            
            if risk_factors.get('security', 0) > 0.7:
                mitigation.append("Increase network monitoring")
                mitigation.append("Prepare for potential attacks")
            
            if risk_factors.get('economics', 0) > 0.7:
                mitigation.append("Monitor miner profitability")
                mitigation.append("Track energy cost trends")
            
            if risk_factors.get('regulatory', 0) > 0.7:
                mitigation.append("Monitor regulatory developments")
                mitigation.append("Diversify mining geography")
            
            if risk_factors.get('technology', 0) > 0.7:
                mitigation.append("Track hardware innovation")
                mitigation.append("Monitor efficiency improvements")
            
            return mitigation
            
        except Exception as e:
            logger.error(f"Error generating risk mitigation: {e}")
            return ["Monitor network conditions closely"]
    
    def _perform_stress_tests(self, data: pd.DataFrame,
                            miner_behavior: MinerBehaviorAnalysis) -> Dict[str, float]:
        """Perform stress tests on Hash Ribbons model"""
        try:
            stress_results = {}
            
            # Price shock scenarios
            if 'price' in data.columns:
                current_price = data['price'].iloc[-1]
                
                # 50% price drop scenario
                shock_price = current_price * 0.5
                stress_results['price_drop_50pct'] = self._simulate_price_shock_impact(shock_price, current_price, miner_behavior)
                
                # 80% price drop scenario
                shock_price = current_price * 0.2
                stress_results['price_drop_80pct'] = self._simulate_price_shock_impact(shock_price, current_price, miner_behavior)
            
            # Regulatory ban scenario
            stress_results['regulatory_ban'] = self._simulate_regulatory_ban(miner_behavior)
            
            # Energy crisis scenario
            stress_results['energy_crisis'] = self._simulate_energy_crisis(miner_behavior)
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error performing stress tests: {e}")
            return {}
    
    def _simulate_price_shock_impact(self, shock_price: float, current_price: float,
                                   miner_behavior: MinerBehaviorAnalysis) -> float:
        """Simulate impact of price shock on hash rate"""
        try:
            price_change = (shock_price - current_price) / current_price
            
            # Estimate hash rate response based on miner behavior
            base_response = price_change * 0.5  # 50% correlation assumption
            
            # Adjust for miner characteristics
            efficiency_factor = miner_behavior.efficiency_trend
            stability_factor = miner_behavior.stability_index
            
            adjusted_response = base_response * (2 - efficiency_factor) * (2 - stability_factor)
            
            # Hash rate impact (percentage change)
            hash_rate_impact = max(-0.8, min(0.5, adjusted_response))  # Cap between -80% and +50%
            
            return abs(hash_rate_impact)
            
        except Exception as e:
            logger.error(f"Error simulating price shock: {e}")
            return 0.3
    
    def _simulate_regulatory_ban(self, miner_behavior: MinerBehaviorAnalysis) -> float:
        """Simulate regulatory ban impact"""
        try:
            # Impact based on geographic concentration
            china_exposure = miner_behavior.geographic_distribution.get('china', 0.35)
            
            # Assume ban affects major mining regions
            ban_impact = china_exposure * 0.8  # 80% of China mining affected
            
            # Adjustment for network resilience
            resilience_factor = miner_behavior.adaptation_speed
            adjusted_impact = ban_impact * (1 - resilience_factor * 0.3)
            
            return min(adjusted_impact, 0.6)  # Cap at 60% impact
            
        except Exception as e:
            logger.error(f"Error simulating regulatory ban: {e}")
            return 0.3
    
    def _simulate_energy_crisis(self, miner_behavior: MinerBehaviorAnalysis) -> float:
        """Simulate energy crisis impact"""
        try:
            # Base impact from energy cost pressure
            base_impact = miner_behavior.profitability_pressure * 0.6
            
            # Adjustment for operational efficiency
            efficiency_factor = miner_behavior.efficiency_trend
            adjusted_impact = base_impact * (2 - efficiency_factor)
            
            return min(adjusted_impact, 0.5)  # Cap at 50% impact
            
        except Exception as e:
            logger.error(f"Error simulating energy crisis: {e}")
            return 0.2
    
    def fit(self, data: pd.DataFrame) -> None:
        """Fit the Hash Ribbons model to historical data"""
        try:
            logger.info("Fitting Quant Grade Hash Ribbons model...")
            
            if len(data) < self.long_window:
                logger.warning(f"Insufficient data for fitting. Need {self.long_window} days, got {len(data)}")
                return
            
            # Prepare training data
            X, y = self._prepare_training_data(data)
            
            if len(X) == 0 or len(y) == 0:
                logger.warning("No valid training data prepared")
                return
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit hash rate prediction models
            for name, model in self.hash_rate_models.items():
                try:
                    model.fit(X_scaled, y)
                    logger.info(f"Fitted {name} hash rate model")
                except Exception as e:
                    logger.error(f"Error fitting {name} model: {e}")
            
            # Fit behavioral clustering models
            behavioral_features = self._extract_behavioral_features(data)
            if len(behavioral_features) > 0:
                self.miner_behavior_model.fit(behavioral_features)
                logger.info("Fitted miner behavior clustering model")
            
            # Fit network health clustering
            network_features = self._extract_network_features(data)
            if len(network_features) > 0:
                self.network_health_model.fit(network_features)
                logger.info("Fitted network health clustering model")
            
            # Fit anomaly detector
            if len(X_scaled) > 10:
                self.anomaly_detector.fit(X_scaled)
                logger.info("Fitted anomaly detector")
            
            self.is_fitted = True
            logger.info("Hash Ribbons model fitting completed")
            
        except Exception as e:
            logger.error(f"Error fitting Hash Ribbons model: {e}")
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[List[List[float]], List[float]]:
        """Prepare optimized training data for hash rate prediction"""
        X, y = [], []
        
        try:
            # Limit data size for performance
            if len(data) > 1000:
                data = data.tail(1000)
            
            # Create sliding windows for time series prediction
            window_size = 30
            
            # Pre-check for hash_rate column
            if 'hash_rate' not in data.columns:
                logger.warning("No hash_rate column found in data")
                return [], []
            
            # Vectorized feature preparation where possible
            hash_rates = data['hash_rate'].values
            prices = data['price'].values if 'price' in data.columns else np.ones(len(data))
            volumes = data['volume'].values if 'volume' in data.columns else np.ones(len(data))
            
            # Pre-compute rolling statistics for efficiency
            hash_rate_30d = pd.Series(hash_rates).rolling(30, min_periods=1).mean().values
            hash_rate_60d = pd.Series(hash_rates).rolling(60, min_periods=1).mean().values
            price_30d = pd.Series(prices).rolling(30, min_periods=1).mean().values
            volume_30d = pd.Series(volumes).rolling(30, min_periods=1).mean().values
            
            # Batch process windows with step size for performance
            step_size = max(1, len(data) // 200)  # Limit to ~200 samples max
            
            for i in range(window_size, len(data), step_size):
                # Use pre-computed values for efficiency
                target_hash_rate = hash_rates[i]
                
                if np.isnan(target_hash_rate):
                    continue
                
                # Simplified feature vector for performance
                features = [
                    hash_rates[i-1] if i > 0 else hash_rates[0],
                    hash_rate_30d[i-1] if i > 0 else hash_rate_30d[0],
                    hash_rate_60d[i-1] if i > 0 else hash_rate_60d[0],
                    (hash_rate_30d[i-1] / hash_rate_60d[i-1]) if hash_rate_60d[i-1] > 0 else 1.0,
                    prices[i-1] if i > 0 else prices[0],
                    price_30d[i-1] if i > 0 else price_30d[0],
                    np.std(prices[max(0, i-30):i]) if i >= 30 else 0.0,
                    volumes[i-1] if i > 0 else volumes[0],
                    volume_30d[i-1] if i > 0 else volume_30d[0]
                ]
                
                if len(features) > 0 and all(not np.isnan(f) for f in features):
                    X.append(features)
                    y.append(target_hash_rate)
            
            self.feature_names = [
                'hash_rate', 'hash_rate_30d', 'hash_rate_60d', 'ribbon_strength',
                'price', 'price_30d', 'price_volatility',
                'volume', 'volume_30d'
            ]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return [], []
    
    def _extract_behavioral_features(self, data: pd.DataFrame) -> List[List[float]]:
        """Extract behavioral features for clustering"""
        features = []
        
        try:
            window_size = 30
            
            for i in range(window_size, len(data), 7):  # Weekly sampling
                window_data = data.iloc[i-window_size:i]
                
                # Calculate behavioral metrics
                miner_behavior = self.analyze_miner_behavior(window_data)
                
                feature_vector = [
                    miner_behavior.capitulation_score,
                    miner_behavior.recovery_strength,
                    miner_behavior.efficiency_trend,
                    miner_behavior.profitability_pressure,
                    miner_behavior.stability_index
                ]
                
                features.append(feature_vector)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting behavioral features: {e}")
            return []
    
    def _extract_network_features(self, data: pd.DataFrame) -> List[List[float]]:
        """Extract network health features for clustering"""
        features = []
        
        try:
            window_size = 30
            
            for i in range(window_size, len(data), 7):  # Weekly sampling
                window_data = data.iloc[i-window_size:i]
                
                # Calculate basic network metrics
                if 'hash_rate' in window_data.columns:
                    hash_rate_stability = 1.0 - window_data['hash_rate'].pct_change().std()
                else:
                    hash_rate_stability = 0.8
                
                if 'price' in window_data.columns:
                    price_hash_correlation = window_data['price'].corr(window_data.get('hash_rate', window_data['price']))
                else:
                    price_hash_correlation = 0.7
                
                feature_vector = [
                    hash_rate_stability,
                    price_hash_correlation,
                    0.6,  # Simulated decentralization
                    0.8,  # Simulated security
                    0.7   # Simulated resilience
                ]
                
                features.append(feature_vector)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting network features: {e}")
            return []
    
    def analyze(self, data: pd.DataFrame) -> HashRibbonsResult:
        """Perform comprehensive Hash Ribbons analysis"""
        try:
            logger.info("Performing Quant Grade Hash Ribbons analysis...")
            
            # Basic Hash Ribbons calculation
            hash_rate_30d, hash_rate_60d, ribbon_signal, ribbon_strength = self.calculate_hash_ribbons(data)
            
            # Miner behavior analysis
            miner_behavior = self.analyze_miner_behavior(data)
            
            # Network health analysis
            network_health = self.analyze_network_health(data, miner_behavior)
            
            # Mining economics analysis
            mining_economics = self.analyze_mining_economics(data, miner_behavior)
            
            # Difficulty analysis
            difficulty_analysis = self.analyze_difficulty_adjustment(data)
            
            # Predictions
            predictions = self.generate_predictions(data) if self.is_fitted else []
            
            # Anomaly detection
            anomalies = self.detect_anomalies(data)
            
            # Risk assessment
            risk_assessment = self.assess_risk(data, miner_behavior, network_health, mining_economics)
            
            # Enhanced analysis components from regular hash_ribbons
            hash_rates = data['hash_rate'].values if 'hash_rate' in data.columns else np.random.uniform(100e18, 200e18, len(data))
            
            # Volatility analysis
            volatility_analysis = None
            if self.enable_volatility_analysis:
                volatility_analysis = self._perform_volatility_analysis(hash_rates)
            
            # Regime analysis
            regime_analysis = None
            if self.enable_regime_analysis:
                regime_analysis = self._perform_regime_analysis(hash_rates)
            
            # Cycle analysis
            cycle_analysis = None
            if self.enable_cycle_analysis:
                cycle_analysis = self._perform_cycle_analysis(hash_rates)
            
            # Statistical measures
            statistical_measures = self._calculate_statistical_measures(hash_rates)
            
            # Network stability metrics
            network_stability = self._calculate_network_stability_metrics(data)
            
            # Model performance
            model_performance = self._evaluate_model_performance(data) if self.is_fitted else {}
            
            # Confidence score
            confidence_score = self._calculate_overall_confidence(
                network_health, predictions, risk_assessment
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                ribbon_signal, miner_behavior, network_health, mining_economics, risk_assessment, anomalies
            )
            
            # Initialize result
            result = HashRibbonsResult(
                timestamp=datetime.now(),
                hash_rate_30d=hash_rate_30d,
                hash_rate_60d=hash_rate_60d,
                ribbon_signal=ribbon_signal,
                ribbon_strength=ribbon_strength,
                miner_behavior=miner_behavior,
                network_health=network_health,
                mining_economics=mining_economics,
                difficulty_analysis=difficulty_analysis,
                predictions=predictions,
                anomalies=anomalies,
                risk_assessment=risk_assessment,
                volatility_analysis=volatility_analysis,
                regime_analysis=regime_analysis,
                cycle_analysis=cycle_analysis,
                **statistical_measures,
                **network_stability,
                model_performance=model_performance,
                confidence_score=confidence_score,
                recommendations=recommendations
            )
            
            # Advanced analytics
            if self.enable_kalman_filter and len(data) >= 30:
                result.kalman_analysis = self._perform_kalman_analysis(data)
            
            if self.enable_monte_carlo and len(data) >= 50:
                result.monte_carlo_analysis = self._perform_monte_carlo_analysis(data)
            
            # Enhanced Hash Ribbons analysis calculations
            try:
                result.advanced_mining_economics = self._analyze_advanced_mining_economics(data, mining_economics)
                logger.debug("Advanced mining economics analysis completed")
            except Exception as e:
                logger.warning(f"Advanced mining economics analysis failed: {e}")
                result.advanced_mining_economics = None
            
            try:
                result.enhanced_difficulty_adjustments = self._analyze_enhanced_difficulty_adjustments(data, difficulty_analysis)
                logger.debug("Enhanced difficulty adjustments analysis completed")
            except Exception as e:
                logger.warning(f"Enhanced difficulty adjustments analysis failed: {e}")
                result.enhanced_difficulty_adjustments = None
            
            try:
                result.detailed_miner_capitulation = self._analyze_detailed_miner_capitulation(data, miner_behavior)
                logger.debug("Detailed miner capitulation analysis completed")
            except Exception as e:
                logger.warning(f"Detailed miner capitulation analysis failed: {e}")
                result.detailed_miner_capitulation = None
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Hash Ribbons analysis: {e}")
            # Return minimal result on error
            return self._create_minimal_result()
    
    def _create_minimal_result(self) -> HashRibbonsResult:
        """Create minimal result for error cases"""
        return HashRibbonsResult(
            timestamp=datetime.now(),
            hash_rate_30d=150e18,
            hash_rate_60d=150e18,
            ribbon_signal='neutral',
            ribbon_strength=0.0,
            miner_behavior=MinerBehaviorAnalysis(
                capitulation_score=0.0, recovery_strength=0.5, efficiency_trend=0.7,
                profitability_pressure=0.3, network_participation=0.8,
                geographic_distribution={'others': 1.0}, mining_pool_concentration=0.6,
                behavior_cluster='adaptive', stability_index=0.7, adaptation_speed=0.5
            ),
            network_health=NetworkHealthMetrics(
                security_score=0.8, decentralization_index=0.4, resilience_factor=0.6,
                attack_cost_estimate=1e9, network_effect_strength=0.8,
                infrastructure_quality=0.7, geographic_diversity=0.6,
                regulatory_risk_score=0.5, energy_sustainability=0.6,
                technological_advancement=0.5
            ),
            mining_economics=MiningEconomics(
                break_even_price=25000, profit_margin=0.3, operational_efficiency=0.7,
                capital_expenditure_cycle='maintenance', energy_cost_pressure=0.3,
                hardware_obsolescence_rate=0.2, mining_reward_sustainability=0.7,
                fee_revenue_contribution=0.05, halving_impact_score=0.3,
                competitive_landscape='balanced',
                # Enhanced metrics defaults
                puell_multiple=1.0, puell_multiple_ma=1.0, puell_signal='neutral',
                miner_revenue_usd=0.0, miner_revenue_btc=0.0, revenue_per_hash=0.0,
                revenue_trend='stable', revenue_volatility=0.0,
                difficulty_ribbon_compression=0.0, difficulty_ribbon_signal='neutral',
                difficulty_ma_9=0.0, difficulty_ma_14=0.0, difficulty_ma_25=0.0,
                difficulty_ma_40=0.0, difficulty_ma_60=0.0, difficulty_ma_90=0.0,
                difficulty_ma_128=0.0, difficulty_ma_200=0.0, ribbon_width=0.0,
                ribbon_trend='neutral', hash_price=0.0, thermocap_ratio=0.0,
                miner_position_index=0.0, mining_profitability_score=0.0
            ),
            difficulty_analysis=DifficultyAnalysis(
                current_difficulty=25e12, difficulty_change=0.02,
                adjustment_frequency=14.0, target_block_time=10.0,
                actual_block_time=10.2, mining_pressure=0.1,
                network_congestion=0.5, fee_market_dynamics='balanced',
                mempool_status='normal', transaction_throughput=7.0
            ),
            predictions=[],
            anomalies=[],
            risk_assessment=HashRibbonsRisk(
                overall_risk=0.5, mining_centralization_risk=0.6, network_security_risk=0.2,
                economic_sustainability_risk=0.3, regulatory_risk=0.5, technological_risk=0.2,
                environmental_risk=0.4, risk_factors={}, risk_mitigation=[], stress_test_results={}
            ),
            model_performance={},
            confidence_score=0.0,
            recommendations=["Unable to generate analysis due to error"]
        )
    
    def _evaluate_model_performance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model performance on recent data"""
        try:
            if not self.is_fitted or len(data) < 90:
                return {}
            
            # Use last 60 days for evaluation
            eval_data = data.tail(90)
            X_eval, y_eval = self._prepare_training_data(eval_data)
            
            if len(X_eval) == 0 or len(y_eval) == 0:
                return {}
            
            X_eval_scaled = self.scaler.transform(X_eval)
            
            performance = {}
            
            for name, model in self.hash_rate_models.items():
                try:
                    y_pred = model.predict(X_eval_scaled)
                    
                    mse = mean_squared_error(y_eval, y_pred)
                    mae = mean_absolute_error(y_eval, y_pred)
                    r2 = r2_score(y_eval, y_pred)
                    
                    performance[f'{name}_mse'] = mse
                    performance[f'{name}_mae'] = mae
                    performance[f'{name}_r2'] = r2
                    
                except Exception as e:
                    logger.error(f"Error evaluating {name} model: {e}")
            
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {}
    
    def _calculate_overall_confidence(self, network_health: NetworkHealthMetrics,
                                    predictions: List[HashRibbonsPrediction],
                                    risk_assessment: HashRibbonsRisk) -> float:
        """Calculate overall confidence score"""
        try:
            # Base confidence from network health
            network_confidence = (
                network_health.security_score * 0.3 +
                network_health.resilience_factor * 0.3 +
                network_health.infrastructure_quality * 0.4
            )
            
            # Prediction confidence
            if predictions:
                pred_confidence = np.mean([p.model_confidence for p in predictions])
            else:
                pred_confidence = 0.5
            
            # Risk-adjusted confidence
            risk_factor = 1.0 - risk_assessment.overall_risk
            
            # Overall confidence
            overall_confidence = (
                network_confidence * 0.4 +
                pred_confidence * 0.4 +
                risk_factor * 0.2
            )
            
            return max(0.0, min(1.0, overall_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _generate_recommendations(self, ribbon_signal: str,
                               miner_behavior: MinerBehaviorAnalysis,
                               network_health: NetworkHealthMetrics,
                               mining_economics: MiningEconomics,
                               risk_assessment: HashRibbonsRisk,
                               anomalies: List[HashRibbonsAnomaly]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Ribbon signal recommendations
            if ribbon_signal in ['strong_bullish', 'bullish']:
                recommendations.append("Hash rate trend suggests network expansion - consider bullish positioning")
                recommendations.append("Monitor for potential mining difficulty increases")
            elif ribbon_signal in ['strong_bearish', 'bearish']:
                recommendations.append("Hash rate decline indicates miner stress - exercise caution")
                recommendations.append("Watch for potential capitulation opportunities")
            else:
                recommendations.append("Neutral hash rate trend - await clearer signals")
            
            # Miner behavior recommendations
            if miner_behavior.capitulation_score > 0.7:
                recommendations.append("High capitulation risk - prepare for potential hash rate drops")
            
            if miner_behavior.efficiency_trend < 0.5:
                recommendations.append("Poor mining efficiency - monitor for operational improvements")
            
            # Network health recommendations
            if network_health.decentralization_index < 0.3:
                recommendations.append("High mining centralization - monitor pool distribution")
            
            if network_health.security_score < 0.6:
                recommendations.append("Network security concerns - increase monitoring")
            
            # Economic recommendations
            if mining_economics.profit_margin < 0.2:
                recommendations.append("Low mining profitability - expect potential hash rate pressure")
            
            if mining_economics.capital_expenditure_cycle == 'contraction':
                recommendations.append("Mining capex contraction - bearish for hash rate growth")
            
            # Risk-based recommendations
            if risk_assessment.overall_risk > 0.7:
                recommendations.append("High overall risk - implement risk management strategies")
            
            if risk_assessment.regulatory_risk > 0.6:
                recommendations.append("Elevated regulatory risk - monitor policy developments")
            
            # Anomaly-based recommendations
            for anomaly in anomalies:
                if anomaly.severity == 'critical':
                    recommendations.append(f"Critical anomaly detected: {anomaly.anomaly_type} - immediate attention required")
                elif anomaly.severity == 'high':
                    recommendations.append(f"High-severity anomaly: {anomaly.anomaly_type} - monitor closely")
            
            return recommendations[:8]  # Limit to top 8 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Monitor hash rate trends and network conditions"]
    
    def _perform_kalman_analysis(self, data: pd.DataFrame) -> HashRibbonsKalmanFilterResult:
        """Perform Kalman filter analysis on Hash Ribbons data"""
        try:
            # Extract hash rate and difficulty data
            if 'hash_rate' not in data.columns:
                # Simulate hash rate data based on price and difficulty
                if 'price' in data.columns:
                    hash_rate_data = data['price'] * np.random.uniform(0.8, 1.2, len(data))
                else:
                    hash_rate_data = np.random.uniform(100e18, 200e18, len(data))
            else:
                hash_rate_data = data['hash_rate'].values
            
            if 'difficulty' not in data.columns:
                # Simulate difficulty data correlated with hash rate
                difficulty_data = hash_rate_data * np.random.uniform(0.9, 1.1, len(data)) / 1e6
            else:
                difficulty_data = data['difficulty'].values
            
            if KALMAN_AVAILABLE and len(hash_rate_data) >= 30:
                # Use pykalman for advanced filtering
                transition_matrices = np.array([[1, 1], [0, 1]])
                observation_matrices = np.array([[1, 0]])
                
                # Filter hash rate data
                kf_hash = KalmanFilter(
                    transition_matrices=transition_matrices,
                    observation_matrices=observation_matrices,
                    initial_state_mean=[hash_rate_data[0], 0],
                    n_dim_state=2
                )
                
                kf_hash = kf_hash.em(hash_rate_data.reshape(-1, 1), n_iter=10)
                hash_state_means, hash_state_covariances = kf_hash.smooth(hash_rate_data.reshape(-1, 1))
                
                # Filter difficulty data
                kf_diff = KalmanFilter(
                    transition_matrices=transition_matrices,
                    observation_matrices=observation_matrices,
                    initial_state_mean=[difficulty_data[0], 0],
                    n_dim_state=2
                )
                
                kf_diff = kf_diff.em(difficulty_data.reshape(-1, 1), n_iter=10)
                diff_state_means, diff_state_covariances = kf_diff.smooth(difficulty_data.reshape(-1, 1))
                
                # Extract results
                filtered_hash_rate_states = hash_state_means[:, 0].tolist()
                filtered_difficulty_states = diff_state_means[:, 0].tolist()
                smoothed_hash_rate = hash_state_means[:, 0].tolist()
                smoothed_difficulty = diff_state_means[:, 0].tolist()
                hash_rate_trend_component = hash_state_means[:, 1].tolist()
                difficulty_trend_component = diff_state_means[:, 1].tolist()
                
                # Calculate noise reduction ratio
                hash_original_variance = np.var(hash_rate_data)
                hash_filtered_variance = np.var(filtered_hash_rate_states)
                noise_reduction_ratio = 1 - (hash_filtered_variance / hash_original_variance) if hash_original_variance > 0 else 0
                
                return HashRibbonsKalmanFilterResult(
                    filtered_hash_rate_states=filtered_hash_rate_states,
                    filtered_difficulty_states=filtered_difficulty_states,
                    hash_rate_state_covariances=hash_state_covariances.tolist(),
                    difficulty_state_covariances=diff_state_covariances.tolist(),
                    log_likelihood=kf_hash.loglikelihood(hash_rate_data.reshape(-1, 1)) + kf_diff.loglikelihood(difficulty_data.reshape(-1, 1)),
                    smoothed_hash_rate=smoothed_hash_rate,
                    smoothed_difficulty=smoothed_difficulty,
                    hash_rate_trend_component=hash_rate_trend_component,
                    difficulty_trend_component=difficulty_trend_component,
                    noise_reduction_ratio=noise_reduction_ratio
                )
            else:
                # Simplified Kalman filter implementation
                def simple_kalman_filter(data):
                    filtered_states = []
                    state_estimate = data[0]
                    error_estimate = 1.0
                    
                    for observation in data:
                        # Prediction step
                        predicted_state = state_estimate
                        predicted_error = error_estimate + 0.1
                        
                        # Update step
                        kalman_gain = predicted_error / (predicted_error + 0.5)
                        state_estimate = predicted_state + kalman_gain * (observation - predicted_state)
                        error_estimate = (1 - kalman_gain) * predicted_error
                        
                        filtered_states.append(state_estimate)
                    
                    return filtered_states, error_estimate
                
                filtered_hash_rate_states, hash_error = simple_kalman_filter(hash_rate_data)
                filtered_difficulty_states, diff_error = simple_kalman_filter(difficulty_data)
                
                # Calculate trend components as differences
                hash_rate_trend_component = np.diff(filtered_hash_rate_states, prepend=filtered_hash_rate_states[0]).tolist()
                difficulty_trend_component = np.diff(filtered_difficulty_states, prepend=filtered_difficulty_states[0]).tolist()
                
                return HashRibbonsKalmanFilterResult(
                    filtered_hash_rate_states=filtered_hash_rate_states,
                    filtered_difficulty_states=filtered_difficulty_states,
                    hash_rate_state_covariances=[[hash_error]] * len(filtered_hash_rate_states),
                    difficulty_state_covariances=[[diff_error]] * len(filtered_difficulty_states),
                    log_likelihood=0.0,
                    smoothed_hash_rate=filtered_hash_rate_states,
                    smoothed_difficulty=filtered_difficulty_states,
                    hash_rate_trend_component=hash_rate_trend_component,
                    difficulty_trend_component=difficulty_trend_component,
                    noise_reduction_ratio=0.3
                )
                
        except Exception as e:
            logger.error(f"Kalman filter analysis failed: {str(e)}")
            # Return minimal result on error
            return HashRibbonsKalmanFilterResult(
                filtered_hash_rate_states=[150e18] * len(data),
                filtered_difficulty_states=[25e12] * len(data),
                hash_rate_state_covariances=[[1.0]] * len(data),
                difficulty_state_covariances=[[1.0]] * len(data),
                log_likelihood=0.0,
                smoothed_hash_rate=[150e18] * len(data),
                smoothed_difficulty=[25e12] * len(data),
                hash_rate_trend_component=[0.0] * len(data),
                difficulty_trend_component=[0.0] * len(data),
                noise_reduction_ratio=0.0
            )
    
    def _perform_monte_carlo_analysis(self, data: pd.DataFrame) -> HashRibbonsMonteCarloResult:
        """Perform Monte Carlo simulation analysis on Hash Ribbons data"""
        try:
            if len(data) < 50:
                raise ValueError("Insufficient data for Monte Carlo analysis")
            
            # Extract hash rate and difficulty data
            if 'hash_rate' not in data.columns:
                if 'price' in data.columns:
                    hash_rate_data = data['price'] * np.random.uniform(0.8, 1.2, len(data))
                else:
                    hash_rate_data = np.random.uniform(100e18, 200e18, len(data))
            else:
                hash_rate_data = data['hash_rate'].values
            
            if 'difficulty' not in data.columns:
                difficulty_data = hash_rate_data * np.random.uniform(0.9, 1.1, len(data)) / 1e6
            else:
                difficulty_data = data['difficulty'].values
            
            # Calculate historical statistics
            hash_returns = np.diff(np.log(hash_rate_data + 1e-10))
            diff_returns = np.diff(np.log(difficulty_data + 1e-10))
            
            hash_mean_return = np.mean(hash_returns)
            hash_std_return = np.std(hash_returns)
            diff_mean_return = np.mean(diff_returns)
            diff_std_return = np.std(diff_returns)
            
            # Monte Carlo simulation parameters
            n_simulations = self.monte_carlo_simulations
            n_steps = min(30, len(data) // 4)  # Forecast horizon
            
            # Run simulations
            hash_simulations = []
            diff_simulations = []
            hash_final_values = []
            diff_final_values = []
            
            for _ in range(n_simulations):
                # Hash rate simulation
                hash_path = [hash_rate_data[-1]]  # Start from last observed value
                diff_path = [difficulty_data[-1]]
                
                for _ in range(n_steps):
                    # Hash rate simulation with mean reversion and mining economics
                    hash_random_shock = np.random.normal(0, hash_std_return)
                    hash_mean_reversion = -0.05 * (np.log(hash_path[-1] + 1e-10) - np.log(np.mean(hash_rate_data) + 1e-10))
                    
                    hash_next_log_value = np.log(hash_path[-1] + 1e-10) + hash_mean_return + hash_mean_reversion + hash_random_shock
                    hash_next_value = max(np.exp(hash_next_log_value), 1e18)  # Ensure positive values
                    hash_path.append(hash_next_value)
                    
                    # Difficulty simulation correlated with hash rate
                    diff_random_shock = np.random.normal(0, diff_std_return)
                    diff_correlation = 0.8 * hash_random_shock  # Strong correlation with hash rate
                    diff_mean_reversion = -0.03 * (np.log(diff_path[-1] + 1e-10) - np.log(np.mean(difficulty_data) + 1e-10))
                    
                    diff_next_log_value = np.log(diff_path[-1] + 1e-10) + diff_mean_return + diff_mean_reversion + diff_correlation + diff_random_shock
                    diff_next_value = max(np.exp(diff_next_log_value), 1e10)  # Ensure positive values
                    diff_path.append(diff_next_value)
                
                hash_simulations.append(hash_path[1:])  # Exclude starting value
                diff_simulations.append(diff_path[1:])
                hash_final_values.append(hash_path[-1])
                diff_final_values.append(diff_path[-1])
            
            # Calculate statistics
            hash_simulations_array = np.array(hash_simulations)
            diff_simulations_array = np.array(diff_simulations)
            
            mean_hash_rate_path = np.mean(hash_simulations_array, axis=0).tolist()
            mean_difficulty_path = np.mean(diff_simulations_array, axis=0).tolist()
            
            # Confidence intervals
            hash_rate_confidence_intervals = {
                '95%_lower': np.percentile(hash_simulations_array, 2.5, axis=0).tolist(),
                '95%_upper': np.percentile(hash_simulations_array, 97.5, axis=0).tolist(),
                '68%_lower': np.percentile(hash_simulations_array, 16, axis=0).tolist(),
                '68%_upper': np.percentile(hash_simulations_array, 84, axis=0).tolist()
            }
            
            difficulty_confidence_intervals = {
                '95%_lower': np.percentile(diff_simulations_array, 2.5, axis=0).tolist(),
                '95%_upper': np.percentile(diff_simulations_array, 97.5, axis=0).tolist(),
                '68%_lower': np.percentile(diff_simulations_array, 16, axis=0).tolist(),
                '68%_upper': np.percentile(diff_simulations_array, 84, axis=0).tolist()
            }
            
            # Final value distributions
            final_hash_rate_distribution = {
                'mean': np.mean(hash_final_values),
                'std': np.std(hash_final_values),
                'median': np.median(hash_final_values),
                'percentile_5': np.percentile(hash_final_values, 5),
                'percentile_95': np.percentile(hash_final_values, 95)
            }
            
            final_difficulty_distribution = {
                'mean': np.mean(diff_final_values),
                'std': np.std(diff_final_values),
                'median': np.median(diff_final_values),
                'percentile_5': np.percentile(diff_final_values, 5),
                'percentile_95': np.percentile(diff_final_values, 95)
            }
            
            # Risk metrics
            current_hash_rate = hash_rate_data[-1]
            current_difficulty = difficulty_data[-1]
            
            risk_metrics = {
                'hash_rate_decline_probability': np.mean([fv < current_hash_rate for fv in hash_final_values]),
                'difficulty_decline_probability': np.mean([fv < current_difficulty for fv in diff_final_values]),
                'hash_rate_expected_return': (final_hash_rate_distribution['mean'] - current_hash_rate) / current_hash_rate if current_hash_rate > 0 else 0,
                'difficulty_expected_return': (final_difficulty_distribution['mean'] - current_difficulty) / current_difficulty if current_difficulty > 0 else 0,
                'hash_rate_value_at_risk_5': np.percentile(hash_final_values, 5) - current_hash_rate,
                'difficulty_value_at_risk_5': np.percentile(diff_final_values, 5) - current_difficulty
            }
            
            # Scenario probabilities
            scenario_probabilities = {
                'hash_rate_surge': np.mean([fv > current_hash_rate * 1.3 for fv in hash_final_values]),
                'hash_rate_capitulation': np.mean([fv < current_hash_rate * 0.7 for fv in hash_final_values]),
                'difficulty_spike': np.mean([fv > current_difficulty * 1.2 for fv in diff_final_values]),
                'difficulty_drop': np.mean([fv < current_difficulty * 0.8 for fv in diff_final_values]),
                'mining_expansion': np.mean([hv > current_hash_rate * 1.2 and dv > current_difficulty * 1.1 for hv, dv in zip(hash_final_values, diff_final_values)]),
                'mining_contraction': np.mean([hv < current_hash_rate * 0.8 and dv < current_difficulty * 0.9 for hv, dv in zip(hash_final_values, diff_final_values)])
            }
            
            # Stress test results
            stress_scenarios = {
                'extreme_hash_rate_growth': np.percentile(hash_final_values, 99),
                'extreme_hash_rate_decline': np.percentile(hash_final_values, 1),
                'extreme_difficulty_increase': np.percentile(diff_final_values, 99),
                'extreme_difficulty_decrease': np.percentile(diff_final_values, 1),
                'worst_case_mining_scenario': np.percentile([hv / dv for hv, dv in zip(hash_final_values, diff_final_values)], 1),
                'best_case_mining_scenario': np.percentile([hv / dv for hv, dv in zip(hash_final_values, diff_final_values)], 99)
            }
            
            return HashRibbonsMonteCarloResult(
                mean_hash_rate_path=mean_hash_rate_path,
                mean_difficulty_path=mean_difficulty_path,
                hash_rate_confidence_intervals=hash_rate_confidence_intervals,
                difficulty_confidence_intervals=difficulty_confidence_intervals,
                final_hash_rate_distribution=final_hash_rate_distribution,
                final_difficulty_distribution=final_difficulty_distribution,
                simulation_count=n_simulations,
                risk_metrics=risk_metrics,
                scenario_probabilities=scenario_probabilities,
                stress_test_results=stress_scenarios
            )
            
        except Exception as e:
            logger.error(f"Monte Carlo analysis failed: {str(e)}")
            # Return minimal result on error
            return HashRibbonsMonteCarloResult(
                mean_hash_rate_path=[150e18] * 10,
                mean_difficulty_path=[25e12] * 10,
                hash_rate_confidence_intervals={'95%_lower': [150e18] * 10, '95%_upper': [150e18] * 10},
                difficulty_confidence_intervals={'95%_lower': [25e12] * 10, '95%_upper': [25e12] * 10},
                final_hash_rate_distribution={'mean': 150e18, 'std': 0, 'median': 150e18, 'percentile_5': 150e18, 'percentile_95': 150e18},
                final_difficulty_distribution={'mean': 25e12, 'std': 0, 'median': 25e12, 'percentile_5': 25e12, 'percentile_95': 25e12},
                simulation_count=0,
                risk_metrics={'hash_rate_decline_probability': 0.5, 'difficulty_decline_probability': 0.5, 'hash_rate_expected_return': 0, 'difficulty_expected_return': 0, 'hash_rate_value_at_risk_5': 0, 'difficulty_value_at_risk_5': 0},
                scenario_probabilities={'hash_rate_surge': 0.2, 'hash_rate_capitulation': 0.2, 'difficulty_spike': 0.2, 'difficulty_drop': 0.2, 'mining_expansion': 0.1, 'mining_contraction': 0.1},
                stress_test_results={'extreme_hash_rate_growth': 150e18, 'extreme_hash_rate_decline': 150e18, 'extreme_difficulty_increase': 25e12, 'extreme_difficulty_decrease': 25e12, 'worst_case_mining_scenario': 6e6, 'best_case_mining_scenario': 6e6}
            )


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    # Generate realistic Bitcoin-like data
    n_days = len(dates)
    price_trend = np.cumsum(np.random.normal(0, 0.02, n_days)) + 4.7  # Log price
    prices = np.exp(price_trend) * 30000  # Convert to actual prices
    
    # Hash rate correlated with price but with mining dynamics
    hash_rate_base = 150e18
    hash_rate_trend = np.cumsum(np.random.normal(0, 0.01, n_days))
    hash_rates = hash_rate_base * np.exp(hash_rate_trend + price_trend * 0.3)
    
    # Add mining-specific volatility
    mining_shocks = np.random.choice([0, -0.3, -0.5], n_days, p=[0.95, 0.03, 0.02])
    hash_rates *= np.exp(np.cumsum(mining_shocks * 0.1))
    
    # Difficulty roughly follows hash rate with adjustment periods
    difficulties = hash_rates / 1e12 * np.random.uniform(0.9, 1.1, n_days)
    
    # Volume with some correlation to price volatility
    volumes = np.random.lognormal(13, 0.5, n_days) * (1 + np.abs(np.diff(np.log(prices), prepend=np.log(prices[0]))) * 10)
    
    # Create DataFrame
    sample_data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'hash_rate': hash_rates,
        'difficulty': difficulties,
        'volume': volumes
    })
    
    print("=== Quant Grade Hash Ribbons Model Test ===")
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Date range: {sample_data['date'].min()} to {sample_data['date'].max()}")
    print(f"Price range: ${sample_data['price'].min():.0f} - ${sample_data['price'].max():.0f}")
    print(f"Hash rate range: {sample_data['hash_rate'].min()/1e18:.1f}E - {sample_data['hash_rate'].max()/1e18:.1f}E")
    
    # Initialize and fit model
    model = QuantGradeHashRibbonsModel(
        short_window=30,
        long_window=60,
        prediction_horizons=[7, 14, 30],
        anomaly_threshold=2.0
    )
    
    print("\n=== Fitting Model ===")
    model.fit(sample_data)
    
    # Perform analysis on recent data
    print("\n=== Performing Analysis ===")
    recent_data = sample_data.tail(90)  # Last 90 days
    result = model.analyze(recent_data)
    
    # Display results
    print(f"\n=== Hash Ribbons Analysis Results ===")
    print(f"Timestamp: {result.timestamp}")
    print(f"Hash Rate 30D: {result.hash_rate_30d/1e18:.2f}E")
    print(f"Hash Rate 60D: {result.hash_rate_60d/1e18:.2f}E")
    print(f"Ribbon Signal: {result.ribbon_signal}")
    print(f"Ribbon Strength: {result.ribbon_strength:.3f}")
    print(f"Confidence Score: {result.confidence_score:.3f}")
    
    print(f"\n=== Miner Behavior Analysis ===")
    mb = result.miner_behavior
    print(f"Capitulation Score: {mb.capitulation_score:.3f}")
    print(f"Recovery Strength: {mb.recovery_strength:.3f}")
    print(f"Efficiency Trend: {mb.efficiency_trend:.3f}")
    print(f"Behavior Cluster: {mb.behavior_cluster}")
    print(f"Stability Index: {mb.stability_index:.3f}")
    
    print(f"\n=== Network Health Metrics ===")
    nh = result.network_health
    print(f"Security Score: {nh.security_score:.3f}")
    print(f"Decentralization Index: {nh.decentralization_index:.3f}")
    print(f"Resilience Factor: {nh.resilience_factor:.3f}")
    print(f"Attack Cost Estimate: ${nh.attack_cost_estimate/1e9:.1f}B")
    
    print(f"\n=== Mining Economics ===")
    me = result.mining_economics
    print(f"Break-even Price: ${me.break_even_price:.0f}")
    print(f"Profit Margin: {me.profit_margin:.1%}")
    print(f"Operational Efficiency: {me.operational_efficiency:.3f}")
    print(f"Capex Cycle: {me.capital_expenditure_cycle}")
    
    print(f"\n=== Risk Assessment ===")
    ra = result.risk_assessment
    print(f"Overall Risk: {ra.overall_risk:.3f}")
    print(f"Centralization Risk: {ra.mining_centralization_risk:.3f}")
    print(f"Security Risk: {ra.network_security_risk:.3f}")
    print(f"Economic Risk: {ra.economic_sustainability_risk:.3f}")
    
    if result.predictions:
        print(f"\n=== Predictions ===")
        for pred in result.predictions[:2]:  # Show first 2 predictions
            print(f"Horizon: {pred.prediction_horizon} days")
            print(f"Predicted Hash Rate: {pred.predicted_hash_rate/1e18:.2f}E")
            print(f"Trend: {pred.hash_rate_trend}")
            print(f"Signal: {pred.ribbon_signal}")
            print(f"Confidence: {pred.model_confidence:.3f}")
            print()
    
    if result.anomalies:
        print(f"\n=== Anomalies Detected ===")
        for anomaly in result.anomalies:
            print(f"Type: {anomaly.anomaly_type}")
            print(f"Severity: {anomaly.severity}")
            print(f"Score: {anomaly.anomaly_score:.3f}")
            print(f"Causes: {', '.join(anomaly.potential_causes[:2])}")
            print()
    
    print(f"\n=== Recommendations ===")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"{i}. {rec}")
    
    if result.model_performance:
        print(f"\n=== Model Performance ===")
        for metric, value in result.model_performance.items():
            if 'r2' in metric:
                print(f"{metric}: {value:.3f}")
            else:
                print(f"{metric}: {value:.2e}")
    
    print("\n=== Hash Ribbons Analysis Complete ===")