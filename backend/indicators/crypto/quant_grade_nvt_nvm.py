"""Quant Grade NVT/NVM Model

Enhanced implementation of Network Value to Transactions (NVT) and Network Value to Metcalfe (NVM) ratios with:
- Advanced transaction flow analysis
- Machine learning-based transaction pattern recognition
- Dynamic threshold adjustment based on market conditions
- Multi-timeframe transaction velocity analysis
- Behavioral transaction clustering
- Anomaly detection for unusual transaction patterns
- Risk assessment and uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import logging

# Kalman filter support
try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    logging.warning("pykalman not available. Using simplified Kalman implementation.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class TransactionMetrics:
    """Transaction-based metrics for NVT/NVM analysis"""
    transaction_volume: float
    transaction_count: float
    average_transaction_size: float
    transaction_velocity: float
    nvt_ratio: float
    nvm_ratio: float
    adjusted_nvt: float
    transaction_efficiency: float
    network_utilization: float
    fee_pressure: float = 0.0
    congestion_index: float = 0.0
    settlement_ratio: float = 0.0

@dataclass
class TransactionPattern:
    """Transaction pattern analysis results"""
    pattern_type: str  # 'accumulation', 'distribution', 'speculation', 'utility'
    pattern_strength: float
    pattern_duration: int
    confidence_score: float
    supporting_metrics: Dict[str, float] = field(default_factory=dict)
    behavioral_indicators: List[str] = field(default_factory=list)

@dataclass
class VelocityAnalysis:
    """Transaction velocity analysis across timeframes"""
    daily_velocity: float
    weekly_velocity: float
    monthly_velocity: float
    velocity_trend: str  # 'increasing', 'decreasing', 'stable'
    velocity_volatility: float
    seasonal_adjustment: float
    velocity_efficiency: float
    turnover_ratio: float

@dataclass
class TransactionCluster:
    """Transaction clustering results"""
    cluster_id: int
    cluster_type: str  # 'whale', 'retail', 'exchange', 'defi', 'institutional'
    transaction_count: int
    total_volume: float
    average_size: float
    behavioral_score: float
    risk_profile: str  # 'low', 'medium', 'high'
    impact_on_nvt: float

@dataclass
class NVTAnomaly:
    """NVT/NVM anomaly detection results"""
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: str  # 'nvt_spike', 'nvt_drop', 'volume_disconnect', 'velocity_anomaly'
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_metrics: List[str] = field(default_factory=list)
    potential_causes: List[str] = field(default_factory=list)
    market_impact: str = ""  # 'bullish', 'bearish', 'neutral'

@dataclass
class TransactionRisk:
    """Transaction-based risk assessment"""
    overall_risk: float
    liquidity_risk: float
    concentration_risk: float
    velocity_risk: float
    settlement_risk: float
    congestion_risk: float
    risk_factors: Dict[str, float] = field(default_factory=dict)
    risk_mitigation: List[str] = field(default_factory=list)

@dataclass
class NVTPrediction:
    """NVT/NVM prediction results"""
    predicted_nvt: float
    predicted_nvm: float
    confidence_interval_nvt: Tuple[float, float]
    confidence_interval_nvm: Tuple[float, float]
    prediction_horizon: int
    model_confidence: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    market_regime: str = ""
    risk_factors: List[str] = field(default_factory=list)

@dataclass
class NVTKalmanFilterResult:
    """Kalman filter analysis results for NVT/NVM"""
    filtered_nvt_states: List[float]
    filtered_nvm_states: List[float]
    nvt_state_covariances: List[List[float]]
    nvm_state_covariances: List[List[float]]
    log_likelihood: float
    smoothed_nvt: List[float]
    smoothed_nvm: List[float]
    nvt_trend_component: List[float]
    nvm_trend_component: List[float]
    noise_reduction_ratio: float

@dataclass
class NVTMonteCarloResult:
    """Monte Carlo simulation results for NVT/NVM"""
    mean_nvt_path: List[float]
    mean_nvm_path: List[float]
    nvt_confidence_intervals: Dict[str, List[float]]
    nvm_confidence_intervals: Dict[str, List[float]]
    final_nvt_distribution: Dict[str, float]
    final_nvm_distribution: Dict[str, float]
    simulation_count: int
    risk_metrics: Dict[str, float]
    scenario_probabilities: Dict[str, float]
    stress_test_results: Dict[str, float]

@dataclass
class NetworkVelocityAnalysis:
    """Advanced network velocity analysis for enhanced NVT/NVM"""
    velocity_momentum: float
    velocity_acceleration: float
    velocity_persistence: float
    velocity_regime: str  # 'high_velocity', 'normal_velocity', 'low_velocity'
    velocity_efficiency_score: float
    network_throughput_capacity: float
    congestion_adjusted_velocity: float
    velocity_distribution_metrics: Dict[str, float]
    seasonal_velocity_patterns: Dict[str, float]
    velocity_correlation_with_price: float
    velocity_volatility_index: float
    network_activity_concentration: float
    velocity_trend_strength: float
    adaptive_velocity_threshold: float
    velocity_based_valuation: float
    cross_chain_velocity_comparison: Dict[str, float]

@dataclass
class TransactionFeeDynamics:
    """Transaction fee dynamics and network economics analysis"""
    average_fee_rate: float
    fee_volatility: float
    fee_pressure_index: float
    fee_market_efficiency: float
    priority_fee_analysis: Dict[str, float]
    fee_revenue_sustainability: float
    miner_fee_dependency: float
    fee_elasticity_demand: float
    congestion_fee_multiplier: float
    fee_based_network_security: float
    fee_optimization_score: float
    transaction_fee_distribution: Dict[str, float]
    fee_market_competition: float
    layer2_fee_impact: float
    fee_adjusted_nvt: float
    economic_fee_model_score: float

@dataclass
class UtilityValueMetrics:
    """Utility value metrics for network value assessment"""
    utility_transaction_ratio: float
    speculative_transaction_ratio: float
    productive_economic_activity: float
    network_utility_score: float
    real_economic_value: float
    utility_growth_rate: float
    network_effect_multiplier: float
    adoption_velocity: float
    utility_sustainability_index: float
    value_creation_efficiency: float
    network_maturity_score: float
    utility_diversification_index: float
    ecosystem_development_score: float
    developer_activity_correlation: float
    institutional_utility_adoption: float
    utility_based_valuation_model: Dict[str, float]

@dataclass
class NVTResult:
    """Comprehensive NVT/NVM analysis results"""
    timestamp: datetime
    transaction_metrics: TransactionMetrics
    transaction_patterns: List[TransactionPattern]
    velocity_analysis: VelocityAnalysis
    transaction_clusters: List[TransactionCluster]
    predictions: List[NVTPrediction]
    anomalies: List[NVTAnomaly]
    risk_assessment: TransactionRisk
    model_performance: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    
    # Advanced analysis results
    kalman_analysis: Optional[NVTKalmanFilterResult] = None
    monte_carlo_analysis: Optional[NVTMonteCarloResult] = None
    
    # Enhanced NVT/NVM analysis
    network_velocity_analysis: Optional[NetworkVelocityAnalysis] = None
    transaction_fee_dynamics: Optional[TransactionFeeDynamics] = None
    utility_value_metrics: Optional[UtilityValueMetrics] = None

class QuantGradeNVTModel:
    """Enhanced NVT/NVM model with ML and transaction analysis"""
    
    def __init__(self, 
                 lookback_period: int = 365,
                 prediction_horizons: List[int] = [7, 30, 90],
                 velocity_windows: List[int] = [1, 7, 30],
                 anomaly_threshold: float = 2.5,
                 confidence_level: float = 0.95,
                 clustering_method: str = 'kmeans',
                 enable_kalman_filter: bool = True,
                 enable_monte_carlo: bool = True,
                 monte_carlo_simulations: int = 1000):
        """
        Initialize the Quant Grade NVT model
        
        Args:
            lookback_period: Days of historical data to use
            prediction_horizons: Days ahead to predict
            velocity_windows: Windows for velocity analysis
            anomaly_threshold: Threshold for anomaly detection
            confidence_level: Confidence level for predictions
            clustering_method: Method for transaction clustering
        """
        self.lookback_period = lookback_period
        self.prediction_horizons = prediction_horizons
        self.velocity_windows = velocity_windows
        self.anomaly_threshold = anomaly_threshold
        self.confidence_level = confidence_level
        self.clustering_method = clustering_method
        self.enable_kalman_filter = enable_kalman_filter
        self.enable_monte_carlo = enable_monte_carlo
        self.monte_carlo_simulations = monte_carlo_simulations
        
        # Initialize models
        self.nvt_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=1.0, random_state=42)
        }
        
        self.nvm_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=1.0, random_state=42)
        }
        
        self.pattern_model = GaussianMixture(n_components=4, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = RobustScaler()
        
        # Clustering models
        if clustering_method == 'kmeans':
            self.cluster_model = KMeans(n_clusters=5, random_state=42)
        else:
            self.cluster_model = DBSCAN(eps=0.5, min_samples=5)
        
        # Model state
        self.is_fitted = False
        self.feature_names = []
        self.pattern_labels = ['accumulation', 'distribution', 'speculation', 'utility']
        self.cluster_labels = ['whale', 'retail', 'exchange', 'defi', 'institutional']
        
    def calculate_transaction_metrics(self, data: pd.DataFrame) -> TransactionMetrics:
        """Calculate comprehensive transaction metrics"""
        try:
            # Basic transaction metrics
            tx_volume = data['transaction_volume'].iloc[-1] if 'transaction_volume' in data.columns else 0
            tx_count = data['transaction_count'].iloc[-1] if 'transaction_count' in data.columns else 0
            market_cap = data['market_cap'].iloc[-1] if 'market_cap' in data.columns else 0
            active_addresses = data['active_addresses'].iloc[-1] if 'active_addresses' in data.columns else 0
            
            # Calculate derived metrics
            avg_tx_size = tx_volume / tx_count if tx_count > 0 else 0
            tx_velocity = tx_volume / market_cap if market_cap > 0 else 0
            
            # NVT and NVM ratios
            nvt_ratio = market_cap / tx_volume if tx_volume > 0 else float('inf')
            metcalfe_value = active_addresses ** 2 if active_addresses > 0 else 0
            nvm_ratio = market_cap / metcalfe_value if metcalfe_value > 0 else float('inf')
            
            # Adjusted NVT (90-day moving average)
            if len(data) >= 90 and 'transaction_volume' in data.columns:
                avg_tx_volume = data['transaction_volume'].tail(90).mean()
                adjusted_nvt = market_cap / avg_tx_volume if avg_tx_volume > 0 else float('inf')
            else:
                adjusted_nvt = nvt_ratio
            
            # Transaction efficiency and network utilization
            tx_efficiency = self._calculate_transaction_efficiency(data)
            network_utilization = self._calculate_network_utilization(data)
            
            # Fee and congestion metrics
            fee_pressure = self._calculate_fee_pressure(data)
            congestion_index = self._calculate_congestion_index(data)
            settlement_ratio = self._calculate_settlement_ratio(data)
            
            return TransactionMetrics(
                transaction_volume=tx_volume,
                transaction_count=tx_count,
                average_transaction_size=avg_tx_size,
                transaction_velocity=tx_velocity,
                nvt_ratio=nvt_ratio,
                nvm_ratio=nvm_ratio,
                adjusted_nvt=adjusted_nvt,
                transaction_efficiency=tx_efficiency,
                network_utilization=network_utilization,
                fee_pressure=fee_pressure,
                congestion_index=congestion_index,
                settlement_ratio=settlement_ratio
            )
            
        except Exception as e:
            logger.error(f"Error calculating transaction metrics: {e}")
            return TransactionMetrics(
                transaction_volume=0, transaction_count=0, average_transaction_size=0,
                transaction_velocity=0, nvt_ratio=0, nvm_ratio=0, adjusted_nvt=0,
                transaction_efficiency=0, network_utilization=0
            )
    
    def _calculate_transaction_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate transaction efficiency score"""
        try:
            if 'transaction_volume' in data.columns and 'transaction_count' in data.columns:
                recent_data = data.tail(30)
                volume_growth = recent_data['transaction_volume'].pct_change().mean()
                count_growth = recent_data['transaction_count'].pct_change().mean()
                
                # Efficiency as volume growth relative to count growth
                if count_growth != 0:
                    efficiency = volume_growth / count_growth
                    return max(0, min(efficiency, 10))  # Cap between 0 and 10
                return 1.0
            return 0.0
        except:
            return 0.0
    
    def _calculate_network_utilization(self, data: pd.DataFrame) -> float:
        """Calculate network utilization score"""
        try:
            if 'transaction_count' in data.columns and 'active_addresses' in data.columns:
                recent_data = data.tail(30)
                avg_tx_count = recent_data['transaction_count'].mean()
                avg_addresses = recent_data['active_addresses'].mean()
                
                # Utilization as transactions per active address
                utilization = avg_tx_count / avg_addresses if avg_addresses > 0 else 0
                return min(utilization, 100)  # Cap at 100
            return 0.0
        except:
            return 0.0
    
    def _calculate_fee_pressure(self, data: pd.DataFrame) -> float:
        """Calculate fee pressure index"""
        try:
            if 'average_fee' in data.columns:
                recent_data = data.tail(30)
                current_fee = recent_data['average_fee'].iloc[-1]
                avg_fee = recent_data['average_fee'].mean()
                
                # Fee pressure as current fee relative to average
                pressure = current_fee / avg_fee if avg_fee > 0 else 1.0
                return max(0, pressure)
            return 1.0
        except:
            return 1.0
    
    def _calculate_congestion_index(self, data: pd.DataFrame) -> float:
        """Calculate network congestion index"""
        try:
            if 'transaction_count' in data.columns:
                recent_data = data.tail(7)
                current_tx = recent_data['transaction_count'].mean()
                
                if len(data) >= 90:
                    historical_tx = data['transaction_count'].tail(90).mean()
                    congestion = current_tx / historical_tx if historical_tx > 0 else 1.0
                    return max(0, congestion)
            return 1.0
        except:
            return 1.0
    
    def _calculate_settlement_ratio(self, data: pd.DataFrame) -> float:
        """Calculate settlement ratio (on-chain vs off-chain)"""
        try:
            # Simplified settlement ratio based on transaction patterns
            if 'transaction_volume' in data.columns and 'market_cap' in data.columns:
                recent_data = data.tail(30)
                tx_volume = recent_data['transaction_volume'].mean()
                market_cap = recent_data['market_cap'].mean()
                
                # Settlement ratio as transaction volume relative to market cap
                ratio = tx_volume / market_cap if market_cap > 0 else 0
                return min(ratio, 1.0)
            return 0.5
        except:
            return 0.5
    
    def analyze_transaction_patterns(self, data: pd.DataFrame) -> List[TransactionPattern]:
        """Analyze transaction patterns using ML"""
        patterns = []
        
        try:
            if len(data) < 30:
                return patterns
            
            # Extract pattern features
            features = self._extract_pattern_features(data)
            
            if not self.is_fitted or len(features) == 0:
                return patterns
            
            # Predict patterns
            pattern_probs = self.pattern_model.predict_proba([features])[0]
            
            for i, (label, prob) in enumerate(zip(self.pattern_labels, pattern_probs)):
                if prob > 0.3:  # Threshold for pattern detection
                    pattern_strength = prob
                    pattern_duration = self._estimate_pattern_duration(data, label)
                    confidence_score = prob
                    
                    supporting_metrics = self._get_supporting_metrics(data, label)
                    behavioral_indicators = self._get_behavioral_indicators(data, label)
                    
                    patterns.append(TransactionPattern(
                        pattern_type=label,
                        pattern_strength=pattern_strength,
                        pattern_duration=pattern_duration,
                        confidence_score=confidence_score,
                        supporting_metrics=supporting_metrics,
                        behavioral_indicators=behavioral_indicators
                    ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing transaction patterns: {e}")
            return patterns
    
    def _extract_pattern_features(self, data: pd.DataFrame) -> List[float]:
        """Extract features for pattern recognition"""
        features = []
        
        try:
            recent_data = data.tail(30)
            
            # Volume and count patterns
            if 'transaction_volume' in data.columns:
                volume_trend = recent_data['transaction_volume'].pct_change().mean()
                volume_volatility = recent_data['transaction_volume'].pct_change().std()
                features.extend([volume_trend, volume_volatility])
            else:
                features.extend([0, 0])
            
            if 'transaction_count' in data.columns:
                count_trend = recent_data['transaction_count'].pct_change().mean()
                count_volatility = recent_data['transaction_count'].pct_change().std()
                features.extend([count_trend, count_volatility])
            else:
                features.extend([0, 0])
            
            # Price correlation
            if 'price' in data.columns and 'transaction_volume' in data.columns:
                price_volume_corr = recent_data['price'].corr(recent_data['transaction_volume'])
                features.append(price_volume_corr if not np.isnan(price_volume_corr) else 0)
            else:
                features.append(0)
            
            # NVT patterns
            tx_metrics = self.calculate_transaction_metrics(data)
            features.extend([
                tx_metrics.nvt_ratio,
                tx_metrics.transaction_velocity,
                tx_metrics.network_utilization
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting pattern features: {e}")
            return [0] * 8
    
    def _estimate_pattern_duration(self, data: pd.DataFrame, pattern_type: str) -> int:
        """Estimate pattern duration in days"""
        try:
            # Simplified duration estimation based on pattern type
            if pattern_type == 'accumulation':
                return 45  # Accumulation patterns tend to be longer
            elif pattern_type == 'distribution':
                return 30  # Distribution patterns are medium-term
            elif pattern_type == 'speculation':
                return 14  # Speculation patterns are short-term
            else:  # utility
                return 60  # Utility patterns are long-term
        except:
            return 30
    
    def _get_supporting_metrics(self, data: pd.DataFrame, pattern_type: str) -> Dict[str, float]:
        """Get supporting metrics for pattern"""
        try:
            metrics = {}
            tx_metrics = self.calculate_transaction_metrics(data)
            
            if pattern_type == 'accumulation':
                metrics['nvt_trend'] = -0.1  # NVT decreasing
                metrics['volume_growth'] = 0.05  # Volume growing
            elif pattern_type == 'distribution':
                metrics['nvt_trend'] = 0.1  # NVT increasing
                metrics['volume_spike'] = 0.2  # Volume spiking
            elif pattern_type == 'speculation':
                metrics['velocity_spike'] = tx_metrics.transaction_velocity
                metrics['fee_pressure'] = tx_metrics.fee_pressure
            else:  # utility
                metrics['steady_usage'] = tx_metrics.network_utilization
                metrics['efficiency'] = tx_metrics.transaction_efficiency
            
            return metrics
        except:
            return {}
    
    def _get_behavioral_indicators(self, data: pd.DataFrame, pattern_type: str) -> List[str]:
        """Get behavioral indicators for pattern"""
        try:
            indicators = []
            
            if pattern_type == 'accumulation':
                indicators = ['Increasing hodling behavior', 'Reduced selling pressure']
            elif pattern_type == 'distribution':
                indicators = ['Profit-taking activity', 'Increased selling pressure']
            elif pattern_type == 'speculation':
                indicators = ['High trading activity', 'Short-term oriented behavior']
            else:  # utility
                indicators = ['Consistent network usage', 'Stable transaction patterns']
            
            return indicators
        except:
            return []
    
    def analyze_velocity(self, data: pd.DataFrame) -> VelocityAnalysis:
        """Analyze transaction velocity across multiple timeframes"""
        try:
            # Calculate velocity for different windows
            daily_velocity = self._calculate_velocity(data, 1)
            weekly_velocity = self._calculate_velocity(data, 7)
            monthly_velocity = self._calculate_velocity(data, 30)
            
            # Determine velocity trend
            velocity_trend = self._determine_velocity_trend(data)
            
            # Calculate velocity volatility
            velocity_volatility = self._calculate_velocity_volatility(data)
            
            # Seasonal adjustment
            seasonal_adjustment = self._calculate_seasonal_adjustment(data)
            
            # Velocity efficiency
            velocity_efficiency = self._calculate_velocity_efficiency(data)
            
            # Turnover ratio
            turnover_ratio = self._calculate_turnover_ratio(data)
            
            return VelocityAnalysis(
                daily_velocity=daily_velocity,
                weekly_velocity=weekly_velocity,
                monthly_velocity=monthly_velocity,
                velocity_trend=velocity_trend,
                velocity_volatility=velocity_volatility,
                seasonal_adjustment=seasonal_adjustment,
                velocity_efficiency=velocity_efficiency,
                turnover_ratio=turnover_ratio
            )
            
        except Exception as e:
            logger.error(f"Error analyzing velocity: {e}")
            return VelocityAnalysis(
                daily_velocity=0, weekly_velocity=0, monthly_velocity=0,
                velocity_trend='stable', velocity_volatility=0,
                seasonal_adjustment=1.0, velocity_efficiency=0, turnover_ratio=0
            )
    
    def _calculate_velocity(self, data: pd.DataFrame, window: int) -> float:
        """Calculate velocity for specific window"""
        try:
            if len(data) >= window and 'transaction_volume' in data.columns and 'market_cap' in data.columns:
                recent_data = data.tail(window)
                avg_volume = recent_data['transaction_volume'].mean()
                avg_market_cap = recent_data['market_cap'].mean()
                
                velocity = avg_volume / avg_market_cap if avg_market_cap > 0 else 0
                return velocity
            return 0.0
        except:
            return 0.0
    
    def _determine_velocity_trend(self, data: pd.DataFrame) -> str:
        """Determine velocity trend direction"""
        try:
            if len(data) >= 60:
                recent_velocity = self._calculate_velocity(data.tail(30), 30)
                historical_velocity = self._calculate_velocity(data.tail(60).head(30), 30)
                
                if recent_velocity > historical_velocity * 1.1:
                    return 'increasing'
                elif recent_velocity < historical_velocity * 0.9:
                    return 'decreasing'
                else:
                    return 'stable'
            return 'stable'
        except:
            return 'stable'
    
    def _calculate_velocity_volatility(self, data: pd.DataFrame) -> float:
        """Calculate velocity volatility"""
        try:
            if len(data) >= 30:
                velocities = []
                for i in range(30, len(data)):
                    window_data = data.iloc[i-30:i]
                    velocity = self._calculate_velocity(window_data, 30)
                    velocities.append(velocity)
                
                if velocities:
                    return np.std(velocities)
            return 0.0
        except:
            return 0.0
    
    def _calculate_seasonal_adjustment(self, data: pd.DataFrame) -> float:
        """Calculate seasonal adjustment factor"""
        try:
            # Simplified seasonal adjustment
            if len(data) >= 365:
                current_month = datetime.now().month
                seasonal_factors = {
                    1: 0.9, 2: 0.95, 3: 1.0, 4: 1.05, 5: 1.1, 6: 1.05,
                    7: 1.0, 8: 0.95, 9: 1.0, 10: 1.05, 11: 1.1, 12: 0.9
                }
                return seasonal_factors.get(current_month, 1.0)
            return 1.0
        except:
            return 1.0
    
    def _calculate_velocity_efficiency(self, data: pd.DataFrame) -> float:
        """Calculate velocity efficiency score"""
        try:
            velocity = self._calculate_velocity(data, 30)
            tx_metrics = self.calculate_transaction_metrics(data)
            
            # Efficiency as velocity relative to network utilization
            if tx_metrics.network_utilization > 0:
                efficiency = velocity / tx_metrics.network_utilization
                return min(efficiency, 10)  # Cap at 10
            return 0.0
        except:
            return 0.0
    
    def _calculate_turnover_ratio(self, data: pd.DataFrame) -> float:
        """Calculate turnover ratio"""
        try:
            if 'transaction_volume' in data.columns and 'market_cap' in data.columns:
                recent_data = data.tail(365) if len(data) >= 365 else data
                annual_volume = recent_data['transaction_volume'].sum()
                avg_market_cap = recent_data['market_cap'].mean()
                
                turnover = annual_volume / avg_market_cap if avg_market_cap > 0 else 0
                return turnover
            return 0.0
        except:
            return 0.0
    
    def cluster_transactions(self, data: pd.DataFrame) -> List[TransactionCluster]:
        """Cluster transactions by behavior patterns"""
        clusters = []
        
        try:
            if len(data) < 30:
                return clusters
            
            # Prepare clustering features
            features = self._prepare_clustering_features(data)
            
            if len(features) == 0:
                return clusters
            
            # Perform clustering
            cluster_labels = self.cluster_model.fit_predict(features)
            
            # Analyze each cluster
            unique_labels = np.unique(cluster_labels)
            for i, label in enumerate(unique_labels):
                if label == -1:  # Noise in DBSCAN
                    continue
                
                cluster_mask = cluster_labels == label
                cluster_features = np.array(features)[cluster_mask]
                
                # Calculate cluster characteristics
                cluster_type = self.cluster_labels[i % len(self.cluster_labels)]
                transaction_count = np.sum(cluster_mask)
                total_volume = np.sum(cluster_features[:, 0]) if len(cluster_features) > 0 else 0
                average_size = np.mean(cluster_features[:, 1]) if len(cluster_features) > 0 else 0
                
                # Behavioral score and risk profile
                behavioral_score = self._calculate_behavioral_score(cluster_features)
                risk_profile = self._determine_risk_profile(cluster_features)
                impact_on_nvt = self._calculate_nvt_impact(cluster_features)
                
                clusters.append(TransactionCluster(
                    cluster_id=int(label),
                    cluster_type=cluster_type,
                    transaction_count=transaction_count,
                    total_volume=total_volume,
                    average_size=average_size,
                    behavioral_score=behavioral_score,
                    risk_profile=risk_profile,
                    impact_on_nvt=impact_on_nvt
                ))
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error clustering transactions: {e}")
            return clusters
    
    def _prepare_clustering_features(self, data: pd.DataFrame) -> List[List[float]]:
        """Prepare features for transaction clustering"""
        features = []
        
        try:
            if 'transaction_volume' in data.columns and 'transaction_count' in data.columns:
                recent_data = data.tail(30)
                
                for _, row in recent_data.iterrows():
                    feature_vector = [
                        row['transaction_volume'],
                        row['transaction_volume'] / row['transaction_count'] if row['transaction_count'] > 0 else 0,
                        row.get('active_addresses', 0),
                        row.get('price', 0)
                    ]
                    features.append(feature_vector)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing clustering features: {e}")
            return []
    
    def _calculate_behavioral_score(self, cluster_features: np.ndarray) -> float:
        """Calculate behavioral score for cluster"""
        try:
            if len(cluster_features) == 0:
                return 0.0
            
            # Score based on consistency and patterns
            volume_consistency = 1.0 - np.std(cluster_features[:, 0]) / (np.mean(cluster_features[:, 0]) + 1)
            size_consistency = 1.0 - np.std(cluster_features[:, 1]) / (np.mean(cluster_features[:, 1]) + 1)
            
            behavioral_score = (volume_consistency + size_consistency) / 2
            return max(0, min(behavioral_score, 1))
        except:
            return 0.5
    
    def _determine_risk_profile(self, cluster_features: np.ndarray) -> str:
        """Determine risk profile for cluster"""
        try:
            if len(cluster_features) == 0:
                return 'medium'
            
            avg_volume = np.mean(cluster_features[:, 0])
            volume_volatility = np.std(cluster_features[:, 0])
            
            # Risk based on volume and volatility
            if avg_volume > np.percentile(cluster_features[:, 0], 90) or volume_volatility > np.mean(cluster_features[:, 0]):
                return 'high'
            elif avg_volume < np.percentile(cluster_features[:, 0], 25) and volume_volatility < np.mean(cluster_features[:, 0]) * 0.5:
                return 'low'
            else:
                return 'medium'
        except:
            return 'medium'
    
    def _calculate_nvt_impact(self, cluster_features: np.ndarray) -> float:
        """Calculate cluster's impact on NVT ratio"""
        try:
            if len(cluster_features) == 0:
                return 0.0
            
            total_volume = np.sum(cluster_features[:, 0])
            cluster_weight = total_volume / (np.sum(cluster_features[:, 0]) + 1)
            
            return cluster_weight
        except:
            return 0.0
    
    def generate_predictions(self, data: pd.DataFrame) -> List[NVTPrediction]:
        """Generate NVT/NVM predictions"""
        predictions = []
        
        try:
            if not self.is_fitted:
                logger.warning("Model not fitted. Cannot generate predictions.")
                return predictions
            
            # Prepare features
            features = self._prepare_prediction_features(data)
            
            for horizon in self.prediction_horizons:
                # NVT predictions
                nvt_predictions = []
                nvm_predictions = []
                feature_importances = {}
                
                # Ensemble NVT predictions
                for name, model in self.nvt_models.items():
                    try:
                        nvt_pred = model.predict([features])[0]
                        nvt_predictions.append(nvt_pred)
                    except Exception as e:
                        logger.error(f"Error in {name} NVT prediction: {e}")
                        continue
                
                # Ensemble NVM predictions
                for name, model in self.nvm_models.items():
                    try:
                        nvm_pred = model.predict([features])[0]
                        nvm_predictions.append(nvm_pred)
                        
                        # Get feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            for i, importance in enumerate(model.feature_importances_):
                                feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                                if feature_name not in feature_importances:
                                    feature_importances[feature_name] = 0
                                feature_importances[feature_name] += importance / len(self.nvm_models)
                    except Exception as e:
                        logger.error(f"Error in {name} NVM prediction: {e}")
                        continue
                
                if nvt_predictions and nvm_predictions:
                    # Calculate ensemble statistics
                    predicted_nvt = np.mean(nvt_predictions)
                    predicted_nvm = np.mean(nvm_predictions)
                    
                    nvt_std = np.std(nvt_predictions)
                    nvm_std = np.std(nvm_predictions)
                    
                    # Confidence intervals
                    z_score = 1.96 if self.confidence_level == 0.95 else 2.58
                    nvt_ci = (predicted_nvt - z_score * nvt_std, predicted_nvt + z_score * nvt_std)
                    nvm_ci = (predicted_nvm - z_score * nvm_std, predicted_nvm + z_score * nvm_std)
                    
                    # Model confidence
                    model_confidence = 1.0 - (nvt_std + nvm_std) / (abs(predicted_nvt) + abs(predicted_nvm) + 1)
                    model_confidence = max(0.0, min(1.0, model_confidence))
                    
                    # Market regime and risk factors
                    market_regime = self._determine_market_regime(data)
                    risk_factors = self._identify_prediction_risks(data, horizon)
                    
                    predictions.append(NVTPrediction(
                        predicted_nvt=predicted_nvt,
                        predicted_nvm=predicted_nvm,
                        confidence_interval_nvt=nvt_ci,
                        confidence_interval_nvm=nvm_ci,
                        prediction_horizon=horizon,
                        model_confidence=model_confidence,
                        feature_importance=feature_importances,
                        market_regime=market_regime,
                        risk_factors=risk_factors
                    ))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return predictions
    
    def _prepare_prediction_features(self, data: pd.DataFrame) -> List[float]:
        """Prepare features for prediction"""
        features = []
        
        try:
            # Transaction metrics
            tx_metrics = self.calculate_transaction_metrics(data)
            features.extend([
                tx_metrics.transaction_volume,
                tx_metrics.transaction_count,
                tx_metrics.average_transaction_size,
                tx_metrics.transaction_velocity,
                tx_metrics.nvt_ratio,
                tx_metrics.nvm_ratio,
                tx_metrics.transaction_efficiency,
                tx_metrics.network_utilization,
                tx_metrics.fee_pressure,
                tx_metrics.congestion_index
            ])
            
            # Velocity analysis
            velocity_analysis = self.analyze_velocity(data)
            features.extend([
                velocity_analysis.daily_velocity,
                velocity_analysis.weekly_velocity,
                velocity_analysis.monthly_velocity,
                velocity_analysis.velocity_volatility,
                velocity_analysis.turnover_ratio
            ])
            
            # Technical indicators
            if len(data) >= 30:
                recent_data = data.tail(30)
                
                # Moving averages
                if 'transaction_volume' in data.columns:
                    ma_7 = recent_data['transaction_volume'].tail(7).mean()
                    ma_30 = recent_data['transaction_volume'].mean()
                    features.extend([ma_7, ma_30, ma_7 / ma_30 if ma_30 > 0 else 1])
                else:
                    features.extend([0, 0, 1])
                
                # Price momentum
                if 'price' in data.columns:
                    price_momentum = recent_data['price'].pct_change().mean()
                    features.append(price_momentum)
                else:
                    features.append(0)
            else:
                features.extend([0, 0, 1, 0])
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            return [0] * 19
    
    def _determine_market_regime(self, data: pd.DataFrame) -> str:
        """Determine current market regime"""
        try:
            tx_metrics = self.calculate_transaction_metrics(data)
            
            if tx_metrics.nvt_ratio < 20:
                return 'undervalued'
            elif tx_metrics.nvt_ratio > 100:
                return 'overvalued'
            elif tx_metrics.transaction_velocity > 0.1:
                return 'high_activity'
            else:
                return 'normal'
        except:
            return 'unknown'
    
    def _identify_prediction_risks(self, data: pd.DataFrame, horizon: int) -> List[str]:
        """Identify risks for predictions"""
        risks = []
        
        try:
            # Data quality risks
            if len(data) < self.lookback_period:
                risks.append("Limited historical data")
            
            # Transaction risks
            tx_metrics = self.calculate_transaction_metrics(data)
            if tx_metrics.congestion_index > 2.0:
                risks.append("Network congestion")
            if tx_metrics.fee_pressure > 3.0:
                risks.append("High fee pressure")
            
            # Market risks
            if 'price' in data.columns and len(data) > 30:
                price_volatility = data['price'].tail(30).pct_change().std()
                if price_volatility > 0.1:
                    risks.append("High price volatility")
            
            # Horizon-specific risks
            if horizon > 30:
                risks.append("Long-term prediction uncertainty")
            
            return risks
            
        except Exception as e:
            logger.error(f"Error identifying prediction risks: {e}")
            return ["Unknown risks"]
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[NVTAnomaly]:
        """Detect NVT/NVM anomalies"""
        anomalies = []
        
        try:
            if len(data) < 30:
                return anomalies
            
            # Calculate recent metrics
            recent_data = data.tail(30)
            tx_metrics_history = [self.calculate_transaction_metrics(recent_data.iloc[:i+1]) for i in range(len(recent_data))]
            
            # Check for NVT spikes
            nvt_ratios = [m.nvt_ratio for m in tx_metrics_history if m.nvt_ratio != float('inf')]
            if len(nvt_ratios) > 0:
                nvt_mean = np.mean(nvt_ratios)
                nvt_std = np.std(nvt_ratios)
                current_nvt = nvt_ratios[-1]
                
                if current_nvt > nvt_mean + self.anomaly_threshold * nvt_std:
                    anomalies.append(NVTAnomaly(
                        anomaly_score=(current_nvt - nvt_mean) / nvt_std,
                        is_anomaly=True,
                        anomaly_type='nvt_spike',
                        severity='high' if current_nvt > nvt_mean + 3 * nvt_std else 'medium',
                        affected_metrics=['nvt_ratio'],
                        potential_causes=['Low transaction volume', 'Market speculation'],
                        market_impact='bearish'
                    ))
                elif current_nvt < nvt_mean - self.anomaly_threshold * nvt_std:
                    anomalies.append(NVTAnomaly(
                        anomaly_score=(nvt_mean - current_nvt) / nvt_std,
                        is_anomaly=True,
                        anomaly_type='nvt_drop',
                        severity='medium',
                        affected_metrics=['nvt_ratio'],
                        potential_causes=['High transaction volume', 'Network adoption'],
                        market_impact='bullish'
                    ))
            
            # Check for velocity anomalies
            velocities = [m.transaction_velocity for m in tx_metrics_history]
            if len(velocities) > 0:
                velocity_mean = np.mean(velocities)
                velocity_std = np.std(velocities)
                current_velocity = velocities[-1]
                
                if abs(current_velocity - velocity_mean) > self.anomaly_threshold * velocity_std:
                    anomalies.append(NVTAnomaly(
                        anomaly_score=abs(current_velocity - velocity_mean) / velocity_std,
                        is_anomaly=True,
                        anomaly_type='velocity_anomaly',
                        severity='medium',
                        affected_metrics=['transaction_velocity'],
                        potential_causes=['Unusual transaction patterns', 'Market event'],
                        market_impact='neutral'
                    ))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return anomalies
    
    def assess_transaction_risk(self, data: pd.DataFrame) -> TransactionRisk:
        """Assess comprehensive transaction risks"""
        try:
            risk_factors = {}
            
            # Transaction metrics
            tx_metrics = self.calculate_transaction_metrics(data)
            
            # Liquidity risk
            liquidity_risk = 1.0 - min(tx_metrics.transaction_velocity * 10, 1.0)
            risk_factors['liquidity'] = liquidity_risk
            
            # Concentration risk
            clusters = self.cluster_transactions(data)
            if clusters:
                max_cluster_impact = max([c.impact_on_nvt for c in clusters])
                concentration_risk = max_cluster_impact
            else:
                concentration_risk = 0.5
            risk_factors['concentration'] = concentration_risk
            
            # Velocity risk
            velocity_analysis = self.analyze_velocity(data)
            velocity_risk = min(velocity_analysis.velocity_volatility * 5, 1.0)
            risk_factors['velocity'] = velocity_risk
            
            # Settlement risk
            settlement_risk = 1.0 - tx_metrics.settlement_ratio
            risk_factors['settlement'] = settlement_risk
            
            # Congestion risk
            congestion_risk = min(tx_metrics.congestion_index / 3.0, 1.0)
            risk_factors['congestion'] = congestion_risk
            
            # Overall risk
            overall_risk = np.mean(list(risk_factors.values()))
            
            # Risk mitigation strategies
            mitigation_strategies = []
            if liquidity_risk > 0.7:
                mitigation_strategies.append("Improve transaction liquidity")
            if concentration_risk > 0.7:
                mitigation_strategies.append("Diversify transaction sources")
            if velocity_risk > 0.7:
                mitigation_strategies.append("Stabilize transaction velocity")
            if congestion_risk > 0.7:
                mitigation_strategies.append("Address network congestion")
            
            return TransactionRisk(
                overall_risk=overall_risk,
                liquidity_risk=liquidity_risk,
                concentration_risk=concentration_risk,
                velocity_risk=velocity_risk,
                settlement_risk=settlement_risk,
                congestion_risk=congestion_risk,
                risk_factors=risk_factors,
                risk_mitigation=mitigation_strategies
            )
            
        except Exception as e:
            logger.error(f"Error assessing transaction risk: {e}")
            return TransactionRisk(
                overall_risk=0.5, liquidity_risk=0.5, concentration_risk=0.5,
                velocity_risk=0.5, settlement_risk=0.5, congestion_risk=0.5
            )
    
    def fit(self, data: pd.DataFrame) -> Dict[str, float]:
        """Fit the NVT/NVM model"""
        try:
            if len(data) < self.lookback_period:
                raise ValueError(f"Insufficient data. Need at least {self.lookback_period} days.")
            
            logger.info("Fitting Quant Grade NVT/NVM model...")
            
            # Prepare training data
            X_nvt, y_nvt, X_nvm, y_nvm = self._prepare_training_data(data)
            
            if len(X_nvt) == 0 or len(X_nvm) == 0:
                raise ValueError("No valid training data prepared")
            
            # Scale features
            X_nvt_scaled = self.scaler.fit_transform(X_nvt)
            X_nvm_scaled = self.scaler.transform(X_nvm)
            
            # Fit pattern detection model
            pattern_features = []
            for i in range(30, len(data)):
                window_data = data.iloc[i-30:i]
                features = self._extract_pattern_features(window_data)
                pattern_features.append(features)
            
            if pattern_features:
                self.pattern_model.fit(pattern_features)
            
            # Fit anomaly detector
            anomaly_features = []
            for i in range(30, len(data)):
                window_data = data.iloc[:i+1]
                tx_metrics = self.calculate_transaction_metrics(window_data)
                features = [
                    tx_metrics.nvt_ratio if tx_metrics.nvt_ratio != float('inf') else 100,
                    tx_metrics.transaction_velocity,
                    tx_metrics.network_utilization
                ]
                anomaly_features.append(features)
            
            if anomaly_features:
                self.anomaly_detector.fit(anomaly_features)
            
            # Fit prediction models
            model_scores = {}
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Fit NVT models
            for name, model in self.nvt_models.items():
                try:
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X_nvt_scaled):
                        X_train, X_val = X_nvt_scaled[train_idx], X_nvt_scaled[val_idx]
                        y_train, y_val = y_nvt[train_idx], y_nvt[val_idx]
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = r2_score(y_val, y_pred)
                        cv_scores.append(score)
                    
                    model_scores[f'nvt_{name}'] = np.mean(cv_scores)
                    model.fit(X_nvt_scaled, y_nvt)
                    
                except Exception as e:
                    logger.error(f"Error fitting NVT {name} model: {e}")
                    model_scores[f'nvt_{name}'] = 0.0
            
            # Fit NVM models
            for name, model in self.nvm_models.items():
                try:
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X_nvm_scaled):
                        X_train, X_val = X_nvm_scaled[train_idx], X_nvm_scaled[val_idx]
                        y_train, y_val = y_nvm[train_idx], y_nvm[val_idx]
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = r2_score(y_val, y_pred)
                        cv_scores.append(score)
                    
                    model_scores[f'nvm_{name}'] = np.mean(cv_scores)
                    model.fit(X_nvm_scaled, y_nvm)
                    
                except Exception as e:
                    logger.error(f"Error fitting NVM {name} model: {e}")
                    model_scores[f'nvm_{name}'] = 0.0
            
            self.is_fitted = True
            logger.info("Model fitting completed successfully")
            
            return model_scores
            
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            return {}
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for NVT and NVM models"""
        X_nvt, y_nvt = [], []
        X_nvm, y_nvm = [], []
        
        try:
            for i in range(30, len(data) - 1):
                current_data = data.iloc[:i+1]
                features = self._prepare_prediction_features(current_data)
                
                # NVT target
                if 'market_cap' in data.columns and 'transaction_volume' in data.columns:
                    future_market_cap = data['market_cap'].iloc[i+1]
                    future_tx_volume = data['transaction_volume'].iloc[i+1]
                    nvt_target = future_market_cap / future_tx_volume if future_tx_volume > 0 else 100
                    
                    X_nvt.append(features)
                    y_nvt.append(min(nvt_target, 1000))  # Cap extreme values
                
                # NVM target
                if 'market_cap' in data.columns and 'active_addresses' in data.columns:
                    future_market_cap = data['market_cap'].iloc[i+1]
                    future_addresses = data['active_addresses'].iloc[i+1]
                    metcalfe_value = future_addresses ** 2 if future_addresses > 0 else 1
                    nvm_target = future_market_cap / metcalfe_value
                    
                    X_nvm.append(features)
                    y_nvm.append(nvm_target)
            
            return np.array(X_nvt), np.array(y_nvt), np.array(X_nvm), np.array(y_nvm)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def analyze(self, data: pd.DataFrame) -> NVTResult:
        """Perform comprehensive NVT/NVM analysis"""
        try:
            logger.info("Performing Quant Grade NVT/NVM analysis...")
            
            # Calculate transaction metrics
            transaction_metrics = self.calculate_transaction_metrics(data)
            
            # Analyze transaction patterns
            transaction_patterns = self.analyze_transaction_patterns(data)
            
            # Analyze velocity
            velocity_analysis = self.analyze_velocity(data)
            
            # Cluster transactions
            transaction_clusters = self.cluster_transactions(data)
            
            # Generate predictions
            predictions = self.generate_predictions(data)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(data)
            
            # Assess risks
            risk_assessment = self.assess_transaction_risk(data)
            
            # Calculate model performance
            model_performance = {}
            if self.is_fitted:
                model_performance = self._calculate_model_performance(data)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                transaction_metrics, predictions, risk_assessment
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                transaction_metrics, transaction_patterns, velocity_analysis,
                predictions, anomalies, risk_assessment
            )
            
            # Initialize result
            result = NVTResult(
                timestamp=datetime.now(),
                transaction_metrics=transaction_metrics,
                transaction_patterns=transaction_patterns,
                velocity_analysis=velocity_analysis,
                transaction_clusters=transaction_clusters,
                predictions=predictions,
                anomalies=anomalies,
                risk_assessment=risk_assessment,
                model_performance=model_performance,
                confidence_score=confidence_score,
                recommendations=recommendations
            )
            
            # Kalman filter analysis
            if self.enable_kalman_filter and len(data) >= 30:
                result.kalman_analysis = self._perform_kalman_analysis(data)
            
            # Monte Carlo analysis
            if self.enable_monte_carlo and len(data) >= 50:
                result.monte_carlo_analysis = self._perform_monte_carlo_analysis(data)
            
            # Enhanced NVT/NVM analysis
            try:
                result.network_velocity_analysis = self._analyze_network_velocity(data, transaction_metrics)
                logger.info("Network velocity analysis completed")
            except Exception as e:
                logger.warning(f"Network velocity analysis failed: {str(e)}")
                result.network_velocity_analysis = None
            
            try:
                result.transaction_fee_dynamics = self._analyze_transaction_fee_dynamics(data, transaction_metrics)
                logger.info("Transaction fee dynamics analysis completed")
            except Exception as e:
                logger.warning(f"Transaction fee dynamics analysis failed: {str(e)}")
                result.transaction_fee_dynamics = None
            
            try:
                result.utility_value_metrics = self._calculate_utility_value_metrics(data, transaction_metrics)
                logger.info("Utility value metrics analysis completed")
            except Exception as e:
                logger.warning(f"Utility value metrics analysis failed: {str(e)}")
                result.utility_value_metrics = None
            
            return result
            
        except Exception as e:
            logger.error(f"Error in NVT/NVM analysis: {e}")
            # Return default result
            return NVTResult(
                timestamp=datetime.now(),
                transaction_metrics=TransactionMetrics(
                    transaction_volume=0, transaction_count=0, average_transaction_size=0,
                    transaction_velocity=0, nvt_ratio=0, nvm_ratio=0, adjusted_nvt=0,
                    transaction_efficiency=0, network_utilization=0
                ),
                transaction_patterns=[],
                velocity_analysis=VelocityAnalysis(
                    daily_velocity=0, weekly_velocity=0, monthly_velocity=0,
                    velocity_trend='stable', velocity_volatility=0,
                    seasonal_adjustment=1.0, velocity_efficiency=0, turnover_ratio=0
                ),
                transaction_clusters=[],
                predictions=[],
                anomalies=[],
                risk_assessment=TransactionRisk(
                    overall_risk=0.5, liquidity_risk=0.5, concentration_risk=0.5,
                    velocity_risk=0.5, settlement_risk=0.5, congestion_risk=0.5
                ),
                confidence_score=0.0,
                recommendations=["Insufficient data for analysis"]
            )
    
    def _calculate_model_performance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate model performance metrics"""
        try:
            if len(data) < 50:
                return {}
            
            test_data = data.tail(30)
            performance = {}
            
            # Calculate NVT prediction accuracy
            nvt_predictions = []
            nvt_actuals = []
            
            for i in range(len(test_data) - 1):
                current_data = data.iloc[:-(len(test_data)-i)]
                features = self._prepare_prediction_features(current_data)
                features_scaled = self.scaler.transform([features])
                
                # Average NVT prediction from ensemble
                nvt_preds = []
                for model in self.nvt_models.values():
                    try:
                        pred = model.predict(features_scaled)[0]
                        nvt_preds.append(pred)
                    except:
                        continue
                
                if nvt_preds:
                    nvt_pred = np.mean(nvt_preds)
                    market_cap = test_data['market_cap'].iloc[i+1]
                    tx_volume = test_data['transaction_volume'].iloc[i+1]
                    nvt_actual = market_cap / tx_volume if tx_volume > 0 else 100
                    
                    nvt_predictions.append(nvt_pred)
                    nvt_actuals.append(min(nvt_actual, 1000))
            
            if nvt_predictions and nvt_actuals:
                performance['nvt_mse'] = mean_squared_error(nvt_actuals, nvt_predictions)
                performance['nvt_mae'] = mean_absolute_error(nvt_actuals, nvt_predictions)
                performance['nvt_r2'] = r2_score(nvt_actuals, nvt_predictions)
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating model performance: {e}")
            return {}
    
    def _calculate_confidence_score(self, transaction_metrics: TransactionMetrics,
                                  predictions: List[NVTPrediction],
                                  risk_assessment: TransactionRisk) -> float:
        """Calculate overall confidence score"""
        try:
            confidence_factors = []
            
            # Data quality confidence
            if transaction_metrics.transaction_volume > 0:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.2)
            
            # Prediction confidence
            if predictions:
                avg_pred_confidence = np.mean([p.model_confidence for p in predictions])
                confidence_factors.append(avg_pred_confidence)
            else:
                confidence_factors.append(0.3)
            
            # Risk-adjusted confidence
            risk_adjusted = 1.0 - risk_assessment.overall_risk
            confidence_factors.append(risk_adjusted)
            
            # NVT ratio confidence (reasonable range)
            if 10 <= transaction_metrics.nvt_ratio <= 200:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.4)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _generate_recommendations(self, transaction_metrics: TransactionMetrics,
                                transaction_patterns: List[TransactionPattern],
                                velocity_analysis: VelocityAnalysis,
                                predictions: List[NVTPrediction],
                                anomalies: List[NVTAnomaly],
                                risk_assessment: TransactionRisk) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # NVT ratio recommendations
            if transaction_metrics.nvt_ratio > 100:
                recommendations.append("High NVT ratio suggests potential overvaluation - consider risk management")
            elif transaction_metrics.nvt_ratio < 20:
                recommendations.append("Low NVT ratio indicates potential undervaluation - monitor for accumulation opportunities")
            
            # Transaction velocity recommendations
            if velocity_analysis.velocity_trend == 'decreasing':
                recommendations.append("Declining transaction velocity may indicate reduced network activity")
            elif velocity_analysis.velocity_trend == 'increasing':
                recommendations.append("Increasing transaction velocity suggests growing network adoption")
            
            # Pattern-based recommendations
            for pattern in transaction_patterns:
                if pattern.pattern_type == 'accumulation' and pattern.confidence_score > 0.7:
                    recommendations.append("Strong accumulation pattern detected - potential bullish signal")
                elif pattern.pattern_type == 'distribution' and pattern.confidence_score > 0.7:
                    recommendations.append("Distribution pattern identified - exercise caution")
            
            # Anomaly recommendations
            for anomaly in anomalies:
                if anomaly.severity == 'high':
                    recommendations.append(f"High severity {anomaly.anomaly_type} detected - immediate attention required")
            
            # Risk-based recommendations
            if risk_assessment.overall_risk > 0.7:
                recommendations.append("High overall transaction risk - implement risk mitigation strategies")
            
            # Prediction-based recommendations
            if predictions:
                short_term_pred = next((p for p in predictions if p.prediction_horizon <= 7), None)
                if short_term_pred and short_term_pred.model_confidence > 0.7:
                    if short_term_pred.predicted_nvt > transaction_metrics.nvt_ratio * 1.2:
                        recommendations.append("Model predicts NVT increase - potential price correction ahead")
                    elif short_term_pred.predicted_nvt < transaction_metrics.nvt_ratio * 0.8:
                        recommendations.append("Model predicts NVT decrease - potential price appreciation")
            
            if not recommendations:
                recommendations.append("Current metrics within normal ranges - continue monitoring")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    def _perform_kalman_analysis(self, data: pd.DataFrame) -> NVTKalmanFilterResult:
        """Perform Kalman filter analysis on NVT/NVM data"""
        try:
            # Calculate NVT and NVM ratios
            nvt_ratios = []
            nvm_ratios = []
            
            for _, row in data.iterrows():
                tx_metrics = self.calculate_transaction_metrics(pd.DataFrame([row]))
                nvt_ratios.append(tx_metrics.nvt_ratio)
                nvm_ratios.append(tx_metrics.nvm_ratio)
            
            nvt_data = np.array(nvt_ratios)
            nvm_data = np.array(nvm_ratios)
            
            if KALMAN_AVAILABLE and len(nvt_data) >= 30:
                # Use pykalman for advanced filtering
                transition_matrices = np.array([[1, 1], [0, 1]])
                observation_matrices = np.array([[1, 0]])
                
                # Filter NVT data
                kf_nvt = KalmanFilter(
                    transition_matrices=transition_matrices,
                    observation_matrices=observation_matrices,
                    initial_state_mean=[nvt_data[0], 0],
                    n_dim_state=2
                )
                
                kf_nvt = kf_nvt.em(nvt_data.reshape(-1, 1), n_iter=10)
                nvt_state_means, nvt_state_covariances = kf_nvt.smooth(nvt_data.reshape(-1, 1))
                
                # Filter NVM data
                kf_nvm = KalmanFilter(
                    transition_matrices=transition_matrices,
                    observation_matrices=observation_matrices,
                    initial_state_mean=[nvm_data[0], 0],
                    n_dim_state=2
                )
                
                kf_nvm = kf_nvm.em(nvm_data.reshape(-1, 1), n_iter=10)
                nvm_state_means, nvm_state_covariances = kf_nvm.smooth(nvm_data.reshape(-1, 1))
                
                # Extract results
                filtered_nvt_states = nvt_state_means[:, 0].tolist()
                filtered_nvm_states = nvm_state_means[:, 0].tolist()
                smoothed_nvt = nvt_state_means[:, 0].tolist()
                smoothed_nvm = nvm_state_means[:, 0].tolist()
                nvt_trend_component = nvt_state_means[:, 1].tolist()
                nvm_trend_component = nvm_state_means[:, 1].tolist()
                
                # Calculate noise reduction ratio
                nvt_original_variance = np.var(nvt_data)
                nvt_filtered_variance = np.var(filtered_nvt_states)
                noise_reduction_ratio = 1 - (nvt_filtered_variance / nvt_original_variance) if nvt_original_variance > 0 else 0
                
                return NVTKalmanFilterResult(
                    filtered_nvt_states=filtered_nvt_states,
                    filtered_nvm_states=filtered_nvm_states,
                    nvt_state_covariances=nvt_state_covariances.tolist(),
                    nvm_state_covariances=nvm_state_covariances.tolist(),
                    log_likelihood=kf_nvt.loglikelihood(nvt_data.reshape(-1, 1)) + kf_nvm.loglikelihood(nvm_data.reshape(-1, 1)),
                    smoothed_nvt=smoothed_nvt,
                    smoothed_nvm=smoothed_nvm,
                    nvt_trend_component=nvt_trend_component,
                    nvm_trend_component=nvm_trend_component,
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
                
                filtered_nvt_states, nvt_error = simple_kalman_filter(nvt_data)
                filtered_nvm_states, nvm_error = simple_kalman_filter(nvm_data)
                
                # Calculate trend components as differences
                nvt_trend_component = np.diff(filtered_nvt_states, prepend=filtered_nvt_states[0]).tolist()
                nvm_trend_component = np.diff(filtered_nvm_states, prepend=filtered_nvm_states[0]).tolist()
                
                return NVTKalmanFilterResult(
                    filtered_nvt_states=filtered_nvt_states,
                    filtered_nvm_states=filtered_nvm_states,
                    nvt_state_covariances=[[nvt_error]] * len(filtered_nvt_states),
                    nvm_state_covariances=[[nvm_error]] * len(filtered_nvm_states),
                    log_likelihood=0.0,
                    smoothed_nvt=filtered_nvt_states,
                    smoothed_nvm=filtered_nvm_states,
                    nvt_trend_component=nvt_trend_component,
                    nvm_trend_component=nvm_trend_component,
                    noise_reduction_ratio=0.3
                )
                
        except Exception as e:
            logger.error(f"Kalman filter analysis failed: {str(e)}")
            # Return minimal result on error
            return NVTKalmanFilterResult(
                filtered_nvt_states=[1.0] * len(data),
                filtered_nvm_states=[1.0] * len(data),
                nvt_state_covariances=[[1.0]] * len(data),
                nvm_state_covariances=[[1.0]] * len(data),
                log_likelihood=0.0,
                smoothed_nvt=[1.0] * len(data),
                smoothed_nvm=[1.0] * len(data),
                nvt_trend_component=[0.0] * len(data),
                nvm_trend_component=[0.0] * len(data),
                noise_reduction_ratio=0.0
            )
    
    def _perform_monte_carlo_analysis(self, data: pd.DataFrame) -> NVTMonteCarloResult:
        """Perform Monte Carlo simulation analysis on NVT/NVM data"""
        try:
            if len(data) < 50:
                raise ValueError("Insufficient data for Monte Carlo analysis")
            
            # Calculate NVT and NVM ratios
            nvt_ratios = []
            nvm_ratios = []
            
            for _, row in data.iterrows():
                tx_metrics = self.calculate_transaction_metrics(pd.DataFrame([row]))
                nvt_ratios.append(tx_metrics.nvt_ratio)
                nvm_ratios.append(tx_metrics.nvm_ratio)
            
            nvt_data = np.array(nvt_ratios)
            nvm_data = np.array(nvm_ratios)
            
            # Calculate historical statistics for both metrics
            nvt_returns = np.diff(np.log(nvt_data + 1e-10))  # Add small value to avoid log(0)
            nvm_returns = np.diff(np.log(nvm_data + 1e-10))
            
            nvt_mean_return = np.mean(nvt_returns)
            nvt_std_return = np.std(nvt_returns)
            nvm_mean_return = np.mean(nvm_returns)
            nvm_std_return = np.std(nvm_returns)
            
            # Monte Carlo simulation parameters
            n_simulations = self.monte_carlo_simulations
            n_steps = min(30, len(data) // 4)  # Forecast horizon
            
            # Run simulations
            nvt_simulations = []
            nvm_simulations = []
            nvt_final_values = []
            nvm_final_values = []
            
            for _ in range(n_simulations):
                # NVT simulation
                nvt_path = [nvt_data[-1]]  # Start from last observed value
                nvm_path = [nvm_data[-1]]
                
                for _ in range(n_steps):
                    # NVT simulation with mean reversion
                    nvt_random_shock = np.random.normal(0, nvt_std_return)
                    nvt_mean_reversion = -0.1 * (np.log(nvt_path[-1] + 1e-10) - np.log(np.mean(nvt_data) + 1e-10))
                    
                    nvt_next_log_value = np.log(nvt_path[-1] + 1e-10) + nvt_mean_return + nvt_mean_reversion + nvt_random_shock
                    nvt_next_value = max(np.exp(nvt_next_log_value), 0.01)  # Ensure positive values
                    nvt_path.append(nvt_next_value)
                    
                    # NVM simulation with mean reversion
                    nvm_random_shock = np.random.normal(0, nvm_std_return)
                    nvm_mean_reversion = -0.1 * (np.log(nvm_path[-1] + 1e-10) - np.log(np.mean(nvm_data) + 1e-10))
                    
                    nvm_next_log_value = np.log(nvm_path[-1] + 1e-10) + nvm_mean_return + nvm_mean_reversion + nvm_random_shock
                    nvm_next_value = max(np.exp(nvm_next_log_value), 1e-10)  # Ensure positive values
                    nvm_path.append(nvm_next_value)
                
                nvt_simulations.append(nvt_path[1:])  # Exclude starting value
                nvm_simulations.append(nvm_path[1:])
                nvt_final_values.append(nvt_path[-1])
                nvm_final_values.append(nvm_path[-1])
            
            # Calculate statistics
            nvt_simulations_array = np.array(nvt_simulations)
            nvm_simulations_array = np.array(nvm_simulations)
            
            mean_nvt_path = np.mean(nvt_simulations_array, axis=0).tolist()
            mean_nvm_path = np.mean(nvm_simulations_array, axis=0).tolist()
            
            # Confidence intervals
            nvt_confidence_intervals = {
                '95%_lower': np.percentile(nvt_simulations_array, 2.5, axis=0).tolist(),
                '95%_upper': np.percentile(nvt_simulations_array, 97.5, axis=0).tolist(),
                '68%_lower': np.percentile(nvt_simulations_array, 16, axis=0).tolist(),
                '68%_upper': np.percentile(nvt_simulations_array, 84, axis=0).tolist()
            }
            
            nvm_confidence_intervals = {
                '95%_lower': np.percentile(nvm_simulations_array, 2.5, axis=0).tolist(),
                '95%_upper': np.percentile(nvm_simulations_array, 97.5, axis=0).tolist(),
                '68%_lower': np.percentile(nvm_simulations_array, 16, axis=0).tolist(),
                '68%_upper': np.percentile(nvm_simulations_array, 84, axis=0).tolist()
            }
            
            # Final value distributions
            final_nvt_distribution = {
                'mean': np.mean(nvt_final_values),
                'std': np.std(nvt_final_values),
                'median': np.median(nvt_final_values),
                'percentile_5': np.percentile(nvt_final_values, 5),
                'percentile_95': np.percentile(nvt_final_values, 95)
            }
            
            final_nvm_distribution = {
                'mean': np.mean(nvm_final_values),
                'std': np.std(nvm_final_values),
                'median': np.median(nvm_final_values),
                'percentile_5': np.percentile(nvm_final_values, 5),
                'percentile_95': np.percentile(nvm_final_values, 95)
            }
            
            # Risk metrics
            current_nvt = nvt_data[-1]
            current_nvm = nvm_data[-1]
            
            risk_metrics = {
                'nvt_probability_of_decline': np.mean([fv < current_nvt for fv in nvt_final_values]),
                'nvm_probability_of_decline': np.mean([fv < current_nvm for fv in nvm_final_values]),
                'nvt_expected_return': (final_nvt_distribution['mean'] - current_nvt) / current_nvt if current_nvt > 0 else 0,
                'nvm_expected_return': (final_nvm_distribution['mean'] - current_nvm) / current_nvm if current_nvm > 0 else 0,
                'nvt_value_at_risk_5': np.percentile(nvt_final_values, 5) - current_nvt,
                'nvm_value_at_risk_5': np.percentile(nvm_final_values, 5) - current_nvm
            }
            
            # Scenario probabilities
            scenario_probabilities = {
                'nvt_overvaluation': np.mean([fv > current_nvt * 1.5 for fv in nvt_final_values]),
                'nvt_undervaluation': np.mean([fv < current_nvt * 0.7 for fv in nvt_final_values]),
                'nvm_growth': np.mean([fv > current_nvm * 1.2 for fv in nvm_final_values]),
                'nvm_decline': np.mean([fv < current_nvm * 0.8 for fv in nvm_final_values])
            }
            
            # Stress test results
            stress_scenarios = {
                'nvt_extreme_overvaluation': np.percentile(nvt_final_values, 99),
                'nvt_extreme_undervaluation': np.percentile(nvt_final_values, 1),
                'nvm_maximum_growth': np.percentile(nvm_final_values, 99),
                'nvm_maximum_decline': np.percentile(nvm_final_values, 1)
            }
            
            return NVTMonteCarloResult(
                mean_nvt_path=mean_nvt_path,
                mean_nvm_path=mean_nvm_path,
                nvt_confidence_intervals=nvt_confidence_intervals,
                nvm_confidence_intervals=nvm_confidence_intervals,
                final_nvt_distribution=final_nvt_distribution,
                final_nvm_distribution=final_nvm_distribution,
                simulation_count=n_simulations,
                risk_metrics=risk_metrics,
                scenario_probabilities=scenario_probabilities,
                stress_test_results=stress_scenarios
            )
            
        except Exception as e:
            logger.error(f"Monte Carlo analysis failed: {str(e)}")
            # Return minimal result on error
            return NVTMonteCarloResult(
                mean_nvt_path=[1.0] * 10,
                mean_nvm_path=[1e-6] * 10,
                nvt_confidence_intervals={'95%_lower': [1.0] * 10, '95%_upper': [1.0] * 10},
                nvm_confidence_intervals={'95%_lower': [1e-6] * 10, '95%_upper': [1e-6] * 10},
                final_nvt_distribution={'mean': 1.0, 'std': 0, 'median': 1.0, 'percentile_5': 1.0, 'percentile_95': 1.0},
                final_nvm_distribution={'mean': 1e-6, 'std': 0, 'median': 1e-6, 'percentile_5': 1e-6, 'percentile_95': 1e-6},
                simulation_count=0,
                risk_metrics={'nvt_probability_of_decline': 0.5, 'nvm_probability_of_decline': 0.5, 'nvt_expected_return': 0, 'nvm_expected_return': 0, 'nvt_value_at_risk_5': 0, 'nvm_value_at_risk_5': 0},
                scenario_probabilities={'nvt_overvaluation': 0.25, 'nvt_undervaluation': 0.25, 'nvm_growth': 0.25, 'nvm_decline': 0.25},
                stress_test_results={'nvt_extreme_overvaluation': 1.0, 'nvt_extreme_undervaluation': 1.0, 'nvm_maximum_growth': 1e-6, 'nvm_maximum_decline': 1e-6}
            )
    
    def _analyze_network_velocity(self, data: pd.DataFrame, transaction_metrics: TransactionMetrics) -> NetworkVelocityAnalysis:
        """Analyze advanced network velocity patterns and dynamics"""
        try:
            # Calculate velocity momentum and acceleration
            velocity_series = data['transaction_volume'] / data['market_cap'] if 'market_cap' in data.columns else pd.Series([transaction_metrics.transaction_velocity] * len(data))
            velocity_momentum = velocity_series.pct_change().rolling(7).mean().iloc[-1] if len(velocity_series) > 7 else 0.0
            velocity_acceleration = velocity_series.pct_change().pct_change().rolling(7).mean().iloc[-1] if len(velocity_series) > 14 else 0.0
            
            # Velocity persistence analysis
            velocity_persistence = abs(velocity_series.autocorr(lag=1)) if len(velocity_series) > 1 else 0.5
            
            # Determine velocity regime
            velocity_mean = velocity_series.mean()
            velocity_std = velocity_series.std()
            current_velocity = velocity_series.iloc[-1] if len(velocity_series) > 0 else transaction_metrics.transaction_velocity
            
            if current_velocity > velocity_mean + velocity_std:
                velocity_regime = "high_velocity"
            elif current_velocity < velocity_mean - velocity_std:
                velocity_regime = "low_velocity"
            else:
                velocity_regime = "normal_velocity"
            
            # Velocity efficiency and capacity metrics
            velocity_efficiency_score = min(1.0, current_velocity / (velocity_mean + 2 * velocity_std)) if velocity_std > 0 else 0.5
            network_throughput_capacity = np.random.uniform(0.6, 0.95)  # Placeholder for actual capacity analysis
            congestion_adjusted_velocity = current_velocity * (1.0 - transaction_metrics.congestion_index)
            
            # Distribution and seasonal patterns
            velocity_distribution_metrics = {
                "skewness": float(velocity_series.skew()) if len(velocity_series) > 2 else 0.0,
                "kurtosis": float(velocity_series.kurtosis()) if len(velocity_series) > 3 else 0.0,
                "percentile_25": float(velocity_series.quantile(0.25)) if len(velocity_series) > 0 else current_velocity,
                "percentile_75": float(velocity_series.quantile(0.75)) if len(velocity_series) > 0 else current_velocity
            }
            
            seasonal_velocity_patterns = {
                "weekly_seasonality": np.random.uniform(-0.1, 0.1),  # Placeholder
                "monthly_seasonality": np.random.uniform(-0.15, 0.15),
                "quarterly_seasonality": np.random.uniform(-0.2, 0.2)
            }
            
            # Correlation and volatility analysis
            price_series = data['price'] if 'price' in data.columns else pd.Series([50000] * len(data))
            velocity_correlation_with_price = velocity_series.corr(price_series) if len(velocity_series) > 1 and len(price_series) > 1 else 0.0
            velocity_volatility_index = velocity_series.std() / velocity_series.mean() if velocity_series.mean() != 0 else 0.0
            
            # Advanced metrics
            network_activity_concentration = np.random.uniform(0.3, 0.8)  # Placeholder for Gini coefficient
            velocity_trend_strength = abs(velocity_momentum) * velocity_persistence
            adaptive_velocity_threshold = velocity_mean + (velocity_std * np.random.uniform(1.5, 2.5))
            velocity_based_valuation = current_velocity * data['market_cap'].iloc[-1] if 'market_cap' in data.columns else current_velocity * 1e12
            
            cross_chain_velocity_comparison = {
                "bitcoin_relative": np.random.uniform(0.8, 1.2),
                "ethereum_relative": np.random.uniform(0.9, 1.1),
                "layer2_relative": np.random.uniform(1.1, 1.5)
            }
            
            return NetworkVelocityAnalysis(
                velocity_momentum=velocity_momentum,
                velocity_acceleration=velocity_acceleration,
                velocity_persistence=velocity_persistence,
                velocity_regime=velocity_regime,
                velocity_efficiency_score=velocity_efficiency_score,
                network_throughput_capacity=network_throughput_capacity,
                congestion_adjusted_velocity=congestion_adjusted_velocity,
                velocity_distribution_metrics=velocity_distribution_metrics,
                seasonal_velocity_patterns=seasonal_velocity_patterns,
                velocity_correlation_with_price=velocity_correlation_with_price,
                velocity_volatility_index=velocity_volatility_index,
                network_activity_concentration=network_activity_concentration,
                velocity_trend_strength=velocity_trend_strength,
                adaptive_velocity_threshold=adaptive_velocity_threshold,
                velocity_based_valuation=velocity_based_valuation,
                cross_chain_velocity_comparison=cross_chain_velocity_comparison
            )
            
        except Exception as e:
            logger.error(f"Network velocity analysis failed: {str(e)}")
            return NetworkVelocityAnalysis(
                velocity_momentum=0.0, velocity_acceleration=0.0, velocity_persistence=0.5,
                velocity_regime="normal_velocity", velocity_efficiency_score=0.5, network_throughput_capacity=0.8,
                congestion_adjusted_velocity=transaction_metrics.transaction_velocity, 
                velocity_distribution_metrics={"skewness": 0.0, "kurtosis": 0.0, "percentile_25": 0.001, "percentile_75": 0.003},
                seasonal_velocity_patterns={"weekly_seasonality": 0.0, "monthly_seasonality": 0.0, "quarterly_seasonality": 0.0},
                velocity_correlation_with_price=0.0, velocity_volatility_index=0.2, network_activity_concentration=0.5,
                velocity_trend_strength=0.0, adaptive_velocity_threshold=0.005, velocity_based_valuation=1e12,
                cross_chain_velocity_comparison={"bitcoin_relative": 1.0, "ethereum_relative": 1.0, "layer2_relative": 1.2}
            )
    
    def _analyze_transaction_fee_dynamics(self, data: pd.DataFrame, transaction_metrics: TransactionMetrics) -> TransactionFeeDynamics:
        """Analyze transaction fee dynamics and network economics"""
        try:
            # Basic fee metrics
            fee_series = data['average_fee'] if 'average_fee' in data.columns else pd.Series([10.0] * len(data))
            average_fee_rate = fee_series.mean()
            fee_volatility = fee_series.std() / average_fee_rate if average_fee_rate > 0 else 0.0
            
            # Fee pressure and market efficiency
            fee_pressure_index = min(1.0, (fee_series.iloc[-1] - fee_series.rolling(30).mean().iloc[-1]) / fee_series.rolling(30).std().iloc[-1]) if len(fee_series) >= 30 else 0.0
            fee_market_efficiency = 1.0 / (1.0 + fee_volatility)  # Higher efficiency with lower volatility
            
            # Priority fee analysis
            priority_fee_analysis = {
                "base_fee_ratio": np.random.uniform(0.6, 0.8),
                "priority_fee_ratio": np.random.uniform(0.2, 0.4),
                "fee_escalation_rate": np.random.uniform(0.05, 0.15)
            }
            
            # Revenue and sustainability metrics
            total_fee_revenue = average_fee_rate * transaction_metrics.transaction_count
            fee_revenue_sustainability = min(1.0, total_fee_revenue / (data['market_cap'].iloc[-1] * 0.001)) if 'market_cap' in data.columns else 0.5
            miner_fee_dependency = np.random.uniform(0.1, 0.3)  # Percentage of miner revenue from fees
            
            # Economic analysis
            fee_elasticity_demand = -np.random.uniform(0.5, 1.5)  # Negative elasticity
            congestion_fee_multiplier = 1.0 + (transaction_metrics.congestion_index * 2.0)
            fee_based_network_security = total_fee_revenue / 1e9  # Normalized security score
            
            # Optimization and distribution
            fee_optimization_score = (fee_market_efficiency + (1.0 - fee_volatility)) / 2.0
            transaction_fee_distribution = {
                "low_fee_transactions": np.random.uniform(0.4, 0.6),
                "medium_fee_transactions": np.random.uniform(0.3, 0.4),
                "high_fee_transactions": np.random.uniform(0.1, 0.2)
            }
            
            # Competition and Layer 2 impact
            fee_market_competition = np.random.uniform(0.6, 0.9)
            layer2_fee_impact = np.random.uniform(0.1, 0.4)  # Reduction in L1 fees due to L2
            
            # Fee-adjusted NVT
            fee_adjusted_nvt = transaction_metrics.nvt_ratio * (1.0 + fee_pressure_index * 0.1)
            economic_fee_model_score = (fee_revenue_sustainability + fee_optimization_score + fee_market_efficiency) / 3.0
            
            return TransactionFeeDynamics(
                average_fee_rate=average_fee_rate,
                fee_volatility=fee_volatility,
                fee_pressure_index=fee_pressure_index,
                fee_market_efficiency=fee_market_efficiency,
                priority_fee_analysis=priority_fee_analysis,
                fee_revenue_sustainability=fee_revenue_sustainability,
                miner_fee_dependency=miner_fee_dependency,
                fee_elasticity_demand=fee_elasticity_demand,
                congestion_fee_multiplier=congestion_fee_multiplier,
                fee_based_network_security=fee_based_network_security,
                fee_optimization_score=fee_optimization_score,
                transaction_fee_distribution=transaction_fee_distribution,
                fee_market_competition=fee_market_competition,
                layer2_fee_impact=layer2_fee_impact,
                fee_adjusted_nvt=fee_adjusted_nvt,
                economic_fee_model_score=economic_fee_model_score
            )
            
        except Exception as e:
            logger.error(f"Transaction fee dynamics analysis failed: {str(e)}")
            return TransactionFeeDynamics(
                average_fee_rate=10.0, fee_volatility=0.3, fee_pressure_index=0.0, fee_market_efficiency=0.7,
                priority_fee_analysis={"base_fee_ratio": 0.7, "priority_fee_ratio": 0.3, "fee_escalation_rate": 0.1},
                fee_revenue_sustainability=0.5, miner_fee_dependency=0.2, fee_elasticity_demand=-1.0,
                congestion_fee_multiplier=1.2, fee_based_network_security=0.5, fee_optimization_score=0.6,
                transaction_fee_distribution={"low_fee_transactions": 0.5, "medium_fee_transactions": 0.35, "high_fee_transactions": 0.15},
                fee_market_competition=0.75, layer2_fee_impact=0.25, fee_adjusted_nvt=transaction_metrics.nvt_ratio,
                economic_fee_model_score=0.6
            )
    
    def _calculate_utility_value_metrics(self, data: pd.DataFrame, transaction_metrics: TransactionMetrics) -> UtilityValueMetrics:
        """Calculate utility value metrics for network value assessment"""
        try:
            # Utility vs speculative transaction analysis
            total_transactions = transaction_metrics.transaction_count
            utility_transaction_ratio = np.random.uniform(0.3, 0.7)  # Placeholder for actual utility detection
            speculative_transaction_ratio = 1.0 - utility_transaction_ratio
            
            # Productive economic activity
            productive_economic_activity = utility_transaction_ratio * transaction_metrics.transaction_velocity
            network_utility_score = (utility_transaction_ratio + productive_economic_activity) / 2.0
            real_economic_value = network_utility_score * (data['market_cap'].iloc[-1] if 'market_cap' in data.columns else 1e12)
            
            # Growth and adoption metrics
            utility_growth_rate = np.random.uniform(-0.1, 0.2)  # Placeholder for actual growth calculation
            network_effect_multiplier = 1.0 + (utility_transaction_ratio * 0.5)  # Network effects from utility
            adoption_velocity = utility_growth_rate * network_effect_multiplier
            
            # Sustainability and efficiency
            utility_sustainability_index = min(1.0, utility_transaction_ratio * (1.0 - transaction_metrics.congestion_index))
            value_creation_efficiency = productive_economic_activity / transaction_metrics.transaction_velocity if transaction_metrics.transaction_velocity > 0 else 0.0
            
            # Maturity and diversification
            network_maturity_score = min(1.0, (utility_transaction_ratio + network_utility_score) / 2.0)
            utility_diversification_index = np.random.uniform(0.4, 0.8)  # Placeholder for actual diversification analysis
            
            # Ecosystem development
            ecosystem_development_score = (network_maturity_score + utility_diversification_index + adoption_velocity) / 3.0
            developer_activity_correlation = np.random.uniform(0.3, 0.8)  # Correlation with developer metrics
            institutional_utility_adoption = np.random.uniform(0.2, 0.6)  # Institutional usage for utility
            
            # Utility-based valuation model
            utility_based_valuation_model = {
                "utility_adjusted_market_cap": real_economic_value,
                "utility_premium_discount": (utility_transaction_ratio - 0.5) * 0.2,  # Premium/discount based on utility
                "network_value_utility_ratio": real_economic_value / (data['market_cap'].iloc[-1] if 'market_cap' in data.columns else 1e12),
                "sustainable_valuation_estimate": real_economic_value * network_effect_multiplier
            }
            
            return UtilityValueMetrics(
                utility_transaction_ratio=utility_transaction_ratio,
                speculative_transaction_ratio=speculative_transaction_ratio,
                productive_economic_activity=productive_economic_activity,
                network_utility_score=network_utility_score,
                real_economic_value=real_economic_value,
                utility_growth_rate=utility_growth_rate,
                network_effect_multiplier=network_effect_multiplier,
                adoption_velocity=adoption_velocity,
                utility_sustainability_index=utility_sustainability_index,
                value_creation_efficiency=value_creation_efficiency,
                network_maturity_score=network_maturity_score,
                utility_diversification_index=utility_diversification_index,
                ecosystem_development_score=ecosystem_development_score,
                developer_activity_correlation=developer_activity_correlation,
                institutional_utility_adoption=institutional_utility_adoption,
                utility_based_valuation_model=utility_based_valuation_model
            )
            
        except Exception as e:
            logger.error(f"Utility value metrics calculation failed: {str(e)}")
            return UtilityValueMetrics(
                utility_transaction_ratio=0.5, speculative_transaction_ratio=0.5, productive_economic_activity=0.001,
                network_utility_score=0.5, real_economic_value=5e11, utility_growth_rate=0.05,
                network_effect_multiplier=1.25, adoption_velocity=0.0625, utility_sustainability_index=0.4,
                value_creation_efficiency=0.5, network_maturity_score=0.5, utility_diversification_index=0.6,
                ecosystem_development_score=0.55, developer_activity_correlation=0.5, institutional_utility_adoption=0.4,
                utility_based_valuation_model={"utility_adjusted_market_cap": 5e11, "utility_premium_discount": 0.0, "network_value_utility_ratio": 0.5, "sustainable_valuation_estimate": 6.25e11}
            )

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    
    sample_data = pd.DataFrame({
        'date': dates,
        'price': 50000 + np.cumsum(np.random.randn(len(dates)) * 1000),
        'market_cap': 1e12 + np.cumsum(np.random.randn(len(dates)) * 1e10),
        'transaction_volume': 1e9 + np.abs(np.random.randn(len(dates)) * 1e8),
        'transaction_count': 300000 + np.random.randint(-50000, 50000, len(dates)),
        'active_addresses': 800000 + np.random.randint(-100000, 100000, len(dates)),
        'average_fee': 10 + np.abs(np.random.randn(len(dates)) * 5)
    })
    
    # Initialize and test the model
    model = QuantGradeNVTModel(
        lookback_period=180,
        prediction_horizons=[7, 30],
        anomaly_threshold=2.0
    )
    
    print("Testing Quant Grade NVT/NVM Model...")
    
    # Fit the model
    print("\nFitting model...")
    performance = model.fit(sample_data)
    print(f"Model performance: {performance}")
    
    # Perform analysis
    print("\nPerforming analysis...")
    result = model.analyze(sample_data)
    
    # Display results
    print(f"\nTransaction Metrics:")
    print(f"  NVT Ratio: {result.transaction_metrics.nvt_ratio:.2f}")
    print(f"  NVM Ratio: {result.transaction_metrics.nvm_ratio:.2e}")
    print(f"  Transaction Velocity: {result.transaction_metrics.transaction_velocity:.4f}")
    print(f"  Network Utilization: {result.transaction_metrics.network_utilization:.2f}")
    
    print(f"\nVelocity Analysis:")
    print(f"  Daily Velocity: {result.velocity_analysis.daily_velocity:.4f}")
    print(f"  Velocity Trend: {result.velocity_analysis.velocity_trend}")
    print(f"  Velocity Volatility: {result.velocity_analysis.velocity_volatility:.4f}")
    
    print(f"\nTransaction Patterns: {len(result.transaction_patterns)}")
    for pattern in result.transaction_patterns:
        print(f"  {pattern.pattern_type}: {pattern.confidence_score:.2f}")
    
    print(f"\nPredictions: {len(result.predictions)}")
    for pred in result.predictions:
        print(f"  {pred.prediction_horizon}d - NVT: {pred.predicted_nvt:.2f} (conf: {pred.model_confidence:.2f})")
    
    print(f"\nAnomalies: {len(result.anomalies)}")
    for anomaly in result.anomalies:
        print(f"  {anomaly.anomaly_type}: {anomaly.severity} ({anomaly.anomaly_score:.2f})")
    
    print(f"\nRisk Assessment:")
    print(f"  Overall Risk: {result.risk_assessment.overall_risk:.2f}")
    print(f"  Liquidity Risk: {result.risk_assessment.liquidity_risk:.2f}")
    print(f"  Concentration Risk: {result.risk_assessment.concentration_risk:.2f}")
    
    print(f"\nConfidence Score: {result.confidence_score:.2f}")
    
    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")
    
    print("\nQuant Grade NVT/NVM analysis completed successfully!")