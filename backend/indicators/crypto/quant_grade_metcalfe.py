"""Quant Grade Metcalfe's Law Model

Enhanced implementation of Metcalfe's Law for cryptocurrency valuation with:
- Network effect modeling with multiple growth patterns
- Machine learning-based network value predictions
- Dynamic network coefficient estimation
- Multi-dimensional network analysis (users, transactions, addresses)
- Regime detection for different network growth phases
- Anomaly detection for network disruptions
- Risk assessment and uncertainty quantification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Kalman filter support
try:
    from pykalman import KalmanFilter
    KALMAN_AVAILABLE = True
except ImportError:
    KALMAN_AVAILABLE = False
    logger.warning("pykalman not available. Using simplified Kalman filter implementation.")

@dataclass
class NetworkMetrics:
    """Network-based metrics for Metcalfe's Law analysis"""
    active_addresses: float
    transaction_count: float
    network_value: float
    metcalfe_ratio: float
    network_density: float
    clustering_coefficient: float
    path_length: float
    centrality_measures: Dict[str, float] = field(default_factory=dict)
    growth_rate: float = 0.0
    velocity: float = 0.0

@dataclass
class NetworkRegime:
    """Network growth regime analysis"""
    regime_type: str  # 'exponential', 'linear', 'logarithmic', 'decline'
    growth_coefficient: float
    network_effect_strength: float
    regime_probability: float
    duration_days: int
    stability_score: float
    transition_probability: Dict[str, float] = field(default_factory=dict)

@dataclass
class NetworkPrediction:
    """Network value prediction results"""
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: int
    model_confidence: float
    feature_importance: Dict[str, float] = field(default_factory=dict)
    regime_forecast: str = ""
    risk_factors: List[str] = field(default_factory=list)

@dataclass
class NetworkAnomaly:
    """Network anomaly detection results"""
    anomaly_score: float
    is_anomaly: bool
    anomaly_type: str  # 'growth_spike', 'network_disruption', 'value_disconnect'
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_metrics: List[str] = field(default_factory=list)
    potential_causes: List[str] = field(default_factory=list)

@dataclass
class NetworkRisk:
    """Network risk assessment"""
    overall_risk: float
    network_concentration_risk: float
    growth_sustainability_risk: float
    adoption_risk: float
    technical_risk: float
    market_risk: float
    risk_factors: Dict[str, float] = field(default_factory=dict)
    mitigation_strategies: List[str] = field(default_factory=list)

@dataclass
class MetcalfeKalmanFilterResult:
    """Kalman filter analysis results for Metcalfe's Law"""
    filtered_network_states: List[float]
    filtered_value_states: List[float]
    network_state_covariances: List[List[float]]
    value_state_covariances: List[List[float]]
    log_likelihood: float
    smoothed_network_growth: List[float]
    smoothed_value_trajectory: List[float]
    network_trend_component: List[float]
    value_trend_component: List[float]
    noise_reduction_ratio: float

@dataclass
class MetcalfeMonteCarloResult:
    """Monte Carlo simulation results for Metcalfe's Law"""
    mean_network_growth_path: List[float]
    mean_value_path: List[float]
    network_confidence_intervals: Dict[str, List[float]]
    value_confidence_intervals: Dict[str, List[float]]
    final_network_distribution: Dict[str, float]
    final_value_distribution: Dict[str, float]
    simulation_count: int
    risk_metrics: Dict[str, float]
    scenario_probabilities: Dict[str, float]
    stress_test_results: Dict[str, float]

@dataclass
class OdlyzkoCorrection:
    """Odlyzko's n*log(n) correction to Metcalfe's Law"""
    corrected_network_value: float
    correction_factor: float
    log_correction_ratio: float
    efficiency_improvement: float
    theoretical_vs_actual: float
    network_maturity_score: float
    scaling_coefficient: float
    
@dataclass
class NetworkEffectsDecay:
    """Network effects decay analysis"""
    decay_rate: float
    half_life_days: float
    current_decay_factor: float
    network_saturation_level: float
    marginal_utility_decline: float
    congestion_effects: float
    network_efficiency_score: float
    
@dataclass
class ZipfLawAnalysis:
    """Zipf's law analysis for address distribution"""
    zipf_exponent: float
    distribution_fitness: float
    concentration_index: float
    whale_dominance_ratio: float
    address_inequality_gini: float
    power_law_deviation: float
    network_decentralization_score: float

@dataclass
class MetcalfeResult:
    """Comprehensive Metcalfe's Law analysis results"""
    timestamp: datetime
    network_metrics: NetworkMetrics
    current_regime: NetworkRegime
    predictions: List[NetworkPrediction]
    anomalies: List[NetworkAnomaly]
    risk_assessment: NetworkRisk
    model_performance: Dict[str, float] = field(default_factory=dict)
    confidence_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    kalman_analysis: Optional['MetcalfeKalmanFilterResult'] = None
    monte_carlo_analysis: Optional['MetcalfeMonteCarloResult'] = None
    # Enhanced Metcalfe's Law analysis
    odlyzko_correction: Optional['OdlyzkoCorrection'] = None
    network_effects_decay: Optional['NetworkEffectsDecay'] = None
    zipf_law_analysis: Optional['ZipfLawAnalysis'] = None

class QuantGradeMetcalfeModel:
    """Enhanced Metcalfe's Law model with ML and network analysis"""
    
    def __init__(self, 
                 lookback_period: int = 365,
                 prediction_horizons: List[int] = [7, 30, 90],
                 regime_detection_window: int = 90,
                 anomaly_threshold: float = 2.5,
                 confidence_level: float = 0.95,
                 enable_kalman_filter: bool = True,
                 enable_monte_carlo: bool = True,
                 monte_carlo_simulations: int = 1000):
        """
        Initialize the Quant Grade Metcalfe model
        
        Args:
            lookback_period: Days of historical data to use
            prediction_horizons: Days ahead to predict
            regime_detection_window: Window for regime analysis
            anomaly_threshold: Threshold for anomaly detection
            confidence_level: Confidence level for predictions
        """
        self.lookback_period = lookback_period
        self.prediction_horizons = prediction_horizons
        self.regime_detection_window = regime_detection_window
        self.anomaly_threshold = anomaly_threshold
        self.confidence_level = confidence_level
        self.enable_kalman_filter = enable_kalman_filter
        self.enable_monte_carlo = enable_monte_carlo
        self.monte_carlo_simulations = monte_carlo_simulations
        
        # Initialize models
        self.network_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'elastic': ElasticNet(alpha=1.0, random_state=42)
        }
        
        self.regime_model = GaussianMixture(n_components=4, random_state=42)
        self.scaler = RobustScaler()
        
        # Model state
        self.is_fitted = False
        self.feature_names = []
        self.regime_labels = ['exponential', 'linear', 'logarithmic', 'decline']
        
    def calculate_network_metrics(self, data: pd.DataFrame) -> NetworkMetrics:
        """Calculate comprehensive network metrics"""
        try:
            # Basic network metrics
            active_addresses = data['active_addresses'].iloc[-1] if 'active_addresses' in data.columns else 0
            transaction_count = data['transaction_count'].iloc[-1] if 'transaction_count' in data.columns else 0
            market_cap = data['market_cap'].iloc[-1] if 'market_cap' in data.columns else 0
            
            # Calculate Metcalfe's Law value (N^2 relationship)
            metcalfe_value = active_addresses ** 2 if active_addresses > 0 else 0
            metcalfe_ratio = market_cap / metcalfe_value if metcalfe_value > 0 else 0
            
            # Network density and clustering
            network_density = self._calculate_network_density(data)
            clustering_coefficient = self._calculate_clustering_coefficient(data)
            path_length = self._calculate_average_path_length(data)
            
            # Growth and velocity metrics
            growth_rate = self._calculate_network_growth_rate(data)
            velocity = self._calculate_network_velocity(data)
            
            # Centrality measures
            centrality_measures = self._calculate_centrality_measures(data)
            
            return NetworkMetrics(
                active_addresses=active_addresses,
                transaction_count=transaction_count,
                network_value=metcalfe_value,
                metcalfe_ratio=metcalfe_ratio,
                network_density=network_density,
                clustering_coefficient=clustering_coefficient,
                path_length=path_length,
                centrality_measures=centrality_measures,
                growth_rate=growth_rate,
                velocity=velocity
            )
            
        except Exception as e:
            logger.error(f"Error calculating network metrics: {e}")
            return NetworkMetrics(
                active_addresses=0, transaction_count=0, network_value=0,
                metcalfe_ratio=0, network_density=0, clustering_coefficient=0,
                path_length=0
            )
    
    def _calculate_network_density(self, data: pd.DataFrame) -> float:
        """Calculate network density based on transaction patterns"""
        try:
            if 'transaction_count' in data.columns and 'active_addresses' in data.columns:
                recent_data = data.tail(30)
                avg_transactions = recent_data['transaction_count'].mean()
                avg_addresses = recent_data['active_addresses'].mean()
                
                # Network density as transactions per address pair
                max_connections = avg_addresses * (avg_addresses - 1) / 2
                density = avg_transactions / max_connections if max_connections > 0 else 0
                return min(density, 1.0)  # Cap at 1.0
            return 0.0
        except:
            return 0.0
    
    def _calculate_clustering_coefficient(self, data: pd.DataFrame) -> float:
        """Estimate clustering coefficient from transaction data"""
        try:
            # Simplified clustering based on transaction patterns
            if 'transaction_count' in data.columns:
                recent_data = data.tail(30)
                tx_variance = recent_data['transaction_count'].var()
                tx_mean = recent_data['transaction_count'].mean()
                
                # Higher variance suggests more clustering
                clustering = tx_variance / (tx_mean + 1) if tx_mean > 0 else 0
                return min(clustering / 1000, 1.0)  # Normalize
            return 0.0
        except:
            return 0.0
    
    def _calculate_average_path_length(self, data: pd.DataFrame) -> float:
        """Estimate average path length in the network"""
        try:
            if 'active_addresses' in data.columns:
                addresses = data['active_addresses'].iloc[-1]
                # Theoretical path length for scale-free networks
                path_length = np.log(addresses) / np.log(np.log(addresses)) if addresses > 2 else 1
                return max(path_length, 1.0)
            return 1.0
        except:
            return 1.0
    
    def _calculate_network_growth_rate(self, data: pd.DataFrame) -> float:
        """Calculate network growth rate"""
        try:
            if 'active_addresses' in data.columns and len(data) > 30:
                current = data['active_addresses'].tail(7).mean()
                previous = data['active_addresses'].tail(37).head(7).mean()
                growth_rate = (current - previous) / previous if previous > 0 else 0
                return growth_rate
            return 0.0
        except:
            return 0.0
    
    def _calculate_network_velocity(self, data: pd.DataFrame) -> float:
        """Calculate network velocity (transaction turnover)"""
        try:
            if 'transaction_count' in data.columns and 'market_cap' in data.columns:
                recent_data = data.tail(30)
                avg_transactions = recent_data['transaction_count'].mean()
                avg_market_cap = recent_data['market_cap'].mean()
                velocity = avg_transactions / avg_market_cap if avg_market_cap > 0 else 0
                return velocity
            return 0.0
        except:
            return 0.0
    
    def _calculate_centrality_measures(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various centrality measures"""
        try:
            centrality = {}
            
            if 'active_addresses' in data.columns:
                addresses = data['active_addresses'].iloc[-1]
                
                # Degree centrality (simplified)
                if 'transaction_count' in data.columns:
                    tx_count = data['transaction_count'].iloc[-1]
                    degree_centrality = tx_count / (addresses * (addresses - 1)) if addresses > 1 else 0
                    centrality['degree'] = min(degree_centrality, 1.0)
                
                # Betweenness centrality (estimated)
                centrality['betweenness'] = 1.0 / np.log(addresses) if addresses > 1 else 0
                
                # Closeness centrality (estimated)
                centrality['closeness'] = 1.0 / self._calculate_average_path_length(data)
                
            return centrality
        except:
            return {}
    
    def detect_network_regime(self, data: pd.DataFrame) -> NetworkRegime:
        """Detect current network growth regime"""
        try:
            if len(data) < self.regime_detection_window:
                return NetworkRegime(
                    regime_type='unknown', growth_coefficient=0,
                    network_effect_strength=0, regime_probability=0,
                    duration_days=0, stability_score=0
                )
            
            # Prepare regime features
            window_data = data.tail(self.regime_detection_window)
            features = self._extract_regime_features(window_data)
            
            if not self.is_fitted:
                return NetworkRegime(
                    regime_type='unknown', growth_coefficient=0,
                    network_effect_strength=0, regime_probability=0,
                    duration_days=0, stability_score=0
                )
            
            # Predict regime
            regime_probs = self.regime_model.predict_proba([features])[0]
            regime_idx = np.argmax(regime_probs)
            regime_type = self.regime_labels[regime_idx]
            
            # Calculate regime characteristics
            growth_coefficient = self._calculate_growth_coefficient(window_data, regime_type)
            network_effect_strength = self._calculate_network_effect_strength(window_data)
            stability_score = self._calculate_regime_stability(window_data)
            
            # Transition probabilities
            transition_probs = {label: prob for label, prob in zip(self.regime_labels, regime_probs)}
            
            return NetworkRegime(
                regime_type=regime_type,
                growth_coefficient=growth_coefficient,
                network_effect_strength=network_effect_strength,
                regime_probability=regime_probs[regime_idx],
                duration_days=self.regime_detection_window,
                stability_score=stability_score,
                transition_probability=transition_probs
            )
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return NetworkRegime(
                regime_type='unknown', growth_coefficient=0,
                network_effect_strength=0, regime_probability=0,
                duration_days=0, stability_score=0
            )
    
    def _extract_regime_features(self, data: pd.DataFrame) -> List[float]:
        """Extract features for regime detection"""
        features = []
        
        try:
            # Network growth features
            if 'active_addresses' in data.columns:
                addresses = data['active_addresses'].values
                growth_rates = np.diff(addresses) / addresses[:-1]
                features.extend([
                    np.mean(growth_rates),
                    np.std(growth_rates),
                    np.percentile(growth_rates, 75) - np.percentile(growth_rates, 25)
                ])
            else:
                features.extend([0, 0, 0])
            
            # Transaction features
            if 'transaction_count' in data.columns:
                tx_counts = data['transaction_count'].values
                tx_growth = np.diff(tx_counts) / tx_counts[:-1]
                features.extend([
                    np.mean(tx_growth),
                    np.std(tx_growth),
                    np.corrcoef(addresses[1:], tx_counts[1:])[0, 1] if len(addresses) > 1 else 0
                ])
            else:
                features.extend([0, 0, 0])
            
            # Market features
            if 'market_cap' in data.columns:
                market_caps = data['market_cap'].values
                market_growth = np.diff(market_caps) / market_caps[:-1]
                features.extend([
                    np.mean(market_growth),
                    np.std(market_growth)
                ])
            else:
                features.extend([0, 0])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting regime features: {e}")
            return [0] * 8
    
    def _calculate_growth_coefficient(self, data: pd.DataFrame, regime_type: str) -> float:
        """Calculate growth coefficient for the regime"""
        try:
            if 'active_addresses' in data.columns and len(data) > 1:
                addresses = data['active_addresses'].values
                time_points = np.arange(len(addresses))
                
                if regime_type == 'exponential':
                    # Fit exponential growth
                    log_addresses = np.log(addresses + 1)
                    coeff = np.polyfit(time_points, log_addresses, 1)[0]
                elif regime_type == 'linear':
                    # Fit linear growth
                    coeff = np.polyfit(time_points, addresses, 1)[0]
                elif regime_type == 'logarithmic':
                    # Fit logarithmic growth
                    log_time = np.log(time_points + 1)
                    coeff = np.polyfit(log_time, addresses, 1)[0]
                else:  # decline
                    # Negative growth
                    coeff = -np.polyfit(time_points, addresses, 1)[0]
                
                return coeff
            return 0.0
        except:
            return 0.0
    
    def _calculate_network_effect_strength(self, data: pd.DataFrame) -> float:
        """Calculate the strength of network effects"""
        try:
            if 'active_addresses' in data.columns and 'market_cap' in data.columns:
                addresses = data['active_addresses'].values
                market_caps = data['market_cap'].values
                
                # Calculate correlation between N^2 and market cap
                network_values = addresses ** 2
                correlation = np.corrcoef(network_values, market_caps)[0, 1]
                return abs(correlation) if not np.isnan(correlation) else 0.0
            return 0.0
        except:
            return 0.0
    
    def _calculate_regime_stability(self, data: pd.DataFrame) -> float:
        """Calculate regime stability score"""
        try:
            if 'active_addresses' in data.columns:
                addresses = data['active_addresses'].values
                growth_rates = np.diff(addresses) / addresses[:-1]
                
                # Stability as inverse of growth rate volatility
                volatility = np.std(growth_rates)
                stability = 1.0 / (1.0 + volatility)
                return stability
            return 0.0
        except:
            return 0.0
    
    def generate_predictions(self, data: pd.DataFrame) -> List[NetworkPrediction]:
        """Generate network value predictions"""
        predictions = []
        
        try:
            if not self.is_fitted:
                logger.warning("Model not fitted. Cannot generate predictions.")
                return predictions
            
            # Prepare features
            features = self._prepare_prediction_features(data)
            
            for horizon in self.prediction_horizons:
                # Ensemble prediction
                ensemble_predictions = []
                feature_importances = {}
                
                for name, model in self.network_models.items():
                    try:
                        pred = model.predict([features])[0]
                        ensemble_predictions.append(pred)
                        
                        # Get feature importance if available
                        if hasattr(model, 'feature_importances_'):
                            for i, importance in enumerate(model.feature_importances_):
                                feature_name = self.feature_names[i] if i < len(self.feature_names) else f'feature_{i}'
                                if feature_name not in feature_importances:
                                    feature_importances[feature_name] = 0
                                feature_importances[feature_name] += importance / len(self.network_models)
                    except Exception as e:
                        logger.error(f"Error in {name} prediction: {e}")
                        continue
                
                if ensemble_predictions:
                    # Calculate ensemble statistics
                    predicted_value = np.mean(ensemble_predictions)
                    prediction_std = np.std(ensemble_predictions)
                    
                    # Confidence interval
                    z_score = 1.96 if self.confidence_level == 0.95 else 2.58
                    ci_lower = predicted_value - z_score * prediction_std
                    ci_upper = predicted_value + z_score * prediction_std
                    
                    # Model confidence based on prediction agreement
                    model_confidence = 1.0 - (prediction_std / (abs(predicted_value) + 1))
                    model_confidence = max(0.0, min(1.0, model_confidence))
                    
                    # Risk factors
                    risk_factors = self._identify_prediction_risks(data, horizon)
                    
                    predictions.append(NetworkPrediction(
                        predicted_value=predicted_value,
                        confidence_interval=(ci_lower, ci_upper),
                        prediction_horizon=horizon,
                        model_confidence=model_confidence,
                        feature_importance=feature_importances,
                        risk_factors=risk_factors
                    ))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return predictions
    
    def _prepare_prediction_features(self, data: pd.DataFrame) -> List[float]:
        """Prepare optimized features for prediction with vectorization"""
        features = []
        
        try:
            # Limit data size for performance
            if len(data) > 500:
                data = data.tail(500)
            
            # Network metrics
            network_metrics = self.calculate_network_metrics(data)
            features.extend([
                network_metrics.active_addresses,
                network_metrics.transaction_count,
                network_metrics.network_value,
                network_metrics.metcalfe_ratio,
                network_metrics.network_density,
                network_metrics.clustering_coefficient,
                network_metrics.path_length,
                network_metrics.growth_rate,
                network_metrics.velocity
            ])
            
            # Technical indicators (vectorized)
            if len(data) >= 20:
                recent_data = data.tail(20)
                
                # Moving averages (vectorized)
                if 'active_addresses' in data.columns:
                    addr_values = recent_data['active_addresses'].values.astype(np.float32)
                    ma_7 = np.mean(addr_values[-7:]) if len(addr_values) >= 7 else np.mean(addr_values)
                    ma_30 = np.mean(addr_values)
                    ma_ratio = ma_7 / ma_30 if ma_30 > 0 else 1.0
                    features.extend([float(ma_7), float(ma_30), float(ma_ratio)])
                else:
                    features.extend([0.0, 0.0, 1.0])
                
                # Volatility measures (optimized)
                if 'market_cap' in data.columns:
                    market_values = recent_data['market_cap'].values.astype(np.float32)
                    returns = np.diff(market_values) / market_values[:-1]
                    volatility = float(np.std(returns) * np.sqrt(365)) if len(returns) > 0 else 0.0
                    features.append(volatility)
                else:
                    features.append(0.0)
            else:
                features.extend([0.0, 0.0, 1.0, 0.0])
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing prediction features: {e}")
            return [0.0] * 13
    
    def _identify_prediction_risks(self, data: pd.DataFrame, horizon: int) -> List[str]:
        """Identify risks for predictions"""
        risks = []
        
        try:
            # Data quality risks
            if len(data) < self.lookback_period:
                risks.append("Limited historical data")
            
            # Network growth risks
            network_metrics = self.calculate_network_metrics(data)
            if network_metrics.growth_rate < -0.1:
                risks.append("Declining network growth")
            
            # Market risks
            if 'market_cap' in data.columns and len(data) > 30:
                recent_volatility = data['market_cap'].tail(30).pct_change().std()
                if recent_volatility > 0.1:
                    risks.append("High market volatility")
            
            # Horizon-specific risks
            if horizon > 30:
                risks.append("Long-term prediction uncertainty")
            
            return risks
            
        except Exception as e:
            logger.error(f"Error identifying prediction risks: {e}")
            return ["Unknown risks"]
    
    def detect_anomalies(self, data: pd.DataFrame) -> List[NetworkAnomaly]:
        """Detect network anomalies"""
        anomalies = []
        
        try:
            if len(data) < 30:
                return anomalies
            
            # Calculate network metrics for recent period
            recent_data = data.tail(30)
            network_metrics = [self.calculate_network_metrics(recent_data.iloc[:i+1]) for i in range(len(recent_data))]
            
            # Check for growth spikes
            growth_rates = [m.growth_rate for m in network_metrics[-7:]]
            if len(growth_rates) > 0:
                avg_growth = np.mean(growth_rates)
                growth_std = np.std(growth_rates)
                
                if avg_growth > growth_std * self.anomaly_threshold:
                    anomalies.append(NetworkAnomaly(
                        anomaly_score=avg_growth / growth_std,
                        is_anomaly=True,
                        anomaly_type='growth_spike',
                        severity='high' if avg_growth > growth_std * 3 else 'medium',
                        affected_metrics=['growth_rate'],
                        potential_causes=['Network adoption surge', 'Market speculation']
                    ))
            
            # Check for value disconnects
            metcalfe_ratios = [m.metcalfe_ratio for m in network_metrics]
            if len(metcalfe_ratios) > 0:
                ratio_mean = np.mean(metcalfe_ratios)
                ratio_std = np.std(metcalfe_ratios)
                current_ratio = metcalfe_ratios[-1]
                
                if abs(current_ratio - ratio_mean) > ratio_std * self.anomaly_threshold:
                    anomalies.append(NetworkAnomaly(
                        anomaly_score=abs(current_ratio - ratio_mean) / ratio_std,
                        is_anomaly=True,
                        anomaly_type='value_disconnect',
                        severity='high' if abs(current_ratio - ratio_mean) > ratio_std * 3 else 'medium',
                        affected_metrics=['metcalfe_ratio'],
                        potential_causes=['Market mispricing', 'Network utility change']
                    ))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return anomalies
    
    def assess_network_risk(self, data: pd.DataFrame) -> NetworkRisk:
        """Assess comprehensive network risks"""
        try:
            risk_factors = {}
            
            # Network concentration risk
            network_metrics = self.calculate_network_metrics(data)
            concentration_risk = 1.0 - network_metrics.network_density
            risk_factors['concentration'] = concentration_risk
            
            # Growth sustainability risk
            if network_metrics.growth_rate > 0.5:  # Very high growth
                sustainability_risk = 0.8
            elif network_metrics.growth_rate < -0.1:  # Declining
                sustainability_risk = 0.9
            else:
                sustainability_risk = 0.3
            risk_factors['sustainability'] = sustainability_risk
            
            # Adoption risk based on network effects
            adoption_risk = 1.0 - network_metrics.clustering_coefficient
            risk_factors['adoption'] = adoption_risk
            
            # Technical risk
            technical_risk = 0.5  # Base technical risk
            if network_metrics.path_length > 10:
                technical_risk += 0.3
            risk_factors['technical'] = min(technical_risk, 1.0)
            
            # Market risk
            market_risk = 0.4  # Base market risk
            if 'market_cap' in data.columns and len(data) > 30:
                volatility = data['market_cap'].tail(30).pct_change().std()
                market_risk += min(volatility * 2, 0.5)
            risk_factors['market'] = min(market_risk, 1.0)
            
            # Overall risk
            overall_risk = np.mean(list(risk_factors.values()))
            
            # Mitigation strategies
            mitigation_strategies = []
            if concentration_risk > 0.7:
                mitigation_strategies.append("Diversify network participation")
            if sustainability_risk > 0.7:
                mitigation_strategies.append("Monitor growth sustainability")
            if adoption_risk > 0.7:
                mitigation_strategies.append("Improve network utility")
            
            return NetworkRisk(
                overall_risk=overall_risk,
                network_concentration_risk=concentration_risk,
                growth_sustainability_risk=sustainability_risk,
                adoption_risk=adoption_risk,
                technical_risk=risk_factors['technical'],
                market_risk=risk_factors['market'],
                risk_factors=risk_factors,
                mitigation_strategies=mitigation_strategies
            )
            
        except Exception as e:
            logger.error(f"Error assessing network risk: {e}")
            return NetworkRisk(
                overall_risk=0.5, network_concentration_risk=0.5,
                growth_sustainability_risk=0.5, adoption_risk=0.5,
                technical_risk=0.5, market_risk=0.5
            )
    
    def fit(self, data: pd.DataFrame) -> Dict[str, float]:
        """Fit the Metcalfe model"""
        try:
            if len(data) < self.lookback_period:
                raise ValueError(f"Insufficient data. Need at least {self.lookback_period} days.")
            
            logger.info("Fitting Quant Grade Metcalfe model...")
            
            # Prepare training data
            X, y = self._prepare_training_data(data)
            
            if len(X) == 0:
                raise ValueError("No valid training data prepared")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Fit regime detection model
            regime_features = []
            for i in range(self.regime_detection_window, len(data)):
                window_data = data.iloc[i-self.regime_detection_window:i]
                features = self._extract_regime_features(window_data)
                regime_features.append(features)
            
            if regime_features:
                self.regime_model.fit(regime_features)
            
            # Fit prediction models
            model_scores = {}
            tscv = TimeSeriesSplit(n_splits=3)
            
            for name, model in self.network_models.items():
                try:
                    # Cross-validation
                    cv_scores = []
                    for train_idx, val_idx in tscv.split(X_scaled):
                        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_val)
                        score = r2_score(y_val, y_pred)
                        cv_scores.append(score)
                    
                    model_scores[name] = np.mean(cv_scores)
                    
                    # Fit on full data
                    model.fit(X_scaled, y)
                    
                except Exception as e:
                    logger.error(f"Error fitting {name} model: {e}")
                    model_scores[name] = 0.0
            
            self.is_fitted = True
            logger.info("Model fitting completed successfully")
            
            return model_scores
            
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            return {}
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for the model"""
        X, y = [], []
        
        try:
            for i in range(30, len(data) - 1):  # Need lookback and lookahead
                # Features from current and historical data
                current_data = data.iloc[:i+1]
                features = self._prepare_prediction_features(current_data)
                
                # Target: future network value (Metcalfe's Law)
                if 'active_addresses' in data.columns:
                    future_addresses = data['active_addresses'].iloc[i+1]
                    target = future_addresses ** 2
                    
                    X.append(features)
                    y.append(target)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([])
    
    def analyze(self, data: pd.DataFrame) -> MetcalfeResult:
        """Perform comprehensive Metcalfe's Law analysis"""
        try:
            logger.info("Performing Quant Grade Metcalfe analysis...")
            
            # Calculate current network metrics
            network_metrics = self.calculate_network_metrics(data)
            
            # Detect current regime
            current_regime = self.detect_network_regime(data)
            
            # Generate predictions
            predictions = self.generate_predictions(data)
            
            # Detect anomalies
            anomalies = self.detect_anomalies(data)
            
            # Assess risks
            risk_assessment = self.assess_network_risk(data)
            
            # Calculate model performance if fitted
            model_performance = {}
            if self.is_fitted:
                model_performance = self._calculate_model_performance(data)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                network_metrics, current_regime, predictions, risk_assessment
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                network_metrics, current_regime, predictions, anomalies, risk_assessment
            )
            
            # Initialize result
            result = MetcalfeResult(
                timestamp=datetime.now(),
                network_metrics=network_metrics,
                current_regime=current_regime,
                predictions=predictions,
                anomalies=anomalies,
                risk_assessment=risk_assessment,
                model_performance=model_performance,
                confidence_score=confidence_score,
                recommendations=recommendations
            )
            
            # Add Kalman filter analysis if enabled and sufficient data
            if self.enable_kalman_filter and len(data) >= 30:
                result.kalman_analysis = self._perform_kalman_analysis(data)
            
            # Add Monte Carlo analysis if enabled and sufficient data
            if self.enable_monte_carlo and len(data) >= 50:
                result.monte_carlo_analysis = self._perform_monte_carlo_analysis(data)
            
            # Enhanced Metcalfe's Law analysis
            try:
                # Odlyzko's n*log(n) correction
                result.odlyzko_correction = self._calculate_odlyzko_correction(data)
                
                # Network effects decay analysis
                result.network_effects_decay = self._analyze_network_effects_decay(data)
                
                # Zipf's law analysis for address distribution
                result.zipf_law_analysis = self._analyze_zipf_law_distribution(data)
                
            except Exception as e:
                logger.error(f"Error in enhanced Metcalfe analysis: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in Metcalfe analysis: {e}")
            # Return default result
            return MetcalfeResult(
                timestamp=datetime.now(),
                network_metrics=NetworkMetrics(
                    active_addresses=0, transaction_count=0, network_value=0,
                    metcalfe_ratio=0, network_density=0, clustering_coefficient=0,
                    path_length=0
                ),
                current_regime=NetworkRegime(
                    regime_type='unknown', growth_coefficient=0,
                    network_effect_strength=0, regime_probability=0,
                    duration_days=0, stability_score=0
                ),
                predictions=[],
                anomalies=[],
                risk_assessment=NetworkRisk(
                    overall_risk=0.5, network_concentration_risk=0.5,
                    growth_sustainability_risk=0.5, adoption_risk=0.5,
                    technical_risk=0.5, market_risk=0.5
                ),
                confidence_score=0.0,
                recommendations=["Insufficient data for analysis"]
            )
    
    def _calculate_model_performance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate model performance metrics"""
        try:
            if len(data) < 50:
                return {}
            
            # Use recent data for performance evaluation
            test_data = data.tail(30)
            performance = {}
            
            # Calculate prediction accuracy for each model
            for name, model in self.network_models.items():
                try:
                    predictions = []
                    actuals = []
                    
                    for i in range(len(test_data) - 1):
                        current_data = data.iloc[:-(len(test_data)-i)]
                        features = self._prepare_prediction_features(current_data)
                        features_scaled = self.scaler.transform([features])
                        
                        pred = model.predict(features_scaled)[0]
                        actual = test_data['active_addresses'].iloc[i+1] ** 2
                        
                        predictions.append(pred)
                        actuals.append(actual)
                    
                    if predictions and actuals:
                        mse = mean_squared_error(actuals, predictions)
                        mae = mean_absolute_error(actuals, predictions)
                        r2 = r2_score(actuals, predictions)
                        
                        performance[f'{name}_mse'] = mse
                        performance[f'{name}_mae'] = mae
                        performance[f'{name}_r2'] = r2
                
                except Exception as e:
                    logger.error(f"Error calculating performance for {name}: {e}")
                    continue
            
            return performance
            
        except Exception as e:
            logger.error(f"Error calculating model performance: {e}")
            return {}
    
    def _calculate_confidence_score(self, network_metrics: NetworkMetrics,
                                  current_regime: NetworkRegime,
                                  predictions: List[NetworkPrediction],
                                  risk_assessment: NetworkRisk) -> float:
        """Calculate overall confidence score"""
        try:
            confidence_factors = []
            
            # Data quality confidence
            if network_metrics.active_addresses > 0:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.2)
            
            # Regime confidence
            confidence_factors.append(current_regime.regime_probability)
            
            # Prediction confidence
            if predictions:
                avg_pred_confidence = np.mean([p.model_confidence for p in predictions])
                confidence_factors.append(avg_pred_confidence)
            else:
                confidence_factors.append(0.3)
            
            # Risk-adjusted confidence
            risk_adjusted = 1.0 - risk_assessment.overall_risk
            confidence_factors.append(risk_adjusted)
            
            return np.mean(confidence_factors)
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.5
    
    def _generate_recommendations(self, network_metrics: NetworkMetrics,
                                current_regime: NetworkRegime,
                                predictions: List[NetworkPrediction],
                                anomalies: List[NetworkAnomaly],
                                risk_assessment: NetworkRisk) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        try:
            # Network growth recommendations
            if network_metrics.growth_rate < 0:
                recommendations.append("Network showing decline - investigate adoption barriers")
            elif network_metrics.growth_rate > 0.5:
                recommendations.append("Exceptional network growth - monitor sustainability")
            
            # Regime-based recommendations
            if current_regime.regime_type == 'exponential':
                recommendations.append("Exponential growth phase - capitalize on network effects")
            elif current_regime.regime_type == 'decline':
                recommendations.append("Network in decline - urgent intervention needed")
            
            # Prediction-based recommendations
            if predictions:
                short_term_pred = next((p for p in predictions if p.prediction_horizon <= 30), None)
                if short_term_pred and short_term_pred.model_confidence > 0.7:
                    if short_term_pred.predicted_value > network_metrics.network_value * 1.1:
                        recommendations.append("Strong network growth expected - consider expansion")
                    elif short_term_pred.predicted_value < network_metrics.network_value * 0.9:
                        recommendations.append("Network contraction predicted - prepare mitigation")
            
            # Anomaly-based recommendations
            for anomaly in anomalies:
                if anomaly.severity in ['high', 'critical']:
                    recommendations.append(f"Address {anomaly.anomaly_type} - {anomaly.severity} severity")
            
            # Risk-based recommendations
            if risk_assessment.overall_risk > 0.7:
                recommendations.append("High overall risk - implement risk mitigation strategies")
                recommendations.extend(risk_assessment.mitigation_strategies)
            
            # Network health recommendations
            if network_metrics.network_density < 0.3:
                recommendations.append("Low network density - improve connectivity")
            if network_metrics.clustering_coefficient < 0.2:
                recommendations.append("Low clustering - enhance community formation")
            
            return recommendations[:10]  # Limit to top 10 recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to analysis error"]
    
    def _perform_kalman_analysis(self, data: pd.DataFrame) -> MetcalfeKalmanFilterResult:
        """Perform Kalman filter analysis on network metrics"""
        try:
            # Prepare network value time series
            network_values = data['network_value'].values if 'network_value' in data.columns else data['close'].values
            
            if KALMAN_AVAILABLE:
                # Advanced Kalman filtering with pykalman
                from pykalman import KalmanFilter
                
                # State transition matrices for network growth modeling
                transition_matrices = np.array([[1, 1], [0, 1]])
                observation_matrices = np.array([[1, 0]])
                
                kf = KalmanFilter(
                    transition_matrices=transition_matrices,
                    observation_matrices=observation_matrices
                )
                
                state_means, state_covariances = kf.em(network_values.reshape(-1, 1)).smooth()
                
                # Extract filtered values and growth rates
                filtered_values = state_means[:, 0]
                growth_rates = state_means[:, 1]
                
                # Calculate prediction intervals
                prediction_std = np.sqrt(state_covariances[:, 0, 0])
                upper_bound = filtered_values + 2 * prediction_std
                lower_bound = filtered_values - 2 * prediction_std
                
                # Network growth trend analysis
                trend_strength = np.abs(np.mean(growth_rates[-10:]))
                trend_consistency = 1 - np.std(growth_rates[-10:]) / (np.mean(np.abs(growth_rates[-10:])) + 1e-8)
                
                return MetcalfeKalmanFilterResult(
                    filtered_values=filtered_values.tolist(),
                    prediction_intervals={'upper': upper_bound.tolist(), 'lower': lower_bound.tolist()},
                    state_estimates={'network_value': filtered_values.tolist(), 'growth_rate': growth_rates.tolist()},
                    innovation_statistics={'mean': float(np.mean(network_values - filtered_values)), 
                                         'std': float(np.std(network_values - filtered_values))},
                    model_likelihood=float(kf.loglikelihood(network_values.reshape(-1, 1))),
                    trend_analysis={'strength': float(trend_strength), 'consistency': float(trend_consistency)}
                )
            else:
                # Simplified Kalman-like filtering
                filtered_values = []
                prediction_variance = []
                
                # Simple exponential smoothing with adaptive parameters
                alpha = 0.3  # Smoothing parameter
                filtered_val = network_values[0]
                variance = 1.0
                
                for i, obs in enumerate(network_values):
                    if i == 0:
                        filtered_values.append(obs)
                        prediction_variance.append(variance)
                    else:
                        # Prediction step
                        predicted_val = filtered_val
                        predicted_var = variance + 0.1  # Process noise
                        
                        # Update step
                        kalman_gain = predicted_var / (predicted_var + 0.5)  # Observation noise
                        filtered_val = predicted_val + kalman_gain * (obs - predicted_val)
                        variance = (1 - kalman_gain) * predicted_var
                        
                        filtered_values.append(filtered_val)
                        prediction_variance.append(variance)
                
                filtered_values = np.array(filtered_values)
                prediction_std = np.sqrt(prediction_variance)
                
                return MetcalfeKalmanFilterResult(
                    filtered_values=filtered_values.tolist(),
                    prediction_intervals={
                        'upper': (filtered_values + 2 * prediction_std).tolist(),
                        'lower': (filtered_values - 2 * prediction_std).tolist()
                    },
                    state_estimates={'network_value': filtered_values.tolist()},
                    innovation_statistics={
                        'mean': float(np.mean(network_values - filtered_values)),
                        'std': float(np.std(network_values - filtered_values))
                    },
                    model_likelihood=0.0,
                    trend_analysis={'strength': 0.0, 'consistency': 0.0}
                )
                
        except Exception as e:
            logger.error(f"Error in Kalman filter analysis: {e}")
            return MetcalfeKalmanFilterResult(
                filtered_values=[],
                prediction_intervals={'upper': [], 'lower': []},
                state_estimates={},
                innovation_statistics={'mean': 0.0, 'std': 0.0},
                model_likelihood=0.0,
                trend_analysis={'strength': 0.0, 'consistency': 0.0}
            )
    
    def _calculate_odlyzko_correction(self, data: pd.DataFrame) -> OdlyzkoCorrection:
        """Calculate Odlyzko's n*log(n) correction to Metcalfe's Law"""
        try:
            # Get active addresses
            active_addresses = data['active_addresses'].iloc[-1] if 'active_addresses' in data.columns else 0
            market_cap = data['market_cap'].iloc[-1] if 'market_cap' in data.columns else 0
            
            if active_addresses <= 0:
                return OdlyzkoCorrection(
                    corrected_network_value=0, correction_factor=0, log_correction_ratio=0,
                    efficiency_improvement=0, theoretical_vs_actual=0, network_maturity_score=0,
                    scaling_coefficient=0
                )
            
            # Traditional Metcalfe's Law: N^2
            traditional_value = active_addresses ** 2
            
            # Odlyzko's correction: N * log(N)
            log_n = np.log(active_addresses) if active_addresses > 1 else 0
            odlyzko_value = active_addresses * log_n
            
            # Calculate correction metrics
            correction_factor = odlyzko_value / traditional_value if traditional_value > 0 else 0
            log_correction_ratio = log_n / active_addresses if active_addresses > 0 else 0
            
            # Network maturity score (higher for more mature networks)
            maturity_score = min(1.0, log_correction_ratio * 10) if log_correction_ratio > 0 else 0
            
            # Efficiency improvement (how much more realistic the valuation is)
            if market_cap > 0 and traditional_value > 0 and odlyzko_value > 0:
                traditional_ratio = market_cap / traditional_value
                odlyzko_ratio = market_cap / odlyzko_value
                efficiency_improvement = abs(1 - odlyzko_ratio) / abs(1 - traditional_ratio) if traditional_ratio != 1 else 1
            else:
                efficiency_improvement = 0
            
            # Theoretical vs actual comparison
            theoretical_vs_actual = odlyzko_value / market_cap if market_cap > 0 else 0
            
            # Scaling coefficient for network effects
            scaling_coefficient = log_n / 2 if log_n > 0 else 0
            
            return OdlyzkoCorrection(
                corrected_network_value=float(odlyzko_value),
                correction_factor=float(correction_factor),
                log_correction_ratio=float(log_correction_ratio),
                efficiency_improvement=float(efficiency_improvement),
                theoretical_vs_actual=float(theoretical_vs_actual),
                network_maturity_score=float(maturity_score),
                scaling_coefficient=float(scaling_coefficient)
            )
            
        except Exception as e:
            logger.error(f"Error calculating Odlyzko correction: {e}")
            return OdlyzkoCorrection(
                corrected_network_value=0, correction_factor=0, log_correction_ratio=0,
                efficiency_improvement=0, theoretical_vs_actual=0, network_maturity_score=0,
                scaling_coefficient=0
            )
    
    def _analyze_network_effects_decay(self, data: pd.DataFrame) -> NetworkEffectsDecay:
        """Analyze network effects decay and saturation"""
        try:
            if len(data) < 30:
                return NetworkEffectsDecay(
                    decay_rate=0, half_life_days=0, current_decay_factor=1,
                    network_saturation_level=0, marginal_utility_decline=0,
                    congestion_effects=0, network_efficiency_score=1
                )
            
            # Calculate network growth rates
            addresses = data['active_addresses'].fillna(0) if 'active_addresses' in data.columns else pd.Series([0] * len(data))
            transactions = data['transaction_count'].fillna(0) if 'transaction_count' in data.columns else pd.Series([0] * len(data))
            
            # Growth rate analysis
            address_growth = addresses.pct_change().fillna(0)
            tx_growth = transactions.pct_change().fillna(0)
            
            # Calculate decay rate using exponential decay model
            recent_growth = address_growth.tail(30).mean()
            historical_growth = address_growth.head(30).mean() if len(address_growth) > 60 else recent_growth
            
            decay_rate = (historical_growth - recent_growth) / historical_growth if historical_growth != 0 else 0
            decay_rate = max(0, min(1, decay_rate))  # Clamp between 0 and 1
            
            # Half-life calculation (days for growth to halve)
            if decay_rate > 0:
                half_life_days = np.log(0.5) / np.log(1 - decay_rate) if decay_rate < 1 else float('inf')
            else:
                half_life_days = float('inf')
            
            # Current decay factor
            current_decay_factor = 1 - decay_rate
            
            # Network saturation level (based on growth deceleration)
            max_addresses = addresses.max()
            current_addresses = addresses.iloc[-1]
            saturation_level = current_addresses / max_addresses if max_addresses > 0 else 0
            
            # Marginal utility decline (diminishing returns)
            if len(addresses) > 10:
                recent_efficiency = tx_growth.tail(10).mean() / address_growth.tail(10).mean() if address_growth.tail(10).mean() != 0 else 0
                historical_efficiency = tx_growth.head(10).mean() / address_growth.head(10).mean() if address_growth.head(10).mean() != 0 else 0
                marginal_utility_decline = max(0, (historical_efficiency - recent_efficiency) / historical_efficiency) if historical_efficiency != 0 else 0
            else:
                marginal_utility_decline = 0
            
            # Congestion effects (when transaction growth lags address growth)
            congestion_effects = max(0, address_growth.tail(10).mean() - tx_growth.tail(10).mean()) if len(address_growth) > 10 else 0
            
            # Network efficiency score
            efficiency_score = (1 - decay_rate) * (1 - marginal_utility_decline) * (1 - congestion_effects)
            efficiency_score = max(0, min(1, efficiency_score))
            
            return NetworkEffectsDecay(
                decay_rate=float(decay_rate),
                half_life_days=float(half_life_days) if half_life_days != float('inf') else 999999,
                current_decay_factor=float(current_decay_factor),
                network_saturation_level=float(saturation_level),
                marginal_utility_decline=float(marginal_utility_decline),
                congestion_effects=float(congestion_effects),
                network_efficiency_score=float(efficiency_score)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing network effects decay: {e}")
            return NetworkEffectsDecay(
                decay_rate=0, half_life_days=999999, current_decay_factor=1,
                network_saturation_level=0, marginal_utility_decline=0,
                congestion_effects=0, network_efficiency_score=1
            )
    
    def _analyze_zipf_law_distribution(self, data: pd.DataFrame) -> ZipfLawAnalysis:
        """Analyze address distribution using Zipf's law"""
        try:
            # For demonstration, we'll use synthetic distribution analysis
            # In practice, this would require detailed address balance data
            
            active_addresses = data['active_addresses'].iloc[-1] if 'active_addresses' in data.columns else 0
            market_cap = data['market_cap'].iloc[-1] if 'market_cap' in data.columns else 0
            
            if active_addresses <= 0:
                return ZipfLawAnalysis(
                    zipf_exponent=0, distribution_fitness=0, concentration_index=0,
                    whale_dominance_ratio=0, address_inequality_gini=0,
                    power_law_deviation=0, network_decentralization_score=0
                )
            
            # Estimate Zipf exponent (typically around 1 for perfect Zipf distribution)
            # Using network size as proxy for distribution characteristics
            log_addresses = np.log(active_addresses) if active_addresses > 1 else 0
            zipf_exponent = 1.0 + (0.1 * log_addresses / 10)  # Slight deviation from perfect Zipf
            
            # Distribution fitness (how well it follows Zipf's law)
            # Perfect Zipf would be 1.0, deviations reduce this score
            distribution_fitness = 1.0 / (1.0 + abs(zipf_exponent - 1.0))
            
            # Concentration index (Herfindahl-like measure)
            # Estimate based on network maturity
            network_maturity = min(1.0, log_addresses / 15) if log_addresses > 0 else 0
            concentration_index = 0.8 - (0.3 * network_maturity)  # More mature = less concentrated
            
            # Whale dominance ratio (estimated)
            # Assume top 1% holds significant portion, decreases with network growth
            whale_dominance_ratio = 0.9 - (0.4 * network_maturity)
            whale_dominance_ratio = max(0.1, min(0.9, whale_dominance_ratio))
            
            # Address inequality (Gini coefficient estimate)
            # Higher for younger networks, stabilizes for mature networks
            address_inequality_gini = 0.95 - (0.25 * network_maturity)
            address_inequality_gini = max(0.3, min(0.95, address_inequality_gini))
            
            # Power law deviation
            power_law_deviation = abs(zipf_exponent - 1.0)
            
            # Network decentralization score
            decentralization_score = (1 - concentration_index) * (1 - whale_dominance_ratio) * distribution_fitness
            decentralization_score = max(0, min(1, decentralization_score))
            
            return ZipfLawAnalysis(
                zipf_exponent=float(zipf_exponent),
                distribution_fitness=float(distribution_fitness),
                concentration_index=float(concentration_index),
                whale_dominance_ratio=float(whale_dominance_ratio),
                address_inequality_gini=float(address_inequality_gini),
                power_law_deviation=float(power_law_deviation),
                network_decentralization_score=float(decentralization_score)
            )
            
        except Exception as e:
            logger.error(f"Error analyzing Zipf law distribution: {e}")
            return ZipfLawAnalysis(
                zipf_exponent=1.0, distribution_fitness=0, concentration_index=0,
                whale_dominance_ratio=0, address_inequality_gini=0,
                power_law_deviation=0, network_decentralization_score=0
            )
    
    def _perform_monte_carlo_analysis(self, data: pd.DataFrame) -> MetcalfeMonteCarloResult:
        """Perform Monte Carlo simulation for network growth scenarios"""
        try:
            # Prepare network metrics for simulation
            network_values = data['network_value'].values if 'network_value' in data.columns else data['close'].values
            active_addresses = data['active_addresses'].values if 'active_addresses' in data.columns else network_values
            
            # Historical statistics for simulation parameters
            value_returns = np.diff(np.log(network_values + 1e-8))
            address_growth = np.diff(np.log(active_addresses + 1e-8))
            
            historical_stats = {
                'value_return_mean': float(np.mean(value_returns)),
                'value_return_std': float(np.std(value_returns)),
                'address_growth_mean': float(np.mean(address_growth)),
                'address_growth_std': float(np.std(address_growth)),
                'correlation': float(np.corrcoef(value_returns[1:], address_growth[1:])[0, 1]) if len(value_returns) > 1 else 0.0
            }
            
            # Monte Carlo simulation
            simulations = []
            simulation_horizons = [30, 90, 180]  # Days
            
            for horizon in simulation_horizons:
                horizon_simulations = []
                
                for _ in range(self.monte_carlo_simulations):
                    # Simulate correlated network growth
                    random_shocks = np.random.multivariate_normal(
                        [historical_stats['value_return_mean'], historical_stats['address_growth_mean']],
                        [[historical_stats['value_return_std']**2, 
                          historical_stats['correlation'] * historical_stats['value_return_std'] * historical_stats['address_growth_std']],
                         [historical_stats['correlation'] * historical_stats['value_return_std'] * historical_stats['address_growth_std'],
                          historical_stats['address_growth_std']**2]],
                        horizon
                    )
                    
                    # Simulate network value path
                    value_path = [network_values[-1]]
                    address_path = [active_addresses[-1]]
                    
                    for day in range(horizon):
                        # Network value evolution with Metcalfe's law influence
                        new_addresses = address_path[-1] * np.exp(random_shocks[day, 1])
                        metcalfe_factor = (new_addresses / address_path[0]) ** 2  # Metcalfe's law
                        new_value = value_path[-1] * np.exp(random_shocks[day, 0]) * (1 + 0.1 * (metcalfe_factor - 1))
                        
                        value_path.append(max(new_value, 0))
                        address_path.append(max(new_addresses, 0))
                    
                    horizon_simulations.append({
                        'final_value': value_path[-1],
                        'final_addresses': address_path[-1],
                        'max_value': max(value_path),
                        'min_value': min(value_path),
                        'volatility': np.std(np.diff(np.log(np.array(value_path) + 1e-8)))
                    })
                
                simulations.append({
                    'horizon_days': horizon,
                    'scenarios': horizon_simulations
                })
            
            # Calculate confidence intervals and risk metrics
            confidence_intervals = {}
            risk_metrics = {}
            scenario_probabilities = {}
            
            for sim in simulations:
                horizon = sim['horizon_days']
                final_values = [s['final_value'] for s in sim['scenarios']]
                
                confidence_intervals[f'{horizon}d'] = {
                    'p5': float(np.percentile(final_values, 5)),
                    'p25': float(np.percentile(final_values, 25)),
                    'p50': float(np.percentile(final_values, 50)),
                    'p75': float(np.percentile(final_values, 75)),
                    'p95': float(np.percentile(final_values, 95))
                }
                
                current_value = network_values[-1]
                risk_metrics[f'{horizon}d'] = {
                    'var_5': float(np.percentile(final_values, 5) - current_value),
                    'cvar_5': float(np.mean([v for v in final_values if v <= np.percentile(final_values, 5)]) - current_value),
                    'max_drawdown': float(min([s['min_value'] for s in sim['scenarios']]) - current_value),
                    'volatility': float(np.mean([s['volatility'] for s in sim['scenarios']]))
                }
                
                scenario_probabilities[f'{horizon}d'] = {
                    'growth_prob': float(len([v for v in final_values if v > current_value]) / len(final_values)),
                    'decline_prob': float(len([v for v in final_values if v < current_value]) / len(final_values)),
                    'extreme_growth_prob': float(len([v for v in final_values if v > current_value * 1.5]) / len(final_values)),
                    'extreme_decline_prob': float(len([v for v in final_values if v < current_value * 0.5]) / len(final_values))
                }
            
            # Stress test scenarios
            stress_tests = {
                'network_shock': self._simulate_network_shock(network_values, active_addresses, -0.3),
                'adoption_crisis': self._simulate_network_shock(network_values, active_addresses, -0.5),
                'exponential_growth': self._simulate_network_shock(network_values, active_addresses, 0.5)
            }
            
            return MetcalfeMonteCarloResult(
                historical_statistics=historical_stats,
                confidence_intervals=confidence_intervals,
                risk_metrics=risk_metrics,
                scenario_probabilities=scenario_probabilities,
                stress_test_results=stress_tests
            )
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo analysis: {e}")
            return MetcalfeMonteCarloResult(
                historical_statistics={},
                confidence_intervals={},
                risk_metrics={},
                scenario_probabilities={},
                stress_test_results={}
            )
    
    def _simulate_network_shock(self, network_values: np.ndarray, active_addresses: np.ndarray, shock_magnitude: float) -> Dict[str, float]:
        """Simulate network shock scenario"""
        try:
            current_value = network_values[-1]
            current_addresses = active_addresses[-1]
            
            # Apply shock to network metrics
            shocked_addresses = current_addresses * (1 + shock_magnitude)
            metcalfe_impact = (shocked_addresses / current_addresses) ** 2
            shocked_value = current_value * metcalfe_impact
            
            return {
                'shocked_network_value': float(shocked_value),
                'shocked_addresses': float(shocked_addresses),
                'value_impact_pct': float((shocked_value - current_value) / current_value * 100),
                'recovery_time_estimate': float(abs(shock_magnitude) * 30)  # Rough estimate in days
            }
        except Exception as e:
            logger.error(f"Error in shock simulation: {e}")
            return {'shocked_network_value': 0.0, 'shocked_addresses': 0.0, 'value_impact_pct': 0.0, 'recovery_time_estimate': 0.0}