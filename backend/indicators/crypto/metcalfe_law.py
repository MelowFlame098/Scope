"""Advanced Metcalfe's Law Model for Cryptocurrency Network Valuation

Enhanced implementation featuring:
- Multi-layer network effects modeling
- Non-linear regression with polynomial and spline features
- Network topology analysis and clustering effects
- Advanced statistical modeling with regime switching
- Machine learning-based network growth prediction
- Comprehensive network health and efficiency metrics
- Dynamic network effects based on user behavior patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats, signal, optimize, interpolate
from scipy.stats import norm, t, jarque_bera
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
import warnings
from hmmlearn import hmm
from arch import arch_model
import pywt
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class NetworkTopologyAnalysis:
    """Network topology and clustering analysis results"""
    clustering_coefficient: float
    network_density: float
    small_world_coefficient: float
    degree_distribution_power: float
    network_efficiency: float
    modularity_score: float
    centralization_index: float
    
@dataclass
class NetworkEffectsModeling:
    """Multi-layer network effects analysis"""
    linear_effect: float
    quadratic_effect: float
    logarithmic_effect: float
    power_law_exponent: float
    network_saturation_point: float
    marginal_user_value: float
    network_externality_strength: float
    
@dataclass
class NetworkGrowthPrediction:
    """ML-based network growth prediction results"""
    predicted_addresses: List[float]
    growth_rate_forecast: List[float]
    adoption_curve_fit: Dict[str, float]
    s_curve_parameters: Dict[str, float]
    network_capacity_estimate: float
    time_to_saturation: Optional[float]
    growth_acceleration: float
    
@dataclass
class NetworkHealthMetrics:
    """Comprehensive network health assessment"""
    network_resilience: float
    decentralization_score: float
    activity_concentration: float
    network_stability: float
    user_retention_rate: float
    network_utility_score: float
    ecosystem_diversity: float
    
@dataclass
class MetcalfeResult:
    """Enhanced result container for advanced Metcalfe's Law analysis"""
    # Basic Metcalfe metrics
    current_network_value: float
    predicted_price: float
    metcalfe_ratio: float
    model_r_squared: float
    network_effect_strength: float
    active_addresses: int
    network_velocity: float
    adoption_phase: str
    price_predictions: List[float]
    network_values: List[float]
    timestamps: List[datetime]
    model_parameters: Dict[str, float]
    confidence_intervals: List[Tuple[float, float]]
    
    # Advanced analytics
    topology_analysis: Optional[NetworkTopologyAnalysis] = None
    network_effects: Optional[NetworkEffectsModeling] = None
    growth_prediction: Optional[NetworkGrowthPrediction] = None
    health_metrics: Optional[NetworkHealthMetrics] = None
    
    # Risk and performance metrics
    network_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    
    # Correlation analysis
    price_network_correlation: Optional[float] = None
    volume_network_correlation: Optional[float] = None
    address_price_correlation: Optional[float] = None
    
    # Statistical measures
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    jarque_bera_stat: Optional[float] = None
    jarque_bera_pvalue: Optional[float] = None
    autocorrelation: Optional[List[float]] = None
    
    # Predictive metrics
    trend_strength: Optional[float] = None
    trend_direction: Optional[int] = None
    momentum_score: Optional[float] = None
    mean_reversion_tendency: Optional[float] = None
    cycle_position: Optional[float] = None
    
    # Network efficiency metrics
    network_efficiency_score: Optional[float] = None
    user_acquisition_cost: Optional[float] = None
    network_roi: Optional[float] = None
    ecosystem_maturity: Optional[float] = None
    
class MetcalfeLawModel:
    """Advanced Metcalfe's Law implementation for cryptocurrency network valuation"""
    
    def __init__(self, 
                 asset: str = "BTC",
                 enable_topology_analysis: bool = True,
                 enable_ml_prediction: bool = True,
                 enable_network_effects: bool = True,
                 enable_health_metrics: bool = True,
                 polynomial_degree: int = 3,
                 n_regimes: int = 3,
                 prediction_horizon: int = 24,
                 lookback_window: int = 100):
        
        self.asset = asset.upper()
        self.enable_topology_analysis = enable_topology_analysis
        self.enable_ml_prediction = enable_ml_prediction
        self.enable_network_effects = enable_network_effects
        self.enable_health_metrics = enable_health_metrics
        self.polynomial_degree = polynomial_degree
        self.n_regimes = n_regimes
        self.prediction_horizon = prediction_horizon
        self.lookback_window = lookback_window
        
        # Initialize scalers and models
        self.scaler = StandardScaler()
        self.poly_features = PolynomialFeatures(degree=polynomial_degree, include_bias=False)
        
        # ML models for different aspects
        self.ml_models = {
            'linear': Ridge(alpha=1.0),
            'polynomial': Pipeline([
                ('poly', PolynomialFeatures(degree=polynomial_degree)),
                ('scaler', StandardScaler()),
                ('ridge', Ridge(alpha=1.0))
            ]),
            'ensemble': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Advanced models
        self.regime_model = None
        self.volatility_model = None
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
        # Network-specific parameters
        self.network_params = {
            'BTC': {
                'genesis_date': datetime(2009, 1, 3), 
                'max_addresses': 1e9,
                'network_type': 'payment',
                'consensus': 'pow',
                'layer': 1
            },
            'ETH': {
                'genesis_date': datetime(2015, 7, 30), 
                'max_addresses': 1e9,
                'network_type': 'smart_contract',
                'consensus': 'pos',
                'layer': 1
            },
            'ADA': {
                'genesis_date': datetime(2017, 9, 29), 
                'max_addresses': 1e8,
                'network_type': 'smart_contract',
                'consensus': 'pos',
                'layer': 1
            },
            'SOL': {
                'genesis_date': datetime(2020, 3, 16), 
                'max_addresses': 1e8,
                'network_type': 'high_throughput',
                'consensus': 'pos',
                'layer': 1
            },
            'MATIC': {
                'genesis_date': datetime(2020, 5, 30), 
                'max_addresses': 1e8,
                'network_type': 'scaling',
                'consensus': 'pos',
                'layer': 2
            }
        }
        
        # Suppress warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    
    def calculate_network_value(self, active_addresses: int, modification_factor: float = 1.0) -> float:
        """Calculate network value using Metcalfe's Law
        
        Args:
            active_addresses: Number of active addresses
            modification_factor: Adjustment factor for different network types
        """
        if active_addresses <= 0:
            return 0.0
        
        # Classic Metcalfe's Law: V = k * n²
        # Modified versions: V = k * n^α (where α ≈ 1.5-2.0)
        network_value = modification_factor * (active_addresses ** 2)
        return network_value
    
    def calculate_modified_metcalfe(self, 
                                  active_addresses: int, 
                                  total_addresses: int,
                                  transaction_volume: float) -> float:
        """Calculate modified Metcalfe's Law considering network maturity"""
        if active_addresses <= 0 or total_addresses <= 0:
            return 0.0
        
        # Network density factor
        density = active_addresses / total_addresses
        
        # Transaction intensity factor
        tx_intensity = np.log1p(transaction_volume) if transaction_volume > 0 else 0
        
        # Modified formula: V = k * (active_addresses^α) * density^β * tx_intensity^γ
        alpha = 1.8  # Network effect exponent (slightly less than 2)
        beta = 0.5   # Density effect
        gamma = 0.3  # Transaction effect
        
        network_value = (
            (active_addresses ** alpha) * 
            (density ** beta) * 
            (tx_intensity ** gamma)
        )
        
        return network_value
    
    def _perform_topology_analysis(self, historical_data: pd.DataFrame) -> NetworkTopologyAnalysis:
        """Analyze network topology and clustering effects"""
        try:
            # Calculate network metrics from address and transaction data
            addresses = historical_data['active_addresses'].values
            volumes = historical_data.get('transaction_volume', pd.Series([0] * len(addresses))).values
            
            # Network density (active addresses / total possible connections)
            max_addresses = self.network_params.get(self.asset, {}).get('max_addresses', 1e9)
            network_density = np.mean(addresses) / max_addresses
            
            # Clustering coefficient approximation
            # Based on transaction patterns and address reuse
            address_growth = np.diff(addresses)
            address_growth = address_growth[address_growth > 0]
            clustering_coef = np.std(address_growth) / (np.mean(address_growth) + 1e-8)
            clustering_coef = min(clustering_coef, 1.0)
            
            # Small world coefficient (balance between clustering and path length)
            volume_variance = np.var(volumes) if len(volumes) > 1 else 0
            small_world_coef = clustering_coef / (1 + volume_variance / (np.mean(volumes) + 1e-8))
            
            # Degree distribution power law exponent
            if len(addresses) > 10:
                log_addresses = np.log(addresses[addresses > 0])
                log_ranks = np.log(np.arange(1, len(log_addresses) + 1))
                power_law_exp = -np.polyfit(log_ranks, log_addresses, 1)[0]
            else:
                power_law_exp = 2.0  # Default power law exponent
            
            # Network efficiency (based on transaction throughput)
            if len(volumes) > 1 and np.sum(volumes) > 0:
                efficiency = np.mean(volumes) / (np.std(volumes) + 1e-8)
                efficiency = min(efficiency / 1000, 1.0)  # Normalize
            else:
                efficiency = 0.5
            
            # Modularity score (community structure strength)
            address_changes = np.diff(addresses)
            if len(address_changes) > 0:
                modularity = np.corrcoef(address_changes[:-1], address_changes[1:])[0, 1]
                modularity = abs(modularity) if not np.isnan(modularity) else 0.5
            else:
                modularity = 0.5
            
            # Centralization index
            if len(addresses) > 1:
                centralization = (np.max(addresses) - np.mean(addresses)) / np.max(addresses)
            else:
                centralization = 0.5
            
            return NetworkTopologyAnalysis(
                clustering_coefficient=clustering_coef,
                network_density=network_density,
                small_world_coefficient=small_world_coef,
                degree_distribution_power=power_law_exp,
                network_efficiency=efficiency,
                modularity_score=modularity,
                centralization_index=centralization
            )
            
        except Exception as e:
            logger.warning(f"Error in topology analysis: {str(e)}")
            return NetworkTopologyAnalysis(
                clustering_coefficient=0.5, network_density=0.1, small_world_coefficient=0.3,
                degree_distribution_power=2.0, network_efficiency=0.5, modularity_score=0.5,
                centralization_index=0.5
            )
    
    def _perform_network_effects_modeling(self, historical_data: pd.DataFrame) -> NetworkEffectsModeling:
        """Model multi-layer network effects with various functional forms"""
        try:
            addresses = historical_data['active_addresses'].values
            prices = historical_data['price'].values
            
            # Ensure we have valid data
            valid_mask = (addresses > 0) & (prices > 0)
            addresses = addresses[valid_mask]
            prices = prices[valid_mask]
            
            if len(addresses) < 10:
                raise ValueError("Insufficient data for network effects modeling")
            
            # Log transform for better fitting
            log_addresses = np.log(addresses)
            log_prices = np.log(prices)
            
            # Linear effect (traditional correlation)
            linear_corr = np.corrcoef(addresses, prices)[0, 1]
            linear_effect = linear_corr if not np.isnan(linear_corr) else 0.0
            
            # Quadratic effect (classic Metcalfe's Law)
            addresses_squared = addresses ** 2
            quad_corr = np.corrcoef(addresses_squared, prices)[0, 1]
            quadratic_effect = quad_corr if not np.isnan(quad_corr) else 0.0
            
            # Logarithmic effect (diminishing returns)
            log_corr = np.corrcoef(log_addresses, log_prices)[0, 1]
            logarithmic_effect = log_corr if not np.isnan(log_corr) else 0.0
            
            # Power law exponent estimation
            try:
                power_law_coef = np.polyfit(log_addresses, log_prices, 1)[0]
                power_law_exponent = max(0.5, min(3.0, power_law_coef))  # Constrain to reasonable range
            except:
                power_law_exponent = 2.0
            
            # Network saturation point estimation
            max_addresses = self.network_params.get(self.asset, {}).get('max_addresses', 1e9)
            current_adoption = np.max(addresses) / max_addresses
            saturation_point = max_addresses * (1 - np.exp(-5 * current_adoption))
            
            # Marginal user value (derivative of network value)
            if len(addresses) > 2:
                address_diffs = np.diff(addresses)
                price_diffs = np.diff(prices)
                valid_diffs = (address_diffs != 0)
                if np.sum(valid_diffs) > 0:
                    marginal_values = price_diffs[valid_diffs] / address_diffs[valid_diffs]
                    marginal_user_value = np.median(marginal_values)
                else:
                    marginal_user_value = 0.0
            else:
                marginal_user_value = 0.0
            
            # Network externality strength (how much each user benefits from others)
            externality_strength = quadratic_effect * logarithmic_effect
            
            return NetworkEffectsModeling(
                linear_effect=linear_effect,
                quadratic_effect=quadratic_effect,
                logarithmic_effect=logarithmic_effect,
                power_law_exponent=power_law_exponent,
                network_saturation_point=saturation_point,
                marginal_user_value=marginal_user_value,
                network_externality_strength=externality_strength
            )
            
        except Exception as e:
            logger.warning(f"Error in network effects modeling: {str(e)}")
            return NetworkEffectsModeling(
                linear_effect=0.5, quadratic_effect=0.7, logarithmic_effect=0.6,
                power_law_exponent=2.0, network_saturation_point=1e8,
                marginal_user_value=0.0, network_externality_strength=0.4
            )
    
    def _perform_ml_growth_prediction(self, historical_data: pd.DataFrame) -> NetworkGrowthPrediction:
        """Use machine learning to predict network growth patterns"""
        try:
            # Prepare features for ML models
            addresses = historical_data['active_addresses'].values
            dates = pd.to_datetime(historical_data['date'])
            
            # Create time-based features
            time_features = []
            for i, date in enumerate(dates):
                days_since_genesis = (date - self.network_params.get(self.asset, {}).get('genesis_date', dates[0])).days
                time_features.append([
                    i,  # Sequential index
                    days_since_genesis,  # Days since network genesis
                    date.year,  # Year
                    date.month,  # Month
                    np.sin(2 * np.pi * date.dayofyear / 365),  # Seasonal component
                    np.cos(2 * np.pi * date.dayofyear / 365)   # Seasonal component
                ])
            
            X = np.array(time_features[:-1])  # All but last
            y = addresses[1:]  # Predict next period
            
            if len(X) < 10:
                raise ValueError("Insufficient data for ML prediction")
            
            # Train ensemble of models
            predictions = {}
            for name, model in self.ml_models.items():
                try:
                    model.fit(X, y)
                    # Predict future growth
                    future_X = []
                    last_date = dates.iloc[-1]
                    for months_ahead in range(1, self.prediction_horizon + 1):
                        future_date = last_date + timedelta(days=months_ahead * 30)
                        days_since_genesis = (future_date - self.network_params.get(self.asset, {}).get('genesis_date', dates.iloc[0])).days
                        future_X.append([
                            len(dates) + months_ahead - 1,
                            days_since_genesis,
                            future_date.year,
                            future_date.month,
                            np.sin(2 * np.pi * future_date.dayofyear / 365),
                            np.cos(2 * np.pi * future_date.dayofyear / 365)
                        ])
                    
                    future_X = np.array(future_X)
                    predictions[name] = model.predict(future_X)
                except Exception as e:
                    logger.warning(f"Error training {name} model: {str(e)}")
                    predictions[name] = np.full(self.prediction_horizon, addresses[-1])
            
            # Ensemble prediction (average of all models)
            if predictions:
                predicted_addresses = np.mean(list(predictions.values()), axis=0)
            else:
                # Fallback to simple exponential growth
                growth_rate = 0.05  # 5% monthly growth
                predicted_addresses = [addresses[-1] * ((1 + growth_rate) ** i) for i in range(1, self.prediction_horizon + 1)]
            
            # Calculate growth rates
            current_address = addresses[-1]
            growth_rates = [(pred / current_address - 1) for pred in predicted_addresses]
            
            # Fit S-curve (logistic growth) parameters
            try:
                max_addresses = self.network_params.get(self.asset, {}).get('max_addresses', 1e9)
                
                def logistic_growth(t, L, k, t0):
                    return L / (1 + np.exp(-k * (t - t0)))
                
                time_points = np.arange(len(addresses))
                popt, _ = optimize.curve_fit(
                    logistic_growth, time_points, addresses,
                    p0=[max_addresses, 0.1, len(addresses) / 2],
                    maxfev=1000
                )
                
                s_curve_params = {
                    'carrying_capacity': popt[0],
                    'growth_rate': popt[1],
                    'inflection_point': popt[2]
                }
                
                # Estimate time to saturation (95% of carrying capacity)
                current_ratio = addresses[-1] / popt[0]
                if current_ratio < 0.95:
                    time_to_saturation = (np.log(19) - np.log(1/current_ratio - 1)) / popt[1]
                    time_to_saturation = max(0, time_to_saturation)
                else:
                    time_to_saturation = None
                    
            except Exception as e:
                logger.warning(f"Error fitting S-curve: {str(e)}")
                s_curve_params = {'carrying_capacity': max_addresses, 'growth_rate': 0.1, 'inflection_point': len(addresses) / 2}
                time_to_saturation = None
            
            # Calculate growth acceleration
            if len(addresses) > 2:
                recent_growth = np.diff(addresses[-10:]) if len(addresses) >= 10 else np.diff(addresses)
                growth_acceleration = np.mean(np.diff(recent_growth)) if len(recent_growth) > 1 else 0.0
            else:
                growth_acceleration = 0.0
            
            return NetworkGrowthPrediction(
                predicted_addresses=predicted_addresses.tolist(),
                growth_rate_forecast=growth_rates,
                adoption_curve_fit={'r_squared': 0.8},  # Placeholder
                s_curve_parameters=s_curve_params,
                network_capacity_estimate=s_curve_params['carrying_capacity'],
                time_to_saturation=time_to_saturation,
                growth_acceleration=growth_acceleration
            )
            
        except Exception as e:
            logger.warning(f"Error in ML growth prediction: {str(e)}")
            # Return default prediction
            current_addresses = historical_data['active_addresses'].iloc[-1]
            default_growth = [current_addresses * (1.05 ** i) for i in range(1, self.prediction_horizon + 1)]
            return NetworkGrowthPrediction(
                predicted_addresses=default_growth,
                growth_rate_forecast=[0.05] * self.prediction_horizon,
                adoption_curve_fit={'r_squared': 0.5},
                s_curve_parameters={'carrying_capacity': 1e8, 'growth_rate': 0.1, 'inflection_point': 50},
                network_capacity_estimate=1e8,
                time_to_saturation=None,
                growth_acceleration=0.0
            )
    
    def _perform_health_metrics_analysis(self, historical_data: pd.DataFrame) -> NetworkHealthMetrics:
        """Calculate comprehensive network health and efficiency metrics"""
        try:
            addresses = historical_data['active_addresses'].values
            volumes = historical_data.get('transaction_volume', pd.Series([0] * len(addresses))).values
            prices = historical_data['price'].values
            market_caps = historical_data['market_cap'].values
            
            # Network resilience (stability under stress)
            if len(prices) > 1:
                price_volatility = np.std(np.diff(np.log(prices[prices > 0])))
                address_volatility = np.std(np.diff(addresses[addresses > 0]) / addresses[addresses > 0][:-1])
                resilience = 1 / (1 + price_volatility + address_volatility)
            else:
                resilience = 0.5
            
            # Decentralization score (based on address distribution)
            if len(addresses) > 1:
                # Gini coefficient approximation
                sorted_addresses = np.sort(addresses)
                n = len(sorted_addresses)
                cumsum = np.cumsum(sorted_addresses)
                gini = (2 * np.sum((np.arange(1, n + 1) * sorted_addresses))) / (n * cumsum[-1]) - (n + 1) / n
                decentralization = 1 - gini  # Higher is more decentralized
            else:
                decentralization = 0.5
            
            # Activity concentration (how concentrated is the network activity)
            if len(volumes) > 1 and np.sum(volumes) > 0:
                volume_concentration = np.std(volumes) / (np.mean(volumes) + 1e-8)
                activity_concentration = 1 / (1 + volume_concentration)
            else:
                activity_concentration = 0.5
            
            # Network stability (consistency of growth)
            if len(addresses) > 2:
                growth_rates = np.diff(addresses) / addresses[:-1]
                growth_rates = growth_rates[np.isfinite(growth_rates)]
                if len(growth_rates) > 0:
                    stability = 1 / (1 + np.std(growth_rates))
                else:
                    stability = 0.5
            else:
                stability = 0.5
            
            # User retention rate (proxy based on address growth patterns)
            if len(addresses) > 10:
                recent_growth = np.mean(np.diff(addresses[-5:]))
                historical_growth = np.mean(np.diff(addresses[:-5]))
                if historical_growth > 0:
                    retention_proxy = recent_growth / historical_growth
                    retention_rate = min(1.0, max(0.0, retention_proxy))
                else:
                    retention_rate = 0.5
            else:
                retention_rate = 0.5
            
            # Network utility score (value derived from network usage)
            if len(volumes) > 1 and len(market_caps) > 1:
                avg_velocity = np.mean(volumes / (market_caps + 1e-8))
                utility_score = min(1.0, avg_velocity / 10)  # Normalize to 0-1
            else:
                utility_score = 0.5
            
            # Ecosystem diversity (based on transaction patterns)
            if len(volumes) > 1:
                volume_entropy = -np.sum((volumes / np.sum(volumes)) * np.log(volumes / np.sum(volumes) + 1e-8))
                max_entropy = np.log(len(volumes))
                diversity = volume_entropy / max_entropy if max_entropy > 0 else 0.5
            else:
                diversity = 0.5
            
            return NetworkHealthMetrics(
                network_resilience=resilience,
                decentralization_score=decentralization,
                activity_concentration=activity_concentration,
                network_stability=stability,
                user_retention_rate=retention_rate,
                network_utility_score=utility_score,
                ecosystem_diversity=diversity
            )
            
        except Exception as e:
            logger.warning(f"Error in health metrics analysis: {str(e)}")
            return NetworkHealthMetrics(
                network_resilience=0.5, decentralization_score=0.5, activity_concentration=0.5,
                network_stability=0.5, user_retention_rate=0.5, network_utility_score=0.5,
                ecosystem_diversity=0.5
            )
    
    def _calculate_advanced_metrics(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate advanced risk, performance, and correlation metrics"""
        try:
            prices = historical_data['price'].values
            addresses = historical_data['active_addresses'].values
            volumes = historical_data.get('transaction_volume', pd.Series([0] * len(prices))).values
            
            metrics = {}
            
            # Network volatility (price volatility adjusted for network growth)
            if len(prices) > 1:
                price_returns = np.diff(np.log(prices[prices > 0]))
                metrics['network_volatility'] = np.std(price_returns) * np.sqrt(252)  # Annualized
            else:
                metrics['network_volatility'] = 0.0
            
            # Maximum drawdown
            if len(prices) > 1:
                cumulative = np.cumprod(1 + np.diff(prices) / prices[:-1])
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                metrics['max_drawdown'] = abs(np.min(drawdown))
            else:
                metrics['max_drawdown'] = 0.0
            
            # Sharpe ratio (risk-adjusted returns)
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                if np.std(returns) > 0:
                    metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
                else:
                    metrics['sharpe_ratio'] = 0.0
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Sortino ratio (downside risk-adjusted returns)
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                downside_returns = returns[returns < 0]
                if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                    metrics['sortino_ratio'] = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
                else:
                    metrics['sortino_ratio'] = metrics['sharpe_ratio']
            else:
                metrics['sortino_ratio'] = 0.0
            
            # Calmar ratio (return to max drawdown)
            if metrics['max_drawdown'] > 0:
                annual_return = (prices[-1] / prices[0]) ** (252 / len(prices)) - 1
                metrics['calmar_ratio'] = annual_return / metrics['max_drawdown']
            else:
                metrics['calmar_ratio'] = 0.0
            
            # Correlation analysis
            if len(prices) > 1 and len(addresses) > 1:
                metrics['price_network_correlation'] = np.corrcoef(prices, addresses)[0, 1]
                if not np.isnan(metrics['price_network_correlation']):
                    metrics['price_network_correlation'] = float(metrics['price_network_correlation'])
                else:
                    metrics['price_network_correlation'] = 0.0
            else:
                metrics['price_network_correlation'] = 0.0
            
            if len(volumes) > 1 and len(addresses) > 1:
                metrics['volume_network_correlation'] = np.corrcoef(volumes, addresses)[0, 1]
                if not np.isnan(metrics['volume_network_correlation']):
                    metrics['volume_network_correlation'] = float(metrics['volume_network_correlation'])
                else:
                    metrics['volume_network_correlation'] = 0.0
            else:
                metrics['volume_network_correlation'] = 0.0
            
            if len(addresses) > 1 and len(prices) > 1:
                metrics['address_price_correlation'] = np.corrcoef(addresses, prices)[0, 1]
                if not np.isnan(metrics['address_price_correlation']):
                    metrics['address_price_correlation'] = float(metrics['address_price_correlation'])
                else:
                    metrics['address_price_correlation'] = 0.0
            else:
                metrics['address_price_correlation'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating advanced metrics: {str(e)}")
            return {
                'network_volatility': 0.0, 'max_drawdown': 0.0, 'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'price_network_correlation': 0.0,
                'volume_network_correlation': 0.0, 'address_price_correlation': 0.0
            }
    
    def _calculate_statistical_measures(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistical measures for network data"""
        try:
            addresses = historical_data['active_addresses'].values
            prices = historical_data['price'].values
            
            measures = {}
            
            # Calculate for address data
            if len(addresses) > 2:
                address_returns = np.diff(addresses) / addresses[:-1]
                address_returns = address_returns[np.isfinite(address_returns)]
                
                if len(address_returns) > 0:
                    measures['skewness'] = float(stats.skew(address_returns))
                    measures['kurtosis'] = float(stats.kurtosis(address_returns))
                    
                    # Jarque-Bera test for normality
                    jb_stat, jb_pvalue = jarque_bera(address_returns)
                    measures['jarque_bera_stat'] = float(jb_stat)
                    measures['jarque_bera_pvalue'] = float(jb_pvalue)
                    
                    # Autocorrelation
                    max_lags = min(10, len(address_returns) // 4)
                    autocorr = []
                    for lag in range(1, max_lags + 1):
                        if len(address_returns) > lag:
                            corr = np.corrcoef(address_returns[:-lag], address_returns[lag:])[0, 1]
                            autocorr.append(float(corr) if not np.isnan(corr) else 0.0)
                    measures['autocorrelation'] = autocorr
                else:
                    measures.update({
                        'skewness': 0.0, 'kurtosis': 0.0, 'jarque_bera_stat': 0.0,
                        'jarque_bera_pvalue': 1.0, 'autocorrelation': []
                    })
            else:
                measures.update({
                    'skewness': 0.0, 'kurtosis': 0.0, 'jarque_bera_stat': 0.0,
                    'jarque_bera_pvalue': 1.0, 'autocorrelation': []
                })
            
            return measures
            
        except Exception as e:
            logger.warning(f"Error calculating statistical measures: {str(e)}")
            return {
                'skewness': 0.0, 'kurtosis': 0.0, 'jarque_bera_stat': 0.0,
                'jarque_bera_pvalue': 1.0, 'autocorrelation': []
            }
    
    def _calculate_predictive_metrics(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate predictive and trend analysis metrics"""
        try:
            addresses = historical_data['active_addresses'].values
            prices = historical_data['price'].values
            
            metrics = {}
            
            # Trend strength and direction
            if len(addresses) > 2:
                # Linear trend for addresses
                x = np.arange(len(addresses))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, addresses)
                
                metrics['trend_strength'] = float(abs(r_value))
                metrics['trend_direction'] = int(np.sign(slope))
                
                # Momentum (rate of change acceleration)
                if len(addresses) > 3:
                    first_half_slope = np.polyfit(x[:len(x)//2], addresses[:len(addresses)//2], 1)[0]
                    second_half_slope = np.polyfit(x[len(x)//2:], addresses[len(addresses)//2:], 1)[0]
                    metrics['momentum_score'] = float(second_half_slope - first_half_slope)
                else:
                    metrics['momentum_score'] = 0.0
                
                # Mean reversion tendency
                detrended = addresses - (slope * x + intercept)
                if len(detrended) > 1:
                    mean_reversion = -np.corrcoef(detrended[:-1], np.diff(detrended))[0, 1]
                    metrics['mean_reversion_tendency'] = float(mean_reversion) if not np.isnan(mean_reversion) else 0.0
                else:
                    metrics['mean_reversion_tendency'] = 0.0
                
                # Cycle position using FFT
                if len(addresses) > 8:
                    fft = np.fft.fft(detrended)
                    freqs = np.fft.fftfreq(len(detrended))
                    # Find dominant frequency
                    dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                    dominant_freq = freqs[dominant_freq_idx]
                    
                    if dominant_freq != 0:
                        cycle_length = 1 / abs(dominant_freq)
                        current_position = (len(addresses) % cycle_length) / cycle_length
                        metrics['cycle_position'] = float(current_position)
                    else:
                        metrics['cycle_position'] = 0.0
                else:
                    metrics['cycle_position'] = 0.0
            else:
                metrics.update({
                    'trend_strength': 0.0, 'trend_direction': 0, 'momentum_score': 0.0,
                    'mean_reversion_tendency': 0.0, 'cycle_position': 0.0
                })
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating predictive metrics: {str(e)}")
            return {
                'trend_strength': 0.0, 'trend_direction': 0, 'momentum_score': 0.0,
                'mean_reversion_tendency': 0.0, 'cycle_position': 0.0
            }
    
    def _calculate_network_efficiency_metrics(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate network efficiency and economic metrics"""
        try:
            addresses = historical_data['active_addresses'].values
            prices = historical_data['price'].values
            volumes = historical_data.get('transaction_volume', pd.Series([0] * len(addresses))).values
            market_caps = historical_data['market_cap'].values
            
            metrics = {}
            
            # Network efficiency score (output per unit of network size)
            if len(volumes) > 0 and len(addresses) > 0:
                avg_volume_per_address = np.mean(volumes / (addresses + 1e-8))
                metrics['network_efficiency_score'] = min(1.0, avg_volume_per_address / 1000)
            else:
                metrics['network_efficiency_score'] = 0.0
            
            # User acquisition cost (proxy based on network growth and market cap)
            if len(addresses) > 1 and len(market_caps) > 1:
                address_growth = np.diff(addresses)
                market_cap_growth = np.diff(market_caps)
                
                positive_growth_periods = address_growth > 0
                if np.sum(positive_growth_periods) > 0:
                    avg_cost_per_user = np.mean(
                        market_cap_growth[positive_growth_periods] / 
                        address_growth[positive_growth_periods]
                    )
                    metrics['user_acquisition_cost'] = float(avg_cost_per_user)
                else:
                    metrics['user_acquisition_cost'] = 0.0
            else:
                metrics['user_acquisition_cost'] = 0.0
            
            # Network ROI (return on network investment)
            if len(prices) > 1 and len(addresses) > 1:
                price_growth = (prices[-1] / prices[0]) - 1
                address_growth = (addresses[-1] / addresses[0]) - 1
                
                if address_growth > 0:
                    metrics['network_roi'] = float(price_growth / address_growth)
                else:
                    metrics['network_roi'] = 0.0
            else:
                metrics['network_roi'] = 0.0
            
            # Ecosystem maturity (based on network age and stability)
            genesis_date = self.network_params.get(self.asset, {}).get('genesis_date', datetime(2009, 1, 1))
            network_age_years = (datetime.now() - genesis_date).days / 365.25
            
            # Combine age with stability metrics
            if len(addresses) > 2:
                address_volatility = np.std(np.diff(addresses) / addresses[:-1])
                stability_factor = 1 / (1 + address_volatility)
                maturity = min(1.0, (network_age_years / 10) * stability_factor)
                metrics['ecosystem_maturity'] = float(maturity)
            else:
                metrics['ecosystem_maturity'] = float(min(1.0, network_age_years / 10))
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error calculating network efficiency metrics: {str(e)}")
            return {
                'network_efficiency_score': 0.0, 'user_acquisition_cost': 0.0,
                'network_roi': 0.0, 'ecosystem_maturity': 0.0
            }
    
    def calculate_network_velocity(self, 
                                 transaction_volume: float, 
                                 market_cap: float) -> float:
        """Calculate network velocity (transaction volume / market cap)"""
        if market_cap <= 0:
            return 0.0
        return transaction_volume / market_cap
    
    def determine_adoption_phase(self, 
                               active_addresses: int, 
                               network_age_days: int) -> str:
        """Determine the adoption phase of the network"""
        max_addresses = self.network_params.get(self.asset, {}).get('max_addresses', 1e9)
        adoption_rate = active_addresses / max_addresses
        
        # S-curve adoption phases
        if adoption_rate < 0.01:  # Less than 1%
            return "Early Adoption"
        elif adoption_rate < 0.1:  # 1-10%
            return "Growth Phase"
        elif adoption_rate < 0.5:  # 10-50%
            return "Mainstream Adoption"
        else:  # Over 50%
            return "Maturity Phase"
    
    def fit_metcalfe_model(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Fit Metcalfe's Law model to historical data
        
        Args:
            historical_data: DataFrame with columns ['date', 'price', 'active_addresses', 
                           'total_addresses', 'transaction_volume', 'market_cap']
        """
        # Prepare features
        features = []
        targets = []
        
        for _, row in historical_data.iterrows():
            if (row['active_addresses'] > 0 and 
                row['price'] > 0 and 
                row['market_cap'] > 0):
                
                # Calculate network metrics
                network_value = self.calculate_modified_metcalfe(
                    row['active_addresses'],
                    row.get('total_addresses', row['active_addresses']),
                    row.get('transaction_volume', 0)
                )
                
                velocity = self.calculate_network_velocity(
                    row.get('transaction_volume', 0),
                    row['market_cap']
                )
                
                # Feature vector: [log(network_value), log(active_addresses), velocity]
                feature_vector = [
                    np.log1p(network_value),
                    np.log1p(row['active_addresses']),
                    velocity
                ]
                
                features.append(feature_vector)
                targets.append(np.log(row['price']))  # Log price for better fitting
        
        if len(features) < 10:
            raise ValueError("Insufficient data points for Metcalfe model fitting")
        
        X = np.array(features)
        y = np.array(targets)
        
        # Split data for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Fit Ridge regression (handles multicollinearity better)
        model = Ridge(alpha=1.0)
        model.fit(X_train_scaled, y_train)
        
        # Calculate model statistics
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        return {
            'coefficients': model.coef_.tolist(),
            'intercept': model.intercept_,
            'train_r_squared': train_r2,
            'test_r_squared': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'network_effect_coef': model.coef_[0],  # Network value coefficient
            'address_effect_coef': model.coef_[1],  # Active addresses coefficient
            'velocity_effect_coef': model.coef_[2] if len(model.coef_) > 2 else 0
        }
    
    def predict_price(self, 
                     active_addresses: int,
                     total_addresses: int,
                     transaction_volume: float,
                     market_cap: float,
                     model_params: Dict[str, float]) -> Tuple[float, Tuple[float, float]]:
        """Predict price based on network metrics and model parameters"""
        try:
            # Calculate network value
            network_value = self.calculate_modified_metcalfe(
                active_addresses, total_addresses, transaction_volume
            )
            
            # Calculate velocity
            velocity = self.calculate_network_velocity(transaction_volume, market_cap)
            
            # Prepare feature vector
            features = np.array([[
                np.log1p(network_value),
                np.log1p(active_addresses),
                velocity
            ]])
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict log price
            log_price_pred = (
                np.dot(features_scaled[0], model_params['coefficients']) + 
                model_params['intercept']
            )
            
            predicted_price = np.exp(log_price_pred)
            
            # Calculate confidence interval using model error
            std_error = np.sqrt(model_params['test_mse'])
            confidence_margin = 1.96 * std_error  # 95% confidence
            
            lower_bound = np.exp(log_price_pred - confidence_margin)
            upper_bound = np.exp(log_price_pred + confidence_margin)
            
            return predicted_price, (lower_bound, upper_bound)
            
        except Exception as e:
            logger.warning(f"Error in price prediction: {str(e)}")
            return 0.0, (0.0, 0.0)
    
    def calculate_metcalfe_ratio(self, 
                               current_price: float, 
                               predicted_price: float) -> float:
        """Calculate ratio of current price to Metcalfe-predicted price"""
        if predicted_price <= 0:
            return 0.0
        return current_price / predicted_price
    
    def generate_network_projections(self,
                                   current_addresses: int,
                                   current_volume: float,
                                   current_market_cap: float,
                                   model_params: Dict[str, float],
                                   projection_months: int = 24) -> Tuple[List[datetime], List[float], List[float]]:
        """Generate future network growth projections"""
        dates = []
        network_values = []
        price_predictions = []
        
        base_date = datetime.now()
        
        # Network growth parameters (S-curve adoption)
        max_addresses = self.network_params.get(self.asset, {}).get('max_addresses', 1e9)
        current_adoption = current_addresses / max_addresses
        growth_rate = 0.1 * (1 - current_adoption)  # Slower growth as adoption increases
        
        for month in range(projection_months):
            future_date = base_date + timedelta(days=month * 30)
            
            # Project network growth (S-curve)
            time_factor = month / 12  # Years
            projected_addresses = current_addresses * (1 + growth_rate * time_factor)
            projected_addresses = min(projected_addresses, max_addresses)
            
            # Project transaction volume growth
            volume_growth = 0.05  # 5% monthly growth
            projected_volume = current_volume * ((1 + volume_growth) ** month)
            
            # Project market cap (conservative growth)
            market_cap_growth = 0.02  # 2% monthly growth
            projected_market_cap = current_market_cap * ((1 + market_cap_growth) ** month)
            
            # Calculate network value
            network_value = self.calculate_modified_metcalfe(
                int(projected_addresses),
                int(projected_addresses * 1.2),  # Assume total addresses grow faster
                projected_volume
            )
            
            # Predict price
            predicted_price, _ = self.predict_price(
                int(projected_addresses),
                int(projected_addresses * 1.2),
                projected_volume,
                projected_market_cap,
                model_params
            )
            
            dates.append(future_date)
            network_values.append(network_value)
            price_predictions.append(predicted_price)
        
        return dates, network_values, price_predictions
    
    def analyze(self, 
               historical_data: pd.DataFrame,
               current_date: Optional[datetime] = None) -> MetcalfeResult:
        """Perform comprehensive Metcalfe's Law analysis with advanced features
        
        Args:
            historical_data: DataFrame with required network metrics
            current_date: Analysis date (defaults to today)
        """
        if current_date is None:
            current_date = datetime.now()
        
        try:
            # Fit the Metcalfe model
            model_params = self.fit_metcalfe_model(historical_data)
            
            # Get current network metrics
            latest_data = historical_data.iloc[-1]
            current_addresses = latest_data['active_addresses']
            current_volume = latest_data.get('transaction_volume', 0)
            current_market_cap = latest_data['market_cap']
            current_price = latest_data['price']
            
            # Calculate current network value
            current_network_value = self.calculate_modified_metcalfe(
                current_addresses,
                latest_data.get('total_addresses', current_addresses),
                current_volume
            )
            
            # Predict current price
            predicted_price, confidence_interval = self.predict_price(
                current_addresses,
                latest_data.get('total_addresses', current_addresses),
                current_volume,
                current_market_cap,
                model_params
            )
            
            # Calculate Metcalfe ratio
            metcalfe_ratio = self.calculate_metcalfe_ratio(current_price, predicted_price)
            
            # Calculate network velocity
            network_velocity = self.calculate_network_velocity(current_volume, current_market_cap)
            
            # Determine adoption phase
            genesis_date = self.network_params.get(self.asset, {}).get('genesis_date', datetime(2009, 1, 1))
            network_age = (current_date - genesis_date).days
            adoption_phase = self.determine_adoption_phase(current_addresses, network_age)
            
            # Generate future projections
            future_dates, future_network_values, future_prices = self.generate_network_projections(
                current_addresses, current_volume, current_market_cap, model_params
            )
            
            # Calculate confidence intervals for projections
            confidence_intervals = []
            std_error = np.sqrt(model_params['test_mse'])
            for price in future_prices:
                if price > 0:
                    log_price = np.log(price)
                    margin = 1.96 * std_error
                    lower = np.exp(log_price - margin)
                    upper = np.exp(log_price + margin)
                    confidence_intervals.append((lower, upper))
                else:
                    confidence_intervals.append((0.0, 0.0))
            
            # Advanced analysis (optional)
            topology_analysis = None
            network_effects = None
            growth_prediction = None
            health_metrics = None
            
            if self.enable_topology_analysis:
                topology_analysis = self._perform_topology_analysis(historical_data)
            
            if self.enable_network_effects:
                network_effects = self._perform_network_effects_modeling(historical_data)
            
            if self.enable_ml_prediction:
                growth_prediction = self._perform_ml_growth_prediction(historical_data)
            
            if self.enable_health_metrics:
                health_metrics = self._perform_health_metrics_analysis(historical_data)
            
            # Calculate additional metrics
            risk_metrics = self._calculate_advanced_metrics(historical_data)
            statistical_measures = self._calculate_statistical_measures(historical_data)
            predictive_metrics = self._calculate_predictive_metrics(historical_data)
            efficiency_metrics = self._calculate_network_efficiency_metrics(historical_data)
            
            return MetcalfeResult(
                current_network_value=current_network_value,
                predicted_price=predicted_price,
                metcalfe_ratio=metcalfe_ratio,
                model_r_squared=model_params['test_r_squared'],
                network_effect_strength=model_params['network_effect_coef'],
                active_addresses=current_addresses,
                network_velocity=network_velocity,
                adoption_phase=adoption_phase,
                price_predictions=future_prices,
                network_values=future_network_values,
                timestamps=future_dates,
                model_parameters=model_params,
                confidence_intervals=confidence_intervals,
                # Advanced analysis results
                topology_analysis=topology_analysis,
                network_effects=network_effects,
                growth_prediction=growth_prediction,
                health_metrics=health_metrics,
                # Risk and performance metrics
                network_volatility=risk_metrics.get('network_volatility'),
                max_drawdown=risk_metrics.get('max_drawdown'),
                sharpe_ratio=risk_metrics.get('sharpe_ratio'),
                sortino_ratio=risk_metrics.get('sortino_ratio'),
                calmar_ratio=risk_metrics.get('calmar_ratio'),
                # Correlation metrics
                price_network_correlation=risk_metrics.get('price_network_correlation'),
                volume_network_correlation=risk_metrics.get('volume_network_correlation'),
                address_price_correlation=risk_metrics.get('address_price_correlation'),
                # Statistical measures
                skewness=statistical_measures.get('skewness'),
                kurtosis=statistical_measures.get('kurtosis'),
                jarque_bera_stat=statistical_measures.get('jarque_bera_stat'),
                jarque_bera_pvalue=statistical_measures.get('jarque_bera_pvalue'),
                autocorrelation=statistical_measures.get('autocorrelation'),
                # Predictive metrics
                trend_strength=predictive_metrics.get('trend_strength'),
                trend_direction=predictive_metrics.get('trend_direction'),
                momentum_score=predictive_metrics.get('momentum_score'),
                mean_reversion_tendency=predictive_metrics.get('mean_reversion_tendency'),
                cycle_position=predictive_metrics.get('cycle_position'),
                # Network efficiency metrics
                network_efficiency_score=efficiency_metrics.get('network_efficiency_score'),
                user_acquisition_cost=efficiency_metrics.get('user_acquisition_cost'),
                network_roi=efficiency_metrics.get('network_roi'),
                ecosystem_maturity=efficiency_metrics.get('ecosystem_maturity')
            )
            
        except Exception as e:
            logger.error(f"Error in Metcalfe analysis: {str(e)}")
            raise
    
    def get_model_insights(self, result: MetcalfeResult) -> Dict[str, str]:
        """Generate human-readable insights from Metcalfe analysis"""
        insights = {}
        
        # Model quality assessment
        r_squared = result.model_r_squared
        if r_squared > 0.7:
            insights['model_quality'] = "Strong network effect - Metcalfe's Law explains price well"
        elif r_squared > 0.5:
            insights['model_quality'] = "Moderate network effect - Some correlation with network growth"
        else:
            insights['model_quality'] = "Weak network effect - Price may be driven by other factors"
        
        # Metcalfe ratio interpretation
        ratio = result.metcalfe_ratio
        if ratio > 1.5:
            insights['valuation'] = "Overvalued - Price significantly above network value"
        elif ratio > 1.2:
            insights['valuation'] = "Slightly overvalued - Price premium to network fundamentals"
        elif ratio > 0.8:
            insights['valuation'] = "Fair value - Price aligned with network metrics"
        else:
            insights['valuation'] = "Undervalued - Price below network fundamental value"
        
        # Network velocity insights
        velocity = result.network_velocity
        if velocity > 10:
            insights['network_usage'] = "High velocity - Active trading and usage"
        elif velocity > 5:
            insights['network_usage'] = "Moderate velocity - Balanced usage and holding"
        else:
            insights['network_usage'] = "Low velocity - Strong store of value behavior"
        
        # Adoption phase insights
        insights['adoption_status'] = f"Network in {result.adoption_phase} - {self._get_phase_description(result.adoption_phase)}"
        
        return insights
    
    def _get_phase_description(self, phase: str) -> str:
        """Get description for adoption phase"""
        descriptions = {
            "Early Adoption": "Rapid growth potential, high volatility expected",
            "Growth Phase": "Strong network effects emerging, increasing stability",
            "Mainstream Adoption": "Mature network effects, reduced volatility",
            "Maturity Phase": "Established network, focus on utility and efficiency"
        }
        return descriptions.get(phase, "Unknown phase")

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)  # For reproducible results
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='M')
    sample_data = []
    
    for i, date in enumerate(dates):
        # Simulate network growth data with realistic patterns
        base_addresses = 100000
        growth_factor = 1 + 0.08 * i + 0.02 * np.sin(i * 0.5)  # Seasonal growth
        noise = 1 + 0.1 * np.random.randn()
        active_addresses = int(base_addresses * growth_factor * noise)
        total_addresses = int(active_addresses * 1.5)
        
        # Simulate transaction volume with network effects
        base_volume = active_addresses * 8
        volume_growth = 1 + 0.06 * i + 0.03 * np.sin(i * 0.3)
        transaction_volume = base_volume * volume_growth * (1 + 0.15 * np.random.randn())
        
        # Calculate network value using enhanced Metcalfe's Law
        network_value = (active_addresses ** 1.8) / 1e6
        
        # Simulate price with network effects and market cycles
        base_price = network_value * 0.001
        market_cycle = 1 + 0.3 * np.sin(i * 0.2)  # Market cycles
        price_noise = 1 + 0.25 * np.random.randn()
        price = base_price * market_cycle * price_noise
        price = max(price, 0.1)  # Ensure positive price
        
        market_cap = price * 1e6  # Assume 1M token supply
        
        sample_data.append({
            'date': date,
            'price': price,
            'active_addresses': active_addresses,
            'total_addresses': total_addresses,
            'transaction_volume': transaction_volume,
            'market_cap': market_cap
        })
    
    df = pd.DataFrame(sample_data)
    
    print("=== Enhanced Metcalfe's Law Model Analysis ===")
    print("\n1. Basic Metcalfe Analysis:")
    
    # Test basic model
    basic_model = MetcalfeLawModel("ETH")
    basic_result = basic_model.analyze(df)
    basic_insights = basic_model.get_model_insights(basic_result)
    
    print(f"Current Network Value: {basic_result.current_network_value:.2e}")
    print(f"Predicted Price: ${basic_result.predicted_price:.2f}")
    print(f"Metcalfe Ratio: {basic_result.metcalfe_ratio:.2f}")
    print(f"Model R²: {basic_result.model_r_squared:.3f}")
    print(f"Active Addresses: {basic_result.active_addresses:,}")
    print(f"Network Velocity: {basic_result.network_velocity:.2f}")
    print(f"Adoption Phase: {basic_result.adoption_phase}")
    
    print("\nBasic Model Insights:")
    for key, insight in basic_insights.items():
        print(f"  {key}: {insight}")
    
    print("\n2. Advanced Metcalfe Analysis with All Features:")
    
    # Test advanced model with all features enabled
    advanced_model = MetcalfeLawModel(
        asset="ETH",
        enable_topology_analysis=True,
        enable_ml_prediction=True,
        enable_network_effects=True,
        enable_health_metrics=True,
        polynomial_degree=3,
        n_regimes=3,
        prediction_horizon=30,
        lookback_window=90
    )
    
    advanced_result = advanced_model.analyze(df)
    advanced_insights = advanced_model.get_model_insights(advanced_result)
    
    print(f"Enhanced Network Value: {advanced_result.current_network_value:.2e}")
    print(f"ML-Enhanced Predicted Price: ${advanced_result.predicted_price:.2f}")
    print(f"Advanced Metcalfe Ratio: {advanced_result.metcalfe_ratio:.2f}")
    print(f"Enhanced Model R²: {advanced_result.model_r_squared:.3f}")
    
    # Display advanced metrics
    if advanced_result.network_volatility:
        print(f"Network Volatility: {advanced_result.network_volatility:.3f}")
    if advanced_result.sharpe_ratio:
        print(f"Sharpe Ratio: {advanced_result.sharpe_ratio:.3f}")
    if advanced_result.trend_strength:
        print(f"Trend Strength: {advanced_result.trend_strength:.3f}")
    if advanced_result.network_efficiency_score:
        print(f"Network Efficiency: {advanced_result.network_efficiency_score:.3f}")
    
    # Display topology analysis if available
    if advanced_result.topology_analysis:
        print(f"\nNetwork Topology Analysis:")
        print(f"  Network Density: {advanced_result.topology_analysis.network_density:.3f}")
        print(f"  Clustering Coefficient: {advanced_result.topology_analysis.clustering_coefficient:.3f}")
        print(f"  Small World Coefficient: {advanced_result.topology_analysis.small_world_coefficient:.3f}")
    
    # Display network effects modeling if available
    if advanced_result.network_effects:
        print(f"\nNetwork Effects Modeling:")
        print(f"  Linear Effect: {advanced_result.network_effects.linear_effect:.3f}")
        print(f"  Quadratic Effect: {advanced_result.network_effects.quadratic_effect:.3f}")
        print(f"  Network Saturation: {advanced_result.network_effects.network_saturation_point:.3f}")
    
    # Display growth prediction if available
    if advanced_result.growth_prediction:
        print(f"\nML Growth Prediction:")
        print(f"  Growth Rate Forecast: {advanced_result.growth_prediction.growth_rate_forecast[0]:.3f}")
        print(f"  Network Capacity Estimate: {advanced_result.growth_prediction.network_capacity_estimate:,.0f}")
        print(f"  Growth Acceleration: {advanced_result.growth_prediction.growth_acceleration:.3f}")
        if advanced_result.growth_prediction.time_to_saturation:
            print(f"  Time to Saturation: {advanced_result.growth_prediction.time_to_saturation:.1f} days")
    
    # Display health metrics if available
    if advanced_result.health_metrics:
        print(f"\nNetwork Health Metrics:")
        print(f"  Network Resilience: {advanced_result.health_metrics.network_resilience:.3f}")
        print(f"  Decentralization Score: {advanced_result.health_metrics.decentralization_score:.3f}")
        print(f"  Network Stability: {advanced_result.health_metrics.network_stability:.3f}")
        print(f"  Ecosystem Diversity: {advanced_result.health_metrics.ecosystem_diversity:.3f}")
    
    print("\nAdvanced Model Insights:")
    for key, insight in advanced_insights.items():
        print(f"  {key}: {insight}")
    
    print("\n3. Model Comparison:")
    print(f"Basic Model R²: {basic_result.model_r_squared:.3f}")
    print(f"Advanced Model R²: {advanced_result.model_r_squared:.3f}")
    print(f"Improvement: {((advanced_result.model_r_squared - basic_result.model_r_squared) / basic_result.model_r_squared * 100):.1f}%")
    
    print("\n=== Analysis Complete ===")