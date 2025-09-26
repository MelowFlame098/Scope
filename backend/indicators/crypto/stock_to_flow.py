"""Advanced Stock-to-Flow (S2F) Model for Bitcoin and Cryptocurrency Valuation

This module implements a sophisticated Stock-to-Flow model with advanced quantitative features:
- Regime switching models for different market phases
- Kalman filtering for dynamic parameter estimation
- Monte Carlo simulations for uncertainty quantification
- Advanced statistical methods and volatility modeling
- Machine learning enhancements for improved predictions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats, optimize
from scipy.stats import norm, t
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from filterpy.kalman import KalmanFilter
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class RegimeSwitchingResult:
    """Results from regime switching analysis"""
    current_regime: int
    regime_probabilities: List[float]
    regime_parameters: Dict[int, Dict[str, float]]
    regime_transitions: np.ndarray
    regime_history: List[int]
    regime_descriptions: Dict[int, str]

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulations"""
    price_scenarios: np.ndarray
    confidence_bands: Dict[str, Tuple[float, float]]
    var_estimates: Dict[str, float]  # Value at Risk
    expected_returns: Dict[str, float]
    scenario_probabilities: List[float]
    stress_test_results: Dict[str, float]

@dataclass
class KalmanFilterResult:
    """Results from Kalman filtering"""
    filtered_states: np.ndarray
    state_covariances: np.ndarray
    innovation_residuals: np.ndarray
    log_likelihood: float
    adaptive_parameters: Dict[str, List[float]]

@dataclass
class S2FResult:
    """Enhanced result container for advanced Stock-to-Flow analysis"""
    # Core S2F metrics
    current_s2f_ratio: float
    predicted_price: float
    model_r_squared: float
    confidence_interval: Tuple[float, float]
    next_halving_date: Optional[datetime]
    days_to_halving: Optional[int]
    historical_s2f: List[float]
    price_predictions: List[float]
    timestamps: List[datetime]
    model_parameters: Dict[str, float]
    
    # Advanced analytics
    regime_analysis: Optional[RegimeSwitchingResult] = None
    monte_carlo_analysis: Optional[MonteCarloResult] = None
    kalman_analysis: Optional[KalmanFilterResult] = None
    volatility_metrics: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    model_diagnostics: Dict[str, float] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Statistical measures
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
class StockToFlowModel:
    """Advanced Stock-to-Flow model implementation with quantitative enhancements"""
    
    def __init__(self, asset: str = "BTC", enable_regime_switching: bool = True,
                 enable_kalman_filter: bool = True, enable_monte_carlo: bool = True):
        self.asset = asset.upper()
        self.btc_halving_blocks = 210000  # Bitcoin halving every 210,000 blocks
        self.btc_block_time = 10 * 60  # 10 minutes in seconds
        self.btc_genesis_date = datetime(2009, 1, 3)
        
        # Advanced features flags
        self.enable_regime_switching = enable_regime_switching
        self.enable_kalman_filter = enable_kalman_filter
        self.enable_monte_carlo = enable_monte_carlo
        
        # Bitcoin-specific parameters
        if self.asset == "BTC":
            self.initial_reward = 50
            self.total_supply_cap = 21_000_000
        else:
            # Default parameters for other cryptocurrencies
            self.initial_reward = 1
            self.total_supply_cap = 1_000_000_000
            
        # Initialize advanced models
        self.regime_model = None
        self.kalman_filter = None
        self.scaler = StandardScaler()
        
        # Model parameters
        self.n_regimes = 3  # Bull, Bear, Neutral
        self.monte_carlo_simulations = 10000
    
    def calculate_bitcoin_supply_schedule(self, target_date: datetime) -> Tuple[float, float]:
        """Calculate Bitcoin's current supply and annual production rate"""
        days_since_genesis = (target_date - self.btc_genesis_date).days
        blocks_mined = days_since_genesis * 24 * 60 * 60 / self.btc_block_time
        
        # Calculate current block reward based on halving events
        halvings = int(blocks_mined // self.btc_halving_blocks)
        current_reward = self.initial_reward / (2 ** halvings)
        
        # Calculate total supply
        total_supply = 0
        remaining_blocks = blocks_mined
        reward = self.initial_reward
        
        for i in range(halvings + 1):
            if i < halvings:
                # Complete halving periods
                total_supply += self.btc_halving_blocks * reward
                remaining_blocks -= self.btc_halving_blocks
            else:
                # Current period
                total_supply += remaining_blocks * reward
            reward /= 2
        
        # Annual production (flow)
        blocks_per_year = 365.25 * 24 * 60 * 60 / self.btc_block_time
        annual_production = blocks_per_year * current_reward
        
        return total_supply, annual_production
    
    def calculate_s2f_ratio(self, supply: float, annual_production: float) -> float:
        """Calculate Stock-to-Flow ratio"""
        if annual_production <= 0:
            return float('inf')
        return supply / annual_production
    
    def fit_s2f_model(self, historical_data: pd.DataFrame) -> Dict[str, float]:
        """Fit the Stock-to-Flow model to historical data
        
        Args:
            historical_data: DataFrame with columns ['date', 'price', 'supply', 'flow']
        """
        # Calculate S2F ratios
        s2f_ratios = []
        log_prices = []
        
        for _, row in historical_data.iterrows():
            s2f = self.calculate_s2f_ratio(row['supply'], row['flow'])
            if s2f > 0 and s2f != float('inf') and row['price'] > 0:
                s2f_ratios.append(np.log10(s2f))
                log_prices.append(np.log10(row['price']))
        
        if len(s2f_ratios) < 10:
            raise ValueError("Insufficient data points for S2F model fitting")
        
        # Linear regression: log(Price) = a * log(S2F) + b
        X = np.array(s2f_ratios).reshape(-1, 1)
        y = np.array(log_prices)
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Calculate model statistics
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        
        return {
            'slope': model.coef_[0],
            'intercept': model.intercept_,
            'r_squared': r_squared,
            'mse': mse,
            'std_error': np.sqrt(mse)
        }
    
    def predict_price(self, s2f_ratio: float, model_params: Dict[str, float]) -> Tuple[float, Tuple[float, float]]:
        """Predict price based on S2F ratio and model parameters"""
        if s2f_ratio <= 0:
            return 0.0, (0.0, 0.0)
        
        log_s2f = np.log10(s2f_ratio)
        log_price_pred = model_params['slope'] * log_s2f + model_params['intercept']
        predicted_price = 10 ** log_price_pred
        
        # Calculate confidence interval (95%)
        std_error = model_params['std_error']
        confidence_margin = 1.96 * std_error  # 95% confidence
        
        lower_bound = 10 ** (log_price_pred - confidence_margin)
        upper_bound = 10 ** (log_price_pred + confidence_margin)
        
        return predicted_price, (lower_bound, upper_bound)
    
    def calculate_next_halving(self, current_date: datetime) -> Tuple[Optional[datetime], Optional[int]]:
        """Calculate next Bitcoin halving date"""
        if self.asset != "BTC":
            return None, None
        
        days_since_genesis = (current_date - self.btc_genesis_date).days
        blocks_mined = days_since_genesis * 24 * 60 * 60 / self.btc_block_time
        
        # Find next halving block
        current_halving_period = int(blocks_mined // self.btc_halving_blocks)
        next_halving_block = (current_halving_period + 1) * self.btc_halving_blocks
        
        # Calculate blocks remaining
        blocks_remaining = next_halving_block - blocks_mined
        
        # Convert to days
        days_remaining = blocks_remaining * self.btc_block_time / (24 * 60 * 60)
        next_halving_date = current_date + timedelta(days=days_remaining)
        
        return next_halving_date, int(days_remaining)
    
    def generate_future_projections(self, 
                                  model_params: Dict[str, float],
                                  current_date: datetime,
                                  projection_years: int = 10) -> Tuple[List[datetime], List[float], List[float]]:
        """Generate future S2F projections and price predictions"""
        dates = []
        s2f_ratios = []
        price_predictions = []
        
        for days_ahead in range(0, projection_years * 365, 30):  # Monthly projections
            future_date = current_date + timedelta(days=days_ahead)
            
            if self.asset == "BTC":
                supply, flow = self.calculate_bitcoin_supply_schedule(future_date)
            else:
                # Simplified model for other cryptocurrencies
                # Assume decreasing inflation rate
                years_from_now = days_ahead / 365.25
                supply = 1_000_000 * (1 + 0.1 * np.exp(-0.1 * years_from_now))
                flow = supply * 0.05 * np.exp(-0.05 * years_from_now)  # Decreasing inflation
            
            s2f_ratio = self.calculate_s2f_ratio(supply, flow)
            predicted_price, _ = self.predict_price(s2f_ratio, model_params)
            
            dates.append(future_date)
            s2f_ratios.append(s2f_ratio)
            price_predictions.append(predicted_price)
        
        return dates, s2f_ratios, price_predictions
    
    def _setup_kalman_filter(self, n_observations: int) -> KalmanFilter:
        """Initialize Kalman filter for dynamic parameter estimation"""
        # State: [slope, intercept, slope_velocity, intercept_velocity]
        kf = KalmanFilter(dim_x=4, dim_z=1)
        
        # State transition matrix (constant velocity model)
        kf.F = np.array([[1., 0., 1., 0.],
                        [0., 1., 0., 1.],
                        [0., 0., 1., 0.],
                        [0., 0., 0., 1.]])
        
        # Measurement function (observe price)
        kf.H = np.array([[1., 1., 0., 0.]])
        
        # Process noise covariance
        kf.Q = np.eye(4) * 0.01
        
        # Measurement noise covariance
        kf.R = np.array([[0.1]])
        
        # Initial state covariance
        kf.P *= 100
        
        return kf
    
    def _fit_regime_switching_model(self, log_prices: np.ndarray, log_s2f: np.ndarray) -> hmm.GaussianHMM:
        """Fit Hidden Markov Model for regime switching"""
        # Prepare features for regime detection
        features = np.column_stack([log_s2f, log_prices])
        
        # Fit Gaussian HMM
        model = hmm.GaussianHMM(n_components=self.n_regimes, covariance_type="full", 
                               n_iter=100, random_state=42)
        model.fit(features)
        
        return model
    
    def _perform_regime_analysis(self, historical_data: pd.DataFrame) -> RegimeSwitchingResult:
        """Perform regime switching analysis"""
        # Calculate S2F ratios and prepare data
        s2f_ratios = []
        log_prices = []
        
        for _, row in historical_data.iterrows():
            s2f = self.calculate_s2f_ratio(row['supply'], row['flow'])
            if s2f > 0 and s2f != float('inf') and row['price'] > 0:
                s2f_ratios.append(np.log10(s2f))
                log_prices.append(np.log10(row['price']))
        
        if len(s2f_ratios) < 50:
            # Return default result if insufficient data
            return RegimeSwitchingResult(
                current_regime=1,
                regime_probabilities=[0.33, 0.34, 0.33],
                regime_parameters={},
                regime_transitions=np.eye(3),
                regime_history=[1] * len(s2f_ratios),
                regime_descriptions={0: "Bear Market", 1: "Neutral", 2: "Bull Market"}
            )
        
        # Fit regime switching model
        features = np.column_stack([s2f_ratios, log_prices])
        self.regime_model = self._fit_regime_switching_model(np.array(log_prices), np.array(s2f_ratios))
        
        # Get regime predictions
        regime_states = self.regime_model.predict(features)
        regime_probs = self.regime_model.predict_proba(features)
        
        # Extract regime parameters
        regime_parameters = {}
        for i in range(self.n_regimes):
            regime_parameters[i] = {
                'mean_log_price': self.regime_model.means_[i][1],
                'mean_log_s2f': self.regime_model.means_[i][0],
                'volatility': np.sqrt(np.diag(self.regime_model.covars_[i])).mean()
            }
        
        return RegimeSwitchingResult(
            current_regime=regime_states[-1],
            regime_probabilities=regime_probs[-1].tolist(),
            regime_parameters=regime_parameters,
            regime_transitions=self.regime_model.transmat_,
            regime_history=regime_states.tolist(),
            regime_descriptions={0: "Bear Market", 1: "Neutral", 2: "Bull Market"}
        )
    
    def _perform_kalman_analysis(self, historical_data: pd.DataFrame) -> KalmanFilterResult:
        """Perform Kalman filtering for dynamic parameter estimation"""
        # Prepare data
        s2f_ratios = []
        log_prices = []
        
        for _, row in historical_data.iterrows():
            s2f = self.calculate_s2f_ratio(row['supply'], row['flow'])
            if s2f > 0 and s2f != float('inf') and row['price'] > 0:
                s2f_ratios.append(np.log10(s2f))
                log_prices.append(np.log10(row['price']))
        
        if len(s2f_ratios) < 10:
            # Return default result
            return KalmanFilterResult(
                filtered_states=np.zeros((4, len(s2f_ratios))),
                state_covariances=np.zeros((len(s2f_ratios), 4, 4)),
                innovation_residuals=np.zeros(len(s2f_ratios)),
                log_likelihood=0.0,
                adaptive_parameters={'slope': [1.0] * len(s2f_ratios), 'intercept': [0.0] * len(s2f_ratios)}
            )
        
        # Initialize Kalman filter
        kf = self._setup_kalman_filter(len(log_prices))
        
        # Initial state estimate (from simple linear regression)
        X = np.array(s2f_ratios).reshape(-1, 1)
        y = np.array(log_prices)
        simple_model = LinearRegression().fit(X, y)
        
        kf.x = np.array([simple_model.coef_[0], simple_model.intercept_, 0., 0.])
        
        # Run Kalman filter
        filtered_states = []
        state_covariances = []
        residuals = []
        log_likelihood = 0
        
        for i, (s2f_val, price_val) in enumerate(zip(s2f_ratios, log_prices)):
            # Predict
            kf.predict()
            
            # Update measurement matrix for current S2F value
            kf.H = np.array([[s2f_val, 1., 0., 0.]])
            
            # Update with observation
            kf.update(price_val)
            
            filtered_states.append(kf.x.copy())
            state_covariances.append(kf.P.copy())
            
            # Calculate innovation residual
            predicted_price = kf.H @ kf.x
            residual = price_val - predicted_price
            residuals.append(residual[0] if hasattr(residual, '__len__') else residual)
            
            # Update log likelihood
            log_likelihood += kf.log_likelihood
        
        filtered_states = np.array(filtered_states).T
        
        return KalmanFilterResult(
             filtered_states=filtered_states,
             state_covariances=np.array(state_covariances),
             innovation_residuals=np.array(residuals),
             log_likelihood=log_likelihood,
             adaptive_parameters={
                 'slope': filtered_states[0].tolist(),
                 'intercept': filtered_states[1].tolist()
             }
         )
    
    def _perform_monte_carlo_analysis(self, model_params: Dict[str, float], 
                                    current_s2f: float, projection_days: int = 365) -> MonteCarloResult:
        """Perform Monte Carlo simulations for uncertainty quantification"""
        # Parameter uncertainty (based on model standard error)
        slope_std = model_params.get('std_error', 0.1)
        intercept_std = model_params.get('std_error', 0.1)
        
        # Generate parameter samples
        slope_samples = np.random.normal(model_params['slope'], slope_std, self.monte_carlo_simulations)
        intercept_samples = np.random.normal(model_params['intercept'], intercept_std, self.monte_carlo_simulations)
        
        # S2F evolution scenarios (considering halving effects)
        s2f_scenarios = []
        for _ in range(self.monte_carlo_simulations):
            # Add randomness to S2F evolution
            s2f_path = [current_s2f]
            for day in range(1, projection_days + 1):
                # Model S2F growth with some randomness
                growth_rate = 0.0001 + np.random.normal(0, 0.00005)  # Small daily growth
                new_s2f = s2f_path[-1] * (1 + growth_rate)
                s2f_path.append(new_s2f)
            s2f_scenarios.append(s2f_path)
        
        # Price simulations
        price_scenarios = np.zeros((self.monte_carlo_simulations, projection_days + 1))
        
        for i in range(self.monte_carlo_simulations):
            slope = slope_samples[i]
            intercept = intercept_samples[i]
            s2f_path = s2f_scenarios[i]
            
            for j, s2f_val in enumerate(s2f_path):
                if s2f_val > 0:
                    log_s2f = np.log10(s2f_val)
                    log_price = slope * log_s2f + intercept
                    # Add price volatility
                    log_price += np.random.normal(0, 0.1)
                    price_scenarios[i, j] = 10 ** log_price
                else:
                    price_scenarios[i, j] = price_scenarios[i, j-1] if j > 0 else 1000
        
        # Calculate confidence bands
        confidence_levels = [50, 68, 95, 99]
        confidence_bands = {}
        
        for conf in confidence_levels:
            lower_percentile = (100 - conf) / 2
            upper_percentile = 100 - lower_percentile
            
            lower_band = np.percentile(price_scenarios[:, -1], lower_percentile)
            upper_band = np.percentile(price_scenarios[:, -1], upper_percentile)
            confidence_bands[f'{conf}%'] = (lower_band, upper_band)
        
        # Calculate VaR estimates
        final_prices = price_scenarios[:, -1]
        current_price = price_scenarios[:, 0].mean()
        returns = (final_prices - current_price) / current_price
        
        var_estimates = {
            'VaR_95': np.percentile(returns, 5),
            'VaR_99': np.percentile(returns, 1),
            'CVaR_95': returns[returns <= np.percentile(returns, 5)].mean(),
            'CVaR_99': returns[returns <= np.percentile(returns, 1)].mean()
        }
        
        # Expected returns
        expected_returns = {
            'mean_return': returns.mean(),
            'median_return': np.median(returns),
            'std_return': returns.std()
        }
        
        # Stress test scenarios
        stress_scenarios = {
            'bear_market': np.percentile(final_prices, 10),
            'recession': np.percentile(final_prices, 5),
            'black_swan': np.percentile(final_prices, 1),
            'bull_market': np.percentile(final_prices, 90),
            'euphoria': np.percentile(final_prices, 95)
        }
        
        return MonteCarloResult(
            price_scenarios=price_scenarios,
            confidence_bands=confidence_bands,
            var_estimates=var_estimates,
            expected_returns=expected_returns,
            scenario_probabilities=[1/self.monte_carlo_simulations] * self.monte_carlo_simulations,
            stress_test_results=stress_scenarios
        )
    
    def _calculate_advanced_metrics(self, historical_data: pd.DataFrame, 
                                  price_predictions: List[float]) -> Tuple[Dict, Dict, Dict]:
        """Calculate advanced risk and performance metrics"""
        prices = historical_data['price'].values
        returns = np.diff(np.log(prices))
        
        # Volatility metrics
        volatility_metrics = {
            'annualized_volatility': np.std(returns) * np.sqrt(365),
            'rolling_volatility_30d': pd.Series(returns).rolling(30).std().iloc[-1] * np.sqrt(365),
            'volatility_of_volatility': pd.Series(returns).rolling(30).std().std() * np.sqrt(365),
            'garch_volatility': self._estimate_garch_volatility(returns)
        }
        
        # Risk metrics
        risk_metrics = {
            'max_drawdown': self._calculate_max_drawdown(prices),
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': returns.mean() * 365 / abs(self._calculate_max_drawdown(prices)),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'var_95': np.percentile(returns, 5),
            'cvar_95': returns[returns <= np.percentile(returns, 5)].mean()
        }
        
        # Model diagnostics
        actual_log_prices = np.log10(prices[-len(price_predictions):])
        predicted_log_prices = np.log10(price_predictions)
        
        model_diagnostics = {
            'mean_absolute_error': mean_absolute_error(actual_log_prices, predicted_log_prices),
            'root_mean_squared_error': np.sqrt(mean_squared_error(actual_log_prices, predicted_log_prices)),
            'directional_accuracy': self._calculate_directional_accuracy(actual_log_prices, predicted_log_prices),
            'information_ratio': self._calculate_information_ratio(actual_log_prices, predicted_log_prices),
            'tracking_error': np.std(actual_log_prices - predicted_log_prices)
        }
        
        return volatility_metrics, risk_metrics, model_diagnostics
    
    def _estimate_garch_volatility(self, returns: np.ndarray) -> float:
        """Estimate GARCH(1,1) volatility"""
        try:
            # Simple GARCH(1,1) estimation
            mean_return = np.mean(returns)
            squared_residuals = (returns - mean_return) ** 2
            
            # GARCH parameters (simplified estimation)
            omega = np.var(squared_residuals) * 0.1
            alpha = 0.1
            beta = 0.8
            
            # Calculate conditional variance
            cond_var = [np.var(returns)]
            for i in range(1, len(squared_residuals)):
                var_t = omega + alpha * squared_residuals[i-1] + beta * cond_var[i-1]
                cond_var.append(var_t)
            
            return np.sqrt(cond_var[-1]) * np.sqrt(365)
        except:
            return np.std(returns) * np.sqrt(365)
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return np.min(drawdown)
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 365
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(365)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        excess_returns = returns - risk_free_rate / 365
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return np.inf
        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / downside_deviation * np.sqrt(365)
    
    def _calculate_directional_accuracy(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate directional accuracy"""
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))
        return np.mean(actual_direction == predicted_direction)
    
    def _calculate_information_ratio(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate information ratio"""
        tracking_error = actual - predicted
        return np.mean(tracking_error) / np.std(tracking_error) if np.std(tracking_error) > 0 else 0
    
    def analyze(self, 
               historical_data: pd.DataFrame,
               current_date: Optional[datetime] = None,
               projection_days: int = 365) -> S2FResult:
        """Perform comprehensive advanced Stock-to-Flow analysis
        
        Args:
            historical_data: DataFrame with columns ['date', 'price', 'supply', 'flow']
            current_date: Analysis date (defaults to today)
            projection_days: Number of days for future projections
        """
        if current_date is None:
            current_date = datetime.now()
        
        try:
            # Fit the S2F model
            model_params = self.fit_s2f_model(historical_data)
            
            # Calculate current S2F ratio
            if self.asset == "BTC":
                current_supply, current_flow = self.calculate_bitcoin_supply_schedule(current_date)
            else:
                # Use latest data point for other assets
                latest_data = historical_data.iloc[-1]
                current_supply = latest_data['supply']
                current_flow = latest_data['flow']
            
            current_s2f = self.calculate_s2f_ratio(current_supply, current_flow)
            
            # Predict current price
            predicted_price, confidence_interval = self.predict_price(current_s2f, model_params)
            
            # Calculate next halving (Bitcoin only)
            next_halving_date, days_to_halving = self.calculate_next_halving(current_date)
            
            # Generate future projections
            future_dates, future_s2f, future_prices = self.generate_future_projections(
                model_params, current_date
            )
            
            # Prepare historical S2F data
            historical_s2f = []
            for _, row in historical_data.iterrows():
                s2f = self.calculate_s2f_ratio(row['supply'], row['flow'])
                historical_s2f.append(s2f)
            
            # Advanced analytics (optional)
            regime_analysis = None
            monte_carlo_analysis = None
            kalman_analysis = None
            
            if self.enable_regime_switching:
                try:
                    regime_analysis = self._perform_regime_analysis(historical_data)
                except Exception as e:
                    logger.warning(f"Regime switching analysis failed: {e}")
            
            if self.enable_kalman_filter:
                try:
                    kalman_analysis = self._perform_kalman_analysis(historical_data)
                except Exception as e:
                    logger.warning(f"Kalman filter analysis failed: {e}")
            
            if self.enable_monte_carlo:
                try:
                    monte_carlo_analysis = self._perform_monte_carlo_analysis(
                        model_params, current_s2f, projection_days
                    )
                except Exception as e:
                    logger.warning(f"Monte Carlo analysis failed: {e}")
            
            # Calculate advanced metrics
            volatility_metrics, risk_metrics, model_diagnostics = self._calculate_advanced_metrics(
                historical_data, future_prices[:len(historical_data)]
            )
            
            # Calculate statistical measures
            prices = historical_data['price'].values
            returns = np.diff(np.log(prices))
            
            return S2FResult(
                # Core S2F metrics
                current_s2f_ratio=current_s2f,
                predicted_price=predicted_price,
                model_r_squared=model_params['r_squared'],
                confidence_interval=confidence_interval,
                next_halving_date=next_halving_date,
                days_to_halving=days_to_halving,
                historical_s2f=historical_s2f,
                price_predictions=future_prices,
                timestamps=future_dates,
                model_parameters=model_params,
                
                # Advanced analytics
                regime_analysis=regime_analysis,
                monte_carlo_analysis=monte_carlo_analysis,
                kalman_analysis=kalman_analysis,
                volatility_metrics=volatility_metrics,
                risk_metrics=risk_metrics,
                model_diagnostics=model_diagnostics,
                
                # Statistical measures
                sharpe_ratio=risk_metrics.get('sharpe_ratio'),
                max_drawdown=risk_metrics.get('max_drawdown'),
                volatility=volatility_metrics.get('annualized_volatility'),
                skewness=risk_metrics.get('skewness'),
                kurtosis=risk_metrics.get('kurtosis')
            )
            
        except Exception as e:
            logger.error(f"Error in advanced S2F analysis: {str(e)}")
            raise
    
    def get_model_insights(self, result: S2FResult) -> Dict[str, str]:
        """Generate comprehensive human-readable insights from advanced S2F analysis"""
        insights = {}
        
        # Model quality assessment with advanced diagnostics
        r_squared = result.model_r_squared
        if r_squared > 0.8:
            insights['model_quality'] = "Excellent fit - S2F model explains price movements well"
        elif r_squared > 0.6:
            insights['model_quality'] = "Good fit - S2F model has predictive power"
        elif r_squared > 0.4:
            insights['model_quality'] = "Moderate fit - S2F model shows some correlation"
        else:
            insights['model_quality'] = "Poor fit - S2F model may not be reliable"
        
        # Add model diagnostics if available
        if result.model_diagnostics:
            mae = result.model_diagnostics.get('mean_absolute_error', 0)
            dir_acc = result.model_diagnostics.get('directional_accuracy', 0)
            insights['model_quality'] += f" (MAE: {mae:.3f}, Directional Accuracy: {dir_acc:.1%})"
        
        # S2F ratio interpretation with regime context
        s2f = result.current_s2f_ratio
        if s2f > 100:
            insights['scarcity_level'] = "Extremely scarce - Very high S2F ratio indicates strong store of value properties"
        elif s2f > 50:
            insights['scarcity_level'] = "Highly scarce - S2F ratio comparable to precious metals"
        elif s2f > 20:
            insights['scarcity_level'] = "Moderately scarce - Decent store of value characteristics"
        else:
            insights['scarcity_level'] = "Low scarcity - High inflation rate relative to existing supply"
        
        # Regime analysis insights
        if result.regime_analysis:
            regime_desc = result.regime_analysis.regime_descriptions.get(
                result.regime_analysis.current_regime, "Unknown"
            )
            regime_prob = max(result.regime_analysis.regime_probabilities)
            insights['market_regime'] = f"Current regime: {regime_desc} (confidence: {regime_prob:.1%})"
        
        # Risk assessment
        if result.risk_metrics:
            sharpe = result.risk_metrics.get('sharpe_ratio', 0)
            max_dd = result.risk_metrics.get('max_drawdown', 0)
            if sharpe > 1.5:
                risk_assessment = "Excellent risk-adjusted returns"
            elif sharpe > 1.0:
                risk_assessment = "Good risk-adjusted returns"
            elif sharpe > 0.5:
                risk_assessment = "Moderate risk-adjusted returns"
            else:
                risk_assessment = "Poor risk-adjusted returns"
            
            insights['risk_assessment'] = f"{risk_assessment} (Sharpe: {sharpe:.2f}, Max DD: {max_dd:.1%})"
        
        # Volatility insights
        if result.volatility_metrics:
            vol = result.volatility_metrics.get('annualized_volatility', 0)
            if vol > 1.0:
                vol_level = "Extremely high volatility"
            elif vol > 0.6:
                vol_level = "High volatility"
            elif vol > 0.3:
                vol_level = "Moderate volatility"
            else:
                vol_level = "Low volatility"
            
            insights['volatility_analysis'] = f"{vol_level} ({vol:.1%} annualized)"
        
        # Monte Carlo insights
        if result.monte_carlo_analysis:
            var_95 = result.monte_carlo_analysis.var_estimates.get('VaR_95', 0)
            mean_return = result.monte_carlo_analysis.expected_returns.get('mean_return', 0)
            
            insights['monte_carlo_outlook'] = f"Expected return: {mean_return:.1%}, 95% VaR: {var_95:.1%}"
            
            # Stress test insights
            bear_scenario = result.monte_carlo_analysis.stress_test_results.get('bear_market', 0)
            bull_scenario = result.monte_carlo_analysis.stress_test_results.get('bull_market', 0)
            current_price = result.predicted_price
            
            bear_change = (bear_scenario - current_price) / current_price
            bull_change = (bull_scenario - current_price) / current_price
            
            insights['stress_scenarios'] = f"Bear case: {bear_change:.1%}, Bull case: {bull_change:.1%}"
        
        # Halving insights (Bitcoin only) with enhanced analysis
        if result.days_to_halving is not None:
            if result.days_to_halving < 365:
                insights['halving_impact'] = f"Next halving in {result.days_to_halving} days - Expect supply shock and potential price appreciation"
            else:
                insights['halving_impact'] = f"Next halving in {result.days_to_halving // 365} years - Long-term supply reduction trajectory"
            
            # Add S2F projection post-halving
            post_halving_s2f = result.current_s2f_ratio * 2  # Approximate doubling
            insights['halving_impact'] += f" (S2F ratio expected to reach ~{post_halving_s2f:.0f})"
        
        # Kalman filter insights
        if result.kalman_analysis:
            recent_slope = result.kalman_analysis.adaptive_parameters['slope'][-1]
            if recent_slope > result.model_parameters.get('slope', 1) * 1.1:
                insights['parameter_evolution'] = "Model parameters suggest strengthening S2F relationship"
            elif recent_slope < result.model_parameters.get('slope', 1) * 0.9:
                insights['parameter_evolution'] = "Model parameters suggest weakening S2F relationship"
            else:
                insights['parameter_evolution'] = "Model parameters remain stable"
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for Bitcoin
    dates = pd.date_range(start='2010-01-01', end='2024-01-01', freq='D')
    
    # Simulate Bitcoin supply (simplified)
    supply = []
    current_supply = 0
    blocks_per_day = 144  # Approximate
    reward = 50  # Initial reward
    
    for i, date in enumerate(dates):
        if i > 0 and i % (4 * 365) == 0:  # Halving every 4 years (simplified)
            reward = reward / 2
        
        daily_new_supply = blocks_per_day * reward
        current_supply += daily_new_supply
        supply.append(current_supply)
    
    # Simulate price data with S2F relationship + noise
    np.random.seed(42)
    s2f_ratios = []
    flow_data = []
    
    for i, s in enumerate(supply):
        if i == 0:
            s2f_ratios.append(1)
            flow_data.append(blocks_per_day * 50 * 365)  # Initial annual flow
        else:
            daily_flow = supply[i] - supply[i-1]
            annual_flow = daily_flow * 365
            s2f_ratio = s / annual_flow if annual_flow > 0 else s2f_ratios[-1]
            s2f_ratios.append(s2f_ratio)
            flow_data.append(annual_flow)
    
    # Price follows S2F with some noise and regime changes
    log_s2f = np.log(s2f_ratios)
    
    # Add regime-dependent noise
    regime_noise = np.random.normal(0, 0.3, len(log_s2f))
    market_cycles = np.sin(np.arange(len(log_s2f)) * 2 * np.pi / (4 * 365)) * 0.2
    
    log_price = 3.3 * log_s2f - 17.01 + regime_noise + market_cycles
    prices = np.exp(log_price)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'price': prices,
        'supply': supply,
        'flow': flow_data
    })
    
    print("Running Basic S2F Analysis...")
    # Initialize and run basic S2F model
    s2f_model = StockToFlowModel("BTC", enable_regime_switching=False, 
                                enable_kalman_filter=False, enable_monte_carlo=False)
    result = s2f_model.analyze(data)
    
    print("\n=== BASIC S2F ANALYSIS ===")
    print(f"Current S2F Ratio: {result.current_s2f_ratio:.2f}")
    print(f"Predicted Price: ${result.predicted_price:,.2f}")
    print(f"Model R²: {result.model_r_squared:.3f}")
    print(f"Days to next halving: {result.days_to_halving}")
    
    print("\nRunning Advanced S2F Analysis with all features...")
    # Initialize advanced S2F model with all features
    advanced_s2f_model = StockToFlowModel(
        "BTC",
        enable_regime_switching=True,
        enable_kalman_filter=True,
        enable_monte_carlo=True
    )
    
    advanced_result = advanced_s2f_model.analyze(data)
    
    print("\n=== ADVANCED S2F ANALYSIS ===")
    print(f"Current S2F Ratio: {advanced_result.current_s2f_ratio:.2f}")
    print(f"Predicted Price: ${advanced_result.predicted_price:,.2f}")
    print(f"Model R²: {advanced_result.model_r_squared:.3f}")
    
    # Display advanced metrics
    if advanced_result.risk_metrics:
        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio: {advanced_result.risk_metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {advanced_result.risk_metrics.get('max_drawdown', 0):.1%}")
        print(f"  VaR (95%): {advanced_result.risk_metrics.get('var_95', 0):.1%}")
    
    if advanced_result.volatility_metrics:
        print(f"\nVolatility Metrics:")
        print(f"  Annualized Volatility: {advanced_result.volatility_metrics.get('annualized_volatility', 0):.1%}")
        print(f"  Skewness: {advanced_result.skewness:.3f}")
        print(f"  Kurtosis: {advanced_result.kurtosis:.3f}")
    
    if advanced_result.regime_analysis:
        print(f"\nRegime Analysis:")
        print(f"  Current Regime: {advanced_result.regime_analysis.current_regime}")
        print(f"  Regime Probabilities: {[f'{p:.1%}' for p in advanced_result.regime_analysis.regime_probabilities]}")
    
    if advanced_result.monte_carlo_analysis:
        print(f"\nMonte Carlo Analysis:")
        print(f"  Expected Return: {advanced_result.monte_carlo_analysis.expected_returns.get('mean_return', 0):.1%}")
        conf_95 = advanced_result.monte_carlo_analysis.confidence_bands.get('95%', (0, 0))
        print(f"  95% Confidence Interval: [${conf_95[0]:,.0f}, ${conf_95[1]:,.0f}]")
    
    # Print comprehensive insights
    insights = advanced_s2f_model.get_model_insights(advanced_result)
    print("\n=== COMPREHENSIVE MODEL INSIGHTS ===")
    for key, value in insights.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n=== MODEL COMPARISON ===")
    print(f"Basic Model R²: {result.model_r_squared:.3f}")
    print(f"Advanced Model R²: {advanced_result.model_r_squared:.3f}")
    if result.model_r_squared > 0:
        improvement = ((advanced_result.model_r_squared - result.model_r_squared) / result.model_r_squared * 100)
        print(f"Improvement: {improvement:+.1f}%")
    
    print("\n=== FUTURE PROJECTIONS ===")
    print(f"Next 12 months price targets:")
    for i in range(0, min(12, len(advanced_result.price_predictions)), 3):
        date = advanced_result.timestamps[i]
        price = advanced_result.price_predictions[i]
        print(f"  {date.strftime('%Y-%m')}: ${price:,.0f}")