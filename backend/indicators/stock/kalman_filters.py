import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

@dataclass
class KalmanState:
    """Represents the state of a Kalman filter at a given time."""
    state_mean: np.ndarray
    state_covariance: np.ndarray
    timestamp: datetime
    log_likelihood: float = 0.0
    innovation: Optional[np.ndarray] = None
    innovation_covariance: Optional[np.ndarray] = None

@dataclass
class KalmanFilterResults:
    """Results from Kalman filter estimation."""
    filtered_states: List[KalmanState]
    smoothed_states: List[KalmanState]
    predicted_states: List[KalmanState]
    log_likelihood: float
    aic: float
    bic: float
    filtered_values: np.ndarray
    smoothed_values: np.ndarray
    predicted_values: np.ndarray
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]
    model_parameters: Dict[str, Any]
    
@dataclass
class AdaptiveKalmanResults:
    """Results from adaptive Kalman filter."""
    base_results: KalmanFilterResults
    adaptive_noise_variance: np.ndarray
    adaptation_history: List[Dict[str, float]]
    regime_probabilities: Optional[np.ndarray] = None
    detected_changepoints: List[int] = field(default_factory=list)
    
@dataclass
class EnsembleKalmanResults:
    """Results from ensemble Kalman filter."""
    ensemble_mean: np.ndarray
    ensemble_variance: np.ndarray
    individual_forecasts: List[np.ndarray]
    weights: np.ndarray
    ensemble_log_likelihood: float
    model_selection_criteria: Dict[str, float]
    
@dataclass
class ParticleFilterResults:
    """Results from particle filter."""
    particles: List[np.ndarray]
    weights: List[np.ndarray]
    effective_sample_sizes: List[float]
    resampling_times: List[int]
    filtered_estimates: np.ndarray
    confidence_intervals: Dict[str, np.ndarray]
    log_likelihood: float

class BaseKalmanFilter:
    """Base class for Kalman filter implementations."""
    
    def __init__(self, state_dim: int, obs_dim: int):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.reset()
        
    def reset(self):
        """Reset filter to initial state."""
        self.state_mean = np.zeros(self.state_dim)
        self.state_covariance = np.eye(self.state_dim)
        self.transition_matrix = np.eye(self.state_dim)
        self.observation_matrix = np.eye(self.obs_dim, self.state_dim)
        self.transition_noise = np.eye(self.state_dim)
        self.observation_noise = np.eye(self.obs_dim)
        self.log_likelihood = 0.0
        
    def predict(self) -> KalmanState:
        """Prediction step of Kalman filter."""
        # Predict state
        predicted_state = self.transition_matrix @ self.state_mean
        predicted_covariance = (
            self.transition_matrix @ self.state_covariance @ self.transition_matrix.T +
            self.transition_noise
        )
        
        return KalmanState(
            state_mean=predicted_state,
            state_covariance=predicted_covariance,
            timestamp=datetime.now()
        )
        
    def update(self, observation: np.ndarray, predicted_state: KalmanState) -> KalmanState:
        """Update step of Kalman filter."""
        # Innovation
        innovation = observation - self.observation_matrix @ predicted_state.state_mean
        innovation_covariance = (
            self.observation_matrix @ predicted_state.state_covariance @ self.observation_matrix.T +
            self.observation_noise
        )
        
        # Kalman gain
        kalman_gain = (
            predicted_state.state_covariance @ self.observation_matrix.T @
            linalg.inv(innovation_covariance)
        )
        
        # Update state
        updated_state = predicted_state.state_mean + kalman_gain @ innovation
        updated_covariance = (
            predicted_state.state_covariance -
            kalman_gain @ self.observation_matrix @ predicted_state.state_covariance
        )
        
        # Log likelihood
        log_likelihood = -0.5 * (
            np.log(2 * np.pi * linalg.det(innovation_covariance)) +
            innovation.T @ linalg.inv(innovation_covariance) @ innovation
        )
        
        self.state_mean = updated_state
        self.state_covariance = updated_covariance
        self.log_likelihood += log_likelihood
        
        return KalmanState(
            state_mean=updated_state,
            state_covariance=updated_covariance,
            timestamp=datetime.now(),
            log_likelihood=log_likelihood,
            innovation=innovation,
            innovation_covariance=innovation_covariance
        )

class LinearKalmanFilter(BaseKalmanFilter):
    """Standard linear Kalman filter for stock price estimation."""
    
    def __init__(self, model_type: str = 'local_level'):
        self.model_type = model_type
        
        if model_type == 'local_level':
            super().__init__(state_dim=1, obs_dim=1)
            self.setup_local_level()
        elif model_type == 'local_trend':
            super().__init__(state_dim=2, obs_dim=1)
            self.setup_local_trend()
        elif model_type == 'seasonal':
            super().__init__(state_dim=13, obs_dim=1)  # 12 seasonal + 1 level
            self.setup_seasonal()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def setup_local_level(self):
        """Setup local level model (random walk)."""
        self.transition_matrix = np.array([[1.0]])
        self.observation_matrix = np.array([[1.0]])
        self.transition_noise = np.array([[0.1]])
        self.observation_noise = np.array([[1.0]])
        
    def setup_local_trend(self):
        """Setup local trend model (random walk with drift)."""
        self.transition_matrix = np.array([
            [1.0, 1.0],
            [0.0, 1.0]
        ])
        self.observation_matrix = np.array([[1.0, 0.0]])
        self.transition_noise = np.array([
            [0.1, 0.0],
            [0.0, 0.01]
        ])
        self.observation_noise = np.array([[1.0]])
        
    def setup_seasonal(self, period: int = 12):
        """Setup seasonal model."""
        # State: [level, s1, s2, ..., s11] where s_i are seasonal components
        self.transition_matrix = np.zeros((13, 13))
        self.transition_matrix[0, 0] = 1.0  # Level
        
        # Seasonal transition
        self.transition_matrix[1, 1:12] = -1.0
        for i in range(2, 12):
            self.transition_matrix[i, i-1] = 1.0
        self.transition_matrix[12, 11] = 1.0
        
        self.observation_matrix = np.zeros((1, 13))
        self.observation_matrix[0, 0] = 1.0  # Level
        self.observation_matrix[0, 1] = 1.0  # Current seasonal
        
        self.transition_noise = np.eye(13) * 0.01
        self.transition_noise[0, 0] = 0.1  # Level noise
        self.observation_noise = np.array([[1.0]])
        
    def fit(self, data: np.ndarray, optimize_params: bool = True) -> KalmanFilterResults:
        """Fit Kalman filter to data."""
        if optimize_params:
            self._optimize_parameters(data)
            
        return self._run_filter(data)
        
    def _optimize_parameters(self, data: np.ndarray):
        """Optimize filter parameters using maximum likelihood."""
        def objective(params):
            self._set_parameters(params)
            try:
                results = self._run_filter(data)
                return -results.log_likelihood
            except:
                return 1e10
                
        # Initial parameters
        if self.model_type == 'local_level':
            initial_params = [0.1, 1.0]  # transition_noise, observation_noise
        elif self.model_type == 'local_trend':
            initial_params = [0.1, 0.01, 1.0]  # level_noise, trend_noise, obs_noise
        else:
            initial_params = [0.1] + [0.01] * 12 + [1.0]  # level + seasonal + obs
            
        bounds = [(1e-6, 10.0)] * len(initial_params)
        
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        self._set_parameters(result.x)
        
    def _set_parameters(self, params: np.ndarray):
        """Set filter parameters."""
        if self.model_type == 'local_level':
            self.transition_noise = np.array([[params[0]]])
            self.observation_noise = np.array([[params[1]]])
        elif self.model_type == 'local_trend':
            self.transition_noise = np.array([
                [params[0], 0.0],
                [0.0, params[1]]
            ])
            self.observation_noise = np.array([[params[2]]])
        else:  # seasonal
            self.transition_noise = np.diag([params[0]] + list(params[1:13]))
            self.observation_noise = np.array([[params[13]]])
            
    def _run_filter(self, data: np.ndarray) -> KalmanFilterResults:
        """Run forward and backward passes."""
        n = len(data)
        filtered_states = []
        predicted_states = []
        
        self.reset()
        
        # Forward pass
        for i, obs in enumerate(data):
            # Predict
            predicted_state = self.predict()
            predicted_states.append(predicted_state)
            
            # Update
            filtered_state = self.update(np.array([obs]), predicted_state)
            filtered_states.append(filtered_state)
            
        # Backward pass (smoothing)
        smoothed_states = self._smooth(filtered_states)
        
        # Extract values
        filtered_values = np.array([state.state_mean[0] for state in filtered_states])
        smoothed_values = np.array([state.state_mean[0] for state in smoothed_states])
        predicted_values = np.array([state.state_mean[0] for state in predicted_states])
        
        # Calculate metrics
        residuals = data - filtered_values
        standardized_residuals = residuals / np.sqrt([state.innovation_covariance[0, 0] for state in filtered_states])
        
        # Confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(filtered_states)
        
        # Model selection criteria
        k = len(self._get_parameters())
        aic = 2 * k - 2 * self.log_likelihood
        bic = k * np.log(n) - 2 * self.log_likelihood
        
        return KalmanFilterResults(
            filtered_states=filtered_states,
            smoothed_states=smoothed_states,
            predicted_states=predicted_states,
            log_likelihood=self.log_likelihood,
            aic=aic,
            bic=bic,
            filtered_values=filtered_values,
            smoothed_values=smoothed_values,
            predicted_values=predicted_values,
            residuals=residuals,
            standardized_residuals=standardized_residuals,
            confidence_intervals=confidence_intervals,
            model_parameters=self._get_parameters()
        )
        
    def _smooth(self, filtered_states: List[KalmanState]) -> List[KalmanState]:
        """Rauch-Tung-Striebel smoother."""
        n = len(filtered_states)
        smoothed_states = [None] * n
        
        # Initialize with last filtered state
        smoothed_states[-1] = filtered_states[-1]
        
        # Backward pass
        for i in range(n-2, -1, -1):
            # Predict next state
            predicted_mean = self.transition_matrix @ filtered_states[i].state_mean
            predicted_cov = (
                self.transition_matrix @ filtered_states[i].state_covariance @ self.transition_matrix.T +
                self.transition_noise
            )
            
            # Smoother gain
            smoother_gain = (
                filtered_states[i].state_covariance @ self.transition_matrix.T @
                linalg.inv(predicted_cov)
            )
            
            # Smooth
            smoothed_mean = (
                filtered_states[i].state_mean +
                smoother_gain @ (smoothed_states[i+1].state_mean - predicted_mean)
            )
            smoothed_cov = (
                filtered_states[i].state_covariance +
                smoother_gain @ (smoothed_states[i+1].state_covariance - predicted_cov) @ smoother_gain.T
            )
            
            smoothed_states[i] = KalmanState(
                state_mean=smoothed_mean,
                state_covariance=smoothed_cov,
                timestamp=filtered_states[i].timestamp
            )
            
        return smoothed_states
        
    def _calculate_confidence_intervals(self, states: List[KalmanState], confidence: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals."""
        z_score = 1.96 if confidence == 0.95 else 2.576  # 99%
        
        means = np.array([state.state_mean[0] for state in states])
        stds = np.array([np.sqrt(state.state_covariance[0, 0]) for state in states])
        
        return {
            'lower': means - z_score * stds,
            'upper': means + z_score * stds
        }
        
    def _get_parameters(self) -> Dict[str, Any]:
        """Get current filter parameters."""
        return {
            'transition_matrix': self.transition_matrix,
            'observation_matrix': self.observation_matrix,
            'transition_noise': self.transition_noise,
            'observation_noise': self.observation_noise,
            'model_type': self.model_type
        }
        
    def forecast(self, steps: int, last_state: KalmanState) -> Tuple[np.ndarray, np.ndarray]:
        """Generate forecasts."""
        forecasts = []
        variances = []
        
        current_mean = last_state.state_mean
        current_cov = last_state.state_covariance
        
        for _ in range(steps):
            # Predict
            current_mean = self.transition_matrix @ current_mean
            current_cov = (
                self.transition_matrix @ current_cov @ self.transition_matrix.T +
                self.transition_noise
            )
            
            # Observation
            forecast = self.observation_matrix @ current_mean
            variance = (
                self.observation_matrix @ current_cov @ self.observation_matrix.T +
                self.observation_noise
            )
            
            forecasts.append(forecast[0])
            variances.append(variance[0, 0])
            
        return np.array(forecasts), np.array(variances)

class AdaptiveKalmanFilter(LinearKalmanFilter):
    """Adaptive Kalman filter with time-varying parameters."""
    
    def __init__(self, model_type: str = 'local_level', adaptation_method: str = 'innovation'):
        super().__init__(model_type)
        self.adaptation_method = adaptation_method
        self.adaptation_window = 20
        self.forgetting_factor = 0.95
        
    def fit(self, data: np.ndarray) -> AdaptiveKalmanResults:
        """Fit adaptive Kalman filter."""
        n = len(data)
        filtered_states = []
        predicted_states = []
        adaptation_history = []
        adaptive_noise_variance = np.zeros(n)
        
        self.reset()
        
        for i, obs in enumerate(data):
            # Adapt parameters
            if i > self.adaptation_window:
                self._adapt_parameters(data[max(0, i-self.adaptation_window):i+1], i)
                
            # Store current noise variance
            adaptive_noise_variance[i] = self.observation_noise[0, 0]
            
            # Predict and update
            predicted_state = self.predict()
            predicted_states.append(predicted_state)
            
            filtered_state = self.update(np.array([obs]), predicted_state)
            filtered_states.append(filtered_state)
            
            # Store adaptation info
            adaptation_history.append({
                'observation_noise': self.observation_noise[0, 0],
                'transition_noise': self.transition_noise[0, 0] if self.model_type == 'local_level' else self.transition_noise[0, 0],
                'innovation': filtered_state.innovation[0] if filtered_state.innovation is not None else 0.0
            })
            
        # Run standard filter for comparison
        base_results = super()._run_filter(data)
        
        return AdaptiveKalmanResults(
            base_results=base_results,
            adaptive_noise_variance=adaptive_noise_variance,
            adaptation_history=adaptation_history
        )
        
    def _adapt_parameters(self, recent_data: np.ndarray, current_index: int):
        """Adapt filter parameters based on recent performance."""
        if self.adaptation_method == 'innovation':
            self._adapt_by_innovation(recent_data)
        elif self.adaptation_method == 'likelihood':
            self._adapt_by_likelihood(recent_data)
        elif self.adaptation_method == 'variance':
            self._adapt_by_variance(recent_data)
            
    def _adapt_by_innovation(self, recent_data: np.ndarray):
        """Adapt based on innovation sequence."""
        # Calculate recent innovations
        innovations = []
        temp_filter = LinearKalmanFilter(self.model_type)
        temp_filter.transition_noise = self.transition_noise.copy()
        temp_filter.observation_noise = self.observation_noise.copy()
        
        for obs in recent_data[:-1]:
            predicted = temp_filter.predict()
            updated = temp_filter.update(np.array([obs]), predicted)
            if updated.innovation is not None:
                innovations.append(updated.innovation[0])
                
        if innovations:
            innovation_var = np.var(innovations)
            # Adapt observation noise
            self.observation_noise[0, 0] = max(0.01, innovation_var * self.forgetting_factor)
            
    def _adapt_by_likelihood(self, recent_data: np.ndarray):
        """Adapt based on likelihood changes."""
        # Test different noise levels
        noise_levels = [0.1, 0.5, 1.0, 2.0, 5.0]
        best_likelihood = -np.inf
        best_noise = self.observation_noise[0, 0]
        
        for noise in noise_levels:
            temp_filter = LinearKalmanFilter(self.model_type)
            temp_filter.observation_noise = np.array([[noise]])
            temp_filter.transition_noise = self.transition_noise.copy()
            
            try:
                temp_results = temp_filter._run_filter(recent_data)
                if temp_results.log_likelihood > best_likelihood:
                    best_likelihood = temp_results.log_likelihood
                    best_noise = noise
            except:
                continue
                
        self.observation_noise[0, 0] = best_noise
        
    def _adapt_by_variance(self, recent_data: np.ndarray):
        """Adapt based on empirical variance."""
        if len(recent_data) > 1:
            empirical_var = np.var(np.diff(recent_data))
            self.transition_noise[0, 0] = max(0.001, empirical_var * 0.1)
            self.observation_noise[0, 0] = max(0.01, empirical_var * 0.5)

class EnsembleKalmanFilter:
    """Ensemble of Kalman filters for robust estimation."""
    
    def __init__(self, model_types: List[str] = None):
        if model_types is None:
            model_types = ['local_level', 'local_trend', 'seasonal']
            
        self.filters = [LinearKalmanFilter(model_type) for model_type in model_types]
        self.model_types = model_types
        self.weights = np.ones(len(self.filters)) / len(self.filters)
        
    def fit(self, data: np.ndarray) -> EnsembleKalmanResults:
        """Fit ensemble of filters."""
        individual_results = []
        individual_forecasts = []
        log_likelihoods = []
        
        # Fit each filter
        for filter_obj in self.filters:
            try:
                results = filter_obj.fit(data, optimize_params=True)
                individual_results.append(results)
                individual_forecasts.append(results.filtered_values)
                log_likelihoods.append(results.log_likelihood)
            except:
                # Fallback for failed filters
                individual_results.append(None)
                individual_forecasts.append(np.zeros_like(data))
                log_likelihoods.append(-np.inf)
                
        # Calculate weights based on performance
        log_likelihoods = np.array(log_likelihoods)
        valid_indices = np.isfinite(log_likelihoods)
        
        if np.any(valid_indices):
            # Softmax weighting
            exp_ll = np.exp(log_likelihoods[valid_indices] - np.max(log_likelihoods[valid_indices]))
            weights = np.zeros(len(self.filters))
            weights[valid_indices] = exp_ll / np.sum(exp_ll)
        else:
            weights = np.ones(len(self.filters)) / len(self.filters)
            
        self.weights = weights
        
        # Ensemble predictions
        individual_forecasts = np.array(individual_forecasts)
        ensemble_mean = np.average(individual_forecasts, axis=0, weights=weights)
        
        # Ensemble variance (including model uncertainty)
        ensemble_variance = np.zeros_like(ensemble_mean)
        for i, forecast in enumerate(individual_forecasts):
            ensemble_variance += weights[i] * (forecast - ensemble_mean) ** 2
            
        # Model selection criteria
        ensemble_log_likelihood = np.sum(weights * log_likelihoods)
        model_selection_criteria = {
            'ensemble_aic': -2 * ensemble_log_likelihood + 2 * len(self.filters),
            'ensemble_bic': -2 * ensemble_log_likelihood + len(self.filters) * np.log(len(data)),
            'individual_aics': [r.aic if r else np.inf for r in individual_results],
            'individual_bics': [r.bic if r else np.inf for r in individual_results]
        }
        
        return EnsembleKalmanResults(
            ensemble_mean=ensemble_mean,
            ensemble_variance=ensemble_variance,
            individual_forecasts=individual_forecasts.tolist(),
            weights=weights,
            ensemble_log_likelihood=ensemble_log_likelihood,
            model_selection_criteria=model_selection_criteria
        )
        
    def forecast(self, steps: int, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate ensemble forecasts."""
        individual_forecasts = []
        individual_variances = []
        
        for i, filter_obj in enumerate(self.filters):
            try:
                results = filter_obj.fit(data, optimize_params=False)
                last_state = results.filtered_states[-1]
                forecast, variance = filter_obj.forecast(steps, last_state)
                individual_forecasts.append(forecast)
                individual_variances.append(variance)
            except:
                individual_forecasts.append(np.zeros(steps))
                individual_variances.append(np.ones(steps))
                
        # Ensemble forecast
        individual_forecasts = np.array(individual_forecasts)
        ensemble_forecast = np.average(individual_forecasts, axis=0, weights=self.weights)
        
        # Ensemble variance
        ensemble_variance = np.zeros(steps)
        for i, (forecast, variance) in enumerate(zip(individual_forecasts, individual_variances)):
            ensemble_variance += self.weights[i] * (variance + (forecast - ensemble_forecast) ** 2)
            
        return ensemble_forecast, ensemble_variance

class ParticleFilter:
    """Particle filter for non-linear state estimation."""
    
    def __init__(self, n_particles: int = 1000, state_dim: int = 1):
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.particles = None
        self.weights = None
        self.effective_sample_size_threshold = n_particles / 2
        
    def initialize_particles(self, initial_state: np.ndarray, initial_covariance: np.ndarray):
        """Initialize particle cloud."""
        self.particles = np.random.multivariate_normal(
            initial_state, initial_covariance, self.n_particles
        )
        self.weights = np.ones(self.n_particles) / self.n_particles
        
    def predict_particles(self, transition_function, noise_covariance: np.ndarray):
        """Predict particles forward in time."""
        for i in range(self.n_particles):
            self.particles[i] = transition_function(self.particles[i])
            self.particles[i] += np.random.multivariate_normal(
                np.zeros(self.state_dim), noise_covariance
            )
            
    def update_weights(self, observation: np.ndarray, likelihood_function):
        """Update particle weights based on observation."""
        for i in range(self.n_particles):
            self.weights[i] *= likelihood_function(observation, self.particles[i])
            
        # Normalize weights
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.n_particles) / self.n_particles
            
    def resample(self) -> bool:
        """Resample particles if effective sample size is too low."""
        effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        
        if effective_sample_size < self.effective_sample_size_threshold:
            # Systematic resampling
            indices = self._systematic_resample()
            self.particles = self.particles[indices]
            self.weights = np.ones(self.n_particles) / self.n_particles
            return True
            
        return False
        
    def _systematic_resample(self) -> np.ndarray:
        """Systematic resampling algorithm."""
        n = self.n_particles
        positions = (np.arange(n) + np.random.random()) / n
        
        indices = np.zeros(n, dtype=int)
        cumulative_sum = np.cumsum(self.weights)
        
        i, j = 0, 0
        while i < n:
            if positions[i] < cumulative_sum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
                
        return indices
        
    def get_estimate(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get weighted mean and covariance estimate."""
        mean = np.average(self.particles, axis=0, weights=self.weights)
        
        # Weighted covariance
        diff = self.particles - mean
        covariance = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.n_particles):
            covariance += self.weights[i] * np.outer(diff[i], diff[i])
            
        return mean, covariance
        
    def fit(self, data: np.ndarray, transition_function, likelihood_function,
            initial_state: np.ndarray, initial_covariance: np.ndarray,
            process_noise: np.ndarray) -> ParticleFilterResults:
        """Fit particle filter to data."""
        n = len(data)
        
        # Initialize
        self.initialize_particles(initial_state, initial_covariance)
        
        # Storage
        all_particles = []
        all_weights = []
        effective_sample_sizes = []
        resampling_times = []
        filtered_estimates = []
        
        log_likelihood = 0.0
        
        for i, obs in enumerate(data):
            # Predict
            self.predict_particles(transition_function, process_noise)
            
            # Update
            self.update_weights(obs, likelihood_function)
            
            # Store
            all_particles.append(self.particles.copy())
            all_weights.append(self.weights.copy())
            
            # Estimate
            mean, _ = self.get_estimate()
            filtered_estimates.append(mean[0] if self.state_dim == 1 else mean)
            
            # Effective sample size
            ess = 1.0 / np.sum(self.weights ** 2)
            effective_sample_sizes.append(ess)
            
            # Log likelihood
            log_likelihood += np.log(np.mean(self.weights) + 1e-10)
            
            # Resample if needed
            if self.resample():
                resampling_times.append(i)
                
        # Confidence intervals
        confidence_intervals = self._calculate_particle_confidence_intervals(
            all_particles, all_weights
        )
        
        return ParticleFilterResults(
            particles=all_particles,
            weights=all_weights,
            effective_sample_sizes=effective_sample_sizes,
            resampling_times=resampling_times,
            filtered_estimates=np.array(filtered_estimates),
            confidence_intervals=confidence_intervals,
            log_likelihood=log_likelihood
        )
        
    def _calculate_particle_confidence_intervals(self, particles_history: List[np.ndarray],
                                               weights_history: List[np.ndarray],
                                               confidence: float = 0.95) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals from particle distributions."""
        n = len(particles_history)
        lower_percentile = (1 - confidence) / 2 * 100
        upper_percentile = (1 + confidence) / 2 * 100
        
        lower_bounds = []
        upper_bounds = []
        
        for particles, weights in zip(particles_history, weights_history):
            # Weighted percentiles
            sorted_indices = np.argsort(particles[:, 0] if self.state_dim > 1 else particles)
            sorted_weights = weights[sorted_indices]
            cumulative_weights = np.cumsum(sorted_weights)
            
            lower_idx = np.searchsorted(cumulative_weights, lower_percentile / 100)
            upper_idx = np.searchsorted(cumulative_weights, upper_percentile / 100)
            
            if self.state_dim == 1:
                lower_bounds.append(particles[sorted_indices[lower_idx]])
                upper_bounds.append(particles[sorted_indices[upper_idx]])
            else:
                lower_bounds.append(particles[sorted_indices[lower_idx], 0])
                upper_bounds.append(particles[sorted_indices[upper_idx], 0])
                
        return {
            'lower': np.array(lower_bounds),
            'upper': np.array(upper_bounds)
        }

class KalmanFilterAnalyzer:
    """Main class for comprehensive Kalman filter analysis."""
    
    def __init__(self):
        self.linear_filter = None
        self.adaptive_filter = None
        self.ensemble_filter = None
        self.particle_filter = None
        
    def analyze_stock_data(self, data: Union[pd.DataFrame, np.ndarray],
                          price_column: str = 'close',
                          methods: List[str] = None) -> Dict[str, Any]:
        """Comprehensive Kalman filter analysis of stock data."""
        if methods is None:
            methods = ['linear', 'adaptive', 'ensemble']
            
        # Prepare data
        if isinstance(data, pd.DataFrame):
            prices = data[price_column].values
            timestamps = data.index if hasattr(data, 'index') else None
        else:
            prices = data
            timestamps = None
            
        # Log transform for better stationarity
        log_prices = np.log(prices)
        
        results = {}
        
        # Linear Kalman Filter
        if 'linear' in methods:
            results['linear'] = self._analyze_linear_kalman(log_prices)
            
        # Adaptive Kalman Filter
        if 'adaptive' in methods:
            results['adaptive'] = self._analyze_adaptive_kalman(log_prices)
            
        # Ensemble Kalman Filter
        if 'ensemble' in methods:
            results['ensemble'] = self._analyze_ensemble_kalman(log_prices)
            
        # Particle Filter
        if 'particle' in methods:
            results['particle'] = self._analyze_particle_filter(log_prices)
            
        # Model comparison
        results['comparison'] = self._compare_models(results)
        
        # Generate insights
        results['insights'] = self._generate_insights(results, prices)
        
        return results
        
    def _analyze_linear_kalman(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze with linear Kalman filters."""
        models = ['local_level', 'local_trend', 'seasonal']
        results = {}
        
        for model_type in models:
            try:
                filter_obj = LinearKalmanFilter(model_type)
                filter_results = filter_obj.fit(data, optimize_params=True)
                
                # Generate forecasts
                last_state = filter_results.filtered_states[-1]
                forecasts, forecast_vars = filter_obj.forecast(30, last_state)
                
                results[model_type] = {
                    'results': filter_results,
                    'forecasts': forecasts,
                    'forecast_variances': forecast_vars,
                    'filter': filter_obj
                }
            except Exception as e:
                results[model_type] = {'error': str(e)}
                
        return results
        
    def _analyze_adaptive_kalman(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze with adaptive Kalman filters."""
        adaptation_methods = ['innovation', 'likelihood', 'variance']
        results = {}
        
        for method in adaptation_methods:
            try:
                filter_obj = AdaptiveKalmanFilter('local_level', method)
                filter_results = filter_obj.fit(data)
                
                results[method] = {
                    'results': filter_results,
                    'filter': filter_obj
                }
            except Exception as e:
                results[method] = {'error': str(e)}
                
        return results
        
    def _analyze_ensemble_kalman(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze with ensemble Kalman filter."""
        try:
            filter_obj = EnsembleKalmanFilter()
            filter_results = filter_obj.fit(data)
            
            # Generate forecasts
            forecasts, forecast_vars = filter_obj.forecast(30, data)
            
            return {
                'results': filter_results,
                'forecasts': forecasts,
                'forecast_variances': forecast_vars,
                'filter': filter_obj
            }
        except Exception as e:
            return {'error': str(e)}
            
    def _analyze_particle_filter(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze with particle filter."""
        try:
            # Define transition and likelihood functions
            def transition_function(state):
                return state  # Random walk
                
            def likelihood_function(observation, state):
                return np.exp(-0.5 * (observation - state) ** 2)
                
            filter_obj = ParticleFilter(n_particles=1000, state_dim=1)
            
            # Initial conditions
            initial_state = np.array([data[0]])
            initial_covariance = np.array([[1.0]])
            process_noise = np.array([[0.1]])
            
            filter_results = filter_obj.fit(
                data, transition_function, likelihood_function,
                initial_state, initial_covariance, process_noise
            )
            
            return {
                'results': filter_results,
                'filter': filter_obj
            }
        except Exception as e:
            return {'error': str(e)}
            
    def _compare_models(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different Kalman filter models."""
        comparison = {
            'model_rankings': {},
            'performance_metrics': {},
            'best_model': None
        }
        
        # Collect performance metrics
        models_performance = []
        
        # Linear models
        if 'linear' in results:
            for model_type, model_results in results['linear'].items():
                if 'results' in model_results:
                    models_performance.append({
                        'name': f'linear_{model_type}',
                        'aic': model_results['results'].aic,
                        'bic': model_results['results'].bic,
                        'log_likelihood': model_results['results'].log_likelihood
                    })
                    
        # Ensemble model
        if 'ensemble' in results and 'results' in results['ensemble']:
            models_performance.append({
                'name': 'ensemble',
                'aic': results['ensemble']['results'].model_selection_criteria['ensemble_aic'],
                'bic': results['ensemble']['results'].model_selection_criteria['ensemble_bic'],
                'log_likelihood': results['ensemble']['results'].ensemble_log_likelihood
            })
            
        # Rank models
        if models_performance:
            # Sort by AIC (lower is better)
            sorted_by_aic = sorted(models_performance, key=lambda x: x['aic'])
            comparison['model_rankings']['by_aic'] = [model['name'] for model in sorted_by_aic]
            
            # Sort by BIC (lower is better)
            sorted_by_bic = sorted(models_performance, key=lambda x: x['bic'])
            comparison['model_rankings']['by_bic'] = [model['name'] for model in sorted_by_bic]
            
            # Best model (by AIC)
            comparison['best_model'] = sorted_by_aic[0]['name']
            comparison['performance_metrics'] = models_performance
            
        return comparison
        
    def _generate_insights(self, results: Dict[str, Any], original_prices: np.ndarray) -> Dict[str, Any]:
        """Generate insights from Kalman filter analysis."""
        insights = {
            'trend_analysis': {},
            'volatility_analysis': {},
            'forecast_analysis': {},
            'model_recommendations': {},
            'risk_assessment': {}
        }
        
        # Trend analysis
        if 'linear' in results and 'local_trend' in results['linear']:
            trend_results = results['linear']['local_trend']
            if 'results' in trend_results:
                trend_states = trend_results['results'].filtered_states
                if len(trend_states) > 0 and len(trend_states[0].state_mean) > 1:
                    trends = [state.state_mean[1] for state in trend_states]
                    insights['trend_analysis'] = {
                        'average_trend': np.mean(trends),
                        'trend_volatility': np.std(trends),
                        'current_trend': trends[-1] if trends else 0,
                        'trend_direction': 'upward' if trends[-1] > 0 else 'downward' if trends[-1] < 0 else 'sideways'
                    }
                    
        # Volatility analysis
        if 'adaptive' in results:
            for method, adaptive_results in results['adaptive'].items():
                if 'results' in adaptive_results:
                    noise_variance = adaptive_results['results'].adaptive_noise_variance
                    insights['volatility_analysis'][method] = {
                        'average_volatility': np.mean(noise_variance),
                        'volatility_trend': 'increasing' if noise_variance[-1] > np.mean(noise_variance) else 'decreasing',
                        'volatility_regime_changes': len(adaptive_results['results'].detected_changepoints)
                    }
                    
        # Forecast analysis
        best_model = results.get('comparison', {}).get('best_model')
        if best_model and 'linear' in results:
            model_type = best_model.replace('linear_', '')
            if model_type in results['linear'] and 'forecasts' in results['linear'][model_type]:
                forecasts = results['linear'][model_type]['forecasts']
                current_price = original_prices[-1]
                forecast_prices = np.exp(np.log(current_price) + forecasts)
                
                insights['forecast_analysis'] = {
                    'short_term_direction': 'up' if forecast_prices[4] > current_price else 'down',
                    'medium_term_direction': 'up' if forecast_prices[14] > current_price else 'down',
                    'long_term_direction': 'up' if forecast_prices[-1] > current_price else 'down',
                    'expected_return_1w': (forecast_prices[4] / current_price - 1) * 100,
                    'expected_return_1m': (forecast_prices[-1] / current_price - 1) * 100
                }
                
        # Model recommendations
        if 'comparison' in results:
            insights['model_recommendations'] = {
                'best_overall': results['comparison'].get('best_model', 'unknown'),
                'recommended_for_trending': 'linear_local_trend',
                'recommended_for_volatile': 'adaptive_innovation',
                'recommended_for_robust': 'ensemble'
            }
            
        # Risk assessment
        if 'ensemble' in results and 'results' in results['ensemble']:
            ensemble_variance = results['ensemble']['results'].ensemble_variance
            insights['risk_assessment'] = {
                'current_uncertainty': float(ensemble_variance[-1]),
                'average_uncertainty': float(np.mean(ensemble_variance)),
                'uncertainty_trend': 'increasing' if ensemble_variance[-1] > np.mean(ensemble_variance) else 'decreasing',
                'risk_level': 'high' if ensemble_variance[-1] > np.percentile(ensemble_variance, 75) else 'medium' if ensemble_variance[-1] > np.percentile(ensemble_variance, 25) else 'low'
            }
            
        return insights
        
    def plot_results(self, data: np.ndarray, results: Dict[str, Any], 
                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot Kalman filter results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Kalman Filter Analysis Results', fontsize=16)
        
        # Original data
        axes[0, 0].plot(data, label='Original Data', alpha=0.7)
        
        # Linear filters
        if 'linear' in results:
            for model_type, model_results in results['linear'].items():
                if 'results' in model_results:
                    filtered_values = model_results['results'].filtered_values
                    axes[0, 0].plot(filtered_values, label=f'Linear {model_type}', alpha=0.8)
                    
        axes[0, 0].set_title('Linear Kalman Filters')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Ensemble results
        if 'ensemble' in results and 'results' in results['ensemble']:
            ensemble_results = results['ensemble']['results']
            axes[0, 1].plot(data, label='Original Data', alpha=0.7)
            axes[0, 1].plot(ensemble_results.ensemble_mean, label='Ensemble Mean', linewidth=2)
            
            # Confidence intervals
            std = np.sqrt(ensemble_results.ensemble_variance)
            axes[0, 1].fill_between(
                range(len(data)),
                ensemble_results.ensemble_mean - 2*std,
                ensemble_results.ensemble_mean + 2*std,
                alpha=0.3, label='95% Confidence'
            )
            
        axes[0, 1].set_title('Ensemble Kalman Filter')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Adaptive noise variance
        if 'adaptive' in results:
            for method, adaptive_results in results['adaptive'].items():
                if 'results' in adaptive_results:
                    noise_var = adaptive_results['results'].adaptive_noise_variance
                    axes[1, 0].plot(noise_var, label=f'Adaptive {method}')
                    
        axes[1, 0].set_title('Adaptive Noise Variance')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Residuals analysis
        if 'linear' in results and 'local_level' in results['linear']:
            if 'results' in results['linear']['local_level']:
                residuals = results['linear']['local_level']['results'].standardized_residuals
                axes[1, 1].plot(residuals, alpha=0.7)
                axes[1, 1].axhline(y=0, color='r', linestyle='--')
                axes[1, 1].axhline(y=2, color='r', linestyle=':', alpha=0.5)
                axes[1, 1].axhline(y=-2, color='r', linestyle=':', alpha=0.5)
                
        axes[1, 1].set_title('Standardized Residuals')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n = 200
    true_trend = 0.01
    true_level = 100
    
    # Generate synthetic stock price data
    prices = [true_level]
    for i in range(1, n):
        prices.append(prices[-1] * (1 + true_trend + np.random.normal(0, 0.02)))
    
    prices = np.array(prices)
    
    # Create analyzer
    analyzer = KalmanFilterAnalyzer()
    
    # Run analysis
    results = analyzer.analyze_stock_data(prices, methods=['linear', 'adaptive', 'ensemble'])
    
    # Print insights
    print("Kalman Filter Analysis Results:")
    print("=" * 50)
    
    if 'insights' in results:
        insights = results['insights']
        
        print("\nTrend Analysis:")
        if 'trend_analysis' in insights:
            for key, value in insights['trend_analysis'].items():
                print(f"  {key}: {value}")
                
        print("\nForecast Analysis:")
        if 'forecast_analysis' in insights:
            for key, value in insights['forecast_analysis'].items():
                print(f"  {key}: {value}")
                
        print("\nModel Recommendations:")
        if 'model_recommendations' in insights:
            for key, value in insights['model_recommendations'].items():
                print(f"  {key}: {value}")
                
        print("\nRisk Assessment:")
        if 'risk_assessment' in insights:
            for key, value in insights['risk_assessment'].items():
                print(f"  {key}: {value}")
    
    # Plot results
    fig = analyzer.plot_results(np.log(prices), results)
    plt.show()