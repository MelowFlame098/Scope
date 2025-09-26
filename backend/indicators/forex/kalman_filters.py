from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ForexData:
    """Data structure for forex market data"""
    timestamp: List[datetime]
    exchange_rate: List[float]
    interest_rate_domestic: List[float]
    interest_rate_foreign: List[float]
    inflation_domestic: List[float]
    inflation_foreign: List[float]
    volatility: List[float]
    volume: List[float]
    economic_indicators: Dict[str, List[float]]
    
@dataclass
class KalmanState:
    """Kalman filter state representation"""
    state_mean: np.ndarray
    state_covariance: np.ndarray
    log_likelihood: float
    innovation: np.ndarray
    innovation_covariance: np.ndarray
    kalman_gain: np.ndarray
    
@dataclass
class KalmanFilterResult:
    """Results from Kalman filter analysis"""
    filtered_states: List[KalmanState]
    smoothed_states: List[KalmanState]
    predicted_values: List[float]
    prediction_intervals: List[Tuple[float, float]]
    log_likelihood: float
    aic: float
    bic: float
    mse: float
    mae: float
    model_parameters: Dict[str, Any]
    regime_probabilities: Optional[List[float]] = None
    
@dataclass
class ForexKalmanResults:
    """Comprehensive forex Kalman filter results"""
    linear_kalman: KalmanFilterResult
    adaptive_kalman: KalmanFilterResult
    regime_switching: KalmanFilterResult
    ensemble_result: KalmanFilterResult
    model_comparison: Dict[str, Dict[str, float]]
    trading_signals: List[str]
    risk_metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    
class LinearKalmanFilter:
    """Linear Kalman Filter for forex rate estimation"""
    
    def __init__(self, state_dim: int = 2, obs_dim: int = 1):
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.reset_parameters()
        
    def reset_parameters(self):
        """Reset filter parameters"""
        # State transition matrix (random walk with drift)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        
        # Observation matrix
        self.H = np.array([[1.0, 0.0]])
        
        # Process noise covariance
        self.Q = np.eye(self.state_dim) * 0.01
        
        # Observation noise covariance
        self.R = np.array([[0.1]])
        
        # Initial state
        self.x0 = np.zeros(self.state_dim)
        
        # Initial covariance
        self.P0 = np.eye(self.state_dim)
        
    def fit(self, observations: np.ndarray, 
            optimize_params: bool = True) -> KalmanFilterResult:
        """Fit Kalman filter to observations"""
        
        if optimize_params:
            self._optimize_parameters(observations)
            
        # Forward pass (filtering)
        filtered_states = self._forward_pass(observations)
        
        # Backward pass (smoothing)
        smoothed_states = self._backward_pass(filtered_states)
        
        # Calculate predictions and metrics
        predictions = [state.state_mean[0] for state in filtered_states[1:]]
        
        # Prediction intervals
        pred_intervals = []
        for state in filtered_states[1:]:
            std = np.sqrt(state.state_covariance[0, 0])
            pred_intervals.append((state.state_mean[0] - 1.96*std,
                                 state.state_mean[0] + 1.96*std))
        
        # Calculate metrics
        log_likelihood = sum(state.log_likelihood for state in filtered_states)
        n_params = self.state_dim**2 + self.obs_dim**2 + self.state_dim + self.obs_dim
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(len(observations))
        
        if len(predictions) > 0:
            mse = mean_squared_error(observations[1:], predictions)
            mae = mean_absolute_error(observations[1:], predictions)
        else:
            mse = mae = float('inf')
        
        return KalmanFilterResult(
            filtered_states=filtered_states,
            smoothed_states=smoothed_states,
            predicted_values=predictions,
            prediction_intervals=pred_intervals,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            mse=mse,
            mae=mae,
            model_parameters={
                'F': self.F.tolist(),
                'H': self.H.tolist(),
                'Q': self.Q.tolist(),
                'R': self.R.tolist()
            }
        )
    
    def _forward_pass(self, observations: np.ndarray) -> List[KalmanState]:
        """Forward pass (filtering)"""
        states = []
        x = self.x0.copy()
        P = self.P0.copy()
        
        for i, y in enumerate(observations):
            if i == 0:
                # Initialize with first observation
                x[0] = y
                states.append(KalmanState(
                    state_mean=x.copy(),
                    state_covariance=P.copy(),
                    log_likelihood=0.0,
                    innovation=np.array([0.0]),
                    innovation_covariance=self.R.copy(),
                    kalman_gain=np.zeros((self.state_dim, self.obs_dim))
                ))
                continue
            
            # Predict
            x_pred = self.F @ x
            P_pred = self.F @ P @ self.F.T + self.Q
            
            # Update
            y_pred = self.H @ x_pred
            innovation = np.array([y]) - y_pred
            S = self.H @ P_pred @ self.H.T + self.R
            K = P_pred @ self.H.T @ np.linalg.inv(S)
            
            x = x_pred + K @ innovation
            P = P_pred - K @ self.H @ P_pred
            
            # Log likelihood
            log_lik = -0.5 * (np.log(2*np.pi) + np.log(np.linalg.det(S)) + 
                              innovation.T @ np.linalg.inv(S) @ innovation)[0]
            
            states.append(KalmanState(
                state_mean=x.copy(),
                state_covariance=P.copy(),
                log_likelihood=log_lik,
                innovation=innovation,
                innovation_covariance=S,
                kalman_gain=K
            ))
            
        return states
    
    def _backward_pass(self, filtered_states: List[KalmanState]) -> List[KalmanState]:
        """Backward pass (smoothing)"""
        n = len(filtered_states)
        smoothed_states = [None] * n
        
        # Initialize with last filtered state
        smoothed_states[-1] = filtered_states[-1]
        
        for i in range(n-2, -1, -1):
            x_filt = filtered_states[i].state_mean
            P_filt = filtered_states[i].state_covariance
            x_pred = self.F @ x_filt
            P_pred = self.F @ P_filt @ self.F.T + self.Q
            
            A = P_filt @ self.F.T @ np.linalg.inv(P_pred)
            x_smooth = x_filt + A @ (smoothed_states[i+1].state_mean - x_pred)
            P_smooth = P_filt + A @ (smoothed_states[i+1].state_covariance - P_pred) @ A.T
            
            smoothed_states[i] = KalmanState(
                state_mean=x_smooth,
                state_covariance=P_smooth,
                log_likelihood=filtered_states[i].log_likelihood,
                innovation=filtered_states[i].innovation,
                innovation_covariance=filtered_states[i].innovation_covariance,
                kalman_gain=filtered_states[i].kalman_gain
            )
            
        return smoothed_states
    
    def _optimize_parameters(self, observations: np.ndarray):
        """Optimize filter parameters using MLE"""
        def objective(params):
            try:
                # Unpack parameters
                q_diag = np.exp(params[:self.state_dim])
                r_val = np.exp(params[self.state_dim])
                
                # Update parameters
                self.Q = np.diag(q_diag)
                self.R = np.array([[r_val]])
                
                # Calculate negative log likelihood
                states = self._forward_pass(observations)
                return -sum(state.log_likelihood for state in states)
            except:
                return 1e10
        
        # Initial parameters
        initial_params = np.concatenate([
            np.log(np.diag(self.Q)),
            np.log(np.diag(self.R))
        ])
        
        # Optimize
        result = minimize(objective, initial_params, method='L-BFGS-B')
        
        if result.success:
            # Update with optimized parameters
            q_diag = np.exp(result.x[:self.state_dim])
            r_val = np.exp(result.x[self.state_dim])
            self.Q = np.diag(q_diag)
            self.R = np.array([[r_val]])

class AdaptiveKalmanFilter:
    """Adaptive Kalman Filter with time-varying parameters"""
    
    def __init__(self, adaptation_method: str = 'innovation'):
        self.adaptation_method = adaptation_method
        self.base_filter = LinearKalmanFilter()
        
    def fit(self, observations: np.ndarray, 
            economic_indicators: Optional[np.ndarray] = None) -> KalmanFilterResult:
        """Fit adaptive Kalman filter"""
        
        if self.adaptation_method == 'innovation':
            return self._innovation_adaptive(observations)
        elif self.adaptation_method == 'economic':
            return self._economic_adaptive(observations, economic_indicators)
        else:
            return self._variance_adaptive(observations)
    
    def _innovation_adaptive(self, observations: np.ndarray) -> KalmanFilterResult:
        """Innovation-based adaptation"""
        states = []
        x = self.base_filter.x0.copy()
        P = self.base_filter.P0.copy()
        Q = self.base_filter.Q.copy()
        R = self.base_filter.R.copy()
        
        # Adaptation parameters
        alpha = 0.95  # Forgetting factor
        window_size = 10
        innovations = []
        
        for i, y in enumerate(observations):
            if i == 0:
                x[0] = y
                states.append(KalmanState(
                    state_mean=x.copy(),
                    state_covariance=P.copy(),
                    log_likelihood=0.0,
                    innovation=np.array([0.0]),
                    innovation_covariance=R.copy(),
                    kalman_gain=np.zeros((2, 1))
                ))
                continue
            
            # Predict
            x_pred = self.base_filter.F @ x
            P_pred = self.base_filter.F @ P @ self.base_filter.F.T + Q
            
            # Update
            y_pred = self.base_filter.H @ x_pred
            innovation = np.array([y]) - y_pred
            S = self.base_filter.H @ P_pred @ self.base_filter.H.T + R
            K = P_pred @ self.base_filter.H.T @ np.linalg.inv(S)
            
            x = x_pred + K @ innovation
            P = P_pred - K @ self.base_filter.H @ P_pred
            
            # Store innovation
            innovations.append(innovation[0])
            
            # Adapt noise parameters
            if len(innovations) >= window_size:
                recent_innovations = innovations[-window_size:]
                innovation_var = np.var(recent_innovations)
                
                # Adapt R based on innovation variance
                R = np.array([[alpha * R[0,0] + (1-alpha) * innovation_var]])
                
                # Adapt Q based on innovation magnitude
                if abs(innovation[0]) > 2 * np.sqrt(S[0,0]):
                    Q = Q * 1.1  # Increase process noise
                else:
                    Q = Q * 0.99  # Decrease process noise
            
            # Log likelihood
            log_lik = -0.5 * (np.log(2*np.pi) + np.log(np.linalg.det(S)) + 
                              innovation.T @ np.linalg.inv(S) @ innovation)[0]
            
            states.append(KalmanState(
                state_mean=x.copy(),
                state_covariance=P.copy(),
                log_likelihood=log_lik,
                innovation=innovation,
                innovation_covariance=S,
                kalman_gain=K
            ))
        
        # Create result
        predictions = [state.state_mean[0] for state in states[1:]]
        pred_intervals = []
        for state in states[1:]:
            std = np.sqrt(state.state_covariance[0, 0])
            pred_intervals.append((state.state_mean[0] - 1.96*std,
                                 state.state_mean[0] + 1.96*std))
        
        log_likelihood = sum(state.log_likelihood for state in states)
        n_params = 6  # Approximate number of parameters
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(len(observations))
        
        if len(predictions) > 0:
            mse = mean_squared_error(observations[1:], predictions)
            mae = mean_absolute_error(observations[1:], predictions)
        else:
            mse = mae = float('inf')
        
        return KalmanFilterResult(
            filtered_states=states,
            smoothed_states=states,  # No smoothing for adaptive
            predicted_values=predictions,
            prediction_intervals=pred_intervals,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            mse=mse,
            mae=mae,
            model_parameters={
                'adaptation_method': self.adaptation_method,
                'final_Q': Q.tolist(),
                'final_R': R.tolist()
            }
        )
    
    def _economic_adaptive(self, observations: np.ndarray, 
                          economic_indicators: Optional[np.ndarray]) -> KalmanFilterResult:
        """Economic indicator-based adaptation"""
        if economic_indicators is None:
            return self._innovation_adaptive(observations)
        
        # Use economic indicators to adapt noise parameters
        states = []
        x = self.base_filter.x0.copy()
        P = self.base_filter.P0.copy()
        
        for i, (y, econ) in enumerate(zip(observations, economic_indicators)):
            # Adapt noise based on economic uncertainty
            uncertainty_factor = 1 + abs(econ - np.mean(economic_indicators[:i+1]))
            Q = self.base_filter.Q * uncertainty_factor
            R = self.base_filter.R * uncertainty_factor
            
            if i == 0:
                x[0] = y
                states.append(KalmanState(
                    state_mean=x.copy(),
                    state_covariance=P.copy(),
                    log_likelihood=0.0,
                    innovation=np.array([0.0]),
                    innovation_covariance=R.copy(),
                    kalman_gain=np.zeros((2, 1))
                ))
                continue
            
            # Standard Kalman filter steps with adapted noise
            x_pred = self.base_filter.F @ x
            P_pred = self.base_filter.F @ P @ self.base_filter.F.T + Q
            
            y_pred = self.base_filter.H @ x_pred
            innovation = np.array([y]) - y_pred
            S = self.base_filter.H @ P_pred @ self.base_filter.H.T + R
            K = P_pred @ self.base_filter.H.T @ np.linalg.inv(S)
            
            x = x_pred + K @ innovation
            P = P_pred - K @ self.base_filter.H @ P_pred
            
            log_lik = -0.5 * (np.log(2*np.pi) + np.log(np.linalg.det(S)) + 
                              innovation.T @ np.linalg.inv(S) @ innovation)[0]
            
            states.append(KalmanState(
                state_mean=x.copy(),
                state_covariance=P.copy(),
                log_likelihood=log_lik,
                innovation=innovation,
                innovation_covariance=S,
                kalman_gain=K
            ))
        
        # Create result (similar to innovation adaptive)
        predictions = [state.state_mean[0] for state in states[1:]]
        pred_intervals = []
        for state in states[1:]:
            std = np.sqrt(state.state_covariance[0, 0])
            pred_intervals.append((state.state_mean[0] - 1.96*std,
                                 state.state_mean[0] + 1.96*std))
        
        log_likelihood = sum(state.log_likelihood for state in states)
        n_params = 6
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(len(observations))
        
        if len(predictions) > 0:
            mse = mean_squared_error(observations[1:], predictions)
            mae = mean_absolute_error(observations[1:], predictions)
        else:
            mse = mae = float('inf')
        
        return KalmanFilterResult(
            filtered_states=states,
            smoothed_states=states,
            predicted_values=predictions,
            prediction_intervals=pred_intervals,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            mse=mse,
            mae=mae,
            model_parameters={
                'adaptation_method': self.adaptation_method,
                'economic_adaptation': True
            }
        )
    
    def _variance_adaptive(self, observations: np.ndarray) -> KalmanFilterResult:
        """Variance-based adaptation"""
        # Similar to innovation adaptive but focuses on variance changes
        return self._innovation_adaptive(observations)

class RegimeSwitchingKalmanFilter:
    """Regime-switching Kalman Filter for forex markets"""
    
    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.filters = [LinearKalmanFilter() for _ in range(n_regimes)]
        
    def fit(self, observations: np.ndarray) -> KalmanFilterResult:
        """Fit regime-switching Kalman filter"""
        
        # Initialize regime probabilities
        regime_probs = np.ones((len(observations), self.n_regimes)) / self.n_regimes
        transition_matrix = np.ones((self.n_regimes, self.n_regimes)) / self.n_regimes
        
        # EM algorithm for parameter estimation
        for iteration in range(10):  # Max iterations
            # E-step: Calculate regime probabilities
            regime_probs = self._calculate_regime_probabilities(
                observations, transition_matrix)
            
            # M-step: Update parameters
            transition_matrix = self._update_transition_matrix(regime_probs)
            self._update_filter_parameters(observations, regime_probs)
        
        # Final filtering with estimated parameters
        states, predictions = self._regime_switching_filter(
            observations, regime_probs, transition_matrix)
        
        # Calculate metrics
        pred_intervals = []
        for state in states[1:]:
            std = np.sqrt(state.state_covariance[0, 0])
            pred_intervals.append((state.state_mean[0] - 1.96*std,
                                 state.state_mean[0] + 1.96*std))
        
        log_likelihood = sum(state.log_likelihood for state in states)
        n_params = self.n_regimes * 6 + self.n_regimes * (self.n_regimes - 1)
        aic = -2 * log_likelihood + 2 * n_params
        bic = -2 * log_likelihood + n_params * np.log(len(observations))
        
        if len(predictions) > 0:
            mse = mean_squared_error(observations[1:], predictions)
            mae = mean_absolute_error(observations[1:], predictions)
        else:
            mse = mae = float('inf')
        
        return KalmanFilterResult(
            filtered_states=states,
            smoothed_states=states,
            predicted_values=predictions,
            prediction_intervals=pred_intervals,
            log_likelihood=log_likelihood,
            aic=aic,
            bic=bic,
            mse=mse,
            mae=mae,
            model_parameters={
                'n_regimes': self.n_regimes,
                'transition_matrix': transition_matrix.tolist()
            },
            regime_probabilities=regime_probs[:, 0].tolist()
        )
    
    def _calculate_regime_probabilities(self, observations: np.ndarray, 
                                       transition_matrix: np.ndarray) -> np.ndarray:
        """Calculate regime probabilities using forward-backward algorithm"""
        n_obs = len(observations)
        regime_probs = np.zeros((n_obs, self.n_regimes))
        
        # Forward pass
        regime_probs[0] = 1.0 / self.n_regimes
        
        for t in range(1, n_obs):
            for j in range(self.n_regimes):
                # Calculate likelihood for regime j
                likelihood = self._calculate_likelihood(observations[t], j)
                
                # Update probability
                regime_probs[t, j] = likelihood * np.sum(
                    regime_probs[t-1] * transition_matrix[:, j])
            
            # Normalize
            regime_probs[t] /= np.sum(regime_probs[t])
        
        return regime_probs
    
    def _calculate_likelihood(self, observation: float, regime: int) -> float:
        """Calculate observation likelihood for given regime"""
        # Simplified likelihood calculation
        # In practice, this would use the Kalman filter likelihood
        return stats.norm.pdf(observation, loc=0, scale=1)
    
    def _update_transition_matrix(self, regime_probs: np.ndarray) -> np.ndarray:
        """Update transition matrix based on regime probabilities"""
        transition_matrix = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(self.n_regimes):
            for j in range(self.n_regimes):
                numerator = np.sum(regime_probs[:-1, i] * regime_probs[1:, j])
                denominator = np.sum(regime_probs[:-1, i])
                transition_matrix[i, j] = numerator / (denominator + 1e-8)
        
        # Normalize rows
        for i in range(self.n_regimes):
            transition_matrix[i] /= np.sum(transition_matrix[i])
        
        return transition_matrix
    
    def _update_filter_parameters(self, observations: np.ndarray, 
                                 regime_probs: np.ndarray):
        """Update Kalman filter parameters for each regime"""
        for regime in range(self.n_regimes):
            # Weight observations by regime probability
            weights = regime_probs[:, regime]
            
            # Update filter parameters (simplified)
            # In practice, this would involve weighted MLE
            self.filters[regime].Q *= (1 + np.mean(weights))
            self.filters[regime].R *= (1 + np.var(weights))
    
    def _regime_switching_filter(self, observations: np.ndarray, 
                                regime_probs: np.ndarray,
                                transition_matrix: np.ndarray) -> Tuple[List[KalmanState], List[float]]:
        """Apply regime-switching Kalman filter"""
        states = []
        predictions = []
        
        # Initialize
        x = np.zeros(2)
        P = np.eye(2)
        
        for t, y in enumerate(observations):
            if t == 0:
                x[0] = y
                states.append(KalmanState(
                    state_mean=x.copy(),
                    state_covariance=P.copy(),
                    log_likelihood=0.0,
                    innovation=np.array([0.0]),
                    innovation_covariance=np.array([[1.0]]),
                    kalman_gain=np.zeros((2, 1))
                ))
                continue
            
            # Weighted combination of regime-specific filters
            x_combined = np.zeros(2)
            P_combined = np.zeros((2, 2))
            total_likelihood = 0
            
            for regime in range(self.n_regimes):
                # Apply regime-specific filter
                filter_result = self.filters[regime].fit(observations[:t+1])
                regime_state = filter_result.filtered_states[-1]
                
                weight = regime_probs[t, regime]
                x_combined += weight * regime_state.state_mean
                P_combined += weight * regime_state.state_covariance
                total_likelihood += weight * regime_state.log_likelihood
            
            states.append(KalmanState(
                state_mean=x_combined,
                state_covariance=P_combined,
                log_likelihood=total_likelihood,
                innovation=np.array([y - x_combined[0]]),
                innovation_covariance=np.array([[np.sqrt(P_combined[0, 0])]]),
                kalman_gain=np.zeros((2, 1))
            ))
            
            predictions.append(x_combined[0])
        
        return states, predictions

class ForexKalmanAnalyzer:
    """Comprehensive Forex Kalman Filter Analysis"""
    
    def __init__(self):
        self.linear_filter = LinearKalmanFilter()
        self.adaptive_filter = AdaptiveKalmanFilter()
        self.regime_filter = RegimeSwitchingKalmanFilter()
        
    def analyze(self, forex_data: ForexData) -> ForexKalmanResults:
        """Perform comprehensive Kalman filter analysis"""
        
        exchange_rates = np.array(forex_data.exchange_rate)
        
        # Linear Kalman Filter
        linear_result = self.linear_filter.fit(exchange_rates)
        
        # Adaptive Kalman Filter
        economic_indicators = None
        if forex_data.economic_indicators:
            # Use first economic indicator as adaptation signal
            indicator_name = list(forex_data.economic_indicators.keys())[0]
            economic_indicators = np.array(forex_data.economic_indicators[indicator_name])
        
        adaptive_result = self.adaptive_filter.fit(exchange_rates, economic_indicators)
        
        # Regime-Switching Kalman Filter
        regime_result = self.regime_filter.fit(exchange_rates)
        
        # Ensemble result (weighted combination)
        ensemble_result = self._create_ensemble_result(
            [linear_result, adaptive_result, regime_result],
            ['linear', 'adaptive', 'regime_switching']
        )
        
        # Model comparison
        model_comparison = self._compare_models({
            'Linear': linear_result,
            'Adaptive': adaptive_result,
            'Regime-Switching': regime_result,
            'Ensemble': ensemble_result
        })
        
        # Generate trading signals
        trading_signals = self._generate_trading_signals(
            exchange_rates, ensemble_result)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(
            exchange_rates, ensemble_result)
        
        # Generate insights and recommendations
        insights = self._generate_insights(
            forex_data, ensemble_result, model_comparison)
        recommendations = self._generate_recommendations(
            model_comparison, risk_metrics)
        
        return ForexKalmanResults(
            linear_kalman=linear_result,
            adaptive_kalman=adaptive_result,
            regime_switching=regime_result,
            ensemble_result=ensemble_result,
            model_comparison=model_comparison,
            trading_signals=trading_signals,
            risk_metrics=risk_metrics,
            insights=insights,
            recommendations=recommendations
        )
    
    def _create_ensemble_result(self, results: List[KalmanFilterResult], 
                               names: List[str]) -> KalmanFilterResult:
        """Create ensemble result from multiple models"""
        
        # Weight models by inverse AIC (better models get higher weight)
        aics = [result.aic for result in results]
        weights = [1/aic for aic in aics]
        weights = np.array(weights) / np.sum(weights)
        
        # Weighted predictions
        n_predictions = len(results[0].predicted_values)
        ensemble_predictions = np.zeros(n_predictions)
        
        for i, (result, weight) in enumerate(zip(results, weights)):
            if len(result.predicted_values) == n_predictions:
                ensemble_predictions += weight * np.array(result.predicted_values)
        
        # Ensemble metrics
        ensemble_mse = np.mean([result.mse * weight for result, weight in zip(results, weights)])
        ensemble_mae = np.mean([result.mae * weight for result, weight in zip(results, weights)])
        ensemble_log_lik = np.sum([result.log_likelihood * weight for result, weight in zip(results, weights)])
        
        return KalmanFilterResult(
            filtered_states=results[0].filtered_states,  # Use best model's states
            smoothed_states=results[0].smoothed_states,
            predicted_values=ensemble_predictions.tolist(),
            prediction_intervals=results[0].prediction_intervals,
            log_likelihood=ensemble_log_lik,
            aic=2 * len(weights) - 2 * ensemble_log_lik,
            bic=len(weights) * np.log(n_predictions) - 2 * ensemble_log_lik,
            mse=ensemble_mse,
            mae=ensemble_mae,
            model_parameters={
                'ensemble_weights': weights.tolist(),
                'component_models': names
            }
        )
    
    def _compare_models(self, models: Dict[str, KalmanFilterResult]) -> Dict[str, Dict[str, float]]:
        """Compare model performance"""
        comparison = {}
        
        for name, result in models.items():
            comparison[name] = {
                'AIC': result.aic,
                'BIC': result.bic,
                'MSE': result.mse,
                'MAE': result.mae,
                'Log_Likelihood': result.log_likelihood
            }
        
        return comparison
    
    def _generate_trading_signals(self, exchange_rates: np.ndarray, 
                                 result: KalmanFilterResult) -> List[str]:
        """Generate trading signals based on Kalman filter predictions"""
        signals = []
        predictions = result.predicted_values
        
        for i in range(1, len(predictions)):
            if i >= len(exchange_rates):
                break
                
            current_rate = exchange_rates[i]
            predicted_rate = predictions[i-1] if i-1 < len(predictions) else current_rate
            
            # Signal based on prediction vs actual
            if predicted_rate > current_rate * 1.001:  # 0.1% threshold
                signals.append('BUY')
            elif predicted_rate < current_rate * 0.999:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        return signals
    
    def _calculate_risk_metrics(self, exchange_rates: np.ndarray, 
                               result: KalmanFilterResult) -> Dict[str, float]:
        """Calculate risk metrics"""
        predictions = np.array(result.predicted_values)
        actual = exchange_rates[1:len(predictions)+1]
        
        # Prediction errors
        errors = actual - predictions
        
        return {
            'prediction_volatility': np.std(predictions),
            'error_volatility': np.std(errors),
            'max_error': np.max(np.abs(errors)),
            'error_skewness': stats.skew(errors),
            'error_kurtosis': stats.kurtosis(errors),
            'hit_rate': np.mean(np.sign(np.diff(actual)) == np.sign(np.diff(predictions))),
            'sharpe_ratio': np.mean(np.diff(predictions)) / (np.std(np.diff(predictions)) + 1e-8)
        }
    
    def _generate_insights(self, forex_data: ForexData, 
                          result: KalmanFilterResult,
                          model_comparison: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate analytical insights"""
        insights = []
        
        # Model performance insights
        best_model = min(model_comparison.keys(), 
                        key=lambda x: model_comparison[x]['AIC'])
        insights.append(f"Best performing model: {best_model}")
        
        # Volatility insights
        if hasattr(result, 'regime_probabilities') and result.regime_probabilities:
            high_vol_periods = np.sum(np.array(result.regime_probabilities) > 0.7)
            insights.append(f"High volatility regime detected in {high_vol_periods} periods")
        
        # Prediction accuracy
        if result.mse < 0.001:
            insights.append("High prediction accuracy achieved")
        elif result.mse > 0.01:
            insights.append("Prediction accuracy could be improved")
        
        # Economic factor influence
        if forex_data.economic_indicators:
            insights.append("Economic indicators incorporated in adaptive filtering")
        
        return insights
    
    def _generate_recommendations(self, model_comparison: Dict[str, Dict[str, float]],
                                 risk_metrics: Dict[str, float]) -> List[str]:
        """Generate trading recommendations"""
        recommendations = []
        
        # Model selection recommendation
        best_model = min(model_comparison.keys(), 
                        key=lambda x: model_comparison[x]['AIC'])
        recommendations.append(f"Use {best_model} model for primary analysis")
        
        # Risk management
        if risk_metrics['error_volatility'] > 0.02:
            recommendations.append("Implement strict risk management due to high prediction uncertainty")
        
        if risk_metrics['hit_rate'] > 0.6:
            recommendations.append("Model shows good directional accuracy for trend following")
        elif risk_metrics['hit_rate'] < 0.4:
            recommendations.append("Consider contrarian strategies due to low directional accuracy")
        
        # Volatility-based recommendations
        if risk_metrics['prediction_volatility'] > 0.05:
            recommendations.append("High volatility environment - consider shorter holding periods")
        
        return recommendations
    
    def plot_results(self, forex_data: ForexData, results: ForexKalmanResults):
        """Plot comprehensive analysis results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Exchange rates and predictions
        ax1 = axes[0, 0]
        timestamps = forex_data.timestamp
        exchange_rates = forex_data.exchange_rate
        
        ax1.plot(timestamps, exchange_rates, label='Actual', alpha=0.7)
        
        if len(results.ensemble_result.predicted_values) > 0:
            pred_timestamps = timestamps[1:len(results.ensemble_result.predicted_values)+1]
            ax1.plot(pred_timestamps, results.ensemble_result.predicted_values, 
                    label='Ensemble Prediction', alpha=0.8)
        
        ax1.set_title('Exchange Rate Predictions')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Exchange Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Model comparison
        ax2 = axes[0, 1]
        models = list(results.model_comparison.keys())
        aics = [results.model_comparison[model]['AIC'] for model in models]
        
        bars = ax2.bar(models, aics)
        ax2.set_title('Model Comparison (AIC)')
        ax2.set_xlabel('Model')
        ax2.set_ylabel('AIC')
        ax2.tick_params(axis='x', rotation=45)
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Plot 3: Prediction intervals
        ax3 = axes[1, 0]
        if len(results.ensemble_result.prediction_intervals) > 0:
            pred_timestamps = timestamps[1:len(results.ensemble_result.prediction_intervals)+1]
            intervals = results.ensemble_result.prediction_intervals
            lower_bounds = [interval[0] for interval in intervals]
            upper_bounds = [interval[1] for interval in intervals]
            
            ax3.fill_between(pred_timestamps, lower_bounds, upper_bounds, 
                           alpha=0.3, label='95% Confidence Interval')
            ax3.plot(pred_timestamps, results.ensemble_result.predicted_values, 
                    label='Prediction', color='red')
            ax3.plot(timestamps[1:len(intervals)+1], exchange_rates[1:len(intervals)+1], 
                    label='Actual', alpha=0.7)
        
        ax3.set_title('Prediction Intervals')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Exchange Rate')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Regime probabilities (if available)
        ax4 = axes[1, 1]
        if (hasattr(results.regime_switching, 'regime_probabilities') and 
            results.regime_switching.regime_probabilities):
            regime_probs = results.regime_switching.regime_probabilities
            ax4.plot(timestamps[:len(regime_probs)], regime_probs, 
                    label='High Volatility Regime Probability')
            ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
            ax4.set_title('Regime Switching Probabilities')
            ax4.set_xlabel('Time')
            ax4.set_ylabel('Probability')
            ax4.legend()
        else:
            # Plot residuals instead
            if len(results.ensemble_result.predicted_values) > 0:
                residuals = (np.array(exchange_rates[1:len(results.ensemble_result.predicted_values)+1]) - 
                           np.array(results.ensemble_result.predicted_values))
                ax4.plot(timestamps[1:len(residuals)+1], residuals)
                ax4.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                ax4.set_title('Prediction Residuals')
                ax4.set_xlabel('Time')
                ax4.set_ylabel('Residual')
        
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, forex_data: ForexData, results: ForexKalmanResults) -> str:
        """Generate comprehensive analysis report"""
        report = []
        report.append("=== FOREX KALMAN FILTER ANALYSIS REPORT ===")
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Data Period: {forex_data.timestamp[0]} to {forex_data.timestamp[-1]}")
        report.append(f"Number of Observations: {len(forex_data.exchange_rate)}")
        report.append("")
        
        # Model Performance
        report.append("MODEL PERFORMANCE COMPARISON:")
        for model, metrics in results.model_comparison.items():
            report.append(f"{model}:")
            report.append(f"  AIC: {metrics['AIC']:.4f}")
            report.append(f"  BIC: {metrics['BIC']:.4f}")
            report.append(f"  MSE: {metrics['MSE']:.6f}")
            report.append(f"  MAE: {metrics['MAE']:.6f}")
            report.append("")
        
        # Risk Metrics
        report.append("RISK METRICS:")
        for metric, value in results.risk_metrics.items():
            report.append(f"{metric}: {value:.4f}")
        report.append("")
        
        # Trading Signals Summary
        if results.trading_signals:
            signal_counts = {signal: results.trading_signals.count(signal) 
                           for signal in set(results.trading_signals)}
            report.append("TRADING SIGNALS SUMMARY:")
            for signal, count in signal_counts.items():
                percentage = (count / len(results.trading_signals)) * 100
                report.append(f"{signal}: {count} ({percentage:.1f}%)")
            report.append("")
        
        # Insights
        report.append("KEY INSIGHTS:")
        for insight in results.insights:
            report.append(f"• {insight}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        for recommendation in results.recommendations:
            report.append(f"• {recommendation}")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Generate sample forex data
    np.random.seed(42)
    n_points = 100
    
    timestamps = [datetime.now() - timedelta(days=n_points-i) for i in range(n_points)]
    
    # Simulate exchange rate with trend and volatility
    base_rate = 1.2
    trend = np.linspace(0, 0.1, n_points)
    noise = np.random.normal(0, 0.02, n_points)
    exchange_rate = base_rate + trend + noise
    
    # Simulate other economic data
    interest_rate_domestic = np.random.normal(0.02, 0.005, n_points)
    interest_rate_foreign = np.random.normal(0.015, 0.005, n_points)
    inflation_domestic = np.random.normal(0.025, 0.01, n_points)
    inflation_foreign = np.random.normal(0.02, 0.01, n_points)
    volatility = np.random.exponential(0.02, n_points)
    volume = np.random.lognormal(10, 0.5, n_points)
    
    # Economic indicators
    economic_indicators = {
        'gdp_growth': np.random.normal(0.02, 0.01, n_points),
        'unemployment': np.random.normal(0.05, 0.02, n_points),
        'trade_balance': np.random.normal(0, 1000, n_points)
    }
    
    # Create forex data object
    forex_data = ForexData(
        timestamp=timestamps,
        exchange_rate=exchange_rate.tolist(),
        interest_rate_domestic=interest_rate_domestic.tolist(),
        interest_rate_foreign=interest_rate_foreign.tolist(),
        inflation_domestic=inflation_domestic.tolist(),
        inflation_foreign=inflation_foreign.tolist(),
        volatility=volatility.tolist(),
        volume=volume.tolist(),
        economic_indicators=economic_indicators
    )
    
    # Initialize analyzer
    analyzer = ForexKalmanAnalyzer()
    
    # Run analysis
    print("Running Forex Kalman Filter Analysis...")
    results = analyzer.analyze(forex_data)
    
    # Print summary
    print("\n=== ANALYSIS SUMMARY ===")
    print(f"Best Model (by AIC): {min(results.model_comparison.keys(), key=lambda x: results.model_comparison[x]['AIC'])}")
    print(f"Ensemble MSE: {results.ensemble_result.mse:.6f}")
    print(f"Ensemble MAE: {results.ensemble_result.mae:.6f}")
    print(f"Hit Rate: {results.risk_metrics['hit_rate']:.3f}")
    
    print("\nKey Insights:")
    for insight in results.insights:
        print(f"• {insight}")
    
    print("\nRecommendations:")
    for recommendation in results.recommendations:
        print(f"• {recommendation}")
    
    # Generate and save report
    report = analyzer.generate_report(forex_data, results)
    print("\n" + "="*50)
    print(report)
    
    # Plot results
    try:
        analyzer.plot_results(forex_data, results)
    except Exception as e:
        print(f"Plotting failed: {e}")
    
    print("\nAnalysis completed successfully!")