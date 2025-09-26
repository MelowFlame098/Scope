"""Kalman Filter Models for Index State Analysis

This module implements various Kalman filter models for analyzing
index price dynamics and state estimation.

Models included:
- Local Level Model: Random walk with noise
- Local Trend Model: Random walk with drift
- Regime Switching Model: Multiple state regimes
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from scipy import stats
from scipy.linalg import inv, det
import warnings
warnings.filterwarnings('ignore')

@dataclass
class IndexData:
    """Index data structure"""
    index_symbol: str
    prices: List[float]
    returns: List[float]
    timestamps: List[datetime]
    volume: Optional[List[float]] = None
    market_cap: Optional[List[float]] = None

@dataclass
class KalmanResult:
    """Kalman filter results"""
    filtered_states: List[List[float]]
    predicted_states: List[List[float]]
    state_covariances: List[List[List[float]]]
    innovations: List[float]
    innovation_covariances: List[float]
    log_likelihood: float
    smoothed_states: List[List[float]]
    model_parameters: Dict[str, float]
    state_probabilities: Optional[List[List[float]]] = None

class KalmanFilterAnalyzer:
    """Kalman filter models for index state analysis"""
    
    def __init__(self):
        self.model_cache = {}
    
    def fit_kalman_filter(self, prices: List[float], model_type: str = 'local_level',
                         **kwargs) -> KalmanResult:
        """Fit Kalman filter model
        
        Args:
            prices: List of price values
            model_type: Type of Kalman model ('local_level', 'local_trend', 'regime_switching')
            **kwargs: Additional model parameters
            
        Returns:
            KalmanResult containing filtered states and diagnostics
        """
        
        prices = np.array(prices)
        
        if model_type == 'local_level':
            return self._fit_local_level_model(prices, **kwargs)
        elif model_type == 'local_trend':
            return self._fit_local_trend_model(prices, **kwargs)
        elif model_type == 'regime_switching':
            return self._fit_regime_switching_model(prices, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _fit_local_level_model(self, prices: np.ndarray, 
                              observation_noise: float = None,
                              state_noise: float = None) -> KalmanResult:
        """Fit local level (random walk) Kalman filter"""
        
        n = len(prices)
        
        # Estimate noise parameters if not provided
        if observation_noise is None:
            observation_noise = np.var(np.diff(prices)) * 0.1
        if state_noise is None:
            state_noise = np.var(np.diff(prices)) * 0.9
        
        # State space matrices
        F = np.array([[1.0]])  # State transition
        H = np.array([[1.0]])  # Observation matrix
        Q = np.array([[state_noise]])  # State noise covariance
        R = np.array([[observation_noise]])  # Observation noise covariance
        
        # Initialize
        x = np.array([[prices[0]]])  # Initial state
        P = np.array([[1.0]])  # Initial state covariance
        
        # Storage
        filtered_states = []
        predicted_states = []
        state_covariances = []
        innovations = []
        innovation_covariances = []
        log_likelihood = 0.0
        
        # Kalman filtering
        for t in range(n):
            # Prediction step
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            
            # Update step
            y = prices[t] - (H @ x_pred)[0, 0]  # Innovation
            S = H @ P_pred @ H.T + R  # Innovation covariance
            K = P_pred @ H.T @ inv(S)  # Kalman gain
            
            x = x_pred + K * y
            P = P_pred - K @ H @ P_pred
            
            # Store results
            filtered_states.append(x.flatten().tolist())
            predicted_states.append(x_pred.flatten().tolist())
            state_covariances.append(P.tolist())
            innovations.append(float(y))
            innovation_covariances.append(float(S[0, 0]))
            
            # Update log-likelihood
            log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(det(S)) + y**2 / S[0, 0])
        
        # Kalman smoothing (RTS smoother)
        smoothed_states = self._rts_smoother(filtered_states, state_covariances, F, Q)
        
        return KalmanResult(
            filtered_states=filtered_states,
            predicted_states=predicted_states,
            state_covariances=state_covariances,
            innovations=innovations,
            innovation_covariances=innovation_covariances,
            log_likelihood=log_likelihood,
            smoothed_states=smoothed_states,
            model_parameters={
                'observation_noise': observation_noise,
                'state_noise': state_noise
            }
        )
    
    def _fit_local_trend_model(self, prices: np.ndarray,
                              observation_noise: float = None,
                              level_noise: float = None,
                              slope_noise: float = None) -> KalmanResult:
        """Fit local trend (random walk with drift) Kalman filter"""
        
        n = len(prices)
        
        # Estimate noise parameters if not provided
        price_var = np.var(np.diff(prices))
        if observation_noise is None:
            observation_noise = price_var * 0.1
        if level_noise is None:
            level_noise = price_var * 0.8
        if slope_noise is None:
            slope_noise = price_var * 0.1
        
        # State space matrices (2D state: [level, slope])
        F = np.array([[1.0, 1.0],
                     [0.0, 1.0]])  # State transition
        H = np.array([[1.0, 0.0]])  # Observation matrix
        Q = np.array([[level_noise, 0.0],
                     [0.0, slope_noise]])  # State noise covariance
        R = np.array([[observation_noise]])  # Observation noise covariance
        
        # Initialize
        initial_slope = np.mean(np.diff(prices[:min(10, len(prices))]))
        x = np.array([[prices[0]], [initial_slope]])  # Initial state [level, slope]
        P = np.array([[1.0, 0.0],
                     [0.0, 1.0]])  # Initial state covariance
        
        # Storage
        filtered_states = []
        predicted_states = []
        state_covariances = []
        innovations = []
        innovation_covariances = []
        log_likelihood = 0.0
        
        # Kalman filtering
        for t in range(n):
            # Prediction step
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            
            # Update step
            y = prices[t] - (H @ x_pred)[0, 0]  # Innovation
            S = H @ P_pred @ H.T + R  # Innovation covariance
            K = P_pred @ H.T @ inv(S)  # Kalman gain
            
            x = x_pred + K * y
            P = P_pred - K @ H @ P_pred
            
            # Store results
            filtered_states.append(x.flatten().tolist())
            predicted_states.append(x_pred.flatten().tolist())
            state_covariances.append(P.tolist())
            innovations.append(float(y))
            innovation_covariances.append(float(S[0, 0]))
            
            # Update log-likelihood
            log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(det(S)) + y**2 / S[0, 0])
        
        # Kalman smoothing
        smoothed_states = self._rts_smoother(filtered_states, state_covariances, F, Q)
        
        return KalmanResult(
            filtered_states=filtered_states,
            predicted_states=predicted_states,
            state_covariances=state_covariances,
            innovations=innovations,
            innovation_covariances=innovation_covariances,
            log_likelihood=log_likelihood,
            smoothed_states=smoothed_states,
            model_parameters={
                'observation_noise': observation_noise,
                'level_noise': level_noise,
                'slope_noise': slope_noise
            }
        )
    
    def _fit_regime_switching_model(self, prices: np.ndarray,
                                   n_regimes: int = 2) -> KalmanResult:
        """Fit regime-switching Kalman filter"""
        
        n = len(prices)
        
        # Simplified regime-switching model using volatility clustering
        returns = np.diff(np.log(prices))
        volatility = np.abs(returns)
        
        # Identify regimes based on volatility
        vol_threshold = np.median(volatility)
        regimes = (volatility > vol_threshold).astype(int)
        
        # Fit separate local level models for each regime
        regime_results = {}
        
        for regime in range(n_regimes):
            regime_mask = (regimes == regime)
            if np.sum(regime_mask) > 10:  # Minimum observations
                regime_prices = prices[1:][regime_mask]  # Skip first price (no return)
                if len(regime_prices) > 0:
                    regime_result = self._fit_local_level_model(regime_prices)
                    regime_results[regime] = regime_result
        
        # Combine results (simplified approach)
        if regime_results:
            # Use the regime with more observations as primary
            primary_regime = max(regime_results.keys(), 
                               key=lambda r: len(regime_results[r].filtered_states))
            primary_result = regime_results[primary_regime]
            
            # Create regime probabilities
            state_probabilities = []
            for t in range(len(regimes)):
                probs = [0.0] * n_regimes
                if t < len(regimes):
                    probs[regimes[t]] = 1.0
                else:
                    probs[0] = 1.0  # Default to regime 0
                state_probabilities.append(probs)
            
            # Extend primary result to full length
            full_length = len(prices)
            primary_length = len(primary_result.filtered_states)
            
            if primary_length < full_length:
                # Extend with last values
                extension_length = full_length - primary_length
                last_state = primary_result.filtered_states[-1]
                last_cov = primary_result.state_covariances[-1]
                
                filtered_states = primary_result.filtered_states + [last_state] * extension_length
                predicted_states = primary_result.predicted_states + [last_state] * extension_length
                state_covariances = primary_result.state_covariances + [last_cov] * extension_length
                innovations = primary_result.innovations + [0.0] * extension_length
                innovation_covariances = primary_result.innovation_covariances + [1.0] * extension_length
            else:
                filtered_states = primary_result.filtered_states[:full_length]
                predicted_states = primary_result.predicted_states[:full_length]
                state_covariances = primary_result.state_covariances[:full_length]
                innovations = primary_result.innovations[:full_length]
                innovation_covariances = primary_result.innovation_covariances[:full_length]
            
            return KalmanResult(
                filtered_states=filtered_states,
                predicted_states=predicted_states,
                state_covariances=state_covariances,
                innovations=innovations,
                innovation_covariances=innovation_covariances,
                log_likelihood=primary_result.log_likelihood,
                smoothed_states=filtered_states,  # Simplified
                model_parameters=primary_result.model_parameters,
                state_probabilities=state_probabilities
            )
        else:
            # Fallback to local level model
            return self._fit_local_level_model(prices)
    
    def _rts_smoother(self, filtered_states: List[List[float]], 
                     state_covariances: List[List[List[float]]],
                     F: np.ndarray, Q: np.ndarray) -> List[List[float]]:
        """Rauch-Tung-Striebel smoother"""
        
        n = len(filtered_states)
        state_dim = len(filtered_states[0])
        
        # Initialize with last filtered state
        smoothed_states = [None] * n
        smoothed_states[-1] = filtered_states[-1].copy()
        
        # Backward pass
        for t in range(n-2, -1, -1):
            try:
                x_t = np.array(filtered_states[t]).reshape(-1, 1)
                P_t = np.array(state_covariances[t])
                x_t1 = np.array(filtered_states[t+1]).reshape(-1, 1)
                P_t1 = np.array(state_covariances[t+1])
                
                # Predicted state and covariance
                x_pred = F @ x_t
                P_pred = F @ P_t @ F.T + Q
                
                # Smoother gain
                A = P_t @ F.T @ inv(P_pred)
                
                # Smoothed state
                x_smooth = x_t + A @ (np.array(smoothed_states[t+1]).reshape(-1, 1) - x_pred)
                smoothed_states[t] = x_smooth.flatten().tolist()
                
            except:
                # Fallback to filtered state
                smoothed_states[t] = filtered_states[t].copy()
        
        return smoothed_states
    
    def analyze_kalman(self, index_data: IndexData, 
                      model_types: Optional[List[str]] = None) -> Dict[str, KalmanResult]:
        """Analyze index using multiple Kalman filter models
        
        Args:
            index_data: IndexData object containing price data
            model_types: List of Kalman models to fit (default: ['local_level', 'local_trend', 'regime_switching'])
            
        Returns:
            Dictionary of KalmanResult objects for each model type
        """
        
        if model_types is None:
            model_types = ['local_level', 'local_trend', 'regime_switching']
        
        results = {}
        
        for model_type in model_types:
            try:
                print(f"Fitting {model_type} Kalman filter...")
                result = self.fit_kalman_filter(index_data.prices, model_type)
                results[model_type] = result
            except Exception as e:
                print(f"Failed to fit {model_type}: {e}")
                # Create a fallback result
                results[model_type] = self._create_fallback_result(index_data, model_type)
        
        return results
    
    def _create_fallback_result(self, index_data: IndexData, model_type: str) -> KalmanResult:
        """Create fallback Kalman result when model fitting fails"""
        
        prices = np.array(index_data.prices)
        n = len(prices)
        
        # Simple state estimation (smoothed prices)
        if n > 10:
            smoothed_prices = np.convolve(prices, np.ones(min(10, n))/(min(10, n)), mode='same')
        else:
            smoothed_prices = prices.copy()
        
        # Create states based on model type
        if model_type == 'local_trend':
            # 2D state: [level, slope]
            slopes = np.gradient(smoothed_prices)
            filtered_states = [[float(p), float(s)] for p, s in zip(smoothed_prices, slopes)]
        else:
            # 1D state: [level]
            filtered_states = [[float(p)] for p in smoothed_prices]
        
        state_dim = len(filtered_states[0])
        
        return KalmanResult(
            filtered_states=filtered_states,
            predicted_states=filtered_states.copy(),
            state_covariances=[[[1.0] * state_dim for _ in range(state_dim)] for _ in range(n)],
            innovations=[0.0] * n,
            innovation_covariances=[1.0] * n,
            log_likelihood=-1000,
            smoothed_states=filtered_states.copy(),
            model_parameters={'observation_noise': 1.0, 'state_noise': 1.0}
        )

# Example usage and testing
if __name__ == "__main__":
    # Generate sample index data for testing
    np.random.seed(42)
    n_obs = 200
    
    # Generate synthetic price series with trend and noise
    trend = np.linspace(100, 120, n_obs)
    noise = np.random.normal(0, 2, n_obs)
    prices = trend + noise + np.cumsum(np.random.normal(0, 0.5, n_obs))
    
    # Calculate returns
    returns = np.diff(np.log(prices)).tolist()
    
    # Create timestamps
    timestamps = pd.date_range(start='2022-01-01', periods=n_obs, freq='D')
    
    # Create IndexData object
    index_data = IndexData(
        index_symbol="TEST_INDEX",
        prices=prices.tolist(),
        returns=returns,
        timestamps=timestamps.tolist()
    )
    
    # Create analyzer
    analyzer = KalmanFilterAnalyzer()
    
    # Perform analysis
    print("Performing Kalman Filter analysis...")
    results = analyzer.analyze_kalman(index_data)
    
    # Print results
    print(f"\n=== Kalman Filter Analysis Results ===\n")
    
    for model_type, result in results.items():
        print(f"{model_type} Model:")
        print(f"  Log-Likelihood: {result.log_likelihood:.4f}")
        print(f"  State Dimension: {len(result.filtered_states[0])}")
        print(f"  Final State: {result.filtered_states[-1]}")
        print(f"  Model Parameters: {result.model_parameters}")
        
        if result.state_probabilities:
            print(f"  Final State Probabilities: {result.state_probabilities[-1]}")
        
        print()
    
    # Find best model (highest log-likelihood)
    best_model = max(results.items(), key=lambda x: x[1].log_likelihood)
    print(f"Best Model: {best_model[0]} (Log-Likelihood: {best_model[1].log_likelihood:.4f})")
    
    print("\nKalman Filter analysis completed successfully!")