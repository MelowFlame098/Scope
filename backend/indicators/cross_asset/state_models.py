from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

# Conditional imports with fallbacks
try:
    from hmmlearn import hmm
except ImportError:
    hmm = None

try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    GaussianMixture = None

try:
    import scipy.stats as stats
    from scipy.optimize import minimize
except ImportError:
    stats = None
    minimize = None

try:
    from statsmodels.tsa.regime_switching import MarkovRegression
except ImportError:
    MarkovRegression = None

try:
    import ruptures as rpt
except ImportError:
    rpt = None


@dataclass
class CrossAssetData:
    """Data structure for cross-asset analysis"""
    asset_prices: Dict[str, List[float]]
    asset_returns: Dict[str, List[float]]
    timestamps: List[str]
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    volatilities: Optional[Dict[str, float]] = None


@dataclass
class StateModelResults:
    """Results from state model analysis"""
    hmm_states: np.ndarray
    hmm_transition_matrix: np.ndarray
    hmm_means: np.ndarray
    hmm_covariances: np.ndarray
    change_points: List[int]
    regime_statistics: Dict[int, Dict[str, float]]
    model_likelihood: float
    regime_probabilities: Optional[np.ndarray]
    state_durations: Dict[int, float]
    regime_transitions: Dict[Tuple[int, int], float]


class StateModelAnalyzer:
    """State model analysis using HMM and Bayesian methods"""
    
    def __init__(self, n_states: int = 3):
        self.n_states = n_states
        self.hmm_model = None
        self.markov_model = None
    
    def fit_hmm_model(self, returns: np.ndarray) -> Dict[str, Any]:
        """Fit Hidden Markov Model with fallback"""
        if len(returns) < 10:
            return self._fit_hmm_fallback(returns)
        
        if hmm is not None:
            try:
                # Reshape returns for HMM
                X = returns.reshape(-1, 1)
                
                # Fit Gaussian HMM
                model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", random_state=42)
                model.fit(X)
                
                # Get states
                states = model.predict(X)
                
                # Get state probabilities
                state_probs = model.predict_proba(X)
                
                # Calculate likelihood
                log_likelihood = model.score(X)
                
                self.hmm_model = model
                
                return {
                    'states': states,
                    'transition_matrix': model.transmat_,
                    'means': model.means_.flatten(),
                    'covariances': model.covars_.flatten(),
                    'state_probabilities': state_probs,
                    'log_likelihood': log_likelihood,
                    'success': True
                }
                
            except Exception as e:
                print(f"HMM fitting failed: {e}")
                return self._fit_hmm_fallback(returns)
        else:
            return self._fit_hmm_fallback(returns)
    
    def _fit_hmm_fallback(self, returns: np.ndarray) -> Dict[str, Any]:
        """Fallback HMM implementation using simple clustering"""
        if GaussianMixture is not None:
            try:
                # Use Gaussian Mixture Model as fallback
                X = returns.reshape(-1, 1)
                gmm = GaussianMixture(n_components=self.n_states, random_state=42)
                gmm.fit(X)
                
                states = gmm.predict(X)
                state_probs = gmm.predict_proba(X)
                
                # Create mock transition matrix
                transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
                
                # Calculate state statistics
                means = []
                covariances = []
                for i in range(self.n_states):
                    state_returns = returns[states == i]
                    if len(state_returns) > 0:
                        means.append(np.mean(state_returns))
                        covariances.append(np.var(state_returns))
                    else:
                        means.append(0.0)
                        covariances.append(0.01)
                
                return {
                    'states': states,
                    'transition_matrix': transition_matrix,
                    'means': np.array(means),
                    'covariances': np.array(covariances),
                    'state_probabilities': state_probs,
                    'log_likelihood': gmm.score(X),
                    'success': True
                }
                
            except Exception as e:
                print(f"GMM fallback failed: {e}")
        
        # Ultimate fallback: simple quantile-based states
        return self._fit_simple_states(returns)
    
    def _fit_simple_states(self, returns: np.ndarray) -> Dict[str, Any]:
        """Simple state assignment based on return quantiles"""
        if len(returns) == 0:
            return {
                'states': np.array([]),
                'transition_matrix': np.eye(self.n_states),
                'means': np.zeros(self.n_states),
                'covariances': np.ones(self.n_states) * 0.01,
                'state_probabilities': None,
                'log_likelihood': -1000.0,
                'success': False
            }
        
        # Define states based on return quantiles
        if self.n_states == 2:
            threshold = np.median(returns)
            states = (returns > threshold).astype(int)
        elif self.n_states == 3:
            low_threshold = np.percentile(returns, 33.33)
            high_threshold = np.percentile(returns, 66.67)
            states = np.zeros(len(returns), dtype=int)
            states[returns > low_threshold] = 1
            states[returns > high_threshold] = 2
        else:
            # For more states, use equal quantiles
            quantiles = np.linspace(0, 100, self.n_states + 1)
            thresholds = np.percentile(returns, quantiles[1:-1])
            states = np.digitize(returns, thresholds)
        
        # Calculate state statistics
        means = []
        covariances = []
        for i in range(self.n_states):
            state_returns = returns[states == i]
            if len(state_returns) > 0:
                means.append(np.mean(state_returns))
                covariances.append(np.var(state_returns))
            else:
                means.append(0.0)
                covariances.append(0.01)
        
        # Simple transition matrix
        transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        
        return {
            'states': states,
            'transition_matrix': transition_matrix,
            'means': np.array(means),
            'covariances': np.array(covariances),
            'state_probabilities': None,
            'log_likelihood': -100.0,
            'success': True
        }
    
    def detect_change_points_bayesian(self, returns: np.ndarray, method: str = 'pelt') -> List[int]:
        """Detect change points using Bayesian methods with fallback"""
        if len(returns) < 10:
            return []
        
        if rpt is not None:
            try:
                # Use ruptures library for change point detection
                if method == 'pelt':
                    algo = rpt.Pelt(model="rbf").fit(returns)
                    change_points = algo.predict(pen=10)
                elif method == 'binseg':
                    algo = rpt.Binseg(model="l2").fit(returns)
                    change_points = algo.predict(n_bkps=5)
                else:
                    algo = rpt.Window(width=40, model="l2").fit(returns)
                    change_points = algo.predict(n_bkps=5)
                
                # Remove the last point (end of series)
                if change_points and change_points[-1] == len(returns):
                    change_points = change_points[:-1]
                
                return change_points
                
            except Exception as e:
                print(f"Ruptures change point detection failed: {e}")
        
        # Fallback: simple variance-based change point detection
        return self._detect_change_points_simple(returns)
    
    def _detect_change_points_simple(self, returns: np.ndarray, window: int = 20) -> List[int]:
        """Simple change point detection based on rolling variance"""
        if len(returns) < window * 2:
            return []
        
        change_points = []
        
        # Calculate rolling variance
        rolling_var = []
        for i in range(window, len(returns) - window):
            var1 = np.var(returns[i-window:i])
            var2 = np.var(returns[i:i+window])
            rolling_var.append(abs(var2 - var1))
        
        if not rolling_var:
            return []
        
        # Find peaks in variance changes
        threshold = np.mean(rolling_var) + 2 * np.std(rolling_var)
        
        for i, var_change in enumerate(rolling_var):
            if var_change > threshold:
                change_points.append(i + window)
        
        # Remove change points that are too close to each other
        filtered_change_points = []
        for cp in change_points:
            if not filtered_change_points or cp - filtered_change_points[-1] > window:
                filtered_change_points.append(cp)
        
        return filtered_change_points
    
    def calculate_regime_statistics(self, returns: np.ndarray, states: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Calculate statistics for each regime"""
        regime_stats = {}
        
        for state in range(self.n_states):
            state_returns = returns[states == state]
            
            if len(state_returns) > 0:
                regime_stats[state] = {
                    'mean_return': np.mean(state_returns),
                    'volatility': np.std(state_returns),
                    'skewness': stats.skew(state_returns) if stats else 0.0,
                    'kurtosis': stats.kurtosis(state_returns) if stats else 0.0,
                    'min_return': np.min(state_returns),
                    'max_return': np.max(state_returns),
                    'duration': len(state_returns),
                    'frequency': len(state_returns) / len(returns)
                }
            else:
                regime_stats[state] = {
                    'mean_return': 0.0,
                    'volatility': 0.01,
                    'skewness': 0.0,
                    'kurtosis': 0.0,
                    'min_return': 0.0,
                    'max_return': 0.0,
                    'duration': 0,
                    'frequency': 0.0
                }
        
        return regime_stats
    
    def calculate_state_durations(self, states: np.ndarray) -> Dict[int, float]:
        """Calculate average duration for each state"""
        durations = {}
        
        for state in range(self.n_states):
            state_durations = []
            current_duration = 0
            in_state = False
            
            for s in states:
                if s == state:
                    if not in_state:
                        in_state = True
                        current_duration = 1
                    else:
                        current_duration += 1
                else:
                    if in_state:
                        state_durations.append(current_duration)
                        in_state = False
                        current_duration = 0
            
            # Handle case where series ends in the state
            if in_state:
                state_durations.append(current_duration)
            
            durations[state] = np.mean(state_durations) if state_durations else 0.0
        
        return durations
    
    def calculate_regime_transitions(self, states: np.ndarray) -> Dict[Tuple[int, int], float]:
        """Calculate regime transition frequencies"""
        transitions = {}
        total_transitions = 0
        
        # Initialize all possible transitions
        for i in range(self.n_states):
            for j in range(self.n_states):
                transitions[(i, j)] = 0
        
        # Count transitions
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            transitions[(current_state, next_state)] += 1
            total_transitions += 1
        
        # Convert to frequencies
        if total_transitions > 0:
            for key in transitions:
                transitions[key] /= total_transitions
        
        return transitions
    
    def predict_next_state(self, current_state: int, n_steps: int = 1) -> List[Tuple[int, float]]:
        """Predict next state probabilities"""
        if self.hmm_model is None:
            # Simple fallback: assume equal probability for all states
            prob = 1.0 / self.n_states
            return [(i, prob) for i in range(self.n_states)]
        
        try:
            # Use transition matrix for prediction
            current_probs = np.zeros(self.n_states)
            current_probs[current_state] = 1.0
            
            # Apply transition matrix n_steps times
            for _ in range(n_steps):
                current_probs = np.dot(current_probs, self.hmm_model.transmat_)
            
            return [(i, prob) for i, prob in enumerate(current_probs)]
            
        except Exception:
            # Fallback to equal probabilities
            prob = 1.0 / self.n_states
            return [(i, prob) for i in range(self.n_states)]
    
    def analyze_regime_switching(self, returns: np.ndarray) -> StateModelResults:
        """Comprehensive regime switching analysis"""
        print("Performing regime switching analysis...")
        
        if len(returns) == 0:
            return StateModelResults(
                hmm_states=np.array([]),
                hmm_transition_matrix=np.eye(self.n_states),
                hmm_means=np.zeros(self.n_states),
                hmm_covariances=np.ones(self.n_states) * 0.01,
                change_points=[],
                regime_statistics={},
                model_likelihood=-1000.0,
                regime_probabilities=None,
                state_durations={},
                regime_transitions={}
            )
        
        # Fit HMM model
        hmm_results = self.fit_hmm_model(returns)
        
        # Detect change points
        change_points = self.detect_change_points_bayesian(returns)
        
        # Calculate regime statistics
        regime_statistics = self.calculate_regime_statistics(returns, hmm_results['states'])
        
        # Calculate state durations
        state_durations = self.calculate_state_durations(hmm_results['states'])
        
        # Calculate regime transitions
        regime_transitions = self.calculate_regime_transitions(hmm_results['states'])
        
        return StateModelResults(
            hmm_states=hmm_results['states'],
            hmm_transition_matrix=hmm_results['transition_matrix'],
            hmm_means=hmm_results['means'],
            hmm_covariances=hmm_results['covariances'],
            change_points=change_points,
            regime_statistics=regime_statistics,
            model_likelihood=hmm_results['log_likelihood'],
            regime_probabilities=hmm_results['state_probabilities'],
            state_durations=state_durations,
            regime_transitions=regime_transitions
        )
    
    def get_current_regime_info(self, results: StateModelResults) -> Dict[str, Any]:
        """Get information about the current regime"""
        if len(results.hmm_states) == 0:
            return {}
        
        current_state = results.hmm_states[-1]
        
        info = {
            'current_state': int(current_state),
            'state_statistics': results.regime_statistics.get(current_state, {}),
            'expected_duration': results.state_durations.get(current_state, 0.0),
            'transition_probabilities': {}
        }
        
        # Get transition probabilities from current state
        for next_state in range(self.n_states):
            if current_state < len(results.hmm_transition_matrix):
                prob = results.hmm_transition_matrix[current_state, next_state]
                info['transition_probabilities'][next_state] = prob
        
        return info
    
    def detect_regime_anomalies(self, results: StateModelResults, threshold: float = 0.1) -> List[Dict[str, Any]]:
        """Detect anomalous regime behavior"""
        anomalies = []
        
        if len(results.hmm_states) < 10:
            return anomalies
        
        # Detect unusually short regime durations
        states = results.hmm_states
        current_state = states[0]
        current_duration = 1
        
        for i in range(1, len(states)):
            if states[i] == current_state:
                current_duration += 1
            else:
                # State changed
                expected_duration = results.state_durations.get(current_state, 1.0)
                
                if current_duration < expected_duration * threshold:
                    anomalies.append({
                        'type': 'short_regime_duration',
                        'state': current_state,
                        'actual_duration': current_duration,
                        'expected_duration': expected_duration,
                        'end_index': i - 1
                    })
                
                current_state = states[i]
                current_duration = 1
        
        # Detect unexpected transitions
        for i in range(len(states) - 1):
            current_state = states[i]
            next_state = states[i + 1]
            
            if current_state < len(results.hmm_transition_matrix):
                expected_prob = results.hmm_transition_matrix[current_state, next_state]
                
                if expected_prob < threshold:
                    anomalies.append({
                        'type': 'unexpected_transition',
                        'from_state': current_state,
                        'to_state': next_state,
                        'probability': expected_prob,
                        'index': i
                    })
        
        return anomalies


# Example usage
if __name__ == "__main__":
    # Generate sample return data with regime switching
    np.random.seed(42)
    n_periods = 252
    
    # Create regime-switching returns
    returns = []
    states = []
    
    # Regime 1: Low volatility, positive mean
    for _ in range(100):
        returns.append(np.random.normal(0.001, 0.01))
        states.append(0)
    
    # Regime 2: High volatility, negative mean
    for _ in range(80):
        returns.append(np.random.normal(-0.002, 0.03))
        states.append(1)
    
    # Regime 3: Medium volatility, neutral mean
    for _ in range(72):
        returns.append(np.random.normal(0.0005, 0.02))
        states.append(2)
    
    returns = np.array(returns)
    true_states = np.array(states)
    
    # Initialize analyzer
    state_analyzer = StateModelAnalyzer(n_states=3)
    
    # Perform analysis
    results = state_analyzer.analyze_regime_switching(returns)
    
    print("State Model Analysis Results:")
    print(f"Detected {len(np.unique(results.hmm_states))} states")
    print(f"Model likelihood: {results.model_likelihood:.2f}")
    print(f"Change points detected: {len(results.change_points)}")
    
    print("\nRegime Statistics:")
    for state, stats in results.regime_statistics.items():
        print(f"  State {state}:")
        print(f"    Mean Return: {stats['mean_return']:.4f}")
        print(f"    Volatility: {stats['volatility']:.4f}")
        print(f"    Duration: {stats['duration']} periods")
        print(f"    Frequency: {stats['frequency']:.2%}")
    
    print("\nState Durations:")
    for state, duration in results.state_durations.items():
        print(f"  State {state}: {duration:.1f} periods on average")
    
    print("\nTransition Matrix:")
    print(results.hmm_transition_matrix)
    
    # Get current regime info
    current_info = state_analyzer.get_current_regime_info(results)
    print(f"\nCurrent Regime: State {current_info.get('current_state', 'Unknown')}")
    
    # Detect anomalies
    anomalies = state_analyzer.detect_regime_anomalies(results)
    if anomalies:
        print(f"\nDetected {len(anomalies)} regime anomalies:")
        for anomaly in anomalies[:3]:  # Show first 3
            print(f"  {anomaly['type']} at index {anomaly.get('index', anomaly.get('end_index', 'N/A'))}")
    
    # Predict next state
    if len(results.hmm_states) > 0:
        current_state = results.hmm_states[-1]
        predictions = state_analyzer.predict_next_state(current_state)
        print(f"\nNext state predictions from state {current_state}:")
        for state, prob in predictions:
            print(f"  State {state}: {prob:.3f}")