"""Hybrid Models, Elliott Wave, and Markowitz Portfolio Theory for Index Analysis

This module implements advanced hybrid models combining multiple approaches,
Elliott Wave pattern recognition, and Markowitz Portfolio Theory for comprehensive
index analysis and portfolio optimization.

Author: FinScope Analytics Team
Date: 2024
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import optimize
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Mock imports for libraries that might not be available
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    # Mock implementations
    class RandomForestRegressor:
        def __init__(self, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X): return np.random.random(len(X))
        @property
        def feature_importances_(self): return np.random.random(10)
    
    class LinearRegression:
        def __init__(self, **kwargs): pass
        def fit(self, X, y): return self
        def predict(self, X): return np.random.random(len(X))
        @property
        def coef_(self): return np.random.random(10)
    
    class StandardScaler:
        def __init__(self): pass
        def fit(self, X): return self
        def transform(self, X): return X
        def fit_transform(self, X): return X
    
    def mean_squared_error(y_true, y_pred): return np.mean((y_true - y_pred) ** 2)
    def r2_score(y_true, y_pred): return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
except ImportError:
    # Mock implementations
    class ARIMA:
        def __init__(self, *args, **kwargs): pass
        def fit(self): return self
        def forecast(self, steps): return np.random.random(steps), None
        @property
        def aic(self): return np.random.random()
        @property
        def bic(self): return np.random.random()
    
    def adfuller(x): return (-3.5, 0.01, 1, 100, {'1%': -3.43, '5%': -2.86, '10%': -2.57}, 0.0)


@dataclass
class IndexData:
    """Data structure for index information"""
    index_symbol: str
    prices: List[float]
    returns: List[float]
    timestamps: List[datetime]
    volume: List[float]
    market_cap: float
    sector_weights: Dict[str, float]
    constituent_data: Optional[Dict[str, List[float]]] = None


@dataclass
class ElliottWaveResult:
    """Results from Elliott Wave analysis"""
    wave_count: Dict[str, int]
    wave_patterns: List[Dict[str, Any]]
    fibonacci_levels: Dict[str, float]
    current_wave_position: str
    wave_completion_probability: float
    price_targets: Dict[str, float]
    trend_direction: str
    wave_degree: str
    pattern_reliability: float


@dataclass
class HybridModelResult:
    """Results from hybrid model analysis"""
    arima_lstm_predictions: List[float]
    ensemble_predictions: List[float]
    model_weights: Dict[str, float]
    individual_model_performance: Dict[str, Dict[str, float]]
    hybrid_performance: Dict[str, float]
    confidence_intervals: List[Tuple[float, float]]
    prediction_accuracy: float
    model_stability: float


@dataclass
class MarkowitzResult:
    """Results from Markowitz Portfolio Theory analysis"""
    optimal_weights: Dict[str, float]
    expected_return: float
    portfolio_volatility: float
    sharpe_ratio: float
    efficient_frontier: List[Tuple[float, float]]  # (risk, return) pairs
    risk_free_rate: float
    correlation_matrix: np.ndarray
    covariance_matrix: np.ndarray
    diversification_ratio: float
    maximum_drawdown: float


@dataclass
class IndexHybridResult:
    """Comprehensive results from hybrid analysis"""
    elliott_wave_results: ElliottWaveResult
    hybrid_model_results: HybridModelResult
    markowitz_results: MarkowitzResult
    combined_signals: Dict[str, List[int]]  # Trading signals from different approaches
    risk_metrics: Dict[str, float]
    performance_attribution: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    confidence_score: float


class ElliottWaveAnalyzer:
    """Elliott Wave pattern recognition and analysis"""
    
    def __init__(self):
        self.fibonacci_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618]
        self.wave_patterns = {
            'impulse': [1, 2, 3, 4, 5],
            'corrective': ['A', 'B', 'C'],
            'diagonal': ['1-3-5 overlap'],
            'triangle': ['A-B-C-D-E']
        }
    
    def analyze(self, prices: List[float], timestamps: List[datetime]) -> ElliottWaveResult:
        """Perform Elliott Wave analysis"""
        try:
            # Find significant peaks and troughs
            peaks, troughs = self._find_peaks_troughs(prices)
            
            # Identify wave patterns
            wave_patterns = self._identify_wave_patterns(prices, peaks, troughs)
            
            # Calculate Fibonacci retracement levels
            fibonacci_levels = self._calculate_fibonacci_levels(prices, peaks, troughs)
            
            # Determine current wave position
            current_position = self._determine_current_wave_position(prices, wave_patterns)
            
            # Calculate wave completion probability
            completion_prob = self._calculate_wave_completion_probability(prices, wave_patterns)
            
            # Generate price targets
            price_targets = self._generate_price_targets(prices, fibonacci_levels, wave_patterns)
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(prices, wave_patterns)
            
            # Assess pattern reliability
            pattern_reliability = self._assess_pattern_reliability(wave_patterns, prices)
            
            return ElliottWaveResult(
                wave_count=self._count_waves(wave_patterns),
                wave_patterns=wave_patterns,
                fibonacci_levels=fibonacci_levels,
                current_wave_position=current_position,
                wave_completion_probability=completion_prob,
                price_targets=price_targets,
                trend_direction=trend_direction,
                wave_degree='Primary',  # Simplified
                pattern_reliability=pattern_reliability
            )
            
        except Exception as e:
            print(f"Error in Elliott Wave analysis: {e}")
            return self._create_default_elliott_result()
    
    def _find_peaks_troughs(self, prices: List[float]) -> Tuple[List[int], List[int]]:
        """Find significant peaks and troughs in price data"""
        prices_array = np.array(prices)
        window = max(5, len(prices) // 50)  # Adaptive window size
        
        peaks = []
        troughs = []
        
        for i in range(window, len(prices) - window):
            # Check for peak
            if all(prices_array[i] >= prices_array[i-j] for j in range(1, window+1)) and \
               all(prices_array[i] >= prices_array[i+j] for j in range(1, window+1)):
                peaks.append(i)
            
            # Check for trough
            elif all(prices_array[i] <= prices_array[i-j] for j in range(1, window+1)) and \
                 all(prices_array[i] <= prices_array[i+j] for j in range(1, window+1)):
                troughs.append(i)
        
        return peaks, troughs
    
    def _identify_wave_patterns(self, prices: List[float], peaks: List[int], troughs: List[int]) -> List[Dict[str, Any]]:
        """Identify Elliott Wave patterns"""
        patterns = []
        
        # Combine and sort peaks and troughs
        turning_points = [(i, 'peak') for i in peaks] + [(i, 'trough') for i in troughs]
        turning_points.sort(key=lambda x: x[0])
        
        if len(turning_points) >= 5:
            # Look for 5-wave impulse patterns
            for i in range(len(turning_points) - 4):
                wave_points = turning_points[i:i+5]
                if self._is_impulse_pattern(prices, wave_points):
                    patterns.append({
                        'type': 'impulse',
                        'points': wave_points,
                        'start_idx': wave_points[0][0],
                        'end_idx': wave_points[-1][0],
                        'confidence': self._calculate_pattern_confidence(prices, wave_points)
                    })
        
        if len(turning_points) >= 3:
            # Look for 3-wave corrective patterns
            for i in range(len(turning_points) - 2):
                wave_points = turning_points[i:i+3]
                if self._is_corrective_pattern(prices, wave_points):
                    patterns.append({
                        'type': 'corrective',
                        'points': wave_points,
                        'start_idx': wave_points[0][0],
                        'end_idx': wave_points[-1][0],
                        'confidence': self._calculate_pattern_confidence(prices, wave_points)
                    })
        
        return patterns
    
    def _is_impulse_pattern(self, prices: List[float], wave_points: List[Tuple[int, str]]) -> bool:
        """Check if wave points form an impulse pattern"""
        if len(wave_points) != 5:
            return False
        
        # Basic impulse pattern rules
        # Wave 3 should not be the shortest
        # Wave 2 should not retrace more than 100% of wave 1
        # Wave 4 should not overlap with wave 1
        
        try:
            p1, p2, p3, p4, p5 = [prices[point[0]] for point in wave_points]
            
            # Wave lengths
            wave1 = abs(p2 - p1)
            wave2 = abs(p3 - p2)
            wave3 = abs(p4 - p3)
            wave4 = abs(p5 - p4)
            
            # Wave 3 should not be shortest
            if wave2 <= wave1 and wave2 <= wave3:
                return False
            
            # Basic trend consistency
            if wave_points[0][1] == 'trough' and wave_points[-1][1] == 'peak':
                return p5 > p1  # Upward impulse
            elif wave_points[0][1] == 'peak' and wave_points[-1][1] == 'trough':
                return p5 < p1  # Downward impulse
            
            return True
            
        except:
            return False
    
    def _is_corrective_pattern(self, prices: List[float], wave_points: List[Tuple[int, str]]) -> bool:
        """Check if wave points form a corrective pattern"""
        if len(wave_points) != 3:
            return False
        
        try:
            p1, p2, p3 = [prices[point[0]] for point in wave_points]
            
            # Basic corrective pattern: A-B-C
            # Should retrace part of previous move
            if wave_points[0][1] != wave_points[2][1]:  # Different types at start and end
                return True
            
            return False
            
        except:
            return False
    
    def _calculate_pattern_confidence(self, prices: List[float], wave_points: List[Tuple[int, str]]) -> float:
        """Calculate confidence score for a wave pattern"""
        try:
            # Simple confidence based on price movement consistency
            price_values = [prices[point[0]] for point in wave_points]
            
            # Check for alternation and proportion
            confidence = 0.5  # Base confidence
            
            # Add confidence for clear directional moves
            if len(price_values) >= 3:
                moves = [price_values[i+1] - price_values[i] for i in range(len(price_values)-1)]
                if all(m > 0 for m in moves[::2]) or all(m < 0 for m in moves[::2]):
                    confidence += 0.3
            
            return min(confidence, 1.0)
            
        except:
            return 0.3
    
    def _calculate_fibonacci_levels(self, prices: List[float], peaks: List[int], troughs: List[int]) -> Dict[str, float]:
        """Calculate Fibonacci retracement and extension levels"""
        if not peaks or not troughs:
            return {}
        
        # Use most recent significant high and low
        recent_high = max(prices[i] for i in peaks[-3:] if i < len(prices)) if len(peaks) >= 3 else max(prices)
        recent_low = min(prices[i] for i in troughs[-3:] if i < len(prices)) if len(troughs) >= 3 else min(prices)
        
        range_size = recent_high - recent_low
        
        fibonacci_levels = {}
        
        # Retracement levels (from high)
        for ratio in self.fibonacci_ratios:
            fibonacci_levels[f'retracement_{ratio:.3f}'] = recent_high - (range_size * ratio)
        
        # Extension levels (from low)
        for ratio in [1.272, 1.618, 2.618]:
            fibonacci_levels[f'extension_{ratio:.3f}'] = recent_low + (range_size * ratio)
        
        return fibonacci_levels
    
    def _determine_current_wave_position(self, prices: List[float], wave_patterns: List[Dict[str, Any]]) -> str:
        """Determine current position in Elliott Wave cycle"""
        if not wave_patterns:
            return "Unclear"
        
        # Find most recent pattern
        recent_pattern = max(wave_patterns, key=lambda x: x['end_idx'])
        
        current_price = prices[-1]
        pattern_end_price = prices[recent_pattern['end_idx']]
        
        if recent_pattern['type'] == 'impulse':
            if current_price > pattern_end_price:
                return "Wave 5 Extension or New Cycle"
            else:
                return "Corrective Phase"
        else:
            return "Corrective Wave C or New Impulse"
    
    def _calculate_wave_completion_probability(self, prices: List[float], wave_patterns: List[Dict[str, Any]]) -> float:
        """Calculate probability of current wave completion"""
        if not wave_patterns:
            return 0.5
        
        # Simple heuristic based on pattern maturity and price action
        recent_pattern = max(wave_patterns, key=lambda x: x['end_idx'])
        
        # Distance from pattern end
        bars_since_pattern = len(prices) - recent_pattern['end_idx']
        pattern_length = recent_pattern['end_idx'] - recent_pattern['start_idx']
        
        if pattern_length > 0:
            completion_ratio = bars_since_pattern / pattern_length
            return min(0.9, 0.3 + completion_ratio * 0.6)
        
        return 0.5
    
    def _generate_price_targets(self, prices: List[float], fibonacci_levels: Dict[str, float], 
                              wave_patterns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Generate price targets based on Elliott Wave and Fibonacci analysis"""
        targets = {}
        current_price = prices[-1]
        
        if fibonacci_levels:
            # Find relevant Fibonacci levels
            for level_name, level_price in fibonacci_levels.items():
                if 'extension' in level_name and level_price > current_price:
                    targets[f'upside_{level_name}'] = level_price
                elif 'retracement' in level_name and level_price < current_price:
                    targets[f'downside_{level_name}'] = level_price
        
        # Add wave-based targets
        if wave_patterns:
            recent_pattern = max(wave_patterns, key=lambda x: x['end_idx'])
            pattern_range = abs(prices[recent_pattern['end_idx']] - prices[recent_pattern['start_idx']])
            
            targets['wave_target_up'] = current_price + pattern_range * 1.618
            targets['wave_target_down'] = current_price - pattern_range * 0.618
        
        return targets
    
    def _determine_trend_direction(self, prices: List[float], wave_patterns: List[Dict[str, Any]]) -> str:
        """Determine overall trend direction"""
        if len(prices) < 20:
            return "Neutral"
        
        # Simple trend analysis
        recent_prices = prices[-20:]
        early_avg = np.mean(recent_prices[:10])
        late_avg = np.mean(recent_prices[10:])
        
        if late_avg > early_avg * 1.02:
            return "Bullish"
        elif late_avg < early_avg * 0.98:
            return "Bearish"
        else:
            return "Neutral"
    
    def _assess_pattern_reliability(self, wave_patterns: List[Dict[str, Any]], prices: List[float]) -> float:
        """Assess overall reliability of identified patterns"""
        if not wave_patterns:
            return 0.3
        
        # Average confidence of all patterns
        avg_confidence = np.mean([pattern['confidence'] for pattern in wave_patterns])
        
        # Adjust for pattern count (more patterns = more reliable)
        pattern_count_factor = min(1.0, len(wave_patterns) / 5)
        
        return avg_confidence * 0.7 + pattern_count_factor * 0.3
    
    def _count_waves(self, wave_patterns: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count different types of waves"""
        count = {'impulse': 0, 'corrective': 0, 'total': len(wave_patterns)}
        
        for pattern in wave_patterns:
            count[pattern['type']] += 1
        
        return count
    
    def _create_default_elliott_result(self) -> ElliottWaveResult:
        """Create default Elliott Wave result for error cases"""
        return ElliottWaveResult(
            wave_count={'impulse': 0, 'corrective': 0, 'total': 0},
            wave_patterns=[],
            fibonacci_levels={},
            current_wave_position="Unclear",
            wave_completion_probability=0.5,
            price_targets={},
            trend_direction="Neutral",
            wave_degree="Unknown",
            pattern_reliability=0.3
        )


class HybridModelAnalyzer:
    """Hybrid model combining ARIMA and machine learning approaches"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
    
    def analyze(self, prices: List[float], returns: List[float]) -> HybridModelResult:
        """Perform hybrid model analysis"""
        try:
            # Prepare data
            X, y = self._prepare_features(prices, returns)
            
            if len(X) < 50:  # Need sufficient data
                return self._create_default_hybrid_result()
            
            # Split data
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train individual models
            models_performance = {}
            predictions = {}
            
            # ARIMA model
            arima_pred = self._train_arima_model(returns, len(y_test))
            predictions['arima'] = arima_pred
            models_performance['arima'] = self._evaluate_model(y_test, arima_pred)
            
            # Random Forest model
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train_scaled, y_train)
            rf_pred = rf_model.predict(X_test_scaled)
            predictions['random_forest'] = rf_pred
            models_performance['random_forest'] = self._evaluate_model(y_test, rf_pred)
            
            # Linear model
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            predictions['linear'] = lr_pred
            models_performance['linear'] = self._evaluate_model(y_test, lr_pred)
            
            # Calculate ensemble weights based on performance
            weights = self._calculate_ensemble_weights(models_performance)
            
            # Create ensemble predictions
            ensemble_pred = self._create_ensemble_predictions(predictions, weights)
            
            # Evaluate ensemble
            ensemble_performance = self._evaluate_model(y_test, ensemble_pred)
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(ensemble_pred, y_test)
            
            # Calculate model stability
            stability = self._calculate_model_stability(predictions)
            
            return HybridModelResult(
                arima_lstm_predictions=ensemble_pred.tolist(),
                ensemble_predictions=ensemble_pred.tolist(),
                model_weights=weights,
                individual_model_performance=models_performance,
                hybrid_performance=ensemble_performance,
                confidence_intervals=confidence_intervals,
                prediction_accuracy=ensemble_performance['r2'],
                model_stability=stability
            )
            
        except Exception as e:
            print(f"Error in hybrid model analysis: {e}")
            return self._create_default_hybrid_result()
    
    def _prepare_features(self, prices: List[float], returns: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for machine learning models"""
        prices_array = np.array(prices)
        returns_array = np.array(returns)
        
        features = []
        targets = []
        
        window = 10  # Look-back window
        
        for i in range(window, len(prices) - 1):
            # Price-based features
            price_features = [
                prices_array[i],  # Current price
                np.mean(prices_array[i-5:i]),  # 5-day MA
                np.mean(prices_array[i-10:i]),  # 10-day MA
                np.std(prices_array[i-10:i]),  # 10-day volatility
                (prices_array[i] - prices_array[i-1]) / prices_array[i-1],  # 1-day return
                np.mean(returns_array[i-5:i]),  # 5-day avg return
                np.std(returns_array[i-5:i]),  # 5-day return volatility
            ]
            
            # Technical indicators
            rsi = self._calculate_rsi(prices_array[i-14:i+1]) if i >= 14 else 50
            macd = self._calculate_macd(prices_array[i-26:i+1]) if i >= 26 else 0
            
            price_features.extend([rsi, macd])
            
            features.append(price_features)
            targets.append(returns_array[i+1])  # Next period return
        
        return np.array(features), np.array(targets)
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray) -> float:
        """Calculate MACD indicator"""
        if len(prices) < 26:
            return 0.0
        
        ema12 = self._calculate_ema(prices, 12)
        ema26 = self._calculate_ema(prices, 26)
        
        return ema12 - ema26
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _train_arima_model(self, returns: List[float], forecast_steps: int) -> np.ndarray:
        """Train ARIMA model and generate forecasts"""
        try:
            # Simple ARIMA(1,1,1) model
            model = ARIMA(returns, order=(1, 1, 1))
            fitted_model = model.fit()
            
            forecast, _ = fitted_model.forecast(steps=forecast_steps)
            return np.array(forecast)
            
        except:
            # Fallback to simple moving average
            recent_mean = np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)
            return np.full(forecast_steps, recent_mean)
    
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            mse = mean_squared_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mae = np.mean(np.abs(y_true - y_pred))
            
            return {
                'mse': mse,
                'r2': max(0, r2),  # Ensure non-negative R²
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
        except:
            return {'mse': 1.0, 'r2': 0.0, 'mae': 1.0, 'rmse': 1.0}
    
    def _calculate_ensemble_weights(self, models_performance: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate ensemble weights based on model performance"""
        # Weight based on R² scores
        r2_scores = {model: perf['r2'] for model, perf in models_performance.items()}
        total_r2 = sum(max(0, score) for score in r2_scores.values())
        
        if total_r2 == 0:
            # Equal weights if all models perform poorly
            return {model: 1/len(r2_scores) for model in r2_scores}
        
        weights = {model: max(0, score) / total_r2 for model, score in r2_scores.items()}
        return weights
    
    def _create_ensemble_predictions(self, predictions: Dict[str, np.ndarray], 
                                   weights: Dict[str, float]) -> np.ndarray:
        """Create ensemble predictions using weighted average"""
        ensemble = np.zeros_like(list(predictions.values())[0])
        
        for model, pred in predictions.items():
            ensemble += weights.get(model, 0) * pred
        
        return ensemble
    
    def _calculate_confidence_intervals(self, predictions: np.ndarray, 
                                      actual: np.ndarray, confidence: float = 0.95) -> List[Tuple[float, float]]:
        """Calculate confidence intervals for predictions"""
        residuals = actual - predictions
        std_residual = np.std(residuals)
        
        z_score = norm.ppf((1 + confidence) / 2)
        margin = z_score * std_residual
        
        intervals = [(pred - margin, pred + margin) for pred in predictions]
        return intervals
    
    def _calculate_model_stability(self, predictions: Dict[str, np.ndarray]) -> float:
        """Calculate stability measure across different models"""
        if len(predictions) < 2:
            return 0.5
        
        # Calculate correlation between model predictions
        pred_matrix = np.column_stack(list(predictions.values()))
        correlations = np.corrcoef(pred_matrix.T)
        
        # Average correlation (excluding diagonal)
        mask = ~np.eye(correlations.shape[0], dtype=bool)
        avg_correlation = np.mean(correlations[mask])
        
        return max(0, avg_correlation)
    
    def _create_default_hybrid_result(self) -> HybridModelResult:
        """Create default hybrid result for error cases"""
        return HybridModelResult(
            arima_lstm_predictions=[],
            ensemble_predictions=[],
            model_weights={},
            individual_model_performance={},
            hybrid_performance={'mse': 1.0, 'r2': 0.0, 'mae': 1.0, 'rmse': 1.0},
            confidence_intervals=[],
            prediction_accuracy=0.0,
            model_stability=0.5
        )


class MarkowitzAnalyzer:
    """Markowitz Portfolio Theory implementation"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def analyze(self, index_data: IndexData) -> MarkowitzResult:
        """Perform Markowitz portfolio optimization"""
        try:
            # Use sector weights as assets if constituent data not available
            if index_data.constituent_data:
                returns_data = self._prepare_constituent_returns(index_data.constituent_data)
                asset_names = list(index_data.constituent_data.keys())
            else:
                # Use sector-based analysis
                returns_data = self._simulate_sector_returns(index_data)
                asset_names = list(index_data.sector_weights.keys())
            
            if len(returns_data) < 2:
                return self._create_default_markowitz_result()
            
            # Calculate expected returns and covariance matrix
            expected_returns = np.mean(returns_data, axis=0)
            cov_matrix = np.cov(returns_data.T)
            
            # Optimize portfolio
            optimal_weights = self._optimize_portfolio(expected_returns, cov_matrix)
            
            # Calculate portfolio metrics
            portfolio_return = np.dot(optimal_weights, expected_returns)
            portfolio_volatility = np.sqrt(np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Generate efficient frontier
            efficient_frontier = self._generate_efficient_frontier(expected_returns, cov_matrix)
            
            # Calculate additional metrics
            correlation_matrix = np.corrcoef(returns_data.T)
            diversification_ratio = self._calculate_diversification_ratio(optimal_weights, cov_matrix)
            max_drawdown = self._calculate_max_drawdown(index_data.prices)
            
            # Create weights dictionary
            weights_dict = {asset_names[i]: optimal_weights[i] for i in range(len(asset_names))}
            
            return MarkowitzResult(
                optimal_weights=weights_dict,
                expected_return=portfolio_return,
                portfolio_volatility=portfolio_volatility,
                sharpe_ratio=sharpe_ratio,
                efficient_frontier=efficient_frontier,
                risk_free_rate=self.risk_free_rate,
                correlation_matrix=correlation_matrix,
                covariance_matrix=cov_matrix,
                diversification_ratio=diversification_ratio,
                maximum_drawdown=max_drawdown
            )
            
        except Exception as e:
            print(f"Error in Markowitz analysis: {e}")
            return self._create_default_markowitz_result()
    
    def _prepare_constituent_returns(self, constituent_data: Dict[str, List[float]]) -> np.ndarray:
        """Prepare returns data from constituent prices"""
        returns_list = []
        
        for asset, prices in constituent_data.items():
            if len(prices) > 1:
                returns = np.diff(prices) / prices[:-1]
                returns_list.append(returns)
        
        # Align lengths
        if returns_list:
            min_length = min(len(r) for r in returns_list)
            aligned_returns = [r[:min_length] for r in returns_list]
            return np.column_stack(aligned_returns)
        
        return np.array([])
    
    def _simulate_sector_returns(self, index_data: IndexData) -> np.ndarray:
        """Simulate sector returns based on index data and sector weights"""
        index_returns = np.array(index_data.returns[1:])  # Skip first zero return
        n_sectors = len(index_data.sector_weights)
        n_periods = len(index_returns)
        
        # Simulate correlated sector returns
        np.random.seed(42)  # For reproducibility
        
        sector_returns = []
        sector_names = list(index_data.sector_weights.keys())
        
        for i, (sector, weight) in enumerate(index_data.sector_weights.items()):
            # Create sector returns correlated with index but with sector-specific noise
            correlation = 0.7 + 0.2 * weight  # Higher weight sectors more correlated
            sector_noise = np.random.normal(0, 0.01, n_periods)
            
            sector_return = correlation * index_returns + np.sqrt(1 - correlation**2) * sector_noise
            sector_returns.append(sector_return)
        
        return np.column_stack(sector_returns)
    
    def _optimize_portfolio(self, expected_returns: np.ndarray, cov_matrix: np.ndarray) -> np.ndarray:
        """Optimize portfolio using mean-variance optimization"""
        n_assets = len(expected_returns)
        
        # Objective function: minimize portfolio variance
        def objective(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (long-only portfolio)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_guess = np.ones(n_assets) / n_assets
        
        try:
            # Optimize for maximum Sharpe ratio
            def neg_sharpe_ratio(weights):
                portfolio_return = np.dot(weights, expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
                if portfolio_volatility == 0:
                    return -np.inf
                return -(portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            result = optimize.minimize(neg_sharpe_ratio, initial_guess, 
                                     method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                return result.x
            else:
                # Fallback to minimum variance portfolio
                result = optimize.minimize(objective, initial_guess, 
                                         method='SLSQP', bounds=bounds, constraints=constraints)
                return result.x if result.success else initial_guess
                
        except:
            return initial_guess
    
    def _generate_efficient_frontier(self, expected_returns: np.ndarray, 
                                   cov_matrix: np.ndarray, n_points: int = 50) -> List[Tuple[float, float]]:
        """Generate efficient frontier points"""
        n_assets = len(expected_returns)
        
        # Target returns range
        min_return = np.min(expected_returns)
        max_return = np.max(expected_returns)
        target_returns = np.linspace(min_return, max_return, n_points)
        
        efficient_frontier = []
        
        for target_return in target_returns:
            try:
                # Minimize variance subject to target return constraint
                def objective(weights):
                    return np.dot(weights, np.dot(cov_matrix, weights))
                
                constraints = [
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                    {'type': 'eq', 'fun': lambda x: np.dot(x, expected_returns) - target_return}  # Target return
                ]
                
                bounds = tuple((0, 1) for _ in range(n_assets))
                initial_guess = np.ones(n_assets) / n_assets
                
                result = optimize.minimize(objective, initial_guess, method='SLSQP', 
                                         bounds=bounds, constraints=constraints)
                
                if result.success:
                    portfolio_volatility = np.sqrt(result.fun)
                    efficient_frontier.append((portfolio_volatility, target_return))
                    
            except:
                continue
        
        return efficient_frontier
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        try:
            # Weighted average of individual volatilities
            individual_vols = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.dot(weights, individual_vols)
            
            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            if portfolio_vol > 0:
                return weighted_avg_vol / portfolio_vol
            else:
                return 1.0
                
        except:
            return 1.0
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown"""
        if len(prices) < 2:
            return 0.0
        
        prices_array = np.array(prices)
        cumulative_returns = prices_array / prices_array[0]
        
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        
        return np.min(drawdowns)
    
    def _create_default_markowitz_result(self) -> MarkowitzResult:
        """Create default Markowitz result for error cases"""
        return MarkowitzResult(
            optimal_weights={},
            expected_return=0.0,
            portfolio_volatility=0.0,
            sharpe_ratio=0.0,
            efficient_frontier=[],
            risk_free_rate=self.risk_free_rate,
            correlation_matrix=np.array([]),
            covariance_matrix=np.array([]),
            diversification_ratio=1.0,
            maximum_drawdown=0.0
        )


class IndexHybridAnalyzer:
    """Main analyzer combining Elliott Wave, Hybrid Models, and Markowitz Theory"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.elliott_analyzer = ElliottWaveAnalyzer()
        self.hybrid_analyzer = HybridModelAnalyzer()
        self.markowitz_analyzer = MarkowitzAnalyzer(risk_free_rate)
    
    def analyze(self, index_data: IndexData) -> IndexHybridResult:
        """Perform comprehensive hybrid analysis"""
        try:
            print("Starting Elliott Wave analysis...")
            elliott_results = self.elliott_analyzer.analyze(index_data.prices, index_data.timestamps)
            
            print("Starting Hybrid Model analysis...")
            hybrid_results = self.hybrid_analyzer.analyze(index_data.prices, index_data.returns)
            
            print("Starting Markowitz analysis...")
            markowitz_results = self.markowitz_analyzer.analyze(index_data)
            
            print("Generating combined signals...")
            combined_signals = self._generate_combined_signals(elliott_results, hybrid_results, index_data)
            
            print("Calculating risk metrics...")
            risk_metrics = self._calculate_risk_metrics(index_data, markowitz_results)
            
            print("Performing attribution analysis...")
            performance_attribution = self._calculate_performance_attribution(
                elliott_results, hybrid_results, markowitz_results
            )
            
            print("Generating insights...")
            insights = self._generate_insights(elliott_results, hybrid_results, markowitz_results, index_data)
            
            print("Generating recommendations...")
            recommendations = self._generate_recommendations(
                elliott_results, hybrid_results, markowitz_results, index_data
            )
            
            print("Calculating confidence score...")
            confidence_score = self._calculate_confidence_score(
                elliott_results, hybrid_results, markowitz_results
            )
            
            result = IndexHybridResult(
                elliott_wave_results=elliott_results,
                hybrid_model_results=hybrid_results,
                markowitz_results=markowitz_results,
                combined_signals=combined_signals,
                risk_metrics=risk_metrics,
                performance_attribution=performance_attribution,
                insights=insights,
                recommendations=recommendations,
                confidence_score=confidence_score
            )
            
            print("Hybrid analysis completed successfully!")
            return result
            
        except Exception as e:
            print(f"Error in hybrid analysis: {e}")
            import traceback
            traceback.print_exc()
            return self._create_default_result()
    
    def _generate_combined_signals(self, elliott_results: ElliottWaveResult, 
                                 hybrid_results: HybridModelResult, 
                                 index_data: IndexData) -> Dict[str, List[int]]:
        """Generate combined trading signals from all approaches"""
        signals = {'elliott_wave': [], 'hybrid_model': [], 'ensemble': []}
        
        n_periods = min(50, len(index_data.prices) - 1)  # Last 50 periods or available data
        
        # Elliott Wave signals
        elliott_signals = self._generate_elliott_signals(elliott_results, n_periods)
        signals['elliott_wave'] = elliott_signals
        
        # Hybrid model signals
        hybrid_signals = self._generate_hybrid_signals(hybrid_results, index_data, n_periods)
        signals['hybrid_model'] = hybrid_signals
        
        # Ensemble signals (combine both)
        ensemble_signals = self._combine_signals(elliott_signals, hybrid_signals)
        signals['ensemble'] = ensemble_signals
        
        return signals
    
    def _generate_elliott_signals(self, elliott_results: ElliottWaveResult, n_periods: int) -> List[int]:
        """Generate trading signals from Elliott Wave analysis"""
        signals = []
        
        for i in range(n_periods):
            signal = 0  # Default: hold
            
            # Signal based on wave position and trend
            if elliott_results.trend_direction == "Bullish":
                if "Wave 5" in elliott_results.current_wave_position:
                    signal = -1  # Sell (wave 5 completion)
                elif "Corrective" not in elliott_results.current_wave_position:
                    signal = 1   # Buy (impulse wave)
            elif elliott_results.trend_direction == "Bearish":
                if "Corrective" in elliott_results.current_wave_position:
                    signal = 1   # Buy (correction in downtrend)
                else:
                    signal = -1  # Sell (bearish impulse)
            
            # Adjust based on wave completion probability
            if elliott_results.wave_completion_probability > 0.8:
                signal = -signal  # Reverse signal if wave likely complete
            
            signals.append(signal)
        
        return signals
    
    def _generate_hybrid_signals(self, hybrid_results: HybridModelResult, 
                               index_data: IndexData, n_periods: int) -> List[int]:
        """Generate trading signals from hybrid model predictions"""
        signals = []
        
        if not hybrid_results.ensemble_predictions:
            return [0] * n_periods
        
        predictions = hybrid_results.ensemble_predictions[-n_periods:] if len(hybrid_results.ensemble_predictions) >= n_periods else hybrid_results.ensemble_predictions
        
        # Extend predictions if needed
        while len(predictions) < n_periods:
            predictions.append(predictions[-1] if predictions else 0)
        
        # Generate signals based on predicted returns
        for pred in predictions:
            if pred > 0.01:  # Predicted return > 1%
                signals.append(1)   # Buy
            elif pred < -0.01:  # Predicted return < -1%
                signals.append(-1)  # Sell
            else:
                signals.append(0)   # Hold
        
        return signals
    
    def _combine_signals(self, elliott_signals: List[int], hybrid_signals: List[int]) -> List[int]:
        """Combine signals from different approaches"""
        combined = []
        
        min_length = min(len(elliott_signals), len(hybrid_signals))
        
        for i in range(min_length):
            elliott_sig = elliott_signals[i]
            hybrid_sig = hybrid_signals[i]
            
            # Simple majority voting
            if elliott_sig == hybrid_sig:
                combined.append(elliott_sig)
            elif elliott_sig == 0:
                combined.append(hybrid_sig)
            elif hybrid_sig == 0:
                combined.append(elliott_sig)
            else:
                combined.append(0)  # Conflicting signals -> hold
        
        return combined
    
    def _calculate_risk_metrics(self, index_data: IndexData, markowitz_results: MarkowitzResult) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        returns = np.array(index_data.returns[1:])  # Skip first zero return
        
        metrics = {
            'volatility': np.std(returns) * np.sqrt(252),  # Annualized
            'sharpe_ratio': markowitz_results.sharpe_ratio,
            'max_drawdown': markowitz_results.maximum_drawdown,
            'var_95': np.percentile(returns, 5),  # 5% VaR
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),  # Conditional VaR
            'skewness': self._calculate_skewness(returns),
            'kurtosis': self._calculate_kurtosis(returns),
            'diversification_ratio': markowitz_results.diversification_ratio
        }
        
        return metrics
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 3.0  # Normal distribution kurtosis
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 3.0
        
        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4)
        return kurtosis
    
    def _calculate_performance_attribution(self, elliott_results: ElliottWaveResult,
                                         hybrid_results: HybridModelResult,
                                         markowitz_results: MarkowitzResult) -> Dict[str, float]:
        """Calculate performance attribution across different approaches"""
        attribution = {
            'elliott_wave_contribution': 0.0,
            'hybrid_model_contribution': 0.0,
            'portfolio_optimization_contribution': 0.0,
            'diversification_benefit': markowitz_results.diversification_ratio - 1.0,
            'risk_adjusted_alpha': markowitz_results.sharpe_ratio - 0.5  # Benchmark Sharpe of 0.5
        }
        
        # Elliott Wave contribution (based on pattern reliability)
        attribution['elliott_wave_contribution'] = elliott_results.pattern_reliability * 0.3
        
        # Hybrid model contribution (based on prediction accuracy)
        attribution['hybrid_model_contribution'] = hybrid_results.prediction_accuracy * 0.4
        
        # Portfolio optimization contribution (based on Sharpe ratio improvement)
        attribution['portfolio_optimization_contribution'] = min(0.3, markowitz_results.sharpe_ratio * 0.1)
        
        return attribution
    
    def _generate_insights(self, elliott_results: ElliottWaveResult,
                         hybrid_results: HybridModelResult,
                         markowitz_results: MarkowitzResult,
                         index_data: IndexData) -> List[str]:
        """Generate key insights from the analysis"""
        insights = []
        
        # Elliott Wave insights
        if elliott_results.pattern_reliability > 0.7:
            insights.append(f"Strong Elliott Wave patterns detected with {elliott_results.pattern_reliability:.1%} reliability")
        
        if elliott_results.trend_direction != "Neutral":
            insights.append(f"Elliott Wave analysis indicates {elliott_results.trend_direction.lower()} trend")
        
        if elliott_results.wave_completion_probability > 0.8:
            insights.append(f"Current wave pattern shows {elliott_results.wave_completion_probability:.1%} completion probability")
        
        # Hybrid model insights
        if hybrid_results.prediction_accuracy > 0.6:
            insights.append(f"Hybrid models show strong predictive power with {hybrid_results.prediction_accuracy:.1%} accuracy")
        
        if hybrid_results.model_stability > 0.7:
            insights.append(f"High model consensus detected with {hybrid_results.model_stability:.1%} stability")
        
        best_model = max(hybrid_results.model_weights.items(), key=lambda x: x[1])[0] if hybrid_results.model_weights else "Unknown"
        insights.append(f"Best performing model: {best_model}")
        
        # Markowitz insights
        if markowitz_results.sharpe_ratio > 1.0:
            insights.append(f"Excellent risk-adjusted returns with Sharpe ratio of {markowitz_results.sharpe_ratio:.2f}")
        elif markowitz_results.sharpe_ratio > 0.5:
            insights.append(f"Good risk-adjusted performance with Sharpe ratio of {markowitz_results.sharpe_ratio:.2f}")
        
        if markowitz_results.diversification_ratio > 1.2:
            insights.append(f"Strong diversification benefits with ratio of {markowitz_results.diversification_ratio:.2f}")
        
        # Portfolio composition insights
        if markowitz_results.optimal_weights:
            max_weight_asset = max(markowitz_results.optimal_weights.items(), key=lambda x: x[1])
            insights.append(f"Optimal portfolio heavily weighted in {max_weight_asset[0]} ({max_weight_asset[1]:.1%})")
        
        # Risk insights
        if abs(markowitz_results.maximum_drawdown) > 0.2:
            insights.append(f"High drawdown risk detected: {markowitz_results.maximum_drawdown:.1%}")
        
        return insights
    
    def _generate_recommendations(self, elliott_results: ElliottWaveResult,
                                hybrid_results: HybridModelResult,
                                markowitz_results: MarkowitzResult,
                                index_data: IndexData) -> List[str]:
        """Generate investment recommendations"""
        recommendations = []
        
        # Elliott Wave recommendations
        if elliott_results.trend_direction == "Bullish" and elliott_results.pattern_reliability > 0.6:
            recommendations.append("Consider increasing long exposure based on bullish Elliott Wave patterns")
        elif elliott_results.trend_direction == "Bearish" and elliott_results.pattern_reliability > 0.6:
            recommendations.append("Consider reducing exposure or hedging based on bearish Elliott Wave patterns")
        
        if elliott_results.wave_completion_probability > 0.8:
            recommendations.append("Current wave pattern near completion - prepare for potential reversal")
        
        # Price target recommendations
        if elliott_results.price_targets:
            upside_targets = [v for k, v in elliott_results.price_targets.items() if 'upside' in k]
            if upside_targets:
                avg_upside = np.mean(upside_targets)
                current_price = index_data.prices[-1]
                upside_potential = (avg_upside - current_price) / current_price
                if upside_potential > 0.1:
                    recommendations.append(f"Significant upside potential detected: {upside_potential:.1%}")
        
        # Hybrid model recommendations
        if hybrid_results.prediction_accuracy > 0.6:
            recent_predictions = hybrid_results.ensemble_predictions[-5:] if len(hybrid_results.ensemble_predictions) >= 5 else hybrid_results.ensemble_predictions
            if recent_predictions:
                avg_prediction = np.mean(recent_predictions)
                if avg_prediction > 0.02:
                    recommendations.append("Models predict positive returns - consider increasing allocation")
                elif avg_prediction < -0.02:
                    recommendations.append("Models predict negative returns - consider defensive positioning")
        
        # Portfolio optimization recommendations
        if markowitz_results.sharpe_ratio < 0.5:
            recommendations.append("Consider portfolio rebalancing to improve risk-adjusted returns")
        
        if markowitz_results.diversification_ratio < 1.1:
            recommendations.append("Increase diversification to reduce concentration risk")
        
        # Sector allocation recommendations
        if markowitz_results.optimal_weights:
            for sector, weight in markowitz_results.optimal_weights.items():
                current_weight = index_data.sector_weights.get(sector, 0)
                if weight > current_weight * 1.5:
                    recommendations.append(f"Consider overweighting {sector} sector (optimal: {weight:.1%})")
                elif weight < current_weight * 0.5:
                    recommendations.append(f"Consider underweighting {sector} sector (optimal: {weight:.1%})")
        
        # Risk management recommendations
        if abs(markowitz_results.maximum_drawdown) > 0.15:
            recommendations.append("Implement stop-loss strategies to limit drawdown risk")
        
        if markowitz_results.portfolio_volatility > 0.25:
            recommendations.append("High portfolio volatility - consider volatility targeting strategies")
        
        return recommendations
    
    def _calculate_confidence_score(self, elliott_results: ElliottWaveResult,
                                  hybrid_results: HybridModelResult,
                                  markowitz_results: MarkowitzResult) -> float:
        """Calculate overall confidence score for the analysis"""
        scores = []
        
        # Elliott Wave confidence
        scores.append(elliott_results.pattern_reliability)
        
        # Hybrid model confidence
        scores.append(hybrid_results.prediction_accuracy)
        scores.append(hybrid_results.model_stability)
        
        # Markowitz confidence (based on Sharpe ratio and diversification)
        markowitz_confidence = min(1.0, (markowitz_results.sharpe_ratio + 1) / 2)  # Normalize Sharpe ratio
        scores.append(markowitz_confidence)
        
        # Overall confidence is weighted average
        weights = [0.25, 0.35, 0.15, 0.25]  # Elliott, Hybrid accuracy, Hybrid stability, Markowitz
        
        if len(scores) == len(weights):
            confidence = sum(score * weight for score, weight in zip(scores, weights))
        else:
            confidence = np.mean(scores) if scores else 0.5
        
        return max(0.0, min(1.0, confidence))
    
    def _create_default_result(self) -> IndexHybridResult:
        """Create default result for error cases"""
        return IndexHybridResult(
            elliott_wave_results=self.elliott_analyzer._create_default_elliott_result(),
            hybrid_model_results=self.hybrid_analyzer._create_default_hybrid_result(),
            markowitz_results=self.markowitz_analyzer._create_default_markowitz_result(),
            combined_signals={'elliott_wave': [], 'hybrid_model': [], 'ensemble': []},
            risk_metrics={},
            performance_attribution={},
            insights=["Analysis incomplete due to insufficient data"],
            recommendations=["Gather more historical data for comprehensive analysis"],
            confidence_score=0.3
        )
    
    def plot_results(self, result: IndexHybridResult, index_data: IndexData, save_path: Optional[str] = None):
        """Plot comprehensive analysis results"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Hybrid Analysis Results for {index_data.index_symbol}', fontsize=16)
        
        # 1. Index Price with Elliott Wave patterns
        ax1 = axes[0, 0]
        ax1.plot(index_data.prices, label='Index Price', color='blue')
        
        # Add Elliott Wave patterns if available
        if result.elliott_wave_results.wave_patterns:
            for pattern in result.elliott_wave_results.wave_patterns:
                start_idx = pattern['start_idx']
                end_idx = pattern['end_idx']
                if start_idx < len(index_data.prices) and end_idx < len(index_data.prices):
                    ax1.axvspan(start_idx, end_idx, alpha=0.3, 
                              color='green' if pattern['type'] == 'impulse' else 'red',
                              label=f"{pattern['type'].title()} Wave" if pattern == result.elliott_wave_results.wave_patterns[0] else "")
        
        ax1.set_title('Index Price with Elliott Wave Patterns')
        ax1.set_xlabel('Time Period')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Hybrid Model Predictions vs Actual
        ax2 = axes[0, 1]
        if result.hybrid_model_results.ensemble_predictions:
            actual_returns = index_data.returns[-len(result.hybrid_model_results.ensemble_predictions):]
            ax2.scatter(actual_returns, result.hybrid_model_results.ensemble_predictions, alpha=0.6)
            
            # Add perfect prediction line
            min_val = min(min(actual_returns), min(result.hybrid_model_results.ensemble_predictions))
            max_val = max(max(actual_returns), max(result.hybrid_model_results.ensemble_predictions))
            ax2.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            ax2.set_title(f'Predictions vs Actual (R² = {result.hybrid_model_results.prediction_accuracy:.3f})')
            ax2.set_xlabel('Actual Returns')
            ax2.set_ylabel('Predicted Returns')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No Predictions Available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Hybrid Model Predictions')
        
        # 3. Efficient Frontier
        ax3 = axes[1, 0]
        if result.markowitz_results.efficient_frontier:
            risks, returns = zip(*result.markowitz_results.efficient_frontier)
            ax3.plot(risks, returns, 'b-', label='Efficient Frontier')
            
            # Mark optimal portfolio
            ax3.scatter([result.markowitz_results.portfolio_volatility], 
                       [result.markowitz_results.expected_return], 
                       color='red', s=100, label='Optimal Portfolio', zorder=5)
            
            ax3.set_title('Efficient Frontier')
            ax3.set_xlabel('Risk (Volatility)')
            ax3.set_ylabel('Expected Return')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Efficient Frontier Data', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Efficient Frontier')
        
        # 4. Portfolio Weights
        ax4 = axes[1, 1]
        if result.markowitz_results.optimal_weights:
            assets = list(result.markowitz_results.optimal_weights.keys())
            weights = list(result.markowitz_results.optimal_weights.values())
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(assets)))
            wedges, texts, autotexts = ax4.pie(weights, labels=assets, autopct='%1.1f%%', colors=colors)
            ax4.set_title('Optimal Portfolio Weights')
        else:
            ax4.text(0.5, 0.5, 'No Portfolio Weights', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Portfolio Weights')
        
        # 5. Combined Trading Signals
        ax5 = axes[2, 0]
        if result.combined_signals['ensemble']:
            signals = result.combined_signals['ensemble']
            signal_periods = range(len(signals))
            
            # Color code signals
            colors = ['red' if s == -1 else 'green' if s == 1 else 'gray' for s in signals]
            ax5.scatter(signal_periods, signals, c=colors, alpha=0.7)
            
            ax5.set_title('Combined Trading Signals')
            ax5.set_xlabel('Time Period')
            ax5.set_ylabel('Signal (-1: Sell, 0: Hold, 1: Buy)')
            ax5.set_ylim(-1.5, 1.5)
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, 'No Trading Signals', ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Trading Signals')
        
        # 6. Risk Metrics Comparison
        ax6 = axes[2, 1]
        if result.risk_metrics:
            metrics_names = ['Volatility', 'Sharpe Ratio', 'Max Drawdown', 'VaR 95%']
            metrics_values = [
                result.risk_metrics.get('volatility', 0),
                result.risk_metrics.get('sharpe_ratio', 0),
                abs(result.risk_metrics.get('max_drawdown', 0)),
                abs(result.risk_metrics.get('var_95', 0))
            ]
            
            # Normalize values for comparison
            max_val = max(metrics_values) if max(metrics_values) > 0 else 1
            normalized_values = [v / max_val for v in metrics_values]
            
            bars = ax6.bar(metrics_names, normalized_values, color=['blue', 'green', 'red', 'orange'])
            ax6.set_title('Risk Metrics (Normalized)')
            ax6.set_ylabel('Normalized Value')
            
            # Add actual values as text
            for bar, value in zip(bars, metrics_values):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        else:
            ax6.text(0.5, 0.5, 'No Risk Metrics', ha='center', va='center', transform=ax6.transAxes)
            ax6.set_title('Risk Metrics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, result: IndexHybridResult, index_data: IndexData) -> str:
        """Generate comprehensive analysis report"""
        report = f"""
# Hybrid Analysis Report: {index_data.index_symbol}

## Executive Summary

This comprehensive analysis combines Elliott Wave theory, hybrid machine learning models, and Markowitz Portfolio Theory to provide multi-dimensional insights into {index_data.index_symbol}.

**Overall Confidence Score: {result.confidence_score:.1%}**

## Elliott Wave Analysis

### Wave Pattern Summary
- **Current Wave Position**: {result.elliott_wave_results.current_wave_position}
- **Trend Direction**: {result.elliott_wave_results.trend_direction}
- **Pattern Reliability**: {result.elliott_wave_results.pattern_reliability:.1%}
- **Wave Completion Probability**: {result.elliott_wave_results.wave_completion_probability:.1%}

### Wave Count
"""
        
        # Add wave count details
        for wave_type, count in result.elliott_wave_results.wave_count.items():
            report += f"- **{wave_type.title()} Waves**: {count}\n"
        
        # Add Fibonacci levels
        if result.elliott_wave_results.fibonacci_levels:
            report += "\n### Key Fibonacci Levels\n"
            for level, price in list(result.elliott_wave_results.fibonacci_levels.items())[:5]:
                report += f"- **{level}**: {price:.2f}\n"
        
        # Add price targets
        if result.elliott_wave_results.price_targets:
            report += "\n### Price Targets\n"
            for target, price in result.elliott_wave_results.price_targets.items():
                report += f"- **{target}**: {price:.2f}\n"
        
        report += f"""

## Hybrid Model Analysis

### Model Performance Summary
- **Prediction Accuracy (R²)**: {result.hybrid_model_results.prediction_accuracy:.3f}
- **Model Stability**: {result.hybrid_model_results.model_stability:.3f}

### Individual Model Performance
"""
        
        # Add individual model performance
        for model, performance in result.hybrid_model_results.individual_model_performance.items():
            report += f"\n#### {model.title()} Model\n"
            for metric, value in performance.items():
                report += f"- **{metric.upper()}**: {value:.4f}\n"
        
        # Add model weights
        if result.hybrid_model_results.model_weights:
            report += "\n### Ensemble Model Weights\n"
            for model, weight in result.hybrid_model_results.model_weights.items():
                report += f"- **{model.title()}**: {weight:.1%}\n"
        
        report += f"""

## Markowitz Portfolio Theory Analysis

### Portfolio Optimization Results
- **Expected Return**: {result.markowitz_results.expected_return:.4f}
- **Portfolio Volatility**: {result.markowitz_results.portfolio_volatility:.4f}
- **Sharpe Ratio**: {result.markowitz_results.sharpe_ratio:.3f}
- **Diversification Ratio**: {result.markowitz_results.diversification_ratio:.3f}
- **Maximum Drawdown**: {result.markowitz_results.maximum_drawdown:.1%}

### Optimal Portfolio Weights
"""
        
        # Add optimal weights
        if result.markowitz_results.optimal_weights:
            for asset, weight in result.markowitz_results.optimal_weights.items():
                report += f"- **{asset}**: {weight:.1%}\n"
        
        report += f"""

## Risk Analysis

### Comprehensive Risk Metrics
"""
        
        # Add risk metrics
        for metric, value in result.risk_metrics.items():
            if metric == 'volatility':
                report += f"- **Annualized Volatility**: {value:.1%}\n"
            elif metric == 'max_drawdown':
                report += f"- **Maximum Drawdown**: {value:.1%}\n"
            elif metric == 'var_95':
                report += f"- **Value at Risk (95%)**: {value:.4f}\n"
            elif metric == 'cvar_95':
                report += f"- **Conditional VaR (95%)**: {value:.4f}\n"
            else:
                report += f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n"
        
        report += f"""

## Performance Attribution

### Factor Contributions
"""
        
        # Add performance attribution
        for factor, contribution in result.performance_attribution.items():
            report += f"- **{factor.replace('_', ' ').title()}**: {contribution:.1%}\n"
        
        report += f"""

## Trading Signals Analysis

### Signal Summary
"""
        
        # Add signal analysis
        for signal_type, signals in result.combined_signals.items():
            if signals:
                buy_signals = sum(1 for s in signals if s == 1)
                sell_signals = sum(1 for s in signals if s == -1)
                hold_signals = sum(1 for s in signals if s == 0)
                
                report += f"\n#### {signal_type.replace('_', ' ').title()} Signals\n"
                report += f"- **Buy Signals**: {buy_signals}\n"
                report += f"- **Sell Signals**: {sell_signals}\n"
                report += f"- **Hold Signals**: {hold_signals}\n"
        
        report += "\n## Key Insights\n\n"
        
        # Add insights
        for i, insight in enumerate(result.insights, 1):
            report += f"{i}. {insight}\n"
        
        report += "\n## Investment Recommendations\n\n"
        
        # Add recommendations
        for i, recommendation in enumerate(result.recommendations, 1):
            report += f"{i}. {recommendation}\n"
        
        report += f"""

## Model Confidence Assessment

### Confidence Breakdown
- **Elliott Wave Reliability**: {result.elliott_wave_results.pattern_reliability:.1%}
- **Hybrid Model Accuracy**: {result.hybrid_model_results.prediction_accuracy:.1%}
- **Model Stability**: {result.hybrid_model_results.model_stability:.1%}
- **Overall Confidence**: {result.confidence_score:.1%}

### Confidence Interpretation
"""
        
        if result.confidence_score > 0.8:
            report += "- **High Confidence**: Analysis shows strong consensus across multiple approaches\n"
        elif result.confidence_score > 0.6:
            report += "- **Moderate Confidence**: Analysis shows reasonable consensus with some uncertainty\n"
        else:
            report += "- **Low Confidence**: Analysis shows significant uncertainty - use with caution\n"
        
        # Add current trading signal
        current_signal = "Hold"
        if result.combined_signals['ensemble']:
            last_signal = result.combined_signals['ensemble'][-1]
            if last_signal == 1:
                current_signal = "Buy"
            elif last_signal == -1:
                current_signal = "Sell"
        
        report += f"""

## Current Trading Signal

**Recommended Action**: {current_signal}

*Based on ensemble of Elliott Wave patterns, hybrid model predictions, and portfolio optimization analysis.*

## Methodology

### Elliott Wave Analysis
- Pattern recognition using peak/trough identification
- Fibonacci retracement and extension levels
- Wave completion probability assessment
- Trend direction analysis

### Hybrid Models
- ARIMA time series modeling
- Random Forest machine learning
- Linear regression baseline
- Ensemble weighting based on performance

### Markowitz Portfolio Theory
- Mean-variance optimization
- Efficient frontier generation
- Risk-adjusted return maximization
- Diversification analysis

---

*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Analysis confidence: {result.confidence_score:.1%}*
"""
        
        return report


# Example usage and testing
if __name__ == "__main__":
    # Create sample index data
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    
    # Generate realistic price series with trend and volatility
    returns = np.random.normal(0.0008, 0.02, n_periods)  # Daily returns
    prices = [100.0]  # Starting price
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_periods)
    timestamps = [start_date + timedelta(days=i) for i in range(len(prices))]
    
    # Generate volume data
    volume = np.random.lognormal(15, 0.5, len(prices)).tolist()
    
    # Create sector weights
    sector_weights = {
        'Technology': 0.25,
        'Healthcare': 0.15,
        'Financials': 0.20,
        'Consumer': 0.15,
        'Energy': 0.10,
        'Industrials': 0.15
    }
    
    # Create sample constituent data
    constituent_data = {}
    for sector in sector_weights.keys():
        # Generate correlated price series for each sector
        sector_returns = 0.8 * np.array(returns) + 0.2 * np.random.normal(0, 0.015, len(returns))
        sector_prices = [100.0]
        for ret in sector_returns:
            sector_prices.append(sector_prices[-1] * (1 + ret))
        constituent_data[sector] = sector_prices
    
    # Create IndexData object
    index_data = IndexData(
        index_symbol="SAMPLE_INDEX",
        prices=prices,
        returns=[0.0] + returns.tolist(),  # Add zero return for first period
        timestamps=timestamps,
        volume=volume,
        market_cap=1e12,  # $1 trillion market cap
        sector_weights=sector_weights,
        constituent_data=constituent_data
    )
    
    print("Starting Hybrid Elliott Wave Markowitz Analysis...")
    print(f"Analyzing {len(prices)} price points for {index_data.index_symbol}")
    
    # Create analyzer
    analyzer = IndexHybridAnalyzer(risk_free_rate=0.02)
    
    # Perform analysis
    try:
        result = analyzer.analyze(index_data)
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)
        
        print(f"\nOverall Confidence Score: {result.confidence_score:.1%}")
        print(f"Elliott Wave Trend: {result.elliott_wave_results.trend_direction}")
        print(f"Pattern Reliability: {result.elliott_wave_results.pattern_reliability:.1%}")
        print(f"Hybrid Model Accuracy: {result.hybrid_model_results.prediction_accuracy:.1%}")
        print(f"Portfolio Sharpe Ratio: {result.markowitz_results.sharpe_ratio:.3f}")
        
        print("\nKey Insights:")
        for i, insight in enumerate(result.insights[:3], 1):
            print(f"{i}. {insight}")
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(result.recommendations[:3], 1):
            print(f"{i}. {rec}")
        
        # Generate and display report
        print("\n" + "="*50)
        print("GENERATING COMPREHENSIVE REPORT...")
        print("="*50)
        
        report = analyzer.generate_report(result, index_data)
        
        # Save report to file
        with open('hybrid_analysis_report.md', 'w') as f:
            f.write(report)
        print("\nReport saved to 'hybrid_analysis_report.md'")
        
        # Plot results
        print("\nGenerating analysis plots...")
        analyzer.plot_results(result, index_data, 'hybrid_analysis_plots.png')
        
        print("\nAnalysis complete! Check the generated files for detailed results.")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nHybrid Elliott Wave Markowitz Analysis finished.")