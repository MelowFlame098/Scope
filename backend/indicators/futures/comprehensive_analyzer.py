"""Comprehensive Futures Trading Analysis

This module orchestrates momentum, mean reversion, and reinforcement learning analysis
for futures trading, providing combined signals and comprehensive performance metrics.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro

# Conditional imports for advanced features
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Some ensemble features will be limited.")

try:
    from hmmlearn import hmm
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    warnings.warn("hmmlearn not available. HMM regime detection will be limited.")

# Import local modules
from .technical_indicators import MomentumAnalyzer, MomentumResult, FuturesData
from .mean_reversion import MeanReversionAnalyzer, MeanReversionResult
from .rl_models import RLAnalyzer, RLAgentResult

# Conditional imports for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available. Plotting functionality will be limited.")

@dataclass
class FuturesMomentumMeanReversionResult:
    """Combined results from futures momentum, mean reversion, and RL analysis"""
    momentum_results: MomentumResult
    mean_reversion_results: MeanReversionResult
    rl_results: Dict[str, RLAgentResult]
    combined_signals: List[str]
    strategy_performance: Dict[str, Dict[str, float]]
    risk_metrics: Dict[str, float]
    insights: List[str]
    recommendations: List[str]
    model_confidence: Dict[str, float]

@dataclass
class EnsembleModelResults:
    """Results from ensemble modeling"""
    ensemble_forecast: np.ndarray
    individual_forecasts: Dict[str, np.ndarray]
    model_weights: Dict[str, float]
    ensemble_accuracy: float
    individual_accuracies: Dict[str, float]
    feature_importance: Dict[str, float]
    cross_validation_scores: Dict[str, List[float]]
    model_comparison: Dict[str, Dict[str, float]]

@dataclass
class AdvancedRiskMetrics:
    """Advanced risk assessment metrics"""
    value_at_risk: Dict[str, float]  # Different confidence levels
    expected_shortfall: Dict[str, float]
    tail_risk: float
    maximum_drawdown: float
    drawdown_duration: int
    regime_risk: Dict[str, Dict[str, float]]
    stress_test_results: Dict[str, float]
    risk_adjusted_returns: Dict[str, float]  # Sharpe, Sortino, Calmar
    higher_moments: Dict[str, float]  # Skewness, Kurtosis
    liquidity_risk: float
    correlation_risk: float

@dataclass
class MachineLearningInsights:
    """Machine learning based insights"""
    anomaly_scores: np.ndarray
    anomaly_dates: List[datetime]
    clustering_labels: np.ndarray
    cluster_centers: np.ndarray
    pattern_recognition: Dict[str, float]
    feature_importance_ml: Dict[str, float]
    regime_probabilities: Optional[np.ndarray]
    regime_states: Optional[np.ndarray]
    predictive_accuracy: Dict[str, float]
    market_microstructure: Dict[str, float]

@dataclass
class ComprehensiveDiagnostics:
    """Comprehensive model diagnostics"""
    statistical_tests: Dict[str, Dict[str, float]]
    model_validation: Dict[str, Dict[str, float]]
    residual_analysis: Dict[str, float]
    goodness_of_fit: Dict[str, float]
    cross_validation_results: Dict[str, List[float]]
    stability_tests: Dict[str, float]
    overall_quality_score: float

@dataclass
class EnhancedFuturesAnalysisResult:
    """Enhanced comprehensive futures analysis result"""
    basic_result: FuturesMomentumMeanReversionResult
    ensemble_results: Optional[EnsembleModelResults]
    advanced_risk: Optional[AdvancedRiskMetrics]
    ml_insights: Optional[MachineLearningInsights]
    diagnostics: Optional[ComprehensiveDiagnostics]
    enhanced_insights: List[str]
    strategic_recommendations: List[str]
    confidence_intervals: Dict[str, Tuple[float, float]]
    model_uncertainty: Dict[str, float]
    analysis_timestamp: datetime

class FuturesMomentumMeanReversionAnalyzer:
    """Comprehensive analyzer for futures momentum, mean reversion, and RL strategies"""
    
    def __init__(self, enable_ensemble_methods: bool = False,
                 enable_advanced_risk: bool = False,
                 enable_ml_insights: bool = False,
                 enable_comprehensive_diagnostics: bool = False,
                 rolling_window: int = 252,
                 forecast_horizon: int = 5):
        """
        Initialize enhanced futures analyzer
        
        Args:
            enable_ensemble_methods: Enable ensemble modeling
            enable_advanced_risk: Enable advanced risk assessment
            enable_ml_insights: Enable machine learning insights
            enable_comprehensive_diagnostics: Enable comprehensive diagnostics
            rolling_window: Rolling window for calculations
            forecast_horizon: Forecast horizon for predictions
        """
        self.momentum_analyzer = MomentumAnalyzer()
        self.mean_reversion_analyzer = MeanReversionAnalyzer()
        self.rl_analyzer = RLAnalyzer()
        
        # Enhanced features flags
        self.enable_ensemble_methods = enable_ensemble_methods
        self.enable_advanced_risk = enable_advanced_risk
        self.enable_ml_insights = enable_ml_insights
        self.enable_comprehensive_diagnostics = enable_comprehensive_diagnostics
        
        # Parameters
        self.rolling_window = rolling_window
        self.forecast_horizon = forecast_horizon
        
        # Initialize scaler for ML features
        self.scaler = StandardScaler()
        
        # Check library availability
        self.xgboost_available = XGBOOST_AVAILABLE
        self.hmmlearn_available = HMMLEARN_AVAILABLE
    
    def analyze(self, futures_data: FuturesData, 
               rl_agent_types: List[str] = ["PPO", "SAC", "DDPG"],
               rl_train_timesteps: int = 5000) -> FuturesMomentumMeanReversionResult:
        """Perform comprehensive futures analysis"""
        
        try:
            # Perform individual analyses
            print("Performing momentum analysis...")
            momentum_results = self.momentum_analyzer.analyze(futures_data)
            
            print("Performing mean reversion analysis...")
            mean_reversion_results = self.mean_reversion_analyzer.analyze(futures_data)
            
            print("Performing RL analysis...")
            rl_results = self.rl_analyzer.analyze(
                futures_data, rl_agent_types, rl_train_timesteps
            )
            
            # Combine signals from all strategies
            combined_signals = self._combine_signals(
                momentum_results, mean_reversion_results, rl_results
            )
            
            # Calculate strategy performance
            strategy_performance = self._calculate_strategy_performance(
                futures_data, momentum_results, mean_reversion_results, 
                rl_results, combined_signals
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                futures_data, combined_signals
            )
            
            # Generate insights and recommendations
            insights = self._generate_insights(
                momentum_results, mean_reversion_results, rl_results, strategy_performance
            )
            
            recommendations = self._generate_recommendations(
                momentum_results, mean_reversion_results, rl_results, risk_metrics
            )
            
            # Calculate model confidence
            model_confidence = self._calculate_model_confidence(
                momentum_results, mean_reversion_results, rl_results
            )
            
            return FuturesMomentumMeanReversionResult(
                momentum_results=momentum_results,
                mean_reversion_results=mean_reversion_results,
                rl_results=rl_results,
                combined_signals=combined_signals,
                strategy_performance=strategy_performance,
                risk_metrics=risk_metrics,
                insights=insights,
                recommendations=recommendations,
                model_confidence=model_confidence
            )
            
        except Exception as e:
            print(f"Error in comprehensive analysis: {e}")
            return self._create_default_result(futures_data)
    
    def analyze_enhanced(self, futures_data: FuturesData,
                        rl_agent_types: List[str] = ["PPO", "SAC", "DDPG"],
                        rl_train_timesteps: int = 5000) -> EnhancedFuturesAnalysisResult:
        """Perform enhanced comprehensive futures analysis with advanced features"""
        
        try:
            # First perform basic analysis
            basic_result = self.analyze(futures_data, rl_agent_types, rl_train_timesteps)
            
            # Initialize enhanced components
            ensemble_results = None
            advanced_risk = None
            ml_insights = None
            diagnostics = None
            
            # Perform ensemble analysis if enabled
            if self.enable_ensemble_methods:
                print("Performing ensemble modeling...")
                ensemble_results = self._perform_ensemble_analysis(futures_data)
            
            # Perform advanced risk assessment if enabled
            if self.enable_advanced_risk:
                print("Performing advanced risk assessment...")
                advanced_risk = self._perform_advanced_risk_assessment(futures_data, basic_result)
            
            # Generate ML insights if enabled
            if self.enable_ml_insights:
                print("Generating machine learning insights...")
                ml_insights = self._generate_ml_insights(futures_data, basic_result)
            
            # Perform comprehensive diagnostics if enabled
            if self.enable_comprehensive_diagnostics:
                print("Performing comprehensive diagnostics...")
                diagnostics = self._perform_comprehensive_diagnostics(futures_data, basic_result)
            
            # Generate enhanced insights and recommendations
            enhanced_insights = self._generate_enhanced_insights(
                basic_result, ensemble_results, advanced_risk, ml_insights, diagnostics
            )
            
            strategic_recommendations = self._generate_strategic_recommendations(
                basic_result, ensemble_results, advanced_risk, ml_insights, diagnostics
            )
            
            # Calculate confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                futures_data, basic_result, ensemble_results
            )
            
            # Assess model uncertainty
            model_uncertainty = self._assess_model_uncertainty(
                ensemble_results, diagnostics, basic_result
            )
            
            return EnhancedFuturesAnalysisResult(
                basic_result=basic_result,
                ensemble_results=ensemble_results,
                advanced_risk=advanced_risk,
                ml_insights=ml_insights,
                diagnostics=diagnostics,
                enhanced_insights=enhanced_insights,
                strategic_recommendations=strategic_recommendations,
                confidence_intervals=confidence_intervals,
                model_uncertainty=model_uncertainty,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            print(f"Error in enhanced analysis: {e}")
            return self._create_fallback_enhanced_result(futures_data)
    
    def _combine_signals(self, momentum_results: MomentumResult,
                        mean_reversion_results: MeanReversionResult,
                        rl_results: Dict[str, RLAgentResult]) -> List[str]:
        """Combine signals from all strategies using voting mechanism"""
        
        n_periods = len(momentum_results.momentum_signals)
        combined_signals = []
        
        for i in range(n_periods):
            votes = {"BUY": 0, "SELL": 0, "HOLD": 0}
            
            # Momentum vote
            momentum_signal = momentum_results.momentum_signals[i]
            if momentum_signal in ["STRONG_BUY", "BUY"]:
                votes["BUY"] += 2 if momentum_signal == "STRONG_BUY" else 1
            elif momentum_signal in ["STRONG_SELL", "SELL"]:
                votes["SELL"] += 2 if momentum_signal == "STRONG_SELL" else 1
            else:
                votes["HOLD"] += 1
            
            # Mean reversion vote
            mr_signal = mean_reversion_results.mean_reversion_signals[i]
            if mr_signal in ["STRONG_BUY", "BUY"]:
                votes["BUY"] += 2 if mr_signal == "STRONG_BUY" else 1
            elif mr_signal in ["STRONG_SELL", "SELL"]:
                votes["SELL"] += 2 if mr_signal == "STRONG_SELL" else 1
            else:
                votes["HOLD"] += 1
            
            # RL votes
            for agent_type, rl_result in rl_results.items():
                if i < len(rl_result.actions):
                    action = rl_result.actions[i]
                    if action == 1:  # Buy
                        votes["BUY"] += 1
                    elif action == 2:  # Sell
                        votes["SELL"] += 1
                    else:  # Hold
                        votes["HOLD"] += 1
            
            # Determine combined signal
            max_votes = max(votes.values())
            if votes["BUY"] == max_votes and max_votes > votes["HOLD"]:
                if max_votes >= 4:
                    combined_signals.append("STRONG_BUY")
                else:
                    combined_signals.append("BUY")
            elif votes["SELL"] == max_votes and max_votes > votes["HOLD"]:
                if max_votes >= 4:
                    combined_signals.append("STRONG_SELL")
                else:
                    combined_signals.append("SELL")
            else:
                combined_signals.append("HOLD")
        
        return combined_signals
    
    def _calculate_strategy_performance(self, futures_data: FuturesData,
                                      momentum_results: MomentumResult,
                                      mean_reversion_results: MeanReversionResult,
                                      rl_results: Dict[str, RLAgentResult],
                                      combined_signals: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each strategy"""
        
        performance = {}
        
        # Momentum strategy performance
        momentum_returns = self._calculate_signal_returns(
            futures_data.returns, momentum_results.momentum_signals
        )
        performance["momentum"] = {
            "total_return": sum(momentum_returns),
            "sharpe_ratio": self._calculate_sharpe_ratio(momentum_returns),
            "max_drawdown": self._calculate_max_drawdown_from_returns(momentum_returns),
            "win_rate": self._calculate_win_rate(momentum_returns)
        }
        
        # Mean reversion strategy performance
        mr_returns = self._calculate_signal_returns(
            futures_data.returns, mean_reversion_results.mean_reversion_signals
        )
        performance["mean_reversion"] = {
            "total_return": sum(mr_returns),
            "sharpe_ratio": self._calculate_sharpe_ratio(mr_returns),
            "max_drawdown": self._calculate_max_drawdown_from_returns(mr_returns),
            "win_rate": self._calculate_win_rate(mr_returns)
        }
        
        # RL strategy performance
        for agent_type, rl_result in rl_results.items():
            performance[f"rl_{agent_type.lower()}"] = {
                "total_return": rl_result.cumulative_returns[-1] if rl_result.cumulative_returns else 0,
                "sharpe_ratio": rl_result.sharpe_ratio,
                "max_drawdown": rl_result.max_drawdown,
                "win_rate": rl_result.win_rate
            }
        
        # Combined strategy performance
        combined_returns = self._calculate_signal_returns(
            futures_data.returns, combined_signals
        )
        performance["combined"] = {
            "total_return": sum(combined_returns),
            "sharpe_ratio": self._calculate_sharpe_ratio(combined_returns),
            "max_drawdown": self._calculate_max_drawdown_from_returns(combined_returns),
            "win_rate": self._calculate_win_rate(combined_returns)
        }
        
        return performance
    
    def _perform_ensemble_analysis(self, futures_data: FuturesData) -> EnsembleModelResults:
        """Perform ensemble modeling with multiple forecasting methods"""
        
        try:
            # Prepare features for ensemble modeling
            features = self._prepare_ensemble_features(futures_data)
            
            # Initialize models and forecasts
            individual_forecasts = {}
            individual_accuracies = {}
            cross_validation_scores = {}
            
            # Random Forest forecast
            rf_forecast, rf_accuracy, rf_cv_scores = self._random_forest_forecast(features, futures_data.returns)
            individual_forecasts['random_forest'] = rf_forecast
            individual_accuracies['random_forest'] = rf_accuracy
            cross_validation_scores['random_forest'] = rf_cv_scores
            
            # XGBoost forecast (if available)
            if self.xgboost_available:
                xgb_forecast, xgb_accuracy, xgb_cv_scores = self._xgboost_forecast(features, futures_data.returns)
                individual_forecasts['xgboost'] = xgb_forecast
                individual_accuracies['xgboost'] = xgb_accuracy
                cross_validation_scores['xgboost'] = xgb_cv_scores
            
            # Ridge regression forecast
            ridge_forecast, ridge_accuracy, ridge_cv_scores = self._ridge_forecast(features, futures_data.returns)
            individual_forecasts['ridge'] = ridge_forecast
            individual_accuracies['ridge'] = ridge_accuracy
            cross_validation_scores['ridge'] = ridge_cv_scores
            
            # Elastic Net forecast
            elastic_forecast, elastic_accuracy, elastic_cv_scores = self._elastic_net_forecast(features, futures_data.returns)
            individual_forecasts['elastic_net'] = elastic_forecast
            individual_accuracies['elastic_net'] = elastic_accuracy
            cross_validation_scores['elastic_net'] = elastic_cv_scores
            
            # Calculate ensemble weights based on accuracy
            total_accuracy = sum(individual_accuracies.values())
            model_weights = {model: acc / total_accuracy for model, acc in individual_accuracies.items()}
            
            # Create ensemble forecast
            ensemble_forecast = np.zeros(len(list(individual_forecasts.values())[0]))
            for model, forecast in individual_forecasts.items():
                ensemble_forecast += model_weights[model] * forecast
            
            # Calculate ensemble accuracy
            actual_returns = np.array(futures_data.returns[-len(ensemble_forecast):])
            ensemble_accuracy = 1.0 / (1.0 + mean_squared_error(actual_returns, ensemble_forecast))
            
            # Calculate feature importance (using Random Forest)
            feature_importance = self._calculate_ensemble_feature_importance(features, futures_data.returns)
            
            # Model comparison
            model_comparison = self._compare_ensemble_models(individual_accuracies, cross_validation_scores)
            
            return EnsembleModelResults(
                ensemble_forecast=ensemble_forecast,
                individual_forecasts=individual_forecasts,
                model_weights=model_weights,
                ensemble_accuracy=ensemble_accuracy,
                individual_accuracies=individual_accuracies,
                feature_importance=feature_importance,
                cross_validation_scores=cross_validation_scores,
                model_comparison=model_comparison
            )
            
        except Exception as e:
            print(f"Error in ensemble analysis: {e}")
            return self._create_default_ensemble_result()
    
    def _prepare_ensemble_features(self, futures_data: FuturesData) -> np.ndarray:
        """Prepare features for ensemble modeling"""
        
        returns = np.array(futures_data.returns)
        prices = np.array(futures_data.prices)
        
        features_list = []
        
        # Lagged returns (1-5 periods)
        for lag in range(1, 6):
            lagged_returns = np.roll(returns, lag)
            lagged_returns[:lag] = 0  # Fill initial values with 0
            features_list.append(lagged_returns)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            # Rolling mean
            rolling_mean = pd.Series(returns).rolling(window=window, min_periods=1).mean().values
            features_list.append(rolling_mean)
            
            # Rolling std
            rolling_std = pd.Series(returns).rolling(window=window, min_periods=1).std().fillna(0).values
            features_list.append(rolling_std)
        
        # Technical indicators
        # Simple moving averages
        for window in [10, 20, 50]:
            sma = pd.Series(prices).rolling(window=window, min_periods=1).mean().values
            price_to_sma = prices / sma
            features_list.append(price_to_sma)
        
        # Price momentum
        for period in [5, 10, 20]:
            momentum = (prices / np.roll(prices, period)) - 1
            momentum[:period] = 0
            features_list.append(momentum)
        
        # Volatility features
        volatility = pd.Series(returns).rolling(window=20, min_periods=1).std().fillna(0).values
        features_list.append(volatility)
        
        # Volume features (if available)
        if hasattr(futures_data, 'volume') and futures_data.volume:
            volume = np.array(futures_data.volume)
            volume_ma = pd.Series(volume).rolling(window=20, min_periods=1).mean().values
            volume_ratio = volume / volume_ma
            features_list.append(volume_ratio)
        
        # Stack all features
        features = np.column_stack(features_list)
        
        # Handle any remaining NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _random_forest_forecast(self, features: np.ndarray, returns: List[float]) -> Tuple[np.ndarray, float, List[float]]:
        """Generate Random Forest forecast"""
        
        try:
            X, y = self._prepare_supervised_data(features, returns)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            cv_scores = -cv_scores  # Convert to positive MSE
            
            # Fit model and generate forecast
            model.fit(X, y)
            forecast = model.predict(X[-self.forecast_horizon:])
            
            # Calculate accuracy
            accuracy = 1.0 / (1.0 + np.mean(cv_scores))
            
            return forecast, accuracy, cv_scores.tolist()
            
        except Exception as e:
            print(f"Error in Random Forest forecast: {e}")
            return np.zeros(self.forecast_horizon), 0.0, []
    
    def _xgboost_forecast(self, features: np.ndarray, returns: List[float]) -> Tuple[np.ndarray, float, List[float]]:
        """Generate XGBoost forecast"""
        
        try:
            X, y = self._prepare_supervised_data(features, returns)
            
            model = xgb.XGBRegressor(n_estimators=100, random_state=42, max_depth=6)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            cv_scores = -cv_scores
            
            # Fit model and generate forecast
            model.fit(X, y)
            forecast = model.predict(X[-self.forecast_horizon:])
            
            # Calculate accuracy
            accuracy = 1.0 / (1.0 + np.mean(cv_scores))
            
            return forecast, accuracy, cv_scores.tolist()
            
        except Exception as e:
            print(f"Error in XGBoost forecast: {e}")
            return np.zeros(self.forecast_horizon), 0.0, []
    
    def _ridge_forecast(self, features: np.ndarray, returns: List[float]) -> Tuple[np.ndarray, float, List[float]]:
        """Generate Ridge regression forecast"""
        
        try:
            X, y = self._prepare_supervised_data(features, returns)
            
            model = Ridge(alpha=1.0, random_state=42)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            cv_scores = -cv_scores
            
            # Fit model and generate forecast
            model.fit(X, y)
            forecast = model.predict(X[-self.forecast_horizon:])
            
            # Calculate accuracy
            accuracy = 1.0 / (1.0 + np.mean(cv_scores))
            
            return forecast, accuracy, cv_scores.tolist()
            
        except Exception as e:
            print(f"Error in Ridge forecast: {e}")
            return np.zeros(self.forecast_horizon), 0.0, []
    
    def _elastic_net_forecast(self, features: np.ndarray, returns: List[float]) -> Tuple[np.ndarray, float, List[float]]:
        """Generate Elastic Net forecast"""
        
        try:
            X, y = self._prepare_supervised_data(features, returns)
            
            model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
            
            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
            cv_scores = -cv_scores
            
            # Fit model and generate forecast
            model.fit(X, y)
            forecast = model.predict(X[-self.forecast_horizon:])
            
            # Calculate accuracy
            accuracy = 1.0 / (1.0 + np.mean(cv_scores))
            
            return forecast, accuracy, cv_scores.tolist()
            
        except Exception as e:
            print(f"Error in Elastic Net forecast: {e}")
            return np.zeros(self.forecast_horizon), 0.0, []
    
    def _prepare_supervised_data(self, features: np.ndarray, returns: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for supervised learning"""
        
        returns_array = np.array(returns)
        
        # Use features to predict next period returns
        X = features[:-1]  # All but last observation
        y = returns_array[1:]  # All but first observation (shifted targets)
        
        return X, y
    
    def _calculate_ensemble_feature_importance(self, features: np.ndarray, returns: List[float]) -> Dict[str, float]:
        """Calculate feature importance using Random Forest"""
        
        try:
            X, y = self._prepare_supervised_data(features, returns)
            
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Create feature names
            feature_names = []
            feature_names.extend([f'lag_{i}_returns' for i in range(1, 6)])
            for window in [5, 10, 20]:
                feature_names.extend([f'rolling_mean_{window}', f'rolling_std_{window}'])
            for window in [10, 20, 50]:
                feature_names.append(f'price_to_sma_{window}')
            for period in [5, 10, 20]:
                feature_names.append(f'momentum_{period}')
            feature_names.append('volatility')
            
            # Ensure we have the right number of feature names
            n_features = X.shape[1]
            if len(feature_names) < n_features:
                feature_names.extend([f'feature_{i}' for i in range(len(feature_names), n_features)])
            elif len(feature_names) > n_features:
                feature_names = feature_names[:n_features]
            
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            
            return importance_dict
            
        except Exception as e:
            print(f"Error calculating feature importance: {e}")
            return {}
    
    def _compare_ensemble_models(self, individual_accuracies: Dict[str, float], 
                               cross_validation_scores: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Compare ensemble models performance"""
        
        comparison = {}
        
        for model_name in individual_accuracies.keys():
            cv_scores = cross_validation_scores.get(model_name, [])
            
            comparison[model_name] = {
                'accuracy': individual_accuracies[model_name],
                'cv_mean': np.mean(cv_scores) if cv_scores else 0.0,
                'cv_std': np.std(cv_scores) if cv_scores else 0.0,
                'stability': 1.0 / (1.0 + np.std(cv_scores)) if cv_scores else 0.0
            }
        
        return comparison
    
    def _create_default_ensemble_result(self) -> EnsembleModelResults:
        """Create default ensemble result for error cases"""
        
        return EnsembleModelResults(
            ensemble_forecast=np.zeros(self.forecast_horizon),
            individual_forecasts={},
            model_weights={},
            ensemble_accuracy=0.0,
            individual_accuracies={},
            feature_importance={},
            cross_validation_scores={},
            model_comparison={}
        )
    
    def _perform_advanced_risk_assessment(self, returns: List[float], prices: List[float]) -> AdvancedRiskMetrics:
        """Perform comprehensive risk assessment"""
        
        try:
            returns_array = np.array(returns)
            prices_array = np.array(prices)
            
            # Calculate VaR and CVaR
            var_95, var_99 = self._calculate_var(returns_array)
            cvar_95, cvar_99 = self._calculate_cvar(returns_array)
            
            # Calculate maximum drawdown
            max_drawdown, drawdown_duration = self._calculate_max_drawdown(prices_array)
            
            # Calculate tail risk metrics
            tail_ratio = self._calculate_tail_ratio(returns_array)
            
            # Calculate volatility clustering
            volatility_clustering = self._calculate_volatility_clustering(returns_array)
            
            # Calculate stress test scenarios
            stress_scenarios = self._perform_stress_testing(returns_array)
            
            # Calculate risk-adjusted returns
            risk_adjusted_returns = self._calculate_risk_adjusted_returns(returns_array)
            
            return AdvancedRiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                drawdown_duration=drawdown_duration,
                tail_ratio=tail_ratio,
                volatility_clustering=volatility_clustering,
                stress_scenarios=stress_scenarios,
                risk_adjusted_returns=risk_adjusted_returns
            )
            
        except Exception as e:
            print(f"Error in advanced risk assessment: {e}")
            return self._create_default_risk_metrics()
    
    def _calculate_var(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate Value at Risk at 95% and 99% confidence levels"""
        
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        return var_95, var_99
    
    def _calculate_cvar(self, returns: np.ndarray) -> Tuple[float, float]:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        
        var_95, var_99 = self._calculate_var(returns)
        
        # CVaR is the expected value of returns below VaR threshold
        cvar_95 = np.mean(returns[returns <= var_95]) if np.any(returns <= var_95) else var_95
        cvar_99 = np.mean(returns[returns <= var_99]) if np.any(returns <= var_99) else var_99
        
        return cvar_95, cvar_99
    
    def _calculate_max_drawdown(self, prices: np.ndarray) -> Tuple[float, int]:
        """Calculate maximum drawdown and its duration"""
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(prices)
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Find maximum drawdown
        max_drawdown = np.min(drawdown)
        
        # Calculate drawdown duration
        drawdown_periods = np.where(drawdown < 0)[0]
        if len(drawdown_periods) > 0:
            # Find longest consecutive drawdown period
            diff = np.diff(drawdown_periods)
            breaks = np.where(diff > 1)[0]
            if len(breaks) > 0:
                durations = np.diff(np.concatenate(([0], breaks + 1, [len(drawdown_periods)])))
                max_duration = np.max(durations)
            else:
                max_duration = len(drawdown_periods)
        else:
            max_duration = 0
        
        return max_drawdown, max_duration
    
    def _calculate_tail_ratio(self, returns: np.ndarray) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        
        if p5 != 0:
            tail_ratio = abs(p95 / p5)
        else:
            tail_ratio = float('inf')
        
        return tail_ratio
    
    def _calculate_volatility_clustering(self, returns: np.ndarray) -> float:
        """Calculate volatility clustering measure using autocorrelation of squared returns"""
        
        try:
            squared_returns = returns ** 2
            
            # Calculate autocorrelation at lag 1
            if len(squared_returns) > 1:
                correlation = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _perform_stress_testing(self, returns: np.ndarray) -> Dict[str, float]:
        """Perform stress testing scenarios"""
        
        scenarios = {}
        
        try:
            # Market crash scenario (3 standard deviations)
            std_dev = np.std(returns)
            mean_return = np.mean(returns)
            
            scenarios['market_crash_3std'] = mean_return - 3 * std_dev
            scenarios['market_crash_2std'] = mean_return - 2 * std_dev
            
            # Historical worst case
            scenarios['historical_worst'] = np.min(returns)
            
            # Volatility shock (returns with doubled volatility)
            shocked_returns = np.random.normal(mean_return, 2 * std_dev, len(returns))
            scenarios['volatility_shock_var95'] = np.percentile(shocked_returns, 5)
            
        except Exception as e:
            print(f"Error in stress testing: {e}")
            scenarios = {
                'market_crash_3std': 0.0,
                'market_crash_2std': 0.0,
                'historical_worst': 0.0,
                'volatility_shock_var95': 0.0
            }
        
        return scenarios
    
    def _calculate_risk_adjusted_returns(self, returns: np.ndarray) -> Dict[str, float]:
        """Calculate various risk-adjusted return metrics"""
        
        metrics = {}
        
        try:
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Sharpe ratio (assuming risk-free rate = 0)
            metrics['sharpe_ratio'] = mean_return / std_return if std_return != 0 else 0.0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else std_return
            metrics['sortino_ratio'] = mean_return / downside_std if downside_std != 0 else 0.0
            
            # Calmar ratio (return / max drawdown)
            # We'll use a simplified version here
            negative_returns = returns[returns < 0]
            max_loss = np.min(negative_returns) if len(negative_returns) > 0 else -0.01
            metrics['calmar_ratio'] = mean_return / abs(max_loss) if max_loss != 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating risk-adjusted returns: {e}")
            metrics = {
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0
            }
        
        return metrics
    
    def _create_default_risk_metrics(self) -> AdvancedRiskMetrics:
        """Create default risk metrics for error cases"""
        
        return AdvancedRiskMetrics(
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            max_drawdown=0.0,
            drawdown_duration=0,
            tail_ratio=1.0,
            volatility_clustering=0.0,
            stress_scenarios={},
            risk_adjusted_returns={}
        )
    
    def _perform_ml_insights(self, returns: List[float], prices: List[float]) -> MachineLearningInsights:
        """Generate machine learning insights"""
        
        try:
            returns_array = np.array(returns)
            prices_array = np.array(prices)
            
            # Detect market regimes
            regime_analysis = self._detect_market_regimes(returns_array)
            
            # Perform anomaly detection
            anomalies = self._detect_anomalies(returns_array)
            
            # Cluster analysis
            clustering_results = self._perform_clustering_analysis(returns_array)
            
            # Pattern recognition
            patterns = self._recognize_patterns(prices_array)
            
            # Feature importance from ensemble models
            feature_importance = self._analyze_feature_importance(returns_array, prices_array)
            
            return MachineLearningInsights(
                regime_analysis=regime_analysis,
                anomaly_detection=anomalies,
                clustering_results=clustering_results,
                pattern_recognition=patterns,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            print(f"Error in ML insights: {e}")
            return self._create_default_ml_insights()
    
    def _detect_market_regimes(self, returns: np.ndarray) -> Dict[str, Any]:
        """Detect market regimes using Hidden Markov Models or clustering"""
        
        try:
            if self.has_hmmlearn and len(returns) > 20:
                # Use HMM for regime detection
                from hmmlearn import hmm
                
                # Prepare data (returns and volatility)
                volatility = np.abs(returns)
                X = np.column_stack([returns, volatility])
                
                # Fit HMM with 3 states (bull, bear, sideways)
                model = hmm.GaussianHMM(n_components=3, covariance_type="full", random_state=42)
                model.fit(X)
                
                # Predict states
                states = model.predict(X)
                
                # Analyze regimes
                regime_stats = {}
                for state in range(3):
                    state_mask = states == state
                    if np.any(state_mask):
                        regime_stats[f'regime_{state}'] = {
                            'mean_return': np.mean(returns[state_mask]),
                            'volatility': np.std(returns[state_mask]),
                            'frequency': np.mean(state_mask),
                            'periods': np.sum(state_mask)
                        }
                
                return {
                    'method': 'HMM',
                    'n_regimes': 3,
                    'current_regime': int(states[-1]),
                    'regime_probabilities': model.predict_proba(X)[-1].tolist(),
                    'regime_statistics': regime_stats
                }
            else:
                # Fallback to simple volatility-based regime detection
                volatility = np.abs(returns)
                vol_threshold_high = np.percentile(volatility, 75)
                vol_threshold_low = np.percentile(volatility, 25)
                
                regimes = np.where(volatility > vol_threshold_high, 2,  # High volatility
                                 np.where(volatility < vol_threshold_low, 0, 1))  # Low/Medium volatility
                
                regime_stats = {}
                for regime in range(3):
                    regime_mask = regimes == regime
                    if np.any(regime_mask):
                        regime_stats[f'regime_{regime}'] = {
                            'mean_return': np.mean(returns[regime_mask]),
                            'volatility': np.std(returns[regime_mask]),
                            'frequency': np.mean(regime_mask),
                            'periods': np.sum(regime_mask)
                        }
                
                return {
                    'method': 'Volatility-based',
                    'n_regimes': 3,
                    'current_regime': int(regimes[-1]),
                    'regime_statistics': regime_stats
                }
                
        except Exception as e:
            print(f"Error in regime detection: {e}")
            return {'method': 'None', 'error': str(e)}
    
    def _detect_anomalies(self, returns: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies in returns using statistical methods"""
        
        try:
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(returns))
            z_threshold = 3.0
            z_anomalies = np.where(z_scores > z_threshold)[0]
            
            # IQR based anomaly detection
            q1, q3 = np.percentile(returns, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_anomalies = np.where((returns < lower_bound) | (returns > upper_bound))[0]
            
            # Isolation Forest (if available)
            try:
                from sklearn.ensemble import IsolationForest
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = iso_forest.fit_predict(returns.reshape(-1, 1))
                iso_anomalies = np.where(anomaly_labels == -1)[0]
            except:
                iso_anomalies = np.array([])
            
            return {
                'z_score_anomalies': {
                    'indices': z_anomalies.tolist(),
                    'count': len(z_anomalies),
                    'threshold': z_threshold
                },
                'iqr_anomalies': {
                    'indices': iqr_anomalies.tolist(),
                    'count': len(iqr_anomalies),
                    'bounds': [lower_bound, upper_bound]
                },
                'isolation_forest_anomalies': {
                    'indices': iso_anomalies.tolist(),
                    'count': len(iso_anomalies)
                }
            }
            
        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            return {'error': str(e)}
    
    def _perform_clustering_analysis(self, returns: np.ndarray) -> Dict[str, Any]:
        """Perform clustering analysis on returns"""
        
        try:
            # Prepare features for clustering
            features = []
            window_size = min(10, len(returns) // 4)
            
            for i in range(window_size, len(returns)):
                window_returns = returns[i-window_size:i]
                features.append([
                    np.mean(window_returns),
                    np.std(window_returns),
                    np.min(window_returns),
                    np.max(window_returns)
                ])
            
            if len(features) < 3:
                return {'error': 'Insufficient data for clustering'}
            
            features_array = np.array(features)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            # K-means clustering
            n_clusters = min(3, len(features) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # Analyze clusters
            cluster_stats = {}
            for cluster in range(n_clusters):
                cluster_mask = cluster_labels == cluster
                if np.any(cluster_mask):
                    cluster_features = features_array[cluster_mask]
                    cluster_stats[f'cluster_{cluster}'] = {
                        'size': np.sum(cluster_mask),
                        'mean_return': np.mean(cluster_features[:, 0]),
                        'mean_volatility': np.mean(cluster_features[:, 1]),
                        'frequency': np.mean(cluster_mask)
                    }
            
            return {
                'method': 'K-means',
                'n_clusters': n_clusters,
                'cluster_labels': cluster_labels.tolist(),
                'cluster_statistics': cluster_stats,
                'silhouette_score': float(silhouette_score(features_scaled, cluster_labels)) if len(set(cluster_labels)) > 1 else 0.0
            }
            
        except Exception as e:
            print(f"Error in clustering analysis: {e}")
            return {'error': str(e)}
    
    def _recognize_patterns(self, prices: np.ndarray) -> Dict[str, Any]:
        """Recognize common price patterns"""
        
        try:
            patterns = {}
            
            # Trend analysis
            if len(prices) > 10:
                # Linear trend
                x = np.arange(len(prices))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, prices)
                
                patterns['trend'] = {
                    'slope': slope,
                    'r_squared': r_value ** 2,
                    'p_value': p_value,
                    'direction': 'upward' if slope > 0 else 'downward',
                    'strength': abs(r_value)
                }
            
            # Support and resistance levels
            if len(prices) > 20:
                # Find local minima and maxima
                from scipy.signal import argrelextrema
                
                local_minima = argrelextrema(prices, np.less, order=5)[0]
                local_maxima = argrelextrema(prices, np.greater, order=5)[0]
                
                if len(local_minima) > 0:
                    support_levels = prices[local_minima]
                    patterns['support_levels'] = {
                        'levels': support_levels.tolist(),
                        'current_support': float(np.max(support_levels[support_levels <= prices[-1]]) if np.any(support_levels <= prices[-1]) else np.min(support_levels))
                    }
                
                if len(local_maxima) > 0:
                    resistance_levels = prices[local_maxima]
                    patterns['resistance_levels'] = {
                        'levels': resistance_levels.tolist(),
                        'current_resistance': float(np.min(resistance_levels[resistance_levels >= prices[-1]]) if np.any(resistance_levels >= prices[-1]) else np.max(resistance_levels))
                    }
            
            # Volatility patterns
            if len(prices) > 5:
                returns = np.diff(prices) / prices[:-1]
                volatility = np.abs(returns)
                
                patterns['volatility_pattern'] = {
                    'current_volatility': float(np.std(returns[-10:]) if len(returns) >= 10 else np.std(returns)),
                    'average_volatility': float(np.mean(volatility)),
                    'volatility_trend': 'increasing' if len(volatility) > 1 and volatility[-1] > np.mean(volatility[:-1]) else 'decreasing'
                }
            
            return patterns
            
        except Exception as e:
            print(f"Error in pattern recognition: {e}")
            return {'error': str(e)}
    
    def _analyze_feature_importance(self, returns: np.ndarray, prices: np.ndarray) -> Dict[str, float]:
        """Analyze feature importance for predicting returns"""
        
        try:
            # Create features
            features = []
            targets = []
            
            for i in range(10, len(returns)):
                # Features: lagged returns, moving averages, volatility
                feature_vector = [
                    returns[i-1], returns[i-2], returns[i-3],  # Lagged returns
                    np.mean(returns[i-5:i]),  # 5-period moving average
                    np.mean(returns[i-10:i]),  # 10-period moving average
                    np.std(returns[i-5:i]),   # 5-period volatility
                    np.std(returns[i-10:i]),  # 10-period volatility
                ]
                features.append(feature_vector)
                targets.append(returns[i])
            
            if len(features) < 10:
                return {'error': 'Insufficient data for feature importance analysis'}
            
            features_array = np.array(features)
            targets_array = np.array(targets)
            
            # Use Random Forest to determine feature importance
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features_array, targets_array)
            
            feature_names = [
                'lag_1_return', 'lag_2_return', 'lag_3_return',
                'ma_5', 'ma_10', 'vol_5', 'vol_10'
            ]
            
            importance_dict = dict(zip(feature_names, rf.feature_importances_))
            
            return importance_dict
            
        except Exception as e:
            print(f"Error in feature importance analysis: {e}")
            return {'error': str(e)}
    
    def _create_default_ml_insights(self) -> MachineLearningInsights:
        """Create default ML insights for error cases"""
        
        return MachineLearningInsights(
            regime_analysis={},
            anomaly_detection={},
            clustering_results={},
            pattern_recognition={},
            feature_importance={}
        )
    
    def _calculate_signal_returns(self, market_returns: List[float], 
                                signals: List[str]) -> List[float]:
        """Calculate returns based on trading signals"""
        
        strategy_returns = []
        
        for i, (market_return, signal) in enumerate(zip(market_returns, signals)):
            if signal in ["BUY", "STRONG_BUY"]:
                # Long position - gain from positive returns
                strategy_returns.append(market_return)
            elif signal in ["SELL", "STRONG_SELL"]:
                # Short position - gain from negative returns
                strategy_returns.append(-market_return)
            else:
                # Hold - no return
                strategy_returns.append(0.0)
        
        return strategy_returns
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate annualized Sharpe ratio"""
        
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily data)
        return (mean_return / std_return) * np.sqrt(252)
    
    def _calculate_max_drawdown_from_returns(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod([1 + r for r in returns])
        return self._calculate_max_drawdown(cumulative.tolist())
    
    def _calculate_max_drawdown(self, cumulative: List[float]) -> float:
        """Calculate maximum drawdown from cumulative values"""
        
        if len(cumulative) < 2:
            return 0.0
        
        peak = cumulative[0]
        max_dd = 0.0
        
        for value in cumulative[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak if peak != 0 else 0
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_win_rate(self, returns: List[float]) -> float:
        """Calculate win rate"""
        
        if len(returns) == 0:
            return 0.0
        
        winning_periods = sum(1 for r in returns if r > 0)
        return winning_periods / len(returns)
    
    def _calculate_risk_metrics(self, futures_data: FuturesData, 
                              combined_signals: List[str]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        metrics = {}
        
        # Basic market metrics
        returns = futures_data.returns
        if returns:
            metrics["market_volatility"] = np.std(returns)
            metrics["market_skewness"] = stats.skew(returns)
            metrics["market_kurtosis"] = stats.kurtosis(returns)
        
        # Strategy-specific metrics
        strategy_returns = self._calculate_signal_returns(returns, combined_signals)
        if strategy_returns:
            metrics["strategy_volatility"] = np.std(strategy_returns)
            metrics["strategy_skewness"] = stats.skew(strategy_returns)
            metrics["strategy_kurtosis"] = stats.kurtosis(strategy_returns)
            
            # Value at Risk (5%)
            metrics["var_5pct"] = np.percentile(strategy_returns, 5)
            
            # Expected Shortfall (5%)
            var_threshold = metrics["var_5pct"]
            tail_returns = [r for r in strategy_returns if r <= var_threshold]
            metrics["expected_shortfall_5pct"] = np.mean(tail_returns) if tail_returns else 0
        
        # Signal consistency
        signal_changes = sum(1 for i in range(1, len(combined_signals)) 
                           if combined_signals[i] != combined_signals[i-1])
        metrics["signal_stability"] = 1 - (signal_changes / max(len(combined_signals), 1))
        
        return metrics
    
    def _generate_insights(self, momentum_results: MomentumResult,
                         mean_reversion_results: MeanReversionResult,
                         rl_results: Dict[str, RLAgentResult],
                         strategy_performance: Dict[str, Dict[str, float]]) -> List[str]:
        """Generate analytical insights"""
        
        insights = []
        
        # Momentum insights
        avg_momentum = np.mean(momentum_results.momentum_scores)
        if avg_momentum > 0.2:
            insights.append(f"Strong positive momentum detected (avg: {avg_momentum:.2f})")
        elif avg_momentum < -0.2:
            insights.append(f"Strong negative momentum detected (avg: {avg_momentum:.2f})")
        
        # Mean reversion insights
        if mean_reversion_results.adf_pvalue < 0.05:
            insights.append(f"Price series is stationary (ADF p-value: {mean_reversion_results.adf_pvalue:.3f})")
            insights.append(f"Mean reversion half-life: {mean_reversion_results.half_life:.1f} periods")
        
        # Strategy performance insights
        best_strategy = max(strategy_performance.keys(), 
                          key=lambda k: strategy_performance[k]["sharpe_ratio"])
        best_sharpe = strategy_performance[best_strategy]["sharpe_ratio"]
        insights.append(f"Best performing strategy: {best_strategy} (Sharpe: {best_sharpe:.2f})")
        
        # RL insights
        if rl_results:
            best_rl = max(rl_results.keys(), key=lambda k: rl_results[k].sharpe_ratio)
            insights.append(f"Best RL agent: {best_rl} (Win rate: {rl_results[best_rl].win_rate:.1%})")
        
        # Divergence insights
        divergence_count = sum(momentum_results.momentum_divergence)
        if divergence_count > len(momentum_results.momentum_divergence) * 0.1:
            insights.append(f"Frequent momentum divergences detected ({divergence_count} instances)")
        
        return insights
    
    def _generate_recommendations(self, momentum_results: MomentumResult,
                                mean_reversion_results: MeanReversionResult,
                                rl_results: Dict[str, RLAgentResult],
                                risk_metrics: Dict[str, float]) -> List[str]:
        """Generate trading recommendations"""
        
        recommendations = []
        
        # Strategy selection recommendations
        if mean_reversion_results.adf_pvalue < 0.05 and mean_reversion_results.half_life < 20:
            recommendations.append("Market shows strong mean reversion - favor contrarian strategies")
        else:
            recommendations.append("Market shows momentum characteristics - favor trend-following strategies")
        
        # Risk management recommendations
        strategy_vol = risk_metrics.get("strategy_volatility", 0)
        if strategy_vol > 0.03:
            recommendations.append(f"High strategy volatility ({strategy_vol:.1%}) - reduce position sizes")
        
        var_5pct = risk_metrics.get("var_5pct", 0)
        if var_5pct < -0.05:
            recommendations.append(f"High tail risk (5% VaR: {var_5pct:.1%}) - implement strict stop losses")
        
        # Signal stability recommendations
        signal_stability = risk_metrics.get("signal_stability", 1)
        if signal_stability < 0.7:
            recommendations.append("Low signal stability - consider longer-term indicators or signal smoothing")
        
        # RL-specific recommendations
        if rl_results:
            best_rl_agent = max(rl_results.keys(), key=lambda k: rl_results[k].sharpe_ratio)
            recommendations.append(f"Consider using {best_rl_agent} agent for automated trading")
        
        # Technical recommendations
        current_rsi = momentum_results.rsi_values[-1] if momentum_results.rsi_values else 50
        if current_rsi > 70:
            recommendations.append("RSI indicates overbought conditions - consider taking profits")
        elif current_rsi < 30:
            recommendations.append("RSI indicates oversold conditions - consider buying opportunities")
        
        # General recommendations
        recommendations.append("Combine momentum and mean reversion signals for robust trading decisions")
        recommendations.append("Regularly retrain RL models with new market data")
        recommendations.append("Monitor regime changes that may affect strategy effectiveness")
        
        return recommendations
    
    def _calculate_model_confidence(self, momentum_results: MomentumResult,
                                  mean_reversion_results: MeanReversionResult,
                                  rl_results: Dict[str, RLAgentResult]) -> Dict[str, float]:
        """Calculate confidence scores for different models"""
        
        confidence = {}
        
        # Momentum model confidence
        momentum_strength_avg = np.mean(momentum_results.momentum_strength)
        confidence["momentum"] = min(momentum_strength_avg * 2, 1.0)  # Scale to 0-1
        
        # Mean reversion model confidence
        if mean_reversion_results.adf_pvalue < 0.05:
            stationarity_confidence = 1 - mean_reversion_results.adf_pvalue
        else:
            stationarity_confidence = 0.5
        
        half_life_confidence = 1 / (1 + mean_reversion_results.half_life / 20)
        confidence["mean_reversion"] = (stationarity_confidence + half_life_confidence) / 2
        
        # RL model confidence
        for agent_type, rl_result in rl_results.items():
            # Base confidence on win rate and Sharpe ratio
            win_rate_score = rl_result.win_rate
            sharpe_score = max(0, min(rl_result.sharpe_ratio / 2, 1))  # Normalize Sharpe
            confidence[f"rl_{agent_type.lower()}"] = (win_rate_score + sharpe_score) / 2
        
        # Overall confidence
        confidence["overall"] = np.mean(list(confidence.values()))
        
        return confidence
    
    def _create_default_result(self, futures_data: FuturesData) -> FuturesMomentumMeanReversionResult:
        """Create default result for error cases"""
        
        n = len(futures_data.close)
        
        # Create default momentum results
        default_momentum = MomentumResult(
            momentum_scores=[0.0] * n,
            momentum_signals=["HOLD"] * n,
            momentum_strength=[0.5] * n,
            trend_direction=["SIDEWAYS"] * n,
            momentum_divergence=[False] * n,
            rsi_values=[50.0] * n,
            macd_values=[0.0] * n,
            macd_signal=[0.0] * n,
            stochastic_k=[50.0] * n,
            stochastic_d=[50.0] * n,
            williams_r=[-50.0] * n
        )
        
        # Create default mean reversion results
        default_mean_reversion = MeanReversionResult(
            mean_reversion_scores=[0.0] * n,
            mean_reversion_signals=["HOLD"] * n,
            z_scores=[0.0] * n,
            bollinger_upper=futures_data.close.copy(),
            bollinger_middle=futures_data.close.copy(),
            bollinger_lower=futures_data.close.copy(),
            adf_statistic=0.0,
            adf_pvalue=0.5,
            half_life=20.0,
            reversion_probability=[0.5] * n
        )
        
        # Create default RL results
        default_rl = {
            "PPO": RLAgentResult(
                agent_type="PPO",
                actions=[0] * n,
                rewards=[0.0] * n,
                cumulative_returns=[0.0] * n,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_return=0.0
            )
        }
        
        return FuturesMomentumMeanReversionResult(
            momentum_results=default_momentum,
            mean_reversion_results=default_mean_reversion,
            rl_results=default_rl,
            combined_signals=["HOLD"] * n,
            strategy_performance={"combined": {"total_return": 0.0, "sharpe_ratio": 0.0, 
                                             "max_drawdown": 0.0, "win_rate": 0.0}},
            risk_metrics={"market_volatility": 0.02, "signal_stability": 1.0},
            insights=["Analysis completed with default values due to errors"],
            recommendations=["Review data quality and model parameters"],
            model_confidence={"overall": 0.5}
        )
    
    def plot_results(self, futures_data: FuturesData, 
                    results: FuturesMomentumMeanReversionResult):
        """Plot comprehensive analysis results"""
        
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available. Cannot generate plots.")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        timestamps = futures_data.timestamps
        
        # Plot 1: Price and Bollinger Bands
        ax1 = axes[0, 0]
        ax1.plot(timestamps, futures_data.close, label='Price', linewidth=2)
        ax1.plot(timestamps, results.mean_reversion_results.bollinger_upper, 
                label='BB Upper', linestyle='--', alpha=0.7)
        ax1.plot(timestamps, results.mean_reversion_results.bollinger_lower, 
                label='BB Lower', linestyle='--', alpha=0.7)
        ax1.fill_between(timestamps, 
                        results.mean_reversion_results.bollinger_upper,
                        results.mean_reversion_results.bollinger_lower,
                        alpha=0.1)
        
        ax1.set_title('Price and Bollinger Bands', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: RSI
        ax2 = axes[0, 1]
        ax2.plot(timestamps, results.momentum_results.rsi_values, label='RSI', color='purple')
        ax2.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought')
        ax2.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold')
        ax2.axhline(y=50, color='gray', linestyle='-', alpha=0.5)
        
        ax2.set_title('RSI Momentum Indicator', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Z-Scores
        ax3 = axes[1, 0]
        ax3.plot(timestamps, results.mean_reversion_results.z_scores, 
                label='Z-Score', color='orange', linewidth=2)
        ax3.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Overbought (+2σ)')
        ax3.axhline(y=-2, color='green', linestyle='--', alpha=0.7, label='Oversold (-2σ)')
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax3.set_title('Z-Score Mean Reversion', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Z-Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Combined Signals
        ax4 = axes[1, 1]
        signal_numeric = []
        for signal in results.combined_signals:
            if signal == 'STRONG_BUY':
                signal_numeric.append(2)
            elif signal == 'BUY':
                signal_numeric.append(1)
            elif signal == 'HOLD':
                signal_numeric.append(0)
            elif signal == 'SELL':
                signal_numeric.append(-1)
            elif signal == 'STRONG_SELL':
                signal_numeric.append(-2)
            else:
                signal_numeric.append(0)
        
        ax4.plot(timestamps, signal_numeric, marker='o', markersize=3, linewidth=1)
        ax4.set_ylim(-2.5, 2.5)
        ax4.set_yticks([-2, -1, 0, 1, 2])
        ax4.set_yticklabels(['Strong Sell', 'Sell', 'Hold', 'Buy', 'Strong Buy'])
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax4.set_title('Combined Trading Signals', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Signal')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Strategy Performance
        ax5 = axes[2, 0]
        strategies = list(results.strategy_performance.keys())
        sharpe_ratios = [results.strategy_performance[s]['sharpe_ratio'] for s in strategies]
        
        bars = ax5.bar(strategies, sharpe_ratios, 
                      color=['blue', 'green', 'red', 'purple', 'orange'][:len(strategies)])
        ax5.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        for bar, value in zip(bars, sharpe_ratios):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.01 if height >= 0 else height - 0.05,
                    f'{value:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        ax5.set_title('Strategy Performance (Sharpe Ratio)', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Strategy')
        ax5.set_ylabel('Sharpe Ratio')
        ax5.tick_params(axis='x', rotation=45)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Model Confidence
        ax6 = axes[2, 1]
        models = list(results.model_confidence.keys())
        confidences = list(results.model_confidence.values())
        
        bars = ax6.bar(models, confidences, color='skyblue')
        ax6.set_ylim(0, 1)
        
        for bar, value in zip(bars, confidences):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
        
        ax6.set_title('Model Confidence Scores', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Model')
        ax6.set_ylabel('Confidence')
        ax6.tick_params(axis='x', rotation=45)
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_report(self, futures_data: FuturesData, 
                       results: FuturesMomentumMeanReversionResult) -> str:
        """Generate comprehensive analysis report"""
        
        report = f"""
# FUTURES MOMENTUM & MEAN REVERSION ANALYSIS REPORT
## Contract: {futures_data.contract_symbol} ({futures_data.underlying_asset})
## Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### EXECUTIVE SUMMARY
This report presents a comprehensive analysis of momentum and mean reversion characteristics 
for the {futures_data.contract_symbol} futures contract, incorporating advanced machine learning 
and reinforcement learning techniques.

### CURRENT TRADING SIGNAL
**Combined Signal:** {results.combined_signals[-1]}

### MOMENTUM ANALYSIS
**Current Momentum Score:** {results.momentum_results.momentum_scores[-1]:.3f}
**Current RSI:** {results.momentum_results.rsi_values[-1]:.1f}
**Current MACD:** {results.momentum_results.macd_values[-1]:.4f}
**Trend Direction:** {results.momentum_results.trend_direction[-1]}
**Momentum Divergences Detected:** {sum(results.momentum_results.momentum_divergence)}

### MEAN REVERSION ANALYSIS
**Stationarity (ADF p-value):** {results.mean_reversion_results.adf_pvalue:.4f}
**Mean Reversion Half-Life:** {results.mean_reversion_results.half_life:.1f} periods
**Current Z-Score:** {results.mean_reversion_results.z_scores[-1]:.3f}
**Current Mean Reversion Score:** {results.mean_reversion_results.mean_reversion_scores[-1]:.3f}
**Reversion Probability:** {results.mean_reversion_results.reversion_probability[-1]:.1%}

### REINFORCEMENT LEARNING ANALYSIS
"""
        
        for agent_type, rl_result in results.rl_results.items():
            report += f"""
**{agent_type} Agent Performance:**
- Total Return: {rl_result.cumulative_returns[-1]:.2%}
- Sharpe Ratio: {rl_result.sharpe_ratio:.2f}
- Maximum Drawdown: {rl_result.max_drawdown:.2%}
- Win Rate: {rl_result.win_rate:.1%}
- Total Trades: {rl_result.total_trades}
"""
        
        report += f"""

### STRATEGY PERFORMANCE COMPARISON
"""
        
        for strategy, performance in results.strategy_performance.items():
            report += f"""
**{strategy.replace('_', ' ').title()} Strategy:**
- Total Return: {performance['total_return']:.2%}
- Sharpe Ratio: {performance['sharpe_ratio']:.2f}
- Maximum Drawdown: {performance['max_drawdown']:.2%}
- Win Rate: {performance['win_rate']:.1%}
"""
        
        report += f"""

### RISK METRICS
**Market Volatility:** {results.risk_metrics.get('market_volatility', 0):.2%}
**Strategy Volatility:** {results.risk_metrics.get('strategy_volatility', 0):.2%}
**Value at Risk (5%):** {results.risk_metrics.get('var_5pct', 0):.2%}
**Expected Shortfall (5%):** {results.risk_metrics.get('expected_shortfall_5pct', 0):.2%}
**Signal Stability:** {results.risk_metrics.get('signal_stability', 0):.1%}

### KEY INSIGHTS
"""
        
        for i, insight in enumerate(results.insights, 1):
            report += f"{i}. {insight}\n"
        
        report += f"""

### RECOMMENDATIONS
"""
        
        for i, recommendation in enumerate(results.recommendations, 1):
            report += f"{i}. {recommendation}\n"
        
        report += f"""

### MODEL CONFIDENCE SCORES
"""
        
        for model, confidence in results.model_confidence.items():
            report += f"**{model.replace('_', ' ').title()}:** {confidence:.1%}\n"
        
        report += f"""

### METHODOLOGY
This analysis employs multiple complementary approaches:

1. **Momentum Analysis:** RSI, MACD, Stochastic Oscillator, Williams %R
2. **Mean Reversion Analysis:** Bollinger Bands, Z-scores, ADF stationarity test
3. **Reinforcement Learning:** PPO, SAC, and DDPG agents trained on historical data
4. **Signal Combination:** Voting mechanism across all strategies
5. **Risk Assessment:** VaR, Expected Shortfall, volatility metrics

**Disclaimer:** This analysis is for informational purposes only and should not be 
considered as financial advice. Past performance does not guarantee future results.
"""
        
        return report

# Example usage
if __name__ == "__main__":
    # Create sample futures data
    np.random.seed(42)
    n_periods = 252  # One year of daily data
    
    # Generate realistic futures price data
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.02, n_periods)  # Daily returns
    prices = [base_price]
    
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate OHLC data
    high_prices = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    low_prices = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    open_prices = [prices[0]] + prices[:-1]  # Previous close as next open
    
    # Generate volume and open interest
    base_volume = 10000
    volumes = [base_volume * (1 + np.random.normal(0, 0.3)) for _ in range(len(prices))]
    volumes = [max(1000, v) for v in volumes]  # Ensure positive volume
    
    open_interests = [50000 * (1 + np.random.normal(0, 0.1)) for _ in range(len(prices))]
    open_interests = [max(10000, oi) for oi in open_interests]  # Ensure positive OI
    
    # Create timestamps
    start_date = datetime.now() - timedelta(days=n_periods)
    timestamps = [start_date + timedelta(days=i) for i in range(len(prices))]
    
    # Create FuturesData object
    futures_data = FuturesData(
        prices=prices,
        returns=returns.tolist(),
        volume=volumes,
        open_interest=open_interests,
        timestamps=timestamps,
        high=high_prices,
        low=low_prices,
        open=open_prices,
        close=prices,
        contract_symbol="CL_2024_03",
        underlying_asset="Crude Oil"
    )
    
    # Initialize analyzer
    analyzer = FuturesMomentumMeanReversionAnalyzer()
    
    # Perform analysis
    print("Performing comprehensive futures momentum and mean reversion analysis...")
    results = analyzer.analyze(futures_data, rl_train_timesteps=1000)
    
    # Print summary
    print("\n" + "="*80)
    print("FUTURES MOMENTUM & MEAN REVERSION ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\nContract: {futures_data.contract_symbol}")
    print(f"Underlying Asset: {futures_data.underlying_asset}")
    print(f"Analysis Period: {len(futures_data.prices)} periods")
    
    print(f"\nCurrent Signals:")
    print(f"  Combined: {results.combined_signals[-1]}")
    print(f"  Momentum: {results.momentum_results.momentum_signals[-1]}")
    print(f"  Mean Reversion: {results.mean_reversion_results.mean_reversion_signals[-1]}")
    
    print(f"\nBest Strategy: {max(results.strategy_performance.keys(), key=lambda k: results.strategy_performance[k]['sharpe_ratio'])}")
    print(f"Overall Confidence: {results.model_confidence['overall']:.1%}")
    
    # Generate and save report
    report = analyzer.generate_report(futures_data, results)
    print(f"\nFull report generated ({len(report)} characters)")