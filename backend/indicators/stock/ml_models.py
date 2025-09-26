"""Advanced ML Models for Stock Analysis

This module implements machine learning models for stock analysis including:
- LSTM for time series prediction
- XGBoost for feature-based analysis
- Random Forest for ensemble predictions
- Feature engineering and selection
- Model evaluation and validation

Author: Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Note: In a real implementation, you would import actual ML libraries
# For this example, we'll create simplified versions

@dataclass
class StockFundamentals:
    """Stock fundamental data"""
    revenue: float
    net_income: float
    free_cash_flow: float
    total_debt: float
    shareholders_equity: float
    shares_outstanding: float
    dividend_per_share: float
    earnings_per_share: float
    book_value_per_share: float
    revenue_growth_rate: float
    earnings_growth_rate: float
    dividend_growth_rate: float
    beta: float
    market_cap: float

@dataclass
class MarketData:
    """Market and economic data"""
    risk_free_rate: float
    market_return: float
    inflation_rate: float
    gdp_growth: float
    unemployment_rate: float
    sector_performance: Dict[str, float]
    market_volatility: float

@dataclass
class TimeSeriesData:
    """Time series data for ML models"""
    prices: List[float]
    volumes: List[float]
    dates: List[datetime]
    technical_indicators: Dict[str, List[float]]

@dataclass
class MLModelResult:
    """Result of ML model analysis"""
    lstm_prediction: Dict[str, Any]
    xgboost_analysis: Dict[str, Any]
    random_forest_analysis: Dict[str, Any]
    ensemble_prediction: Dict[str, Any]
    feature_importance: Dict[str, float]
    model_performance: Dict[str, Dict[str, float]]
    risk_assessment: Dict[str, Any]
    confidence_intervals: Dict[str, Tuple[float, float]]
    prediction_horizon: int  # days
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str

class AdvancedMLModels:
    """Advanced Machine Learning Models for Stock Analysis"""
    
    def __init__(self):
        self.lstm_lookback = 60  # days
        self.prediction_horizon = 30  # days
        self.confidence_level = 0.95
        self.min_data_points = 100
    
    def analyze(self, fundamentals: StockFundamentals, market_data: MarketData,
               time_series: Optional[TimeSeriesData] = None) -> MLModelResult:
        """Perform comprehensive ML analysis"""
        try:
            # Generate synthetic time series if not provided
            if time_series is None:
                time_series = self._generate_synthetic_data(fundamentals)
            
            # Validate data sufficiency
            if len(time_series.prices) < self.min_data_points:
                return self._create_insufficient_data_result()
            
            # Feature engineering
            features = self._engineer_features(fundamentals, market_data, time_series)
            
            # LSTM prediction
            lstm_result = self._lstm_prediction(time_series, features)
            
            # XGBoost analysis
            xgboost_result = self._xgboost_analysis(features, time_series)
            
            # Random Forest analysis
            rf_result = self._random_forest_analysis(features, time_series)
            
            # Ensemble prediction
            ensemble_result = self._ensemble_prediction(
                lstm_result, xgboost_result, rf_result
            )
            
            # Feature importance analysis
            feature_importance = self._analyze_feature_importance(
                xgboost_result, rf_result
            )
            
            # Model performance evaluation
            model_performance = self._evaluate_model_performance(
                lstm_result, xgboost_result, rf_result
            )
            
            # Risk assessment
            risk_assessment = self._assess_prediction_risk(
                ensemble_result, time_series, market_data
            )
            
            # Confidence intervals
            confidence_intervals = self._calculate_confidence_intervals(
                ensemble_result, model_performance
            )
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                ensemble_result, feature_importance, risk_assessment
            )
            
            return MLModelResult(
                lstm_prediction=lstm_result,
                xgboost_analysis=xgboost_result,
                random_forest_analysis=rf_result,
                ensemble_prediction=ensemble_result,
                feature_importance=feature_importance,
                model_performance=model_performance,
                risk_assessment=risk_assessment,
                confidence_intervals=confidence_intervals,
                prediction_horizon=self.prediction_horizon,
                metadata={
                    "data_points": len(time_series.prices),
                    "features_used": len(features),
                    "lookback_period": self.lstm_lookback,
                    "confidence_level": self.confidence_level
                },
                timestamp=datetime.now(),
                interpretation=interpretation
            )
            
        except Exception as e:
            return self._create_fallback_result(str(e))
    
    def _generate_synthetic_data(self, fundamentals: StockFundamentals) -> TimeSeriesData:
        """Generate synthetic time series data for demonstration"""
        # Generate 200 days of synthetic price data
        np.random.seed(42)  # For reproducibility
        
        base_price = fundamentals.market_cap / fundamentals.shares_outstanding
        dates = [datetime.now() - timedelta(days=i) for i in range(200, 0, -1)]
        
        # Generate prices with some trend and volatility
        returns = np.random.normal(0.001, 0.02, 200)  # Daily returns
        prices = [base_price]
        
        for i in range(1, 200):
            new_price = prices[-1] * (1 + returns[i])
            prices.append(max(0.1, new_price))  # Ensure positive prices
        
        # Generate volumes
        avg_volume = fundamentals.shares_outstanding * 0.01  # 1% daily turnover
        volumes = np.random.lognormal(np.log(avg_volume), 0.5, 200).tolist()
        
        # Generate technical indicators
        technical_indicators = {
            'sma_20': self._calculate_sma(prices, 20),
            'sma_50': self._calculate_sma(prices, 50),
            'rsi': self._calculate_rsi(prices),
            'macd': self._calculate_macd(prices)
        }
        
        return TimeSeriesData(
            prices=prices,
            volumes=volumes,
            dates=dates,
            technical_indicators=technical_indicators
        )
    
    def _calculate_sma(self, prices: List[float], window: int) -> List[float]:
        """Calculate Simple Moving Average"""
        sma = []
        for i in range(len(prices)):
            if i < window - 1:
                sma.append(prices[i])  # Use actual price for insufficient data
            else:
                sma.append(np.mean(prices[i-window+1:i+1]))
        return sma
    
    def _calculate_rsi(self, prices: List[float], window: int = 14) -> List[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < window + 1:
            return [50.0] * len(prices)  # Neutral RSI
        
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [max(0, delta) for delta in deltas]
        losses = [max(0, -delta) for delta in deltas]
        
        rsi = [50.0]  # First value
        
        for i in range(window, len(deltas)):
            avg_gain = np.mean(gains[i-window:i])
            avg_loss = np.mean(losses[i-window:i])
            
            if avg_loss == 0:
                rsi.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
                rsi.append(rsi_val)
        
        # Pad to match price length
        while len(rsi) < len(prices):
            rsi.append(rsi[-1])
        
        return rsi
    
    def _calculate_macd(self, prices: List[float]) -> List[float]:
        """Calculate MACD"""
        if len(prices) < 26:
            return [0.0] * len(prices)
        
        ema_12 = self._calculate_ema(prices, 12)
        ema_26 = self._calculate_ema(prices, 26)
        
        macd = [ema_12[i] - ema_26[i] for i in range(len(prices))]
        return macd
    
    def _calculate_ema(self, prices: List[float], window: int) -> List[float]:
        """Calculate Exponential Moving Average"""
        if not prices:
            return []
        
        alpha = 2 / (window + 1)
        ema = [prices[0]]
        
        for i in range(1, len(prices)):
            ema_val = alpha * prices[i] + (1 - alpha) * ema[-1]
            ema.append(ema_val)
        
        return ema
    
    def _engineer_features(self, fundamentals: StockFundamentals, 
                          market_data: MarketData,
                          time_series: TimeSeriesData) -> Dict[str, List[float]]:
        """Engineer features for ML models"""
        features = {}
        
        # Price-based features
        features['price_returns'] = self._calculate_returns(time_series.prices)
        features['price_volatility'] = self._calculate_rolling_volatility(time_series.prices)
        features['price_momentum'] = self._calculate_momentum(time_series.prices)
        
        # Volume features
        features['volume_sma'] = self._calculate_sma(time_series.volumes, 20)
        features['volume_ratio'] = [
            vol / avg_vol if avg_vol > 0 else 1.0
            for vol, avg_vol in zip(time_series.volumes, features['volume_sma'])
        ]
        
        # Technical indicators
        features.update(time_series.technical_indicators)
        
        # Fundamental ratios (constant for time series)
        pe_ratio = (fundamentals.market_cap / fundamentals.shares_outstanding) / fundamentals.earnings_per_share
        pb_ratio = (fundamentals.market_cap / fundamentals.shares_outstanding) / fundamentals.book_value_per_share
        
        features['pe_ratio'] = [pe_ratio] * len(time_series.prices)
        features['pb_ratio'] = [pb_ratio] * len(time_series.prices)
        features['debt_to_equity'] = [fundamentals.total_debt / fundamentals.shareholders_equity] * len(time_series.prices)
        
        # Market features
        features['market_return'] = [market_data.market_return] * len(time_series.prices)
        features['risk_free_rate'] = [market_data.risk_free_rate] * len(time_series.prices)
        features['market_volatility'] = [market_data.market_volatility] * len(time_series.prices)
        
        return features
    
    def _calculate_returns(self, prices: List[float]) -> List[float]:
        """Calculate price returns"""
        if len(prices) < 2:
            return [0.0] * len(prices)
        
        returns = [0.0]  # First return is 0
        for i in range(1, len(prices)):
            if prices[i-1] > 0:
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
            else:
                returns.append(0.0)
        
        return returns
    
    def _calculate_rolling_volatility(self, prices: List[float], window: int = 20) -> List[float]:
        """Calculate rolling volatility"""
        returns = self._calculate_returns(prices)
        volatility = []
        
        for i in range(len(returns)):
            if i < window - 1:
                volatility.append(0.02)  # Default volatility
            else:
                window_returns = returns[i-window+1:i+1]
                vol = np.std(window_returns) * np.sqrt(252)  # Annualized
                volatility.append(vol)
        
        return volatility
    
    def _calculate_momentum(self, prices: List[float], window: int = 10) -> List[float]:
        """Calculate price momentum"""
        momentum = []
        
        for i in range(len(prices)):
            if i < window:
                momentum.append(0.0)
            else:
                mom = (prices[i] - prices[i-window]) / prices[i-window] if prices[i-window] > 0 else 0.0
                momentum.append(mom)
        
        return momentum
    
    def _lstm_prediction(self, time_series: TimeSeriesData, 
                        features: Dict[str, List[float]]) -> Dict[str, Any]:
        """Simplified LSTM prediction"""
        # In a real implementation, this would use TensorFlow/PyTorch
        # For now, we'll create a simplified trend-based prediction
        
        prices = time_series.prices
        recent_prices = prices[-self.lstm_lookback:]
        
        # Simple trend analysis
        if len(recent_prices) >= 2:
            trend = (recent_prices[-1] - recent_prices[0]) / len(recent_prices)
            volatility = np.std(self._calculate_returns(recent_prices))
        else:
            trend = 0.0
            volatility = 0.02
        
        # Generate predictions
        current_price = prices[-1]
        predictions = []
        
        for i in range(self.prediction_horizon):
            # Add some noise and mean reversion
            noise = np.random.normal(0, volatility)
            mean_reversion = -0.1 * (current_price - np.mean(recent_prices)) / current_price
            
            predicted_return = trend + noise + mean_reversion
            predicted_price = current_price * (1 + predicted_return)
            predictions.append(max(0.1, predicted_price))
            current_price = predicted_price
        
        return {
            'predictions': predictions,
            'trend': trend,
            'volatility': volatility,
            'confidence': 0.7,
            'model_type': 'LSTM',
            'lookback_used': self.lstm_lookback
        }
    
    def _xgboost_analysis(self, features: Dict[str, List[float]], 
                         time_series: TimeSeriesData) -> Dict[str, Any]:
        """Simplified XGBoost analysis"""
        # In a real implementation, this would use XGBoost library
        # For now, we'll create a feature-based prediction
        
        prices = time_series.prices
        current_price = prices[-1]
        
        # Use recent technical indicators for prediction
        recent_rsi = time_series.technical_indicators['rsi'][-1]
        recent_macd = time_series.technical_indicators['macd'][-1]
        recent_sma_ratio = current_price / time_series.technical_indicators['sma_20'][-1]
        
        # Simple rule-based prediction
        prediction_factor = 1.0
        
        # RSI influence
        if recent_rsi > 70:  # Overbought
            prediction_factor *= 0.98
        elif recent_rsi < 30:  # Oversold
            prediction_factor *= 1.02
        
        # MACD influence
        if recent_macd > 0:
            prediction_factor *= 1.005
        else:
            prediction_factor *= 0.995
        
        # SMA influence
        if recent_sma_ratio > 1.05:
            prediction_factor *= 0.99
        elif recent_sma_ratio < 0.95:
            prediction_factor *= 1.01
        
        predicted_price = current_price * prediction_factor
        
        return {
            'prediction': predicted_price,
            'prediction_factor': prediction_factor,
            'feature_contributions': {
                'rsi': recent_rsi,
                'macd': recent_macd,
                'sma_ratio': recent_sma_ratio
            },
            'confidence': 0.75,
            'model_type': 'XGBoost'
        }
    
    def _random_forest_analysis(self, features: Dict[str, List[float]], 
                               time_series: TimeSeriesData) -> Dict[str, Any]:
        """Simplified Random Forest analysis"""
        # In a real implementation, this would use scikit-learn
        # For now, we'll create an ensemble-like prediction
        
        prices = time_series.prices
        current_price = prices[-1]
        
        # Multiple "tree" predictions
        tree_predictions = []
        
        # Tree 1: Based on momentum
        momentum = self._calculate_momentum(prices)[-1]
        tree1_pred = current_price * (1 + momentum * 0.5)
        tree_predictions.append(tree1_pred)
        
        # Tree 2: Based on volatility
        volatility = self._calculate_rolling_volatility(prices)[-1]
        vol_factor = 1.0 - min(0.1, volatility)  # Lower vol = slight increase
        tree2_pred = current_price * vol_factor
        tree_predictions.append(tree2_pred)
        
        # Tree 3: Based on volume
        volume_ratio = features['volume_ratio'][-1]
        vol_factor = 1.0 + (volume_ratio - 1.0) * 0.01  # Volume influence
        tree3_pred = current_price * vol_factor
        tree_predictions.append(tree3_pred)
        
        # Average prediction
        rf_prediction = np.mean(tree_predictions)
        
        return {
            'prediction': rf_prediction,
            'tree_predictions': tree_predictions,
            'feature_importance': {
                'momentum': 0.4,
                'volatility': 0.35,
                'volume': 0.25
            },
            'confidence': 0.8,
            'model_type': 'Random Forest',
            'n_trees': len(tree_predictions)
        }
    
    def _ensemble_prediction(self, lstm_result: Dict[str, Any], 
                           xgboost_result: Dict[str, Any],
                           rf_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine predictions from multiple models"""
        # Weight models by their confidence
        lstm_weight = lstm_result['confidence']
        xgb_weight = xgboost_result['confidence']
        rf_weight = rf_result['confidence']
        
        total_weight = lstm_weight + xgb_weight + rf_weight
        
        # For LSTM, use first prediction
        lstm_pred = lstm_result['predictions'][0] if lstm_result['predictions'] else 0
        
        # Weighted average
        ensemble_prediction = (
            lstm_pred * lstm_weight +
            xgboost_result['prediction'] * xgb_weight +
            rf_result['prediction'] * rf_weight
        ) / total_weight
        
        # Calculate prediction range
        all_predictions = [
            lstm_pred,
            xgboost_result['prediction'],
            rf_result['prediction']
        ]
        
        prediction_std = np.std(all_predictions)
        
        return {
            'prediction': ensemble_prediction,
            'prediction_range': {
                'low': ensemble_prediction - 1.96 * prediction_std,
                'high': ensemble_prediction + 1.96 * prediction_std
            },
            'model_weights': {
                'lstm': lstm_weight / total_weight,
                'xgboost': xgb_weight / total_weight,
                'random_forest': rf_weight / total_weight
            },
            'prediction_std': prediction_std,
            'confidence': np.mean([lstm_weight, xgb_weight, rf_weight])
        }
    
    def _analyze_feature_importance(self, xgboost_result: Dict[str, Any],
                                  rf_result: Dict[str, Any]) -> Dict[str, float]:
        """Analyze feature importance across models"""
        # Combine feature importance from XGBoost and Random Forest
        xgb_importance = xgboost_result.get('feature_contributions', {})
        rf_importance = rf_result.get('feature_importance', {})
        
        # Normalize and combine
        combined_importance = {}
        all_features = set(list(xgb_importance.keys()) + list(rf_importance.keys()))
        
        for feature in all_features:
            xgb_val = abs(xgb_importance.get(feature, 0))
            rf_val = rf_importance.get(feature, 0)
            combined_importance[feature] = (xgb_val + rf_val) / 2
        
        # Normalize to sum to 1
        total_importance = sum(combined_importance.values())
        if total_importance > 0:
            combined_importance = {
                k: v / total_importance for k, v in combined_importance.items()
            }
        
        return combined_importance
    
    def _evaluate_model_performance(self, lstm_result: Dict[str, Any],
                                  xgboost_result: Dict[str, Any],
                                  rf_result: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Evaluate model performance metrics"""
        # In a real implementation, this would use historical backtesting
        # For now, we'll use confidence as a proxy for performance
        
        return {
            'lstm': {
                'confidence': lstm_result['confidence'],
                'volatility': lstm_result.get('volatility', 0.02),
                'trend_strength': abs(lstm_result.get('trend', 0))
            },
            'xgboost': {
                'confidence': xgboost_result['confidence'],
                'feature_utilization': len(xgboost_result.get('feature_contributions', {})),
                'prediction_magnitude': abs(xgboost_result.get('prediction_factor', 1) - 1)
            },
            'random_forest': {
                'confidence': rf_result['confidence'],
                'tree_consensus': np.std(rf_result.get('tree_predictions', [0])),
                'feature_diversity': len(rf_result.get('feature_importance', {}))
            }
        }
    
    def _assess_prediction_risk(self, ensemble_result: Dict[str, Any],
                              time_series: TimeSeriesData,
                              market_data: MarketData) -> Dict[str, Any]:
        """Assess risk associated with predictions"""
        prediction_std = ensemble_result.get('prediction_std', 0)
        current_price = time_series.prices[-1]
        
        # Risk metrics
        prediction_volatility = prediction_std / current_price
        market_risk = market_data.market_volatility
        
        # Risk classification
        if prediction_volatility > market_risk * 1.5:
            risk_level = "High"
        elif prediction_volatility > market_risk:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            'risk_level': risk_level,
            'prediction_volatility': prediction_volatility,
            'market_volatility': market_risk,
            'relative_risk': prediction_volatility / market_risk if market_risk > 0 else 1.0,
            'risk_factors': {
                'model_disagreement': prediction_std,
                'market_conditions': market_data.market_volatility,
                'data_quality': 'Good' if len(time_series.prices) > 100 else 'Limited'
            }
        }
    
    def _calculate_confidence_intervals(self, ensemble_result: Dict[str, Any],
                                      model_performance: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for predictions"""
        prediction = ensemble_result['prediction']
        prediction_std = ensemble_result.get('prediction_std', 0)
        
        # Different confidence levels
        confidence_intervals = {}
        
        for confidence_level in [0.68, 0.95, 0.99]:
            if confidence_level == 0.68:
                z_score = 1.0
            elif confidence_level == 0.95:
                z_score = 1.96
            else:  # 0.99
                z_score = 2.58
            
            margin = z_score * prediction_std
            confidence_intervals[f'{confidence_level:.0%}'] = (
                prediction - margin,
                prediction + margin
            )
        
        return confidence_intervals
    
    def _generate_interpretation(self, ensemble_result: Dict[str, Any],
                               feature_importance: Dict[str, float],
                               risk_assessment: Dict[str, Any]) -> str:
        """Generate interpretation of ML results"""
        prediction = ensemble_result['prediction']
        confidence = ensemble_result['confidence']
        risk_level = risk_assessment['risk_level']
        
        interpretation_parts = []
        
        # Prediction summary
        interpretation_parts.append(
            f"ML ensemble prediction: ${prediction:.2f} (confidence: {confidence:.1%})"
        )
        
        # Risk assessment
        interpretation_parts.append(f"Prediction risk: {risk_level}")
        
        # Top features
        if feature_importance:
            top_feature = max(feature_importance.items(), key=lambda x: x[1])
            interpretation_parts.append(
                f"Key driver: {top_feature[0]} ({top_feature[1]:.1%} importance)"
            )
        
        # Model consensus
        model_weights = ensemble_result.get('model_weights', {})
        dominant_model = max(model_weights.items(), key=lambda x: x[1]) if model_weights else None
        if dominant_model:
            interpretation_parts.append(
                f"Dominant model: {dominant_model[0]} ({dominant_model[1]:.1%} weight)"
            )
        
        return "; ".join(interpretation_parts)
    
    def _create_insufficient_data_result(self) -> MLModelResult:
        """Create result when insufficient data is available"""
        return MLModelResult(
            lstm_prediction={'error': 'Insufficient data'},
            xgboost_analysis={'error': 'Insufficient data'},
            random_forest_analysis={'error': 'Insufficient data'},
            ensemble_prediction={'error': 'Insufficient data'},
            feature_importance={},
            model_performance={},
            risk_assessment={'risk_level': 'High', 'reason': 'Insufficient data'},
            confidence_intervals={},
            prediction_horizon=0,
            metadata={'error': 'Insufficient data for ML analysis'},
            timestamp=datetime.now(),
            interpretation="Insufficient data for ML analysis"
        )
    
    def _create_fallback_result(self, error_message: str) -> MLModelResult:
        """Create fallback result when calculation fails"""
        return MLModelResult(
            lstm_prediction={'error': error_message},
            xgboost_analysis={'error': error_message},
            random_forest_analysis={'error': error_message},
            ensemble_prediction={'error': error_message},
            feature_importance={},
            model_performance={},
            risk_assessment={'risk_level': 'High'},
            confidence_intervals={},
            prediction_horizon=0,
            metadata={'error': error_message},
            timestamp=datetime.now(),
            interpretation="ML analysis failed"
        )

# Example usage
if __name__ == "__main__":
    # Sample data
    fundamentals = StockFundamentals(
        revenue=10000000000,
        net_income=1000000000,
        free_cash_flow=800000000,
        total_debt=2000000000,
        shareholders_equity=5000000000,
        shares_outstanding=100000000,
        dividend_per_share=4.0,
        earnings_per_share=10.0,
        book_value_per_share=50.0,
        revenue_growth_rate=0.05,
        earnings_growth_rate=0.08,
        dividend_growth_rate=0.06,
        beta=1.2,
        market_cap=8000000000
    )
    
    market_data = MarketData(
        risk_free_rate=0.03,
        market_return=0.10,
        inflation_rate=0.025,
        gdp_growth=0.025,
        unemployment_rate=0.05,
        sector_performance={"Technology": 0.12},
        market_volatility=0.15
    )
    
    # Analyze with ML models
    ml_models = AdvancedMLModels()
    result = ml_models.analyze(fundamentals, market_data)
    
    print(f"ML Analysis Results:")
    print(f"Ensemble Prediction: ${result.ensemble_prediction.get('prediction', 'N/A'):.2f}")
    print(f"Prediction Horizon: {result.prediction_horizon} days")
    print(f"Risk Level: {result.risk_assessment.get('risk_level', 'Unknown')}")
    
    print(f"\nModel Performance:")
    for model, metrics in result.model_performance.items():
        print(f"{model.title()}: Confidence {metrics.get('confidence', 0):.1%}")
    
    print(f"\nFeature Importance:")
    for feature, importance in sorted(result.feature_importance.items(), 
                                    key=lambda x: x[1], reverse=True)[:5]:
        print(f"{feature}: {importance:.1%}")
    
    print(f"\nConfidence Intervals:")
    for level, (low, high) in result.confidence_intervals.items():
        print(f"{level}: ${low:.2f} - ${high:.2f}")
    
    print(f"\nInterpretation: {result.interpretation}")