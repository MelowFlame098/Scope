# Risk Predictor
# Phase 9: AI-First Platform Implementation

import logging
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskType(Enum):
    MARKET_RISK = "market_risk"
    CREDIT_RISK = "credit_risk"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"
    CONCENTRATION_RISK = "concentration_risk"
    VOLATILITY_RISK = "volatility_risk"
    TAIL_RISK = "tail_risk"
    SYSTEMIC_RISK = "systemic_risk"

class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    EXTREME = "extreme"

class TimeHorizon(Enum):
    INTRADAY = "intraday"  # 1 day
    SHORT_TERM = "short_term"  # 1 week
    MEDIUM_TERM = "medium_term"  # 1 month
    LONG_TERM = "long_term"  # 3 months

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    cvar_95: float  # Conditional VaR (95%)
    cvar_99: float  # Conditional VaR (99%)
    max_drawdown: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_risk: float

@dataclass
class RiskPrediction:
    risk_type: RiskType
    risk_level: RiskLevel
    probability: float
    confidence: float
    time_horizon: TimeHorizon
    risk_score: float  # 0-100
    contributing_factors: Dict[str, float]
    recommendations: List[str]
    metrics: RiskMetrics
    metadata: Dict[str, Any]
    created_at: datetime

@dataclass
class PortfolioRiskAssessment:
    portfolio_id: str
    overall_risk_score: float
    risk_level: RiskLevel
    risk_predictions: Dict[RiskType, RiskPrediction]
    diversification_score: float
    concentration_risks: List[Dict[str, Any]]
    stress_test_results: Dict[str, float]
    recommendations: List[str]
    created_at: datetime

class RiskPredictor:
    """Advanced risk prediction and assessment engine"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.risk_models = {}
        self.anomaly_detectors = {}
        self.scalers = {}
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskLevel.VERY_LOW: (0, 20),
            RiskLevel.LOW: (20, 40),
            RiskLevel.MEDIUM: (40, 60),
            RiskLevel.HIGH: (60, 80),
            RiskLevel.VERY_HIGH: (80, 95),
            RiskLevel.EXTREME: (95, 100)
        }
        
        # Market regime indicators
        self.market_regimes = {
            'bull_market': {'volatility': 0.15, 'trend': 0.05},
            'bear_market': {'volatility': 0.25, 'trend': -0.05},
            'sideways': {'volatility': 0.20, 'trend': 0.0},
            'crisis': {'volatility': 0.40, 'trend': -0.15}
        }
        
        # Initialize models
        asyncio.create_task(self._initialize_models())
        
        logger.info("Risk predictor initialized")
    
    async def predict_risk(self, portfolio_data: Dict[str, Any], 
                          risk_type: RiskType = RiskType.MARKET_RISK,
                          time_horizon: TimeHorizon = TimeHorizon.SHORT_TERM) -> RiskPrediction:
        """Predict specific risk for portfolio"""
        try:
            # Extract features from portfolio data
            features = await self._extract_risk_features(portfolio_data)
            
            # Calculate risk metrics
            metrics = await self._calculate_risk_metrics(portfolio_data)
            
            # Predict risk level
            risk_score = await self._predict_risk_score(features, risk_type, time_horizon)
            
            # Determine risk level
            risk_level = self._determine_risk_level(risk_score)
            
            # Calculate probability and confidence
            probability = await self._calculate_risk_probability(features, risk_type)
            confidence = await self._calculate_prediction_confidence(features, risk_type)
            
            # Identify contributing factors
            contributing_factors = await self._identify_risk_factors(features, risk_type)
            
            # Generate recommendations
            recommendations = await self._generate_risk_recommendations(risk_type, risk_level, contributing_factors)
            
            return RiskPrediction(
                risk_type=risk_type,
                risk_level=risk_level,
                probability=probability,
                confidence=confidence,
                time_horizon=time_horizon,
                risk_score=risk_score,
                contributing_factors=contributing_factors,
                recommendations=recommendations,
                metrics=metrics,
                metadata={
                    'portfolio_size': len(portfolio_data.get('positions', [])),
                    'total_value': portfolio_data.get('total_value', 0),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting {risk_type.value} risk: {e}")
            return self._create_fallback_risk_prediction(risk_type, time_horizon)
    
    async def assess_portfolio_risk(self, portfolio_data: Dict[str, Any]) -> PortfolioRiskAssessment:
        """Comprehensive portfolio risk assessment"""
        try:
            # Predict all risk types
            risk_predictions = {}
            for risk_type in RiskType:
                prediction = await self.predict_risk(portfolio_data, risk_type)
                risk_predictions[risk_type] = prediction
            
            # Calculate overall risk score
            overall_risk_score = await self._calculate_overall_risk_score(risk_predictions)
            
            # Determine overall risk level
            overall_risk_level = self._determine_risk_level(overall_risk_score)
            
            # Calculate diversification score
            diversification_score = await self._calculate_diversification_score(portfolio_data)
            
            # Identify concentration risks
            concentration_risks = await self._identify_concentration_risks(portfolio_data)
            
            # Perform stress tests
            stress_test_results = await self._perform_stress_tests(portfolio_data)
            
            # Generate portfolio-level recommendations
            recommendations = await self._generate_portfolio_recommendations(
                overall_risk_level, diversification_score, concentration_risks
            )
            
            return PortfolioRiskAssessment(
                portfolio_id=portfolio_data.get('portfolio_id', 'unknown'),
                overall_risk_score=overall_risk_score,
                risk_level=overall_risk_level,
                risk_predictions=risk_predictions,
                diversification_score=diversification_score,
                concentration_risks=concentration_risks,
                stress_test_results=stress_test_results,
                recommendations=recommendations,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error assessing portfolio risk: {e}")
            return self._create_fallback_portfolio_assessment(portfolio_data)
    
    async def detect_risk_anomalies(self, portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect risk anomalies in portfolio"""
        try:
            anomalies = []
            
            # Extract features
            features = await self._extract_risk_features(portfolio_data)
            
            # Check for anomalies in different risk dimensions
            for risk_type in RiskType:
                detector = self.anomaly_detectors.get(risk_type)
                if detector:
                    # Prepare features for this risk type
                    risk_features = self._prepare_risk_features(features, risk_type)
                    
                    # Detect anomalies
                    anomaly_scores = detector.decision_function(risk_features.reshape(1, -1))
                    is_anomaly = detector.predict(risk_features.reshape(1, -1))[0] == -1
                    
                    if is_anomaly:
                        anomalies.append({
                            'risk_type': risk_type.value,
                            'anomaly_score': float(anomaly_scores[0]),
                            'severity': 'high' if anomaly_scores[0] < -0.5 else 'medium',
                            'description': f"Anomalous {risk_type.value} detected",
                            'timestamp': datetime.now().isoformat()
                        })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting risk anomalies: {e}")
            return []
    
    async def calculate_var(self, returns: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
        """Calculate Value at Risk (VaR) and Conditional VaR"""
        try:
            returns_array = np.array(returns)
            
            if len(returns_array) < 30:
                logger.warning("Insufficient data for reliable VaR calculation")
                return {'var': 0.0, 'cvar': 0.0}
            
            # Sort returns
            sorted_returns = np.sort(returns_array)
            
            # Calculate VaR
            var_index = int((1 - confidence_level) * len(sorted_returns))
            var = -sorted_returns[var_index] if var_index < len(sorted_returns) else 0.0
            
            # Calculate Conditional VaR (Expected Shortfall)
            tail_returns = sorted_returns[:var_index] if var_index > 0 else sorted_returns[:1]
            cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else 0.0
            
            # Parametric VaR (assuming normal distribution)
            mean_return = np.mean(returns_array)
            std_return = np.std(returns_array)
            z_score = stats.norm.ppf(1 - confidence_level)
            parametric_var = -(mean_return + z_score * std_return)
            
            # Monte Carlo VaR
            mc_var = await self._monte_carlo_var(returns_array, confidence_level)
            
            return {
                'historical_var': float(var),
                'conditional_var': float(cvar),
                'parametric_var': float(parametric_var),
                'monte_carlo_var': float(mc_var),
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {'var': 0.0, 'cvar': 0.0}
    
    async def _initialize_models(self):
        """Initialize risk prediction models"""
        try:
            # Initialize anomaly detectors for each risk type
            for risk_type in RiskType:
                self.anomaly_detectors[risk_type] = IsolationForest(
                    contamination=0.1,
                    random_state=42
                )
                
                # Initialize risk classification models
                self.risk_models[risk_type] = RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                )
                
                # Initialize scalers
                self.scalers[risk_type] = StandardScaler()
            
            # Train models with synthetic data (in production, use historical data)
            await self._train_models_with_synthetic_data()
            
            logger.info("Risk prediction models initialized")
            
        except Exception as e:
            logger.error(f"Error initializing risk models: {e}")
    
    async def _extract_risk_features(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract risk-relevant features from portfolio data"""
        try:
            features = {}
            
            positions = portfolio_data.get('positions', [])
            total_value = portfolio_data.get('total_value', 1.0)
            
            if not positions:
                return self._get_default_features()
            
            # Portfolio composition features
            features['position_count'] = len(positions)
            features['avg_position_size'] = total_value / len(positions) if positions else 0
            
            # Concentration features
            position_weights = [pos.get('value', 0) / total_value for pos in positions]
            features['max_position_weight'] = max(position_weights) if position_weights else 0
            features['herfindahl_index'] = sum(w**2 for w in position_weights)
            
            # Sector/Asset class diversification
            sectors = [pos.get('sector', 'unknown') for pos in positions]
            unique_sectors = len(set(sectors))
            features['sector_diversification'] = unique_sectors / len(positions) if positions else 0
            
            # Volatility features
            volatilities = [pos.get('volatility', 0.2) for pos in positions]
            features['avg_volatility'] = np.mean(volatilities)
            features['max_volatility'] = max(volatilities) if volatilities else 0
            features['volatility_dispersion'] = np.std(volatilities) if len(volatilities) > 1 else 0
            
            # Correlation features
            correlations = []
            for i, pos1 in enumerate(positions):
                for j, pos2 in enumerate(positions[i+1:], i+1):
                    corr = pos1.get('correlation_with_market', 0.5)
                    correlations.append(corr)
            
            features['avg_correlation'] = np.mean(correlations) if correlations else 0.5
            features['max_correlation'] = max(correlations) if correlations else 0.5
            
            # Liquidity features
            liquidities = [pos.get('liquidity_score', 0.5) for pos in positions]
            features['avg_liquidity'] = np.mean(liquidities)
            features['min_liquidity'] = min(liquidities) if liquidities else 0.5
            
            # Market cap features
            market_caps = [pos.get('market_cap', 1e9) for pos in positions]
            features['avg_market_cap'] = np.mean(market_caps)
            features['small_cap_ratio'] = sum(1 for mc in market_caps if mc < 2e9) / len(market_caps)
            
            # Geographic diversification
            countries = [pos.get('country', 'US') for pos in positions]
            unique_countries = len(set(countries))
            features['geographic_diversification'] = unique_countries / len(positions) if positions else 0
            
            # Time-based features
            now = datetime.now()
            features['hour_of_day'] = now.hour
            features['day_of_week'] = now.weekday()
            features['month'] = now.month
            
            # Market regime features
            market_regime = await self._detect_market_regime()
            features.update(market_regime)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting risk features: {e}")
            return self._get_default_features()
    
    async def _calculate_risk_metrics(self, portfolio_data: Dict[str, Any]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        try:
            # Get portfolio returns (synthetic for demo)
            returns = self._generate_synthetic_returns(portfolio_data)
            
            # Calculate VaR metrics
            var_results = await self.calculate_var(returns, 0.95)
            var_99_results = await self.calculate_var(returns, 0.99)
            
            # Calculate other metrics
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Sharpe ratio
            risk_free_rate = 0.02  # 2% risk-free rate
            excess_returns = np.mean(returns) - risk_free_rate/252
            sharpe_ratio = excess_returns / np.std(returns) if np.std(returns) > 0 else 0
            
            # Sortino ratio
            downside_returns = [r for r in returns if r < 0]
            downside_std = np.std(downside_returns) if downside_returns else np.std(returns)
            sortino_ratio = excess_returns / downside_std if downside_std > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = np.min(drawdowns)
            
            # Beta (correlation with market)
            market_returns = np.random.normal(0.0008, 0.015, len(returns))  # Synthetic market
            beta = np.cov(returns, market_returns)[0, 1] / np.var(market_returns) if np.var(market_returns) > 0 else 1.0
            
            # Correlation risk
            correlation_risk = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
            
            return RiskMetrics(
                var_95=var_results.get('historical_var', 0.0),
                var_99=var_99_results.get('historical_var', 0.0),
                cvar_95=var_results.get('conditional_var', 0.0),
                cvar_99=var_99_results.get('conditional_var', 0.0),
                max_drawdown=float(max_drawdown),
                volatility=float(volatility),
                sharpe_ratio=float(sharpe_ratio),
                sortino_ratio=float(sortino_ratio),
                beta=float(beta),
                correlation_risk=float(abs(correlation_risk))
            )
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return self._get_default_risk_metrics()
    
    async def _predict_risk_score(self, features: Dict[str, float], 
                                 risk_type: RiskType, time_horizon: TimeHorizon) -> float:
        """Predict risk score for specific risk type"""
        try:
            # Prepare features for the specific risk type
            risk_features = self._prepare_risk_features(features, risk_type)
            
            # Base risk score calculation
            base_score = 0.0
            
            if risk_type == RiskType.MARKET_RISK:
                base_score = (
                    features.get('avg_volatility', 0.2) * 30 +
                    features.get('max_correlation', 0.5) * 20 +
                    features.get('herfindahl_index', 0.1) * 25 +
                    (1 - features.get('sector_diversification', 0.5)) * 25
                )
            
            elif risk_type == RiskType.CONCENTRATION_RISK:
                base_score = (
                    features.get('max_position_weight', 0.1) * 40 +
                    features.get('herfindahl_index', 0.1) * 30 +
                    (1 - features.get('sector_diversification', 0.5)) * 30
                )
            
            elif risk_type == RiskType.LIQUIDITY_RISK:
                base_score = (
                    (1 - features.get('avg_liquidity', 0.5)) * 50 +
                    (1 - features.get('min_liquidity', 0.5)) * 30 +
                    features.get('small_cap_ratio', 0.2) * 20
                )
            
            elif risk_type == RiskType.VOLATILITY_RISK:
                base_score = (
                    features.get('avg_volatility', 0.2) * 40 +
                    features.get('max_volatility', 0.2) * 30 +
                    features.get('volatility_dispersion', 0.05) * 30
                )
            
            else:
                # Generic risk calculation
                base_score = (
                    features.get('avg_volatility', 0.2) * 25 +
                    features.get('herfindahl_index', 0.1) * 25 +
                    (1 - features.get('avg_liquidity', 0.5)) * 25 +
                    features.get('max_correlation', 0.5) * 25
                )
            
            # Adjust for time horizon
            horizon_multiplier = {
                TimeHorizon.INTRADAY: 0.5,
                TimeHorizon.SHORT_TERM: 1.0,
                TimeHorizon.MEDIUM_TERM: 1.3,
                TimeHorizon.LONG_TERM: 1.6
            }.get(time_horizon, 1.0)
            
            # Adjust for market regime
            market_stress = features.get('market_stress', 0.0)
            regime_multiplier = 1.0 + market_stress
            
            final_score = base_score * horizon_multiplier * regime_multiplier
            
            # Ensure score is between 0 and 100
            return max(0.0, min(100.0, final_score))
            
        except Exception as e:
            logger.error(f"Error predicting risk score: {e}")
            return 50.0  # Default medium risk
    
    async def _calculate_risk_probability(self, features: Dict[str, float], risk_type: RiskType) -> float:
        """Calculate probability of risk event occurring"""
        try:
            # Use logistic function to convert features to probability
            risk_indicators = {
                RiskType.MARKET_RISK: ['avg_volatility', 'market_stress', 'max_correlation'],
                RiskType.LIQUIDITY_RISK: ['min_liquidity', 'small_cap_ratio'],
                RiskType.CONCENTRATION_RISK: ['max_position_weight', 'herfindahl_index'],
                RiskType.VOLATILITY_RISK: ['avg_volatility', 'volatility_dispersion']
            }
            
            indicators = risk_indicators.get(risk_type, ['avg_volatility', 'market_stress'])
            
            # Calculate weighted sum of indicators
            weighted_sum = sum(features.get(indicator, 0.5) for indicator in indicators)
            normalized_sum = weighted_sum / len(indicators)
            
            # Convert to probability using sigmoid function
            probability = 1 / (1 + np.exp(-5 * (normalized_sum - 0.5)))
            
            return float(probability)
            
        except Exception as e:
            logger.error(f"Error calculating risk probability: {e}")
            return 0.5
    
    async def _calculate_prediction_confidence(self, features: Dict[str, float], risk_type: RiskType) -> float:
        """Calculate confidence in risk prediction"""
        try:
            # Confidence based on data quality and model certainty
            data_quality_factors = [
                features.get('position_count', 0) / 20,  # More positions = higher confidence
                min(1.0, features.get('avg_liquidity', 0.5) * 2),  # Higher liquidity = higher confidence
                1.0 - features.get('volatility_dispersion', 0.1) * 5  # Lower dispersion = higher confidence
            ]
            
            # Calculate base confidence
            base_confidence = np.mean([max(0.0, min(1.0, factor)) for factor in data_quality_factors])
            
            # Adjust for market conditions
            market_stress = features.get('market_stress', 0.0)
            stress_adjustment = 1.0 - market_stress * 0.3
            
            final_confidence = base_confidence * stress_adjustment
            
            return max(0.1, min(1.0, final_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating prediction confidence: {e}")
            return 0.5
    
    async def _identify_risk_factors(self, features: Dict[str, float], risk_type: RiskType) -> Dict[str, float]:
        """Identify contributing risk factors"""
        try:
            factors = {}
            
            if risk_type == RiskType.MARKET_RISK:
                factors = {
                    'volatility': features.get('avg_volatility', 0.2) * 100,
                    'correlation': features.get('max_correlation', 0.5) * 100,
                    'concentration': features.get('herfindahl_index', 0.1) * 100,
                    'diversification': (1 - features.get('sector_diversification', 0.5)) * 100
                }
            
            elif risk_type == RiskType.LIQUIDITY_RISK:
                factors = {
                    'low_liquidity_positions': (1 - features.get('min_liquidity', 0.5)) * 100,
                    'small_cap_exposure': features.get('small_cap_ratio', 0.2) * 100,
                    'market_stress': features.get('market_stress', 0.0) * 100
                }
            
            elif risk_type == RiskType.CONCENTRATION_RISK:
                factors = {
                    'large_positions': features.get('max_position_weight', 0.1) * 100,
                    'sector_concentration': (1 - features.get('sector_diversification', 0.5)) * 100,
                    'geographic_concentration': (1 - features.get('geographic_diversification', 0.5)) * 100
                }
            
            else:
                factors = {
                    'general_risk': 50.0,
                    'market_conditions': features.get('market_stress', 0.0) * 100
                }
            
            # Normalize factors to sum to 100
            total = sum(factors.values())
            if total > 0:
                factors = {k: (v / total) * 100 for k, v in factors.items()}
            
            return factors
            
        except Exception as e:
            logger.error(f"Error identifying risk factors: {e}")
            return {'unknown_factor': 100.0}
    
    async def _generate_risk_recommendations(self, risk_type: RiskType, risk_level: RiskLevel, 
                                           contributing_factors: Dict[str, float]) -> List[str]:
        """Generate risk mitigation recommendations"""
        try:
            recommendations = []
            
            # General recommendations based on risk level
            if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
                recommendations.append("Consider reducing position sizes to lower overall risk exposure")
                recommendations.append("Implement stop-loss orders to limit potential losses")
            
            # Specific recommendations based on risk type
            if risk_type == RiskType.MARKET_RISK:
                if 'volatility' in contributing_factors and contributing_factors['volatility'] > 30:
                    recommendations.append("Reduce exposure to high-volatility assets")
                if 'correlation' in contributing_factors and contributing_factors['correlation'] > 30:
                    recommendations.append("Diversify into uncorrelated assets")
            
            elif risk_type == RiskType.CONCENTRATION_RISK:
                if 'large_positions' in contributing_factors and contributing_factors['large_positions'] > 30:
                    recommendations.append("Reduce concentration in large positions")
                if 'sector_concentration' in contributing_factors and contributing_factors['sector_concentration'] > 30:
                    recommendations.append("Diversify across different sectors")
            
            elif risk_type == RiskType.LIQUIDITY_RISK:
                if 'low_liquidity_positions' in contributing_factors and contributing_factors['low_liquidity_positions'] > 30:
                    recommendations.append("Increase allocation to liquid assets")
                if 'small_cap_exposure' in contributing_factors and contributing_factors['small_cap_exposure'] > 30:
                    recommendations.append("Consider reducing small-cap exposure")
            
            # Add default recommendation if none generated
            if not recommendations:
                recommendations.append("Monitor risk metrics regularly and adjust portfolio as needed")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Monitor portfolio risk regularly"]
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine risk level from risk score"""
        for level, (min_score, max_score) in self.risk_thresholds.items():
            if min_score <= risk_score < max_score:
                return level
        return RiskLevel.EXTREME
    
    def _prepare_risk_features(self, features: Dict[str, float], risk_type: RiskType) -> np.ndarray:
        """Prepare features for specific risk type"""
        try:
            # Select relevant features for each risk type
            risk_feature_map = {
                RiskType.MARKET_RISK: ['avg_volatility', 'max_correlation', 'herfindahl_index', 'sector_diversification'],
                RiskType.LIQUIDITY_RISK: ['avg_liquidity', 'min_liquidity', 'small_cap_ratio'],
                RiskType.CONCENTRATION_RISK: ['max_position_weight', 'herfindahl_index', 'sector_diversification'],
                RiskType.VOLATILITY_RISK: ['avg_volatility', 'max_volatility', 'volatility_dispersion']
            }
            
            relevant_features = risk_feature_map.get(risk_type, list(features.keys())[:5])
            feature_vector = [features.get(feature, 0.0) for feature in relevant_features]
            
            return np.array(feature_vector)
            
        except Exception as e:
            logger.error(f"Error preparing risk features: {e}")
            return np.array([0.0] * 5)
    
    async def _detect_market_regime(self) -> Dict[str, float]:
        """Detect current market regime"""
        try:
            # Simplified market regime detection
            # In production, this would analyze real market data
            
            # Generate synthetic market indicators
            volatility = np.random.uniform(0.1, 0.4)
            trend = np.random.uniform(-0.1, 0.1)
            
            # Determine regime
            if volatility > 0.3:
                regime = 'crisis'
                stress_level = 0.8
            elif volatility > 0.25 and trend < -0.02:
                regime = 'bear_market'
                stress_level = 0.6
            elif volatility < 0.18 and trend > 0.02:
                regime = 'bull_market'
                stress_level = 0.2
            else:
                regime = 'sideways'
                stress_level = 0.4
            
            return {
                'market_regime': regime,
                'market_volatility': volatility,
                'market_trend': trend,
                'market_stress': stress_level
            }
            
        except Exception as e:
            logger.error(f"Error detecting market regime: {e}")
            return {
                'market_regime': 'unknown',
                'market_volatility': 0.2,
                'market_trend': 0.0,
                'market_stress': 0.5
            }
    
    async def _monte_carlo_var(self, returns: np.ndarray, confidence_level: float, 
                              num_simulations: int = 10000) -> float:
        """Calculate VaR using Monte Carlo simulation"""
        try:
            if len(returns) < 10:
                return 0.0
            
            # Fit distribution to returns
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Generate random scenarios
            simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
            
            # Calculate VaR
            var_index = int((1 - confidence_level) * num_simulations)
            sorted_returns = np.sort(simulated_returns)
            var = -sorted_returns[var_index] if var_index < len(sorted_returns) else 0.0
            
            return float(var)
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo VaR: {e}")
            return 0.0
    
    def _generate_synthetic_returns(self, portfolio_data: Dict[str, Any], days: int = 252) -> List[float]:
        """Generate synthetic returns for demonstration"""
        try:
            # Base parameters
            annual_return = 0.08
            annual_volatility = 0.15
            
            # Adjust based on portfolio characteristics
            positions = portfolio_data.get('positions', [])
            if positions:
                avg_volatility = np.mean([pos.get('volatility', 0.15) for pos in positions])
                annual_volatility = avg_volatility
            
            # Generate returns
            daily_return = annual_return / 252
            daily_volatility = annual_volatility / np.sqrt(252)
            
            returns = np.random.normal(daily_return, daily_volatility, days)
            
            return returns.tolist()
            
        except Exception as e:
            logger.error(f"Error generating synthetic returns: {e}")
            return [0.0] * days
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values"""
        return {
            'position_count': 10,
            'avg_position_size': 10000,
            'max_position_weight': 0.1,
            'herfindahl_index': 0.1,
            'sector_diversification': 0.5,
            'avg_volatility': 0.2,
            'max_volatility': 0.3,
            'volatility_dispersion': 0.05,
            'avg_correlation': 0.5,
            'max_correlation': 0.7,
            'avg_liquidity': 0.7,
            'min_liquidity': 0.5,
            'avg_market_cap': 5e9,
            'small_cap_ratio': 0.2,
            'geographic_diversification': 0.3,
            'market_stress': 0.3
        }
    
    def _get_default_risk_metrics(self) -> RiskMetrics:
        """Get default risk metrics"""
        return RiskMetrics(
            var_95=0.02,
            var_99=0.04,
            cvar_95=0.03,
            cvar_99=0.06,
            max_drawdown=-0.15,
            volatility=0.15,
            sharpe_ratio=0.8,
            sortino_ratio=1.0,
            beta=1.0,
            correlation_risk=0.3
        )
    
    def _create_fallback_risk_prediction(self, risk_type: RiskType, time_horizon: TimeHorizon) -> RiskPrediction:
        """Create fallback risk prediction"""
        return RiskPrediction(
            risk_type=risk_type,
            risk_level=RiskLevel.MEDIUM,
            probability=0.5,
            confidence=0.3,
            time_horizon=time_horizon,
            risk_score=50.0,
            contributing_factors={'unknown': 100.0},
            recommendations=["Monitor risk regularly"],
            metrics=self._get_default_risk_metrics(),
            metadata={'fallback': True},
            created_at=datetime.now()
        )
    
    def _create_fallback_portfolio_assessment(self, portfolio_data: Dict[str, Any]) -> PortfolioRiskAssessment:
        """Create fallback portfolio assessment"""
        return PortfolioRiskAssessment(
            portfolio_id=portfolio_data.get('portfolio_id', 'unknown'),
            overall_risk_score=50.0,
            risk_level=RiskLevel.MEDIUM,
            risk_predictions={},
            diversification_score=0.5,
            concentration_risks=[],
            stress_test_results={},
            recommendations=["Monitor portfolio risk"],
            created_at=datetime.now()
        )
    
    async def _calculate_overall_risk_score(self, risk_predictions: Dict[RiskType, RiskPrediction]) -> float:
        """Calculate overall portfolio risk score"""
        try:
            if not risk_predictions:
                return 50.0
            
            # Weight different risk types
            risk_weights = {
                RiskType.MARKET_RISK: 0.3,
                RiskType.CONCENTRATION_RISK: 0.2,
                RiskType.LIQUIDITY_RISK: 0.2,
                RiskType.VOLATILITY_RISK: 0.15,
                RiskType.CREDIT_RISK: 0.1,
                RiskType.OPERATIONAL_RISK: 0.05
            }
            
            weighted_score = 0.0
            total_weight = 0.0
            
            for risk_type, prediction in risk_predictions.items():
                weight = risk_weights.get(risk_type, 0.1)
                weighted_score += prediction.risk_score * weight
                total_weight += weight
            
            return weighted_score / total_weight if total_weight > 0 else 50.0
            
        except Exception as e:
            logger.error(f"Error calculating overall risk score: {e}")
            return 50.0
    
    async def _calculate_diversification_score(self, portfolio_data: Dict[str, Any]) -> float:
        """Calculate portfolio diversification score"""
        try:
            positions = portfolio_data.get('positions', [])
            if not positions:
                return 0.0
            
            # Calculate various diversification metrics
            total_value = portfolio_data.get('total_value', 1.0)
            
            # Position size diversification
            position_weights = [pos.get('value', 0) / total_value for pos in positions]
            herfindahl_index = sum(w**2 for w in position_weights)
            position_diversification = 1 - herfindahl_index
            
            # Sector diversification
            sectors = [pos.get('sector', 'unknown') for pos in positions]
            unique_sectors = len(set(sectors))
            sector_diversification = min(1.0, unique_sectors / 10)  # Assume 10 sectors is fully diversified
            
            # Geographic diversification
            countries = [pos.get('country', 'US') for pos in positions]
            unique_countries = len(set(countries))
            geographic_diversification = min(1.0, unique_countries / 5)  # Assume 5 countries is well diversified
            
            # Asset class diversification
            asset_classes = [pos.get('asset_class', 'equity') for pos in positions]
            unique_asset_classes = len(set(asset_classes))
            asset_class_diversification = min(1.0, unique_asset_classes / 4)  # 4 main asset classes
            
            # Weighted average
            diversification_score = (
                position_diversification * 0.4 +
                sector_diversification * 0.3 +
                geographic_diversification * 0.2 +
                asset_class_diversification * 0.1
            )
            
            return max(0.0, min(1.0, diversification_score))
            
        except Exception as e:
            logger.error(f"Error calculating diversification score: {e}")
            return 0.5
    
    async def _identify_concentration_risks(self, portfolio_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify concentration risks in portfolio"""
        try:
            concentration_risks = []
            positions = portfolio_data.get('positions', [])
            total_value = portfolio_data.get('total_value', 1.0)
            
            if not positions:
                return concentration_risks
            
            # Position concentration
            for pos in positions:
                weight = pos.get('value', 0) / total_value
                if weight > 0.1:  # More than 10% in single position
                    concentration_risks.append({
                        'type': 'position_concentration',
                        'symbol': pos.get('symbol', 'unknown'),
                        'weight': weight,
                        'severity': 'high' if weight > 0.2 else 'medium',
                        'description': f"Large position concentration: {weight:.1%}"
                    })
            
            # Sector concentration
            sector_weights = {}
            for pos in positions:
                sector = pos.get('sector', 'unknown')
                weight = pos.get('value', 0) / total_value
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            for sector, weight in sector_weights.items():
                if weight > 0.3:  # More than 30% in single sector
                    concentration_risks.append({
                        'type': 'sector_concentration',
                        'sector': sector,
                        'weight': weight,
                        'severity': 'high' if weight > 0.5 else 'medium',
                        'description': f"High sector concentration in {sector}: {weight:.1%}"
                    })
            
            return concentration_risks
            
        except Exception as e:
            logger.error(f"Error identifying concentration risks: {e}")
            return []
    
    async def _perform_stress_tests(self, portfolio_data: Dict[str, Any]) -> Dict[str, float]:
        """Perform stress tests on portfolio"""
        try:
            # Generate synthetic returns for stress testing
            returns = self._generate_synthetic_returns(portfolio_data, 252)
            
            stress_scenarios = {
                'market_crash_2008': {'return_shock': -0.3, 'volatility_shock': 2.0},
                'covid_crash_2020': {'return_shock': -0.25, 'volatility_shock': 1.8},
                'interest_rate_shock': {'return_shock': -0.15, 'volatility_shock': 1.3},
                'inflation_shock': {'return_shock': -0.1, 'volatility_shock': 1.2},
                'liquidity_crisis': {'return_shock': -0.2, 'volatility_shock': 1.5}
            }
            
            stress_results = {}
            
            for scenario, shocks in stress_scenarios.items():
                # Apply shocks to returns
                shocked_returns = np.array(returns) + shocks['return_shock']
                shocked_returns = shocked_returns * shocks['volatility_shock']
                
                # Calculate portfolio impact
                portfolio_return = np.sum(shocked_returns)
                max_drawdown = np.min(np.cumsum(shocked_returns))
                
                stress_results[scenario] = {
                    'portfolio_return': float(portfolio_return),
                    'max_drawdown': float(max_drawdown)
                }
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Error performing stress tests: {e}")
            return {}
    
    async def _generate_portfolio_recommendations(self, risk_level: RiskLevel, 
                                                diversification_score: float,
                                                concentration_risks: List[Dict[str, Any]]) -> List[str]:
        """Generate portfolio-level recommendations"""
        try:
            recommendations = []
            
            # Risk level recommendations
            if risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH, RiskLevel.EXTREME]:
                recommendations.append("Consider reducing overall portfolio risk through position sizing")
                recommendations.append("Implement hedging strategies to protect against downside risk")
            
            # Diversification recommendations
            if diversification_score < 0.5:
                recommendations.append("Improve portfolio diversification across sectors and asset classes")
            
            if diversification_score < 0.3:
                recommendations.append("Consider adding international exposure for geographic diversification")
            
            # Concentration risk recommendations
            if concentration_risks:
                recommendations.append("Address concentration risks by reducing large position sizes")
                
                sector_concentrations = [risk for risk in concentration_risks if risk['type'] == 'sector_concentration']
                if sector_concentrations:
                    recommendations.append("Diversify across different sectors to reduce sector concentration risk")
            
            # Default recommendation
            if not recommendations:
                recommendations.append("Continue monitoring portfolio risk metrics and maintain current allocation")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating portfolio recommendations: {e}")
            return ["Monitor portfolio risk regularly"]
    
    async def _train_models_with_synthetic_data(self):
        """Train models with synthetic data for demonstration"""
        try:
            # Generate synthetic training data
            n_samples = 1000
            n_features = 10
            
            for risk_type in RiskType:
                # Generate features
                X = np.random.random((n_samples, n_features))
                
                # Generate labels (risk levels)
                risk_scores = np.random.random(n_samples) * 100
                y = [self._determine_risk_level(score).value for score in risk_scores]
                
                # Train anomaly detector
                self.anomaly_detectors[risk_type].fit(X)
                
                # Train risk classifier
                # Note: This is simplified - in production, use proper categorical encoding
                y_numeric = [list(RiskLevel).index(RiskLevel(label)) for label in y]
                self.risk_models[risk_type].fit(X, y_numeric)
                
                # Fit scaler
                self.scalers[risk_type].fit(X)
            
            logger.info("Models trained with synthetic data")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")