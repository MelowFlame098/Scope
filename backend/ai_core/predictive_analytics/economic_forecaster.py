# Economic Forecaster
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EconomicIndicator(Enum):
    GDP_GROWTH = "gdp_growth"
    INFLATION_RATE = "inflation_rate"
    UNEMPLOYMENT_RATE = "unemployment_rate"
    INTEREST_RATE = "interest_rate"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    INDUSTRIAL_PRODUCTION = "industrial_production"
    RETAIL_SALES = "retail_sales"
    HOUSING_STARTS = "housing_starts"
    TRADE_BALANCE = "trade_balance"
    MONEY_SUPPLY = "money_supply"
    YIELD_CURVE_SPREAD = "yield_curve_spread"
    CREDIT_SPREADS = "credit_spreads"
    COMMODITY_PRICES = "commodity_prices"
    CURRENCY_STRENGTH = "currency_strength"
    STOCK_MARKET_PERFORMANCE = "stock_market_performance"

class ForecastHorizon(Enum):
    SHORT_TERM = "short_term"      # 1-3 months
    MEDIUM_TERM = "medium_term"    # 3-12 months
    LONG_TERM = "long_term"        # 1-3 years

class ForecastConfidence(Enum):
    LOW = "low"                    # 0-50%
    MODERATE = "moderate"          # 50-75%
    HIGH = "high"                  # 75-90%
    VERY_HIGH = "very_high"        # 90%+

class EconomicRegime(Enum):
    EXPANSION = "expansion"
    PEAK = "peak"
    CONTRACTION = "contraction"
    TROUGH = "trough"
    STAGFLATION = "stagflation"
    DEFLATION = "deflation"
    RECOVERY = "recovery"

@dataclass
class EconomicData:
    indicator: EconomicIndicator
    value: float
    timestamp: datetime
    source: str
    quality_score: float  # 0-1
    seasonal_adjusted: bool
    revision_history: List[float]

@dataclass
class EconomicForecast:
    indicator: EconomicIndicator
    horizon: ForecastHorizon
    forecast_value: float
    confidence_interval: Tuple[float, float]
    confidence_level: ForecastConfidence
    methodology: str
    key_drivers: List[str]
    risks: List[str]
    forecast_date: datetime
    model_accuracy: float

@dataclass
class RegimeAnalysis:
    current_regime: EconomicRegime
    regime_probability: float
    regime_duration: int  # months
    transition_probabilities: Dict[EconomicRegime, float]
    key_indicators: List[str]
    regime_characteristics: Dict[str, float]
    analysis_timestamp: datetime

@dataclass
class EconomicScenario:
    scenario_name: str
    probability: float
    time_horizon: int  # months
    indicator_forecasts: Dict[EconomicIndicator, float]
    market_implications: Dict[str, float]
    policy_implications: List[str]
    scenario_description: str
    creation_timestamp: datetime

class EconomicForecaster:
    """Advanced economic forecasting and regime analysis engine"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.economic_data = {}
        self.forecast_cache = {}
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'random_state': 42
            },
            'linear': {
                'alpha': 1.0
            }
        }
        
        # Initialize models for each indicator
        self._initialize_models()
        
        # Economic regime transition matrix
        self.regime_transitions = {
            EconomicRegime.EXPANSION: {
                EconomicRegime.EXPANSION: 0.7,
                EconomicRegime.PEAK: 0.2,
                EconomicRegime.CONTRACTION: 0.1
            },
            EconomicRegime.PEAK: {
                EconomicRegime.PEAK: 0.3,
                EconomicRegime.CONTRACTION: 0.6,
                EconomicRegime.EXPANSION: 0.1
            },
            EconomicRegime.CONTRACTION: {
                EconomicRegime.CONTRACTION: 0.6,
                EconomicRegime.TROUGH: 0.3,
                EconomicRegime.RECOVERY: 0.1
            },
            EconomicRegime.TROUGH: {
                EconomicRegime.TROUGH: 0.2,
                EconomicRegime.RECOVERY: 0.7,
                EconomicRegime.CONTRACTION: 0.1
            },
            EconomicRegime.RECOVERY: {
                EconomicRegime.RECOVERY: 0.4,
                EconomicRegime.EXPANSION: 0.5,
                EconomicRegime.TROUGH: 0.1
            }
        }
        
        logger.info("Economic forecaster initialized")
    
    async def forecast_indicator(self, indicator: EconomicIndicator,
                               horizon: ForecastHorizon,
                               use_ensemble: bool = True) -> Optional[EconomicForecast]:
        """Generate forecast for specific economic indicator"""
        try:
            # Check cache first
            cache_key = f"{indicator.value}_{horizon.value}"
            if cache_key in self.forecast_cache:
                cached_forecast = self.forecast_cache[cache_key]
                if (datetime.now() - cached_forecast.forecast_date).hours < 6:
                    return cached_forecast
            
            # Get historical data
            historical_data = await self._get_indicator_data(indicator)
            if historical_data is None or len(historical_data) < 50:
                logger.warning(f"Insufficient data for {indicator.value}")
                return None
            
            # Prepare features
            features = await self._prepare_features(indicator, historical_data)
            if features is None:
                return None
            
            # Generate forecast
            if use_ensemble:
                forecast_result = await self._ensemble_forecast(indicator, features, horizon)
            else:
                forecast_result = await self._single_model_forecast(indicator, features, horizon)
            
            if forecast_result is None:
                return None
            
            # Create forecast object
            forecast = EconomicForecast(
                indicator=indicator,
                horizon=horizon,
                forecast_value=forecast_result['value'],
                confidence_interval=forecast_result['confidence_interval'],
                confidence_level=forecast_result['confidence_level'],
                methodology=forecast_result['methodology'],
                key_drivers=forecast_result['key_drivers'],
                risks=forecast_result['risks'],
                forecast_date=datetime.now(),
                model_accuracy=forecast_result['accuracy']
            )
            
            # Cache forecast
            self.forecast_cache[cache_key] = forecast
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error forecasting {indicator.value}: {e}")
            return None
    
    async def analyze_economic_regime(self) -> Optional[RegimeAnalysis]:
        """Analyze current economic regime and transition probabilities"""
        try:
            # Get key economic indicators
            key_indicators = [
                EconomicIndicator.GDP_GROWTH,
                EconomicIndicator.INFLATION_RATE,
                EconomicIndicator.UNEMPLOYMENT_RATE,
                EconomicIndicator.YIELD_CURVE_SPREAD
            ]
            
            indicator_values = {}
            for indicator in key_indicators:
                data = await self._get_indicator_data(indicator)
                if data is not None and len(data) > 0:
                    indicator_values[indicator] = data.iloc[-1]
            
            if len(indicator_values) < 3:
                logger.warning("Insufficient data for regime analysis")
                return None
            
            # Determine current regime
            current_regime = await self._classify_regime(indicator_values)
            
            # Calculate regime probability
            regime_probability = await self._calculate_regime_probability(
                current_regime, indicator_values
            )
            
            # Estimate regime duration
            regime_duration = await self._estimate_regime_duration(current_regime)
            
            # Calculate transition probabilities
            transition_probs = self.regime_transitions.get(current_regime, {})
            
            # Identify key indicators for this regime
            key_regime_indicators = await self._get_regime_indicators(current_regime)
            
            # Get regime characteristics
            regime_chars = await self._get_regime_characteristics(
                current_regime, indicator_values
            )
            
            return RegimeAnalysis(
                current_regime=current_regime,
                regime_probability=regime_probability,
                regime_duration=regime_duration,
                transition_probabilities=transition_probs,
                key_indicators=key_regime_indicators,
                regime_characteristics=regime_chars,
                analysis_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing economic regime: {e}")
            return None
    
    async def generate_economic_scenarios(self, num_scenarios: int = 5) -> List[EconomicScenario]:
        """Generate multiple economic scenarios with probabilities"""
        try:
            scenarios = []
            
            # Define base scenarios
            base_scenarios = [
                {
                    'name': 'Baseline Growth',
                    'probability': 0.4,
                    'description': 'Moderate economic growth with stable inflation and employment'
                },
                {
                    'name': 'Accelerated Growth',
                    'probability': 0.2,
                    'description': 'Strong economic expansion with rising inflation pressures'
                },
                {
                    'name': 'Economic Slowdown',
                    'probability': 0.25,
                    'description': 'Slowing growth with potential recession risks'
                },
                {
                    'name': 'Stagflation',
                    'probability': 0.1,
                    'description': 'Low growth combined with persistent high inflation'
                },
                {
                    'name': 'Deflationary Spiral',
                    'probability': 0.05,
                    'description': 'Economic contraction with falling prices and high unemployment'
                }
            ]
            
            for i, base_scenario in enumerate(base_scenarios[:num_scenarios]):
                try:
                    # Generate indicator forecasts for this scenario
                    indicator_forecasts = await self._generate_scenario_forecasts(
                        base_scenario['name']
                    )
                    
                    # Calculate market implications
                    market_implications = await self._calculate_market_implications(
                        indicator_forecasts
                    )
                    
                    # Generate policy implications
                    policy_implications = await self._generate_policy_implications(
                        base_scenario['name'], indicator_forecasts
                    )
                    
                    scenario = EconomicScenario(
                        scenario_name=base_scenario['name'],
                        probability=base_scenario['probability'],
                        time_horizon=12,  # 12 months
                        indicator_forecasts=indicator_forecasts,
                        market_implications=market_implications,
                        policy_implications=policy_implications,
                        scenario_description=base_scenario['description'],
                        creation_timestamp=datetime.now()
                    )
                    
                    scenarios.append(scenario)
                    
                except Exception as e:
                    logger.error(f"Error generating scenario {base_scenario['name']}: {e}")
                    continue
            
            return scenarios
            
        except Exception as e:
            logger.error(f"Error generating economic scenarios: {e}")
            return []
    
    async def forecast_multiple_indicators(self, indicators: List[EconomicIndicator],
                                         horizon: ForecastHorizon) -> Dict[EconomicIndicator, EconomicForecast]:
        """Generate forecasts for multiple indicators simultaneously"""
        try:
            forecasts = {}
            
            # Create tasks for parallel processing
            tasks = []
            for indicator in indicators:
                task = asyncio.create_task(
                    self.forecast_indicator(indicator, horizon)
                )
                tasks.append((indicator, task))
            
            # Wait for all forecasts to complete
            for indicator, task in tasks:
                try:
                    forecast = await task
                    if forecast:
                        forecasts[indicator] = forecast
                except Exception as e:
                    logger.error(f"Error forecasting {indicator.value}: {e}")
                    continue
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error forecasting multiple indicators: {e}")
            return {}
    
    def _initialize_models(self):
        """Initialize ML models for each indicator"""
        try:
            for indicator in EconomicIndicator:
                self.models[indicator] = {
                    'random_forest': RandomForestRegressor(**self.model_configs['random_forest']),
                    'gradient_boosting': GradientBoostingRegressor(**self.model_configs['gradient_boosting']),
                    'linear': Ridge(**self.model_configs['linear'])
                }
                self.scalers[indicator] = StandardScaler()
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
    
    async def _get_indicator_data(self, indicator: EconomicIndicator,
                                periods: int = 120) -> Optional[pd.Series]:
        """Get historical data for economic indicator"""
        try:
            # In a real implementation, this would fetch from economic data APIs
            # For now, we'll generate synthetic but realistic data
            
            np.random.seed(hash(indicator.value) % 2**32)
            
            # Define indicator characteristics
            indicator_params = {
                EconomicIndicator.GDP_GROWTH: {'mean': 2.5, 'std': 1.2, 'trend': 0.001},
                EconomicIndicator.INFLATION_RATE: {'mean': 2.1, 'std': 1.5, 'trend': 0.002},
                EconomicIndicator.UNEMPLOYMENT_RATE: {'mean': 5.5, 'std': 2.1, 'trend': -0.001},
                EconomicIndicator.INTEREST_RATE: {'mean': 3.8, 'std': 2.5, 'trend': 0.001},
                EconomicIndicator.CONSUMER_CONFIDENCE: {'mean': 95, 'std': 15, 'trend': 0.01},
                EconomicIndicator.INDUSTRIAL_PRODUCTION: {'mean': 102, 'std': 8, 'trend': 0.02},
                EconomicIndicator.YIELD_CURVE_SPREAD: {'mean': 1.2, 'std': 1.1, 'trend': -0.001}
            }
            
            params = indicator_params.get(indicator, {'mean': 100, 'std': 10, 'trend': 0})
            
            # Generate time series with trend and seasonality
            dates = pd.date_range(end=datetime.now(), periods=periods, freq='M')
            
            # Base trend
            trend = np.arange(periods) * params['trend']
            
            # Seasonal component (annual cycle)
            seasonal = 0.1 * params['std'] * np.sin(2 * np.pi * np.arange(periods) / 12)
            
            # Random noise
            noise = np.random.normal(0, params['std'], periods)
            
            # Combine components
            values = params['mean'] + trend + seasonal + noise
            
            # Add some autocorrelation
            for i in range(1, len(values)):
                values[i] += 0.3 * (values[i-1] - params['mean'])
            
            return pd.Series(values, index=dates)
            
        except Exception as e:
            logger.error(f"Error getting indicator data: {e}")
            return None
    
    async def _prepare_features(self, indicator: EconomicIndicator,
                              data: pd.Series) -> Optional[pd.DataFrame]:
        """Prepare features for forecasting model"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Lagged values
            for lag in [1, 2, 3, 6, 12]:
                if len(data) > lag:
                    features[f'lag_{lag}'] = data.shift(lag)
            
            # Moving averages
            for window in [3, 6, 12]:
                if len(data) > window:
                    features[f'ma_{window}'] = data.rolling(window=window).mean()
            
            # Trend features
            features['trend'] = np.arange(len(data))
            features['trend_squared'] = features['trend'] ** 2
            
            # Seasonal features
            features['month'] = data.index.month
            features['quarter'] = data.index.quarter
            
            # Volatility features
            features['volatility_3m'] = data.rolling(window=3).std()
            features['volatility_12m'] = data.rolling(window=12).std()
            
            # Rate of change
            features['pct_change_1m'] = data.pct_change(1)
            features['pct_change_3m'] = data.pct_change(3)
            features['pct_change_12m'] = data.pct_change(12)
            
            # Remove rows with NaN values
            features = features.dropna()
            
            if len(features) < 20:
                return None
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None
    
    async def _ensemble_forecast(self, indicator: EconomicIndicator,
                               features: pd.DataFrame,
                               horizon: ForecastHorizon) -> Optional[Dict[str, Any]]:
        """Generate ensemble forecast using multiple models"""
        try:
            # Prepare target variable
            target_data = await self._get_indicator_data(indicator)
            if target_data is None:
                return None
            
            # Align features and target
            aligned_data = pd.concat([features, target_data.rename('target')], axis=1).dropna()
            
            if len(aligned_data) < 30:
                return None
            
            X = aligned_data.drop('target', axis=1)
            y = aligned_data['target']
            
            # Time series split for validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            model_predictions = {}
            model_scores = {}
            
            # Train and validate each model
            for model_name, model in self.models[indicator].items():
                try:
                    predictions = []
                    scores = []
                    
                    for train_idx, val_idx in tscv.split(X):
                        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_val_scaled = scaler.transform(X_val)
                        
                        # Train model
                        model.fit(X_train_scaled, y_train)
                        
                        # Predict
                        pred = model.predict(X_val_scaled)
                        predictions.extend(pred)
                        
                        # Calculate score
                        score = 1 - mean_absolute_error(y_val, pred) / np.std(y_val)
                        scores.append(max(0, score))  # Ensure non-negative
                    
                    model_predictions[model_name] = np.mean(predictions)
                    model_scores[model_name] = np.mean(scores)
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    continue
            
            if not model_predictions:
                return None
            
            # Calculate ensemble prediction (weighted by scores)
            total_score = sum(model_scores.values())
            if total_score == 0:
                # Equal weights if all scores are zero
                weights = {name: 1/len(model_scores) for name in model_scores}
            else:
                weights = {name: score/total_score for name, score in model_scores.items()}
            
            ensemble_prediction = sum(
                weights[name] * pred for name, pred in model_predictions.items()
            )
            
            # Calculate confidence interval
            prediction_std = np.std(list(model_predictions.values()))
            confidence_interval = (
                ensemble_prediction - 1.96 * prediction_std,
                ensemble_prediction + 1.96 * prediction_std
            )
            
            # Determine confidence level
            avg_score = np.mean(list(model_scores.values()))
            if avg_score > 0.8:
                confidence_level = ForecastConfidence.VERY_HIGH
            elif avg_score > 0.6:
                confidence_level = ForecastConfidence.HIGH
            elif avg_score > 0.4:
                confidence_level = ForecastConfidence.MODERATE
            else:
                confidence_level = ForecastConfidence.LOW
            
            # Identify key drivers
            key_drivers = await self._identify_key_drivers(indicator, features)
            
            # Identify risks
            risks = await self._identify_forecast_risks(indicator, horizon)
            
            return {
                'value': ensemble_prediction,
                'confidence_interval': confidence_interval,
                'confidence_level': confidence_level,
                'methodology': 'Ensemble (RF + GB + Linear)',
                'key_drivers': key_drivers,
                'risks': risks,
                'accuracy': avg_score
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble forecast: {e}")
            return None
    
    async def _single_model_forecast(self, indicator: EconomicIndicator,
                                   features: pd.DataFrame,
                                   horizon: ForecastHorizon) -> Optional[Dict[str, Any]]:
        """Generate forecast using single best model"""
        try:
            # Use Random Forest as default single model
            model = self.models[indicator]['random_forest']
            
            # Prepare data
            target_data = await self._get_indicator_data(indicator)
            if target_data is None:
                return None
            
            aligned_data = pd.concat([features, target_data.rename('target')], axis=1).dropna()
            
            if len(aligned_data) < 20:
                return None
            
            X = aligned_data.drop('target', axis=1)
            y = aligned_data['target']
            
            # Scale features
            scaler = self.scalers[indicator]
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model.fit(X_scaled, y)
            
            # Make prediction (using last available features)
            last_features = X.iloc[-1:]
            last_features_scaled = scaler.transform(last_features)
            prediction = model.predict(last_features_scaled)[0]
            
            # Estimate confidence interval (simplified)
            prediction_std = np.std(y) * 0.2  # Rough estimate
            confidence_interval = (
                prediction - 1.96 * prediction_std,
                prediction + 1.96 * prediction_std
            )
            
            # Calculate model accuracy
            train_pred = model.predict(X_scaled)
            accuracy = 1 - mean_absolute_error(y, train_pred) / np.std(y)
            accuracy = max(0, accuracy)
            
            # Determine confidence level
            if accuracy > 0.8:
                confidence_level = ForecastConfidence.VERY_HIGH
            elif accuracy > 0.6:
                confidence_level = ForecastConfidence.HIGH
            elif accuracy > 0.4:
                confidence_level = ForecastConfidence.MODERATE
            else:
                confidence_level = ForecastConfidence.LOW
            
            # Identify key drivers
            key_drivers = await self._identify_key_drivers(indicator, features)
            
            # Identify risks
            risks = await self._identify_forecast_risks(indicator, horizon)
            
            return {
                'value': prediction,
                'confidence_interval': confidence_interval,
                'confidence_level': confidence_level,
                'methodology': 'Random Forest',
                'key_drivers': key_drivers,
                'risks': risks,
                'accuracy': accuracy
            }
            
        except Exception as e:
            logger.error(f"Error in single model forecast: {e}")
            return None
    
    async def _classify_regime(self, indicator_values: Dict[EconomicIndicator, float]) -> EconomicRegime:
        """Classify current economic regime based on indicators"""
        try:
            # Simple rule-based classification
            gdp_growth = indicator_values.get(EconomicIndicator.GDP_GROWTH, 2.0)
            inflation = indicator_values.get(EconomicIndicator.INFLATION_RATE, 2.0)
            unemployment = indicator_values.get(EconomicIndicator.UNEMPLOYMENT_RATE, 5.0)
            yield_spread = indicator_values.get(EconomicIndicator.YIELD_CURVE_SPREAD, 1.0)
            
            # Recession indicators
            if gdp_growth < 0 or yield_spread < 0 or unemployment > 8:
                if gdp_growth < -2:
                    return EconomicRegime.CONTRACTION
                else:
                    return EconomicRegime.TROUGH
            
            # Stagflation
            if gdp_growth < 1 and inflation > 4:
                return EconomicRegime.STAGFLATION
            
            # Strong growth
            if gdp_growth > 4 and unemployment < 4:
                return EconomicRegime.PEAK
            
            # Recovery
            if gdp_growth > 2 and unemployment > 6:
                return EconomicRegime.RECOVERY
            
            # Default to expansion
            return EconomicRegime.EXPANSION
            
        except Exception as e:
            logger.error(f"Error classifying regime: {e}")
            return EconomicRegime.EXPANSION
    
    async def _calculate_regime_probability(self, regime: EconomicRegime,
                                          indicator_values: Dict[EconomicIndicator, float]) -> float:
        """Calculate probability of current regime classification"""
        try:
            # Simple confidence scoring based on how well indicators align
            confidence_scores = []
            
            gdp_growth = indicator_values.get(EconomicIndicator.GDP_GROWTH, 2.0)
            inflation = indicator_values.get(EconomicIndicator.INFLATION_RATE, 2.0)
            unemployment = indicator_values.get(EconomicIndicator.UNEMPLOYMENT_RATE, 5.0)
            
            if regime == EconomicRegime.EXPANSION:
                confidence_scores.append(min(1.0, max(0.0, gdp_growth / 3.0)))
                confidence_scores.append(min(1.0, max(0.0, (8 - unemployment) / 4.0)))
            elif regime == EconomicRegime.CONTRACTION:
                confidence_scores.append(min(1.0, max(0.0, -gdp_growth / 2.0)))
                confidence_scores.append(min(1.0, max(0.0, (unemployment - 5) / 3.0)))
            elif regime == EconomicRegime.STAGFLATION:
                confidence_scores.append(min(1.0, max(0.0, (4 - gdp_growth) / 4.0)))
                confidence_scores.append(min(1.0, max(0.0, (inflation - 3) / 2.0)))
            
            return np.mean(confidence_scores) if confidence_scores else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating regime probability: {e}")
            return 0.5
    
    async def _estimate_regime_duration(self, regime: EconomicRegime) -> int:
        """Estimate typical duration of economic regime in months"""
        try:
            # Historical average durations
            durations = {
                EconomicRegime.EXPANSION: 58,  # ~5 years
                EconomicRegime.PEAK: 6,        # ~6 months
                EconomicRegime.CONTRACTION: 11, # ~11 months
                EconomicRegime.TROUGH: 4,      # ~4 months
                EconomicRegime.RECOVERY: 18,   # ~1.5 years
                EconomicRegime.STAGFLATION: 24, # ~2 years
                EconomicRegime.DEFLATION: 36   # ~3 years
            }
            
            return durations.get(regime, 12)
            
        except Exception as e:
            logger.error(f"Error estimating regime duration: {e}")
            return 12
    
    async def _get_regime_indicators(self, regime: EconomicRegime) -> List[str]:
        """Get key indicators for specific regime"""
        try:
            indicators = {
                EconomicRegime.EXPANSION: [
                    "GDP Growth", "Employment Growth", "Consumer Confidence", "Business Investment"
                ],
                EconomicRegime.CONTRACTION: [
                    "GDP Decline", "Rising Unemployment", "Falling Industrial Production", "Credit Contraction"
                ],
                EconomicRegime.STAGFLATION: [
                    "High Inflation", "Low Growth", "Rising Commodity Prices", "Wage Pressures"
                ],
                EconomicRegime.RECOVERY: [
                    "Improving Employment", "Rising Consumer Spending", "Inventory Rebuilding", "Credit Expansion"
                ]
            }
            
            return indicators.get(regime, ["Economic Activity", "Price Levels", "Employment"])
            
        except Exception as e:
            logger.error(f"Error getting regime indicators: {e}")
            return []
    
    async def _get_regime_characteristics(self, regime: EconomicRegime,
                                        indicator_values: Dict[EconomicIndicator, float]) -> Dict[str, float]:
        """Get characteristics of current regime"""
        try:
            characteristics = {
                'growth_rate': indicator_values.get(EconomicIndicator.GDP_GROWTH, 2.0),
                'inflation_rate': indicator_values.get(EconomicIndicator.INFLATION_RATE, 2.0),
                'unemployment_rate': indicator_values.get(EconomicIndicator.UNEMPLOYMENT_RATE, 5.0),
                'interest_rate': indicator_values.get(EconomicIndicator.INTEREST_RATE, 3.0),
                'volatility_index': 0.15,  # Placeholder
                'credit_conditions': 0.5   # Placeholder (0=tight, 1=loose)
            }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Error getting regime characteristics: {e}")
            return {}
    
    async def _generate_scenario_forecasts(self, scenario_name: str) -> Dict[EconomicIndicator, float]:
        """Generate indicator forecasts for specific scenario"""
        try:
            # Define scenario parameters
            scenario_params = {
                'Baseline Growth': {
                    EconomicIndicator.GDP_GROWTH: 2.5,
                    EconomicIndicator.INFLATION_RATE: 2.2,
                    EconomicIndicator.UNEMPLOYMENT_RATE: 4.5,
                    EconomicIndicator.INTEREST_RATE: 4.0
                },
                'Accelerated Growth': {
                    EconomicIndicator.GDP_GROWTH: 4.0,
                    EconomicIndicator.INFLATION_RATE: 3.5,
                    EconomicIndicator.UNEMPLOYMENT_RATE: 3.5,
                    EconomicIndicator.INTEREST_RATE: 5.5
                },
                'Economic Slowdown': {
                    EconomicIndicator.GDP_GROWTH: 0.5,
                    EconomicIndicator.INFLATION_RATE: 1.5,
                    EconomicIndicator.UNEMPLOYMENT_RATE: 6.5,
                    EconomicIndicator.INTEREST_RATE: 2.5
                },
                'Stagflation': {
                    EconomicIndicator.GDP_GROWTH: 0.8,
                    EconomicIndicator.INFLATION_RATE: 5.0,
                    EconomicIndicator.UNEMPLOYMENT_RATE: 7.0,
                    EconomicIndicator.INTEREST_RATE: 6.0
                },
                'Deflationary Spiral': {
                    EconomicIndicator.GDP_GROWTH: -2.0,
                    EconomicIndicator.INFLATION_RATE: -1.0,
                    EconomicIndicator.UNEMPLOYMENT_RATE: 9.0,
                    EconomicIndicator.INTEREST_RATE: 0.5
                }
            }
            
            return scenario_params.get(scenario_name, {})
            
        except Exception as e:
            logger.error(f"Error generating scenario forecasts: {e}")
            return {}
    
    async def _calculate_market_implications(self, indicator_forecasts: Dict[EconomicIndicator, float]) -> Dict[str, float]:
        """Calculate market implications of economic forecasts"""
        try:
            implications = {}
            
            gdp_growth = indicator_forecasts.get(EconomicIndicator.GDP_GROWTH, 2.0)
            inflation = indicator_forecasts.get(EconomicIndicator.INFLATION_RATE, 2.0)
            interest_rate = indicator_forecasts.get(EconomicIndicator.INTEREST_RATE, 3.0)
            
            # Stock market implications
            implications['stock_market_return'] = gdp_growth * 2.5 - inflation * 0.5
            
            # Bond market implications
            implications['bond_market_return'] = -interest_rate * 0.8 + inflation * 0.3
            
            # Currency implications
            implications['currency_strength'] = interest_rate * 0.4 - inflation * 0.2
            
            # Commodity implications
            implications['commodity_return'] = inflation * 1.2 + gdp_growth * 0.5
            
            # Real estate implications
            implications['real_estate_return'] = gdp_growth * 1.5 - interest_rate * 0.8
            
            return implications
            
        except Exception as e:
            logger.error(f"Error calculating market implications: {e}")
            return {}
    
    async def _generate_policy_implications(self, scenario_name: str,
                                          indicator_forecasts: Dict[EconomicIndicator, float]) -> List[str]:
        """Generate policy implications for scenario"""
        try:
            implications = []
            
            gdp_growth = indicator_forecasts.get(EconomicIndicator.GDP_GROWTH, 2.0)
            inflation = indicator_forecasts.get(EconomicIndicator.INFLATION_RATE, 2.0)
            unemployment = indicator_forecasts.get(EconomicIndicator.UNEMPLOYMENT_RATE, 5.0)
            
            # Monetary policy implications
            if inflation > 3.0:
                implications.append("Consider tightening monetary policy to combat inflation")
            elif inflation < 1.0:
                implications.append("Consider loosening monetary policy to support growth")
            
            # Fiscal policy implications
            if gdp_growth < 1.0:
                implications.append("Consider fiscal stimulus to support economic growth")
            elif gdp_growth > 4.0:
                implications.append("Consider fiscal restraint to prevent overheating")
            
            # Employment policy implications
            if unemployment > 7.0:
                implications.append("Implement job creation and training programs")
            
            # Regulatory implications
            if scenario_name == 'Stagflation':
                implications.append("Review supply chain regulations and trade policies")
            
            return implications[:5]  # Return top 5 implications
            
        except Exception as e:
            logger.error(f"Error generating policy implications: {e}")
            return []
    
    async def _identify_key_drivers(self, indicator: EconomicIndicator,
                                  features: pd.DataFrame) -> List[str]:
        """Identify key drivers for indicator forecast"""
        try:
            # Generic key drivers based on indicator type
            drivers = {
                EconomicIndicator.GDP_GROWTH: [
                    "Consumer spending", "Business investment", "Government spending", "Net exports"
                ],
                EconomicIndicator.INFLATION_RATE: [
                    "Energy prices", "Labor costs", "Supply chain conditions", "Monetary policy"
                ],
                EconomicIndicator.UNEMPLOYMENT_RATE: [
                    "Economic growth", "Business confidence", "Labor market policies", "Automation trends"
                ],
                EconomicIndicator.INTEREST_RATE: [
                    "Inflation expectations", "Economic growth", "Central bank policy", "Global rates"
                ]
            }
            
            return drivers.get(indicator, ["Economic conditions", "Market sentiment", "Policy changes"])
            
        except Exception as e:
            logger.error(f"Error identifying key drivers: {e}")
            return []
    
    async def _identify_forecast_risks(self, indicator: EconomicIndicator,
                                     horizon: ForecastHorizon) -> List[str]:
        """Identify risks to forecast accuracy"""
        try:
            # Generic risks based on horizon and indicator
            risks = []
            
            if horizon == ForecastHorizon.LONG_TERM:
                risks.extend(["Structural economic changes", "Policy regime shifts", "Technological disruption"])
            
            if indicator in [EconomicIndicator.INFLATION_RATE, EconomicIndicator.INTEREST_RATE]:
                risks.extend(["Central bank policy surprises", "Global economic shocks"])
            
            if indicator == EconomicIndicator.GDP_GROWTH:
                risks.extend(["Geopolitical events", "Financial market volatility", "Trade policy changes"])
            
            # Add general risks
            risks.extend(["Data revisions", "Model uncertainty", "Unforeseen events"])
            
            return risks[:5]  # Return top 5 risks
            
        except Exception as e:
            logger.error(f"Error identifying forecast risks: {e}")
            return []

# Export main classes
__all__ = ['EconomicForecaster', 'EconomicForecast', 'EconomicIndicator', 'ForecastHorizon',
           'RegimeAnalysis', 'EconomicScenario', 'EconomicRegime', 'ForecastConfidence']