import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ForexFeatureCategory(Enum):
    PRICE_ACTION = "price_action"
    TECHNICAL = "technical"
    ECONOMIC = "economic"
    CARRY_TRADE = "carry_trade"
    SENTIMENT = "sentiment"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    INTERVENTION = "intervention"
    SEASONALITY = "seasonality"
    FLOW = "flow"
    RISK = "risk"
    MOMENTUM = "momentum"

class EconomicIndicator(Enum):
    GDP_GROWTH = "gdp_growth"
    INFLATION_RATE = "inflation_rate"
    UNEMPLOYMENT_RATE = "unemployment_rate"
    INTEREST_RATE = "interest_rate"
    CURRENT_ACCOUNT = "current_account"
    TRADE_BALANCE = "trade_balance"
    GOVERNMENT_DEBT = "government_debt"
    PMI_MANUFACTURING = "pmi_manufacturing"
    PMI_SERVICES = "pmi_services"
    CONSUMER_CONFIDENCE = "consumer_confidence"
    RETAIL_SALES = "retail_sales"
    INDUSTRIAL_PRODUCTION = "industrial_production"
    MONEY_SUPPLY = "money_supply"
    FOREIGN_RESERVES = "foreign_reserves"
    CREDIT_RATING = "credit_rating"

@dataclass
class ForexFeatureSet:
    """Comprehensive feature set for forex analysis"""
    timestamp: datetime
    currency_pair: str
    
    # Price action features
    price_features: Dict[str, float] = field(default_factory=dict)
    
    # Technical indicators
    technical_features: Dict[str, float] = field(default_factory=dict)
    
    # Economic indicators
    economic_features: Dict[str, float] = field(default_factory=dict)
    
    # Carry trade features
    carry_trade_features: Dict[str, float] = field(default_factory=dict)
    
    # Sentiment features
    sentiment_features: Dict[str, float] = field(default_factory=dict)
    
    # Volatility features
    volatility_features: Dict[str, float] = field(default_factory=dict)
    
    # Correlation features
    correlation_features: Dict[str, float] = field(default_factory=dict)
    
    # Intervention features
    intervention_features: Dict[str, float] = field(default_factory=dict)
    
    # Seasonality features
    seasonality_features: Dict[str, float] = field(default_factory=dict)
    
    # Flow features
    flow_features: Dict[str, float] = field(default_factory=dict)
    
    # Risk features
    risk_features: Dict[str, float] = field(default_factory=dict)
    
    # Momentum features
    momentum_features: Dict[str, float] = field(default_factory=dict)
    
    # Feature importance scores
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    # Feature quality metrics
    feature_quality: Dict[str, float] = field(default_factory=dict)
    
    def get_all_features(self) -> Dict[str, float]:
        """Get all features as a single dictionary"""
        all_features = {}
        all_features.update(self.price_features)
        all_features.update(self.technical_features)
        all_features.update(self.economic_features)
        all_features.update(self.carry_trade_features)
        all_features.update(self.sentiment_features)
        all_features.update(self.volatility_features)
        all_features.update(self.correlation_features)
        all_features.update(self.intervention_features)
        all_features.update(self.seasonality_features)
        all_features.update(self.flow_features)
        all_features.update(self.risk_features)
        all_features.update(self.momentum_features)
        return all_features
    
    def get_features_by_category(self, category: ForexFeatureCategory) -> Dict[str, float]:
        """Get features by category"""
        category_map = {
            ForexFeatureCategory.PRICE_ACTION: self.price_features,
            ForexFeatureCategory.TECHNICAL: self.technical_features,
            ForexFeatureCategory.ECONOMIC: self.economic_features,
            ForexFeatureCategory.CARRY_TRADE: self.carry_trade_features,
            ForexFeatureCategory.SENTIMENT: self.sentiment_features,
            ForexFeatureCategory.VOLATILITY: self.volatility_features,
            ForexFeatureCategory.CORRELATION: self.correlation_features,
            ForexFeatureCategory.INTERVENTION: self.intervention_features,
            ForexFeatureCategory.SEASONALITY: self.seasonality_features,
            ForexFeatureCategory.FLOW: self.flow_features,
            ForexFeatureCategory.RISK: self.risk_features,
            ForexFeatureCategory.MOMENTUM: self.momentum_features
        }
        return category_map.get(category, {})

class ForexPriceActionFeatures:
    """Extract price action features for forex"""
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize price action feature extractor
        
        Args:
            lookback_periods: List of periods to calculate features over
        """
        self.lookback_periods = lookback_periods or [5, 10, 20, 50, 100]
        logger.info(f"Initialized ForexPriceActionFeatures with periods: {self.lookback_periods}")
    
    def extract_features(self, prices: List[float], volumes: List[float] = None) -> Dict[str, float]:
        """Extract price action features"""
        try:
            features = {}
            
            if len(prices) < max(self.lookback_periods):
                logger.warning(f"Insufficient price data: {len(prices)} < {max(self.lookback_periods)}")
                return features
            
            prices_array = np.array(prices)
            
            # Current price metrics
            current_price = prices_array[-1]
            features['current_price'] = current_price
            
            # Returns for different periods
            for period in self.lookback_periods:
                if len(prices) > period:
                    # Simple return
                    period_return = (current_price - prices_array[-period-1]) / prices_array[-period-1]
                    features[f'return_{period}d'] = period_return
                    
                    # Log return
                    log_return = np.log(current_price / prices_array[-period-1])
                    features[f'log_return_{period}d'] = log_return
                    
                    # Volatility (rolling standard deviation of returns)
                    if len(prices) > period + 1:
                        period_prices = prices_array[-period-1:]
                        period_returns = np.diff(period_prices) / period_prices[:-1]
                        features[f'volatility_{period}d'] = np.std(period_returns)
                        features[f'mean_return_{period}d'] = np.mean(period_returns)
                    
                    # Price position relative to period high/low
                    period_high = np.max(prices_array[-period:])
                    period_low = np.min(prices_array[-period:])
                    if period_high > period_low:
                        features[f'price_position_{period}d'] = (current_price - period_low) / (period_high - period_low)
                    
                    # Distance from moving average
                    period_ma = np.mean(prices_array[-period:])
                    features[f'ma_distance_{period}d'] = (current_price - period_ma) / period_ma
            
            # Price momentum and acceleration
            if len(prices) >= 3:
                # First derivative (velocity)
                velocity = prices_array[-1] - prices_array[-2]
                features['price_velocity'] = velocity / prices_array[-2]
                
                # Second derivative (acceleration)
                if len(prices) >= 4:
                    prev_velocity = prices_array[-2] - prices_array[-3]
                    acceleration = velocity - prev_velocity
                    features['price_acceleration'] = acceleration / prices_array[-3]
            
            # Support and resistance levels
            if len(prices) >= 20:
                recent_prices = prices_array[-20:]
                
                # Local maxima and minima
                local_maxima = []
                local_minima = []
                
                for i in range(1, len(recent_prices) - 1):
                    if recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i+1]:
                        local_maxima.append(recent_prices[i])
                    elif recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i+1]:
                        local_minima.append(recent_prices[i])
                
                if local_maxima:
                    resistance_level = np.mean(local_maxima)
                    features['resistance_distance'] = (current_price - resistance_level) / resistance_level
                
                if local_minima:
                    support_level = np.mean(local_minima)
                    features['support_distance'] = (current_price - support_level) / support_level
            
            # Price patterns
            if len(prices) >= 10:
                recent_prices = prices_array[-10:]
                
                # Trend strength
                x = np.arange(len(recent_prices))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_prices)
                features['trend_slope'] = slope / np.mean(recent_prices)
                features['trend_strength'] = r_value ** 2
                features['trend_significance'] = 1 - p_value if p_value < 1 else 0
            
            # Volume-weighted features (if volume available)
            if volumes and len(volumes) == len(prices):
                volumes_array = np.array(volumes)
                
                # Volume-weighted average price (VWAP)
                if len(prices) >= 20:
                    recent_prices = prices_array[-20:]
                    recent_volumes = volumes_array[-20:]
                    if np.sum(recent_volumes) > 0:
                        vwap = np.sum(recent_prices * recent_volumes) / np.sum(recent_volumes)
                        features['vwap_distance'] = (current_price - vwap) / vwap
                
                # Volume trend
                if len(volumes) >= 10:
                    recent_volumes = volumes_array[-10:]
                    volume_trend = (recent_volumes[-1] - np.mean(recent_volumes[:-1])) / np.mean(recent_volumes[:-1])
                    features['volume_trend'] = volume_trend
            
            # Price gaps
            if len(prices) >= 2:
                gap = (prices_array[-1] - prices_array[-2]) / prices_array[-2]
                features['price_gap'] = gap
                features['gap_magnitude'] = abs(gap)
            
            # Fractal dimension (complexity measure)
            if len(prices) >= 50:
                recent_prices = prices_array[-50:]
                features['fractal_dimension'] = self._calculate_fractal_dimension(recent_prices)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting price action features: {e}")
            return {}
    
    def _calculate_fractal_dimension(self, prices: np.ndarray) -> float:
        """Calculate fractal dimension using box-counting method"""
        try:
            # Normalize prices
            normalized_prices = (prices - np.min(prices)) / (np.max(prices) - np.min(prices))
            
            # Simple fractal dimension approximation
            n = len(normalized_prices)
            if n < 10:
                return 1.5  # Default value
            
            # Calculate path length
            path_length = np.sum(np.abs(np.diff(normalized_prices)))
            
            # Fractal dimension approximation
            if path_length > 0:
                fractal_dim = 1 + np.log(path_length) / np.log(n)
                return max(1.0, min(2.0, fractal_dim))  # Clamp between 1 and 2
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"Error calculating fractal dimension: {e}")
            return 1.5

class ForexEconomicFeatures:
    """Extract economic indicator features for forex"""
    
    def __init__(self, base_currency: str = "USD", quote_currency: str = "EUR"):
        """
        Initialize economic feature extractor
        
        Args:
            base_currency: Base currency code
            quote_currency: Quote currency code
        """
        self.base_currency = base_currency
        self.quote_currency = quote_currency
        logger.info(f"Initialized ForexEconomicFeatures for {base_currency}/{quote_currency}")
    
    def extract_features(self, base_economic_data: Dict[str, float], 
                        quote_economic_data: Dict[str, float],
                        global_data: Dict[str, float] = None) -> Dict[str, float]:
        """Extract economic features"""
        try:
            features = {}
            
            if global_data is None:
                global_data = {}
            
            # Interest rate differential (most important for forex)
            base_rate = base_economic_data.get('interest_rate', 0.0)
            quote_rate = quote_economic_data.get('interest_rate', 0.0)
            features['interest_rate_differential'] = base_rate - quote_rate
            features['interest_rate_ratio'] = base_rate / (quote_rate + 0.01) if quote_rate > -0.01 else 0
            
            # Real interest rate differential
            base_inflation = base_economic_data.get('inflation_rate', 0.0)
            quote_inflation = quote_economic_data.get('inflation_rate', 0.0)
            base_real_rate = base_rate - base_inflation
            quote_real_rate = quote_rate - quote_inflation
            features['real_rate_differential'] = base_real_rate - quote_real_rate
            
            # Economic growth differential
            base_gdp = base_economic_data.get('gdp_growth', 0.0)
            quote_gdp = quote_economic_data.get('gdp_growth', 0.0)
            features['gdp_growth_differential'] = base_gdp - quote_gdp
            features['gdp_growth_ratio'] = (base_gdp + 2) / (quote_gdp + 2)  # Add 2 to avoid negative issues
            
            # Inflation differential
            features['inflation_differential'] = base_inflation - quote_inflation
            
            # Unemployment differential
            base_unemployment = base_economic_data.get('unemployment_rate', 0.0)
            quote_unemployment = quote_economic_data.get('unemployment_rate', 0.0)
            features['unemployment_differential'] = quote_unemployment - base_unemployment  # Lower is better
            
            # Current account balance differential
            base_current_account = base_economic_data.get('current_account', 0.0)
            quote_current_account = quote_economic_data.get('current_account', 0.0)
            features['current_account_differential'] = base_current_account - quote_current_account
            
            # Trade balance differential
            base_trade_balance = base_economic_data.get('trade_balance', 0.0)
            quote_trade_balance = quote_economic_data.get('trade_balance', 0.0)
            features['trade_balance_differential'] = base_trade_balance - quote_trade_balance
            
            # Government debt differential (lower is better)
            base_debt = base_economic_data.get('government_debt', 0.0)
            quote_debt = quote_economic_data.get('government_debt', 0.0)
            features['debt_differential'] = quote_debt - base_debt  # Lower debt is better
            
            # PMI differentials
            base_pmi_mfg = base_economic_data.get('pmi_manufacturing', 50.0)
            quote_pmi_mfg = quote_economic_data.get('pmi_manufacturing', 50.0)
            features['pmi_manufacturing_differential'] = base_pmi_mfg - quote_pmi_mfg
            
            base_pmi_svc = base_economic_data.get('pmi_services', 50.0)
            quote_pmi_svc = quote_economic_data.get('pmi_services', 50.0)
            features['pmi_services_differential'] = base_pmi_svc - quote_pmi_svc
            
            # Consumer confidence differential
            base_confidence = base_economic_data.get('consumer_confidence', 0.0)
            quote_confidence = quote_economic_data.get('consumer_confidence', 0.0)
            features['consumer_confidence_differential'] = base_confidence - quote_confidence
            
            # Retail sales differential
            base_retail = base_economic_data.get('retail_sales', 0.0)
            quote_retail = quote_economic_data.get('retail_sales', 0.0)
            features['retail_sales_differential'] = base_retail - quote_retail
            
            # Industrial production differential
            base_industrial = base_economic_data.get('industrial_production', 0.0)
            quote_industrial = quote_economic_data.get('industrial_production', 0.0)
            features['industrial_production_differential'] = base_industrial - quote_industrial
            
            # Money supply growth differential
            base_money_supply = base_economic_data.get('money_supply', 0.0)
            quote_money_supply = quote_economic_data.get('money_supply', 0.0)
            features['money_supply_differential'] = base_money_supply - quote_money_supply
            
            # Foreign reserves differential
            base_reserves = base_economic_data.get('foreign_reserves', 0.0)
            quote_reserves = quote_economic_data.get('foreign_reserves', 0.0)
            features['foreign_reserves_differential'] = base_reserves - quote_reserves
            
            # Credit rating differential
            base_rating = base_economic_data.get('credit_rating', 0.0)
            quote_rating = quote_economic_data.get('credit_rating', 0.0)
            features['credit_rating_differential'] = base_rating - quote_rating
            
            # Economic surprise indices
            base_surprise = base_economic_data.get('economic_surprise_index', 0.0)
            quote_surprise = quote_economic_data.get('economic_surprise_index', 0.0)
            features['economic_surprise_differential'] = base_surprise - quote_surprise
            
            # Global risk factors
            features['global_risk_appetite'] = global_data.get('risk_appetite', 0.0)
            features['global_liquidity'] = global_data.get('global_liquidity', 0.0)
            features['commodity_prices'] = global_data.get('commodity_index', 0.0)
            features['global_growth'] = global_data.get('global_gdp_growth', 0.0)
            
            # VIX and risk-off indicators
            features['vix_level'] = global_data.get('vix', 20.0)
            features['risk_off_indicator'] = 1.0 if global_data.get('vix', 20.0) > 25.0 else 0.0
            
            # Central bank policy indicators
            features['base_cb_hawkish'] = base_economic_data.get('central_bank_hawkishness', 0.0)
            features['quote_cb_hawkish'] = quote_economic_data.get('central_bank_hawkishness', 0.0)
            features['cb_policy_divergence'] = features['base_cb_hawkish'] - features['quote_cb_hawkish']
            
            # Economic momentum indicators
            base_momentum = self._calculate_economic_momentum(base_economic_data)
            quote_momentum = self._calculate_economic_momentum(quote_economic_data)
            features['economic_momentum_differential'] = base_momentum - quote_momentum
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting economic features: {e}")
            return {}
    
    def _calculate_economic_momentum(self, economic_data: Dict[str, float]) -> float:
        """Calculate overall economic momentum score"""
        try:
            momentum_indicators = {
                'gdp_growth': 0.25,
                'pmi_manufacturing': 0.15,
                'pmi_services': 0.15,
                'consumer_confidence': 0.10,
                'retail_sales': 0.10,
                'industrial_production': 0.10,
                'unemployment_rate': -0.15  # Negative weight (lower is better)
            }
            
            momentum_score = 0.0
            total_weight = 0.0
            
            for indicator, weight in momentum_indicators.items():
                if indicator in economic_data:
                    value = economic_data[indicator]
                    
                    # Normalize values
                    if indicator == 'unemployment_rate':
                        normalized_value = max(0, 10 - value) / 10  # Invert unemployment
                    elif indicator in ['pmi_manufacturing', 'pmi_services']:
                        normalized_value = (value - 50) / 50  # PMI around 50
                    else:
                        normalized_value = max(-1, min(1, value / 5))  # General normalization
                    
                    momentum_score += weight * normalized_value
                    total_weight += abs(weight)
            
            return momentum_score / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating economic momentum: {e}")
            return 0.0

class ForexCarryTradeFeatures:
    """Extract carry trade related features"""
    
    def __init__(self):
        logger.info("Initialized ForexCarryTradeFeatures")
    
    def extract_features(self, base_rate: float, quote_rate: float,
                        base_volatility: float, quote_volatility: float,
                        correlation_with_risk: float = 0.0,
                        funding_costs: Dict[str, float] = None) -> Dict[str, float]:
        """Extract carry trade features"""
        try:
            features = {}
            
            if funding_costs is None:
                funding_costs = {}
            
            # Basic carry trade metrics
            carry = base_rate - quote_rate
            features['carry_rate'] = carry
            features['carry_absolute'] = abs(carry)
            
            # Risk-adjusted carry
            avg_volatility = (base_volatility + quote_volatility) / 2
            if avg_volatility > 0:
                features['risk_adjusted_carry'] = carry / avg_volatility
                features['carry_to_vol_ratio'] = abs(carry) / avg_volatility
            else:
                features['risk_adjusted_carry'] = 0.0
                features['carry_to_vol_ratio'] = 0.0
            
            # Carry trade attractiveness score
            # Higher positive carry with lower volatility is more attractive
            if avg_volatility > 0:
                attractiveness = carry / (1 + avg_volatility)
                features['carry_attractiveness'] = attractiveness
            else:
                features['carry_attractiveness'] = carry
            
            # Funding currency characteristics
            features['funding_rate'] = min(base_rate, quote_rate)
            features['target_rate'] = max(base_rate, quote_rate)
            features['rate_spread'] = features['target_rate'] - features['funding_rate']
            
            # Volatility differential
            features['volatility_differential'] = base_volatility - quote_volatility
            features['volatility_ratio'] = base_volatility / (quote_volatility + 0.001)
            
            # Risk correlation adjustment
            features['risk_correlation'] = correlation_with_risk
            features['risk_adjusted_attractiveness'] = features['carry_attractiveness'] * (1 - abs(correlation_with_risk))
            
            # Funding costs
            base_funding_cost = funding_costs.get('base_currency', 0.0)
            quote_funding_cost = funding_costs.get('quote_currency', 0.0)
            features['funding_cost_differential'] = base_funding_cost - quote_funding_cost
            
            # Net carry after funding costs
            net_carry = carry - abs(base_funding_cost - quote_funding_cost)
            features['net_carry'] = net_carry
            
            # Carry trade sustainability indicators
            # High carry with stable rates is more sustainable
            rate_stability = 1.0 / (1.0 + abs(base_rate - quote_rate) * 0.1)
            features['carry_sustainability'] = carry * rate_stability
            
            # Carry trade risk indicators
            features['carry_risk_high'] = 1.0 if abs(carry) > 0.03 and avg_volatility > 0.015 else 0.0
            features['carry_risk_medium'] = 1.0 if abs(carry) > 0.02 and avg_volatility > 0.01 else 0.0
            
            # Optimal carry trade conditions
            optimal_conditions = (
                abs(carry) > 0.015 and  # Sufficient carry
                avg_volatility < 0.012 and  # Low volatility
                abs(correlation_with_risk) < 0.5  # Low risk correlation
            )
            features['optimal_carry_conditions'] = 1.0 if optimal_conditions else 0.0
            
            # Carry trade momentum
            # This would typically use historical data, simplified here
            features['carry_momentum'] = carry * (1 + features['carry_attractiveness'])
            
            # Currency strength indicators for carry trade
            features['base_currency_strength'] = base_rate - avg_volatility
            features['quote_currency_strength'] = quote_rate - quote_volatility
            features['relative_currency_strength'] = features['base_currency_strength'] - features['quote_currency_strength']
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting carry trade features: {e}")
            return {}

class ForexInterventionFeatures:
    """Extract central bank intervention related features"""
    
    def __init__(self):
        logger.info("Initialized ForexInterventionFeatures")
    
    def extract_features(self, current_price: float, historical_prices: List[float],
                        central_bank_data: Dict[str, Any] = None,
                        economic_data: Dict[str, float] = None) -> Dict[str, float]:
        """Extract intervention-related features"""
        try:
            features = {}
            
            if central_bank_data is None:
                central_bank_data = {}
            if economic_data is None:
                economic_data = {}
            
            # Price deviation from historical norms
            if len(historical_prices) >= 100:
                long_term_mean = np.mean(historical_prices[-100:])
                long_term_std = np.std(historical_prices[-100:])
                
                if long_term_std > 0:
                    z_score = (current_price - long_term_mean) / long_term_std
                    features['price_z_score'] = z_score
                    features['extreme_deviation'] = 1.0 if abs(z_score) > 2.0 else 0.0
                    features['moderate_deviation'] = 1.0 if abs(z_score) > 1.5 else 0.0
                else:
                    features['price_z_score'] = 0.0
                    features['extreme_deviation'] = 0.0
                    features['moderate_deviation'] = 0.0
            
            # Rapid price movement indicators
            if len(historical_prices) >= 5:
                recent_prices = historical_prices[-5:]
                price_change_5d = (current_price - recent_prices[0]) / recent_prices[0]
                features['price_change_5d'] = price_change_5d
                features['rapid_appreciation'] = 1.0 if price_change_5d > 0.05 else 0.0
                features['rapid_depreciation'] = 1.0 if price_change_5d < -0.05 else 0.0
            
            # Volatility spike indicators
            if len(historical_prices) >= 20:
                recent_returns = np.diff(historical_prices[-20:]) / historical_prices[-20:-1]
                current_volatility = np.std(recent_returns)
                
                if len(historical_prices) >= 100:
                    historical_returns = np.diff(historical_prices[-100:-20]) / historical_prices[-100:-21]
                    historical_volatility = np.std(historical_returns)
                    
                    if historical_volatility > 0:
                        volatility_ratio = current_volatility / historical_volatility
                        features['volatility_ratio'] = volatility_ratio
                        features['volatility_spike'] = 1.0 if volatility_ratio > 2.0 else 0.0
                    else:
                        features['volatility_ratio'] = 1.0
                        features['volatility_spike'] = 0.0
            
            # Central bank communication indicators
            features['cb_verbal_intervention'] = central_bank_data.get('verbal_intervention_score', 0.0)
            features['cb_hawkish_tone'] = central_bank_data.get('hawkish_tone', 0.0)
            features['cb_dovish_tone'] = central_bank_data.get('dovish_tone', 0.0)
            features['cb_intervention_threat'] = central_bank_data.get('intervention_threat', 0.0)
            
            # Policy divergence indicators
            features['policy_divergence'] = central_bank_data.get('policy_divergence_score', 0.0)
            features['rate_decision_surprise'] = central_bank_data.get('rate_decision_surprise', 0.0)
            
            # Economic imbalance indicators
            current_account = economic_data.get('current_account_balance', 0.0)
            trade_balance = economic_data.get('trade_balance', 0.0)
            
            features['current_account_deficit'] = 1.0 if current_account < -0.03 else 0.0
            features['trade_deficit'] = 1.0 if trade_balance < -0.02 else 0.0
            features['twin_deficit'] = 1.0 if (current_account < -0.03 and trade_balance < -0.02) else 0.0
            
            # Foreign reserves indicators
            foreign_reserves = economic_data.get('foreign_reserves_months', 0.0)
            features['low_reserves'] = 1.0 if foreign_reserves < 3.0 else 0.0
            features['adequate_reserves'] = 1.0 if foreign_reserves > 6.0 else 0.0
            
            # Intervention probability model (simplified)
            intervention_factors = [
                features.get('extreme_deviation', 0.0) * 0.3,
                features.get('rapid_appreciation', 0.0) * 0.2,
                features.get('rapid_depreciation', 0.0) * 0.2,
                features.get('volatility_spike', 0.0) * 0.15,
                features.get('cb_intervention_threat', 0.0) * 0.15
            ]
            
            intervention_probability = min(1.0, sum(intervention_factors))
            features['intervention_probability'] = intervention_probability
            
            # Intervention effectiveness indicators
            features['market_depth'] = central_bank_data.get('market_depth_score', 0.5)
            features['intervention_history'] = central_bank_data.get('historical_success_rate', 0.5)
            
            # Coordination indicators
            features['g7_coordination'] = central_bank_data.get('g7_coordination_probability', 0.0)
            features['bilateral_coordination'] = central_bank_data.get('bilateral_coordination', 0.0)
            
            # Market positioning indicators (contrarian to intervention)
            features['speculative_positioning'] = central_bank_data.get('speculative_net_position', 0.0)
            features['commercial_positioning'] = central_bank_data.get('commercial_net_position', 0.0)
            
            # Intervention timing indicators
            features['month_end_effect'] = 1.0 if datetime.now().day > 25 else 0.0
            features['quarter_end_effect'] = 1.0 if (datetime.now().month % 3 == 0 and datetime.now().day > 25) else 0.0
            
            # Political pressure indicators
            features['political_pressure'] = central_bank_data.get('political_pressure_score', 0.0)
            features['election_proximity'] = central_bank_data.get('election_proximity_months', 12.0) / 12.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting intervention features: {e}")
            return {}

class ForexSeasonalityFeatures:
    """Extract seasonality and calendar effect features"""
    
    def __init__(self):
        logger.info("Initialized ForexSeasonalityFeatures")
    
    def extract_features(self, timestamp: datetime, currency_pair: str) -> Dict[str, float]:
        """Extract seasonality features"""
        try:
            features = {}
            
            # Time-based features
            features['month'] = timestamp.month
            features['quarter'] = (timestamp.month - 1) // 3 + 1
            features['day_of_month'] = timestamp.day
            features['day_of_week'] = timestamp.weekday()  # 0 = Monday
            features['hour'] = timestamp.hour
            
            # Cyclical encoding for time features
            features['month_sin'] = np.sin(2 * np.pi * timestamp.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * timestamp.month / 12)
            features['day_sin'] = np.sin(2 * np.pi * timestamp.day / 31)
            features['day_cos'] = np.cos(2 * np.pi * timestamp.day / 31)
            features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
            
            # Trading session indicators
            features['asian_session'] = 1.0 if 0 <= timestamp.hour < 9 else 0.0
            features['european_session'] = 1.0 if 7 <= timestamp.hour < 16 else 0.0
            features['american_session'] = 1.0 if 13 <= timestamp.hour < 22 else 0.0
            features['session_overlap_eur_us'] = 1.0 if 13 <= timestamp.hour < 16 else 0.0
            features['session_overlap_asia_eur'] = 1.0 if 7 <= timestamp.hour < 9 else 0.0
            
            # Weekend and holiday effects
            features['is_weekend'] = 1.0 if timestamp.weekday() >= 5 else 0.0
            features['is_monday'] = 1.0 if timestamp.weekday() == 0 else 0.0
            features['is_friday'] = 1.0 if timestamp.weekday() == 4 else 0.0
            
            # Month-end and quarter-end effects
            features['month_end'] = 1.0 if timestamp.day > 25 else 0.0
            features['quarter_end'] = 1.0 if (timestamp.month % 3 == 0 and timestamp.day > 25) else 0.0
            features['year_end'] = 1.0 if (timestamp.month == 12 and timestamp.day > 25) else 0.0
            
            # Seasonal patterns by currency pair
            seasonal_patterns = self._get_seasonal_patterns(currency_pair, timestamp)
            features.update(seasonal_patterns)
            
            # Holiday calendar effects
            holiday_effects = self._get_holiday_effects(timestamp, currency_pair)
            features.update(holiday_effects)
            
            # Economic calendar effects
            features['first_friday'] = 1.0 if (timestamp.weekday() == 4 and 1 <= timestamp.day <= 7) else 0.0  # NFP
            features['fomc_week'] = self._is_fomc_week(timestamp)
            features['ecb_week'] = self._is_ecb_week(timestamp)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting seasonality features: {e}")
            return {}
    
    def _get_seasonal_patterns(self, currency_pair: str, timestamp: datetime) -> Dict[str, float]:
        """Get currency-specific seasonal patterns"""
        patterns = {}
        
        try:
            # USD strength patterns
            if 'USD' in currency_pair:
                # USD tends to be stronger in Q4
                patterns['usd_q4_strength'] = 1.0 if timestamp.month >= 10 else 0.0
                # USD weakness in summer months
                patterns['usd_summer_weakness'] = 1.0 if 6 <= timestamp.month <= 8 else 0.0
            
            # EUR patterns
            if 'EUR' in currency_pair:
                # EUR weakness in August (vacation effect)
                patterns['eur_august_weakness'] = 1.0 if timestamp.month == 8 else 0.0
                # EUR strength in Q1
                patterns['eur_q1_strength'] = 1.0 if timestamp.month <= 3 else 0.0
            
            # JPY patterns
            if 'JPY' in currency_pair:
                # JPY fiscal year end (March)
                patterns['jpy_fiscal_year_end'] = 1.0 if timestamp.month == 3 else 0.0
                # JPY golden week effect (early May)
                patterns['jpy_golden_week'] = 1.0 if (timestamp.month == 5 and timestamp.day <= 7) else 0.0
            
            # GBP patterns
            if 'GBP' in currency_pair:
                # GBP tax year end (April)
                patterns['gbp_tax_year_end'] = 1.0 if timestamp.month == 4 else 0.0
            
            # AUD patterns
            if 'AUD' in currency_pair:
                # AUD commodity cycle correlation
                patterns['aud_commodity_season'] = 1.0 if timestamp.month in [3, 4, 9, 10] else 0.0
            
            # CAD patterns
            if 'CAD' in currency_pair:
                # CAD oil price correlation seasonality
                patterns['cad_oil_season'] = 1.0 if timestamp.month in [5, 6, 7, 8] else 0.0
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting seasonal patterns: {e}")
            return {}
    
    def _get_holiday_effects(self, timestamp: datetime, currency_pair: str) -> Dict[str, float]:
        """Get holiday calendar effects"""
        effects = {}
        
        try:
            # Major holidays that affect forex markets
            # Christmas/New Year period
            effects['christmas_period'] = 1.0 if (timestamp.month == 12 and timestamp.day >= 20) or (timestamp.month == 1 and timestamp.day <= 5) else 0.0
            
            # Thanksgiving week (US)
            if timestamp.month == 11:
                # Fourth Thursday of November
                thanksgiving_day = 22 + (3 - datetime(timestamp.year, 11, 22).weekday()) % 7
                if thanksgiving_day - 3 <= timestamp.day <= thanksgiving_day + 1:
                    effects['thanksgiving_week'] = 1.0
                else:
                    effects['thanksgiving_week'] = 0.0
            else:
                effects['thanksgiving_week'] = 0.0
            
            # Easter effects (simplified - typically March/April)
            effects['easter_period'] = 1.0 if timestamp.month in [3, 4] and timestamp.weekday() in [4, 0] else 0.0
            
            return effects
            
        except Exception as e:
            logger.error(f"Error getting holiday effects: {e}")
            return {}
    
    def _is_fomc_week(self, timestamp: datetime) -> float:
        """Check if it's FOMC meeting week (simplified)"""
        try:
            # FOMC meets 8 times per year, roughly every 6-7 weeks
            # Simplified: assume meetings in specific weeks
            fomc_months = [1, 3, 5, 6, 7, 9, 11, 12]
            if timestamp.month in fomc_months:
                # Assume meetings in 2nd or 3rd week of the month
                if 8 <= timestamp.day <= 21:
                    return 1.0
            return 0.0
        except:
            return 0.0
    
    def _is_ecb_week(self, timestamp: datetime) -> float:
        """Check if it's ECB meeting week (simplified)"""
        try:
            # ECB meets 8 times per year
            ecb_months = [1, 3, 4, 6, 7, 9, 10, 12]
            if timestamp.month in ecb_months:
                # Assume meetings in 1st or 2nd week of the month
                if 1 <= timestamp.day <= 14:
                    return 1.0
            return 0.0
        except:
            return 0.0

class ForexFeatureEngineer:
    """Main feature engineering class for forex"""
    
    def __init__(self, currency_pair: str = "EURUSD"):
        """
        Initialize forex feature engineer
        
        Args:
            currency_pair: Currency pair to analyze (e.g., "EURUSD")
        """
        self.currency_pair = currency_pair
        self.base_currency = currency_pair[:3]
        self.quote_currency = currency_pair[3:]
        
        # Initialize feature extractors
        self.price_action_extractor = ForexPriceActionFeatures()
        self.economic_extractor = ForexEconomicFeatures(self.base_currency, self.quote_currency)
        self.carry_trade_extractor = ForexCarryTradeFeatures()
        self.intervention_extractor = ForexInterventionFeatures()
        self.seasonality_extractor = ForexSeasonalityFeatures()
        
        # Feature selection and preprocessing
        self.feature_selector = None
        self.scaler = StandardScaler()
        self.pca = None
        
        logger.info(f"Initialized ForexFeatureEngineer for {currency_pair}")
    
    def extract_all_features(self, 
                           prices: List[float],
                           volumes: List[float] = None,
                           base_economic_data: Dict[str, float] = None,
                           quote_economic_data: Dict[str, float] = None,
                           global_data: Dict[str, float] = None,
                           central_bank_data: Dict[str, Any] = None,
                           timestamp: datetime = None) -> ForexFeatureSet:
        """Extract all features for forex analysis"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            if base_economic_data is None:
                base_economic_data = {}
            if quote_economic_data is None:
                quote_economic_data = {}
            if global_data is None:
                global_data = {}
            if central_bank_data is None:
                central_bank_data = {}
            
            # Initialize feature set
            feature_set = ForexFeatureSet(
                timestamp=timestamp,
                currency_pair=self.currency_pair
            )
            
            # Extract price action features
            feature_set.price_features = self.price_action_extractor.extract_features(prices, volumes)
            
            # Extract economic features
            feature_set.economic_features = self.economic_extractor.extract_features(
                base_economic_data, quote_economic_data, global_data
            )
            
            # Extract carry trade features
            if len(prices) > 0:
                base_rate = base_economic_data.get('interest_rate', 0.0)
                quote_rate = quote_economic_data.get('interest_rate', 0.0)
                base_vol = feature_set.price_features.get('volatility_20d', 0.01)
                quote_vol = base_vol  # Simplified
                
                feature_set.carry_trade_features = self.carry_trade_extractor.extract_features(
                    base_rate, quote_rate, base_vol, quote_vol
                )
            
            # Extract intervention features
            feature_set.intervention_features = self.intervention_extractor.extract_features(
                prices[-1] if prices else 0.0, prices, central_bank_data, base_economic_data
            )
            
            # Extract seasonality features
            feature_set.seasonality_features = self.seasonality_extractor.extract_features(
                timestamp, self.currency_pair
            )
            
            # Extract additional technical features
            feature_set.technical_features = self._extract_technical_features(prices)
            
            # Extract volatility features
            feature_set.volatility_features = self._extract_volatility_features(prices)
            
            # Extract momentum features
            feature_set.momentum_features = self._extract_momentum_features(prices)
            
            # Extract risk features
            feature_set.risk_features = self._extract_risk_features(prices, global_data)
            
            # Calculate feature importance (simplified)
            feature_set.feature_importance = self._calculate_feature_importance(feature_set)
            
            # Calculate feature quality
            feature_set.feature_quality = self._calculate_feature_quality(feature_set)
            
            return feature_set
            
        except Exception as e:
            logger.error(f"Error extracting all features: {e}")
            return ForexFeatureSet(timestamp=timestamp or datetime.now(), currency_pair=self.currency_pair)
    
    def _extract_technical_features(self, prices: List[float]) -> Dict[str, float]:
        """Extract technical indicator features"""
        try:
            features = {}
            
            if len(prices) < 20:
                return features
            
            prices_array = np.array(prices)
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                if len(prices) >= period:
                    ma = np.mean(prices_array[-period:])
                    features[f'ma_{period}'] = ma
                    features[f'price_to_ma_{period}'] = (prices_array[-1] - ma) / ma
            
            # RSI (simplified)
            if len(prices) >= 14:
                returns = np.diff(prices_array[-15:])
                gains = np.where(returns > 0, returns, 0)
                losses = np.where(returns < 0, -returns, 0)
                
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    features['rsi'] = rsi
                    features['rsi_overbought'] = 1.0 if rsi > 70 else 0.0
                    features['rsi_oversold'] = 1.0 if rsi < 30 else 0.0
            
            # Bollinger Bands
            if len(prices) >= 20:
                ma_20 = np.mean(prices_array[-20:])
                std_20 = np.std(prices_array[-20:])
                upper_band = ma_20 + 2 * std_20
                lower_band = ma_20 - 2 * std_20
                
                features['bb_upper'] = upper_band
                features['bb_lower'] = lower_band
                features['bb_position'] = (prices_array[-1] - lower_band) / (upper_band - lower_band) if upper_band > lower_band else 0.5
                features['bb_squeeze'] = 1.0 if (upper_band - lower_band) / ma_20 < 0.1 else 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting technical features: {e}")
            return {}
    
    def _extract_volatility_features(self, prices: List[float]) -> Dict[str, float]:
        """Extract volatility-related features"""
        try:
            features = {}
            
            if len(prices) < 10:
                return features
            
            prices_array = np.array(prices)
            returns = np.diff(prices_array) / prices_array[:-1]
            
            # Historical volatility for different periods
            for period in [5, 10, 20, 50]:
                if len(returns) >= period:
                    period_vol = np.std(returns[-period:]) * np.sqrt(252)  # Annualized
                    features[f'volatility_{period}d'] = period_vol
            
            # Volatility of volatility
            if len(prices) >= 50:
                rolling_vols = []
                for i in range(10, len(returns)):
                    vol = np.std(returns[i-10:i])
                    rolling_vols.append(vol)
                
                if rolling_vols:
                    features['vol_of_vol'] = np.std(rolling_vols)
            
            # GARCH-like features (simplified)
            if len(returns) >= 20:
                # Volatility clustering
                abs_returns = np.abs(returns[-20:])
                features['volatility_clustering'] = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1] if len(abs_returns) > 1 else 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting volatility features: {e}")
            return {}
    
    def _extract_momentum_features(self, prices: List[float]) -> Dict[str, float]:
        """Extract momentum-related features"""
        try:
            features = {}
            
            if len(prices) < 10:
                return features
            
            prices_array = np.array(prices)
            
            # Price momentum for different periods
            for period in [5, 10, 20]:
                if len(prices) > period:
                    momentum = (prices_array[-1] - prices_array[-period-1]) / prices_array[-period-1]
                    features[f'momentum_{period}d'] = momentum
            
            # Rate of change
            if len(prices) >= 10:
                roc = (prices_array[-1] - prices_array[-10]) / prices_array[-10]
                features['rate_of_change'] = roc
            
            # Momentum acceleration
            if len(prices) >= 15:
                momentum_5d = (prices_array[-1] - prices_array[-6]) / prices_array[-6]
                momentum_10d = (prices_array[-6] - prices_array[-11]) / prices_array[-11]
                features['momentum_acceleration'] = momentum_5d - momentum_10d
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting momentum features: {e}")
            return {}
    
    def _extract_risk_features(self, prices: List[float], global_data: Dict[str, float]) -> Dict[str, float]:
        """Extract risk-related features"""
        try:
            features = {}
            
            # VaR calculation (simplified)
            if len(prices) >= 20:
                returns = np.diff(prices) / prices[:-1]
                var_95 = np.percentile(returns, 5)
                var_99 = np.percentile(returns, 1)
                
                features['var_95'] = var_95
                features['var_99'] = var_99
                features['expected_shortfall'] = np.mean(returns[returns <= var_95])
            
            # Maximum drawdown
            if len(prices) >= 20:
                peak = np.maximum.accumulate(prices)
                drawdown = (prices - peak) / peak
                features['max_drawdown'] = np.min(drawdown)
                features['current_drawdown'] = drawdown[-1]
            
            # Global risk indicators
            features['global_vix'] = global_data.get('vix', 20.0)
            features['risk_on_off'] = 1.0 if global_data.get('vix', 20.0) > 25.0 else -1.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting risk features: {e}")
            return {}
    
    def _calculate_feature_importance(self, feature_set: ForexFeatureSet) -> Dict[str, float]:
        """Calculate feature importance scores (simplified)"""
        try:
            importance = {}
            
            # Economic features are generally most important for forex
            for feature in feature_set.economic_features.keys():
                if 'differential' in feature:
                    importance[feature] = 0.9
                elif 'rate' in feature:
                    importance[feature] = 0.8
                else:
                    importance[feature] = 0.6
            
            # Carry trade features
            for feature in feature_set.carry_trade_features.keys():
                importance[feature] = 0.7
            
            # Price action features
            for feature in feature_set.price_features.keys():
                if 'return' in feature or 'volatility' in feature:
                    importance[feature] = 0.6
                else:
                    importance[feature] = 0.4
            
            # Other features get default importance
            all_features = feature_set.get_all_features()
            for feature in all_features.keys():
                if feature not in importance:
                    importance[feature] = 0.3
            
            return importance
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {e}")
            return {}
    
    def _calculate_feature_quality(self, feature_set: ForexFeatureSet) -> Dict[str, float]:
        """Calculate feature quality scores"""
        try:
            quality = {}
            
            all_features = feature_set.get_all_features()
            
            for feature_name, feature_value in all_features.items():
                # Quality based on value characteristics
                if np.isnan(feature_value) or np.isinf(feature_value):
                    quality[feature_name] = 0.0
                elif abs(feature_value) > 100:  # Extreme values
                    quality[feature_name] = 0.3
                elif abs(feature_value) < 1e-10:  # Too small values
                    quality[feature_name] = 0.4
                else:
                    quality[feature_name] = 0.8
            
            return quality
            
        except Exception as e:
            logger.error(f"Error calculating feature quality: {e}")
            return {}
    
    def prepare_ml_features(self, feature_sets: List[ForexFeatureSet], 
                          target_values: List[float] = None,
                          feature_selection: bool = True,
                          scaling: bool = True,
                          pca_components: int = None) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for machine learning"""
        try:
            if not feature_sets:
                return np.array([]), []
            
            # Convert feature sets to DataFrame
            feature_data = []
            feature_names = None
            
            for feature_set in feature_sets:
                all_features = feature_set.get_all_features()
                
                if feature_names is None:
                    feature_names = list(all_features.keys())
                
                # Ensure consistent feature order
                feature_vector = [all_features.get(name, 0.0) for name in feature_names]
                feature_data.append(feature_vector)
            
            feature_matrix = np.array(feature_data)
            
            # Handle missing values
            feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Feature selection
            if feature_selection and target_values is not None and len(target_values) == len(feature_sets):
                if self.feature_selector is None:
                    n_features = min(50, feature_matrix.shape[1])  # Select top 50 features
                    self.feature_selector = SelectKBest(score_func=f_regression, k=n_features)
                    feature_matrix = self.feature_selector.fit_transform(feature_matrix, target_values)
                    
                    # Update feature names
                    selected_indices = self.feature_selector.get_support(indices=True)
                    feature_names = [feature_names[i] for i in selected_indices]
                else:
                    feature_matrix = self.feature_selector.transform(feature_matrix)
            
            # Scaling
            if scaling:
                if not hasattr(self.scaler, 'mean_') or self.scaler.mean_ is None:
                    feature_matrix = self.scaler.fit_transform(feature_matrix)
                else:
                    feature_matrix = self.scaler.transform(feature_matrix)
            
            # PCA dimensionality reduction
            if pca_components is not None and pca_components < feature_matrix.shape[1]:
                if self.pca is None:
                    self.pca = PCA(n_components=pca_components)
                    feature_matrix = self.pca.fit_transform(feature_matrix)
                    
                    # Update feature names for PCA components
                    feature_names = [f'pca_component_{i}' for i in range(pca_components)]
                else:
                    feature_matrix = self.pca.transform(feature_matrix)
            
            return feature_matrix, feature_names
            
        except Exception as e:
            logger.error(f"Error preparing ML features: {e}")
            return np.array([]), []
    
    def get_feature_statistics(self, feature_sets: List[ForexFeatureSet]) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of features"""
        try:
            if not feature_sets:
                return {}
            
            # Collect all features
            all_features_data = {}
            
            for feature_set in feature_sets:
                all_features = feature_set.get_all_features()
                
                for feature_name, feature_value in all_features.items():
                    if feature_name not in all_features_data:
                        all_features_data[feature_name] = []
                    all_features_data[feature_name].append(feature_value)
            
            # Calculate statistics
            statistics = {}
            
            for feature_name, values in all_features_data.items():
                values_array = np.array(values)
                values_array = values_array[~np.isnan(values_array)]  # Remove NaN values
                
                if len(values_array) > 0:
                    statistics[feature_name] = {
                        'mean': float(np.mean(values_array)),
                        'std': float(np.std(values_array)),
                        'min': float(np.min(values_array)),
                        'max': float(np.max(values_array)),
                        'median': float(np.median(values_array)),
                        'q25': float(np.percentile(values_array, 25)),
                        'q75': float(np.percentile(values_array, 75)),
                        'count': len(values_array),
                        'missing_ratio': (len(values) - len(values_array)) / len(values)
                    }
                else:
                    statistics[feature_name] = {
                        'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0,
                        'median': 0.0, 'q25': 0.0, 'q75': 0.0,
                        'count': 0, 'missing_ratio': 1.0
                    }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error calculating feature statistics: {e}")
            return {}
    
    def validate_features(self, feature_set: ForexFeatureSet) -> Dict[str, List[str]]:
        """Validate feature quality and return issues"""
        try:
            issues = {
                'missing_values': [],
                'infinite_values': [],
                'extreme_values': [],
                'low_quality': []
            }
            
            all_features = feature_set.get_all_features()
            
            for feature_name, feature_value in all_features.items():
                # Check for missing values
                if np.isnan(feature_value):
                    issues['missing_values'].append(feature_name)
                
                # Check for infinite values
                elif np.isinf(feature_value):
                    issues['infinite_values'].append(feature_name)
                
                # Check for extreme values
                elif abs(feature_value) > 1000:
                    issues['extreme_values'].append(feature_name)
                
                # Check feature quality
                quality_score = feature_set.feature_quality.get(feature_name, 0.0)
                if quality_score < 0.5:
                    issues['low_quality'].append(feature_name)
            
            return issues
            
        except Exception as e:
            logger.error(f"Error validating features: {e}")
            return {'missing_values': [], 'infinite_values': [], 'extreme_values': [], 'low_quality': []}

# Example usage
if __name__ == "__main__":
    # Initialize feature engineer
    engineer = ForexFeatureEngineer("EURUSD")
    
    # Sample data
    sample_prices = [1.1000 + 0.001 * np.sin(i * 0.1) + 0.0005 * np.random.randn() for i in range(100)]
    sample_volumes = [1000 + 100 * np.random.randn() for _ in range(100)]
    
    # Sample economic data
    base_economic = {
        'interest_rate': 0.025,
        'inflation_rate': 0.02,
        'gdp_growth': 0.015,
        'unemployment_rate': 0.08,
        'current_account': -0.02,
        'trade_balance': 0.01,
        'pmi_manufacturing': 52.0,
        'pmi_services': 54.0,
        'consumer_confidence': 0.1
    }
    
    quote_economic = {
        'interest_rate': 0.015,
        'inflation_rate': 0.018,
        'gdp_growth': 0.012,
        'unemployment_rate': 0.075,
        'current_account': 0.01,
        'trade_balance': -0.005,
        'pmi_manufacturing': 48.0,
        'pmi_services': 51.0,
        'consumer_confidence': -0.05
    }
    
    global_data = {
        'vix': 22.0,
        'risk_appetite': 0.1,
        'global_liquidity': 0.05,
        'commodity_index': 0.02,
        'global_gdp_growth': 0.03
    }
    
    central_bank_data = {
        'verbal_intervention_score': 0.2,
        'hawkish_tone': 0.3,
        'dovish_tone': 0.1,
        'intervention_threat': 0.0,
        'policy_divergence_score': 0.4,
        'market_depth_score': 0.7,
        'historical_success_rate': 0.6
    }
    
    # Extract features
    feature_set = engineer.extract_all_features(
        prices=sample_prices,
        volumes=sample_volumes,
        base_economic_data=base_economic,
        quote_economic_data=quote_economic,
        global_data=global_data,
        central_bank_data=central_bank_data,
        timestamp=datetime.now()
    )
    
    print(f"Extracted {len(feature_set.get_all_features())} features for {feature_set.currency_pair}")
    print(f"Feature categories:")
    print(f"  - Price action: {len(feature_set.price_features)}")
    print(f"  - Economic: {len(feature_set.economic_features)}")
    print(f"  - Carry trade: {len(feature_set.carry_trade_features)}")
    print(f"  - Technical: {len(feature_set.technical_features)}")
    print(f"  - Intervention: {len(feature_set.intervention_features)}")
    print(f"  - Seasonality: {len(feature_set.seasonality_features)}")
    print(f"  - Volatility: {len(feature_set.volatility_features)}")
    print(f"  - Momentum: {len(feature_set.momentum_features)}")
    print(f"  - Risk: {len(feature_set.risk_features)}")
    
    # Validate features
    issues = engineer.validate_features(feature_set)
    print(f"\nFeature validation:")
    for issue_type, feature_list in issues.items():
        if feature_list:
            print(f"  - {issue_type}: {len(feature_list)} features")
    
    # Sample ML preparation
    feature_sets = [feature_set]  # In practice, you'd have multiple feature sets
    target_values = [0.001]  # Sample target (e.g., next period return)
    
    ml_features, feature_names = engineer.prepare_ml_features(
        feature_sets, target_values, feature_selection=False, scaling=True
    )
    
    print(f"\nML-ready features: {ml_features.shape}")
    print(f"Feature names: {len(feature_names)}")
    
    # Feature statistics
    stats = engineer.get_feature_statistics(feature_sets)
    print(f"\nFeature statistics calculated for {len(stats)} features")
    
    # Display some key economic features
    print(f"\nKey economic features:")
    for feature_name, value in feature_set.economic_features.items():
        if 'differential' in feature_name:
            print(f"  - {feature_name}: {value:.4f}")
    
    print(f"\nKey carry trade features:")
    for feature_name, value in feature_set.carry_trade_features.items():
        if 'carry' in feature_name:
            print(f"  - {feature_name}: {value:.4f}")