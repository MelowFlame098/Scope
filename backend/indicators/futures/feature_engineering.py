import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FuturesContractType(Enum):
    ENERGY = "energy"
    METALS = "metals"
    AGRICULTURE = "agriculture"
    CURRENCIES = "currencies"
    INTEREST_RATES = "interest_rates"
    EQUITY_INDEX = "equity_index"
    VOLATILITY = "volatility"

class SeasonalityType(Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    HARVEST = "harvest"
    WEATHER = "weather"
    CALENDAR = "calendar"

@dataclass
class FuturesFeatureSet:
    """Comprehensive feature set for futures analysis"""
    contract_symbol: str
    timestamp: datetime
    
    # Price action features
    price_features: Dict[str, float] = field(default_factory=dict)
    
    # Technical indicators
    technical_features: Dict[str, float] = field(default_factory=dict)
    
    # Term structure features
    term_structure_features: Dict[str, float] = field(default_factory=dict)
    
    # Basis and carry features
    basis_features: Dict[str, float] = field(default_factory=dict)
    
    # Volatility features
    volatility_features: Dict[str, float] = field(default_factory=dict)
    
    # Volume and open interest features
    volume_features: Dict[str, float] = field(default_factory=dict)
    
    # Seasonality features
    seasonality_features: Dict[str, float] = field(default_factory=dict)
    
    # Market microstructure features
    microstructure_features: Dict[str, float] = field(default_factory=dict)
    
    # Cross-asset features
    cross_asset_features: Dict[str, float] = field(default_factory=dict)
    
    # Macro economic features
    macro_features: Dict[str, float] = field(default_factory=dict)
    
    # Sentiment features
    sentiment_features: Dict[str, float] = field(default_factory=dict)
    
    # Risk features
    risk_features: Dict[str, float] = field(default_factory=dict)
    
    # Regime features
    regime_features: Dict[str, float] = field(default_factory=dict)
    
    # Feature metadata
    feature_quality: float = 1.0
    missing_data_ratio: float = 0.0
    feature_count: int = 0
    
    def get_all_features(self) -> Dict[str, float]:
        """Get all features as a single dictionary"""
        all_features = {}
        all_features.update(self.price_features)
        all_features.update(self.technical_features)
        all_features.update(self.term_structure_features)
        all_features.update(self.basis_features)
        all_features.update(self.volatility_features)
        all_features.update(self.volume_features)
        all_features.update(self.seasonality_features)
        all_features.update(self.microstructure_features)
        all_features.update(self.cross_asset_features)
        all_features.update(self.macro_features)
        all_features.update(self.sentiment_features)
        all_features.update(self.risk_features)
        all_features.update(self.regime_features)
        return all_features
    
    def get_feature_vector(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Get features as numpy array"""
        all_features = self.get_all_features()
        
        if feature_names:
            return np.array([all_features.get(name, 0.0) for name in feature_names])
        else:
            return np.array(list(all_features.values()))

class FuturesPriceActionFeatures:
    """Price action feature engineering for futures"""
    
    @staticmethod
    def calculate_price_features(prices: List[float], 
                               volumes: Optional[List[float]] = None,
                               lookback_periods: List[int] = [5, 10, 20, 50]) -> Dict[str, float]:
        """Calculate price action features"""
        features = {}
        
        if len(prices) < 2:
            return features
        
        try:
            prices_array = np.array(prices)
            
            # Returns
            returns = np.diff(prices_array) / prices_array[:-1]
            
            # Basic price features
            features['current_price'] = prices_array[-1]
            features['price_change_1d'] = returns[-1] if len(returns) > 0 else 0
            features['price_change_pct_1d'] = returns[-1] * 100 if len(returns) > 0 else 0
            
            # Multi-period returns
            for period in lookback_periods:
                if len(prices_array) > period:
                    period_return = (prices_array[-1] - prices_array[-period-1]) / prices_array[-period-1]
                    features[f'return_{period}d'] = period_return
                    features[f'return_pct_{period}d'] = period_return * 100
            
            # Price momentum
            if len(prices_array) >= 10:
                # Price momentum (slope of linear regression)
                x = np.arange(len(prices_array[-10:]))
                slope, _, r_value, _, _ = stats.linregress(x, prices_array[-10:])
                features['price_momentum_10d'] = slope
                features['price_trend_strength_10d'] = abs(r_value)
            
            # Price acceleration (second derivative)
            if len(returns) >= 5:
                returns_change = np.diff(returns)
                features['price_acceleration_5d'] = np.mean(returns_change[-5:])
            
            # Price volatility
            for period in [5, 10, 20]:
                if len(returns) >= period:
                    vol = np.std(returns[-period:]) * np.sqrt(252)
                    features[f'volatility_{period}d'] = vol
            
            # Price levels relative to recent history
            for period in [20, 50, 100]:
                if len(prices_array) >= period:
                    recent_prices = prices_array[-period:]
                    current_price = prices_array[-1]
                    
                    features[f'price_percentile_{period}d'] = stats.percentileofscore(recent_prices, current_price) / 100
                    features[f'price_zscore_{period}d'] = (current_price - np.mean(recent_prices)) / (np.std(recent_prices) + 1e-8)
            
            # Gap analysis
            if len(prices_array) >= 2:
                # Overnight gap (assuming daily data)
                gap = (prices_array[-1] - prices_array[-2]) / prices_array[-2]
                features['overnight_gap'] = gap
                features['gap_magnitude'] = abs(gap)
            
            # Volume-weighted features (if volume available)
            if volumes and len(volumes) == len(prices):
                volumes_array = np.array(volumes)
                
                # VWAP
                if len(volumes_array) >= 20:
                    vwap_20 = np.sum(prices_array[-20:] * volumes_array[-20:]) / np.sum(volumes_array[-20:])
                    features['vwap_20d'] = vwap_20
                    features['price_vs_vwap_20d'] = (prices_array[-1] - vwap_20) / vwap_20
                
                # Volume-weighted momentum
                if len(volumes_array) >= 10:
                    vw_returns = returns[-9:] * volumes_array[-9:] / np.sum(volumes_array[-9:])
                    features['volume_weighted_momentum_10d'] = np.sum(vw_returns)
            
        except Exception as e:
            logger.error(f"Error calculating price features: {e}")
        
        return features

class FuturesTermStructureFeatures:
    """Term structure feature engineering for futures"""
    
    @staticmethod
    def calculate_term_structure_features(contract_prices: Dict[str, float],
                                        contract_expiries: Dict[str, datetime],
                                        current_date: datetime = None) -> Dict[str, float]:
        """Calculate term structure features"""
        features = {}
        
        if len(contract_prices) < 2:
            return features
        
        try:
            if current_date is None:
                current_date = datetime.now()
            
            # Sort contracts by expiry
            sorted_contracts = sorted(contract_expiries.items(), key=lambda x: x[1])
            
            # Calculate time to expiry and prices
            times_to_expiry = []
            prices = []
            
            for contract, expiry in sorted_contracts:
                if contract in contract_prices:
                    tte = (expiry - current_date).days / 365.25  # Years to expiry
                    times_to_expiry.append(tte)
                    prices.append(contract_prices[contract])
            
            if len(prices) < 2:
                return features
            
            times_array = np.array(times_to_expiry)
            prices_array = np.array(prices)
            
            # Term structure slope
            if len(prices_array) >= 2:
                # Linear slope
                slope, _, r_value, _, _ = stats.linregress(times_array, prices_array)
                features['term_structure_slope'] = slope
                features['term_structure_r_squared'] = r_value ** 2
                
                # Contango/Backwardation
                if slope > 0:
                    features['contango_strength'] = slope
                    features['backwardation_strength'] = 0
                    features['market_structure'] = 1  # Contango
                else:
                    features['contango_strength'] = 0
                    features['backwardation_strength'] = abs(slope)
                    features['market_structure'] = -1  # Backwardation
            
            # Calendar spreads
            for i in range(len(prices_array) - 1):
                spread = prices_array[i+1] - prices_array[i]
                time_diff = times_array[i+1] - times_array[i]
                
                if time_diff > 0:
                    spread_per_year = spread / time_diff
                    features[f'calendar_spread_{i+1}'] = spread
                    features[f'calendar_spread_annualized_{i+1}'] = spread_per_year
            
            # Term structure curvature (if 3+ contracts)
            if len(prices_array) >= 3:
                # Fit quadratic
                coeffs = np.polyfit(times_array, prices_array, 2)
                features['term_structure_curvature'] = coeffs[0]  # Quadratic coefficient
                
                # Butterfly spread (middle - average of wings)
                if len(prices_array) >= 3:
                    butterfly = prices_array[1] - (prices_array[0] + prices_array[2]) / 2
                    features['butterfly_spread'] = butterfly
            
            # Front month vs back month
            if len(prices_array) >= 2:
                front_back_ratio = prices_array[0] / prices_array[-1]
                features['front_back_ratio'] = front_back_ratio
                features['front_back_spread'] = prices_array[0] - prices_array[-1]
            
            # Term structure volatility
            if len(prices_array) >= 3:
                price_changes = np.diff(prices_array)
                features['term_structure_volatility'] = np.std(price_changes)
            
            # Roll yield estimation
            if len(prices_array) >= 2 and len(times_array) >= 2:
                # Estimate roll yield for front contract
                front_price = prices_array[0]
                next_price = prices_array[1]
                time_to_roll = times_array[0]  # Assuming rolling at expiry
                
                if time_to_roll > 0:
                    roll_yield = (next_price - front_price) / front_price / time_to_roll
                    features['estimated_roll_yield'] = roll_yield
            
        except Exception as e:
            logger.error(f"Error calculating term structure features: {e}")
        
        return features

class FuturesBasisFeatures:
    """Basis and carry feature engineering for futures"""
    
    @staticmethod
    def calculate_basis_features(futures_price: float,
                               spot_price: float,
                               risk_free_rate: float,
                               dividend_yield: float,
                               time_to_expiry: float,
                               storage_cost: float = 0.0) -> Dict[str, float]:
        """Calculate basis and carry features"""
        features = {}
        
        try:
            # Basic basis
            basis = futures_price - spot_price
            basis_pct = basis / spot_price if spot_price != 0 else 0
            
            features['basis_absolute'] = basis
            features['basis_percentage'] = basis_pct * 100
            
            # Theoretical fair value
            if time_to_expiry > 0:
                # Cost of carry model
                carry_rate = risk_free_rate - dividend_yield + storage_cost
                theoretical_futures = spot_price * np.exp(carry_rate * time_to_expiry)
                
                features['theoretical_futures_price'] = theoretical_futures
                features['fair_value_basis'] = futures_price - theoretical_futures
                features['fair_value_basis_pct'] = (futures_price - theoretical_futures) / theoretical_futures * 100
                
                # Implied convenience yield
                if futures_price > 0 and spot_price > 0:
                    implied_rate = np.log(futures_price / spot_price) / time_to_expiry
                    convenience_yield = risk_free_rate + storage_cost - implied_rate
                    features['implied_convenience_yield'] = convenience_yield
                
                # Carry return
                carry_return = (theoretical_futures - spot_price) / spot_price
                features['carry_return'] = carry_return
                features['carry_return_annualized'] = carry_return / time_to_expiry if time_to_expiry > 0 else 0
            
            # Basis momentum (requires historical data - placeholder)
            features['basis_momentum'] = 0  # Would need historical basis data
            
        except Exception as e:
            logger.error(f"Error calculating basis features: {e}")
        
        return features

class FuturesSeasonalityFeatures:
    """Seasonality feature engineering for futures"""
    
    @staticmethod
    def calculate_seasonality_features(current_date: datetime,
                                     contract_type: FuturesContractType,
                                     historical_prices: Optional[List[Tuple[datetime, float]]] = None) -> Dict[str, float]:
        """Calculate seasonality features"""
        features = {}
        
        try:
            # Calendar-based seasonality
            month = current_date.month
            day_of_year = current_date.timetuple().tm_yday
            quarter = (month - 1) // 3 + 1
            
            # Cyclical encoding of time features
            features['month_sin'] = np.sin(2 * np.pi * month / 12)
            features['month_cos'] = np.cos(2 * np.pi * month / 12)
            features['day_of_year_sin'] = np.sin(2 * np.pi * day_of_year / 365)
            features['day_of_year_cos'] = np.cos(2 * np.pi * day_of_year / 365)
            features['quarter'] = quarter
            
            # Contract-specific seasonality patterns
            if contract_type == FuturesContractType.ENERGY:
                # Energy seasonality (heating/cooling seasons)
                features['heating_season'] = 1 if month in [11, 12, 1, 2, 3] else 0
                features['cooling_season'] = 1 if month in [6, 7, 8, 9] else 0
                features['shoulder_season'] = 1 if month in [4, 5, 10] else 0
                
            elif contract_type == FuturesContractType.AGRICULTURE:
                # Agricultural seasonality (planting/harvest)
                features['planting_season'] = 1 if month in [3, 4, 5] else 0
                features['growing_season'] = 1 if month in [6, 7, 8] else 0
                features['harvest_season'] = 1 if month in [9, 10, 11] else 0
                features['winter_season'] = 1 if month in [12, 1, 2] else 0
                
            elif contract_type == FuturesContractType.METALS:
                # Industrial demand patterns
                features['industrial_high_season'] = 1 if month in [3, 4, 5, 9, 10, 11] else 0
                features['summer_slowdown'] = 1 if month in [7, 8] else 0
                features['year_end_effect'] = 1 if month == 12 else 0
            
            # Historical seasonality analysis (if data available)
            if historical_prices and len(historical_prices) > 365:
                # Calculate historical seasonal patterns
                monthly_returns = {i: [] for i in range(1, 13)}
                
                for i in range(1, len(historical_prices)):
                    date, price = historical_prices[i]
                    prev_date, prev_price = historical_prices[i-1]
                    
                    if prev_price != 0:
                        monthly_return = (price - prev_price) / prev_price
                        monthly_returns[date.month].append(monthly_return)
                
                # Average seasonal returns
                for month_num in range(1, 13):
                    if monthly_returns[month_num]:
                        avg_return = np.mean(monthly_returns[month_num])
                        features[f'historical_seasonal_return_month_{month_num}'] = avg_return
                
                # Current month seasonal bias
                if monthly_returns[month]:
                    features['current_month_seasonal_bias'] = np.mean(monthly_returns[month])
                    features['current_month_seasonal_volatility'] = np.std(monthly_returns[month])
            
            # Holiday effects
            # Major holidays that affect commodity markets
            holidays = {
                (1, 1): 'new_year',
                (7, 4): 'independence_day',
                (11, 4): 'thanksgiving_week',  # Approximate
                (12, 25): 'christmas'
            }
            
            for (holiday_month, holiday_day), holiday_name in holidays.items():
                days_to_holiday = abs((datetime(current_date.year, holiday_month, holiday_day) - current_date).days)
                if days_to_holiday <= 5:  # Within 5 days of holiday
                    features[f'{holiday_name}_effect'] = 1
                else:
                    features[f'{holiday_name}_effect'] = 0
            
        except Exception as e:
            logger.error(f"Error calculating seasonality features: {e}")
        
        return features

class FuturesVolumeFeatures:
    """Volume and open interest feature engineering"""
    
    @staticmethod
    def calculate_volume_features(volumes: List[float],
                                open_interests: Optional[List[float]] = None,
                                prices: Optional[List[float]] = None) -> Dict[str, float]:
        """Calculate volume and open interest features"""
        features = {}
        
        if not volumes or len(volumes) < 2:
            return features
        
        try:
            volumes_array = np.array(volumes)
            
            # Basic volume features
            features['current_volume'] = volumes_array[-1]
            features['volume_change_1d'] = volumes_array[-1] - volumes_array[-2] if len(volumes_array) > 1 else 0
            features['volume_change_pct_1d'] = (volumes_array[-1] - volumes_array[-2]) / volumes_array[-2] * 100 if len(volumes_array) > 1 and volumes_array[-2] != 0 else 0
            
            # Volume moving averages and ratios
            for period in [5, 10, 20]:
                if len(volumes_array) >= period:
                    vol_ma = np.mean(volumes_array[-period:])
                    features[f'volume_ma_{period}d'] = vol_ma
                    features[f'volume_ratio_{period}d'] = volumes_array[-1] / vol_ma if vol_ma != 0 else 1
            
            # Volume volatility
            if len(volumes_array) >= 10:
                vol_changes = np.diff(volumes_array[-10:])
                features['volume_volatility_10d'] = np.std(vol_changes)
            
            # Volume trend
            if len(volumes_array) >= 10:
                x = np.arange(len(volumes_array[-10:]))
                slope, _, r_value, _, _ = stats.linregress(x, volumes_array[-10:])
                features['volume_trend_10d'] = slope
                features['volume_trend_strength_10d'] = abs(r_value)
            
            # Volume percentiles
            for period in [20, 50]:
                if len(volumes_array) >= period:
                    recent_volumes = volumes_array[-period:]
                    current_volume = volumes_array[-1]
                    features[f'volume_percentile_{period}d'] = stats.percentileofscore(recent_volumes, current_volume) / 100
            
            # Price-volume relationship
            if prices and len(prices) == len(volumes):
                prices_array = np.array(prices)
                
                # On-balance volume
                if len(prices_array) >= 2:
                    price_changes = np.diff(prices_array)
                    obv_changes = np.where(price_changes > 0, volumes_array[1:], 
                                         np.where(price_changes < 0, -volumes_array[1:], 0))
                    features['obv_10d'] = np.sum(obv_changes[-10:]) if len(obv_changes) >= 10 else 0
                
                # Volume-weighted average price deviation
                if len(volumes_array) >= 10:
                    recent_prices = prices_array[-10:]
                    recent_volumes = volumes_array[-10:]
                    vwap = np.sum(recent_prices * recent_volumes) / np.sum(recent_volumes)
                    features['price_vwap_deviation_10d'] = (prices_array[-1] - vwap) / vwap * 100
                
                # Price-volume correlation
                if len(prices_array) >= 20:
                    price_returns = np.diff(prices_array[-20:]) / prices_array[-20:-1]
                    volume_changes = np.diff(volumes_array[-20:]) / volumes_array[-20:-1]
                    
                    if len(price_returns) > 0 and len(volume_changes) > 0:
                        correlation = np.corrcoef(price_returns, volume_changes)[0, 1]
                        features['price_volume_correlation_20d'] = correlation if not np.isnan(correlation) else 0
            
            # Open interest features
            if open_interests and len(open_interests) == len(volumes):
                oi_array = np.array(open_interests)
                
                features['current_open_interest'] = oi_array[-1]
                features['oi_change_1d'] = oi_array[-1] - oi_array[-2] if len(oi_array) > 1 else 0
                features['oi_change_pct_1d'] = (oi_array[-1] - oi_array[-2]) / oi_array[-2] * 100 if len(oi_array) > 1 and oi_array[-2] != 0 else 0
                
                # Volume to open interest ratio
                features['volume_oi_ratio'] = volumes_array[-1] / oi_array[-1] if oi_array[-1] != 0 else 0
                
                # Open interest trend
                if len(oi_array) >= 10:
                    x = np.arange(len(oi_array[-10:]))
                    slope, _, r_value, _, _ = stats.linregress(x, oi_array[-10:])
                    features['oi_trend_10d'] = slope
                    features['oi_trend_strength_10d'] = abs(r_value)
            
        except Exception as e:
            logger.error(f"Error calculating volume features: {e}")
        
        return features

class FuturesFeatureEngineer:
    """Main feature engineering class for futures"""
    
    def __init__(self, contract_type: FuturesContractType = FuturesContractType.ENERGY):
        self.contract_type = contract_type
        self.price_features = FuturesPriceActionFeatures()
        self.term_structure_features = FuturesTermStructureFeatures()
        self.basis_features = FuturesBasisFeatures()
        self.seasonality_features = FuturesSeasonalityFeatures()
        self.volume_features = FuturesVolumeFeatures()
        
        # Feature scalers
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler()
        }
        
        # PCA for dimensionality reduction
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        
    def engineer_features(self,
                         contract_symbol: str,
                         prices: List[float],
                         volumes: List[float],
                         timestamps: List[datetime],
                         contract_prices: Optional[Dict[str, float]] = None,
                         contract_expiries: Optional[Dict[str, datetime]] = None,
                         spot_price: Optional[float] = None,
                         risk_free_rate: float = 0.02,
                         dividend_yield: float = 0.0,
                         open_interests: Optional[List[float]] = None,
                         historical_data: Optional[List[Tuple[datetime, float]]] = None) -> FuturesFeatureSet:
        """Engineer comprehensive feature set for futures"""
        
        current_timestamp = timestamps[-1] if timestamps else datetime.now()
        
        # Initialize feature set
        feature_set = FuturesFeatureSet(
            contract_symbol=contract_symbol,
            timestamp=current_timestamp
        )
        
        try:
            # Price action features
            feature_set.price_features = self.price_features.calculate_price_features(
                prices=prices,
                volumes=volumes
            )
            
            # Term structure features
            if contract_prices and contract_expiries:
                feature_set.term_structure_features = self.term_structure_features.calculate_term_structure_features(
                    contract_prices=contract_prices,
                    contract_expiries=contract_expiries,
                    current_date=current_timestamp
                )
            
            # Basis features
            if spot_price and contract_expiries:
                # Calculate time to expiry for current contract
                if contract_symbol in contract_expiries:
                    time_to_expiry = (contract_expiries[contract_symbol] - current_timestamp).days / 365.25
                    
                    feature_set.basis_features = self.basis_features.calculate_basis_features(
                        futures_price=prices[-1],
                        spot_price=spot_price,
                        risk_free_rate=risk_free_rate,
                        dividend_yield=dividend_yield,
                        time_to_expiry=time_to_expiry
                    )
            
            # Seasonality features
            feature_set.seasonality_features = self.seasonality_features.calculate_seasonality_features(
                current_date=current_timestamp,
                contract_type=self.contract_type,
                historical_prices=historical_data
            )
            
            # Volume features
            feature_set.volume_features = self.volume_features.calculate_volume_features(
                volumes=volumes,
                open_interests=open_interests,
                prices=prices
            )
            
            # Technical indicators
            feature_set.technical_features = self._calculate_technical_features(prices, volumes)
            
            # Volatility features
            feature_set.volatility_features = self._calculate_volatility_features(prices)
            
            # Risk features
            feature_set.risk_features = self._calculate_risk_features(prices)
            
            # Calculate feature quality metrics
            all_features = feature_set.get_all_features()
            feature_set.feature_count = len(all_features)
            
            # Calculate missing data ratio
            missing_count = sum(1 for v in all_features.values() if v == 0 or np.isnan(v))
            feature_set.missing_data_ratio = missing_count / len(all_features) if all_features else 0
            
            # Feature quality score
            feature_set.feature_quality = max(0, 1 - feature_set.missing_data_ratio)
            
        except Exception as e:
            logger.error(f"Error engineering features: {e}")
        
        return feature_set
    
    def _calculate_technical_features(self, prices: List[float], volumes: List[float]) -> Dict[str, float]:
        """Calculate technical indicator features"""
        features = {}
        
        if len(prices) < 10:
            return features
        
        try:
            prices_array = np.array(prices)
            
            # Moving averages
            for period in [5, 10, 20, 50]:
                if len(prices_array) >= period:
                    ma = np.mean(prices_array[-period:])
                    features[f'ma_{period}'] = ma
                    features[f'price_ma_ratio_{period}'] = prices_array[-1] / ma
            
            # RSI
            if len(prices_array) >= 15:
                returns = np.diff(prices_array)
                gains = np.where(returns > 0, returns, 0)
                losses = np.where(returns < 0, -returns, 0)
                
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                
                if avg_loss != 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                    features['rsi_14'] = rsi
            
            # Bollinger Bands
            if len(prices_array) >= 20:
                ma_20 = np.mean(prices_array[-20:])
                std_20 = np.std(prices_array[-20:])
                
                upper_band = ma_20 + 2 * std_20
                lower_band = ma_20 - 2 * std_20
                
                features['bb_upper'] = upper_band
                features['bb_lower'] = lower_band
                features['bb_position'] = (prices_array[-1] - lower_band) / (upper_band - lower_band)
            
            # MACD
            if len(prices_array) >= 26:
                ema_12 = self._calculate_ema(prices_array, 12)
                ema_26 = self._calculate_ema(prices_array, 26)
                macd = ema_12[-1] - ema_26[-1]
                features['macd'] = macd
                
                if len(prices_array) >= 35:
                    macd_series = ema_12[-9:] - ema_26[-9:]
                    signal = self._calculate_ema(macd_series, 9)[-1]
                    features['macd_signal'] = signal
                    features['macd_histogram'] = macd - signal
            
        except Exception as e:
            logger.error(f"Error calculating technical features: {e}")
        
        return features
    
    def _calculate_volatility_features(self, prices: List[float]) -> Dict[str, float]:
        """Calculate volatility-based features"""
        features = {}
        
        if len(prices) < 10:
            return features
        
        try:
            prices_array = np.array(prices)
            returns = np.diff(prices_array) / prices_array[:-1]
            
            # Historical volatility
            for period in [10, 20, 50]:
                if len(returns) >= period:
                    vol = np.std(returns[-period:]) * np.sqrt(252)
                    features[f'historical_vol_{period}d'] = vol
            
            # GARCH-like volatility clustering
            if len(returns) >= 20:
                squared_returns = returns ** 2
                vol_clustering = np.corrcoef(squared_returns[-19:], squared_returns[-20:-1])[0, 1]
                features['volatility_clustering'] = vol_clustering if not np.isnan(vol_clustering) else 0
            
            # Volatility of volatility
            if len(returns) >= 30:
                rolling_vols = []
                for i in range(10, len(returns)):
                    vol = np.std(returns[i-10:i]) * np.sqrt(252)
                    rolling_vols.append(vol)
                
                if rolling_vols:
                    vol_of_vol = np.std(rolling_vols)
                    features['volatility_of_volatility'] = vol_of_vol
            
        except Exception as e:
            logger.error(f"Error calculating volatility features: {e}")
        
        return features
    
    def _calculate_risk_features(self, prices: List[float]) -> Dict[str, float]:
        """Calculate risk-based features"""
        features = {}
        
        if len(prices) < 10:
            return features
        
        try:
            prices_array = np.array(prices)
            returns = np.diff(prices_array) / prices_array[:-1]
            
            # Value at Risk (VaR)
            for confidence in [0.95, 0.99]:
                if len(returns) >= 20:
                    var = np.percentile(returns[-20:], (1 - confidence) * 100)
                    features[f'var_{int(confidence*100)}'] = var
            
            # Expected Shortfall (Conditional VaR)
            if len(returns) >= 20:
                var_95 = np.percentile(returns[-20:], 5)
                tail_returns = returns[-20:][returns[-20:] <= var_95]
                if len(tail_returns) > 0:
                    expected_shortfall = np.mean(tail_returns)
                    features['expected_shortfall_95'] = expected_shortfall
            
            # Maximum drawdown
            if len(prices_array) >= 10:
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown)
                features['max_drawdown'] = max_drawdown
            
            # Skewness and kurtosis
            if len(returns) >= 20:
                features['returns_skewness'] = stats.skew(returns[-20:])
                features['returns_kurtosis'] = stats.kurtosis(returns[-20:])
            
        except Exception as e:
            logger.error(f"Error calculating risk features: {e}")
        
        return features
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate exponential moving average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def prepare_ml_features(self, 
                          feature_sets: List[FuturesFeatureSet],
                          target_features: Optional[List[str]] = None,
                          scale_features: bool = True,
                          apply_pca: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for machine learning"""
        
        if not feature_sets:
            return np.array([]), []
        
        # Get all feature names
        all_feature_names = set()
        for fs in feature_sets:
            all_feature_names.update(fs.get_all_features().keys())
        
        feature_names = sorted(list(all_feature_names))
        
        if target_features:
            feature_names = [name for name in feature_names if name in target_features]
        
        # Create feature matrix
        feature_matrix = []
        for fs in feature_sets:
            features = fs.get_all_features()
            row = [features.get(name, 0.0) for name in feature_names]
            feature_matrix.append(row)
        
        feature_matrix = np.array(feature_matrix)
        
        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        if scale_features and len(feature_matrix) > 1:
            feature_matrix = self.scalers['standard'].fit_transform(feature_matrix)
        
        # Apply PCA
        if apply_pca and len(feature_matrix) > 1 and feature_matrix.shape[1] > 10:
            feature_matrix = self.pca.fit_transform(feature_matrix)
            feature_names = [f'PC_{i+1}' for i in range(feature_matrix.shape[1])]
        
        return feature_matrix, feature_names

# Example usage
if __name__ == "__main__":
    print("=== Futures Feature Engineering Demo ===")
    
    # Create sample data
    np.random.seed(42)
    n_days = 100
    
    # Generate sample price data
    base_price = 75.0
    price_changes = np.random.normal(0, 0.02, n_days)
    prices = [base_price]
    for change in price_changes:
        prices.append(prices[-1] * (1 + change))
    
    # Generate sample volume data
    volumes = np.random.lognormal(10, 0.5, n_days + 1).tolist()
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=n_days)
    timestamps = [start_date + timedelta(days=i) for i in range(n_days + 1)]
    
    # Sample contract data
    contract_prices = {
        'CL_2024_06': 75.20,
        'CL_2024_07': 75.45,
        'CL_2024_08': 75.70,
        'CL_2024_09': 75.90
    }
    
    contract_expiries = {
        'CL_2024_06': datetime(2024, 6, 20),
        'CL_2024_07': datetime(2024, 7, 20),
        'CL_2024_08': datetime(2024, 8, 20),
        'CL_2024_09': datetime(2024, 9, 20)
    }
    
    # Initialize feature engineer
    feature_engineer = FuturesFeatureEngineer(contract_type=FuturesContractType.ENERGY)
    
    print(f"\nEngineering features for {len(prices)} price points...")
    
    # Engineer features
    feature_set = feature_engineer.engineer_features(
        contract_symbol='CL_2024_06',
        prices=prices,
        volumes=volumes,
        timestamps=timestamps,
        contract_prices=contract_prices,
        contract_expiries=contract_expiries,
        spot_price=74.80
    )
    
    print(f"\n=== FEATURE SUMMARY ===")
    print(f"Contract: {feature_set.contract_symbol}")
    print(f"Timestamp: {feature_set.timestamp}")
    print(f"Total Features: {feature_set.feature_count}")
    print(f"Feature Quality: {feature_set.feature_quality:.3f}")
    print(f"Missing Data Ratio: {feature_set.missing_data_ratio:.3f}")
    
    # Display feature categories
    feature_categories = [
        ('Price Features', feature_set.price_features),
        ('Technical Features', feature_set.technical_features),
        ('Term Structure Features', feature_set.term_structure_features),
        ('Basis Features', feature_set.basis_features),
        ('Volatility Features', feature_set.volatility_features),
        ('Volume Features', feature_set.volume_features),
        ('Seasonality Features', feature_set.seasonality_features),
        ('Risk Features', feature_set.risk_features)
    ]
    
    for category_name, features in feature_categories:
        if features:
            print(f"\n=== {category_name.upper()} ===")
            for name, value in list(features.items())[:5]:  # Show first 5 features
                print(f"  {name}: {value:.4f}")
            if len(features) > 5:
                print(f"  ... and {len(features) - 5} more features")
    
    # Test ML preparation
    print(f"\n=== ML PREPARATION TEST ===")
    
    # Create multiple feature sets for ML preparation
    feature_sets = []
    for i in range(5):
        # Simulate different time periods
        end_idx = len(prices) - i * 10
        start_idx = max(0, end_idx - 50)
        
        subset_prices = prices[start_idx:end_idx]
        subset_volumes = volumes[start_idx:end_idx]
        subset_timestamps = timestamps[start_idx:end_idx]
        
        if len(subset_prices) > 10:
            fs = feature_engineer.engineer_features(
                contract_symbol='CL_2024_06',
                prices=subset_prices,
                volumes=subset_volumes,
                timestamps=subset_timestamps,
                contract_prices=contract_prices,
                contract_expiries=contract_expiries,
                spot_price=74.80
            )
            feature_sets.append(fs)
    
    # Prepare for ML
    feature_matrix, feature_names = feature_engineer.prepare_ml_features(
        feature_sets=feature_sets,
        scale_features=True,
        apply_pca=False
    )
    
    print(f"Feature Matrix Shape: {feature_matrix.shape}")
    print(f"Number of Features: {len(feature_names)}")
    print(f"Sample Feature Names: {feature_names[:10]}")
    
    print("\n=== Futures Feature Engineering Complete ===")