from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class AssetType(Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    INDEX = "index"
    CROSS_ASSET = "cross_asset"

class IndicatorCategory(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MACHINE_LEARNING = "machine_learning"
    STATISTICAL = "statistical"

@dataclass
class IndicatorResult:
    name: str
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory

class CryptoComprehensiveIndicators:
    """Comprehensive crypto-specific indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def stock_to_flow_model(self, data: pd.DataFrame, asset: str = "BTC") -> IndicatorResult:
        """Stock-to-Flow model for Bitcoin and other cryptocurrencies"""
        try:
            # Bitcoin-specific parameters (can be adjusted for other cryptos)
            if asset.upper() == "BTC":
                # Bitcoin halving schedule
                halving_dates = pd.to_datetime([
                    '2012-11-28', '2016-07-09', '2020-05-11', '2024-04-20'
                ])
                initial_reward = 50
                block_time = 10  # minutes
                blocks_per_day = 24 * 60 / block_time
            else:
                # Generic parameters for other cryptos
                halving_dates = pd.to_datetime(['2020-01-01'])  # Placeholder
                initial_reward = 25
                block_time = 2
                blocks_per_day = 24 * 60 / block_time
            
            # Calculate current reward based on halving schedule
            current_reward = initial_reward
            for halving_date in halving_dates:
                if data.index[-1] >= halving_date:
                    current_reward /= 2
            
            # Calculate daily production (flow)
            daily_production = current_reward * blocks_per_day
            
            # Estimate circulating supply (stock)
            # This is simplified - in practice, you'd use actual supply data
            days_since_genesis = (data.index - data.index[0]).days
            cumulative_supply = np.cumsum(np.full(len(data), daily_production))
            
            # Stock-to-Flow ratio
            stock_to_flow = cumulative_supply / daily_production
            
            # S2F model price prediction
            # Based on PlanB's model: Price = exp(3.3 * ln(S2F) - 15.7)
            predicted_price = np.exp(3.3 * np.log(stock_to_flow) - 15.7)
            
            # Calculate model deviation
            actual_price = data['close']
            price_deviation = (actual_price - predicted_price) / predicted_price * 100
            
            result_df = pd.DataFrame({
                'price': actual_price,
                'stock_to_flow': stock_to_flow,
                'predicted_price': predicted_price,
                'price_deviation': price_deviation,
                'daily_production': daily_production,
                'estimated_supply': cumulative_supply
            }, index=data.index)
            
            return IndicatorResult(
                name="Stock-to-Flow Model",
                values=result_df,
                metadata={
                    'asset': asset,
                    'current_s2f': stock_to_flow.iloc[-1],
                    'current_deviation': price_deviation.iloc[-1],
                    'model_r2': np.corrcoef(np.log(actual_price), np.log(predicted_price))[0,1]**2,
                    'interpretation': 'Higher S2F suggests higher scarcity and potential price appreciation'
                },
                confidence=0.75,
                timestamp=datetime.now(),
                asset_type=AssetType.CRYPTO,
                category=IndicatorCategory.FUNDAMENTAL
            )
            
        except Exception as e:
            logger.error(f"Error calculating Stock-to-Flow: {e}")
            return self._empty_result("Stock-to-Flow Model", AssetType.CRYPTO)
    
    def metcalfes_law(self, data: pd.DataFrame, network_data: Optional[pd.DataFrame] = None) -> IndicatorResult:
        """Metcalfe's Law valuation model"""
        try:
            # If network data not provided, estimate from price and volume
            if network_data is None:
                # Proxy for network activity using volume and price
                network_activity = np.sqrt(data['volume'] * data['close'])
                active_addresses = network_activity / network_activity.mean() * 1000000  # Normalized estimate
            else:
                active_addresses = network_data['active_addresses']
            
            # Metcalfe's Law: Network value proportional to square of users
            network_value = active_addresses ** 2
            
            # Normalize to price scale
            price_scale_factor = data['close'].mean() / network_value.mean()
            metcalfe_price = network_value * price_scale_factor
            
            # Calculate NVT-like ratio using Metcalfe's value
            nvt_metcalfe = data['close'] / (network_value / 1e6)  # Scaled for readability
            
            # Price deviation from Metcalfe's prediction
            metcalfe_deviation = (data['close'] - metcalfe_price) / metcalfe_price * 100
            
            result_df = pd.DataFrame({
                'price': data['close'],
                'active_addresses': active_addresses,
                'network_value': network_value,
                'metcalfe_price': metcalfe_price,
                'metcalfe_deviation': metcalfe_deviation,
                'nvt_metcalfe': nvt_metcalfe
            }, index=data.index)
            
            return IndicatorResult(
                name="Metcalfe's Law",
                values=result_df,
                metadata={
                    'current_network_value': network_value.iloc[-1],
                    'current_deviation': metcalfe_deviation.iloc[-1],
                    'correlation_with_price': np.corrcoef(data['close'], metcalfe_price)[0,1],
                    'interpretation': 'Network value grows with square of active users'
                },
                confidence=0.65,
                timestamp=datetime.now(),
                asset_type=AssetType.CRYPTO,
                category=IndicatorCategory.FUNDAMENTAL
            )
            
        except Exception as e:
            logger.error(f"Error calculating Metcalfe's Law: {e}")
            return self._empty_result("Metcalfe's Law", AssetType.CRYPTO)
    
    def nvt_nvm_ratio(self, data: pd.DataFrame, network_data: Optional[pd.DataFrame] = None) -> IndicatorResult:
        """Network Value to Transactions (NVT) and Network Value to Metcalfe (NVM) ratios"""
        try:
            # Calculate market cap (proxy)
            market_cap = data['close'] * data['volume']  # Simplified proxy
            
            # Transaction volume (use trading volume as proxy if on-chain data unavailable)
            if network_data is not None and 'transaction_volume' in network_data.columns:
                transaction_volume = network_data['transaction_volume']
            else:
                # Use trading volume as proxy for transaction volume
                transaction_volume = data['volume']
            
            # NVT Ratio
            nvt_ratio = market_cap / transaction_volume
            
            # NVT Signal (90-day MA of transaction volume)
            nvt_signal = market_cap / transaction_volume.rolling(90).mean()
            
            # NVM Ratio (Network Value to Metcalfe)
            if network_data is not None and 'active_addresses' in network_data.columns:
                active_addresses = network_data['active_addresses']
            else:
                # Estimate active addresses from volume
                active_addresses = np.sqrt(data['volume']) * 1000
            
            metcalfe_value = active_addresses ** 2
            nvm_ratio = market_cap / metcalfe_value
            
            # Z-scores for mean reversion signals
            nvt_zscore = (nvt_ratio - nvt_ratio.rolling(365).mean()) / nvt_ratio.rolling(365).std()
            nvm_zscore = (nvm_ratio - nvm_ratio.rolling(365).mean()) / nvm_ratio.rolling(365).std()
            
            result_df = pd.DataFrame({
                'price': data['close'],
                'market_cap': market_cap,
                'transaction_volume': transaction_volume,
                'nvt_ratio': nvt_ratio,
                'nvt_signal': nvt_signal,
                'nvm_ratio': nvm_ratio,
                'nvt_zscore': nvt_zscore,
                'nvm_zscore': nvm_zscore,
                'active_addresses': active_addresses
            }, index=data.index)
            
            return IndicatorResult(
                name="NVT/NVM Ratios",
                values=result_df,
                metadata={
                    'current_nvt': nvt_ratio.iloc[-1],
                    'current_nvm': nvm_ratio.iloc[-1],
                    'nvt_percentile': (nvt_ratio.iloc[-1] > nvt_ratio).mean() * 100,
                    'nvm_percentile': (nvm_ratio.iloc[-1] > nvm_ratio).mean() * 100,
                    'interpretation': 'High NVT/NVM suggests overvaluation, low suggests undervaluation'
                },
                confidence=0.70,
                timestamp=datetime.now(),
                asset_type=AssetType.CRYPTO,
                category=IndicatorCategory.FUNDAMENTAL
            )
            
        except Exception as e:
            logger.error(f"Error calculating NVT/NVM: {e}")
            return self._empty_result("NVT/NVM Ratios", AssetType.CRYPTO)
    
    def crypto_logarithmic_regression(self, data: pd.DataFrame) -> IndicatorResult:
        """Logarithmic regression model for long-term crypto price trends"""
        try:
            # Prepare data
            prices = data['close'].dropna()
            log_prices = np.log(prices)
            
            # Time variable (days since start)
            time_days = (prices.index - prices.index[0]).days
            
            # Fit logarithmic regression: log(price) = a + b*log(time) + c*time
            X = np.column_stack([
                np.ones(len(time_days)),  # Intercept
                np.log(time_days + 1),    # Log time component
                time_days / 365.25        # Linear time component (years)
            ])
            
            # Remove any invalid values
            valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(log_prices)
            X_clean = X[valid_mask]
            y_clean = log_prices[valid_mask]
            
            if len(X_clean) < 10:
                raise ValueError("Insufficient valid data points")
            
            # Fit regression
            coefficients = np.linalg.lstsq(X_clean, y_clean, rcond=None)[0]
            
            # Generate predictions
            log_predicted = X @ coefficients
            predicted_prices = np.exp(log_predicted)
            
            # Calculate regression bands (support and resistance)
            residuals = log_prices[valid_mask] - (X_clean @ coefficients)
            std_residual = np.std(residuals)
            
            upper_band = np.exp(log_predicted + 2 * std_residual)
            lower_band = np.exp(log_predicted - 2 * std_residual)
            
            # Price position within bands
            band_position = (prices - predicted_prices) / (upper_band - lower_band)
            
            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            result_df = pd.DataFrame({
                'price': prices,
                'log_regression': predicted_prices,
                'upper_band': upper_band,
                'lower_band': lower_band,
                'band_position': band_position,
                'price_deviation': (prices - predicted_prices) / predicted_prices * 100
            }, index=prices.index)
            
            return IndicatorResult(
                name="Crypto Logarithmic Regression",
                values=result_df,
                metadata={
                    'coefficients': coefficients.tolist(),
                    'r_squared': r_squared,
                    'current_band_position': band_position.iloc[-1],
                    'trend_strength': abs(coefficients[2]),  # Linear time coefficient
                    'interpretation': 'Band position > 0.5 suggests overvaluation, < -0.5 undervaluation'
                },
                confidence=0.80,
                timestamp=datetime.now(),
                asset_type=AssetType.CRYPTO,
                category=IndicatorCategory.STATISTICAL
            )
            
        except Exception as e:
            logger.error(f"Error calculating logarithmic regression: {e}")
            return self._empty_result("Crypto Logarithmic Regression", AssetType.CRYPTO)
    
    def _empty_result(self, name: str, asset_type: AssetType) -> IndicatorResult:
        """Return empty result for error cases"""
        return IndicatorResult(
            name=name,
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.TECHNICAL
        )
    
    async def calculate_indicator(self, indicator_name: str, data: pd.DataFrame, **kwargs) -> IndicatorResult:
        """Calculate specific crypto indicator based on name"""
        indicator_map = {
            'stock_to_flow': self.stock_to_flow_model,
            'metcalfes_law': self.metcalfes_law,
            'nvt_nvm': self.nvt_nvm_ratio,
            'crypto_log_regression': self.crypto_logarithmic_regression,
        }
        
        if indicator_name not in indicator_map:
            raise ValueError(f"Unknown crypto indicator: {indicator_name}")
        
        return indicator_map[indicator_name](data, **kwargs)