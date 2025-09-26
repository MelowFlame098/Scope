from typing import Optional, Dict, Any, List, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Statistical and ML imports
try:
    from scipy import stats
    from scipy.signal import find_peaks
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, f_regression
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy/Scikit-learn not available. Some features will be limited.")

logger = logging.getLogger(__name__)

class CryptoFeatureCategory(Enum):
    PRICE_ACTION = "price_action"
    TECHNICAL = "technical"
    ONCHAIN = "onchain"
    MARKET_STRUCTURE = "market_structure"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    CROSS_ASSET = "cross_asset"

@dataclass
class CryptoFeatureSet:
    """Comprehensive crypto feature set"""
    price_features: Dict[str, float]
    technical_features: Dict[str, float]
    onchain_features: Dict[str, float]
    market_structure_features: Dict[str, float]
    volatility_features: Dict[str, float]
    momentum_features: Dict[str, float]
    sentiment_features: Dict[str, float]
    macro_features: Dict[str, float]
    cross_asset_features: Dict[str, float]
    feature_importance: Dict[str, float]
    timestamp: datetime

class CryptoPriceActionFeatures:
    """Price action and basic OHLCV features for crypto"""
    
    @staticmethod
    def calculate_price_features(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive price action features"""
        features = {}
        
        if len(data) < 2:
            return features
        
        # Basic price features
        features['current_price'] = data['close'].iloc[-1]
        features['price_change_1d'] = (data['close'].iloc[-1] / data['close'].iloc[-2] - 1) if len(data) >= 2 else 0
        
        # Multi-timeframe returns
        for period in [1, 3, 7, 14, 30, 90]:
            if len(data) > period:
                features[f'return_{period}d'] = (data['close'].iloc[-1] / data['close'].iloc[-period-1] - 1)
        
        # Price position features
        if len(data) >= 20:
            high_20 = data['high'].tail(20).max()
            low_20 = data['low'].tail(20).min()
            features['price_position_20d'] = (data['close'].iloc[-1] - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
        
        # Gap analysis
        if len(data) >= 2:
            prev_close = data['close'].iloc[-2]
            curr_open = data['open'].iloc[-1]
            features['gap_size'] = (curr_open - prev_close) / prev_close
            features['gap_filled'] = 1 if ((curr_open > prev_close and data['low'].iloc[-1] <= prev_close) or 
                                          (curr_open < prev_close and data['high'].iloc[-1] >= prev_close)) else 0
        
        # Intraday features
        features['intraday_return'] = (data['close'].iloc[-1] / data['open'].iloc[-1] - 1)
        features['high_low_ratio'] = data['high'].iloc[-1] / data['low'].iloc[-1] if data['low'].iloc[-1] > 0 else 1
        features['close_position'] = ((data['close'].iloc[-1] - data['low'].iloc[-1]) / 
                                    (data['high'].iloc[-1] - data['low'].iloc[-1])) if data['high'].iloc[-1] != data['low'].iloc[-1] else 0.5
        
        return features

class CryptoTechnicalFeatures:
    """Technical analysis features for crypto"""
    
    @staticmethod
    def calculate_moving_averages(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate various moving averages and crossovers"""
        features = {}
        close = data['close']
        
        # Simple moving averages
        ma_periods = [7, 14, 21, 50, 100, 200]
        for period in ma_periods:
            if len(close) >= period:
                ma = close.tail(period).mean()
                features[f'sma_{period}'] = ma
                features[f'price_to_sma_{period}'] = close.iloc[-1] / ma - 1
        
        # Exponential moving averages
        ema_periods = [12, 26, 50]
        for period in ema_periods:
            if len(close) >= period:
                ema = close.ewm(span=period).mean().iloc[-1]
                features[f'ema_{period}'] = ema
                features[f'price_to_ema_{period}'] = close.iloc[-1] / ema - 1
        
        # Moving average crossovers
        if len(close) >= 50:
            sma_20 = close.tail(20).mean()
            sma_50 = close.tail(50).mean()
            features['ma_crossover_20_50'] = sma_20 / sma_50 - 1
        
        return features
    
    @staticmethod
    def calculate_momentum_indicators(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate momentum-based technical indicators"""
        features = {}
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data.get('volume', pd.Series([1] * len(data)))
        
        # RSI
        if len(close) >= 14:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi_14'] = rsi.iloc[-1]
            features['rsi_overbought'] = 1 if rsi.iloc[-1] > 70 else 0
            features['rsi_oversold'] = 1 if rsi.iloc[-1] < 30 else 0
        
        # Stochastic Oscillator
        if len(data) >= 14:
            lowest_low = low.rolling(window=14).min()
            highest_high = high.rolling(window=14).max()
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            features['stoch_k'] = k_percent.iloc[-1]
            features['stoch_d'] = k_percent.rolling(window=3).mean().iloc[-1]
        
        # MACD
        if len(close) >= 26:
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            histogram = macd - signal
            
            features['macd'] = macd.iloc[-1]
            features['macd_signal'] = signal.iloc[-1]
            features['macd_histogram'] = histogram.iloc[-1]
            features['macd_bullish'] = 1 if macd.iloc[-1] > signal.iloc[-1] else 0
        
        # Williams %R
        if len(data) >= 14:
            highest_high = high.rolling(window=14).max()
            lowest_low = low.rolling(window=14).min()
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            features['williams_r'] = williams_r.iloc[-1]
        
        # Commodity Channel Index (CCI)
        if len(data) >= 20:
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            features['cci_20'] = cci.iloc[-1]
        
        return features
    
    @staticmethod
    def calculate_volatility_indicators(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate volatility-based indicators"""
        features = {}
        close = data['close']
        high = data['high']
        low = data['low']
        
        # Bollinger Bands
        if len(close) >= 20:
            sma_20 = close.rolling(window=20).mean()
            std_20 = close.rolling(window=20).std()
            
            bb_upper = sma_20 + (2 * std_20)
            bb_lower = sma_20 - (2 * std_20)
            
            features['bb_upper'] = bb_upper.iloc[-1]
            features['bb_lower'] = bb_lower.iloc[-1]
            features['bb_width'] = (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / sma_20.iloc[-1]
            features['bb_position'] = (close.iloc[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # Average True Range (ATR)
        if len(data) >= 14:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            features['atr_14'] = atr.iloc[-1]
            features['atr_ratio'] = atr.iloc[-1] / close.iloc[-1]
        
        # Keltner Channels
        if len(data) >= 20:
            ema_20 = close.ewm(span=20).mean()
            if 'atr_14' in features:
                kc_upper = ema_20 + (2 * features['atr_14'])
                kc_lower = ema_20 - (2 * features['atr_14'])
                features['kc_position'] = (close.iloc[-1] - kc_lower) / (kc_upper - kc_lower)
        
        return features

class CryptoOnChainFeatures:
    """On-chain analysis features for crypto"""
    
    @staticmethod
    def calculate_network_features(onchain_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate network-based features"""
        features = {}
        
        if not onchain_data:
            # Return default/mock values if no on-chain data available
            return {
                'active_addresses': 0.5,
                'transaction_count': 0.5,
                'hash_rate': 0.5,
                'difficulty': 0.5,
                'network_value': 0.5
            }
        
        # Active addresses
        if 'active_addresses' in onchain_data:
            features['active_addresses'] = onchain_data['active_addresses']
            features['active_addresses_ma7'] = np.mean(onchain_data.get('active_addresses_history', [features['active_addresses']])[-7:])
        
        # Transaction metrics
        if 'transaction_count' in onchain_data:
            features['transaction_count'] = onchain_data['transaction_count']
            features['tx_count_growth'] = onchain_data.get('tx_count_growth', 0)
        
        # Mining metrics (for Bitcoin)
        if 'hash_rate' in onchain_data:
            features['hash_rate'] = onchain_data['hash_rate']
            features['hash_rate_ma30'] = np.mean(onchain_data.get('hash_rate_history', [features['hash_rate']])[-30:])
        
        if 'difficulty' in onchain_data:
            features['difficulty'] = onchain_data['difficulty']
            features['difficulty_adjustment'] = onchain_data.get('difficulty_adjustment', 0)
        
        # Network value metrics
        if 'market_cap' in onchain_data and 'realized_cap' in onchain_data:
            features['mvrv_ratio'] = onchain_data['market_cap'] / onchain_data['realized_cap']
        
        return features
    
    @staticmethod
    def calculate_flow_features(flow_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate exchange flow and holder behavior features"""
        features = {}
        
        if not flow_data:
            return {
                'exchange_inflow': 0.0,
                'exchange_outflow': 0.0,
                'net_flow': 0.0,
                'whale_activity': 0.5
            }
        
        # Exchange flows
        if 'exchange_inflow' in flow_data and 'exchange_outflow' in flow_data:
            features['exchange_inflow'] = flow_data['exchange_inflow']
            features['exchange_outflow'] = flow_data['exchange_outflow']
            features['net_flow'] = flow_data['exchange_outflow'] - flow_data['exchange_inflow']
            features['flow_ratio'] = flow_data['exchange_outflow'] / flow_data['exchange_inflow'] if flow_data['exchange_inflow'] > 0 else 1
        
        # Whale activity
        if 'large_transactions' in flow_data:
            features['whale_activity'] = flow_data['large_transactions']
        
        # HODLer behavior
        if 'long_term_holders' in flow_data:
            features['lth_supply'] = flow_data['long_term_holders']
        
        return features

class CryptoMarketStructureFeatures:
    """Market microstructure features for crypto"""
    
    @staticmethod
    def calculate_liquidity_features(data: pd.DataFrame) -> Dict[str, float]:
        """Calculate liquidity and market structure features"""
        features = {}
        
        if 'volume' not in data.columns:
            return features
        
        close = data['close']
        volume = data['volume']
        high = data['high']
        low = data['low']
        
        # Volume features
        if len(volume) >= 20:
            features['volume_sma_20'] = volume.tail(20).mean()
            features['volume_ratio'] = volume.iloc[-1] / features['volume_sma_20']
        
        # Price-Volume features
        if len(data) >= 2:
            price_change = close.pct_change().iloc[-1]
            volume_change = volume.pct_change().iloc[-1]
            features['price_volume_correlation'] = price_change * volume_change
        
        # On-Balance Volume (OBV)
        if len(data) >= 10:
            obv = np.where(close > close.shift(1), volume, 
                          np.where(close < close.shift(1), -volume, 0)).cumsum()
            features['obv'] = obv[-1]
            features['obv_ma10'] = np.mean(obv[-10:])
        
        # Volume Weighted Average Price (VWAP)
        if len(data) >= 1:
            typical_price = (high + low + close) / 3
            vwap = (typical_price * volume).sum() / volume.sum()
            features['vwap'] = vwap
            features['price_to_vwap'] = close.iloc[-1] / vwap - 1
        
        # Accumulation/Distribution Line
        if len(data) >= 10:
            money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
            money_flow_multiplier = money_flow_multiplier.fillna(0)
            money_flow_volume = money_flow_multiplier * volume
            ad_line = money_flow_volume.cumsum()
            features['ad_line'] = ad_line.iloc[-1]
        
        return features
    
    @staticmethod
    def calculate_spread_features(orderbook_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate bid-ask spread and orderbook features"""
        features = {}
        
        if not orderbook_data:
            return {
                'bid_ask_spread': 0.001,  # Default 0.1% spread
                'orderbook_imbalance': 0.0,
                'market_depth': 0.5
            }
        
        # Bid-ask spread
        if 'best_bid' in orderbook_data and 'best_ask' in orderbook_data:
            bid = orderbook_data['best_bid']
            ask = orderbook_data['best_ask']
            mid_price = (bid + ask) / 2
            features['bid_ask_spread'] = (ask - bid) / mid_price
        
        # Order book imbalance
        if 'bid_volume' in orderbook_data and 'ask_volume' in orderbook_data:
            bid_vol = orderbook_data['bid_volume']
            ask_vol = orderbook_data['ask_volume']
            features['orderbook_imbalance'] = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        
        # Market depth
        if 'total_bid_volume' in orderbook_data and 'total_ask_volume' in orderbook_data:
            total_volume = orderbook_data['total_bid_volume'] + orderbook_data['total_ask_volume']
            features['market_depth'] = total_volume
        
        return features

class CryptoSentimentFeatures:
    """Sentiment and social media features for crypto"""
    
    @staticmethod
    def calculate_sentiment_features(sentiment_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate sentiment-based features"""
        features = {}
        
        if not sentiment_data:
            return {
                'social_sentiment': 0.5,
                'fear_greed_index': 50.0,
                'news_sentiment': 0.5,
                'reddit_activity': 0.5
            }
        
        # Social media sentiment
        if 'twitter_sentiment' in sentiment_data:
            features['twitter_sentiment'] = sentiment_data['twitter_sentiment']
        
        if 'reddit_sentiment' in sentiment_data:
            features['reddit_sentiment'] = sentiment_data['reddit_sentiment']
        
        # Fear & Greed Index
        if 'fear_greed_index' in sentiment_data:
            features['fear_greed_index'] = sentiment_data['fear_greed_index']
            features['extreme_fear'] = 1 if sentiment_data['fear_greed_index'] < 25 else 0
            features['extreme_greed'] = 1 if sentiment_data['fear_greed_index'] > 75 else 0
        
        # News sentiment
        if 'news_sentiment' in sentiment_data:
            features['news_sentiment'] = sentiment_data['news_sentiment']
        
        # Social activity metrics
        if 'social_volume' in sentiment_data:
            features['social_volume'] = sentiment_data['social_volume']
        
        return features

class CryptoMacroFeatures:
    """Macro-economic features affecting crypto"""
    
    @staticmethod
    def calculate_macro_features(macro_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """Calculate macro-economic features"""
        features = {}
        
        if not macro_data:
            return {
                'dxy_index': 100.0,
                'gold_price': 2000.0,
                'vix_index': 20.0,
                'interest_rates': 0.05
            }
        
        # Dollar strength
        if 'dxy' in macro_data:
            features['dxy_index'] = macro_data['dxy']
            features['dxy_change'] = macro_data.get('dxy_change', 0)
        
        # Traditional safe havens
        if 'gold_price' in macro_data:
            features['gold_price'] = macro_data['gold_price']
            features['gold_change'] = macro_data.get('gold_change', 0)
        
        # Market volatility
        if 'vix' in macro_data:
            features['vix_index'] = macro_data['vix']
            features['high_vix'] = 1 if macro_data['vix'] > 30 else 0
        
        # Interest rates
        if 'fed_rate' in macro_data:
            features['fed_rate'] = macro_data['fed_rate']
        
        # Inflation
        if 'cpi' in macro_data:
            features['cpi'] = macro_data['cpi']
        
        return features

class CryptoCrossAssetFeatures:
    """Cross-asset correlation features"""
    
    @staticmethod
    def calculate_correlation_features(price_data: pd.DataFrame, 
                                     other_assets: Optional[Dict[str, pd.Series]] = None) -> Dict[str, float]:
        """Calculate cross-asset correlation features"""
        features = {}
        
        if not other_assets or len(price_data) < 30:
            return {
                'btc_correlation': 0.8,  # Default high correlation with BTC
                'eth_correlation': 0.7,
                'sp500_correlation': 0.3,
                'gold_correlation': 0.1
            }
        
        crypto_returns = price_data['close'].pct_change().dropna()
        
        # Calculate correlations with other assets
        for asset_name, asset_prices in other_assets.items():
            if len(asset_prices) >= len(crypto_returns):
                asset_returns = asset_prices.pct_change().dropna()
                
                # Align the series
                common_dates = crypto_returns.index.intersection(asset_returns.index)
                if len(common_dates) >= 20:
                    corr = crypto_returns.loc[common_dates].corr(asset_returns.loc[common_dates])
                    features[f'{asset_name}_correlation'] = corr if not np.isnan(corr) else 0
                    
                    # Rolling correlation
                    rolling_corr = crypto_returns.loc[common_dates].rolling(30).corr(asset_returns.loc[common_dates])
                    features[f'{asset_name}_rolling_corr'] = rolling_corr.iloc[-1] if not np.isnan(rolling_corr.iloc[-1]) else 0
        
        return features

class CryptoAdvancedFeatureEngineering:
    """Advanced feature engineering pipeline for crypto"""
    
    def __init__(self, 
                 feature_selection: bool = True,
                 scaling_method: str = 'robust',
                 pca_components: Optional[int] = None):
        
        self.feature_selection = feature_selection
        self.scaling_method = scaling_method
        self.pca_components = pca_components
        
        # Initialize components
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        
        # Feature calculators
        self.price_calculator = CryptoPriceActionFeatures()
        self.technical_calculator = CryptoTechnicalFeatures()
        self.onchain_calculator = CryptoOnChainFeatures()
        self.market_calculator = CryptoMarketStructureFeatures()
        self.sentiment_calculator = CryptoSentimentFeatures()
        self.macro_calculator = CryptoMacroFeatures()
        self.cross_asset_calculator = CryptoCrossAssetFeatures()
    
    def extract_all_features(self, 
                           price_data: pd.DataFrame,
                           onchain_data: Optional[Dict[str, Any]] = None,
                           sentiment_data: Optional[Dict[str, Any]] = None,
                           macro_data: Optional[Dict[str, Any]] = None,
                           other_assets: Optional[Dict[str, pd.Series]] = None,
                           orderbook_data: Optional[Dict[str, Any]] = None,
                           flow_data: Optional[Dict[str, Any]] = None) -> CryptoFeatureSet:
        """Extract comprehensive feature set"""
        
        # Extract features from each category
        price_features = self.price_calculator.calculate_price_features(price_data)
        
        # Technical features
        ma_features = self.technical_calculator.calculate_moving_averages(price_data)
        momentum_features = self.technical_calculator.calculate_momentum_indicators(price_data)
        volatility_features = self.technical_calculator.calculate_volatility_indicators(price_data)
        technical_features = {**ma_features, **momentum_features, **volatility_features}
        
        # On-chain features
        network_features = self.onchain_calculator.calculate_network_features(onchain_data)
        flow_features = self.onchain_calculator.calculate_flow_features(flow_data)
        onchain_features = {**network_features, **flow_features}
        
        # Market structure features
        liquidity_features = self.market_calculator.calculate_liquidity_features(price_data)
        spread_features = self.market_calculator.calculate_spread_features(orderbook_data)
        market_structure_features = {**liquidity_features, **spread_features}
        
        # Other features
        sentiment_features = self.sentiment_calculator.calculate_sentiment_features(sentiment_data)
        macro_features = self.macro_calculator.calculate_macro_features(macro_data)
        cross_asset_features = self.cross_asset_calculator.calculate_correlation_features(price_data, other_assets)
        
        # Calculate feature importance (simplified)
        all_features = {
            **price_features, **technical_features, **onchain_features,
            **market_structure_features, **sentiment_features, 
            **macro_features, **cross_asset_features
        }
        
        feature_importance = self._calculate_feature_importance(all_features)
        
        return CryptoFeatureSet(
            price_features=price_features,
            technical_features=technical_features,
            onchain_features=onchain_features,
            market_structure_features=market_structure_features,
            volatility_features=volatility_features,
            momentum_features=momentum_features,
            sentiment_features=sentiment_features,
            macro_features=macro_features,
            cross_asset_features=cross_asset_features,
            feature_importance=feature_importance,
            timestamp=datetime.now()
        )
    
    def _calculate_feature_importance(self, features: Dict[str, float]) -> Dict[str, float]:
        """Calculate feature importance scores"""
        # Simplified importance based on feature categories and volatility
        importance = {}
        
        for feature_name, value in features.items():
            # Base importance
            base_importance = 0.5
            
            # Category-based importance
            if any(keyword in feature_name.lower() for keyword in ['price', 'return', 'change']):
                base_importance = 0.9
            elif any(keyword in feature_name.lower() for keyword in ['volume', 'rsi', 'macd']):
                base_importance = 0.8
            elif any(keyword in feature_name.lower() for keyword in ['onchain', 'flow', 'whale']):
                base_importance = 0.7
            elif any(keyword in feature_name.lower() for keyword in ['sentiment', 'fear']):
                base_importance = 0.6
            
            # Adjust based on value magnitude (normalized)
            if abs(value) > 1:
                base_importance *= 1.1
            elif abs(value) < 0.1:
                base_importance *= 0.9
            
            importance[feature_name] = min(1.0, base_importance)
        
        return importance
    
    def prepare_features_for_ml(self, feature_set: CryptoFeatureSet) -> Tuple[np.ndarray, List[str]]:
        """Prepare features for machine learning"""
        # Combine all features
        all_features = {
            **feature_set.price_features,
            **feature_set.technical_features,
            **feature_set.onchain_features,
            **feature_set.market_structure_features,
            **feature_set.volatility_features,
            **feature_set.momentum_features,
            **feature_set.sentiment_features,
            **feature_set.macro_features,
            **feature_set.cross_asset_features
        }
        
        # Convert to arrays
        feature_names = list(all_features.keys())
        feature_values = np.array([all_features[name] for name in feature_names]).reshape(1, -1)
        
        # Handle NaN values
        feature_values = np.nan_to_num(feature_values, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply scaling
        if self.scaler is None:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:  # robust
                self.scaler = RobustScaler()
            
            # Fit scaler (in practice, you'd fit on training data)
            self.scaler.fit(feature_values)
        
        scaled_features = self.scaler.transform(feature_values)
        
        # Apply PCA if specified
        if self.pca_components and SCIPY_AVAILABLE:
            if self.pca is None:
                self.pca = PCA(n_components=self.pca_components)
                self.pca.fit(scaled_features)
            
            scaled_features = self.pca.transform(scaled_features)
            feature_names = [f'PC_{i+1}' for i in range(self.pca_components)]
        
        return scaled_features[0], feature_names

# Example usage
if __name__ == "__main__":
    # Create sample crypto data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Simulate Bitcoin OHLCV data
    n_days = len(dates)
    returns = np.random.randn(n_days) * 0.03
    prices = 45000 * np.exp(np.cumsum(returns))
    
    crypto_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n_days) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(n_days)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(n_days)) * 0.02),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_days)
    }, index=dates)
    
    # Sample additional data
    onchain_data = {
        'active_addresses': 950000,
        'transaction_count': 250000,
        'hash_rate': 450000000,
        'market_cap': 900000000000,
        'realized_cap': 600000000000
    }
    
    sentiment_data = {
        'fear_greed_index': 65,
        'twitter_sentiment': 0.6,
        'reddit_sentiment': 0.55
    }
    
    macro_data = {
        'dxy': 103.5,
        'vix': 18.2,
        'gold_price': 2050.0
    }
    
    # Initialize feature engineering
    feature_engineer = CryptoAdvancedFeatureEngineering(
        feature_selection=True,
        scaling_method='robust',
        pca_components=None
    )
    
    # Extract comprehensive features
    feature_set = feature_engineer.extract_all_features(
        price_data=crypto_data,
        onchain_data=onchain_data,
        sentiment_data=sentiment_data,
        macro_data=macro_data
    )
    
    print("\n=== Crypto Advanced Feature Engineering ===")
    
    print(f"\n--- PRICE FEATURES ({len(feature_set.price_features)}) ---")
    for name, value in list(feature_set.price_features.items())[:5]:
        print(f"{name}: {value:.4f}")
    
    print(f"\n--- TECHNICAL FEATURES ({len(feature_set.technical_features)}) ---")
    for name, value in list(feature_set.technical_features.items())[:5]:
        print(f"{name}: {value:.4f}")
    
    print(f"\n--- ON-CHAIN FEATURES ({len(feature_set.onchain_features)}) ---")
    for name, value in list(feature_set.onchain_features.items())[:5]:
        print(f"{name}: {value:.4f}")
    
    print(f"\n--- SENTIMENT FEATURES ({len(feature_set.sentiment_features)}) ---")
    for name, value in feature_set.sentiment_features.items():
        print(f"{name}: {value:.4f}")
    
    # Prepare for ML
    ml_features, feature_names = feature_engineer.prepare_features_for_ml(feature_set)
    
    print(f"\n--- ML-READY FEATURES ---")
    print(f"Total features: {len(ml_features)}")
    print(f"Feature vector shape: {ml_features.shape}")
    
    # Top important features
    top_features = sorted(feature_set.feature_importance.items(), 
                         key=lambda x: x[1], reverse=True)[:10]
    
    print(f"\n--- TOP 10 IMPORTANT FEATURES ---")
    for name, importance in top_features:
        print(f"{name}: {importance:.3f}")