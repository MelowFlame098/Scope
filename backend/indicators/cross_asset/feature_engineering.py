from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class CrossAssetFeatureCategory(Enum):
    """Categories of cross-asset features"""
    PRICE_ACTION = "price_action"
    TECHNICAL = "technical"
    VOLATILITY = "volatility"
    CORRELATION = "correlation"
    FACTOR_MODELS = "factor_models"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    REGIME = "regime"
    MACRO_ECONOMIC = "macro_economic"
    SENTIMENT = "sentiment"
    FLOW = "flow"
    RISK = "risk"

class CrossAssetScalingMethod(Enum):
    """Scaling methods for cross-asset features"""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    QUANTILE = "quantile"
    NONE = "none"

@dataclass
class CrossAssetFeatureSet:
    """Complete set of cross-asset features"""
    price_features: Dict[str, pd.DataFrame]
    technical_features: Dict[str, pd.DataFrame]
    volatility_features: Dict[str, pd.DataFrame]
    correlation_features: pd.DataFrame
    factor_features: pd.DataFrame
    momentum_features: Dict[str, pd.DataFrame]
    mean_reversion_features: Dict[str, pd.DataFrame]
    regime_features: pd.DataFrame
    macro_features: Optional[pd.DataFrame] = None
    sentiment_features: Optional[pd.DataFrame] = None
    flow_features: Optional[pd.DataFrame] = None
    risk_features: Optional[pd.DataFrame] = None
    feature_metadata: Dict[str, Any] = None
    timestamp: datetime = None

class CrossAssetPriceActionFeatures:
    """Price action features for cross-asset analysis"""
    
    def __init__(self, lookback_periods: List[int] = [5, 10, 20, 50, 100]):
        self.lookback_periods = lookback_periods
    
    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate price action features for each asset class"""
        features = {}
        
        for asset_class, df in data.items():
            if 'close' not in df.columns:
                continue
            
            asset_features = pd.DataFrame(index=df.index)
            
            # Returns
            returns = df['close'].pct_change()
            asset_features['returns'] = returns
            
            # Log returns
            asset_features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Price ratios and levels
            for period in self.lookback_periods:
                if len(df) > period:
                    # Price ratios
                    asset_features[f'price_ratio_{period}d'] = df['close'] / df['close'].shift(period)
                    
                    # High/Low ratios
                    if 'high' in df.columns and 'low' in df.columns:
                        asset_features[f'hl_ratio_{period}d'] = (
                            df['high'].rolling(period).max() / df['low'].rolling(period).min()
                        )
                    
                    # Close position in range
                    if 'high' in df.columns and 'low' in df.columns:
                        high_period = df['high'].rolling(period).max()
                        low_period = df['low'].rolling(period).min()
                        asset_features[f'close_position_{period}d'] = (
                            (df['close'] - low_period) / (high_period - low_period + 1e-8)
                        )
            
            # Gap analysis
            if 'open' in df.columns:
                asset_features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
                asset_features['gap_filled'] = (
                    ((df['low'] <= df['close'].shift(1)) & (df['gap'] > 0)) |
                    ((df['high'] >= df['close'].shift(1)) & (df['gap'] < 0))
                ).astype(int)
            
            # Volume-price features
            if 'volume' in df.columns:
                asset_features['volume_price_trend'] = (
                    returns * df['volume'].pct_change()
                )
                
                # Price-volume correlation
                for period in [10, 20, 50]:
                    if len(df) > period:
                        corr = returns.rolling(period).corr(df['volume'].pct_change())
                        asset_features[f'price_volume_corr_{period}d'] = corr
            
            features[asset_class] = asset_features.dropna(how='all')
        
        return features

class CrossAssetTechnicalFeatures:
    """Technical indicators for cross-asset analysis"""
    
    def __init__(self, periods: List[int] = [10, 20, 50, 100, 200]):
        self.periods = periods
    
    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate technical features for each asset class"""
        features = {}
        
        for asset_class, df in data.items():
            if 'close' not in df.columns:
                continue
            
            asset_features = pd.DataFrame(index=df.index)
            
            # Moving averages and ratios
            for period in self.periods:
                if len(df) > period:
                    ma = df['close'].rolling(period).mean()
                    asset_features[f'ma_{period}'] = ma
                    asset_features[f'price_ma_ratio_{period}'] = df['close'] / ma
                    
                    # EMA
                    ema = df['close'].ewm(span=period).mean()
                    asset_features[f'ema_{period}'] = ema
                    asset_features[f'price_ema_ratio_{period}'] = df['close'] / ema
            
            # RSI
            for period in [14, 30]:
                rsi = self._calculate_rsi(df['close'], period)
                asset_features[f'rsi_{period}'] = rsi
            
            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(df['close'])
            asset_features['macd_line'] = macd_line
            asset_features['macd_signal'] = macd_signal
            asset_features['macd_histogram'] = macd_histogram
            
            # Bollinger Bands
            for period in [20, 50]:
                bb_upper, bb_middle, bb_lower, bb_width, bb_position = self._calculate_bollinger_bands(
                    df['close'], period
                )
                asset_features[f'bb_upper_{period}'] = bb_upper
                asset_features[f'bb_lower_{period}'] = bb_lower
                asset_features[f'bb_width_{period}'] = bb_width
                asset_features[f'bb_position_{period}'] = bb_position
            
            # Stochastic Oscillator
            if 'high' in df.columns and 'low' in df.columns:
                stoch_k, stoch_d = self._calculate_stochastic(df, 14, 3)
                asset_features['stoch_k'] = stoch_k
                asset_features['stoch_d'] = stoch_d
            
            # Williams %R
            if 'high' in df.columns and 'low' in df.columns:
                williams_r = self._calculate_williams_r(df, 14)
                asset_features['williams_r'] = williams_r
            
            # Average True Range (ATR)
            if 'high' in df.columns and 'low' in df.columns:
                atr = self._calculate_atr(df, 14)
                asset_features['atr'] = atr
                asset_features['atr_ratio'] = atr / df['close']
            
            features[asset_class] = asset_features.dropna(how='all')
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        width = (upper - lower) / ma
        position = (prices - lower) / (upper - lower)
        return upper, ma, lower, width, position
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """Calculate Stochastic Oscillator"""
        lowest_low = df['low'].rolling(k_period).min()
        highest_high = df['high'].rolling(k_period).max()
        k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14):
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(period).max()
        lowest_low = df['low'].rolling(period).min()
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        return williams_r
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close_prev = np.abs(df['high'] - df['close'].shift())
        low_close_prev = np.abs(df['low'] - df['close'].shift())
        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(period).mean()
        return atr

class CrossAssetVolatilityFeatures:
    """Volatility features for cross-asset analysis"""
    
    def __init__(self, periods: List[int] = [5, 10, 20, 50, 100]):
        self.periods = periods
    
    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate volatility features for each asset class"""
        features = {}
        
        for asset_class, df in data.items():
            if 'close' not in df.columns:
                continue
            
            asset_features = pd.DataFrame(index=df.index)
            returns = df['close'].pct_change()
            
            # Historical volatility
            for period in self.periods:
                if len(df) > period:
                    vol = returns.rolling(period).std() * np.sqrt(252)
                    asset_features[f'volatility_{period}d'] = vol
                    
                    # Volatility of volatility
                    vol_of_vol = vol.rolling(period).std()
                    asset_features[f'vol_of_vol_{period}d'] = vol_of_vol
                    
                    # Volatility ratios
                    if period > 5:
                        short_vol = returns.rolling(5).std() * np.sqrt(252)
                        asset_features[f'vol_ratio_{period}d'] = short_vol / vol
            
            # GARCH-like features
            asset_features['squared_returns'] = returns ** 2
            asset_features['abs_returns'] = np.abs(returns)
            
            # Exponentially weighted volatility
            for alpha in [0.94, 0.97, 0.99]:
                ewm_vol = returns.ewm(alpha=alpha).std() * np.sqrt(252)
                asset_features[f'ewm_vol_{int(alpha*100)}'] = ewm_vol
            
            # Realized volatility components
            if 'high' in df.columns and 'low' in df.columns and 'open' in df.columns:
                # Parkinson volatility
                parkinson_vol = self._calculate_parkinson_volatility(df)
                asset_features['parkinson_vol'] = parkinson_vol
                
                # Garman-Klass volatility
                gk_vol = self._calculate_garman_klass_volatility(df)
                asset_features['garman_klass_vol'] = gk_vol
                
                # Rogers-Satchell volatility
                rs_vol = self._calculate_rogers_satchell_volatility(df)
                asset_features['rogers_satchell_vol'] = rs_vol
            
            # Volatility clustering
            for period in [10, 20]:
                if len(returns) > period:
                    vol_clustering = self._calculate_volatility_clustering(returns, period)
                    asset_features[f'vol_clustering_{period}d'] = vol_clustering
            
            features[asset_class] = asset_features.dropna(how='all')
        
        return features
    
    def _calculate_parkinson_volatility(self, df: pd.DataFrame, period: int = 20):
        """Calculate Parkinson volatility estimator"""
        hl_ratio = np.log(df['high'] / df['low'])
        parkinson = np.sqrt((1 / (4 * np.log(2))) * hl_ratio ** 2)
        return parkinson.rolling(period).mean() * np.sqrt(252)
    
    def _calculate_garman_klass_volatility(self, df: pd.DataFrame, period: int = 20):
        """Calculate Garman-Klass volatility estimator"""
        hl = np.log(df['high'] / df['low'])
        co = np.log(df['close'] / df['open'])
        gk = 0.5 * hl ** 2 - (2 * np.log(2) - 1) * co ** 2
        return np.sqrt(gk.rolling(period).mean() * 252)
    
    def _calculate_rogers_satchell_volatility(self, df: pd.DataFrame, period: int = 20):
        """Calculate Rogers-Satchell volatility estimator"""
        ho = np.log(df['high'] / df['open'])
        hc = np.log(df['high'] / df['close'])
        lo = np.log(df['low'] / df['open'])
        lc = np.log(df['low'] / df['close'])
        rs = ho * hc + lo * lc
        return np.sqrt(rs.rolling(period).mean() * 252)
    
    def _calculate_volatility_clustering(self, returns: pd.Series, period: int):
        """Calculate volatility clustering measure"""
        abs_returns = np.abs(returns)
        vol_autocorr = abs_returns.rolling(period).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
        )
        return vol_autocorr

class CrossAssetCorrelationFeatures:
    """Correlation features for cross-asset analysis"""
    
    def __init__(self, periods: List[int] = [10, 20, 50, 100, 200]):
        self.periods = periods
    
    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate correlation features across asset classes"""
        # Prepare returns data
        returns_data = {}
        for asset_class, df in data.items():
            if 'close' in df.columns:
                returns_data[asset_class] = df['close'].pct_change()
        
        if len(returns_data) < 2:
            return pd.DataFrame()
        
        # Align all return series
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if returns_df.empty:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=returns_df.index)
        asset_pairs = [(i, j) for i in returns_df.columns for j in returns_df.columns if i < j]
        
        # Rolling correlations
        for period in self.periods:
            if len(returns_df) > period:
                for asset1, asset2 in asset_pairs:
                    corr = returns_df[asset1].rolling(period).corr(returns_df[asset2])
                    features[f'corr_{asset1}_{asset2}_{period}d'] = corr
        
        # Dynamic conditional correlation (DCC) features
        features.update(self._calculate_dcc_features(returns_df))
        
        # Correlation regime features
        features.update(self._calculate_correlation_regime_features(returns_df))
        
        # Cross-asset momentum correlations
        features.update(self._calculate_momentum_correlations(returns_df))
        
        # Volatility correlations
        features.update(self._calculate_volatility_correlations(returns_df))
        
        return features.dropna(how='all')
    
    def _calculate_dcc_features(self, returns_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Dynamic Conditional Correlation features"""
        features = {}
        
        try:
            # Simplified DCC implementation
            # In practice, you would use a proper DCC-GARCH model
            
            # Calculate standardized residuals (simplified)
            standardized_returns = returns_df.div(returns_df.rolling(20).std(), axis=0)
            
            # Dynamic correlation proxy
            for i, asset1 in enumerate(returns_df.columns):
                for j, asset2 in enumerate(returns_df.columns[i+1:], i+1):
                    # Rolling correlation with exponential weighting
                    ewm_corr = standardized_returns[asset1].ewm(span=30).corr(
                        standardized_returns[asset2].ewm(span=30)
                    )
                    features[f'dcc_{asset1}_{asset2}'] = ewm_corr
        
        except Exception as e:
            print(f"DCC calculation error: {e}")
        
        return features
    
    def _calculate_correlation_regime_features(self, returns_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate correlation regime features"""
        features = {}
        
        try:
            # Average correlation
            corr_matrix = returns_df.rolling(50).corr()
            
            # Extract upper triangle correlations for each time period
            n_assets = len(returns_df.columns)
            avg_corr_series = []
            max_corr_series = []
            min_corr_series = []
            
            for date in returns_df.index[49:]:  # Start after 50-day window
                date_corr = corr_matrix.loc[date]
                if not date_corr.isna().all().all():
                    # Get upper triangle values
                    upper_triangle = np.triu(date_corr.values, k=1)
                    upper_values = upper_triangle[upper_triangle != 0]
                    
                    if len(upper_values) > 0:
                        avg_corr_series.append(np.mean(upper_values))
                        max_corr_series.append(np.max(upper_values))
                        min_corr_series.append(np.min(upper_values))
                    else:
                        avg_corr_series.append(np.nan)
                        max_corr_series.append(np.nan)
                        min_corr_series.append(np.nan)
                else:
                    avg_corr_series.append(np.nan)
                    max_corr_series.append(np.nan)
                    min_corr_series.append(np.nan)
            
            # Create series with proper index
            corr_index = returns_df.index[49:]
            features['avg_correlation'] = pd.Series(avg_corr_series, index=corr_index)
            features['max_correlation'] = pd.Series(max_corr_series, index=corr_index)
            features['min_correlation'] = pd.Series(min_corr_series, index=corr_index)
            features['corr_dispersion'] = features['max_correlation'] - features['min_correlation']
        
        except Exception as e:
            print(f"Correlation regime calculation error: {e}")
        
        return features
    
    def _calculate_momentum_correlations(self, returns_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate momentum-based correlations"""
        features = {}
        
        try:
            # Calculate momentum for each asset
            momentum_data = {}
            for asset in returns_df.columns:
                momentum_data[asset] = returns_df[asset].rolling(20).sum()
            
            momentum_df = pd.DataFrame(momentum_data)
            
            # Calculate correlations between momentum series
            for i, asset1 in enumerate(momentum_df.columns):
                for j, asset2 in enumerate(momentum_df.columns[i+1:], i+1):
                    mom_corr = momentum_df[asset1].rolling(50).corr(momentum_df[asset2])
                    features[f'momentum_corr_{asset1}_{asset2}'] = mom_corr
        
        except Exception as e:
            print(f"Momentum correlation calculation error: {e}")
        
        return features
    
    def _calculate_volatility_correlations(self, returns_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volatility-based correlations"""
        features = {}
        
        try:
            # Calculate volatility for each asset
            vol_data = {}
            for asset in returns_df.columns:
                vol_data[asset] = returns_df[asset].rolling(20).std()
            
            vol_df = pd.DataFrame(vol_data)
            
            # Calculate correlations between volatility series
            for i, asset1 in enumerate(vol_df.columns):
                for j, asset2 in enumerate(vol_df.columns[i+1:], i+1):
                    vol_corr = vol_df[asset1].rolling(50).corr(vol_df[asset2])
                    features[f'volatility_corr_{asset1}_{asset2}'] = vol_corr
        
        except Exception as e:
            print(f"Volatility correlation calculation error: {e}")
        
        return features

class CrossAssetFactorFeatures:
    """Factor model features for cross-asset analysis"""
    
    def __init__(self, n_factors: int = 5, factor_method: str = 'pca'):
        self.n_factors = n_factors
        self.factor_method = factor_method
        self.factor_model = None
        self.scaler = StandardScaler()
    
    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate factor model features"""
        # Prepare returns data
        returns_data = {}
        for asset_class, df in data.items():
            if 'close' in df.columns:
                returns_data[asset_class] = df['close'].pct_change()
        
        if len(returns_data) < 2:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty or len(returns_df) < self.n_factors * 2:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=returns_df.index)
        
        # Fit factor model
        if self.factor_method == 'pca':
            self.factor_model = PCA(n_components=self.n_factors)
        else:
            self.factor_model = FactorAnalysis(n_components=self.n_factors)
        
        # Scale returns
        scaled_returns = self.scaler.fit_transform(returns_df)
        
        # Fit factor model
        factors = self.factor_model.fit_transform(scaled_returns)
        
        # Add factor loadings as features
        for i in range(self.n_factors):
            features[f'factor_{i+1}'] = factors[:, i]
        
        # Calculate factor-specific features
        features.update(self._calculate_factor_momentum(factors))
        features.update(self._calculate_factor_volatility(factors))
        features.update(self._calculate_factor_loadings_features(returns_df))
        
        return features.dropna(how='all')
    
    def _calculate_factor_momentum(self, factors: np.ndarray) -> Dict[str, pd.Series]:
        """Calculate factor momentum features"""
        features = {}
        
        try:
            factors_df = pd.DataFrame(factors, columns=[f'factor_{i+1}' for i in range(factors.shape[1])])
            
            for col in factors_df.columns:
                # Factor momentum
                features[f'{col}_momentum_5d'] = factors_df[col].rolling(5).sum()
                features[f'{col}_momentum_20d'] = factors_df[col].rolling(20).sum()
                
                # Factor trend
                features[f'{col}_trend'] = factors_df[col].rolling(10).apply(
                    lambda x: stats.linregress(range(len(x)), x)[0] if len(x) > 1 else 0
                )
        
        except Exception as e:
            print(f"Factor momentum calculation error: {e}")
        
        return features
    
    def _calculate_factor_volatility(self, factors: np.ndarray) -> Dict[str, pd.Series]:
        """Calculate factor volatility features"""
        features = {}
        
        try:
            factors_df = pd.DataFrame(factors, columns=[f'factor_{i+1}' for i in range(factors.shape[1])])
            
            for col in factors_df.columns:
                # Factor volatility
                features[f'{col}_volatility'] = factors_df[col].rolling(20).std()
                
                # Factor volatility of volatility
                vol = factors_df[col].rolling(10).std()
                features[f'{col}_vol_of_vol'] = vol.rolling(10).std()
        
        except Exception as e:
            print(f"Factor volatility calculation error: {e}")
        
        return features
    
    def _calculate_factor_loadings_features(self, returns_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate rolling factor loadings features"""
        features = {}
        
        try:
            # Rolling factor analysis
            window = 50
            if len(returns_df) < window:
                return features
            
            loading_series = {asset: [] for asset in returns_df.columns}
            dates = []
            
            for i in range(window, len(returns_df)):
                window_data = returns_df.iloc[i-window:i]
                
                try:
                    # Fit factor model on window
                    scaled_window = self.scaler.fit_transform(window_data)
                    temp_factor_model = PCA(n_components=min(self.n_factors, len(returns_df.columns)))
                    temp_factor_model.fit(scaled_window)
                    
                    # Get loadings (components)
                    loadings = temp_factor_model.components_
                    
                    # Store first factor loading for each asset
                    for j, asset in enumerate(returns_df.columns):
                        loading_series[asset].append(loadings[0, j] if len(loadings) > 0 else 0)
                    
                    dates.append(returns_df.index[i])
                
                except:
                    # Handle singular matrix or other errors
                    for asset in returns_df.columns:
                        loading_series[asset].append(0)
                    dates.append(returns_df.index[i])
            
            # Convert to features
            for asset in returns_df.columns:
                if loading_series[asset]:
                    features[f'{asset}_factor1_loading'] = pd.Series(
                        loading_series[asset], index=dates
                    )
        
        except Exception as e:
            print(f"Factor loadings calculation error: {e}")
        
        return features

class CrossAssetMomentumFeatures:
    """Momentum features for cross-asset analysis"""
    
    def __init__(self, periods: List[int] = [5, 10, 20, 50, 100]):
        self.periods = periods
    
    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate momentum features for each asset class"""
        features = {}
        
        for asset_class, df in data.items():
            if 'close' not in df.columns:
                continue
            
            asset_features = pd.DataFrame(index=df.index)
            returns = df['close'].pct_change()
            
            # Price momentum
            for period in self.periods:
                if len(df) > period:
                    # Simple momentum
                    momentum = (df['close'] / df['close'].shift(period)) - 1
                    asset_features[f'momentum_{period}d'] = momentum
                    
                    # Risk-adjusted momentum
                    vol = returns.rolling(period).std()
                    risk_adj_momentum = momentum / (vol + 1e-8)
                    asset_features[f'risk_adj_momentum_{period}d'] = risk_adj_momentum
            
            # Cross-sectional momentum (relative to other assets)
            # This would be calculated at the portfolio level
            
            # Time series momentum
            for period in [10, 20]:
                if len(returns) > period:
                    ts_momentum = returns.rolling(period).sum()
                    asset_features[f'ts_momentum_{period}d'] = ts_momentum
            
            # Momentum acceleration
            for period in [20, 50]:
                if len(df) > period + 10:
                    short_momentum = (df['close'] / df['close'].shift(10)) - 1
                    long_momentum = (df['close'] / df['close'].shift(period)) - 1
                    momentum_accel = short_momentum - long_momentum
                    asset_features[f'momentum_accel_{period}d'] = momentum_accel
            
            features[asset_class] = asset_features.dropna(how='all')
        
        return features

class CrossAssetMeanReversionFeatures:
    """Mean reversion features for cross-asset analysis"""
    
    def __init__(self, periods: List[int] = [10, 20, 50, 100]):
        self.periods = periods
    
    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate mean reversion features for each asset class"""
        features = {}
        
        for asset_class, df in data.items():
            if 'close' not in df.columns:
                continue
            
            asset_features = pd.DataFrame(index=df.index)
            
            # Distance from moving average
            for period in self.periods:
                if len(df) > period:
                    ma = df['close'].rolling(period).mean()
                    distance = (df['close'] - ma) / ma
                    asset_features[f'ma_distance_{period}d'] = distance
                    
                    # Z-score
                    std = df['close'].rolling(period).std()
                    z_score = (df['close'] - ma) / (std + 1e-8)
                    asset_features[f'z_score_{period}d'] = z_score
            
            # Half-life of mean reversion
            returns = df['close'].pct_change()
            for period in [50, 100]:
                if len(returns) > period:
                    half_life = self._calculate_half_life(returns, period)
                    asset_features[f'half_life_{period}d'] = half_life
            
            # Hurst exponent (measure of mean reversion vs momentum)
            for period in [50, 100]:
                if len(df) > period:
                    hurst = self._calculate_hurst_exponent(df['close'], period)
                    asset_features[f'hurst_{period}d'] = hurst
            
            features[asset_class] = asset_features.dropna(how='all')
        
        return features
    
    def _calculate_half_life(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate half-life of mean reversion"""
        half_lives = []
        
        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            
            try:
                # Simple AR(1) model: r_t = alpha + beta * r_{t-1} + epsilon_t
                y = window_returns.iloc[1:].values
                x = window_returns.iloc[:-1].values
                
                if len(y) > 1 and len(x) > 1:
                    beta = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
                    
                    if beta < 1 and beta > 0:
                        half_life = -np.log(2) / np.log(beta)
                    else:
                        half_life = np.inf
                else:
                    half_life = np.inf
            except:
                half_life = np.inf
            
            half_lives.append(half_life)
        
        return pd.Series(half_lives, index=returns.index[window:])
    
    def _calculate_hurst_exponent(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Hurst exponent using R/S analysis"""
        hurst_values = []
        
        for i in range(window, len(prices)):
            window_prices = prices.iloc[i-window:i]
            
            try:
                # Calculate log returns
                log_returns = np.log(window_prices / window_prices.shift(1)).dropna()
                
                if len(log_returns) < 10:
                    hurst_values.append(0.5)
                    continue
                
                # R/S analysis
                n = len(log_returns)
                mean_return = log_returns.mean()
                
                # Cumulative deviations
                cumulative_deviations = (log_returns - mean_return).cumsum()
                
                # Range
                R = cumulative_deviations.max() - cumulative_deviations.min()
                
                # Standard deviation
                S = log_returns.std()
                
                if S > 0:
                    # Hurst exponent approximation
                    hurst = np.log(R/S) / np.log(n)
                else:
                    hurst = 0.5
                
                hurst_values.append(max(0, min(1, hurst)))
            
            except:
                hurst_values.append(0.5)
        
        return pd.Series(hurst_values, index=prices.index[window:])

class CrossAssetRegimeFeatures:
    """Regime detection features for cross-asset analysis"""
    
    def __init__(self, regime_window: int = 50):
        self.regime_window = regime_window
    
    def calculate_features(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate regime features across asset classes"""
        # Prepare returns data
        returns_data = {}
        for asset_class, df in data.items():
            if 'close' in df.columns:
                returns_data[asset_class] = df['close'].pct_change()
        
        if len(returns_data) < 2:
            return pd.DataFrame()
        
        returns_df = pd.DataFrame(returns_data).dropna()
        
        if returns_df.empty:
            return pd.DataFrame()
        
        features = pd.DataFrame(index=returns_df.index)
        
        # Market stress indicators
        features.update(self._calculate_stress_indicators(returns_df))
        
        # Volatility regime indicators
        features.update(self._calculate_volatility_regime_indicators(returns_df))
        
        # Correlation regime indicators
        features.update(self._calculate_correlation_regime_indicators(returns_df))
        
        # Risk-on/Risk-off indicators
        features.update(self._calculate_risk_regime_indicators(returns_df))
        
        return features.dropna(how='all')
    
    def _calculate_stress_indicators(self, returns_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate market stress indicators"""
        features = {}
        
        try:
            # Maximum drawdown indicator
            for asset in returns_df.columns:
                cumulative_returns = (1 + returns_df[asset]).cumprod()
                rolling_max = cumulative_returns.rolling(self.regime_window).max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                features[f'{asset}_max_drawdown'] = drawdown.rolling(self.regime_window).min()
            
            # Cross-asset stress (average of worst performers)
            daily_ranks = returns_df.rank(axis=1, ascending=False)
            worst_performers = (daily_ranks >= len(returns_df.columns) * 0.7).sum(axis=1)
            features['stress_indicator'] = worst_performers.rolling(10).mean()
        
        except Exception as e:
            print(f"Stress indicator calculation error: {e}")
        
        return features
    
    def _calculate_volatility_regime_indicators(self, returns_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate volatility regime indicators"""
        features = {}
        
        try:
            # Average volatility across assets
            vol_data = {}
            for asset in returns_df.columns:
                vol_data[asset] = returns_df[asset].rolling(20).std() * np.sqrt(252)
            
            vol_df = pd.DataFrame(vol_data)
            features['avg_volatility'] = vol_df.mean(axis=1)
            features['max_volatility'] = vol_df.max(axis=1)
            features['vol_dispersion'] = vol_df.std(axis=1)
            
            # Volatility regime classification
            avg_vol = features['avg_volatility']
            vol_percentiles = avg_vol.rolling(252).quantile([0.33, 0.67])
            
            # This is a simplified approach - in practice you'd use more sophisticated methods
            features['vol_regime'] = pd.Series(index=avg_vol.index, dtype=float)
            features['vol_regime'] = 1  # Default to medium volatility
            
        except Exception as e:
            print(f"Volatility regime calculation error: {e}")
        
        return features
    
    def _calculate_correlation_regime_indicators(self, returns_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate correlation regime indicators"""
        features = {}
        
        try:
            # Rolling average correlation
            corr_values = []
            for i in range(self.regime_window, len(returns_df)):
                window_data = returns_df.iloc[i-self.regime_window:i]
                corr_matrix = window_data.corr()
                
                # Get upper triangle correlations
                upper_triangle = np.triu(corr_matrix.values, k=1)
                upper_values = upper_triangle[upper_triangle != 0]
                
                if len(upper_values) > 0:
                    avg_corr = np.mean(upper_values)
                else:
                    avg_corr = 0
                
                corr_values.append(avg_corr)
            
            features['avg_correlation'] = pd.Series(
                corr_values, index=returns_df.index[self.regime_window:]
            )
        
        except Exception as e:
            print(f"Correlation regime calculation error: {e}")
        
        return features
    
    def _calculate_risk_regime_indicators(self, returns_df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate risk-on/risk-off regime indicators"""
        features = {}
        
        try:
            # Simplified risk-on/risk-off indicator
            # In practice, you'd use specific asset classes like stocks vs bonds
            
            if len(returns_df.columns) >= 2:
                # Assume first asset is risky (e.g., stocks), second is safe (e.g., bonds)
                risky_asset = returns_df.iloc[:, 0]
                safe_asset = returns_df.iloc[:, 1] if len(returns_df.columns) > 1 else -risky_asset
                
                # Risk-on indicator: risky assets outperforming safe assets
                risk_on_indicator = (risky_asset - safe_asset).rolling(10).mean()
                features['risk_on_indicator'] = risk_on_indicator
                
                # Flight-to-quality indicator
                flight_to_quality = (-risky_asset + safe_asset).rolling(10).mean()
                features['flight_to_quality'] = flight_to_quality
        
        except Exception as e:
            print(f"Risk regime calculation error: {e}")
        
        return features

class CrossAssetFeatureEngineering:
    """Main class for cross-asset feature engineering"""
    
    def __init__(self, 
                 feature_categories: List[CrossAssetFeatureCategory] = None,
                 scaling_method: CrossAssetScalingMethod = CrossAssetScalingMethod.STANDARD):
        
        if feature_categories is None:
            feature_categories = list(CrossAssetFeatureCategory)
        
        self.feature_categories = feature_categories
        self.scaling_method = scaling_method
        
        # Initialize feature calculators
        self.price_features = CrossAssetPriceActionFeatures()
        self.technical_features = CrossAssetTechnicalFeatures()
        self.volatility_features = CrossAssetVolatilityFeatures()
        self.correlation_features = CrossAssetCorrelationFeatures()
        self.factor_features = CrossAssetFactorFeatures()
        self.momentum_features = CrossAssetMomentumFeatures()
        self.mean_reversion_features = CrossAssetMeanReversionFeatures()
        self.regime_features = CrossAssetRegimeFeatures()
        
        # Scalers
        self.scalers = {}
        self._initialize_scalers()
    
    def _initialize_scalers(self):
        """Initialize scalers based on scaling method"""
        if self.scaling_method == CrossAssetScalingMethod.STANDARD:
            self.scalers['default'] = StandardScaler()
        elif self.scaling_method == CrossAssetScalingMethod.MINMAX:
            self.scalers['default'] = MinMaxScaler()
        elif self.scaling_method == CrossAssetScalingMethod.ROBUST:
            self.scalers['default'] = RobustScaler()
        else:
            self.scalers['default'] = None
    
    def calculate_all_features(self, 
                             data: Dict[str, pd.DataFrame],
                             economic_indicators: Optional[Dict[str, pd.DataFrame]] = None,
                             sentiment_data: Optional[Dict[str, pd.DataFrame]] = None) -> CrossAssetFeatureSet:
        """Calculate all cross-asset features"""
        
        feature_set = CrossAssetFeatureSet(
            price_features={},
            technical_features={},
            volatility_features={},
            correlation_features=pd.DataFrame(),
            factor_features=pd.DataFrame(),
            momentum_features={},
            mean_reversion_features={},
            regime_features=pd.DataFrame(),
            timestamp=datetime.now()
        )
        
        # Calculate features by category
        if CrossAssetFeatureCategory.PRICE_ACTION in self.feature_categories:
            feature_set.price_features = self.price_features.calculate_features(data)
        
        if CrossAssetFeatureCategory.TECHNICAL in self.feature_categories:
            feature_set.technical_features = self.technical_features.calculate_features(data)
        
        if CrossAssetFeatureCategory.VOLATILITY in self.feature_categories:
            feature_set.volatility_features = self.volatility_features.calculate_features(data)
        
        if CrossAssetFeatureCategory.CORRELATION in self.feature_categories:
            feature_set.correlation_features = self.correlation_features.calculate_features(data)
        
        if CrossAssetFeatureCategory.FACTOR_MODELS in self.feature_categories:
            feature_set.factor_features = self.factor_features.calculate_features(data)
        
        if CrossAssetFeatureCategory.MOMENTUM in self.feature_categories:
            feature_set.momentum_features = self.momentum_features.calculate_features(data)
        
        if CrossAssetFeatureCategory.MEAN_REVERSION in self.feature_categories:
            feature_set.mean_reversion_features = self.mean_reversion_features.calculate_features(data)
        
        if CrossAssetFeatureCategory.REGIME in self.feature_categories:
            feature_set.regime_features = self.regime_features.calculate_features(data)
        
        # Add external data if provided
        if economic_indicators:
            feature_set.macro_features = self._process_economic_indicators(economic_indicators)
        
        if sentiment_data:
            feature_set.sentiment_features = self._process_sentiment_data(sentiment_data)
        
        # Calculate feature metadata
        feature_set.feature_metadata = self._calculate_feature_metadata(feature_set)
        
        return feature_set
    
    def prepare_ml_features(self, 
                          feature_set: CrossAssetFeatureSet,
                          target_asset: str = None,
                          feature_selection: bool = True,
                          n_features: int = 50) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Prepare features for machine learning"""
        
        # Combine all features into a single DataFrame
        combined_features = self._combine_features(feature_set)
        
        if combined_features.empty:
            return pd.DataFrame(), {}
        
        # Handle missing values
        combined_features = self._handle_missing_values(combined_features)
        
        # Scale features
        if self.scaling_method != CrossAssetScalingMethod.NONE:
            combined_features = self._scale_features(combined_features)
        
        # Feature selection
        if feature_selection and target_asset and len(combined_features.columns) > n_features:
            combined_features, selection_info = self._select_features(
                combined_features, target_asset, n_features
            )
        else:
            selection_info = {'method': 'none', 'selected_features': list(combined_features.columns)}
        
        # Prepare metadata
        ml_metadata = {
            'n_features': len(combined_features.columns),
            'n_samples': len(combined_features),
            'scaling_method': self.scaling_method.value,
            'feature_selection': selection_info,
            'feature_categories': [cat.value for cat in self.feature_categories],
            'timestamp': datetime.now()
        }
        
        return combined_features, ml_metadata
    
    def _combine_features(self, feature_set: CrossAssetFeatureSet) -> pd.DataFrame:
        """Combine all features into a single DataFrame"""
        all_features = []
        
        # Add asset-specific features
        for asset_features in [feature_set.price_features, feature_set.technical_features,
                              feature_set.volatility_features, feature_set.momentum_features,
                              feature_set.mean_reversion_features]:
            for asset_class, features_df in asset_features.items():
                if not features_df.empty:
                    # Prefix column names with asset class
                    prefixed_features = features_df.add_prefix(f'{asset_class}_')
                    all_features.append(prefixed_features)
        
        # Add cross-asset features
        for features_df in [feature_set.correlation_features, feature_set.factor_features,
                           feature_set.regime_features]:
            if not features_df.empty:
                all_features.append(features_df)
        
        # Add external features
        for features_df in [feature_set.macro_features, feature_set.sentiment_features,
                           feature_set.flow_features, feature_set.risk_features]:
            if features_df is not None and not features_df.empty:
                all_features.append(features_df)
        
        if not all_features:
            return pd.DataFrame()
        
        # Combine all features
        combined = pd.concat(all_features, axis=1, sort=False)
        
        # Remove duplicate columns
        combined = combined.loc[:, ~combined.columns.duplicated()]
        
        return combined
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Forward fill then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Fill remaining NaNs with 0
        df = df.fillna(0)
        
        return df
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using the specified method"""
        if self.scalers['default'] is None:
            return df
        
        try:
            scaled_data = self.scalers['default'].fit_transform(df)
            return pd.DataFrame(scaled_data, index=df.index, columns=df.columns)
        except Exception as e:
            print(f"Scaling error: {e}")
            return df
    
    def _select_features(self, 
                        df: pd.DataFrame, 
                        target_asset: str, 
                        n_features: int) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Select most important features"""
        try:
            # Create a simple target (next period return)
            # In practice, you'd pass the actual target
            target = df.iloc[:, 0].shift(-1).dropna()  # Simplified target
            features = df.iloc[:-1]  # Align with target
            
            # Align indices
            common_index = features.index.intersection(target.index)
            features = features.loc[common_index]
            target = target.loc[common_index]
            
            if len(features) == 0 or len(target) == 0:
                return df, {'method': 'failed', 'selected_features': list(df.columns)}
            
            # Use SelectKBest with f_regression
            selector = SelectKBest(score_func=f_regression, k=min(n_features, len(features.columns)))
            selected_features = selector.fit_transform(features, target)
            
            # Get selected feature names
            selected_indices = selector.get_support(indices=True)
            selected_columns = features.columns[selected_indices]
            
            selection_info = {
                'method': 'SelectKBest_f_regression',
                'selected_features': list(selected_columns),
                'feature_scores': dict(zip(selected_columns, selector.scores_[selected_indices]))
            }
            
            return df[selected_columns], selection_info
        
        except Exception as e:
            print(f"Feature selection error: {e}")
            return df, {'method': 'failed', 'selected_features': list(df.columns)}
    
    def _process_economic_indicators(self, economic_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process economic indicators"""
        # Simplified processing - in practice, you'd implement more sophisticated methods
        processed_features = []
        
        for indicator_name, indicator_data in economic_data.items():
            if not indicator_data.empty and 'value' in indicator_data.columns:
                # Basic processing
                processed = pd.DataFrame(index=indicator_data.index)
                processed[f'{indicator_name}_level'] = indicator_data['value']
                processed[f'{indicator_name}_change'] = indicator_data['value'].pct_change()
                processed[f'{indicator_name}_ma'] = indicator_data['value'].rolling(12).mean()
                
                processed_features.append(processed)
        
        if processed_features:
            return pd.concat(processed_features, axis=1)
        else:
            return pd.DataFrame()
    
    def _process_sentiment_data(self, sentiment_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Process sentiment data"""
        # Simplified processing
        processed_features = []
        
        for sentiment_name, sentiment_df in sentiment_data.items():
            if not sentiment_df.empty and 'sentiment_score' in sentiment_df.columns:
                processed = pd.DataFrame(index=sentiment_df.index)
                processed[f'{sentiment_name}_sentiment'] = sentiment_df['sentiment_score']
                processed[f'{sentiment_name}_sentiment_ma'] = sentiment_df['sentiment_score'].rolling(5).mean()
                
                processed_features.append(processed)
        
        if processed_features:
            return pd.concat(processed_features, axis=1)
        else:
            return pd.DataFrame()
    
    def _calculate_feature_metadata(self, feature_set: CrossAssetFeatureSet) -> Dict[str, Any]:
        """Calculate metadata about the feature set"""
        metadata = {
            'feature_categories': [cat.value for cat in self.feature_categories],
            'scaling_method': self.scaling_method.value,
            'timestamp': datetime.now()
        }
        
        # Count features by category
        feature_counts = {}
        
        for asset_class, features_df in feature_set.price_features.items():
            feature_counts[f'price_features_{asset_class}'] = len(features_df.columns)
        
        for asset_class, features_df in feature_set.technical_features.items():
            feature_counts[f'technical_features_{asset_class}'] = len(features_df.columns)
        
        feature_counts['correlation_features'] = len(feature_set.correlation_features.columns)
        feature_counts['factor_features'] = len(feature_set.factor_features.columns)
        feature_counts['regime_features'] = len(feature_set.regime_features.columns)
        
        metadata['feature_counts'] = feature_counts
        metadata['total_features'] = sum(feature_counts.values())
        
        return metadata
    
    def get_feature_importance(self, 
                             features: pd.DataFrame, 
                             target: pd.Series,
                             method: str = 'mutual_info') -> Dict[str, float]:
        """Calculate feature importance"""
        try:
            # Align features and target
            common_index = features.index.intersection(target.index)
            aligned_features = features.loc[common_index]
            aligned_target = target.loc[common_index]
            
            if len(aligned_features) == 0 or len(aligned_target) == 0:
                return {}
            
            if method == 'mutual_info':
                importance_scores = mutual_info_regression(aligned_features, aligned_target)
            else:  # correlation
                importance_scores = []
                for col in aligned_features.columns:
                    corr = abs(aligned_features[col].corr(aligned_target))
                    importance_scores.append(corr if not np.isnan(corr) else 0)
            
            return dict(zip(aligned_features.columns, importance_scores))
        
        except Exception as e:
            print(f"Feature importance calculation error: {e}")
            return {}
    
    def validate_features(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Validate feature quality"""
        validation_results = {
            'n_features': len(features.columns),
            'n_samples': len(features),
            'missing_values': features.isnull().sum().sum(),
            'missing_percentage': (features.isnull().sum().sum() / (len(features) * len(features.columns))) * 100,
            'infinite_values': np.isinf(features.select_dtypes(include=[np.number])).sum().sum(),
            'constant_features': (features.nunique() == 1).sum(),
            'duplicate_features': len(features.columns) - len(features.T.drop_duplicates()),
            'feature_correlations': self._check_high_correlations(features),
            'feature_variance': features.var().describe().to_dict(),
            'timestamp': datetime.now()
        }
        
        return validation_results
    
    def _check_high_correlations(self, features: pd.DataFrame, threshold: float = 0.95) -> Dict[str, List[str]]:
        """Check for highly correlated features"""
        try:
            corr_matrix = features.corr().abs()
            high_corr_pairs = {}
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > threshold:
                        feature1 = corr_matrix.columns[i]
                        feature2 = corr_matrix.columns[j]
                        
                        if feature1 not in high_corr_pairs:
                            high_corr_pairs[feature1] = []
                        high_corr_pairs[feature1].append(feature2)
            
            return high_corr_pairs
        
        except Exception as e:
            print(f"Correlation check error: {e}")
            return {}
    
    def get_feature_statistics(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive feature statistics"""
        try:
            numeric_features = features.select_dtypes(include=[np.number])
            
            stats = {
                'basic_stats': numeric_features.describe().to_dict(),
                'skewness': numeric_features.skew().to_dict(),
                'kurtosis': numeric_features.kurtosis().to_dict(),
                'feature_types': features.dtypes.to_dict(),
                'memory_usage': features.memory_usage(deep=True).to_dict(),
                'timestamp': datetime.now()
            }
            
            return stats
        
        except Exception as e:
            print(f"Feature statistics calculation error: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Sample cross-asset data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create sample data for different asset classes
    np.random.seed(42)
    
    sample_data = {
        'equities': pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02),
            'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02) + np.random.rand(len(dates)) * 2,
            'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02) - np.random.rand(len(dates)) * 2,
            'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.02),
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates),
        
        'bonds': pd.DataFrame({
            'open': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01),
            'high': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01) + np.random.rand(len(dates)) * 0.5,
            'low': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01) - np.random.rand(len(dates)) * 0.5,
            'close': 100 + np.cumsum(np.random.randn(len(dates)) * 0.01),
            'volume': np.random.randint(500000, 5000000, len(dates))
        }, index=dates),
        
        'commodities': pd.DataFrame({
            'open': 50 + np.cumsum(np.random.randn(len(dates)) * 0.03),
            'high': 50 + np.cumsum(np.random.randn(len(dates)) * 0.03) + np.random.rand(len(dates)) * 3,
            'low': 50 + np.cumsum(np.random.randn(len(dates)) * 0.03) - np.random.rand(len(dates)) * 3,
            'close': 50 + np.cumsum(np.random.randn(len(dates)) * 0.03),
            'volume': np.random.randint(100000, 1000000, len(dates))
        }, index=dates),
        
        'currencies': pd.DataFrame({
            'open': 1.0 + np.cumsum(np.random.randn(len(dates)) * 0.005),
            'high': 1.0 + np.cumsum(np.random.randn(len(dates)) * 0.005) + np.random.rand(len(dates)) * 0.01,
            'low': 1.0 + np.cumsum(np.random.randn(len(dates)) * 0.005) - np.random.rand(len(dates)) * 0.01,
            'close': 1.0 + np.cumsum(np.random.randn(len(dates)) * 0.005),
            'volume': np.random.randint(10000000, 100000000, len(dates))
        }, index=dates)
    }
    
    print("Cross-Asset Feature Engineering Example")
    print("=" * 50)
    
    # Initialize feature engineering
    feature_categories = [
        CrossAssetFeatureCategory.PRICE_ACTION,
        CrossAssetFeatureCategory.TECHNICAL,
        CrossAssetFeatureCategory.VOLATILITY,
        CrossAssetFeatureCategory.CORRELATION,
        CrossAssetFeatureCategory.FACTOR_MODELS,
        CrossAssetFeatureCategory.MOMENTUM,
        CrossAssetFeatureCategory.MEAN_REVERSION,
        CrossAssetFeatureCategory.REGIME
    ]
    
    feature_engineer = CrossAssetFeatureEngineering(
        feature_categories=feature_categories,
        scaling_method=CrossAssetScalingMethod.STANDARD
    )
    
    # Calculate all features
    print("\n1. Calculating comprehensive feature set...")
    feature_set = feature_engineer.calculate_all_features(sample_data)
    
    print(f"   - Price action features: {len(feature_set.price_features)} asset classes")
    print(f"   - Technical features: {len(feature_set.technical_features)} asset classes")
    print(f"   - Volatility features: {len(feature_set.volatility_features)} asset classes")
    print(f"   - Correlation features: {len(feature_set.correlation_features.columns)} features")
    print(f"   - Factor features: {len(feature_set.factor_features.columns)} features")
    print(f"   - Momentum features: {len(feature_set.momentum_features)} asset classes")
    print(f"   - Mean reversion features: {len(feature_set.mean_reversion_features)} asset classes")
    print(f"   - Regime features: {len(feature_set.regime_features.columns)} features")
    
    # Prepare ML features
    print("\n2. Preparing features for machine learning...")
    ml_features, ml_metadata = feature_engineer.prepare_ml_features(
        feature_set,
        target_asset='equities',
        feature_selection=True,
        n_features=30
    )
    
    print(f"   - Total ML features: {ml_metadata['n_features']}")
    print(f"   - Total samples: {ml_metadata['n_samples']}")
    print(f"   - Scaling method: {ml_metadata['scaling_method']}")
    
    # Feature validation
    print("\n3. Validating feature quality...")
    validation_results = feature_engineer.validate_features(ml_features)
    
    print(f"   - Missing values: {validation_results['missing_values']}")
    print(f"   - Missing percentage: {validation_results['missing_percentage']:.2f}%")
    print(f"   - Constant features: {validation_results['constant_features']}")
    print(f"   - Duplicate features: {validation_results['duplicate_features']}")
    
    # Feature statistics
    print("\n4. Feature statistics...")
    feature_stats = feature_engineer.get_feature_statistics(ml_features)
    
    if 'basic_stats' in feature_stats and ml_features.columns.tolist():
        sample_feature = ml_features.columns[0]
        if sample_feature in feature_stats['basic_stats']:
            print(f"   - Sample feature ({sample_feature}) mean: {feature_stats['basic_stats'][sample_feature]['mean']:.4f}")
            print(f"   - Sample feature ({sample_feature}) std: {feature_stats['basic_stats'][sample_feature]['std']:.4f}")
    
    # Feature importance (simplified example)
    if len(ml_features) > 1 and len(ml_features.columns) > 0:
        print("\n5. Feature importance analysis...")
        # Create a simple target for demonstration
        target = ml_features.iloc[:, 0].shift(-1).dropna()
        
        importance_scores = feature_engineer.get_feature_importance(
            ml_features.iloc[:-1],  # Align with target
            target,
            method='correlation'
        )
        
        if importance_scores:
            # Show top 5 most important features
            sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
            print("   - Top 5 most important features:")
            for i, (feature, score) in enumerate(sorted_importance[:5]):
                print(f"     {i+1}. {feature}: {score:.4f}")
    
    print("\nCross-asset feature engineering completed successfully!")
    print(f"Final feature matrix shape: {ml_features.shape}")