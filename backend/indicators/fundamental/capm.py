"""Capital Asset Pricing Model (CAPM) Analysis

The CAPM calculates the expected return of an asset based on its systematic risk (beta)
relative to the market. This model is fundamental for portfolio management, risk assessment,
and determining whether an asset is fairly priced relative to its risk.

Author: Assistant
Date: 2024
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

logger = logging.getLogger(__name__)


class AssetType(Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"
    FUTURES = "futures"
    INDEX = "index"
    CROSS_ASSET = "cross_asset"


class IndicatorCategory(Enum):
    FUNDAMENTAL = "fundamental"
    RISK = "risk"
    VALUATION = "valuation"


@dataclass
class CAPMResult:
    """Result of CAPM calculation"""
    name: str
    beta: float
    alpha: float
    expected_return: float
    risk_free_rate: float
    market_return: float
    market_premium: float
    r_squared: float
    systematic_risk: float
    unsystematic_risk: float
    sharpe_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class CAPMIndicator:
    """Capital Asset Pricing Model Calculator with Advanced Risk Analysis"""
    
    def __init__(self, risk_free_rate: float = 0.02, lookback_period: int = 252,
                 rolling_window: int = 63, min_periods: int = 30):
        """
        Initialize CAPM calculator
        
        Args:
            risk_free_rate: Risk-free rate (default: 2%)
            lookback_period: Period for beta calculation (default: 252 days)
            rolling_window: Rolling window for time-varying analysis (default: 63 days)
            min_periods: Minimum periods required for calculation (default: 30)
        """
        self.risk_free_rate = risk_free_rate
        self.lookback_period = lookback_period
        self.rolling_window = rolling_window
        self.min_periods = min_periods
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, asset_data: pd.DataFrame, market_data: pd.DataFrame,
                 custom_risk_free_rate: Optional[float] = None,
                 asset_type: AssetType = AssetType.STOCK) -> CAPMResult:
        """
        Calculate CAPM analysis for given asset and market data
        
        Args:
            asset_data: Asset price data DataFrame with 'close' column
            market_data: Market index data DataFrame with 'close' column
            custom_risk_free_rate: Override default risk-free rate
            asset_type: Type of asset being analyzed
            
        Returns:
            CAPMResult containing risk and return analysis
        """
        try:
            # Use custom risk-free rate if provided
            risk_free_rate = custom_risk_free_rate or self.risk_free_rate
            
            # Align data and calculate returns
            aligned_data = self._align_data(asset_data, market_data)
            asset_returns, market_returns = self._calculate_returns(aligned_data)
            
            # Calculate excess returns
            asset_excess = asset_returns - risk_free_rate / 252  # Daily risk-free rate
            market_excess = market_returns - risk_free_rate / 252
            
            # Calculate beta and alpha using full period
            beta, alpha, r_squared = self._calculate_beta_alpha(asset_excess, market_excess)
            
            # Calculate expected return using CAPM
            market_return = market_returns.mean() * 252  # Annualized
            market_premium = market_return - risk_free_rate
            expected_return = risk_free_rate + beta * market_premium
            
            # Risk decomposition
            systematic_risk, unsystematic_risk = self._decompose_risk(asset_returns, market_returns, beta)
            
            # Performance metrics
            sharpe_ratio = self._calculate_sharpe_ratio(asset_returns, risk_free_rate)
            treynor_ratio = self._calculate_treynor_ratio(asset_returns, beta, risk_free_rate)
            jensen_alpha = self._calculate_jensen_alpha(asset_returns, market_returns, beta, risk_free_rate)
            
            # Generate signals
            signals = self._generate_signals(beta, alpha, jensen_alpha, sharpe_ratio, r_squared)
            
            # Create time series analysis
            values_df = self._create_time_series(aligned_data, asset_returns, market_returns, 
                                               risk_free_rate, beta)
            
            # Calculate confidence
            confidence = self._calculate_confidence(r_squared, len(asset_returns))
            
            return CAPMResult(
                name="Capital Asset Pricing Model",
                beta=beta,
                alpha=alpha,
                expected_return=expected_return,
                risk_free_rate=risk_free_rate,
                market_return=market_return,
                market_premium=market_premium,
                r_squared=r_squared,
                systematic_risk=systematic_risk,
                unsystematic_risk=unsystematic_risk,
                sharpe_ratio=sharpe_ratio,
                treynor_ratio=treynor_ratio,
                jensen_alpha=jensen_alpha,
                values=values_df,
                metadata={
                    'lookback_period': self.lookback_period,
                    'rolling_window': self.rolling_window,
                    'asset_volatility': asset_returns.std() * np.sqrt(252),
                    'market_volatility': market_returns.std() * np.sqrt(252),
                    'correlation': asset_returns.corrwith(market_returns).iloc[0] if len(asset_returns) > 0 else 0,
                    'tracking_error': self._calculate_tracking_error(asset_returns, market_returns, beta),
                    'information_ratio': self._calculate_information_ratio(asset_returns, market_returns, beta),
                    'downside_beta': self._calculate_downside_beta(asset_returns, market_returns),
                    'upside_beta': self._calculate_upside_beta(asset_returns, market_returns),
                    'beta_stability': self._analyze_beta_stability(values_df),
                    'risk_attribution': self._analyze_risk_attribution(systematic_risk, unsystematic_risk),
                    'market_timing': self._analyze_market_timing(asset_returns, market_returns),
                    'regime_analysis': self._analyze_market_regimes(asset_returns, market_returns),
                    'interpretation': self._get_interpretation(beta, alpha, jensen_alpha, r_squared)
                },
                confidence=confidence,
                timestamp=datetime.now(),
                asset_type=asset_type,
                category=IndicatorCategory.FUNDAMENTAL,
                signals=signals
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating CAPM: {e}")
            return self._empty_result(asset_type)
    
    def _align_data(self, asset_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
        """Align asset and market data by date"""
        # Ensure both have 'close' column
        if 'close' not in asset_data.columns or 'close' not in market_data.columns:
            raise ValueError("Both asset_data and market_data must have 'close' column")
        
        # Align by index (dates)
        aligned = pd.DataFrame({
            'asset_price': asset_data['close'],
            'market_price': market_data['close']
        }).dropna()
        
        # Limit to lookback period
        if len(aligned) > self.lookback_period:
            aligned = aligned.tail(self.lookback_period)
        
        if len(aligned) < self.min_periods:
            raise ValueError(f"Insufficient data: {len(aligned)} periods, minimum required: {self.min_periods}")
        
        return aligned
    
    def _calculate_returns(self, aligned_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calculate daily returns for asset and market"""
        asset_returns = aligned_data['asset_price'].pct_change().dropna()
        market_returns = aligned_data['market_price'].pct_change().dropna()
        
        # Remove outliers (beyond 3 standard deviations)
        asset_returns = self._remove_outliers(asset_returns)
        market_returns = self._remove_outliers(market_returns)
        
        # Align returns
        common_index = asset_returns.index.intersection(market_returns.index)
        asset_returns = asset_returns.loc[common_index]
        market_returns = market_returns.loc[common_index]
        
        return asset_returns, market_returns
    
    def _remove_outliers(self, returns: pd.Series, threshold: float = 3.0) -> pd.Series:
        """Remove outliers beyond threshold standard deviations"""
        mean_return = returns.mean()
        std_return = returns.std()
        
        lower_bound = mean_return - threshold * std_return
        upper_bound = mean_return + threshold * std_return
        
        return returns[(returns >= lower_bound) & (returns <= upper_bound)]
    
    def _calculate_beta_alpha(self, asset_excess: pd.Series, market_excess: pd.Series) -> Tuple[float, float, float]:
        """Calculate beta, alpha, and R-squared using linear regression"""
        if len(asset_excess) != len(market_excess) or len(asset_excess) < self.min_periods:
            return 1.0, 0.0, 0.0
        
        # Prepare data for regression
        X = market_excess.values.reshape(-1, 1)
        y = asset_excess.values
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)
        
        beta = model.coef_[0]
        alpha = model.intercept_
        
        # Calculate R-squared
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        return beta, alpha, r_squared
    
    def _decompose_risk(self, asset_returns: pd.Series, market_returns: pd.Series, 
                       beta: float) -> Tuple[float, float]:
        """Decompose total risk into systematic and unsystematic components"""
        # Total variance
        total_variance = asset_returns.var()
        
        # Systematic variance (explained by market)
        market_variance = market_returns.var()
        systematic_variance = (beta ** 2) * market_variance
        
        # Unsystematic variance (residual)
        unsystematic_variance = max(0, total_variance - systematic_variance)
        
        # Convert to standard deviations (annualized)
        systematic_risk = np.sqrt(systematic_variance * 252)
        unsystematic_risk = np.sqrt(unsystematic_variance * 252)
        
        return systematic_risk, unsystematic_risk
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    
    def _calculate_treynor_ratio(self, returns: pd.Series, beta: float, risk_free_rate: float) -> float:
        """Calculate Treynor ratio"""
        if beta == 0:
            return 0.0
        
        excess_return = returns.mean() * 252 - risk_free_rate
        return excess_return / beta
    
    def _calculate_jensen_alpha(self, asset_returns: pd.Series, market_returns: pd.Series,
                               beta: float, risk_free_rate: float) -> float:
        """Calculate Jensen's alpha"""
        asset_return = asset_returns.mean() * 252
        market_return = market_returns.mean() * 252
        
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        jensen_alpha = asset_return - expected_return
        
        return jensen_alpha
    
    def _calculate_tracking_error(self, asset_returns: pd.Series, market_returns: pd.Series, 
                                 beta: float) -> float:
        """Calculate tracking error"""
        # Expected returns based on beta
        expected_asset_returns = beta * market_returns
        
        # Tracking error is std dev of excess returns
        tracking_error = (asset_returns - expected_asset_returns).std() * np.sqrt(252)
        
        return tracking_error
    
    def _calculate_information_ratio(self, asset_returns: pd.Series, market_returns: pd.Series, 
                                   beta: float) -> float:
        """Calculate information ratio"""
        # Active returns (asset vs beta-adjusted market)
        expected_asset_returns = beta * market_returns
        active_returns = asset_returns - expected_asset_returns
        
        if active_returns.std() == 0:
            return 0.0
        
        return (active_returns.mean() * 252) / (active_returns.std() * np.sqrt(252))
    
    def _calculate_downside_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta during market downturns"""
        # Filter for negative market returns
        down_market = market_returns < 0
        
        if down_market.sum() < 10:  # Need at least 10 observations
            return 1.0
        
        asset_down = asset_returns[down_market]
        market_down = market_returns[down_market]
        
        if market_down.std() == 0:
            return 1.0
        
        downside_beta = asset_down.cov(market_down) / market_down.var()
        return downside_beta
    
    def _calculate_upside_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta during market upturns"""
        # Filter for positive market returns
        up_market = market_returns > 0
        
        if up_market.sum() < 10:  # Need at least 10 observations
            return 1.0
        
        asset_up = asset_returns[up_market]
        market_up = market_returns[up_market]
        
        if market_up.std() == 0:
            return 1.0
        
        upside_beta = asset_up.cov(market_up) / market_up.var()
        return upside_beta
    
    def _generate_signals(self, beta: float, alpha: float, jensen_alpha: float, 
                         sharpe_ratio: float, r_squared: float) -> List[str]:
        """Generate investment signals based on CAPM analysis"""
        signals = []
        
        # Beta-based signals
        if beta > 1.5:
            signals.append("HIGH_BETA")
        elif beta > 1.2:
            signals.append("AGGRESSIVE")
        elif beta < 0.5:
            signals.append("LOW_BETA")
        elif beta < 0.8:
            signals.append("DEFENSIVE")
        else:
            signals.append("MARKET_NEUTRAL")
        
        # Alpha signals
        if alpha > 0.02:  # 2% monthly alpha
            signals.append("POSITIVE_ALPHA")
        elif alpha < -0.02:
            signals.append("NEGATIVE_ALPHA")
        
        # Jensen's alpha signals
        if jensen_alpha > 0.05:  # 5% annual excess return
            signals.append("OUTPERFORMING")
        elif jensen_alpha < -0.05:
            signals.append("UNDERPERFORMING")
        
        # Risk-adjusted performance
        if sharpe_ratio > 1.0:
            signals.append("EXCELLENT_RISK_ADJUSTED")
        elif sharpe_ratio > 0.5:
            signals.append("GOOD_RISK_ADJUSTED")
        elif sharpe_ratio < 0:
            signals.append("POOR_RISK_ADJUSTED")
        
        # Model fit quality
        if r_squared > 0.7:
            signals.append("HIGH_MARKET_CORRELATION")
        elif r_squared < 0.3:
            signals.append("LOW_MARKET_CORRELATION")
        
        # Combined signals
        if beta > 1.2 and jensen_alpha > 0.03:
            signals.append("HIGH_BETA_OUTPERFORMER")
        elif beta < 0.8 and jensen_alpha > 0.02:
            signals.append("LOW_BETA_OUTPERFORMER")
        
        return signals
    
    def _create_time_series(self, aligned_data: pd.DataFrame, asset_returns: pd.Series,
                           market_returns: pd.Series, risk_free_rate: float, 
                           static_beta: float) -> pd.DataFrame:
        """Create time series DataFrame with CAPM analysis"""
        # Calculate rolling beta
        rolling_beta = self._calculate_rolling_beta(asset_returns, market_returns)
        
        # Calculate rolling alpha
        rolling_alpha = self._calculate_rolling_alpha(asset_returns, market_returns, rolling_beta)
        
        # Calculate expected returns based on CAPM
        market_excess = market_returns - risk_free_rate / 252
        expected_returns = risk_free_rate / 252 + rolling_beta * market_excess
        
        # Calculate rolling Sharpe ratio
        rolling_sharpe = self._calculate_rolling_sharpe(asset_returns, risk_free_rate)
        
        # Calculate cumulative returns
        asset_cumulative = (1 + asset_returns).cumprod()
        market_cumulative = (1 + market_returns).cumprod()
        expected_cumulative = (1 + expected_returns).cumprod()
        
        # Calculate rolling correlation
        rolling_correlation = asset_returns.rolling(self.rolling_window).corr(market_returns)
        
        # Calculate active returns (asset vs expected)
        active_returns = asset_returns - expected_returns
        
        result_df = pd.DataFrame({
            'asset_price': aligned_data['asset_price'].reindex(asset_returns.index, method='ffill'),
            'market_price': aligned_data['market_price'].reindex(asset_returns.index, method='ffill'),
            'asset_returns': asset_returns,
            'market_returns': market_returns,
            'rolling_beta': rolling_beta,
            'rolling_alpha': rolling_alpha,
            'expected_returns': expected_returns,
            'active_returns': active_returns,
            'rolling_sharpe': rolling_sharpe,
            'rolling_correlation': rolling_correlation,
            'asset_cumulative': asset_cumulative,
            'market_cumulative': market_cumulative,
            'expected_cumulative': expected_cumulative,
            'risk_regime': self._classify_risk_regime(rolling_beta)
        }, index=asset_returns.index)
        
        return result_df
    
    def _calculate_rolling_beta(self, asset_returns: pd.Series, market_returns: pd.Series) -> pd.Series:
        """Calculate rolling beta"""
        def rolling_beta_calc(window_data):
            if len(window_data) < self.min_periods:
                return np.nan
            
            asset_window = window_data['asset']
            market_window = window_data['market']
            
            if market_window.std() == 0:
                return 1.0
            
            return asset_window.cov(market_window) / market_window.var()
        
        # Combine returns for rolling calculation
        combined = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        })
        
        rolling_beta = combined.rolling(window=self.rolling_window, min_periods=self.min_periods).apply(
            lambda x: rolling_beta_calc(x), raw=False
        )['asset']
        
        return rolling_beta
    
    def _calculate_rolling_alpha(self, asset_returns: pd.Series, market_returns: pd.Series,
                                rolling_beta: pd.Series) -> pd.Series:
        """Calculate rolling alpha"""
        # Rolling mean returns
        rolling_asset_mean = asset_returns.rolling(self.rolling_window).mean()
        rolling_market_mean = market_returns.rolling(self.rolling_window).mean()
        
        # Alpha = Asset Return - Beta * Market Return
        rolling_alpha = rolling_asset_mean - rolling_beta * rolling_market_mean
        
        return rolling_alpha
    
    def _calculate_rolling_sharpe(self, returns: pd.Series, risk_free_rate: float) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        excess_returns = returns - risk_free_rate / 252
        
        rolling_mean = excess_returns.rolling(self.rolling_window).mean()
        rolling_std = excess_returns.rolling(self.rolling_window).std()
        
        rolling_sharpe = (rolling_mean * 252) / (rolling_std * np.sqrt(252))
        
        return rolling_sharpe
    
    def _classify_risk_regime(self, rolling_beta: pd.Series) -> pd.Series:
        """Classify risk regimes based on rolling beta"""
        def classify_beta(beta):
            if pd.isna(beta):
                return "UNKNOWN"
            elif beta < 0.5:
                return "LOW_RISK"
            elif beta < 0.8:
                return "DEFENSIVE"
            elif beta < 1.2:
                return "MARKET_RISK"
            elif beta < 1.5:
                return "AGGRESSIVE"
            else:
                return "HIGH_RISK"
        
        return rolling_beta.apply(classify_beta)
    
    def _analyze_beta_stability(self, values_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze beta stability over time"""
        if 'rolling_beta' not in values_df.columns:
            return {'stability': 'UNKNOWN'}
        
        rolling_beta = values_df['rolling_beta'].dropna()
        
        if len(rolling_beta) < 10:
            return {'stability': 'INSUFFICIENT_DATA'}
        
        # Calculate beta volatility
        beta_volatility = rolling_beta.std()
        beta_mean = rolling_beta.mean()
        
        # Coefficient of variation
        cv = abs(beta_volatility / beta_mean) if beta_mean != 0 else float('inf')
        
        # Stability classification
        if cv < 0.1:
            stability = "VERY_STABLE"
        elif cv < 0.2:
            stability = "STABLE"
        elif cv < 0.4:
            stability = "MODERATE"
        else:
            stability = "UNSTABLE"
        
        return {
            'stability': stability,
            'beta_volatility': beta_volatility,
            'coefficient_of_variation': cv,
            'beta_range': [rolling_beta.min(), rolling_beta.max()],
            'trend': 'INCREASING' if rolling_beta.iloc[-10:].mean() > rolling_beta.iloc[:10].mean() else 'DECREASING'
        }
    
    def _analyze_risk_attribution(self, systematic_risk: float, unsystematic_risk: float) -> Dict[str, Any]:
        """Analyze risk attribution"""
        total_risk = np.sqrt(systematic_risk**2 + unsystematic_risk**2)
        
        if total_risk == 0:
            return {'systematic_pct': 0, 'unsystematic_pct': 0, 'diversification_ratio': 0}
        
        systematic_pct = (systematic_risk / total_risk) * 100
        unsystematic_pct = (unsystematic_risk / total_risk) * 100
        
        # Diversification ratio (higher is better)
        diversification_ratio = systematic_risk / total_risk
        
        return {
            'systematic_pct': systematic_pct,
            'unsystematic_pct': unsystematic_pct,
            'total_risk': total_risk,
            'diversification_ratio': diversification_ratio,
            'risk_profile': 'WELL_DIVERSIFIED' if systematic_pct > 70 else 'CONCENTRATED'
        }
    
    def _analyze_market_timing(self, asset_returns: pd.Series, market_returns: pd.Series) -> Dict[str, Any]:
        """Analyze market timing ability"""
        # Calculate up/down market performance
        up_market = market_returns > market_returns.median()
        down_market = ~up_market
        
        up_market_performance = asset_returns[up_market].mean() if up_market.sum() > 0 else 0
        down_market_performance = asset_returns[down_market].mean() if down_market.sum() > 0 else 0
        
        # Market timing score
        market_up_return = market_returns[up_market].mean() if up_market.sum() > 0 else 0
        market_down_return = market_returns[down_market].mean() if down_market.sum() > 0 else 0
        
        timing_score = 0
        if market_up_return != 0 and market_down_return != 0:
            up_ratio = up_market_performance / market_up_return
            down_ratio = down_market_performance / market_down_return
            timing_score = up_ratio - down_ratio
        
        return {
            'up_market_performance': up_market_performance * 252,  # Annualized
            'down_market_performance': down_market_performance * 252,
            'timing_score': timing_score,
            'timing_ability': 'GOOD' if timing_score > 0.1 else 'POOR' if timing_score < -0.1 else 'NEUTRAL'
        }
    
    def _analyze_market_regimes(self, asset_returns: pd.Series, market_returns: pd.Series) -> Dict[str, Any]:
        """Analyze performance across different market regimes"""
        # Define market regimes based on volatility
        market_vol = market_returns.rolling(21).std()  # 21-day rolling volatility
        vol_median = market_vol.median()
        
        high_vol = market_vol > vol_median * 1.5
        low_vol = market_vol < vol_median * 0.5
        normal_vol = ~(high_vol | low_vol)
        
        regimes = {
            'high_volatility': {
                'asset_return': asset_returns[high_vol].mean() * 252 if high_vol.sum() > 0 else 0,
                'market_return': market_returns[high_vol].mean() * 252 if high_vol.sum() > 0 else 0,
                'periods': high_vol.sum()
            },
            'normal_volatility': {
                'asset_return': asset_returns[normal_vol].mean() * 252 if normal_vol.sum() > 0 else 0,
                'market_return': market_returns[normal_vol].mean() * 252 if normal_vol.sum() > 0 else 0,
                'periods': normal_vol.sum()
            },
            'low_volatility': {
                'asset_return': asset_returns[low_vol].mean() * 252 if low_vol.sum() > 0 else 0,
                'market_return': market_returns[low_vol].mean() * 252 if low_vol.sum() > 0 else 0,
                'periods': low_vol.sum()
            }
        }
        
        return regimes
    
    def _calculate_confidence(self, r_squared: float, data_length: int) -> float:
        """Calculate confidence score based on model fit and data quality"""
        confidence = 0.3  # Base confidence
        
        # Adjust based on R-squared
        confidence += r_squared * 0.4
        
        # Adjust based on data length
        if data_length >= 252:
            confidence += 0.2
        elif data_length >= 126:
            confidence += 0.1
        
        # Adjust based on minimum periods
        if data_length >= self.min_periods * 2:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _get_interpretation(self, beta: float, alpha: float, jensen_alpha: float, 
                          r_squared: float) -> str:
        """Get interpretation of CAPM results"""
        # Beta interpretation
        if beta > 1.2:
            beta_desc = "high-beta (aggressive)"
        elif beta < 0.8:
            beta_desc = "low-beta (defensive)"
        else:
            beta_desc = "market-beta (neutral)"
        
        # Alpha interpretation
        if jensen_alpha > 0.03:
            alpha_desc = "with strong outperformance"
        elif jensen_alpha > 0:
            alpha_desc = "with modest outperformance"
        elif jensen_alpha > -0.03:
            alpha_desc = "with market-level performance"
        else:
            alpha_desc = "with underperformance"
        
        # Model fit
        if r_squared > 0.7:
            fit_desc = "High model reliability"
        elif r_squared > 0.4:
            fit_desc = "Moderate model reliability"
        else:
            fit_desc = "Low model reliability"
        
        return f"Asset exhibits {beta_desc} characteristics {alpha_desc}. {fit_desc}."
    
    def _empty_result(self, asset_type: AssetType) -> CAPMResult:
        """Return empty result for error cases"""
        return CAPMResult(
            name="Capital Asset Pricing Model",
            beta=1.0,
            alpha=0.0,
            expected_return=0.0,
            risk_free_rate=self.risk_free_rate,
            market_return=0.0,
            market_premium=0.0,
            r_squared=0.0,
            systematic_risk=0.0,
            unsystematic_risk=0.0,
            sharpe_ratio=0.0,
            treynor_ratio=0.0,
            jensen_alpha=0.0,
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.FUNDAMENTAL,
            signals=["ERROR"]
        )
    
    def get_chart_data(self, result: CAPMResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'capm_analysis',
            'name': 'Capital Asset Pricing Model',
            'data': {
                'asset_returns': result.values['asset_returns'].tolist() if 'asset_returns' in result.values.columns else [],
                'market_returns': result.values['market_returns'].tolist() if 'market_returns' in result.values.columns else [],
                'rolling_beta': result.values['rolling_beta'].tolist() if 'rolling_beta' in result.values.columns else [],
                'expected_returns': result.values['expected_returns'].tolist() if 'expected_returns' in result.values.columns else [],
                'active_returns': result.values['active_returns'].tolist() if 'active_returns' in result.values.columns else []
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'capm_metrics': {
                'beta': result.beta,
                'alpha': result.alpha,
                'expected_return': result.expected_return,
                'jensen_alpha': result.jensen_alpha,
                'sharpe_ratio': result.sharpe_ratio,
                'treynor_ratio': result.treynor_ratio,
                'r_squared': result.r_squared
            },
            'risk_metrics': {
                'systematic_risk': result.systematic_risk,
                'unsystematic_risk': result.unsystematic_risk,
                'total_risk': np.sqrt(result.systematic_risk**2 + result.unsystematic_risk**2)
            },
            'series': [
                {
                    'name': 'Asset Cumulative Returns',
                    'data': result.values['asset_cumulative'].tolist() if 'asset_cumulative' in result.values.columns else [],
                    'color': '#2196F3',
                    'type': 'line',
                    'lineWidth': 2
                },
                {
                    'name': 'Market Cumulative Returns',
                    'data': result.values['market_cumulative'].tolist() if 'market_cumulative' in result.values.columns else [],
                    'color': '#4CAF50',
                    'type': 'line',
                    'lineWidth': 2
                },
                {
                    'name': 'Expected Returns (CAPM)',
                    'data': result.values['expected_cumulative'].tolist() if 'expected_cumulative' in result.values.columns else [],
                    'color': '#FF9800',
                    'type': 'line',
                    'lineWidth': 2,
                    'dashStyle': 'Dash'
                },
                {
                    'name': 'Rolling Beta',
                    'data': result.values['rolling_beta'].tolist() if 'rolling_beta' in result.values.columns else [],
                    'color': '#9C27B0',
                    'type': 'line',
                    'lineWidth': 1,
                    'yAxis': 1
                }
            ]
        }


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate correlated returns
    market_returns = np.random.randn(252) * 0.01
    asset_returns = 1.2 * market_returns + np.random.randn(252) * 0.005  # Beta = 1.2
    
    market_prices = 100 * (1 + market_returns).cumprod()
    asset_prices = 50 * (1 + asset_returns).cumprod()
    
    asset_data = pd.DataFrame({'close': asset_prices}, index=dates)
    market_data = pd.DataFrame({'close': market_prices}, index=dates)
    
    # Calculate CAPM
    capm_calculator = CAPMIndicator()
    result = capm_calculator.calculate(asset_data, market_data, asset_type=AssetType.STOCK)
    
    print(f"CAPM Analysis:")
    print(f"Beta: {result.beta:.3f}")
    print(f"Alpha: {result.alpha:.4f}")
    print(f"Expected Return: {result.expected_return:.2%}")
    print(f"Jensen's Alpha: {result.jensen_alpha:.2%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"Treynor Ratio: {result.treynor_ratio:.3f}")
    print(f"R-squared: {result.r_squared:.3f}")
    print(f"Systematic Risk: {result.systematic_risk:.2%}")
    print(f"Unsystematic Risk: {result.unsystematic_risk:.2%}")
    print(f"Signals: {', '.join(result.signals)}")