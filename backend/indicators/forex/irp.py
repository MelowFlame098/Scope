"""Interest Rate Parity (IRP) Analysis

Interest Rate Parity is a fundamental theory in international finance that relates
interest rates, exchange rates, and inflation. It suggests that the difference in
interest rates between two countries should equal the expected change in exchange rates.

This module implements both Covered Interest Rate Parity (CIRP) and Uncovered
Interest Rate Parity (UIRP) with advanced carry trade analysis.

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
from scipy.optimize import minimize

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
    ECONOMIC = "economic"
    MONETARY = "monetary"


@dataclass
class IRPResult:
    """Result of Interest Rate Parity calculation"""
    name: str
    covered_irp_rate: float
    uncovered_irp_rate: float
    current_rate: float
    cirp_deviation: float
    uirp_deviation: float
    carry_trade_signal: str
    carry_trade_return: float
    risk_adjusted_carry: float
    forward_premium: float
    interest_differential: float
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class IRPIndicator:
    """Interest Rate Parity Calculator with Advanced Carry Trade Analysis"""
    
    def __init__(self, base_country: str = "US", quote_country: str = "EU",
                 risk_free_proxy: str = "government_bond", carry_threshold: float = 0.5,
                 volatility_window: int = 63):
        """
        Initialize IRP calculator
        
        Args:
            base_country: Base currency country code (default: "US")
            quote_country: Quote currency country code (default: "EU")
            risk_free_proxy: Type of risk-free rate proxy (default: "government_bond")
            carry_threshold: Minimum interest differential for carry trade (default: 0.5%)
            volatility_window: Window for volatility calculations (default: 63 days)
        """
        self.base_country = base_country
        self.quote_country = quote_country
        self.risk_free_proxy = risk_free_proxy
        self.carry_threshold = carry_threshold
        self.volatility_window = volatility_window
        self.logger = logging.getLogger(__name__)
        
        # Initialize country interest rate data
        self.interest_rate_data = self._initialize_interest_rates()
    
    def calculate(self, data: pd.DataFrame, interest_rates: Optional[Dict] = None,
                 forward_rates: Optional[pd.Series] = None, custom_base: Optional[str] = None,
                 custom_quote: Optional[str] = None, asset_type: AssetType = AssetType.FOREX) -> IRPResult:
        """
        Calculate Interest Rate Parity analysis
        
        Args:
            data: Exchange rate data DataFrame with 'close' column
            interest_rates: Dictionary containing interest rates for both countries
            forward_rates: Forward exchange rates (if available)
            custom_base: Override base country
            custom_quote: Override quote country
            asset_type: Type of asset being analyzed
            
        Returns:
            IRPResult containing IRP analysis
        """
        try:
            # Use custom countries if provided
            base_country = custom_base or self.base_country
            quote_country = custom_quote or self.quote_country
            
            # Prepare interest rate data
            rate_data = self._prepare_interest_rate_data(interest_rates, base_country, quote_country)
            
            # Calculate Covered Interest Rate Parity (CIRP)
            cirp_rate, cirp_deviation = self._calculate_covered_irp(data, rate_data, forward_rates)
            
            # Calculate Uncovered Interest Rate Parity (UIRP)
            uirp_rate, uirp_deviation = self._calculate_uncovered_irp(data, rate_data)
            
            # Carry trade analysis
            carry_signal, carry_return, risk_adj_carry = self._analyze_carry_trade(
                data, rate_data, cirp_deviation, uirp_deviation
            )
            
            # Forward premium calculation
            forward_premium = self._calculate_forward_premium(rate_data, forward_rates)
            
            # Risk analysis
            risk_metrics = self._calculate_risk_metrics(data, rate_data)
            
            # Generate comprehensive signals
            signals = self._generate_signals(cirp_deviation, uirp_deviation, carry_signal,
                                           rate_data, risk_metrics)
            
            # Create time series data
            values_df = self._create_time_series(data, rate_data, cirp_rate, uirp_rate,
                                               forward_rates, risk_metrics)
            
            # Calculate confidence
            confidence = self._calculate_confidence(rate_data, len(data), forward_rates is not None)
            
            current_rate = data['close'].iloc[-1]
            interest_differential = rate_data['interest_differential']
            
            return IRPResult(
                name="Interest Rate Parity",
                covered_irp_rate=cirp_rate.iloc[-1] if isinstance(cirp_rate, pd.Series) else cirp_rate,
                uncovered_irp_rate=uirp_rate.iloc[-1] if isinstance(uirp_rate, pd.Series) else uirp_rate,
                current_rate=current_rate,
                cirp_deviation=cirp_deviation.iloc[-1] if isinstance(cirp_deviation, pd.Series) else cirp_deviation,
                uirp_deviation=uirp_deviation.iloc[-1] if isinstance(uirp_deviation, pd.Series) else uirp_deviation,
                carry_trade_signal=carry_signal,
                carry_trade_return=carry_return,
                risk_adjusted_carry=risk_adj_carry,
                forward_premium=forward_premium,
                interest_differential=interest_differential,
                values=values_df,
                metadata={
                    'base_country': base_country,
                    'quote_country': quote_country,
                    'base_interest_rate': rate_data['base_rate'],
                    'quote_interest_rate': rate_data['quote_rate'],
                    'interest_differential': interest_differential,
                    'carry_threshold': self.carry_threshold,
                    'volatility_analysis': risk_metrics,
                    'cirp_arbitrage_opportunity': abs(cirp_deviation.iloc[-1]) > 0.1 if isinstance(cirp_deviation, pd.Series) else abs(cirp_deviation) > 0.1,
                    'uirp_forecast_accuracy': self._test_uirp_forecast_accuracy(data, rate_data),
                    'carry_trade_performance': self._analyze_carry_performance(values_df),
                    'term_structure_analysis': self._analyze_term_structure(rate_data),
                    'currency_risk_premium': self._estimate_currency_risk_premium(data, rate_data),
                    'interpretation': self._get_interpretation(cirp_deviation.iloc[-1] if isinstance(cirp_deviation, pd.Series) else cirp_deviation,
                                                            uirp_deviation.iloc[-1] if isinstance(uirp_deviation, pd.Series) else uirp_deviation,
                                                            carry_signal, interest_differential)
                },
                confidence=confidence,
                timestamp=datetime.now(),
                asset_type=asset_type,
                category=IndicatorCategory.MONETARY,
                signals=signals
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating IRP: {e}")
            return self._empty_result(asset_type)
    
    def _initialize_interest_rates(self) -> Dict[str, Dict[str, float]]:
        """Initialize default interest rate data for major currencies"""
        return {
            'US': {
                'short_term': 0.0525,  # 5.25% Fed Funds Rate
                'long_term': 0.045,    # 4.5% 10-year Treasury
                'policy_rate': 0.0525,
                'real_rate': 0.025
            },
            'EU': {
                'short_term': 0.04,    # 4.0% ECB Rate
                'long_term': 0.035,    # 3.5% 10-year Bund
                'policy_rate': 0.04,
                'real_rate': 0.018
            },
            'JP': {
                'short_term': -0.001,  # -0.1% BOJ Rate
                'long_term': 0.008,    # 0.8% 10-year JGB
                'policy_rate': -0.001,
                'real_rate': -0.006
            },
            'GB': {
                'short_term': 0.0525,  # 5.25% BOE Rate
                'long_term': 0.042,    # 4.2% 10-year Gilt
                'policy_rate': 0.0525,
                'real_rate': 0.022
            },
            'CA': {
                'short_term': 0.05,    # 5.0% BOC Rate
                'long_term': 0.038,    # 3.8% 10-year Bond
                'policy_rate': 0.05,
                'real_rate': 0.028
            },
            'AU': {
                'short_term': 0.041,   # 4.1% RBA Rate
                'long_term': 0.042,    # 4.2% 10-year Bond
                'policy_rate': 0.041,
                'real_rate': 0.013
            },
            'CH': {
                'short_term': 0.0175,  # 1.75% SNB Rate
                'long_term': 0.008,    # 0.8% 10-year Bond
                'policy_rate': 0.0175,
                'real_rate': 0.008
            },
            'NZ': {
                'short_term': 0.055,   # 5.5% RBNZ Rate
                'long_term': 0.048,    # 4.8% 10-year Bond
                'policy_rate': 0.055,
                'real_rate': 0.025
            }
        }
    
    def _prepare_interest_rate_data(self, interest_rates: Optional[Dict], 
                                   base_country: str, quote_country: str) -> Dict[str, Any]:
        """Prepare interest rate data for calculations"""
        # Use provided rates or defaults
        if interest_rates:
            base_rate = interest_rates.get('base_rate', 
                                         self.interest_rate_data.get(base_country, {}).get('short_term', 0.025))
            quote_rate = interest_rates.get('quote_rate', 
                                          self.interest_rate_data.get(quote_country, {}).get('short_term', 0.02))
        else:
            base_rate = self.interest_rate_data.get(base_country, {}).get('short_term', 0.025)
            quote_rate = self.interest_rate_data.get(quote_country, {}).get('short_term', 0.02)
        
        # Additional rate data
        base_long_rate = self.interest_rate_data.get(base_country, {}).get('long_term', base_rate)
        quote_long_rate = self.interest_rate_data.get(quote_country, {}).get('long_term', quote_rate)
        
        base_real_rate = self.interest_rate_data.get(base_country, {}).get('real_rate', base_rate - 0.025)
        quote_real_rate = self.interest_rate_data.get(quote_country, {}).get('real_rate', quote_rate - 0.02)
        
        return {
            'base_rate': base_rate,
            'quote_rate': quote_rate,
            'interest_differential': base_rate - quote_rate,
            'base_long_rate': base_long_rate,
            'quote_long_rate': quote_long_rate,
            'long_rate_differential': base_long_rate - quote_long_rate,
            'base_real_rate': base_real_rate,
            'quote_real_rate': quote_real_rate,
            'real_rate_differential': base_real_rate - quote_real_rate,
            'yield_curve_slope_base': base_long_rate - base_rate,
            'yield_curve_slope_quote': quote_long_rate - quote_rate
        }
    
    def _calculate_covered_irp(self, data: pd.DataFrame, rate_data: Dict[str, Any],
                              forward_rates: Optional[pd.Series] = None) -> Tuple[pd.Series, pd.Series]:
        """Calculate Covered Interest Rate Parity"""
        # CIRP: F = S * (1 + r_base) / (1 + r_quote)
        # Where F is forward rate, S is spot rate
        
        spot_rate = data['close']
        base_rate = rate_data['base_rate']
        quote_rate = rate_data['quote_rate']
        
        # Time to maturity (assuming 1 year for simplicity, can be adjusted)
        time_to_maturity = 1.0
        
        # Theoretical forward rate based on CIRP
        theoretical_forward = spot_rate * ((1 + base_rate * time_to_maturity) / 
                                         (1 + quote_rate * time_to_maturity))
        
        if forward_rates is not None:
            # Use actual forward rates if available
            actual_forward = forward_rates
            # CIRP deviation: (Actual Forward - Theoretical Forward) / Theoretical Forward
            cirp_deviation = (actual_forward - theoretical_forward) / theoretical_forward * 100
        else:
            # If no forward rates, assume current spot rate as "forward" (simplified)
            actual_forward = spot_rate
            cirp_deviation = (actual_forward - theoretical_forward) / theoretical_forward * 100
        
        return theoretical_forward, cirp_deviation
    
    def _calculate_uncovered_irp(self, data: pd.DataFrame, rate_data: Dict[str, Any]) -> Tuple[pd.Series, pd.Series]:
        """Calculate Uncovered Interest Rate Parity"""
        # UIRP: Expected change in exchange rate = Interest rate differential
        # E[ΔS/S] = r_base - r_quote
        
        spot_rate = data['close']
        interest_diff = rate_data['interest_differential']
        
        # Calculate expected future spot rate based on UIRP
        # S_expected = S_current * (1 + interest_differential)
        expected_future_rate = spot_rate * (1 + interest_diff)
        
        # Calculate actual rate changes (using forward-looking if available, or historical)
        if len(data) > 252:  # If we have more than 1 year of data
            # Use 1-year forward rate change
            actual_rate_change = spot_rate.pct_change(252).shift(-252)  # 1-year forward change
        else:
            # Use available data period
            period = min(63, len(data) // 2)  # Quarterly or half available period
            actual_rate_change = spot_rate.pct_change(period).shift(-period)
        
        # UIRP deviation: Actual change vs Expected change
        expected_change = pd.Series(interest_diff, index=spot_rate.index)
        uirp_deviation = (actual_rate_change - expected_change) * 100
        
        return expected_future_rate, uirp_deviation.fillna(0)
    
    def _analyze_carry_trade(self, data: pd.DataFrame, rate_data: Dict[str, Any],
                           cirp_deviation: pd.Series, uirp_deviation: pd.Series) -> Tuple[str, float, float]:
        """Analyze carry trade opportunities"""
        interest_diff = rate_data['interest_differential']
        
        # Calculate exchange rate volatility
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(self.volatility_window).std() * np.sqrt(252)  # Annualized
        current_volatility = volatility.iloc[-1] if not volatility.empty else 0.15
        
        # Carry trade return (interest differential)
        carry_return = interest_diff * 100  # Convert to percentage
        
        # Risk-adjusted carry (Sharpe-like ratio)
        if current_volatility > 0:
            risk_adjusted_carry = carry_return / (current_volatility * 100)
        else:
            risk_adjusted_carry = 0
        
        # Generate carry trade signal
        if abs(interest_diff) < self.carry_threshold / 100:
            carry_signal = "NO_CARRY"
        elif interest_diff > self.carry_threshold / 100:
            if risk_adjusted_carry > 0.5:  # Good risk-adjusted return
                carry_signal = "STRONG_CARRY_LONG"
            else:
                carry_signal = "WEAK_CARRY_LONG"
        else:  # interest_diff < -self.carry_threshold / 100
            if risk_adjusted_carry < -0.5:  # Good risk-adjusted return for short
                carry_signal = "STRONG_CARRY_SHORT"
            else:
                carry_signal = "WEAK_CARRY_SHORT"
        
        # Adjust signal based on UIRP deviation (momentum factor)
        recent_uirp = uirp_deviation.iloc[-21:].mean()  # Recent 21-day average
        if abs(recent_uirp) > 2:  # Significant UIRP deviation
            if (interest_diff > 0 and recent_uirp < -1) or (interest_diff < 0 and recent_uirp > 1):
                # UIRP suggests currency will move against carry trade
                if "STRONG" in carry_signal:
                    carry_signal = carry_signal.replace("STRONG", "WEAK")
                elif "WEAK" in carry_signal:
                    carry_signal = "NO_CARRY"
        
        return carry_signal, carry_return, risk_adjusted_carry
    
    def _calculate_forward_premium(self, rate_data: Dict[str, Any], 
                                  forward_rates: Optional[pd.Series] = None) -> float:
        """Calculate forward premium/discount"""
        interest_diff = rate_data['interest_differential']
        
        # Theoretical forward premium based on interest rate differential
        # Forward Premium ≈ Interest Rate Differential (for small rates)
        theoretical_premium = interest_diff * 100
        
        return theoretical_premium
    
    def _calculate_risk_metrics(self, data: pd.DataFrame, rate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive risk metrics"""
        returns = data['close'].pct_change().dropna()
        
        # Volatility measures
        daily_vol = returns.std()
        annualized_vol = daily_vol * np.sqrt(252)
        
        # Rolling volatility
        rolling_vol = returns.rolling(self.volatility_window).std() * np.sqrt(252)
        vol_of_vol = rolling_vol.std()  # Volatility of volatility
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        # Value at Risk (VaR)
        var_95 = returns.quantile(0.05)
        var_99 = returns.quantile(0.01)
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Correlation with interest rate differential (if time series available)
        interest_diff = rate_data['interest_differential']
        
        # Currency beta (sensitivity to interest rate changes)
        # Simplified calculation
        currency_beta = returns.corr(pd.Series(interest_diff, index=returns.index[:1]).reindex(returns.index, method='ffill'))
        
        return {
            'daily_volatility': daily_vol,
            'annualized_volatility': annualized_vol,
            'volatility_of_volatility': vol_of_vol,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown,
            'currency_beta': currency_beta if not pd.isna(currency_beta) else 0,
            'current_volatility_percentile': self._calculate_volatility_percentile(rolling_vol)
        }
    
    def _calculate_volatility_percentile(self, rolling_vol: pd.Series) -> float:
        """Calculate current volatility percentile"""
        if len(rolling_vol.dropna()) < 20:
            return 50.0  # Default to median if insufficient data
        
        current_vol = rolling_vol.iloc[-1]
        return stats.percentileofscore(rolling_vol.dropna(), current_vol)
    
    def _test_uirp_forecast_accuracy(self, data: pd.DataFrame, rate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test UIRP forecast accuracy"""
        try:
            if len(data) < 126:  # Need at least 6 months of data
                return {'accuracy': 'INSUFFICIENT_DATA'}
            
            spot_rate = data['close']
            interest_diff = rate_data['interest_differential']
            
            # Calculate 3-month forward predictions vs actual
            prediction_horizon = 63  # 3 months
            
            # UIRP prediction: future rate change = interest differential
            uirp_predictions = pd.Series(interest_diff, index=spot_rate.index)
            
            # Actual rate changes
            actual_changes = spot_rate.pct_change(prediction_horizon).shift(-prediction_horizon)
            
            # Calculate forecast errors
            forecast_errors = actual_changes - uirp_predictions
            forecast_errors = forecast_errors.dropna()
            
            if len(forecast_errors) < 10:
                return {'accuracy': 'INSUFFICIENT_DATA'}
            
            # Accuracy metrics
            mae = forecast_errors.abs().mean()  # Mean Absolute Error
            rmse = np.sqrt((forecast_errors ** 2).mean())  # Root Mean Square Error
            bias = forecast_errors.mean()  # Forecast bias
            
            # Direction accuracy
            predicted_direction = np.sign(uirp_predictions)
            actual_direction = np.sign(actual_changes)
            direction_accuracy = (predicted_direction == actual_direction).mean()
            
            return {
                'mae': mae,
                'rmse': rmse,
                'bias': bias,
                'direction_accuracy': direction_accuracy,
                'forecast_quality': 'GOOD' if direction_accuracy > 0.6 else 'MODERATE' if direction_accuracy > 0.4 else 'POOR'
            }
            
        except Exception as e:
            self.logger.warning(f"UIRP forecast accuracy test failed: {e}")
            return {'accuracy': 'ERROR'}
    
    def _analyze_carry_performance(self, values_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze historical carry trade performance"""
        try:
            if 'carry_return' not in values_df.columns or len(values_df) < 63:
                return {'performance': 'INSUFFICIENT_DATA'}
            
            carry_returns = values_df['carry_return'].dropna()
            
            # Performance metrics
            total_return = carry_returns.sum()
            annualized_return = carry_returns.mean() * 252
            volatility = carry_returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + carry_returns / 100).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Win rate
            win_rate = (carry_returns > 0).mean()
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'performance_rating': self._rate_carry_performance(sharpe_ratio, max_drawdown, win_rate)
            }
            
        except Exception as e:
            self.logger.warning(f"Carry performance analysis failed: {e}")
            return {'performance': 'ERROR'}
    
    def _rate_carry_performance(self, sharpe: float, max_dd: float, win_rate: float) -> str:
        """Rate carry trade performance"""
        score = 0
        
        # Sharpe ratio scoring
        if sharpe > 1.0:
            score += 3
        elif sharpe > 0.5:
            score += 2
        elif sharpe > 0:
            score += 1
        
        # Max drawdown scoring (less negative is better)
        if max_dd > -0.05:  # Less than 5% drawdown
            score += 3
        elif max_dd > -0.10:  # Less than 10% drawdown
            score += 2
        elif max_dd > -0.20:  # Less than 20% drawdown
            score += 1
        
        # Win rate scoring
        if win_rate > 0.6:
            score += 3
        elif win_rate > 0.5:
            score += 2
        elif win_rate > 0.4:
            score += 1
        
        # Overall rating
        if score >= 7:
            return "EXCELLENT"
        elif score >= 5:
            return "GOOD"
        elif score >= 3:
            return "MODERATE"
        else:
            return "POOR"
    
    def _analyze_term_structure(self, rate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze yield curve and term structure implications"""
        base_slope = rate_data['yield_curve_slope_base']
        quote_slope = rate_data['yield_curve_slope_quote']
        slope_differential = base_slope - quote_slope
        
        # Interpret yield curve shapes
        def interpret_curve(slope):
            if slope > 0.02:  # More than 2% slope
                return "STEEP"
            elif slope > 0.005:  # More than 0.5% slope
                return "NORMAL"
            elif slope > -0.005:  # Flat
                return "FLAT"
            else:  # Inverted
                return "INVERTED"
        
        base_curve_shape = interpret_curve(base_slope)
        quote_curve_shape = interpret_curve(quote_slope)
        
        # Term structure implications for currency
        if slope_differential > 0.01:
            term_structure_signal = "BULLISH_BASE_CURRENCY"
        elif slope_differential < -0.01:
            term_structure_signal = "BEARISH_BASE_CURRENCY"
        else:
            term_structure_signal = "NEUTRAL"
        
        return {
            'base_curve_slope': base_slope,
            'quote_curve_slope': quote_slope,
            'slope_differential': slope_differential,
            'base_curve_shape': base_curve_shape,
            'quote_curve_shape': quote_curve_shape,
            'term_structure_signal': term_structure_signal
        }
    
    def _estimate_currency_risk_premium(self, data: pd.DataFrame, rate_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate currency risk premium"""
        try:
            returns = data['close'].pct_change().dropna()
            interest_diff = rate_data['interest_differential']
            
            # Estimate risk premium as excess return over interest differential
            # Risk Premium = Actual Return - Interest Differential
            
            # Calculate rolling excess returns
            rolling_returns = returns.rolling(252).sum()  # Annual returns
            excess_returns = rolling_returns - interest_diff
            
            current_risk_premium = excess_returns.iloc[-1] if not excess_returns.empty else 0
            avg_risk_premium = excess_returns.mean()
            
            # Risk premium volatility
            risk_premium_vol = excess_returns.std()
            
            return {
                'current_risk_premium': current_risk_premium,
                'average_risk_premium': avg_risk_premium,
                'risk_premium_volatility': risk_premium_vol,
                'risk_premium_interpretation': self._interpret_risk_premium(current_risk_premium)
            }
            
        except Exception as e:
            self.logger.warning(f"Risk premium estimation failed: {e}")
            return {'risk_premium': 'ERROR'}
    
    def _interpret_risk_premium(self, risk_premium: float) -> str:
        """Interpret currency risk premium"""
        if risk_premium > 0.02:  # More than 2%
            return "HIGH_RISK_PREMIUM"
        elif risk_premium > 0.005:  # More than 0.5%
            return "MODERATE_RISK_PREMIUM"
        elif risk_premium > -0.005:  # Between -0.5% and 0.5%
            return "LOW_RISK_PREMIUM"
        else:  # Less than -0.5%
            return "NEGATIVE_RISK_PREMIUM"
    
    def _generate_signals(self, cirp_deviation: pd.Series, uirp_deviation: pd.Series,
                         carry_signal: str, rate_data: Dict[str, Any], 
                         risk_metrics: Dict[str, Any]) -> List[str]:
        """Generate comprehensive trading signals"""
        signals = []
        
        # CIRP arbitrage signals
        current_cirp = cirp_deviation.iloc[-1] if isinstance(cirp_deviation, pd.Series) else cirp_deviation
        if abs(current_cirp) > 0.1:  # More than 0.1% deviation
            if current_cirp > 0.1:
                signals.append("CIRP_ARBITRAGE_SELL")
            else:
                signals.append("CIRP_ARBITRAGE_BUY")
        
        # UIRP momentum signals
        current_uirp = uirp_deviation.iloc[-1] if isinstance(uirp_deviation, pd.Series) else uirp_deviation
        if abs(current_uirp) > 1:  # More than 1% deviation
            if current_uirp > 1:
                signals.append("UIRP_MOMENTUM_SELL")
            else:
                signals.append("UIRP_MOMENTUM_BUY")
        
        # Carry trade signals
        signals.append(f"CARRY_{carry_signal}")
        
        # Interest rate differential signals
        interest_diff = rate_data['interest_differential']
        if abs(interest_diff) > 0.02:  # More than 2% differential
            if interest_diff > 0.02:
                signals.append("HIGH_YIELD_ADVANTAGE")
            else:
                signals.append("LOW_YIELD_DISADVANTAGE")
        
        # Volatility-based signals
        vol_percentile = risk_metrics.get('current_volatility_percentile', 50)
        if vol_percentile > 80:
            signals.append("HIGH_VOLATILITY_REGIME")
        elif vol_percentile < 20:
            signals.append("LOW_VOLATILITY_REGIME")
        
        # Risk-adjusted signals
        annualized_vol = risk_metrics.get('annualized_volatility', 0.15)
        if interest_diff / annualized_vol > 0.5:  # Good risk-adjusted carry
            signals.append("ATTRACTIVE_RISK_ADJUSTED_CARRY")
        elif interest_diff / annualized_vol < -0.5:
            signals.append("POOR_RISK_ADJUSTED_CARRY")
        
        # Term structure signals
        slope_diff = rate_data.get('yield_curve_slope_base', 0) - rate_data.get('yield_curve_slope_quote', 0)
        if abs(slope_diff) > 0.01:
            if slope_diff > 0.01:
                signals.append("YIELD_CURVE_BULLISH")
            else:
                signals.append("YIELD_CURVE_BEARISH")
        
        return signals
    
    def _create_time_series(self, data: pd.DataFrame, rate_data: Dict[str, Any],
                           cirp_rate: pd.Series, uirp_rate: pd.Series,
                           forward_rates: Optional[pd.Series], 
                           risk_metrics: Dict[str, Any]) -> pd.DataFrame:
        """Create comprehensive time series DataFrame"""
        # Base data
        result_df = pd.DataFrame({
            'exchange_rate': data['close'],
            'cirp_theoretical_rate': cirp_rate,
            'uirp_expected_rate': uirp_rate
        }, index=data.index)
        
        # Add forward rates if available
        if forward_rates is not None:
            result_df['forward_rate'] = forward_rates
            result_df['forward_premium'] = (forward_rates - data['close']) / data['close'] * 100
        
        # Interest rate data (constant series)
        result_df['interest_differential'] = rate_data['interest_differential']
        result_df['base_rate'] = rate_data['base_rate']
        result_df['quote_rate'] = rate_data['quote_rate']
        
        # Deviations
        result_df['cirp_deviation'] = (data['close'] - cirp_rate) / cirp_rate * 100
        result_df['uirp_deviation'] = (data['close'] - uirp_rate) / uirp_rate * 100
        
        # Carry trade metrics
        result_df['carry_return'] = rate_data['interest_differential'] * 100
        
        # Volatility measures
        returns = data['close'].pct_change()
        result_df['volatility'] = returns.rolling(self.volatility_window).std() * np.sqrt(252) * 100
        result_df['risk_adjusted_carry'] = result_df['carry_return'] / result_df['volatility']
        
        # Rolling correlations and other metrics
        result_df['momentum'] = data['close'].pct_change(21) * 100  # 21-day momentum
        result_df['volatility_percentile'] = result_df['volatility'].rolling(252).rank(pct=True) * 100
        
        # Signal zones
        result_df['irp_signal_zone'] = self._classify_irp_zone(result_df['cirp_deviation'], 
                                                              result_df['uirp_deviation'])
        
        return result_df
    
    def _classify_irp_zone(self, cirp_deviation: pd.Series, uirp_deviation: pd.Series) -> pd.Series:
        """Classify IRP signal zones"""
        def classify_zone(cirp_dev, uirp_dev):
            if pd.isna(cirp_dev) or pd.isna(uirp_dev):
                return "UNKNOWN"
            
            # Strong arbitrage opportunity
            if abs(cirp_dev) > 0.5:
                return "STRONG_ARBITRAGE"
            
            # Moderate arbitrage
            elif abs(cirp_dev) > 0.1:
                return "MODERATE_ARBITRAGE"
            
            # UIRP momentum
            elif abs(uirp_dev) > 2:
                return "MOMENTUM_OPPORTUNITY"
            
            # Fair value
            else:
                return "FAIR_VALUE"
        
        return pd.Series([classify_zone(c, u) for c, u in zip(cirp_deviation, uirp_deviation)], 
                        index=cirp_deviation.index)
    
    def _calculate_confidence(self, rate_data: Dict[str, Any], data_length: int, 
                             has_forward_rates: bool) -> float:
        """Calculate confidence score"""
        confidence = 0.3  # Base confidence
        
        # Data length adjustment
        if data_length >= 1260:  # 5 years
            confidence += 0.2
        elif data_length >= 252:  # 1 year
            confidence += 0.15
        elif data_length >= 63:  # 3 months
            confidence += 0.1
        
        # Interest rate differential significance
        if abs(rate_data['interest_differential']) > 0.01:  # More than 1%
            confidence += 0.15
        
        # Forward rates availability
        if has_forward_rates:
            confidence += 0.2
        
        # Rate data quality (assuming we have good data for major currencies)
        if rate_data.get('base_rate', 0) > 0 and rate_data.get('quote_rate', 0) >= 0:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _get_interpretation(self, cirp_deviation: float, uirp_deviation: float,
                          carry_signal: str, interest_differential: float) -> str:
        """Get interpretation of IRP results"""
        # CIRP interpretation
        if abs(cirp_deviation) > 0.5:
            cirp_text = "significant arbitrage opportunity exists"
        elif abs(cirp_deviation) > 0.1:
            cirp_text = "moderate arbitrage opportunity exists"
        else:
            cirp_text = "no significant arbitrage opportunity"
        
        # UIRP interpretation
        if abs(uirp_deviation) > 2:
            uirp_text = "strong momentum signal present"
        elif abs(uirp_deviation) > 1:
            uirp_text = "moderate momentum signal present"
        else:
            uirp_text = "no significant momentum signal"
        
        # Carry trade interpretation
        if "STRONG" in carry_signal:
            carry_text = "attractive carry trade opportunity"
        elif "WEAK" in carry_signal:
            carry_text = "marginal carry trade opportunity"
        else:
            carry_text = "no significant carry trade opportunity"
        
        # Interest rate context
        if abs(interest_differential) > 0.02:
            rate_context = f"with significant {abs(interest_differential)*100:.1f}% interest rate differential"
        else:
            rate_context = "with minimal interest rate differential"
        
        return f"CIRP analysis shows {cirp_text}. UIRP analysis indicates {uirp_text}. Current conditions suggest {carry_text} {rate_context}."
    
    def _empty_result(self, asset_type: AssetType) -> IRPResult:
        """Return empty result for error cases"""
        return IRPResult(
            name="Interest Rate Parity",
            covered_irp_rate=0.0,
            uncovered_irp_rate=0.0,
            current_rate=0.0,
            cirp_deviation=0.0,
            uirp_deviation=0.0,
            carry_trade_signal="ERROR",
            carry_trade_return=0.0,
            risk_adjusted_carry=0.0,
            forward_premium=0.0,
            interest_differential=0.0,
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.MONETARY,
            signals=["ERROR"]
        )
    
    def get_chart_data(self, result: IRPResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'irp_analysis',
            'name': 'Interest Rate Parity',
            'data': {
                'exchange_rate': result.values['exchange_rate'].tolist() if 'exchange_rate' in result.values.columns else [],
                'cirp_rate': result.values['cirp_theoretical_rate'].tolist() if 'cirp_theoretical_rate' in result.values.columns else [],
                'uirp_rate': result.values['uirp_expected_rate'].tolist() if 'uirp_expected_rate' in result.values.columns else [],
                'cirp_deviation': result.values['cirp_deviation'].tolist() if 'cirp_deviation' in result.values.columns else [],
                'uirp_deviation': result.values['uirp_deviation'].tolist() if 'uirp_deviation' in result.values.columns else [],
                'carry_return': result.values['carry_return'].tolist() if 'carry_return' in result.values.columns else []
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'irp_metrics': {
                'current_rate': result.current_rate,
                'cirp_rate': result.covered_irp_rate,
                'uirp_rate': result.uncovered_irp_rate,
                'cirp_deviation': result.cirp_deviation,
                'uirp_deviation': result.uirp_deviation,
                'carry_signal': result.carry_trade_signal,
                'carry_return': result.carry_trade_return,
                'risk_adjusted_carry': result.risk_adjusted_carry,
                'interest_differential': result.interest_differential
            },
            'series': [
                {
                    'name': 'Exchange Rate',
                    'data': result.values['exchange_rate'].tolist() if 'exchange_rate' in result.values.columns else [],
                    'color': '#2196F3',
                    'type': 'line',
                    'lineWidth': 2
                },
                {
                    'name': 'CIRP Theoretical Rate',
                    'data': result.values['cirp_theoretical_rate'].tolist() if 'cirp_theoretical_rate' in result.values.columns else [],
                    'color': '#4CAF50',
                    'type': 'line',
                    'lineWidth': 2,
                    'dashStyle': 'Dash'
                },
                {
                    'name': 'UIRP Expected Rate',
                    'data': result.values['uirp_expected_rate'].tolist() if 'uirp_expected_rate' in result.values.columns else [],
                    'color': '#FF9800',
                    'type': 'line',
                    'lineWidth': 2,
                    'dashStyle': 'Dot'
                },
                {
                    'name': 'CIRP Deviation %',
                    'data': result.values['cirp_deviation'].tolist() if 'cirp_deviation' in result.values.columns else [],
                    'color': '#9C27B0',
                    'type': 'line',
                    'lineWidth': 1,
                    'yAxis': 1
                },
                {
                    'name': 'Carry Return %',
                    'data': result.values['carry_return'].tolist() if 'carry_return' in result.values.columns else [],
                    'color': '#F44336',
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
    
    # Generate sample USD/JPY exchange rate data
    np.random.seed(42)
    returns = np.random.randn(252) * 0.008  # 0.8% daily volatility
    prices = 150 * (1 + returns).cumprod()  # Starting at 150 USD/JPY
    
    sample_data = pd.DataFrame({'close': prices}, index=dates)
    
    # Sample interest rate data
    sample_rates = {
        'base_rate': 0.0525,  # US 5.25%
        'quote_rate': -0.001  # Japan -0.1%
    }
    
    # Calculate IRP
    irp_calculator = IRPIndicator(base_country="US", quote_country="JP")
    result = irp_calculator.calculate(sample_data, sample_rates, asset_type=AssetType.FOREX)
    
    print(f"IRP Analysis (USD/JPY):")
    print(f"Current Rate: {result.current_rate:.2f}")
    print(f"CIRP Theoretical Rate: {result.covered_irp_rate:.2f}")
    print(f"UIRP Expected Rate: {result.uncovered_irp_rate:.2f}")
    print(f"CIRP Deviation: {result.cirp_deviation:.3f}%")
    print(f"UIRP Deviation: {result.uirp_deviation:.3f}%")
    print(f"Carry Trade Signal: {result.carry_trade_signal}")
    print(f"Carry Return: {result.carry_trade_return:.2f}%")
    print(f"Risk-Adjusted Carry: {result.risk_adjusted_carry:.2f}")
    print(f"Interest Differential: {result.interest_differential:.3f}%")
    print(f"Signals: {', '.join(result.signals)}")