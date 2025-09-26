"""Balance of Payments (BOP) Model Analysis

The Balance of Payments model analyzes currency movements based on a country's
international transactions, including trade balance, capital flows, and financial
account movements. This model helps identify currency pressures from fundamental
economic flows.

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
from scipy.signal import savgol_filter

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
    FLOW_BASED = "flow_based"


@dataclass
class BOPResult:
    """Result of Balance of Payments analysis"""
    name: str
    current_account_balance: float
    capital_account_balance: float
    financial_account_balance: float
    bop_pressure_index: float
    trade_balance_impact: float
    capital_flow_impact: float
    reserve_change_impact: float
    currency_pressure_signal: str
    flow_sustainability: str
    external_vulnerability: float
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class BOPIndicator:
    """Balance of Payments Model Calculator with Advanced Flow Analysis"""
    
    def __init__(self, base_country: str = "US", quote_country: str = "EU",
                 smoothing_window: int = 21, pressure_threshold: float = 1.5):
        """
        Initialize BOP calculator
        
        Args:
            base_country: Base currency country code (default: "US")
            quote_country: Quote currency country code (default: "EU")
            smoothing_window: Window for smoothing BOP data (default: 21 days)
            pressure_threshold: Threshold for significant BOP pressure (default: 1.5)
        """
        self.base_country = base_country
        self.quote_country = quote_country
        self.smoothing_window = smoothing_window
        self.pressure_threshold = pressure_threshold
        self.logger = logging.getLogger(__name__)
        
        # Initialize country BOP data
        self.bop_data = self._initialize_bop_data()
    
    def calculate(self, data: pd.DataFrame, bop_data: Optional[Dict] = None,
                 trade_data: Optional[Dict] = None, capital_flows: Optional[Dict] = None,
                 custom_base: Optional[str] = None, custom_quote: Optional[str] = None,
                 asset_type: AssetType = AssetType.FOREX) -> BOPResult:
        """
        Calculate Balance of Payments analysis
        
        Args:
            data: Exchange rate data DataFrame with 'close' column
            bop_data: Dictionary containing BOP components
            trade_data: Dictionary containing trade balance data
            capital_flows: Dictionary containing capital flow data
            custom_base: Override base country
            custom_quote: Override quote country
            asset_type: Type of asset being analyzed
            
        Returns:
            BOPResult containing BOP analysis
        """
        try:
            # Use custom countries if provided
            base_country = custom_base or self.base_country
            quote_country = custom_quote or self.quote_country
            
            # Prepare BOP data
            bop_components = self._prepare_bop_data(bop_data, trade_data, capital_flows, 
                                                   base_country, quote_country, len(data))
            
            # Calculate current account analysis
            current_account = self._analyze_current_account(data, bop_components)
            
            # Calculate capital account analysis
            capital_account = self._analyze_capital_account(data, bop_components)
            
            # Calculate financial account analysis
            financial_account = self._analyze_financial_account(data, bop_components)
            
            # Calculate BOP pressure index
            bop_pressure = self._calculate_bop_pressure(current_account, capital_account, 
                                                       financial_account, bop_components)
            
            # Analyze individual flow impacts
            trade_impact = self._analyze_trade_impact(data, bop_components)
            capital_impact = self._analyze_capital_flow_impact(data, bop_components)
            reserve_impact = self._analyze_reserve_impact(data, bop_components)
            
            # Generate currency pressure signals
            pressure_signal, sustainability = self._generate_pressure_signals(
                bop_pressure, current_account, capital_account, financial_account
            )
            
            # Calculate external vulnerability
            vulnerability = self._calculate_external_vulnerability(bop_components, data)
            
            # Generate comprehensive signals
            signals = self._generate_signals(bop_pressure, current_account, capital_account,
                                           financial_account, trade_impact, capital_impact,
                                           vulnerability, pressure_signal)
            
            # Create time series data
            values_df = self._create_time_series(data, bop_components, current_account,
                                               capital_account, financial_account, 
                                               bop_pressure, trade_impact, capital_impact)
            
            # Calculate confidence
            confidence = self._calculate_confidence(bop_components, len(data))
            
            return BOPResult(
                name="Balance of Payments Model",
                current_account_balance=current_account['balance'],
                capital_account_balance=capital_account['balance'],
                financial_account_balance=financial_account['balance'],
                bop_pressure_index=bop_pressure['index'],
                trade_balance_impact=trade_impact['impact'],
                capital_flow_impact=capital_impact['impact'],
                reserve_change_impact=reserve_impact['impact'],
                currency_pressure_signal=pressure_signal,
                flow_sustainability=sustainability,
                external_vulnerability=vulnerability,
                values=values_df,
                metadata={
                    'base_country': base_country,
                    'quote_country': quote_country,
                    'bop_components': bop_components,
                    'current_account_analysis': current_account,
                    'capital_account_analysis': capital_account,
                    'financial_account_analysis': financial_account,
                    'pressure_analysis': bop_pressure,
                    'flow_decomposition': {
                        'trade_impact': trade_impact,
                        'capital_impact': capital_impact,
                        'reserve_impact': reserve_impact
                    },
                    'vulnerability_assessment': self._assess_vulnerability_components(bop_components),
                    'flow_persistence': self._analyze_flow_persistence(values_df),
                    'seasonal_patterns': self._analyze_seasonal_patterns(bop_components),
                    'crisis_indicators': self._identify_crisis_indicators(bop_pressure, vulnerability),
                    'interpretation': self._get_interpretation(bop_pressure, pressure_signal, 
                                                            current_account, vulnerability)
                },
                confidence=confidence,
                timestamp=datetime.now(),
                asset_type=asset_type,
                category=IndicatorCategory.FLOW_BASED,
                signals=signals
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating BOP: {e}")
            return self._empty_result(asset_type)
    
    def _initialize_bop_data(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default BOP data for major economies"""
        return {
            'US': {
                'current_account': -0.035,  # -3.5% of GDP
                'trade_balance': -0.028,    # -2.8% of GDP
                'services_balance': 0.008,   # 0.8% of GDP
                'income_balance': 0.002,     # 0.2% of GDP
                'transfers_balance': -0.005, # -0.5% of GDP
                'capital_account': 0.001,    # 0.1% of GDP
                'fdi_inflows': 0.015,       # 1.5% of GDP
                'fdi_outflows': -0.020,     # -2.0% of GDP
                'portfolio_inflows': 0.025,  # 2.5% of GDP
                'portfolio_outflows': -0.018, # -1.8% of GDP
                'other_investment': 0.010,   # 1.0% of GDP
                'reserve_changes': 0.002,    # 0.2% of GDP
                'gdp_usd': 25000,           # GDP in billions USD
                'external_debt': 0.95       # External debt as % of GDP
            },
            'EU': {
                'current_account': 0.025,   # 2.5% of GDP
                'trade_balance': 0.035,     # 3.5% of GDP
                'services_balance': 0.005,   # 0.5% of GDP
                'income_balance': -0.008,    # -0.8% of GDP
                'transfers_balance': -0.007, # -0.7% of GDP
                'capital_account': 0.002,    # 0.2% of GDP
                'fdi_inflows': 0.012,       # 1.2% of GDP
                'fdi_outflows': -0.015,     # -1.5% of GDP
                'portfolio_inflows': 0.020,  # 2.0% of GDP
                'portfolio_outflows': -0.022, # -2.2% of GDP
                'other_investment': -0.005,  # -0.5% of GDP
                'reserve_changes': 0.001,    # 0.1% of GDP
                'gdp_usd': 17000,           # GDP in billions USD
                'external_debt': 1.15       # External debt as % of GDP
            },
            'JP': {
                'current_account': 0.032,   # 3.2% of GDP
                'trade_balance': 0.008,     # 0.8% of GDP
                'services_balance': -0.012,  # -1.2% of GDP
                'income_balance': 0.035,     # 3.5% of GDP
                'transfers_balance': -0.002, # -0.2% of GDP
                'capital_account': 0.001,    # 0.1% of GDP
                'fdi_inflows': 0.005,       # 0.5% of GDP
                'fdi_outflows': -0.018,     # -1.8% of GDP
                'portfolio_inflows': 0.015,  # 1.5% of GDP
                'portfolio_outflows': -0.025, # -2.5% of GDP
                'other_investment': 0.008,   # 0.8% of GDP
                'reserve_changes': 0.003,    # 0.3% of GDP
                'gdp_usd': 4200,            # GDP in billions USD
                'external_debt': 0.65       # External debt as % of GDP
            },
            'GB': {
                'current_account': -0.025,  # -2.5% of GDP
                'trade_balance': -0.018,    # -1.8% of GDP
                'services_balance': 0.045,   # 4.5% of GDP
                'income_balance': -0.035,    # -3.5% of GDP
                'transfers_balance': -0.017, # -1.7% of GDP
                'capital_account': 0.002,    # 0.2% of GDP
                'fdi_inflows': 0.025,       # 2.5% of GDP
                'fdi_outflows': -0.030,     # -3.0% of GDP
                'portfolio_inflows': 0.055,  # 5.5% of GDP
                'portfolio_outflows': -0.048, # -4.8% of GDP
                'other_investment': 0.015,   # 1.5% of GDP
                'reserve_changes': 0.001,    # 0.1% of GDP
                'gdp_usd': 3100,            # GDP in billions USD
                'external_debt': 3.25       # External debt as % of GDP (very high for UK)
            },
            'CA': {
                'current_account': -0.018,  # -1.8% of GDP
                'trade_balance': 0.015,     # 1.5% of GDP
                'services_balance': -0.025,  # -2.5% of GDP
                'income_balance': -0.012,    # -1.2% of GDP
                'transfers_balance': 0.004,  # 0.4% of GDP
                'capital_account': 0.001,    # 0.1% of GDP
                'fdi_inflows': 0.022,       # 2.2% of GDP
                'fdi_outflows': -0.028,     # -2.8% of GDP
                'portfolio_inflows': 0.035,  # 3.5% of GDP
                'portfolio_outflows': -0.032, # -3.2% of GDP
                'other_investment': 0.008,   # 0.8% of GDP
                'reserve_changes': 0.002,    # 0.2% of GDP
                'gdp_usd': 2100,            # GDP in billions USD
                'external_debt': 1.18       # External debt as % of GDP
            },
            'AU': {
                'current_account': -0.015,  # -1.5% of GDP
                'trade_balance': 0.025,     # 2.5% of GDP
                'services_balance': -0.018,  # -1.8% of GDP
                'income_balance': -0.028,    # -2.8% of GDP
                'transfers_balance': 0.006,  # 0.6% of GDP
                'capital_account': 0.002,    # 0.2% of GDP
                'fdi_inflows': 0.018,       # 1.8% of GDP
                'fdi_outflows': -0.012,     # -1.2% of GDP
                'portfolio_inflows': 0.028,  # 2.8% of GDP
                'portfolio_outflows': -0.025, # -2.5% of GDP
                'other_investment': 0.005,   # 0.5% of GDP
                'reserve_changes': 0.001,    # 0.1% of GDP
                'gdp_usd': 1550,            # GDP in billions USD
                'external_debt': 1.05       # External debt as % of GDP
            }
        }
    
    def _prepare_bop_data(self, bop_data: Optional[Dict], trade_data: Optional[Dict],
                         capital_flows: Optional[Dict], base_country: str, 
                         quote_country: str, data_length: int) -> Dict[str, Any]:
        """Prepare BOP data for analysis"""
        # Use provided data or defaults
        base_bop = bop_data.get('base', {}) if bop_data else {}
        quote_bop = bop_data.get('quote', {}) if bop_data else {}
        
        # Get default data
        base_defaults = self.bop_data.get(base_country, self.bop_data['US'])
        quote_defaults = self.bop_data.get(quote_country, self.bop_data['EU'])
        
        # Merge with defaults
        for key, default_value in base_defaults.items():
            if key not in base_bop:
                base_bop[key] = default_value
        
        for key, default_value in quote_defaults.items():
            if key not in quote_bop:
                quote_bop[key] = default_value
        
        # Add trade data if provided
        if trade_data:
            base_bop.update(trade_data.get('base', {}))
            quote_bop.update(trade_data.get('quote', {}))
        
        # Add capital flow data if provided
        if capital_flows:
            base_bop.update(capital_flows.get('base', {}))
            quote_bop.update(capital_flows.get('quote', {}))
        
        # Create time series (simplified - in practice would use actual time series data)
        dates = pd.date_range(end=datetime.now(), periods=data_length, freq='D')
        
        # Add some realistic variation to the data
        np.random.seed(42)
        variation_factor = 0.1  # 10% variation
        
        result = {
            'base_country_data': base_bop,
            'quote_country_data': quote_bop,
            'dates': dates,
            'base_current_account': self._create_time_series_component(
                base_bop['current_account'], data_length, variation_factor
            ),
            'quote_current_account': self._create_time_series_component(
                quote_bop['current_account'], data_length, variation_factor
            ),
            'base_capital_account': self._create_time_series_component(
                base_bop['capital_account'], data_length, variation_factor
            ),
            'quote_capital_account': self._create_time_series_component(
                quote_bop['capital_account'], data_length, variation_factor
            ),
            'base_financial_account': self._create_time_series_component(
                base_bop.get('fdi_inflows', 0) + base_bop.get('portfolio_inflows', 0) + 
                base_bop.get('fdi_outflows', 0) + base_bop.get('portfolio_outflows', 0) + 
                base_bop.get('other_investment', 0), data_length, variation_factor
            ),
            'quote_financial_account': self._create_time_series_component(
                quote_bop.get('fdi_inflows', 0) + quote_bop.get('portfolio_inflows', 0) + 
                quote_bop.get('fdi_outflows', 0) + quote_bop.get('portfolio_outflows', 0) + 
                quote_bop.get('other_investment', 0), data_length, variation_factor
            )
        }
        
        return result
    
    def _create_time_series_component(self, base_value: float, length: int, 
                                     variation: float) -> pd.Series:
        """Create time series with realistic variation around base value"""
        # Generate random walk around base value
        noise = np.random.randn(length) * variation * abs(base_value)
        
        # Add some trend and seasonality
        trend = np.linspace(0, 0.1 * base_value, length)
        seasonal = 0.05 * abs(base_value) * np.sin(2 * np.pi * np.arange(length) / 252)
        
        values = base_value + trend + seasonal + noise
        
        # Smooth the series
        if length > 10:
            window_size = min(self.smoothing_window, length // 3)
            if window_size % 2 == 0:
                window_size += 1
            values = savgol_filter(values, window_size, 3)
        
        return pd.Series(values)
    
    def _analyze_current_account(self, data: pd.DataFrame, bop_components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current account impact on currency"""
        base_ca = bop_components['base_current_account']
        quote_ca = bop_components['quote_current_account']
        
        # Current account differential
        ca_differential = base_ca - quote_ca
        
        # Current account balance (latest)
        current_balance = ca_differential.iloc[-1]
        
        # Trend analysis
        ca_trend = ca_differential.rolling(63).mean().pct_change(21).iloc[-1]  # 21-day trend
        
        # Sustainability analysis
        ca_volatility = ca_differential.rolling(63).std().iloc[-1]
        ca_persistence = ca_differential.autocorr(lag=21)  # 21-day autocorrelation
        
        # Impact on currency (positive CA differential = currency strength)
        if current_balance > 0.01:  # More than 1% of GDP
            ca_signal = "STRONG_POSITIVE"
        elif current_balance > 0.005:  # More than 0.5% of GDP
            ca_signal = "MODERATE_POSITIVE"
        elif current_balance > -0.005:  # Between -0.5% and 0.5%
            ca_signal = "NEUTRAL"
        elif current_balance > -0.01:  # Between -1% and -0.5%
            ca_signal = "MODERATE_NEGATIVE"
        else:
            ca_signal = "STRONG_NEGATIVE"
        
        return {
            'balance': current_balance,
            'trend': ca_trend,
            'volatility': ca_volatility,
            'persistence': ca_persistence,
            'signal': ca_signal,
            'time_series': ca_differential,
            'sustainability_score': self._calculate_ca_sustainability(base_ca, quote_ca)
        }
    
    def _analyze_capital_account(self, data: pd.DataFrame, bop_components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze capital account impact on currency"""
        base_ka = bop_components['base_capital_account']
        quote_ka = bop_components['quote_capital_account']
        
        # Capital account differential
        ka_differential = base_ka - quote_ka
        
        # Current balance
        current_balance = ka_differential.iloc[-1]
        
        # Trend and volatility
        ka_trend = ka_differential.rolling(63).mean().pct_change(21).iloc[-1]
        ka_volatility = ka_differential.rolling(63).std().iloc[-1]
        
        # Signal generation
        if abs(current_balance) < 0.001:  # Less than 0.1% of GDP
            ka_signal = "MINIMAL_IMPACT"
        elif current_balance > 0:
            ka_signal = "POSITIVE"
        else:
            ka_signal = "NEGATIVE"
        
        return {
            'balance': current_balance,
            'trend': ka_trend,
            'volatility': ka_volatility,
            'signal': ka_signal,
            'time_series': ka_differential
        }
    
    def _analyze_financial_account(self, data: pd.DataFrame, bop_components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial account impact on currency"""
        base_fa = bop_components['base_financial_account']
        quote_fa = bop_components['quote_financial_account']
        
        # Financial account differential
        fa_differential = base_fa - quote_fa
        
        # Current balance
        current_balance = fa_differential.iloc[-1]
        
        # Trend analysis
        fa_trend = fa_differential.rolling(63).mean().pct_change(21).iloc[-1]
        fa_volatility = fa_differential.rolling(63).std().iloc[-1]
        
        # Flow composition analysis (using base country data)
        base_data = bop_components['base_country_data']
        fdi_flows = base_data.get('fdi_inflows', 0) + base_data.get('fdi_outflows', 0)
        portfolio_flows = base_data.get('portfolio_inflows', 0) + base_data.get('portfolio_outflows', 0)
        other_flows = base_data.get('other_investment', 0)
        
        # Signal generation based on flow composition and magnitude
        if abs(current_balance) > 0.02:  # More than 2% of GDP
            if current_balance > 0:
                fa_signal = "STRONG_INFLOWS"
            else:
                fa_signal = "STRONG_OUTFLOWS"
        elif abs(current_balance) > 0.01:  # More than 1% of GDP
            if current_balance > 0:
                fa_signal = "MODERATE_INFLOWS"
            else:
                fa_signal = "MODERATE_OUTFLOWS"
        else:
            fa_signal = "BALANCED_FLOWS"
        
        return {
            'balance': current_balance,
            'trend': fa_trend,
            'volatility': fa_volatility,
            'signal': fa_signal,
            'time_series': fa_differential,
            'flow_composition': {
                'fdi_component': fdi_flows,
                'portfolio_component': portfolio_flows,
                'other_component': other_flows
            },
            'flow_stability': self._assess_flow_stability(fdi_flows, portfolio_flows, other_flows)
        }
    
    def _calculate_bop_pressure(self, current_account: Dict[str, Any], 
                               capital_account: Dict[str, Any],
                               financial_account: Dict[str, Any],
                               bop_components: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive BOP pressure index"""
        # Weights for different BOP components
        ca_weight = 0.4   # Current account weight
        ka_weight = 0.1   # Capital account weight
        fa_weight = 0.5   # Financial account weight
        
        # Normalize components to [-1, 1] scale
        ca_normalized = np.tanh(current_account['balance'] * 20)  # Scale factor 20
        ka_normalized = np.tanh(capital_account['balance'] * 100)  # Scale factor 100 (smaller amounts)
        fa_normalized = np.tanh(financial_account['balance'] * 10)  # Scale factor 10
        
        # Calculate weighted pressure index
        pressure_index = (ca_weight * ca_normalized + 
                         ka_weight * ka_normalized + 
                         fa_weight * fa_normalized)
        
        # Time series of pressure index
        ca_ts = current_account['time_series']
        ka_ts = capital_account['time_series']
        fa_ts = financial_account['time_series']
        
        pressure_ts = (ca_weight * np.tanh(ca_ts * 20) + 
                      ka_weight * np.tanh(ka_ts * 100) + 
                      fa_weight * np.tanh(fa_ts * 10))
        
        # Pressure volatility and persistence
        pressure_volatility = pressure_ts.rolling(63).std().iloc[-1]
        pressure_persistence = pressure_ts.autocorr(lag=21)
        
        # Pressure signal
        if pressure_index > self.pressure_threshold:
            pressure_signal = "STRONG_APPRECIATION_PRESSURE"
        elif pressure_index > 0.5:
            pressure_signal = "MODERATE_APPRECIATION_PRESSURE"
        elif pressure_index > -0.5:
            pressure_signal = "BALANCED_PRESSURE"
        elif pressure_index > -self.pressure_threshold:
            pressure_signal = "MODERATE_DEPRECIATION_PRESSURE"
        else:
            pressure_signal = "STRONG_DEPRECIATION_PRESSURE"
        
        return {
            'index': pressure_index,
            'signal': pressure_signal,
            'time_series': pressure_ts,
            'volatility': pressure_volatility,
            'persistence': pressure_persistence,
            'components': {
                'current_account_contribution': ca_weight * ca_normalized,
                'capital_account_contribution': ka_weight * ka_normalized,
                'financial_account_contribution': fa_weight * fa_normalized
            }
        }
    
    def _analyze_trade_impact(self, data: pd.DataFrame, bop_components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trade balance impact on currency"""
        base_data = bop_components['base_country_data']
        quote_data = bop_components['quote_country_data']
        
        # Trade balance differential
        base_trade = base_data.get('trade_balance', 0)
        quote_trade = quote_data.get('trade_balance', 0)
        trade_differential = base_trade - quote_trade
        
        # Trade impact assessment
        if abs(trade_differential) > 0.03:  # More than 3% of GDP difference
            impact_magnitude = "HIGH"
        elif abs(trade_differential) > 0.015:  # More than 1.5% of GDP difference
            impact_magnitude = "MODERATE"
        else:
            impact_magnitude = "LOW"
        
        # Direction
        if trade_differential > 0:
            impact_direction = "POSITIVE"
        else:
            impact_direction = "NEGATIVE"
        
        return {
            'impact': trade_differential,
            'magnitude': impact_magnitude,
            'direction': impact_direction,
            'base_trade_balance': base_trade,
            'quote_trade_balance': quote_trade
        }
    
    def _analyze_capital_flow_impact(self, data: pd.DataFrame, bop_components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze capital flow impact on currency"""
        base_data = bop_components['base_country_data']
        quote_data = bop_components['quote_country_data']
        
        # Net capital flows (inflows - outflows)
        base_net_fdi = base_data.get('fdi_inflows', 0) + base_data.get('fdi_outflows', 0)
        base_net_portfolio = base_data.get('portfolio_inflows', 0) + base_data.get('portfolio_outflows', 0)
        base_net_capital = base_net_fdi + base_net_portfolio + base_data.get('other_investment', 0)
        
        quote_net_fdi = quote_data.get('fdi_inflows', 0) + quote_data.get('fdi_outflows', 0)
        quote_net_portfolio = quote_data.get('portfolio_inflows', 0) + quote_data.get('portfolio_outflows', 0)
        quote_net_capital = quote_net_fdi + quote_net_portfolio + quote_data.get('other_investment', 0)
        
        capital_differential = base_net_capital - quote_net_capital
        
        # Impact assessment
        if abs(capital_differential) > 0.025:  # More than 2.5% of GDP
            impact_magnitude = "HIGH"
        elif abs(capital_differential) > 0.01:  # More than 1% of GDP
            impact_magnitude = "MODERATE"
        else:
            impact_magnitude = "LOW"
        
        return {
            'impact': capital_differential,
            'magnitude': impact_magnitude,
            'base_net_flows': base_net_capital,
            'quote_net_flows': quote_net_capital,
            'flow_composition': {
                'base_fdi': base_net_fdi,
                'base_portfolio': base_net_portfolio,
                'quote_fdi': quote_net_fdi,
                'quote_portfolio': quote_net_portfolio
            }
        }
    
    def _analyze_reserve_impact(self, data: pd.DataFrame, bop_components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reserve changes impact on currency"""
        base_data = bop_components['base_country_data']
        quote_data = bop_components['quote_country_data']
        
        base_reserves = base_data.get('reserve_changes', 0)
        quote_reserves = quote_data.get('reserve_changes', 0)
        
        reserve_differential = base_reserves - quote_reserves
        
        # Reserve impact (positive reserve changes can indicate intervention)
        if abs(reserve_differential) > 0.005:  # More than 0.5% of GDP
            impact_magnitude = "SIGNIFICANT"
        else:
            impact_magnitude = "MINIMAL"
        
        return {
            'impact': reserve_differential,
            'magnitude': impact_magnitude,
            'base_reserve_change': base_reserves,
            'quote_reserve_change': quote_reserves
        }
    
    def _generate_pressure_signals(self, bop_pressure: Dict[str, Any], 
                                  current_account: Dict[str, Any],
                                  capital_account: Dict[str, Any],
                                  financial_account: Dict[str, Any]) -> Tuple[str, str]:
        """Generate currency pressure and sustainability signals"""
        pressure_signal = bop_pressure['signal']
        
        # Sustainability analysis
        ca_balance = current_account['balance']
        fa_balance = financial_account['balance']
        
        # Check if financial flows can sustain current account deficit
        if ca_balance < -0.02:  # Large current account deficit
            if fa_balance > abs(ca_balance):  # Financial inflows cover deficit
                sustainability = "SUSTAINABLE_WITH_INFLOWS"
            else:
                sustainability = "POTENTIALLY_UNSUSTAINABLE"
        elif ca_balance > 0.02:  # Large current account surplus
            sustainability = "HIGHLY_SUSTAINABLE"
        else:
            sustainability = "BALANCED_SUSTAINABLE"
        
        return pressure_signal, sustainability
    
    def _calculate_external_vulnerability(self, bop_components: Dict[str, Any], 
                                        data: pd.DataFrame) -> float:
        """Calculate external vulnerability index"""
        base_data = bop_components['base_country_data']
        
        # External vulnerability factors
        external_debt_ratio = base_data.get('external_debt', 1.0)
        current_account_deficit = max(0, -base_data.get('current_account', 0))
        
        # Portfolio flow dependency (hot money)
        portfolio_inflows = base_data.get('portfolio_inflows', 0)
        total_inflows = (base_data.get('fdi_inflows', 0) + 
                        portfolio_inflows + 
                        base_data.get('other_investment', 0))
        
        portfolio_dependency = portfolio_inflows / max(total_inflows, 0.001)
        
        # Exchange rate volatility (from price data)
        returns = data['close'].pct_change().dropna()
        fx_volatility = returns.rolling(63).std().iloc[-1] * np.sqrt(252) if len(returns) > 63 else 0.15
        
        # Vulnerability index (0-1 scale, higher = more vulnerable)
        vulnerability = (
            0.3 * min(external_debt_ratio, 2.0) / 2.0 +  # External debt component
            0.3 * min(current_account_deficit * 20, 1.0) +  # CA deficit component
            0.2 * portfolio_dependency +  # Hot money dependency
            0.2 * min(fx_volatility * 5, 1.0)  # FX volatility component
        )
        
        return min(vulnerability, 1.0)
    
    def _calculate_ca_sustainability(self, base_ca: pd.Series, quote_ca: pd.Series) -> float:
        """Calculate current account sustainability score"""
        ca_diff = base_ca - quote_ca
        
        # Factors affecting sustainability
        ca_volatility = ca_diff.rolling(63).std().iloc[-1]
        ca_trend = ca_diff.rolling(126).mean().pct_change(63).iloc[-1]  # 3-month trend
        ca_level = abs(ca_diff.iloc[-1])
        
        # Sustainability score (0-1, higher = more sustainable)
        volatility_score = max(0, 1 - ca_volatility * 50)  # Lower volatility = higher score
        trend_score = max(0, 1 - abs(ca_trend) * 10)  # Stable trend = higher score
        level_score = max(0, 1 - ca_level * 20)  # Moderate level = higher score
        
        sustainability = (volatility_score + trend_score + level_score) / 3
        
        return sustainability
    
    def _assess_flow_stability(self, fdi_flows: float, portfolio_flows: float, 
                              other_flows: float) -> str:
        """Assess stability of capital flows"""
        total_flows = abs(fdi_flows) + abs(portfolio_flows) + abs(other_flows)
        
        if total_flows == 0:
            return "NO_FLOWS"
        
        # FDI is considered most stable
        fdi_share = abs(fdi_flows) / total_flows
        portfolio_share = abs(portfolio_flows) / total_flows
        
        if fdi_share > 0.6:
            return "HIGHLY_STABLE"
        elif fdi_share > 0.4:
            return "MODERATELY_STABLE"
        elif portfolio_share > 0.6:
            return "VOLATILE_FLOWS"
        else:
            return "MIXED_STABILITY"
    
    def _assess_vulnerability_components(self, bop_components: Dict[str, Any]) -> Dict[str, Any]:
        """Assess individual vulnerability components"""
        base_data = bop_components['base_country_data']
        
        return {
            'external_debt_risk': 'HIGH' if base_data.get('external_debt', 1.0) > 1.5 else 'MODERATE' if base_data.get('external_debt', 1.0) > 1.0 else 'LOW',
            'current_account_risk': 'HIGH' if base_data.get('current_account', 0) < -0.05 else 'MODERATE' if base_data.get('current_account', 0) < -0.02 else 'LOW',
            'capital_flow_risk': self._assess_capital_flow_risk(base_data),
            'reserve_adequacy': 'ADEQUATE' if base_data.get('reserve_changes', 0) >= 0 else 'CONCERNING'
        }
    
    def _assess_capital_flow_risk(self, country_data: Dict[str, Any]) -> str:
        """Assess capital flow risk"""
        portfolio_inflows = country_data.get('portfolio_inflows', 0)
        fdi_inflows = country_data.get('fdi_inflows', 0)
        
        total_inflows = portfolio_inflows + fdi_inflows
        
        if total_inflows <= 0:
            return 'LOW'
        
        hot_money_ratio = portfolio_inflows / total_inflows
        
        if hot_money_ratio > 0.7:
            return 'HIGH'
        elif hot_money_ratio > 0.4:
            return 'MODERATE'
        else:
            return 'LOW'
    
    def _analyze_flow_persistence(self, values_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze persistence of BOP flows"""
        try:
            if 'bop_pressure' not in values_df.columns or len(values_df) < 63:
                return {'persistence': 'INSUFFICIENT_DATA'}
            
            pressure_series = values_df['bop_pressure'].dropna()
            
            # Calculate autocorrelations at different lags
            lag_1 = pressure_series.autocorr(lag=1)
            lag_21 = pressure_series.autocorr(lag=21)  # Monthly
            lag_63 = pressure_series.autocorr(lag=63)  # Quarterly
            
            # Persistence classification
            if lag_21 > 0.7:
                persistence = 'HIGH_PERSISTENCE'
            elif lag_21 > 0.4:
                persistence = 'MODERATE_PERSISTENCE'
            else:
                persistence = 'LOW_PERSISTENCE'
            
            return {
                'persistence_classification': persistence,
                'lag_1_autocorr': lag_1,
                'lag_21_autocorr': lag_21,
                'lag_63_autocorr': lag_63
            }
            
        except Exception as e:
            self.logger.warning(f"Flow persistence analysis failed: {e}")
            return {'persistence': 'ERROR'}
    
    def _analyze_seasonal_patterns(self, bop_components: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze seasonal patterns in BOP flows"""
        try:
            # Simple seasonal analysis based on month
            current_month = datetime.now().month
            
            # Typical seasonal patterns (simplified)
            seasonal_factors = {
                1: {'trade': -0.1, 'capital': 0.05},   # January: weak trade, some capital inflows
                2: {'trade': -0.05, 'capital': 0.02},  # February: recovering
                3: {'trade': 0.0, 'capital': 0.0},     # March: neutral
                4: {'trade': 0.05, 'capital': -0.02},  # April: improving trade
                5: {'trade': 0.1, 'capital': -0.05},   # May: strong trade
                6: {'trade': 0.08, 'capital': -0.03},  # June: good trade
                7: {'trade': 0.05, 'capital': 0.0},    # July: moderate
                8: {'trade': 0.02, 'capital': 0.02},   # August: slowing
                9: {'trade': 0.0, 'capital': 0.05},    # September: neutral trade, capital inflows
                10: {'trade': -0.02, 'capital': 0.08}, # October: weak trade, strong capital
                11: {'trade': -0.05, 'capital': 0.1},  # November: weaker trade, very strong capital
                12: {'trade': -0.08, 'capital': 0.05}  # December: weak trade, moderate capital
            }
            
            current_seasonal = seasonal_factors.get(current_month, {'trade': 0, 'capital': 0})
            
            return {
                'current_month': current_month,
                'seasonal_trade_factor': current_seasonal['trade'],
                'seasonal_capital_factor': current_seasonal['capital'],
                'seasonal_interpretation': self._interpret_seasonal_factors(current_seasonal)
            }
            
        except Exception as e:
            self.logger.warning(f"Seasonal analysis failed: {e}")
            return {'seasonal_patterns': 'ERROR'}
    
    def _interpret_seasonal_factors(self, seasonal: Dict[str, float]) -> str:
        """Interpret seasonal factors"""
        trade_factor = seasonal['trade']
        capital_factor = seasonal['capital']
        
        if trade_factor > 0.05 and capital_factor < -0.02:
            return "STRONG_TRADE_SEASON"
        elif trade_factor < -0.05 and capital_factor > 0.05:
            return "CAPITAL_INFLOW_SEASON"
        elif abs(trade_factor) < 0.02 and abs(capital_factor) < 0.02:
            return "NEUTRAL_SEASON"
        else:
            return "MIXED_SEASONAL_EFFECTS"
    
    def _identify_crisis_indicators(self, bop_pressure: Dict[str, Any], 
                                   vulnerability: float) -> Dict[str, Any]:
        """Identify potential crisis indicators"""
        crisis_score = 0
        indicators = []
        
        # High BOP pressure
        if abs(bop_pressure['index']) > 2.0:
            crisis_score += 2
            indicators.append('EXTREME_BOP_PRESSURE')
        elif abs(bop_pressure['index']) > 1.5:
            crisis_score += 1
            indicators.append('HIGH_BOP_PRESSURE')
        
        # High external vulnerability
        if vulnerability > 0.8:
            crisis_score += 2
            indicators.append('HIGH_EXTERNAL_VULNERABILITY')
        elif vulnerability > 0.6:
            crisis_score += 1
            indicators.append('MODERATE_EXTERNAL_VULNERABILITY')
        
        # High pressure volatility
        if bop_pressure.get('volatility', 0) > 0.5:
            crisis_score += 1
            indicators.append('VOLATILE_BOP_FLOWS')
        
        # Crisis risk assessment
        if crisis_score >= 4:
            crisis_risk = 'HIGH'
        elif crisis_score >= 2:
            crisis_risk = 'MODERATE'
        else:
            crisis_risk = 'LOW'
        
        return {
            'crisis_risk_level': crisis_risk,
            'crisis_score': crisis_score,
            'crisis_indicators': indicators
        }
    
    def _generate_signals(self, bop_pressure: Dict[str, Any], current_account: Dict[str, Any],
                         capital_account: Dict[str, Any], financial_account: Dict[str, Any],
                         trade_impact: Dict[str, Any], capital_impact: Dict[str, Any],
                         vulnerability: float, pressure_signal: str) -> List[str]:
        """Generate comprehensive BOP signals"""
        signals = []
        
        # Primary BOP pressure signals
        signals.append(f"BOP_{pressure_signal}")
        
        # Current account signals
        ca_signal = current_account['signal']
        if ca_signal != "NEUTRAL":
            signals.append(f"CURRENT_ACCOUNT_{ca_signal}")
        
        # Financial account signals
        fa_signal = financial_account['signal']
        signals.append(f"FINANCIAL_{fa_signal}")
        
        # Trade balance signals
        if trade_impact['magnitude'] != "LOW":
            signals.append(f"TRADE_{trade_impact['magnitude']}_{trade_impact['direction']}")
        
        # Capital flow signals
        if capital_impact['magnitude'] != "LOW":
            signals.append(f"CAPITAL_FLOWS_{capital_impact['magnitude']}")
        
        # Vulnerability signals
        if vulnerability > 0.7:
            signals.append("HIGH_EXTERNAL_VULNERABILITY")
        elif vulnerability > 0.4:
            signals.append("MODERATE_EXTERNAL_VULNERABILITY")
        
        # Flow stability signals
        flow_stability = financial_account.get('flow_stability', 'UNKNOWN')
        if flow_stability in ['VOLATILE_FLOWS', 'MIXED_STABILITY']:
            signals.append(f"CAPITAL_{flow_stability}")
        
        # Sustainability signals
        ca_sustainability = current_account.get('sustainability_score', 0.5)
        if ca_sustainability < 0.3:
            signals.append("UNSUSTAINABLE_CA_POSITION")
        elif ca_sustainability > 0.7:
            signals.append("SUSTAINABLE_CA_POSITION")
        
        # Combined signals
        pressure_index = bop_pressure['index']
        if pressure_index > 1.0 and vulnerability < 0.3:
            signals.append("STRONG_FUNDAMENTALS")
        elif pressure_index < -1.0 and vulnerability > 0.7:
            signals.append("WEAK_FUNDAMENTALS")
        
        return signals
    
    def _create_time_series(self, data: pd.DataFrame, bop_components: Dict[str, Any],
                           current_account: Dict[str, Any], capital_account: Dict[str, Any],
                           financial_account: Dict[str, Any], bop_pressure: Dict[str, Any],
                           trade_impact: Dict[str, Any], capital_impact: Dict[str, Any]) -> pd.DataFrame:
        """Create comprehensive time series DataFrame"""
        result_df = pd.DataFrame({
            'exchange_rate': data['close'],
            'current_account_balance': current_account['time_series'],
            'capital_account_balance': capital_account['time_series'],
            'financial_account_balance': financial_account['time_series'],
            'bop_pressure': bop_pressure['time_series']
        }, index=data.index)
        
        # Add component contributions
        result_df['ca_contribution'] = bop_pressure['components']['current_account_contribution']
        result_df['ka_contribution'] = bop_pressure['components']['capital_account_contribution']
        result_df['fa_contribution'] = bop_pressure['components']['financial_account_contribution']
        
        # Add rolling statistics
        result_df['bop_pressure_ma'] = result_df['bop_pressure'].rolling(21).mean()
        result_df['bop_volatility'] = result_df['bop_pressure'].rolling(63).std()
        
        # Add trade and capital flow impacts (constant series)
        result_df['trade_impact'] = trade_impact['impact']
        result_df['capital_flow_impact'] = capital_impact['impact']
        
        # Add exchange rate momentum for comparison
        result_df['fx_momentum'] = data['close'].pct_change(21)
        
        # Add pressure zones
        result_df['pressure_zone'] = self._classify_pressure_zone(result_df['bop_pressure'])
        
        return result_df
    
    def _classify_pressure_zone(self, pressure_series: pd.Series) -> pd.Series:
        """Classify BOP pressure zones"""
        def classify_pressure(pressure):
            if pd.isna(pressure):
                return "UNKNOWN"
            elif pressure > 1.5:
                return "STRONG_APPRECIATION"
            elif pressure > 0.5:
                return "MODERATE_APPRECIATION"
            elif pressure > -0.5:
                return "BALANCED"
            elif pressure > -1.5:
                return "MODERATE_DEPRECIATION"
            else:
                return "STRONG_DEPRECIATION"
        
        return pressure_series.apply(classify_pressure)
    
    def _calculate_confidence(self, bop_components: Dict[str, Any], data_length: int) -> float:
        """Calculate confidence score"""
        confidence = 0.4  # Base confidence
        
        # Data length adjustment
        if data_length >= 1260:  # 5 years
            confidence += 0.2
        elif data_length >= 252:  # 1 year
            confidence += 0.15
        
        # BOP data completeness
        base_data = bop_components['base_country_data']
        required_fields = ['current_account', 'trade_balance', 'fdi_inflows', 'portfolio_inflows']
        completeness = sum(1 for field in required_fields if field in base_data) / len(required_fields)
        confidence += 0.2 * completeness
        
        # Country data quality (major economies have better data)
        if bop_components.get('base_country_data', {}).get('gdp_usd', 0) > 1000:  # Large economy
            confidence += 0.1
        
        return min(0.9, confidence)
    
    def _get_interpretation(self, bop_pressure: Dict[str, Any], pressure_signal: str,
                          current_account: Dict[str, Any], vulnerability: float) -> str:
        """Get interpretation of BOP results"""
        pressure_index = bop_pressure['index']
        ca_balance = current_account['balance']
        
        # Main pressure assessment
        if abs(pressure_index) > 1.5:
            pressure_desc = "significant currency pressure"
        elif abs(pressure_index) > 0.5:
            pressure_desc = "moderate currency pressure"
        else:
            pressure_desc = "balanced currency flows"
        
        # Direction
        if pressure_index > 0:
            direction = "appreciation"
        else:
            direction = "depreciation"
        
        # Current account context
        if abs(ca_balance) > 0.02:
            ca_context = f"with {'surplus' if ca_balance > 0 else 'deficit'} of {abs(ca_balance)*100:.1f}% of GDP"
        else:
            ca_context = "with balanced current account"
        
        # Vulnerability context
        if vulnerability > 0.7:
            vuln_context = "High external vulnerability suggests caution."
        elif vulnerability > 0.4:
            vuln_context = "Moderate external vulnerability requires monitoring."
        else:
            vuln_context = "Low external vulnerability supports stability."
        
        return f"BOP analysis indicates {pressure_desc} toward {direction} {ca_context}. {vuln_context}"
    
    def _empty_result(self, asset_type: AssetType) -> BOPResult:
        """Return empty result for error cases"""
        return BOPResult(
            name="Balance of Payments Model",
            current_account_balance=0.0,
            capital_account_balance=0.0,
            financial_account_balance=0.0,
            bop_pressure_index=0.0,
            trade_balance_impact=0.0,
            capital_flow_impact=0.0,
            reserve_change_impact=0.0,
            currency_pressure_signal="ERROR",
            flow_sustainability="ERROR",
            external_vulnerability=0.0,
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.FLOW_BASED,
            signals=["ERROR"]
        )
    
    def get_chart_data(self, result: BOPResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'bop_analysis',
            'name': 'Balance of Payments Model',
            'data': {
                'exchange_rate': result.values['exchange_rate'].tolist() if 'exchange_rate' in result.values.columns else [],
                'bop_pressure': result.values['bop_pressure'].tolist() if 'bop_pressure' in result.values.columns else [],
                'current_account': result.values['current_account_balance'].tolist() if 'current_account_balance' in result.values.columns else [],
                'financial_account': result.values['financial_account_balance'].tolist() if 'financial_account_balance' in result.values.columns else [],
                'pressure_ma': result.values['bop_pressure_ma'].tolist() if 'bop_pressure_ma' in result.values.columns else []
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'bop_metrics': {
                'current_account_balance': result.current_account_balance,
                'capital_account_balance': result.capital_account_balance,
                'financial_account_balance': result.financial_account_balance,
                'bop_pressure_index': result.bop_pressure_index,
                'trade_balance_impact': result.trade_balance_impact,
                'capital_flow_impact': result.capital_flow_impact,
                'currency_pressure_signal': result.currency_pressure_signal,
                'external_vulnerability': result.external_vulnerability
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
                    'name': 'BOP Pressure Index',
                    'data': result.values['bop_pressure'].tolist() if 'bop_pressure' in result.values.columns else [],
                    'color': '#4CAF50',
                    'type': 'line',
                    'lineWidth': 2,
                    'yAxis': 1
                },
                {
                    'name': 'BOP Pressure MA',
                    'data': result.values['bop_pressure_ma'].tolist() if 'bop_pressure_ma' in result.values.columns else [],
                    'color': '#FF9800',
                    'type': 'line',
                    'lineWidth': 1,
                    'yAxis': 1
                },
                {
                    'name': 'Current Account Balance',
                    'data': result.values['current_account_balance'].tolist() if 'current_account_balance' in result.values.columns else [],
                    'color': '#9C27B0',
                    'type': 'line',
                    'lineWidth': 1,
                    'yAxis': 2
                },
                {
                    'name': 'Financial Account Balance',
                    'data': result.values['financial_account_balance'].tolist() if 'financial_account_balance' in result.values.columns else [],
                    'color': '#F44336',
                    'type': 'line',
                    'lineWidth': 1,
                    'yAxis': 2
                }
            ]
        }


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Generate sample EUR/USD exchange rate data
    np.random.seed(42)
    returns = np.random.randn(252) * 0.008
    prices = 1.10 * (1 + returns).cumprod()
    
    sample_data = pd.DataFrame({'close': prices}, index=dates)
    
    # Sample BOP data
    sample_bop = {
        'base': {
            'current_account': -0.035,
            'trade_balance': -0.028,
            'fdi_inflows': 0.015,
            'portfolio_inflows': 0.025
        },
        'quote': {
            'current_account': 0.025,
            'trade_balance': 0.035,
            'fdi_inflows': 0.012,
            'portfolio_inflows': 0.020
        }
    }
    
    # Calculate BOP
    bop_calculator = BOPIndicator(base_country="US", quote_country="EU")
    
    result = bop_calculator.calculate(
        data=sample_data,
        bop_data=sample_bop,
        asset_type=AssetType.FOREX
    )
    
    print(f"BOP Pressure Index: {result.bop_pressure_index:.4f}")
    print(f"Current Account Balance: {result.current_account_balance:.4f}")
    print(f"Financial Account Balance: {result.financial_account_balance:.4f}")
    print(f"Currency Pressure Signal: {result.currency_pressure_signal}")
    print(f"External Vulnerability: {result.external_vulnerability:.4f}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Signals: {result.signals}")
    
    # Get chart data
    chart_data = bop_calculator.get_chart_data(result)
    print(f"\nChart data prepared with {len(chart_data['series'])} series")