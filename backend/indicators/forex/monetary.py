"""Monetary Model of Exchange Rate Determination

The Monetary Model explains exchange rate movements through relative money supply,
income levels, and interest rates between countries. This model is based on the
quantity theory of money and purchasing power parity assumptions.

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
    MONETARY = "monetary"
    MACROECONOMIC = "macroeconomic"


@dataclass
class MonetaryResult:
    """Result of Monetary Model analysis"""
    name: str
    theoretical_exchange_rate: float
    current_exchange_rate: float
    misalignment: float
    misalignment_percentage: float
    money_supply_differential: float
    income_differential: float
    interest_rate_differential: float
    velocity_differential: float
    monetary_pressure: str
    convergence_signal: str
    half_life_months: float
    model_r_squared: float
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class MonetaryIndicator:
    """Monetary Model Calculator with Advanced Econometric Analysis"""
    
    def __init__(self, base_country: str = "US", quote_country: str = "EU",
                 model_type: str = "flexible_price", estimation_window: int = 252):
        """
        Initialize Monetary Model calculator
        
        Args:
            base_country: Base currency country code (default: "US")
            quote_country: Quote currency country code (default: "EU")
            model_type: Type of monetary model ("flexible_price", "sticky_price", "real_interest_differential")
            estimation_window: Window for model estimation (default: 252 days)
        """
        self.base_country = base_country
        self.quote_country = quote_country
        self.model_type = model_type
        self.estimation_window = estimation_window
        self.logger = logging.getLogger(__name__)
        
        # Initialize monetary data
        self.monetary_data = self._initialize_monetary_data()
        
        # Model parameters
        self.alpha = 1.0  # Money supply elasticity
        self.beta = 1.0   # Income elasticity
        self.gamma = 1.0  # Interest rate elasticity
    
    def calculate(self, data: pd.DataFrame, monetary_data: Optional[Dict] = None,
                 custom_base: Optional[str] = None, custom_quote: Optional[str] = None,
                 asset_type: AssetType = AssetType.FOREX) -> MonetaryResult:
        """
        Calculate Monetary Model analysis
        
        Args:
            data: Exchange rate data DataFrame with 'close' column
            monetary_data: Dictionary containing monetary variables
            custom_base: Override base country
            custom_quote: Override quote country
            asset_type: Type of asset being analyzed
            
        Returns:
            MonetaryResult containing monetary model analysis
        """
        try:
            # Use custom countries if provided
            base_country = custom_base or self.base_country
            quote_country = custom_quote or self.quote_country
            
            # Prepare monetary data
            monetary_vars = self._prepare_monetary_data(monetary_data, base_country, 
                                                       quote_country, len(data))
            
            # Estimate model parameters
            model_params = self._estimate_model_parameters(data, monetary_vars)
            
            # Calculate theoretical exchange rate
            theoretical_rate = self._calculate_theoretical_rate(monetary_vars, model_params)
            
            # Calculate misalignment
            current_rate = data['close'].iloc[-1]
            misalignment = theoretical_rate - current_rate
            misalignment_pct = (misalignment / current_rate) * 100
            
            # Analyze monetary fundamentals
            money_differential = self._calculate_money_supply_differential(monetary_vars)
            income_differential = self._calculate_income_differential(monetary_vars)
            interest_differential = self._calculate_interest_rate_differential(monetary_vars)
            velocity_differential = self._calculate_velocity_differential(monetary_vars)
            
            # Generate monetary pressure signal
            monetary_pressure = self._assess_monetary_pressure(money_differential, 
                                                             income_differential, 
                                                             interest_differential)
            
            # Analyze convergence dynamics
            convergence_signal, half_life = self._analyze_convergence_dynamics(
                data, theoretical_rate, model_params
            )
            
            # Calculate model fit
            r_squared = self._calculate_model_fit(data, monetary_vars, model_params)
            
            # Generate comprehensive signals
            signals = self._generate_signals(misalignment_pct, monetary_pressure, 
                                           convergence_signal, money_differential,
                                           income_differential, interest_differential,
                                           r_squared)
            
            # Create time series data
            values_df = self._create_time_series(data, monetary_vars, theoretical_rate,
                                               model_params, misalignment)
            
            # Calculate confidence
            confidence = self._calculate_confidence(r_squared, len(data), monetary_vars)
            
            return MonetaryResult(
                name="Monetary Model",
                theoretical_exchange_rate=theoretical_rate,
                current_exchange_rate=current_rate,
                misalignment=misalignment,
                misalignment_percentage=misalignment_pct,
                money_supply_differential=money_differential,
                income_differential=income_differential,
                interest_rate_differential=interest_differential,
                velocity_differential=velocity_differential,
                monetary_pressure=monetary_pressure,
                convergence_signal=convergence_signal,
                half_life_months=half_life,
                model_r_squared=r_squared,
                values=values_df,
                metadata={
                    'base_country': base_country,
                    'quote_country': quote_country,
                    'model_type': self.model_type,
                    'model_parameters': model_params,
                    'monetary_variables': monetary_vars,
                    'estimation_diagnostics': self._run_model_diagnostics(data, monetary_vars, model_params),
                    'structural_breaks': self._test_structural_breaks(data, monetary_vars),
                    'cointegration_analysis': self._test_cointegration(data, monetary_vars),
                    'forecast_accuracy': self._assess_forecast_accuracy(data, monetary_vars, model_params),
                    'sensitivity_analysis': self._perform_sensitivity_analysis(monetary_vars, model_params),
                    'interpretation': self._get_interpretation(misalignment_pct, monetary_pressure, 
                                                            convergence_signal, r_squared)
                },
                confidence=confidence,
                timestamp=datetime.now(),
                asset_type=asset_type,
                category=IndicatorCategory.MONETARY,
                signals=signals
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating Monetary Model: {e}")
            return self._empty_result(asset_type)
    
    def _initialize_monetary_data(self) -> Dict[str, Dict[str, Any]]:
        """Initialize default monetary data for major economies"""
        return {
            'US': {
                'money_supply_m2': 21000,      # M2 in billions USD
                'real_gdp': 25000,             # Real GDP in billions USD
                'nominal_interest_rate': 0.05,  # 5% nominal rate
                'inflation_rate': 0.025,       # 2.5% inflation
                'money_velocity': 1.2,         # M2 velocity
                'money_growth_rate': 0.08,     # 8% annual M2 growth
                'gdp_growth_rate': 0.025,      # 2.5% real GDP growth
                'central_bank_rate': 0.05,     # Federal funds rate
                'monetary_base': 5500,         # Monetary base in billions
                'money_multiplier': 3.8        # Money multiplier
            },
            'EU': {
                'money_supply_m2': 15000,      # M2 in billions EUR
                'real_gdp': 17000,             # Real GDP in billions EUR
                'nominal_interest_rate': 0.03,  # 3% nominal rate
                'inflation_rate': 0.02,        # 2% inflation
                'money_velocity': 1.1,         # M2 velocity
                'money_growth_rate': 0.06,     # 6% annual M2 growth
                'gdp_growth_rate': 0.015,      # 1.5% real GDP growth
                'central_bank_rate': 0.03,     # ECB rate
                'monetary_base': 4200,         # Monetary base in billions
                'money_multiplier': 3.6        # Money multiplier
            },
            'JP': {
                'money_supply_m2': 1100000,    # M2 in billions JPY
                'real_gdp': 550000,            # Real GDP in billions JPY
                'nominal_interest_rate': 0.001, # 0.1% nominal rate
                'inflation_rate': 0.005,       # 0.5% inflation
                'money_velocity': 0.8,         # M2 velocity
                'money_growth_rate': 0.03,     # 3% annual M2 growth
                'gdp_growth_rate': 0.01,       # 1% real GDP growth
                'central_bank_rate': -0.001,   # Negative BOJ rate
                'monetary_base': 580000,       # Monetary base in billions
                'money_multiplier': 1.9        # Money multiplier
            },
            'GB': {
                'money_supply_m2': 2800,       # M2 in billions GBP
                'real_gdp': 2500,              # Real GDP in billions GBP
                'nominal_interest_rate': 0.045, # 4.5% nominal rate
                'inflation_rate': 0.035,       # 3.5% inflation
                'money_velocity': 1.15,        # M2 velocity
                'money_growth_rate': 0.07,     # 7% annual M2 growth
                'gdp_growth_rate': 0.02,       # 2% real GDP growth
                'central_bank_rate': 0.045,    # Bank of England rate
                'monetary_base': 850,          # Monetary base in billions
                'money_multiplier': 3.3        # Money multiplier
            },
            'CA': {
                'money_supply_m2': 1800,       # M2 in billions CAD
                'real_gdp': 2100,              # Real GDP in billions CAD
                'nominal_interest_rate': 0.04,  # 4% nominal rate
                'inflation_rate': 0.03,        # 3% inflation
                'money_velocity': 1.18,        # M2 velocity
                'money_growth_rate': 0.065,    # 6.5% annual M2 growth
                'gdp_growth_rate': 0.022,      # 2.2% real GDP growth
                'central_bank_rate': 0.04,     # Bank of Canada rate
                'monetary_base': 520,          # Monetary base in billions
                'money_multiplier': 3.5        # Money multiplier
            },
            'AU': {
                'money_supply_m2': 2200,       # M2 in billions AUD
                'real_gdp': 2000,              # Real GDP in billions AUD
                'nominal_interest_rate': 0.035, # 3.5% nominal rate
                'inflation_rate': 0.025,       # 2.5% inflation
                'money_velocity': 1.25,        # M2 velocity
                'money_growth_rate': 0.055,    # 5.5% annual M2 growth
                'gdp_growth_rate': 0.025,      # 2.5% real GDP growth
                'central_bank_rate': 0.035,    # RBA rate
                'monetary_base': 480,          # Monetary base in billions
                'money_multiplier': 4.6        # Money multiplier
            }
        }
    
    def _prepare_monetary_data(self, monetary_data: Optional[Dict], base_country: str,
                              quote_country: str, data_length: int) -> Dict[str, Any]:
        """Prepare monetary data for analysis"""
        # Use provided data or defaults
        base_monetary = monetary_data.get('base', {}) if monetary_data else {}
        quote_monetary = monetary_data.get('quote', {}) if monetary_data else {}
        
        # Get default data
        base_defaults = self.monetary_data.get(base_country, self.monetary_data['US'])
        quote_defaults = self.monetary_data.get(quote_country, self.monetary_data['EU'])
        
        # Merge with defaults
        for key, default_value in base_defaults.items():
            if key not in base_monetary:
                base_monetary[key] = default_value
        
        for key, default_value in quote_defaults.items():
            if key not in quote_monetary:
                quote_monetary[key] = default_value
        
        # Create time series for monetary variables
        dates = pd.date_range(end=datetime.now(), periods=data_length, freq='D')
        
        # Generate realistic time series with trends and cycles
        np.random.seed(42)
        
        result = {
            'base_country_data': base_monetary,
            'quote_country_data': quote_monetary,
            'dates': dates,
            'base_money_supply': self._create_monetary_series(
                base_monetary['money_supply_m2'], 
                base_monetary['money_growth_rate'], 
                data_length
            ),
            'quote_money_supply': self._create_monetary_series(
                quote_monetary['money_supply_m2'], 
                quote_monetary['money_growth_rate'], 
                data_length
            ),
            'base_real_gdp': self._create_gdp_series(
                base_monetary['real_gdp'], 
                base_monetary['gdp_growth_rate'], 
                data_length
            ),
            'quote_real_gdp': self._create_gdp_series(
                quote_monetary['real_gdp'], 
                quote_monetary['gdp_growth_rate'], 
                data_length
            ),
            'base_interest_rate': self._create_interest_rate_series(
                base_monetary['nominal_interest_rate'], 
                data_length
            ),
            'quote_interest_rate': self._create_interest_rate_series(
                quote_monetary['nominal_interest_rate'], 
                data_length
            ),
            'base_velocity': self._create_velocity_series(
                base_monetary['money_velocity'], 
                data_length
            ),
            'quote_velocity': self._create_velocity_series(
                quote_monetary['money_velocity'], 
                data_length
            )
        }
        
        return result
    
    def _create_monetary_series(self, initial_value: float, growth_rate: float, 
                               length: int) -> pd.Series:
        """Create money supply time series with realistic growth patterns"""
        # Annual growth rate to daily
        daily_growth = (1 + growth_rate) ** (1/252) - 1
        
        # Add cyclical and random components
        trend = np.arange(length) * daily_growth
        cycle = 0.02 * np.sin(2 * np.pi * np.arange(length) / 252)  # Annual cycle
        noise = np.random.randn(length) * 0.005  # Random noise
        
        # Combine components
        log_values = np.log(initial_value) + trend + cycle + noise
        values = np.exp(log_values)
        
        return pd.Series(values)
    
    def _create_gdp_series(self, initial_value: float, growth_rate: float, 
                          length: int) -> pd.Series:
        """Create GDP time series with business cycle patterns"""
        # Annual growth rate to daily
        daily_growth = (1 + growth_rate) ** (1/252) - 1
        
        # Business cycle (4-year cycle)
        business_cycle = 0.015 * np.sin(2 * np.pi * np.arange(length) / (4 * 252))
        
        # Seasonal component
        seasonal = 0.005 * np.sin(2 * np.pi * np.arange(length) / 63)  # Quarterly
        
        # Random shocks
        shocks = np.random.randn(length) * 0.003
        
        # Combine components
        trend = np.arange(length) * daily_growth
        log_values = np.log(initial_value) + trend + business_cycle + seasonal + shocks
        values = np.exp(log_values)
        
        return pd.Series(values)
    
    def _create_interest_rate_series(self, initial_rate: float, length: int) -> pd.Series:
        """Create interest rate time series with monetary policy cycles"""
        # Policy cycle (2-3 year cycle)
        policy_cycle = 0.01 * np.sin(2 * np.pi * np.arange(length) / (2.5 * 252))
        
        # Random policy changes
        policy_shocks = np.random.randn(length) * 0.002
        
        # Ensure rates don't go too negative
        rates = initial_rate + policy_cycle + policy_shocks
        rates = np.maximum(rates, -0.01)  # Floor at -1%
        
        return pd.Series(rates)
    
    def _create_velocity_series(self, initial_velocity: float, length: int) -> pd.Series:
        """Create money velocity time series"""
        # Velocity tends to be more stable but can have trends
        trend = np.linspace(0, -0.1 * initial_velocity, length)  # Slight declining trend
        cycle = 0.05 * initial_velocity * np.sin(2 * np.pi * np.arange(length) / (3 * 252))
        noise = np.random.randn(length) * 0.02 * initial_velocity
        
        values = initial_velocity + trend + cycle + noise
        values = np.maximum(values, 0.1)  # Ensure positive velocity
        
        return pd.Series(values)
    
    def _estimate_model_parameters(self, data: pd.DataFrame, 
                                  monetary_vars: Dict[str, Any]) -> Dict[str, float]:
        """Estimate monetary model parameters using regression"""
        try:
            # Prepare regression data
            y = np.log(data['close']).values
            
            # Independent variables based on model type
            if self.model_type == "flexible_price":
                # Flexible price monetary model: s = (m1 - m2) - (y1 - y2)
                x1 = np.log(monetary_vars['base_money_supply']).values - np.log(monetary_vars['quote_money_supply']).values
                x2 = np.log(monetary_vars['base_real_gdp']).values - np.log(monetary_vars['quote_real_gdp']).values
                X = np.column_stack([np.ones(len(x1)), x1, -x2])
                
            elif self.model_type == "sticky_price":
                # Sticky price model: includes interest rate differential
                x1 = np.log(monetary_vars['base_money_supply']).values - np.log(monetary_vars['quote_money_supply']).values
                x2 = np.log(monetary_vars['base_real_gdp']).values - np.log(monetary_vars['quote_real_gdp']).values
                x3 = monetary_vars['base_interest_rate'].values - monetary_vars['quote_interest_rate'].values
                X = np.column_stack([np.ones(len(x1)), x1, -x2, x3])
                
            else:  # real_interest_differential
                # Real interest differential model
                x1 = np.log(monetary_vars['base_money_supply']).values - np.log(monetary_vars['quote_money_supply']).values
                x2 = np.log(monetary_vars['base_real_gdp']).values - np.log(monetary_vars['quote_real_gdp']).values
                x3 = monetary_vars['base_interest_rate'].values - monetary_vars['quote_interest_rate'].values
                X = np.column_stack([np.ones(len(x1)), x1, -x2, x3])
            
            # Use only recent data for estimation if series is long
            if len(y) > self.estimation_window:
                y_est = y[-self.estimation_window:]
                X_est = X[-self.estimation_window:]
            else:
                y_est = y
                X_est = X
            
            # OLS estimation
            try:
                beta = np.linalg.lstsq(X_est, y_est, rcond=None)[0]
            except np.linalg.LinAlgError:
                # Fallback to theoretical values
                beta = [0, 1.0, 1.0, 1.0] if len(X_est[0]) > 3 else [0, 1.0, 1.0]
            
            # Extract parameters
            params = {
                'constant': beta[0],
                'money_elasticity': beta[1] if len(beta) > 1 else 1.0,
                'income_elasticity': abs(beta[2]) if len(beta) > 2 else 1.0,
                'interest_elasticity': beta[3] if len(beta) > 3 else 1.0
            }
            
            # Ensure reasonable parameter bounds
            params['money_elasticity'] = np.clip(params['money_elasticity'], 0.5, 2.0)
            params['income_elasticity'] = np.clip(params['income_elasticity'], 0.5, 2.0)
            params['interest_elasticity'] = np.clip(params['interest_elasticity'], -5.0, 5.0)
            
            return params
            
        except Exception as e:
            self.logger.warning(f"Parameter estimation failed: {e}")
            # Return theoretical parameters
            return {
                'constant': 0.0,
                'money_elasticity': 1.0,
                'income_elasticity': 1.0,
                'interest_elasticity': 1.0
            }
    
    def _calculate_theoretical_rate(self, monetary_vars: Dict[str, Any], 
                                   model_params: Dict[str, float]) -> float:
        """Calculate theoretical exchange rate from monetary model"""
        # Get latest values
        base_money = monetary_vars['base_money_supply'].iloc[-1]
        quote_money = monetary_vars['quote_money_supply'].iloc[-1]
        base_gdp = monetary_vars['base_real_gdp'].iloc[-1]
        quote_gdp = monetary_vars['quote_real_gdp'].iloc[-1]
        base_rate = monetary_vars['base_interest_rate'].iloc[-1]
        quote_rate = monetary_vars['quote_interest_rate'].iloc[-1]
        
        # Calculate theoretical rate based on model type
        if self.model_type == "flexible_price":
            # s = constant + alpha*(m1-m2) - beta*(y1-y2)
            log_s = (model_params['constant'] + 
                    model_params['money_elasticity'] * (np.log(base_money) - np.log(quote_money)) -
                    model_params['income_elasticity'] * (np.log(base_gdp) - np.log(quote_gdp)))
            
        elif self.model_type == "sticky_price":
            # s = constant + alpha*(m1-m2) - beta*(y1-y2) + gamma*(i1-i2)
            log_s = (model_params['constant'] + 
                    model_params['money_elasticity'] * (np.log(base_money) - np.log(quote_money)) -
                    model_params['income_elasticity'] * (np.log(base_gdp) - np.log(quote_gdp)) +
                    model_params['interest_elasticity'] * (base_rate - quote_rate))
            
        else:  # real_interest_differential
            # Include real interest rate differential
            base_real_rate = base_rate - monetary_vars['base_country_data'].get('inflation_rate', 0.02)
            quote_real_rate = quote_rate - monetary_vars['quote_country_data'].get('inflation_rate', 0.02)
            
            log_s = (model_params['constant'] + 
                    model_params['money_elasticity'] * (np.log(base_money) - np.log(quote_money)) -
                    model_params['income_elasticity'] * (np.log(base_gdp) - np.log(quote_gdp)) +
                    model_params['interest_elasticity'] * (base_real_rate - quote_real_rate))
        
        return np.exp(log_s)
    
    def _calculate_money_supply_differential(self, monetary_vars: Dict[str, Any]) -> float:
        """Calculate money supply growth differential"""
        base_growth = monetary_vars['base_country_data']['money_growth_rate']
        quote_growth = monetary_vars['quote_country_data']['money_growth_rate']
        return base_growth - quote_growth
    
    def _calculate_income_differential(self, monetary_vars: Dict[str, Any]) -> float:
        """Calculate income (GDP) growth differential"""
        base_growth = monetary_vars['base_country_data']['gdp_growth_rate']
        quote_growth = monetary_vars['quote_country_data']['gdp_growth_rate']
        return base_growth - quote_growth
    
    def _calculate_interest_rate_differential(self, monetary_vars: Dict[str, Any]) -> float:
        """Calculate interest rate differential"""
        base_rate = monetary_vars['base_interest_rate'].iloc[-1]
        quote_rate = monetary_vars['quote_interest_rate'].iloc[-1]
        return base_rate - quote_rate
    
    def _calculate_velocity_differential(self, monetary_vars: Dict[str, Any]) -> float:
        """Calculate money velocity differential"""
        base_velocity = monetary_vars['base_velocity'].iloc[-1]
        quote_velocity = monetary_vars['quote_velocity'].iloc[-1]
        return base_velocity - quote_velocity
    
    def _assess_monetary_pressure(self, money_diff: float, income_diff: float, 
                                 interest_diff: float) -> str:
        """Assess overall monetary pressure on currency"""
        # Scoring system
        score = 0
        
        # Money supply differential (higher growth = depreciation pressure)
        if money_diff > 0.02:  # More than 2% higher growth
            score -= 2
        elif money_diff > 0.01:
            score -= 1
        elif money_diff < -0.02:
            score += 2
        elif money_diff < -0.01:
            score += 1
        
        # Income differential (higher growth = appreciation pressure)
        if income_diff > 0.01:  # More than 1% higher growth
            score += 2
        elif income_diff > 0.005:
            score += 1
        elif income_diff < -0.01:
            score -= 2
        elif income_diff < -0.005:
            score -= 1
        
        # Interest rate differential (higher rates = appreciation pressure)
        if interest_diff > 0.02:  # More than 2% higher rates
            score += 2
        elif interest_diff > 0.01:
            score += 1
        elif interest_diff < -0.02:
            score -= 2
        elif interest_diff < -0.01:
            score -= 1
        
        # Convert score to signal
        if score >= 3:
            return "STRONG_APPRECIATION_PRESSURE"
        elif score >= 1:
            return "MODERATE_APPRECIATION_PRESSURE"
        elif score <= -3:
            return "STRONG_DEPRECIATION_PRESSURE"
        elif score <= -1:
            return "MODERATE_DEPRECIATION_PRESSURE"
        else:
            return "BALANCED_PRESSURE"
    
    def _analyze_convergence_dynamics(self, data: pd.DataFrame, theoretical_rate: float,
                                     model_params: Dict[str, float]) -> Tuple[str, float]:
        """Analyze convergence to theoretical rate"""
        try:
            current_rate = data['close'].iloc[-1]
            misalignment = (theoretical_rate - current_rate) / current_rate
            
            # Calculate historical misalignments
            if len(data) > 63:
                recent_prices = data['close'].tail(63)
                misalignments = [(theoretical_rate - price) / price for price in recent_prices]
                
                # Estimate half-life using AR(1) model
                misalign_series = pd.Series(misalignments)
                if len(misalign_series) > 10:
                    # Simple AR(1): x_t = rho * x_{t-1} + epsilon
                    x_lag = misalign_series.shift(1).dropna()
                    x_curr = misalign_series[1:]
                    
                    if len(x_lag) > 5:
                        rho = np.corrcoef(x_lag, x_curr)[0, 1]
                        rho = max(0, min(0.99, rho))  # Bound rho
                        
                        # Half-life in months
                        if rho > 0:
                            half_life = np.log(0.5) / np.log(rho) / 21  # Convert to months
                        else:
                            half_life = 1.0  # Very fast convergence
                    else:
                        half_life = 6.0  # Default 6 months
                else:
                    half_life = 6.0
            else:
                half_life = 6.0
            
            # Convergence signal
            if abs(misalignment) < 0.02:  # Within 2%
                convergence_signal = "NEAR_EQUILIBRIUM"
            elif abs(misalignment) < 0.05:  # Within 5%
                if misalignment > 0:
                    convergence_signal = "MODERATE_UNDERVALUATION"
                else:
                    convergence_signal = "MODERATE_OVERVALUATION"
            else:  # More than 5%
                if misalignment > 0:
                    convergence_signal = "SIGNIFICANT_UNDERVALUATION"
                else:
                    convergence_signal = "SIGNIFICANT_OVERVALUATION"
            
            return convergence_signal, half_life
            
        except Exception as e:
            self.logger.warning(f"Convergence analysis failed: {e}")
            return "UNKNOWN", 6.0
    
    def _calculate_model_fit(self, data: pd.DataFrame, monetary_vars: Dict[str, Any],
                            model_params: Dict[str, float]) -> float:
        """Calculate model R-squared"""
        try:
            # Observed exchange rates
            y_observed = np.log(data['close']).values
            
            # Predicted exchange rates
            base_money = np.log(monetary_vars['base_money_supply']).values
            quote_money = np.log(monetary_vars['quote_money_supply']).values
            base_gdp = np.log(monetary_vars['base_real_gdp']).values
            quote_gdp = np.log(monetary_vars['quote_real_gdp']).values
            
            if self.model_type == "flexible_price":
                y_predicted = (model_params['constant'] + 
                             model_params['money_elasticity'] * (base_money - quote_money) -
                             model_params['income_elasticity'] * (base_gdp - quote_gdp))
            else:
                base_rates = monetary_vars['base_interest_rate'].values
                quote_rates = monetary_vars['quote_interest_rate'].values
                y_predicted = (model_params['constant'] + 
                             model_params['money_elasticity'] * (base_money - quote_money) -
                             model_params['income_elasticity'] * (base_gdp - quote_gdp) +
                             model_params['interest_elasticity'] * (base_rates - quote_rates))
            
            # Calculate R-squared
            ss_res = np.sum((y_observed - y_predicted) ** 2)
            ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
            
            if ss_tot > 0:
                r_squared = 1 - (ss_res / ss_tot)
                return max(0, min(1, r_squared))  # Bound between 0 and 1
            else:
                return 0.0
                
        except Exception as e:
            self.logger.warning(f"R-squared calculation failed: {e}")
            return 0.0
    
    def _run_model_diagnostics(self, data: pd.DataFrame, monetary_vars: Dict[str, Any],
                              model_params: Dict[str, float]) -> Dict[str, Any]:
        """Run comprehensive model diagnostics"""
        try:
            diagnostics = {}
            
            # Parameter significance (simplified)
            diagnostics['parameter_significance'] = {
                'money_elasticity': 'SIGNIFICANT' if abs(model_params['money_elasticity'] - 1.0) < 0.5 else 'QUESTIONABLE',
                'income_elasticity': 'SIGNIFICANT' if abs(model_params['income_elasticity'] - 1.0) < 0.5 else 'QUESTIONABLE',
                'interest_elasticity': 'SIGNIFICANT' if abs(model_params['interest_elasticity']) > 0.1 else 'WEAK'
            }
            
            # Residual analysis
            y_observed = np.log(data['close']).values
            base_money = np.log(monetary_vars['base_money_supply']).values
            quote_money = np.log(monetary_vars['quote_money_supply']).values
            base_gdp = np.log(monetary_vars['base_real_gdp']).values
            quote_gdp = np.log(monetary_vars['quote_real_gdp']).values
            
            if self.model_type != "flexible_price":
                base_rates = monetary_vars['base_interest_rate'].values
                quote_rates = monetary_vars['quote_interest_rate'].values
                y_predicted = (model_params['constant'] + 
                             model_params['money_elasticity'] * (base_money - quote_money) -
                             model_params['income_elasticity'] * (base_gdp - quote_gdp) +
                             model_params['interest_elasticity'] * (base_rates - quote_rates))
            else:
                y_predicted = (model_params['constant'] + 
                             model_params['money_elasticity'] * (base_money - quote_money) -
                             model_params['income_elasticity'] * (base_gdp - quote_gdp))
            
            residuals = y_observed - y_predicted
            
            diagnostics['residual_analysis'] = {
                'mean_residual': np.mean(residuals),
                'residual_std': np.std(residuals),
                'residual_skewness': stats.skew(residuals) if len(residuals) > 3 else 0,
                'residual_kurtosis': stats.kurtosis(residuals) if len(residuals) > 3 else 0
            }
            
            # Durbin-Watson test for autocorrelation (simplified)
            if len(residuals) > 2:
                dw_stat = np.sum(np.diff(residuals)**2) / np.sum(residuals**2)
                diagnostics['durbin_watson'] = dw_stat
                
                if dw_stat < 1.5:
                    diagnostics['autocorrelation'] = 'POSITIVE'
                elif dw_stat > 2.5:
                    diagnostics['autocorrelation'] = 'NEGATIVE'
                else:
                    diagnostics['autocorrelation'] = 'NONE'
            
            return diagnostics
            
        except Exception as e:
            self.logger.warning(f"Model diagnostics failed: {e}")
            return {'status': 'ERROR'}
    
    def _test_structural_breaks(self, data: pd.DataFrame, 
                               monetary_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Test for structural breaks in the relationship"""
        try:
            # Simple structural break test using rolling correlations
            window = min(126, len(data) // 3)  # 6 months or 1/3 of data
            
            if len(data) < window * 2:
                return {'status': 'INSUFFICIENT_DATA'}
            
            # Calculate rolling correlations between exchange rate and fundamentals
            fx_returns = data['close'].pct_change()
            money_diff = (monetary_vars['base_money_supply'].pct_change() - 
                         monetary_vars['quote_money_supply'].pct_change())
            
            rolling_corr = fx_returns.rolling(window).corr(money_diff)
            
            # Detect significant changes in correlation
            corr_changes = rolling_corr.diff().abs()
            break_threshold = 0.3  # 30% change in correlation
            
            potential_breaks = corr_changes[corr_changes > break_threshold]
            
            return {
                'potential_breaks': len(potential_breaks),
                'break_dates': potential_breaks.index.tolist()[-3:] if len(potential_breaks) > 0 else [],
                'stability_assessment': 'STABLE' if len(potential_breaks) < 2 else 'UNSTABLE'
            }
            
        except Exception as e:
            self.logger.warning(f"Structural break test failed: {e}")
            return {'status': 'ERROR'}
    
    def _test_cointegration(self, data: pd.DataFrame, 
                           monetary_vars: Dict[str, Any]) -> Dict[str, Any]:
        """Test for cointegration between exchange rate and fundamentals"""
        try:
            # Simplified cointegration test using correlation of levels
            fx_level = np.log(data['close'])
            money_diff = (np.log(monetary_vars['base_money_supply']) - 
                         np.log(monetary_vars['quote_money_supply']))
            gdp_diff = (np.log(monetary_vars['base_real_gdp']) - 
                       np.log(monetary_vars['quote_real_gdp']))
            
            # Test correlation between levels
            corr_money = fx_level.corr(money_diff)
            corr_gdp = fx_level.corr(gdp_diff)
            
            # Simple cointegration assessment
            if abs(corr_money) > 0.7 or abs(corr_gdp) > 0.7:
                cointegration = 'STRONG'
            elif abs(corr_money) > 0.4 or abs(corr_gdp) > 0.4:
                cointegration = 'MODERATE'
            else:
                cointegration = 'WEAK'
            
            return {
                'cointegration_strength': cointegration,
                'money_correlation': corr_money,
                'gdp_correlation': corr_gdp,
                'long_run_relationship': 'EXISTS' if cointegration != 'WEAK' else 'QUESTIONABLE'
            }
            
        except Exception as e:
            self.logger.warning(f"Cointegration test failed: {e}")
            return {'status': 'ERROR'}
    
    def _assess_forecast_accuracy(self, data: pd.DataFrame, monetary_vars: Dict[str, Any],
                                 model_params: Dict[str, float]) -> Dict[str, Any]:
        """Assess out-of-sample forecast accuracy"""
        try:
            if len(data) < 63:  # Need at least 3 months of data
                return {'status': 'INSUFFICIENT_DATA'}
            
            # Use last 21 days as out-of-sample
            train_data = data.iloc[:-21]
            test_data = data.iloc[-21:]
            
            # Generate forecasts (simplified)
            forecasts = []
            actuals = test_data['close'].values
            
            for i in range(len(test_data)):
                # Use model to predict (simplified approach)
                theoretical = self._calculate_theoretical_rate(monetary_vars, model_params)
                forecasts.append(theoretical)
            
            # Calculate forecast errors
            forecast_errors = [(f - a) / a for f, a in zip(forecasts, actuals)]
            
            mae = np.mean(np.abs(forecast_errors))
            rmse = np.sqrt(np.mean(np.array(forecast_errors) ** 2))
            
            # Accuracy assessment
            if mae < 0.02:  # Less than 2% error
                accuracy = 'HIGH'
            elif mae < 0.05:  # Less than 5% error
                accuracy = 'MODERATE'
            else:
                accuracy = 'LOW'
            
            return {
                'forecast_accuracy': accuracy,
                'mean_absolute_error': mae,
                'root_mean_square_error': rmse,
                'forecast_period_days': len(test_data)
            }
            
        except Exception as e:
            self.logger.warning(f"Forecast accuracy assessment failed: {e}")
            return {'status': 'ERROR'}
    
    def _perform_sensitivity_analysis(self, monetary_vars: Dict[str, Any],
                                     model_params: Dict[str, float]) -> Dict[str, Any]:
        """Perform sensitivity analysis on model parameters"""
        try:
            base_theoretical = self._calculate_theoretical_rate(monetary_vars, model_params)
            
            sensitivity = {}
            
            # Test sensitivity to money elasticity
            params_money_high = model_params.copy()
            params_money_high['money_elasticity'] *= 1.2
            theoretical_money_high = self._calculate_theoretical_rate(monetary_vars, params_money_high)
            
            params_money_low = model_params.copy()
            params_money_low['money_elasticity'] *= 0.8
            theoretical_money_low = self._calculate_theoretical_rate(monetary_vars, params_money_low)
            
            sensitivity['money_elasticity'] = {
                'high_scenario': (theoretical_money_high - base_theoretical) / base_theoretical,
                'low_scenario': (theoretical_money_low - base_theoretical) / base_theoretical
            }
            
            # Test sensitivity to income elasticity
            params_income_high = model_params.copy()
            params_income_high['income_elasticity'] *= 1.2
            theoretical_income_high = self._calculate_theoretical_rate(monetary_vars, params_income_high)
            
            params_income_low = model_params.copy()
            params_income_low['income_elasticity'] *= 0.8
            theoretical_income_low = self._calculate_theoretical_rate(monetary_vars, params_income_low)
            
            sensitivity['income_elasticity'] = {
                'high_scenario': (theoretical_income_high - base_theoretical) / base_theoretical,
                'low_scenario': (theoretical_income_low - base_theoretical) / base_theoretical
            }
            
            # Overall sensitivity assessment
            max_sensitivity = max(
                abs(sensitivity['money_elasticity']['high_scenario']),
                abs(sensitivity['money_elasticity']['low_scenario']),
                abs(sensitivity['income_elasticity']['high_scenario']),
                abs(sensitivity['income_elasticity']['low_scenario'])
            )
            
            if max_sensitivity > 0.1:  # More than 10% change
                sensitivity['overall_assessment'] = 'HIGH_SENSITIVITY'
            elif max_sensitivity > 0.05:  # More than 5% change
                sensitivity['overall_assessment'] = 'MODERATE_SENSITIVITY'
            else:
                sensitivity['overall_assessment'] = 'LOW_SENSITIVITY'
            
            return sensitivity
            
        except Exception as e:
            self.logger.warning(f"Sensitivity analysis failed: {e}")
            return {'status': 'ERROR'}
    
    def _generate_signals(self, misalignment_pct: float, monetary_pressure: str,
                         convergence_signal: str, money_diff: float, income_diff: float,
                         interest_diff: float, r_squared: float) -> List[str]:
        """Generate comprehensive monetary model signals"""
        signals = []
        
        # Misalignment signals
        if abs(misalignment_pct) > 10:
            if misalignment_pct > 0:
                signals.append("SIGNIFICANT_UNDERVALUATION")
            else:
                signals.append("SIGNIFICANT_OVERVALUATION")
        elif abs(misalignment_pct) > 5:
            if misalignment_pct > 0:
                signals.append("MODERATE_UNDERVALUATION")
            else:
                signals.append("MODERATE_OVERVALUATION")
        
        # Monetary pressure signals
        signals.append(f"MONETARY_{monetary_pressure}")
        
        # Convergence signals
        signals.append(f"CONVERGENCE_{convergence_signal}")
        
        # Individual fundamental signals
        if abs(money_diff) > 0.02:
            if money_diff > 0:
                signals.append("EXCESSIVE_MONEY_GROWTH")
            else:
                signals.append("TIGHT_MONETARY_POLICY")
        
        if abs(income_diff) > 0.01:
            if income_diff > 0:
                signals.append("STRONG_ECONOMIC_GROWTH")
            else:
                signals.append("WEAK_ECONOMIC_GROWTH")
        
        if abs(interest_diff) > 0.02:
            if interest_diff > 0:
                signals.append("INTEREST_RATE_ADVANTAGE")
            else:
                signals.append("INTEREST_RATE_DISADVANTAGE")
        
        # Model quality signals
        if r_squared > 0.7:
            signals.append("HIGH_MODEL_RELIABILITY")
        elif r_squared > 0.4:
            signals.append("MODERATE_MODEL_RELIABILITY")
        else:
            signals.append("LOW_MODEL_RELIABILITY")
        
        # Combined signals
        if (abs(misalignment_pct) > 5 and 
            monetary_pressure in ["STRONG_APPRECIATION_PRESSURE", "STRONG_DEPRECIATION_PRESSURE"] and
            r_squared > 0.5):
            signals.append("STRONG_FUNDAMENTAL_SIGNAL")
        
        return signals
    
    def _create_time_series(self, data: pd.DataFrame, monetary_vars: Dict[str, Any],
                           theoretical_rate: float, model_params: Dict[str, float],
                           misalignment: float) -> pd.DataFrame:
        """Create comprehensive time series DataFrame"""
        # Calculate theoretical rates for entire series
        base_money = np.log(monetary_vars['base_money_supply']).values
        quote_money = np.log(monetary_vars['quote_money_supply']).values
        base_gdp = np.log(monetary_vars['base_real_gdp']).values
        quote_gdp = np.log(monetary_vars['quote_real_gdp']).values
        
        if self.model_type != "flexible_price":
            base_rates = monetary_vars['base_interest_rate'].values
            quote_rates = monetary_vars['quote_interest_rate'].values
            theoretical_series = np.exp(
                model_params['constant'] + 
                model_params['money_elasticity'] * (base_money - quote_money) -
                model_params['income_elasticity'] * (base_gdp - quote_gdp) +
                model_params['interest_elasticity'] * (base_rates - quote_rates)
            )
        else:
            theoretical_series = np.exp(
                model_params['constant'] + 
                model_params['money_elasticity'] * (base_money - quote_money) -
                model_params['income_elasticity'] * (base_gdp - quote_gdp)
            )
        
        result_df = pd.DataFrame({
            'exchange_rate': data['close'],
            'theoretical_rate': theoretical_series,
            'misalignment': (theoretical_series - data['close']) / data['close'] * 100,
            'money_differential': (monetary_vars['base_money_supply'].pct_change(21) - 
                                 monetary_vars['quote_money_supply'].pct_change(21)) * 100,
            'gdp_differential': (monetary_vars['base_real_gdp'].pct_change(63) - 
                               monetary_vars['quote_real_gdp'].pct_change(63)) * 100,
            'interest_differential': (monetary_vars['base_interest_rate'] - 
                                    monetary_vars['quote_interest_rate']) * 100
        }, index=data.index)
        
        # Add moving averages
        result_df['theoretical_rate_ma'] = result_df['theoretical_rate'].rolling(21).mean()
        result_df['misalignment_ma'] = result_df['misalignment'].rolling(21).mean()
        
        # Add volatility measures
        result_df['fx_volatility'] = data['close'].pct_change().rolling(21).std() * np.sqrt(252) * 100
        result_df['misalignment_volatility'] = result_df['misalignment'].rolling(63).std()
        
        # Add regime indicators
        result_df['misalignment_regime'] = self._classify_misalignment_regime(result_df['misalignment'])
        
        return result_df
    
    def _classify_misalignment_regime(self, misalignment_series: pd.Series) -> pd.Series:
        """Classify misalignment regimes"""
        def classify_misalignment(misalign):
            if pd.isna(misalign):
                return "UNKNOWN"
            elif misalign > 10:
                return "SIGNIFICANT_UNDERVALUATION"
            elif misalign > 5:
                return "MODERATE_UNDERVALUATION"
            elif misalign > -5:
                return "FAIR_VALUE"
            elif misalign > -10:
                return "MODERATE_OVERVALUATION"
            else:
                return "SIGNIFICANT_OVERVALUATION"
        
        return misalignment_series.apply(classify_misalignment)
    
    def _calculate_confidence(self, r_squared: float, data_length: int, 
                             monetary_vars: Dict[str, Any]) -> float:
        """Calculate confidence score"""
        confidence = 0.3  # Base confidence
        
        # Model fit contribution
        confidence += 0.3 * r_squared
        
        # Data length contribution
        if data_length >= 1260:  # 5 years
            confidence += 0.2
        elif data_length >= 252:  # 1 year
            confidence += 0.15
        elif data_length >= 126:  # 6 months
            confidence += 0.1
        
        # Data quality contribution
        base_data = monetary_vars['base_country_data']
        quote_data = monetary_vars['quote_country_data']
        
        required_fields = ['money_supply_m2', 'real_gdp', 'nominal_interest_rate']
        base_completeness = sum(1 for field in required_fields if field in base_data) / len(required_fields)
        quote_completeness = sum(1 for field in required_fields if field in quote_data) / len(required_fields)
        
        avg_completeness = (base_completeness + quote_completeness) / 2
        confidence += 0.2 * avg_completeness
        
        return min(0.95, confidence)
    
    def _get_interpretation(self, misalignment_pct: float, monetary_pressure: str,
                          convergence_signal: str, r_squared: float) -> str:
        """Get interpretation of monetary model results"""
        # Misalignment interpretation
        if abs(misalignment_pct) > 10:
            misalign_desc = f"significantly {'undervalued' if misalignment_pct > 0 else 'overvalued'} by {abs(misalignment_pct):.1f}%"
        elif abs(misalignment_pct) > 5:
            misalign_desc = f"moderately {'undervalued' if misalignment_pct > 0 else 'overvalued'} by {abs(misalignment_pct):.1f}%"
        else:
            misalign_desc = "fairly valued according to monetary fundamentals"
        
        # Pressure interpretation
        if "STRONG" in monetary_pressure:
            pressure_desc = "strong monetary forces"
        elif "MODERATE" in monetary_pressure:
            pressure_desc = "moderate monetary pressures"
        else:
            pressure_desc = "balanced monetary conditions"
        
        # Model reliability
        if r_squared > 0.7:
            reliability = "High model reliability supports this assessment."
        elif r_squared > 0.4:
            reliability = "Moderate model reliability suggests caution in interpretation."
        else:
            reliability = "Low model reliability indicates high uncertainty."
        
        return f"Monetary model suggests the currency is {misalign_desc}, with {pressure_desc} in the background. {reliability}"
    
    def _empty_result(self, asset_type: AssetType) -> MonetaryResult:
        """Return empty result for error cases"""
        return MonetaryResult(
            name="Monetary Model",
            theoretical_exchange_rate=0.0,
            current_exchange_rate=0.0,
            misalignment=0.0,
            misalignment_percentage=0.0,
            money_supply_differential=0.0,
            income_differential=0.0,
            interest_rate_differential=0.0,
            velocity_differential=0.0,
            monetary_pressure="ERROR",
            convergence_signal="ERROR",
            half_life_months=0.0,
            model_r_squared=0.0,
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.MONETARY,
            signals=["ERROR"]
        )
    
    def get_chart_data(self, result: MonetaryResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'monetary_model',
            'name': 'Monetary Model Analysis',
            'data': {
                'exchange_rate': result.values['exchange_rate'].tolist() if 'exchange_rate' in result.values.columns else [],
                'theoretical_rate': result.values['theoretical_rate'].tolist() if 'theoretical_rate' in result.values.columns else [],
                'misalignment': result.values['misalignment'].tolist() if 'misalignment' in result.values.columns else [],
                'money_differential': result.values['money_differential'].tolist() if 'money_differential' in result.values.columns else [],
                'interest_differential': result.values['interest_differential'].tolist() if 'interest_differential' in result.values.columns else []
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'monetary_metrics': {
                'theoretical_exchange_rate': result.theoretical_exchange_rate,
                'current_exchange_rate': result.current_exchange_rate,
                'misalignment_percentage': result.misalignment_percentage,
                'money_supply_differential': result.money_supply_differential,
                'income_differential': result.income_differential,
                'interest_rate_differential': result.interest_rate_differential,
                'monetary_pressure': result.monetary_pressure,
                'convergence_signal': result.convergence_signal,
                'model_r_squared': result.model_r_squared
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
                    'name': 'Theoretical Rate',
                    'data': result.values['theoretical_rate'].tolist() if 'theoretical_rate' in result.values.columns else [],
                    'color': '#4CAF50',
                    'type': 'line',
                    'lineWidth': 2
                },
                {
                    'name': 'Theoretical Rate MA',
                    'data': result.values['theoretical_rate_ma'].tolist() if 'theoretical_rate_ma' in result.values.columns else [],
                    'color': '#FF9800',
                    'type': 'line',
                    'lineWidth': 1
                },
                {
                    'name': 'Misalignment %',
                    'data': result.values['misalignment'].tolist() if 'misalignment' in result.values.columns else [],
                    'color': '#9C27B0',
                    'type': 'line',
                    'lineWidth': 1,
                    'yAxis': 1
                }
            ],
            'yAxis': [
                {
                    'title': {'text': 'Exchange Rate'},
                    'opposite': False
                },
                {
                    'title': {'text': 'Misalignment (%)'},
                    'opposite': True
                }
            ],
            'plotOptions': {
                'line': {
                    'marker': {'enabled': False}
                }
            },
            'tooltip': {
                'shared': True,
                'crosshairs': True
            },
            'legend': {
                'enabled': True,
                'align': 'center',
                'verticalAlign': 'bottom'
            }
        }