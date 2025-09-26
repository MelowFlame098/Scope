"""Discounted Cash Flow (DCF) Valuation Model

The DCF model estimates the intrinsic value of a company by projecting its future
free cash flows and discounting them back to present value. This fundamental
analysis technique helps determine if a stock is undervalued or overvalued.

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
    VALUATION = "valuation"
    FINANCIAL = "financial"


@dataclass
class DCFResult:
    """Result of DCF calculation"""
    name: str
    fair_value_per_share: float
    enterprise_value: float
    equity_value: float
    upside_downside_pct: float
    projected_fcf: List[float]
    terminal_value: float
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class DCFIndicator:
    """Discounted Cash Flow Valuation Calculator with Advanced Analysis"""
    
    def __init__(self, projection_years: int = 5, terminal_growth: float = 0.025,
                 discount_rate: float = 0.10, growth_rate: float = 0.05):
        """
        Initialize DCF calculator
        
        Args:
            projection_years: Number of years to project cash flows (default: 5)
            terminal_growth: Long-term growth rate for terminal value (default: 2.5%)
            discount_rate: Weighted Average Cost of Capital (WACC) (default: 10%)
            growth_rate: Expected FCF growth rate (default: 5%)
        """
        self.projection_years = projection_years
        self.terminal_growth = terminal_growth
        self.discount_rate = discount_rate
        self.growth_rate = growth_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame, financials: Optional[Dict] = None,
                 custom_growth_rate: Optional[float] = None,
                 custom_discount_rate: Optional[float] = None,
                 asset_type: AssetType = AssetType.STOCK) -> DCFResult:
        """
        Calculate DCF valuation for given financial data
        
        Args:
            data: Price data DataFrame with 'close' column
            financials: Dictionary containing financial metrics
            custom_growth_rate: Override default growth rate
            custom_discount_rate: Override default discount rate
            asset_type: Type of asset being analyzed
            
        Returns:
            DCFResult containing valuation analysis
        """
        try:
            # Use custom rates if provided
            growth_rate = custom_growth_rate or self.growth_rate
            discount_rate = custom_discount_rate or self.discount_rate
            
            # Validate inputs
            if discount_rate <= self.terminal_growth:
                discount_rate = self.terminal_growth + 0.02
                self.logger.warning(f"Adjusted discount rate to {discount_rate:.2%} to exceed terminal growth")
            
            # Prepare financial data
            financials = self._prepare_financials(data, financials)
            
            # Project future cash flows
            projected_fcf = self._project_cash_flows(financials['free_cash_flow'], growth_rate)
            
            # Calculate terminal value
            terminal_value = self._calculate_terminal_value(projected_fcf[-1], discount_rate)
            
            # Calculate present values
            pv_fcf = self._calculate_present_values(projected_fcf, discount_rate)
            pv_terminal = terminal_value / (1 + discount_rate) ** self.projection_years
            
            # Enterprise and equity values
            enterprise_value = sum(pv_fcf) + pv_terminal
            equity_value = self._calculate_equity_value(enterprise_value, financials)
            
            # Fair value per share
            shares_outstanding = financials.get('shares_outstanding', 1e9)
            fair_value_per_share = equity_value / shares_outstanding
            
            # Valuation analysis
            current_price = data['close'].iloc[-1]
            upside_downside = (fair_value_per_share - current_price) / current_price * 100
            
            # Generate signals
            signals = self._generate_signals(current_price, fair_value_per_share, upside_downside)
            
            # Create time series data
            values_df = self._create_time_series(data, fair_value_per_share, financials)
            
            # Calculate confidence based on data quality
            confidence = self._calculate_confidence(financials, len(data))
            
            return DCFResult(
                name="Discounted Cash Flow",
                fair_value_per_share=fair_value_per_share,
                enterprise_value=enterprise_value,
                equity_value=equity_value,
                upside_downside_pct=upside_downside,
                projected_fcf=projected_fcf,
                terminal_value=terminal_value,
                values=values_df,
                metadata={
                    'growth_rate': growth_rate,
                    'discount_rate': discount_rate,
                    'terminal_growth': self.terminal_growth,
                    'projection_years': self.projection_years,
                    'current_price': current_price,
                    'pv_fcf': pv_fcf,
                    'pv_terminal': pv_terminal,
                    'net_debt': financials.get('net_debt', 0),
                    'cash': financials.get('cash', 0),
                    'shares_outstanding': shares_outstanding,
                    'fcf_margin': financials.get('fcf_margin', 0.15),
                    'revenue_growth': financials.get('revenue_growth', growth_rate),
                    'sensitivity_analysis': self._sensitivity_analysis(financials, current_price),
                    'scenario_analysis': self._scenario_analysis(financials, current_price),
                    'valuation_multiples': self._calculate_valuation_multiples(current_price, fair_value_per_share, financials),
                    'interpretation': self._get_interpretation(upside_downside)
                },
                confidence=confidence,
                timestamp=datetime.now(),
                asset_type=asset_type,
                category=IndicatorCategory.FUNDAMENTAL,
                signals=signals
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating DCF: {e}")
            return self._empty_result(asset_type)
    
    def _prepare_financials(self, data: pd.DataFrame, financials: Optional[Dict]) -> Dict[str, Any]:
        """Prepare and validate financial data"""
        if financials is None:
            # Estimate financial metrics from market data
            current_price = data['close'].iloc[-1]
            market_cap = current_price * 1e9  # Assume 1B shares
            
            # Industry average estimates
            revenue = market_cap * 1.2  # P/S ratio of ~0.83
            fcf = revenue * 0.15  # 15% FCF margin
            
            financials = {
                'free_cash_flow': fcf,
                'revenue': revenue,
                'shares_outstanding': 1e9,
                'net_debt': 0,
                'cash': revenue * 0.1,  # 10% of revenue in cash
                'fcf_margin': 0.15,
                'revenue_growth': self.growth_rate
            }
        
        # Validate and set defaults
        required_fields = ['free_cash_flow', 'shares_outstanding']
        for field in required_fields:
            if field not in financials or financials[field] <= 0:
                if field == 'free_cash_flow':
                    financials[field] = data['close'].iloc[-1] * 1e8  # Estimate
                elif field == 'shares_outstanding':
                    financials[field] = 1e9  # Default 1B shares
        
        # Set optional fields with defaults
        financials.setdefault('net_debt', 0)
        financials.setdefault('cash', 0)
        financials.setdefault('fcf_margin', 0.15)
        financials.setdefault('revenue_growth', self.growth_rate)
        
        return financials
    
    def _project_cash_flows(self, base_fcf: float, growth_rate: float) -> List[float]:
        """Project future free cash flows"""
        projected_fcf = []
        
        for year in range(1, self.projection_years + 1):
            # Apply declining growth rate for more realistic projections
            adjusted_growth = growth_rate * (0.95 ** (year - 1))  # Gradual decline
            fcf_year = base_fcf * (1 + adjusted_growth) ** year
            projected_fcf.append(fcf_year)
        
        return projected_fcf
    
    def _calculate_terminal_value(self, final_fcf: float, discount_rate: float) -> float:
        """Calculate terminal value using Gordon Growth Model"""
        terminal_fcf = final_fcf * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - self.terminal_growth)
        return terminal_value
    
    def _calculate_present_values(self, projected_fcf: List[float], discount_rate: float) -> List[float]:
        """Calculate present values of projected cash flows"""
        pv_fcf = []
        
        for i, fcf in enumerate(projected_fcf):
            pv = fcf / (1 + discount_rate) ** (i + 1)
            pv_fcf.append(pv)
        
        return pv_fcf
    
    def _calculate_equity_value(self, enterprise_value: float, financials: Dict[str, Any]) -> float:
        """Calculate equity value from enterprise value"""
        net_debt = financials.get('net_debt', 0)
        cash = financials.get('cash', 0)
        
        # Equity Value = Enterprise Value - Net Debt + Cash
        equity_value = enterprise_value - net_debt + cash
        
        return max(0, equity_value)  # Equity value cannot be negative
    
    def _generate_signals(self, current_price: float, fair_value: float, upside_downside: float) -> List[str]:
        """Generate investment signals based on DCF analysis"""
        signals = []
        
        # Valuation signals
        if upside_downside > 20:
            signals.append("STRONG_BUY")
        elif upside_downside > 10:
            signals.append("BUY")
        elif upside_downside > -10:
            signals.append("HOLD")
        elif upside_downside > -20:
            signals.append("SELL")
        else:
            signals.append("STRONG_SELL")
        
        # Additional signals
        if upside_downside > 50:
            signals.append("DEEPLY_UNDERVALUED")
        elif upside_downside < -50:
            signals.append("DEEPLY_OVERVALUED")
        
        if abs(upside_downside) < 5:
            signals.append("FAIRLY_VALUED")
        
        # Price momentum vs valuation
        if upside_downside > 15:
            signals.append("VALUE_OPPORTUNITY")
        elif upside_downside < -15:
            signals.append("OVERVALUATION_RISK")
        
        return signals
    
    def _create_time_series(self, data: pd.DataFrame, fair_value: float, financials: Dict[str, Any]) -> pd.DataFrame:
        """Create time series DataFrame with DCF analysis"""
        # Create DCF time series (simplified - assumes constant fair value)
        dcf_values = np.full(len(data), fair_value)
        price_to_dcf = data['close'] / fair_value
        upside_downside = (fair_value - data['close']) / data['close'] * 100
        
        # Calculate rolling metrics
        rolling_window = min(252, len(data))  # 1 year or available data
        price_volatility = data['close'].rolling(rolling_window).std()
        price_trend = data['close'].rolling(rolling_window).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
        
        result_df = pd.DataFrame({
            'price': data['close'],
            'dcf_fair_value': dcf_values,
            'price_to_dcf': price_to_dcf,
            'upside_downside': upside_downside,
            'price_volatility': price_volatility,
            'price_trend': price_trend,
            'valuation_zone': self._classify_valuation_zone(price_to_dcf)
        }, index=data.index)
        
        return result_df
    
    def _classify_valuation_zone(self, price_to_dcf: pd.Series) -> pd.Series:
        """Classify valuation zones based on price-to-DCF ratio"""
        def classify_value(ratio):
            if ratio < 0.7:
                return "DEEP_VALUE"
            elif ratio < 0.9:
                return "UNDERVALUED"
            elif ratio < 1.1:
                return "FAIR_VALUE"
            elif ratio < 1.3:
                return "OVERVALUED"
            else:
                return "EXPENSIVE"
        
        return price_to_dcf.apply(classify_value)
    
    def _calculate_confidence(self, financials: Dict[str, Any], data_length: int) -> float:
        """Calculate confidence score based on data quality"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on data availability
        if 'revenue' in financials and financials['revenue'] > 0:
            confidence += 0.1
        if 'net_debt' in financials:
            confidence += 0.1
        if 'cash' in financials:
            confidence += 0.1
        if data_length >= 252:  # At least 1 year of data
            confidence += 0.1
        if data_length >= 1260:  # At least 5 years of data
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _sensitivity_analysis(self, financials: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Perform sensitivity analysis on key assumptions"""
        base_fcf = financials['free_cash_flow']
        shares = financials['shares_outstanding']
        
        # Test different growth and discount rates
        growth_rates = [self.growth_rate - 0.02, self.growth_rate, self.growth_rate + 0.02]
        discount_rates = [self.discount_rate - 0.01, self.discount_rate, self.discount_rate + 0.01]
        
        sensitivity_matrix = {}
        
        for gr in growth_rates:
            for dr in discount_rates:
                if dr > self.terminal_growth:
                    # Quick DCF calculation
                    projected_fcf = [base_fcf * (1 + gr) ** year for year in range(1, self.projection_years + 1)]
                    terminal_value = projected_fcf[-1] * (1 + self.terminal_growth) / (dr - self.terminal_growth)
                    
                    pv_fcf = sum([fcf / (1 + dr) ** (i + 1) for i, fcf in enumerate(projected_fcf)])
                    pv_terminal = terminal_value / (1 + dr) ** self.projection_years
                    
                    enterprise_value = pv_fcf + pv_terminal
                    equity_value = self._calculate_equity_value(enterprise_value, financials)
                    fair_value = equity_value / shares
                    
                    sensitivity_matrix[f"g{gr:.1%}_d{dr:.1%}"] = {
                        'fair_value': fair_value,
                        'upside': (fair_value - current_price) / current_price * 100
                    }
        
        return sensitivity_matrix
    
    def _scenario_analysis(self, financials: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """Perform scenario analysis (bull, base, bear cases)"""
        base_fcf = financials['free_cash_flow']
        shares = financials['shares_outstanding']
        
        scenarios = {
            'bear': {'growth_mult': 0.5, 'discount_add': 0.02, 'terminal_mult': 0.8},
            'base': {'growth_mult': 1.0, 'discount_add': 0.0, 'terminal_mult': 1.0},
            'bull': {'growth_mult': 1.5, 'discount_add': -0.01, 'terminal_mult': 1.2}
        }
        
        scenario_results = {}
        
        for scenario_name, params in scenarios.items():
            growth_rate = self.growth_rate * params['growth_mult']
            discount_rate = max(self.discount_rate + params['discount_add'], self.terminal_growth + 0.01)
            terminal_growth = self.terminal_growth * params['terminal_mult']
            
            # Calculate scenario DCF
            projected_fcf = [base_fcf * (1 + growth_rate) ** year for year in range(1, self.projection_years + 1)]
            terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
            
            pv_fcf = sum([fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(projected_fcf)])
            pv_terminal = terminal_value / (1 + discount_rate) ** self.projection_years
            
            enterprise_value = pv_fcf + pv_terminal
            equity_value = self._calculate_equity_value(enterprise_value, financials)
            fair_value = equity_value / shares
            
            scenario_results[scenario_name] = {
                'fair_value': fair_value,
                'upside': (fair_value - current_price) / current_price * 100,
                'probability': 0.33  # Equal probability for simplicity
            }
        
        # Calculate probability-weighted fair value
        weighted_fair_value = sum([result['fair_value'] * result['probability'] 
                                 for result in scenario_results.values()])
        
        scenario_results['weighted'] = {
            'fair_value': weighted_fair_value,
            'upside': (weighted_fair_value - current_price) / current_price * 100
        }
        
        return scenario_results
    
    def _calculate_valuation_multiples(self, current_price: float, fair_value: float, 
                                     financials: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate relevant valuation multiples"""
        multiples = {}
        
        # Price-to-DCF ratio
        multiples['price_to_dcf'] = current_price / fair_value
        
        # Implied multiples based on DCF
        if 'revenue' in financials and financials['revenue'] > 0:
            shares = financials['shares_outstanding']
            revenue_per_share = financials['revenue'] / shares
            multiples['dcf_implied_ps'] = fair_value / revenue_per_share
            multiples['current_ps'] = current_price / revenue_per_share
        
        if 'free_cash_flow' in financials and financials['free_cash_flow'] > 0:
            shares = financials['shares_outstanding']
            fcf_per_share = financials['free_cash_flow'] / shares
            multiples['dcf_implied_p_fcf'] = fair_value / fcf_per_share
            multiples['current_p_fcf'] = current_price / fcf_per_share
        
        return multiples
    
    def _get_interpretation(self, upside_downside: float) -> str:
        """Get interpretation of DCF results"""
        if upside_downside > 30:
            return "Stock appears significantly undervalued with strong upside potential"
        elif upside_downside > 15:
            return "Stock appears undervalued with moderate upside potential"
        elif upside_downside > -15:
            return "Stock appears fairly valued with limited upside/downside"
        elif upside_downside > -30:
            return "Stock appears overvalued with moderate downside risk"
        else:
            return "Stock appears significantly overvalued with high downside risk"
    
    def _empty_result(self, asset_type: AssetType) -> DCFResult:
        """Return empty result for error cases"""
        return DCFResult(
            name="Discounted Cash Flow",
            fair_value_per_share=0.0,
            enterprise_value=0.0,
            equity_value=0.0,
            upside_downside_pct=0.0,
            projected_fcf=[],
            terminal_value=0.0,
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.FUNDAMENTAL,
            signals=["ERROR"]
        )
    
    def get_chart_data(self, result: DCFResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'dcf_valuation',
            'name': 'DCF Valuation Analysis',
            'data': {
                'price': result.values['price'].tolist() if 'price' in result.values.columns else [],
                'fair_value': result.values['dcf_fair_value'].tolist() if 'dcf_fair_value' in result.values.columns else [],
                'upside_downside': result.values['upside_downside'].tolist() if 'upside_downside' in result.values.columns else []
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'valuation': {
                'fair_value_per_share': result.fair_value_per_share,
                'current_upside_downside': result.upside_downside_pct,
                'enterprise_value': result.enterprise_value,
                'equity_value': result.equity_value
            },
            'series': [
                {
                    'name': 'Stock Price',
                    'data': result.values['price'].tolist() if 'price' in result.values.columns else [],
                    'color': '#2196F3',
                    'type': 'line',
                    'lineWidth': 2
                },
                {
                    'name': 'DCF Fair Value',
                    'data': result.values['dcf_fair_value'].tolist() if 'dcf_fair_value' in result.values.columns else [],
                    'color': '#4CAF50',
                    'type': 'line',
                    'lineWidth': 2,
                    'dashStyle': 'Dash'
                }
            ],
            'zones': [
                {
                    'value': result.fair_value_per_share * 0.9,
                    'color': 'rgba(76, 175, 80, 0.1)',
                    'label': 'Undervalued Zone'
                },
                {
                    'value': result.fair_value_per_share * 1.1,
                    'color': 'rgba(244, 67, 54, 0.1)',
                    'label': 'Overvalued Zone'
                }
            ]
        }


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    sample_prices = 100 + np.cumsum(np.random.randn(252) * 0.02)
    sample_data = pd.DataFrame({'close': sample_prices}, index=dates)
    
    # Sample financials
    sample_financials = {
        'free_cash_flow': 5e9,  # $5B FCF
        'revenue': 50e9,  # $50B revenue
        'shares_outstanding': 1e9,  # 1B shares
        'net_debt': 2e9,  # $2B net debt
        'cash': 10e9,  # $10B cash
        'fcf_margin': 0.10
    }
    
    # Calculate DCF
    dcf_calculator = DCFIndicator()
    result = dcf_calculator.calculate(sample_data, sample_financials, AssetType.STOCK)
    
    print(f"DCF Analysis:")
    print(f"Fair Value per Share: ${result.fair_value_per_share:.2f}")
    print(f"Current Price: ${result.metadata['current_price']:.2f}")
    print(f"Upside/Downside: {result.upside_downside_pct:.1f}%")
    print(f"Enterprise Value: ${result.enterprise_value/1e9:.1f}B")
    print(f"Signals: {', '.join(result.signals)}")