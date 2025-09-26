"""Dividend Discount Model (DDM) Valuation

The DDM values a stock based on the present value of its expected future dividends.
This model is particularly useful for dividend-paying stocks and includes both
simple Gordon Growth Model and multi-stage DDM implementations.

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
    DIVIDEND = "dividend"


@dataclass
class DDMResult:
    """Result of DDM calculation"""
    name: str
    simple_fair_value: float
    multistage_fair_value: float
    current_dividend_yield: float
    projected_dividends: List[float]
    terminal_value: float
    upside_simple: float
    upside_multistage: float
    values: pd.DataFrame
    metadata: Dict[str, Any]
    confidence: float
    timestamp: datetime
    asset_type: AssetType
    category: IndicatorCategory
    signals: List[str]


class DDMIndicator:
    """Dividend Discount Model Calculator with Advanced Analysis"""
    
    def __init__(self, required_return: float = 0.08, growth_rate: float = 0.03,
                 high_growth_years: int = 5, stable_growth_rate: float = 0.02):
        """
        Initialize DDM calculator
        
        Args:
            required_return: Required rate of return (default: 8%)
            growth_rate: Expected dividend growth rate (default: 3%)
            high_growth_years: Years of high growth in multi-stage model (default: 5)
            stable_growth_rate: Long-term stable growth rate (default: 2%)
        """
        self.required_return = required_return
        self.growth_rate = growth_rate
        self.high_growth_years = high_growth_years
        self.stable_growth_rate = stable_growth_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate(self, data: pd.DataFrame, dividend_data: Optional[Dict] = None,
                 custom_required_return: Optional[float] = None,
                 custom_growth_rate: Optional[float] = None,
                 asset_type: AssetType = AssetType.STOCK) -> DDMResult:
        """
        Calculate DDM valuation for given dividend data
        
        Args:
            data: Price data DataFrame with 'close' column
            dividend_data: Dictionary containing dividend information
            custom_required_return: Override default required return
            custom_growth_rate: Override default growth rate
            asset_type: Type of asset being analyzed
            
        Returns:
            DDMResult containing valuation analysis
        """
        try:
            # Use custom rates if provided
            required_return = custom_required_return or self.required_return
            growth_rate = custom_growth_rate or self.growth_rate
            
            # Validate inputs
            if required_return <= growth_rate:
                required_return = growth_rate + 0.02
                self.logger.warning(f"Adjusted required return to {required_return:.2%} to exceed growth rate")
            
            # Prepare dividend data
            dividend_data = self._prepare_dividend_data(data, dividend_data)
            
            # Calculate simple DDM (Gordon Growth Model)
            simple_fair_value = self._calculate_simple_ddm(dividend_data, growth_rate, required_return)
            
            # Calculate multi-stage DDM
            multistage_fair_value, projected_dividends, terminal_value = self._calculate_multistage_ddm(
                dividend_data, growth_rate, required_return
            )
            
            # Current analysis
            current_price = data['close'].iloc[-1]
            current_dividend = dividend_data['annual_dividend']
            current_dividend_yield = current_dividend / current_price
            
            # Calculate upside/downside
            upside_simple = (simple_fair_value - current_price) / current_price * 100
            upside_multistage = (multistage_fair_value - current_price) / current_price * 100
            
            # Generate signals
            signals = self._generate_signals(current_price, simple_fair_value, multistage_fair_value, 
                                           current_dividend_yield, dividend_data)
            
            # Create time series data
            values_df = self._create_time_series(data, simple_fair_value, multistage_fair_value, 
                                               current_dividend_yield, dividend_data)
            
            # Calculate confidence
            confidence = self._calculate_confidence(dividend_data, len(data))
            
            return DDMResult(
                name="Dividend Discount Model",
                simple_fair_value=simple_fair_value,
                multistage_fair_value=multistage_fair_value,
                current_dividend_yield=current_dividend_yield,
                projected_dividends=projected_dividends,
                terminal_value=terminal_value,
                upside_simple=upside_simple,
                upside_multistage=upside_multistage,
                values=values_df,
                metadata={
                    'required_return': required_return,
                    'growth_rate': growth_rate,
                    'stable_growth_rate': self.stable_growth_rate,
                    'high_growth_years': self.high_growth_years,
                    'current_price': current_price,
                    'annual_dividend': current_dividend,
                    'payout_ratio': dividend_data.get('payout_ratio', 0.4),
                    'dividend_growth_history': dividend_data.get('dividend_history', []),
                    'dividend_sustainability': self._analyze_dividend_sustainability(dividend_data),
                    'yield_analysis': self._analyze_yield(current_dividend_yield, dividend_data),
                    'growth_analysis': self._analyze_growth_prospects(dividend_data, growth_rate),
                    'sensitivity_analysis': self._sensitivity_analysis(dividend_data, current_price, required_return, growth_rate),
                    'peer_comparison': self._peer_comparison(current_dividend_yield, dividend_data),
                    'interpretation': self._get_interpretation(upside_simple, upside_multistage, current_dividend_yield)
                },
                confidence=confidence,
                timestamp=datetime.now(),
                asset_type=asset_type,
                category=IndicatorCategory.FUNDAMENTAL,
                signals=signals
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating DDM: {e}")
            return self._empty_result(asset_type)
    
    def _prepare_dividend_data(self, data: pd.DataFrame, dividend_data: Optional[Dict]) -> Dict[str, Any]:
        """Prepare and validate dividend data"""
        if dividend_data is None:
            # Estimate dividend data from market data
            current_price = data['close'].iloc[-1]
            estimated_yield = 0.025  # 2.5% yield assumption
            annual_dividend = current_price * estimated_yield
            
            dividend_data = {
                'annual_dividend': annual_dividend,
                'dividend_history': [annual_dividend * 0.95, annual_dividend],
                'payout_ratio': 0.4,
                'dividend_frequency': 4  # Quarterly
            }
        
        # Validate and set defaults
        if 'annual_dividend' not in dividend_data or dividend_data['annual_dividend'] <= 0:
            current_price = data['close'].iloc[-1]
            dividend_data['annual_dividend'] = current_price * 0.02  # 2% yield default
        
        # Set optional fields with defaults
        dividend_data.setdefault('payout_ratio', 0.4)
        dividend_data.setdefault('dividend_frequency', 4)
        dividend_data.setdefault('dividend_history', [dividend_data['annual_dividend']])
        
        # Calculate historical growth if history available
        if len(dividend_data['dividend_history']) >= 2:
            history = dividend_data['dividend_history']
            historical_growth = (history[-1] / history[0]) ** (1 / (len(history) - 1)) - 1
            dividend_data['historical_growth'] = historical_growth
        else:
            dividend_data['historical_growth'] = self.growth_rate
        
        return dividend_data
    
    def _calculate_simple_ddm(self, dividend_data: Dict[str, Any], growth_rate: float, 
                             required_return: float) -> float:
        """Calculate simple DDM using Gordon Growth Model"""
        current_dividend = dividend_data['annual_dividend']
        next_dividend = current_dividend * (1 + growth_rate)
        
        # Gordon Growth Model: P = D1 / (r - g)
        fair_value = next_dividend / (required_return - growth_rate)
        
        return fair_value
    
    def _calculate_multistage_ddm(self, dividend_data: Dict[str, Any], growth_rate: float, 
                                 required_return: float) -> Tuple[float, List[float], float]:
        """Calculate multi-stage DDM with high growth then stable growth"""
        current_dividend = dividend_data['annual_dividend']
        
        # Stage 1: High growth period
        high_growth_rate = min(growth_rate * 1.5, 0.15)  # Cap at 15%
        stage1_dividends = []
        
        for year in range(1, self.high_growth_years + 1):
            # Gradually declining growth rate
            year_growth = high_growth_rate * (0.9 ** (year - 1))
            dividend = current_dividend * (1 + year_growth) ** year
            stage1_dividends.append(dividend)
        
        # Stage 2: Terminal value with stable growth
        terminal_dividend = stage1_dividends[-1] * (1 + self.stable_growth_rate)
        terminal_value = terminal_dividend / (required_return - self.stable_growth_rate)
        
        # Present value calculations
        pv_stage1 = sum([div / (1 + required_return) ** (i + 1) 
                        for i, div in enumerate(stage1_dividends)])
        pv_terminal = terminal_value / (1 + required_return) ** self.high_growth_years
        
        multistage_fair_value = pv_stage1 + pv_terminal
        
        return multistage_fair_value, stage1_dividends, terminal_value
    
    def _generate_signals(self, current_price: float, simple_fair_value: float, 
                         multistage_fair_value: float, dividend_yield: float, 
                         dividend_data: Dict[str, Any]) -> List[str]:
        """Generate investment signals based on DDM analysis"""
        signals = []
        
        # Valuation signals (using multistage as primary)
        upside = (multistage_fair_value - current_price) / current_price * 100
        
        if upside > 20:
            signals.append("STRONG_BUY")
        elif upside > 10:
            signals.append("BUY")
        elif upside > -10:
            signals.append("HOLD")
        elif upside > -20:
            signals.append("SELL")
        else:
            signals.append("STRONG_SELL")
        
        # Dividend yield signals
        if dividend_yield > 0.06:  # 6%+
            signals.append("HIGH_YIELD")
        elif dividend_yield > 0.04:  # 4-6%
            signals.append("ATTRACTIVE_YIELD")
        elif dividend_yield < 0.01:  # <1%
            signals.append("LOW_YIELD")
        
        # Dividend growth signals
        historical_growth = dividend_data.get('historical_growth', 0)
        if historical_growth > 0.08:  # 8%+ growth
            signals.append("STRONG_DIVIDEND_GROWTH")
        elif historical_growth > 0.03:  # 3-8% growth
            signals.append("STEADY_DIVIDEND_GROWTH")
        elif historical_growth < 0:
            signals.append("DIVIDEND_CUTS")
        
        # Sustainability signals
        payout_ratio = dividend_data.get('payout_ratio', 0.5)
        if payout_ratio > 0.8:
            signals.append("HIGH_PAYOUT_RISK")
        elif payout_ratio < 0.3:
            signals.append("CONSERVATIVE_PAYOUT")
        
        # Value vs growth signals
        simple_upside = (simple_fair_value - current_price) / current_price * 100
        if abs(simple_upside - upside) > 15:
            signals.append("MODEL_DIVERGENCE")
        
        # Income vs appreciation potential
        if dividend_yield > 0.04 and upside > 10:
            signals.append("INCOME_AND_GROWTH")
        elif dividend_yield > 0.05 and upside < 5:
            signals.append("INCOME_FOCUSED")
        
        return signals
    
    def _create_time_series(self, data: pd.DataFrame, simple_fair_value: float, 
                           multistage_fair_value: float, dividend_yield: float, 
                           dividend_data: Dict[str, Any]) -> pd.DataFrame:
        """Create time series DataFrame with DDM analysis"""
        # Create DDM time series (simplified - assumes constant fair values)
        simple_values = np.full(len(data), simple_fair_value)
        multistage_values = np.full(len(data), multistage_fair_value)
        
        # Calculate ratios and metrics
        price_to_simple = data['close'] / simple_fair_value
        price_to_multistage = data['close'] / multistage_fair_value
        
        # Estimated dividend yield over time (assuming constant dividend)
        estimated_yield = dividend_data['annual_dividend'] / data['close']
        
        # Rolling metrics
        rolling_window = min(252, len(data))
        price_volatility = data['close'].rolling(rolling_window).std()
        yield_volatility = estimated_yield.rolling(rolling_window).std()
        
        result_df = pd.DataFrame({
            'price': data['close'],
            'ddm_simple': simple_values,
            'ddm_multistage': multistage_values,
            'price_to_simple': price_to_simple,
            'price_to_multistage': price_to_multistage,
            'estimated_yield': estimated_yield,
            'price_volatility': price_volatility,
            'yield_volatility': yield_volatility,
            'valuation_zone': self._classify_valuation_zone(price_to_multistage)
        }, index=data.index)
        
        return result_df
    
    def _classify_valuation_zone(self, price_to_ddm: pd.Series) -> pd.Series:
        """Classify valuation zones based on price-to-DDM ratio"""
        def classify_value(ratio):
            if ratio < 0.8:
                return "DEEP_VALUE"
            elif ratio < 0.95:
                return "UNDERVALUED"
            elif ratio < 1.05:
                return "FAIR_VALUE"
            elif ratio < 1.2:
                return "OVERVALUED"
            else:
                return "EXPENSIVE"
        
        return price_to_ddm.apply(classify_value)
    
    def _analyze_dividend_sustainability(self, dividend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dividend sustainability"""
        payout_ratio = dividend_data.get('payout_ratio', 0.5)
        historical_growth = dividend_data.get('historical_growth', 0)
        
        # Sustainability score (0-100)
        sustainability_score = 100
        
        # Adjust for payout ratio
        if payout_ratio > 0.8:
            sustainability_score -= 30
        elif payout_ratio > 0.6:
            sustainability_score -= 15
        elif payout_ratio < 0.3:
            sustainability_score -= 5  # Too conservative might indicate issues
        
        # Adjust for growth consistency
        if historical_growth < 0:
            sustainability_score -= 40
        elif historical_growth < 0.02:
            sustainability_score -= 20
        elif historical_growth > 0.15:
            sustainability_score -= 10  # Unsustainable high growth
        
        # Determine sustainability level
        if sustainability_score >= 80:
            level = "HIGH"
        elif sustainability_score >= 60:
            level = "MEDIUM"
        elif sustainability_score >= 40:
            level = "LOW"
        else:
            level = "AT_RISK"
        
        return {
            'sustainability_score': sustainability_score,
            'sustainability_level': level,
            'payout_ratio': payout_ratio,
            'historical_growth': historical_growth,
            'risk_factors': self._identify_dividend_risks(dividend_data)
        }
    
    def _identify_dividend_risks(self, dividend_data: Dict[str, Any]) -> List[str]:
        """Identify potential dividend risks"""
        risks = []
        
        payout_ratio = dividend_data.get('payout_ratio', 0.5)
        historical_growth = dividend_data.get('historical_growth', 0)
        
        if payout_ratio > 0.9:
            risks.append("VERY_HIGH_PAYOUT_RATIO")
        elif payout_ratio > 0.7:
            risks.append("HIGH_PAYOUT_RATIO")
        
        if historical_growth < -0.05:
            risks.append("DECLINING_DIVIDENDS")
        elif historical_growth < 0:
            risks.append("RECENT_DIVIDEND_CUTS")
        
        if historical_growth > 0.2:
            risks.append("UNSUSTAINABLE_GROWTH_RATE")
        
        # Check dividend history volatility
        dividend_history = dividend_data.get('dividend_history', [])
        if len(dividend_history) >= 3:
            volatility = np.std(dividend_history) / np.mean(dividend_history)
            if volatility > 0.2:
                risks.append("VOLATILE_DIVIDEND_HISTORY")
        
        return risks
    
    def _analyze_yield(self, current_yield: float, dividend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dividend yield characteristics"""
        # Industry benchmarks (simplified)
        low_yield_threshold = 0.02
        high_yield_threshold = 0.06
        
        if current_yield < low_yield_threshold:
            yield_category = "LOW_YIELD"
            attractiveness = "GROWTH_FOCUSED"
        elif current_yield > high_yield_threshold:
            yield_category = "HIGH_YIELD"
            attractiveness = "INCOME_FOCUSED"
        else:
            yield_category = "MODERATE_YIELD"
            attractiveness = "BALANCED"
        
        # Yield sustainability
        payout_ratio = dividend_data.get('payout_ratio', 0.5)
        if current_yield > 0.08 and payout_ratio > 0.8:
            sustainability = "QUESTIONABLE"
        elif current_yield > 0.06 and payout_ratio > 0.7:
            sustainability = "MODERATE"
        else:
            sustainability = "SUSTAINABLE"
        
        return {
            'current_yield': current_yield,
            'yield_category': yield_category,
            'attractiveness': attractiveness,
            'sustainability': sustainability,
            'yield_percentile': self._estimate_yield_percentile(current_yield)
        }
    
    def _estimate_yield_percentile(self, yield_value: float) -> int:
        """Estimate yield percentile (simplified)"""
        # Rough market percentiles
        if yield_value < 0.01:
            return 10
        elif yield_value < 0.02:
            return 25
        elif yield_value < 0.03:
            return 50
        elif yield_value < 0.05:
            return 75
        else:
            return 90
    
    def _analyze_growth_prospects(self, dividend_data: Dict[str, Any], growth_rate: float) -> Dict[str, Any]:
        """Analyze dividend growth prospects"""
        historical_growth = dividend_data.get('historical_growth', 0)
        payout_ratio = dividend_data.get('payout_ratio', 0.5)
        
        # Growth sustainability
        if payout_ratio < 0.4:
            growth_potential = "HIGH"  # Room to increase dividends
        elif payout_ratio < 0.6:
            growth_potential = "MODERATE"
        else:
            growth_potential = "LIMITED"
        
        # Growth consistency
        if historical_growth > 0.05 and historical_growth < 0.15:
            consistency = "CONSISTENT"
        elif abs(historical_growth - growth_rate) < 0.02:
            consistency = "STABLE"
        else:
            consistency = "VOLATILE"
        
        return {
            'historical_growth': historical_growth,
            'expected_growth': growth_rate,
            'growth_potential': growth_potential,
            'consistency': consistency,
            'payout_ratio': payout_ratio
        }
    
    def _sensitivity_analysis(self, dividend_data: Dict[str, Any], current_price: float,
                             required_return: float, growth_rate: float) -> Dict[str, Any]:
        """Perform sensitivity analysis on key assumptions"""
        current_dividend = dividend_data['annual_dividend']
        
        # Test different required returns and growth rates
        returns = [required_return - 0.01, required_return, required_return + 0.01]
        growth_rates = [growth_rate - 0.01, growth_rate, growth_rate + 0.01]
        
        sensitivity_matrix = {}
        
        for rr in returns:
            for gr in growth_rates:
                if rr > gr:
                    # Simple DDM calculation
                    next_dividend = current_dividend * (1 + gr)
                    fair_value = next_dividend / (rr - gr)
                    upside = (fair_value - current_price) / current_price * 100
                    
                    sensitivity_matrix[f"r{rr:.1%}_g{gr:.1%}"] = {
                        'fair_value': fair_value,
                        'upside': upside
                    }
        
        return sensitivity_matrix
    
    def _peer_comparison(self, current_yield: float, dividend_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare with peer averages (simplified)"""
        # Industry averages (simplified assumptions)
        industry_avg_yield = 0.035
        industry_avg_growth = 0.04
        industry_avg_payout = 0.5
        
        historical_growth = dividend_data.get('historical_growth', 0)
        payout_ratio = dividend_data.get('payout_ratio', 0.5)
        
        return {
            'yield_vs_peers': 'ABOVE' if current_yield > industry_avg_yield else 'BELOW',
            'growth_vs_peers': 'ABOVE' if historical_growth > industry_avg_growth else 'BELOW',
            'payout_vs_peers': 'ABOVE' if payout_ratio > industry_avg_payout else 'BELOW',
            'yield_premium': (current_yield - industry_avg_yield) * 100,
            'growth_premium': (historical_growth - industry_avg_growth) * 100
        }
    
    def _calculate_confidence(self, dividend_data: Dict[str, Any], data_length: int) -> float:
        """Calculate confidence score based on data quality"""
        confidence = 0.5  # Base confidence
        
        # Adjust based on dividend history
        dividend_history = dividend_data.get('dividend_history', [])
        if len(dividend_history) >= 5:
            confidence += 0.2
        elif len(dividend_history) >= 3:
            confidence += 0.1
        
        # Adjust based on payout ratio availability
        if 'payout_ratio' in dividend_data:
            confidence += 0.1
        
        # Adjust based on price data length
        if data_length >= 252:
            confidence += 0.1
        if data_length >= 1260:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _get_interpretation(self, upside_simple: float, upside_multistage: float, 
                          dividend_yield: float) -> str:
        """Get interpretation of DDM results"""
        avg_upside = (upside_simple + upside_multistage) / 2
        
        if avg_upside > 20:
            valuation = "significantly undervalued"
        elif avg_upside > 10:
            valuation = "undervalued"
        elif avg_upside > -10:
            valuation = "fairly valued"
        elif avg_upside > -20:
            valuation = "overvalued"
        else:
            valuation = "significantly overvalued"
        
        if dividend_yield > 0.05:
            income_aspect = "with attractive dividend income"
        elif dividend_yield > 0.03:
            income_aspect = "with moderate dividend income"
        else:
            income_aspect = "with limited dividend income"
        
        return f"Stock appears {valuation} {income_aspect}"
    
    def _empty_result(self, asset_type: AssetType) -> DDMResult:
        """Return empty result for error cases"""
        return DDMResult(
            name="Dividend Discount Model",
            simple_fair_value=0.0,
            multistage_fair_value=0.0,
            current_dividend_yield=0.0,
            projected_dividends=[],
            terminal_value=0.0,
            upside_simple=0.0,
            upside_multistage=0.0,
            values=pd.DataFrame(),
            metadata={'error': 'Calculation failed'},
            confidence=0.0,
            timestamp=datetime.now(),
            asset_type=asset_type,
            category=IndicatorCategory.FUNDAMENTAL,
            signals=["ERROR"]
        )
    
    def get_chart_data(self, result: DDMResult) -> Dict[str, Any]:
        """Prepare data for chart visualization"""
        return {
            'type': 'ddm_valuation',
            'name': 'Dividend Discount Model',
            'data': {
                'price': result.values['price'].tolist() if 'price' in result.values.columns else [],
                'simple_fair_value': result.values['ddm_simple'].tolist() if 'ddm_simple' in result.values.columns else [],
                'multistage_fair_value': result.values['ddm_multistage'].tolist() if 'ddm_multistage' in result.values.columns else [],
                'dividend_yield': result.values['estimated_yield'].tolist() if 'estimated_yield' in result.values.columns else []
            },
            'signals': result.signals,
            'metadata': result.metadata,
            'valuation': {
                'simple_fair_value': result.simple_fair_value,
                'multistage_fair_value': result.multistage_fair_value,
                'current_dividend_yield': result.current_dividend_yield,
                'upside_simple': result.upside_simple,
                'upside_multistage': result.upside_multistage
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
                    'name': 'DDM Simple',
                    'data': result.values['ddm_simple'].tolist() if 'ddm_simple' in result.values.columns else [],
                    'color': '#4CAF50',
                    'type': 'line',
                    'lineWidth': 2,
                    'dashStyle': 'Dash'
                },
                {
                    'name': 'DDM Multi-stage',
                    'data': result.values['ddm_multistage'].tolist() if 'ddm_multistage' in result.values.columns else [],
                    'color': '#FF9800',
                    'type': 'line',
                    'lineWidth': 2,
                    'dashStyle': 'DashDot'
                }
            ],
            'dividend_info': {
                'annual_dividend': result.metadata.get('annual_dividend', 0),
                'projected_dividends': result.projected_dividends,
                'dividend_yield': result.current_dividend_yield,
                'payout_ratio': result.metadata.get('payout_ratio', 0)
            }
        }


# Example usage
if __name__ == "__main__":
    # Sample data
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    sample_prices = 50 + np.cumsum(np.random.randn(252) * 0.01)
    sample_data = pd.DataFrame({'close': sample_prices}, index=dates)
    
    # Sample dividend data
    sample_dividend_data = {
        'annual_dividend': 2.0,  # $2 annual dividend
        'dividend_history': [1.8, 1.9, 2.0],  # 3-year history
        'payout_ratio': 0.6,  # 60% payout ratio
        'dividend_frequency': 4  # Quarterly payments
    }
    
    # Calculate DDM
    ddm_calculator = DDMIndicator()
    result = ddm_calculator.calculate(sample_data, sample_dividend_data, AssetType.STOCK)
    
    print(f"DDM Analysis:")
    print(f"Simple Fair Value: ${result.simple_fair_value:.2f}")
    print(f"Multi-stage Fair Value: ${result.multistage_fair_value:.2f}")
    print(f"Current Price: ${result.metadata['current_price']:.2f}")
    print(f"Dividend Yield: {result.current_dividend_yield:.2%}")
    print(f"Upside (Simple): {result.upside_simple:.1f}%")
    print(f"Upside (Multi-stage): {result.upside_multistage:.1f}%")
    print(f"Signals: {', '.join(result.signals)}")