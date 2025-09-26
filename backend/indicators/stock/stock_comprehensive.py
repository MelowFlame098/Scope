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

class StockComprehensiveIndicators:
    """Comprehensive stock-specific indicators"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def discounted_cash_flow(self, data: pd.DataFrame, financials: Optional[Dict] = None, 
                           growth_rate: float = 0.05, discount_rate: float = 0.10, 
                           terminal_growth: float = 0.02) -> IndicatorResult:
        """Discounted Cash Flow (DCF) valuation model"""
        try:
            # Default financial data if not provided
            if financials is None:
                # Estimate based on market data
                market_cap = data['close'].iloc[-1] * 1e9  # Assume 1B shares
                revenue = market_cap * 0.8  # Revenue estimate
                fcf = revenue * 0.15  # FCF margin estimate
                financials = {
                    'free_cash_flow': fcf,
                    'revenue': revenue,
                    'shares_outstanding': 1e9
                }
            
            # Project future cash flows (5 years)
            projection_years = 5
            current_fcf = financials['free_cash_flow']
            
            projected_fcf = []
            for year in range(1, projection_years + 1):
                fcf_year = current_fcf * (1 + growth_rate) ** year
                projected_fcf.append(fcf_year)
            
            # Terminal value
            terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (discount_rate - terminal_growth)
            
            # Discount all cash flows to present value
            pv_fcf = []
            for i, fcf in enumerate(projected_fcf):
                pv = fcf / (1 + discount_rate) ** (i + 1)
                pv_fcf.append(pv)
            
            pv_terminal = terminal_value / (1 + discount_rate) ** projection_years
            
            # Enterprise value and equity value
            enterprise_value = sum(pv_fcf) + pv_terminal
            equity_value = enterprise_value  # Assuming no net debt
            
            # Fair value per share
            shares_outstanding = financials.get('shares_outstanding', 1e9)
            fair_value_per_share = equity_value / shares_outstanding
            
            # Current price vs fair value
            current_price = data['close'].iloc[-1]
            upside_downside = (fair_value_per_share - current_price) / current_price * 100
            
            # Create time series of DCF values (simplified)
            dcf_values = np.full(len(data), fair_value_per_share)
            price_to_dcf = data['close'] / fair_value_per_share
            
            result_df = pd.DataFrame({
                'price': data['close'],
                'dcf_fair_value': dcf_values,
                'price_to_dcf': price_to_dcf,
                'upside_downside': upside_downside
            }, index=data.index)
            
            return IndicatorResult(
                name="Discounted Cash Flow",
                values=result_df,
                metadata={
                    'fair_value_per_share': fair_value_per_share,
                    'enterprise_value': enterprise_value,
                    'upside_downside_pct': upside_downside,
                    'growth_rate': growth_rate,
                    'discount_rate': discount_rate,
                    'terminal_growth': terminal_growth,
                    'projected_fcf': projected_fcf,
                    'interpretation': 'Positive upside suggests undervaluation'
                },
                confidence=0.75,
                timestamp=datetime.now(),
                asset_type=AssetType.STOCK,
                category=IndicatorCategory.FUNDAMENTAL
            )
            
        except Exception as e:
            logger.error(f"Error calculating DCF: {e}")
            return self._empty_result("Discounted Cash Flow", AssetType.STOCK)
    
    def dividend_discount_model(self, data: pd.DataFrame, dividend_data: Optional[Dict] = None,
                              growth_rate: float = 0.03, required_return: float = 0.08) -> IndicatorResult:
        """Dividend Discount Model (DDM) for dividend-paying stocks"""
        try:
            # Default dividend data if not provided
            if dividend_data is None:
                # Estimate dividend yield from market data
                estimated_yield = 0.02  # 2% yield assumption
                current_dividend = data['close'].iloc[-1] * estimated_yield
                dividend_data = {
                    'annual_dividend': current_dividend,
                    'dividend_history': [current_dividend * 0.95, current_dividend]
                }
            
            current_dividend = dividend_data['annual_dividend']
            
            # Gordon Growth Model: P = D1 / (r - g)
            if required_return <= growth_rate:
                # Adjust required return if it's too low
                required_return = growth_rate + 0.02
            
            next_dividend = current_dividend * (1 + growth_rate)
            ddm_fair_value = next_dividend / (required_return - growth_rate)
            
            # Multi-stage DDM (high growth then stable growth)
            high_growth_years = 5
            high_growth_rate = growth_rate * 1.5  # Higher initial growth
            stable_growth_rate = 0.02  # Long-term stable growth
            
            # Stage 1: High growth dividends
            stage1_dividends = []
            for year in range(1, high_growth_years + 1):
                dividend = current_dividend * (1 + high_growth_rate) ** year
                stage1_dividends.append(dividend)
            
            # Stage 2: Terminal value with stable growth
            terminal_dividend = stage1_dividends[-1] * (1 + stable_growth_rate)
            terminal_value = terminal_dividend / (required_return - stable_growth_rate)
            
            # Present value calculation
            pv_stage1 = sum([div / (1 + required_return) ** (i + 1) 
                           for i, div in enumerate(stage1_dividends)])
            pv_terminal = terminal_value / (1 + required_return) ** high_growth_years
            
            multistage_fair_value = pv_stage1 + pv_terminal
            
            # Current price analysis
            current_price = data['close'].iloc[-1]
            dividend_yield = current_dividend / current_price
            
            # Create time series
            ddm_simple = np.full(len(data), ddm_fair_value)
            ddm_multistage = np.full(len(data), multistage_fair_value)
            
            result_df = pd.DataFrame({
                'price': data['close'],
                'ddm_simple': ddm_simple,
                'ddm_multistage': ddm_multistage,
                'dividend_yield': dividend_yield,
                'price_to_ddm': data['close'] / ddm_simple
            }, index=data.index)
            
            return IndicatorResult(
                name="Dividend Discount Model",
                values=result_df,
                metadata={
                    'simple_fair_value': ddm_fair_value,
                    'multistage_fair_value': multistage_fair_value,
                    'current_dividend_yield': dividend_yield,
                    'growth_rate': growth_rate,
                    'required_return': required_return,
                    'upside_simple': (ddm_fair_value - current_price) / current_price * 100,
                    'upside_multistage': (multistage_fair_value - current_price) / current_price * 100,
                    'interpretation': 'Higher fair value suggests undervaluation'
                },
                confidence=0.70,
                timestamp=datetime.now(),
                asset_type=AssetType.STOCK,
                category=IndicatorCategory.FUNDAMENTAL
            )
            
        except Exception as e:
            logger.error(f"Error calculating DDM: {e}")
            return self._empty_result("Dividend Discount Model", AssetType.STOCK)
    
    def capm_analysis(self, data: pd.DataFrame, market_data: Optional[pd.DataFrame] = None,
                     risk_free_rate: float = 0.02) -> IndicatorResult:
        """Capital Asset Pricing Model (CAPM) analysis"""
        try:
            # Use S&P 500 proxy if market data not provided
            if market_data is None:
                # Create synthetic market data based on stock data
                market_returns = data['close'].pct_change() * 0.7 + np.random.normal(0, 0.01, len(data))
                market_data = pd.DataFrame({'close': (1 + market_returns).cumprod() * 100})
            
            # Calculate returns
            stock_returns = data['close'].pct_change().dropna()
            market_returns = market_data['close'].pct_change().dropna()
            
            # Align data
            common_index = stock_returns.index.intersection(market_returns.index)
            stock_returns = stock_returns[common_index]
            market_returns = market_returns[common_index]
            
            if len(stock_returns) < 30:
                raise ValueError("Insufficient data for CAPM analysis")
            
            # Calculate beta using rolling window
            window = min(252, len(stock_returns))  # 1 year or available data
            
            # Rolling beta calculation
            rolling_beta = []
            rolling_alpha = []
            rolling_r_squared = []
            
            for i in range(window, len(stock_returns) + 1):
                y = stock_returns.iloc[i-window:i]
                x = market_returns.iloc[i-window:i]
                
                # Linear regression: stock_return = alpha + beta * market_return
                covariance = np.cov(x, y)[0, 1]
                market_variance = np.var(x)
                
                if market_variance > 0:
                    beta = covariance / market_variance
                    alpha = np.mean(y) - beta * np.mean(x)
                    
                    # R-squared
                    y_pred = alpha + beta * x
                    ss_res = np.sum((y - y_pred) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                else:
                    beta, alpha, r_squared = 1.0, 0.0, 0.0
                
                rolling_beta.append(beta)
                rolling_alpha.append(alpha)
                rolling_r_squared.append(r_squared)
            
            # Pad with NaN for initial period
            beta_series = pd.Series([np.nan] * (window - 1) + rolling_beta, index=stock_returns.index)
            alpha_series = pd.Series([np.nan] * (window - 1) + rolling_alpha, index=stock_returns.index)
            r_squared_series = pd.Series([np.nan] * (window - 1) + rolling_r_squared, index=stock_returns.index)
            
            # Expected return using CAPM
            market_risk_premium = market_returns.mean() * 252 - risk_free_rate  # Annualized
            expected_return = risk_free_rate + beta_series * market_risk_premium
            
            # Sharpe ratio
            excess_returns = stock_returns.rolling(window).mean() * 252 - risk_free_rate
            volatility = stock_returns.rolling(window).std() * np.sqrt(252)
            sharpe_ratio = excess_returns / volatility
            
            # Treynor ratio
            treynor_ratio = excess_returns / beta_series
            
            result_df = pd.DataFrame({
                'price': data['close'],
                'beta': beta_series,
                'alpha': alpha_series,
                'expected_return': expected_return,
                'sharpe_ratio': sharpe_ratio,
                'treynor_ratio': treynor_ratio,
                'r_squared': r_squared_series
            }, index=data.index)
            
            return IndicatorResult(
                name="CAPM Analysis",
                values=result_df,
                metadata={
                    'current_beta': beta_series.iloc[-1] if not pd.isna(beta_series.iloc[-1]) else 1.0,
                    'current_alpha': alpha_series.iloc[-1] if not pd.isna(alpha_series.iloc[-1]) else 0.0,
                    'risk_free_rate': risk_free_rate,
                    'market_risk_premium': market_risk_premium,
                    'current_expected_return': expected_return.iloc[-1] if not pd.isna(expected_return.iloc[-1]) else risk_free_rate,
                    'interpretation': 'Beta > 1 indicates higher volatility than market, Alpha > 0 suggests outperformance'
                },
                confidence=0.80,
                timestamp=datetime.now(),
                asset_type=AssetType.STOCK,
                category=IndicatorCategory.FUNDAMENTAL
            )
            
        except Exception as e:
            logger.error(f"Error calculating CAPM: {e}")
            return self._empty_result("CAPM Analysis", AssetType.STOCK)
    
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
        """Calculate specific stock indicator based on name"""
        indicator_map = {
            'dcf': self.discounted_cash_flow,
            'ddm': self.dividend_discount_model,
            'capm': self.capm_analysis,
        }
        
        if indicator_name not in indicator_map:
            raise ValueError(f"Unknown stock indicator: {indicator_name}")
        
        return indicator_map[indicator_name](data, **kwargs)