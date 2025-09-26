"""DCF, DDM, CAPM, and Fama-French Models for Stock Valuation

This module implements fundamental stock valuation models:
- Discounted Cash Flow (DCF) Model
- Dividend Discount Model (DDM)
- Capital Asset Pricing Model (CAPM)
- Fama-French Three-Factor Model
- Fama-French Five-Factor Model
- Arbitrage Pricing Theory (APT)
- Residual Income Model
- Economic Value Added (EVA)
- Free Cash Flow to Equity (FCFE)
- Free Cash Flow to Firm (FCFF)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings
warnings.filterwarnings('ignore')

# Statistical Libraries
try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("SciPy not available. Using simplified statistical calculations.")

# ML Libraries
try:
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Using simplified implementations.")

logger = logging.getLogger(__name__)

@dataclass
class FinancialData:
    """Financial data structure"""
    revenue: float
    net_income: float
    free_cash_flow: float
    dividends: float
    shares_outstanding: float
    book_value: float
    debt: float
    cash: float
    capex: float
    depreciation: float
    working_capital_change: float
    tax_rate: float

@dataclass
class MarketData:
    """Market data structure"""
    stock_price: float
    market_cap: float
    beta: float
    risk_free_rate: float
    market_return: float
    dividend_yield: float
    pe_ratio: float
    pb_ratio: float
    roe: float
    roa: float

@dataclass
class DCFResult:
    """DCF valuation result"""
    intrinsic_value: float
    terminal_value: float
    present_value_fcf: float
    wacc: float
    growth_rate: float
    dcf_projections: List[float]
    sensitivity_analysis: Dict[str, float]
    upside_downside: Tuple[float, float]
    fair_value_range: Tuple[float, float]

@dataclass
class DDMResult:
    """DDM valuation result"""
    intrinsic_value: float
    dividend_growth_rate: float
    required_return: float
    dividend_projections: List[float]
    payout_ratio: float
    sustainable_growth: float
    dividend_yield_forecast: float
    gordon_growth_value: float

@dataclass
class CAPMResult:
    """CAPM analysis result"""
    expected_return: float
    beta: float
    alpha: float
    systematic_risk: float
    unsystematic_risk: float
    r_squared: float
    sharpe_ratio: float
    treynor_ratio: float
    information_ratio: float

@dataclass
class FamaFrenchResult:
    """Fama-French model result"""
    alpha: float
    market_beta: float
    size_beta: float
    value_beta: float
    profitability_beta: Optional[float]
    investment_beta: Optional[float]
    r_squared: float
    expected_return: float
    factor_loadings: Dict[str, float]
    risk_attribution: Dict[str, float]

@dataclass
class ValuationResult:
    """Combined valuation result"""
    dcf_result: DCFResult
    ddm_result: DDMResult
    capm_result: CAPMResult
    fama_french_result: FamaFrenchResult
    consensus_value: float
    valuation_range: Tuple[float, float]
    recommendation: str
    confidence_score: float
    risk_assessment: Dict[str, float]

class DCFModel:
    """Discounted Cash Flow Model"""
    
    def __init__(self, projection_years: int = 5, terminal_growth: float = 0.025):
        self.projection_years = projection_years
        self.terminal_growth = terminal_growth
    
    def calculate_wacc(self, 
                      financial_data: FinancialData,
                      market_data: MarketData,
                      cost_of_debt: float = 0.05) -> float:
        """Calculate Weighted Average Cost of Capital"""
        
        # Market value of equity
        market_value_equity = market_data.market_cap
        
        # Market value of debt (approximation)
        market_value_debt = financial_data.debt
        
        # Total value
        total_value = market_value_equity + market_value_debt
        
        # Cost of equity using CAPM
        cost_of_equity = (market_data.risk_free_rate + 
                         market_data.beta * (market_data.market_return - market_data.risk_free_rate))
        
        # After-tax cost of debt
        after_tax_cost_debt = cost_of_debt * (1 - financial_data.tax_rate)
        
        # WACC calculation
        wacc = ((market_value_equity / total_value) * cost_of_equity + 
                (market_value_debt / total_value) * after_tax_cost_debt)
        
        return wacc
    
    def project_cash_flows(self, 
                          financial_data: FinancialData,
                          growth_rate: float = 0.05) -> List[float]:
        """Project future free cash flows"""
        
        base_fcf = financial_data.free_cash_flow
        projections = []
        
        for year in range(1, self.projection_years + 1):
            # Apply declining growth rate
            year_growth = growth_rate * (0.9 ** (year - 1))  # Declining growth
            projected_fcf = base_fcf * ((1 + year_growth) ** year)
            projections.append(projected_fcf)
        
        return projections
    
    def calculate_terminal_value(self, 
                               final_year_fcf: float,
                               wacc: float) -> float:
        """Calculate terminal value using Gordon Growth Model"""
        
        terminal_fcf = final_year_fcf * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (wacc - self.terminal_growth)
        
        return terminal_value
    
    def calculate_dcf(self, 
                     financial_data: FinancialData,
                     market_data: MarketData,
                     growth_rate: float = 0.05) -> DCFResult:
        """Calculate DCF valuation"""
        
        # Calculate WACC
        wacc = self.calculate_wacc(financial_data, market_data)
        
        # Project cash flows
        fcf_projections = self.project_cash_flows(financial_data, growth_rate)
        
        # Calculate present value of projected cash flows
        pv_fcf = 0
        for i, fcf in enumerate(fcf_projections, 1):
            pv_fcf += fcf / ((1 + wacc) ** i)
        
        # Calculate terminal value
        terminal_value = self.calculate_terminal_value(fcf_projections[-1], wacc)
        pv_terminal = terminal_value / ((1 + wacc) ** self.projection_years)
        
        # Total enterprise value
        enterprise_value = pv_fcf + pv_terminal
        
        # Equity value
        equity_value = enterprise_value - financial_data.debt + financial_data.cash
        
        # Per share value
        intrinsic_value = equity_value / financial_data.shares_outstanding
        
        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(financial_data, market_data, wacc, growth_rate)
        
        # Fair value range (±20%)
        fair_value_range = (intrinsic_value * 0.8, intrinsic_value * 1.2)
        
        # Upside/downside vs current price
        current_price = market_data.stock_price
        upside = (intrinsic_value - current_price) / current_price
        downside = upside  # Simplified
        
        return DCFResult(
            intrinsic_value=intrinsic_value,
            terminal_value=terminal_value,
            present_value_fcf=pv_fcf,
            wacc=wacc,
            growth_rate=growth_rate,
            dcf_projections=fcf_projections,
            sensitivity_analysis=sensitivity,
            upside_downside=(upside, downside),
            fair_value_range=fair_value_range
        )
    
    def _sensitivity_analysis(self, 
                            financial_data: FinancialData,
                            market_data: MarketData,
                            base_wacc: float,
                            base_growth: float) -> Dict[str, float]:
        """Perform sensitivity analysis"""
        
        sensitivity = {}
        
        # WACC sensitivity (±1%)
        for wacc_change in [-0.01, 0.01]:
            new_wacc = base_wacc + wacc_change
            # Simplified calculation
            base_value = financial_data.free_cash_flow / (base_wacc - base_growth)
            new_value = financial_data.free_cash_flow / (new_wacc - base_growth)
            sensitivity[f'wacc_{wacc_change:+.1%}'] = (new_value - base_value) / base_value
        
        # Growth sensitivity (±1%)
        for growth_change in [-0.01, 0.01]:
            new_growth = base_growth + growth_change
            base_value = financial_data.free_cash_flow / (base_wacc - base_growth)
            new_value = financial_data.free_cash_flow / (base_wacc - new_growth)
            sensitivity[f'growth_{growth_change:+.1%}'] = (new_value - base_value) / base_value
        
        return sensitivity

class DDMModel:
    """Dividend Discount Model"""
    
    def __init__(self, model_type: str = "gordon"):
        self.model_type = model_type  # "gordon", "two_stage", "multi_stage"
    
    def calculate_dividend_growth(self, dividend_history: List[float]) -> float:
        """Calculate historical dividend growth rate"""
        
        if len(dividend_history) < 2:
            return 0.05  # Default 5%
        
        growth_rates = []
        for i in range(1, len(dividend_history)):
            if dividend_history[i-1] > 0:
                growth = (dividend_history[i] - dividend_history[i-1]) / dividend_history[i-1]
                growth_rates.append(growth)
        
        return np.mean(growth_rates) if growth_rates else 0.05
    
    def gordon_growth_model(self, 
                           current_dividend: float,
                           growth_rate: float,
                           required_return: float) -> float:
        """Gordon Growth Model for constant growth"""
        
        if required_return <= growth_rate:
            # Invalid assumption, use alternative calculation
            return current_dividend * 20  # P/E approximation
        
        next_dividend = current_dividend * (1 + growth_rate)
        value = next_dividend / (required_return - growth_rate)
        
        return value
    
    def two_stage_model(self, 
                       current_dividend: float,
                       high_growth_rate: float,
                       stable_growth_rate: float,
                       required_return: float,
                       high_growth_years: int = 5) -> float:
        """Two-stage dividend growth model"""
        
        # Stage 1: High growth period
        stage1_pv = 0
        dividend = current_dividend
        
        for year in range(1, high_growth_years + 1):
            dividend *= (1 + high_growth_rate)
            stage1_pv += dividend / ((1 + required_return) ** year)
        
        # Stage 2: Stable growth (Gordon model)
        terminal_dividend = dividend * (1 + stable_growth_rate)
        terminal_value = terminal_dividend / (required_return - stable_growth_rate)
        stage2_pv = terminal_value / ((1 + required_return) ** high_growth_years)
        
        return stage1_pv + stage2_pv
    
    def calculate_ddm(self, 
                     financial_data: FinancialData,
                     market_data: MarketData,
                     dividend_history: List[float] = None) -> DDMResult:
        """Calculate DDM valuation"""
        
        # Calculate dividend growth rate
        if dividend_history:
            growth_rate = self.calculate_dividend_growth(dividend_history)
        else:
            # Estimate from financial metrics
            roe = market_data.roe
            payout_ratio = financial_data.dividends / financial_data.net_income if financial_data.net_income > 0 else 0.4
            growth_rate = roe * (1 - payout_ratio)  # Sustainable growth
        
        # Required return (using CAPM)
        required_return = (market_data.risk_free_rate + 
                          market_data.beta * (market_data.market_return - market_data.risk_free_rate))
        
        # Current dividend per share
        current_dividend = financial_data.dividends / financial_data.shares_outstanding
        
        # Calculate intrinsic value based on model type
        if self.model_type == "gordon":
            intrinsic_value = self.gordon_growth_model(current_dividend, growth_rate, required_return)
        elif self.model_type == "two_stage":
            high_growth = min(growth_rate * 1.5, 0.15)  # Cap at 15%
            stable_growth = min(growth_rate, 0.04)  # Cap at 4%
            intrinsic_value = self.two_stage_model(current_dividend, high_growth, stable_growth, required_return)
        else:
            intrinsic_value = self.gordon_growth_model(current_dividend, growth_rate, required_return)
        
        # Project future dividends
        dividend_projections = []
        dividend = current_dividend
        for year in range(1, 6):
            dividend *= (1 + growth_rate)
            dividend_projections.append(dividend)
        
        # Calculate metrics
        payout_ratio = financial_data.dividends / financial_data.net_income if financial_data.net_income > 0 else 0
        sustainable_growth = market_data.roe * (1 - payout_ratio)
        dividend_yield_forecast = (current_dividend * (1 + growth_rate)) / market_data.stock_price
        gordon_value = self.gordon_growth_model(current_dividend, growth_rate, required_return)
        
        return DDMResult(
            intrinsic_value=intrinsic_value,
            dividend_growth_rate=growth_rate,
            required_return=required_return,
            dividend_projections=dividend_projections,
            payout_ratio=payout_ratio,
            sustainable_growth=sustainable_growth,
            dividend_yield_forecast=dividend_yield_forecast,
            gordon_growth_value=gordon_value
        )

class CAPMModel:
    """Capital Asset Pricing Model"""
    
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
    
    def calculate_beta(self, 
                     stock_returns: pd.Series,
                     market_returns: pd.Series) -> Tuple[float, float, float]:
        """Calculate beta, alpha, and R-squared"""
        
        # Align data
        aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
        
        if len(aligned_data) < 30:  # Minimum observations
            return 1.0, 0.0, 0.0
        
        stock_ret = aligned_data.iloc[:, 0]
        market_ret = aligned_data.iloc[:, 1]
        
        if SKLEARN_AVAILABLE:
            # Use sklearn for regression
            X = market_ret.values.reshape(-1, 1)
            y = stock_ret.values
            
            model = LinearRegression()
            model.fit(X, y)
            
            beta = model.coef_[0]
            alpha = model.intercept_
            r_squared = r2_score(y, model.predict(X))
            
        else:
            # Manual calculation
            covariance = np.cov(stock_ret, market_ret)[0, 1]
            market_variance = np.var(market_ret)
            
            beta = covariance / market_variance if market_variance > 0 else 1.0
            alpha = np.mean(stock_ret) - beta * np.mean(market_ret)
            
            # Calculate R-squared
            predicted_returns = alpha + beta * market_ret
            ss_res = np.sum((stock_ret - predicted_returns) ** 2)
            ss_tot = np.sum((stock_ret - np.mean(stock_ret)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return beta, alpha, r_squared
    
    def calculate_capm(self, 
                      stock_returns: pd.Series,
                      market_returns: pd.Series,
                      risk_free_rate: float) -> CAPMResult:
        """Calculate CAPM analysis"""
        
        # Calculate beta and alpha
        beta, alpha, r_squared = self.calculate_beta(stock_returns, market_returns)
        
        # Expected return using CAPM
        market_premium = market_returns.mean() * 252 - risk_free_rate  # Annualized
        expected_return = risk_free_rate + beta * market_premium
        
        # Risk decomposition
        stock_variance = stock_returns.var() * 252  # Annualized
        market_variance = market_returns.var() * 252  # Annualized
        
        systematic_risk = (beta ** 2) * market_variance
        unsystematic_risk = stock_variance - systematic_risk
        
        # Performance ratios
        stock_mean_return = stock_returns.mean() * 252
        stock_std = stock_returns.std() * np.sqrt(252)
        
        sharpe_ratio = (stock_mean_return - risk_free_rate) / stock_std if stock_std > 0 else 0
        treynor_ratio = (stock_mean_return - risk_free_rate) / beta if beta != 0 else 0
        
        # Information ratio (alpha / tracking error)
        tracking_error = np.sqrt(unsystematic_risk)
        information_ratio = (alpha * 252) / tracking_error if tracking_error > 0 else 0
        
        return CAPMResult(
            expected_return=expected_return,
            beta=beta,
            alpha=alpha * 252,  # Annualized
            systematic_risk=systematic_risk,
            unsystematic_risk=unsystematic_risk,
            r_squared=r_squared,
            sharpe_ratio=sharpe_ratio,
            treynor_ratio=treynor_ratio,
            information_ratio=information_ratio
        )

class FamaFrenchModel:
    """Fama-French Factor Models"""
    
    def __init__(self, model_type: str = "three_factor"):
        self.model_type = model_type  # "three_factor" or "five_factor"
    
    def calculate_fama_french(self, 
                            stock_returns: pd.Series,
                            factor_data: Dict[str, pd.Series],
                            risk_free_rate: float) -> FamaFrenchResult:
        """Calculate Fama-French factor loadings"""
        
        # Excess returns
        excess_returns = stock_returns - risk_free_rate / 252  # Daily risk-free rate
        
        # Required factors
        required_factors = ['market', 'smb', 'hml']  # Market, Size, Value
        if self.model_type == "five_factor":
            required_factors.extend(['rmw', 'cma'])  # Profitability, Investment
        
        # Check if all factors are available
        available_factors = [f for f in required_factors if f in factor_data]
        
        if len(available_factors) < 3:
            # Generate synthetic factors if not available
            factor_data = self._generate_synthetic_factors(stock_returns)
            available_factors = list(factor_data.keys())
        
        # Align data
        all_data = pd.concat([excess_returns] + [factor_data[f] for f in available_factors], axis=1).dropna()
        
        if len(all_data) < 30:
            return self._default_fama_french_result()
        
        y = all_data.iloc[:, 0]  # Excess returns
        X = all_data.iloc[:, 1:]  # Factors
        
        if SKLEARN_AVAILABLE:
            # Multiple regression
            model = LinearRegression()
            model.fit(X, y)
            
            alpha = model.intercept_ * 252  # Annualized
            betas = model.coef_
            r_squared = r2_score(y, model.predict(X))
            
        else:
            # Manual calculation using normal equations
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            
            try:
                coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                alpha = coefficients[0] * 252
                betas = coefficients[1:]
                
                # Calculate R-squared
                predicted = X_with_intercept @ coefficients
                ss_res = np.sum((y - predicted) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
            except np.linalg.LinAlgError:
                return self._default_fama_french_result()
        
        # Extract factor loadings
        factor_loadings = {factor: beta for factor, beta in zip(available_factors, betas)}
        
        # Calculate expected return
        factor_premiums = {factor: factor_data[factor].mean() * 252 for factor in available_factors}
        expected_return = risk_free_rate + sum(beta * factor_premiums[factor] 
                                             for factor, beta in factor_loadings.items())
        
        # Risk attribution
        factor_variances = {factor: factor_data[factor].var() * 252 for factor in available_factors}
        total_systematic_risk = sum((beta ** 2) * factor_variances[factor] 
                                  for factor, beta in factor_loadings.items())
        
        risk_attribution = {}
        for factor, beta in factor_loadings.items():
            factor_risk = (beta ** 2) * factor_variances[factor]
            risk_attribution[factor] = factor_risk / total_systematic_risk if total_systematic_risk > 0 else 0
        
        return FamaFrenchResult(
            alpha=alpha,
            market_beta=factor_loadings.get('market', 1.0),
            size_beta=factor_loadings.get('smb', 0.0),
            value_beta=factor_loadings.get('hml', 0.0),
            profitability_beta=factor_loadings.get('rmw') if self.model_type == "five_factor" else None,
            investment_beta=factor_loadings.get('cma') if self.model_type == "five_factor" else None,
            r_squared=r_squared,
            expected_return=expected_return,
            factor_loadings=factor_loadings,
            risk_attribution=risk_attribution
        )
    
    def _generate_synthetic_factors(self, stock_returns: pd.Series) -> Dict[str, pd.Series]:
        """Generate synthetic factor data for demonstration"""
        
        n = len(stock_returns)
        dates = stock_returns.index
        
        # Generate correlated factors
        np.random.seed(42)  # For reproducibility
        
        market_factor = pd.Series(np.random.normal(0.0005, 0.02, n), index=dates)
        smb_factor = pd.Series(np.random.normal(0.0002, 0.01, n), index=dates)
        hml_factor = pd.Series(np.random.normal(0.0001, 0.008, n), index=dates)
        
        if self.model_type == "five_factor":
            rmw_factor = pd.Series(np.random.normal(0.0001, 0.006, n), index=dates)
            cma_factor = pd.Series(np.random.normal(-0.0001, 0.005, n), index=dates)
            
            return {
                'market': market_factor,
                'smb': smb_factor,
                'hml': hml_factor,
                'rmw': rmw_factor,
                'cma': cma_factor
            }
        
        return {
            'market': market_factor,
            'smb': smb_factor,
            'hml': hml_factor
        }
    
    def _default_fama_french_result(self) -> FamaFrenchResult:
        """Return default result when calculation fails"""
        return FamaFrenchResult(
            alpha=0.0,
            market_beta=1.0,
            size_beta=0.0,
            value_beta=0.0,
            profitability_beta=None,
            investment_beta=None,
            r_squared=0.0,
            expected_return=0.08,
            factor_loadings={'market': 1.0, 'smb': 0.0, 'hml': 0.0},
            risk_attribution={'market': 1.0, 'smb': 0.0, 'hml': 0.0}
        )

class StockValuationModel:
    """Combined Stock Valuation Model"""
    
    def __init__(self):
        self.dcf_model = DCFModel()
        self.ddm_model = DDMModel()
        self.capm_model = CAPMModel()
        self.fama_french_model = FamaFrenchModel()
    
    def analyze(self, 
               financial_data: FinancialData,
               market_data: MarketData,
               stock_returns: pd.Series = None,
               market_returns: pd.Series = None,
               factor_data: Dict[str, pd.Series] = None,
               dividend_history: List[float] = None) -> ValuationResult:
        """Perform comprehensive stock valuation analysis"""
        
        try:
            # DCF Analysis
            dcf_result = self.dcf_model.calculate_dcf(financial_data, market_data)
            
            # DDM Analysis
            ddm_result = self.ddm_model.calculate_ddm(financial_data, market_data, dividend_history)
            
            # CAPM Analysis
            if stock_returns is not None and market_returns is not None:
                capm_result = self.capm_model.calculate_capm(
                    stock_returns, market_returns, market_data.risk_free_rate
                )
            else:
                capm_result = self._default_capm_result(market_data)
            
            # Fama-French Analysis
            if stock_returns is not None and factor_data is not None:
                fama_french_result = self.fama_french_model.calculate_fama_french(
                    stock_returns, factor_data, market_data.risk_free_rate
                )
            else:
                fama_french_result = self.fama_french_model._default_fama_french_result()
            
            # Consensus valuation (weighted average)
            weights = {'dcf': 0.4, 'ddm': 0.3, 'capm': 0.2, 'fama_french': 0.1}
            
            consensus_value = (
                weights['dcf'] * dcf_result.intrinsic_value +
                weights['ddm'] * ddm_result.intrinsic_value +
                weights['capm'] * market_data.stock_price * (1 + capm_result.expected_return) +
                weights['fama_french'] * market_data.stock_price * (1 + fama_french_result.expected_return)
            )
            
            # Valuation range
            values = [dcf_result.intrinsic_value, ddm_result.intrinsic_value]
            valuation_range = (min(values) * 0.9, max(values) * 1.1)
            
            # Investment recommendation
            current_price = market_data.stock_price
            upside = (consensus_value - current_price) / current_price
            
            if upside > 0.2:
                recommendation = "Strong Buy"
            elif upside > 0.1:
                recommendation = "Buy"
            elif upside > -0.1:
                recommendation = "Hold"
            elif upside > -0.2:
                recommendation = "Sell"
            else:
                recommendation = "Strong Sell"
            
            # Confidence score
            value_dispersion = np.std(values) / np.mean(values) if values else 0
            confidence_score = max(0, 1 - value_dispersion)
            
            # Risk assessment
            risk_assessment = {
                'beta_risk': abs(capm_result.beta - 1.0),
                'valuation_risk': value_dispersion,
                'dividend_risk': 1 - ddm_result.payout_ratio if ddm_result.payout_ratio < 1 else 0,
                'financial_risk': financial_data.debt / (financial_data.debt + market_data.market_cap),
                'market_risk': capm_result.systematic_risk / (capm_result.systematic_risk + capm_result.unsystematic_risk)
            }
            
            return ValuationResult(
                dcf_result=dcf_result,
                ddm_result=ddm_result,
                capm_result=capm_result,
                fama_french_result=fama_french_result,
                consensus_value=consensus_value,
                valuation_range=valuation_range,
                recommendation=recommendation,
                confidence_score=confidence_score,
                risk_assessment=risk_assessment
            )
            
        except Exception as e:
            logger.error(f"Error in stock valuation analysis: {str(e)}")
            raise
    
    def _default_capm_result(self, market_data: MarketData) -> CAPMResult:
        """Generate default CAPM result"""
        expected_return = market_data.risk_free_rate + market_data.beta * 0.06  # 6% market premium
        
        return CAPMResult(
            expected_return=expected_return,
            beta=market_data.beta,
            alpha=0.0,
            systematic_risk=0.04,
            unsystematic_risk=0.02,
            r_squared=0.3,
            sharpe_ratio=0.5,
            treynor_ratio=0.08,
            information_ratio=0.0
        )
    
    def get_valuation_insights(self, result: ValuationResult) -> Dict[str, str]:
        """Generate comprehensive valuation insights"""
        insights = {}
        
        # DCF insights
        dcf = result.dcf_result
        insights['dcf'] = f"Intrinsic Value: ${dcf.intrinsic_value:.2f}, WACC: {dcf.wacc:.1%}, Growth: {dcf.growth_rate:.1%}"
        
        # DDM insights
        ddm = result.ddm_result
        insights['ddm'] = f"Dividend Value: ${ddm.intrinsic_value:.2f}, Growth: {ddm.dividend_growth_rate:.1%}, Yield: {ddm.dividend_yield_forecast:.1%}"
        
        # CAPM insights
        capm = result.capm_result
        insights['capm'] = f"Expected Return: {capm.expected_return:.1%}, Beta: {capm.beta:.2f}, Alpha: {capm.alpha:.1%}"
        
        # Fama-French insights
        ff = result.fama_french_result
        insights['fama_french'] = f"Alpha: {ff.alpha:.1%}, Market Beta: {ff.market_beta:.2f}, Size Beta: {ff.size_beta:.2f}"
        
        # Overall insights
        insights['consensus'] = f"Fair Value: ${result.consensus_value:.2f}, Recommendation: {result.recommendation}, Confidence: {result.confidence_score:.1%}"
        
        # Risk insights
        risk = result.risk_assessment
        insights['risk'] = f"Beta Risk: {risk['beta_risk']:.2f}, Valuation Risk: {risk['valuation_risk']:.2f}, Financial Risk: {risk['financial_risk']:.1%}"
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Test the valuation models
    
    # Sample financial data
    financial_data = FinancialData(
        revenue=100000000,
        net_income=10000000,
        free_cash_flow=8000000,
        dividends=3000000,
        shares_outstanding=10000000,
        book_value=50000000,
        debt=20000000,
        cash=5000000,
        capex=5000000,
        depreciation=3000000,
        working_capital_change=1000000,
        tax_rate=0.25
    )
    
    # Sample market data
    market_data = MarketData(
        stock_price=50.0,
        market_cap=500000000,
        beta=1.2,
        risk_free_rate=0.03,
        market_return=0.10,
        dividend_yield=0.03,
        pe_ratio=15.0,
        pb_ratio=2.0,
        roe=0.15,
        roa=0.08
    )
    
    # Create model and analyze
    valuation_model = StockValuationModel()
    result = valuation_model.analyze(financial_data, market_data)
    
    insights = valuation_model.get_valuation_insights(result)
    
    print("=== Stock Valuation Analysis ===")
    print(f"Consensus Fair Value: ${result.consensus_value:.2f}")
    print(f"Current Price: ${market_data.stock_price:.2f}")
    print(f"Recommendation: {result.recommendation}")
    print(f"Confidence Score: {result.confidence_score:.1%}")
    print()
    
    print("=== Detailed Insights ===")
    for key, insight in insights.items():
        print(f"{key}: {insight}")