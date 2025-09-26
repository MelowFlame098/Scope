"""Gordon Growth Model for Dividend-Paying Stocks

This module implements comprehensive dividend analysis models:
- Gordon Growth Model (Constant Growth DDM)
- Multi-Stage Dividend Growth Models
- Dividend Sustainability Analysis
- Dividend Policy Evaluation
- Dividend Aristocrat Analysis
- Dividend Yield Strategies
- Payout Ratio Optimization
- Dividend Coverage Analysis
- Dividend Growth Forecasting
- Dividend Risk Assessment
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
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.metrics import r2_score, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Using simplified implementations.")

logger = logging.getLogger(__name__)

@dataclass
class DividendData:
    """Dividend data structure"""
    dividend_per_share: float
    dividend_yield: float
    payout_ratio: float
    dividend_coverage: float
    free_cash_flow_per_share: float
    earnings_per_share: float
    book_value_per_share: float
    return_on_equity: float
    debt_to_equity: float
    current_ratio: float

@dataclass
class DividendHistory:
    """Historical dividend data"""
    dates: List[datetime]
    dividends: List[float]
    special_dividends: List[float]
    dividend_frequency: str  # 'quarterly', 'semi-annual', 'annual'
    ex_dividend_dates: List[datetime]
    payment_dates: List[datetime]
    years_of_growth: int
    consecutive_increases: int

@dataclass
class GordonGrowthResult:
    """Gordon Growth Model result"""
    intrinsic_value: float
    dividend_growth_rate: float
    required_return: float
    terminal_value: float
    present_value_growth_phase: float
    fair_value_range: Tuple[float, float]
    sensitivity_analysis: Dict[str, float]
    margin_of_safety: float
    dividend_yield_on_cost: float

@dataclass
class MultiStageResult:
    """Multi-stage dividend growth result"""
    intrinsic_value: float
    stage1_value: float
    stage2_value: float
    stage3_value: Optional[float]
    growth_rates: List[float]
    stage_durations: List[int]
    transition_points: List[float]
    npv_by_stage: List[float]

@dataclass
class DividendSustainabilityResult:
    """Dividend sustainability analysis"""
    sustainability_score: float
    payout_sustainability: float
    earnings_stability: float
    cash_flow_coverage: float
    debt_service_capacity: float
    dividend_cut_probability: float
    sustainable_payout_ratio: float
    years_of_coverage: float
    stress_test_results: Dict[str, float]

@dataclass
class DividendAristocratResult:
    """Dividend Aristocrat analysis"""
    aristocrat_status: bool
    years_of_increases: int
    average_growth_rate: float
    growth_consistency: float
    dividend_cagr: float
    yield_on_original_cost: float
    total_return_contribution: float
    aristocrat_score: float
    peer_comparison: Dict[str, float]

@dataclass
class DividendStrategyResult:
    """Dividend investment strategy result"""
    strategy_type: str
    target_yield: float
    growth_potential: float
    risk_level: float
    income_stability: float
    capital_appreciation: float
    total_return_forecast: float
    strategy_score: float
    recommended_allocation: float

@dataclass
class DividendAnalysisResult:
    """Combined dividend analysis result"""
    gordon_growth: GordonGrowthResult
    multi_stage: MultiStageResult
    sustainability: DividendSustainabilityResult
    aristocrat_analysis: DividendAristocratResult
    strategy_recommendation: DividendStrategyResult
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    model_confidence: float

class GordonGrowthModel:
    """Gordon Growth Model Implementation"""
    
    def __init__(self, risk_free_rate: float = 0.03, market_risk_premium: float = 0.06):
        self.risk_free_rate = risk_free_rate
        self.market_risk_premium = market_risk_premium
    
    def calculate_required_return(self, beta: float, company_risk_premium: float = 0.0) -> float:
        """Calculate required return using CAPM with company-specific adjustments"""
        return self.risk_free_rate + beta * self.market_risk_premium + company_risk_premium
    
    def estimate_growth_rate(self, dividend_history: DividendHistory, method: str = "historical") -> float:
        """Estimate dividend growth rate using various methods"""
        
        dividends = np.array(dividend_history.dividends)
        
        if len(dividends) < 2:
            return 0.03  # Default 3% growth
        
        if method == "historical":
            # Simple historical average
            growth_rates = []
            for i in range(1, len(dividends)):
                if dividends[i-1] > 0:
                    growth = (dividends[i] - dividends[i-1]) / dividends[i-1]
                    growth_rates.append(growth)
            return np.mean(growth_rates) if growth_rates else 0.03
        
        elif method == "cagr":
            # Compound Annual Growth Rate
            if dividends[0] > 0 and len(dividends) > 1:
                years = len(dividends) - 1
                cagr = (dividends[-1] / dividends[0]) ** (1/years) - 1
                return cagr
            return 0.03
        
        elif method == "regression":
            # Linear regression on log dividends
            if SKLEARN_AVAILABLE and len(dividends) >= 3:
                log_dividends = np.log(dividends[dividends > 0])
                if len(log_dividends) >= 3:
                    X = np.arange(len(log_dividends)).reshape(-1, 1)
                    model = LinearRegression()
                    model.fit(X, log_dividends)
                    return model.coef_[0]  # Growth rate from log-linear trend
            return self.estimate_growth_rate(dividend_history, "cagr")
        
        elif method == "sustainable":
            # Sustainable growth rate (ROE × Retention Ratio)
            # This requires additional financial data
            return 0.05  # Placeholder
        
        else:
            return self.estimate_growth_rate(dividend_history, "historical")
    
    def calculate_gordon_growth(self, 
                              dividend_data: DividendData,
                              dividend_history: DividendHistory,
                              beta: float = 1.0,
                              growth_method: str = "historical") -> GordonGrowthResult:
        """Calculate Gordon Growth Model valuation"""
        
        # Estimate growth rate
        growth_rate = self.estimate_growth_rate(dividend_history, growth_method)
        
        # Calculate required return
        required_return = self.calculate_required_return(beta)
        
        # Ensure growth rate is less than required return
        if growth_rate >= required_return:
            growth_rate = required_return * 0.8  # Cap at 80% of required return
        
        # Current dividend
        current_dividend = dividend_data.dividend_per_share
        
        # Next year's expected dividend
        next_dividend = current_dividend * (1 + growth_rate)
        
        # Gordon Growth Model calculation
        if required_return > growth_rate:
            intrinsic_value = next_dividend / (required_return - growth_rate)
        else:
            # Fallback to P/E based valuation
            intrinsic_value = dividend_data.earnings_per_share * 15  # 15x P/E
        
        # Terminal value (same as intrinsic value in Gordon model)
        terminal_value = intrinsic_value
        
        # Present value of growth phase (infinite in Gordon model)
        present_value_growth_phase = intrinsic_value
        
        # Fair value range (±20%)
        fair_value_range = (intrinsic_value * 0.8, intrinsic_value * 1.2)
        
        # Sensitivity analysis
        sensitivity = self._sensitivity_analysis(
            current_dividend, growth_rate, required_return
        )
        
        # Margin of safety (assuming current price is dividend yield based)
        current_price = current_dividend / dividend_data.dividend_yield if dividend_data.dividend_yield > 0 else intrinsic_value
        margin_of_safety = (intrinsic_value - current_price) / intrinsic_value if intrinsic_value > 0 else 0
        
        # Dividend yield on cost (future yield based on current price)
        dividend_yield_on_cost = next_dividend / current_price if current_price > 0 else dividend_data.dividend_yield
        
        return GordonGrowthResult(
            intrinsic_value=intrinsic_value,
            dividend_growth_rate=growth_rate,
            required_return=required_return,
            terminal_value=terminal_value,
            present_value_growth_phase=present_value_growth_phase,
            fair_value_range=fair_value_range,
            sensitivity_analysis=sensitivity,
            margin_of_safety=margin_of_safety,
            dividend_yield_on_cost=dividend_yield_on_cost
        )
    
    def _sensitivity_analysis(self, 
                            current_dividend: float,
                            base_growth: float,
                            base_required_return: float) -> Dict[str, float]:
        """Perform sensitivity analysis on key parameters"""
        
        sensitivity = {}
        base_value = current_dividend * (1 + base_growth) / (base_required_return - base_growth)
        
        # Growth rate sensitivity (±1%)
        for delta in [-0.01, 0.01]:
            new_growth = base_growth + delta
            if new_growth < base_required_return:
                new_value = current_dividend * (1 + new_growth) / (base_required_return - new_growth)
                sensitivity[f'growth_{delta:+.1%}'] = (new_value - base_value) / base_value
        
        # Required return sensitivity (±1%)
        for delta in [-0.01, 0.01]:
            new_required_return = base_required_return + delta
            if new_required_return > base_growth:
                new_value = current_dividend * (1 + base_growth) / (new_required_return - base_growth)
                sensitivity[f'required_return_{delta:+.1%}'] = (new_value - base_value) / base_value
        
        return sensitivity

class MultiStageDividendModel:
    """Multi-Stage Dividend Growth Model"""
    
    def __init__(self, risk_free_rate: float = 0.03):
        self.risk_free_rate = risk_free_rate
    
    def calculate_two_stage(self, 
                           dividend_data: DividendData,
                           high_growth_rate: float,
                           stable_growth_rate: float,
                           required_return: float,
                           high_growth_years: int = 5) -> MultiStageResult:
        """Two-stage dividend growth model"""
        
        current_dividend = dividend_data.dividend_per_share
        
        # Stage 1: High growth period
        stage1_value = 0
        dividend = current_dividend
        
        for year in range(1, high_growth_years + 1):
            dividend *= (1 + high_growth_rate)
            pv_dividend = dividend / ((1 + required_return) ** year)
            stage1_value += pv_dividend
        
        # Stage 2: Stable growth (Gordon model)
        terminal_dividend = dividend * (1 + stable_growth_rate)
        if required_return > stable_growth_rate:
            terminal_value = terminal_dividend / (required_return - stable_growth_rate)
            stage2_value = terminal_value / ((1 + required_return) ** high_growth_years)
        else:
            stage2_value = dividend * 20  # Fallback valuation
        
        intrinsic_value = stage1_value + stage2_value
        
        return MultiStageResult(
            intrinsic_value=intrinsic_value,
            stage1_value=stage1_value,
            stage2_value=stage2_value,
            stage3_value=None,
            growth_rates=[high_growth_rate, stable_growth_rate],
            stage_durations=[high_growth_years, float('inf')],
            transition_points=[high_growth_years],
            npv_by_stage=[stage1_value, stage2_value]
        )
    
    def calculate_three_stage(self, 
                             dividend_data: DividendData,
                             growth_rates: List[float],
                             stage_durations: List[int],
                             required_return: float) -> MultiStageResult:
        """Three-stage dividend growth model"""
        
        if len(growth_rates) != 3 or len(stage_durations) != 2:
            raise ValueError("Three-stage model requires 3 growth rates and 2 finite stage durations")
        
        current_dividend = dividend_data.dividend_per_share
        stage_values = []
        dividend = current_dividend
        year = 0
        
        # Stage 1: High growth
        stage1_value = 0
        for _ in range(stage_durations[0]):
            year += 1
            dividend *= (1 + growth_rates[0])
            pv_dividend = dividend / ((1 + required_return) ** year)
            stage1_value += pv_dividend
        stage_values.append(stage1_value)
        
        # Stage 2: Declining growth
        stage2_value = 0
        for _ in range(stage_durations[1]):
            year += 1
            dividend *= (1 + growth_rates[1])
            pv_dividend = dividend / ((1 + required_return) ** year)
            stage2_value += pv_dividend
        stage_values.append(stage2_value)
        
        # Stage 3: Stable growth (terminal value)
        terminal_dividend = dividend * (1 + growth_rates[2])
        if required_return > growth_rates[2]:
            terminal_value = terminal_dividend / (required_return - growth_rates[2])
            stage3_value = terminal_value / ((1 + required_return) ** year)
        else:
            stage3_value = dividend * 15  # Fallback
        stage_values.append(stage3_value)
        
        intrinsic_value = sum(stage_values)
        
        return MultiStageResult(
            intrinsic_value=intrinsic_value,
            stage1_value=stage1_value,
            stage2_value=stage2_value,
            stage3_value=stage3_value,
            growth_rates=growth_rates,
            stage_durations=stage_durations + [float('inf')],
            transition_points=[stage_durations[0], sum(stage_durations)],
            npv_by_stage=stage_values
        )

class DividendSustainabilityAnalyzer:
    """Analyze dividend sustainability"""
    
    def __init__(self):
        self.weights = {
            'payout_ratio': 0.25,
            'earnings_stability': 0.20,
            'cash_flow_coverage': 0.25,
            'debt_capacity': 0.15,
            'business_quality': 0.15
        }
    
    def analyze_sustainability(self, 
                             dividend_data: DividendData,
                             dividend_history: DividendHistory,
                             financial_metrics: Dict[str, float] = None) -> DividendSustainabilityResult:
        """Analyze dividend sustainability"""
        
        # Payout ratio sustainability
        payout_sustainability = self._analyze_payout_ratio(dividend_data)
        
        # Earnings stability
        earnings_stability = self._analyze_earnings_stability(dividend_history)
        
        # Cash flow coverage
        cash_flow_coverage = self._analyze_cash_flow_coverage(dividend_data)
        
        # Debt service capacity
        debt_service_capacity = self._analyze_debt_capacity(dividend_data)
        
        # Overall sustainability score
        sustainability_score = (
            self.weights['payout_ratio'] * payout_sustainability +
            self.weights['earnings_stability'] * earnings_stability +
            self.weights['cash_flow_coverage'] * cash_flow_coverage +
            self.weights['debt_capacity'] * debt_service_capacity +
            self.weights['business_quality'] * 0.7  # Default business quality
        )
        
        # Dividend cut probability
        dividend_cut_probability = max(0, 1 - sustainability_score)
        
        # Sustainable payout ratio
        sustainable_payout_ratio = min(0.6, dividend_data.payout_ratio * sustainability_score)
        
        # Years of coverage
        if dividend_data.free_cash_flow_per_share > 0:
            years_of_coverage = dividend_data.free_cash_flow_per_share / dividend_data.dividend_per_share
        else:
            years_of_coverage = 0
        
        # Stress test
        stress_test_results = self._stress_test_dividends(dividend_data)
        
        return DividendSustainabilityResult(
            sustainability_score=sustainability_score,
            payout_sustainability=payout_sustainability,
            earnings_stability=earnings_stability,
            cash_flow_coverage=cash_flow_coverage,
            debt_service_capacity=debt_service_capacity,
            dividend_cut_probability=dividend_cut_probability,
            sustainable_payout_ratio=sustainable_payout_ratio,
            years_of_coverage=years_of_coverage,
            stress_test_results=stress_test_results
        )
    
    def _analyze_payout_ratio(self, dividend_data: DividendData) -> float:
        """Analyze payout ratio sustainability"""
        payout = dividend_data.payout_ratio
        
        if payout <= 0.4:
            return 1.0  # Very sustainable
        elif payout <= 0.6:
            return 0.8  # Sustainable
        elif payout <= 0.8:
            return 0.6  # Moderate
        elif payout <= 1.0:
            return 0.3  # Risky
        else:
            return 0.1  # Unsustainable
    
    def _analyze_earnings_stability(self, dividend_history: DividendHistory) -> float:
        """Analyze earnings stability based on dividend history"""
        dividends = np.array(dividend_history.dividends)
        
        if len(dividends) < 3:
            return 0.5  # Insufficient data
        
        # Calculate coefficient of variation
        cv = np.std(dividends) / np.mean(dividends) if np.mean(dividends) > 0 else 1
        
        # Convert to stability score (lower CV = higher stability)
        stability = max(0, 1 - cv)
        
        # Bonus for consecutive increases
        consecutive_bonus = min(0.2, dividend_history.consecutive_increases * 0.02)
        
        return min(1.0, stability + consecutive_bonus)
    
    def _analyze_cash_flow_coverage(self, dividend_data: DividendData) -> float:
        """Analyze free cash flow coverage"""
        coverage = dividend_data.dividend_coverage
        
        if coverage >= 3.0:
            return 1.0  # Excellent coverage
        elif coverage >= 2.0:
            return 0.8  # Good coverage
        elif coverage >= 1.5:
            return 0.6  # Adequate coverage
        elif coverage >= 1.0:
            return 0.3  # Weak coverage
        else:
            return 0.1  # Insufficient coverage
    
    def _analyze_debt_capacity(self, dividend_data: DividendData) -> float:
        """Analyze debt service capacity"""
        debt_to_equity = dividend_data.debt_to_equity
        current_ratio = dividend_data.current_ratio
        
        # Debt level score
        if debt_to_equity <= 0.3:
            debt_score = 1.0
        elif debt_to_equity <= 0.6:
            debt_score = 0.7
        elif debt_to_equity <= 1.0:
            debt_score = 0.4
        else:
            debt_score = 0.2
        
        # Liquidity score
        if current_ratio >= 2.0:
            liquidity_score = 1.0
        elif current_ratio >= 1.5:
            liquidity_score = 0.8
        elif current_ratio >= 1.0:
            liquidity_score = 0.5
        else:
            liquidity_score = 0.2
        
        return (debt_score + liquidity_score) / 2
    
    def _stress_test_dividends(self, dividend_data: DividendData) -> Dict[str, float]:
        """Stress test dividend sustainability under various scenarios"""
        
        base_coverage = dividend_data.dividend_coverage
        
        scenarios = {
            'recession_mild': 0.8,  # 20% earnings decline
            'recession_severe': 0.6,  # 40% earnings decline
            'interest_rate_rise': 0.9,  # 10% impact from higher rates
            'sector_downturn': 0.7,  # 30% sector-specific decline
            'credit_crunch': 0.5  # 50% impact from credit issues
        }
        
        stress_results = {}
        for scenario, impact_factor in scenarios.items():
            stressed_coverage = base_coverage * impact_factor
            sustainability = min(1.0, stressed_coverage)
            stress_results[scenario] = sustainability
        
        return stress_results

class DividendAristocratAnalyzer:
    """Analyze Dividend Aristocrat characteristics"""
    
    def __init__(self, min_years_for_aristocrat: int = 25):
        self.min_years_for_aristocrat = min_years_for_aristocrat
    
    def analyze_aristocrat_status(self, 
                                dividend_history: DividendHistory,
                                dividend_data: DividendData) -> DividendAristocratResult:
        """Analyze dividend aristocrat characteristics"""
        
        # Check aristocrat status
        years_of_increases = dividend_history.consecutive_increases
        aristocrat_status = years_of_increases >= self.min_years_for_aristocrat
        
        # Calculate average growth rate
        dividends = np.array(dividend_history.dividends)
        if len(dividends) >= 2:
            growth_rates = []
            for i in range(1, len(dividends)):
                if dividends[i-1] > 0:
                    growth = (dividends[i] - dividends[i-1]) / dividends[i-1]
                    growth_rates.append(growth)
            average_growth_rate = np.mean(growth_rates) if growth_rates else 0
        else:
            average_growth_rate = 0
        
        # Growth consistency (lower standard deviation = higher consistency)
        if len(growth_rates) > 1:
            growth_consistency = max(0, 1 - np.std(growth_rates))
        else:
            growth_consistency = 0.5
        
        # Dividend CAGR
        if len(dividends) >= 2 and dividends[0] > 0:
            years = len(dividends) - 1
            dividend_cagr = (dividends[-1] / dividends[0]) ** (1/years) - 1
        else:
            dividend_cagr = average_growth_rate
        
        # Yield on original cost (hypothetical)
        if len(dividends) >= 2:
            yield_on_original_cost = dividends[-1] / dividends[0] * dividend_data.dividend_yield
        else:
            yield_on_original_cost = dividend_data.dividend_yield
        
        # Total return contribution from dividends
        total_return_contribution = dividend_data.dividend_yield + average_growth_rate
        
        # Aristocrat score
        aristocrat_score = (
            0.3 * min(1.0, years_of_increases / self.min_years_for_aristocrat) +
            0.2 * min(1.0, average_growth_rate / 0.1) +  # Normalize to 10% growth
            0.2 * growth_consistency +
            0.15 * min(1.0, dividend_data.dividend_yield / 0.04) +  # Normalize to 4% yield
            0.15 * min(1.0, dividend_data.dividend_coverage / 2.0)  # Normalize to 2x coverage
        )
        
        # Peer comparison (simplified)
        peer_comparison = {
            'growth_vs_peers': 0.0,  # Would need peer data
            'yield_vs_peers': 0.0,
            'consistency_vs_peers': 0.0,
            'coverage_vs_peers': 0.0
        }
        
        return DividendAristocratResult(
            aristocrat_status=aristocrat_status,
            years_of_increases=years_of_increases,
            average_growth_rate=average_growth_rate,
            growth_consistency=growth_consistency,
            dividend_cagr=dividend_cagr,
            yield_on_original_cost=yield_on_original_cost,
            total_return_contribution=total_return_contribution,
            aristocrat_score=aristocrat_score,
            peer_comparison=peer_comparison
        )

class DividendStrategyAnalyzer:
    """Analyze dividend investment strategies"""
    
    def __init__(self):
        self.strategies = {
            'high_yield': {'min_yield': 0.04, 'max_payout': 0.8, 'min_coverage': 1.2},
            'dividend_growth': {'min_growth': 0.05, 'max_payout': 0.6, 'min_years': 10},
            'aristocrat': {'min_years': 25, 'min_yield': 0.02, 'min_coverage': 1.5},
            'balanced': {'min_yield': 0.025, 'min_growth': 0.03, 'max_payout': 0.7}
        }
    
    def analyze_strategy_fit(self, 
                           dividend_data: DividendData,
                           dividend_history: DividendHistory,
                           aristocrat_result: DividendAristocratResult) -> DividendStrategyResult:
        """Analyze which dividend strategy fits best"""
        
        strategy_scores = {}
        
        # High Yield Strategy
        high_yield_score = self._score_high_yield_strategy(dividend_data)
        strategy_scores['high_yield'] = high_yield_score
        
        # Dividend Growth Strategy
        growth_score = self._score_growth_strategy(dividend_data, aristocrat_result)
        strategy_scores['dividend_growth'] = growth_score
        
        # Aristocrat Strategy
        aristocrat_score = self._score_aristocrat_strategy(dividend_data, aristocrat_result)
        strategy_scores['aristocrat'] = aristocrat_score
        
        # Balanced Strategy
        balanced_score = self._score_balanced_strategy(dividend_data, aristocrat_result)
        strategy_scores['balanced'] = balanced_score
        
        # Select best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_strategy]
        
        # Calculate strategy metrics
        target_yield = dividend_data.dividend_yield
        growth_potential = aristocrat_result.average_growth_rate
        risk_level = 1 - dividend_data.dividend_coverage / 3.0  # Higher coverage = lower risk
        income_stability = aristocrat_result.growth_consistency
        capital_appreciation = growth_potential * 0.5  # Simplified
        total_return_forecast = target_yield + growth_potential
        recommended_allocation = best_score  # Use score as allocation weight
        
        return DividendStrategyResult(
            strategy_type=best_strategy,
            target_yield=target_yield,
            growth_potential=growth_potential,
            risk_level=max(0, min(1, risk_level)),
            income_stability=income_stability,
            capital_appreciation=capital_appreciation,
            total_return_forecast=total_return_forecast,
            strategy_score=best_score,
            recommended_allocation=recommended_allocation
        )
    
    def _score_high_yield_strategy(self, dividend_data: DividendData) -> float:
        """Score fit for high yield strategy"""
        criteria = self.strategies['high_yield']
        
        yield_score = min(1.0, dividend_data.dividend_yield / criteria['min_yield'])
        payout_score = 1.0 if dividend_data.payout_ratio <= criteria['max_payout'] else 0.5
        coverage_score = min(1.0, dividend_data.dividend_coverage / criteria['min_coverage'])
        
        return (yield_score + payout_score + coverage_score) / 3
    
    def _score_growth_strategy(self, dividend_data: DividendData, aristocrat_result: DividendAristocratResult) -> float:
        """Score fit for dividend growth strategy"""
        criteria = self.strategies['dividend_growth']
        
        growth_score = min(1.0, aristocrat_result.average_growth_rate / criteria['min_growth'])
        payout_score = 1.0 if dividend_data.payout_ratio <= criteria['max_payout'] else 0.5
        years_score = min(1.0, aristocrat_result.years_of_increases / criteria['min_years'])
        
        return (growth_score + payout_score + years_score) / 3
    
    def _score_aristocrat_strategy(self, dividend_data: DividendData, aristocrat_result: DividendAristocratResult) -> float:
        """Score fit for aristocrat strategy"""
        criteria = self.strategies['aristocrat']
        
        years_score = 1.0 if aristocrat_result.years_of_increases >= criteria['min_years'] else 0.3
        yield_score = min(1.0, dividend_data.dividend_yield / criteria['min_yield'])
        coverage_score = min(1.0, dividend_data.dividend_coverage / criteria['min_coverage'])
        
        return (years_score + yield_score + coverage_score) / 3
    
    def _score_balanced_strategy(self, dividend_data: DividendData, aristocrat_result: DividendAristocratResult) -> float:
        """Score fit for balanced strategy"""
        criteria = self.strategies['balanced']
        
        yield_score = min(1.0, dividend_data.dividend_yield / criteria['min_yield'])
        growth_score = min(1.0, aristocrat_result.average_growth_rate / criteria['min_growth'])
        payout_score = 1.0 if dividend_data.payout_ratio <= criteria['max_payout'] else 0.7
        
        return (yield_score + growth_score + payout_score) / 3

class GordonGrowthAnalyzer:
    """Combined Gordon Growth and Dividend Analysis"""
    
    def __init__(self):
        self.gordon_model = GordonGrowthModel()
        self.multi_stage_model = MultiStageDividendModel()
        self.sustainability_analyzer = DividendSustainabilityAnalyzer()
        self.aristocrat_analyzer = DividendAristocratAnalyzer()
        self.strategy_analyzer = DividendStrategyAnalyzer()
    
    def analyze(self, 
               dividend_data: DividendData,
               dividend_history: DividendHistory,
               beta: float = 1.0,
               financial_metrics: Dict[str, float] = None) -> DividendAnalysisResult:
        """Perform comprehensive dividend analysis"""
        
        try:
            # Gordon Growth Analysis
            gordon_result = self.gordon_model.calculate_gordon_growth(
                dividend_data, dividend_history, beta
            )
            
            # Multi-Stage Analysis
            high_growth = min(0.15, gordon_result.dividend_growth_rate * 1.5)
            stable_growth = min(0.04, gordon_result.dividend_growth_rate * 0.8)
            
            multi_stage_result = self.multi_stage_model.calculate_two_stage(
                dividend_data, high_growth, stable_growth, gordon_result.required_return
            )
            
            # Sustainability Analysis
            sustainability_result = self.sustainability_analyzer.analyze_sustainability(
                dividend_data, dividend_history, financial_metrics
            )
            
            # Aristocrat Analysis
            aristocrat_result = self.aristocrat_analyzer.analyze_aristocrat_status(
                dividend_history, dividend_data
            )
            
            # Strategy Analysis
            strategy_result = self.strategy_analyzer.analyze_strategy_fit(
                dividend_data, dividend_history, aristocrat_result
            )
            
            # Risk Metrics
            risk_metrics = {
                'dividend_cut_risk': sustainability_result.dividend_cut_probability,
                'payout_risk': max(0, dividend_data.payout_ratio - 0.6),
                'coverage_risk': max(0, 2.0 - dividend_data.dividend_coverage) / 2.0,
                'growth_risk': max(0, 0.15 - aristocrat_result.average_growth_rate) / 0.15,
                'yield_risk': max(0, dividend_data.dividend_yield - 0.08) / 0.08
            }
            
            # Performance Metrics
            performance_metrics = {
                'total_return': dividend_data.dividend_yield + aristocrat_result.average_growth_rate,
                'income_component': dividend_data.dividend_yield,
                'growth_component': aristocrat_result.average_growth_rate,
                'risk_adjusted_return': (dividend_data.dividend_yield + aristocrat_result.average_growth_rate) / (1 + sum(risk_metrics.values())),
                'dividend_cagr': aristocrat_result.dividend_cagr
            }
            
            # Model Confidence
            confidence_factors = [
                sustainability_result.sustainability_score,
                aristocrat_result.growth_consistency,
                min(1.0, dividend_data.dividend_coverage / 2.0),
                min(1.0, dividend_history.years_of_growth / 10.0)
            ]
            model_confidence = np.mean(confidence_factors)
            
            return DividendAnalysisResult(
                gordon_growth=gordon_result,
                multi_stage=multi_stage_result,
                sustainability=sustainability_result,
                aristocrat_analysis=aristocrat_result,
                strategy_recommendation=strategy_result,
                risk_metrics=risk_metrics,
                performance_metrics=performance_metrics,
                model_confidence=model_confidence
            )
            
        except Exception as e:
            logger.error(f"Error in dividend analysis: {str(e)}")
            raise
    
    def get_dividend_insights(self, result: DividendAnalysisResult) -> Dict[str, str]:
        """Generate comprehensive dividend insights"""
        insights = {}
        
        # Gordon Growth insights
        gordon = result.gordon_growth
        insights['gordon_growth'] = f"Fair Value: ${gordon.intrinsic_value:.2f}, Growth: {gordon.dividend_growth_rate:.1%}, Yield on Cost: {gordon.dividend_yield_on_cost:.1%}"
        
        # Multi-stage insights
        multi = result.multi_stage
        insights['multi_stage'] = f"Two-Stage Value: ${multi.intrinsic_value:.2f}, Stage 1: ${multi.stage1_value:.2f}, Stage 2: ${multi.stage2_value:.2f}"
        
        # Sustainability insights
        sustain = result.sustainability
        insights['sustainability'] = f"Score: {sustain.sustainability_score:.1%}, Cut Risk: {sustain.dividend_cut_probability:.1%}, Coverage: {sustain.years_of_coverage:.1f} years"
        
        # Aristocrat insights
        aristocrat = result.aristocrat_analysis
        insights['aristocrat'] = f"Status: {'Yes' if aristocrat.aristocrat_status else 'No'}, Years: {aristocrat.years_of_increases}, Growth: {aristocrat.average_growth_rate:.1%}"
        
        # Strategy insights
        strategy = result.strategy_recommendation
        insights['strategy'] = f"Best Fit: {strategy.strategy_type.title()}, Score: {strategy.strategy_score:.1%}, Total Return: {strategy.total_return_forecast:.1%}"
        
        # Risk insights
        risk = result.risk_metrics
        insights['risk'] = f"Cut Risk: {risk['dividend_cut_risk']:.1%}, Payout Risk: {risk['payout_risk']:.1%}, Coverage Risk: {risk['coverage_risk']:.1%}"
        
        # Performance insights
        perf = result.performance_metrics
        insights['performance'] = f"Total Return: {perf['total_return']:.1%}, Income: {perf['income_component']:.1%}, Growth: {perf['growth_component']:.1%}"
        
        # Overall insights
        insights['overall'] = f"Model Confidence: {result.model_confidence:.1%}, Risk-Adj Return: {perf['risk_adjusted_return']:.1%}"
        
        return insights

# Example usage and testing
if __name__ == "__main__":
    # Test the Gordon Growth model
    
    # Sample dividend data
    dividend_data = DividendData(
        dividend_per_share=2.50,
        dividend_yield=0.035,
        payout_ratio=0.55,
        dividend_coverage=1.8,
        free_cash_flow_per_share=4.50,
        earnings_per_share=4.55,
        book_value_per_share=25.0,
        return_on_equity=0.18,
        debt_to_equity=0.4,
        current_ratio=1.5
    )
    
    # Sample dividend history
    dividend_history = DividendHistory(
        dates=[datetime(2020, 1, 1), datetime(2021, 1, 1), datetime(2022, 1, 1), datetime(2023, 1, 1)],
        dividends=[2.20, 2.30, 2.40, 2.50],
        special_dividends=[0.0, 0.0, 0.0, 0.0],
        dividend_frequency='quarterly',
        ex_dividend_dates=[],
        payment_dates=[],
        years_of_growth=15,
        consecutive_increases=15
    )
    
    # Create analyzer and run analysis
    analyzer = GordonGrowthAnalyzer()
    result = analyzer.analyze(dividend_data, dividend_history, beta=1.1)
    
    insights = analyzer.get_dividend_insights(result)
    
    print("=== Gordon Growth & Dividend Analysis ===")
    print(f"Gordon Growth Fair Value: ${result.gordon_growth.intrinsic_value:.2f}")
    print(f"Multi-Stage Fair Value: ${result.multi_stage.intrinsic_value:.2f}")
    print(f"Sustainability Score: {result.sustainability.sustainability_score:.1%}")
    print(f"Aristocrat Status: {'Yes' if result.aristocrat_analysis.aristocrat_status else 'No'}")
    print(f"Recommended Strategy: {result.strategy_recommendation.strategy_type.title()}")
    print(f"Model Confidence: {result.model_confidence:.1%}")
    print()
    
    print("=== Detailed Insights ===")
    for key, insight in insights.items():
        print(f"{key}: {insight}")