"""Financial Ratios Calculator for Stock Analysis

This module implements comprehensive financial ratio calculations including:
- Valuation ratios (P/E, P/B, PEG, etc.)
- Profitability ratios (ROE, ROA, ROI, etc.)
- Liquidity ratios (Current, Quick, Cash ratios)
- Leverage ratios (Debt-to-Equity, Interest Coverage, etc.)
- Efficiency ratios (Asset Turnover, Inventory Turnover, etc.)
- Market ratios (Dividend Yield, Market Cap ratios)

Author: Assistant
Date: 2024
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class StockFundamentals:
    """Stock fundamental data"""
    revenue: float
    net_income: float
    free_cash_flow: float
    total_debt: float
    shareholders_equity: float
    shares_outstanding: float
    dividend_per_share: float
    earnings_per_share: float
    book_value_per_share: float
    revenue_growth_rate: float
    earnings_growth_rate: float
    dividend_growth_rate: float
    beta: float
    market_cap: float
    # Additional fields for comprehensive ratio analysis
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    inventory: Optional[float] = None
    accounts_receivable: Optional[float] = None
    cash_and_equivalents: Optional[float] = None
    long_term_debt: Optional[float] = None
    interest_expense: Optional[float] = None
    operating_income: Optional[float] = None
    gross_profit: Optional[float] = None
    cost_of_goods_sold: Optional[float] = None

@dataclass
class FinancialRatiosResult:
    """Result of financial ratios analysis"""
    valuation_ratios: Dict[str, float]
    profitability_ratios: Dict[str, float]
    liquidity_ratios: Dict[str, float]
    leverage_ratios: Dict[str, float]
    efficiency_ratios: Dict[str, float]
    market_ratios: Dict[str, float]
    ratio_analysis: Dict[str, Dict[str, Any]]
    peer_comparison: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    overall_score: Dict[str, float]
    red_flags: List[str]
    strengths: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    interpretation: str

class FinancialRatiosCalculator:
    """Financial Ratios Calculator"""
    
    def __init__(self):
        # Industry benchmarks (simplified - in practice, these would be sector-specific)
        self.benchmarks = {
            'pe_ratio': {'excellent': 15, 'good': 20, 'average': 25, 'poor': 35},
            'pb_ratio': {'excellent': 1.0, 'good': 1.5, 'average': 2.5, 'poor': 4.0},
            'roe': {'excellent': 0.20, 'good': 0.15, 'average': 0.10, 'poor': 0.05},
            'roa': {'excellent': 0.10, 'good': 0.07, 'average': 0.05, 'poor': 0.02},
            'current_ratio': {'excellent': 2.0, 'good': 1.5, 'average': 1.2, 'poor': 1.0},
            'debt_to_equity': {'excellent': 0.3, 'good': 0.5, 'average': 0.8, 'poor': 1.2},
            'interest_coverage': {'excellent': 10, 'good': 5, 'average': 2.5, 'poor': 1.5}
        }
    
    def calculate(self, fundamentals: StockFundamentals, 
                 historical_data: Optional[List[StockFundamentals]] = None) -> FinancialRatiosResult:
        """Calculate comprehensive financial ratios"""
        try:
            # Calculate all ratio categories
            valuation_ratios = self._calculate_valuation_ratios(fundamentals)
            profitability_ratios = self._calculate_profitability_ratios(fundamentals)
            liquidity_ratios = self._calculate_liquidity_ratios(fundamentals)
            leverage_ratios = self._calculate_leverage_ratios(fundamentals)
            efficiency_ratios = self._calculate_efficiency_ratios(fundamentals)
            market_ratios = self._calculate_market_ratios(fundamentals)
            
            # Analyze ratios
            ratio_analysis = self._analyze_ratios({
                **valuation_ratios, **profitability_ratios, **liquidity_ratios,
                **leverage_ratios, **efficiency_ratios, **market_ratios
            })
            
            # Peer comparison (simplified)
            peer_comparison = self._compare_to_peers(ratio_analysis)
            
            # Trend analysis
            trend_analysis = self._analyze_trends(fundamentals, historical_data)
            
            # Overall scoring
            overall_score = self._calculate_overall_score(ratio_analysis)
            
            # Identify red flags and strengths
            red_flags = self._identify_red_flags(ratio_analysis, fundamentals)
            strengths = self._identify_strengths(ratio_analysis, fundamentals)
            
            # Generate interpretation
            interpretation = self._generate_interpretation(
                overall_score, red_flags, strengths
            )
            
            return FinancialRatiosResult(
                valuation_ratios=valuation_ratios,
                profitability_ratios=profitability_ratios,
                liquidity_ratios=liquidity_ratios,
                leverage_ratios=leverage_ratios,
                efficiency_ratios=efficiency_ratios,
                market_ratios=market_ratios,
                ratio_analysis=ratio_analysis,
                peer_comparison=peer_comparison,
                trend_analysis=trend_analysis,
                overall_score=overall_score,
                red_flags=red_flags,
                strengths=strengths,
                metadata={
                    'total_ratios_calculated': len(valuation_ratios) + len(profitability_ratios) + 
                                             len(liquidity_ratios) + len(leverage_ratios) + 
                                             len(efficiency_ratios) + len(market_ratios),
                    'data_completeness': self._assess_data_completeness(fundamentals)
                },
                timestamp=datetime.now(),
                interpretation=interpretation
            )
            
        except Exception as e:
            return self._create_fallback_result(str(e))
    
    def _calculate_valuation_ratios(self, fundamentals: StockFundamentals) -> Dict[str, float]:
        """Calculate valuation ratios"""
        ratios = {}
        
        # Price per share
        price_per_share = fundamentals.market_cap / fundamentals.shares_outstanding
        
        # P/E Ratio
        if fundamentals.earnings_per_share > 0:
            ratios['pe_ratio'] = price_per_share / fundamentals.earnings_per_share
        else:
            ratios['pe_ratio'] = float('inf')
        
        # P/B Ratio
        if fundamentals.book_value_per_share > 0:
            ratios['pb_ratio'] = price_per_share / fundamentals.book_value_per_share
        else:
            ratios['pb_ratio'] = float('inf')
        
        # PEG Ratio
        if fundamentals.earnings_growth_rate > 0 and ratios.get('pe_ratio', 0) != float('inf'):
            ratios['peg_ratio'] = ratios['pe_ratio'] / (fundamentals.earnings_growth_rate * 100)
        else:
            ratios['peg_ratio'] = float('inf')
        
        # Price-to-Sales Ratio
        revenue_per_share = fundamentals.revenue / fundamentals.shares_outstanding
        if revenue_per_share > 0:
            ratios['ps_ratio'] = price_per_share / revenue_per_share
        else:
            ratios['ps_ratio'] = float('inf')
        
        # Price-to-Cash Flow Ratio
        if fundamentals.free_cash_flow > 0:
            fcf_per_share = fundamentals.free_cash_flow / fundamentals.shares_outstanding
            ratios['pcf_ratio'] = price_per_share / fcf_per_share
        else:
            ratios['pcf_ratio'] = float('inf')
        
        # Enterprise Value ratios (simplified)
        enterprise_value = fundamentals.market_cap + fundamentals.total_debt
        if fundamentals.revenue > 0:
            ratios['ev_revenue'] = enterprise_value / fundamentals.revenue
        else:
            ratios['ev_revenue'] = float('inf')
        
        if fundamentals.operating_income and fundamentals.operating_income > 0:
            ratios['ev_ebitda'] = enterprise_value / fundamentals.operating_income  # Simplified EBITDA
        else:
            ratios['ev_ebitda'] = float('inf')
        
        return ratios
    
    def _calculate_profitability_ratios(self, fundamentals: StockFundamentals) -> Dict[str, float]:
        """Calculate profitability ratios"""
        ratios = {}
        
        # Return on Equity (ROE)
        if fundamentals.shareholders_equity > 0:
            ratios['roe'] = fundamentals.net_income / fundamentals.shareholders_equity
        else:
            ratios['roe'] = 0.0
        
        # Return on Assets (ROA)
        if fundamentals.total_assets and fundamentals.total_assets > 0:
            ratios['roa'] = fundamentals.net_income / fundamentals.total_assets
        else:
            # Estimate total assets as equity + debt
            estimated_assets = fundamentals.shareholders_equity + fundamentals.total_debt
            if estimated_assets > 0:
                ratios['roa'] = fundamentals.net_income / estimated_assets
            else:
                ratios['roa'] = 0.0
        
        # Net Profit Margin
        if fundamentals.revenue > 0:
            ratios['net_profit_margin'] = fundamentals.net_income / fundamentals.revenue
        else:
            ratios['net_profit_margin'] = 0.0
        
        # Gross Profit Margin
        if fundamentals.gross_profit and fundamentals.revenue > 0:
            ratios['gross_profit_margin'] = fundamentals.gross_profit / fundamentals.revenue
        else:
            ratios['gross_profit_margin'] = 0.0
        
        # Operating Margin
        if fundamentals.operating_income and fundamentals.revenue > 0:
            ratios['operating_margin'] = fundamentals.operating_income / fundamentals.revenue
        else:
            ratios['operating_margin'] = 0.0
        
        # Return on Investment (ROI) - simplified
        total_investment = fundamentals.shareholders_equity + fundamentals.total_debt
        if total_investment > 0:
            ratios['roi'] = fundamentals.net_income / total_investment
        else:
            ratios['roi'] = 0.0
        
        return ratios
    
    def _calculate_liquidity_ratios(self, fundamentals: StockFundamentals) -> Dict[str, float]:
        """Calculate liquidity ratios"""
        ratios = {}
        
        # Current Ratio
        if fundamentals.current_assets and fundamentals.current_liabilities:
            if fundamentals.current_liabilities > 0:
                ratios['current_ratio'] = fundamentals.current_assets / fundamentals.current_liabilities
            else:
                ratios['current_ratio'] = float('inf')
        else:
            ratios['current_ratio'] = 0.0
        
        # Quick Ratio (Acid Test)
        if (fundamentals.current_assets and fundamentals.inventory and 
            fundamentals.current_liabilities and fundamentals.current_liabilities > 0):
            quick_assets = fundamentals.current_assets - fundamentals.inventory
            ratios['quick_ratio'] = quick_assets / fundamentals.current_liabilities
        else:
            ratios['quick_ratio'] = 0.0
        
        # Cash Ratio
        if (fundamentals.cash_and_equivalents and fundamentals.current_liabilities and 
            fundamentals.current_liabilities > 0):
            ratios['cash_ratio'] = fundamentals.cash_and_equivalents / fundamentals.current_liabilities
        else:
            ratios['cash_ratio'] = 0.0
        
        # Working Capital Ratio
        if fundamentals.current_assets and fundamentals.current_liabilities:
            working_capital = fundamentals.current_assets - fundamentals.current_liabilities
            if fundamentals.revenue > 0:
                ratios['working_capital_ratio'] = working_capital / fundamentals.revenue
            else:
                ratios['working_capital_ratio'] = 0.0
        else:
            ratios['working_capital_ratio'] = 0.0
        
        return ratios
    
    def _calculate_leverage_ratios(self, fundamentals: StockFundamentals) -> Dict[str, float]:
        """Calculate leverage ratios"""
        ratios = {}
        
        # Debt-to-Equity Ratio
        if fundamentals.shareholders_equity > 0:
            ratios['debt_to_equity'] = fundamentals.total_debt / fundamentals.shareholders_equity
        else:
            ratios['debt_to_equity'] = float('inf')
        
        # Debt-to-Assets Ratio
        total_assets = fundamentals.total_assets or (fundamentals.shareholders_equity + fundamentals.total_debt)
        if total_assets > 0:
            ratios['debt_to_assets'] = fundamentals.total_debt / total_assets
        else:
            ratios['debt_to_assets'] = 0.0
        
        # Equity Ratio
        if total_assets > 0:
            ratios['equity_ratio'] = fundamentals.shareholders_equity / total_assets
        else:
            ratios['equity_ratio'] = 0.0
        
        # Interest Coverage Ratio
        if fundamentals.interest_expense and fundamentals.interest_expense > 0:
            # Use operating income or estimate from net income
            operating_income = fundamentals.operating_income or (fundamentals.net_income + fundamentals.interest_expense)
            ratios['interest_coverage'] = operating_income / fundamentals.interest_expense
        else:
            ratios['interest_coverage'] = float('inf')
        
        # Debt Service Coverage (simplified)
        if fundamentals.free_cash_flow > 0 and fundamentals.interest_expense:
            ratios['debt_service_coverage'] = fundamentals.free_cash_flow / fundamentals.interest_expense
        else:
            ratios['debt_service_coverage'] = 0.0
        
        # Long-term Debt to Equity
        if fundamentals.long_term_debt and fundamentals.shareholders_equity > 0:
            ratios['long_term_debt_to_equity'] = fundamentals.long_term_debt / fundamentals.shareholders_equity
        else:
            ratios['long_term_debt_to_equity'] = 0.0
        
        return ratios
    
    def _calculate_efficiency_ratios(self, fundamentals: StockFundamentals) -> Dict[str, float]:
        """Calculate efficiency ratios"""
        ratios = {}
        
        # Asset Turnover
        total_assets = fundamentals.total_assets or (fundamentals.shareholders_equity + fundamentals.total_debt)
        if total_assets > 0:
            ratios['asset_turnover'] = fundamentals.revenue / total_assets
        else:
            ratios['asset_turnover'] = 0.0
        
        # Equity Turnover
        if fundamentals.shareholders_equity > 0:
            ratios['equity_turnover'] = fundamentals.revenue / fundamentals.shareholders_equity
        else:
            ratios['equity_turnover'] = 0.0
        
        # Inventory Turnover
        if fundamentals.inventory and fundamentals.cost_of_goods_sold:
            if fundamentals.inventory > 0:
                ratios['inventory_turnover'] = fundamentals.cost_of_goods_sold / fundamentals.inventory
            else:
                ratios['inventory_turnover'] = float('inf')
        else:
            ratios['inventory_turnover'] = 0.0
        
        # Receivables Turnover
        if fundamentals.accounts_receivable and fundamentals.accounts_receivable > 0:
            ratios['receivables_turnover'] = fundamentals.revenue / fundamentals.accounts_receivable
        else:
            ratios['receivables_turnover'] = 0.0
        
        # Days Sales Outstanding (DSO)
        if ratios.get('receivables_turnover', 0) > 0:
            ratios['days_sales_outstanding'] = 365 / ratios['receivables_turnover']
        else:
            ratios['days_sales_outstanding'] = 0.0
        
        # Days Inventory Outstanding (DIO)
        if ratios.get('inventory_turnover', 0) > 0:
            ratios['days_inventory_outstanding'] = 365 / ratios['inventory_turnover']
        else:
            ratios['days_inventory_outstanding'] = 0.0
        
        return ratios
    
    def _calculate_market_ratios(self, fundamentals: StockFundamentals) -> Dict[str, float]:
        """Calculate market ratios"""
        ratios = {}
        
        # Dividend Yield
        price_per_share = fundamentals.market_cap / fundamentals.shares_outstanding
        if price_per_share > 0:
            ratios['dividend_yield'] = fundamentals.dividend_per_share / price_per_share
        else:
            ratios['dividend_yield'] = 0.0
        
        # Dividend Payout Ratio
        if fundamentals.earnings_per_share > 0:
            ratios['dividend_payout_ratio'] = fundamentals.dividend_per_share / fundamentals.earnings_per_share
        else:
            ratios['dividend_payout_ratio'] = 0.0
        
        # Earnings Yield
        if price_per_share > 0:
            ratios['earnings_yield'] = fundamentals.earnings_per_share / price_per_share
        else:
            ratios['earnings_yield'] = 0.0
        
        # Market-to-Book Ratio (same as P/B)
        if fundamentals.book_value_per_share > 0:
            ratios['market_to_book'] = price_per_share / fundamentals.book_value_per_share
        else:
            ratios['market_to_book'] = float('inf')
        
        # Price-to-Tangible Book Value (simplified)
        ratios['price_to_tangible_book'] = ratios['market_to_book']  # Simplified
        
        return ratios
    
    def _analyze_ratios(self, all_ratios: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Analyze ratios against benchmarks"""
        analysis = {}
        
        for ratio_name, ratio_value in all_ratios.items():
            if ratio_name in self.benchmarks:
                benchmark = self.benchmarks[ratio_name]
                
                # Determine rating
                if ratio_name in ['debt_to_equity']:  # Lower is better
                    if ratio_value <= benchmark['excellent']:
                        rating = 'Excellent'
                    elif ratio_value <= benchmark['good']:
                        rating = 'Good'
                    elif ratio_value <= benchmark['average']:
                        rating = 'Average'
                    elif ratio_value <= benchmark['poor']:
                        rating = 'Poor'
                    else:
                        rating = 'Very Poor'
                else:  # Higher is better
                    if ratio_value >= benchmark['excellent']:
                        rating = 'Excellent'
                    elif ratio_value >= benchmark['good']:
                        rating = 'Good'
                    elif ratio_value >= benchmark['average']:
                        rating = 'Average'
                    elif ratio_value >= benchmark['poor']:
                        rating = 'Poor'
                    else:
                        rating = 'Very Poor'
                
                analysis[ratio_name] = {
                    'value': ratio_value,
                    'rating': rating,
                    'benchmark': benchmark,
                    'interpretation': self._interpret_ratio(ratio_name, ratio_value, rating)
                }
            else:
                # No benchmark available
                analysis[ratio_name] = {
                    'value': ratio_value,
                    'rating': 'N/A',
                    'benchmark': None,
                    'interpretation': self._interpret_ratio(ratio_name, ratio_value, 'N/A')
                }
        
        return analysis
    
    def _interpret_ratio(self, ratio_name: str, value: float, rating: str) -> str:
        """Interpret individual ratio"""
        interpretations = {
            'pe_ratio': f"P/E of {value:.1f} indicates {'expensive' if value > 25 else 'reasonable' if value > 15 else 'cheap'} valuation",
            'pb_ratio': f"P/B of {value:.1f} suggests {'premium' if value > 2 else 'fair' if value > 1 else 'discount'} to book value",
            'roe': f"ROE of {value:.1%} shows {'strong' if value > 0.15 else 'adequate' if value > 0.10 else 'weak'} profitability",
            'roa': f"ROA of {value:.1%} indicates {'efficient' if value > 0.07 else 'average' if value > 0.03 else 'poor'} asset utilization",
            'current_ratio': f"Current ratio of {value:.1f} suggests {'strong' if value > 1.5 else 'adequate' if value > 1.2 else 'weak'} liquidity",
            'debt_to_equity': f"D/E of {value:.1f} indicates {'low' if value < 0.5 else 'moderate' if value < 1.0 else 'high'} leverage",
            'dividend_yield': f"Dividend yield of {value:.1%} is {'high' if value > 0.04 else 'moderate' if value > 0.02 else 'low'}"
        }
        
        return interpretations.get(ratio_name, f"{ratio_name}: {value:.3f} ({rating})")
    
    def _compare_to_peers(self, ratio_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare ratios to peer averages (simplified)"""
        # In a real implementation, this would use actual peer data
        peer_comparison = {
            'better_than_peers': [],
            'worse_than_peers': [],
            'peer_percentile': {}
        }
        
        for ratio_name, analysis in ratio_analysis.items():
            rating = analysis['rating']
            if rating in ['Excellent', 'Good']:
                peer_comparison['better_than_peers'].append(ratio_name)
                peer_comparison['peer_percentile'][ratio_name] = 75  # Simplified
            elif rating in ['Poor', 'Very Poor']:
                peer_comparison['worse_than_peers'].append(ratio_name)
                peer_comparison['peer_percentile'][ratio_name] = 25  # Simplified
            else:
                peer_comparison['peer_percentile'][ratio_name] = 50  # Average
        
        return peer_comparison
    
    def _analyze_trends(self, current: StockFundamentals, 
                       historical: Optional[List[StockFundamentals]]) -> Dict[str, Any]:
        """Analyze ratio trends over time"""
        if not historical or len(historical) < 2:
            return {
                'trend_analysis': 'Insufficient historical data',
                'improving_ratios': [],
                'declining_ratios': []
            }
        
        # Calculate trends for key ratios
        improving_ratios = []
        declining_ratios = []
        
        # ROE trend
        current_roe = current.net_income / current.shareholders_equity if current.shareholders_equity > 0 else 0
        historical_roe = [h.net_income / h.shareholders_equity if h.shareholders_equity > 0 else 0 for h in historical[-3:]]
        
        if len(historical_roe) >= 2 and current_roe > np.mean(historical_roe):
            improving_ratios.append('ROE')
        elif len(historical_roe) >= 2 and current_roe < np.mean(historical_roe):
            declining_ratios.append('ROE')
        
        # Debt-to-Equity trend
        current_de = current.total_debt / current.shareholders_equity if current.shareholders_equity > 0 else 0
        historical_de = [h.total_debt / h.shareholders_equity if h.shareholders_equity > 0 else 0 for h in historical[-3:]]
        
        if len(historical_de) >= 2 and current_de < np.mean(historical_de):
            improving_ratios.append('Debt-to-Equity')
        elif len(historical_de) >= 2 and current_de > np.mean(historical_de):
            declining_ratios.append('Debt-to-Equity')
        
        return {
            'trend_analysis': f'Analyzed {len(historical)} historical periods',
            'improving_ratios': improving_ratios,
            'declining_ratios': declining_ratios,
            'trend_strength': 'Strong' if len(improving_ratios) > len(declining_ratios) else 'Weak'
        }
    
    def _calculate_overall_score(self, ratio_analysis: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate overall financial health score"""
        category_scores = {
            'valuation': 0.0,
            'profitability': 0.0,
            'liquidity': 0.0,
            'leverage': 0.0,
            'efficiency': 0.0,
            'market': 0.0
        }
        
        category_counts = {k: 0 for k in category_scores.keys()}
        
        # Score mapping
        score_map = {
            'Excellent': 5,
            'Good': 4,
            'Average': 3,
            'Poor': 2,
            'Very Poor': 1,
            'N/A': 3
        }
        
        # Categorize ratios and calculate scores
        ratio_categories = {
            'valuation': ['pe_ratio', 'pb_ratio', 'peg_ratio', 'ps_ratio', 'pcf_ratio'],
            'profitability': ['roe', 'roa', 'net_profit_margin', 'gross_profit_margin', 'operating_margin'],
            'liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio'],
            'leverage': ['debt_to_equity', 'debt_to_assets', 'interest_coverage'],
            'efficiency': ['asset_turnover', 'inventory_turnover', 'receivables_turnover'],
            'market': ['dividend_yield', 'dividend_payout_ratio', 'earnings_yield']
        }
        
        for category, ratios in ratio_categories.items():
            total_score = 0
            count = 0
            
            for ratio in ratios:
                if ratio in ratio_analysis:
                    rating = ratio_analysis[ratio]['rating']
                    total_score += score_map.get(rating, 3)
                    count += 1
            
            if count > 0:
                category_scores[category] = total_score / count
                category_counts[category] = count
        
        # Overall score (weighted average)
        weights = {
            'valuation': 0.20,
            'profitability': 0.25,
            'liquidity': 0.15,
            'leverage': 0.20,
            'efficiency': 0.10,
            'market': 0.10
        }
        
        overall_score = sum(
            category_scores[cat] * weights[cat] 
            for cat in category_scores.keys()
            if category_counts[cat] > 0
        )
        
        return {
            'overall': overall_score,
            **category_scores,
            'max_score': 5.0,
            'percentage': (overall_score / 5.0) * 100
        }
    
    def _identify_red_flags(self, ratio_analysis: Dict[str, Dict[str, Any]], 
                           fundamentals: StockFundamentals) -> List[str]:
        """Identify financial red flags"""
        red_flags = []
        
        # Check for poor ratings in critical ratios
        critical_ratios = ['roe', 'current_ratio', 'debt_to_equity', 'interest_coverage']
        
        for ratio in critical_ratios:
            if ratio in ratio_analysis:
                rating = ratio_analysis[ratio]['rating']
                if rating in ['Poor', 'Very Poor']:
                    red_flags.append(f"Poor {ratio.replace('_', ' ').title()}: {rating}")
        
        # Specific red flags
        if fundamentals.net_income < 0:
            red_flags.append("Negative net income")
        
        if fundamentals.free_cash_flow < 0:
            red_flags.append("Negative free cash flow")
        
        if fundamentals.earnings_growth_rate < -0.1:
            red_flags.append("Declining earnings (>10%)")
        
        # High debt levels
        if 'debt_to_equity' in ratio_analysis:
            de_ratio = ratio_analysis['debt_to_equity']['value']
            if de_ratio > 2.0:
                red_flags.append(f"Very high debt-to-equity ratio: {de_ratio:.1f}")
        
        # Low liquidity
        if 'current_ratio' in ratio_analysis:
            current_ratio = ratio_analysis['current_ratio']['value']
            if current_ratio < 1.0:
                red_flags.append(f"Current ratio below 1.0: {current_ratio:.1f}")
        
        return red_flags
    
    def _identify_strengths(self, ratio_analysis: Dict[str, Dict[str, Any]], 
                           fundamentals: StockFundamentals) -> List[str]:
        """Identify financial strengths"""
        strengths = []
        
        # Check for excellent ratings
        for ratio_name, analysis in ratio_analysis.items():
            if analysis['rating'] == 'Excellent':
                strengths.append(f"Excellent {ratio_name.replace('_', ' ').title()}")
        
        # Specific strengths
        if fundamentals.earnings_growth_rate > 0.15:
            strengths.append(f"Strong earnings growth: {fundamentals.earnings_growth_rate:.1%}")
        
        if fundamentals.revenue_growth_rate > 0.10:
            strengths.append(f"Strong revenue growth: {fundamentals.revenue_growth_rate:.1%}")
        
        if fundamentals.dividend_per_share > 0 and fundamentals.dividend_growth_rate > 0.05:
            strengths.append(f"Growing dividend: {fundamentals.dividend_growth_rate:.1%} growth")
        
        # High profitability
        if 'roe' in ratio_analysis and ratio_analysis['roe']['value'] > 0.20:
            strengths.append(f"High ROE: {ratio_analysis['roe']['value']:.1%}")
        
        # Strong liquidity
        if 'current_ratio' in ratio_analysis and ratio_analysis['current_ratio']['value'] > 2.0:
            strengths.append(f"Strong liquidity: Current ratio {ratio_analysis['current_ratio']['value']:.1f}")
        
        return strengths
    
    def _generate_interpretation(self, overall_score: Dict[str, float], 
                               red_flags: List[str], strengths: List[str]) -> str:
        """Generate overall interpretation"""
        score_pct = overall_score.get('percentage', 0)
        
        if score_pct >= 80:
            health_rating = "Excellent"
        elif score_pct >= 70:
            health_rating = "Good"
        elif score_pct >= 60:
            health_rating = "Average"
        elif score_pct >= 50:
            health_rating = "Below Average"
        else:
            health_rating = "Poor"
        
        interpretation_parts = [
            f"Financial Health: {health_rating} ({score_pct:.0f}/100)"
        ]
        
        if strengths:
            interpretation_parts.append(f"Key Strengths: {len(strengths)} identified")
        
        if red_flags:
            interpretation_parts.append(f"Red Flags: {len(red_flags)} identified")
        
        # Best and worst categories
        category_scores = {k: v for k, v in overall_score.items() 
                          if k not in ['overall', 'max_score', 'percentage']}
        
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            worst_category = min(category_scores.items(), key=lambda x: x[1])
            
            interpretation_parts.append(
                f"Strongest: {best_category[0].title()} ({best_category[1]:.1f}/5.0)"
            )
            interpretation_parts.append(
                f"Weakest: {worst_category[0].title()} ({worst_category[1]:.1f}/5.0)"
            )
        
        return "; ".join(interpretation_parts)
    
    def _assess_data_completeness(self, fundamentals: StockFundamentals) -> float:
        """Assess completeness of fundamental data"""
        required_fields = [
            'revenue', 'net_income', 'shareholders_equity', 'total_debt',
            'shares_outstanding', 'earnings_per_share', 'book_value_per_share'
        ]
        
        optional_fields = [
            'total_assets', 'current_assets', 'current_liabilities',
            'inventory', 'accounts_receivable', 'cash_and_equivalents',
            'long_term_debt', 'interest_expense', 'operating_income',
            'gross_profit', 'cost_of_goods_sold'
        ]
        
        required_count = sum(1 for field in required_fields 
                           if getattr(fundamentals, field, None) is not None)
        optional_count = sum(1 for field in optional_fields 
                           if getattr(fundamentals, field, None) is not None)
        
        required_completeness = required_count / len(required_fields)
        optional_completeness = optional_count / len(optional_fields)
        
        # Weight required fields more heavily
        overall_completeness = (required_completeness * 0.7 + optional_completeness * 0.3)
        
        return overall_completeness
    
    def _create_fallback_result(self, error_message: str) -> FinancialRatiosResult:
        """Create fallback result when calculation fails"""
        return FinancialRatiosResult(
            valuation_ratios={},
            profitability_ratios={},
            liquidity_ratios={},
            leverage_ratios={},
            efficiency_ratios={},
            market_ratios={},
            ratio_analysis={},
            peer_comparison={},
            trend_analysis={},
            overall_score={'overall': 0.0, 'percentage': 0.0},
            red_flags=["Calculation failed"],
            strengths=[],
            metadata={'error': error_message},
            timestamp=datetime.now(),
            interpretation="Financial ratio analysis failed"
        )

# Example usage
if __name__ == "__main__":
    # Sample comprehensive data
    fundamentals = StockFundamentals(
        revenue=10000000000,  # $10B
        net_income=1000000000,  # $1B
        free_cash_flow=800000000,  # $800M
        total_debt=2000000000,  # $2B
        shareholders_equity=5000000000,  # $5B
        shares_outstanding=100000000,  # 100M
        dividend_per_share=4.0,
        earnings_per_share=10.0,
        book_value_per_share=50.0,
        revenue_growth_rate=0.05,
        earnings_growth_rate=0.08,
        dividend_growth_rate=0.06,
        beta=1.2,
        market_cap=8000000000,  # $8B
        # Additional comprehensive data
        total_assets=7000000000,  # $7B
        current_assets=3000000000,  # $3B
        current_liabilities=1500000000,  # $1.5B
        inventory=500000000,  # $500M
        accounts_receivable=800000000,  # $800M
        cash_and_equivalents=1000000000,  # $1B
        long_term_debt=1500000000,  # $1.5B
        interest_expense=100000000,  # $100M
        operating_income=1200000000,  # $1.2B
        gross_profit=4000000000,  # $4B
        cost_of_goods_sold=6000000000  # $6B
    )
    
    # Calculate ratios
    calculator = FinancialRatiosCalculator()
    result = calculator.calculate(fundamentals)
    
    print(f"Financial Ratios Analysis Results:")
    print(f"Overall Score: {result.overall_score['percentage']:.0f}/100 ({result.overall_score['overall']:.1f}/5.0)")
    
    print(f"\nValuation Ratios:")
    for ratio, value in result.valuation_ratios.items():
        if value != float('inf'):
            print(f"{ratio.replace('_', ' ').title()}: {value:.2f}")
    
    print(f"\nProfitability Ratios:")
    for ratio, value in result.profitability_ratios.items():
        print(f"{ratio.replace('_', ' ').title()}: {value:.2%}")
    
    print(f"\nLiquidity Ratios:")
    for ratio, value in result.liquidity_ratios.items():
        if value != float('inf'):
            print(f"{ratio.replace('_', ' ').title()}: {value:.2f}")
    
    print(f"\nLeverage Ratios:")
    for ratio, value in result.leverage_ratios.items():
        if value != float('inf'):
            print(f"{ratio.replace('_', ' ').title()}: {value:.2f}")
    
    if result.strengths:
        print(f"\nKey Strengths:")
        for strength in result.strengths[:5]:  # Top 5
            print(f"- {strength}")
    
    if result.red_flags:
        print(f"\nRed Flags:")
        for flag in result.red_flags[:5]:  # Top 5
            print(f"- {flag}")
    
    print(f"\nInterpretation: {result.interpretation}")