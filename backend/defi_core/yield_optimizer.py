"""DeFi Yield Optimizer

This module provides intelligent yield farming optimization across multiple DeFi protocols.
It analyzes yield opportunities, calculates optimal allocation strategies, and manages
risk-adjusted returns for users.

Features:
- Multi-protocol yield analysis
- Risk-adjusted return calculations
- Automated yield farming strategies
- Impermanent loss protection
- Gas cost optimization
- Portfolio rebalancing

Author: FinScope AI Team
Version: 1.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

class YieldStrategy(Enum):
    """Available yield farming strategies"""
    CONSERVATIVE = "conservative"  # Low risk, stable returns
    MODERATE = "moderate"  # Balanced risk/reward
    AGGRESSIVE = "aggressive"  # High risk, high potential returns
    CUSTOM = "custom"  # User-defined strategy

class RiskLevel(Enum):
    """Risk levels for yield opportunities"""
    VERY_LOW = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4
    VERY_HIGH = 5

@dataclass
class YieldOpportunity:
    """Represents a yield farming opportunity"""
    protocol_name: str
    pool_address: str
    token_pair: Tuple[str, str]
    apy: Decimal
    tvl: Decimal
    risk_level: RiskLevel
    impermanent_loss_risk: Decimal
    gas_cost_estimate: Decimal
    minimum_deposit: Decimal
    lock_period: Optional[int] = None  # Days
    additional_rewards: List[str] = field(default_factory=list)
    strategy_type: str = "liquidity_mining"
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class OptimizationResult:
    """Result of yield optimization"""
    total_apy: Decimal
    risk_score: float
    allocations: List[Dict[str, Any]]
    expected_returns: Dict[str, Decimal]
    gas_costs: Decimal
    rebalance_frequency: int  # Days
    confidence_score: float
    warnings: List[str] = field(default_factory=list)

@dataclass
class UserProfile:
    """User's yield farming profile"""
    risk_tolerance: RiskLevel
    investment_amount: Decimal
    preferred_tokens: List[str]
    excluded_protocols: List[str] = field(default_factory=list)
    max_gas_percentage: float = 0.05  # 5% of investment
    rebalance_threshold: float = 0.1  # 10% deviation
    auto_compound: bool = True

class YieldOptimizer:
    """Advanced DeFi Yield Optimizer
    
    Analyzes yield opportunities across multiple protocols and provides
    optimized allocation strategies based on user preferences and risk tolerance.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Yield Optimizer
        
        Args:
            config: Configuration dictionary containing:
                - protocols: List of supported protocols
                - risk_parameters: Risk calculation parameters
                - optimization_settings: Optimization algorithm settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.YieldOptimizer")
        
        # Yield opportunities cache
        self.opportunities: Dict[str, YieldOpportunity] = {}
        self.cache_ttl = timedelta(minutes=10)
        self.last_update = datetime.min
        
        # Risk parameters
        self.risk_params = config.get('risk_parameters', {
            'impermanent_loss_weight': 0.3,
            'protocol_risk_weight': 0.2,
            'liquidity_risk_weight': 0.2,
            'smart_contract_risk_weight': 0.3
        })
        
        # Optimization settings
        self.optimization_settings = config.get('optimization_settings', {
            'max_protocols': 5,
            'min_allocation_percentage': 0.05,  # 5%
            'rebalance_threshold': 0.1,  # 10%
            'gas_optimization': True
        })
        
        # Protocol risk scores (would be loaded from external data)
        self.protocol_risk_scores = {
            'uniswap': 2,
            'sushiswap': 2,
            'compound': 1,
            'aave': 1,
            'curve': 2,
            'balancer': 3,
            'pancakeswap': 3,
            'quickswap': 3
        }
        
        self.logger.info("Yield Optimizer initialized")
    
    async def get_yield_opportunities(self, force_refresh: bool = False) -> List[YieldOpportunity]:
        """Get current yield opportunities
        
        Args:
            force_refresh: Force refresh of opportunities data
            
        Returns:
            List of available yield opportunities
        """
        try:
            # Check if cache is still valid
            if not force_refresh and self._is_cache_valid():
                return list(self.opportunities.values())
            
            # Fetch fresh opportunities
            await self._fetch_opportunities()
            return list(self.opportunities.values())
            
        except Exception as e:
            self.logger.error(f"Error getting yield opportunities: {e}")
            return []
    
    async def _fetch_opportunities(self):
        """Fetch yield opportunities from various protocols"""
        self.logger.info("Fetching yield opportunities...")
        
        # Mock opportunities for demonstration
        mock_opportunities = [
            YieldOpportunity(
                protocol_name="uniswap_v3",
                pool_address="0x1234567890123456789012345678901234567890",
                token_pair=("USDC", "ETH"),
                apy=Decimal('12.5'),
                tvl=Decimal('50000000'),  # $50M
                risk_level=RiskLevel.MEDIUM,
                impermanent_loss_risk=Decimal('0.15'),
                gas_cost_estimate=Decimal('50'),
                minimum_deposit=Decimal('100'),
                additional_rewards=["UNI"]
            ),
            YieldOpportunity(
                protocol_name="compound",
                pool_address="0x2345678901234567890123456789012345678901",
                token_pair=("USDC", "USDC"),  # Lending
                apy=Decimal('8.2'),
                tvl=Decimal('200000000'),  # $200M
                risk_level=RiskLevel.LOW,
                impermanent_loss_risk=Decimal('0.0'),
                gas_cost_estimate=Decimal('30'),
                minimum_deposit=Decimal('1'),
                strategy_type="lending"
            ),
            YieldOpportunity(
                protocol_name="curve",
                pool_address="0x3456789012345678901234567890123456789012",
                token_pair=("USDC", "USDT"),
                apy=Decimal('15.8'),
                tvl=Decimal('100000000'),  # $100M
                risk_level=RiskLevel.MEDIUM,
                impermanent_loss_risk=Decimal('0.05'),
                gas_cost_estimate=Decimal('40'),
                minimum_deposit=Decimal('50'),
                additional_rewards=["CRV", "CVX"]
            ),
            YieldOpportunity(
                protocol_name="aave",
                pool_address="0x4567890123456789012345678901234567890123",
                token_pair=("ETH", "ETH"),  # Lending
                apy=Decimal('6.5'),
                tvl=Decimal('300000000'),  # $300M
                risk_level=RiskLevel.LOW,
                impermanent_loss_risk=Decimal('0.0'),
                gas_cost_estimate=Decimal('35'),
                minimum_deposit=Decimal('0.01'),
                strategy_type="lending"
            ),
            YieldOpportunity(
                protocol_name="sushiswap",
                pool_address="0x5678901234567890123456789012345678901234",
                token_pair=("SUSHI", "ETH"),
                apy=Decimal('25.3'),
                tvl=Decimal('20000000'),  # $20M
                risk_level=RiskLevel.HIGH,
                impermanent_loss_risk=Decimal('0.25'),
                gas_cost_estimate=Decimal('45'),
                minimum_deposit=Decimal('10'),
                additional_rewards=["SUSHI"]
            )
        ]
        
        # Store opportunities
        self.opportunities.clear()
        for opp in mock_opportunities:
            key = f"{opp.protocol_name}_{opp.pool_address}"
            self.opportunities[key] = opp
        
        self.last_update = datetime.now()
        self.logger.info(f"Fetched {len(mock_opportunities)} yield opportunities")
    
    def _is_cache_valid(self) -> bool:
        """Check if opportunities cache is still valid"""
        return datetime.now() - self.last_update < self.cache_ttl
    
    async def optimize_yield(self, user_profile: UserProfile) -> OptimizationResult:
        """Optimize yield allocation for user
        
        Args:
            user_profile: User's investment profile and preferences
            
        Returns:
            Optimized allocation strategy
        """
        try:
            self.logger.info(f"Optimizing yield for {user_profile.investment_amount} investment")
            
            # Get available opportunities
            opportunities = await self.get_yield_opportunities()
            
            # Filter opportunities based on user preferences
            filtered_opportunities = self._filter_opportunities(opportunities, user_profile)
            
            if not filtered_opportunities:
                return OptimizationResult(
                    total_apy=Decimal('0'),
                    risk_score=0.0,
                    allocations=[],
                    expected_returns={},
                    gas_costs=Decimal('0'),
                    rebalance_frequency=30,
                    confidence_score=0.0,
                    warnings=["No suitable opportunities found"]
                )
            
            # Calculate optimal allocations
            allocations = self._calculate_optimal_allocations(filtered_opportunities, user_profile)
            
            # Calculate expected returns and metrics
            total_apy, risk_score, expected_returns, gas_costs = self._calculate_metrics(
                allocations, user_profile.investment_amount
            )
            
            # Determine rebalance frequency
            rebalance_frequency = self._calculate_rebalance_frequency(allocations, user_profile)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(allocations, filtered_opportunities)
            
            # Generate warnings
            warnings = self._generate_warnings(allocations, user_profile)
            
            result = OptimizationResult(
                total_apy=total_apy,
                risk_score=risk_score,
                allocations=allocations,
                expected_returns=expected_returns,
                gas_costs=gas_costs,
                rebalance_frequency=rebalance_frequency,
                confidence_score=confidence_score,
                warnings=warnings
            )
            
            self.logger.info(f"Optimization complete: {total_apy:.2f}% APY, Risk: {risk_score:.2f}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error optimizing yield: {e}")
            return OptimizationResult(
                total_apy=Decimal('0'),
                risk_score=0.0,
                allocations=[],
                expected_returns={},
                gas_costs=Decimal('0'),
                rebalance_frequency=30,
                confidence_score=0.0,
                warnings=[f"Optimization error: {str(e)}"]
            )
    
    def _filter_opportunities(self, opportunities: List[YieldOpportunity], 
                            user_profile: UserProfile) -> List[YieldOpportunity]:
        """Filter opportunities based on user preferences"""
        filtered = []
        
        for opp in opportunities:
            # Check risk tolerance
            if opp.risk_level.value > user_profile.risk_tolerance.value:
                continue
            
            # Check excluded protocols
            if opp.protocol_name in user_profile.excluded_protocols:
                continue
            
            # Check minimum deposit
            if opp.minimum_deposit > user_profile.investment_amount:
                continue
            
            # Check gas costs
            max_gas = user_profile.investment_amount * Decimal(str(user_profile.max_gas_percentage))
            if opp.gas_cost_estimate > max_gas:
                continue
            
            # Check preferred tokens
            if user_profile.preferred_tokens:
                token_match = any(token in opp.token_pair for token in user_profile.preferred_tokens)
                if not token_match:
                    continue
            
            filtered.append(opp)
        
        return filtered
    
    def _calculate_optimal_allocations(self, opportunities: List[YieldOpportunity], 
                                     user_profile: UserProfile) -> List[Dict[str, Any]]:
        """Calculate optimal allocation percentages"""
        if not opportunities:
            return []
        
        # Sort opportunities by risk-adjusted return
        scored_opportunities = []
        for opp in opportunities:
            risk_adjusted_apy = self._calculate_risk_adjusted_return(opp)
            scored_opportunities.append((opp, risk_adjusted_apy))
        
        scored_opportunities.sort(key=lambda x: x[1], reverse=True)
        
        # Apply allocation strategy based on user profile
        if user_profile.risk_tolerance == RiskLevel.VERY_LOW:
            return self._conservative_allocation(scored_opportunities, user_profile)
        elif user_profile.risk_tolerance == RiskLevel.LOW:
            return self._moderate_allocation(scored_opportunities, user_profile)
        elif user_profile.risk_tolerance in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
            return self._aggressive_allocation(scored_opportunities, user_profile)
        else:
            return self._balanced_allocation(scored_opportunities, user_profile)
    
    def _calculate_risk_adjusted_return(self, opportunity: YieldOpportunity) -> Decimal:
        """Calculate risk-adjusted return for an opportunity"""
        base_apy = opportunity.apy
        
        # Adjust for impermanent loss risk
        il_penalty = opportunity.impermanent_loss_risk * Decimal('10')  # 10% penalty per 1% IL risk
        
        # Adjust for protocol risk
        protocol_risk = self.protocol_risk_scores.get(opportunity.protocol_name, 3)
        risk_penalty = Decimal(str(protocol_risk)) * Decimal('2')  # 2% penalty per risk level
        
        # Adjust for liquidity (TVL)
        if opportunity.tvl < Decimal('10000000'):  # Less than $10M
            liquidity_penalty = Decimal('5')  # 5% penalty
        elif opportunity.tvl < Decimal('50000000'):  # Less than $50M
            liquidity_penalty = Decimal('2')  # 2% penalty
        else:
            liquidity_penalty = Decimal('0')
        
        risk_adjusted = base_apy - il_penalty - risk_penalty - liquidity_penalty
        return max(risk_adjusted, Decimal('0'))
    
    def _conservative_allocation(self, scored_opportunities: List[Tuple[YieldOpportunity, Decimal]], 
                               user_profile: UserProfile) -> List[Dict[str, Any]]:
        """Conservative allocation strategy - focus on low-risk opportunities"""
        allocations = []
        remaining_percentage = Decimal('1.0')
        
        # Only use top 2-3 lowest risk opportunities
        low_risk_opps = [opp for opp, _ in scored_opportunities if opp.risk_level.value <= 2][:3]
        
        if not low_risk_opps:
            return allocations
        
        # Equal allocation among low-risk opportunities
        allocation_per_opp = remaining_percentage / len(low_risk_opps)
        
        for opp in low_risk_opps:
            if allocation_per_opp >= Decimal(str(self.optimization_settings['min_allocation_percentage'])):
                allocations.append({
                    'protocol': opp.protocol_name,
                    'pool_address': opp.pool_address,
                    'token_pair': opp.token_pair,
                    'allocation_percentage': float(allocation_per_opp),
                    'amount': user_profile.investment_amount * allocation_per_opp,
                    'expected_apy': opp.apy,
                    'risk_level': opp.risk_level.name,
                    'strategy_type': opp.strategy_type
                })
        
        return allocations
    
    def _moderate_allocation(self, scored_opportunities: List[Tuple[YieldOpportunity, Decimal]], 
                           user_profile: UserProfile) -> List[Dict[str, Any]]:
        """Moderate allocation strategy - balanced approach"""
        allocations = []
        
        # Use top 3-4 opportunities with mixed risk levels
        selected_opps = scored_opportunities[:4]
        
        # Weight allocation by risk-adjusted return
        total_score = sum(score for _, score in selected_opps)
        
        for opp, score in selected_opps:
            if total_score > 0:
                allocation_percentage = float(score / total_score)
                
                if allocation_percentage >= self.optimization_settings['min_allocation_percentage']:
                    allocations.append({
                        'protocol': opp.protocol_name,
                        'pool_address': opp.pool_address,
                        'token_pair': opp.token_pair,
                        'allocation_percentage': allocation_percentage,
                        'amount': user_profile.investment_amount * Decimal(str(allocation_percentage)),
                        'expected_apy': opp.apy,
                        'risk_level': opp.risk_level.name,
                        'strategy_type': opp.strategy_type
                    })
        
        return allocations
    
    def _aggressive_allocation(self, scored_opportunities: List[Tuple[YieldOpportunity, Decimal]], 
                             user_profile: UserProfile) -> List[Dict[str, Any]]:
        """Aggressive allocation strategy - focus on highest returns"""
        allocations = []
        
        # Use top 2-3 highest yielding opportunities
        top_opps = scored_opportunities[:3]
        
        # Allocate more to highest yielding
        weights = [0.5, 0.3, 0.2]  # Decreasing weights
        
        for i, (opp, _) in enumerate(top_opps):
            if i < len(weights):
                allocation_percentage = weights[i]
                
                allocations.append({
                    'protocol': opp.protocol_name,
                    'pool_address': opp.pool_address,
                    'token_pair': opp.token_pair,
                    'allocation_percentage': allocation_percentage,
                    'amount': user_profile.investment_amount * Decimal(str(allocation_percentage)),
                    'expected_apy': opp.apy,
                    'risk_level': opp.risk_level.name,
                    'strategy_type': opp.strategy_type
                })
        
        return allocations
    
    def _balanced_allocation(self, scored_opportunities: List[Tuple[YieldOpportunity, Decimal]], 
                           user_profile: UserProfile) -> List[Dict[str, Any]]:
        """Balanced allocation strategy"""
        return self._moderate_allocation(scored_opportunities, user_profile)
    
    def _calculate_metrics(self, allocations: List[Dict[str, Any]], 
                         investment_amount: Decimal) -> Tuple[Decimal, float, Dict[str, Decimal], Decimal]:
        """Calculate portfolio metrics"""
        if not allocations:
            return Decimal('0'), 0.0, {}, Decimal('0')
        
        # Calculate weighted average APY
        total_apy = Decimal('0')
        total_risk_score = 0.0
        total_gas_costs = Decimal('0')
        expected_returns = {}
        
        for allocation in allocations:
            weight = Decimal(str(allocation['allocation_percentage']))
            apy = allocation['expected_apy']
            
            total_apy += apy * weight
            
            # Calculate risk contribution
            risk_level_map = {'VERY_LOW': 1, 'LOW': 2, 'MEDIUM': 3, 'HIGH': 4, 'VERY_HIGH': 5}
            risk_value = risk_level_map.get(allocation['risk_level'], 3)
            total_risk_score += risk_value * float(weight)
            
            # Estimate gas costs (simplified)
            gas_cost = Decimal('50') * weight  # Base gas cost per allocation
            total_gas_costs += gas_cost
            
            # Expected annual returns
            annual_return = allocation['amount'] * (apy / Decimal('100'))
            expected_returns[allocation['protocol']] = annual_return
        
        return total_apy, total_risk_score, expected_returns, total_gas_costs
    
    def _calculate_rebalance_frequency(self, allocations: List[Dict[str, Any]], 
                                     user_profile: UserProfile) -> int:
        """Calculate optimal rebalance frequency in days"""
        if not allocations:
            return 30
        
        # More volatile allocations need more frequent rebalancing
        avg_risk = sum(self._get_risk_value(alloc['risk_level']) for alloc in allocations) / len(allocations)
        
        if avg_risk <= 2:
            return 90  # Quarterly for low risk
        elif avg_risk <= 3:
            return 30  # Monthly for medium risk
        else:
            return 14  # Bi-weekly for high risk
    
    def _get_risk_value(self, risk_level: str) -> int:
        """Convert risk level string to numeric value"""
        risk_map = {'VERY_LOW': 1, 'LOW': 2, 'MEDIUM': 3, 'HIGH': 4, 'VERY_HIGH': 5}
        return risk_map.get(risk_level, 3)
    
    def _calculate_confidence_score(self, allocations: List[Dict[str, Any]], 
                                  opportunities: List[YieldOpportunity]) -> float:
        """Calculate confidence score for the optimization"""
        if not allocations or not opportunities:
            return 0.0
        
        # Factors affecting confidence:
        # 1. Number of opportunities considered
        # 2. TVL of selected protocols
        # 3. Diversification
        
        opportunity_factor = min(len(opportunities) / 10, 1.0)  # More opportunities = higher confidence
        
        # TVL factor
        total_tvl = sum(opp.tvl for opp in opportunities if any(
            alloc['protocol'] == opp.protocol_name for alloc in allocations
        ))
        tvl_factor = min(float(total_tvl) / 100000000, 1.0)  # Normalize to $100M
        
        # Diversification factor
        diversification_factor = min(len(allocations) / 5, 1.0)  # Up to 5 protocols
        
        confidence = (opportunity_factor + tvl_factor + diversification_factor) / 3
        return round(confidence, 2)
    
    def _generate_warnings(self, allocations: List[Dict[str, Any]], 
                         user_profile: UserProfile) -> List[str]:
        """Generate warnings for the user"""
        warnings = []
        
        if not allocations:
            warnings.append("No suitable allocations found")
            return warnings
        
        # Check for high risk allocations
        high_risk_count = sum(1 for alloc in allocations if self._get_risk_value(alloc['risk_level']) >= 4)
        if high_risk_count > 0:
            warnings.append(f"{high_risk_count} high-risk allocation(s) detected")
        
        # Check for concentration risk
        max_allocation = max(alloc['allocation_percentage'] for alloc in allocations)
        if max_allocation > 0.6:
            warnings.append("High concentration risk - consider diversifying")
        
        # Check for impermanent loss risk
        il_protocols = [alloc['protocol'] for alloc in allocations if alloc['strategy_type'] == 'liquidity_mining']
        if len(il_protocols) > len(allocations) * 0.7:
            warnings.append("High impermanent loss exposure - monitor token price correlations")
        
        return warnings
    
    async def get_portfolio_performance(self, allocations: List[Dict[str, Any]], 
                                      days: int = 30) -> Dict[str, Any]:
        """Get historical performance data for a portfolio
        
        Args:
            allocations: Current portfolio allocations
            days: Number of days to analyze
            
        Returns:
            Performance metrics dictionary
        """
        try:
            # Mock performance data
            total_return = Decimal('0')
            protocol_performance = {}
            
            for allocation in allocations:
                # Simulate daily returns (simplified)
                daily_return = allocation['expected_apy'] / Decimal('365')
                period_return = daily_return * days / 100
                allocation_return = allocation['amount'] * period_return
                
                total_return += allocation_return
                protocol_performance[allocation['protocol']] = {
                    'return': float(allocation_return),
                    'return_percentage': float(period_return * 100),
                    'apy': float(allocation['expected_apy'])
                }
            
            return {
                'total_return': float(total_return),
                'total_return_percentage': float(total_return / sum(Decimal(str(a['amount'])) for a in allocations) * 100),
                'protocol_performance': protocol_performance,
                'analysis_period_days': days,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio performance: {e}")
            return {}
    
    async def suggest_rebalancing(self, current_allocations: List[Dict[str, Any]], 
                                user_profile: UserProfile) -> Dict[str, Any]:
        """Suggest portfolio rebalancing
        
        Args:
            current_allocations: Current portfolio allocations
            user_profile: User's investment profile
            
        Returns:
            Rebalancing suggestions
        """
        try:
            # Get current optimal allocations
            optimal_result = await self.optimize_yield(user_profile)
            
            if not optimal_result.allocations:
                return {'rebalance_needed': False, 'suggestions': []}
            
            # Compare current vs optimal
            suggestions = []
            rebalance_needed = False
            
            # Create lookup for current allocations
            current_lookup = {alloc['protocol']: alloc for alloc in current_allocations}
            optimal_lookup = {alloc['protocol']: alloc for alloc in optimal_result.allocations}
            
            # Check for significant deviations
            for protocol, optimal_alloc in optimal_lookup.items():
                current_alloc = current_lookup.get(protocol)
                
                if not current_alloc:
                    # New protocol to add
                    suggestions.append({
                        'action': 'add',
                        'protocol': protocol,
                        'target_percentage': optimal_alloc['allocation_percentage'],
                        'reason': 'New high-yield opportunity'
                    })
                    rebalance_needed = True
                else:
                    # Check for significant deviation
                    current_pct = current_alloc['allocation_percentage']
                    optimal_pct = optimal_alloc['allocation_percentage']
                    deviation = abs(current_pct - optimal_pct)
                    
                    if deviation > user_profile.rebalance_threshold:
                        action = 'increase' if optimal_pct > current_pct else 'decrease'
                        suggestions.append({
                            'action': action,
                            'protocol': protocol,
                            'current_percentage': current_pct,
                            'target_percentage': optimal_pct,
                            'deviation': deviation,
                            'reason': f'Allocation drift beyond {user_profile.rebalance_threshold*100}% threshold'
                        })
                        rebalance_needed = True
            
            # Check for protocols to remove
            for protocol, current_alloc in current_lookup.items():
                if protocol not in optimal_lookup:
                    suggestions.append({
                        'action': 'remove',
                        'protocol': protocol,
                        'current_percentage': current_alloc['allocation_percentage'],
                        'reason': 'No longer optimal or available'
                    })
                    rebalance_needed = True
            
            return {
                'rebalance_needed': rebalance_needed,
                'suggestions': suggestions,
                'optimal_apy': float(optimal_result.total_apy),
                'current_risk_score': optimal_result.risk_score,
                'estimated_gas_cost': float(optimal_result.gas_costs),
                'confidence_score': optimal_result.confidence_score
            }
            
        except Exception as e:
            self.logger.error(f"Error suggesting rebalancing: {e}")
            return {'rebalance_needed': False, 'suggestions': [], 'error': str(e)}
    
    async def shutdown(self):
        """Gracefully shutdown yield optimizer"""
        self.logger.info("Shutting down yield optimizer...")
        self.opportunities.clear()
        self.logger.info("Yield optimizer shutdown complete")

# Export main classes
__all__ = [
    'YieldOptimizer', 'YieldOpportunity', 'OptimizationResult', 
    'UserProfile', 'YieldStrategy', 'RiskLevel'
]