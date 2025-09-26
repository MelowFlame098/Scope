"""DeFi Liquidity Manager

This module manages liquidity provision across multiple DeFi protocols, including
liquidity pool analysis, position monitoring, impermanent loss tracking, and
automated liquidity optimization strategies.

Features:
- Multi-protocol liquidity management
- Impermanent loss monitoring and protection
- Automated position rebalancing
- Liquidity pool analytics
- Fee optimization
- Risk management

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

class PoolType(Enum):
    """Types of liquidity pools"""
    CONSTANT_PRODUCT = "constant_product"  # Uniswap V2 style
    CONCENTRATED = "concentrated"  # Uniswap V3 style
    STABLE = "stable"  # Curve style
    WEIGHTED = "weighted"  # Balancer style
    ORDERBOOK = "orderbook"  # dYdX style

class PositionStatus(Enum):
    """Status of liquidity positions"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    WITHDRAWN = "withdrawn"
    ERROR = "error"

@dataclass
class LiquidityPool:
    """Represents a liquidity pool"""
    protocol: str
    pool_address: str
    token_a: str
    token_b: str
    pool_type: PoolType
    fee_tier: Decimal
    tvl: Decimal
    volume_24h: Decimal
    apy: Decimal
    reserve_a: Decimal
    reserve_b: Decimal
    price_range: Optional[Tuple[Decimal, Decimal]] = None  # For concentrated liquidity
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class LiquidityPosition:
    """Represents a user's liquidity position"""
    position_id: str
    pool: LiquidityPool
    user_address: str
    amount_a: Decimal
    amount_b: Decimal
    liquidity_tokens: Decimal
    entry_price_a: Decimal
    entry_price_b: Decimal
    current_price_a: Decimal
    current_price_b: Decimal
    fees_earned: Decimal
    impermanent_loss: Decimal
    status: PositionStatus
    created_at: datetime
    last_updated: datetime = field(default_factory=datetime.now)
    price_range: Optional[Tuple[Decimal, Decimal]] = None  # For concentrated liquidity

@dataclass
class ImpermanentLossAnalysis:
    """Analysis of impermanent loss for a position"""
    current_il_percentage: Decimal
    current_il_usd: Decimal
    max_il_percentage: Decimal
    max_il_usd: Decimal
    break_even_price_ratio: Decimal
    fees_vs_il_ratio: Decimal
    net_performance: Decimal
    risk_level: str

@dataclass
class LiquidityStrategy:
    """Liquidity provision strategy"""
    name: str
    description: str
    target_pools: List[str]
    rebalance_threshold: Decimal
    il_protection_threshold: Decimal
    fee_optimization: bool
    auto_compound: bool
    risk_parameters: Dict[str, Any]

class LiquidityManager:
    """Advanced DeFi Liquidity Manager
    
    Manages liquidity positions across multiple protocols with intelligent
    monitoring, optimization, and risk management capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Liquidity Manager
        
        Args:
            config: Configuration dictionary containing:
                - protocols: Supported protocols configuration
                - monitoring_settings: Position monitoring parameters
                - risk_settings: Risk management parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.LiquidityManager")
        
        # Data storage
        self.pools: Dict[str, LiquidityPool] = {}
        self.positions: Dict[str, LiquidityPosition] = {}
        self.strategies: Dict[str, LiquidityStrategy] = {}
        
        # Cache settings
        self.cache_ttl = timedelta(minutes=5)
        self.last_pool_update = datetime.min
        
        # Monitoring settings
        self.monitoring_settings = config.get('monitoring_settings', {
            'update_interval': 300,  # 5 minutes
            'il_alert_threshold': 0.05,  # 5%
            'fee_collection_threshold': 0.01,  # 1%
            'rebalance_threshold': 0.1  # 10%
        })
        
        # Risk settings
        self.risk_settings = config.get('risk_settings', {
            'max_il_tolerance': 0.2,  # 20%
            'min_fee_apy': 0.05,  # 5%
            'max_position_size': 0.3,  # 30% of portfolio
            'diversification_requirement': 3  # Minimum 3 different pools
        })
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        self.logger.info("Liquidity Manager initialized")
    
    def _initialize_default_strategies(self):
        """Initialize default liquidity strategies"""
        strategies = [
            LiquidityStrategy(
                name="conservative",
                description="Low-risk stable coin pairs with minimal IL",
                target_pools=["USDC-USDT", "USDC-DAI", "USDT-DAI"],
                rebalance_threshold=Decimal('0.05'),
                il_protection_threshold=Decimal('0.02'),
                fee_optimization=True,
                auto_compound=True,
                risk_parameters={'max_il': 0.05, 'min_tvl': 10000000}
            ),
            LiquidityStrategy(
                name="moderate",
                description="Balanced approach with major token pairs",
                target_pools=["ETH-USDC", "BTC-USDC", "ETH-USDT"],
                rebalance_threshold=Decimal('0.1'),
                il_protection_threshold=Decimal('0.1'),
                fee_optimization=True,
                auto_compound=True,
                risk_parameters={'max_il': 0.15, 'min_tvl': 5000000}
            ),
            LiquidityStrategy(
                name="aggressive",
                description="High-yield opportunities with higher risk",
                target_pools=["ETH-ALT", "BTC-ALT", "ALT-USDC"],
                rebalance_threshold=Decimal('0.15'),
                il_protection_threshold=Decimal('0.2'),
                fee_optimization=True,
                auto_compound=False,
                risk_parameters={'max_il': 0.3, 'min_tvl': 1000000}
            )
        ]
        
        for strategy in strategies:
            self.strategies[strategy.name] = strategy
    
    async def get_available_pools(self, force_refresh: bool = False) -> List[LiquidityPool]:
        """Get available liquidity pools
        
        Args:
            force_refresh: Force refresh of pool data
            
        Returns:
            List of available liquidity pools
        """
        try:
            if not force_refresh and self._is_pool_cache_valid():
                return list(self.pools.values())
            
            await self._fetch_pools()
            return list(self.pools.values())
            
        except Exception as e:
            self.logger.error(f"Error getting available pools: {e}")
            return []
    
    async def _fetch_pools(self):
        """Fetch liquidity pools from various protocols"""
        self.logger.info("Fetching liquidity pools...")
        
        # Mock pools for demonstration
        mock_pools = [
            LiquidityPool(
                protocol="uniswap_v3",
                pool_address="0x1111111111111111111111111111111111111111",
                token_a="USDC",
                token_b="ETH",
                pool_type=PoolType.CONCENTRATED,
                fee_tier=Decimal('0.003'),  # 0.3%
                tvl=Decimal('50000000'),
                volume_24h=Decimal('10000000'),
                apy=Decimal('15.5'),
                reserve_a=Decimal('25000000'),
                reserve_b=Decimal('12500'),
                price_range=(Decimal('1800'), Decimal('2200'))
            ),
            LiquidityPool(
                protocol="uniswap_v2",
                pool_address="0x2222222222222222222222222222222222222222",
                token_a="USDC",
                token_b="USDT",
                pool_type=PoolType.CONSTANT_PRODUCT,
                fee_tier=Decimal('0.003'),
                tvl=Decimal('100000000'),
                volume_24h=Decimal('5000000'),
                apy=Decimal('8.2'),
                reserve_a=Decimal('50000000'),
                reserve_b=Decimal('50000000')
            ),
            LiquidityPool(
                protocol="curve",
                pool_address="0x3333333333333333333333333333333333333333",
                token_a="USDC",
                token_b="USDT",
                pool_type=PoolType.STABLE,
                fee_tier=Decimal('0.0004'),  # 0.04%
                tvl=Decimal('200000000'),
                volume_24h=Decimal('20000000'),
                apy=Decimal('12.8'),
                reserve_a=Decimal('100000000'),
                reserve_b=Decimal('100000000')
            ),
            LiquidityPool(
                protocol="balancer",
                pool_address="0x4444444444444444444444444444444444444444",
                token_a="ETH",
                token_b="BTC",
                pool_type=PoolType.WEIGHTED,
                fee_tier=Decimal('0.01'),  # 1%
                tvl=Decimal('30000000'),
                volume_24h=Decimal('3000000'),
                apy=Decimal('18.5'),
                reserve_a=Decimal('7500'),
                reserve_b=Decimal('500')
            ),
            LiquidityPool(
                protocol="sushiswap",
                pool_address="0x5555555555555555555555555555555555555555",
                token_a="SUSHI",
                token_b="ETH",
                pool_type=PoolType.CONSTANT_PRODUCT,
                fee_tier=Decimal('0.003'),
                tvl=Decimal('15000000'),
                volume_24h=Decimal('2000000'),
                apy=Decimal('25.3'),
                reserve_a=Decimal('5000000'),
                reserve_b=Decimal('2500')
            )
        ]
        
        # Store pools
        self.pools.clear()
        for pool in mock_pools:
            key = f"{pool.protocol}_{pool.pool_address}"
            self.pools[key] = pool
        
        self.last_pool_update = datetime.now()
        self.logger.info(f"Fetched {len(mock_pools)} liquidity pools")
    
    def _is_pool_cache_valid(self) -> bool:
        """Check if pool cache is still valid"""
        return datetime.now() - self.last_pool_update < self.cache_ttl
    
    async def add_liquidity(self, 
                           pool_address: str, 
                           amount_a: Decimal, 
                           amount_b: Decimal,
                           user_address: str,
                           price_range: Optional[Tuple[Decimal, Decimal]] = None) -> Dict[str, Any]:
        """Add liquidity to a pool
        
        Args:
            pool_address: Pool contract address
            amount_a: Amount of token A
            amount_b: Amount of token B
            user_address: User's wallet address
            price_range: Price range for concentrated liquidity (optional)
            
        Returns:
            Transaction result and position information
        """
        try:
            # Find the pool
            pool = None
            for p in self.pools.values():
                if p.pool_address == pool_address:
                    pool = p
                    break
            
            if not pool:
                return {
                    'success': False,
                    'error': 'Pool not found',
                    'transaction_hash': None
                }
            
            # Validate amounts
            if amount_a <= 0 or amount_b <= 0:
                return {
                    'success': False,
                    'error': 'Invalid amounts',
                    'transaction_hash': None
                }
            
            # Calculate liquidity tokens (simplified)
            liquidity_tokens = (amount_a * amount_b) ** Decimal('0.5')
            
            # Create position
            position_id = f"pos_{user_address}_{pool_address}_{datetime.now().timestamp()}"
            position = LiquidityPosition(
                position_id=position_id,
                pool=pool,
                user_address=user_address,
                amount_a=amount_a,
                amount_b=amount_b,
                liquidity_tokens=liquidity_tokens,
                entry_price_a=pool.reserve_b / pool.reserve_a if pool.reserve_a > 0 else Decimal('1'),
                entry_price_b=pool.reserve_a / pool.reserve_b if pool.reserve_b > 0 else Decimal('1'),
                current_price_a=pool.reserve_b / pool.reserve_a if pool.reserve_a > 0 else Decimal('1'),
                current_price_b=pool.reserve_a / pool.reserve_b if pool.reserve_b > 0 else Decimal('1'),
                fees_earned=Decimal('0'),
                impermanent_loss=Decimal('0'),
                status=PositionStatus.ACTIVE,
                created_at=datetime.now(),
                price_range=price_range
            )
            
            # Store position
            self.positions[position_id] = position
            
            # Mock transaction hash
            tx_hash = f"0x{'1' * 64}"
            
            self.logger.info(f"Added liquidity position {position_id} to {pool.protocol}")
            
            return {
                'success': True,
                'position_id': position_id,
                'transaction_hash': tx_hash,
                'liquidity_tokens': float(liquidity_tokens),
                'estimated_apy': float(pool.apy),
                'pool_info': {
                    'protocol': pool.protocol,
                    'token_pair': f"{pool.token_a}-{pool.token_b}",
                    'fee_tier': float(pool.fee_tier),
                    'tvl': float(pool.tvl)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error adding liquidity: {e}")
            return {
                'success': False,
                'error': str(e),
                'transaction_hash': None
            }
    
    async def remove_liquidity(self, position_id: str, percentage: Decimal = Decimal('1.0')) -> Dict[str, Any]:
        """Remove liquidity from a position
        
        Args:
            position_id: Position identifier
            percentage: Percentage of position to remove (0.0 to 1.0)
            
        Returns:
            Transaction result
        """
        try:
            if position_id not in self.positions:
                return {
                    'success': False,
                    'error': 'Position not found',
                    'transaction_hash': None
                }
            
            position = self.positions[position_id]
            
            if position.status != PositionStatus.ACTIVE:
                return {
                    'success': False,
                    'error': 'Position is not active',
                    'transaction_hash': None
                }
            
            # Validate percentage
            if percentage <= 0 or percentage > 1:
                return {
                    'success': False,
                    'error': 'Invalid percentage',
                    'transaction_hash': None
                }
            
            # Calculate amounts to remove
            remove_amount_a = position.amount_a * percentage
            remove_amount_b = position.amount_b * percentage
            remove_liquidity = position.liquidity_tokens * percentage
            
            # Update position
            if percentage == Decimal('1.0'):
                position.status = PositionStatus.WITHDRAWN
            else:
                position.amount_a -= remove_amount_a
                position.amount_b -= remove_amount_b
                position.liquidity_tokens -= remove_liquidity
            
            position.last_updated = datetime.now()
            
            # Mock transaction hash
            tx_hash = f"0x{'2' * 64}"
            
            self.logger.info(f"Removed {float(percentage * 100)}% liquidity from position {position_id}")
            
            return {
                'success': True,
                'transaction_hash': tx_hash,
                'removed_amount_a': float(remove_amount_a),
                'removed_amount_b': float(remove_amount_b),
                'fees_collected': float(position.fees_earned * percentage),
                'remaining_liquidity': float(position.liquidity_tokens) if percentage < 1 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error removing liquidity: {e}")
            return {
                'success': False,
                'error': str(e),
                'transaction_hash': None
            }
    
    async def get_user_positions(self, user_address: str) -> List[Dict[str, Any]]:
        """Get all positions for a user
        
        Args:
            user_address: User's wallet address
            
        Returns:
            List of user's liquidity positions
        """
        try:
            user_positions = []
            
            for position in self.positions.values():
                if position.user_address.lower() == user_address.lower():
                    # Update position with current data
                    await self._update_position(position)
                    
                    # Calculate current value
                    current_value_a = position.amount_a * position.current_price_a
                    current_value_b = position.amount_b * position.current_price_b
                    total_value = current_value_a + current_value_b
                    
                    # Calculate IL analysis
                    il_analysis = await self._calculate_impermanent_loss(position)
                    
                    user_positions.append({
                        'position_id': position.position_id,
                        'protocol': position.pool.protocol,
                        'token_pair': f"{position.pool.token_a}-{position.pool.token_b}",
                        'pool_address': position.pool.pool_address,
                        'amount_a': float(position.amount_a),
                        'amount_b': float(position.amount_b),
                        'liquidity_tokens': float(position.liquidity_tokens),
                        'current_value_usd': float(total_value),
                        'fees_earned': float(position.fees_earned),
                        'impermanent_loss': {
                            'percentage': float(il_analysis.current_il_percentage),
                            'usd_amount': float(il_analysis.current_il_usd),
                            'net_performance': float(il_analysis.net_performance)
                        },
                        'apy': float(position.pool.apy),
                        'status': position.status.value,
                        'created_at': position.created_at.isoformat(),
                        'last_updated': position.last_updated.isoformat(),
                        'price_range': position.price_range if position.price_range else None
                    })
            
            return user_positions
            
        except Exception as e:
            self.logger.error(f"Error getting user positions: {e}")
            return []
    
    async def _update_position(self, position: LiquidityPosition):
        """Update position with current market data"""
        try:
            # Update current prices (mock implementation)
            # In real implementation, this would fetch from price oracles
            
            # Simulate price changes
            import random
            price_change = Decimal(str(random.uniform(-0.05, 0.05)))  # ±5% change
            
            position.current_price_a = position.entry_price_a * (1 + price_change)
            position.current_price_b = position.entry_price_b * (1 - price_change)
            
            # Update fees earned (simplified)
            days_active = (datetime.now() - position.created_at).days
            if days_active > 0:
                daily_fee_rate = position.pool.apy / Decimal('365') / Decimal('100')
                total_value = position.amount_a + position.amount_b
                position.fees_earned = total_value * daily_fee_rate * Decimal(str(days_active))
            
            position.last_updated = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error updating position: {e}")
    
    async def _calculate_impermanent_loss(self, position: LiquidityPosition) -> ImpermanentLossAnalysis:
        """Calculate impermanent loss for a position"""
        try:
            # Calculate price ratio change
            entry_ratio = position.entry_price_a / position.entry_price_b
            current_ratio = position.current_price_a / position.current_price_b
            ratio_change = current_ratio / entry_ratio
            
            # Calculate impermanent loss percentage
            # IL = 2 * sqrt(ratio) / (1 + ratio) - 1
            sqrt_ratio = ratio_change ** Decimal('0.5')
            il_multiplier = 2 * sqrt_ratio / (1 + ratio_change) - 1
            il_percentage = abs(il_multiplier) * 100
            
            # Calculate IL in USD
            initial_value = position.amount_a + position.amount_b
            il_usd = initial_value * abs(il_multiplier)
            
            # Calculate net performance (fees - IL)
            net_performance = position.fees_earned - il_usd
            
            # Determine risk level
            if il_percentage < 2:
                risk_level = "LOW"
            elif il_percentage < 5:
                risk_level = "MEDIUM"
            elif il_percentage < 10:
                risk_level = "HIGH"
            else:
                risk_level = "VERY_HIGH"
            
            return ImpermanentLossAnalysis(
                current_il_percentage=il_percentage,
                current_il_usd=il_usd,
                max_il_percentage=il_percentage,  # Simplified
                max_il_usd=il_usd,
                break_even_price_ratio=entry_ratio,
                fees_vs_il_ratio=position.fees_earned / il_usd if il_usd > 0 else Decimal('inf'),
                net_performance=net_performance,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating impermanent loss: {e}")
            return ImpermanentLossAnalysis(
                current_il_percentage=Decimal('0'),
                current_il_usd=Decimal('0'),
                max_il_percentage=Decimal('0'),
                max_il_usd=Decimal('0'),
                break_even_price_ratio=Decimal('1'),
                fees_vs_il_ratio=Decimal('1'),
                net_performance=Decimal('0'),
                risk_level="UNKNOWN"
            )
    
    async def analyze_pool_performance(self, pool_address: str, days: int = 30) -> Dict[str, Any]:
        """Analyze historical performance of a liquidity pool
        
        Args:
            pool_address: Pool contract address
            days: Number of days to analyze
            
        Returns:
            Pool performance analysis
        """
        try:
            # Find the pool
            pool = None
            for p in self.pools.values():
                if p.pool_address == pool_address:
                    pool = p
                    break
            
            if not pool:
                return {'error': 'Pool not found'}
            
            # Mock historical data
            daily_volumes = [float(pool.volume_24h) * random.uniform(0.5, 1.5) for _ in range(days)]
            daily_fees = [vol * float(pool.fee_tier) for vol in daily_volumes]
            daily_apy = [float(pool.apy) * random.uniform(0.8, 1.2) for _ in range(days)]
            
            avg_volume = sum(daily_volumes) / len(daily_volumes)
            avg_fees = sum(daily_fees) / len(daily_fees)
            avg_apy = sum(daily_apy) / len(daily_apy)
            
            # Calculate volatility
            volume_volatility = np.std(daily_volumes) / avg_volume if avg_volume > 0 else 0
            apy_volatility = np.std(daily_apy) / avg_apy if avg_apy > 0 else 0
            
            return {
                'pool_address': pool_address,
                'protocol': pool.protocol,
                'token_pair': f"{pool.token_a}-{pool.token_b}",
                'analysis_period_days': days,
                'current_metrics': {
                    'tvl': float(pool.tvl),
                    'volume_24h': float(pool.volume_24h),
                    'apy': float(pool.apy),
                    'fee_tier': float(pool.fee_tier)
                },
                'historical_averages': {
                    'avg_daily_volume': avg_volume,
                    'avg_daily_fees': avg_fees,
                    'avg_apy': avg_apy
                },
                'volatility_metrics': {
                    'volume_volatility': volume_volatility,
                    'apy_volatility': apy_volatility
                },
                'risk_assessment': {
                    'liquidity_risk': 'LOW' if pool.tvl > 50000000 else 'MEDIUM' if pool.tvl > 10000000 else 'HIGH',
                    'impermanent_loss_risk': 'LOW' if pool.pool_type == PoolType.STABLE else 'MEDIUM' if 'USD' in f"{pool.token_a}{pool.token_b}" else 'HIGH',
                    'protocol_risk': 'LOW' if pool.protocol in ['uniswap_v2', 'uniswap_v3', 'curve'] else 'MEDIUM'
                },
                'recommendations': self._generate_pool_recommendations(pool, avg_apy, volume_volatility)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing pool performance: {e}")
            return {'error': str(e)}
    
    def _generate_pool_recommendations(self, pool: LiquidityPool, avg_apy: float, volatility: float) -> List[str]:
        """Generate recommendations for a pool"""
        recommendations = []
        
        if avg_apy > 20:
            recommendations.append("High APY - monitor for sustainability")
        
        if volatility > 0.5:
            recommendations.append("High volatility - consider smaller position size")
        
        if pool.tvl < 1000000:
            recommendations.append("Low TVL - higher slippage risk")
        
        if pool.pool_type == PoolType.STABLE:
            recommendations.append("Stable pair - low impermanent loss risk")
        
        if pool.volume_24h / pool.tvl > 0.5:
            recommendations.append("High volume/TVL ratio - good fee generation")
        
        return recommendations
    
    async def suggest_rebalancing(self, user_address: str) -> Dict[str, Any]:
        """Suggest position rebalancing for a user
        
        Args:
            user_address: User's wallet address
            
        Returns:
            Rebalancing suggestions
        """
        try:
            positions = await self.get_user_positions(user_address)
            
            if not positions:
                return {'rebalance_needed': False, 'suggestions': []}
            
            suggestions = []
            
            for position in positions:
                # Check for high impermanent loss
                il_percentage = position['impermanent_loss']['percentage']
                if il_percentage > self.risk_settings['max_il_tolerance'] * 100:
                    suggestions.append({
                        'position_id': position['position_id'],
                        'action': 'reduce_or_exit',
                        'reason': f"High impermanent loss: {il_percentage:.2f}%",
                        'urgency': 'HIGH'
                    })
                
                # Check for low fee generation
                net_performance = position['impermanent_loss']['net_performance']
                if net_performance < 0:
                    suggestions.append({
                        'position_id': position['position_id'],
                        'action': 'monitor_closely',
                        'reason': f"Negative net performance: ${net_performance:.2f}",
                        'urgency': 'MEDIUM'
                    })
                
                # Check for out-of-range positions (concentrated liquidity)
                if position['price_range']:
                    # This would check if current price is outside the range
                    suggestions.append({
                        'position_id': position['position_id'],
                        'action': 'adjust_range',
                        'reason': "Consider adjusting price range for better capital efficiency",
                        'urgency': 'LOW'
                    })
            
            return {
                'rebalance_needed': len(suggestions) > 0,
                'suggestions': suggestions,
                'total_positions': len(positions),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error suggesting rebalancing: {e}")
            return {'rebalance_needed': False, 'suggestions': [], 'error': str(e)}
    
    async def get_optimal_pools(self, strategy_name: str, investment_amount: Decimal) -> List[Dict[str, Any]]:
        """Get optimal pools for a given strategy
        
        Args:
            strategy_name: Name of the strategy
            investment_amount: Amount to invest
            
        Returns:
            List of optimal pools with allocation suggestions
        """
        try:
            if strategy_name not in self.strategies:
                return []
            
            strategy = self.strategies[strategy_name]
            pools = await self.get_available_pools()
            
            # Filter pools based on strategy
            suitable_pools = []
            for pool in pools:
                token_pair = f"{pool.token_a}-{pool.token_b}"
                
                # Check if pool matches strategy targets
                for target in strategy.target_pools:
                    if target in token_pair or any(token in target for token in [pool.token_a, pool.token_b]):
                        # Apply strategy risk parameters
                        if pool.tvl >= strategy.risk_parameters.get('min_tvl', 0):
                            suitable_pools.append(pool)
                        break
            
            # Sort by APY and risk
            suitable_pools.sort(key=lambda p: float(p.apy), reverse=True)
            
            # Calculate allocations
            optimal_pools = []
            total_pools = min(len(suitable_pools), 3)  # Max 3 pools
            
            for i, pool in enumerate(suitable_pools[:total_pools]):
                # Simple allocation: decreasing weights
                weights = [0.5, 0.3, 0.2]
                allocation_pct = weights[i] if i < len(weights) else 0.1
                allocation_amount = investment_amount * Decimal(str(allocation_pct))
                
                optimal_pools.append({
                    'pool_address': pool.pool_address,
                    'protocol': pool.protocol,
                    'token_pair': f"{pool.token_a}-{pool.token_b}",
                    'allocation_percentage': allocation_pct * 100,
                    'allocation_amount': float(allocation_amount),
                    'expected_apy': float(pool.apy),
                    'tvl': float(pool.tvl),
                    'fee_tier': float(pool.fee_tier),
                    'risk_level': self._assess_pool_risk(pool),
                    'estimated_daily_fees': float(allocation_amount * pool.apy / Decimal('365') / Decimal('100'))
                })
            
            return optimal_pools
            
        except Exception as e:
            self.logger.error(f"Error getting optimal pools: {e}")
            return []
    
    def _assess_pool_risk(self, pool: LiquidityPool) -> str:
        """Assess risk level of a pool"""
        risk_score = 0
        
        # TVL risk
        if pool.tvl < 1000000:
            risk_score += 3
        elif pool.tvl < 10000000:
            risk_score += 2
        else:
            risk_score += 1
        
        # Pool type risk
        if pool.pool_type == PoolType.STABLE:
            risk_score += 1
        elif pool.pool_type == PoolType.CONSTANT_PRODUCT:
            risk_score += 2
        else:
            risk_score += 3
        
        # Token risk
        stable_tokens = ['USDC', 'USDT', 'DAI', 'BUSD']
        if pool.token_a in stable_tokens and pool.token_b in stable_tokens:
            risk_score += 1
        elif pool.token_a in stable_tokens or pool.token_b in stable_tokens:
            risk_score += 2
        else:
            risk_score += 3
        
        if risk_score <= 4:
            return "LOW"
        elif risk_score <= 6:
            return "MEDIUM"
        else:
            return "HIGH"
    
    async def shutdown(self):
        """Gracefully shutdown liquidity manager"""
        self.logger.info("Shutting down liquidity manager...")
        self.pools.clear()
        self.positions.clear()
        self.strategies.clear()
        self.logger.info("Liquidity manager shutdown complete")

# Export main classes
__all__ = [
    'LiquidityManager', 'LiquidityPool', 'LiquidityPosition', 
    'ImpermanentLossAnalysis', 'LiquidityStrategy', 'PoolType', 'PositionStatus'
]