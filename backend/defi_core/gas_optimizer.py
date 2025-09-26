"""Gas Optimizer

This module provides intelligent gas optimization capabilities for DeFi transactions,
helping users minimize transaction costs through gas price analysis, timing optimization,
and transaction batching strategies.

Features:
- Real-time gas price monitoring
- Gas price prediction
- Optimal timing recommendations
- Transaction batching
- MEV protection strategies
- Cross-chain gas comparison
- Gas-efficient route planning

Supported Networks:
- Ethereum Mainnet
- Polygon
- Arbitrum
- Optimism
- Avalanche
- BSC
- Fantom

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
import statistics
from collections import defaultdict, deque
import hashlib

class GasPriority(Enum):
    """Gas price priority levels"""
    SLOW = "slow"
    STANDARD = "standard"
    FAST = "fast"
    INSTANT = "instant"

class TransactionType(Enum):
    """Types of DeFi transactions"""
    SWAP = "swap"
    LIQUIDITY_ADD = "liquidity_add"
    LIQUIDITY_REMOVE = "liquidity_remove"
    STAKE = "stake"
    UNSTAKE = "unstake"
    CLAIM_REWARDS = "claim_rewards"
    BRIDGE = "bridge"
    LENDING = "lending"
    BORROWING = "borrowing"
    NFT_TRADE = "nft_trade"
    GOVERNANCE = "governance"

class NetworkId(Enum):
    """Supported blockchain networks"""
    ETHEREUM = 1
    BSC = 56
    POLYGON = 137
    AVALANCHE = 43114
    FANTOM = 250
    ARBITRUM = 42161
    OPTIMISM = 10
    HARMONY = 1666600000
    MOONBEAM = 1284
    CRONOS = 25

@dataclass
class GasPrice:
    """Gas price information for a network"""
    network_id: NetworkId
    slow: Decimal
    standard: Decimal
    fast: Decimal
    instant: Decimal
    base_fee: Optional[Decimal] = None
    priority_fee: Optional[Decimal] = None
    timestamp: datetime = field(default_factory=datetime.now)
    block_number: Optional[int] = None
    network_congestion: float = 0.0  # 0-1 scale

@dataclass
class GasEstimate:
    """Gas usage estimate for a transaction"""
    transaction_type: TransactionType
    network_id: NetworkId
    estimated_gas: int
    gas_price: GasPrice
    total_cost_usd: Decimal
    confidence_score: float
    factors: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class OptimizationRecommendation:
    """Gas optimization recommendation"""
    recommendation_type: str
    description: str
    potential_savings_usd: Decimal
    potential_savings_percentage: float
    implementation_difficulty: str  # "easy", "medium", "hard"
    time_sensitive: bool
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchTransaction:
    """Batched transaction for gas optimization"""
    batch_id: str
    transactions: List[Dict[str, Any]]
    estimated_gas_savings: Decimal
    estimated_time_savings: int  # minutes
    total_gas_cost: Decimal
    individual_gas_cost: Decimal
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None

class GasOptimizer:
    """Advanced Gas Optimization Engine
    
    Provides intelligent gas optimization strategies to minimize transaction costs
    across multiple blockchain networks through price analysis, timing optimization,
    and transaction batching.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Gas Optimizer
        
        Args:
            config: Configuration dictionary containing:
                - gas_apis: Gas price API configurations
                - optimization_settings: Optimization parameters
                - network_settings: Network-specific settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.GasOptimizer")
        
        # Gas price data storage
        self.gas_prices: Dict[NetworkId, GasPrice] = {}
        self.price_history: Dict[NetworkId, deque] = defaultdict(lambda: deque(maxlen=1440))  # 24 hours of minute data
        
        # Transaction estimates and batches
        self.gas_estimates: Dict[str, GasEstimate] = {}
        self.batch_transactions: Dict[str, BatchTransaction] = {}
        
        # Optimization settings
        self.optimization_settings = config.get('optimization_settings', {
            'max_wait_time_minutes': 60,
            'min_savings_threshold': 0.05,  # 5%
            'batch_timeout_minutes': 30,
            'prediction_confidence_threshold': 0.7
        })
        
        # Network settings
        self.network_settings = config.get('network_settings', {
            NetworkId.ETHEREUM: {
                'base_gas_estimates': {
                    TransactionType.SWAP: 150000,
                    TransactionType.LIQUIDITY_ADD: 200000,
                    TransactionType.LIQUIDITY_REMOVE: 180000,
                    TransactionType.STAKE: 120000,
                    TransactionType.UNSTAKE: 100000,
                    TransactionType.CLAIM_REWARDS: 80000,
                    TransactionType.BRIDGE: 250000,
                    TransactionType.LENDING: 180000,
                    TransactionType.BORROWING: 200000,
                    TransactionType.NFT_TRADE: 100000,
                    TransactionType.GOVERNANCE: 90000
                },
                'gas_multiplier': 1.0,
                'native_token_price_usd': 2000
            },
            NetworkId.POLYGON: {
                'base_gas_estimates': {
                    TransactionType.SWAP: 120000,
                    TransactionType.LIQUIDITY_ADD: 160000,
                    TransactionType.LIQUIDITY_REMOVE: 140000,
                    TransactionType.STAKE: 100000,
                    TransactionType.UNSTAKE: 80000,
                    TransactionType.CLAIM_REWARDS: 60000,
                    TransactionType.BRIDGE: 200000,
                    TransactionType.LENDING: 140000,
                    TransactionType.BORROWING: 160000,
                    TransactionType.NFT_TRADE: 80000,
                    TransactionType.GOVERNANCE: 70000
                },
                'gas_multiplier': 1.2,
                'native_token_price_usd': 0.8
            },
            NetworkId.ARBITRUM: {
                'base_gas_estimates': {
                    TransactionType.SWAP: 100000,
                    TransactionType.LIQUIDITY_ADD: 140000,
                    TransactionType.LIQUIDITY_REMOVE: 120000,
                    TransactionType.STAKE: 80000,
                    TransactionType.UNSTAKE: 70000,
                    TransactionType.CLAIM_REWARDS: 50000,
                    TransactionType.BRIDGE: 180000,
                    TransactionType.LENDING: 120000,
                    TransactionType.BORROWING: 140000,
                    TransactionType.NFT_TRADE: 70000,
                    TransactionType.GOVERNANCE: 60000
                },
                'gas_multiplier': 0.8,
                'native_token_price_usd': 2000
            }
        })
        
        # Initialize gas price monitoring
        self._initialize_gas_monitoring()
        
        self.logger.info("Gas Optimizer initialized")
    
    def _initialize_gas_monitoring(self):
        """Initialize gas price monitoring for supported networks"""
        # Mock initial gas prices
        initial_prices = {
            NetworkId.ETHEREUM: GasPrice(
                network_id=NetworkId.ETHEREUM,
                slow=Decimal('20'),
                standard=Decimal('30'),
                fast=Decimal('45'),
                instant=Decimal('60'),
                base_fee=Decimal('25'),
                priority_fee=Decimal('2'),
                network_congestion=0.6
            ),
            NetworkId.POLYGON: GasPrice(
                network_id=NetworkId.POLYGON,
                slow=Decimal('30'),
                standard=Decimal('35'),
                fast=Decimal('45'),
                instant=Decimal('60'),
                network_congestion=0.3
            ),
            NetworkId.ARBITRUM: GasPrice(
                network_id=NetworkId.ARBITRUM,
                slow=Decimal('0.1'),
                standard=Decimal('0.2'),
                fast=Decimal('0.3'),
                instant=Decimal('0.5'),
                network_congestion=0.2
            ),
            NetworkId.OPTIMISM: GasPrice(
                network_id=NetworkId.OPTIMISM,
                slow=Decimal('0.001'),
                standard=Decimal('0.002'),
                fast=Decimal('0.003'),
                instant=Decimal('0.005'),
                network_congestion=0.1
            )
        }
        
        for network_id, gas_price in initial_prices.items():
            self.gas_prices[network_id] = gas_price
            self.price_history[network_id].append({
                'timestamp': datetime.now(),
                'standard_price': gas_price.standard,
                'congestion': gas_price.network_congestion
            })
        
        self.logger.info(f"Initialized gas monitoring for {len(initial_prices)} networks")
    
    async def get_current_gas_prices(self, network_id: NetworkId) -> Optional[GasPrice]:
        """Get current gas prices for a network
        
        Args:
            network_id: Target blockchain network
            
        Returns:
            Current gas price information
        """
        try:
            # In a real implementation, this would fetch from gas APIs
            # For now, return cached/mock data with some variation
            if network_id not in self.gas_prices:
                return None
            
            base_price = self.gas_prices[network_id]
            
            # Add some realistic variation
            import random
            variation = random.uniform(0.9, 1.1)
            
            current_price = GasPrice(
                network_id=network_id,
                slow=base_price.slow * Decimal(str(variation * 0.8)),
                standard=base_price.standard * Decimal(str(variation)),
                fast=base_price.fast * Decimal(str(variation * 1.2)),
                instant=base_price.instant * Decimal(str(variation * 1.5)),
                base_fee=base_price.base_fee * Decimal(str(variation)) if base_price.base_fee else None,
                priority_fee=base_price.priority_fee * Decimal(str(variation)) if base_price.priority_fee else None,
                timestamp=datetime.now(),
                network_congestion=min(1.0, base_price.network_congestion * variation)
            )
            
            # Update cache
            self.gas_prices[network_id] = current_price
            
            # Add to history
            self.price_history[network_id].append({
                'timestamp': current_price.timestamp,
                'standard_price': current_price.standard,
                'congestion': current_price.network_congestion
            })
            
            return current_price
            
        except Exception as e:
            self.logger.error(f"Error getting gas prices for {network_id}: {e}")
            return None
    
    async def estimate_transaction_gas(self, 
                                     transaction_type: TransactionType,
                                     network_id: NetworkId,
                                     priority: GasPriority = GasPriority.STANDARD,
                                     transaction_data: Optional[Dict[str, Any]] = None) -> Optional[GasEstimate]:
        """Estimate gas cost for a transaction
        
        Args:
            transaction_type: Type of DeFi transaction
            network_id: Target blockchain network
            priority: Gas price priority level
            transaction_data: Additional transaction parameters
            
        Returns:
            Gas cost estimate
        """
        try:
            # Get current gas prices
            gas_price = await self.get_current_gas_prices(network_id)
            if not gas_price:
                return None
            
            # Get base gas estimate
            network_config = self.network_settings.get(network_id, {})
            base_estimates = network_config.get('base_gas_estimates', {})
            base_gas = base_estimates.get(transaction_type, 150000)
            
            # Apply network-specific multiplier
            gas_multiplier = network_config.get('gas_multiplier', 1.0)
            estimated_gas = int(base_gas * gas_multiplier)
            
            # Adjust based on transaction complexity
            if transaction_data:
                estimated_gas = self._adjust_gas_for_complexity(estimated_gas, transaction_data)
            
            # Get gas price based on priority
            priority_prices = {
                GasPriority.SLOW: gas_price.slow,
                GasPriority.STANDARD: gas_price.standard,
                GasPriority.FAST: gas_price.fast,
                GasPriority.INSTANT: gas_price.instant
            }
            
            selected_gas_price = priority_prices[priority]
            
            # Calculate total cost in USD
            native_token_price = network_config.get('native_token_price_usd', 1)
            gas_cost_native = (Decimal(estimated_gas) * selected_gas_price) / Decimal('1e9')  # Convert from gwei
            total_cost_usd = gas_cost_native * Decimal(str(native_token_price))
            
            # Calculate confidence score
            confidence_score = self._calculate_gas_confidence(gas_price, network_id)
            
            # Generate optimization factors
            factors = self._analyze_gas_factors(gas_price, transaction_type, network_id)
            
            # Generate alternatives
            alternatives = await self._generate_gas_alternatives(transaction_type, network_id, gas_price)
            
            estimate = GasEstimate(
                transaction_type=transaction_type,
                network_id=network_id,
                estimated_gas=estimated_gas,
                gas_price=gas_price,
                total_cost_usd=total_cost_usd,
                confidence_score=confidence_score,
                factors=factors,
                alternatives=alternatives
            )
            
            # Cache estimate
            estimate_id = self._generate_estimate_id(transaction_type, network_id, priority)
            self.gas_estimates[estimate_id] = estimate
            
            return estimate
            
        except Exception as e:
            self.logger.error(f"Error estimating gas: {e}")
            return None
    
    def _adjust_gas_for_complexity(self, base_gas: int, transaction_data: Dict[str, Any]) -> int:
        """Adjust gas estimate based on transaction complexity"""
        multiplier = 1.0
        
        # Token count factor
        token_count = transaction_data.get('token_count', 1)
        if token_count > 1:
            multiplier *= (1 + (token_count - 1) * 0.3)
        
        # Slippage tolerance factor
        slippage = transaction_data.get('slippage_tolerance', 0.005)
        if slippage > 0.01:  # High slippage might require more complex routing
            multiplier *= 1.2
        
        # Route complexity factor
        route_hops = transaction_data.get('route_hops', 1)
        if route_hops > 2:
            multiplier *= (1 + (route_hops - 2) * 0.4)
        
        # MEV protection factor
        if transaction_data.get('mev_protection', False):
            multiplier *= 1.1
        
        return int(base_gas * multiplier)
    
    def _calculate_gas_confidence(self, gas_price: GasPrice, network_id: NetworkId) -> float:
        """Calculate confidence score for gas estimate"""
        confidence = 0.8  # Base confidence
        
        # Network congestion factor
        if gas_price.network_congestion < 0.3:
            confidence += 0.1
        elif gas_price.network_congestion > 0.7:
            confidence -= 0.2
        
        # Price stability factor (based on recent history)
        if network_id in self.price_history and len(self.price_history[network_id]) > 10:
            recent_prices = [p['standard_price'] for p in list(self.price_history[network_id])[-10:]]
            price_volatility = statistics.stdev(recent_prices) / statistics.mean(recent_prices)
            
            if price_volatility < 0.1:  # Low volatility
                confidence += 0.1
            elif price_volatility > 0.3:  # High volatility
                confidence -= 0.1
        
        # Data freshness factor
        data_age = (datetime.now() - gas_price.timestamp).total_seconds() / 60  # Minutes
        if data_age > 5:
            confidence -= min(0.2, data_age / 30)
        
        return max(0.1, min(1.0, confidence))
    
    def _analyze_gas_factors(self, gas_price: GasPrice, transaction_type: TransactionType, network_id: NetworkId) -> List[str]:
        """Analyze factors affecting gas costs"""
        factors = []
        
        # Network congestion
        if gas_price.network_congestion > 0.7:
            factors.append("High network congestion - consider waiting or using L2")
        elif gas_price.network_congestion < 0.3:
            factors.append("Low network congestion - good time to transact")
        
        # Gas price level
        if network_id in self.price_history and len(self.price_history[network_id]) > 60:
            recent_avg = statistics.mean([p['standard_price'] for p in list(self.price_history[network_id])[-60:]])
            current_price = gas_price.standard
            
            if current_price > recent_avg * Decimal('1.2'):
                factors.append("Gas prices 20% above recent average")
            elif current_price < recent_avg * Decimal('0.8'):
                factors.append("Gas prices 20% below recent average - good opportunity")
        
        # Transaction type specific factors
        if transaction_type in [TransactionType.SWAP, TransactionType.NFT_TRADE]:
            factors.append("Time-sensitive transaction - consider fast/instant priority")
        elif transaction_type in [TransactionType.CLAIM_REWARDS, TransactionType.GOVERNANCE]:
            factors.append("Non-urgent transaction - can wait for lower gas prices")
        
        # Network specific factors
        if network_id == NetworkId.ETHEREUM:
            factors.append("Consider L2 alternatives for lower costs")
        elif network_id in [NetworkId.POLYGON, NetworkId.ARBITRUM, NetworkId.OPTIMISM]:
            factors.append("Already on low-cost network")
        
        return factors
    
    async def _generate_gas_alternatives(self, 
                                       transaction_type: TransactionType, 
                                       network_id: NetworkId, 
                                       current_gas_price: GasPrice) -> List[Dict[str, Any]]:
        """Generate alternative options for gas optimization"""
        alternatives = []
        
        # Different priority levels
        priorities = [GasPriority.SLOW, GasPriority.STANDARD, GasPriority.FAST, GasPriority.INSTANT]
        current_priority = GasPriority.STANDARD
        
        for priority in priorities:
            if priority != current_priority:
                priority_prices = {
                    GasPriority.SLOW: current_gas_price.slow,
                    GasPriority.STANDARD: current_gas_price.standard,
                    GasPriority.FAST: current_gas_price.fast,
                    GasPriority.INSTANT: current_gas_price.instant
                }
                
                alternatives.append({
                    'type': 'priority_change',
                    'description': f"Use {priority.value} priority",
                    'gas_price': float(priority_prices[priority]),
                    'estimated_time': self._get_priority_time(priority),
                    'cost_difference_percentage': float((priority_prices[priority] - current_gas_price.standard) / current_gas_price.standard * 100)
                })
        
        # Alternative networks
        if network_id == NetworkId.ETHEREUM:
            l2_networks = [NetworkId.POLYGON, NetworkId.ARBITRUM, NetworkId.OPTIMISM]
            for l2_network in l2_networks:
                if l2_network in self.gas_prices:
                    l2_gas = self.gas_prices[l2_network]
                    alternatives.append({
                        'type': 'network_change',
                        'description': f"Use {l2_network.name} instead",
                        'network': l2_network.name,
                        'estimated_savings': 70,  # Typical L2 savings
                        'additional_steps': ['Bridge assets to L2', 'Execute transaction', 'Bridge back if needed']
                    })
        
        # Timing alternatives
        if current_gas_price.network_congestion > 0.5:
            alternatives.append({
                'type': 'timing_optimization',
                'description': 'Wait for lower gas prices',
                'estimated_wait_time': '30-60 minutes',
                'potential_savings': '15-30%',
                'recommendation': 'Monitor gas prices and execute when congestion decreases'
            })
        
        # Batching alternatives
        if transaction_type in [TransactionType.CLAIM_REWARDS, TransactionType.GOVERNANCE]:
            alternatives.append({
                'type': 'batching',
                'description': 'Batch with other transactions',
                'potential_savings': '20-40%',
                'requirements': 'Wait for more transactions to batch'
            })
        
        return alternatives
    
    def _get_priority_time(self, priority: GasPriority) -> str:
        """Get estimated confirmation time for priority level"""
        time_estimates = {
            GasPriority.SLOW: "10-30 minutes",
            GasPriority.STANDARD: "3-5 minutes",
            GasPriority.FAST: "1-2 minutes",
            GasPriority.INSTANT: "< 1 minute"
        }
        return time_estimates.get(priority, "Unknown")
    
    async def get_optimization_recommendations(self, 
                                             user_address: str,
                                             transaction_types: List[TransactionType],
                                             network_id: NetworkId) -> List[OptimizationRecommendation]:
        """Get personalized gas optimization recommendations
        
        Args:
            user_address: User's wallet address
            transaction_types: Types of transactions user wants to execute
            network_id: Target blockchain network
            
        Returns:
            List of optimization recommendations
        """
        try:
            recommendations = []
            
            # Get current gas prices
            gas_price = await self.get_current_gas_prices(network_id)
            if not gas_price:
                return recommendations
            
            # Timing optimization
            if gas_price.network_congestion > 0.6:
                timing_rec = OptimizationRecommendation(
                    recommendation_type="timing",
                    description="Wait for lower gas prices during off-peak hours",
                    potential_savings_usd=Decimal('10'),
                    potential_savings_percentage=25.0,
                    implementation_difficulty="easy",
                    time_sensitive=False,
                    details={
                        'optimal_times': ['2-6 AM UTC', '10-12 PM UTC'],
                        'current_congestion': gas_price.network_congestion,
                        'estimated_wait': '30-60 minutes'
                    }
                )
                recommendations.append(timing_rec)
            
            # Network optimization
            if network_id == NetworkId.ETHEREUM:
                network_rec = OptimizationRecommendation(
                    recommendation_type="network",
                    description="Consider using Layer 2 networks for significant savings",
                    potential_savings_usd=Decimal('50'),
                    potential_savings_percentage=70.0,
                    implementation_difficulty="medium",
                    time_sensitive=False,
                    details={
                        'recommended_networks': ['Polygon', 'Arbitrum', 'Optimism'],
                        'bridge_time': '10-30 minutes',
                        'bridge_cost': '$5-15'
                    }
                )
                recommendations.append(network_rec)
            
            # Batching optimization
            batchable_types = [TransactionType.CLAIM_REWARDS, TransactionType.GOVERNANCE, TransactionType.STAKE]
            user_batchable = [t for t in transaction_types if t in batchable_types]
            
            if len(user_batchable) > 1:
                batch_rec = OptimizationRecommendation(
                    recommendation_type="batching",
                    description="Batch multiple transactions to save on gas costs",
                    potential_savings_usd=Decimal('20'),
                    potential_savings_percentage=35.0,
                    implementation_difficulty="easy",
                    time_sensitive=False,
                    details={
                        'batchable_transactions': [t.value for t in user_batchable],
                        'estimated_savings_per_tx': '$5-10',
                        'batch_timeout': '30 minutes'
                    }
                )
                recommendations.append(batch_rec)
            
            # Priority optimization
            if any(t in [TransactionType.CLAIM_REWARDS, TransactionType.GOVERNANCE] for t in transaction_types):
                priority_rec = OptimizationRecommendation(
                    recommendation_type="priority",
                    description="Use slow priority for non-urgent transactions",
                    potential_savings_usd=Decimal('8'),
                    potential_savings_percentage=20.0,
                    implementation_difficulty="easy",
                    time_sensitive=False,
                    details={
                        'slow_priority_savings': '20-30%',
                        'additional_wait_time': '10-20 minutes',
                        'suitable_transactions': ['Claim rewards', 'Governance voting']
                    }
                )
                recommendations.append(priority_rec)
            
            # MEV protection optimization
            if any(t in [TransactionType.SWAP, TransactionType.NFT_TRADE] for t in transaction_types):
                mev_rec = OptimizationRecommendation(
                    recommendation_type="mev_protection",
                    description="Use MEV protection to avoid front-running",
                    potential_savings_usd=Decimal('15'),
                    potential_savings_percentage=0.0,  # Not direct savings, but protection
                    implementation_difficulty="medium",
                    time_sensitive=True,
                    details={
                        'protection_methods': ['Flashbots Protect', 'Private mempools'],
                        'additional_cost': '$1-3',
                        'protection_level': 'High'
                    }
                )
                recommendations.append(mev_rec)
            
            # Sort by potential savings
            recommendations.sort(key=lambda r: r.potential_savings_usd, reverse=True)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting optimization recommendations: {e}")
            return []
    
    async def create_transaction_batch(self, 
                                     user_address: str,
                                     transactions: List[Dict[str, Any]],
                                     network_id: NetworkId) -> Optional[BatchTransaction]:
        """Create a batch of transactions for gas optimization
        
        Args:
            user_address: User's wallet address
            transactions: List of transactions to batch
            network_id: Target blockchain network
            
        Returns:
            Batch transaction object
        """
        try:
            if len(transactions) < 2:
                return None
            
            # Calculate individual gas costs
            individual_costs = []
            total_individual_gas = 0
            
            for tx in transactions:
                tx_type = TransactionType(tx.get('type', 'swap'))
                estimate = await self.estimate_transaction_gas(tx_type, network_id)
                if estimate:
                    individual_costs.append(estimate.total_cost_usd)
                    total_individual_gas += estimate.estimated_gas
            
            if not individual_costs:
                return None
            
            # Calculate batch gas cost (typically 10-20% savings)
            batch_gas_savings = 0.15  # 15% average savings
            batch_gas = int(total_individual_gas * (1 - batch_gas_savings))
            
            # Get current gas price
            gas_price = await self.get_current_gas_prices(network_id)
            if not gas_price:
                return None
            
            # Calculate batch cost
            network_config = self.network_settings.get(network_id, {})
            native_token_price = network_config.get('native_token_price_usd', 1)
            gas_cost_native = (Decimal(batch_gas) * gas_price.standard) / Decimal('1e9')
            batch_cost = gas_cost_native * Decimal(str(native_token_price))
            
            # Calculate savings
            individual_total = sum(individual_costs)
            gas_savings = individual_total - batch_cost
            
            # Generate batch ID
            batch_id = self._generate_batch_id(user_address, transactions)
            
            # Create batch transaction
            batch = BatchTransaction(
                batch_id=batch_id,
                transactions=transactions,
                estimated_gas_savings=gas_savings,
                estimated_time_savings=len(transactions) * 2,  # 2 minutes per transaction saved
                total_gas_cost=batch_cost,
                individual_gas_cost=individual_total,
                expires_at=datetime.now() + timedelta(minutes=self.optimization_settings['batch_timeout_minutes'])
            )
            
            # Store batch
            self.batch_transactions[batch_id] = batch
            
            self.logger.info(f"Created transaction batch {batch_id} with {len(transactions)} transactions")
            
            return batch
            
        except Exception as e:
            self.logger.error(f"Error creating transaction batch: {e}")
            return None
    
    def _generate_batch_id(self, user_address: str, transactions: List[Dict[str, Any]]) -> str:
        """Generate unique batch ID"""
        data = f"{user_address}_{len(transactions)}_{datetime.now().timestamp()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def _generate_estimate_id(self, transaction_type: TransactionType, network_id: NetworkId, priority: GasPriority) -> str:
        """Generate unique estimate ID"""
        data = f"{transaction_type.value}_{network_id.value}_{priority.value}_{datetime.now().timestamp()}"
        return hashlib.sha256(data.encode()).hexdigest()[:12]
    
    async def predict_gas_prices(self, network_id: NetworkId, hours_ahead: int = 1) -> Dict[str, Any]:
        """Predict future gas prices
        
        Args:
            network_id: Target blockchain network
            hours_ahead: Hours to predict ahead
            
        Returns:
            Gas price predictions
        """
        try:
            if network_id not in self.price_history or len(self.price_history[network_id]) < 60:
                return {'error': 'Insufficient historical data'}
            
            # Get historical data
            history = list(self.price_history[network_id])
            prices = [float(p['standard_price']) for p in history[-60:]]  # Last hour
            congestion = [p['congestion'] for p in history[-60:]]
            
            # Simple trend analysis
            recent_trend = (prices[-1] - prices[-30]) / prices[-30] if len(prices) >= 30 else 0
            volatility = statistics.stdev(prices[-30:]) if len(prices) >= 30 else 0
            
            # Predict based on trend and patterns
            current_price = prices[-1]
            predicted_price = current_price * (1 + recent_trend * hours_ahead)
            
            # Add confidence based on volatility
            confidence = max(0.3, 1 - (volatility / current_price))
            
            # Generate prediction ranges
            uncertainty = volatility * hours_ahead
            low_estimate = max(0, predicted_price - uncertainty)
            high_estimate = predicted_price + uncertainty
            
            return {
                'network': network_id.name,
                'prediction_time': (datetime.now() + timedelta(hours=hours_ahead)).isoformat(),
                'predicted_price_gwei': round(predicted_price, 2),
                'confidence_score': round(confidence, 2),
                'price_range': {
                    'low': round(low_estimate, 2),
                    'high': round(high_estimate, 2)
                },
                'trend': 'increasing' if recent_trend > 0.05 else 'decreasing' if recent_trend < -0.05 else 'stable',
                'recommendation': self._generate_price_recommendation(predicted_price, current_price, confidence)
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting gas prices: {e}")
            return {'error': str(e)}
    
    def _generate_price_recommendation(self, predicted_price: float, current_price: float, confidence: float) -> str:
        """Generate recommendation based on price prediction"""
        price_change = (predicted_price - current_price) / current_price
        
        if confidence < 0.5:
            return "Low confidence prediction - monitor prices closely"
        elif price_change > 0.1:
            return "Prices expected to rise - consider transacting now"
        elif price_change < -0.1:
            return "Prices expected to fall - consider waiting"
        else:
            return "Prices expected to remain stable"
    
    async def get_gas_analytics(self, network_id: Optional[NetworkId] = None) -> Dict[str, Any]:
        """Get gas usage analytics
        
        Args:
            network_id: Specific network to analyze (optional)
            
        Returns:
            Gas analytics data
        """
        try:
            analytics = {}
            
            networks_to_analyze = [network_id] if network_id else list(self.gas_prices.keys())
            
            for net_id in networks_to_analyze:
                if net_id not in self.price_history:
                    continue
                
                history = list(self.price_history[net_id])
                if len(history) < 10:
                    continue
                
                prices = [float(p['standard_price']) for p in history]
                congestion_levels = [p['congestion'] for p in history]
                
                # Calculate statistics
                current_price = prices[-1]
                avg_price_24h = statistics.mean(prices)
                min_price_24h = min(prices)
                max_price_24h = max(prices)
                price_volatility = statistics.stdev(prices) / avg_price_24h if avg_price_24h > 0 else 0
                
                # Congestion analysis
                avg_congestion = statistics.mean(congestion_levels)
                peak_congestion_times = []
                
                # Find peak congestion periods
                for i, congestion in enumerate(congestion_levels):
                    if congestion > 0.7:
                        timestamp = history[i]['timestamp']
                        peak_congestion_times.append(timestamp.strftime('%H:%M'))
                
                analytics[net_id.name] = {
                    'current_price_gwei': round(current_price, 2),
                    'avg_price_24h_gwei': round(avg_price_24h, 2),
                    'min_price_24h_gwei': round(min_price_24h, 2),
                    'max_price_24h_gwei': round(max_price_24h, 2),
                    'price_volatility': round(price_volatility, 3),
                    'avg_congestion': round(avg_congestion, 2),
                    'peak_congestion_times': peak_congestion_times[-10:],  # Last 10 peak times
                    'price_trend': 'increasing' if current_price > avg_price_24h * 1.05 else 'decreasing' if current_price < avg_price_24h * 0.95 else 'stable',
                    'optimal_transaction_times': self._get_optimal_times(history),
                    'last_updated': datetime.now().isoformat()
                }
            
            # Cross-network comparison
            if len(analytics) > 1:
                analytics['comparison'] = self._generate_network_comparison(analytics)
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting gas analytics: {e}")
            return {}
    
    def _get_optimal_times(self, history: List[Dict[str, Any]]) -> List[str]:
        """Identify optimal transaction times based on historical data"""
        # Group by hour and find average prices
        hourly_prices = defaultdict(list)
        
        for record in history:
            hour = record['timestamp'].hour
            hourly_prices[hour].append(float(record['standard_price']))
        
        # Calculate average price per hour
        hourly_averages = {}
        for hour, prices in hourly_prices.items():
            hourly_averages[hour] = statistics.mean(prices)
        
        # Find hours with lowest average prices
        sorted_hours = sorted(hourly_averages.items(), key=lambda x: x[1])
        optimal_hours = [f"{hour:02d}:00-{hour+1:02d}:00 UTC" for hour, _ in sorted_hours[:3]]
        
        return optimal_hours
    
    def _generate_network_comparison(self, analytics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cross-network comparison"""
        networks = [k for k in analytics.keys() if k != 'comparison']
        
        if len(networks) < 2:
            return {}
        
        # Find cheapest and most expensive networks
        current_prices = {net: analytics[net]['current_price_gwei'] for net in networks}
        cheapest = min(current_prices, key=current_prices.get)
        most_expensive = max(current_prices, key=current_prices.get)
        
        # Calculate savings
        savings_percentage = ((current_prices[most_expensive] - current_prices[cheapest]) / current_prices[most_expensive]) * 100
        
        return {
            'cheapest_network': cheapest,
            'most_expensive_network': most_expensive,
            'potential_savings_percentage': round(savings_percentage, 1),
            'price_difference_gwei': round(current_prices[most_expensive] - current_prices[cheapest], 2),
            'recommendation': f"Consider using {cheapest} for {savings_percentage:.0f}% gas savings"
        }
    
    async def shutdown(self):
        """Gracefully shutdown gas optimizer"""
        self.logger.info("Shutting down gas optimizer...")
        self.gas_prices.clear()
        self.price_history.clear()
        self.gas_estimates.clear()
        self.batch_transactions.clear()
        self.logger.info("Gas optimizer shutdown complete")

# Export main classes
__all__ = [
    'GasOptimizer', 'GasPrice', 'GasEstimate', 'OptimizationRecommendation',
    'BatchTransaction', 'GasPriority', 'TransactionType', 'NetworkId'
]