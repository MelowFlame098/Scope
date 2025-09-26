"""Cross-Chain Bridge Manager

This module provides cross-chain asset transfer capabilities, enabling users to move
assets between different blockchain networks efficiently and securely. It supports
multiple bridge protocols and provides intelligent routing for optimal transfers.

Features:
- Multi-protocol bridge support
- Intelligent route optimization
- Cross-chain transaction monitoring
- Fee optimization
- Security validation
- Bridge aggregation

Supported Bridges:
- Polygon Bridge (Ethereum <-> Polygon)
- Arbitrum Bridge (Ethereum <-> Arbitrum)
- Optimism Bridge (Ethereum <-> Optimism)
- Avalanche Bridge (Ethereum <-> Avalanche)
- BSC Bridge (Ethereum <-> BSC)
- Multichain (formerly AnySwap)
- Hop Protocol
- Synapse Protocol

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
import hashlib
from collections import defaultdict

class BridgeStatus(Enum):
    """Status of bridge transfers"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class BridgeType(Enum):
    """Types of bridge protocols"""
    NATIVE = "native"  # Official chain bridges
    THIRD_PARTY = "third_party"  # Third-party bridges
    LIQUIDITY = "liquidity"  # Liquidity-based bridges
    LOCK_MINT = "lock_mint"  # Lock & mint bridges
    ATOMIC_SWAP = "atomic_swap"  # Atomic swaps

class ChainId(Enum):
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
class BridgeRoute:
    """Represents a bridge route between chains"""
    bridge_name: str
    bridge_type: BridgeType
    source_chain: ChainId
    destination_chain: ChainId
    supported_tokens: List[str]
    fee_percentage: Decimal
    fixed_fee: Decimal
    min_amount: Decimal
    max_amount: Decimal
    estimated_time: int  # Minutes
    security_score: int  # 1-10
    is_active: bool = True
    daily_volume_limit: Optional[Decimal] = None

@dataclass
class BridgeQuote:
    """Quote for a cross-chain transfer"""
    route: BridgeRoute
    input_amount: Decimal
    output_amount: Decimal
    fees: Decimal
    gas_cost: Decimal
    total_cost: Decimal
    estimated_time: int
    price_impact: Decimal
    confidence_score: float
    warnings: List[str] = field(default_factory=list)

@dataclass
class BridgeTransaction:
    """Cross-chain bridge transaction"""
    transaction_id: str
    user_address: str
    route: BridgeRoute
    token_address: str
    amount: Decimal
    source_tx_hash: Optional[str]
    destination_tx_hash: Optional[str]
    status: BridgeStatus
    created_at: datetime
    updated_at: datetime
    estimated_completion: datetime
    fees_paid: Decimal
    gas_used: Optional[int] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class CrossChainBridge:
    """Advanced Cross-Chain Bridge Manager
    
    Provides intelligent cross-chain asset transfer capabilities with
    route optimization, fee minimization, and security validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Cross-Chain Bridge Manager
        
        Args:
            config: Configuration dictionary containing:
                - bridge_configs: Bridge protocol configurations
                - security_settings: Security validation parameters
                - optimization_settings: Route optimization parameters
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.CrossChainBridge")
        
        # Bridge routes and transactions
        self.routes: Dict[str, BridgeRoute] = {}
        self.transactions: Dict[str, BridgeTransaction] = {}
        
        # Cache settings
        self.quote_cache: Dict[str, Tuple[BridgeQuote, datetime]] = {}
        self.cache_ttl = timedelta(minutes=2)  # Short TTL for quotes
        
        # Security settings
        self.security_settings = config.get('security_settings', {
            'min_security_score': 7,
            'max_amount_per_tx': 100000,  # $100k
            'daily_limit_per_user': 500000,  # $500k
            'require_confirmation': True
        })
        
        # Optimization settings
        self.optimization_settings = config.get('optimization_settings', {
            'prioritize_speed': False,
            'prioritize_cost': True,
            'max_routes_to_compare': 5,
            'slippage_tolerance': 0.01  # 1%
        })
        
        # Initialize bridge routes
        self._initialize_bridge_routes()
        
        self.logger.info("Cross-Chain Bridge Manager initialized")
    
    def _initialize_bridge_routes(self):
        """Initialize supported bridge routes"""
        # Define supported bridge routes
        bridge_routes = [
            # Ethereum <-> Polygon
            BridgeRoute(
                bridge_name="polygon_pos",
                bridge_type=BridgeType.NATIVE,
                source_chain=ChainId.ETHEREUM,
                destination_chain=ChainId.POLYGON,
                supported_tokens=["ETH", "USDC", "USDT", "DAI", "WBTC"],
                fee_percentage=Decimal('0.001'),  # 0.1%
                fixed_fee=Decimal('0'),
                min_amount=Decimal('10'),
                max_amount=Decimal('1000000'),
                estimated_time=30,  # 30 minutes
                security_score=9
            ),
            # Ethereum <-> Arbitrum
            BridgeRoute(
                bridge_name="arbitrum_bridge",
                bridge_type=BridgeType.NATIVE,
                source_chain=ChainId.ETHEREUM,
                destination_chain=ChainId.ARBITRUM,
                supported_tokens=["ETH", "USDC", "USDT", "DAI", "WBTC"],
                fee_percentage=Decimal('0.0005'),  # 0.05%
                fixed_fee=Decimal('0'),
                min_amount=Decimal('5'),
                max_amount=Decimal('1000000'),
                estimated_time=15,  # 15 minutes
                security_score=9
            ),
            # Ethereum <-> Optimism
            BridgeRoute(
                bridge_name="optimism_bridge",
                bridge_type=BridgeType.NATIVE,
                source_chain=ChainId.ETHEREUM,
                destination_chain=ChainId.OPTIMISM,
                supported_tokens=["ETH", "USDC", "USDT", "DAI", "WBTC"],
                fee_percentage=Decimal('0.0005'),  # 0.05%
                fixed_fee=Decimal('0'),
                min_amount=Decimal('5'),
                max_amount=Decimal('1000000'),
                estimated_time=20,  # 20 minutes
                security_score=9
            ),
            # Ethereum <-> BSC
            BridgeRoute(
                bridge_name="bsc_bridge",
                bridge_type=BridgeType.THIRD_PARTY,
                source_chain=ChainId.ETHEREUM,
                destination_chain=ChainId.BSC,
                supported_tokens=["ETH", "USDC", "USDT", "DAI", "WBTC"],
                fee_percentage=Decimal('0.002'),  # 0.2%
                fixed_fee=Decimal('5'),
                min_amount=Decimal('20'),
                max_amount=Decimal('500000'),
                estimated_time=10,  # 10 minutes
                security_score=7
            ),
            # Ethereum <-> Avalanche
            BridgeRoute(
                bridge_name="avalanche_bridge",
                bridge_type=BridgeType.NATIVE,
                source_chain=ChainId.ETHEREUM,
                destination_chain=ChainId.AVALANCHE,
                supported_tokens=["ETH", "USDC", "USDT", "DAI", "WBTC"],
                fee_percentage=Decimal('0.001'),  # 0.1%
                fixed_fee=Decimal('0'),
                min_amount=Decimal('10'),
                max_amount=Decimal('1000000'),
                estimated_time=25,  # 25 minutes
                security_score=8
            ),
            # Multichain routes (various pairs)
            BridgeRoute(
                bridge_name="multichain",
                bridge_type=BridgeType.THIRD_PARTY,
                source_chain=ChainId.ETHEREUM,
                destination_chain=ChainId.FANTOM,
                supported_tokens=["USDC", "USDT", "DAI", "WBTC"],
                fee_percentage=Decimal('0.003'),  # 0.3%
                fixed_fee=Decimal('2'),
                min_amount=Decimal('50'),
                max_amount=Decimal('200000'),
                estimated_time=5,  # 5 minutes
                security_score=6
            ),
            # Hop Protocol (L2 <-> L2)
            BridgeRoute(
                bridge_name="hop_protocol",
                bridge_type=BridgeType.LIQUIDITY,
                source_chain=ChainId.POLYGON,
                destination_chain=ChainId.ARBITRUM,
                supported_tokens=["USDC", "USDT", "DAI", "ETH"],
                fee_percentage=Decimal('0.002'),  # 0.2%
                fixed_fee=Decimal('1'),
                min_amount=Decimal('10'),
                max_amount=Decimal('100000'),
                estimated_time=3,  # 3 minutes
                security_score=8
            )
        ]
        
        # Store routes
        for route in bridge_routes:
            route_key = f"{route.bridge_name}_{route.source_chain.value}_{route.destination_chain.value}"
            self.routes[route_key] = route
            
            # Add reverse route if applicable
            if route.bridge_type in [BridgeType.NATIVE, BridgeType.LIQUIDITY]:
                reverse_route = BridgeRoute(
                    bridge_name=route.bridge_name,
                    bridge_type=route.bridge_type,
                    source_chain=route.destination_chain,
                    destination_chain=route.source_chain,
                    supported_tokens=route.supported_tokens,
                    fee_percentage=route.fee_percentage,
                    fixed_fee=route.fixed_fee,
                    min_amount=route.min_amount,
                    max_amount=route.max_amount,
                    estimated_time=route.estimated_time * 2,  # Reverse usually takes longer
                    security_score=route.security_score
                )
                reverse_key = f"{route.bridge_name}_{route.destination_chain.value}_{route.source_chain.value}"
                self.routes[reverse_key] = reverse_route
        
        self.logger.info(f"Initialized {len(self.routes)} bridge routes")
    
    async def get_available_routes(self, 
                                 source_chain: ChainId, 
                                 destination_chain: ChainId, 
                                 token: str) -> List[BridgeRoute]:
        """Get available bridge routes for a token transfer
        
        Args:
            source_chain: Source blockchain
            destination_chain: Destination blockchain
            token: Token symbol to transfer
            
        Returns:
            List of available bridge routes
        """
        try:
            available_routes = []
            
            for route in self.routes.values():
                if (route.source_chain == source_chain and 
                    route.destination_chain == destination_chain and
                    token in route.supported_tokens and
                    route.is_active):
                    available_routes.append(route)
            
            # Sort by security score and estimated time
            available_routes.sort(key=lambda r: (-r.security_score, r.estimated_time))
            
            return available_routes
            
        except Exception as e:
            self.logger.error(f"Error getting available routes: {e}")
            return []
    
    async def get_bridge_quote(self, 
                              source_chain: ChainId,
                              destination_chain: ChainId,
                              token: str,
                              amount: Decimal,
                              user_address: str) -> List[BridgeQuote]:
        """Get bridge quotes for a transfer
        
        Args:
            source_chain: Source blockchain
            destination_chain: Destination blockchain
            token: Token symbol to transfer
            amount: Amount to transfer
            user_address: User's wallet address
            
        Returns:
            List of bridge quotes sorted by optimization preference
        """
        try:
            # Check cache first
            cache_key = f"{source_chain.value}_{destination_chain.value}_{token}_{amount}_{user_address}"
            if self._is_quote_cache_valid(cache_key):
                cached_quote, _ = self.quote_cache[cache_key]
                return [cached_quote]
            
            # Get available routes
            routes = await self.get_available_routes(source_chain, destination_chain, token)
            
            if not routes:
                return []
            
            quotes = []
            
            for route in routes[:self.optimization_settings['max_routes_to_compare']]:
                try:
                    quote = await self._calculate_quote(route, token, amount, user_address)
                    if quote:
                        quotes.append(quote)
                except Exception as e:
                    self.logger.warning(f"Error calculating quote for {route.bridge_name}: {e}")
                    continue
            
            # Sort quotes based on optimization preferences
            quotes = self._sort_quotes(quotes)
            
            # Cache the best quote
            if quotes:
                self.quote_cache[cache_key] = (quotes[0], datetime.now())
            
            return quotes
            
        except Exception as e:
            self.logger.error(f"Error getting bridge quotes: {e}")
            return []
    
    async def _calculate_quote(self, 
                              route: BridgeRoute, 
                              token: str, 
                              amount: Decimal, 
                              user_address: str) -> Optional[BridgeQuote]:
        """Calculate quote for a specific route"""
        try:
            # Validate amount limits
            if amount < route.min_amount or amount > route.max_amount:
                return None
            
            # Calculate fees
            percentage_fee = amount * route.fee_percentage
            total_fees = percentage_fee + route.fixed_fee
            
            # Estimate gas costs (simplified)
            gas_cost = self._estimate_gas_cost(route.source_chain, route.destination_chain)
            
            # Calculate output amount
            output_amount = amount - total_fees
            
            # Calculate total cost
            total_cost = total_fees + gas_cost
            
            # Calculate price impact (simplified)
            price_impact = total_cost / amount if amount > 0 else Decimal('0')
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(route, amount)
            
            # Generate warnings
            warnings = self._generate_quote_warnings(route, amount, total_cost)
            
            return BridgeQuote(
                route=route,
                input_amount=amount,
                output_amount=output_amount,
                fees=total_fees,
                gas_cost=gas_cost,
                total_cost=total_cost,
                estimated_time=route.estimated_time,
                price_impact=price_impact,
                confidence_score=confidence_score,
                warnings=warnings
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating quote: {e}")
            return None
    
    def _estimate_gas_cost(self, source_chain: ChainId, destination_chain: ChainId) -> Decimal:
        """Estimate gas costs for bridge transaction"""
        # Simplified gas cost estimation
        base_costs = {
            ChainId.ETHEREUM: Decimal('50'),  # $50 base cost
            ChainId.POLYGON: Decimal('1'),    # $1 base cost
            ChainId.ARBITRUM: Decimal('5'),   # $5 base cost
            ChainId.OPTIMISM: Decimal('5'),   # $5 base cost
            ChainId.BSC: Decimal('2'),        # $2 base cost
            ChainId.AVALANCHE: Decimal('3'),  # $3 base cost
            ChainId.FANTOM: Decimal('1'),     # $1 base cost
        }
        
        source_cost = base_costs.get(source_chain, Decimal('10'))
        dest_cost = base_costs.get(destination_chain, Decimal('10'))
        
        return source_cost + dest_cost
    
    def _calculate_confidence_score(self, route: BridgeRoute, amount: Decimal) -> float:
        """Calculate confidence score for a quote"""
        score = 0.0
        
        # Security score factor (40%)
        score += (route.security_score / 10) * 0.4
        
        # Bridge type factor (20%)
        type_scores = {
            BridgeType.NATIVE: 1.0,
            BridgeType.LIQUIDITY: 0.9,
            BridgeType.THIRD_PARTY: 0.7,
            BridgeType.LOCK_MINT: 0.8,
            BridgeType.ATOMIC_SWAP: 0.9
        }
        score += type_scores.get(route.bridge_type, 0.5) * 0.2
        
        # Amount factor (20%) - higher confidence for amounts within optimal range
        optimal_range = route.max_amount * Decimal('0.1')  # 10% of max
        if amount <= optimal_range:
            score += 0.2
        else:
            score += 0.2 * (1 - min(float(amount / route.max_amount), 1.0))
        
        # Time factor (20%) - faster is better
        max_time = 120  # 2 hours
        time_score = max(0, (max_time - route.estimated_time) / max_time)
        score += time_score * 0.2
        
        return round(score, 2)
    
    def _generate_quote_warnings(self, route: BridgeRoute, amount: Decimal, total_cost: Decimal) -> List[str]:
        """Generate warnings for a quote"""
        warnings = []
        
        # High fee warning
        fee_percentage = (total_cost / amount) * 100 if amount > 0 else 0
        if fee_percentage > 5:
            warnings.append(f"High fees: {fee_percentage:.2f}% of transfer amount")
        
        # Security warning
        if route.security_score < self.security_settings['min_security_score']:
            warnings.append(f"Lower security score: {route.security_score}/10")
        
        # Large amount warning
        if amount > self.security_settings['max_amount_per_tx']:
            warnings.append("Large transfer amount - consider splitting into smaller transactions")
        
        # Third-party bridge warning
        if route.bridge_type == BridgeType.THIRD_PARTY:
            warnings.append("Third-party bridge - additional smart contract risk")
        
        # Long transfer time warning
        if route.estimated_time > 60:
            warnings.append(f"Long transfer time: ~{route.estimated_time} minutes")
        
        return warnings
    
    def _sort_quotes(self, quotes: List[BridgeQuote]) -> List[BridgeQuote]:
        """Sort quotes based on optimization preferences"""
        if self.optimization_settings['prioritize_speed']:
            # Sort by time, then cost
            quotes.sort(key=lambda q: (q.estimated_time, q.total_cost))
        elif self.optimization_settings['prioritize_cost']:
            # Sort by cost, then time
            quotes.sort(key=lambda q: (q.total_cost, q.estimated_time))
        else:
            # Sort by confidence score
            quotes.sort(key=lambda q: q.confidence_score, reverse=True)
        
        return quotes
    
    def _is_quote_cache_valid(self, cache_key: str) -> bool:
        """Check if cached quote is still valid"""
        if cache_key not in self.quote_cache:
            return False
        
        _, timestamp = self.quote_cache[cache_key]
        return datetime.now() - timestamp < self.cache_ttl
    
    async def initiate_bridge_transfer(self, 
                                      quote: BridgeQuote, 
                                      user_address: str,
                                      destination_address: Optional[str] = None) -> Dict[str, Any]:
        """Initiate a cross-chain bridge transfer
        
        Args:
            quote: Selected bridge quote
            user_address: User's source wallet address
            destination_address: Destination wallet address (optional)
            
        Returns:
            Transaction initiation result
        """
        try:
            # Validate security requirements
            security_check = self._validate_security(quote, user_address)
            if not security_check['valid']:
                return {
                    'success': False,
                    'error': security_check['reason'],
                    'transaction_id': None
                }
            
            # Generate transaction ID
            transaction_id = self._generate_transaction_id(quote, user_address)
            
            # Create transaction record
            transaction = BridgeTransaction(
                transaction_id=transaction_id,
                user_address=user_address,
                route=quote.route,
                token_address="0x" + "0" * 40,  # Mock token address
                amount=quote.input_amount,
                source_tx_hash=None,
                destination_tx_hash=None,
                status=BridgeStatus.PENDING,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                estimated_completion=datetime.now() + timedelta(minutes=quote.estimated_time),
                fees_paid=quote.fees,
                metadata={
                    'destination_address': destination_address or user_address,
                    'quote_confidence': quote.confidence_score,
                    'warnings': quote.warnings
                }
            )
            
            # Store transaction
            self.transactions[transaction_id] = transaction
            
            # In a real implementation, this would:
            # 1. Build the bridge transaction
            # 2. Submit to the blockchain
            # 3. Monitor for confirmation
            
            # Mock successful initiation
            source_tx_hash = f"0x{'1' * 64}"
            transaction.source_tx_hash = source_tx_hash
            transaction.status = BridgeStatus.CONFIRMED
            transaction.updated_at = datetime.now()
            
            self.logger.info(f"Initiated bridge transfer {transaction_id}")
            
            return {
                'success': True,
                'transaction_id': transaction_id,
                'source_tx_hash': source_tx_hash,
                'estimated_completion': transaction.estimated_completion.isoformat(),
                'bridge_info': {
                    'bridge_name': quote.route.bridge_name,
                    'source_chain': quote.route.source_chain.name,
                    'destination_chain': quote.route.destination_chain.name,
                    'estimated_time_minutes': quote.estimated_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error initiating bridge transfer: {e}")
            return {
                'success': False,
                'error': str(e),
                'transaction_id': None
            }
    
    def _validate_security(self, quote: BridgeQuote, user_address: str) -> Dict[str, Any]:
        """Validate security requirements for a transfer"""
        # Check security score
        if quote.route.security_score < self.security_settings['min_security_score']:
            return {
                'valid': False,
                'reason': f"Bridge security score too low: {quote.route.security_score}/10"
            }
        
        # Check amount limits
        if quote.input_amount > self.security_settings['max_amount_per_tx']:
            return {
                'valid': False,
                'reason': f"Amount exceeds maximum per transaction: ${self.security_settings['max_amount_per_tx']}"
            }
        
        # Check daily limits (simplified)
        daily_total = self._get_user_daily_volume(user_address)
        if daily_total + quote.input_amount > self.security_settings['daily_limit_per_user']:
            return {
                'valid': False,
                'reason': f"Amount exceeds daily limit: ${self.security_settings['daily_limit_per_user']}"
            }
        
        return {'valid': True, 'reason': None}
    
    def _get_user_daily_volume(self, user_address: str) -> Decimal:
        """Get user's daily transfer volume"""
        today = datetime.now().date()
        daily_volume = Decimal('0')
        
        for tx in self.transactions.values():
            if (tx.user_address.lower() == user_address.lower() and
                tx.created_at.date() == today and
                tx.status in [BridgeStatus.CONFIRMED, BridgeStatus.PROCESSING, BridgeStatus.COMPLETED]):
                daily_volume += tx.amount
        
        return daily_volume
    
    def _generate_transaction_id(self, quote: BridgeQuote, user_address: str) -> str:
        """Generate unique transaction ID"""
        data = f"{user_address}_{quote.route.bridge_name}_{datetime.now().timestamp()}_{quote.input_amount}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    async def get_transaction_status(self, transaction_id: str) -> Dict[str, Any]:
        """Get status of a bridge transaction
        
        Args:
            transaction_id: Transaction identifier
            
        Returns:
            Transaction status information
        """
        try:
            if transaction_id not in self.transactions:
                return {'error': 'Transaction not found'}
            
            transaction = self.transactions[transaction_id]
            
            # Update transaction status (mock implementation)
            await self._update_transaction_status(transaction)
            
            return {
                'transaction_id': transaction_id,
                'status': transaction.status.value,
                'source_chain': transaction.route.source_chain.name,
                'destination_chain': transaction.route.destination_chain.name,
                'bridge_name': transaction.route.bridge_name,
                'amount': float(transaction.amount),
                'fees_paid': float(transaction.fees_paid),
                'source_tx_hash': transaction.source_tx_hash,
                'destination_tx_hash': transaction.destination_tx_hash,
                'created_at': transaction.created_at.isoformat(),
                'updated_at': transaction.updated_at.isoformat(),
                'estimated_completion': transaction.estimated_completion.isoformat(),
                'progress_percentage': self._calculate_progress_percentage(transaction),
                'error_message': transaction.error_message
            }
            
        except Exception as e:
            self.logger.error(f"Error getting transaction status: {e}")
            return {'error': str(e)}
    
    async def _update_transaction_status(self, transaction: BridgeTransaction):
        """Update transaction status based on blockchain state"""
        try:
            # Mock status progression
            now = datetime.now()
            elapsed = (now - transaction.created_at).total_seconds() / 60  # Minutes
            
            if transaction.status == BridgeStatus.CONFIRMED and elapsed > 2:
                transaction.status = BridgeStatus.PROCESSING
                transaction.updated_at = now
            elif transaction.status == BridgeStatus.PROCESSING and elapsed > transaction.route.estimated_time:
                transaction.status = BridgeStatus.COMPLETED
                transaction.destination_tx_hash = f"0x{'2' * 64}"
                transaction.updated_at = now
            
        except Exception as e:
            self.logger.error(f"Error updating transaction status: {e}")
    
    def _calculate_progress_percentage(self, transaction: BridgeTransaction) -> int:
        """Calculate transaction progress percentage"""
        status_progress = {
            BridgeStatus.PENDING: 10,
            BridgeStatus.CONFIRMED: 30,
            BridgeStatus.PROCESSING: 70,
            BridgeStatus.COMPLETED: 100,
            BridgeStatus.FAILED: 0,
            BridgeStatus.CANCELLED: 0
        }
        
        base_progress = status_progress.get(transaction.status, 0)
        
        # Add time-based progress for processing status
        if transaction.status == BridgeStatus.PROCESSING:
            elapsed = (datetime.now() - transaction.created_at).total_seconds() / 60
            time_progress = min(elapsed / transaction.route.estimated_time, 1.0) * 40  # 40% for time
            return min(int(base_progress + time_progress), 100)
        
        return base_progress
    
    async def get_user_transactions(self, user_address: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get user's bridge transaction history
        
        Args:
            user_address: User's wallet address
            limit: Maximum number of transactions to return
            
        Returns:
            List of user's bridge transactions
        """
        try:
            user_transactions = []
            
            for transaction in self.transactions.values():
                if transaction.user_address.lower() == user_address.lower():
                    # Update status
                    await self._update_transaction_status(transaction)
                    
                    user_transactions.append({
                        'transaction_id': transaction.transaction_id,
                        'status': transaction.status.value,
                        'bridge_name': transaction.route.bridge_name,
                        'source_chain': transaction.route.source_chain.name,
                        'destination_chain': transaction.route.destination_chain.name,
                        'amount': float(transaction.amount),
                        'fees_paid': float(transaction.fees_paid),
                        'source_tx_hash': transaction.source_tx_hash,
                        'destination_tx_hash': transaction.destination_tx_hash,
                        'created_at': transaction.created_at.isoformat(),
                        'estimated_completion': transaction.estimated_completion.isoformat(),
                        'progress_percentage': self._calculate_progress_percentage(transaction)
                    })
            
            # Sort by creation time (newest first)
            user_transactions.sort(key=lambda x: x['created_at'], reverse=True)
            
            return user_transactions[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting user transactions: {e}")
            return []
    
    async def get_bridge_analytics(self) -> Dict[str, Any]:
        """Get bridge usage analytics
        
        Returns:
            Bridge analytics data
        """
        try:
            total_transactions = len(self.transactions)
            total_volume = sum(tx.amount for tx in self.transactions.values())
            
            # Status distribution
            status_counts = defaultdict(int)
            for tx in self.transactions.values():
                status_counts[tx.status.value] += 1
            
            # Bridge usage
            bridge_usage = defaultdict(int)
            bridge_volume = defaultdict(Decimal)
            for tx in self.transactions.values():
                bridge_usage[tx.route.bridge_name] += 1
                bridge_volume[tx.route.bridge_name] += tx.amount
            
            # Chain pair popularity
            chain_pairs = defaultdict(int)
            for tx in self.transactions.values():
                pair = f"{tx.route.source_chain.name}->{tx.route.destination_chain.name}"
                chain_pairs[pair] += 1
            
            return {
                'total_transactions': total_transactions,
                'total_volume_usd': float(total_volume),
                'status_distribution': dict(status_counts),
                'bridge_usage': {
                    'by_count': dict(bridge_usage),
                    'by_volume': {k: float(v) for k, v in bridge_volume.items()}
                },
                'popular_chain_pairs': dict(sorted(chain_pairs.items(), key=lambda x: x[1], reverse=True)[:10]),
                'average_transaction_size': float(total_volume / total_transactions) if total_transactions > 0 else 0,
                'supported_routes': len(self.routes),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting bridge analytics: {e}")
            return {}
    
    async def shutdown(self):
        """Gracefully shutdown cross-chain bridge manager"""
        self.logger.info("Shutting down cross-chain bridge manager...")
        self.routes.clear()
        self.transactions.clear()
        self.quote_cache.clear()
        self.logger.info("Cross-chain bridge manager shutdown complete")

# Export main classes
__all__ = [
    'CrossChainBridge', 'BridgeRoute', 'BridgeQuote', 'BridgeTransaction',
    'BridgeStatus', 'BridgeType', 'ChainId'
]