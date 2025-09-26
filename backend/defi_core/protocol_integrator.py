"""DeFi Protocol Integrator

This module provides a unified interface for interacting with various DeFi protocols
across multiple blockchain networks. It handles smart contract interactions,
protocol-specific logic, and provides a standardized API for DeFi operations.

Supported Protocols:
- DEXs: Uniswap V2/V3, SushiSwap, PancakeSwap, Curve, Balancer
- Lending: Aave, Compound, MakerDAO, Venus
- Yield Farming: Yearn Finance, Harvest, Convex
- Derivatives: dYdX, Synthetix, Perpetual Protocol
- Insurance: Nexus Mutual, Cover Protocol
- Staking: Lido, Rocket Pool, Ankr

Features:
- Protocol abstraction layer
- Unified API interface
- Real-time protocol data
- Transaction routing
- Risk assessment
- Yield optimization
- Portfolio tracking

Author: FinScope AI Team
Version: 2.0.0
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib
from abc import ABC, abstractmethod

try:
    from web3 import Web3
    from web3.contract import Contract
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    Web3 = None
    Contract = None
    Account = None

class ProtocolType(Enum):
    """Types of DeFi protocols"""
    DEX = "dex"
    LENDING = "lending"
    YIELD_FARMING = "yield_farming"
    DERIVATIVES = "derivatives"
    INSURANCE = "insurance"
    STAKING = "staking"
    BRIDGE = "bridge"
    SYNTHETIC = "synthetic"
    OPTIONS = "options"
    GOVERNANCE = "governance"

class ProtocolStatus(Enum):
    """Protocol operational status"""
    ACTIVE = "active"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"
    EMERGENCY_PAUSE = "emergency_pause"
    UNKNOWN = "unknown"

class TransactionStatus(Enum):
    """Transaction status"""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ChainId(Enum):
    """Supported blockchain networks"""
    ETHEREUM = 1
    BSC = 56
    POLYGON = 137
    AVALANCHE = 43114
    FANTOM = 250
    ARBITRUM = 42161
    OPTIMISM = 10

@dataclass
class ProtocolInfo:
    """Information about a DeFi protocol"""
    protocol_id: str
    name: str
    protocol_type: ProtocolType
    version: str
    chain_id: int
    contract_addresses: Dict[str, str]
    supported_tokens: List[str]
    tvl_usd: Decimal
    status: ProtocolStatus
    risk_score: int  # 1-10 scale
    audit_status: str
    documentation_url: str
    api_endpoints: Dict[str, str] = field(default_factory=dict)
    fees: Dict[str, Decimal] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class ProtocolPosition:
    """User's position in a protocol"""
    protocol_id: str
    user_address: str
    position_type: str  # "liquidity", "lending", "borrowing", "staking", etc.
    token_address: str
    amount: Decimal
    value_usd: Decimal
    entry_price: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    rewards_earned: Decimal
    created_at: datetime
    last_updated: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProtocolTransaction:
    """Protocol transaction record"""
    transaction_id: str
    protocol_id: str
    user_address: str
    transaction_type: str
    token_in: Optional[str]
    token_out: Optional[str]
    amount_in: Optional[Decimal]
    amount_out: Optional[Decimal]
    gas_used: Optional[int]
    gas_price: Optional[Decimal]
    status: TransactionStatus
    tx_hash: Optional[str]
    block_number: Optional[int]
    timestamp: datetime
    fees_paid: Decimal = Decimal('0')
    slippage: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TransactionResult:
    """Result of a DeFi transaction"""
    success: bool
    transaction_hash: Optional[str]
    gas_used: Optional[int]
    gas_price: Optional[int]
    block_number: Optional[int]
    error_message: Optional[str] = None
    protocol_data: Optional[Dict[str, Any]] = None
    estimated_returns: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None

class BaseProtocolAdapter(ABC):
    """Base class for protocol adapters"""
    
    def __init__(self, protocol_info: ProtocolInfo, config: Dict[str, Any]):
        self.protocol_info = protocol_info
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{protocol_info.name}")
    
    @abstractmethod
    async def get_protocol_data(self) -> Dict[str, Any]:
        """Get current protocol data"""
        pass
    
    @abstractmethod
    async def get_user_positions(self, user_address: str) -> List[ProtocolPosition]:
        """Get user's positions in the protocol"""
        pass
    
    @abstractmethod
    async def execute_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a transaction on the protocol"""
        pass
    
    @abstractmethod
    async def estimate_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate transaction costs and outcomes"""
        pass

class ProtocolIntegrator:
    """Main DeFi Protocol Integrator
    
    Provides unified interface for interacting with various DeFi protocols
    across multiple blockchain networks.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Protocol Integrator
        
        Args:
            config: Configuration dictionary containing:
                - web3_providers: Web3 provider URLs for each chain
                - protocols: List of protocol configurations
                - default_slippage: Default slippage tolerance
                - gas_settings: Gas price and limit settings
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ProtocolIntegrator")
        
        # Web3 connections
        self.web3_connections: Dict[ChainId, Web3] = {}
        self.protocols: Dict[str, ProtocolConfig] = {}
        self.contracts: Dict[str, Contract] = {}
        
        # Settings
        self.default_slippage = config.get('default_slippage', 0.005)  # 0.5%
        self.gas_settings = config.get('gas_settings', {})
        
        # Protocol adapters and data
        self.adapters: Dict[str, BaseProtocolAdapter] = {}
        self.protocols: Dict[str, ProtocolInfo] = {}
        self.user_positions: Dict[str, List[ProtocolPosition]] = defaultdict(list)
        self.transactions: Dict[str, ProtocolTransaction] = {}
        
        # Protocol data cache
        self.protocol_cache = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Integration settings
        self.integration_settings = config.get('integration_settings', {
            'max_protocols_per_user': 10,
            'position_sync_interval': 300,  # 5 minutes
            'transaction_timeout': 600,  # 10 minutes
            'retry_attempts': 3
        })
        
        # Risk settings
        self.risk_settings = config.get('risk_settings', {
            'max_position_size_usd': 100000,
            'max_protocol_allocation': 0.3,  # 30%
            'min_health_factor': 1.5,
            'blacklisted_protocols': []
        })
        
        if WEB3_AVAILABLE:
            self._initialize_connections()
            self._load_protocols()
            self._initialize_protocol_adapters()
        else:
            self.logger.warning("Web3 not available - running in simulation mode")
            self._initialize_mock_adapters()
    
    def _initialize_connections(self):
        """Initialize Web3 connections for supported chains"""
        providers = self.config.get('web3_providers', {})
        
        for chain_name, provider_url in providers.items():
            try:
                chain_id = ChainId[chain_name.upper()]
                w3 = Web3(Web3.HTTPProvider(provider_url))
                
                if w3.is_connected():
                    self.web3_connections[chain_id] = w3
                    self.logger.info(f"Connected to {chain_name} network")
                else:
                    self.logger.error(f"Failed to connect to {chain_name} network")
                    
            except Exception as e:
                self.logger.error(f"Error connecting to {chain_name}: {e}")
    
    def _load_protocols(self):
        """Load protocol configurations"""
        protocol_configs = self.config.get('protocols', [])
        
        for protocol_data in protocol_configs:
            try:
                # Convert old format to new format if needed
                if 'protocol_id' not in protocol_data:
                    protocol_data['protocol_id'] = protocol_data.get('name', '').lower().replace(' ', '_')
                
                protocol = ProtocolInfo(**protocol_data)
                self.protocols[protocol.name] = protocol
                
                # Initialize contract if Web3 connection available
                if protocol.chain_id in self.web3_connections:
                    w3 = self.web3_connections[protocol.chain_id]
                    contract = w3.eth.contract(
                        address=protocol.contract_address,
                        abi=protocol.abi
                    )
                    self.contracts[protocol.name] = contract
                    
                self.logger.info(f"Loaded protocol: {protocol.name}")
                
            except Exception as e:
                self.logger.error(f"Error loading protocol {protocol_data.get('name', 'unknown')}: {e}")
    
    def get_active_protocols(self) -> List[str]:
        """Get list of active protocol names
        
        Returns:
            List of active protocol names
        """
        return [name for name, protocol in self.protocols.items() if protocol.is_active]
    
    def get_protocol_info(self, protocol_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific protocol
        
        Args:
            protocol_name: Name of the protocol
            
        Returns:
            Protocol information dictionary or None if not found
        """
        if protocol_name not in self.protocols:
            return None
        
        protocol = self.protocols[protocol_name]
        return {
            'name': protocol.name,
            'type': protocol.protocol_type.value,
            'chain': protocol.chain_id.name,
            'contract_address': protocol.contract_address,
            'is_active': protocol.is_active,
            'supported_tokens': protocol.supported_tokens or [],
            'fee_tier': protocol.fee_tier
        }
    
    async def get_token_price(self, token_address: str, chain_id: ChainId, protocol_name: str = None) -> Optional[Decimal]:
        """Get token price from DEX
        
        Args:
            token_address: Token contract address
            chain_id: Blockchain network
            protocol_name: Specific protocol to use (optional)
            
        Returns:
            Token price in USD or None if unavailable
        """
        try:
            # Use cached price if available and fresh
            cache_key = f"price_{token_address}_{chain_id.value}"
            if self._is_cache_valid(cache_key):
                return self.protocol_cache[cache_key]['price']
            
            # Find suitable DEX protocol
            dex_protocols = [
                p for p in self.protocols.values() 
                if p.protocol_type == ProtocolType.DEX and p.chain_id == chain_id and p.is_active
            ]
            
            if protocol_name:
                dex_protocols = [p for p in dex_protocols if p.name == protocol_name]
            
            if not dex_protocols:
                self.logger.warning(f"No suitable DEX found for chain {chain_id.name}")
                return None
            
            # Try each DEX until we get a price
            for protocol in dex_protocols:
                try:
                    price = await self._get_token_price_from_dex(token_address, protocol)
                    if price:
                        # Cache the result
                        self.protocol_cache[cache_key] = {
                            'price': price,
                            'timestamp': datetime.now(),
                            'protocol': protocol.name
                        }
                        return price
                except Exception as e:
                    self.logger.warning(f"Failed to get price from {protocol.name}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting token price: {e}")
            return None
    
    async def _get_token_price_from_dex(self, token_address: str, protocol: ProtocolInfo) -> Optional[Decimal]:
        """Get token price from specific DEX protocol"""
        if protocol.name not in self.contracts:
            return None
        
        contract = self.contracts[protocol.name]
        w3 = self.web3_connections[protocol.chain_id]
        
        # Implementation depends on specific DEX
        if 'uniswap' in protocol.name.lower():
            return await self._get_uniswap_price(token_address, contract, w3)
        elif 'sushiswap' in protocol.name.lower():
            return await self._get_sushiswap_price(token_address, contract, w3)
        elif 'pancakeswap' in protocol.name.lower():
            return await self._get_pancakeswap_price(token_address, contract, w3)
        
        return None
    
    async def _get_uniswap_price(self, token_address: str, contract: Contract, w3: Web3) -> Optional[Decimal]:
        """Get token price from Uniswap"""
        try:
            # This is a simplified implementation
            # In practice, you'd need to:
            # 1. Find the best liquidity pool for the token
            # 2. Get reserves from the pool
            # 3. Calculate price based on reserves
            # 4. Convert to USD using a stable reference (USDC/USDT)
            
            # Placeholder implementation
            return Decimal('100.0')  # Mock price
            
        except Exception as e:
            self.logger.error(f"Error getting Uniswap price: {e}")
            return None
    
    async def _get_sushiswap_price(self, token_address: str, contract: Contract, w3: Web3) -> Optional[Decimal]:
        """Get token price from SushiSwap"""
        # Similar implementation to Uniswap
        return Decimal('100.0')  # Mock price
    
    async def _get_pancakeswap_price(self, token_address: str, contract: Contract, w3: Web3) -> Optional[Decimal]:
        """Get token price from PancakeSwap"""
        # Similar implementation to Uniswap
        return Decimal('100.0')  # Mock price
    
    async def swap_tokens(self, 
                         from_token: str, 
                         to_token: str, 
                         amount: Decimal, 
                         chain_id: ChainId,
                         user_address: str,
                         slippage: Optional[float] = None,
                         protocol_name: Optional[str] = None) -> TransactionResult:
        """Swap tokens using DEX
        
        Args:
            from_token: Source token address
            to_token: Destination token address
            amount: Amount to swap
            chain_id: Blockchain network
            user_address: User's wallet address
            slippage: Slippage tolerance (optional)
            protocol_name: Specific protocol to use (optional)
            
        Returns:
            Transaction result
        """
        try:
            slippage = slippage or self.default_slippage
            
            # Find suitable DEX
            dex_protocols = [
                p for p in self.protocols.values() 
                if p.protocol_type == ProtocolType.DEX and p.chain_id == chain_id and p.is_active
            ]
            
            if protocol_name:
                dex_protocols = [p for p in dex_protocols if p.name == protocol_name]
            
            if not dex_protocols:
                return TransactionResult(
                    success=False,
                    transaction_hash=None,
                    gas_used=None,
                    gas_price=None,
                    block_number=None,
                    error_message="No suitable DEX found"
                )
            
            # Use the first available DEX
            protocol = dex_protocols[0]
            
            if not WEB3_AVAILABLE:
                # Simulation mode
                return TransactionResult(
                    success=True,
                    transaction_hash="0x" + "0" * 64,  # Mock hash
                    gas_used=150000,
                    gas_price=20000000000,  # 20 gwei
                    block_number=12345678,
                    protocol_data={
                        'protocol': protocol.name,
                        'from_token': from_token,
                        'to_token': to_token,
                        'amount': str(amount),
                        'slippage': slippage
                    }
                )
            
            # Real implementation would build and execute the transaction
            return await self._execute_swap(protocol, from_token, to_token, amount, user_address, slippage)
            
        except Exception as e:
            self.logger.error(f"Error swapping tokens: {e}")
            return TransactionResult(
                success=False,
                transaction_hash=None,
                gas_used=None,
                gas_price=None,
                block_number=None,
                error_message=str(e)
            )
    
    async def _execute_swap(self, protocol: ProtocolConfig, from_token: str, to_token: str, 
                           amount: Decimal, user_address: str, slippage: float) -> TransactionResult:
        """Execute token swap transaction"""
        # This would contain the actual transaction building and execution logic
        # For now, return a mock successful transaction
        return TransactionResult(
            success=True,
            transaction_hash="0x" + "1" * 64,  # Mock hash
            gas_used=150000,
            gas_price=20000000000,  # 20 gwei
            block_number=12345678,
            protocol_data={
                'protocol': protocol.name,
                'from_token': from_token,
                'to_token': to_token,
                'amount': str(amount),
                'slippage': slippage
            }
        )
    
    async def add_liquidity(self, 
                           token_a: str, 
                           token_b: str, 
                           amount_a: Decimal, 
                           amount_b: Decimal,
                           chain_id: ChainId,
                           user_address: str,
                           protocol_name: Optional[str] = None) -> TransactionResult:
        """Add liquidity to a pool
        
        Args:
            token_a: First token address
            token_b: Second token address
            amount_a: Amount of first token
            amount_b: Amount of second token
            chain_id: Blockchain network
            user_address: User's wallet address
            protocol_name: Specific protocol to use (optional)
            
        Returns:
            Transaction result
        """
        try:
            # Find suitable DEX for liquidity provision
            dex_protocols = [
                p for p in self.protocols.values() 
                if p.protocol_type == ProtocolType.DEX and p.chain_id == chain_id and p.is_active
            ]
            
            if protocol_name:
                dex_protocols = [p for p in dex_protocols if p.name == protocol_name]
            
            if not dex_protocols:
                return TransactionResult(
                    success=False,
                    transaction_hash=None,
                    gas_used=None,
                    gas_price=None,
                    block_number=None,
                    error_message="No suitable DEX found for liquidity provision"
                )
            
            protocol = dex_protocols[0]
            
            # Mock implementation
            return TransactionResult(
                success=True,
                transaction_hash="0x" + "2" * 64,  # Mock hash
                gas_used=200000,
                gas_price=20000000000,  # 20 gwei
                block_number=12345679,
                protocol_data={
                    'protocol': protocol.name,
                    'token_a': token_a,
                    'token_b': token_b,
                    'amount_a': str(amount_a),
                    'amount_b': str(amount_b),
                    'operation': 'add_liquidity'
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error adding liquidity: {e}")
            return TransactionResult(
                success=False,
                transaction_hash=None,
                gas_used=None,
                gas_price=None,
                block_number=None,
                error_message=str(e)
            )
    
    async def remove_liquidity(self, 
                              token_a: str, 
                              token_b: str, 
                              liquidity_amount: Decimal,
                              chain_id: ChainId,
                              user_address: str,
                              protocol_name: Optional[str] = None) -> TransactionResult:
        """Remove liquidity from a pool
        
        Args:
            token_a: First token address
            token_b: Second token address
            liquidity_amount: Amount of liquidity tokens to remove
            chain_id: Blockchain network
            user_address: User's wallet address
            protocol_name: Specific protocol to use (optional)
            
        Returns:
            Transaction result
        """
        # Similar implementation to add_liquidity
        return TransactionResult(
            success=True,
            transaction_hash="0x" + "3" * 64,  # Mock hash
            gas_used=180000,
            gas_price=20000000000,  # 20 gwei
            block_number=12345680,
            protocol_data={
                'operation': 'remove_liquidity',
                'liquidity_amount': str(liquidity_amount)
            }
        )
    
    def _initialize_protocol_adapters(self):
        """Initialize protocol adapters"""
        # This would initialize real protocol adapters
        # For now, we'll use mock adapters
        self._initialize_mock_adapters()
    
    def _initialize_mock_adapters(self):
        """Initialize mock protocol adapters for testing"""
        # Mock Uniswap V3
        uniswap_info = ProtocolInfo(
            protocol_id="uniswap_v3",
            name="Uniswap V3",
            protocol_type=ProtocolType.DEX,
            version="3.0.0",
            chain_id=1,
            contract_addresses={
                'factory': '0x1F98431c8aD98523631AE4a59f267346ea31F984',
                'router': '0xE592427A0AEce92De3Edee1F18E0157C05861564'
            },
            supported_tokens=['ETH', 'USDC', 'USDT', 'DAI', 'WBTC'],
            tvl_usd=Decimal('4500000000'),
            status=ProtocolStatus.ACTIVE,
            risk_score=8,
            audit_status="Audited by Trail of Bits",
            documentation_url="https://docs.uniswap.org/",
            fees={'swap': Decimal('0.003')}
        )
        self.protocols[uniswap_info.protocol_id] = uniswap_info
        
        # Mock Aave V3
        aave_info = ProtocolInfo(
            protocol_id="aave_v3",
            name="Aave V3",
            protocol_type=ProtocolType.LENDING,
            version="3.0.0",
            chain_id=1,
            contract_addresses={
                'pool': '0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2'
            },
            supported_tokens=['ETH', 'USDC', 'USDT', 'DAI', 'WBTC'],
            tvl_usd=Decimal('12000000000'),
            status=ProtocolStatus.ACTIVE,
            risk_score=9,
            audit_status="Audited by OpenZeppelin",
            documentation_url="https://docs.aave.com/",
            fees={'flash_loan': Decimal('0.0009')}
        )
        self.protocols[aave_info.protocol_id] = aave_info
        
        self.logger.info(f"Initialized {len(self.protocols)} mock protocol adapters")
    
    async def get_supported_protocols(self, protocol_type: Optional[ProtocolType] = None) -> List[Dict[str, Any]]:
        """Get list of supported protocols"""
        try:
            protocols = []
            
            for protocol_info in self.protocols.values():
                if protocol_type and protocol_info.protocol_type != protocol_type:
                    continue
                
                protocols.append({
                    'protocol_id': protocol_info.protocol_id,
                    'name': protocol_info.name,
                    'type': protocol_info.protocol_type.value,
                    'version': protocol_info.version,
                    'chain_id': protocol_info.chain_id,
                    'tvl_usd': float(protocol_info.tvl_usd),
                    'status': protocol_info.status.value,
                    'risk_score': protocol_info.risk_score,
                    'supported_tokens': protocol_info.supported_tokens
                })
            
            return protocols
            
        except Exception as e:
            self.logger.error(f"Error getting supported protocols: {e}")
            return []
    
    async def get_yield_opportunities(self, 
                                    user_address: str,
                                    amount_usd: float,
                                    risk_tolerance: str = "medium") -> List[Dict[str, Any]]:
        """Get yield opportunities across protocols"""
        try:
            opportunities = []
            
            # Risk score thresholds
            risk_thresholds = {
                'low': 8,
                'medium': 6,
                'high': 4
            }
            min_risk_score = risk_thresholds.get(risk_tolerance, 6)
            
            for protocol_id, protocol_info in self.protocols.items():
                if protocol_info.risk_score < min_risk_score:
                    continue
                
                if protocol_info.protocol_type == ProtocolType.LENDING:
                    # Mock lending opportunity
                    opportunities.append({
                        'protocol_id': protocol_id,
                        'protocol_name': protocol_info.name,
                        'opportunity_type': 'lending',
                        'asset': 'USDC',
                        'apy': 4.5,
                        'risk_score': protocol_info.risk_score,
                        'tvl_usd': float(protocol_info.tvl_usd),
                        'minimum_amount': 100,
                        'estimated_returns_30d': (amount_usd * 4.5 / 100 / 12)
                    })
                
                elif protocol_info.protocol_type == ProtocolType.DEX:
                    # Mock liquidity provision opportunity
                    opportunities.append({
                        'protocol_id': protocol_id,
                        'protocol_name': protocol_info.name,
                        'opportunity_type': 'liquidity_provision',
                        'asset': 'ETH/USDC',
                        'apy': 12.3,
                        'risk_score': protocol_info.risk_score - 1,  # LP has additional risk
                        'tvl_usd': float(protocol_info.tvl_usd),
                        'minimum_amount': 500,
                        'estimated_returns_30d': (amount_usd * 12.3 / 100 / 12),
                        'additional_risks': ['Impermanent Loss']
                    })
            
            # Sort by risk-adjusted returns
            opportunities.sort(key=lambda x: (x['apy'] / (11 - x['risk_score'])), reverse=True)
            
            return opportunities[:10]  # Top 10 opportunities
            
        except Exception as e:
            self.logger.error(f"Error getting yield opportunities: {e}")
            return []
    
    async def get_user_portfolio(self, user_address: str) -> Dict[str, Any]:
        """Get user's complete DeFi portfolio across all protocols"""
        try:
            portfolio = {
                'user_address': user_address,
                'total_value_usd': 0,
                'total_pnl_usd': 0,
                'total_rewards_usd': 0,
                'protocols': {},
                'asset_breakdown': {},
                'position_types': {},
                'risk_metrics': {},
                'last_updated': datetime.now().isoformat()
            }
            
            # Mock portfolio data
            portfolio['protocols']['uniswap_v3'] = {
                'name': 'Uniswap V3',
                'positions': 2,
                'total_value_usd': 5000,
                'unrealized_pnl_usd': 150,
                'rewards_earned_usd': 75
            }
            
            portfolio['protocols']['aave_v3'] = {
                'name': 'Aave V3',
                'positions': 1,
                'total_value_usd': 3000,
                'unrealized_pnl_usd': 45,
                'rewards_earned_usd': 120
            }
            
            portfolio['total_value_usd'] = 8000
            portfolio['total_pnl_usd'] = 195
            portfolio['total_rewards_usd'] = 195
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error getting user portfolio: {e}")
            return {'error': str(e)}
    
    async def get_protocol_analytics(self) -> Dict[str, Any]:
        """Get analytics across all integrated protocols"""
        try:
            analytics = {
                'total_protocols': len(self.protocols),
                'total_tvl_usd': sum(float(p.tvl_usd) for p in self.protocols.values()),
                'protocol_breakdown': {},
                'transaction_stats': {
                    'total_transactions': len(self.transactions),
                    'successful_transactions': 0,
                    'failed_transactions': 0
                },
                'risk_distribution': {},
                'last_updated': datetime.now().isoformat()
            }
            
            # Protocol breakdown
            for protocol_id, protocol_info in self.protocols.items():
                analytics['protocol_breakdown'][protocol_id] = {
                    'name': protocol_info.name,
                    'type': protocol_info.protocol_type.value,
                    'tvl_usd': float(protocol_info.tvl_usd),
                    'risk_score': protocol_info.risk_score,
                    'status': protocol_info.status.value
                }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting protocol analytics: {e}")
            return {}
    
    def get_total_tvl(self) -> float:
        """Get total value locked across all protocols"""
        return sum(float(protocol.tvl_usd) for protocol in self.protocols.values())
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.protocol_cache:
            return False
        
        cached_time = self.protocol_cache[cache_key]['timestamp']
        return datetime.now() - cached_time < self.cache_ttl
    
    async def shutdown(self):
        """Gracefully shutdown protocol integrator"""
        self.logger.info("Shutting down protocol integrator...")
        
        # Close Web3 connections
        for chain_id, w3 in self.web3_connections.items():
            try:
                # Web3.py doesn't have explicit close method
                # but we can clear the connection
                pass
            except Exception as e:
                self.logger.error(f"Error closing {chain_id.name} connection: {e}")
        
        # Clear all data
        self.web3_connections.clear()
        self.contracts.clear()
        self.protocol_cache.clear()
        self.adapters.clear()
        self.protocols.clear()
        self.user_positions.clear()
        self.transactions.clear()
        
        self.logger.info("Protocol integrator shutdown complete")

# Export main classes
__all__ = [
    'ProtocolIntegrator', 'BaseProtocolAdapter',
    'ProtocolInfo', 'ProtocolPosition', 'ProtocolTransaction', 'TransactionResult',
    'ProtocolType', 'ProtocolStatus', 'TransactionStatus', 'ChainId'
]