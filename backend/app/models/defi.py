"""DeFi (Decentralized Finance) related database models.

This module contains models for:
- DeFi protocols and platforms
- Liquidity pools and yield farming
- Staking and governance tokens
- Cross-chain bridge transactions
- DeFi portfolio tracking
"""

from typing import Optional, Dict, Any, List
from decimal import Decimal
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, Integer,
    ForeignKey, Enum as SQLEnum, UniqueConstraint, Index,
    Numeric, CheckConstraint, Date
)
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import relationship, validates
from datetime import datetime, date
import uuid
from enum import Enum

from app.models.base import BaseModel, TimestampMixin


class BlockchainNetwork(str, Enum):
    """Supported blockchain networks."""
    ETHEREUM = "ethereum"
    BINANCE_SMART_CHAIN = "binance_smart_chain"
    POLYGON = "polygon"
    AVALANCHE = "avalanche"
    FANTOM = "fantom"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    SOLANA = "solana"
    CARDANO = "cardano"
    POLKADOT = "polkadot"
    COSMOS = "cosmos"
    TERRA = "terra"
    NEAR = "near"
    ALGORAND = "algorand"


class ProtocolType(str, Enum):
    """DeFi protocol types."""
    DEX = "dex"  # Decentralized Exchange
    LENDING = "lending"
    BORROWING = "borrowing"
    YIELD_FARMING = "yield_farming"
    LIQUIDITY_MINING = "liquidity_mining"
    STAKING = "staking"
    GOVERNANCE = "governance"
    INSURANCE = "insurance"
    DERIVATIVES = "derivatives"
    SYNTHETIC = "synthetic"
    BRIDGE = "bridge"
    LAUNCHPAD = "launchpad"
    NFT_MARKETPLACE = "nft_marketplace"
    GAMING = "gaming"
    METAVERSE = "metaverse"


class PoolType(str, Enum):
    """Liquidity pool types."""
    CONSTANT_PRODUCT = "constant_product"  # Uniswap V2 style
    CONSTANT_SUM = "constant_sum"
    WEIGHTED = "weighted"  # Balancer style
    STABLE = "stable"  # Curve style
    CONCENTRATED = "concentrated"  # Uniswap V3 style
    ORDERBOOK = "orderbook"
    LENDING = "lending"
    STAKING = "staking"


class TransactionType(str, Enum):
    """DeFi transaction types."""
    SWAP = "swap"
    ADD_LIQUIDITY = "add_liquidity"
    REMOVE_LIQUIDITY = "remove_liquidity"
    STAKE = "stake"
    UNSTAKE = "unstake"
    CLAIM_REWARDS = "claim_rewards"
    LEND = "lend"
    BORROW = "borrow"
    REPAY = "repay"
    LIQUIDATE = "liquidate"
    BRIDGE = "bridge"
    GOVERNANCE_VOTE = "governance_vote"
    NFT_MINT = "nft_mint"
    NFT_TRADE = "nft_trade"


class TransactionStatus(str, Enum):
    """Transaction status."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REVERTED = "reverted"


class StakingType(str, Enum):
    """Staking types."""
    VALIDATOR = "validator"
    DELEGATED = "delegated"
    LIQUIDITY = "liquidity"
    GOVERNANCE = "governance"
    YIELD_FARMING = "yield_farming"
    SINGLE_ASSET = "single_asset"
    LP_TOKEN = "lp_token"


class RewardType(str, Enum):
    """Reward types."""
    TOKEN = "token"
    LP_TOKEN = "lp_token"
    NFT = "nft"
    GOVERNANCE_TOKEN = "governance_token"
    NATIVE_TOKEN = "native_token"
    STABLE_COIN = "stable_coin"


class DeFiProtocol(BaseModel):
    """DeFi protocol model."""
    
    __tablename__ = "defi_protocols"
    
    # Protocol information
    name = Column(
        String(100),
        nullable=False,
        unique=True,
        doc="Protocol name"
    )
    
    symbol = Column(
        String(20),
        nullable=True,
        doc="Protocol token symbol"
    )
    
    protocol_type = Column(
        SQLEnum(ProtocolType),
        nullable=False,
        doc="Protocol type"
    )
    
    blockchain_network = Column(
        SQLEnum(BlockchainNetwork),
        nullable=False,
        doc="Primary blockchain network"
    )
    
    # Contract information
    contract_address = Column(
        String(100),
        nullable=True,
        doc="Main contract address"
    )
    
    contract_addresses = Column(
        JSONB,
        nullable=True,
        doc="All contract addresses"
    )
    
    # Protocol details
    description = Column(
        Text,
        nullable=True,
        doc="Protocol description"
    )
    
    website_url = Column(
        String(200),
        nullable=True,
        doc="Official website URL"
    )
    
    documentation_url = Column(
        String(200),
        nullable=True,
        doc="Documentation URL"
    )
    
    # Financial metrics
    total_value_locked = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Total Value Locked (TVL)"
    )
    
    market_cap = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Market capitalization"
    )
    
    volume_24h = Column(
        Numeric(30, 8),
        nullable=True,
        doc="24-hour trading volume"
    )
    
    # Risk and security
    audit_status = Column(
        String(50),
        nullable=True,
        doc="Audit status"
    )
    
    audit_reports = Column(
        JSONB,
        nullable=True,
        doc="Audit report information"
    )
    
    risk_score = Column(
        Numeric(5, 2),
        nullable=True,
        doc="Risk score (1-10)"
    )
    
    # Status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Protocol is active"
    )
    
    is_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Protocol is verified"
    )
    
    # Metadata
    supported_networks = Column(
        ARRAY(String),
        nullable=True,
        doc="Supported blockchain networks"
    )
    
    features = Column(
        ARRAY(String),
        nullable=True,
        doc="Protocol features"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional protocol metadata"
    )
    
    # Relationships
    pools = relationship("LiquidityPool", back_populates="protocol")
    positions = relationship("DeFiPosition", back_populates="protocol")
    transactions = relationship("DeFiTransaction", back_populates="protocol")
    
    # Constraints
    __table_args__ = (
        Index('idx_defi_protocol_type_network', 'protocol_type', 'blockchain_network'),
        Index('idx_defi_protocol_tvl', 'total_value_locked'),
        Index('idx_defi_protocol_active', 'is_active'),
    )
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<DeFiProtocol(id={self.id}, name={self.name}, "
            f"type={self.protocol_type.value}, network={self.blockchain_network.value})>"
        )


class LiquidityPool(BaseModel):
    """Liquidity pool model."""
    
    __tablename__ = "liquidity_pools"
    
    # Foreign keys
    protocol_id = Column(
        UUID(as_uuid=True),
        ForeignKey("defi_protocols.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to DeFi protocol"
    )
    
    # Pool information
    name = Column(
        String(100),
        nullable=False,
        doc="Pool name"
    )
    
    symbol = Column(
        String(50),
        nullable=True,
        doc="Pool token symbol"
    )
    
    pool_type = Column(
        SQLEnum(PoolType),
        nullable=False,
        doc="Pool type"
    )
    
    # Contract information
    pool_address = Column(
        String(100),
        nullable=False,
        doc="Pool contract address"
    )
    
    lp_token_address = Column(
        String(100),
        nullable=True,
        doc="LP token contract address"
    )
    
    # Token composition
    token_addresses = Column(
        ARRAY(String),
        nullable=False,
        doc="Token contract addresses in pool"
    )
    
    token_symbols = Column(
        ARRAY(String),
        nullable=False,
        doc="Token symbols in pool"
    )
    
    token_weights = Column(
        ARRAY(Numeric),
        nullable=True,
        doc="Token weights in pool"
    )
    
    # Pool metrics
    total_liquidity = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Total liquidity in USD"
    )
    
    volume_24h = Column(
        Numeric(30, 8),
        nullable=True,
        doc="24-hour trading volume"
    )
    
    volume_7d = Column(
        Numeric(30, 8),
        nullable=True,
        doc="7-day trading volume"
    )
    
    fees_24h = Column(
        Numeric(30, 8),
        nullable=True,
        doc="24-hour fees collected"
    )
    
    # APY and rewards
    apy = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Annual Percentage Yield"
    )
    
    apr = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Annual Percentage Rate"
    )
    
    reward_tokens = Column(
        ARRAY(String),
        nullable=True,
        doc="Reward token addresses"
    )
    
    reward_rates = Column(
        JSONB,
        nullable=True,
        doc="Reward emission rates"
    )
    
    # Pool parameters
    fee_rate = Column(
        Numeric(10, 6),
        nullable=True,
        doc="Pool fee rate"
    )
    
    swap_fee = Column(
        Numeric(10, 6),
        nullable=True,
        doc="Swap fee percentage"
    )
    
    # Status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Pool is active"
    )
    
    is_incentivized = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Pool has additional rewards"
    )
    
    # Metadata
    pool_data = Column(
        JSONB,
        nullable=True,
        doc="Additional pool data"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Pool metadata"
    )
    
    # Relationships
    protocol = relationship("DeFiProtocol", back_populates="pools")
    positions = relationship("DeFiPosition", back_populates="pool")
    
    # Constraints
    __table_args__ = (
        Index('idx_liquidity_pool_protocol_type', 'protocol_id', 'pool_type'),
        Index('idx_liquidity_pool_address', 'pool_address'),
        Index('idx_liquidity_pool_liquidity', 'total_liquidity'),
        Index('idx_liquidity_pool_apy', 'apy'),
        UniqueConstraint('protocol_id', 'pool_address', name='uq_pool_protocol_address'),
    )
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<LiquidityPool(id={self.id}, name={self.name}, "
            f"type={self.pool_type.value}, liquidity={self.total_liquidity})>"
        )


class DeFiPosition(BaseModel):
    """DeFi position model."""
    
    __tablename__ = "defi_positions"
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    portfolio_id = Column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to portfolio"
    )
    
    protocol_id = Column(
        UUID(as_uuid=True),
        ForeignKey("defi_protocols.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to DeFi protocol"
    )
    
    pool_id = Column(
        UUID(as_uuid=True),
        ForeignKey("liquidity_pools.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to liquidity pool"
    )
    
    # Position information
    position_type = Column(
        String(50),
        nullable=False,
        doc="Position type (liquidity, staking, lending, etc.)"
    )
    
    wallet_address = Column(
        String(100),
        nullable=False,
        doc="User wallet address"
    )
    
    # Token information
    token_address = Column(
        String(100),
        nullable=True,
        doc="Primary token contract address"
    )
    
    token_symbol = Column(
        String(20),
        nullable=True,
        doc="Primary token symbol"
    )
    
    # Position details
    amount = Column(
        Numeric(30, 18),
        nullable=False,
        doc="Position amount"
    )
    
    amount_usd = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Position value in USD"
    )
    
    entry_price = Column(
        Numeric(30, 18),
        nullable=True,
        doc="Entry price"
    )
    
    current_price = Column(
        Numeric(30, 18),
        nullable=True,
        doc="Current price"
    )
    
    # Rewards and earnings
    unclaimed_rewards = Column(
        JSONB,
        nullable=True,
        doc="Unclaimed rewards by token"
    )
    
    claimed_rewards = Column(
        JSONB,
        nullable=True,
        doc="Total claimed rewards by token"
    )
    
    total_earned = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Total earned in USD"
    )
    
    # Performance metrics
    pnl = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Profit and Loss in USD"
    )
    
    pnl_percentage = Column(
        Numeric(10, 4),
        nullable=True,
        doc="P&L percentage"
    )
    
    apy_earned = Column(
        Numeric(10, 4),
        nullable=True,
        doc="APY earned"
    )
    
    # Timing
    opened_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        doc="Position opening time"
    )
    
    last_updated = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        doc="Last update time"
    )
    
    closed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Position closing time"
    )
    
    # Status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Position is active"
    )
    
    # Metadata
    position_data = Column(
        JSONB,
        nullable=True,
        doc="Additional position data"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Position metadata"
    )
    
    # Relationships
    user = relationship("User")
    portfolio = relationship("Portfolio")
    protocol = relationship("DeFiProtocol", back_populates="positions")
    pool = relationship("LiquidityPool", back_populates="positions")
    transactions = relationship("DeFiTransaction", back_populates="position")
    
    # Constraints
    __table_args__ = (
        Index('idx_defi_position_user_protocol', 'user_id', 'protocol_id'),
        Index('idx_defi_position_wallet', 'wallet_address'),
        Index('idx_defi_position_active', 'is_active'),
        Index('idx_defi_position_opened', 'opened_at'),
    )
    
    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable.
        
        Returns:
            True if position has positive P&L
        """
        return self.pnl is not None and self.pnl > 0
    
    @property
    def duration_days(self) -> Optional[int]:
        """Get position duration in days.
        
        Returns:
            Duration in days
        """
        if self.closed_at:
            return (self.closed_at - self.opened_at).days
        return (datetime.utcnow() - self.opened_at).days
    
    def close_position(self) -> None:
        """Close the position."""
        self.is_active = False
        self.closed_at = datetime.utcnow()
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<DeFiPosition(id={self.id}, type={self.position_type}, "
            f"amount={self.amount}, pnl={self.pnl})>"
        )


class DeFiTransaction(BaseModel):
    """DeFi transaction model."""
    
    __tablename__ = "defi_transactions"
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    protocol_id = Column(
        UUID(as_uuid=True),
        ForeignKey("defi_protocols.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to DeFi protocol"
    )
    
    position_id = Column(
        UUID(as_uuid=True),
        ForeignKey("defi_positions.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to DeFi position"
    )
    
    # Transaction information
    transaction_hash = Column(
        String(100),
        nullable=False,
        unique=True,
        doc="Blockchain transaction hash"
    )
    
    transaction_type = Column(
        SQLEnum(TransactionType),
        nullable=False,
        doc="Transaction type"
    )
    
    blockchain_network = Column(
        SQLEnum(BlockchainNetwork),
        nullable=False,
        doc="Blockchain network"
    )
    
    # Addresses
    from_address = Column(
        String(100),
        nullable=False,
        doc="From wallet address"
    )
    
    to_address = Column(
        String(100),
        nullable=False,
        doc="To contract/wallet address"
    )
    
    # Token information
    token_in_address = Column(
        String(100),
        nullable=True,
        doc="Input token contract address"
    )
    
    token_in_symbol = Column(
        String(20),
        nullable=True,
        doc="Input token symbol"
    )
    
    token_in_amount = Column(
        Numeric(30, 18),
        nullable=True,
        doc="Input token amount"
    )
    
    token_out_address = Column(
        String(100),
        nullable=True,
        doc="Output token contract address"
    )
    
    token_out_symbol = Column(
        String(20),
        nullable=True,
        doc="Output token symbol"
    )
    
    token_out_amount = Column(
        Numeric(30, 18),
        nullable=True,
        doc="Output token amount"
    )
    
    # Financial details
    value_usd = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Transaction value in USD"
    )
    
    gas_fee = Column(
        Numeric(30, 18),
        nullable=True,
        doc="Gas fee paid"
    )
    
    gas_fee_usd = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Gas fee in USD"
    )
    
    protocol_fee = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Protocol fee paid"
    )
    
    # Block information
    block_number = Column(
        Integer,
        nullable=True,
        doc="Block number"
    )
    
    block_timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        doc="Block timestamp"
    )
    
    # Status
    status = Column(
        SQLEnum(TransactionStatus),
        default=TransactionStatus.PENDING,
        nullable=False,
        doc="Transaction status"
    )
    
    # Additional data
    transaction_data = Column(
        JSONB,
        nullable=True,
        doc="Additional transaction data"
    )
    
    logs = Column(
        JSONB,
        nullable=True,
        doc="Transaction logs"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Transaction metadata"
    )
    
    # Relationships
    user = relationship("User")
    protocol = relationship("DeFiProtocol", back_populates="transactions")
    position = relationship("DeFiPosition", back_populates="transactions")
    
    # Constraints
    __table_args__ = (
        Index('idx_defi_transaction_user_type', 'user_id', 'transaction_type'),
        Index('idx_defi_transaction_protocol_type', 'protocol_id', 'transaction_type'),
        Index('idx_defi_transaction_hash', 'transaction_hash'),
        Index('idx_defi_transaction_timestamp', 'block_timestamp'),
        Index('idx_defi_transaction_status', 'status'),
        Index('idx_defi_transaction_from_address', 'from_address'),
    )
    
    @property
    def is_successful(self) -> bool:
        """Check if transaction is successful.
        
        Returns:
            True if transaction is confirmed
        """
        return self.status == TransactionStatus.CONFIRMED
    
    @property
    def total_cost(self) -> Optional[Decimal]:
        """Get total transaction cost including fees.
        
        Returns:
            Total cost in USD
        """
        if self.value_usd is None:
            return None
        
        total = self.value_usd
        if self.gas_fee_usd:
            total += self.gas_fee_usd
        if self.protocol_fee:
            total += self.protocol_fee
        
        return total
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<DeFiTransaction(id={self.id}, hash={self.transaction_hash[:10]}..., "
            f"type={self.transaction_type.value}, status={self.status.value})>"
        )


class StakingPosition(BaseModel):
    """Staking position model."""
    
    __tablename__ = "staking_positions"
    
    # Foreign keys
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    protocol_id = Column(
        UUID(as_uuid=True),
        ForeignKey("defi_protocols.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to DeFi protocol"
    )
    
    # Staking information
    staking_type = Column(
        SQLEnum(StakingType),
        nullable=False,
        doc="Staking type"
    )
    
    validator_address = Column(
        String(100),
        nullable=True,
        doc="Validator address (for delegated staking)"
    )
    
    # Token information
    token_address = Column(
        String(100),
        nullable=False,
        doc="Staked token contract address"
    )
    
    token_symbol = Column(
        String(20),
        nullable=False,
        doc="Staked token symbol"
    )
    
    # Staking details
    staked_amount = Column(
        Numeric(30, 18),
        nullable=False,
        doc="Staked token amount"
    )
    
    staked_value_usd = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Staked value in USD"
    )
    
    # Rewards
    reward_token_address = Column(
        String(100),
        nullable=True,
        doc="Reward token contract address"
    )
    
    reward_token_symbol = Column(
        String(20),
        nullable=True,
        doc="Reward token symbol"
    )
    
    unclaimed_rewards = Column(
        Numeric(30, 18),
        nullable=True,
        doc="Unclaimed reward amount"
    )
    
    claimed_rewards = Column(
        Numeric(30, 18),
        nullable=True,
        doc="Total claimed rewards"
    )
    
    rewards_value_usd = Column(
        Numeric(30, 8),
        nullable=True,
        doc="Total rewards value in USD"
    )
    
    # Performance
    apy = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Annual Percentage Yield"
    )
    
    apr = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Annual Percentage Rate"
    )
    
    # Lock period
    lock_period_days = Column(
        Integer,
        nullable=True,
        doc="Lock period in days"
    )
    
    unlock_date = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Unlock date"
    )
    
    # Timing
    staked_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        doc="Staking start time"
    )
    
    last_claim_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last reward claim time"
    )
    
    unstaked_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Unstaking time"
    )
    
    # Status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Staking position is active"
    )
    
    is_locked = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Tokens are locked"
    )
    
    # Metadata
    staking_data = Column(
        JSONB,
        nullable=True,
        doc="Additional staking data"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Staking metadata"
    )
    
    # Relationships
    user = relationship("User")
    protocol = relationship("DeFiProtocol")
    
    # Constraints
    __table_args__ = (
        Index('idx_staking_position_user_protocol', 'user_id', 'protocol_id'),
        Index('idx_staking_position_type', 'staking_type'),
        Index('idx_staking_position_active', 'is_active'),
        Index('idx_staking_position_unlock', 'unlock_date'),
    )
    
    @property
    def is_unlocked(self) -> bool:
        """Check if staking position is unlocked.
        
        Returns:
            True if position is unlocked
        """
        if not self.is_locked or self.unlock_date is None:
            return True
        return datetime.utcnow() >= self.unlock_date
    
    @property
    def days_staked(self) -> int:
        """Get number of days staked.
        
        Returns:
            Days staked
        """
        end_date = self.unstaked_at or datetime.utcnow()
        return (end_date - self.staked_at).days
    
    @property
    def total_rewards_earned(self) -> Optional[Decimal]:
        """Get total rewards earned.
        
        Returns:
            Total rewards (claimed + unclaimed)
        """
        total = Decimal('0')
        if self.claimed_rewards:
            total += self.claimed_rewards
        if self.unclaimed_rewards:
            total += self.unclaimed_rewards
        return total if total > 0 else None
    
    def unstake(self) -> None:
        """Unstake the position."""
        self.is_active = False
        self.unstaked_at = datetime.utcnow()
    
    def claim_rewards(self, amount: Decimal) -> None:
        """Claim rewards.
        
        Args:
            amount: Amount of rewards claimed
        """
        if self.claimed_rewards is None:
            self.claimed_rewards = Decimal('0')
        self.claimed_rewards += amount
        
        if self.unclaimed_rewards is not None:
            self.unclaimed_rewards = max(Decimal('0'), self.unclaimed_rewards - amount)
        
        self.last_claim_at = datetime.utcnow()
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<StakingPosition(id={self.id}, type={self.staking_type.value}, "
            f"amount={self.staked_amount}, apy={self.apy})>"
        )