"""Trading-related database models.

This module contains models for:
- Trading orders and executions
- Trading strategies and algorithms
- Risk management and position sizing
- Broker integrations and accounts
"""

from typing import Optional, Dict, Any, List
from decimal import Decimal
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, Integer,
    ForeignKey, Enum as SQLEnum, UniqueConstraint, Index,
    Numeric, CheckConstraint
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from datetime import datetime
import uuid
from enum import Enum

from app.models.base import BaseModel, SoftDeleteModel, AuditModel
from app.models.portfolio import AssetType


class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    BRACKET = "bracket"
    OCO = "oco"  # One-Cancels-Other
    ICEBERG = "iceberg"
    TWAP = "twap"  # Time-Weighted Average Price
    VWAP = "vwap"  # Volume-Weighted Average Price


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"
    BUY_TO_COVER = "buy_to_cover"
    SELL_SHORT = "sell_short"


class OrderStatus(str, Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    SUSPENDED = "suspended"


class TimeInForce(str, Enum):
    """Time in force options."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    GTD = "gtd"  # Good Till Date
    ATC = "atc"  # At The Close
    ATO = "ato"  # At The Open


class PositionSide(str, Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class StrategyType(str, Enum):
    """Trading strategy types."""
    MANUAL = "manual"
    ALGORITHMIC = "algorithmic"
    COPY_TRADING = "copy_trading"
    SOCIAL_TRADING = "social_trading"
    ROBO_ADVISOR = "robo_advisor"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    GRID_TRADING = "grid_trading"
    DCA = "dca"  # Dollar Cost Averaging


class StrategyStatus(str, Enum):
    """Strategy status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    BACKTESTING = "backtesting"


class BrokerAccount(SoftDeleteModel):
    """Broker account model."""
    
    __tablename__ = "broker_accounts"
    
    # Foreign key to user
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    # Account information
    broker_name = Column(
        String(100),
        nullable=False,
        doc="Broker name"
    )
    
    account_number = Column(
        String(100),
        nullable=False,
        doc="Broker account number"
    )
    
    account_name = Column(
        String(100),
        nullable=True,
        doc="Account display name"
    )
    
    account_type = Column(
        String(50),
        nullable=True,
        doc="Account type (cash, margin, etc.)"
    )
    
    # Connection details
    api_key = Column(
        String(255),
        nullable=True,
        doc="Encrypted API key"
    )
    
    api_secret = Column(
        String(255),
        nullable=True,
        doc="Encrypted API secret"
    )
    
    access_token = Column(
        String(500),
        nullable=True,
        doc="OAuth access token"
    )
    
    refresh_token = Column(
        String(500),
        nullable=True,
        doc="OAuth refresh token"
    )
    
    # Status and settings
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Account is active"
    )
    
    is_connected = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Connection status"
    )
    
    is_paper_trading = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Paper trading account"
    )
    
    auto_sync = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Auto sync positions and orders"
    )
    
    # Account balances
    cash_balance = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Cash balance"
    )
    
    buying_power = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Buying power"
    )
    
    portfolio_value = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Total portfolio value"
    )
    
    # Connection tracking
    last_sync = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last synchronization"
    )
    
    last_error = Column(
        Text,
        nullable=True,
        doc="Last connection error"
    )
    
    # Metadata
    settings = Column(
        JSONB,
        nullable=True,
        doc="Broker-specific settings"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional account metadata"
    )
    
    # Relationships
    user = relationship("User")
    orders = relationship(
        "TradingOrder",
        back_populates="broker_account",
        cascade="all, delete-orphan"
    )
    positions = relationship(
        "Position",
        back_populates="broker_account",
        cascade="all, delete-orphan"
    )
    
    # Constraints
    __table_args__ = (
        UniqueConstraint(
            'user_id', 'broker_name', 'account_number',
            name='uq_broker_account_user_broker_number'
        ),
        Index('idx_broker_account_user_active', 'user_id', 'is_active'),
        Index('idx_broker_account_broker', 'broker_name'),
    )
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<BrokerAccount(id={self.id}, broker={self.broker_name}, "
            f"account={self.account_number})>"
        )


class TradingOrder(AuditModel):
    """Trading order model."""
    
    __tablename__ = "trading_orders"
    
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
    
    broker_account_id = Column(
        UUID(as_uuid=True),
        ForeignKey("broker_accounts.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to broker account"
    )
    
    strategy_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trading_strategies.id", ondelete="SET NULL"),
        nullable=True,
        doc="Reference to trading strategy"
    )
    
    # Order identification
    client_order_id = Column(
        String(100),
        nullable=False,
        unique=True,
        doc="Client-side order ID"
    )
    
    broker_order_id = Column(
        String(100),
        nullable=True,
        doc="Broker-side order ID"
    )
    
    # Asset information
    symbol = Column(
        String(20),
        nullable=False,
        doc="Asset symbol"
    )
    
    asset_type = Column(
        SQLEnum(AssetType),
        nullable=False,
        doc="Asset type"
    )
    
    # Order details
    order_type = Column(
        SQLEnum(OrderType),
        nullable=False,
        doc="Order type"
    )
    
    order_side = Column(
        SQLEnum(OrderSide),
        nullable=False,
        doc="Order side"
    )
    
    quantity = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Order quantity"
    )
    
    # Pricing
    limit_price = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Limit price"
    )
    
    stop_price = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Stop price"
    )
    
    trailing_amount = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Trailing stop amount"
    )
    
    trailing_percent = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Trailing stop percentage"
    )
    
    # Execution details
    time_in_force = Column(
        SQLEnum(TimeInForce),
        default=TimeInForce.DAY,
        nullable=False,
        doc="Time in force"
    )
    
    good_till_date = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Good till date"
    )
    
    # Status and execution
    status = Column(
        SQLEnum(OrderStatus),
        default=OrderStatus.PENDING,
        nullable=False,
        doc="Order status"
    )
    
    filled_quantity = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Filled quantity"
    )
    
    average_fill_price = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Average fill price"
    )
    
    # Timing
    submitted_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Order submission time"
    )
    
    filled_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Order fill time"
    )
    
    cancelled_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Order cancellation time"
    )
    
    # Error handling
    rejection_reason = Column(
        Text,
        nullable=True,
        doc="Order rejection reason"
    )
    
    # Fees and costs
    commission = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Commission paid"
    )
    
    fees = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Additional fees"
    )
    
    # Metadata
    notes = Column(
        Text,
        nullable=True,
        doc="Order notes"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional order metadata"
    )
    
    # Relationships
    user = relationship("User")
    portfolio = relationship("Portfolio")
    broker_account = relationship("BrokerAccount", back_populates="orders")
    strategy = relationship("TradingStrategy", back_populates="orders")
    executions = relationship(
        "OrderExecution",
        back_populates="order",
        cascade="all, delete-orphan"
    )
    
    # Constraints
    __table_args__ = (
        Index('idx_trading_order_user_status', 'user_id', 'status'),
        Index('idx_trading_order_symbol_status', 'symbol', 'status'),
        Index('idx_trading_order_broker_id', 'broker_order_id'),
        Index('idx_trading_order_submitted', 'submitted_at'),
        CheckConstraint('quantity > 0', name='ck_trading_order_quantity_positive'),
        CheckConstraint('filled_quantity >= 0', name='ck_trading_order_filled_quantity_positive'),
        CheckConstraint('filled_quantity <= quantity', name='ck_trading_order_filled_lte_quantity'),
    )
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Calculate remaining quantity to fill.
        
        Returns:
            Remaining quantity
        """
        return self.quantity - self.filled_quantity
    
    @property
    def fill_percentage(self) -> Decimal:
        """Calculate fill percentage.
        
        Returns:
            Fill percentage
        """
        if self.quantity == 0:
            return Decimal('0')
        return (self.filled_quantity / self.quantity) * 100
    
    @property
    def is_filled(self) -> bool:
        """Check if order is completely filled.
        
        Returns:
            True if order is filled
        """
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        """Check if order is active.
        
        Returns:
            True if order is active
        """
        return self.status in [
            OrderStatus.PENDING,
            OrderStatus.SUBMITTED,
            OrderStatus.ACCEPTED,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    @property
    def total_value(self) -> Optional[Decimal]:
        """Calculate total order value.
        
        Returns:
            Total value or None if no fill price
        """
        if self.average_fill_price:
            return self.filled_quantity * self.average_fill_price
        elif self.limit_price:
            return self.quantity * self.limit_price
        return None
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<TradingOrder(id={self.id}, symbol={self.symbol}, "
            f"side={self.order_side.value}, quantity={self.quantity}, "
            f"status={self.status.value})>"
        )


class OrderExecution(BaseModel):
    """Order execution/fill model."""
    
    __tablename__ = "order_executions"
    
    # Foreign key to order
    order_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trading_orders.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to trading order"
    )
    
    # Execution details
    execution_id = Column(
        String(100),
        nullable=True,
        doc="Broker execution ID"
    )
    
    quantity = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Executed quantity"
    )
    
    price = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Execution price"
    )
    
    # Timing
    executed_at = Column(
        DateTime(timezone=True),
        nullable=False,
        doc="Execution timestamp"
    )
    
    # Costs
    commission = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Commission for this execution"
    )
    
    fees = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Fees for this execution"
    )
    
    # Metadata
    venue = Column(
        String(50),
        nullable=True,
        doc="Execution venue"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional execution metadata"
    )
    
    # Relationships
    order = relationship("TradingOrder", back_populates="executions")
    
    # Constraints
    __table_args__ = (
        Index('idx_order_execution_order_time', 'order_id', 'executed_at'),
        CheckConstraint('quantity > 0', name='ck_order_execution_quantity_positive'),
        CheckConstraint('price > 0', name='ck_order_execution_price_positive'),
    )
    
    @property
    def gross_value(self) -> Decimal:
        """Calculate gross execution value.
        
        Returns:
            Gross value
        """
        return self.quantity * self.price
    
    @property
    def net_value(self) -> Decimal:
        """Calculate net execution value.
        
        Returns:
            Net value after fees
        """
        return self.gross_value - self.commission - self.fees
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<OrderExecution(id={self.id}, order_id={self.order_id}, "
            f"quantity={self.quantity}, price={self.price})>"
        )


class Position(BaseModel):
    """Trading position model."""
    
    __tablename__ = "positions"
    
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
    
    broker_account_id = Column(
        UUID(as_uuid=True),
        ForeignKey("broker_accounts.id", ondelete="CASCADE"),
        nullable=True,
        doc="Reference to broker account"
    )
    
    # Asset information
    symbol = Column(
        String(20),
        nullable=False,
        doc="Asset symbol"
    )
    
    asset_type = Column(
        SQLEnum(AssetType),
        nullable=False,
        doc="Asset type"
    )
    
    # Position details
    side = Column(
        SQLEnum(PositionSide),
        nullable=False,
        doc="Position side"
    )
    
    quantity = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Position quantity"
    )
    
    average_price = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Average entry price"
    )
    
    current_price = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Current market price"
    )
    
    # P&L calculations
    unrealized_pnl = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Unrealized profit/loss"
    )
    
    realized_pnl = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Realized profit/loss"
    )
    
    # Cost basis
    cost_basis = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Total cost basis"
    )
    
    market_value = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Current market value"
    )
    
    # Timing
    opened_at = Column(
        DateTime(timezone=True),
        nullable=False,
        doc="Position open time"
    )
    
    closed_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Position close time"
    )
    
    last_updated = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        doc="Last update time"
    )
    
    # Status
    is_open = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Position is open"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional position metadata"
    )
    
    # Relationships
    user = relationship("User")
    portfolio = relationship("Portfolio")
    broker_account = relationship("BrokerAccount", back_populates="positions")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint(
            'broker_account_id', 'symbol', 'side',
            name='uq_position_account_symbol_side'
        ),
        Index('idx_position_user_symbol', 'user_id', 'symbol'),
        Index('idx_position_portfolio_open', 'portfolio_id', 'is_open'),
        CheckConstraint('quantity >= 0', name='ck_position_quantity_positive'),
        CheckConstraint('average_price > 0', name='ck_position_average_price_positive'),
    )
    
    @property
    def total_pnl(self) -> Decimal:
        """Calculate total P&L.
        
        Returns:
            Total profit/loss
        """
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def pnl_percentage(self) -> Decimal:
        """Calculate P&L percentage.
        
        Returns:
            P&L percentage
        """
        if self.cost_basis == 0:
            return Decimal('0')
        return (self.total_pnl / self.cost_basis) * 100
    
    def update_market_data(self, current_price: Decimal) -> None:
        """Update position with current market data.
        
        Args:
            current_price: Current market price
        """
        self.current_price = current_price
        self.market_value = self.quantity * current_price
        
        if self.side == PositionSide.LONG:
            self.unrealized_pnl = (current_price - self.average_price) * self.quantity
        elif self.side == PositionSide.SHORT:
            self.unrealized_pnl = (self.average_price - current_price) * self.quantity
        else:
            self.unrealized_pnl = Decimal('0')
        
        self.last_updated = datetime.utcnow()
    
    def close_position(self, close_price: Decimal) -> None:
        """Close the position.
        
        Args:
            close_price: Closing price
        """
        self.current_price = close_price
        self.realized_pnl += self.unrealized_pnl
        self.unrealized_pnl = Decimal('0')
        self.is_open = False
        self.closed_at = datetime.utcnow()
        self.last_updated = datetime.utcnow()
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<Position(id={self.id}, symbol={self.symbol}, "
            f"side={self.side.value}, quantity={self.quantity})>"
        )


class TradingStrategy(SoftDeleteModel):
    """Trading strategy model."""
    
    __tablename__ = "trading_strategies"
    
    # Foreign key to user
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    # Strategy information
    name = Column(
        String(100),
        nullable=False,
        doc="Strategy name"
    )
    
    description = Column(
        Text,
        nullable=True,
        doc="Strategy description"
    )
    
    strategy_type = Column(
        SQLEnum(StrategyType),
        nullable=False,
        doc="Strategy type"
    )
    
    # Status and settings
    status = Column(
        SQLEnum(StrategyStatus),
        default=StrategyStatus.INACTIVE,
        nullable=False,
        doc="Strategy status"
    )
    
    is_public = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Strategy is public"
    )
    
    # Configuration
    parameters = Column(
        JSONB,
        nullable=True,
        doc="Strategy parameters"
    )
    
    risk_settings = Column(
        JSONB,
        nullable=True,
        doc="Risk management settings"
    )
    
    # Performance tracking
    total_return = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Total return"
    )
    
    total_return_percentage = Column(
        Numeric(10, 4),
        default=0,
        nullable=False,
        doc="Total return percentage"
    )
    
    win_rate = Column(
        Numeric(5, 2),
        nullable=True,
        doc="Win rate percentage"
    )
    
    sharpe_ratio = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Sharpe ratio"
    )
    
    max_drawdown = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Maximum drawdown"
    )
    
    # Activity tracking
    total_trades = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Total number of trades"
    )
    
    winning_trades = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of winning trades"
    )
    
    losing_trades = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of losing trades"
    )
    
    # Timing
    started_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Strategy start time"
    )
    
    stopped_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Strategy stop time"
    )
    
    last_trade_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last trade time"
    )
    
    # Code and logic
    strategy_code = Column(
        Text,
        nullable=True,
        doc="Strategy implementation code"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional strategy metadata"
    )
    
    # Relationships
    user = relationship("User")
    orders = relationship(
        "TradingOrder",
        back_populates="strategy",
        cascade="all, delete-orphan"
    )
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='uq_trading_strategy_user_name'),
        Index('idx_trading_strategy_user_status', 'user_id', 'status'),
        Index('idx_trading_strategy_type', 'strategy_type'),
        Index('idx_trading_strategy_public', 'is_public'),
    )
    
    @property
    def is_active(self) -> bool:
        """Check if strategy is active.
        
        Returns:
            True if strategy is active
        """
        return self.status == StrategyStatus.ACTIVE
    
    def start(self) -> None:
        """Start the strategy."""
        self.status = StrategyStatus.ACTIVE
        self.started_at = datetime.utcnow()
    
    def stop(self) -> None:
        """Stop the strategy."""
        self.status = StrategyStatus.STOPPED
        self.stopped_at = datetime.utcnow()
    
    def pause(self) -> None:
        """Pause the strategy."""
        self.status = StrategyStatus.PAUSED
    
    def resume(self) -> None:
        """Resume the strategy."""
        self.status = StrategyStatus.ACTIVE
    
    def update_performance(self) -> None:
        """Update strategy performance metrics."""
        # Calculate win rate
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
        else:
            self.win_rate = None
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<TradingStrategy(id={self.id}, name={self.name}, "
            f"type={self.strategy_type.value}, status={self.status.value})>"
        )