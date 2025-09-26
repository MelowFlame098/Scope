"""Portfolio-related database models.

This module contains models for:
- User portfolios and holdings
- Transactions and trade history
- Portfolio performance tracking
- Asset allocation and rebalancing
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


class PortfolioType(str, Enum):
    """Portfolio types."""
    PERSONAL = "personal"
    RETIREMENT = "retirement"
    TRADING = "trading"
    CRYPTO = "crypto"
    DEFI = "defi"
    PAPER = "paper"  # Paper trading
    DEMO = "demo"    # Demo account


class TransactionType(str, Enum):
    """Transaction types."""
    BUY = "buy"
    SELL = "sell"
    DIVIDEND = "dividend"
    INTEREST = "interest"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
    TRANSFER_IN = "transfer_in"
    TRANSFER_OUT = "transfer_out"
    SPLIT = "split"
    MERGER = "merger"
    SPINOFF = "spinoff"
    FEE = "fee"
    TAX = "tax"
    ADJUSTMENT = "adjustment"


class TransactionStatus(str, Enum):
    """Transaction status."""
    PENDING = "pending"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    SETTLED = "settled"


class AssetType(str, Enum):
    """Asset types."""
    STOCK = "stock"
    ETF = "etf"
    MUTUAL_FUND = "mutual_fund"
    BOND = "bond"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    FOREX = "forex"
    OPTION = "option"
    FUTURE = "future"
    CASH = "cash"
    REAL_ESTATE = "real_estate"
    ALTERNATIVE = "alternative"


class Portfolio(SoftDeleteModel):
    """User portfolio model."""
    
    __tablename__ = "portfolios"
    
    # Foreign key to user
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    # Portfolio information
    name = Column(
        String(100),
        nullable=False,
        doc="Portfolio name"
    )
    
    description = Column(
        Text,
        nullable=True,
        doc="Portfolio description"
    )
    
    portfolio_type = Column(
        SQLEnum(PortfolioType),
        default=PortfolioType.PERSONAL,
        nullable=False,
        doc="Portfolio type"
    )
    
    # Portfolio settings
    base_currency = Column(
        String(3),
        default="USD",
        nullable=False,
        doc="Base currency for portfolio"
    )
    
    is_default = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Default portfolio for user"
    )
    
    is_public = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Public portfolio visibility"
    )
    
    # Performance tracking
    initial_value = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Initial portfolio value"
    )
    
    current_value = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Current portfolio value"
    )
    
    total_invested = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Total amount invested"
    )
    
    total_return = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Total return amount"
    )
    
    total_return_percentage = Column(
        Numeric(10, 4),
        default=0,
        nullable=False,
        doc="Total return percentage"
    )
    
    # Risk metrics
    beta = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Portfolio beta"
    )
    
    sharpe_ratio = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Sharpe ratio"
    )
    
    volatility = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Portfolio volatility"
    )
    
    max_drawdown = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Maximum drawdown"
    )
    
    # Allocation targets
    target_allocation = Column(
        JSONB,
        nullable=True,
        doc="Target asset allocation"
    )
    
    rebalance_threshold = Column(
        Numeric(5, 2),
        default=5.0,
        nullable=False,
        doc="Rebalancing threshold percentage"
    )
    
    auto_rebalance = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Enable automatic rebalancing"
    )
    
    # Metadata
    last_updated = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        doc="Last portfolio update"
    )
    
    performance_data = Column(
        JSONB,
        nullable=True,
        doc="Historical performance data"
    )
    
    # Relationships
    user = relationship("User")
    holdings = relationship(
        "Holding",
        back_populates="portfolio",
        cascade="all, delete-orphan"
    )
    transactions = relationship(
        "Transaction",
        back_populates="portfolio",
        cascade="all, delete-orphan"
    )
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('user_id', 'name', name='uq_user_portfolio_name'),
        Index('idx_portfolio_user_type', 'user_id', 'portfolio_type'),
        Index('idx_portfolio_public', 'is_public'),
        CheckConstraint('current_value >= 0', name='ck_portfolio_current_value_positive'),
        CheckConstraint('total_invested >= 0', name='ck_portfolio_total_invested_positive'),
    )
    
    @validates('name')
    def validate_name(self, key, name):
        """Validate portfolio name.
        
        Args:
            key: Field name
            name: Portfolio name
            
        Returns:
            Validated name
            
        Raises:
            ValueError: If name is invalid
        """
        if not name or not name.strip():
            raise ValueError("Portfolio name is required")
        
        name = name.strip()
        if len(name) < 1 or len(name) > 100:
            raise ValueError("Portfolio name must be between 1 and 100 characters")
        
        return name
    
    @property
    def total_return_amount(self) -> Decimal:
        """Calculate total return amount.
        
        Returns:
            Total return amount
        """
        return self.current_value - self.total_invested
    
    @property
    def total_return_percent(self) -> Decimal:
        """Calculate total return percentage.
        
        Returns:
            Total return percentage
        """
        if self.total_invested == 0:
            return Decimal('0')
        
        return (self.total_return_amount / self.total_invested) * 100
    
    def update_performance(self) -> None:
        """Update portfolio performance metrics."""
        # Calculate current value from holdings
        total_value = sum(holding.current_value for holding in self.holdings)
        self.current_value = total_value
        
        # Update return metrics
        self.total_return = self.total_return_amount
        self.total_return_percentage = self.total_return_percent
        
        # Update timestamp
        self.last_updated = datetime.utcnow()
    
    def get_allocation(self) -> Dict[str, Decimal]:
        """Get current asset allocation.
        
        Returns:
            Dictionary of asset type to percentage allocation
        """
        if self.current_value == 0:
            return {}
        
        allocation = {}
        for holding in self.holdings:
            asset_type = holding.asset_type.value
            percentage = (holding.current_value / self.current_value) * 100
            
            if asset_type in allocation:
                allocation[asset_type] += percentage
            else:
                allocation[asset_type] = percentage
        
        return allocation
    
    def needs_rebalancing(self) -> bool:
        """Check if portfolio needs rebalancing.
        
        Returns:
            True if rebalancing is needed
        """
        if not self.target_allocation or not self.auto_rebalance:
            return False
        
        current_allocation = self.get_allocation()
        
        for asset_type, target_percent in self.target_allocation.items():
            current_percent = current_allocation.get(asset_type, 0)
            deviation = abs(current_percent - target_percent)
            
            if deviation > self.rebalance_threshold:
                return True
        
        return False
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return f"<Portfolio(id={self.id}, name={self.name}, user_id={self.user_id})>"


class Holding(BaseModel):
    """Portfolio holding model."""
    
    __tablename__ = "holdings"
    
    # Foreign key to portfolio
    portfolio_id = Column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to portfolio"
    )
    
    # Asset information
    symbol = Column(
        String(20),
        nullable=False,
        doc="Asset symbol/ticker"
    )
    
    asset_type = Column(
        SQLEnum(AssetType),
        nullable=False,
        doc="Type of asset"
    )
    
    asset_name = Column(
        String(200),
        nullable=True,
        doc="Full asset name"
    )
    
    # Position information
    quantity = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Quantity held"
    )
    
    average_cost = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Average cost per unit"
    )
    
    current_price = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Current price per unit"
    )
    
    # Calculated values
    cost_basis = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Total cost basis"
    )
    
    current_value = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Current market value"
    )
    
    unrealized_gain_loss = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Unrealized gain/loss"
    )
    
    unrealized_gain_loss_percentage = Column(
        Numeric(10, 4),
        nullable=False,
        doc="Unrealized gain/loss percentage"
    )
    
    # Dividend tracking
    total_dividends = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Total dividends received"
    )
    
    dividend_yield = Column(
        Numeric(10, 4),
        nullable=True,
        doc="Current dividend yield"
    )
    
    # Metadata
    first_purchase_date = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="First purchase date"
    )
    
    last_updated = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        doc="Last price update"
    )
    
    # Additional data
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional holding metadata"
    )
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")
    
    # Constraints
    __table_args__ = (
        UniqueConstraint('portfolio_id', 'symbol', name='uq_portfolio_holding_symbol'),
        Index('idx_holding_portfolio_symbol', 'portfolio_id', 'symbol'),
        Index('idx_holding_asset_type', 'asset_type'),
        CheckConstraint('quantity >= 0', name='ck_holding_quantity_positive'),
        CheckConstraint('average_cost >= 0', name='ck_holding_average_cost_positive'),
        CheckConstraint('current_price >= 0', name='ck_holding_current_price_positive'),
    )
    
    @validates('symbol')
    def validate_symbol(self, key, symbol):
        """Validate asset symbol.
        
        Args:
            key: Field name
            symbol: Asset symbol
            
        Returns:
            Validated symbol
            
        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not symbol.strip():
            raise ValueError("Asset symbol is required")
        
        symbol = symbol.strip().upper()
        if len(symbol) < 1 or len(symbol) > 20:
            raise ValueError("Asset symbol must be between 1 and 20 characters")
        
        return symbol
    
    def update_values(self, current_price: Decimal) -> None:
        """Update holding values based on current price.
        
        Args:
            current_price: Current market price
        """
        self.current_price = current_price
        self.current_value = self.quantity * current_price
        self.cost_basis = self.quantity * self.average_cost
        self.unrealized_gain_loss = self.current_value - self.cost_basis
        
        if self.cost_basis > 0:
            self.unrealized_gain_loss_percentage = (
                (self.unrealized_gain_loss / self.cost_basis) * 100
            )
        else:
            self.unrealized_gain_loss_percentage = Decimal('0')
        
        self.last_updated = datetime.utcnow()
    
    def add_shares(self, quantity: Decimal, price: Decimal) -> None:
        """Add shares to holding (buy transaction).
        
        Args:
            quantity: Number of shares to add
            price: Price per share
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        # Calculate new average cost
        total_cost = (self.quantity * self.average_cost) + (quantity * price)
        total_quantity = self.quantity + quantity
        
        self.average_cost = total_cost / total_quantity
        self.quantity = total_quantity
        
        # Update values
        self.update_values(self.current_price)
    
    def remove_shares(self, quantity: Decimal) -> Decimal:
        """Remove shares from holding (sell transaction).
        
        Args:
            quantity: Number of shares to remove
            
        Returns:
            Realized gain/loss from the sale
            
        Raises:
            ValueError: If quantity is invalid
        """
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        if quantity > self.quantity:
            raise ValueError("Cannot sell more shares than held")
        
        # Calculate realized gain/loss
        sale_cost_basis = quantity * self.average_cost
        sale_proceeds = quantity * self.current_price
        realized_gain_loss = sale_proceeds - sale_cost_basis
        
        # Update quantity
        self.quantity -= quantity
        
        # Update values
        self.update_values(self.current_price)
        
        return realized_gain_loss
    
    @property
    def weight_in_portfolio(self) -> Decimal:
        """Calculate weight of holding in portfolio.
        
        Returns:
            Weight as percentage
        """
        if not self.portfolio or self.portfolio.current_value == 0:
            return Decimal('0')
        
        return (self.current_value / self.portfolio.current_value) * 100
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return f"<Holding(id={self.id}, symbol={self.symbol}, quantity={self.quantity})>"


class Transaction(AuditModel):
    """Transaction model for tracking all portfolio transactions."""
    
    __tablename__ = "transactions"
    
    # Foreign key to portfolio
    portfolio_id = Column(
        UUID(as_uuid=True),
        ForeignKey("portfolios.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to portfolio"
    )
    
    # Transaction information
    transaction_type = Column(
        SQLEnum(TransactionType),
        nullable=False,
        doc="Type of transaction"
    )
    
    status = Column(
        SQLEnum(TransactionStatus),
        default=TransactionStatus.PENDING,
        nullable=False,
        doc="Transaction status"
    )
    
    # Asset information (nullable for cash transactions)
    symbol = Column(
        String(20),
        nullable=True,
        doc="Asset symbol/ticker"
    )
    
    asset_type = Column(
        SQLEnum(AssetType),
        nullable=True,
        doc="Type of asset"
    )
    
    # Transaction details
    quantity = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Quantity of asset"
    )
    
    price = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Price per unit"
    )
    
    amount = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Total transaction amount"
    )
    
    fees = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Transaction fees"
    )
    
    taxes = Column(
        Numeric(20, 8),
        default=0,
        nullable=False,
        doc="Transaction taxes"
    )
    
    # Timing
    transaction_date = Column(
        DateTime(timezone=True),
        nullable=False,
        doc="Transaction date"
    )
    
    settlement_date = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Settlement date"
    )
    
    # External references
    external_id = Column(
        String(100),
        nullable=True,
        doc="External transaction ID"
    )
    
    broker_id = Column(
        String(100),
        nullable=True,
        doc="Broker transaction ID"
    )
    
    # Additional information
    notes = Column(
        Text,
        nullable=True,
        doc="Transaction notes"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional transaction data"
    )
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="transactions")
    
    # Constraints
    __table_args__ = (
        Index('idx_transaction_portfolio_date', 'portfolio_id', 'transaction_date'),
        Index('idx_transaction_symbol_date', 'symbol', 'transaction_date'),
        Index('idx_transaction_type_status', 'transaction_type', 'status'),
        Index('idx_transaction_external_id', 'external_id'),
        CheckConstraint('amount != 0', name='ck_transaction_amount_nonzero'),
        CheckConstraint('fees >= 0', name='ck_transaction_fees_positive'),
        CheckConstraint('taxes >= 0', name='ck_transaction_taxes_positive'),
    )
    
    @validates('symbol')
    def validate_symbol(self, key, symbol):
        """Validate asset symbol.
        
        Args:
            key: Field name
            symbol: Asset symbol
            
        Returns:
            Validated symbol
        """
        if symbol:
            symbol = symbol.strip().upper()
            if len(symbol) < 1 or len(symbol) > 20:
                raise ValueError("Asset symbol must be between 1 and 20 characters")
        
        return symbol
    
    @property
    def net_amount(self) -> Decimal:
        """Calculate net transaction amount including fees and taxes.
        
        Returns:
            Net amount
        """
        if self.transaction_type in [TransactionType.BUY, TransactionType.WITHDRAWAL]:
            return self.amount + self.fees + self.taxes
        else:
            return self.amount - self.fees - self.taxes
    
    @property
    def is_buy_transaction(self) -> bool:
        """Check if transaction is a buy transaction.
        
        Returns:
            True if buy transaction
        """
        return self.transaction_type in [
            TransactionType.BUY,
            TransactionType.DEPOSIT,
            TransactionType.TRANSFER_IN,
            TransactionType.DIVIDEND,
            TransactionType.INTEREST
        ]
    
    @property
    def is_sell_transaction(self) -> bool:
        """Check if transaction is a sell transaction.
        
        Returns:
            True if sell transaction
        """
        return self.transaction_type in [
            TransactionType.SELL,
            TransactionType.WITHDRAWAL,
            TransactionType.TRANSFER_OUT,
            TransactionType.FEE,
            TransactionType.TAX
        ]
    
    def execute(self) -> None:
        """Mark transaction as executed."""
        self.status = TransactionStatus.EXECUTED
        if not self.settlement_date:
            self.settlement_date = self.transaction_date
    
    def cancel(self, reason: str = None) -> None:
        """Cancel the transaction.
        
        Args:
            reason: Cancellation reason
        """
        self.status = TransactionStatus.CANCELLED
        if reason and self.metadata:
            self.metadata['cancellation_reason'] = reason
        elif reason:
            self.metadata = {'cancellation_reason': reason}
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<Transaction(id={self.id}, type={self.transaction_type.value}, "
            f"symbol={self.symbol}, amount={self.amount})>"
        )