"""Market data related database models.

This module contains models for:
- Financial instruments and securities
- Price data and historical quotes
- Market indices and benchmarks
- Economic indicators and events
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
from app.models.portfolio import AssetType


class MarketStatus(str, Enum):
    """Market status."""
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"
    AFTER_HOURS = "after_hours"
    HOLIDAY = "holiday"
    MAINTENANCE = "maintenance"


class Exchange(str, Enum):
    """Stock exchanges."""
    NYSE = "NYSE"
    NASDAQ = "NASDAQ"
    AMEX = "AMEX"
    LSE = "LSE"      # London Stock Exchange
    TSE = "TSE"      # Tokyo Stock Exchange
    HKEX = "HKEX"    # Hong Kong Exchange
    SSE = "SSE"      # Shanghai Stock Exchange
    SZSE = "SZSE"    # Shenzhen Stock Exchange
    TSX = "TSX"      # Toronto Stock Exchange
    ASX = "ASX"      # Australian Securities Exchange
    BSE = "BSE"      # Bombay Stock Exchange
    NSE = "NSE"      # National Stock Exchange of India
    CRYPTO = "CRYPTO"  # Cryptocurrency exchanges
    FOREX = "FOREX"   # Foreign exchange
    COMMODITY = "COMMODITY"  # Commodity exchanges


class Sector(str, Enum):
    """Market sectors."""
    TECHNOLOGY = "technology"
    HEALTHCARE = "healthcare"
    FINANCIALS = "financials"
    CONSUMER_DISCRETIONARY = "consumer_discretionary"
    CONSUMER_STAPLES = "consumer_staples"
    INDUSTRIALS = "industrials"
    ENERGY = "energy"
    UTILITIES = "utilities"
    MATERIALS = "materials"
    REAL_ESTATE = "real_estate"
    TELECOMMUNICATIONS = "telecommunications"
    CRYPTOCURRENCY = "cryptocurrency"
    COMMODITIES = "commodities"
    FOREX = "forex"


class PriceType(str, Enum):
    """Price data types."""
    REAL_TIME = "real_time"
    DELAYED = "delayed"
    END_OF_DAY = "end_of_day"
    INTRADAY = "intraday"
    HISTORICAL = "historical"


class Timeframe(str, Enum):
    """Chart timeframes."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"
    YEAR_1 = "1y"


class Security(BaseModel):
    """Financial security/instrument model."""
    
    __tablename__ = "securities"
    
    # Basic information
    symbol = Column(
        String(20),
        unique=True,
        nullable=False,
        index=True,
        doc="Security symbol/ticker"
    )
    
    name = Column(
        String(200),
        nullable=False,
        doc="Full security name"
    )
    
    asset_type = Column(
        SQLEnum(AssetType),
        nullable=False,
        doc="Type of asset"
    )
    
    # Exchange and market information
    exchange = Column(
        SQLEnum(Exchange),
        nullable=True,
        doc="Primary exchange"
    )
    
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        doc="Trading currency"
    )
    
    country = Column(
        String(2),
        nullable=True,
        doc="Country code (ISO 2-letter)"
    )
    
    # Classification
    sector = Column(
        SQLEnum(Sector),
        nullable=True,
        doc="Market sector"
    )
    
    industry = Column(
        String(100),
        nullable=True,
        doc="Industry classification"
    )
    
    # Security details
    isin = Column(
        String(12),
        nullable=True,
        unique=True,
        doc="International Securities Identification Number"
    )
    
    cusip = Column(
        String(9),
        nullable=True,
        doc="Committee on Uniform Securities Identification Procedures"
    )
    
    # Market data
    market_cap = Column(
        Numeric(20, 2),
        nullable=True,
        doc="Market capitalization"
    )
    
    shares_outstanding = Column(
        Numeric(20, 0),
        nullable=True,
        doc="Shares outstanding"
    )
    
    # Status and metadata
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Security is actively traded"
    )
    
    is_tradable = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Security can be traded"
    )
    
    listing_date = Column(
        Date,
        nullable=True,
        doc="Initial listing date"
    )
    
    delisting_date = Column(
        Date,
        nullable=True,
        doc="Delisting date if applicable"
    )
    
    # Additional information
    description = Column(
        Text,
        nullable=True,
        doc="Security description"
    )
    
    website = Column(
        String(500),
        nullable=True,
        doc="Company website"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional security metadata"
    )
    
    # Data source tracking
    data_source = Column(
        String(50),
        nullable=True,
        doc="Primary data source"
    )
    
    last_updated = Column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
        doc="Last metadata update"
    )
    
    # Relationships
    price_data = relationship(
        "PriceData",
        back_populates="security",
        cascade="all, delete-orphan"
    )
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_security_symbol_exchange', 'symbol', 'exchange'),
        Index('idx_security_asset_type', 'asset_type'),
        Index('idx_security_sector', 'sector'),
        Index('idx_security_active', 'is_active'),
        Index('idx_security_tradable', 'is_tradable'),
    )
    
    @validates('symbol')
    def validate_symbol(self, key, symbol):
        """Validate security symbol.
        
        Args:
            key: Field name
            symbol: Security symbol
            
        Returns:
            Validated symbol
            
        Raises:
            ValueError: If symbol is invalid
        """
        if not symbol or not symbol.strip():
            raise ValueError("Security symbol is required")
        
        symbol = symbol.strip().upper()
        if len(symbol) < 1 or len(symbol) > 20:
            raise ValueError("Security symbol must be between 1 and 20 characters")
        
        return symbol
    
    @validates('name')
    def validate_name(self, key, name):
        """Validate security name.
        
        Args:
            key: Field name
            name: Security name
            
        Returns:
            Validated name
            
        Raises:
            ValueError: If name is invalid
        """
        if not name or not name.strip():
            raise ValueError("Security name is required")
        
        name = name.strip()
        if len(name) < 1 or len(name) > 200:
            raise ValueError("Security name must be between 1 and 200 characters")
        
        return name
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return f"<Security(symbol={self.symbol}, name={self.name}, type={self.asset_type.value})>"


class PriceData(TimestampMixin):
    """Price data model for securities."""
    
    __tablename__ = "price_data"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier"
    )
    
    # Foreign key to security
    security_id = Column(
        UUID(as_uuid=True),
        ForeignKey("securities.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to security"
    )
    
    # Price information
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        doc="Price timestamp"
    )
    
    open_price = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Opening price"
    )
    
    high_price = Column(
        Numeric(20, 8),
        nullable=True,
        doc="High price"
    )
    
    low_price = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Low price"
    )
    
    close_price = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Closing price"
    )
    
    adjusted_close = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Adjusted closing price"
    )
    
    # Volume information
    volume = Column(
        Numeric(20, 0),
        nullable=True,
        doc="Trading volume"
    )
    
    volume_weighted_price = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Volume weighted average price"
    )
    
    # Market data
    bid_price = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Bid price"
    )
    
    ask_price = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Ask price"
    )
    
    bid_size = Column(
        Numeric(20, 0),
        nullable=True,
        doc="Bid size"
    )
    
    ask_size = Column(
        Numeric(20, 0),
        nullable=True,
        doc="Ask size"
    )
    
    # Metadata
    timeframe = Column(
        SQLEnum(Timeframe),
        nullable=False,
        doc="Data timeframe"
    )
    
    price_type = Column(
        SQLEnum(PriceType),
        default=PriceType.HISTORICAL,
        nullable=False,
        doc="Type of price data"
    )
    
    data_source = Column(
        String(50),
        nullable=True,
        doc="Data source provider"
    )
    
    # Additional data
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional price metadata"
    )
    
    # Relationships
    security = relationship("Security", back_populates="price_data")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint(
            'security_id', 'timestamp', 'timeframe',
            name='uq_price_data_security_timestamp_timeframe'
        ),
        Index('idx_price_data_security_timestamp', 'security_id', 'timestamp'),
        Index('idx_price_data_timestamp', 'timestamp'),
        Index('idx_price_data_timeframe', 'timeframe'),
        CheckConstraint('close_price > 0', name='ck_price_data_close_price_positive'),
        CheckConstraint('volume >= 0', name='ck_price_data_volume_positive'),
    )
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Calculate bid-ask spread.
        
        Returns:
            Bid-ask spread or None if data unavailable
        """
        if self.bid_price and self.ask_price:
            return self.ask_price - self.bid_price
        return None
    
    @property
    def spread_percentage(self) -> Optional[Decimal]:
        """Calculate bid-ask spread as percentage.
        
        Returns:
            Spread percentage or None if data unavailable
        """
        spread = self.spread
        if spread and self.bid_price and self.bid_price > 0:
            return (spread / self.bid_price) * 100
        return None
    
    @property
    def price_change(self) -> Optional[Decimal]:
        """Calculate price change from open to close.
        
        Returns:
            Price change or None if open price unavailable
        """
        if self.open_price:
            return self.close_price - self.open_price
        return None
    
    @property
    def price_change_percentage(self) -> Optional[Decimal]:
        """Calculate price change percentage.
        
        Returns:
            Price change percentage or None if open price unavailable
        """
        change = self.price_change
        if change and self.open_price and self.open_price > 0:
            return (change / self.open_price) * 100
        return None
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<PriceData(security_id={self.security_id}, "
            f"timestamp={self.timestamp}, close={self.close_price})>"
        )


class MarketIndex(BaseModel):
    """Market index model."""
    
    __tablename__ = "market_indices"
    
    # Basic information
    symbol = Column(
        String(20),
        unique=True,
        nullable=False,
        index=True,
        doc="Index symbol"
    )
    
    name = Column(
        String(200),
        nullable=False,
        doc="Index name"
    )
    
    description = Column(
        Text,
        nullable=True,
        doc="Index description"
    )
    
    # Index details
    base_value = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Base index value"
    )
    
    base_date = Column(
        Date,
        nullable=True,
        doc="Base date for index calculation"
    )
    
    currency = Column(
        String(3),
        default="USD",
        nullable=False,
        doc="Index currency"
    )
    
    # Classification
    category = Column(
        String(50),
        nullable=True,
        doc="Index category"
    )
    
    region = Column(
        String(50),
        nullable=True,
        doc="Geographic region"
    )
    
    # Constituents
    constituent_count = Column(
        Integer,
        nullable=True,
        doc="Number of constituents"
    )
    
    constituents = Column(
        JSONB,
        nullable=True,
        doc="Index constituents and weights"
    )
    
    # Status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Index is active"
    )
    
    # Metadata
    provider = Column(
        String(100),
        nullable=True,
        doc="Index provider"
    )
    
    methodology = Column(
        Text,
        nullable=True,
        doc="Index calculation methodology"
    )
    
    website = Column(
        String(500),
        nullable=True,
        doc="Index provider website"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional index metadata"
    )
    
    # Relationships
    price_data = relationship(
        "IndexPriceData",
        back_populates="index",
        cascade="all, delete-orphan"
    )
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_market_index_category', 'category'),
        Index('idx_market_index_region', 'region'),
        Index('idx_market_index_active', 'is_active'),
    )
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return f"<MarketIndex(symbol={self.symbol}, name={self.name})>"


class IndexPriceData(TimestampMixin):
    """Price data for market indices."""
    
    __tablename__ = "index_price_data"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier"
    )
    
    # Foreign key to index
    index_id = Column(
        UUID(as_uuid=True),
        ForeignKey("market_indices.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to market index"
    )
    
    # Price information
    timestamp = Column(
        DateTime(timezone=True),
        nullable=False,
        doc="Price timestamp"
    )
    
    open_value = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Opening value"
    )
    
    high_value = Column(
        Numeric(20, 8),
        nullable=True,
        doc="High value"
    )
    
    low_value = Column(
        Numeric(20, 8),
        nullable=True,
        doc="Low value"
    )
    
    close_value = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Closing value"
    )
    
    # Metadata
    timeframe = Column(
        SQLEnum(Timeframe),
        nullable=False,
        doc="Data timeframe"
    )
    
    data_source = Column(
        String(50),
        nullable=True,
        doc="Data source provider"
    )
    
    # Relationships
    index = relationship("MarketIndex", back_populates="price_data")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint(
            'index_id', 'timestamp', 'timeframe',
            name='uq_index_price_data_index_timestamp_timeframe'
        ),
        Index('idx_index_price_data_index_timestamp', 'index_id', 'timestamp'),
        Index('idx_index_price_data_timestamp', 'timestamp'),
        CheckConstraint('close_value > 0', name='ck_index_price_data_close_value_positive'),
    )
    
    @property
    def value_change(self) -> Optional[Decimal]:
        """Calculate value change from open to close.
        
        Returns:
            Value change or None if open value unavailable
        """
        if self.open_value:
            return self.close_value - self.open_value
        return None
    
    @property
    def value_change_percentage(self) -> Optional[Decimal]:
        """Calculate value change percentage.
        
        Returns:
            Value change percentage or None if open value unavailable
        """
        change = self.value_change
        if change and self.open_value and self.open_value > 0:
            return (change / self.open_value) * 100
        return None
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<IndexPriceData(index_id={self.index_id}, "
            f"timestamp={self.timestamp}, close={self.close_value})>"
        )


class EconomicIndicator(BaseModel):
    """Economic indicator model."""
    
    __tablename__ = "economic_indicators"
    
    # Basic information
    code = Column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
        doc="Indicator code"
    )
    
    name = Column(
        String(200),
        nullable=False,
        doc="Indicator name"
    )
    
    description = Column(
        Text,
        nullable=True,
        doc="Indicator description"
    )
    
    # Classification
    category = Column(
        String(50),
        nullable=True,
        doc="Indicator category"
    )
    
    frequency = Column(
        String(20),
        nullable=True,
        doc="Release frequency"
    )
    
    unit = Column(
        String(50),
        nullable=True,
        doc="Unit of measurement"
    )
    
    # Source information
    source = Column(
        String(100),
        nullable=True,
        doc="Data source"
    )
    
    source_url = Column(
        String(500),
        nullable=True,
        doc="Source URL"
    )
    
    # Status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Indicator is active"
    )
    
    # Metadata
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional indicator metadata"
    )
    
    # Relationships
    data_points = relationship(
        "EconomicData",
        back_populates="indicator",
        cascade="all, delete-orphan"
    )
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_economic_indicator_category', 'category'),
        Index('idx_economic_indicator_active', 'is_active'),
    )
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return f"<EconomicIndicator(code={self.code}, name={self.name})>"


class EconomicData(TimestampMixin):
    """Economic data points."""
    
    __tablename__ = "economic_data"
    
    # Primary key
    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier"
    )
    
    # Foreign key to indicator
    indicator_id = Column(
        UUID(as_uuid=True),
        ForeignKey("economic_indicators.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to economic indicator"
    )
    
    # Data information
    date = Column(
        Date,
        nullable=False,
        doc="Data date"
    )
    
    value = Column(
        Numeric(20, 8),
        nullable=False,
        doc="Indicator value"
    )
    
    # Revisions
    is_preliminary = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Preliminary data point"
    )
    
    is_revised = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Revised data point"
    )
    
    revision_count = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Number of revisions"
    )
    
    # Metadata
    data_source = Column(
        String(50),
        nullable=True,
        doc="Data source"
    )
    
    metadata = Column(
        JSONB,
        nullable=True,
        doc="Additional data metadata"
    )
    
    # Relationships
    indicator = relationship("EconomicIndicator", back_populates="data_points")
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint(
            'indicator_id', 'date',
            name='uq_economic_data_indicator_date'
        ),
        Index('idx_economic_data_indicator_date', 'indicator_id', 'date'),
        Index('idx_economic_data_date', 'date'),
    )
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return (
            f"<EconomicData(indicator_id={self.indicator_id}, "
            f"date={self.date}, value={self.value})>"
        )