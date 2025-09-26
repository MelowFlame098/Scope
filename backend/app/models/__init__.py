"""Database models for FinScope application.

This package contains all SQLAlchemy models for the application,
organized by domain and functionality.
"""

from app.models.base import BaseModel, TimestampMixin, SoftDeleteMixin
from app.models.user import User, UserProfile, UserSettings
from app.models.portfolio import Portfolio, Position, Transaction
from app.models.market import MarketData, Symbol, Exchange
from app.models.trading import Order, Trade, TradingStrategy
from app.models.analytics import Analysis, Prediction, Alert
from app.models.defi import DeFiProtocol, LiquidityPool, YieldFarm

__all__ = [
    # Base models
    "BaseModel",
    "TimestampMixin",
    "SoftDeleteMixin",
    
    # User models
    "User",
    "UserProfile",
    "UserSettings",
    
    # Portfolio models
    "Portfolio",
    "Position",
    "Transaction",
    
    # Market models
    "MarketData",
    "Symbol",
    "Exchange",
    
    # Trading models
    "Order",
    "Trade",
    "TradingStrategy",
    
    # Analytics models
    "Analysis",
    "Prediction",
    "Alert",
    
    # DeFi models
    "DeFiProtocol",
    "LiquidityPool",
    "YieldFarm",
]