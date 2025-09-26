"""
Prisma Client for Python Backend
Provides database access through Prisma ORM
"""

import asyncio
import os
from typing import Optional
from prisma import Prisma
from prisma.models import User, Portfolio, Trade, Position
import logging

logger = logging.getLogger(__name__)

class PrismaClient:
    """Singleton Prisma client for database operations"""
    
    _instance: Optional['PrismaClient'] = None
    _prisma: Optional[Prisma] = None
    
    def __new__(cls) -> 'PrismaClient':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    async def connect(self) -> None:
        """Initialize and connect to the database"""
        if self._prisma is None:
            self._prisma = Prisma()
            await self._prisma.connect()
            logger.info("Connected to database via Prisma")
    
    async def disconnect(self) -> None:
        """Disconnect from the database"""
        if self._prisma:
            await self._prisma.disconnect()
            logger.info("Disconnected from database")
    
    @property
    def db(self) -> Prisma:
        """Get the Prisma client instance"""
        if self._prisma is None:
            raise RuntimeError("Prisma client not connected. Call connect() first.")
        return self._prisma
    
    # User operations
    async def create_user(self, email: str, username: str, hashed_password: str, **kwargs) -> User:
        """Create a new user"""
        return await self.db.user.create(
            data={
                'email': email,
                'username': username,
                'hashedPassword': hashed_password,
                **kwargs
            }
        )
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return await self.db.user.find_unique(where={'email': email})
    
    async def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return await self.db.user.find_unique(where={'username': username})
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return await self.db.user.find_unique(where={'id': user_id})
    
    # Portfolio operations
    async def create_portfolio(self, user_id: str, name: str, **kwargs) -> Portfolio:
        """Create a new portfolio"""
        return await self.db.portfolio.create(
            data={
                'userId': user_id,
                'name': name,
                'totalValue': 0,
                'cashBalance': kwargs.get('cashBalance', 100000),  # Default paper trading balance
                'isPaperTrade': kwargs.get('isPaperTrade', True),
                **kwargs
            }
        )
    
    async def get_user_portfolios(self, user_id: str) -> list[Portfolio]:
        """Get all portfolios for a user"""
        return await self.db.portfolio.find_many(
            where={'userId': user_id},
            include={'positions': True, 'trades': True}
        )
    
    # Trade operations
    async def create_trade(self, user_id: str, portfolio_id: str, **trade_data) -> Trade:
        """Create a new trade"""
        return await self.db.trade.create(
            data={
                'userId': user_id,
                'portfolioId': portfolio_id,
                **trade_data
            }
        )
    
    async def get_user_trades(self, user_id: str, limit: int = 100) -> list[Trade]:
        """Get recent trades for a user"""
        return await self.db.trade.find_many(
            where={'userId': user_id},
            order_by={'createdAt': 'desc'},
            take=limit
        )
    
    # Position operations
    async def update_position(self, portfolio_id: str, symbol: str, **position_data) -> Position:
        """Update or create a position"""
        return await self.db.position.upsert(
            where={
                'portfolioId_symbol': {
                    'portfolioId': portfolio_id,
                    'symbol': symbol
                }
            },
            data={
                'portfolioId': portfolio_id,
                'symbol': symbol,
                **position_data
            },
            create={
                'portfolioId': portfolio_id,
                'symbol': symbol,
                **position_data
            }
        )

# Global instance
prisma_client = PrismaClient()

async def get_prisma() -> PrismaClient:
    """Dependency injection for FastAPI"""
    if prisma_client._prisma is None:
        await prisma_client.connect()
    return prisma_client