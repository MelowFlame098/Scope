from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, desc, func
from datetime import datetime

from .base import BaseRepository
from ..models import Portfolio, PortfolioHolding, Asset
from ..schemas import PortfolioCreate, PortfolioUpdate, PortfolioHoldingCreate, PortfolioHoldingUpdate

class PortfolioRepository(BaseRepository[Portfolio, PortfolioCreate, PortfolioUpdate]):
    """
    Repository for Portfolio model with portfolio management operations
    """
    
    def __init__(self):
        super().__init__(Portfolio)
    
    def get_user_portfolios(self, db: Session, *, user_id: str) -> List[Portfolio]:
        """
        Get all portfolios for a user
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            List of user's portfolios
        """
        return (
            db.query(Portfolio)
            .filter(Portfolio.user_id == user_id)
            .options(joinedload(Portfolio.holdings))
            .all()
        )
    
    def get_portfolio_with_holdings(self, db: Session, *, portfolio_id: str) -> Optional[Portfolio]:
        """
        Get portfolio with all holdings loaded
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            
        Returns:
            Portfolio with holdings or None
        """
        return (
            db.query(Portfolio)
            .filter(Portfolio.id == portfolio_id)
            .options(joinedload(Portfolio.holdings))
            .first()
        )
    
    def get_public_portfolios(self, db: Session, *, skip: int = 0, limit: int = 100) -> List[Portfolio]:
        """
        Get public portfolios
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of public portfolios
        """
        return (
            db.query(Portfolio)
            .filter(Portfolio.is_public == True)
            .options(joinedload(Portfolio.user))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def calculate_portfolio_value(self, db: Session, *, portfolio_id: str) -> Dict[str, Any]:
        """
        Calculate total portfolio value and performance metrics
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            
        Returns:
            Dictionary with portfolio metrics
        """
        portfolio = self.get_portfolio_with_holdings(db, portfolio_id=portfolio_id)
        if not portfolio:
            return {}
        
        total_value = 0.0
        total_cost = 0.0
        holdings_data = []
        
        for holding in portfolio.holdings:
            # Get current asset price
            asset = db.query(Asset).filter(Asset.symbol == holding.symbol).first()
            current_price = asset.current_price if asset and asset.current_price else 0.0
            
            holding_value = holding.quantity * current_price
            holding_cost = holding.quantity * holding.average_cost
            
            total_value += holding_value
            total_cost += holding_cost
            
            holdings_data.append({
                'symbol': holding.symbol,
                'quantity': holding.quantity,
                'average_cost': holding.average_cost,
                'current_price': current_price,
                'value': holding_value,
                'cost': holding_cost,
                'pnl': holding_value - holding_cost,
                'pnl_percentage': ((holding_value - holding_cost) / holding_cost * 100) if holding_cost > 0 else 0
            })
        
        total_pnl = total_value - total_cost
        total_pnl_percentage = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            'portfolio_id': portfolio_id,
            'total_value': total_value,
            'total_cost': total_cost,
            'total_pnl': total_pnl,
            'total_pnl_percentage': total_pnl_percentage,
            'holdings': holdings_data,
            'currency': portfolio.currency
        }

class PortfolioHoldingRepository(BaseRepository[PortfolioHolding, PortfolioHoldingCreate, PortfolioHoldingUpdate]):
    """
    Repository for PortfolioHolding model
    """
    
    def __init__(self):
        super().__init__(PortfolioHolding)
    
    def get_portfolio_holdings(self, db: Session, *, portfolio_id: str) -> List[PortfolioHolding]:
        """
        Get all holdings for a portfolio
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            
        Returns:
            List of portfolio holdings
        """
        return (
            db.query(PortfolioHolding)
            .filter(PortfolioHolding.portfolio_id == portfolio_id)
            .all()
        )
    
    def get_holding_by_symbol(self, db: Session, *, portfolio_id: str, symbol: str) -> Optional[PortfolioHolding]:
        """
        Get a specific holding by portfolio and symbol
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            symbol: Asset symbol
            
        Returns:
            Portfolio holding or None
        """
        return (
            db.query(PortfolioHolding)
            .filter(
                and_(
                    PortfolioHolding.portfolio_id == portfolio_id,
                    PortfolioHolding.symbol == symbol.upper()
                )
            )
            .first()
        )
    
    def add_or_update_holding(
        self, 
        db: Session, 
        *, 
        portfolio_id: str, 
        symbol: str, 
        quantity: float, 
        price: float,
        transaction_type: str = "buy"
    ) -> PortfolioHolding:
        """
        Add new holding or update existing one
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            symbol: Asset symbol
            quantity: Quantity to add/remove
            price: Transaction price
            transaction_type: 'buy' or 'sell'
            
        Returns:
            Updated or created portfolio holding
        """
        existing_holding = self.get_holding_by_symbol(db, portfolio_id=portfolio_id, symbol=symbol)
        
        if existing_holding:
            if transaction_type == "buy":
                # Calculate new average cost
                total_cost = (existing_holding.quantity * existing_holding.average_cost) + (quantity * price)
                new_quantity = existing_holding.quantity + quantity
                new_average_cost = total_cost / new_quantity if new_quantity > 0 else 0
                
                existing_holding.quantity = new_quantity
                existing_holding.average_cost = new_average_cost
            elif transaction_type == "sell":
                existing_holding.quantity -= quantity
                
                # Remove holding if quantity becomes zero or negative
                if existing_holding.quantity <= 0:
                    db.delete(existing_holding)
                    db.commit()
                    return None
            
            existing_holding.last_transaction_date = datetime.utcnow()
            existing_holding.updated_at = datetime.utcnow()
            
            db.commit()
            db.refresh(existing_holding)
            return existing_holding
        else:
            # Create new holding (only for buy transactions)
            if transaction_type == "buy":
                new_holding = PortfolioHolding(
                    portfolio_id=portfolio_id,
                    symbol=symbol.upper(),
                    quantity=quantity,
                    average_cost=price,
                    first_purchase_date=datetime.utcnow(),
                    last_transaction_date=datetime.utcnow()
                )
                
                db.add(new_holding)
                db.commit()
                db.refresh(new_holding)
                return new_holding
            else:
                raise ValueError("Cannot sell an asset that is not in the portfolio")
    
    def get_user_holdings_by_symbol(self, db: Session, *, user_id: str, symbol: str) -> List[PortfolioHolding]:
        """
        Get all holdings of a specific symbol across user's portfolios
        
        Args:
            db: Database session
            user_id: User ID
            symbol: Asset symbol
            
        Returns:
            List of holdings for the symbol
        """
        return (
            db.query(PortfolioHolding)
            .join(Portfolio)
            .filter(
                and_(
                    Portfolio.user_id == user_id,
                    PortfolioHolding.symbol == symbol.upper()
                )
            )
            .all()
        )
    
    def get_top_holdings_by_value(self, db: Session, *, portfolio_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get top holdings by current value
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            limit: Maximum number of holdings to return
            
        Returns:
            List of holdings with current values
        """
        holdings = self.get_portfolio_holdings(db, portfolio_id=portfolio_id)
        holdings_with_value = []
        
        for holding in holdings:
            asset = db.query(Asset).filter(Asset.symbol == holding.symbol).first()
            current_price = asset.current_price if asset and asset.current_price else 0.0
            value = holding.quantity * current_price
            
            holdings_with_value.append({
                'holding': holding,
                'current_price': current_price,
                'value': value
            })
        
        # Sort by value and return top holdings
        holdings_with_value.sort(key=lambda x: x['value'], reverse=True)
        return holdings_with_value[:limit]

# Create repository instances
portfolio_repository = PortfolioRepository()
portfolio_holding_repository = PortfolioHoldingRepository()