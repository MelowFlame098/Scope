from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from datetime import datetime

from ..repositories.portfolio import portfolio_repository, portfolio_holding_repository
from ..repositories.asset import asset_repository
from ..models import Portfolio, PortfolioHolding
from ..schemas import PortfolioCreate, PortfolioUpdate, PortfolioHoldingCreate

class PortfolioService:
    """
    Service class for portfolio-related business logic
    """
    
    def __init__(self):
        self.portfolio_repo = portfolio_repository
        self.holding_repo = portfolio_holding_repository
        self.asset_repo = asset_repository
    
    def create_portfolio(self, db: Session, *, user_id: str, portfolio_in: PortfolioCreate) -> Portfolio:
        """
        Create a new portfolio for a user
        
        Args:
            db: Database session
            user_id: User ID
            portfolio_in: Portfolio creation data
            
        Returns:
            Created portfolio instance
        """
        portfolio_data = portfolio_in.dict()
        portfolio_data["user_id"] = user_id
        
        portfolio = self.portfolio_repo.create(db, obj_in=portfolio_data)
        return portfolio
    
    def get_user_portfolios(self, db: Session, *, user_id: str) -> List[Portfolio]:
        """
        Get all portfolios for a user
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            List of user's portfolios
        """
        return self.portfolio_repo.get_user_portfolios(db, user_id=user_id)
    
    def get_portfolio_details(self, db: Session, *, portfolio_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed portfolio information including holdings and performance
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)
            
        Returns:
            Portfolio details dictionary or None
        """
        portfolio = self.portfolio_repo.get_portfolio_with_holdings(db, portfolio_id=portfolio_id)
        
        if not portfolio or portfolio.user_id != user_id:
            return None
        
        # Calculate portfolio metrics
        portfolio_metrics = self.portfolio_repo.calculate_portfolio_value(db, portfolio_id=portfolio_id)
        
        # Get portfolio performance over time (simplified)
        performance_data = self._calculate_portfolio_performance(db, portfolio_id)
        
        return {
            "portfolio": portfolio,
            "metrics": portfolio_metrics,
            "performance": performance_data,
            "holdings_count": len(portfolio.holdings),
            "last_updated": portfolio.updated_at
        }
    
    def add_holding(
        self, 
        db: Session, 
        *, 
        portfolio_id: str, 
        user_id: str,
        symbol: str, 
        quantity: float, 
        price: float
    ) -> Optional[PortfolioHolding]:
        """
        Add or update a holding in a portfolio
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)
            symbol: Asset symbol
            quantity: Quantity to add
            price: Purchase price
            
        Returns:
            Updated holding or None
        """
        # Verify portfolio ownership
        portfolio = self.portfolio_repo.get(db, portfolio_id)
        if not portfolio or portfolio.user_id != user_id:
            return None
        
        # Ensure asset exists
        asset = self.asset_repo.get_by_symbol(db, symbol=symbol)
        if not asset:
            raise ValueError(f"Asset {symbol} not found")
        
        # Add or update holding
        holding = self.holding_repo.add_or_update_holding(
            db,
            portfolio_id=portfolio_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            transaction_type="buy"
        )
        
        # Update portfolio timestamp
        portfolio.updated_at = datetime.utcnow()
        db.commit()
        
        return holding
    
    def sell_holding(
        self, 
        db: Session, 
        *, 
        portfolio_id: str, 
        user_id: str,
        symbol: str, 
        quantity: float, 
        price: float
    ) -> Optional[PortfolioHolding]:
        """
        Sell a holding from a portfolio
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)
            symbol: Asset symbol
            quantity: Quantity to sell
            price: Sale price
            
        Returns:
            Updated holding or None
        """
        # Verify portfolio ownership
        portfolio = self.portfolio_repo.get(db, portfolio_id)
        if not portfolio or portfolio.user_id != user_id:
            return None
        
        # Check if holding exists and has sufficient quantity
        existing_holding = self.holding_repo.get_holding_by_symbol(
            db, portfolio_id=portfolio_id, symbol=symbol
        )
        
        if not existing_holding or existing_holding.quantity < quantity:
            raise ValueError("Insufficient holdings to sell")
        
        # Sell holding
        holding = self.holding_repo.add_or_update_holding(
            db,
            portfolio_id=portfolio_id,
            symbol=symbol,
            quantity=quantity,
            price=price,
            transaction_type="sell"
        )
        
        # Update portfolio timestamp
        portfolio.updated_at = datetime.utcnow()
        db.commit()
        
        return holding
    
    def remove_holding(self, db: Session, *, portfolio_id: str, user_id: str, symbol: str) -> bool:
        """
        Remove a holding completely from a portfolio
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)
            symbol: Asset symbol
            
        Returns:
            True if removed successfully, False otherwise
        """
        # Verify portfolio ownership
        portfolio = self.portfolio_repo.get(db, portfolio_id)
        if not portfolio or portfolio.user_id != user_id:
            return False
        
        # Find and remove holding
        holding = self.holding_repo.get_holding_by_symbol(
            db, portfolio_id=portfolio_id, symbol=symbol
        )
        
        if holding:
            self.holding_repo.delete(db, id=holding.id)
            portfolio.updated_at = datetime.utcnow()
            db.commit()
            return True
        
        return False
    
    def get_portfolio_performance(self, db: Session, *, portfolio_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get portfolio performance metrics
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)
            
        Returns:
            Performance metrics dictionary or None
        """
        # Verify portfolio ownership
        portfolio = self.portfolio_repo.get(db, portfolio_id)
        if not portfolio or portfolio.user_id != user_id:
            return None
        
        return self._calculate_portfolio_performance(db, portfolio_id)
    
    def get_portfolio_allocation(self, db: Session, *, portfolio_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """
        Get portfolio asset allocation
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)
            
        Returns:
            Allocation data dictionary or None
        """
        # Verify portfolio ownership
        portfolio = self.portfolio_repo.get(db, portfolio_id)
        if not portfolio or portfolio.user_id != user_id:
            return None
        
        # Calculate portfolio value
        portfolio_metrics = self.portfolio_repo.calculate_portfolio_value(db, portfolio_id=portfolio_id)
        
        if not portfolio_metrics or portfolio_metrics['total_value'] == 0:
            return {
                "allocations": [],
                "by_category": {},
                "total_value": 0
            }
        
        # Calculate allocations
        allocations = []
        category_allocations = {}
        
        for holding_data in portfolio_metrics['holdings']:
            symbol = holding_data['symbol']
            value = holding_data['value']
            percentage = (value / portfolio_metrics['total_value']) * 100
            
            # Get asset category
            asset = self.asset_repo.get_by_symbol(db, symbol=symbol)
            category = asset.category if asset else "unknown"
            
            allocations.append({
                "symbol": symbol,
                "value": value,
                "percentage": percentage,
                "category": category
            })
            
            # Aggregate by category
            if category not in category_allocations:
                category_allocations[category] = {"value": 0, "percentage": 0}
            
            category_allocations[category]["value"] += value
            category_allocations[category]["percentage"] += percentage
        
        return {
            "allocations": sorted(allocations, key=lambda x: x['value'], reverse=True),
            "by_category": category_allocations,
            "total_value": portfolio_metrics['total_value']
        }
    
    def update_portfolio(self, db: Session, *, portfolio_id: str, user_id: str, portfolio_in: PortfolioUpdate) -> Optional[Portfolio]:
        """
        Update portfolio information
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)
            portfolio_in: Portfolio update data
            
        Returns:
            Updated portfolio or None
        """
        portfolio = self.portfolio_repo.get(db, portfolio_id)
        if not portfolio or portfolio.user_id != user_id:
            return None
        
        updated_portfolio = self.portfolio_repo.update(db, db_obj=portfolio, obj_in=portfolio_in)
        return updated_portfolio
    
    def delete_portfolio(self, db: Session, *, portfolio_id: str, user_id: str) -> bool:
        """
        Delete a portfolio
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            user_id: User ID (for authorization)
            
        Returns:
            True if deleted successfully, False otherwise
        """
        portfolio = self.portfolio_repo.get(db, portfolio_id)
        if not portfolio or portfolio.user_id != user_id:
            return False
        
        self.portfolio_repo.delete(db, id=portfolio_id)
        return True
    
    def _calculate_portfolio_performance(self, db: Session, portfolio_id: str) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics (simplified version)
        
        Args:
            db: Database session
            portfolio_id: Portfolio ID
            
        Returns:
            Performance metrics dictionary
        """
        # Get current portfolio value
        current_metrics = self.portfolio_repo.calculate_portfolio_value(db, portfolio_id=portfolio_id)
        
        # For now, return basic metrics
        # In a full implementation, you would track historical values
        return {
            "current_value": current_metrics.get('total_value', 0),
            "total_cost": current_metrics.get('total_cost', 0),
            "total_pnl": current_metrics.get('total_pnl', 0),
            "total_pnl_percentage": current_metrics.get('total_pnl_percentage', 0),
            "daily_change": 0,  # Would need historical data
            "weekly_change": 0,  # Would need historical data
            "monthly_change": 0,  # Would need historical data
        }
    
    def get_user_total_portfolio_value(self, db: Session, *, user_id: str) -> Dict[str, Any]:
        """
        Get total portfolio value across all user portfolios
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Total portfolio metrics
        """
        portfolios = self.get_user_portfolios(db, user_id=user_id)
        
        total_value = 0.0
        total_cost = 0.0
        portfolio_count = len(portfolios)
        
        for portfolio in portfolios:
            metrics = self.portfolio_repo.calculate_portfolio_value(db, portfolio_id=portfolio.id)
            total_value += metrics.get('total_value', 0)
            total_cost += metrics.get('total_cost', 0)
        
        total_pnl = total_value - total_cost
        total_pnl_percentage = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return {
            "total_value": total_value,
            "total_cost": total_cost,
            "total_pnl": total_pnl,
            "total_pnl_percentage": total_pnl_percentage,
            "portfolio_count": portfolio_count
        }

# Create service instance
portfolio_service = PortfolioService()