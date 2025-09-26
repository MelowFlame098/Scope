import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_HALF_UP

from database import get_db, SessionLocal
from db_models import User, Portfolio, PortfolioHolding, Asset
from schemas import (
    PortfolioResponse, PortfolioHoldingResponse, PortfolioCreate,
    PortfolioHoldingCreate, AssetResponse
)
from market_data import MarketDataService
from notification_service import NotificationService, NotificationType, NotificationChannel

logger = logging.getLogger(__name__)

class PortfolioService:
    def __init__(self):
        self.market_service = MarketDataService()
        self.notification_service = NotificationService()
        
    async def create_portfolio(
        self,
        user_id: str,
        portfolio_data: PortfolioCreate,
        db: Session
    ) -> PortfolioResponse:
        """Create a new portfolio for a user."""
        try:
            portfolio = Portfolio(
                user_id=user_id,
                name=portfolio_data.name,
                description=portfolio_data.description,
                is_public=portfolio_data.is_public,
                currency=portfolio_data.currency
            )
            
            db.add(portfolio)
            db.commit()
            db.refresh(portfolio)
            
            logger.info(f"Portfolio created: {portfolio.id} for user {user_id}")
            
            return await self._portfolio_to_response(portfolio, db)
            
        except Exception as e:
            logger.error(f"Error creating portfolio for user {user_id}: {e}")
            db.rollback()
            raise
    
    async def get_user_portfolios(
        self,
        user_id: str,
        db: Session
    ) -> List[PortfolioResponse]:
        """Get all portfolios for a user."""
        try:
            portfolios = db.query(Portfolio).filter(
                Portfolio.user_id == user_id
            ).all()
            
            portfolio_responses = []
            for portfolio in portfolios:
                response = await self._portfolio_to_response(portfolio, db)
                portfolio_responses.append(response)
            
            return portfolio_responses
            
        except Exception as e:
            logger.error(f"Error getting portfolios for user {user_id}: {e}")
            raise
    
    async def get_portfolio(
        self,
        portfolio_id: str,
        user_id: str,
        db: Session
    ) -> Optional[PortfolioResponse]:
        """Get a specific portfolio."""
        try:
            portfolio = db.query(Portfolio).filter(
                and_(
                    Portfolio.id == portfolio_id,
                    or_(
                        Portfolio.user_id == user_id,
                        Portfolio.is_public == True
                    )
                )
            ).first()
            
            if not portfolio:
                return None
            
            return await self._portfolio_to_response(portfolio, db)
            
        except Exception as e:
            logger.error(f"Error getting portfolio {portfolio_id}: {e}")
            raise
    
    async def add_holding(
        self,
        portfolio_id: str,
        user_id: str,
        holding_data: PortfolioHoldingCreate,
        db: Session
    ) -> PortfolioHoldingResponse:
        """Add or update a holding in a portfolio."""
        try:
            # Verify portfolio ownership
            portfolio = db.query(Portfolio).filter(
                and_(
                    Portfolio.id == portfolio_id,
                    Portfolio.user_id == user_id
                )
            ).first()
            
            if not portfolio:
                raise ValueError("Portfolio not found or access denied")
            
            # Check if holding already exists
            existing_holding = db.query(PortfolioHolding).filter(
                and_(
                    PortfolioHolding.portfolio_id == portfolio_id,
                    PortfolioHolding.symbol == holding_data.symbol
                )
            ).first()
            
            if existing_holding:
                # Update existing holding (average cost calculation)
                total_cost = (existing_holding.quantity * existing_holding.average_cost) + \
                           (holding_data.quantity * holding_data.average_cost)
                total_quantity = existing_holding.quantity + holding_data.quantity
                
                if total_quantity > 0:
                    existing_holding.average_cost = total_cost / total_quantity
                    existing_holding.quantity = total_quantity
                    existing_holding.last_transaction_date = datetime.utcnow()
                    
                    if existing_holding.first_purchase_date is None:
                        existing_holding.first_purchase_date = datetime.utcnow()
                    
                    db.commit()
                    db.refresh(existing_holding)
                    
                    return await self._holding_to_response(existing_holding, db)
                else:
                    # Remove holding if quantity becomes 0 or negative
                    db.delete(existing_holding)
                    db.commit()
                    return None
            else:
                # Create new holding
                holding = PortfolioHolding(
                    portfolio_id=portfolio_id,
                    symbol=holding_data.symbol,
                    quantity=holding_data.quantity,
                    average_cost=holding_data.average_cost,
                    first_purchase_date=datetime.utcnow(),
                    last_transaction_date=datetime.utcnow()
                )
                
                db.add(holding)
                db.commit()
                db.refresh(holding)
                
                return await self._holding_to_response(holding, db)
                
        except Exception as e:
            logger.error(f"Error adding holding to portfolio {portfolio_id}: {e}")
            db.rollback()
            raise
    
    async def remove_holding(
        self,
        portfolio_id: str,
        holding_id: str,
        user_id: str,
        db: Session
    ) -> bool:
        """Remove a holding from a portfolio."""
        try:
            # Verify portfolio ownership
            portfolio = db.query(Portfolio).filter(
                and_(
                    Portfolio.id == portfolio_id,
                    Portfolio.user_id == user_id
                )
            ).first()
            
            if not portfolio:
                return False
            
            # Remove holding
            holding = db.query(PortfolioHolding).filter(
                and_(
                    PortfolioHolding.id == holding_id,
                    PortfolioHolding.portfolio_id == portfolio_id
                )
            ).first()
            
            if holding:
                db.delete(holding)
                db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing holding {holding_id} from portfolio {portfolio_id}: {e}")
            db.rollback()
            raise
    
    async def calculate_portfolio_performance(
        self,
        portfolio_id: str,
        user_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio performance metrics."""
        try:
            portfolio = db.query(Portfolio).filter(
                and_(
                    Portfolio.id == portfolio_id,
                    or_(
                        Portfolio.user_id == user_id,
                        Portfolio.is_public == True
                    )
                )
            ).first()
            
            if not portfolio:
                raise ValueError("Portfolio not found or access denied")
            
            holdings = db.query(PortfolioHolding).filter(
                PortfolioHolding.portfolio_id == portfolio_id
            ).all()
            
            if not holdings:
                return {
                    "total_value": 0,
                    "total_cost": 0,
                    "total_profit_loss": 0,
                    "total_profit_loss_percentage": 0,
                    "holdings_performance": [],
                    "asset_allocation": {},
                    "performance_metrics": {}
                }
            
            total_value = 0
            total_cost = 0
            holdings_performance = []
            asset_allocation = {}
            
            for holding in holdings:
                try:
                    # Get current market price
                    asset_data = await self.market_service.get_asset_details(holding.symbol)
                    current_price = asset_data.current_price if asset_data else 0
                    
                    # Calculate holding metrics
                    holding_cost = holding.quantity * holding.average_cost
                    holding_value = holding.quantity * current_price
                    holding_pnl = holding_value - holding_cost
                    holding_pnl_pct = (holding_pnl / holding_cost * 100) if holding_cost > 0 else 0
                    
                    total_value += holding_value
                    total_cost += holding_cost
                    
                    holdings_performance.append({
                        "symbol": holding.symbol,
                        "quantity": holding.quantity,
                        "average_cost": holding.average_cost,
                        "current_price": current_price,
                        "total_cost": holding_cost,
                        "current_value": holding_value,
                        "profit_loss": holding_pnl,
                        "profit_loss_percentage": holding_pnl_pct,
                        "weight": 0  # Will be calculated after total_value is known
                    })
                    
                    # Asset allocation by category
                    if asset_data:
                        category = asset_data.category
                        if category not in asset_allocation:
                            asset_allocation[category] = 0
                        asset_allocation[category] += holding_value
                
                except Exception as e:
                    logger.error(f"Error calculating performance for holding {holding.symbol}: {e}")
                    continue
            
            # Calculate weights and normalize asset allocation
            if total_value > 0:
                for holding_perf in holdings_performance:
                    holding_perf["weight"] = (holding_perf["current_value"] / total_value) * 100
                
                for category in asset_allocation:
                    asset_allocation[category] = (asset_allocation[category] / total_value) * 100
            
            # Calculate overall portfolio metrics
            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            
            # Calculate additional performance metrics
            performance_metrics = await self._calculate_advanced_metrics(
                holdings_performance, portfolio, db
            )
            
            return {
                "total_value": round(total_value, 2),
                "total_cost": round(total_cost, 2),
                "total_profit_loss": round(total_pnl, 2),
                "total_profit_loss_percentage": round(total_pnl_pct, 2),
                "holdings_performance": holdings_performance,
                "asset_allocation": asset_allocation,
                "performance_metrics": performance_metrics
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio performance for {portfolio_id}: {e}")
            raise
    
    async def _calculate_advanced_metrics(
        self,
        holdings_performance: List[Dict[str, Any]],
        portfolio: Portfolio,
        db: Session
    ) -> Dict[str, Any]:
        """Calculate advanced portfolio performance metrics."""
        try:
            # Calculate diversification metrics
            num_holdings = len(holdings_performance)
            
            # Calculate concentration risk (Herfindahl Index)
            herfindahl_index = sum(
                (holding["weight"] / 100) ** 2 for holding in holdings_performance
            )
            
            # Calculate portfolio beta (simplified - would need historical data for accurate calculation)
            # For now, we'll use a placeholder
            portfolio_beta = 1.0  # TODO: Implement proper beta calculation
            
            # Calculate portfolio volatility (placeholder)
            portfolio_volatility = 0.15  # TODO: Implement proper volatility calculation
            
            # Calculate Sharpe ratio (placeholder)
            risk_free_rate = 0.02  # 2% risk-free rate
            portfolio_return = sum(
                holding["profit_loss_percentage"] * (holding["weight"] / 100)
                for holding in holdings_performance
            ) / 100
            
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            return {
                "num_holdings": num_holdings,
                "concentration_risk": round(herfindahl_index, 4),
                "diversification_score": round(1 - herfindahl_index, 4),
                "portfolio_beta": round(portfolio_beta, 2),
                "portfolio_volatility": round(portfolio_volatility * 100, 2),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "portfolio_return": round(portfolio_return * 100, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating advanced metrics: {e}")
            return {}
    
    async def get_portfolio_history(
        self,
        portfolio_id: str,
        user_id: str,
        days: int = 30,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """Get portfolio value history over time."""
        # TODO: Implement portfolio history tracking
        # This would require storing daily portfolio snapshots
        # For now, return placeholder data
        
        history = []
        end_date = datetime.utcnow()
        
        for i in range(days):
            date = end_date - timedelta(days=i)
            # Placeholder data - in real implementation, fetch from portfolio_history table
            history.append({
                "date": date.isoformat(),
                "total_value": 10000 + (i * 100),  # Placeholder
                "profit_loss": i * 50,  # Placeholder
                "profit_loss_percentage": (i * 0.5)  # Placeholder
            })
        
        return list(reversed(history))
    
    async def rebalance_portfolio(
        self,
        portfolio_id: str,
        user_id: str,
        target_allocation: Dict[str, float],
        db: Session
    ) -> Dict[str, Any]:
        """Generate portfolio rebalancing recommendations."""
        try:
            # Get current portfolio performance
            current_performance = await self.calculate_portfolio_performance(
                portfolio_id, user_id, db
            )
            
            current_allocation = current_performance["asset_allocation"]
            total_value = current_performance["total_value"]
            
            rebalancing_actions = []
            
            for category, target_pct in target_allocation.items():
                current_pct = current_allocation.get(category, 0)
                difference = target_pct - current_pct
                target_value = total_value * (target_pct / 100)
                current_value = total_value * (current_pct / 100)
                action_value = target_value - current_value
                
                if abs(difference) > 1:  # Only suggest rebalancing if difference > 1%
                    action = "buy" if difference > 0 else "sell"
                    rebalancing_actions.append({
                        "category": category,
                        "current_percentage": round(current_pct, 2),
                        "target_percentage": target_pct,
                        "difference_percentage": round(difference, 2),
                        "action": action,
                        "action_value": round(abs(action_value), 2)
                    })
            
            return {
                "rebalancing_needed": len(rebalancing_actions) > 0,
                "actions": rebalancing_actions,
                "current_allocation": current_allocation,
                "target_allocation": target_allocation
            }
            
        except Exception as e:
            logger.error(f"Error generating rebalancing recommendations for {portfolio_id}: {e}")
            raise
    
    async def _portfolio_to_response(
        self,
        portfolio: Portfolio,
        db: Session
    ) -> PortfolioResponse:
        """Convert Portfolio model to response schema."""
        try:
            # Get holdings
            holdings = db.query(PortfolioHolding).filter(
                PortfolioHolding.portfolio_id == portfolio.id
            ).all()
            
            holdings_responses = []
            total_value = 0
            total_cost = 0
            
            for holding in holdings:
                holding_response = await self._holding_to_response(holding, db)
                holdings_responses.append(holding_response)
                
                if holding_response.current_value:
                    total_value += holding_response.current_value
                total_cost += holding.quantity * holding.average_cost
            
            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0
            
            return PortfolioResponse(
                id=portfolio.id,
                user_id=portfolio.user_id,
                name=portfolio.name,
                description=portfolio.description,
                is_public=portfolio.is_public,
                currency=portfolio.currency,
                created_at=portfolio.created_at,
                updated_at=portfolio.updated_at,
                holdings=holdings_responses,
                total_value=round(total_value, 2),
                total_profit_loss=round(total_pnl, 2),
                total_profit_loss_percentage=round(total_pnl_pct, 2)
            )
            
        except Exception as e:
            logger.error(f"Error converting portfolio to response: {e}")
            raise
    
    async def _holding_to_response(
        self,
        holding: PortfolioHolding,
        db: Session
    ) -> PortfolioHoldingResponse:
        """Convert PortfolioHolding model to response schema."""
        try:
            # Get current market price
            asset_data = await self.market_service.get_asset_details(holding.symbol)
            current_price = asset_data.current_price if asset_data else 0
            
            # Calculate metrics
            total_cost = holding.quantity * holding.average_cost
            current_value = holding.quantity * current_price
            profit_loss = current_value - total_cost
            profit_loss_pct = (profit_loss / total_cost * 100) if total_cost > 0 else 0
            
            return PortfolioHoldingResponse(
                id=holding.id,
                portfolio_id=holding.portfolio_id,
                symbol=holding.symbol,
                quantity=holding.quantity,
                average_cost=holding.average_cost,
                first_purchase_date=holding.first_purchase_date,
                last_transaction_date=holding.last_transaction_date,
                current_value=round(current_value, 2),
                profit_loss=round(profit_loss, 2),
                profit_loss_percentage=round(profit_loss_pct, 2)
            )
            
        except Exception as e:
            logger.error(f"Error converting holding to response: {e}")
            raise

# Global portfolio service instance
portfolio_service = PortfolioService()