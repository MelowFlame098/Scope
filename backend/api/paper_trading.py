from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime

from database import get_db
from auth import get_current_active_user
from db_models import User
from paper_trading_service import paper_trading_service

router = APIRouter(prefix="/api/paper-trading", tags=["paper-trading"])

# Pydantic models for request/response
class CreateOrderRequest(BaseModel):
    portfolio_id: str
    symbol: str
    side: str = Field(..., regex="^(buy|sell)$")
    quantity: float = Field(..., gt=0)
    order_type: str = Field(default="market", regex="^(market|limit|stop|stop_limit)$")
    limit_price: Optional[float] = Field(None, gt=0)
    stop_price: Optional[float] = Field(None, gt=0)
    time_in_force: str = Field(default="day", regex="^(day|gtc|ioc|fok)$")

class OrderResponse(BaseModel):
    order_id: str
    status: str
    symbol: str
    side: str
    quantity: float
    order_type: str
    created_at: str

class CancelOrderResponse(BaseModel):
    order_id: str
    status: str
    cancelled_at: str

class PositionResponse(BaseModel):
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    pnl_percentage: float

class PortfolioSummaryResponse(BaseModel):
    portfolio_id: str
    total_value: float
    total_invested: float
    available_cash: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    pnl_percentage: float
    positions: List[PositionResponse]
    updated_at: str

@router.post("/orders", response_model=OrderResponse)
async def create_order(
    order_request: CreateOrderRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create a new paper trading order
    """
    try:
        result = await paper_trading_service.create_paper_order(
            db=db,
            user_id=current_user.id,
            portfolio_id=order_request.portfolio_id,
            symbol=order_request.symbol.upper(),
            side=order_request.side,
            quantity=order_request.quantity,
            order_type=order_request.order_type,
            limit_price=order_request.limit_price,
            stop_price=order_request.stop_price,
            time_in_force=order_request.time_in_force
        )
        
        return OrderResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create order"
        )

@router.delete("/orders/{order_id}", response_model=CancelOrderResponse)
async def cancel_order(
    order_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Cancel a pending paper trading order
    """
    try:
        result = await paper_trading_service.cancel_order(
            db=db,
            user_id=current_user.id,
            order_id=order_id
        )
        
        return CancelOrderResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel order"
        )

@router.get("/portfolio/{portfolio_id}/summary", response_model=PortfolioSummaryResponse)
async def get_portfolio_summary(
    portfolio_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get real-time portfolio summary with current positions and P&L
    """
    try:
        result = await paper_trading_service.get_portfolio_summary(
            db=db,
            user_id=current_user.id,
            portfolio_id=portfolio_id
        )
        
        return PortfolioSummaryResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get portfolio summary"
        )

@router.get("/orders")
async def get_orders(
    portfolio_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get user's paper trading orders with optional filtering
    """
    try:
        from db_models import Order, Portfolio
        from sqlalchemy import and_, desc
        
        query = db.query(Order).join(Portfolio).filter(Portfolio.user_id == current_user.id)
        
        if portfolio_id:
            query = query.filter(Order.portfolio_id == portfolio_id)
        
        if status:
            query = query.filter(Order.status == status)
        
        orders = query.order_by(desc(Order.created_at)).limit(limit).all()
        
        return [
            {
                "order_id": order.id,
                "portfolio_id": order.portfolio_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "order_type": order.order_type,
                "limit_price": order.limit_price,
                "stop_price": order.stop_price,
                "status": order.status,
                "filled_quantity": order.filled_quantity,
                "average_fill_price": order.average_fill_price,
                "commission": order.commission,
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat() if order.updated_at else None
            }
            for order in orders
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get orders"
        )

@router.get("/trades")
async def get_trades(
    portfolio_id: Optional[str] = None,
    symbol: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get user's paper trading trade history
    """
    try:
        from db_models import Trade, Portfolio
        from sqlalchemy import and_, desc
        
        query = db.query(Trade).join(Portfolio).filter(Portfolio.user_id == current_user.id)
        
        if portfolio_id:
            query = query.filter(Trade.portfolio_id == portfolio_id)
        
        if symbol:
            query = query.filter(Trade.symbol == symbol.upper())
        
        trades = query.order_by(desc(Trade.timestamp)).limit(limit).all()
        
        return [
            {
                "trade_id": trade.id,
                "order_id": trade.order_id,
                "portfolio_id": trade.portfolio_id,
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "price": trade.price,
                "commission": trade.commission,
                "timestamp": trade.timestamp.isoformat()
            }
            for trade in trades
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get trades"
        )

@router.get("/positions")
async def get_positions(
    portfolio_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get current positions for a portfolio
    """
    try:
        from db_models import Position, Portfolio
        from sqlalchemy import and_
        
        # Verify portfolio ownership
        portfolio = db.query(Portfolio).filter(
            and_(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
        ).first()
        
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )
        
        positions = db.query(Position).filter(Position.portfolio_id == portfolio_id).all()
        
        result = []
        for position in positions:
            # Get current price for real-time data
            current_price = await paper_trading_service.market_data_service.get_current_price(position.symbol)
            
            if current_price:
                market_value = position.quantity * current_price
                unrealized_pnl = (current_price - position.average_cost) * position.quantity
                pnl_percentage = (unrealized_pnl / (position.quantity * position.average_cost)) * 100 if position.quantity > 0 else 0
                
                result.append({
                    "symbol": position.symbol,
                    "quantity": position.quantity,
                    "average_cost": position.average_cost,
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pnl": unrealized_pnl,
                    "realized_pnl": position.realized_pnl,
                    "pnl_percentage": pnl_percentage,
                    "updated_at": position.updated_at.isoformat() if position.updated_at else None
                })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get positions"
        )

@router.get("/market-hours")
async def get_market_hours():
    """
    Get current market hours and status (simplified for paper trading)
    """
    return {
        "is_open": True,  # Paper trading is always open
        "next_open": None,
        "next_close": None,
        "timezone": "UTC",
        "note": "Paper trading is available 24/7"
    }

@router.get("/performance/{portfolio_id}")
async def get_portfolio_performance(
    portfolio_id: str,
    days: int = 30,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Get portfolio performance metrics over time
    """
    try:
        from db_models import Portfolio, Trade
        from sqlalchemy import and_, func
        from datetime import datetime, timedelta
        
        # Verify portfolio ownership
        portfolio = db.query(Portfolio).filter(
            and_(Portfolio.id == portfolio_id, Portfolio.user_id == current_user.id)
        ).first()
        
        if not portfolio:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Portfolio not found"
            )
        
        # Get trades within the specified period
        start_date = datetime.utcnow() - timedelta(days=days)
        trades = db.query(Trade).filter(
            and_(
                Trade.portfolio_id == portfolio_id,
                Trade.timestamp >= start_date
            )
        ).order_by(Trade.timestamp).all()
        
        # Calculate daily P&L (simplified)
        daily_pnl = {}
        total_invested = 0
        
        for trade in trades:
            date_key = trade.timestamp.date().isoformat()
            if date_key not in daily_pnl:
                daily_pnl[date_key] = 0
            
            if trade.side == 'buy':
                total_invested += trade.quantity * trade.price
            else:
                # Simplified P&L calculation
                daily_pnl[date_key] += trade.quantity * trade.price
        
        # Get current portfolio summary
        summary = await paper_trading_service.get_portfolio_summary(db, current_user.id, portfolio_id)
        
        return {
            "portfolio_id": portfolio_id,
            "period_days": days,
            "total_trades": len(trades),
            "total_invested": total_invested,
            "current_value": summary["total_value"],
            "total_pnl": summary["total_pnl"],
            "pnl_percentage": summary["pnl_percentage"],
            "daily_pnl": daily_pnl,
            "updated_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get portfolio performance"
        )