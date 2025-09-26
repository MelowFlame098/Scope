from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import (
    PortfolioCreate,
    PortfolioUpdate,
    PortfolioResponse,
    PortfolioHoldingCreate,
    PortfolioHoldingUpdate,
    PortfolioHoldingResponse,
    TransactionCreate,
    MessageResponse,
    PaginatedResponse,
    PortfolioPerformance
)
from ..services.portfolio_service import PortfolioService
from ..core.security import get_current_user
from ..models import User

router = APIRouter()


@router.get("/", response_model=List[PortfolioResponse])
def get_user_portfolios(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get all portfolios for the current user.
    """
    portfolio_service = PortfolioService(db)
    portfolios = portfolio_service.get_user_portfolios(current_user.id)
    return [PortfolioResponse.from_orm(portfolio) for portfolio in portfolios]


@router.post("/", response_model=PortfolioResponse)
def create_portfolio(
    portfolio_data: PortfolioCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Create a new portfolio.
    """
    portfolio_service = PortfolioService(db)
    
    # Check if portfolio name already exists for this user
    existing_portfolios = portfolio_service.get_user_portfolios(current_user.id)
    if any(p.name == portfolio_data.name for p in existing_portfolios):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Portfolio with this name already exists"
        )
    
    portfolio = portfolio_service.create_portfolio(
        user_id=current_user.id,
        portfolio_data=portfolio_data
    )
    return PortfolioResponse.from_orm(portfolio)


@router.get("/{portfolio_id}", response_model=PortfolioResponse)
def get_portfolio(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get a specific portfolio.
    """
    portfolio_service = PortfolioService(db)
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this portfolio"
        )
    
    return PortfolioResponse.from_orm(portfolio)


@router.put("/{portfolio_id}", response_model=PortfolioResponse)
def update_portfolio(
    portfolio_id: int,
    portfolio_update: PortfolioUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update a portfolio.
    """
    portfolio_service = PortfolioService(db)
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this portfolio"
        )
    
    # Check if new name conflicts with existing portfolios
    if portfolio_update.name and portfolio_update.name != portfolio.name:
        existing_portfolios = portfolio_service.get_user_portfolios(current_user.id)
        if any(p.name == portfolio_update.name and p.id != portfolio_id for p in existing_portfolios):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Portfolio with this name already exists"
            )
    
    updated_portfolio = portfolio_service.update_portfolio(
        portfolio_id, portfolio_update
    )
    return PortfolioResponse.from_orm(updated_portfolio)


@router.delete("/{portfolio_id}", response_model=MessageResponse)
def delete_portfolio(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Delete a portfolio.
    """
    portfolio_service = PortfolioService(db)
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this portfolio"
        )
    
    portfolio_service.delete_portfolio(portfolio_id)
    
    return MessageResponse(
        message="Portfolio deleted successfully",
        success=True
    )


@router.get("/{portfolio_id}/performance", response_model=PortfolioPerformance)
def get_portfolio_performance(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get portfolio performance metrics.
    """
    portfolio_service = PortfolioService(db)
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this portfolio"
        )
    
    performance = portfolio_service.calculate_portfolio_performance(portfolio_id)
    return performance


# Portfolio Holdings endpoints
@router.get("/{portfolio_id}/holdings", response_model=List[PortfolioHoldingResponse])
def get_portfolio_holdings(
    portfolio_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get all holdings in a portfolio.
    """
    portfolio_service = PortfolioService(db)
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this portfolio"
        )
    
    holdings = portfolio_service.get_portfolio_holdings(portfolio_id)
    return [PortfolioHoldingResponse.from_orm(holding) for holding in holdings]


@router.post("/{portfolio_id}/holdings", response_model=PortfolioHoldingResponse)
def add_holding(
    portfolio_id: int,
    holding_data: PortfolioHoldingCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Add a new holding to a portfolio.
    """
    portfolio_service = PortfolioService(db)
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this portfolio"
        )
    
    holding = portfolio_service.add_holding(
        portfolio_id=portfolio_id,
        asset_id=holding_data.asset_id,
        quantity=holding_data.quantity,
        purchase_price=holding_data.purchase_price
    )
    return PortfolioHoldingResponse.from_orm(holding)


@router.put("/{portfolio_id}/holdings/{holding_id}", response_model=PortfolioHoldingResponse)
def update_holding(
    portfolio_id: int,
    holding_id: int,
    holding_update: PortfolioHoldingUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update a portfolio holding.
    """
    portfolio_service = PortfolioService(db)
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this portfolio"
        )
    
    # Verify holding belongs to this portfolio
    holding = portfolio_service.get_holding(holding_id)
    if not holding or holding.portfolio_id != portfolio_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Holding not found in this portfolio"
        )
    
    updated_holding = portfolio_service.update_holding(
        holding_id, holding_update
    )
    return PortfolioHoldingResponse.from_orm(updated_holding)


@router.delete("/{portfolio_id}/holdings/{holding_id}", response_model=MessageResponse)
def remove_holding(
    portfolio_id: int,
    holding_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Remove a holding from a portfolio.
    """
    portfolio_service = PortfolioService(db)
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this portfolio"
        )
    
    # Verify holding belongs to this portfolio
    holding = portfolio_service.get_holding(holding_id)
    if not holding or holding.portfolio_id != portfolio_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Holding not found in this portfolio"
        )
    
    portfolio_service.remove_holding(holding_id)
    
    return MessageResponse(
        message="Holding removed successfully",
        success=True
    )


# Transaction endpoints
@router.post("/{portfolio_id}/holdings/{holding_id}/buy", response_model=PortfolioHoldingResponse)
def buy_more_shares(
    portfolio_id: int,
    holding_id: int,
    transaction_data: TransactionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Buy more shares of an existing holding.
    """
    portfolio_service = PortfolioService(db)
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this portfolio"
        )
    
    # Verify holding belongs to this portfolio
    holding = portfolio_service.get_holding(holding_id)
    if not holding or holding.portfolio_id != portfolio_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Holding not found in this portfolio"
        )
    
    updated_holding = portfolio_service.buy_more_shares(
        holding_id=holding_id,
        quantity=transaction_data.quantity,
        price=transaction_data.price
    )
    return PortfolioHoldingResponse.from_orm(updated_holding)


@router.post("/{portfolio_id}/holdings/{holding_id}/sell", response_model=PortfolioHoldingResponse)
def sell_shares(
    portfolio_id: int,
    holding_id: int,
    transaction_data: TransactionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Sell shares of an existing holding.
    """
    portfolio_service = PortfolioService(db)
    portfolio = portfolio_service.get_portfolio(portfolio_id)
    
    if not portfolio:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Portfolio not found"
        )
    
    if portfolio.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to modify this portfolio"
        )
    
    # Verify holding belongs to this portfolio
    holding = portfolio_service.get_holding(holding_id)
    if not holding or holding.portfolio_id != portfolio_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Holding not found in this portfolio"
        )
    
    if transaction_data.quantity > holding.quantity:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot sell more shares than owned"
        )
    
    updated_holding = portfolio_service.sell_shares(
        holding_id=holding_id,
        quantity=transaction_data.quantity,
        price=transaction_data.price
    )
    
    if updated_holding:
        return PortfolioHoldingResponse.from_orm(updated_holding)
    else:
        # Holding was completely sold and removed
        return MessageResponse(
            message="All shares sold, holding removed from portfolio",
            success=True
        )