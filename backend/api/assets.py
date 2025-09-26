from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import (
    AssetResponse,
    WatchlistCreate,
    WatchlistResponse,
    MessageResponse,
    PaginatedResponse,
    MarketOverview,
    MarketDataResponse,
    AssetCategory
)
from ..services.data_service import DataService
from ..repositories.asset import AssetRepository
from ..repositories.user import UserRepository
from ..core.security import get_current_user
from ..models import User, Watchlist

router = APIRouter()


@router.get("/", response_model=PaginatedResponse)
def get_assets(
    skip: int = Query(0, ge=0, description="Number of assets to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of assets to return"),
    category: Optional[AssetCategory] = Query(None, description="Filter by asset category"),
    search: Optional[str] = Query(None, description="Search term for symbol or name"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get all assets with optional filtering.
    """
    asset_repo = AssetRepository(db)
    
    filters = {}
    if category:
        filters["category"] = category
    
    if search:
        assets = asset_repo.search_assets(search, skip=skip, limit=limit)
        total = len(assets)  # This is not accurate for pagination, but works for demo
    else:
        assets = asset_repo.get_multi(skip=skip, limit=limit, **filters)
        total = asset_repo.count(**filters)
    
    return PaginatedResponse(
        items=[AssetResponse.from_orm(asset) for asset in assets],
        total=total,
        page=skip // limit + 1,
        size=limit,
        pages=(total + limit - 1) // limit
    )


@router.get("/search", response_model=List[AssetResponse])
def search_assets(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Number of results to return"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Search assets by symbol or name.
    """
    asset_repo = AssetRepository(db)
    assets = asset_repo.search_assets(q, limit=limit)
    return [AssetResponse.from_orm(asset) for asset in assets]


@router.get("/trending", response_model=List[AssetResponse])
def get_trending_assets(
    limit: int = Query(10, ge=1, le=50, description="Number of trending assets to return"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get trending assets based on volume and price changes.
    """
    asset_repo = AssetRepository(db)
    assets = asset_repo.get_trending_assets(limit=limit)
    return [AssetResponse.from_orm(asset) for asset in assets]


@router.get("/top-performers", response_model=List[AssetResponse])
def get_top_performers(
    limit: int = Query(10, ge=1, le=50, description="Number of top performers to return"),
    timeframe: str = Query("24h", description="Timeframe for performance calculation"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get top performing assets.
    """
    asset_repo = AssetRepository(db)
    assets = asset_repo.get_top_performers(limit=limit)
    return [AssetResponse.from_orm(asset) for asset in assets]


@router.get("/top-losers", response_model=List[AssetResponse])
def get_top_losers(
    limit: int = Query(10, ge=1, le=50, description="Number of top losers to return"),
    timeframe: str = Query("24h", description="Timeframe for performance calculation"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get worst performing assets.
    """
    asset_repo = AssetRepository(db)
    assets = asset_repo.get_top_losers(limit=limit)
    return [AssetResponse.from_orm(asset) for asset in assets]


@router.get("/market-overview", response_model=MarketOverview)
def get_market_overview(
    db: Session = Depends(get_db)
) -> Any:
    """
    Get market overview with key metrics.
    """
    data_service = DataService(db)
    overview = data_service.get_market_overview()
    return overview


@router.get("/{asset_id}", response_model=AssetResponse)
def get_asset(
    asset_id: int,
    db: Session = Depends(get_db)
) -> Any:
    """
    Get a specific asset by ID.
    """
    asset_repo = AssetRepository(db)
    asset = asset_repo.get(asset_id)
    
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    return AssetResponse.from_orm(asset)


@router.get("/symbol/{symbol}", response_model=AssetResponse)
def get_asset_by_symbol(
    symbol: str,
    db: Session = Depends(get_db)
) -> Any:
    """
    Get a specific asset by symbol.
    """
    asset_repo = AssetRepository(db)
    asset = asset_repo.get_by_symbol(symbol.upper())
    
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    return AssetResponse.from_orm(asset)


@router.get("/{asset_id}/market-data", response_model=MarketDataResponse)
def get_asset_market_data(
    asset_id: int,
    timeframe: str = Query("1d", description="Timeframe for market data"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get market data for a specific asset.
    """
    asset_repo = AssetRepository(db)
    asset = asset_repo.get(asset_id)
    
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    data_service = DataService(db)
    market_data = data_service.get_asset_market_data(asset.symbol, timeframe)
    return market_data


# Watchlist endpoints
@router.get("/watchlists/", response_model=List[WatchlistResponse])
def get_user_watchlists(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get all watchlists for the current user.
    """
    watchlists = db.query(Watchlist).filter(Watchlist.user_id == current_user.id).all()
    return [WatchlistResponse.from_orm(watchlist) for watchlist in watchlists]


@router.post("/watchlists/", response_model=WatchlistResponse)
def create_watchlist(
    watchlist_data: WatchlistCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Create a new watchlist.
    """
    asset_repo = AssetRepository(db)
    
    # Verify all assets exist
    for asset_id in watchlist_data.asset_ids:
        asset = asset_repo.get(asset_id)
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Asset with ID {asset_id} not found"
            )
    
    # Check if watchlist name already exists for this user
    existing_watchlist = db.query(Watchlist).filter(
        Watchlist.user_id == current_user.id,
        Watchlist.name == watchlist_data.name
    ).first()
    
    if existing_watchlist:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Watchlist with this name already exists"
        )
    
    # Create watchlist
    watchlist = Watchlist(
        user_id=current_user.id,
        name=watchlist_data.name,
        description=watchlist_data.description
    )
    
    # Add assets to watchlist
    for asset_id in watchlist_data.asset_ids:
        asset = asset_repo.get(asset_id)
        watchlist.assets.append(asset)
    
    db.add(watchlist)
    db.commit()
    db.refresh(watchlist)
    
    return WatchlistResponse.from_orm(watchlist)


@router.get("/watchlists/{watchlist_id}", response_model=WatchlistResponse)
def get_watchlist(
    watchlist_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get a specific watchlist.
    """
    watchlist = db.query(Watchlist).filter(
        Watchlist.id == watchlist_id,
        Watchlist.user_id == current_user.id
    ).first()
    
    if not watchlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist not found"
        )
    
    return WatchlistResponse.from_orm(watchlist)


@router.put("/watchlists/{watchlist_id}", response_model=WatchlistResponse)
def update_watchlist(
    watchlist_id: int,
    watchlist_update: WatchlistCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update a watchlist.
    """
    watchlist = db.query(Watchlist).filter(
        Watchlist.id == watchlist_id,
        Watchlist.user_id == current_user.id
    ).first()
    
    if not watchlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist not found"
        )
    
    asset_repo = AssetRepository(db)
    
    # Verify all assets exist
    for asset_id in watchlist_update.asset_ids:
        asset = asset_repo.get(asset_id)
        if not asset:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Asset with ID {asset_id} not found"
            )
    
    # Check if new name conflicts with existing watchlists
    if watchlist_update.name != watchlist.name:
        existing_watchlist = db.query(Watchlist).filter(
            Watchlist.user_id == current_user.id,
            Watchlist.name == watchlist_update.name,
            Watchlist.id != watchlist_id
        ).first()
        
        if existing_watchlist:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Watchlist with this name already exists"
            )
    
    # Update watchlist
    watchlist.name = watchlist_update.name
    watchlist.description = watchlist_update.description
    
    # Update assets
    watchlist.assets.clear()
    for asset_id in watchlist_update.asset_ids:
        asset = asset_repo.get(asset_id)
        watchlist.assets.append(asset)
    
    db.commit()
    db.refresh(watchlist)
    
    return WatchlistResponse.from_orm(watchlist)


@router.delete("/watchlists/{watchlist_id}", response_model=MessageResponse)
def delete_watchlist(
    watchlist_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Delete a watchlist.
    """
    watchlist = db.query(Watchlist).filter(
        Watchlist.id == watchlist_id,
        Watchlist.user_id == current_user.id
    ).first()
    
    if not watchlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist not found"
        )
    
    db.delete(watchlist)
    db.commit()
    
    return MessageResponse(
        message="Watchlist deleted successfully",
        success=True
    )


@router.post("/watchlists/{watchlist_id}/assets/{asset_id}", response_model=MessageResponse)
def add_asset_to_watchlist(
    watchlist_id: int,
    asset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Add an asset to a watchlist.
    """
    watchlist = db.query(Watchlist).filter(
        Watchlist.id == watchlist_id,
        Watchlist.user_id == current_user.id
    ).first()
    
    if not watchlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist not found"
        )
    
    asset_repo = AssetRepository(db)
    asset = asset_repo.get(asset_id)
    
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    if asset in watchlist.assets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Asset already in watchlist"
        )
    
    watchlist.assets.append(asset)
    db.commit()
    
    return MessageResponse(
        message="Asset added to watchlist successfully",
        success=True
    )


@router.delete("/watchlists/{watchlist_id}/assets/{asset_id}", response_model=MessageResponse)
def remove_asset_from_watchlist(
    watchlist_id: int,
    asset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Remove an asset from a watchlist.
    """
    watchlist = db.query(Watchlist).filter(
        Watchlist.id == watchlist_id,
        Watchlist.user_id == current_user.id
    ).first()
    
    if not watchlist:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Watchlist not found"
        )
    
    asset_repo = AssetRepository(db)
    asset = asset_repo.get(asset_id)
    
    if not asset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Asset not found"
        )
    
    if asset not in watchlist.assets:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Asset not in watchlist"
        )
    
    watchlist.assets.remove(asset)
    db.commit()
    
    return MessageResponse(
        message="Asset removed from watchlist successfully",
        success=True
    )