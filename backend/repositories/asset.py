from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from datetime import datetime, timedelta

from .base import BaseRepository
from ..models import Asset
from ..schemas import AssetCreate, AssetUpdate

class AssetRepository(BaseRepository[Asset, AssetCreate, AssetUpdate]):
    """
    Repository for Asset model with financial data specific operations
    """
    
    def __init__(self):
        super().__init__(Asset)
    
    def get_by_symbol(self, db: Session, *, symbol: str) -> Optional[Asset]:
        """
        Get asset by symbol
        
        Args:
            db: Database session
            symbol: Asset symbol (e.g., BTC, AAPL)
            
        Returns:
            Asset instance or None
        """
        return db.query(Asset).filter(Asset.symbol == symbol.upper()).first()
    
    def get_by_category(self, db: Session, *, category: str, skip: int = 0, limit: int = 100) -> List[Asset]:
        """
        Get assets by category
        
        Args:
            db: Database session
            category: Asset category (crypto, stock, forex, commodity, index)
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of assets in the category
        """
        return (
            db.query(Asset)
            .filter(Asset.category == category)
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def search_assets(self, db: Session, *, query: str, category: Optional[str] = None) -> List[Asset]:
        """
        Search assets by symbol or name
        
        Args:
            db: Database session
            query: Search query
            category: Optional category filter
            
        Returns:
            List of matching assets
        """
        search_filter = or_(
            Asset.symbol.ilike(f"%{query.upper()}%"),
            Asset.name.ilike(f"%{query}%")
        )
        
        base_query = db.query(Asset).filter(search_filter)
        
        if category:
            base_query = base_query.filter(Asset.category == category)
        
        return base_query.limit(50).all()
    
    def get_top_performers(self, db: Session, *, category: Optional[str] = None, limit: int = 10) -> List[Asset]:
        """
        Get top performing assets by 24h price change
        
        Args:
            db: Database session
            category: Optional category filter
            limit: Maximum number of assets to return
            
        Returns:
            List of top performing assets
        """
        query = db.query(Asset).filter(Asset.price_change_percentage_24h.isnot(None))
        
        if category:
            query = query.filter(Asset.category == category)
        
        return (
            query
            .order_by(desc(Asset.price_change_percentage_24h))
            .limit(limit)
            .all()
        )
    
    def get_top_losers(self, db: Session, *, category: Optional[str] = None, limit: int = 10) -> List[Asset]:
        """
        Get worst performing assets by 24h price change
        
        Args:
            db: Database session
            category: Optional category filter
            limit: Maximum number of assets to return
            
        Returns:
            List of worst performing assets
        """
        query = db.query(Asset).filter(Asset.price_change_percentage_24h.isnot(None))
        
        if category:
            query = query.filter(Asset.category == category)
        
        return (
            query
            .order_by(Asset.price_change_percentage_24h)
            .limit(limit)
            .all()
        )
    
    def get_by_market_cap_range(
        self, 
        db: Session, 
        *, 
        min_cap: Optional[float] = None, 
        max_cap: Optional[float] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Asset]:
        """
        Get assets within a market cap range
        
        Args:
            db: Database session
            min_cap: Minimum market cap
            max_cap: Maximum market cap
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of assets within the market cap range
        """
        query = db.query(Asset).filter(Asset.market_cap.isnot(None))
        
        if min_cap is not None:
            query = query.filter(Asset.market_cap >= min_cap)
        
        if max_cap is not None:
            query = query.filter(Asset.market_cap <= max_cap)
        
        return (
            query
            .order_by(desc(Asset.market_cap))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_stale_prices(self, db: Session, *, hours: int = 1) -> List[Asset]:
        """
        Get assets with stale price data (not updated within specified hours)
        
        Args:
            db: Database session
            hours: Number of hours to consider data stale
            
        Returns:
            List of assets with stale price data
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return (
            db.query(Asset)
            .filter(
                or_(
                    Asset.last_price_update < cutoff_time,
                    Asset.last_price_update.is_(None)
                )
            )
            .all()
        )
    
    def update_price_data(
        self, 
        db: Session, 
        *, 
        symbol: str, 
        price_data: Dict[str, Any]
    ) -> Optional[Asset]:
        """
        Update asset price and market data
        
        Args:
            db: Database session
            symbol: Asset symbol
            price_data: Dictionary containing price and market data
            
        Returns:
            Updated asset instance or None
        """
        asset = self.get_by_symbol(db, symbol=symbol)
        if not asset:
            return None
        
        # Update price fields
        if 'current_price' in price_data:
            asset.current_price = price_data['current_price']
        
        if 'price_change_24h' in price_data:
            asset.price_change_24h = price_data['price_change_24h']
        
        if 'price_change_percentage_24h' in price_data:
            asset.price_change_percentage_24h = price_data['price_change_percentage_24h']
        
        if 'market_cap' in price_data:
            asset.market_cap = price_data['market_cap']
        
        if 'volume_24h' in price_data:
            asset.volume_24h = price_data['volume_24h']
        
        if 'technical_indicators' in price_data:
            asset.technical_indicators = price_data['technical_indicators']
        
        asset.last_price_update = datetime.utcnow()
        asset.updated_at = datetime.utcnow()
        
        db.commit()
        db.refresh(asset)
        return asset
    
    def get_trending_assets(self, db: Session, *, limit: int = 20) -> List[Asset]:
        """
        Get trending assets based on volume and price change
        
        Args:
            db: Database session
            limit: Maximum number of assets to return
            
        Returns:
            List of trending assets
        """
        return (
            db.query(Asset)
            .filter(
                and_(
                    Asset.volume_24h.isnot(None),
                    Asset.price_change_percentage_24h.isnot(None)
                )
            )
            .order_by(
                desc(Asset.volume_24h * abs(Asset.price_change_percentage_24h))
            )
            .limit(limit)
            .all()
        )

# Create repository instance
asset_repository = AssetRepository()