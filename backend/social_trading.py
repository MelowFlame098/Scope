from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import json
from enum import Enum

router = APIRouter(prefix="/api/social-trading", tags=["social-trading"])

class TraderTier(str, Enum):
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    DIAMOND = "diamond"

class PostType(str, Enum):
    TRADE = "trade"
    ANALYSIS = "analysis"
    DISCUSSION = "discussion"
    STRATEGY = "strategy"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Pydantic Models
class SocialTrader(BaseModel):
    id: str
    username: str
    displayName: str
    avatar: str
    tier: TraderTier
    followers: int
    following: int
    totalReturn: float
    winRate: float
    avgHoldTime: str
    riskScore: float
    copiers: int
    aum: float  # Assets Under Management
    joinDate: datetime
    isVerified: bool
    badges: List[str]
    bio: str
    tradingStyle: str
    preferredAssets: List[str]

class SocialPost(BaseModel):
    id: str
    authorId: str
    author: SocialTrader
    type: PostType
    title: str
    content: str
    images: List[str] = []
    tags: List[str] = []
    likes: int
    comments: int
    shares: int
    views: int
    isLiked: bool = False
    isBookmarked: bool = False
    createdAt: datetime
    updatedAt: datetime
    tradingData: Optional[Dict[str, Any]] = None

class TradingStrategy(BaseModel):
    id: str
    name: str
    description: str
    authorId: str
    author: SocialTrader
    performance: float
    maxDrawdown: float
    sharpeRatio: float
    winRate: float
    totalTrades: int
    followers: int
    copiers: int
    riskLevel: RiskLevel
    timeframe: str
    assets: List[str]
    minInvestment: float
    fees: float
    isActive: bool
    isFollowing: bool = False
    isCopying: bool = False
    createdAt: datetime
    updatedAt: datetime

class LeaderboardEntry(BaseModel):
    rank: int
    trader: SocialTrader
    metric: str
    value: float
    change: float
    period: str

class CopyTradeSettings(BaseModel):
    traderId: str
    allocation: float
    maxRisk: float
    stopLoss: Optional[float] = None
    takeProfit: Optional[float] = None
    copyRatio: float = 1.0
    isActive: bool = True

class Comment(BaseModel):
    id: str
    postId: str
    authorId: str
    author: SocialTrader
    content: str
    likes: int
    isLiked: bool = False
    createdAt: datetime
    replies: List['Comment'] = []

class FollowRequest(BaseModel):
    traderId: str

class CopyTradeRequest(BaseModel):
    traderId: str
    settings: CopyTradeSettings

class PostRequest(BaseModel):
    type: PostType
    title: str
    content: str
    tags: List[str] = []
    tradingData: Optional[Dict[str, Any]] = None

class CommentRequest(BaseModel):
    postId: str
    content: str

# Service Class
class SocialTradingService:
    def __init__(self):
        self.traders = {}
        self.posts = {}
        self.strategies = {}
        self.comments = {}
        self.follows = {}
        self.copy_trades = {}
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize with mock data for demonstration"""
        # Mock traders
        mock_traders = [
            {
                "id": "trader-1",
                "username": "crypto_master_2024",
                "displayName": "Alex Chen",
                "avatar": "/api/placeholder/50/50",
                "tier": TraderTier.PLATINUM,
                "followers": 15420,
                "following": 234,
                "totalReturn": 156.7,
                "winRate": 78.5,
                "avgHoldTime": "3.2 days",
                "riskScore": 6.8,
                "copiers": 1250,
                "aum": 2500000,
                "joinDate": datetime.now() - timedelta(days=365),
                "isVerified": True,
                "badges": ["Top Performer", "Risk Manager", "Crypto Expert"],
                "bio": "Professional crypto trader with 5+ years experience. Focus on DeFi and altcoins.",
                "tradingStyle": "Swing Trading",
                "preferredAssets": ["BTC", "ETH", "SOL", "AVAX"]
            },
            {
                "id": "trader-2",
                "username": "forex_queen",
                "displayName": "Sarah Johnson",
                "avatar": "/api/placeholder/50/50",
                "tier": TraderTier.GOLD,
                "followers": 8930,
                "following": 156,
                "totalReturn": 89.3,
                "winRate": 82.1,
                "avgHoldTime": "1.5 days",
                "riskScore": 4.2,
                "copiers": 890,
                "aum": 1200000,
                "joinDate": datetime.now() - timedelta(days=280),
                "isVerified": True,
                "badges": ["Consistent Performer", "Low Risk"],
                "bio": "Conservative forex trader specializing in major currency pairs.",
                "tradingStyle": "Scalping",
                "preferredAssets": ["EUR/USD", "GBP/USD", "USD/JPY"]
            }
        ]
        
        for trader_data in mock_traders:
            trader = SocialTrader(**trader_data)
            self.traders[trader.id] = trader
        
        # Mock posts
        mock_posts = [
            {
                "id": "post-1",
                "authorId": "trader-1",
                "author": self.traders["trader-1"],
                "type": PostType.ANALYSIS,
                "title": "Bitcoin Technical Analysis: Bullish Breakout Expected",
                "content": "After analyzing the recent price action, I believe BTC is setting up for a significant breakout above $45k. Key indicators showing strong momentum...",
                "tags": ["Bitcoin", "Technical Analysis", "Bullish"],
                "likes": 234,
                "comments": 45,
                "shares": 23,
                "views": 1250,
                "createdAt": datetime.now() - timedelta(hours=2),
                "updatedAt": datetime.now() - timedelta(hours=2),
                "tradingData": {
                    "symbol": "BTC/USD",
                    "entryPrice": 43000,
                    "targetPrice": 47000,
                    "stopLoss": 41000
                }
            },
            {
                "id": "post-2",
                "authorId": "trader-2",
                "author": self.traders["trader-2"],
                "type": PostType.TRADE,
                "title": "EUR/USD Long Position",
                "content": "Opened a long position on EUR/USD at 1.0850. Expecting a move to 1.0920 based on ECB policy outlook.",
                "tags": ["EUR/USD", "Forex", "Long"],
                "likes": 156,
                "comments": 28,
                "shares": 12,
                "views": 890,
                "createdAt": datetime.now() - timedelta(hours=4),
                "updatedAt": datetime.now() - timedelta(hours=4),
                "tradingData": {
                    "symbol": "EUR/USD",
                    "side": "long",
                    "entryPrice": 1.0850,
                    "quantity": 100000,
                    "targetPrice": 1.0920,
                    "stopLoss": 1.0800
                }
            }
        ]
        
        for post_data in mock_posts:
            post = SocialPost(**post_data)
            self.posts[post.id] = post
        
        # Mock strategies
        mock_strategies = [
            {
                "id": "strategy-1",
                "name": "Momentum Breakout Strategy",
                "description": "A systematic approach to trading momentum breakouts with strict risk management.",
                "authorId": "trader-1",
                "author": self.traders["trader-1"],
                "performance": 23.5,
                "maxDrawdown": -8.2,
                "sharpeRatio": 1.85,
                "winRate": 68.5,
                "totalTrades": 145,
                "followers": 1250,
                "copiers": 890,
                "riskLevel": RiskLevel.MEDIUM,
                "timeframe": "4H - 1D",
                "assets": ["BTC", "ETH", "SOL"],
                "minInvestment": 1000,
                "fees": 0.2,
                "isActive": True,
                "createdAt": datetime.now() - timedelta(days=30),
                "updatedAt": datetime.now() - timedelta(days=1)
            }
        ]
        
        for strategy_data in mock_strategies:
            strategy = TradingStrategy(**strategy_data)
            self.strategies[strategy.id] = strategy
    
    async def get_social_feed(self, user_id: str, limit: int = 20, offset: int = 0) -> List[SocialPost]:
        """Get social trading feed"""
        posts = list(self.posts.values())
        posts.sort(key=lambda x: x.createdAt, reverse=True)
        return posts[offset:offset + limit]
    
    async def get_top_traders(self, metric: str = "return", period: str = "30d", limit: int = 10) -> List[LeaderboardEntry]:
        """Get top traders leaderboard"""
        traders = list(self.traders.values())
        
        if metric == "return":
            traders.sort(key=lambda x: x.totalReturn, reverse=True)
        elif metric == "winrate":
            traders.sort(key=lambda x: x.winRate, reverse=True)
        elif metric == "followers":
            traders.sort(key=lambda x: x.followers, reverse=True)
        elif metric == "copiers":
            traders.sort(key=lambda x: x.copiers, reverse=True)
        
        leaderboard = []
        for i, trader in enumerate(traders[:limit]):
            entry = LeaderboardEntry(
                rank=i + 1,
                trader=trader,
                metric=metric,
                value=getattr(trader, metric.replace("return", "totalReturn").replace("winrate", "winRate")),
                change=5.2,  # Mock change
                period=period
            )
            leaderboard.append(entry)
        
        return leaderboard
    
    async def get_trading_strategies(self, user_id: str, limit: int = 20) -> List[TradingStrategy]:
        """Get available trading strategies"""
        strategies = list(self.strategies.values())
        strategies.sort(key=lambda x: x.performance, reverse=True)
        return strategies[:limit]
    
    async def get_trader_profile(self, trader_id: str) -> SocialTrader:
        """Get trader profile"""
        if trader_id not in self.traders:
            raise HTTPException(status_code=404, detail="Trader not found")
        return self.traders[trader_id]
    
    async def follow_trader(self, user_id: str, trader_id: str) -> bool:
        """Follow a trader"""
        if trader_id not in self.traders:
            raise HTTPException(status_code=404, detail="Trader not found")
        
        if user_id not in self.follows:
            self.follows[user_id] = set()
        
        if trader_id not in self.follows[user_id]:
            self.follows[user_id].add(trader_id)
            self.traders[trader_id].followers += 1
            return True
        return False
    
    async def unfollow_trader(self, user_id: str, trader_id: str) -> bool:
        """Unfollow a trader"""
        if user_id in self.follows and trader_id in self.follows[user_id]:
            self.follows[user_id].remove(trader_id)
            self.traders[trader_id].followers -= 1
            return True
        return False
    
    async def copy_trader(self, user_id: str, settings: CopyTradeSettings) -> str:
        """Start copying a trader"""
        if settings.traderId not in self.traders:
            raise HTTPException(status_code=404, detail="Trader not found")
        
        copy_id = f"copy-{user_id}-{settings.traderId}-{int(datetime.now().timestamp())}"
        self.copy_trades[copy_id] = {
            "id": copy_id,
            "userId": user_id,
            "settings": settings,
            "createdAt": datetime.now(),
            "isActive": True
        }
        
        self.traders[settings.traderId].copiers += 1
        return copy_id
    
    async def stop_copying_trader(self, user_id: str, trader_id: str) -> bool:
        """Stop copying a trader"""
        for copy_id, copy_data in self.copy_trades.items():
            if (copy_data["userId"] == user_id and 
                copy_data["settings"].traderId == trader_id and 
                copy_data["isActive"]):
                copy_data["isActive"] = False
                self.traders[trader_id].copiers -= 1
                return True
        return False
    
    async def create_post(self, user_id: str, post_data: PostRequest) -> SocialPost:
        """Create a new social post"""
        if user_id not in self.traders:
            raise HTTPException(status_code=404, detail="User not found")
        
        post_id = f"post-{int(datetime.now().timestamp())}"
        post = SocialPost(
            id=post_id,
            authorId=user_id,
            author=self.traders[user_id],
            type=post_data.type,
            title=post_data.title,
            content=post_data.content,
            tags=post_data.tags,
            likes=0,
            comments=0,
            shares=0,
            views=0,
            createdAt=datetime.now(),
            updatedAt=datetime.now(),
            tradingData=post_data.tradingData
        )
        
        self.posts[post_id] = post
        return post
    
    async def like_post(self, user_id: str, post_id: str) -> bool:
        """Like/unlike a post"""
        if post_id not in self.posts:
            raise HTTPException(status_code=404, detail="Post not found")
        
        post = self.posts[post_id]
        # Toggle like (simplified logic)
        if not post.isLiked:
            post.likes += 1
            post.isLiked = True
        else:
            post.likes -= 1
            post.isLiked = False
        
        return post.isLiked
    
    async def add_comment(self, user_id: str, comment_data: CommentRequest) -> Comment:
        """Add comment to a post"""
        if comment_data.postId not in self.posts:
            raise HTTPException(status_code=404, detail="Post not found")
        
        if user_id not in self.traders:
            raise HTTPException(status_code=404, detail="User not found")
        
        comment_id = f"comment-{int(datetime.now().timestamp())}"
        comment = Comment(
            id=comment_id,
            postId=comment_data.postId,
            authorId=user_id,
            author=self.traders[user_id],
            content=comment_data.content,
            likes=0,
            createdAt=datetime.now()
        )
        
        self.comments[comment_id] = comment
        self.posts[comment_data.postId].comments += 1
        
        return comment
    
    async def get_user_copy_trades(self, user_id: str) -> List[Dict[str, Any]]:
        """Get user's active copy trades"""
        user_copies = []
        for copy_id, copy_data in self.copy_trades.items():
            if copy_data["userId"] == user_id and copy_data["isActive"]:
                trader = self.traders[copy_data["settings"].traderId]
                user_copies.append({
                    "id": copy_id,
                    "trader": trader,
                    "settings": copy_data["settings"],
                    "performance": 12.5,  # Mock performance
                    "pnl": 1250.0,  # Mock P&L
                    "createdAt": copy_data["createdAt"]
                })
        return user_copies

# Initialize service
social_trading_service = SocialTradingService()

# API Endpoints
@router.get("/feed", response_model=List[SocialPost])
async def get_social_feed(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user_id: str = "current-user"  # In real app, get from auth
):
    """Get social trading feed"""
    return await social_trading_service.get_social_feed(user_id, limit, offset)

@router.get("/traders/top", response_model=List[LeaderboardEntry])
async def get_top_traders(
    metric: str = Query("return", regex="^(return|winrate|followers|copiers)$"),
    period: str = Query("30d", regex="^(7d|30d|90d|1y)$"),
    limit: int = Query(10, ge=1, le=50)
):
    """Get top traders leaderboard"""
    return await social_trading_service.get_top_traders(metric, period, limit)

@router.get("/strategies", response_model=List[TradingStrategy])
async def get_trading_strategies(
    limit: int = Query(20, ge=1, le=100),
    user_id: str = "current-user"
):
    """Get available trading strategies"""
    return await social_trading_service.get_trading_strategies(user_id, limit)

@router.get("/traders/{trader_id}", response_model=SocialTrader)
async def get_trader_profile(trader_id: str):
    """Get trader profile"""
    return await social_trading_service.get_trader_profile(trader_id)

@router.post("/follow")
async def follow_trader(
    request: FollowRequest,
    user_id: str = "current-user"
):
    """Follow a trader"""
    success = await social_trading_service.follow_trader(user_id, request.traderId)
    return {"success": success, "message": "Trader followed successfully" if success else "Already following"}

@router.delete("/follow/{trader_id}")
async def unfollow_trader(
    trader_id: str,
    user_id: str = "current-user"
):
    """Unfollow a trader"""
    success = await social_trading_service.unfollow_trader(user_id, trader_id)
    return {"success": success, "message": "Trader unfollowed successfully" if success else "Not following"}

@router.post("/copy")
async def copy_trader(
    request: CopyTradeRequest,
    user_id: str = "current-user"
):
    """Start copying a trader"""
    copy_id = await social_trading_service.copy_trader(user_id, request.settings)
    return {"copyId": copy_id, "message": "Copy trading started successfully"}

@router.delete("/copy/{trader_id}")
async def stop_copying_trader(
    trader_id: str,
    user_id: str = "current-user"
):
    """Stop copying a trader"""
    success = await social_trading_service.stop_copying_trader(user_id, trader_id)
    return {"success": success, "message": "Copy trading stopped" if success else "Not copying this trader"}

@router.post("/posts", response_model=SocialPost)
async def create_post(
    request: PostRequest,
    user_id: str = "current-user"
):
    """Create a new social post"""
    return await social_trading_service.create_post(user_id, request)

@router.post("/posts/{post_id}/like")
async def like_post(
    post_id: str,
    user_id: str = "current-user"
):
    """Like/unlike a post"""
    is_liked = await social_trading_service.like_post(user_id, post_id)
    return {"liked": is_liked, "message": "Post liked" if is_liked else "Post unliked"}

@router.post("/comments", response_model=Comment)
async def add_comment(
    request: CommentRequest,
    user_id: str = "current-user"
):
    """Add comment to a post"""
    return await social_trading_service.add_comment(user_id, request)

@router.get("/copy-trades", response_model=List[Dict[str, Any]])
async def get_user_copy_trades(
    user_id: str = "current-user"
):
    """Get user's active copy trades"""
    return await social_trading_service.get_user_copy_trades(user_id)

@router.get("/stats")
async def get_social_stats():
    """Get social trading platform statistics"""
    return {
        "totalTraders": len(social_trading_service.traders),
        "totalPosts": len(social_trading_service.posts),
        "totalStrategies": len(social_trading_service.strategies),
        "activeCopyTrades": len([ct for ct in social_trading_service.copy_trades.values() if ct["isActive"]]),
        "totalVolume": 125000000,  # Mock volume
        "avgReturn": 15.7  # Mock average return
    }