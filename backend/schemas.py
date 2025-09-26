from pydantic import BaseModel, EmailStr, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Enums
class RiskTolerance(str, Enum):
    conservative = "conservative"
    moderate = "moderate"
    aggressive = "aggressive"

class AssetCategory(str, Enum):
    CRYPTO = "crypto"
    STOCK = "stock"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    stock = "stock"
    crypto = "crypto"
    etf = "etf"
    bond = "bond"
    commodity = "commodity"

class ForumCategory(str, Enum):
    CRYPTO = "crypto"
    STOCKS = "stocks"
    FOREX = "forex"
    GENERAL = "general"
    ANALYSIS = "analysis"

class NewsCategory(str, Enum):
    general = "general"
    crypto = "crypto"
    stocks = "stocks"
    market = "market"
    technology = "technology"
    regulation = "regulation"

class TransactionType(str, Enum):
    buy = "buy"
    sell = "sell"

class InsightType(str, Enum):
    ANALYSIS = "analysis"
    PREDICTION = "prediction"
    ALERT = "alert"
    RECOMMENDATION = "recommendation"

class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class VoteType(str, Enum):
    UP = "up"
    DOWN = "down"

class Timeframe(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

# Base schemas
class BaseSchema(BaseModel):
    class Config:
        from_attributes = True

# User schemas
class UserBase(BaseSchema):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    risk_tolerance: Optional[RiskTolerance] = RiskTolerance.moderate
    investment_goals: Optional[List[str]] = []

class UserCreate(UserBase):
    password: str
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v
    
    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if not v.isalnum():
            raise ValueError('Username must contain only alphanumeric characters')
        return v

class UserUpdate(BaseSchema):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    preferred_assets: Optional[List[str]] = None
    risk_tolerance: Optional[RiskLevel] = None
    trading_experience: Optional[str] = None
    investment_goals: Optional[List[str]] = None
    is_active: Optional[bool] = None

class UserInDB(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

class User(UserInDB):
    pass

class UserResponse(UserBase):
    id: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    avatar_url: Optional[str] = None
    bio: Optional[str] = None
    location: Optional[str] = None
    website: Optional[str] = None
    
    class Config:
        from_attributes = True

# Authentication schemas
class Token(BaseSchema):
    access_token: str
    refresh_token: str
    token_type: str
    user: User

class TokenRefresh(BaseSchema):
    refresh_token: str

class LoginRequest(BaseSchema):
    email: EmailStr
    password: str

class PasswordChange(BaseSchema):
    current_password: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class PasswordReset(BaseSchema):
    email: EmailStr

class PasswordResetConfirm(BaseSchema):
    token: str
    new_password: str
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v

class TokenData(BaseModel):
    username: Optional[str] = None

# Asset schemas
class AssetBase(BaseSchema):
    symbol: str
    name: str
    category: AssetCategory
    current_price: Optional[float] = None
    price_change_24h: Optional[float] = None
    price_change_percentage_24h: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    logo_url: Optional[str] = None

class AssetCreate(AssetBase):
    pass

class AssetUpdate(BaseSchema):
    name: Optional[str] = None
    current_price: Optional[float] = None
    price_change_24h: Optional[float] = None
    price_change_percentage_24h: Optional[float] = None
    market_cap: Optional[float] = None
    volume_24h: Optional[float] = None
    logo_url: Optional[str] = None
    last_price_update: Optional[datetime] = None

class AssetInDB(AssetBase):
    id: int
    last_price_update: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

class Asset(AssetInDB):
    pass

class AssetResponse(AssetBase):
    id: int
    last_price_update: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Market data schemas
class MarketDataPoint(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

class ChartData(BaseModel):
    symbol: str
    timeframe: str
    data: List[MarketDataPoint]
    indicators: Optional[Dict[str, List[float]]] = None

class MarketOverview(BaseModel):
    total_market_cap: float
    total_volume_24h: float
    btc_dominance: Optional[float] = None
    fear_greed_index: Optional[int] = None
    active_cryptocurrencies: Optional[int] = None
    market_cap_change_24h: Optional[float] = None

class MarketDataResponse(BaseModel):
    overview: MarketOverview
    top_gainers: List[AssetResponse]
    top_losers: List[AssetResponse]
    trending: List[AssetResponse]
    timestamp: datetime

# Watchlist schemas
class WatchlistBase(BaseSchema):
    name: str
    description: Optional[str] = None

class WatchlistCreate(WatchlistBase):
    asset_ids: Optional[List[int]] = []

class WatchlistUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None

class WatchlistInDB(WatchlistBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime

class Watchlist(WatchlistInDB):
    assets: List[Asset] = []

class WatchlistResponse(WatchlistBase):
    id: int
    user_id: int
    assets: List[AssetResponse] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Portfolio schemas
class PortfolioBase(BaseSchema):
    name: str
    description: Optional[str] = None
    is_public: bool = False

class PortfolioCreate(PortfolioBase):
    pass

class PortfolioUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None

class PortfolioInDB(PortfolioBase):
    id: int
    user_id: int
    total_value: float
    total_cost: float
    total_return: float
    total_return_percentage: float
    created_at: datetime
    updated_at: datetime

class Portfolio(PortfolioInDB):
    holdings: List['PortfolioHolding'] = []

class PortfolioResponse(PortfolioBase):
    id: int
    user_id: int
    total_value: float
    total_cost: float
    total_return: float
    total_return_percentage: float
    holdings: List['PortfolioHoldingResponse'] = []
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

# Portfolio Holding schemas
class PortfolioHoldingBase(BaseSchema):
    quantity: float
    average_cost: float

class PortfolioHoldingCreate(PortfolioHoldingBase):
    asset_id: int
    portfolio_id: int

class PortfolioHoldingUpdate(BaseSchema):
    quantity: Optional[float] = None
    average_cost: Optional[float] = None

class PortfolioHoldingInDB(PortfolioHoldingBase):
    id: int
    portfolio_id: int
    asset_id: int
    current_value: float
    total_return: float
    total_return_percentage: float
    created_at: datetime
    updated_at: datetime

class PortfolioHolding(PortfolioHoldingInDB):
    asset: Asset

class PortfolioHoldingResponse(PortfolioHoldingBase):
    id: int
    portfolio_id: int
    asset_id: int
    current_value: float
    total_return: float
    total_return_percentage: float
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Transaction schemas
class TransactionCreate(BaseSchema):
    portfolio_id: int
    asset_id: int
    transaction_type: TransactionType
    quantity: float
    price: float
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('Quantity must be positive')
        return v
    
    @validator('price')
    def validate_price(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v

# News schemas
class NewsArticleBase(BaseSchema):
    title: str
    content: Optional[str] = None
    url: str
    source: str
    author: Optional[str] = None
    category: NewsCategory = NewsCategory.general
    related_symbols: Optional[List[str]] = []
    image_url: Optional[str] = None
    published_at: Optional[datetime] = None

class NewsArticleCreate(NewsArticleBase):
    pass

class NewsArticleUpdate(BaseSchema):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[NewsCategory] = None
    related_symbols: Optional[List[str]] = None
    image_url: Optional[str] = None
    views: Optional[int] = None

class NewsArticleInDB(NewsArticleBase):
    id: int
    views: int
    sentiment_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

class NewsArticle(NewsArticleInDB):
    pass

# Forum schemas
class ForumPostBase(BaseSchema):
    title: str
    content: str
    category: Optional[str] = None
    tags: Optional[List[str]] = []

class ForumPostCreate(ForumPostBase):
    pass

class ForumPostUpdate(BaseSchema):
    title: Optional[str] = None
    content: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None

class ForumPostInDB(ForumPostBase):
    id: int
    user_id: int
    upvotes: int
    downvotes: int
    comment_count: int
    created_at: datetime
    updated_at: datetime

class ForumPost(ForumPostInDB):
    user: User
    comments: List['Comment'] = []

class ForumPostResponse(ForumPostBase):
    id: int
    user_id: int
    upvotes: int
    downvotes: int
    comment_count: int
    created_at: datetime
    updated_at: datetime
    user: UserResponse
    comments: List['CommentResponse'] = []

    class Config:
        from_attributes = True

# Comment schemas
class CommentBase(BaseSchema):
    content: str

class CommentCreate(CommentBase):
    post_id: int
    parent_id: Optional[int] = None

class CommentUpdate(BaseSchema):
    content: Optional[str] = None

class CommentInDB(CommentBase):
    id: int
    post_id: int
    user_id: int
    parent_id: Optional[int] = None
    upvotes: int
    downvotes: int
    created_at: datetime
    updated_at: datetime

class Comment(CommentInDB):
    user: User
    replies: List['Comment'] = []

class CommentResponse(CommentBase):
    id: int
    post_id: int
    user_id: int
    parent_id: Optional[int] = None
    upvotes: int
    downvotes: int
    created_at: datetime
    updated_at: datetime
    user: UserResponse
    replies: List['CommentResponse'] = []

    class Config:
        from_attributes = True

class VoteCreate(BaseModel):
    vote_type: str  # 'up' or 'down'
    
    @validator('vote_type')
    def validate_vote_type(cls, v):
        if v not in ['up', 'down']:
            raise ValueError('Vote type must be "up" or "down"')
        return v

# News schemas
class NewsBase(BaseModel):
    title: str
    content: Optional[str] = None
    summary: Optional[str] = None
    url: str
    source: str
    author: Optional[str] = None

class NewsCreate(NewsBase):
    category: Optional[str] = None
    tags: Optional[List[str]] = []
    related_symbols: Optional[List[str]] = []
    published_at: Optional[datetime] = None

class NewsResponse(NewsBase):
    id: str
    category: Optional[str] = None
    tags: List[str] = []
    related_symbols: List[str] = []
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    image_url: Optional[str] = None
    published_at: Optional[datetime] = None
    views: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# AI Insight schemas
class AIInsightBase(BaseModel):
    symbol: Optional[str] = None
    insight_type: InsightType
    title: str
    content: str

class AIInsightCreate(AIInsightBase):
    model_name: Optional[str] = None
    confidence_score: Optional[float] = None
    timeframe: Optional[Timeframe] = None
    risk_level: Optional[RiskLevel] = None
    expires_at: Optional[datetime] = None
    
    model_config = {"protected_namespaces": ()}

class AIInsightResponse(AIInsightBase):
    id: str
    model_name: Optional[str] = None
    confidence_score: Optional[float] = None
    timeframe: Optional[Timeframe] = None
    risk_level: Optional[RiskLevel] = None
    is_validated: bool
    validation_score: Optional[float] = None
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    model_config = {"protected_namespaces": (), "from_attributes": True}

# Trading Model schemas
class TradingModelBase(BaseModel):
    name: str
    description: Optional[str] = None
    category: str  # technical, fundamental, ml, sentiment
    asset_class: str  # crypto, stocks, forex, cross-asset

class TradingModelCreate(TradingModelBase):
    parameters: Optional[Dict[str, Any]] = {}
    indicators: Optional[List[str]] = []
    is_public: Optional[bool] = False

class TradingModelResponse(TradingModelBase):
    id: str
    parameters: Dict[str, Any]
    indicators: List[str]
    accuracy: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    is_active: bool
    is_public: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

class ModelPredictionBase(BaseModel):
    symbol: str
    prediction_type: str  # price, direction, signal
    predicted_value: Optional[float] = None
    confidence: Optional[float] = None
    timeframe: str  # 1h, 4h, 1d, 1w
    target_date: datetime

class ModelPredictionCreate(ModelPredictionBase):
    model_id: str
    
    model_config = {"protected_namespaces": ()}

class ModelPredictionResponse(ModelPredictionBase):
    id: str
    model_id: str
    actual_value: Optional[float] = None
    is_correct: Optional[bool] = None
    created_at: datetime
    validated_at: Optional[datetime] = None
    
    model_config = {"protected_namespaces": (), "from_attributes": True}

# Analysis request schemas
class AnalysisRequest(BaseModel):
    symbol: str
    analysis_type: str = "comprehensive"  # technical, fundamental, sentiment, comprehensive
    timeframe: Optional[str] = "1d"
    indicators: Optional[List[str]] = []

class ExplanationRequest(BaseModel):
    data: Dict[str, Any]
    context: Optional[str] = ""
    explanation_type: Optional[str] = "simple"  # simple, detailed, technical

# WebSocket message schemas
class WebSocketMessage(BaseModel):
    type: str
    data: Dict[str, Any]
    timestamp: datetime

class MarketUpdateMessage(WebSocketMessage):
    type: str = "market_update"

class NewsUpdateMessage(WebSocketMessage):
    type: str = "news_update"

class AlertMessage(WebSocketMessage):
    type: str = "alert"

# Search schemas
class SearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    limit: Optional[int] = 20
    offset: Optional[int] = 0

class SearchResponse(BaseModel):
    results: List[Union[AssetResponse, NewsResponse, ForumPostResponse]]
    total: int
    query: str
    category: Optional[str] = None

# Error schemas
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None

# Success schemas
class SuccessResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None

# Additional Response schemas
class MessageResponse(BaseSchema):
    message: str
    success: bool = True

class PaginatedResponse(BaseSchema):
    items: List[Any]
    total: int
    page: int
    size: int
    pages: int

class PortfolioPerformance(BaseSchema):
    total_value: float
    total_cost: float
    total_return: float
    total_return_percentage: float
    daily_change: float
    daily_change_percentage: float
    asset_allocation: Dict[str, float]
    top_performers: List[Dict[str, Any]]
    top_losers: List[Dict[str, Any]]

class DataRefreshResult(BaseSchema):
    success: bool
    crypto_updated: int
    stocks_updated: int
    news_created: int
    errors: List[str]
    last_refresh: datetime

# Trading Model schemas (updated)
class TradingModelInDB(TradingModelBase):
    id: int
    user_id: int
    is_active: bool
    accuracy_score: Optional[float] = None
    created_at: datetime
    updated_at: datetime

class TradingModel(TradingModelInDB):
    user: User
    predictions: List['ModelPrediction'] = []

class TradingModelUpdate(BaseSchema):
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    target_symbols: Optional[List[str]] = None
    is_active: Optional[bool] = None
    accuracy_score: Optional[float] = None

# Model Prediction schemas (updated)
class ModelPredictionInDB(ModelPredictionBase):
    id: int
    model_id: int
    actual_value: Optional[float] = None
    is_validated: bool
    created_at: datetime
    updated_at: datetime
    
    model_config = {"protected_namespaces": ()}

class ModelPrediction(ModelPredictionInDB):
    model: TradingModel

class ModelPredictionUpdate(BaseSchema):
    actual_value: Optional[float] = None
    is_validated: Optional[bool] = None

# Update forward references
Portfolio.model_rebuild()
ForumPost.model_rebuild()
Comment.model_rebuild()
TradingModel.model_rebuild()