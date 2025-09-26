from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Profile information
    avatar_url = Column(String, nullable=True)
    bio = Column(Text, nullable=True)
    location = Column(String, nullable=True)
    website = Column(String, nullable=True)
    
    # Trading preferences
    preferred_assets = Column(JSON, default=list)
    risk_tolerance = Column(String, default="medium")  # low, medium, high
    trading_experience = Column(String, default="beginner")  # beginner, intermediate, advanced
    
    # Relationships
    watchlists = relationship("Watchlist", back_populates="user", cascade="all, delete-orphan")
    forum_posts = relationship("ForumPost", back_populates="author", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="author", cascade="all, delete-orphan")
    portfolios = relationship("Portfolio", back_populates="user", cascade="all, delete-orphan")

class Asset(Base):
    __tablename__ = "assets"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)  # crypto, stock, forex, commodity, index
    exchange = Column(String, nullable=True)
    
    # Current market data
    current_price = Column(Float, nullable=True)
    price_change_24h = Column(Float, nullable=True)
    price_change_percentage_24h = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    volume_24h = Column(Float, nullable=True)
    
    # Additional metadata
    description = Column(Text, nullable=True)
    website = Column(String, nullable=True)
    logo_url = Column(String, nullable=True)
    
    # Technical indicators (cached)
    technical_indicators = Column(JSON, default=dict)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_price_update = Column(DateTime, nullable=True)

class Watchlist(Base):
    __tablename__ = "watchlists"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    symbol = Column(String, nullable=False)
    category = Column(String, nullable=False)
    notes = Column(Text, nullable=True)
    
    # Alert settings
    price_alert_enabled = Column(Boolean, default=False)
    price_alert_above = Column(Float, nullable=True)
    price_alert_below = Column(Float, nullable=True)
    
    added_at = Column(DateTime, default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="watchlists")

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False, default="Default Portfolio")
    description = Column(Text, nullable=True)
    
    # Portfolio settings
    is_public = Column(Boolean, default=False)
    currency = Column(String, default="USD")
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="portfolios")
    holdings = relationship("PortfolioHolding", back_populates="portfolio", cascade="all, delete-orphan")

class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    average_cost = Column(Float, nullable=False)
    
    # Transaction history
    first_purchase_date = Column(DateTime, nullable=True)
    last_transaction_date = Column(DateTime, default=func.now())
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio", back_populates="holdings")

class ForumPost(Base):
    __tablename__ = "forum_posts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String, nullable=False)  # crypto, stocks, forex, general, analysis
    tags = Column(JSON, default=list)
    
    # Author information
    author_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Engagement metrics
    likes = Column(Integer, default=0)
    dislikes = Column(Integer, default=0)
    views = Column(Integer, default=0)
    
    # Post status
    is_pinned = Column(Boolean, default=False)
    is_locked = Column(Boolean, default=False)
    is_deleted = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    author = relationship("User", back_populates="forum_posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    votes = relationship("PostVote", back_populates="post", cascade="all, delete-orphan")

class Comment(Base):
    __tablename__ = "comments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(String, ForeignKey("forum_posts.id"), nullable=False)
    author_id = Column(String, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    
    # Comment threading
    parent_comment_id = Column(String, ForeignKey("comments.id"), nullable=True)
    
    # Engagement
    likes = Column(Integer, default=0)
    
    # Status
    is_deleted = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    post = relationship("ForumPost", back_populates="comments")
    author = relationship("User", back_populates="comments")
    parent_comment = relationship("Comment", remote_side=[id])
    votes = relationship("CommentVote", back_populates="comment", cascade="all, delete-orphan")

class PostVote(Base):
    __tablename__ = "post_votes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(String, ForeignKey("forum_posts.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    vote_type = Column(String, nullable=False)  # 'up' or 'down'
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    post = relationship("ForumPost", back_populates="votes")

class CommentVote(Base):
    __tablename__ = "comment_votes"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    comment_id = Column(String, ForeignKey("comments.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    vote_type = Column(String, nullable=False)  # 'up' or 'down'
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    comment = relationship("Comment", back_populates="votes")

class NewsArticle(Base):
    __tablename__ = "news_articles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    url = Column(String, nullable=False)
    source = Column(String, nullable=False)
    author = Column(String, nullable=True)
    
    # Categorization
    category = Column(String, nullable=True)  # crypto, stocks, forex, general
    tags = Column(JSON, default=list)
    related_symbols = Column(JSON, default=list)
    
    # Sentiment analysis
    sentiment_score = Column(Float, nullable=True)  # -1 to 1
    sentiment_label = Column(String, nullable=True)  # positive, negative, neutral
    
    # Metadata
    image_url = Column(String, nullable=True)
    published_at = Column(DateTime, nullable=True)
    
    # Engagement
    views = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class AIInsight(Base):
    __tablename__ = "ai_insights"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = Column(String, nullable=True)
    insight_type = Column(String, nullable=False)  # analysis, prediction, alert, recommendation
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    
    # AI model information
    model_name = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)  # 0 to 1
    
    # Insight metadata
    timeframe = Column(String, nullable=True)  # short, medium, long
    risk_level = Column(String, nullable=True)  # low, medium, high
    
    # Validation
    is_validated = Column(Boolean, default=False)
    validation_score = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime, nullable=True)

class TradingModel(Base):
    __tablename__ = "trading_models"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String, nullable=False)  # technical, fundamental, ml, sentiment
    asset_class = Column(String, nullable=False)  # crypto, stocks, forex, cross-asset
    
    # Model configuration
    parameters = Column(JSON, default=dict)
    indicators = Column(JSON, default=list)
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class ModelPrediction(Base):
    __tablename__ = "model_predictions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    model_id = Column(String, ForeignKey("trading_models.id"), nullable=False)
    symbol = Column(String, nullable=False)
    
    # Prediction data
    prediction_type = Column(String, nullable=False)  # price, direction, signal
    predicted_value = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    timeframe = Column(String, nullable=False)  # 1h, 4h, 1d, 1w
    
    # Validation
    actual_value = Column(Float, nullable=True)
    is_correct = Column(Boolean, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    target_date = Column(DateTime, nullable=False)
    validated_at = Column(DateTime, nullable=True)

class Order(Base):
    __tablename__ = "orders"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # buy, sell
    quantity = Column(Float, nullable=False)
    order_type = Column(String, nullable=False)  # market, limit, stop, etc.
    limit_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    time_in_force = Column(String, nullable=False)  # day, gtc, etc.
    status = Column(String, nullable=False)  # pending, filled, cancelled, etc.
    filled_quantity = Column(Float, default=0.0)
    average_fill_price = Column(Float, nullable=True)
    commission = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio")
    trades = relationship("Trade", back_populates="order")

class Trade(Base):
    __tablename__ = "trades"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    order_id = Column(String, ForeignKey("orders.id"), nullable=False)
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, nullable=False)
    side = Column(String, nullable=False)  # buy, sell
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    
    timestamp = Column(DateTime, default=func.now())
    
    # Relationships
    order = relationship("Order", back_populates="trades")
    portfolio = relationship("Portfolio")

class Position(Base):
    __tablename__ = "positions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = Column(String, ForeignKey("portfolios.id"), nullable=False)
    symbol = Column(String, nullable=False)
    quantity = Column(Float, nullable=False)
    average_cost = Column(Float, nullable=False)
    market_value = Column(Float, nullable=True)
    unrealized_pnl = Column(Float, nullable=True)
    realized_pnl = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    portfolio = relationship("Portfolio")

class ForumComment(Base):
    __tablename__ = "forum_comments"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    post_id = Column(String, ForeignKey("forum_posts.id"), nullable=False)
    author_id = Column(String, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    parent_comment_id = Column(String, ForeignKey("forum_comments.id"), nullable=True)
    
    # Engagement
    likes = Column(Integer, default=0)
    dislikes = Column(Integer, default=0)
    
    # Status
    is_deleted = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    post = relationship("ForumPost")
    author = relationship("User")
    parent_comment = relationship("ForumComment", remote_side=[id])

class ForumCategory(Base):
    __tablename__ = "forum_categories"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)
    description = Column(Text, nullable=True)
    color = Column(String, nullable=True)  # hex color code
    icon = Column(String, nullable=True)  # icon name or url
    
    # Hierarchy
    parent_category_id = Column(String, ForeignKey("forum_categories.id"), nullable=True)
    
    # Settings
    is_active = Column(Boolean, default=True)
    sort_order = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    parent_category = relationship("ForumCategory", remote_side=[id])

class UserFollow(Base):
    __tablename__ = "user_follows"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    follower_id = Column(String, ForeignKey("users.id"), nullable=False)
    following_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    follower = relationship("User", foreign_keys=[follower_id])
    following = relationship("User", foreign_keys=[following_id])

class Notification(Base):
    __tablename__ = "notifications"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    notification_type = Column(String, nullable=False)
    priority = Column(String, default="medium")
    status = Column(String, default="pending")
    title = Column(String, nullable=False)
    message = Column(Text, nullable=False)
    
    # Delivery channels
    channels = Column(JSON, default=list)
    
    # Metadata
    notification_metadata = Column(JSON, default=dict)
    action_url = Column(String, nullable=True)
    action_text = Column(String, nullable=True)
    image_url = Column(String, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    scheduled_at = Column(DateTime, nullable=True)
    sent_at = Column(DateTime, nullable=True)
    read_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User")

class ChartTemplate(Base):
    __tablename__ = "chart_templates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    chart_type = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    indicators = Column(JSON, default=list)
    annotations = Column(JSON, default=list)
    settings = Column(JSON, default=dict)
    
    is_public = Column(Boolean, default=False)
    is_default = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User")

class TechnicalIndicator(Base):
    __tablename__ = "technical_indicators"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    name = Column(String, nullable=False)
    indicator_type = Column(String, nullable=False)
    parameters = Column(JSON, default=dict)
    formula = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    
    is_custom = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User")

class ChartAnnotation(Base):
    __tablename__ = "chart_annotations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)
    annotation_type = Column(String, nullable=False)
    coordinates = Column(JSON, nullable=False)
    style = Column(JSON, default=dict)
    text = Column(Text, nullable=True)
    
    is_public = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User")

class NotificationPreference(Base):
    __tablename__ = "notification_preferences"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    notification_type = Column(String, nullable=False)
    enabled = Column(Boolean, default=True)
    channels = Column(JSON, default=list)
    frequency = Column(String, default="immediate")
    min_priority = Column(String, default="low")
    
    # Quiet hours
    quiet_hours_start = Column(String, nullable=True)
    quiet_hours_end = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User")

class PriceAlert(Base):
    __tablename__ = "price_alerts"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    symbol = Column(String, nullable=False)
    condition = Column(String, nullable=False)  # above, below, crosses_above, crosses_below, percent_change
    target_value = Column(Float, nullable=False)
    comparison_value = Column(Float, nullable=True)  # for percent change alerts
    
    # Status
    is_active = Column(Boolean, default=True)
    triggered_at = Column(DateTime, nullable=True)
    
    # Metadata
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    expires_at = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User")