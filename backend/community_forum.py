"""Community Forum for FinScope - Phase 7 Implementation

Provides comprehensive community features including discussions,
user interactions, content moderation, and social networking.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc

from database import get_db
from db_models import User, ForumPost, ForumComment, ForumCategory, UserFollow
from auth import get_current_user

logger = logging.getLogger(__name__)

class PostType(str, Enum):
    """Types of forum posts"""
    DISCUSSION = "discussion"
    QUESTION = "question"
    ANALYSIS = "analysis"
    NEWS = "news"
    STRATEGY = "strategy"
    ALERT = "alert"
    POLL = "poll"
    EDUCATIONAL = "educational"

class PostStatus(str, Enum):
    """Post status"""
    DRAFT = "draft"
    PUBLISHED = "published"
    HIDDEN = "hidden"
    DELETED = "deleted"
    FLAGGED = "flagged"
    FEATURED = "featured"

class ContentCategory(str, Enum):
    """Content categories"""
    STOCKS = "stocks"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITIES = "commodities"
    OPTIONS = "options"
    FUTURES = "futures"
    BONDS = "bonds"
    ETFS = "etfs"
    GENERAL = "general"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    MARKET_NEWS = "market_news"
    TRADING_STRATEGIES = "trading_strategies"
    PORTFOLIO_MANAGEMENT = "portfolio_management"
    RISK_MANAGEMENT = "risk_management"

class UserRole(str, Enum):
    """User roles in community"""
    MEMBER = "member"
    CONTRIBUTOR = "contributor"
    EXPERT = "expert"
    MODERATOR = "moderator"
    ADMIN = "admin"

class ReactionType(str, Enum):
    """Types of reactions"""
    LIKE = "like"
    DISLIKE = "dislike"
    LOVE = "love"
    LAUGH = "laugh"
    ANGRY = "angry"
    INSIGHTFUL = "insightful"
    HELPFUL = "helpful"

@dataclass
class PostMetrics:
    """Post engagement metrics"""
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    comments: int = 0
    shares: int = 0
    bookmarks: int = 0
    engagement_score: float = 0.0
    trending_score: float = 0.0

@dataclass
class UserStats:
    """User community statistics"""
    posts_count: int = 0
    comments_count: int = 0
    likes_received: int = 0
    followers_count: int = 0
    following_count: int = 0
    reputation_score: int = 0
    badges: List[str] = None
    join_date: datetime = None
    last_active: datetime = None

class PostRequest(BaseModel):
    """Request for creating/updating posts"""
    title: str = Field(..., min_length=5, max_length=200)
    content: str = Field(..., min_length=10, max_length=50000)
    post_type: PostType = PostType.DISCUSSION
    category: ContentCategory = ContentCategory.GENERAL
    tags: List[str] = Field(default=[], max_items=10)
    symbols: List[str] = Field(default=[], max_items=20)
    is_anonymous: bool = False
    allow_comments: bool = True
    status: PostStatus = PostStatus.PUBLISHED
    
class PostResponse(BaseModel):
    """Response for post operations"""
    id: str
    title: str
    content: str
    post_type: PostType
    category: ContentCategory
    author_id: str
    author_name: str
    author_avatar: Optional[str]
    tags: List[str]
    symbols: List[str]
    status: PostStatus
    created_at: datetime
    updated_at: Optional[datetime]
    metrics: PostMetrics
    is_following_author: bool = False
    user_reaction: Optional[ReactionType] = None
    is_bookmarked: bool = False

class CommentRequest(BaseModel):
    """Request for creating comments"""
    content: str = Field(..., min_length=1, max_length=5000)
    parent_comment_id: Optional[str] = None
    is_anonymous: bool = False

class CommentResponse(BaseModel):
    """Response for comment operations"""
    id: str
    content: str
    author_id: str
    author_name: str
    author_avatar: Optional[str]
    post_id: str
    parent_comment_id: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    likes: int = 0
    dislikes: int = 0
    replies_count: int = 0
    user_reaction: Optional[ReactionType] = None
    replies: List['CommentResponse'] = []

class ForumSearchRequest(BaseModel):
    """Request for forum search"""
    query: Optional[str] = None
    category: Optional[ContentCategory] = None
    post_type: Optional[PostType] = None
    tags: List[str] = []
    symbols: List[str] = []
    author_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    sort_by: str = "created_at"  # created_at, updated_at, likes, comments, views
    sort_order: str = "desc"  # asc, desc
    page: int = Field(default=1, ge=1)
    limit: int = Field(default=20, ge=1, le=100)

class ForumSearchResponse(BaseModel):
    """Response for forum search"""
    posts: List[PostResponse]
    total_count: int
    page: int
    limit: int
    total_pages: int
    has_next: bool
    has_prev: bool

class CommunityForum:
    """Advanced community forum system"""
    
    def __init__(self):
        # Content moderation settings
        self.max_posts_per_hour = 10
        self.max_comments_per_hour = 50
        self.spam_detection_enabled = True
        self.auto_moderation_enabled = True
        
        # Reputation system
        self.reputation_weights = {
            "post_created": 5,
            "comment_created": 2,
            "like_received": 1,
            "helpful_reaction": 3,
            "post_featured": 20,
            "expert_badge": 100
        }
        
        # Trending algorithm weights
        self.trending_weights = {
            "recency": 0.3,
            "engagement": 0.4,
            "velocity": 0.3
        }
        
        # Cache for frequently accessed data
        self._trending_posts_cache = {}
        self._user_stats_cache = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def create_post(
        self,
        request: PostRequest,
        user_id: str,
        db: Session
    ) -> PostResponse:
        """Create a new forum post"""
        try:
            # Check rate limits
            if not await self._check_rate_limits(user_id, "post", db):
                raise ValueError("Rate limit exceeded for posts")
            
            # Content moderation
            if self.auto_moderation_enabled:
                moderation_result = await self._moderate_content(
                    request.title + " " + request.content
                )
                if not moderation_result["approved"]:
                    raise ValueError(f"Content moderation failed: {moderation_result['reason']}")
            
            # Generate post ID
            post_id = self._generate_post_id()
            
            # Create post record
            post = ForumPost(
                id=post_id,
                title=request.title,
                content=request.content,
                post_type=request.post_type.value,
                category=request.category.value,
                author_id=user_id,
                tags=request.tags,
                symbols=request.symbols,
                status=request.status.value,
                is_anonymous=request.is_anonymous,
                allow_comments=request.allow_comments,
                created_at=datetime.utcnow()
            )
            
            db.add(post)
            db.commit()
            
            # Update user reputation
            await self._update_user_reputation(
                user_id, "post_created", db
            )
            
            # Clear relevant caches
            self._clear_cache(["trending_posts", f"user_stats_{user_id}"])
            
            # Create response
            response = await self._build_post_response(post, user_id, db)
            
            logger.info(f"Post {post_id} created by user {user_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error creating post: {str(e)}")
            db.rollback()
            raise
    
    async def get_post(
        self,
        post_id: str,
        user_id: Optional[str],
        db: Session
    ) -> PostResponse:
        """Get a specific post"""
        try:
            post = db.query(ForumPost).filter(
                ForumPost.id == post_id,
                ForumPost.status != PostStatus.DELETED.value
            ).first()
            
            if not post:
                raise ValueError(f"Post {post_id} not found")
            
            # Increment view count
            post.views = (post.views or 0) + 1
            db.commit()
            
            # Build response
            response = await self._build_post_response(post, user_id, db)
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting post: {str(e)}")
            raise
    
    async def update_post(
        self,
        post_id: str,
        request: PostRequest,
        user_id: str,
        db: Session
    ) -> PostResponse:
        """Update an existing post"""
        try:
            post = db.query(ForumPost).filter(
                ForumPost.id == post_id,
                ForumPost.author_id == user_id
            ).first()
            
            if not post:
                raise ValueError(f"Post {post_id} not found or not authorized")
            
            # Content moderation for updated content
            if self.auto_moderation_enabled:
                moderation_result = await self._moderate_content(
                    request.title + " " + request.content
                )
                if not moderation_result["approved"]:
                    raise ValueError(f"Content moderation failed: {moderation_result['reason']}")
            
            # Update post fields
            post.title = request.title
            post.content = request.content
            post.post_type = request.post_type.value
            post.category = request.category.value
            post.tags = request.tags
            post.symbols = request.symbols
            post.status = request.status.value
            post.allow_comments = request.allow_comments
            post.updated_at = datetime.utcnow()
            
            db.commit()
            
            # Clear caches
            self._clear_cache(["trending_posts"])
            
            response = await self._build_post_response(post, user_id, db)
            
            logger.info(f"Post {post_id} updated by user {user_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error updating post: {str(e)}")
            db.rollback()
            raise
    
    async def delete_post(
        self,
        post_id: str,
        user_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Delete a post"""
        try:
            post = db.query(ForumPost).filter(
                ForumPost.id == post_id,
                ForumPost.author_id == user_id
            ).first()
            
            if not post:
                raise ValueError(f"Post {post_id} not found or not authorized")
            
            # Soft delete
            post.status = PostStatus.DELETED.value
            post.updated_at = datetime.utcnow()
            
            db.commit()
            
            # Clear caches
            self._clear_cache(["trending_posts", f"user_stats_{user_id}"])
            
            logger.info(f"Post {post_id} deleted by user {user_id}")
            
            return {
                "post_id": post_id,
                "status": "deleted",
                "message": "Post deleted successfully"
            }
            
        except Exception as e:
            logger.error(f"Error deleting post: {str(e)}")
            db.rollback()
            raise
    
    async def add_comment(
        self,
        post_id: str,
        request: CommentRequest,
        user_id: str,
        db: Session
    ) -> CommentResponse:
        """Add a comment to a post"""
        try:
            # Check if post exists and allows comments
            post = db.query(ForumPost).filter(
                ForumPost.id == post_id,
                ForumPost.status == PostStatus.PUBLISHED.value
            ).first()
            
            if not post:
                raise ValueError(f"Post {post_id} not found")
            
            if not post.allow_comments:
                raise ValueError("Comments are not allowed on this post")
            
            # Check rate limits
            if not await self._check_rate_limits(user_id, "comment", db):
                raise ValueError("Rate limit exceeded for comments")
            
            # Content moderation
            if self.auto_moderation_enabled:
                moderation_result = await self._moderate_content(request.content)
                if not moderation_result["approved"]:
                    raise ValueError(f"Content moderation failed: {moderation_result['reason']}")
            
            # Generate comment ID
            comment_id = self._generate_comment_id()
            
            # Create comment record
            comment = ForumComment(
                id=comment_id,
                content=request.content,
                post_id=post_id,
                author_id=user_id,
                parent_comment_id=request.parent_comment_id,
                is_anonymous=request.is_anonymous,
                created_at=datetime.utcnow()
            )
            
            db.add(comment)
            
            # Update post comment count
            post.comments_count = (post.comments_count or 0) + 1
            
            db.commit()
            
            # Update user reputation
            await self._update_user_reputation(
                user_id, "comment_created", db
            )
            
            # Build response
            response = await self._build_comment_response(comment, user_id, db)
            
            logger.info(f"Comment {comment_id} added to post {post_id} by user {user_id}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error adding comment: {str(e)}")
            db.rollback()
            raise
    
    async def react_to_post(
        self,
        post_id: str,
        reaction_type: ReactionType,
        user_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Add or update reaction to a post"""
        try:
            # Check if post exists
            post = db.query(ForumPost).filter(
                ForumPost.id == post_id,
                ForumPost.status == PostStatus.PUBLISHED.value
            ).first()
            
            if not post:
                raise ValueError(f"Post {post_id} not found")
            
            # Handle reaction logic (simplified - would use separate reaction table)
            # For now, just update like/dislike counts
            if reaction_type == ReactionType.LIKE:
                post.likes_count = (post.likes_count or 0) + 1
                
                # Update author reputation
                await self._update_user_reputation(
                    post.author_id, "like_received", db
                )
            
            elif reaction_type == ReactionType.DISLIKE:
                post.dislikes_count = (post.dislikes_count or 0) + 1
            
            elif reaction_type == ReactionType.HELPFUL:
                post.helpful_count = (post.helpful_count or 0) + 1
                
                # Update author reputation
                await self._update_user_reputation(
                    post.author_id, "helpful_reaction", db
                )
            
            db.commit()
            
            # Clear caches
            self._clear_cache(["trending_posts"])
            
            logger.info(f"User {user_id} reacted to post {post_id} with {reaction_type.value}")
            
            return {
                "post_id": post_id,
                "reaction": reaction_type.value,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error reacting to post: {str(e)}")
            db.rollback()
            raise
    
    async def search_posts(
        self,
        request: ForumSearchRequest,
        user_id: Optional[str],
        db: Session
    ) -> ForumSearchResponse:
        """Search forum posts"""
        try:
            # Build query
            query = db.query(ForumPost).filter(
                ForumPost.status == PostStatus.PUBLISHED.value
            )
            
            # Apply filters
            if request.query:
                query = query.filter(
                    or_(
                        ForumPost.title.ilike(f"%{request.query}%"),
                        ForumPost.content.ilike(f"%{request.query}%")
                    )
                )
            
            if request.category:
                query = query.filter(ForumPost.category == request.category.value)
            
            if request.post_type:
                query = query.filter(ForumPost.post_type == request.post_type.value)
            
            if request.author_id:
                query = query.filter(ForumPost.author_id == request.author_id)
            
            if request.tags:
                for tag in request.tags:
                    query = query.filter(ForumPost.tags.contains([tag]))
            
            if request.symbols:
                for symbol in request.symbols:
                    query = query.filter(ForumPost.symbols.contains([symbol]))
            
            if request.date_from:
                query = query.filter(ForumPost.created_at >= request.date_from)
            
            if request.date_to:
                query = query.filter(ForumPost.created_at <= request.date_to)
            
            # Apply sorting
            if request.sort_by == "created_at":
                sort_column = ForumPost.created_at
            elif request.sort_by == "updated_at":
                sort_column = ForumPost.updated_at
            elif request.sort_by == "likes":
                sort_column = ForumPost.likes_count
            elif request.sort_by == "comments":
                sort_column = ForumPost.comments_count
            elif request.sort_by == "views":
                sort_column = ForumPost.views
            else:
                sort_column = ForumPost.created_at
            
            if request.sort_order == "desc":
                query = query.order_by(desc(sort_column))
            else:
                query = query.order_by(asc(sort_column))
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination
            offset = (request.page - 1) * request.limit
            posts = query.offset(offset).limit(request.limit).all()
            
            # Build responses
            post_responses = []
            for post in posts:
                response = await self._build_post_response(post, user_id, db)
                post_responses.append(response)
            
            # Calculate pagination info
            total_pages = (total_count + request.limit - 1) // request.limit
            has_next = request.page < total_pages
            has_prev = request.page > 1
            
            return ForumSearchResponse(
                posts=post_responses,
                total_count=total_count,
                page=request.page,
                limit=request.limit,
                total_pages=total_pages,
                has_next=has_next,
                has_prev=has_prev
            )
            
        except Exception as e:
            logger.error(f"Error searching posts: {str(e)}")
            raise
    
    async def get_trending_posts(
        self,
        category: Optional[ContentCategory],
        limit: int,
        user_id: Optional[str],
        db: Session
    ) -> List[PostResponse]:
        """Get trending posts"""
        try:
            # Check cache
            cache_key = f"trending_{category.value if category else 'all'}_{limit}"
            if cache_key in self._trending_posts_cache:
                cached_data = self._trending_posts_cache[cache_key]
                if datetime.utcnow() - cached_data["timestamp"] < timedelta(seconds=self._cache_ttl):
                    return cached_data["posts"]
            
            # Calculate trending scores
            posts = await self._calculate_trending_posts(category, limit, db)
            
            # Build responses
            post_responses = []
            for post in posts:
                response = await self._build_post_response(post, user_id, db)
                post_responses.append(response)
            
            # Cache results
            self._trending_posts_cache[cache_key] = {
                "posts": post_responses,
                "timestamp": datetime.utcnow()
            }
            
            return post_responses
            
        except Exception as e:
            logger.error(f"Error getting trending posts: {str(e)}")
            return []
    
    async def get_user_stats(
        self,
        user_id: str,
        db: Session
    ) -> UserStats:
        """Get user community statistics"""
        try:
            # Check cache
            cache_key = f"user_stats_{user_id}"
            if cache_key in self._user_stats_cache:
                cached_data = self._user_stats_cache[cache_key]
                if datetime.utcnow() - cached_data["timestamp"] < timedelta(seconds=self._cache_ttl):
                    return cached_data["stats"]
            
            # Get user
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError(f"User {user_id} not found")
            
            # Calculate statistics
            posts_count = db.query(ForumPost).filter(
                ForumPost.author_id == user_id,
                ForumPost.status != PostStatus.DELETED.value
            ).count()
            
            comments_count = db.query(ForumComment).filter(
                ForumComment.author_id == user_id
            ).count()
            
            # Get total likes received
            likes_received = db.query(ForumPost).filter(
                ForumPost.author_id == user_id
            ).with_entities(
                db.func.sum(ForumPost.likes_count)
            ).scalar() or 0
            
            # Get follower counts (would need UserFollow table)
            followers_count = 0  # TODO: Implement follower system
            following_count = 0  # TODO: Implement following system
            
            # Calculate reputation score
            reputation_score = (
                posts_count * self.reputation_weights["post_created"] +
                comments_count * self.reputation_weights["comment_created"] +
                likes_received * self.reputation_weights["like_received"]
            )
            
            # Determine badges
            badges = self._calculate_user_badges(
                posts_count, comments_count, likes_received, reputation_score
            )
            
            stats = UserStats(
                posts_count=posts_count,
                comments_count=comments_count,
                likes_received=likes_received,
                followers_count=followers_count,
                following_count=following_count,
                reputation_score=reputation_score,
                badges=badges,
                join_date=user.created_at,
                last_active=user.last_login
            )
            
            # Cache results
            self._user_stats_cache[cache_key] = {
                "stats": stats,
                "timestamp": datetime.utcnow()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting user stats: {str(e)}")
            return UserStats()
    
    async def _build_post_response(
        self,
        post: ForumPost,
        user_id: Optional[str],
        db: Session
    ) -> PostResponse:
        """Build post response with user-specific data"""
        try:
            # Get author info
            author = db.query(User).filter(User.id == post.author_id).first()
            author_name = "Anonymous" if post.is_anonymous else (author.username if author else "Unknown")
            author_avatar = None if post.is_anonymous else (author.avatar_url if author else None)
            
            # Calculate metrics
            metrics = PostMetrics(
                views=post.views or 0,
                likes=post.likes_count or 0,
                dislikes=post.dislikes_count or 0,
                comments=post.comments_count or 0,
                shares=post.shares_count or 0,
                bookmarks=post.bookmarks_count or 0
            )
            
            # User-specific data
            is_following_author = False  # TODO: Implement following system
            user_reaction = None  # TODO: Implement user reactions
            is_bookmarked = False  # TODO: Implement bookmarking
            
            return PostResponse(
                id=post.id,
                title=post.title,
                content=post.content,
                post_type=PostType(post.post_type),
                category=ContentCategory(post.category),
                author_id=post.author_id,
                author_name=author_name,
                author_avatar=author_avatar,
                tags=post.tags or [],
                symbols=post.symbols or [],
                status=PostStatus(post.status),
                created_at=post.created_at,
                updated_at=post.updated_at,
                metrics=metrics,
                is_following_author=is_following_author,
                user_reaction=user_reaction,
                is_bookmarked=is_bookmarked
            )
            
        except Exception as e:
            logger.error(f"Error building post response: {str(e)}")
            raise
    
    async def _build_comment_response(
        self,
        comment: ForumComment,
        user_id: Optional[str],
        db: Session
    ) -> CommentResponse:
        """Build comment response with user-specific data"""
        try:
            # Get author info
            author = db.query(User).filter(User.id == comment.author_id).first()
            author_name = "Anonymous" if comment.is_anonymous else (author.username if author else "Unknown")
            author_avatar = None if comment.is_anonymous else (author.avatar_url if author else None)
            
            # Get replies count
            replies_count = db.query(ForumComment).filter(
                ForumComment.parent_comment_id == comment.id
            ).count()
            
            return CommentResponse(
                id=comment.id,
                content=comment.content,
                author_id=comment.author_id,
                author_name=author_name,
                author_avatar=author_avatar,
                post_id=comment.post_id,
                parent_comment_id=comment.parent_comment_id,
                created_at=comment.created_at,
                updated_at=comment.updated_at,
                likes=comment.likes_count or 0,
                dislikes=comment.dislikes_count or 0,
                replies_count=replies_count
            )
            
        except Exception as e:
            logger.error(f"Error building comment response: {str(e)}")
            raise
    
    async def _check_rate_limits(
        self,
        user_id: str,
        action_type: str,
        db: Session
    ) -> bool:
        """Check if user has exceeded rate limits"""
        try:
            now = datetime.utcnow()
            hour_ago = now - timedelta(hours=1)
            
            if action_type == "post":
                recent_posts = db.query(ForumPost).filter(
                    ForumPost.author_id == user_id,
                    ForumPost.created_at >= hour_ago
                ).count()
                
                return recent_posts < self.max_posts_per_hour
            
            elif action_type == "comment":
                recent_comments = db.query(ForumComment).filter(
                    ForumComment.author_id == user_id,
                    ForumComment.created_at >= hour_ago
                ).count()
                
                return recent_comments < self.max_comments_per_hour
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limits: {str(e)}")
            return True  # Allow on error
    
    async def _moderate_content(self, content: str) -> Dict[str, Any]:
        """Moderate content for spam, inappropriate language, etc."""
        try:
            # Simple content moderation (would integrate with AI moderation service)
            
            # Check for spam patterns
            spam_keywords = ["spam", "scam", "get rich quick", "guaranteed profit"]
            content_lower = content.lower()
            
            for keyword in spam_keywords:
                if keyword in content_lower:
                    return {
                        "approved": False,
                        "reason": f"Content contains spam keyword: {keyword}"
                    }
            
            # Check content length
            if len(content) < 5:
                return {
                    "approved": False,
                    "reason": "Content too short"
                }
            
            # Check for excessive caps
            caps_ratio = sum(1 for c in content if c.isupper()) / len(content)
            if caps_ratio > 0.7:
                return {
                    "approved": False,
                    "reason": "Excessive use of capital letters"
                }
            
            return {"approved": True, "reason": "Content approved"}
            
        except Exception as e:
            logger.error(f"Error moderating content: {str(e)}")
            return {"approved": True, "reason": "Moderation error - approved by default"}
    
    async def _update_user_reputation(
        self,
        user_id: str,
        action: str,
        db: Session
    ):
        """Update user reputation score"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                points = self.reputation_weights.get(action, 0)
                user.reputation_score = (user.reputation_score or 0) + points
                db.commit()
                
                # Clear user stats cache
                cache_key = f"user_stats_{user_id}"
                if cache_key in self._user_stats_cache:
                    del self._user_stats_cache[cache_key]
            
        except Exception as e:
            logger.error(f"Error updating user reputation: {str(e)}")
    
    async def _calculate_trending_posts(
        self,
        category: Optional[ContentCategory],
        limit: int,
        db: Session
    ) -> List[ForumPost]:
        """Calculate trending posts using engagement metrics"""
        try:
            # Get recent posts (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            
            query = db.query(ForumPost).filter(
                ForumPost.status == PostStatus.PUBLISHED.value,
                ForumPost.created_at >= week_ago
            )
            
            if category:
                query = query.filter(ForumPost.category == category.value)
            
            posts = query.all()
            
            # Calculate trending scores
            scored_posts = []
            for post in posts:
                score = self._calculate_trending_score(post)
                scored_posts.append((post, score))
            
            # Sort by score and return top posts
            scored_posts.sort(key=lambda x: x[1], reverse=True)
            return [post for post, score in scored_posts[:limit]]
            
        except Exception as e:
            logger.error(f"Error calculating trending posts: {str(e)}")
            return []
    
    def _calculate_trending_score(self, post: ForumPost) -> float:
        """Calculate trending score for a post"""
        try:
            now = datetime.utcnow()
            age_hours = (now - post.created_at).total_seconds() / 3600
            
            # Recency score (decays over time)
            recency_score = max(0, 100 - (age_hours / 24) * 10)
            
            # Engagement score
            likes = post.likes_count or 0
            comments = post.comments_count or 0
            views = post.views or 0
            
            engagement_score = (
                likes * 3 +
                comments * 5 +
                views * 0.1
            )
            
            # Velocity score (engagement per hour)
            velocity_score = engagement_score / max(age_hours, 1)
            
            # Combined trending score
            trending_score = (
                recency_score * self.trending_weights["recency"] +
                engagement_score * self.trending_weights["engagement"] +
                velocity_score * self.trending_weights["velocity"]
            )
            
            return trending_score
            
        except Exception as e:
            logger.error(f"Error calculating trending score: {str(e)}")
            return 0
    
    def _calculate_user_badges(
        self,
        posts_count: int,
        comments_count: int,
        likes_received: int,
        reputation_score: int
    ) -> List[str]:
        """Calculate user badges based on activity"""
        badges = []
        
        # Post-based badges
        if posts_count >= 100:
            badges.append("Prolific Poster")
        elif posts_count >= 50:
            badges.append("Active Poster")
        elif posts_count >= 10:
            badges.append("Regular Poster")
        elif posts_count >= 1:
            badges.append("First Post")
        
        # Comment-based badges
        if comments_count >= 500:
            badges.append("Discussion Master")
        elif comments_count >= 100:
            badges.append("Active Commenter")
        
        # Engagement badges
        if likes_received >= 1000:
            badges.append("Community Favorite")
        elif likes_received >= 100:
            badges.append("Well Liked")
        
        # Reputation badges
        if reputation_score >= 1000:
            badges.append("Expert Contributor")
        elif reputation_score >= 500:
            badges.append("Trusted Member")
        elif reputation_score >= 100:
            badges.append("Active Member")
        
        return badges
    
    def _clear_cache(self, cache_keys: List[str]):
        """Clear specific cache entries"""
        for key in cache_keys:
            if key.startswith("trending_posts"):
                keys_to_remove = [k for k in self._trending_posts_cache.keys() if k.startswith("trending_")]
                for k in keys_to_remove:
                    del self._trending_posts_cache[k]
            elif key in self._user_stats_cache:
                del self._user_stats_cache[key]
    
    def _generate_post_id(self) -> str:
        """Generate unique post ID"""
        import uuid
        return f"POST_{uuid.uuid4().hex[:8].upper()}"
    
    def _generate_comment_id(self) -> str:
        """Generate unique comment ID"""
        import uuid
        return f"COMM_{uuid.uuid4().hex[:8].upper()}"

# Global community forum instance
community_forum = CommunityForum()

def get_community_forum() -> CommunityForum:
    """Get community forum instance"""
    return community_forum