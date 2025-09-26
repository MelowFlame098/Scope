import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import and_, or_, desc, func, case
from textblob import TextBlob
import re

from database import get_db, SessionLocal
from db_models import (
    User, ForumPost, Comment, PostVote, CommentVote,
    Asset, Watchlist
)
from schemas import (
    ForumPostResponse, ForumPostCreate, CommentResponse, CommentCreate,
    ForumCategory, VoteType, UserResponse
)
from notification_service import NotificationService, NotificationType, NotificationChannel

logger = logging.getLogger(__name__)

class ForumService:
    def __init__(self):
        self.notification_service = NotificationService()
        
    async def create_post(
        self,
        user_id: str,
        post_data: ForumPostCreate,
        db: Session
    ) -> ForumPostResponse:
        """Create a new forum post."""
        try:
            # Validate user exists
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            
            # Extract mentioned symbols and tags
            symbols = self._extract_symbols(post_data.content)
            tags = self._extract_tags(post_data.content)
            
            # Analyze sentiment
            sentiment_score, sentiment_label = self._analyze_sentiment(post_data.content)
            
            # Create post
            post = ForumPost(
                user_id=user_id,
                title=post_data.title,
                content=post_data.content,
                category=post_data.category,
                symbols=symbols,
                tags=tags,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label
            )
            
            db.add(post)
            db.commit()
            db.refresh(post)
            
            logger.info(f"Forum post created: {post.id} by user {user_id}")
            
            # Send notifications to followers (if implemented)
            await self._notify_followers(user_id, post, db)
            
            return await self._post_to_response(post, user_id, db)
            
        except Exception as e:
            logger.error(f"Error creating forum post: {e}")
            db.rollback()
            raise
    
    async def get_posts(
        self,
        category: Optional[ForumCategory] = None,
        symbol: Optional[str] = None,
        user_id: Optional[str] = None,
        sort_by: str = "recent",
        limit: int = 20,
        offset: int = 0,
        current_user_id: Optional[str] = None,
        db: Session = None
    ) -> List[ForumPostResponse]:
        """Get forum posts with filtering and sorting."""
        try:
            if db is None:
                db = SessionLocal()
            
            # Build query
            query = db.query(ForumPost).options(
                joinedload(ForumPost.user),
                joinedload(ForumPost.comments)
            )
            
            # Apply filters
            if category:
                query = query.filter(ForumPost.category == category)
            
            if symbol:
                query = query.filter(ForumPost.symbols.contains([symbol]))
            
            if user_id:
                query = query.filter(ForumPost.user_id == user_id)
            
            # Apply sorting
            if sort_by == "recent":
                query = query.order_by(desc(ForumPost.created_at))
            elif sort_by == "popular":
                # Sort by vote score (upvotes - downvotes)
                query = query.outerjoin(PostVote).group_by(ForumPost.id).order_by(
                    desc(func.sum(case(
                        (PostVote.vote_type == VoteType.UPVOTE, 1),
                        (PostVote.vote_type == VoteType.DOWNVOTE, -1),
                        else_=0
                    )))
                )
            elif sort_by == "discussed":
                # Sort by number of comments
                query = query.outerjoin(Comment).group_by(ForumPost.id).order_by(
                    desc(func.count(Comment.id))
                )
            
            # Apply pagination
            posts = query.offset(offset).limit(limit).all()
            
            # Convert to response format
            post_responses = []
            for post in posts:
                response = await self._post_to_response(post, current_user_id, db)
                post_responses.append(response)
            
            return post_responses
            
        except Exception as e:
            logger.error(f"Error getting forum posts: {e}")
            return []
        finally:
            if db:
                db.close()
    
    async def get_post(
        self,
        post_id: str,
        current_user_id: Optional[str] = None,
        db: Session = None
    ) -> Optional[ForumPostResponse]:
        """Get a specific forum post with comments."""
        try:
            if db is None:
                db = SessionLocal()
            
            post = db.query(ForumPost).options(
                joinedload(ForumPost.user),
                joinedload(ForumPost.comments).joinedload(Comment.user)
            ).filter(ForumPost.id == post_id).first()
            
            if not post:
                return None
            
            # Increment view count
            post.view_count = (post.view_count or 0) + 1
            db.commit()
            
            return await self._post_to_response(post, current_user_id, db)
            
        except Exception as e:
            logger.error(f"Error getting forum post {post_id}: {e}")
            return None
        finally:
            if db:
                db.close()
    
    async def update_post(
        self,
        post_id: str,
        user_id: str,
        post_data: ForumPostCreate,
        db: Session
    ) -> Optional[ForumPostResponse]:
        """Update a forum post."""
        try:
            post = db.query(ForumPost).filter(
                and_(
                    ForumPost.id == post_id,
                    ForumPost.user_id == user_id
                )
            ).first()
            
            if not post:
                return None
            
            # Update post fields
            post.title = post_data.title
            post.content = post_data.content
            post.category = post_data.category
            post.symbols = self._extract_symbols(post_data.content)
            post.tags = self._extract_tags(post_data.content)
            post.updated_at = datetime.utcnow()
            
            # Re-analyze sentiment
            sentiment_score, sentiment_label = self._analyze_sentiment(post_data.content)
            post.sentiment_score = sentiment_score
            post.sentiment_label = sentiment_label
            
            db.commit()
            db.refresh(post)
            
            return await self._post_to_response(post, user_id, db)
            
        except Exception as e:
            logger.error(f"Error updating forum post {post_id}: {e}")
            db.rollback()
            raise
    
    async def delete_post(
        self,
        post_id: str,
        user_id: str,
        db: Session
    ) -> bool:
        """Delete a forum post."""
        try:
            post = db.query(ForumPost).filter(
                and_(
                    ForumPost.id == post_id,
                    ForumPost.user_id == user_id
                )
            ).first()
            
            if not post:
                return False
            
            # Delete associated votes and comments
            db.query(PostVote).filter(PostVote.post_id == post_id).delete()
            db.query(CommentVote).filter(
                CommentVote.comment_id.in_(
                    db.query(Comment.id).filter(Comment.post_id == post_id)
                )
            ).delete()
            db.query(Comment).filter(Comment.post_id == post_id).delete()
            
            # Delete the post
            db.delete(post)
            db.commit()
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting forum post {post_id}: {e}")
            db.rollback()
            raise
    
    async def vote_post(
        self,
        post_id: str,
        user_id: str,
        vote_type: VoteType,
        db: Session
    ) -> Dict[str, Any]:
        """Vote on a forum post."""
        try:
            # Check if post exists
            post = db.query(ForumPost).filter(ForumPost.id == post_id).first()
            if not post:
                raise ValueError("Post not found")
            
            # Check existing vote
            existing_vote = db.query(PostVote).filter(
                and_(
                    PostVote.post_id == post_id,
                    PostVote.user_id == user_id
                )
            ).first()
            
            if existing_vote:
                if existing_vote.vote_type == vote_type:
                    # Remove vote if same type
                    db.delete(existing_vote)
                else:
                    # Update vote type
                    existing_vote.vote_type = vote_type
                    existing_vote.created_at = datetime.utcnow()
            else:
                # Create new vote
                vote = PostVote(
                    post_id=post_id,
                    user_id=user_id,
                    vote_type=vote_type
                )
                db.add(vote)
            
            db.commit()
            
            # Calculate vote counts
            vote_counts = self._get_post_vote_counts(post_id, db)
            
            # Notify post author (if not voting on own post)
            if post.user_id != user_id:
                await self._notify_vote(post, user_id, vote_type, db)
            
            return vote_counts
            
        except Exception as e:
            logger.error(f"Error voting on post {post_id}: {e}")
            db.rollback()
            raise
    
    async def add_comment(
        self,
        post_id: str,
        user_id: str,
        comment_data: CommentCreate,
        db: Session
    ) -> CommentResponse:
        """Add a comment to a forum post."""
        try:
            # Validate post exists
            post = db.query(ForumPost).filter(ForumPost.id == post_id).first()
            if not post:
                raise ValueError("Post not found")
            
            # Validate user exists
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                raise ValueError("User not found")
            
            # Analyze sentiment
            sentiment_score, sentiment_label = self._analyze_sentiment(comment_data.content)
            
            # Create comment
            comment = Comment(
                post_id=post_id,
                user_id=user_id,
                content=comment_data.content,
                parent_id=comment_data.parent_id,
                sentiment_score=sentiment_score,
                sentiment_label=sentiment_label
            )
            
            db.add(comment)
            db.commit()
            db.refresh(comment)
            
            # Notify post author and parent comment author
            await self._notify_comment(post, comment, db)
            
            return await self._comment_to_response(comment, user_id, db)
            
        except Exception as e:
            logger.error(f"Error adding comment to post {post_id}: {e}")
            db.rollback()
            raise
    
    async def vote_comment(
        self,
        comment_id: str,
        user_id: str,
        vote_type: VoteType,
        db: Session
    ) -> Dict[str, Any]:
        """Vote on a comment."""
        try:
            # Check if comment exists
            comment = db.query(Comment).filter(Comment.id == comment_id).first()
            if not comment:
                raise ValueError("Comment not found")
            
            # Check existing vote
            existing_vote = db.query(CommentVote).filter(
                and_(
                    CommentVote.comment_id == comment_id,
                    CommentVote.user_id == user_id
                )
            ).first()
            
            if existing_vote:
                if existing_vote.vote_type == vote_type:
                    # Remove vote if same type
                    db.delete(existing_vote)
                else:
                    # Update vote type
                    existing_vote.vote_type = vote_type
                    existing_vote.created_at = datetime.utcnow()
            else:
                # Create new vote
                vote = CommentVote(
                    comment_id=comment_id,
                    user_id=user_id,
                    vote_type=vote_type
                )
                db.add(vote)
            
            db.commit()
            
            # Calculate vote counts
            vote_counts = self._get_comment_vote_counts(comment_id, db)
            
            return vote_counts
            
        except Exception as e:
            logger.error(f"Error voting on comment {comment_id}: {e}")
            db.rollback()
            raise
    
    async def get_trending_topics(
        self,
        hours: int = 24,
        limit: int = 10,
        db: Session = None
    ) -> List[Dict[str, Any]]:
        """Get trending topics based on recent activity."""
        try:
            if db is None:
                db = SessionLocal()
            
            # Get recent posts
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Query for trending symbols
            symbol_query = db.query(
                func.unnest(ForumPost.symbols).label('symbol'),
                func.count().label('mention_count')
            ).filter(
                ForumPost.created_at >= cutoff_time
            ).group_by('symbol').order_by(desc('mention_count')).limit(limit)
            
            trending_symbols = symbol_query.all()
            
            # Query for trending tags
            tag_query = db.query(
                func.unnest(ForumPost.tags).label('tag'),
                func.count().label('mention_count')
            ).filter(
                ForumPost.created_at >= cutoff_time
            ).group_by('tag').order_by(desc('mention_count')).limit(limit)
            
            trending_tags = tag_query.all()
            
            return {
                "trending_symbols": [
                    {"symbol": row.symbol, "mentions": row.mention_count}
                    for row in trending_symbols
                ],
                "trending_tags": [
                    {"tag": row.tag, "mentions": row.mention_count}
                    for row in trending_tags
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return {"trending_symbols": [], "trending_tags": []}
        finally:
            if db:
                db.close()
    
    async def get_user_activity(
        self,
        user_id: str,
        activity_type: str = "all",
        limit: int = 20,
        db: Session = None
    ) -> Dict[str, Any]:
        """Get user's forum activity."""
        try:
            if db is None:
                db = SessionLocal()
            
            activity = {}
            
            if activity_type in ["all", "posts"]:
                posts = db.query(ForumPost).filter(
                    ForumPost.user_id == user_id
                ).order_by(desc(ForumPost.created_at)).limit(limit).all()
                
                activity["posts"] = [
                    await self._post_to_response(post, user_id, db)
                    for post in posts
                ]
            
            if activity_type in ["all", "comments"]:
                comments = db.query(Comment).options(
                    joinedload(Comment.post)
                ).filter(
                    Comment.user_id == user_id
                ).order_by(desc(Comment.created_at)).limit(limit).all()
                
                activity["comments"] = [
                    await self._comment_to_response(comment, user_id, db)
                    for comment in comments
                ]
            
            return activity
            
        except Exception as e:
            logger.error(f"Error getting user activity for {user_id}: {e}")
            return {}
        finally:
            if db:
                db.close()
    
    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock/crypto symbols from text."""
        # Look for patterns like $AAPL, $BTC, etc.
        symbol_pattern = r'\$([A-Z]{1,10})'
        symbols = re.findall(symbol_pattern, text.upper())
        return list(set(symbols))  # Remove duplicates
    
    def _extract_tags(self, text: str) -> List[str]:
        """Extract hashtags from text."""
        # Look for patterns like #trading, #crypto, etc.
        tag_pattern = r'#([a-zA-Z0-9_]+)'
        tags = re.findall(tag_pattern, text.lower())
        return list(set(tags))  # Remove duplicates
    
    def _analyze_sentiment(self, text: str) -> tuple[float, str]:
        """Analyze sentiment of text."""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
            
            return round(polarity, 3), label
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0, "neutral"
    
    def _get_post_vote_counts(self, post_id: str, db: Session) -> Dict[str, int]:
        """Get vote counts for a post."""
        votes = db.query(PostVote).filter(PostVote.post_id == post_id).all()
        
        upvotes = sum(1 for vote in votes if vote.vote_type == VoteType.UPVOTE)
        downvotes = sum(1 for vote in votes if vote.vote_type == VoteType.DOWNVOTE)
        
        return {
            "upvotes": upvotes,
            "downvotes": downvotes,
            "score": upvotes - downvotes
        }
    
    def _get_comment_vote_counts(self, comment_id: str, db: Session) -> Dict[str, int]:
        """Get vote counts for a comment."""
        votes = db.query(CommentVote).filter(CommentVote.comment_id == comment_id).all()
        
        upvotes = sum(1 for vote in votes if vote.vote_type == VoteType.UPVOTE)
        downvotes = sum(1 for vote in votes if vote.vote_type == VoteType.DOWNVOTE)
        
        return {
            "upvotes": upvotes,
            "downvotes": downvotes,
            "score": upvotes - downvotes
        }
    
    async def _post_to_response(
        self,
        post: ForumPost,
        current_user_id: Optional[str],
        db: Session
    ) -> ForumPostResponse:
        """Convert ForumPost model to response schema."""
        # Get vote counts
        vote_counts = self._get_post_vote_counts(post.id, db)
        
        # Get user's vote if logged in
        user_vote = None
        if current_user_id:
            vote = db.query(PostVote).filter(
                and_(
                    PostVote.post_id == post.id,
                    PostVote.user_id == current_user_id
                )
            ).first()
            user_vote = vote.vote_type if vote else None
        
        # Get comment count
        comment_count = db.query(Comment).filter(Comment.post_id == post.id).count()
        
        # Convert comments
        comments = []
        if hasattr(post, 'comments') and post.comments:
            for comment in post.comments:
                comment_response = await self._comment_to_response(comment, current_user_id, db)
                comments.append(comment_response)
        
        return ForumPostResponse(
            id=post.id,
            user_id=post.user_id,
            user=UserResponse(
                id=post.user.id,
                username=post.user.username,
                email=post.user.email,
                full_name=post.user.full_name,
                created_at=post.user.created_at
            ) if post.user else None,
            title=post.title,
            content=post.content,
            category=post.category,
            symbols=post.symbols or [],
            tags=post.tags or [],
            sentiment_score=post.sentiment_score,
            sentiment_label=post.sentiment_label,
            view_count=post.view_count or 0,
            upvotes=vote_counts["upvotes"],
            downvotes=vote_counts["downvotes"],
            vote_score=vote_counts["score"],
            user_vote=user_vote,
            comment_count=comment_count,
            comments=comments,
            created_at=post.created_at,
            updated_at=post.updated_at
        )
    
    async def _comment_to_response(
        self,
        comment: Comment,
        current_user_id: Optional[str],
        db: Session
    ) -> CommentResponse:
        """Convert Comment model to response schema."""
        # Get vote counts
        vote_counts = self._get_comment_vote_counts(comment.id, db)
        
        # Get user's vote if logged in
        user_vote = None
        if current_user_id:
            vote = db.query(CommentVote).filter(
                and_(
                    CommentVote.comment_id == comment.id,
                    CommentVote.user_id == current_user_id
                )
            ).first()
            user_vote = vote.vote_type if vote else None
        
        return CommentResponse(
            id=comment.id,
            post_id=comment.post_id,
            user_id=comment.user_id,
            user=UserResponse(
                id=comment.user.id,
                username=comment.user.username,
                email=comment.user.email,
                full_name=comment.user.full_name,
                created_at=comment.user.created_at
            ) if comment.user else None,
            content=comment.content,
            parent_id=comment.parent_id,
            sentiment_score=comment.sentiment_score,
            sentiment_label=comment.sentiment_label,
            upvotes=vote_counts["upvotes"],
            downvotes=vote_counts["downvotes"],
            vote_score=vote_counts["score"],
            user_vote=user_vote,
            created_at=comment.created_at,
            updated_at=comment.updated_at
        )
    
    async def _notify_followers(self, user_id: str, post: ForumPost, db: Session):
        """Notify followers about new post."""
        # TODO: Implement follower system and notifications
        pass
    
    async def _notify_vote(self, post: ForumPost, voter_id: str, vote_type: VoteType, db: Session):
        """Notify post author about vote."""
        try:
            voter = db.query(User).filter(User.id == voter_id).first()
            if not voter:
                return
            
            message = f"{voter.username} {'upvoted' if vote_type == VoteType.UPVOTE else 'downvoted'} your post: {post.title}"
            
            await self.notification_service.send_notification(
                user_id=post.user_id,
                notification_type=NotificationType.FORUM_VOTE,
                message=message,
                channels=[NotificationChannel.WEBSOCKET],
                data={
                    "post_id": post.id,
                    "voter_username": voter.username,
                    "vote_type": vote_type.value
                }
            )
            
        except Exception as e:
            logger.error(f"Error sending vote notification: {e}")
    
    async def _notify_comment(self, post: ForumPost, comment: Comment, db: Session):
        """Notify about new comment."""
        try:
            commenter = db.query(User).filter(User.id == comment.user_id).first()
            if not commenter:
                return
            
            # Notify post author
            if post.user_id != comment.user_id:
                message = f"{commenter.username} commented on your post: {post.title}"
                
                await self.notification_service.send_notification(
                    user_id=post.user_id,
                    notification_type=NotificationType.FORUM_COMMENT,
                    message=message,
                    channels=[NotificationChannel.WEBSOCKET],
                    data={
                        "post_id": post.id,
                        "comment_id": comment.id,
                        "commenter_username": commenter.username
                    }
                )
            
            # Notify parent comment author if it's a reply
            if comment.parent_id:
                parent_comment = db.query(Comment).filter(Comment.id == comment.parent_id).first()
                if parent_comment and parent_comment.user_id != comment.user_id:
                    message = f"{commenter.username} replied to your comment"
                    
                    await self.notification_service.send_notification(
                        user_id=parent_comment.user_id,
                        notification_type=NotificationType.FORUM_COMMENT,
                        message=message,
                        channels=[NotificationChannel.WEBSOCKET],
                        data={
                            "post_id": post.id,
                            "comment_id": comment.id,
                            "parent_comment_id": comment.parent_id,
                            "commenter_username": commenter.username
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error sending comment notification: {e}")

# Global forum service instance
forum_service = ForumService()