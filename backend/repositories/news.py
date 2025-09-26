from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
from datetime import datetime, timedelta

from .base import BaseRepository
from ..models import NewsArticle, AIInsight
from ..schemas import NewsArticleCreate, NewsArticleUpdate, AIInsightCreate, AIInsightUpdate

class NewsRepository(BaseRepository[NewsArticle, NewsArticleCreate, NewsArticleUpdate]):
    """
    Repository for NewsArticle model with news-specific operations
    """
    
    def __init__(self):
        super().__init__(NewsArticle)
    
    def get_by_url(self, db: Session, *, url: str) -> Optional[NewsArticle]:
        """
        Get news article by URL
        
        Args:
            db: Database session
            url: Article URL
            
        Returns:
            NewsArticle instance or None
        """
        return db.query(NewsArticle).filter(NewsArticle.url == url).first()
    
    def get_by_category(
        self, 
        db: Session, 
        *, 
        category: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[NewsArticle]:
        """
        Get news articles by category
        
        Args:
            db: Database session
            category: News category
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of news articles
        """
        return (
            db.query(NewsArticle)
            .filter(NewsArticle.category == category)
            .order_by(desc(NewsArticle.published_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_by_symbols(
        self, 
        db: Session, 
        *, 
        symbols: List[str], 
        skip: int = 0, 
        limit: int = 100
    ) -> List[NewsArticle]:
        """
        Get news articles related to specific symbols
        
        Args:
            db: Database session
            symbols: List of asset symbols
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of related news articles
        """
        # Convert symbols to uppercase for consistency
        symbols_upper = [symbol.upper() for symbol in symbols]
        
        return (
            db.query(NewsArticle)
            .filter(
                or_(*[
                    NewsArticle.related_symbols.contains([symbol]) 
                    for symbol in symbols_upper
                ])
            )
            .order_by(desc(NewsArticle.published_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_recent_news(
        self, 
        db: Session, 
        *, 
        hours: int = 24, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[NewsArticle]:
        """
        Get recent news articles within specified hours
        
        Args:
            db: Database session
            hours: Number of hours to look back
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of recent news articles
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        return (
            db.query(NewsArticle)
            .filter(NewsArticle.published_at >= cutoff_time)
            .order_by(desc(NewsArticle.published_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def search_news(
        self, 
        db: Session, 
        *, 
        query: str, 
        category: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[NewsArticle]:
        """
        Search news articles by title and content
        
        Args:
            db: Database session
            query: Search query
            category: Optional category filter
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of matching news articles
        """
        search_filter = or_(
            NewsArticle.title.ilike(f"%{query}%"),
            NewsArticle.content.ilike(f"%{query}%"),
            NewsArticle.summary.ilike(f"%{query}%")
        )
        
        base_query = db.query(NewsArticle).filter(search_filter)
        
        if category:
            base_query = base_query.filter(NewsArticle.category == category)
        
        return (
            base_query
            .order_by(desc(NewsArticle.published_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_trending_news(self, db: Session, *, limit: int = 20) -> List[NewsArticle]:
        """
        Get trending news based on views and recency
        
        Args:
            db: Database session
            limit: Maximum number of articles to return
            
        Returns:
            List of trending news articles
        """
        # Get news from last 7 days and sort by views
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        
        return (
            db.query(NewsArticle)
            .filter(NewsArticle.published_at >= cutoff_time)
            .order_by(desc(NewsArticle.views))
            .limit(limit)
            .all()
        )
    
    def get_sentiment_analysis(
        self, 
        db: Session, 
        *, 
        symbol: Optional[str] = None, 
        category: Optional[str] = None,
        hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get sentiment analysis for news articles
        
        Args:
            db: Database session
            symbol: Optional symbol filter
            category: Optional category filter
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with sentiment statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        query = db.query(NewsArticle).filter(
            and_(
                NewsArticle.published_at >= cutoff_time,
                NewsArticle.sentiment_score.isnot(None)
            )
        )
        
        if symbol:
            query = query.filter(NewsArticle.related_symbols.contains([symbol.upper()]))
        
        if category:
            query = query.filter(NewsArticle.category == category)
        
        articles = query.all()
        
        if not articles:
            return {
                'total_articles': 0,
                'average_sentiment': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0
            }
        
        sentiment_scores = [article.sentiment_score for article in articles]
        average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        
        positive_count = len([s for s in sentiment_scores if s > 0.1])
        negative_count = len([s for s in sentiment_scores if s < -0.1])
        neutral_count = len(sentiment_scores) - positive_count - negative_count
        
        return {
            'total_articles': len(articles),
            'average_sentiment': average_sentiment,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'sentiment_distribution': {
                'positive': positive_count / len(articles) * 100,
                'negative': negative_count / len(articles) * 100,
                'neutral': neutral_count / len(articles) * 100
            }
        }
    
    def increment_views(self, db: Session, *, article_id: str) -> Optional[NewsArticle]:
        """
        Increment view count for an article
        
        Args:
            db: Database session
            article_id: Article ID
            
        Returns:
            Updated article or None
        """
        article = self.get(db, article_id)
        if article:
            article.views += 1
            db.commit()
            db.refresh(article)
        return article

class AIInsightRepository(BaseRepository[AIInsight, AIInsightCreate, AIInsightUpdate]):
    """
    Repository for AIInsight model with AI-specific operations
    """
    
    def __init__(self):
        super().__init__(AIInsight)
    
    def get_by_symbol(
        self, 
        db: Session, 
        *, 
        symbol: str, 
        insight_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AIInsight]:
        """
        Get AI insights for a specific symbol
        
        Args:
            db: Database session
            symbol: Asset symbol
            insight_type: Optional insight type filter
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of AI insights
        """
        query = db.query(AIInsight).filter(AIInsight.symbol == symbol.upper())
        
        if insight_type:
            query = query.filter(AIInsight.insight_type == insight_type)
        
        return (
            query
            .order_by(desc(AIInsight.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_active_insights(
        self, 
        db: Session, 
        *, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[AIInsight]:
        """
        Get active (non-expired) AI insights
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of active AI insights
        """
        current_time = datetime.utcnow()
        
        return (
            db.query(AIInsight)
            .filter(
                or_(
                    AIInsight.expires_at.is_(None),
                    AIInsight.expires_at > current_time
                )
            )
            .order_by(desc(AIInsight.created_at))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def get_high_confidence_insights(
        self, 
        db: Session, 
        *, 
        min_confidence: float = 0.8,
        skip: int = 0,
        limit: int = 100
    ) -> List[AIInsight]:
        """
        Get high confidence AI insights
        
        Args:
            db: Database session
            min_confidence: Minimum confidence score
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of high confidence insights
        """
        return (
            db.query(AIInsight)
            .filter(AIInsight.confidence_score >= min_confidence)
            .order_by(desc(AIInsight.confidence_score))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def validate_insight(self, db: Session, *, insight_id: str, validation_score: float) -> Optional[AIInsight]:
        """
        Validate an AI insight with a score
        
        Args:
            db: Database session
            insight_id: Insight ID
            validation_score: Validation score (0-1)
            
        Returns:
            Updated insight or None
        """
        insight = self.get(db, insight_id)
        if insight:
            insight.is_validated = True
            insight.validation_score = validation_score
            db.commit()
            db.refresh(insight)
        return insight

# Create repository instances
news_repository = NewsRepository()
ai_insight_repository = AIInsightRepository()