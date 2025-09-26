from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import (
    NewsArticleResponse,
    AIInsightResponse,
    PaginatedResponse,
    NewsCategory
)
from ..repositories.news import NewsRepository, AIInsightRepository
from ..services.data_service import DataService
from ..core.security import get_current_user
from ..models import User

router = APIRouter()


@router.get("/", response_model=PaginatedResponse)
def get_news(
    skip: int = Query(0, ge=0, description="Number of articles to skip"),
    limit: int = Query(50, ge=1, le=200, description="Number of articles to return"),
    category: Optional[NewsCategory] = Query(None, description="Filter by news category"),
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols to filter by"),
    search: Optional[str] = Query(None, description="Search term for title or content"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get news articles with optional filtering.
    """
    news_repo = NewsRepository(db)
    
    if search:
        articles = news_repo.search_news(search, skip=skip, limit=limit)
        total = len(articles)  # This is not accurate for pagination, but works for demo
    elif symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        articles = news_repo.get_by_symbols(symbol_list, skip=skip, limit=limit)
        total = len(articles)
    elif category:
        articles = news_repo.get_by_category(category, skip=skip, limit=limit)
        total = news_repo.count(category=category)
    else:
        articles = news_repo.get_recent_news(skip=skip, limit=limit)
        total = news_repo.count()
    
    return PaginatedResponse(
        items=[NewsArticleResponse.from_orm(article) for article in articles],
        total=total,
        page=skip // limit + 1,
        size=limit,
        pages=(total + limit - 1) // limit
    )


@router.get("/trending", response_model=List[NewsArticleResponse])
def get_trending_news(
    limit: int = Query(20, ge=1, le=100, description="Number of trending articles to return"),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get trending news articles based on views and recency.
    """
    news_repo = NewsRepository(db)
    articles = news_repo.get_trending_news(limit=limit, hours=hours)
    return [NewsArticleResponse.from_orm(article) for article in articles]


@router.get("/recent", response_model=List[NewsArticleResponse])
def get_recent_news(
    limit: int = Query(20, ge=1, le=100, description="Number of recent articles to return"),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get recent news articles.
    """
    news_repo = NewsRepository(db)
    articles = news_repo.get_recent_news(limit=limit, hours=hours)
    return [NewsArticleResponse.from_orm(article) for article in articles]


@router.get("/search", response_model=List[NewsArticleResponse])
def search_news(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Number of results to return"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Search news articles by title or content.
    """
    news_repo = NewsRepository(db)
    articles = news_repo.search_news(q, limit=limit)
    return [NewsArticleResponse.from_orm(article) for article in articles]


@router.get("/sentiment", response_model=dict)
def get_sentiment_analysis(
    symbols: Optional[str] = Query(None, description="Comma-separated list of symbols"),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get sentiment analysis for news articles.
    """
    news_repo = NewsRepository(db)
    
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
        sentiment_data = news_repo.get_sentiment_analysis(symbol_list, hours=hours)
    else:
        sentiment_data = news_repo.get_sentiment_analysis(hours=hours)
    
    return sentiment_data


@router.get("/{article_id}", response_model=NewsArticleResponse)
def get_news_article(
    article_id: int,
    db: Session = Depends(get_db)
) -> Any:
    """
    Get a specific news article by ID.
    """
    news_repo = NewsRepository(db)
    article = news_repo.get(article_id)
    
    if not article:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="News article not found"
        )
    
    # Increment view count
    news_repo.increment_views(article_id)
    
    return NewsArticleResponse.from_orm(article)


@router.get("/category/{category}", response_model=List[NewsArticleResponse])
def get_news_by_category(
    category: NewsCategory,
    limit: int = Query(20, ge=1, le=100, description="Number of articles to return"),
    skip: int = Query(0, ge=0, description="Number of articles to skip"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get news articles by category.
    """
    news_repo = NewsRepository(db)
    articles = news_repo.get_by_category(category, skip=skip, limit=limit)
    return [NewsArticleResponse.from_orm(article) for article in articles]


@router.get("/symbol/{symbol}", response_model=List[NewsArticleResponse])
def get_news_by_symbol(
    symbol: str,
    limit: int = Query(20, ge=1, le=100, description="Number of articles to return"),
    skip: int = Query(0, ge=0, description="Number of articles to skip"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get news articles related to a specific symbol.
    """
    news_repo = NewsRepository(db)
    articles = news_repo.get_by_symbols([symbol.upper()], skip=skip, limit=limit)
    return [NewsArticleResponse.from_orm(article) for article in articles]


# AI Insights endpoints
@router.get("/insights/", response_model=List[AIInsightResponse])
def get_ai_insights(
    skip: int = Query(0, ge=0, description="Number of insights to skip"),
    limit: int = Query(20, ge=1, le=100, description="Number of insights to return"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    min_confidence: Optional[float] = Query(None, ge=0, le=1, description="Minimum confidence score"),
    active_only: bool = Query(True, description="Only return active insights"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get AI insights with optional filtering.
    """
    insight_repo = AIInsightRepository(db)
    
    if symbol:
        insights = insight_repo.get_by_symbol(symbol.upper(), skip=skip, limit=limit)
    elif min_confidence is not None:
        insights = insight_repo.get_high_confidence_insights(
            min_confidence=min_confidence, skip=skip, limit=limit
        )
    elif active_only:
        insights = insight_repo.get_active_insights(skip=skip, limit=limit)
    else:
        insights = insight_repo.get_multi(skip=skip, limit=limit)
    
    return [AIInsightResponse.from_orm(insight) for insight in insights]


@router.get("/insights/trending", response_model=List[AIInsightResponse])
def get_trending_insights(
    limit: int = Query(10, ge=1, le=50, description="Number of trending insights to return"),
    hours: int = Query(24, ge=1, le=168, description="Time window in hours"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get trending AI insights based on confidence and recency.
    """
    insight_repo = AIInsightRepository(db)
    insights = insight_repo.get_high_confidence_insights(
        min_confidence=0.7, limit=limit
    )
    return [AIInsightResponse.from_orm(insight) for insight in insights]


@router.get("/insights/symbol/{symbol}", response_model=List[AIInsightResponse])
def get_insights_by_symbol(
    symbol: str,
    limit: int = Query(10, ge=1, le=50, description="Number of insights to return"),
    skip: int = Query(0, ge=0, description="Number of insights to skip"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get AI insights for a specific symbol.
    """
    insight_repo = AIInsightRepository(db)
    insights = insight_repo.get_by_symbol(symbol.upper(), skip=skip, limit=limit)
    return [AIInsightResponse.from_orm(insight) for insight in insights]


@router.get("/insights/{insight_id}", response_model=AIInsightResponse)
def get_ai_insight(
    insight_id: int,
    db: Session = Depends(get_db)
) -> Any:
    """
    Get a specific AI insight by ID.
    """
    insight_repo = AIInsightRepository(db)
    insight = insight_repo.get(insight_id)
    
    if not insight:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="AI insight not found"
        )
    
    return AIInsightResponse.from_orm(insight)


@router.post("/insights/{insight_id}/validate", response_model=dict)
def validate_insight(
    insight_id: int,
    is_accurate: bool = Query(..., description="Whether the insight was accurate"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Validate an AI insight (user feedback).
    """
    insight_repo = AIInsightRepository(db)
    insight = insight_repo.get(insight_id)
    
    if not insight:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="AI insight not found"
        )
    
    # In a real application, you would store user feedback
    # and use it to improve the AI model
    insight_repo.validate_insight(insight_id, is_accurate)
    
    return {
        "message": "Insight validation recorded",
        "insight_id": insight_id,
        "is_accurate": is_accurate
    }


@router.post("/refresh", response_model=dict)
def refresh_news_data(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Manually refresh news data (authenticated users only).
    """
    data_service = DataService(db)
    
    try:
        result = data_service.fetch_news_data()
        return {
            "message": "News data refreshed successfully",
            "articles_created": result.get("articles_created", 0),
            "success": True
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh news data: {str(e)}"
        )