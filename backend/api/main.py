from fastapi import APIRouter

from .auth import router as auth_router
from .users import router as users_router
from .portfolios import router as portfolios_router
from .assets import router as assets_router
from .news import router as news_router
from .ml_endpoints import router as ml_router
from .chart_analysis_endpoints import router as chart_analysis_router
from .screen_analysis_endpoints import router as screen_analysis_router
from .indicators_endpoints import router as indicators_router

api_router = APIRouter()

# Include all route modules with their prefixes
api_router.include_router(
    auth_router,
    prefix="/auth",
    tags=["authentication"]
)

api_router.include_router(
    users_router,
    prefix="/users",
    tags=["users"]
)

api_router.include_router(
    portfolios_router,
    prefix="/portfolios",
    tags=["portfolios"]
)

api_router.include_router(
    assets_router,
    prefix="/assets",
    tags=["assets"]
)

api_router.include_router(
    news_router,
    prefix="/news",
    tags=["news"]
)

api_router.include_router(
    ml_router,
    prefix="/ml",
    tags=["machine-learning"]
)

api_router.include_router(
    chart_analysis_router,
    tags=["chart-analysis"]
)

api_router.include_router(
    screen_analysis_router,
    prefix="/screen-analysis",
    tags=["screen-analysis"]
)

api_router.include_router(
    indicators_router,
    tags=["indicators"]
)

# Health check endpoint
@api_router.get("/health", tags=["health"])
def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "message": "FinScope API is running"
    }