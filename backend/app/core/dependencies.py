"""Dependency injection for FinScope application.

This module provides dependency injection functions for:
- Database sessions
- Authentication
- Feature access
- Service instances
- Configuration
"""

from typing import Generator, Optional, Any
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import logging

# Import from existing modules
from database import get_db as _get_db
from auth import get_current_user as _get_current_user
from app.config.settings import get_settings, Settings
from app.core.feature_registry import registry
from app.core.exceptions import (
    AuthenticationException,
    AuthorizationException,
    FeatureUnavailableException
)

logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


# Database dependency
def get_db() -> Generator[Session, None, None]:
    """Get database session.
    
    Yields:
        Database session
    """
    yield from _get_db()


# Settings dependency
def get_settings_dependency() -> Settings:
    """Get application settings.
    
    Returns:
        Application settings instance
    """
    return get_settings()


# Authentication dependencies
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    """Get current authenticated user.
    
    Args:
        credentials: HTTP authorization credentials
        db: Database session
        
    Returns:
        Current user instance
        
    Raises:
        AuthenticationException: If authentication fails
    """
    try:
        # Use existing auth function
        return _get_current_user(credentials.credentials, db)
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        raise AuthenticationException("Invalid authentication credentials")


def get_optional_user(
    request: Request,
    db: Session = Depends(get_db)
):
    """Get current user if authenticated, None otherwise.
    
    Args:
        request: HTTP request
        db: Database session
        
    Returns:
        Current user instance or None
    """
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        return _get_current_user(token, db)
    except Exception:
        return None


def require_admin(
    current_user = Depends(get_current_user)
):
    """Require admin privileges.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user if admin
        
    Raises:
        AuthorizationException: If user is not admin
    """
    if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
        raise AuthorizationException("Admin privileges required")
    return current_user


def require_premium(
    current_user = Depends(get_current_user)
):
    """Require premium subscription.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user if premium
        
    Raises:
        AuthorizationException: If user doesn't have premium
    """
    if not hasattr(current_user, 'is_premium') or not current_user.is_premium:
        raise AuthorizationException("Premium subscription required")
    return current_user


# Feature dependencies
def get_feature(feature_name: str):
    """Create a dependency for a specific feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Dependency function that returns the feature instance
    """
    def _get_feature() -> Any:
        feature = registry.get_feature(feature_name)
        if feature is None:
            raise FeatureUnavailableException(
                f"Feature '{feature_name}' is not available",
                feature_name=feature_name
            )
        return feature
    
    return _get_feature


def get_optional_feature(feature_name: str):
    """Create a dependency for an optional feature.
    
    Args:
        feature_name: Name of the feature
        
    Returns:
        Dependency function that returns the feature instance or None
    """
    def _get_optional_feature() -> Optional[Any]:
        return registry.get_feature(feature_name)
    
    return _get_optional_feature


# Specific feature dependencies
def get_ai_core():
    """Get AI Core feature.
    
    Returns:
        AI Core instance
        
    Raises:
        FeatureUnavailableException: If AI Core is not available
    """
    return get_feature("ai_core")()


def get_optional_ai_core():
    """Get AI Core feature if available.
    
    Returns:
        AI Core instance or None
    """
    return get_optional_feature("ai_core")()


def get_defi_core():
    """Get DeFi Core feature.
    
    Returns:
        DeFi Core instance
        
    Raises:
        FeatureUnavailableException: If DeFi Core is not available
    """
    return get_feature("defi_core")()


def get_optional_defi_core():
    """Get DeFi Core feature if available.
    
    Returns:
        DeFi Core instance or None
    """
    return get_optional_feature("defi_core")()


def get_langchain():
    """Get LangChain integration.
    
    Returns:
        LangChain integration instance
        
    Raises:
        FeatureUnavailableException: If LangChain is not available
    """
    return get_feature("langchain")()


def get_optional_langchain():
    """Get LangChain integration if available.
    
    Returns:
        LangChain integration instance or None
    """
    return get_optional_feature("langchain")()


def get_technical_analysis():
    """Get Technical Analysis service.
    
    Returns:
        Technical Analysis service instance
        
    Raises:
        FeatureUnavailableException: If Technical Analysis is not available
    """
    return get_feature("technical_analysis")()


def get_optional_technical_analysis():
    """Get Technical Analysis service if available.
    
    Returns:
        Technical Analysis service instance or None
    """
    return get_optional_feature("technical_analysis")()


def get_ml_pipeline():
    """Get ML Pipeline service.
    
    Returns:
        ML Pipeline service instance
        
    Raises:
        FeatureUnavailableException: If ML Pipeline is not available
    """
    return get_feature("ml_pipeline")()


def get_optional_ml_pipeline():
    """Get ML Pipeline service if available.
    
    Returns:
        ML Pipeline service instance or None
    """
    return get_optional_feature("ml_pipeline")()


# Service dependencies (for existing services)
def get_websocket_manager(request: Request):
    """Get WebSocket manager from app state.
    
    Args:
        request: HTTP request
        
    Returns:
        WebSocket manager instance
    """
    return request.app.state.websocket_manager


def get_feature_summary(request: Request):
    """Get feature summary from app state.
    
    Args:
        request: HTTP request
        
    Returns:
        Feature summary dictionary
    """
    return request.app.state.feature_summary


# Validation dependencies
def validate_symbol(symbol: str) -> str:
    """Validate trading symbol.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        Validated symbol
        
    Raises:
        HTTPException: If symbol is invalid
    """
    if not symbol or len(symbol) < 1 or len(symbol) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid symbol format"
        )
    return symbol.upper()


def validate_pagination(skip: int = 0, limit: int = 100) -> tuple[int, int]:
    """Validate pagination parameters.
    
    Args:
        skip: Number of items to skip
        limit: Maximum number of items to return
        
    Returns:
        Validated skip and limit values
        
    Raises:
        HTTPException: If parameters are invalid
    """
    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Skip must be non-negative"
        )
    
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Limit must be between 1 and 1000"
        )
    
    return skip, limit


# Rate limiting dependency
def check_rate_limit(request: Request):
    """Check if request is rate limited.
    
    This is handled by middleware, but can be used for additional checks.
    
    Args:
        request: HTTP request
    """
    # Rate limiting is handled by middleware
    # This can be used for additional endpoint-specific rate limiting
    pass


# Request context dependency
def get_request_context(request: Request) -> dict:
    """Get request context information.
    
    Args:
        request: HTTP request
        
    Returns:
        Request context dictionary
    """
    return {
        "request_id": getattr(request.state, 'request_id', None),
        "method": request.method,
        "url": str(request.url),
        "client_ip": request.client.host if request.client else "unknown",
        "user_agent": request.headers.get("user-agent", "unknown")
    }


# Health check dependency
def get_health_status(request: Request) -> dict:
    """Get application health status.
    
    Args:
        request: HTTP request
        
    Returns:
        Health status dictionary
    """
    settings = get_settings()
    
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "features_loaded": len(registry.get_available_features()),
        "total_features": len(registry.get_all_features())
    }