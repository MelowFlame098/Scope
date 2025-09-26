"""Middleware components for FinScope application.

This module provides various middleware components for:
- Request/response logging
- Rate limiting
- Security headers
- Performance monitoring
- Error handling
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from typing import Callable
import time
import logging
import uuid
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from app.config.settings import Settings
from app.core.exceptions import RateLimitException

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"Response {request_id}: {response.status_code} "
                f"in {process_time:.3f}s"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = str(process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"Error {request_id}: {str(e)} in {process_time:.3f}s",
                exc_info=True
            )
            raise


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(self, app: FastAPI, requests_per_minute: int = 100):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
        self.cleanup_interval = 60  # seconds
        self.last_cleanup = time.time()
    
    def _cleanup_old_requests(self):
        """Remove old request timestamps."""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            cutoff_time = current_time - 60  # 1 minute ago
            
            for client_ip in list(self.requests.keys()):
                self.requests[client_ip] = [
                    timestamp for timestamp in self.requests[client_ip]
                    if timestamp > cutoff_time
                ]
                
                # Remove empty entries
                if not self.requests[client_ip]:
                    del self.requests[client_ip]
            
            self.last_cleanup = current_time
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Get requests from the last minute
        recent_requests = [
            timestamp for timestamp in self.requests[client_ip]
            if timestamp > minute_ago
        ]
        
        return len(recent_requests) >= self.requests_per_minute
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Cleanup old requests periodically
        self._cleanup_old_requests()
        
        # Check rate limit
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            raise RateLimitException(
                message="Too many requests",
                limit=self.requests_per_minute,
                reset_time=60
            )
        
        # Record this request
        self.requests[client_ip].append(time.time())
        
        return await call_next(request)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware for adding security headers."""
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=()"
        )
        
        return response


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring application performance."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.request_times = []
        self.slow_request_threshold = 1.0  # seconds
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        self.request_times.append(process_time)
        
        # Keep only last 1000 requests
        if len(self.request_times) > 1000:
            self.request_times = self.request_times[-1000:]
        
        # Log slow requests
        if process_time > self.slow_request_threshold:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {process_time:.3f}s"
            )
        
        # Add performance headers
        response.headers["X-Response-Time"] = f"{process_time:.3f}"
        
        return response
    
    def get_average_response_time(self) -> float:
        """Get average response time."""
        if not self.request_times:
            return 0.0
        return sum(self.request_times) / len(self.request_times)
    
    def get_slow_requests_count(self) -> int:
        """Get count of slow requests."""
        return sum(
            1 for time in self.request_times
            if time > self.slow_request_threshold
        )


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for centralized error handling."""
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        try:
            return await call_next(request)
        except Exception as e:
            # Log the error with request context
            request_id = getattr(request.state, 'request_id', 'unknown')
            logger.error(
                f"Unhandled error in request {request_id}: {str(e)}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "client_ip": request.client.host if request.client else "unknown"
                }
            )
            
            # Re-raise the exception to be handled by FastAPI's exception handlers
            raise


class CacheControlMiddleware(BaseHTTPMiddleware):
    """Middleware for adding cache control headers."""
    
    def __init__(self, app: FastAPI, default_max_age: int = 300):
        super().__init__(app)
        self.default_max_age = default_max_age
        self.cache_rules = {
            "/health": 60,  # Cache health checks for 1 minute
            "/features": 300,  # Cache feature status for 5 minutes
            "/api/v1/market-data": 60,  # Cache market data for 1 minute
        }
    
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)
        
        # Skip caching for non-GET requests
        if request.method != "GET":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            return response
        
        # Apply cache rules
        path = request.url.path
        max_age = self.default_max_age
        
        for rule_path, rule_max_age in self.cache_rules.items():
            if path.startswith(rule_path):
                max_age = rule_max_age
                break
        
        response.headers["Cache-Control"] = f"public, max-age={max_age}"
        response.headers["ETag"] = f'"{hash(str(response.body))}"'
        
        return response


def setup_middleware(app: FastAPI, settings: Settings):
    """Setup all middleware for the application.
    
    Args:
        app: FastAPI application instance
        settings: Application settings
    """
    # Add middleware in reverse order (last added = first executed)
    
    # Cache control (outermost)
    app.add_middleware(CacheControlMiddleware)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Performance monitoring
    performance_middleware = PerformanceMonitoringMiddleware(app)
    app.add_middleware(PerformanceMonitoringMiddleware)
    
    # Store performance middleware in app state for metrics endpoint
    app.state.performance_middleware = performance_middleware
    
    # Rate limiting
    if not settings.is_development:
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.rate_limit_per_minute
        )
    
    # Error handling
    app.add_middleware(ErrorHandlingMiddleware)
    
    # Request logging (innermost)
    if settings.monitoring.log_level == "DEBUG" or settings.is_development:
        app.add_middleware(RequestLoggingMiddleware)
    
    logger.info("Middleware setup complete")


# Metrics endpoint for performance monitoring
def get_performance_metrics(app: FastAPI) -> dict:
    """Get performance metrics from middleware.
    
    Args:
        app: FastAPI application instance
        
    Returns:
        Dictionary with performance metrics
    """
    if not hasattr(app.state, 'performance_middleware'):
        return {"error": "Performance middleware not available"}
    
    middleware = app.state.performance_middleware
    
    return {
        "average_response_time": middleware.get_average_response_time(),
        "slow_requests_count": middleware.get_slow_requests_count(),
        "total_requests": len(middleware.request_times),
        "slow_request_threshold": middleware.slow_request_threshold
    }