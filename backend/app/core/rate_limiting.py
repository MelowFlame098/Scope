"""Rate limiting module for API endpoints.

This module provides:
- Redis-based rate limiting
- IP-based and user-based rate limiting
- Sliding window rate limiting
- Configurable limits per endpoint
- Rate limit headers
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import json
from fastapi import HTTPException, status
import redis.asyncio as redis
from app.config.settings import get_settings
from app.core.logging_config import get_logger

logger = get_logger("rate_limiting")
settings = get_settings()


class RateLimitExceeded(HTTPException):
    """Rate limit exceeded exception."""
    
    def __init__(self, detail: str, retry_after: int = None):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(retry_after)} if retry_after else None
        )


class RateLimiter:
    """Redis-based rate limiter with sliding window."""
    
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Ensure Redis connection is initialized."""
        if not self._initialized:
            try:
                self.redis_client = redis.from_url(
                    settings.redis.url,
                    password=settings.redis.password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # Test connection
                await self.redis_client.ping()
                self._initialized = True
                logger.info("Rate limiter Redis connection established")
            except Exception as e:
                logger.error(f"Failed to connect to Redis for rate limiting: {e}")
                # Fallback to in-memory rate limiting (not recommended for production)
                self.redis_client = None
                self._initialized = True
    
    async def check_rate_limit(
        self,
        key: str,
        max_requests: int,
        window_seconds: int,
        identifier: str = None
    ) -> bool:
        """Check if request is within rate limit.
        
        Args:
            key: Unique identifier for the rate limit (e.g., 'login:192.168.1.1')
            max_requests: Maximum number of requests allowed
            window_seconds: Time window in seconds
            identifier: Optional identifier for logging
        
        Returns:
            True if within limit
        
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        await self._ensure_initialized()
        
        if not self.redis_client:
            # Fallback: allow all requests if Redis is not available
            logger.warning("Redis not available, skipping rate limiting")
            return True
        
        try:
            current_time = datetime.utcnow().timestamp()
            window_start = current_time - window_seconds
            
            # Use Redis sorted set for sliding window
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(current_time): current_time})
            
            # Set expiration
            pipe.expire(key, window_seconds + 1)
            
            results = await pipe.execute()
            current_count = results[1]
            
            if current_count >= max_requests:
                # Get the oldest request time to calculate retry_after
                oldest_requests = await self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest_requests:
                    oldest_time = oldest_requests[0][1]
                    retry_after = int(oldest_time + window_seconds - current_time) + 1
                else:
                    retry_after = window_seconds
                
                logger.warning(
                    f"Rate limit exceeded for {identifier or key}: "
                    f"{current_count}/{max_requests} in {window_seconds}s"
                )
                
                raise RateLimitExceeded(
                    detail=f"Rate limit exceeded. Try again in {retry_after} seconds.",
                    retry_after=retry_after
                )
            
            return True
            
        except RateLimitExceeded:
            raise
        except Exception as e:
            logger.error(f"Rate limiting error for {key}: {e}")
            # Allow request if rate limiting fails
            return True
    
    async def get_rate_limit_status(
        self,
        key: str,
        max_requests: int,
        window_seconds: int
    ) -> Dict[str, Any]:
        """Get current rate limit status.
        
        Returns:
            Dictionary with rate limit information
        """
        await self._ensure_initialized()
        
        if not self.redis_client:
            return {
                "requests_made": 0,
                "requests_remaining": max_requests,
                "reset_time": None,
                "retry_after": 0
            }
        
        try:
            current_time = datetime.utcnow().timestamp()
            window_start = current_time - window_seconds
            
            # Clean old entries and count current
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, 0, window_start)
            pipe.zcard(key)
            pipe.zrange(key, 0, 0, withscores=True)  # Get oldest request
            
            results = await pipe.execute()
            current_count = results[1]
            oldest_requests = results[2]
            
            requests_remaining = max(0, max_requests - current_count)
            
            # Calculate reset time
            if oldest_requests:
                oldest_time = oldest_requests[0][1]
                reset_time = oldest_time + window_seconds
                retry_after = max(0, int(reset_time - current_time))
            else:
                reset_time = current_time + window_seconds
                retry_after = 0
            
            return {
                "requests_made": current_count,
                "requests_remaining": requests_remaining,
                "reset_time": datetime.fromtimestamp(reset_time).isoformat(),
                "retry_after": retry_after
            }
            
        except Exception as e:
            logger.error(f"Error getting rate limit status for {key}: {e}")
            return {
                "requests_made": 0,
                "requests_remaining": max_requests,
                "reset_time": None,
                "retry_after": 0
            }
    
    async def increment_attempt(
        self,
        key: str,
        expire_seconds: int = 3600
    ) -> int:
        """Increment attempt counter.
        
        Args:
            key: Counter key
            expire_seconds: Expiration time in seconds
        
        Returns:
            Current count
        """
        await self._ensure_initialized()
        
        if not self.redis_client:
            return 0
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, expire_seconds)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.error(f"Error incrementing attempt for {key}: {e}")
            return 0
    
    async def get_attempt_count(self, key: str) -> int:
        """Get current attempt count.
        
        Args:
            key: Counter key
        
        Returns:
            Current count
        """
        await self._ensure_initialized()
        
        if not self.redis_client:
            return 0
        
        try:
            count = await self.redis_client.get(key)
            return int(count) if count else 0
        except Exception as e:
            logger.error(f"Error getting attempt count for {key}: {e}")
            return 0
    
    async def clear_attempts(self, key: str) -> bool:
        """Clear attempt counter.
        
        Args:
            key: Counter key
        
        Returns:
            True if cleared successfully
        """
        await self._ensure_initialized()
        
        if not self.redis_client:
            return True
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error clearing attempts for {key}: {e}")
            return False
    
    async def reset_rate_limit(self, key: str) -> bool:
        """Reset rate limit for a key.
        
        Args:
            key: Rate limit key
        
        Returns:
            True if reset successfully
        """
        await self._ensure_initialized()
        
        if not self.redis_client:
            return True
        
        try:
            await self.redis_client.delete(key)
            logger.info(f"Rate limit reset for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error resetting rate limit for {key}: {e}")
            return False
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self._initialized = False
            logger.info("Rate limiter Redis connection closed")


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


# Rate limiting decorators and middleware
class RateLimitMiddleware:
    """Rate limiting middleware for FastAPI."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request, call_next):
        """Process request with rate limiting."""
        # Skip rate limiting for certain paths
        skip_paths = ["/health", "/docs", "/openapi.json", "/redoc"]
        if any(request.url.path.startswith(path) for path in skip_paths):
            return await call_next(request)
        
        # Apply global rate limiting
        client_ip = request.client.host
        try:
            await self.rate_limiter.check_rate_limit(
                f"global:{client_ip}",
                max_requests=100,  # 100 requests per minute
                window_seconds=60,
                identifier=client_ip
            )
        except RateLimitExceeded:
            raise
        
        response = await call_next(request)
        
        # Add rate limit headers
        try:
            status = await self.rate_limiter.get_rate_limit_status(
                f"global:{client_ip}",
                max_requests=100,
                window_seconds=60
            )
            
            response.headers["X-RateLimit-Limit"] = "100"
            response.headers["X-RateLimit-Remaining"] = str(status["requests_remaining"])
            response.headers["X-RateLimit-Reset"] = str(status["retry_after"])
            
        except Exception as e:
            logger.error(f"Error adding rate limit headers: {e}")
        
        return response


# Endpoint-specific rate limiting configurations
RATE_LIMIT_CONFIGS = {
    "auth_register": {"max_requests": 5, "window_seconds": 3600},  # 5 per hour
    "auth_login": {"max_requests": 10, "window_seconds": 900},     # 10 per 15 min
    "auth_password_reset": {"max_requests": 3, "window_seconds": 3600},  # 3 per hour
    "trading_order": {"max_requests": 100, "window_seconds": 60},  # 100 per minute
    "market_data": {"max_requests": 1000, "window_seconds": 60},   # 1000 per minute
    "portfolio_update": {"max_requests": 50, "window_seconds": 60}, # 50 per minute
}


def get_rate_limit_config(endpoint: str) -> Dict[str, int]:
    """Get rate limit configuration for endpoint."""
    return RATE_LIMIT_CONFIGS.get(endpoint, {"max_requests": 60, "window_seconds": 60})