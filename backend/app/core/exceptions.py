"""Custom exceptions for FinScope application.

This module defines a hierarchy of custom exceptions that provide
clear error handling and meaningful error messages throughout the application.
"""

from typing import Optional, Dict, Any


class FinScopeException(Exception):
    """Base exception class for FinScope application.
    
    All custom exceptions should inherit from this class.
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "message": self.message,
            "error_code": self.error_code,
            "details": self.details,
            "type": self.__class__.__name__
        }


class ValidationException(FinScopeException):
    """Exception raised for data validation errors."""
    
    def __init__(
        self,
        message: str = "Validation error",
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value
        
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details
        )


class AuthenticationException(FinScopeException):
    """Exception raised for authentication errors."""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            **kwargs
        )


class AuthorizationException(FinScopeException):
    """Exception raised for authorization errors."""
    
    def __init__(self, message: str = "Access denied", **kwargs):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            **kwargs
        )


class ServiceException(FinScopeException):
    """Exception raised for service-level errors."""
    
    def __init__(
        self,
        message: str = "Service error",
        service_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if service_name:
            details["service"] = service_name
        
        super().__init__(
            message=message,
            error_code="SERVICE_ERROR",
            details=details
        )


class DatabaseException(FinScopeException):
    """Exception raised for database-related errors."""
    
    def __init__(self, message: str = "Database error", **kwargs):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            **kwargs
        )


class ExternalAPIException(FinScopeException):
    """Exception raised for external API errors."""
    
    def __init__(
        self,
        message: str = "External API error",
        api_name: Optional[str] = None,
        status_code: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if api_name:
            details["api"] = api_name
        if status_code:
            details["status_code"] = status_code
        
        super().__init__(
            message=message,
            error_code="EXTERNAL_API_ERROR",
            details=details
        )


class TradingException(FinScopeException):
    """Exception raised for trading-related errors."""
    
    def __init__(
        self,
        message: str = "Trading error",
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if order_id:
            details["order_id"] = order_id
        if symbol:
            details["symbol"] = symbol
        
        super().__init__(
            message=message,
            error_code="TRADING_ERROR",
            details=details
        )


class PortfolioException(FinScopeException):
    """Exception raised for portfolio-related errors."""
    
    def __init__(
        self,
        message: str = "Portfolio error",
        portfolio_id: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if portfolio_id:
            details["portfolio_id"] = portfolio_id
        
        super().__init__(
            message=message,
            error_code="PORTFOLIO_ERROR",
            details=details
        )


class MarketDataException(FinScopeException):
    """Exception raised for market data errors."""
    
    def __init__(
        self,
        message: str = "Market data error",
        symbol: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if symbol:
            details["symbol"] = symbol
        if provider:
            details["provider"] = provider
        
        super().__init__(
            message=message,
            error_code="MARKET_DATA_ERROR",
            details=details
        )


class AIException(FinScopeException):
    """Exception raised for AI/ML related errors."""
    
    def __init__(
        self,
        message: str = "AI service error",
        model_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if model_name:
            details["model"] = model_name
        
        super().__init__(
            message=message,
            error_code="AI_ERROR",
            details=details
        )


class DeFiException(FinScopeException):
    """Exception raised for DeFi-related errors."""
    
    def __init__(
        self,
        message: str = "DeFi error",
        protocol: Optional[str] = None,
        network: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if protocol:
            details["protocol"] = protocol
        if network:
            details["network"] = network
        
        super().__init__(
            message=message,
            error_code="DEFI_ERROR",
            details=details
        )


class RateLimitException(FinScopeException):
    """Exception raised when rate limits are exceeded."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        reset_time: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if limit:
            details["limit"] = limit
        if reset_time:
            details["reset_time"] = reset_time
        
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_ERROR",
            details=details
        )


class ConfigurationException(FinScopeException):
    """Exception raised for configuration errors."""
    
    def __init__(
        self,
        message: str = "Configuration error",
        config_key: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if config_key:
            details["config_key"] = config_key
        
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details
        )


class FeatureUnavailableException(FinScopeException):
    """Exception raised when a required feature is not available."""
    
    def __init__(
        self,
        message: str = "Feature not available",
        feature_name: Optional[str] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if feature_name:
            details["feature"] = feature_name
        
        super().__init__(
            message=message,
            error_code="FEATURE_UNAVAILABLE",
            details=details
        )