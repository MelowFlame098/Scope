"""Base service class for FinScope application.

This module provides a base service class that all services should inherit from,
establishing consistent patterns for:
- Dependency injection
- Error handling
- Logging
- Configuration access
- Database operations
- Caching
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Type, TypeVar, Generic
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
import logging
import asyncio
from datetime import datetime, timedelta
import json

from app.config.settings import Settings, get_settings
from app.core.exceptions import (
    ServiceException,
    DatabaseException,
    ValidationException,
    ConfigurationException
)

# Type variables
T = TypeVar('T')
ServiceType = TypeVar('ServiceType', bound='BaseService')


class BaseService(ABC):
    """Base service class for all FinScope services.
    
    Provides common functionality for:
    - Configuration access
    - Logging
    - Error handling
    - Database operations
    - Caching
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize base service.
        
        Args:
            settings: Application settings (optional)
        """
        self.settings = settings or get_settings()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        
        # Service metadata
        self.service_name = self.__class__.__name__
        self.service_version = "1.0.0"
        self.initialized_at = datetime.utcnow()
        
        self.logger.info(f"Initialized {self.service_name} service")
    
    @property
    def is_healthy(self) -> bool:
        """Check if service is healthy.
        
        Returns:
            True if service is healthy
        """
        try:
            return self._health_check()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return False
    
    def _health_check(self) -> bool:
        """Perform service-specific health check.
        
        Override this method in subclasses for custom health checks.
        
        Returns:
            True if service is healthy
        """
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status information.
        
        Returns:
            Service status dictionary
        """
        return {
            "service_name": self.service_name,
            "service_version": self.service_version,
            "initialized_at": self.initialized_at.isoformat(),
            "is_healthy": self.is_healthy,
            "uptime_seconds": (datetime.utcnow() - self.initialized_at).total_seconds()
        }
    
    # Error handling methods
    def handle_error(self, error: Exception, context: str = "") -> None:
        """Handle service errors consistently.
        
        Args:
            error: The exception that occurred
            context: Additional context information
            
        Raises:
            ServiceException: Wrapped service exception
        """
        error_msg = f"{self.service_name} error"
        if context:
            error_msg += f" in {context}"
        error_msg += f": {str(error)}"
        
        self.logger.error(error_msg, exc_info=True)
        
        # Convert specific exceptions
        if isinstance(error, SQLAlchemyError):
            raise DatabaseException(error_msg, original_error=error)
        elif isinstance(error, ValueError):
            raise ValidationException(error_msg, original_error=error)
        else:
            raise ServiceException(error_msg, original_error=error)
    
    def validate_input(self, data: Any, validator: callable) -> Any:
        """Validate input data.
        
        Args:
            data: Data to validate
            validator: Validation function
            
        Returns:
            Validated data
            
        Raises:
            ValidationException: If validation fails
        """
        try:
            return validator(data)
        except Exception as e:
            raise ValidationException(f"Input validation failed: {e}")
    
    # Database operation helpers
    def safe_db_operation(self, operation: callable, db: Session, *args, **kwargs) -> Any:
        """Safely execute database operation with error handling.
        
        Args:
            operation: Database operation function
            db: Database session
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
            
        Raises:
            DatabaseException: If database operation fails
        """
        try:
            result = operation(db, *args, **kwargs)
            db.commit()
            return result
        except SQLAlchemyError as e:
            db.rollback()
            self.handle_error(e, "database operation")
        except Exception as e:
            db.rollback()
            self.handle_error(e, "database operation")
    
    async def safe_async_db_operation(self, operation: callable, db: Session, *args, **kwargs) -> Any:
        """Safely execute async database operation with error handling.
        
        Args:
            operation: Async database operation function
            db: Database session
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
            
        Raises:
            DatabaseException: If database operation fails
        """
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(db, *args, **kwargs)
            else:
                result = operation(db, *args, **kwargs)
            db.commit()
            return result
        except SQLAlchemyError as e:
            db.rollback()
            self.handle_error(e, "async database operation")
        except Exception as e:
            db.rollback()
            self.handle_error(e, "async database operation")
    
    # Caching methods
    def cache_get(self, key: str) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None
        
        # Check TTL
        if key in self._cache_ttl:
            if datetime.utcnow() > self._cache_ttl[key]:
                self.cache_delete(key)
                return None
        
        return self._cache[key]
    
    def cache_set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
        """
        self._cache[key] = value
        if ttl_seconds > 0:
            self._cache_ttl[key] = datetime.utcnow() + timedelta(seconds=ttl_seconds)
    
    def cache_delete(self, key: str) -> None:
        """Delete value from cache.
        
        Args:
            key: Cache key
        """
        self._cache.pop(key, None)
        self._cache_ttl.pop(key, None)
    
    def cache_clear(self) -> None:
        """Clear all cached values."""
        self._cache.clear()
        self._cache_ttl.clear()
    
    # Configuration helpers
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value.
        
        Args:
            key: Configuration key (dot notation supported)
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            # Support dot notation (e.g., "database.url")
            keys = key.split('.')
            value = self.settings
            
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default
            
            return value
        except Exception:
            return default
    
    def require_config(self, key: str) -> Any:
        """Get required configuration value.
        
        Args:
            key: Configuration key
            
        Returns:
            Configuration value
            
        Raises:
            ConfigurationException: If configuration key is missing
        """
        value = self.get_config(key)
        if value is None:
            raise ConfigurationException(f"Required configuration '{key}' is missing")
        return value
    
    # Utility methods
    def serialize_data(self, data: Any) -> str:
        """Serialize data to JSON string.
        
        Args:
            data: Data to serialize
            
        Returns:
            JSON string
        """
        try:
            return json.dumps(data, default=str)
        except Exception as e:
            self.logger.error(f"Failed to serialize data: {e}")
            return "{}"
    
    def deserialize_data(self, data: str) -> Any:
        """Deserialize JSON string to data.
        
        Args:
            data: JSON string
            
        Returns:
            Deserialized data
        """
        try:
            return json.loads(data)
        except Exception as e:
            self.logger.error(f"Failed to deserialize data: {e}")
            return None
    
    # Async helpers
    async def run_in_background(self, operation: callable, *args, **kwargs) -> Any:
        """Run operation in background thread.
        
        Args:
            operation: Operation to run
            *args: Operation arguments
            **kwargs: Operation keyword arguments
            
        Returns:
            Operation result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, operation, *args, **kwargs)
    
    # Lifecycle methods
    async def startup(self) -> None:
        """Service startup hook.
        
        Override this method in subclasses for custom startup logic.
        """
        self.logger.info(f"Starting up {self.service_name} service")
    
    async def shutdown(self) -> None:
        """Service shutdown hook.
        
        Override this method in subclasses for custom shutdown logic.
        """
        self.logger.info(f"Shutting down {self.service_name} service")
        self.cache_clear()
    
    # Abstract methods (optional to override)
    def initialize(self) -> None:
        """Initialize service-specific resources.
        
        Override this method in subclasses for custom initialization.
        """
        pass
    
    def cleanup(self) -> None:
        """Cleanup service-specific resources.
        
        Override this method in subclasses for custom cleanup.
        """
        pass


class AsyncBaseService(BaseService):
    """Async version of BaseService for services that need async operations."""
    
    def __init__(self, settings: Optional[Settings] = None):
        """Initialize async base service.
        
        Args:
            settings: Application settings (optional)
        """
        super().__init__(settings)
        self._async_tasks: List[asyncio.Task] = []
    
    async def create_task(self, coro) -> asyncio.Task:
        """Create and track async task.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Created task
        """
        task = asyncio.create_task(coro)
        self._async_tasks.append(task)
        return task
    
    async def wait_for_tasks(self, timeout: Optional[float] = None) -> None:
        """Wait for all tracked tasks to complete.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        if self._async_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._async_tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                self.logger.warning("Some tasks did not complete within timeout")
            finally:
                self._async_tasks.clear()
    
    async def shutdown(self) -> None:
        """Async service shutdown with task cleanup."""
        await super().shutdown()
        
        # Cancel remaining tasks
        for task in self._async_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation
        if self._async_tasks:
            await asyncio.gather(*self._async_tasks, return_exceptions=True)
        
        self._async_tasks.clear()


class ServiceRegistry:
    """Registry for managing service instances."""
    
    def __init__(self):
        """Initialize service registry."""
        self._services: Dict[str, BaseService] = {}
        self._service_types: Dict[str, Type[BaseService]] = {}
    
    def register_service_type(self, name: str, service_class: Type[BaseService]) -> None:
        """Register a service type.
        
        Args:
            name: Service name
            service_class: Service class
        """
        self._service_types[name] = service_class
    
    def create_service(self, name: str, **kwargs) -> BaseService:
        """Create and register a service instance.
        
        Args:
            name: Service name
            **kwargs: Service initialization arguments
            
        Returns:
            Created service instance
            
        Raises:
            ServiceException: If service type is not registered
        """
        if name not in self._service_types:
            raise ServiceException(f"Service type '{name}' is not registered")
        
        service_class = self._service_types[name]
        service = service_class(**kwargs)
        self._services[name] = service
        
        return service
    
    def get_service(self, name: str) -> Optional[BaseService]:
        """Get service instance by name.
        
        Args:
            name: Service name
            
        Returns:
            Service instance or None if not found
        """
        return self._services.get(name)
    
    def get_all_services(self) -> Dict[str, BaseService]:
        """Get all registered service instances.
        
        Returns:
            Dictionary of service instances
        """
        return self._services.copy()
    
    async def startup_all(self) -> None:
        """Start up all registered services."""
        for service in self._services.values():
            await service.startup()
    
    async def shutdown_all(self) -> None:
        """Shut down all registered services."""
        for service in self._services.values():
            await service.shutdown()


# Global service registry instance
service_registry = ServiceRegistry()