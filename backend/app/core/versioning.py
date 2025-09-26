"""API versioning for FinScope application.

This module provides:
- Version detection from headers/URL
- Version-specific routing
- Backward compatibility
- Version deprecation warnings
- API evolution management
"""

from typing import Dict, List, Optional, Callable, Any, Union
from fastapi import Request, HTTPException, status
from fastapi.routing import APIRouter
from enum import Enum
import re
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from packaging import version

logger = logging.getLogger(__name__)


class APIVersion(Enum):
    """Supported API versions."""
    V1 = "1.0"
    V1_1 = "1.1"
    V2 = "2.0"
    
    @classmethod
    def get_latest(cls) -> 'APIVersion':
        """Get the latest API version.
        
        Returns:
            Latest API version
        """
        versions = [version.parse(v.value) for v in cls]
        latest = max(versions)
        
        for api_version in cls:
            if version.parse(api_version.value) == latest:
                return api_version
        
        return cls.V1  # Fallback
    
    @classmethod
    def get_default(cls) -> 'APIVersion':
        """Get the default API version for new clients.
        
        Returns:
            Default API version
        """
        return cls.V2  # Current stable version
    
    @classmethod
    def from_string(cls, version_str: str) -> Optional['APIVersion']:
        """Parse API version from string.
        
        Args:
            version_str: Version string (e.g., "1.0", "v2.0", "2")
            
        Returns:
            API version or None if invalid
        """
        # Clean version string
        clean_version = re.sub(r'^v', '', version_str.lower())
        
        # Handle major version only (e.g., "1" -> "1.0", "2" -> "2.0")
        if re.match(r'^\d+$', clean_version):
            clean_version += ".0"
        
        for api_version in cls:
            if api_version.value == clean_version:
                return api_version
        
        return None
    
    def is_deprecated(self) -> bool:
        """Check if this version is deprecated.
        
        Returns:
            True if version is deprecated
        """
        # V1.0 is deprecated
        return self == APIVersion.V1
    
    def deprecation_date(self) -> Optional[datetime]:
        """Get deprecation date for this version.
        
        Returns:
            Deprecation date or None if not deprecated
        """
        if self == APIVersion.V1:
            return datetime(2024, 6, 1)  # Example deprecation date
        return None
    
    def sunset_date(self) -> Optional[datetime]:
        """Get sunset date for this version.
        
        Returns:
            Sunset date or None if no sunset planned
        """
        if self == APIVersion.V1:
            return datetime(2024, 12, 31)  # Example sunset date
        return None


@dataclass
class VersionInfo:
    """Version information container."""
    version: APIVersion
    requested_version: Optional[str]
    source: str  # 'header', 'url', 'default'
    is_deprecated: bool
    deprecation_date: Optional[datetime]
    sunset_date: Optional[datetime]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "version": self.version.value,
            "requested_version": self.requested_version,
            "source": self.source,
            "is_deprecated": self.is_deprecated,
            "deprecation_date": self.deprecation_date.isoformat() if self.deprecation_date else None,
            "sunset_date": self.sunset_date.isoformat() if self.sunset_date else None
        }


class VersionDetector:
    """API version detection from requests."""
    
    def __init__(self):
        """Initialize version detector."""
        self.header_names = [
            "API-Version",
            "X-API-Version",
            "Accept-Version"
        ]
        self.url_pattern = re.compile(r'/v(\d+(?:\.\d+)?)')
    
    def detect_version(self, request: Request) -> VersionInfo:
        """Detect API version from request.
        
        Args:
            request: HTTP request
            
        Returns:
            Version information
        """
        # Try to get version from headers
        for header_name in self.header_names:
            header_value = request.headers.get(header_name)
            if header_value:
                api_version = APIVersion.from_string(header_value)
                if api_version:
                    return self._create_version_info(
                        api_version, header_value, "header"
                    )
        
        # Try to get version from URL path
        url_match = self.url_pattern.search(str(request.url.path))
        if url_match:
            version_str = url_match.group(1)
            api_version = APIVersion.from_string(version_str)
            if api_version:
                return self._create_version_info(
                    api_version, version_str, "url"
                )
        
        # Try to get version from Accept header (content negotiation)
        accept_header = request.headers.get("Accept", "")
        version_match = re.search(r'version=([\d\.]+)', accept_header)
        if version_match:
            version_str = version_match.group(1)
            api_version = APIVersion.from_string(version_str)
            if api_version:
                return self._create_version_info(
                    api_version, version_str, "accept_header"
                )
        
        # Use default version
        default_version = APIVersion.get_default()
        return self._create_version_info(default_version, None, "default")
    
    def _create_version_info(
        self,
        api_version: APIVersion,
        requested_version: Optional[str],
        source: str
    ) -> VersionInfo:
        """Create version info object.
        
        Args:
            api_version: Detected API version
            requested_version: Original requested version string
            source: Source of version detection
            
        Returns:
            Version information
        """
        return VersionInfo(
            version=api_version,
            requested_version=requested_version,
            source=source,
            is_deprecated=api_version.is_deprecated(),
            deprecation_date=api_version.deprecation_date(),
            sunset_date=api_version.sunset_date()
        )


class VersionedRouter:
    """Router that handles multiple API versions."""
    
    def __init__(self):
        """Initialize versioned router."""
        self.routers: Dict[APIVersion, APIRouter] = {}
        self.detector = VersionDetector()
    
    def add_router(self, version: APIVersion, router: APIRouter) -> None:
        """Add router for specific version.
        
        Args:
            version: API version
            router: FastAPI router
        """
        self.routers[version] = router
        logger.info(f"Added router for API version {version.value}")
    
    def get_router(self, version: APIVersion) -> Optional[APIRouter]:
        """Get router for specific version.
        
        Args:
            version: API version
            
        Returns:
            Router or None if not found
        """
        return self.routers.get(version)
    
    def get_all_routers(self) -> Dict[APIVersion, APIRouter]:
        """Get all registered routers.
        
        Returns:
            Dictionary of version to router mappings
        """
        return self.routers.copy()
    
    def create_version_middleware(self) -> Callable:
        """Create middleware for version detection and validation.
        
        Returns:
            Middleware function
        """
        async def version_middleware(request: Request, call_next):
            # Detect version
            version_info = self.detector.detect_version(request)
            
            # Store version info in request state
            request.state.version_info = version_info
            
            # Check if version is supported
            if version_info.version not in self.routers:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"API version {version_info.version.value} is not supported"
                )
            
            # Check if version is sunset
            if version_info.sunset_date and datetime.utcnow() > version_info.sunset_date:
                raise HTTPException(
                    status_code=status.HTTP_410_GONE,
                    detail=f"API version {version_info.version.value} has been sunset"
                )
            
            # Process request
            response = await call_next(request)
            
            # Add version headers to response
            response.headers["API-Version"] = version_info.version.value
            response.headers["API-Version-Source"] = version_info.source
            
            # Add deprecation warnings
            if version_info.is_deprecated:
                response.headers["Deprecation"] = "true"
                if version_info.deprecation_date:
                    response.headers["Deprecation-Date"] = version_info.deprecation_date.isoformat()
                if version_info.sunset_date:
                    response.headers["Sunset"] = version_info.sunset_date.isoformat()
                
                # Add warning header
                warning_msg = f"API version {version_info.version.value} is deprecated"
                if version_info.sunset_date:
                    warning_msg += f" and will be sunset on {version_info.sunset_date.date()}"
                response.headers["Warning"] = f'299 - "{warning_msg}"'
            
            return response
        
        return version_middleware


class VersionCompatibility:
    """Handle backward compatibility between API versions."""
    
    def __init__(self):
        """Initialize version compatibility handler."""
        self.transformers: Dict[tuple[APIVersion, APIVersion], Callable] = {}
    
    def register_transformer(
        self,
        from_version: APIVersion,
        to_version: APIVersion,
        transformer: Callable
    ) -> None:
        """Register data transformer between versions.
        
        Args:
            from_version: Source version
            to_version: Target version
            transformer: Transformation function
        """
        self.transformers[(from_version, to_version)] = transformer
        logger.info(f"Registered transformer from {from_version.value} to {to_version.value}")
    
    def transform_request(
        self,
        data: Any,
        from_version: APIVersion,
        to_version: APIVersion
    ) -> Any:
        """Transform request data between versions.
        
        Args:
            data: Request data
            from_version: Source version
            to_version: Target version
            
        Returns:
            Transformed data
        """
        transformer = self.transformers.get((from_version, to_version))
        if transformer:
            return transformer(data)
        return data
    
    def transform_response(
        self,
        data: Any,
        from_version: APIVersion,
        to_version: APIVersion
    ) -> Any:
        """Transform response data between versions.
        
        Args:
            data: Response data
            from_version: Source version
            to_version: Target version
            
        Returns:
            Transformed data
        """
        transformer = self.transformers.get((from_version, to_version))
        if transformer:
            return transformer(data)
        return data


# Utility functions
def get_request_version(request: Request) -> VersionInfo:
    """Get version info from request state.
    
    Args:
        request: HTTP request
        
    Returns:
        Version information
    """
    return getattr(request.state, 'version_info', None)


def require_version(min_version: APIVersion):
    """Decorator to require minimum API version.
    
    Args:
        min_version: Minimum required version
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(request: Request, *args, **kwargs):
            version_info = get_request_version(request)
            if not version_info:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="API version not detected"
                )
            
            current_version = version.parse(version_info.version.value)
            required_version = version.parse(min_version.value)
            
            if current_version < required_version:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"API version {min_version.value} or higher required"
                )
            
            return func(request, *args, **kwargs)
        return wrapper
    return decorator


def version_specific(version: APIVersion):
    """Decorator to mark endpoint as version-specific.
    
    Args:
        version: Specific API version
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(request: Request, *args, **kwargs):
            version_info = get_request_version(request)
            if not version_info or version_info.version != version:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Endpoint only available in API version {version.value}"
                )
            
            return func(request, *args, **kwargs)
        return wrapper
    return decorator


# Global instances
versioned_router = VersionedRouter()
version_compatibility = VersionCompatibility()


# Example transformers
def transform_v1_to_v2_user(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform user data from v1 to v2 format.
    
    Args:
        data: V1 user data
        
    Returns:
        V2 user data
    """
    # Example: V2 uses 'full_name' instead of separate 'first_name' and 'last_name'
    if 'first_name' in data and 'last_name' in data:
        data['full_name'] = f"{data['first_name']} {data['last_name']}"
        data.pop('first_name', None)
        data.pop('last_name', None)
    
    return data


def transform_v2_to_v1_user(data: Dict[str, Any]) -> Dict[str, Any]:
    """Transform user data from v2 to v1 format.
    
    Args:
        data: V2 user data
        
    Returns:
        V1 user data
    """
    # Example: V1 uses separate 'first_name' and 'last_name'
    if 'full_name' in data:
        parts = data['full_name'].split(' ', 1)
        data['first_name'] = parts[0]
        data['last_name'] = parts[1] if len(parts) > 1 else ''
        data.pop('full_name', None)
    
    return data


# Register example transformers
version_compatibility.register_transformer(
    APIVersion.V1, APIVersion.V2, transform_v1_to_v2_user
)
version_compatibility.register_transformer(
    APIVersion.V2, APIVersion.V1, transform_v2_to_v1_user
)