"""Authentication and authorization system for FinScope.

This module provides:
- User authentication (login/logout)
- JWT token management
- Role-based access control
- Session management
- Password security
- API key authentication
"""

from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt
import secrets
import hashlib
from enum import Enum
from dataclasses import dataclass
import logging
from functools import wraps

from app.config.settings import get_settings
from app.core.exceptions import (
    AuthenticationException,
    AuthorizationException,
    ValidationException
)
from app.core.logging_config import get_logger
from app.core.security import SecurityManager

logger = get_logger("auth")
security_manager = SecurityManager()


class UserRole(str, Enum):
    """User roles for authorization."""
    ADMIN = "admin"
    PREMIUM = "premium"
    STANDARD = "standard"
    READONLY = "readonly"
    GUEST = "guest"


class TokenType(str, Enum):
    """Token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    RESET = "reset"
    VERIFICATION = "verification"


class AuthProvider(str, Enum):
    """Authentication providers."""
    LOCAL = "local"
    GOOGLE = "google"
    GITHUB = "github"
    MICROSOFT = "microsoft"
    API_KEY = "api_key"


class LoginRequest(BaseModel):
    """Login request model."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    remember_me: bool = False
    provider: AuthProvider = AuthProvider.LOCAL


class RegisterRequest(BaseModel):
    """Registration request model."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    terms_accepted: bool = Field(..., description="Must accept terms")
    
    class Config:
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePassword123!",
                "first_name": "John",
                "last_name": "Doe",
                "terms_accepted": True
            }
        }


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int
    scope: Optional[str] = None


class UserInfo(BaseModel):
    """User information model."""
    user_id: str
    email: str
    first_name: str
    last_name: str
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime] = None
    provider: AuthProvider = AuthProvider.LOCAL
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        from_attributes = True


@dataclass
class AuthContext:
    """Authentication context."""
    user_id: str
    email: str
    role: UserRole
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    scopes: List[str]
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if token is expired.
        
        Returns:
            True if token is expired
        """
        return datetime.utcnow() > self.expires_at
    
    @property
    def time_until_expiry(self) -> timedelta:
        """Get time until token expires.
        
        Returns:
            Time until expiry
        """
        return self.expires_at - datetime.utcnow()
    
    def has_scope(self, scope: str) -> bool:
        """Check if context has specific scope.
        
        Args:
            scope: Scope to check
            
        Returns:
            True if scope is present
        """
        return scope in self.scopes
    
    def has_role(self, required_role: UserRole) -> bool:
        """Check if user has required role or higher.
        
        Args:
            required_role: Required role
            
        Returns:
            True if user has required role or higher
        """
        role_hierarchy = {
            UserRole.GUEST: 0,
            UserRole.READONLY: 1,
            UserRole.STANDARD: 2,
            UserRole.PREMIUM: 3,
            UserRole.ADMIN: 4
        }
        
        user_level = role_hierarchy.get(self.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level


class AuthenticationManager:
    """Authentication and authorization manager."""
    
    def __init__(self):
        """Initialize authentication manager."""
        self.settings = get_settings()
        self.security_settings = self.settings.security
        
        # Password hashing
        self.pwd_context = CryptContext(
            schemes=["bcrypt"],
            deprecated="auto",
            bcrypt__rounds=12
        )
        
        # JWT settings
        self.algorithm = "HS256"
        self.access_token_expire = timedelta(minutes=self.security_settings.access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=self.security_settings.refresh_token_expire_days)
        
        # Security
        self.bearer_scheme = HTTPBearer(auto_error=False)
        
        # Session storage (in production, use Redis or database)
        self._active_sessions: Dict[str, AuthContext] = {}
        self._revoked_tokens: set = set()
        
        logger.info("Authentication manager initialized")
    
    def hash_password(self, password: str) -> str:
        """Hash a password.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash.
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password is correct
        """
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(
        self,
        user_id: str,
        email: str,
        role: UserRole,
        scopes: Optional[List[str]] = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token.
        
        Args:
            user_id: User ID
            email: User email
            role: User role
            scopes: Token scopes
            expires_delta: Custom expiration time
            
        Returns:
            JWT token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + self.access_token_expire
        
        payload = {
            "sub": user_id,
            "email": email,
            "role": role.value,
            "type": TokenType.ACCESS.value,
            "scopes": scopes or [],
            "iat": datetime.utcnow(),
            "exp": expire,
            "jti": secrets.token_urlsafe(32)  # JWT ID for revocation
        }
        
        token = jwt.encode(
            payload,
            self.security_settings.secret_key,
            algorithm=self.algorithm
        )
        
        logger.debug(f"Access token created for user {user_id}")
        return token
    
    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT refresh token.
        
        Args:
            user_id: User ID
            expires_delta: Custom expiration time
            
        Returns:
            JWT refresh token string
        """
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + self.refresh_token_expire
        
        payload = {
            "sub": user_id,
            "type": TokenType.REFRESH.value,
            "iat": datetime.utcnow(),
            "exp": expire,
            "jti": secrets.token_urlsafe(32)
        }
        
        token = jwt.encode(
            payload,
            self.security_settings.secret_key,
            algorithm=self.algorithm
        )
        
        logger.debug(f"Refresh token created for user {user_id}")
        return token
    
    def verify_token(self, token: str) -> AuthContext:
        """Verify and decode JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Authentication context
            
        Raises:
            AuthenticationException: If token is invalid
        """
        try:
            # Check if token is revoked
            if token in self._revoked_tokens:
                raise AuthenticationException("Token has been revoked")
            
            # Decode token
            payload = jwt.decode(
                token,
                self.security_settings.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Extract token data
            user_id = payload.get("sub")
            email = payload.get("email")
            role_str = payload.get("role")
            token_type_str = payload.get("type")
            scopes = payload.get("scopes", [])
            issued_at = datetime.fromtimestamp(payload.get("iat", 0))
            expires_at = datetime.fromtimestamp(payload.get("exp", 0))
            
            if not user_id:
                raise AuthenticationException("Invalid token: missing user ID")
            
            # Convert enums
            try:
                role = UserRole(role_str) if role_str else UserRole.GUEST
                token_type = TokenType(token_type_str) if token_type_str else TokenType.ACCESS
            except ValueError:
                raise AuthenticationException("Invalid token: invalid role or type")
            
            # Create auth context
            auth_context = AuthContext(
                user_id=user_id,
                email=email or "",
                role=role,
                token_type=token_type,
                issued_at=issued_at,
                expires_at=expires_at,
                scopes=scopes
            )
            
            # Check expiration
            if auth_context.is_expired:
                raise AuthenticationException("Token has expired")
            
            return auth_context
            
        except JWTError as e:
            logger.warning(f"JWT verification failed: {e}")
            raise AuthenticationException(f"Invalid token: {e}")
    
    def revoke_token(self, token: str) -> None:
        """Revoke a token.
        
        Args:
            token: Token to revoke
        """
        self._revoked_tokens.add(token)
        logger.info("Token revoked")
    
    def create_api_key(
        self,
        user_id: str,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_at: Optional[datetime] = None
    ) -> str:
        """Create API key.
        
        Args:
            user_id: User ID
            name: API key name
            scopes: API key scopes
            expires_at: Expiration time
            
        Returns:
            API key string
        """
        # Generate API key
        key_data = f"{user_id}:{name}:{datetime.utcnow().isoformat()}"
        api_key = hashlib.sha256(key_data.encode()).hexdigest()
        
        # Store API key info (in production, store in database)
        # This is a simplified implementation
        
        logger.info(f"API key created for user {user_id}: {name}")
        return f"fs_{api_key[:32]}"
    
    def verify_api_key(self, api_key: str) -> AuthContext:
        """Verify API key.
        
        Args:
            api_key: API key to verify
            
        Returns:
            Authentication context
            
        Raises:
            AuthenticationException: If API key is invalid
        """
        # This is a simplified implementation
        # In production, verify against database
        
        if not api_key.startswith("fs_"):
            raise AuthenticationException("Invalid API key format")
        
        # For demo purposes, create a basic context
        # In production, look up the actual user and permissions
        return AuthContext(
            user_id="api_user",
            email="api@finscope.com",
            role=UserRole.STANDARD,
            token_type=TokenType.API_KEY,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(days=365),
            scopes=["read", "write"]
        )
    
    async def authenticate_request(self, request: Request) -> Optional[AuthContext]:
        """Authenticate incoming request.
        
        Args:
            request: FastAPI request
            
        Returns:
            Authentication context or None
        """
        # Try Bearer token first
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            try:
                return self.verify_token(token)
            except AuthenticationException:
                pass
        
        # Try API key
        api_key = request.headers.get("X-API-Key")
        if api_key:
            try:
                return self.verify_api_key(api_key)
            except AuthenticationException:
                pass
        
        return None
    
    def create_session(
        self,
        auth_context: AuthContext,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> str:
        """Create user session.
        
        Args:
            auth_context: Authentication context
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session ID
        """
        session_id = secrets.token_urlsafe(32)
        
        # Update context with session info
        auth_context.session_id = session_id
        auth_context.ip_address = ip_address
        auth_context.user_agent = user_agent
        
        # Store session
        self._active_sessions[session_id] = auth_context
        
        logger.info(f"Session created for user {auth_context.user_id}: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[AuthContext]:
        """Get session by ID.
        
        Args:
            session_id: Session ID
            
        Returns:
            Authentication context or None
        """
        return self._active_sessions.get(session_id)
    
    def revoke_session(self, session_id: str) -> None:
        """Revoke user session.
        
        Args:
            session_id: Session ID to revoke
        """
        if session_id in self._active_sessions:
            del self._active_sessions[session_id]
            logger.info(f"Session revoked: {session_id}")
    
    def revoke_user_sessions(self, user_id: str) -> int:
        """Revoke all sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Number of sessions revoked
        """
        revoked_count = 0
        sessions_to_revoke = []
        
        for session_id, context in self._active_sessions.items():
            if context.user_id == user_id:
                sessions_to_revoke.append(session_id)
        
        for session_id in sessions_to_revoke:
            self.revoke_session(session_id)
            revoked_count += 1
        
        logger.info(f"Revoked {revoked_count} sessions for user {user_id}")
        return revoked_count
    
    def get_user_sessions(self, user_id: str) -> List[AuthContext]:
        """Get all active sessions for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of authentication contexts
        """
        return [
            context for context in self._active_sessions.values()
            if context.user_id == user_id
        ]
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        expired_sessions = []
        
        for session_id, context in self._active_sessions.items():
            if context.is_expired:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.revoke_session(session_id)
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
        return len(expired_sessions)


# Global authentication manager
auth_manager = AuthenticationManager()


# Dependency functions for FastAPI
async def get_current_user(request: Request) -> AuthContext:
    """Get current authenticated user.
    
    Args:
        request: FastAPI request
        
    Returns:
        Authentication context
        
    Raises:
        HTTPException: If authentication fails
    """
    auth_context = await auth_manager.authenticate_request(request)
    
    if not auth_context:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return auth_context


async def get_current_active_user(current_user: AuthContext = Depends(get_current_user)) -> AuthContext:
    """Get current active user.
    
    Args:
        current_user: Current user context
        
    Returns:
        Authentication context
        
    Raises:
        HTTPException: If user is inactive
    """
    # In a real implementation, check if user is active in database
    return current_user


async def get_admin_user(current_user: AuthContext = Depends(get_current_active_user)) -> AuthContext:
    """Get current admin user.
    
    Args:
        current_user: Current user context
        
    Returns:
        Authentication context
        
    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.has_role(UserRole.ADMIN):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return current_user


async def get_premium_user(current_user: AuthContext = Depends(get_current_active_user)) -> AuthContext:
    """Get current premium user.
    
    Args:
        current_user: Current user context
        
    Returns:
        Authentication context
        
    Raises:
        HTTPException: If user is not premium
    """
    if not current_user.has_role(UserRole.PREMIUM):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium access required"
        )
    
    return current_user


# Authorization decorators
def require_role(required_role: UserRole):
    """Decorator to require specific role.
    
    Args:
        required_role: Required user role
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract auth context from kwargs
            current_user = kwargs.get('current_user')
            if not current_user or not isinstance(current_user, AuthContext):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not current_user.has_role(required_role):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role {required_role.value} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def require_scope(required_scope: str):
    """Decorator to require specific scope.
    
    Args:
        required_scope: Required scope
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract auth context from kwargs
            current_user = kwargs.get('current_user')
            if not current_user or not isinstance(current_user, AuthContext):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not current_user.has_scope(required_scope):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Scope {required_scope} required"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Utility functions
def get_auth_manager() -> AuthenticationManager:
    """Get authentication manager instance.
    
    Returns:
        Authentication manager
    """
    return auth_manager


def create_test_user_token(
    user_id: str = "test_user",
    role: UserRole = UserRole.STANDARD
) -> str:
    """Create test user token for development.
    
    Args:
        user_id: Test user ID
        role: Test user role
        
    Returns:
        JWT token
    """
    return auth_manager.create_access_token(
        user_id=user_id,
        email=f"{user_id}@test.com",
        role=role,
        scopes=["read", "write"]
    )