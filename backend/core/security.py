from datetime import datetime, timedelta
from typing import Any, Union, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from .config import settings
from ..database import get_db
from ..models import User
from ..schemas import TokenData

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login"
)


def create_access_token(
    subject: Union[str, Any], expires_delta: timedelta = None
) -> str:
    """
    Create a JWT access token.
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "access"}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def create_refresh_token(
    subject: Union[str, Any], expires_delta: timedelta = None
) -> str:
    """
    Create a JWT refresh token.
    """
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    
    to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Hash a password.
    """
    return pwd_context.hash(password)


def verify_token(token: str, token_type: str = "access") -> Optional[str]:
    """
    Verify a JWT token and return the subject (username).
    """
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        token_type_claim: str = payload.get("type")
        
        if username is None or token_type_claim != token_type:
            return None
        
        return username
    except JWTError:
        return None


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    Get the current authenticated user from the JWT token.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != "access":
            raise credentials_exception
        
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    
    user = db.query(User).filter(User.username == token_data.username).first()
    if user is None:
        raise credentials_exception
    
    return user


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get the current active user.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


def get_current_active_superuser(current_user: User = Depends(get_current_user)) -> User:
    """
    Get the current active superuser.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user


def create_password_reset_token(email: str) -> str:
    """
    Create a password reset token.
    """
    delta = timedelta(hours=settings.EMAIL_RESET_TOKEN_EXPIRE_HOURS)
    now = datetime.utcnow()
    expires = now + delta
    exp = expires.timestamp()
    encoded_jwt = jwt.encode(
        {"exp": exp, "nbf": now, "sub": email, "type": "password_reset"},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def verify_password_reset_token(token: str) -> Optional[str]:
    """
    Verify a password reset token and return the email.
    """
    try:
        decoded_token = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_type = decoded_token.get("type")
        if token_type != "password_reset":
            return None
        return decoded_token["sub"]
    except JWTError:
        return None


def generate_api_key() -> str:
    """
    Generate a secure API key for external integrations.
    """
    import secrets
    return secrets.token_urlsafe(32)


def validate_api_key(api_key: str, db: Session) -> Optional[User]:
    """
    Validate an API key and return the associated user.
    Note: This would require an api_keys table in a real implementation.
    """
    # In a real implementation, you would:
    # 1. Query the api_keys table
    # 2. Check if the key is active and not expired
    # 3. Return the associated user
    # For now, this is a placeholder
    return None


def check_permissions(user: User, resource: str, action: str) -> bool:
    """
    Check if a user has permission to perform an action on a resource.
    This is a basic implementation - in a real app you might use a more
    sophisticated permission system like RBAC or ABAC.
    """
    # Superusers can do anything
    if user.is_superuser:
        return True
    
    # Basic permission checks
    if not user.is_active:
        return False
    
    # Resource-specific permissions
    if resource == "portfolio":
        # Users can manage their own portfolios
        return action in ["read", "create", "update", "delete"]
    
    elif resource == "user":
        # Users can read and update their own profile
        return action in ["read", "update"]
    
    elif resource == "asset":
        # All users can read asset data
        return action == "read"
    
    elif resource == "news":
        # All users can read news
        return action == "read"
    
    elif resource == "admin":
        # Only superusers can access admin functions
        return user.is_superuser
    
    # Default deny
    return False


def require_permission(resource: str, action: str):
    """
    Decorator to require specific permissions for an endpoint.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get current user from kwargs (assumes it's passed as dependency)
            current_user = kwargs.get('current_user')
            if not current_user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            if not check_permissions(current_user, resource, action):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Insufficient permissions"
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def create_email_verification_token(email: str) -> str:
    """
    Create an email verification token.
    """
    delta = timedelta(hours=24)  # Email verification expires in 24 hours
    now = datetime.utcnow()
    expires = now + delta
    exp = expires.timestamp()
    encoded_jwt = jwt.encode(
        {"exp": exp, "nbf": now, "sub": email, "type": "email_verification"},
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def verify_email_verification_token(token: str) -> Optional[str]:
    """
    Verify an email verification token and return the email.
    """
    try:
        decoded_token = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM]
        )
        token_type = decoded_token.get("type")
        if token_type != "email_verification":
            return None
        return decoded_token["sub"]
    except JWTError:
        return None


def is_token_blacklisted(token: str) -> bool:
    """
    Check if a token is blacklisted.
    In a real implementation, you would check against a blacklist stored in Redis or database.
    """
    # Placeholder implementation
    return False


def blacklist_token(token: str) -> None:
    """
    Add a token to the blacklist.
    In a real implementation, you would store this in Redis or database.
    """
    # Placeholder implementation
    pass