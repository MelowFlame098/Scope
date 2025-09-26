from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv

from database import get_db
from db_models import User, Asset, Portfolio, Watchlist, ForumPost, Comment, NewsArticle
from schemas import TokenData

load_dotenv()

# Security configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
security = HTTPBearer()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[TokenData]:
    """Verify JWT token and return token data."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        token_data = TokenData(email=email)
        return token_data
    except JWTError:
        return None

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email from database."""
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password."""
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)) -> User:
    """Get current authenticated user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        token_data = verify_token(token)
        if token_data is None or token_data.email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (must be active)."""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def get_optional_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security), db: Session = Depends(get_db)) -> Optional[User]:
    """Get current user if authenticated, otherwise return None."""
    if not credentials:
        return None
    
    try:
        token = credentials.credentials
        token_data = verify_token(token)
        if token_data is None or token_data.email is None:
            return None
        
        user = get_user_by_email(db, email=token_data.email)
        return user if user and user.is_active else None
    except:
        return None

def create_user(db: Session, email: str, username: str, password: str, full_name: Optional[str] = None) -> User:
    """Create a new user."""
    # Check if user already exists
    if get_user_by_email(db, email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check if username is taken
    existing_username = db.query(User).filter(User.username == username).first()
    if existing_username:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create new user
    hashed_password = get_password_hash(password)
    user = User(
        email=email,
        username=username,
        hashed_password=hashed_password,
        full_name=full_name,
        is_active=True,
        is_verified=False
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user

def update_user_password(db: Session, user: User, new_password: str) -> User:
    """Update user password."""
    user.hashed_password = get_password_hash(new_password)
    db.commit()
    db.refresh(user)
    return user

def verify_user_email(db: Session, user: User) -> User:
    """Mark user email as verified."""
    user.is_verified = True
    db.commit()
    db.refresh(user)
    return user

def deactivate_user(db: Session, user: User) -> User:
    """Deactivate user account."""
    user.is_active = False
    db.commit()
    db.refresh(user)
    return user

def reactivate_user(db: Session, user: User) -> User:
    """Reactivate user account."""
    user.is_active = True
    db.commit()
    db.refresh(user)
    return user

# Admin functions
def is_admin(user: User) -> bool:
    """Check if user has admin privileges."""
    # For now, check if user email is in admin list
    # In production, you might want to add an is_admin field to User model
    admin_emails = os.getenv("ADMIN_EMAILS", "").split(",")
    return user.email in admin_emails

def require_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """Dependency that requires admin privileges."""
    if not is_admin(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

# Rate limiting (basic implementation)
class RateLimiter:
    def __init__(self):
        self.requests = {}
    
    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed based on rate limit."""
        now = datetime.utcnow()
        
        if key not in self.requests:
            self.requests[key] = []
        
        # Remove old requests outside the window
        self.requests[key] = [
            req_time for req_time in self.requests[key]
            if (now - req_time).total_seconds() < window
        ]
        
        # Check if under limit
        if len(self.requests[key]) < limit:
            self.requests[key].append(now)
            return True
        
        return False

# Global rate limiter instance
rate_limiter = RateLimiter()

def check_rate_limit(key: str, limit: int = 100, window: int = 3600):
    """Rate limiting dependency."""
    if not rate_limiter.is_allowed(key, limit, window):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

# Password validation
def validate_password(password: str) -> bool:
    """Validate password strength."""
    if len(password) < 8:
        return False
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    return has_upper and has_lower and has_digit and has_special

def get_password_requirements() -> dict:
    """Get password requirements for frontend validation."""
    return {
        "min_length": 8,
        "require_uppercase": True,
        "require_lowercase": True,
        "require_digit": True,
        "require_special": True,
        "special_chars": "!@#$%^&*()_+-=[]{}|;:,.<>?"
    }

# Session management
def invalidate_user_sessions(db: Session, user: User):
    """Invalidate all user sessions (for logout all devices)."""
    # In a production system, you might want to maintain a blacklist of tokens
    # or use a different approach like storing session IDs in the database
    pass

# Two-factor authentication (placeholder)
def generate_2fa_secret() -> str:
    """Generate 2FA secret for user."""
    # Implementation would depend on your 2FA provider (e.g., TOTP)
    pass

def verify_2fa_token(secret: str, token: str) -> bool:
    """Verify 2FA token."""
    # Implementation would depend on your 2FA provider
    pass