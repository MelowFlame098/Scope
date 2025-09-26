from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session
import os
from fastapi import HTTPException, status

from ..repositories.user import user_repository
from ..models import User
from ..schemas import UserCreate, UserUpdate

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

class AuthService:
    """
    Service for handling authentication and authorization
    """
    
    def __init__(self):
        self.user_repo = user_repository
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """
        Hash a password
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        return pwd_context.hash(password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token
        
        Args:
            data: Data to encode in token
            expires_delta: Optional expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """
        Create a JWT refresh token
        
        Args:
            data: Data to encode in token
            
        Returns:
            JWT refresh token string
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Optional[Dict[str, Any]]:
        """
        Verify and decode a JWT token
        
        Args:
            token: JWT token string
            token_type: Expected token type (access or refresh)
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Check token type
            if payload.get("type") != token_type:
                return None
            
            # Check expiration
            exp = payload.get("exp")
            if exp is None or datetime.fromtimestamp(exp) < datetime.utcnow():
                return None
            
            return payload
        
        except JWTError:
            return None
    
    def authenticate_user(self, db: Session, email: str, password: str) -> Optional[User]:
        """
        Authenticate a user with email and password
        
        Args:
            db: Database session
            email: User email
            password: Plain text password
            
        Returns:
            User object if authentication successful, None otherwise
        """
        user = self.user_repo.get_by_email(db, email=email)
        
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        self.user_repo.update_last_login(db, user_id=user.id)
        
        return user
    
    def create_user(self, db: Session, user_create: UserCreate) -> User:
        """
        Create a new user
        
        Args:
            db: Database session
            user_create: User creation data
            
        Returns:
            Created user
            
        Raises:
            HTTPException: If email or username already exists
        """
        # Check if email already exists
        if self.user_repo.get_by_email(db, email=user_create.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Check if username already exists
        if self.user_repo.get_by_username(db, username=user_create.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Hash password
        hashed_password = self.get_password_hash(user_create.password)
        
        # Create user data
        user_data = {
            "email": user_create.email,
            "username": user_create.username,
            "full_name": user_create.full_name,
            "hashed_password": hashed_password,
            "risk_tolerance": user_create.risk_tolerance,
            "investment_goals": user_create.investment_goals,
            "is_active": True
        }
        
        return self.user_repo.create(db, obj_in=user_data)
    
    def get_current_user(self, db: Session, token: str) -> User:
        """
        Get current user from JWT token
        
        Args:
            db: Database session
            token: JWT access token
            
        Returns:
            Current user
            
        Raises:
            HTTPException: If token is invalid or user not found
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        payload = self.verify_token(token, token_type="access")
        
        if payload is None:
            raise credentials_exception
        
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        user = self.user_repo.get(db, id=user_id)
        if user is None:
            raise credentials_exception
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Inactive user"
            )
        
        return user
    
    def refresh_access_token(self, db: Session, refresh_token: str) -> Dict[str, str]:
        """
        Create new access token from refresh token
        
        Args:
            db: Database session
            refresh_token: JWT refresh token
            
        Returns:
            New access token
            
        Raises:
            HTTPException: If refresh token is invalid
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        payload = self.verify_token(refresh_token, token_type="refresh")
        
        if payload is None:
            raise credentials_exception
        
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
        
        user = self.user_repo.get(db, id=user_id)
        if user is None or not user.is_active:
            raise credentials_exception
        
        # Create new access token
        access_token = self.create_access_token(data={"sub": str(user.id)})
        
        return {"access_token": access_token, "token_type": "bearer"}
    
    def change_password(self, db: Session, user: User, current_password: str, new_password: str) -> bool:
        """
        Change user password
        
        Args:
            db: Database session
            user: User object
            current_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully
            
        Raises:
            HTTPException: If current password is incorrect
        """
        if not self.verify_password(current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Incorrect current password"
            )
        
        # Hash new password
        hashed_password = self.get_password_hash(new_password)
        
        # Update user
        update_data = {"hashed_password": hashed_password}
        self.user_repo.update(db, db_obj=user, obj_in=update_data)
        
        return True
    
    def reset_password(self, db: Session, email: str) -> Dict[str, str]:
        """
        Generate password reset token
        
        Args:
            db: Database session
            email: User email
            
        Returns:
            Password reset token
            
        Raises:
            HTTPException: If user not found
        """
        user = self.user_repo.get_by_email(db, email=email)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Create reset token (expires in 1 hour)
        reset_token = self.create_access_token(
            data={"sub": str(user.id), "purpose": "password_reset"},
            expires_delta=timedelta(hours=1)
        )
        
        return {"reset_token": reset_token}
    
    def confirm_password_reset(self, db: Session, token: str, new_password: str) -> bool:
        """
        Confirm password reset with token
        
        Args:
            db: Database session
            token: Password reset token
            new_password: New password
            
        Returns:
            True if password reset successful
            
        Raises:
            HTTPException: If token is invalid
        """
        payload = self.verify_token(token, token_type="access")
        
        if payload is None or payload.get("purpose") != "password_reset":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        user_id: int = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token"
            )
        
        user = self.user_repo.get(db, id=user_id)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Hash new password
        hashed_password = self.get_password_hash(new_password)
        
        # Update user
        update_data = {"hashed_password": hashed_password}
        self.user_repo.update(db, db_obj=user, obj_in=update_data)
        
        return True
    
    def login_user(self, db: Session, email: str, password: str) -> Dict[str, Any]:
        """
        Login user and return tokens
        
        Args:
            db: Database session
            email: User email
            password: User password
            
        Returns:
            Access and refresh tokens with user info
            
        Raises:
            HTTPException: If authentication fails
        """
        user = self.authenticate_user(db, email, password)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Create tokens
        access_token = self.create_access_token(data={"sub": str(user.id)})
        refresh_token = self.create_refresh_token(data={"sub": str(user.id)})
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name,
                "risk_tolerance": user.risk_tolerance
            }
        }
    
    def check_email_availability(self, db: Session, email: str) -> bool:
        """
        Check if email is available
        
        Args:
            db: Database session
            email: Email to check
            
        Returns:
            True if email is available
        """
        return self.user_repo.get_by_email(db, email=email) is None
    
    def check_username_availability(self, db: Session, username: str) -> bool:
        """
        Check if username is available
        
        Args:
            db: Database session
            username: Username to check
            
        Returns:
            True if username is available
        """
        return self.user_repo.get_by_username(db, username=username) is None

# Create service instance
auth_service = AuthService()