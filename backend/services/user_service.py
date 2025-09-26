from typing import Optional, List
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import JWTError, jwt
import os

from ..repositories.user import user_repository
from ..models import User
from ..schemas import UserCreate, UserUpdate, UserInDB

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class UserService:
    """
    Service class for user-related business logic
    """
    
    def __init__(self):
        self.repository = user_repository
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash
        
        Args:
            plain_password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches, False otherwise
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
    
    def authenticate_user(self, db: Session, email: str, password: str) -> Optional[User]:
        """
        Authenticate a user with email and password
        
        Args:
            db: Database session
            email: User email
            password: Plain text password
            
        Returns:
            User instance if authentication successful, None otherwise
        """
        user = self.repository.get_by_email(db, email=email)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token
        
        Args:
            data: Data to encode in token
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[str]:
        """
        Verify a JWT token and extract user email
        
        Args:
            token: JWT token
            
        Returns:
            User email if token is valid, None otherwise
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            email: str = payload.get("sub")
            if email is None:
                return None
            return email
        except JWTError:
            return None
    
    def get_current_user(self, db: Session, token: str) -> Optional[User]:
        """
        Get current user from JWT token
        
        Args:
            db: Database session
            token: JWT token
            
        Returns:
            User instance if token is valid, None otherwise
        """
        email = self.verify_token(token)
        if email is None:
            return None
        
        user = self.repository.get_by_email(db, email=email)
        return user
    
    def create_user(self, db: Session, *, user_in: UserCreate) -> User:
        """
        Create a new user
        
        Args:
            db: Database session
            user_in: User creation data
            
        Returns:
            Created user instance
        """
        # Check if user already exists
        existing_user = self.repository.get_by_email(db, email=user_in.email)
        if existing_user:
            raise ValueError("User with this email already exists")
        
        existing_username = self.repository.get_by_username(db, username=user_in.username)
        if existing_username:
            raise ValueError("User with this username already exists")
        
        # Hash password
        hashed_password = self.get_password_hash(user_in.password)
        
        # Create user data
        user_data = user_in.dict()
        user_data.pop("password")
        user_data["hashed_password"] = hashed_password
        
        # Create user
        user = self.repository.create(db, obj_in=user_data)
        return user
    
    def update_user(self, db: Session, *, user_id: str, user_in: UserUpdate) -> Optional[User]:
        """
        Update user information
        
        Args:
            db: Database session
            user_id: User ID
            user_in: User update data
            
        Returns:
            Updated user instance or None
        """
        user = self.repository.get(db, user_id)
        if not user:
            return None
        
        update_data = user_in.dict(exclude_unset=True)
        
        # Hash password if provided
        if "password" in update_data:
            update_data["hashed_password"] = self.get_password_hash(update_data.pop("password"))
        
        updated_user = self.repository.update(db, db_obj=user, obj_in=update_data)
        return updated_user
    
    def get_user_by_id(self, db: Session, *, user_id: str) -> Optional[User]:
        """
        Get user by ID
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            User instance or None
        """
        return self.repository.get(db, user_id)
    
    def get_user_by_email(self, db: Session, *, email: str) -> Optional[User]:
        """
        Get user by email
        
        Args:
            db: Database session
            email: User email
            
        Returns:
            User instance or None
        """
        return self.repository.get_by_email(db, email=email)
    
    def get_user_by_username(self, db: Session, *, username: str) -> Optional[User]:
        """
        Get user by username
        
        Args:
            db: Database session
            username: Username
            
        Returns:
            User instance or None
        """
        return self.repository.get_by_username(db, username=username)
    
    def search_users(self, db: Session, *, query: str, skip: int = 0, limit: int = 100) -> List[User]:
        """
        Search users
        
        Args:
            db: Database session
            query: Search query
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of matching users
        """
        return self.repository.search_users(db, query=query, skip=skip, limit=limit)
    
    def activate_user(self, db: Session, *, user_id: str) -> Optional[User]:
        """
        Activate a user account
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None
        """
        return self.repository.activate_user(db, user_id=user_id)
    
    def deactivate_user(self, db: Session, *, user_id: str) -> Optional[User]:
        """
        Deactivate a user account
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None
        """
        return self.repository.deactivate_user(db, user_id=user_id)
    
    def update_user_login(self, db: Session, *, user_id: str) -> Optional[User]:
        """
        Update user's last login timestamp
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None
        """
        return self.repository.update_last_login(db, user_id=user_id)
    
    def change_password(self, db: Session, *, user_id: str, current_password: str, new_password: str) -> bool:
        """
        Change user password
        
        Args:
            db: Database session
            user_id: User ID
            current_password: Current password
            new_password: New password
            
        Returns:
            True if password changed successfully, False otherwise
        """
        user = self.repository.get(db, user_id)
        if not user:
            return False
        
        # Verify current password
        if not self.verify_password(current_password, user.hashed_password):
            return False
        
        # Update password
        hashed_new_password = self.get_password_hash(new_password)
        user.hashed_password = hashed_new_password
        user.updated_at = datetime.utcnow()
        
        db.commit()
        return True
    
    def is_email_available(self, db: Session, *, email: str) -> bool:
        """
        Check if email is available for registration
        
        Args:
            db: Database session
            email: Email to check
            
        Returns:
            True if email is available, False otherwise
        """
        user = self.repository.get_by_email(db, email=email)
        return user is None
    
    def is_username_available(self, db: Session, *, username: str) -> bool:
        """
        Check if username is available for registration
        
        Args:
            db: Database session
            username: Username to check
            
        Returns:
            True if username is available, False otherwise
        """
        user = self.repository.get_by_username(db, username=username)
        return user is None

# Create service instance
user_service = UserService()