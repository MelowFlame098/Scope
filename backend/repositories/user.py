from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from .base import BaseRepository
from ..models import User
from ..schemas import UserCreate, UserUpdate

class UserRepository(BaseRepository[User, UserCreate, UserUpdate]):
    """
    Repository for User model with specific business logic
    """
    
    def __init__(self):
        super().__init__(User)
    
    def get_by_email(self, db: Session, *, email: str) -> Optional[User]:
        """
        Get user by email address
        
        Args:
            db: Database session
            email: User email
            
        Returns:
            User instance or None
        """
        return db.query(User).filter(User.email == email).first()
    
    def get_by_username(self, db: Session, *, username: str) -> Optional[User]:
        """
        Get user by username
        
        Args:
            db: Database session
            username: Username
            
        Returns:
            User instance or None
        """
        return db.query(User).filter(User.username == username).first()
    
    def get_active_users(self, db: Session, *, skip: int = 0, limit: int = 100) -> List[User]:
        """
        Get all active users
        
        Args:
            db: Database session
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of active users
        """
        return db.query(User).filter(User.is_active == True).offset(skip).limit(limit).all()
    
    def search_users(self, db: Session, *, query: str, skip: int = 0, limit: int = 100) -> List[User]:
        """
        Search users by username, email, or full name
        
        Args:
            db: Database session
            query: Search query
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of matching users
        """
        search_filter = or_(
            User.username.ilike(f"%{query}%"),
            User.email.ilike(f"%{query}%"),
            User.full_name.ilike(f"%{query}%")
        )
        
        return (
            db.query(User)
            .filter(and_(User.is_active == True, search_filter))
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    def activate_user(self, db: Session, *, user_id: str) -> Optional[User]:
        """
        Activate a user account
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None
        """
        user = self.get(db, user_id)
        if user:
            user.is_active = True
            user.is_verified = True
            db.commit()
            db.refresh(user)
        return user
    
    def deactivate_user(self, db: Session, *, user_id: str) -> Optional[User]:
        """
        Deactivate a user account
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None
        """
        user = self.get(db, user_id)
        if user:
            user.is_active = False
            db.commit()
            db.refresh(user)
        return user
    
    def update_last_login(self, db: Session, *, user_id: str) -> Optional[User]:
        """
        Update user's last login timestamp
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            Updated user instance or None
        """
        from datetime import datetime
        
        user = self.get(db, user_id)
        if user:
            user.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(user)
        return user
    
    def get_users_by_risk_tolerance(self, db: Session, *, risk_level: str) -> List[User]:
        """
        Get users by risk tolerance level
        
        Args:
            db: Database session
            risk_level: Risk tolerance level (low, medium, high)
            
        Returns:
            List of users with specified risk tolerance
        """
        return db.query(User).filter(
            and_(
                User.is_active == True,
                User.risk_tolerance == risk_level
            )
        ).all()

# Create repository instance
user_repository = UserRepository()