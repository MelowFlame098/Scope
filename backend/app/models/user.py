"""User-related database models.

This module contains models for:
- User accounts and authentication
- User profiles and personal information
- User settings and preferences
- User sessions and API keys
"""

from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, String, Boolean, DateTime, Text, Integer,
    ForeignKey, Enum as SQLEnum, UniqueConstraint, Index
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship, validates
from datetime import datetime
import uuid
from enum import Enum
import re

from app.models.base import BaseModel, SoftDeleteModel, AuditModel
from app.core.auth import UserRole, AuthProvider


class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING_VERIFICATION = "pending_verification"
    LOCKED = "locked"


class NotificationPreference(str, Enum):
    """Notification preferences."""
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    IN_APP = "in_app"
    NONE = "none"


class ThemePreference(str, Enum):
    """UI theme preferences."""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"


class User(SoftDeleteModel):
    """User account model."""
    
    __tablename__ = "users"
    
    # Authentication fields
    email = Column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        doc="User email address (unique)"
    )
    
    password_hash = Column(
        String(255),
        nullable=True,  # Nullable for OAuth users
        doc="Hashed password"
    )
    
    # Basic information
    first_name = Column(
        String(100),
        nullable=False,
        doc="User first name"
    )
    
    last_name = Column(
        String(100),
        nullable=False,
        doc="User last name"
    )
    
    # Account status
    status = Column(
        SQLEnum(UserStatus),
        default=UserStatus.PENDING_VERIFICATION,
        nullable=False,
        doc="Account status"
    )
    
    role = Column(
        SQLEnum(UserRole),
        default=UserRole.STANDARD,
        nullable=False,
        doc="User role for authorization"
    )
    
    # Verification and security
    is_verified = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Email verification status"
    )
    
    verification_token = Column(
        String(255),
        nullable=True,
        doc="Email verification token"
    )
    
    verification_expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Verification token expiration"
    )
    
    # Password reset
    reset_token = Column(
        String(255),
        nullable=True,
        doc="Password reset token"
    )
    
    reset_expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Reset token expiration"
    )
    
    # Authentication provider
    auth_provider = Column(
        SQLEnum(AuthProvider),
        default=AuthProvider.LOCAL,
        nullable=False,
        doc="Authentication provider"
    )
    
    provider_id = Column(
        String(255),
        nullable=True,
        doc="External provider user ID"
    )
    
    # Activity tracking
    last_login = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last login timestamp"
    )
    
    last_activity = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last activity timestamp"
    )
    
    login_count = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Total login count"
    )
    
    # Failed login tracking
    failed_login_attempts = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Failed login attempts counter"
    )
    
    locked_until = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Account lock expiration"
    )
    
    # Subscription and billing
    subscription_tier = Column(
        String(50),
        default="free",
        nullable=False,
        doc="Subscription tier"
    )
    
    subscription_expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Subscription expiration"
    )
    
    # Relationships
    profile = relationship(
        "UserProfile",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    settings = relationship(
        "UserSettings",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan"
    )
    
    api_keys = relationship(
        "UserAPIKey",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    sessions = relationship(
        "UserSession",
        back_populates="user",
        cascade="all, delete-orphan"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email_status', 'email', 'status'),
        Index('idx_user_provider', 'auth_provider', 'provider_id'),
        Index('idx_user_verification', 'verification_token'),
        Index('idx_user_reset', 'reset_token'),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        """Validate email format.
        
        Args:
            key: Field name
            email: Email value
            
        Returns:
            Validated email
            
        Raises:
            ValueError: If email format is invalid
        """
        if not email:
            raise ValueError("Email is required")
        
        # Basic email validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, email):
            raise ValueError("Invalid email format")
        
        return email.lower().strip()
    
    @validates('first_name', 'last_name')
    def validate_name(self, key, name):
        """Validate name fields.
        
        Args:
            key: Field name
            name: Name value
            
        Returns:
            Validated name
            
        Raises:
            ValueError: If name is invalid
        """
        if not name or not name.strip():
            raise ValueError(f"{key} is required")
        
        name = name.strip()
        if len(name) < 1 or len(name) > 100:
            raise ValueError(f"{key} must be between 1 and 100 characters")
        
        return name
    
    @property
    def full_name(self) -> str:
        """Get full name.
        
        Returns:
            Full name string
        """
        return f"{self.first_name} {self.last_name}"
    
    @property
    def is_active(self) -> bool:
        """Check if user is active.
        
        Returns:
            True if user is active
        """
        return (
            self.status == UserStatus.ACTIVE and
            not self.is_deleted and
            (not self.locked_until or self.locked_until < datetime.utcnow())
        )
    
    @property
    def is_premium(self) -> bool:
        """Check if user has premium subscription.
        
        Returns:
            True if user has premium access
        """
        return (
            self.role in [UserRole.PREMIUM, UserRole.ADMIN] or
            (
                self.subscription_tier in ["premium", "enterprise"] and
                (
                    not self.subscription_expires_at or
                    self.subscription_expires_at > datetime.utcnow()
                )
            )
        )
    
    def update_last_login(self) -> None:
        """Update last login timestamp and increment login count."""
        self.last_login = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.login_count += 1
        self.failed_login_attempts = 0  # Reset failed attempts on successful login
    
    def update_last_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def increment_failed_login(self, max_attempts: int = 5, lockout_duration: int = 30) -> None:
        """Increment failed login attempts and lock account if needed.
        
        Args:
            max_attempts: Maximum failed attempts before lockout
            lockout_duration: Lockout duration in minutes
        """
        self.failed_login_attempts += 1
        
        if self.failed_login_attempts >= max_attempts:
            self.locked_until = datetime.utcnow() + timedelta(minutes=lockout_duration)
            self.status = UserStatus.LOCKED
    
    def unlock_account(self) -> None:
        """Unlock user account."""
        self.locked_until = None
        self.failed_login_attempts = 0
        if self.status == UserStatus.LOCKED:
            self.status = UserStatus.ACTIVE
    
    def verify_email(self) -> None:
        """Mark email as verified."""
        self.is_verified = True
        self.verification_token = None
        self.verification_expires_at = None
        if self.status == UserStatus.PENDING_VERIFICATION:
            self.status = UserStatus.ACTIVE
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return f"<User(id={self.id}, email={self.email}, role={self.role.value})>"


class UserProfile(BaseModel):
    """User profile with additional information."""
    
    __tablename__ = "user_profiles"
    
    # Foreign key to user
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        doc="Reference to user"
    )
    
    # Personal information
    phone = Column(
        String(20),
        nullable=True,
        doc="Phone number"
    )
    
    date_of_birth = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Date of birth"
    )
    
    country = Column(
        String(100),
        nullable=True,
        doc="Country of residence"
    )
    
    timezone = Column(
        String(50),
        default="UTC",
        nullable=False,
        doc="User timezone"
    )
    
    language = Column(
        String(10),
        default="en",
        nullable=False,
        doc="Preferred language"
    )
    
    # Professional information
    occupation = Column(
        String(100),
        nullable=True,
        doc="User occupation"
    )
    
    company = Column(
        String(100),
        nullable=True,
        doc="Company name"
    )
    
    experience_level = Column(
        String(20),
        nullable=True,
        doc="Trading/investing experience level"
    )
    
    # Profile customization
    avatar_url = Column(
        String(500),
        nullable=True,
        doc="Profile avatar URL"
    )
    
    bio = Column(
        Text,
        nullable=True,
        doc="User biography"
    )
    
    # Social links
    website = Column(
        String(500),
        nullable=True,
        doc="Personal website"
    )
    
    linkedin = Column(
        String(500),
        nullable=True,
        doc="LinkedIn profile"
    )
    
    twitter = Column(
        String(500),
        nullable=True,
        doc="Twitter profile"
    )
    
    # Investment preferences
    risk_tolerance = Column(
        String(20),
        nullable=True,
        doc="Risk tolerance level"
    )
    
    investment_goals = Column(
        JSONB,
        nullable=True,
        doc="Investment goals and objectives"
    )
    
    # Relationship
    user = relationship("User", back_populates="profile")
    
    @validates('phone')
    def validate_phone(self, key, phone):
        """Validate phone number format.
        
        Args:
            key: Field name
            phone: Phone number
            
        Returns:
            Validated phone number
        """
        if phone:
            # Remove all non-digit characters
            phone = re.sub(r'\D', '', phone)
            if len(phone) < 10 or len(phone) > 15:
                raise ValueError("Invalid phone number length")
        return phone
    
    @validates('website', 'linkedin', 'twitter')
    def validate_url(self, key, url):
        """Validate URL format.
        
        Args:
            key: Field name
            url: URL value
            
        Returns:
            Validated URL
        """
        if url:
            url = url.strip()
            if not url.startswith(('http://', 'https://')):
                url = f"https://{url}"
            
            # Basic URL validation
            url_pattern = r'^https?://[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            if not re.match(url_pattern, url):
                raise ValueError(f"Invalid {key} URL format")
        
        return url
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return f"<UserProfile(user_id={self.user_id})>"


class UserSettings(BaseModel):
    """User application settings and preferences."""
    
    __tablename__ = "user_settings"
    
    # Foreign key to user
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        doc="Reference to user"
    )
    
    # UI preferences
    theme = Column(
        SQLEnum(ThemePreference),
        default=ThemePreference.AUTO,
        nullable=False,
        doc="UI theme preference"
    )
    
    dashboard_layout = Column(
        JSONB,
        nullable=True,
        doc="Dashboard layout configuration"
    )
    
    default_currency = Column(
        String(3),
        default="USD",
        nullable=False,
        doc="Default currency code"
    )
    
    # Notification preferences
    email_notifications = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Enable email notifications"
    )
    
    push_notifications = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Enable push notifications"
    )
    
    sms_notifications = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Enable SMS notifications"
    )
    
    notification_preferences = Column(
        JSONB,
        nullable=True,
        doc="Detailed notification preferences"
    )
    
    # Trading preferences
    auto_refresh_interval = Column(
        Integer,
        default=30,
        nullable=False,
        doc="Auto refresh interval in seconds"
    )
    
    default_chart_timeframe = Column(
        String(10),
        default="1D",
        nullable=False,
        doc="Default chart timeframe"
    )
    
    trading_preferences = Column(
        JSONB,
        nullable=True,
        doc="Trading-specific preferences"
    )
    
    # Privacy settings
    profile_visibility = Column(
        String(20),
        default="private",
        nullable=False,
        doc="Profile visibility setting"
    )
    
    share_portfolio = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Allow portfolio sharing"
    )
    
    # API settings
    api_access_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Enable API access"
    )
    
    api_rate_limit = Column(
        Integer,
        default=1000,
        nullable=False,
        doc="API rate limit per hour"
    )
    
    # Feature flags
    beta_features_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Enable beta features"
    )
    
    ai_features_enabled = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Enable AI features"
    )
    
    defi_features_enabled = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Enable DeFi features"
    )
    
    # Relationship
    user = relationship("User", back_populates="settings")
    
    def get_notification_preference(self, notification_type: str) -> bool:
        """Get notification preference for specific type.
        
        Args:
            notification_type: Type of notification
            
        Returns:
            True if notifications are enabled for this type
        """
        if not self.notification_preferences:
            return True  # Default to enabled
        
        return self.notification_preferences.get(notification_type, True)
    
    def set_notification_preference(self, notification_type: str, enabled: bool) -> None:
        """Set notification preference for specific type.
        
        Args:
            notification_type: Type of notification
            enabled: Whether to enable notifications
        """
        if not self.notification_preferences:
            self.notification_preferences = {}
        
        self.notification_preferences[notification_type] = enabled
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return f"<UserSettings(user_id={self.user_id})>"


class UserAPIKey(BaseModel):
    """User API keys for programmatic access."""
    
    __tablename__ = "user_api_keys"
    
    # Foreign key to user
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    # API key information
    name = Column(
        String(100),
        nullable=False,
        doc="API key name/description"
    )
    
    key_hash = Column(
        String(255),
        nullable=False,
        unique=True,
        doc="Hashed API key"
    )
    
    key_prefix = Column(
        String(10),
        nullable=False,
        doc="API key prefix for identification"
    )
    
    # Permissions and limits
    scopes = Column(
        JSONB,
        nullable=False,
        default=list,
        doc="API key scopes/permissions"
    )
    
    rate_limit = Column(
        Integer,
        default=1000,
        nullable=False,
        doc="Rate limit per hour"
    )
    
    # Status and expiration
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="API key active status"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="API key expiration"
    )
    
    # Usage tracking
    last_used = Column(
        DateTime(timezone=True),
        nullable=True,
        doc="Last usage timestamp"
    )
    
    usage_count = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Total usage count"
    )
    
    # Relationship
    user = relationship("User", back_populates="api_keys")
    
    @property
    def is_expired(self) -> bool:
        """Check if API key is expired.
        
        Returns:
            True if API key is expired
        """
        return (
            self.expires_at is not None and
            self.expires_at < datetime.utcnow()
        )
    
    @property
    def is_valid(self) -> bool:
        """Check if API key is valid for use.
        
        Returns:
            True if API key is valid
        """
        return self.is_active and not self.is_expired
    
    def update_usage(self) -> None:
        """Update API key usage statistics."""
        self.last_used = datetime.utcnow()
        self.usage_count += 1
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return f"<UserAPIKey(id={self.id}, name={self.name}, user_id={self.user_id})>"


class UserSession(BaseModel):
    """User session tracking."""
    
    __tablename__ = "user_sessions"
    
    # Foreign key to user
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to user"
    )
    
    # Session information
    session_token = Column(
        String(255),
        nullable=False,
        unique=True,
        doc="Session token"
    )
    
    # Client information
    ip_address = Column(
        String(45),
        nullable=True,
        doc="Client IP address"
    )
    
    user_agent = Column(
        Text,
        nullable=True,
        doc="Client user agent"
    )
    
    device_info = Column(
        JSONB,
        nullable=True,
        doc="Device information"
    )
    
    # Session status
    is_active = Column(
        Boolean,
        default=True,
        nullable=False,
        doc="Session active status"
    )
    
    expires_at = Column(
        DateTime(timezone=True),
        nullable=False,
        doc="Session expiration"
    )
    
    # Activity tracking
    last_activity = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        doc="Last activity timestamp"
    )
    
    # Relationship
    user = relationship("User", back_populates="sessions")
    
    # Indexes
    __table_args__ = (
        Index('idx_session_token', 'session_token'),
        Index('idx_session_user_active', 'user_id', 'is_active'),
        Index('idx_session_expires', 'expires_at'),
    )
    
    @property
    def is_expired(self) -> bool:
        """Check if session is expired.
        
        Returns:
            True if session is expired
        """
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_valid(self) -> bool:
        """Check if session is valid.
        
        Returns:
            True if session is valid
        """
        return self.is_active and not self.is_expired
    
    def update_activity(self) -> None:
        """Update session activity timestamp."""
        self.last_activity = datetime.utcnow()
    
    def invalidate(self) -> None:
        """Invalidate the session."""
        self.is_active = False
    
    def __repr__(self) -> str:
        """String representation.
        
        Returns:
            String representation
        """
        return f"<UserSession(id={self.id}, user_id={self.user_id}, active={self.is_active})>"