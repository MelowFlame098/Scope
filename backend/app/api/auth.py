"""Enhanced authentication API with email verification and CAPTCHA.

This module provides:
- User registration with email verification
- Login with CAPTCHA protection
- Password reset functionality
- Multi-factor authentication
- Rate limiting and security features
"""

from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
from sqlalchemy.orm import Session
import secrets
import asyncio

from app.core.dependencies import get_db, get_current_user
from app.core.auth import AuthenticationManager, UserRole, LoginRequest, RegisterRequest, TokenResponse
from app.core.exceptions import AuthenticationException, ValidationException
from app.core.rate_limiting import RateLimiter
from app.core.logging_config import get_logger
from app.services.email_service import get_email_service
from app.services.captcha_service import get_captcha_service, CaptchaType
from app.models.user import User, UserProfile, UserSettings
from app.config.settings import get_settings

logger = get_logger("auth_api")
settings = get_settings()
auth_manager = AuthenticationManager()
security = HTTPBearer()
rate_limiter = RateLimiter()

router = APIRouter(prefix="/auth", tags=["authentication"])


class EnhancedRegisterRequest(BaseModel):
    """Enhanced registration request with CAPTCHA."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    confirm_password: str
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    terms_accepted: bool = Field(..., description="Must accept terms")
    privacy_accepted: bool = Field(..., description="Must accept privacy policy")
    
    # CAPTCHA fields
    captcha_type: str = Field(..., description="CAPTCHA type")
    captcha_response: Optional[str] = None
    captcha_challenge_id: Optional[str] = None
    recaptcha_token: Optional[str] = None
    
    @validator("confirm_password")
    def passwords_match(cls, v, values):
        if "password" in values and v != values["password"]:
            raise ValueError("Passwords do not match")
        return v
    
    @validator("terms_accepted")
    def terms_must_be_accepted(cls, v):
        if not v:
            raise ValueError("Terms of service must be accepted")
        return v
    
    @validator("privacy_accepted")
    def privacy_must_be_accepted(cls, v):
        if not v:
            raise ValueError("Privacy policy must be accepted")
        return v


class EnhancedLoginRequest(BaseModel):
    """Enhanced login request with CAPTCHA."""
    email: EmailStr
    password: str
    remember_me: bool = False
    
    # CAPTCHA fields (required after failed attempts)
    captcha_type: Optional[str] = None
    captcha_response: Optional[str] = None
    captcha_challenge_id: Optional[str] = None
    recaptcha_token: Optional[str] = None


class EmailVerificationRequest(BaseModel):
    """Email verification request."""
    token: str = Field(..., description="Verification token")


class ResendVerificationRequest(BaseModel):
    """Resend verification email request."""
    email: EmailStr = Field(..., description="Email address to resend verification to")


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr
    captcha_type: str
    captcha_response: Optional[str] = None
    captcha_challenge_id: Optional[str] = None
    recaptcha_token: Optional[str] = None


class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation request."""
    token: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator("confirm_password")
    def passwords_match(cls, v, values):
        if "password" in values and v != values["new_password"]:
            raise ValueError("Passwords do not match")
        return v


class UserResponse(BaseModel):
    """User response model."""
    id: str
    email: str
    first_name: str
    last_name: str
    role: UserRole
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    subscription_tier: Optional[str]


@router.post("/captcha/challenge", summary="Create CAPTCHA challenge")
async def create_captcha_challenge(
    captcha_type: str,
    captcha_service = Depends(get_captcha_service)
):
    """Create a new CAPTCHA challenge."""
    try:
        if captcha_type not in [CaptchaType.IMAGE, CaptchaType.MATH]:
            raise HTTPException(
                status_code=400,
                detail="Unsupported CAPTCHA type"
            )
        
        result = await captcha_service.create_challenge(captcha_type)
        if not result.success:
            raise HTTPException(
                status_code=500,
                detail=result.error_message or "Failed to create CAPTCHA challenge"
            )
        
        return {
            "challenge_id": result.challenge_id,
            "challenge_data": result.challenge_data
        }
    except Exception as e:
        logger.error(f"Failed to create CAPTCHA challenge: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )


@router.post("/register", response_model=Dict[str, str], summary="Register new user")
async def register(
    request: Request,
    user_data: EnhancedRegisterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    email_service = Depends(get_email_service),
    captcha_service = Depends(get_captcha_service)
):
    """Register a new user with email verification and CAPTCHA."""
    try:
        # Rate limiting
        client_ip = request.client.host
        await rate_limiter.check_rate_limit(
            f"register:{client_ip}",
            max_requests=5,
            window_seconds=3600  # 5 registrations per hour
        )
        
        # Verify CAPTCHA
        captcha_verified = False
        if user_data.captcha_type == CaptchaType.RECAPTCHA_V2:
            if not user_data.recaptcha_token:
                raise HTTPException(status_code=400, detail="reCAPTCHA token required")
            
            result = await captcha_service.verify_recaptcha_v2(
                user_data.recaptcha_token,
                client_ip
            )
            captcha_verified = result.success
        
        elif user_data.captcha_type == CaptchaType.RECAPTCHA_V3:
            if not user_data.recaptcha_token:
                raise HTTPException(status_code=400, detail="reCAPTCHA token required")
            
            result = await captcha_service.verify_recaptcha_v3(
                user_data.recaptcha_token,
                "register",
                client_ip
            )
            captcha_verified = result.success
        
        elif user_data.captcha_type in [CaptchaType.IMAGE, CaptchaType.MATH]:
            if not user_data.captcha_challenge_id or not user_data.captcha_response:
                raise HTTPException(status_code=400, detail="CAPTCHA challenge ID and response required")
            
            result = await captcha_service.verify_challenge(
                user_data.captcha_challenge_id,
                user_data.captcha_response
            )
            captcha_verified = result.success
        
        if not captcha_verified:
            raise HTTPException(
                status_code=400,
                detail="CAPTCHA verification failed"
            )
        
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="User with this email already exists"
            )
        
        # Create user
        user = User(
            email=user_data.email,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            role=UserRole.STANDARD,
            is_active=True,
            is_verified=False,
            subscription_tier="free"
        )
        
        # Set password
        user.password_hash = auth_manager.get_password_hash(user_data.password)
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        # Create user profile
        profile = UserProfile(
            user_id=user.id,
            timezone="UTC",
            language="en",
            country="US"
        )
        db.add(profile)
        
        # Create user settings
        settings_obj = UserSettings(
            user_id=user.id,
            theme="light",
            notifications_enabled=True,
            email_notifications=True,
            push_notifications=True,
            trading_notifications=True
        )
        db.add(settings_obj)
        db.commit()
        
        # Generate verification token
        verification_token = email_service.generate_verification_token(str(user.id), user.email)
        await email_service.store_verification_token(str(user.id), user.email, verification_token)
        
        # Send verification email in background
        verification_url = f"{settings.frontend_url}/verify-email?token={verification_token}"
        background_tasks.add_task(
            email_service.send_verification_email,
            user,
            verification_url
        )
        
        logger.info(f"User registered successfully: {user.email}")
        
        return {
            "message": "Registration successful. Please check your email for verification.",
            "user_id": str(user.id)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="Registration failed. Please try again."
        )


@router.post("/verify-email", summary="Verify email address")
async def verify_email(
    request: EmailVerificationRequest,
    db: Session = Depends(get_db),
    email_service = Depends(get_email_service)
):
    """Verify user email address."""
    try:
        # Verify token
        token_data = await email_service.verify_email_token(request.token)
        if not token_data:
            raise HTTPException(
                status_code=400,
                detail="Invalid or expired verification token"
            )
        
        # Get user
        user = db.query(User).filter(User.id == token_data["user_id"]).first()
        if not user:
            raise HTTPException(
                status_code=404,
                detail="User not found"
            )
        
        if user.email != token_data["email"]:
            raise HTTPException(
                status_code=400,
                detail="Email mismatch"
            )
        
        # Update user verification status
        user.is_verified = True
        user.email_verified_at = datetime.utcnow()
        db.commit()
        
        logger.info(f"Email verified successfully: {user.email}")
        
        return {"message": "Email verified successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Email verification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Email verification failed"
        )


@router.post("/resend-verification", summary="Resend verification email")
async def resend_verification_email(
    request: ResendVerificationRequest,
    db: Session = Depends(get_db),
    email_service = Depends(get_email_service)
):
    """Resend email verification email."""
    try:
        # Get user by email
        user = db.query(User).filter(User.email == request.email).first()
        if not user:
            # Don't reveal if email exists or not for security
            return {"message": "If the email exists and is not verified, a verification email has been sent"}
        
        # Check if user is already verified
        if user.is_verified:
            return {"message": "Email is already verified"}
        
        # Generate new verification token
        token = email_service.generate_verification_token(str(user.id), user.email)
        
        # Store token in Redis
        await email_service.store_verification_token(str(user.id), user.email, token)
        
        # Create verification URL
        verification_url = f"{settings.frontend_url}/auth/verify-email?token={token}&email={user.email}"
        
        # Send verification email
        await email_service.send_verification_email(user, verification_url)
        
        logger.info(f"Verification email resent to {user.email}")
        
        return {"message": "If the email exists and is not verified, a verification email has been sent"}
    
    except Exception as e:
        logger.error(f"Failed to resend verification email: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to resend verification email"
        )


@router.post("/login", response_model=TokenResponse, summary="User login")
async def login(
    request: Request,
    login_data: EnhancedLoginRequest,
    db: Session = Depends(get_db),
    captcha_service = Depends(get_captcha_service)
):
    """Authenticate user and return tokens."""
    try:
        client_ip = request.client.host
        
        # Rate limiting
        await rate_limiter.check_rate_limit(
            f"login:{client_ip}",
            max_requests=10,
            window_seconds=900  # 10 attempts per 15 minutes
        )
        
        # Check if CAPTCHA is required (after failed attempts)
        failed_attempts = await rate_limiter.get_attempt_count(f"login_failed:{login_data.email}")
        captcha_required = failed_attempts >= 3
        
        if captcha_required:
            if not login_data.captcha_type:
                raise HTTPException(
                    status_code=400,
                    detail="CAPTCHA verification required after multiple failed attempts"
                )
            
            # Verify CAPTCHA
            captcha_verified = False
            if login_data.captcha_type == CaptchaType.RECAPTCHA_V2:
                if not login_data.recaptcha_token:
                    raise HTTPException(status_code=400, detail="reCAPTCHA token required")
                
                result = await captcha_service.verify_recaptcha_v2(
                    login_data.recaptcha_token,
                    client_ip
                )
                captcha_verified = result.success
            
            elif login_data.captcha_type in [CaptchaType.IMAGE, CaptchaType.MATH]:
                if not login_data.captcha_challenge_id or not login_data.captcha_response:
                    raise HTTPException(status_code=400, detail="CAPTCHA challenge ID and response required")
                
                result = await captcha_service.verify_challenge(
                    login_data.captcha_challenge_id,
                    login_data.captcha_response
                )
                captcha_verified = result.success
            
            if not captcha_verified:
                raise HTTPException(
                    status_code=400,
                    detail="CAPTCHA verification failed"
                )
        
        # Authenticate user
        try:
            user = auth_manager.authenticate_user(db, login_data.email, login_data.password)
        except AuthenticationException as e:
            # Increment failed attempts
            await rate_limiter.increment_attempt(f"login_failed:{login_data.email}", expire_seconds=3600)
            raise HTTPException(
                status_code=401,
                detail=str(e)
            )
        
        # Check if email is verified
        if not user.is_verified:
            raise HTTPException(
                status_code=401,
                detail="Please verify your email address before logging in"
            )
        
        # Generate tokens
        access_token = auth_manager.create_access_token(
            data={"sub": str(user.id), "email": user.email, "role": user.role}
        )
        
        refresh_token = None
        if login_data.remember_me:
            refresh_token = auth_manager.create_refresh_token(
                data={"sub": str(user.id), "email": user.email}
            )
        
        # Update last login
        user.last_login = datetime.utcnow()
        user.login_count = (user.login_count or 0) + 1
        db.commit()
        
        # Clear failed attempts
        await rate_limiter.clear_attempts(f"login_failed:{login_data.email}")
        
        logger.info(f"User logged in successfully: {user.email}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.security.access_token_expire_minutes * 60
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Login failed. Please try again."
        )


@router.get("/me", response_model=UserResponse, summary="Get current user")
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current authenticated user information."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        role=current_user.role,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        subscription_tier=current_user.subscription_tier
    )