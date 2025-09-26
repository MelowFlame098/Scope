from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import (
    Token,
    TokenRefresh,
    UserCreate,
    UserResponse,
    LoginRequest,
    PasswordChange,
    PasswordReset,
    PasswordResetConfirm,
    MessageResponse
)
from ..services.auth_service import AuthService
from ..services.user_service import UserService
from ..core.config import settings
from ..core.security import get_current_user
from ..models import User

router = APIRouter()


@router.post("/register", response_model=UserResponse)
def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
) -> Any:
    """
    Register a new user.
    """
    user_service = UserService(db)
    auth_service = AuthService(db)
    
    # Check if user already exists
    if user_service.get_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    if user_service.get_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user
    user = auth_service.create_user(user_data)
    return UserResponse.from_orm(user)


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests.
    """
    auth_service = AuthService(db)
    
    user = auth_service.authenticate_user(
        email_or_username=form_data.username,
        password=form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    access_token = auth_service.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    refresh_token = auth_service.create_refresh_token(
        data={"sub": user.username}, expires_delta=refresh_token_expires
    )
    
    # Update last login
    user_service = UserService(db)
    user_service.update_last_login(user.id)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post("/login/json", response_model=Token)
def login_json(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
) -> Any:
    """
    JSON login endpoint.
    """
    auth_service = AuthService(db)
    
    user = auth_service.authenticate_user(
        email_or_username=login_data.email_or_username,
        password=login_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email/username or password"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    refresh_token_expires = timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    
    access_token = auth_service.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    refresh_token = auth_service.create_refresh_token(
        data={"sub": user.username}, expires_delta=refresh_token_expires
    )
    
    # Update last login
    user_service = UserService(db)
    user_service.update_last_login(user.id)
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post("/refresh", response_model=Token)
def refresh_token(
    token_data: TokenRefresh,
    db: Session = Depends(get_db)
) -> Any:
    """
    Refresh access token using refresh token.
    """
    auth_service = AuthService(db)
    
    try:
        username = auth_service.verify_refresh_token(token_data.refresh_token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )
    
    user_service = UserService(db)
    user = user_service.get_by_username(username)
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth_service.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return Token(
        access_token=access_token,
        refresh_token=token_data.refresh_token,
        token_type="bearer"
    )


@router.post("/change-password", response_model=MessageResponse)
def change_password(
    password_data: PasswordChange,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Change user password.
    """
    auth_service = AuthService(db)
    user_service = UserService(db)
    
    # Verify current password
    if not auth_service.verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    user_service.change_password(current_user.id, password_data.new_password)
    
    return MessageResponse(
        message="Password changed successfully",
        success=True
    )


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(
    reset_data: PasswordReset,
    db: Session = Depends(get_db)
) -> Any:
    """
    Request password reset.
    """
    user_service = UserService(db)
    
    user = user_service.get_by_email(reset_data.email)
    if not user:
        # Don't reveal if email exists or not
        return MessageResponse(
            message="If the email exists, a reset link has been sent",
            success=True
        )
    
    # In a real application, you would:
    # 1. Generate a secure reset token
    # 2. Store it in the database with expiration
    # 3. Send email with reset link
    
    # For now, just return success message
    return MessageResponse(
        message="If the email exists, a reset link has been sent",
        success=True
    )


@router.post("/reset-password/confirm", response_model=MessageResponse)
def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    db: Session = Depends(get_db)
) -> Any:
    """
    Confirm password reset with token.
    """
    # In a real application, you would:
    # 1. Verify the reset token
    # 2. Check if it's not expired
    # 3. Update the user's password
    # 4. Invalidate the reset token
    
    # For now, just return success message
    return MessageResponse(
        message="Password reset successfully",
        success=True
    )


@router.get("/me", response_model=UserResponse)
def get_current_user_info(
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get current user information.
    """
    return UserResponse.from_orm(current_user)


@router.post("/logout", response_model=MessageResponse)
def logout(
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Logout user (invalidate token).
    In a real application, you might want to blacklist the token.
    """
    return MessageResponse(
        message="Successfully logged out",
        success=True
    )