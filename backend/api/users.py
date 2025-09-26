from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session

from ..database import get_db
from ..schemas import (
    UserResponse,
    UserUpdate,
    MessageResponse,
    PaginatedResponse
)
from ..services.user_service import UserService
from ..core.security import get_current_user, get_current_active_superuser
from ..models import User

router = APIRouter()


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(
    current_user: User = Depends(get_current_user)
) -> Any:
    """
    Get current user profile.
    """
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
def update_current_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update current user profile.
    """
    user_service = UserService(db)
    
    # Check if email is being changed and if it's already taken
    if user_update.email and user_update.email != current_user.email:
        existing_user = user_service.get_by_email(user_update.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Check if username is being changed and if it's already taken
    if user_update.username and user_update.username != current_user.username:
        existing_user = user_service.get_by_username(user_update.username)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    updated_user = user_service.update(current_user.id, user_update.dict(exclude_unset=True))
    return UserResponse.from_orm(updated_user)


@router.delete("/me", response_model=MessageResponse)
def delete_current_user(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Delete current user account.
    """
    user_service = UserService(db)
    user_service.delete(current_user.id)
    
    return MessageResponse(
        message="Account deleted successfully",
        success=True
    )


@router.post("/me/deactivate", response_model=MessageResponse)
def deactivate_current_user(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Any:
    """
    Deactivate current user account.
    """
    user_service = UserService(db)
    user_service.deactivate_user(current_user.id)
    
    return MessageResponse(
        message="Account deactivated successfully",
        success=True
    )


@router.get("/check-email", response_model=dict)
def check_email_availability(
    email: str = Query(..., description="Email to check"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Check if email is available.
    """
    user_service = UserService(db)
    is_available = user_service.is_email_available(email)
    
    return {
        "email": email,
        "available": is_available
    }


@router.get("/check-username", response_model=dict)
def check_username_availability(
    username: str = Query(..., description="Username to check"),
    db: Session = Depends(get_db)
) -> Any:
    """
    Check if username is available.
    """
    user_service = UserService(db)
    is_available = user_service.is_username_available(username)
    
    return {
        "username": username,
        "available": is_available
    }


# Admin endpoints
@router.get("/", response_model=PaginatedResponse)
def get_users(
    skip: int = Query(0, ge=0, description="Number of users to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of users to return"),
    search: str = Query(None, description="Search term for username or email"),
    is_active: bool = Query(None, description="Filter by active status"),
    current_user: User = Depends(get_current_active_superuser),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get all users (admin only).
    """
    user_service = UserService(db)
    
    filters = {}
    if is_active is not None:
        filters["is_active"] = is_active
    
    if search:
        users = user_service.search_users(search, skip=skip, limit=limit)
        total = len(users)  # This is not accurate for pagination, but works for demo
    else:
        users = user_service.get_multi(skip=skip, limit=limit, **filters)
        total = user_service.count(**filters)
    
    return PaginatedResponse(
        items=[UserResponse.from_orm(user) for user in users],
        total=total,
        page=skip // limit + 1,
        size=limit,
        pages=(total + limit - 1) // limit
    )


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    current_user: User = Depends(get_current_active_superuser),
    db: Session = Depends(get_db)
) -> Any:
    """
    Get user by ID (admin only).
    """
    user_service = UserService(db)
    user = user_service.get(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.from_orm(user)


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_superuser),
    db: Session = Depends(get_db)
) -> Any:
    """
    Update user by ID (admin only).
    """
    user_service = UserService(db)
    
    user = user_service.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Check if email is being changed and if it's already taken
    if user_update.email and user_update.email != user.email:
        existing_user = user_service.get_by_email(user_update.email)
        if existing_user and existing_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
    
    # Check if username is being changed and if it's already taken
    if user_update.username and user_update.username != user.username:
        existing_user = user_service.get_by_username(user_update.username)
        if existing_user and existing_user.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    updated_user = user_service.update(user_id, user_update.dict(exclude_unset=True))
    return UserResponse.from_orm(updated_user)


@router.delete("/{user_id}", response_model=MessageResponse)
def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_active_superuser),
    db: Session = Depends(get_db)
) -> Any:
    """
    Delete user by ID (admin only).
    """
    user_service = UserService(db)
    
    user = user_service.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user_service.delete(user_id)
    
    return MessageResponse(
        message="User deleted successfully",
        success=True
    )


@router.post("/{user_id}/activate", response_model=MessageResponse)
def activate_user(
    user_id: int,
    current_user: User = Depends(get_current_active_superuser),
    db: Session = Depends(get_db)
) -> Any:
    """
    Activate user by ID (admin only).
    """
    user_service = UserService(db)
    
    user = user_service.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user_service.activate_user(user_id)
    
    return MessageResponse(
        message="User activated successfully",
        success=True
    )


@router.post("/{user_id}/deactivate", response_model=MessageResponse)
def deactivate_user(
    user_id: int,
    current_user: User = Depends(get_current_active_superuser),
    db: Session = Depends(get_db)
) -> Any:
    """
    Deactivate user by ID (admin only).
    """
    user_service = UserService(db)
    
    user = user_service.get(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    user_service.deactivate_user(user_id)
    
    return MessageResponse(
        message="User deactivated successfully",
        success=True
    )