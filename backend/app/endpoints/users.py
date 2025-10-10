"""
User Endpoints - Handle user registration, authentication, and profile management
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, status, Depends, Form
from fastapi.responses import JSONResponse
from database.models import UserCreate, UserUpdate, UserResponse
from app.services.user_service import UserService

router = APIRouter()

@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """
    Register a new user
    """
    try:
        user = await UserService.create_user(user_data)
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login")
async def login_user(
    username: str = Form(...),
    password: str = Form(...)
):
    """
    Authenticate user with username and password
    """
    try:
        user = await UserService.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )
        
        return JSONResponse({
            "message": "Login successful",
            "user": user.dict(),
            "access_token": "dummy_token"  # In real app, generate JWT token
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@router.get("/profile/{user_id}", response_model=UserResponse)
async def get_user_profile(user_id: str):
    """
    Get user profile by ID
    """
    try:
        user = await UserService.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user profile: {str(e)}"
        )

@router.put("/profile/{user_id}", response_model=UserResponse)
async def update_user_profile(user_id: str, user_data: UserUpdate):
    """
    Update user profile
    """
    try:
        user = await UserService.update_user(user_id, user_data)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user profile: {str(e)}"
        )

@router.get("/users", response_model=List[UserResponse])
async def get_all_users(skip: int = 0, limit: int = 100):
    """
    Get all users (admin endpoint)
    """
    try:
        users = await UserService.get_all_users(skip, limit)
        return users
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get users: {str(e)}"
        )

@router.delete("/users/{user_id}")
async def delete_user(user_id: str):
    """
    Delete user (soft delete)
    """
    try:
        success = await UserService.delete_user(user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return JSONResponse({
            "message": "User deleted successfully"
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}"
        )

@router.get("/users/{user_id}/eeg-files")
async def get_user_eeg_files(user_id: str):
    """
    Get user's EEG files
    """
    try:
        user = await UserService.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return JSONResponse({
            "user_id": user_id,
            "eeg_files": user.eeg_files,
            "total_files": len(user.eeg_files)
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user EEG files: {str(e)}"
        )
