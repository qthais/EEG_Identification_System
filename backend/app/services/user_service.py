"""
User Service - Handles all user-related database operations
"""
from typing import Optional, List
from datetime import datetime
from bson import ObjectId
from passlib.context import CryptContext
from fastapi import HTTPException, status
from database.db import user_collection
from database.models import User, UserCreate, UserUpdate, UserResponse

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserService:
    """Service class for user operations"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    @staticmethod
    async def create_user(user_data: UserCreate) -> UserResponse:
        """Create a new user"""
        # Check if username already exists
        existing_user = await user_collection.find_one({"username": user_data.username})
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )
        
        # Check if email already exists
        existing_email = await user_collection.find_one({"email": user_data.email})
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Hash password
        hashed_password = UserService.get_password_hash(user_data.password)
        
        # Create user document
        user_doc = {
            "username": user_data.username,
            "email": user_data.email,
            "password": hashed_password,
            "full_name": user_data.full_name,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "eeg_files": [],
            "is_active": True
        }
        
        # Insert user
        result = await user_collection.insert_one(user_doc)
        
        # Return user without password
        user_doc["_id"] = str(result.inserted_id)
        return UserResponse(**user_doc)
    
    @staticmethod
    async def get_user_by_id(user_id: str) -> Optional[UserResponse]:
        """Get user by ID"""
        if not ObjectId.is_valid(user_id):
            return None
        
        user = await user_collection.find_one({"_id": ObjectId(user_id)})
        if not user:
            return None
        
        user["_id"] = str(user["_id"])
        return UserResponse(**user)
    
    @staticmethod
    async def get_user_by_username(username: str) -> Optional[User]:
        """Get user by username (includes password for authentication)"""
        user = await user_collection.find_one({"username": username})
        if not user:
            return None
        
        user["_id"] = str(user["_id"])
        return User(**user)
    
    @staticmethod
    async def get_user_by_email(email: str) -> Optional[UserResponse]:
        """Get user by email"""
        user = await user_collection.find_one({"email": email})
        if not user:
            return None
        
        user["_id"] = str(user["_id"])
        return UserResponse(**user)
    
    @staticmethod
    async def update_user(user_id: str, user_data: UserUpdate) -> Optional[UserResponse]:
        """Update user information"""
        if not ObjectId.is_valid(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user ID"
            )
        
        # Check if user exists
        existing_user = await user_collection.find_one({"_id": ObjectId(user_id)})
        if not existing_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Prepare update data
        update_data = {"updated_at": datetime.utcnow()}
        
        if user_data.email is not None:
            # Check if email is already taken by another user
            email_user = await user_collection.find_one({
                "email": user_data.email,
                "_id": {"$ne": ObjectId(user_id)}
            })
            if email_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            update_data["email"] = user_data.email
        
        if user_data.full_name is not None:
            update_data["full_name"] = user_data.full_name
        
        if user_data.is_active is not None:
            update_data["is_active"] = user_data.is_active
        
        # Update user
        await user_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_data}
        )
        
        # Return updated user
        return await UserService.get_user_by_id(user_id)
    
    @staticmethod
    async def add_eeg_file_to_user(user_id: str, filename: str) -> bool:
        """Add EEG filename to user's eeg_files list"""
        if not ObjectId.is_valid(user_id):
            return False
        
        result = await user_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$addToSet": {"eeg_files": filename},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return result.modified_count > 0
    
    @staticmethod
    async def remove_eeg_file_from_user(user_id: str, filename: str) -> bool:
        """Remove EEG filename from user's eeg_files list"""
        if not ObjectId.is_valid(user_id):
            return False
        
        result = await user_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$pull": {"eeg_files": filename},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        return result.modified_count > 0
    
    @staticmethod
    async def authenticate_user(username: str, password: str) -> Optional[UserResponse]:
        """Authenticate user with username and password"""
        user = await UserService.get_user_by_username(username)
        if not user:
            return None
        
        if not UserService.verify_password(password, user.password):
            return None
        
        if not user.is_active:
            return None
        
        # Return user without password
        user_dict = user.dict()
        user_dict.pop("password", None)
        return UserResponse(**user_dict)
    
    @staticmethod
    async def get_all_users(skip: int = 0, limit: int = 100) -> List[UserResponse]:
        """Get all users with pagination"""
        users = []
        cursor = user_collection.find().skip(skip).limit(limit)
        
        async for user in cursor:
            user["_id"] = str(user["_id"])
            users.append(UserResponse(**user))
        
        return users
    
    @staticmethod
    async def delete_user(user_id: str) -> bool:
        """Delete a user (soft delete by setting is_active to False)"""
        if not ObjectId.is_valid(user_id):
            return False
        
        result = await user_collection.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "is_active": False,
                    "updated_at": datetime.utcnow()
                }
            }
        )
        
        return result.modified_count > 0
