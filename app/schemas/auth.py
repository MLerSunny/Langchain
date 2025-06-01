from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from .common import BaseResponse

class Token(BaseModel):
    """Token response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Type of token")
    expires_at: datetime = Field(..., description="Token expiration timestamp")
    refresh_token: Optional[str] = Field(None, description="Refresh token for obtaining new access tokens")

class TokenData(BaseModel):
    """Token payload data model."""
    sub: str = Field(..., description="Subject (user ID)")
    exp: datetime = Field(..., description="Expiration time")
    lob: Optional[str] = Field(None, description="Line of business")
    state: Optional[str] = Field(None, description="State information")
    permissions: Optional[list[str]] = Field(default=[], description="User permissions")

class UserBase(BaseModel):
    """Base user model."""
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., description="Username")
    full_name: Optional[str] = Field(None, description="User's full name")
    lob: Optional[str] = Field(None, description="Line of business")

class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8, description="User password")

class UserUpdate(BaseModel):
    """User update model."""
    email: Optional[EmailStr] = Field(None, description="User email address")
    username: Optional[str] = Field(None, description="Username")
    full_name: Optional[str] = Field(None, description="User's full name")
    password: Optional[str] = Field(None, min_length=8, description="User password")
    lob: Optional[str] = Field(None, description="Line of business")

class UserInDB(UserBase):
    """User model as stored in database."""
    id: str = Field(..., description="User ID")
    hashed_password: str = Field(..., description="Hashed password")
    is_active: bool = Field(default=True, description="Whether the user is active")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="User creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

class UserResponse(BaseResponse):
    """User response model."""
    user: UserInDB = Field(..., description="User information")

class TokenResponse(BaseResponse):
    """Token response model with metadata."""
    token: Token = Field(..., description="Authentication token information") 