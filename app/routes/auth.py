from fastapi import APIRouter, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from core.settings import settings
from app.schemas.auth import (
    Token,
    TokenData,
    UserCreate,
    UserUpdate,
    UserResponse,
    TokenResponse
)
from app.schemas.errors import (
    AuthenticationErrorResponse,
    AuthorizationErrorResponse,
    ValidationErrorResponse
)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
    responses={
        401: {"model": AuthenticationErrorResponse},
        403: {"model": AuthorizationErrorResponse},
        422: {"model": ValidationErrorResponse}
    }
)

logger = logging.getLogger(__name__)
security = HTTPBearer()

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> Token:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm
    )
    return Token(
        access_token=encoded_jwt,
        token_type="bearer",
        expires_at=expire
    )

@router.post("/token", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def login_for_access_token(
    request: Request,
    user_data: UserCreate
) -> TokenResponse:
    """
    Get access token for user authentication.
    
    Args:
        user_data: User credentials for authentication
        
    Returns:
        TokenResponse: Access token and metadata
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Validate user credentials (implement your authentication logic here)
        # This is a placeholder - implement actual user validation
        if not user_data.email or not user_data.password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
            
        # Create access token
        token = create_access_token(
            data={"sub": user_data.email},
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        
        return TokenResponse(
            status="success",
            token=token,
            metadata={
                "user_email": user_data.email,
                "token_type": "bearer"
            }
        )
        
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

@router.post("/refresh", response_model=TokenResponse, status_code=status.HTTP_200_OK)
async def refresh_token(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> TokenResponse:
    """
    Refresh access token using refresh token.
    
    Args:
        credentials: Current access token
        
    Returns:
        TokenResponse: New access token and metadata
        
    Raises:
        HTTPException: If token refresh fails
    """
    try:
        # Decode and validate current token
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        
        # Create new token
        token = create_access_token(
            data={"sub": payload["sub"]},
            expires_delta=timedelta(minutes=settings.access_token_expire_minutes)
        )
        
        return TokenResponse(
            status="success",
            token=token,
            metadata={
                "user_email": payload["sub"],
                "token_type": "bearer"
            }
        )
        
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        ) 