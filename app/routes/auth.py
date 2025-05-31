from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from core.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)

security = HTTPBearer()

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
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
    return encoded_jwt

async def validate_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Validate JWT token and return user data."""
    try:
        token = credentials.credentials
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError as e:
        logger.error(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials"
        )

@router.post("/token")
async def get_token(request: Request):
    """Get a new access token."""
    try:
        # Get user data from request
        user_data = await request.json()
        
        # Create token
        token = create_access_token(
            data={
                "sub": user_data.get("username", "anonymous"),
                "lob": user_data.get("lob", "general"),
                "state": user_data.get("state")
            }
        )
        
        return {"token": token}
    except Exception as e:
        logger.error(f"Error creating token: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 