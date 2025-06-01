from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class BaseResponse(BaseModel):
    """Base response model for all API responses."""
    status: str = Field(..., description="Status of the response")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of the response")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class ErrorResponse(BaseResponse):
    """Standard error response model."""
    error_code: str = Field(..., description="Error code for the error")
    error_message: str = Field(..., description="Human readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")

class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=10, ge=1, le=100, description="Number of items per page")

class PaginatedResponse(BaseResponse):
    """Paginated response model."""
    total: int = Field(..., description="Total number of items")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of items per page")
    total_pages: int = Field(..., description="Total number of pages")
    items: list = Field(..., description="List of items in the current page") 