from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from .common import ErrorResponse

class ValidationError(BaseModel):
    """Validation error details."""
    loc: List[str] = Field(..., description="Location of the error in the request")
    msg: str = Field(..., description="Error message")
    type: str = Field(..., description="Type of validation error")

class ValidationErrorResponse(ErrorResponse):
    """Response model for validation errors."""
    error_code: Literal["VALIDATION_ERROR"] = "VALIDATION_ERROR"
    details: List[ValidationError] = Field(..., description="List of validation errors")

class AuthenticationErrorResponse(ErrorResponse):
    """Response model for authentication errors."""
    error_code: Literal["AUTHENTICATION_ERROR"] = "AUTHENTICATION_ERROR"
    error_message: Literal["Authentication failed"] = "Authentication failed"

class AuthorizationErrorResponse(ErrorResponse):
    """Response model for authorization errors."""
    error_code: Literal["AUTHORIZATION_ERROR"] = "AUTHORIZATION_ERROR"
    error_message: Literal["Not authorized to perform this action"] = "Not authorized to perform this action"

class NotFoundErrorResponse(ErrorResponse):
    """Response model for not found errors."""
    error_code: Literal["NOT_FOUND"] = "NOT_FOUND"
    error_message: Literal["Resource not found"] = "Resource not found"

class ServerErrorResponse(ErrorResponse):
    """Response model for server errors."""
    error_code: Literal["SERVER_ERROR"] = "SERVER_ERROR"
    error_message: Literal["Internal server error"] = "Internal server error"

# Dictionary mapping error codes to their response models
ERROR_RESPONSES = {
    "VALIDATION_ERROR": ValidationErrorResponse,
    "AUTHENTICATION_ERROR": AuthenticationErrorResponse,
    "AUTHORIZATION_ERROR": AuthorizationErrorResponse,
    "NOT_FOUND": NotFoundErrorResponse,
    "SERVER_ERROR": ServerErrorResponse
} 