"""Schema definitions for the API."""

from .common import (
    BaseResponse,
    ErrorResponse,
    PaginationParams,
    PaginatedResponse
)

from .errors import (
    ValidationError,
    ValidationErrorResponse,
    AuthenticationErrorResponse,
    AuthorizationErrorResponse,
    NotFoundErrorResponse,
    ServerErrorResponse,
    ERROR_RESPONSES
)

from .auth import (
    Token,
    TokenData,
    UserBase,
    UserCreate,
    UserUpdate,
    UserInDB,
    UserResponse,
    TokenResponse
)

from .query import (
    QueryRequest,
    Source,
    Metrics,
    QueryResponse,
    StreamChunk
)

from .documents import (
    DocumentMetadata,
    DocumentBase,
    DocumentCreate,
    DocumentUpdate,
    DocumentInDB,
    DocumentResponse,
    DocumentListResponse,
    DocumentProcessRequest,
    DocumentProcessResponse
)

__all__ = [
    # Common schemas
    "BaseResponse",
    "ErrorResponse",
    "PaginationParams",
    "PaginatedResponse",
    
    # Error schemas
    "ValidationError",
    "ValidationErrorResponse",
    "AuthenticationErrorResponse",
    "AuthorizationErrorResponse",
    "NotFoundErrorResponse",
    "ServerErrorResponse",
    "ERROR_RESPONSES",
    
    # Auth schemas
    "Token",
    "TokenData",
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserInDB",
    "UserResponse",
    "TokenResponse",
    
    # Query schemas
    "QueryRequest",
    "Source",
    "Metrics",
    "QueryResponse",
    "StreamChunk",
    
    # Document schemas
    "DocumentMetadata",
    "DocumentBase",
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentInDB",
    "DocumentResponse",
    "DocumentListResponse",
    "DocumentProcessRequest",
    "DocumentProcessResponse"
] 