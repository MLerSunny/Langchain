from fastapi import APIRouter, HTTPException, Request, Security, Depends, status, UploadFile, File
from typing import List, Optional
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from app.schemas.documents import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentListResponse,
    DocumentProcessRequest,
    DocumentProcessResponse
)
from app.schemas.errors import (
    AuthenticationErrorResponse,
    AuthorizationErrorResponse,
    ValidationErrorResponse,
    ServerErrorResponse,
    NotFoundErrorResponse
)
from core.settings import settings
from core.document_processor import DocumentProcessor

router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
    responses={
        401: {"model": AuthenticationErrorResponse},
        403: {"model": AuthorizationErrorResponse},
        404: {"model": NotFoundErrorResponse},
        422: {"model": ValidationErrorResponse},
        500: {"model": ServerErrorResponse}
    }
)

logger = logging.getLogger(__name__)
security = HTTPBearer()

def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> dict:
    """Validate JWT token and return user info."""
    try:
        payload = jwt.decode(
            credentials.credentials, 
            settings.jwt_secret, 
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    request: Request = None,
    user: dict = Depends(validate_token)
) -> DocumentResponse:
    """
    Upload and process a document.
    
    Args:
        file: The document file to upload
        request: FastAPI request object
        user: Authenticated user information
        
    Returns:
        DocumentResponse containing the processed document information
        
    Raises:
        HTTPException: If document processing fails
    """
    try:
        processor = request.app.state.document_processor
        document = await processor.process_upload(file, user.get("sub"))
        return DocumentResponse.from_orm(document)
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/process", response_model=DocumentProcessResponse, status_code=status.HTTP_200_OK)
async def process_document(
    request_body: DocumentProcessRequest,
    request: Request,
    user: dict = Depends(validate_token)
) -> DocumentProcessResponse:
    """
    Process document content directly.
    
    Args:
        request_body: Document processing request containing content and metadata
        request: FastAPI request object
        user: Authenticated user information
        
    Returns:
        DocumentProcessResponse containing processing results
        
    Raises:
        HTTPException: If document processing fails
    """
    try:
        processor = request.app.state.document_processor
        result = await processor.process_content(
            content=request_body.content,
            metadata=request_body.metadata,
            chunk_size=request_body.chunk_size,
            user_id=user.get("sub")
        )
        return DocumentProcessResponse(
            document_id=result.document_id,
            chunks=result.chunks,
            processing_time=result.processing_time,
            metadata=result.metadata
        )
    except Exception as e:
        logger.error(f"Error processing document content: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{document_id}", response_model=DocumentResponse, status_code=status.HTTP_200_OK)
async def get_document(
    document_id: str,
    request: Request,
    user: dict = Depends(validate_token)
) -> DocumentResponse:
    """
    Get document by ID.
    
    Args:
        document_id: ID of the document to retrieve
        request: FastAPI request object
        user: Authenticated user information
        
    Returns:
        DocumentResponse containing the document information
        
    Raises:
        HTTPException: If document not found or retrieval fails
    """
    try:
        processor = request.app.state.document_processor
        document = await processor.get_document(document_id)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        return DocumentResponse.from_orm(document)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/", response_model=DocumentListResponse, status_code=status.HTTP_200_OK)
async def list_documents(
    skip: int = 0,
    limit: int = 10,
    request: Request = None,
    user: dict = Depends(validate_token)
) -> DocumentListResponse:
    """
    List documents with pagination.
    
    Args:
        skip: Number of documents to skip
        limit: Maximum number of documents to return
        request: FastAPI request object
        user: Authenticated user information
        
    Returns:
        DocumentListResponse containing list of documents and pagination info
        
    Raises:
        HTTPException: If document listing fails
    """
    try:
        processor = request.app.state.document_processor
        documents = await processor.list_documents(skip=skip, limit=limit)
        total = await processor.count_documents()
        return DocumentListResponse(
            items=[DocumentResponse.from_orm(doc) for doc in documents],
            total=total,
            skip=skip,
            limit=limit
        )
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.put("/{document_id}", response_model=DocumentResponse, status_code=status.HTTP_200_OK)
async def update_document(
    document_id: str,
    document_update: DocumentUpdate,
    request: Request,
    user: dict = Depends(validate_token)
) -> DocumentResponse:
    """
    Update document metadata.
    
    Args:
        document_id: ID of the document to update
        document_update: Updated document information
        request: FastAPI request object
        user: Authenticated user information
        
    Returns:
        DocumentResponse containing the updated document information
        
    Raises:
        HTTPException: If document not found or update fails
    """
    try:
        processor = request.app.state.document_processor
        document = await processor.update_document(document_id, document_update)
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        return DocumentResponse.from_orm(document)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    request: Request,
    user: dict = Depends(validate_token)
) -> None:
    """
    Delete a document.
    
    Args:
        document_id: ID of the document to delete
        request: FastAPI request object
        user: Authenticated user information
        
    Raises:
        HTTPException: If document not found or deletion fails
    """
    try:
        processor = request.app.state.document_processor
        success = await processor.delete_document(document_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 