from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from .common import BaseResponse, PaginationParams

class DocumentMetadata(BaseModel):
    """Document metadata model."""
    title: Optional[str] = Field(None, description="Document title")
    author: Optional[str] = Field(None, description="Document author")
    source: Optional[str] = Field(None, description="Document source")
    url: Optional[HttpUrl] = Field(None, description="Document URL")
    created_at: Optional[datetime] = Field(None, description="Document creation date")
    updated_at: Optional[datetime] = Field(None, description="Document last update date")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    custom_metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom metadata fields")

class DocumentBase(BaseModel):
    """Base document model."""
    content: str = Field(..., description="Document content")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="Document metadata")

class DocumentCreate(DocumentBase):
    """Document creation model."""
    pass

class DocumentUpdate(BaseModel):
    """Document update model."""
    content: Optional[str] = Field(None, description="Document content")
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")

class DocumentInDB(DocumentBase):
    """Document model as stored in database."""
    id: str = Field(..., description="Document ID")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    embedding_id: Optional[str] = Field(None, description="ID of the document's embedding")
    chunk_count: int = Field(default=0, description="Number of chunks in the document")

class DocumentResponse(BaseResponse):
    """Document response model."""
    document: DocumentInDB = Field(..., description="Document information")

class DocumentListResponse(BaseResponse):
    """Document list response model."""
    documents: List[DocumentInDB] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Number of documents per page")
    total_pages: int = Field(..., description="Total number of pages")

class DocumentProcessRequest(BaseModel):
    """Document processing request model."""
    content: str = Field(..., description="Document content to process")
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")
    chunk_size: Optional[int] = Field(None, ge=1, description="Size of document chunks")
    chunk_overlap: Optional[int] = Field(None, ge=0, description="Overlap between chunks")

class DocumentProcessResponse(BaseResponse):
    """Document processing response model."""
    document_id: str = Field(..., description="ID of the processed document")
    chunk_count: int = Field(..., description="Number of chunks created")
    processing_time: float = Field(..., description="Time taken for processing in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata") 