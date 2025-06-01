"""
Document Processor for handling document uploads, processing, and management.
"""

import logging
from typing import List, Optional, Dict, Any
from fastapi import UploadFile
from datetime import datetime
import uuid

from core.document_manager import DocumentManager
from core.settings import settings

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles document processing and management operations."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.document_manager = DocumentManager()
        self.documents: Dict[str, Dict[str, Any]] = {}  # In-memory document storage
    
    async def process_upload(self, file: UploadFile, user_id: str) -> Dict[str, Any]:
        """
        Process an uploaded document.
        
        Args:
            file: The uploaded file
            user_id: ID of the user uploading the document
            
        Returns:
            Dictionary containing document information
        """
        try:
            content = await file.read()
            text = content.decode('utf-8')
            
            document_id = str(uuid.uuid4())
            document = {
                "id": document_id,
                "filename": file.filename,
                "content": text,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "metadata": {
                    "content_type": file.content_type,
                    "size": len(content)
                }
            }
            
            # Store document
            self.documents[document_id] = document
            
            # Process document with document manager
            self.document_manager.add_documents([{
                "text": text,
                "metadata": {
                    "document_id": document_id,
                    "filename": file.filename,
                    "user_id": user_id
                }
            }])
            
            return document
            
        except Exception as e:
            logger.error(f"Error processing upload: {e}")
            raise
    
    async def process_content(
        self,
        content: str,
        metadata: Dict[str, Any],
        chunk_size: Optional[int] = None,
        user_id: str = None
    ) -> Dict[str, Any]:
        """
        Process document content directly.
        
        Args:
            content: Document content
            metadata: Document metadata
            chunk_size: Size of text chunks
            user_id: ID of the user
            
        Returns:
            Dictionary containing processing results
        """
        try:
            document_id = str(uuid.uuid4())
            
            # Process with document manager
            self.document_manager.add_documents([{
                "text": content,
                "metadata": {
                    "document_id": document_id,
                    "user_id": user_id,
                    **metadata
                }
            }])
            
            return {
                "document_id": document_id,
                "chunks": len(content.split()),  # Simple chunk count
                "processing_time": 0.0,  # TODO: Add actual processing time
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Error processing content: {e}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Document information if found, None otherwise
        """
        return self.documents.get(document_id)
    
    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List documents with pagination.
        
        Args:
            skip: Number of documents to skip
            limit: Maximum number of documents to return
            
        Returns:
            List of documents
        """
        documents = list(self.documents.values())
        return documents[skip:skip + limit]
    
    async def count_documents(self) -> int:
        """
        Get total number of documents.
        
        Returns:
            Total document count
        """
        return len(self.documents)
    
    async def update_document(
        self,
        document_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Update document information.
        
        Args:
            document_id: ID of the document
            updates: Dictionary of updates
            
        Returns:
            Updated document if found, None otherwise
        """
        if document_id not in self.documents:
            return None
            
        document = self.documents[document_id]
        document.update(updates)
        document["updated_at"] = datetime.utcnow()
        
        return document
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            True if successful, False otherwise
        """
        if document_id not in self.documents:
            return False
            
        del self.documents[document_id]
        return True

    async def cleanup(self) -> None:
        """
        Clean up resources and perform necessary cleanup operations.
        """
        try:
            # Clear in-memory document storage
            self.documents.clear()
            
            # Clean up document manager resources
            if hasattr(self.document_manager, 'cleanup'):
                await self.document_manager.cleanup()
                
            logger.info("Document processor cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during document processor cleanup: {e}")
            raise 