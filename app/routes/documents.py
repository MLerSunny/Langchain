from fastapi import APIRouter, HTTPException, Depends, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import List, Optional, Dict, Any
import json
import logging
from datetime import datetime

from core.rag_engine import RAGEngine
from core.auth import validate_token

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/process")
async def process_document(
    request: Request,
    file: UploadFile = File(...),
    metadata: Optional[Dict[str, Any]] = None,
    user: dict = Depends(validate_token)
):
    """Process a single document."""
    try:
        # Get RAG engine instance
        rag_engine = request.app.state.rag_engine
        
        # Read file content
        content = await file.read()
        text = content.decode("utf-8")
        
        # Process document
        documents = rag_engine.process_document(
            document=text,
            metadata=metadata or {}
        )
        
        return {
            "status": "success",
            "document_count": len(documents),
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "user": user.get("sub", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/batch")
async def process_documents_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    metadata: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    parallel: bool = True,
    user: dict = Depends(validate_token)
):
    """Process multiple documents in batches."""
    try:
        # Get RAG engine instance
        rag_engine = request.app.state.rag_engine
        
        # Read file contents
        documents = []
        for file in files:
            content = await file.read()
            text = content.decode("utf-8")
            documents.append(text)
        
        # Process documents
        processed_docs = rag_engine.process_documents_batch(
            documents=documents,
            metadata=metadata or {},
            batch_size=batch_size,
            parallel=parallel
        )
        
        return {
            "status": "success",
            "document_count": len(processed_docs),
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "user": user.get("sub", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Error processing documents batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/stream")
async def process_documents_stream(
    request: Request,
    files: List[UploadFile] = File(...),
    metadata: Optional[Dict[str, Any]] = None,
    batch_size: Optional[int] = None,
    user: dict = Depends(validate_token)
):
    """Process multiple documents and stream results."""
    try:
        # Get RAG engine instance
        rag_engine = request.app.state.rag_engine
        
        # Read file contents
        documents = []
        for file in files:
            content = await file.read()
            text = content.decode("utf-8")
            documents.append(text)
        
        def stream_generator():
            for batch in rag_engine.process_documents_stream(
                documents=documents,
                metadata=metadata or {},
                batch_size=batch_size
            ):
                yield json.dumps({
                    "documents": [
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata
                        }
                        for doc in batch
                    ],
                    "metadata": {
                        "timestamp": datetime.utcnow().isoformat(),
                        "user": user.get("sub", "unknown")
                    }
                }) + "\n"
        
        return StreamingResponse(
            stream_generator(),
            media_type="application/x-ndjson"
        )
        
    except Exception as e:
        logger.error(f"Error streaming document processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_document_status(
    request: Request,
    user: dict = Depends(validate_token)
):
    """Get document processing status."""
    try:
        # Get RAG engine instance
        rag_engine = request.app.state.rag_engine
        
        # Get vector store stats
        stats = rag_engine.vector_store.get_stats()
        
        return {
            "status": "success",
            "stats": stats,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "user": user.get("sub", "unknown")
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting document status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 