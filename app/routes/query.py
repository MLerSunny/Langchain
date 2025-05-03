from fastapi import APIRouter, HTTPException, Request, Security, Depends
from typing import List, Dict, Any, Optional
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from app.schemas import QueryRequest, QueryResponse
from core.settings import settings

router = APIRouter(tags=["Query"])
logger = logging.getLogger(__name__)
security = HTTPBearer()

def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)):
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
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/models")
async def list_models(request: Request, user: dict = Depends(validate_token)):
    """List available models."""
    try:
        models = await request.app.state.model_client.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query(request_body: QueryRequest, request: Request, user: dict = Depends(validate_token)):
    """Query endpoint for RAG-based QA."""
    try:
        # Store user in request state for middleware/handlers
        request.state.user = user
        
        context_docs = []
        sources = []
        
        # Retrieve context from vector store if available
        if request.app.state.vectorstore is not None:
            try:
                retrieval_results = request.app.state.vectorstore.similarity_search_with_score(
                    request_body.question, k=request_body.top_k
                )
                
                if retrieval_results:
                    # Extract documents and metadata
                    context_docs = [doc for doc, _ in retrieval_results]
                    sources = [
                        doc.metadata.get("source", "Unknown")
                        for doc, _ in retrieval_results
                    ]
                    
            except Exception as e:
                logger.error(f"Error retrieving from vector store: {e}")
                # Continue without context
        
        # Generate answer using the model
        answer = await request.app.state.model_client.generate(
            query=request_body.question,
            context_docs=context_docs,
            model=settings.model_name,
            temperature=request_body.temperature,
            max_tokens=settings.max_tokens,
        )
        
        return QueryResponse(
            answer=answer["text"],
            sources=sources if sources else []
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 