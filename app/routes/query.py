from fastapi import APIRouter, HTTPException, Request, Security, Depends, status
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import json
from datetime import datetime
import os

from app.schemas.query import (
    QueryRequest,
    QueryResponse,
    StreamChunk,
    Source,
    Metrics
)
from app.schemas.errors import (
    AuthenticationErrorResponse,
    AuthorizationErrorResponse,
    ValidationErrorResponse,
    ServerErrorResponse
)
from core.settings import settings
from core.rag_engine import RAGEngine

router = APIRouter(
    prefix="/query",
    tags=["Query"],
    responses={
        401: {"model": AuthenticationErrorResponse},
        403: {"model": AuthorizationErrorResponse},
        422: {"model": ValidationErrorResponse},
        500: {"model": ServerErrorResponse}
    }
)

logger = logging.getLogger(__name__)
security = HTTPBearer()

def validate_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Validate JWT token and return user info."""
    # Try config, then environment
    jwt_secret = settings.get('security', {}).get('jwt_secret') or os.environ.get('JWT_SECRET')
    jwt_algorithm = settings.get('security', {}).get('jwt_algorithm') or os.environ.get('JWT_ALGORITHM', 'HS256')
    if not jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT secret not configured in config or environment."
        )
    try:
        payload = jwt.decode(
            credentials.credentials, 
            jwt_secret, 
            algorithms=[jwt_algorithm]
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.get("/models", response_model=Dict[str, List[str]], status_code=status.HTTP_200_OK)
async def list_models(
    request: Request,
    user: dict = Depends(validate_token)
) -> Dict[str, List[str]]:
    """
    List available models.
    
    Args:
        request: FastAPI request object
        user: Authenticated user information
        
    Returns:
        Dict containing list of available models
        
    Raises:
        HTTPException: If listing models fails
    """
    try:
        models = await request.app.state.model_client.list_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/metrics", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_metrics(
    request: Request,
    user: dict = Depends(validate_token)
) -> Dict[str, Any]:
    """
    Get system metrics.
    
    Args:
        request: FastAPI request object
        user: Authenticated user information
        
    Returns:
        Dict containing system metrics
        
    Raises:
        HTTPException: If metrics retrieval fails
    """
    try:
        metrics = request.app.state.metrics_collector.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/query", response_model=QueryResponse)
async def process_query(
    request_body: QueryRequest,
    request: Request
):
    """Process a query and return response."""
    try:
        # Process query
        response, sources = request.app.state.rag_engine.process_query(
            query=request_body.question,
            stream=request_body.stream
        )
        
        # Convert Document objects to Source objects
        source_objects = []
        for doc in sources:
            source_objects.append(Source(
                document_id=doc.metadata.get('source', 'unknown'),
                content=doc.page_content,
                metadata=doc.metadata,
                score=doc.metadata.get('score', 1.0)
            ))
        
        # Get metrics
        metrics_dict = request.app.state.rag_engine.get_metrics()
        metrics = Metrics(**metrics_dict)
        
        # Clean up response by removing prompt template and context
        if response.startswith("System:") or response.startswith("Context:"):
            response = response.split("Assistant:")[-1].strip()
        if response.startswith("Based on the provided context"):
            response = response.replace("Based on the provided context, ", "")
        
        return QueryResponse(
            status="success",
            answer=response,
            sources=source_objects,
            metrics=metrics,
            metadata={
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            status="error",
            answer="An error occurred while processing your query.",
            sources=[],
            metrics=Metrics(),
            metadata={
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        ) 