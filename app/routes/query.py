from fastapi import APIRouter, HTTPException, Request, Security, Depends
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional, Union
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
import json
from datetime import datetime

from app.schemas import QueryRequest, QueryResponse
from core.settings import settings
from core.rag_engine import RAGEngine
from core.auth import validate_token

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
        
        # Get RAG engine instance
        rag_engine = request.app.state.rag_engine
        
        # Process query
        if request_body.stream:
            def stream_generator():
                for chunk, sources in rag_engine.process_query(
                    query=request_body.question,
                    stream=True,
                    reference_answer=request_body.reference_answer
                ):
                    yield json.dumps({
                        "text": chunk,
                        "sources": sources,
                        "metadata": {
                            "timestamp": datetime.utcnow().isoformat(),
                            "user": user.get("sub", "unknown")
                        },
                        "metrics": {
                            "retrieval_latency": rag_engine.metrics.retrieval_latency,
                            "generation_latency": rag_engine.metrics.generation_latency,
                            "cache_hits": rag_engine.metrics.cache_hits,
                            "cache_misses": rag_engine.metrics.cache_misses,
                            "token_usage": rag_engine.metrics.token_usage,
                            "rouge_scores": rag_engine.metrics.rouge_scores,
                            "bert_scores": rag_engine.metrics.bert_scores
                        }
                    }) + "\n"
            
            return StreamingResponse(
                stream_generator(),
                media_type="application/x-ndjson"
            )
        else:
            # Process query without streaming
            response, sources = rag_engine.process_query(
                query=request_body.question,
                stream=False,
                reference_answer=request_body.reference_answer
            )
            
            return QueryResponse(
                answer=response,
                sources=sources,
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "user": user.get("sub", "unknown")
                },
                metrics={
                    "retrieval_latency": rag_engine.metrics.retrieval_latency,
                    "generation_latency": rag_engine.metrics.generation_latency,
                    "cache_hits": rag_engine.metrics.cache_hits,
                    "cache_misses": rag_engine.metrics.cache_misses,
                    "token_usage": rag_engine.metrics.token_usage,
                    "rouge_scores": rag_engine.metrics.rouge_scores,
                    "bert_scores": rag_engine.metrics.bert_scores
                }
            )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 