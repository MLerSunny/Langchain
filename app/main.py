"""
FastAPI application for RAG-based question answering system.

This module sets up the FastAPI application with endpoints for RAG-based QA.
"""

import logging
import os
import sys
import redis
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query, Depends, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi_limiter import FastAPILimiter, limiter
from jose import jwt, JWTError
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Qdrant
from pydantic import BaseModel
import uvicorn
import traceback
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.proxy import OllamaProxyClient
from app.schemas import QueryRequest, QueryResponse
from core.settings import settings
from app.routes import query, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="DeepSeek LLM API",
    description="API for DeepSeek RAG and fine-tuning service",
    version="1.0.0",
)

# Security configuration
security = HTTPBearer()

# Add CORS middleware with tightened origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit local
        "http://localhost:8000",  # FastAPI local
        "http://localhost:3000",  # Frontend local
        "https://api.yourorganization.com"  # Production domain
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    return response

# Initialize rate limiter
@app.on_event("startup")
async def setup_limiter():
    """Initialize rate limiter with Redis."""
    try:
        await FastAPILimiter.init(redis.from_url("redis://redis:6379"))
        logger.info("Rate limiter initialized with Redis")
    except Exception as e:
        logger.error(f"Failed to initialize rate limiter: {e}")

# Add rate limiting middleware
app.middleware("http")(limiter.limit("30/minute"))

# Initialize the vector store
@app.on_event("startup")
async def startup_db_client():
    """Initialize the vector store and model client on startup."""
    try:
        # Initialize model client
        app.state.model_client = OllamaProxyClient()
        logger.info("Model client initialized")
        
        # Initialize embeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        if settings.vector_db == "qdrant":
            # Connect to Qdrant
            from langchain_community.vectorstores import Qdrant
            from qdrant_client import QdrantClient
            
            qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
            client = QdrantClient(url=qdrant_url)
            
            try:
                # Check if collection exists
                collections = client.get_collections().collections
                collection_exists = any(c.name == "insurance_docs" for c in collections)
                
                if collection_exists:
                    app.state.vectorstore = Qdrant(
                        client=client,
                        collection_name="insurance_docs",
                        embedding_function=embeddings,
                    )
                    logger.info(f"Connected to Qdrant vector store at {qdrant_url}")
                else:
                    logger.warning("Qdrant collection does not exist, please run ingest script")
                    app.state.vectorstore = None
            except Exception as e:
                logger.error(f"Error connecting to Qdrant: {e}")
                app.state.vectorstore = None
        else:
            # Default to Chroma
            chroma_path = settings.chroma_path
            
            # Check if the vector store exists
            if not os.path.exists(chroma_path):
                logger.warning(f"Vector store directory not found at {chroma_path}")
                logger.warning("Please run the ingest script to create the vector store")
                app.state.vectorstore = None
            else:
                # Initialize Chroma vector store
                app.state.vectorstore = Chroma(
                    persist_directory=chroma_path,
                    embedding_function=embeddings,
                )
                logger.info(f"Vector store loaded from {chroma_path}")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Continue without the vector store
        app.state.vectorstore = None

# Include routers from modules
app.include_router(health.router)
app.include_router(query.router)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "DeepSeek LLM API is running",
        "docs_url": "/docs",
        "version": "1.0.0",
    }

# JWT token functions
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=30)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return encoded_jwt

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

# Exception handling
@app.exception_handler(Exception)
async def handle_general_exception(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(traceback.format_exc())
    return {
        "status": "error",
        "message": "An unexpected error occurred",
        "details": str(exc) if app.debug else "Contact administrator for details",
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.fastapi_port,
        reload=True,
    ) 