"""
Main application module for the RAG system.
Consolidates functionality from various app directories.
"""

import os
import logging
import signal
import sys
from typing import Dict, Any
from datetime import datetime, timedelta
from dotenv import load_dotenv
import asyncio
from jose import jwt

# Load environment variables from .env file
load_dotenv()

import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from langchain.schema import Document

from core.settings import settings, HOST, FASTAPI_PORT
from core.rag_engine import RAGEngine
from core.metrics import MetricsCollector
from core.fine_tuning import FineTuningManager
from app.routes import auth, query, documents, fine_tuning
from core.document_processor import DocumentProcessor
from core.model_client import ModelClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG API",
    description="""
    A powerful RAG (Retrieval-Augmented Generation) API that provides:
    - Document management and processing
    - Semantic search and querying
    - Authentication and authorization
    - Model management
    
    This API follows OpenAPI 3.0 specification and provides interactive documentation.
    """,
    version="1.0.0",
    docs_url=None,  # Disable default docs
    redoc_url=None,  # Disable default redoc
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get("security.allowed_origins", ["http://localhost:8501"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router)
app.include_router(query.router)
app.include_router(documents.router)
app.include_router(fine_tuning.router)

# Global state for graceful shutdown
app.state.shutting_down = False
app.state.shutdown_event = asyncio.Event()

def handle_signal(signum, frame):
    """Handle shutdown signals gracefully."""
    if not app.state.shutting_down:
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        app.state.shutting_down = True
        # Set the shutdown event
        if hasattr(app.state, "shutdown_event"):
            app.state.shutdown_event.set()
        # Exit with status code 0 to indicate clean shutdown
        os._exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

async def cleanup_component(component, component_name: str):
    """Helper function to clean up a component with proper error handling."""
    if hasattr(component, "cleanup"):
        try:
            cleanup_method = getattr(component, "cleanup")
            if asyncio.iscoroutinefunction(cleanup_method):
                try:
                    await asyncio.wait_for(cleanup_method(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning(f"Timeout cleaning up {component_name}")
                except asyncio.CancelledError:
                    logger.warning(f"Cleanup cancelled for {component_name}")
                except Exception as e:
                    logger.warning(f"Error during async cleanup of {component_name}: {str(e)}")
            else:
                try:
                    cleanup_method()
                except Exception as e:
                    logger.warning(f"Error during sync cleanup of {component_name}: {str(e)}")
            logger.info(f"{component_name} resources cleaned up")
        except Exception as e:
            logger.warning(f"Error accessing cleanup method for {component_name}: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    try:
        # Initialize components
        app.state.rag_engine = RAGEngine()
        app.state.metrics_collector = MetricsCollector()
        app.state.fine_tuning_manager = FineTuningManager()
        app.state.document_processor = DocumentProcessor()
        app.state.model_client = ModelClient()
        
        # Add test document about RAG
        rag_doc = Document(
            page_content="""
            Retrieval-Augmented Generation (RAG) is a technique that enhances large language models (LLMs) 
            by retrieving relevant information from external knowledge sources before generating responses. 
            This approach helps to:
            1. Improve accuracy by providing up-to-date information
            2. Reduce hallucinations by grounding responses in retrieved facts
            3. Enable the model to access domain-specific knowledge
            4. Make the system more transparent by showing sources
            
            RAG works by first retrieving relevant documents or passages from a knowledge base, then using 
            this context along with the user's query to generate a more informed and accurate response.
            """,
            metadata={"source": "system", "type": "rag_description"}
        )
        app.state.rag_engine.add_documents([rag_doc])
        
        # Log startup information
        logger.info("Application components initialized successfully")
        logger.info(f"Server running on {HOST}:{FASTAPI_PORT}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
        else:
            logger.warning("No GPU available. Some operations may be slow on CPU.")
            
    except Exception as e:
        logger.error(f"Error initializing application components: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    if app.state.shutting_down:
        logger.info("Shutdown already in progress, skipping cleanup")
        return

    try:
        app.state.shutting_down = True
        # Cleanup resources in reverse order of initialization
        components = [
            ("model_client", "Model client"),
            ("document_processor", "Document processor"),
            ("fine_tuning_manager", "Fine tuning manager"),
            ("metrics_collector", "Metrics collector"),
            ("rag_engine", "RAG engine")
        ]
        
        for component_name, component_display in components:
            if hasattr(app.state, component_name):
                component = getattr(app.state, component_name)
                await cleanup_component(component, component_display)
        
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        # Don't raise the error to allow graceful shutdown

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Dict containing health status
    """
    try:
        status = {
            "status": "healthy",
            "components": {
                "rag_engine": "operational",
                "metrics_collector": "operational",
                "fine_tuning_manager": "operational",
                "document_processor": "operational",
                "model_client": "operational"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        return status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

# Debug token endpoint for testing
@app.get("/debug/token", tags=["Debug"])
async def get_debug_token(lob: str = "auto", state: str = "CA") -> dict:
    """
    Get a debug token for testing purposes.
    
    Args:
        lob: Line of business filter
        state: State code filter
        
    Returns:
        Dict containing debug token
    """
    jwt_secret = settings.get('security', {}).get('jwt_secret') or os.environ.get('JWT_SECRET')
    jwt_algorithm = settings.get('security', {}).get('jwt_algorithm') or os.environ.get('JWT_ALGORITHM', 'HS256')
    payload = {
        "sub": "debug_user",
        "lob": lob,
        "state": state,
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, jwt_secret, algorithm=jwt_algorithm)
    return {"token": token}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Welcome to the RAG API. Visit /docs for documentation."}

def custom_openapi() -> Dict[str, Any]:
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
        
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add security schemes
    openapi_schema["components"] = {
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT"
            }
        }
    }
    
    # Add security requirement to all operations
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            operation["security"] = [{"bearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI endpoint."""
    return get_swagger_ui_html(
        openapi_url="/openapi.json",
        title=f"{app.title} - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.9.0/swagger-ui.css",
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=FASTAPI_PORT,
        reload=True,
        reload_dirs=["app", "core"],  # Only watch specific directories
        reload_delay=2.0,  # Increase delay to prevent rapid reloads
        workers=1,  # Use single worker for stability
        log_level="info",
        loop="asyncio",  # Use asyncio event loop
        limit_concurrency=100,  # Limit concurrent connections
        backlog=2048,  # Increase connection backlog
        timeout_keep_alive=30,  # Keep-alive timeout
        timeout_graceful_shutdown=30  # Graceful shutdown timeout
    ) 