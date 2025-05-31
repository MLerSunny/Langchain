"""
Main application module for the RAG system.
Consolidates functionality from various app directories.
"""

import os
import logging
from typing import List, Optional, Dict, Any
import time
from datetime import datetime

import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from core.settings import settings, HOST, FASTAPI_PORT
from core.rag import RAGEngine
from core.metrics import MetricsCollector
from core.fine_tuning import FineTuningManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG + Fine-tuning API",
    description="API for RAG system with model management and document processing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_engine = None
metrics_collector = None
fine_tuning_manager = None

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    global rag_engine, metrics_collector, fine_tuning_manager
    try:
        # Initialize components
        rag_engine = RAGEngine()
        metrics_collector = MetricsCollector()
        fine_tuning_manager = FineTuningManager()
        
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
    try:
        # Add any cleanup code here
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")

class QueryRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    query: str
    context: Optional[str] = None
    max_tokens: Optional[int] = settings.get("generation.max_tokens", 4096)
    temperature: Optional[float] = settings.get("generation.temperature", 0.7)

class FineTuningRequest(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    training_data: List[Dict[str, str]]
    epochs: Optional[int] = settings.get("training.epochs", 3)
    batch_size: Optional[int] = settings.get("training.batch_size", 32)
    learning_rate: Optional[float] = settings.get("training.learning_rate", 2e-5)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        status = {
            "status": "healthy",
            "components": {
                "rag_engine": rag_engine is not None,
                "metrics_collector": metrics_collector is not None,
                "fine_tuning_manager": fine_tuning_manager is not None
            },
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
        
        # Add GPU information if available
        if torch.cuda.is_available():
            status["gpu"] = {
                "available": True,
                "device": torch.cuda.get_device_name(0),
                "cuda_version": torch.version.cuda
            }
        else:
            status["gpu"] = {
                "available": False
            }
            
        return status
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service unhealthy: {str(e)}"
        )

@app.get("/metrics")
async def get_metrics():
    """Get system metrics."""
    try:
        metrics = metrics_collector.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Process a query through the RAG system."""
    try:
        if rag_engine is None:
            raise HTTPException(status_code=503, detail="RAG engine not initialized")
            
        start_time = time.time()
        
        # Process query
        response, sources = rag_engine.process_query(
            request.query,
            context=request.context,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        # Record metrics in background
        if metrics_collector is not None:
            background_tasks.add_task(
                metrics_collector.record_query,
                request.query,
                time.time() - start_time
            )
        
        return {
            "answer": response,
            "sources": sources,
            "processing_time": time.time() - start_time
        }
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine-tune")
async def fine_tune(request: FineTuningRequest, background_tasks: BackgroundTasks):
    """Start a fine-tuning job."""
    try:
        job_id = fine_tuning_manager.start_training(
            model_name=request.model_name,
            training_data=request.training_data,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate
        )
        
        return {"job_id": job_id, "status": "started"}
        
    except Exception as e:
        logger.error(f"Error starting fine-tuning: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fine-tuning-status/{job_id}")
async def get_fine_tuning_status(job_id: str):
    """Get the status of a fine-tuning job."""
    try:
        status = fine_tuning_manager.get_job_status(job_id)
        return status
    except Exception as e:
        logger.error(f"Error getting fine-tuning status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Debug endpoints
@app.get("/debug/token")
async def get_debug_token(lob: str = "general", state: Optional[str] = None):
    """Generate a debug token for testing."""
    try:
        token = auth.create_access_token(
            data={"sub": "debug_user", "lob": lob, "state": state}
        )
        return {"token": token}
    except Exception as e:
        logger.error(f"Error generating debug token: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the RAG + Fine-tuning API"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=HOST,
        port=FASTAPI_PORT,
        reload=True
    ) 