from fastapi import APIRouter, Request
from core.settings import settings

router = APIRouter(tags=["Health"])

@router.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    vectorstore_status = "available" if hasattr(request.app.state, "vectorstore") and request.app.state.vectorstore is not None else "not available"
    vector_db_type = settings.vector_db
    model_status = "available"  # Assume model is always available
    
    return {
        "status": "healthy",
        "vectorstore": vectorstore_status,
        "vector_db_type": vector_db_type,
        "model": model_status,
        "model_name": settings.model_name,
    } 