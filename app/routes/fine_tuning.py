from fastapi import APIRouter, HTTPException, Request, Security, Depends, status, BackgroundTasks
from typing import Dict, Any
import logging
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from app.schemas.fine_tuning import FineTuningRequest, FineTuningResponse
from app.schemas.errors import (
    AuthenticationErrorResponse,
    AuthorizationErrorResponse,
    ValidationErrorResponse,
    ServerErrorResponse
)
from core.settings import settings
from core.fine_tuning import FineTuningManager

router = APIRouter(
    prefix="/fine-tuning",
    tags=["Fine-tuning"],
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
    try:
        payload = jwt.decode(
            credentials.credentials, 
            settings.jwt_secret, 
            algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

@router.post("/", response_model=FineTuningResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_fine_tuning(
    request_body: FineTuningRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    user: dict = Depends(validate_token)
) -> FineTuningResponse:
    """
    Start a fine-tuning job.
    
    Args:
        request_body: Fine-tuning configuration
        request: FastAPI request object
        background_tasks: Background tasks manager
        user: Authenticated user information
        
    Returns:
        FineTuningResponse containing job information
        
    Raises:
        HTTPException: If fine-tuning job creation fails
    """
    try:
        fine_tuning_manager = request.app.state.fine_tuning_manager
        job_id = fine_tuning_manager.start_training(
            model_name=request_body.model_name,
            training_data=request_body.training_data,
            epochs=request_body.epochs,
            batch_size=request_body.batch_size,
            learning_rate=request_body.learning_rate
        )
        
        return FineTuningResponse(
            job_id=job_id,
            status="started",
            created_at=request.app.state.fine_tuning_manager.get_job_created_at(job_id),
            updated_at=request.app.state.fine_tuning_manager.get_job_updated_at(job_id)
        )
        
    except Exception as e:
        logger.error(f"Error starting fine-tuning: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{job_id}", response_model=FineTuningResponse, status_code=status.HTTP_200_OK)
async def get_fine_tuning_status(
    job_id: str,
    request: Request,
    user: dict = Depends(validate_token)
) -> FineTuningResponse:
    """
    Get the status of a fine-tuning job.
    
    Args:
        job_id: ID of the fine-tuning job
        request: FastAPI request object
        user: Authenticated user information
        
    Returns:
        FineTuningResponse containing job status
        
    Raises:
        HTTPException: If job not found or status retrieval fails
    """
    try:
        fine_tuning_manager = request.app.state.fine_tuning_manager
        status = fine_tuning_manager.get_job_status(job_id)
        
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Fine-tuning job {job_id} not found"
            )
            
        return FineTuningResponse(
            job_id=job_id,
            status=status["status"],
            progress=status.get("progress"),
            metrics=status.get("metrics"),
            error=status.get("error"),
            created_at=status["created_at"],
            updated_at=status["updated_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting fine-tuning status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) 