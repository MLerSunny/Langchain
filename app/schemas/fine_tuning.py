from typing import List, Dict, Optional
from pydantic import BaseModel, ConfigDict

from core.settings import settings

class FineTuningRequest(BaseModel):
    """Request model for starting a fine-tuning job."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    training_data: List[Dict[str, str]]
    epochs: Optional[int] = settings.get("training.epochs", 3)
    batch_size: Optional[int] = settings.get("training.batch_size", 32)
    learning_rate: Optional[float] = settings.get("training.learning_rate", 2e-5)

class FineTuningResponse(BaseModel):
    """Response model for fine-tuning job status."""
    job_id: str
    status: str
    progress: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    created_at: str
    updated_at: str 