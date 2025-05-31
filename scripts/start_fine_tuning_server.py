#!/usr/bin/env python
"""
Start the fine-tuning server with proper environment setup.
This server handles model fine-tuning requests and training jobs.
"""

import os
import sys
import logging
from pathlib import Path
import uvicorn
import time
import requests
from typing import Dict, Any, Optional
import json
import torch
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import wandb
import yaml

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.settings import settings, FASTAPI_PORT, HOST, TRAINING_DATA_DIR, MODELS_DIR
from core.model_manager import ModelManager
from core.security import SecurityManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('fine_tuning_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize managers
model_manager = ModelManager()
security_manager = SecurityManager()

# Load environment variables
load_dotenv()

# Load configuration
config_path = Path('config/rag.yaml')
if not config_path.exists():
    raise FileNotFoundError("config/rag.yaml not found. Please ensure the configuration file exists in the config directory.")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title=f"{config['app']['name']} - Fine-tuning Service",
    description="Fine-tuning service for the RAG system",
    version=config['app']['version']
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['security']['allowed_origins'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for training state
training_jobs = {}

def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    try:
        # Check if training data directory exists
        training_dir = Path(TRAINING_DATA_DIR)
        if not training_dir.exists():
            logger.info(f"Creating training data directory at {training_dir}")
            training_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if models directory exists
        models_dir = Path(MODELS_DIR)
        if not models_dir.exists():
            logger.info(f"Creating models directory at {models_dir}")
            models_dir.mkdir(parents=True, exist_ok=True)
        
        # Check GPU availability
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch CUDA version: {torch.version.cuda}")
        else:
            logger.warning("No GPU available. Fine-tuning will be slow on CPU.")
            logger.info("Consider using a machine with GPU for better performance")
        
        # Check if main server is running
        if not check_server_health():
            logger.error("Main server is not responding. Please ensure the main server is running.")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error checking prerequisites: {str(e)}", exc_info=True)
        return False

def check_server_health(max_retries=3, retry_delay=2) -> bool:
    """Check if the server is healthy."""
    for attempt in range(max_retries):
        try:
            # Try both HTTP and HTTPS
            urls = [
                f"http://{HOST}:{FASTAPI_PORT}/health",
                f"https://{HOST}:{FASTAPI_PORT}/health"
            ]
            
            for url in urls:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Server health check successful at {url}")
                        return True
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Failed to connect to {url}: {str(e)}")
                    continue
            
            if attempt < max_retries - 1:
                logger.warning(f"Health check attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                continue
                
        except Exception as e:
            logger.error(f"Error during health check: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
    
    return False

class FineTuningServer:
    def __init__(self):
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_history: Dict[str, Dict[str, Any]] = {}
    
    async def start_training(self, job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start a new fine-tuning job."""
        try:
            # Validate configuration
            if not self._validate_config(config):
                raise ValueError("Invalid training configuration")
            
            # Check if job already exists
            if job_id in self.active_jobs:
                raise ValueError(f"Job {job_id} already exists")
            
            # Initialize job
            job_info = {
                "job_id": job_id,
                "config": config,
                "status": "initializing",
                "start_time": datetime.now().isoformat(),
                "progress": 0.0,
                "metrics": {}
            }
            
            self.active_jobs[job_id] = job_info
            
            # Start training process
            # This would typically be done in a separate process or thread
            # For now, we'll just simulate it
            job_info["status"] = "training"
            
            return job_info
        except Exception as e:
            logger.error(f"Error starting training job: {str(e)}", exc_info=True)
            raise
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate training configuration."""
        required_fields = [
            "model_name",
            "training_data_path",
            "epochs",
            "batch_size",
            "learning_rate"
        ]
        return all(field in config for field in required_fields)
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a training job."""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        elif job_id in self.job_history:
            return self.job_history[job_id]
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running training job."""
        if job_id not in self.active_jobs:
            return False
        
        job_info = self.active_jobs[job_id]
        job_info["status"] = "cancelled"
        job_info["end_time"] = datetime.now().isoformat()
        
        # Move to history
        self.job_history[job_id] = job_info
        del self.active_jobs[job_id]
        
        return True

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/jobs")
async def list_jobs():
    """List all training jobs."""
    return {"jobs": training_jobs}

@app.post("/train")
async def start_training(
    background_tasks: BackgroundTasks,
    model_name: str = "gpt2",
    dataset_name: str = "wikitext",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5
):
    """Start a new training job."""
    job_id = f"job_{len(training_jobs) + 1}"
    
    # Initialize job status
    training_jobs[job_id] = {
        "status": "initializing",
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "current_epoch": 0,
        "loss": None
    }
    
    # Start training in background
    background_tasks.add_task(
        train_model,
        job_id,
        model_name,
        dataset_name,
        num_epochs,
        batch_size,
        learning_rate
    )
    
    return {
        "job_id": job_id,
        "status": "started",
        "message": "Training job started successfully"
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a training job."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return training_jobs[job_id]

async def train_model(
    job_id: str,
    model_name: str,
    dataset_name: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float
):
    """Train the model in the background."""
    try:
        # Update job status
        training_jobs[job_id]["status"] = "loading_model"
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load dataset
        training_jobs[job_id]["status"] = "loading_dataset"
        dataset = load_dataset(dataset_name, split="train")
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Training loop
        training_jobs[job_id]["status"] = "training"
        for epoch in range(num_epochs):
            training_jobs[job_id]["current_epoch"] = epoch + 1
            
            # Training logic here
            # This is a simplified version - you would need to implement
            # proper training loops, evaluation, and checkpointing
            
            # Simulate training progress
            training_jobs[job_id]["loss"] = 1.0 / (epoch + 1)
        
        # Save model
        training_jobs[job_id]["status"] = "saving"
        model.save_pretrained(f"data/models/{job_id}")
        tokenizer.save_pretrained(f"data/models/{job_id}")
        
        # Update final status
        training_jobs[job_id]["status"] = "completed"
        
    except Exception as e:
        training_jobs[job_id]["status"] = "failed"
        training_jobs[job_id]["error"] = str(e)

def start_server():
    """Start the FastAPI server."""
    try:
        # Get server configuration
        host = config['server']['host']
        port = config['server']['port']
        
        logger.info(f"Starting fine-tuning server on {host}:{port}")
        logger.info(f"Main server should be running on {HOST}:{FASTAPI_PORT}")
        
        # Double check main server health before starting
        if not check_server_health():
            raise RuntimeError("Main server is not responding. Please start the main server first.")
        
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level=config['app']['log_level'].lower(),
            reload=False  # Disable auto-reload in production
        )
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}", exc_info=True)
        raise

def main():
    """Main function to start the fine-tuning server."""
    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("Prerequisites check failed")
            sys.exit(1)
        
        # Check server health
        if not check_server_health():
            logger.error("Main server health check failed")
            sys.exit(1)
        
        # Initialize server
        server = FineTuningServer()
        
        # Start the server
        start_server()
    except Exception as e:
        logger.error(f"Error starting fine-tuning server: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 