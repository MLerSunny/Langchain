import logging
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import uuid
from datetime import datetime
import threading
import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from datasets import Dataset

from core.settings import settings

logger = logging.getLogger(__name__)

class FineTuningManager:
    def __init__(self):
        """Initialize fine-tuning manager."""
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.job_lock = threading.Lock()
        
        # Create jobs directory
        self.jobs_dir = Path('data/jobs')
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing jobs
        self._load_jobs()
    
    def start_training(
        self,
        model_name: str,
        training_data: List[Dict[str, str]],
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-5
    ) -> str:
        """Start a fine-tuning job."""
        try:
            job_id = str(uuid.uuid4())
            
            # Create job record
            self.jobs[job_id] = {
                "model_name": model_name,
                "status": "pending",
                "start_time": datetime.now().isoformat(),
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "training_data_size": len(training_data),
                "progress": 0,
                "error": None
            }
            
            # TODO: Implement actual fine-tuning logic here
            # For now, we'll just simulate a job
            logger.info(f"Started fine-tuning job {job_id} for model {model_name}")
            
            return job_id
            
        except Exception as e:
            logger.error(f"Error starting fine-tuning job: {str(e)}")
            raise
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a fine-tuning job."""
        try:
            if job_id not in self.jobs:
                raise ValueError(f"Job {job_id} not found")
                
            return self.jobs[job_id]
            
        except Exception as e:
            logger.error(f"Error getting job status: {str(e)}")
            raise
    
    def _run_fine_tuning(
        self,
        job_id: str,
        model_name: str,
        training_data: List[Dict[str, str]],
        epochs: int,
        batch_size: int,
        learning_rate: float
    ):
        """Run fine-tuning process."""
        try:
            # Update job status
            with self.job_lock:
                self.jobs[job_id]['status'] = 'running'
                self._save_job(job_id)
            
            # Load model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Prepare dataset
            dataset = Dataset.from_list(training_data)
            
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'],
                    padding='max_length',
                    truncation=True,
                    max_length=512
                )
            
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=f"data/models/{job_id}",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                learning_rate=learning_rate,
                save_strategy="epoch",
                evaluation_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="loss"
            )
            
            # Set up trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
            )
            
            # Train model
            trainer.train()
            
            # Save model
            trainer.save_model()
            
            # Update job status
            with self.job_lock:
                self.jobs[job_id].update({
                    'status': 'completed',
                    'end_time': datetime.now().isoformat(),
                    'progress': 100,
                    'metrics': trainer.state.log_history[-1]
                })
                self._save_job(job_id)
            
        except Exception as e:
            logger.error(f"Error in fine-tuning job {job_id}: {e}")
            
            # Update job status
            with self.job_lock:
                self.jobs[job_id].update({
                    'status': 'failed',
                    'end_time': datetime.now().isoformat(),
                    'error': str(e)
                })
                self._save_job(job_id)
    
    def _save_job(self, job_id: str):
        """Save job info to disk."""
        try:
            job_file = self.jobs_dir / f"{job_id}.json"
            with open(job_file, 'w') as f:
                json.dump(self.jobs[job_id], f)
                
        except Exception as e:
            logger.error(f"Error saving job {job_id}: {e}")
            raise
    
    def _load_jobs(self):
        """Load existing jobs from disk."""
        try:
            for job_file in self.jobs_dir.glob("*.json"):
                with open(job_file, 'r') as f:
                    job_info = json.load(f)
                    self.jobs[job_info['id']] = job_info
                    
        except Exception as e:
            logger.error(f"Error loading jobs: {e}")
            raise 