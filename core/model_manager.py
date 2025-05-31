"""
Model Manager for handling both HuggingFace and Ollama models.
Provides a unified interface for model operations including loading, fine-tuning, and conversion.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from core.settings import settings
from core.model_optimizer import get_optimized_model, get_optimized_training_args

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model operations for both HuggingFace and Ollama models."""
    
    def __init__(self, base_model_path: str = None):
        """
        Initialize the model manager.
        
        Args:
            base_model_path: Path to the base model (HuggingFace format)
        """
        self.base_model_path = base_model_path or settings.get('llm.model')
        self.models_dir = Path(settings.get('paths.models_dir', 'data/models'))
        self.ollama_models_dir = Path(os.path.expanduser("~/.ollama/models"))
        
    def download_huggingface_model(self, model_name: str) -> str:
        """
        Download a model from HuggingFace Hub.
        
        Args:
            model_name: Name of the model on HuggingFace Hub
            
        Returns:
            Path to the downloaded model
        """
        try:
            logger.info(f"Downloading model {model_name} from HuggingFace Hub")
            model_path = self.models_dir / model_name.split("/")[-1]
            
            # Download model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Save locally
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            logger.info(f"Model downloaded and saved to {model_path}")
            return str(model_path)
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise
    
    def load_model_for_training(
        self,
        use_4bit: bool = True,
        use_8bit: bool = False,
        use_flash_attention: bool = True,
        use_lora: bool = True,
        lora_rank: int = 64,
        lora_alpha: int = 128,
        lora_dropout: float = 0.05
    ) -> Tuple[Any, Any]:
        """
        Load model for fine-tuning with optimizations.
        
        Args:
            use_4bit: Whether to use 4-bit quantization
            use_8bit: Whether to use 8-bit quantization
            use_flash_attention: Whether to use flash attention
            use_lora: Whether to use LoRA
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            
        Returns:
            Tuple of (model, tokenizer)
        """
        return get_optimized_model(
            model_name=self.base_model_path,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            use_flash_attention=use_flash_attention,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
    
    def save_fine_tuned_model(
        self,
        model: Any,
        tokenizer: Any,
        output_name: str,
        merge_lora: bool = True
    ) -> str:
        """
        Save a fine-tuned model.
        
        Args:
            model: The fine-tuned model
            tokenizer: The tokenizer
            output_name: Name for the saved model
            merge_lora: Whether to merge LoRA weights with base model
            
        Returns:
            Path to the saved model
        """
        try:
            output_path = self.models_dir / output_name
            
            if merge_lora and isinstance(model, PeftModel):
                logger.info("Merging LoRA weights with base model")
                model = model.merge_and_unload()
            
            # Save model and tokenizer
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            logger.info(f"Model saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def convert_to_ollama_format(
        self,
        model_path: str,
        ollama_name: str
    ) -> bool:
        """
        Convert a HuggingFace model to Ollama format.
        
        Args:
            model_path: Path to the HuggingFace model
            ollama_name: Name for the Ollama model
            
        Returns:
            True if conversion successful
        """
        try:
            # Create Ollama Modelfile
            modelfile_path = Path(model_path) / "Modelfile"
            with open(modelfile_path, "w") as f:
                f.write(f"""FROM {model_path}
PARAMETER temperature 0.7
PARAMETER top_p 0.7
PARAMETER stop "Human:"
PARAMETER stop "Assistant:"
""")
            
            # Create Ollama model
            import subprocess
            result = subprocess.run(
                ["ollama", "create", ollama_name, "-f", str(modelfile_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info(f"Successfully created Ollama model: {ollama_name}")
                return True
            else:
                logger.error(f"Error creating Ollama model: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error converting model to Ollama format: {e}")
            return False
    
    def get_available_models(self) -> Dict[str, list]:
        """
        Get lists of available models for both HuggingFace and Ollama.
        
        Returns:
            Dictionary with 'huggingface' and 'ollama' model lists
        """
        models = {
            "huggingface": [],
            "ollama": []
        }
        
        # Get HuggingFace models
        if self.models_dir.exists():
            models["huggingface"] = [
                d.name for d in self.models_dir.iterdir()
                if d.is_dir() and (d / "config.json").exists()
            ]
        
        # Get Ollama models
        if self.ollama_models_dir.exists():
            models["ollama"] = [
                d.name for d in self.ollama_models_dir.iterdir()
                if d.is_dir()
            ]
        
        return models 