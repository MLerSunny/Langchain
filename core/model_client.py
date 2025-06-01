"""
Model Client for handling model inference and client-side operations.
Provides a unified interface for model interactions.
"""

import logging
from typing import Dict, Any, Optional, List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.settings import settings
from core.model_manager import ModelManager

logger = logging.getLogger(__name__)

class ModelClient:
    """Handles model inference and client-side operations."""
    
    def __init__(self):
        """Initialize the model client."""
        self.model_manager = ModelManager()
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    async def load_model(self, model_name: Optional[str] = None) -> None:
        """
        Load a model for inference.
        
        Args:
            model_name: Name of the model to load (optional)
        """
        try:
            model_name = model_name or settings.get('llm.model')
            logger.info(f"Loading model: {model_name}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing generated text and metadata
        """
        try:
            if not self.model or not self.tokenizer:
                await self.load_model()
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {
                "text": generated_text,
                "metadata": {
                    "model": self.model.name_or_path,
                    "max_length": max_length,
                    "temperature": temperature,
                    "top_p": top_p
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    async def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get list of available models.
        
        Returns:
            Dictionary containing lists of available models
        """
        return self.model_manager.get_available_models()
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            if self.model:
                del self.model
            if self.tokenizer:
                del self.tokenizer
            torch.cuda.empty_cache()
            logger.info("Model client resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise 