import os
import logging
from typing import List, Dict, Any, Union, Tuple
from langchain_core.documents import Document
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from .base import BaseLLMBackend
from core.settings import settings
from core.exceptions import RAGError

logger = logging.getLogger(__name__)

class HuggingFaceBackend(BaseLLMBackend):
    """HuggingFace backend implementation."""
    
    def __init__(self):
        self.llm = None
        self.tokenizer = None
        self.model = None
        self.metrics = {
            "generation_latency": 0.0,
            "token_usage": 0,
            "successful_generations": 0,
            "failed_generations": 0
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize HuggingFace model with configuration."""
        try:
            llm_config = config.get('llm', {})
            model_name = llm_config['model']
            
            # Get model loading parameters
            use_8bit = llm_config.get('use_8bit', False)
            device_map = llm_config.get('device_map', 'auto')
            max_memory = llm_config.get('max_memory', {0: "3GB"})
            
            # Initialize tokenizer
            logger.info(f"Loading tokenizer for model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=config.get('huggingface_token'),
                trust_remote_code=True
            )
            
            # Initialize model with appropriate parameters
            logger.info(f"Loading model: {model_name}")
            model_kwargs = {
                "token": config.get('huggingface_token'),
                "use_safetensors": True,
                "device_map": device_map,
                "torch_dtype": "auto",
                "trust_remote_code": True
            }
            
            if use_8bit:
                model_kwargs.update({
                    "load_in_8bit": True,
                    "max_memory": max_memory,
                    "llm_int8_enable_fp32_cpu_offload": True
                })
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # Create pipeline
            logger.info("Creating text generation pipeline")
            pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=llm_config.get('max_tokens', 1024),
                temperature=llm_config.get('temperature', 0.7),
                top_p=llm_config.get('top_p', 0.95),
                repetition_penalty=llm_config.get('repetition_penalty', 1.15),
                device_map=device_map
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info("HuggingFace backend initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing HuggingFace model: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        documents: List[Document],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[str, Tuple[str, List[Document]]]:
        """Generate a response using HuggingFace model."""
        try:
            # Prepare context from retrieved documents
            context = "\n".join([doc.page_content for doc in documents])
            full_prompt = prompt.format(context=context)
            
            # Generate response
            response = self.llm(full_prompt)
            
            # Update metrics
            self.metrics["successful_generations"] += 1
            self.metrics["token_usage"] += len(response.split())
            
            return response, documents
        except Exception as e:
            logger.error(f"Error generating response with HuggingFace: {str(e)}")
            self.metrics["failed_generations"] += 1
            raise
    
    def cleanup(self) -> None:
        """Cleanup HuggingFace resources."""
        try:
            if self.model:
                del self.model
                self.model = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            if self.llm:
                del self.llm
                self.llm = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get HuggingFace metrics."""
        return self.metrics 