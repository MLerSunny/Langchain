import logging
from typing import List, Dict, Any, Union, Tuple
from langchain_core.documents import Document
from .base import BaseLLMBackend

logger = logging.getLogger(__name__)

class VLLMBackend(BaseLLMBackend):
    """vLLM backend implementation."""
    
    def __init__(self):
        self.llm = None
        self.sampling_params = None
        self.metrics = {
            "generation_latency": 0.0,
            "token_usage": 0,
            "successful_generations": 0,
            "failed_generations": 0
        }
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize vLLM with configuration."""
        try:
            # Import vLLM here to avoid import errors when not using this backend
            try:
                from vllm import LLM, SamplingParams
            except ImportError as e:
                logger.error("Failed to import vLLM. Please install it with: pip install vllm")
                raise ImportError("vLLM is not installed. Please install it with: pip install vllm") from e
            
            llm_config = config.get('llm', {})
            self.sampling_params = SamplingParams(
                temperature=llm_config.get('temperature', 0.7),
                top_p=llm_config.get('top_p', 0.95),
                max_tokens=llm_config.get('max_tokens', 4096),
                frequency_penalty=llm_config.get('frequency_penalty', 0.0),
                presence_penalty=llm_config.get('presence_penalty', 0.0)
            )
            
            logger.info(f"Initializing vLLM with model: {llm_config['model']}")
            self.llm = LLM(
                model=llm_config['model'],
                dtype="auto",
                tensor_parallel_size=config.get('optimization', {}).get('tensor_parallel_size', 1),
                gpu_memory_utilization=config.get('optimization', {}).get('gpu_memory_utilization', 0.9),
                max_model_len=llm_config.get('max_tokens', 4096)
            )
            logger.info("vLLM backend initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vLLM: {str(e)}")
            raise
    
    def generate(
        self,
        prompt: str,
        documents: List[Document],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[str, Tuple[str, List[Document]]]:
        """Generate a response using vLLM."""
        try:
            if not self.llm:
                raise RuntimeError("vLLM backend not initialized")
            
            # Prepare context from retrieved documents
            context = "\n".join([doc.page_content for doc in documents])
            full_prompt = prompt.format(context=context)
            
            # Generate response
            outputs = self.llm.generate(
                prompts=[full_prompt],
                sampling_params=self.sampling_params
            )
            response = outputs[0].outputs[0].text
            
            # Update metrics
            self.metrics["successful_generations"] += 1
            self.metrics["token_usage"] += len(response.split())
            
            return response, documents
        except Exception as e:
            logger.error(f"Error generating response with vLLM: {str(e)}")
            self.metrics["failed_generations"] += 1
            raise
    
    def cleanup(self) -> None:
        """Cleanup vLLM resources."""
        try:
            if self.llm:
                del self.llm
                self.llm = None
            if self.sampling_params:
                del self.sampling_params
                self.sampling_params = None
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get vLLM metrics."""
        return self.metrics 