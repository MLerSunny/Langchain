import logging
from typing import Dict, Type
from .base import BaseLLMBackend

logger = logging.getLogger(__name__)

class LLMBackendFactory:
    """Factory for creating and managing LLM backends."""
    
    _backends: Dict[str, Type[BaseLLMBackend]] = {}
    
    @classmethod
    def get_backend(cls, backend_name: str) -> BaseLLMBackend:
        """
        Get a backend instance by name.
        
        Args:
            backend_name: Name of the backend ('vllm' or 'huggingface')
            
        Returns:
            An instance of the requested backend
            
        Raises:
            ValueError: If the backend name is not supported
        """
        if backend_name not in cls._backends:
            if backend_name == 'vllm':
                from .vllm_backend import VLLMBackend
                cls._backends['vllm'] = VLLMBackend
            elif backend_name == 'huggingface':
                from .huggingface_backend import HuggingFaceBackend
                cls._backends['huggingface'] = HuggingFaceBackend
            else:
                raise ValueError(
                    f"Unsupported backend: {backend_name}. "
                    f"Supported backends: ['vllm', 'huggingface']"
                )
        
        return cls._backends[backend_name]()
    
    @classmethod
    def register_backend(cls, name: str, backend_class: Type[BaseLLMBackend]) -> None:
        """
        Register a new backend.
        
        Args:
            name: Name to register the backend under
            backend_class: The backend class to register
        """
        if not issubclass(backend_class, BaseLLMBackend):
            raise ValueError("Backend class must inherit from BaseLLMBackend")
        
        cls._backends[name] = backend_class
        logger.info(f"Registered new backend: {name}")
    
    @classmethod
    def get_supported_backends(cls) -> list[str]:
        """Get list of supported backend names."""
        return ['vllm', 'huggingface'] 