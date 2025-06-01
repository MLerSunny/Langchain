from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, Tuple
from langchain_core.documents import Document

class BaseLLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the LLM backend with configuration."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        documents: List[Document],
        stream: bool = False,
        **kwargs: Any
    ) -> Union[str, Tuple[str, List[Document]]]:
        """Generate a response using the LLM."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about the LLM's performance."""
        pass 