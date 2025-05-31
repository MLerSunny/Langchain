class RAGError(Exception):
    """Base exception for RAG system errors."""
    pass

class RetrievalError(RAGError):
    """Exception for retrieval errors in RAG."""
    pass

class GenerationError(RAGError):
    """Exception for generation errors in RAG."""
    pass

class ConfigError(RAGError):
    """Exception for configuration errors in RAG."""
    pass 