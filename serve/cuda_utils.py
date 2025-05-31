import torch
import logging

logger = logging.getLogger(__name__)

def get_device():
    """Get the appropriate device for model inference."""
    try:
        if torch.cuda.is_available():
            # Check if CUDA is properly initialized
            torch.cuda.init()
            # Get available memory
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            if free_memory < 2 * 1024 * 1024 * 1024:  # Less than 2GB free
                logger.warning("CUDA memory is low, falling back to CPU")
                return "cpu"
            return "cuda"
        return "cpu"
    except Exception as e:
        logger.warning(f"Error initializing CUDA: {e}, falling back to CPU")
        return "cpu"

def clear_cuda_memory():
    """Clear CUDA memory if available."""
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Error clearing CUDA memory: {e}") 