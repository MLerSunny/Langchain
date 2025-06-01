"""
Settings management for the RAG system.
"""

import os
from pathlib import Path
from .config import Config
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default configuration paths
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "rag.yaml")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_VECTOR_STORE_DIR = os.path.join(DEFAULT_DATA_DIR, "vector_store")

# Create settings instance
config = Config(DEFAULT_CONFIG_PATH)

# Add JWT secret and algorithm from environment variables
def _add_jwt_settings(config):
    config.jwt_secret = os.getenv("JWT_SECRET")
    config.jwt_algorithm = os.getenv("JWT_ALGORITHM")

_add_jwt_settings(config)

# Export commonly used paths
DATA_DIR = config.get("paths.data_dir", DEFAULT_DATA_DIR)
CHROMA_PERSIST_DIRECTORY = config.get("vector_store.persist_directory", DEFAULT_VECTOR_STORE_DIR)
TRAINING_DATA_DIR = config.get("paths.training_data_dir", os.path.join(DEFAULT_DATA_DIR, "training", "data"))
MODELS_DIR = config.get("paths.models_dir", os.path.join(DEFAULT_DATA_DIR, "models"))
LOGS_DIR = config.get("paths.logs_dir", os.path.join(DEFAULT_DATA_DIR, "logs"))
CACHE_DIR = config.get("paths.cache_dir", os.path.join(DEFAULT_DATA_DIR, "cache"))
METRICS_DIR = config.get("paths.metrics_dir", os.path.join(DEFAULT_DATA_DIR, "metrics"))
JOBS_DIR = config.get("paths.jobs_dir", os.path.join(DEFAULT_DATA_DIR, "jobs"))

# Export server settings
HOST = config.get("server.host", "0.0.0.0")
FASTAPI_PORT = config.get("server.fastapi_port", 8000)
STREAMLIT_PORT = config.get("server.streamlit_port", 8501)

# Export model settings
MODEL_NAME = config.get("llm.model")
EMBEDDING_MODEL = config.get("embeddings.model")

# Export RAG settings
CHUNK_SIZE = config.get("chunking.chunk_size", 1000)
CHUNK_OVERLAP = config.get("chunking.chunk_overlap", 200)
MAX_CHUNKS = config.get("chunking.max_chunks", 10)

# Export security settings
API_KEY = config.get("security.api_key")
ALLOWED_ORIGINS = config.get("security.allowed_origins", ["http://localhost:8501"])
RATE_LIMIT = config.get("security.rate_limit", 100)

# Export optimization settings
USE_GPU = config.get("optimization.use_gpu", False)
BATCH_SIZE = config.get("optimization.batch_size", 64)
MAX_WORKERS = config.get("optimization.max_workers", 8)
USE_FP16 = config.get("optimization.use_fp16", True)
USE_FLASH_ATTENTION = config.get("optimization.use_flash_attention", True)

# Export monitoring settings
MONITORING_ENABLED = config.get("monitoring.enabled", True)
LOG_LEVEL = config.get("monitoring.log_level", "INFO")
METRICS_ENABLED = config.get("monitoring.metrics", [])

# Export error handling settings
RETRY_ATTEMPTS = config.get("error_handling.retry_attempts", 3)
RETRY_DELAY = config.get("error_handling.retry_delay", 1000)
FALLBACK_RESPONSES = config.get("error_handling.fallback_responses", {})

# Export feature flags
FEATURES = config.get("features", {})

# Create a settings dictionary that includes all configuration values
settings = {
    "config_path": DEFAULT_CONFIG_PATH,
    "llm": config.get("llm", {}),
    "vector_store": {
        "persist_directory": config.get("vector_store.persist_directory", DEFAULT_VECTOR_STORE_DIR),
        "collection_name": config.get("vector_store.collection_name", "documents"),
        "type": config.get("vector_store.type", "chroma"),
        "dimension": config.get("vector_store.dimension", 384),
        "index_params": config.get("vector_store.index_params", {}),
        "mode": config.get("vector_store.mode", "append")
    },
    "optimization": config.get("optimization", {}),
    "query": config.get("query", {}),
    "error_handling": config.get("error_handling", {}),
    "generation": config.get("generation", {}),
    "server": config.get("server", {}),
    "embeddings": {
        "model": os.path.join(MODELS_DIR, "embeddings"),
        "device": "auto",
        "trust_remote_code": True,
        "normalize_embeddings": True,
        "cache_dir": os.path.join(CACHE_DIR, "embeddings"),
        "batch_size": BATCH_SIZE,
        "local_files_only": True
    }
}

def get(key: str, default: Any = None) -> Any:
    """Get a setting value by key."""
    try:
        # Split the key into parts
        parts = key.split('.')
        
        # Start with the settings dictionary
        value = settings
        
        # Traverse the dictionary
        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
                if value is None:
                    return default
            else:
                return default
                
        return value
    except Exception as e:
        print(f"[DEBUG] Error getting setting {key}: {str(e)}")
        return default 