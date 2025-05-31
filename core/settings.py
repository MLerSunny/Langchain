"""
Settings management for the RAG system.
"""

import os
from pathlib import Path
from .config import Config

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Default configuration paths
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "rag.yaml")
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_VECTOR_STORE_DIR = os.path.join(DEFAULT_DATA_DIR, "vector_store")

# Create settings instance
settings = Config(DEFAULT_CONFIG_PATH)

# Export commonly used paths
DATA_DIR = settings.get("paths.data_dir", DEFAULT_DATA_DIR)
CHROMA_PERSIST_DIRECTORY = settings.get("vector_store.persist_directory", DEFAULT_VECTOR_STORE_DIR)
TRAINING_DATA_DIR = settings.get("paths.training_data_dir", os.path.join(DEFAULT_DATA_DIR, "training", "data"))
MODELS_DIR = settings.get("paths.models_dir", os.path.join(DEFAULT_DATA_DIR, "models"))
LOGS_DIR = settings.get("paths.logs_dir", os.path.join(DEFAULT_DATA_DIR, "logs"))
CACHE_DIR = settings.get("paths.cache_dir", os.path.join(DEFAULT_DATA_DIR, "cache"))
METRICS_DIR = settings.get("paths.metrics_dir", os.path.join(DEFAULT_DATA_DIR, "metrics"))
JOBS_DIR = settings.get("paths.jobs_dir", os.path.join(DEFAULT_DATA_DIR, "jobs"))

# Export server settings
HOST = settings.get("server.host", "0.0.0.0")
FASTAPI_PORT = settings.get("server.fastapi_port", 8000)
STREAMLIT_PORT = settings.get("server.streamlit_port", 8501)

# Export model settings
MODEL_NAME = settings.get("llm.model")
EMBEDDING_MODEL = settings.get("embeddings.model")

# Export RAG settings
CHUNK_SIZE = settings.get("chunking.chunk_size", 1000)
CHUNK_OVERLAP = settings.get("chunking.chunk_overlap", 200)
MAX_CHUNKS = settings.get("chunking.max_chunks", 10)

# Export security settings
API_KEY = settings.get("security.api_key")
ALLOWED_ORIGINS = settings.get("security.allowed_origins", ["http://localhost:8501"])
RATE_LIMIT = settings.get("security.rate_limit", 100)

# Export optimization settings
USE_GPU = settings.get("optimization.use_gpu", False)
BATCH_SIZE = settings.get("optimization.batch_size", 64)
MAX_WORKERS = settings.get("optimization.max_workers", 8)
USE_FP16 = settings.get("optimization.use_fp16", True)
USE_FLASH_ATTENTION = settings.get("optimization.use_flash_attention", True)

# Export monitoring settings
MONITORING_ENABLED = settings.get("monitoring.enabled", True)
LOG_LEVEL = settings.get("monitoring.log_level", "INFO")
METRICS_ENABLED = settings.get("monitoring.metrics", [])

# Export error handling settings
RETRY_ATTEMPTS = settings.get("error_handling.retry_attempts", 3)
RETRY_DELAY = settings.get("error_handling.retry_delay", 1000)
FALLBACK_RESPONSES = settings.get("error_handling.fallback_responses", {})

# Export feature flags
FEATURES = settings.get("features", {})

# Export Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") 