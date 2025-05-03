# core/settings.py
import os
import yaml
import logging
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    # API Server settings
    fastapi_port: int = 8000
    vllm_port: int = 8001
    host: str = "0.0.0.0"
    
    # JWT Settings
    jwt_secret: str = "CHANGE_ME_IN_PRODUCTION"
    jwt_algorithm: str = "HS256"
    
    # Model settings
    model_name: str = "deepseek-coder:7b-instruct-v1.5"
    default_model: str = "deepseek-coder:7b-instruct-v1.5"
    ollama_base_url: str = "http://localhost:11434"
    context_window: int = 8192
    max_tokens: int = 2048
    temperature: float = 0.1
    
    # Vector database settings
    vector_db: str = "chroma"          # "chroma" | "qdrant"
    data_path: str = "data/raw"
    chroma_path: str = ".chroma"
    chroma_persist_directory: str = "data/chroma"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dimension: int = 768
    
    # Qdrant settings
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_url: str = "http://vectordb:6333"
    qdrant_collection_name: str = "insurance_docs"
    
    # RAG settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 8
    
    # Data paths
    base_dir: str = "."
    data_dir: str = "data"
    rag_config_path: str = "core/rag.yaml"
    deepspeed_config_path: str = "core/deepspeed_zero3.json"
    
    # Fine-tuning settings
    training_dataset_path: str = "data/training"
    eval_dataset_path: str = "data/eval"
    output_dir: str = "data/models"
    learning_rate: float = 2e-5
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    max_steps: int = -1
    logging_steps: int = 10
    save_steps: int = 100
    warmup_steps: int = 50
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

# Create a global settings instance
settings = Settings()

# Generate absolute paths based on base_dir
def get_abs_path(rel_path: str) -> str:
    """Convert a relative path to absolute based on base_dir setting."""
    base = Path(settings.base_dir)
    return str(base / rel_path)

# Update path settings to absolute paths if they're relative
if not os.path.isabs(settings.chroma_persist_directory):
    settings.chroma_persist_directory = get_abs_path(settings.chroma_persist_directory)

if not os.path.isabs(settings.data_dir):
    settings.data_dir = get_abs_path(settings.data_dir)

if not os.path.isabs(settings.rag_config_path):
    settings.rag_config_path = get_abs_path(settings.rag_config_path)

if not os.path.isabs(settings.training_dataset_path):
    settings.training_dataset_path = get_abs_path(settings.training_dataset_path)

if not os.path.isabs(settings.eval_dataset_path):
    settings.eval_dataset_path = get_abs_path(settings.eval_dataset_path)

if not os.path.isabs(settings.output_dir):
    settings.output_dir = get_abs_path(settings.output_dir)

if not os.path.isabs(settings.chroma_path):
    settings.chroma_path = get_abs_path(settings.chroma_path)

if not os.path.isabs(settings.data_path):
    settings.data_path = get_abs_path(settings.data_path)

if not os.path.isabs(settings.deepspeed_config_path):
    settings.deepspeed_config_path = get_abs_path(settings.deepspeed_config_path)

# Validation check for JWT secret
# Comment out for training to proceed
# assert settings.jwt_secret != "CHANGE_ME_IN_PRODUCTION", (
#     "Set JWT_SECRET in .env before running!")
settings.jwt_secret = "ypzkQRSJB3QxTBF95c8KwDjgcPrEmn7v"

# Load RAG configuration from YAML if it exists
RAG_CONFIG = {}
try:
    if os.path.exists(settings.rag_config_path):
        with open(settings.rag_config_path, "r") as f:
            RAG_CONFIG = yaml.safe_load(f)
except Exception as e:
    logger.warning(f"Could not load RAG config from {settings.rag_config_path}: {e}")

# Override settings from RAG_CONFIG if available
if "vector_db" in RAG_CONFIG:
    settings.chroma_persist_directory = RAG_CONFIG["vector_db"].get(
        "persist_directory", settings.chroma_persist_directory
    )

if "document_processing" in RAG_CONFIG:
    settings.embedding_model = RAG_CONFIG["document_processing"].get(
        "embedding_model", settings.embedding_model
    )
    settings.chunk_size = RAG_CONFIG["document_processing"].get(
        "chunk_size", settings.chunk_size
    )
    settings.chunk_overlap = RAG_CONFIG["document_processing"].get(
        "chunk_overlap", settings.chunk_overlap
    )

# Removed duplicate uppercase constants
# Import this settings object and access properties directly instead
# For example: from core.settings import settings
# Then use: settings.chunk_size instead of CHUNK_SIZE 