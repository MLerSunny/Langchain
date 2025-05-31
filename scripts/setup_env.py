#!/usr/bin/env python
"""
Setup environment for the RAG system.
"""

import os
import sys
import json
import yaml
from pathlib import Path
import shutil
from dotenv import load_dotenv

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.settings import settings, DEFAULT_CONFIG_PATH

def create_default_config():
    """Create default configuration file."""
    config_path = Path('config/rag.yaml')
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    default_config = {
        "app": {
            "name": "Insurance Assistant",
            "version": "1.0.0",
            "environment": "development",
            "debug": True,
            "log_level": "INFO"
        },
        "rag": {
            "chunk_size": 256,
            "chunk_overlap": 200,
            "max_chunks": 10,
            "similarity_threshold": 0.7,
            "cache_results": True,
            "cache_ttl": 7200
        },
        "llm": {
            "model": "deepseek-ai/deepseek-llm-7b-base",
            "temperature": 0.7,
            "max_tokens": 4096,
            "top_p": 0.95,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0
        },
        "vector_store": {
            "type": "chroma",
            "persist_directory": "data/vector_store/chroma",
            "collection_name": "documents",
            "dimension": 384,
            "index_params": {
                "hnsw_ef_construction": 100,
                "hnsw_m": 16
            }
        },
        "embeddings": {
            "model": "all-MiniLM-L6-v2",
            "cache_dir": "data/cache",
            "batch_size": 64
        },
        "security": {
            "api_key": "your-secret-key-here",
            "allowed_origins": ["http://localhost:8501"],
            "rate_limit": 100,
            "max_file_size": 10485760
        },
        "server": {
            "host": "localhost",
            "port": 8000,
            "fastapi_port": 8000,
            "streamlit_port": 8501
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)
    
    print(f"Created default rag.yaml file at {config_path.absolute()}")

def create_default_env():
    """Create default .env file."""
    env_path = Path('.env')
    
    default_env = f"""CHROMA_PERSIST_DIRECTORY=data/vector_store/chroma
DATA_DIR=data
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
HOST=localhost
"""
    
    with open(env_path, 'w') as f:
        f.write(default_env)
    
    print(f"Created default .env file at {env_path.absolute()}")

def create_directories():
    """Create necessary directories."""
    directories = [
        'data',
        'data/vector_store',
        'data/vector_store/chroma',
        'data/cache',
        'data/models',
        'data/logs',
        'data/metrics',
        'data/jobs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def main():
    """Main function to setup environment."""
    try:
        # Create default configuration
        create_default_config()
        
        # Create default .env file
        create_default_env()
        
        # Create necessary directories
        create_directories()
        
        print("Environment setup completed successfully!")
        
    except Exception as e:
        print(f"Error setting up environment: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 