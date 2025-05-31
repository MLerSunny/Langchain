#!/usr/bin/env python
"""
Initialize the vector database with proper configuration.
"""

import os
import sys
import logging
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.settings import settings, CHROMA_PERSIST_DIRECTORY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load configuration
config_path = Path('config/rag.yaml')
if not config_path.exists():
    raise FileNotFoundError("config/rag.yaml not found. Please ensure the configuration file exists in the config directory.")

with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

def init_vector_db():
    """Initialize the vector database."""
    try:
        # Create vector store directory if it doesn't exist
        persist_dir = Path(config['vector_store']['persist_directory'])
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Vector database initialized at {persist_dir.absolute()}")
        return True
    except Exception as e:
        logger.error(f"Error initializing vector database: {str(e)}")
        return False

def main():
    """Main function to initialize vector database."""
    try:
        if init_vector_db():
            logger.info("Vector database initialization completed successfully!")
        else:
            logger.error("Vector database initialization failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during vector database initialization: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 