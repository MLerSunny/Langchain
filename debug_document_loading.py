#!/usr/bin/env python
"""
Debug script to test document loading from various file types.
This will help diagnose why text files aren't being processed.
"""

import os
import sys
import logging

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import document loaders
from langchain_community.document_loaders import (
    PyPDFLoader, 
    TextLoader,
    UnstructuredFileLoader
)
from core.settings import settings

def test_load_specific_files():
    """
    Attempt to load specific files with detailed debugging.
    """
    logger.info("Starting debug document loading test")
    
    # Directory to test
    test_dir = "data/raw/auto/CA"
    logger.info(f"Testing document loading from directory: {test_dir}")
    
    # List all files
    if os.path.exists(test_dir):
        files = os.listdir(test_dir)
        logger.info(f"Files in directory: {files}")
    else:
        logger.error(f"Directory {test_dir} does not exist")
        return
    
    # Try to load each file with specific loaders
    for file in files:
        file_path = os.path.join(test_dir, file)
        file_ext = os.path.splitext(file)[1].lower()
        
        logger.info(f"\n----- Testing file: {file} -----")
        logger.info(f"Extension: {file_ext}")
        
        try:
            # Test PDF loader for PDF files
            if file_ext == ".pdf":
                logger.info("Attempting to load with PyPDFLoader")
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                logger.info(f"Successfully loaded {len(docs)} pages with PyPDFLoader")
            
            # Test Text loader for TXT files
            elif file_ext == ".txt":
                # Try with different encodings
                encodings = ["utf-8", "latin-1", "utf-16", "cp1252"]
                for encoding in encodings:
                    try:
                        logger.info(f"Attempting to load with TextLoader using {encoding} encoding")
                        loader = TextLoader(file_path, encoding=encoding)
                        docs = loader.load()
                        logger.info(f"Successfully loaded with TextLoader using {encoding} encoding")
                        logger.info(f"Content preview: {docs[0].page_content[:100]}...")
                        break
                    except Exception as e:
                        logger.warning(f"Failed with encoding {encoding}: {e}")
                
                # Also try with UnstructuredFileLoader as fallback
                try:
                    logger.info("Attempting to load with UnstructuredFileLoader")
                    loader = UnstructuredFileLoader(file_path)
                    docs = loader.load()
                    logger.info(f"Successfully loaded with UnstructuredFileLoader")
                    logger.info(f"Content preview: {docs[0].page_content[:100]}...")
                except Exception as e:
                    logger.warning(f"Failed with UnstructuredFileLoader: {e}")
            
            # Try generic loader for other files
            else:
                logger.info("Attempting to load with UnstructuredFileLoader")
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
                logger.info(f"Successfully loaded with UnstructuredFileLoader")
        
        except Exception as e:
            logger.error(f"Error loading {file_path}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_load_specific_files()
    logger.info("Debug test completed") 