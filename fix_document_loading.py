#!/usr/bin/env python
"""
Fix document loading issues in the ShareGPT conversion UI.
This script diagnoses why some document types are not being displayed in the UI.
"""

import os
import sys
import logging
import traceback
from typing import List, Dict, Any

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add parent directory to path for importing project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import document loaders
from scripts.convert_to_sharegpt import load_documents, split_documents
from core.settings import settings

def diagnose_document_loading():
    """
    Diagnose document loading issues by testing each step separately.
    """
    # Test directory
    test_dir = "data/raw/auto/CA"
    logger.info(f"Testing document loading from {test_dir}")
    
    # Step 1: Validate the directory exists
    if not os.path.exists(test_dir):
        logger.error(f"Directory {test_dir} does not exist!")
        return
    
    # Step 2: List files in the directory
    files = os.listdir(test_dir)
    logger.info(f"Files in directory: {files}")
    
    # Step 3: Test load_documents function
    try:
        logger.info("Testing load_documents function...")
        documents = load_documents(test_dir)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Step 4: Print document details for debugging
        for i, doc in enumerate(documents):
            logger.info(f"Document {i+1}:")
            logger.info(f"  Source: {doc.metadata.get('source')}")
            logger.info(f"  Topic: {doc.metadata.get('topic')}")
            logger.info(f"  Content length: {len(doc.page_content)} characters")
            logger.info(f"  First 100 chars: {doc.page_content[:100]}...")
        
        # Step 5: Test split_documents function
        logger.info("Testing split_documents function...")
        chunks = split_documents(documents, chunk_size=1024, chunk_overlap=128)
        logger.info(f"Split into {len(chunks)} chunks")
        
        # Step 6: Check if each original document has chunks
        doc_sources = set(doc.metadata.get('source') for doc in documents)
        chunk_sources = set(chunk.metadata.get('source') for chunk in chunks)
        
        for source in doc_sources:
            if source not in chunk_sources:
                logger.warning(f"Document {source} has no chunks!")
            else:
                source_chunks = [c for c in chunks if c.metadata.get('source') == source]
                logger.info(f"Document {source} has {len(source_chunks)} chunks")
        
        # Step 7: Print sample chunks
        for i, chunk in enumerate(chunks[:3]):
            logger.info(f"Sample chunk {i+1}:")
            logger.info(f"  Source: {chunk.metadata.get('source')}")
            logger.info(f"  Topic: {chunk.metadata.get('topic')}")
            logger.info(f"  Content length: {len(chunk.page_content)} characters")
            logger.info(f"  First 100 chars: {chunk.page_content[:100]}...")
        
        # Step 8: Check for small documents that might not split properly
        small_docs = [doc for doc in documents if len(doc.page_content) < 300]
        if small_docs:
            logger.warning(f"Found {len(small_docs)} small documents that might not split properly:")
            for doc in small_docs:
                logger.warning(f"  {doc.metadata.get('source')} - Length: {len(doc.page_content)}")
    
    except Exception as e:
        logger.error(f"Error during document loading: {e}")
        logger.error(traceback.format_exc())

def fix_document_loading():
    """
    Apply fixes to the document loading process based on diagnosis.
    """
    # This will be implemented after diagnostics reveals the issue
    logger.info("Implementing fixes will be done after diagnostics completes")

if __name__ == "__main__":
    logger.info("Starting document loading diagnosis")
    diagnose_document_loading()
    logger.info("Diagnosis complete") 