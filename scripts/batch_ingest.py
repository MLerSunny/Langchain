"""
Batch document ingestion functionality.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path

from langchain_core.documents import Document
from scripts.convert_to_sharegpt import load_documents, split_documents

logger = logging.getLogger(__name__)

def process_directory(
    directory_path: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    max_chunks: Optional[int] = None,
    allowed_extensions: Optional[List[str]] = None
) -> List[Document]:
    """
    Process all documents in a directory and its subdirectories.
    
    Args:
        directory_path: Path to the directory containing documents
        chunk_size: Size of document chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        max_chunks: Maximum number of chunks to process (None for all)
        allowed_extensions: List of allowed file extensions (None for all)
        
    Returns:
        List of processed document chunks
    """
    if allowed_extensions is None:
        allowed_extensions = ['.txt', '.pdf', '.docx', '.md']
    
    all_documents = []
    directory = Path(directory_path)
    
    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            # Check if file extension is allowed
            if file_ext not in allowed_extensions:
                logger.info(f"Skipping file with unsupported extension: {file_path}")
                continue
            
            try:
                # Load document
                documents = load_documents(file_path)
                if not documents:
                    logger.warning(f"No content found in file: {file_path}")
                    continue
                
                # Split into chunks
                chunks = split_documents(documents, chunk_size, chunk_overlap, max_chunks)
                all_documents.extend(chunks)
                
                logger.info(f"Successfully processed file: {file_path}")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                continue
    
    return all_documents 