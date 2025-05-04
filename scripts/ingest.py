#!/usr/bin/env python
"""
Script for ingesting documents and creating vector embeddings for RAG.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Add the project root to Python path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from core.settings import CHROMA_PERSIST_DIRECTORY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def create_vector_db_from_documents(
    documents: List[Document],
    persist_directory: Optional[str] = CHROMA_PERSIST_DIRECTORY,
    embedding_model_name: str = "all-MiniLM-L6-v2"
) -> Any:
    """
    Create a vector database from documents and persist it to disk.
    
    Args:
        documents: List of documents to add to vector store
        persist_directory: Directory to persist the vector store
        embedding_model_name: Name of the embedding model to use
    
    Returns:
        The Chroma vector store object
    """
    logger.info(f"Creating vector database at {persist_directory} with {len(documents)} documents")
    
    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Initialize the embedding function
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create and persist the vector store
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the vector store
    vectorstore.persist()
    
    logger.info(f"Vector database created with {len(documents)} documents and persisted to {persist_directory}")
    
    return vectorstore

def main():
    """
    Main function for ingesting documents from command line.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest documents and create vector embeddings for RAG"
    )
    
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing documents to process"
    )
    
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=CHROMA_PERSIST_DIRECTORY,
        help="Directory to persist vector store"
    )
    
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the embedding model to use"
    )
    
    args = parser.parse_args()
    
    # Import here to avoid circular imports
    from convert_to_sharegpt import load_documents, split_documents
    
    # Load and split documents
    documents = load_documents(args.input_dir)
    chunks = split_documents(documents)
    
    # Create vector store
    create_vector_db_from_documents(
        documents=chunks,
        persist_directory=args.persist_dir,
        embedding_model_name=args.embedding_model
    )

if __name__ == "__main__":
    main() 