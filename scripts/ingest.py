#!/usr/bin/env python3
"""
Document ingestion script.

This script handles loading documents from various sources,
chunking them, embedding them, and storing them in a vector database.
"""

import argparse
import logging
import os
import sys
from typing import List, Optional

import chromadb
from langchain.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    PDFMinerLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.settings import (
    CHROMA_PERSIST_DIRECTORY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Map file extensions to document loaders
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".pdf": (PyPDFLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def load_documents(source_dir: str) -> List:
    """
    Load documents from the specified directory.

    Args:
        source_dir: Path to the directory containing documents.

    Returns:
        List of loaded documents.
    """
    if not os.path.exists(source_dir):
        logger.error(f"Directory '{source_dir}' does not exist")
        sys.exit(1)

    all_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            # Skip hidden files
            if file.startswith("."):
                continue
                
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext in LOADER_MAPPING:
                all_files.append(file_path)
                
    if not all_files:
        logger.error(f"No supported documents found in '{source_dir}'")
        sys.exit(1)
        
    logger.info(f"Found {len(all_files)} files to process")
    
    # Load documents using the appropriate loader
    documents = []
    for file_path in all_files:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in LOADER_MAPPING:
            loader_class, loader_args = LOADER_MAPPING[file_ext]
            try:
                logger.info(f"Loading '{file_path}'")
                loader = loader_class(file_path, **loader_args)
                documents.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading '{file_path}': {e}")
    
    return documents


def process_documents(
    documents: List,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List:
    """
    Process documents by splitting them into chunks.

    Args:
        documents: List of documents to process.
        chunk_size: Size of each chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of processed document chunks.
    """
    if not documents:
        logger.error("No documents to process")
        sys.exit(1)
        
    logger.info(f"Splitting {len(documents)} documents into chunks")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")
    
    return chunks


def create_vectorstore(
    chunks: List,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    embedding_model_name: str = EMBEDDING_MODEL,
) -> None:
    """
    Create a vector store from the document chunks.

    Args:
        chunks: List of document chunks.
        persist_directory: Directory to persist the vector store.
        embedding_model_name: Name of the embedding model to use.
    """
    if not chunks:
        logger.error("No chunks to embed")
        sys.exit(1)
        
    logger.info(f"Creating embeddings using '{embedding_model_name}'")
    
    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    
    # Create and persist a Chroma vector store
    logger.info(f"Creating vector store at '{persist_directory}'")
    try:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )
        vectorstore.persist()
        logger.info(f"Vector store created successfully with {len(chunks)} documents")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        sys.exit(1)


def main():
    """Main function to run the ingestion process."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into a vector database"
    )
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        required=True,
        help="Source directory containing documents to ingest",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"Chunk size for document splitting (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help=f"Chunk overlap for document splitting (default: {CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default=CHROMA_PERSIST_DIRECTORY,
        help=f"Directory to persist vector store (default: {CHROMA_PERSIST_DIRECTORY})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help=f"Embedding model to use (default: {EMBEDDING_MODEL})",
    )
    
    args = parser.parse_args()
    
    logger.info("Starting document ingestion process")
    
    # Load documents
    documents = load_documents(args.source)
    
    # Process documents
    chunks = process_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    
    # Create and persist vector store
    create_vectorstore(
        chunks,
        persist_directory=args.persist_dir,
        embedding_model_name=args.embedding_model,
    )
    
    logger.info("Document ingestion completed successfully")


if __name__ == "__main__":
    main() 