"""
Ingest module for loading and splitting documents for insurance RAG system.
This module loads mixed document types from data/raw/, 
processes them with custom chunking, and stores them in Chroma.
"""

import os
import re
import sys
import argparse
import shutil
import logging
import hashlib
from typing import List, Optional, Dict, Any, Set

import chromadb
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    CSVLoader,
    UnstructuredFileLoader
)
from langchain_community.vectorstores import Chroma, Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def extract_metadata_from_path(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from file path based on directory structure.
    Assumes path structure like: data/raw/{lob}/{state}/filename
    
    Args:
        file_path: Path to the document
        
    Returns:
        Dictionary containing metadata
    """
    parts = file_path.split(os.sep)
    # Default values
    metadata = {
        "source": file_path,
        "lob": "general",  # Line of Business
        "state": "all"     # State code
    }
    
    # Try to extract LOB and State from path if available
    if len(parts) >= 4:  # data/raw/lob/state/filename
        if parts[-3].lower() not in ("raw", "data"):
            metadata["lob"] = parts[-3].lower()
        if len(parts[-2]) == 2:  # State code is typically 2 characters
            metadata["state"] = parts[-2].upper()
    
    return metadata


def calculate_document_hash(content: str) -> str:
    """
    Calculate MD5 hash of document content.
    
    Args:
        content: Document content
        
    Returns:
        MD5 hash as a hex string
    """
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def load_documents(source_dir: str) -> List[Document]:
    """
    Load documents from a specified source directory with multiple file types.
    
    Args:
        source_dir: Directory containing documents
        
    Returns:
        List of Document objects with metadata
    """
    documents = []
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            try:
                # Extract metadata from path
                metadata = extract_metadata_from_path(file_path)
                
                # Load based on file type
                if file_ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                elif file_ext in [".docx", ".doc"]:
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                elif file_ext == ".csv":
                    loader = CSVLoader(file_path)
                    docs = loader.load()
                else:
                    # Try to load as a generic file
                    loader = UnstructuredFileLoader(file_path)
                    docs = loader.load()
                
                # Add metadata to each document
                for doc in docs:
                    # Calculate document hash and add to metadata
                    doc_hash = calculate_document_hash(doc.page_content)
                    doc.metadata.update(metadata)
                    doc.metadata["doc_hash"] = doc_hash
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    return documents


def load_and_split(file_paths: List[str], chunk_size: int = 256, chunk_overlap: int = 32) -> List[Document]:
    """
    Load and split documents from a list of file paths
    
    Args:
        file_paths: List of paths to documents
        chunk_size: Size of chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of chunked Document objects
    """
    documents = []
    for file_path in file_paths:
        try:
            file_ext = os.path.splitext(file_path)[1].lower()
            metadata = extract_metadata_from_path(file_path)
            
            if file_ext == ".pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
            elif file_ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
            elif file_ext == ".csv":
                loader = CSVLoader(file_path)
                docs = loader.load()
            else:
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
            
            for doc in docs:
                doc.metadata.update(metadata)
            
            documents.extend(docs)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    # Use the SentenceTransformersTokenTextSplitter for better chunking
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="all-MiniLM-L6-v2",
        tokens_per_chunk=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    return chunks


def split_documents(
    documents: List[Document],
    chunk_size: int = 256,
    chunk_overlap: int = 32
) -> List[Document]:
    """
    Split documents into smaller chunks using sentence-aware token splitting.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        
    Returns:
        List of chunked Document objects
    """
    # Use the SentenceTransformersTokenTextSplitter for better chunking
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="all-MiniLM-L6-v2",
        tokens_per_chunk=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    return chunks


def get_existing_document_hashes(db_type: str, persist_dir: str = None, collection_name: str = "documents") -> Set[str]:
    """
    Get hashes of documents already in the vector store.
    
    Args:
        db_type: Type of vector database ("chroma" or "qdrant")
        persist_dir: Directory where Chroma DB is stored
        collection_name: Collection name for Qdrant
        
    Returns:
        Set of document hashes already in the database
    """
    existing_hashes = set()
    
    try:
        if db_type == "chroma" and os.path.exists(persist_dir):
            client = chromadb.PersistentClient(path=persist_dir)
            if collection_name in client.list_collections():
                collection = client.get_collection(name=collection_name)
                # Get all metadata
                result = collection.get(include=["metadatas"])
                if result and "metadatas" in result:
                    for metadata in result["metadatas"]:
                        if metadata and "doc_hash" in metadata:
                            existing_hashes.add(metadata["doc_hash"])
                logger.info(f"Found {len(existing_hashes)} existing document hashes in Chroma")
                
        elif db_type == "qdrant":
            from qdrant_client import QdrantClient
            qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
            client = QdrantClient(url=qdrant_url)
            
            collections = client.get_collections().collections
            if any(c.name == collection_name for c in collections):
                # Get points with payload
                points = client.scroll(
                    collection_name=collection_name,
                    with_payload=True,
                    limit=10000  # Adjust based on expected document count
                )[0]
                
                for point in points:
                    if point.payload and "doc_hash" in point.payload:
                        existing_hashes.add(point.payload["doc_hash"])
                logger.info(f"Found {len(existing_hashes)} existing document hashes in Qdrant")
    except Exception as e:
        logger.error(f"Error getting existing document hashes: {e}")
    
    return existing_hashes


def main():
    """Main function to run the ingestion process."""
    parser = argparse.ArgumentParser(description="Ingest documents into vector database")
    parser.add_argument("--source-dir", "-s", 
                        default=settings.data_path,
                        help="Directory containing documents to ingest")
    parser.add_argument("--persist-dir", "-p", 
                        default=settings.chroma_path,
                        help="Directory to persist the vector store")
    parser.add_argument("--chunk-size", "-c", 
                        type=int,
                        default=256,
                        help="Chunk size for document splitting in tokens")
    parser.add_argument("--chunk-overlap", "-o", 
                        type=int,
                        default=32,
                        help="Chunk overlap for document splitting in tokens")
    parser.add_argument("--embedding-model", "-e", 
                        default="all-MiniLM-L6-v2",
                        help="Embedding model to use")
    parser.add_argument("--rebuild", "-r", 
                        action="store_true",
                        help="Rebuild the vector store")
    parser.add_argument("--db-type", "-d",
                        choices=["chroma", "qdrant"],
                        default=settings.vector_db,
                        help="Vector database type to use")
    parser.add_argument("--mode", "-m",
                       choices=["upsert", "recreate"],
                       default="recreate",
                       help="Ingest mode: 'upsert' to add new documents only, 'recreate' to rebuild")
    
    args = parser.parse_args()
    
    # Check if source directory exists
    if not os.path.isdir(args.source_dir):
        logger.error(f"Error: {args.source_dir} is not a valid directory")
        return
    
    # Set up persistence directory
    persist_dir = args.persist_dir
    collection_name = "documents" if args.db_type == "chroma" else "insurance_docs"
    
    # Check if we should rebuild the database (either rebuild flag or recreate mode)
    if (args.rebuild or args.mode == "recreate") and os.path.exists(persist_dir):
        logger.info(f"Rebuilding vector store: removing {persist_dir}")
        shutil.rmtree(persist_dir)
    
    # Load documents
    documents = load_documents(args.source_dir)
    if not documents:
        logger.warning("No documents found or loaded successfully")
        return
    
    # Get existing document hashes for upsert mode
    existing_hashes = set()
    if args.mode == "upsert":
        existing_hashes = get_existing_document_hashes(
            db_type=args.db_type, 
            persist_dir=persist_dir,
            collection_name=collection_name
        )
        logger.info(f"Found {len(existing_hashes)} documents already in the database")
    
    # Filter out documents that already exist in upsert mode
    if args.mode == "upsert" and existing_hashes:
        original_count = len(documents)
        documents = [
            doc for doc in documents 
            if "doc_hash" in doc.metadata and doc.metadata["doc_hash"] not in existing_hashes
        ]
        skipped_count = original_count - len(documents)
        logger.info(f"Skipping {skipped_count} documents that already exist in the database")
        
        if not documents:
            logger.info("No new documents to ingest")
            return
    
    # Split documents
    chunks = split_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Initialize embeddings model
    logger.info(f"Initializing embeddings model: {args.embedding_model}")
    device = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"
    embedding_model = HuggingFaceEmbeddings(
        model_name=args.embedding_model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    if args.db_type == "chroma" or settings.vector_db == "chroma":
        # Store in Chroma
        logger.info(f"Storing {len(chunks)} document chunks in Chroma at {persist_dir}")
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_dir
        )
        db.persist()
        logger.info(f"Successfully ingested {len(chunks)} document chunks into Chroma at {persist_dir}")
    elif args.db_type == "qdrant" or settings.vector_db == "qdrant":
        try:
            from langchain_community.vectorstores import Qdrant
            from qdrant_client import QdrantClient
            
            # Setup Qdrant client
            qdrant_url = f"http://{settings.qdrant_host}:{settings.qdrant_port}"
            collection_name = "insurance_docs"
            
            logger.info(f"Storing {len(chunks)} document chunks in Qdrant at {qdrant_url}")
            client = QdrantClient(url=qdrant_url)
            
            # Create collection if it doesn't exist
            collections = client.get_collections().collections
            collection_exists = any(c.name == collection_name for c in collections)
            
            # Store documents in Qdrant
            db = Qdrant.from_documents(
                documents=chunks,
                embedding=embedding_model,
                url=qdrant_url,
                collection_name=collection_name,
                force_recreate=args.rebuild or args.mode == "recreate"
            )
            logger.info(f"Successfully ingested {len(chunks)} document chunks into Qdrant collection '{collection_name}'")
        except ImportError:
            logger.error("Error: qdrant-client package not installed. Please install with 'pip install qdrant-client'")
            return
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            return


if __name__ == "__main__":
    main() 