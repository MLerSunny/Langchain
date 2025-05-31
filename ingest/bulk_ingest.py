"""
Bulk Ingest module for loading and splitting documents for insurance RAG system.
This module loads mixed document types from data/raw/, 
processes them with custom chunking, and stores them in Chroma.
Use this for production and large-scale ingestion.
"""

print("RUNNING BULK INGEST FROM:", __file__)

import os
import re
import sys
import argparse
import shutil
import logging
import hashlib
import json
import datetime
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
from tqdm import tqdm
import time
import traceback

import chromadb
from langchain_community.document_loaders import (
    PyPDFLoader, 
    Docx2txtLoader, 
    CSVLoader
)
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document
import yaml
import torch
from chromadb.config import Settings

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.settings import CHROMA_PERSIST_DIRECTORY, settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/bulk_ingest_run.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        self.total_files = 0
        self.processed_files = 0
        self.total_chunks = 0
        self.processed_chunks = 0
        self.current_file = ""
        self.current_operation = ""
        self.operation_start_time = time.time()
        self.operation_times = {
            "loading": [],
            "Loading": [],
            "splitting": [],
            "embedding": [],
            "storing": []
        }
        self.failed_files = []
        self.failed_chunks = []
        
    def update_file_progress(self, current_file: str, operation: str):
        # Calculate time for previous operation
        if self.current_operation:
            elapsed = time.time() - self.operation_start_time
            self.operation_times[self.current_operation].append(elapsed)
        
        self.current_file = current_file
        self.current_operation = operation
        self.processed_files += 1
        self.operation_start_time = time.time()
        
        # Calculate overall progress
        elapsed = time.time() - self.start_time
        files_per_second = self.processed_files / elapsed if elapsed > 0 else 0
        progress = (self.processed_files / self.total_files * 100) if self.total_files > 0 else 0
        
        # Calculate estimated time remaining
        if files_per_second > 0:
            remaining_files = self.total_files - self.processed_files
            eta_seconds = remaining_files / files_per_second
            eta_str = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"
        else:
            eta_str = "ETA: Calculating..."
        
        print(f"\nProcessing: {self.current_file} ({self.processed_files}/{self.total_files})")
        print(f"Operation: {operation}")
        print(f"Progress: {progress:.1f}%")
        print(f"Speed: {files_per_second:.2f} files/sec")
        print(f"Time elapsed: {datetime.timedelta(seconds=int(elapsed))}")
        print(eta_str)
        
    def update_chunk_progress(self, total_chunks: int):
        self.total_chunks = total_chunks
        self.processed_chunks += 1
        
        # Calculate chunk processing metrics
        elapsed = time.time() - self.operation_start_time
        chunks_per_second = self.processed_chunks / elapsed if elapsed > 0 else 0
        progress = (self.processed_chunks / self.total_chunks * 100) if self.total_chunks > 0 else 0
        
        # Calculate estimated time remaining for chunks
        if chunks_per_second > 0:
            remaining_chunks = self.total_chunks - self.processed_chunks
            eta_seconds = remaining_chunks / chunks_per_second
            eta_str = f"ETA: {datetime.timedelta(seconds=int(eta_seconds))}"
        else:
            eta_str = "ETA: Calculating..."
        
        print(f"\nChunk Processing:")
        print(f"Progress: {progress:.1f}% ({self.processed_chunks}/{self.total_chunks} chunks)")
        print(f"Speed: {chunks_per_second:.2f} chunks/sec")
        print(f"Time elapsed: {datetime.timedelta(seconds=int(elapsed))}")
        print(eta_str)
        
    def record_failure(self, item: str, error: str, item_type: str = "file"):
        """Record failed items for retry."""
        if item_type == "file":
            self.failed_files.append({"path": item, "error": error})
        else:
            self.failed_chunks.append({"chunk": item, "error": error})
    
    def print_summary(self):
        elapsed = time.time() - self.start_time
        
        print("\n\n=== Ingestion Summary ===")
        print(f"Total time: {datetime.timedelta(seconds=int(elapsed))}")
        print(f"Files processed: {self.processed_files}/{self.total_files}")
        print(f"Total chunks generated: {self.total_chunks}")
        
        # Print operation timing statistics
        print("\nOperation Timing Statistics:")
        for operation, times in self.operation_times.items():
            if times:
                avg_time = sum(times) / len(times)
                print(f"{operation.capitalize()}:")
                print(f"  Average time: {avg_time:.2f} seconds")
                print(f"  Total time: {sum(times):.2f} seconds")
                print(f"  Operations: {len(times)}")
        
        # Print failure statistics
        if self.failed_files or self.failed_chunks:
            print("\nFailure Statistics:")
            if self.failed_files:
                print(f"Failed files: {len(self.failed_files)}")
                for failure in self.failed_files:
                    print(f"  - {failure['path']}: {failure['error']}")
            if self.failed_chunks:
                print(f"Failed chunks: {len(self.failed_chunks)}")
                for failure in self.failed_chunks:
                    print(f"  - Chunk {failure['chunk']}: {failure['error']}")
        
        print(f"\nAverage processing time per file: {elapsed/self.processed_files:.2f} seconds")
        print(f"Average chunks per file: {self.total_chunks/self.processed_files:.2f}")

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


def validate_sharegpt_json(file_path: str) -> bool:
    """
    Validate if a JSON file matches the ShareGPT structure.
    Expected structure:
    [
      {
        "conversations": [
          {"from": "human", "value": "..."},
          {"from": "gpt" or "assistant", "value": "..."}
        ]
      }
    ]
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            return False
        for item in data:
            if not isinstance(item, dict) or "conversations" not in item:
                return False
            conversations = item["conversations"]
            if not isinstance(conversations, list):
                return False
            for conv in conversations:
                if not isinstance(conv, dict) or "from" not in conv or "value" not in conv:
                    return False
                if conv["from"] not in ["human", "gpt", "assistant"]:  # Accept both "gpt" and "assistant"
                    return False
        return True
    except Exception as e:
        logger.error(f"Error validating ShareGPT JSON {file_path}: {str(e)}")
        return False


def load_documents(source_paths: List[str], progress_tracker: ProgressTracker) -> List[Document]:
    """
    Load documents from a list of file paths.
    Enhanced: Stricter ShareGPT validation, metadata enrichment, logging, summary.
    """
    documents = []
    processed_files = 0
    skipped_files = 0
    errored_files = 0
    total_files = len(source_paths)
    progress_tracker.total_files = total_files
    processed_count = 0
    
    print(f"\nStarting document ingestion. Total files to process: {total_files}")
    
    for file_path in source_paths:
        try:
            processed_count += 1
            file = os.path.basename(file_path)
            print(f"\nProcessing file {processed_count}/{total_files}: {file}")
            progress_tracker.update_file_progress(file, "Loading")
            
            file_ext = os.path.splitext(file)[1].strip().lower()
            print(f"[DEBUG] file_ext for {file}: '{file_ext}'")
            if file_ext == ".jsonl":
                logger.info(f"Skipping .jsonl file (not supported for ShareGPT ingestion): {file_path}")
                skipped_files += 1
                continue
            
            print(f"Loading file: {file_path}")
            metadata = extract_metadata_from_path(file_path)
            metadata["original_filename"] = file
            metadata["ingestion_timestamp"] = datetime.datetime.now().isoformat()
            
            # Load based on file type
            print(f"Using loader for file type: {file_ext}")
            try:
                if file_ext == ".json":
                    if validate_sharegpt_json(file_path):
                        logger.info(f"Valid ShareGPT JSON file: {file_path}")
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        docs = []
                        for item in data:
                            for conv in item["conversations"]:
                                doc = Document(page_content=conv["value"], metadata=metadata)
                                docs.append(doc)
                    else:
                        logger.warning(f"Invalid ShareGPT JSON file: {file_path}, skipping.")
                        skipped_files += 1
                        continue
                elif file_ext == ".pdf":
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                elif file_ext in [".docx", ".doc"]:
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                elif file_ext == ".csv":
                    loader = CSVLoader(file_path)
                    docs = loader.load()
                elif file_ext == ".txt":
                    # For text files, try to detect encoding and load directly
                    try:
                        print(f"Attempting to load text file directly: {file_path}")
                        import chardet
                        with open(file_path, 'rb') as f:
                            raw_data = f.read()
                            result = chardet.detect(raw_data)
                            encoding = result['encoding'] if result['encoding'] else 'utf-8'
                            print(f"Detected encoding: {encoding}")
                        
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                            if not content.strip():
                                raise ValueError("File is empty or contains only whitespace")
                            docs = [Document(page_content=content, metadata=metadata)]
                            print(f"Successfully loaded text file: {file_path}")
                    except Exception as e:
                        error_msg = f"Error in direct text loading: {str(e)}"
                        print(error_msg)
                        logger.error(error_msg)
                        
                        # Fallback to UnstructuredLoader
                        try:
                            print(f"Attempting to load with UnstructuredLoader: {file_path}")
                            loader = UnstructuredLoader(file_path, encoding=encoding)
                            docs = loader.load()
                            print(f"Successfully loaded text file using UnstructuredLoader: {file_path}")
                        except Exception as e2:
                            error_msg = f"Error using UnstructuredLoader: {str(e2)}"
                            print(error_msg)
                            logger.error(error_msg)
                            traceback.print_exc()
                            raise Exception(f"Failed to load text file. Direct loading error: {str(e)}. UnstructuredLoader error: {str(e2)}")
                else:
                    # For other file types, try UnstructuredLoader
                    try:
                        loader = UnstructuredLoader(file_path)
                        docs = loader.load()
                    except Exception as e:
                        logger.error(f"Error loading {file_path} with UnstructuredLoader: {str(e)}")
                        traceback.print_exc()
                        raise
                
                if not docs:
                    raise ValueError("No documents were loaded from the file")
                
                print(f"Loaded {len(docs)} documents from {file_path}")
                
                for doc in docs:
                    if not doc.page_content.strip():
                        print(f"Warning: Empty document content in {file_path}")
                        continue
                    doc_hash = calculate_document_hash(doc.page_content)
                    doc.metadata.update(metadata)
                    doc.metadata["doc_hash"] = doc_hash
                documents.extend(docs)
                logger.info(f"Successfully processed {file_path}")
                processed_files += 1
                
            except Exception as e:
                error_msg = f"Error loading {file_path}: {str(e)}"
                logger.error(error_msg)
                print(error_msg)
                traceback.print_exc()
                errored_files += 1
                continue
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            logger.error(error_msg)
            print(error_msg)
            traceback.print_exc()
            errored_files += 1
            continue
    
    print("\n\nIngestion Summary:")
    print(f"Total files processed: {processed_files}")
    print(f"Files skipped: {skipped_files}")
    print(f"Files with errors: {errored_files}")
    print(f"Total documents loaded: {len(documents)}")
    
    return documents


def split_documents(documents: List[Document], config: dict, progress_tracker: ProgressTracker) -> List[Document]:
    """Split documents into chunks using configuration from rag.yaml."""
    print("\nSplitting documents into chunks...")
    chunking_config = config['chunking']
    embeddings_config = config['embeddings']
    
    print(f"Initializing text splitter with chunk size: {chunking_config['chunk_size']}")
    text_splitter = SentenceTransformersTokenTextSplitter(
        model_name=embeddings_config['model'],
        chunk_size=chunking_config['chunk_size'],
        chunk_overlap=chunking_config['chunk_overlap']
    )
    
    chunks = []
    failed_docs = []
    print(f"Processing {len(documents)} documents...")
    
    for i, doc in enumerate(tqdm(documents, desc="Splitting documents")):
        try:
            doc_start_time = time.time()
            print(f"\nSplitting document {i+1}/{len(documents)}")
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
            
            # Update progress and timing
            doc_time = time.time() - doc_start_time
            progress_tracker.operation_times["splitting"].append(doc_time)
            progress_tracker.update_chunk_progress(len(chunks))
            
            print(f"Created {len(doc_chunks)} chunks from document {i+1}")
            print(f"Processing time: {doc_time:.2f} seconds")
            print(f"Average time per chunk: {doc_time/len(doc_chunks):.2f} seconds")
            
        except Exception as e:
            error_msg = f"Error splitting document {i+1}: {str(e)}"
            print(error_msg)
            progress_tracker.record_failure(f"doc_{i+1}", error_msg, "file")
            failed_docs.append(doc)
            continue
    
    # Retry failed documents with different chunking parameters
    if failed_docs:
        print(f"\nRetrying {len(failed_docs)} failed documents with adjusted parameters...")
        retry_config = config.copy()
        retry_config['chunking']['chunk_size'] = max(100, chunking_config['chunk_size'] // 2)
        retry_config['chunking']['chunk_overlap'] = max(10, chunking_config['chunk_overlap'] // 2)
        
        retry_chunks = split_documents(failed_docs, retry_config, progress_tracker)
        chunks.extend(retry_chunks)
    
    print(f"\nSplit {len(documents)} documents into {len(chunks)} chunks")
    return chunks


def store_documents(vector_store: Chroma, chunks: List[Document], batch_size: int = 100, progress_tracker: ProgressTracker = None) -> None:
    """Store documents in vector store with progress tracking and error recovery."""
    print(f"\nStoring {len(chunks)} chunks in vector database...")
    total_chunks = len(chunks)
    failed_chunks = []
    max_retries = 3  # Maximum number of retries for failed chunks
    retry_count = 0
    
    # Process in smaller batches to show progress
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        batch_start_time = time.time()
        print(f"\nProcessing batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size}")
        print(f"Storing chunks {i+1} to {min(i+batch_size, total_chunks)} of {total_chunks}")
        
        try:
            # Store batch
            vector_store.add_documents(batch)
            
            # Calculate and display progress
            progress = min(i + batch_size, total_chunks) / total_chunks * 100
            batch_time = time.time() - batch_start_time
            chunks_per_second = len(batch) / batch_time if batch_time > 0 else 0
            
            print(f"Progress: {progress:.1f}% complete")
            print(f"Batch processing time: {batch_time:.2f} seconds")
            print(f"Processing speed: {chunks_per_second:.2f} chunks/second")
            
            # Display memory usage
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            print(f"Memory usage: {memory_info.rss / 1024 / 1024:.1f} MB")
            
            # Update progress tracker if provided
            if progress_tracker:
                progress_tracker.operation_times["storing"].append(batch_time)
            
        except Exception as e:
            error_msg = f"Error storing batch: {str(e)}"
            print(error_msg)
            if progress_tracker:
                progress_tracker.record_failure(f"batch_{i//batch_size + 1}", error_msg, "chunk")
            failed_chunks.extend(batch)
            continue
    
    # Retry failed chunks with smaller batch size
    if failed_chunks and retry_count < max_retries:
        retry_count += 1
        print(f"\nRetrying {len(failed_chunks)} failed chunks (attempt {retry_count}/{max_retries})...")
        retry_batch_size = max(1, batch_size // 2)
        store_documents(vector_store, failed_chunks, retry_batch_size, progress_tracker)
    elif failed_chunks:
        print(f"\nFailed to store {len(failed_chunks)} chunks after {max_retries} retries")
    
    print("\nStorage complete!")


def load_and_split(file_paths: List[str], config: dict) -> List[Document]:
    """Load documents from file paths and split them into chunks."""
    documents = []
    for file_path in file_paths:
        loader = UnstructuredLoader(file_path)
        documents.extend(loader.load())
    
    return split_documents(documents, config)


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


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize embedding model from configuration.
    
    Returns:
        Initialized HuggingFaceEmbeddings instance
    """
    try:
        model_name = settings.get('embeddings.model')
        device = settings.get('embeddings.device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        model_kwargs = {
            'trust_remote_code': settings.get('embeddings.trust_remote_code', True),
            'device': device
        }
        encode_kwargs = {
            'normalize_embeddings': settings.get('embeddings.normalize_embeddings', True)
        }
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=settings.get('embeddings.cache_dir')
        )
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        raise


def print_vectordb_health(persist_dir, embedding_model):
    from langchain_community.vectorstores import Chroma
    try:
        db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
        collections = db._client.list_collections()
        print("[VectorDB Health] Collections:", [c.name for c in collections])
        for c in collections:
            collection = db._client.get_collection(c.name)
            print(f"[VectorDB Health] Collection '{c.name}' has {collection.count()} vectors.")
        results = db.similarity_search("What is AI?", k=1)
        if results:
            print("[VectorDB Health] Sample result metadata:", results[0].metadata)
            print("[VectorDB Health] Sample result content:", results[0].page_content[:100])
        else:
            print("[VectorDB Health] No results found for test query.")
        import os
        db_files = os.listdir(persist_dir)
        print("[VectorDB Health] Files in DB directory:", db_files)
    except Exception as e:
        print("[VectorDB Health] Error during health check:", e)


def load_config(config_path: str = "config/rag.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_vector_store(config: dict, rebuild: bool = False):
    """Create or load vector store using configuration from rag.yaml."""
    print("\nInitializing vector store...")
    vector_store_config = config['vector_store']
    embeddings_config = config['embeddings']
    
    # Initialize embeddings
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embeddings_config['model'],
        model_kwargs={"device": "cuda" if os.environ.get("USE_GPU", "false").lower() == "true" else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Configure HNSW index parameters
    hnsw_config = {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": vector_store_config['index_params']['hnsw_ef_construction'],
        "hnsw:search_ef": vector_store_config['index_params']['hnsw_ef_construction'],
        "hnsw:M": vector_store_config['index_params']['hnsw_m']
    }
    
    print("Creating ChromaDB client...")
    # Create persistent ChromaDB client
    client = chromadb.PersistentClient(
        path=vector_store_config['persist_directory'],
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=rebuild
        )
    )
    
    print("Creating/loading collection...")
    # Create or get collection with HNSW index
    collection = client.get_or_create_collection(
        name=vector_store_config['collection_name'],
        metadata={"hnsw:space": "cosine"}
    )
    
    print("Initializing Chroma vector store...")
    # Initialize Chroma with the configured client and index
    vector_store = Chroma(
        client=client,
        collection_name=vector_store_config['collection_name'],
        embedding_function=embeddings,
        persist_directory=vector_store_config['persist_directory']
    )
    
    return vector_store


def get_chroma_metrics(vector_store: Chroma) -> dict:
    """Get metrics about the current state of ChromaDB."""
    try:
        collection = vector_store._collection
        count = collection.count()
        # Get all metadata to analyze document sources
        result = collection.get(include=["metadatas"])
        sources = {}
        if result and "metadatas" in result:
            for metadata in result["metadatas"]:
                if metadata and "source" in metadata:
                    source = metadata["source"]
                    sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_documents": count,
            "unique_sources": len(sources),
            "documents_per_source": sources
        }
    except Exception as e:
        logger.error(f"Error getting ChromaDB metrics: {str(e)}")
        return {
            "total_documents": 0,
            "unique_sources": 0,
            "documents_per_source": {}
        }

def main():
    """Main function to process documents and store them in the vector database."""
    parser = argparse.ArgumentParser(description="Process documents and store them in the vector database")
    parser.add_argument("--source-dir", "-s", type=str, required=True, help="Directory containing source documents")
    parser.add_argument("--config", type=str, default="config/rag.yaml", help="Path to configuration file")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector store")
    parser.add_argument("--mode", type=str, choices=["append", "rebuild"], default="append", help="Ingestion mode")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of files to process in each batch")
    args = parser.parse_args()
    
    print("\nStarting bulk ingestion process...")
    print(f"Source directory: {args.source_dir}")
    print(f"Configuration file: {args.config}")
    print(f"Mode: {args.mode}")
    print(f"Batch size: {args.batch_size}")
    
    # Initialize progress tracker
    progress_tracker = ProgressTracker()
    
    try:
        # Load configuration
        print("Loading configuration...")
        config = load_config(args.config)
        
        # Process source directory
        source_path = Path(args.source_dir)
        if source_path.is_file():
            file_paths = [str(source_path)]
        else:
            file_paths = [str(p) for p in source_path.glob("**/*") if p.is_file()]
        
        print(f"\nFound {len(file_paths)} files to process")
        progress_tracker.total_files = len(file_paths)
        
        # Create vector store
        print("Creating vector store...")
        vector_store = create_vector_store(config, rebuild=args.rebuild)
        
        # Get initial metrics
        print("\n=== Initial ChromaDB Metrics ===")
        initial_metrics = get_chroma_metrics(vector_store)
        print(f"Total documents: {initial_metrics['total_documents']}")
        print(f"Unique sources: {initial_metrics['unique_sources']}")
        print("\nDocuments per source:")
        for source, count in initial_metrics['documents_per_source'].items():
            print(f"  - {source}: {count}")
        
        # Process documents in batches
        total_processed = 0
        total_documents = 0
        
        for i in range(0, len(file_paths), args.batch_size):
            batch_files = file_paths[i:i + args.batch_size]
            print(f"\nProcessing batch {i//args.batch_size + 1} of {(len(file_paths) + args.batch_size - 1)//args.batch_size}")
            
            try:
                # Process documents
                print("Loading and processing documents...")
                documents = load_documents(batch_files, progress_tracker)
                
                if documents:
                    # Split documents
                    chunks = split_documents(documents, config, progress_tracker)
                    
                    # Store documents with progress tracking
                    store_documents(vector_store, chunks)
                    
                    total_processed += len(batch_files)
                    total_documents += len(chunks)
                    
                    print(f"\nBatch completed. Total files processed: {total_processed}/{len(file_paths)}")
                    print(f"Total chunks stored: {total_documents}")
                else:
                    print("No documents to store in this batch")
                    
            except Exception as e:
                print(f"Error processing batch: {str(e)}")
                print("Continuing with next batch...")
                continue
        
        # Get final metrics
        print("\n=== Final ChromaDB Metrics ===")
        final_metrics = get_chroma_metrics(vector_store)
        print(f"Total documents: {final_metrics['total_documents']}")
        print(f"Unique sources: {final_metrics['unique_sources']}")
        print("\nDocuments per source:")
        for source, count in final_metrics['documents_per_source'].items():
            print(f"  - {source}: {count}")
        
        # Print metrics comparison
        print("\n=== Metrics Comparison ===")
        print(f"Initial document count: {initial_metrics['total_documents']}")
        print(f"Final document count: {final_metrics['total_documents']}")
        print(f"New documents added: {final_metrics['total_documents'] - initial_metrics['total_documents']}")
        print(f"New sources added: {final_metrics['unique_sources'] - initial_metrics['unique_sources']}")
        
        # Print final summary
        progress_tracker.print_summary()
        
        # Health check
        print("\nPerforming health check...")
        test_query = "test query"
        results = vector_store.similarity_search(test_query, k=1)
        if results:
            print("Vector store health check passed")
        else:
            print("Vector store health check failed")
        
        print("\nBulk ingestion completed!")
        print(f"Total files processed: {total_processed}")
        print(f"Total chunks stored: {total_documents}")
        
    except Exception as e:
        print(f"Fatal error in main process: {str(e)}")
        raise


if __name__ == "__main__":
    main() 