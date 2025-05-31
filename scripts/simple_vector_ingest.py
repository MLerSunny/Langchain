# Renamed from ingest.py to simple_vector_ingest.py
# This script provides simple utilities for ingesting a single file into the vector database.
# Use this for quick, one-off ingestion tasks, not for bulk or production ingestion.

import os
import yaml
import torch
from typing import List, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, Docx2txtLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from core.settings import CHROMA_PERSIST_DIRECTORY

def get_embedding_model(config_path: str = "config/rag.yaml") -> HuggingFaceEmbeddings:
    """
    Initialize embedding model from configuration.
    
    Args:
        config_path: Path to the RAG configuration file
        
    Returns:
        Initialized HuggingFaceEmbeddings instance
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        embeddings_config = config['embeddings']
        
        model_name = embeddings_config['model']
        device = embeddings_config.get('device', 'auto')
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        model_kwargs = {
            'trust_remote_code': embeddings_config.get('trust_remote_code', True),
            'device': device
        }
        encode_kwargs = {
            'normalize_embeddings': embeddings_config.get('normalize_embeddings', True)
        }
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=embeddings_config.get('cache_dir')
        )
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        raise

def load_document(file_path: str):
    """Load a document based on its file extension."""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    loaders = {
        '.txt': TextLoader,
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.md': UnstructuredMarkdownLoader
    }
    
    if file_extension not in loaders:
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    loader = loaders[file_extension](file_path)
    return loader.load()

def split_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Split documents into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def create_vector_db_from_documents(
    documents: List,
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    config_path: str = "config/rag.yaml"
):
    """Create a vector database from documents."""
    embeddings = get_embedding_model(config_path)
    
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    vector_db.persist()
    return vector_db

def process_file(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, config_path: str = "config/rag.yaml"):
    """Process a single file and add it to the vector database."""
    try:
        # Load document
        documents = load_document(file_path)
        
        # Split documents
        split_docs = split_documents(documents, chunk_size, chunk_overlap)
        
        # Create/update vector database
        vector_db = create_vector_db_from_documents(split_docs, config_path=config_path)
        
        return True, f"Successfully processed {file_path}"
    except Exception as e:
        return False, f"Error processing {file_path}: {str(e)}" 