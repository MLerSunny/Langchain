"""
Document Manager for handling document loading, processing, and vector database operations.
Consolidates functionality from various document management scripts.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

from core.settings import settings

logger = logging.getLogger(__name__)

# Export DocumentManager
__all__ = ['DocumentManager']

class DocumentManager:
    """Manages document loading, processing, and vector database operations."""
    
    def __init__(self):
        """Initialize the document manager."""
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=settings.chroma_persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get embedding configuration from settings
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
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=settings.get('embeddings.cache_dir')
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
    
    def init_vector_db(self, collection_name: str = None) -> None:
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the collection to create
        """
        try:
            collection_name = collection_name or settings.qdrant_collection_name
            self.chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the vector database.
        
        Returns:
            List of collection names
        """
        try:
            return [col.name for col in self.chroma_client.list_collections()]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """
        Add documents to the vector database.
        
        Args:
            documents: List of documents to add
            collection_name: Name of the collection
            metadata: Additional metadata for the documents
            
        Returns:
            True if successful
        """
        try:
            collection_name = collection_name or settings.qdrant_collection_name
            
            # Split documents into chunks
            texts = []
            metadatas = []
            for doc in documents:
                chunks = self.text_splitter.split_text(doc["text"])
                texts.extend(chunks)
                metadatas.extend([{
                    **doc.get("metadata", {}),
                    **(metadata or {}),
                    "chunk": i
                } for i in range(len(chunks))])
            
            # Create or get collection
            try:
                collection = self.chroma_client.get_collection(collection_name)
            except:
                self.init_vector_db(collection_name)
                collection = self.chroma_client.get_collection(collection_name)
            
            # Add documents
            collection.add(
                documents=texts,
                metadatas=metadatas,
                ids=[f"{i}" for i in range(len(texts))]
            )
            
            logger.info(f"Added {len(texts)} chunks to collection {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    def search_documents(
        self,
        query: str,
        collection_name: str = None,
        n_results: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents in the vector database.
        
        Args:
            query: Search query
            collection_name: Name of the collection
            n_results: Number of results to return
            
        Returns:
            List of matching documents
        """
        try:
            collection_name = collection_name or settings.qdrant_collection_name
            n_results = n_results or settings.top_k
            
            collection = self.chroma_client.get_collection(collection_name)
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            return [{
                "text": doc,
                "metadata": meta,
                "distance": dist
            } for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )]
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection from the vector database.
        
        Args:
            collection_name: Name of the collection
            
        Returns:
            True if successful
        """
        try:
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False 