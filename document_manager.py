#!/usr/bin/env python
"""
Document management functionality for the RAG system.
"""

import os
import sys
import logging
import tempfile
import shutil
import time
import uuid
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
import pandas as pd
import hashlib
from langchain_core.documents import Document

# Add the project root to Python path to import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.simple_vector_ingest import create_vector_db_from_documents
from scripts.convert_to_sharegpt import load_documents, split_documents, get_file_extension
from scripts.batch_ingest import process_directory
from core.settings import settings, CHROMA_PERSIST_DIRECTORY, DATA_DIR

# Constants
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

__all__ = ['document_manager_tab']

def get_chroma_client(persist_directory: str = CHROMA_PERSIST_DIRECTORY):
    """
    Get a ChromaDB client for the vector database.
    
    Args:
        persist_directory: Directory where the vector database is stored
        
    Returns:
        ChromaDB client
    """
    from chromadb import PersistentClient
    from chromadb.config import Settings
    
    try:
        client = PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        return client
    except Exception as e:
        st.error(f"Error connecting to ChromaDB: {e}")
        return None

def list_documents(collection_name: str = "langchain", limit: int = 1000) -> List[Dict]:
    """
    List all documents in the vector database.
    
    Args:
        collection_name: Name of the collection to query
        limit: Maximum number of documents to return
        
    Returns:
        List of documents with their metadata
    """
    client = get_chroma_client()
    if not client:
        return []
    
    try:
        collection = client.get_collection(collection_name)
        if not collection:
            return []
        
        # Get all documents
        result = collection.get(include=["metadatas", "documents"], limit=limit)
        
        documents = []
        for i, doc_id in enumerate(result.get("ids", [])):
            metadata = result.get("metadatas", [])[i] if i < len(result.get("metadatas", [])) else {}
            content = result.get("documents", [])[i] if i < len(result.get("documents", [])) else ""
            
            # Create a document dict
            document = {
                "id": doc_id,
                "content": content,
                "source": metadata.get("source", "Unknown"),
                "topic": metadata.get("topic", "Unknown"),
                "lob": metadata.get("lob", "Unknown"),
                "state": metadata.get("state", "Unknown"),
                "added_at": metadata.get("added_at", "Unknown"),
                "content_preview": content[:100] + "..." if len(content) > 100 else content
            }
            documents.append(document)
        
        return documents
    except Exception as e:
        st.error(f"Error listing documents: {e}")
        return []

def delete_document(doc_id: str, collection_name: str = "langchain") -> bool:
    """
    Delete a document from the vector database.
    
    Args:
        doc_id: ID of the document to delete
        collection_name: Name of the collection to delete from
        
    Returns:
        True if successful, False otherwise
    """
    client = get_chroma_client()
    if not client:
        return False
    
    try:
        collection = client.get_collection(collection_name)
        if not collection:
            return False
        
        # Delete the document
        collection.delete(ids=[doc_id])
        return True
    except Exception as e:
        st.error(f"Error deleting document: {e}")
        return False

def add_documents_to_vectordb(
    documents: List[Document],
    persist_directory: str = CHROMA_PERSIST_DIRECTORY,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    max_chunks: Optional[int] = None
) -> int:
    """
    Add documents to the vector database.
    
    Args:
        documents: List of documents to add
        persist_directory: Directory to persist the vector store
        embedding_model_name: Name of the embedding model to use
        chunk_size: Size of document chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        max_chunks: Maximum number of chunks to process (None for all)
        
    Returns:
        Number of chunks added to vector store
    """
    try:
        # Split documents into chunks
        chunks = split_documents(documents, chunk_size, chunk_overlap, max_chunks)
        
        # Add chunks to vector store
        vectorstore = create_vector_db_from_documents(
            documents=chunks,
            persist_directory=persist_directory,
            embedding_model_name=embedding_model_name
        )
        
        return len(chunks)
    except Exception as e:
        st.error(f"Error adding documents to vector database: {e}")
        return 0

def display_document_count_by_source(documents: List[Dict]) -> None:
    """
    Display a bar chart of document counts by source.
    
    Args:
        documents: List of documents with metadata
    """
    # Count documents by source
    source_counts = {}
    for doc in documents:
        source = doc.get("source", "Unknown")
        if source in source_counts:
            source_counts[source] += 1
        else:
            source_counts[source] = 1
    
    # Create DataFrame
    df = pd.DataFrame(
        {"Source": list(source_counts.keys()), "Count": list(source_counts.values())}
    )
    
    # Sort by count
    df = df.sort_values("Count", ascending=False)
    
    # Display as bar chart
    st.bar_chart(df.set_index("Source"))

def search_documents(query: str, collection_name: str = "langchain", limit: int = 10) -> List[Dict]:
    """
    Search for documents in the vector database.
    
    Args:
        query: Query string
        collection_name: Name of the collection to search
        limit: Maximum number of results to return
        
    Returns:
        List of matching documents
    """
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    
    try:
        # Initialize embedding function
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Connect to vector store
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        
        # Search for documents
        results = vectorstore.similarity_search_with_score(query, k=limit)
        
        # Format results
        documents = []
        for doc, score in results:
            document = {
                "id": doc.metadata.get("id", "Unknown"),
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "topic": doc.metadata.get("topic", "Unknown"),
                "lob": doc.metadata.get("lob", "Unknown"),
                "state": doc.metadata.get("state", "Unknown"),
                "added_at": doc.metadata.get("added_at", "Unknown"),
                "content_preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "relevance_score": float(score)
            }
            documents.append(document)
        
        return documents
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return []

def document_manager_tab():
    """Tab for managing documents in the vector database."""
    st.markdown("## 📑 Document Management")
    st.markdown(
        "Manage documents in the vector database for RAG. View, add, and delete documents directly."
    )
    
    # Initialize session state for vector DB
    if "vector_db_documents" not in st.session_state:
        st.session_state.vector_db_documents = []
    
    if "last_refresh_time" not in st.session_state:
        st.session_state.last_refresh_time = time.time()
    
    # Create tabs for different operations
    tabs = st.tabs(["📋 View Documents", "➕ Add Documents", "🔍 Search Documents"])
    
    with tabs[0]:  # View Documents tab
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.subheader("Documents in Vector Database")
        
        with col3:
            if st.button("🔄 Refresh", key="refresh_docs"):
                st.session_state.vector_db_documents = list_documents()
                st.session_state.last_refresh_time = time.time()
        
        # Show last refresh time
        st.caption(f"Last refreshed: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state.last_refresh_time))}")
        
        # If no documents loaded yet, load them
        if not st.session_state.vector_db_documents:
            with st.spinner("Loading documents from vector database..."):
                st.session_state.vector_db_documents = list_documents()
        
        # Display documents count
        col1, col2 = st.columns(2)
        col1.metric("Total Documents", len(st.session_state.vector_db_documents))
        
        # Count unique sources
        unique_sources = set(doc.get("source", "Unknown") for doc in st.session_state.vector_db_documents)
        col2.metric("Unique Sources", len(unique_sources))
        
        # Display chart of document counts by source
        st.subheader("Document Distribution")
        display_document_count_by_source(st.session_state.vector_db_documents)
        
        # Filters
        st.subheader("Filter Documents")
        col1, col2 = st.columns(2)
        
        with col1:
            # Get unique sources
            sources = ["All"] + sorted(list(unique_sources))
            selected_source = st.selectbox("Filter by Source", sources)
        
        with col2:
            # Get unique topics
            unique_topics = set(doc.get("topic", "Unknown") for doc in st.session_state.vector_db_documents)
            topics = ["All"] + sorted(list(unique_topics))
            selected_topic = st.selectbox("Filter by Topic", topics)
        
        # Apply filters
        filtered_docs = st.session_state.vector_db_documents
        if selected_source != "All":
            filtered_docs = [doc for doc in filtered_docs if doc.get("source") == selected_source]
        
        if selected_topic != "All":
            filtered_docs = [doc for doc in filtered_docs if doc.get("topic") == selected_topic]
        
        # Create table of documents
        st.subheader("Document List")
        
        if not filtered_docs:
            st.info("No documents match the selected filters.")
        else:
            # Create a DataFrame for display
            df_data = []
            for i, doc in enumerate(filtered_docs):
                df_data.append({
                    "ID": i + 1,
                    "Source": doc.get("source", "Unknown"),
                    "Topic": doc.get("topic", "Unknown"),
                    "Preview": doc.get("content_preview", ""),
                    "Doc ID": doc.get("id", "")
                })
            
            df = pd.DataFrame(df_data)
            
            # Display as DataFrame
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Document deletion
            st.subheader("Delete Document")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                doc_index = st.number_input(
                    "Select document by ID",
                    min_value=1,
                    max_value=len(filtered_docs),
                    value=1,
                    step=1
                )
            
            with col2:
                delete_button = st.button("🗑️ Delete", key="delete_doc")
            
            if delete_button:
                # Get document ID
                doc_id = filtered_docs[doc_index - 1].get("id")
                
                # Confirm deletion
                confirm = st.checkbox(f"Confirm deletion of document {doc_index}?")
                
                if confirm:
                    with st.spinner("Deleting document..."):
                        success = delete_document(doc_id)
                        
                        if success:
                            st.success(f"Document {doc_index} deleted successfully!")
                            
                            # Refresh documents
                            st.session_state.vector_db_documents = list_documents()
                            st.rerun()
                        else:
                            st.error(f"Failed to delete document {doc_index}.")
    
    with tabs[1]:  # Add Documents tab
        st.subheader("Add Documents to Vector Database")
        
        # Document source selection
        source_option = st.radio(
            "Document Source", 
            ["Upload Files", "Use Existing Path"],
            index=0,
            help="Choose where to get the documents from"
        )
        
        # Document processing parameters
        st.subheader("Processing Parameters")
        
        col1, col2, col3 = st.columns(3)
        chunk_size = col1.slider(
            "Chunk Size (tokens)", 
            128, 2048, DEFAULT_CHUNK_SIZE,
            help="Size of document chunks in tokens. Smaller chunks may improve retrieval but lose context."
        )
        
        chunk_overlap = col2.slider(
            "Chunk Overlap (tokens)", 
            0, 512, DEFAULT_CHUNK_OVERLAP,
            help="Overlap between chunks in tokens. Higher overlap helps maintain context between chunks."
        )
        
        max_chunks = col3.slider(
            "Max Chunks to Process", 
            10, 1000, 100,
            help="Maximum number of chunks to process. Lower this if you experience memory issues."
        )
        
        # Add metadata fields
        st.subheader("Document Metadata")
        
        col1, col2 = st.columns(2)
        with col1:
            topic = st.text_input(
                "Topic/Category", 
                help="Topic or category for the documents (e.g., 'Insurance', 'Finance')"
            )
        
        with col2:
            lob = st.selectbox(
                "Line of Business", 
                ["general", "auto", "home", "life", "health", "finance", "other"],
                index=0,
                help="Line of business these documents belong to"
            )
        
        # Conditionally show upload or path input based on selection
        if source_option == "Upload Files":
            st.subheader("Upload Documents")
            uploaded_files = st.file_uploader(
                "Upload Documents", 
                accept_multiple_files=True,
                type=["pdf", "docx", "doc", "csv", "txt", "html", "json"],
                help="Upload documents in PDF, DOCX, CSV, TXT, HTML, JSON format"
            )
            
            if uploaded_files:
                upload_button = st.button("📤 Upload to Vector DB", type="primary")
                
                if upload_button:
                    with st.spinner("Processing documents..."):
                        # Save uploaded files to a temporary directory
                        temp_dir = tempfile.mkdtemp()
                        
                        try:
                            # Save each uploaded file
                            for uploaded_file in uploaded_files:
                                file_path = os.path.join(temp_dir, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                            
                            # Load documents
                            documents = load_documents(temp_dir)
                            
                            # Add metadata
                            for doc in documents:
                                doc.metadata["topic"] = topic if topic else doc.metadata.get("topic", "Unknown")
                                doc.metadata["lob"] = lob
                                doc.metadata["added_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Generate an ID if none exists
                                if "id" not in doc.metadata:
                                    # Create a stable ID based on content + source
                                    content_hash = hashlib.md5((doc.page_content + str(doc.metadata.get("source", ""))).encode()).hexdigest()
                                    doc.metadata["id"] = f"doc-{content_hash}"
                            
                            # Process and add to vector DB
                            chunks_added = add_documents_to_vectordb(
                                documents=documents,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                max_chunks=max_chunks
                            )
                            
                            st.success(f"Successfully processed {len(documents)} documents, adding {chunks_added} chunks to the vector database.")
                            
                            # Refresh document list
                            st.session_state.vector_db_documents = list_documents()
                        except Exception as e:
                            st.error(f"Error processing documents: {e}")
                        finally:
                            # Clean up temporary directory
                            shutil.rmtree(temp_dir)
        else:
            st.subheader("Use Existing Path")
            existing_path = st.text_input(
                "Document Path", 
                os.path.join(DATA_DIR, "raw"),
                help="Path to directory containing documents"
            )
            
            # Check if path exists
            if existing_path and os.path.exists(existing_path):
                path_button = st.button("📂 Process Directory", type="primary")
                
                if path_button:
                    with st.spinner("Processing documents..."):
                        try:
                            # Load documents
                            documents = load_documents(existing_path)
                            
                            # Add metadata
                            for doc in documents:
                                doc.metadata["topic"] = topic if topic else doc.metadata.get("topic", "Unknown")
                                doc.metadata["lob"] = lob
                                doc.metadata["added_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                                
                                # Generate an ID if none exists
                                if "id" not in doc.metadata:
                                    # Create a stable ID based on content + source
                                    content_hash = hashlib.md5((doc.page_content + str(doc.metadata.get("source", ""))).encode()).hexdigest()
                                    doc.metadata["id"] = f"doc-{content_hash}"
                            
                            # Process and add to vector DB
                            chunks_added = add_documents_to_vectordb(
                                documents=documents,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                                max_chunks=max_chunks
                            )
                            
                            st.success(f"Successfully processed {len(documents)} documents, adding {chunks_added} chunks to the vector database.")
                            
                            # Refresh document list
                            st.session_state.vector_db_documents = list_documents()
                        except Exception as e:
                            st.error(f"Error processing documents: {e}")
            else:
                st.warning("Path does not exist. Please enter a valid directory path.")
    
    with tabs[2]:  # Search Documents tab
        st.subheader("Search Documents")
        
        search_query = st.text_input(
            "Search Query",
            placeholder="Enter your search query here...",
            help="Enter a query to search for documents in the vector database"
        )
        
        max_results = st.slider(
            "Max Results",
            min_value=1,
            max_value=100,
            value=10,
            help="Maximum number of search results to return"
        )
        
        search_button = st.button("🔍 Search", type="primary", key="search_button")
        
        if search_button and search_query:
            with st.spinner("Searching documents..."):
                results = search_documents(search_query, limit=max_results)
                
                if not results:
                    st.info("No documents found matching your query.")
                else:
                    st.success(f"Found {len(results)} matching documents.")
                    
                    # Display results
                    for i, doc in enumerate(results):
                        with st.expander(f"Result {i+1}: {doc.get('source', 'Unknown')}"):
                            st.markdown(f"**Source:** {doc.get('source', 'Unknown')}")
                            st.markdown(f"**Topic:** {doc.get('topic', 'Unknown')}")
                            st.markdown(f"**Relevance Score:** {doc.get('relevance_score', 0):.4f}")
                            st.markdown("**Content:**")
                            st.markdown(doc.get("content", ""))
                            
                            # Show metadata
                            with st.expander("Metadata"):
                                st.json({
                                    "id": doc.get("id", "Unknown"),
                                    "source": doc.get("source", "Unknown"),
                                    "topic": doc.get("topic", "Unknown"),
                                    "lob": doc.get("lob", "Unknown"),
                                    "state": doc.get("state", "Unknown"),
                                    "added_at": doc.get("added_at", "Unknown")
                                })
        
        # Add documentation
        with st.expander("📚 Search Tips"):
            st.markdown("""
            ### Search Tips
            
            * Use natural language queries for best results
            * Include specific keywords related to your topic
            * Be specific with technical terms
            * Try different phrasings if you don't get good results
            * The search uses semantic similarity, not keyword matching
            
            ### How Vector Search Works
            
            1. Your query is converted to a vector embedding
            2. The system finds documents with similar embeddings
            3. Results are ranked by similarity score
            4. This allows finding conceptually similar content, not just keyword matches
            """) 