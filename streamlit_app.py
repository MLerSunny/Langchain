#!/usr/bin/env python
"""
Streamlit application for the RAG + Fine-tuning System for DeepSeek models.
This provides a user interface for document ingestion, RAG querying, and fine-tuning.
"""

import os
import sys
import json
import tempfile
import shutil
import time
import platform
import psutil
import requests
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
import yaml
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add the project root to Python path to import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our project modules
from scripts.convert_to_sharegpt import (
    load_documents, 
    split_documents,
    generate_questions_from_content,
    enhance_answer,
    convert_to_sharegpt_format,
    check_llm_availability
)
from core.settings import settings, DATA_DIR, OLLAMA_BASE_URL, CHROMA_PERSIST_DIRECTORY, FASTAPI_PORT, MODEL_NAME
from document_manager import document_manager_tab
from metrics_tab import display_metrics_tab
from core.security import SecurityManager
from core.model_manager import ModelManager

# Load environment variables
load_dotenv()

# Use settings from core/settings.py (which loads rag.yaml)
from core.settings import settings, DATA_DIR, OLLAMA_BASE_URL, CHROMA_PERSIST_DIRECTORY, FASTAPI_PORT, MODEL_NAME

# Get RAG configuration from settings
RAG_CONFIG = settings.get('rag', {})
logger.info("Loaded RAG configuration from rag.yaml")

# Get configuration values
MAX_MEMORY_PERCENT = settings.get('optimization.max_memory_percent', 80)
DEFAULT_CHUNK_SIZE = settings.get('rag.chunk_size', 1024)
DEFAULT_CHUNK_OVERLAP = settings.get('rag.chunk_overlap', 128)
CACHE_TTL = settings.get('cache.ttl', 600)  # 10 minutes default

# Set page configuration
st.set_page_config(
    page_title="RAG System Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state variables
if 'model_name' not in st.session_state:
    st.session_state.model_name = MODEL_NAME  # Use MODEL_NAME from settings
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'sharegpt_data' not in st.session_state:
    st.session_state.sharegpt_data = []
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "idle"
if 'current_chunk_index' not in st.session_state:
    st.session_state.current_chunk_index = 0
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Initialize security manager
security_manager = SecurityManager()

# App styling
st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 20px;
    border-radius: 4px 4px 0px 0px;
}
.system-status {
    padding: 0.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.status-ok {
    background-color: rgba(0, 255, 0, 0.1);
    border: 1px solid rgba(0, 255, 0, 0.2);
}
.status-warning {
    background-color: rgba(255, 165, 0, 0.1);
    border: 1px solid rgba(255, 165, 0, 0.2);
}
.status-error {
    background-color: rgba(255, 0, 0, 0.1);
    border: 1px solid rgba(255, 0, 0, 0.2);
}
</style>
""", unsafe_allow_html=True)

# Load system prompt templates
SYSTEM_PROMPT_TEMPLATES = {
    "Insurance": "You are an insurance expert assistant. You provide accurate, helpful information about insurance policies, coverages, and claims. Answer questions based on your knowledge of insurance best practices and regulations.",
    "Finance": "You are a finance expert assistant. You provide accurate, helpful information about financial planning, investments, and money management. Answer questions based on your knowledge of financial best practices and regulations.",
    "Healthcare": "You are a healthcare expert assistant. You provide accurate, helpful information about medical conditions, treatments, and healthcare systems. Answer questions based on your knowledge of healthcare best practices.",
    "Legal": "You are a legal expert assistant. You provide accurate, helpful information about laws, regulations, and legal processes. Answer questions based on your knowledge of legal best practices.",
    "Technology": "You are a technology expert assistant. You provide accurate, helpful information about software, hardware, and digital services. Answer questions based on your knowledge of technology best practices.",
    "Education": "You are an education expert assistant. You provide accurate, helpful information about learning methods, educational systems, and teaching approaches. Answer questions based on your knowledge of educational best practices.",
    "General": "You are a helpful assistant that provides accurate, informative responses based on the content provided. Answer questions factually using only the information available in the provided content."
}

def display_system_status():
    st.markdown(
        '<div class="system-status status-ok">✅ System is running. All services appear healthy.</div>',
        unsafe_allow_html=True
    )

def convert_documents_tab():
    """Tab for converting documents to ShareGPT format."""
    st.header("Convert Documents to ShareGPT Format")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload documents to convert",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Save uploaded files
        temp_dir = save_uploaded_files(uploaded_files)
        
        # Document processing options
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input(
                "Chunk Size",
                min_value=256,
                max_value=4096,
                value=settings.chunk_size,  # Use centralized setting
                step=256
            )
        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap",
                min_value=0,
                max_value=512,
                value=settings.chunk_overlap,  # Use centralized setting
                step=64
            )
        
        # Processing options
        generate_questions = st.checkbox("Generate questions from content", value=True)
        enhance_answers = st.checkbox("Enhance answers with additional context", value=True)
        
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                try:
                    # Load and split documents
                    documents = load_documents(temp_dir)
                    st.session_state.chunks = split_documents(
                        documents,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                    
                    # Display chunks
                    display_chunks_table(st.session_state.chunks)
                    
                    # Generate ShareGPT format
                    if generate_questions:
                        st.session_state.sharegpt_data = []
                        for i, chunk in enumerate(st.session_state.chunks):
                            questions = generate_questions_from_content(chunk.page_content)
                            for question in questions:
                                answer = enhance_answer(chunk.page_content, question) if enhance_answers else chunk.page_content
                                sharegpt_item = convert_to_sharegpt_format(
                                    question,
                                    answer,
                                    system_prompt=SYSTEM_PROMPT_TEMPLATES["General"]
                                )
                                st.session_state.sharegpt_data.append(sharegpt_item)
                    
                    # Display ShareGPT preview
                    if st.session_state.sharegpt_data:
                        display_sharegpt_preview(st.session_state.sharegpt_data)
                        
                        # Save ShareGPT data
                        if st.button("Save ShareGPT Data"):
                            save_path = os.path.join(DATA_DIR, "training", "sharegpt_data.json")
                            with open(save_path, "w", encoding="utf-8") as f:
                                json.dump(st.session_state.sharegpt_data, f, indent=2)
                            st.success(f"ShareGPT data saved to {save_path}")
                    
                except Exception as e:
                    st.error(f"Error processing documents: {str(e)}")
                    logger.error(f"Error processing documents: {str(e)}", exc_info=True)
                finally:
                    # Cleanup
                    shutil.rmtree(temp_dir, ignore_errors=True)

def query_rag_tab():
    """Tab for querying the RAG system."""
    st.header("Query RAG System")
    
    # Query input
    query = st.text_area("Enter your question", height=100)
    
    if query:
        # Query options
        col1, col2 = st.columns(2)
        with col1:
            num_results = st.number_input("Number of results", min_value=1, max_value=10, value=3)
        with col2:
            similarity_threshold = st.slider(
                "Similarity threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05
            )
        
        if st.button("Submit Query"):
            with st.spinner("Processing query..."):
                try:
                    # Make request to RAG API
                    response = requests.post(
                        f"http://{settings.get('api.host')}:{settings.get('api.fastapi_port')}/query",
                        json={
                            "query": query,
                            "num_results": num_results,
                            "similarity_threshold": similarity_threshold
                        }
                    )
                    
                    if response.status_code == 200:
                        results = response.json()
                        
                        # Display results
                        st.subheader("Results")
                        for i, result in enumerate(results["results"], 1):
                            with st.expander(f"Result {i} (Score: {result['score']:.2f})"):
                                st.markdown(result["content"])
                                if "metadata" in result:
                                    st.json(result["metadata"])
                    else:
                        st.error(f"Error querying RAG system: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    logger.error(f"Error processing query: {str(e)}", exc_info=True)

def init_session_state():
    """Initialize session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def login():
    """Handle user login."""
    st.title("Login")
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # In a real app, you would validate against a database
        if username and password:
            # For demo purposes, accept any non-empty credentials
            st.session_state.authenticated = True
            st.session_state.user_id = username
            st.success("Login successful!")
            st.experimental_rerun()
        else:
            st.error("Please enter both username and password")

def chat_interface():
    """Main chat interface."""
    security_manager = st.session_state.security_manager
    model_manager = st.session_state.model_manager
    st.title("AI Chat Assistant")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    if prompt := st.chat_input("What would you like to know?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    is_valid, error = security_manager.validate_input(prompt)
                    if not is_valid:
                        st.error(error)
                        return
                    response = model_manager.generate_response(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
                except Exception as e:
                    logger.error(f"Error generating response: {str(e)}")
                    st.error("Sorry, I encountered an error. Please try again.")

def document_upload():
    """Handle document uploads."""
    security_manager = st.session_state.security_manager
    document_manager = st.session_state.get('document_manager')
    st.title("Document Upload")
    uploaded_file = st.file_uploader("Choose a file", type=settings.allowed_file_types)
    if uploaded_file is not None:
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        is_valid, error = security_manager.validate_file(file_path)
        if not is_valid:
            st.error(error)
            return
        try:
            with st.spinner("Processing document..."):
                if document_manager:
                    document_manager.process_document(file_path)
            st.success("Document processed successfully!")
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            st.error("Error processing document. Please try again.")

def query_rag_api(query: str, collection_name: str = "documents", n_results: int = 5):
    """Query the RAG API."""
    try:
        response = requests.post(
            f"http://{settings.get('api.host')}:{settings.get('api.fastapi_port')}/query",
            json={
                "query": query,
                "collection_name": collection_name,
                "n_results": n_results
            }
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Error querying RAG API: {str(e)}")
        return None

def get_training_jobs():
    """Get list of training jobs."""
    try:
        response = requests.get(
            f"http://{settings.get('api.host')}:{settings.get('api.fastapi_port') + 1}/jobs"
        )
        response.raise_for_status()
        return response.json()['jobs']
    except Exception as e:
        st.error(f"Error getting training jobs: {str(e)}")
        return {}

def fine_tuning_tab():
    st.header("Fine Tuning")
    st.info("Fine tuning functionality coming soon. (Implement this tab as needed)")

def main():
    init_session_state()
    if 'security_manager' not in st.session_state:
        st.session_state.security_manager = SecurityManager()
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()

    # Add all desired tabs to the sidebar
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "Convert to ShareGPT",
            "Document Management",
            "QueryRag",
            "Fine Tuning",
            "Metrics Tab"
        ]
    )

    if not st.session_state.authenticated:
        login()
    else:
        if page == "Convert to ShareGPT":
            convert_documents_tab()
        elif page == "Document Management":
            document_manager_tab()
        elif page == "QueryRag":
            query_rag_tab()
        elif page == "Fine Tuning":
            fine_tuning_tab()
        elif page == "Metrics Tab":
            display_metrics_tab()

        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_id = None
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main() 