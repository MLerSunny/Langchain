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
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.colored_header import colored_header
from streamlit_extras.app_logo import add_logo

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
from core.settings import settings, DATA_DIR, OLLAMA_BASE_URL

# Constants for memory management
MAX_MEMORY_PERCENT = 80  # Maximum memory usage percentage
DEFAULT_CHUNK_SIZE = 1024
DEFAULT_CHUNK_OVERLAP = 128

# Cache time in seconds (10 minutes)
CACHE_TTL = 600

# Set page configuration
st.set_page_config(
    page_title="DeepSeek RAG + Fine-tuning System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for model name
if 'model_name' not in st.session_state:
    # Default to deepseek-llm:7b
    st.session_state.model_name = "deepseek-llm:7b"

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

@st.cache_data(ttl=CACHE_TTL)
def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    return os.path.splitext(filename)[1].lower()

@st.cache_data(ttl=CACHE_TTL)
def get_supported_file_types() -> List[str]:
    """Get a list of supported file types."""
    return [".pdf", ".docx", ".doc", ".csv", ".txt", ".html", ".htm", ".json"]

@st.cache_data(ttl=CACHE_TTL)
def check_system_status() -> Dict[str, Any]:
    """
    Check the status of various system components.
    
    Returns:
        Dict with status information about various components
    """
    status = {
        "memory": {
            "available": True,
            "message": "Memory usage is within acceptable limits",
            "percent": 0,
            "level": "ok"
        },
        "disk": {
            "available": True,
            "message": "Disk space is sufficient",
            "percent": 0,
            "level": "ok"
        },
        "ollama": {
            "available": check_llm_availability(),
            "message": "Ollama service is " + ("available" if check_llm_availability() else "unavailable"),
            "level": "ok" if check_llm_availability() else "error"
        },
        "data_dir": {
            "available": os.path.exists(DATA_DIR),
            "message": f"Data directory exists at {DATA_DIR}" if os.path.exists(DATA_DIR) else f"Data directory not found at {DATA_DIR}",
            "level": "ok" if os.path.exists(DATA_DIR) else "error"
        },
        "overall": {
            "available": True,
            "message": "All systems operational",
            "level": "ok"
        }
    }
    
    # Check memory usage
    memory = psutil.virtual_memory()
    status["memory"]["percent"] = memory.percent
    
    if memory.percent > 90:
        status["memory"]["available"] = False
        status["memory"]["message"] = f"Memory usage is critically high ({memory.percent}%)"
        status["memory"]["level"] = "error"
    elif memory.percent > 75:
        status["memory"]["message"] = f"Memory usage is high ({memory.percent}%)"
        status["memory"]["level"] = "warning"
    
    # Check disk space
    disk = psutil.disk_usage('/')
    status["disk"]["percent"] = disk.percent
    
    if disk.percent > 90:
        status["disk"]["available"] = False
        status["disk"]["message"] = f"Disk space is critically low ({disk.percent}%)"
        status["disk"]["level"] = "error"
    elif disk.percent > 75:
        status["disk"]["message"] = f"Disk space is running low ({disk.percent}%)"
        status["disk"]["level"] = "warning"
    
    # Determine overall status
    if not status["ollama"]["available"] or not status["memory"]["available"] or not status["disk"]["available"]:
        status["overall"]["available"] = False
        status["overall"]["message"] = "Some system components are unavailable"
        status["overall"]["level"] = "error"
    elif status["memory"]["level"] == "warning" or status["disk"]["level"] == "warning":
        status["overall"]["message"] = "System is operational but some resources are limited"
        status["overall"]["level"] = "warning"
    
    return status

def save_uploaded_files(uploaded_files) -> str:
    """Save uploaded files to a temporary directory and return the path."""
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Save each uploaded file
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    
    return temp_dir

def display_documents_table(documents):
    """Display a table of documents with their metadata."""
    if not documents:
        st.warning("No documents loaded")
        return
    
    # Create a DataFrame from document metadata
    docs_data = []
    
    # Group documents by source file to show a more consolidated view
    sources = {}
    for doc in documents:
        source = doc.metadata.get("source", "Unknown")
        if source not in sources:
            sources[source] = {
                "count": 1,
                "total_chars": len(doc.page_content),
                "preview": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                "topic": doc.metadata.get("topic", "Unknown")
            }
        else:
            sources[source]["count"] += 1
            sources[source]["total_chars"] += len(doc.page_content)
    
    # Create a summary dataframe 
    for i, (source, info) in enumerate(sources.items()):
        # Extract file type from source
        file_type = os.path.splitext(source)[1].lower() if "." in source else ""
        
        docs_data.append({
            "ID": i,
            "Source": source,
            "File Type": file_type,
            "Topic": info["topic"],
            "Pages/Sections": info["count"],
            "Total Length (chars)": info["total_chars"],
            "Content Preview": info["preview"]
        })
    
    # Display as a DataFrame
    df = pd.DataFrame(docs_data)
    st.dataframe(df, use_container_width=True)
    
    # Display some statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Documents", len(sources))
    col1.metric("Total Pages/Sections", len(documents))
    
    avg_length = sum(len(doc.page_content) for doc in documents) / len(documents)
    col2.metric("Average Length", f"{int(avg_length)} chars")
    
    total_chars = sum(len(doc.page_content) for doc in documents)
    col3.metric("Total Content Size", f"{total_chars:,} chars")
    
    # Show distribution of document lengths by file type
    if len(docs_data) > 1:
        fig = px.bar(
            df, 
            x="Source", 
            y="Total Length (chars)",
            color="File Type",
            title="Document Size by Source",
            labels={"Total Length (chars)": "Character Count", "Source": "Document Source"}
        )
        st.plotly_chart(fig, use_container_width=True)

def display_chunks_table(chunks):
    """Display a table of document chunks."""
    if not chunks:
        st.warning("No chunks created")
        return
    
    # Create a DataFrame from chunks
    chunks_data = []
    for i, chunk in enumerate(chunks):
        # Get the first 100 characters of content
        content_preview = chunk.page_content[:100] + "..." if len(chunk.page_content) > 100 else chunk.page_content
        
        # Extract file type from source
        source = chunk.metadata.get("source", "Unknown")
        file_type = os.path.splitext(source)[1].lower() if "." in source else ""
        
        # Add to data
        chunks_data.append({
            "ID": i,
            "Source": source,
            "File Type": file_type,
            "Topic": chunk.metadata.get("topic", "Unknown"),
            "Length (chars)": len(chunk.page_content),
            "Content Preview": content_preview
        })
    
    # Display as a DataFrame with filtering
    df = pd.DataFrame(chunks_data)
    
    # Add filters for different document types
    col1, col2 = st.columns(2)
    with col1:
        file_types = ["All"] + sorted(df["File Type"].unique().tolist())
        selected_type = st.selectbox("Filter by File Type", file_types)
    
    with col2:
        sources = ["All"] + sorted(df["Source"].unique().tolist())
        selected_source = st.selectbox("Filter by Source", sources)
    
    # Apply filters
    filtered_df = df.copy()
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["File Type"] == selected_type]
    if selected_source != "All":
        filtered_df = filtered_df[filtered_df["Source"] == selected_source]
    
    # Display filtered dataframe
    st.dataframe(filtered_df, use_container_width=True)
    
    # Display some statistics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", len(chunks))
    col1.metric("Showing", len(filtered_df))
    
    avg_length = sum(len(chunk.page_content) for chunk in chunks) / len(chunks)
    col2.metric("Average Chunk Length", f"{int(avg_length)} chars")
    
    total_chars = sum(len(chunk.page_content) for chunk in chunks)
    col3.metric("Total Content Size", f"{total_chars:,} chars")
    
    # Count chunks per source
    source_counts = df.groupby(["Source", "File Type"]).size().reset_index(name="Chunk Count")
    
    # Show distribution of chunks by source and file type
    fig = px.bar(
        source_counts, 
        x="Source", 
        y="Chunk Count",
        color="File Type",
        title="Chunks Distribution by Source",
        labels={"Chunk Count": "Number of Chunks", "Source": "Document Source"}
    )
    st.plotly_chart(fig, use_container_width=True)

def display_sharegpt_preview(sharegpt_data, max_display=5):
    """Display a preview of the ShareGPT format data."""
    if not sharegpt_data:
        st.warning("No ShareGPT data created")
        return
    
    # Show number of generated conversations
    st.metric("Generated Conversations", len(sharegpt_data))
    
    # Display a limited number of examples
    for i, conversation in enumerate(sharegpt_data[:max_display]):
        with st.expander(f"Conversation {i+1}"):
            for msg in conversation["conversations"]:
                role = msg["from"]
                content = msg["value"]
                
                if role == "system":
                    st.markdown(f"**System:**\n\n{content}")
                elif role == "human":
                    st.markdown(f"**Human:**\n\n{content}")
                elif role in ["assistant", "gpt"]:
                    st.markdown(f"**Assistant:**\n\n{content}")
    
    if len(sharegpt_data) > max_display:
        st.info(f"Showing {max_display} of {len(sharegpt_data)} conversations")

def display_system_status():
    """Display system status information in the sidebar."""
    status = check_system_status()
    
    with st.sidebar:
        st.markdown("### System Status")
        
        # Overall status
        overall_level = status["overall"]["level"]
        status_class = f"system-status status-{overall_level}"
        
        st.markdown(f"""
        <div class="{status_class}">
            <strong>Overall Status:</strong> {status["overall"]["message"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Component statuses
        components = {
            "Ollama LLM": status["ollama"],
            "Memory": status["memory"],
            "Disk Space": status["disk"],
            "Data Directory": status["data_dir"]
        }
        
        for name, component in components.items():
            level = component["level"]
            emoji = "✅" if level == "ok" else "⚠️" if level == "warning" else "❌"
            
            # For memory and disk, show percentage
            extra_info = f" ({component.get('percent')}%)" if "percent" in component else ""
            
            st.markdown(f"{emoji} **{name}**{extra_info}: {component['message']}")
        
        # Show system info
        st.markdown("### System Information")
        st.markdown(f"**OS:** {platform.system()} {platform.release()}")
        st.markdown(f"**Python:** {platform.python_version()}")
        
        memory = psutil.virtual_memory()
        st.markdown(f"**Total Memory:** {memory.total / (1024 ** 3):.1f} GB")
        
        # Add refresh button
        if st.button("Refresh Status"):
            # Clear the cache for check_system_status
            check_system_status.clear()
            st.rerun()

@st.cache_data(ttl=CACHE_TTL)
def get_available_models() -> List[str]:
    """
    Get a list of available models from Ollama.
    
    Returns:
        List of available model names
    """
    default_models = ["deepseek-llm:7b", "deepseek-r1:32b", "deepseek-coder:instruct", "deepseek-coder:latest"]
    
    try:
        # Try to connect to Ollama and get available models
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        
        if response.status_code != 200:
            return default_models
        
        # Extract model names
        models = response.json().get("models", [])
        model_names = [model.get("name") for model in models]
        
        if not model_names:
            return default_models
            
        # Put deepseek-llm:7b first if it exists in the list
        if "deepseek-llm:7b" in model_names:
            model_names.remove("deepseek-llm:7b")
            return ["deepseek-llm:7b"] + model_names
        
        # Put deepseek-r1:32b second if it exists
        if "deepseek-r1:32b" in model_names:
            model_names.remove("deepseek-r1:32b")
            model_names = ["deepseek-r1:32b"] + model_names
            
        return model_names
        
    except requests.exceptions.RequestException:
        # Return default models if can't connect to Ollama
        return default_models

def convert_documents_tab():
    """Tab for converting documents to ShareGPT format."""
    st.markdown("## 📄 Convert Documents to ShareGPT Format")
    st.markdown(
        "Upload documents in various formats and convert them to ShareGPT format for fine-tuning. "
        "The system will process the documents, generate questions, and create conversation pairs."
    )
    
    # Get system status for memory-aware parameter suggestions
    status = check_system_status()
    memory_usage = status["memory"]["percent"]
    
    # Adjust default parameters based on memory usage
    suggested_batch_size = 20 if memory_usage < 70 else 10 if memory_usage < 85 else 5
    suggested_chunk_size = DEFAULT_CHUNK_SIZE if memory_usage < 70 else int(DEFAULT_CHUNK_SIZE * 0.75)
    
    with st.form("convert_form"):
        # File upload or path input selection
        st.markdown(f"**Step 1:** Choose document source")
        doc_source = st.radio(
            "Select document source",
            ["Upload documents", "Use existing documents path"],
            help="Either upload new documents or specify a path to existing documents"
        )
        
        if doc_source == "Upload documents":
            # File upload (original functionality)
            supported_types = get_supported_file_types()
            st.markdown(f"**Upload your documents** (Supported formats: {', '.join(supported_types)})")
            uploaded_files = st.file_uploader(
                "Upload Documents", 
                accept_multiple_files=True, 
                type=[ext[1:] for ext in supported_types]  # Remove the dot
            )
            existing_path = None
        else:
            # Path selection for existing documents
            st.markdown("**Specify path to existing documents**")
            existing_path = st.text_input(
                "Documents Path",
                value=os.path.join(DATA_DIR, "raw"),
                help="Path to directory containing existing documents (read-only access)"
            )
            st.info("⚠️ The system will only read from this path without modifying the original documents.")
            uploaded_files = None
        
        # Chunking parameters
        st.markdown("**Step 2:** Configure document processing")
        col1, col2, col3 = st.columns(3)
        chunk_size = col1.slider(
            "Chunk Size (tokens)", 
            256, 2048, suggested_chunk_size,
            help="Size of document chunks in tokens. Smaller chunks provide more focused but less context-aware results."
        )
        chunk_overlap = col2.slider(
            "Chunk Overlap (tokens)", 
            0, 512, DEFAULT_CHUNK_OVERLAP,
            help="Overlap between chunks in tokens. Higher overlap helps maintain context between chunks."
        )
        max_chunks = col3.slider(
            "Max Chunks to Process", 
            10, 100, suggested_batch_size,
            help="Maximum number of chunks to process. Lower this if you experience memory issues."
        )
        
        # ShareGPT conversion parameters
        st.markdown("**Step 3:** Configure conversion to ShareGPT format")
        
        # Domain selection and system prompt
        col1, col2 = st.columns(2)
        with col1:
            domain = st.selectbox(
                "Select Domain", 
                list(SYSTEM_PROMPT_TEMPLATES.keys()),
                index=0,
                help="Select the domain most relevant to your documents"
            )
        
        system_prompt = st.text_area(
            "System Prompt", 
            SYSTEM_PROMPT_TEMPLATES[domain],
            height=100,
            help="System prompt that sets the context for the AI assistant"
        )
        
        col1, col2, col3 = st.columns(3)
        questions_per_chunk = col1.slider(
            "Questions Per Chunk", 
            1, 5, 2,
            help="Number of questions to generate per document chunk"
        )
        enhance_answers = col2.checkbox(
            "Enhance Answers with LLM", 
            True,
            help="Use LLM to create more structured and focused answers"
        )
        max_workers = col3.slider(
            "Parallel Workers",
            1, 8, 4,
            help="Number of parallel workers for processing. Higher values are faster but use more memory."
        )
        
        # Output options
        st.markdown("**Step 4:** Configure output")
        output_filename = st.text_input(
            "Output Filename", 
            "generated_conversations.json",
            help="Filename for the generated ShareGPT format data"
        )
        
        # Submit button
        submit_button = st.form_submit_button("Process Documents")
    
    # LLM availability warning
    if not check_llm_availability():
        st.warning(
            "⚠️ Ollama LLM service is not available. The system will use rule-based question generation, "
            "which may produce less specific questions. Make sure Ollama is running for better results."
        )
    
    # Process when the form is submitted
    if submit_button and (uploaded_files or existing_path):
        # Set up a progress tracking system
        progress_container = st.empty()
        status_container = st.empty()
        result_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
        
        with status_container:
            if uploaded_files:
                status = st.info("Saving uploaded files...")
            else:
                status = st.info("Reading from existing documents path...")
        
        try:
            # Handle document source based on selection
            if uploaded_files:
                # Save uploaded files (original functionality)
                temp_dir = save_uploaded_files(uploaded_files)
                source_dir = temp_dir
            else:
                # Use existing path directly
                if not os.path.exists(existing_path):
                    st.error(f"Path {existing_path} does not exist!")
                    return
                source_dir = existing_path
            
            # Update progress
            progress_bar.progress(10)
            status.info("Loading documents...")
            
            # Load documents
            documents = load_documents(source_dir)
            
            # Update progress
            progress_bar.progress(30)
            
            with result_container:
                st.subheader("Loaded Documents")
                display_documents_table(documents)
            
            status.info("Splitting documents into chunks...")
            
            # Split documents
            chunks = split_documents(documents, chunk_size, chunk_overlap, max_chunks)
            
            # Update progress
            progress_bar.progress(50)
            
            with result_container:
                st.subheader("Document Chunks")
                display_chunks_table(chunks)
            
            status.info("Converting to ShareGPT format...")
            
            # Convert to ShareGPT format
            sharegpt_data = convert_to_sharegpt_format(
                chunks,
                system_prompt=system_prompt,
                questions_per_chunk=questions_per_chunk,
                enhance_answers=enhance_answers,
                max_workers=max_workers,
                batch_size=max_chunks
            )
            
            # Update progress
            progress_bar.progress(90)
            
            # Save the output
            output_dir = os.path.join(DATA_DIR, "training")
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
            
            # Update progress
            progress_bar.progress(100)
            status.success(f"Processing complete! Saved to {output_path}")
            
            with result_container:
                st.subheader("ShareGPT Format Preview")
                display_sharegpt_preview(sharegpt_data)
                
                # Download button for the generated file
                with open(output_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                    
                st.download_button(
                    label="Download ShareGPT Data",
                    data=file_content,
                    file_name=output_filename,
                    mime="application/json"
                )
                
                # Information on next steps
                st.info("""
                **Next Steps**:
                1. Review the generated ShareGPT data
                2. Use it for fine-tuning by going to the "Fine-tune Model" tab
                3. Or download and modify it manually before fine-tuning
                """)
            
        except Exception as e:
            status.error(f"Error processing documents: {e}")
        finally:
            # Clean up temporary directory only if we created one
            if 'temp_dir' in locals() and uploaded_files:
                shutil.rmtree(temp_dir)
    
    elif submit_button:
        st.warning("Please either upload documents or specify a valid path to existing documents.")
    
    # Add documentation section
    with st.expander("📚 Documentation & Tips"):
        st.markdown("""
        ### Document Processing
        
        **Two ways to provide documents:**
        1. **Upload Documents**: Upload new documents directly through the interface
        2. **Use Existing Path**: Specify a path to documents that already exist on your drive
        
        **Important**: When using an existing documents path, the system:
        - Uses read-only access to the original documents
        - Never modifies or deletes the original files
        - Creates processed versions and vector embeddings in its own data directory
        - Maintains the original files unchanged for reference
        
        This approach lets you use your existing document repositories for RAG and fine-tuning without duplicating or altering them.
        
        ### How Document Conversion Works
        
        1. **Document Loading**: Files are loaded using LangChain's document loaders
        2. **Chunking**: Documents are split into manageable chunks based on your settings
        3. **Question Generation**: Questions are generated for each chunk using AI or rule-based methods
        4. **Answer Enhancement**: Answers are enhanced to be more structured and focused
        5. **ShareGPT Format**: Everything is converted to ShareGPT format for fine-tuning
        
        ### Tips for Best Results
        
        - **Chunk Size**: Smaller chunks (256-512) work better for focused information, larger chunks (1024-2048) for maintaining context
        - **Questions Per Chunk**: 2-3 questions per chunk usually provides good coverage
        - **System Prompt**: Customize the system prompt to match your domain for more relevant responses
        - **Memory Usage**: If you experience out-of-memory errors, reduce max chunks and parallel workers
        
        ### Troubleshooting
        
        - **Question Generation Failed**: Ensure Ollama is running and the model is available
        - **Processing Slow**: Reduce the number of chunks or parallel workers
        - **Poor Quality Questions**: Try a different domain or customize the system prompt
        """)

def query_rag_tab():
    """Tab for querying the RAG system."""
    st.markdown("## 🔍 Query RAG System")
    
    if not check_llm_availability():
        st.warning(
            "⚠️ Ollama LLM service is not available. Please start Ollama to use the RAG system."
        )
        
        # Show instructions for starting Ollama
        with st.expander("How to Start Ollama"):
            st.markdown("""
            ### Starting Ollama
            
            1. **Open a terminal/command prompt**
            2. **Run the following command**:
            ```
            ollama serve
            ```
            3. **In another terminal, pull the DeepSeek model**:
            ```
            ollama pull deepseek-r1:32b
            ```
            
            You can also use the provided Makefile command:
            ```
            make ollama
            ```
            
            After starting Ollama, refresh this page.
            """)
        return
    
    st.markdown("Coming soon! This tab will allow querying the RAG system.")

def fine_tune_tab():
    """Tab for fine-tuning models."""
    st.markdown("## 🧠 Fine-tune Model")
    
    if not check_llm_availability():
        st.warning(
            "⚠️ Ollama LLM service is not available. Please start Ollama to use the fine-tuning system."
        )
        return
    
    st.markdown("Coming soon! This tab will provide an interface for fine-tuning models.")

def app_settings_tab():
    """Tab for app settings."""
    st.markdown("## ⚙️ Settings")
    
    # App settings section
    st.subheader("App Settings")
    
    # Theme settings
    theme = st.selectbox(
        "Theme",
        ["Light", "Dark"],
        index=0,
        help="Select the app theme"
    )
    
    # Model settings
    st.subheader("Model Settings")
    
    # Get available models from Ollama
    available_models = get_available_models()
    
    # Model selection using a selectbox
    model = st.selectbox(
        "Model Name",
        available_models,
        index=available_models.index(st.session_state.model_name) if st.session_state.model_name in available_models else 0,
        help="Select a DeepSeek model to use"
    )
    
    # Ollama URL
    ollama_url = st.text_input(
        "Ollama URL",
        OLLAMA_BASE_URL,
        help="URL of the Ollama service"
    )
    
    # System settings
    st.subheader("System Settings")
    
    # Data directory
    data_dir = st.text_input(
        "Data Directory",
        DATA_DIR,
        help="Directory for storing data"
    )
    
    # Apply button
    if st.button("Apply Settings"):
        # Update session state with new values
        st.session_state.model_name = model
        st.success("Settings updated successfully!")
        # Force refresh to apply changes
        st.rerun()
    
    # System information
    st.subheader("System Information")
    
    # Show OS info
    os_info = f"{platform.system()} {platform.release()}"
    st.markdown(f"**Operating System**: {os_info}")
    
    # Show Python version
    st.markdown(f"**Python Version**: {platform.python_version()}")
    
    # Show memory usage
    memory = psutil.virtual_memory()
    st.markdown(f"**Memory Usage**: {memory.percent}% of {memory.total / (1024 ** 3):.1f} GB")
    
    # Show disk usage
    disk = psutil.disk_usage('/')
    st.markdown(f"**Disk Usage**: {disk.percent}% of {disk.total / (1024 ** 3):.1f} GB")

def main():
    """Main function to run the Streamlit app."""
    # Display system status in sidebar
    display_system_status()
    
    # App header
    st.title("DeepSeek RAG + Fine-tuning System")
    st.markdown("A complete system for building RAG applications and fine-tuning DeepSeek models")
    
    # Create tabs
    tabs = st.tabs([
        "📄 Convert Documents", 
        "🔍 Query RAG", 
        "🧠 Fine-tune Model", 
        "⚙️ Settings"
    ])
    
    # Tab content
    with tabs[0]:
        convert_documents_tab()
    
    with tabs[1]:
        query_rag_tab()
    
    with tabs[2]:
        fine_tune_tab()
    
    with tabs[3]:
        app_settings_tab()
    
    # Footer
    st.markdown("---")
    st.markdown("DeepSeek RAG + Fine-tuning System | v1.0.0")

if __name__ == "__main__":
    main() 