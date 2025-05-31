#!/usr/bin/env python
"""
Start the RAG server with proper environment setup.
"""

import os
import sys
import logging
from pathlib import Path
import uvicorn
import time
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.settings import (
    settings, CHROMA_PERSIST_DIRECTORY, FASTAPI_PORT, HOST,
    MODEL_NAME, EMBEDDING_MODEL, ALLOWED_ORIGINS, LOG_LEVEL,
    DATA_DIR, MODELS_DIR, CACHE_DIR, LOGS_DIR
)

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, "rag_server.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        DATA_DIR,
        CHROMA_PERSIST_DIRECTORY,
        MODELS_DIR,
        CACHE_DIR,
        LOGS_DIR
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

# Initialize FastAPI app
app = FastAPI(
    title=settings.get("app.name", "Insurance Assistant"),
    description="Retrieval-Augmented Generation API",
    version=settings.get("app.version", "1.0.0")
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB client
client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIRECTORY,
    settings=Settings(anonymized_telemetry=False)
)

# Initialize embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.get("/collections")
async def list_collections():
    """List all collections in the vector database."""
    try:
        collections = client.list_collections()
        return {"collections": [col.name for col in collections]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_documents(query: str, collection_name: str = None, n_results: int = None):
    """Query documents from the vector database."""
    try:
        collection_name = collection_name or settings.get("vector_store.collection_name", "documents")
        n_results = n_results or settings.get("retrieval.max_results", 5)
        
        collection = client.get_collection(collection_name)
        query_embedding = embedding_model.encode(query).tolist()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        return {
            "query": query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def check_prerequisites():
    """Check if all prerequisites are met."""
    try:
        # Ensure all required directories exist
        ensure_directories()
        
        # Check if .env file exists
        env_file = Path(".env")
        if not env_file.exists():
            logger.warning(".env file not found. Creating default .env file...")
            with open(env_file, "w") as f:
                f.write(f"""CHROMA_PERSIST_DIRECTORY={CHROMA_PERSIST_DIRECTORY}
DATA_DIR={DATA_DIR}
FASTAPI_PORT={FASTAPI_PORT}
STREAMLIT_PORT=8501
HOST={HOST}
""")
        
        return True
    except Exception as e:
        logger.error(f"Error checking prerequisites: {str(e)}", exc_info=True)
        return False

def check_server_health(max_retries=3, retry_delay=2):
    """Check if the server is healthy."""
    for attempt in range(max_retries):
        try:
            response = requests.get(f"http://{HOST}:{FASTAPI_PORT}/health")
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
    return False

def start_server():
    """Start the FastAPI server."""
    uvicorn.run(
        app,
        host=HOST,
        port=FASTAPI_PORT,
        log_level=LOG_LEVEL.lower()
    )

def main():
    """Main function to start the RAG server."""
    try:
        # Check prerequisites
        if not check_prerequisites():
            logger.error("Prerequisites check failed")
            sys.exit(1)
        
        # Start the server
        logger.info(f"Starting RAG server on {HOST}:{FASTAPI_PORT}")
        start_server()
    except Exception as e:
        logger.error(f"Error starting RAG server: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 