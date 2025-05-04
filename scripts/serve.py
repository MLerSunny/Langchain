#!/usr/bin/env python3
"""
Script to start the RAG API service.
This makes it easy to start the FastAPI server for the RAG system.
"""

import os
import sys
import logging
import subprocess
import signal
import time
from pathlib import Path

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.settings import settings, FASTAPI_PORT

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Flag to track if we should continue running
keep_running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully exit."""
    global keep_running
    logger.info("Stopping RAG API service...")
    keep_running = False

def start_rag_service():
    """Start the RAG API service."""
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    logger.info(f"Starting RAG API service on port {FASTAPI_PORT}...")
    
    # Command to run the FastAPI server - changed to explicitly reference serve/main.py
    cmd = [
        "uvicorn", 
        "serve.main:app", 
        "--host", "0.0.0.0",
        "--port", str(FASTAPI_PORT),
        "--reload"
    ]
    
    try:
        # Start the server process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        logger.info(f"RAG API service started with PID {process.pid}")
        logger.info(f"API will be available at http://localhost:{FASTAPI_PORT}")
        
        # Monitor the process and log output
        while keep_running and process.poll() is None:
            output = process.stdout.readline()
            if output:
                print(output.strip())
            time.sleep(0.1)
        
        # Terminate the process if still running
        if process.poll() is None:
            logger.info("Terminating RAG API service...")
            process.terminate()
            process.wait(timeout=5)
        
        return_code = process.poll()
        logger.info(f"RAG API service exited with code {return_code}")
        
    except Exception as e:
        logger.error(f"Error starting RAG API service: {e}")
        return 1
    
    return 0

def main():
    """Main entry point."""
    # Check if the vector database exists
    chroma_dir = settings.chroma_persist_directory
    if not os.path.exists(chroma_dir):
        logger.error(f"Vector database not found at {chroma_dir}")
        logger.error("Please ingest documents first using the 'Convert Documents' tab or the ingest.py script")
        return 1
    
    # Start the RAG service
    return start_rag_service()

if __name__ == "__main__":
    sys.exit(main()) 