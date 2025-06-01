#!/usr/bin/env python
"""
CLI entry point for RAG + Fine-tuning system for DeepSeek models.
This script provides a unified CLI for all components.
"""

import os
os.environ["PYTHONMALLOC"] = "malloc"
os.environ["CYGWIN"] = "heap_chunk_in_mb=2048,tp_num_c_bufs=1024"
os.environ["LOG_LEVEL"] = "DEBUG"

import sys
import argparse
import subprocess
import logging
from pathlib import Path
from fastapi import HTTPException, FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import json
import uuid

# Import the config to make it available
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.settings import (
    settings,
    DATA_DIR,
    CHROMA_PERSIST_DIRECTORY,
    FASTAPI_PORT,
    HOST
)

from core.rag import RAGEngine
from core.metrics import MetricsCollector
from core.fine_tuning import FineTuningManager
from scripts.start_fine_tuning_server import FineTuningServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add debug logging for settings
logger.info(f"Loading settings from: {settings.get('app.config_path', 'config/rag.yaml')}")
logger.info(f"Generation settings: {settings.get('generation')}")

app = FastAPI(
    title="Fine-tuning API",
    description="API for managing model fine-tuning jobs",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
rag_engine = RAGEngine()
metrics_collector = MetricsCollector()
fine_tuning_manager = FineTuningManager()
fine_tuning_server = FineTuningServer()

class QueryRequest(BaseModel):
    query: str
    context: Optional[str] = None
    max_tokens: Optional[int] = settings.get("llm", {}).get("max_tokens", 2048)
    temperature: Optional[float] = settings.get("llm", {}).get("temperature", 0.7)

class FineTuningRequest(BaseModel):
    model_name: str
    training_data: List[Dict[str, str]]
    epochs: Optional[int] = settings.get("fine_tuning.default_epochs", 3)
    batch_size: Optional[int] = settings.get("fine_tuning.default_batch_size", 8)
    learning_rate: Optional[float] = settings.get("fine_tuning.default_learning_rate", 2e-5)

class TrainingConfig(BaseModel):
    model_name: str
    training_data_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    additional_params: Optional[Dict[str, Any]] = None

class JobResponse(BaseModel):
    job_id: str
    status: str
    start_time: str
    progress: float
    metrics: Dict[str, Any]

def validate_config(settings):
    required_keys = [
        ("llm", "max_tokens", 2048),
        ("llm", "temperature", 0.7),
        ("generation", "max_tokens", 2048),
        ("generation", "temperature", 0.7),
        ("server", "fastapi_port", 8000),
        ("server", "host", "localhost"),
    ]
    for section, key, default in required_keys:
        value = settings.get(section, {}).get(key, None)
        if value is None:
            logger.warning(f"Config key '{section}.{key}' is missing. Using default: {default}")

def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description="RAG + Fine-tuning System for DeepSeek models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents into the vector database")
    ingest_parser.add_argument(
        "--source_dir", "-s", 
        default=os.path.join(DATA_DIR, "raw"),
        help="Directory containing documents to ingest"
    )
    ingest_parser.add_argument(
        "--rebuild", "-r", 
        action="store_true",
        help="Rebuild the vector database"
    )
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the RAG API server")
    serve_parser.add_argument(
        "--port", "-p",
        type=int,
        default=FASTAPI_PORT,
        help="Port to run the server on"
    )
    serve_parser.add_argument(
        "--host", "-H",
        default=HOST,
        help="Host to bind the server to"
    )
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser("finetune", help="Fine-tune a DeepSeek model")
    finetune_parser.add_argument(
        "--dataset_dir", "-d",
        default=os.path.join(DATA_DIR, "training"),
        help="Directory containing training data"
    )
    
    # Query command for quick testing
    query_parser = subparsers.add_parser("query", help="Query the RAG system directly")
    query_parser.add_argument(
        "question",
        help="Question to ask the RAG system"
    )
    query_parser.add_argument(
        "--lob", "-l",
        default="auto",
        help="Line of business filter"
    )
    query_parser.add_argument(
        "--state", "-s",
        default="CA",
        help="State code filter (2 letters)"
    )
    
    # Kill command
    kill_parser = subparsers.add_parser("kill", help="Kill all running python processes (use with caution)")
    # Status command
    status_parser = subparsers.add_parser("status", help="Show status of all running python processes")
    
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to custom config YAML file (default: config/rag.yaml)"
    )
    
    return parser

def run_ingest(args):
    """Run document ingestion process."""
    logger.info("Starting document ingestion...")
    cmd = [
        sys.executable, 
        "ingest/bulk_ingest.py",
        "--source-dir", args.source_dir
    ]
    
    if args.rebuild:
        cmd.append("--rebuild")
    
    # Stream output live to the terminal
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        logger.info("Document ingestion completed successfully.")
    else:
        logger.error("Document ingestion failed.")
        sys.exit(1)

def run_serve(args):
    """Start the RAG API server."""
    logger.info(f"Starting RAG API server on {args.host}:{args.port}...")
    
    # Check if vector database exists
    if not os.path.exists(CHROMA_PERSIST_DIRECTORY):
        logger.error(f"Vector database not found at {CHROMA_PERSIST_DIRECTORY}. Run 'ingest' command first.")
        sys.exit(1)
    
    # Use uvicorn to run the FastAPI server
    cmd = [
        "uvicorn", 
        "app.main:app", 
        "--host", args.host,
        "--port", str(args.port),
        "--reload"
    ]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        logger.info("Server stopped.")

def run_finetune(args):
    """Run fine-tuning process."""
    logger.info("Starting model fine-tuning...")
    
    # Check if dataset directory exists
    if not os.path.exists(args.dataset_dir):
        logger.error(f"Dataset directory not found at {args.dataset_dir}")
        sys.exit(1)
    
    cmd = [
        sys.executable,
        "finetune/trainer.py",
        "--dataset_dir", args.dataset_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Fine-tuning completed successfully.")
        logger.info(result.stdout)
    else:
        logger.error("Fine-tuning failed.")
        logger.error(result.stderr)
        sys.exit(1)

def run_query(args):
    """Run a direct query against the RAG system."""
    import requests
    import json
    
    # First get a debug token for testing
    token_url = f"http://localhost:{FASTAPI_PORT}/debug/token?lob={args.lob}&state={args.state}"
    try:
        token_response = requests.get(token_url)
        token_response.raise_for_status()  # Raise exception for non-200 status codes
        token = token_response.json()["token"]
    except Exception as e:
        logger.error(f"Failed to get authentication token: {e}")
        logger.error("Make sure the RAG API server is running (use 'serve' command)")
        sys.exit(1)
    
    # Now make the actual query
    query_url = f"http://localhost:{FASTAPI_PORT}/query/query"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    data = {
        "question": args.question,
        "max_tokens": settings.get('llm', {}).get('max_tokens', 2048),
        "temperature": settings.get('llm', {}).get('temperature', 0.7),
        "stream": False
    }
    
    try:
        response = requests.post(query_url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for non-200 status codes
        result = response.json()
        
        print("\nAnswer:")
        print(result["answer"])
        print("\nSources:")
        for source in result.get("sources", []):
            print(f"- {source}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Query failed: {e}")
        if hasattr(e.response, 'text'):
            logger.error(f"Response: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

def run_kill(args):
    """Kill all running python processes (use with caution)."""
    import subprocess
    import platform
    logger.info("Killing all running python processes...")
    try:
        if platform.system() == "Windows":
            # Use PowerShell to kill all python* processes
            subprocess.run(["powershell", "-Command", "Get-Process python* | Stop-Process -Force"], check=True)
        else:
            # Use pkill on Unix
            subprocess.run(["pkill", "-f", "python"], check=True)
        logger.info("All python processes killed.")
    except Exception as e:
        logger.error(f"Failed to kill python processes: {e}")
        sys.exit(1)

def run_status(args):
    """Show status of all running python processes."""
    import subprocess
    import platform
    logger.info("Listing all running python processes...")
    try:
        if platform.system() == "Windows":
            # Use PowerShell to list all python* processes with details
            result = subprocess.run([
                "powershell", "-Command",
                "Get-Process python* | Select-Object Id,StartTime,ProcessName,Path | Format-Table -AutoSize"
            ], capture_output=True, text=True)
            print(result.stdout)
        else:
            # Use ps on Unix
            result = subprocess.run([
                "ps", "-eo", "pid,lstart,cmd"], capture_output=True, text=True)
            lines = [line for line in result.stdout.splitlines() if "python" in line]
            print("\n".join(lines))
        logger.info("Status listed above.")
    except Exception as e:
        logger.error(f"Failed to get python process status: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    # If a custom config is provided, reload settings
    if getattr(args, 'config', None):
        from core.settings import settings as global_settings
        from core.config import Config
        # Create a new Config instance with the custom path
        global_settings = Config(args.config)
        validate_config(global_settings)
    else:
        validate_config(settings)
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "serve":
        run_serve(args)
    elif args.command == "finetune":
        run_finetune(args)
    elif args.command == "query":
        run_query(args)
    elif args.command == "kill":
        run_kill(args)
    elif args.command == "status":
        run_status(args)

if __name__ == "__main__":
    main() 