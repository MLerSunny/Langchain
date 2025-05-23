#!/usr/bin/env python
"""
Main entry point for RAG + Fine-tuning system for DeepSeek models.
This script provides a unified CLI for all components.
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Import the config to make it available
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from core.settings import (
    settings,
    DATA_DIR,
    CHROMA_PERSIST_DIRECTORY,
    FASTAPI_PORT,
    HOST
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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
    
    return parser

def run_ingest(args):
    """Run document ingestion process."""
    logger.info("Starting document ingestion...")
    cmd = [
        sys.executable, 
        "ingest/ingest.py",
        "--source_dir", args.source_dir
    ]
    
    if args.rebuild:
        cmd.append("--rebuild")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        logger.info("Document ingestion completed successfully.")
        logger.info(result.stdout)
    else:
        logger.error("Document ingestion failed.")
        logger.error(result.stderr)
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
        "serve.main:app", 
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
        token = token_response.json()["token"]
    except Exception as e:
        logger.error(f"Failed to get authentication token: {e}")
        logger.error("Make sure the RAG API server is running (use 'serve' command)")
        sys.exit(1)
    
    # Now make the actual query
    query_url = f"http://localhost:{FASTAPI_PORT}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    data = {
        "model": "deepseek-insurance",
        "messages": [
            {"role": "user", "content": args.question}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(query_url, headers=headers, json=data)
        result = response.json()
        
        # Extract and print the answer
        answer = result["choices"][0]["message"]["content"]
        logger.info(f"\nQuestion: {args.question}")
        logger.info(f"\nAnswer: {answer}")
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        sys.exit(1)

def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command == "ingest":
        run_ingest(args)
    elif args.command == "serve":
        run_serve(args)
    elif args.command == "finetune":
        run_finetune(args)
    elif args.command == "query":
        run_query(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 