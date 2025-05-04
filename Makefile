# Makefile for RAG + Fine-tuning Project

.PHONY: setup ingest serve finetune test clean help docker-build docker-up docker-down lint install-hooks streamlit rag

# Default Python command
PYTHON = python

# Port configuration
FASTAPI_PORT = 8000
VLLM_PORT = 8001
STREAMLIT_PORT = 8501

# Configuration variables
SOURCE_DIR = data/raw
TRAIN_DATA = data/training
EVAL_DATA = data/eval
MODEL_NAME = deepseek-llm:7b

help:
	@echo "Available commands:"
	@echo "  setup         - Run setup script to install dependencies"
	@echo "  ingest        - Ingest documents into the vector database"
	@echo "  serve         - Start the FastAPI server"
	@echo "  rag           - Start the RAG API server"
	@echo "  vllm          - Start vLLM server"
	@echo "  ollama        - Pull and run the DeepSeek model in Ollama"
	@echo "  finetune      - Run fine-tuning process on DeepSeek model"
	@echo "  streamlit     - Start the Streamlit UI"
	@echo "  test          - Run all tests"
	@echo "  docker-build  - Build Docker image"
	@echo "  docker-up     - Start all services with Docker Compose"
	@echo "  docker-down   - Stop all Docker services"
	@echo "  lint          - Run linters (ruff, black, isort)"
	@echo "  install-hooks - Install pre-commit hooks"
	@echo "  clean         - Clean up temporary files and databases"

# Setup commands for different platforms
ifeq ($(OS),Windows_NT)
setup:
	powershell.exe -ExecutionPolicy Bypass -File scripts/setup.ps1
else
setup:
	sudo bash scripts/setup.sh
endif

# Ingestion command
ifeq ($(OS),Windows_NT)
ingest:
	@echo "Ingesting documents from $(SOURCE_DIR)..."
	powershell.exe -ExecutionPolicy Bypass -File scripts/ingest.ps1 -dataDir $(SOURCE_DIR)
else
ingest:
	@echo "Ingesting documents from $(SOURCE_DIR)..."
	bash scripts/ingest.sh --data-dir $(SOURCE_DIR)
endif

# Server command
ifeq ($(OS),Windows_NT)
serve:
	@echo "Starting FastAPI server on port $(FASTAPI_PORT)..."
	powershell.exe -ExecutionPolicy Bypass -File scripts/serve.ps1 -port $(FASTAPI_PORT)
else
serve:
	@echo "Starting FastAPI server on port $(FASTAPI_PORT)..."
	bash scripts/serve.sh --port $(FASTAPI_PORT)
endif

# RAG server command
rag:
	@echo "Starting RAG API server on port $(FASTAPI_PORT)..."
	$(PYTHON) -m scripts.serve

vllm:
	@echo "Starting vLLM server on port $(VLLM_PORT)..."
	$(PYTHON) -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port $(VLLM_PORT) --model $(MODEL_NAME)

ollama:
	@echo "Pulling DeepSeek model in Ollama..."
	ollama pull $(MODEL_NAME)
	@echo "Starting Ollama service..."
	ollama serve

# Fine-tuning command
ifeq ($(OS),Windows_NT)
finetune:
	@echo "Fine-tuning DeepSeek model..."
	powershell.exe -ExecutionPolicy Bypass -File scripts/finetune.ps1
else
finetune:
	@echo "Fine-tuning DeepSeek model..."
	bash scripts/finetune.sh
endif

# Streamlit command
streamlit:
	@echo "Starting Streamlit UI on port $(STREAMLIT_PORT)..."
	streamlit run streamlit_app.py --server.port $(STREAMLIT_PORT) --server.address 0.0.0.0

# Test command
ifeq ($(OS),Windows_NT)
test:
	@echo "Running tests..."
	powershell.exe -ExecutionPolicy Bypass -File scripts/test.ps1
else
test:
	@echo "Running tests..."
	bash scripts/test.sh
endif

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t rag-app:latest .

docker-up:
	@echo "Starting all services with Docker Compose..."
	docker-compose up -d

docker-down:
	@echo "Stopping all Docker services..."
	docker-compose down

# Linting commands
lint:
	@echo "Running linters..."
	ruff check .
	black .
	isort .

install-hooks:
	@echo "Installing pre-commit hooks..."
	pre-commit install

clean:
	@echo "Cleaning up..."
	rm -rf data/chroma
	rm -rf data/qdrant
	rm -rf data/models
	rm -rf __pycache__
	rm -rf app/__pycache__
	rm -rf scripts/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache 