# RAG + Fine-tuning System for DeepSeek Models

A complete system for building RAG (Retrieval-Augmented Generation) applications and fine-tuning DeepSeek models on Windows with NVIDIA RTX GPUs.

## Features

- Document ingestion pipeline with support for multiple file formats
- Vector database for semantic search using ChromaDB
- FastAPI server with RAG-based question answering
- vLLM integration for faster inference
- Fine-tuning capabilities using LoRA/QLoRA
- PowerShell setup script for Windows environments
- Centralized configuration system for easy customization
- Unified CLI interface for all system components
- Streamlit UI for interactive chat experience

## System Requirements

- Windows 10 or higher
- NVIDIA GPU with at least 8GB VRAM (RTX series recommended)
- 16GB+ system RAM
- Python 3.11
- Administrator access (for setup script)

## Quick Start

### Setup

1. Clone this repository:

   ```
   git clone <repository-url>
   cd <repository-name>
   ```

2. Run the setup script (as Administrator):

   ```
   make setup
   ```

   This will install:
   - CUDA 12.1
   - PyTorch with CUDA support
   - Intel oneAPI optimization libraries
   - bitsandbytes for quantization
   - All required Python dependencies
   - Ollama for model hosting

3. Pull the DeepSeek model via Ollama:

   ```
   make ollama
   ```

### Using the Unified CLI Interface

The system provides a unified command-line interface for all components:

```
python main.py <command> [options]
```

Available commands:

- `ingest`: Ingest documents into the vector database
- `serve`: Start the RAG API server
- `finetune`: Fine-tune a DeepSeek model
- `query`: Query the RAG system directly

Example usage:

```bash
# Ingest documents from a custom directory and rebuild the vector database
python main.py ingest --source_dir data/custom_docs --rebuild

# Start the API server on a custom port
python main.py serve --port 9000

# Fine-tune a model with a custom dataset
python main.py finetune --dataset_dir data/custom_training

# Query the system directly
python main.py query "What is a retrieval-augmented generation?" --lob general --state CA
```

### Using the RAG System with Make

1. Prepare your documents by placing them in the `data/raw` directory

2. Ingest the documents into the vector database:

   ```
   make ingest
   ```

3. Start the FastAPI server:

   ```
   make serve
   ```

4. In a separate terminal, start vLLM for faster inference (optional):

   ```
   make vllm
   ```

5. Access the API at <http://localhost:8080>

### Using the Streamlit UI

The system includes a Streamlit chat interface for interacting with the DeepSeek model:

```bash
# Start the Streamlit app
make streamlit
```

or

```bash
streamlit run streamlit_app.py
```

Access the UI at <http://localhost:8501>

### Fine-tuning Models

1. Prepare your training data in ShareGPT format (JSON):

   ```json
   {
     "conversations": [
       {"from": "user", "value": "What is the capital of France?"},
       {"from": "assistant", "value": "The capital of France is Paris."}
     ]
   }
   ```

2. Place your training data in `data/training/` directory

3. Run the fine-tuning process:

   ```
   python main.py finetune
   ```

## Configuration System

The system uses a centralized configuration approach with all settings defined in `core/settings.py`. Configuration is loaded from:

1. Environment variables (highest priority)
2. `core/rag.yaml` for RAG-specific settings
3. Default values in `core/settings.py`

Key configuration groups:

- **Base directories**: Paths for data, models, and other directories
- **API server settings**: Host, ports, and API configuration
- **Model settings**: Default model, Ollama URL, context window, generation parameters
- **Vector database settings**: ChromaDB configuration, embedding model
- **RAG settings**: Chunk sizes, retrieval parameters, prompts
- **Fine-tuning settings**: Training parameters, dataset paths, learning rates

To customize the configuration:

1. Edit `core/rag.yaml` for RAG-specific settings
2. Set environment variables for temporary changes
3. Modify `core/settings.py` for permanent changes

Example environment variables:

```bash
# Set custom directories
export BASE_DIR=/path/to/custom/dir
export DATA_DIR=/path/to/data

# Change API settings
export FASTAPI_PORT=9000
export VLLM_PORT=8100

# Modify model behavior
export DEFAULT_MODEL=deepseek-coder:33b-instruct
export TEMPERATURE=0.2
export MAX_TOKENS=4096
```

## Data Organization

The system uses the following data organization:

- `data/raw/` - Original source documents (PDFs, DOCs, etc.)
- `data/processed/` - Cleaned and processed documents
- `data/chroma/` - ChromaDB vector database
- `data/training/` - Training datasets for fine-tuning
- `data/eval/` - Evaluation datasets for fine-tuning
- `data/models/` - Fine-tuned models

**Note**: For production deployment, move large sample files to external storage and provide a download script to reduce repository size and Docker build context.

## Pulling DeepSeek Models via Ollama

DeepSeek models can be easily pulled and run using Ollama:

```bash
# Pull the model
ollama pull deepseek-coder:7b-instruct-v1.5

# Run the model
ollama run deepseek-coder:7b-instruct-v1.5
```

You can also use the provided make command:

```bash
make ollama
```

This will pull the latest DeepSeek models and start the Ollama service.

## Fine-tuning Guide

The system supports fine-tuning DeepSeek models using parameter-efficient methods:

1. Prepare your training data in the required format (JSONL with 'prompt' and 'completion' fields)

2. Configure the fine-tuning parameters in `core/settings.py` or pass them as command-line arguments:

   ```bash
   python scripts/finetune.py \
     --model_name_or_path deepseek-coder:7b-instruct-v1.5 \
     --dataset_path data/training/custom_data.jsonl \
     --eval_dataset_path data/eval/custom_eval.jsonl \
     --use_lora True \
     --lora_rank 8 \
     --use_4bit True
   ```

3. The fine-tuned model will be saved to the directory specified in `core/settings.py` or passed via `--output_dir`

4. To use the fine-tuned model, you can load it with the `transformers` and `peft` libraries or convert it to GGUF format for use with Ollama

## API Reference

The FastAPI server provides the following endpoints:

- `GET /`: Root endpoint
- `GET /health`: Health check endpoint
- `GET /models`: List available models
- `POST /query`: Query endpoint for RAG-based QA

Example query:

```bash
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is a retrieval-augmented generation?",
    "model": "deepseek-coder:7b-instruct-v1.5",
    "temperature": 0.1,
    "max_tokens": 500,
    "use_rag": true,
    "k": 3
  }'
```

## Development

### Project Structure

```
.
├── app/                  # FastAPI application
│   ├── main.py           # Main application file
│   ├── proxy.py          # vLLM proxy implementation
│   ├── routes/           # API route definitions
│   └── utils/            # Utility functions
├── Chatbot-Deepseek/     # Streamlit UI application
│   └── app.py            # Streamlit app code
├── core/                 # Core configuration
│   └── settings.py       # Settings and configuration
├── data/                 # Data directories
│   ├── chroma/           # ChromaDB vector store
│   ├── eval/             # Evaluation datasets
│   ├── models/           # Fine-tuned models
│   ├── processed/        # Processed documents
│   ├── raw/              # Source documents
│   └── training/         # Training datasets
├── examples/             # Example notebooks and code
│   └── app.ipynb         # RAG example notebook
├── scripts/              # Scripts for various operations
│   ├── finetune.py       # Fine-tuning script
│   ├── ingest.py         # Document ingestion script
│   └── setup.ps1         # Windows setup script
├── tests/                # Test files
│   ├── app/              # UI tests
│   └── test_app.py       # API tests
├── docker-compose.yml    # Docker compose configuration
├── Makefile              # Makefile for common operations
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies
```

### Running Tests

```bash
make test
```

## License

[MIT License](LICENSE)

## Acknowledgments

- [DeepSeek](https://github.com/deepseek-ai/DeepSeek-Coder) for the base models
- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [vLLM](https://github.com/vllm-project/vllm) for fast inference
- [Ollama](https://github.com/ollama/ollama) for model hosting
- [PEFT](https://github.com/huggingface/peft) for parameter-efficient fine-tuning

# Fine-tuning Instructions

This repository includes scripts for fine-tuning language models on custom datasets. The following guide will help you run the fine-tuning process.

## Prerequisites

- Python 3.7+
- PyTorch
- Transformers library
- PEFT (Parameter-Efficient Fine-Tuning) library

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Fine-tuning on Windows

To run fine-tuning on Windows, use the following command:

```powershell
.\scripts\finetune.ps1 -skipDeepSpeed -modelName facebook/opt-350m
```

### Troubleshooting Windows Issues

When running on Windows, you might encounter several issues:

1. **Missing DeepSpeed**: If DeepSpeed is not installed or doesn't work on your system, use the `-skipDeepSpeed` flag to use standard training.

2. **CUDA Compatibility**: If CUDA is not available, the script will automatically disable 4-bit quantization and fp16 training.

3. **Model Access**: Some models like `deepseek-ai/deepseek-llm-32b-instruct` require authentication. Use public models like `facebook/opt-350m` if you don't have access.

4. **JWT Secret**: The script contains a validation check for JWT secrets. This has been configured in the `core/settings.py` file.

## Fine-tuning on Linux

To run fine-tuning on Linux, use the following command:

```bash
./scripts/finetune.sh
```

## Parameters

The fine-tuning script accepts the following parameters:

- `-modelName`: Model name or path (default: "facebook/opt-350m")
- `-configFile`: DeepSpeed config file (default: "core/deepspeed_zero3.json")
- `-outputDir`: Output directory for checkpoints (default: "checkpoints/r1_ins_lora")
- `-trainEpochs`: Number of training epochs (default: 3)
- `-batchSize`: Batch size (default: 1)
- `-gradAccumSteps`: Gradient accumulation steps (default: 8)
- `-skipDeepSpeed`: Skip using DeepSpeed for training

## Output

The trained model will be saved in the output directory specified by `-outputDir`. The model can then be used for inference or further fine-tuning.
