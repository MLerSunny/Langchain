# RAG + Fine-tuning System

This is a Retrieval-Augmented Generation (RAG) system with fine-tuning capabilities, built using FastAPI and LangChain.

## Features

- Document ingestion and vector storage using FAISS
- Query processing with RAG
- Model fine-tuning capabilities
- Metrics tracking
- RESTful API endpoints
- Streamlit web interface

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
uvicorn app.main:app --reload --port 8000
```

4. Start the Streamlit interface (in a separate terminal):
```bash
streamlit run streamlit_app.py --server.port 8501
```

## API Endpoints

- `GET /health`: Health check endpoint
- `GET /metrics`: Get system metrics
- `POST /query`: Process a query through the RAG system
- `POST /fine-tune`: Start a fine-tuning job
- `GET /fine-tuning-status/{job_id}`: Get fine-tuning job status

## Usage

1. Add documents to the system:
```python
from core.rag import RAGEngine

rag = RAGEngine()
rag.add_documents(["Your document text here"])
```

2. Query the system:
```python
response, sources = rag.process_query("Your question here")
```

3. Start fine-tuning:
```python
from core.fine_tuning import FineTuningManager

manager = FineTuningManager()
job_id = manager.start_training(
    model_name="gpt2",
    training_data=[{"text": "Your training data"}]
)
```

## Architecture

- `app/main.py`: FastAPI application
- `core/rag.py`: RAG engine implementation
- `core/metrics.py`: Metrics collection
- `core/fine_tuning.py`: Fine-tuning management
- `streamlit_app.py`: Web interface

## License

MIT

# Configuration System

The project uses a comprehensive configuration system to manage all settings and parameters. This makes it easy to customize the behavior of the application without modifying the code.

## Configuration Files

The following configuration files are used:

- `config/model_optimization.yaml`: Model optimization settings
- `config/rag.yaml`: RAG (Retrieval-Augmented Generation) settings
- `config/security.yaml`: Security and access control settings

## Creating Configuration Files

To create the initial configuration files with default values, run:
```bash
python scripts/create_config.py
```

This will create all necessary configuration files with sensible defaults. You can then modify these files according to your needs.

## Validating Configuration

To validate your configuration files, run:
```bash
python scripts/validate_config.py
```

This will check all configuration files for:
- Required sections and fields
- Correct data types
- Valid values
- Security settings

## Configuration Structure

### Model Optimization Configuration

```yaml
quantization:
  use_4bit: true
  use_8bit: false
  dtype: float16
  quantization_type: nf4

lora:
  use_lora: true
  rank: 8
  alpha: 16
  dropout: 0.1
  target_modules: ["q_proj", "v_proj"]
  bias: none
  task_type: CAUSAL_LM

memory:
  max_memory: 8GB
  offload_folder: data/offload
  use_cache: true
  gradient_checkpointing: true
  data_loader:
    num_workers: 4
    pin_memory: true

training:
  mixed_precision: fp16
  optimizer: adamw_torch
  report_to: ["tensorboard"]
  data_loader:
    batch_size: 8
    gradient_accumulation_steps: 4

model_loading:
  dtype: float16
  device_map: auto
  use_cache: true
  max_memory: 8GB

tokenizer:
  padding_token: <pad>
  max_length: 2048
  padding_side: right
  truncation_side: right
```

### RAG Configuration

```yaml
chunking:
  chunk_size: 512
  chunk_overlap: 50
  max_chunks: 10

retrieval:
  similarity_threshold: 0.7
  max_results: 5
  rerank: true

vector_store:
  type: chroma
  persist_directory: data/chroma
  collection_name: documents

embeddings:
  model: sentence-transformers/all-MiniLM-L6-v2
  cache_dir: data/embeddings_cache

cache:
  ttl: 3600
  max_size: 1000

query:
  query_template: "Answer the following question based on the context: {question}"
  system_prompt: "You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely."

generation:
  max_tokens: 512
  temperature: 0.7
  top_p: 0.9
  frequency_penalty: 0.0
  presence_penalty: 0.0

error_handling:
  max_retries: 3
  retry_delay: 1
  fallback_model: gpt-3.5-turbo
```

### Security Configuration

```yaml
rate_limits:
  requests_per_minute: 60
  requests_per_hour: 1000
  max_query_length: 1000

file_validation:
  max_file_size: 10485760  # 10MB
  allowed_types: [".txt", ".pdf", ".json", ".csv", ".md"]

input_validation:
  max_length: 10000
  sensitive_patterns:
    - \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b  # Email
    - \b\d{3}[-.]?\d{3}[-.]?\d{4}\b  # Phone
    - \b\d{16}\b  # Credit card
    - \b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b  # API key

security_headers:
  x_frame_options: DENY
  x_content_type_options: nosniff
  x_xss_protection: "1; mode=block"
  strict_transport_security: "max-age=31536000; includeSubDomains"

jwt:
  access_token_expiry: 3600  # 1 hour
  refresh_token_expiry: 604800  # 7 days
  algorithm: HS256

cors:
  allowed_origins: ["*"]
  allowed_methods: ["GET", "POST", "PUT", "DELETE"]
  allowed_headers: ["*"]
  exposed_headers: ["Content-Length", "X-Request-ID"]
  max_age: 3600

password_policy:
  min_length: 12
  require_uppercase: true
  require_lowercase: true
  require_numbers: true
  require_special: true
  max_age_days: 90
  history_size: 5

session:
  timeout_minutes: 30
  max_failed_attempts: 5
  lockout_duration_minutes: 15

api_security:
  max_requests_per_ip: 1000
  max_requests_per_user: 100
  require_api_key: true
```

## Environment Variables

The following environment variables can be used to override configuration file paths:

- `MODEL_OPTIMIZATION_CONFIG_PATH`: Path to model optimization configuration file
- `RAG_CONFIG_PATH`: Path to RAG configuration file
- `SECURITY_CONFIG_PATH`: Path to security configuration file

## Best Practices

1. Always validate your configuration files before deploying
2. Use environment variables for sensitive information
3. Keep configuration files in version control
4. Document any changes to configuration structure
5. Use the validation script to catch configuration errors early
