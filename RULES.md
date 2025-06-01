# Codebase Rules and Guidelines

This document outlines the rules and guidelines for maintaining the codebase integrity and preventing breaking changes.

## 1. LLM Backend Configuration

### Required Structure in `rag.yaml`
```yaml
llm:
  backend: "huggingface"  # Options: "huggingface" or "vllm"
  model: "microsoft/phi-2"
  temperature: 0.7
  max_tokens: 2048
  top_p: 0.95
  frequency_penalty: 0.0
  presence_penalty: 0.0
  repetition_penalty: 1.15
```

### Rules
- Always maintain both `huggingface` and `vllm` backends
- Default backend must be `huggingface` for Windows compatibility
- Keep all LLM parameters in the configuration file
- Do not hardcode LLM settings in the code

## 2. ShareGPT/Ollama Configuration

### Required Structure in `rag.yaml`
```yaml
ollama_llm:
  model: "deepseek-llm:7b"
  base_url: "http://localhost:11434"
  temperature: 0.7
  max_tokens: 4096
  top_p: 0.95
  frequency_penalty: 0.0
  presence_penalty: 0.0
  context_window: 4096
  stop_sequences: []
  timeout: 120
  retry_attempts: 3
  retry_delay: 2

sharegpt:
  query:
    query_template: |
      Generate {num_questions} relevant questions based on the following content.
      The questions should be specific, clear, and test understanding of key concepts.

      Content:
      {content}

      Questions:
    response_format: "newline"
    questions_per_chunk: 2
    min_question_length: 10
    max_question_length: 200
```

### Rules
- Keep these sections in `rag.yaml` for document ingestion
- Maintain all parameters for both Ollama and ShareGPT configurations
- Do not remove or modify the query template structure

## 3. Document Ingestion

### Required Structure in `rag.yaml`
```yaml
chunking:
  chunk_size: 256
  chunk_overlap: 200
  max_chunks: 10
  chunk_strategy: "recursive"
  min_chunk_size: 100
  max_chunk_size: 2000
  split_by: "paragraph"

vector_store:
  type: "chroma"
  persist_directory: "data/vector_store/chroma"
  collection_name: "documents"
  dimension: 384
```

### Rules
- Always use chunking settings from `rag.yaml`
- Maintain vector store configuration
- Keep document processing pipeline in `scripts/convert_to_sharegpt.py`

## 4. Code Structure

### Required Directory Structure
```
core/
  ├── llm_backends/
  │   ├── base.py
  │   ├── huggingface_backend.py
  │   ├── vllm_backend.py
  │   └── factory.py
  ├── rag_engine.py
  └── settings.py
scripts/
  ├── convert_to_sharegpt.py
  ├── batch_ingest.py
  └── simple_vector_ingest.py
```

### Rules
- Keep LLM backend abstraction in `core/llm_backends/`
- Maintain document ingestion pipeline in `scripts/convert_to_sharegpt.py`
- Keep RAG engine implementation in `core/rag_engine.py`
- Preserve metrics collection in `RAGMetrics` class

## 5. Error Handling

### Required Structure in `rag.yaml`
```yaml
error_handling:
  retry_attempts: 3
  retry_delay: 1000
  fallback_responses:
    generation_error: "I apologize, but I encountered an error while generating a response. Please try again."
    retrieval_error: "I apologize, but I encountered an error while retrieving information. Please try again."
    validation_error: "I apologize, but I couldn't generate a satisfactory response. Please try rephrasing your question."
```

### Rules
- Always maintain fallback responses
- Keep retry configuration
- Preserve error handling structure

## 6. Metrics and Monitoring

### Required Structure in `rag.yaml`
```yaml
monitoring:
  enabled: true
  log_level: "INFO"
  metrics:
    - "latency"
    - "throughput"
    - "error_rate"
    - "token_usage"
```

### Rules
- Keep all metrics collection in `RAGMetrics` class
- Maintain monitoring configuration
- Preserve metrics structure

## 7. File Paths

### Required Structure in `rag.yaml`
```yaml
paths:
  data_dir: "./data"
  training_data_dir: "./data/training/data"
  models_dir: "./data/models"
  logs_dir: "./data/logs"
  cache_dir: "./data/cache"
  metrics_dir: "./data/metrics"
  jobs_dir: "./data/jobs"
```

### Rules
- Keep all data directories in `rag.yaml`
- Maintain consistent path structure
- Do not hardcode paths in the code

## 8. Dependencies

### Required Versions
- Python: 3.11.9
- CUDA: 12.1
- vLLM: 0.9.0.1
- HuggingFace Transformers
- ChromaDB
- Sentence Transformers

### Rules
- Maintain compatibility with listed versions
- Document any version changes
- Test with all dependencies

## 9. Testing Requirements

### Pre-change Checklist
1. Test LLM backend initialization
2. Test document ingestion
3. Test RAG query processing
4. Verify metrics collection
5. Check error handling

### Rules
- Run all tests before making changes
- Document test results
- Maintain test coverage

## 10. Documentation

### Required Documentation
- Configuration options in `rag.yaml`
- Docstrings in all Python files
- Changes to configuration structure

### Rules
- Keep all configuration options documented
- Maintain docstrings
- Document configuration changes

## 11. Config-Driven Approach

### Required Structure in `rag.yaml`
```yaml
# All configuration must be in rag.yaml
config:
  version: "1.0"
  environment: "development"  # Options: "development", "staging", "production"
  debug: false
  strict_mode: true  # Enforce all configuration rules
```

### Rules
- All configuration must be in `rag.yaml`
- No hardcoded values in code
- Use `Config` class to access settings
- Follow hierarchical structure in YAML
- Document all configuration options
- Version control the configuration
- Use environment-specific configs
- Validate config on startup
- Keep sensitive data in `.env` only
- Use type hints for config values

### Configuration Access Pattern
```python
from core.config import Config

# Initialize config
config = Config("config/rag.yaml")

# Access settings
llm_backend = config.get("llm.backend")
model_name = config.get("llm.model")
```

### Best Practices
1. Group related settings together
2. Use descriptive names
3. Provide default values
4. Document all options
5. Validate on load
6. Use type hints
7. Keep it DRY
8. Version control
9. Environment aware
10. Secure sensitive data

## Validation

To ensure these rules are followed, run the validation script:
```bash
python scripts/validate_rules.py
```

This will check:
1. Configuration file structure
2. Required directories and files
3. Code structure compliance
4. Dependency versions
5. Documentation completeness 