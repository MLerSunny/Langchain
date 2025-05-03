# Quickstart Guide

This guide will help you get started with the RAG and Fine-tuning system.

## Prerequisites

- Python 3.10+
- Docker and Docker Compose (for containerized setup)
- GPU with CUDA support (recommended for training)

## Installation

### Option 1: Local Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Set up a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy the environment file and configure it:

   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

### Option 2: Docker Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Copy the environment file and configure it:

   ```bash
   cp env.example .env
   # Edit .env with your settings
   ```

3. Build and start the containers:

   ```bash
   docker-compose up --build
   ```

## Configuration

The application uses a centralized configuration system through environment variables and the `.env` file. Here are the key settings:

### Core Settings

Edit the `.env` file to configure:

| Variable | Description | Default |
|----------|-------------|---------|
| `JWT_SECRET` | Secret key for JWT authentication | `CHANGE_ME_IN_PRODUCTION` |
| `VECTOR_DB` | Vector database backend to use (`chroma` or `qdrant`) | `chroma` |
| `DATA_PATH` | Path to raw data files | `data/raw` |
| `CHROMA_PATH` | Path to Chroma DB storage | `.chroma` |
| `QDRANT_HOST` | Hostname for Qdrant server | `localhost` |
| `QDRANT_PORT` | Port for Qdrant server | `6333` |
| `MODEL_NAME` | Model name or path to use for inference | `deepseek-coder:7b-instruct-v1.5` |

### Security Notice

⚠️ **Important**: The application will refuse to start if `JWT_SECRET` is set to the default value. Always set a proper secret in production!

## Usage

### 1. Document Ingestion

First, ingest documents to create the vector database:

```bash
python -m ingest.ingest --source-dir data/raw
```

This will process documents from the `data/raw` directory and store embeddings in the configured vector database.

#### Ingestion Modes

The system supports two ingestion modes:

- **Upsert mode** (default): Only new or modified documents are processed and added to the database. Documents that haven't changed since the last ingestion are skipped. This is now the default mode to improve efficiency and prevent unnecessary re-embedding of unchanged files.
- **Recreate mode**: Rebuilds the vector database from scratch, removing all existing embeddings. Use this mode when you want to completely refresh your vector store or change embedding parameters.

```bash
# Use upsert mode (default)
python -m ingest.ingest --source-dir data/raw --mode upsert

# Use recreate mode for complete re-indexing
python -m ingest.ingest --source-dir data/raw --mode recreate
```

#### Switching Between Vector Databases

You can choose between Chroma (embedded) and Qdrant (client-server):

```bash
# Use Chroma (default)
export VECTOR_DB=chroma
# OR
# Use Qdrant 
export VECTOR_DB=qdrant
```

Or specify at runtime:

```bash
python -m ingest.ingest --db-type qdrant
```

### 2. Start the API Server

Run the API server:

```bash
python -m app.main
```

Or using the Makefile:

```bash
make serve
```

The API will be available at <http://localhost:8000>

### 3. Fine-tune the Model (Optional)

To fine-tune the model on your domain-specific data:

```bash
python -m finetune.trainer --dataset_dir data/training
```

## Architecture

The system uses a Retrieval-Augmented Generation (RAG) approach with optional fine-tuning:

![Architecture Diagram](arch.drawio)

## API Authentication

The API uses JWT tokens for authentication. To make authenticated requests:

1. Get a valid JWT token from your authentication system
2. Include it in your requests:

```bash
curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is covered in my policy?", "top_k": 5}'
```

## Rate Limiting

API requests are rate-limited to 30 requests per minute per client to prevent abuse.

## License

See the LICENSE file for details.
