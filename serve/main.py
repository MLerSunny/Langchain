"""
FastAPI server for local Insurance RAG with DeepSeek LLM integration.
This server provides an OpenAI-compatible API endpoint and handles:
1. JWT auth with insurance metadata extraction
2. Context retrieval from Chroma using filters
3. Forwarding to a vLLM server with streaming support
"""

import os
import json
import jwt
import httpx
import logging
import sys
import torch
import functools
import asyncio
import warnings
import importlib
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.embeddings import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import LLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.vectorstores import Qdrant
from langchain_core.documents import Document

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.settings import (
    settings,
    RAG_CONFIG
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Retrieve configuration
retriever_config = RAG_CONFIG.get("retriever", {})
prompts_config = RAG_CONFIG.get("prompts", {})
vector_db_config = RAG_CONFIG.get("vector_db", {})

# Initialize FastAPI
app = FastAPI(title="Insurance RAG API", 
              description="Local RAG system with fine-tuned DeepSeek LLM")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT settings
JWT_SECRET = settings.jwt_secret
JWT_ALGORITHM = settings.jwt_algorithm

# vLLM server settings
VLLM_HOST = os.getenv("VLLM_HOST", f"http://llm:8001")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Vector store settings from centralized config
COLLECTION_NAME = vector_db_config.get("collection_name", settings.qdrant_collection_name)

# Model Request/Response schemas
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = settings.temperature
    top_p: float = 0.95
    max_tokens: Optional[int] = settings.max_tokens
    stream: bool = False
    stop: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

# State to track vector store connection
class AppState:
    embeddings = None
    vectorstore = None
    db_type = "chroma"  # Default DB type

app_state = AppState()

# Helper function to stream responses from vLLM
async def stream_chat_response(response: httpx.Response):
    """Stream the chat response from vLLM server."""
    try:
        async for chunk in response.aiter_bytes():
            if chunk:
                yield chunk
    except Exception as e:
        logger.error(f"Error in streaming response: {e}")
        yield b'{"error": "Error in streaming response"}'

# Function to check vLLM server health and fallback to Ollama if needed
async def check_llm_server_health():
    global VLLM_HOST
    
    # First try the configured VLLM_HOST
    try:
        async with httpx.AsyncClient() as client:
            # Different endpoints for vLLM and Ollama
            if "11434" in VLLM_HOST:  # This looks like Ollama
                response = await client.get(f"{VLLM_HOST}/api/tags", timeout=3.0)
            else:  # Try vLLM endpoint
                response = await client.get(f"{VLLM_HOST}/health", timeout=3.0)
                
            if response.status_code == 200:
                logger.info(f"Successfully connected to LLM server at {VLLM_HOST}")
                return True
    except Exception as e:
        logger.warning(f"LLM server at {VLLM_HOST} not available: {e}")
    
    # Always try Ollama as fallback if the first attempt failed
    try:
        logger.info(f"Trying to use Ollama as fallback at {OLLAMA_URL}")
        VLLM_HOST = OLLAMA_URL  # Set to Ollama URL
        
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags", timeout=3.0)
            if response.status_code == 200:
                logger.info(f"Successfully connected to Ollama at {OLLAMA_URL}")
                return True
    except Exception as e:
        logger.error(f"Ollama server not available: {e}")
    
    # If all attempts fail, check if localhost:11434 works
    if OLLAMA_URL != "http://localhost:11434" and VLLM_HOST != "http://localhost:11434":
        try:
            test_url = "http://localhost:11434"
            logger.info(f"Trying localhost Ollama at {test_url}")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{test_url}/api/tags", timeout=3.0)
                if response.status_code == 200:
                    logger.info(f"Successfully connected to localhost Ollama")
                    VLLM_HOST = test_url
                    return True
        except Exception as e:
            logger.error(f"Localhost Ollama not available: {e}")
    
    return False

# Initialize vector store if needed
def initialize_empty_vector_store():
    """Create an empty vector store if none exists."""
    logger.info("Creating empty vector store...")
    
    # Create a sample document so the vector store is initialized
    sample_docs = [
        Document(
            page_content="This is a sample document. Please ingest real documents for better results.",
            metadata={"source": "sample", "lob": "general", "state": "ALL"}
        )
    ]
    
    # Create the vector store
    try:
        os.makedirs(settings.chroma_persist_directory, exist_ok=True)
        
        # Use a direct call to Chroma here, not depending on app_state
        embedding_function = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        vector_store = Chroma.from_documents(
            documents=sample_docs,
            embedding=embedding_function,
            persist_directory=settings.chroma_persist_directory
        )
        vector_store.persist()
        logger.info("Successfully created sample vector store")
        return vector_store
    except Exception as e:
        logger.error(f"Failed to initialize vector store: {e}")
        return None

# Initialize embedding model and vector store
@app.on_event("startup")
async def startup_db_client():
    # Check LLM server health
    llm_available = await check_llm_server_health()
    if not llm_available:
        logger.warning("No LLM server available. API will have limited functionality.")
    
    # Initialize embeddings model
    app_state.embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Try to connect to Qdrant first, fallback to Chroma
    try:
        from qdrant_client import QdrantClient
        qdrant_url = settings.qdrant_url
        client = QdrantClient(url=qdrant_url)
        # Check if Qdrant is up
        collections = client.get_collections().collections
        collection_exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if collection_exists:
            app_state.vectorstore = Qdrant(
                client=client,
                collection_name=COLLECTION_NAME,
                embedding_function=app_state.embeddings
            )
            app_state.db_type = "qdrant"
            logger.info(f"Connected to Qdrant at {qdrant_url}, collection: {COLLECTION_NAME}")
        else:
            logger.warning(f"Qdrant collection {COLLECTION_NAME} not found, falling back to Chroma")
            # Fallback to Chroma
            if os.path.exists(settings.chroma_persist_directory):
                try:
                    app_state.vectorstore = Chroma(
                        persist_directory=settings.chroma_persist_directory,
                        embedding_function=app_state.embeddings,
                    )
                    # Check if there are documents in the store
                    if app_state.vectorstore._collection.count() == 0:
                        logger.warning("Chroma database is empty, initializing with a sample document")
                        app_state.vectorstore = initialize_empty_vector_store()
                    
                    app_state.db_type = "chroma"
                    logger.info(f"Connected to Chroma at {settings.chroma_persist_directory}")
                except Exception as e:
                    logger.error(f"Error connecting to Chroma: {e}")
                    app_state.vectorstore = initialize_empty_vector_store()
            else:
                logger.warning(f"Creating Chroma directory at {settings.chroma_persist_directory}")
                os.makedirs(settings.chroma_persist_directory, exist_ok=True)
                app_state.vectorstore = initialize_empty_vector_store()
                app_state.db_type = "chroma"
    except ImportError:
        logger.warning("Qdrant client not installed, using Chroma")
        # Use Chroma
        if os.path.exists(settings.chroma_persist_directory):
            try:
                app_state.vectorstore = Chroma(
                    persist_directory=settings.chroma_persist_directory,
                    embedding_function=app_state.embeddings,
                )
                # Check if there are documents in the store
                if app_state.vectorstore._collection.count() == 0:
                    logger.warning("Chroma database is empty, initializing with a sample document")
                    app_state.vectorstore = initialize_empty_vector_store()
                
                app_state.db_type = "chroma"
                logger.info(f"Connected to Chroma at {settings.chroma_persist_directory}")
            except Exception as e:
                logger.error(f"Error connecting to Chroma: {e}")
                app_state.vectorstore = initialize_empty_vector_store()
        else:
            logger.warning(f"Creating Chroma directory at {settings.chroma_persist_directory}")
            os.makedirs(settings.chroma_persist_directory, exist_ok=True)
            app_state.vectorstore = initialize_empty_vector_store()
            app_state.db_type = "chroma"
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        # Fallback to Chroma
        if os.path.exists(settings.chroma_persist_directory):
            try:
                app_state.vectorstore = Chroma(
                    persist_directory=settings.chroma_persist_directory,
                    embedding_function=app_state.embeddings,
                )
                # Check if there are documents in the store
                if app_state.vectorstore._collection.count() == 0:
                    logger.warning("Chroma database is empty, initializing with a sample document")
                    app_state.vectorstore = initialize_empty_vector_store()
                
                app_state.db_type = "chroma"
                logger.info(f"Connected to Chroma at {settings.chroma_persist_directory}")
            except Exception as e:
                logger.error(f"Error connecting to Chroma: {e}")
                app_state.vectorstore = initialize_empty_vector_store()
        else:
            logger.warning(f"Creating Chroma directory at {settings.chroma_persist_directory}")
            os.makedirs(settings.chroma_persist_directory, exist_ok=True)
            app_state.vectorstore = initialize_empty_vector_store()
            app_state.db_type = "chroma"

    # Initialize vector store if needed
    if not app_state.vectorstore or (hasattr(app_state.vectorstore, '_collection') and app_state.vectorstore._collection.count() == 0):
        logger.warning("Vector store is empty or not available, initializing with sample document")
        app_state.vectorstore = initialize_empty_vector_store()
        app_state.db_type = "chroma"

# JWT auth dependency
async def get_token_data(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    
    except jwt.PyJWTError as e:
        logger.error(f"JWT validation error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")
    except Exception as e:
        logger.error(f"Auth error: {e}")
        raise HTTPException(status_code=401, detail="Authentication error")

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "ok", "vectorstore": app_state.db_type if app_state.vectorstore else "not_connected"}

# Generate a sample JWT token for testing
@app.get("/debug/token")
async def create_test_token(lob: str = "auto", state: str = "CA"):
    """Generate a test JWT token with insurance metadata (for development only)"""
    payload = {
        "sub": "test-user",
        "lob": lob.lower(),
        "state": state.upper(),
        "exp": datetime.utcnow() + timedelta(days=1)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"token": token}

# Cache for retrieval results
@functools.lru_cache(maxsize=256)
def retrieve_docs_cached(query: str, lob: str, state: str, k: int):
    """Cached version of document retrieval"""
    # This function will be called by the async retriever
    # but will cache results for repeated queries
    
    if not app_state.vectorstore:
        return "No vector store available. Please ingest documents first."
    
    # Set up metadata filters
    metadata_filters = {}
    if lob and lob.lower() != "general" and lob.lower() != "all":
        metadata_filters["lob"] = lob.lower()
    if state and state.lower() != "all":
        metadata_filters["state"] = state.upper()

    # Get retriever parameters from config
    search_type = retriever_config.get("search_type", "mmr")
    lambda_mult = retriever_config.get("lambda_mult", 0.3)
    
    try:
        # Create retriever with metadata filtering
        if search_type == "mmr":
            retriever = app_state.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "lambda_mult": lambda_mult,
                    "filter": metadata_filters if metadata_filters else None
                }
            )
        else:
            retriever = app_state.vectorstore.as_retriever(
                search_kwargs={
                    "k": k,
                    "filter": metadata_filters if metadata_filters else None
                }
            )
        
        # Get documents
        docs = retriever.invoke(query)
        return docs
    except Exception as e:
        logger.error(f"Error during document retrieval: {e}")
        return f"Error during document retrieval: {e}"

# Async retriever for concurrent document retrieval
async def retrieve_context(query: str, lob: str = None, state: str = None, k: int = None):
    """Async wrapper for document retrieval with metadata filtering"""
    if not k:
        k = retriever_config.get("num_documents", 3)
    
    if not app_state.vectorstore:
        return "No vector store available. Please ingest documents first."
    
    # Use a thread pool to call the synchronous function
    loop = asyncio.get_event_loop()
    docs = await loop.run_in_executor(
        None, retrieve_docs_cached, query, lob, state, k
    )
    
    if isinstance(docs, str):  # Error message
        return docs
    
    # Format the documents into a context string
    context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        content = doc.page_content.strip()
        context += f"Document {i+1} [Source: {source}]:\n{content}\n\n"
    
    return context if context else "No relevant documents found."

# Format prompt with context and templates
def format_prompt(context: str, query: str):
    """
    Format the prompt for the LLM.
    
    Args:
        context: Retrieved context
        query: User question
        
    Returns:
        List of formatted messages
    """
    system_template = prompts_config.get("system_template", 
        "You are an insurance expert assistant. Answer the question based strictly on the provided context. If the information isn't in the context, say you don't know.")
    user_template = prompts_config.get("user_template", 
        "Context:\n{context}\n\nQuestion: {question}")
    
    # Fill in the user template with context and question
    user_message = user_template.format(context=context, question=query)
    
    # Return as a list of message dictionaries
    return [
        {"role": "system", "content": system_template},
        {"role": "user", "content": user_message}
    ]

# OpenAI-compatible chat completions endpoint with streaming support
@app.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest, 
    token_data: Dict = Depends(get_token_data)
):
    # Extract insurance metadata from JWT
    lob = token_data.get("lob", "general")
    state = token_data.get("state", "all")
    logger.info(f"Request for LOB: {lob}, State: {state}")
    
    # Get user query from the last user message
    user_query = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_query = msg.content
            break
    
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in the request")
    
    # Retrieve context based on query and metadata
    context = await retrieve_context(user_query, lob, state)
    
    # Format prompt with context and query
    formatted_messages = format_prompt(context, user_query)
    
    # Prepare request for vLLM
    vllm_messages = formatted_messages  # No need to transform, already in correct format
    
    vllm_payload = {
        "model": request.model,
        "messages": vllm_messages,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_tokens": request.max_tokens,
        "stream": request.stream
    }
    
    if request.stop:
        vllm_payload["stop"] = request.stop
    
    # Forward to vLLM server
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # If streaming is requested, return a streaming response
            if request.stream:
                response = await client.post(
                    f"{VLLM_HOST}/v1/chat/completions",
                    json=vllm_payload,
                    headers={"Content-Type": "application/json"},
                    timeout=None,  # No timeout for streaming
                )
                
                if response.status_code != 200:
                    logger.error(f"vLLM server error: {response.text}")
                    raise HTTPException(
                        status_code=response.status_code, 
                        detail=f"vLLM server error: {response.text}"
                    )
                
                return StreamingResponse(
                    stream_chat_response(response),
                    media_type="text/event-stream"
                )
            
            # Non-streaming request
            else:
                response = await client.post(
                    f"{VLLM_HOST}/v1/chat/completions",
                    json=vllm_payload,
                    headers={"Content-Type": "application/json"},
                )
                
                if response.status_code != 200:
                    logger.error(f"vLLM server error: {response.text}")
                    raise HTTPException(
                        status_code=response.status_code, 
                        detail=f"vLLM server error: {response.text}"
                    )
                
                vllm_response = response.json()
                
                # Transform vLLM response to OpenAI format
                return ChatCompletionResponse(
                    id=vllm_response.get("id", "chatcmpl-default"),
                    created=vllm_response.get("created", int(datetime.now().timestamp())),
                    model=vllm_response.get("model", request.model),
                    choices=[
                        ChatCompletionChoice(
                            index=i,
                            message=Message(
                                role=choice["message"]["role"],
                                content=choice["message"]["content"]
                            ),
                            finish_reason=choice.get("finish_reason", "stop")
                        )
                        for i, choice in enumerate(vllm_response.get("choices", []))
                    ],
                    usage=Usage(
                        prompt_tokens=vllm_response.get("usage", {}).get("prompt_tokens", 0),
                        completion_tokens=vllm_response.get("usage", {}).get("completion_tokens", 0),
                        total_tokens=vllm_response.get("usage", {}).get("total_tokens", 0)
                    )
                )
            
    except httpx.HTTPError as e:
        logger.error(f"Error communicating with vLLM server: {e}")
        raise HTTPException(status_code=503, detail=f"vLLM server communication error: {str(e)}")

# Direct query endpoint - simpler interface for testing
@app.post("/query")
async def query(
    request: Request,
    token_data: Dict = Depends(get_token_data)
):
    """Simple query endpoint for direct testing"""
    # Parse request body
    body = await request.json()
    question = body.get("question")
    lob = body.get("lob", "general")
    state = body.get("state", "all")
    k = body.get("k", settings.top_k)
    
    if not question:
        raise HTTPException(status_code=400, detail="Question is required")
    
    # Retrieve context
    context = await retrieve_context(question, lob, state, k)
    
    # Format prompt
    formatted_messages = format_prompt(context, question)
    
    # Prepare request for vLLM
    vllm_messages = formatted_messages  # No need to transform, already in correct format
    
    vllm_payload = {
        "model": settings.default_model,
        "messages": vllm_messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
        "stream": False
    }
    
    # Forward to vLLM server
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json=vllm_payload,
                headers={"Content-Type": "application/json"},
            )
            
            if response.status_code != 200:
                logger.error(f"vLLM server error: {response.text}")
                raise HTTPException(
                    status_code=response.status_code, 
                    detail=f"vLLM server error: {response.text}"
                )
            
            vllm_response = response.json()
            
            # Extract the answer from the response
            answer = vllm_response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Extract sources from context
            sources = []
            if context:
                context_docs = context.split("\n\n")
                for doc in context_docs:
                    if doc.strip() and "Source:" in doc:
                        source_line = doc.split("Source:")[1].strip() if "Source:" in doc else ""
                        if source_line and source_line not in sources:
                            sources.append(source_line)
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "metadata": {
                    "lob": lob,
                    "state": state,
                    "documents_retrieved": k
                }
            }
            
    except httpx.HTTPError as e:
        logger.error(f"Error communicating with vLLM server: {e}")
        raise HTTPException(status_code=503, detail=f"vLLM server communication error: {str(e)}")

# Vector DB Status Model
class VectorDBStatus(BaseModel):
    db_type: str
    version: str
    is_connected: bool
    is_compatible: bool
    collection_name: Optional[str] = None
    docs_count: Optional[int] = None
    persist_directory: Optional[str] = None
    in_memory: bool = False
    compatibility_message: Optional[str] = None

@app.get("/vector-db-status")
async def vector_db_status():
    """Check the status of the vector database including version and compatibility."""
    status = VectorDBStatus(
        db_type=app_state.db_type,
        version="unknown",
        is_connected=app_state.vectorstore is not None,
        is_compatible=False,
        in_memory=not settings.chroma_persist_directory,
        compatibility_message="Vector database status unknown"
    )
    
    # Get ChromaDB version if available
    try:
        if app_state.db_type == "chroma":
            import chromadb
            status.version = chromadb.__version__
            
            # Check compatibility
            version_parts = status.version.split('.')
            major, minor = int(version_parts[0]), int(version_parts[1])
            
            # Check if we're using the version specified in requirements.txt
            req_version = "0.4.18"  # Minimum version from requirements.txt
            req_parts = req_version.split('.')
            req_major, req_minor = int(req_parts[0]), int(req_parts[1])
            
            if major > req_major or (major == req_major and minor >= req_minor):
                status.is_compatible = True
                status.compatibility_message = f"ChromaDB version {status.version} is compatible with required version {req_version}"
            else:
                status.compatibility_message = f"ChromaDB version {status.version} may not be compatible with required version {req_version}"
            
            # Get collection details if connected
            if app_state.vectorstore:
                try:
                    status.persist_directory = settings.chroma_persist_directory or "in-memory"
                    # Try to get collection count
                    collection = app_state.vectorstore._collection
                    count = collection.count()
                    status.docs_count = count
                except Exception as e:
                    logger.warning(f"Could not get collection details: {e}")
        
        elif app_state.db_type == "qdrant":
            # For Qdrant
            try:
                import qdrant_client
                status.version = qdrant_client.__version__
                status.is_compatible = True
                status.collection_name = COLLECTION_NAME
                status.compatibility_message = f"Qdrant version {status.version} is connected"
                
                # Get collection count if connected
                if app_state.vectorstore:
                    try:
                        count = app_state.vectorstore._client.count(
                            collection_name=COLLECTION_NAME
                        ).count
                        status.docs_count = count
                    except Exception as e:
                        logger.warning(f"Could not get collection count: {e}")
            except ImportError:
                status.compatibility_message = "Qdrant client not installed"
        
    except ImportError:
        status.compatibility_message = f"Could not import {app_state.db_type} package"
    except Exception as e:
        status.compatibility_message = f"Error checking {app_state.db_type} version: {e}"
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve.main:app", host=settings.host, port=settings.fastapi_port, reload=True) 