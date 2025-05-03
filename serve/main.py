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
from typing import List, Dict, Any, Optional, Union, AsyncGenerator
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from langchain_community.vectorstores import Chroma, Qdrant
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

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

# Initialize embedding model and vector store
@app.on_event("startup")
async def startup_db_client():
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
                app_state.vectorstore = Chroma(
                    persist_directory=settings.chroma_persist_directory,
                    embedding_function=app_state.embeddings,
                )
                app_state.db_type = "chroma"
                logger.info(f"Connected to Chroma at {settings.chroma_persist_directory}")
            else:
                logger.error(f"Vector database not found! Please ingest documents first.")
                app_state.vectorstore = None
    except ImportError:
        logger.warning("Qdrant client not installed, using Chroma")
        # Use Chroma
        if os.path.exists(settings.chroma_persist_directory):
            app_state.vectorstore = Chroma(
                persist_directory=settings.chroma_persist_directory,
                embedding_function=app_state.embeddings,
            )
            app_state.db_type = "chroma" 
            logger.info(f"Connected to Chroma at {settings.chroma_persist_directory}")
        else:
            logger.error(f"Chroma directory {settings.chroma_persist_directory} not found!")
            app_state.vectorstore = None
    except Exception as e:
        logger.error(f"Error connecting to Qdrant: {e}")
        # Fallback to Chroma
        if os.path.exists(settings.chroma_persist_directory):
            app_state.vectorstore = Chroma(
                persist_directory=settings.chroma_persist_directory,
                embedding_function=app_state.embeddings,
            )
            app_state.db_type = "chroma"
            logger.info(f"Connected to Chroma at {settings.chroma_persist_directory}")
        else:
            logger.error(f"Vector database not found! Please ingest documents first.")
            app_state.vectorstore = None

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
    if lob:
        metadata_filters["lob"] = lob
    if state:
        metadata_filters["state"] = state

    # Get retriever parameters from config
    search_type = retriever_config.get("search_type", "mmr")
    lambda_mult = retriever_config.get("lambda_mult", 0.3)
    
    # Create retriever with metadata filtering
    if search_type == "mmr":
        retriever = app_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "lambda_mult": lambda_mult,
                "filter": metadata_filters
            }
        )
    else:
        retriever = app_state.vectorstore.as_retriever(
            search_kwargs={
                "k": k,
                "filter": metadata_filters
            }
        )
    
    # Get documents
    docs = retriever.get_relevant_documents(query)
    
    # Format context
    formatted_context = ""
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        lob = doc.metadata.get("lob", "unknown")
        state = doc.metadata.get("state", "unknown")
        formatted_context += f"[Document {i+1}] Source: {source}, LOB: {lob}, State: {state}\n"
        formatted_context += f"{doc.page_content}\n\n"
    
    return formatted_context

# Async wrapper for the cached retrieval function
async def retrieve_context(query: str, lob: str = None, state: str = None, k: int = None):
    """
    Retrieve relevant context from the vector store.
    
    Args:
        query: The user's question
        lob: Line of business filter
        state: State code filter
        k: Number of documents to retrieve
        
    Returns:
        List of retrieved documents and their content as a string
    """
    k = k or settings.top_k
    
    # Use a thread pool to run the sync retrieval function
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, retrieve_docs_cached, query, lob or "general", state or "all", k
    )

# Format prompt with context and templates
def format_prompt(context: str, query: str):
    """
    Format the prompt for the LLM.
    
    Args:
        context: Retrieved context
        query: User question
        
    Returns:
        Formatted prompt
    """
    system_template = prompts_config.get("system_template", 
        "You are an insurance expert assistant. Answer the question based strictly on the provided context. If the information isn't in the context, say you don't know.")
    user_template = prompts_config.get("user_template", 
        "Context:\n{context}\n\nQuestion: {question}")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("user", user_template)
    ])
    
    formatted_prompt = prompt.format_messages(context=context, question=query)
    return formatted_prompt

# Helper function to stream responses from vLLM
async def stream_chat_response(response: httpx.Response):
    """Stream the chat response from vLLM server."""
    async for line in response.aiter_lines():
        if line.strip():
            if line.startswith("data:"):
                line = line[5:].strip()
            if line == "[DONE]":
                break
            try:
                yield f"data: {line}\n\n"
            except Exception as e:
                logger.error(f"Error streaming response: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"

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
    vllm_messages = [{"role": msg.type, "content": msg.content} for msg in formatted_messages]
    
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
    question: str,
    lob: str = "general", 
    state: str = "all",
    k: int = None,
    token_data: Dict = Depends(get_token_data)
):
    """Simple query endpoint for direct testing"""
    # Retrieve context
    context = await retrieve_context(question, lob, state, k or settings.top_k)
    
    # Format prompt
    formatted_messages = format_prompt(context, question)
    
    # Prepare request for vLLM
    vllm_messages = [{"role": msg.type, "content": msg.content} for msg in formatted_messages]
    
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
            
            return {
                "question": question,
                "answer": answer,
                "metadata": {
                    "lob": lob,
                    "state": state
                }
            }
            
    except httpx.HTTPError as e:
        logger.error(f"Error communicating with vLLM server: {e}")
        raise HTTPException(status_code=503, detail=f"vLLM server communication error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.fastapi_port, reload=True) 