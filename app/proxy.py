"""
vLLM proxy implementation for Ollama models.

This module implements a client for interfacing with models via Ollama API
and a proxy for vLLM integration.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional

import aiohttp
from langchain.schema import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.settings import (
    OLLAMA_BASE_URL,
    VLLM_PORT,
    DEFAULT_MODEL,
    CONTEXT_WINDOW,
    MAX_TOKENS,
    TEMPERATURE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class OllamaProxyClient:
    """
    Client for interacting with Ollama models via the Ollama API.
    Also implements a proxy to vLLM when available.
    """

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        """
        Initialize the client.

        Args:
            base_url: Base URL for the Ollama API.
        """
        self.base_url = base_url
        self.vllm_url = f"http://localhost:{VLLM_PORT}/v1/completions"
        
    async def _create_session(self) -> aiohttp.ClientSession:
        """
        Create an aiohttp client session.

        Returns:
            An aiohttp client session.
        """
        return aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600),
            connector=aiohttp.TCPConnector(limit=10),
        )

    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models from Ollama.

        Returns:
            List of available models.
        """
        try:
            async with await self._create_session() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("models", [])
                    else:
                        error_text = await response.text()
                        logger.error(f"Error listing models: {error_text}")
                        raise Exception(f"Failed to list models: {error_text}")
        except Exception as e:
            logger.error(f"Error connecting to Ollama API: {e}")
            raise Exception(f"Failed to connect to Ollama API: {e}")

    async def _try_vllm(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> Optional[Dict[str, Any]]:
        """
        Try to use vLLM for generation if available.

        Args:
            prompt: The prompt to send to the model.
            model: The model to use.
            temperature: The temperature to use for generation.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            The response from vLLM, or None if vLLM is not available.
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            
            async with await self._create_session() as session:
                async with session.post(
                    self.vllm_url, json=payload, timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extract the generated text
                        text = result.get("choices", [{}])[0].get("text", "")
                        
                        # Return in a format compatible with our API
                        return {
                            "text": text,
                            "model": model,
                            "total_tokens": result.get("usage", {}).get("total_tokens", 0),
                        }
                    else:
                        # vLLM not available or error
                        return None
        except Exception as e:
            logger.debug(f"vLLM not available or error: {e}")
            return None

    async def generate(
        self,
        query: str,
        context_docs: Optional[List[Document]] = None,
        model: str = DEFAULT_MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
    ) -> Dict[str, Any]:
        """
        Generate a response to a query.

        Args:
            query: The query to send to the model.
            context_docs: Optional context documents for RAG.
            model: The model to use.
            temperature: The temperature to use for generation.
            max_tokens: The maximum number of tokens to generate.

        Returns:
            The response from the model.
        """
        # Prepare the context from the retrieved documents
        context = ""
        if context_docs:
            context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Prepare the prompt
        if context:
            prompt = f"""You are a helpful AI assistant. Use the following information to answer the question. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context:
            {context}

            Question: {query}

            Answer:"""
        else:
            prompt = f"""You are a helpful AI assistant. Answer the following question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Question: {query}

            Answer:"""
        
        # Try vLLM first
        vllm_response = await self._try_vllm(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if vllm_response:
            logger.info("Using vLLM for generation")
            return vllm_response
        
        # Fall back to Ollama API
        logger.info("Falling back to Ollama API for generation")
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "temperature": temperature,
                "num_predict": max_tokens,
                "stream": False,
            }
            
            async with await self._create_session() as session:
                async with session.post(
                    f"{self.base_url}/api/generate", json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Return in a format compatible with our API
                        return {
                            "text": result.get("response", ""),
                            "model": model,
                            "total_tokens": result.get("eval_count", 0),
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Error generating response: {error_text}")
                        raise Exception(f"Failed to generate response: {error_text}")
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise Exception(f"Failed to generate response: {e}")


async def test_client():
    """Test the client."""
    client = OllamaProxyClient()
    
    # List models
    models = await client.list_models()
    logger.info("Available models: %s", models)
    
    # Test generation
    response = await client.generate(
        query="What is the capital of France?",
        model=DEFAULT_MODEL,
    )
    logger.debug("LLM response: %s", response)


if __name__ == "__main__":
    asyncio.run(test_client()) 