"""
Tests for the FastAPI application.

This module contains tests for the FastAPI application endpoints.
"""

import os
import sys
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from langchain.schema import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import app
from app.proxy import OllamaProxyClient


# Create a test client
client = TestClient(app)


def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"


@mock.patch.object(OllamaProxyClient, "list_models")
async def test_list_models(mock_list_models):
    """Test the list models endpoint."""
    # Mock the list_models method
    mock_list_models.return_value = [
        {"name": "deepseek-coder:7b-instruct-v1.5"},
        {"name": "llama3:8b"},
    ]
    
    # Initialize the model client in the app state
    app.state.model_client = OllamaProxyClient()
    
    # Make the request
    response = client.get("/models")
    
    # Check the response
    assert response.status_code == 200
    assert "models" in response.json()
    assert len(response.json()["models"]) == 2


@mock.patch.object(OllamaProxyClient, "generate")
async def test_query_endpoint(mock_generate):
    """Test the query endpoint."""
    # Mock the generate method
    mock_generate.return_value = {
        "text": "This is a test response",
        "model": "deepseek-coder:7b-instruct-v1.5",
        "total_tokens": 10,
    }
    
    # Initialize the model client in the app state
    app.state.model_client = OllamaProxyClient()
    
    # Create a test vector store with mock documents
    class MockVectorStore:
        def similarity_search_with_score(self, query, k=3):
            return [
                (
                    Document(
                        page_content="Test document content",
                        metadata={"source": "test.txt"},
                    ),
                    0.95,
                )
            ]
    
    # Set the mock vector store in the app state
    app.state.vectorstore = MockVectorStore()
    
    # Make the request
    response = client.post(
        "/query",
        json={
            "query": "What is a test?",
            "model": "deepseek-coder:7b-instruct-v1.5",
            "temperature": 0.1,
            "max_tokens": 100,
            "use_rag": True,
            "k": 1,
        },
    )
    
    # Check the response
    assert response.status_code == 200
    assert "answer" in response.json()
    assert response.json()["answer"] == "This is a test response"
    assert "sources" in response.json()
    assert len(response.json()["sources"]) == 1 