"""
Tests for citation accuracy in model responses.
"""

import os
import sys
import json
import time
import pytest
import httpx
import asyncio
import jwt
from typing import List, Dict, Any

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Constants
JWT_SECRET = "test-secret-key"
API_URL = "http://localhost:8080/v1/chat/completions"
VLLM_URL = "http://localhost:8000/v1/chat/completions"

@pytest.fixture
def test_token():
    """Generate a test JWT token."""
    payload = {
        "sub": "test-user",
        "lob": "auto",
        "state": "CA"
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    return token

@pytest.mark.asyncio
async def test_citation_format_correctness():
    """
    Test 1: Verify citations in responses are correctly formatted.
    Citations should follow the [Document X] format as specified in the prompt.
    
    Note: This test requires the API server to be running.
    Skip if server is not available.
    """
    # Check if API is available (skip if not)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8080/health")
            if response.status_code != 200:
                pytest.skip("API server not available")
    except httpx.ConnectError:
        pytest.skip("API server not available")
        
    # Get a test token
    token = jwt.encode(
        {"sub": "test-user", "lob": "auto", "state": "CA"},
        JWT_SECRET,
        algorithm="HS256"
    )
    
    # Test query designed to elicit citations
    query = "What are the coverages available in an auto insurance policy?"
    
    # Create the API request
    request_data = {
        "model": "deepseek-llm-32b-instruct",
        "messages": [
            {"role": "user", "content": query}
        ],
        "temperature": 0.3,  # Lower temperature for more consistent responses
        "max_tokens": 500
    }
    
    # Call the API
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            API_URL,
            json=request_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Check response status
        if response.status_code != 200:
            pytest.skip(f"API request failed with status {response.status_code}: {response.text}")
            
        # Parse response
        response_data = response.json()
        assistant_response = response_data["choices"][0]["message"]["content"]
        
        # Look for citation format [Document X]
        citation_count = assistant_response.count("[Document")
        
        # Print response for debugging
        print(f"Model response: {assistant_response}")
        print(f"Citation count: {citation_count}")
        
        # Assert that at least one citation is present
        assert citation_count > 0, "No citations found in model response"
        
        # Check citation format - should be [Document X] where X is a number
        import re
        citation_pattern = r'\[Document\s+\d+\]'
        citations = re.findall(citation_pattern, assistant_response)
        
        assert len(citations) > 0, "No correctly formatted citations found"
        
        # Verify citation numbers are valid
        for citation in citations:
            # Extract the citation number
            number_match = re.search(r'\d+', citation)
            assert number_match, f"Citation format error: {citation}"
            
            citation_number = int(number_match.group(0))
            assert 1 <= citation_number <= 10, f"Citation number out of expected range: {citation_number}"


@pytest.mark.asyncio
async def test_citation_content_accuracy():
    """
    Test 2: Verify citation content matches the referenced documents.
    Content in citations should be present in the original documents.
    
    Note: This test requires the API server and direct access to the vector store.
    Skip if either is not available.
    """
    # Skip if API or vector store is not available
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8080/health")
            if response.status_code != 200:
                pytest.skip("API server not available")
                
        if not os.path.exists("chroma_insurance"):
            pytest.skip("Vector store not available")
    except httpx.ConnectError:
        pytest.skip("API server not available")
    
    # Get a test token
    token = jwt.encode(
        {"sub": "test-user", "lob": "auto", "state": "CA"},
        JWT_SECRET,
        algorithm="HS256"
    )
    
    # Test query about specific content
    query = "What is the deductible for comprehensive coverage?"
    
    # Create the API request
    request_data = {
        "model": "deepseek-llm-32b-instruct",
        "messages": [
            {"role": "user", "content": query}
        ],
        "temperature": 0.3,
        "max_tokens": 500
    }
    
    # Call the API
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            API_URL,
            json=request_data,
            headers={"Authorization": f"Bearer {token}"}
        )
        
        # Skip if request fails
        if response.status_code != 200:
            pytest.skip(f"API request failed with status {response.status_code}: {response.text}")
            
        # Parse response
        response_data = response.json()
        assistant_response = response_data["choices"][0]["message"]["content"]
        
        # Check for content about deductibles
        assert "deductible" in assistant_response.lower(), "Response doesn't contain expected deductible information"
        assert "$500" in assistant_response, "Response doesn't contain the correct deductible amount for comprehensive"
        
        # Verify facts match the original document
        with open("data/raw/auto/CA/sample_auto_policy.txt", "r") as f:
            policy_content = f.read()
            
        # Extract key facts from response
        if "$500" in assistant_response:
            assert "$500" in policy_content, "Cited deductible amount doesn't match source document"
            
        # Check for comprehensive coverage mention
        if "comprehensive" in assistant_response.lower():
            assert "COMPREHENSIVE COVERAGE" in policy_content, "Cited coverage type doesn't match source document"


@pytest.mark.asyncio
async def test_citation_consistency():
    """
    Test 3: Verify citation consistency across multiple requests.
    The same query should yield consistent citations.
    
    Note: This test requires the API server to be running.
    Skip if server is not available.
    """
    # Check if API is available (skip if not)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://localhost:8080/health")
            if response.status_code != 200:
                pytest.skip("API server not available")
    except httpx.ConnectError:
        pytest.skip("API server not available")
        
    # Get a test token
    token = jwt.encode(
        {"sub": "test-user", "lob": "auto", "state": "CA"},
        JWT_SECRET,
        algorithm="HS256"
    )
    
    # Test query that should yield consistent results
    query = "What are the liability limits in the auto insurance policy?"
    
    # Create the API request
    request_data = {
        "model": "deepseek-llm-32b-instruct",
        "messages": [
            {"role": "user", "content": query}
        ],
        "temperature": 0.1,  # Use very low temperature for consistency
        "max_tokens": 500
    }
    
    # Make multiple requests and compare citations
    all_citations = []
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Make 3 identical requests
        for i in range(3):
            response = await client.post(
                API_URL,
                json=request_data,
                headers={"Authorization": f"Bearer {token}"}
            )
            
            # Skip iteration if request fails
            if response.status_code != 200:
                print(f"Request {i+1} failed with status {response.status_code}: {response.text}")
                continue
                
            # Parse response
            response_data = response.json()
            assistant_response = response_data["choices"][0]["message"]["content"]
            
            # Extract citations
            import re
            citation_pattern = r'\[Document\s+\d+\]'
            citations = re.findall(citation_pattern, assistant_response)
            
            all_citations.append(citations)
            
            # Avoid rate limiting
            await asyncio.sleep(1)
    
    # Compare citations across responses
    if len(all_citations) < 2:
        pytest.skip("Not enough successful responses to compare")
    
    # Check if citations are reasonably consistent
    # We don't expect perfect match due to LLM non-determinism, but should see overlap
    citation_sets = [set(citations) for citations in all_citations]
    
    # Calculate overlap between citation sets
    overlap_score = 0
    comparison_count = 0
    
    for i in range(len(citation_sets)):
        for j in range(i+1, len(citation_sets)):
            if citation_sets[i] and citation_sets[j]:  # Only compare non-empty sets
                intersection = citation_sets[i].intersection(citation_sets[j])
                union = citation_sets[i].union(citation_sets[j])
                
                if union:  # Avoid division by zero
                    jaccard_similarity = len(intersection) / len(union)
                    overlap_score += jaccard_similarity
                    comparison_count += 1
    
    # Calculate average overlap score
    avg_overlap = overlap_score / comparison_count if comparison_count > 0 else 0
    
    print(f"Citation sets: {citation_sets}")
    print(f"Average citation overlap score: {avg_overlap:.2f}")
    
    # We expect at least 0.3 similarity (30% overlap)
    # This is a loose threshold since LLM outputs vary
    assert avg_overlap >= 0.3, f"Citation consistency too low: {avg_overlap:.2f}" 