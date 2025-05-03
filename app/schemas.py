from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="The question to answer")
    temperature: float = Field(0.1, description="Temperature for the model's output")
    top_k: int = Field(5, description="Number of documents to retrieve from vector store")
    model: Optional[str] = Field(None, description="Model to use for generation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is a retrieval-augmented generation?",
                "temperature": 0.1,
                "top_k": 5
            }
        }

class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str = Field(..., description="The answer to the question")
    sources: List[str] = Field([], description="Sources of the information used to answer the question")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Retrieval-Augmented Generation (RAG) is a technique that combines the retrieval of documents from a knowledge base with the generation capabilities of large language models. It allows models to access external information to improve factuality and accuracy of responses.",
                "sources": ["https://example.com/document1.pdf", "https://example.com/document2.pdf"]
            }
        } 