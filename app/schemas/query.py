from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from .common import BaseResponse

class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(..., min_length=1, description="The question to be answered")
    reference_answer: Optional[str] = Field(None, description="Optional reference answer for evaluation")
    stream: bool = Field(default=False, description="Whether to stream the response")
    max_tokens: Optional[int] = Field(None, ge=1, description="Maximum number of tokens in the response")
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    context_window: Optional[int] = Field(None, ge=1, description="Size of the context window")

class Source(BaseModel):
    """Source document information."""
    document_id: str = Field(..., description="ID of the source document")
    content: str = Field(..., description="Content snippet from the source")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Source metadata")
    score: float = Field(..., description="Relevance score of the source")

class Metrics(BaseModel):
    """Query metrics."""
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    token_usage: int = 0
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    query_success_rate: float = 0.0
    response_quality: float = 0.0
    rouge_scores: Dict[str, float] = Field(default_factory=lambda: {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'rougeLsum': 0.0
    })
    bert_scores: Dict[str, float] = Field(default_factory=lambda: {
        'bert_precision': 0.0,
        'bert_recall': 0.0,
        'bert_f1': 0.0
    })
    bleu_score: float = 0.0
    meteor_score: float = 0.0
    wer_score: float = 0.0
    perplexity: float = 0.0
    semantic_similarity: float = 0.0

class QueryResponse(BaseResponse):
    """Query response model."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(default_factory=list, description="Source documents used")
    metrics: Metrics = Field(..., description="Query performance metrics")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the query"
    )

class StreamChunk(BaseModel):
    """Streaming response chunk."""
    chunk: str = Field(..., description="Text chunk of the response")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")
    sources: Optional[List[Source]] = Field(None, description="Source documents used")
    metrics: Optional[Metrics] = Field(None, description="Query performance metrics") 