from fastapi import APIRouter, Request, Depends, HTTPException
from typing import Dict, Any
import logging
from datetime import datetime

from core.auth import validate_token

router = APIRouter(tags=["Metrics"])
logger = logging.getLogger(__name__)

@router.get("/metrics")
async def get_metrics(request: Request, user: dict = Depends(validate_token)):
    """Get current metrics from the RAG engine."""
    try:
        # Get RAG engine instance
        rag_engine = request.app.state.rag_engine
        
        # Get current metrics
        metrics = {
            "retrieval_latency": rag_engine.metrics.retrieval_latency,
            "generation_latency": rag_engine.metrics.generation_latency,
            "cache_hits": rag_engine.metrics.cache_hits,
            "cache_misses": rag_engine.metrics.cache_misses,
            "token_usage": rag_engine.metrics.token_usage,
            "rouge_scores": rag_engine.metrics.rouge_scores,
            "bert_scores": rag_engine.metrics.bert_scores,
            "bleu_score": rag_engine.metrics.bleu_score,
            "meteor_score": rag_engine.metrics.meteor_score,
            "wer_score": rag_engine.metrics.wer_score,
            "perplexity": rag_engine.metrics.perplexity,
            "semantic_similarity": rag_engine.metrics.semantic_similarity,
            "evaluation_history": rag_engine.metrics.evaluation_history,
            "error_history": rag_engine.metrics.error_history,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat(),
                "user": user.get("sub", "unknown")
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) 