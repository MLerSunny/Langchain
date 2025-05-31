import logging
from typing import Dict, Any, List, Optional, Union
import evaluate
import torch
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime
import json
from pathlib import Path
from collections import deque
import time

logger = logging.getLogger(__name__)

class MetricsEvaluator:
    """Evaluates RAG responses using various metrics from HuggingFace evaluate."""
    
    def __init__(self, cache_size: int = 128):
        """Initialize the metrics evaluator with HuggingFace metrics.
        
        Args:
            cache_size: Size of the LRU cache for metric calculations
        """
        # Initialize metrics with error handling
        try:
            self.rouge = evaluate.load('rouge')
            self.bertscore = evaluate.load('bertscore')
            self.bleu = evaluate.load('bleu')
            self.meteor = evaluate.load('meteor')
            self.wer = evaluate.load('wer')
            self.perplexity = evaluate.load('perplexity')
            
            # Initialize thread pool for parallel processing
            self.executor = ThreadPoolExecutor(max_workers=4)
            
            # Thread-local storage for device
            self._thread_local = threading.local()
            
        except Exception as e:
            logger.error(f"Error initializing metrics: {e}")
            raise RuntimeError(f"Failed to initialize metrics: {e}")
    
    @property
    def device(self) -> str:
        """Get the current device for the thread."""
        if not hasattr(self._thread_local, 'device'):
            self._thread_local.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self._thread_local.device
    
    @lru_cache(maxsize=128)
    def calculate_rouge_scores(self, reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores between reference and candidate text.
        
        Args:
            reference: Reference text
            candidate: Candidate text to evaluate
            
        Returns:
            Dictionary of ROUGE scores
        """
        try:
            scores = self.rouge.compute(
                predictions=[candidate],
                references=[reference],
                use_stemmer=True
            )
            return {
                'rouge1': scores['rouge1'],
                'rouge2': scores['rouge2'],
                'rougeL': scores['rougeL'],
                'rougeLsum': scores['rougeLsum']
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0,
                'rougeLsum': 0.0
            }
    
    @lru_cache(maxsize=128)
    def calculate_bert_score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """
        Calculate BERT score between reference and candidate texts.
        
        Args:
            references: List of reference texts
            candidates: List of candidate texts to evaluate
            
        Returns:
            Dictionary of BERT score metrics
        """
        try:
            scores = self.bertscore.compute(
                predictions=candidates,
                references=references,
                lang="en",
                rescale_with_baseline=True,
                device=self.device
            )
            
            return {
                'bert_precision': float(np.mean(scores['precision'])),
                'bert_recall': float(np.mean(scores['recall'])),
                'bert_f1': float(np.mean(scores['f1']))
            }
        except Exception as e:
            logger.error(f"Error calculating BERT score: {e}")
            return {
                'bert_precision': 0.0,
                'bert_recall': 0.0,
                'bert_f1': 0.0
            }
    
    @lru_cache(maxsize=128)
    def calculate_bleu_score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Calculate BLEU score."""
        try:
            scores = self.bleu.compute(
                predictions=candidates,
                references=[[ref] for ref in references]
            )
            return {'bleu': scores['bleu']}
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return {'bleu': 0.0}
    
    @lru_cache(maxsize=128)
    def calculate_meteor_score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Calculate METEOR score."""
        try:
            scores = self.meteor.compute(
                predictions=candidates,
                references=references
            )
            return {'meteor': scores['meteor']}
        except Exception as e:
            logger.error(f"Error calculating METEOR score: {e}")
            return {'meteor': 0.0}
    
    @lru_cache(maxsize=128)
    def calculate_wer_score(self, references: List[str], candidates: List[str]) -> Dict[str, float]:
        """Calculate Word Error Rate."""
        try:
            scores = self.wer.compute(
                predictions=candidates,
                references=references
            )
            return {'wer': scores['wer']}
        except Exception as e:
            logger.error(f"Error calculating WER score: {e}")
            return {'wer': 0.0}
    
    @lru_cache(maxsize=128)
    def calculate_perplexity(self, texts: List[str]) -> Dict[str, float]:
        """Calculate perplexity score."""
        try:
            scores = self.perplexity.compute(
                predictions=texts,
                model_id='gpt2'
            )
            return {'perplexity': scores['perplexity']}
        except Exception as e:
            logger.error(f"Error calculating perplexity: {e}")
            return {'perplexity': 0.0}
    
    def evaluate_response(
        self,
        reference: str,
        candidate: str,
        calculate_all: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a response using all available metrics.
        
        Args:
            reference: Reference text
            candidate: Candidate text to evaluate
            calculate_all: Whether to calculate all metrics
            
        Returns:
            Dictionary of all metric scores
        """
        try:
            # Calculate ROUGE scores
            rouge_scores = self.calculate_rouge_scores(reference, candidate)
            
            # Calculate other metrics if requested
            other_scores = {}
            if calculate_all:
                # Use thread pool for parallel computation
                futures = []
                futures.append(self.executor.submit(self.calculate_bert_score, [reference], [candidate]))
                futures.append(self.executor.submit(self.calculate_bleu_score, [reference], [candidate]))
                futures.append(self.executor.submit(self.calculate_meteor_score, [reference], [candidate]))
                futures.append(self.executor.submit(self.calculate_wer_score, [reference], [candidate]))
                futures.append(self.executor.submit(self.calculate_perplexity, [candidate]))
                
                # Collect results
                for future in futures:
                    other_scores.update(future.result())
            
            # Combine all scores
            return {**rouge_scores, **other_scores}
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {
                'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0,
                'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0,
                'bleu': 0.0, 'meteor': 0.0, 'wer': 0.0,
                'perplexity': 0.0
            }
    
    def evaluate_batch(
        self,
        references: List[str],
        candidates: List[str],
        calculate_all: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate a batch of responses using all available metrics.
        
        Args:
            references: List of reference texts
            candidates: List of candidate texts to evaluate
            calculate_all: Whether to calculate all metrics
            
        Returns:
            Dictionary of average metric scores
        """
        try:
            # Calculate ROUGE scores
            rouge_scores = self.rouge.compute(
                predictions=candidates,
                references=references,
                use_stemmer=True
            )
            
            # Calculate other metrics if requested
            other_scores = {}
            if calculate_all:
                # Use thread pool for parallel computation
                futures = []
                futures.append(self.executor.submit(self.calculate_bert_score, references, candidates))
                futures.append(self.executor.submit(self.calculate_bleu_score, references, candidates))
                futures.append(self.executor.submit(self.calculate_meteor_score, references, candidates))
                futures.append(self.executor.submit(self.calculate_wer_score, references, candidates))
                futures.append(self.executor.submit(self.calculate_perplexity, candidates))
                
                # Collect results
                for future in futures:
                    other_scores.update(future.result())
            
            # Combine all scores
            return {**rouge_scores, **other_scores}
            
        except Exception as e:
            logger.error(f"Error evaluating batch: {e}")
            return {
                'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0, 'rougeLsum': 0.0,
                'bert_precision': 0.0, 'bert_recall': 0.0, 'bert_f1': 0.0,
                'bleu': 0.0, 'meteor': 0.0, 'wer': 0.0,
                'perplexity': 0.0
            }
    
    def __del__(self):
        """Cleanup resources."""
        try:
            self.executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Error shutting down executor: {e}")

class MetricsCollector:
    def __init__(self, history_size: int = 1000):
        """Initialize metrics collector."""
        self.history_size = history_size
        
        # Initialize metrics storage
        self.retrieval_latencies = deque(maxlen=history_size)
        self.generation_latencies = deque(maxlen=history_size)
        self.total_latencies = deque(maxlen=history_size)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize evaluation metrics
        self.rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'rougeLsum': []
        }
        self.bert_scores = {
            'bert_precision': [],
            'bert_recall': [],
            'bert_f1': []
        }
        self.bleu_scores = []
        self.meteor_scores = []
        self.wer_scores = []
        self.perplexity_scores = []
        
        # Initialize error history
        self.error_history = []
        
        # Load existing metrics if available
        self._load_metrics()
        
        self.query_history: List[Dict[str, Any]] = []
        self.start_time = datetime.now()
    
    def update_metrics(
        self,
        retrieval_latency: float,
        generation_latency: float,
        total_latency: float,
        cache_hit: bool = False
    ):
        """Update metrics with new values."""
        try:
            # Update latencies
            self.retrieval_latencies.append(retrieval_latency)
            self.generation_latencies.append(generation_latency)
            self.total_latencies.append(total_latency)
            
            # Update cache stats
            if cache_hit:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            
            # Save metrics
            self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
            raise
    
    def update_evaluation_metrics(
        self,
        rouge_scores: Dict[str, float],
        bert_scores: Dict[str, float],
        bleu_score: float,
        meteor_score: float,
        wer_score: float,
        perplexity: float
    ):
        """Update evaluation metrics."""
        try:
            # Update ROUGE scores
            for metric, score in rouge_scores.items():
                self.rouge_scores[metric].append(score)
            
            # Update BERT scores
            for metric, score in bert_scores.items():
                self.bert_scores[metric].append(score)
            
            # Update other scores
            self.bleu_scores.append(bleu_score)
            self.meteor_scores.append(meteor_score)
            self.wer_scores.append(wer_score)
            self.perplexity_scores.append(perplexity)
            
            # Save metrics
            self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error updating evaluation metrics: {e}")
            raise
    
    def add_error(self, error: str, context: Dict[str, Any]):
        """Add error to history."""
        try:
            self.error_history.append({
                'timestamp': datetime.now().isoformat(),
                'error': error,
                'context': context
            })
            
            # Save metrics
            self._save_metrics()
            
        except Exception as e:
            logger.error(f"Error adding error to history: {e}")
            raise
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        try:
            # Calculate averages
            avg_retrieval_latency = np.mean(self.retrieval_latencies) if self.retrieval_latencies else 0
            avg_generation_latency = np.mean(self.generation_latencies) if self.generation_latencies else 0
            avg_total_latency = np.mean(self.total_latencies) if self.total_latencies else 0
            
            # Calculate cache hit rate
            total_cache = self.cache_hits + self.cache_misses
            cache_hit_rate = self.cache_hits / total_cache if total_cache > 0 else 0
            
            # Calculate average evaluation metrics
            avg_rouge_scores = {
                metric: np.mean(scores) if scores else 0
                for metric, scores in self.rouge_scores.items()
            }
            
            avg_bert_scores = {
                metric: np.mean(scores) if scores else 0
                for metric, scores in self.bert_scores.items()
            }
            
            return {
                'retrieval_latency': avg_retrieval_latency,
                'generation_latency': avg_generation_latency,
                'total_latency': avg_total_latency,
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'cache_hit_rate': cache_hit_rate,
                'rouge_scores': avg_rouge_scores,
                'bert_scores': avg_bert_scores,
                'bleu_score': np.mean(self.bleu_scores) if self.bleu_scores else 0,
                'meteor_score': np.mean(self.meteor_scores) if self.meteor_scores else 0,
                'wer_score': np.mean(self.wer_scores) if self.wer_scores else 0,
                'perplexity': np.mean(self.perplexity_scores) if self.perplexity_scores else 0,
                'error_history': self.error_history[-10:] if self.error_history else [],
                'evaluation_history': self._get_evaluation_history()
            }
            
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            raise
    
    def _get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get evaluation history."""
        try:
            history = []
            for i in range(len(self.rouge_scores['rouge1'])):
                entry = {
                    'timestamp': datetime.now().isoformat(),
                    'rouge_scores': {
                        metric: scores[i] if i < len(scores) else 0
                        for metric, scores in self.rouge_scores.items()
                    },
                    'bert_scores': {
                        metric: scores[i] if i < len(scores) else 0
                        for metric, scores in self.bert_scores.items()
                    },
                    'bleu_score': self.bleu_scores[i] if i < len(self.bleu_scores) else 0,
                    'meteor_score': self.meteor_scores[i] if i < len(self.meteor_scores) else 0,
                    'wer_score': self.wer_scores[i] if i < len(self.wer_scores) else 0,
                    'perplexity': self.perplexity_scores[i] if i < len(self.perplexity_scores) else 0
                }
                history.append(entry)
            return history
            
        except Exception as e:
            logger.error(f"Error getting evaluation history: {e}")
            raise
    
    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            metrics_dir = Path('data/metrics')
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = metrics_dir / 'metrics.json'
            metrics = {
                'retrieval_latencies': list(self.retrieval_latencies),
                'generation_latencies': list(self.generation_latencies),
                'total_latencies': list(self.total_latencies),
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'rouge_scores': self.rouge_scores,
                'bert_scores': self.bert_scores,
                'bleu_scores': self.bleu_scores,
                'meteor_scores': self.meteor_scores,
                'wer_scores': self.wer_scores,
                'perplexity_scores': self.perplexity_scores,
                'error_history': self.error_history
            }
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f)
                
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
            raise
    
    def _load_metrics(self):
        """Load metrics from disk."""
        try:
            metrics_file = Path('data/metrics/metrics.json')
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                self.retrieval_latencies = deque(metrics.get('retrieval_latencies', []), maxlen=self.history_size)
                self.generation_latencies = deque(metrics.get('generation_latencies', []), maxlen=self.history_size)
                self.total_latencies = deque(metrics.get('total_latencies', []), maxlen=self.history_size)
                self.cache_hits = metrics.get('cache_hits', 0)
                self.cache_misses = metrics.get('cache_misses', 0)
                self.rouge_scores = metrics.get('rouge_scores', {})
                self.bert_scores = metrics.get('bert_scores', {})
                self.bleu_scores = metrics.get('bleu_scores', [])
                self.meteor_scores = metrics.get('meteor_scores', [])
                self.wer_scores = metrics.get('wer_scores', [])
                self.perplexity_scores = metrics.get('perplexity_scores', [])
                self.error_history = metrics.get('error_history', [])
                
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            raise

    def record_query(self, query: str, processing_time: float):
        """Record a query and its processing time."""
        try:
            self.query_history.append({
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "processing_time": processing_time
            })
        except Exception as e:
            logger.error(f"Error recording query metrics: {str(e)}")
            
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        try:
            total_queries = len(self.query_history)
            avg_processing_time = (
                sum(q["processing_time"] for q in self.query_history) / total_queries
                if total_queries > 0 else 0
            )
            
            return {
                "total_queries": total_queries,
                "average_processing_time": avg_processing_time,
                "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
                "recent_queries": self.query_history[-10:] if self.query_history else []
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {
                "error": str(e),
                "total_queries": 0,
                "average_processing_time": 0,
                "uptime_seconds": 0,
                "recent_queries": []
            } 