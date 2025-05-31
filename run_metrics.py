import requests
import json
from pprint import pprint
from core.settings import FASTAPI_PORT

def fetch_metrics():
    """Fetch metrics from the FastAPI server."""
    try:
        response = requests.get(f"http://localhost:{FASTAPI_PORT}/metrics")
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching metrics: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error connecting to metrics server: {e}")
        return None

def display_metrics(metrics):
    """Display metrics in a readable format."""
    if not metrics:
        print("No metrics available")
        return

    print("\n=== Basic Metrics ===")
    print(f"Retrieval Latency: {metrics.get('retrieval_latency', 0):.2f}s")
    print(f"Generation Latency: {metrics.get('generation_latency', 0):.2f}s")
    total_latency = (metrics.get('retrieval_latency', 0) or 0) + (metrics.get('generation_latency', 0) or 0)
    print(f"Total Latency: {total_latency:.2f}s")
    
    cache_hits = metrics.get('cache_hits', 0)
    cache_misses = metrics.get('cache_misses', 0)
    total_cache = cache_hits + cache_misses
    print(f"\nCache Hits: {cache_hits}")
    print(f"Cache Misses: {cache_misses}")
    if total_cache > 0:
        print(f"Cache Hit Rate: {(cache_hits / total_cache * 100):.1f}%")

    print("\n=== ROUGE Scores ===")
    rouge_scores = metrics.get('rouge_scores', {})
    print(f"ROUGE-1: {rouge_scores.get('rouge1', 0):.3f}")
    print(f"ROUGE-2: {rouge_scores.get('rouge2', 0):.3f}")
    print(f"ROUGE-L: {rouge_scores.get('rougeL', 0):.3f}")
    print(f"ROUGE-Lsum: {rouge_scores.get('rougeLsum', 0):.3f}")

    print("\n=== BERT Scores ===")
    bert_scores = metrics.get('bert_scores', {})
    print(f"BERT Precision: {bert_scores.get('bert_precision', 0):.3f}")
    print(f"BERT Recall: {bert_scores.get('bert_recall', 0):.3f}")
    print(f"BERT F1: {bert_scores.get('bert_f1', 0):.3f}")

    print("\n=== Additional Metrics ===")
    print(f"BLEU Score: {metrics.get('bleu_score', 0):.3f}")
    print(f"METEOR Score: {metrics.get('meteor_score', 0):.3f}")
    print(f"WER Score: {metrics.get('wer_score', 0):.3f}")
    print(f"Perplexity: {metrics.get('perplexity', 0):.3f}")
    print(f"Semantic Similarity: {metrics.get('semantic_similarity', 0):.3f}")

    if metrics.get('error_history'):
        print("\n=== Recent Errors ===")
        for error in metrics['error_history'][-5:]:  # Show last 5 errors
            print(f"Time: {error.get('timestamp')}")
            print(f"Type: {error.get('type')}")
            print(f"Message: {error.get('message')}")
            print("---")

if __name__ == "__main__":
    print("Fetching metrics...")
    metrics = fetch_metrics()
    display_metrics(metrics) 