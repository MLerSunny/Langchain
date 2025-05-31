import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from datetime import datetime
import logging

from core.settings import settings, FASTAPI_PORT

logger = logging.getLogger(__name__)

def format_metric(val, decimals=3):
    if val is None or (isinstance(val, (int, float)) and val == 0):
        return "—"
    return f"{val:.{decimals}f}"

def display_metrics_tab():
    """Display metrics tab."""
    st.markdown("## 📊 <b>Metrics</b>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Add a button to refresh metrics
    if st.button("🔄 Run Metrics", help="Fetch the latest metrics from the server"):
        st.session_state['metrics_refresh'] = True

    # Only fetch metrics if the button is pressed or on first load
    if st.session_state.get('metrics_refresh', True):
        with st.spinner("Fetching latest metrics..."):
            try:
                response = requests.get(f"http://localhost:{FASTAPI_PORT}/metrics")
                st.session_state['metrics_data'] = response.json() if response.status_code == 200 else None
            except Exception as e:
                st.session_state['metrics_data'] = None
        st.session_state['metrics_refresh'] = False

    metrics = st.session_state.get('metrics_data', None)
    if metrics is None:
        st.error("Failed to fetch metrics from API")
        return

    # Display basic metrics
    st.markdown("### Basic Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval Latency", format_metric(metrics.get('retrieval_latency'), 2) + "s")
    col2.metric("Generation Latency", format_metric(metrics.get('generation_latency'), 2) + "s")
    total_latency = (metrics.get('retrieval_latency', 0) or 0) + (metrics.get('generation_latency', 0) or 0)
    col3.metric("Total Latency", format_metric(total_latency, 2) + "s")
    
    col1, col2, col3 = st.columns(3)
    cache_hits = metrics.get('cache_hits', 0)
    cache_misses = metrics.get('cache_misses', 0)
    total_cache = cache_hits + cache_misses
    col1.metric("Cache Hits", cache_hits if cache_hits else "—")
    col2.metric("Cache Misses", cache_misses if cache_misses else "—")
    col3.metric("Cache Hit Rate", f"{(cache_hits / total_cache * 100):.1f}%" if total_cache > 0 else "—")
    
    # Display ROUGE scores
    st.markdown("### ROUGE Scores")
    rouge_scores = metrics.get('rouge_scores', {})
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROUGE-1", format_metric(rouge_scores.get('rouge1')))
    col2.metric("ROUGE-2", format_metric(rouge_scores.get('rouge2')))
    col3.metric("ROUGE-L", format_metric(rouge_scores.get('rougeL')))
    col4.metric("ROUGE-Lsum", format_metric(rouge_scores.get('rougeLsum')))
    
    # Display BERT scores
    st.markdown("### BERT Scores")
    bert_scores = metrics.get('bert_scores', {})
    col1, col2, col3 = st.columns(3)
    col1.metric("BERT Precision", format_metric(bert_scores.get('bert_precision')))
    col2.metric("BERT Recall", format_metric(bert_scores.get('bert_recall')))
    col3.metric("BERT F1", format_metric(bert_scores.get('bert_f1')))
    
    # Display additional metrics
    st.markdown("### Additional Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("BLEU Score", format_metric(metrics.get('bleu_score')))
    col2.metric("METEOR Score", format_metric(metrics.get('meteor_score')))
    col3.metric("WER Score", format_metric(metrics.get('wer_score')))
    
    col1, col2 = st.columns(2)
    col1.metric("Perplexity", format_metric(metrics.get('perplexity')))
    col2.metric("Semantic Similarity", format_metric(metrics.get('semantic_similarity')))
    
    # Display historical metrics
    st.markdown("### Historical Metrics")
    
    # Create DataFrame for historical data
    history_data = []
    for entry in metrics.get('evaluation_history', []):
        history_data.append({
            'timestamp': entry.get('timestamp'),
            'rouge1': entry.get('rouge_scores', {}).get('rouge1', 0),
            'rouge2': entry.get('rouge_scores', {}).get('rouge2', 0),
            'rougeL': entry.get('rouge_scores', {}).get('rougeL', 0),
            'rougeLsum': entry.get('rouge_scores', {}).get('rougeLsum', 0),
            'bert_precision': entry.get('bert_scores', {}).get('bert_precision', 0),
            'bert_recall': entry.get('bert_scores', {}).get('bert_recall', 0),
            'bert_f1': entry.get('bert_scores', {}).get('bert_f1', 0),
            'bleu_score': entry.get('bleu_score', 0),
            'meteor_score': entry.get('meteor_score', 0),
            'wer_score': entry.get('wer_score', 0),
            'perplexity': entry.get('perplexity', 0),
            'semantic_similarity': entry.get('semantic_similarity', 0)
        })
    
    if history_data:
        df = pd.DataFrame(history_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Plot ROUGE scores over time
        fig_rouge = px.line(
            df,
            x='timestamp',
            y=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
            title='ROUGE Scores Over Time',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig_rouge, use_container_width=True)
        
        # Plot BERT scores over time
        fig_bert = px.line(
            df,
            x='timestamp',
            y=['bert_precision', 'bert_recall', 'bert_f1'],
            title='BERT Scores Over Time',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig_bert, use_container_width=True)
        
        # Plot additional metrics over time
        fig_additional = px.line(
            df,
            x='timestamp',
            y=['bleu_score', 'meteor_score', 'wer_score', 'perplexity', 'semantic_similarity'],
            title='Additional Metrics Over Time',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig_additional, use_container_width=True)
    else:
        st.markdown("🛈 <b>No historical metrics data available yet.</b>", unsafe_allow_html=True)
    
    # Display error history
    if metrics.get('error_history'):
        st.markdown("### Error History")
        error_df = pd.DataFrame(metrics['error_history'])
        st.dataframe(error_df, use_container_width=True)
    else:
        st.markdown("🛈 <b>No error history available.</b>", unsafe_allow_html=True) 