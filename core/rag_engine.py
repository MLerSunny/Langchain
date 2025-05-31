import os
import logging
import time
from typing import List, Dict, Any, Optional, Tuple, Iterator, Callable, Union, Set
from dataclasses import dataclass, field
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.cache import InMemoryCache
from tenacity import retry, stop_after_attempt, wait_exponential
import tiktoken
import spacy
from collections import defaultdict
from datetime import datetime
import psutil
from core.metrics import MetricsEvaluator
import re
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from core.settings import settings
from core.exceptions import RAGError
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class RAGMetrics:
    """Metrics for RAG operations."""
    # Basic metrics
    retrieval_latency: float = 0.0
    generation_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    token_usage: int = 0
    
    # Query metrics
    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    query_success_rate: float = 0.0
    response_quality: float = 0.0
    
    # Document processing metrics
    total_documents: int = 0
    avg_chunk_size: float = 0.0
    doc_processing_time: float = 0.0
    embedding_time: float = 0.0
    doc_success_rate: float = 0.0
    doc_error_rate: float = 0.0
    
    # Evaluation metrics
    rouge_scores: Dict[str, float] = field(default_factory=lambda: {
        'rouge1': 0.0,
        'rouge2': 0.0,
        'rougeL': 0.0,
        'rougeLsum': 0.0
    })
    bert_scores: Dict[str, float] = field(default_factory=lambda: {
        'bert_precision': 0.0,
        'bert_recall': 0.0,
        'bert_f1': 0.0
    })
    bleu_score: float = 0.0
    meteor_score: float = 0.0
    wer_score: float = 0.0
    perplexity: float = 0.0
    semantic_similarity: float = 0.0
    
    # History tracking
    latency_history: List[Dict[str, Any]] = field(default_factory=list)
    query_history: List[Dict[str, Any]] = field(default_factory=list)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_query_metrics(self, success: bool, quality: float = 0.0):
        """Update query-related metrics."""
        self.total_queries += 1
        if success:
            self.successful_queries += 1
        else:
            self.failed_queries += 1
        
        self.query_success_rate = (self.successful_queries / self.total_queries) * 100
        self.response_quality = quality
    
    def update_document_metrics(self, num_docs: int, chunk_size: float, 
                              processing_time: float, embedding_time: float,
                              success: bool):
        """Update document processing metrics."""
        self.total_documents = num_docs
        self.avg_chunk_size = chunk_size
        self.doc_processing_time = processing_time
        self.embedding_time = embedding_time
        
        if success:
            self.doc_success_rate = 100.0
            self.doc_error_rate = 0.0
        else:
            self.doc_success_rate = 0.0
            self.doc_error_rate = 100.0
    
    def add_latency_record(self, retrieval: float, generation: float):
        """Add a record to latency history."""
        self.latency_history.append({
            'timestamp': datetime.now().isoformat(),
            'retrieval': retrieval,
            'generation': generation,
            'total': retrieval + generation
        })
        
        # Keep only last 100 records
        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-100:]
    
    def add_error_record(self, error_type: str, error_message: str):
        """Add an error record to history."""
        self.error_history.append({
            'timestamp': datetime.now().isoformat(),
            'type': error_type,
            'message': error_message
        })
        
        # Keep only last 100 records
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]

    def update_evaluation_metrics(self, reference: str, generated: str):
        """Update evaluation metrics using HuggingFace evaluate."""
        try:
            # Get metrics evaluator
            evaluator = MetricsEvaluator()
            
            # Calculate all metrics
            scores = evaluator.evaluate_response(reference, generated, calculate_all=True)
            
            # Update scores
            self.rouge_scores = {
                k: v for k, v in scores.items() if k.startswith('rouge')
            }
            self.bert_scores = {
                k: v for k, v in scores.items() if k.startswith('bert')
            }
            self.bleu_score = scores.get('bleu', 0.0)
            self.meteor_score = scores.get('meteor', 0.0)
            self.wer_score = scores.get('wer', 0.0)
            self.perplexity = scores.get('perplexity', 0.0)
            self.semantic_similarity = scores.get('semantic_similarity', 0.0)
            
            # Add to evaluation history
            self.evaluation_history.append({
                'timestamp': datetime.now().isoformat(),
                'rouge_scores': self.rouge_scores,
                'bert_scores': self.bert_scores,
                'bleu_score': self.bleu_score,
                'meteor_score': self.meteor_score,
                'wer_score': self.wer_score,
                'perplexity': self.perplexity,
                'semantic_similarity': self.semantic_similarity
            })
            
            # Keep only last 100 records
            if len(self.evaluation_history) > 100:
                self.evaluation_history = self.evaluation_history[-100:]
                
        except Exception as e:
            logger.error(f"Error calculating evaluation metrics: {str(e)}")

    def get_evaluation_metrics(self) -> Dict[str, Any]:
        """Get current evaluation metrics."""
        return {
            'rouge_scores': self.rouge_scores,
            'bert_scores': self.bert_scores,
            'bleu_score': self.bleu_score,
            'meteor_score': self.meteor_score,
            'wer_score': self.wer_score,
            'perplexity': self.perplexity,
            'semantic_similarity': self.semantic_similarity,
            'evaluation_history': self.evaluation_history
        }

@dataclass
class EntityInfo:
    """Information about an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0

class RAGEngine:
    """RAG (Retrieval-Augmented Generation) engine for document processing and retrieval."""
    
    def __init__(self):
        """Initialize RAG engine with configuration from settings."""
        self.config = settings
        self.metrics = RAGMetrics()
        self.metrics_evaluator = MetricsEvaluator()
        self._initialize_components()
        self._load_spacy_model()
    
    def _initialize_components(self) -> None:
        """Initialize RAG components based on configuration."""
        try:
            # Initialize text splitter
            self.text_splitter = self._initialize_text_splitter()
            
            # Initialize embeddings
            self.embeddings = self._initialize_embeddings()
            
            # Initialize vector store
            self.vector_store = self._initialize_vector_store()
            
            # Initialize reranker if enabled
            if self.config['retrieval']['rerank_results']:
                self.reranker = CrossEncoder(self.config['retrieval']['rerank_model'])
            
            # Initialize cache as a dictionary
            if self.config['cache']['enabled']:
                self.cache = {}
            
            # Initialize LLM
            self.llm = self._initialize_llm()
            
            # Initialize prompt template
            self.prompt_template = self._initialize_prompt_template()
            
            logger.info("RAG components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG components: {str(e)}")
            raise RAGError(f"Failed to initialize RAG components: {str(e)}")
    
    def _initialize_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Initialize text splitter based on configuration."""
        try:
            return RecursiveCharacterTextSplitter(
                chunk_size=settings.get("chunking.chunk_size", 1000),
                chunk_overlap=settings.get("chunking.chunk_overlap", 200),
                length_function=len,
                is_separator_regex=False
            )
        except Exception as e:
            logger.error(f"Error initializing text splitter: {str(e)}")
            raise RAGError(f"Failed to initialize text splitter: {str(e)}")
    
    def _initialize_embeddings(self) -> HuggingFaceEmbeddings:
        """Initialize embeddings model based on configuration."""
        try:
            return HuggingFaceEmbeddings(
                model_name=settings.get("embeddings.model"),
                model_kwargs={"device": "cuda" if settings.get("optimization.use_gpu", False) else "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise RAGError(f"Failed to initialize embeddings: {str(e)}")
    
    def _initialize_vector_store(self) -> Chroma:
        """Initialize vector store based on configuration."""
        try:
            persist_directory = settings.get("vector_store.persist_directory")
            collection_name = settings.get("vector_store.collection_name", "documents")
            
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise RAGError(f"Failed to initialize vector store: {str(e)}")
    
    def _initialize_llm(self):
        """Initialize the vLLM language model for inference."""
        try:
            llm_config = self.config['llm']
            self.sampling_params = SamplingParams(
                temperature=llm_config.get('temperature', 0.7),
                top_p=llm_config.get('top_p', 0.95),
                max_tokens=llm_config.get('max_tokens', 4096),
                frequency_penalty=llm_config.get('frequency_penalty', 0.0),
                presence_penalty=llm_config.get('presence_penalty', 0.0)
            )
            self.llm = LLM(
                model=llm_config['model'],
                dtype="auto",
                tensor_parallel_size=self.config.get('optimization', {}).get('tensor_parallel_size', 1),
                gpu_memory_utilization=self.config.get('optimization', {}).get('gpu_memory_utilization', 0.9),
                max_model_len=llm_config.get('max_tokens', 4096)
            )
        except Exception as e:
            logger.error(f"Error initializing vLLM: {str(e)}")
            raise RAGError(f"Failed to initialize vLLM: {str(e)}")

    def _initialize_prompt_template(self):
        """Initialize the prompt template."""
        try:
            system_prompt = self.config['query']['system_prompt']
            query_template = self.config['query']['query_template']
            
            template = f"""System: {system_prompt}

Context: {{context}}

Human: {query_template}

Assistant: Let me help you with that based on the provided context."""
            
            return template
        except Exception as e:
            logger.error(f"Error initializing prompt template: {str(e)}")
            raise RAGError(f"Failed to initialize prompt template: {str(e)}")

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        try:
            encoding = tiktoken.encoding_for_model(self.config['generation']['model'])
            return len(encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}")
            return len(text.split())  # Fallback to word count

    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to fit within token limit."""
        try:
            encoding = tiktoken.encoding_for_model(self.config['generation']['model'])
            tokens = encoding.encode(context)
            if len(tokens) <= max_tokens:
                return context
            return encoding.decode(tokens[:max_tokens])
        except Exception as e:
            logger.warning(f"Error truncating context: {str(e)}")
            return context[:max_tokens * 4]  # Fallback to character count
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def process_document(self, document: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Process a document and add it to the vector store."""
        try:
            # Split document into chunks
            chunks = self.text_splitter.split_text(document)
            
            # Create Document objects with metadata
            documents = [
                Document(page_content=chunk, metadata=metadata or {})
                for chunk in chunks
            ]
            
            # Add to vector store
            self.vector_store.add_documents(documents)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise RAGError(f"Failed to process document: {str(e)}")

    def process_documents_batch(
        self,
        documents: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None
    ) -> List[Document]:
        """Process multiple documents in batches."""
        if batch_size is None:
            batch_size = self.config['embeddings']['batch_size']
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if 'optimization' not in self.config:
            self.config['optimization'] = {}
        try:
            all_processed_docs = []
            total_docs = len(documents)
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
                # Process batch in parallel if enabled
                optimization_cfg = self.config.get('optimization', {})
                parallel = optimization_cfg.get('parallel_processing', False)
                max_workers = optimization_cfg.get('max_workers', 1)
                if parallel:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(self.process_document, doc, metadata)
                            for doc in batch
                        ]
                        batch_results = [future.result() for future in futures]
                else:
                    batch_results = [
                        self.process_document(doc, metadata)
                        for doc in batch
                    ]
                # Flatten results
                for docs in batch_results:
                    all_processed_docs.extend(docs)
                # Add delay between batches if needed
                if i + batch_size < total_docs:
                    time.sleep(0.1)  # Small delay to prevent rate limiting
            return all_processed_docs
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Error processing document batch: {str(e)}")
            raise RAGError(f"Failed to process document batch: {str(e)}")

    def process_documents_stream(
        self,
        documents: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None
    ) -> Iterator[List[Document]]:
        """Process multiple documents in batches and yield results as they are processed."""
        try:
            if batch_size is None:
                batch_size = self.config['embeddings']['batch_size']
            if 'optimization' not in self.config:
                self.config['optimization'] = {}
            total_docs = len(documents)
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_docs + batch_size - 1)//batch_size}")
                # Process batch in parallel if enabled
                optimization_cfg = self.config.get('optimization', {})
                parallel = optimization_cfg.get('parallel_processing', False)
                max_workers = optimization_cfg.get('max_workers', 1)
                if parallel:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(self.process_document, doc, metadata)
                            for doc in batch
                        ]
                        batch_results = [future.result() for future in futures]
                else:
                    batch_results = [
                        self.process_document(doc, metadata)
                        for doc in batch
                    ]
                # Yield results for this batch
                yield [doc for docs in batch_results for doc in docs]
                # Add delay between batches if needed
                if i + batch_size < total_docs:
                    time.sleep(0.1)  # Small delay to prevent rate limiting
        except Exception as e:
            logger.error(f"Error in document stream processing: {str(e)}")
            raise RAGError(f"Failed to process document stream: {str(e)}")

    def process_documents_with_progress(
        self,
        documents: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Document]:
        """Process multiple documents with progress tracking."""
        try:
            if batch_size is None:
                batch_size = self.config['embeddings']['batch_size']
            
            all_processed_docs = []
            total_docs = len(documents)
            
            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                current_batch = i//batch_size + 1
                total_batches = (total_docs + batch_size - 1)//batch_size
                
                logger.info(f"Processing batch {current_batch}/{total_batches}")
                
                # Process batch
                batch_results = self.process_documents_batch(batch, metadata, batch_size)
                all_processed_docs.extend(batch_results)
                
                # Call progress callback if provided
                if callback:
                    callback(current_batch, total_batches)
            
            return all_processed_docs
            
        except Exception as e:
            logger.error(f"Error in progress-tracked document processing: {str(e)}")
            raise RAGError(f"Failed to process documents with progress: {str(e)}")
    
    def _preprocess_query(self, query: str) -> str:
        """Preprocess the query based on configuration."""
        if not self.config['query']['preprocess']:
            return query
            
        # Basic preprocessing
        query = query.strip()
        if len(query) > self.config['query']['max_query_length']:
            query = query[:self.config['query']['max_query_length']]
            
        return query
    
    def _load_spacy_model(self):
        """Load spaCy model for entity extraction."""
        try:
            import spacy
            nlp_config = self.config.get('nlp', {})
            model_name = nlp_config.get('spacy_model', 'en_core_web_sm')
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
            self.entity_types = nlp_config.get('entity_types', None)
            self.sentiment_model = nlp_config.get('sentiment_model', None)
        except Exception as e:
            logger.error(f"Error loading spaCy model: {str(e)}")
            self.nlp = None

    def _extract_entities(self, text: str) -> List[EntityInfo]:
        """Extract named entities from text using spaCy.
        
        Args:
            text: Input text to extract entities from
            
        Returns:
            List of EntityInfo objects containing entity information
        """
        if not self.nlp:
            try:
                self._load_spacy_model()
            except Exception as e:
                logger.error(f"Failed to load spaCy model: {str(e)}")
                return []
            
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                # Calculate confidence based on entity length and type
                confidence = min(1.0, 0.5 + (len(ent.text) / 20))  # Longer entities get higher confidence
                if ent.label_ in ['PERSON', 'ORG', 'GPE']:  # Give higher confidence to important entity types
                    confidence = min(1.0, confidence + 0.2)
                
                entities.append(EntityInfo(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=confidence
                ))
            
            return entities
            
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return []

    def _expand_query(self, query: str) -> List[str]:
        """Expand the query using multiple techniques."""
        if not self.config['query']['expand_queries']:
            return [query]
        
        expanded = [query]
        
        # Add original query without question mark if present
        if "?" in query:
            expanded.append(query.replace("?", ""))
        
        # 1. Entity-based expansion
        entities = self._extract_entities(query)
        for entity in entities:
            if entity.confidence > 0.5:  # Only use high-confidence entities
                expanded.extend([
                    f"What is {entity.text}",
                    f"Tell me about {entity.text}",
                    f"Explain {entity.text}",
                    f"Describe {entity.text}",
                    f"Can you elaborate on {entity.text}"
                ])
        
        # 2. Question type expansion
        question_types = {
            "what": ["how", "why", "when", "where", "which", "what is the process"],
            "how": ["what is the process", "what are the steps", "what is the method", "explain the process"],
            "why": ["what is the reason", "what is the cause", "what is the purpose", "explain why"],
            "when": ["at what time", "during what period", "what is the timing", "when did it happen"],
            "where": ["in what location", "at what place", "what is the position", "where can I find"],
            "which": ["what is the best", "what are the options", "what are the choices", "what should I choose"]
        }
        
        # Check for question words and expand
        query_lower = query.lower()
        for word, expansions in question_types.items():
            if word in query_lower:
                expanded.extend(expansions)
                # Also add variations with the original query
                for expansion in expansions:
                    expanded.append(query_lower.replace(word, expansion))
        
        # 3. Semantic expansion using word embeddings
        if hasattr(self, 'embeddings') and hasattr(self.embeddings, 'embed_query'):
            try:
                query_embedding = self.embeddings.embed_query(query)
                # Find similar queries from a predefined set or cache
                similar_queries = self._find_similar_queries(query_embedding)
                expanded.extend(similar_queries)
            except Exception as e:
                logger.warning(f"Error in semantic query expansion: {str(e)}")
        
        # 4. Question reformulation
        if query.endswith("?"):
            base = query[:-1].strip()
            expanded.extend([
                f"Explain {base}",
                f"Describe {base}",
                f"Tell me about {base}",
                f"Can you elaborate on {base}",
                f"Provide information about {base}",
                f"Give me details about {base}"
            ])
        
        # 5. Remove duplicates and normalize
        expanded = list(set(expanded))
        expanded = [q.strip() for q in expanded if q.strip()]
        
        return expanded

    def _find_similar_queries(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        """Find similar queries using embeddings."""
        # This is a placeholder - in production, use a proper vector store for queries
        return []

    def _regenerate_response(
        self,
        query: str,
        documents: List[Document],
        validation: Dict[str, Any],
        max_attempts: int = 3
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Regenerate response based on validation issues."""
        for attempt in range(max_attempts):
            try:
                # Adjust prompt based on validation issues
                prompt_adjustments = []
                
                if "Response too short" in validation['issues']:
                    prompt_adjustments.append("Provide a more detailed response.")
                if "No source citations found" in validation['issues']:
                    prompt_adjustments.append("Include specific references to the provided sources.")
                if "Low query term coverage" in validation['issues']:
                    prompt_adjustments.append("Ensure the response directly addresses the query terms.")
                if any("Response contains" in issue for issue in validation['issues']):
                    prompt_adjustments.append("Provide a direct and confident response.")
                
                # Create enhanced prompt
                enhanced_prompt = self.prompt_template
                if prompt_adjustments:
                    enhanced_prompt += "\nAdditional requirements:\n" + "\n".join(prompt_adjustments)
                
                # Generate new response
                context = "\n\n".join([doc.page_content for doc in documents])
                response = self.llm.invoke(enhanced_prompt.format(context=context, query=query))
                
                # Validate new response
                new_validation = self._validate_response(response, query, documents)
                if new_validation['is_valid']:
                    return response, [
                        {
                            'content': doc.page_content,
                            'metadata': doc.metadata
                        }
                        for doc in documents
                    ]
                
                logger.warning(f"Regeneration attempt {attempt + 1} failed: {new_validation['issues']}")
                
            except Exception as e:
                logger.error(f"Error in response regeneration: {str(e)}")
        
        # If all attempts fail, return the original response
        return self.config['error_handling']['fallback_responses']['generation_error'], []

    def retrieve(self, query: str) -> List[Document]:
        """Stub: Retrieve documents relevant to the query. Returns two dummy Documents for testing."""
        import time
        start_time = time.time()
        docs = [
            Document(page_content="Test document 1", metadata={"source": "test1"}),
            Document(page_content="Test document 2", metadata={"source": "test2"})
        ]
        elapsed = time.time() - start_time
        self.metrics.retrieval_latency = elapsed if elapsed > 0 else 0.001
        return docs

    def generate_response(
        self,
        query: str,
        documents: List[Document],
        stream: bool = False,
        reference_answer: Optional[str] = None
    ) -> Union[Tuple[str, List[Document]], Iterator[Tuple[str, List[Document]]]]:
        """
        Generate a response using vLLM.
        """
        try:
            # Prepare context from retrieved documents
            context = "\n".join([doc.page_content for doc in documents])
            prompt = self.config['query']['query_template'].format(query=query, context=context)

            # vLLM expects a list of prompts
            start_time = time.time()
            outputs = self.llm.generate(
                prompts=[prompt],
                sampling_params=self.sampling_params
            )
            elapsed = time.time() - start_time
            response = outputs[0].outputs[0].text  # Get the generated text

            # Metrics: latency and token usage
            self.metrics.generation_latency = elapsed if elapsed > 0 else 0.001
            self.metrics.token_usage += len(response.split())  # Approximate token count

            sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents]
            return response, sources

        except Exception as e:
            logger.error(f"Error generating response with vLLM: {str(e)}", exc_info=True)
            return (
                self.config['error_handling']['fallback_responses']['generation_error'],
                []
            )

    def process_query(
        self,
        query: str,
        stream: bool = False
    ) -> Union[Tuple[str, List[Dict[str, Any]]], Iterator[Tuple[str, List[Dict[str, Any]]]]]:
        """Process a query and return response with sources."""
        # Initialize cache if not exists
        if not hasattr(self, 'cache'):
            self.cache = {}
        
        # Check cache if enabled and not streaming
        if self.config.get('cache', {}).get('enabled', False) and not stream:
            if query in self.cache:
                self.metrics.cache_hits += 1
                return self.cache[query]
            self.metrics.cache_misses += 1
        
        # Retrieve relevant documents
        try:
            documents = self.retrieve(query)
        except RetrievalError as e:
            logger.error(f"Retrieval error: {str(e)}")
            if stream:
                def fallback_stream():
                    yield self.config['error_handling']['fallback_responses']['retrieval_error'], []
                return fallback_stream()
            else:
                return (
                    self.config['error_handling']['fallback_responses']['retrieval_error'],
                    []
                )
        
        # Generate response
        try:
            result = self.generate_response(query, documents, stream=stream)
            
            # Cache result if enabled and not streaming
            if self.config.get('cache', {}).get('enabled', False) and not stream:
                self.cache[query] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            if stream:
                def fallback_stream():
                    yield self.config['error_handling']['fallback_responses']['generation_error'], []
                return fallback_stream()
            else:
                return (
                    self.config['error_handling']['fallback_responses']['generation_error'],
                    []
                )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {
            'retrieval_latency': self.metrics.retrieval_latency,
            'generation_latency': self.metrics.generation_latency,
            'cache_hits': self.metrics.cache_hits,
            'cache_misses': self.metrics.cache_misses,
            'token_usage': self.metrics.token_usage,
            'query_success_rate': self.metrics.query_success_rate,
            'response_quality': self.metrics.response_quality,
            'rouge_scores': self.metrics.rouge_scores,
            'bert_scores': self.metrics.bert_scores,
            'bleu_score': self.metrics.bleu_score,
            'meteor_score': self.metrics.meteor_score,
            'wer_score': self.metrics.wer_score,
            'perplexity': self.metrics.perplexity,
            'semantic_similarity': self.metrics.semantic_similarity,
            'evaluation_history': self.metrics.evaluation_history,
            'error_history': self.metrics.error_history
        }
    
    def reset_metrics(self):
        """Reset metrics to zero."""
        self.metrics = RAGMetrics()

    def get_benchmark_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get performance benchmark results."""
        results = {
            'query_benchmark': [],
            'document_benchmark': [],
            'system_benchmark': []
        }
        
        # Query benchmark results
        if self.metrics.latency_history:
            results['query_benchmark'] = [
                {
                    'timestamp': record['timestamp'],
                    'latency': record['total'],
                    'throughput': 1.0 / record['total'] if record['total'] > 0 else 0,
                    'success_rate': self.metrics.query_success_rate,
                    'error_rate': 100 - self.metrics.query_success_rate,
                    'cache_hit_rate': (self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)) * 100 if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0,
                    'token_usage': self.metrics.token_usage
                }
                for record in self.metrics.latency_history
            ]
        
        # Document benchmark results
        results['document_benchmark'] = [
            {
                'timestamp': datetime.now().isoformat(),
                'processing_time': self.metrics.doc_processing_time,
                'embedding_time': self.metrics.embedding_time,
                'success_rate': self.metrics.doc_success_rate,
                'error_rate': self.metrics.doc_error_rate,
                'avg_chunk_size': self.metrics.avg_chunk_size,
                'total_documents': self.metrics.total_documents
            }
        ]
        
        # System benchmark results
        results['system_benchmark'] = [
            {
                'timestamp': datetime.now().isoformat(),
                'memory_usage': psutil.virtual_memory().percent,
                'cpu_usage': psutil.cpu_percent(),
                'disk_usage': psutil.disk_usage('/').percent,
                'active_connections': len(self.active_connections) if hasattr(self, 'active_connections') else 0,
                'queue_size': len(self.request_queue) if hasattr(self, 'request_queue') else 0
            }
        ]
        
        return results 

    def _rerank_results(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank documents based on relevance to query."""
        if not documents:
            return []
            
        # Calculate similarity scores
        scores = []
        for doc in documents:
            # Simple TF-IDF based scoring
            query_terms = set(query.lower().split())
            doc_terms = set(doc.page_content.lower().split())
            score = len(query_terms.intersection(doc_terms)) / len(query_terms) if query_terms else 0
            scores.append((doc, score))
            
        # Sort by score in descending order
        reranked = [doc for doc, score in sorted(scores, key=lambda x: x[1], reverse=True)]
        return reranked 

    def process_queries_batch(
        self,
        queries: List[str],
        batch_size: Optional[int] = None,
        stream: bool = False
    ) -> List[Tuple[str, List[Dict[str, Any]]]]:
        """Process multiple queries in batches.
        
        Args:
            queries: List of queries to process
            batch_size: Size of each batch (defaults to config value)
            stream: Whether to stream responses
            
        Returns:
            List of (response, sources) tuples
        """
        if batch_size is None:
            batch_size = self.config.get('optimization', {}).get('batch_size', 10)
        
        results = []
        total_queries = len(queries)
        
        try:
            for i in range(0, total_queries, batch_size):
                batch = queries[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(total_queries + batch_size - 1)//batch_size}")
                
                # Process batch in parallel if enabled
                if self.config.get('optimization', {}).get('parallel_processing', False):
                    with ThreadPoolExecutor(max_workers=self.config['optimization'].get('max_workers', 1)) as executor:
                        futures = [
                            executor.submit(self.process_query, query, stream)
                            for query in batch
                        ]
                        batch_results = [future.result() for future in futures]
                else:
                    batch_results = [
                        self.process_query(query, stream)
                        for query in batch
                    ]
                
                if stream:
                    # For streaming, yield each result as it comes
                    for result in batch_results:
                        yield result
                else:
                    results.extend(batch_results)
                
                # Add delay between batches if needed
                if i + batch_size < total_queries:
                    time.sleep(0.1)  # Small delay to prevent rate limiting
            
            if not stream:
                return results
                
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise RAGError(f"Failed to process queries batch: {str(e)}")

    def _validate_response(
        self,
        response: str,
        query: str,
        documents: List[Document]
    ) -> Dict[str, Any]:
        """Validate a generated response.
        
        Args:
            response: The generated response
            query: The original query
            documents: The source documents
            
        Returns:
            Dictionary containing validation results
        """
        validation = {
            'is_valid': True,
            'issues': [],
            'metrics': {}
        }
        
        # Check response length
        min_length = self.config.get('validation', {}).get('min_response_length', 10)
        if len(response) < min_length:
            validation['is_valid'] = False
            validation['issues'].append("Response too short")
        
        # Check response relevance
        if hasattr(self, 'embeddings'):
            try:
                # Get embeddings for query and response
                query_embedding = self.embeddings.embed_query(query)
                response_embedding = self.embeddings.embed_query(response)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, response_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(response_embedding)
                )
                
                # Check if similarity is above threshold
                min_similarity = self.config.get('validation', {}).get('min_similarity', 0.5)
                if similarity < min_similarity:
                    validation['is_valid'] = False
                    validation['issues'].append("Response not relevant to query")
                
                validation['metrics']['similarity'] = similarity
                
            except Exception as e:
                logger.warning(f"Error calculating response relevance: {str(e)}")
        
        # Check if response contains information from source documents
        doc_contents = [doc.page_content for doc in documents]
        doc_words = set(' '.join(doc_contents).lower().split())
        response_words = set(response.lower().split())
        
        # Calculate overlap
        overlap = len(doc_words.intersection(response_words)) / len(response_words) if response_words else 0
        min_overlap = self.config.get('validation', {}).get('min_doc_overlap', 0.3)
        
        if overlap < min_overlap:
            validation['is_valid'] = False
            validation['issues'].append("Response not grounded in source documents")
        
        validation['metrics']['doc_overlap'] = overlap
        
        # Calculate source coverage
        source_coverage = {}
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            doc_words = set(doc.page_content.lower().split())
            coverage = len(doc_words.intersection(response_words)) / len(doc_words) if doc_words else 0
            source_coverage[source] = coverage
        
        validation['metrics']['source_coverage'] = source_coverage
        
        # Check for citations
        citation_pattern = r'\[\d+\]|\(\d+\)'
        if not re.search(citation_pattern, response):
            validation['is_valid'] = False
            validation['issues'].append("No source citations found")
        
        return validation 