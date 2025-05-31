#!/usr/bin/env python
"""
Convert various document formats to ShareGPT format for fine-tuning.

This script processes documents from different formats (PDF, DOCX, CSV, HTML, etc.)
and converts them into ShareGPT conversation format suitable for fine-tuning.
"""

import os
import sys
import json
import logging
import argparse
import time
import re
import random
import requests
import threading
import psutil
import yaml
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import (
    Docx2txtLoader, 
    CSVLoader,
    UnstructuredFileLoader,
    BSHTMLLoader,
    JSONLoader,
    TextLoader,
    DirectoryLoader
)
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_core.documents import Document

from core.settings import settings, DATA_DIR, OLLAMA_BASE_URL, EMBEDDING_MODEL

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Change to DEBUG level
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('convert_to_sharegpt.log')
    ]
)
logger = logging.getLogger(__name__)

# Global variables for LLM availability
LLM_AVAILABLE = None
MAX_BATCH_SIZE = 20  # Maximum number of documents to process at once

# Global variables for process control
STOP_PROCESSING = threading.Event()
PROGRESS_CALLBACK = None
PROGRESS_LOCK = threading.Lock()
CHUNKS_PROCESSED = 0
CURRENT_CHUNK = 0

@dataclass
class ProcessingMetrics:
    """Class to track processing metrics"""
    start_time: datetime = field(default_factory=datetime.now)
    total_documents: int = 0
    processed_documents: int = 0
    total_chunks: int = 0
    processed_chunks: int = 0
    total_questions: int = 0
    processed_questions: int = 0
    file_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    errors: List[str] = field(default_factory=list)
    memory_usage: List[float] = field(default_factory=list)
    
    def update_memory_usage(self):
        """Update current memory usage"""
        process = psutil.Process()
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
    
    def get_elapsed_time(self) -> str:
        """Get formatted elapsed time"""
        elapsed = datetime.now() - self.start_time
        return str(timedelta(seconds=int(elapsed.total_seconds())))
    
    def get_progress_percentage(self) -> float:
        """Calculate overall progress percentage"""
        if self.total_documents == 0:
            return 0.0
        return (self.processed_documents / self.total_documents) * 100
    
    def get_memory_usage(self) -> str:
        """Get current memory usage in MB"""
        if not self.memory_usage:
            return "0 MB"
        return f"{self.memory_usage[-1]:.2f} MB"
    
    def log_metrics(self, logger):
        """Log current metrics"""
        logger.info("=== Processing Metrics ===")
        logger.info(f"Elapsed Time: {self.get_elapsed_time()}")
        logger.info(f"Progress: {self.get_progress_percentage():.1f}%")
        logger.info(f"Documents: {self.processed_documents}/{self.total_documents}")
        logger.info(f"Chunks: {self.processed_chunks}/{self.total_chunks}")
        logger.info(f"Questions: {self.processed_questions}/{self.total_questions}")
        logger.info(f"Memory Usage: {self.get_memory_usage()}")
        logger.info("File Types Processed:")
        for file_type, count in self.file_types.items():
            logger.info(f"  {file_type}: {count}")
        if self.errors:
            logger.info(f"Errors: {len(self.errors)}")
            for error in self.errors[-5:]:  # Show last 5 errors
                logger.info(f"  - {error}")
        logger.info("=======================")

# Create global metrics instance
metrics = ProcessingMetrics()

def check_llm_availability(max_retries: int = 3, retry_delay: int = 2) -> bool:
    """
    Check if the LLM service (Ollama) is available with retry logic.
    Only checks if the model specified in config is available. Does not override config.
    """
    global LLM_AVAILABLE
    if LLM_AVAILABLE is not None:
        return LLM_AVAILABLE
    model_name = settings.get("llm.model")
    for attempt in range(max_retries):
        try:
            response = requests.get(
                f"{OLLAMA_BASE_URL}/api/tags",
                timeout=10
            )
            if response.status_code != 200:
                logger.warning(f"Ollama server returned status code {response.status_code} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                LLM_AVAILABLE = False
                return False
            models = response.json().get("models", [])
            model_names = [model.get("name") for model in models]
            if model_name in model_names:
                logger.info(f"Model '{model_name}' from config is available.")
                LLM_AVAILABLE = True
                return True
            logger.warning(f"Model '{model_name}' from config not found in available models: {model_names}")
            LLM_AVAILABLE = False
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            LLM_AVAILABLE = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error checking LLM availability: {str(e)}")
            LLM_AVAILABLE = False
            return False
    LLM_AVAILABLE = False
    return False

def get_file_extension(filename: str) -> str:
    """
    Get file extension from filename.
    
    Args:
        filename: Path or filename
        
    Returns:
        Lowercase file extension with dot (e.g., '.pdf')
    """
    return os.path.splitext(filename)[1].lower()

def load_documents(source_dir: str) -> List[Document]:
    """
    Load documents from a specified source directory with multiple file types.
    """
    documents = []
    
    # Convert to absolute path and normalize
    abs_source_dir = os.path.normpath(os.path.abspath(source_dir))
    logger.info(f"Loading documents from: {abs_source_dir}")
    
    # Define loaders for different file types
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".doc": Docx2txtLoader,
        ".csv": CSVLoader,
        ".txt": TextLoader,
        ".html": BSHTMLLoader,
        ".htm": BSHTMLLoader,
        ".json": JSONLoader,
    }
    
    for root, _, files in os.walk(abs_source_dir):
        for file in files:
            file_path = os.path.normpath(os.path.join(root, file))
            file_ext = get_file_extension(file)
            
            try:
                logger.info(f"Processing file: {file_path}")
                logger.info(f"File type: {file_ext}")
                
                # Update file type metrics
                metrics.file_types[file_ext] += 1
                
                # Extract topic from path
                topic = os.path.splitext(file)[0]
                metadata = {
                    "source": file_path,
                    "topic": topic,
                }
                
                # Use specific loader if available, otherwise use UnstructuredFileLoader
                if file_ext in loaders:
                    if file_ext == ".json":
                        try:
                            # First try with content field
                            loader = loaders[file_ext](
                                file_path,
                                text_content=True,
                                jq_schema=".content"
                            )
                            docs = loader.load()
                        except Exception as e:
                            logger.warning(f"Failed to load JSON with .content schema, trying alternative: {str(e)}")
                            try:
                                # Try with text field
                                loader = loaders[file_ext](
                                    file_path,
                                    text_content=True,
                                    jq_schema=".text"
                                )
                                docs = loader.load()
                            except Exception as e:
                                logger.warning(f"Failed to load JSON with .text schema, trying raw content: {str(e)}")
                                # Try loading raw JSON content
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = json.load(f)
                                    if isinstance(content, dict):
                                        # Convert dict to string representation
                                        content = json.dumps(content, ensure_ascii=False)
                                    docs = [Document(page_content=str(content), metadata=metadata)]
                    else:
                        loader = loaders[file_ext](file_path)
                        docs = loader.load()
                else:
                    # Try to load as a generic file
                    loader = UnstructuredFileLoader(file_path)
                    docs = loader.load()
                
                # Add metadata to each document
                for doc in docs:
                    doc.metadata.update(metadata)
                
                # Update metrics
                metrics.total_documents += len(docs)
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path}")
                
            except Exception as e:
                error_msg = f"Error loading {file_path}: {str(e)}"
                logger.error(error_msg)
                metrics.errors.append(error_msg)
    
    return documents

def split_documents(
    documents: List[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    max_chunks: int = None
) -> List[Document]:
    """
    Split documents into chunks using sentence transformers token splitter.
    """
    try:
        text_splitter = SentenceTransformersTokenTextSplitter(
            tokens_per_chunk=chunk_size,
            chunk_overlap=chunk_overlap,
            model_name=EMBEDDING_MODEL
        )
        chunks = []
        for doc in documents:
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
            metrics.processed_documents += 1
            metrics.total_chunks += len(doc_chunks)
            logger.info(f"Split document into {len(doc_chunks)} chunks")
        if max_chunks and len(chunks) > max_chunks:
            chunks = chunks[:max_chunks]
            logger.info(f"Limited chunks to {max_chunks}")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise

def rule_based_question_generation(content: str, num_questions: int = 3) -> List[str]:
    """
    Generate questions from content using rule-based approaches when LLM is unavailable.
    
    Args:
        content: Document content text
        num_questions: Number of questions to generate
        
    Returns:
        List of generated questions
    """
    questions = []
    
    # Truncate content if too long
    max_content_length = 4000
    if len(content) > max_content_length:
        content = content[:max_content_length]
    
    # Extract potential topics from the first paragraph
    first_paragraph = content.split("\n\n")[0] if "\n\n" in content else content[:500]
    
    # 1. Extract key entities and create "What is X?" questions
    words = re.findall(r'\b[A-Z][a-zA-Z]{5,}\b', content)
    entities = list(set([w for w in words if len(w) > 5]))[:5]  # Limit to 5 unique entities
    
    for entity in entities[:min(len(entities), num_questions)]:
        questions.append(f"What is {entity} and why is it important?")
    
    # 2. Add generic questions based on content length
    generic_questions = [
        f"What are the main points discussed in this document?",
        f"How does this information relate to industry best practices?",
        f"What are the key takeaways from this content?",
        f"Can you summarize the most important information in this document?",
        f"What practical applications does this information have?",
        f"How would you explain this information to someone new to the topic?"
    ]
    
    # Add generic questions if we don't have enough entity-based ones
    while len(questions) < num_questions:
        if generic_questions:
            questions.append(generic_questions.pop(0))
        else:
            break
    
    # 3. Add topic-specific question based on document metadata if available
    if len(questions) < num_questions and "topic" in content.lower():
        topic_match = re.search(r'\b(regarding|about|on|for)\s+([a-zA-Z\s]+)', content.lower())
        if topic_match:
            topic = topic_match.group(2).strip()
            questions.append(f"What specific information does this document provide about {topic}?")
    
    # Generate templated questions if we still need more
    templates = [
        "What are the requirements for {}?",
        "How does {} work in practice?",
        "What challenges might arise with {}?",
        "How can {} be improved?",
        "What are the benefits of {}?"
    ]
    
    # Try to extract potential subjects from the content
    potential_subjects = re.findall(r'(?:the|a|an)\s+([a-zA-Z]{5,}(?:\s+[a-zA-Z]+){0,2})', content.lower())
    unique_subjects = list(set([s.strip() for s in potential_subjects if len(s) > 5]))
    
    # Generate questions from templates and subjects
    if unique_subjects and len(questions) < num_questions:
        for template in templates:
            if len(questions) >= num_questions:
                break
            if unique_subjects:
                subject = unique_subjects.pop(0)
                questions.append(template.format(subject))
    
    # Ensure we have exactly the requested number of questions
    if len(questions) > num_questions:
        questions = questions[:num_questions]
    
    # If we still don't have enough questions, add very generic ones
    while len(questions) < num_questions:
        questions.append(f"What are the implications of this information for the industry?")
    
    return questions

def generate_questions_from_content(content: str, num_questions: int = 3) -> List[str]:
    """
    Generate questions from content using LLM or rule-based approach.
    
    Args:
        content: Text content to generate questions from
        num_questions: Number of questions to generate
        
    Returns:
        List of generated questions
    """
    try:
        # Check if LLM is available
        if check_llm_availability():
            # Use LLM for question generation
            prompt = f"""Generate {num_questions} relevant questions based on the following content.
            The questions should be specific, clear, and test understanding of key concepts.
            
            Content:
            {content}
            
            Questions:"""
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.get("llm.model"),
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                questions = response.json()["response"].split("\n")
                questions = [q.strip() for q in questions if q.strip()]
                return questions[:num_questions]
        
        # Fallback to rule-based approach
        return rule_based_question_generation(content, num_questions)
        
    except Exception as e:
        logger.error(f"Error generating questions: {str(e)}", exc_info=True)
        return rule_based_question_generation(content, num_questions)

def enhance_answer(content: str, question: str) -> str:
    """
    Enhance answer with additional context and formatting.
    
    Args:
        content: Original content
        question: Question being answered
        
    Returns:
        Enhanced answer
    """
    try:
        # Check if LLM is available
        if check_llm_availability():
            # Use LLM to enhance answer
            prompt = f"""Given the following content and question, provide a clear and comprehensive answer.
            Include relevant details and examples from the content.
            
            Content:
            {content}
            
            Question:
            {question}
            
            Answer:"""
            
            response = requests.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": settings.get("llm.model"),
                    "prompt": prompt,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                return response.json()["response"].strip()
        
        # Fallback to simple answer
        return content
        
    except Exception as e:
        logger.error(f"Error enhancing answer: {str(e)}", exc_info=True)
        return content

def process_document_chunk(chunk_data):
    """
    Process a single document chunk to generate questions and answers.
    Used for parallel processing.
    
    Args:
        chunk_data: Tuple containing (index, document, system_prompt, questions_per_chunk, enhance_answers)
        
    Returns:
        List of conversation objects
    """
    global CHUNKS_PROCESSED, CURRENT_CHUNK
    
    i, doc, system_prompt, questions_per_chunk, enhance_answers = chunk_data
    conversations = []
    
    # Check if processing should stop
    if STOP_PROCESSING.is_set():
        logger.info(f"Skipping document chunk {i} due to stop request")
        return []
    
    logger.info(f"Processing document chunk {i}")
    
    try:
        # Update global progress counters
        with PROGRESS_LOCK:
            CHUNKS_PROCESSED += 1
            CURRENT_CHUNK = i
            if PROGRESS_CALLBACK:
                try:
                    PROGRESS_CALLBACK(i, CHUNKS_PROCESSED)
                except Exception as e:
                    logger.warning(f"Progress callback error for chunk {i}: {e}")
        
        # Extract content from document with validation
        if not hasattr(doc, 'page_content') or not doc.page_content:
            logger.error(f"Invalid document format for chunk {i}")
            return []
            
        content = doc.page_content
        source = doc.metadata.get("source", "unknown")
        topic = doc.metadata.get("topic", "unknown")
        
        # Generate questions based on the content
        try:
            questions = generate_questions_from_content(content, questions_per_chunk)
            if not questions:
                logger.warning(f"No questions generated for chunk {i}")
                return []
        except Exception as e:
            logger.error(f"Error generating questions for chunk {i}: {e}")
            return []
        
        # Create conversation for each question
        for q in questions:
            if STOP_PROCESSING.is_set():
                break
                
            try:
                # Create the answer
                if enhance_answers:
                    answer = enhance_answer(content, q)
                else:
                    answer = content
                
                # Validate answer
                if not answer or len(answer.strip()) == 0:
                    logger.warning(f"Empty answer generated for question in chunk {i}")
                    continue
                
                # Create conversation object
                conv = {
                    "id": f"chunk_{i}_q_{hash(q) % 10000}",
                    "source": source,
                    "topic": topic,
                    "conversations": [
                        {
                            "from": "human",
                            "value": q
                        },
                        {
                            "from": "gpt",
                            "value": answer
                        }
                    ]
                }
                
                # Add system prompt if provided
                if system_prompt:
                    conv["system"] = system_prompt
                    
                conversations.append(conv)
                
            except Exception as e:
                logger.error(f"Error processing question in chunk {i}: {e}")
                continue
        
        logger.info(f"Generated {len(conversations)} conversations from chunk {i}")
        return conversations
        
    except Exception as e:
        logger.error(f"Error processing chunk {i}: {e}")
        return []

def convert_to_sharegpt_format(
    question: str,
    answer: str,
    system_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convert a question-answer pair to ShareGPT format.
    
    Args:
        question: Question text
        answer: Answer text
        system_prompt: Optional system prompt
        
    Returns:
        Dictionary in ShareGPT format
    """
    try:
        # Create conversation
        conversation = []
        
        # Add system message if provided
        if system_prompt:
            conversation.append({
                "from": "system",
                "value": system_prompt
            })
        
        # Add human question
        conversation.append({
            "from": "human",
            "value": question
        })
        
        # Add gpt answer
        conversation.append({
            "from": "gpt",
            "value": answer
        })
        
        return {
            "conversations": conversation,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source": "document_processing"
            }
        }
        
    except Exception as e:
        logger.error(f"Error converting to ShareGPT format: {str(e)}", exc_info=True)
        raise

def process_chunks_to_sharegpt(
    chunks: list,
    system_prompt: str = None,
    questions_per_chunk: int = 2,
    enhance_answers: bool = False
) -> list:
    """
    Process all chunks, generate Q&A pairs, and convert to ShareGPT format.
    """
    sharegpt_data = []
    metrics.total_chunks = len(chunks)
    metrics.total_questions = len(chunks) * questions_per_chunk
    
    for i, doc in enumerate(chunks):
        metrics.processed_chunks = i + 1
        metrics.update_memory_usage()
        
        content = getattr(doc, 'page_content', None)
        if not content:
            continue
            
        # Generate questions
        questions = generate_questions_from_content(content, questions_per_chunk)
        for q in questions:
            metrics.processed_questions += 1
            if enhance_answers:
                answer = enhance_answer(content, q)
            else:
                answer = content
            sharegpt_data.append(
                convert_to_sharegpt_format(q, answer, system_prompt=system_prompt)
            )
            
        # Log metrics periodically
        if (i + 1) % 10 == 0:  # Log every 10 chunks
            metrics.log_metrics(logger)
    
    return sharegpt_data

def main():
    """Main function to convert documents to ShareGPT format."""
    try:
        parser = argparse.ArgumentParser(
            description="Convert documents to ShareGPT format for fine-tuning"
        )
        parser.add_argument(
            "--source_dirs", "-s",
            type=str,
            nargs="+",  # Allow multiple source directories
            default=[os.path.join(DATA_DIR, "raw"), os.path.join(DATA_DIR, "insurance")],
            help="Directories containing documents to convert (space-separated)"
        )
        parser.add_argument(
            "--output_file", "-o",
            type=str,
            default=os.path.join(DATA_DIR, "training", "generated_conversations.json"),
            help="Output file for ShareGPT format data"
        )
        parser.add_argument(
            "--chunk_size", "-c",
            type=int,
            default=256,
            help="Size of document chunks in tokens"
        )
        parser.add_argument(
            "--chunk_overlap", "-v",
            type=int,
            default=128,
            help="Overlap between chunks in tokens"
        )
        parser.add_argument(
            "--questions_per_chunk", "-q",
            type=int,
            default=2,
            help="Number of questions to generate per chunk"
        )
        parser.add_argument(
            "--system_prompt", "-p",
            type=str,
            default=None,
            help="System prompt to use in conversations"
        )
        parser.add_argument(
            "--enhance_answers", "-e",
            action="store_true",
            help="Enhance answers using LLM"
        )
        parser.add_argument(
            "--max_workers", "-w",
            type=int,
            default=4,
            help="Maximum number of parallel workers"
        )
        parser.add_argument(
            "--max_chunks", "-m",
            type=int,
            default=None,
            help="Maximum number of chunks to process (for memory management)"
        )
        parser.add_argument(
            "--batch_size", "-b",
            type=int,
            default=MAX_BATCH_SIZE,
            help=f"Batch size for processing (default: {MAX_BATCH_SIZE})"
        )
        
        # Load chunk_size from rag.yaml
        try:
            with open(os.path.join(os.path.dirname(__file__), '../config/rag.yaml'), 'r') as f:
                rag_config = yaml.safe_load(f)
                chunk_size_default = rag_config.get('rag', {}).get('chunk_size', 256)
        except Exception as e:
            chunk_size_default = 256
            logger.warning(f"Could not load chunk_size from rag.yaml: {e}")
        
        args = parser.parse_args()
        
        # Log the model name from config
        logger.info(f"Using model from config: {settings.get('llm.model')}")
        # Check if LLM is available
        llm_status = "available" if check_llm_availability() else "unavailable"
        logger.info(f"LLM service status: {llm_status}")
        
        # Load documents from all source directories
        all_documents = []
        for source_dir in args.source_dirs:
            logger.info(f"Processing documents from {source_dir}")
            try:
                documents = load_documents(source_dir)
                if documents:
                    all_documents.extend(documents)
                    metrics.processed_documents += len(documents)
                    logger.info(f"Loaded {len(documents)} documents from {source_dir}")
                else:
                    logger.warning(f"No documents found in {source_dir}")
            except Exception as e:
                logger.error(f"Error loading documents from {source_dir}: {str(e)}", exc_info=True)
                continue
        
        if not all_documents:
            logger.error("No documents found in any source directory. Exiting.")
            sys.exit(1)
        
        # Split documents into chunks
        try:
            chunks = split_documents(all_documents, args.chunk_size, args.chunk_overlap, args.max_chunks)
            logger.info(f"Successfully split documents into {len(chunks)} chunks")
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}", exc_info=True)
            sys.exit(1)
        
        # Convert to ShareGPT format
        try:
            sharegpt_data = process_chunks_to_sharegpt(
                chunks,
                system_prompt=args.system_prompt,
                questions_per_chunk=args.questions_per_chunk,
                enhance_answers=args.enhance_answers
            )
            logger.info(f"Successfully processed {len(sharegpt_data)} conversations")
        except Exception as e:
            logger.error(f"Error processing chunks: {str(e)}", exc_info=True)
            sys.exit(1)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        # Write to output file
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully wrote {len(sharegpt_data)} conversations to {args.output_file}")
        except Exception as e:
            logger.error(f"Error writing output file: {str(e)}", exc_info=True)
            sys.exit(1)
        
        # Log final metrics
        metrics.log_metrics(logger)
        
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}", exc_info=True)
        metrics.errors.append(f"Main processing error: {str(e)}")
        metrics.log_metrics(logger)
        sys.exit(1)

if __name__ == "__main__":
    main() 