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
from typing import List, Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed, Future

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.document_loaders import (
    PyPDFLoader, 
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

from core.settings import settings, DATA_DIR, OLLAMA_BASE_URL

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
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

def check_llm_availability() -> bool:
    """
    Check if the LLM service (Ollama) is available.
    
    Returns:
        bool: True if LLM is available, False otherwise
    """
    global LLM_AVAILABLE
    
    # Return cached result if we've already checked
    if LLM_AVAILABLE is not None:
        return LLM_AVAILABLE
    
    try:
        # Try to connect to Ollama and check if the model is available
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags")
        
        if response.status_code != 200:
            logger.warning(f"Ollama server is running but returned status code {response.status_code}")
            LLM_AVAILABLE = False
            return False
        
        # Check if our model is available
        models = response.json().get("models", [])
        model_names = [model.get("name") for model in models]
        
        # If deepseek-llm:7b is available, prioritize it
        if "deepseek-llm:7b" in model_names:
            settings.model_name = "deepseek-llm:7b"
            logger.info(f"Using preferred model deepseek-llm:7b")
            LLM_AVAILABLE = True
            return True
        
        # If deepseek-r1:32b is available but we should use rule-based approach for large models
        if "deepseek-r1:32b" in model_names:
            settings.model_name = "deepseek-r1:32b"
            logger.info(f"Using preferred model deepseek-r1:32b")
            # For 32B models, we'll use rule-based approach to avoid timeouts
            logger.warning("Large 32B model detected - will use rule-based processing to avoid timeouts")
            LLM_AVAILABLE = False
            return False
        
        if settings.model_name in model_names:
            # Also check if model name contains markers of very large models
            if "32b" in settings.model_name.lower() or "32-b" in settings.model_name.lower():
                logger.warning(f"Large model {settings.model_name} detected - will use rule-based processing to avoid timeouts")
                LLM_AVAILABLE = False
                return False
            # For 7B models, they're fine to use with LLM
            elif "7b" in settings.model_name.lower() or "7-b" in settings.model_name.lower():
                logger.info(f"Using 7B model {settings.model_name} for LLM processing")
                LLM_AVAILABLE = True
                return True
                
            logger.info(f"LLM model {settings.model_name} is available")
            LLM_AVAILABLE = True
            return True
        else:
            logger.warning(f"Model {settings.model_name} not found in available models: {model_names}")
            
            # If we have any model available, we can still use it
            if models:
                # Use the first available model that's not a huge model
                for model in models:
                    model_name = model['name']
                    # Prefer 7B models
                    if "7b" in model_name.lower() or "7-b" in model_name.lower():
                        settings.model_name = model_name
                        logger.info(f"Will use available 7B model: {settings.model_name} instead")
                        LLM_AVAILABLE = True
                        return True
                    # Avoid 32B models
                    elif "32b" not in model_name.lower() and "32-b" not in model_name.lower():
                        settings.model_name = model_name
                        logger.info(f"Will use available model: {settings.model_name} instead")
                        LLM_AVAILABLE = True
                        return True
            
            LLM_AVAILABLE = False
            return False
            
    except requests.exceptions.RequestException as e:
        logger.warning(f"Ollama server is not available: {e}")
        LLM_AVAILABLE = False
        return False

def load_documents(source_dir: str) -> List[Document]:
    """
    Load documents from a specified source directory with multiple file types.
    
    Args:
        source_dir: Directory containing documents
        
    Returns:
        List of Document objects with metadata
    """
    documents = []
    
    logger.info(f"Loading documents from {source_dir}")
    
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
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            try:
                # Extract topic from path
                topic = os.path.splitext(file)[0]
                metadata = {
                    "source": file_path,
                    "topic": topic,
                }
                
                # Use specific loader if available, otherwise use UnstructuredFileLoader
                if file_ext in loaders:
                    if file_ext == ".json":
                        # JSON loader requires a jq-like string to specify the content field
                        loader = loaders[file_ext](file_path, jq=".content", text_content=True)
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
                
                documents.extend(docs)
                logger.info(f"Loaded {len(docs)} documents from {file_path}")
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    return documents

def split_documents(
    documents: List[Document],
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    max_chunks: int = None
) -> List[Document]:
    """
    Split documents into smaller chunks using sentence-aware token splitting.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of chunks in tokens
        chunk_overlap: Overlap between chunks in tokens
        max_chunks: Maximum number of chunks to return (for memory management)
        
    Returns:
        List of chunked Document objects
    """
    if not documents:
        logger.warning("No documents to split")
        return []
        
    # Use the SentenceTransformersTokenTextSplitter for better chunking
    splitter = SentenceTransformersTokenTextSplitter(
        model_name="all-MiniLM-L6-v2",
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
    
    # Limit chunks if max_chunks is specified (for memory management)
    if max_chunks and len(chunks) > max_chunks:
        logger.warning(f"Limiting to {max_chunks} chunks to manage memory usage")
        # Prioritize chunks from different documents for diversity
        chunks = chunks[:max_chunks]
    
    return chunks

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
    Generate relevant questions from document content using Ollama API or fallback to rule-based.
    
    Args:
        content: Document content text
        num_questions: Number of questions to generate
        
    Returns:
        List of generated questions
    """
    # First check if LLM is available
    llm_available = check_llm_availability()
    
    if not llm_available:
        logger.warning("LLM not available. Using rule-based question generation.")
        return rule_based_question_generation(content, num_questions)
    
    import requests
    import json
    
    # Truncate content if too long
    max_content_length = 1000  # Reduced from 2000 to help prevent timeouts with large models
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."
    
    prompt = f"""
    Given the following content, generate {num_questions} specific, relevant questions that could be asked about this information. 
    Make sure the questions are clear, focused, and can be answered using the provided content.
    
    CONTENT:
    {content}
    
    QUESTIONS (provide exactly {num_questions} questions, numbered 1-{num_questions}):
    """
    
    try:
        # Call Ollama API to generate questions
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            },
            timeout=300  # Increased timeout for large models (was 120)
        )
        
        if response.status_code != 200:
            logger.error(f"Error calling Ollama API: {response.text}")
            return rule_based_question_generation(content, num_questions)
        
        result = response.json()
        generated_text = result.get("response", "")
        
        # Parse questions from response
        questions = []
        for line in generated_text.split('\n'):
            line = line.strip()
            # Extract questions that contain a question mark
            if '?' in line and (line[0].isdigit() or line.lower().startswith("question")):
                # Remove any numbering or "Question:" prefix
                question = line.split(".", 1)[-1].split(":", 1)[-1].strip()
                questions.append(question)
        
        if not questions:
            # If no questions were successfully parsed, look for lines with question marks
            questions = [line.strip() for line in generated_text.split('\n') 
                        if '?' in line and len(line.strip()) > 10]
        
        # Ensure we have the requested number of questions
        if not questions or len(questions) < num_questions:
            # If LLM didn't generate enough questions, add some from rule-based approach
            rule_questions = rule_based_question_generation(content, num_questions - len(questions))
            questions.extend(rule_questions)
            
        return questions[:num_questions]
        
    except Exception as e:
        logger.error(f"Error generating questions with LLM: {e}")
        # Fallback to rule-based generation
        return rule_based_question_generation(content, num_questions)

def enhance_answer(content: str, question: str) -> str:
    """
    Enhance document content to create a more structured answer to the question.
    
    Args:
        content: Original document content
        question: The question being asked
        
    Returns:
        Enhanced answer text
    """
    # First check if LLM is available
    llm_available = check_llm_availability()
    
    if not llm_available:
        logger.warning("LLM not available. Returning original content as answer.")
        # Try to extract the most relevant parts of the content as the answer
        sentences = content.split('.')
        relevant_sentences = []
        
        # Look for sentences that might contain keywords from the question
        question_keywords = set(re.findall(r'\b\w{4,}\b', question.lower()))
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in question_keywords):
                relevant_sentences.append(sentence)
        
        # If we found relevant sentences, join them; otherwise return the original content
        if relevant_sentences:
            return '. '.join(relevant_sentences[:5]) + '.'
        return content
    
    import requests
    
    # Truncate content if too long
    max_content_length = 1000  # Reduced from 2000 to help prevent timeouts
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."
    
    prompt = f"""
    Given the following QUESTION and CONTENT from a document, create a well-structured, 
    comprehensive answer that directly addresses the question using only information from the content.
    
    QUESTION: {question}
    
    CONTENT:
    {content}
    
    ANSWER:
    """
    
    try:
        # Call Ollama API to generate enhanced answer
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": settings.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more factual responses
                    "top_p": 0.9,
                }
            },
            timeout=360  # Increased timeout for large models (was 180)
        )
        
        if response.status_code != 200:
            logger.error(f"Error calling Ollama API: {response.text}")
            return content
        
        result = response.json()
        enhanced_answer = result.get("response", "").strip()
        
        if enhanced_answer:
            return enhanced_answer
        return content
        
    except Exception as e:
        logger.error(f"Error enhancing answer: {e}")
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
    
    # Update global progress counters
    with PROGRESS_LOCK:
        CHUNKS_PROCESSED += 1
        CURRENT_CHUNK = i
        # Call progress callback with the values directly
        if PROGRESS_CALLBACK:
            try:
                PROGRESS_CALLBACK(i, CHUNKS_PROCESSED)
            except Exception as e:
                # If callback errors, just log it but continue processing
                logger.warning(f"Progress callback error for chunk {i}: {e}")
    
    try:
        # Extract content from document
        content = doc.page_content
        source = doc.metadata.get("source", "unknown")
        topic = doc.metadata.get("topic", "unknown")
        
        # Generate questions based on the content
        questions = generate_questions_from_content(content, questions_per_chunk)
        
        # Create conversation for each question
        for q in questions:
            if STOP_PROCESSING.is_set():
                break
                
            # Create the answer
            if enhance_answers:
                answer = enhance_answer(content, q)
            else:
                answer = content
            
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
        
        logger.info(f"Generated {len(conversations)} conversations from chunk {i}")
        return conversations
        
    except Exception as e:
        logger.error(f"Error processing chunk {i}: {e}")
        return []

def convert_to_sharegpt_format(
    documents: List[Document], 
    system_prompt: Optional[str] = None,
    questions_per_chunk: int = 2,
    enhance_answers: bool = True,
    max_workers: int = 4,
    batch_size: int = None,
    progress_callback: Callable = None,
) -> List[Dict]:
    """
    Convert documents to ShareGPT format by generating questions and answers.
    
    Args:
        documents: List of Document objects to process
        system_prompt: Optional system prompt to set context
        questions_per_chunk: Number of questions to generate per document
        enhance_answers: Whether to enhance answers with LLM
        max_workers: Maximum number of parallel workers
        batch_size: Maximum number of documents to process at once
        progress_callback: Optional callback function to report progress
        
    Returns:
        List of conversation objects in ShareGPT format
    """
    global PROGRESS_CALLBACK, STOP_PROCESSING, CHUNKS_PROCESSED, CURRENT_CHUNK
    
    if not documents:
        logger.warning("No documents to convert")
        return []
    
    # Set progress callback if provided
    PROGRESS_CALLBACK = progress_callback
    
    # Reset progress tracking
    STOP_PROCESSING.clear()
    CHUNKS_PROCESSED = 0
    CURRENT_CHUNK = 0
    
    # Set batch size if specified
    if batch_size:
        documents = documents[:batch_size]
    
    # Create tasks for parallel processing
    tasks = [
        (i, doc, system_prompt, questions_per_chunk, enhance_answers)
        for i, doc in enumerate(documents)
    ]
    
    logger.info(f"Converting {len(documents)} documents to ShareGPT format with {max_workers} workers")
    
    # Define a wrapper for error handling
    def safe_process(chunk_data):
        try:
            return process_document_chunk(chunk_data)
        except Exception as e:
            logger.error(f"Error in worker thread: {e}")
            return []
    
    # Process documents in parallel
    results = []
    
    # Adjust workers count if fewer documents than workers
    actual_workers = min(max_workers, len(documents))
    
    try:
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(safe_process, task): task for task in tasks}
            
            # Process results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    conversations = future.result()
                    results.extend(conversations)
                except Exception as e:
                    logger.error(f"Task {task[0]} generated an exception: {e}")
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        STOP_PROCESSING.set()
    
    # Sort results by source and ID for consistency
    results.sort(key=lambda x: (x.get("source", ""), x.get("id", "")))
    
    logger.info(f"Generated {len(results)} conversations in ShareGPT format")
    return results

def main():
    """Main function to convert documents to ShareGPT format."""
    parser = argparse.ArgumentParser(
        description="Convert documents to ShareGPT format for fine-tuning"
    )
    parser.add_argument(
        "--source_dir", "-s",
        type=str,
        default=os.path.join(DATA_DIR, "raw"),
        help="Directory containing documents to convert"
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
        default=1024,
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
    
    args = parser.parse_args()
    
    # Check if LLM is available
    llm_status = "available" if check_llm_availability() else "unavailable"
    logger.info(f"LLM service status: {llm_status}")
    
    # Load documents
    documents = load_documents(args.source_dir)
    if not documents:
        logger.error("No documents found. Exiting.")
        sys.exit(1)
    
    # Split documents into chunks
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap, args.max_chunks)
    
    # Convert to ShareGPT format
    sharegpt_data = convert_to_sharegpt_format(
        chunks,
        system_prompt=args.system_prompt,
        questions_per_chunk=args.questions_per_chunk,
        enhance_answers=args.enhance_answers,
        max_workers=args.max_workers,
        batch_size=args.batch_size
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Write to output file
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(sharegpt_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Written {len(sharegpt_data)} conversations to {args.output_file}")

if __name__ == "__main__":
    main() 