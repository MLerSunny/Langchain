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
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        
        if settings.model_name in model_names:
            logger.info(f"LLM model {settings.model_name} is available")
            LLM_AVAILABLE = True
            return True
        else:
            logger.warning(f"Model {settings.model_name} not found in available models: {model_names}")
            
            # If we have any model available, we can still use it
            if models:
                logger.info(f"Will use available model: {models[0]['name']} instead")
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
    max_content_length = 4000
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
            timeout=30  # Set a timeout to avoid hanging
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
            timeout=45  # Longer timeout for answer generation
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
    i, doc, system_prompt, questions_per_chunk, enhance_answers = chunk_data
    conversations = []
    
    logger.info(f"Processing document chunk {i}")
    
    # Generate questions from document content
    questions = generate_questions_from_content(doc.page_content, questions_per_chunk)
    
    for question in questions:
        # Create conversation object
        conversation = {"conversations": []}
        
        # Add system prompt
        conversation["conversations"].append({
            "from": "system",
            "value": system_prompt
        })
        
        # Add user question
        conversation["conversations"].append({
            "from": "human",
            "value": question
        })
        
        # Add assistant response
        answer = doc.page_content
        if enhance_answers:
            answer = enhance_answer(doc.page_content, question)
        
        conversation["conversations"].append({
            "from": "assistant",
            "value": answer
        })
        
        conversations.append(conversation)
    
    return conversations

def convert_to_sharegpt_format(
    documents: List[Document], 
    system_prompt: Optional[str] = None,
    questions_per_chunk: int = 2,
    enhance_answers: bool = True,
    max_workers: int = 4,
    batch_size: int = None
) -> List[Dict]:
    """
    Convert document chunks to ShareGPT format for fine-tuning.
    
    Args:
        documents: List of document chunks
        system_prompt: Optional system prompt to include
        questions_per_chunk: Number of questions to generate per chunk
        enhance_answers: Whether to enhance answers using LLM
        max_workers: Maximum number of parallel workers
        batch_size: Limit processing to this many chunks (for memory management)
        
    Returns:
        List of conversation objects in ShareGPT format
    """
    if not system_prompt:
        # Load domain-specific system prompts
        system_prompts = {
            "insurance": """You are an insurance expert assistant. You provide accurate, helpful information about insurance policies, coverages, and claims. Answer questions based on your knowledge of insurance best practices and regulations.""",
            "finance": """You are a finance expert assistant. You provide accurate, helpful information about financial planning, investments, and money management. Answer questions based on your knowledge of financial best practices and regulations.""",
            "healthcare": """You are a healthcare expert assistant. You provide accurate, helpful information about medical conditions, treatments, and healthcare systems. Answer questions based on your knowledge of healthcare best practices.""",
            "legal": """You are a legal expert assistant. You provide accurate, helpful information about laws, regulations, and legal processes. Answer questions based on your knowledge of legal best practices.""",
            "technology": """You are a technology expert assistant. You provide accurate, helpful information about software, hardware, and digital services. Answer questions based on your knowledge of technology best practices.""",
            "general": """You are a helpful assistant that provides accurate, informative responses based on the content provided. Answer questions factually using only the information available in the provided content."""
        }
        
        # Try to detect document domain from content or metadata
        domain = "general"
        if documents:
            sample_text = documents[0].page_content.lower()
            sample_topic = documents[0].metadata.get("topic", "").lower()
            
            # Check for domain keywords in content or metadata
            for key in system_prompts.keys():
                if key in sample_text or key in sample_topic:
                    domain = key
                    break
            
            logger.info(f"Detected document domain: {domain}")
        
        system_prompt = system_prompts[domain]
    
    sharegpt_data = []
    
    # Apply memory management - limit the number of chunks to process
    if batch_size and batch_size < len(documents):
        logger.warning(f"Limiting processing to {batch_size} chunks for memory management")
        # Select a diverse sample from the documents
        documents = documents[:batch_size]
    
    # Prepare data for parallel processing
    chunk_data = [
        (i, doc, system_prompt, questions_per_chunk, enhance_answers) 
        for i, doc in enumerate(documents)
    ]
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(process_document_chunk, data): data 
            for data in chunk_data
        }
        
        for future in as_completed(future_to_chunk):
            try:
                conversations = future.result()
                sharegpt_data.extend(conversations)
            except Exception as e:
                i = future_to_chunk[future][0]
                logger.error(f"Error processing chunk {i}: {e}")
    
    logger.info(f"Created {len(sharegpt_data)} conversation examples")
    return sharegpt_data

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