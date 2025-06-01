import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

from core.settings import settings
from core.exceptions import RAGError

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine."""
        try:
            # Get embedding model name with fallback
            embedding_model = settings.get("embeddings", {}).get("model", "sentence-transformers/all-MiniLM-L6-v2")
            if not embedding_model:
                raise RAGError("Embedding model not specified in configuration")
            
            logger.info(f"Initializing embeddings with model: {embedding_model}")
            
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={
                    "device": "cuda" if settings.get("optimization", {}).get("use_gpu", False) else "cpu",
                    "trust_remote_code": True
                },
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # Initialize components
            self.vector_store = None
            self.qa_chain = None
            self._initialize_llm()
            
        except Exception as e:
            logger.error(f"Error initializing RAG engine: {str(e)}")
            raise RAGError(f"Failed to initialize RAG engine: {str(e)}")
        
    def _initialize_llm(self):
        """Initialize the language model."""
        try:
            llm_config = settings.get("llm", {})
            model_name = llm_config.get("model", "microsoft/phi-2")
            if not model_name:
                raise RAGError("LLM model not specified in configuration")
                
            logger.info(f"Initializing LLM with model: {model_name}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_TOKEN"),
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=os.getenv("HUGGINGFACE_TOKEN"),
                use_safetensors=True,
                device_map="auto",
                load_in_8bit=True,  # Enable 8-bit quantization
                max_memory={0: "3GB"},  # Limit GPU memory usage
                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading for 32-bit modules
                torch_dtype=torch.float16,  # Use float16 for better memory efficiency
                trust_remote_code=True
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=llm_config.get("max_tokens", 1024),
                temperature=llm_config.get("temperature", 0.7),
                top_p=llm_config.get("top_p", 0.95),
                repetition_penalty=llm_config.get("repetition_penalty", 1.15),
                device_map="auto"
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            raise RAGError(f"Failed to initialize LLM: {str(e)}")
            
    def add_documents(self, documents: List[str]):
        """Add documents to the vector store."""
        try:
            # Split documents
            chunking_config = settings.get("chunking", {})
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunking_config.get("chunk_size", 1000),
                chunk_overlap=chunking_config.get("chunk_overlap", 200)
            )
            texts = text_splitter.split_documents(documents)
            
            # Create or update vector store
            vector_store_config = settings.get("vector_store", {})
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    texts,
                    self.embeddings,
                    persist_directory=vector_store_config.get("persist_directory"),
                    collection_name=vector_store_config.get("collection_name", "documents")
                )
            else:
                self.vector_store.add_documents(texts)
                
            # Update QA chain
            retrieval_config = settings.get("retrieval", {})
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": retrieval_config.get("max_results", 3)}
                )
            )
            
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
            raise RAGError(f"Failed to add documents: {str(e)}")
            
    def process_query(
        self,
        query: str,
        context: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """Process a query through the RAG system."""
        try:
            if self.qa_chain is None:
                raise RAGError("No documents have been added to the system")
                
            # Get relevant documents
            retrieval_config = settings.get("retrieval", {})
            docs = self.vector_store.similarity_search(
                query,
                k=retrieval_config.get("max_results", 3)
            )
            sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            
            # Generate response
            response = self.qa_chain.run(query)
            
            return response, sources
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise RAGError(f"Failed to process query: {str(e)}") 