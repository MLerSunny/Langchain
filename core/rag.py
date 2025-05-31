import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from core.settings import settings
from core.exceptions import RAGError

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        """Initialize the RAG engine."""
        try:
            # Initialize embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.get("embeddings.model"),
                model_kwargs={
                    "device": "cuda" if settings.get("optimization.use_gpu", False) else "cpu",
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
            model_name = settings.get("llm.model")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_safetensors=True,
                device_map="auto",
                load_in_8bit=True,  # Enable 8-bit quantization
                max_memory={0: "3GB"}  # Limit GPU memory usage
            )
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=settings.get("llm.max_tokens", 1024),
                temperature=settings.get("llm.temperature", 0.7),
                top_p=settings.get("llm.top_p", 0.95),
                repetition_penalty=settings.get("llm.repetition_penalty", 1.15),
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
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.get("chunking.chunk_size", 1000),
                chunk_overlap=settings.get("chunking.chunk_overlap", 200)
            )
            texts = text_splitter.split_documents(documents)
            
            # Create or update vector store
            if self.vector_store is None:
                self.vector_store = Chroma.from_documents(
                    texts,
                    self.embeddings,
                    persist_directory=settings.get("vector_store.persist_directory"),
                    collection_name=settings.get("vector_store.collection_name", "documents")
                )
            else:
                self.vector_store.add_documents(texts)
                
            # Update QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_kwargs={"k": settings.get("retrieval.max_results", 3)}
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
            docs = self.vector_store.similarity_search(
                query,
                k=settings.get("retrieval.max_results", 3)
            )
            sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            
            # Generate response
            response = self.qa_chain.run(query)
            
            return response, sources
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise RAGError(f"Failed to process query: {str(e)}") 