import os
import sys
from pathlib import Path

# Add the project root to Python path to import project modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Fix the import path for HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Import our project modules
from core.settings import settings

def initialize_vector_db():
    """Create a minimal vector DB so the RAG server can start."""
    print("Initializing vector database...")
    # Create the directory if it doesn't exist
    os.makedirs(settings.chroma_persist_directory, exist_ok=True)

    # First check if there are already documents in the DB
    try:
        # Initialize the embedding function
        embedding_function = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # Try to open the existing DB
        vector_store = Chroma(
            persist_directory=settings.chroma_persist_directory,
            embedding_function=embedding_function
        )
        
        # Check if there are documents
        count = vector_store._collection.count()
        if count > 0:
            print(f"Vector database already contains {count} documents. No need to initialize.")
            return
    except Exception as e:
        print(f"Could not check existing vector database: {e}")
        # Continue with initialization

    # Create a sample document
    sample_docs = [
        Document(
            page_content="This is a sample document for testing RAG functionality.",
            metadata={"source": "sample", "topic": "test"}
        )
    ]

    # Initialize the embedding function
    embedding_function = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True}
    )

    # Create the vector store - persist_directory automatically saves
    # No need to call persist() separately anymore
    vector_store = Chroma.from_documents(
        documents=sample_docs,
        embedding=embedding_function,
        persist_directory=settings.chroma_persist_directory
    )
    
    # Don't call persist() - it's not needed anymore
    print(f"Vector database initialized at {settings.chroma_persist_directory} with 1 sample document")

if __name__ == "__main__":
    initialize_vector_db()
    print("Done! RAG server should now work.")
