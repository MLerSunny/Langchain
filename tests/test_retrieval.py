"""
Tests for the retrieval process, including ingest idempotency, 
retrieval latency, and citation integrity.
"""

import os
import sys
import time
import shutil
import unittest
import tempfile
import pytest
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from ingest.bulk_ingest import load_documents, split_documents, extract_metadata_from_path

# Test constants
PERSIST_DIR = "test_chroma_db"
TEST_DATA_DIR = "data/raw"
COLLECTION_NAME = "test_insurance_docs"
MAX_RETRIEVAL_LATENCY_MS = 50  # Maximum acceptable latency in milliseconds

class TestRetrieval:
    """Test cases for document retrieval functionality."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        # Initialize vector store
        self.embeddings = OpenAIEmbeddings()
        self.db = Chroma(
            collection_name="test_collection",
            embedding_function=self.embeddings,
            persist_directory="test_db"
        )
        
        # Initialize retriever
        self.retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 5,
                "lambda_mult": 0.7
            }
        )
        
        # Add test documents
        test_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt"}),
            Document(page_content="Test document 2", metadata={"source": "test2.txt"}),
            Document(page_content="Test document 3", metadata={"source": "test3.txt"})
        ]
        self.db.add_documents(test_docs)
        
        yield
        
        # Cleanup
        if os.path.exists("test_db"):
            import shutil
            shutil.rmtree("test_db")
    
    def split_documents(self, documents):
        """Split documents into chunks."""
        if not documents:
            raise ValueError("No documents provided")
        
        chunks = []
        for doc in documents:
            # Simple splitting by sentences
            sentences = doc.page_content.split('.')
            for sentence in sentences:
                if sentence.strip():
                    chunks.append(Document(
                        page_content=sentence.strip(),
                        metadata=doc.metadata
                    ))
        return chunks
    
    def filter_documents(self, documents, metadata_filter):
        """Filter documents by metadata."""
        if not documents:
            return []
        
        filtered = []
        for doc in documents:
            matches = True
            for key, value in metadata_filter.items():
                if doc.metadata.get(key) != value:
                    matches = False
                    break
            if matches:
                filtered.append(doc)
        return filtered
    
    def get_relevant_documents(self, query):
        """Get relevant documents for a query."""
        return self.retriever.get_relevant_documents(query)
    
    @classmethod
    def setup_class(cls):
        """Set up test environment once before all tests."""
        # Initialize the embedding model
        cls.embedding_model = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            deployment="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            chunk_size=1000
        )
        
        # Make sure test data exists
        assert os.path.exists(TEST_DATA_DIR), f"Test data directory {TEST_DATA_DIR} does not exist"
        
        # Create a clean test DB for each test run
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        
        # Sample queries for latency testing
        cls.test_queries = [
            "What does comprehensive auto insurance cover?",
            "What is the deductible for collision coverage?",
            "Are rental cars covered in my policy?",
            "What are the exclusions in my policy?",
            "How is my premium calculated?",
            "What is the good driver discount?",
            "What are the liability limits?",
            "How does uninsured motorist coverage work?",
            "What happens if my car is totaled?",
            "Do I need special coverage for a new car?"
        ]
    
    @classmethod
    def teardown_class(cls):
        """Clean up after all tests are done."""
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
    
    def setup_method(self):
        """Set up test environment before each test method."""
        self.test_docs = [
            Document(page_content="Test document 1", metadata={"source": "test1.txt", "topic": "test"}),
            Document(page_content="Test document 2", metadata={"source": "test2.txt", "topic": "test"}),
            Document(page_content="Test document 3", metadata={"source": "test3.txt", "topic": "test"})
        ]
        self.chunks = []
        self.retriever = None
    
    def teardown_method(self):
        """Clean up after each test method."""
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
    
    def test_ingest_idempotency(self):
        """
        Test 1: Verify that running ingest multiple times produces consistent results.
        The vector store should maintain integrity and not duplicate documents.
        """
        # Get initial document count and IDs
        initial_count = self.db._collection.count()
        initial_ids = set(self.db._collection.get()["ids"])
        
        # Run ingest process again with same data
        chunks2 = self.split_documents(self.test_docs)
        db2 = Chroma.from_documents(
            documents=chunks2,
            embedding=self.embedding_model,
            persist_directory=PERSIST_DIR,
            collection_name=f"{COLLECTION_NAME}_2"
        )
        db2.persist()
        
        # Get count and IDs after second ingest
        second_count = db2._collection.count()
        second_ids = set(db2._collection.get()["ids"])
        
        # The counts should be the same
        assert initial_count == len(self.chunks), "Initial document count doesn't match chunk count"
        assert second_count == len(chunks2), "Second document count doesn't match chunk count"
        assert len(initial_ids) == initial_count, "Some document IDs are duplicated in initial ingest"
        assert len(second_ids) == second_count, "Some document IDs are duplicated in second ingest"
        
        # The content should be consistent (same number of documents)
        assert len(chunks2) == len(self.chunks), "Second ingest produced different number of chunks"
        
        # Rebuild check
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        
        # Create a new DB
        db3 = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embedding_model,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
        db3.persist()
        
        # Verify rebuild produces same count
        rebuild_count = db3._collection.count()
        assert rebuild_count == initial_count, "Rebuilt database has different document count"
    
    def test_retrieval_latency(self):
        """
        Test 2: Verify that retrieval latency is below the maximum threshold.
        Each query should complete in under 50ms on average.
        """
        # Measure retrieval time for each query
        latencies = []
        for query in self.test_queries:
            start_time = time.time()
            docs = self.get_relevant_documents(query)
            end_time = time.time()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Make sure we got some results
            assert len(docs) > 0, f"No documents retrieved for query: {query}"
        
        # Calculate average latency
        avg_latency = sum(latencies) / len(latencies)
        max_latency = max(latencies)
        
        # Print latencies for debugging
        print(f"Average retrieval latency: {avg_latency:.2f}ms")
        print(f"Maximum retrieval latency: {max_latency:.2f}ms")
        print(f"Individual latencies (ms): {[round(lat, 2) for lat in latencies]}")
        
        # Skip strict timing assertion in CI environments or slower systems
        if not os.environ.get("CI"):
            assert avg_latency < MAX_RETRIEVAL_LATENCY_MS, f"Average retrieval latency ({avg_latency:.2f}ms) exceeds threshold ({MAX_RETRIEVAL_LATENCY_MS}ms)"
    
    def test_citation_integrity(self):
        """
        Test 3: Verify that citations refer to actual documents and contain correct content.
        Citations should match the original sources.
        """
        # Get documents for a specific query
        query = "What is the deductible for comprehensive coverage?"
        docs = self.get_relevant_documents(query)
        
        # Extract citation sources
        sources = [doc.metadata.get("source", "unknown") for doc in docs]
        
        # Check that sources exist
        for source in sources:
            assert os.path.exists(source), f"Citation source does not exist: {source}"
        
        # Check content integrity - look for specific text from citations in original files
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            if os.path.exists(source):
                with open(source, "r", encoding="utf-8") as f:
                    original_content = f.read()
                
                # Get a snippet of the document content
                snippet = doc.page_content[:50]  # First 50 chars as a sample
                
                # The snippet should be in the original document
                # We lower() both to handle case differences in chunking
                assert snippet.lower() in original_content.lower(), f"Citation content doesn't match original document. Snippet: {snippet}"
    
    def test_empty_document_handling(self):
        """Test handling of empty documents."""
        empty_doc = Document(page_content="", metadata={"source": "empty.txt"})
        chunks = self.split_documents([empty_doc])
        assert len(chunks) == 0, "Empty documents should not produce chunks"
    
    def test_large_document_handling(self):
        """Test handling of very large documents."""
        large_content = "Test content " * 10000  # Create a large document
        large_doc = Document(page_content=large_content, metadata={"source": "large.txt"})
        chunks = self.split_documents([large_doc])
        assert len(chunks) > 0, "Large documents should be split into chunks"
        assert all(len(chunk.page_content) <= self.retriever.chunk_size for chunk in chunks), \
            "All chunks should be within size limit"
    
    def test_special_characters(self):
        """Test handling of documents with special characters."""
        special_chars = "!@#$%^&*()_+{}|:<>?[]\\;',./~`"
        doc = Document(page_content=special_chars, metadata={"source": "special.txt"})
        chunks = self.split_documents([doc])
        assert len(chunks) > 0, "Documents with special characters should be processed"
    
    def test_unicode_characters(self):
        """Test handling of documents with Unicode characters."""
        unicode_content = "Test content with Unicode: 你好世界 🌍"
        doc = Document(page_content=unicode_content, metadata={"source": "unicode.txt"})
        chunks = self.split_documents([doc])
        assert len(chunks) > 0, "Documents with Unicode characters should be processed"
        assert unicode_content in chunks[0].page_content, "Unicode content should be preserved"
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved during chunking."""
        metadata = {
            "source": "test.txt",
            "topic": "test",
            "custom_field": "custom_value"
        }
        doc = Document(page_content="Test content", metadata=metadata)
        chunks = self.split_documents([doc])
        assert all(chunk.metadata == metadata for chunk in chunks), \
            "Metadata should be preserved in all chunks"
    
    def test_chunk_overlap(self):
        """Test that chunk overlap is working correctly."""
        content = "This is a test document that should be split into multiple chunks with overlap."
        doc = Document(page_content=content, metadata={"source": "overlap.txt"})
        chunks = self.split_documents([doc])
        
        if len(chunks) > 1:
            # Check that consecutive chunks have some overlap
            for i in range(len(chunks) - 1):
                overlap = set(chunks[i].page_content.split()) & set(chunks[i+1].page_content.split())
                assert len(overlap) > 0, "Consecutive chunks should have some overlap"
    
    def test_retrieval_consistency(self):
        """Test that retrieval returns consistent results for the same query."""
        query = "test query"
        results1 = self.get_relevant_documents(query)
        results2 = self.get_relevant_documents(query)
        
        # Compare document IDs or content
        assert [r.page_content for r in results1] == [r.page_content for r in results2], \
            "Retrieval should be consistent for the same query"
    
    def test_retrieval_relevance(self):
        """Test that retrieved documents are relevant to the query."""
        query = "specific test content"
        results = self.get_relevant_documents(query)
        
        # Check that results contain query terms
        query_terms = set(query.lower().split())
        for result in results:
            content_terms = set(result.page_content.lower().split())
            assert len(query_terms & content_terms) > 0, \
                "Retrieved documents should contain query terms"
    
    def test_error_handling(self):
        """Test error handling for various edge cases."""
        # Test with None input
        with pytest.raises(ValueError):
            self.split_documents(None)
        
        # Test with invalid document
        with pytest.raises(ValueError):
            self.split_documents([{"invalid": "document"}])
        
        # Test with invalid chunk size
        with pytest.raises(ValueError):
            self.retriever.chunk_size = -1
            self.split_documents(self.test_docs)
    
    def test_performance(self):
        """Test retrieval performance with larger datasets."""
        # Create a larger test dataset
        large_docs = [
            Document(page_content=f"Test document {i}", metadata={"source": f"test{i}.txt"})
            for i in range(100)
        ]
        
        start_time = time.time()
        chunks = self.split_documents(large_docs)
        processing_time = time.time() - start_time
        
        assert processing_time < 5.0, "Document processing should complete within 5 seconds"
        assert len(chunks) > 0, "Should process large datasets successfully"
    
    def test_metadata_filtering(self):
        """Test filtering documents by metadata."""
        # Create documents with different metadata
        docs = [
            Document(page_content="Content 1", metadata={"category": "A", "priority": "high"}),
            Document(page_content="Content 2", metadata={"category": "B", "priority": "low"}),
            Document(page_content="Content 3", metadata={"category": "A", "priority": "medium"})
        ]
        
        # Test filtering by single metadata field
        filtered_docs = self.filter_documents(docs, {"category": "A"})
        assert len(filtered_docs) == 2, "Should filter by single metadata field"
        
        # Test filtering by multiple metadata fields
        filtered_docs = self.filter_documents(docs, {"category": "A", "priority": "high"})
        assert len(filtered_docs) == 1, "Should filter by multiple metadata fields"
        
        # Test filtering with non-existent metadata
        filtered_docs = self.filter_documents(docs, {"nonexistent": "value"})
        assert len(filtered_docs) == 0, "Should handle non-existent metadata fields"
    
    def test_section_preservation(self):
        """Test that important sections are preserved during chunking."""
        # Create a document with clear sections
        content = """
        SECTION 1: Introduction
        This is the introduction section.
        
        SECTION 2: Main Content
        This is the main content section.
        
        SECTION 3: Conclusion
        This is the conclusion section.
        """
        doc = Document(page_content=content, metadata={"source": "sections.txt"})
        
        # Split into chunks
        chunks = self.split_documents([doc])
        
        # Define important sections to check
        important_sections = ["SECTION 1", "SECTION 2", "SECTION 3"]
        
        # Check which sections are preserved
        preserved_sections = []
        missing_sections = []
        
        for section in important_sections:
            found = False
            for chunk in chunks:
                if section in chunk.page_content:
                    preserved_sections.append(section)
                    found = True
                    break
            if not found:
                missing_sections.append(section)
        
        # Calculate preservation percentage
        preservation_rate = len(preserved_sections) / len(important_sections) * 100
        
        # Print preservation statistics
        print(f"Section preservation rate: {preservation_rate:.1f}%")
        print(f"Preserved sections: {preserved_sections}")
        if missing_sections:
            print(f"Missing sections: {missing_sections}")
        
        # Assert that at least 80% of important sections are preserved
        assert preservation_rate >= 80, f"Section preservation rate too low: {preservation_rate:.1f}%"
        
        # Check for section integrity in individual chunks
        section_in_chunks = False
        for chunk in chunks:
            if "SECTION" in chunk.page_content:
                section_in_chunks = True
                # Make sure the section has some content
                section_pos = chunk.page_content.find("SECTION")
                assert len(chunk.page_content) > section_pos + 20, "Section header is cut off without content"
        
        assert section_in_chunks, "No section headers found in any chunks" 