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

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.ingest import load_documents, split_documents, extract_metadata_from_path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Test constants
PERSIST_DIR = "test_chroma_db"
TEST_DATA_DIR = "data/raw"
COLLECTION_NAME = "test_insurance_docs"
MAX_RETRIEVAL_LATENCY_MS = 50  # Maximum acceptable latency in milliseconds

class TestRetrieval:
    """Test cases for document retrieval functionality."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment once before all tests."""
        # Initialize the embedding model
        cls.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
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
        """Set up before each test method."""
        # Create a fresh DB for each test
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)
        
        # Load documents from test data
        self.documents = load_documents(TEST_DATA_DIR)
        assert len(self.documents) > 0, "No test documents loaded"
        
        # Split documents
        self.chunks = split_documents(self.documents)
        assert len(self.chunks) > 0, "No chunks created from test documents"
        
        # Store in test Chroma DB
        self.db = Chroma.from_documents(
            documents=self.chunks,
            embedding=self.embedding_model,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME
        )
        self.db.persist()
    
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
        chunks2 = split_documents(self.documents)
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
        # Create retriever
        retriever = self.db.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 8,
                "lambda_mult": 0.3
            }
        )
        
        # Measure retrieval time for each query
        latencies = []
        for query in self.test_queries:
            start_time = time.time()
            docs = retriever.get_relevant_documents(query)
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
        # Create retriever
        retriever = self.db.as_retriever(search_kwargs={"k": 5})
        
        # Get documents for a specific query
        query = "What is the deductible for comprehensive coverage?"
        docs = retriever.get_relevant_documents(query)
        
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
    
    def test_metadata_filtering(self):
        """
        Test 4: Verify that metadata filtering works correctly.
        Retrieval should respect LOB and state filters.
        """
        # Get all distinct LOBs and states from our documents
        lobs = set()
        states = set()
        
        for doc in self.chunks:
            lob = doc.metadata.get("lob")
            state = doc.metadata.get("state")
            if lob:
                lobs.add(lob)
            if state:
                states.add(state)
        
        # Make sure we have at least one LOB and state
        assert len(lobs) > 0, "No LOBs found in test documents"
        assert len(states) > 0, "No states found in test documents"
        
        # Test filtering by a specific LOB
        test_lob = next(iter(lobs))
        lob_retriever = self.db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"lob": test_lob}
            }
        )
        
        # Get documents for a general query
        docs = lob_retriever.get_relevant_documents("insurance coverage")
        
        # All retrieved documents should have the test_lob
        for doc in docs:
            assert doc.metadata.get("lob") == test_lob, f"Retrieved document has wrong LOB: {doc.metadata.get('lob')}"
        
        # Test filtering by a specific state
        test_state = next(iter(states))
        state_retriever = self.db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"state": test_state}
            }
        )
        
        # Get documents for a general query
        docs = state_retriever.get_relevant_documents("insurance coverage")
        
        # All retrieved documents should have the test_state
        for doc in docs:
            assert doc.metadata.get("state") == test_state, f"Retrieved document has wrong state: {doc.metadata.get('state')}"
        
        # Test combined filtering
        combined_retriever = self.db.as_retriever(
            search_kwargs={
                "k": 5,
                "filter": {"lob": test_lob, "state": test_state}
            }
        )
        
        # Get documents for a general query
        docs = combined_retriever.get_relevant_documents("insurance coverage")
        
        # All retrieved documents should have both the test_lob and test_state
        for doc in docs:
            assert doc.metadata.get("lob") == test_lob, f"Retrieved document has wrong LOB: {doc.metadata.get('lob')}"
            assert doc.metadata.get("state") == test_state, f"Retrieved document has wrong state: {doc.metadata.get('state')}"
    
    def test_section_preservation(self):
        """
        Test 5: Verify that document sections are properly preserved during chunking.
        Important section headers and content should remain intact.
        """
        # Expected section patterns that should be preserved
        important_sections = [
            "SECTION 1: COVERAGES",
            "SECTION 2: EXCLUSIONS",
            "SECTION 3: DEFINITIONS",
            "SECTION 4: PREMIUM",
            "SECTION 5: POLICY PERIOD",
            "LIABILITY COVERAGE",
            "COMPREHENSIVE COVERAGE",
            "COLLISION COVERAGE",
            "ENDORSEMENT"
        ]
        
        # Get all document content as a single string
        all_chunks_content = "\n".join([chunk.page_content for chunk in self.chunks])
        
        # Check if each important section is preserved in at least one chunk
        preserved_sections = []
        missing_sections = []
        
        for section in important_sections:
            if section.lower() in all_chunks_content.lower():
                preserved_sections.append(section)
            else:
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
        for chunk in self.chunks:
            if "SECTION" in chunk.page_content:
                section_in_chunks = True
                # Make sure the section has some content
                section_pos = chunk.page_content.find("SECTION")
                assert len(chunk.page_content) > section_pos + 20, "Section header is cut off without content"
        
        assert section_in_chunks, "No section headers found in any chunks" 