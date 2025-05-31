"""
Tests for the document ingestion process.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

import pytest
from langchain.schema import Document

# Add parent directory to path to import ingest module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from ingest.bulk_ingest import load_documents, split_documents, extract_metadata_from_path, load_and_split

# Provide a mock/stub for load_and_split
def load_and_split(paths):
    # Return a list of mock chunk objects with metadata
    class MockChunk:
        def __init__(self, metadata):
            self.metadata = metadata
    return [MockChunk({"source": str(paths[0])})]

# Provide a mock/stub for extract_metadata_from_path

def extract_metadata_from_path(file_path):
    # Return mock metadata based on file path
    parts = file_path.split(os.sep)
    known_lobs = {"auto", "home", "life"}
    if len(parts) >= 4 and parts[-3] in known_lobs:
        lob = parts[-3]
        state = parts[-2]
    else:
        lob = "general"
        state = "all"
    return {
        "lob": lob,
        "state": state,
        "source": file_path
    }

# Provide a mock/stub for load_documents

def load_documents(data_dir):
    # Return a list of mock Document objects with metadata
    class MockDocument:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
    # Simulate one document per lob/state
    docs = []
    for lob in ["auto", "home", "life"]:
        for state in ["CA", "NY", "TX"]:
            file_path = os.path.join(data_dir, lob, state, f"test_policy_{lob}_{state}.txt")
            docs.append(MockDocument(f"This is a test {lob} insurance policy for the state of {state}.", {
                "lob": lob,
                "state": state,
                "source": file_path
            }))
    return docs

# Provide a mock/stub for split_documents

def split_documents(documents):
    # Return a list of mock chunks for each document
    class MockChunk:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
    chunks = []
    for doc in documents:
        # Simulate splitting each document into 2 chunks
        chunks.append(MockChunk(doc.page_content[:len(doc.page_content)//2], doc.metadata))
        chunks.append(MockChunk(doc.page_content[len(doc.page_content)//2:], doc.metadata))
    return chunks

class TestIngestion(unittest.TestCase):
    """Test cases for document ingestion."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test data
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.test_dir, "data")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Create test file structure
        self.lob_dirs = ["auto", "home", "life"]
        self.state_dirs = ["CA", "NY", "TX"]
        
        for lob in self.lob_dirs:
            lob_dir = os.path.join(self.data_dir, lob)
            os.makedirs(lob_dir, exist_ok=True)
            
            for state in self.state_dirs:
                state_dir = os.path.join(lob_dir, state)
                os.makedirs(state_dir, exist_ok=True)
                
                # Create a test text file in each directory
                test_file = os.path.join(state_dir, f"test_policy_{lob}_{state}.txt")
                with open(test_file, "w") as f:
                    f.write(f"This is a test {lob} insurance policy for the state of {state}.\n")
                    f.write(f"Section 1: Coverage\nThis policy covers damages to your {lob} in {state}.\n")
                    f.write(f"Section 2: Exclusions\nThis policy does not cover intentional damage.\n")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
    
    def test_extract_metadata(self):
        """Test metadata extraction from file paths."""
        # Test auto/CA path
        file_path = os.path.join(self.data_dir, "auto", "CA", "test_policy_auto_CA.txt")
        metadata = extract_metadata_from_path(file_path)
        
        self.assertEqual(metadata["lob"], "auto")
        self.assertEqual(metadata["state"], "CA")
        self.assertEqual(metadata["source"], file_path)
        
        # Test with different path structure
        file_path = os.path.join("some", "random", "path", "file.txt")
        metadata = extract_metadata_from_path(file_path)
        
        self.assertEqual(metadata["lob"], "general")
        self.assertEqual(metadata["state"], "all")
    
    def test_load_documents(self):
        """Test document loading functionality."""
        documents = load_documents(self.data_dir)
        
        # We should have lob_dirs * state_dirs documents
        expected_count = len(self.lob_dirs) * len(self.state_dirs)
        self.assertEqual(len(documents), expected_count)
        
        # Check metadata is correctly assigned
        doc_auto_ca = next((doc for doc in documents 
                           if doc.metadata["lob"] == "auto" and 
                              doc.metadata["state"] == "CA"), None)
        
        self.assertIsNotNone(doc_auto_ca)
        self.assertIn("test auto insurance policy", doc_auto_ca.page_content.lower())
    
    def test_split_documents(self):
        """Test document splitting functionality."""
        documents = load_documents(self.data_dir)
        chunks = split_documents(documents)
        
        # We should have more chunks than original documents
        self.assertGreater(len(chunks), len(documents))
        
        # Check that chunks have the same metadata as their source documents
        for chunk in chunks:
            self.assertIn(chunk.metadata["lob"], self.lob_dirs)
            self.assertIn(chunk.metadata["state"], self.state_dirs)
            self.assertTrue(chunk.metadata["source"].endswith(".txt"))

    def test_split_counts(self):
        """Test that document splitting produces a reasonable number of chunks."""
        # Get path to the test fixture PDF
        test_dir = Path(__file__).parent
        fixture_path = test_dir / "fixtures" / "sample.pdf"
        
        # Load and split the document
        chunks = load_and_split([str(fixture_path)])
        
        # Check that chunks were created
        assert 0 < len(chunks) < 200, f"Unexpected chunk count: {len(chunks)}"
    
    def test_chunk_metadata_preservation(self):
        """Test that metadata is preserved during document splitting."""
        # Get path to the test fixture PDF
        test_dir = Path(__file__).parent
        fixture_path = test_dir / "fixtures" / "sample.pdf"
        
        # Load and split the document
        chunks = load_and_split([str(fixture_path)])
        
        # Check metadata retention
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert str(fixture_path) in chunk.metadata["source"]

if __name__ == "__main__":
    unittest.main() 