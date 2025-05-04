#!/usr/bin/env python3
"""
Script to list all documents in ChromaDB with their metadata
"""

import os
import sys
import chromadb
from chromadb.config import Settings

# Add the project root to the path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.settings import settings

def list_chroma_documents():
    """List all documents in ChromaDB with their metadata."""
    # Connect to ChromaDB
    persist_directory = settings.chroma_persist_directory
    print(f"Connecting to ChromaDB at: {persist_directory}")
    
    try:
        client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get all collections in ChromaDB v0.6.0+
        collection_names = client.list_collections()
        print(f"Found {len(collection_names)} collections")
        
        if not collection_names:
            print("No collections found in ChromaDB.")
            return
        
        # Go through each collection
        for collection_name in collection_names:
            print(f"\nCollection: {collection_name}")
            
            # Get the collection
            collection = client.get_collection(name=collection_name)
            
            # Get all items in the collection
            items = collection.get(include=["metadatas", "documents"])
            doc_count = len(items["ids"]) if "ids" in items else 0
            print(f"  Total documents: {doc_count}")
            
            # Print the raw count from the collection API as well
            print(f"  Collection count(): {collection.count()}")
            
            # Display each document with its metadata
            if doc_count > 0:
                print("\n  Documents:")
                for i, doc_id in enumerate(items["ids"]):
                    metadata = items["metadatas"][i] if "metadatas" in items else {}
                    source = metadata.get("source", "Unknown")
                    lob = metadata.get("lob", "Unknown")
                    state = metadata.get("state", "Unknown")
                    
                    # Get a preview of the document content
                    content = items["documents"][i] if "documents" in items else ""
                    preview = content[:100] + "..." if len(content) > 100 else content
                    
                    print(f"  {i+1}. ID: {doc_id}")
                    print(f"     Source: {source}")
                    print(f"     LOB: {lob}, State: {state}")
                    print(f"     Content preview: {preview}")
                    print()
    
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return

if __name__ == "__main__":
    list_chroma_documents() 