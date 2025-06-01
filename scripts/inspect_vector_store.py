import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import yaml

# Load config
CONFIG_PATH = "config/rag.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

persist_dir = os.path.abspath(config['vector_store']['persist_directory'])
collection_name = config['vector_store']['collection_name']
embedding_model = config['embeddings']['model']

print(f"[INFO] Inspecting vector store at: {persist_dir}")
print(f"[INFO] Collection name: {collection_name}")
print(f"[INFO] Embedding model: {embedding_model}")

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
vector_store = Chroma(persist_directory=persist_dir, embedding_function=embeddings, collection_name=collection_name)

# Get all documents/chunks
print("\n[INFO] Listing all chunks in the vector store:")
collection = vector_store._collection
result = collection.get(include=["documents", "metadatas"])

if not result or not result.get("documents"):
    print("[WARN] No documents found in the vector store.")
else:
    for i, (doc, meta) in enumerate(zip(result["documents"], result["metadatas"])):
        print(f"--- Chunk {i+1} ---")
        print(f"Source: {meta.get('source', 'N/A')}")
        print(f"Content: {doc[:200]}{'...' if len(doc) > 200 else ''}")
        print()

# Optional: search for a keyword
import sys
if len(sys.argv) > 1:
    keyword = sys.argv[1]
    print(f"\n[INFO] Searching for keyword: '{keyword}' in chunk contents...")
    found = False
    for i, (doc, meta) in enumerate(zip(result["documents"], result["metadatas"])):
        if keyword.lower() in doc.lower():
            print(f"[MATCH] Chunk {i+1} (Source: {meta.get('source', 'N/A')}):")
            print(f"{doc[:300]}{'...' if len(doc) > 300 else ''}")
            print()
            found = True
    if not found:
        print("[INFO] No chunks found containing the keyword.") 