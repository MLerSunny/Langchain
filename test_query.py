import requests
import json

def test_rag_query():
    url = "http://localhost:8000/query"
    payload = {
        "query": "What is Retrieval-Augmented Generation (RAG)?"
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes
        result = response.json()
        print("\nQuery Response:")
        print("Answer:", result.get("answer"))
        print("\nSources:")
        for source in result.get("sources", []):
            print("- Content:", source.get("content")[:100], "...")
            print("  Metadata:", source.get("metadata"))
        print("\nProcessing Time:", result.get("processing_time"), "seconds")
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding response: {e}")

if __name__ == "__main__":
    test_rag_query() 