"""
Example Python client for newbieAR API
"""
import requests
import json

# Configuration
BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "newbie_admin_dev_key_change_in_production"
HEADERS = {"X-API-Key": API_KEY}


def health_check():
    """Check API health"""
    response = requests.get("http://localhost:8000/health")
    print("Health:", response.json())


def upload_document(file_path: str, collection_name: str):
    """Upload document for ingestion"""
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {
            "collection_name": collection_name,
            "chunk_strategy": "hybrid",
        }

        response = requests.post(
            f"{BASE_URL}/ingest/vectordb",
            headers=HEADERS,
            files=files,
            data=data,
        )

        print("Upload response:", response.json())
        return response.json()["job_id"]


def check_job(job_id: str):
    """Check ingestion job status"""
    response = requests.get(
        f"{BASE_URL}/ingest/jobs/{job_id}",
        headers=HEADERS,
    )
    print("Job status:", response.json())


def basic_rag_search(query: str, collection_name: str):
    """Perform basic RAG search"""
    payload = {
        "user_id": "example_user",
        "query": query,
        "collection_name": collection_name,
        "top_k": 5,
    }

    response = requests.post(
        f"{BASE_URL}/retrieval/basic-rag",
        headers=HEADERS,
        json=payload,
    )

    print("Search results:", json.dumps(response.json(), indent=2))


def chat_with_agent(message: str, collection_name: str, session_id: str = None):
    """Chat with agent (synchronous)"""
    payload = {
        "user_id": "example_user",
        "message": message,
        "collection_name": collection_name,
        "session_id": session_id,
        "top_k": 5,
    }

    response = requests.post(
        f"{BASE_URL}/agents/basic-rag/chat",
        headers=HEADERS,
        json=payload,
    )

    result = response.json()
    print("\nAgent response:", result["message"])
    print("Session ID:", result["session_id"])
    print("Citations:", len(result["citations"]))

    return result["session_id"]


def stream_chat(message: str, collection_name: str):
    """Stream chat with agent (SSE)"""
    payload = {
        "user_id": "example_user",
        "message": message,
        "collection_name": collection_name,
        "top_k": 5,
    }

    response = requests.post(
        f"{BASE_URL}/agents/basic-rag/stream",
        headers=HEADERS,
        json=payload,
        stream=True,
    )

    print("\nStreaming response:")
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith('data: '):
                print(decoded[6:])


if __name__ == "__main__":
    print("=== newbieAR API Client Examples ===\n")

    # Health check
    health_check()

    # Example: Upload and search
    # job_id = upload_document("path/to/document.pdf", "my_collection")
    # check_job(job_id)

    # Example: Search
    # basic_rag_search("What is RAG?", "my_collection")

    # Example: Chat
    # session_id = chat_with_agent("Explain transformers", "my_collection")

    # Example: Streaming chat
    # stream_chat("What are the benefits of RAG?", "my_collection")
