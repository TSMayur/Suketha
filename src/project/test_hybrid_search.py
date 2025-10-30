import logging
import time
import requests # Import the requests library
import json

# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    # Get client, collection name, Milvus URI, AND TEI_ENDPOINT from config
    from .config import client, COLLECTION_NAME, AZURE_MILVUS_URI, TEI_ENDPOINT
except ModuleNotFoundError:
    print("Error: Could not import from config.py.")
    print("Ensure config.py exists and poetry environment is active.")
    exit(1)
except ImportError as e:
    print(f"Error: Could not import required names from config.py: {e}")
    exit(1)

# Import necessary Milvus classes
from pymilvus import AnnSearchRequest, RRFRanker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
QUERY_TEXT = "Which continents does Topray Solar have a global presence in?"
SEARCH_LIMIT = 10 # How many results to retrieve ultimately
OUTPUT_FIELDS = ["chunk_id", "doc_name", "chunk_text"] # Fields to return

# Field names in your Milvus collection
SPARSE_VECTOR_FIELD = "sparse_vector"
DENSE_VECTOR_FIELD = "dense_vector"
TEXT_FIELD_FOR_BM25_INPUT = "chunk_text"

# --- Embedding Function ---
def get_dense_embedding_from_tei(text: str, endpoint_url: str) -> list[float]:
    """Generates a dense embedding for the given text by calling the TEI endpoint."""
    logger.info(f"Requesting dense query embedding from TEI endpoint: {endpoint_url}")
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({"inputs": text}) # Standard TEI input format

    try:
        response = requests.post(f"{endpoint_url}/embed", headers=headers, data=payload, timeout=10) # Added /embed path
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # TEI typically returns a list of embeddings, even for a single input
        embeddings = response.json()
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], list):
             logger.info("Dense embedding received successfully from TEI.")
             # Assuming the endpoint normalizes embeddings; if not, add normalization here.
             return embeddings[0]
        else:
             logger.error(f"Unexpected response format from TEI: {embeddings}")
             raise ValueError("TEI response format is incorrect.")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling TEI endpoint: {e}")
        raise
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        logger.error(f"Error processing TEI response: {e}")
        raise

# --- Main Test Function ---
def test_hybrid_search():
    logger.info(f"--- Running Hybrid Search (Native BM25 + TEI Dense) for Top {SEARCH_LIMIT} ---")
    logger.info(f"Query: \"{QUERY_TEXT}\"")
    logger.info(f"Collection: {COLLECTION_NAME}\n")

    try:
        # 1. Generate Dense Query Embedding via TEI
        if not TEI_ENDPOINT:
            logger.error("TEI_ENDPOINT is not defined in config.py or .env file.")
            return
        dense_query_embedding = get_dense_embedding_from_tei(QUERY_TEXT, TEI_ENDPOINT)

        # 2. Connect (client is imported from config)
        logger.info(f"Using Milvus client connected to {AZURE_MILVUS_URI}")
        if not client.has_collection(COLLECTION_NAME):
             logger.error(f"Collection '{COLLECTION_NAME}' not found!")
             return

        # 3. Prepare Search Requests
        # Sparse Request (using raw text)
        sparse_search_params = {"metric_type": "BM25"}
        sparse_request = AnnSearchRequest(
            data=[QUERY_TEXT],
            anns_field=SPARSE_VECTOR_FIELD,
            param=sparse_search_params,
            limit=SEARCH_LIMIT * 2
        )

        # Dense Request (using TEI embedding)
        dense_search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        dense_request = AnnSearchRequest(
            data=[dense_query_embedding], # Use the vector from TEI
            anns_field=DENSE_VECTOR_FIELD,
            param=dense_search_params,
            limit=SEARCH_LIMIT * 2
        )

        # 4. Perform Hybrid Search
        logger.info(f"Running hybrid search for Top {SEARCH_LIMIT} results...")
        start_time = time.time()
        results = client.hybrid_search(
            collection_name=COLLECTION_NAME,
            reqs=[sparse_request, dense_request],
            rerank=RRFRanker(),
            limit=SEARCH_LIMIT,
            output_fields=OUTPUT_FIELDS
        )
        duration = time.time() - start_time
        logger.info(f"Search completed in {duration:.4f} seconds.")

        # 5. Print results
        print(f"\n--- Top {SEARCH_LIMIT} Hybrid Search Results (RRF with TEI Query Embedding) ---")
        if not results or not results[0]:
            print("\nNo results found.")
            return

        for rank, hit in enumerate(results[0], 1):
            score = hit.score
            doc_name = hit.entity.get('doc_name', 'N/A')
            chunk_text_snippet = hit.entity.get('chunk_text', '')[:200]
            print(f"\n--- Rank {rank} (Score: {score:.4f}) ---")
            print(f"Doc: {doc_name}")
            print(f"Chunk: {chunk_text_snippet}...")

    except Exception as e:
        logger.error("--- An Error Occurred ---", exc_info=True)
        print(f"\nError: {e}")
        print("\nCheck connection details, collection name, field names, and TEI endpoint availability/response.")

    finally:
        logger.info("Test finished.")


if __name__ == "__main__":
    test_hybrid_search()
