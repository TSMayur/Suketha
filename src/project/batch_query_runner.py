# src/project/batch_query_runner.py

import json
import logging
from pathlib import Path
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
AZURE_MILVUS_URI = "http://135.235.255.241:19530"
AZURE_MILVUS_TOKEN = "SecurePassword123"

COLLECTION_NAME = "testCollection2"
SCRIPT_DIR = Path(__file__).parent
INPUT_QUERIES_FILE = SCRIPT_DIR / "1.json"
OUTPUT_SUBMISSION_DIR = SCRIPT_DIR / "submission_files"

# CRITICAL: MUST match the model used during indexing
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
SEARCH_LIMIT_PER_QUERY = 5

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_submission_files():
    logger.info("--- Starting Batch Query Runner for Submission ---")

    # --- 1. Load Queries ---
    if not INPUT_QUERIES_FILE.exists():
        logger.error(f"FATAL: Input query file not found at '{INPUT_QUERIES_FILE}'.")
        return

    with open(INPUT_QUERIES_FILE, 'r', encoding='utf-8') as f:
        queries_data = json.load(f)
    logger.info(f"Loaded {len(queries_data)} queries from '{INPUT_QUERIES_FILE}'.")

    # --- 2. Initialize Models and Connect to Milvus ---
    try:
        logger.info(f"Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        logger.info("Embedding model loaded successfully.")

        logger.info(f"Connecting to Milvus at {AZURE_MILVUS_URI}...")
        client = MilvusClient(uri=AZURE_MILVUS_URI, token=AZURE_MILVUS_TOKEN)
        
        # Verify collection exists
        if not client.has_collection(COLLECTION_NAME):
            logger.error(f"FATAL: Collection '{COLLECTION_NAME}' does not exist.")
            available = client.list_collections()
            logger.error(f"Available collections: {available}")
            return
        
        # Check collection stats
        stats = client.get_collection_stats(COLLECTION_NAME)
        row_count = stats.get('row_count', 0)
        logger.info(f"Collection '{COLLECTION_NAME}' has {row_count} entities.")
        
        if row_count == 0:
            logger.error("FATAL: Collection is empty. Ensure data has been ingested.")
            return

        logger.info("Connection successful. Ready to process queries.")

    except Exception as e:
        logger.error(f"FATAL: Failed to initialize: {e}", exc_info=True)
        return

    # --- 3. Prepare Output Directory ---
    OUTPUT_SUBMISSION_DIR.mkdir(exist_ok=True)
    logger.info(f"Output directory ready: {OUTPUT_SUBMISSION_DIR}")

    # --- 4. Process Each Query ---
    successful = 0
    failed = 0
    
    for i, item in enumerate(queries_data):
        query_num = item.get("query_num")
        query_text = item.get("query")

        if not query_text or not query_num:
            logger.warning(f"Skipping item {i+1}: missing query_num or query")
            failed += 1
            continue

        logger.info(f"Query {i+1}/{len(queries_data)} (ID: {query_num})")
        
        try:
            # Generate query embedding - SAME model as indexing
            query_vector = embedding_model.encode(
                query_text, 
                normalize_embeddings=True
            )
            
            # Search with proper parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}  # HNSW parameter
            }

            search_results = client.search(
                collection_name=COLLECTION_NAME,
                data=[query_vector.tolist()],
                limit=SEARCH_LIMIT_PER_QUERY,
                search_params=search_params,
                anns_field="dense_vector",  # Correct field from schema
                output_fields=["file_name"]  # Correct field from schema
            )
            
            # Extract unique results
            retrieved_docs = []
            seen = set()
            
            if search_results and len(search_results) > 0:
                for hit in search_results[0]:
                    # Get the doc identifier
                    doc_id = hit['entity'].get('doc_id', '')
                    
                    # If doc_id is like "doc123" but you need "doc123.txt", adjust here
                    # Based on instructions, seems like no .txt extension needed
                    if doc_id and doc_id not in seen:
                        retrieved_docs.append(doc_id)
                        seen.add(doc_id)
                        logger.info(f"  - {doc_id} (score: {hit.get('distance', 'N/A')})")
            
            if not retrieved_docs:
                logger.warning(f"No results for query {query_num}")
            
            # Create output in required format
            output_data = {
                "query": query_text,
                "response": retrieved_docs
            }

            # Save individual file
            output_file = OUTPUT_SUBMISSION_DIR / f"{query_num}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            
            successful += 1
            logger.info(f"  -> Saved {len(retrieved_docs)} results")

        except Exception as e:
            logger.error(f"FAILED query {query_num}: {e}", exc_info=True)
            failed += 1

    # --- 5. Summary ---
    logger.info("=" * 60)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info(f"Successful: {successful}/{len(queries_data)}")
    logger.info(f"Failed: {failed}/{len(queries_data)}")
    logger.info(f"Output: {OUTPUT_SUBMISSION_DIR}")
    logger.info(f"Next: Zip '{OUTPUT_SUBMISSION_DIR.name}' as PS04_TEAM_NAME.zip")
    logger.info("=" * 60)


if __name__ == "__main__":
    generate_submission_files()