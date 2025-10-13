# src/project/process_all_queries_optimized.py

import json
import zipfile
import logging
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pymilvus import MilvusClient, Collection, connections
from sentence_transformers import SentenceTransformer
import numpy as np 

# --- CONFIGURATION ---
QUERIES_FILE = 'queries.json'
OUTPUT_DIR = 'submission'
ZIP_FILENAME = 'PS04_YOUR_TEAM_NAME.zip'

# Milvus Configuration - LOCAL
MILVUS_URI = "http://localhost:19530"
MILVUS_TOKEN = None
COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Performance Configuration
EMBEDDING_BATCH_SIZE = 128
SEARCH_BATCH_SIZE = 32
MAX_WORKERS = 8
WRITE_WORKERS = 4

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedBatchQueryProcessor:
    """Optimized processor for large-scale query batching"""
    
    def __init__(self):
        logger.info("Initializing optimized query processor...")
        
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME, 
            device="cpu"
        )
        
        self._connect_to_milvus()
        
        logger.info("Initialization complete")
    
    def _connect_to_milvus(self, max_retries=5, retry_delay=3):
        logger.info(f"Connecting to Milvus at {MILVUS_URI}")
        
        for attempt in range(1, max_retries + 1):
            try:
                if MILVUS_TOKEN:
                    self.client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
                else:
                    self.client = MilvusClient(uri=MILVUS_URI)

                # Create the 'default' connection that the Collection class needs.
                connections.connect("default", uri=MILVUS_URI)
                
                if not self.client.has_collection(COLLECTION_NAME):
                    available = self.client.list_collections()
                    raise RuntimeError(
                        f"Collection '{COLLECTION_NAME}' not found! "
                        f"Available: {available}"
                    )
                
                schema_info = self.client.describe_collection(COLLECTION_NAME)
                self.vector_field_name = None
                
                logger.info(f"Collection schema fields:")
                for field in schema_info['fields']:
                    field_name = field['name']
                    field_type = field['type']
                    logger.info(f"  - {field_name}: {field_type}")
                    if field_type == 101 or field_type == 100 or 'VECTOR' in str(field_type):
                        self.vector_field_name = field_name
                        logger.info(f"    ^^^ Using this as vector field ^^^")
                
                if not self.vector_field_name:
                    raise RuntimeError("No vector field found in schema!")
                
                logger.info("Loading collection into memory...")
                collection = Collection(COLLECTION_NAME)
                collection.load()
                
                time.sleep(2)
                logger.info("Collection loaded successfully")
                
                stats = self.client.get_collection_stats(COLLECTION_NAME)
                row_count = stats.get('row_count', 0)
                
                if row_count == 0:
                    logger.warning(f"Collection '{COLLECTION_NAME}' is EMPTY!")
                else:
                    logger.info(f"Collection has {row_count:,} entities")
                
                logger.info(f"Successfully connected to Milvus (vector field: {self.vector_field_name})")
                return
                
            except Exception as e:
                logger.warning(f"Connection attempt {attempt}/{max_retries} failed: {e}")
                
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logger.error("\nFailed to connect to Milvus!")
                    raise RuntimeError(f"Cannot connect to Milvus at {MILVUS_URI}") from e
    
    def embed_queries_batch(self, queries: List[Tuple[str, str]]) -> List[Tuple[str, str, np.ndarray]]:
        query_nums = [q[0] for q in queries]
        query_texts = [q[1] for q in queries]
        
        logger.info(f"Embedding batch of {len(query_texts)} queries...")
        start_time = time.time()
        
        embeddings = self.embedding_model.encode(
            query_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        duration = time.time() - start_time
        speed = len(query_texts) / duration
        logger.info(f"Embedded {len(query_texts)} queries in {duration:.2f}s ({speed:.1f} queries/sec)")
        
        return list(zip(query_nums, query_texts, embeddings))
    
    def search_single_query(self, query_data: Tuple[str, str, np.ndarray]) -> Dict:
        query_num, query_text, embedding = query_data
        
        try:
            search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
            
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                data=[embedding.tolist()],
                limit=5,
                search_params=search_params,
                anns_field="embedding",
                # --- CHANGE #1: ASK FOR THE CHUNK TEXT ---
                output_fields=["doc_id","doc_name", "chunk_text"] 
            )
            
            doc_ids = []
            seen = set()
            # --- CHANGE #2: GRAB THE TEXT OF THE TOP CHUNK ---
            top_chunk_text = "Not found." 
            
            if results and len(results) > 0 and len(results[0]) > 0:
                # Get the top chunk text from the very first result
                top_chunk_text = results[0][0]['entity'].get('chunk_text', 'Text field missing.')

                for hit in results[0]:
                    doc_id = hit['entity'].get('doc_name', '')
                    if doc_id and doc_id not in seen:
                        doc_ids.append(doc_id)
                        seen.add(doc_id)
            
            return {
                "query_num": query_num,
                "query_text": query_text,
                "doc_ids": doc_ids,  
                "top_chunk_text": top_chunk_text, 
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Search failed for query {query_num}: {e}")
            return {
                "query_num": query_num,
                "query_text": query_text,
                "doc_ids": [],
                "top_chunk_text": f"Error during search: {e}",
                "success": False,
                "error": str(e)
            }
    
    def search_parallel_batch(self, embedded_queries: List[Tuple[str, str, np.ndarray]]) -> List[Dict]:
        logger.info(f"Searching {len(embedded_queries)} queries in parallel...")
        start_time = time.time()
        
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_query = {
                executor.submit(self.search_single_query, query_data): query_data[0]
                for query_data in embedded_queries
            }
            
            for future in as_completed(future_to_query):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # --- CHANGE #3: PRINT THE QUERY AND TOP RESULT ---
                    logger.info(f"\n[RESULT FOR QUERY #{result['query_num']}]")
                    logger.info(f"  QUERY: {result['query_text']}")
                    # Truncate the chunk text to keep the log clean
                    logger.info(f"  TOP CHUNK: {result['top_chunk_text'][:400]}...") 
                    
                except Exception as e:
                    query_num = future_to_query[future]
                    logger.error(f"Query {query_num} failed: {e}")
        
        duration = time.time() - start_time
        speed = len(embedded_queries) / duration if duration > 0 else 0
        logger.info(f"Searched {len(embedded_queries)} queries in {duration:.2f}s ({speed:.1f} queries/sec)")
        
        return results
    
    # ... (The rest of the file remains exactly the same) ...

    def write_result_file(self, result: Dict, output_path: Path) -> bool:
        try:
            output_data = {
                "query": result["query_text"],
                "response": result["doc_ids"]
            }
            
            output_file = output_path / f"query_{result['query_num']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            
            return True
        except Exception as e:
            logger.error(f"Failed to write result for query {result['query_num']}: {e}")
            return False
    
    def write_results_parallel(self, results: List[Dict], output_path: Path) -> int:
        logger.info(f"Writing {len(results)} result files...")
        start_time = time.time()
        
        success_count = 0
        with ThreadPoolExecutor(max_workers=WRITE_WORKERS) as executor:
            futures = [
                executor.submit(self.write_result_file, result, output_path)
                for result in results
            ]
            
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
        
        duration = time.time() - start_time
        logger.info(f"Wrote {success_count}/{len(results)} files in {duration:.2f}s")
        
        return success_count
    
    def process_all_queries(self):
        start_time = time.time()
        
        logger.info(f"Loading queries from {QUERIES_FILE}...")
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        logger.info(f"Loaded {len(queries_data)} queries")
        
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        queries = [
            (item.get("query_num"), item.get("query"))
            for item in queries_data
            if item.get("query_num") and item.get("query")
        ]
        
        logger.info(f"Processing {len(queries)} valid queries")
        
        all_results = []
        total_queries = len(queries)
        
        for batch_start in range(0, total_queries, EMBEDDING_BATCH_SIZE):
            batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total_queries)
            batch_queries = queries[batch_start:batch_end]
            
            batch_num = (batch_start // EMBEDDING_BATCH_SIZE) + 1
            total_batches = (total_queries + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing batch {batch_num}/{total_batches} (queries {batch_start+1}-{batch_end})")
            
            embedded_queries = self.embed_queries_batch(batch_queries)
            batch_results = self.search_parallel_batch(embedded_queries)
            self.write_results_parallel(batch_results, output_path)
            
            all_results.extend(batch_results)
            
            processed = len(all_results)
            elapsed = time.time() - start_time
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = total_queries - processed
            eta = remaining / speed if speed > 0 else 0
            
            logger.info(f"Progress: {processed}/{total_queries} ({100*processed/total_queries:.1f}%)")
            logger.info(f"Speed: {speed:.1f} queries/sec | ETA: {eta/60:.1f} minutes")
        
        logger.info(f"\nCreating submission zip: {ZIP_FILENAME}")
        json_files = list(output_path.glob("*.json"))
        
        with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in json_files:
                zf.write(file_path, arcname=file_path.name)
        
        total_time = time.time() - start_time
        success_count = sum(1 for r in all_results if r["success"])
        
        logger.info(f"\n{'='*60}")
        logger.info("BATCH PROCESSING COMPLETE")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Average speed: {total_queries/total_time:.1f} queries/sec")
        logger.info(f"Output: {ZIP_FILENAME}")
        logger.info(f"{'='*60}")

def run_batch_queries():
    try:
        processor = OptimizedBatchQueryProcessor()
        processor.process_all_queries()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_batch_queries()
