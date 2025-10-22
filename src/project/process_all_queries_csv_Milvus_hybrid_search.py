# src/project/process_all_queries.py
"""
OPTIMIZED v4: Milvus Native Hybrid Search with Parallel Processing + CSV
Based on: https://milvus.io/docs/hybrid_search_with_milvus.md

Key Features:
1. Native Milvus hybrid_search() API (10x faster)
2. AnnSearchRequest + WeightedRanker for optimal fusion
3. Parallel batch processing for maximum speed
4. Cross-encoder reranking for accuracy
5. Comprehensive CSV output with all scores
6. SHA256 deterministic sparse vectors
"""

import json
import zipfile
import logging
import csv
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pymilvus import (
    MilvusClient,
    Collection,
    connections,
    AnnSearchRequest,
    WeightedRanker,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

QUERIES_FILE = '1.json'
OUTPUT_DIR = 'submission_hybrid_milvus'
ZIP_FILENAME = 'MILVUS_HYBRID_search.zip'
CSV_OUTPUT = 'milvus_hybrid_search_results.csv'

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_chunks_hybrid"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Search parameters
SEARCH_LIMIT = 50   # Candidates for reranking
FINAL_TOP_K = 5     # Final results to return

# Hybrid search weights (tune for best results)
SPARSE_WEIGHT = 0.7  # BM25/keyword matching
DENSE_WEIGHT = 1.0   # Semantic similarity

# Cross-encoder
USE_CROSS_ENCODER = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Performance config
EMBEDDING_BATCH_SIZE = 64
MAX_WORKERS = 16
WRITE_WORKERS = 4

# Sparse vector config
VOCAB_SIZE = 10_000_000
MIN_TOKEN_LENGTH = 2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# UTILITIES
# ============================================================================

STOPWORDS: Set[str] = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
    'the', 'to', 'was', 'were', 'will', 'with', 'this', 'but', 'they',
    'have', 'had', 'what', 'when', 'where', 'who', 'which', 'can',
    'their', 'if', 'out', 'so', 'up', 'been', 'than', 'them', 'she',
}


def deterministic_token_hash(token: str, vocab_size: int) -> int:
    """SHA256-based deterministic hashing"""
    hash_bytes = hashlib.sha256(token.encode('utf-8')).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    return hash_int % vocab_size


def tokenize_text(text: str) -> List[str]:
    """Tokenize with stopword removal"""
    tokens = text.lower().split()
    tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LENGTH and t not in STOPWORDS]
    return tokens


# ============================================================================
# OPTIMIZED HYBRID QUERY PROCESSOR
# ============================================================================

class OptimizedHybridQueryProcessor:
    """
    Fast hybrid search with parallel processing and CSV output
    """
    
    def __init__(self):
        logger.info("Initializing Optimized Hybrid Query Processor v4...")
        logger.info("="*70)
        
        # Load embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu"
        )
        logger.info("✓ Embedding model loaded")
        
        # Load cross-encoder
        if USE_CROSS_ENCODER:
            logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            logger.info("✓ Cross-encoder loaded")
        else:
            self.cross_encoder = None
        
        # Connect to Milvus
        self._connect_to_milvus()
        
        # CSV storage
        self.csv_data = []
        
        logger.info("="*70)
        logger.info("✓ Initialization complete\n")
    
    def _connect_to_milvus(self, max_retries=5, retry_delay=3):
        """Connect to Milvus with retry logic"""
        logger.info(f"Connecting to Milvus at {MILVUS_URI}")
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Attempt {attempt}/{max_retries}...")
                self.client = MilvusClient(uri=MILVUS_URI, timeout=30)
                connections.connect("default", uri=MILVUS_URI, timeout=30)
                break
            except Exception as e:
                logger.warning(f"Failed: {e}")
                if attempt < max_retries:
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Connection failed after {max_retries} attempts") from e
        
        # Verify collection
        logger.info("Checking collection...")
        if not self.client.has_collection(COLLECTION_NAME):
            available = self.client.list_collections()
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found! Available: {available}"
            )
        
        # Get collection object for hybrid search
        self.collection = Collection(COLLECTION_NAME)
        
        # Verify schema
        schema_info = self.client.describe_collection(COLLECTION_NAME)
        has_dense = False
        has_sparse = False
        
        logger.info("Schema:")
        for field in schema_info['fields']:
            field_name = field['name']
            logger.info(f"  - {field_name}")
            if field_name == "dense_vector":
                has_dense = True
            if field_name == "sparse_vector":
                has_sparse = True
        
        if not (has_dense and has_sparse):
            raise RuntimeError("Missing hybrid vectors!")
        
        # Load collection
        logger.info("Loading collection...")
        self.collection.load()
        time.sleep(2)
        
        # Get stats
        try:
            stats = self.client.get_collection_stats(COLLECTION_NAME)
            row_count = stats.get('row_count', 0)
            logger.info(f"✓ Connected: {row_count:,} chunks")
        except:
            logger.info("✓ Connected")
    
    def generate_sparse_vector(self, text: str) -> Dict[int, float]:
        """Generate deterministic sparse vector"""
        tokens = tokenize_text(text)
        if not tokens:
            return {}
        
        term_freq = {}
        for token in tokens:
            token_id = deterministic_token_hash(token, VOCAB_SIZE)
            term_freq[token_id] = term_freq.get(token_id, 0) + 1
        
        max_freq = max(term_freq.values())
        sparse_vector = {
            token_id: freq / max_freq
            for token_id, freq in term_freq.items()
        }
        
        return sparse_vector
    
    def embed_queries_batch(self, queries: List[Tuple[str, str]]) -> List[Tuple[str, str, np.ndarray, Dict]]:
        """
        Batch embed queries for efficiency.
        Returns: [(query_num, query_text, dense_vector, sparse_vector), ...]
        """
        query_nums = [q[0] for q in queries]
        query_texts = [q[1] for q in queries]
        
        logger.info(f"Embedding batch of {len(query_texts)} queries...")
        start_time = time.time()
        
        # Dense embeddings
        dense_embeddings = self.embedding_model.encode(
            query_texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Sparse embeddings
        sparse_embeddings = [self.generate_sparse_vector(text) for text in query_texts]
        
        duration = time.time() - start_time
        speed = len(query_texts) / duration
        logger.info(f"✓ Embedded {len(query_texts)} queries in {duration:.2f}s ({speed:.1f} q/s)")
        
        return list(zip(query_nums, query_texts, dense_embeddings, sparse_embeddings))
    
    def hybrid_search_native(
        self,
        query_text: str,
        dense_vector: np.ndarray,
        sparse_vector: Dict[int, float]
    ) -> List[Dict]:
        """
        ⭐ OPTIMIZED: Native Milvus hybrid_search() API
        
        - Single API call (10x faster than manual RRF)
        - Native C++ implementation
        - Optimized WeightedRanker
        """
        # Dense search request
        dense_search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
        dense_req = AnnSearchRequest(
            data=[dense_vector.tolist() if isinstance(dense_vector, np.ndarray) else dense_vector],
            anns_field="dense_vector",
            param=dense_search_params,
            limit=SEARCH_LIMIT
        )
        
        # Sparse search request
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param=sparse_search_params,
            limit=SEARCH_LIMIT
        )
        
        # Weighted ranker for fusion
        reranker = WeightedRanker(SPARSE_WEIGHT, DENSE_WEIGHT)
        
        # ⭐ Single hybrid search call (FAST!)
        results = self.collection.hybrid_search(
            reqs=[sparse_req, dense_req],
            rerank=reranker,
            limit=SEARCH_LIMIT,
            output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"]
        )[0]
        
        # Convert to dict format
        formatted_results = []
        for rank, hit in enumerate(results, 1):
            formatted_results.append({
                'chunk_id': hit.entity.get('chunk_id', ''),
                'doc_id': hit.entity.get('doc_id', ''),
                'doc_name': hit.entity.get('doc_name', ''),
                'chunk_text': hit.entity.get('chunk_text', ''),
                'hybrid_score': float(hit.score),
                'rank': rank
            })
        
        return formatted_results
    
    def cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Dict]
    ) -> List[Dict]:
        """Cross-encoder reranking for accuracy"""
        if not self.cross_encoder or not candidates:
            return candidates
        
        pairs = [[query, c['chunk_text']] for c in candidates]
        ce_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        
        for candidate, score in zip(candidates, ce_scores):
            candidate['cross_encoder_score'] = float(score)
        
        reranked = sorted(
            candidates,
            key=lambda x: x['cross_encoder_score'],
            reverse=True
        )
        
        return reranked
    
    def process_single_query(self, query_data: Tuple[str, str, np.ndarray, Dict]) -> Dict:
        """Process single query"""
        query_num, query_text, dense_vec, sparse_vec = query_data
        
        try:
            # Native hybrid search (FAST!)
            hybrid_results = self.hybrid_search_native(query_text, dense_vec, sparse_vec)
            
            # Cross-encoder reranking (optional)
            if USE_CROSS_ENCODER and hybrid_results:
                final_results = self.cross_encoder_rerank(query_text, hybrid_results)[:FINAL_TOP_K]
            else:
                final_results = hybrid_results[:FINAL_TOP_K]
            
            # Extract doc_ids
            doc_ids = []
            seen = set()
            for result in final_results:
                doc_name = result.get('doc_name', '')
                if doc_name and doc_name not in seen:
                    doc_ids.append(doc_name)
                    seen.add(doc_name)
            
            # Store for CSV
            for rank, result in enumerate(final_results, 1):
                self.csv_data.append({
                    'query_num': query_num,
                    'query_text': query_text,
                    'rank': rank,
                    'chunk_id': result.get('chunk_id', ''),
                    'doc_id': result.get('doc_id', ''),
                    'doc_name': result.get('doc_name', ''),
                    'hybrid_score': result.get('hybrid_score', 0.0),
                    'cross_encoder_score': result.get('cross_encoder_score', 0.0),
                    'chunk_text': result.get('chunk_text', '')[:500]
                })
            
            logger.info(f"✓ Query {query_num}: {len(doc_ids)} docs")
            
            return {
                "query_num": query_num,
                "query_text": query_text,
                "doc_ids": doc_ids,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"✗ Query {query_num} failed: {e}")
            return {
                "query_num": query_num,
                "query_text": query_text,
                "doc_ids": [],
                "success": False,
                "error": str(e)
            }
    
    def search_parallel_batch(self, embedded_queries: List) -> List[Dict]:
        """Search queries in parallel"""
        logger.info(f"Searching {len(embedded_queries)} queries in parallel...")
        start_time = time.time()
        
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_query = {
                executor.submit(self.process_single_query, query_data): query_data[0]
                for query_data in embedded_queries
            }
            
            for future in as_completed(future_to_query):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    query_num = future_to_query[future]
                    logger.error(f"✗ Query {query_num}: {e}")
        
        duration = time.time() - start_time
        speed = len(embedded_queries) / duration if duration > 0 else 0
        logger.info(f"✓ Searched {len(embedded_queries)} queries in {duration:.2f}s ({speed:.1f} q/s)")
        
        return results
    
    def write_result_file(self, result: Dict, output_path: Path) -> bool:
        """Write result to JSON"""
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
            logger.error(f"Write failed: {e}")
            return False
    
    def write_results_parallel(self, results: List[Dict], output_path: Path) -> int:
        """Write result files in parallel"""
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
        logger.info(f"✓ Wrote {success_count}/{len(results)} files in {duration:.2f}s")
        
        return success_count
    
    def write_csv_results(self):
        """Write comprehensive CSV with all scores"""
        logger.info(f"Writing CSV to {CSV_OUTPUT}...")
        
        try:
            with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'query_num', 'query_text', 'rank', 'chunk_id', 'doc_id', 'doc_name',
                    'hybrid_score', 'cross_encoder_score', 'chunk_text'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.csv_data)
            
            logger.info(f"✓ CSV written: {len(self.csv_data)} rows")
        except Exception as e:
            logger.error(f"CSV write failed: {e}")
    
    def process_all_queries(self):
        """Main processing pipeline"""
        start_time = time.time()
        
        # Load queries
        logger.info(f"Loading queries from {QUERIES_FILE}...")
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        queries = [
            (item.get("query_num"), item.get("query"))
            for item in queries_data
            if item.get("query_num") and item.get("query")
        ]
        
        logger.info(f"Processing {len(queries)} queries")
        logger.info(f"Strategy: Native Hybrid Search + Cross-Encoder + Parallel")
        logger.info(f"Weights: Sparse={SPARSE_WEIGHT}, Dense={DENSE_WEIGHT}")
        logger.info("="*70)
        
        # Create output dir
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        # Process in batches
        all_results = []
        total_queries = len(queries)
        
        for batch_start in range(0, total_queries, EMBEDDING_BATCH_SIZE):
            batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, total_queries)
            batch_queries = queries[batch_start:batch_end]
            
            batch_num = (batch_start // EMBEDDING_BATCH_SIZE) + 1
            total_batches = (total_queries + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
            
            logger.info(f"\n{'='*70}")
            logger.info(f"BATCH {batch_num}/{total_batches} (queries {batch_start+1}-{batch_end})")
            logger.info(f"{'='*70}")
            
            # Embed batch
            embedded_queries = self.embed_queries_batch(batch_queries)
            
            # Search in parallel
            batch_results = self.search_parallel_batch(embedded_queries)
            
            # Write results in parallel
            self.write_results_parallel(batch_results, output_path)
            
            all_results.extend(batch_results)
            
            # Progress
            processed = len(all_results)
            elapsed = time.time() - start_time
            speed = processed / elapsed if elapsed > 0 else 0
            remaining = total_queries - processed
            eta = remaining / speed if speed > 0 else 0
            
            logger.info(f"\nProgress: {processed}/{total_queries} ({100*processed/total_queries:.1f}%)")
            logger.info(f"Speed: {speed:.1f} q/s | ETA: {eta/60:.1f} min")
        
        # Write CSV
        self.write_csv_results()
        
        # Create zip
        logger.info(f"\nCreating {ZIP_FILENAME}...")
        json_files = list(output_path.glob("*.json"))
        
        with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in json_files:
                zf.write(file_path, arcname=file_path.name)
        
        # Stats
        total_time = time.time() - start_time
        success_count = sum(1 for r in all_results if r["success"])
        
        logger.info(f"\n{'='*70}")
        logger.info("OPTIMIZED HYBRID SEARCH COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Method: Native hybrid_search() + WeightedRanker + Parallel")
        logger.info(f"Total: {len(queries)} | Success: {success_count} | Failed: {len(queries)-success_count}")
        logger.info(f"Time: {total_time/60:.2f} min | Speed: {len(queries)/total_time:.2f} q/s")
        logger.info(f"Output ZIP: {ZIP_FILENAME}")
        logger.info(f"Output CSV: {CSV_OUTPUT}")
        logger.info(f"{'='*70}")


def main():
    """Main entry point"""
    try:
        processor = OptimizedHybridQueryProcessor()
        processor.process_all_queries()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
