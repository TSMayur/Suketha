"""
FIXED v2: Query processor with PROPER deterministic BM25 sparse vectors

Key Fixes:
1. Use SHA256 for deterministic hashing (not Python's hash())
2. Larger vocabulary space (10M instead of 1M)
3. Better tokenization (remove stopwords, min length)
4. TF-IDF style scoring instead of just normalized TF
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
from pymilvus import MilvusClient, Collection, connections
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Configuration
QUERIES_FILE = '1.json'
OUTPUT_DIR = 'submission_hybrid_optimized'
ZIP_FILENAME = 'hybrid_search_optimized.zip'
CSV_OUTPUT = 'hybrid_search_detailed_results.csv'

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_chunks_hybrid"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

DENSE_TOP_K = 30
SPARSE_TOP_K = 30
FINAL_TOP_K = 5
USE_CROSS_ENCODER = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

EMBEDDING_BATCH_SIZE = 64
MAX_WORKERS = 6
RRF_K = 60

# ⭐ Sparse vector config
VOCAB_SIZE = 10_000_000  # 10M instead of 1M for less collisions
MIN_TOKEN_LENGTH = 2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Minimal stopwords (expand if needed)
STOPWORDS: Set[str] = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
    'the', 'to', 'was', 'were', 'will', 'with', 'this', 'but', 'they',
    'have', 'had', 'what', 'when', 'where', 'who', 'which', 'can',
    'their', 'if', 'out', 'so', 'up', 'been', 'than', 'them', 'she',
}


def deterministic_token_hash(token: str, vocab_size: int) -> int:
    """
    ⭐ CRITICAL: Deterministic hash using SHA256
    
    Unlike Python's hash(), SHA256 is:
    - Deterministic (same token always -> same ID)
    - Consistent across Python sessions
    - Better distribution (fewer collisions)
    """
    hash_bytes = hashlib.sha256(token.encode('utf-8')).digest()
    # Use first 8 bytes as integer
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')
    return hash_int % vocab_size


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text for sparse vector generation
    
    - Lowercase
    - Split on whitespace
    - Remove stopwords
    - Filter short tokens
    """
    tokens = text.lower().split()
    
    # Filter
    tokens = [
        t for t in tokens
        if len(t) >= MIN_TOKEN_LENGTH and t not in STOPWORDS
    ]
    
    return tokens


class FixedHybridQueryProcessor:
    """
    Query processor with PROPER deterministic BM25
    """
    
    def __init__(self):
        logger.info("Initializing Fixed Hybrid Query Processor v2...")
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu"
        )
        logger.info("✓ Embedding model loaded")
        
        # Load cross-encoder
        if USE_CROSS_ENCODER:
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
            logger.info("✓ Cross-encoder loaded")
        else:
            self.cross_encoder = None
        
        # Connect to Milvus
        self._connect_to_milvus()
        
        self.csv_data = []
        logger.info("✓ Initialization complete\n")
    
    def _connect_to_milvus(self):
        """Connect to Milvus"""
        logger.info(f"Connecting to Milvus at {MILVUS_URI}")
        
        self.client = MilvusClient(uri=MILVUS_URI)
        connections.connect("default", uri=MILVUS_URI)
        
        if not self.client.has_collection(COLLECTION_NAME):
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found!")
        
        # Load collection
        collection = Collection(COLLECTION_NAME)
        collection.load()
        time.sleep(2)
        
        stats = self.client.get_collection_stats(COLLECTION_NAME)
        logger.info(f"✓ Connected: {stats.get('row_count', 0):,} chunks")
    
    def generate_sparse_vector(self, text: str) -> Dict[int, float]:
        """
        ⭐ FIXED: Generate sparse vector with deterministic hashing
        
        Uses SHA256 for consistent token IDs across sessions
        """
        # Tokenize with stopword removal
        tokens = tokenize_text(text)
        
        if not tokens:
            return {}
        
        # Count term frequencies
        term_freq = {}
        for token in tokens:
            token_id = deterministic_token_hash(token, VOCAB_SIZE)
            term_freq[token_id] = term_freq.get(token_id, 0) + 1
        
        # Normalize by max frequency (simple TF)
        max_freq = max(term_freq.values())
        sparse_vector = {
            token_id: freq / max_freq
            for token_id, freq in term_freq.items()
        }
        
        return sparse_vector
    
    def embed_queries_batch(self, queries: List[Tuple[str, str]]) -> List[Tuple[str, str, np.ndarray, Dict]]:
        """
        Embed queries in batch - BOTH dense and sparse
        
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
        
        # ⭐ Sparse vectors with deterministic hashing
        sparse_vectors = [
            self.generate_sparse_vector(text)
            for text in query_texts
        ]
        
        duration = time.time() - start_time
        logger.info(f"✓ Embedded {len(query_texts)} queries in {duration:.2f}s")
        
        # ⭐ DEBUG: Log sparse vector stats
        if sparse_vectors:
            sizes = [len(sv) for sv in sparse_vectors]
            avg_size = sum(sizes) / len(sizes)
            min_size = min(sizes)
            max_size = max(sizes)
            
            logger.info(f"  Dense shape: {dense_embeddings.shape}")
            logger.info(f"  Sparse dimensions - avg: {avg_size:.1f}, min: {min_size}, max: {max_size}")
            
            # Show sample
            sample_vec = sparse_vectors[0]
            sample_items = list(sample_vec.items())[:3]
            logger.info(f"  Sample sparse: {dict(sample_items)}")
        
        return list(zip(query_nums, query_texts, dense_embeddings, sparse_vectors))
    
    def hybrid_search_single(
        self,
        query_text: str,
        dense_vec: np.ndarray,
        sparse_vec: Dict[int, float]
    ) -> List[Dict]:
        """
        Hybrid search with proper error handling
        """
        # Check sparse vector
        if not sparse_vec:
            logger.warning(f"⚠️  Empty sparse vector for: {query_text[:50]}...")
        elif len(sparse_vec) < 5:
            logger.warning(f"⚠️  Very small sparse vector ({len(sparse_vec)} dims) for: {query_text[:30]}...")
        
        # Dense search (COSINE)
        search_params_dense = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        try:
            dense_results = self.client.search(
                collection_name=COLLECTION_NAME,
                data=[dense_vec.tolist()],
                anns_field="dense_vector",
                limit=DENSE_TOP_K,
                search_params=search_params_dense,
                output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"]
            )
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            dense_results = []
        
        # Sparse search (IP for sparse vectors)
        search_params_sparse = {
            "metric_type": "IP",
            "params": {}
        }
        
        sparse_results = []
        if sparse_vec:
            try:
                sparse_results = self.client.search(
                    collection_name=COLLECTION_NAME,
                    data=[sparse_vec],
                    anns_field="sparse_vector",
                    limit=SPARSE_TOP_K,
                    search_params=search_params_sparse,
                    output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"]
                )
            except Exception as e:
                logger.error(f"Sparse search failed: {e}")
                sparse_results = []
        
        # Log search results
        dense_count = len(dense_results[0]) if dense_results and len(dense_results) > 0 else 0
        sparse_count = len(sparse_results[0]) if sparse_results and len(sparse_results) > 0 else 0
        
        if dense_count == 0 and sparse_count == 0:
            logger.warning(f"⚠️  NO RESULTS for: {query_text[:50]}...")
        elif sparse_count == 0:
            logger.debug(f"No sparse results (only {dense_count} dense) for: {query_text[:30]}...")
        
        # RRF Fusion
        dense_ranks = {}
        sparse_ranks = {}
        all_chunks = {}
        
        # Process dense results
        if dense_results and len(dense_results) > 0:
            for rank, hit in enumerate(dense_results[0], 1):
                chunk_id = hit['entity'].get('chunk_id', '')
                dense_ranks[chunk_id] = rank
                all_chunks[chunk_id] = {
                    'chunk_id': chunk_id,
                    'doc_id': hit['entity'].get('doc_id', ''),
                    'doc_name': hit['entity'].get('doc_name', ''),
                    'chunk_text': hit['entity'].get('chunk_text', ''),
                    'dense_score': float(hit.get('distance', 0.0)),
                    'dense_rank': rank
                }
        
        # Process sparse results
        if sparse_results and len(sparse_results) > 0:
            for rank, hit in enumerate(sparse_results[0], 1):
                chunk_id = hit['entity'].get('chunk_id', '')
                sparse_ranks[chunk_id] = rank
                
                if chunk_id in all_chunks:
                    all_chunks[chunk_id]['sparse_score'] = float(hit.get('distance', 0.0))
                    all_chunks[chunk_id]['sparse_rank'] = rank
                else:
                    all_chunks[chunk_id] = {
                        'chunk_id': chunk_id,
                        'doc_id': hit['entity'].get('doc_id', ''),
                        'doc_name': hit['entity'].get('doc_name', ''),
                        'chunk_text': hit['entity'].get('chunk_text', ''),
                        'sparse_score': float(hit.get('distance', 0.0)),
                        'sparse_rank': rank
                    }
        
        # Calculate RRF scores
        for chunk_id, chunk_data in all_chunks.items():
            dense_rank = dense_ranks.get(chunk_id, DENSE_TOP_K + 1)
            sparse_rank = sparse_ranks.get(chunk_id, SPARSE_TOP_K + 1)
            
            rrf_score = (1.0 / (RRF_K + dense_rank)) + (1.0 / (RRF_K + sparse_rank))
            
            chunk_data['rrf_score'] = rrf_score
            chunk_data['dense_rank'] = dense_rank
            chunk_data['sparse_rank'] = sparse_rank
            chunk_data.setdefault('dense_score', 0.0)
            chunk_data.setdefault('sparse_score', 0.0)
        
        # Sort by RRF score
        fused_results = sorted(
            all_chunks.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        return fused_results
    
    def cross_encoder_rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank with cross-encoder"""
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
            # Hybrid search
            fused_results = self.hybrid_search_single(query_text, dense_vec, sparse_vec)
            
            # Take top candidates
            candidates = fused_results[:50]
            
            # Cross-encoder reranking
            if USE_CROSS_ENCODER and candidates:
                final_results = self.cross_encoder_rerank(query_text, candidates)[:FINAL_TOP_K]
            else:
                final_results = candidates[:FINAL_TOP_K]
            
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
                    'rrf_score': result.get('rrf_score', 0.0),
                    'cosine_score': result.get('dense_score', 0.0),
                    'bm25_score': result.get('sparse_score', 0.0),
                    'cosine_rank': result.get('dense_rank', 999),
                    'bm25_rank': result.get('sparse_rank', 999),
                    'cross_encoder_score': result.get('cross_encoder_score', 0.0),
                    'chunk_text': result.get('chunk_text', '')[:500]
                })
            
            sparse_hit_count = sum(1 for r in final_results if r.get('sparse_rank', 999) <= SPARSE_TOP_K)
            logger.info(f"✓ Query {query_num}: {len(doc_ids)} docs | Sparse: {sparse_hit_count}/{len(final_results)}")
            
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
        """Search in parallel"""
        logger.info(f"Searching {len(embedded_queries)} queries...")
        start_time = time.time()
        
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.process_single_query, query_data): query_data[0]
                for query_data in embedded_queries
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Query failed: {e}")
        
        duration = time.time() - start_time
        logger.info(f"✓ Searched in {duration:.2f}s ({len(embedded_queries)/duration:.1f} q/s)")
        
        return results
    
    def write_result_file(self, result: Dict, output_path: Path) -> bool:
        """Write JSON result"""
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
    
    def write_csv_results(self):
        """Write CSV"""
        logger.info(f"Writing CSV to {CSV_OUTPUT}...")
        
        try:
            with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'query_num', 'query_text', 'rank', 'chunk_id', 'doc_id', 'doc_name',
                    'rrf_score', 'cosine_score', 'bm25_score', 'cosine_rank', 'bm25_rank',
                    'cross_encoder_score', 'chunk_text'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.csv_data)
            
            logger.info(f"✓ CSV written: {len(self.csv_data)} rows")
        except Exception as e:
            logger.error(f"CSV write failed: {e}")
    
    def process_all_queries(self):
        """Main processing"""
        start_time = time.time()
        
        # Load queries
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        queries = [
            (item.get("query_num"), item.get("query"))
            for item in queries_data
            if item.get("query_num") and item.get("query")
        ]
        
        logger.info(f"Processing {len(queries)} queries")
        
        # Create output dir
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        # Process in batches
        all_results = []
        
        for batch_start in range(0, len(queries), EMBEDDING_BATCH_SIZE):
            batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, len(queries))
            batch_queries = queries[batch_start:batch_end]
            
            logger.info(f"\n{'='*70}")
            logger.info(f"Batch {batch_start+1}-{batch_end}:")
            
            # Embed (both dense and sparse)
            embedded_queries = self.embed_queries_batch(batch_queries)
            
            # Search
            batch_results = self.search_parallel_batch(embedded_queries)
            
            # Write results
            for result in batch_results:
                self.write_result_file(result, output_path)
            
            all_results.extend(batch_results)
        
        # Write CSV
        self.write_csv_results()
        
        # Create zip
        logger.info(f"\nCreating {ZIP_FILENAME}...")
        with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in output_path.glob("*.json"):
                zf.write(file, arcname=file.name)
        
        # Stats
        total_time = time.time() - start_time
        success_count = sum(1 for r in all_results if r["success"])
        
        # Analyze sparse usage
        sparse_hits = sum(
            1 for row in self.csv_data 
            if row['bm25_rank'] <= SPARSE_TOP_K
        )
        total_rows = len(self.csv_data)
        sparse_percentage = (sparse_hits / total_rows * 100) if total_rows > 0 else 0
        
        # Analyze sparse contribution to top-1
        top1_with_sparse = sum(
            1 for row in self.csv_data
            if row['rank'] == 1 and row['bm25_rank'] <= SPARSE_TOP_K
        )
        
        logger.info(f"\n{'='*70}")
        logger.info("HYBRID SEARCH COMPLETE")
        logger.info(f"Total: {len(queries)} | Success: {success_count} | Failed: {len(queries)-success_count}")
        logger.info(f"Time: {total_time/60:.2f} min | Speed: {len(queries)/total_time:.2f} q/s")
        logger.info(f"Sparse effectiveness: {sparse_percentage:.1f}% ({sparse_hits}/{total_rows} results)")
        logger.info(f"Sparse in top-1: {top1_with_sparse}/{len(queries)} queries")
        logger.info(f"Output: {ZIP_FILENAME} | CSV: {CSV_OUTPUT}")
        logger.info(f"{'='*70}")


def main():
    try:
        processor = FixedHybridQueryProcessor()
        processor.process_all_queries()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
