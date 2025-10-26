"""
FINAL OPTIMIZED: RRF + Cross-Encoder + Memory Management

Combines best of both worlds:
- ✅ RRF fusion (no weight tuning needed)
- ✅ Cross-encoder reranking (accuracy)
- ✅ Aggressive memory management (for 41k+ queries)
- ✅ Connection refresh (prevents degradation)
- ✅ Parallel processing (speed)
- ✅ CSV output with all scores
"""

import json
import zipfile
import logging
import csv
import hashlib
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Tuple, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pymilvus import MilvusClient, Collection, connections
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import torch

# Stopwords for BM25
STOPWORDS: Set[str] = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
    'the', 'to', 'was', 'were', 'will', 'with', 'this', 'but', 'they',
    'have', 'had', 'what', 'when', 'where', 'who', 'which', 'can',
    'their', 'if', 'out', 'so', 'up', 'been', 'than', 'them', 'she',
}

# Configuration
QUERIES_FILE = 'random_1000_queries.json'
OUTPUT_DIR = 'submission_hybrid_rf'
ZIP_FILENAME = 'PS04_RRF.zip'
CSV_OUTPUT = 'Final_rrf(10_search)_results_with_scores.csv'

MILVUS_URI = "http://4.213.199.69:19530"
TOKEN = "SecurePassword123"
COLLECTION_NAME = "rag_chunks_hybrid"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Search parameters
SEARCH_LIMIT = 10
FINAL_TOP_K = 5
RRF_K = 10  # Standard RRF parameter

# Cross-encoder
USE_CROSS_ENCODER = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Performance (optimized for 8-core 32GB VM)
EMBEDDING_BATCH_SIZE = 36
PROCESS_BATCH_SIZE = 72
MAX_WORKERS = 12
CROSS_ENCODER_BATCH = 36

# Sparse vector config
VOCAB_SIZE = 10_000_000
MIN_TOKEN_LENGTH = 2

# Memory management
MEMORY_CHECK_INTERVAL = 25
MAX_MEMORY_PERCENT = 75
CONNECTION_REFRESH_INTERVAL = 1000

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    """
    Tokenize text for BM25 (matches pipeline approach)
    Simple whitespace + lowercase, minimal filtering
    """
    tokens = text.lower().split()
    # Only filter very short tokens and stopwords
    return [t for t in tokens if len(t) >= MIN_TOKEN_LENGTH and t not in STOPWORDS]


def generate_sparse_vector_bm25(text: str) -> Dict[int, float]:
    """
    Generate BM25-style sparse vector (matches your pipeline exactly)
    
    This matches complete_pipeline_hybrid.py:
    - Tokenize with lowercase + whitespace split
    - Count term frequencies
    - Normalize by max frequency
    - Use hash(token) as index
    """
    tokens = tokenize_text(text)
    
    if not tokens:
        return {}
    
    # Count term frequencies
    term_freq = {}
    for token in tokens:
        token_id = deterministic_token_hash(token,VOCAB_SIZE)
        term_freq[token_id] = term_freq.get(token_id, 0) + 1
    
    # Normalize by max frequency (BM25-like)
    max_freq = max(term_freq.values())
    sparse_vector = {
        idx: freq / max_freq 
        for idx, freq in term_freq.items()
    }
    
    return sparse_vector


def check_memory() -> Dict[str, float]:
    """Check current memory usage"""
    mem = psutil.virtual_memory()
    return {
        'percent': mem.percent,
        'available_gb': mem.available / (1024**3),
        'used_gb': mem.used / (1024**3),
        'total_gb': mem.total / (1024**3)
    }


def aggressive_cleanup():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for _ in range(3):
        gc.collect()
    
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass


def reciprocal_rank_fusion(
    sparse_results: List[Dict],
    dense_results: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    RRF: Combines rankings without needing score normalization
    Formula: RRF_score = sum(1 / (k + rank_i))
    """
    rrf_scores = {}
    
    # Process sparse results
    for rank, result in enumerate(sparse_results, 1):
        chunk_id = result['chunk_id']
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                'chunk_id': chunk_id,
                'doc_id': result.get('doc_id', ''),
                'doc_name': result.get('doc_name', ''),
                'chunk_text': result.get('chunk_text', ''),
                'rrf_score': 0.0,
                'sparse_rank': rank,
                'dense_rank': None,
                'sparse_score': result.get('distance', 0.0),
                'dense_score': None
            }
        rrf_scores[chunk_id]['rrf_score'] += 1.0 / (k + rank)
    
    # Process dense results
    for rank, result in enumerate(dense_results, 1):
        chunk_id = result['chunk_id']
        if chunk_id not in rrf_scores:
            rrf_scores[chunk_id] = {
                'chunk_id': chunk_id,
                'doc_id': result.get('doc_id', ''),
                'doc_name': result.get('doc_name', ''),
                'chunk_text': result.get('chunk_text', ''),
                'rrf_score': 0.0,
                'sparse_rank': None,
                'dense_rank': rank,
                'sparse_score': None,
                'dense_score': result.get('distance', 0.0)
            }
        else:
            rrf_scores[chunk_id]['dense_rank'] = rank
            rrf_scores[chunk_id]['dense_score'] = result.get('distance', 0.0)
        
        rrf_scores[chunk_id]['rrf_score'] += 1.0 / (k + rank)
    
    # Sort by RRF score
    ranked = sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)
    return ranked


class OptimizedRRFProcessor:
    """Final optimized RRF + Cross-Encoder processor with memory management"""
    
    def __init__(self):
        logger.info("="*70)
        logger.info("FINAL OPTIMIZED: RRF + BM25 (Pipeline-Compatible) + Cross-Encoder")
        logger.info("="*70)
        
        # Memory tracking
        self.initial_memory = check_memory()
        logger.info(f"💾 Initial memory: {self.initial_memory['used_gb']:.1f}GB / {self.initial_memory['total_gb']:.1f}GB")
        
        # Load dense embedding model
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        self.embedding_model.eval()
        logger.info("✓ Embedding model loaded")
        
        if USE_CROSS_ENCODER:
            logger.info(f"Loading cross-encoder: {CROSS_ENCODER_MODEL}")
            self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL, device="cpu")
            logger.info("✓ Cross-encoder loaded")
        else:
            self.cross_encoder = None
        
        # Connect to Milvus
        self.client = None
        self.collection = None
        self._connect_to_milvus()
        
        # Statistics
        self.queries_processed = 0
        self.total_queries = 0
        self.connection_refreshes = 0
        self.csv_data = []
        
        logger.info(f"⚡ Workers: {MAX_WORKERS}, Batch: {PROCESS_BATCH_SIZE}")
        logger.info("="*70)
        logger.info("✓ Ready - Using pipeline-compatible BM25!\n")
    
    def _connect_to_milvus(self, max_retries=3):
        """Connect to Milvus with cleanup"""
        try:
            if hasattr(self, 'collection') and self.collection:
                self.collection.release()
            if hasattr(self, 'client') and self.client:
                self.client.close()
            connections.disconnect("default")
        except:
            pass
        
        logger.info(f"Connecting to Milvus at {MILVUS_URI}")
        
        for attempt in range(1, max_retries + 1):
            try:
                self.client = MilvusClient(uri=MILVUS_URI, token=TOKEN, timeout=20)
                connections.connect("default", uri=MILVUS_URI, token=TOKEN, timeout=20)
                break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Connection failed") from e
        
        if not self.client.has_collection(COLLECTION_NAME):
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found!")
        
        self.collection = Collection(COLLECTION_NAME)
        self.collection.load()
        time.sleep(1)
        
        logger.info("✓ Connected to Milvus")
    
    def _refresh_connection_if_needed(self):
        """Periodically refresh Milvus connection"""
        if self.queries_processed % CONNECTION_REFRESH_INTERVAL == 0 and self.queries_processed > 0:
            logger.info("🔄 Refreshing Milvus connection...")
            try:
                self._connect_to_milvus()
                self.connection_refreshes += 1
                logger.info("✓ Connection refreshed")
            except Exception as e:
                logger.error(f"Connection refresh failed: {e}")
    
    def embed_queries_batch(self, queries: List[Tuple[str, str]]) -> List[Tuple]:
        """Batch embed with BM25 sparse vectors (pipeline-compatible)"""
        query_nums = [q[0] for q in queries]
        query_texts = [q[1] for q in queries]
        
        # Dense embeddings with no_grad
        with torch.no_grad():
            dense_embeddings = self.embedding_model.encode(
                query_texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # BM25 sparse embeddings (matches pipeline)
        sparse_embeddings = [generate_sparse_vector_bm25(text) for text in query_texts]
        
        result = list(zip(query_nums, query_texts, dense_embeddings, sparse_embeddings))
        
        # Clear references
        del dense_embeddings
        del sparse_embeddings
        
        return result
    
    def search_single_vector(
        self,
        vector_type: str,
        vector_data,
        limit: int
    ) -> List[Dict]:
        """Search with single vector type"""
        try:
            if vector_type == "dense":
                search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
                results = self.client.search(
                    collection_name=COLLECTION_NAME,
                    data=[vector_data.tolist() if isinstance(vector_data, np.ndarray) else vector_data],
                    limit=limit,
                    search_params=search_params,
                    anns_field="dense_vector",
                    output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"]
                )
            else:  # sparse
                search_params = {"metric_type": "IP", "params": {}}
                results = self.client.search(
                    collection_name=COLLECTION_NAME,
                    data=[vector_data],
                    limit=limit,
                    search_params=search_params,
                    anns_field="sparse_vector",
                    output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"]
                )
            
            formatted = []
            for hit in results[0]:
                formatted.append({
                    'chunk_id': hit['entity'].get('chunk_id', ''),
                    'doc_id': hit['entity'].get('doc_id', ''),
                    'doc_name': hit['entity'].get('doc_name', ''),
                    'chunk_text': hit['entity'].get('chunk_text', ''),
                    'distance': float(hit['distance'])
                })
            
            # Clear references
            del results
            return formatted
            
        except Exception as e:
            logger.error(f"{vector_type} search failed: {e}")
            return []
    
    def hybrid_search_rrf(
        self,
        dense_vector: np.ndarray,
        sparse_vector: Dict[int, float]
    ) -> List[Dict]:
        """Hybrid search using RRF"""
        sparse_results = self.search_single_vector("sparse", sparse_vector, SEARCH_LIMIT)
        dense_results = self.search_single_vector("dense", dense_vector, SEARCH_LIMIT)
        
        rrf_results = reciprocal_rank_fusion(sparse_results, dense_results, k=RRF_K)
        
        # Clear references
        del sparse_results
        del dense_results
        
        return rrf_results
    
    def cross_encoder_rerank_batch(
        self,
        queries_and_candidates: List[Tuple[str, List[Dict]]]
    ) -> List[List[Dict]]:
        """Batch cross-encoder reranking with memory management"""
        if not self.cross_encoder or not queries_and_candidates:
            return [candidates for _, candidates in queries_and_candidates]
        
        # Flatten all pairs
        all_pairs = []
        pair_boundaries = [0]
        
        for query, candidates in queries_and_candidates:
            pairs = [[query, c['chunk_text']] for c in candidates]
            all_pairs.extend(pairs)
            pair_boundaries.append(len(all_pairs))
        
        # Batch predict with no_grad
        with torch.no_grad():
            all_scores = self.cross_encoder.predict(
                all_pairs,
                batch_size=CROSS_ENCODER_BATCH,
                show_progress_bar=False
            )
        
        # Reconstruct results
        reranked_results = []
        for i, (query, candidates) in enumerate(queries_and_candidates):
            start_idx = pair_boundaries[i]
            end_idx = pair_boundaries[i + 1]
            scores = all_scores[start_idx:end_idx]
            
            for candidate, score in zip(candidates, scores):
                candidate['cross_encoder_score'] = float(score)
            
            reranked = sorted(candidates, key=lambda x: x['cross_encoder_score'], reverse=True)
            reranked_results.append(reranked)
        
        # Clear references
        del all_pairs
        del all_scores
        del pair_boundaries
        
        return reranked_results
    
    def process_single_query(self, query_data: Tuple) -> Dict:
        """Process single query"""
        query_num, query_text, dense_vec, sparse_vec = query_data
        
        try:
            # RRF hybrid search
            rrf_results = self.hybrid_search_rrf(dense_vec, sparse_vec)
            
            if not rrf_results:
                return {
                    "query_num": query_num,
                    "query_text": query_text,
                    "candidates": [],
                    "doc_ids": [],
                    "success": True
                }
            
            return {
                "query_num": query_num,
                "query_text": query_text,
                "candidates": rrf_results[:SEARCH_LIMIT],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"✗ Query {query_num} failed: {e}")
            return {
                "query_num": query_num,
                "query_text": query_text,
                "candidates": [],
                "doc_ids": [],
                "success": False,
                "error": str(e)
            }
    
    def search_parallel_batch(self, embedded_queries: List) -> List[Dict]:
        """Search in parallel"""
        start = time.time()
        
        results = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.process_single_query, qdata): qdata[0]
                for qdata in embedded_queries
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Query failed: {e}")
        
        duration = time.time() - start
        speed = len(embedded_queries) / duration if duration > 0 else 0
        logger.info(f"  Search: {len(embedded_queries)} queries in {duration:.2f}s ({speed:.1f} q/s)")
        
        return results
    
    def finalize_results(self, results: List[Dict]) -> List[Dict]:
        """Batch cross-encoder reranking and finalize"""
        successful = [r for r in results if r['success'] and r.get('candidates')]
        
        if USE_CROSS_ENCODER and successful:
            queries_and_candidates = [
                (r['query_text'], r['candidates'])
                for r in successful
            ]
            
            start = time.time()
            reranked_batches = self.cross_encoder_rerank_batch(queries_and_candidates)
            duration = time.time() - start
            logger.info(f"  Rerank: {len(successful)} queries in {duration:.2f}s")
            
            # Update results and collect CSV data
            for result, reranked in zip(successful, reranked_batches):
                final_results = reranked[:FINAL_TOP_K]
                
                # Extract doc_ids
                doc_ids = []
                seen = set()
                for candidate in final_results:
                    doc_name = candidate.get('doc_name', '')
                    if doc_name and doc_name not in seen:
                        doc_ids.append(doc_name)
                        seen.add(doc_name)
                
                result['doc_ids'] = doc_ids
                
                # CSV data
                for rank, candidate in enumerate(final_results, 1):
                    self.csv_data.append({
                        'query_num': result['query_num'],
                        'query_text': result['query_text'],
                        'rank': rank,
                        'chunk_id': candidate.get('chunk_id', ''),
                        'doc_id': candidate.get('doc_id', ''),
                        'doc_name': candidate.get('doc_name', ''),
                        'rrf_score': candidate.get('rrf_score', 0.0),
                        'sparse_rank': candidate.get('sparse_rank') or 0,
                        'dense_rank': candidate.get('dense_rank') or 0,
                        'cross_encoder_score': candidate.get('cross_encoder_score', 0.0),
                        'chunk_text': candidate.get('chunk_text', '')[:500]
                    })
                
                result['candidates'] = None
            
            del queries_and_candidates
            del reranked_batches
        else:
            # No reranking
            for result in successful:
                candidates = result.get('candidates', [])[:FINAL_TOP_K]
                
                doc_ids = []
                seen = set()
                for candidate in candidates:
                    doc_name = candidate.get('doc_name', '')
                    if doc_name and doc_name not in seen:
                        doc_ids.append(doc_name)
                        seen.add(doc_name)
                
                result['doc_ids'] = doc_ids
                
                # CSV data (no cross-encoder scores)
                for rank, candidate in enumerate(candidates, 1):
                    self.csv_data.append({
                        'query_num': result['query_num'],
                        'query_text': result['query_text'],
                        'rank': rank,
                        'chunk_id': candidate.get('chunk_id', ''),
                        'doc_id': candidate.get('doc_id', ''),
                        'doc_name': candidate.get('doc_name', ''),
                        'rrf_score': candidate.get('rrf_score', 0.0),
                        'sparse_rank': candidate.get('sparse_rank') or 0,
                        'dense_rank': candidate.get('dense_rank') or 0,
                        'cross_encoder_score': 0.0,
                        'chunk_text': candidate.get('chunk_text', '')[:500]
                    })
                
                result['candidates'] = None
        
        # Final cleanup
        for result in results:
            result.pop('candidates', None)
        
        return results
    
    def write_results_batch(self, results: List[Dict], output_path: Path):
        """Write JSON results"""
        for result in results:
            try:
                output_data = {
                    "query": result["query_text"],
                    "response": result.get("doc_ids", [])
                }
                output_file = output_path / f"query_{result['query_num']}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=4)
            except Exception as e:
                logger.error(f"Write failed: {e}")
    
    def write_csv_results(self):
        """Write CSV with all scores"""
        logger.info(f"\n📊 Writing CSV to {CSV_OUTPUT}...")
        
        try:
            sorted_data = sorted(self.csv_data, key=lambda x: (int(x['query_num']), x['rank']))
            
            with open(CSV_OUTPUT, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'query_num', 'query_text', 'rank', 'chunk_id', 'doc_id', 'doc_name',
                    'rrf_score', 'sparse_rank', 'dense_rank', 'cross_encoder_score', 'chunk_text'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sorted_data)
            
            logger.info(f"✅ CSV: {len(sorted_data)} rows")
        except Exception as e:
            logger.error(f"CSV failed: {e}")
    
    def process_all_queries(self):
        """Main pipeline with memory management"""
        pipeline_start = time.time()
        
        # Load queries
        with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
            queries_data = json.load(f)
        
        queries = [
            (item.get("query_num"), item.get("query"))
            for item in queries_data
            if item.get("query_num") and item.get("query")
        ]
        
        self.total_queries = len(queries)
        
        logger.info(f"Processing {self.total_queries} queries")
        logger.info(f"Method: RRF (k={RRF_K}) + Cross-Encoder")
        logger.info("="*70)
        
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        all_results = []
        recent_batch_times = []
        
        # Process in batches
        for process_start in range(0, len(queries), PROCESS_BATCH_SIZE):
            process_end = min(process_start + PROCESS_BATCH_SIZE, len(queries))
            process_batch = queries[process_start:process_end]
            
            batch_start_time = time.time()
            logger.info(f"\n{'='*70}")
            logger.info(f"BATCH: {process_start+1}-{process_end} ({len(process_batch)} queries)")
            logger.info(f"{'='*70}")
            
            # Refresh connection periodically
            self._refresh_connection_if_needed()
            
            batch_results = []
            
            # Subdivide into embedding batches
            for embed_start in range(0, len(process_batch), EMBEDDING_BATCH_SIZE):
                embed_end = min(embed_start + EMBEDDING_BATCH_SIZE, len(process_batch))
                embed_batch = process_batch[embed_start:embed_end]
                
                embedded = self.embed_queries_batch(embed_batch)
                search_results = self.search_parallel_batch(embedded)
                batch_results.extend(search_results)
                
                del embedded
            
            # Batch reranking
            batch_results = self.finalize_results(batch_results)
            
            # Write results
            self.write_results_batch(batch_results, output_path)
            
            all_results.extend(batch_results)
            self.queries_processed = len(all_results)
            
            # Statistics
            batch_time = time.time() - batch_start_time
            recent_batch_times.append(batch_time)
            if len(recent_batch_times) > 10:
                recent_batch_times.pop(0)
            
            avg_recent = sum(recent_batch_times) / len(recent_batch_times)
            elapsed = time.time() - pipeline_start
            progress = self.queries_processed / self.total_queries * 100
            overall_speed = self.queries_processed / elapsed
            
            queries_remaining = self.total_queries - self.queries_processed
            batches_remaining = queries_remaining / PROCESS_BATCH_SIZE
            eta_seconds = batches_remaining * avg_recent
            
            logger.info(f"\n📊 Batch: {batch_time:.2f}s (avg: {avg_recent:.2f}s)")
            logger.info(f"📊 Progress: {self.queries_processed}/{self.total_queries} ({progress:.1f}%)")
            logger.info(f"⚡ Speed: {overall_speed:.2f} q/s")
            logger.info(f"⏱️  ETA: {eta_seconds/60:.1f} min")
            
            # Memory management
            if self.queries_processed % MEMORY_CHECK_INTERVAL == 0:
                mem = check_memory()
                mem_growth = mem['used_gb'] - self.initial_memory['used_gb']
                logger.info(f"💾 Memory: {mem['percent']:.1f}% ({mem['used_gb']:.1f}GB, +{mem_growth:.1f}GB)")
                
                if mem['percent'] > MAX_MEMORY_PERCENT:
                    logger.warning("⚠️  High memory! Cleanup...")
                    aggressive_cleanup()
                    mem_after = check_memory()
                    logger.info(f"💾 After: {mem_after['percent']:.1f}%")
            
            # Cleanup after batch
            aggressive_cleanup()
            del batch_results
        
        # Write CSV
        self.write_csv_results()
        
        # Create zip
        logger.info(f"\n📦 Creating {ZIP_FILENAME}...")
        with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in output_path.glob("*.json"):
                zf.write(file, arcname=file.name)
        
        # Final stats
        total_time = time.time() - pipeline_start
        success_count = sum(1 for r in all_results if r["success"])
        final_mem = check_memory()
        mem_growth = final_mem['used_gb'] - self.initial_memory['used_gb']
        
        logger.info(f"\n{'='*70}")
        logger.info("✅ COMPLETE - Pipeline-Compatible BM25!")
        logger.info(f"{'='*70}")
        logger.info(f"Queries: {len(queries)} | Success: {success_count}")
        logger.info(f"Time: {total_time/60:.2f} min | Speed: {len(queries)/total_time:.2f} q/s")
        logger.info(f"Memory growth: {mem_growth:.1f}GB")
        logger.info(f"Connection refreshes: {self.connection_refreshes}")
        logger.info(f"Output: {ZIP_FILENAME}")
        logger.info(f"CSV: {CSV_OUTPUT} ({len(self.csv_data)} rows)")
        logger.info(f"Method: BM25 (hash-based, max-freq normalized) + RRF + Cross-Encoder")
        logger.info(f"{'='*70}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.collection:
                self.collection.release()
            if self.client:
                self.client.close()
            connections.disconnect("default")
        except:
            pass


def main():
    processor = None
    try:
        processor = OptimizedRRFProcessor()
        processor.process_all_queries()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        if processor:
            processor.cleanup()
        aggressive_cleanup()


if __name__ == "__main__":
    main()
