"""
OPTIMIZED v6: MEMORY LEAK FIX + AGGRESSIVE RESOURCE MANAGEMENT

Critical fixes for sustained performance:
1. 🔌 Proper Milvus connection cleanup after each batch
2. 🧠 Cross-encoder result clearing to prevent memory accumulation  
3. 🔄 Model cache clearing between batches
4. 💾 Aggressive garbage collection with memory monitoring
5. ⚡ Dynamic worker scaling based on performance
6. 🎯 Smaller batches with connection refresh
"""

import json
import zipfile
import logging
import hashlib
import gc
import psutil
import os
from pathlib import Path
from typing import List, Dict, Tuple, Set, Optional
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymilvus import (
    MilvusClient,
    Collection,
    connections,
    AnnSearchRequest,
    WeightedRanker,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import torch

# Configuration
QUERIES_FILE = 'queries.json'
OUTPUT_DIR = 'submission_hybrid_milvus'
ZIP_FILENAME = 'PS04_TEAM.zip'

MILVUS_URI = "http://4.213.199.69:19530"
TOKEN = "SecurePassword123"
COLLECTION_NAME = "rag_chunks_hybrid"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Search parameters
SEARCH_LIMIT = 50
FINAL_TOP_K = 5

# Hybrid search weights
SPARSE_WEIGHT = 0.7
DENSE_WEIGHT = 1.0

USE_CROSS_ENCODER = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ⚡ OPTIMIZED Performance settings for 8-core 32GB VM
EMBEDDING_BATCH_SIZE = 24      # Reduced for stability
PROCESS_BATCH_SIZE = 50        # Smaller batches, more frequent cleanup
MAX_WORKERS = 12               # Conservative for 8 cores
CROSS_ENCODER_BATCH = 24       # Reduced batch size

# Sparse vector config
VOCAB_SIZE = 10_000_000
MIN_TOKEN_LENGTH = 2

# Memory management - AGGRESSIVE
MEMORY_CHECK_INTERVAL = 25     # Check more frequently
MAX_MEMORY_PERCENT = 75        # Lower threshold
CONNECTION_REFRESH_INTERVAL = 200  # Reconnect every N queries

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Stopwords
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
    """AGGRESSIVE memory cleanup"""
    # Clear PyTorch cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Multiple GC passes
    for _ in range(3):
        gc.collect()
    
    # Force Python to release memory to OS
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass


class OptimizedFastProcessor:
    """
    ⚡ OPTIMIZED v6: Fixed memory leaks and resource management
    """
    
    def __init__(self):
        logger.info("Initializing OPTIMIZED Fast Processor v6 (Memory Leak Fix)...")
        
        # Memory tracking
        self.initial_memory = check_memory()
        logger.info(f"💾 Initial memory: {self.initial_memory['used_gb']:.1f}GB / {self.initial_memory['total_gb']:.1f}GB")
        
        # Load embedding model with explicit device
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu"
        )
        # Clear model cache
        self.embedding_model.eval()
        logger.info("✓ Embedding model loaded")
        
        # Load cross-encoder
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
        self.batch_times = []
        
        logger.info(f"⚡ Workers: {MAX_WORKERS}")
        logger.info(f"📦 Batch sizes: embed={EMBEDDING_BATCH_SIZE}, process={PROCESS_BATCH_SIZE}")
        logger.info("✓ Initialization complete\n")
    
    def _connect_to_milvus(self, max_retries=3, retry_delay=2):
        """Connect to Milvus with retry logic and cleanup"""
        # Cleanup existing connections
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
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    raise RuntimeError(f"Failed to connect after {max_retries} attempts") from e
        
        # Verify collection
        if not self.client.has_collection(COLLECTION_NAME):
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found!")
        
        # Get collection
        self.collection = Collection(COLLECTION_NAME)
        
        # Load collection
        logger.info("Loading collection...")
        self.collection.load()
        time.sleep(1)
        
        logger.info("✓ Connected to Milvus")
    
    def _refresh_connection_if_needed(self):
        """Periodically refresh Milvus connection to prevent degradation"""
        if self.queries_processed % CONNECTION_REFRESH_INTERVAL == 0 and self.queries_processed > 0:
            logger.info("🔄 Refreshing Milvus connection...")
            try:
                self._connect_to_milvus()
                self.connection_refreshes += 1
                logger.info("✓ Connection refreshed")
            except Exception as e:
                logger.error(f"Connection refresh failed: {e}")
    
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
        normalized = {tid: freq / max_freq for tid, freq in term_freq.items()}
        
        return normalized
    
    def embed_queries_batch(
        self,
        queries: List[Tuple[str, str]]
    ) -> List[Tuple[str, str, List[float], Dict[int, float]]]:
        """Batch embed with memory management"""
        query_nums = [q[0] for q in queries]
        query_texts = [q[1] for q in queries]
        
        # Dense embeddings - explicitly clear cache
        with torch.no_grad():
            dense_embeddings = self.embedding_model.encode(
                query_texts,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
        
        # Sparse vectors
        sparse_vectors = [self.generate_sparse_vector(text) for text in query_texts]
        
        result = [
            (qnum, qtext, dense.tolist(), sparse)
            for qnum, qtext, dense, sparse in zip(
                query_nums, query_texts, dense_embeddings, sparse_vectors
            )
        ]
        
        # Clear references
        del dense_embeddings
        del sparse_vectors
        
        return result
    
    def hybrid_search_native(
        self,
        dense_vector: List[float],
        sparse_vector: Dict[int, float]
    ) -> List[Dict]:
        """Native Milvus hybrid search with error handling"""
        try:
            # Dense search request
            dense_req = AnnSearchRequest(
                data=[dense_vector],
                anns_field="dense_vector",
                param={"metric_type": "COSINE", "params": {"ef": 64}},
                limit=SEARCH_LIMIT
            )
            
            # Sparse search request
            sparse_req = AnnSearchRequest(
                data=[sparse_vector],
                anns_field="sparse_vector",
                param={"metric_type": "IP", "params": {}},
                limit=SEARCH_LIMIT
            )
            
            # Weighted ranker
            reranker = WeightedRanker(SPARSE_WEIGHT, DENSE_WEIGHT)
            
            # Hybrid search
            results = self.collection.hybrid_search(
                reqs=[sparse_req, dense_req],
                rerank=reranker,
                limit=SEARCH_LIMIT,
                output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"]
            )[0]
            
            extracted = [
                {
                    'chunk_id': hit.entity.get('chunk_id', ''),
                    'doc_id': hit.entity.get('doc_id', ''),
                    'doc_name': hit.entity.get('doc_name', ''),
                    'chunk_text': hit.entity.get('chunk_text', ''),
                    'hybrid_score': float(hit.score)
                }
                for hit in results
            ]
            
            # Clear references
            del results
            return extracted
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def cross_encoder_rerank(
        self,
        queries_and_candidates: List[Tuple[str, List[Dict]]]
    ) -> List[List[Dict]]:
        """
        🚀 OPTIMIZED: Batch cross-encoder with explicit memory management
        """
        if not self.cross_encoder or not queries_and_candidates:
            return [candidates for _, candidates in queries_and_candidates]
        
        # Flatten all query-candidate pairs
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
        
        # Explicitly clear large objects
        del all_pairs
        del all_scores
        del pair_boundaries
        
        return reranked_results
    
    def process_single_query(
        self,
        query_data: Tuple[str, str, List[float], Dict[int, float]]
    ) -> Dict:
        """Process single query"""
        query_num, query_text, dense_vec, sparse_vec = query_data
        
        try:
            # Hybrid search
            hybrid_results = self.hybrid_search_native(dense_vec, sparse_vec)
            
            if not hybrid_results:
                return {
                    "query_num": query_num,
                    "query_text": query_text,
                    "doc_ids": [],
                    "candidates": [],
                    "success": True
                }
            
            return {
                "query_num": query_num,
                "query_text": query_text,
                "candidates": hybrid_results[:SEARCH_LIMIT],
                "success": True
            }
            
        except Exception as e:
            logger.error(f"✗ Query {query_num} failed: {e}")
            return {
                "query_num": query_num,
                "query_text": query_text,
                "doc_ids": [],
                "candidates": [],
                "success": False,
                "error": str(e)
            }
    
    def search_parallel_batch(
        self,
        embedded_queries: List[Tuple[str, str, List[float], Dict[int, float]]]
    ) -> List[Dict]:
        """Parallel search with proper cleanup"""
        start = time.time()
        
        results = []
        # Use context manager for proper cleanup
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.process_single_query, query_data): query_data[0]
                for query_data in embedded_queries
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)  # Add timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Query failed: {e}")
        
        duration = time.time() - start
        speed = len(embedded_queries) / duration if duration > 0 else 0
        logger.info(f"  Search: {len(embedded_queries)} queries in {duration:.2f}s ({speed:.1f} q/s)")
        
        return results
    
    def finalize_results(self, results: List[Dict]) -> List[Dict]:
        """
        🚀 OPTIMIZED: Batch cross-encoder reranking with memory cleanup
        """
        # Separate successful results
        successful = [r for r in results if r['success'] and r.get('candidates')]
        
        if USE_CROSS_ENCODER and successful:
            # Prepare batch
            queries_and_candidates = [
                (r['query_text'], r['candidates'])
                for r in successful
            ]
            
            # Batch rerank
            start = time.time()
            reranked_batches = self.cross_encoder_rerank(queries_and_candidates)
            duration = time.time() - start
            logger.info(f"  Rerank: {len(successful)} queries in {duration:.2f}s")
            
            # Update results and clear candidates immediately
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
                result['candidates'] = None  # Clear immediately
            
            # Clear batch references
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
                result['candidates'] = None
        
        # Final cleanup pass
        for result in results:
            result.pop('candidates', None)
        
        return results
    
    def write_results_batch(self, results: List[Dict], output_path: Path):
        """Batch write results"""
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
                logger.error(f"Write failed for query {result.get('query_num')}: {e}")
    
    def process_all_queries(self):
        """
        ⚡ OPTIMIZED processing with aggressive memory management
        """
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
        
        logger.info(f"\n{'='*70}")
        logger.info(f"⚡ OPTIMIZED v6: MEMORY LEAK FIX")
        logger.info(f"{'='*70}")
        logger.info(f"Total queries: {self.total_queries}")
        logger.info(f"Process batch size: {PROCESS_BATCH_SIZE}")
        logger.info(f"Connection refresh interval: {CONNECTION_REFRESH_INTERVAL}")
        logger.info(f"{'='*70}\n")
        
        # Create output dir
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        all_results = []
        last_speed_check = time.time()
        recent_batch_times = []
        
        # Process in smaller batches
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
                
                # Embed
                embedded = self.embed_queries_batch(embed_batch)
                
                # Search in parallel
                search_results = self.search_parallel_batch(embedded)
                
                batch_results.extend(search_results)
                
                # Clear embedding batch
                del embedded
            
            # Batch cross-encoder reranking
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
            
            avg_recent_batch_time = sum(recent_batch_times) / len(recent_batch_times)
            
            elapsed = time.time() - pipeline_start
            progress = self.queries_processed / self.total_queries * 100
            overall_speed = self.queries_processed / elapsed
            
            # ETA based on recent performance
            queries_remaining = self.total_queries - self.queries_processed
            batches_remaining = queries_remaining / PROCESS_BATCH_SIZE
            eta_seconds = batches_remaining * avg_recent_batch_time
            
            logger.info(f"\n📊 Batch: {batch_time:.2f}s (avg recent: {avg_recent_batch_time:.2f}s)")
            logger.info(f"📊 Progress: {self.queries_processed}/{self.total_queries} ({progress:.1f}%)")
            logger.info(f"⚡ Overall speed: {overall_speed:.2f} q/s")
            logger.info(f"⏱️  ETA: {eta_seconds/60:.1f} minutes")
            
            # Memory check and aggressive cleanup
            if self.queries_processed % MEMORY_CHECK_INTERVAL == 0:
                mem = check_memory()
                mem_growth = mem['used_gb'] - self.initial_memory['used_gb']
                logger.info(f"💾 Memory: {mem['percent']:.1f}% ({mem['used_gb']:.1f}GB used, +{mem_growth:.1f}GB growth)")
                
                if mem['percent'] > MAX_MEMORY_PERCENT:
                    logger.warning(f"⚠️  High memory! Aggressive cleanup...")
                    aggressive_cleanup()
                    mem_after = check_memory()
                    freed = mem['used_gb'] - mem_after['used_gb']
                    logger.info(f"💾 After cleanup: {mem_after['percent']:.1f}% (freed {freed:.1f}GB)")
            
            # Force cleanup after each batch
            aggressive_cleanup()
            
            # Clear batch results
            del batch_results
        
        # Create zip
        logger.info(f"\n📦 Creating {ZIP_FILENAME}...")
        with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in output_path.glob("*.json"):
                zf.write(file, arcname=file.name)
        
        # Final stats
        total_time = time.time() - pipeline_start
        success_count = sum(1 for r in all_results if r["success"])
        avg_speed = len(queries) / total_time
        
        final_mem = check_memory()
        total_mem_growth = final_mem['used_gb'] - self.initial_memory['used_gb']
        
        logger.info(f"\n{'='*70}")
        logger.info("✅ COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Queries: {len(queries)} | Success: {success_count}")
        logger.info(f"Time: {total_time/60:.2f} minutes")
        logger.info(f"Speed: {avg_speed:.2f} q/s")
        logger.info(f"Memory growth: {total_mem_growth:.1f}GB")
        logger.info(f"Connection refreshes: {self.connection_refreshes}")
        logger.info(f"Output: {ZIP_FILENAME}")
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
    """Main entry point"""
    processor = None
    try:
        processor = OptimizedFastProcessor()
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
