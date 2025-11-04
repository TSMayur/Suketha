"""
FINAL OPTIMIZED: Native Hybrid Search + Cross-Encoder + Memory Management

Uses Milvus native hybrid_search with:
- ✅ BM25 sparse search (native)
- ✅ Dense vector search
- ✅ RRF ranker (built-in)
- ✅ Cross-encoder reranking
- ✅ Memory management
- ✅ Parallel processing
"""

import json
import zipfile
import logging
import csv
import gc
import psutil
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pymilvus import AnnSearchRequest, RRFRanker, Collection, connections
from .config import client, COLLECTION_NAME, AZURE_MILVUS_URI, AZURE_MILVUS_TOKEN
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import torch

# Configuration
QUERIES_FILE = 'new_queries.json'
OUTPUT_DIR = 'submission_hybrid_native'
ZIP_FILENAME = 'PS04_Shortlisting.zip'
CSV_OUTPUT = 'Final_shortlisting.csv'

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Search parameters
SEARCH_LIMIT = 50
FINAL_TOP_K = 5
CANDIDATE_MULTIPLIER = 2  # Retrieve SEARCH_LIMIT * 2 candidates before final ranking

# Cross-encoder
USE_CROSS_ENCODER = True
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Performance
EMBEDDING_BATCH_SIZE = 36
PROCESS_BATCH_SIZE = 72
MAX_WORKERS = 12
CROSS_ENCODER_BATCH = 36

# Memory management
MEMORY_CHECK_INTERVAL = 25
MAX_MEMORY_PERCENT = 75
CONNECTION_REFRESH_INTERVAL = 1000

# Field names in Milvus collection
SPARSE_VECTOR_FIELD = "sparse_vector"
DENSE_VECTOR_FIELD = "dense_vector"
OUTPUT_FIELDS = ["chunk_id", "doc_id", "doc_name", "chunk_text"]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


class NativeHybridProcessor:
    """Native Milvus hybrid_search with Cross-Encoder reranking"""
    
    def __init__(self):
        logger.info("="*70)
        logger.info("NATIVE HYBRID SEARCH: BM25 + Dense + RRF + Cross-Encoder")
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
        self.collection = None
        self._connect_to_milvus()
        
        # Statistics
        self.queries_processed = 0
        self.total_queries = 0
        self.connection_refreshes = 0
        self.csv_data = []
        
        logger.info(f"⚡ Workers: {MAX_WORKERS}, Batch: {PROCESS_BATCH_SIZE}")
        logger.info("="*70)
        logger.info("✓ Ready - Using native Milvus hybrid_search!\n")
    
    def _connect_to_milvus(self, max_retries=3):
        """Connect to Milvus with cleanup"""
        try:
            if hasattr(self, 'collection') and self.collection:
                self.collection.release()
            connections.disconnect("default")
        except:
            pass
        
        logger.info(f"Connecting to Milvus at {AZURE_MILVUS_URI}")
        
        for attempt in range(1, max_retries + 1):
            try:
                connections.connect("default", uri=AZURE_MILVUS_URI, token=AZURE_MILVUS_TOKEN, timeout=20)
                break
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Connection failed") from e
        
        if not client.has_collection(COLLECTION_NAME):
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
        """Batch embed queries (dense only, BM25 uses raw text)"""
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
        
        result = list(zip(query_nums, query_texts, dense_embeddings))
        
        # Clear references
        del dense_embeddings
        
        return result
    
    def hybrid_search_native(
        self,
        query_text: str,
        dense_vector: np.ndarray
    ) -> List[Dict]:
        """Native Milvus hybrid_search with BM25 + Dense + RRF"""
        try:
            # Prepare sparse search request (BM25 with raw text)
            sparse_search_params = {
                "metric_type": "BM25",
                "params": {
                    "inverted_index_algo": "DAAT_MAXSCORE",
                    "bm25_k1": 1.2,
                    "bm25_b": 0.75
                }
            }
            sparse_request = AnnSearchRequest(
                data=[query_text],
                anns_field=SPARSE_VECTOR_FIELD,
                param=sparse_search_params,
                limit=SEARCH_LIMIT * CANDIDATE_MULTIPLIER
            )
            
            # Prepare dense search request
            dense_search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 200}
            }
            dense_request = AnnSearchRequest(
                data=[dense_vector.tolist()],
                anns_field=DENSE_VECTOR_FIELD,
                param=dense_search_params,
                limit=SEARCH_LIMIT * CANDIDATE_MULTIPLIER
            )
            
            # Perform native hybrid search with RRF
            results = client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=[sparse_request, dense_request],
                ranker=RRFRanker(),
                limit=SEARCH_LIMIT,
                output_fields=OUTPUT_FIELDS
            )
            
            # Format results
            formatted = []
            if results and results[0]:
                for rank, hit in enumerate(results[0], 1):
                    formatted.append({
                        'rank': rank,
                        'chunk_id': hit.entity.get('chunk_id', ''),
                        'doc_id': hit.entity.get('doc_id', ''),
                        'doc_name': hit.entity.get('doc_name', ''),
                        'chunk_text': hit.entity.get('chunk_text', ''),
                        'hybrid_score': float(hit.score),
                        'distance': float(hit.distance) if hasattr(hit, 'distance') else float(hit.score)
                    })
            
            del results
            return formatted
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    def process_single_query(self, query_data: Tuple) -> Dict:
        """Process single query with native hybrid search"""
        query_num, query_text, dense_vec = query_data
        
        try:
            # Native hybrid search
            hybrid_results = self.hybrid_search_native(query_text, dense_vec)
            
            if not hybrid_results:
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
                "candidates": hybrid_results,
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
    
    def cross_encoder_rerank_batch(
        self,
        queries_and_candidates: List[Tuple[str, List[Dict]]]
    ) -> List[List[Dict]]:
        """Batch cross-encoder reranking"""
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
    
    def finalize_results(self, results: List[Dict]) -> List[Dict]:
        """Apply cross-encoder reranking and finalize"""
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
                        'hybrid_score': candidate.get('hybrid_score', 0.0),
                        'initial_rank': candidate.get('rank', 0),
                        'cross_encoder_score': candidate.get('cross_encoder_score', 0.0),
                        'chunk_text': candidate.get('chunk_text', '')[:2000]
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
                
                # CSV data
                for rank, candidate in enumerate(candidates, 1):
                    self.csv_data.append({
                        'query_num': result['query_num'],
                        'query_text': result['query_text'],
                        'rank': rank,
                        'chunk_id': candidate.get('chunk_id', ''),
                        'doc_id': candidate.get('doc_id', ''),
                        'doc_name': candidate.get('doc_name', ''),
                        'hybrid_score': candidate.get('hybrid_score', 0.0),
                        'initial_rank': candidate.get('rank', 0),
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
                    'hybrid_score', 'initial_rank', 'cross_encoder_score', 'chunk_text'
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
        logger.info(f"Method: Native Hybrid Search (BM25 + Dense) with RRF + Cross-Encoder")
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
        logger.info("✅ COMPLETE - Native Hybrid Search!")
        logger.info(f"{'='*70}")
        logger.info(f"Queries: {len(queries)} | Success: {success_count}")
        logger.info(f"Time: {total_time/60:.2f} min | Speed: {len(queries)/total_time:.2f} q/s")
        logger.info(f"Memory growth: {mem_growth:.1f}GB")
        logger.info(f"Connection refreshes: {self.connection_refreshes}")
        logger.info(f"Output: {ZIP_FILENAME}")
        logger.info(f"CSV: {CSV_OUTPUT} ({len(self.csv_data)} rows)")
        logger.info(f"Method: Native Milvus hybrid_search (BM25 + Dense + RRF) + Cross-Encoder")
        logger.info(f"{'='*70}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.collection:
                self.collection.release()
            connections.disconnect("default")
        except:
            pass


def main():
    processor = None
    try:
        processor = NativeHybridProcessor()
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
