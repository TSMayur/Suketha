"""
100% MATCHING Milvus Documentation Format
- Exact same search patterns as official docs
- Same metric types, params, request order
- Using mpnet for embeddings (as requested)
- Custom sparse vectors (SHA256 hash-based)
"""

import json
import zipfile
import logging
import hashlib
import gc
import psutil
import csv
from pathlib import Path
from typing import List, Dict, Tuple, Set
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymilvus import (
    connections,
    utility,
    Collection,
    AnnSearchRequest,
    WeightedRanker,
)
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

# ============================================================================
# CONFIGURATION
# ============================================================================

# File paths
QUERIES_FILE = 'random_1000_queries.json'
OUTPUT_DIR = 'subm'
ZIP_FILENAME = 'PS04.zip'
CSV_OUTPUT = 'new_milvus.csv'

# Milvus connection
MILVUS_URI = "http://4.213.199.69:19530"
TOKEN = "SecurePassword123"
COLLECTION_NAME = "rag_chunks_hybrid"

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Search parameters
SEARCH_LIMIT = 50
FINAL_TOP_K = 5

# Hybrid search weights (same as docs)
SPARSE_WEIGHT = 0.7
DENSE_WEIGHT = 1.0

# Feature flags
USE_CROSS_ENCODER = True

# Performance settings
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

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# STOPWORDS
# ============================================================================

STOPWORDS: Set[str] = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'or', 'that',
    'the', 'to', 'was', 'were', 'will', 'with', 'this', 'but', 'they',
    'have', 'had', 'what', 'when', 'where', 'who', 'which', 'can',
    'their', 'if', 'out', 'so', 'up', 'been', 'than', 'them', 'she',
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


def generate_sparse_vector(text: str) -> Dict[int, float]:
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


# ============================================================================
# MILVUS CONNECTION (100% SAME AS DOCS)
# ============================================================================

def connect_to_milvus(uri: str, token: str = None):
    """Connect to Milvus - exactly as docs"""
    logger.info(f"Connecting to Milvus at {uri}")
    if token:
        connections.connect(uri=uri, token=token)
    else:
        connections.connect(uri=uri)
    logger.info("✓ Connected to Milvus")


def get_collection(collection_name: str) -> Collection:
    """Get and load collection - exactly as docs"""
    if not utility.has_collection(collection_name):
        raise RuntimeError(f"Collection '{collection_name}' not found!")
    
    col = Collection(collection_name)
    col.load()
    logger.info(f"✓ Collection '{collection_name}' loaded")
    return col


# ============================================================================
# SEARCH FUNCTIONS (100% MATCHING DOCS)
# ============================================================================

def dense_search(col, query_dense_embedding, limit=10):
    """Dense search - EXACTLY as documentation"""
    search_params = {"metric_type": "IP", "params": {}}
    res = col.search(
        [query_dense_embedding],
        anns_field="dense_vector",
        limit=limit,
        output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"],
        param=search_params,
    )[0]
    return [
        {
            'chunk_id': hit.entity.get('chunk_id', ''),
            'doc_id': hit.entity.get('doc_id', ''),
            'doc_name': hit.entity.get('doc_name', ''),
            'chunk_text': hit.entity.get('chunk_text', ''),
            'score': float(hit.score),
        }
        for hit in res
    ]


def sparse_search(col, query_sparse_embedding, limit=10):
    """Sparse search - EXACTLY as documentation"""
    search_params = {
        "metric_type": "IP",
        "params": {},
    }
    res = col.search(
        [query_sparse_embedding],
        anns_field="sparse_vector",
        limit=limit,
        output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"],
        param=search_params,
    )[0]
    return [
        {
            'chunk_id': hit.entity.get('chunk_id', ''),
            'doc_id': hit.entity.get('doc_id', ''),
            'doc_name': hit.entity.get('doc_name', ''),
            'chunk_text': hit.entity.get('chunk_text', ''),
            'score': float(hit.score),
        }
        for hit in res
    ]


def hybrid_search(
    col,
    query_dense_embedding,
    query_sparse_embedding,
    sparse_weight=1.0,
    dense_weight=1.0,
    limit=10,
):
    """Hybrid search - EXACTLY as documentation"""
    dense_search_params = {"metric_type": "IP", "params": {}}
    dense_req = AnnSearchRequest(
        [query_dense_embedding], "dense_vector", dense_search_params, limit=limit
    )
    sparse_search_params = {"metric_type": "IP", "params": {}}
    sparse_req = AnnSearchRequest(
        [query_sparse_embedding], "sparse_vector", sparse_search_params, limit=limit
    )
    rerank = WeightedRanker(sparse_weight, dense_weight)
    res = col.hybrid_search(
        [sparse_req, dense_req], rerank=rerank, limit=limit, output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"]
    )[0]
    return [
        {
            'chunk_id': hit.entity.get('chunk_id', ''),
            'doc_id': hit.entity.get('doc_id', ''),
            'doc_name': hit.entity.get('doc_name', ''),
            'chunk_text': hit.entity.get('chunk_text', ''),
            'score': float(hit.score),
        }
        for hit in res
    ]


# ============================================================================
# EMBEDDING GENERATION
# ============================================================================

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load sentence transformer model"""
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device="cpu")
    model.eval()
    logger.info("✓ Embedding model loaded")
    return model


def load_cross_encoder(model_name: str) -> CrossEncoder:
    """Load cross-encoder model"""
    logger.info(f"Loading cross-encoder: {model_name}")
    model = CrossEncoder(model_name, device="cpu")
    logger.info("✓ Cross-encoder loaded")
    return model


def generate_embeddings(
    embedding_model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 24
) -> List[List[float]]:
    """Generate dense embeddings"""
    with torch.no_grad():
        embeddings = embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    return embeddings.tolist()


# ============================================================================
# CROSS-ENCODER RERANKING
# ============================================================================

def cross_encoder_rerank(
    cross_encoder: CrossEncoder,
    query: str,
    candidates: List[Dict],
    batch_size: int = 24
) -> List[Dict]:
    """Rerank using cross-encoder"""
    if not candidates:
        return []
    
    pairs = [[query, c['chunk_text']] for c in candidates]
    
    with torch.no_grad():
        scores = cross_encoder.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False
        )
    
    for candidate, score in zip(candidates, scores):
        candidate['cross_encoder_score'] = float(score)
    
    reranked = sorted(candidates, key=lambda x: x['cross_encoder_score'], reverse=True)
    
    return reranked


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_single_query(
    col: Collection,
    query_num: str,
    query_text: str,
    dense_embedding: List[float],
    sparse_embedding: Dict[int, float]
) -> Dict:
    """Process a single query using hybrid search"""
    try:
        # Use hybrid_search function (same as docs)
        results = hybrid_search(
            col,
            dense_embedding,
            sparse_embedding,
            sparse_weight=SPARSE_WEIGHT,
            dense_weight=DENSE_WEIGHT,
            limit=SEARCH_LIMIT
        )
        
        return {
            "query_num": query_num,
            "query_text": query_text,
            "candidates": results,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"✗ Query {query_num} failed: {e}")
        return {
            "query_num": query_num,
            "query_text": query_text,
            "candidates": [],
            "success": False,
            "error": str(e)
        }


def process_queries_parallel(
    col: Collection,
    queries_data: List[Tuple[str, str, List[float], Dict[int, float]]],
    max_workers: int = 12
) -> List[Dict]:
    """Process queries in parallel"""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_query,
                col,
                query_num,
                query_text,
                dense_emb,
                sparse_emb
            ): query_num
            for query_num, query_text, dense_emb, sparse_emb in queries_data
        }
        
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                logger.error(f"Query processing failed: {e}")
    
    return results


# ============================================================================
# RESULT FINALIZATION
# ============================================================================

def finalize_results_with_reranking(
    cross_encoder: CrossEncoder,
    results: List[Dict],
    csv_data: List[Dict]
) -> List[Dict]:
    """Apply cross-encoder reranking and extract top-K doc_ids"""
    for result in results:
        if not result['success'] or not result.get('candidates'):
            result['doc_ids'] = []
            continue
        
        # Rerank
        reranked = cross_encoder_rerank(
            cross_encoder,
            result['query_text'],
            result['candidates'],
            batch_size=CROSS_ENCODER_BATCH
        )
        
        # Take top-K
        top_k = reranked[:FINAL_TOP_K]
        
        # Extract unique doc_ids
        doc_ids = []
        seen = set()
        for candidate in top_k:
            doc_name = candidate.get('doc_name', '')
            if doc_name and doc_name not in seen:
                doc_ids.append(doc_name)
                seen.add(doc_name)
        
        result['doc_ids'] = doc_ids
        
        # Collect CSV data
        for rank, candidate in enumerate(reranked, 1):
            csv_data.append({
                'query_num': result['query_num'],
                'query_text': result['query_text'],
                'rank': rank,
                'chunk_id': candidate.get('chunk_id', ''),
                'doc_id': candidate.get('doc_id', ''),
                'doc_name': candidate.get('doc_name', ''),
                'hybrid_score': candidate.get('score', 0.0),
                'cross_encoder_score': candidate.get('cross_encoder_score', 0.0),
                'chunk_text': candidate.get('chunk_text', '')[:500]
            })
        
        result['candidates'] = None
    
    return results


def finalize_results_without_reranking(
    results: List[Dict],
    csv_data: List[Dict]
) -> List[Dict]:
    """Extract top-K doc_ids without reranking"""
    for result in results:
        if not result['success'] or not result.get('candidates'):
            result['doc_ids'] = []
            continue
        
        candidates = result['candidates'][:FINAL_TOP_K]
        
        # Extract unique doc_ids
        doc_ids = []
        seen = set()
        for candidate in candidates:
            doc_name = candidate.get('doc_name', '')
            if doc_name and doc_name not in seen:
                doc_ids.append(doc_name)
                seen.add(doc_name)
        
        result['doc_ids'] = doc_ids
        
        # Collect CSV data
        for rank, candidate in enumerate(result['candidates'], 1):
            csv_data.append({
                'query_num': result['query_num'],
                'query_text': result['query_text'],
                'rank': rank,
                'chunk_id': candidate.get('chunk_id', ''),
                'doc_id': candidate.get('doc_id', ''),
                'doc_name': candidate.get('doc_name', ''),
                'hybrid_score': candidate.get('score', 0.0),
                'cross_encoder_score': 0.0,
                'chunk_text': candidate.get('chunk_text', '')[:500]
            })
        
        result['candidates'] = None
    
    return results


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def write_results_to_json(results: List[Dict], output_dir: Path):
    """Write results to individual JSON files"""
    output_dir.mkdir(exist_ok=True)
    
    for result in results:
        try:
            output_data = {
                "query": result["query_text"],
                "response": result.get("doc_ids", [])
            }
            
            output_file = output_dir / f"query_{result['query_num']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
        except Exception as e:
            logger.error(f"Write failed for query {result.get('query_num')}: {e}")


def write_csv_results(csv_data: List[Dict], csv_file: str):
    """Write all search results to CSV"""
    logger.info(f"\n📊 Writing CSV results to {csv_file}...")
    
    try:
        sorted_data = sorted(csv_data, key=lambda x: (int(x['query_num']), x['rank']))
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'query_num', 'query_text', 'rank', 'chunk_id',
                'doc_id', 'doc_name', 'hybrid_score', 'cross_encoder_score',
                'chunk_text'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            writer.writerows(sorted_data)
        
        logger.info(f"✅ CSV written: {len(sorted_data)} rows")
    except Exception as e:
        logger.error(f"❌ Failed to write CSV: {e}")


def create_submission_zip(output_dir: Path, zip_file: str):
    """Create submission ZIP file"""
    logger.info(f"\n📦 Creating {zip_file}...")
    
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file in output_dir.glob("*.json"):
            zf.write(file, arcname=file.name)
    
    logger.info(f"✓ ZIP created: {zip_file}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main pipeline - 100% matching Milvus documentation pattern"""
    pipeline_start = time.time()
    initial_memory = check_memory()
    
    logger.info(f"\n{'='*70}")
    logger.info("HYBRID SEARCH - 100% Milvus Documentation Format")
    logger.info(f"{'='*70}")
    logger.info(f"💾 Initial memory: {initial_memory['used_gb']:.1f}GB / {initial_memory['total_gb']:.1f}GB\n")
    
    # ========================================================================
    # 1. LOAD QUERIES
    # ========================================================================
    logger.info("📂 Loading queries...")
    with open(QUERIES_FILE, 'r', encoding='utf-8') as f:
        queries_json = json.load(f)
    
    queries = [
        (item.get("query_num"), item.get("query"))
        for item in queries_json
        if item.get("query_num") and item.get("query")
    ]
    
    total_queries = len(queries)
    logger.info(f"✓ Loaded {total_queries} queries\n")
    
    # ========================================================================
    # 2. LOAD MODELS
    # ========================================================================
    embedding_model = load_embedding_model(EMBEDDING_MODEL_NAME)
    
    if USE_CROSS_ENCODER:
        cross_encoder = load_cross_encoder(CROSS_ENCODER_MODEL)
    else:
        cross_encoder = None
    
    # ========================================================================
    # 3. CONNECT TO MILVUS (same as docs)
    # ========================================================================
    connect_to_milvus(uri=MILVUS_URI, token=TOKEN)
    col = get_collection(COLLECTION_NAME)
    
    # ========================================================================
    # 4. PROCESS QUERIES IN BATCHES
    # ========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"PROCESSING {total_queries} QUERIES")
    logger.info(f"Batch size: {PROCESS_BATCH_SIZE}")
    logger.info(f"Workers: {MAX_WORKERS}")
    logger.info(f"{'='*70}\n")
    
    output_dir = Path(OUTPUT_DIR)
    all_results = []
    csv_data = []
    queries_processed = 0
    
    for batch_start in range(0, len(queries), PROCESS_BATCH_SIZE):
        batch_end = min(batch_start + PROCESS_BATCH_SIZE, len(queries))
        batch_queries = queries[batch_start:batch_end]
        
        batch_start_time = time.time()
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH: {batch_start+1}-{batch_end} ({len(batch_queries)} queries)")
        logger.info(f"{'='*70}")
        
        # Generate embeddings for batch
        query_nums = [q[0] for q in batch_queries]
        query_texts = [q[1] for q in batch_queries]
        
        logger.info("Generating embeddings...")
        dense_embeddings = generate_embeddings(embedding_model, query_texts, EMBEDDING_BATCH_SIZE)
        sparse_embeddings = [generate_sparse_vector(text) for text in query_texts]
        
        # Prepare data for parallel processing
        queries_data = [
            (qnum, qtext, dense, sparse)
            for qnum, qtext, dense, sparse in zip(
                query_nums, query_texts, dense_embeddings, sparse_embeddings
            )
        ]
        
        # Search in parallel
        logger.info("Searching...")
        search_start = time.time()
        batch_results = process_queries_parallel(col, queries_data, MAX_WORKERS)
        search_time = time.time() - search_start
        logger.info(f"  Search: {len(batch_queries)} queries in {search_time:.2f}s")
        
        # Rerank
        if USE_CROSS_ENCODER:
            logger.info("Reranking...")
            batch_results = finalize_results_with_reranking(cross_encoder, batch_results, csv_data)
        else:
            batch_results = finalize_results_without_reranking(batch_results, csv_data)
        
        # Write results
        write_results_to_json(batch_results, output_dir)
        
        all_results.extend(batch_results)
        queries_processed += len(batch_queries)
        
        # Statistics
        batch_time = time.time() - batch_start_time
        elapsed = time.time() - pipeline_start
        progress = queries_processed / total_queries * 100
        overall_speed = queries_processed / elapsed
        
        logger.info(f"\n📊 Batch time: {batch_time:.2f}s")
        logger.info(f"📊 Progress: {queries_processed}/{total_queries} ({progress:.1f}%)")
        logger.info(f"⚡ Overall speed: {overall_speed:.2f} q/s")
        
        # Memory check
        if queries_processed % MEMORY_CHECK_INTERVAL == 0:
            mem = check_memory()
            logger.info(f"💾 Memory: {mem['percent']:.1f}% ({mem['used_gb']:.1f}GB used)")
            
            if mem['percent'] > MAX_MEMORY_PERCENT:
                logger.warning("⚠️  High memory! Running cleanup...")
                aggressive_cleanup()
        
        # Cleanup
        aggressive_cleanup()
        del dense_embeddings, sparse_embeddings, queries_data, batch_results
    
    # ========================================================================
    # 5. WRITE CSV AND CREATE ZIP
    # ========================================================================
    write_csv_results(csv_data, CSV_OUTPUT)
    create_submission_zip(output_dir, ZIP_FILENAME)
    
    # ========================================================================
    # 6. FINAL STATISTICS
    # ========================================================================
    total_time = time.time() - pipeline_start
    success_count = sum(1 for r in all_results if r["success"])
    final_mem = check_memory()
    mem_growth = final_mem['used_gb'] - initial_memory['used_gb']
    
    logger.info(f"\n{'='*70}")
    logger.info("✅ PIPELINE COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Queries: {total_queries} | Success: {success_count}")
    logger.info(f"Time: {total_time/60:.2f} minutes")
    logger.info(f"Speed: {total_queries/total_time:.2f} q/s")
    logger.info(f"Memory growth: {mem_growth:.1f}GB")
    logger.info(f"Output: {ZIP_FILENAME}")
    logger.info(f"CSV: {CSV_OUTPUT} ({len(csv_data)} rows)")
    logger.info(f"{'='*70}")
    
    # ========================================================================
    # 7. CLEANUP
    # ========================================================================
    try:
        connections.disconnect("default")
        logger.info("✓ Disconnected from Milvus")
    except:
        pass
    aggressive_cleanup()


if __name__ == "__main__":
    main()
