# src/project/process_queries_milvus_hybrid.py
"""
Query processing with Milvus NATIVE hybrid search:
1. Dense vector search (HNSW on dense_vector)
2. Sparse vector search (BM25 on sparse_vector) 
3. Hybrid search with RRF (Reciprocal Rank Fusion)
4. Cross-encoder reranking (optional)
"""

import json
import zipfile
import logging
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import time
from pymilvus import MilvusClient, Collection, connections
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

# Configuration
QUERIES_FILE = 'queries.json'
OUTPUT_DIR = 'submission_hybrid_milvus'
ZIP_FILENAME = 'PS04_MILVUS_HYBRID.zip'
CSV_OUTPUT = 'milvus_hybrid_results.csv'

MILVUS_URI = "http://localhost:19530"
COLLECTION_NAME = "rag_chunks_hybrid"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Search parameters
DENSE_TOP_K = 20
SPARSE_TOP_K = 20
FINAL_TOP_K = 5
USE_CROSS_ENCODER = True  # Set to False for faster queries

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MilvusHybridQueryProcessor:
    """
    Query processor using Milvus native hybrid search
    """
    
    def __init__(self):
        logger.info("Initializing Milvus hybrid query processor...")
        
        # Load embedding model for dense vectors
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu"
        )
        
        # Load cross-encoder for reranking (optional)
        if USE_CROSS_ENCODER:
            logger.info("Loading cross-encoder for reranking...")
            self.cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
        else:
            self.cross_encoder = None
        
        # Connect to Milvus
        self._connect_to_milvus()
        
        # Storage for CSV
        self.csv_data = []
        
        logger.info("Initialization complete")
    
    def _connect_to_milvus(self):
        """Connect to Milvus and verify hybrid collection"""
        logger.info(f"Connecting to Milvus at {MILVUS_URI}")
        
        self.client = MilvusClient(uri=MILVUS_URI)
        connections.connect("default", uri=MILVUS_URI)
        
        if not self.client.has_collection(COLLECTION_NAME):
            available = self.client.list_collections()
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found! "
                f"Available: {available}\n"
                f"Run: poetry run python -m project.schema_setup_hybrid"
            )
        
        # Verify schema
        schema_info = self.client.describe_collection(COLLECTION_NAME)
        
        has_dense = False
        has_sparse = False
        
        logger.info("Collection schema:")
        for field in schema_info['fields']:
            field_name = field['name']
            field_type = field['type']
            logger.info(f"  - {field_name}: {field_type}")
            
            if field_name == "dense_vector":
                has_dense = True
            if field_name == "sparse_vector":
                has_sparse = True
        
        if not (has_dense and has_sparse):
            raise RuntimeError(
                "Collection missing hybrid vectors!\n"
                "Expected: dense_vector, sparse_vector"
            )
        
        # Load collection
        collection = Collection(COLLECTION_NAME)
        collection.load()
        time.sleep(2)
        
        stats = self.client.get_collection_stats(COLLECTION_NAME)
        row_count = stats.get('row_count', 0)
        
        logger.info(f"✅ Connected to hybrid collection: {row_count:,} chunks")
    
    def generate_sparse_vector(self, text: str) -> Dict[int, float]:
        """Generate BM25 sparse vector (same as pipeline)"""
        tokens = text.lower().split()
        
        term_freq = {}
        for token in tokens:
            token_id = hash(token) % 1000000
            term_freq[token_id] = term_freq.get(token_id, 0) + 1
        
        max_freq = max(term_freq.values()) if term_freq else 1
        sparse_vector = {
            idx: freq / max_freq
            for idx, freq in term_freq.items()
        }
        
        return sparse_vector
    
    def embed_query(self, query_text: str) -> Tuple[List[float], Dict[int, float]]:
        """
        Generate both dense and sparse vectors for query.
        
        Returns:
            (dense_vector, sparse_vector)
        """
        # Dense vector
        dense = self.embedding_model.encode(
            query_text,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()
        
        # Sparse vector
        sparse = self.generate_sparse_vector(query_text)
        
        return dense, sparse
    
    def hybrid_search(
        self,
        query_text: str,
        dense_vector: List[float],
        sparse_vector: Dict[int, float]
    ) -> List[Dict]:
        """
        Perform Milvus native hybrid search with RRF fusion.
        
        RRF (Reciprocal Rank Fusion):
        score = 1/(k + rank_dense) + 1/(k + rank_sparse)
        """
        # Search parameters
        search_params_dense = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }
        
        search_params_sparse = {
            "metric_type": "BM25",
            "params": {}
        }
        
        # === Dense Vector Search ===
        dense_results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[dense_vector],
            anns_field="dense_vector",
            limit=DENSE_TOP_K,
            search_params=search_params_dense,
            output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"]
        )
        
        # === Sparse Vector Search (BM25) ===
        sparse_results = self.client.search(
            collection_name=COLLECTION_NAME,
            data=[sparse_vector],
            anns_field="sparse_vector",
            limit=SPARSE_TOP_K,
            search_params=search_params_sparse,
            output_fields=["chunk_id", "doc_id", "doc_name", "chunk_text"]
        )
        
        # === RRF Fusion ===
        # Reciprocal Rank Fusion combines rankings from both searches
        rrf_k = 60  # Standard RRF parameter
        
        # Build rank maps
        dense_ranks = {}
        sparse_ranks = {}
        all_chunks = {}
        
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
            
            rrf_score = (1.0 / (rrf_k + dense_rank)) + (1.0 / (rrf_k + sparse_rank))
            
            chunk_data['rrf_score'] = rrf_score
            chunk_data['dense_rank'] = dense_rank
            chunk_data['sparse_rank'] = sparse_rank
            
            # Fill missing scores
            chunk_data.setdefault('dense_score', 0.0)
            chunk_data.setdefault('sparse_score', 0.0)
        
        # Sort by RRF score
        fused_results = sorted(
            all_chunks.values(),
            key=lambda x: x['rrf_score'],
            reverse=True
        )
        
        logger.info(f"Hybrid search: {len(dense_results[0])} dense + {len(sparse_results[0])} sparse = {len(fused_results)} fused")
        
        return fused_results
    
    def cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Dict]
    ) -> List[Dict]:
        """Rerank with cross-encoder"""
        if not self.cross_encoder or not candidates:
            return candidates
        
        pairs = [[query, c['chunk_text']] for c in candidates]
        ce_scores = self.cross_encoder.predict(pairs)
        
        for candidate, score in zip(candidates, ce_scores):
            candidate['cross_encoder_score'] = float(score)
        
        reranked = sorted(
            candidates,
            key=lambda x: x['cross_encoder_score'],
            reverse=True
        )
        
        logger.info(f"Cross-encoder reranked {len(candidates)} candidates")
        return reranked
    
    def process_single_query(self, query_data: Tuple[str, str]) -> Dict:
        """Process a single query with hybrid search"""
        query_num, query_text = query_data
        
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Query {query_num}: {query_text}")
            
            # Generate embeddings
            dense_vec, sparse_vec = self.embed_query(query_text)
            
            # Hybrid search
            fused_results = self.hybrid_search(query_text, dense_vec, sparse_vec)
            
            # Take top candidates for reranking
            candidates = fused_results[:25]
            
            # Cross-encoder reranking
            if USE_CROSS_ENCODER:
                final_results = self.cross_encoder_rerank(query_text, candidates)[:FINAL_TOP_K]
            else:
                final_results = candidates[:FINAL_TOP_K]
            
            # Extract doc_ids for submission
            doc_ids = []
            seen = set()
            for result in final_results:
                doc_name = result['doc_name']
                if doc_name and doc_name not in seen:
                    doc_ids.append(doc_name)
                    seen.add(doc_name)
            
            # Log results
            logger.info(f"Top-{FINAL_TOP_K} Results:")
            for rank, result in enumerate(final_results, 1):
                logger.info(f"\n[Rank {rank}] {result['doc_name']}")
                logger.info(f"  RRF: {result['rrf_score']:.4f} | Dense: {result['dense_score']:.4f} | Sparse: {result['sparse_score']:.4f}")
                if USE_CROSS_ENCODER:
                    logger.info(f"  Cross-Encoder: {result.get('cross_encoder_score', 0):.4f}")
                logger.info(f"  Text: {result['chunk_text'][:150]}...")
            
            # Store for CSV
            for rank, result in enumerate(final_results, 1):
                self.csv_data.append({
                    'query_num': query_num,
                    'query_text': query_text,
                    'rank': rank,
                    'chunk_id': result['chunk_id'],
                    'doc_id': result['doc_id'],
                    'doc_name': result['doc_name'],
                    'rrf_score': result['rrf_score'],
                    'dense_score': result['dense_score'],
                    'sparse_score': result['sparse_score'],
                    'dense_rank': result['dense_rank'],
                    'sparse_rank': result['sparse_rank'],
                    'cross_encoder_score': result.get('cross_encoder_score', 0),
                    'chunk_text': result['chunk_text'][:500]
                })
            
            return {
                "query_num": query_num,
                "query_text": query_text,
                "doc_ids": doc_ids,
                "detailed_results": final_results,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Query {query_num} failed: {e}", exc_info=True)
            return {
                "query_num": query_num,
                "query_text": query_text,
                "doc_ids": [],
                "detailed_results": [],
                "success": False,
                "error": str(e)
            }
    
    def write_result_file(self, result: Dict, output_path: Path) -> bool:
        """Write result to JSON file"""
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
            logger.error(f"Failed to write query {result['query_num']}: {e}")
            return False
    
    def write_csv_results(self, csv_path: str):
        """Write detailed results to CSV"""
        logger.info(f"Writing CSV results to {csv_path}...")
        
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'query_num', 'query_text', 'rank', 'chunk_id', 'doc_id', 'doc_name',
                    'rrf_score', 'dense_score', 'sparse_score', 'dense_rank', 'sparse_rank',
                    'cross_encoder_score', 'chunk_text'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.csv_data)
            
            logger.info(f"✅ CSV written: {len(self.csv_data)} rows")
        except Exception as e:
            logger.error(f"Failed to write CSV: {e}")
    
    def process_all_queries(self):
        """Process all queries with Milvus hybrid search"""
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
        
        logger.info(f"Processing {len(queries)} queries with Milvus hybrid search")
        
        # Create output directory
        output_path = Path(OUTPUT_DIR)
        output_path.mkdir(exist_ok=True)
        
        # Process queries
        all_results = []
        for i, query_data in enumerate(queries, 1):
            logger.info(f"\n{'='*70}")
            logger.info(f"QUERY {i}/{len(queries)}")
            logger.info(f"{'='*70}")
            
            result = self.process_single_query(query_data)
            all_results.append(result)
            
            # Write result file
            self.write_result_file(result, output_path)
            
            # Progress
            elapsed = time.time() - start_time
            speed = i / elapsed if elapsed > 0 else 0
            remaining = len(queries) - i
            eta = remaining / speed if speed > 0 else 0
            
            logger.info(f"\nProgress: {i}/{len(queries)} ({100*i/len(queries):.1f}%)")
            logger.info(f"Speed: {speed:.2f} queries/sec | ETA: {eta/60:.1f} minutes")
        
        # Write CSV
        self.write_csv_results(CSV_OUTPUT)
        
        # Create submission zip
        logger.info(f"\nCreating submission zip: {ZIP_FILENAME}")
        json_files = list(output_path.glob("*.json"))
        
        with zipfile.ZipFile(ZIP_FILENAME, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in json_files:
                zf.write(file_path, arcname=file_path.name)
        
        # Final statistics
        total_time = time.time() - start_time
        success_count = sum(1 for r in all_results if r["success"])
        
        logger.info(f"\n{'='*70}")
        logger.info("MILVUS HYBRID SEARCH COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Total queries: {len(queries)}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {len(queries) - success_count}")
        logger.info(f"Total time: {total_time/60:.2f} minutes")
        logger.info(f"Average speed: {len(queries)/total_time:.2f} queries/sec")
        logger.info(f"Output: {ZIP_FILENAME}")
        logger.info(f"CSV: {CSV_OUTPUT}")
        logger.info(f"{'='*70}")


def main():
    """Main entry point"""
    try:
        processor = MilvusHybridQueryProcessor()
        processor.process_all_queries()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()