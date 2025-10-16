# src/project/hybrid_reranker.py
"""
Hybrid Retrieval + Reranking Strategy:
1. BM25 sparse retrieval (top-k candidates)
2. Dense vector search (cosine similarity)
3. Score fusion (BM25 + Cosine)
4. Cross-encoder reranking (final top-k)
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridReranker:
    """
    Implements BM25 + Vector Search + Cross-Encoder Reranking
    """
    
    def __init__(
        self,
        db_path: str = "documents.db",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize the hybrid reranker.
        
        Args:
            db_path: Path to SQLite database with chunks table
            cross_encoder_model: HuggingFace cross-encoder model name
        """
        self.db_path = db_path
        
        # Load cross-encoder for reranking
        logger.info(f"Loading cross-encoder: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        # Load BM25 index from database
        self._load_bm25_index()
        
        logger.info("HybridReranker initialized successfully")
    
    def _load_bm25_index(self):
        """Load all chunks from database and build BM25 index"""
        logger.info("Building BM25 index from database...")
        
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Fetch all chunks
        cur.execute("""
            SELECT chunk_id, doc_id, doc_name, chunk_text 
            FROM chunks
            ORDER BY chunk_id
        """)
        
        self.chunks_data = []
        corpus = []
        
        for row in cur.fetchall():
            chunk_id, doc_id, doc_name, chunk_text = row
            self.chunks_data.append({
                'chunk_id': chunk_id,
                'doc_id': doc_id,
                'doc_name': doc_name,
                'chunk_text': chunk_text
            })
            # Tokenize for BM25
            corpus.append(chunk_text.lower().split())
        
        conn.close()
        
        # Build BM25 index
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"BM25 index built with {len(self.chunks_data)} chunks")
    
    def bm25_search(self, query: str, top_k: int = 25) -> List[Dict[str, Any]]:
        """
        Perform BM25 sparse retrieval.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of dicts with chunk data and BM25 scores
        """
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # Only include non-zero scores
                result = self.chunks_data[idx].copy()
                result['bm25_score'] = float(bm25_scores[idx])
                results.append(result)
        
        logger.info(f"BM25 retrieved {len(results)} results")
        return results
    
    def fuse_scores(
        self,
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        alpha: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Fuse BM25 and vector search scores.
        
        Args:
            bm25_results: Results from BM25 search
            vector_results: Results from vector search
            alpha: Weight for BM25 (1-alpha for vector). 0.5 = equal weight
            
        Returns:
            Fused results with combined scores
        """
        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(r['bm25_score'] for r in bm25_results)
            min_bm25 = min(r['bm25_score'] for r in bm25_results)
            bm25_range = max_bm25 - min_bm25 if max_bm25 != min_bm25 else 1.0
            
            for r in bm25_results:
                r['bm25_score_norm'] = (r['bm25_score'] - min_bm25) / bm25_range
        
        # Create lookup for vector scores by chunk_id
        vector_lookup = {
            r['chunk_id']: r['cosine_similarity'] 
            for r in vector_results
        }
        
        # Combine scores
        fused_results = {}
        
        # Add BM25 results
        for r in bm25_results:
            chunk_id = r['chunk_id']
            bm25_norm = r.get('bm25_score_norm', 0)
            vector_score = vector_lookup.get(chunk_id, 0)
            
            fused_score = alpha * bm25_norm + (1 - alpha) * vector_score
            
            fused_results[chunk_id] = {
                'chunk_id': chunk_id,
                'doc_id': r['doc_id'],
                'doc_name': r['doc_name'],
                'chunk_text': r['chunk_text'],
                'bm25_score': r['bm25_score'],
                'bm25_score_norm': bm25_norm,
                'cosine_similarity': vector_score,
                'fused_score': fused_score
            }
        
        # Add vector-only results (not in BM25)
        for r in vector_results:
            chunk_id = r['chunk_id']
            if chunk_id not in fused_results:
                vector_score = r['cosine_similarity']
                fused_score = (1 - alpha) * vector_score
                
                fused_results[chunk_id] = {
                    'chunk_id': chunk_id,
                    'doc_id': r['doc_id'],
                    'doc_name': r['doc_name'],
                    'chunk_text': r['chunk_text'],
                    'bm25_score': 0,
                    'bm25_score_norm': 0,
                    'cosine_similarity': vector_score,
                    'fused_score': fused_score
                }
        
        # Sort by fused score
        fused_list = sorted(
            fused_results.values(),
            key=lambda x: x['fused_score'],
            reverse=True
        )
        
        logger.info(f"Fused {len(fused_list)} unique results")
        return fused_list
    
    def cross_encoder_rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Original query
            candidates: List of candidate chunks
            top_k: Number of final results
            
        Returns:
            Reranked results with cross-encoder scores
        """
        if not candidates:
            return []
        
        # Prepare query-passage pairs
        pairs = [[query, candidate['chunk_text']] for candidate in candidates]
        
        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Add scores to candidates
        for candidate, score in zip(candidates, ce_scores):
            candidate['cross_encoder_score'] = float(score)
        
        # Sort by cross-encoder score
        reranked = sorted(
            candidates,
            key=lambda x: x['cross_encoder_score'],
            reverse=True
        )[:top_k]
        
        logger.info(f"Cross-encoder reranked to top-{top_k}")
        return reranked
    
    def hybrid_search(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        bm25_top_k: int = 25,
        fusion_alpha: float = 0.5,
        final_top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Complete hybrid search pipeline.
        
        Args:
            query: Search query
            vector_results: Results from Milvus vector search
            bm25_top_k: Number of BM25 candidates
            fusion_alpha: Weight for BM25 in fusion (0.5 = equal)
            final_top_k: Final number of results after reranking
            
        Returns:
            Final reranked results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting hybrid search for query: '{query}'")
        
        # Step 1: BM25 retrieval
        bm25_results = self.bm25_search(query, top_k=bm25_top_k)
        
        # Step 2: Score fusion
        fused_results = self.fuse_scores(
            bm25_results, 
            vector_results,
            alpha=fusion_alpha
        )
        
        # Take top candidates for reranking
        candidates = fused_results[:min(25, len(fused_results))]
        
        # Step 3: Cross-encoder reranking
        final_results = self.cross_encoder_rerank(
            query,
            candidates,
            top_k=final_top_k
        )
        
        # Log final results
        logger.info(f"\nFinal Top-{final_top_k} Results:")
        for rank, result in enumerate(final_results, 1):
            logger.info(f"\n[Rank {rank}]")
            logger.info(f"  Doc: {result['doc_name']}")
            logger.info(f"  Cross-Encoder: {result['cross_encoder_score']:.4f}")
            logger.info(f"  Fused Score: {result['fused_score']:.4f}")
            logger.info(f"  BM25: {result['bm25_score']:.4f} | Cosine: {result['cosine_similarity']:.4f}")
            logger.info(f"  Text: {result['chunk_text'][:200]}...")
        
        logger.info(f"{'='*60}\n")
        return final_results


# Standalone test function
if __name__ == "__main__":
    # Example usage
    reranker = HybridReranker(
        db_path="documents.db",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    
    # Simulate vector search results
    mock_vector_results = [
        {
            'chunk_id': 'doc1_chunk_0',
            'doc_id': 'doc1',
            'doc_name': 'example.txt',
            'chunk_text': 'This is about machine learning and AI.',
            'cosine_similarity': 0.85
        },
        {
            'chunk_id': 'doc1_chunk_1',
            'doc_id': 'doc1',
            'doc_name': 'example.txt',
            'chunk_text': 'Deep learning models require large datasets.',
            'cosine_similarity': 0.78
        }
    ]
    
    results = reranker.hybrid_search(
        query="What is machine learning?",
        vector_results=mock_vector_results,
        bm25_top_k=25,
        fusion_alpha=0.5,
        final_top_k=5
    )
    
    print("\nHybrid search complete!")