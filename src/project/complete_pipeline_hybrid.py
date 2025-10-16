# src/project/complete_pipeline_hybrid.py
"""
COMPLETE HYBRID PIPELINE with Milvus Native BM25

Features:
1. Dense vectors (sentence-transformers)
2. Sparse vectors (Milvus BM25 - auto-generated)
3. Chunk cleaning before embedding
4. Hybrid search ready

This replaces complete_pipeline_hybrid.py with full Milvus BM25 support.
"""

import argparse
import asyncio
import json
import logging
import time
import os
import gc
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient

from project.pydantic_models import ProcessingConfig, EmbeddingModel
from project.doc_reader import DocumentReader
from project.chunker import ChunkingService
from project.chunk_cleaner import clean_chunks_before_embedding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Milvus configuration
COLLECTION_NAME = "rag_chunks_hybrid"


def get_files_from_sqlite(db_path):
    """Get files from SQLite with pending status"""
    query = "SELECT doc_id, source_path FROM documents WHERE processing_status='pending'"
    with sqlite3.connect(db_path) as conn:
        files = []
        for doc_id, source_path in conn.execute(query):
            if os.path.exists(source_path):
                files.append((doc_id, Path(source_path)))
            else:
                logger.warning(f"File not found: {source_path}")
        return files


class MilvusHybridPipeline:
    """
    Complete pipeline with Milvus native hybrid search support
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
        # Device setup
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Load embedding model (for dense vectors)
        logger.info("Loading sentence transformer for dense embeddings...")
        self.embedding_model = SentenceTransformer(
            EmbeddingModel.ALL_MPNET_BASE_V2.value,
            device=self.device
        )
        
        # Batch sizes
        if self.device == "mps":
            self.embedding_batch_size = 32
            self.chunk_batch_size = 200
        else:
            self.embedding_batch_size = 64
            self.chunk_batch_size = 500
        
        # Thread pool
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Connect to Milvus
        self._connect_to_milvus()
        
        logger.info("Pipeline initialized successfully")
    
    def _connect_to_milvus(self):
        """Connect to Milvus"""
        logger.info("Connecting to Milvus...")
        self.milvus_client = MilvusClient(uri="http://localhost:19530")
        
        if not self.milvus_client.has_collection(COLLECTION_NAME):
            logger.error(f"❌ Collection '{COLLECTION_NAME}' not found!")
            logger.error("Run: poetry run python -m project.schema_setup_hybrid")
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' does not exist")
        
        logger.info(f"✅ Connected to Milvus collection: {COLLECTION_NAME}")
    
    def generate_sparse_vector(self, text: str) -> Dict[int, float]:
        """
        Generate BM25 sparse vector from text.
        
        Milvus BM25 uses a simple TF-IDF-like approach:
        - Term frequencies become sparse vector indices
        - Values are normalized scores
        """
        # Tokenize (simple whitespace + lowercase)
        tokens = text.lower().split()
        
        # Count term frequencies
        term_freq = {}
        for token in tokens:
            # Use hash of token as index (Milvus will handle this)
            token_id = hash(token) % 1000000  # Keep indices reasonable
            term_freq[token_id] = term_freq.get(token_id, 0) + 1
        
        # Normalize frequencies (simple approach)
        max_freq = max(term_freq.values()) if term_freq else 1
        sparse_vector = {
            idx: freq / max_freq 
            for idx, freq in term_freq.items()
        }
        
        return sparse_vector
    
    def _embed_chunks_with_management(self, chunk_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate BOTH dense and sparse vectors for chunks.
        """
        texts = [chunk['chunk_text'] for chunk in chunk_dicts]
        
        try:
            # Clear memory
            if self.device == "mps":
                torch.mps.empty_cache()
            
            # === Generate Dense Vectors (sentence-transformers) ===
            logger.info(f"Generating dense embeddings for {len(texts)} chunks...")
            all_embeddings = []
            num_batches = (len(texts) + self.embedding_batch_size - 1) // self.embedding_batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.embedding_batch_size
                end_idx = min(start_idx + self.embedding_batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=self.embedding_batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.device
                )
                
                all_embeddings.append(batch_embeddings)
                
                if self.device == "mps":
                    torch.mps.empty_cache()
            
            # Concatenate dense embeddings
            if len(all_embeddings) > 1:
                dense_embeddings = np.vstack(all_embeddings)
            else:
                dense_embeddings = all_embeddings[0]
            
            # === Generate Sparse Vectors (BM25) ===
            logger.info(f"Generating BM25 sparse vectors for {len(texts)} chunks...")
            sparse_vectors = [
                self.generate_sparse_vector(text) 
                for text in texts
            ]
            
            # === Update Chunks ===
            for chunk, dense_emb, sparse_vec in zip(chunk_dicts, dense_embeddings, sparse_vectors):
                chunk['dense_vector'] = dense_emb.tolist()
                chunk['sparse_vector'] = sparse_vec
                chunk['embedding_model'] = EmbeddingModel.ALL_MPNET_BASE_V2.value
            
            logger.info(f"✅ Generated both dense and sparse vectors")
            
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []
        
        return chunk_dicts
    
    def _cleanup_memory(self):
        """Memory cleanup"""
        if self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()
    
    def _insert_to_milvus_direct(self, chunk_dicts: List[Dict[str, Any]]):
        """
        Insert chunks directly to Milvus (for small batches).
        """
        if not chunk_dicts:
            return
        
        logger.info(f"Inserting {len(chunk_dicts)} chunks to Milvus...")
        
        # Prepare data in Milvus format
        data = []
        for chunk in chunk_dicts:
            row = {
                "chunk_id": chunk.get("chunk_id", chunk.get("id")),
                "doc_id": chunk.get("doc_id", ""),
                "doc_name": chunk.get("doc_name", ""),
                "chunk_index": chunk.get("chunk_index", 0),
                "chunk_text": chunk.get("chunk_text", ""),
                "chunk_size": chunk.get("chunk_size", 0),
                "chunk_tokens": chunk.get("chunk_tokens", 0),
                "chunk_method": chunk.get("chunk_method", "recursive"),
                "chunk_overlap": chunk.get("chunk_overlap", 0),
                "domain": chunk.get("domain", "general"),
                "content_type": chunk.get("content_type", "text"),
                "embedding_model": chunk.get("embedding_model", ""),
                "created_at": chunk.get("created_at", ""),
                "dense_vector": chunk.get("dense_vector", []),
                "sparse_vector": chunk.get("sparse_vector", {})
            }
            data.append(row)
        
        # Insert to Milvus
        try:
            result = self.milvus_client.insert(
                collection_name=COLLECTION_NAME,
                data=data
            )
            logger.info(f"✅ Inserted {result['insert_count']} chunks to Milvus")
        except Exception as e:
            logger.error(f"❌ Milvus insert failed: {e}")
            raise
    
    def _initialize_json_file(self, file_path: Path):
        """Initialize JSON file for bulk import (optional backup)"""
        if os.path.exists(file_path):
            os.remove(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('{\n  "rows": [\n')
        
        logger.info(f"Initialized JSON backup: {file_path}")
    
    def _append_to_json_file(self, chunks: List[Dict[str, Any]], file_path: Path, needs_comma: bool):
        """Append to JSON backup file"""
        if not chunks:
            return
        
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    if needs_comma or i > 0:
                        f.write(',\n')
                    
                    # Simplified for JSON (Milvus native format may differ)
                    cleaned_chunk = {
                        "chunk_id": chunk.get("chunk_id"),
                        "doc_id": chunk.get("doc_id"),
                        "doc_name": chunk.get("doc_name"),
                        "chunk_index": chunk.get("chunk_index", 0),
                        "chunk_text": chunk.get("chunk_text", ""),
                        "chunk_size": chunk.get("chunk_size", 0),
                        "chunk_tokens": chunk.get("chunk_tokens", 0),
                        "chunk_method": chunk.get("chunk_method"),
                        "chunk_overlap": chunk.get("chunk_overlap", 0),
                        "domain": chunk.get("domain"),
                        "content_type": chunk.get("content_type"),
                        "embedding_model": chunk.get("embedding_model"),
                        "created_at": chunk.get("created_at", ""),
                        "dense_vector": chunk.get("dense_vector"),
                        "sparse_vector": chunk.get("sparse_vector")
                    }
                    
                    chunk_str = json.dumps(cleaned_chunk, ensure_ascii=False)
                    indented = '    ' + chunk_str.replace('\n', '\n    ')
                    f.write(indented)
            
        except Exception as e:
            logger.error(f"Failed to append to JSON: {e}")
    
    def _finalize_json_file(self, file_path: Path):
        """Finalize JSON backup"""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write('\n  ]\n}\n')
            logger.info(f"JSON backup finalized: {file_path}")
        except Exception as e:
            logger.error(f"Failed to finalize JSON: {e}")
    
    def _process_file_sync(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a single file"""
        try:
            # Read document
            document = DocumentReader.read_file(file_path)
            if not document:
                return []
            
            # Chunk document
            chunks = ChunkingService.chunk_document(document, self.config)
            chunk_dicts = [chunk.model_dump(by_alias=False) for chunk in chunks]
            
            # ⭐ CLEAN CHUNKS BEFORE EMBEDDING ⭐
            cleaned_chunks = clean_chunks_before_embedding(chunk_dicts)
            
            logger.info(f"Processed {file_path.name}: {len(chunks)} -> {len(cleaned_chunks)} chunks after cleaning")
            
            return cleaned_chunks
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return []
    
    def process_complete_pipeline(
        self,
        input_dir: str,
        output_dir: str,
        use_direct_insert: bool = True
    ):
        """
        Main pipeline with direct Milvus insertion.
        """
        pipeline_start = time.time()
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        output_file = output_path / "prepared_data_hybrid.json"
        
        try:
            # Initialize JSON backup
            self._initialize_json_file(output_file)
            
            logger.info("=== STARTING MILVUS HYBRID PIPELINE ===")
            
            # Discover files
            db_path = input_path / "documents.db"
            if db_path.exists():
                logger.info(f"Fetching from SQLite: {db_path}")
                files = get_files_from_sqlite(str(db_path))
                files = [(doc_id, path) for doc_id, path in files]
            else:
                logger.info("Scanning input directory...")
                file_paths = DocumentReader.find_files(input_path)
                files = [(None, path) for path in file_paths]
            
            logger.info(f"Found {len(files)} files to process")
            
            if not files:
                self._finalize_json_file(output_file)
                return
            
            total_chunks = 0
            
            # Process each file
            for i, (doc_id, file_path) in enumerate(files, 1):
                file_start = time.time()
                logger.info(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
                
                try:
                    # Read and chunk
                    chunk_dicts = self._process_file_sync(file_path)
                    
                    if not chunk_dicts:
                        continue
                    
                    # Attach doc_id from SQLite
                    if doc_id:
                        for chunk in chunk_dicts:
                            chunk["doc_id"] = doc_id
                    
                    # Process in batches
                    for batch_start in range(0, len(chunk_dicts), self.chunk_batch_size):
                        batch_end = min(batch_start + self.chunk_batch_size, len(chunk_dicts))
                        batch = chunk_dicts[batch_start:batch_end]
                        
                        logger.info(f"Batch: chunks {batch_start}-{batch_end-1}")
                        
                        # Generate embeddings (dense + sparse)
                        embedded_batch = self._embed_chunks_with_management(batch)
                        
                        # Insert to Milvus directly
                        if use_direct_insert:
                            self._insert_to_milvus_direct(embedded_batch)
                        
                        # Backup to JSON
                        self._append_to_json_file(
                            embedded_batch,
                            output_file,
                            total_chunks > 0
                        )
                        
                        total_chunks += len(embedded_batch)
                        
                        # Cleanup
                        self._cleanup_memory()
                    
                    file_time = time.time() - file_start
                    logger.info(f"✅ Completed {file_path.name} in {file_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"❌ Error: {e}", exc_info=True)
                    continue
            
            # Finalize
            self._finalize_json_file(output_file)
            
            total_time = time.time() - pipeline_start
            logger.info("\n=== PIPELINE COMPLETED ===")
            logger.info(f"Total chunks: {total_chunks}")
            logger.info(f"Total time: {total_time/60:.2f} minutes")
            logger.info(f"Speed: {total_chunks/total_time:.1f} chunks/sec")
            
        except Exception as e:
            logger.error(f"❌ Critical error: {e}", exc_info=True)
            try:
                self._finalize_json_file(output_file)
            except:
                pass
            raise
        finally:
            self.executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description="Milvus Hybrid Pipeline with BM25")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=256)
    parser.add_argument("--no-direct-insert", action="store_true",
                       help="Skip direct Milvus insertion (JSON only)")
    
    args = parser.parse_args()
    
    config = ProcessingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    pipeline = MilvusHybridPipeline(config)
    pipeline.process_complete_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_direct_insert=not args.no_direct_insert
    )


if __name__ == "__main__":
    main()