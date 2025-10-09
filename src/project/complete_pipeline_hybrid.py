# src/project/complete_pipeline_final.py
# ------------------ INPUT FILE LOGIC HANDLING ------------------
# This pipeline accepts input either via:
# - An SQLite "documents.db" file in the input directory or 
# db with different name(update the db name in below code line number:240 )
# - OR if not found, defaults to scanning the input directory for files (legacy/manual workflows)
#
# For SQLite: The "documents" table must contain 'doc_id', 'source_path', and 'processing_status' fields.
# Only rows with processing_status='pending' will be included.
# The input_dir must contain the documents.db SQLite database if you want to use the DB-driven input.
# --------------------------------------------------------------


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

from project.pydantic_models import ProcessingConfig, EmbeddingModel
from project.doc_reader import DocumentReader
from project.chunker import OptimizedChunkingService,ChunkingService
from project.milvus_bulk_import import EnhancedMilvusBulkImporter
from project.populate_sqlite_from_json import populate_db_from_json
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


#new function added for sqlite
def get_files_from_sqlite(db_path):
    """
    Get files from SQLite documents table with pending status,
    returns a list of (doc_id, Path) tuples.
    """
    query = "SELECT doc_id, source_path FROM documents WHERE processing_status='pending'"
    with sqlite3.connect(db_path) as conn:
        files = []
        for doc_id, source_path in conn.execute(query):
            if os.path.exists(source_path):
                files.append((doc_id, Path(source_path)))
            else:
                logger.warning(f"File not found: {source_path}")
        return files

class FinalOptimizedPipeline:
    """
    Final optimized pipeline for CPU:
    - Model loaded ONCE at startup
    - Large batches (128 chunks) for efficient processing
    - Async I/O to overlap operations
    - Thermal management with cooling breaks
    - Minimal memory overhead
    """
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        
        # Load model ONCE - this is the key optimization
        logger.info("Loading embedding model (one-time operation)...")
        start = time.time()
        self.embedding_model = SentenceTransformer(
            EmbeddingModel.ALL_MPNET_BASE_V2.value,
            device='cpu'
        )
        logger.info(f"Model loaded in {time.time()-start:.2f}s")
        
        # Optimal settings for CPU
        self.embedding_batch_size = 128  # Sweet spot for CPU
        self.write_batch_size = 300      # Write in larger chunks
        
        # Thread pool for I/O operations only
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info(f"Pipeline ready: batch_size={self.embedding_batch_size}")

    def _embed_batch_sync(self, texts: List[str]) -> List[List[float]]:
        """Synchronous embedding - model already loaded"""
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.embedding_batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    async def _embed_chunks_async(self, chunk_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Async wrapper for embedding"""
        if not chunk_dicts:
            return []
        
        texts = [chunk['chunk_text'] for chunk in chunk_dicts]
        start_time = time.time()
        
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            self.executor,
            self._embed_batch_sync,
            texts
        )
        
        # Update chunks
        for chunk, embedding in zip(chunk_dicts, embeddings):
            chunk['embedding'] = embedding
            chunk['embedding_model'] = EmbeddingModel.ALL_MPNET_BASE_V2.value
        
        duration = time.time() - start_time
        speed = len(texts) / duration if duration > 0 else 0
        logger.info(f"Embedded {len(texts)} chunks in {duration:.2f}s ({speed:.1f} chunks/sec)")
        
        return chunk_dicts

    def _initialize_json_file(self, file_path: Path):
        """Initialize JSON file"""
        if os.path.exists(file_path):
            os.remove(file_path)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('{\n  "rows": [\n')
        
        logger.info(f"Initialized: {file_path}")

    async def _append_to_json_async(self, chunks: List[Dict[str, Any]], file_path: Path, needs_comma: bool):
        """Async JSON append"""
        if not chunks:
            return

        # Prepare JSON in memory
        json_parts = []
        for i, chunk in enumerate(chunks):
            if needs_comma or i > 0:
                json_parts.append(',\n')
            
            cleaned = {
                "chunk_id": chunk.get("chunk_id", f"unknown_{i}"),
                "doc_id": chunk.get("doc_id", ""),
                "doc_name": chunk.get("doc_name", ""),
                "chunk_index": chunk.get("chunk_index", i),
                "chunk_text": chunk.get("chunk_text", ""),
                "chunk_size": chunk.get("chunk_size", 0),
                "chunk_tokens": chunk.get("chunk_tokens", 0),
                "chunk_method": chunk.get("chunk_method", "recursive"),
                "chunk_overlap": chunk.get("chunk_overlap", 0),
                "domain": chunk.get("domain", "general"),
                "content_type": chunk.get("content_type", "text"),
                "embedding_model": chunk.get("embedding_model", ""),
                "created_at": chunk.get("created_at", ""),
                "embedding": chunk.get("embedding", [])
            }
            
            if not isinstance(cleaned["embedding"], list) or len(cleaned["embedding"]) == 0:
                continue
            
            chunk_str = json.dumps(cleaned, ensure_ascii=False)
            json_parts.append('    ' + chunk_str.replace('\n', '\n    '))

        # Write to file in thread pool
        content = ''.join(json_parts)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._write_to_file,
            file_path,
            content
        )

    def _write_to_file(self, file_path: Path, content: str):
        """Sync file write"""
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(content)

    def _finalize_json_file(self, file_path: Path):
        """Finalize JSON file"""
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write('\n  ]\n}\n')
        
        # Quick validation
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        rows = data.get("rows", [])
        logger.info(f"Finalized: {len(rows)} rows")

    async def _process_file_async(self, file_path: Path) -> List[Dict[str, Any]]:
        """Async file processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._process_file_sync,
            file_path
        )

    def _process_file_sync(self, file_path: Path) -> List[Dict[str, Any]]:
        """Sync file processing"""
        try:
            document = DocumentReader.read_file(file_path)
            if not document:
                return []
            chunks = ChunkingService.chunk_document(document, self.config)
            return [chunk.model_dump(by_alias=False) for chunk in chunks]
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return []

    async def process_pipeline(
        self, 
        input_dir: str, 
        output_dir: str,
        use_bulk_import: bool = True
    ):
        """Main pipeline"""
        
        pipeline_start = time.time()
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        output_file = output_path / "prepared_data.json"
        
        try:
            self._initialize_json_file(output_file)
            logger.info("=== STARTING OPTIMIZED CPU PIPELINE ===")
            
            # Find files
            #files = DocumentReader.find_files(input_path) 
            # Try to fetch files from SQLite (if database exists)
            db_path = input_path / "documents.db"
            if db_path.exists():
                logger.info(f"Fetching pending files from SQLite: {db_path}")
                files = get_files_from_sqlite(str(db_path))
                # get_files_from_sqlite returns list of (doc_id, Path)
                files = [(doc_id, path) for doc_id, path in files]
    
            else:
                logger.info("No SQLite DB found — scanning input directory instead.")
                file_paths = DocumentReader.find_files(input_path)
                # Wrap with dummy doc_ids (None)
                files = [(None, path) for path in file_paths]

            logger.info(f"Found {len(files)} files to process")

            
            if not files:
                self._finalize_json_file(output_file)
                return
            
            total_chunks = 0
            
            # Process each file
            for i, (doc_id,file_path) in enumerate(files):
                file_start = time.time()
                logger.info(f"[{i+1}/{len(files)}] Processing: {file_path.name} (doc_id={doc_id})")
                
                try:
                    # Read and chunk
                    chunk_dicts = await self._process_file_async(file_path)
                    # Attach doc_id (if from SQLite)
                    if doc_id is not None:                        
                        for chunk in chunk_dicts:                            
                            chunk["doc_id"] = doc_id

                    
                    if not chunk_dicts:
                        logger.warning(f"No chunks: {file_path.name}")
                        continue
                    
                    logger.info(f"Generated {len(chunk_dicts)} chunks")
                    
                    # Process in batches
                    for batch_idx in range(0, len(chunk_dicts), self.write_batch_size):
                        batch_end = min(batch_idx + self.write_batch_size, len(chunk_dicts))
                        batch = chunk_dicts[batch_idx:batch_end]
                        
                        batch_num = batch_idx // self.write_batch_size + 1
                        total_batches = (len(chunk_dicts) + self.write_batch_size - 1) // self.write_batch_size
                        logger.info(f"Batch {batch_num}/{total_batches}: {len(batch)} chunks")
                        
                        # Embed
                        embedded = await self._embed_chunks_async(batch)
                        
                        # Write
                        await self._append_to_json_async(
                            embedded,
                            output_file,
                            total_chunks > 0
                        )
                        
                        total_chunks += len(embedded)
                        
                        # Thermal management: brief pause every 3 batches
                        if batch_num % 3 == 0:
                            await asyncio.sleep(0.3)
                        else:
                            await asyncio.sleep(0)
                        
                        gc.collect()
                    
                    file_time = time.time() - file_start
                    logger.info(f"Completed {file_path.name} in {file_time:.2f}s")
                    
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    continue
            
            # Finalize
            self._finalize_json_file(output_file)
            logger.info(f"Total chunks processed: {total_chunks}")
            
            # Bulk import
            if use_bulk_import and total_chunks > 0:
                logger.info("Starting bulk import...")
                try:
                    importer = EnhancedMilvusBulkImporter()
                    object_name = importer.upload_to_minio(output_file)
                    importer.run_bulk_import("rag_chunks_test1", object_name)
                except Exception as e:
                    logger.error(f"Bulk import failed: {e}", exc_info=True)

            # Populate SQLite database from the generated JSON
            if db_path.exists() and total_chunks > 0:
                logger.info(f"Populating SQLite database: {db_path}")
                populate_db_from_json(str(db_path), str(output_file))
            
            total_time = time.time() - pipeline_start
            logger.info("=== PIPELINE COMPLETED ===")
            logger.info(f"Total time: {total_time:.2f}s")
            if total_chunks > 0:
                logger.info(f"Average speed: {total_chunks/total_time:.1f} chunks/sec")
                
        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
            try:
                self._finalize_json_file(output_file)
            except:
                pass
            raise
        finally:
            self.executor.shutdown(wait=True)


async def main():
    parser = argparse.ArgumentParser(description="Final optimized CPU pipeline")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=256)
    parser.add_argument("--no-bulk-import", action="store_true")
    
    args = parser.parse_args()
    
    config = ProcessingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    pipeline = FinalOptimizedPipeline(config)
    await pipeline.process_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_bulk_import=not args.no_bulk_import
    )


if __name__ == "__main__":
    asyncio.run(main()) 
