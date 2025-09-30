# src/project/complete_pipeline.py

import argparse
import asyncio
import json
import logging
import time
import os
import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Iterator
import torch
import numpy as np

from project.pydantic_models import ProcessingConfig, EmbeddingModel
from project.doc_reader import DocumentReader
from project.chunker import OptimizedChunkingService
from project.milvus_bulk_import import EnhancedMilvusBulkImporter
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CompleteOptimizedPipeline:
    """Pipeline optimized for MPS memory management and consistent performance."""
    
    def __init__(self, config: ProcessingConfig, max_workers: int = 2):
        self.config = config
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info(f"Initializing embedding model on {self.device}")
        
        self.embedding_model = SentenceTransformer(
            EmbeddingModel.ALL_MPNET_BASE_V2.value,
            device=self.device
        )
        
        # Conservative batch sizes for consistent MPS performance
        if self.device == "mps":
            self.embedding_batch_size = 32  # Small batch for consistent performance
            self.chunk_batch_size = 200     # Process chunks in smaller groups
        else:
            self.embedding_batch_size = 64
            self.chunk_batch_size = 500
        
        # Pre-warm the model with small batch
        logger.info("Pre-warming embedding model...")
        dummy_texts = ["This is a test sentence."] * 5
        _ = self.embedding_model.encode(dummy_texts, show_progress_bar=False)
        if self.device == "mps":
            torch.mps.empty_cache()
        logger.info("Model pre-warming complete")
        

    def _embed_chunks_with_mps_management(self, chunk_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed chunks with proper MPS memory management."""
        texts = [chunk['chunk_text'] for chunk in chunk_dicts]
        
        try:
            # Clear memory before processing
            if self.device == "mps":
                torch.mps.empty_cache()
            
            # Process in small, consistent batches
            all_embeddings = []
            num_batches = (len(texts) + self.embedding_batch_size - 1) // self.embedding_batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.embedding_batch_size
                end_idx = min(start_idx + self.embedding_batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                # Process batch
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    batch_size=self.embedding_batch_size,
                    show_progress_bar=False,  # Disable progress bar for cleaner logs
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.device
                )
                
                all_embeddings.append(batch_embeddings)
                
                # Clean up after each batch
                if self.device == "mps":
                    torch.mps.empty_cache()
            
            # Concatenate all embeddings
            if len(all_embeddings) > 1:
                embeddings = np.vstack(all_embeddings)
            else:
                embeddings = all_embeddings[0]
            
            # Update chunks with embeddings
            for chunk, embedding in zip(chunk_dicts, embeddings):
                chunk['embedding'] = embedding.tolist()  # This is correct
                #chunk.pop('embedding_vector', None)      # Remove this line if it exists
                chunk['embedding_model'] = EmbeddingModel.ALL_MPNET_BASE_V2.value
                
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []
            
        return chunk_dicts

    def _cleanup_memory(self):
        """Aggressive memory cleanup for MPS."""
        if self.device == "mps":
            torch.mps.empty_cache()
        gc.collect()

    # Fixed JSON writing methods for complete_pipeline.py
    # Replace the existing methods in your CompleteOptimizedPipeline class

    import json
    import os
    import logging
    from pathlib import Path
    from typing import List, Dict, Any

    logger = logging.getLogger(__name__)

    def _initialize_json_file(self, file_path: Path):
        """Creates an empty file and writes the initial JSON structure."""
        try:
            # Remove existing file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Removed existing file: {file_path}")
            
            # Create directory if it doesn't exist
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('{\n  "rows": [\n')
                
            logger.info(f"Initialized JSON file: {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize JSON file: {e}")
            raise

    def _append_to_json_file(self, chunks: List[Dict[str, Any]], file_path: Path, needs_comma: bool):
        """Appends a list of chunks to the JSON file, ensuring valid syntax."""
        if not chunks:
            logger.warning("No chunks to append")
            return

        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                for i, chunk in enumerate(chunks):
                    # Add comma before this chunk if needed
                    if needs_comma or i > 0:
                        f.write(',\n')
                    
                    # Clean the chunk data - ensure all required fields are present
                    cleaned_chunk = {
                        "chunk_id": chunk.get("chunk_id", chunk.get("id", f"unknown_{i}")),
                        "doc_id": chunk.get("doc_id", ""),
                        "chunk_index": chunk.get("chunk_index", i),
                        "chunk_text": chunk.get("chunk_text", ""),
                        "chunk_size": chunk.get("chunk_size", len(str(chunk.get("chunk_text", "")))) or 0,
                        "chunk_tokens": chunk.get("chunk_tokens"),
                        "chunk_method": chunk.get("chunk_method", "recursive"),
                        "chunk_overlap": chunk.get("chunk_overlap", 0),
                        "domain": chunk.get("domain", "general"),
                        "content_type": chunk.get("content_type", "text"),
                        "embedding_model": chunk.get("embedding_model") or "not_embedded",
                        "created_at": chunk.get("created_at",""),
                        "embedding": chunk.get("embedding")  # This should be a list of floats
                    }
                    
                    # Validate embedding field
                    if not isinstance(cleaned_chunk["embedding"], list):
                        logger.error(f"Invalid embedding type for chunk {cleaned_chunk['chunk_id']}: {type(cleaned_chunk['embedding'])}")
                        continue
                    # Validate embedding field - skip chunk if it's missing
                    if not isinstance(cleaned_chunk["embedding"], list) or len(cleaned_chunk["embedding"]) == 0:
                        logger.error(f"Skipping chunk {cleaned_chunk['chunk_id']} due to missing or invalid embedding.")
                        continue # CRITICAL: Skip this chunk
                    
                    if len(cleaned_chunk["embedding"]) == 0:
                        logger.error(f"Empty embedding for chunk {cleaned_chunk['chunk_id']}")
                        continue
                    
                    # Pretty-print each chunk with indentation
                    chunk_str = json.dumps(cleaned_chunk, indent=4, ensure_ascii=False)
                    indented_chunk_str = '    ' + chunk_str.replace('\n', '\n    ')
                    f.write(indented_chunk_str)

            logger.info(f"Appended {len(chunks)} chunks to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to append chunks to JSON file: {e}")
            raise

    def _finalize_json_file(self, file_path: Path):
        """Finalizes the JSON file by adding the closing characters."""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write('\n  ]\n}\n')
            logger.info(f"JSON file '{file_path.name}' has been finalized.")
            
            # Validate the JSON file
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                rows = data.get("rows", [])
                logger.info(f"JSON validation successful: {len(rows)} rows")
                
                if rows:
                    first_row = rows[0]
                    logger.info(f"First row keys: {list(first_row.keys())}")
                    
                    if "embedding" in first_row:
                        embedding = first_row["embedding"]
                        if isinstance(embedding, list):
                            logger.info(f"Embedding validation successful: {len(embedding)} dimensions")
                        else:
                            logger.error(f"Invalid embedding type: {type(embedding)}")
                    else:
                        logger.error("No embedding field in first row")
                        
            except json.JSONDecodeError as e:
                logger.error(f"JSON validation failed: {e}")
                raise
            except Exception as e:
                logger.error(f"JSON validation error: {e}")
                
        except Exception as e:
            logger.error(f"Failed to finalize JSON file: {e}")
            raise

    # Fixed method for processing large files with better JSON writing
    def _process_large_file_in_batches(
        self, 
        chunk_dicts: List[Dict[str, Any]], 
        output_file_path: Path,
        needs_comma: bool
    ) -> int:
        """Process large files in smaller batches to maintain consistent performance."""
        total_processed = 0
        num_batches = (len(chunk_dicts) + self.chunk_batch_size - 1) // self.chunk_batch_size
        
        logger.info(f"Processing {len(chunk_dicts)} chunks in {num_batches} batches")
        
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * self.chunk_batch_size
            end_idx = min(start_idx + self.chunk_batch_size, len(chunk_dicts))
            
            batch_chunks = chunk_dicts[start_idx:end_idx]
            logger.info(f"Processing batch {batch_idx + 1}/{num_batches}: chunks {start_idx}-{end_idx-1}")
            
            try:
                # Embed batch with MPS management
                embedded_chunks = self._embed_chunks_with_mps_management(batch_chunks)
                
                # Append to file
                self._append_to_json_file(
                    embedded_chunks, 
                    output_file_path, 
                    needs_comma or total_processed > 0
                )
                
                total_processed += len(embedded_chunks)
                batch_duration = time.time() - batch_start_time
                speed = len(embedded_chunks) / batch_duration if batch_duration > 0 else 0
                
                logger.info(f"Batch {batch_idx + 1} completed in {batch_duration:.2f}s ({speed:.1f} chunks/sec)")
                
                # Update needs_comma for next batch
                needs_comma = True
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_idx + 1}: {e}")
                # Continue with next batch
                continue
                
            # Aggressive cleanup between batches
            self._cleanup_memory()
        
        return total_processed

    # Complete replacement for process_complete_pipeline method
    def process_complete_pipeline(
        self, 
        input_dir: str, 
        output_dir: str,
        use_bulk_import: bool = True
    ) -> None:
        """Pipeline with chunked processing for large files and robust JSON writing."""
        
        pipeline_start_time = time.time()
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        output_file_path = output_path / "prepared_data.json"
        
        try:
            # Initialize JSON file with proper error handling
            self._initialize_json_file(output_file_path)

            logger.info("=== STARTING MPS-OPTIMIZED PIPELINE ===")
            
            # Step 1: File Discovery
            logger.info("Step 1: Discovering files...")
            files_to_process = DocumentReader.find_files(input_path)
            logger.info(f"Found {len(files_to_process)} files")
            
            if not files_to_process:
                logger.warning("No files found to process")
                self._finalize_json_file(output_file_path)
                return
                    
            total_chunks_processed = 0

            # Process each file
            for i, file_path in enumerate(files_to_process):
                file_start_time = time.time()
                logger.info(f"--- Processing file {i+1}/{len(files_to_process)}: {file_path.name} ---")
                
                try:
                    # Step 2: Read and Chunk
                    chunking_start_time = time.time()
                    chunk_dicts = self._process_file_sync(file_path)
                    chunking_duration = time.time() - chunking_start_time
                    
                    if not chunk_dicts:
                        logger.warning(f"No chunks generated for {file_path.name}.")
                        continue
                        
                    logger.info(f"Chunking finished in {chunking_duration:.2f}s. Found {len(chunk_dicts)} chunks.")
                    
                    # Step 3: Process chunks in smaller batches for large files
                    if len(chunk_dicts) > self.chunk_batch_size:
                        logger.info(f"Large file detected. Processing in batches of {self.chunk_batch_size}")
                        chunks_processed = self._process_large_file_in_batches(
                            chunk_dicts, output_file_path, total_chunks_processed > 0
                        )
                    else:
                        # Process smaller files normally
                        embedding_start_time = time.time()
                        embedded_chunks = self._embed_chunks_with_mps_management(chunk_dicts)
                        embedding_duration = time.time() - embedding_start_time
                        
                        embedding_speed = len(embedded_chunks) / embedding_duration if embedding_duration > 0 else 0
                        logger.info(f"Embedding finished in {embedding_duration:.2f}s ({embedding_speed:.1f} chunks/sec)")
                        
                        self._append_to_json_file(embedded_chunks, output_file_path, total_chunks_processed > 0)
                        chunks_processed = len(embedded_chunks)
                    
                    total_chunks_processed += chunks_processed
                    file_duration = time.time() - file_start_time
                    logger.info(f"--- Finished processing {file_path.name} in {file_duration:.2f}s ---")
                    
                except Exception as e:
                    logger.error(f"Error processing file {file_path.name}: {e}")
                    logger.exception("Full traceback:")
                    continue

            # Always finalize the JSON file
            self._finalize_json_file(output_file_path)
            logger.info(f"--- All files processed. Total chunks: {total_chunks_processed} ---")

            # Step 4: Storage (Bulk Import)
            storage_start = time.time()
            if use_bulk_import and total_chunks_processed > 0:
                logger.info("Step 4: Starting Bulk Import into Milvus...")
                try:
                    from project.milvus_bulk_import import EnhancedMilvusBulkImporter
                    importer = EnhancedMilvusBulkImporter()
                    object_name = importer.upload_to_minio(output_file_path)
                    importer.run_bulk_import("rag_chunks", object_name)
                except Exception as e:
                    logger.error(f"Bulk import process failed: {e}", exc_info=True)
            else:
                logger.info("Step 4: Skipping bulk import.")

            storage_time = time.time() - storage_start
            total_time = time.time() - pipeline_start_time
            
            logger.info("=== PIPELINE COMPLETED ===")
            logger.info(f"Total processing time: {total_time:.2f}s")
            if total_chunks_processed > 0:
                logger.info(f"Average processing speed: {total_chunks_processed/total_time:.1f} chunks/second")
                
        except Exception as e:
            logger.error(f"Critical error in pipeline: {e}", exc_info=True)
            
            # Try to finalize JSON file even if there was an error
            try:
                self._finalize_json_file(output_file_path)
                logger.info("JSON file finalized despite error")
            except Exception as finalize_error:
                logger.error(f"Could not finalize JSON file: {finalize_error}")
            raise

    # Additional validation method
    def _validate_chunk_data(self, chunk: Dict[str, Any]) -> bool:
        """Validate chunk data before writing to JSON"""
        required_fields = [
            "chunk_id", "doc_id", "chunk_index", "chunk_text", 
            "chunk_size", "chunk_method", "chunk_overlap", 
            "domain", "content_type", "embedding"
        ]
        
        for field in required_fields:
            if field not in chunk:
                logger.error(f"Missing required field '{field}' in chunk")
                return False
        
        # Validate embedding specifically
        embedding = chunk.get("embedding")
        if not isinstance(embedding, list):
            logger.error(f"Embedding must be a list, got {type(embedding)}")
            return False
            
        if len(embedding) == 0:
            logger.error("Embedding cannot be empty")
            return False
            
        # Check if all embedding values are numbers
        for i, val in enumerate(embedding[:5]):  # Check first 5 values
            if not isinstance(val, (int, float)):
                logger.error(f"Embedding contains non-numeric value at index {i}: {type(val)}")
                return False
        
        return True

    def _process_file_sync(self, file_path: Path) -> List[Dict[str, Any]]:
        """Synchronous file processing for a single file"""
        try:
            document = DocumentReader.read_file(file_path)
            if not document: 
                return []
            chunks = OptimizedChunkingService.chunk_document(document, self.config)
            return [chunk.model_dump(by_alias=False) for chunk in chunks]
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return []

def main():
    parser = argparse.ArgumentParser(description="MPS-optimized RAG pipeline")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=256)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--no-bulk-import", action="store_true", help="Skip the Milvus bulk import step.")
    parser.add_argument("--embedding-batch-size", type=int, default=32,
                       help="Embedding batch size (keep small for MPS stability)")
    parser.add_argument("--chunk-batch-size", type=int, default=200,
                       help="Number of chunks to process in each file batch")
    
    args = parser.parse_args()
    
    config = ProcessingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    pipeline = CompleteOptimizedPipeline(config, max_workers=args.max_workers)
    
    # Override batch sizes if specified
    pipeline.embedding_batch_size = args.embedding_batch_size
    pipeline.chunk_batch_size = args.chunk_batch_size
    
    pipeline.process_complete_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_bulk_import=not args.no_bulk_import
    )

if __name__ == "__main__":
    main()