# src/project/tei_pipeline.py

import argparse
import json
import logging
import time
import os
import gc
from pathlib import Path
from typing import List, Dict, Any
from pymilvus import MilvusClient, DataType

from project.pydantic_models import ProcessingConfig, ChunkingMethod
from project.doc_reader import DocumentReader
from project.chunker import OptimizedChunkingService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TEIMilvusPipeline:
    """Pipeline using Milvus native embedding with Hugging Face TEI."""
    
    def __init__(self, config: ProcessingConfig, tei_endpoint: str = "http://localhost:8080"):
        self.config = config
        self.tei_endpoint = tei_endpoint
        self.client = None
        self.collection_name = "rag_chunks_tei"
        
        # Initialize local MPS preprocessing for text optimization
        import torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        if self.device == "mps":
            logger.info("Using MPS (Metal Performance Shaders) for local preprocessing")
            torch.backends.mps.allow_tf32 = True
        else:
            logger.info("MPS not available, using CPU for preprocessing")
        
    def _setup_milvus_client(self):
        """Initialize Milvus client and collection with TEI embedding function."""
        self.client = MilvusClient(uri="http://localhost:19530")
        
        # Drop existing collection if it exists
        if self.client.has_collection(self.collection_name):
            logger.info(f"Dropping existing collection: {self.collection_name}")
            self.client.drop_collection(self.collection_name)
        
        # Create schema
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        
        # Add fields to match your existing pipeline
        schema.add_field("chunk_id", DataType.VARCHAR, max_length=255, is_primary=True)
        schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
        schema.add_field("chunk_index", DataType.INT64)
        schema.add_field("chunk_text", DataType.VARCHAR, max_length=65535)
        schema.add_field("chunk_size", DataType.INT64)
        schema.add_field("chunk_tokens", DataType.INT64)
        schema.add_field("chunk_method", DataType.VARCHAR, max_length=50)
        schema.add_field("chunk_overlap", DataType.INT64)
        schema.add_field("domain", DataType.VARCHAR, max_length=100)
        schema.add_field("content_type", DataType.VARCHAR, max_length=50)
        schema.add_field("embedding_model", DataType.VARCHAR, max_length=200)
        schema.add_field("created_at", DataType.VARCHAR, max_length=50)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=768)  # DistilRoBERTa uses 768 dims
        
        # TEI will generate embeddings manually via HTTP requests
        # No need for Milvus embedding functions
        
        # Index parameters
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        
        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            consistency_level="Bounded"
        )
        
        logger.info(f"Created collection: {self.collection_name} with TEI embedding function")
    
    def process_complete_pipeline(self, input_dir: str, output_dir: str, use_direct_insert: bool = True):
        """Process pipeline using TEI for embeddings."""
        
        pipeline_start_time = time.time()
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info("=== STARTING TEI-MILVUS PIPELINE ===")
        
        # Setup Milvus client
        self._setup_milvus_client()
        
        # Step 1: File Discovery
        logger.info("Step 1: Discovering files...")
        files_to_process = DocumentReader.find_files(input_path)
        logger.info(f"Found {len(files_to_process)} files")
        
        if not files_to_process:
            logger.warning("No files found to process")
            return
        
        total_chunks_processed = 0
        all_chunk_data = []
        
        # Process each file
        for i, file_path in enumerate(files_to_process):
            file_start_time = time.time()
            logger.info(f"--- Processing file {i+1}/{len(files_to_process)}: {file_path.name} ---")
            
            # Step 2: Read and Chunk (no embedding here)
            chunk_dicts = self._process_file_sync(file_path)
            
            if not chunk_dicts:
                logger.warning(f"No chunks generated for {file_path.name}.")
                continue
            
            logger.info(f"Generated {len(chunk_dicts)} chunks for {file_path.name}")
            
            # Prepare data for Milvus (without embeddings - TEI will handle this)
            processed_chunks = self._prepare_chunks_for_tei(chunk_dicts)
            all_chunk_data.extend(processed_chunks)
            
            total_chunks_processed += len(processed_chunks)
            file_duration = time.time() - file_start_time
            logger.info(f"--- Finished processing {file_path.name} in {file_duration:.2f}s ---")
        
        # Step 3: Insert into Milvus (TEI handles embeddings automatically)
        if use_direct_insert and total_chunks_processed > 0:
            logger.info("Step 3: Inserting chunks into Milvus (TEI will generate embeddings)...")
            self._insert_chunks_with_tei(all_chunk_data)
        
        total_time = time.time() - pipeline_start_time
        
        logger.info("=== TEI-MILVUS PIPELINE COMPLETED ===")
        logger.info(f"Total processing time: {total_time:.2f}s")
        logger.info(f"Total chunks processed: {total_chunks_processed}")
        if total_chunks_processed > 0:
            logger.info(f"Average processing speed: {total_chunks_processed/total_time:.1f} chunks/second")
    
    def _process_file_sync(self, file_path: Path) -> List[Dict[str, Any]]:
        """Synchronous file processing for a single file."""
        try:
            document = DocumentReader.read_file(file_path)
            if not document: 
                return []
            chunks = OptimizedChunkingService.chunk_document(document, self.config)
            return [chunk.model_dump() for chunk in chunks]
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")
            return []
    
    def _prepare_chunks_for_tei(self, chunk_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare chunks with TEI embeddings."""
        import requests
        
        prepared_chunks = []
        texts = [chunk.get("chunk_text", "") for chunk in chunk_dicts]
        
        # Get embeddings from TEI in batches
        embeddings = []
        batch_size = 32  # Process in smaller batches
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                response = requests.post(
                    f"{self.tei_endpoint}/embed",
                    json={"inputs": batch_texts},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                batch_embeddings = response.json()
                
                # TEI returns embeddings directly as arrays, not wrapped
                if isinstance(batch_embeddings, list) and len(batch_embeddings) > 0:
                    if isinstance(batch_embeddings[0], list):
                        # Direct array format
                        embeddings.extend(batch_embeddings)
                    else:
                        # Single embedding case
                        embeddings.append(batch_embeddings)
                else:
                    logger.error(f"Unexpected TEI response format: {type(batch_embeddings)}")
                    return []
                    
            except Exception as e:
                logger.error(f"TEI batch embedding failed: {e}")
                logger.error(f"Response status: {response.status_code if 'response' in locals() else 'N/A'}")
                logger.error(f"Response text: {response.text if 'response' in locals() else 'N/A'}")
                return []
        
        for chunk, embedding in zip(chunk_dicts, embeddings):
            # Ensure embedding is a proper list of floats
            if isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding):
                embedding_vector = [float(x) for x in embedding]
            else:
                logger.error(f"Invalid embedding format: {type(embedding)}, content: {embedding}")
                continue
                
            prepared_chunk = {
                "chunk_id": chunk.get("id", chunk.get("chunk_id")),
                "doc_id": chunk.get("doc_id"),
                "chunk_index": chunk.get("chunk_index", 0),
                "chunk_text": chunk.get("chunk_text", ""),
                "chunk_size": chunk.get("chunk_size", 0),
                "chunk_tokens": chunk.get("chunk_tokens"),
                "chunk_method": chunk.get("chunk_method", ChunkingMethod.RECURSIVE.value),
                "chunk_overlap": chunk.get("chunk_overlap", 0),
                "domain": chunk.get("domain", "general"),
                "content_type": chunk.get("content_type", "text"),
                "embedding_model": "huggingface-tei",
                "created_at": chunk.get("created_at", ""),
                "embedding": embedding_vector  # Properly formatted float array
            }
            prepared_chunks.append(prepared_chunk)
        
        return prepared_chunks
    
    def _insert_chunks_with_tei(self, chunk_data: List[Dict[str, Any]]):
        """Insert chunks into Milvus collection with TEI embedding generation."""
        batch_size = 100
        total_batches = (len(chunk_data) + batch_size - 1) // batch_size
        
        logger.info(f"Inserting {len(chunk_data)} chunks in {total_batches} batches")
        
        for batch_idx in range(total_batches):
            batch_start_time = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunk_data))
            
            batch_chunks = chunk_data[start_idx:end_idx]
            
            try:
                # Insert batch - Milvus with TEI will automatically generate embeddings
                # from the chunk_text field
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch_chunks
                )
                
                batch_duration = time.time() - batch_start_time
                logger.info(f"Inserted batch {batch_idx + 1}/{total_batches}: "
                           f"{len(batch_chunks)} chunks in {batch_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error inserting batch {batch_idx + 1}: {e}")
                continue
        
        # Flush to ensure all data is written
        self.client.flush(collection_name=self.collection_name)
        logger.info("All chunks inserted and flushed to Milvus")
    
    def search_with_tei(self, query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search using TEI embeddings."""
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_text],  # TEI will embed this automatically
                limit=limit,
                output_fields=["chunk_id", "doc_id", "chunk_text", "domain", "content_type"]
            )
            
            search_results = []
            for i, hit in enumerate(results[0]):
                result = {
                    "rank": i + 1,
                    "chunk_id": hit["entity"]["chunk_id"],
                    "doc_id": hit["entity"]["doc_id"],
                    "chunk_text": hit["entity"]["chunk_text"],
                    "domain": hit["entity"]["domain"],
                    "content_type": hit["entity"]["content_type"],
                    "distance": hit["distance"],
                    "similarity_score": 1.0 - hit["distance"]  # Convert distance to similarity
                }
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            if self.client and self.client.has_collection(self.collection_name):
                stats = self.client.get_collection_stats(self.collection_name)
                return {
                    "total_chunks": stats["row_count"],
                    "collection_name": self.collection_name,
                    "status": "ready",
                    "embedding_function": "huggingface-tei"
                }
            else:
                return {"total_chunks": 0, "status": "no_collection"}
        except Exception as e:
            return {"error": str(e), "status": "error", "total_chunks": 0}


def main():
    parser = argparse.ArgumentParser(description="TEI-Milvus optimized RAG pipeline")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=256)
    parser.add_argument("--tei-endpoint", type=str, default="http://localhost:8080",
                       help="TEI server endpoint")
    parser.add_argument("--no-direct-insert", action="store_true", 
                       help="Skip direct insertion into Milvus")
    
    args = parser.parse_args()
    
    config = ProcessingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    pipeline = TEIMilvusPipeline(config, tei_endpoint=args.tei_endpoint)
    
    pipeline.process_complete_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_direct_insert=not args.no_direct_insert
    )


if __name__ == "__main__":
    main()