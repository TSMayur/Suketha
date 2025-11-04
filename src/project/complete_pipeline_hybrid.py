"""
STREAMING PIPELINE - Parallel Producer/Consumer with True Streaming
- Read file in chunks (10-50MB blocks) - NO FULL FILE IN MEMORY
- Parse → Chunk → Queue → Insert in parallel
- Memory capped at ~150MB (1500 chunks max in queue)
- Producer and Consumer run in separate threads
- Comprehensive logging for performance tracking
"""
import argparse
import json
import logging
import time
import os
from pathlib import Path
from typing import List, Dict, Any, Iterator
import sqlite3
from queue import Queue
from dataclasses import dataclass, field
from datetime import datetime
import threading

from .config import client, COLLECTION_NAME
from project.pydantic_models import ProcessingConfig, Document, DocumentType
from project.doc_reader import DocumentReader
from project.chunker import ChunkingService
from project.chunk_cleaner import clean_chunks_before_embedding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineMetrics:
    """Track detailed pipeline metrics"""
    total_files: int = 0
    files_processed: int = 0
    total_chunks: int = 0
    total_read_time: float = 0.0
    total_chunk_time: float = 0.0
    total_clean_time: float = 0.0
    total_insert_time: float = 0.0
    total_wait_time: float = 0.0
    batch_count: int = 0
    batch_times: List[float] = field(default_factory=list)
    chunks_per_batch: List[int] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    
    def add_batch_insert(self, batch_size: int, insert_time: float):
        """Record batch insert metrics"""
        self.batch_count += 1
        self.batch_times.append(insert_time)
        self.chunks_per_batch.append(batch_size)
        self.total_insert_time += insert_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        elapsed = time.time() - self.start_time
        
        avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        min_batch_time = min(self.batch_times) if self.batch_times else 0
        max_batch_time = max(self.batch_times) if self.batch_times else 0
        
        return {
            "total_elapsed_sec": elapsed,
            "total_elapsed_min": elapsed / 60,
            "files_processed": self.files_processed,
            "total_chunks": self.total_chunks,
            "chunks_per_sec": self.total_chunks / elapsed if elapsed > 0 else 0,
            "batches": {
                "count": self.batch_count,
                "avg_size": sum(self.chunks_per_batch) / len(self.chunks_per_batch) if self.chunks_per_batch else 0,
                "avg_time_sec": avg_batch_time,
                "min_time_sec": min_batch_time,
                "max_time_sec": max_batch_time,
                "total_time_sec": self.total_insert_time,
                "avg_throughput": sum(self.chunks_per_batch) / sum(self.batch_times) if sum(self.batch_times) > 0 else 0
            },
            "time_breakdown": {
                "read_sec": self.total_read_time,
                "read_pct": (self.total_read_time / elapsed * 100) if elapsed > 0 else 0,
                "chunk_sec": self.total_chunk_time,
                "chunk_pct": (self.total_chunk_time / elapsed * 100) if elapsed > 0 else 0,
                "clean_sec": self.total_clean_time,
                "clean_pct": (self.total_clean_time / elapsed * 100) if elapsed > 0 else 0,
                "insert_sec": self.total_insert_time,
                "insert_pct": (self.total_insert_time / elapsed * 100) if elapsed > 0 else 0,
                "wait_sec": self.total_wait_time,
                "wait_pct": (self.total_wait_time / elapsed * 100) if elapsed > 0 else 0,
            }
        }


class StreamingFileReader:
    """
    True streaming file reader - yields text in chunks (10-50MB blocks).
    NEVER loads entire file into memory.
    """
    
    @staticmethod
    def stream_read_text_file(file_path: Path, chunk_size_mb: int = 10) -> Iterator[str]:
        """
        Stream read a text file in chunks.
        Yields blocks of text (~10MB each) without loading entire file.
        """
        chunk_size = chunk_size_mb * 1024 * 1024  # Convert MB to bytes
        
        logger.info(f"  📄 Streaming read: {chunk_size_mb}MB chunks")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            buffer = ""
            bytes_read = 0
            
            while True:
                # Read chunk
                chunk = f.read(chunk_size)
                if not chunk:
                    # Yield remaining buffer
                    if buffer:
                        yield buffer
                    break
                
                buffer += chunk
                bytes_read += len(chunk.encode('utf-8'))
                
                # Find last complete line to avoid splitting mid-sentence
                last_newline = buffer.rfind('\n')
                
                if last_newline > 0:
                    # Yield complete lines
                    yield buffer[:last_newline + 1]
                    buffer = buffer[last_newline + 1:]
                
                # Log progress
                if bytes_read % (50 * 1024 * 1024) == 0:  # Every 50MB
                    logger.debug(f"     └─ Read {bytes_read / (1024*1024):.1f}MB...")
    
    @staticmethod
    def stream_read_tsv_file(file_path: Path, chunk_rows: int = 10000) -> Iterator[List[Dict[str, Any]]]:
        """
        Stream read a TSV file in chunks.
        Yields batches of rows without loading entire file.
        """
        logger.info(f"  📄 Streaming TSV read: {chunk_rows} rows per chunk")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read header
            header = f.readline().strip().split('\t')
            
            batch = []
            row_count = 0
            
            for line in f:
                row_count += 1
                values = line.strip().split('\t')
                
                # Create dict from header and values
                if len(values) == len(header):
                    row_dict = dict(zip(header, values))
                    batch.append(row_dict)
                
                # Yield batch when full
                if len(batch) >= chunk_rows:
                    yield batch
                    batch = []
                    
                    if row_count % 50000 == 0:
                        logger.debug(f"     └─ Read {row_count} rows...")
            
            # Yield remaining rows
            if batch:
                yield batch


class StreamingPipeline:
    """
    Streaming pipeline with producer/consumer running in parallel threads.
    """
    
    def __init__(self, config: ProcessingConfig, num_consumers: int):
        self.config = config
        self.insert_batch_size = 50  # REDUCED: Start with 50 chunks per batch
        self.max_tokens_per_chunk = 380  # TEI limit: max 380 tokens per chunk
        self.queue_max_size = 1500  # Bounded queue (blocks when full)
        self.num_consumers = num_consumers 
        self.chunk_queue = Queue(maxsize=self.queue_max_size)
        self.metrics = PipelineMetrics()
        self.stream_chunk_size_mb = 10  # Read files in 10MB chunks
        self._connect_to_milvus()
    
    def _connect_to_milvus(self):
        """Verify Milvus connection"""
        if not client.has_collection(COLLECTION_NAME):
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found!")
        logger.info(f"✅ Connected to Milvus: {COLLECTION_NAME}")

        # Add thread lock for metrics (multiple consumers writing)
        self.metrics_lock = threading.Lock()

    def _split_oversized_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split chunk if exceeds token limit.
        DEFENSIVE: This function does NOT trust incoming chunk_tokens,
        as they are estimates. It calculates its own estimate.
        """
        text = chunk['chunk_text']
        
        # Use a safe, hardcoded char/token ratio to estimate.
        # We can't trust chunk.get('chunk_tokens').
        # 3.0 is a very safe "worst-case" ratio.
        SAFE_CHARS_PER_TOKEN = 3.0
        
        estimated_tokens = len(text) / SAFE_CHARS_PER_TOKEN
        
        # Now check if our OWN estimate is within the limit.
        if estimated_tokens <= self.max_tokens_per_chunk:
            chunk['chunk_tokens'] = int(estimated_tokens) # Update with our better estimate
            return [chunk]

        # If we're here, the text is definitely too long. We MUST split it.
        logger.debug(f"  ⚠️  Splitting oversized text: {len(text)} chars / {estimated_tokens:.0f} estimated tokens")
        
        # Calculate target characters per split, using our safe ratio and a buffer
        target_chars = int(self.max_tokens_per_chunk * SAFE_CHARS_PER_TOKEN * 0.9)
        
        # Handle edge case where target_chars is 0
        if target_chars == 0:
            logger.warning(f"  ⚠️  target_chars is 0, using a fallback size.")
            target_chars = self.max_tokens_per_chunk * 2

        split_chunks = []
        start = 0
        part_idx = 0
        
        while start < len(text):
            end = min(start + target_chars, len(text))
            
            if end < len(text):
                # Try to find a smart break point
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                last_space = text.rfind(' ', start, end)
                
                break_point = max(last_period, last_newline, last_space)
                if break_point > start:
                    end = break_point + 1
                # else: We do a "hard" cut at target_chars
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                new_chunk = chunk.copy()
                new_chunk['chunk_id'] = f"{chunk['chunk_id']}_part{part_idx}"
                new_chunk['chunk_text'] = chunk_text
                new_chunk['chunk_size'] = len(chunk_text)
                # Set the new *estimated* token count
                new_chunk['chunk_tokens'] = int(len(chunk_text) / SAFE_CHARS_PER_TOKEN)
                split_chunks.append(new_chunk)
                part_idx += 1
            
            start = end
            
        logger.debug(f"  ✂️  Split into {len(split_chunks)} parts")
        return split_chunks
    
    def _insert_batch(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Insert batch to Milvus with detailed logging.
        Returns number of chunks inserted.
        """
        if not chunks:
            return 0
        
        batch_start = time.time()
        batch_size = len(chunks)
        
        logger.info(f"┌─ 📦 BATCH INSERT START ────────────────────")
        logger.info(f"│  Batch size: {batch_size} chunks")
        logger.info(f"│  Queue size: {self.chunk_queue.qsize()}/{self.queue_max_size}")
        logger.info(f"│  Total processed so far: {self.metrics.total_chunks}")
        
        # Prepare data
        prep_start = time.time()
        data = []
        for chunk in chunks:
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
                "embedding_model": "tei-bm25-hybrid",
                "created_at": chunk.get("created_at", ""),
            }
            data.append(row)
        prep_time = time.time() - prep_start
        
        logger.info(f"│  Data preparation: {prep_time:.3f}s")
        
        # Insert to Milvus
        insert_start = time.time()
        try:
            result = client.insert(
                collection_name=COLLECTION_NAME,
                data=data,
                timeout=180  # FIX: Added comma and 3-minute timeout for TEI
            )
            insert_time = time.time() - insert_start
            
            # Calculate throughput
            throughput = batch_size / insert_time if insert_time > 0 else 0
            
            logger.info(f"│  Milvus insert: {insert_time:.3f}s")
            logger.info(f"│  Insert count: {result['insert_count']}")
            logger.info(f"│  Throughput: {throughput:.1f} chunks/sec")
            
            total_batch_time = time.time() - batch_start
            logger.info(f"│  Total batch time: {total_batch_time:.3f}s")
            logger.info(f"└─ ✅ BATCH INSERT COMPLETE ────────────────")
            
            # Record metrics (thread-safe)
            with self.metrics_lock:
                self.metrics.add_batch_insert(result['insert_count'], insert_time)
            
            return result['insert_count']
            
        except Exception as e:
            insert_time = time.time() - insert_start
            logger.error(f"│  ❌ INSERT FAILED after {insert_time:.3f}s")
            logger.error(f"│  Error: {e}")
            logger.error(f"└─ ❌ BATCH INSERT FAILED ──────────────────")
            raise
    
    def _stream_process_file(self, file_path: Path, doc_id: str = None) -> Iterator[Dict[str, Any]]:
        """
        TRUE STREAMING: Process file in small chunks without loading entire file.
        Yields chunks one at a time.
        """
        file_start = time.time()
        
        logger.info(f"\n╔═══════════════════════════════════════════")
        logger.info(f"║ 📂 FILE: {file_path.name}")
        logger.info(f"║ Size: {file_path.stat().st_size / (1024*1024):.2f} MB")
        logger.info(f"╚═══════════════════════════════════════════")
        
        try:
            file_ext = file_path.suffix.lower()
            
            # Determine file type and stream accordingly
            if file_ext in ['.txt', '.md', '.json', '.csv']:
                yield from self._stream_process_text_file(file_path, doc_id)
            elif file_ext in ['.tsv']:
                yield from self._stream_process_tsv_file(file_path, doc_id)
            else:
                # Fallback to old method for unsupported types
                logger.warning(f"  ⚠️  Unsupported streaming for {file_ext}, using full read")
                yield from self._fallback_process_file(file_path, doc_id)
            
            total_time = time.time() - file_start
            logger.info(f"  ⏱️  Total file time: {total_time:.3f}s")
            
        except Exception as e:
            logger.error(f"  ❌ Error processing {file_path.name}: {e}")
            return
    
    def _stream_process_text_file(self, file_path: Path, doc_id: str = None) -> Iterator[Dict[str, Any]]:
        """Stream process text-based files"""
        read_start = time.time()
        text_buffer = ""
        chunk_index = 0
        yielded_count = 0
        
        # Stream read file in chunks
        for text_chunk in StreamingFileReader.stream_read_text_file(file_path, self.stream_chunk_size_mb):
            text_buffer += text_chunk
            
            # When buffer is large enough, chunk it
            if len(text_buffer) > self.config.chunk_size * 3:
                # Create document from buffer
                from project.pydantic_models import Document
                doc = Document(
                    id=doc_id or f"doc_{file_path.stem}",  # FIX: Changed 'doc_id' to 'id'
                    title=file_path.name,                 # FIX: Changed 'doc_name' to 'title'
                    content=text_buffer,
                    document_type=DocumentType.TXT,     # FIX: Added required field (or .JSON, etc.)
                    source=str(file_path)                 # FIX: Added required field
                )
                
                # Chunk the buffer
                chunk_start = time.time()
                chunks = ChunkingService.chunk_document(doc, self.config)
                self.metrics.total_chunk_time += time.time() - chunk_start
                
                # Process chunks
                for chunk in chunks:
                    chunk_dict = chunk.model_dump(by_alias=False)
                    if doc_id:
                        chunk_dict["doc_id"] = doc_id
                    chunk_dict["chunk_index"] = chunk_index
                    chunk_index += 1
                    
                    # Clean and split
                    cleaned = clean_chunks_before_embedding([chunk_dict])
                    if cleaned:
                        split_chunks = self._split_oversized_chunk(cleaned[0])
                        for split_chunk in split_chunks:
                            yield split_chunk
                            yielded_count += 1
                
                # Keep overlap for continuity
                text_buffer = text_buffer[-self.config.chunk_overlap:]
        
        # Process remaining buffer
        if text_buffer.strip():
            from project.pydantic_models import Document
            doc = Document(
                id=doc_id or f"doc_{file_path.stem}",
                title=file_path.name,
                content=text_buffer,
                document_type=DocumentType.TXT,
                source=str(file_path)
            )
            
            chunks = ChunkingService.chunk_document(doc, self.config)
            for chunk in chunks:
                chunk_dict = chunk.model_dump(by_alias=False)
                if doc_id:
                    chunk_dict["doc_id"] = doc_id
                chunk_dict["chunk_index"] = chunk_index
                chunk_index += 1
                
                cleaned = clean_chunks_before_embedding([chunk_dict])
                if cleaned:
                    split_chunks = self._split_oversized_chunk(cleaned[0])
                    for split_chunk in split_chunks:
                        yield split_chunk
                        yielded_count += 1
        
        read_time = time.time() - read_start
        self.metrics.total_read_time += read_time
        logger.info(f"  ✅ Streamed {yielded_count} chunks in {read_time:.3f}s")
    
    def _stream_process_tsv_file(self, file_path: Path, doc_id: str = None) -> Iterator[Dict[str, Any]]:
        """Stream process TSV files"""
        read_start = time.time()
        chunk_index = 0
        yielded_count = 0
        
        # Stream read TSV in batches
        for row_batch in StreamingFileReader.stream_read_tsv_file(file_path, chunk_rows=10000):
            # Convert rows to text
            text_content = "\n".join([
                " | ".join([f"{k}: {v}" for k, v in row.items()])
                for row in row_batch
            ])
            
            # Create document from batch
            from project.pydantic_models import Document
            doc = Document(
                id=doc_id or f"doc_{file_path.stem}",
                title=file_path.name,
                content=text_content,
                document_type=DocumentType.TSV,
                source=str(file_path)
            )
            
            # Chunk the batch
            chunks = ChunkingService.chunk_document(doc, self.config)
            
            for chunk in chunks:
                chunk_dict = chunk.model_dump(by_alias=False)
                if doc_id:
                    chunk_dict["doc_id"] = doc_id
                chunk_dict["chunk_index"] = chunk_index
                chunk_index += 1
                
                cleaned = clean_chunks_before_embedding([chunk_dict])
                if cleaned:
                    split_chunks = self._split_oversized_chunk(cleaned[0])
                    for split_chunk in split_chunks:
                        yield split_chunk
                        yielded_count += 1
        
        read_time = time.time() - read_start
        self.metrics.total_read_time += read_time
        logger.info(f"  ✅ Streamed {yielded_count} chunks in {read_time:.3f}s")
    
    def _fallback_process_file(self, file_path: Path, doc_id: str = None) -> Iterator[Dict[str, Any]]:
        """Fallback for unsupported file types"""
        read_start = time.time()
        document = DocumentReader.read_file(file_path)
        read_time = time.time() - read_start
        self.metrics.total_read_time += read_time
        
        if not document:
            logger.warning(f"  ⚠️  Could not read: {file_path.name}")
            return
        
        chunk_start = time.time()
        chunks = ChunkingService.chunk_document(document, self.config)
        self.metrics.total_chunk_time += time.time() - chunk_start
        
        yielded_count = 0
        for chunk in chunks:
            chunk_dict = chunk.model_dump(by_alias=False)
            if doc_id:
                chunk_dict["doc_id"] = doc_id
            
            cleaned = clean_chunks_before_embedding([chunk_dict])
            if cleaned:
                split_chunks = self._split_oversized_chunk(cleaned[0])
                for split_chunk in split_chunks:
                    yield split_chunk
                    yielded_count += 1
        
        logger.info(f"  ✅ Processed {yielded_count} chunks (fallback)")
    
    def _producer(self, files: List[tuple]):
        """
        Producer: Read files and push chunks to bounded queue.
        Blocks when queue is full (backpressure).
        RUNS IN MAIN THREAD.
        """
        logger.info(f"\n🏭 PRODUCER START: {len(files)} files to process")
        
        for file_idx, (doc_id, file_path) in enumerate(files, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"[{file_idx}/{len(files)}] Processing file {file_idx}")
            logger.info(f"{'='*80}")
            
            file_chunks = 0
            
            # Stream chunks from file
            for chunk in self._stream_process_file(file_path, doc_id):
                # Check if queue is full (will block here if needed)
                if self.chunk_queue.full():
                    wait_start = time.time()
                    logger.warning(f"  ⏸️  Queue FULL ({self.queue_max_size}) - waiting for consumer...")
                    
                    # Put will block here until space available
                    self.chunk_queue.put(chunk)
                    
                    wait_time = time.time() - wait_start
                    logger.info(f"  ▶️  Queue space available after {wait_time:.3f}s")
                    self.metrics.total_wait_time += wait_time
                else:
                    # Put chunk in queue (non-blocking)
                    self.chunk_queue.put(chunk)
                
                file_chunks += 1
                
                # Log queue status periodically
                if file_chunks % 500 == 0:
                    logger.debug(f"  📊 Queue: {self.chunk_queue.qsize()}/{self.queue_max_size} | File chunks: {file_chunks}")
            
            self.metrics.files_processed += 1
            logger.info(f"  ✅ File complete: {file_chunks} chunks queued")
            logger.info(f"  📊 Files: {self.metrics.files_processed}/{len(files)}")
        
        logger.info(f"\n🏭 PRODUCER COMPLETE: All files processed")
    
    def _consumer(self, output_file: Path, worker_id: int):
        """
        Consumer: Pull chunks from queue and insert in batches.
        RUNS IN SEPARATE THREAD (one of multiple workers).
        """
        logger.info(f"\n🔨 CONSUMER-{worker_id} START: Waiting for chunks...")
        
        chunk_buffer = []
        
        while True:
            # Get chunk from queue (blocks if empty)
            chunk = self.chunk_queue.get()
            
            # Check for sentinel (end of processing)
            if chunk is None:
                logger.info(f"\n🔨 CONSUMER-{worker_id}: Received end signal")
                break
            
            chunk_buffer.append(chunk)
            
            # Insert when buffer reaches batch size
            if len(chunk_buffer) >= self.insert_batch_size:
                logger.info(f"\n🔄 [Worker-{worker_id}] Buffer full ({len(chunk_buffer)}) - inserting batch...")
                
                # --- TRY/EXCEPT BLOCK FOR FAULT TOLERANCE ---
                try:
                    inserted = self._insert_batch(chunk_buffer)
                    self._append_json_backup(chunk_buffer, output_file, self.metrics.total_chunks > 0)
                    
                    with self.metrics_lock:
                        self.metrics.total_chunks += inserted
                    
                    logger.info(f"💾 [Worker-{worker_id}] Total chunks inserted: {self.metrics.total_chunks}")
                    logger.info(f"📊 Average speed: {self.metrics.total_chunks/(time.time()-self.metrics.start_time):.1f} chunks/sec\n")
                    
                except Exception as e:
                    logger.error(f"!!!!!!!! [Worker-{worker_id}] FAILED TO INSERT BATCH. SKIPPING. !!!!!!!!")
                    logger.error(f"Error: {e}")
                # --- END OF BLOCK ---

                # We clear the buffer whether it succeeded or failed, to move on
                chunk_buffer.clear()  # Free memory
        
        # Insert remaining chunks
        if chunk_buffer:
            logger.info(f"\n🔄 [Worker-{worker_id}] Inserting final batch ({len(chunk_buffer)} chunks)...")
            
            try:
                inserted = self._insert_batch(chunk_buffer)
                self._append_json_backup(chunk_buffer, output_file, self.metrics.total_chunks > 0)
                
                with self.metrics_lock:
                    self.metrics.total_chunks += inserted
                    
            except Exception as e:
                logger.error(f"!!!!!!!! [Worker-{worker_id}] FAILED TO INSERT FINAL BATCH. !!!!!!!!")
                logger.error(f"Error: {e}")
        
        logger.info(f"\n🔨 CONSUMER-{worker_id} COMPLETE")
    
    def process_files(self, input_dir: str, output_dir: str):
        """
        Main processing loop - producer/consumer run IN PARALLEL.
        Consumer runs in separate thread, producer in main thread.
        """
        logger.info("\n" + "="*80)
        logger.info("🚀 STREAMING PIPELINE STARTING")
        logger.info("="*80)
        logger.info(f"Config:")
        logger.info(f"  • Chunk size: {self.config.chunk_size}")
        logger.info(f"  • Chunk overlap: {self.config.chunk_overlap}")
        logger.info(f"  • Insert batch size: {self.insert_batch_size}")
        logger.info(f"  • Queue max size: {self.queue_max_size}")
        logger.info(f"  • Consumer threads: {self.num_consumers}")
        logger.info(f"  • Stream chunk size: {self.stream_chunk_size_mb}MB")
        logger.info(f"  • Max tokens/chunk: {self.max_tokens_per_chunk}")
        logger.info("="*80)
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        output_file = output_path / "processed_chunks.json"
        
        # Discover files
        db_path = input_path / "shortlist.db"
        if db_path.exists():
            logger.info(f"\n📂 Loading from SQLite: {db_path}")
            files = self._get_files_from_sqlite(str(db_path))
        else:
            logger.info(f"\n📂 Scanning directory: {input_path}")
            file_paths = DocumentReader.find_files(input_path)
            files = [(None, path) for path in file_paths]
        
        self.metrics.total_files = len(files)
        logger.info(f"✅ Found {len(files)} files to process\n")
        
        if not files:
            logger.warning("⚠️  No files to process!")
            return
        
        # Initialize JSON backup
        self._init_json_backup(output_file)

        # Start multiple consumer threads
        consumer_threads = []
        logger.info(f"\n🚀 Starting {self.num_consumers} consumer threads...\n")

        for i in range(self.num_consumers):
            consumer_thread = threading.Thread(
                target=self._consumer,
                args=(output_file, i+1),
                name=f"Consumer-{i+1}"
            )
            consumer_thread.start()
            consumer_threads.append(consumer_thread)
            logger.info(f"✅ Consumer thread {i+1} started")

        logger.info(f"\n✅ All {self.num_consumers} consumers running\n")

        # Run producer in main thread
        self._producer(files)

        # Signal all consumers to stop by sending sentinel for each
        for i in range(self.num_consumers):
            self.chunk_queue.put(None)

        # Wait for all consumers to finish
        logger.info(f"\n⏳ Waiting for {self.num_consumers} consumers to finish...")
        for i, thread in enumerate(consumer_threads, 1):
            thread.join()
            logger.info(f"✅ Consumer thread {i} completed")

        logger.info("✅ All consumer threads completed")
        
        # Finalize
        self._finalize_json_backup(output_file)
        
        # Print comprehensive summary
        self._print_summary(output_file)
    
    def _print_summary(self, output_file: Path):
        """Print detailed pipeline summary"""
        summary = self.metrics.get_summary()
        
        logger.info("\n" + "="*80)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        logger.info(f"\n📊 OVERALL STATISTICS:")
        logger.info(f"  • Files processed: {summary['files_processed']}")
        logger.info(f"  • Total chunks: {summary['total_chunks']}")
        logger.info(f"  • Total time: {summary['total_elapsed_min']:.2f} min ({summary['total_elapsed_sec']:.1f}s)")
        logger.info(f"  • Average throughput: {summary['chunks_per_sec']:.1f} chunks/sec")
        
        logger.info(f"\n📦 BATCH STATISTICS:")
        logger.info(f"  • Total batches: {summary['batches']['count']}")
        logger.info(f"  • Average batch size: {summary['batches']['avg_size']:.1f} chunks")
        logger.info(f"  • Average batch time: {summary['batches']['avg_time_sec']:.3f}s")
        logger.info(f"  • Min batch time: {summary['batches']['min_time_sec']:.3f}s")
        logger.info(f"  • Max batch time: {summary['batches']['max_time_sec']:.3f}s")
        logger.info(f"  • Total insert time: {summary['batches']['total_time_sec']:.2f}s")
        logger.info(f"  • Average batch throughput: {summary['batches']['avg_throughput']:.1f} chunks/sec")
        
        logger.info(f"\n⏱️  TIME BREAKDOWN:")
        logger.info(f"  • Reading files: {summary['time_breakdown']['read_sec']:.2f}s ({summary['time_breakdown']['read_pct']:.1f}%)")
        logger.info(f"  • Chunking: {summary['time_breakdown']['chunk_sec']:.2f}s ({summary['time_breakdown']['chunk_pct']:.1f}%)")
        logger.info(f"  • Cleaning: {summary['time_breakdown']['clean_sec']:.2f}s ({summary['time_breakdown']['clean_pct']:.1f}%)")
        logger.info(f"  • Inserting: {summary['time_breakdown']['insert_sec']:.2f}s ({summary['time_breakdown']['insert_pct']:.1f}%)")
        logger.info(f"  • Queue waiting: {summary['time_breakdown']['wait_sec']:.2f}s ({summary['time_breakdown']['wait_pct']:.1f}%)")
        
        logger.info(f"\n💾 OUTPUT:")
        logger.info(f"  • Backup file: {output_file}")
        logger.info(f"  • File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
        
        logger.info("\n" + "="*80)
        
        # Save metrics to JSON
        metrics_file = output_file.parent / "pipeline_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"📈 Detailed metrics saved to: {metrics_file}")
        logger.info("="*80 + "\n")
    
    def _get_files_from_sqlite(self, db_path: str):
        """Load files from SQLite"""
        query = """
        SELECT doc_id, source_path
        FROM documents
        WHERE processing_status='pending'
        ORDER BY CAST(REPLACE(REPLACE(doc_name, 'doc_', ''), '.txt', '') AS INTEGER)
        """
        with sqlite3.connect(db_path) as conn:
            files = []
            for doc_id, source_path in conn.execute(query):
                if os.path.exists(source_path):
                    files.append((doc_id, Path(source_path)))
            return files
    
    def _init_json_backup(self, file_path: Path):
        """Initialize JSON backup"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write('{\n  "rows": [\n')
    
    def _append_json_backup(self, chunks: List[Dict[str, Any]], file_path: Path, needs_comma: bool):
        """Append chunks to JSON"""
        with open(file_path, 'a', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks):
                if needs_comma or i > 0:
                    f.write(',\n')
                
                cleaned_chunk = {
                    "chunk_id": chunk.get("chunk_id"),
                    "doc_id": chunk.get("doc_id"),
                    "doc_name": chunk.get("doc_name"),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "chunk_tokens": chunk.get("chunk_tokens", 0),
                }
                
                chunk_str = json.dumps(cleaned_chunk, ensure_ascii=False)
                f.write('    ' + chunk_str)
    
    def _finalize_json_backup(self, file_path: Path):
        """Finalize JSON"""
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write('\n  ]\n}\n')


def main():
    parser = argparse.ArgumentParser(description="Streaming Pipeline with Bounded Queue and Threading")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for Milvus inserts (REDUCED to 50)")
    parser.add_argument("--queue-size", type=int, default=1500, help="Max chunks in queue")
    parser.add_argument("--stream-mb", type=int, default=10, help="Stream read chunk size in MB")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel consumer threads (start with 2-4)")

    args = parser.parse_args()

    config = ProcessingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )

    pipeline = StreamingPipeline(config, num_consumers=args.workers)   
    pipeline.insert_batch_size = args.batch_size  # Use the batch size from args
    pipeline.queue_max_size = args.queue_size
    pipeline.stream_chunk_size_mb = args.stream_mb
    pipeline.process_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
