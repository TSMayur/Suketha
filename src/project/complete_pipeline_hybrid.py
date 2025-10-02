
import argparse
import asyncio
import json
import logging
import time
import os
import gc
from pathlib import Path
from typing import List, Dict, Any
import multiprocessing as mp
from multiprocessing import Pool

from project.pydantic_models import ProcessingConfig, EmbeddingModel
from project.doc_reader import DocumentReader
from project.chunker import OptimizedChunkingService
from project.milvus_bulk_import import EnhancedMilvusBulkImporter
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model - loaded once per worker
_MODEL = None

def _init_worker():
    """Initialize model ONCE when worker starts"""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(
            EmbeddingModel.ALL_MPNET_BASE_V2.value,
            device='cpu'
        )
        logger.info(f"Worker {mp.current_process().name} initialized")

def _embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed using pre-loaded global model"""
    global _MODEL
    return _MODEL.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()


class FixedAsyncPipeline:
    """Fixed pipeline with proper worker pool management"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.workers = max(4, mp.cpu_count() - 2)  # More workers
        
        # Create pool ONCE and reuse it
        self.pool = Pool(
            processes=self.workers,
            initializer=_init_worker
        )
        
        logger.info(f"Pipeline: {self.workers} workers, models pre-loaded")

    async def _embed_async(self, chunks: List[Dict]) -> List[Dict]:
        """Async embedding with persistent pool"""
        if not chunks:
            return []
        
        texts = [c['chunk_text'] for c in chunks]
        start = time.time()
        
        # Larger batches - 50 chunks per worker task
        batch_size = 50
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        # Submit to pool asynchronously
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(None, self.pool.apply, _embed_batch, (batch,))
            for batch in batches
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Flatten
        embeddings = [e for batch in results for e in batch]
        
        # Update chunks
        for c, emb in zip(chunks, embeddings):
            c['embedding'] = emb
            c['embedding_model'] = EmbeddingModel.ALL_MPNET_BASE_V2.value
        
        speed = len(texts) / (time.time() - start)
        logger.info(f"Embedded {len(texts)} in {time.time()-start:.2f}s ({speed:.1f}/s)")
        
        del texts, batches, embeddings, results
        gc.collect()
        
        return chunks

    async def _write_async(self, chunks: List[Dict], path: Path, first: bool):
        """Async JSON write"""
        if not chunks:
            return
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._write_sync, chunks, path, first)

    def _write_sync(self, chunks: List[Dict], path: Path, first: bool):
        """Synchronous write"""
        with open(path, 'a', encoding='utf-8') as f:
            for i, c in enumerate(chunks):
                if not first or i > 0:
                    f.write(',\n')
                
                data = {
                    "chunk_id": c.get("chunk_id", f"c_{i}"),
                    "doc_id": c.get("doc_id", ""),
                    "chunk_index": c.get("chunk_index", i),
                    "chunk_text": c.get("chunk_text", ""),
                    "chunk_size": c.get("chunk_size", 0),
                    "chunk_tokens": c.get("chunk_tokens", 0),
                    "chunk_method": c.get("chunk_method", "recursive"),
                    "chunk_overlap": c.get("chunk_overlap", 0),
                    "domain": c.get("domain", "general"),
                    "content_type": c.get("content_type", "text"),
                    "embedding_model": c.get("embedding_model", ""),
                    "created_at": c.get("created_at", ""),
                    "embedding": c.get("embedding", [])
                }
                
                if not data["embedding"]:
                    continue
                
                f.write('    ' + json.dumps(data, ensure_ascii=False))

    async def _process_file_async(self, fp: Path) -> List[Dict]:
        """Async file processing"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._process_sync, fp)

    def _process_sync(self, fp: Path) -> List[Dict]:
        """Sync processing"""
        try:
            doc = DocumentReader.read_file(fp)
            if not doc:
                return []
            chunks = OptimizedChunkingService.chunk_document(doc, self.config)
            return [c.model_dump(by_alias=False) for c in chunks]
        except Exception as e:
            logger.error(f"Error: {e}")
            return []

    async def run_async(self, input_dir: str, output_dir: str, bulk_import: bool = True):
        """Main async pipeline"""
        start = time.time()
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        output_file = output_path / "prepared_data.json"
        
        # Init file
        if output_file.exists():
            output_file.unlink()
        with open(output_file, 'w') as f:
            f.write('{\n  "rows": [\n')
        
        logger.info("=== FIXED ASYNC PIPELINE ===")
        
        files = DocumentReader.find_files(Path(input_dir))
        logger.info(f"Files: {len(files)}")
        
        if not files:
            with open(output_file, 'a') as f:
                f.write('\n  ]\n}\n')
            return
        
        total = 0
        first_write = True
        
        for i, fp in enumerate(files):
            logger.info(f"File {i+1}/{len(files)}: {fp.name}")
            
            try:
                # Process file
                chunk_dicts = await self._process_file_async(fp)
                
                if not chunk_dicts:
                    continue
                
                logger.info(f"Chunks: {len(chunk_dicts)}")
                
                # Process in 300-chunk batches (larger!)
                batch_size = 300
                for j in range(0, len(chunk_dicts), batch_size):
                    batch = chunk_dicts[j:j+batch_size]
                    batch_num = j // batch_size + 1
                    logger.info(f"Batch {batch_num}: {len(batch)} chunks")
                    
                    # Embed
                    embedded = await self._embed_async(batch)
                    
                    # Write
                    await self._write_async(embedded, output_file, first_write)
                    first_write = False
                    total += len(embedded)
                    
                    del batch, embedded
                    gc.collect()
                    await asyncio.sleep(0)
                
                del chunk_dicts
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error: {e}")
                continue
        
        # Finalize
        with open(output_file, 'a') as f:
            f.write('\n  ]\n}\n')
        
        logger.info(f"Total: {total} chunks")
        
        # Import
        if bulk_import and total > 0:
            try:
                logger.info("Importing...")
                imp = EnhancedMilvusBulkImporter()
                obj = imp.upload_to_minio(output_file)
                imp.run_bulk_import("rag_chunks", obj)
            except Exception as e:
                logger.error(f"Import failed: {e}")
        
        elapsed = time.time() - start
        logger.info(f"=== DONE in {elapsed:.2f}s ===")
        logger.info(f"Speed: {total/elapsed:.1f} chunks/sec")
        
        # Cleanup
        self.pool.close()
        self.pool.join()

    def __del__(self):
        """Cleanup pool on deletion"""
        if hasattr(self, 'pool'):
            self.pool.close()
            self.pool.join()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=256)
    parser.add_argument("--no-bulk-import", action="store_true")
    
    args = parser.parse_args()
    
    config = ProcessingConfig(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    pipeline = FixedAsyncPipeline(config)
    await pipeline.run_async(
        args.input_dir,
        args.output_dir,
        not args.no_bulk_import
    )


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    asyncio.run(main())