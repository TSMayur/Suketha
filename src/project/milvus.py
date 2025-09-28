# src/project/milvus_optimized.py

import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any
from project.pydantic_models import Chunk, Document, SearchResult
import logging

logger = logging.getLogger(__name__)

class OptimizedMilvusVectorStore:
    _vectorstore = None
    _embeddings = None
    _connected = False
    _executor = ThreadPoolExecutor(max_workers=2)

    @classmethod
    def _wait_for_milvus(cls, max_retries=10, delay=2):
        """Faster Milvus connection with shorter timeouts"""
        logger.info("Connecting to Milvus...")
        for attempt in range(max_retries):
            try:
                from pymilvus import connections
                connections.connect("default", host="localhost", port="19530", timeout=5)
                if connections.get_connection_addr("default"):
                    logger.info("Milvus connected successfully")
                    return True
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                    time.sleep(delay)
                else:
                    raise ConnectionError(f"Milvus connection failed after {max_retries} attempts: {e}")
        return False

    @classmethod
    def get_client(cls):
        if cls._embeddings is None:
            logger.info("Loading embeddings for Milvus client...")
            cls._embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'mps' if hasattr(torch.backends, 'mps') else 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        if not cls._connected:
            cls._wait_for_milvus()
            cls._connected = True
        return True

    @classmethod
    def setup_schema(cls, class_name: str = "rag_chunks"):
        cls.get_client()
        connection_args = {"uri": "http://localhost:19530"}
        try:
            cls._vectorstore = Milvus(
                embedding_function=cls._embeddings,
                collection_name=class_name,
                connection_args=connection_args,
                consistency_level="Eventually",
                drop_old=False,
                vector_field="embedding_vector",
                text_field="chunk_text",
                enable_dynamic_field=True,
                index_params={
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
            )
            logger.info(f"Collection '{class_name}' ready")
        except Exception as e:
            logger.error(f"Error setting up Milvus schema: {e}")
            raise e

    @classmethod
    async def insert_chunks_async(cls, chunk_dicts: List[Dict], class_name: str = "rag_chunks"):
        """Optimized async insertion with larger batches and corrected metadata."""
        if cls._vectorstore is None:
            cls.setup_schema(class_name)
        
        from langchain_core.documents import Document as LangChainDoc
        
        logger.info(f"Preparing {len(chunk_dicts)} chunks for insertion...")
        
        docs = []
        ids = []
        for chunk in chunk_dicts:
            # CORRECTED METADATA: Include all fields required by the schema
            metadata = {
                "chunk_id": chunk.get("chunk_id"),
                "doc_id": chunk.get("doc_id"),
                "chunk_index": chunk.get("chunk_index", 0),
                "chunk_text": chunk.get("chunk_text", ""), # Add text for potential filtering
                "chunk_size": chunk.get("chunk_size", 0),
                "chunk_tokens": chunk.get("chunk_tokens"),
                "chunk_method": chunk.get("chunk_method"),
                "chunk_overlap": chunk.get("chunk_overlap"),
                "domain": chunk.get("domain"),
                "content_type": chunk.get("content_type"),
                "embedding_model": chunk.get("embedding_model"),
                "created_at": chunk.get("created_at")
            }
            
            doc = LangChainDoc(
                page_content=chunk['chunk_text'],
                metadata=metadata
            )
            docs.append(doc)
            ids.append(chunk.get("chunk_id"))

        batch_size = 100
        total_batches = (len(docs) + batch_size - 1) // batch_size
        
        async def insert_batch(batch_docs, batch_ids, batch_num):
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(
                    cls._executor,
                    lambda: cls._vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
                )
                logger.info(f"Inserted batch {batch_num}/{total_batches}: {len(batch_docs)} chunks")
                return len(batch_docs)
            except Exception as e:
                logger.error(f"Error inserting batch {batch_num}: {e}")
                return 0

        semaphore = asyncio.Semaphore(2)
        
        async def bounded_insert(batch_docs, batch_ids, batch_num):
            async with semaphore:
                return await insert_batch(batch_docs, batch_ids, batch_num)

        tasks = []
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            task = bounded_insert(batch_docs, batch_ids, batch_num)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_inserts = sum(r for r in results if isinstance(r, int))
        logger.info(f"Completed insertion: {successful_inserts}/{len(docs)} chunks successful")

    # ... (other methods remain the same) ...

    @classmethod
    def insert_chunks_sync(cls, chunk_dicts: List[Dict], class_name: str = "rag_chunks"):
        """Synchronous version for compatibility"""
        asyncio.run(cls.insert_chunks_async(chunk_dicts, class_name))

    # Keep existing methods for backward compatibility
    @classmethod
    def search_by_text(cls, query_text: str, limit: int = 5) -> List[SearchResult]:
        if cls._vectorstore is None:
            cls.setup_schema()
        try:
            results_with_scores = cls._vectorstore.similarity_search_with_relevance_scores(
                query=query_text,
                k=limit
            )

            search_results = []
            for rank, (doc, score) in enumerate(results_with_scores, 1):
                similarity_score = float(score)
                distance = 1.0 - similarity_score
                
                # Create chunk from search result
                chunk = Chunk(
                    id=doc.metadata.get("chunk_id", f"chunk_{rank}"),
                    doc_id=doc.metadata.get("doc_id", ""),
                    chunk_text=doc.page_content,
                    chunk_index=doc.metadata.get("chunk_index", 0),
                    chunk_method="unknown",  # Add default
                    chunk_size=len(doc.page_content),
                    chunk_overlap=0,  # Add default
                    content_type=doc.metadata.get("content_type", "text")
                )
                
                search_result = SearchResult(
                    chunk=chunk,
                    similarity_score=similarity_score,
                    distance=distance,
                    rank=rank
                )
                search_results.append(search_result)
            return search_results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    @classmethod
    def get_stats(cls, class_name: str = "rag_chunks") -> Dict[str, Any]:
        try:
            cls.get_client()
            from pymilvus import Collection, utility
            if utility.has_collection(class_name):
                collection = Collection(class_name)
                collection.flush()
                return {
                    "total_chunks": collection.num_entities,
                    "status": "ready"
                }
            else:
                return {"total_chunks": 0, "status": "no_collection"}
        except Exception as e:
            return {"error": str(e), "status": "error", "total_chunks": 0}

    @classmethod
    def clear_all_data(cls, class_name: str = "rag_chunks"):
        try:
            logger.info("Clearing database...")
            from pymilvus import utility
            cls.get_client()
            if utility.has_collection(class_name):
                utility.drop_collection(class_name)
            cls._vectorstore = None
            cls.setup_schema(class_name)
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Clear error: {e}")

# Alias for backward compatibility
MilvusVectorStore = OptimizedMilvusVectorStore