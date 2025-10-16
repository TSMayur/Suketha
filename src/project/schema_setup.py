# src/project/schema_setup_hybrid.py
"""
Create Milvus collection with BOTH dense and sparse vectors for hybrid search.
Dense: sentence-transformers embeddings (768D)
Sparse: BM25 vectors (built-in Milvus BM25)
"""

from pymilvus import MilvusClient, DataType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "rag_chunks_hybrid"

def create_hybrid_collection():
    """Create collection with both dense and sparse vector fields"""
    
    client = MilvusClient(uri="http://localhost:19530")
    
    # Drop existing collection if it exists
    if client.has_collection(COLLECTION_NAME):
        logger.info(f"Dropping existing collection: {COLLECTION_NAME}")
        client.drop_collection(COLLECTION_NAME)
    
    logger.info(f"Creating hybrid collection: {COLLECTION_NAME}")
    
    # Define schema with BOTH vector types
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True
    )
    
    # === Metadata Fields ===
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=255, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
    schema.add_field("doc_name", DataType.VARCHAR, max_length=255)
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
    
    # === Dense Vector Field (from sentence-transformers) ===
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=768  # all-mpnet-base-v2 dimension
    )
    
    # === Sparse Vector Field (BM25 - Milvus will generate this) ===
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR  # Special type for sparse vectors
    )
    
    # === Create Indexes ===
    index_params = client.prepare_index_params()
    
    # Index for dense vector (semantic search)
    index_params.add_index(
        field_name="dense_vector",
        index_type="HNSW",
        metric_type="COSINE",
        params={
            "M": 16,
            "efConstruction": 200
        }
    )
    
    # Index for sparse vector (BM25 search)
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",  # Optimized for sparse vectors
        metric_type="IP",  # BM25 similarity metric
        params={}
    )
    
    # Create the collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        consistency_level="Bounded"
    )
    
    logger.info("✅ Hybrid collection created successfully!")
    logger.info(f"   - Dense vectors: 768D FLOAT_VECTOR with HNSW index")
    logger.info(f"   - Sparse vectors: SPARSE_FLOAT_VECTOR with BM25 index")
    
    return client


def verify_collection():
    """Verify the collection was created correctly"""
    client = MilvusClient(uri="http://localhost:19530")
    
    if not client.has_collection(COLLECTION_NAME):
        logger.error(f"❌ Collection '{COLLECTION_NAME}' not found!")
        return False
    
    schema_info = client.describe_collection(COLLECTION_NAME)
    
    logger.info("\n📋 Collection Schema:")
    for field in schema_info['fields']:
        field_name = field['name']
        field_type = field['type']
        logger.info(f"   - {field_name}: {field_type}")
    
    logger.info("\n✅ Collection verified successfully!")
    return True


if __name__ == "__main__":
    # Create collection
    create_hybrid_collection()
    
    # Verify it worked
    verify_collection()