# src/project/schema_setup.py
"""
Create Milvus collection with BOTH dense and sparse vectors for hybrid search.
Dense: TEI embeddings (768D)
Sparse: BM25 vectors (built-in Milvus BM25)
"""

from pymilvus import MilvusClient, DataType, Function, FunctionType
from .config import client, COLLECTION_NAME, TEI_ENDPOINT
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_hybrid_collection():
    """Create collection with both dense and sparse vector fields"""
    
    # Drop existing collection if it exists
    if client.has_collection(COLLECTION_NAME):
        logger.info(f"Dropping existing collection: {COLLECTION_NAME}")
        client.drop_collection(COLLECTION_NAME)
    
    logger.info(f"Creating hybrid collection: {COLLECTION_NAME}")
    
    # Define schema with BOTH vector types
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=False  # Disable to avoid JSON index issues
    )
    
    # === Metadata Fields ===
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=255, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
    schema.add_field("doc_name", DataType.VARCHAR, max_length=255)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("chunk_text", DataType.VARCHAR, max_length=65535, enable_analyzer=True)
    schema.add_field("chunk_size", DataType.INT64)
    schema.add_field("chunk_tokens", DataType.INT64)
    schema.add_field("chunk_method", DataType.VARCHAR, max_length=50)
    schema.add_field("chunk_overlap", DataType.INT64)
    schema.add_field("domain", DataType.VARCHAR, max_length=100)
    schema.add_field("content_type", DataType.VARCHAR, max_length=50)
    schema.add_field("embedding_model", DataType.VARCHAR, max_length=200)
    schema.add_field("created_at", DataType.VARCHAR, max_length=50)
    
    # === Dense Vector Field (from TEI) ===
    schema.add_field(
        field_name="dense_vector",
        datatype=DataType.FLOAT_VECTOR,
        dim=768  # all-mpnet-base-v2 dimension
    )
    
    # === Sparse Vector Field (BM25 - Milvus auto-generated) ===
    schema.add_field(
        field_name="sparse_vector",
        datatype=DataType.SPARSE_FLOAT_VECTOR
    )
    
    # === TEI Function (Dense Embeddings) ===
    tei_function = Function(
        name="tei_func",
        function_type=FunctionType.TEXTEMBEDDING,
        input_field_names=["chunk_text"],
        output_field_names=["dense_vector"],
        params={
            "provider": "TEI",
            "endpoint": TEI_ENDPOINT,
        }
    )
    
    # === BM25 Function (Sparse Embeddings) ===
    bm25_function = Function(
        name="text_bm25_emb",
        input_field_names=["chunk_text"],
        output_field_names=["sparse_vector"],
        function_type=FunctionType.BM25,
    )
    
    schema.add_function(tei_function)
    schema.add_function(bm25_function)
    
    # === Index Parameters ===
    index_params = client.prepare_index_params()
    
    # Primary key index
    index_params.add_index(
        field_name="chunk_id",
        index_type="AUTOINDEX"
    )
    
    # Dense vector index
    index_params.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )
    
    # Sparse vector index (BM25)
    index_params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75
        }
    )
    
    # Create the collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        consistency_level="Bounded"
    )
    
    logger.info("✅ Hybrid collection created successfully!")
    
    return client


def verify_collection():
    """Verify collection was created correctly"""
    
    if not client.has_collection(COLLECTION_NAME):
        logger.error(f"❌ Collection '{COLLECTION_NAME}' not found!")
        return False
    
    schema_info = client.describe_collection(COLLECTION_NAME)
    
    logger.info("\n📋 Collection Schema:")
    for field in schema_info['fields']:
        field_name = field['name']
        field_type = field['type']
        logger.info(f"   - {field_name}: {field_type}")
    
    # Check functions
    logger.info("\n🔧 Configured Functions:")
    if 'functions' in schema_info:
        for func in schema_info['functions']:
            logger.info(f"   - {func.get('name', 'Unknown')}: {func.get('type', 'Unknown')}")
    
    logger.info("\n✅ Collection verified successfully!")
    return True


if __name__ == "__main__":
    # Create collection
    create_hybrid_collection()
    
    # Verify it worked
    verify_collection()
