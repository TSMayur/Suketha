from pymilvus import MilvusClient, DataType

COLLECTION_NAME = "rag_chunks_test1"

def create_collection():
    client = MilvusClient(uri="http://localhost:19530")
    
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    
    # Define a schema that PERFECTLY matches the pipeline's output
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
    
    # --- Fields that are ACTUALLY CREATED by the pipeline ---
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=255, is_primary=True)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
    schema.add_field("doc_name", DataType.VARCHAR, max_length=255) # Added this line
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
    # The vector field is named 'embedding' in the pipeline's output
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=768)
    
    index_params = client.prepare_index_params()
    index_params.add_index(
        # Index the 'embedding' field
        field_name="embedding",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        consistency_level="Bounded"
    )
    
    print(f"Milvus collection '{COLLECTION_NAME}' created successfully!")

if __name__ == "__main__":
    create_collection()