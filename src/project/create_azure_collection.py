#!/usr/bin/env python3
# Create NEW collection on Azure Milvus instance

from pymilvus import MilvusClient, DataType
import sys

# Azure Configuration
MILVUS_URI = "http://135.235.255.241:19530"
MILVUS_TOKEN = "SecurePassword123"
COLLECTION_NAME = "rag_chunks_new"  # NEW collection name to avoid conflicts

def create_azure_collection():
    """Create NEW collection on Azure Milvus with proper schema"""
    
    print(f"Connecting to Azure Milvus at {MILVUS_URI}...")
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    
    # List existing collections
    existing = client.list_collections()
    print(f"\nExisting collections: {existing}")
    
    # Check if collection already exists
    if client.has_collection(COLLECTION_NAME):
        print(f"\nCollection '{COLLECTION_NAME}' already exists.")
        response = input("Drop and recreate? (yes/no): ")
        if response.lower() != 'yes':
            print("Keeping existing collection. Exiting.")
            return
        print(f"Dropping existing collection '{COLLECTION_NAME}'...")
        client.drop_collection(COLLECTION_NAME)
    
    print(f"\nCreating collection '{COLLECTION_NAME}'...")
    
    # Define schema matching your data structure
    schema = MilvusClient.create_schema(
        auto_id=False, 
        enable_dynamic_field=True
    )
    
    # Add all fields from your chunks
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
    
    # Vector field - 768 dimensions for all-mpnet-base-v2
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=768)
    
    # Create index for fast similarity search
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",  # Fast and accurate
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )
    
    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong"
    )
    
    print(f"Collection '{COLLECTION_NAME}' created successfully on Azure!")
    
    # Verify
    schema_info = client.describe_collection(COLLECTION_NAME)
    print(f"\nCollection schema fields:")
    for field in schema_info['fields']:
        print(f"  - {field['name']}: {field['type']}")
    
    # Load into memory
    from pymilvus import Collection
    collection = Collection(COLLECTION_NAME)
    collection.load()
    print(f"\nCollection loaded into memory and ready!")
    
    print(f"\nNext steps:")
    print(f"1. Upload your prepared JSON to Azure MinIO")
    print(f"2. Run bulk import to load chunks")
    print(f"3. Query with process_all_queries.py")


if __name__ == "__main__":
    create_azure_collection()
#!/usr/bin/env python3
# Create collection on Azure Milvus instance

from pymilvus import MilvusClient, DataType

# Azure Configuration
MILVUS_URI = "http://135.235.255.241:19530"
MILVUS_TOKEN = "SecurePassword123"
COLLECTION_NAME = "testCollection2"

def create_azure_collection():
    """Create collection on Azure Milvus with proper schema"""
    
    print(f"Connecting to Azure Milvus at {MILVUS_URI}...")
    client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
    
    # Check if collection already exists
    if client.has_collection(COLLECTION_NAME):
        response = input(f"\n⚠️  Collection '{COLLECTION_NAME}' already exists. Drop and recreate? (yes/no): ")
        if response.lower() == 'yes':
            print(f"Dropping existing collection '{COLLECTION_NAME}'...")
            client.drop_collection(COLLECTION_NAME)
        else:
            print("Keeping existing collection. Exiting.")
            return
    
    print(f"\nCreating collection '{COLLECTION_NAME}'...")
    
    # Define schema matching your data structure
    schema = MilvusClient.create_schema(
        auto_id=False, 
        enable_dynamic_field=True
    )
    
    # Add all fields from your chunks
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
    
    # Vector field - 768 dimensions for all-mpnet-base-v2
    schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=768)
    
    # Create index for fast similarity search
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="HNSW",  # Fast and accurate
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )
    
    # Create collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        schema=schema,
        index_params=index_params,
        consistency_level="Strong"
    )
    
    print(f"✅ Collection '{COLLECTION_NAME}' created successfully!")
    
    # Verify
    schema_info = client.describe_collection(COLLECTION_NAME)
    print(f"\nCollection schema fields:")
    for field in schema_info['fields']:
        print(f"  - {field['name']}: {field['type']}")
    
    print(f"\n✅ Ready to ingest data into Azure Milvus!")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Vector field: embedding (768 dimensions)")
    print(f"Index: HNSW with COSINE metric")


if __name__ == "__main__":
    create_azure_collection()