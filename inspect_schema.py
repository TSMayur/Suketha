# Quick schema inspector using MilvusClient API
from pymilvus import MilvusClient

AZURE_MILVUS_URI = "http://135.235.255.241:19530"
AZURE_MILVUS_TOKEN = "SecurePassword123"
COLLECTION_NAME = "testCollection2"

client = MilvusClient(uri=AZURE_MILVUS_URI, token=AZURE_MILVUS_TOKEN)

# Describe collection
info = client.describe_collection(COLLECTION_NAME)
print("=== COLLECTION SCHEMA ===")
print(f"Collection: {COLLECTION_NAME}")
print(f"Fields:")
for field in info['fields']:
    print(f"  - {field['name']}: {field['type']}")
    if 'dim' in field:
        print(f"    (dimensions: {field['dim']})")

# Get a sample record to see actual data
print("\n=== SAMPLE RECORD ===")
results = client.query(
    collection_name=COLLECTION_NAME,
    filter="",  # No filter, get any record
    output_fields=["*"],  # All fields
    limit=1
)

if results:
    record = results[0]
    print("Fields in actual data:")
    for key, value in record.items():
        if key == 'embedding':  # Don't print huge vector
            print(f"  - {key}: [vector with {len(value)} dimensions]")
        elif isinstance(value, str) and len(value) > 100:
            print(f"  - {key}: {value[:100]}...")
        else:
            print(f"  - {key}: {value}")
else:
    print("No records found!")