import logging
from pymilvus import MilvusClient

# --- CONFIGURATION ---
# Replace these with your actual credentials
AZURE_MILVUS_URI = "http://135.235.255.241:19530"  # From the shared screenshot
AZURE_MILVUS_TOKEN = "SecurePassword123"  # From the shared screenshot (if using password auth)

COLLECTION_NAME = "testCollection2" 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def view_schema():
    try:
        logging.info(f"Connecting to Milvus at {AZURE_MILVUS_URI}...")
        client = MilvusClient(uri=AZURE_MILVUS_URI, token=AZURE_MILVUS_TOKEN)
        
        logging.info(f"Fetching schema for collection: '{COLLECTION_NAME}'")
        info = client.describe_collection(COLLECTION_NAME)
        
        print("\n--- ACTUAL MILVUS SCHEMA ---")
        for field in info['fields']:
            print(f"  - Name: '{field['name']}', Type: {field['type']}")
        print("--------------------------\n")
        
        logging.info("Find the field with type 'FLOAT_VECTOR'. That is your vector field name.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    view_schema()