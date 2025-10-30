from dotenv import load_dotenv
import os
from pymilvus import MilvusClient
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# --- CONFIGURATION ---
# Replace these with your actual credentials
AZURE_MILVUS_URI = os.getenv("AZURE_MILVUS_URI")  # e.g., "milvus-xyz123.milvus.azure.com:19530"
AZURE_MILVUS_TOKEN = os.getenv("AZURE_MILVUS_TOKEN")  # e.g., "your_milvus_token_here"
COLLECTION_NAME = os.getenv("COLLECTION_NAME")  # e.g., "my_collection"

TEI_ENDPOINT = os.getenv("TEI_ENDPOINT")  # e.g., "http://

logging.info(f"Connecting to Milvus at {AZURE_MILVUS_URI}...")
client = MilvusClient(uri=AZURE_MILVUS_URI, token=AZURE_MILVUS_TOKEN)
        
