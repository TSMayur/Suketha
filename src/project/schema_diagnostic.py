# src/project/schema_diagnostic.py

import json
import logging
from pathlib import Path
from pymilvus import utility, connections, Collection

# --- CONFIGURATION ---
COLLECTION_NAME = "rag_chunks"
JSON_FILE_PATH = "prepared_for_upload/prepared_data.json"
MILVUS_ALIAS = "diagnostic_connection"
# ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def diagnose_schema_mismatch():
    """
    Connects to Milvus, reads the collection schema, and compares it against
    the fields in the prepared JSON file to find discrepancies.
    """
    # --- 1. Get Schema from Milvus ---
    try:
        # **THE FIX IS HERE**: Explicitly connect before using the Collection object
        connections.connect(MILVUS_ALIAS, host="localhost", port="19530")
        
        if not utility.has_collection(COLLECTION_NAME, using=MILVUS_ALIAS):
            logging.error(f"Milvus collection '{COLLECTION_NAME}' does not exist. Please run schema_setup.py first.")
            connections.disconnect(MILVUS_ALIAS)
            return

        collection = Collection(COLLECTION_NAME, using=MILVUS_ALIAS)
        milvus_schema = collection.schema
        milvus_fields = {field.name for field in milvus_schema.fields}
        logging.info(f"Successfully fetched schema for '{COLLECTION_NAME}' from Milvus.")
        print("\n--- Milvus Schema Fields ---")
        for field in sorted(list(milvus_fields)):
            print(f"- {field}")

    except Exception as e:
        logging.error(f"Failed to connect to Milvus or fetch schema: {e}")
        return
    finally:
        if MILVUS_ALIAS in connections.list_connections():
            connections.disconnect(MILVUS_ALIAS)

    # --- 2. Get Fields from JSON data ---
    try:
        file_path = Path(JSON_FILE_PATH)
        if not file_path.exists():
            logging.error(f"JSON file not found at: '{file_path}'. Please run the pipeline first.")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle case where 'rows' might be empty
            first_record = data.get("rows", [{}])[0] if data.get("rows") else {}
            json_fields = set(first_record.keys())
        
        logging.info(f"Successfully loaded first record from '{file_path}'.")
        print("\n--- JSON Data Fields (from first record) ---")
        for field in sorted(list(json_fields)):
            print(f"- {field}")

    except Exception as e:
        logging.error(f"Failed to read or parse JSON file: {e}")
        return

    # --- 3. Compare and Report ---
    print("\n--- DIAGNOSTIC REPORT ---")
    
    missing_in_json = milvus_fields - json_fields
    if missing_in_json:
        print(f"\n❌ ERROR: Fields required by Milvus schema but MISSING in your JSON data:")
        for field in sorted(list(missing_in_json)):
            print(f"  - {field}")

    extra_in_json = json_fields - milvus_fields
    if extra_in_json:
        print(f"\n⚠️ WARNING: Fields found in your JSON data but NOT DEFINED in your Milvus schema:")
        for field in sorted(list(extra_in_json)):
            print(f"  - {field}")

    if not missing_in_json and not extra_in_json:
        print("\n✅ SUCCESS: All fields match perfectly between your JSON data and the Milvus schema.")
    
    print("\n---------------------------\n")


if __name__ == "__main__":
    diagnose_schema_mismatch()