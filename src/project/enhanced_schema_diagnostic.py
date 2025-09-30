# src/project/enhanced_schema_diagnostic.py

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
    Enhanced diagnostic that shows actual field types and sample data structure
    """
    # --- 1. Get Schema from Milvus ---
    try:
        connections.connect(MILVUS_ALIAS, host="localhost", port="19530")
        
        if not utility.has_collection(COLLECTION_NAME, using=MILVUS_ALIAS):
            logging.error(f"Milvus collection '{COLLECTION_NAME}' does not exist. Please run schema_setup.py first.")
            connections.disconnect(MILVUS_ALIAS)
            return

        collection = Collection(COLLECTION_NAME, using=MILVUS_ALIAS)
        milvus_schema = collection.schema
        milvus_fields = {}
        
        print("\n--- Milvus Schema Fields (with types) ---")
        for field in milvus_schema.fields:
            milvus_fields[field.name] = str(field.dtype)
            print(f"- {field.name}: {field.dtype}")

        logging.info(f"Successfully fetched schema for '{COLLECTION_NAME}' from Milvus.")

    except Exception as e:
        logging.error(f"Failed to connect to Milvus or fetch schema: {e}")
        return
    finally:
        if MILVUS_ALIAS in connections.list_connections():
            connections.disconnect(MILVUS_ALIAS)

    # --- 2. Analyze JSON data structure ---
    try:
        file_path = Path(JSON_FILE_PATH)
        if not file_path.exists():
            logging.error(f"JSON file not found at: '{file_path}'. Please run the pipeline first.")
            return

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"\n--- JSON File Structure Analysis ---")
        print(f"Root keys in JSON: {list(data.keys())}")
        
        if "rows" in data:
            rows = data["rows"]
            print(f"Number of rows: {len(rows)}")
            
            if len(rows) > 0:
                first_record = rows[0]
                json_fields = set(first_record.keys())
                
                print(f"\n--- JSON Data Fields (from first record) ---")
                for field in sorted(list(json_fields)):
                    value = first_record[field]
                    value_type = type(value).__name__
                    
                    # Show sample values for key fields
                    if isinstance(value, list) and len(value) > 0:
                        print(f"- {field}: {value_type} (length: {len(value)}, first element type: {type(value[0]).__name__})")
                    elif isinstance(value, str) and len(value) > 50:
                        print(f"- {field}: {value_type} ('{value[:50]}...')")
                    else:
                        print(f"- {field}: {value_type} ({value})")
                        
                # --- 3. Compare and Report ---
                print("\n--- DETAILED DIAGNOSTIC REPORT ---")
                
                milvus_field_names = set(milvus_fields.keys())
                missing_in_json = milvus_field_names - json_fields
                extra_in_json = json_fields - milvus_field_names
                
                if missing_in_json:
                    print(f"\n❌ ERROR: Fields required by Milvus schema but MISSING in your JSON data:")
                    for field in sorted(list(missing_in_json)):
                        expected_type = milvus_fields[field]
                        print(f"  - {field} (expected type: {expected_type})")

                if extra_in_json:
                    print(f"\n⚠️ WARNING: Fields found in your JSON data but NOT DEFINED in your Milvus schema:")
                    for field in sorted(list(extra_in_json)):
                        value_type = type(first_record[field]).__name__
                        print(f"  - {field} (actual type: {value_type})")

                # Check for vector field issues specifically
                vector_fields = [k for k, v in milvus_fields.items() if 'VECTOR' in v]
                print(f"\n--- Vector Field Analysis ---")
                for vec_field in vector_fields:
                    if vec_field in json_fields:
                        vec_data = first_record[vec_field]
                        if isinstance(vec_data, list):
                            print(f"✅ {vec_field}: Found list with {len(vec_data)} dimensions")
                            if len(vec_data) > 0:
                                print(f"   First few values: {vec_data[:5]}")
                        else:
                            print(f"❌ {vec_field}: Expected list, found {type(vec_data).__name__}")
                    else:
                        print(f"❌ {vec_field}: Missing from JSON data")

                if not missing_in_json and not extra_in_json:
                    print("\n✅ SUCCESS: All fields match perfectly between your JSON data and the Milvus schema.")
                    
                # Show sample record structure
                print(f"\n--- Sample Record Structure ---")
                print(json.dumps(first_record, indent=2)[:500] + "..." if len(json.dumps(first_record, indent=2)) > 500 else json.dumps(first_record, indent=2))
                
            else:
                print("❌ No rows found in the JSON data!")
        else:
            print("❌ No 'rows' key found in JSON data structure!")
            print(f"Available keys: {list(data.keys())}")
            
    except Exception as e:
        logging.error(f"Failed to read or parse JSON file: {e}")
        return

    print("\n---------------------------\n")


if __name__ == "__main__":
    diagnose_schema_mismatch()