# src/project/milvus_bulk_import.py

import logging
import json
from pathlib import Path
from minio import Minio
from pymilvus import utility, connections, Collection
import time 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- MinIO/S3 Connection Details ---
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"
BUCKET_NAME = "milvus-bulk-import"

# --- Milvus Connection Details ---
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

class EnhancedMilvusBulkImporter:
    def __init__(self):
        self.minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        logging.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    def validate_json_structure(self, file_path: Path, collection_name: str):
        """Validate JSON structure before upload"""
        logging.info("🔍 Validating JSON structure before upload...")
        
        try:
            # Get Milvus schema
            if not utility.has_collection(collection_name):
                logging.error(f"❌ Collection '{collection_name}' does not exist!")
                return False
                
            collection = Collection(collection_name)
            schema = collection.schema
            expected_fields = {field.name: str(field.dtype) for field in schema.fields}
            
            logging.info(f"📋 Expected Milvus fields:")
            for field, dtype in expected_fields.items():
                logging.info(f"   - {field}: {dtype}")
            
            # Read and validate JSON
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            logging.info(f"📄 JSON root structure: {list(data.keys())}")
            
            if "rows" not in data:
                logging.error(f"❌ JSON missing 'rows' key. Found keys: {list(data.keys())}")
                return False
                
            rows = data["rows"]
            if not rows:
                logging.error("❌ No data rows found in JSON")
                return False
                
            logging.info(f"✅ Found {len(rows)} data rows")
            
            # Validate first record
            first_record = rows[0]
            actual_fields = set(first_record.keys())
            expected_field_names = set(expected_fields.keys())
            
            logging.info(f"📝 First record fields:")
            for field in sorted(actual_fields):
                value = first_record[field]
                if isinstance(value, list):
                    logging.info(f"   - {field}: list[{len(value)} elements]")
                    if len(value) > 0:
                        logging.info(f"     First element type: {type(value[0]).__name__}")
                elif isinstance(value, str) and len(value) > 50:
                    logging.info(f"   - {field}: string ('{value[:50]}...')")
                else:
                    logging.info(f"   - {field}: {type(value).__name__} = {value}")
            
            # Check mismatches
            missing_fields = expected_field_names - actual_fields
            extra_fields = actual_fields - expected_field_names
            
            if missing_fields:
                logging.error(f"❌ Missing required fields: {missing_fields}")
                return False
                
            if extra_fields:
                logging.warning(f"⚠️  Extra fields (will be ignored): {extra_fields}")
            
            # Validate vector field specifically
            vector_fields = [name for name, dtype in expected_fields.items() if 'VECTOR' in dtype]
            for vec_field in vector_fields:
                if vec_field in first_record:
                    vec_data = first_record[vec_field]
                    if not isinstance(vec_data, list):
                        logging.error(f"❌ Vector field '{vec_field}' should be list, got {type(vec_data).__name__}")
                        return False
                    if not vec_data:
                        logging.error(f"❌ Vector field '{vec_field}' is empty")
                        return False
                    logging.info(f"✅ Vector field '{vec_field}': {len(vec_data)} dimensions")
                    
            logging.info("✅ JSON structure validation passed!")
            return True
            
        except Exception as e:
            logging.error(f"❌ JSON validation failed: {e}")
            return False

    def upload_to_minio(self, file_path: Path) -> str:
        """Uploads a single prepared data file to the MinIO bucket."""
        try:
            if not self.minio_client.bucket_exists(BUCKET_NAME):
                self.minio_client.make_bucket(BUCKET_NAME)
                logging.info(f"Created MinIO bucket '{BUCKET_NAME}'")
            
            object_name = file_path.name
            logging.info(f"📤 Uploading {file_path.name} to MinIO...")
            
            # Get file size for logging
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            logging.info(f"📊 File size: {file_size:.2f} MB")
            
            self.minio_client.fput_object(
                bucket_name=BUCKET_NAME,
                object_name=object_name,
                file_path=str(file_path),
            )
            logging.info(f"✅ Successfully uploaded {file_path.name} to MinIO")
            return object_name
        except Exception as e:
            logging.error(f"❌ Failed to upload file to MinIO: {e}")
            raise

    def run_bulk_import(self, collection_name: str, file_name: str):
        """Triggers and monitors the Milvus bulk import job with enhanced logging."""
        logging.info("🚀 Starting Milvus bulk import job...")
        
        try:
            # Validate collection exists
            if not utility.has_collection(collection_name):
                logging.error(f"❌ Collection '{collection_name}' does not exist!")
                return
                
            # Get collection info
            collection = Collection(collection_name)
            collection_info = collection.describe()
            logging.info(f"📋 Collection info: {collection_info}")
            
            # Try different path formats
            file_paths_to_try = [
                file_name,  # Just filename
                f"{BUCKET_NAME}/{file_name}",  # Bucket/filename
                f"/{BUCKET_NAME}/{file_name}",  # /Bucket/filename
            ]
            
            for attempt, file_path in enumerate(file_paths_to_try, 1):
                logging.info(f"📥 Attempt {attempt}: Trying file path format: {file_path}")
                
                try:
                    task_id_response = utility.do_bulk_insert(
                        collection_name=collection_name,
                        files=[file_path]
                    )
                    task_id = task_id_response[0] if isinstance(task_id_response, list) else task_id_response
                    logging.info(f"✅ Bulk insert job started with Task ID: {task_id}")
                    
                    # Monitor the job
                    self._monitor_bulk_import(task_id, file_path)
                    return  # Success, exit function
                    
                except Exception as e:
                    logging.warning(f"   Failed with format '{file_path}': {e}")
                    if attempt < len(file_paths_to_try):
                        logging.info("   Trying next format...")
                        continue
                    else:
                        logging.error("   All path formats failed!")
                        raise
                        
        except Exception as e:
            logging.error(f"❌ An error occurred during bulk import: {e}", exc_info=True)

    def _monitor_bulk_import(self, task_id, file_path):
        """Separate method to monitor bulk import progress"""
        check_count = 0
        while True:
            check_count += 1
            state = utility.get_bulk_insert_state(task_id=task_id)
            
            logging.info(f"📊 Check #{check_count} - Task {task_id} state: {state.state_name}")
            
            if hasattr(state, 'row_count'):
                logging.info(f"   Rows processed: {state.row_count}")
            if hasattr(state, 'progress'):
                logging.info(f"   Progress: {state.progress}")
            if hasattr(state, 'infos'):
                logging.info(f"   Info: {state.infos}")
                
            if state.state_name in ["Completed", "Failed"]:
                if state.state_name == "Failed":
                    logging.error(f"❌ Bulk import failed!")
                    logging.error(f"   Reason: {state.failed_reason}")
                    raise Exception(f"Bulk import failed: {state.failed_reason}")
                else:
                    logging.info(f"✅ Bulk import completed successfully!")
                    if hasattr(state, 'row_count'):
                        logging.info(f"   Total rows imported: {state.row_count}")
                break
                
            time.sleep(5)
    def run_with_validation(self, collection_name: str, file_path: Path):
        """Run complete bulk import process with validation"""
        logging.info("🔄 Starting enhanced bulk import process...")
        
        # Step 1: Validate JSON structure
        if not self.validate_json_structure(file_path, collection_name):
            logging.error("❌ Validation failed. Aborting bulk import.")
            return
        
        # Step 2: Upload to MinIO
        try:
            object_name = self.upload_to_minio(file_path)
        except Exception as e:
            logging.error(f"❌ Upload failed. Aborting bulk import.")
            return
        
        # Step 3: Run bulk import
        self.run_bulk_import(collection_name, object_name)


def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Milvus bulk import with validation")
    parser.add_argument("--file", type=str, default="prepared_for_upload/prepared_data.json")
    parser.add_argument("--collection", type=str, default="rag_chunks")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        logging.error(f"❌ File not found: {file_path}")
        return
    
    importer = EnhancedMilvusBulkImporter()
    importer.run_with_validation(args.collection, file_path)


if __name__ == "__main__":
    main()