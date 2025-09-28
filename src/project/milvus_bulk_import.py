# src/project/milvus_bulk_importer.py

import logging
from pathlib import Path
from minio import Minio
from pymilvus import utility, connections
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

class MilvusBulkImporter:
    def __init__(self):
        self.minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        logging.info(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    def upload_to_minio(self, file_path: Path) -> str:
        """Uploads a single prepared data file to the MinIO bucket."""
        try:
            if not self.minio_client.bucket_exists(BUCKET_NAME):
                self.minio_client.make_bucket(BUCKET_NAME)
                logging.info(f"Created MinIO bucket '{BUCKET_NAME}'")
            
            object_name = file_path.name
            self.minio_client.fput_object(
                bucket_name=BUCKET_NAME,
                object_name=object_name,
                file_path=str(file_path),
            )
            logging.info(f"Successfully uploaded {file_path.name} to MinIO")
            return object_name
        except Exception as e:
            logging.error(f"Failed to upload file to MinIO: {e}")
            raise

    def run_bulk_import(self, collection_name: str, file_name: str):
        """Triggers and monitors the Milvus bulk import job."""
        logging.info("Starting Milvus bulk import job...")
        try:
            task_id_response = utility.do_bulk_insert(
                collection_name=collection_name,
                files=[file_name]
            )
            task_id = task_id_response[0] if isinstance(task_id_response, list) else task_id_response
            logging.info(f"Bulk insert job started with Task ID: {task_id}")

            # Monitor the job until it completes or fails
            while True:
                state = utility.get_bulk_insert_state(task_id=task_id)
                logging.info(f"Task {task_id} state: {state.state_name}")
                if state.state_name in ["Completed", "Failed"]:
                    if state.state_name == "Failed":
                        logging.error(f"Bulk import failed: {state.failed_reason}")
                    break
                time.sleep(5) # Wait 5 seconds before checking again

        except Exception as e:
            logging.error(f"An error occurred during bulk import: {e}", exc_info=True)