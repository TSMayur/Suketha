# src/project/storage_manager.py

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class StorageManager:
    """Handles saving prepared data to local files."""

    @staticmethod
    def save_as_ndjson(chunks: List[Dict[str, Any]], output_path: Path, filename: str):
        """
        Saves a list of chunk dictionaries to a NDJSON file, which is
        efficient for streaming and bulk imports. Each line is a separate JSON object.
        """
        output_file = output_path / filename
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk) + "\n")
            logging.info(f"Successfully saved {len(chunks)} chunks to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save data to {output_file}: {e}")

    @staticmethod
    def save_as_json(chunks: List[Dict[str, Any]], output_path: Path, filename: str):
        """
        Saves a list of chunk dictionaries to a single JSON file in the format
        required by some bulk import processes (with a root "rows" key).
        """
        output_file = output_path / filename
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                # The data is already a list of dicts, wrap it in a "rows" key
                json.dump({"rows": chunks}, f, indent=4)
            logging.info(f"Successfully saved {len(chunks)} chunks to {output_file}")
        except Exception as e:
            logging.error(f"Failed to save data to {output_file}: {e}")