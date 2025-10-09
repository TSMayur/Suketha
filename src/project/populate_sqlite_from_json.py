import sqlite3
import json
import argparse
from pathlib import Path

def populate_db_from_json(db_path: str, json_path: str):
    """
    Populates the 'chunks' table in a SQLite database from a JSON file.

    Args:
        db_path (str): The path to the SQLite database file.
        json_path (str): The path to the JSON file containing the chunk data.
    """
    
    if not Path(json_path).exists():
        print(f"Error: JSON file not found at {json_path}")
        return
        
    if not Path(db_path).exists():
        print(f"Error: Database not found at {db_path}. Please run sqlite_steup.py first.")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rows = data.get("rows", [])
    if not rows:
        print("No rows found in the JSON file. Nothing to insert.")
        return

    # Prepare data for bulk insertion
    chunks_to_insert = []
    for row in rows:
        chunks_to_insert.append((
            row.get("chunk_id"),
            row.get("doc_id"),
            row.get("doc_name"),
            row.get("chunk_index"),
            row.get("chunk_text"),
            row.get("chunk_size"),
            row.get("chunk_tokens"),
            row.get("chunk_method"),
            row.get("chunk_overlap"),
            row.get("domain"),
            row.get("content_type"),
            row.get("embedding_model"),
        ))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Use executemany for efficient bulk insertion
    try:
        cur.executemany("""
        INSERT INTO chunks (
            chunk_id, doc_id,doc_name, chunk_index, chunk_text, chunk_size, 
            chunk_tokens, chunk_method, chunk_overlap, domain, 
            content_type, embedding_model
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, chunks_to_insert)
        
        conn.commit()
        print(f"Successfully inserted {len(chunks_to_insert)} chunks into the database.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Populate SQLite DB from a JSON file.")
    parser.add_argument("--db-path", type=str, required=True, help="Path to the SQLite database file.")
    parser.add_argument("--json-path", type=str, required=True, help="Path to the JSON input file.")
    
    args = parser.parse_args()
    populate_db_from_json(args.db_path, args.json_path)