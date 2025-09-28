import os
import sqlite3
import numpy as np
import json
from pathlib import Path
from project.processor import DocumentProcessor
from project.pydantic_models import ProcessingConfig

DATA_DIR = r"D:\genai\RAG\test"
BATCH_SIZE = 20
NDJSON_FILE = r"chunks_bulk.ndjson"
SQLITE_DB = r"rag_chunks.db"

def get_all_files(directory, extensions=None):
    extensions = extensions or [".json", ".txt", ".csv", ".tsv"]
    return [str(p) for p in Path(directory).rglob("*") if p.suffix.lower() in extensions]

def convert_chunks_to_dicts(document, chunks):
    dicts = []
    for i, chunk in enumerate(chunks):
        dicts.append({
            "chunk_id": f"{document.id}_chunk_{i}",
            "doc_id": document.id,  # always use doc_id!
            "chunk_index": i,
            "chunk_text": chunk.content,
            "chunk_size": len(chunk.content),
            "chunk_tokens": len(chunk.content.split()),
            "chunk_method": getattr(chunk, "chunking_method", "recursive").value if hasattr(getattr(chunk, "chunking_method", "recursive"), "value") else str(getattr(chunk, "chunking_method", "recursive")),
            "chunk_overlap": 50,
            "start_position": None,
            "end_position": None,
            "domain": getattr(document, "domain", "general"),
            "content_type": getattr(document, "file_type", "unknown").value if hasattr(getattr(document, "file_type", "unknown"), "value") else str(getattr(document, "file_type", "unknown")),
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "vector_id": None,
            "embedding_timestamp": None,
            "created_at": None,
            "embedding_vector": chunk.embedding
        })
    return dicts

def append_ndjson(chunk_dicts, path):
    with open(path, "a", encoding="utf-8") as f:
        for chunk in chunk_dicts:
            out = {k: chunk[k] for k in [
                "chunk_id", "doc_id", "chunk_index", "chunk_text", "chunk_size",
                "chunk_tokens", "chunk_method", "chunk_overlap",
                "start_position", "end_position", "domain", "content_type",
                "embedding_model", "vector_id", "embedding_timestamp",
                "created_at", "embedding_vector"
            ]}
            f.write(json.dumps(out) + "\n")

def bulk_insert_sqlite_chunks(chunk_dicts, db_path=SQLITE_DB):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    chunk_rows = []
    for chunk in chunk_dicts:
        chunk_rows.append((
            chunk['chunk_id'], chunk['doc_id'], chunk['chunk_index'],
            chunk['chunk_text'], chunk['chunk_size'], chunk['chunk_tokens'],
            chunk['chunk_method'], chunk['chunk_overlap'],
            chunk.get('start_position'), chunk.get('end_position'),
            chunk['domain'], chunk['content_type'], chunk['embedding_model'],
            np.array(chunk['embedding_vector'], dtype='float32').tobytes(),
            chunk.get('vector_id'), chunk.get('embedding_timestamp'),
            chunk.get('created_at')
        ))
    cur.executemany("""INSERT OR IGNORE INTO chunks (
        chunk_id, doc_id, chunk_index, chunk_text, chunk_size, chunk_tokens,
        chunk_method, chunk_overlap, start_position, end_position, domain,
        content_type, embedding_model, embedding_vector, vector_id,
        embedding_timestamp, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        chunk_rows)
    conn.commit()
    conn.close()
    print(f"Inserted {len(chunk_rows)} chunks into SQLite.")

def main():
    if os.path.exists(NDJSON_FILE):
        os.remove(NDJSON_FILE)
    config = ProcessingConfig()
    processor = DocumentProcessor(config)
    file_list = get_all_files(DATA_DIR)
    print(f"Found {len(file_list)} files.")
    for batch_start in range(0, len(file_list), BATCH_SIZE):
        batch_files = file_list[batch_start:batch_start+BATCH_SIZE]
        all_chunk_dicts = []
        for file_path in batch_files:
            print(f"Processing {file_path}")
            try:
                document, chunks = processor.process_document(file_path)
                chunk_dicts = convert_chunks_to_dicts(document, chunks)
                all_chunk_dicts.extend(chunk_dicts)
            except Exception as e:
                print(f"Error: {e} for {file_path}")
        if all_chunk_dicts:
            append_ndjson(all_chunk_dicts, NDJSON_FILE)
            bulk_insert_sqlite_chunks(all_chunk_dicts, db_path=SQLITE_DB)
        print(f"Batch: {batch_start//BATCH_SIZE + 1} processed.")
    print("All batches processed. NDJSON ready for Milvus.")

if __name__ == "__main__":
    main()
