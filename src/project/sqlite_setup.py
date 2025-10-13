import sqlite3
import argparse

def create_sqlite_db(db_path="documents.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Documents Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        filename TEXT NOT NULL,
        doc_name TEXT NOT NULL,
        file_extension TEXT NOT NULL,
        header_exists INTEGER,
        file_size INTEGER NOT NULL,
        domain TEXT NOT NULL,
        content_type TEXT NOT NULL,
        language TEXT DEFAULT 'en',
        encoding TEXT DEFAULT 'utf-8',
        ingestion_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_processed DATETIME,
        processing_status TEXT DEFAULT 'pending',
        error_message TEXT,
        total_chars INTEGER,
        total_words INTEGER,
        estimated_tokens INTEGER,
        domain_metadata TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Chunks Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY,
        doc_id TEXT NOT NULL,
        doc_name TEXT NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        chunk_size INTEGER NOT NULL,
        chunk_tokens INTEGER,
        chunk_method TEXT NOT NULL,
        chunk_overlap INTEGER DEFAULT 0,
        start_position INTEGER,
        end_position INTEGER,
        domain TEXT NOT NULL,
        content_type TEXT NOT NULL,
        embedding_model TEXT,
        embedding_vector BLOB,
        vector_id TEXT,
        embedding_timestamp DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
    print(f"SQLite tables in '{db_path}' created successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and set up the SQLite database.")
    parser.add_argument("--db-path", default="documents.db", help="Path to the SQLite database file.")
    args = parser.parse_args()
    create_sqlite_db(args.db_path)
