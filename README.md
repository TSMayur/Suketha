# Advanced RAG Pipeline with Streaming Ingestion

This project implements a high-performance, database-driven RAG pipeline. It's designed for ingesting large quantities of documents, processing them in a memory-efficient streaming fashion, and querying them with an optimized hybrid search.

The pipeline uses **SQLite** to manage file ingestion state and **Milvus** as the vector store for a robust, scalable solution.

## Key Features

* **🚀 Streaming Ingestion Pipeline:** (`complete_pipeline_hybrid.py`) Uses a parallel producer/consumer model to read, chunk, and ingest documents without loading entire files into memory.
* **🔍 Optimized Hybrid Search:** (`process_all_queries_csv.py`) Implements a sophisticated query process using Milvus's native hybrid search (BM25 + Dense Vectors), RRF reranking, and a final Cross-Encoder pass for state-of-the-art accuracy.
* **📋 Database-Driven Workflow:** The entire ingestion process is managed by an SQLite database.
    * `sqlite_setup.py`: Initializes the database.
    * `file_meta_loader.py`: Scans source directories and registers files as 'pending' in the DB.
    * `complete_pipeline_hybrid.py`: Reads the 'pending' files from the DB, processes them, and inserts them into Milvus.
* **🐳 Dockerized Infrastructure:** Includes a `docker-compose.yml` file to instantly launch a local Milvus, MinIO, and etcd stack.
* **🧹 Advanced Text Processing:**
    * `doc_reader.py`: Supports multiple file types (PDF, TXT, DOCX, CSV, etc.).
    * `chunk_cleaner.py`: Scrubs noise, metadata, and malformed JSON from text chunks before embedding.
* **📦 Modern Python Stack:** Managed with **Poetry** for reproducible dependencies.

---

## 🚀 Getting Started

### Prerequisites

* **Python 3.10+**
* **[Poetry](https://python-poetry.org/docs/#installation)**
* **[Docker](https://www.docker.com/get-started)** & Docker Compose

### 1. Clone & Install Dependencies

```sh
# Clone the repository
git clone [https://github.com/TSMayur/Suketha.git](https://github.com/TSMayur/Suketha.git)
cd Suketha

# Install all dependencies using Poetry
poetry install
````

### 2\. Configure Your Environment

You must create a `.env` file in the root of the project. This file tells the scripts how to connect to Milvus and other services.

**➡️ Create a file named `.env`** and add the following:

```ini
# --- Milvus Connection ---
# Use this for the local Docker setup
AZURE_MILVUS_URI="http://localhost:19530"
AZURE_MILVUS_TOKEN=""

# (Optional) Or, use this for a cloud (Azure) instance
# AZURE_MILVUS_URI="your-azure-milvus-uri.milvus.azure.com:19530"
# AZURE_MILVUS_TOKEN="your-azure-milvus-token"

# --- Collection Name ---
COLLECTION_NAME="rag_hybrid_chunks"

# --- TEI Endpoint ---
# This is required by schema_setup.py for the embedding function.
# You must provide a URL to a running Text Embeddings Inference container.
TEI_ENDPOINT="http://your-tei-embedding-service-url"
```

### 3\. Start Local Infrastructure

This command starts Milvus, MinIO, and etcd using the `docker-compose.yml` file.

```sh
docker-compose up -d
```

You can view the Milvus UI (Attu) at [http://localhost:8000](https://www.google.com/search?q=http://localhost:8000).

-----

## ⚙️ Usage: The Main Workflow

### Step 1: (One-Time) Create the Milvus Collection

Before you can ingest data, you must create the collection in Milvus. This script configures the schema for hybrid search (BM25 sparse + dense vectors).

```sh
poetry run python -m project.schema_setup
```

*(This only needs to be run once. It reads the `COLLECTION_NAME` and `TEI_ENDPOINT` from your `.env` file).*

### Step 2: Ingest Your Documents

This is a multi-step process that uses SQLite to track files.

1.  **Add your files** (PDFs, TXT, DOCX, etc.) into a source directory, for example: `data/test`.

2.  **Clean up old database files** (optional, but recommended for a fresh start):

    ```sh
    rm -f data/hybrid/documents.db
    rm -f prepared_for_upload/processed_chunks.json
    ```

3.  **Create a fresh SQLite database** to track your files:

    ```sh
    poetry run python src/project/sqlite_setup.py --db-path data/hybrid/documents.db
    ```

4.  **Scan your data folder** and register all files in the database:

    ```sh
    poetry run python src/project/file_meta_loader.py data/test --db data/hybrid/documents.db
    ```

5.  **Run the main streaming pipeline.** This script reads the "pending" files from the SQLite DB, processes them in parallel, and inserts them into Milvus:

    ```sh
    poetry run python -m project.complete_pipeline_hybrid \
        --input-dir data/hybrid \
        --output-dir prepared_for_upload
    ```

### Step 3: Run Queries

Once your data is ingested, you can run queries.

1.  **Edit the query file:** Add your questions to `new_queries.json`. It should look like this:

    ```json
    [
      {
        "query_num": "1",
        "query": "What is the capital of France?"
      },
      {
        "query_num": "2",
        "query": "Describe the process of photosynthesis."
      }
    ]
    ```

2.  **Run the query script:**

    ```sh
    poetry run python -m project.process_all_queries_csv
    ```

This will:

1.  Embed all queries.
2.  Perform a parallel hybrid search (BM25 + Dense) in Milvus.
3.  Rerank the results using a Cross-Encoder.
4.  Generate a `submission/` folder with individual JSON results.
5.  Create a detailed `Final_shortlistingchunks100rerank.csv` file with all results, scores, and chunk text for analysis.

-----

## 📂 Project Structure

```
.
|-- Data
├── docker-compose.yml        # Infrastructure (Milvus, MinIO, etcd)
├── pyproject.toml            # Poetry dependencies
├── poetry.lock               # Exact dependency versions
├── .env                      # <-- Your local configuration (you must create this)
├── new_queries.json          # Input queries for batch processing
└── src
    └── project
        ├── config.py             # Loads .env file and configures Milvus client
        │
        ├── # 1. INGESTION WORKFLOW
        ├── sqlite_setup.py       # Creates SQLite tables (documents, chunks)
        ├── file_meta_loader.py   # Scans folders, populates 'documents' table
        ├── complete_pipeline_hybrid.py # <-- MAIN: Streaming Ingestion (Producer/Consumer)
        │
        ├── # 2. QUERY WORKFLOW
        ├── process_all_queries_csv.py # <-- MAIN: Hybrid Search + Reranking
        │
        ├── # 3. MILVUS & SCHEMA
        ├── schema_setup.py       # Creates Milvus collection schema (hybrid)
        ├── milvus_bulk_import.py # Utility for bulk-loading from MinIO
        │
        ├── # 4. CORE UTILITIES
        ├── doc_reader.py         # Reads .pdf, .txt, .docx, etc.
        ├── chunker.py            # Splits documents into chunks
        ├── chunk_cleaner.py      # Cleans text, removes noise
        └── pydantic_models.py    # Core data models (Document, Chunk)
```

## Main Dependencies

A complete list is in `pyproject.toml`. Key libraries include:

  * `pymilvus`: The Milvus client.
  * `sentence-transformers`: For query embedding.
  * `cross-encoder`: For reranking.
  * `pydantic`: For data models.
  * `aiosqlite`: For async SQLite operations.
  * `psutil`: For memory management.

<!-- end list -->
