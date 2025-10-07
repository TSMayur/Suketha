# Advanced RAG Pipeline with Milvus & Poetry

This project implements a comprehensive and optimized Retrieval-Augmented Generation (RAG) pipeline designed for efficient document processing, embedding, and querying. It leverages state-of-the-art tools like Milvus for vector storage and SentenceTransformers for embeddings, all managed within a clean Poetry environment.

## Features

  * **Multi-Format Document Processing**: Ingests and processes a wide range of document formats including PDF, TXT, JSON, DOCX, CSV, and TSV.
  * **Optimized Chunking**: Implements various chunking strategies (recursive, Spacy, NLTK) to effectively segment documents.
  * **High-Performance Embedding**: Utilizes `sentence-transformers/all-mpnet-base-v2` for generating high-quality embeddings, with optimizations for both CPU and GPU workflows.
  * **Milvus Integration**: Employs Milvus as a robust and scalable vector store for storing and querying document chunks.
  * **Bulk Import**: Features a streamlined bulk import process using MinIO for efficient data loading into Milvus.
  * **Parallel Query Processing**: Optimized for handling large batches of queries concurrently for high-throughput searching.
  * **Dockerized Environment**: Comes with a `docker-compose.yml` for easy setup of the required services (Milvus, MinIO, etcd).
  * **Dependency Management**: Uses **Poetry** for clear, deterministic dependency management.

## Getting Started

Follow these instructions to get a local copy of the project up and running.

### Prerequisites

  * [Docker](https://www.docker.com/get-started) and Docker Compose
  * [Poetry](https://www.google.com/search?q=https://python-poetry.org/docs/%23installation) (for managing Python dependencies)
  * Python 3.10+

### Installation

1.  **Clone the repository:**

    ```sh
    git clone https://github.com/TSMayur/Suketha.git
    cd Suketha
    ```

2.  **Start the infrastructure services:**
    This command launches Milvus, MinIO, and other dependencies in Docker containers.

    ```sh
    docker-compose up -d
    ```

3.  **Install Python dependencies using Poetry:**
    This will create a virtual environment and install all the packages listed in `pyproject.toml`.

    ```sh
    poetry install
    ```

## Usage

All scripts should be run using `poetry run`. This ensures they execute within the correct virtual environment with all dependencies available.

### 1\. Set Up the Milvus Collection

First, create the Milvus collection with the correct schema.

```sh
poetry run python -m project.schema_setup
```

### 2\. Process and Ingest Documents

Place your source documents (PDFs, TXT files, etc.) into a directory (e.g., `data/`). Then, run the complete processing pipeline. This script will read the documents, chunk them, generate embeddings, and bulk import the data into Milvus.

Choose the appropriate pipeline for your hardware:

```sh
# For CPU or hybrid environments
poetry run python -m project.complete_pipeline_hybrid --input-dir data/ --output-dir prepared_data/

# For GPU (MPS) environments
poetry run python -m project.complete_pipeline_gpu --input-dir data/ --output-dir prepared_data/
```

### 3\. Run Queries

Once your data is indexed, you can process a batch of queries from a JSON file.

  * **For JSON output:**
    ```sh
    poetry run python -m project.process_all_queries
    ```
  * **For detailed CSV output:**
    ```sh
    poetry run python -m project.process_all_queries_csv
    ```

You can also query the documents interactively:

```sh
poetry run python -m project.query_document
```

## Project Structure

```
├── docker-compose.yml      # Docker configuration for infrastructure
├── pyproject.toml          # Poetry dependencies and project metadata
└── src
    └── project
        ├── chunker.py              # Document chunking logic
        ├── doc_reader.py           # Reads various document formats
        ├── embedder.py             # Generates embeddings for text chunks
        ├── milvus.py               # Milvus client and search functions
        ├── milvus_bulk_import.py   # Bulk import data into Milvus
        ├── complete_pipeline_gpu.py # Main data processing pipeline (GPU optimized)
        ├── complete_pipeline_hybrid.py # Main data processing pipeline (CPU/Hybrid)
        ├── process_all_queries.py  # Batch query processing script
        ├── pydantic_models.py      # Core data models for the project
        └── ...
```

## Main Dependencies

  * `pymilvus`: For interacting with Milvus.
  * `sentence-transformers`: For text embeddings.
  * `langchain`: For various text processing utilities.
  * `pypdf`: For reading PDF files.
  * `minio`: For the bulk import process.

A full list of dependencies can be found in the **`pyproject.toml`** file.