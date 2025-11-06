# Complete RAG Pipeline Documentation

**Key Technologies**: Python Poetry, Milvus vector database, SQLite, sentence-transformers, LangChain, Pydantic

---

## Phase 1: Initial Project Setup and Poetry Introduction

### The Poetry Decision

**Question Asked:** "What is Poetry and why should we use it?"

**Decision Made:** Use Poetry as the dependency and environment management tool instead of traditional pip and requirements.txt.

### Rationale:

- Deterministic dependency resolution
- Automatic virtual environment management
- Lock files for reproducible builds
- Simplified package publishing
- Better dependency conflict resolution

### Initial Setup Steps

Instead of using a `requirements.txt`, Poetry's `pyproject.toml` file was used. This provides better version pinning and environment reproducibility.

```bash
# Install Poetry
pip install poetry

# Create project
poetry new your_project_name
cd your_project_name

# Install dependencies
poetry add langchain pydantic sentence-transformers pymilvus
```

### Project Structure Established

```
rag-project/
├── src/
│   └── project/
│       ├── __init__.py
│       ├── models.py
│       ├── pdf_reader.py
│       ├── chunker.py
│       ├── embedder.py
│       └── main.py
├── data/
│   └── input/
├── pyproject.toml
└── poetry.lock
```

**Key Decision**: Modular architecture with separate files for each concern (separation of concerns principle).

### Initial Database: Why Weaviate Was Used First

- The first working version was built for Weaviate (open-source, Docker-deployable vector DB).
- Weaviate was chosen because:
  - It had simple LangChain integration and REST API.
  - Local testing was easy, with Docker or cloud support.
  - No license fee for basic usage.

### Why The Shift to Milvus?

The main motivators and potential reasons:

1. **Performance & Scalability**: Milvus provides higher scalability and performance for very large datasets, supporting billions of vectors.

2. **Cost and Accessibility**: Milvus is completely free and open-source for all features, unlike Weaviate's advanced features which may require a license or SaaS plan for bigger production loads.

3. **Community & Ecosystem**: Milvus has a rapidly growing community, extensive documentation, and is widely recognized in the AI vector database ecosystem.

4. **Flexible Schema and Index Support**: Milvus allows custom field schemas and supports various ANN (Approximate Nearest Neighbor) index types (like HNSW, IVF_FLAT), useful for tuning vector search speed and accuracy.

5. **First-Class LangChain and PyMilvus Support**: Tight, officially maintained integrations that work out of the box, and better error handling as the project scales.

---

## Phase 2: Document Processing and File Readers

### Problem Statement

Need to process multiple document formats: PDF, TXT, JSON, DOCX, CSV, and TSV (including TSV with hash symbols in filenames).

### Implementation Strategy

**File: doc_reader.py**

Created a unified document loader that:
- Detects file type by extension
- Routes to appropriate parser
- Returns standardized Document objects
- Handles errors gracefully

### File Type Support

| File Type | Library Used | Special Handling |
|-----------|-------------|------------------|
| PDF | PyPDF2 | Page-by-page text extraction |
| TXT | Built-in | Direct read with encoding detection |
| JSON | json module | Structure-preserving parsing |
| DOCX | python-docx | Paragraph extraction |
| CSV | pandas | DataFrame to text conversion |
| TSV | pandas (sep='\t') | Tab-separated parsing |
| TSV# | Custom detection | Hash symbol filename handling |

### Key Code Pattern

```python
def load_document(file_path: str) -> Document:
    file_type = detect_file_type(file_path)
    
    if file_type == FileType.PDF:
        return _load_pdf(file_path)
    elif file_type == FileType.CSV:
        return _load_csv(file_path)
    # ... routing logic
```

**Decision Point**: Use pandas for CSV/TSV instead of csv module for better data type handling and easier preprocessing.

---

## Phase 3: Chunking Strategy Development

### The Chunking Problem

Large documents cannot be embedded as single units. Need to split into smaller, semantically meaningful chunks while maintaining context.

### Chunking Methods Implemented

#### 1. Recursive Text Splitting (Default for Text/PDF)

**Parameters Decided:**
- Chunk size: 1024 characters
- Overlap: 256 characters
- Separators: Paragraphs → Sentences → Words → Characters

**Rationale**: Overlap ensures no information loss at boundaries. Hierarchical splitting preserves natural text structure.

```python
def create_chunks(
    document_id: str, 
    text: str, 
    chunk_size: int = 1024,
    overlap: int = 256
) -> List[Chunk]:
    # Recursive splitting logic
```

#### 2. JSON-Aware Chunking

For JSON documents, preserve object structure:
- Split by top-level keys
- Maintain key-value relationships
- Handle nested objects appropriately

#### 3. CSV/TSV Row-Based Chunking

**Major Decision Point**: After initial implementation, realized CSV/TSV files were being treated as plain text, breaking row structure.

**Solution Implemented:**

```python
def csv_tsv_chunking(document: Document, config: ProcessingConfig) -> List[Chunk]:
    df = pd.read_csv(io.StringIO(document.content), sep=separator)
    
    # Group rows into chunks based on byte size
    max_bytes = 2048  # 2KB per chunk
    current_rows = []
    
    for i, row in df.iterrows():
        row_dict = {col: str(row[col]) for col in df.columns}
        row_text = json.dumps(row_dict)
        
        if len(row_text) + running_length > max_bytes:
            # Flush current chunk
            yield create_chunk(current_rows)
            current_rows = [row_dict]
        else:
            current_rows.append(row_dict)
```

**Key Insight**: Each chunk should contain complete rows, not partial rows split across chunks. Metadata includes column names and row ranges.

### Chunking Metadata

Every chunk includes:
- `chunk_id`: Unique identifier
- `doc_id`: Parent document reference
- `chunk_index`: Position in document
- `chunking_method`: Method used (recursive, JSON, CSV, etc.)
- `chunk_overlap`: Overlap size
- `start_position`: Character offset in original document
- `end_position`: End character offset
- `content_type`: File type of source

---

## Phase 4: Embedding Generation

### Model Selection

**Chosen Model**: `sentence-transformers/all-mpnet-base-v2`

**Why This Model:**
- 768-dimensional embeddings (good balance of quality and size)
- Trained on diverse text corpora
- Strong semantic understanding
- Fast inference on CPU
- Normalization-friendly for cosine similarity

### Embedding Pipeline

```python
def add_embeddings_to_chunks(chunks: List[Chunk]) -> List[Chunk]:
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    for chunk in chunks:
        # Normalize for cosine similarity
        embedding = model.encode(
            chunk.content,
            normalize_embeddings=True
        ).tolist()
        chunk.embedding = embedding
    
    return chunks
```

**Batch Processing Decision**: Process chunks in batches of 32 for optimal GPU/CPU utilization.

### Normalization Strategy

All embeddings are L2-normalized at generation time because:
- Milvus uses COSINE metric
- Normalized vectors make cosine similarity equivalent to dot product
- Faster search operations
- Consistent similarity ranges (0 to 1)

---

## Phase 5: Vector Database Architecture - Milvus

### Why Milvus?

**Evaluation Criteria:**

| Feature | Milvus | Alternatives |
|---------|--------|--------------|
| Scalability | Excellent (billions of vectors) | Limited |
| Performance | HNSW index, sub-ms latency | Varies |
| Deployment | Docker, cloud-ready | Complex |
| Schema flexibility | Dynamic fields supported | Limited |
| Hybrid search | Supported | Limited |

**Decision**: Milvus for production-grade vector search with flexibility for future enhancements.

### Docker Setup

**File: docker-compose.yml**

```yaml
version: '3.4'
services:
  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - ./milvus_data:/var/lib/milvus
```

**Key Decision**: Use Docker Compose (not bare docker run) for:
- Reproducibility across team members
- Easy configuration management
- Volume mounting for data persistence
- Multi-service orchestration (Milvus + etcd + MinIO)

### Schema Design Evolution

#### Initial Schema (Had Issues)

```python
schema.add_field("chunk_id", DataType.VARCHAR, is_primary=True)
schema.add_field("content", DataType.VARCHAR)
schema.add_field("embedding_vector", DataType.FLOAT_VECTOR, dim=768)
```

**Problem Encountered**: Insert failures with error "missing field content_type".

#### Final Schema (After Fixes)

**File: schema_setup.py**

```python
def create_collection():
    schema = MilvusClient.create_schema(auto_id=False)
    
    # Primary key
    schema.add_field("chunk_id", DataType.VARCHAR, max_length=255, is_primary=True)
    
    # Document relationship
    schema.add_field("doc_id", DataType.VARCHAR, max_length=255)
    schema.add_field("chunk_index", DataType.INT64)
    
    # Content
    schema.add_field("chunk_text", DataType.VARCHAR, max_length=65535)
    schema.add_field("chunk_size", DataType.INT64)
    schema.add_field("chunk_tokens", DataType.INT64)
    
    # Chunking metadata
    schema.add_field("chunk_method", DataType.VARCHAR, max_length=100)
    schema.add_field("chunk_overlap", DataType.INT64)
    schema.add_field("start_position", DataType.INT64)
    schema.add_field("end_position", DataType.INT64)
    
    # File metadata
    schema.add_field("domain", DataType.VARCHAR, max_length=100)
    schema.add_field("content_type", DataType.VARCHAR, max_length=50)
    schema.add_field("embedding_model", DataType.VARCHAR, max_length=200)
    
    # Required for storage
    schema.add_field("vector_id", DataType.VARCHAR, max_length=255)
    schema.add_field("embedding_timestamp", DataType.VARCHAR, max_length=50)
    schema.add_field("created_at", DataType.VARCHAR, max_length=50)
    
    # Vector field
    schema.add_field("embedding_vector", DataType.FLOAT_VECTOR, dim=768)
    
    # Indexes
    index_params = client.prepare_index_params()
    index_params.add_index("chunk_index", index_type="STL_SORT")
    index_params.add_index("embedding_vector", index_type="HNSW", metric_type="COSINE")
    
    client.create_collection(
        collection_name="rag_chunks",
        schema=schema,
        index_params=index_params,
        consistency_level="Bounded"
    )
```

### Critical Schema Errors Fixed

1. **Missing vector_id field**: Added as required unique identifier for vector operations
2. **Missing content_type field**: Added to track source file type
3. **Wrong index type on string fields**: Removed STL_SORT from VARCHAR fields (only valid for numeric)
4. **Incomplete position tracking**: Added proper start_position and end_position calculation

---

## Phase 6: Dual Storage Architecture

### The Problem

Vector search (Milvus) is excellent for semantic similarity but:
- Limited metadata query capabilities
- No complex filtering on non-vector fields
- Difficult to track document lineage
- No easy way to get document-level statistics

### Solution: SQLite + Milvus Hybrid

**File: sqlite_setup.py**

```python
def create_tables(db_path):
    conn = sqlite3.connect(db_path)
    
    # Documents table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_type TEXT NOT NULL,
            file_size INTEGER,
            processing_status TEXT DEFAULT 'pending',
            chunk_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Chunks table (mirror of Milvus)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            chunk_size INTEGER,
            chunk_method TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        )
    """)
```

### Data Flow

```
Document → Load → Chunk → Embed → Store in:
                                    ├─ Milvus (vectors + metadata)
                                    └─ SQLite (metadata + lineage)
```

**Benefits:**
- Fast semantic search via Milvus
- Complex metadata queries via SQLite
- Document tracking and status management
- Easy analytics and reporting
- Backup and recovery options

---

## Phase 7: Bulk Processing and Pipeline Optimization

### Initial Approach (Failed)

Used multiprocessing for parallel document processing:

**Problems:**
- Memory exhaustion on large files
- Process spawn overhead
- Difficulty debugging
- Inconsistent behavior across OS

### Final Approach (Serial Processing)

**File: bulk_upload.py**

```python
def process_single_file(file_path):
    stats = {
        "filename": os.path.basename(file_path),
        "load_time": 0,
        "chunk_time": 0,
        "embed_time": 0,
        "storage_time": 0
    }
    
    # Load and chunk
    t0 = time.time()
    document, chunks = processor.process_document(file_path)
    stats["load_time"] = time.time() - t0
    
    # Embed
    t0 = time.time()
    chunks_with_embeddings = add_embeddings_to_chunks(chunks)
    stats["embed_time"] = time.time() - t0
    
    # Store
    t0 = time.time()
    store_chunks_milvus(chunks_with_embeddings)
    store_chunks_sqlite(chunks_with_embeddings)
    stats["storage_time"] = time.time() - t0
    
    return stats

# Process all files serially
for file_path in all_files:
    try:
        stats = process_single_file(file_path)
        log_stats(stats)
    except Exception as e:
        log_error(file_path, e)
```

**Why Serial Processing Won:**
- Predictable memory usage
- Easy error handling and logging
- Consistent performance
- Simple debugging
- Sufficient speed for most use cases

### Memory Management

- Process one file at a time
- Clear chunk lists after storage
- Garbage collect after each file
- Batch insert to Milvus (100 chunks per batch)

---

## Phase 8: Pydantic Models and Type Safety

### Why Pydantic?

**Decision**: Use Pydantic for all data models instead of plain dictionaries.

**File: pydantic_models.py**

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class FileType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    JSON = "json"
    CSV = "csv"
    TSV = "tsv"

class ChunkingMethod(str, Enum):
    RECURSIVE = "recursive"
    CHARACTER = "character"
    JSON = "json"
    CSV_ROW = "csv_row"

class Document(BaseModel):
    id: str
    title: str
    content: str = Field(..., min_length=10)
    file_type: FileType
    metadata: Dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    id: str
    doc_id: str
    content: str = Field(..., min_length=10)
    chunk_index: int
    chunking_method: ChunkingMethod
    embedding: Optional[List[float]] = None
    
    # Schema-required fields
    chunk_overlap: int = 0
    start_position: int = 0
    end_position: int = 0
    vector_id: str = ""
    content_type: str = ""
```

**Benefits:**
- Compile-time type checking
- Automatic validation
- Clear documentation
- Easy refactoring
- IDE autocomplete support

---

## Phase 9: Query and Retrieval System

### Semantic Search Implementation

**File: query_engine.py**

```python
def search_similar_chunks(query: str, top_k: int = 5):
    # Generate query embedding
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    query_vector = model.encode(query, normalize_embeddings=True).tolist()
    
    # Search Milvus
    client = MilvusClient(uri="http://localhost:19530")
    results = client.search(
        collection_name="rag_chunks",
        data=[query_vector],
        anns_field="embedding_vector",
        search_params={"metric_type": "COSINE", "params": {"ef": 64}},
        output_fields=["chunk_id", "chunk_text", "doc_id", "doc_name"],
        limit=top_k
    )
    
    return results
```

### HNSW Index Parameters

**Chosen Settings:**
- Index type: HNSW (Hierarchical Navigable Small World)
- Metric: COSINE
- ef (search-time): 64
- M (graph connectivity): 16 (default)

**Rationale:**
- HNSW provides best speed/accuracy tradeoff
- ef=64 balances recall and latency
- COSINE metric matches normalized embeddings

### Hybrid Search Considerations

For future enhancement, planned hybrid search combining:
- Vector similarity (semantic)
- BM25 keyword matching (lexical)
- Metadata filtering (structured)

---

## Phase 10: Evaluation Framework

### The Baseline Problem

**Question**: How do we know if our RAG system is actually good?

**Answer**: Compare against established baselines using standard IR metrics.

### Baseline Methods Implemented

#### 1. BM25 (Best Match 25)

Classic keyword-based ranking:

```python
from rank_bm25 import BM25Okapi

corpus_texts = [chunk.content for chunk in chunks]
bm25 = BM25Okapi([text.split() for text in corpus_texts])
scores = bm25.get_scores(query.split())
```

#### 2. TF-IDF with Cosine Similarity

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
corpus_vectors = vectorizer.fit_transform(corpus_texts)
query_vector = vectorizer.transform([query])
scores = cosine_similarity(query_vector, corpus_vectors)
```

#### 3. Random Baseline (Control)

Random selection to establish lower bound.

### LLM-Based Relevance Annotation

**Decision**: Use Groq API for automatic relevance scoring instead of manual annotation.

**Relevance Scale:**
- 0: Not relevant
- 1: Slightly relevant
- 2: Moderately relevant
- 3: Highly relevant

```python
def llm_relevance(query: str, doc_text: str) -> int:
    prompt = f"""
    Query: {query}
    Document: {doc_text}
    
    Rate relevance (0-3): 0=not relevant, 3=highly relevant.
    Respond with only the integer.
    """
    
    response = groq_client.chat.completions.create(
        model="mixtral-8x7b-32768",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return int(response.choices[0].message.content.strip())
```

### Evaluation Metrics

#### Precision@K

Fraction of top-K results that are relevant:

```python
def precision_at_k(results: List[str], relevance_map: Dict[str, int], k: int) -> float:
    top_k = results[:k]
    relevant = [doc for doc in top_k if relevance_map.get(doc, 0) > 0]
    return len(relevant) / k
```

#### Recall@K

Fraction of all relevant documents found in top-K:

```python
def recall_at_k(results: List[str], relevance_map: Dict[str, int], k: int) -> float:
    all_relevant = [doc for doc, score in relevance_map.items() if score > 0]
    top_k = results[:k]
    found = [doc for doc in top_k if doc in all_relevant]
    return len(found) / len(all_relevant) if all_relevant else 0
```

#### nDCG@K (Normalized Discounted Cumulative Gain)

Measures ranking quality with position-aware weighting:

```python
def dcg(scores: List[int]) -> float:
    return sum((2**s - 1) / np.log2(i+2) for i, s in enumerate(scores))

def ndcg_at_k(results: List[str], relevance_map: Dict[str, int], k: int) -> float:
    top_k_results = results[:k]
    actual_scores = [relevance_map.get(doc, 0) for doc in top_k_results]
    ideal_scores = sorted(relevance_map.values(), reverse=True)[:k]
    
    return dcg(actual_scores) / dcg(ideal_scores) if dcg(ideal_scores) > 0 else 0
```

### Pooling Strategy

To reduce annotation burden:

1. Collect top-K results from all methods (RAG, BM25, TF-IDF, Random)
2. Create union of all results (pool)
3. Annotate only documents in pool
4. Evaluate all methods using same annotations

---

## Key Lessons Learned

### Technical Decisions

1. **Poetry over pip**: Better dependency management, worth the learning curve
2. **Modular architecture**: Separate files for separate concerns made debugging easier
3. **Serial over parallel**: Simpler is often better; avoid premature optimization
4. **Pydantic models**: Type safety catches bugs early
5. **Dual storage**: Milvus for vectors, SQLite for metadata - best of both worlds

### Debugging Patterns

1. **Schema errors**: Always verify field names match across all insertion points
2. **Memory issues**: Process one file at a time, garbage collect aggressively
3. **Embedding mismatches**: Normalize at generation time, not search time
4. **CSV chunking**: Row-level granularity preserves table structure

### Performance Optimizations

1. **Batch embedding**: Process 32 chunks at a time
2. **Batch Milvus insertion**: 100 chunks per batch
3. **Connection pooling**: Reuse Milvus client across requests
4. **Index selection**: HNSW with COSINE for best speed/quality

---

## Production Checklist

- [x] Document processing for all file types
- [x] Configurable chunking strategies
- [x] Embedding generation with normalization
- [x] Milvus schema with all required fields
- [x] SQLite metadata storage
- [x] Error handling and logging
- [x] Evaluation framework with baselines
- [ ] API endpoint for queries
- [ ] Monitoring and observability
- [ ] Automated testing
- [ ] CI/CD pipeline
- [ ] Documentation for end users

---

## Appendix: Complete File Listing

### Core Pipeline Files

1. **pydantic_models.py** - Data models with validation
2. **doc_reader.py** - Multi-format document loader
3. **chunker.py** - Chunking strategies for all file types
4. **embedder.py** - Embedding generation
5. **schema_setup.py** - Milvus schema creation
6. **milvus.py** - Milvus operations wrapper
7. **sqlite_setup.py** - SQLite schema and operations
8. **bulk_upload.py** - Batch processing pipeline
9. **query_engine.py** - Search and retrieval
10. **baseline_evaluation.py** - Evaluation framework

### Configuration Files

- **pyproject.toml** - Poetry dependencies
- **docker-compose.yml** - Milvus deployment
- **.env** - API keys and configuration

### Data Flow Summary

```
Input Documents
    ↓
doc_reader.py (Load)
    ↓
chunker.py (Split)
    ↓
embedder.py (Vectorize)
    ↓
    ├─→ milvus.py (Vector Storage)
    └─→ sqlite_setup.py (Metadata Storage)
    ↓
query_engine.py (Retrieval)
    ↓
baseline_evaluation.py (Quality Assessment)
```

---

## Conclusion

This RAG pipeline represents a complete journey from initial setup to production-ready system with evaluation. Every decision was made deliberately, every error was debugged systematically, and every component was built to be maintainable and extensible.

The modular architecture allows easy swapping of components (e.g., different embedding models, different vector databases) without rewriting the entire system. The dual storage approach provides flexibility for both semantic search and structured queries. The evaluation framework ensures continuous quality monitoring.

### Final Architecture Principles:

1. **Modularity** - One responsibility per file
2. **Type Safety** - Pydantic models everywhere
3. **Observability** - Logging at every step
4. **Testability** - Each component can be tested independently
5. **Scalability** - Batch processing and efficient storage
6. **Maintainability** - Clear documentation and consistent patterns

This documentation serves as both a historical record of the development process and a guide for future developers working on the system.
