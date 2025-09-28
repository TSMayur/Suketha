# src/project/pydantic_models_complete.py

from pydantic import BaseModel, Field, field_serializer
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class DocumentType(str, Enum):
    PDF = "pdf"
    TXT = "txt"
    JSON = "json"
    DOCX = "docx"
    CSV = "csv"
    TSV = "tsv"

class ChunkingMethod(str, Enum):
    RECURSIVE = "recursive"
    JSON = "json"
    SPACY = "spacy"
    NLTK = "nltk"
    MARKDOWN = "markdown_header"
    COMBINED = "combined"

class EmbeddingModel(str, Enum):
    ALL_MPNET_BASE_V2 = "sentence-transformers/all-mpnet-base-v2"

class Document(BaseModel):
    id: str
    content: str
    document_type: DocumentType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str

class Chunk(BaseModel):
    chunk_id: str = Field(..., alias="id")
    doc_id: str
    chunk_index: int
    chunk_text: str
    chunk_size: int
    chunk_tokens: Optional[int] = None
    chunk_method: ChunkingMethod
    chunk_overlap: int
    domain: str = "general"
    content_type: str
    embedding_model: Optional[str] = None
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    @field_serializer('created_at')
    def serialize_dt(self, dt: datetime, _info):
        return dt.isoformat()

class SearchResult(BaseModel):
    """Missing SearchResult model"""
    chunk: Chunk
    similarity_score: float
    distance: float
    rank: int

class ProcessingConfig(BaseModel):
    chunking_method: ChunkingMethod = ChunkingMethod.RECURSIVE
    chunk_size: int = 1024
    chunk_overlap: int = 256