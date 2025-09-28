# src/project/embedder_optimized.py

import torch
import numpy as np
from typing import List, Optional
from project.pydantic_models import Chunk, EmbeddingModel
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

class OptimizedEmbedder:
    _instance = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._model is None:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model once with optimal settings for M2 Mac"""
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        logger.info(f"Loading embedding model on {device}")
        self._model = SentenceTransformer(
            EmbeddingModel.ALL_MPNET_BASE_V2.value,
            device=device
        )
        
        # Optimize for M2 Mac
        if device == "mps":
            # Set optimal batch size for M2 Mac memory
            self.optimal_batch_size = 64
            # Enable MPS optimizations
            torch.backends.mps.allow_tf32 = True
        else:
            self.optimal_batch_size = 32
            
        logger.info(f"Model loaded. Optimal batch size: {self.optimal_batch_size}")
    
    def embed_texts_batch(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """Optimized batch embedding with proper memory management"""
        if not texts:
            return []
            
        batch_size = batch_size or self.optimal_batch_size
        all_embeddings = []
        
        # Process in optimal batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Use sentence-transformers directly for better performance
            with torch.no_grad():
                batch_embeddings = self._model.encode(
                    batch_texts,
                    batch_size=len(batch_texts),
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            
            all_embeddings.extend(batch_embeddings.tolist())
            
            # Clear MPS cache periodically on M2 Mac
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        return all_embeddings
    
    def embed_chunks_optimized(self, chunks: List[Chunk]) -> List[Chunk]:
        """Optimized chunk embedding"""
        if not chunks:
            return []
        
        texts = [chunk.chunk_text for chunk in chunks]
        embeddings = self.embed_texts_batch(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
        return chunks

# Singleton instance
embedder = OptimizedEmbedder()

def embed_chunks(chunks: List[Chunk], encoder=None) -> List[Chunk]:
    """Main embedding function - now uses optimized embedder"""
    return embedder.embed_chunks_optimized(chunks)