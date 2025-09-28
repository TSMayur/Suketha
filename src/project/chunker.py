# src/project/chunker_optimized.py

import logging
import json
from typing import List, Dict, Any
import tiktoken

from .pydantic_models import Chunk, ChunkingMethod, ProcessingConfig, Document, DocumentType
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    RecursiveJsonSplitter,
    NLTKTextSplitter,
    SpacyTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter
)

logger = logging.getLogger(__name__)

try:
    TOKENIZER = tiktoken.get_encoding("cl100k_base")
except:
    TOKENIZER = None

class DatasetType:
    """Dataset type detection based on content patterns"""
    CORD19 = "CORD-19"
    WIKIHOP = "WikiHop"
    HOTPOTQA = "HotpotQA"
    EU_LAW = "EU_Law"
    APT_NOTES = "APT_Notes"
    ROBUSTQA = "RobustQA"
    MULTIHOPQA = "MultiHopQA"
    GENERAL = "general"

class OptimizedChunkingService:
    """
    Dataset-aware chunking service implementing strategies from the hackathon guide
    """
    
    @staticmethod
    def detect_dataset_type(document: Document) -> str:
        """Detect dataset type based on document content and metadata"""
        content = document.content.lower()
        source = document.source.lower()
        
        if "cord" in source or "covid" in source:
            return DatasetType.CORD19
        elif "wikihop" in source:
            return DatasetType.WIKIHOP
        elif "hotpot" in source:
            return DatasetType.HOTPOTQA
        elif "eu" in source or "legal" in source or "law" in source:
            return DatasetType.EU_LAW
        elif "apt" in source or "threat" in source or "cyber" in source:
            return DatasetType.APT_NOTES
        elif "robust" in source:
            return DatasetType.ROBUSTQA
        elif "multihop" in source:
            return DatasetType.MULTIHOPQA
        
        if "abstract" in content and "methods" in content and "results" in content:
            return DatasetType.CORD19
        elif "article" in content and "section" in content and ("directive" in content or "regulation" in content):
            return DatasetType.EU_LAW
        elif ("ttp" in content or "ioc" in content or "malware" in content):
            return DatasetType.APT_NOTES
        
        return DatasetType.GENERAL

    @staticmethod
    def get_chunking_strategy(dataset_type: str, config: ProcessingConfig) -> Dict[str, Any]:
        """
        Get chunking strategy, but NOW ALWAYS uses the chunk size and overlap from the config.
        """
        
        # All strategies will now use the config values
        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap

        strategies = {
            DatasetType.CORD19: {
                "splitter": "recursive", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                "separators": ["\n\n", "\n", " ", ""],
                "description": "Preserves paragraphs/sections for scientific arguments"
            },
            DatasetType.WIKIHOP: {
                "splitter": "spacy", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                "description": "Fact-centric chunks for multi-hop linking"
            },
            DatasetType.HOTPOTQA: {
                "splitter": "spacy", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                "description": "Precise supporting sentences for reasoning"
            },
            DatasetType.EU_LAW: {
                "splitter": "markdown_header", "headers": ["# Article", "## Section", "### Paragraph"],
                "fallback": { "splitter": "recursive", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap },
                "description": "Preserves article boundaries for legal referencing"
            },
            DatasetType.APT_NOTES: {
                "splitter": "combined",
                "primary": { "splitter": "markdown_header", "headers": ["# TTPs", "## IoCs", "### Indicators"] },
                "fallback": { "splitter": "recursive", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap },
                "description": "Keeps indicators intact while separating narrative"
            },
            DatasetType.ROBUSTQA: {
                "splitter": "spacy", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                "description": "Perturbations stay tied to question context"
            },
            DatasetType.MULTIHOPQA: {
                "splitter": "nltk", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                "add_metadata": True, "description": "Fine-grained chunks for stepwise reasoning"
            },
            DatasetType.GENERAL: {
                "splitter": "recursive", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                "separators": ["\n\n", "\n", " ", ""], "description": "General purpose chunking"
            }
        }
        
        return strategies.get(dataset_type, strategies[DatasetType.GENERAL])

    @staticmethod
    def chunk_document(document: Document, config: ProcessingConfig) -> List[Chunk]:
        """Main chunking method with dataset-aware strategies"""
        
        dataset_type = OptimizedChunkingService.detect_dataset_type(document)
        logger.info(f"Detected dataset type: {dataset_type} for document: {document.id}")
        
        strategy = OptimizedChunkingService.get_chunking_strategy(dataset_type, config)
        logger.info(f"Using strategy: {strategy['description']}")
        
        chunks = OptimizedChunkingService._apply_chunking_strategy(
            document, strategy, dataset_type
        )
        
        return chunks

    @staticmethod
    def _apply_chunking_strategy(
        document: Document, 
        strategy: Dict[str, Any], 
        dataset_type: str
    ) -> List[Chunk]:
        """Apply the selected chunking strategy"""
        
        splitter_type = strategy["splitter"]
        
        try:
            if splitter_type == "recursive":
                return OptimizedChunkingService._recursive_chunking(document, strategy, dataset_type)
            elif splitter_type == "spacy":
                return OptimizedChunkingService._spacy_chunking(document, strategy, dataset_type)
            elif splitter_type == "nltk":
                return OptimizedChunkingService._nltk_chunking(document, strategy, dataset_type)
            elif splitter_type == "markdown_header":
                return OptimizedChunkingService._markdown_header_chunking(document, strategy, dataset_type)
            elif splitter_type == "combined":
                return OptimizedChunkingService._combined_chunking(document, strategy, dataset_type)
            elif splitter_type == "json":
                return OptimizedChunkingService._json_chunking(document, strategy, dataset_type)
            else:
                logger.warning(f"Unknown splitter type: {splitter_type}, using recursive")
                return OptimizedChunkingService._recursive_chunking(document, strategy, dataset_type)
                
        except Exception as e:
            logger.error(f"Error in chunking strategy {splitter_type}: {e}")
            fallback_strategy = {
                "chunk_size": 1024, "chunk_overlap": 256,
                "separators": ["\n\n", "\n", " ", ""]
            }
            return OptimizedChunkingService._recursive_chunking(document, fallback_strategy, dataset_type)

    # ... (The rest of the helper methods _recursive_chunking, _spacy_chunking, etc., remain exactly the same) ...

    @staticmethod
    def _recursive_chunking(document: Document, strategy: Dict[str, Any], dataset_type: str) -> List[Chunk]:
        """Recursive character text splitter - most reliable"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=strategy["chunk_size"],
            chunk_overlap=strategy["chunk_overlap"],
            separators=strategy.get("separators", ["\n\n", "\n", " ", ""])
        )
        texts = splitter.split_text(document.content)
        return OptimizedChunkingService._create_chunks_from_texts(texts, document, dataset_type, strategy)

    @staticmethod
    def _spacy_chunking(document: Document, strategy: Dict[str, Any], dataset_type: str) -> List[Chunk]:
        """SpaCy-based sentence chunking"""
        try:
            splitter = SpacyTextSplitter(
                chunk_size=strategy["chunk_size"],
                chunk_overlap=strategy.get("chunk_overlap", 0)
            )
            texts = splitter.split_text(document.content)
            return OptimizedChunkingService._create_chunks_from_texts(texts, document, dataset_type, strategy)
        except Exception as e:
            logger.warning(f"SpaCy chunking failed: {e}, falling back to recursive")
            return OptimizedChunkingService._recursive_chunking(document, strategy, dataset_type)

    @staticmethod
    def _nltk_chunking(document: Document, strategy: Dict[str, Any], dataset_type: str) -> List[Chunk]:
        """NLTK-based sentence chunking"""
        try:
            splitter = NLTKTextSplitter(
                chunk_size=strategy["chunk_size"],
                chunk_overlap=strategy.get("chunk_overlap", 0)
            )
            texts = splitter.split_text(document.content)
            return OptimizedChunkingService._create_chunks_from_texts(texts, document, dataset_type, strategy)
        except Exception as e:
            logger.warning(f"NLTK chunking failed: {e}, falling back to recursive")
            return OptimizedChunkingService._recursive_chunking(document, strategy, dataset_type)

    @staticmethod
    def _markdown_header_chunking(document: Document, strategy: Dict[str, Any], dataset_type: str) -> List[Chunk]:
        """Markdown header-based chunking for structured documents"""
        try:
            headers_to_split = strategy.get("headers", ["#", "##", "###"])
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split)
            docs = splitter.split_text(document.content)
            
            texts = []
            for doc in docs:
                header_context = " ".join([f"{k}: {v}" for k, v in doc.metadata.items()])
                text = f"{header_context}\n{doc.page_content}" if header_context else doc.page_content
                texts.append(text)
            
            if not texts and "fallback" in strategy:
                return OptimizedChunkingService._apply_chunking_strategy(
                    document, strategy["fallback"], dataset_type
                )
                
            return OptimizedChunkingService._create_chunks_from_texts(texts, document, dataset_type, strategy)
            
        except Exception as e:
            logger.warning(f"Markdown header chunking failed: {e}")
            if "fallback" in strategy:
                return OptimizedChunkingService._apply_chunking_strategy(
                    document, strategy["fallback"], dataset_type
                )
            return OptimizedChunkingService._recursive_chunking(document, strategy, dataset_type)

    @staticmethod
    def _combined_chunking(document: Document, strategy: Dict[str, Any], dataset_type: str) -> List[Chunk]:
        """Combined chunking strategy"""
        try:
            primary_chunks = OptimizedChunkingService._apply_chunking_strategy(
                document, strategy["primary"], dataset_type
            )
            
            if len(primary_chunks) < 3 and "fallback" in strategy:
                fallback_chunks = OptimizedChunkingService._apply_chunking_strategy(
                    document, strategy["fallback"], dataset_type
                )
                return primary_chunks + fallback_chunks
                
            return primary_chunks
            
        except Exception as e:
            logger.warning(f"Combined chunking failed: {e}")
            if "fallback" in strategy:
                return OptimizedChunkingService._apply_chunking_strategy(
                    document, strategy["fallback"], dataset_type
                )
            return OptimizedChunkingService._recursive_chunking(document, strategy, dataset_type)

    @staticmethod
    def _json_chunking(document: Document, strategy: Dict[str, Any], dataset_type: str) -> List[Chunk]:
        """JSON-specific chunking"""
        splitter = RecursiveJsonSplitter(
            max_chunk_size=strategy.get("chunk_size", 3000)
        )
        try:
            if not document.content.strip(): return []
            json_data = json.loads(document.content)
            if not json_data: return []
            texts = [json.dumps(chunk) for chunk in splitter.split_json(json_data=json_data)]
            return OptimizedChunkingService._create_chunks_from_texts(texts, document, dataset_type, strategy)
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"JSON chunking failed for {document.id}: {e}, falling back to recursive")
            return OptimizedChunkingService._recursive_chunking(document, strategy, dataset_type)

    @staticmethod
    def _create_chunks_from_texts(texts: List[str], document: Document, dataset_type: str, strategy: Dict[str, Any]) -> List[Chunk]:
        """Create Chunk objects from text segments with enhanced metadata"""
        chunks = []
        for i, text in enumerate(texts):
            if len(text.strip()) >= 10:
                token_count = len(TOKENIZER.encode(text)) if TOKENIZER else None
                
                chunk = Chunk(
                    id=f"{document.id}_chunk_{i}",
                    doc_id=document.id,
                    chunk_text=text.strip(),
                    chunk_index=i,
                    chunk_method=ChunkingMethod.RECURSIVE,
                    chunk_size=len(text),
                    chunk_tokens=token_count,
                    chunk_overlap=strategy.get("chunk_overlap", 0),
                    content_type=document.document_type.value,
                    domain=dataset_type,
                    embedding_model=None,
                    embedding=None
                )
                chunks.append(chunk)
        return chunks