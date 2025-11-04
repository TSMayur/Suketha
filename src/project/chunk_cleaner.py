# src/project/chunk_cleaner.py
"""
Clean and preprocess chunks before embedding to improve retrieval quality.

This module handles:
1. JSON structure extraction
2. Metadata removal
3. Text normalization
4. Quality filtering
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkCleaner:
    """
    Clean and preprocess chunk text for better embedding quality
    """
    
    def __init__(self, min_chunk_length: int = 50, max_chunk_length: int = 2000):
        """
        Initialize the chunk cleaner.
        
        Args:
            min_chunk_length: Minimum characters to keep a chunk
            max_chunk_length: Maximum characters (truncate if longer)
        """
        self.min_chunk_length = min_chunk_length
        self.max_chunk_length = max_chunk_length
        
        # Patterns to remove
        self.noise_patterns = [
            r'\{"url":\s*"[^"]*"',  # URL fields
            r'"snippet":\s*"[^"]*"',  # Snippet fields
            r'"related_links":\s*null',  # Null fields
            r'"title":\s*"[^"]*"',  # Title fields (sometimes)
            r'/url\?q=http[^\s]*',  # Encoded URLs
            r'&sa=U&ved=[^\s]*',  # Google tracking params
            r'\n\s*\n\s*\n',  # Multiple newlines
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.noise_patterns
        ]
    
    def extract_text_from_json(self, text: str) -> str:
        """
        Extract actual text content from JSON structures.
        
        Handles formats like:
        - {"context": [[...], [...]]}
        - {"search_results": [...]}
        - Plain nested arrays
        """
        try:
            # Try to parse as JSON
            data = json.loads(text)
            
            # Extract text from different JSON structures
            if isinstance(data, dict):
                return self._extract_from_dict(data)
            elif isinstance(data, list):
                return self._extract_from_list(data)
            else:
                return str(data)
                
        except (json.JSONDecodeError, TypeError):
            # Not JSON or malformed, return as-is
            return text
    
    def _extract_from_dict(self, data: dict) -> str:
        """Extract text from dictionary structures"""
        texts = []
        
        # Handle common structures
        if "context" in data:
            # Format: {"context": [[title, [sentences]], ...]}
            for item in data.get("context", []):
                if isinstance(item, list) and len(item) >= 2:
                    title = item[0]
                    content = item[1]
                    
                    if isinstance(content, list):
                        # Join sentences
                        texts.append(" ".join(str(s) for s in content))
                    elif isinstance(content, str):
                        texts.append(content)
        
        elif "search_results" in data:
            # Format: {"search_results": [{"snippet": "...", ...}, ...]}
            for result in data.get("search_results", []):
                if isinstance(result, dict):
                    snippet = result.get("snippet", "")
                    if snippet:
                        texts.append(snippet)
        
        else:
            # Generic extraction: get all string values
            texts = self._extract_all_strings(data)
        
        return " ".join(texts)
    
    def _extract_from_list(self, data: list) -> str:
        """Extract text from list structures"""
        texts = []
        
        for item in data:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, list):
                # Recursive extraction
                texts.append(self._extract_from_list(item))
            elif isinstance(item, dict):
                texts.append(self._extract_from_dict(item))
        
        return " ".join(texts)
    
    def _extract_all_strings(self, data: Any) -> List[str]:
        """Recursively extract all strings from nested structure"""
        strings = []
        
        if isinstance(data, str):
            if len(data) > 10 and not data.startswith("http"):
                strings.append(data)
        elif isinstance(data, dict):
            for value in data.values():
                strings.extend(self._extract_all_strings(value))
        elif isinstance(data, list):
            for item in data:
                strings.extend(self._extract_all_strings(item))
        
        return strings
    
    def remove_noise(self, text: str) -> str:
        """
        Remove noise patterns from text.
        
        Removes:
        - URLs and tracking parameters
        - JSON metadata fields
        - Excessive whitespace
        """
        cleaned = text
        
        # Apply all noise removal patterns
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub(' ', cleaned)
        
        # Additional cleaning
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize whitespace
        cleaned = cleaned.strip()
        
        return cleaned
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for better embedding quality.
        
        - Fix encoding issues
        - Normalize unicode
        - Remove special characters
        """
        # Fix common encoding issues
        replacements = {
            'â€"': '—',
            'â€œ': '"',
            'â€': '"',
            'â€™': "'",
            'Ã¼': 'ü',
            'Ã¶': 'ö',
            'Ã¤': 'ä',
            'Ã': 'ß',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove excessive punctuation
        text = re.sub(r'\.{3,}', '...', text)  # Multiple periods
        text = re.sub(r'\s*\.\s*\.\s*\.\s*', '...', text)  # Spaced periods
        
        return text
    
    def clean_chunk(self, chunk_text: str) -> Optional[str]:
        """
        Main cleaning pipeline for a single chunk.
        
        Returns None if chunk should be filtered out.
        """
        # Step 1: Try to extract from JSON
        text = self.extract_text_from_json(chunk_text)
        
        # Step 2: Remove noise patterns
        text = self.remove_noise(text)
        
        # Step 3: Normalize text
        text = self.normalize_text(text)
        
        # Step 4: Quality check
        if len(text) < self.min_chunk_length:
            logger.debug(f"Filtered out short chunk: {len(text)} chars")
            return None
        
        # Truncate if too long
        if len(text) > self.max_chunk_length:
            text = text[:self.max_chunk_length] + "..."
            logger.debug(f"Truncated long chunk to {self.max_chunk_length} chars")
        
        return text
    
    def clean_chunks_batch(self, chunk_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean a batch of chunks.
        
        Args:
            chunk_dicts: List of chunk dictionaries with 'chunk_text' field
            
        Returns:
            Cleaned chunks (filtered and modified)
        """
        cleaned_chunks = []
        filtered_count = 0
        
        for chunk in chunk_dicts:
            original_text = chunk.get('chunk_text', '')
            
            # Clean the text
            cleaned_text = self.clean_chunk(original_text)
            
            if cleaned_text is None:
                filtered_count += 1
                continue
            
            # Update chunk with cleaned text
            chunk['chunk_text'] = cleaned_text
            chunk['chunk_size'] = len(cleaned_text)
            
            # Recalculate tokens (approximate)
            chunk['chunk_tokens'] = len(cleaned_text.split())
            
            cleaned_chunks.append(chunk)
        
        logger.debug(f"Cleaned {len(chunk_dicts)} chunks -> {len(cleaned_chunks)} kept, {filtered_count} filtered")
        
        return cleaned_chunks
    
    def preview_cleaning(self, text: str, max_length: int = 200) -> Dict[str, str]:
        """
        Preview what cleaning will do to a text sample.
        
        Useful for debugging and validation.
        """
        cleaned = self.clean_chunk(text)
        
        return {
            'original': text[:max_length] + ('...' if len(text) > max_length else ''),
            'cleaned': (cleaned[:max_length] + ('...' if len(cleaned) > max_length else '')) if cleaned else '[FILTERED OUT]',
            'original_length': len(text),
            'cleaned_length': len(cleaned) if cleaned else 0,
            'filtered': cleaned is None
        }


# Convenience function for integration
def clean_chunks_before_embedding(chunk_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convenience function to clean chunks before embedding.
    
    Usage in pipeline:
        chunks = clean_chunks_before_embedding(chunk_dicts)
    """
    cleaner = ChunkCleaner(min_chunk_length=50, max_chunk_length=2000)
    return cleaner.clean_chunks_batch(chunk_dicts)


# Test/Demo
if __name__ == "__main__":
    cleaner = ChunkCleaner()
    
    # Test case 1: JSON structure
    test_json = '''{
        "context": [
            ["Dashiell Hammett", ["He was an American author.", "Created Sam Spade."]],
            ["The Maltese Falcon", ["Published in 1929.", "Classic detective novel."]]
        ]
    }'''
    
    print("Test 1: JSON Structure")
    preview = cleaner.preview_cleaning(test_json)
    print(f"Original: {preview['original']}")
    print(f"Cleaned:  {preview['cleaned']}\n")
    
    # Test case 2: Noisy text
    test_noisy = '''{"url": "/url?q=http://example.com&sa=U&ved=123", "snippet": "This is the actual content we want to extract.", "related_links": null}'''
    
    print("Test 2: Noisy Metadata")
    preview = cleaner.preview_cleaning(test_noisy)
    print(f"Original: {preview['original']}")
    print(f"Cleaned:  {preview['cleaned']}\n")
    
    # Test case 3: Clean text
    test_clean = "Dashiell Hammett was an American author of hard-boiled detective novels."
    
    print("Test 3: Already Clean Text")
    preview = cleaner.preview_cleaning(test_clean)
    print(f"Original: {preview['original']}")
    print(f"Cleaned:  {preview['cleaned']}")
