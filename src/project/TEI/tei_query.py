# src/project/tei_query.py

import logging
from typing import List, Dict, Any
from project.tei_pipeline import TEIMilvusPipeline
from project.pydantic_models import ProcessingConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TEIQueryEngine:
    """Query engine for TEI-powered Milvus collections."""
    
    def __init__(self, collection_name: str = "rag_chunks_tei"):
        self.collection_name = collection_name
        config = ProcessingConfig()
        self.pipeline = TEIMilvusPipeline(config)
        
        # Initialize client connection
        if self.pipeline.client is None:
            self.pipeline._setup_milvus_client()
        
        print(f"TEI Query Engine ready for collection: {collection_name}")
    
    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search using TEI embeddings."""
        results = self.pipeline.search_with_tei(query, limit)
        
        if results:
            print(f"\nFound {len(results)} results using TEI embeddings:")
            for result in results:
                print(f"\nRank {result['rank']}:")
                print(f"Similarity: {result['similarity_score']:.3f}")
                print(f"Chunk ID: {result['chunk_id']}")
                print(f"Domain: {result['domain']}")
                print(f"Content: {result['chunk_text'][:400]}...")
        else:
            print("No results found")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.pipeline.get_stats()


def search_with_tei(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search documents using TEI embeddings."""
    engine = TEIQueryEngine()
    return engine.search(query, limit)


def main():
    print("TEI-POWERED DOCUMENT SEARCH")
    print("=" * 25)
    
    engine = TEIQueryEngine()
    stats = engine.get_stats()
    total_chunks = stats.get('total_chunks', 0)
    
    if total_chunks == 0:
        print("No data found. Run tei_pipeline.py first.")
        return
    
    print(f"Database: {total_chunks} chunks (TEI embeddings)")
    print("Type 'quit' to exit")
    
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        
        try:
            search_with_tei(query, limit=3)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye")
        