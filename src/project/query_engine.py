from typing import List
from project.pydantic_models import SearchResult
from project.milvus import MilvusVectorStore

class QueryEngine:
    def __init__(self):
        print("Query engine ready")

    def search(self, query: str, limit: int = 5) -> List[SearchResult]:
        results = MilvusVectorStore.search_by_text(query, limit)
        
        if results:
            print(f"\nFound {len(results)} results:")
            for result in results:
                print(f"\nRank {result.rank}:")
                print(f"Similarity: {result.similarity_score:.3f}")
                print(f"Content: {result.chunk.content[:400]}...")
        else:
            print("No results found")
        
        return results

def search_documents(query: str, limit: int = 5) -> List[SearchResult]:
    engine = QueryEngine()
    return engine.search(query, limit)
