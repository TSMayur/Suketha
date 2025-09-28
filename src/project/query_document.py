from project.query_engine import search_documents
from project.milvus import MilvusVectorStore

def main():
    print("DOCUMENT SEARCH")
    print("=" * 20)
    
    stats = MilvusVectorStore.get_stats()
    total_chunks = stats.get('total_chunks', 0)
    
    if total_chunks == 0:
        print("No data found. Run process_document.py first.")
        return
    
    print(f"Database: {total_chunks} chunks")
    print("Type 'quit' to exit")
    
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        
        try:
            search_documents(query, limit=3)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye")
