from project.milvus import MilvusVectorStore

if __name__ == "__main__":
    print("Clearing Milvus collection...")
    MilvusVectorStore.clear_all_data("rag_chunks")
    print("Done!")
