from pymilvus import MilvusClient

COLLECTION_NAME = "rag_chunks"

def view_schema():
    client = MilvusClient(uri="http://localhost:19530")
    info = client.describe_collection(COLLECTION_NAME)
    print("Collection Name:", info["collection_name"])
    print("Description:", info.get("description", ""))
    print("Fields:")
    for f in info["fields"]:
        print(f"  - {f['name']}: {f['type']} (Primary: {f.get('is_primary', False)})")

if __name__ == "__main__":
    view_schema()
