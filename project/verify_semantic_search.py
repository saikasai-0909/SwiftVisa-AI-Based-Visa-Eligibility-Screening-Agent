import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Chroma
client = chromadb.PersistentClient(path="project/chroma_db")
collection = client.get_collection("swiftvisa_embeddings")

# Test query
query = "Can I apply for Graduate visa after completing my degree?"

# Generate query embedding
query_embedding = model.encode(query, normalize_embeddings=True).tolist()

# Perform similarity search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)

# Display results
print("\n🔍 QUERY:", query)
print("\nTOP MATCHES:\n")
for i, doc in enumerate(results["documents"][0], start=1):
    print(f"{i}. {doc[:300]}...\n")
