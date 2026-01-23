import os
import json
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHUNKS_FILE = os.path.join(BASE_DIR, "output", "all_visa_chunks.json")
VECTOR_DB_FILE = os.path.join(BASE_DIR, "output", "vector_store.json")

if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(
        f"Chunks file not found: {CHUNKS_FILE}\n"
        "Run ChunkingVisa.py first."
    )

print(f"Loading chunks from: {CHUNKS_FILE}")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data.get("chunks", [])

print(f"Total chunks loaded: {len(chunks)}")

if not chunks:
    raise ValueError("No chunks found! Chunking step failed.")

print("\nLoading embedding model: all-MiniLM-L6-v2")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded successfully.")

print("\nCreating embeddings...")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("Embeddings generated.")

print("\nBuilding vector store JSON...")

vector_store = {
    "model": "all-MiniLM-L6-v2",
    "dimension": embeddings.shape[1],
    "vectors": []
}

for chunk, embedding in zip(chunks, embeddings):
    vector_store["vectors"].append({
        "id": chunk["id"],
        "visa_type": chunk["visa_type"],
        "text": chunk["text"],
        "metadata": chunk["metadata"],
        "embedding": embedding.tolist()
    })

with open(VECTOR_DB_FILE, "w", encoding="utf-8") as f:
    json.dump(vector_store, f, indent=2, ensure_ascii=False)

print("\n EMBEDDINGS CREATED SUCCESSFULLY")
print(f"Saved at: {VECTOR_DB_FILE}")
