import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VECTOR_DB_DIR = os.path.join(BASE_DIR, "chroma_vector_db")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CHUNKS_FILE = os.path.join(OUTPUT_DIR, "all_visa_chunks.json")
RESULT_FILE = os.path.join(OUTPUT_DIR, "vector_db_output.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "vector_db_summary.json")

COLLECTION_NAME = "visa_policy_chunks"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model loaded.")


client = chromadb.PersistentClient(
    path=VECTOR_DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

if not os.path.exists(CHUNKS_FILE):
    raise FileNotFoundError(
        f"Chunks file not found: {CHUNKS_FILE}\n"
        "Run ChunkingVisa.py first."
    )

print(f"Loading chunks from: {CHUNKS_FILE}")

with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

chunks = data.get("chunks", [])

if not chunks:
    raise ValueError("No chunks found in JSON.")

print(f"Total chunks loaded: {len(chunks)}")

ids = []
documents = []
metadatas = []

for chunk in chunks:
    ids.append(chunk["id"])
    documents.append(chunk["text"])

    metadata = chunk["metadata"].copy()
    metadata["visa_type"] = chunk["visa_type"]
    metadatas.append(metadata)

print("\nCreating embeddings...")

embeddings = model.encode(
    documents,
    batch_size=32,
    show_progress_bar=True
).tolist()

print("Embeddings created.")

collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings
)

print("\n Indexing completed: embeddings stored in ChromaDB")

query_text = "What are the financial requirements for student visa?"
print(f"\nRunning test query: {query_text}")

query_embedding = model.encode([query_text]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

output_data = {
    "query": query_text,
    "results": []
}

for doc, meta, dist in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    output_data["results"].append({
        "retrieved_text": doc,
        "metadata": meta,
        "distance_score": dist
    })

with open(RESULT_FILE, "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

summary_data = {
    "collection_name": COLLECTION_NAME,
    "total_vectors_in_db": collection.count(),
    "embedding_model": "all-MiniLM-L6-v2",
    "test_query": query_text
}

with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, ensure_ascii=False, indent=2)

print("\n Output files saved:")
print(f"- {RESULT_FILE}")
print(f"- {SUMMARY_FILE}")
print(" Total indexed vectors:", collection.count())
