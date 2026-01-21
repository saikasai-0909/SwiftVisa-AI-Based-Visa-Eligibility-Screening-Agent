import json
import os
import chromadb
from chromadb.config import Settings

# -------- PATHS (ABSOLUTE) --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
INPUT_FILE = os.path.join(BASE_DIR, "chunks_embeddings.json")

# Create Chroma directory if not exists
os.makedirs(CHROMA_DIR, exist_ok=True)

print("📁 Chroma path:", CHROMA_DIR)

# -------- LOAD DATA --------
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -------- CREATE PERSISTENT CLIENT --------
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name="swiftvisa_embeddings",
    metadata={"hnsw:space": "cosine"}
)

# -------- STORE VECTORS --------
collection.add(
    ids=[c["chunk_id"] for c in chunks],                # Unique IDs
    embeddings=[c["embedding"] for c in chunks],        # Vector embeddings
    documents=[c["text"] for c in chunks],              # Original text
    metadatas=[
        {
            "file": c["file"],
            "section": c["section_title"]
        }
        for c in chunks
    ]
)

print("✅ Stored vectors:", collection.count())
