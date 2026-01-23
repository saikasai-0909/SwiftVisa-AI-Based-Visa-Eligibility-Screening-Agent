import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =========================================================
# STEP 1: LOAD CHUNKS JSON
# =========================================================
CHUNKS_JSON = "visa_chunks.json"       # output from chunking step
INDEX_FILE = "visa_faiss.index"        # FAISS index output
META_FILE = "visa_metadata.pkl"        # metadata output

with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print("✅ Total chunks loaded:", len(chunks))

texts = [c["text"] for c in chunks]

metadatas = []
for c in chunks:
    metadatas.append({
        "source": c.get("source"),
        "page_number": c.get("page_number"),
        "chunk_id": c.get("chunk_id"),
        "uuid": c.get("uuid"),
    })

print("✅ Example text chunk preview:\n", texts[0][:200])


# =========================================================
# STEP 2: LOAD EMBEDDING MODEL (FREE)
# =========================================================
# This is a light + good embedding model
model_name = "all-MiniLM-L6-v2"

print(f"\n⏳ Loading embedding model: {model_name}")
embed_model = SentenceTransformer(model_name)


# =========================================================
# STEP 3: CREATE EMBEDDINGS
# =========================================================
print("\n⏳ Creating embeddings...")
embeddings = embed_model.encode(
    texts,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("✅ Embeddings created.")
print("✅ Embedding shape:", embeddings.shape)  # (num_chunks, vector_dim)


# =========================================================
# STEP 4: CREATE FAISS INDEX
# =========================================================
dimension = embeddings.shape[1]

print("\n⏳ Creating FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype("float32"))

print("✅ FAISS vectors indexed:", index.ntotal)


# =========================================================
# STEP 5: SAVE INDEX + METADATA
# =========================================================
faiss.write_index(index, INDEX_FILE)

with open(META_FILE, "wb") as f:
    pickle.dump(metadatas, f)

print("\n✅ Saved FAISS index:", INDEX_FILE)
print("✅ Saved metadata:", META_FILE)


# =========================================================
# STEP 6: TEST RETRIEVAL (SAMPLE QUERY)
# =========================================================
query = "What are requirements for student visa financial proof?"
query_vec = embed_model.encode([query], convert_to_numpy=True)

k = 5
distances, indices = index.search(query_vec.astype("float32"), k)

print("\n🔎 Query:", query)
print("\n✅ Top Retrieved Chunks:\n")

for rank, idx in enumerate(indices[0], start=1):
    print(f"--- Rank {rank} | Chunk ID: {metadatas[idx]['chunk_id']} | Page: {metadatas[idx]['page_number']} ---")
    print(texts[idx][:500])
    print()
