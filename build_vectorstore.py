# build_vectorstore.py  

import json
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/faiss_index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

all_texts = []
all_metadata = []

print("Loading chunks...")

for file in CHUNKS_DIR.glob("*.json"):
    chunks = json.loads(file.read_text(encoding="utf-8"))
    for idx, chunk in enumerate(chunks):
        all_texts.append(chunk)
        all_metadata.append({"source": file.stem, "chunk_id": idx})

print("Generating embeddings...")
embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=True)
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, str(INDEX_DIR / "index.faiss"))

np.save(str(INDEX_DIR / "embeddings.npy"), embeddings)
with open(INDEX_DIR / "texts.json", "w", encoding="utf-8") as f:
    json.dump(all_texts, f, ensure_ascii=False, indent=2)

with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(all_metadata, f, ensure_ascii=False, indent=2)

print("Vectorstore saved at:", INDEX_DIR)

