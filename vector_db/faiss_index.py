import pickle
import faiss
import numpy as np
from pathlib import Path

PKL_PATH = Path("vector_db/embeddings.pkl")
INDEX_PATH = Path("vector_db/faiss.index")


def build_faiss_index():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"].astype("float32")

    dim = embeddings.shape[1]

    # Inner Product index (cosine similarity if vectors are normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    print(f"✅ FAISS index built with {index.ntotal} vectors")


def load_faiss_index():
    return faiss.read_index(str(INDEX_PATH))


if __name__ == "__main__":
    build_faiss_index()
