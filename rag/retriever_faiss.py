"""
FAISS-based retriever using precomputed embeddings.pkl
"""

import pickle
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.schema import Document


# -------------------------
# Paths
# -------------------------
PROJ_ROOT = Path(__file__).resolve().parent.parent
PKL_PATH = PROJ_ROOT / "vector_db" / "embeddings.pkl"
FAISS_INDEX_PATH = PROJ_ROOT / "vector_db" / "faiss.index"


# -------------------------
# Embedding model
# -------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------
# Loaders
# -------------------------
def _load_data():
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    return data["texts"], data["metadata"]


def _load_faiss_index():
    return faiss.read_index(str(FAISS_INDEX_PATH))


# -------------------------
# Retriever
# -------------------------
def retrieve_documents(
    query: str,
    k: int = 6,
    filters: dict | None = None,
    min_score: float = 0.3
):
    """
    Retrieve top-k relevant documents using FAISS with metadata filtering.
    """

    texts, metadatas = _load_data()
    index = _load_faiss_index()

    query_vec = model.encode([query], normalize_embeddings=True).astype("float32")

    # Over-fetch for filtering
    scores, indices = index.search(query_vec, k * 5)

    # Debug: show what FAISS found
    print("\n🔍 Raw retrieved metadata (before filtering):")
    for i in indices[0][:10]:
        print(metadatas[i])

    docs = []

    for idx, score in zip(indices[0], scores[0]):
        if score < min_score:
            continue

        meta = metadatas[idx]

        # -------- Normalized metadata filtering --------
        if filters:
            mismatch = False
            for key, value in filters.items():
                meta_val = str(meta.get(key, "")).lower().replace("-", "").replace(" ", "")
                filter_val = str(value).lower().replace("-", "").replace(" ", "")

                # Normalize country aliases
                if meta_val in ["unitedstates", "us"]:
                    meta_val = "usa"
                if filter_val in ["unitedstates", "us"]:
                    filter_val = "usa"

                if meta_val != filter_val:
                    mismatch = True
                    break

            if mismatch:
                continue

        docs.append(
            Document(
                page_content=texts[idx],
                metadata={**meta, "score": float(score)}
            )
        )

        if len(docs) >= k:
            break

    return docs
