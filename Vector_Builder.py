"""
src/vector_builder.py

Optimized FAISS Vector Store for SwiftVisa
Enhancements implemented:
1. Embedding model cached (no multiple loads)
4. Embeddings cached → no need to regenerate
5. Embedding float16 quantization
8. Optional Cross-Encoder reranker for higher accuracy
14. Full OOP modular design
"""

import os
import json
import pickle
from typing import List, Dict

import numpy as np
from tqdm import tqdm
import faiss

from sentence_transformers import SentenceTransformer, CrossEncoder


# ============================================================
# 1. Embedding Model Wrapper (cached)
# ============================================================
class EmbeddingModel:
    _instance = None

    def __new__(cls, model_name: str):
        if cls._instance is None:
            print(f"[MODEL] Loading embedding model: {model_name}")
            cls._instance = super().__new__(cls)
            cls._instance.model = SentenceTransformer(model_name)
        return cls._instance

    def encode(self, texts, batch_size=64):
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i + batch_size]
            embs = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(embs)
        return np.vstack(embeddings)


# ============================================================
# 2. Reranker Wrapper (optional)
# ============================================================
class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6-v2"):
        print(f"[MODEL] Loading reranker: {model_name}")
        self.reranker = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[str]):
        pairs = [[query, c] for c in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked


# ============================================================
# 3. FAISS Vector Store Class
# ============================================================
class FaissVectorStore:
    def __init__(self, index_dir: str, model_name: str = "thenlper/gte-base"):
        self.index_dir = index_dir
        self.model_name = model_name
        os.makedirs(index_dir, exist_ok=True)

        # file paths
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "metadatas.pkl")
        self.text_path = os.path.join(index_dir, "texts.pkl")
        self.emb_path = os.path.join(index_dir, "embeddings.npy")

        self.index = None
        self.metadatas = None
        self.texts = None

        # cached embedding model
        self.embedder = EmbeddingModel(model_name)

    # -------------------------------------------
    # Normalize vectors (IP search requires L2 norm)
    # -------------------------------------------
    @staticmethod
    def normalize(x: np.ndarray):
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        norm[norm == 0] = 1e-12
        return x / norm

    # -------------------------------------------
    # Build FAISS index (with cached embeddings)
    # -------------------------------------------
    def build(self, chunks_jsonl: str, batch_size=64):
        docs = []
        with open(chunks_jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                docs.append(json.loads(line))

        self.texts = [d["text"] for d in docs]
        self.metadatas = [d["metadata"] for d in docs]
        print(f"[INFO] Loaded {len(self.texts)} chunks")

        # ----- Load or Compute Embeddings -----
        if os.path.exists(self.emb_path):
            print("[CACHE] Loading existing embeddings...")
            embs = np.load(self.emb_path)
        else:
            print("[EMB] Computing embeddings...")
            embs = self.embedder.encode(self.texts, batch_size=batch_size)
            embs = embs.astype("float16")  # quantization for speed + memory
            np.save(self.emb_path, embs)

        # Normalize for cosine similarity
        embs = self.normalize(embs.astype("float32"))

        dim = embs.shape[1]
        print(f"[INFO] Vector dimension: {dim}")

        # Build FAISS index
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)

        # Save index + metadata
        faiss.write_index(self.index, self.index_path)
        pickle.dump(self.metadatas, open(self.meta_path, "wb"))
        pickle.dump(self.texts, open(self.text_path, "wb"))

        print(f"[OK] FAISS index created at: {self.index_path}")

    # -------------------------------------------
    # Load FAISS index
    # -------------------------------------------
    def load(self):
        self.index = faiss.read_index(self.index_path)
        self.metadatas = pickle.load(open(self.meta_path, "rb"))
        self.texts = pickle.load(open(self.text_path, "rb"))
        print("[INFO] Index loaded successfully")

    # -------------------------------------------
    # Query FAISS (FILTERED + RERANKED)
    # -------------------------------------------
    def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict = None,
        rerank: bool = False
    ):
        if self.index is None:
            self.load()

        # Encode query
        q_emb = self.embedder.model.encode([query], convert_to_numpy=True)
        q_emb = self.normalize(q_emb.astype("float32"))

        # Over-fetch for better recall
        D, I = self.index.search(q_emb, top_k * 3)

        results = []
        for score, idx in zip(D[0], I[0]):
            meta = self.metadatas[idx]

            # -------- Metadata Filtering --------
            if filters:
                for k, v in filters.items():
                    if meta.get(k) != v:
                        break
                else:
                    results.append({
                        "score": float(score),
                        "metadata": meta,
                        "text": self.texts[idx]
                    })
            else:
                results.append({
                    "score": float(score),
                    "metadata": meta,
                    "text": self.texts[idx]
                })

            if len(results) >= top_k:
                break

        # -------- Cross-Encoder Reranking --------
        if rerank and results:
            reranker = Reranker()
            texts = [r["text"] for r in results]
            scores = reranker.reranker.predict([[query, t] for t in texts])

            for r, s in zip(results, scores):
                r["score"] = float(s)

            results = sorted(results, key=lambda x: x["score"], reverse=True)

        return results



if __name__ == "__main__":
    store = FaissVectorStore(
        index_dir=r"C:\\Project\vector_db\faiss",
        model_name="thenlper/gte-base"
    )

    store.build(
        chunks_jsonl=r"C:\\Project\\Data\\all_chunks.jsonl",
        batch_size=64
    )

    print("Index build complete!")
