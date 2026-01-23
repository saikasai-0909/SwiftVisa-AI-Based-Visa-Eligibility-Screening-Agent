"""
Builds FAISS vector store for SwiftVisa
Pipeline:
PDF → Text → Chunks → Embeddings → FAISS index + embeddings.pkl
"""

import os
import json
import pickle
import faiss
import numpy as np
from typing import Dict
from pathlib import Path
from sentence_transformers import SentenceTransformer

from ingest.extract_text import extract_text_from_pdf, save_text
from ingest.chunker import chunk_text, write_jsonl


# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_TEXT = PROJECT_ROOT / "data_text"
CHUNKS_FILE = PROJECT_ROOT / "chunks" / "visa_chunks.jsonl"
VECTOR_DIR = PROJECT_ROOT / "vector_db"
FAISS_PATH = VECTOR_DIR / "faiss.index"
PKL_PATH = VECTOR_DIR / "embeddings.pkl"


# ---------------- PDF Discovery ----------------
def discover_pdfs() -> Dict[str, Dict]:
    files = [f for f in os.listdir(DATA_RAW) if f.lower().endswith(".pdf")]

    mapping = {}

    for f in files:
        meta = {
            "country": "unknown",
            "source": f
        }

        lf = f.lower()

        if "uk" in lf:
            meta["country"] = "UK"
        elif "us" in lf or "usa" in lf or "uscis" in lf:
            meta["country"] = "USA"
        elif "canada" in lf or "ca" in lf:
            meta["country"] = "Canada"
        elif "australia" in lf or "au" in lf:
            meta["country"] = "Australia"

        mapping[f] = meta

    return mapping


# ---------------- FAISS Builder ----------------
def build_faiss(chunks_file: Path):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = []
    metadatas = []

    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
            metadatas.append(obj["metadata"])

    print(f"Embedding {len(texts)} chunks...")

    embeddings = model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))

    VECTOR_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(FAISS_PATH))

    with open(PKL_PATH, "wb") as f:
        pickle.dump({"texts": texts, "metadata": metadatas}, f)

    print("FAISS index and embeddings.pkl created.")


# ---------------- Main Ingestion ----------------
def ingest_all():
    mapping = discover_pdfs()

    if not mapping:
        print("No PDFs found in data_raw/")
        return

    DATA_TEXT.mkdir(exist_ok=True)
    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    for fname, meta in mapping.items():
        pdf_path = DATA_RAW / fname

        print(f"Processing {fname}")

        text = extract_text_from_pdf(str(pdf_path))
        if not text.strip():
            print(f"⚠ No text in {fname}")
            continue

        txt_out = DATA_TEXT / fname.replace(".pdf", ".txt")
        save_text(text, str(txt_out))

        chunks = chunk_text(text, meta)
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks generated.")
        return

    write_jsonl(all_chunks, str(CHUNKS_FILE))
    print(f"Wrote {len(all_chunks)} chunks")

    build_faiss(CHUNKS_FILE)


if __name__ == "__main__":
    ingest_all()
