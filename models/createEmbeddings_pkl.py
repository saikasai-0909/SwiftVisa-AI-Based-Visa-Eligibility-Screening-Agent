import json
import pickle
from sentence_transformers import SentenceTransformer
from pathlib import Path

CHUNKS_DIR = Path("chunks")                 # where JSONL files are stored
OUTPUT_PATH = Path("vector_db/embeddings.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")


def load_all_chunks():
    """
    Load all chunk JSONL files into memory.
    Each line is expected to be:
    { "id": ..., "text": ..., "metadata": ... }
    """
    chunks = []

    for jsonl_file in CHUNKS_DIR.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                chunks.append(json.loads(line))

    return chunks


def create_embeddings_pkl():
    chunks = load_all_chunks()

    if not chunks:
        raise RuntimeError("❌ No chunks found. Run ingestion first.")

    texts = [c["text"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    data = {
        "embeddings": embeddings,
        "texts": texts,
        "metadata": metadata
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(data, f)

    print(f"✅ embeddings.pkl created with {len(texts)} vectors")


if __name__ == "__main__":
    create_embeddings_pkl()
