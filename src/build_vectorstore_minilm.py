import json
from pathlib import Path
from langchain_core.documents import Document

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


CHUNKS_PATH = "data/chunks/all_chunks.json"
VECTORSTORE_PATH = "vectorstore/visa_db_minilm"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def build_vectorstore():
    print("🔹 Loading chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    documents = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "visa_type": chunk["metadata"]["visa_type"],
                    "section": chunk["metadata"]["section"],
                    "country": chunk["metadata"].get("country", "UK"),
                    "chunk_id": chunk["chunk_id"],
                }
            )
        )

    print(f"✓ Loaded {len(documents)} documents")

    print("🔹 Loading MiniLM embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},   # change to "cuda" if GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    print("🔹 Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    Path(VECTORSTORE_PATH).parent.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)

    print(f"✅ Vectorstore saved to {VECTORSTORE_PATH}")
    print(f"📐 Vector dimension: {vectorstore.index.d}")


if __name__ == "__main__":
    build_vectorstore()
