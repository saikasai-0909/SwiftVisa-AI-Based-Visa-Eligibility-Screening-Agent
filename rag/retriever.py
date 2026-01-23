"""
Retriever wrapper using Chroma + MiniLM embeddings.
"""
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PERSIST_DIR = os.path.join(PROJ_ROOT, "vector_db", "chroma")

def load_retriever(k: int = 6):
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if not os.path.exists(PERSIST_DIR):
        raise FileNotFoundError(f"Chroma DB not found at {PERSIST_DIR}. Run ingest.build_vectorstore first.")
    store = Chroma(persist_directory=PERSIST_DIR, embedding_function=emb)
    # return as retriever object compatible with langchain
    return store.as_retriever(search_kwargs={"k": k})
