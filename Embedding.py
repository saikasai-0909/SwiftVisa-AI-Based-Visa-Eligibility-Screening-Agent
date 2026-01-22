import os
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_DIR = "data/pdf"
OUTPUT_DIR = "chunk_output"
VECTOR_STORE_DIR = "visa_vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PDF_CONFIGS = [
    {"path": "UK Student visa.pdf", "visa_type": "student", "country": "UK"},
    {"path": "UK Work visa.pdf", "visa_type": "work", "country": "UK"},
    {"path": "UK Visitor visa.pdf", "visa_type": "visitor", "country": "UK"},
    {"path": "Graduate visa - GOV.UK.pdf", "visa_type": "graduate", "country": "UK"},
    {"path": "Health and Care Worker visa - GOV.UK.pdf", "visa_type": "healthcare_worker", "country": "UK"},
]

def load_docs_with_metadata():
    all_docs = []
    base = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base, PDF_DIR)

    for cfg in PDF_CONFIGS:
        file_path = os.path.join(pdf_dir, cfg["path"])

        if not os.path.exists(file_path):
            print(f"[ERROR] File not found → {file_path}")
            continue

        print(f"[INFO] Loading: {file_path}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        for doc in pages:
            doc.metadata["visa_type"] = cfg["visa_type"]
            doc.metadata["country"] = cfg["country"]
            doc.metadata["source"] = cfg["path"]

        all_docs.extend(pages)

    return all_docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(docs)

def save_chunks(chunks):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i, chunk in enumerate(chunks):
        filename = os.path.join(OUTPUT_DIR, f"chunk_{i+1}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write("=== METADATA ===\n")
            for key, value in chunk.metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n=== CONTENT ===\n")
            f.write(chunk.page_content)

    print(f"[SUCCESS] Saved {len(chunks)} chunks to: {OUTPUT_DIR}")

def build_vector_store(chunks):
    print("[INFO] Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME
    )

    print("[INFO] Building FAISS vector database...")
    vector_db = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_db.save_local(VECTOR_STORE_DIR)

    print(f"[SUCCESS] Vector store saved at: {VECTOR_STORE_DIR}")

if __name__ == "__main__":
    pages = load_docs_with_metadata()
    print(f"[INFO] Loaded {len(pages)} pages")

    chunks = chunk_documents(pages)
    print(f"[INFO] Created {len(chunks)} chunks")

    save_chunks(chunks)          
    build_vector_store(chunks)