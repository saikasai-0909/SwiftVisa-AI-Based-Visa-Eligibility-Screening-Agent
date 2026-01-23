import os
import shutil
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
# UPDATED: Points to your existing 'pdfs' folder
DATA_PATH = "./pdfs"  
DB_PATH = "./chroma_db"

def create_vector_db():
    print("🚀 Starting Database Creation...")
    
    # 1. CHECK FOLDER
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ Error: The folder '{DATA_PATH}' does not exist.")
        print("Please make sure your PDF files are inside the 'pdfs' folder.")
        return

    # 2. LOAD PDFS
    print(f"📂 Loading PDFs from '{DATA_PATH}'...")
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("⚠️ No PDFs found! Please check if the files end in .pdf")
        return

    print(f"✅ Loaded {len(documents)} pages.")

    # 3. SPLIT TEXT (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} text chunks.")

    # 4. CREATE DATABASE
    print("🧠 Generating Embeddings (This may take a moment)...")
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Clear old database to prevent duplicates
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        print("♻️  Cleared old database files.")

    # Create new DB
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embedding_function, 
        persist_directory=DB_PATH
    )
    
    print(f"🎉 SUCCESS: Database successfully created at '{DB_PATH}'.")
    print("👉 You can now run: streamlit run app.py")

if __name__ == "__main__":
    create_vector_db()