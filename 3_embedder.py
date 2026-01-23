import json
import os
import shutil
# We use the standard Chroma interface
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document

# --- CONFIGURATION ---
INPUT_FILE = "chunked_visa_data.json"
DB_PATH = "./visa_db"
EMBED_MODEL = "nomic-embed-text"  # Make sure you pulled this in Ollama

def create_vector_db():
    # 1. Load the Chunks
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} missing. Run Step 2 first.")
        return

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} chunks. Preparing to embed...")

    # 2. Convert JSON objects back to LangChain Documents
    documents = []
    for item in data:
        doc = Document(
            page_content=item['text'],
            metadata=item['metadata'], 
            id=item['id']  # We explicitly pass the ID here
        )
        documents.append(doc)

    # 3. Clear old DB (Optional - keeps things fresh)
    if os.path.exists(DB_PATH):
        print("Removing old database to ensure a fresh start...")
        shutil.rmtree(DB_PATH)

    # 4. Initialize Embedding Model
    print(f"Connecting to Ollama ({EMBED_MODEL})...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    # 5. Create and Persist Database
    # This effectively "saves" the math vectors to your hard drive
    print("Embedding data... (This might take 1-2 minutes)")
    
    # We process in batches automatically
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("\n[SUCCESS] Vector Database created at: " + DB_PATH)
    print("You are now ready to run the Chat Agent (main.py)!")

if __name__ == "__main__":
    create_vector_db()