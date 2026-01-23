import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Define the path where you saved the DB (Check your ingestion script for this path!)
# Common paths: "./chroma_db", "db", or just local folder
DB_PATH = "./chroma_db" 

def check_database():
    print(f"🕵️ Checking database at: {os.path.abspath(DB_PATH)}")
    
    if not os.path.exists(DB_PATH):
        print("❌ ERROR: The database folder does not exist here.")
        print("   -> Did you run the cleaning/ingestion script?")
        print("   -> did it save to a different folder?")
        return

    # 2. Setup Embeddings (Must be SAME as used in cleaning.py)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 3. Load DB
    try:
        vector_db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
        
        # 4. Count documents
        # Note: Chroma doesn't have a direct .count() in some versions, 
        # so we try a dummy search or get collection.
        count = vector_db._collection.count()
        
        print(f"✅ Database Loaded!")
        print(f"📚 Total Document Chunks Found: {count}")
        
        if count == 0:
            print("⚠️ WARNING: The database exists but is EMPTY.")
            print("   -> Your cleaning script ran but didn't save chunks.")
        else:
            print("🎉 Success! The data is there.")
            
            # Test a retrieval
            print("\n🧪 Testing Retrieval for 'Student Visa'...")
            results = vector_db.similarity_search("Student Visa requirements", k=2)
            for i, doc in enumerate(results):
                print(f"   Result {i+1}: {doc.page_content[:100]}...")

    except Exception as e:
        print(f"❌ Error loading DB: {e}")

if __name__ == "__main__":
    check_database()