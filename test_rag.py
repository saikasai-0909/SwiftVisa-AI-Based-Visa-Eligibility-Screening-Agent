"""
Quick test to verify RAG system is working
"""
import os
from retriever import VisaRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "visa_faiss.index")
FAISS_METADATA_PATH = os.path.join(BASE_DIR, "visa_faiss_metadata.pkl")

print("=" * 80)
print("Testing RAG Retriever...")
print("=" * 80)

try:
    # Load retriever
    print("\n1. Loading retriever...")
    retriever = VisaRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH
    )
    print("✅ Retriever loaded successfully!")
    
    # Test query
    print("\n2. Testing query: 'Student visa requirements'")
    results = retriever.retrieve("Student visa requirements", top_k=3)
    
    print(f"\n✅ Retrieved {len(results)} documents")
    
    # Display results
    for i, doc in enumerate(results, 1):
        print(f"\n--- Document {i} ---")
        print(f"Source: {doc['metadata'].get('source', 'Unknown')}")
        print(f"Score: {doc.get('score', 'N/A')}")
        print(f"Text preview: {doc['text'][:200]}...")
    
    print("\n" + "=" * 80)
    print("✅ RAG System is working correctly!")
    print("=" * 80)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
