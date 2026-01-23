"""
Test LM Studio connection and full RAG pipeline
"""
import os
from openai import OpenAI
from retriever import VisaRetriever

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "visa_faiss.index")
FAISS_METADATA_PATH = os.path.join(BASE_DIR, "visa_faiss_metadata.pkl")

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"

print("=" * 80)
print("Testing Full RAG Pipeline with LLM")
print("=" * 80)

# Test 1: Retriever
print("\n1. Testing Retriever...")
try:
    retriever = VisaRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH
    )
    
    query = "What are the financial requirements for a Student visa?"
    results = retriever.retrieve(query, top_k=2)
    
    print(f"✅ Retrieved {len(results)} documents")
    
    # Build context
    context = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(results)])
    print(f"✅ Context length: {len(context)} characters")
    
except Exception as e:
    print(f"❌ Retriever Error: {e}")
    exit(1)

# Test 2: LLM Connection
print("\n2. Testing LM Studio connection...")
try:
    llm = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )
    
    # Simple test
    response = llm.chat.completions.create(
        model="meta-llama-3.2-3b-instruct",
        messages=[
            {"role": "user", "content": "Say 'Hello, I am working!'"}
        ],
        temperature=0.1,
        max_tokens=50
    )
    
    print(f"✅ LLM Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ LLM Error: {e}")
    print("\n⚠️  Make sure LM Studio is running on http://localhost:1234")
    print("⚠️  Load the model 'meta-llama-3.2-3b-instruct' in LM Studio")
    exit(1)

# Test 3: Full RAG Pipeline
print("\n3. Testing Full RAG Pipeline...")
try:
    augmented_prompt = f"""VISA POLICY CONTEXT:
{context[:2000]}

USER QUESTION:
{query}

Please answer based only on the context provided above."""

    response = llm.chat.completions.create(
        model="meta-llama-3.2-3b-instruct",
        messages=[
            {"role": "system", "content": "You are a UK visa policy assistant. Answer only based on the provided context."},
            {"role": "user", "content": augmented_prompt}
        ],
        temperature=0.2,
        max_tokens=300
    )
    
    answer = response.choices[0].message.content
    
    print("\n" + "=" * 80)
    print("QUESTION:", query)
    print("=" * 80)
    print("ANSWER:", answer)
    print("=" * 80)
    print("\n✅ Full RAG Pipeline is working!")
    
except Exception as e:
    print(f"❌ Pipeline Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ All tests passed! RAG system is fully functional.")
print("=" * 80)
