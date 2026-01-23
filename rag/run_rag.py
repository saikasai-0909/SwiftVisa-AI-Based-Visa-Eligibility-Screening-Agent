from rag.rag_engine import run_rag
from models.local_llm import local_llm

print("🚀 Starting RAG...")

query = "What are the eligibility criteria for H1B visa?"

user_profile = {
    "country": "USA",
    "visa_type": "H1B"
}

print("🔍 Running RAG pipeline...")
answer, docs = run_rag(
    query=query,
    llm_callable=local_llm,
    user_profile=user_profile
)

print("✅ LLM responded")

print("\n===== ANSWER =====\n")
print(answer)
