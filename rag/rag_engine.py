"""
SwiftVisa RAG Engine
Country-aware, Visa-agnostic, FAISS-based Retrieval-Augmented Generation
"""

from typing import List, Tuple
from langchain.schema import Document

from rag.prompt_builder import build_prompt
from rag.retriever_faiss import retrieve_documents


# -------------------------
# Intent Detection
# -------------------------
def detect_intent(query: str) -> str:
    q = query.lower()

    if any(w in q for w in ["eligible", "eligibility", "qualify", "criteria", "requirements"]):
        return "eligibility"

    if any(w in q for w in ["documents", "fees", "processing", "duration", "cost"]):
        return "procedural"

    return "informational"


# -------------------------
# Main RAG Pipeline
# -------------------------
def run_rag(
    query: str,
    llm_callable,
    user_profile: dict | None = None,
    k: int = 6
) -> Tuple[str, List[Document]]:

    # Safety guard
    if not user_profile or not isinstance(user_profile, dict):
        user_profile = {}

    # 1️⃣ Detect intent
    intent = detect_intent(query)

    # 2️⃣ Build metadata filters (ONLY country)
    filters = None
    country = user_profile.get("country")

    if country:
        filters = {
            "country": country
        }

    # 3️⃣ Retrieve relevant chunks
    docs = retrieve_documents(
        query=query,
        k=8 if intent == "eligibility" else k,
        filters=filters
    )

    if not docs:
        return "I could not find this in the official visa policy documents.", []

    # 4️⃣ Build grounded prompt
    prompt = build_prompt(
        query=query,
        docs=docs,
        user_profile=user_profile,
        intent=intent
    )

    # 5️⃣ Generate answer
    answer = llm_callable(prompt)

    return answer, docs
