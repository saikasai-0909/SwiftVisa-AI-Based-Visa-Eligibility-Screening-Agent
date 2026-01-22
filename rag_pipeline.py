"""
RAG Pipeline for SwiftVisa
- Visa-aware retrieval
- Metadata filtering
- Cross-encoder reranking
- Strict eligibility decision enforcement
"""

import requests
import math
from typing import List
from Vector_Builder import FaissVectorStore

# ============================================================
# Confidence Score Calculator
# ============================================================
def compute_confidence(retrieved_chunks: list) -> int:
    if not retrieved_chunks:
        return 0

    scores = [r.get("score", 0) for r in retrieved_chunks]
    avg_score = sum(scores) / len(scores)

    confidence = avg_score * math.log(1 + len(retrieved_chunks)) * 20
    return int(min(100, max(0, confidence)))


# ============================================================
# Official GOV.UK Visa Links
# ============================================================
def get_official_visa_url(visa_type: str) -> str:
    mapping = {
        "student": "https://www.gov.uk/student-visa",
        "skilled_worker": "https://www.gov.uk/skilled-worker-visa",
        "graduate": "https://www.gov.uk/graduate-visa",
        "health_care_worker": "https://www.gov.uk/health-care-worker-visa",
        "visitor": "https://www.gov.uk/standard-visitor-visa"
    }
    return mapping.get(visa_type, "https://www.gov.uk/browse/visas-immigration")


# ============================================================
# Query → Metadata Filter Extraction
# ============================================================
def extract_filters(question: str) -> dict:
    q = question.lower()
    filters = {"country": "uk"}

    if "student" in q:
        filters["visa_type"] = "student"
    elif "skilled" in q:
        filters["visa_type"] = "skilled_worker"
    elif "graduate" in q:
        filters["visa_type"] = "graduate"
    elif "health" in q:
        filters["visa_type"] = "health_care_worker"
    elif "visitor" in q or "visit" in q:
        filters["visa_type"] = "visitor"

    return filters


# ============================================================
# Retriever
# ============================================================
class Retriever:
    def __init__(self, vector_store: FaissVectorStore, top_k=10, rerank=True):
        self.vector_store = vector_store
        self.top_k = top_k
        self.rerank = rerank

    def retrieve(self, query: str):
        filters = extract_filters(query)
        return self.vector_store.query(
            query=query,
            top_k=self.top_k,
            filters=filters,
            rerank=self.rerank
        )


# ============================================================
# Context Builder
# ============================================================
class ContextBuilder:
    def __init__(self, max_chars=3500):
        self.max_chars = max_chars

    def build(self, chunks: List[dict]) -> str:
        context = []
        total = 0

        for c in chunks:
            header = f"[{c['metadata'].get('visa_type', 'unknown')} visa policy]\n"
            text = header + c["text"].strip()

            if total + len(text) > self.max_chars:
                break

            context.append(text)
            total += len(text)

        return "\n\n---\n\n".join(context)


# ============================================================
# Prompt Builder (STRICT)
# ============================================================
class PromptBuilder:
    def __init__(self, prompt_path: str):
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_rules = f.read()

    def build(self, context: str, question: str) -> str:
        return f"""
SYSTEM ROLE:
You are a UK Home Office visa eligibility officer.
You MUST make a decision.
You are NOT allowed to ask questions.

{self.system_rules}

========================
OFFICIAL POLICY CONTEXT
========================
{context}

========================
APPLICANT DETAILS
========================
{question}

========================
MANDATORY OUTPUT FORMAT
========================
Eligibility Status:
<Eligible / Not Eligible / Conditionally Eligible>

Reasons:
- Bullet points

Required Documents:
- Bullet points

Recommendations:
- Bullet points

Explanation:
Clear justification strictly based on the policy.
"""


# ============================================================
# Ollama LLM
# ============================================================
class OllamaLLM:
    def __init__(self, model="llama3.2", temperature=0.1):
        self.model = model
        self.temperature = temperature
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False
        }
        res = requests.post(self.url, json=payload, timeout=120)
        res.raise_for_status()
        return res.json()["response"].strip()


# ============================================================
# RAG Pipeline
# ============================================================
class RAGPipeline:
    def __init__(
        self,
        vector_index_dir: str,
        embedding_model: str,
        ollama_model: str,
        prompt_path: str
    ):
        self.vector_store = FaissVectorStore(
            index_dir=vector_index_dir,
            model_name=embedding_model
        )

        self.retriever = Retriever(self.vector_store)
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder(prompt_path)
        self.llm = OllamaLLM(model=ollama_model)

    def answer(self, question: str):
        retrieved = self.retriever.retrieve(question)

        if not retrieved:
            return {
                "question": question,
                "answer": "I could not find this information in the official visa policy documents.",
                "confidence_score": 0,
                "citations": []
            }

        top_chunks = retrieved[:10]
        context = self.context_builder.build(top_chunks)

        prompt = self.prompt_builder.build(context, question)
        answer = self.llm.generate(prompt)

        confidence = compute_confidence(top_chunks)

        seen = set()
        citations = []

        for r in top_chunks:
            meta = r["metadata"]
            key = (meta.get("source_file"), meta.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)

            visa_type = meta.get("visa_type", "unknown")
            citations.append({
                "source_file": meta.get("source_file"),
                "visa_type": visa_type,
                "chunk_index": meta.get("chunk_index"),
                "official_link": get_official_visa_url(visa_type)
            })

        return {
            "question": question,
            "answer": answer,
            "confidence_score": confidence,
            "citations": citations
        }
