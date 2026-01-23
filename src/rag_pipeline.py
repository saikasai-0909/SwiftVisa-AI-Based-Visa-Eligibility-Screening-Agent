import os
import time
from pathlib import Path
from typing import List, Dict

from huggingface_hub import InferenceClient
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ===================== HUGGING FACE SETUP =====================
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def get_client():
    """Gets the HF client using the terminal environment variable"""
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("⚠️ Warning: HUGGINGFACE_API_KEY not found in environment variables.")
    return InferenceClient(api_key=api_key)

client = get_client()

# ===================== RAG PIPELINE =====================
class EnhancedVisaRAGPipeline:
    def __init__(
        self,
        vectorstore_path="vectorstore/visa_db_minilm",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        top_k=3,
        temperature=0.0,
        top_p=0.9,
    ):
        self.vectorstore_path = Path(vectorstore_path)
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.temperature = temperature
        self.top_p = top_p

        print("=" * 70)
        print("ENHANCED VISA RAG PIPELINE (Hugging Face + Llama 3.1)")
        print("=" * 70)
        print(f"LLM Model       : {MODEL_ID}")
        print(f"Embedding Model : {self.embedding_model}")
        print(f"Vectorstore Path: {self.vectorstore_path}")
        print(f"Top-K Retrieval : {self.top_k}")
        print("=" * 70 + "\n")

        self.embeddings = self._load_embeddings()
        self.vectorstore = self._load_vectorstore()

        print("✓ Pipeline ready!\n")

    def _load_embeddings(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        return embeddings

    def _load_vectorstore(self):
        if not self.vectorstore_path.exists():
            raise FileNotFoundError(f"Vector store not found at {self.vectorstore_path}")

        vectorstore = FAISS.load_local(
            str(self.vectorstore_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore

    def _format_docs(self, docs):
        return "\n\n".join([f"[Document {i}]\n{doc.page_content}" for i, doc in enumerate(docs, 1)])

    def _extract_citations(self, docs) -> List[Dict]:
        citations = []
        for i, doc in enumerate(docs, 1):
            citations.append({
                "citation_id": i,
                "visa_type": doc.metadata.get("visa_type", "N/A"),
                "section": doc.metadata.get("section", "N/A"),
                "country": doc.metadata.get("country", "UK"),
                "chunk_id": doc.metadata.get("chunk_id", "N/A"),
                "full_content": doc.page_content,
            })
        return citations

    def _build_prompt(self, context: str, question: str) -> str:
        return f"""You are an expert UK immigration visa eligibility officer.
STRICT RULES:
- Use ONLY the provided policy documents.
- If the answer is missing, say: "I don't have enough information in the provided documents."

POLICY DOCUMENTS:
{context}

QUESTION:
{question}

ANSWER:"""

    def _extract_visa_type(self, question: str) -> str:
        question_lower = question.lower()
        visa_types = ["Graduate Visa", "Student Visa", "Skilled Worker Visa", "Health & Care Visa", "Visitor Visa"]
        for visa_type in visa_types:
            if visa_type.lower() in question_lower:
                return visa_type
        return None

    def query(self, question: str, show_sources: bool = True) -> Dict:
        # 1. Retrieve
        visa_type = self._extract_visa_type(question)
        if visa_type:
            retrieved_docs = self.vectorstore.similarity_search(question, k=self.top_k, filter={"visa_type": visa_type})
        else:
            retrieved_docs = self.vectorstore.similarity_search(question, k=self.top_k)
        
        context = self._format_docs(retrieved_docs)
        prompt = self._build_prompt(context, question)

        # 2. Generate (Hugging Face)
        try:
            response = client.chat_completion(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            if "503" in str(e):
                print("⌛ Model loading on HF servers... retrying in 10s")
                time.sleep(10)
                return self.query(question, show_sources)
            raise e

        # 3. Format Output
        citations = self._extract_citations(retrieved_docs)
        result = {
            "question": question,
            "answer": answer,
            "citations": citations,
            "num_sources": len(retrieved_docs),
        }

        if show_sources:
            self._display_result(result)
        return result

    def _display_result(self, result: Dict):
        print("\n" + "=" * 70 + "\nANSWER\n" + "=" * 70)
        print(result["answer"])
        print(f"\nSources used: {result['num_sources']}")

# ===================== MAIN =====================
if __name__ == "__main__":
    rag = EnhancedVisaRAGPipeline()
    while True:
        q = input("\n🔍 Your Question: ").strip()
        if q.lower() in ["quit", "exit", "q"]: break
        if q: rag.query(q)