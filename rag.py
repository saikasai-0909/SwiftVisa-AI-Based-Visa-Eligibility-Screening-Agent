"""
AI Visa Eligibility Screening – RAG Pipeline
Retriever (FAISS) + LLaMA 3.2 3B (LM Studio)
"""

import os
from openai import OpenAI
from retriever import VisaRetriever

# =====================================================
# Paths
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "visa_faiss.index")
FAISS_METADATA_PATH = os.path.join(BASE_DIR, "visa_faiss_metadata.pkl")

# =====================================================
# LLM Configuration (LM Studio)
# =====================================================
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL_NAME = "meta-llama-3.2-3b-instruct"

TEMPERATURE = 0.2          # low = factual
TOP_K_CHUNKS = 3           # retriever top-k

# =====================================================
# SYSTEM PROMPT (VERY IMPORTANT)
# =====================================================
SYSTEM_PROMPT = """
You are an AI Visa Eligibility Screening Assistant.

Your task:
- Help users understand UK visa eligibility based ONLY on official visa policy documents.

Rules:
- Use ONLY the provided context.
- Do NOT use external or prior knowledge.
- Do NOT guess or assume.
- If information is missing, clearly say:
  "This information is not mentioned in the visa policy."

Response Style:
- Use clear headings.
- Use bullet points where applicable.
- Be concise, factual, and accurate.
- No legal advice. No opinions.
"""

# =====================================================
# Load Retriever
# =====================================================
def load_retriever():
    print("[INFO] Loading FAISS retriever...")
    return VisaRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH
    )

# =====================================================
# Load LLM Client
# =====================================================
def load_llm():
    print("[INFO] Connecting to LM Studio...")
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )

# =====================================================
# RAG Pipeline
# =====================================================
def answer_question(question: str):
    retriever = load_retriever()
    llm = load_llm()

    # -----------------------------
    # Retrieve Top-K Chunks
    # -----------------------------
    retrieved_docs = retriever.retrieve(question, top_k=TOP_K_CHUNKS)

    if not retrieved_docs:
        return "❌ No relevant visa policy found."

    # -----------------------------
    # Build Context (without page numbers)
    # -----------------------------
    context_blocks = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc["metadata"]
        context_blocks.append(
            f"""
--- Document {i} ---
Source: {os.path.basename(meta.get('source', 'Unknown'))}

{doc['text']}
"""
        )

    context = "\n".join(context_blocks)

    # -----------------------------
    # Augmented Prompt
    # -----------------------------
    augmented_prompt = f"""
VISA POLICY CONTEXT:
{context}

USER QUESTION:
{question}
"""

    # -----------------------------
    # LLM Call
    # -----------------------------
    try:
        response = llm.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt},
            ],
            temperature=TEMPERATURE,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"[WARN] LLM unavailable: {e}")
        print("[INFO] Returning retrieved context only")
        return context

# =====================================================
# MAIN (Interactive Chatbot with Loop)
# =====================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🇬🇧 AI VISA ELIGIBILITY SCREENING (RAG)")
    print("=" * 80)
    print("\n💡 Tips:")
    print("   • Ask questions about UK visa requirements, costs, eligibility")
    print("   • Type 'exit' or 'quit' to stop")
    print("   • Type 'help' for example questions")
    print("=" * 80)

    while True:
        question = input("\n❓ Your question (or 'exit'): ").strip()

        # Handle exit commands
        if question.lower() in ['exit', 'quit', 'q', 'bye']:
            print("\n👋 Thank you for using the Visa RAG System!")
            break
        
        # Handle help command
        if question.lower() in ['help', 'h', '?']:
            print("\n📖 Example questions:")
            print("   • What are the eligibility requirements for a student visa?")
            print("   • How much does a graduate visa cost?")
            print("   • Can I work on a visitor visa?")
            print("   • What documents do I need for a health care worker visa?")
            continue
        
        # Validate question length
        if len(question) < 5:
            print("⚠️  Please enter a valid visa question (at least 5 characters)")
            continue

        # Process question
        print("\n🔍 Processing with RAG...\n")
        answer = answer_question(question)
        print("\n" + "=" * 80)
        print("📌 ANSWER")
        print("=" * 80)
        print(answer)
        print("=" * 80)
