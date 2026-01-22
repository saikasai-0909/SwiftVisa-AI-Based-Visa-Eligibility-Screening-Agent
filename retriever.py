import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from openai import OpenAI

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_DIR = os.path.join(BASE_DIR, "data", "pdf")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "visa_vector_store")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL_NAME = "meta-llama-3.2-3b-instruct"

PDF_CONFIGS = [
    {"path": "UK Student visa.pdf", "visa_type": "student", "country": "UK"},
    {"path": "UK Work visa.pdf", "visa_type": "work", "country": "UK"},
    {"path": "UK Visitor visa.pdf", "visa_type": "visitor", "country": "UK"},
    {"path": "Graduate visa - GOV.UK.pdf", "visa_type": "graduate", "country": "UK"},
    {
        "path": "Health and Care Worker visa - GOV.UK.pdf",
        "visa_type": "healthcare_worker",
        "country": "UK",
    },
]
SYSTEM_PROMPT = """
You are a UK Visa Eligibility Screening Assistant. Assess visa eligibility based on the policy documents provided.

Instructions:
- Read the visa policy provided carefully
- Review the applicant's information against the policy
- Explain your assessment in a natural, conversational way
- Reference what the policy says and how the applicant meets or doesn't meet it
- Provide clear reasoning, not rigid checklists

OUTPUT FORMAT:

Status: ELIGIBLE or NOT ELIGIBLE

Reason:
Explain the applicant's eligibility assessment. Start by stating what the policy requires, then explain how the applicant's situation aligns with those requirements. Be thorough, natural, and conversational in your explanation. Reference specific requirements from the policy and how they apply to this case.

Example tone: "The policy requires X. The applicant has provided Y, which meets this requirement because... Additionally, the applicant needs to have Z for the next requirement..."
"""


def load_docs_with_metadata():
    documents = []

    for cfg in PDF_CONFIGS:
        file_path = os.path.join(PDF_DIR, cfg["path"])

        if not os.path.exists(file_path):
            print(f"[ERROR] Missing file → {file_path}")
            continue

        print(f"[INFO] Loading PDF → {cfg['path']}")
        loader = PyPDFLoader(file_path)
        pages = loader.load()

        for page in pages:
            page.metadata["visa_type"] = cfg["visa_type"]
            page.metadata["country"] = cfg["country"]
            page.metadata["source"] = cfg["path"]

        documents.extend(pages)

    return documents


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_documents(documents)


# Cache for embeddings to avoid reloading
_cached_embeddings = None

def get_embeddings():
    """Load embeddings only once and cache them"""
    global _cached_embeddings
    if _cached_embeddings is None:
        print("[INFO] ⏳ Loading embedding model (this takes 30-60 seconds on first load)...")
        print(f"[INFO] Model: {EMBEDDING_MODEL_NAME}")
        try:
            _cached_embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},  # Use CPU to avoid GPU memory issues
                encode_kwargs={"normalize_embeddings": True}
            )
            print("[SUCCESS] ✅ Embedding model loaded and cached!")
        except Exception as e:
            print(f"[ERROR] ❌ Failed to load embedding model: {e}")
            raise
    return _cached_embeddings


def build_vector_store(chunks):
    print("[INFO] Loading embedding model...")
    embeddings = get_embeddings()

    print("[INFO] Creating FAISS vector database...")
    vector_db = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    vector_db.save_local(VECTOR_STORE_DIR)

    print("[SUCCESS] Vector database created.")

def load_retriever():
    embeddings = get_embeddings()

    vector_db = FAISS.load_local(
        VECTOR_STORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    return retriever

def load_llm():
    client = OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )
    return client

def rag_answer(question, retriever, llm_client, visa_type: str | None = None, critical_flags: list | None = None):
    # Expand query with key terms to help retrieval focus on suitability sections
    extra_terms = [
        "visa eligibility requirements",
        "suitability",
        "refusal",
        "english language",
        "financial requirement",
        "documents",
    ]

    critical_flags = critical_flags or []
    if any("CRIMINAL HISTORY" in flag for flag in critical_flags):
        extra_terms.extend([
            "criminal history",
            "criminal record",
            "good character",
            "spent convictions",
        ])

    expanded_query = f"{question}\n\nKeywords: {', '.join(extra_terms)}"

    docs = retriever.invoke(expanded_query)

    if not docs:
        return "Status: UNDETERMINED\n\nNo relevant visa policy text was retrieved."

    # Filter by visa_type FIRST (before logging)
    if visa_type:
        filtered_docs = [d for d in docs if d.metadata.get("visa_type") == visa_type]
        if filtered_docs:
            docs = filtered_docs[:4]  # Cap at 4 docs for this visa type
        else:
            # Fallback: use any docs if none match the visa type
            docs = docs[:4]
    else:
        docs = docs[:4]

    # Log retrieved documents for debugging
    print(f"\n[RAG] Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        visa = doc.metadata.get('visa_type', 'unknown')
        print(f"  [{i+1}] Visa Type: {visa} | Source: {doc.metadata.get('source', 'policy')} | Length: {len(doc.page_content)} chars")

    context = "\n\n".join(
        f"[SOURCE: {doc.metadata.get('source', 'policy')}]\n{doc.page_content}"
        for doc in docs
    )

# ---- SAFETY: limit context size to prevent LLM crash ----
    MAX_CONTEXT_CHARS = 3000
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]
        print(f"[WARNING] Context truncated to {MAX_CONTEXT_CHARS} chars")

    prompt = f"""
VISA POLICY CONTEXT:
{context}

APPLICANT QUERY:
{question}
"""

    try:
        response = llm_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] LLM API call failed: {str(e)}")
        return f"Status: UNDETERMINED\n\nReason:\n- System Error: The AI model crashed. Try: 1) Reload the model in LM Studio, 2) Use a smaller/more stable model, or 3) Switch to a cloud API.\n\nError Details: {str(e)}"


if __name__ == "__main__":

    if not os.path.exists(VECTOR_STORE_DIR):
        docs = load_docs_with_metadata()
        print(f"[INFO] Pages loaded: {len(docs)}")

        chunks = chunk_documents(docs)
        print(f"[INFO] Chunks created: {len(chunks)}")

        build_vector_store(chunks)
        
    retriever = load_retriever()
    llm_client = load_llm()

    print("\n=== VISA RAG SYSTEM READY ===")

    while True:
        question = input("\nAsk a visa question (type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        answer = rag_answer(question, retriever, llm_client)
        print("\nANSWER:\n", answer)