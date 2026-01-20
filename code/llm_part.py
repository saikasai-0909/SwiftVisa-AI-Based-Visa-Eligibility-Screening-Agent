import chromadb
import requests
import json
from sentence_transformers import SentenceTransformer
import os
os.makedirs("rag_logs", exist_ok=True)


# ==============================
# PATHS & CONFIG
# ==============================

CHROMA_DB_PATH = r"C:\Users\kriti\OneDrive\Desktop\Infosys\5vectordb..5\chroma_db"
LM_STUDIO_URL = "http://127.0.0.1:1234/v1/chat/completions"
EMBEDDING_MODEL = "intfloat/e5-large-v2"
LLM_MODEL_NAME = "llama-3.2-3b-instruct"

TOP_K = 3

# ==============================
# LOAD EMBEDDING MODEL (QUERY SIDE)
# ==============================

embedder = SentenceTransformer(EMBEDDING_MODEL)

# ==============================
# LOAD CHROMA DB (PERSISTED)
# ==============================

client = chromadb.Client(
    chromadb.config.Settings(
        persist_directory=CHROMA_DB_PATH,
        anonymized_telemetry=False,
        is_persistent=True
    )
)

collection = client.get_or_create_collection("visa_rules")

# ==============================
# LLM PROMPT FUNCTION
# ==============================

def ask_llm(context: str, question: str) -> str:
    prompt = f"""
You are an expert immigration assistant.

You must answer the user’s question using ONLY the information provided in the CONTEXT section below.
Do NOT use any external knowledge.
Do NOT guess or assume anything.
If the information required to answer the question is missing or incomplete, clearly say so.

Your task is to:
1. Read the provided policy text carefully.
2. Extract only the relevant rules.
3. Produce a clear, structured, and accurate answer.

If multiple conditions apply, list them clearly.
If exceptions or variations exist, mention them explicitly.
If eligibility cannot be determined, state what information is missing.

When listing missing information:
- Do NOT repeat user profile details that are already provided.
- Only list missing eligibility-related conditions required by policy.
- For each missing item, explain WHY it is required according to policy.
- Phrase missing items as actionable clarifications the applicant can respond to.

CONTEXT:
--------
{context}

TASK / QUERY:
---------
{question}

Answer strictly in the following format:

Summary:
- 

Detailed Explanations:
- 

Conditions / Variations (if any):
- 

Missing Informations:
- 
"""

    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 700
    }

    response = requests.post(
        LM_STUDIO_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    return response.json()["choices"][0]["message"]["content"]

# ==============================
# MAIN QUERY PIPELINE
# ==============================

from datetime import datetime

def answer_question(query_text: str):
    query_embedding = embedder.encode(
        "query: " + query_text,
        normalize_embeddings=True
    )


    # Step 2: Retrieve across ALL visa types
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=TOP_K
    )

    # Step 3: Build context + citations
    context_blocks = []
    citations = []

    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), start=1
    ):
        citation_id = f"[C{i}]"

        block = f"""
{citation_id}
[VISA TYPE: {meta['visa_type']} | CHUNK: {meta['chunk_index']}]
{doc}
"""
        context_blocks.append(block)

        citations.append({
            "citation_id": citation_id,
            "visa_type": meta["visa_type"],
            "chunk_index": meta["chunk_index"],
            "text": doc
        })

    # Step 4: Join context
    context = "\n\n".join(context_blocks)

    # Step 5: Ask LLM
    final_answer = ask_llm(context, query_text)

    # Step 6: Save EVERYTHING to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = "C:\\Users\\kriti\\OneDrive\\Desktop\\Infosys\\code\\rag_logs"

    filename = os.path.join(
        OUTPUT_DIR,
        f"rag_output_{timestamp}.json"
    )


    output_data = {
        "question": query_text,
        "retrieved_chunks": citations,
        "final_answer": final_answer
    }

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)

    # Step 7: Minimal terminal output
    print(f"✅ RAG output saved successfully to {filename}")
    return final_answer

     


# ==============================
# MAIN FUNCTION
# ==============================

if __name__ == "__main__":
    question = "What is the eligibility nedded for a Graduate visa?"
    

    answer = answer_question(question)
    print("\nFINAL ANSWER:\n")
    print(answer)
