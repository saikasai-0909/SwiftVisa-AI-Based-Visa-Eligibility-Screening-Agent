import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from groq import Groq

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "chroma_vector_db")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

COLLECTION_NAME = "visa_policy_chunks"
TOP_K = 5
TEMPERATURE = 0.2

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading Groq client...")
llm = Groq(
    api_key=os.getenv("GROQ_API_KEY")  
)

chroma_client = chromadb.PersistentClient(
    path=VECTOR_DB_DIR,
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_collection(name=COLLECTION_NAME)

def run_query(query_text: str, visa_type: str):

    print(f"\n🔍 Query: {query_text}")
    print(f"🎯 Visa Type: {visa_type}")

    query_embedding = embedder.encode([query_text]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K * 2
    )

    selected_chunks = []
    context_texts = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        if meta.get("visa_type") == visa_type:
            selected_chunks.append({
                "source_file": meta.get("source_file"),
                "chunk_id": meta.get("chunk_id"),
                "distance": round(dist, 3),
                "text": doc
            })
            context_texts.append(doc)

        if len(selected_chunks) == TOP_K:
            break

    if not selected_chunks:
        raise RuntimeError("No relevant policy chunks found.")

    prompt = f"""
You are a visa eligibility officer.

Rules:
- Use ONLY the policy context below
- Answer ONLY in bullet points
- Do NOT add outside knowledge

Policy Context:
{" ".join(context_texts)}

Question:
{query_text}
"""

    response = llm.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()

    print("\n===== ANSWER =====\n")
    for line in answer.split("\n"):
        if line.strip():
            print(f"- {line.lstrip('-• ')}")

    output = {
        "query": query_text,
        "visa_type": visa_type,
        "top_k": TOP_K,
        "retrieved_chunks": selected_chunks,
        "llm_answer": answer
    }

    out_path = os.path.join(OUTPUT_DIR, "retriever_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n Output saved to: {out_path}")

if __name__ == "__main__":
    run_query(
        query_text="What are the eligibility requirements for student visa?",
        visa_type="Student Visa"
    )
