import json
import faiss
import requests
from sentence_transformers import SentenceTransformer

# Load embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

# Load FAISS index and texts
INDEX_PATH = "data/faiss_index/index.faiss"
TEXTS_PATH = "data/faiss_index/texts.json"

index = faiss.read_index(INDEX_PATH)

with open(TEXTS_PATH, "r", encoding="utf-8") as f:
    texts = json.load(f)

# Retrieve top-K relevant chunks
def retrieve_chunks(question, k=3):
    """
    Converts the user question into an embedding
    and retrieves top-K similar chunks using FAISS
    """
    query_embedding = model.encode([question])
    distances, indices = index.search(query_embedding, k)

    retrieved_chunks = [texts[i] for i in indices[0]]
    return retrieved_chunks

# Remove duplicate lines from context
def remove_duplicate_lines(text):
    seen = set()
    unique_lines = []

    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            unique_lines.append(cleaned)

    return "\n".join(unique_lines)

# Build strict RAG prompt
def build_prompt(context, question):
    return f"""
You are an AI visa eligibility assistant.

STRICT RULES:
- Use ONLY the information provided in the context.
- Do NOT add external knowledge.
- Do NOT write paragraphs.
- Do NOT add notes, disclaimers, summaries, or explanations.
- Extract information ONLY as bullet points.
- Each bullet point must contain ONE requirement.
- If the answer is not found, say:
  "The information is not available in the provided documents."

FORMAT:
- Use hyphen (-) for bullets
- Bullet points ONLY

Context:
{context}

Question:
{question}

Answer:
"""

# Call LM Studio (local LLM)
def ask_lmstudio(prompt):
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "local-model",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2
        }
    )

    result = response.json()

    if "choices" not in result:
        return "Error: No response from LM Studio."

    return result["choices"][0]["message"]["content"]

# Clean final LLM output
def clean_llm_output(answer):
    seen = set()
    cleaned_lines = []

    for line in answer.splitlines():
        line = line.strip()

        # Remove notes or disclaimers
        if line.lower().startswith("note"):
            continue

        # Keep unique bullet points only
        if line and line not in seen:
            seen.add(line)
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines)

# Main execution
if __name__ == "__main__":
    question = input("Ask a question: ")

    # Retrieve chunks
    retrieved_chunks = retrieve_chunks(question, k=3)

    # Build context
    context = "\n\n".join(retrieved_chunks)

    # Remove duplicate context lines
    context = remove_duplicate_lines(context)

    # Build prompt
    prompt = build_prompt(context, question)

    # Ask LLM
    answer = ask_lmstudio(prompt)

    # Final cleaning (dedup + remove notes)
    answer = clean_llm_output(answer)

    print("\nAnswer:\n")
    print(answer)






