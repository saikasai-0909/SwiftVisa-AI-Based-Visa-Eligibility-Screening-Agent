import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# Retrieval & model settings

TOP_K = 5                     
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

GEMINI_MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.0   

# Configure Gemini API

genai.configure(api_key=os.getenv("GEMINI_KEY"))

# Load embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# Initialize ChromaDB client

client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

# Load existing collection
collection = client.get_collection("swiftvisa_embeddings")

# User query
query = "What is the eligibility criteria for a graduate visa?"

# RETRIEVER

# Generate query embedding

query_embedding = embedding_model.encode(
    query,
    normalize_embeddings=True
).tolist()

# Query ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=TOP_K,
    include=["documents", "metadatas", "distances"]
)

# Extract results
retrieved_docs = results["documents"][0]
retrieved_metadata = results["metadatas"][0]
retrieved_distances = results["distances"][0]

# Convert cosine distance → similarity
similarity_scores = [1 - d for d in retrieved_distances]
avg_similarity = sum(similarity_scores) / len(similarity_scores)

# Prepare context for LLM
context_blocks = []
used_chunks_info = []

for i, doc in enumerate(retrieved_docs):
    meta = retrieved_metadata[i]
    context_blocks.append(
        f"[Chunk {i+1}]\n{doc}"
    )
    used_chunks_info.append(
        f"Chunk {i+1} | File: {meta.get('file')} | Section: {meta.get('section')}"
    )

context_text = "\n\n".join(context_blocks)

# ================= PROMPT =================

prompt = f"""
You are a UK visa policy assistant.

RULES (STRICT):
- Answer ONLY using the policy content provided below.
- If information is missing, clearly say "Not specified in policy".
- Do NOT use external knowledge.
- Answer in POINT-WISE format.
- Be factual and concise.

QUESTION:
{query}

POLICY CONTENT:
{context_text}

FINAL ANSWER (Point-wise):
"""

# Initialize Gemini model
model = genai.GenerativeModel(GEMINI_MODEL)

# Generate answer
response = model.generate_content(
    prompt,
    generation_config={
        "temperature": TEMPERATURE
    }
)

answer_text = response.text

# ================= CONFIDENCE =================

used_chunks_count = len(retrieved_docs)

confidence_score = (
    0.7 * avg_similarity +
    0.3 * (used_chunks_count / TOP_K)
)

confidence_percentage = round(confidence_score * 100, 2)

# ================= OUTPUT =================


print("\n================= ANSWER =================\n")
print(answer_text)

print("\n============= USED EMBEDDINGS =============\n")
for info in used_chunks_info:
    print("-", info)

print("\n=============== CONFIDENCE ================\n")
print(f"Estimated Confidence: {confidence_percentage}%")
