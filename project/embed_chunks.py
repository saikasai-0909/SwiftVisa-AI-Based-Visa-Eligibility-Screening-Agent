import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Input chunks and output embeddings file
INPUT_FILE = "project/chunks.json"
OUTPUT_FILE = "project/chunks_embeddings.json"

# Load sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load chunked text data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Generate embeddings for each chunk

for chunk in tqdm(chunks, desc="Embedding chunks"):
    embedding = model.encode(
        chunk["text"],
        normalize_embeddings=True   # cosine similarity ready
    )
    chunk["embedding"] = embedding.tolist()

# Save chunks with embeddings
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)

print("✅ SUCCESS: Embeddings created")
