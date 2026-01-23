chunks = []

with open(r"C:\Users\SHAIK SAAJITH\OneDrive\Documents\visa_chunks.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

current_chunk = None
current_text = []

for line in lines:
    line = line.strip()

    # Detect metadata dictionary line
    if line.startswith("{") and "source" in line:
        # Save previous chunk
        if current_chunk:
            current_chunk["text"] = " ".join(current_text).strip()
            chunks.append(current_chunk)

        # Start new chunk
        current_chunk = eval(line)
        current_text = []

    else:
        # Collect text lines
        if line:
            current_text.append(line)

# Save last chunk
if current_chunk:
    current_chunk["text"] = " ".join(current_text).strip()
    chunks.append(current_chunk)

print("Total chunks loaded:", len(chunks))


# ----------------------------
# STEP 2: EXTRACT TEXT + METADATA
# ----------------------------

texts = []
metadatas = []

for c in chunks:
    texts.append(c["text"])
    metadatas.append({
        "source": c["source"],
        "page": c["page_number"],
        "chunk_id": c["chunk_id"],
        "uuid": c["uuid"]
    })

print("Sample chunk text:\n", texts[0][:200])


# ----------------------------
# STEP 3: FREE LOCAL EMBEDDINGS
# ----------------------------

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(texts)

print("Embedding vector size:", embeddings.shape[1])


# ----------------------------
# STEP 4: INDEXING (FAISS)
# ----------------------------

import faiss
import numpy as np

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings).astype("float32"))

print("Total vectors indexed:", index.ntotal)


# ----------------------------
# STEP 5: QUERY
# ----------------------------

query = "Who can apply for a Graduate visa?"
query_embedding = model.encode([query])

# ----------------------------
# STEP 6: RETRIEVER
# ----------------------------

k = 3
distances, indices = index.search(
    np.array(query_embedding).astype("float32"),
    k
)


# ----------------------------
# STEP 7: OUTPUT
# ----------------------------

print("\n Retrieved Chunks:\n")

for idx in indices[0]:
    print(texts[idx])
    print("Source:", metadatas[idx])
    print("-" * 80)


# ----------------------------
# STEP 8: BUILD CONTEXT
# ----------------------------

context = " ".join([texts[i] for i in indices[0]])

print("\n Retrieved Context:\n")
print(context[:500])  # preview


# ----------------------------
# STEP 9: LOAD LOCAL LLM
# ----------------------------

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="distilgpt2"
)


# ----------------------------
# STEP 10: PROMPT LLM
# ----------------------------

prompt = f"""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}

Answer:
"""


# ----------------------------
# STEP 11: GENERATE ANSWER
# ----------------------------

response = generator(
    prompt,
    max_length=300,
    num_return_sequences=1
)

print("\n FINAL ANSWER:\n")
print(response[0]["generated_text"])
