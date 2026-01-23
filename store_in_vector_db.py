import json
import numpy as np
import faiss
import pickle

# Load the data with embeddings
print("Loading data from JSONL file...")
data = []
with open('combined_visa_with_embeddings.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data)} chunks")

# Prepare data for FAISS
print("Preparing embeddings for FAISS...")
embeddings = []
documents = []
metadatas = []

for i, item in enumerate(data):
    embeddings.append(item['embedding'])
    documents.append(item['text'])
    metadatas.append({
        'id': item['id'],
        'index': i,
        **item.get('metadata', {})
    })

# Convert to numpy array
embeddings_array = np.array(embeddings, dtype='float32')
print(f"Embeddings shape: {embeddings_array.shape}")

# Create FAISS index
dimension = embeddings_array.shape[1]
print(f"Creating FAISS index with dimension: {dimension}")

# Using IndexFlatL2 for exact search (L2 distance)
# For larger datasets, consider IndexIVFFlat or IndexHNSWFlat
index = faiss.IndexFlatL2(dimension)

# Add vectors to the index
print("Adding vectors to FAISS index...")
index.add(embeddings_array)

# Save the FAISS index
print("Saving FAISS index...")
faiss.write_index(index, "visa_faiss.index")

# Save the metadata and documents separately
print("Saving metadata and documents...")
with open('visa_faiss_metadata.pkl', 'wb') as f:
    pickle.dump({
        'documents': documents,
        'metadatas': metadatas
    }, f)

print(f"\nSuccessfully stored {len(data)} documents in FAISS!")
print(f"Index location: ./visa_faiss.index")
print(f"Metadata location: ./visa_faiss_metadata.pkl")
print(f"\nIndex stats:")
print(f"- Dimension: {dimension}")
print(f"- Total vectors: {index.ntotal}")
print(f"- Index type: IndexFlatL2 (exact L2 search)")
