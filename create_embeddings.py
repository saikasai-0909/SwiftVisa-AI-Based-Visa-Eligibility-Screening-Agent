import json
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the sentence transformer model
print("Loading sentence transformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Read the JSONL file
print("Reading JSONL file...")
data = []
with open('combined_visa_metadata.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line))

print(f"Loaded {len(data)} chunks")

# Extract text from each chunk
texts = [item['text'] for item in data]

# Generate embeddings
print("Generating embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

print(f"Generated embeddings shape: {embeddings.shape}")

# Save embeddings and data
print("Saving embeddings...")
np.save('visa_embeddings.npy', embeddings)

# Save the data with embeddings in a new file
with open('combined_visa_with_embeddings.jsonl', 'w', encoding='utf-8') as f:
    for item, embedding in zip(data, embeddings):
        item['embedding'] = embedding.tolist()
        f.write(json.dumps(item) + '\n')

print("Done! Files created:")
print("- visa_embeddings.npy (numpy array of embeddings)")
print("- combined_visa_with_embeddings.jsonl (JSONL with embeddings included)")
