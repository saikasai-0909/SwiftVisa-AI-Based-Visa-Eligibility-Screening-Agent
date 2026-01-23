import pickle

with open("vector_db/embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print("Embeddings:", len(data["embeddings"]))
print("Texts:", len(data["texts"]))
print("Metadata:", len(data["metadata"]))
