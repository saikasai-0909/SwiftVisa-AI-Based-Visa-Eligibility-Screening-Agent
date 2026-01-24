import numpy as np
import json

embeddings = np.load("data/faiss_index/embeddings.npy")

emb_list = embeddings.tolist()

output_data = {
    "shape": embeddings.shape,      
    "embeddings": emb_list          
}

with open("embeddings.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print("Saved embeddings with shape to embeddings.json")



