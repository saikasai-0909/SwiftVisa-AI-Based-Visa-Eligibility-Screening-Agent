from rag.retriever_faiss import retrieve_faiss

results = retrieve_faiss(
    "What are the eligibility criteria for H1B visa?",
    top_k=5
)

for r in results:
    print("\nScore:", r["score"])
    print("Country:", r["metadata"].get("country"))
    print("Visa:", r["metadata"].get("visa_type"))
    print("Text:", r["text"][:300])
