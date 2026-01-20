import json
import requests
import os

API_URL = "http://127.0.0.1:1234/v1/chat/completions"

CHUNK_PATH = r"C:\\Users\\kriti\\OneDrive\\Desktop\\Infosys\\chunks"   # YOUR PATH


def load_chunks(visa_type):
    file_path = os.path.join(CHUNK_PATH, f"{visa_type}.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ask(visa_type, question):
    chunks = load_chunks(visa_type)

    # TEMP METHOD: use first chunks for reply
    context = "\n\n---\n\n".join(chunk["text"] for chunk in chunks[:6])

    prompt = f"""
    You are a helpful UK visa assistant.
    Answer ONLY based on context below.

    Context:
    {context}

    Question: {question}
    """

    response = requests.post(API_URL, json={
        "model": "local-llama",
        "messages": [{"role": "user", "content": prompt}]
    })

    return response.json()["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(ask("Graduate", "What is the eligibility for a Graduate Visa?"))
