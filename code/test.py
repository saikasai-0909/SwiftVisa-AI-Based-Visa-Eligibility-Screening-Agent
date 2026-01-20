import requests

API_URL = "http://127.0.0.1:1234/v1/chat/completions"

payload = {
    "model": "local-llama",
    "messages": [
        {"role": "user", "content": "hi"}
    ]
}

response = requests.post(API_URL, json=payload)

print("\n=== RAW RESPONSE ===")
print(response.json())

print("\n=== MODEL SAYS ===")
print(response.json()["choices"][0]["message"]["content"])
