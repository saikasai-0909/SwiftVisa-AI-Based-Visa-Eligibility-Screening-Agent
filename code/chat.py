import requests

API_URL = "http://127.0.0.1:1234/v1/chat/completions"

print("Chat started! Type 'exit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Ending chat...")
        break

    payload = {
        "model": "local-llama",
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    response = requests.post(API_URL, json=payload)
    reply = response.json()["choices"][0]["message"]["content"]

    print("LLM:", reply)
