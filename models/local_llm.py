from openai import OpenAI

client = OpenAI(
    base_url="http://10.209.129.217:1234/v1",
    api_key="lm-studio",
    timeout=30.0  # 🔥 ADD THIS
)

def local_llm(prompt: str, temperature: float = 0.0) -> str:
    print("🧠 Sending prompt to local LLM...")

    response = client.chat.completions.create(
        model="llama-3.2-3b-instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a visa policy assistant. "
                    "Answer strictly using the provided policy snippets. "
                    "If information is missing, say so clearly."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    print("🧠 Local LLM finished generation")
    return response.choices[0].message.content.strip()
