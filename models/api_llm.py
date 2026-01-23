"""
API-based LLM helper (OpenAI). Use this if local model is unavailable or you prefer hosted inference.
MIT-compliant sample code — does not include API keys.
"""

from typing import Callable
import os

def openai_callable(api_key: str = None, model: str = "gpt-3.5-turbo"):
    try:
        import openai
    except Exception:
        raise RuntimeError("openai package is not installed. pip install openai to use API LLM.")

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("No OpenAI API key provided. Set OPENAI_API_KEY env var or pass api_key.")

    openai.api_key = api_key

    def llm(prompt: str) -> str:
        # Convert prompt into a single user message for chat completion
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    return llm
