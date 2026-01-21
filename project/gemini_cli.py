import os
import sys
from google import genai
from google.genai.errors import APIError


# Initialize Gemini client
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    sys.exit(1)

# Gemini model to use
MODEL = "gemini-2.5-flash"

def get_gemini_response(prompt: str):
    # Send prompt to Gemini and print response
    print("🤖 Thinking...")
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        # Display formatted output
        print("\n" + "="*50)
        print("✨ Gemini Response ✨")
        print("="*50)
        print(response.text)
        print("="*50 + "\n")

    except APIError as e:
        # Handle Gemini API errors
        print(f"\n❌ API Error: Failed to get response from Gemini. Details: {e}")
    except Exception as e:
        # Handle unexpected errors
        print(f"\n❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Read prompt from command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python gemini_cli.py \"Your question or prompt here\"")
        print("Example: python gemini_cli.py \"Write a Python function for a Fibonacci sequence\"")
        sys.exit(1)

    prompt_text = " ".join(sys.argv[1:])
    
    # Get response from Gemini
    get_gemini_response(prompt_text)