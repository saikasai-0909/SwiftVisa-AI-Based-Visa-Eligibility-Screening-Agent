# chunk_and_save.py
from pathlib import Path
import json
import tiktoken   
from typing import List

TXT_DIR = Path("data/txt")
CHUNKS_DIR = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

def char_chunk(text: str, chunk_size=500, overlap=100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]

def token_chunk(text: str, chunk_tokens=700, overlap_tokens=150, encoding_name="cl100k_base"):
    enc = tiktoken.get_encoding(encoding_name)
    toks = enc.encode(text)
    chunks = []
    i = 0
    while i < len(toks):
        chunk = toks[i:i+chunk_tokens]
        chunks.append(enc.decode(chunk))
        i += chunk_tokens - overlap_tokens
    return [c for c in chunks if len(c) > 20]

def main():
    for txt_file in TXT_DIR.glob("*.txt"):
        text = txt_file.read_text(encoding="utf-8")
        chunks = char_chunk(text, chunk_size=1000, overlap=200)
        out = CHUNKS_DIR / (txt_file.stem + ".json")
        out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Saved chunks for", txt_file.name, "->", out)

if __name__ == "__main__":
    main()
