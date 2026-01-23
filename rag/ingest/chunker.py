"""
Chunk text into JSONL chunks with metadata.
Original, MIT-compliant code.
"""
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

from typing import List, Dict

def chunk_text(text: str, metadata: Dict, chunk_size: int = 700, overlap: int = 150) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    parts = splitter.split_text(text)
    chunks = []
    for i, p in enumerate(parts):
        chunks.append({
            "id": f"{metadata.get('country','na')}_{metadata.get('visa_type','na')}_{i}",
            "text": p,
            "metadata": metadata
        })
    return chunks

def write_jsonl(chunks, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
