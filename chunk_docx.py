import os
import uuid
import json
from docx import Document

DOCX_PATH = r"C:\Users\SHAIK SAAJITH\OneDrive\Documents\visa_chunks.docx"
OUTPUT_JSON = "visa_chunks.json"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def chunk_text(text: str, chunk_size=900, overlap=150):
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


doc = Document(DOCX_PATH)

# Join all paragraphs
full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
full_text = full_text.replace("\n", " ").strip()

chunks = chunk_text(full_text, CHUNK_SIZE, CHUNK_OVERLAP)

data = []
for i, c in enumerate(chunks):
    data.append({
        "source": os.path.basename(DOCX_PATH),
        "page_number": None,
        "chunk_id": i,
        "uuid": str(uuid.uuid4()),
        "text": c
    })

print("✅ Total DOCX chunks:", len(data))

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Saved chunks to {OUTPUT_JSON}")
