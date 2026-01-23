import os
import uuid
import json
from pypdf import PdfReader

# ----------------------------
# STEP 1: LOAD PDF
# ----------------------------
PDF_PATH = r"C:\Users\SHAIK SAAJITH\OneDrive\Documents\UK_visa.pdf"
OUTPUT_JSON = "visa_chunks.json"

# ----------------------------
# STEP 2: CONFIG
# ----------------------------
CHUNK_SIZE = 900        # characters per chunk
CHUNK_OVERLAP = 150     # overlap for continuity


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


# ----------------------------
# STEP 3: EXTRACT + CHUNK PAGE WISE
# ----------------------------
reader = PdfReader(PDF_PATH)

all_chunks = []
chunk_id = 0

for page_num, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() or ""
    page_text = page_text.replace("\n", " ").strip()

    if not page_text:
        continue

    page_chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)

    for c in page_chunks:
        all_chunks.append({
            "source": os.path.basename(PDF_PATH),
            "page_number": page_num,
            "chunk_id": chunk_id,
            "uuid": str(uuid.uuid4()),
            "text": c
        })
        chunk_id += 1

print("✅ Total chunks created:", len(all_chunks))

# ----------------------------
# STEP 4: SAVE TO JSON
# ----------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"✅ Saved chunks to {OUTPUT_JSON}")
