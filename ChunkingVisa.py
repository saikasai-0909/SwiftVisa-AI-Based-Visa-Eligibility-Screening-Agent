import os
import json
from PyPDF2 import PdfReader
from datetime import datetime

CONFIG_FILE = "config.json"

if not os.path.exists(CONFIG_FILE):
    raise FileNotFoundError(
        f"{CONFIG_FILE} not found. Create it in the same folder as this script."
    )

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

PDF_FOLDER = config.get("pdf_folder", "DataSets")
OUTPUT_FOLDER = config.get("output_folder", "output")
MAX_WORDS = int(config.get("max_words", 200))
OVERLAP_WORDS = int(config.get("overlap_words", 50))

print("Looking for PDFs in:", os.path.abspath(PDF_FOLDER))

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages_text = []

    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
            if text:
                text = " ".join(text.split())
                pages_text.append(text)
        except Exception as e:
            print(f"[WARN] Error reading page {page_num} in {pdf_path}: {e}")

    return " ".join(pages_text)


def create_word_chunks(
    text: str,
    max_words: int,
    overlap_words: int,
    source_file: str,
    visa_type: str,
):
    words = text.split()
    total_words = len(words)
    chunks = []

    if total_words == 0:
        return chunks

    step = max(max_words - overlap_words, 1)
    index = 0
    chunk_id = 1

    while index < total_words:
        start_idx = index
        end_idx = min(index + max_words, total_words)

        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words)

        chunks.append({
            "id": f"{visa_type.lower().replace(' ', '_')}_chunk_{chunk_id}",
            "visa_type": visa_type,
            "text": chunk_text,
            "metadata": {
                "source_file": source_file,
                "chunk_id": chunk_id,
                "start_word_index": start_idx,
                "end_word_index": end_idx - 1,
                "word_count": len(chunk_words),
                "total_words_in_document": total_words
            }
        })

        index += step
        chunk_id += 1

    return chunks



def process_all_pdfs():
    all_chunks = []
    timestamp = datetime.utcnow().isoformat() + "Z"

    if not os.path.exists(PDF_FOLDER):
        raise FileNotFoundError(f"PDF folder not found: {PDF_FOLDER}")

    pdf_files = [
        f for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        raise RuntimeError("No PDF files found in DataSets folder.")

    print("\n=== STARTING PDF CHUNKING ===\n")

    for filename in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, filename)
        visa_type = os.path.splitext(filename)[0]

        print(f"Processing: {visa_type}")

        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            print(f"[WARN] No text found in {filename}")
            continue

        chunks = create_word_chunks(
            text=text,
            max_words=MAX_WORDS,
            overlap_words=OVERLAP_WORDS,
            source_file=filename,
            visa_type=visa_type
        )

        for chunk in chunks:
            chunk["metadata"]["created_at"] = timestamp
            all_chunks.append(chunk)

        print(f" -> {len(chunks)} chunks created")

    output_path = os.path.join(OUTPUT_FOLDER, "all_visa_chunks.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"chunks": all_chunks}, f, indent=2, ensure_ascii=False)

    print("\n=== CHUNKING COMPLETED ===")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Saved at: {output_path}\n")


if __name__ == "__main__":
    process_all_pdfs()
