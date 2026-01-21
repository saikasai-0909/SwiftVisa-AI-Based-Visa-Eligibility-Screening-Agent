import os
import re
import json
import tiktoken

# Input / Output paths
INPUT_FOLDER = "project/formatted_texts"
OUTPUT_FILE = "project/chunks.json"

# Chunking configuration
MAX_TOKENS = 300
OVERLAP = 50

# Tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    # Count tokens in text
    return len(tokenizer.encode(text))

def read_files(folder):
    # Read all .txt files from folder
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".txt"):
            path = os.path.join(folder, file)
            with open(path, "r", encoding="utf-8") as f:
                docs.append({
                    "file": file,
                    "text": f.read().strip()
                })
    return docs

def split_sections(text):
    

    # Step 1: Newline before numbered headings
    text = re.sub(r"(?<!\n)(\d+\.\s+[A-Z])", r"\n\1", text)

    # Step 2: Match only the heading part
    pattern = re.compile(r"\n(\d+\.\s+[A-Z][A-Za-z ]{2,60})")

    matches = list(pattern.finditer(text))

    sections = []
    if not matches:
        print("❌ WARNING: No headings detected in file!")
        return []

    for i, match in enumerate(matches):
        raw_title = match.group(1).strip()

        # stop title at first long sentence
        title = raw_title.split(" A ")[0].split(" You ")[0].strip()

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        content = text[start:end].strip()

        if len(content) > 50: 
            sections.append((title, content))

    return sections

def token_sliding_window(text):
    # Token-based sliding window chunking
    tokens = tokenizer.encode(text)
    chunks = []

    start = 0
    while start < len(tokens):
        end = start + MAX_TOKENS
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append(chunk_text)
        start += MAX_TOKENS - OVERLAP

    return chunks

def hybrid_chunking(docs):
    # Section-wise + token-wise chunking
    all_chunks = []

    for doc in docs:
        sections = split_sections(doc["text"])

        for sec_idx, (title, content) in enumerate(sections, start=1):
            sub_chunks = token_sliding_window(content)

            for sub_idx, chunk in enumerate(sub_chunks, start=1):
                all_chunks.append({
                    "chunk_id": f"{doc['file']}_{sec_idx}_{sub_idx}",
                    "file": doc["file"],
                    "section_title": title,
                    "text": chunk.strip(),
                    "token_count": count_tokens(chunk)
                })

    return all_chunks

if __name__ == "__main__":
    docs = read_files(INPUT_FOLDER)

    if not docs:
        print("❌ No TXT files found in folder.")
        exit()

    chunks = hybrid_chunking(docs)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print(f"✅ SUCCESS: {len(chunks)} PERFECT hybrid chunks created")
