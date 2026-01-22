# src/chunking.py

import os
import json
import hashlib
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Logging
# ============================================================
def log(msg: str):
    print(f"[INFO] {msg}")


# ============================================================
# File Loaders
# ============================================================
def load_text(txt_path: str) -> str:
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Failed to read text file {txt_path}: {e}")


def load_meta(meta_path: str) -> Dict:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to read metadata {meta_path}: {e}")


# ============================================================
# Metadata Normalization
# ============================================================
def normalize_meta(meta: Dict) -> Dict:
    defaults = {
        "country": "unknown",
        "source_file": None,
        "forms_present": False
    }

    for k, v in defaults.items():
        meta.setdefault(k, v)

    return meta


# ============================================================
# Visa Type Detection
# ============================================================
def detect_visa_type(source_file: str, text: str) -> str:
    """
    Detect visa type using UK GOV PDF naming conventions
    Priority: filename > content
    """

    fname = source_file.lower()
    content = text.lower()

    # -------- Filename-based detection (PRIMARY) --------
    if "graduate visa" in fname:
        return "graduate"

    if "health and care worker" in fname:
        return "health_care_worker"

    if "skilled worker" in fname:
        return "skilled_worker"

    if "student visa" in fname:
        return "student"

    if "standard visitor" in fname or "visit the uk" in fname:
        return "visitor"

    # -------- Content-based detection (FALLBACK) in case if I add more file in Raw Data --------
    if "graduate visa" in content:
        return "graduate"

    if "health and care worker visa" in content:
        return "health_care_worker"

    if "skilled worker visa" in content:
        return "skilled_worker"

    if "student visa" in content:
        return "student"

    if "standard visitor visa" in content:
        return "visitor"

    return "unknown"


# ============================================================
# Hashing & Deduplication
# ============================================================
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def deduplicate(docs: List[Dict]) -> List[Dict]:
    seen = set()
    unique_docs = []

    for d in docs:
        h = hash_text(d["text"])
        if h not in seen:
            seen.add(h)
            unique_docs.append(d)

    return unique_docs


# ============================================================
# Chunking Logic
# ============================================================
def chunk_single_doc(
    text: str,
    meta: Dict,
    source_file: str,
    chunk_size: int,
    chunk_overlap: int
) -> List[Dict]:

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    visa_type = detect_visa_type(source_file, text)
    chunks = splitter.split_text(text)

    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "text": chunk,
            "metadata": {
                "country": meta["country"] if meta["country"] else "uk",
                "visa_type": visa_type,
                "source_file": source_file,
                "chunk_index": i,
                "char_length": len(chunk)
            }
        })

    return docs


# ============================================================
# Pipeline Orchestrator
# ============================================================
def chunk_all(
    cleaned_dir: str,
    meta_dir: str,
    output_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict]:

    log("Starting chunking pipeline...")
    all_docs = []

    for fname in sorted(os.listdir(cleaned_dir)):
        if not fname.endswith("_clean.txt"):
            continue

        txt_path = os.path.join(cleaned_dir, fname)
        meta_path = os.path.join(
            meta_dir, fname.replace("_clean.txt", "_meta.json")
        )

        if not os.path.exists(meta_path):
            log(f"Skipping {fname} (metadata missing)")
            continue

        text = load_text(txt_path)
        meta = normalize_meta(load_meta(meta_path))
        meta["source_file"] = fname
        meta["country"] = "uk"
        docs = chunk_single_doc(
            text=text,
            meta=meta,
            source_file=fname,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        all_docs.extend(docs)

    before = len(all_docs)
    all_docs = deduplicate(all_docs)
    after = len(all_docs)

    log(f"Removed {before - after} duplicate chunks")
    log(f"Final chunk count: {after}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for d in all_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    avg_len = sum(d["metadata"]["char_length"] for d in all_docs) / max(len(all_docs), 1)
    log(f"Average chunk length: {int(avg_len)} chars")
    log(f"Saved chunks → {output_file}")

    return all_docs


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    cleaned_dir = r"C:\\Project\\Data\\Visa_Docs_Cleaned"
    meta_dir = r"C:\\Project\\Data\\Visa_Docs_Cleaned"
    output_file = r"C:\\Project\\Data\\all_chunks.jsonl"

    chunk_all(
        cleaned_dir=cleaned_dir,
        meta_dir=meta_dir,
        output_file=output_file
    )

    print("\n[DONE] Chunking pipeline complete.")
