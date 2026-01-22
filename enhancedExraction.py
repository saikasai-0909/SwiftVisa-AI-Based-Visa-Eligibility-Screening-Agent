import os
import re
import json
import pdfplumber
import fitz  # PyMuPDF
from pypdf import PdfReader
from unidecode import unidecode

# --------------------------------------------
#     TEXT EXTRACTION (pdfplumber)
# --------------------------------------------
def extract_text_from_pdf(pdf_path):
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_text.append(text)
    return "\n".join(all_text), len(all_text)


# --------------------------------------------
#     CLEANING FUNCTION
# --------------------------------------------
def clean_policy_text(text):
    text = unidecode(text)

    # Remove timestamps like: "12/5/25, 6:49 PM"
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(AM|PM)?", " ", text)

    # Remove page numbers: "Page X of Y"
    text = re.sub(r"Page\s+\d+\s+of\s+\d+", " ", text, flags=re.I)
    text = re.sub(r"\n\d+\n", "\n", text)

    # Remove repeated gov headers/footers
    noise_patterns = [
        r"Government of India.*",
        r"Home Office.*",
        r"UK Visas and Immigration.*",
        r"Ministry of External Affairs.*",
        r"Copyright.*",
        r"Disclaimer.*",
        r"Printed from.*",
        r"Print .* - GOV\.UK",
        r"https?://www\.gov\.uk[^\s]*",
    ]
    for p in noise_patterns:
        text = re.sub(p, " ", text, flags=re.I)

    # Fix broken line-joins: if a new line doesn't end with punctuation, join it
    text = re.sub(r"(?<![.!?])\n(?=[A-Za-z])", " ", text)

    # Normalize multiple spaces/newlines
    text = re.sub(r"\s+", " ", text)

    # Keep Appendix references recognizable
    text = re.sub(r"Appendix\s+([A-Za-z0-9IVXLC]+)", r"\nAppendix \1\n", text)

    # Keep Form references
    text = re.sub(r"Form\s+([A-Za-z0-9]+)", r"\nForm \1\n", text)

    return text.strip()


# --------------------------------------------
#     EXTRACT PRINTED URLS (from text)
# --------------------------------------------
def extract_printed_urls(text):
    urls = re.findall(r"https?://[^\s,)]+", text)
    return list(dict.fromkeys(urls))  # dedupe & preserve order


# --------------------------------------------
#     EXTRACT CLICKABLE HYPERLINKS (annotations)
# --------------------------------------------
def extract_annotation_links(pdf_path):
    doc = fitz.open(pdf_path)
    links = []
    for pno in range(len(doc)):
        page = doc[pno]
        for link in page.get_links():
            uri = link.get("uri")
            if uri:
                links.append({"page": pno+1, "uri": uri})
    return links


# --------------------------------------------
#     DETECT ACROFORM FIELDS (if any)
# --------------------------------------------
def detect_form_fields(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        fields = reader.get_fields()
        return fields if fields else None
    except:
        return None


# --------------------------------------------
#     EXTRACT TABLES (pdfplumber simple attempt)
# --------------------------------------------
def extract_tables(pdf_path):
    tables_all = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                table = page.extract_table()
                if table:
                    tables_all.append({"page": i+1, "table": table})
            except:
                pass
    return tables_all


# --------------------------------------------
#     PROCESS A SINGLE PDF
# --------------------------------------------
def process_pdf(pdf_path, output_dir):
    base = os.path.splitext(os.path.basename(pdf_path))[0]

    # Extract text
    raw_text, page_count = extract_text_from_pdf(pdf_path)

    # Clean text
    cleaned = clean_policy_text(raw_text)

    # Extract links
    printed_urls = extract_printed_urls(raw_text)
    annotation_links = extract_annotation_links(pdf_path)

    # Extract forms (AcroForm)
    forms = detect_form_fields(pdf_path)

    # Extract tables (basic)
    tables = extract_tables(pdf_path)

    # Build metadata
    meta = {
        "source_file": os.path.basename(pdf_path),
        "pages": page_count,
        "printed_urls": printed_urls,
        "annotation_links": annotation_links,
        "forms_present": bool(forms),
        "form_fields": forms,
        "tables": tables,
    }

    # Write outputs
    txt_path = os.path.join(output_dir, base + "_clean.txt")
    meta_path = os.path.join(output_dir, base + "_meta.json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Cleaned text saved → {txt_path}")
    print(f"[OK] Metadata saved → {meta_path}")


# --------------------------------------------
#     MAIN BATCH PROCESSOR
# --------------------------------------------
def process_all_pdfs(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            print("\n====================================")
            print(f"Processing → {filename}")
            print("====================================")
            process_pdf(pdf_path, output_dir)


# --------------------------------------------
#     RUNNER
# --------------------------------------------
if __name__ == "__main__":
    INPUT_DIR = r"C:\\Project\Data\\Raw_PDFs"            # change to your folder
    OUTPUT_DIR = r"C:\\Project\Data\\Visa_Docs_Cleaned"   # cleaned output

    process_all_pdfs(INPUT_DIR, OUTPUT_DIR)
