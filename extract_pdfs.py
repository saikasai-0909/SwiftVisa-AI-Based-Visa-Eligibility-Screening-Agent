import fitz  
from pathlib import Path

INPUT_DIR = Path("data/pdfs")
OUT_DIR = Path("data/txt")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    full = []
    for page in doc:
        text = page.get_text("text")
        full.append(text)
    return "\n".join(full)

def main():
    for pdf in INPUT_DIR.glob("*.pdf"):
        print("Processing:", pdf)
        text = extract_text_from_pdf(pdf)
        out_file = OUT_DIR / (pdf.stem + ".txt")
        out_file.write_text(text, encoding="utf-8")
        print("Wrote:", out_file)

if __name__ == "__main__":
    main()
