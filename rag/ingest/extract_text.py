import os
import fitz  # PyMuPDF
from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path


def extract_with_pymupdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        return "".join(page.get_text() for page in doc)
    except:
        return ""


def extract_with_pypdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        return "".join(page.extract_text() or "" for page in reader.pages)
    except:
        return ""


def extract_with_ocr(pdf_path):
    text = ""
    images = convert_from_path(pdf_path)
    for img in images:
        text += pytesseract.image_to_string(img)
    return text


def extract_text_from_pdf(pdf_path):
    text = extract_with_pymupdf(pdf_path)

    if text.strip():
        print(f"📄 PyMuPDF: {os.path.basename(pdf_path)}")
        return text

    text = extract_with_pypdf(pdf_path)

    if text.strip():
        print(f"📄 PyPDF: {os.path.basename(pdf_path)}")
        return text

    print(f"📸 OCR: {os.path.basename(pdf_path)}")
    return extract_with_ocr(pdf_path)


def save_text(text, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
