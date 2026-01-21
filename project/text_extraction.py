import pdfplumber
import os
import re

# Input PDF folder and output text folder
PDF_FOLDER = "project/visa_pdfs"
OUTPUT_FOLDER = "project/cleaned_text"

# Create output folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def clean_text(text: str) -> str:
    # Remove date + GOV.UK header/footer lines
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2},.*?GOV\.UK", "", text)

    # Remove GOV.UK page URLs
    text = re.sub(r"https://www\.gov\.uk/.*?\d+/\d+", "", text)

    # Convert relative links to full GOV.UK URLs
    text = re.sub(
        r"\(/([a-zA-Z0-9\-\/]+)\)",
        r"https://www.gov.uk/\1",
        text
    )
    # Normalize excessive newlines

    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trim whitespace on each line

    text = "\n".join(line.strip() for line in text.splitlines())

    return text.strip()


# Process each PDF file
for file_name in os.listdir(PDF_FOLDER):
    if not file_name.lower().endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_FOLDER, file_name)
    output_path = os.path.join(
        OUTPUT_FOLDER, file_name.replace(".pdf", ".txt")
    )

    print(f"Processing → {file_name}")

    full_text = []

    # Extract text page by page
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text = text.replace("\t", " ")
                full_text.append(text)

    # Merge all pages
    merged_text = "\n\n".join(full_text)

    # Clean extracted text
    cleaned_text = clean_text(merged_text)

    # Save cleaned output
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f" Saved clean output → {output_path}")

print("\n ALL PDFs EXTRACTED & CLEANED PERFECTLY")
