import os
import re
import json
from langchain_community.document_loaders import PyPDFLoader

def clean_extracted_text(text):
    """
    Cleans the ENGLISH text extracted from the PDF.
    """
    # 1. Remove "Page X of Y" and "Published for..." footers
    text = re.sub(r'Page \d+ of \d+.*', '', text)
    text = re.sub(r'Published for Home Office staff on.*', '', text)
    
    # 2. Remove "Version X.0" headers
    text = re.sub(r'Version \d+\.\d+', '', text)
    
    # 3. Fix line breaks (restore paragraphs)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    
    # 4. Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_all_pdfs(pdf_folder):
    all_cleaned_data = []
    
    files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    print(f"Found {len(files)} PDF files. Starting processing...")

    for filename in files:
        file_path = os.path.join(pdf_folder, filename)
        print(f"Processing: {filename}...")
        
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            for page in pages:
                # Clean the text
                cleaned_text = clean_extracted_text(page.page_content)
                
                # Only keep pages that actually have text
                if len(cleaned_text) > 50:
                    record = {
                        "filename": filename,
                        "page_number": page.metadata.get('page', 0) + 1,
                        "content": cleaned_text
                    }
                    all_cleaned_data.append(record)
            
        except Exception as e:
            print(f"   [ERROR] Could not read {filename}. Reason: {e}")
            
    return all_cleaned_data

if __name__ == "__main__":
    pdf_folder = "./pdfs" 
    output_file = "cleaned_visa_data.json"
    
    if not os.path.exists(pdf_folder):
        print(f"ERROR: The folder '{pdf_folder}' does not exist.")
    else:
        # 1. Process all PDFs
        data = process_all_pdfs(pdf_folder)
        
        # 2. Save to JSON
        if data:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            
            print(f"\n[SUCCESS] Processed {len(data)} pages total.")
            print(f"All cleaned data saved to: {output_file}")
            print("You can now proceed to Step 2 (Chunking).")
        else:
            print("[FAILURE] No data found.")