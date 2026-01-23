import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIGURATION: MAP FILENAMES TO VISA CATEGORIES ---
# This is crucial. It tells the AI which rules belong to which visa.
CATEGORY_MAP = {
    "General_visitor_V15.0EXT.pdf": "visitor_visa",
    "Health-and-Care-Visa-Guidance+04+August+2025_updated.pdf": "healthcare_visa",
    "Skilled+worker.pdf": "skilled_worker_visa",
    "Short-term_student__English_language_.pdf": "short_term_student_visa",
    "Student+and+Child+Student.pdf": "student_visa"
}

def load_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_chunks(data):
    print("Initializing Chunker...")
    
    # Chunk Size 1000: Good for capturing full legal paragraphs
    # Overlap 200: Ensures we don't cut a sentence in half
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    all_chunks = []
    
    print(f"Processing {len(data)} pages...")
    
    for page in data:
        filename = page['filename']
        content = page['content']
        page_num = page['page_number']
        
        # Determine Category based on filename
        # If filename isn't in our map, default to 'general'
        category = CATEGORY_MAP.get(filename, "general_immigration_rules")
        
        # Split the text
        splits = text_splitter.split_text(content)
        
        # Save chunks with their metadata
        for i, text_chunk in enumerate(splits):
            chunk_record = {
                "id": f"{filename}_p{page_num}_{i}",
                "text": text_chunk,
                "metadata": {
                    "source": filename,
                    "page": page_num,
                    "category": category # <--- THIS IS THE KEY FOR SCREENING
                }
            }
            all_chunks.append(chunk_record)
            
    return all_chunks

if __name__ == "__main__":
    input_file = "cleaned_visa_data.json"
    output_file = "chunked_visa_data.json"
    
    if os.path.exists(input_file):
        # 1. Load Cleaned Data
        raw_data = load_data(input_file)
        
        # 2. Chunk It
        final_chunks = create_chunks(raw_data)
        
        # 3. Save to new JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_chunks, f, indent=4)
            
        print(f"\n[SUCCESS] Created {len(final_chunks)} chunks.")
        print(f"Saved to: {output_file}")
        print("You are ready for Step 3 (Embedding)!")
        
    else:
        print(f"Error: {input_file} not found. Run Step 1 first.")