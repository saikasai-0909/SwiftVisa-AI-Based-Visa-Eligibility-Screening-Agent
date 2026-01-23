import os
import re
import json
from pathlib import Path
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VisaDocumentProcessor:
    def __init__(self, raw_data_dir="data/raw", cleaned_dir="data/cleaned", chunks_dir="data/chunks"):
        self.raw_data_dir = Path(raw_data_dir)
        self.cleaned_dir = Path(cleaned_dir)
        self.chunks_dir = Path(chunks_dir)
        
        # Create directories if they don't exist
        self.cleaned_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
        # Visa type mapping
        self.visa_mapping = {
            "Graduate visa - GOV.UK.pdf": {
                "visa_type": "Graduate Visa",
                "country": "UK",
                "category": "post-study"
            },
            "Student visa - GOV.UK.pdf": {
                "visa_type": "Student Visa",
                "country": "UK",
                "category": "study"
            },
            "Health and Care Worker visa - GOV.UK.pdf": {
                "visa_type": "Health and Care Worker Visa",
                "country": "UK",
                "category": "work"
            },
            "Skilled Worker visa - GOV.UK.pdf": {
                "visa_type": "Skilled Worker Visa",
                "country": "UK",
                "category": "work"
            },
            "Visit the UK as a Standard Visitor - GOV.UK.pdf": {
                "visa_type": "Standard Visitor Visa",
                "country": "UK",
                "category": "visit"
            }
        }
    
    def clean_text(self, text):
        """Clean extracted PDF text"""
        
        # Remove timestamps and page numbers
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s+[AP]M', '', text)
        
        # Remove "Print [Document] - GOV.UK" headers
        text = re.sub(r'Print .+ - GOV\.UK', '', text)
        text = re.sub(r'\d{1,2}/\d{1,2}/\d{2,4}.*?GOV\.UK', '', text)
        
        # Remove URLs
        text = re.sub(r'https?://[^\s]+', '', text)
        
        # Remove footer copyright
        text = re.sub(r'All content is available under.*?© Crown copyright', '', text, flags=re.DOTALL)
        
        # Remove "Part of" breadcrumbs
        text = re.sub(r'Part of\s+.*?\n', '', text)
        
        # Fix excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Fix broken bullets
        text = re.sub(r'^\s*[•·○]\s*', '- ', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def detect_section(self, text):
        """Detect section type from text content"""
        
        text_lower = text.lower()
        
        # Section keywords (ordered by priority)
        section_patterns = [
            ("eligibility", ["eligib", "requirements", "must have", "must be", "you can apply if"]),
            ("costs", ["fee", "cost", "pay", "£", "surcharge", "how much"]),
            ("salary", ["salary", "paid", "going rate", "minimum salary", "£", "per year"]),
            ("documents", ["documents", "provide", "certificate", "passport", "proof"]),
            ("application", ["apply", "application", "online", "how to apply"]),
            ("family", ["partner", "children", "dependant", "family member"]),
            ("timeline", ["how long", "decision", "weeks", "processing"]),
            ("english", ["english", "language", "ielts", "b1", "b2"]),
            ("extending", ["extend", "extension", "switch", "update"]),
        ]
        
        for section_name, keywords in section_patterns:
            if any(keyword in text_lower for keyword in keywords):
                return section_name
        
        return "overview"
    
    def process_single_document(self, filename):
        """Process a single PDF document"""
        
        filepath = self.raw_data_dir / filename
        
        if not filepath.exists():
            print(f"❌ File not found: {filename}")
            return None
        
        print(f"📄 Processing: {filename}")
        
        # Get metadata
        metadata = self.visa_mapping.get(filename, {
            "visa_type": "Unknown",
            "country": "UK",
            "category": "general"
        })
        
        # Load PDF
        try:
            loader = PyPDFLoader(str(filepath))
            pages = loader.load()
        except Exception as e:
            print(f"❌ Error loading PDF: {e}")
            return None
        
        # Combine all pages
        raw_text = "\n\n".join([page.page_content for page in pages])
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Add document header
        header = f"""Visa Type: {metadata['visa_type']}
Country: {metadata['country']}
Category: {metadata['category']}
Source: Official UK Government Immigration Guidelines

---

"""
        
        cleaned_text = header + cleaned_text
        
        # Save cleaned version
        cleaned_filename = filename.replace('.pdf', '_cleaned.txt')
        cleaned_path = self.cleaned_dir / cleaned_filename
        
        with open(cleaned_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        print(f"  ✓ Cleaned and saved to: {cleaned_filename}")
        
        return {
            "filename": filename,
            "visa_type": metadata["visa_type"],
            "country": metadata["country"],
            "category": metadata["category"],
            "text": cleaned_text,
            "word_count": len(cleaned_text.split())
        }
    
    def chunk_document(self, doc_data):
        """Chunk a cleaned document"""
        
        print(f"✂️  Chunking: {doc_data['visa_type']}")
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=[
                "\n## ",      # Major sections
                "\n### ",     # Subsections
                "\n\n",       # Paragraphs
                "\n",         # Lines
                ". ",         # Sentences
                " ",          # Words
            ],
            length_function=len,
        )
        
        # Split text
        text_chunks = text_splitter.split_text(doc_data["text"])
        
        # Add metadata to each chunk
        chunks_with_metadata = []
        
        for i, chunk_text in enumerate(text_chunks):
            chunk = {
                "chunk_id": f"{doc_data['visa_type'].replace(' ', '_')}_{i}",
                "text": chunk_text,
                "metadata": {
                    "visa_type": doc_data["visa_type"],
                    "country": doc_data["country"],
                    "category": doc_data["category"],
                    "section": self.detect_section(chunk_text),
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "source_file": doc_data["filename"]
                }
            }
            chunks_with_metadata.append(chunk)
        
        print(f"  ✓ Created {len(chunks_with_metadata)} chunks")
        
        return chunks_with_metadata
    
    def process_all_documents(self):
        """Process all documents in the raw data directory"""
        
        print("\n" + "="*60)
        print("🚀 STARTING VISA DOCUMENT PROCESSING")
        print("="*60 + "\n")
        
        all_cleaned_docs = []
        all_chunks = []
        
        # Process each document
        for filename in self.visa_mapping.keys():
            cleaned_doc = self.process_single_document(filename)
            
            if cleaned_doc:
                all_cleaned_docs.append(cleaned_doc)
                
                # Chunk the cleaned document
                chunks = self.chunk_document(cleaned_doc)
                all_chunks.extend(chunks)
        
        # Save all chunks to JSON
        chunks_file = self.chunks_dir / "all_chunks.json"
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*60)
        print("✅ PROCESSING COMPLETE")
        print("="*60)
        print(f"📊 Summary:")
        print(f"  - Documents processed: {len(all_cleaned_docs)}")
        print(f"  - Total chunks created: {len(all_chunks)}")
        print(f"  - Chunks saved to: {chunks_file}")
        print("="*60 + "\n")
        
        return all_cleaned_docs, all_chunks
    
    def validate_chunks(self, chunks):
        """Validate chunk quality"""
        
        print("\n🔍 VALIDATING CHUNKS...")
        
        issues = []
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]
            metadata = chunk["metadata"]
            
            # Check 1: Minimum length
            if len(text.strip()) < 100:
                issues.append(f"❌ {chunk_id}: Too short ({len(text)} chars)")
            
            # Check 2: No URL artifacts
            if "http" in text or "www." in text:
                issues.append(f"⚠️  {chunk_id}: Contains URLs")
            
            # Check 3: Metadata completeness
            required_keys = ["visa_type", "country", "section"]
            missing = [k for k in required_keys if k not in metadata]
            if missing:
                issues.append(f"❌ {chunk_id}: Missing metadata: {missing}")
        
        if issues:
            print(f"\n⚠️  Found {len(issues)} issues:")
            for issue in issues[:10]:  # Show first 10
                print(f"  {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("✅ All chunks validated successfully!")
        
        return len(issues) == 0


# Main execution function
def main():
    # Initialize processor
    processor = VisaDocumentProcessor()
    
    # Process all documents
    cleaned_docs, chunks = processor.process_all_documents()
    
    # Validate chunks
    processor.validate_chunks(chunks)
    
    # Print sample chunk
    if chunks:
        print("\n" + "="*60)
        print("📝 SAMPLE CHUNK")
        print("="*60)
        sample = chunks[0]
        print(f"Chunk ID: {sample['chunk_id']}")
        print(f"Visa Type: {sample['metadata']['visa_type']}")
        print(f"Section: {sample['metadata']['section']}")
        print(f"Text Preview:\n{sample['text'][:300]}...")
        print("="*60)


if __name__ == "__main__":
    main()