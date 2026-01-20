"""
SwiftVisa - UK Visa Document Chunking Script
Processes UK visa policy documents and prepares them for vector storage
"""

import os
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from typing import List, Dict
import json

# UK Visa Types
UK_VISA_TYPES = {
    "student": ["student_visa", "tier_4", "student route"],
    "graduate": ["graduate_visa", "graduate route", "post-study"],
    "skilled_worker": ["skilled_worker", "tier_2", "work visa"],
    "health_care_worker": ["health_care_worker", "health and care", "nhs"],
    "visitor": ["standard_visitor", "tourist", "visitor visa"]
}

class UKVisaDocumentChunker:
    """Handles chunking of UK visa policy documents"""
    
    def __init__(self, data_path: str = "KB/uk_policies"):
        self.data_path = data_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
    def load_documents(self) -> List:
        """Load all PDF documents from the UK policies folder"""
        print(f"📂 Loading documents from: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            print(f"❌ Path not found: {self.data_path}")
            print(f"💡 Creating directory...")
            os.makedirs(self.data_path, exist_ok=True)
            return []
        
        # Load PDFs
        loader = DirectoryLoader(
            self.data_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        try:
            documents = loader.load()
            print(f"✅ Loaded {len(documents)} document pages")
            return documents
        except Exception as e:
            print(f"❌ Error loading documents: {e}")
            return []
    
    def extract_visa_type(self, doc) -> str:
        """Extract visa type from document metadata or content"""
        # Try filename first
        source = doc.metadata.get("source", "").lower()
        
        for visa_type, keywords in UK_VISA_TYPES.items():
            if any(keyword in source for keyword in keywords):
                return visa_type
        
        # Try content (first 500 chars)
        content = doc.page_content[:500].lower()
        for visa_type, keywords in UK_VISA_TYPES.items():
            if any(keyword in content for keyword in keywords):
                return visa_type
        
        return "general"
    
    def chunk_documents(self, documents: List) -> List:
        """Split documents into chunks with metadata"""
        print(f"\n✂️  Chunking documents...")
        
        all_chunks = []
        
        for doc in documents:
            # Split document
            splits = self.text_splitter.split_documents([doc])
            
            # Extract visa type
            visa_type = self.extract_visa_type(doc)
            
            # Add enhanced metadata to each chunk
            for i, split in enumerate(splits):
                split.metadata.update({
                    "country": "UK",
                    "visa_type": visa_type,
                    "source_file": os.path.basename(doc.metadata.get("source", "")),
                    "chunk_id": i,
                    "total_chunks": len(splits)
                })
                all_chunks.append(split)
        
        print(f"✅ Created {len(all_chunks)} chunks")
        return all_chunks
    
    def analyze_chunks(self, chunks: List):
        """Analyze and display chunk statistics"""
        if not chunks:
            print("⚠️  No chunks to analyze")
            return
        
        print("\n📊 Chunk Analysis:")
        print("=" * 60)
        
        # Overall statistics
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        print(f"Total chunks: {len(chunks)}")
        print(f"Average chunk size: {sum(chunk_lengths) / len(chunk_lengths):.0f} characters")
        print(f"Min chunk size: {min(chunk_lengths)} characters")
        print(f"Max chunk size: {max(chunk_lengths)} characters")
        
        # Breakdown by visa type
        print("\n📋 Chunks by Visa Type:")
        visa_counts = {}
        for chunk in chunks:
            visa_type = chunk.metadata.get("visa_type", "unknown")
            visa_counts[visa_type] = visa_counts.get(visa_type, 0) + 1
        
        for visa_type, count in sorted(visa_counts.items()):
            print(f"  • {visa_type.replace('_', ' ').title()}: {count} chunks")
        
        # Sample chunk
        print("\n📄 Sample Chunk:")
        print("-" * 60)
        sample = chunks[0]
        print(f"Content: {sample.page_content[:300]}...")
        print(f"\nMetadata: {json.dumps(sample.metadata, indent=2)}")
        print("-" * 60)
    
    def save_chunks_json(self, chunks: List, output_path: str = "chunks_preview.json"):
        """Save chunks to JSON for inspection"""
        chunks_data = []
        for chunk in chunks:
            chunks_data.append({
                "content": chunk.page_content,
                "metadata": chunk.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Chunks saved to: {output_path}")
    
    def process_all(self) -> List:
        """Complete pipeline: load, chunk, analyze"""
        print("🚀 Starting UK Visa Document Processing...")
        print("=" * 60)
        
        # Load documents
        documents = self.load_documents()
        
        if not documents:
            print("\n⚠️  No documents found!")
            print(f"📁 Please add PDF files to: {self.data_path}")
            print("\nExpected files:")
            print("  • student_visa.pdf")
            print("  • graduate_visa.pdf")
            print("  • skilled_worker_visa.pdf")
            print("  • health_care_worker_visa.pdf")
            print("  • standard_visitor_visa.pdf")
            return []
        
        # Chunk documents
        chunks = self.chunk_documents(documents)
        
        # Analyze
        self.analyze_chunks(chunks)
        
        # Save preview
        self.save_chunks_json(chunks)
        
        print("\n✅ Processing complete!")
        return chunks


def main():
    """Main execution"""
    # Initialize chunker
    chunker = UKVisaDocumentChunker(data_path="KB/uk_policies")
    
    # Process documents
    chunks = chunker.process_all()
    
    return chunks


if __name__ == "__main__":
    chunks = main()