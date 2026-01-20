"""
SwiftVisa - FAISS Vector Store Builder
Creates embeddings using Sentence Transformers and builds FAISS index
"""

import os
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss

from chunk_documents import UKVisaDocumentChunker

# Setup logging - FIXED for Windows encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/vectorstore_build.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VectorStoreBuilder:
    """Builds FAISS vector store from document chunks"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        vectorstore_path: str = "vectorstore/faiss_index",
        model_cache_dir: str = "models/sentence_transformer"
    ):
        """
        Initialize Vector Store Builder
        
        Args:
            model_name: Sentence transformer model name
            vectorstore_path: Path to save FAISS index
            model_cache_dir: Path to cache the embedding model
        """
        self.model_name = model_name
        self.vectorstore_path = vectorstore_path
        self.model_cache_dir = model_cache_dir
        
        # Create directories
        os.makedirs(self.vectorstore_path, exist_ok=True)
        os.makedirs(self.model_cache_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        # Initialize embedding model
        logger.info(f"[MODEL] Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(
            model_name,
            cache_folder=model_cache_dir
        )
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(f"[SUCCESS] Model loaded. Embedding dimension: {self.embedding_dim}")
        
    def create_embeddings(self, chunks: List) -> np.ndarray:
        """
        Create embeddings for all chunks
        
        Args:
            chunks: List of document chunks
            
        Returns:
            numpy array of embeddings
        """
        logger.info(f"[PROCESS] Creating embeddings for {len(chunks)} chunks...")
        
        # Extract text content from chunks
        texts = [chunk.page_content for chunk in chunks]
        
        # Generate embeddings with progress bar
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        logger.info(f"[SUCCESS] Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build FAISS index from embeddings
        
        Args:
            embeddings: numpy array of embeddings
            
        Returns:
            FAISS index
        """
        logger.info("[BUILD] Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index (using IndexFlatIP for cosine similarity)
        index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Add embeddings to index
        index.add(embeddings)
        
        logger.info(f"[SUCCESS] FAISS index built. Total vectors: {index.ntotal}")
        return index
    
    def save_vectorstore(
        self,
        index: faiss.Index,
        chunks: List,
        metadata: Dict
    ):
        """
        Save FAISS index and metadata to disk
        
        Args:
            index: FAISS index
            chunks: Document chunks
            metadata: Additional metadata
        """
        logger.info(f"[SAVE] Saving vector store to: {self.vectorstore_path}")
        
        # Save FAISS index
        index_path = os.path.join(self.vectorstore_path, "index.faiss")
        faiss.write_index(index, index_path)
        logger.info(f"[SUCCESS] FAISS index saved: {index_path}")
        
        # Prepare chunk metadata
        chunks_metadata = []
        for i, chunk in enumerate(chunks):
            chunks_metadata.append({
                "id": i,
                "content": chunk.page_content,
                "metadata": chunk.metadata
            })
        
        # Save chunks with pickle (for LangChain compatibility)
        chunks_path = os.path.join(self.vectorstore_path, "index.pkl")
        with open(chunks_path, 'wb') as f:
            pickle.dump(chunks_metadata, f)
        logger.info(f"[SUCCESS] Chunks metadata saved: {chunks_path}")
        
        # Save build metadata
        build_info = {
            "build_date": datetime.now().isoformat(),
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "total_chunks": len(chunks),
            "total_vectors": index.ntotal,
            **metadata
        }
        
        metadata_path = os.path.join("vectorstore", "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(build_info, f, indent=2)
        logger.info(f"[SUCCESS] Build metadata saved: {metadata_path}")
    
    def load_vectorstore(self):
        """
        Load existing vector store from disk
        
        Returns:
            tuple: (index, chunks_metadata)
        """
        index_path = os.path.join(self.vectorstore_path, "index.faiss")
        chunks_path = os.path.join(self.vectorstore_path, "index.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            logger.error("[ERROR] Vector store not found!")
            return None, None
        
        logger.info(f"[LOAD] Loading vector store from: {self.vectorstore_path}")
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        logger.info(f"[SUCCESS] FAISS index loaded. Total vectors: {index.ntotal}")
        
        # Load chunks metadata
        with open(chunks_path, 'rb') as f:
            chunks_metadata = pickle.load(f)
        logger.info(f"[SUCCESS] Chunks metadata loaded. Total chunks: {len(chunks_metadata)}")
        
        return index, chunks_metadata
    
    def test_search(self, index: faiss.Index, chunks_metadata: List, query: str, k: int = 3):
        """
        Test search functionality
        
        Args:
            index: FAISS index
            chunks_metadata: Chunk metadata
            query: Search query
            k: Number of results to return
        """
        logger.info(f"\n[TEST] Testing search with query: '{query}'")
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = index.search(query_embedding, k)
        
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print(f"{'='*80}")
        
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            chunk = chunks_metadata[idx]
            print(f"\n[RESULT {i+1}] (Similarity: {distance:.4f})")
            print(f"Visa Type: {chunk['metadata'].get('visa_type', 'N/A')}")
            print(f"Source: {chunk['metadata'].get('source_file', 'N/A')}")
            print(f"Content: {chunk['content'][:300]}...")
            print("-" * 80)
    
    def build_complete_pipeline(self):
        """
        Complete pipeline: chunk → embed → index → save
        """
        logger.info("[START] Starting Complete Vector Store Build Pipeline")
        logger.info("="*80)
        
        # Step 1: Chunk documents
        logger.info("\n[STEP 1] Chunking Documents")
        chunker = UKVisaDocumentChunker(data_path="KB/uk_policies")
        chunks = chunker.process_all()
        
        if not chunks:
            logger.error("[ERROR] No chunks created. Exiting.")
            return
        
        # Step 2: Create embeddings
        logger.info("\n[STEP 2] Creating Embeddings")
        embeddings = self.create_embeddings(chunks)
        
        # Step 3: Build FAISS index
        logger.info("\n[STEP 3] Building FAISS Index")
        index = self.build_faiss_index(embeddings)
        
        # Step 4: Save vector store
        logger.info("\n[STEP 4] Saving Vector Store")
        
        # Collect statistics
        visa_counts = {}
        for chunk in chunks:
            visa_type = chunk.metadata.get("visa_type", "unknown")
            visa_counts[visa_type] = visa_counts.get(visa_type, 0) + 1
        
        metadata = {
            "visa_types": list(visa_counts.keys()),
            "chunks_per_visa": visa_counts,
            "source_docs": len(set(c.metadata.get("source_file", "") for c in chunks))
        }
        
        self.save_vectorstore(index, chunks, metadata)
        
        # Step 5: Test retrieval
        logger.info("\n[STEP 5] Testing Retrieval")
        chunks_metadata = [
            {
                "id": i,
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for i, chunk in enumerate(chunks)
        ]
        
        # Test queries
        test_queries = [
            "What are the requirements for a student visa?",
            "How much money do I need for a graduate visa?",
            "Can I work with a visitor visa?"
        ]
        
        for query in test_queries:
            self.test_search(index, chunks_metadata, query, k=2)
        
        logger.info("\n[SUCCESS] Vector Store Build Complete!")
        logger.info("="*80)
        logger.info(f"[SUMMARY] Statistics:")
        logger.info(f"  * Total chunks: {len(chunks)}")
        logger.info(f"  * Total vectors: {index.ntotal}")
        logger.info(f"  * Embedding dimension: {self.embedding_dim}")
        logger.info(f"  * Model: {self.model_name}")
        logger.info(f"  * Storage path: {self.vectorstore_path}")
        logger.info("="*80)


def main():
    """Main execution"""
    # Initialize builder
    builder = VectorStoreBuilder(
        model_name="all-MiniLM-L6-v2",  # Fast and efficient
        vectorstore_path="vectorstore/faiss_index",
        model_cache_dir="models/sentence_transformer"
    )
    
    # Run complete pipeline
    builder.build_complete_pipeline()


if __name__ == "__main__":
    main()