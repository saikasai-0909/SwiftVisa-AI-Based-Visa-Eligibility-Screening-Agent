"""
SwiftVisa - RAG Retriever System
Retrieves relevant visa policy chunks from FAISS vector store
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer

from config.config import (
    EMBEDDING_CONFIG,
    FAISS_CONFIG,
    RETRIEVAL_CONFIG,
    LOGGING_CONFIG
)

# Setup logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class VisaPolicyRetriever:
    """Retrieves relevant visa policy chunks using FAISS"""
    
    def __init__(
        self,
        index_path: str = None,
        embedding_model_name: str = None,
        top_k: int = None
    ):
        """
        Initialize the retriever
        
        Args:
            index_path: Path to FAISS index directory
            embedding_model_name: Name of embedding model
            top_k: Number of chunks to retrieve
        """
        self.index_path = index_path or FAISS_CONFIG["index_path"]
        self.embedding_model_name = embedding_model_name or EMBEDDING_CONFIG["model_name"]
        self.top_k = top_k or RETRIEVAL_CONFIG["top_k"]
        
        logger.info("🔄 Initializing VisaPolicyRetriever...")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            cache_folder=EMBEDDING_CONFIG["cache_dir"]
        )
        
        # Load FAISS index and chunks
        self.index, self.chunks_metadata = self._load_vectorstore()
        
        logger.info("✅ VisaPolicyRetriever initialized successfully")
    
    def _load_vectorstore(self) -> Tuple[faiss.Index, List[Dict]]:
        """Load FAISS index and chunk metadata"""
        import pickle
        
        index_file = os.path.join(self.index_path, "index.faiss")
        chunks_file = os.path.join(self.index_path, "index.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(chunks_file):
            raise FileNotFoundError(
                f"Vector store not found at {self.index_path}. "
                "Please run build_vectorstore.py first!"
            )
        
        # Load FAISS index
        index = faiss.read_index(index_file)
        logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
        
        # Load chunks metadata
        with open(chunks_file, 'rb') as f:
            chunks_metadata = pickle.load(f)
        logger.info(f"Loaded {len(chunks_metadata)} chunk metadata entries")
        
        return index, chunks_metadata
    
    def _create_query_embedding(self, query: str) -> np.ndarray:
        """Create embedding for query"""
        embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        return embedding
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        visa_type_filter: str = None,
        score_threshold: float = None
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of results to return (overrides default)
            visa_type_filter: Filter by specific visa type
            score_threshold: Minimum similarity score
            
        Returns:
            List of retrieved chunks with metadata and scores
        """
        k = top_k or self.top_k
        threshold = score_threshold or RETRIEVAL_CONFIG["score_threshold"]
        
        logger.info(f"Retrieving top {k} chunks for query: '{query}'")
        
        # Create query embedding
        query_embedding = self._create_query_embedding(query)
        
        # Search FAISS index
        # Retrieve more if we're filtering, so we have enough after filtering
        search_k = k * 3 if visa_type_filter else k
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Prepare results
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if score < threshold:
                continue
            
            chunk = self.chunks_metadata[idx]
            
            # Apply visa type filter if specified
            if visa_type_filter:
                chunk_visa_type = chunk['metadata'].get('visa_type', '')
                if chunk_visa_type != visa_type_filter:
                    continue
            
            results.append({
                'content': chunk['content'],
                'metadata': chunk['metadata'],
                'score': float(score),
                'chunk_id': chunk['id']
            })
            
            # Stop if we have enough results
            if len(results) >= k:
                break
        
        logger.info(f"Retrieved {len(results)} relevant chunks")
        return results
    
    def retrieve_with_context(
        self,
        query: str,
        user_context: Dict = None,
        top_k: int = None
    ) -> Dict:
        """
        Retrieve chunks with additional user context for better filtering
        
        Args:
            query: User query
            user_context: User profile information (visa_type, nationality, etc.)
            top_k: Number of results
            
        Returns:
            Dictionary with retrieved chunks and metadata
        """
        visa_type_filter = None
        if user_context and 'visa_type' in user_context:
            visa_type_filter = user_context['visa_type']
        
        chunks = self.retrieve(
            query=query,
            top_k=top_k,
            visa_type_filter=visa_type_filter
        )
        
        return {
            'query': query,
            'chunks': chunks,
            'total_retrieved': len(chunks),
            'visa_type_filter': visa_type_filter,
            'user_context': user_context or {}
        }
    
    def format_context_for_llm(self, retrieved_chunks: List[Dict], max_chars: int = 3000) -> str:
        """
        Format retrieved chunks into context string for LLM
        
        Args:
            retrieved_chunks: List of retrieved chunk dictionaries
            max_chars: Maximum characters for context (to prevent overflow)
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return "No relevant policy information found."
        
        context_parts = []
        total_chars = 0
        
        for i, chunk in enumerate(retrieved_chunks, 1):
            visa_type = chunk['metadata'].get('visa_type', 'unknown').replace('_', ' ').title()
            source = chunk['metadata'].get('source_file', 'unknown')
            content = chunk['content']
            score = chunk['score']
            
            # Create source entry
            entry = f"[Source {i}] {visa_type} - {source} (Relevance: {score:.3f})\n{content}\n"
            
            # Check if adding this would exceed max_chars
            if total_chars + len(entry) > max_chars and context_parts:
                logger.warning(f"Context truncated at {total_chars} chars to prevent overflow")
                break
            
            context_parts.append(entry)
            total_chars += len(entry)
        
        return "\n".join(context_parts)
    
    def get_statistics(self) -> Dict:
        """Get retriever statistics"""
        visa_counts = {}
        for chunk in self.chunks_metadata:
            visa_type = chunk['metadata'].get('visa_type', 'unknown')
            visa_counts[visa_type] = visa_counts.get(visa_type, 0) + 1
        
        return {
            'total_chunks': len(self.chunks_metadata),
            'total_vectors': self.index.ntotal,
            'visa_types': visa_counts,
            'embedding_model': self.embedding_model_name,
            'top_k': self.top_k
        }


# Test function
def test_retriever():
    """Test the retriever with sample queries"""
    print("\n" + "="*80)
    print("🧪 Testing VisaPolicyRetriever")
    print("="*80)
    
    retriever = VisaPolicyRetriever()
    
    # Display statistics
    stats = retriever.get_statistics()
    print(f"\n📊 Retriever Statistics:")
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Total vectors: {stats['total_vectors']}")
    print(f"  Embedding model: {stats['embedding_model']}")
    
    # Test queries
    test_queries = [
        {
            "query": "What are the financial requirements for a student visa?",
            "visa_type": "student"
        },
        {
            "query": "Can I work with a graduate visa?",
            "visa_type": "graduate"
        },
        {
            "query": "What is the minimum salary for skilled worker visa?",
            "visa_type": "skilled_worker"
        }
    ]
    
    for test in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {test['query']}")
        print(f"Filter: {test['visa_type']}")
        print(f"{'='*80}")
        
        results = retriever.retrieve(
            query=test['query'],
            visa_type_filter=test['visa_type'],
            top_k=3
        )
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 Result {i} (Score: {result['score']:.4f})")
            print(f"Visa Type: {result['metadata'].get('visa_type', 'N/A')}")
            print(f"Content: {result['content'][:200]}...")
        
        # Show formatted context
        print(f"\n📝 Formatted Context for LLM:")
        print("-" * 80)
        context = retriever.format_context_for_llm(results)
        print(context[:500] + "...")


if __name__ == "__main__":
    test_retriever()