import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional


class VisaRetriever:
    """
    Retriever for UK Visa documentation using FAISS vector database.
    Supports semantic search for RAG (Retrieval-Augmented Generation) applications.
    """
    
    def __init__(self, 
                 index_path: str = "visa_faiss.index",
                 metadata_path: str = "visa_faiss_metadata.pkl",
                 model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever.
        
        Args:
            index_path: Path to the FAISS index file
            metadata_path: Path to the metadata pickle file
            model_name: Name of the sentence transformer model
        """
        print(f"Loading FAISS index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"✓ Loaded index with {self.index.ntotal} vectors")
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadatas = data['metadatas']
        print(f"✓ Loaded {len(self.documents)} documents")
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("✓ Model loaded successfully\n")
    
    def retrieve(self, 
                 query: str, 
                 top_k: int = 5,
                 return_scores: bool = True) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant documents for a given query.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            return_scores: Whether to include similarity scores
            
        Returns:
            List of dictionaries containing documents, metadata, and optionally scores
        """
        # Generate embedding for the query
        query_embedding = self.model.encode([query]).astype('float32')
        
        # Search the FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            result = {
                'rank': i + 1,
                'text': self.documents[idx],
                'metadata': self.metadatas[idx],
            }
            if return_scores:
                result['distance'] = float(distance)
                result['similarity_score'] = 1 / (1 + float(distance))  # Convert distance to similarity
            results.append(result)
        
        return results
    
    def retrieve_context(self, 
                        query: str, 
                        top_k: int = 3,
                        max_length: int = 2000) -> str:
        """
        Retrieve and format context for RAG applications.
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve
            max_length: Maximum total character length of context
            
        Returns:
            Formatted context string suitable for LLM prompts
        """
        results = self.retrieve(query, top_k=top_k, return_scores=False)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            text = result['text']
            metadata = result['metadata']
            
            # Format with source information
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'N/A')
            formatted = f"[Source: {source}, Page: {page}]\n{text}\n"
            
            # Check if adding this would exceed max_length
            if current_length + len(formatted) > max_length:
                break
                
            context_parts.append(formatted)
            current_length += len(formatted)
        
        return "\n".join(context_parts)
