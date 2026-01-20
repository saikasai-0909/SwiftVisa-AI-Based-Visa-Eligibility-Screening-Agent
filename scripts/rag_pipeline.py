import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import json
from typing import Dict, List, Optional
from datetime import datetime

from retriever import VisaPolicyRetriever
from llm_integration import LMStudioLLM
from config.config import LOGGING_CONFIG

# Setup logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG["log_file"]),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SwiftVisaRAG:
    """Complete RAG pipeline for visa eligibility screening"""
    
    def __init__(self):
        """Initialize RAG pipeline with retriever and LLM"""
        logger.info(" Initializing SwiftVisa RAG Pipeline...")
        
        # Initialize components
        self.retriever = VisaPolicyRetriever()
        self.llm = LMStudioLLM()
        
        # Conversation history for multi-turn conversations
        self.conversation_history = []
        
        logger.info(" SwiftVisa RAG Pipeline ready!")
    
    def answer_question(
        self,
        question: str,
        visa_type: Optional[str] = None,
        top_k: int = 3
    ) -> Dict:
        """
        Answer a visa-related question using RAG
        
        Args:
            question: User's question
            visa_type: Optional filter for specific visa type
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing question: '{question}'")
        
        # Step 1: Retrieve relevant context
        retrieved = self.retriever.retrieve(
            query=question,
            visa_type_filter=visa_type,
            top_k=top_k
        )
        
        if not retrieved:
            return {
                "answer": "I couldn't find relevant information in the visa policies to answer your question. Please try rephrasing or ask about a specific visa type.",
                "sources": [],
                "retrieved_chunks": 0
            }
        
        # Step 2: Format context for LLM
        context = self.retriever.format_context_for_llm(retrieved)
        
        # Step 3: Get answer from LLM
        answer = self.llm.answer_question(
            question=question,
            context=context
        )
        
        # Step 4: Prepare response
        sources = [
            {
                "visa_type": chunk['metadata'].get('visa_type', 'unknown'),
                "source_file": chunk['metadata'].get('source_file', 'unknown'),
                "relevance_score": chunk['score']
            }
            for chunk in retrieved
        ]
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": len(retrieved),
            "visa_type_filter": visa_type,
            "timestamp": datetime.now().isoformat()
        }
    
    def evaluate_eligibility(
        self,
        user_profile: Dict,
        visa_type: str
    ) -> Dict:
        """
        Evaluate user's visa eligibility
        
        Args:
            user_profile: Dictionary with user information
            visa_type: Type of visa to evaluate
            
        Returns:
            Dictionary with eligibility evaluation
        """
        logger.info(f"Evaluating eligibility for {visa_type}")
        logger.info(f"User profile: {json.dumps(user_profile, indent=2)}")
        
        # Step 1: Create query from user profile and visa type
        query = f"What are the requirements for {visa_type.replace('_', ' ')} visa?"
        
        # Step 2: Retrieve relevant policy information
        retrieved = self.retriever.retrieve(
            query=query,
            visa_type_filter=visa_type,
            top_k=5  # Reduced from 7 to prevent context overflow
        )
        
        if not retrieved:
            return {
                "evaluation": f"No policy information found for {visa_type} visa.",
                "eligibility": "Unknown",
                "sources": []
            }
        
        # Step 3: Format context
        context = self.retriever.format_context_for_llm(retrieved)
        
        # Step 4: Get eligibility evaluation from LLM
        evaluation = self.llm.evaluate_eligibility(
            user_profile=user_profile,
            visa_type=visa_type,
            context=context
        )
        
        # Step 5: Prepare response
        sources = [
            {
                "visa_type": chunk['metadata'].get('visa_type', 'unknown'),
                "source_file": chunk['metadata'].get('source_file', 'unknown'),
                "relevance_score": chunk['score']
            }
            for chunk in retrieved
        ]
        
        return {
            "visa_type": visa_type,
            "user_profile": user_profile,
            "evaluation": evaluation,
            "sources": sources,
            "retrieved_chunks": len(retrieved),
            "timestamp": datetime.now().isoformat()
        }
    
    def chat(
        self,
        user_message: str,
        use_retrieval: bool = True,
        visa_type: Optional[str] = None
    ) -> Dict:
        """
        Chat interface with optional retrieval
        
        Args:
            user_message: User's message
            use_retrieval: Whether to use RAG retrieval
            visa_type: Optional visa type filter
            
        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Chat message: '{user_message}'")
        
        context = None
        retrieved_chunks = 0
        
        # Retrieve context if enabled
        if use_retrieval:
            retrieved = self.retriever.retrieve(
                query=user_message,
                visa_type_filter=visa_type,
                top_k=5
            )
            
            if retrieved:
                context = self.retriever.format_context_for_llm(retrieved)
                retrieved_chunks = len(retrieved)
        
        # Get response from LLM
        response = self.llm.chat(
            user_message=user_message,
            conversation_history=self.conversation_history,
            context=context
        )
        
        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Keep only last 10 messages
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        return {
            "user_message": user_message,
            "response": response,
            "retrieved_chunks": retrieved_chunks,
            "timestamp": datetime.now().isoformat()
        }
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")
    
    def get_statistics(self) -> Dict:
        """Get pipeline statistics"""
        return {
            "retriever_stats": self.retriever.get_statistics(),
            "llm_config": {
                "model": self.llm.model,
                "base_url": self.llm.base_url,
                "temperature": self.llm.temperature
            },
            "conversation_length": len(self.conversation_history)
        }


# Interactive testing
def interactive_test():
    """Interactive testing interface"""
    print("\n" + "="*80)
    print(" SwiftVisa RAG Pipeline - Interactive Test")
    print("="*80)
    print("\nCommands:")
    print("  'q <question>' - Ask a question")
    print("  'e' - Evaluate eligibility (will prompt for details)")
    print("  'stats' - Show statistics")
    print("  'reset' - Reset conversation")
    print("  'quit' - Exit")
    print("="*80)
    
    try:
        rag = SwiftVisaRAG()
    except Exception as e:
        print(f"\n Error initializing RAG pipeline: {e}")
        return
    
    while True:
        try:
            user_input = input("\n You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("\n Goodbye!")
                break
            
            elif user_input.lower() == 'stats':
                stats = rag.get_statistics()
                print(f"\n Pipeline Statistics:")
                print(json.dumps(stats, indent=2))
            
            elif user_input.lower() == 'reset':
                rag.reset_conversation()
                print(" Conversation reset")
            
            elif user_input.lower().startswith('q '):
                question = user_input[2:].strip()
                
                # Ask for visa type filter
                print("\nVisa type filter (press Enter for all):")
                print("  student | graduate | skilled_worker | health_care_worker | visitor")
                visa_filter = input("Filter: ").strip().lower() or None
                
                result = rag.answer_question(question, visa_type=visa_filter)
                
                print(f"\n Answer:")
                print("-" * 80)
                print(result['answer'])
                print("-" * 80)
                print(f"\n Sources: {result['retrieved_chunks']} chunks retrieved")
                for source in result['sources'][:3]:
                    print(f"  • {source['visa_type']} - {source['source_file']} (score: {source['relevance_score']:.3f})")
            
            elif user_input.lower() == 'e':
                print("\n Eligibility Evaluation")
                print("-" * 80)
                
                # Select visa type
                print("\nSelect visa type:")
                print("  1. Student Visa")
                print("  2. Graduate Visa")
                print("  3. Skilled Worker Visa")
                print("  4. Health Care Worker Visa")
                print("  5. Standard Visitor Visa")
                
                visa_choice = input("Choice (1-5): ").strip()
                visa_map = {
                    "1": "student",
                    "2": "graduate",
                    "3": "skilled_worker",
                    "4": "health_care_worker",
                    "5": "visitor"
                }
                
                visa_type = visa_map.get(visa_choice)
                if not visa_type:
                    print("Invalid choice")
                    continue
                
                # Collect user profile
                print(f"\nEnter details for {visa_type.replace('_', ' ').title()} Visa:")
                user_profile = {
                    "Age": input("Age: ").strip(),
                    "Nationality": input("Nationality: ").strip(),
                    "Education": input("Education: ").strip(),
                    "Employment Status": input("Employment Status: ").strip(),
                }
                
                if visa_type in ["student", "graduate"]:
                    user_profile["English Test Score"] = input("English Test Score (e.g., IELTS 6.5): ").strip()
                    user_profile["Financial Proof"] = input("Financial Proof (amount): ").strip()
                
                if visa_type in ["skilled_worker", "health_care_worker"]:
                    user_profile["Job Offer"] = input("Job Offer (Yes/No): ").strip()
                    user_profile["Salary"] = input("Salary: ").strip()
                
                # Evaluate
                print("\n Evaluating eligibility...")
                result = rag.evaluate_eligibility(user_profile, visa_type)
                
                print(f"\n Eligibility Evaluation:")
                print("=" * 80)
                print(result['evaluation'])
                print("=" * 80)
                print(f"\n Based on {result['retrieved_chunks']} policy documents")
            
            else:
                # Default chat
                result = rag.chat(user_input)
                print(f"\n {result['response']}")
        
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}")
            logger.exception("Error in interactive test")


# Automated test
def automated_test():
    """Run automated tests"""
    print("\n" + "="*80)
    print(" SwiftVisa RAG Pipeline - Automated Tests")
    print("="*80)
    
    try:
        rag = SwiftVisaRAG()
        
        # Test 1: Question Answering
        print("\n Test 1: Question Answering")
        print("-" * 80)
        
        questions = [
            ("What are the financial requirements for a student visa?", "student"),
            ("Can I work with a graduate visa?", "graduate"),
            ("What is the minimum salary for a skilled worker visa?", "skilled_worker"),
        ]
        
        for question, visa_type in questions:
            print(f"\n {question}")
            result = rag.answer_question(question, visa_type=visa_type)
            print(f" Answer ({result['retrieved_chunks']} sources):")
            print(result['answer'][:300] + "...")
        
        # Test 2: Eligibility Evaluation
        print("\n" + "="*80)
        print(" Test 2: Eligibility Evaluation")
        print("-" * 80)
        
        test_profile = {
            "Age": "24",
            "Nationality": "Indian",
            "Education": "Bachelor's Degree",
            "English Test Score": "IELTS 7.0",
            "Financial Proof": "£12,000"
        }
        
        print(f"\nUser Profile: {json.dumps(test_profile, indent=2)}")
        result = rag.evaluate_eligibility(test_profile, "student")
        print(f"\n Evaluation:")
        print(result['evaluation'][:400] + "...")
        
        print("\n" + "="*80)
        print(" All Tests Complete!")
        print("="*80)
        
    except Exception as e:
        print(f"\n Error: {e}")
        logger.exception("Error in automated test")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "auto":
        automated_test()
    else:
        interactive_test()