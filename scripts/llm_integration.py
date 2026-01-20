"""
SwiftVisa - LLM Integration with LM Studio
Handles communication with Llama 3 model via LM Studio
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import requests
import json
from typing import Dict, List, Optional

from config.config import (
    LM_STUDIO_CONFIG,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    ELIGIBILITY_PROMPT_TEMPLATE,
    LOGGING_CONFIG
)

# Setup logging
logging.basicConfig(
    level=LOGGING_CONFIG["level"],
    format=LOGGING_CONFIG["format"]
)
logger = logging.getLogger(__name__)


class LMStudioLLM:
    """LLM interface for LM Studio"""
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None
    ):
        """
        Initialize LM Studio LLM client
        
        Args:
            base_url: LM Studio API endpoint
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.base_url = base_url or LM_STUDIO_CONFIG["base_url"]
        self.model = model or LM_STUDIO_CONFIG["model"]
        self.temperature = temperature or LM_STUDIO_CONFIG["temperature"]
        self.max_tokens = max_tokens or LM_STUDIO_CONFIG["max_tokens"]
        
        logger.info("🤖 Initializing LM Studio LLM client...")
        logger.info(f"  Base URL: {self.base_url}")
        logger.info(f"  Model: {self.model}")
        
        # Test connection
        self._test_connection()
        
        logger.info("✅ LM Studio LLM client initialized successfully")
    
    def _test_connection(self):
        """Test connection to LM Studio server"""
        try:
            response = requests.get(
                f"{self.base_url}/models",
                timeout=5
            )
            if response.status_code == 200:
                logger.info("✅ Successfully connected to LM Studio")
            else:
                logger.warning(f"⚠️  LM Studio responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to connect to LM Studio: {e}")
            logger.error(f"Please ensure LM Studio is running at {self.base_url}")
            raise ConnectionError(f"Cannot connect to LM Studio at {self.base_url}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        stream: bool = False
    ) -> str:
        """
        Generate response from LLM
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            stream: Whether to stream response
            
        Returns:
            Generated text response
        """
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
            "stream": stream
        }
        
        try:
            logger.debug(f"Sending request to LM Studio")
            logger.debug(f"Message count: {len(messages)}")
            
            # Log message lengths
            for i, msg in enumerate(messages):
                logger.debug(f"Message {i} ({msg['role']}): {len(msg['content'])} chars")
            
            response = requests.post(
                url,
                json=payload,
                timeout=600  # Increased to 600 seconds for slower GPUs
            )
            
            # Better error handling
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"LM Studio error response: {error_detail}")
                
                # Check if it's a context length issue
                if "context" in error_detail.lower() or "length" in error_detail.lower():
                    raise ValueError(
                        "Context length exceeded. Try reducing the number of retrieved chunks "
                        "or shortening the prompt."
                    )
            
            response.raise_for_status()
            
            result = response.json()
            generated_text = result['choices'][0]['message']['content']
            
            logger.debug(f"Received response from LLM ({len(generated_text)} chars)")
            return generated_text
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error calling LM Studio API: {e}")
            logger.error(f"Response content: {e.response.text if hasattr(e, 'response') else 'No response'}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling LM Studio API: {e}")
            raise
    
    def answer_question(
        self,
        question: str,
        context: str,
        system_prompt: str = None
    ) -> str:
        """
        Answer a question based on provided context
        
        Args:
            question: User question
            context: Retrieved policy context
            system_prompt: Optional custom system prompt
            
        Returns:
            Answer from LLM
        """
        sys_prompt = system_prompt or SYSTEM_PROMPT
        
        user_message = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_message}
        ]
        
        logger.info(f"Answering question: '{question}'")
        response = self.generate(messages)
        
        return response
    
    def evaluate_eligibility(
        self,
        user_profile: Dict,
        visa_type: str,
        context: str
    ) -> str:
        """
        Evaluate visa eligibility based on user profile and policy context
        
        Args:
            user_profile: User information dict
            visa_type: Type of visa being evaluated
            context: Retrieved policy context
            
        Returns:
            Eligibility evaluation from LLM
        """
        # Format user profile
        profile_str = "\n".join([f"{k}: {v}" for k, v in user_profile.items()])
        
        user_message = ELIGIBILITY_PROMPT_TEMPLATE.format(
            context=context,
            user_profile=profile_str,
            visa_type=visa_type.replace('_', ' ').title()
        )
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
        
        logger.info(f"Evaluating eligibility for {visa_type}")
        response = self.generate(messages, temperature=0.2)  # Lower temp for consistency
        
        return response
    
    def chat(
        self,
        user_message: str,
        conversation_history: List[Dict] = None,
        context: str = None
    ) -> str:
        """
        Chat with context awareness
        
        Args:
            user_message: User's message
            conversation_history: Previous conversation
            context: Optional context to include
            
        Returns:
            LLM response
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add context if provided
        if context:
            enhanced_message = f"CONTEXT:\n{context}\n\nUSER QUESTION:\n{user_message}"
        else:
            enhanced_message = user_message
        
        messages.append({"role": "user", "content": enhanced_message})
        
        return self.generate(messages)


# Test function
def test_llm():
    """Test LLM integration"""
    print("\n" + "="*80)
    print("🧪 Testing LM Studio LLM Integration")
    print("="*80)
    
    try:
        llm = LMStudioLLM()
        
        # Test 1: Simple question answering
        print("\n📝 Test 1: Simple Question")
        print("-" * 80)
        
        test_context = """
        UK Student Visa Financial Requirements:
        - You must have at least £1,334 per month for living costs if studying in London
        - You must have at least £1,023 per month for living costs if studying outside London
        - You must show you have had these funds for at least 28 consecutive days
        """
        
        question = "How much money do I need for a student visa if I'm studying in London?"
        
        print(f"Question: {question}")
        print(f"\nContext provided: {test_context[:100]}...")
        
        answer = llm.answer_question(question, test_context)
        print(f"\n🤖 LLM Answer:\n{answer}")
        
        # Test 2: Eligibility evaluation
        print("\n" + "="*80)
        print("📝 Test 2: Eligibility Evaluation")
        print("-" * 80)
        
        user_profile = {
            "Age": 25,
            "Nationality": "Indian",
            "Education": "Bachelor's Degree in Computer Science",
            "English Test": "IELTS 7.0",
            "University Offer": "Yes - University of London",
            "Financial Proof": "£15,000 in bank account"
        }
        
        print(f"User Profile:")
        for k, v in user_profile.items():
            print(f"  {k}: {v}")
        
        evaluation = llm.evaluate_eligibility(
            user_profile=user_profile,
            visa_type="student",
            context=test_context
        )
        
        print(f"\n🤖 LLM Evaluation:\n{evaluation}")
        
        print("\n" + "="*80)
        print("✅ LLM Integration Test Complete!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("\nPlease ensure:")
        print("1. LM Studio is running")
        print("2. Server is accessible at http://192.168.1.38:1234")
        print("3. meta-llama-3-8b-instruct model is loaded")


if __name__ == "__main__":
    test_llm()