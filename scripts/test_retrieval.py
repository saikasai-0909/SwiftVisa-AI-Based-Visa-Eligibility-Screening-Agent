"""
SwiftVisa - Test Retrieval System
Test the FAISS vector store with various queries
"""

import json
from build_vectorstore import VectorStoreBuilder


class RetrievalTester:
    """Test retrieval functionality"""
    
    def __init__(self):
        self.builder = VectorStoreBuilder()
        self.index, self.chunks_metadata = self.builder.load_vectorstore()
        
        if self.index is None:
            print("❌ Vector store not found. Please run build_vectorstore.py first!")
            exit(1)
    
    def interactive_search(self):
        """Interactive search interface"""
        print("\n" + "="*80)
        print("🔍 SwiftVisa - Interactive Search")
        print("="*80)
        print("\nType your query or 'quit' to exit")
        print("Examples:")
        print("  • What are the English language requirements for student visa?")
        print("  • How long can I stay with a visitor visa?")
        print("  • Can I bring my family on a skilled worker visa?")
        print("="*80)
        
        while True:
            query = input("\n💬 Your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not query:
                continue
            
            self.builder.test_search(self.index, self.chunks_metadata, query, k=3)
    
    def run_test_suite(self):
        """Run comprehensive test queries"""
        print("\n" + "="*80)
        print("🧪 Running Test Suite")
        print("="*80)
        
        test_cases = [
            # Student Visa
            {
                "category": "Student Visa",
                "queries": [
                    "What are the financial requirements for a student visa?",
                    "Can I work while on a student visa?",
                    "What documents do I need for student visa application?"
                ]
            },
            # Graduate Visa
            {
                "category": "Graduate Visa",
                "queries": [
                    "How long is the graduate visa valid for?",
                    "What are the eligibility criteria for graduate visa?",
                    "Can I switch from student to graduate visa?"
                ]
            },
            # Skilled Worker
            {
                "category": "Skilled Worker Visa",
                "queries": [
                    "What is the minimum salary for skilled worker visa?",
                    "Do I need a job offer for skilled worker visa?",
                    "What occupations are eligible for skilled worker visa?"
                ]
            },
            # Health Care Worker
            {
                "category": "Health Care Worker Visa",
                "queries": [
                    "What are the requirements for health care worker visa?",
                    "Is there a fee discount for NHS workers?",
                    "What jobs qualify for health care worker visa?"
                ]
            },
            # Visitor Visa
            {
                "category": "Standard Visitor Visa",
                "queries": [
                    "How long can I stay on a visitor visa?",
                    "Can I work with a standard visitor visa?",
                    "What activities are allowed on visitor visa?"
                ]
            }
        ]
        
        for test_case in test_cases:
            print(f"\n{'='*80}")
            print(f"📋 Testing: {test_case['category']}")
            print(f"{'='*80}")
            
            for query in test_case['queries']:
                self.builder.test_search(self.index, self.chunks_metadata, query, k=2)
                print()
    
    def check_coverage(self):
        """Check coverage of all visa types"""
        print("\n" + "="*80)
        print("📊 Vector Store Coverage Analysis")
        print("="*80)
        
        # Count chunks by visa type
        visa_counts = {}
        for chunk in self.chunks_metadata:
            visa_type = chunk['metadata'].get('visa_type', 'unknown')
            visa_counts[visa_type] = visa_counts.get(visa_type, 0) + 1
        
        print(f"\nTotal Chunks: {len(self.chunks_metadata)}")
        print(f"Total Vectors: {self.index.ntotal}")
        print(f"\nBreakdown by Visa Type:")
        
        for visa_type, count in sorted(visa_counts.items()):
            percentage = (count / len(self.chunks_metadata)) * 100
            print(f"  • {visa_type.replace('_', ' ').title()}: {count} chunks ({percentage:.1f}%)")
        
        # Load metadata
        try:
            with open("vectorstore/metadata.json", 'r') as f:
                metadata = json.load(f)
            
            print(f"\n📅 Build Information:")
            print(f"  • Build Date: {metadata.get('build_date', 'N/A')}")
            print(f"  • Model: {metadata.get('model_name', 'N/A')}")
            print(f"  • Embedding Dimension: {metadata.get('embedding_dim', 'N/A')}")
            print(f"  • Source Documents: {metadata.get('source_docs', 'N/A')}")
        except:
            pass
        
        print("="*80)


def main():
    """Main execution"""
    tester = RetrievalTester()
    
    print("\n🎯 Choose an option:")
    print("1. Run comprehensive test suite")
    print("2. Interactive search")
    print("3. Check vector store coverage")
    print("4. All of the above")
    
    choice = input("\nYour choice (1-4): ").strip()
    
    if choice == "1":
        tester.run_test_suite()
    elif choice == "2":
        tester.interactive_search()
    elif choice == "3":
        tester.check_coverage()
    elif choice == "4":
        tester.check_coverage()
        tester.run_test_suite()
        tester.interactive_search()
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()