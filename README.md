# SwiftVisa – AI-Based Visa Eligibility Screening Agent

SwiftVisa is an AI-powered visa eligibility screening system designed to assist users in understanding their eligibility for UK visas using structured inputs and publicly available immigration policy documents.

The system leverages Retrieval-Augmented Generation (RAG) to provide policy-grounded, transparent, and explainable eligibility insights through an interactive interface.

## Key Features
- AI-driven visa eligibility assessment
- Retrieval-Augmented Generation (RAG) architecture
- Semantic search using FAISS vector database
- PDF-based policy document ingestion and chunking
- Streamlit-based interactive user interface
- Reduced hallucination through document-grounded responses

## System Architecture
1. Publicly available visa policy documents are used as reference material  
2. Documents are processed offline and split into semantic chunks  
3. Embeddings are generated using Sentence Transformers  
4. FAISS is used for efficient vector-based retrieval  
5. Relevant policy context is passed to the language model  
6. Responses are generated based on retrieved content  

## Tech Stack
- Python
- Streamlit
- Llama 3.1
- Hugging Face Transformers
- Sentence Transformers (MiniLM)
- FAISS
- PyPDF

## Disclaimer
This project is an assistive decision-support system built for educational purposes.  
It does not replace official legal or immigration advice.
