# 🇬🇧 UK Visa Eligibility Checker

A professional decision support system for analyzing UK Visa eligibility against the **January 2026 Immigration Rules**. Built with Streamlit, Pinecone, and Ollama (RAG).

## 🚀 Features

- **Identity Profile Verification**: Collect and verify 18+ identity entities.
- **Route-Specific Eligibility**: Support for multiple visa categories:
  - Student Visa
  - Skilled Worker Visa
  - Graduate Visa
  - Health & Care Visa
  - Visitor Visa
- **RAG-Powered Assessment**: Uses Retrieval-Augmented Generation (Pinecone + Ollama) to provide official-style assessment reports.
- **GOV.UK Inspired UI**: Clean, professional theme based on UK Government design principles.
- **Downloadable Reports**: Generate and download comprehensive eligibility assessment reports.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Embeddings**: HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database**: [Pinecone](https://www.pinecone.io/)
- **LLM Engine**: [Ollama](https://ollama.com/) (using `llama2` or compatible models)
- **Framework**: [LangChain](https://www.langchain.com/)

## 📋 Prerequisites

1. **Python 3.8+**
2. **Pinecone API Key**: Required for vector search functionality.
3. **Ollama**: Installed and running locally for the assessment engine.
   ```bash
   ollama pull llama2
   ollama serve
   ```

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd Visa
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   PINECONE_API_KEY=your_pinecone_api_key
   OLLAMA_URL=http://localhost:11434/api/generate
   OLLAMA_MODEL=llama2
   ```

## 🏃 Running the Application

Start the Streamlit server:
```bash
streamlit run app.py
```

## 📤 Deployment to Streamlit Cloud

1. Push your code to a GitHub repository.
2. Log in to [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your GitHub repository.
4. Add your **Secrets** (Environment Variables) in the Streamlit Cloud dashboard:
   - `PINECONE_API_KEY`
5. Click **Deploy**.

> [!NOTE]
> For Streamlit Cloud deployment, ensure your Ollama endpoint is accessible or replace the local Ollama logic with a cloud-based LLM provider (like Groq or OpenAI) for production use.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
