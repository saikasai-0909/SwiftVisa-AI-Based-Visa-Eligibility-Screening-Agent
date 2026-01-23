🛂 SwiftVisa: AI-Based Visa Eligibility Screening Agent

🔍 AI-powered UK visa eligibility screening system built using Streamlit and Large Language Models (LLMs).

This project helps users determine whether they are ELIGIBLE or NOT ELIGIBLE for various UK visa categories by evaluating user-provided information against official UK visa policy documents.

📌 Project Overview
SwiftVisa is designed as a policy-driven eligibility assistant that:
Collects applicant details through a guided UI
Compares inputs with official UK visa rules
Produces a clear eligibility decision with reasoning


🎯 Supported Visa Types
Student Visa
Graduate Visa
Skilled Worker Visa
Health & Care Visa
Visitor Visa

Each visa type includes dedicated eligibility entities and validation rules based on UK Home Office policies.

⚙️ How the System Works
User selects a visa type
User enters:
Common applicant details
Visa-specific eligibility details

The system:
Loads relevant UK visa policy PDF
Extracts policy content
Sends structured data to an LLM (Groq – LLaMA)
The model returns:
ELIGIBLE / NOT ELIGIBLE
Bullet-point reasons
Final decision summary

✨ Key Features
📄 Policy-based eligibility evaluation
🤖 AI-assisted reasoning (LLM-powered)
🧠 Deterministic, rule-aligned decisions
🖥️ Clean and professional Streamlit UI
🔐 No storage of user personal data
📚 PDF-based UK visa policy reference

🧱 Tech Stack
Python
Streamlit
Groq API (LLaMA models)
PyPDF2
HTML & CSS (custom UI styling)

📂 Project Structure
├── app.py                          # Main Streamlit application
├── ChunkingVisa.py                 # Policy chunking logic
├── embeddings.py                   # Text embedding generation
├── retriever.py                    # Policy retrieval logic
├── Vectordatabase.py               # Vector database handling
├── config.json                     # Configuration file
├── users.json                      # User-related configuration
├── requirements.txt                # Project dependencies
├── DataSets/
│   ├── Student Visa.pdf
│   ├── Graduate Visa.pdf
│   ├── Skilled Worker Visa.pdf
│   ├── Health Visa.pdf
│   └── Visitor Visa.pdf

🔐 Security Note

API keys are not hardcoded
Environment variables are used for sensitive credentials
No user data is stored or logged

🚀 Deployment
This project is designed for deployment using Streamlit Cloud.
Deployment may require repository owner permissions.

📜 License
This project is licensed under the MIT License.

👤 Author
V. Kiran Kumar Reddy
Project developed as part of internship work
