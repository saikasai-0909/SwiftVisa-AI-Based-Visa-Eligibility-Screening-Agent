SwiftVisa: AI-Based Visa Eligibility Screening Agent
Overview

SwiftVisa is an AI-powered UK visa eligibility screening system built using Streamlit and Large Language Models (LLMs).
The application evaluates whether an applicant is ELIGIBLE or NOT ELIGIBLE for different UK visa categories by comparing user-provided information against official UK visa policy documents.

The system is designed to provide:

Transparent eligibility decisions

Policy-driven reasoning

A structured and user-friendly screening experience

Supported Visa Types

Student Visa

Graduate Visa

Skilled Worker Visa

Health & Care Visa

Visitor Visa

Each visa type has its own dedicated eligibility entities and rules, aligned with UK immigration policy documents.

How the System Works

The user selects a visa type

The user enters:

Common applicant details

Visa-specific eligibility information

The system:

Extracts relevant text from official UK visa policy PDF files

Structures the applicant data

Sends both policy rules and user inputs to an LLM

The AI model evaluates the information and returns:

ELIGIBLE or NOT ELIGIBLE

Clear bullet-point reasoning

A final decision summary

Key Features

📄 Policy-driven eligibility evaluation using official UK visa PDFs

🤖 AI-assisted reasoning with deterministic, rule-aligned outputs

🧠 Visa-specific eligibility entities for accurate screening

🖥️ Clean, professional Streamlit UI

🔐 Secure API key handling using environment variables

🚫 No storage of user personal data

🌐 Cloud-deployable architecture

Technology Stack

Python

Streamlit

Groq API (LLaMA models)

PyPDF2 (PDF policy extraction)

HTML / CSS (custom UI styling)

Project Structure
├── app.py                  # Main Streamlit application
├── DataSets/               # UK visa policy PDF documents
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation

Security Practices

API keys are never hard-coded

Secrets are managed using environment variables

GitHub secret scanning compliance maintained

No sensitive data is logged or stored

Deployment

The application is designed for deployment on Streamlit Cloud using:

GitHub repository integration

Branch-based deployment

Secure secret configuration

Once deployed, the application is accessible via a public Streamlit URL.

Disclaimer

This application is intended for educational and screening purposes only.
It does not constitute legal advice or an official immigration decision.
Final visa decisions are made solely by UK Visas and Immigration (UKVI).

Author

V.KiranKumarReddy
AI-Based Visa Eligibility Screening Project
