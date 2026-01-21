README FILE

**"NOTE- INSTRUCTION FOR RUNNING THE PROJECT ": INSTALL ALL DEPENDENCIES AND THEN RUN embedding.ipynb ONCE FOR GENERATING THE EMBEDDING IN VECTOR DATABSE (since it caused error while pushing) AND THEN RUN THE app.py TO SUCESSFULLY RUN THE PROJECT.**

SwiftVisa — A Visa Eligibility Assistant (RAG + LLM + Streamlit)

SwiftVisa is an intelligent visa eligibility assistant built for providing grounded, explainable and non-hallucinatory responses based on official VISA policy documents. It help to surf across many Visa Categories for various countries using a Retrieval-Augmented Generation (RAG) architecture. 

---

Project Goals

- Assist users in determining the visa eligibility
- Provide structured, explainable outputs grounded in policy chunks
- Avoid hallucination by using RAG-based retrieval
- Support both quick eligibility check(user get only a yes/no answer stating whether or not he/she is eligible for that visa) and detailed analysis (providing the user the leverage to ask their queries only related to the visa.)
- Collect visa-specific applicant profile information through UI
- Future-proof and extensible architecture

---

Supported Visa Categories (UK)

- Skilled Worker Visa
- Graduate Visa
- Student Visa
- Health & Care Worker Visa
- Visitor Visa

---

System Architecture Overview

End-to-End Pipeline

User → Streamlit UI → RAG Engine → Vector DB → LM Studio LLM → Structured Output


RAG Components

| Component | Technology |
| Chunking | Semantic chunking technique with overlap |
| Embeddings | E5-Large (retrieval optimized) |
| Vector DB | ChromaDB |
| Retrieval | ANN + cosine similarity (based on semantic retrieval techniques) |
| LLM | Llama-3.2-3B via LM Studio |
| Output | Structured + grounded + explainable |

---

Data Source

Official UK GOV policy PDFs for:

- Skilled Worker
- Graduate
- Student
- Health & Care Worker
- Visitor

PDFs were pre-processed, chunked, embedded and stored in Chroma.

---

Retrieval Strategy

- Top-K semantic retrieval (K = 3)
- A hardcoded visa-type filtering
- Policy chunks retrieved + cited


---

Embedding Details

Model: `intfloat/e5-large-v2`

Reasons for selection:

- Retrieval-optimized query/passage asymmetry
- High Recall@K performance
- Lower hallucination during grounding
- Outperforms MiniLM, USE etc. in policy/legal text

---

Metrics observed:

Recall@1 ≈ 70%+
Recall@3 ≈ 95%+
Recall@5 ≈ 99%
MRR ≈ ~0.85


---

Vector Database

ChromaDB (persistent)

Features used:

- Persistent indexing
- Metadata storage
- Cosine similarity search

Stored for each chunk:
chunk_text
visa_type
chunk_index
embedding
source_file


---

LLM Integration

- Local inference via LM Studio
- Model: Llama 3.2 — 3B Instruct
- Zero temperature (deterministic)
- JSON/structured response enforcement

Responsibilities of LLM:

- Interpret user profile
- Compare eligibility conditions
- Explain reasoning
- Flag missing information
- Avoid hallucination

---

UI / UX Layer (Streamlit):

Designed to emulate a visa application workflow through:

Phase 0 — Intent Selection
> Quick vs Detailed mode

Phase 1 — Common Applicant Entities
> Passport, funds, English, travel intent, etc.

Phase 2 — Visa Category Selection

Phase 3 — Visa-Specific Fields

Phase 4 — Task Mode
> Ask query or compute eligibility (asking query is there only for detailed analysis)

Phase 5 — RAG Result + Explanation

---

Eligibility Result Modes

✔ Quick mode
- Returns: Eligible / Not Eligible / Depends
- Progressive disclosure for missing information

✔ Detailed mode
- Open-ended Q/A grounded in policy chunks
- A detailed explanation with raw facts and figures from the provided information about the eligibility as well as regarding the question asked

---


Also required:

- Local LM Studio running on `http://127.0.0.1:1234/v1/chat/completions`
- ChromaDB persistence path configured
- Embedded policy chunks available

---

Run Instructions:

```IN TERMINAL
streamlit run app.py





