from rag_pipeline import RAGPipeline
import re

# -------------------------------
# Initialize RAG
# -------------------------------
rag = RAGPipeline(
    vector_index_dir=r"C:\\Project\\vector_db\\faiss",
    embedding_model="thenlper/gte-base",
    ollama_model="llama3.2",
    prompt_path=r"C:\\Project\\src\\Prompts\\visa_prompt.txt"
)

# -------------------------------------------------
# Public function (called from Streamlit)
# -------------------------------------------------
def run_eligibility_check(user_payload: dict) -> dict:
    query = build_eligibility_query(user_payload)

    result = rag.answer(query)
    answer_text = result["answer"]
    print(answer_text)
    return {
        "eligibility_status": extract_single_value(answer_text, "Eligibility Status"),
        "reasons": extract_bullets(answer_text, "Reasons"),
        "required_documents": extract_bullets(answer_text, "Required Documents"),

        "missing_documents": extract_bullets(answer_text, "Missing Documents"),
        "recommendations": extract_bullets(answer_text, "Recommendations"),
        "summary": extract_paragraph(answer_text, "Explanation"),
        "citations": result.get("citations", [])
    }


# --------------------------------
# Query Builder
# --------------------------------
def build_eligibility_query(payload: dict) -> str:
    common = payload["common_details"]
    visa_specific = payload.get("visa_specific_details", {})

    return f"""
The following information is a COMPLETE UK visa application.
All applicant details required for eligibility assessment have been provided.
Do NOT ask for additional information.

TASK:
Assess eligibility for a UK {common["visa_type"]}.

APPLICANT DETAILS:
Full Name: {common["full_name"]}
Date of Birth: {common["date_of_birth"]}
Nationality: {common["nationality"]}
Passport Number: {common["passport_number"]}
Passport Issue Date: {common["passport_issue_date"]}
Passport Expiry Date: {common["passport_expiry_date"]}
Country of Application / Current Location: {common["country_of_application"]}
Purpose of Visit: {common["purpose_of_visit"]}
Intended Travel / Start Date: {common["intended_travel_date"]}
Intended Length of Stay: {common["intended_length_of_stay"]}
Funds Available: {common["funds_available"]}
English Language Requirement Met: {common["english_language_met"]}
Criminal History Declaration: {common["criminal_history"]}
Previous UK Visa Refusal: {common["previous_uk_visa_refusal"]}

VISA-SPECIFIC DETAILS:
{format_visa_specific(visa_specific)}

INSTRUCTIONS:
- Use ONLY the retrieved UK visa policy documents
- Do NOT request additional details
- You MUST determine eligibility
- Follow the OUTPUT FORMAT exactly
"""

def format_visa_specific(data: dict) -> str:
    if not data:
        return "None"

    return "\n".join(
        f"{k.replace('_', ' ').title()}: {v}" for k, v in data.items()
    )

# --------------------------------
# ROBUST PARSERS
# --------------------------------
def extract_single_value(text: str, section: str) -> str:
    pattern = rf"{section}:\s*\n?\s*(.+)"
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1).strip() if match else "Not explicitly stated"


def extract_bullets(text: str, section: str) -> list:
    pattern = rf"{section}:\s*(.*?)(?:\n[A-Z][a-z ]+:|\Z)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)

    if not match:
        return ["None"]

    block = match.group(1).strip()
    items = []

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("-", "•", "*")):
            items.append(line.lstrip("-•* ").strip())
        else:
            items.append(line)   # <-- KEY FIX

    return items if items else ["None"]


def extract_paragraph(text: str, section: str) -> str:
    pattern = rf"{section}:\s*(.*)"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else "None"
