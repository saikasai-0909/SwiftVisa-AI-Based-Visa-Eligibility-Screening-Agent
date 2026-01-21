import os
import json
import chromadb
import google.generativeai as genai

# --- CONFIGURATION ---
api_key = os.getenv("GEMINI_KEY") 
genai.configure(api_key=api_key)
MODEL_NAME = "gemini-2.5-flash"

# Setup Chroma
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name="swiftvisa_embeddings")

def run_rag_pipeline(user_profile: dict):
    # 1. Retrieval
    visa_type = user_profile.get("Profile", {}).get("visa_type", "UK Visa")
    query = f"UK {visa_type} visa eligibility criteria, mandatory documents, financial requirements, process steps, conditions of stay."
    
    results = collection.query(query_texts=[query], n_results=8)
    context = "\n\n".join(results["documents"][0])

    # 2. Decision Prompt (WITH 95% SAFETY CAP)
    prompt = f"""
    You are a robotic logic engine for UK Visa processing.

    POLICY CONTEXT (The Rules):
    {context}

    APPLICANT DATA (The Facts):
    {json.dumps(user_profile, indent=2)}

    *** CRITICAL INSTRUCTIONS ***
    1. **ELIGIBILITY VERDICT (STRICT):**
       - Determine "ELIGIBLE" or "NOT ELIGIBLE" based *ONLY* on the specific Yes/No/Value answers provided in APPLICANT DATA.
       - If a requirement (e.g., "TB Test") is NOT mentioned in APPLICANT DATA, **DO NOT** let it affect the verdict. Do not fail the user for missing parameters. Only mark "NOT ELIGIBLE" if a *provided* answer explicitly violates a rule.

    2. **CONFIDENCE SCORE LOGIC (SAFETY CAP):**
       - Calculate confidence based on how many mandatory requirements are explicitly satisfied by the data.
       - **IMPORTANT:** NEVER output 100%. The maximum allowed score is **95%**.
       - Reason: Final discretion always lies with the UK Home Office caseworker. 95% represents "Theoretically Eligible based on provided data."

    3. **GUIDANCE (MAXIMUM DETAIL):**
       - **Document Checklist:** List standard mandatory documents from the POLICY CONTEXT (e.g., "TB Test", "Valid Passport") even if unmentioned by user.
       - **Next Steps:** Extract the FULL application process flow (e.g., Pay IHS -> Biometrics -> Interview -> Wait times).
       - **Dos (Permissions):** STRICTLY list what the user CAN do **after arriving in the UK** (e.g., "Work up to 20 hours/week", "Bring dependents").
       - **Don'ts (Restrictions):** STRICTLY list what the user CANNOT do **after arriving in the UK** (e.g., "Claim public funds", "Switch visa category").
       - **Future Options:** Extract all mentioned routes for extension, settlement (ILR), or switching.
    4. **NEUTRALITY RULE (MANDATORY):**
       - **NEVER** assume the applicant's gender based on their name.
       - **ALWAYS** use "The applicant" or "They/Their" in your reasoning and explanation.
       - Do not use "He/Him" or "She/Her" under any circumstances.
       
    OUTPUT FORMAT (JSON ONLY):
    {{
        "verdict": "ELIGIBLE" or "NOT ELIGIBLE",
        "confidence_score": 0-95,
        "explanation": "Reasoning based strictly on the provided answers.",
        "satisfied_requirements": ["List of provided inputs that passed"],
        "unsatisfied_requirements": ["List of provided inputs that failed"],
        "checklist": ["Comprehensive list of ALL required documents based on Policy Context"],
        "next_steps": ["Step 1", "Step 2", "Step 3..."],
        "dos": ["You CAN work...", "You CAN study..."],
        "donts": ["You CANNOT claim funds...", "You CANNOT overstay..."],
        "future_options": ["Option 1", "Option 2..."],
        "remedy": "Advice for failures (or null)"
    }}
    """

    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt, 
        generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
    )
    
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return {
            "verdict": "ERROR", 
            "explanation": "Technical error in analysis.",
            "confidence_score": 0,
            "checklist": [],
            "next_steps": []
        }