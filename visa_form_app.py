"""
UK Visa Application Form with RAG Integration
Streamlit UI for collecting visa application details and querying visa policies
"""

import streamlit as st
import os
from datetime import datetime, date
from retriever import VisaRetriever
from openai import OpenAI

# =====================================================
# Configuration
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "visa_faiss.index")
FAISS_METADATA_PATH = os.path.join(BASE_DIR, "visa_faiss_metadata.pkl")

LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"
LLM_MODEL_NAME = "meta-llama-3.2-3b-instruct"

# =====================================================
# Initialize Session State
# =====================================================
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

# =====================================================
# Helper Functions
# =====================================================
@st.cache_resource
def load_retriever():
    """Load FAISS retriever"""
    return VisaRetriever(
        index_path=FAISS_INDEX_PATH,
        metadata_path=FAISS_METADATA_PATH
    )

def get_llm_client():
    """Get LLM client"""
    return OpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )

def query_visa_policy(question: str, user_context: dict = None):
    """Query visa policy using RAG"""
    try:
        retriever = load_retriever()
        llm = get_llm_client()
        
        # Retrieve relevant documents with more results for better context
        retrieved_docs = retriever.retrieve(question, top_k=5)
        
        if not retrieved_docs:
            return "❌ No relevant visa policy found."
        
        # Build context
        context_blocks = []
        for i, doc in enumerate(retrieved_docs, 1):
            meta = doc["metadata"]
            context_blocks.append(
                f"""--- Document {i} ---
Source: {os.path.basename(meta.get('source', 'Unknown'))}

{doc['text']}"""
            )
        
        context = "\n".join(context_blocks)
        
        # Add user context if provided
        user_info = ""
        if user_context:
            user_info = "\n\nUSER APPLICATION DETAILS:\n"
            for key, value in user_context.items():
                if value:  # Only include non-empty values
                    user_info += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        # Build augmented prompt
        augmented_prompt = f"""VISA POLICY CONTEXT:
{context}
{user_info}

USER QUESTION:
{question}
"""
        
        # Call LLM for detailed analysis
        response = llm.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": """You are an AI Visa Eligibility Screening Assistant for UK visas.

Your task:
- Analyze UK visa eligibility based ONLY on the official visa policy documents provided.
- Provide specific, detailed answers using the context.
- When user details are provided, assess their eligibility against the requirements.

Rules:
- Use ONLY the provided context from visa policy documents.
- Do NOT use external knowledge or make assumptions.
- If specific information is missing from the documents, clearly state: "This information is not available in the provided visa policy documents."
- Be detailed and cite specific requirements (fees, amounts, conditions).
- Structure your response with clear sections and bullet points.

Response Format:
1. Eligibility Requirements (list key requirements)
2. Financial Requirements (if applicable - specific amounts)
3. Required Documents (list documents needed)
4. Fees and Timeline (if available)
5. Assessment (if user details provided - evaluate their situation)
"""},
                {"role": "user", "content": augmented_prompt},
            ],
            temperature=0.2,
        )
        
        detailed_answer = response.choices[0].message.content.strip()
        
        # Make eligibility determination
        eligibility_prompt = f"""Based on the following analysis and user details, make a clear ELIGIBLE or NOT ELIGIBLE determination.

USER DETAILS:
{user_info}

ANALYSIS:
{detailed_answer}

Respond with ONLY ONE of these formats:
- "ELIGIBLE: [brief reason why they are eligible]"
- "NOT ELIGIBLE: [brief reason why they are not eligible]"
- "INSUFFICIENT INFORMATION: [what information is missing]"

Be strict - only say ELIGIBLE if all key requirements are clearly met based on the provided information."""

        eligibility_response = llm.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a visa eligibility assessor. Make clear determinations."},
                {"role": "user", "content": eligibility_prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )
        
        eligibility_decision = eligibility_response.choices[0].message.content.strip()
        
        return {"decision": eligibility_decision, "details": detailed_answer}
    
    except Exception as e:
        return {
            "decision": "ERROR: Unable to connect to LLM",
            "details": f"⚠️ Error: {str(e)}\n\nPlease ensure LM Studio is running on http://localhost:1234"
        }

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="UK Visa Application Form",
    page_icon="🇬🇧",
    layout="wide"
)

# =====================================================
# Main UI
# =====================================================
st.title("🇬🇧 UK Visa Application Form")
st.markdown("Fill out the form below and get instant eligibility guidance using AI-powered visa policy retrieval")

# =====================================================
# Visa Type Selection
# =====================================================
visa_type = st.selectbox(
    "Select Visa Type",
    ["", "Graduate Visa", "Student Visa", "Skilled Worker Visa", 
     "Health & Care Visa", "Visitor Visa"],
    help="Choose the type of UK visa you want to apply for"
)

if not visa_type:
    st.info("👆 Please select a visa type to begin")
    st.stop()

# =====================================================
# Common Fields (All Visa Types)
# =====================================================
st.header("📋 Personal Information")

col1, col2 = st.columns(2)

with col1:
    full_name = st.text_input("Full Name (as per passport) *", key="full_name")
    date_of_birth = st.date_input("Date of Birth *", min_value=date(1924, 1, 1), max_value=date.today(), key="dob")
    nationality = st.text_input("Nationality *", key="nationality")
    passport_number = st.text_input("Passport Number *", key="passport_num")
    passport_issue_date = st.date_input("Passport Issue Date *", key="pass_issue")

with col2:
    passport_expiry_date = st.date_input("Passport Expiry Date *", key="pass_expiry")
    current_location = st.text_input("Country of Application / Current Location *", key="location")
    intended_travel_date = st.date_input("Intended Travel / Start Date", key="travel_date")
    length_of_stay = st.number_input("Intended Length of Stay (months)", min_value=0, max_value=120, key="stay_length")
    funds_available = st.number_input("Funds Available (£)", min_value=0, step=100, key="funds")

st.header("📞 Contact Information")
col1, col2 = st.columns(2)
with col1:
    email = st.text_input("Email Address *", key="email")
    phone = st.text_input("Phone Number *", key="phone")
with col2:
    address = st.text_area("Current Address *", key="address")

st.header("📝 Declarations")
col1, col2, col3 = st.columns(3)
with col1:
    english_requirement = st.selectbox("English Language Requirement Met", ["", "Yes", "No"], key="english")
with col2:
    criminal_history = st.selectbox("Criminal History Declaration", ["", "Yes", "No"], key="criminal")
with col3:
    previous_refusal = st.selectbox("Previous UK Visa Refusal", ["", "Yes", "No"], key="refusal")

# =====================================================
# Visa-Specific Fields
# =====================================================
st.header(f"🎯 {visa_type} Specific Information")

visa_specific_data = {}

# ============ GRADUATE VISA ============
if visa_type == "Graduate Visa":
    col1, col2 = st.columns(2)
    
    with col1:
        visa_specific_data['currently_in_uk'] = st.selectbox("Currently in UK? *", ["", "Yes", "No"])
        visa_specific_data['current_uk_visa_type'] = st.text_input("Current UK Visa Type (e.g., Student/Tier 4)")
        visa_specific_data['course_completed'] = st.selectbox("Course Completed? *", ["", "Yes", "No"])
        visa_specific_data['course_level_completed'] = st.selectbox(
            "Course Level Completed",
            ["", "Bachelor's Degree", "Master's Degree", "PhD", "Other"]
        )
        visa_specific_data['education_provider_is_licensed'] = st.selectbox(
            "Education Provider is Licensed? *",
            ["", "Yes", "No"]
        )
    
    with col2:
        visa_specific_data['provider_reported_completion'] = st.selectbox(
            "Provider Reported Completion to Home Office?",
            ["", "Yes", "No"]
        )
        visa_specific_data['original_cas_reference'] = st.text_input("Original CAS Reference Number")
        visa_specific_data['student_visa_valid_on_application'] = st.selectbox(
            "Student Visa Valid on Application Date?",
            ["", "Yes", "No"]
        )

# ============ STUDENT VISA ============
elif visa_type == "Student Visa":
    col1, col2 = st.columns(2)
    
    with col1:
        visa_specific_data['has_cas'] = st.selectbox("Have CAS (Confirmation of Acceptance for Studies)? *", ["", "Yes", "No"])
        visa_specific_data['cas_reference_number'] = st.text_input("CAS Reference Number")
        visa_specific_data['education_provider_is_licensed'] = st.selectbox(
            "Education Provider is Licensed? *",
            ["", "Yes", "No"]
        )
        visa_specific_data['course_level'] = st.selectbox(
            "Course Level",
            ["", "Bachelor's Degree", "Master's Degree", "PhD", "Foundation", "Pre-sessional English", "Other"]
        )
        visa_specific_data['course_full_time'] = st.selectbox("Course Full-Time? *", ["", "Yes", "No"])
        visa_specific_data['course_start_date'] = st.date_input("Course Start Date")
    
    with col2:
        visa_specific_data['course_end_date'] = st.date_input("Course End Date")
        visa_specific_data['course_duration_months'] = st.number_input("Course Duration (months)", min_value=0, max_value=120)
        visa_specific_data['meets_financial_requirement'] = st.selectbox(
            "Meets Financial Requirement? *",
            ["", "Yes", "No"]
        )
        visa_specific_data['funds_held_for_28_days'] = st.selectbox(
            "Funds Held for 28 Days?",
            ["", "Yes", "No"]
        )
        visa_specific_data['english_requirement_met_student'] = st.selectbox(
            "English Requirement Met?",
            ["", "Yes", "No"]
        )

# ============ SKILLED WORKER VISA ============
elif visa_type == "Skilled Worker Visa":
    col1, col2 = st.columns(2)
    
    with col1:
        visa_specific_data['job_offer_confirmed'] = st.selectbox("Job Offer Confirmed? *", ["", "Yes", "No"])
        visa_specific_data['employer_is_licensed_sponsor'] = st.selectbox(
            "Employer is Licensed Sponsor? *",
            ["", "Yes", "No"]
        )
        visa_specific_data['certificate_of_sponsorship_issued'] = st.selectbox(
            "Certificate of Sponsorship Issued?",
            ["", "Yes", "No"]
        )
        visa_specific_data['cos_reference_number'] = st.text_input("CoS Reference Number")
        visa_specific_data['job_title'] = st.text_input("Job Title")
        visa_specific_data['soc_code'] = st.text_input("SOC Code (Standard Occupational Classification)")
    
    with col2:
        visa_specific_data['job_is_eligible_occupation'] = st.selectbox(
            "Job is Eligible Occupation?",
            ["", "Yes", "No"]
        )
        visa_specific_data['salary_offered'] = st.number_input("Salary Offered (£ per year)", min_value=0, step=1000)
        visa_specific_data['meets_minimum_salary_threshold'] = st.selectbox(
            "Meets Minimum Salary Threshold?",
            ["", "Yes", "No"]
        )
        visa_specific_data['english_requirement_met_worker'] = st.selectbox(
            "English Requirement Met?",
            ["", "Yes", "No"]
        )
        visa_specific_data['criminal_record_certificate_required'] = st.selectbox(
            "Criminal Record Certificate Required?",
            ["", "Yes", "No"]
        )
        visa_specific_data['criminal_record_certificate_provided'] = st.selectbox(
            "Criminal Record Certificate Provided?",
            ["", "Yes", "No", "N/A"]
        )

# ============ HEALTH & CARE VISA ============
elif visa_type == "Health & Care Visa":
    col1, col2 = st.columns(2)
    
    with col1:
        visa_specific_data['job_offer_confirmed'] = st.selectbox("Job Offer Confirmed? *", ["", "Yes", "No"])
        visa_specific_data['employer_is_licensed_healthcare_sponsor'] = st.selectbox(
            "Employer is Licensed Healthcare Sponsor? *",
            ["", "Yes", "No"]
        )
        visa_specific_data['certificate_of_sponsorship_issued'] = st.selectbox(
            "Certificate of Sponsorship Issued?",
            ["", "Yes", "No"]
        )
        visa_specific_data['cos_reference_number'] = st.text_input("CoS Reference Number")
        visa_specific_data['job_title'] = st.text_input("Job Title (e.g., Nurse, Doctor)")
        visa_specific_data['soc_code'] = st.text_input("SOC Code")
    
    with col2:
        visa_specific_data['job_is_eligible_healthcare_role'] = st.selectbox(
            "Job is Eligible Healthcare Role?",
            ["", "Yes", "No"]
        )
        visa_specific_data['salary_offered'] = st.number_input("Salary Offered (£ per year)", min_value=0, step=1000)
        visa_specific_data['meets_healthcare_salary_rules'] = st.selectbox(
            "Meets Healthcare Salary Rules?",
            ["", "Yes", "No"]
        )
        visa_specific_data['professional_registration_required'] = st.selectbox(
            "Professional Registration Required?",
            ["", "Yes", "No"]
        )
        visa_specific_data['professional_registration_provided'] = st.selectbox(
            "Professional Registration Provided?",
            ["", "Yes", "No", "N/A"]
        )
        visa_specific_data['english_requirement_met_health'] = st.selectbox(
            "English Requirement Met?",
            ["", "Yes", "No"]
        )

# ============ VISITOR VISA ============
elif visa_type == "Visitor Visa":
    col1, col2 = st.columns(2)
    
    with col1:
        visa_specific_data['purpose_of_visit'] = st.text_area("Purpose of Visit *")
        visa_specific_data['purpose_is_permitted'] = st.selectbox(
            "Purpose is Permitted Under Visitor Rules?",
            ["", "Yes", "No", "Unsure"]
        )
        visa_specific_data['intended_length_of_stay_months'] = st.number_input(
            "Intended Length of Stay (months)",
            min_value=0.0,
            max_value=12.0,
            step=0.5
        )
        visa_specific_data['stay_within_6_months_limit'] = st.selectbox(
            "Stay Within 6 Months Limit?",
            ["", "Yes", "No"]
        )
    
    with col2:
        visa_specific_data['accommodation_arranged'] = st.selectbox(
            "Accommodation Arranged?",
            ["", "Yes", "No"]
        )
        visa_specific_data['return_or_onward_travel_planned'] = st.selectbox(
            "Return or Onward Travel Planned?",
            ["", "Yes", "No"]
        )
        visa_specific_data['intends_to_leave_uk'] = st.selectbox(
            "Intends to Leave UK After Visit?",
            ["", "Yes", "No"]
        )
        visa_specific_data['sufficient_funds_for_stay'] = st.selectbox(
            "Sufficient Funds for Stay?",
            ["", "Yes", "No"]
        )

# =====================================================
# Query Section
# =====================================================
st.header("🤖 Ask AI About Your Eligibility")

# Collect all user data
user_data = {
    "visa_type": visa_type,
    "full_name": full_name,
    "date_of_birth": str(date_of_birth),
    "nationality": nationality,
    "passport_number": passport_number,
    "current_location": current_location,
    "funds_available": f"£{funds_available}",
    "english_requirement_met": english_requirement,
    "criminal_history": criminal_history,
    "previous_uk_visa_refusal": previous_refusal,
}

# Add visa-specific data
user_data.update(visa_specific_data)

col1, col2 = st.columns([3, 1])

with col1:
    custom_question = st.text_input(
        "Ask a specific question (optional)",
        placeholder="e.g., What documents do I need? Am I eligible? What are the costs?"
    )

with col2:
    st.write("")
    st.write("")
    ask_button = st.button("🔍 Check Eligibility", use_container_width=True)

# Build comprehensive eligibility query based on user data
def build_eligibility_query(visa_type, user_data, visa_specific_data):
    """Build a detailed query based on user's application details"""
    
    queries = []
    
    if visa_type == "Student Visa":
        # Build specific queries for student visa
        queries.append(f"Student Visa eligibility requirements for {user_data.get('nationality', 'international')} nationals")
        
        if visa_specific_data.get('course_level'):
            queries.append(f"Student Visa requirements for {visa_specific_data['course_level']} level course")
        
        if visa_specific_data.get('has_cas') == 'Yes':
            queries.append("Student Visa CAS (Confirmation of Acceptance for Studies) requirements and process")
        
        if user_data.get('funds_available'):
            queries.append(f"Student Visa financial requirements, maintenance funds, and 28-day rule")
        
        queries.append("Student Visa required documents, application fees, and processing time")
        
    elif visa_type == "Graduate Visa":
        queries.append("Graduate Visa eligibility requirements and qualifying courses")
        
        if visa_specific_data.get('course_level_completed'):
            queries.append(f"Graduate Visa eligibility for {visa_specific_data['course_level_completed']} graduates")
        
        if visa_specific_data.get('currently_in_uk') == 'Yes':
            queries.append("Graduate Visa application process for applicants currently in the UK")
        
        queries.append("Graduate Visa application fees, documents required, and work rights")
        
    elif visa_type == "Skilled Worker Visa":
        queries.append("Skilled Worker Visa eligibility requirements and sponsorship certificate")
        
        if visa_specific_data.get('salary_offered'):
            queries.append(f"Skilled Worker Visa minimum salary threshold and going rate requirements")
        
        if visa_specific_data.get('job_title'):
            queries.append(f"Skilled Worker Visa eligible occupation codes and job requirements")
        
        queries.append("Skilled Worker Visa application fees, English language requirement, and documents needed")
        
    elif visa_type == "Health & Care Visa":
        queries.append("Health and Care Worker Visa eligibility requirements and eligible healthcare roles")
        
        if visa_specific_data.get('job_title'):
            queries.append(f"Health and Care Worker Visa requirements for healthcare professionals")
        
        if visa_specific_data.get('salary_offered'):
            queries.append("Health and Care Worker Visa salary requirements and reduced fees")
        
        queries.append("Health and Care Worker Visa professional registration requirements and application process")
        
    elif visa_type == "Visitor Visa":
        queries.append("UK Visitor Visa eligibility requirements and permitted activities")
        
        if visa_specific_data.get('purpose_of_visit'):
            queries.append(f"UK Visitor Visa rules for {visa_specific_data['purpose_of_visit']}")
        
        if visa_specific_data.get('intended_length_of_stay_months'):
            queries.append("UK Visitor Visa duration limits and 6-month rule")
        
        queries.append("UK Visitor Visa application fees, financial evidence, and prohibited activities")
    
    return " | ".join(queries)


# Default questions based on visa type
default_questions = {
    "Graduate Visa": "What are the eligibility requirements and application process for a UK Graduate Visa? What documents are needed?",
    "Student Visa": "What are the eligibility requirements, financial requirements, and documents needed for a UK Student Visa?",
    "Skilled Worker Visa": "What are the eligibility requirements, salary thresholds, and sponsorship requirements for a UK Skilled Worker Visa?",
    "Health & Care Visa": "What are the eligibility requirements, salary rules, and professional registration requirements for a UK Health & Care Visa?",
    "Visitor Visa": "What are the eligibility requirements, permitted activities, and restrictions for a UK Visitor Visa?"
}

if ask_button:
    if custom_question:
        question = custom_question
    else:
        # Build comprehensive query from user data
        question = build_eligibility_query(visa_type, user_data, visa_specific_data)
    
    if not question:
        st.warning("Please enter a question or use the default eligibility check.")
    else:
        with st.spinner("🔍 Searching visa policies and analyzing your application..."):
            # Query the RAG system with enhanced search
            result = query_visa_policy(question, user_data)
            
            # Extract decision and details
            decision = result.get("decision", "")
            answer = result.get("details", "")
            
            # Display eligibility decision prominently
            st.markdown("---")
            
            # Determine status and color
            if decision.startswith("ELIGIBLE:"):
                st.success("✅ **ELIGIBILITY STATUS: ELIGIBLE**")
                st.markdown(f"**Reason:** {decision.replace('ELIGIBLE:', '').strip()}")
            elif decision.startswith("NOT ELIGIBLE:"):
                st.error("❌ **ELIGIBILITY STATUS: NOT ELIGIBLE**")
                st.markdown(f"**Reason:** {decision.replace('NOT ELIGIBLE:', '').strip()}")
            elif decision.startswith("INSUFFICIENT INFORMATION:"):
                st.warning("⚠️ **ELIGIBILITY STATUS: INSUFFICIENT INFORMATION**")
                st.markdown(f"**Missing:** {decision.replace('INSUFFICIENT INFORMATION:', '').strip()}")
            else:
                st.info("ℹ️ **ELIGIBILITY STATUS: UNDER REVIEW**")
                st.markdown(f"{decision}")
            
            st.markdown("---")
            st.markdown("### 📋 Detailed Eligibility Assessment")
            st.markdown(answer)
            st.markdown("---")
            
            # Additional specific checks based on user input
            st.markdown("### 🔍 Specific Checks Based on Your Application")
            
            specific_checks = []
            
            if visa_type == "Student Visa":
                if visa_specific_data.get('has_cas') == 'Yes':
                    specific_checks.append("✅ You have a CAS - this is required for Student Visa")
                else:
                    specific_checks.append("⚠️ You need a CAS (Confirmation of Acceptance for Studies) from a licensed sponsor")
                
                if visa_specific_data.get('meets_financial_requirement') == 'Yes':
                    specific_checks.append("✅ Financial requirement met")
                else:
                    specific_checks.append("⚠️ You must meet financial requirements")
            
            elif visa_type == "Skilled Worker Visa":
                if visa_specific_data.get('certificate_of_sponsorship_issued') == 'Yes':
                    specific_checks.append("✅ Certificate of Sponsorship issued")
                else:
                    specific_checks.append("⚠️ You need a Certificate of Sponsorship from a licensed sponsor")
                
                if visa_specific_data.get('meets_minimum_salary_threshold') == 'Yes':
                    specific_checks.append("✅ Salary meets minimum threshold")
                else:
                    specific_checks.append("⚠️ Check if salary meets minimum threshold requirements")
            
            elif visa_type == "Graduate Visa":
                if visa_specific_data.get('course_completed') == 'Yes':
                    specific_checks.append("✅ Course completed")
                else:
                    specific_checks.append("⚠️ Course must be completed to apply")
                
                if visa_specific_data.get('currently_in_uk') == 'Yes':
                    specific_checks.append("✅ Currently in UK - you can apply from within the UK")
            
            if specific_checks:
                for check in specific_checks:
                    st.markdown(f"- {check}")
            
            st.markdown("---")
            
            # Show data summary
            with st.expander("📊 Application Data Summary"):
                st.json(user_data)
            
            # Show what was searched
            with st.expander("🔎 Search Query Used"):
                st.code(question, language="text")

# =====================================================
# Footer
# =====================================================
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This is an AI-powered tool for guidance purposes only. Always verify information with official UK government sources and consult an immigration advisor for legal advice.")
st.caption("🔗 Official UK Government Visa Information: https://www.gov.uk/browse/visas-immigration")
