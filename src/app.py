
import streamlit as st
from datetime import datetime, date
import sys
from pathlib import Path
import time
import os
import requests
from huggingface_hub import InferenceClient

# Import validation layers
from validation_layers import HardValidator, format_hard_validation_report, ValidationStatus

# Import your RAG pipeline
try:
    from rag_pipeline import EnhancedVisaRAGPipeline
except ImportError:
    st.error("⚠️ Could not import RAG pipeline. Make sure rag_pipeline.py is in the same directory.")
    st.stop()

st.set_page_config(
    page_title="SwiftVisa - UK Visa Eligibility Checker",
    layout="wide",
    page_icon=" "
)

# Custom CSS for better UX
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
    }
    .eligibility-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Hugging Face API Configuration
# Using Llama 3.1 8B (more modern and faster than Llama 2)
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"

def get_hf_api_key():
    """Get Hugging Face API key from terminal environment or Streamlit secrets"""
    # 1. Try to get from terminal environment (e.g., export HUGGINGFACE_API_KEY=...)
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    
    # 2. If not found, try Streamlit secrets
    if not api_key:
        api_key = st.secrets.get("HUGGINGFACE_API_KEY")
        
    return api_key

# Initialize the modern client globally
api_key = get_hf_api_key()
client = InferenceClient(api_key=api_key) if api_key else None

def query_llama_hf(prompt: str, max_tokens: int = 1000) -> str:
    if not client:
        st.error("⚠️ Hugging Face API key not found. Please set HUGGINGFACE_API_KEY in your terminal.")
        return None
    
    try:
        # The InferenceClient automatically handles the "410 Gone" routing
        response = client.chat_completion(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.1, # Keep it low for factual visa info
        )
        return response.choices[0].message.content
            
    except Exception as e:
        # Handle the 503 "Model is loading" error gracefully
        if "503" in str(e):
            st.warning("⌛ Llama is waking up on the HF servers. Retrying in 10 seconds...")
            time.sleep(10)
            return query_llama_hf(prompt, max_tokens)
        
        st.error(f"⚠️ API Error: {str(e)}")
        return None

# Initialize RAG Pipeline (cached to avoid reloading)
@st.cache_resource
def load_rag_pipeline():
    """Load RAG pipeline once and cache it"""
    try:
        rag = EnhancedVisaRAGPipeline(
            vectorstore_path="vectorstore/visa_db_minilm",
            top_k=3,
            temperature=0.0,
            top_p=0.9,
        )
        return rag
    except Exception as e:
        st.error(f"Failed to load RAG pipeline: {e}")
        return None

# Load pipeline
rag_pipeline = load_rag_pipeline()

# Initialize hard validator
hard_validator = HardValidator()

def format_user_profile_for_query(user_data: dict, visa_type: str) -> str:
    """Convert form data into a structured query for RAG"""
    personal = user_data["personal_info"]
    visa_specific = user_data.get("visa_specific", {})
    
    # Build a comprehensive eligibility query
    query = f"""You are a UK visa eligibility expert. Analyze the following application and provide a detailed assessment.

Visa Type: {visa_type}

Applicant Profile:
- Name: {personal['full_name']}
- Date of Birth: {personal['date_of_birth']}
- Nationality: {personal['nationality']}
- Current Location: {personal['current_location']}
- Funds Available: £{personal['funds_available']:,}
- English Requirement Met: {personal['english_requirement_met']}
- Criminal History: {personal['criminal_history']}
- Previous UK Visa Refusal: {personal['previous_uk_refusal']}
- Intended Travel Date: {personal['intended_travel_date']}
- Intended Stay Duration: {personal['intended_length_of_stay']} months
"""
    
    # Add visa-specific details
    if visa_type == "Graduate Visa" and visa_specific:
        query += f"""
Graduate Visa Details:
- Currently in UK: {visa_specific.get('currently_in_uk')}
- Current Visa Type: {visa_specific.get('current_uk_visa_type')}
- Course Completed: {visa_specific.get('course_completed')}
- Course Level: {visa_specific.get('course_level_completed')}
- Education Provider Licensed: {visa_specific.get('education_provider_licensed')}
- Completion Reported to Home Office: {visa_specific.get('provider_reported_completion')}
- CAS Reference: {visa_specific.get('original_cas_reference')}
- Student Visa Valid on Application Date: {visa_specific.get('student_visa_valid')}
"""
    elif visa_type == "Student Visa" and visa_specific:
        query += f"""
Student Visa Details:
- Has CAS: {visa_specific.get('has_cas')}
- CAS Reference: {visa_specific.get('cas_reference')}
- Education Provider Licensed: {visa_specific.get('education_provider_licensed')}
- Course Level: {visa_specific.get('course_level')}
- Course Full-Time: {visa_specific.get('course_full_time')}
- Course Start Date: {visa_specific.get('course_start_date')}
- Course Duration: {visa_specific.get('course_duration')} months
- Meets Financial Requirement: {visa_specific.get('meets_financial_requirement')}
- Funds Held for 28 Days: {visa_specific.get('funds_held_28_days')}
"""
    elif visa_type == "Skilled Worker Visa" and visa_specific:
        query += f"""
Skilled Worker Visa Details:
- Job Offer Confirmed: {visa_specific.get('job_offer_confirmed')}
- Employer Licensed Sponsor: {visa_specific.get('employer_licensed_sponsor')}
- Certificate of Sponsorship Issued: {visa_specific.get('cos_issued')}
- CoS Reference: {visa_specific.get('cos_reference')}
- Job Title: {visa_specific.get('job_title')}
- SOC Code: {visa_specific.get('soc_code')}
- Job is Eligible Occupation: {visa_specific.get('job_eligible_occupation')}
- Salary Offered: £{visa_specific.get('salary_offered', 0):,} per year
- Meets Minimum Salary Threshold: {visa_specific.get('meets_salary_threshold')}
"""
    elif visa_type == "Health & Care Visa" and visa_specific:
        query += f"""
Health & Care Visa Details:
- Job Offer Confirmed: {visa_specific.get('job_offer_confirmed')}
- Employer Licensed Healthcare Sponsor: {visa_specific.get('employer_healthcare_sponsor')}
- Certificate of Sponsorship Issued: {visa_specific.get('cos_issued')}
- CoS Reference: {visa_specific.get('cos_reference')}
- Job Title: {visa_specific.get('job_title')}
- SOC Code: {visa_specific.get('soc_code')}
- Job is Eligible Healthcare Role: {visa_specific.get('job_eligible_healthcare')}
- Salary Offered: £{visa_specific.get('salary_offered', 0):,} per year
- Meets Healthcare Salary Rules: {visa_specific.get('meets_healthcare_salary')}
"""
    elif visa_type == "Visitor Visa" and visa_specific:
        query += f"""
Visitor Visa Details:
- Purpose of Visit: {visa_specific.get('visitor_purpose')}
- Purpose Permitted: {visa_specific.get('purpose_permitted')}
- Stay Length: {visa_specific.get('stay_length_months')} months
- Accommodation Arranged: {visa_specific.get('accommodation_arranged')}
- Return Travel Planned: {visa_specific.get('return_travel_planned')}
- Intends to Leave UK: {visa_specific.get('intends_to_leave')}
- Sufficient Funds: {visa_specific.get('sufficient_funds')}
"""
    
    query += f"""
Based on the above profile and the official UK {visa_type} requirements, provide:

1. **Eligibility Assessment**: State clearly if the applicant is ELIGIBLE, NOT ELIGIBLE, or needs FURTHER REVIEW
2. **Requirements Analysis**: List which requirements are met and which are not met
3. **Required Documents**: Specify all documents needed for the application
4. **Recommendations**: Provide actionable next steps and advice

Respond in a professional, clear, and helpful manner."""
    
    return query

def check_eligibility_with_llama(user_data: dict, visa_type: str, retrieved_context: str = ""):
    """
    Use Llama via Hugging Face API for eligibility assessment
    """
    try:
        # Format the query with user data
        query = format_user_profile_for_query(user_data, visa_type)
        
        # If we have RAG context, add it
        if retrieved_context:
            full_prompt = f"""Context from UK Visa Policy Documents:
{retrieved_context}

{query}"""
        else:
            full_prompt = query
        
        # Query Llama model
        response = query_llama_hf(full_prompt)
        
        if not response:
            return {
                "status": "error",
                "message": "Failed to get response from Llama model"
            }
        
        # Parse response to determine eligibility
        answer_lower = response.lower()
        
        if "eligible" in answer_lower and "not eligible" not in answer_lower:
            eligibility = "ELIGIBLE"
        elif "not eligible" in answer_lower or "ineligible" in answer_lower:
            eligibility = "NOT ELIGIBLE"
        else:
            eligibility = "FURTHER REVIEW NEEDED"
        
        return {
            "status": "success",
            "eligibility": eligibility,
            "explanation": response,
            "model": "Llama via Hugging Face"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

# ===================== UI STARTS HERE =====================

# Centered Header
st.markdown("""
<div class="main-header">
    <h1>SwiftVisa</h1>
    <h3>UK Visa Eligibility Checker</h3>
    <p>Fast, accurate eligibility assessment powered by Llama AI</p>
</div>
""", unsafe_allow_html=True)

# Check if API key is configured
api_key = get_hf_api_key()
if not api_key:
    st.warning("""
    ⚠️ **Hugging Face API Key Not Configured**
    
    To use this application, you need to set up your Hugging Face API key:
    
    1. Get a free API key from [Hugging Face](https://huggingface.co/settings/tokens)
    2. Set it as an environment variable: `HUGGINGFACE_API_KEY=your_key_here`
    3. Or add it to `.streamlit/secrets.toml`:
       ```
       HUGGINGFACE_API_KEY = "your_key_here"
       ```
    """)

st.markdown("---")

# Visa Type Selection
visa_type = st.selectbox(
    "Select Visa Type",
    ["Graduate Visa", "Student Visa", "Skilled Worker Visa", "Health & Care Visa", "Visitor Visa"],
    help="Choose the visa type you want to apply for"
)

st.markdown("---")

# Common Entities Section
st.header("📋 Personal Information")

col1, col2 = st.columns(2)

with col1:
    full_name = st.text_input("Full Name (as per passport)*", help="Enter your name exactly as it appears on your passport")
    date_of_birth = st.date_input(
        "Date of Birth*",
        min_value=date(1924, 1, 1),
        max_value=date.today(),
        help="Your date of birth"
    )
    nationality = st.text_input("Nationality*", help="Your country of citizenship")
    passport_number = st.text_input("Passport Number*")
    passport_issue_date = st.date_input("Passport Issue Date*", max_value=date.today())
    passport_expiry_date = st.date_input(
        "Passport Expiry Date*",
        min_value=date.today(),
        help="Your passport must be valid for at least 6 months beyond your stay"
    )

with col2:
    current_location = st.text_input(
        "Country of Application / Current Location*",
        help="Where you will be applying from"
    )
    purpose_of_visit = st.text_area("Purpose of Visit*")
    intended_travel_date = st.date_input(
        "Intended Travel / Start Date*",
        min_value=date.today(),
        help="When you plan to travel to the UK"
    )
    intended_length_of_stay = st.number_input(
        "Intended Length of Stay (months)*",
        min_value=1,
        max_value=60,
        help="How long you plan to stay in the UK"
    )
    funds_available = st.number_input(
        "Funds Available (£)*",
        min_value=0,
        step=100,
        help="Total funds you have available to support yourself"
    )

st.markdown("---")

col3, col4 = st.columns(2)

with col3:
    english_requirement_met = st.radio(
        "English Language Requirement Met?*",
        ["Yes", "No"],
        help="Have you passed an approved English test or have a degree taught in English?"
    )
    criminal_history = st.radio(
        "Criminal History Declaration*",
        ["Yes", "No"],
        help="Do you have any criminal convictions?"
    )

with col4:
    previous_uk_refusal = st.radio(
        "Previous UK Visa Refusal?*",
        ["Yes", "No"],
        help="Have you ever been refused a UK visa?"
    )

st.markdown("---")

st.subheader("📞 Contact Information")

col5, col6, col7 = st.columns(3)

with col5:
    email = st.text_input("Email Address*")

with col6:
    phone = st.text_input("Phone Number*")

with col7:
    current_address = st.text_area("Current Address*")

st.markdown("---")

# Visa-Specific Sections
visa_specific_data = {}

if visa_type == "Graduate Visa":
    st.header("🎓 Graduate Visa Specific Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        currently_in_uk = st.radio("Currently in UK?*", ["Yes", "No"], key="grad_uk")
        current_uk_visa_type = st.selectbox("Current UK Visa Type*", ["Student", "Tier 4", "Other"])
        course_completed = st.radio("Course Completed?*", ["Yes", "No"])
        course_level_completed = st.selectbox(
            "Course Level Completed*",
            ["Bachelor's Degree", "Master's Degree", "PhD/Doctorate", "Other"]
        )
    
    with col2:
        education_provider_licensed = st.radio("Education Provider is Licensed?*", ["Yes", "No"], key="grad_licensed")
        provider_reported_completion = st.radio("Provider Reported Completion to Home Office?*", ["Yes", "No"])
        original_cas_reference = st.text_input("Original CAS Reference Number*")
        student_visa_valid = st.radio("Student Visa Valid on Application Date?*", ["Yes", "No"])
    
    visa_specific_data = {
        "currently_in_uk": currently_in_uk,
        "current_uk_visa_type": current_uk_visa_type,
        "course_completed": course_completed,
        "course_level_completed": course_level_completed,
        "education_provider_licensed": education_provider_licensed,
        "provider_reported_completion": provider_reported_completion,
        "original_cas_reference": original_cas_reference,
        "student_visa_valid": student_visa_valid
    }

elif visa_type == "Student Visa":
    st.header("🎓 Student Visa Specific Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        has_cas = st.radio("Do you have a CAS (Confirmation of Acceptance for Studies)?*", ["Yes", "No"])
        cas_reference = st.text_input("CAS Reference Number*")
        education_provider_licensed = st.radio("Education Provider is Licensed?*", ["Yes", "No"], key="student_licensed")
        course_level = st.selectbox(
            "Course Level*",
            ["Below Degree Level", "Bachelor's Degree", "Master's Degree", "PhD/Doctorate"]
        )
        course_full_time = st.radio("Is the Course Full-Time?*", ["Yes", "No"])
    
    with col2:
        course_start_date = st.date_input("Course Start Date*", min_value=date.today())
        course_end_date = st.date_input("Course End Date*", min_value=date.today())
        course_duration = st.number_input("Course Duration (months)*", min_value=1, max_value=60)
        meets_financial_requirement = st.radio("Meets Financial Requirement?*", ["Yes", "No"])
        funds_held_28_days = st.radio("Funds Held for 28 Days?*", ["Yes", "No"])
    
    visa_specific_data = {
        "has_cas": has_cas,
        "cas_reference": cas_reference,
        "education_provider_licensed": education_provider_licensed,
        "course_level": course_level,
        "course_full_time": course_full_time,
        "course_start_date": str(course_start_date),
        "course_end_date": str(course_end_date),
        "course_duration": course_duration,
        "meets_financial_requirement": meets_financial_requirement,
        "funds_held_28_days": funds_held_28_days
    }

elif visa_type == "Skilled Worker Visa":
    st.header("💼 Skilled Worker Visa Specific Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_offer_confirmed = st.radio("Job Offer Confirmed?*", ["Yes", "No"], key="skilled_job")
        employer_licensed_sponsor = st.radio("Employer is Licensed Sponsor?*", ["Yes", "No"])
        cos_issued = st.radio("Certificate of Sponsorship Issued?*", ["Yes", "No"], key="skilled_cos")
        cos_reference = st.text_input("CoS Reference Number*", key="skilled_cos_ref")
        job_title = st.text_input("Job Title*")
        soc_code = st.text_input("SOC Code*", help="Standard Occupational Classification code")
    
    with col2:
        job_eligible_occupation = st.radio("Job is Eligible Occupation?*", ["Yes", "No"])
        salary_offered = st.number_input("Salary Offered (£ per year)*", min_value=0, step=1000)
        meets_salary_threshold = st.radio("Meets Minimum Salary Threshold?*", ["Yes", "No"])
        criminal_record_required = st.radio("Criminal Record Certificate Required?*", ["Yes", "No"], key="skilled_criminal_req")
        criminal_record_provided = "N/A"
        if criminal_record_required == "Yes":
            criminal_record_provided = st.radio("Criminal Record Certificate Provided?*", ["Yes", "No"])
    
    visa_specific_data = {
        "job_offer_confirmed": job_offer_confirmed,
        "employer_licensed_sponsor": employer_licensed_sponsor,
        "cos_issued": cos_issued,
        "cos_reference": cos_reference,
        "job_title": job_title,
        "soc_code": soc_code,
        "job_eligible_occupation": job_eligible_occupation,
        "salary_offered": salary_offered,
        "meets_salary_threshold": meets_salary_threshold,
        "criminal_record_required": criminal_record_required,
        "criminal_record_provided": criminal_record_provided
    }

elif visa_type == "Health & Care Visa":
    st.header("🏥 Health & Care Visa Specific Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        job_offer_confirmed = st.radio("Job Offer Confirmed?*", ["Yes", "No"], key="health_job")
        employer_healthcare_sponsor = st.radio("Employer is Licensed Healthcare Sponsor?*", ["Yes", "No"])
        cos_issued = st.radio("Certificate of Sponsorship Issued?*", ["Yes", "No"], key="health_cos")
        cos_reference = st.text_input("CoS Reference Number*", key="health_cos_ref")
        job_title = st.text_input("Job Title*", key="health_job_title")
        soc_code = st.text_input("SOC Code*", key="health_soc")
    
    with col2:
        job_eligible_healthcare = st.radio("Job is Eligible Healthcare Role?*", ["Yes", "No"])
        salary_offered = st.number_input("Salary Offered (£ per year)*", min_value=0, step=1000, key="health_salary")
        meets_healthcare_salary = st.radio("Meets Healthcare Salary Rules?*", ["Yes", "No"])
        professional_reg_required = st.radio("Professional Registration Required?*", ["Yes", "No"])
        professional_reg_provided = "N/A"
        if professional_reg_required == "Yes":
            professional_reg_provided = st.radio("Professional Registration Provided?*", ["Yes", "No"])
    
    visa_specific_data = {
        "job_offer_confirmed": job_offer_confirmed,
        "employer_healthcare_sponsor": employer_healthcare_sponsor,
        "cos_issued": cos_issued,
        "cos_reference": cos_reference,
        "job_title": job_title,
        "soc_code": soc_code,
        "job_eligible_healthcare": job_eligible_healthcare,
        "salary_offered": salary_offered,
        "meets_healthcare_salary": meets_healthcare_salary,
        "professional_reg_required": professional_reg_required,
        "professional_reg_provided": professional_reg_provided
    }

elif visa_type == "Visitor Visa":
    st.header("✈️ Visitor Visa Specific Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        visitor_purpose = st.text_area("Detailed Purpose of Visit*")
        purpose_permitted = st.radio("Purpose is Permitted Under Visitor Rules?*", ["Yes", "No"])
        stay_length_months = st.number_input("Intended Length of Stay (months)*", min_value=1, max_value=6)
        stay_within_limit = st.radio("Stay Within 6 Months Limit?*", ["Yes", "No"])
    
    with col2:
        accommodation_arranged = st.radio("Accommodation Arranged?*", ["Yes", "No"])
        return_travel_planned = st.radio("Return or Onward Travel Planned?*", ["Yes", "No"])
        intends_to_leave = st.radio("Intends to Leave UK After Visit?*", ["Yes", "No"])
        sufficient_funds = st.radio("Sufficient Funds for Stay?*", ["Yes", "No"])
    
    visa_specific_data = {
        "visitor_purpose": visitor_purpose,
        "purpose_permitted": purpose_permitted,
        "stay_length_months": stay_length_months,
        "stay_within_limit": stay_within_limit,
        "accommodation_arranged": accommodation_arranged,
        "return_travel_planned": return_travel_planned,
        "intends_to_leave": intends_to_leave,
        "sufficient_funds": sufficient_funds
    }

st.markdown("---")

# Submit Button
col1, col2, col3 = st.columns([1, 1, 1])

with col2:
    submit_button = st.button("🔍 Check Eligibility", type="primary", use_container_width=True)

if submit_button:
    # Basic field validation
    missing_fields = []
    if not full_name:
        missing_fields.append("Full Name")
    if not nationality:
        missing_fields.append("Nationality")
    if not passport_number:
        missing_fields.append("Passport Number")
    if not current_location:
        missing_fields.append("Current Location")
    if not purpose_of_visit:
        missing_fields.append("Purpose of Visit")
    if not email:
        missing_fields.append("Email Address")
    if not phone:
        missing_fields.append("Phone Number")
    if not current_address:
        missing_fields.append("Current Address")
    
    # Show error if fields are missing
    if missing_fields:
        st.error("⚠️ **Please fill in the following required fields:**")
        for field in missing_fields:
            st.write(f"- {field}")
    elif not api_key:
        st.error("⚠️ **Hugging Face API key is required to proceed. Please configure it first.**")
    else:
        # Collect all data
        user_data = {
            "visa_type": visa_type,
            "personal_info": {
                "full_name": full_name,
                "date_of_birth": str(date_of_birth),
                "nationality": nationality,
                "passport_number": passport_number,
                "passport_issue_date": str(passport_issue_date),
                "passport_expiry_date": str(passport_expiry_date),
                "current_location": current_location,
                "purpose_of_visit": purpose_of_visit,
                "intended_travel_date": str(intended_travel_date),
                "intended_length_of_stay": intended_length_of_stay,
                "funds_available": funds_available,
                "english_requirement_met": english_requirement_met,
                "criminal_history": criminal_history,
                "previous_uk_refusal": previous_uk_refusal,
                "email": email,
                "phone": phone,
                "current_address": current_address
            },
            "visa_specific": visa_specific_data
        }
        
        # ========== LAYER 1: HARD VALIDATION ==========
        st.markdown("---")
        st.subheader("⚡ Running Fast Validation Checks...")
        
        with st.spinner('Validating your application data...'):
            validation_report = hard_validator.validate(user_data, visa_type)
        
        # Display hard validation results
        st.markdown(format_hard_validation_report(validation_report))
        
        # ========== LAYER 2: LLAMA + RAG (Only if Layer 1 passes) ==========
        if validation_report.passed:
            st.markdown("---")
            st.subheader("🤖 Analyzing with Llama AI...")
            
            with st.spinner('Performing detailed policy analysis with Llama...'):
                start_time = time.time()
                
                # Optionally retrieve context from RAG pipeline if available
                retrieved_context = ""
                if rag_pipeline:
                    try:
                        query = format_user_profile_for_query(user_data, visa_type)
                        rag_result = rag_pipeline.query(query, show_sources=False)
                        retrieved_context = rag_result.get("answer", "")
                    except:
                        pass
                
                # Use Llama for analysis
                result = check_eligibility_with_llama(user_data, visa_type, retrieved_context)
                analysis_time = (time.time() - start_time) * 1000
            
            # Display results
            if result["status"] == "success":
                st.markdown("---")
                st.header("📊 Complete Eligibility Assessment")
                
                st.caption(f"✨ Powered by {result.get('model', 'Llama AI')} | Analysis time: {analysis_time:.0f}ms")
                
                # Show eligibility status with appropriate styling
                if result["eligibility"] == "ELIGIBLE":
                    st.success(f"### ✅ You appear to be **ELIGIBLE** for {visa_type}")
                    st.info("**Next Steps:** Review the detailed guidance below and prepare your application documents.")
                elif result["eligibility"] == "NOT ELIGIBLE":
                    st.error(f"### ❌ Based on the information provided, you may **NOT be eligible** for {visa_type}")
                    st.warning("**Important:** Please review the reasons below carefully. You may want to consult with an immigration advisor.")
                else:
                    st.warning(f"### ⚠️ Your application requires **FURTHER REVIEW**")
                    st.info("Some aspects of your application need additional assessment. Please review the guidance below.")
                
                # Show detailed explanation
                st.markdown("### 📝 Detailed Assessment")
                st.markdown(result["explanation"])
                
                st.markdown("---")
                # Additional recommendations
                st.markdown("### 💡 Important Reminders")
                st.info("""
                - **This is an automated assessment** based on the information you provided
                - Always verify current requirements on the official GOV.UK website
                - Application requirements can change - check before submitting
                - Consider consulting with a licensed immigration advisor for complex cases
                - Ensure all documents are prepared before starting your application
                """)
            
            else:
                st.error(f"❌ Error during analysis: {result.get('message', 'Unknown error occurred')}")
        
        else:
            # Hard validation failed - show helpful guidance
            st.markdown("---")
            st.info("""
            ### 📖 What to do next?
            
            1. **Review the issues** listed above carefully
            2. **Gather the correct information** or documents
            3. **Update your responses** and try again
            4. **Need help?** Consider consulting with an immigration advisor
            
            *Remember: This checker helps you understand requirements before applying. 
            Always verify current rules on the official GOV.UK website.*
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p><strong>SwiftVisa</strong> - Powered by AI and Official UK Government Guidance</p>
    <p>This tool provides guidance only. Always verify requirements on <a href='https://www.gov.uk/browse/visas-immigration' target='_blank'>GOV.UK</a></p>
    <p>For official applications, visit <a href='https://www.gov.uk/apply-uk-visa' target='_blank'>www.gov.uk/apply-uk-visa</a></p>
</div>
""", unsafe_allow_html=True)