"""
AI SwiftVisa - UK Visa Eligibility Screening Application
A beautiful, modern UI for checking UK visa eligibility
REAL-TIME SCORING: Uses FAISS database for policy-based scoring
"""

import streamlit as st
from retriever import VisaRetriever

# Page configuration
st.set_page_config(
    page_title="Visa Eligibility Screening",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0c1929 0%, #1a3a5c 50%, #0c1929 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Header Styles */
    .header-container {
        display: flex;
        align-items: center;
        padding: 20px 40px;
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        margin-bottom: 30px;
    }
    
    .visa-logo {
        background: linear-gradient(135deg, #1a4b8c, #2d6cb5);
        color: white;
        padding: 15px 25px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: 700;
        margin-right: 20px;
    }
    
    .header-text h1 {
        color: white;
        font-size: 28px;
        margin: 0;
        font-weight: 600;
    }
    
    .header-text p {
        color: #8ba3c4;
        font-size: 14px;
        margin: 5px 0 0 0;
    }
    
    /* Hero Section */
    .hero-section {
        text-align: left;
        padding: 40px 0;
        position: relative;
    }
    
    .hero-title {
        font-size: 48px;
        font-weight: 700;
        color: white;
        line-height: 1.2;
        margin-bottom: 20px;
    }
    
    .hero-title span {
        color: #64b5f6;
    }
    
    .hero-subtitle {
        font-size: 18px;
        color: #b0c4de;
        max-width: 500px;
        line-height: 1.6;
    }
    
    /* Visa Cards Container */
    .visa-cards-title {
        font-size: 32px;
        font-weight: 600;
        color: white;
        margin: 40px 0 10px 0;
    }
    
    .visa-cards-subtitle {
        color: #8ba3c4;
        font-size: 16px;
        margin-bottom: 30px;
    }
    
    /* Card Styles */
    .visa-card {
        background: linear-gradient(145deg, #1e3a5f, #152a45);
        border-radius: 16px;
        padding: 30px 25px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
        height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    .visa-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        border-color: #64b5f6;
    }
    
    .card-icon {
        font-size: 40px;
        margin-bottom: 15px;
    }
    
    .card-title {
        color: white;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    
    .card-subtitle {
        color: #8ba3c4;
        font-size: 14px;
    }
    
    /* Form Styles */
    .form-container {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 40px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .form-title {
        color: white;
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .form-subtitle {
        color: #8ba3c4;
        font-size: 16px;
        margin-bottom: 30px;
    }
    
    /* Input Styles */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        color: white !important;
        padding: 12px 15px !important;
    }
    
    /* Fix for Selectbox - prevent cut off */
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        border-radius: 10px !important;
        min-height: 45px !important;
    }
    
    .stSelectbox > div > div > div {
        color: white !important;
        padding: 8px 12px !important;
        overflow: visible !important;
    }
    
    /* Dropdown menu styling */
    .stSelectbox [data-baseweb="select"] {
        min-height: 45px !important;
    }
    
    .stSelectbox [data-baseweb="popover"] {
        background: #1e3a5f !important;
    }
    
    .stSelectbox ul {
        background: #1e3a5f !important;
    }
    
    .stSelectbox li {
        color: white !important;
        padding: 10px 15px !important;
    }
    
    .stSelectbox li:hover {
        background: #2d5a8c !important;
    }
    
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label,
    .stDateInput > label,
    .stRadio > label {
        color: #b0c4de !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #2196f3, #1976d2) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 40px !important;
        font-size: 18px !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 30px rgba(33, 150, 243, 0.4) !important;
    }
    
    /* Result Styles */
    .result-container {
        background: rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 40px;
        margin-top: 30px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .eligible-badge {
        background: linear-gradient(135deg, #4caf50, #2e7d32);
        color: white;
        padding: 20px 40px;
        border-radius: 15px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(76, 175, 80, 0.3);
    }
    
    .not-eligible-badge {
        background: linear-gradient(135deg, #f44336, #c62828);
        color: white;
        padding: 20px 40px;
        border-radius: 15px;
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(244, 67, 54, 0.3);
    }
    
    .result-details {
        color: #e0e0e0;
        font-size: 16px;
        line-height: 1.8;
    }
    
    /* Section Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        margin: 40px 0;
    }
    
    /* World Map Background */
    .world-bg {
        position: relative;
        background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 500"><ellipse cx="500" cy="250" rx="450" ry="220" fill="none" stroke="%231a3a5c" stroke-width="2"/></svg>');
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
    }
    
    /* Animated Elements */
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    
    .floating {
        animation: float 3s ease-in-out infinite;
    }
    
    /* Section Headers */
    .section-header {
        color: #64b5f6;
        font-size: 20px;
        font-weight: 600;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(100, 181, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'selected_visa' not in st.session_state:
    st.session_state.selected_visa = None
if 'result' not in st.session_state:
    st.session_state.result = None
if 'eligibility_status' not in st.session_state:
    st.session_state.eligibility_status = None
if 'score' not in st.session_state:
    st.session_state.score = None
if 'score_details' not in st.session_state:
    st.session_state.score_details = None
if 'form_data' not in st.session_state:
    st.session_state.form_data = {}
if 'policy_context' not in st.session_state:
    st.session_state.policy_context = None

# Load retriever for FAISS database
@st.cache_resource
def load_retriever():
    try:
        return VisaRetriever()
    except Exception as e:
        st.warning(f"FAISS database not available: {e}")
        return None

# Get policy information from FAISS
def get_policy_info(visa_type, retriever):
    """Retrieve relevant policy information from FAISS database"""
    if retriever is None:
        return None
    try:
        query = f"UK {visa_type} eligibility requirements documents needed"
        results = retriever.retrieve(query, top_k=3)
        return results
    except Exception as e:
        return None

# Check eligibility based on form data (rule-based) with SCORING
def check_eligibility_rules(visa_type, form_data):
    """Check eligibility based on the Yes/No answers in the form with scoring"""
    issues = []
    passed = []
    scores = {}  # Track individual requirement scores
    
    # Define weights for different requirement types
    CRITICAL_WEIGHT = 15  # Must-have requirements
    IMPORTANT_WEIGHT = 10  # Important but not critical
    MINOR_WEIGHT = 5  # Nice to have
    
    total_possible = 0
    total_earned = 0
    
    # Common checks (apply to all visa types)
    # English Language Requirement
    total_possible += CRITICAL_WEIGHT
    if form_data.get("english_language_requirement_met") == "No":
        issues.append("❌ English language requirement NOT met")
        scores["English Language"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
    elif form_data.get("english_language_requirement_met") == "Yes":
        passed.append("✅ English language requirement met")
        total_earned += CRITICAL_WEIGHT
        scores["English Language"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
    else:
        scores["English Language"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
    
    # Criminal History
    total_possible += IMPORTANT_WEIGHT
    if form_data.get("criminal_history_declaration") == "Yes":
        issues.append("⚠️ Criminal history declared - may require additional review")
        total_earned += IMPORTANT_WEIGHT // 2  # Partial score - not automatic rejection
        scores["Criminal History"] = {"earned": IMPORTANT_WEIGHT // 2, "max": IMPORTANT_WEIGHT, "status": "warning"}
    elif form_data.get("criminal_history_declaration") == "No":
        passed.append("✅ No criminal history")
        total_earned += IMPORTANT_WEIGHT
        scores["Criminal History"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
    else:
        scores["Criminal History"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
    
    # Previous UK Visa Refusal
    total_possible += IMPORTANT_WEIGHT
    if form_data.get("previous_uk_visa_refusal") == "Yes":
        issues.append("⚠️ Previous UK visa refusal - may affect application")
        total_earned += IMPORTANT_WEIGHT // 2
        scores["Previous Visa Refusal"] = {"earned": IMPORTANT_WEIGHT // 2, "max": IMPORTANT_WEIGHT, "status": "warning"}
    elif form_data.get("previous_uk_visa_refusal") == "No":
        passed.append("✅ No previous UK visa refusal")
        total_earned += IMPORTANT_WEIGHT
        scores["Previous Visa Refusal"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
    else:
        scores["Previous Visa Refusal"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
    
    # Student Visa specific checks
    if visa_type == "Student Visa":
        # CAS - Critical
        total_possible += CRITICAL_WEIGHT
        if form_data.get("has_cas") == "No":
            issues.append("❌ No CAS (Confirmation of Acceptance for Studies) - REQUIRED")
            scores["CAS Document"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("has_cas") == "Yes":
            passed.append("✅ Has valid CAS")
            total_earned += CRITICAL_WEIGHT
            scores["CAS Document"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["CAS Document"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        # Licensed Provider - Critical
        total_possible += CRITICAL_WEIGHT
        if form_data.get("education_provider_is_licensed") == "No":
            issues.append("❌ Education provider is NOT a licensed sponsor - REQUIRED")
            scores["Licensed Provider"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("education_provider_is_licensed") == "Yes":
            passed.append("✅ Education provider is licensed")
            total_earned += CRITICAL_WEIGHT
            scores["Licensed Provider"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Licensed Provider"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        # Full-time Course - Important
        total_possible += IMPORTANT_WEIGHT
        if form_data.get("course_full_time") == "No":
            issues.append("⚠️ Course is not full-time - may affect eligibility")
            total_earned += IMPORTANT_WEIGHT // 2
            scores["Full-time Course"] = {"earned": IMPORTANT_WEIGHT // 2, "max": IMPORTANT_WEIGHT, "status": "warning"}
        elif form_data.get("course_full_time") == "Yes":
            passed.append("✅ Course is full-time")
            total_earned += IMPORTANT_WEIGHT
            scores["Full-time Course"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
        else:
            scores["Full-time Course"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
        
        # Financial Requirement - Critical
        total_possible += CRITICAL_WEIGHT
        if form_data.get("meets_financial_requirement") == "No":
            issues.append("❌ Financial requirements NOT met - REQUIRED")
            scores["Financial Requirement"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("meets_financial_requirement") == "Yes":
            passed.append("✅ Financial requirements met")
            total_earned += CRITICAL_WEIGHT
            scores["Financial Requirement"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Financial Requirement"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        # 28 Days Funds - Critical
        total_possible += CRITICAL_WEIGHT
        if form_data.get("funds_held_for_28_days") == "No":
            issues.append("❌ Funds NOT held for 28 days - REQUIRED")
            scores["28 Days Funds"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("funds_held_for_28_days") == "Yes":
            passed.append("✅ Funds held for 28 days")
            total_earned += CRITICAL_WEIGHT
            scores["28 Days Funds"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["28 Days Funds"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        # Course English Requirement - Important
        total_possible += IMPORTANT_WEIGHT
        if form_data.get("english_requirement_met") == "No":
            issues.append("❌ English requirement NOT met for course")
            scores["Course English"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "fail"}
        elif form_data.get("english_requirement_met") == "Yes":
            passed.append("✅ English requirement met for course")
            total_earned += IMPORTANT_WEIGHT
            scores["Course English"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
        else:
            scores["Course English"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
    
    # Graduate Visa specific checks
    elif visa_type == "Graduate Visa":
        total_possible += CRITICAL_WEIGHT
        if form_data.get("currently_in_uk") == "No":
            issues.append("❌ Not currently in UK - must be in UK to apply")
            scores["Currently in UK"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("currently_in_uk") == "Yes":
            passed.append("✅ Currently in UK")
            total_earned += CRITICAL_WEIGHT
            scores["Currently in UK"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Currently in UK"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("course_completed") == "No":
            issues.append("❌ Course not completed - must complete course first")
            scores["Course Completed"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("course_completed") == "Yes":
            passed.append("✅ Course completed")
            total_earned += CRITICAL_WEIGHT
            scores["Course Completed"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Course Completed"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("education_provider_is_licensed") == "No":
            issues.append("❌ Education provider was NOT licensed - REQUIRED")
            scores["Licensed Provider"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("education_provider_is_licensed") == "Yes":
            passed.append("✅ Education provider was licensed")
            total_earned += CRITICAL_WEIGHT
            scores["Licensed Provider"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Licensed Provider"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("provider_reported_completion_to_home_office") == "No":
            issues.append("❌ Provider has NOT reported completion to Home Office")
            scores["Completion Reported"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("provider_reported_completion_to_home_office") == "Yes":
            passed.append("✅ Provider reported completion to Home Office")
            total_earned += CRITICAL_WEIGHT
            scores["Completion Reported"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Completion Reported"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("student_visa_valid_on_application_date") == "No":
            issues.append("❌ Student visa will NOT be valid on application date")
            scores["Visa Valid"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("student_visa_valid_on_application_date") == "Yes":
            passed.append("✅ Student visa valid on application date")
            total_earned += CRITICAL_WEIGHT
            scores["Visa Valid"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Visa Valid"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
    
    # Skilled Worker Visa specific checks
    elif visa_type == "Skilled Worker Visa":
        total_possible += CRITICAL_WEIGHT
        if form_data.get("job_offer_confirmed") == "No":
            issues.append("❌ No confirmed job offer - REQUIRED")
            scores["Job Offer"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("job_offer_confirmed") == "Yes":
            passed.append("✅ Job offer confirmed")
            total_earned += CRITICAL_WEIGHT
            scores["Job Offer"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Job Offer"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("employer_is_licensed_sponsor") == "No":
            issues.append("❌ Employer is NOT a licensed sponsor - REQUIRED")
            scores["Licensed Sponsor"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("employer_is_licensed_sponsor") == "Yes":
            passed.append("✅ Employer is licensed sponsor")
            total_earned += CRITICAL_WEIGHT
            scores["Licensed Sponsor"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Licensed Sponsor"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("certificate_of_sponsorship_issued") == "No":
            issues.append("❌ Certificate of Sponsorship NOT issued - REQUIRED")
            scores["CoS Issued"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("certificate_of_sponsorship_issued") == "Yes":
            passed.append("✅ Certificate of Sponsorship issued")
            total_earned += CRITICAL_WEIGHT
            scores["CoS Issued"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["CoS Issued"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += IMPORTANT_WEIGHT
        if form_data.get("job_is_eligible_occupation") == "No":
            issues.append("❌ Job is NOT an eligible occupation")
            scores["Eligible Occupation"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "fail"}
        elif form_data.get("job_is_eligible_occupation") == "Yes":
            passed.append("✅ Job is eligible occupation")
            total_earned += IMPORTANT_WEIGHT
            scores["Eligible Occupation"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
        else:
            scores["Eligible Occupation"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("meets_minimum_salary_threshold") == "No":
            issues.append("❌ Does NOT meet minimum salary threshold - REQUIRED")
            scores["Salary Threshold"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("meets_minimum_salary_threshold") == "Yes":
            passed.append("✅ Meets minimum salary threshold")
            total_earned += CRITICAL_WEIGHT
            scores["Salary Threshold"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Salary Threshold"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += IMPORTANT_WEIGHT
        if form_data.get("english_requirement_met") == "No":
            issues.append("❌ English requirement NOT met")
            scores["English Requirement"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "fail"}
        elif form_data.get("english_requirement_met") == "Yes":
            passed.append("✅ English requirement met")
            total_earned += IMPORTANT_WEIGHT
            scores["English Requirement"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
        else:
            scores["English Requirement"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
    
    # Health and Care Worker Visa specific checks
    elif visa_type == "Health and Care Worker Visa":
        total_possible += CRITICAL_WEIGHT
        if form_data.get("job_offer_confirmed") == "No":
            issues.append("❌ No confirmed job offer - REQUIRED")
            scores["Job Offer"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("job_offer_confirmed") == "Yes":
            passed.append("✅ Job offer confirmed")
            total_earned += CRITICAL_WEIGHT
            scores["Job Offer"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Job Offer"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("employer_is_licensed_healthcare_sponsor") == "No":
            issues.append("❌ Employer is NOT a licensed healthcare sponsor - REQUIRED")
            scores["Healthcare Sponsor"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("employer_is_licensed_healthcare_sponsor") == "Yes":
            passed.append("✅ Employer is licensed healthcare sponsor")
            total_earned += CRITICAL_WEIGHT
            scores["Healthcare Sponsor"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Healthcare Sponsor"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("certificate_of_sponsorship_issued") == "No":
            issues.append("❌ Certificate of Sponsorship NOT issued - REQUIRED")
            scores["CoS Issued"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("certificate_of_sponsorship_issued") == "Yes":
            passed.append("✅ Certificate of Sponsorship issued")
            total_earned += CRITICAL_WEIGHT
            scores["CoS Issued"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["CoS Issued"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += IMPORTANT_WEIGHT
        if form_data.get("job_is_eligible_healthcare_role") == "No":
            issues.append("❌ Job is NOT an eligible healthcare role")
            scores["Healthcare Role"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "fail"}
        elif form_data.get("job_is_eligible_healthcare_role") == "Yes":
            passed.append("✅ Job is eligible healthcare role")
            total_earned += IMPORTANT_WEIGHT
            scores["Healthcare Role"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
        else:
            scores["Healthcare Role"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("meets_healthcare_salary_rules") == "No":
            issues.append("❌ Does NOT meet healthcare salary rules")
            scores["Salary Rules"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("meets_healthcare_salary_rules") == "Yes":
            passed.append("✅ Meets healthcare salary rules")
            total_earned += CRITICAL_WEIGHT
            scores["Salary Rules"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Salary Rules"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += IMPORTANT_WEIGHT
        if form_data.get("professional_registration_required") == "Yes" and form_data.get("professional_registration_provided") == "No":
            issues.append("❌ Professional registration required but NOT provided")
            scores["Professional Registration"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "fail"}
        elif form_data.get("professional_registration_provided") == "Yes":
            passed.append("✅ Professional registration provided")
            total_earned += IMPORTANT_WEIGHT
            scores["Professional Registration"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
        elif form_data.get("professional_registration_required") == "No":
            passed.append("✅ Professional registration not required")
            total_earned += IMPORTANT_WEIGHT
            scores["Professional Registration"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
        else:
            scores["Professional Registration"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
        
        total_possible += IMPORTANT_WEIGHT
        if form_data.get("english_requirement_met") == "No":
            issues.append("❌ English requirement NOT met")
            scores["English Requirement"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "fail"}
        elif form_data.get("english_requirement_met") == "Yes":
            passed.append("✅ English requirement met")
            total_earned += IMPORTANT_WEIGHT
            scores["English Requirement"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
        else:
            scores["English Requirement"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
    
    # Visitor Visa specific checks
    elif visa_type == "Visitor Visa":
        total_possible += CRITICAL_WEIGHT
        if form_data.get("purpose_is_permitted_under_visitor_rules") == "No":
            issues.append("❌ Purpose is NOT permitted under visitor rules")
            scores["Permitted Purpose"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("purpose_is_permitted_under_visitor_rules") == "Yes":
            passed.append("✅ Purpose permitted under visitor rules")
            total_earned += CRITICAL_WEIGHT
            scores["Permitted Purpose"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Permitted Purpose"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("stay_within_6_months_limit") == "No":
            issues.append("❌ Stay exceeds 6 months limit - NOT allowed")
            scores["Stay Duration"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("stay_within_6_months_limit") == "Yes":
            passed.append("✅ Stay within 6 months limit")
            total_earned += CRITICAL_WEIGHT
            scores["Stay Duration"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Stay Duration"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += MINOR_WEIGHT
        if form_data.get("accommodation_arranged") == "No":
            issues.append("⚠️ Accommodation not arranged")
            scores["Accommodation"] = {"earned": 0, "max": MINOR_WEIGHT, "status": "warning"}
        elif form_data.get("accommodation_arranged") == "Yes":
            passed.append("✅ Accommodation arranged")
            total_earned += MINOR_WEIGHT
            scores["Accommodation"] = {"earned": MINOR_WEIGHT, "max": MINOR_WEIGHT, "status": "pass"}
        else:
            scores["Accommodation"] = {"earned": 0, "max": MINOR_WEIGHT, "status": "pending"}
        
        total_possible += IMPORTANT_WEIGHT
        if form_data.get("return_or_onward_travel_planned") == "No":
            issues.append("❌ No return/onward travel planned - may affect application")
            scores["Return Travel"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "fail"}
        elif form_data.get("return_or_onward_travel_planned") == "Yes":
            passed.append("✅ Return/onward travel planned")
            total_earned += IMPORTANT_WEIGHT
            scores["Return Travel"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
        else:
            scores["Return Travel"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
        
        total_possible += CRITICAL_WEIGHT
        if form_data.get("intends_to_leave_uk_after_visit") == "No":
            issues.append("❌ Does NOT intend to leave UK - will be REJECTED")
            scores["Intent to Leave"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "fail"}
        elif form_data.get("intends_to_leave_uk_after_visit") == "Yes":
            passed.append("✅ Intends to leave UK after visit")
            total_earned += CRITICAL_WEIGHT
            scores["Intent to Leave"] = {"earned": CRITICAL_WEIGHT, "max": CRITICAL_WEIGHT, "status": "pass"}
        else:
            scores["Intent to Leave"] = {"earned": 0, "max": CRITICAL_WEIGHT, "status": "pending"}
        
        total_possible += IMPORTANT_WEIGHT
        if form_data.get("sufficient_funds_for_stay") == "No":
            issues.append("❌ Insufficient funds for stay")
            scores["Sufficient Funds"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "fail"}
        elif form_data.get("sufficient_funds_for_stay") == "Yes":
            passed.append("✅ Sufficient funds for stay")
            total_earned += IMPORTANT_WEIGHT
            scores["Sufficient Funds"] = {"earned": IMPORTANT_WEIGHT, "max": IMPORTANT_WEIGHT, "status": "pass"}
        else:
            scores["Sufficient Funds"] = {"earned": 0, "max": IMPORTANT_WEIGHT, "status": "pending"}
    
    # Calculate percentage score
    if total_possible > 0:
        percentage_score = round((total_earned / total_possible) * 100, 1)
    else:
        percentage_score = 0
    
    # Determine overall eligibility based on score and critical issues
    critical_issues = [i for i in issues if "REQUIRED" in i or "REJECTED" in i or "must" in i.lower()]
    
    if critical_issues:
        status = "NOT ELIGIBLE"
    elif percentage_score >= 80:
        status = "ELIGIBLE"
    elif percentage_score >= 50:
        status = "POTENTIALLY ELIGIBLE"
    else:
        status = "NOT ELIGIBLE"
    
    return status, passed, issues, percentage_score, total_earned, total_possible, scores

# Build query from form data
def build_query(visa_type, form_data):
    base_query = f"Check eligibility for UK {visa_type}. "
    details = []
    
    for key, value in form_data.items():
        if value and value != "" and value != "Select":
            details.append(f"{key}: {value}")
    
    return base_query + "Applicant details: " + ", ".join(details) + ". What are the requirements and is this person eligible?"

# Navigation functions
def go_to_form(visa_type):
    st.session_state.selected_visa = visa_type
    st.session_state.page = 'form'
    st.session_state.result = None
    st.session_state.eligibility_status = None

def go_home():
    st.session_state.page = 'home'
    st.session_state.selected_visa = None
    st.session_state.result = None
    st.session_state.eligibility_status = None
    st.session_state.score = None
    st.session_state.score_details = None

# =====================================================
# HOME PAGE
# =====================================================
def render_home():
    # Header
    st.markdown("""
    <div class="header-container">
        <div class="visa-logo">VISA</div>
        <div class="header-text">
            <h1>Visa Eligibility Screening</h1>
            <p>AI-based UK Visa Eligibility Assistant</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero Section
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.markdown("""
        <div class="hero-section">
            <h1 class="hero-title">Your Global Journey<br>Starts with a<br><span>Simple Step.</span></h1>
            <p class="hero-subtitle">Our platform helps applicants quickly understand their eligibility for different UK visa categories before applying.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;" class="floating">
            <div style="font-size: 150px; opacity: 0.9;">🌍</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Image row
        st.markdown("""
        <div style="display: flex; justify-content: center; gap: 15px; margin-top: 10px;">
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;">
                <span style="font-size: 45px;">🎓</span>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;">
                <span style="font-size: 45px;">💼</span>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;">
                <span style="font-size: 45px;">🏥</span>
            </div>
            <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px;">
                <span style="font-size: 45px;">✈️</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Divider
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Visa Type Selection Header
    st.markdown("""
    <h2 class="visa-cards-title">Select Visa Type</h2>
    <p class="visa-cards-subtitle">Choose a visa category to begin your eligibility screening</p>
    """, unsafe_allow_html=True)
    
    # Visa Cards
    cols = st.columns(5)
    
    visa_types = [
        {"name": "Student Visa", "icon": "🎓", "key": "Student Visa", "color": "#4fc3f7"},
        {"name": "Graduate Visa", "icon": "🎓", "key": "Graduate Visa", "color": "#81c784"},
        {"name": "Skilled Worker Visa", "icon": "💼", "key": "Skilled Worker Visa", "color": "#ffb74d"},
        {"name": "Health & Care Visa", "icon": "🏥", "key": "Health and Care Worker Visa", "color": "#f06292"},
        {"name": "Visitor Visa", "icon": "✈️", "key": "Visitor Visa", "color": "#9575cd"}
    ]
    
    for i, visa in enumerate(visa_types):
        with cols[i]:
            st.markdown(f"""
            <div class="visa-card">
                <div class="card-icon">{visa['icon']}</div>
                <div class="card-title">{visa['name']}</div>
                <div class="card-subtitle">Check requirements</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"Select", key=f"btn_{visa['key']}", use_container_width=True):
                go_to_form(visa['key'])
                st.rerun()

# =====================================================
# FORM PAGE - REAL-TIME SCORING
# =====================================================
def render_form():
    visa_type = st.session_state.selected_visa
    
    # Load FAISS retriever
    retriever = load_retriever()
    
    # Back button row
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("← Back to Home", key="back_btn"):
            go_home()
            st.rerun()
    
    # Form Header with icon
    icons = {
        "Student Visa": "🎓",
        "Graduate Visa": "🎓", 
        "Skilled Worker Visa": "💼",
        "Health and Care Worker Visa": "🏥",
        "Visitor Visa": "✈️"
    }
    
    st.markdown(f"""
    <div style="text-align: center; margin: 20px 0;">
        <div style="font-size: 60px; margin-bottom: 15px;">{icons.get(visa_type, '📋')}</div>
        <h1 style="color: white; font-size: 32px; margin-bottom: 10px;">{visa_type}</h1>
        <p style="color: #8ba3c4; font-size: 16px;">Fill in your details below to check your eligibility</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Form Container with submit button
    with st.form("visa_form"):
        form_data = {}
        
        # Import datetime for date handling
        from datetime import date, timedelta
        today = date.today()
        
        # =====================================================
        # COMMON ENTITIES - For All Visa Types
        # =====================================================
        st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            form_data["full_name"] = st.text_input("Full Name (as per passport) *")
            form_data["date_of_birth"] = st.date_input(
                "Date of Birth *",
                value=date(2000, 1, 1),
                min_value=date(1940, 1, 1),
                max_value=today - timedelta(days=365*16)
            )
            form_data["nationality"] = st.text_input("Nationality *")
        with col2:
            form_data["passport_number"] = st.text_input("Passport Number *")
            form_data["passport_issue_date"] = st.date_input(
                "Passport Issue Date",
                value=today - timedelta(days=365),
                min_value=date(2000, 1, 1),
                max_value=today
            )
            form_data["passport_expiry_date"] = st.date_input(
                "Passport Expiry Date *",
                value=today + timedelta(days=365*5),
                min_value=today,
                max_value=date(2040, 12, 31)
            )
        with col3:
            form_data["country_of_application"] = st.text_input("Country of Application / Current Location *")
            form_data["email_address"] = st.text_input("Email Address *")
            form_data["phone_number"] = st.text_input("Phone Number *")
        
        st.markdown('<div class="section-header">🛂 Visa Application Details</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            form_data["visa_type_applying_for"] = visa_type
            st.text_input("Visa Type Applying For", value=visa_type, disabled=True)
            form_data["purpose_of_visit"] = st.text_input("Purpose of Visit *")
        with col2:
            form_data["intended_travel_date"] = st.date_input(
                "Intended Travel / Start Date *",
                value=today + timedelta(days=30),
                min_value=today,
                max_value=date(2030, 12, 31)
            )
            form_data["intended_length_of_stay"] = st.text_input("Intended Length of Stay *")
        with col3:
            form_data["funds_available"] = st.number_input("Funds Available (GBP) *", min_value=0, value=10000)
            form_data["current_address"] = st.text_input("Current Address")
        
        st.markdown('<div class="section-header">📋 Declaration</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            form_data["english_language_requirement_met"] = st.selectbox("English Language Requirement Met? *", ["Select", "Yes", "No"])
        with col2:
            form_data["criminal_history_declaration"] = st.selectbox("Criminal History Declaration *", ["Select", "Yes", "No"])
        with col3:
            form_data["previous_uk_visa_refusal"] = st.selectbox("Previous UK Visa Refusal? *", ["Select", "Yes", "No"])
        
        # =====================================================
        # VISA-SPECIFIC FIELDS
        # =====================================================
        
        if visa_type == "Student Visa":
            # Student Visa – Eligibility Entities
            st.markdown('<div class="section-header">📚 CAS & Course Information</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                form_data["has_cas"] = st.selectbox("Do you have a CAS (Confirmation of Acceptance for Studies)? *", ["Select", "Yes", "No"])
                form_data["cas_reference_number"] = st.text_input("CAS Reference Number *")
                form_data["education_provider_is_licensed"] = st.selectbox("Is your education provider a licensed sponsor? *", ["Select", "Yes", "No"])
            with col2:
                form_data["course_level"] = st.selectbox("Course Level *", 
                    ["Select", "English Language Course", "Foundation", "Undergraduate", "Postgraduate", "Doctoral/PhD"])
                form_data["course_full_time"] = st.selectbox("Is the course full-time? *", ["Select", "Yes", "No"])
            
            st.markdown('<div class="section-header">📅 Course Duration</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                form_data["course_start_date"] = st.date_input(
                    "Course Start Date *",
                    value=today + timedelta(days=60),
                    min_value=today,
                    max_value=date(2030, 12, 31)
                )
            with col2:
                form_data["course_end_date"] = st.date_input(
                    "Course End Date *",
                    value=today + timedelta(days=365),
                    min_value=today,
                    max_value=date(2035, 12, 31)
                )
            with col3:
                form_data["course_duration_months"] = st.number_input("Course Duration (months) *", min_value=1, max_value=72, value=12)
            
            st.markdown('<div class="section-header">💰 Financial Requirements</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                form_data["meets_financial_requirement"] = st.selectbox("Meets financial requirement? *", ["Select", "Yes", "No"])
            with col2:
                form_data["funds_held_for_28_days"] = st.selectbox("Funds held for 28 days? *", ["Select", "Yes", "No"])
            with col3:
                form_data["english_requirement_met"] = st.selectbox("English requirement met? *", ["Select", "Yes", "No"])
        
        elif visa_type == "Graduate Visa":
            # Graduate Visa – Eligibility Entities
            st.markdown('<div class="section-header">🎓 Current UK Status</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                form_data["currently_in_uk"] = st.selectbox("Are you currently in the UK? *", ["Select", "Yes", "No"])
                form_data["current_uk_visa_type"] = st.selectbox("Current UK Visa Type *", 
                    ["Select", "Student Visa", "Tier 4 (General) Student", "Other"])
            with col2:
                form_data["course_completed"] = st.selectbox("Have you completed your course? *", ["Select", "Yes", "No"])
                form_data["course_level_completed"] = st.selectbox("Course Level Completed *",
                    ["Select", "Undergraduate (Bachelor's)", "Postgraduate (Master's)", "Doctoral (PhD)", "PGCE", "PGDE"])
            with col3:
                form_data["education_provider_is_licensed"] = st.selectbox("Is your education provider a licensed sponsor? *", ["Select", "Yes", "No"])
            
            st.markdown('<div class="section-header">📋 Course Completion Details</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                form_data["provider_reported_completion_to_home_office"] = st.selectbox("Has your provider reported course completion to the Home Office? *", ["Select", "Yes", "No"])
                form_data["original_cas_reference"] = st.text_input("Original CAS Reference Number")
            with col2:
                form_data["student_visa_valid_on_application_date"] = st.selectbox("Will your Student Visa be valid on application date? *", ["Select", "Yes", "No"])
        
        elif visa_type == "Skilled Worker Visa":
            # Skilled Worker Visa – Eligibility Entities
            st.markdown('<div class="section-header">💼 Job Offer & Sponsorship</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                form_data["job_offer_confirmed"] = st.selectbox("Do you have a confirmed job offer? *", ["Select", "Yes", "No"])
                form_data["employer_is_licensed_sponsor"] = st.selectbox("Is your employer a licensed sponsor? *", ["Select", "Yes", "No"])
                form_data["certificate_of_sponsorship_issued"] = st.selectbox("Has Certificate of Sponsorship been issued? *", ["Select", "Yes", "No"])
            with col2:
                form_data["cos_reference_number"] = st.text_input("Certificate of Sponsorship (CoS) Reference Number *")
                form_data["job_title"] = st.text_input("Job Title *")
                form_data["soc_code"] = st.text_input("SOC (Standard Occupational Classification) Code *")
            
            st.markdown('<div class="section-header">💷 Salary & Eligibility</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                form_data["job_is_eligible_occupation"] = st.selectbox("Is the job an eligible occupation? *", ["Select", "Yes", "No"])
            with col2:
                form_data["salary_offered"] = st.number_input("Salary Offered (GBP/year) *", min_value=0, value=26200)
            with col3:
                form_data["meets_minimum_salary_threshold"] = st.selectbox("Meets minimum salary threshold? *", ["Select", "Yes", "No"])
            
            st.markdown('<div class="section-header">📜 Additional Requirements</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                form_data["english_requirement_met"] = st.selectbox("English requirement met? *", ["Select", "Yes", "No"])
            with col2:
                form_data["criminal_record_certificate_required"] = st.selectbox("Criminal record certificate required? *", ["Select", "Yes", "No"])
            with col3:
                form_data["criminal_record_certificate_provided"] = st.selectbox("Criminal record certificate provided?", ["Select", "Yes", "No", "N/A"])
        
        elif visa_type == "Health and Care Worker Visa":
            # Health & Care Visa – Eligibility Entities
            st.markdown('<div class="section-header">🏥 Job Offer & Sponsorship</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                form_data["job_offer_confirmed"] = st.selectbox("Do you have a confirmed job offer? *", ["Select", "Yes", "No"])
                form_data["employer_is_licensed_healthcare_sponsor"] = st.selectbox("Is your employer a licensed healthcare sponsor? *", ["Select", "Yes", "No"])
                form_data["certificate_of_sponsorship_issued"] = st.selectbox("Has Certificate of Sponsorship been issued? *", ["Select", "Yes", "No"])
            with col2:
                form_data["cos_reference_number"] = st.text_input("Certificate of Sponsorship (CoS) Reference Number *")
                form_data["job_title"] = st.text_input("Job Title *")
                form_data["soc_code"] = st.text_input("SOC (Standard Occupational Classification) Code *")
            
            st.markdown('<div class="section-header">💷 Healthcare Role & Salary</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                form_data["job_is_eligible_healthcare_role"] = st.selectbox("Is the job an eligible healthcare role? *", ["Select", "Yes", "No"])
            with col2:
                form_data["salary_offered"] = st.number_input("Salary Offered (GBP/year) *", min_value=0, value=23200)
            with col3:
                form_data["meets_healthcare_salary_rules"] = st.selectbox("Meets healthcare salary rules? *", ["Select", "Yes", "No"])
            
            st.markdown('<div class="section-header">📜 Professional Registration</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                form_data["professional_registration_required"] = st.selectbox("Professional registration required? *", ["Select", "Yes", "No"])
            with col2:
                form_data["professional_registration_provided"] = st.selectbox("Professional registration provided?", ["Select", "Yes", "No", "N/A"])
            with col3:
                form_data["english_requirement_met"] = st.selectbox("English requirement met? *", ["Select", "Yes", "No"])
        
        elif visa_type == "Visitor Visa":
            # Visitor Visa – Eligibility Entities
            st.markdown('<div class="section-header">✈️ Visit Purpose & Duration</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                form_data["purpose_of_visit"] = st.selectbox("Purpose of Visit *",
                    ["Select", "Tourism", "Visiting Family/Friends", "Business Meeting", "Conference/Event", 
                     "Medical Treatment", "Academic Research (up to 6 months)", "Creative/Entertainment", 
                     "Sports Activities", "Transit", "Other Permitted Activities"])
                form_data["purpose_is_permitted_under_visitor_rules"] = st.selectbox("Is your purpose permitted under visitor rules? *", ["Select", "Yes", "No", "Not Sure"])
            with col2:
                form_data["intended_length_of_stay_months"] = st.number_input("Intended Length of Stay (months) *", min_value=0.0, max_value=12.0, value=1.0, step=0.5)
                form_data["stay_within_6_months_limit"] = st.selectbox("Is your stay within the 6 months limit? *", ["Select", "Yes", "No"])
            
            st.markdown('<div class="section-header">🏠 Accommodation & Travel</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                form_data["accommodation_arranged"] = st.selectbox("Accommodation arranged? *", ["Select", "Yes", "No"])
            with col2:
                form_data["return_or_onward_travel_planned"] = st.selectbox("Return or onward travel planned? *", ["Select", "Yes", "No"])
            with col3:
                form_data["intends_to_leave_uk_after_visit"] = st.selectbox("Intends to leave UK after visit? *", ["Select", "Yes", "No"])
            
            st.markdown('<div class="section-header">💰 Financial Support</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                form_data["sufficient_funds_for_stay"] = st.selectbox("Sufficient funds for stay? *", ["Select", "Yes", "No"])
            with col2:
                form_data["funds_available_visitor"] = st.number_input("Funds Available for Visit (GBP)", min_value=0, value=3000)
        
        # =====================================================
        # SUBMIT BUTTON
        # =====================================================
        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Check My Eligibility", use_container_width=True, type="primary")
    
    # =====================================================
    # RESULTS DISPLAY (Below Form)
    # =====================================================
    if submitted:
        if not form_data.get("full_name") or not form_data.get("nationality"):
            st.error("⚠️ Please fill in Name and Nationality fields")
        else:
            # Calculate eligibility score
            rule_status, passed_checks, failed_checks, percentage_score, earned, total, scores = check_eligibility_rules(visa_type, form_data)
            
            # Determine status
            if percentage_score >= 80:
                score_bg = "linear-gradient(135deg, #4caf50, #2e7d32)"
                status_text = "ELIGIBLE"
                status_emoji = "✅"
                status_color = "#4caf50"
            elif percentage_score >= 50:
                score_bg = "linear-gradient(135deg, #ff9800, #f57c00)"
                status_text = "POTENTIALLY ELIGIBLE"
                status_emoji = "⚠️"
                status_color = "#ff9800"
            else:
                score_bg = "linear-gradient(135deg, #f44336, #c62828)"
                status_text = "NOT ELIGIBLE"
                status_emoji = "❌"
                status_color = "#f44336"
            
            # Display Score Card
            st.markdown(f"""
            <div style="background: {score_bg}; border-radius: 20px; padding: 40px; text-align: center; margin: 30px 0;">
                <div style="font-size: 14px; color: rgba(255,255,255,0.8); margin-bottom: 10px;">ELIGIBILITY SCORE</div>
                <div style="font-size: 72px; font-weight: 700; color: white;">{percentage_score}%</div>
                <div style="font-size: 18px; color: rgba(255,255,255,0.9);">{earned} / {total} points</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Status Banner
            st.markdown(f"""
            <div style="background: {status_color}; border-radius: 15px; padding: 20px; text-align: center; margin-bottom: 30px;">
                <span style="font-size: 24px; font-weight: 600; color: white;">{status_emoji} {status_text} - {"All requirements met" if percentage_score >= 80 else "Requirements not met"}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Score Breakdown
            st.markdown("""
            <div style="background: rgba(255,255,255,0.05); border-radius: 15px; padding: 25px; margin-bottom: 20px;">
                <h3 style="color: #64b5f6; margin-bottom: 20px;">📊 Score Breakdown</h3>
            """, unsafe_allow_html=True)
            
            for req_name, req_data in scores.items():
                status = req_data['status']
                if status == 'pass':
                    icon = "✅"
                    color = "#4caf50"
                elif status == 'fail':
                    icon = "❌"
                    color = "#f44336"
                elif status == 'warning':
                    icon = "⚠️"
                    color = "#ff9800"
                else:
                    icon = "⭕"
                    color = "#9e9e9e"
                
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <span style="color: white; font-size: 14px;">{icon} {req_name}</span>
                    <span style="color: {color}; font-size: 14px; font-weight: 600;">{req_data['earned']}/{req_data['max']} points</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Requirements Met/Failed
            if passed_checks:
                st.markdown("""
                <div style="background: rgba(76, 175, 80, 0.1); border-radius: 15px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(76, 175, 80, 0.3);">
                    <h4 style="color: #4caf50; margin-bottom: 15px;">✅ Requirements Met</h4>
                """, unsafe_allow_html=True)
                for check in passed_checks:
                    st.markdown(f'<div style="color: #a5d6a7; padding: 5px 0;">• {check}</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            if failed_checks:
                st.markdown("""
                <div style="background: rgba(244, 67, 54, 0.1); border-radius: 15px; padding: 20px; margin-bottom: 20px; border: 1px solid rgba(244, 67, 54, 0.3);">
                    <h4 style="color: #f44336; margin-bottom: 15px;">❌ Issues Found</h4>
                """, unsafe_allow_html=True)
                for check in failed_checks:
                    st.markdown(f'<div style="color: #ef9a9a; padding: 5px 0;">• {check}</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Policy Information from FAISS
            policy_info = get_policy_info(visa_type, retriever)
            if policy_info:
                st.markdown("""
                <div style="background: rgba(100, 181, 246, 0.1); border-radius: 15px; padding: 20px; border: 1px solid rgba(100, 181, 246, 0.3);">
                    <h4 style="color: #64b5f6; margin-bottom: 15px;">📖 Relevant Policy Information</h4>
                """, unsafe_allow_html=True)
                for i, info in enumerate(policy_info[:3]):
                    st.markdown(f'<div style="color: #b0c4de; padding: 10px 0; border-bottom: 1px solid rgba(255,255,255,0.1);">{info["text"][:500]}...</div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Store in session state for navigation
            st.session_state.score = percentage_score
            st.session_state.score_details = {
                "earned": earned,
                "total": total,
                "breakdown": scores,
                "passed": passed_checks,
                "failed": failed_checks
            }
            st.session_state.result = "completed"
            st.session_state.eligibility_status = status_text
            
            # Add action buttons
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Check Another Visa Type", use_container_width=True):
                    go_home()
                    st.rerun()
            with col2:
                if st.button("📝 Start New Application", use_container_width=True):
                    st.session_state.result = None
                    st.session_state.eligibility_status = None
                    st.session_state.score = None
                    st.session_state.score_details = None
                    st.rerun()

# =====================================================
# MAIN APP
# =====================================================
def main():
    if st.session_state.page == 'home':
        render_home()
    elif st.session_state.page == 'form':
        render_form()

if __name__ == "__main__":
    main()
