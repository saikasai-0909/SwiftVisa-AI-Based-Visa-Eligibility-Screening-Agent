import streamlit as st
from groq import Groq
from PyPDF2 import PdfReader
from datetime import date
import os

st.set_page_config(
    page_title="SwiftVisa | UK Visa Eligibility",
    page_icon="🛂",
    layout="wide"
)

st.markdown("""
<style>

header,
footer,
.stDeployButton {
    display: none !important;
}

.stApp {
    background: linear-gradient(
        135deg,
        #EEF2F7 0%,
        #E6EBF2 50%,
        #F4F6FA 100%
    );
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.title-center {
    text-align: center;
    margin-bottom: 2.5rem;
}

.title-center h1 {
    font-size: 2.6rem;
    font-weight: 800;
    color: #1F2933;
    margin-bottom: 0.4rem;
}

.title-center h3 {
    font-size: 1.2rem;
    font-weight: 500;
    color: #475569;
}

.page-card {
    background: #FFFFFF;
    padding: 2.2rem;
    border-radius: 22px;
    box-shadow: 0 16px 38px rgba(0,0,0,0.14);
    margin-bottom: 2rem;
}

.visa-card {
    background: #FFFFFF;
    border-radius: 20px;
    padding: 1rem;
    box-shadow: 0 14px 30px rgba(0,0,0,0.16);
    transition: all 0.25s ease;
    text-align: center;
}

.visa-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 26px 52px rgba(0,0,0,0.25);
}

.visa-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1F2933;
    margin-top: 0.8rem;
}

.visa-description {
    font-size: 0.9rem;
    color: #475569;
    min-height: 70px;
    margin-bottom: 10px;
}

h1, h2, h3 {
    color: #1F2933 !important;
    font-weight: 700;
}

input,
textarea,
select {
    background-color: #FAFAFA !important;
    border-radius: 10px !important;
    border: 1px solid #CBD5E1 !important;
    padding: 0.55rem !important;
    color: #1F2933 !important;
}

label,
div[data-baseweb="radio"] span {
    color: #1F2933 !important;
    font-weight: 500;
}

.stButton button {
    width: 100%;
    background: linear-gradient(90deg, #1D4ED8, #3B82F6);
    color: #FFFFFF;
    font-weight: 700;
    border-radius: 14px;
    padding: 0.7rem 1.1rem;
    border: none;
    transition: all 0.2s ease;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 22px rgba(0,0,0,0.25);
}

.result-box {
    background: #FFFFFF;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0 18px 40px rgba(0,0,0,0.15);
    max-width: 900px;
    margin: auto;
    font-size: 1.05rem;
    line-height: 1.6;
}


::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-thumb {
    background: #CBD5E1;
    border-radius: 6px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

 <div style="
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 2.5rem;
        text-align: center;
    ">
        <h1 style="
            font-size: 2.6rem;
            font-weight: 800;
            color: #1F2933;
            margin-bottom: 0.4rem;
        ">
            🛂 SwiftVisa – UK Visa Eligibility System
        </h1>

        <h3 style="
            font-size: 1.2rem;
            font-weight: 500;
            color: #475569;
        ">
            Select a Visa Type
        </h3>
    </div>
</style>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state.page = "visa_selection"
if "visa" not in st.session_state:
    st.session_state.visa = None
if "common" not in st.session_state:
    st.session_state.common = {}
if "visa_data" not in st.session_state:
    st.session_state.visa_data = {}

client = Groq(
    api_key=os.getenv("GROQ_API_KEY"))

POLICY_FILES = {
    "Student Visa": "DataSets/Student Visa.pdf",
    "Graduate Visa": "DataSets/Graduate Visa.pdf",
    "Skilled Worker Visa": "DataSets/Work Visa.pdf",
    "Health & Care Visa": "DataSets/Health Visa.pdf",
    "Visitor Visa": "DataSets/Visitor Visa.pdf"
}

VISA_IMAGES = {
    "Student Visa": "https://images.unsplash.com/photo-1541339907198-e08756dedf3f",
    "Graduate Visa": "https://images.unsplash.com/photo-1513258496099-48168024aec0",
    "Skilled Worker Visa": "https://images.unsplash.com/photo-1521737604893-d14cc237f11d",
    "Health & Care Visa": "https://images.unsplash.com/photo-1584515933487-779824d29309",
    "Visitor Visa": "https://images.unsplash.com/photo-1504609813442-a8924e83f76e"
}

VISA_DESCRIPTIONS = {
    "Student Visa": "Study full-time at a licensed UK education provider.",
    "Graduate Visa": "Stay and work in the UK after completing a UK degree.",
    "Skilled Worker Visa": "Work in the UK with a licensed employer sponsor.",
    "Health & Care Visa": "Work in eligible healthcare roles in the UK.",
    "Visitor Visa": "Short-term visits for tourism, business or family."
}

@st.cache_data
def load_policy(path):
    reader = PdfReader(path)
    text = ""
    for p in reader.pages:
        if p.extract_text():
            text += p.extract_text()
    return text[:5000]

if st.session_state.page == "visa_selection":
    title_left, title_center, title_right = st.columns([2, 2.5, 2])
    with title_center:
        st.title("🛂 SwiftVisa ")

    sub_left, sub_center, sub_right = st.columns([1.5, 3, 1])
    with sub_center:
        st.subheader("AI-Based Visa Eligibility Screening Agent")

    row1 = st.columns(3)
    visas_row1 = [
        "Student Visa",
        "Graduate Visa",
        "Skilled Worker Visa"
    ]

    for i, visa in enumerate(visas_row1):
        with row1[i]:
            st.image(VISA_IMAGES[visa], width="stretch")
            st.markdown(f"### {visa}")
            st.caption(VISA_DESCRIPTIONS[visa])
            if st.button(f"Apply – {visa}", key=visa):
                st.session_state.visa = visa
                st.session_state.page = "application"
                st.rerun()

    spacer_left, col_center1, col_center2, spacer_right = st.columns([0.5, 1, 1, 0.5])

    visas_row2 = [
        "Health & Care Visa",
        "Visitor Visa"
    ]

    for col, visa in zip([col_center1, col_center2], visas_row2):
        with col:
            st.image(VISA_IMAGES[visa], width="stretch")
            st.markdown(f"### {visa}")
            st.caption(VISA_DESCRIPTIONS[visa])
            if st.button(f"Apply – {visa}", key=visa + "_2"):
                st.session_state.visa = visa
                st.session_state.page = "application"
                st.rerun()

elif st.session_state.page == "application":

    st.markdown("<div class='page-card'>", unsafe_allow_html=True)
    st.header("Applicant Details")

    previous_refusal = st.radio("Previous UK Visa Refusal?", ["No", "Yes"])
    refusal_year = None
    if previous_refusal == "Yes":
        refusal_year = st.number_input(
            "Year of Refusal",
            min_value=1990,
            max_value=date.today().year,
            step=1
        )

    st.session_state.common = {
        "full_name": st.text_input("Full Name (as per passport)"),
        "date_of_birth": st.date_input("Date of Birth", max_value=date.today()),
        "nationality": st.text_input("Nationality"),
        "passport_number": st.text_input("Passport Number"),
        "passport_issue_date": st.date_input("Passport Issue Date", max_value=date.today()),
        "passport_expiry_date": st.date_input("Passport Expiry Date", min_value=date.today()),
        "country_of_application": st.text_input("Country of Application"),
        "purpose_of_visit": st.text_input("Purpose of Visit"),
        "intended_travel_date": st.date_input("Intended Travel Date", min_value=date.today()),
        "length_of_stay_months": st.number_input("Length of Stay (Months)", min_value=1),
        "funds_available": st.number_input("Funds Available (£)", step=500),
        "english_language_requirement_met": st.radio("English Requirement Met?", ["Yes","No"]),
        "criminal_history": st.radio("Criminal History?", ["Yes","No"]),
        "previous_uk_visa_refusal": previous_refusal,
        "previous_uk_visa_refusal_year": refusal_year,
        "email": st.text_input("Email"),
        "phone": st.text_input("Phone Number"),
        "address": st.text_area("Current Address")
    }

    if st.button("Next → Visa Specific Details"):
        st.session_state.page = "visa_specific"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "visa_specific":

    st.markdown("<div class='page-card'>", unsafe_allow_html=True)
    st.header(f"{st.session_state.visa} – Visa Specific Details")

    v = st.session_state.visa

    if v == "Student Visa":
        st.session_state.visa_data = {
            "has_cas": st.radio("CAS Issued?", ["Yes","No"]),
            "cas_reference_number": st.text_input("CAS Reference Number"),
            "education_provider_is_licensed": st.radio("Licensed Provider?", ["Yes","No"]),
            "course_level": st.selectbox("Course Level", ["Bachelor","Master","PhD"]),
            "course_full_time": st.radio("Full Time Course?", ["Yes","No"]),
            "course_start_date": st.date_input("Course Start Date"),
            "course_end_date": st.date_input("Course End Date"),
            "course_duration_months": st.number_input("Course Duration (Months)", min_value=1),
            "meets_financial_requirement": st.radio("Financial Requirement Met?", ["Yes","No"]),
            "funds_held_for_28_days": st.radio("Funds Held 28 Days?", ["Yes","No"]),
            "english_requirement_met": st.radio("English Requirement Met?", ["Yes","No"])
        }

    elif v == "Graduate Visa":
        st.session_state.visa_data = {
            "currently_in_uk": st.radio("Currently in UK?", ["Yes","No"]),
            "current_uk_visa_type": st.selectbox("Current Visa Type", ["Student","Tier 4"]),
            "course_completed": st.radio("Course Completed?", ["Yes","No"]),
            "course_level_completed": st.selectbox("Course Level Completed", ["Bachelor","Master","PhD"]),
            "education_provider_is_licensed": st.radio("Licensed Provider?", ["Yes","No"]),
            "provider_reported_completion": st.radio("Reported to Home Office?", ["Yes","No"]),
            "original_cas_reference": st.text_input("Original CAS Reference"),
            "student_visa_valid": st.radio("Student Visa Valid?", ["Yes","No"])
        }

    elif v == "Skilled Worker Visa":
        st.session_state.visa_data = {
            "job_offer_confirmed": st.radio("Job Offer Confirmed?", ["Yes","No"]),
            "employer_is_licensed_sponsor": st.radio("Licensed Sponsor?", ["Yes","No"]),
            "certificate_of_sponsorship_issued": st.radio("CoS Issued?", ["Yes","No"]),
            "cos_reference_number": st.text_input("CoS Reference Number"),
            "job_title": st.text_input("Job Title"),
            "soc_code": st.text_input("SOC Code"),
            "job_is_eligible_occupation": st.radio("Eligible Occupation?", ["Yes","No"]),
            "salary_offered": st.number_input("Salary (£)", step=1000),
            "meets_minimum_salary_threshold": st.radio("Salary Threshold Met?", ["Yes","No"]),
            "english_requirement_met": st.radio("English Requirement Met?", ["Yes","No"]),
            "criminal_record_certificate_required": st.radio("Criminal Record Required?", ["Yes","No"]),
            "criminal_record_certificate_provided": st.radio("Criminal Record Provided?", ["Yes","No"])
        }

    elif v == "Health & Care Visa":
        st.session_state.visa_data = {
            "job_offer_confirmed": st.radio("Job Offer Confirmed?", ["Yes","No"]),
            "employer_is_licensed_healthcare_sponsor": st.radio("Licensed Healthcare Sponsor?", ["Yes","No"]),
            "certificate_of_sponsorship_issued": st.radio("CoS Issued?", ["Yes","No"]),
            "cos_reference_number": st.text_input("CoS Reference Number"),
            "job_title": st.text_input("Job Title"),
            "soc_code": st.text_input("SOC Code"),
            "job_is_eligible_healthcare_role": st.radio("Eligible Role?", ["Yes","No"]),
            "salary_offered": st.number_input("Salary (£)", step=1000),
            "meets_healthcare_salary_rules": st.radio("Meets Salary Rules?", ["Yes","No"]),
            "professional_registration_required": st.radio("Registration Required?", ["Yes","No"]),
            "professional_registration_provided": st.radio("Registration Provided?", ["Yes","No"]),
            "english_requirement_met": st.radio("English Requirement Met?", ["Yes","No"])
        }

    else:
        st.session_state.visa_data = {
            "purpose_of_visit": st.selectbox("Purpose", ["Tourism","Business","Family"]),
            "purpose_is_permitted_under_visitor_rules": st.radio("Purpose Permitted?", ["Yes","No"]),
            "intended_length_of_stay_months": st.number_input("Stay Length (Months)", min_value=1, max_value=6),
            "stay_within_6_months_limit": st.radio("Within 6 Months?", ["Yes","No"]),
            "accommodation_arranged": st.radio("Accommodation Arranged?", ["Yes","No"]),
            "return_or_onward_travel_planned": st.radio("Return Travel Planned?", ["Yes","No"]),
            "intends_to_leave_uk_after_visit": st.radio("Will Leave UK?", ["Yes","No"]),
            "sufficient_funds_for_stay": st.radio("Sufficient Funds?", ["Yes","No"])
        }

    if st.button("Check Eligibility"):
        st.session_state.page = "result"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

elif st.session_state.page == "result":

    policy = load_policy(POLICY_FILES[st.session_state.visa])

    prompt = f"""
You are a UK immigration eligibility officer.

POLICY RULES:
{policy}

COMMON DETAILS:
{st.session_state.common}

VISA DETAILS:
{st.session_state.visa_data}

TASK:
1. Decide ELIGIBLE or NOT ELIGIBLE
2. Give bullet-point reasons
3. End with final decision line
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        messages=[{"role":"user","content":prompt}]
    )

    st.markdown(f"<div class='page-card'>{response.choices[0].message.content}</div>", unsafe_allow_html=True)

    if st.button("Start New Application"):
        st.session_state.clear()
        st.rerun()
