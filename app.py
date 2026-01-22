import streamlit as st
import sys
import os
from datetime import date

# -----------------------------
# Backend import
# -----------------------------
sys.path.append(os.path.abspath("src"))
from run_rag_pipeline import run_eligibility_check

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="SwiftVisa | UK Visa Eligibility",
    page_icon="🛂",
    layout="wide"
)

# -----------------------------
# Custom Background + Styling
# -----------------------------
st.markdown(
    """
    <style>
    /* ============================
       UK GOV INSPIRED THEME
       ============================ */

    :root {
        --gov-blue: #0B5ED7;
        --gov-dark-blue: #084298;
        --gov-bg: #F3F7FB;
        --gov-card: #FFFFFF;
        --gov-border: #DEE2E6;
        --gov-text: #212529;
        --gov-muted: #6C757D;
        --success: #198754;
        --warning: #FFC107;
        --error: #DC3545;
    }

    /* Page Background */
    body {
        background-color: var(--gov-bg);
        color: var(--gov-text);
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
    }

    /* Main container */
    .block-container {
        max-width: 1200px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .swiftvisa-header {
        background: linear-gradient(90deg, var(--gov-blue), var(--gov-dark-blue));
        padding: 2.2rem 1rem;
        border-radius: 14px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.15);
    }

    .swiftvisa-header h1 {
        font-size: 2.3rem;
        margin-bottom: 0.4rem;
    }

    .swiftvisa-header p {
        font-size: 1rem;
        opacity: 0.95;
    }

    /* Card layout */
    .card {
        background: var(--gov-card);
        border-radius: 14px;
        padding: 1.6rem;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        border: 1px solid var(--gov-border);
        animation: fadeUp 0.5s ease-in;
    }

    /* Section titles */
    .section-title {
        text-align: center;
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--gov-dark-blue);
        margin-bottom: 1rem;
    }
    .centered-title::after {
    content: "";
    display: block;
    width: 80px;
    height: 5px;
    background: var(--gov-blue);
    margin: 8px auto 0 auto;
    border-radius: 2px;
}


    /* Buttons */
    .stButton > button {
        background-color: var(--gov-blue);
        color: white;
        border-radius: 8px;
        padding: 0.6em 1.5em;
        font-weight: 600;
        transition: all 0.25s ease;
        border: none;
    }

    .stButton > button:hover {
        background-color: var(--gov-dark-blue);
        transform: translateY(-2px);
    }

    /* Inputs */
    input, textarea, select {
        border-radius: 8px !important;
        border: 1px solid var(--gov-border) !important;
    }

    /* Info / Alerts */
    .stAlert {
        border-radius: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: var(--gov-muted);
        font-size: 0.85rem;
        margin-top: 2rem;
    }

    /* Animation */
    @keyframes fadeUp {
        from {
            opacity: 0;
            transform: translateY(16px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="swiftvisa-header">
        <h1>🛂 SwiftVisa: AI-Based Visa Eligibility Screening Agent</h1>
        <p>Official UK visa eligibility analysis based strictly on government policy</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Helpers
# -----------------------------
def yes_no(label):
    return st.radio(label, ["Yes", "No"], horizontal=True)

# ======================================================
# STEP 1: VISA TYPE (OUTSIDE FORM — IMPORTANT)
# ======================================================
st.subheader("📄 Visa Application")

visa_type = st.selectbox(
    "Visa Type Applying For",
    [
        "Visitor Visa",
        "Health & Care Visa",
        "Skilled Worker Visa",
        "Graduate Visa",
        "Student Visa"
    ],
    key="visa_type"
)

# ======================================================
# STEP 2: FORM (COMMON + VISA-SPECIFIC)
# ======================================================
with st.form("visa_form"):

    # ==============================
    # COMMON ENTITIES
    # ==============================
    st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        justify-content:center;
        gap:6px;
        font-size:clamp(1.1rem, 2vw, 1.3rem);
        font-weight:600;
        position:relative;
        margin:1.1rem 0 1.4rem 0;
    ">
        <span>👤</span>
        <span>Applicant Information</span>
        <div style="
            position:absolute;
            bottom:-5px;
            width:60px;
            height:3px;
            background:#0B5ED7;
            border-radius:2px;
        "></div>
    </div>
    """,
    unsafe_allow_html=True
)


    c1, c2 = st.columns(2)
    with c1:
        full_name = st.text_input("Full Name (as per passport)")
        dob = st.date_input("Date of Birth")
        nationality = st.text_input("Nationality")
        passport_number = st.text_input("Passport Number")
        passport_issue_date = st.date_input("Passport Issue Date")

    with c2:
        passport_expiry_date = st.date_input("Passport Expiry Date")
        country_of_application = st.text_input("Country of Application / Current Location")
        purpose_of_visit = st.text_input("Purpose of Visit")
        intended_travel_date = st.date_input("Intended Travel / Start Date")

    c3, c4 = st.columns(2)
    with c3:
        intended_length_of_stay = st.number_input(
            "Intended Length of Stay (months)", min_value=1, max_value=60
        )
        funds_available = st.number_input("Funds Available (£)", min_value=0)

    with c4:
        english_language_met = yes_no("English Language Requirement Met")
        criminal_history = yes_no("Criminal History Declaration")
        previous_refusal = yes_no("Previous UK Visa Refusal")

   
    st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        justify-content:center;
        gap:6px;
        font-size:clamp(0.95rem, 1.6vw, 1.15rem);
        font-weight:600;
        position:relative;
        margin:1.2rem 0 1.5rem 0;
    ">
        <span>📞</span>
        <span>Contact Details</span>
        <div style="
            position:absolute;
            bottom:-5px;
            width:55px;
            height:3px;
            background:#198754;
            border-radius:2px;
        "></div>
    </div>
    """,
    unsafe_allow_html=True
)




    c5, c6 = st.columns(2)
    with c5:
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")
    with c6:
        address = st.text_area("Current Address")

    # ==============================
    # VISA-SPECIFIC ENTITIES
    # ==============================
    st.markdown(
    """
    <div style="
        display:flex;
        align-items:center;
        justify-content:center;
        gap:6px;
        font-size:clamp(1.05rem, 1.9vw, 1.25rem);
        font-weight:600;
        position:relative;
        margin:1.1rem 0 1.4rem 0;
    ">
        <span>📌</span>
        <span>Visa-Specific Information</span>
        <div style="
            position:absolute;
            bottom:-5px;
            width:60px;
            height:3px;
            background:#FFC107;
            border-radius:2px;
        "></div>
    </div>
    """,
    unsafe_allow_html=True
)



    visa_specific = {}

    # -------- Graduate Visa --------
    if visa_type == "Graduate Visa":
        visa_specific = {
            "currently_in_uk": yes_no("Currently in the UK"),
            "current_uk_visa_type": st.selectbox("Current UK Visa Type", ["Student", "Tier 4"]),
            "course_completed": yes_no("Course Completed"),
            "course_level_completed": st.text_input("Course Level Completed"),
            "education_provider_is_licensed": yes_no("Education Provider is Licensed"),
            "provider_reported_completion_to_home_office": yes_no(
                "Provider Reported Completion to Home Office"
            ),
            "original_cas_reference": st.text_input("Original CAS Reference"),
            "student_visa_valid_on_application_date": yes_no(
                "Student Visa Valid on Application Date"
            )
        }

    # -------- Student Visa --------
    elif visa_type == "Student Visa":
        visa_specific = {
            "has_cas": yes_no("CAS Issued"),
            "cas_reference_number": st.text_input("CAS Reference Number"),
            "education_provider_is_licensed": yes_no("Education Provider is Licensed"),
            "course_level": st.text_input("Course Level"),
            "course_full_time": yes_no("Course is Full-Time"),
            "course_start_date": st.date_input("Course Start Date"),
            "course_end_date": st.date_input("Course End Date"),
            "course_duration_months": st.number_input("Course Duration (months)", 1, 60),
            "meets_financial_requirement": yes_no("Meets Financial Requirement"),
            "funds_held_for_28_days": yes_no("Funds Held for 28 Days"),
            "english_requirement_met": yes_no("English Requirement Met")
        }

    # -------- Skilled Worker Visa --------
    elif visa_type == "Skilled Worker Visa":
        visa_specific = {
            "job_offer_confirmed": yes_no("Job Offer Confirmed"),
            "employer_is_licensed_sponsor": yes_no("Employer is Licensed Sponsor"),
            "certificate_of_sponsorship_issued": yes_no("Certificate of Sponsorship Issued"),
            "cos_reference_number": st.text_input("CoS Reference Number"),
            "job_title": st.text_input("Job Title"),
            "soc_code": st.text_input("SOC Code"),
            "job_is_eligible_occupation": yes_no("Job is Eligible Occupation"),
            "salary_offered": st.number_input("Salary Offered (£)", min_value=0),
            "meets_minimum_salary_threshold": yes_no("Meets Minimum Salary Threshold"),
            "english_requirement_met": yes_no("English Requirement Met"),
            "criminal_record_certificate_required": yes_no("Criminal Record Certificate Required"),
            "criminal_record_certificate_provided": yes_no("Criminal Record Certificate Provided")
        }

    # -------- Health & Care Visa --------
    elif visa_type == "Health & Care Visa":
        visa_specific = {
            "job_offer_confirmed": yes_no("Job Offer Confirmed"),
            "employer_is_licensed_healthcare_sponsor": yes_no(
                "Employer is Licensed Healthcare Sponsor"
            ),
            "certificate_of_sponsorship_issued": yes_no("Certificate of Sponsorship Issued"),
            "cos_reference_number": st.text_input("CoS Reference Number"),
            "job_title": st.text_input("Job Title"),
            "soc_code": st.text_input("SOC Code"),
            "job_is_eligible_healthcare_role": yes_no("Job is Eligible Healthcare Role"),
            "salary_offered": st.number_input("Salary Offered (£)", min_value=0),
            "meets_healthcare_salary_rules": yes_no("Meets Healthcare Salary Rules"),
            "professional_registration_required": yes_no("Professional Registration Required"),
            "professional_registration_provided": yes_no("Professional Registration Provided"),
            "english_requirement_met": yes_no("English Requirement Met")
        }

    # -------- Visitor Visa --------
    elif visa_type == "Visitor Visa":
        visa_specific = {
            "purpose_of_visit": purpose_of_visit,
            "purpose_is_permitted_under_visitor_rules": yes_no(
                "Purpose Permitted Under Visitor Rules"
            ),
            "intended_length_of_stay_months": intended_length_of_stay,
            "stay_within_6_months_limit": yes_no("Stay Within 6 Months"),
            "accommodation_arranged": yes_no("Accommodation Arranged"),
            "return_or_onward_travel_planned": yes_no("Return / Onward Travel Planned"),
            "intends_to_leave_uk_after_visit": yes_no("Intends to Leave UK After Visit"),
            "sufficient_funds_for_stay": yes_no("Sufficient Funds for Stay")
        }

    submitted = st.form_submit_button("🔍 Assess Eligibility")



def validate_required_fields(common_details, visa_specific):
    missing_fields = []

    # ---------- Common mandatory fields ----------
    required_common = {
        "Full Name": common_details["full_name"],
        "Nationality": common_details["nationality"],
        "Passport Number": common_details["passport_number"],
        "Country of Application": common_details["country_of_application"],
        "Purpose of Visit": common_details["purpose_of_visit"],
        "Email": common_details["email"],
        "Phone": common_details["phone"],
    }

    for label, value in required_common.items():
        if not value or str(value).strip() == "":
            missing_fields.append(label)

    # ---------- Visa-specific mandatory fields ----------
    for key, value in visa_specific.items():
        if isinstance(value, str) and value.strip() == "":
            missing_fields.append(key.replace("_", " ").title())

    return missing_fields

# ======================================================
# STEP 3: SUBMIT → RAG
# ======================================================
if submitted:
    payload = {
        "common_details": {
            "full_name": full_name,
            "date_of_birth": str(dob),
            "nationality": nationality,
            "passport_number": passport_number,
            "passport_issue_date": str(passport_issue_date),
            "passport_expiry_date": str(passport_expiry_date),
            "country_of_application": country_of_application,
            "visa_type": visa_type,
            "purpose_of_visit": purpose_of_visit,
            "intended_travel_date": str(intended_travel_date),
            "intended_length_of_stay": intended_length_of_stay,
            "funds_available": funds_available,
            "english_language_met": english_language_met,
            "criminal_history": criminal_history,
            "previous_uk_visa_refusal": previous_refusal,
            "email": email,
            "phone": phone,
            "address": address
        },
        "visa_specific_details": visa_specific
    }

    # -----------------------------
    # VALIDATION CHECK
    # -----------------------------
    missing_fields = validate_required_fields(
        payload["common_details"],
        payload["visa_specific_details"]
    )

    if missing_fields:
        st.error("⚠️ Please fill all required fields before submitting.")
        with st.expander("Missing Fields"):
            for field in missing_fields:
                st.markdown(f"- {field}")
        st.stop()   # 🚨 stops execution → RAG NOT called

    # -----------------------------
    # SAFE TO CALL RAG
    # -----------------------------
    with st.spinner("Analyzing eligibility using official UK visa policy..."):
        result = run_eligibility_check(payload)
        st.progress(100)

    st.divider()
   
    st.markdown(
    """
    <div style="text-align:center;font-weight:600;margin-bottom:1rem;">
        🔍 Eligibility Results
    </div>
    """,
    unsafe_allow_html=True
)

    status = result["eligibility_status"]
    if "eligible" in status.lower() and "not" not in status.lower():
        st.success(status)
    elif "conditional" in status.lower():
        st.warning(status)
    else:
        st.error(status)

    st.markdown("### 📌 Reasons")
    

    for r in result["reasons"]:
        st.markdown(f"- {r}")

    st.subheader("📄 Required Documents")
    for doc in result["required_documents"]:
        st.write(f"- {doc}")


    st.markdown("### 💡 Recommendations")
    for rec in result["recommendations"]:
        st.markdown(f"- {rec}")

    st.markdown("### Explanation")
    st.write(result["summary"])
st.markdown("</div>", unsafe_allow_html=True)

