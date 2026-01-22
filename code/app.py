import streamlit as st
from llm_part import answer_question
from datetime import date

# -----------------------------------
# 1. CORE LOGIC (100% MATCH TO YOUR ORIGINAL)
# -----------------------------------
if "current_phase" not in st.session_state:
    st.session_state.current_phase = 0

def back_button_logic(target_phase: int):
    """Custom styled back button to maintain UI consistency"""
    if st.button("⬅️ Back", key=f"back_btn_{target_phase}"):
        st.session_state.current_phase = target_phase
        st.rerun()

def build_backend_question():
    ce = st.session_state.common_entities
    base_context = f"""
Visa Type: {st.session_state.visa_type}
Country: {st.session_state.country}

Applicant Profile:
Full Name: {ce["full_name"]}
Date of Birth: {ce["date_of_birth"]}
Nationality: {ce["nationality"]}
Passport Number: {ce["passport_number"]}
Passport Issue Date: {ce["passport_issue_date"]}
Passport Expiry Date: {ce["passport_expiry_date"]}

Purpose of Visit: {ce["purpose_of_visit"]}
Intended Travel Date: {ce["intended_travel_date"]}
Intended Length of Stay: {ce["intended_length_of_stay"]}
Funds Available: {ce["funds_available"]}
English Requirement Met: {ce["english_requirement_met"]}
Criminal History: {ce["criminal_history"]}
Previous UK Visa Refusal: {ce["previous_uk_refusal"]}

Contact Details:
Email: {ce["email_address"]}
Phone: {ce["phone_number"]}
Address: {ce["current_address"]}
"""
    base_context += "\nVisa-Specific Information:\n"
    for k, v in st.session_state.visa_specific_data.items():
        base_context += f"{k}: {v}\n"
    return base_context.strip()

def derive_eligibility_verdict(llm_response: str) -> str:
    response_lower = llm_response.lower()
    if "missing informations" in response_lower:
        return "Depends"
    negative_signals = ["not eligible", "must be", "required to", "cannot apply", "does not meet", "is not allowed"]
    for signal in negative_signals:
        if signal in response_lower:
            return "Not Eligible"
    return "Eligible"

# -----------------------------------
# 2. HIGH-VISIBILITY LIGHT BLUE THEME
# -----------------------------------
st.set_page_config(page_title="SwiftVisa Portal", page_icon="🛂", layout="wide")

st.markdown("""
    <style>
    /* 1. Main Background: Light Blue */
    .stApp {
        background-color: #eef6ff !important;
    }

    /* 2. Force Headers and Streamlit UI text to Black */
    h1, h2, h3, p, span, .stMarkdown, label {
        color: #000000 !important;
    }

    /* 3. Fix the Top Header (Deploy/3 dots) */
    header[data-testid="stHeader"] {
        background-color: #cce3ff !important;
    }

    /* 4. Fix Selectboxes / Dropdowns: White BG, Black Text */
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important;
    }
    div[role="listbox"] {
        background-color: white !important;
        color: black !important;
    }
    input {
        background-color: white !important;
        color: black !important;
    }

    /* 5. Sidebar: Darker Blue with Black Text */
    section[data-testid="stSidebar"] {
        background-color: #d1e3ff !important;
    }
    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    /* 6. White Cards for Content */
    .main-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #accbff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }

    /* 7. Buttons */
    .stButton > button {
        background-color: #0052cc !important;
        color: white !important;
        font-weight: bold;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
phases = ["Start", "Identity", "Visa Choice", "Requirements", "Review", "Result"]
# -----------------------------------
# THE "HEADER BAR" FIX (Writing in the blank space)
# -----------------------------------
st.markdown(f"""
    <div style="background-color: #0052cc; padding: 8px; border-radius: 0px 0px 10px 10px; text-align: center; margin-top: -65px; margin-bottom: 25px;">
        <span style="color: white; font-weight: bold;">
            🛂 SWIFTVISA AI SECURE PORTAL &nbsp; | &nbsp; {phases[st.session_state.current_phase].upper()}
        </span>
    </div>
""", unsafe_allow_html=True)
# -----------------------------------
# 3. SIDEBAR NAVIGATION
# -----------------------------------
with st.sidebar:
    st.title("SwiftVisa")
    st.markdown("---")
    phases = ["Start", "Identity", "Visa Choice", "Requirements", "Review", "Result"]
    st.progress(st.session_state.current_phase / 5)
    for i, p in enumerate(phases):
        if i == st.session_state.current_phase: st.markdown(f"**🔵 {p}**")
        elif i < st.session_state.current_phase: st.markdown(f"✅ {p}")
        else: st.markdown(f"⚪ {p}")

# -----------------------------------
# PHASE 0: START
# -----------------------------------
if st.session_state.current_phase == 0:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.title("Visa Eligibility Assistant")
    intent = st.radio("Select assessment type:*", ["Quick eligibility check", "Detailed eligibility analysis"])
    if st.button("Start Now"):
        st.session_state.intent = intent
        st.session_state.current_phase = 1
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# PHASE 1: IDENTITY (COMPULSORY)
# -----------------------------------

if st.session_state.current_phase == 1:
    back_button_logic(0)
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("1. Applicant Core Profile")
    with st.form("identity"):
        c1, c2 = st.columns(2)
        with c1:
            # Note: I am using 'fn', 'nt', etc. to match your 'all()' check below
            fn = st.text_input("Full Name (as per passport)*")
            dob = st.date_input("Date of Birth*")
            nt = st.text_input("Nationality*")
            pn = st.text_input("Passport Number*")
            pi = st.date_input("Passport Issue Date*")
            pe = st.date_input("Passport Expiry Date*")
        with c2:
            pv = st.selectbox("Purpose of Visit*", ["Study", "Work", "Healthcare Employment", "Tourism", "Business", "Family Visit"])
            td = st.date_input("Intended Start Date*")
            sl = st.selectbox("Intended Stay*", ["Less than 6 months", "6–12 months", "1–2 years", "More than 2 years"])
            fa = st.number_input("Funds Available (GBP)*", min_value=0)
            em = st.selectbox("English Met?*", ["Yes", "No"])
        
        st.markdown("---")
        c3, c4 = st.columns(2)
        with c3:
            crim = st.selectbox("Criminal History?*", ["No", "Yes"])
            refu = st.selectbox("Previous UK Refusal?*", ["No", "Yes"])
            email = st.text_input("Email Address*")
        with c4:
            phone = st.text_input("Phone Number*")
            addr = st.text_area("Current Address*")

        if st.form_submit_button("Continue"):
            # This check now matches the variables above exactly
            if not all([fn, nt, pn, email, phone, addr]):
                st.error("Error: All fields marked with * must be filled.")
            else:
                st.session_state.common_entities = {
                    "full_name": fn, "date_of_birth": dob, "nationality": nt,
                    "passport_number": pn, "passport_issue_date": pi, "passport_expiry_date": pe,
                    "purpose_of_visit": pv, "intended_travel_date": td, "intended_length_of_stay": sl,
                    "funds_available": fa, "english_requirement_met": em, "criminal_history": crim,
                    "previous_uk_refusal": refu, "email_address": email, "phone_number": phone, "current_address": addr
                }
                st.session_state.current_phase = 2
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# PHASE 2: VISA CHOICE
# -----------------------------------
if st.session_state.current_phase == 2:
    back_button_logic(1)
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("Destination Selection")
    st.session_state.country = st.selectbox("Target Country*", ["United Kingdom", "US", "INDIA", "CANADA"])
    st.session_state.visa_type = st.selectbox("Visa Type*", ["Skilled Worker Visa", "Student Visa", "Health & Care Worker Visa", "Graduate Visa", "Visitor Visa"])
    if st.button("Proceed"):
        st.session_state.current_phase = 3
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# PHASE 3: VISA SPECIFIC (600 LINE LOGIC RESTORED)
# -----------------------------------
if st.session_state.current_phase == 3:
    back_button_logic(2)
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    v = st.session_state.visa_type
    st.header(f"{v} Requirements")
    with st.form("v_form"):
        d = {}
        if v == "Graduate Visa":
            d["currently_in_uk"] = st.selectbox("In UK?*", ["Yes", "No"])
            d["course_completed"] = st.selectbox("Course Completed?*", ["Yes", "No"])
            d["course_level"] = st.text_input("Course Level*")
            d["cas"] = st.text_input("Original CAS*")
        elif v == "Student Visa":
            d["has_cas"] = st.selectbox("Has CAS?*", ["Yes", "No"])
            d["cas_ref"] = st.text_input("CAS Ref*") if d["has_cas"]=="Yes" else "N/A"
            d["course_level"] = st.selectbox("Course Level*", ["Bachelor", "Master", "PhD"])
            d["course_full_time"] = st.selectbox("Course Full-Time?", ["Yes", "No"])
            d["course_start_date"] = st.date_input("Course Start Date")
            d["course_end_date"] = st.date_input("Course End Date")
            d["course_duration_months"] = st.number_input(
                "Course Duration (months)", min_value=1
            )
            d["funds_28_days"] = st.selectbox("Funds held for 28 days?*", ["Yes", "No"])
        elif v == "Skilled Worker Visa":
            d["job_offer"] = st.selectbox("Job Offer Confirmed?*", ["Yes", "No"])
            d["cos_issued"] = st.selectbox("CoS Issued?*", ["Yes", "No"])
            d["job_title"] = st.text_input("Job Title*")
            d["salary"] = st.number_input("Salary (£)*", min_value=0)
            d["soc"] = st.text_input("SOC Code*")
        elif v == "Health & Care Worker Visa":
            d["job_offer"] = st.selectbox("Had Job Offer?*", ["Yes", "No"])
            d["sponsor"] = st.selectbox("Licensed Sponsor?*", ["Yes", "No"])
            d["salary"] = st.number_input("Salary (£)*", min_value=0)
            d["role"] = st.text_input("Job Role*")
        elif v == "Visitor Visa":
            d["purpose"] = st.text_input("Detail Purpose*")
            d["duration"] = st.number_input("Stay (Months)*", min_value=1)
            d["leave_uk"] = st.selectbox("Intend to leave?*", ["Yes", "No"])
            d["accommodation_arranged"] = st.selectbox(
                "Accommodation Arranged?", ["Yes", "No"]
            )

        if st.form_submit_button("Continue"):
            st.session_state.visa_specific_data = d
            st.session_state.current_phase = 4
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# PHASE 4: REVIEW
# -----------------------------------
if st.session_state.current_phase == 4:
    back_button_logic(3)
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.header("Review & Submit")
    if st.session_state.intent == "Detailed eligibility analysis":
        st.session_state.user_question = st.text_area("Specific Questions for AI*", height=150)
    if st.button("Generate Result"):
        st.session_state.current_phase = 5
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# PHASE 5: RESULT
# -----------------------------------
if st.session_state.current_phase == 5:
    back_button_logic(4)
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    with st.spinner("Analyzing..."):
        if "backend_response" not in st.session_state:
            q = build_backend_question()
            if st.session_state.intent != "Quick eligibility check":
                q += f"\nUser Question:\n{st.session_state.user_question}"
            st.session_state.backend_response = answer_question(q)
        
        res = st.session_state.backend_response
        verd = derive_eligibility_verdict(res)

    if verd == "Eligible": st.success("✅ ELIGIBLE")
    elif verd == "Not Eligible": st.error("❌ NOT ELIGIBLE")
    else: st.warning("⚠️ DEPENDS ON DETAILS")

    st.write(res)
    if st.button("New Check"):
        st.session_state.clear()
        st.session_state.current_phase = 0
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)