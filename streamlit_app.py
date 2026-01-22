import time
import streamlit as st
from datetime import date

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="SwiftVisa — AI Visa Screening Agent",
    page_icon="🤖",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================
if "step" not in st.session_state:
    st.session_state.step = 1

if "selected_visa" not in st.session_state:
    st.session_state.selected_visa = None

if "result" not in st.session_state:
    st.session_state.result = None

# Chatbot state (ONLY USED IN STEP 1)
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "bot", "text": "Hi 👋 I’m SwiftVisa Assistant. Tell me your purpose (Study/Work/Visit/Healthcare) and I will suggest the right visa."}
    ]

# =========================================================
# UI STYLES
# =========================================================
st.markdown("""
<style>
.stApp{
    background: radial-gradient(circle at 20% 15%, rgba(79,70,229,0.22), transparent 35%),
                radial-gradient(circle at 80% 20%, rgba(34,197,94,0.16), transparent 40%),
                radial-gradient(circle at 60% 85%, rgba(6,182,212,0.20), transparent 35%),
                linear-gradient(180deg, #06101d 0%, #070b12 100%);
}
header[data-testid="stHeader"]{background:transparent;}
footer{visibility:hidden;}

.block-container{
    max-width: 1400px !important;
    padding-top: 0.8rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Prevent column stacking on wide screens */
div[data-testid="stHorizontalBlock"]{
    flex-wrap: nowrap !important;
    gap: 20px !important;
}
div[data-testid="column"]{
    min-width: 380px !important;
}

.title{
    font-size: 54px;
    font-weight: 1000;
    letter-spacing: -1px;
    color: #eaf2ff;
    margin-bottom: 0px;
}
.subtitle{
    font-size: 14px;
    color: rgba(234,242,255,0.72);
    margin-top: 6px;
}

.pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:10px 16px;
    border-radius:999px;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.14);
    color: rgba(234,242,255,0.92);
    font-weight: 900;
    font-size: 12px;
}

.panel{
    border-radius: 22px;
    padding: 18px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 22px 80px rgba(0,0,0,0.55);
    backdrop-filter: blur(16px);
}

/* Visa card */
.visa-card{
    width: 100%;
    border-radius: 22px;
    padding: 18px;
    margin-bottom: 14px;
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow: 0 18px 60px rgba(0,0,0,0.45);
    min-height: 155px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.visa-ico{
    width: 56px;height: 56px;
    display:flex;align-items:center;justify-content:center;
    border-radius: 16px;
    background: rgba(125,211,252,0.16);
    border: 1px solid rgba(125,211,252,0.25);
    font-size: 28px;
}
.visa-title{margin-top: 10px;font-size: 19px;font-weight: 1000;color: #eaf2ff;}
.visa-desc{margin-top: 6px;font-size: 13px;color: rgba(234,242,255,0.72);}

.good{
    border-radius: 18px;
    padding: 14px;
    border: 1px solid rgba(34,197,94,0.35);
    background: linear-gradient(135deg, rgba(34,197,94,0.18), rgba(6,182,212,0.10));
    color: rgba(234,242,255,0.95);
}
.bad{
    border-radius: 18px;
    padding: 14px;
    border: 1px solid rgba(239,68,68,0.35);
    background: linear-gradient(135deg, rgba(239,68,68,0.18), rgba(244,63,94,0.10));
    color: rgba(234,242,255,0.95);
}
.explain{
    margin-top: 10px;
    border-radius: 16px;
    padding: 14px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(0,0,0,0.24);
    color: rgba(234,242,255,0.90);
    white-space: pre-wrap;
    line-height: 1.65;
}

button[kind="secondary"], button[kind="primary"]{
    width: 100% !important;
    border-radius: 16px !important;
    font-weight: 1000 !important;
}
label{
    color: rgba(234,242,255,0.92) !important;
    font-weight: 900 !important;
}
.stTextInput input, .stTextArea textarea, .stNumberInput input, .stDateInput input{
    background: rgba(255,255,255,0.10) !important;
    color: rgba(234,242,255,0.95) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    border-radius: 14px !important;
}

/* Small select button (Step 1) */
div.step1-select div.stButton>button{
    width: 110px !important;
    margin-top: 6px;
    border-radius: 18px !important;
}

/* Floating chatbot button */
.chat-float{
    position: fixed;
    right: 22px;
    bottom: 22px;
    z-index: 9999;
}
.chat-float button{
    background: linear-gradient(135deg, #4f46e5, #22c55e);
    color: #fff;
    border: 0;
    border-radius: 24px;
    padding: 12px 18px;
    font-weight: 1000;
    cursor: pointer;
    box-shadow: 0 16px 40px rgba(0,0,0,0.35);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def yes_no_required(label, key=None):
    return st.radio(label, ["Select", "Yes", "No"], index=0, horizontal=True, key=key)

def normalize_yn(val: str):
    if val == "Yes":
        return True
    if val == "No":
        return False
    return None

def validate_required(fields: dict):
    """Return list of keys missing/invalid."""
    missing = []
    for k, v in fields.items():
        if v is None:
            missing.append(k)
        elif isinstance(v, str) and not v.strip():
            missing.append(k)
        elif v == "Select":
            missing.append(k)
    return missing

def check_visa_eligibility(payload: dict) -> dict:
    """
    Simple rule-based decision using your doc structure.
    ELIGIBLE only when mandatory fields complete + all key Y/N rules are Yes.
    """

    visa_type = payload.get("visa_type_applying_for", "")

    missing_fields = []
    failed = []

    # -------------------------
    # COMMON REQUIRED (from doc)
    # -------------------------
    common_required = {
        "full_name": payload.get("full_name"),
        "date_of_birth": payload.get("date_of_birth"),
        "nationality": payload.get("nationality"),
        "passport_number": payload.get("passport_number"),
        "passport_issue_date": payload.get("passport_issue_date"),
        "passport_expiry_date": payload.get("passport_expiry_date"),
        "country_of_application": payload.get("country_of_application"),
        "purpose_of_visit": payload.get("purpose_of_visit"),
        "intended_travel_start_date": payload.get("intended_travel_start_date"),
        "intended_length_of_stay": payload.get("intended_length_of_stay"),
        "funds_available": payload.get("funds_available"),
        "english_requirement_met_common": payload.get("english_requirement_met_common"),
        "criminal_history_declaration": payload.get("criminal_history_declaration"),
        "previous_uk_visa_refusal": payload.get("previous_uk_visa_refusal"),
        "email": payload.get("email"),
        "phone": payload.get("phone"),
        "current_address": payload.get("current_address"),
    }
    missing_fields += validate_required(common_required)

    # if common missing => stop
    if missing_fields:
        return {
            "status": "INELIGIBLE",
            "missing_fields": missing_fields,
            "failed_rules": [],
            "visa_type": visa_type
        }

    # -------------------------
    # VISA SPECIFIC RULES
    # -------------------------
    # Graduate Visa
    if visa_type == "Graduate visa":
        req = {
            "currently_in_uk": payload.get("currently_in_uk"),
            "current_uk_visa_type": payload.get("current_uk_visa_type"),
            "course_completed": payload.get("course_completed"),
            "course_level_completed": payload.get("course_level_completed"),
            "education_provider_is_licensed": payload.get("education_provider_is_licensed"),
            "provider_reported_completion_to_home_office": payload.get("provider_reported_completion_to_home_office"),
            "original_cas_reference": payload.get("original_cas_reference"),
            "student_visa_valid_on_application_date": payload.get("student_visa_valid_on_application_date"),
        }
        missing_fields += validate_required(req)
        if missing_fields:
            return {"status": "INELIGIBLE", "missing_fields": missing_fields, "failed_rules": [], "visa_type": visa_type}

        if payload["currently_in_uk"] != "Yes": failed.append("Must be in UK")
        if payload["course_completed"] != "Yes": failed.append("Course must be completed")
        if payload["education_provider_is_licensed"] != "Yes": failed.append("Provider must be licensed")
        if payload["provider_reported_completion_to_home_office"] != "Yes": failed.append("Provider must report completion")
        if payload["student_visa_valid_on_application_date"] != "Yes": failed.append("Student/Tier4 visa must be valid")

    # Student Visa
    elif visa_type == "Student visa":
        req = {
            "has_cas": payload.get("has_cas"),
            "cas_reference_number": payload.get("cas_reference_number"),
            "education_provider_is_licensed": payload.get("education_provider_is_licensed"),
            "course_level": payload.get("course_level"),
            "course_full_time": payload.get("course_full_time"),
            "course_start_date": payload.get("course_start_date"),
            "course_end_date": payload.get("course_end_date"),
            "course_duration_months": payload.get("course_duration_months"),
            "meets_financial_requirement": payload.get("meets_financial_requirement"),
            "funds_held_for_28_days": payload.get("funds_held_for_28_days"),
            "english_requirement_met": payload.get("english_requirement_met"),
        }
        missing_fields += validate_required(req)
        if missing_fields:
            return {"status": "INELIGIBLE", "missing_fields": missing_fields, "failed_rules": [], "visa_type": visa_type}

        if payload["has_cas"] != "Yes": failed.append("CAS is required")
        if payload["education_provider_is_licensed"] != "Yes": failed.append("Provider must be licensed")
        if payload["meets_financial_requirement"] != "Yes": failed.append("Financial requirement not met")
        if payload["funds_held_for_28_days"] != "Yes": failed.append("Funds not held for 28 days")
        if payload["english_requirement_met"] != "Yes": failed.append("English requirement not met")

    # Skilled Worker Visa
    elif visa_type == "Skilled worker visa":
        req = {
            "job_offer_confirmed": payload.get("job_offer_confirmed"),
            "employer_is_licensed_sponsor": payload.get("employer_is_licensed_sponsor"),
            "certificate_of_sponsorship_issued": payload.get("certificate_of_sponsorship_issued"),
            "cos_reference_number": payload.get("cos_reference_number"),
            "job_title": payload.get("job_title"),
            "soc_code": payload.get("soc_code"),
            "job_is_eligible_occupation": payload.get("job_is_eligible_occupation"),
            "salary_offered": payload.get("salary_offered"),
            "meets_minimum_salary_threshold": payload.get("meets_minimum_salary_threshold"),
            "english_requirement_met": payload.get("english_requirement_met"),
            "criminal_record_certificate_required": payload.get("criminal_record_certificate_required"),
            "criminal_record_certificate_provided": payload.get("criminal_record_certificate_provided"),
        }
        missing_fields += validate_required(req)
        if missing_fields:
            return {"status": "INELIGIBLE", "missing_fields": missing_fields, "failed_rules": [], "visa_type": visa_type}

        if payload["job_offer_confirmed"] != "Yes": failed.append("Job offer not confirmed")
        if payload["employer_is_licensed_sponsor"] != "Yes": failed.append("Employer must be licensed sponsor")
        if payload["certificate_of_sponsorship_issued"] != "Yes": failed.append("CoS must be issued")
        if payload["job_is_eligible_occupation"] != "Yes": failed.append("Occupation not eligible")
        if payload["meets_minimum_salary_threshold"] != "Yes": failed.append("Salary threshold not met")
        if payload["english_requirement_met"] != "Yes": failed.append("English not met")

        # criminal cert logic
        if payload["criminal_record_certificate_required"] == "Yes" and payload["criminal_record_certificate_provided"] != "Yes":
            failed.append("Criminal record certificate required but not provided")

    # Health & Care Visa
    elif visa_type == "Health and care visa":
        req = {
            "job_offer_confirmed": payload.get("job_offer_confirmed"),
            "employer_is_licensed_healthcare_sponsor": payload.get("employer_is_licensed_healthcare_sponsor"),
            "certificate_of_sponsorship_issued": payload.get("certificate_of_sponsorship_issued"),
            "cos_reference_number": payload.get("cos_reference_number"),
            "job_title": payload.get("job_title"),
            "soc_code": payload.get("soc_code"),
            "job_is_eligible_healthcare_role": payload.get("job_is_eligible_healthcare_role"),
            "salary_offered": payload.get("salary_offered"),
            "meets_healthcare_salary_rules": payload.get("meets_healthcare_salary_rules"),
            "professional_registration_required": payload.get("professional_registration_required"),
            "professional_registration_provided": payload.get("professional_registration_provided"),
            "english_requirement_met": payload.get("english_requirement_met"),
        }
        missing_fields += validate_required(req)
        if missing_fields:
            return {"status": "INELIGIBLE", "missing_fields": missing_fields, "failed_rules": [], "visa_type": visa_type}

        if payload["job_offer_confirmed"] != "Yes": failed.append("Job offer not confirmed")
        if payload["employer_is_licensed_healthcare_sponsor"] != "Yes": failed.append("Healthcare employer must be licensed sponsor")
        if payload["certificate_of_sponsorship_issued"] != "Yes": failed.append("CoS must be issued")
        if payload["job_is_eligible_healthcare_role"] != "Yes": failed.append("Role not eligible")
        if payload["meets_healthcare_salary_rules"] != "Yes": failed.append("Healthcare salary rules not met")
        if payload["english_requirement_met"] != "Yes": failed.append("English not met")

        if payload["professional_registration_required"] == "Yes" and payload["professional_registration_provided"] != "Yes":
            failed.append("Professional registration required but not provided")

    # Visitor Visa
    elif visa_type == "Visitor visa":
        req = {
            "purpose_of_visit_specific": payload.get("purpose_of_visit_specific"),
            "purpose_is_permitted_under_visitor_rules": payload.get("purpose_is_permitted_under_visitor_rules"),
            "intended_length_of_stay_months": payload.get("intended_length_of_stay_months"),
            "stay_within_6_months_limit": payload.get("stay_within_6_months_limit"),
            "accommodation_arranged": payload.get("accommodation_arranged"),
            "return_or_onward_travel_planned": payload.get("return_or_onward_travel_planned"),
            "intends_to_leave_uk_after_visit": payload.get("intends_to_leave_uk_after_visit"),
            "sufficient_funds_for_stay": payload.get("sufficient_funds_for_stay"),
        }
        missing_fields += validate_required(req)
        if missing_fields:
            return {"status": "INELIGIBLE", "missing_fields": missing_fields, "failed_rules": [], "visa_type": visa_type}

        if payload["purpose_is_permitted_under_visitor_rules"] != "Yes": failed.append("Purpose not permitted")
        if payload["stay_within_6_months_limit"] != "Yes": failed.append("Stay exceeds 6 months")
        if payload["intends_to_leave_uk_after_visit"] != "Yes": failed.append("Must intend to leave UK after visit")
        if payload["sufficient_funds_for_stay"] != "Yes": failed.append("Insufficient funds")

    # final decision
    if failed:
        return {"status": "INELIGIBLE", "missing_fields": [], "failed_rules": failed, "visa_type": visa_type}

    return {"status": "ELIGIBLE", "missing_fields": [], "failed_rules": [], "visa_type": visa_type}


# =========================================================
# CHATBOT (ONLY STEP 1)
# =========================================================
def chatbot_reply(user_msg: str) -> str:
    msg = user_msg.lower()

    if any(k in msg for k in ["study", "student", "college", "university", "masters", "ms"]):
        return "✅ You should select **Student Visa** (for studies)."
    if any(k in msg for k in ["post study", "graduate", "psw"]):
        return "✅ You should select **Graduate Visa** (post-study work)."
    if any(k in msg for k in ["job", "work", "employment", "sponsor", "skilled worker"]):
        return "✅ You should select **Skilled Worker Visa** (job + sponsor)."
    if any(k in msg for k in ["doctor", "nurse", "health", "care", "nhs", "hospital"]):
        return "✅ You should select **Health & Care Visa** (healthcare sponsored)."
    if any(k in msg for k in ["travel", "tour", "visit", "holiday", "trip"]):
        return "✅ You should select **Visitor Visa** (tourism / short visit)."

    return "Tell me your purpose: **Study / Work / Visit / Healthcare** and I will suggest the correct visa ✅"


# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="title">SwiftVisa — AI Visa Screening Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Interactive UK visa eligibility screening → Clear decision + reasons.</div>', unsafe_allow_html=True)
st.write("")

p1, p2, p3, p4 = st.columns([1.2, 1.2, 1.5, 6])
with p1:
    st.markdown('<div class="pill">🟢 Mandatory clarity</div>', unsafe_allow_html=True)
with p2:
    st.markdown('<div class="pill">🧠 Eligibility decision</div>', unsafe_allow_html=True)
with p3:
    st.markdown('<div class="pill">📌 Reasons</div>', unsafe_allow_html=True)


# =========================================================
# STEP 1 — VISA SELECTION (CHANGED ONLY THIS PART)
# =========================================================
if st.session_state.step == 1:

    st.markdown('<span class="pill">Step 1 — Select Visa Type</span>', unsafe_allow_html=True)
    st.write("")

    visas = [
        ("Student Visa", "🎓", "University CAS + sponsor checks"),
        ("Graduate Visa", "🎓", "Post-study visa eligibility rules"),
        ("Skilled Worker Visa", "🧳", "Job sponsorship and salary rules"),
        ("Health & Care Visa", "🩺", "Healthcare sponsor eligibility"),
        ("Visitor Visa", "✈️", "Short visit conditions"),
    ]

    # Cards in ONE ROW like your image
    cols = st.columns(5, gap="large")

    for i, (title, icon, desc) in enumerate(visas):
        with cols[i]:
            st.markdown(f"""
            <div class="visa-card">
                <div>
                    <div class="visa-ico">{icon}</div>
                    <div class="visa-title">{title}</div>
                    <div class="visa-desc">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Select button BELOW the card (like image)
            st.markdown('<div class="step1-select">', unsafe_allow_html=True)
            if st.button("Select", key=f"select_{title}"):
                st.session_state.selected_visa = title
                st.session_state.step = 2
                st.session_state.result = None
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Chatbot button
    st.write("")
    btn1, btn2 = st.columns([1.2, 8])
    with btn1:
        if st.button("💬 Chatbot Suggestion"):
            st.session_state.chat_open = True
    with btn2:
        st.caption("Ask in human language like: *Which visa is best for me if I want to study in UK?*")

    # Chatbot box
    if st.session_state.chat_open:
        with st.expander("💬 SwiftVisa Chatbot (Suggestion)", expanded=True):
            for m in st.session_state.chat_history:
                if m["role"] == "bot":
                    st.markdown(f"**🤖 Bot:** {m['text']}")
                else:
                    st.markdown(f"**🧑 You:** {m['text']}")

            user_msg = st.text_input("Type here...", key="chat_input")
            if st.button("Send", key="send_chat"):
                if user_msg.strip():
                    st.session_state.chat_history.append({"role": "user", "text": user_msg})
                    st.session_state.chat_history.append({"role": "bot", "text": chatbot_reply(user_msg)})
                    st.rerun()


# =========================================================
# STEP 2 — INPUTS + DECISION (DO NOT CHANGE ANYTHING)
# =========================================================
if st.session_state.step == 2:
    visa_map = {
        "Student Visa": "Student visa",
        "Graduate Visa": "Graduate visa",
        "Skilled Worker Visa": "Skilled worker visa",
        "Health & Care Visa": "Health and care visa",
        "Visitor Visa": "Visitor visa",
    }
    visa_type = visa_map[st.session_state.selected_visa]

    st.markdown(f'<span class="pill">Step 2 — Enter Inputs • Visa: {st.session_state.selected_visa}</span>',
                unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([1.55, 1], gap="large")

    # ----------------------
    # LEFT: INPUTS
    # ----------------------
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown("### ✅ 1) Common Inputs (Mandatory)")
        c1, c2 = st.columns(2)
        with c1:
            full_name = st.text_input("Full Name (as per passport)", placeholder="e.g., Abhi")
            date_of_birth = st.date_input("Date of Birth", value=date(2000, 1, 1))
            nationality = st.text_input("Nationality", placeholder="e.g., Indian")
            passport_number = st.text_input("Passport Number", placeholder="e.g., N1234567")
            passport_issue_date = st.date_input("Passport Issue Date", value=date(2020, 1, 1))
            passport_expiry_date = st.date_input("Passport Expiry Date", value=date(2030, 1, 1))
            current_address = st.text_area("Current Address", placeholder="Your full current address...", height=80)

        with c2:
            email = st.text_input("Email Address", placeholder="e.g., abhi@gmail.com")
            phone = st.text_input("Phone Number", placeholder="+91 XXXXX XXXXX")
            country_of_application = st.text_input("Country of Application / Current Location", placeholder="e.g., India")
            purpose_of_visit = st.text_input("Purpose of Visit", placeholder="e.g., Study / Work / Visit")
            intended_travel_start_date = st.date_input("Intended Travel / Start Date", value=date.today())
            intended_length_of_stay = st.text_input("Intended Length of Stay", placeholder="e.g., 2 years")
            funds_available = st.number_input("Funds Available (GBP)", min_value=0.0, step=100.0)

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            english_requirement_met_common = yes_no_required("English Language Requirement Met?")
        with cc2:
            criminal_history_declaration = yes_no_required("Criminal History Declaration?")
        with cc3:
            previous_uk_visa_refusal = yes_no_required("Previous UK Visa Refusal?")

        st.write("")
        st.markdown("### 📌 2) Visa-Specific Inputs (Mandatory)")

        visa_inputs = {"visa_type_applying_for": visa_type}

        # ========== Graduate Visa
        if visa_type == "Graduate visa":
            a, b = st.columns(2)
            with a:
                visa_inputs["currently_in_uk"] = yes_no_required("Currently in UK?")
                visa_inputs["current_uk_visa_type"] = st.selectbox("Current UK visa type", ["Select", "Student", "Tier 4", "Other"], index=0)
                visa_inputs["course_completed"] = yes_no_required("Course completed?")
                visa_inputs["course_level_completed"] = st.text_input("Course level completed", placeholder="e.g., Masters")
            with b:
                visa_inputs["education_provider_is_licensed"] = yes_no_required("Education provider licensed?")
                visa_inputs["provider_reported_completion_to_home_office"] = yes_no_required("Provider reported completion to Home Office?")
                visa_inputs["original_cas_reference"] = st.text_input("Original CAS reference", placeholder="CAS123...")
                visa_inputs["student_visa_valid_on_application_date"] = yes_no_required("Student/Tier4 visa valid on application date?")

        # ========== Student Visa
        elif visa_type == "Student visa":
            a, b = st.columns(2)
            with a:
                visa_inputs["has_cas"] = yes_no_required("Has CAS?")
                visa_inputs["cas_reference_number"] = st.text_input("CAS reference number", placeholder="CAS123...")
                visa_inputs["education_provider_is_licensed"] = yes_no_required("Education provider licensed?")
                visa_inputs["course_level"] = st.text_input("Course level", placeholder="e.g., UG/PG")
                visa_inputs["course_full_time"] = yes_no_required("Course full-time?")
            with b:
                visa_inputs["course_start_date"] = st.date_input("Course start date", value=date.today())
                visa_inputs["course_end_date"] = st.date_input("Course end date", value=date.today())
                visa_inputs["course_duration_months"] = st.number_input("Course duration (months)", min_value=1, step=1)
                visa_inputs["meets_financial_requirement"] = yes_no_required("Meets financial requirement?")
                visa_inputs["funds_held_for_28_days"] = yes_no_required("Funds held for 28 days?")
                visa_inputs["english_requirement_met"] = yes_no_required("English requirement met?")

        # ========== Skilled Worker Visa
        elif visa_type == "Skilled worker visa":
            a, b = st.columns(2)
            with a:
                visa_inputs["job_offer_confirmed"] = yes_no_required("Job offer confirmed?")
                visa_inputs["employer_is_licensed_sponsor"] = yes_no_required("Employer is licensed sponsor?")
                visa_inputs["certificate_of_sponsorship_issued"] = yes_no_required("Certificate of Sponsorship issued?")
                visa_inputs["cos_reference_number"] = st.text_input("CoS reference number", placeholder="COS123...")
                visa_inputs["job_title"] = st.text_input("Job title", placeholder="e.g., Data Analyst")
                visa_inputs["soc_code"] = st.text_input("SOC code", placeholder="e.g., 2135")
            with b:
                visa_inputs["job_is_eligible_occupation"] = yes_no_required("Job is eligible occupation?")
                visa_inputs["salary_offered"] = st.number_input("Salary offered (GBP)", min_value=0.0, step=500.0)
                visa_inputs["meets_minimum_salary_threshold"] = yes_no_required("Meets minimum salary threshold?")
                visa_inputs["english_requirement_met"] = yes_no_required("English requirement met?")
                visa_inputs["criminal_record_certificate_required"] = yes_no_required("Criminal record certificate required?")
                visa_inputs["criminal_record_certificate_provided"] = yes_no_required("Criminal record certificate provided?")

        # ========== Health & Care Visa
        elif visa_type == "Health and care visa":
            a, b = st.columns(2)
            with a:
                visa_inputs["job_offer_confirmed"] = yes_no_required("Job offer confirmed?")
                visa_inputs["employer_is_licensed_healthcare_sponsor"] = yes_no_required("Employer is licensed healthcare sponsor?")
                visa_inputs["certificate_of_sponsorship_issued"] = yes_no_required("Certificate of Sponsorship issued?")
                visa_inputs["cos_reference_number"] = st.text_input("CoS reference number", placeholder="COS123...")
                visa_inputs["job_title"] = st.text_input("Job title", placeholder="e.g., Nurse")
                visa_inputs["soc_code"] = st.text_input("SOC code", placeholder="e.g., 2231")
            with b:
                visa_inputs["job_is_eligible_healthcare_role"] = yes_no_required("Job is eligible healthcare role?")
                visa_inputs["salary_offered"] = st.number_input("Salary offered (GBP)", min_value=0.0, step=500.0)
                visa_inputs["meets_healthcare_salary_rules"] = yes_no_required("Meets healthcare salary rules?")
                visa_inputs["professional_registration_required"] = yes_no_required("Professional registration required?")
                visa_inputs["professional_registration_provided"] = yes_no_required("Professional registration provided?")
                visa_inputs["english_requirement_met"] = yes_no_required("English requirement met?")

        # ========== Visitor Visa
        elif visa_type == "Visitor visa":
            a, b = st.columns(2)
            with a:
                visa_inputs["purpose_of_visit_specific"] = st.text_input("Purpose of visit (visitor)", placeholder="Tourism / Family / Business")
                visa_inputs["purpose_is_permitted_under_visitor_rules"] = yes_no_required("Purpose permitted under visitor rules?")
                visa_inputs["intended_length_of_stay_months"] = st.number_input("Intended length of stay (months)", min_value=1, step=1)
                visa_inputs["stay_within_6_months_limit"] = yes_no_required("Stay within 6 months limit?")
            with b:
                visa_inputs["accommodation_arranged"] = yes_no_required("Accommodation arranged?")
                visa_inputs["return_or_onward_travel_planned"] = yes_no_required("Return or onward travel planned?")
                visa_inputs["intends_to_leave_uk_after_visit"] = yes_no_required("Intends to leave UK after visit?")
                visa_inputs["sufficient_funds_for_stay"] = yes_no_required("Sufficient funds for stay?")

        st.write("")
        colA, colB = st.columns(2)
        with colA:
            if st.button("⬅ Back to visa selection"):
                st.session_state.step = 1
                st.session_state.selected_visa = None
                st.session_state.result = None
                st.rerun()

        with colB:
            run_btn = st.button("🤖 Run Screening", type="primary")

        st.markdown("</div>", unsafe_allow_html=True)

    # ----------------------
    # RIGHT: DECISION OUTPUT
    # ----------------------
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        st.markdown("""
        <div style="border-radius:22px;padding:16px;background:rgba(255,255,255,0.08);
                    border:1px solid rgba(255,255,255,0.14);text-align:center;">
            <div style="font-size:66px;">🤖</div>
            <div style="font-weight:1000;color:#eaf2ff;font-size:15px;">SwiftBot Eligibility Unit</div>
            <div style="margin-top:4px;color:rgba(234,242,255,0.70);font-size:13px;">
                Monitoring your inputs & generating decision...
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("### ✅ Decision Output")

        if run_btn:
            payload = {
                "visa_type_applying_for": visa_type,

                # common
                "full_name": full_name,
                "date_of_birth": date_of_birth,
                "nationality": nationality,
                "passport_number": passport_number,
                "passport_issue_date": passport_issue_date,
                "passport_expiry_date": passport_expiry_date,
                "country_of_application": country_of_application,
                "purpose_of_visit": purpose_of_visit,
                "intended_travel_start_date": intended_travel_start_date,
                "intended_length_of_stay": intended_length_of_stay,
                "funds_available": funds_available,
                "english_requirement_met_common": english_requirement_met_common,
                "criminal_history_declaration": criminal_history_declaration,
                "previous_uk_visa_refusal": previous_uk_visa_refusal,
                "email": email,
                "phone": phone,
                "current_address": current_address,

                # visa inputs
                **visa_inputs
            }

            prog = st.progress(0)
            for pct, msg in [(20, "Scanning inputs..."), (55, "Validating rules..."), (80, "Generating decision..."), (100, "Complete")]:
                time.sleep(0.35)
                prog.progress(pct)
                st.caption(msg)

            st.session_state.result = check_visa_eligibility(payload)

        if st.session_state.result:
            r = st.session_state.result

            if r["status"] == "ELIGIBLE":
                st.markdown(f"""
                <div class="good">
                    ✅ <b>ELIGIBLE</b><br>
                    Visa Type: <b>{visa_type}</b><br><br>
                    Access Granted ✅
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<div class='explain'>All required eligibility conditions are satisfied based on your inputs.</div>", unsafe_allow_html=True)

            else:
                st.markdown(f"""
                <div class="bad">
                    ❌ <b>INELIGIBLE</b><br>
                    Visa Type: <b>{visa_type}</b><br><br>
                    Access Denied ❌
                </div>
                """, unsafe_allow_html=True)

                if r.get("missing_fields"):
                    miss_txt = "\n".join([f"• {m.replace('_',' ').title()}" for m in r["missing_fields"]])
                    st.markdown(f"<div class='explain'>⚠️ Missing mandatory inputs:\n{miss_txt}</div>", unsafe_allow_html=True)
                else:
                    fail_txt = "\n".join([f"• {x}" for x in r.get("failed_rules", [])])
                    st.markdown(f"<div class='explain'>❌ Failed eligibility conditions:\n{fail_txt}</div>", unsafe_allow_html=True)

        else:
            st.markdown("<div class='explain'>SYSTEM IDLE ⏳\nWaiting for screening request…</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
