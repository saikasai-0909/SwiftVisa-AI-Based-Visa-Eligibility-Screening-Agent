import streamlit as st
from datetime import date

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(
    page_title="UK Visa Eligibility System",
    layout="wide"
)

# ================================
# SESSION STATE
# ================================
if "page" not in st.session_state:
    st.session_state.page = 1


st.markdown("""
<style>
.header-box {
    background-color: #0b1f3a;
    padding: 20px;
    border-radius: 8px;
    color: white;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="header-box">
    <img src="C:/Users/MOUNIKA RAMYA/Downloads/UK_Government_Overseas_Logo.svg" width="120">
    <h2>UK Visa Eligibility Screening Agent</h2>
    <p>Official pre-assessment system for UK visa applications</p>
</div>
""", unsafe_allow_html=True)


# ================================
# PAGE 1 – WELCOME
# ================================
if st.session_state.page == 1:
    st.markdown("""
    <h3 style="text-align:center;">AI Powered Immigration Assessment System</h3>

    <p style="font-size:18px;">
    This system helps determine whether an applicant is eligible for a UK visa based on
    official UK immigration rules.
    </p>

    <p>
    ✔ Validates applicant data<br>
    ✔ Applies visa eligibility rules<br>
    ✔ Gives clear rejection reasons<br>
    ✔ Provides AI-based recommendations
    </p>
    """, unsafe_allow_html=True)

    if st.button("➡️ Start Application"):
        st.session_state.page = 2
        st.rerun()

# ================================
# PAGE 2 – COMMON DETAILS
# ================================
elif st.session_state.page == 2:

    st.header("Common Applicant Details")

    col1, col2 = st.columns(2)

    with col1:
        full_name = st.text_input("Full Name *")
        dob = st.date_input("Date of Birth *", min_value=date(1930,1,1))
        nationality = st.text_input("Nationality")
        passport_number = st.text_input("Passport Number *")
        passport_issue = st.date_input("Passport Issue Date *", min_value=date(1930,1,1))
        passport_expiry = st.date_input("Passport Expiry Date *", min_value=date.today())

    with col2:
        country = st.text_input("Country of Application *")
        visa_type = st.selectbox(
            "Visa Type *",
            ["-", "Student Visa", "Graduate Visa", "Skilled Worker Visa",
             "Health and Care Visa", "Visitor Visa"]
        )
        purpose = st.text_input("Purpose of Visit")
        travel_date = st.date_input("Intended Travel Date")
        stay_months = st.number_input("Length of Stay (months)", min_value=1)
        funds = st.number_input("Funds Available (£)", min_value=0)

        english = st.selectbox("English Requirement Met *", ["-", "Yes", "No"])
        criminal = st.selectbox("Criminal History *", ["-", "Yes", "No"])
        refusal = st.selectbox("Previous UK Visa Refusal *", ["-", "Yes", "No"])
        phone = st.text_input("Phone Number *")

    if st.button("Save & Continue"):
        st.session_state.common = locals()
        st.session_state.page = 3
        st.rerun()

# ================================
# PAGE 3 – VISA SPECIFIC DETAILS
# ================================
elif st.session_state.page == 3:

    visa = st.session_state.common["visa_type"]

    st.header(f"{visa} – Eligibility Details")

    data = {}

    if visa == "Student Visa":
        data["has_cas"] = st.selectbox("Has CAS *", ["-", "Yes", "No"])
        data["provider_licensed"] = st.selectbox("Education Provider Licensed *", ["-", "Yes", "No"])
        data["course_full_time"] = st.selectbox("Course Full Time *", ["-", "Yes", "No"])
        data["financial_met"] = st.selectbox("Financial Requirement Met *", ["-", "Yes", "No"])
        data["funds_28"] = st.selectbox("Funds Held for 28 Days *", ["-", "Yes", "No"])
        data["english"] = st.selectbox("English Requirement Met *", ["-", "Yes", "No"])

    elif visa == "Graduate Visa":
        data["in_uk"] = st.selectbox("Currently in UK *", ["-", "Yes", "No"])
        data["visa_type"] = st.selectbox("Current Visa Type *", ["-", "Student", "Tier 4"])
        data["completed"] = st.selectbox("Course Completed *", ["-", "Yes", "No"])
        data["provider_reported"] = st.selectbox("Provider Reported Completion *", ["-", "Yes", "No"])

    elif visa == "Skilled Worker Visa":
        data["job_offer"] = st.selectbox("Job Offer Confirmed *", ["-", "Yes", "No"])
        data["sponsor"] = st.selectbox("Licensed Sponsor *", ["-", "Yes", "No"])
        data["cos"] = st.selectbox("CoS Issued *", ["-", "Yes", "No"])
        data["salary_met"] = st.selectbox("Salary Requirement Met *", ["-", "Yes", "No"])
        data["english"] = st.selectbox("English Requirement Met *", ["-", "Yes", "No"])

    elif visa == "Health and Care Visa":
        data["job_offer"] = st.selectbox("Job Offer Confirmed *", ["-", "Yes", "No"])
        data["sponsor"] = st.selectbox("Healthcare Sponsor *", ["-", "Yes", "No"])
        data["cos"] = st.selectbox("CoS Issued *", ["-", "Yes", "No"])
        data["registration"] = st.selectbox("Professional Registration *", ["-", "Yes", "No"])

    elif visa == "Visitor Visa":
        data["purpose_ok"] = st.selectbox("Purpose Allowed *", ["-", "Yes", "No"])
        data["stay_ok"] = st.selectbox("Stay Within 6 Months *", ["-", "Yes", "No"])
        data["funds_ok"] = st.selectbox("Sufficient Funds *", ["-", "Yes", "No"])
        data["return"] = st.selectbox("Return Ticket *", ["-", "Yes", "No"])

    if st.button("Check Eligibility"):
        st.session_state.visa_data = data
        st.session_state.page = 4
        st.rerun()

# ================================
# PAGE 4 – DECISION ENGINE
# ================================
elif st.session_state.page == 4:

    reasons = []
    c = st.session_state.common
    v = st.session_state.visa_data

    # ---- Common Checks ----
    if c["english"] == "No":
        reasons.append("English language requirement not met.")

    if c["criminal"] == "Yes":
        reasons.append("Criminal history affects eligibility.")

    if c["refusal"] == "Yes":
        reasons.append("Previous visa refusal impacts approval.")

    # ---- Visa Logic ----
    if c["visa_type"] == "Student Visa":
        if v["has_cas"] == "No":
            reasons.append("CAS is mandatory.")
        if v["financial_met"] == "No":
            reasons.append("Financial requirement not met.")
        if v["funds_28"] == "No":
            reasons.append("Funds not held for 28 days.")

    elif c["visa_type"] == "Graduate Visa":
        if v["completed"] == "No":
            reasons.append("Course not completed.")
        if v["provider_reported"] == "No":
            reasons.append("Provider has not reported completion.")

    elif c["visa_type"] == "Skilled Worker Visa":
        if v["job_offer"] == "No":
            reasons.append("Job offer required.")
        if v["salary_met"] == "No":
            reasons.append("Salary threshold not met.")

    elif c["visa_type"] == "Health and Care Visa":
        if v["registration"] == "No":
            reasons.append("Professional registration required.")

    elif c["visa_type"] == "Visitor Visa":
        if v["stay_ok"] == "No":
            reasons.append("Stay exceeds permitted duration.")

    # ---- OUTPUT ----
    st.header("Eligibility Result")

    if not reasons:
        st.success("✅ ELIGIBLE")
        st.markdown("### AI Explanation")
        st.write("Based on the information provided, you meet all UK visa eligibility requirements.")

    else:
        st.error("❌ NOT ELIGIBLE")
        st.markdown("### Reasons:")
        for r in reasons:
            st.write(f"- {r}")

        st.markdown("### AI Recommendation")
        st.write("You may reapply after correcting the above issues or selecting a more suitable visa type.")
