import streamlit as st
from datetime import date

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="SwiftVisa - UK Visa Eligibility System",
    layout="wide"
)

# ---------------- PROFESSIONAL STYLING ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }
    
    .gov-header {
        background: linear-gradient(135deg, #012169 0%, #1a3a6b 100%);
        padding: 2rem 3rem;
        border-radius: 0;
        color: white;
        margin: -1rem -1rem 2rem -1rem;
        box-shadow: 0 4px 20px rgba(1, 33, 105, 0.3);
    }
    
    .gov-logo {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .gov-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        color: white;
    }
    
    .gov-subtitle {
        font-size: 1rem;
        opacity: 0.95;
        margin-top: 0.5rem;
        font-weight: 400;
    }
    
    .section-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
        border: none ;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fc 100%);
        padding: 3rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        border: 2px solid #e8ecf1;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.06);
        border-top: 3px solid #d4351c;
        transition: transform 0.2s;
    }
    
    .feature-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #012169 0%, #1a3a6b 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2.5rem;
        border-radius: 8px;
        border: none;
        font-size: 1.05rem;
        box-shadow: 0 4px 16px rgba(1, 33, 105, 0.3);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #1a3a6b 0%, #012169 100%);
        box-shadow: 0 6px 20px rgba(1, 33, 105, 0.4);
        transform: translateY(-2px);
    }
    
    h1, h2, h3 {
        color: #012169;
        font-weight: 700;
    }
    
    .success-box {
        background: linear-gradient(135deg, #00703c 0%, #00883f 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 6px 24px rgba(0, 112, 60, 0.3);
    }
    
    .error-box {
        background: linear-gradient(135deg, #d4351c 0%, #b02e14 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.3rem;
        font-weight: 600;
        box-shadow: 0 6px 24px rgba(212, 53, 28, 0.3);
    }
    
    .recommendation-box {
        background: #f8f9fc;
        border-left: 4px solid #1d70b8;
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1.5rem;
    }
    
    .warning-box {
        background: #fff7e6;
        border-left: 5px solid #f47738;
        padding: 1.5rem;
        margin: 1.5rem 0;
        border-radius: 8px;
    }
    
    label {
        font-weight: 600 !important;
        color: #0b0c0c !important;
    }
    
    .stSelectbox, .stTextInput, .stTextArea, .stDateInput, .stNumberInput {
        margin-bottom: 1rem;
    }
    
    .progress-bar {
        background: #e8ecf1;
        height: 6px;
        border-radius: 3px;
        margin: 2rem 0;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, #012169 0%, #1d70b8 100%);
        height: 100%;
        border-radius: 3px;
        transition: width 0.3s;
    }
    
    .reason-item {
        background: white;
        border-left: 5px solid #d4351c;
        padding: 1rem;
        margin: 0.5rem 0;
        font-weight: 600;
        border-radius: 4px;
    }
    
    .rec-item {
        background: white;
        border-left: 5px solid #1d70b8;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = 1

# ---------------- HEADER ----------------
st.markdown("""
<div class="gov-header">
    <div class="gov-logo">
        <span style="font-size: 3.5rem;">🌍</span>
        <div>
            <h1 class="gov-title">SwiftVisa</h1>
            <p class="gov-subtitle">AI-Powered UK Visa Eligibility Screening System</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Progress indicator
progress = (st.session_state.page - 1) / 3 * 100
st.markdown(f"""
<div class="progress-bar">
    <div class="progress-fill" style="width: {progress}%"></div>
</div>
""", unsafe_allow_html=True)

# =====================================================
# PAGE 1 – WELCOME
# =====================================================
if st.session_state.page == 1:
    st.markdown("""
    <div class="welcome-card">
        <h1 style="color: #012169; margin-bottom: 1rem;">Welcome to SwiftVisa 🌍</h1>
        <p style="font-size: 1.2rem; color: #505a5f; margin-bottom: 2rem;">
            Your intelligent assistant for UK visa eligibility assessment
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-grid">
        <div class="feature-card">
            <div class="feature-icon">✅</div>
            <h3>Accurate Assessment</h3>
            <p>AI-powered evaluation based on official UK visa guidelines</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3>Multiple Visa Types</h3>
            <p>Support for Student, Graduate, Skilled Worker, and more</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h3>Instant Results</h3>
            <p>Get immediate feedback on your eligibility status</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">🔒</div>
            <h3>Secure & Private</h3>
            <p>Your data is processed securely and confidentially</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button(" Start Your Application", use_container_width=True):
            st.session_state.page = 2
            st.rerun()

# =====================================================
# PAGE 2 – COMMON DETAILS
# =====================================================
elif st.session_state.page == 2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 📋 Applicant Information")
    st.markdown("Please provide accurate information as per your official documents. Fields marked with * are mandatory.")
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Personal Details")
        full_name = st.text_input("Full Name (as per passport) *", placeholder="John Doe")
        dob = st.date_input("Date of Birth *", min_value=date(1930, 1, 1), max_value=date.today(), format="DD/MM/YYYY")
        nationality = st.text_input("Nationality *", placeholder="Indian")
        passport_number = st.text_input("Passport Number *", placeholder="A1234567")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Passport Information")
        passport_issue = st.date_input("Passport Issue Date *", min_value=date(1930, 1, 1), max_value=date.today(), format="DD/MM/YYYY")
        passport_expiry = st.date_input("Passport Expiry Date *", min_value=date.today(), format="DD/MM/YYYY")
        country = st.text_input("Country of Application *", placeholder="India")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Visa Details")
    
    col3, col4 = st.columns(2)
    with col3:
        visa_type = st.selectbox("Visa Type *", ["-", "Student Visa", "Graduate Visa", "Skilled Worker Visa", "Health and Care Visa", "Visitor Visa"])
        purpose = st.text_input("Purpose of Visit")
        travel_date = st.date_input("Intended Travel Date", min_value=date.today(), format="DD/MM/YYYY")
        
    with col4:
        stay_months = st.number_input("Length of Stay (Months) *", min_value=1, value=1)
        funds = st.number_input("Funds Available (£) *", min_value=0, value=0)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Background Information")
    
    col5, col6 = st.columns(2)
    with col5:
        english = st.selectbox("English Language Requirement Met *", ["-", "Yes", "No"])
        criminal = st.selectbox("Criminal History Declaration *", ["-", "Yes", "No"])
        
    with col6:
        refusal = st.selectbox("Previous UK Visa Refusal *", ["-", "Yes", "No"])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Contact Details")
    
    col7, col8 = st.columns(2)
    with col7:
        email = st.text_input("Email Address", placeholder="example@email.com")
        phone = st.text_input("Phone Number *", placeholder="+91 1234567890")
    with col8:
        address = st.text_area("Current Address", placeholder="Enter your full address")
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button("Continue to Visa-Specific Details ", use_container_width=True):
            errors = []
            if not full_name:
                errors.append("Full name is required")
            if not nationality:
                errors.append("Nationality is required")
            if not passport_number:
                errors.append("Passport number is required")
            if not country:
                errors.append("Country of application is required")
            if visa_type == "-":
                errors.append("Please select a visa type")
            if english == "-":
                errors.append("Please confirm English language requirement status")
            if criminal == "-":
                errors.append("Please declare criminal history status")
            if refusal == "-":
                errors.append("Please declare previous visa refusal status")
            if not phone:
                errors.append("Phone number is required")
                
            if errors:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("### ** Please complete all mandatory fields")
                for error in errors:
                    st.markdown(f"• {error}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.session_state.common = locals()
                st.session_state.page = 3
                st.rerun()

# =====================================================
# PAGE 3 – VISA SPECIFIC DETAILS
# =====================================================
elif st.session_state.page == 3:
    v = st.session_state.common["visa_type"]
    
    st.markdown(f'<div class="section-card">', unsafe_allow_html=True)
    st.markdown(f"## 🎯 {v} – Specific Requirements")
    st.markdown("Please provide the following information specific to your visa type. Fields marked with * are mandatory.")
    st.markdown('</div>', unsafe_allow_html=True)

    data = {}
    
    st.markdown('<div class="section-card">', unsafe_allow_html=True)

    if v == "Student Visa":
        st.markdown("### 📚 Student Visa Requirements")
        col1, col2 = st.columns(2)
        with col1:
            data["has_cas"] = st.selectbox("Has CAS *", ["-", "Yes", "No"])
            data["cas_ref"] = st.text_input("CAS Reference Number")
            data["provider"] = st.selectbox("Education Provider Licensed *", ["-", "Yes", "No"])
            data["course_level"] = st.text_input("Course Level", placeholder="Bachelor's / Master's / PhD")
            data["full_time"] = st.selectbox("Course Full Time *", ["-", "Yes", "No"])
            data["duration"] = st.number_input("Course Duration (Months) *", min_value=1, value=1)
        with col2:
            data["course_start"] = st.date_input("Course Start Date *", format="DD/MM/YYYY")
            data["course_end"] = st.date_input("Course End Date *", format="DD/MM/YYYY")
            data["finance"] = st.selectbox("Financial Requirement Met *", ["-", "Yes", "No"])
            data["funds_28"] = st.selectbox("Funds Held for 28 Days *", ["-", "Yes", "No"])
            data["english"] = st.selectbox("English Requirement Met *", ["-", "Yes", "No"])

    elif v == "Graduate Visa":
        st.markdown("### 🎓 Graduate Visa Requirements")
        col1, col2 = st.columns(2)
        with col1:
            data["in_uk"] = st.selectbox("Currently in UK *", ["-", "Yes", "No"])
            data["visa_type"] = st.selectbox("Current Visa Type *", ["-", "Student", "Tier 4"])
            data["completed"] = st.selectbox("Course Completed *", ["-", "Yes", "No"])
            data["course_level"] = st.text_input("Course Level Completed", placeholder="Bachelor's / Master's / PhD")
        with col2:
            data["provider"] = st.selectbox("Provider Licensed *", ["-", "Yes", "No"])
            data["reported"] = st.selectbox("Provider Reported Completion *", ["-", "Yes", "No"])
            data["cas"] = st.text_input("Original CAS Reference")
            data["visa_valid"] = st.selectbox("Student Visa Valid *", ["-", "Yes", "No"])

    elif v == "Skilled Worker Visa":
        st.markdown("### 💼 Skilled Worker Visa Requirements")
        col1, col2 = st.columns(2)
        with col1:
            data["job_offer"] = st.selectbox("Job Offer Confirmed *", ["-", "Yes", "No"])
            data["sponsor"] = st.selectbox("Licensed Sponsor *", ["-", "Yes", "No"])
            data["cos"] = st.selectbox("CoS Issued *", ["-", "Yes", "No"])
            data["cos_ref"] = st.text_input("CoS Reference Number")
            data["job_title"] = st.text_input("Job Title")
            data["soc"] = st.text_input("SOC Code")
        with col2:
            data["eligible"] = st.selectbox("Eligible Occupation *", ["-", "Yes", "No"])
            data["salary"] = st.number_input("Salary Offered (£)", min_value=0, value=0)
            data["salary_met"] = st.selectbox("Salary Threshold Met *", ["-", "Yes", "No"])
            data["english"] = st.selectbox("English Requirement Met *", ["-", "Yes", "No"])
            data["crc_req"] = st.selectbox("Criminal Record Required *", ["-", "Yes", "No"])
            data["crc_prov"] = st.selectbox("Criminal Record Provided *", ["-", "Yes", "No"])

    elif v == "Health and Care Visa":
        st.markdown("### 🏥 Health and Care Visa Requirements")
        col1, col2 = st.columns(2)
        with col1:
            data["job_offer"] = st.selectbox("Job Offer Confirmed *", ["-", "Yes", "No"])
            data["sponsor"] = st.selectbox("Healthcare Sponsor *", ["-", "Yes", "No"])
            data["cos"] = st.selectbox("CoS Issued *", ["-", "Yes", "No"])
            data["cos_ref"] = st.text_input("CoS Reference Number")
            data["job_title"] = st.text_input("Job Title")
            data["soc"] = st.text_input("SOC Code")
        with col2:
            data["eligible"] = st.selectbox("Eligible Role *", ["-", "Yes", "No"])
            data["salary"] = st.number_input("Salary Offered (£)", min_value=0, value=0)
            data["salary_rules"] = st.selectbox("Meets Salary Rules *", ["-", "Yes", "No"])
            data["reg_req"] = st.selectbox("Registration Required *", ["-", "Yes", "No"])
            data["reg_prov"] = st.selectbox("Registration Provided *", ["-", "Yes", "No"])
            data["english"] = st.selectbox("English Requirement Met *", ["-", "Yes", "No"])

    elif v == "Visitor Visa":
        st.markdown("### 🌍 Visitor Visa Requirements")
        col1, col2 = st.columns(2)
        with col1:
            data["purpose"] = st.text_input("Purpose of Visit")
            data["permitted"] = st.selectbox("Purpose Permitted *", ["-", "Yes", "No"])
            data["stay"] = st.number_input("Length of Stay (Months)", min_value=1, value=1)
            data["limit"] = st.selectbox("Within 6 Months *", ["-", "Yes", "No"])
        with col2:
            data["accommodation"] = st.selectbox("Accommodation Arranged *", ["-", "Yes", "No"])
            data["return"] = st.selectbox("Return Ticket *", ["-", "Yes", "No"])
            data["leave"] = st.selectbox("Intends to Leave UK *", ["-", "Yes", "No"])
            data["funds"] = st.selectbox("Sufficient Funds *", ["-", "Yes", "No"])
    
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button(" Check Eligibility", use_container_width=True):
            errors = []
            
            # Validate mandatory fields for each visa type
            if v == "Student Visa":
                if data["has_cas"] == "-":
                    errors.append("CAS confirmation is required")
                if data["provider"] == "-":
                    errors.append("Education provider licensing status is required")
                if data["full_time"] == "-":
                    errors.append("Course type (full-time/part-time) is required")
                if data["finance"] == "-":
                    errors.append("Financial requirement status is required")
                if data["funds_28"] == "-":
                    errors.append("28-day funds confirmation is required")
                if data["english"] == "-":
                    errors.append("English language requirement status is required")
                    
            elif v == "Graduate Visa":
                if data["in_uk"] == "-":
                    errors.append("UK residence status is required")
                if data["visa_type"] == "-":
                    errors.append("Current visa type is required")
                if data["completed"] == "-":
                    errors.append("Course completion status is required")
                if data["provider"] == "-":
                    errors.append("Provider licensing status is required")
                if data["reported"] == "-":
                    errors.append("Provider reporting status is required")
                if data["visa_valid"] == "-":
                    errors.append("Student visa validity status is required")
                    
            elif v == "Skilled Worker Visa":
                if data["job_offer"] == "-":
                    errors.append("Job offer confirmation is required")
                if data["sponsor"] == "-":
                    errors.append("Sponsor licensing status is required")
                if data["cos"] == "-":
                    errors.append("CoS issuance status is required")
                if data["eligible"] == "-":
                    errors.append("Occupation eligibility status is required")
                if data["salary_met"] == "-":
                    errors.append("Salary threshold status is required")
                if data["english"] == "-":
                    errors.append("English language requirement status is required")
                if data["crc_req"] == "-":
                    errors.append("Criminal record certificate requirement status is required")
                if data["crc_prov"] == "-":
                    errors.append("Criminal record certificate provision status is required")
                    
            elif v == "Health and Care Visa":
                if data["job_offer"] == "-":
                    errors.append("Job offer confirmation is required")
                if data["sponsor"] == "-":
                    errors.append("Healthcare sponsor status is required")
                if data["cos"] == "-":
                    errors.append("CoS issuance status is required")
                if data["eligible"] == "-":
                    errors.append("Role eligibility status is required")
                if data["salary_rules"] == "-":
                    errors.append("Salary requirements status is required")
                if data["reg_req"] == "-":
                    errors.append("Professional registration requirement status is required")
                if data["reg_prov"] == "-":
                    errors.append("Professional registration provision status is required")
                if data["english"] == "-":
                    errors.append("English language requirement status is required")
                    
            elif v == "Visitor Visa":
                if data["permitted"] == "-":
                    errors.append("Purpose permission status is required")
                if data["limit"] == "-":
                    errors.append("Stay duration confirmation is required")
                if data["accommodation"] == "-":
                    errors.append("Accommodation status is required")
                if data["return"] == "-":
                    errors.append("Return ticket status is required")
                if data["leave"] == "-":
                    errors.append("Intent to leave confirmation is required")
                if data["funds"] == "-":
                    errors.append("Sufficient funds confirmation is required")
            
            if errors:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.markdown("### ** Please complete all mandatory fields")
                for error in errors:
                    st.markdown(f"• {error}")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.session_state.visa_data = data
                st.session_state.page = 4
                st.rerun()

# =====================================================
# PAGE 4 – RESULT
# =====================================================
elif st.session_state.page == 4:
    reasons = []
    recommendations = []
    c = st.session_state.common
    v = st.session_state.visa_data

    # Age check
    age = (date.today() - c["dob"]).days // 365
    if age < 18:
        reasons.append("Applicant must be 18 years or older")
        recommendations.append("Wait until you turn 18 before applying for a UK visa")

    # Common checks
    if c["english"] == "No":
        reasons.append("English language requirement not met")
        recommendations.append("Take an approved English language test (IELTS, TOEFL, PTE) and achieve the required score for your visa category")
    
    if c["criminal"] == "Yes":
        reasons.append("Criminal history declared")
        recommendations.append("Provide detailed information about your convictions and seek legal advice. Some convictions may not affect your application")
    
    if c["refusal"] == "Yes":
        reasons.append("Previous UK visa refusal on record")
        recommendations.append("Address the reasons for your previous refusal before reapplying. Consider obtaining legal advice to strengthen your application")

    # STUDENT VISA checks
    if c["visa_type"] == "Student Visa":
        if v["has_cas"] == "No":
            reasons.append("Confirmation of Acceptance for Studies (CAS) is mandatory")
            recommendations.append("Apply to a UK educational institution and obtain a CAS from a licensed Tier 4 sponsor before applying for your visa")
        if v["provider"] == "No":
            reasons.append("Education provider must be licensed by UKVI")
            recommendations.append("Ensure you're applying to a licensed education provider. Check the UK government's register of licensed sponsors")
        if v["full_time"] == "No":
            reasons.append("Course must be full-time")
            recommendations.append("Student visas generally require full-time study. Confirm with your education provider that your course meets this requirement")
        if v["finance"] == "No":
            reasons.append("Financial requirement not met")
            recommendations.append("Ensure you have at least £1,023 per month (for up to 9 months) for courses in London or £820 per month outside London, plus your course fees")
        if v["funds_28"] == "No":
            reasons.append("Funds must be held for 28 consecutive days")
            recommendations.append("Maintain the required amount in your bank account for at least 28 consecutive days before applying")
        if v["english"] == "No":
            reasons.append("English language requirement not met for Student Visa")
            recommendations.append("Provide evidence of English proficiency through an approved test or by studying in a majority English-speaking country")

    # GRADUATE VISA checks
    elif c["visa_type"] == "Graduate Visa":
        if v["in_uk"] == "No":
            reasons.append("Must be in the UK to apply for Graduate Visa")
            recommendations.append("Graduate Visas can only be applied for from within the UK. You must have a valid Student visa")
        if v["visa_type"] not in ["Student", "Tier 4"]:
            reasons.append("Must currently hold a Student or Tier 4 visa")
            recommendations.append("Ensure you're on a Student or Tier 4 visa when applying for a Graduate Visa")
        if v["completed"] == "No":
            reasons.append("Course must be completed")
            recommendations.append("Complete your course successfully and ensure your education provider confirms completion to UKVI")
        if v["provider"] == "No":
            reasons.append("Education provider must be licensed")
            recommendations.append("Your provider must be a licensed sponsor. Contact them to confirm their licensing status")
        if v["reported"] == "No":
            reasons.append("Provider must report course completion to UKVI")
            recommendations.append("Contact your education provider to ensure they've reported your successful completion to UK Visas and Immigration")
        if v["visa_valid"] == "No":
            reasons.append("Student visa must be valid when applying")
            recommendations.append("Apply for a Graduate Visa before your Student visa expires. You can apply from within the UK up to 3 months before your course end date")

    # SKILLED WORKER VISA checks
    elif c["visa_type"] == "Skilled Worker Visa":
        if v["job_offer"] == "No":
            reasons.append("You must have a confirmed job offer")
            recommendations.append("Secure a job offer from a UK employer who is a licensed sponsor before applying")
        if v["sponsor"] == "No":
            reasons.append("Your employer must be a licensed UK sponsor")
            recommendations.append("Ensure your prospective employer holds a valid UK sponsor licence. They can apply for one through UKVI")
        if v["cos"] == "No":
            reasons.append("You must have a Certificate of Sponsorship (CoS)")
            recommendations.append("Your employer must issue you a CoS. Each CoS is unique to you and your job")
        if v["eligible"] == "No":
            reasons.append("The job must be on the list of eligible occupations")
            recommendations.append("Check that your job role is at RQF level 3 or above (equivalent to A level) and appears on the eligible occupations list")
        if v["salary_met"] == "No":
            reasons.append("The salary does not meet the minimum threshold")
            recommendations.append("Your salary must be at least £38,700 per year or the 'going rate' for your job, whichever is higher. Negotiate with your employer to meet this requirement")
        if v["english"] == "No":
            reasons.append("English language requirement not met for Skilled Worker Visa")
            recommendations.append("Provide evidence of English proficiency at B1 level or above (CEFR scale) through an approved English language test")
        if v["crc_req"] == "Yes" and v["crc_prov"] == "No":
            reasons.append("Criminal record certificate required but not provided")
            recommendations.append("Obtain a criminal record certificate from relevant authorities if your role requires it")

    # HEALTH AND CARE VISA checks
    elif c["visa_type"] == "Health and Care Visa":
        if v["job_offer"] == "No":
            reasons.append("Confirmed job offer in health or social care is required")
            recommendations.append("Secure a job offer from an NHS, NHS supplier, or adult social care provider who is a licensed sponsor")
        if v["sponsor"] == "No":
            reasons.append("Employer must be a licensed healthcare sponsor")
            recommendations.append("Ensure your employer is a licensed Health and Care Worker sponsor")
        if v["cos"] == "No":
            reasons.append("Certificate of Sponsorship (CoS) is required")
            recommendations.append("Your healthcare employer must issue you a CoS before you can apply")
        if v["eligible"] == "No":
            reasons.append("Job must be an eligible healthcare role")
            recommendations.append("Check that your role appears on the Health and Care Worker visa eligible occupation list")
        if v["salary_rules"] == "No":
            reasons.append("Salary does not meet requirements")
            recommendations.append("Your salary must meet the minimum threshold for your occupation. Health and Care Worker visas have lower salary thresholds than standard Skilled Worker visas")
        if v["reg_req"] == "Yes" and v["reg_prov"] == "No":
            reasons.append("Professional registration is required")
            recommendations.append("Obtain registration with the appropriate UK regulatory body (e.g., NMC for nurses, GMC for doctors, HCPC for allied health professionals)")
        if v["english"] == "No":
            reasons.append("English language requirement not met")
            recommendations.append("Provide evidence of English proficiency at B1 level (CEFR scale) through an approved English language test")

    # VISITOR VISA checks
    elif c["visa_type"] == "Visitor Visa":
        if v["permitted"] == "No":
            reasons.append("Purpose of visit is not permitted under visitor visa rules")
            recommendations.append("Visitor visas are for tourism, visiting family/friends, business meetings, or short courses. Ensure your purpose aligns with these categories")
        if v["limit"] == "No":
            reasons.append("Intended stay exceeds 6-month maximum")
            recommendations.append("Standard visitor visas allow stays up to 6 months. Consider shortening your trip or exploring other visa categories if you need to stay longer")
        if v["accommodation"] == "No":
            reasons.append("Accommodation must be arranged")
            recommendations.append("Book accommodation in advance (hotel, rental property, or provide details of friends/family you'll stay with)")
        if v["return"] == "No":
            reasons.append("Return ticket or proof of onward travel required")
            recommendations.append("Book a return ticket or demonstrate clear plans to leave the UK at the end of your visit")
        if v["leave"] == "No":
            reasons.append("Must intend to leave the UK at end of visit")
            recommendations.append("Provide evidence of ties to your home country (job, property, family) to demonstrate your intention to return")
        if v["funds"] == "No":
            reasons.append("Insufficient funds for visit")
            recommendations.append("Show bank statements proving you can support yourself during your stay without working or accessing public funds")

    # Display results
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("##  Eligibility Assessment Results")
    st.markdown('</div>', unsafe_allow_html=True)

    if not reasons:
        st.markdown("""
        <div class="success-box">
            ✅ ELIGIBLE FOR UK VISA APPLICATION
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="recommendation-box">
            <h3> AI Recommendation</h3>
            <p style="font-size: 1.1rem; line-height: 1.6;">
                Congratulations! Your profile satisfies all the requirements for a <strong>{}</strong>. 
                You may proceed with the official UK visa application through the UK Government's official portal.
            </p>
            <p style="margin-top: 1rem;">
                <strong>Next Steps:</strong><br>
                1. Visit the official UK Government visa portal<br>
                2. Complete the online application form<br>
                3. Pay the application fee and immigration health surcharge<br>
                4. Book and attend your biometrics appointment<br>
                5. Submit all required supporting documents
            </p>
            <p style="margin-top: 1rem; font-style: italic; color: #505a5f;">
                <strong>Important:</strong> This is an eligibility check only. Meeting these requirements does not guarantee visa approval. 
                The final decision rests with UK Visas and Immigration.
            </p>
        </div>
        """.format(c["visa_type"]), unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div class="error-box">
            ❌ NOT CURRENTLY ELIGIBLE
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("###  Issues Identified:")
        for r in reasons:
            st.markdown(f'<div class="reason-item">• {r}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="recommendation-box">
            <h3> AI Recommendations to Improve Your Eligibility</h3>
            <p style="font-size: 1.05rem; margin-bottom: 1.5rem;">
                Please address the following issues before proceeding with your application:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        for idx, rec in enumerate(recommendations, 1):
            st.markdown(f"""
            <div class="rec-item">
                <strong style="color: #012169;">Recommendation {idx}:</strong>
                <p style="margin-top: 0.5rem; font-size: 1.05rem; line-height: 1.6;">{rec}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="recommendation-box">
            <h4>Need Additional Support?</h4>
            <ul style="line-height: 1.8;">
                <li>Visit the official UK government visa guidance at <strong>gov.uk/browse/visas-immigration</strong></li>
                <li>Contact an immigration advisor or solicitor for professional assistance</li>
                <li>Review the detailed eligibility requirements for your specific visa type</li>
                <li>Ensure all documents are prepared before reapplying</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        if st.button(" Start New Application", use_container_width=True):
            st.session_state.page = 1
            st.rerun()