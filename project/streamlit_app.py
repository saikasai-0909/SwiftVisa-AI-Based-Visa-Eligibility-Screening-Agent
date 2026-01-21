#final version of streamlit app
import streamlit as st
from new_rag_pipeline import run_rag_pipeline 
from datetime import date, datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="SwiftVisa Eligibility Checker",
    page_icon="SV",
    layout="centered"
)

# =========================
# SESSION STATE INIT
# =========================
if "stage" not in st.session_state:
    st.session_state.stage = 0 

if "user_profile" not in st.session_state:
    st.session_state.user_profile = {}

if "visa_answers" not in st.session_state:
    st.session_state.visa_answers = {}

if "eligibility_result" not in st.session_state:
    st.session_state.eligibility_result = None
YES_NO = ["Yes", "No"]
official_cascot_url = "https://cascotweb.warwick.ac.uk/#/classification/soc2020"

# --- HELPERS ---
def get_date_val(data_dict, key, default_date=date.today()):
    date_str = data_dict.get(key)
    if date_str:
        try:
            return datetime.strptime(date_str, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return default_date
    return default_date

def get_index(options_list, current_value):
    try:
        return options_list.index(current_value)
    except (ValueError, KeyError):
        return 0

# =========================
# DYNAMIC CSS SYSTEM
# =========================
if st.session_state.stage == 0:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        color: #f8fafc;
        padding-top: 0rem;
    }
    .logo-box {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 0px;
        height: 80px;
        animation: fadeInDown 1.5s ease-out;
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .typing-title {
        font-size: 3.2rem;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        overflow: hidden;
        border-right: .15em solid #38bdf8;
        white-space: nowrap;
        margin: 0 auto;
        letter-spacing: .10em;
        width: 11ch; 
        animation: typing 2.5s steps(11, end), blink-caret .75s step-end infinite;
    }
    @keyframes typing { from { width: 0 } to { width: 11ch } }
    @keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: #38bdf8; } }
    .slide-tagline {
        font-size: 1.3rem;
        text-align: center;
        color: #38bdf8;
        margin-top: 10px;
        opacity: 0;
        transform: translateX(50px);
        animation: slideIn 0.8s forwards ease-out;
        animation-delay: 2.6s;
    }
    @keyframes slideIn { to { opacity: 1; transform: translateX(0); } }
    .home-desc {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 20px 0;
        text-align: center;
        line-height: 1.6;
        font-size: 1rem;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #38bdf8, #2563eb);
        color: white;
        border: none;
        padding: 15px 40px;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 50px;
        transition: all 0.3s ease;
        display: block;
        margin: 0 auto;
        box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #93c5fd, #bfdbfe);
        color: #020617;
    }
    h1 { text-align: center; color: #1e3a8a; margin-bottom: 0px; }
    .stRadio > div { display: flex !important; flex-direction: row !important; gap: 16px !important; }
    .verdict-badge {
        padding: 8px 16px; border-radius: 20px; font-weight: bold;
        display: inline-block; margin-bottom: 15px; text-transform: uppercase;
    }
    .badge-eligible { background-color: #dcfce7; color: #166534; border: 1px solid #166534; }
    .badge-not-eligible { background-color: #fee2e2; color: #991b1b; border: 1px solid #991b1b; }
    .stProgress > div > div > div > div { background-color: #1e3a8a; }
    </style>
    """, unsafe_allow_html=True)

# ==================================================
# STAGE 0 – HOME PAGE (Updated with Feature Boxes)
# ==================================================
if st.session_state.stage == 0:
   
    st.markdown("""
    <style>
    .trust-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin: 30px 0;
    }
    .trust-card {
        background: rgba(255, 255, 255, 0.08);
        border-top: 3px solid #38bdf8;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        flex: 1;
        transition: transform 0.3s ease;
    }
    .trust-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.12);
    }
    .trust-icon {
        font-size: 1.8rem;
        margin-bottom: 10px;
        display: block;
    }
    .trust-card h4 {
        color: #38bdf8;
        margin-bottom: 8px;
        font-size: 1.1rem;
    }
    .trust-card p {
        font-size: 0.85rem;
        color: #cbd5e1;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)

    # Logo, Title, and Tagline
    col1, col2, col3 = st.columns([2.2, 1, 2.2]) 
    with col2:
        st.markdown('<div class="logo-box">', unsafe_allow_html=True)
        st.image("logo.jpg") 
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.markdown('<div class="typing-title">SwiftVisa</div>', unsafe_allow_html=True)
    st.markdown('<div class="slide-tagline">Navigate Your UK Visa Journey with AI Precision</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="home-desc">
        Welcome to the next generation of immigration assistance. <b>SwiftVisa</b> processes official 
        UK Home Office policy documents in real-time to verify your eligibility. 
        Forget the guesswork—get a clear, grounded decision for your visa application in seconds.
    </div>
    """, unsafe_allow_html=True)

    # Three Feature/Trust Boxes ---
    t_col1, t_col2, t_col3 = st.columns(3)
    
    with t_col1:
        st.markdown("""
        <div class="trust-card">
            <span class="trust-icon">⚖️</span>
            <h4>Policy Grounded</h4>
            <p>Directly analyzed from the latest UK Government immigration rules and guidance.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with t_col2:
        st.markdown("""
        <div class="trust-card">
            <span class="trust-icon">🧠</span>
            <h4>AI Reasoning</h4>
            <p>Sophisticated neural processing that understands complex criteria without rigid rules.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with t_col3:
        st.markdown("""
        <div class="trust-card">
            <span class="trust-icon">⚡</span>
            <h4>Instant Clarity</h4>
            <p>Receive personalized and detailed explanation with extended support in seconds.</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("##") 

    if st.button("Get Started ➜"):
        st.session_state.stage = 1
        st.rerun()

# ==================================================
# STAGE 1 – COMPACT SQUARE VISA SELECTION (CENTERED)
# ==================================================
elif st.session_state.stage == 1:
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
    }

    /* Card Shell - Added bottom padding to make room for button */
    .visa-card-shell {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-top: 4px solid #CF142B; 
        border-radius: 12px;
        padding: 20px 15px 60px 15px; /* Padding for button space */
        text-align: center;
        
        height: 320px; 
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    
    .title-text { 
        height: 45px; 
        color: #38bdf8; 
        font-weight: bold; 
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        margin-top: 10px;
    }

    .desc-text { 
        height: 70px; 
        color: #cbd5e1; 
        font-size: 0.85rem;
        line-height: 1.4;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    
    /* 1. Target the CONTAINER of the button to force centering */
    div.stButton {
        display: flex;
        justify-content: center; 
        width: 100%;             
    }

    /* 2. Target the BUTTON itself for styling & pull-up */
    div.stButton > button {
        background-color: #D4AF37 !important; 
        color: #000 !important;
        font-weight: bold !important;
        border: none !important;
        width: 110px !important; 
        height: 38px !important;
        border-radius: 8px !important;
        
        /* Pull UP into the card */
        position: relative;
        top: -55px; 
        z-index: 99;
    }

    div.stButton > button:hover {
        background-color: #FFD700 !important;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align: center; color: white;'>Select Your Visa Type</h1>", unsafe_allow_html=True)
    st.write("##")

    visas = [
        {"name": "Standard Visitor", "desc": "Short-term stay for tourism, family visits, or business.", "img": "https://cdn-icons-png.flaticon.com/512/826/826070.png"},
        {"name": "Student Visa", "desc": "For international students admitted to licensed UK institutions.", "img": "https://cdn-icons-png.flaticon.com/512/2997/2997312.png"},
        {"name": "Skilled Worker", "desc": "For professionals with job offers from licensed UK sponsors.", "img": "https://cdn-icons-png.flaticon.com/512/3095/3095221.png"},
        {"name": "Health & Care", "desc": "Specialized route for medical professionals with job offers.", "img": "https://cdn-icons-png.flaticon.com/512/2966/2966327.png"},
        {"name": "Graduate Visa", "desc": "Work in the UK for 2+ years after completing your degree.", "img": "https://cdn-icons-png.flaticon.com/512/2490/2490354.png"}
    ]

    def render_card(visa, idx):
        # 1. Render Card Shell
        st.markdown(f"""
        <div class="visa-card-shell">
            <img src="{visa['img']}" width="65">
            <div class="title-text">{visa['name']}</div>
            <div class="desc-text">{visa['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Render Button (Now centered by parent div.stButton style)
        if st.button("Select", key=f"v_{idx}"):
            st.session_state.user_profile['visa_type'] = visa['name']
            st.session_state.stage = 2
            st.rerun()

    # Layout Row 1 (3 items)
    c1, c2, c3 = st.columns(3)
    with c1: render_card(visas[0], 0)
    with c2: render_card(visas[1], 1)
    with c3: render_card(visas[2], 2)

    # Layout Row 2 (2 items centered)
    st.write("###")
    s1, c4, c5, s2 = st.columns([0.5, 1, 1, 0.5])
    with c4: render_card(visas[3], 3)
    with c5: render_card(visas[4], 4)

    st.write("##")
    if st.button("⬅ Back to Home", key="back_home"):
        st.session_state.stage = 0
        st.rerun()

# ==================================================
# STAGE 2: PERSONAL DETAILS (Dark Theme)
# ==================================================
elif st.session_state.stage == 2:
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); color: #f8fafc; }
    .form-box {
        background: rgba(255, 255, 255, 0.05); padding: 30px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;
    }
    .section-title { color: #38bdf8; font-size: 1.8rem; font-weight: bold; margin-bottom: 5px; }
    .stTextInput label, .stDateInput label, .stNumberInput label, .stTextArea label, .stRadio label { color: #e2e8f0 !important; font-weight: 500; }
    div.stButton > button { background-color: #D4AF37 !important; color: black !important; border: none; font-weight: bold; }
    div.stButton > button:hover { background-color: #FFD700 !important; }
    
    /* TOOLTIP FIX (WHITE) */
    .stTooltipIcon { color: #ffffff !important; }
    [data-testid="stTooltipIcon"] > div > svg { fill: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

    st.progress(25)
    visa = st.session_state.user_profile.get('visa_type', 'Selected Visa')
    st.markdown(f'<div class="form-box"><div class="section-title">Personal Details</div><div style="color:#cbd5e1;">Application for: <b>{visa}</b></div></div>', unsafe_allow_html=True)

    p = st.session_state.user_profile
    c1, c2 = st.columns(2)
    with c1:
        # Added Tooltips here
        fn = st.text_input("Full Name", value=p.get("full_name", ""), help="Enter your name exactly as it appears on your travel document.")
        nat = st.text_input("Nationality", value=p.get("nationality", ""), help="The country where you hold citizenship.")
        pn = st.text_input("Passport Number", value=p.get("passport_number", ""), help="The number on your current valid passport.")
        email = st.text_input("Email", value=p.get("email", ""), help="We will use this for updates.")
        phone = st.text_input("Phone", value=p.get("phone", ""))
    with c2:
        dob = st.date_input("Date of Birth", value=get_date_val(p, "date_of_birth", date(2000, 1, 1)))
        loc = st.text_input("Current Location", value=p.get("current_location", ""), help="The country from which you are submitting your application.")
        pid = st.date_input("Passport Issue Date", value=get_date_val(p, "passport_issue_date"))
        ped = st.date_input("Passport Expiry Date", value=get_date_val(p, "passport_expiry_date"))
    addr = st.text_area("Address", value=p.get("address", ""))

    st.write("##")
    b1, b2 = st.columns([1, 1])
    if b1.button("⬅ Back"):
        st.session_state.stage = 1
        st.rerun()
    if b2.button("Next: Journey Details ➜"):
        if not fn or not nat or not pn or not loc: st.error("Fill mandatory fields.")
        elif pid < dob: st.error("Invalid dates.")
        else:
            st.session_state.user_profile.update({"full_name": fn, "nationality": nat, "passport_number": pn, "email": email, "date_of_birth": str(dob), "current_location": loc, "passport_issue_date": str(pid), "passport_expiry_date": str(ped), "phone": phone, "address": addr})
            st.session_state.stage = 3
            st.rerun()

# ==================================================
# STAGE 3: JOURNEY DETAILS
# ==================================================
elif st.session_state.stage == 3:
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); color: #f8fafc; }
    .form-box {
        background: rgba(255, 255, 255, 0.05); padding: 30px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;
    }
    .section-title { color: #D4AF37; font-size: 1.8rem; font-weight: bold; }
    .stTextInput label, .stDateInput label, .stNumberInput label, .stRadio label { color: #e2e8f0 !important; font-size: 1rem !important; }
    div[role="radiogroup"] label p { color: #cbd5e1 !important; }
    div.stButton > button { background-color: #D4AF37 !important; color: black !important; border: none; font-weight: bold; }
    div.stButton > button:hover { background-color: #FFD700 !important; }
    
    /* TOOLTIP FIX (WHITE) */
    .stTooltipIcon { color: #ffffff !important; }
    [data-testid="stTooltipIcon"] > div > svg { fill: #ffffff !important; }
    </style>
    """, unsafe_allow_html=True)

    st.progress(50)
    p = st.session_state.user_profile
    st.markdown('<div class="form-box"><div class="section-title">Journey Details</div><div style="color:#cbd5e1;">Trip details</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        purpose = st.text_input("Main Purpose of Visit", value=p.get("purpose", ""), help="e.g. Tourism, Study, Work, Visiting Family")
        funds = st.number_input("Funds Available (£)", min_value=0, value=int(p.get("funds", 0)), help="Total savings available for your stay in the UK.")
        english = st.radio("English Requirement Met?", YES_NO, index=get_index(YES_NO, p.get("english_met", "No")), horizontal=True, help="Have you passed an approved English test or do you have a degree taught in English?")
        refusal = st.radio("Previous Visa Refusal?", YES_NO, index=get_index(YES_NO, p.get("previous_refusal", "No")), horizontal=True, help="Have you ever been refused a visa for the UK or any other country?")
    with c2:
        travel_date = st.date_input("Intended Travel Date", value=get_date_val(p, "travel_date"))
        stay_len = st.number_input("Length of Stay (Months)", min_value=1, value=int(p.get("length_of_stay", 6)))
        criminal = st.radio("Criminal History?", YES_NO, index=get_index(YES_NO, p.get("criminal_history", "No")), horizontal=True, help="Do you have any criminal convictions in any country?")

    st.write("##")
    b1, b2 = st.columns([1, 1])
    if b1.button("⬅ Back"):
        st.session_state.stage = 2
        st.rerun()
    if b2.button("Next: Visa Specifics ➜"):
        st.session_state.user_profile.update({"purpose": purpose, "travel_date": str(travel_date), "funds": funds, "length_of_stay": stay_len, "criminal_history": criminal, "english_met": english, "previous_refusal": refusal})
        st.session_state.stage = 4
        st.rerun()

# ==================================================
# STAGE 4: VISA SPECIFIC DETAILS
# ==================================================
elif st.session_state.stage == 4:
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); color: #f8fafc; }
    .form-box {
        background: rgba(255, 255, 255, 0.05); padding: 30px; border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 20px;
    }
    .section-title { color: #D4AF37; font-size: 1.8rem; font-weight: bold; }
    .stTextInput label, .stDateInput label, .stNumberInput label, .stRadio label, .stSelectbox label { 
        color: #e2e8f0 !important; font-size: 1rem !important; 
    }
    div[role="radiogroup"] label p { color: #cbd5e1 !important; }
    
    /* GOLD BUTTONS */
    div.stButton > button {
        background-color: #D4AF37 !important; color: black !important; border: none; font-weight: bold;
    }
    div.stButton > button:hover { background-color: #FFD700 !important; }
    
    /* --- FORCE WHITE TOOLTIPS (The Nuclear Fix) --- */
    /* 1. Target the outer container */
    [data-testid="stTooltipIcon"] {
        color: #ffffff !important;
    }
    /* 2. Target the SVG element itself */
    [data-testid="stTooltipIcon"] > div > svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    /* 3. Target the internal path of the icon */
    [data-testid="stTooltipIcon"] > div > svg > path {
        fill: #ffffff !important;
        color: #ffffff !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.progress(75)
    
    selected_visa = st.session_state.user_profile.get('visa_type', 'Standard Visitor')
    ans = st.session_state.visa_answers
    va = {} 

    st.markdown(f"""
    <div class="form-box">
        <div class="section-title">{selected_visa} Specifics</div>
        <div style="color: #cbd5e1;">Detailed eligibility questions.</div>
    </div>
    """, unsafe_allow_html=True)

    # --- SKILLED WORKER VISA (3-Column Layout) ---
    if selected_visa == "Skilled Worker":
        c1, c2, c3 = st.columns(3)
        
        # Column 1: General Job Details
        with c1:
            va["job_offer_confirmed"] = st.radio("Job offer confirmed?", YES_NO, index=get_index(YES_NO, ans.get("job_offer_confirmed", "Yes")), horizontal=True)
            va["employer_is_licensed_sponsor"] = st.radio("Employer is licensed sponsor?", YES_NO, index=get_index(YES_NO, ans.get("employer_is_licensed_sponsor", "Yes")), horizontal=True)
            va["certificate_of_sponsorship_issued"] = st.radio("CoS issued?", YES_NO, index=get_index(YES_NO, ans.get("certificate_of_sponsorship_issued", "Yes")), horizontal=True)
            va["cos_reference_number"] = st.text_input("CoS reference number", value=ans.get("cos_reference_number", ""), help="The digital reference number from your UK employer.")

        # Column 2: Eligibility & Salary
        with c2:
            va["job_is_eligible_occupation"] = st.radio("Job is eligible occupation?", YES_NO, index=get_index(YES_NO, ans.get("job_is_eligible_occupation", "Yes")), horizontal=True)
            va["salary_offered"] = st.number_input("Salary offered (£)", min_value=0, value=ans.get("salary_offered", 0))
            va["meets_minimum_salary_threshold"] = st.radio("Meets salary threshold?", YES_NO, index=get_index(YES_NO, ans.get("meets_minimum_salary_threshold", "Yes")), horizontal=True, help="Usually at least £38,700 unless you are a 'new entrant'.")
            va["english_requirement_met"] = st.radio("English requirement met?", YES_NO, index=get_index(YES_NO, ans.get("english_requirement_met", "Yes")), horizontal=True)
            va["criminal_record_certificate_required"] = st.radio("Criminal record cert required?", YES_NO, index=get_index(YES_NO, ans.get("criminal_record_certificate_required", "Yes")), horizontal=True)
            va["criminal_record_certificate_provided"] = st.radio("Criminal record cert provided?", YES_NO, index=get_index(YES_NO, ans.get("criminal_record_certificate_provided", "Yes")), horizontal=True)

        # Column 3: SOC Code Tool
        with c3:
            st.markdown("##### 🛡️ SOC Verification")
            st.info(f"""
                **Step 1:** [Open CASCOT Tool]({official_cascot_url})  
                **Step 2:** Search for your 4-digit code.  
                **Step 3:** Enter below.
            """)
            
            st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <a href="{official_cascot_url}" target="_blank" style="text-decoration: none;">
                        <button style="background-color: #1e3a8a; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%;">
                            🔍 Open CASCOT Tool ➜
                        </button>
                    </a>
                </div>
            """, unsafe_allow_html=True)

            va["job_title"] = st.text_input("Verified Job Title", value=ans.get("job_title", ""), placeholder="e.g., Software Developer")
            va["soc_code"] = st.text_input("Verified SOC code", value=ans.get("soc_code", ""), placeholder="e.g., 2134", help="This 4-digit code determines your minimum salary threshold.")
            
            if va["soc_code"] and (not va["soc_code"].isdigit() or len(va["soc_code"]) != 4):
                st.error("⚠️ Invalid Format: SOC codes must be exactly 4 digits.")

    # --- HEALTH & CARE VISA (3-Column Layout) ---
    elif selected_visa == "Health & Care":
        c1, c2, c3 = st.columns(3)
        
        # Column 1: General Details
        with c1:
            va["job_offer_confirmed"] = st.radio("Job offer confirmed?", YES_NO, index=get_index(YES_NO, ans.get("job_offer_confirmed", "Yes")), horizontal=True)
            va["employer_is_licensed_healthcare_sponsor"] = st.radio("Licensed healthcare sponsor?", YES_NO, index=get_index(YES_NO, ans.get("employer_is_licensed_healthcare_sponsor", "Yes")), horizontal=True)
            va["certificate_of_sponsorship_issued"] = st.radio("CoS issued?", YES_NO, index=get_index(YES_NO, ans.get("certificate_of_sponsorship_issued", "Yes")), horizontal=True)
            va["cos_reference_number"] = st.text_input("CoS reference number", value=ans.get("cos_reference_number", ""), help="Reference number from your NHS/Care provider.")

        # Column 2: Specific Requirements
        with c2:
            va["job_is_eligible_healthcare_role"] = st.radio("Eligible healthcare role?", YES_NO, index=get_index(YES_NO, ans.get("job_is_eligible_healthcare_role", "Yes")), horizontal=True)
            va["salary_offered"] = st.number_input("Salary offered (£)", min_value=0, value=ans.get("salary_offered", 0))
            va["meets_healthcare_salary_rules"] = st.radio("Meets salary rules?", YES_NO, index=get_index(YES_NO, ans.get("meets_healthcare_salary_rules", "Yes")), horizontal=True)
            va["professional_registration_required"] = st.radio("Registration required?", YES_NO, index=get_index(YES_NO, ans.get("professional_registration_required", "Yes")), horizontal=True)
            va["professional_registration_provided"] = st.radio("Registration provided?", YES_NO, index=get_index(YES_NO, ans.get("professional_registration_provided", "Yes")), horizontal=True)
            va["english_requirement_met"] = st.radio("English requirement met?", YES_NO, index=get_index(YES_NO, ans.get("english_requirement_met", "Yes")), horizontal=True)

        # Column 3: SOC Verification
        with c3:
            st.markdown("##### 🏥 Role Verification")
            st.info(f"""
                **Find Healthcare SOC Code:**
                1. [Open CASCOT Tool]({official_cascot_url}).
                2. Search role (e.g. 'Nurse').
                3. Enter details below.
            """)
            
            st.markdown(f"""
                <div style="margin-bottom: 15px;">
                    <a href="{official_cascot_url}" target="_blank" style="text-decoration: none;">
                        <button style="background-color: #0d9488; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; font-weight: bold; width: 100%;">
                            🔍 Healthcare Tool ➜
                        </button>
                    </a>
                </div>
            """, unsafe_allow_html=True)

            va["job_title"] = st.text_input("Verified Job Title", value=ans.get("job_title", ""), placeholder="e.g., Senior Care Worker")
            va["soc_code"] = st.text_input("Verified SOC code", value=ans.get("soc_code", ""), placeholder="e.g., 6135", help="The 4-digit code identifying your healthcare role.")
            
            if va["soc_code"] and (not va["soc_code"].isdigit() or len(va["soc_code"]) != 4):
                st.warning("⚠️ Tip: SOC codes are usually 4 digits.")

    # --- GRADUATE VISA (Standard 2-Column) ---
    elif selected_visa == "Graduate Visa":
        c1, c2 = st.columns(2)
        with c1:
            va["currently_in_uk"] = st.radio("Currently in the UK?", YES_NO, index=get_index(YES_NO, ans.get("currently_in_uk", "Yes")), horizontal=True)
            va["current_uk_visa_type"] = st.selectbox("Current Visa Type", ["Student", "Tier 4", "Other"], index=0)
            va["course_completed"] = st.radio("Course completed?", YES_NO, index=get_index(YES_NO, ans.get("course_completed", "Yes")), horizontal=True)
            va["original_cas_reference"] = st.text_input("Original CAS Reference", value=ans.get("original_cas_reference", ""), help="The unique reference number from your Student visa.")
        with c2:
            va["education_provider_is_licensed"] = st.radio("Provider licensed?", YES_NO, index=get_index(YES_NO, ans.get("education_provider_is_licensed", "Yes")), horizontal=True)
            va["provider_reported_completion"] = st.radio("Completion reported to Home Office?", YES_NO, index=get_index(YES_NO, ans.get("provider_reported_completion", "Yes")), horizontal=True)
            va["student_visa_valid"] = st.radio("Student Visa valid?", YES_NO, index=get_index(YES_NO, ans.get("student_visa_valid", "Yes")), horizontal=True)
            va["course_level_completed"] = st.selectbox("Course Level", ["Bachelor's", "Master's", "PhD", "Other"], index=0, help="e.g. Bachelor's or Master's degree.")

    # --- STUDENT VISA (Standard 2-Column) ---
    elif selected_visa == "Student Visa":
        c1, c2 = st.columns(2)
        with c1:
            va["has_cas"] = st.radio("Do you have a CAS?", YES_NO, index=get_index(YES_NO, ans.get("has_cas", "Yes")), horizontal=True)
            va["cas_reference_number"] = st.text_input("CAS Reference Number", value=ans.get("cas_reference_number", ""), help="The 14-digit number provided by your university.")
            va["education_provider_is_licensed"] = st.radio("Provider licensed?", YES_NO, index=get_index(YES_NO, ans.get("education_provider_is_licensed", "Yes")), horizontal=True)
            va["course_level"] = st.selectbox("Course Level", ["RQF Level 3", "RQF 4/5", "Degree (6+)", "Pre-sessional"], index=0, help="RQF level (e.g., Level 7 for Masters).")
            va["course_full_time"] = st.radio("Course full-time?", YES_NO, index=get_index(YES_NO, ans.get("course_full_time", "Yes")), horizontal=True)
        with c2:
            va["course_start_date"] = str(st.date_input("Start Date", value=get_date_val(ans, "course_start_date")))
            va["course_end_date"] = str(st.date_input("End Date", value=get_date_val(ans, "course_end_date")))
            va["course_duration_months"] = st.number_input("Duration (Months)", min_value=1, value=ans.get("course_duration_months", 12))
            va["meets_financial_requirement"] = st.radio("Meets financial reqs?", YES_NO, index=get_index(YES_NO, ans.get("meets_financial_requirement", "Yes")), horizontal=True)
            va["funds_held_for_28_days"] = st.radio("Funds held 28+ days?", YES_NO, index=get_index(YES_NO, ans.get("funds_held_for_28_days", "Yes")), horizontal=True, help="Funds must have been in your account for 28 days prior to application.")

    # --- STANDARD VISITOR (Standard 2-Column) ---
    elif selected_visa == "Standard Visitor":
        c1, c2 = st.columns(2)
        with c1:
            va["purpose_of_visit"] = st.text_input("Purpose of Visit", value=ans.get("purpose_of_visit", ""))
            va["purpose_is_permitted_under_visitor_rules"] = st.radio("Purpose permitted?", YES_NO, index=get_index(YES_NO, ans.get("purpose_is_permitted_under_visitor_rules", "Yes")), horizontal=True)
            va["intended_length_of_stay_months"] = st.number_input("Intended Stay (Months)", min_value=0, max_value=6, value=ans.get("intended_length_of_stay_months", 1))
            va["stay_within_6_months_limit"] = st.radio("Stay < 6 months?", YES_NO, index=get_index(YES_NO, ans.get("stay_within_6_months_limit", "Yes")), horizontal=True)
        with c2:
            va["accommodation_arranged"] = st.radio("Accommodation arranged?", YES_NO, index=get_index(YES_NO, ans.get("accommodation_arranged", "Yes")), horizontal=True)
            va["return_or_onward_travel_planned"] = st.radio("Return travel planned?", YES_NO, index=get_index(YES_NO, ans.get("return_or_onward_travel_planned", "Yes")), horizontal=True)
            va["intends_to_leave_uk_after_visit"] = st.radio("Intend to leave after?", YES_NO, index=get_index(YES_NO, ans.get("intends_to_leave_uk_after_visit", "Yes")), horizontal=True)
            va["sufficient_funds_for_stay"] = st.radio("Sufficient funds?", YES_NO, index=get_index(YES_NO, ans.get("sufficient_funds_for_stay", "Yes")), horizontal=True)

    # Fallback for unrecognized visa types
    else:
        st.warning(f"Configuration for '{selected_visa}' not found. Please go back and select a standard visa type.")

    st.write("##")

    # --- NAVIGATION ---
    b1, b2 = st.columns([1, 1])
    if b1.button("⬅ Back"):
        st.session_state.stage = 3
        st.rerun()
        
    if b2.button("Next: Check Eligibility ➜"):
        st.session_state.visa_answers = va
        st.session_state.stage = 5
        st.rerun()

# ==================================================
# STAGE 5: PROCESSING PLACEHOLDER
# ==================================================

elif st.session_state.stage == 5:
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); color: #f8fafc; }
    .section-title { color: #D4AF37; font-size: 2rem; font-weight: bold; text-align: center; margin-bottom: 20px; }
    .review-item { margin-bottom: 10px; font-size: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 5px; }
    .review-key { color: #38bdf8; font-weight: bold; }
    .review-val { color: #e2e8f0; float: right; text-align: right; }
    
    /* Result Cards */
    .result-box {
        background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 25px; margin-bottom: 20px; border: 1px solid rgba(255,255,255,0.1);
    }
    .badge {
        padding: 10px 20px; border-radius: 8px; font-weight: bold; text-transform: uppercase; 
        display: block; width: 100%; text-align: center; font-size: 1.5rem; margin-bottom: 10px;
    }
    .badge-eligible { background-color: #dcfce7; color: #166534; border: 2px solid #166534; }
    .badge-not-eligible { background-color: #fee2e2; color: #991b1b; border: 2px solid #991b1b; }
    
    /* --- FORCE TEXT VISIBILITY --- */
    .stCheckbox label p { color: #ffffff !important; font-size: 1rem; }
    .stCheckbox label { color: #ffffff !important; }
    
    /* Gold Button */
    div.stButton > button {
        background-color: #D4AF37 !important; color: black !important; border: none; font-weight: bold;
    }
    div.stButton > button:hover { background-color: #FFD700 !important; }
    
    /* General Text Override */
    p, li, h1, h2, h3, h4, h5, h6, span, div { color: #ffffff !important; }
    
    /* Revert specific UI elements */
    .badge-eligible { color: #166534 !important; }
    .badge-not-eligible { color: #991b1b !important; }
    div.stButton > button { color: black !important; }
    </style>
    """, unsafe_allow_html=True)

    st.progress(100)
    st.markdown('<div class="section-title">Review Application</div>', unsafe_allow_html=True)

    # --- REVIEW SECTION (Collapsible) ---
    def display_data(data):
        for k, v in data.items():
            if v:
                clean_key = k.replace("_", " ").title()
                st.markdown(f'<div class="review-item"><span class="review-key">{clean_key}</span><span class="review-val">{v}</span></div>', unsafe_allow_html=True)

    with st.expander("👤 Personal Details", expanded=False):
        p_data = {k: v for k, v in st.session_state.user_profile.items() if k not in ['visa_type', 'purpose', 'travel_date', 'funds', 'length_of_stay', 'criminal_history', 'english_met', 'previous_refusal']}
        display_data(p_data)
        if st.button("Edit Personal Details"):
            st.session_state.stage = 2
            st.rerun()

    with st.expander("✈️ Journey Details", expanded=False):
        j_keys = ['purpose', 'travel_date', 'funds', 'length_of_stay', 'criminal_history', 'english_met', 'previous_refusal']
        j_data = {k: st.session_state.user_profile.get(k) for k in j_keys}
        display_data(j_data)
        if st.button("Edit Journey Details"):
            st.session_state.stage = 3
            st.rerun()

    with st.expander("📋 Visa Specifics", expanded=False):
        display_data(st.session_state.visa_answers)
        if st.button("Edit Visa Details"):
            st.session_state.stage = 4
            st.rerun()

    st.write("##")

    # --- CHECK ELIGIBILITY BUTTON ---
    if st.session_state.eligibility_result is None:
        if st.button("✅ Check Eligibility", use_container_width=True):
            with st.spinner("consulting UK Home Office policies..."):
                full_app = {
                    "Profile": st.session_state.user_profile,
                    "Visa_Specifics": st.session_state.visa_answers
                }
                try:
                    result = run_rag_pipeline(full_app)
                    st.session_state.eligibility_result = result
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

    # --- DISPLAY RESULTS ---
    if st.session_state.eligibility_result:
        res = st.session_state.eligibility_result
        verdict = res.get("verdict", "UNKNOWN")
        confidence = res.get("confidence_score", 0)

        st.write("---")
        
        # 1. VERDICT BADGE
        if verdict == "ELIGIBLE":
            st.markdown(f'<div class="badge badge-eligible">✅ ELIGIBLE</div>', unsafe_allow_html=True)
            st.balloons()
        else:
            st.markdown(f'<div class="badge badge-not-eligible">❌ NOT ELIGIBLE</div>', unsafe_allow_html=True)
        
        # 2. CONFIDENCE SCORE
        st.markdown(f"<h3 style='text-align:center; margin-top:10px;'>AI Confidence Score: {confidence}%</h3>", unsafe_allow_html=True)

        # 3. EXPLANATION (Bullets)
        raw_explanation = res.get('explanation', '')
        points = [p.strip() for p in raw_explanation.split('.') if p.strip()]
        explanation_html = "".join([f"<li>{p}.</li>" for p in points])

        st.markdown(f"""
        <div class="result-box">
            <h3 style="color:#D4AF37; margin-top:0;">Assessment Summary</h3>
            <ul style="margin-left: 20px;">{explanation_html}</ul>
        </div>
        """, unsafe_allow_html=True)

        # 4. ELIGIBLE FLOW
        if verdict == "ELIGIBLE":
            # Checklist
            if res.get("checklist"):
                st.subheader("📝 Document Checklist")
                st.info("Check off the documents you already have:")
                for doc in res.get("checklist", []):
                    st.checkbox(doc, key=doc)
                st.write("##")
            
            # Next Steps
            if res.get("next_steps"):
                st.subheader("🚀 Next Steps")
                for i, step in enumerate(res.get("next_steps", [])):
                    st.markdown(f"**{i+1}.** {step}")
                st.write("##")

            # Permissions (Dos/Donts)
            if res.get("dos") or res.get("donts"):
                st.subheader("🇬🇧 Living in the UK: Permissions & Restrictions")
                
                if res.get("dos"):
                    st.success("**✅ You CAN:**")
                    for item in res.get("dos", []):
                        st.write(f"• {item}")
                        
                if res.get("donts"):
                    st.error("**❌ You CANNOT:**")
                    for item in res.get("donts", []):
                        st.write(f"• {item}")
                st.write("##")
            
            # Future Options
            if res.get("future_options"):
                st.subheader("🔮 Future Options")
                for opt in res.get("future_options", []):
                    st.info(f"👉 {opt}")

        # 5. NOT ELIGIBLE FLOW
        else:
            st.subheader("🔍 Gap Analysis")
            
            if res.get("satisfied_requirements"):
                st.success("**✅ Requirements Met**")
                for req in res.get("satisfied_requirements", []):
                    st.write(f"• {req}")
            
            if res.get("unsatisfied_requirements"):
                st.error("**❌ Requirements Missing**")
                for req in res.get("unsatisfied_requirements", []):
                    st.write(f"• {req}")

            st.write("##")
            
            st.markdown("""
            <div class="result-box" style="border-left: 5px solid #ef4444;">
                <h3 style="color: #ef4444; margin-top:0;">💡 How to Become Eligible</h3>
                <p>{}</p>
            </div>
            """.format(res.get("remedy", "Review the missing requirements above.")), unsafe_allow_html=True)

        st.write("---")
        if st.button("Start New Application", type="primary", use_container_width=True):
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()