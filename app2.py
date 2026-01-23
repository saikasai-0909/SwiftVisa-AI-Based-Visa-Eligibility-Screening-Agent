import streamlit as st
from datetime import date

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SwiftVisa | AI Eligibility Checker",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SAFETY: INITIALIZE VARIABLES ---
stay_months = "N/A"
duration = "N/A"
funds_total = "N/A"
english = "N/A"
passport_issue = None
passport_expiry = None
travel_date = None
criminal = "No"
refusals = "No"
full_name = ""
nationality = ""
email = ""
phone = ""
current_address = ""
location = "Outside UK"

# --- HELPER FUNCTION: FIX CHECKBOX TRANSLATION ---
# Critical: Converts True/False to "Yes"/"No" so the AI understands it.
def bool_to_text(value):
    return "Yes" if value else "No"

# --- BACKEND CONNECTION ---
try:
    from visa_agent import get_agent_response
except ImportError:
    # Fallback if file is missing (prevents crash, shows clear error)
    def get_agent_response(key, prompt):
        return "⚠️ Error: Backend file (visa_agent.py) not found. Please ensure it is in the same folder."

# --- 2. SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/a/ae/Flag_of_the_United_Kingdom.svg", width=80)
    
    st.markdown("""
        <div style="margin-top: 10px; text-align: left;">
            <h2 style="margin: 0; padding: 0; font-size: 22px; color: #F8FAFC;">SwiftVisa AI</h2>
            <p style="margin: 0; font-size: 14px; opacity: 0.8; color: #F8FAFC;">Immigration Assistant</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Dark Mode Toggle
    dark_mode = st.toggle("🌙 Dark Mode", value=False)
    
    st.markdown("---")
    
    st.markdown("### 📋 How It Works")
    st.info("""
    **Step 1: Choose Visa**
    Select your target visa category.
    
    **Step 2: Enter Details**
    Fill out the dynamic form.
    
    **Step 3: AI Analysis**
    Our AI scans the Home Office rules.
    """)
    st.caption("© 2026 SwiftVisa Project")

# --- 3. DYNAMIC THEME ENGINE ---
if dark_mode:
    bg_image = "https://images.unsplash.com/photo-1475274047050-1d0c0975c63e?auto=format&fit=crop&w=2000&q=80"
    overlay_color = "rgba(15, 23, 42, 0.9)"
    card_bg = "#1E293B"
    text_color = "#F8FAFC"
    subtitle_color = "#94A3B8"
    input_bg = "#F1F5F9" 
    input_text = "#000000" 
    border_color = "#475569"
    shadow_color = "rgba(0,0,0,0.5)"
else:
    bg_image = "https://images.unsplash.com/photo-1513002749550-c59d786b8e6c?q=80&w=2574&auto=format&fit=crop"
    overlay_color = "rgba(255, 255, 255, 0.3)"
    card_bg = "rgba(255, 255, 255, 0.92)"
    text_color = "#0F172A"
    subtitle_color = "#475569"
    input_bg = "#F8FAFC"
    input_text = "#0F172A"
    border_color = "#E2E8F0"
    shadow_color = "rgba(0,0,0,0.08)"

# --- 4. CSS INJECTION (FIXED FOR VISIBILITY) ---
st.markdown(f"""
    <style>
    .stApp {{ background-image: url('{bg_image}'); background-size: cover; background-position: center; background-attachment: fixed; }}
    .stApp::before {{ content: ""; position: absolute; top: 0; left: 0; width: 100%; height: 100%; background-color: {overlay_color}; pointer-events: none; z-index: -1; }}
    h1, h2, h3, label, .stMarkdown, p, li {{ color: {text_color} !important; font-family: 'Segoe UI', sans-serif; }}
    section[data-testid="stSidebar"] {{ background-color: #0F172A; }}
    section[data-testid="stSidebar"] * {{ color: #F8FAFC !important; }}
    [data-testid="stForm"] {{ background-color: {card_bg}; padding: 30px; border-radius: 12px; box-shadow: 0 4px 20px {shadow_color}; border: 1px solid {border_color}; }}
    .stTextInput>div>div, .stSelectbox>div>div, .stDateInput>div>div {{ background-color: {input_bg} !important; border: 1px solid {border_color}; color: {input_text} !important; border-radius: 6px; }}
    div[data-baseweb="input"] input {{ color: {input_text} !important; -webkit-text-fill-color: {input_text} !important; }}
    div[data-baseweb="select"] > div {{ color: {input_text} !important; }}
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[data-baseweb="menu"] {{ background-color: #FFFFFF !important; }}
    li[data-baseweb="option"], li[role="option"] {{ background-color: #FFFFFF !important; color: #000000 !important; }}
    li[data-baseweb="option"] div {{ color: #000000 !important; }}
    li[data-baseweb="option"]:hover {{ background-color: #F1F5F9 !important; color: #000000 !important; }}
    div[data-testid="stFormSubmitButton"] button {{ width: 100%; background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important; color: #FFFFFF !important; font-weight: 700 !important; padding: 12px; border-radius: 8px; border: none !important; }}
    
    /* --- CRITICAL VISIBILITY FIX --- */
    /* Forces text inside Success/Error boxes to be Dark Red/Green, ignoring global white text rule */
    .success-box, .success-box h3, .success-box p, .success-box li {{ 
        background-color: #ECFDF5 !important; 
        color: #064E3B !important; 
        border-color: #10B981 !important;
        border-width: 1px;
        border-style: solid;
    }}
    .error-box, .error-box h3, .error-box p, .error-box li {{ 
        background-color: #FEF2F2 !important; 
        color: #7F1D1D !important; 
        border-color: #EF4444 !important;
        border-width: 1px;
        border-style: solid;
    }}
    .success-box, .error-box {{
        padding: 20px; 
        border-radius: 8px; 
    }}
    </style>
""", unsafe_allow_html=True)

# --- 5. HEADER ---
col_h1, col_h2, col_h3 = st.columns([1, 8, 1])
with col_h2:
    st.markdown("<h1 style='text-align: center;'>SwiftVisa : UK Visa Eligibility Checker</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; color: {subtitle_color}; opacity: 0.9; font-weight: 500;'>Official 2025/26 Home Office Regulation Checker</div>", unsafe_allow_html=True)

# --- 6. VISA SELECTION ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    visa_category = st.selectbox(
        "Select your visa path:",
        ["Select a category...", "Student Visa", "Graduate Visa", "Skilled Worker Visa", "Health & Care Visa", "Visitor Visa"]
    )

# --- 7. MAIN LOGIC ---
if visa_category == "Select a category...":
    # LANDING PAGE
    st.markdown("<br>", unsafe_allow_html=True)
    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image("https://images.unsplash.com/photo-1513635269975-59663e0ac1ad?auto=format&fit=crop&w=600&q=80", use_container_width=True)
            st.markdown("### 🎓 Education")
            st.caption("Student & Graduate Routes")
        with c2:
            st.image("https://images.unsplash.com/photo-1520986606214-8b456906c813?auto=format&fit=crop&w=600&q=80", use_container_width=True)
            st.markdown("### 🎡 Tourism")
            st.caption("Standard Visitor & Transit")
        with c3:
            st.image("https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?auto=format&fit=crop&w=600&q=80", use_container_width=True)
            st.markdown("### 💼 Business")
            st.caption("Skilled Worker & Health Care")
    
    st.markdown("<br>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1: st.info("✅ **Latest Rules 2025**")
    with f2: st.info("⚡ **AI Powered Analysis**")
    with f3: st.info("🔒 **Secure Processing**")

else:
    # APPLICATION FORM
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.form("eligibility_form"):
        st.markdown(f"### 📝 Application: {visa_category}")
        st.markdown("---")

        # --- 1. Personal & Passport Details ---
        st.markdown("#### 👤 1. Personal & Passport Details")
        c_1, c_2, c_3 = st.columns(3)
        with c_1:
            full_name = st.text_input("Full Name (as per passport)")
            nationality = st.selectbox("Nationality", ["Select Country...", "India", "Nigeria", "China", "United States", "Pakistan", "Bangladesh", "United Kingdom", "Canada", "Other"])
            passport_num = st.text_input("Passport Number")
            email = st.text_input("Email Address")
        with c_2:
            dob = st.date_input("Date of Birth", min_value=date(1950,1,1))
            passport_issue = st.date_input("Passport Issue Date")
            passport_expiry = st.date_input("Passport Expiry Date")
            phone = st.text_input("Phone Number")
        with c_3:
            location = st.selectbox("Country of Application", ["Outside UK", "Inside UK"])
            travel_date = st.date_input("Intended Travel/Start Date")
            english = st.selectbox("English Language Met?", ["Yes - Native", "Yes - Degree", "Yes - SELT", "No"])
        
        current_address = st.text_area("Current Address")
        st.markdown("---")
        
        # --- 2. Specific Requirements ---
        st.markdown(f"#### 💼 2. {visa_category} Requirements")
        user_story = ""

        if "Student" in visa_category:
            sc1, sc2 = st.columns(2)
            with sc1:
                has_cas = st.radio("Do you have a CAS?", ["Yes", "No"], horizontal=True)
                cas_ref = st.text_input("CAS Reference Number")
                course_level = st.selectbox("Course Level", ["Bachelor/Master (RQF 6+)", "PhD", "Below Degree"])
                full_time = st.radio("Is course Full-Time?", ["Yes", "No"], horizontal=True)
                funds_avail = st.text_input("Funds Available (£)")
            with sc2:
                course_start = st.date_input("Course Start Date")
                course_end = st.date_input("Course End Date")
                school_licensed = st.checkbox("Is Education Provider Licensed?")
                funds_held = st.checkbox("Funds held for 28 consecutive days?")
            
            # CORRECTED: Wraps checkboxes in bool_to_text()
            user_story = f"Visa: Student. Name: {full_name}. Nationality: {nationality}. Location: {location}. CAS: {has_cas}. CAS Ref: {cas_ref}. Course Level: {course_level}. Full Time: {full_time}. Provider Licensed: {bool_to_text(school_licensed)}. Funds Held 28 Days: {bool_to_text(funds_held)}. English: {english}."

        elif "Graduate" in visa_category:
            gc1, gc2 = st.columns(2)
            with gc1:
                currently_in_uk = st.radio("Are you currently in the UK?", ["Yes", "No"], horizontal=True)
                current_visa = st.selectbox("Current UK Visa Type", ["Student / Tier 4", "Other"])
                course_completed = st.radio("Course Completed?", ["Yes", "No"], horizontal=True)
                course_level_comp = st.selectbox("Level Completed", ["Bachelor", "Master", "PhD", "Other"])
            with gc2:
                uni_licensed = st.checkbox("Was Provider Licensed?")
                reported = st.checkbox("Did Provider report completion to Home Office?")
                orig_cas = st.text_input("Original CAS Reference (from Student Visa)")
                visa_valid = st.checkbox("Is current Student Visa still valid?")

            # CORRECTED: Wraps checkboxes in bool_to_text()
            user_story = f"Visa: Graduate. Name: {full_name}. In UK: {currently_in_uk}. Current Visa: {current_visa}. Completed: {course_completed}. Level: {course_level_comp}. Licensed: {bool_to_text(uni_licensed)}. Reported to HO: {bool_to_text(reported)}. Original CAS: {orig_cas}. Visa Valid: {bool_to_text(visa_valid)}."

        elif "Skilled Worker" in visa_category:
            wc1, wc2 = st.columns(2)
            with wc1:
                job_offer = st.radio("Job Offer Confirmed?", ["Yes", "No"], horizontal=True)
                employer_licensed = st.checkbox("Employer is Licensed Sponsor?")
                cos_issued = st.checkbox("CoS Issued?")
                cos_ref = st.text_input("CoS Reference Number")
                job_title = st.text_input("Job Title")
            with wc2:
                soc_code = st.text_input("SOC Code")
                eligible_occ = st.checkbox("Is Job an Eligible Occupation?")
                salary = st.number_input("Salary Offered (£)", step=1000)
                crim_cert = st.radio("Criminal Record Certificate Provided?", ["Yes", "No/Not Required"], horizontal=True)

            # CORRECTED: Wraps checkboxes in bool_to_text()
            user_story = f"Visa: Skilled Worker. Name: {full_name}. Job Offer: {job_offer}. Sponsor Licensed: {bool_to_text(employer_licensed)}. CoS Issued: {bool_to_text(cos_issued)}. Job: {job_title}. Eligible Job: {bool_to_text(eligible_occ)}. Salary: {salary}. Criminal Cert: {crim_cert}. English: {english}."

        elif "Health" in visa_category:
            hc1, hc2 = st.columns(2)
            with hc1:
                job_offer = st.radio("Job Offer Confirmed?", ["Yes", "No"], horizontal=True)
                employer_licensed = st.checkbox("Employer is Licensed Health Sponsor?")
                cos_ref = st.text_input("CoS Reference Number")
                job_title = st.text_input("Job Title")
                soc_code = st.text_input("SOC Code")
            with hc2:
                eligible_role = st.checkbox("Is Job an Eligible Health Role?")
                salary = st.number_input("Salary Offered (£)", step=1000)
                prof_reg = st.radio("Professional Registration Provided?", ["Yes", "No/NA"], horizontal=True)
            
            # CORRECTED: Wraps checkboxes in bool_to_text()
            user_story = f"Visa: Health & Care. Name: {full_name}. Job Offer: {job_offer}. Licensed Sponsor: {bool_to_text(employer_licensed)}. Eligible Role: {bool_to_text(eligible_role)}. Salary: {salary}. Prof Reg: {prof_reg}. English: {english}."

        elif "Visitor" in visa_category:
            vc1, vc2 = st.columns(2)
            with vc1:
                purpose = st.selectbox("Purpose of Visit", ["Tourism", "Business", "Family", "Medical"])
                stay_months = st.slider("Intended Stay (Months)", 1, 12, 1)
                accommodation = st.checkbox("Accommodation Arranged?")
                onward_travel = st.checkbox("Return/Onward Travel Planned?")
            with vc2:
                funds_visit = st.radio("Sufficient Funds?", ["Yes", "No"], horizontal=True)
                intent_leave = st.checkbox("Intend to leave UK after visit?")
            
            # CORRECTED: Wraps checkboxes in bool_to_text()
            user_story = f"Visa: Visitor. Name: {full_name}. Purpose: {purpose}. Stay: {stay_months} months. Accommodation: {bool_to_text(accommodation)}. Onward Travel: {bool_to_text(onward_travel)}. Funds: {funds_visit}. Intent to Leave: {bool_to_text(intent_leave)}."

        st.markdown("---")
        
        # --- 3. Declarations ---
        st.markdown("#### ⚖️ 3. Declarations")
        hc1, hc2 = st.columns(2)
        with hc1:
            refusals = st.radio("Previous UK Visa Refusals?", ["No", "Yes"], horizontal=True)
        with hc2:
            criminal = st.radio("Criminal History Declaration?", ["No", "Yes"], horizontal=True)
        
        user_story += f" Previous Refusals: {refusals}. Criminal Record: {criminal}."

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🚀 Check Eligibility Now")

    # --- RESULTS ---
    if submit_btn:
        if not full_name or nationality == "Select Country...":
            st.warning("⚠️ Please fill in your Full Name and Nationality.")
        else:
            with st.spinner("🤖 AI is analyzing Home Office Rules..."):
                key_map = {
                    "Student Visa": "student_visa",
                    "Graduate Visa": "graduate_visa",
                    "Skilled Worker Visa": "skilled_worker_visa",
                    "Health & Care Visa": "healthcare_visa",
                    "Visitor Visa": "visitor_visa"
                }
                backend_key = key_map.get(visa_category, "general")
                
                try:
                    final_response = get_agent_response(backend_key, user_story)
                    
                    # --- CRITICAL DISPLAY FIX: Convert newlines to HTML breaks ---
                    # This ensures the bullet points are visible in the HTML box
                    formatted_response = final_response.replace('\n', '<br>')
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    with st.container():
                         st.markdown(f"""<div style='background-color: {card_bg}; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px {shadow_color};'>
                                <h3>📋 Assessment Result</h3>
                                </div>""", unsafe_allow_html=True)
                    
                         if "NOT ELIGIBLE" in final_response.upper():
                            st.markdown(f"<div class='error-box'><h3>❌ Likely Not Eligible</h3>{formatted_response}</div>", unsafe_allow_html=True)
                         elif "ELIGIBLE" in final_response.upper():
                            st.markdown(f"<div class='success-box'><h3>✅ Likely Eligible</h3>{formatted_response}</div>", unsafe_allow_html=True)
                         else:
                            st.info(formatted_response)
                except Exception as e:
                    st.error(f"System Error: {str(e)}")