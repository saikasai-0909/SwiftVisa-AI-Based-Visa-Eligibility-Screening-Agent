import streamlit as st
import base64
import datetime
from comparison_tool import compare_visa_types, create_comparison_summary, get_best_match
from document_checklist import get_checklist_for_visa
from pdf_export import generate_eligibility_pdf
from retriever import load_retriever, load_llm, rag_answer
from eligibility_result import parse_eligibility_result
from ui.common import render_common
from ui.student import render as student_ui
from ui.graduate import render as graduate_ui
from ui.skilled import render as skilled_ui
from ui.health import render as health_ui
from ui.visitor import render as visitor_ui


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Visa Eligibility Screening",
    page_icon="🛂",
    layout="wide"
)

st.markdown("<style>header {visibility: hidden;}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------
# VISA KEY MAPPING
# ---------------------------------------------------------
VISA_KEY_MAP = {
    "skilled": "work",
    "health": "healthcare_worker",
}

DISPLAY_LABELS = {
    "student": "Student Visa",
    "graduate": "Graduate Visa",
    "work": "Skilled Worker Visa",
    "healthcare_worker": "Health & Care Visa",
    "visitor": "Visitor Visa",
}

def to_data_key(ui_key: str) -> str:
    return VISA_KEY_MAP.get(ui_key, ui_key)

def to_label(data_key: str) -> str:
    return DISPLAY_LABELS.get(data_key, data_key.replace("_", " ").title())


# ---------------------------------------------------------
# LOAD IMAGES
# ---------------------------------------------------------
def load_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = load_base64("assets/visa bg1.jpeg")
merged_img = load_base64("assets/Merged.jpeg")


# ---------------------------------------------------------
# GLOBAL CSS
# ---------------------------------------------------------
st.markdown(f"""
<style>

.stApp {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    inset: 0;
    background: rgba(2, 6, 23, 0.88);
    z-index: -1;
}}

h1, h2, h3 {{
    color: #ffffff;
}}

input, textarea {{
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
}}

.stForm button {{
    background-color: #0f172a !important;
    color: #ffffff !important;
    border-radius: 10px;
}}

/* ---- VISA CARD BUTTON STYLE ---- */
button[kind="secondary"] {{
    background: #ffffff !important;
    border-radius: 18px !important;
    padding: 22px !important;
    height: 160px;
    text-align: left;
    box-shadow: 0 20px 40px rgba(0,0,0,0.12);
    border: none !important;
    font-size: 16px;
}}

button[kind="secondary"]:hover {{
    transform: translateY(-6px);
    box-shadow: 0 25px 60px rgba(0,0,0,0.18);
}}

button[kind="secondary"] * {{
    color: #0f172a !important;
}}

button[kind="secondary"] em {{
    color: #64748b !important;
    font-style: normal;
}}
/* Make select / dropdown fields white */
div[data-baseweb="select"] > div {{
    background-color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #cbd5e1 !important;
}}

div[data-baseweb="select"] span {{
    color: #0f172a !important;
    font-weight: 500;
}}

/* Dropdown options panel */
div[role="listbox"] {{
    background-color: #ffffff !important;
}}

/* Options text */
div[role="option"] {{
    color: #0f172a !important;
}}
/* FIX selected value text in selectbox */
div[data-baseweb="select"] div[aria-selected="true"],
div[data-baseweb="select"] div {{
    color: #0f172a !important;
}}

/* Fix placeholder / selected text */
div[data-baseweb="select"] span {{
    color: #0f172a !important;
    font-weight: 500;
}}

/* Fix single-value select text */
div[data-baseweb="select"] > div > div > div {{
    color: #0f172a !important;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "visa" not in st.session_state:
    st.session_state.visa = None

if "result" not in st.session_state:
    st.session_state.result = None

if "applicant_data" not in st.session_state:
    st.session_state.applicant_data = None


# ---------------------------------------------------------
# INIT RAG (Lazy Loading - Load Only When Needed)
# ---------------------------------------------------------
@st.cache_resource
def get_retriever():
    print("[STARTUP] Loading embedding model (first time only, takes 30-60 seconds)...")
    return load_retriever()

@st.cache_resource
def get_llm():
    print("[STARTUP] Connecting to LM Studio...")
    return load_llm()

# Don't load at startup - wait for user to select visa type
retriever = None
llm = None


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
logo_col, title_col = st.columns([0.8, 9.2], gap="small")

with logo_col:
    st.markdown("<div style='padding-top:8px;'>", unsafe_allow_html=True)
    st.image("assets/logo.jpeg", width=95)
    st.markdown("</div>", unsafe_allow_html=True)

with title_col:
    st.markdown(
        """
        <h3 style="margin-bottom:4px;">Visa Eligibility Screening</h3>
        <span style="color:#e2e8f0;">AI-based UK Visa Eligibility Assistant</span>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------
left, right = st.columns([2.2, 1.8], gap="large")

with left:
    st.markdown(
        """
        <h1>Your Global Journey Starts with a Simple Step.</h1>

        <p style="font-size:20px;line-height:1.6;color:#e5e7eb;max-width:640px;">
            Our platform helps applicants quickly understand their eligibility
            for different UK visa categories before applying.
        </p>
        <p style="font-size:18px;line-height:1.6;color:#e5e7eb;max-width:640px;">
            Take the guesswork out of immigration. Our intelligent screening system
            evaluates your profile against the latest regulations to find your
            fastest path to a successful visa application.
        </p>
        """,
        unsafe_allow_html=True
    )

with right:
    st.markdown(
        f"""
        <div style="display:flex;justify-content:center;align-items:center;height:100%;">
            <img src="data:image/png;base64,{merged_img}"
                 style="max-height:340px;width:auto;max-width:100%;border-radius:18px;box-shadow:0 10px 30px rgba(0,0,0,0.2);">
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# VISA SELECTION (NO BLACK BUTTONS)
# ---------------------------------------------------------
st.markdown("<div style='margin-top:30px;'></div>", unsafe_allow_html=True)
st.markdown("## Select Visa Type")
st.caption("Choose a visa category to begin your eligibility screening")

c1, c2, c3, c4, c5 = st.columns(5)

def visa_card(col, icon, title, key, visa_value):
    with col:
        if st.button(
            f"{icon}\n\n**{title}**\n\n_Check requirements_",
            key=key,
            use_container_width=True
        ):
            st.session_state.visa = visa_value
            st.session_state.result = None


visa_card(c1, "🎓", "Student Visa", "student_btn", "student")
visa_card(c2, "🎓", "Graduate Visa", "graduate_btn", "graduate")
visa_card(c3, "💼", "Skilled Worker Visa", "skilled_btn", "skilled")
visa_card(c4, "🏥", "Health & Care Visa", "health_btn", "health")
visa_card(c5, "✈️", "Visitor Visa", "visitor_btn", "visitor")


# ---------------------------------------------------------
# FORM (APPEARS BELOW CARDS)
# ---------------------------------------------------------
if st.session_state.visa:

    st.markdown("## 📝 Applicant Details")

    with st.form("eligibility_form"):

        st.subheader("Common Details")
        common_data = render_common()

        st.subheader("Visa-Specific Details")

        if st.session_state.visa == "student":
            visa_data = student_ui()
        elif st.session_state.visa == "graduate":
            visa_data = graduate_ui()
        elif st.session_state.visa == "skilled":
            visa_data = skilled_ui()
        elif st.session_state.visa == "health":
            visa_data = health_ui()
        else:
            visa_data = visitor_ui()

        submitted = st.form_submit_button("Check Eligibility")

        if submitted:
            # Validate passport dates
            if common_data.get('passport_expiry_date') <= common_data.get('passport_issue_date'):
                st.error("❌ Passport expiry date must be after issue date. Please correct the dates.")
                st.stop()
            
            # Check if passport is expired
            if common_data.get('passport_expiry_date') < datetime.date.today():
                st.error("❌ Passport has expired. A valid passport is required for visa application.")
                st.stop()
            
            # Basic required-field check to prevent empty submissions
            combined_data = {**common_data, **visa_data}
            missing_fields = []

            for key, value in combined_data.items():
                if isinstance(value, str) and not value.strip():
                    missing_fields.append(key)
                elif isinstance(value, (int, float)) and value == 0:
                    missing_fields.append(key)

            if missing_fields:
                pretty_names = [field.replace("_", " ").title() for field in missing_fields]
                st.warning(f"Missing fields: {', '.join(pretty_names)}. Please complete them before checking eligibility.")
                st.stop()

            with st.spinner("Analyzing visa policy documents..."):
                # Build detailed question with emphasis on critical checks
                critical_flags = []
                if common_data.get('criminal_history') == 'Yes':
                    critical_flags.append("CRIMINAL HISTORY PRESENT")
                if common_data.get('previous_uk_refusal') == 'Yes':
                    critical_flags.append("PREVIOUS UK VISA REFUSAL")
                if common_data.get('english_requirement_met') == 'No':
                    critical_flags.append("ENGLISH REQUIREMENT NOT MET")
                
                critical_section = f"\nCRITICAL ALERTS: {', '.join(critical_flags)}\n" if critical_flags else ""
                
                detailed_question = f"""
Assess UK {st.session_state.visa} visa eligibility. Check ALL policy requirements including criminal history, visa refusal history, financial requirements, and English language requirements.
{critical_section}
APPLICANT INFORMATION:
Full Name: {common_data.get('full_name', 'N/A')}
Nationality: {common_data.get('nationality', 'N/A')}
Date of Birth: {common_data.get('dob', 'N/A')}
Passport Expiry Date: {common_data.get('passport_expiry_date', 'N/A')}
Current Location: {common_data.get('current_location', 'N/A')}

BACKGROUND & COMPLIANCE:
Criminal History: {common_data.get('criminal_history', 'N/A')}
Previous UK Visa Refusal: {common_data.get('previous_uk_refusal', 'N/A')}
English Language Requirement Met: {common_data.get('english_requirement_met', 'N/A')}

FINANCIAL DETAILS:
Funds Available: £{common_data.get('funds_available', 0)}

TRAVEL DETAILS:
Purpose: {common_data.get('purpose_of_visit', 'N/A')}
Intended Start Date: {common_data.get('intended_start_date', 'N/A')}
Length of Stay: {common_data.get('length_of_stay', 'N/A')}

{st.session_state.visa.upper()} VISA REQUIREMENTS:
{chr(10).join(f'{k.replace("_", " ").title()}: {v}' for k, v in visa_data.items())}

Based on UK visa policy documents, assess if this applicant meets ALL requirements. If ANY requirement is not met, the applicant is NOT ELIGIBLE. Provide bullet-point assessment for each requirement.
"""
                # Load models only when needed (on first form submission)
                with st.spinner("⏳ It takes (2-3 minutes) please wait..."):
                    if retriever is None:
                        retriever = get_retriever()
                    if llm is None:
                        llm = get_llm()
                    
                    llm_output = rag_answer(
                        detailed_question,
                        retriever,
                        llm,
                        visa_type=st.session_state.visa,
                        critical_flags=critical_flags
                    )
                    st.session_state.result = parse_eligibility_result(llm_output, combined_data)
                    st.session_state.applicant_data = combined_data


# ---------------------------------------------------------
# RESULT
# ---------------------------------------------------------
if st.session_state.result:
    parsed = st.session_state.result

    st.markdown("## 📊 Eligibility Result")

    result_tab, checklist_tab, compare_tab, export_tab = st.tabs([
        "Result",
        "Document Checklist",
        "Visa Comparison",
        "Export PDF",
    ])

    with result_tab:
        if parsed["status"] == "ELIGIBLE":
            st.success("✅ ELIGIBLE")
        else:
            st.error("❌ NOT ELIGIBLE")

        st.markdown("### Reason")
        reason_text = parsed["reason"]
        if "•" in reason_text or "-" in reason_text[:50]:
            st.markdown(reason_text)
        else:
            st.write(reason_text)

    with checklist_tab:
        data_key = to_data_key(st.session_state.visa)
        checklist = get_checklist_for_visa(data_key)

        st.markdown("### Required Documents")
        if not checklist:
            st.info("No checklist available for this visa type yet.")
        else:
            for category, documents in checklist.items():
                st.markdown(f"**{category}**")
                for idx, document in enumerate(documents):
                    st.checkbox(
                        document,
                        key=f"{data_key}-{category}-{idx}",
                        value=False,
                    )

    with compare_tab:
        st.markdown("### Compare Visa Options")
        ui_options = ["student", "graduate", "skilled", "health", "visitor"]
        selected_ui = st.multiselect(
            "Select visa types to compare",
            options=ui_options,
            default=[st.session_state.visa] if st.session_state.visa else [],
            format_func=lambda key: to_label(to_data_key(key)),
        )

        if selected_ui:
            data_keys = [to_data_key(key) for key in selected_ui]
            comparison_df = compare_visa_types(data_keys)
            if comparison_df.empty:
                st.info("No comparison data available for the selected visas.")
            else:
                st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("Select at least one visa type to compare.")

        if st.session_state.applicant_data:
            recommendations = get_best_match(st.session_state.applicant_data)
            if recommendations:
                rec_labels = ", ".join(to_label(rec) for rec in recommendations)
                st.markdown(f"**Recommended visas for your profile:** {rec_labels}")
                summary_text = create_comparison_summary(st.session_state.applicant_data)
                st.text_area(
                    "Recommendation summary",
                    summary_text,
                    height=220,
                    disabled=True,
                )

    with export_tab:
        st.markdown("### Export Assessment")
        if st.session_state.applicant_data:
            pdf_bytes = generate_eligibility_pdf(
                st.session_state.applicant_data,
                parsed,
                to_label(to_data_key(st.session_state.visa)),
            )
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"visa_eligibility_{st.session_state.visa}.pdf",
                mime="application/pdf",
            )
        else:
            st.info("Submit your details to enable PDF export.")