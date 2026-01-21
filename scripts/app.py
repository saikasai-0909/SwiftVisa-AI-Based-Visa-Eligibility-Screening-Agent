import streamlit as st
import sys
import os
from datetime import date
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from rag_pipeline import SwiftVisaRAG
from utils.form_fields import COMMON_FIELDS, VISA_FIELDS_MAP, VISA_DISPLAY_NAMES

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="SwiftVisa – UK Visa Eligibility Screening",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL STYLES
# --------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: #1F2933;
}

/* Main background */
.stMainBlockContainer {
    background-color: #FFFFFF;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #E6F0FF;
    background-image: linear-gradient(135deg, #E6F0FF 0%, #F0E8FF 100%);
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    padding-top: 2rem;
}

/* Sidebar text */
[data-testid="stSidebar"] .stMarkdown {
    color: #1F2933;
}

[data-testid="stSidebar"] .stMarkdown h1, 
[data-testid="stSidebar"] .stMarkdown h2, 
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 {
    color: #002CA6;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #4B5563;
    font-size: 13px;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #C80730;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] [data-testid="stSelectbox"] {
    accent-color: #002CA6;
}

.header-container {
    position: relative;
    margin-bottom: 2rem;
}

.header-content {
    position: relative;
    z-index: 2;
}

.main-title {
    font-size: 32px;
    font-weight: 700;
    margin: 0 0 0.25rem 0;
    color: #002CA6;
}

.subtitle {
    font-size: 15px;
    color: #C80730;
    margin-top: 0.25rem;
    font-weight: 500;
}

.section-title {
    font-size: 20px;
    font-weight: 600;
    margin: 1.5rem 0 0.75rem 0;
    color: #002CA6;
}

.subsection-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #C80730;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

.notice {
    background-color: #EFEBF9;
    border-left: 3px solid #002CA6;
    padding: 1rem 1.25rem;
    font-size: 13px;
    line-height: 1.6;
    color: #1F2933;
    margin-top: 1rem;
}

.visa-card {
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    border: 2px solid #E7D180;
    border-radius: 8px;
    overflow: hidden;
    transition: all 0.3s ease;
    cursor: pointer;
    min-height: 220px;
    display: flex;
    align-items: flex-end;
    position: relative;
}

.visa-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(to bottom, rgba(0,44,166,0) 40%, rgba(0,44,166,0.8) 100%);
    z-index: 1;
}

.visa-card:hover {
    border-color: #C80730;
    box-shadow: 0 4px 16px rgba(200, 7, 48, 0.2);
    transform: translateY(-2px);
}

.visa-card-content {
    padding: 1.5rem;
    position: relative;
    z-index: 2;
    width: 100%;
}

.visa-card-title {
    font-size: 16px;
    font-weight: 700;
    color: #EFEBF9;
    margin: 0 0 0.5rem 0;
}

.visa-card-desc {
    font-size: 13px;
    color: #F0F0F0;
    line-height: 1.5;
    margin: 0;
}

.result-box {
    border-left: 4px solid;
    padding: 1.5rem;
    margin-top: 1.5rem;
    background-color: #FFFFFF;
    border-radius: 4px;
}

.result-eligible { border-color: #059669; background-color: #ECFDF5; }
.result-warning { border-color: #E7D180; background-color: #FFFBF0; }
.result-error { border-color: #DC2626; background-color: #FEE2E2; }

.result-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.result-eligible .result-title { color: #059669; }
.result-warning .result-title { color: #E7D180; }
.result-error .result-title { color: #DC2626; }

.result-text {
    font-size: 14px;
    line-height: 1.6;
    color: #1F2933;
}

.requirements-box {
    background-color: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 1.25rem;
    margin-top: 1.5rem;
}

.requirements-title {
    font-size: 14px;
    font-weight: 600;
    color: #1F2933;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.requirements-title i {
    color: #C80730;
}

.requirements-list {
    margin: 0;
    padding-left: 1.5rem;
}

.requirements-list li {
    margin-bottom: 0.5rem;
    color: #4B5563;
    line-height: 1.5;
}

.requirements-list li:last-child {
    margin-bottom: 0;
}

.info-note {
    background-color: #EFF6FF;
    border: 1px solid #93C5FD;
    border-radius: 6px;
    padding: 1rem;
    margin-top: 1rem;
    font-size: 13px;
    color: #1E40AF;
}

.critical-issues {
    background-color: #FEE2E2;
    border-left: 3px solid #DC2626;
    padding: 1rem;
    margin-top: 1rem;
    border-radius: 4px;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# UTILITY: Image cropping
# --------------------------------------------------
def crop_header_image(image_path, target_width=1400, target_height=200):
    """Crop image to wide rectangle for header"""
    try:
        img = Image.open(image_path)
        
        # Crop to center with target aspect ratio
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height
        
        if img_ratio > target_ratio:
            # Image is wider, crop sides
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            # Image is taller, crop top/bottom
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))
        
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        st.warning(f"Could not load header image: {e}")
        return None

def crop_card_image(image_path, target_width=200, target_height=120):
    """Crop image to card size"""
    try:
        img = Image.open(image_path)
        
        # Crop to center with target aspect ratio
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height
        
        if img_ratio > target_ratio:
            new_width = int(img.height * target_ratio)
            left = (img.width - new_width) // 2
            img = img.crop((left, 0, left + new_width, img.height))
        else:
            new_height = int(img.width / target_ratio)
            top = (img.height - new_height) // 2
            img = img.crop((0, top, img.width, top + new_height))
        
        img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        return None

# --------------------------------------------------
# ELIGIBILITY PARSER - DIRECT FROM LLM RESPONSE
# --------------------------------------------------
def parse_eligibility(evaluation_text):
    """Parse evaluation directly from LLM response by looking for 'verdict:' line"""
    evaluation_lower = evaluation_text.lower()
    
    # First, try to find the verdict line
    lines = evaluation_text.split('\n')
    verdict_line = None
    verdict_value = None
    
    for line in lines:
        if line.lower().startswith('verdict:'):
            verdict_line = line
            # Extract the verdict value
            parts = line.split(':', 1)
            if len(parts) > 1:
                verdict_value = parts[1].strip()
            break
    
    # If we found a verdict line, use it
    if verdict_value:
        if 'eligible' in verdict_value.lower() and 'not' not in verdict_value.lower():
            return "ELIGIBLE", "eligible"
        elif 'not eligible' in verdict_value.lower():
            return "NOT ELIGIBLE", "error"
        elif 'unclear' in verdict_value.lower():
            return "NEEDS REVIEW", "warning"
    
    # Fallback: Look for keywords if no explicit verdict line
    if 'eligible' in evaluation_lower:
        if 'not eligible' in evaluation_lower:
            return "NOT ELIGIBLE", "error"
        elif 'likely eligible' in evaluation_lower or 'appears eligible' in evaluation_lower:
            return "ELIGIBLE", "eligible"
        else:
            return "ELIGIBLE", "eligible"
    
    # More fallback logic
    positive_indicators = ['meets all requirements', 'satisfies all', 'qualifies for']
    negative_indicators = ['does not meet', 'insufficient', 'below the required', 'ineligible']
    
    pos_count = sum(1 for phrase in positive_indicators if phrase in evaluation_lower)
    neg_count = sum(1 for phrase in negative_indicators if phrase in evaluation_lower)
    
    if pos_count > neg_count:
        return "ELIGIBLE", "eligible"
    elif neg_count > pos_count:
        return "NOT ELIGIBLE", "error"
    else:
        return "NEEDS REVIEW", "warning"

def extract_sections(evaluation_text):
    """Extract different sections from the evaluation text"""
    sections = {
        'explanation': '',
        'missing_requirements': [],
        'additional_info': []
    }
    
    text_lower = evaluation_text.lower()
    lines = evaluation_text.split('\n')
    
    # Try to find and extract sections based on common patterns
    current_section = None
    for line in lines:
        line_lower = line.lower().strip()
        
        # Skip empty lines and verdict line
        if not line.strip() or line_lower.startswith('verdict:'):
            continue
        
        # Detect section headers
        if 'explanation:' in line_lower:
            current_section = 'explanation'
            sections['explanation'] = line.split(':', 1)[1].strip() if ':' in line else ''
            continue
        elif 'missing requirements:' in line_lower:
            current_section = 'missing_requirements'
            continue
        elif 'additional information needed:' in line_lower:
            current_section = 'additional_info'
            continue
        elif 'eligibility summary:' in line_lower:
            current_section = 'explanation'
            sections['explanation'] = line.split(':', 1)[1].strip() if ':' in line else ''
            continue
        
        # Add content to current section
        if current_section:
            line_clean = line.strip()
            if line_clean:
                if current_section == 'explanation':
                    sections['explanation'] += ' ' + line_clean
                elif current_section == 'missing_requirements':
                    # Clean bullet points
                    line_clean = line_clean.lstrip('-•* ').strip()
                    if line_clean:
                        sections['missing_requirements'].append(line_clean)
                elif current_section == 'additional_info':
                    line_clean = line_clean.lstrip('-•* ').strip()
                    if line_clean:
                        sections['additional_info'].append(line_clean)
    
    # Clean up explanation
    sections['explanation'] = sections['explanation'].strip()
    
    return sections

# --------------------------------------------------
# INIT RAG
# --------------------------------------------------
if "rag" not in st.session_state:
    try:
        st.session_state.rag = SwiftVisaRAG()
        st.session_state.ready = True
    except Exception as e:
        st.session_state.ready = False
        st.error("System initialisation failed.")
        st.code(str(e))

# --------------------------------------------------
# HEADER
# --------------------------------------------------
def render_header():
    header_img = crop_header_image("assets/header__image.png", 1600, 250)
    
    if header_img:
        st.image(header_img, use_container_width=True)
    
    st.markdown("<div class='main-title'>SwiftVisa</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>UK Visa Eligibility Screening Tool</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="notice">
    This tool provides a preliminary eligibility assessment based on publicly available UK immigration rules. 
    It does not constitute legal advice and does not guarantee visa approval. Always verify requirements 
    using official UK government sources.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

# --------------------------------------------------
# FORM FIELD RENDERER WITH DATE VALIDATION
# --------------------------------------------------
def calculate_age(birth_date):
    """Calculate age from birth date"""
    today = date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

def render_field(label, cfg, key, visa_type=None):
    kwargs = {"key": key}
    
    # Add help text if conditional field
    help_text = None
    if cfg.get("conditional_on"):
        parent = cfg.get("conditional_on")
        show_when = cfg.get("show_when")
        help_text = f"Only required if '{parent}' is '{show_when}'"
    
    if cfg["type"] == "text":
        return st.text_input(label, help=help_text, **kwargs)
    
    if cfg["type"] == "number":
        return st.number_input(label, min_value=0.0, help=help_text, **kwargs)
    
    if cfg["type"] == "date":
        # Set appropriate date ranges
        if "birth" in label.lower():
            # For date of birth: allow ages from 16 to 100 years
            min_date = date(1924, 1, 1)  # ~100 years ago
            max_date = date(2008, 12, 31)  # 16+ years old
            value = date(2000, 1, 1)  # Default to 24 years old
        else:
            # For other dates (travel, course start, etc.)
            min_date = date.today()
            max_date = date(2030, 12, 31)
            value = date.today()
        
        return st.date_input(
            label,
            value=value,
            min_value=min_date,
            max_value=max_date,
            help=help_text,
            **kwargs
        )
    
    if cfg["type"] == "select":
        return st.selectbox(label, [""] + cfg["options"], help=help_text, **kwargs)

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
render_header()

if not st.session_state.ready:
    st.stop()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("**Eligibility Assessment**")
    st.markdown("")
    
    visa_type = st.selectbox(
        "Visa type",
        [""] + list(VISA_DISPLAY_NAMES.keys()),
        format_func=lambda x: VISA_DISPLAY_NAMES.get(x, "Select a visa") if x else "Select a visa"
    )
    
    st.markdown("---")
    st.markdown("**Required Information**")
    st.markdown("""
- Passport details
- Travel dates
- Financial information
- Course/job documentation
    """)
    
    st.markdown("---")
    st.markdown("**About SwiftVisa**")
    st.markdown("""
SwiftVisa uses advanced **RAG (Retrieval-Augmented Generation)** technology powered by state-of-the-art **Large Language Models** to provide accurate visa eligibility assessments based on real-time UK immigration data.

**Key Features:**
- AI-powered intelligent screening
- Real-time policy updates
- Instant preliminary assessments
- Evidence-based recommendations

*This tool provides guidance only and is not a substitute for official advice.*
    """)

# --------------------------------------------------
# LANDING VIEW
# --------------------------------------------------
if not visa_type:
    st.markdown("### Select a visa type to begin")
    st.markdown("")
    
    col1, col2 = st.columns(2)
    
    visa_info = [
        ("Student Visa", "For applicants accepted to a UK educational institution", "assets/student.png"),
        ("Graduate Visa", "For students who completed eligible UK studies", "assets/graduate.png"),
        ("Skilled Worker Visa", "For skilled professionals with a UK job offer", "assets/skilledworker.png"),
        ("Visitor Visa", "For tourism, business or family visits", "assets/standardvisitor.png"),
    ]
    
    for idx, (visa_name, description, image_path) in enumerate(visa_info):
        col = col1 if idx % 2 == 0 else col2
        
        with col:
            card_img = crop_card_image(image_path, 400, 250)
            
            # Convert image to base64 for CSS background
            import base64
            from io import BytesIO
            
            if card_img:
                buffered = BytesIO()
                card_img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                bg_image = f"url(data:image/png;base64,{img_base64})"
            else:
                bg_image = "none"
            
            st.markdown(f"""
            <div class="visa-card" style="background-image: {bg_image};">
                <div class="visa-card-content">
                    <p class="visa-card-title">{visa_name}</p>
                    <p class="visa-card-desc">{description}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("")
    
    st.stop()

# --------------------------------------------------
# APPLICATION FORM
# --------------------------------------------------
st.markdown(f"<div class='section-title'>{VISA_DISPLAY_NAMES[visa_type]} Application</div>", unsafe_allow_html=True)
st.markdown("")

st.markdown("<div class='subsection-title'>Personal Information</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
user_data = {}

fields = list(COMMON_FIELDS.items())
mid = len(fields) // 2

with col1:
    for k, v in fields[:mid]:
        user_data[k] = render_field(k, v, f"common_{k}", visa_type)

with col2:
    for k, v in fields[mid:]:
        user_data[k] = render_field(k, v, f"common_{k}", visa_type)

st.markdown("<div class='subsection-title'>Visa-Specific Details</div>", unsafe_allow_html=True)

vfields = VISA_FIELDS_MAP[visa_type]
vlist = list(vfields.items())
mid = len(vlist) // 2

c1, c2 = st.columns(2)

# Track which column we're in
col_idx = 0

for k, v in vlist:
    # Check if this field is conditional
    is_conditional = v.get("conditional_on") is not None
    
    if is_conditional:
        # Get the parent field value
        parent_field = v.get("conditional_on")
        show_when_value = v.get("show_when")
        parent_value = user_data.get(parent_field)
        
        # Only show if parent field matches the required value
        if parent_value != show_when_value:
            continue  # Skip this field
    
    # Render in alternating columns
    current_col = c1 if col_idx < mid else c2
    
    with current_col:
        user_data[k] = render_field(k, v, f"{visa_type}_{k}", visa_type)
    
    col_idx += 1

# --------------------------------------------------
# SUBMIT
# --------------------------------------------------
st.markdown("---")
submit = st.button("Evaluate eligibility", use_container_width=True, type="primary")

if submit:
    with st.spinner("Evaluating your eligibility..."):
        try:
            # Calculate age and add to profile
            dob = user_data.get("Date of Birth")
            if dob and isinstance(dob, date):
                age = calculate_age(dob)
                user_data["Calculated Age"] = age
            
            # Format user profile
            profile = {}
            for k, v in user_data.items():
                if v:
                    # Convert dates to strings
                    if isinstance(v, date):
                        profile[k] = v.strftime("%Y-%m-%d")
                    else:
                        profile[k] = str(v)
            
            result = st.session_state.rag.evaluate_eligibility(profile, visa_type)
        except Exception as e:
            st.error(f"Error during evaluation: {e}")
            import traceback
            with st.expander("Error Details"):
                st.code(traceback.format_exc())
            st.stop()

    st.markdown(f"<div class='section-title'>Eligibility Assessment</div>", unsafe_allow_html=True)
    st.markdown("")

    # Parse eligibility from LLM response
    status, cls = parse_eligibility(result["evaluation"])
    
    # Map to your original class names
    if cls == "eligible":
        result_class = "result-eligible"
        title = "✓ Likely Eligible"
    elif cls == "error":
        result_class = "result-error"
        title = "✗ Not Eligible"
    else:
        result_class = "result-warning"
        title = "⚠ Needs Further Review"

    # Extract sections from the evaluation
    sections = extract_sections(result["evaluation"])
    
    # Display the verdict
    st.markdown(f"""
    <div class="result-box {result_class}">
        <div class="result-title">{title}</div>
        <div class="result-text">
            {sections['explanation'] if sections['explanation'] else "Based on the information provided, here is your eligibility assessment."}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display missing requirements section
    if sections['missing_requirements']:
        st.markdown("""
        <div class="requirements-box">
            <div class="requirements-title">
                <i>⚠</i> Missing or Unmet Requirements
            </div>
            <ul class="requirements-list">
        """, unsafe_allow_html=True)
        
        for req in sections['missing_requirements']:
            st.markdown(f"<li>{req}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Display additional information needed
    if sections['additional_info']:
        st.markdown("""
        <div class="requirements-box">
            <div class="requirements-title">
                <i>ℹ</i> Additional Information Required
            </div>
            <ul class="requirements-list">
        """, unsafe_allow_html=True)
        
        for info in sections['additional_info']:
            st.markdown(f"<li>{info}</li>", unsafe_allow_html=True)
        
        st.markdown("</ul></div>", unsafe_allow_html=True)
    
    # Add a note about official verification
    st.markdown("""
    <div class="info-note">
        <strong>Important:</strong> This is a preliminary assessment based on the information provided. 
        For official visa decisions, always consult the UK government's official immigration guidance 
        or seek advice from a qualified immigration advisor.
    </div>
    """, unsafe_allow_html=True)
    
    # Optional: Show raw response for debugging
    with st.expander("View detailed assessment"):
        st.markdown("**Raw LLM Response:**")
        st.code(result["evaluation"])