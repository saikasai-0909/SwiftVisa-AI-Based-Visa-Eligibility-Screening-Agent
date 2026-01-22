from fpdf import FPDF
from datetime import datetime
import io

def clean_text(text):
    """Remove problematic characters for PDF encoding"""
    if not text:
        return ""
    # Replace special characters
    text = str(text)
    text = text.replace("✓", "✓").replace("✗", "X")
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")
    # Remove any remaining problematic unicode
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

def generate_eligibility_pdf(applicant_data, eligibility_result, visa_type):
    """Generate PDF report of visa eligibility assessment"""
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Set colors and fonts
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(0, 102, 153)  # Teal color
    pdf.cell(0, 10, "UK Visa Eligibility Assessment", ln=True, align="C")
    
    # Add timestamp
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    timestamp = datetime.now().strftime('%d %B %Y at %H:%M')
    pdf.cell(0, 5, f"Generated on: {timestamp}", ln=True, align="C")
    
    # Add separator
    pdf.set_draw_color(0, 102, 153)
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(8)
    
    # Visa Type and Status
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 0, 0)
    visa_clean = clean_text(visa_type.upper())
    pdf.cell(0, 8, f"Visa Type: {visa_clean}", ln=True)
    
    # Status Badge
    status = eligibility_result.get("status", "UNDETERMINED")
    if status == "ELIGIBLE":
        pdf.set_text_color(34, 139, 34)  # Green
        status_text = "ELIGIBLE"
    else:
        pdf.set_text_color(178, 34, 34)  # Red
        status_text = "NOT ELIGIBLE"
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 8, f"Status: {status_text}", ln=True)
    pdf.ln(5)
    
    # Applicant Information Section
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 102, 153)
    pdf.cell(0, 8, "Applicant Information", ln=True)
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    
    applicant_fields = [
        ("Full Name", applicant_data.get("full_name")),
        ("Nationality", applicant_data.get("nationality")),
        ("Date of Birth", applicant_data.get("dob")),
        ("Current Location", applicant_data.get("current_location")),
    ]
    
    for label, value in applicant_fields:
        value_clean = clean_text(str(value))
        pdf.cell(50, 6, f"{label}:", border=0)
        pdf.cell(0, 6, value_clean, ln=True)
    
    pdf.ln(5)
    
    # Compliance Information
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 102, 153)
    pdf.cell(0, 8, "Compliance & Background", ln=True)
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    
    compliance_fields = [
        ("Criminal History", applicant_data.get("criminal_history")),
        ("Previous UK Visa Refusal", applicant_data.get("previous_uk_refusal")),
        ("English Requirement Met", applicant_data.get("english_requirement_met")),
    ]
    
    for label, value in compliance_fields:
        value_clean = clean_text(str(value))
        pdf.cell(80, 6, f"{label}:", border=0)
        pdf.cell(0, 6, value_clean, ln=True)
    
    pdf.ln(5)
    
    # Financial Information
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 102, 153)
    pdf.cell(0, 8, "Financial Details", ln=True)
    
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    funds = applicant_data.get('funds_available', 0)
    pdf.cell(50, 6, "Funds Available:", border=0)
    pdf.cell(0, 6, f"GBP {funds}", ln=True)
    
    pdf.ln(8)
    
    # Assessment Reason
    pdf.set_font("Arial", "B", 12)
    pdf.set_text_color(0, 102, 153)
    pdf.cell(0, 8, "Assessment Details", ln=True)
    
    pdf.set_font("Arial", "", 9)
    pdf.set_text_color(0, 0, 0)
    
    reason = eligibility_result.get("reason", "No detailed explanation available.")
    reason_clean = clean_text(reason)
    pdf.multi_cell(0, 5, reason_clean)
    
    pdf.ln(8)
    
    # Footer
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(150, 150, 150)
    pdf.cell(0, 5, "This assessment is based on the latest UK visa policy documents.", ln=True, align="C")
    pdf.cell(0, 5, "For official guidance, please visit www.gov.uk/student-visa", ln=True, align="C")
    
    return bytes(pdf.output())
