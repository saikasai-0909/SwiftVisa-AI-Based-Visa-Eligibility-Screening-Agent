"""
Small helpers for PDF generation/reading.
Original, MIT-compliant code.
"""
from fpdf import FPDF
import os

def text_to_pdf(text: str, out_path: str, title: str = "Report"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    # Optional header
    pdf.cell(0, 10, title, ln=True)
    pdf.ln(4)
    pdf.multi_cell(0, 6, text)
    pdf.output(out_path)

