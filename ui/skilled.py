import streamlit as st

def render():
    # Job Offer & Sponsorship Row
    col1, col2 = st.columns(2)
    with col1:
        job_offer_confirmed = st.selectbox("Job offer confirmed?", ["Yes", "No"])
    with col2:
        employer_is_licensed_sponsor = st.selectbox(
            "Employer is licensed sponsor?", ["Yes", "No"]
        )
    
    # COS Row
    col1, col2 = st.columns(2)
    with col1:
        certificate_of_sponsorship_issued = st.selectbox(
            "Certificate of Sponsorship issued?", ["Yes", "No"]
        )
    with col2:
        cos_reference_number = st.text_input("COS Reference Number")
    
    # Job Details Row
    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Job Title")
    with col2:
        soc_code = st.text_input("SOC Code")
    
    # Eligibility & Salary Row
    col1, col2 = st.columns(2)
    with col1:
        job_is_eligible_occupation = st.selectbox(
            "Job is eligible occupation?", ["Yes", "No"]
        )
    with col2:
        salary_offered = st.number_input("Salary Offered (£)", min_value=0)
    
    # Requirements Row
    col1, col2, col3 = st.columns(3)
    with col1:
        meets_minimum_salary_threshold = st.selectbox(
            "Meets salary threshold?", ["Yes", "No"]
        )
    with col2:
        english_requirement_met = st.selectbox(
            "English requirement met?", ["Yes", "No"]
        )
    with col3:
        criminal_record_certificate_required = st.selectbox(
            "Criminal cert required?", ["Yes", "No"]
        )
    
    # Certificate Provided (if required)
    if criminal_record_certificate_required == "Yes":
        criminal_record_certificate_provided = st.selectbox(
            "Criminal record certificate provided?", ["Yes", "No"]
        )
    else:
        criminal_record_certificate_provided = "N/A"
    
    return {
        "job_offer_confirmed": job_offer_confirmed,
        "employer_is_licensed_sponsor": employer_is_licensed_sponsor,
        "certificate_of_sponsorship_issued": certificate_of_sponsorship_issued,
        "cos_reference_number": cos_reference_number,
        "job_title": job_title,
        "soc_code": soc_code,
        "job_is_eligible_occupation": job_is_eligible_occupation,
        "salary_offered": salary_offered,
        "meets_minimum_salary_threshold": meets_minimum_salary_threshold,
        "english_requirement_met": english_requirement_met,
        "criminal_record_certificate_required": criminal_record_certificate_required,
        "criminal_record_certificate_provided": criminal_record_certificate_provided,
    }