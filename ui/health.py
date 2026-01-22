import streamlit as st

def render():
    # Job Offer & Sponsorship Row
    col1, col2 = st.columns(2)
    with col1:
        job_offer_confirmed = st.selectbox("Job offer confirmed?", ["Yes", "No"])
    with col2:
        employer_is_licensed_healthcare_sponsor = st.selectbox(
            "Employer is licensed healthcare sponsor?", ["Yes", "No"]
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
        job_is_eligible_healthcare_role = st.selectbox(
            "Job is eligible healthcare role?", ["Yes", "No"]
        )
    with col2:
        salary_offered = st.number_input("Salary Offered (£)", min_value=0)
    
    # Requirements Row
    col1, col2, col3 = st.columns(3)
    with col1:
        meets_healthcare_salary_rules = st.selectbox(
            "Meets salary rules?", ["Yes", "No"]
        )
    with col2:
        professional_registration_required = st.selectbox(
            "Professional reg required?", ["Yes", "No"]
        )
    with col3:
        english_requirement_met = st.selectbox(
            "English requirement met?", ["Yes", "No"]
        )
    
    # Professional Registration (if required)
    if professional_registration_required == "Yes":
        professional_registration_provided = st.selectbox(
            "Professional registration provided?", ["Yes", "No"]
        )
    else:
        professional_registration_provided = "N/A"
    
    return {
        "job_offer_confirmed": job_offer_confirmed,
        "employer_is_licensed_healthcare_sponsor": employer_is_licensed_healthcare_sponsor,
        "certificate_of_sponsorship_issued": certificate_of_sponsorship_issued,
        "cos_reference_number": cos_reference_number,
        "job_title": job_title,
        "soc_code": soc_code,
        "job_is_eligible_healthcare_role": job_is_eligible_healthcare_role,
        "salary_offered": salary_offered,
        "meets_healthcare_salary_rules": meets_healthcare_salary_rules,
        "professional_registration_required": professional_registration_required,
        "professional_registration_provided": professional_registration_provided,
        "english_requirement_met": english_requirement_met,
    }