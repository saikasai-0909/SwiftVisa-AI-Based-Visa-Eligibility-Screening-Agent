import streamlit as st

def render():
    # Current Status Row
    col1, col2 = st.columns(2)
    with col1:
        currently_in_uk = st.selectbox("Currently in the UK?", ["Yes", "No"])
    with col2:
        current_uk_visa_type = st.selectbox(
            "Current UK Visa Type", ["Student", "Tier 4"]
        )
    
    # Course Completion Row
    col1, col2 = st.columns(2)
    with col1:
        course_completed = st.selectbox("Course completed?", ["Yes", "No"])
    with col2:
        course_level_completed = st.text_input("Course Level Completed")
    
    # Provider Information Row
    col1, col2 = st.columns(2)
    with col1:
        education_provider_is_licensed = st.selectbox(
            "Education provider is licensed?", ["Yes", "No"]
        )
    with col2:
        provider_reported_completion = st.selectbox(
            "Provider reported completion to Home Office?", ["Yes", "No"]
        )
    
    # CAS & Visa Status Row
    col1, col2 = st.columns(2)
    with col1:
        original_cas_reference = st.text_input("Original CAS Reference")
    with col2:
        student_visa_valid = st.selectbox(
            "Student visa valid on application date?", ["Yes", "No"]
        )
    
    return {
        "currently_in_uk": currently_in_uk,
        "current_uk_visa_type": current_uk_visa_type,
        "course_completed": course_completed,
        "course_level_completed": course_level_completed,
        "education_provider_is_licensed": education_provider_is_licensed,
        "provider_reported_completion": provider_reported_completion,
        "original_cas_reference": original_cas_reference,
        "student_visa_valid": student_visa_valid,
    }