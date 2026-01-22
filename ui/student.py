import streamlit as st
import datetime

def render():
    # CAS Information Row
    col1, col2 = st.columns(2)
    with col1:
        has_cas = st.selectbox("Do you have a CAS?", ["Yes", "No"])
    with col2:
        cas_reference_number = st.text_input("CAS Reference Number")
    
    # Provider & Course Level Row
    col1, col2 = st.columns(2)
    with col1:
        education_provider_is_licensed = st.selectbox(
            "Education provider is licensed?", ["Yes", "No"]
        )
    with col2:
        # Course dropdown with durations
        course_options = {
            "Master's (MSc/MA)": 12,
            "Master's (Research)": 24,
            "MBA": 12,
            "PhD": 36,
            "Bachelor's Degree": 36,
            "Foundation Course": 9,
            "Pre-sessional English": 3,
            "Other": 12
        }
        course_level = st.selectbox("Course Level", list(course_options.keys()))
        # Get duration based on selected course
        auto_duration = course_options[course_level]
    
    # Course Details Row
    col1, col2 = st.columns(2)
    with col1:
        course_full_time = st.selectbox("Course is full-time?", ["Yes", "No"])
    with col2:
        # Duration auto-filled but editable
        course_duration_months = st.number_input(
            "Course Duration (months)", 
            min_value=1, 
            value=auto_duration,
            help="Duration is auto-filled based on course type but can be changed"
        )
    
    # Course Dates Row
    col1, col2 = st.columns(2)
    with col1:
        # Default: course starts in 3 months (typical application timeline)
        default_start = datetime.date.today() + datetime.timedelta(days=90)
        course_start_date = st.date_input(
            "Course Start Date",
            value=default_start,
            min_value=datetime.date.today()
        )
    with col2:
        # Calculate end date based on duration
        default_end = course_start_date + datetime.timedelta(days=int(course_duration_months * 30))
        course_end_date = st.date_input(
            "Course End Date",
            value=default_end,
            min_value=course_start_date
        )
    
    # Financial Requirements Row
    col1, col2, col3 = st.columns(3)
    with col1:
        meets_financial_requirement = st.selectbox(
            "Meets financial requirement?", ["Yes", "No"]
        )
    with col2:
        funds_held_for_28_days = st.selectbox(
            "Funds held for 28 days?", ["Yes", "No"]
        )
    with col3:
        english_requirement_met = st.selectbox(
            "English requirement met?", ["Yes", "No"]
        )
    
    return {
        "has_cas": has_cas,
        "cas_reference_number": cas_reference_number,
        "education_provider_is_licensed": education_provider_is_licensed,
        "course_level": course_level,
        "course_full_time": course_full_time,
        "course_start_date": course_start_date,
        "course_end_date": course_end_date,
        "course_duration_months": course_duration_months,
        "meets_financial_requirement": meets_financial_requirement,
        "funds_held_for_28_days": funds_held_for_28_days,
        "english_requirement_met": english_requirement_met,
    }