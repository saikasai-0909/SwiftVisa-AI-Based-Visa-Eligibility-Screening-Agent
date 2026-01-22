import streamlit as st
import datetime

def render_common():
    # Personal Information Row
    col1, col2 = st.columns(2)
    with col1:
        full_name = st.text_input("Full Name (as per passport)")
    with col2:
        nationality = st.text_input("Nationality")
    
    # Date of Birth Row
    col1, col2 = st.columns(2)
    with col1:
        dob = st.date_input(
            "Date of Birth",
            value=datetime.date(2000, 1, 1),
            min_value=datetime.date(1950, 1, 1),
            max_value=datetime.date.today()
        )
    with col2:
        passport_number = st.text_input("Passport Number")
    
    # Passport Dates Row
    col1, col2 = st.columns(2)
    with col1:
        # Default: issued 2 years ago
        default_issue = datetime.date.today() - datetime.timedelta(days=730)
        passport_issue_date = st.date_input(
            "Passport Issue Date",
            value=default_issue,
            max_value=datetime.date.today()
        )
    with col2:
        # Default: expires 8 years from now (total 10 year passport)
        default_expiry = datetime.date.today() + datetime.timedelta(days=2920)
        passport_expiry_date = st.date_input(
            "Passport Expiry Date",
            value=default_expiry,
            min_value=datetime.date.today()
        )
    
    # Location (Full Width)
    current_location = st.text_input("Country of Application / Current Location")
    
    # Purpose & Travel Date Row
    col1, col2 = st.columns(2)
    with col1:
        purpose_of_visit = st.text_input("Purpose of Visit")
    with col2:
        # Default: travel starts in 2 months
        default_travel = datetime.date.today() + datetime.timedelta(days=60)
        intended_start_date = st.date_input(
            "Intended Travel / Start Date",
            value=default_travel,
            min_value=datetime.date.today()
        )
    
    # Length of Stay & Funds Row
    col1, col2 = st.columns(2)
    with col1:
        length_of_stay = st.text_input("Intended Length of Stay", value="12 months")
    with col2:
        funds_available = st.number_input("Funds Available (£)", min_value=0, value=15000)
    
    # Compliance Questions Row
    col1, col2, col3 = st.columns(3)
    with col1:
        english_requirement_met = st.selectbox(
            "English Requirement Met", ["Yes", "No"]
        )
    with col2:
        criminal_history = st.selectbox(
            "Criminal History", ["No", "Yes"]
        )
    with col3:
        previous_uk_refusal = st.selectbox(
            "Previous UK Refusal", ["No", "Yes"]
        )
    
    # Contact Information Row
    col1, col2 = st.columns(2)
    with col1:
        email = st.text_input("Email Address")
    with col2:
        phone = st.text_input("Phone Number")
    
    # Address (Full Width)
    address = st.text_area("Current Address")
    
    return {
        "full_name": full_name,
        "dob": dob,
        "nationality": nationality,
        "passport_number": passport_number,
        "passport_issue_date": passport_issue_date,
        "passport_expiry_date": passport_expiry_date,
        "current_location": current_location,
        "purpose_of_visit": purpose_of_visit,
        "intended_start_date": intended_start_date,
        "length_of_stay": length_of_stay,
        "funds_available": funds_available,
        "english_requirement_met": english_requirement_met,
        "criminal_history": criminal_history,
        "previous_uk_refusal": previous_uk_refusal,
        "email": email,
        "phone": phone,
        "address": address,
    }