import streamlit as st

def render():
    # Purpose Row
    col1, col2 = st.columns(2)
    with col1:
        purpose_of_visit = st.text_input(
            "Purpose of Visit",
            key="visitor_purpose_of_visit"
        )
    with col2:
        purpose_is_permitted_under_visitor_rules = st.selectbox(
            "Is the purpose permitted under Visitor rules?",
            ["Yes", "No"],
            key="visitor_purpose_permitted"
        )
    
    # Stay Duration Row
    col1, col2 = st.columns(2)
    with col1:
        intended_length_of_stay_months = st.number_input(
            "Intended Length of Stay (months)",
            min_value=0,
            max_value=12,
            step=1,
            key="visitor_length_of_stay"
        )
    with col2:
        stay_within_6_months_limit = st.selectbox(
            "Will the stay be within 6 months?",
            ["Yes", "No"],
            key="visitor_within_6_months"
        )
    
    # Arrangements Row
    col1, col2 = st.columns(2)
    with col1:
        accommodation_arranged = st.selectbox(
            "Is accommodation arranged?",
            ["Yes", "No"],
            key="visitor_accommodation"
        )
    with col2:
        return_or_onward_travel_planned = st.selectbox(
            "Is return or onward travel planned?",
            ["Yes", "No"],
            key="visitor_return_travel"
        )
    
    # Intent & Funds Row
    col1, col2 = st.columns(2)
    with col1:
        intends_to_leave_uk_after_visit = st.selectbox(
            "Do you intend to leave the UK after the visit?",
            ["Yes", "No"],
            key="visitor_intent_leave"
        )
    with col2:
        sufficient_funds_for_stay = st.selectbox(
            "Do you have sufficient funds for the stay?",
            ["Yes", "No"],
            key="visitor_funds"
        )
    
    return {
        "purpose_of_visit": purpose_of_visit,
        "purpose_is_permitted_under_visitor_rules": purpose_is_permitted_under_visitor_rules,
        "intended_length_of_stay_months": intended_length_of_stay_months,
        "stay_within_6_months_limit": stay_within_6_months_limit,
        "accommodation_arranged": accommodation_arranged,
        "return_or_onward_travel_planned": return_or_onward_travel_planned,
        "intends_to_leave_uk_after_visit": intends_to_leave_uk_after_visit,
        "sufficient_funds_for_stay": sufficient_funds_for_stay,
    }