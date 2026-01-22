import pandas as pd
from typing import Dict, List

VISA_COMPARISON_DATA = {
    "student": {
        "description": "For studying at a higher education provider",
        "duration": "Length of course + 4 months",
        "estimated_cost": "£5,000 - £50,000+",
        "key_requirements": [
            "Confirmation of Acceptance (CAS)",
            "CEFR level B2 English (or equivalent)",
            "Proof of financial support",
            "No criminal history issues"
        ],
        "processing_time": "3 weeks",
        "main_benefit": "Study and gain UK qualification"
    },
    "graduate": {
        "description": "For recent graduates to work and gain experience",
        "duration": "2-3 years",
        "estimated_cost": "£0 - £5,000",
        "key_requirements": [
            "Bachelor degree or higher",
            "English proficiency",
            "No criminal history issues"
        ],
        "processing_time": "3 weeks",
        "main_benefit": "Work experience post-graduation"
    },
    "work": {
        "description": "For skilled workers with job offer",
        "duration": "Up to 5 years",
        "estimated_cost": "£719 - £1,035",
        "key_requirements": [
            "Job offer from UK employer",
            "Certificate of Sponsorship (CoS)",
            "CEFR level B2 English",
            "Proof of funds",
            "Point-based assessment"
        ],
        "processing_time": "4-8 weeks",
        "main_benefit": "Employment in UK"
    },
    "visitor": {
        "description": "For tourism, visiting family, or short business trips",
        "duration": "Up to 6 months",
        "estimated_cost": "£0 - £100",
        "key_requirements": [
            "Valid passport",
            "Proof of funds",
            "Return flight ticket",
            "Accommodation details"
        ],
        "processing_time": "1-3 weeks",
        "main_benefit": "Temporary stay in UK"
    },
    "healthcare_worker": {
        "description": "Fast-track visa for health and care workers",
        "duration": "Up to 5 years",
        "estimated_cost": "Free (or heavily subsidized)",
        "key_requirements": [
            "Job offer from UK health/care provider",
            "Health and care professional qualification",
            "English proficiency",
            "Health clearance"
        ],
        "processing_time": "3-4 weeks",
        "main_benefit": "Work in NHS or care sector with benefits"
    }
}

def compare_visa_types(selected_visas: List[str]) -> pd.DataFrame:
    """Create comparison DataFrame for selected visa types"""
    
    data = {}
    
    for visa in selected_visas:
        if visa in VISA_COMPARISON_DATA:
            visa_info = VISA_COMPARISON_DATA[visa]
            data[visa.replace("_", " ").upper()] = {
                "Description": visa_info["description"],
                "Duration": visa_info["duration"],
                "Estimated Cost": visa_info["estimated_cost"],
                "Processing Time": visa_info["processing_time"],
                "Main Benefit": visa_info["main_benefit"]
            }
    
    df = pd.DataFrame(data).T
    return df

def get_visa_details(visa_type: str) -> Dict:
    """Get detailed information about a specific visa type"""
    return VISA_COMPARISON_DATA.get(visa_type, {})

def get_best_match(applicant_profile: Dict) -> List[str]:
    """
    Recommend visa types based on applicant profile
    Returns list of suitable visa types in order of suitability
    """
    
    recommendations = []
    
    # Student visa
    if applicant_profile.get("purpose_of_visit") == "Studies":
        recommendations.append("student")
    
    # Graduate visa
    if applicant_profile.get("qualification_level") in ["Bachelor's", "Master's", "PhD"]:
        recommendations.append("graduate")
    
    # Work visa
    if applicant_profile.get("purpose_of_visit") == "Work":
        recommendations.append("work")
    
    # Visitor visa
    if applicant_profile.get("purpose_of_visit") == "Tourism":
        recommendations.append("visitor")
    
    # Healthcare worker
    if applicant_profile.get("purpose_of_visit") == "Work" and \
       applicant_profile.get("sector") == "Healthcare":
        recommendations.insert(0, "healthcare_worker")
    
    # Default to visitor if no match
    if not recommendations:
        recommendations.append("visitor")
    
    return recommendations

def create_comparison_summary(applicant_data: Dict) -> str:
    """Create a text summary of visa comparison based on applicant data"""
    
    purpose = applicant_data.get("purpose_of_visit", "Unknown")
    recommendations = get_best_match(applicant_data)
    
    summary = f"""
VISA TYPE RECOMMENDATIONS FOR YOUR PROFILE
==========================================

Based on your profile:
- Purpose of Visit: {purpose}
- Nationality: {applicant_data.get('nationality')}

RECOMMENDED VISA TYPES (in order of suitability):

"""
    
    for i, visa in enumerate(recommendations, 1):
        visa_info = VISA_COMPARISON_DATA.get(visa, {})
        summary += f"\n{i}. {visa.replace('_', ' ').upper()}\n"
        summary += f"   Description: {visa_info.get('description', 'N/A')}\n"
        summary += f"   Duration: {visa_info.get('duration', 'N/A')}\n"
        summary += f"   Processing Time: {visa_info.get('processing_time', 'N/A')}\n"
        summary += f"   Main Benefit: {visa_info.get('main_benefit', 'N/A')}\n"
    
    return summary
