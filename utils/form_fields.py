"""
SwiftVisa - Form Field Definitions
Defines all form fields for each visa type based on eligibility criteria
"""

# Common fields for all visa types
COMMON_FIELDS = {
    "Full Name": {"type": "text", "required": True},
    "Date of Birth": {"type": "date", "required": True},
    "Nationality": {"type": "text", "required": True},
    "Passport Number": {"type": "text", "required": True},
    "Email Address": {"type": "text", "required": True},
    "Phone Number": {"type": "text", "required": True},
    "Current Location": {"type": "text", "required": True},
    "Intended Travel Date": {"type": "date", "required": True},
    "Intended Length of Stay (months)": {"type": "number", "required": True},
    "Funds Available (£)": {"type": "number", "required": True},
    "English Language Requirement Met": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Criminal History": {"type": "select", "options": ["Yes", "No"], "required": True},
    "tuberculosis Test results available": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Previous UK Visa Refusal": {"type": "select", "options": ["Yes", "No"], "required": True},
}

# Student Visa specific fields
STUDENT_VISA_FIELDS = {
    "Has CAS (Confirmation of Acceptance)": {"type": "select", "options": ["Yes", "No"], "required": True},
    "CAS Reference Number": {"type": "text", "required": False},
    "Education Provider is Licensed": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Course Level": {"type": "select", "options": ["Undergraduate", "Postgraduate", "PhD", "Foundation"], "required": True},
    "Course is Full-Time": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Course Start Date": {"type": "date", "required": True},
    "Course Duration (months)": {"type": "number", "required": True},
    "Meets Financial Requirement": {"type": "select", "options": ["Yes", "No", "Not Sure"], "required": True},
    "Funds Held for 28 Days": {"type": "select", "options": ["Yes", "No"], "required": True},
    "ATAS Certificate Required": {"type": "select", "options": ["Yes", "No", "Not Sure"], "required": True, "conditional_trigger": True},
    "ATAS Certificate Provided": {"type": "select", "options": ["Yes", "No"], "required": False, "conditional_on": "ATAS Certificate Required", "show_when": "Yes"},
}

# Graduate Visa specific fields
GRADUATE_VISA_FIELDS = {
    "Currently in UK": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Current UK Visa Type": {"type": "select", "options": ["Student Visa", "Tier 4", "Other", "N/A"], "required": True},
    "Course Completed": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Course Level Completed": {"type": "select", "options": ["Bachelor's", "Master's", "PhD"], "required": True},
    "Education Provider is Licensed": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Provider Reported Completion to Home Office": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Original CAS Reference": {"type": "text", "required": False},
    "Student Visa Valid on Application Date": {"type": "select", "options": ["Yes", "No"], "required": True},
}

# Skilled Worker Visa specific fields
SKILLED_WORKER_VISA_FIELDS = {
    "Job Offer Confirmed": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Employer is Licensed Sponsor": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Certificate of Sponsorship Issued": {"type": "select", "options": ["Yes", "No"], "required": True},
    "CoS Reference Number": {"type": "text", "required": False, "conditional_on": "Certificate of Sponsorship Issued", "show_when": "Yes"},
    "Job Title": {"type": "text", "required": True},
    "SOC Code": {"type": "text", "required": False},
    "Job is Eligible Occupation": {"type": "select", "options": ["Yes", "No", "Not Sure"], "required": True},
    "Salary Offered (£)": {"type": "number", "required": True},
    "Meets Minimum Salary Threshold": {"type": "select", "options": ["Yes", "No", "Not Sure"], "required": True},
    "Criminal Record Certificate Required": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Criminal Record Certificate Provided": {"type": "select", "options": ["Yes", "No", "N/A"], "required": False, "conditional_on": "Criminal Record Certificate Required", "show_when": "Yes"},
}

# Health & Care Worker Visa specific fields
HEALTH_CARE_WORKER_VISA_FIELDS = {
    "Job Offer Confirmed": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Employer is Licensed Healthcare Sponsor": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Certificate of Sponsorship Issued": {"type": "select", "options": ["Yes", "No"], "required": True},
    "CoS Reference Number": {"type": "text", "required": False, "conditional_on": "Certificate of Sponsorship Issued", "show_when": "Yes"},
    "Job Title": {"type": "text", "required": True},
    "SOC Code": {"type": "text", "required": False},
    "Job is Eligible Healthcare Role": {"type": "select", "options": ["Yes", "No", "Not Sure"], "required": True},
    "Salary Offered (£)": {"type": "number", "required": True},
    "Meets Healthcare Salary Rules": {"type": "select", "options": ["Yes", "No", "Not Sure"], "required": True},
    "Professional Registration Required": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Professional Registration Provided": {"type": "select", "options": ["Yes", "No", "N/A"], "required": False, "conditional_on": "Professional Registration Required", "show_when": "Yes"},
}

# Standard Visitor Visa specific fields
VISITOR_VISA_FIELDS = {
    "Purpose of Visit": {"type": "select", "options": ["Tourism", "Business", "Family Visit", "Medical", "Other"], "required": True},
    "Purpose is Permitted": {"type": "select", "options": ["Yes", "No", "Not Sure"], "required": True},
    "Stay Within 6 Months Limit": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Accommodation Arranged": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Return/Onward Travel Planned": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Intends to Leave UK After Visit": {"type": "select", "options": ["Yes", "No"], "required": True},
    "Sufficient Funds for Stay": {"type": "select", "options": ["Yes", "No"], "required": True},
}

# Visa type mapping
VISA_FIELDS_MAP = {
    "student": STUDENT_VISA_FIELDS,
    "graduate": GRADUATE_VISA_FIELDS,
    "skilled_worker": SKILLED_WORKER_VISA_FIELDS,
    "health_care_worker": HEALTH_CARE_WORKER_VISA_FIELDS,
    "visitor": VISITOR_VISA_FIELDS,
}

VISA_DISPLAY_NAMES = {
    "student": "Student Visa",
    "graduate": "Graduate Visa",
    "skilled_worker": "Skilled Worker Visa",
    "health_care_worker": "Health & Care Worker Visa",
    "visitor": "Standard Visitor Visa",
}