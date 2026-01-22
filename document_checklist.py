DOCUMENT_REQUIREMENTS = {
    "student": {
        "Essential Documents": [
            "Current passport or other valid travel documentation",
            "Confirmation of Acceptance for Studies (CAS) reference number",
            "Proof of financial support (if required)",
            "English language qualification (if required)",
        ],
        "Conditional Documents": [
            "Academic Technology Approval Scheme (ATAS) certificate (if studying sensitive topics at RQF level 7+)",
            "TB test certificate (if required based on your country)",
            "Financial evidence (may be requested)",
        ],
        "Supporting Documents": [
            "Previous passport (if applicable)",
            "Birth certificate or travel document for dependants",
            "Marriage certificate (if applicable)",
            "Divorce documents (if applicable)",
            "Sponsor letter (if someone is sponsoring you)",
        ]
    },
    "graduate": {
        "Essential Documents": [
            "Current passport or other valid travel documentation",
            "Degree certificate or evidence of qualification",
            "Proof of funds (if required)",
        ],
        "Conditional Documents": [
            "English language qualification (if required)",
            "TB test certificate (if required)",
            "ATAS certificate (if applicable)",
        ],
        "Supporting Documents": [
            "Previous passports",
            "Birth certificate",
            "Marriage or divorce documents",
            "Job offer letter (if applicable)",
        ]
    },
    "work": {
        "Essential Documents": [
            "Current valid passport",
            "Certificate of sponsorship (CoS) reference number",
            "Proof of qualifications",
            "English language evidence",
            "Proof of funds",
        ],
        "Conditional Documents": [
            "TB test certificate",
            "ATAS certificate (if required)",
            "Documents related to any criminal convictions",
        ],
        "Supporting Documents": [
            "Previous passport",
            "Birth certificate",
            "Marriage or divorce documents",
            "Additional qualifications",
            "Professional licenses",
        ]
    },
    "visitor": {
        "Essential Documents": [
            "Current valid passport",
            "Return travel tickets or itinerary",
            "Proof of funds to cover stay",
        ],
        "Conditional Documents": [
            "Accommodation details (booking confirmation or invitation letter)",
            "TB test certificate (if required)",
        ],
        "Supporting Documents": [
            "Employment letter",
            "Bank statements",
            "Previous travel history",
            "Invitation letter from UK contact",
        ]
    },
    "healthcare_worker": {
        "Essential Documents": [
            "Current valid passport",
            "Job offer or employment contract",
            "Professional qualifications",
            "English language evidence",
            "Proof of funds",
        ],
        "Conditional Documents": [
            "TB test certificate",
            "Health clearance",
            "ATAS certificate (if required)",
        ],
        "Supporting Documents": [
            "Previous employment letters",
            "Professional registrations",
            "Training certificates",
        ]
    }
}

def get_checklist_for_visa(visa_type):
    """Get document checklist for a specific visa type"""
    return DOCUMENT_REQUIREMENTS.get(visa_type, {})

def create_checklist_data(visa_type):
    """Create structured checklist data with status tracking"""
    checklist = get_checklist_for_visa(visa_type)
    checklist_data = []
    
    for category, documents in checklist.items():
        for doc in documents:
            checklist_data.append({
                "Category": category,
                "Document": doc,
                "Status": "☐ Not Submitted",
                "Notes": ""
            })
    
    return checklist_data
