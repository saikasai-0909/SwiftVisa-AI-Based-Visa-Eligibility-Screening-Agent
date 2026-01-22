def parse_eligibility_result(llm_output: str, applicant_data: dict = None):
    result = {
        "status": "NOT ELIGIBLE",
        "reason": ""
    }

    if not llm_output or not isinstance(llm_output, str):
        result["reason"] = "- No explanation returned by the system."
        return result

    text = llm_output.strip()
    lower = text.lower()

    # STATUS - prefer Final Assessment, then Status line, then keywords
    if "final assessment" in lower:
        for line in text.splitlines():
            if "final assessment" in line.lower():
                if "not eligible" in line.lower():
                    result["status"] = "NOT ELIGIBLE"
                elif "eligible" in line.lower():
                    result["status"] = "ELIGIBLE"
                break
    elif "status:" in lower:
        for line in text.splitlines():
            if "status:" in line.lower():
                if "not eligible" in line.lower():
                    result["status"] = "NOT ELIGIBLE"
                elif "eligible" in line.lower():
                    result["status"] = "ELIGIBLE"
                break
    elif "not eligible" in lower:
        result["status"] = "NOT ELIGIBLE"
    elif "eligible" in lower:
        result["status"] = "ELIGIBLE"

    # REASON - try multiple patterns (case-insensitive)
    reason_found = False
    
    # Pattern 1: Look for "Reason:" (case-insensitive)
    if "reason:" in lower:
        idx = lower.index("reason:")
        result["reason"] = text[idx + len("reason:"):].strip()
        reason_found = True
    
    # Pattern 2: Look for "Final Assessment:" if no reason found
    if not reason_found and "final assessment:" in lower:
        idx = lower.index("final assessment:")
        result["reason"] = text[idx:].strip()
        reason_found = True
    
    # Pattern 3: If status line exists, take everything after it
    if not reason_found and "status:" in lower:
        for i, line in enumerate(text.splitlines()):
            if "status:" in line.lower():
                # Get all lines after status
                remaining_lines = text.splitlines()[i+1:]
                result["reason"] = "\n".join(remaining_lines).strip()
                reason_found = True
                break
    
    # Fallback: if still no reason, use the full output
    if not reason_found or not result["reason"]:
        # Remove the status line and use the rest
        lines = [line for line in text.splitlines() if "status:" not in line.lower()]
        result["reason"] = "\n".join(lines).strip()
    
    # If still empty, show a message with the raw output for debugging
    if not result["reason"]:
        result["reason"] = f"No detailed explanation found.\n\nRaw LLM Output:\n{text}"

    # ============ HARD OVERRIDE: Check applicant_data for critical disqualifiers ============
    if applicant_data:
        # If Criminal History is "Yes", ALWAYS NOT ELIGIBLE
        if applicant_data.get("criminal_history") == "Yes":
            result["status"] = "NOT ELIGIBLE"
            result["reason"] = "❌ DISQUALIFIED: Applicant has declared criminal history. UK visa policy states that applicants with criminal convictions are typically not eligible for a UK visa.\n\n" + result["reason"]
        
        # If Previous UK Visa Refusal is "Yes", ALWAYS NOT ELIGIBLE
        elif applicant_data.get("previous_uk_refusal") == "Yes":
            result["status"] = "NOT ELIGIBLE"
            result["reason"] = "❌ DISQUALIFIED: Applicant has a previous UK visa refusal on record. This is a critical barrier to eligibility.\n\n" + result["reason"]
        
        # If English Requirement is "No", ALWAYS NOT ELIGIBLE
        elif applicant_data.get("english_requirement_met") == "No":
            result["status"] = "NOT ELIGIBLE"
            result["reason"] = "❌ DISQUALIFIED: Applicant does not meet the English language requirement. This is mandatory for UK visa eligibility.\n\n" + result["reason"]
    
    # Safety: Check for red flags in reason text
    reason_lower = result["reason"].lower()
    
    # If criminal history is mentioned as a concern/issue, mark as NOT ELIGIBLE
    if "criminal history" in reason_lower:
        if any(word in reason_lower for word in ["is a concern", "critical alert", "must investigate", "further investigation may be necessary", "requires further"]):
            result["status"] = "NOT ELIGIBLE"
    
    # If any requirement is explicitly marked "Not Met", force NOT ELIGIBLE
    if ": not met" in reason_lower:
        result["status"] = "NOT ELIGIBLE"
    
    # Check if the final assessment says "not met" or "does not meet" in key sections
    if "final" in reason_lower and ("not eligible" in reason_lower or "does not meet" in reason_lower):
        # Only mark as NOT ELIGIBLE if it's in a final verdict context
        lines = reason_lower.split('\n')
        for i, line in enumerate(lines):
            if 'final' in line and ('not eligible' in line or 'does not meet' in line):
                result["status"] = "NOT ELIGIBLE"
                break
    
    # If there's a recommendation section suggesting further checks AND it's a serious concern
    if "recommendation:" in reason_lower and ("must" in reason_lower or "should not" in reason_lower):
        result["status"] = "NOT ELIGIBLE"

    return result