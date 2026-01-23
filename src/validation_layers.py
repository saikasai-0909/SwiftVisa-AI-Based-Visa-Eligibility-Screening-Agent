"""
SwiftVisa - Two-Layer Validation System
Layer 1: Hard Validation (Deterministic, Fast)
Layer 2: RAG + LLM (Soft Reasoning, Only if Layer 1 passes)
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ValidationStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


@dataclass
class ValidationResult:
    status: ValidationStatus
    message: str
    field: str
    is_critical: bool = False


@dataclass
class HardValidationReport:
    passed: bool
    critical_failures: List[ValidationResult]
    warnings: List[ValidationResult]
    validation_time_ms: float


class HardValidator:
    """Layer 1: Fast deterministic validation"""
    
    # Pre-computed constants for speed
    MIN_PASSPORT_VALIDITY_MONTHS = 6
    VISITOR_MAX_STAY_MONTHS = 6
    GRADUATE_VISA_MIN_COURSE_MONTHS = 12
    STUDENT_VISA_MIN_FUNDS_LONDON = 1529
    STUDENT_VISA_MIN_FUNDS_OUTSIDE_LONDON = 1171
    SKILLED_WORKER_MIN_SALARY = 25000
    HEALTH_CARE_MIN_SALARY = 25000
    
    def __init__(self):
        self.validation_start = None
    
    def validate(self, user_data: dict, visa_type: str) -> HardValidationReport:
        """Run all hard validations"""
        self.validation_start = datetime.now()
        
        critical_failures = []
        warnings = []
        
        # Common validations (parallel execution possible)
        validations = [
            self._validate_passport_validity,
            self._validate_dates,
            self._validate_funds,
            self._validate_english_requirement,
        ]
        
        # Add visa-specific validations
        visa_validators = {
            "Graduate Visa": self._validate_graduate_visa,
            "Student Visa": self._validate_student_visa,
            "Skilled Worker Visa": self._validate_skilled_worker,
            "Health & Care Visa": self._validate_health_care,
            "Visitor Visa": self._validate_visitor_visa,
        }
        
        if visa_type in visa_validators:
            validations.append(visa_validators[visa_type])
        
        # Execute all validations
        for validator in validations:
            result = validator(user_data, visa_type)
            if result:
                if result.is_critical and result.status == ValidationStatus.FAIL:
                    critical_failures.append(result)
                elif result.status == ValidationStatus.WARNING:
                    warnings.append(result)
        
        validation_time = (datetime.now() - self.validation_start).total_seconds() * 1000
        
        return HardValidationReport(
            passed=len(critical_failures) == 0,
            critical_failures=critical_failures,
            warnings=warnings,
            validation_time_ms=validation_time
        )
    
    def _validate_passport_validity(self, user_data: dict, visa_type: str) -> Optional[ValidationResult]:
        """Check passport validity - CRITICAL"""
        personal = user_data.get("personal_info", {})
        
        try:
            expiry_str = personal.get("passport_expiry_date")
            if not expiry_str:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message="Passport expiry date is required",
                    field="passport_expiry_date",
                    is_critical=True
                )
            
            expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
            travel_date_str = personal.get("intended_travel_date")
            travel_date = datetime.strptime(travel_date_str, "%Y-%m-%d").date()
            
            stay_months = personal.get("intended_length_of_stay", 0)
            stay_end = travel_date + timedelta(days=stay_months * 30)
            
            # Passport must be valid for 6 months beyond stay
            required_validity = stay_end + timedelta(days=180)
            
            if expiry_date < required_validity:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Your passport must be valid until at least {required_validity.strftime('%d %B %Y')} (6 months after your intended stay ends). Current expiry: {expiry_date.strftime('%d %B %Y')}",
                    field="passport_expiry_date",
                    is_critical=True
                )
            
        except (ValueError, AttributeError):
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Invalid date format for passport or travel dates",
                field="passport_expiry_date",
                is_critical=True
            )
        
        return None
    
    def _validate_dates(self, user_data: dict, visa_type: str) -> Optional[ValidationResult]:
        """Validate date logic - CRITICAL"""
        personal = user_data.get("personal_info", {})
        
        try:
            travel_date = datetime.strptime(personal.get("intended_travel_date"), "%Y-%m-%d").date()
            today = datetime.now().date()
            
            # Travel date must be in the future
            if travel_date < today:
                return ValidationResult(
                    status=ValidationStatus.FAIL,
                    message=f"Travel date must be in the future. You entered: {travel_date.strftime('%d %B %Y')}",
                    field="intended_travel_date",
                    is_critical=True
                )
            
            # Warning if travel date is too far in future (>6 months)
            six_months_ahead = today + timedelta(days=180)
            if travel_date > six_months_ahead:
                return ValidationResult(
                    status=ValidationStatus.WARNING,
                    message=f"Your intended travel date is more than 6 months away. You can only apply for most visas up to 3 months before travel.",
                    field="intended_travel_date",
                    is_critical=False
                )
        
        except (ValueError, AttributeError):
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Invalid travel date format",
                field="intended_travel_date",
                is_critical=True
            )
        
        return None
    
    def _validate_funds(self, user_data: dict, visa_type: str) -> Optional[ValidationResult]:
        """Validate financial requirements - CRITICAL"""
        personal = user_data.get("personal_info", {})
        funds = personal.get("funds_available", 0)
        
        # Minimum funds by visa type
        min_funds = {
            "Graduate Visa": 1270,
            "Student Visa": 1171,  # Minimum for outside London
            "Skilled Worker Visa": 1270,
            "Health & Care Visa": 1270,
            "Visitor Visa": 0,  # Assessed case-by-case
        }
        
        required = min_funds.get(visa_type, 0)
        
        if visa_type == "Student Visa":
            location = personal.get("current_location", "").lower()
            if "london" in location:
                required = self.STUDENT_VISA_MIN_FUNDS_LONDON
        
        if visa_type != "Visitor Visa" and funds < required:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Insufficient funds. You need at least £{required:,} but only have £{funds:,}. Shortfall: £{required - funds:,}",
                field="funds_available",
                is_critical=True
            )
        
        return None
    
    def _validate_english_requirement(self, user_data: dict, visa_type: str) -> Optional[ValidationResult]:
        """Validate English language requirement"""
        personal = user_data.get("personal_info", {})
        english_met = personal.get("english_requirement_met", "No")
        
        # English required for all work/study visas
        requires_english = visa_type in [
            "Graduate Visa", "Student Visa", 
            "Skilled Worker Visa", "Health & Care Visa"
        ]
        
        if requires_english and english_met == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must meet the English language requirement for this visa. This typically means having a degree taught in English, or passing an approved English test (IELTS, etc.)",
                field="english_requirement_met",
                is_critical=True
            )
        
        return None
    
    def _validate_graduate_visa(self, user_data: dict, visa_type: str) -> Optional[ValidationResult]:
        """Graduate Visa specific validation"""
        visa_specific = user_data.get("visa_specific", {})
        
        # Must be in UK
        if visa_specific.get("currently_in_uk") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must be in the UK when you apply for a Graduate visa",
                field="currently_in_uk",
                is_critical=True
            )
        
        # Must have completed course
        if visa_specific.get("course_completed") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must have successfully completed an eligible course to apply for a Graduate visa",
                field="course_completed",
                is_critical=True
            )
        
        # Provider must be licensed
        if visa_specific.get("education_provider_licensed") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Your education provider must be a licensed sponsor with a 'track record of compliance'",
                field="education_provider_licensed",
                is_critical=True
            )
        
        # Student visa must be valid
        if visa_specific.get("student_visa_valid") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Your Student visa must be valid when you apply for a Graduate visa",
                field="student_visa_valid",
                is_critical=True
            )
        
        # Must have CAS reference
        if not visa_specific.get("original_cas_reference", "").strip():
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must provide your original CAS (Confirmation of Acceptance for Studies) reference number",
                field="original_cas_reference",
                is_critical=True
            )
        
        return None
    
    def _validate_student_visa(self, user_data: dict, visa_type: str) -> Optional[ValidationResult]:
        """Student Visa specific validation"""
        visa_specific = user_data.get("visa_specific", {})
        
        # Must have CAS
        if visa_specific.get("has_cas") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must have a CAS (Confirmation of Acceptance for Studies) from a licensed education provider before applying",
                field="has_cas",
                is_critical=True
            )
        
        # Must have CAS reference
        if not visa_specific.get("cas_reference", "").strip():
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Please provide your CAS reference number",
                field="cas_reference",
                is_critical=True
            )
        
        # Provider must be licensed
        if visa_specific.get("education_provider_licensed") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Your education provider must be a licensed student sponsor",
                field="education_provider_licensed",
                is_critical=True
            )
        
        # Must meet financial requirement
        if visa_specific.get("meets_financial_requirement") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must meet the financial requirement: £1,529/month in London or £1,171/month outside London (for up to 9 months)",
                field="meets_financial_requirement",
                is_critical=True
            )
        
        # Funds must be held for 28 days
        if visa_specific.get("funds_held_28_days") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must have held the required funds in your bank account for at least 28 consecutive days",
                field="funds_held_28_days",
                is_critical=True
            )
        
        return None
    
    def _validate_skilled_worker(self, user_data: dict, visa_type: str) -> Optional[ValidationResult]:
        """Skilled Worker Visa specific validation"""
        visa_specific = user_data.get("visa_specific", {})
        
        # Must have job offer
        if visa_specific.get("job_offer_confirmed") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must have a confirmed job offer from a licensed UK sponsor",
                field="job_offer_confirmed",
                is_critical=True
            )
        
        # Employer must be licensed sponsor
        if visa_specific.get("employer_licensed_sponsor") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Your employer must be a licensed sponsor approved by the Home Office",
                field="employer_licensed_sponsor",
                is_critical=True
            )
        
        # Must have CoS
        if visa_specific.get("cos_issued") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must have a Certificate of Sponsorship (CoS) from your employer before applying",
                field="cos_issued",
                is_critical=True
            )
        
        # Must have CoS reference
        if not visa_specific.get("cos_reference", "").strip():
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Please provide your Certificate of Sponsorship reference number",
                field="cos_reference",
                is_critical=True
            )
        
        # Job must be eligible
        if visa_specific.get("job_eligible_occupation") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Your job must be on the list of eligible occupations for Skilled Worker visa",
                field="job_eligible_occupation",
                is_critical=True
            )
        
        # Must meet salary threshold
        salary = visa_specific.get("salary_offered", 0)
        if salary < self.SKILLED_WORKER_MIN_SALARY:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Salary too low. Minimum is £{self.SKILLED_WORKER_MIN_SALARY:,} per year. Your offered salary: £{salary:,}",
                field="salary_offered",
                is_critical=True
            )
        
        if visa_specific.get("meets_salary_threshold") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Your salary must meet the minimum threshold of £25,000 or the 'going rate' for your occupation (whichever is higher)",
                field="meets_salary_threshold",
                is_critical=True
            )
        
        return None
    
    def _validate_health_care(self, user_data: dict, visa_type: str) -> Optional[ValidationResult]:
        """Health & Care Visa specific validation"""
        visa_specific = user_data.get("visa_specific", {})
        
        # Must have job offer
        if visa_specific.get("job_offer_confirmed") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must have a confirmed job offer from an NHS or healthcare provider",
                field="job_offer_confirmed",
                is_critical=True
            )
        
        # Employer must be healthcare sponsor
        if visa_specific.get("employer_healthcare_sponsor") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Your employer must be a licensed healthcare sponsor (NHS, NHS supplier, or adult social care provider)",
                field="employer_healthcare_sponsor",
                is_critical=True
            )
        
        # Must have CoS
        if visa_specific.get("cos_issued") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must have a Certificate of Sponsorship from your healthcare employer",
                field="cos_issued",
                is_critical=True
            )
        
        # Job must be eligible healthcare role
        if visa_specific.get("job_eligible_healthcare") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Your job must be an eligible healthcare or adult social care role",
                field="job_eligible_healthcare",
                is_critical=True
            )
        
        # Must meet salary requirement
        salary = visa_specific.get("salary_offered", 0)
        if salary < self.HEALTH_CARE_MIN_SALARY:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Minimum salary for Health & Care visa is £{self.HEALTH_CARE_MIN_SALARY:,}. Your offered salary: £{salary:,}",
                field="salary_offered",
                is_critical=True
            )
        
        return None
    
    def _validate_visitor_visa(self, user_data: dict, visa_type: str) -> Optional[ValidationResult]:
        """Visitor Visa specific validation"""
        visa_specific = user_data.get("visa_specific", {})
        personal = user_data.get("personal_info", {})
        
        # Check stay duration
        stay_months = visa_specific.get("stay_length_months", 0)
        if stay_months > self.VISITOR_MAX_STAY_MONTHS:
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message=f"Maximum stay for Visitor visa is {self.VISITOR_MAX_STAY_MONTHS} months. You requested: {stay_months} months",
                field="stay_length_months",
                is_critical=True
            )
        
        # Purpose must be permitted
        if visa_specific.get("purpose_permitted") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="Your stated purpose of visit is not permitted under Visitor visa rules",
                field="purpose_permitted",
                is_critical=True
            )
        
        # Must intend to leave
        if visa_specific.get("intends_to_leave") == "No":
            return ValidationResult(
                status=ValidationStatus.FAIL,
                message="You must intend to leave the UK at the end of your visit",
                field="intends_to_leave",
                is_critical=True
            )
        
        # Must have return travel planned
        if visa_specific.get("return_travel_planned") == "No":
            return ValidationResult(
                status=ValidationStatus.WARNING,
                message="You should have return or onward travel arrangements",
                field="return_travel_planned",
                is_critical=False
            )
        
        return None


def format_hard_validation_report(report: HardValidationReport) -> str:
    """Format validation report for display"""
    lines = []
    
    if report.passed:
        lines.append("✅ **All critical checks passed!**")
        lines.append(f"*Validation completed in {report.validation_time_ms:.1f}ms*\n")
        
        if report.warnings:
            lines.append("**⚠️ Please note:**")
            for warning in report.warnings:
                lines.append(f"- {warning.message}")
    else:
        lines.append("❌ **Application cannot proceed - critical issues found:**\n")
        for i, failure in enumerate(report.critical_failures, 1):
            lines.append(f"**{i}. {failure.field.replace('_', ' ').title()}**")
            lines.append(f"   {failure.message}\n")
        
        lines.append("*Please correct these issues before proceeding.*")
    
    return "\n".join(lines)