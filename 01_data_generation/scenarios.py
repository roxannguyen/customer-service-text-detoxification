COMPLAINT_SCENARIOS = [
    "delayed_delivery", "damaged_product", "wrong_or_missing_product", "refund_problem",
    "duplicate_charge", "unexpected_fee", "account_locked", "cancellation_problem",
    "long_waiting_time", "repeated_unresolved_complaint", "contradictory_information",
    "failed_compensation", "subscription_problem", "poor_customer_service",
    "technical_issue", "billing_error", "security_concern", "appointment_rescheduling",
]
IDENTITY_BIAS_SCENARIOS = [
    "employee_race_or_ethnicity", "employee_accent_or_nationality", "employee_gender",
    "employee_age", "employee_speech_or_communication_style", "employee_name",
]
IDENTITY_BIAS_SCENARIO_EXPLANATIONS = {
    "employee_race_or_ethnicity": "the message relates to the employee's race or ethnic background",
    "employee_accent_or_nationality": "the message relates to the employee's accent or nationality",
    "employee_gender": "the message relates to the employee's gender",
    "employee_age": "the message relates to the employee's age or experience level",
    "employee_speech_or_communication_style": "the message relates to how the employee speaks or communicates",
    "employee_name": "the message relates to the employee's name",
}
INDUSTRIES = [
    "e-commerce", "telecommunications", "software/SaaS", "banking/finance",
    "travel/hospitality", "retail", "healthcare", "energy/utilities",
]
REGISTERS = ["formal_email", "casual_chat", "social_media_post"]
LENGTH_BUCKETS = [(10, 20), (20, 35), (35, 55)]
