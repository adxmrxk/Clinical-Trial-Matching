from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class BiologicalSex(str, Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


class PatientProfile(BaseModel):
    """
    Structured patient profile built from conversational intake.
    Based on common eligibility criteria across clinical trials.
    """

    # Demographics (Phase 1 - Baseline)
    age: Optional[int] = Field(None, description="Patient age in years")
    biological_sex: Optional[BiologicalSex] = Field(None, description="Biological sex")

    # Primary Condition (Phase 1 - Baseline)
    primary_condition: Optional[str] = Field(None, description="Primary diagnosis/condition")
    condition_stage: Optional[str] = Field(None, description="Stage or severity of condition")
    diagnosis_date: Optional[str] = Field(None, description="When condition was diagnosed")

    # Location (Phase 1 - Baseline)
    country: Optional[str] = Field(None, description="Country of residence")
    state_province: Optional[str] = Field(None, description="State or province")
    city: Optional[str] = Field(None, description="City")
    willing_to_travel: Optional[bool] = Field(None, description="Willing to travel for trial")

    # Medical History (Phase 2 - Trial-driven)
    comorbidities: List[str] = Field(default_factory=list, description="Other medical conditions")
    prior_treatments: List[str] = Field(default_factory=list, description="Previous treatments received")
    current_medications: List[str] = Field(default_factory=list, description="Current medications")
    allergies: List[str] = Field(default_factory=list, description="Known allergies")

    # Lab Values (Phase 2 - Trial-driven)
    lab_values: dict = Field(default_factory=dict, description="Recent lab test results")

    # Lifestyle (Phase 3 - Gap-filling)
    smoking_status: Optional[str] = Field(None, description="Smoking status")
    alcohol_use: Optional[str] = Field(None, description="Alcohol consumption")
    pregnancy_status: Optional[str] = Field(None, description="Pregnancy status if applicable")

    # Performance Status (Phase 2 - Trial-driven)
    ecog_status: Optional[int] = Field(None, ge=0, le=5, description="ECOG performance status 0-5")

    # Extracted attributes that don't fit standard fields
    additional_attributes: dict = Field(default_factory=dict, description="Other extracted attributes")


class PatientProfileUpdate(BaseModel):
    """Partial update to patient profile from conversation turn."""
    extracted_attributes: dict = Field(default_factory=dict)
    confidence_scores: dict = Field(default_factory=dict)
    raw_text: str = Field(..., description="Original user message")


class ConversationState(BaseModel):
    """Tracks the state of the conversation and patient profile."""
    session_id: str
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)
    phase: int = Field(1, description="Current questioning phase (1-3)")
    messages: List[dict] = Field(default_factory=list)
    missing_required_fields: List[str] = Field(default_factory=list)
    candidate_trials: List[str] = Field(default_factory=list, description="Trial IDs being considered")

    # Track which baseline questions have been asked (for questions where empty is valid)
    asked_medications: bool = Field(False, description="Whether we've asked about current medications")
    asked_prior_treatments: bool = Field(False, description="Whether we've asked about prior treatments")
