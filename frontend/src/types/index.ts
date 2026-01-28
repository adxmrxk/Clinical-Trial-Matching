export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

export interface StructuredCriterion {
  criterion_id: string;
  original_text: string;
  criterion_type: 'inclusion' | 'exclusion';
  attribute?: string;
  operator?: string;
  value?: string;
  status: 'satisfied' | 'violated' | 'unknown';
  patient_value?: string;
  explanation?: string;
}

export interface ClinicalTrialRaw {
  nct_id: string;
  title: string;
  brief_summary?: string;
  overall_status: string;
  phase?: string;
  conditions: string[];
  eligibility_criteria_text?: string;
  minimum_age?: string;
  maximum_age?: string;
  sex?: string;
  lead_sponsor?: string;
  locations: Array<{
    facility?: string;
    city?: string;
    state?: string;
    country?: string;
  }>;
  contacts?: Array<{
    name?: string;
    email?: string;
    phone?: string;
  }>;
}

export interface TrialMatch {
  trial: ClinicalTrialRaw;
  eligibility_status: EligibilityStatus;
  criteria_satisfied: StructuredCriterion[];
  criteria_violated: StructuredCriterion[];
  criteria_unknown: StructuredCriterion[];
  explanation: string;
  confidence_score: number;
  missing_information: string[];
}

// Simplified version for UI display
export interface ClinicalTrial {
  id: string;  // NCT ID
  nctId: string;  // NCT ID for display
  title: string;
  status: string;
  condition: string;
  location: string;
  eligibility: EligibilityStatus;
  explanation: string;
  criteriaMatched: string[];
  criteriaViolated: string[];
  criteriaUnknown: string[];
  // Additional details
  phase?: string;
  briefSummary?: string;
  ageRange?: string;
  sponsor?: string;
  startDate?: string;
  completionDate?: string;
  enrollmentCount?: number;
  contactName?: string;
  contactEmail?: string;
  contactPhone?: string;
  officialUrl: string;  // Link to ClinicalTrials.gov
}

export interface PatientProfile {
  age?: number;
  biological_sex?: string;
  primary_condition?: string;
  condition_stage?: string;
  country?: string;
  state_province?: string;
  city?: string;
  comorbidities?: string[];
  current_medications?: string[];
  prior_treatments?: string[];
  allergies?: string[];
}

export type EligibilityStatus = 'eligible' | 'ineligible' | 'uncertain';

export interface ChatResponse {
  session_id: string;
  message: Message;
  messages: Message[];  // Multiple messages for phase transitions
  trial_matches: TrialMatch[];
  patient_profile_updated: boolean;
  current_phase: number;
  follow_up_questions: string[];
}
