"""
Rule-Based Clinical Trial Eligibility Matcher

Based on methodologies from:
- PMC6993990: GATE/JAPE pattern matching with dictionaries
- PMC11570988: Hierarchical rule-based algorithm with regex
- TrialMatcher: NER + structured matching approach

This module implements a pure rule-based approach for matching
patient profiles against clinical trial eligibility criteria.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta


# =============================================================================
# STEP 1: CRITERION CATEGORIES AND PATTERN DICTIONARIES
# =============================================================================

class CriterionCategory(str, Enum):
    """Categories of eligibility criteria for rule-based matching."""
    DEMOGRAPHIC = "demographic"          # Age, sex, location
    DIAGNOSIS = "diagnosis"              # Primary condition, staging
    LAB_VALUE = "lab_value"              # Blood counts, biomarkers
    MEDICATION = "medication"            # Current/prior medications
    COMORBIDITY = "comorbidity"          # Other medical conditions
    TREATMENT_HISTORY = "treatment_history"  # Prior treatments
    LIFESTYLE = "lifestyle"              # Smoking, alcohol, pregnancy
    PERFORMANCE_STATUS = "performance_status"  # ECOG, Karnofsky
    TEMPORAL = "temporal"                # Time-based constraints
    GENETIC = "genetic"                  # Biomarkers, mutations
    ORGAN_FUNCTION = "organ_function"    # Kidney, liver function


@dataclass
class MedicalDictionary:
    """
    Dictionaries for medical term matching.
    Based on PMC6993990: "14 manually crafted dictionaries with terms,
    abbreviations, and synonyms"
    """

    # Condition synonyms (expandable)
    conditions: Dict[str, List[str]] = field(default_factory=lambda: {
        "diabetes": ["diabetes", "diabetic", "dm", "dm2", "dm1", "type 2 diabetes",
                     "type 1 diabetes", "t2dm", "t1dm", "diabetes mellitus"],
        "hypertension": ["hypertension", "htn", "high blood pressure", "elevated bp"],
        "heart_disease": ["heart disease", "cardiac disease", "cad", "coronary artery disease",
                          "heart failure", "chf", "congestive heart failure", "mi",
                          "myocardial infarction", "heart attack"],
        "cancer": ["cancer", "carcinoma", "malignancy", "malignant", "tumor", "tumour",
                   "neoplasm", "oncologic", "metastatic", "metastasis"],
        "kidney_disease": ["kidney disease", "renal disease", "ckd", "chronic kidney disease",
                           "renal failure", "kidney failure", "esrd", "end stage renal"],
        "liver_disease": ["liver disease", "hepatic disease", "cirrhosis", "hepatitis",
                          "liver failure", "hepatic failure"],
        "lung_disease": ["lung disease", "pulmonary disease", "copd", "asthma",
                         "respiratory disease", "pulmonary fibrosis"],
        "stroke": ["stroke", "cva", "cerebrovascular accident", "tia",
                   "transient ischemic attack", "brain attack"],
        "hiv": ["hiv", "human immunodeficiency virus", "aids", "hiv positive", "hiv+"],
        "depression": ["depression", "major depressive disorder", "mdd", "depressive disorder"],
        "anxiety": ["anxiety", "anxiety disorder", "gad", "generalized anxiety"],
    })

    # ECOG status descriptions
    ecog_descriptions: Dict[int, List[str]] = field(default_factory=lambda: {
        0: ["fully active", "no restrictions", "normal activity", "asymptomatic"],
        1: ["restricted", "light work", "ambulatory", "symptomatic but ambulatory"],
        2: ["ambulatory", "up and about", "self-care", "unable to work",
            "50% of waking hours"],
        3: ["limited self-care", "confined to bed", "more than 50% of day in bed"],
        4: ["completely disabled", "cannot self-care", "totally confined to bed"],
        5: ["dead", "deceased"],
    })

    # Smoking status terms
    smoking_terms: Dict[str, List[str]] = field(default_factory=lambda: {
        "never": ["never smoked", "non-smoker", "nonsmoker", "never smoker",
                  "no smoking history", "no tobacco"],
        "former": ["former smoker", "ex-smoker", "quit smoking", "stopped smoking",
                   "previous smoker", "past smoker"],
        "current": ["current smoker", "active smoker", "smokes", "smoking",
                    "tobacco user", "cigarette user"],
    })

    # Pregnancy status terms
    pregnancy_terms: Dict[str, List[str]] = field(default_factory=lambda: {
        "pregnant": ["pregnant", "pregnancy", "expecting", "with child", "gravid"],
        "not_pregnant": ["not pregnant", "non-pregnant", "negative pregnancy test"],
        "postpartum": ["postpartum", "post-partum", "after delivery", "nursing",
                       "breastfeeding", "lactating"],
    })

    # Common lab test synonyms
    lab_tests: Dict[str, List[str]] = field(default_factory=lambda: {
        "hemoglobin": ["hemoglobin", "hgb", "hb", "haemoglobin"],
        "hematocrit": ["hematocrit", "hct"],
        "wbc": ["wbc", "white blood cell", "white blood count", "leukocyte"],
        "platelet": ["platelet", "plt", "platelet count", "thrombocyte"],
        "creatinine": ["creatinine", "cr", "serum creatinine", "scr"],
        "egfr": ["egfr", "gfr", "glomerular filtration rate", "estimated gfr"],
        "bilirubin": ["bilirubin", "tbili", "total bilirubin"],
        "alt": ["alt", "sgpt", "alanine aminotransferase", "alanine transaminase"],
        "ast": ["ast", "sgot", "aspartate aminotransferase", "aspartate transaminase"],
        "hba1c": ["hba1c", "a1c", "hemoglobin a1c", "glycated hemoglobin"],
        "inr": ["inr", "international normalized ratio", "prothrombin time"],
        "potassium": ["potassium", "k", "k+", "serum potassium"],
        "sodium": ["sodium", "na", "na+", "serum sodium"],
        "glucose": ["glucose", "blood sugar", "fasting glucose", "blood glucose"],
    })

    # Negation terms (for detecting excluded conditions)
    negation_terms: List[str] = field(default_factory=lambda: [
        "no", "not", "none", "without", "absence of", "negative for",
        "denies", "denied", "no history of", "no evidence of", "ruled out",
        "exclude", "excluded", "never had", "free of", "lacks"
    ])


# =============================================================================
# STEP 2: REGEX PATTERNS FOR NUMERIC CONSTRAINT EXTRACTION
# =============================================================================

@dataclass
class NumericPatterns:
    """
    Regex patterns for extracting numeric constraints from criteria.
    Based on PMC6993990 and PMC11570988 methodologies.
    """

    # Age patterns
    age_patterns: List[str] = field(default_factory=lambda: [
        r"(?:age|aged?)\s*(?:of\s*)?([<>=≤≥]+)\s*(\d+)",  # "age >= 18"
        r"(\d+)\s*(?:years?|yrs?)\s*(?:of age|old)?\s*(?:or\s*)?(older|younger|and above|and below|or more|or less)?",  # "18 years or older"
        r"(?:at least|minimum|minimum age[:\s]*)\s*(\d+)",  # "at least 18"
        r"(?:no more than|maximum|maximum age[:\s]*)\s*(\d+)",  # "no more than 65"
        r"(?:between|from)\s*(\d+)\s*(?:and|to|-)\s*(\d+)\s*(?:years?)?",  # "between 18 and 65"
        r"(?:older than|greater than|above|over)\s*(\d+)",  # "older than 18"
        r"(?:younger than|less than|under|below)\s*(\d+)",  # "younger than 65"
    ])

    # Lab value patterns with units
    lab_value_patterns: List[str] = field(default_factory=lambda: [
        # Pattern: "hemoglobin >= 10 g/dL"
        r"([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s*([<>=≤≥]+)\s*(\d+\.?\d*)\s*([a-zA-Z/%]+)?",
        # Pattern: "creatinine of 1.5 mg/dL or less"
        r"([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s*(?:of|at|level)?\s*(\d+\.?\d*)\s*([a-zA-Z/%]+)?\s*(?:or\s*)?(more|less|higher|lower|above|below)?",
        # Pattern: "platelet count between 100,000 and 400,000"
        r"([a-zA-Z]+(?:\s+[a-zA-Z]+)?)\s*(?:between|from)\s*(\d+[,\d]*)\s*(?:and|to|-)\s*(\d+[,\d]*)",
        # Pattern: "ANC > 1500/mm3" or "ANC > 1.5 x 10^9/L"
        r"([a-zA-Z]+)\s*([<>=≤≥]+)\s*(\d+\.?\d*)\s*(?:x\s*10\^?\d+)?\s*(/[a-zA-Z³]+)?",
    ])

    # ECOG/Performance status patterns
    ecog_patterns: List[str] = field(default_factory=lambda: [
        r"ecog\s*(?:performance\s*)?(?:status|score|ps)?\s*(?:of\s*)?([<>=≤≥]+)?\s*(\d)",
        r"(?:performance\s*status|ps)\s*(?:of\s*)?([<>=≤≥]+)?\s*(\d)",
        r"ecog\s*(\d)\s*(?:or\s*)?(less|lower|better|worse)?",
        r"ecog\s*(\d)\s*(?:-|to)\s*(\d)",  # "ECOG 0-2"
    ])

    # Time duration patterns
    temporal_patterns: List[str] = field(default_factory=lambda: [
        r"(?:within|in the (?:past|last)|during the (?:past|last))\s*(\d+)\s*(days?|weeks?|months?|years?)",
        r"(?:at least|minimum of?)\s*(\d+)\s*(days?|weeks?|months?|years?)\s*(?:since|after|before|prior)",
        r"(?:no|not)\s*(?:more than|less than)\s*(\d+)\s*(days?|weeks?|months?|years?)",
        r"(\d+)\s*(days?|weeks?|months?|years?)\s*(?:ago|prior|before|since)",
    ])

    # Stage patterns (cancer staging)
    stage_patterns: List[str] = field(default_factory=lambda: [
        r"stage\s*([I]{1,4}|[0-4]|[IViv]{1,4})[ABC]?",  # "Stage III", "Stage 3", "Stage IIIA"
        r"(?:t|tumor)\s*([0-4])[abc]?\s*n\s*([0-3])[abc]?\s*m\s*([0-1])",  # TNM: T2N1M0
        r"(?:locally\s*)?(?:advanced|metastatic|early|late)\s*(?:stage)?",
    ])


# Global instances
MEDICAL_DICTIONARY = MedicalDictionary()
NUMERIC_PATTERNS = NumericPatterns()


# =============================================================================
# STEP 3: RULE-BASED EVALUATORS FOR EACH CRITERION TYPE
# =============================================================================

@dataclass
class EvaluationResult:
    """Result of evaluating a single criterion."""
    status: str  # "satisfied", "violated", "unknown"
    patient_value: Optional[str] = None
    explanation: str = ""
    missing_attribute: Optional[str] = None
    confidence: float = 1.0  # 1.0 for rule-based (deterministic)


class RuleBasedEvaluator:
    """
    Core rule-based evaluator for clinical trial eligibility criteria.

    Implements the hierarchical approach from PMC11570988:
    1. Parse criterion to identify category and constraints
    2. Extract relevant patient data
    3. Apply category-specific rules
    4. Return deterministic result
    """

    def __init__(self):
        self.dictionary = MEDICAL_DICTIONARY
        self.patterns = NUMERIC_PATTERNS

    # -------------------------------------------------------------------------
    # MAIN EVALUATION ENTRY POINT
    # -------------------------------------------------------------------------

    def evaluate(
        self,
        criterion_text: str,
        criterion_type: str,  # "inclusion" or "exclusion"
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient_profile: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate a criterion against a patient profile using rules.

        Args:
            criterion_text: Original text of the criterion
            criterion_type: "inclusion" or "exclusion"
            attribute: Parsed attribute (e.g., "age", "hemoglobin")
            operator: Parsed operator (e.g., ">=", "has", "not")
            value: Parsed threshold or value
            patient_profile: Patient data dictionary

        Returns:
            EvaluationResult with status, explanation, and confidence
        """
        # Detect the category of this criterion
        category = self._detect_category(criterion_text, attribute)

        # Route to appropriate evaluator
        evaluators = {
            CriterionCategory.DEMOGRAPHIC: self._evaluate_demographic,
            CriterionCategory.LAB_VALUE: self._evaluate_lab_value,
            CriterionCategory.PERFORMANCE_STATUS: self._evaluate_performance_status,
            CriterionCategory.LIFESTYLE: self._evaluate_lifestyle,
            CriterionCategory.COMORBIDITY: self._evaluate_comorbidity,
            CriterionCategory.MEDICATION: self._evaluate_medication,
            CriterionCategory.TREATMENT_HISTORY: self._evaluate_treatment_history,
            CriterionCategory.DIAGNOSIS: self._evaluate_diagnosis,
            CriterionCategory.TEMPORAL: self._evaluate_temporal,
            CriterionCategory.ORGAN_FUNCTION: self._evaluate_organ_function,
        }

        evaluator = evaluators.get(category)
        if evaluator:
            result = evaluator(
                criterion_text, attribute, operator, value, patient_profile
            )
            # Handle exclusion criteria inversion
            # For exclusion: if patient HAS the condition -> VIOLATED
            # The evaluator returns whether the criterion is met
            return result

        # If no specific evaluator, return unknown
        return EvaluationResult(
            status="unknown",
            explanation=f"No rule-based evaluator for category: {category}",
            missing_attribute=attribute
        )

    def _detect_category(
        self,
        criterion_text: str,
        attribute: Optional[str]
    ) -> CriterionCategory:
        """Detect the category of a criterion based on text and attribute."""
        text_lower = criterion_text.lower()
        attr_lower = (attribute or "").lower()

        # Demographic criteria
        if attr_lower in ["age", "sex", "gender", "biological_sex"]:
            return CriterionCategory.DEMOGRAPHIC
        if any(term in text_lower for term in ["years old", "years of age", "male", "female"]):
            return CriterionCategory.DEMOGRAPHIC

        # Performance status
        if any(term in text_lower for term in ["ecog", "karnofsky", "performance status", "ps "]):
            return CriterionCategory.PERFORMANCE_STATUS

        # Lifestyle - CHECK BEFORE LAB VALUES to avoid false matches
        if any(term in text_lower for term in ["smok", "tobacco", "alcohol", "pregnant", "pregnancy",
                                                 "breastfeed", "lactating", "drug use", "substance"]):
            return CriterionCategory.LIFESTYLE

        # Lab values - use word boundary matching for short synonyms
        lab_indicators = list(self.dictionary.lab_tests.keys())
        if attr_lower in lab_indicators:
            return CriterionCategory.LAB_VALUE

        # Check for lab value terms with proper word boundaries
        for lab_name, synonyms in self.dictionary.lab_tests.items():
            for syn in synonyms:
                # For short synonyms (<=2 chars), require word boundaries
                if len(syn) <= 2:
                    pattern = rf'\b{re.escape(syn)}\b'
                    if re.search(pattern, text_lower):
                        return CriterionCategory.LAB_VALUE
                else:
                    if syn in text_lower:
                        return CriterionCategory.LAB_VALUE

        # Medications
        if any(term in text_lower for term in ["medication", "taking", "receiving", "treatment with",
                                                 "therapy", "drug", "on treatment"]):
            return CriterionCategory.MEDICATION

        # Treatment history
        if any(term in text_lower for term in ["prior", "previous", "history of treatment",
                                                 "received", "undergone", "had treatment"]):
            return CriterionCategory.TREATMENT_HISTORY

        # Comorbidities
        if any(term in text_lower for term in ["history of", "diagnosis of", "comorbid",
                                                 "concurrent", "other condition"]):
            return CriterionCategory.COMORBIDITY

        # Organ function
        if any(term in text_lower for term in ["liver function", "renal function", "kidney function",
                                                 "hepatic", "cardiac function", "lung function"]):
            return CriterionCategory.ORGAN_FUNCTION

        # Temporal
        if any(term in text_lower for term in ["within", "days", "weeks", "months", "years ago",
                                                 "prior to", "before", "since"]):
            return CriterionCategory.TEMPORAL

        # Default to diagnosis
        return CriterionCategory.DIAGNOSIS

    # -------------------------------------------------------------------------
    # DEMOGRAPHIC EVALUATOR (Age, Sex)
    # -------------------------------------------------------------------------

    def _evaluate_demographic(
        self,
        criterion_text: str,
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate demographic criteria (age, sex)."""
        attr = (attribute or "").lower()
        text_lower = criterion_text.lower()

        # AGE EVALUATION
        if attr == "age" or "age" in text_lower or "year" in text_lower:
            patient_age = patient.get("age")
            if patient_age is None:
                return EvaluationResult(
                    status="unknown",
                    explanation="Patient age not provided",
                    missing_attribute="age"
                )

            # Try to extract age constraint from text if not parsed
            min_age, max_age = self._extract_age_range(criterion_text, operator, value)

            if min_age is not None and max_age is not None:
                # Range check
                if min_age <= patient_age <= max_age:
                    return EvaluationResult(
                        status="satisfied",
                        patient_value=str(patient_age),
                        explanation=f"Patient age {patient_age} is within range {min_age}-{max_age}"
                    )
                else:
                    return EvaluationResult(
                        status="violated",
                        patient_value=str(patient_age),
                        explanation=f"Patient age {patient_age} is outside range {min_age}-{max_age}"
                    )
            elif min_age is not None:
                if patient_age >= min_age:
                    return EvaluationResult(
                        status="satisfied",
                        patient_value=str(patient_age),
                        explanation=f"Patient age {patient_age} meets minimum {min_age}"
                    )
                else:
                    return EvaluationResult(
                        status="violated",
                        patient_value=str(patient_age),
                        explanation=f"Patient age {patient_age} below minimum {min_age}"
                    )
            elif max_age is not None:
                if patient_age <= max_age:
                    return EvaluationResult(
                        status="satisfied",
                        patient_value=str(patient_age),
                        explanation=f"Patient age {patient_age} meets maximum {max_age}"
                    )
                else:
                    return EvaluationResult(
                        status="violated",
                        patient_value=str(patient_age),
                        explanation=f"Patient age {patient_age} exceeds maximum {max_age}"
                    )

        # SEX EVALUATION
        if attr in ["sex", "gender", "biological_sex"] or any(
            term in text_lower for term in ["male", "female", "sex", "gender"]
        ):
            patient_sex = patient.get("biological_sex")
            if not patient_sex:
                return EvaluationResult(
                    status="unknown",
                    explanation="Patient sex not provided",
                    missing_attribute="biological_sex"
                )

            # Handle enum or string
            if hasattr(patient_sex, 'value'):
                patient_sex = patient_sex.value
            patient_sex = patient_sex.lower()

            criterion_sex = (value or "").lower()

            # Extract sex requirement from text if not parsed
            if not criterion_sex:
                if "female" in text_lower and "male" not in text_lower.replace("female", ""):
                    criterion_sex = "female"
                elif "male" in text_lower:
                    criterion_sex = "male"

            if criterion_sex in ["all", "both", ""]:
                return EvaluationResult(
                    status="satisfied",
                    patient_value=patient_sex,
                    explanation="Trial accepts all sexes"
                )
            elif criterion_sex == patient_sex:
                return EvaluationResult(
                    status="satisfied",
                    patient_value=patient_sex,
                    explanation=f"Patient sex ({patient_sex}) matches requirement"
                )
            else:
                return EvaluationResult(
                    status="violated",
                    patient_value=patient_sex,
                    explanation=f"Trial requires {criterion_sex}, patient is {patient_sex}"
                )

        return EvaluationResult(
            status="unknown",
            explanation="Could not parse demographic criterion"
        )

    def _extract_age_range(
        self,
        text: str,
        operator: Optional[str],
        value: Optional[str]
    ) -> Tuple[Optional[int], Optional[int]]:
        """Extract age range from criterion text."""
        min_age, max_age = None, None
        text_lower = text.lower()

        # First try parsed values
        if value:
            try:
                val = int(value)
                if operator in [">=", "≥", ">", "at least", "minimum"]:
                    min_age = val if operator in [">=", "≥", "at least", "minimum"] else val + 1
                elif operator in ["<=", "≤", "<", "at most", "maximum"]:
                    max_age = val if operator in ["<=", "≤", "at most", "maximum"] else val - 1
                elif operator == "between":
                    # Value might be "18-65" format
                    if "-" in value:
                        parts = value.split("-")
                        min_age, max_age = int(parts[0]), int(parts[1])
            except ValueError:
                pass

        # Try regex patterns
        for pattern in self.patterns.age_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                groups = match.groups()
                if "between" in pattern or "from" in pattern:
                    min_age = int(groups[0])
                    max_age = int(groups[1])
                elif "older" in text_lower or "at least" in text_lower or "or more" in text_lower:
                    min_age = int(groups[0]) if groups[0].isdigit() else int(groups[1])
                elif "younger" in text_lower or "no more than" in text_lower or "or less" in text_lower:
                    max_age = int(groups[0]) if groups[0].isdigit() else int(groups[1])
                elif len(groups) >= 2 and groups[0] in [">=", "≥", ">"]:
                    min_age = int(groups[1])
                elif len(groups) >= 2 and groups[0] in ["<=", "≤", "<"]:
                    max_age = int(groups[1])
                break

        return min_age, max_age

    # -------------------------------------------------------------------------
    # LAB VALUE EVALUATOR
    # -------------------------------------------------------------------------

    def _evaluate_lab_value(
        self,
        criterion_text: str,
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate lab value criteria.
        Based on PMC6993990: "HBA1c met when values fall between 6.5% and 9.5%"
        """
        text_lower = criterion_text.lower()

        # Normalize the lab test name
        lab_name = self._normalize_lab_name(attribute or text_lower)
        if not lab_name:
            return EvaluationResult(
                status="unknown",
                explanation="Could not identify lab test",
                missing_attribute="lab_values"
            )

        # Get patient lab values
        lab_values = patient.get("lab_values", {})
        patient_value = lab_values.get(lab_name)

        if patient_value is None:
            # Try alternate names
            for normalized, synonyms in self.dictionary.lab_tests.items():
                if normalized == lab_name:
                    for syn in synonyms:
                        if syn in lab_values:
                            patient_value = lab_values[syn]
                            break

        if patient_value is None:
            return EvaluationResult(
                status="unknown",
                explanation=f"Patient {lab_name} value not provided",
                missing_attribute=f"lab_values.{lab_name}"
            )

        # Extract threshold from criterion
        threshold, comparison = self._extract_lab_threshold(criterion_text, operator, value)

        if threshold is None:
            return EvaluationResult(
                status="unknown",
                explanation=f"Could not extract threshold for {lab_name}"
            )

        # Compare values
        try:
            patient_val = float(str(patient_value).replace(",", ""))
            threshold_val = float(str(threshold).replace(",", ""))

            if comparison == ">=":
                met = patient_val >= threshold_val
            elif comparison == ">":
                met = patient_val > threshold_val
            elif comparison == "<=":
                met = patient_val <= threshold_val
            elif comparison == "<":
                met = patient_val < threshold_val
            elif comparison == "==":
                met = abs(patient_val - threshold_val) < 0.01
            elif comparison == "between":
                # threshold is tuple (min, max)
                met = threshold[0] <= patient_val <= threshold[1]
            else:
                return EvaluationResult(
                    status="unknown",
                    explanation=f"Unknown comparison operator: {comparison}"
                )

            status = "satisfied" if met else "violated"
            return EvaluationResult(
                status=status,
                patient_value=str(patient_value),
                explanation=f"Patient {lab_name}={patient_value}, criterion requires {comparison} {threshold}"
            )

        except (ValueError, TypeError) as e:
            return EvaluationResult(
                status="unknown",
                explanation=f"Error comparing values: {e}"
            )

    def _normalize_lab_name(self, text: str) -> Optional[str]:
        """Normalize lab test name to standard form."""
        text_lower = text.lower()
        for normalized, synonyms in self.dictionary.lab_tests.items():
            if normalized in text_lower or any(syn in text_lower for syn in synonyms):
                return normalized
        return None

    def _extract_lab_threshold(
        self,
        text: str,
        operator: Optional[str],
        value: Optional[str]
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Extract threshold and comparison from lab criterion."""
        # Try parsed values first
        if value and operator:
            op_map = {
                ">=": ">=", "≥": ">=", "greater than or equal": ">=",
                "<=": "<=", "≤": "<=", "less than or equal": "<=",
                ">": ">", "greater than": ">", "above": ">", "more than": ">",
                "<": "<", "less than": "<", "below": "<",
                "=": "==", "equals": "==", "equal to": "==",
            }
            normalized_op = op_map.get(operator.lower(), operator)
            try:
                return float(value.replace(",", "")), normalized_op
            except ValueError:
                pass

        # Try regex patterns
        for pattern in self.patterns.lab_value_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                groups = match.groups()
                # Pattern dependent parsing
                for g in groups:
                    if g and re.match(r'\d+\.?\d*', str(g)):
                        # Found a number
                        threshold = float(g.replace(",", ""))
                        # Determine comparison
                        if "or more" in text.lower() or "at least" in text.lower() or ">=" in text:
                            return threshold, ">="
                        elif "or less" in text.lower() or "at most" in text.lower() or "<=" in text:
                            return threshold, "<="
                        elif ">" in text:
                            return threshold, ">"
                        elif "<" in text:
                            return threshold, "<"
                        else:
                            return threshold, ">="  # Default for requirements
                break

        return None, None

    # -------------------------------------------------------------------------
    # PERFORMANCE STATUS EVALUATOR (ECOG)
    # -------------------------------------------------------------------------

    def _evaluate_performance_status(
        self,
        criterion_text: str,
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate ECOG/performance status criteria.
        ECOG 0-5 scale: lower is better (0=fully active, 5=dead)
        """
        patient_ecog = patient.get("ecog_status")
        if patient_ecog is None:
            return EvaluationResult(
                status="unknown",
                explanation="Patient ECOG status not provided",
                missing_attribute="ecog_status"
            )

        # Extract ECOG requirement
        text_lower = criterion_text.lower()
        required_ecog = None
        comparison = "<="  # Default: ECOG X or less (better)

        # Try parsed value
        if value:
            try:
                required_ecog = int(value)
            except ValueError:
                pass

        # Try regex patterns
        if required_ecog is None:
            for pattern in self.patterns.ecog_patterns:
                match = re.search(pattern, text_lower)
                if match:
                    groups = [g for g in match.groups() if g and g.isdigit()]
                    if groups:
                        if len(groups) == 2:
                            # Range like "ECOG 0-2"
                            required_ecog = int(groups[1])  # Max of range
                        else:
                            required_ecog = int(groups[0])
                        break

        if required_ecog is None:
            # Try simple number extraction
            ecog_match = re.search(r'ecog\s*(?:\w+\s*)*(\d)', text_lower)
            if ecog_match:
                required_ecog = int(ecog_match.group(1))

        if required_ecog is None:
            return EvaluationResult(
                status="unknown",
                explanation="Could not parse ECOG requirement from criterion"
            )

        # Determine comparison type
        if "or better" in text_lower or "or less" in text_lower:
            comparison = "<="
        elif "or worse" in text_lower or "or greater" in text_lower:
            comparison = ">="

        # Evaluate
        if comparison == "<=":
            met = patient_ecog <= required_ecog
        else:
            met = patient_ecog >= required_ecog

        status = "satisfied" if met else "violated"
        return EvaluationResult(
            status=status,
            patient_value=str(patient_ecog),
            explanation=f"Patient ECOG={patient_ecog}, criterion requires {comparison} {required_ecog}"
        )

    # -------------------------------------------------------------------------
    # LIFESTYLE EVALUATOR (Smoking, Alcohol, Pregnancy)
    # -------------------------------------------------------------------------

    def _evaluate_lifestyle(
        self,
        criterion_text: str,
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate lifestyle criteria (smoking, alcohol, pregnancy)."""
        text_lower = criterion_text.lower()
        is_negated = self._is_negated(criterion_text)

        # SMOKING
        if any(term in text_lower for term in ["smok", "tobacco", "cigarette"]):
            patient_smoking = patient.get("smoking_status")
            if not patient_smoking:
                return EvaluationResult(
                    status="unknown",
                    explanation="Patient smoking status not provided",
                    missing_attribute="smoking_status"
                )

            patient_smoking = patient_smoking.lower()
            patient_is_smoker = patient_smoking in ["current", "active", "smoker", "yes"]

            # Criterion wants non-smoker (exclusion of smokers)
            if is_negated or "non-smok" in text_lower or "never" in text_lower:
                # Requires non-smoker
                if patient_is_smoker:
                    return EvaluationResult(
                        status="violated",
                        patient_value=patient_smoking,
                        explanation="Criterion requires non-smoker, patient is a smoker"
                    )
                else:
                    return EvaluationResult(
                        status="satisfied",
                        patient_value=patient_smoking,
                        explanation="Patient is not a current smoker"
                    )
            else:
                # Criterion mentions smoking (might be requiring smokers, rare)
                return EvaluationResult(
                    status="satisfied" if patient_is_smoker else "unknown",
                    patient_value=patient_smoking,
                    explanation=f"Patient smoking status: {patient_smoking}"
                )

        # PREGNANCY
        if any(term in text_lower for term in ["pregnan", "lactating", "breastfeed", "nursing"]):
            patient_pregnancy = patient.get("pregnancy_status")
            patient_sex = patient.get("biological_sex")

            # If male, pregnancy not applicable
            if patient_sex:
                sex_val = patient_sex.value if hasattr(patient_sex, 'value') else patient_sex
                if sex_val.lower() == "male":
                    return EvaluationResult(
                        status="satisfied",
                        patient_value="male",
                        explanation="Pregnancy criterion not applicable to male patient"
                    )

            if not patient_pregnancy:
                return EvaluationResult(
                    status="unknown",
                    explanation="Patient pregnancy status not provided",
                    missing_attribute="pregnancy_status"
                )

            patient_is_pregnant = patient_pregnancy.lower() in ["pregnant", "yes", "positive"]

            # Most trials exclude pregnant patients
            if is_negated or "not pregnant" in text_lower or "must not be" in text_lower:
                if patient_is_pregnant:
                    return EvaluationResult(
                        status="violated",
                        patient_value=patient_pregnancy,
                        explanation="Criterion excludes pregnant patients"
                    )
                else:
                    return EvaluationResult(
                        status="satisfied",
                        patient_value=patient_pregnancy,
                        explanation="Patient is not pregnant"
                    )

        # ALCOHOL
        if any(term in text_lower for term in ["alcohol", "drink", "ethanol"]):
            patient_alcohol = patient.get("alcohol_use")
            if not patient_alcohol:
                return EvaluationResult(
                    status="unknown",
                    explanation="Patient alcohol use not provided",
                    missing_attribute="alcohol_use"
                )
            # Basic evaluation
            return EvaluationResult(
                status="satisfied",  # Default to satisfied unless heavy use detected
                patient_value=patient_alcohol,
                explanation=f"Patient alcohol use: {patient_alcohol}"
            )

        return EvaluationResult(
            status="unknown",
            explanation="Could not parse lifestyle criterion"
        )

    # =========================================================================
    # STEP 4: NEGATION DETECTION
    # Based on PMC6993990: "mentions involved in negated patterns were ignored"
    # =========================================================================

    def _is_negated(self, text: str) -> bool:
        """
        Detect if criterion text contains negation.
        Based on PMC6993990: Uses stopword lists like 'not', 'no', 'none'
        """
        text_lower = text.lower()

        # Check for negation terms at the start or before key phrases
        for neg_term in self.dictionary.negation_terms:
            # Pattern: negation term followed by content
            pattern = rf'\b{re.escape(neg_term)}\b'
            if re.search(pattern, text_lower):
                return True

        return False

    def _extract_negated_concept(self, text: str) -> Tuple[bool, str]:
        """
        Extract the negated concept from text.
        Returns (is_negated, concept)
        """
        text_lower = text.lower()

        for neg_term in self.dictionary.negation_terms:
            if neg_term in text_lower:
                # Get the text after the negation
                idx = text_lower.find(neg_term)
                remaining = text[idx + len(neg_term):].strip()
                return True, remaining

        return False, text

    # =========================================================================
    # STEP 5: LIST/SET MATCHING FOR COMORBIDITIES AND MEDICATIONS
    # Based on TrialMatcher: Dictionary-based matching with synonyms
    # =========================================================================

    def _evaluate_comorbidity(
        self,
        criterion_text: str,
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate comorbidity/medical history criteria.
        Uses dictionary matching with synonym expansion.
        """
        text_lower = criterion_text.lower()
        is_negated = self._is_negated(criterion_text)

        # Get patient comorbidities
        patient_conditions = patient.get("comorbidities", [])
        if isinstance(patient_conditions, str):
            patient_conditions = [patient_conditions]
        patient_conditions_lower = [c.lower() for c in patient_conditions]

        # Add primary condition to check
        primary = patient.get("primary_condition")
        if primary:
            patient_conditions_lower.append(primary.lower())

        # Identify which condition the criterion is about
        target_condition = None
        for condition_key, synonyms in self.dictionary.conditions.items():
            if any(syn in text_lower for syn in synonyms):
                target_condition = condition_key
                break

        if not target_condition:
            # Couldn't identify the condition being referenced
            return EvaluationResult(
                status="unknown",
                explanation="Could not identify condition in criterion",
                missing_attribute="comorbidities"
            )

        # Check if patient has this condition
        patient_has_condition = False
        matched_term = None

        for syn in self.dictionary.conditions[target_condition]:
            for patient_cond in patient_conditions_lower:
                if syn in patient_cond or patient_cond in syn:
                    patient_has_condition = True
                    matched_term = patient_cond
                    break
            if patient_has_condition:
                break

        # Determine result based on negation
        # Exclusion criterion with negation: "No history of heart disease"
        #   - Patient HAS heart disease -> VIOLATED
        #   - Patient NO heart disease -> SATISFIED
        if is_negated:
            # Criterion says patient should NOT have this condition
            if patient_has_condition:
                return EvaluationResult(
                    status="violated",
                    patient_value=matched_term or str(patient_conditions),
                    explanation=f"Patient has {target_condition}, but criterion excludes it"
                )
            else:
                return EvaluationResult(
                    status="satisfied",
                    patient_value="none reported",
                    explanation=f"Patient does not have {target_condition}"
                )
        else:
            # Criterion requires patient to have this condition (rare for inclusion)
            if patient_has_condition:
                return EvaluationResult(
                    status="satisfied",
                    patient_value=matched_term,
                    explanation=f"Patient has required condition: {target_condition}"
                )
            else:
                # Check if we have enough info
                if not patient_conditions:
                    return EvaluationResult(
                        status="unknown",
                        explanation="Patient comorbidities not provided",
                        missing_attribute="comorbidities"
                    )
                return EvaluationResult(
                    status="violated",
                    patient_value="none reported",
                    explanation=f"Patient does not have required: {target_condition}"
                )

    def _evaluate_medication(
        self,
        criterion_text: str,
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate medication-related criteria.
        Checks current medications against criterion requirements.
        """
        text_lower = criterion_text.lower()
        is_negated = self._is_negated(criterion_text)

        # Get patient medications
        patient_meds = patient.get("current_medications", [])
        if isinstance(patient_meds, str):
            patient_meds = [patient_meds]
        patient_meds_lower = [m.lower() for m in patient_meds]

        # Extract drug/medication names from criterion
        # This is simplified - a real system would use RxNorm or similar
        drug_patterns = [
            r'\b([a-zA-Z]+(?:mab|nib|ine|ole|ide|ate|ol)\b)',  # Common drug suffixes
            r'\b(aspirin|metformin|insulin|warfarin|heparin)\b',  # Common drugs
        ]

        mentioned_drugs = []
        for pattern in drug_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            mentioned_drugs.extend(matches)

        # Also check for value field if provided
        if value:
            mentioned_drugs.append(value.lower())

        if not mentioned_drugs:
            # Check for generic medication mentions
            if "medication" in text_lower or "drug" in text_lower or "treatment" in text_lower:
                if is_negated:
                    # "No concurrent medications" type criterion
                    if patient_meds:
                        return EvaluationResult(
                            status="unknown",
                            patient_value=str(patient_meds),
                            explanation="Patient has medications, need to verify if excluded"
                        )
                    else:
                        return EvaluationResult(
                            status="satisfied",
                            patient_value="none",
                            explanation="Patient has no current medications"
                        )
            return EvaluationResult(
                status="unknown",
                explanation="Could not identify specific medication in criterion"
            )

        # Check if patient is taking any mentioned drugs
        patient_taking = []
        for drug in mentioned_drugs:
            for patient_med in patient_meds_lower:
                if drug in patient_med or patient_med in drug:
                    patient_taking.append(patient_med)

        if is_negated:
            # Patient should NOT be taking these medications
            if patient_taking:
                return EvaluationResult(
                    status="violated",
                    patient_value=str(patient_taking),
                    explanation=f"Patient is taking excluded medication(s): {patient_taking}"
                )
            else:
                return EvaluationResult(
                    status="satisfied",
                    patient_value="not taking excluded meds",
                    explanation="Patient is not taking excluded medications"
                )
        else:
            # Patient SHOULD be taking (or not taking) these medications
            if patient_taking:
                return EvaluationResult(
                    status="satisfied",
                    patient_value=str(patient_taking),
                    explanation=f"Patient is taking required medication(s)"
                )
            else:
                if not patient_meds:
                    return EvaluationResult(
                        status="unknown",
                        explanation="Patient medications not provided",
                        missing_attribute="current_medications"
                    )
                return EvaluationResult(
                    status="unknown",
                    patient_value="medication not found",
                    explanation=f"Cannot confirm if patient takes: {mentioned_drugs}"
                )

    def _evaluate_treatment_history(
        self,
        criterion_text: str,
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate prior treatment criteria."""
        text_lower = criterion_text.lower()
        is_negated = self._is_negated(criterion_text)

        # Get patient's prior treatments
        prior_treatments = patient.get("prior_treatments", [])
        if isinstance(prior_treatments, str):
            prior_treatments = [prior_treatments]
        treatments_lower = [t.lower() for t in prior_treatments]

        # Common treatment types to check
        treatment_keywords = [
            "chemotherapy", "radiation", "surgery", "immunotherapy",
            "targeted therapy", "hormone therapy", "stem cell", "transplant",
            "resection", "biopsy"
        ]

        mentioned_treatments = []
        for keyword in treatment_keywords:
            if keyword in text_lower:
                mentioned_treatments.append(keyword)

        if value:
            mentioned_treatments.append(value.lower())

        if not mentioned_treatments:
            return EvaluationResult(
                status="unknown",
                explanation="Could not identify specific treatment in criterion"
            )

        # Check if patient has had these treatments
        patient_had = []
        for treatment in mentioned_treatments:
            for prior in treatments_lower:
                if treatment in prior or prior in treatment:
                    patient_had.append(prior)

        if is_negated:
            # Patient should NOT have had this treatment
            if patient_had:
                return EvaluationResult(
                    status="violated",
                    patient_value=str(patient_had),
                    explanation=f"Patient had excluded treatment: {patient_had}"
                )
            else:
                return EvaluationResult(
                    status="satisfied",
                    patient_value="no prior excluded treatments",
                    explanation="Patient has not had excluded treatments"
                )
        else:
            # Patient SHOULD have had this treatment (for some trials)
            if patient_had:
                return EvaluationResult(
                    status="satisfied",
                    patient_value=str(patient_had),
                    explanation=f"Patient has required prior treatment"
                )
            else:
                if not prior_treatments:
                    return EvaluationResult(
                        status="unknown",
                        explanation="Patient prior treatments not provided",
                        missing_attribute="prior_treatments"
                    )
                return EvaluationResult(
                    status="unknown",
                    patient_value="treatment not found",
                    explanation=f"Cannot confirm prior treatment: {mentioned_treatments}"
                )

    def _evaluate_diagnosis(
        self,
        criterion_text: str,
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate diagnosis-related criteria."""
        text_lower = criterion_text.lower()

        primary_condition = patient.get("primary_condition")
        condition_stage = patient.get("condition_stage")

        if not primary_condition:
            return EvaluationResult(
                status="unknown",
                explanation="Patient primary condition not provided",
                missing_attribute="primary_condition"
            )

        primary_lower = primary_condition.lower()

        # Check if criterion matches patient's condition
        condition_match = False

        # Check against dictionary
        for condition_key, synonyms in self.dictionary.conditions.items():
            # Check if criterion mentions this condition
            if any(syn in text_lower for syn in synonyms):
                # Check if patient has this condition
                if any(syn in primary_lower for syn in synonyms):
                    condition_match = True
                    break

        # Also do direct text matching
        if not condition_match:
            # Simple word overlap
            criterion_words = set(text_lower.split())
            patient_words = set(primary_lower.split())
            if criterion_words & patient_words:
                condition_match = True

        # Check for staging
        if "stage" in text_lower:
            stage_match = re.search(r'stage\s*([0-4]|[ivIV]+|[a-dA-D])', text_lower)
            if stage_match and condition_stage:
                required_stage = stage_match.group(1).upper()
                patient_stage = condition_stage.upper()
                if required_stage in patient_stage or patient_stage in required_stage:
                    return EvaluationResult(
                        status="satisfied",
                        patient_value=f"{primary_condition}, {condition_stage}",
                        explanation="Patient condition and stage match criterion"
                    )

        if condition_match:
            return EvaluationResult(
                status="satisfied",
                patient_value=primary_condition,
                explanation="Patient condition matches criterion"
            )

        return EvaluationResult(
            status="unknown",
            patient_value=primary_condition,
            explanation="Could not confirm condition match"
        )

    def _evaluate_organ_function(
        self,
        criterion_text: str,
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate organ function criteria (typically lab-based)."""
        text_lower = criterion_text.lower()

        # Map organ function to relevant labs
        organ_lab_map = {
            "liver": ["alt", "ast", "bilirubin"],
            "hepatic": ["alt", "ast", "bilirubin"],
            "kidney": ["creatinine", "egfr"],
            "renal": ["creatinine", "egfr"],
            "cardiac": ["troponin", "bnp"],
            "bone marrow": ["wbc", "platelet", "hemoglobin"],
            "hematologic": ["wbc", "platelet", "hemoglobin"],
        }

        # Find relevant organ
        relevant_labs = []
        for organ, labs in organ_lab_map.items():
            if organ in text_lower:
                relevant_labs.extend(labs)

        if not relevant_labs:
            return EvaluationResult(
                status="unknown",
                explanation="Could not identify organ function labs needed"
            )

        # Check if we have these lab values
        patient_labs = patient.get("lab_values", {})
        available_labs = {}
        for lab in relevant_labs:
            if lab in patient_labs:
                available_labs[lab] = patient_labs[lab]

        if not available_labs:
            return EvaluationResult(
                status="unknown",
                explanation=f"Needed lab values not provided: {relevant_labs}",
                missing_attribute="lab_values"
            )

        # For "adequate function" criteria, we'd need specific thresholds
        # This is a simplified version
        return EvaluationResult(
            status="satisfied",
            patient_value=str(available_labs),
            explanation=f"Organ function labs available: {available_labs}"
        )

    # =========================================================================
    # STEP 6: TEMPORAL CONSTRAINT HANDLING
    # Based on PMC6993990: "chronological difference between narratives"
    # =========================================================================

    def _evaluate_temporal(
        self,
        criterion_text: str,
        attribute: Optional[str],
        operator: Optional[str],
        value: Optional[str],
        patient: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Evaluate time-based criteria.
        Examples: "within 6 months", "at least 2 weeks since", etc.
        """
        text_lower = criterion_text.lower()

        # Extract time duration
        duration_days = self._extract_duration(text_lower)

        if duration_days is None:
            return EvaluationResult(
                status="unknown",
                explanation="Could not parse temporal constraint"
            )

        # Check for relevant dates in patient profile
        diagnosis_date = patient.get("diagnosis_date")

        # Try to identify what event this temporal constraint applies to
        if "diagnosis" in text_lower or "diagnosed" in text_lower:
            if not diagnosis_date:
                return EvaluationResult(
                    status="unknown",
                    explanation="Diagnosis date required for temporal criterion",
                    missing_attribute="diagnosis_date"
                )

            try:
                # Parse date (assuming ISO format or common formats)
                diag_date = self._parse_date(diagnosis_date)
                if diag_date:
                    days_since = (datetime.now() - diag_date).days

                    if "within" in text_lower or "less than" in text_lower:
                        met = days_since <= duration_days
                    elif "at least" in text_lower or "more than" in text_lower:
                        met = days_since >= duration_days
                    else:
                        met = days_since <= duration_days  # Default to within

                    return EvaluationResult(
                        status="satisfied" if met else "violated",
                        patient_value=f"{days_since} days since diagnosis",
                        explanation=f"Diagnosis was {days_since} days ago, criterion requires {'<=' if 'within' in text_lower else '>='} {duration_days} days"
                    )
            except:
                pass

        # For treatment-related temporal constraints
        if "treatment" in text_lower or "therapy" in text_lower:
            # Would need treatment dates from patient profile
            return EvaluationResult(
                status="unknown",
                explanation="Treatment dates needed for temporal evaluation",
                missing_attribute="treatment_dates"
            )

        return EvaluationResult(
            status="unknown",
            explanation="Could not evaluate temporal criterion without specific dates"
        )

    def _extract_duration(self, text: str) -> Optional[int]:
        """Extract duration in days from temporal text."""
        text_lower = text.lower()

        for pattern in self.patterns.temporal_patterns:
            match = re.search(pattern, text_lower)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    try:
                        number = int(groups[0])
                        unit = groups[1].lower()

                        # Convert to days
                        if "day" in unit:
                            return number
                        elif "week" in unit:
                            return number * 7
                        elif "month" in unit:
                            return number * 30
                        elif "year" in unit:
                            return number * 365
                    except ValueError:
                        continue

        return None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats."""
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%B %d, %Y",
            "%d %B %Y",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None


# =============================================================================
# STEP 7: AGGREGATION LOGIC FOR OVERALL ELIGIBILITY
# Based on TrialGPT: Criterion-level → Trial-level aggregation
# =============================================================================

@dataclass
class TrialEligibilityResult:
    """Aggregated result for a patient-trial match."""
    status: str  # "eligible", "ineligible", "uncertain"
    satisfied_criteria: List[Dict[str, Any]] = field(default_factory=list)
    violated_criteria: List[Dict[str, Any]] = field(default_factory=list)
    unknown_criteria: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    explanation: str = ""
    missing_information: List[str] = field(default_factory=list)


class RuleBasedMatcher:
    """
    Main entry point for rule-based clinical trial matching.

    Aggregates criterion-level evaluations into trial-level decisions.
    Based on TrialGPT's hierarchical design.
    """

    def __init__(self):
        self.evaluator = RuleBasedEvaluator()

    def match_patient_to_trial(
        self,
        patient_profile: Dict[str, Any],
        trial_criteria: List[Dict[str, Any]]
    ) -> TrialEligibilityResult:
        """
        Match a patient against all criteria for a trial.

        Args:
            patient_profile: Patient data dictionary
            trial_criteria: List of criteria dicts with:
                - criterion_id: str
                - original_text: str
                - criterion_type: "inclusion" or "exclusion"
                - attribute: Optional[str]
                - operator: Optional[str]
                - value: Optional[str]

        Returns:
            TrialEligibilityResult with aggregated status
        """
        satisfied = []
        violated = []
        unknown = []
        missing_info = []

        for criterion in trial_criteria:
            result = self.evaluator.evaluate(
                criterion_text=criterion.get("original_text", ""),
                criterion_type=criterion.get("criterion_type", "inclusion"),
                attribute=criterion.get("attribute"),
                operator=criterion.get("operator"),
                value=criterion.get("value"),
                patient_profile=patient_profile
            )

            # Build criterion result
            crit_result = {
                "criterion_id": criterion.get("criterion_id"),
                "original_text": criterion.get("original_text"),
                "criterion_type": criterion.get("criterion_type"),
                "status": result.status,
                "patient_value": result.patient_value,
                "explanation": result.explanation
            }

            if result.status == "satisfied":
                satisfied.append(crit_result)
            elif result.status == "violated":
                violated.append(crit_result)
            else:
                unknown.append(crit_result)
                if result.missing_attribute:
                    missing_info.append(result.missing_attribute)

        # Determine overall eligibility
        status = self._determine_eligibility(satisfied, violated, unknown, trial_criteria)

        # Calculate confidence
        total = len(trial_criteria)
        known = len(satisfied) + len(violated)
        confidence = known / total if total > 0 else 0.0

        # Generate explanation
        explanation = self._generate_explanation(status, satisfied, violated, unknown)

        return TrialEligibilityResult(
            status=status,
            satisfied_criteria=satisfied,
            violated_criteria=violated,
            unknown_criteria=unknown,
            confidence_score=confidence,
            explanation=explanation,
            missing_information=list(set(missing_info))
        )

    def _determine_eligibility(
        self,
        satisfied: List[Dict],
        violated: List[Dict],
        unknown: List[Dict],
        all_criteria: List[Dict]
    ) -> str:
        """
        Determine overall eligibility from criterion results.

        Logic:
        - If ANY inclusion criterion is VIOLATED → INELIGIBLE
        - If ANY exclusion criterion is VIOLATED → INELIGIBLE
        - If ANY criterion is UNKNOWN → UNCERTAIN
        - Otherwise → ELIGIBLE
        """
        # Check for violated inclusion criteria
        for crit in violated:
            if crit.get("criterion_type") == "inclusion":
                return "ineligible"

        # Check for violated exclusion criteria
        # Note: For exclusion, VIOLATED means patient HAS the exclusionary condition
        for crit in violated:
            if crit.get("criterion_type") == "exclusion":
                return "ineligible"

        # If any unknown, status is uncertain
        if unknown:
            return "uncertain"

        # All criteria satisfied
        return "eligible"

    def _generate_explanation(
        self,
        status: str,
        satisfied: List[Dict],
        violated: List[Dict],
        unknown: List[Dict]
    ) -> str:
        """Generate human-readable explanation."""
        if status == "eligible":
            return f"Patient meets all {len(satisfied)} eligibility criteria for this trial."

        if status == "ineligible":
            reasons = [c.get("explanation", c.get("original_text", ""))[:100]
                       for c in violated[:3]]
            return f"Patient does not meet eligibility: {'; '.join(reasons)}"

        if status == "uncertain":
            missing = [c.get("original_text", "")[:50] for c in unknown[:3]]
            return f"Need more information to determine eligibility. Unclear: {'; '.join(missing)}"

        return "Eligibility could not be determined."


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_matcher() -> RuleBasedMatcher:
    """Factory function to create a rule-based matcher instance."""
    return RuleBasedMatcher()


def evaluate_single_criterion(
    criterion_text: str,
    criterion_type: str,
    patient_profile: Dict[str, Any],
    attribute: Optional[str] = None,
    operator: Optional[str] = None,
    value: Optional[str] = None
) -> EvaluationResult:
    """
    Convenience function to evaluate a single criterion.

    Example:
        result = evaluate_single_criterion(
            criterion_text="Age >= 18 years",
            criterion_type="inclusion",
            patient_profile={"age": 25, "biological_sex": "male"},
            attribute="age",
            operator=">=",
            value="18"
        )
    """
    evaluator = RuleBasedEvaluator()
    return evaluator.evaluate(
        criterion_text=criterion_text,
        criterion_type=criterion_type,
        attribute=attribute,
        operator=operator,
        value=value,
        patient_profile=patient_profile
    )
