import json
import re
from typing import Any, Dict, List, Tuple
from .base_agent import BaseAgent
from ..schemas.patient import PatientProfile


class PatientProfilingAgent(BaseAgent):
    """
    Agent responsible for extracting structured patient information
    from conversational free-text input. Includes validation to catch
    incorrect inputs (e.g., "18" for gender, "male" for age).
    """

    def __init__(self):
        super().__init__(
            name="Patient Profiling Agent",
            description="Extract structured patient attributes from natural language conversation."
        )

        # Valid values for validation
        self.valid_sexes = {"male", "female", "other", "m", "f"}
        self.valid_yes_no = {"yes", "no", "true", "false", "yeah", "yep", "nope", "sure", "okay", "ok"}

        # Common country names (partial list for validation)
        self.common_countries = {
            "usa", "us", "united states", "america", "canada", "uk", "united kingdom",
            "australia", "germany", "france", "india", "china", "japan", "brazil",
            "mexico", "spain", "italy", "netherlands", "sweden", "norway", "denmark"
        }

    def get_system_prompt(self) -> str:
        return """You are a medical information extraction specialist. Your job is to extract patient attributes from conversational text.

Extract ONLY information that is explicitly stated. Do not infer or assume.

Return a JSON object with these fields (use null for unknown, use empty list [] for "none"):
{
    "age": <integer or null>,
    "biological_sex": <"male", "female", "other", or null>,
    "primary_condition": <string or null>,
    "condition_stage": <string or null>,
    "diagnosis_date": <string or null - accepts relative terms like "last year", "2 months ago", "recently", "2023", etc.>,
    "country": <string or null>,
    "state_province": <string or null>,
    "city": <string or null>,
    "willing_to_travel": <true, false, or null - true if they say yes/willing, false if they prefer close/local/no travel>,
    "comorbidities": [<list of strings, or empty [] if they say "none">],
    "current_medications": [<list of strings, or empty [] if they say "none" or "not taking any">],
    "prior_treatments": [<list of strings, or empty [] if they say "none" or "haven't tried any">],
    "allergies": [<list of strings>],
    "smoking_status": <string or null>,
    "additional_attributes": {<any other relevant medical info>}
}

IMPORTANT extraction rules:
- For diagnosis_date: Accept any time reference like "last month", "2 years ago", "recently", "January 2024", etc.
- For willing_to_travel: "yes", "willing", "open to it" = true; "no", "prefer local", "close to home" = false
- For medications/treatments: If they say "none", "nothing", "not taking any", return empty list []
- Be flexible with informal language - extract the intent even if not perfectly phrased"""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract patient attributes from user message.

        Input:
            - message: The user's message text
            - current_profile: Existing patient profile to update

        Output:
            - extracted_attributes: New attributes found in this message
            - updated_profile: Merged profile with new attributes
            - confidence: Confidence in extractions
            - validation_errors: List of validation error messages (for re-prompting)
        """
        message = input_data.get("message", "")
        current_profile = input_data.get("current_profile", PatientProfile())

        prompt = f"""Extract patient information from this message:

"{message}"

Current known profile:
{current_profile.model_dump_json(indent=2)}

Return JSON with any NEW information found in this message."""

        try:
            response = await self.llm.generate_json(prompt, self.get_system_prompt())

            # Parse the JSON response
            extracted = json.loads(response)

            # Validate and fix the extracted data
            validated, validation_errors = self._validate_and_fix_extraction(extracted, message)

            # Merge validated data with existing profile
            updated_profile = self._merge_profiles(current_profile, validated)

            return {
                "extracted_attributes": validated,
                "updated_profile": updated_profile,
                "confidence": self._calculate_confidence(validated),
                "validation_errors": validation_errors
            }

        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            return {
                "extracted_attributes": {},
                "updated_profile": current_profile,
                "confidence": 0.0,
                "validation_errors": []
            }

    def _merge_profiles(self, current: PatientProfile, new_data: dict) -> PatientProfile:
        """Merge new extracted data into existing profile."""
        profile_dict = current.model_dump()

        for key, value in new_data.items():
            if value is not None and key in profile_dict:
                if isinstance(value, list) and isinstance(profile_dict[key], list):
                    # Extend lists without duplicates
                    existing = set(profile_dict[key])
                    profile_dict[key] = list(existing.union(set(value)))
                elif isinstance(value, dict) and isinstance(profile_dict[key], dict):
                    # Merge dicts
                    profile_dict[key].update(value)
                else:
                    # Overwrite scalar values
                    profile_dict[key] = value

        return PatientProfile(**profile_dict)

    def _calculate_confidence(self, extracted: dict) -> float:
        """Calculate confidence based on how many fields were extracted."""
        key_fields = ["age", "biological_sex", "primary_condition", "country"]
        extracted_key_fields = sum(1 for f in key_fields if extracted.get(f) is not None)
        return extracted_key_fields / len(key_fields)

    def _validate_and_fix_extraction(self, extracted: dict, original_message: str) -> Tuple[dict, List[str]]:
        """
        Validate extracted data and fix common mistakes.
        Returns (validated_data, list_of_validation_errors)

        IMPORTANT: Error messages are BRIEF - just state what's needed.
        Do NOT repeat the full question - the user already answered.
        """
        errors = []
        validated = {}

        for key, value in extracted.items():
            if value is None:
                continue

            # Validate AGE
            if key == "age":
                if isinstance(value, str):
                    # Try to extract number from string
                    numbers = re.findall(r'\d+', str(value))
                    if numbers:
                        value = int(numbers[0])
                    else:
                        # Check if it's actually a sex value
                        if str(value).lower() in self.valid_sexes:
                            errors.append("age:number")  # Brief type indicator
                            continue
                        else:
                            errors.append("age:number")
                            continue

                if isinstance(value, (int, float)):
                    value = int(value)
                    if value < 0 or value > 120:
                        errors.append("age:valid_range")
                        continue
                    validated[key] = value
                else:
                    errors.append("age:number")

            # Validate BIOLOGICAL SEX
            elif key == "biological_sex":
                value_lower = str(value).lower().strip()

                # Check if it's actually a number (meant for age)
                if value_lower.isdigit():
                    errors.append("sex:male_or_female")
                    continue

                # Normalize sex values
                if value_lower in {"m", "male", "man", "boy"}:
                    validated[key] = "male"
                elif value_lower in {"f", "female", "woman", "girl"}:
                    validated[key] = "female"
                elif value_lower in {"other", "non-binary", "nonbinary", "nb"}:
                    validated[key] = "other"
                else:
                    errors.append("sex:male_or_female")

            # Validate COUNTRY
            elif key == "country":
                value_str = str(value).strip()

                # Check if it's a number (wrong field)
                if value_str.isdigit():
                    errors.append("country:name")
                    continue

                # Check if it looks like a sex value
                if value_str.lower() in self.valid_sexes:
                    errors.append("country:name")
                    continue

                validated[key] = value_str

            # Validate STATE/PROVINCE
            elif key == "state_province":
                value_str = str(value).strip()

                if value_str.isdigit():
                    errors.append("state:name")
                    continue

                validated[key] = value_str

            # Validate WILLING TO TRAVEL (boolean)
            elif key == "willing_to_travel":
                if isinstance(value, bool):
                    validated[key] = value
                elif isinstance(value, str):
                    value_lower = value.lower().strip()
                    if value_lower in {"yes", "true", "yeah", "yep", "sure", "okay", "ok", "willing", "open"}:
                        validated[key] = True
                    elif value_lower in {"no", "false", "nope", "not", "prefer not", "local", "close"}:
                        validated[key] = False
                    else:
                        errors.append("travel:yes_or_no")

            # Validate DIAGNOSIS DATE
            elif key == "diagnosis_date":
                value_str = str(value).strip()

                # Check if it's just a number that might be age
                if value_str.isdigit() and int(value_str) < 150:
                    # Could be a year or could be age - accept years
                    if int(value_str) > 1900 and int(value_str) <= 2026:
                        validated[key] = value_str
                    else:
                        errors.append("diagnosis:date_or_timeframe")
                        continue
                else:
                    validated[key] = value_str

            # Validate MEDICATIONS (should be a list)
            elif key == "current_medications":
                if isinstance(value, list):
                    validated[key] = value
                elif isinstance(value, str):
                    if value.lower() in {"none", "nothing", "no", "n/a", "na", "nil"}:
                        validated[key] = []
                    else:
                        # Split by common delimiters
                        meds = re.split(r'[,;]|\s+and\s+', value)
                        validated[key] = [m.strip() for m in meds if m.strip()]

            # Validate PRIOR TREATMENTS (should be a list)
            elif key == "prior_treatments":
                if isinstance(value, list):
                    validated[key] = value
                elif isinstance(value, str):
                    if value.lower() in {"none", "nothing", "no", "n/a", "na", "nil"}:
                        validated[key] = []
                    else:
                        treatments = re.split(r'[,;]|\s+and\s+', value)
                        validated[key] = [t.strip() for t in treatments if t.strip()]

            # Pass through other fields
            else:
                validated[key] = value

        return validated, errors


# Singleton instance
patient_profiling_agent = PatientProfilingAgent()
