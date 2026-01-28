from typing import Any, Dict, List, Optional, Set
import random
from .base_agent import BaseAgent
from ..schemas.patient import PatientProfile


class QuestionGenerationAgent(BaseAgent):
    """
    Generates follow-up questions based on identified information gaps.
    OPTIMIZED: No LLM calls - uses rule-based question generation for speed.
    """

    def __init__(self):
        super().__init__(
            name="Question Generation Agent",
            description="Generates targeted follow-up questions for patients"
        )

        # Track asked questions per session to avoid duplicates
        self.asked_questions: Dict[str, Set[str]] = {}

        # Phase 1 baseline questions
        self.phase1_questions = {
            "primary_condition": {
                "question": "What medical condition are you hoping to find a trial for?",
                "priority": 0
            },
            "age": {
                "question": "May I ask how old you are?",
                "priority": 1
            },
            "biological_sex": {
                "question": "What is your biological sex (male/female)?",
                "priority": 2
            },
            "country": {
                "question": "What country are you located in?",
                "priority": 3
            },
            "state_province": {
                "question": "Which state or province are you in?",
                "priority": 4
            },
            "willing_to_travel": {
                "question": "Would you be willing to travel for a trial?",
                "priority": 5
            },
            "diagnosis_date": {
                "question": "When were you first diagnosed? A rough timeframe is fine.",
                "priority": 6
            },
            "current_medications": {
                "question": "Are you currently taking any medications? List them or say 'none'.",
                "priority": 7,
                "tracks_field": "asked_medications"
            },
            "prior_treatments": {
                "question": "What treatments have you already tried? Or 'none' if you haven't.",
                "priority": 8,
                "tracks_field": "asked_prior_treatments"
            }
        }

        # Pre-built Phase 2/3 questions for common criteria (NO LLM needed)
        self.criteria_questions = {
            "ecog": "What is your current activity level? Can you work and do daily activities normally, or do you need some rest during the day?",
            "ecog_status": "What is your current activity level? Can you work and do daily activities normally, or do you need some rest during the day?",
            "performance_status": "How would you describe your daily activity level? Fully active, or do you need rest periods?",
            "other_cancers": "Have you ever been diagnosed with any other type of cancer besides your current condition?",
            "cancer_history": "Have you ever been diagnosed with any other type of cancer?",
            "metastatic": "Has your cancer spread to other parts of your body (metastasized)?",
            "metastasis": "Has your cancer spread to other parts of your body?",
            "brain_metastases": "Have you been told if your cancer has spread to your brain?",
            "liver": "Do you have any liver conditions or problems?",
            "kidney": "Do you have any kidney conditions or problems?",
            "heart": "Do you have any heart conditions?",
            "cardiac": "Do you have any heart or cardiovascular conditions?",
            "diabetes": "Do you have diabetes?",
            "hiv": "This is optional - do you know your HIV status?",
            "hepatitis": "Have you ever been diagnosed with hepatitis B or C?",
            "pregnant": "Are you currently pregnant or planning to become pregnant?",
            "pregnancy": "Are you currently pregnant or planning to become pregnant?",
            "breastfeeding": "Are you currently breastfeeding?",
            "surgery": "Have you had any surgeries related to your condition? If so, when?",
            "radiation": "Have you received radiation therapy? If so, when was your last treatment?",
            "chemotherapy": "Have you received chemotherapy? If so, what type and when?",
            "immunotherapy": "Have you received any immunotherapy treatments?",
            "hormone": "Have you received any hormone therapy?",
            "biomarker": "Do you know any of your biomarker or genetic test results (like HER2, BRCA, etc.)?",
            "mutation": "Have you had any genetic testing done? Do you know of any mutations?",
            "stage": "What stage is your condition? (e.g., Stage 1, 2, 3, or 4)",
            "grade": "Do you know the grade of your cancer?",
            "comorbidities": "Do you have any other medical conditions I should know about?",
            "autoimmune": "Do you have any autoimmune conditions (like lupus, rheumatoid arthritis, etc.)?",
            "transplant": "Have you ever had an organ transplant?",
            "blood_clot": "Have you ever had blood clots or clotting disorders?",
            "bleeding": "Do you have any bleeding disorders?",
            "allergy": "Do you have any drug allergies I should know about?",
            "smoking": "Do you currently smoke or have you smoked in the past?",
            "alcohol": "How would you describe your alcohol consumption?",
        }

    def get_system_prompt(self) -> str:
        return ""  # Not used - no LLM calls

    def _get_session_asked(self, session_id: str) -> Set[str]:
        """Get set of asked question attributes for a session."""
        if session_id not in self.asked_questions:
            self.asked_questions[session_id] = set()
        return self.asked_questions[session_id]

    def _mark_asked(self, session_id: str, attribute: str):
        """Mark an attribute as asked for a session."""
        self._get_session_asked(session_id).add(attribute.lower())

    def _was_asked(self, session_id: str, attribute: str) -> bool:
        """Check if an attribute was already asked (with smart matching for similar topics)."""
        attr_lower = attribute.lower()
        asked = self._get_session_asked(session_id)

        # Direct match
        if attr_lower in asked:
            return True

        # Smart matching for similar topics
        similar_groups = [
            {"cancer", "cancers", "other_cancer", "other_cancers", "cancer_history", "malignancy", "tumor"},
            {"heart", "cardiac", "cardiovascular", "heart_disease", "heart_condition"},
            {"liver", "hepatic", "liver_disease", "liver_function"},
            {"kidney", "renal", "kidney_disease", "kidney_function"},
            {"diabetes", "diabetic", "blood_sugar", "glucose"},
            {"pregnant", "pregnancy", "pregnancy_status", "reproductive"},
            {"smoke", "smoking", "smoking_status", "tobacco", "cigarette"},
            {"alcohol", "alcohol_use", "drinking"},
            {"metastatic", "metastasis", "metastases", "spread"},
            {"surgery", "surgical", "operation"},
            {"chemo", "chemotherapy", "chemo_therapy"},
            {"radiation", "radiotherapy", "radiation_therapy"},
            {"immunotherapy", "immuno", "immune_therapy"},
            {"hormone", "hormonal", "hormone_therapy"},
            {"biomarker", "marker", "genetic", "mutation", "gene"},
            {"ecog", "performance", "performance_status", "activity_level"},
            {"stage", "staging", "cancer_stage"},
            {"comorbidity", "comorbidities", "other_conditions", "medical_conditions"},
            {"allergy", "allergies", "allergic"},
            {"transplant", "organ_transplant"},
            {"autoimmune", "auto_immune", "autoimmune_disease"},
        ]

        for group in similar_groups:
            # Check if attribute belongs to this group
            if any(term in attr_lower or attr_lower in term for term in group):
                # Check if any term from this group was already asked
                for asked_topic in asked:
                    if any(term in asked_topic or asked_topic in term for term in group):
                        return True

        return False

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate follow-up questions - NO LLM CALLS for speed."""
        profile: PatientProfile = input_data.get("patient_profile", PatientProfile())
        phase: int = input_data.get("phase", 1)
        gaps: List[dict] = input_data.get("gaps", [])
        asked_medications: bool = input_data.get("asked_medications", False)
        asked_prior_treatments: bool = input_data.get("asked_prior_treatments", False)
        session_id: str = input_data.get("session_id", "default")
        answered_topics: List[str] = input_data.get("answered_topics", [])
        phase1_asked: set = input_data.get("phase1_asked", set())
        phase2_asked: set = input_data.get("phase2_asked", set())
        phase2_questions_count: int = input_data.get("phase2_questions_count", 0)

        # Add answered_topics to the session tracking
        for topic in answered_topics:
            self._mark_asked(session_id, topic)

        if phase == 1:
            return self._generate_phase1_questions(
                profile, asked_medications, asked_prior_treatments,
                session_id, phase1_asked
            )
        else:
            return self._generate_phase2_3_questions(
                profile, gaps, phase, session_id,
                phase2_asked, phase2_questions_count
            )

    def _generate_phase1_questions(
        self,
        profile: PatientProfile,
        asked_medications: bool,
        asked_prior_treatments: bool,
        session_id: str = "default",
        phase1_asked: set = None
    ) -> Dict[str, Any]:
        """Generate baseline questions (Phase 1) - NO LLM.

        IMPORTANT: Once a question has been asked, we do NOT repeat it.
        If the user's answer was invalid, the validation error handling
        will prompt them to correct it without re-asking the full question.
        """
        profile_dict = profile.model_dump()
        phase1_asked = phase1_asked or set()
        missing = []

        for attr, q_info in self.phase1_questions.items():
            # CRITICAL: Skip if this question was already asked - NEVER repeat
            if attr in phase1_asked:
                continue

            value = profile_dict.get(attr)

            # Special handling for tracked fields
            if attr == "current_medications" and asked_medications:
                continue
            if attr == "prior_treatments" and asked_prior_treatments:
                continue

            if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
                missing.append({
                    "attribute": attr,
                    "question": q_info["question"],
                    "priority": q_info["priority"],
                    "phase": 1,
                    "tracks_field": q_info.get("tracks_field")
                })

        missing.sort(key=lambda x: x["priority"])
        questions = missing[:1]

        if questions:
            q = questions[0]
            suggested_response = q["question"]
        else:
            suggested_response = None

        return {
            "questions": questions,
            "phase": 1,
            "phase_explanation": "Collecting baseline information.",
            "suggested_response": suggested_response,
            "all_baseline_collected": len(missing) == 0,
            "next_tracks_field": questions[0].get("tracks_field") if questions else None
        }

    def _generate_phase2_3_questions(
        self,
        profile: PatientProfile,
        gaps: List[dict],
        phase: int,
        session_id: str,
        phase2_asked: set = None,
        phase2_questions_count: int = 0
    ) -> Dict[str, Any]:
        """Generate trial-driven questions (Phase 2/3) - NO LLM, uses pre-built questions.

        IMPORTANT: Once a Phase 2 question has been asked, we do NOT repeat it.
        If the user's answer was invalid, validation error handling will
        prompt them to correct it without re-asking the full question.

        Phase 2 is capped at 5 questions total, then we show trials.
        """
        phase2_asked = phase2_asked or set()

        # If we've asked 5 questions OR no gaps, we're done
        if phase2_questions_count >= 5 or not gaps:
            return {
                "questions": [],
                "phase": phase,
                "phase_explanation": "Ready to show trial results.",
                "suggested_response": None
            }

        questions = []
        profile_dict = profile.model_dump()

        for gap in gaps:
            attr = gap.get("attribute", "").lower().strip()

            # CRITICAL: Skip if this Phase 2 question was already asked - NEVER repeat
            if attr in phase2_asked:
                continue

            # Skip if already asked this session (backup check)
            if self._was_asked(session_id, attr):
                continue

            # Skip if we already have this info in profile
            if attr in profile_dict and profile_dict[attr]:
                continue

            # Find matching pre-built question
            question_text = None
            for key, q_text in self.criteria_questions.items():
                if key in attr or attr in key:
                    question_text = q_text
                    break

            # Fallback: generate simple question from attribute name
            if not question_text:
                readable_attr = attr.replace("_", " ").replace("-", " ")
                question_text = f"Could you tell me about your {readable_attr}?"

            questions.append({
                "attribute": attr,
                "question": question_text,
                "phase": phase
            })

            # Mark as asked
            self._mark_asked(session_id, attr)

            # Only ask 1 question at a time
            if len(questions) >= 1:
                break

        if questions:
            # Add variety to the response
            acks = ["Got it!", "Thanks!", "Perfect!", "Great!", "Noted!"]
            intros_p2 = [
                "To check your eligibility for these trials - ",
                "Some trials have specific requirements - ",
                "To narrow down the best matches - ",
            ]
            intros_p3 = [
                "Almost done! ",
                "Just one more thing - ",
                "Nearly there! ",
            ]

            ack = random.choice(acks)
            if phase == 2:
                intro = random.choice(intros_p2)
            else:
                intro = random.choice(intros_p3)

            suggested_response = f"{ack} {intro}{questions[0]['question'].lower()}"
        else:
            suggested_response = None

        return {
            "questions": questions,
            "phase": phase,
            "phase_explanation": f"Phase {phase}: {'Trial-driven' if phase == 2 else 'Final'} questions.",
            "suggested_response": suggested_response
        }

    def clear_session(self, session_id: str):
        """Clear asked questions for a session."""
        if session_id in self.asked_questions:
            del self.asked_questions[session_id]


# Singleton instance
question_generation_agent = QuestionGenerationAgent()
