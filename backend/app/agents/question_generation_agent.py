from typing import Any, Dict, List, Optional
import json
from .base_agent import BaseAgent
from ..schemas.patient import PatientProfile


class QuestionGenerationAgent(BaseAgent):
    """
    Generates natural, empathetic follow-up questions based on identified
    information gaps. Implements the three-phase questioning strategy:

    Phase 1: Baseline screening (universal attributes)
    Phase 2: Trial-driven questioning (criteria-aware)
    Phase 3: Gap-filling and clarification (adaptive)
    """

    def __init__(self):
        super().__init__(
            name="Question Generation Agent",
            description="Generates targeted, criteria-driven follow-up questions for patients"
        )

        # Phase 1 baseline questions (asked first, regardless of trials)
        # Phrased warmly and conversationally
        # Priority determines the order questions are asked (lower = earlier)
        self.phase1_questions = {
            "primary_condition": {
                "question": "What medical condition are you hoping to find a trial for?",
                "context": "This is the starting point for finding relevant trials for you.",
                "priority": 0
            },
            "age": {
                "question": "May I ask how old you are? This helps match you with trials that have specific age requirements.",
                "context": "Different trials accept different age ranges.",
                "priority": 1
            },
            "biological_sex": {
                "question": "Could you share your biological sex? Some trials are designed specifically for certain groups.",
                "context": "Certain trials focus on conditions that affect specific biological sexes differently.",
                "priority": 2
            },
            "country": {
                "question": "What country are you located in? This helps me find trials you can actually access.",
                "context": "Clinical trials are conducted at specific locations.",
                "priority": 3
            },
            "state_province": {
                "question": "Which state or province are you in? This helps me find trials closer to you.",
                "context": "Finding nearby trials makes participation more convenient.",
                "priority": 4
            },
            "willing_to_travel": {
                "question": "Would you be willing to travel to participate in a clinical trial, or would you prefer trials closer to home?",
                "context": "Some excellent trials may require travel; knowing your preference helps narrow options.",
                "priority": 5
            },
            "diagnosis_date": {
                "question": "When were you first diagnosed with this condition? A rough timeframe is fine, like 'last year' or '3 months ago'.",
                "context": "Some trials are for newly diagnosed patients while others are for those with longer history.",
                "priority": 6
            },
            "current_medications": {
                "question": "Are you currently taking any medications? If so, could you list them? If not, just let me know.",
                "context": "Some trials have requirements about current medications or may interact with them.",
                "priority": 7,
                "tracks_field": "asked_medications"
            },
            "prior_treatments": {
                "question": "What treatments have you already tried for this condition, if any? This could include medications, therapies, or procedures.",
                "context": "Many trials are looking for patients who have or haven't tried specific treatments.",
                "priority": 8,
                "tracks_field": "asked_prior_treatments"
            }
        }

        # Sensitive topics that need careful phrasing
        self.sensitive_topics = {
            "pregnancy_status": "pregnancy or reproductive health",
            "smoking_status": "tobacco use",
            "alcohol_use": "alcohol consumption",
            "mental_health": "mental health conditions",
            "hiv_status": "HIV status",
            "substance_use": "substance use"
        }

    def get_system_prompt(self) -> str:
        return """You are a compassionate clinical trial assistant helping patients find suitable trials.

Your role is to generate follow-up questions that:
1. Are natural and conversational, not clinical or robotic
2. Are empathetic and non-judgmental
3. Explain why the information is needed (transparency)
4. Avoid medical jargon when possible
5. Never ask about multiple unrelated topics in one question
6. Respect patient privacy and comfort

For sensitive topics (pregnancy, substance use, mental health), be especially gentle and explain that:
- The information is optional
- It's only used to find appropriate trials
- Their privacy is protected

Always prioritize the patient's comfort while gathering necessary information."""

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate follow-up questions based on current phase and information gaps.

        Input:
            patient_profile: PatientProfile - Current patient profile
            phase: int - Current conversation phase (1, 2, or 3)
            gaps: List[dict] - Information gaps from Gap Analysis Agent (for phases 2-3)
            trial_context: Optional context about specific trials
            asked_medications: bool - Whether medications question has been asked
            asked_prior_treatments: bool - Whether prior treatments question has been asked

        Output:
            questions: List of question objects with text and metadata
            phase_explanation: Why these questions are being asked
            suggested_response: A natural response incorporating questions
            next_tracks_field: Optional field to mark as asked after this response
        """
        profile: PatientProfile = input_data.get("patient_profile", PatientProfile())
        phase: int = input_data.get("phase", 1)
        gaps: List[dict] = input_data.get("gaps", [])
        trial_context: Optional[str] = input_data.get("trial_context")
        asked_medications: bool = input_data.get("asked_medications", False)
        asked_prior_treatments: bool = input_data.get("asked_prior_treatments", False)

        if phase == 1:
            return await self._generate_phase1_questions(profile, asked_medications, asked_prior_treatments)
        elif phase == 2:
            return await self._generate_phase2_questions(profile, gaps, trial_context)
        else:  # Phase 3
            return await self._generate_phase3_questions(profile, gaps, trial_context)

    async def _generate_phase1_questions(
        self,
        profile: PatientProfile,
        asked_medications: bool = False,
        asked_prior_treatments: bool = False
    ) -> Dict[str, Any]:
        """Generate baseline screening questions (Phase 1)."""
        profile_dict = profile.model_dump()
        missing = []

        # Find missing Phase 1 attributes
        for attr, q_info in self.phase1_questions.items():
            value = profile_dict.get(attr)

            # Special handling for medications and prior_treatments (tracked separately)
            if attr == "current_medications":
                if not asked_medications:
                    missing.append({
                        "attribute": attr,
                        "question": q_info["question"],
                        "context": q_info["context"],
                        "priority": q_info["priority"],
                        "phase": 1,
                        "tracks_field": "asked_medications"
                    })
                continue

            if attr == "prior_treatments":
                if not asked_prior_treatments:
                    missing.append({
                        "attribute": attr,
                        "question": q_info["question"],
                        "context": q_info["context"],
                        "priority": q_info["priority"],
                        "phase": 1,
                        "tracks_field": "asked_prior_treatments"
                    })
                continue

            # For all other fields, check if value is None or empty
            if value is None or value == "":
                missing.append({
                    "attribute": attr,
                    "question": q_info["question"],
                    "context": q_info["context"],
                    "priority": q_info["priority"],
                    "phase": 1
                })

        # Sort by priority
        missing.sort(key=lambda x: x["priority"])

        # Take top 1 question at a time for natural conversation flow
        questions = missing[:1]

        # Generate a natural response incorporating the questions
        if questions:
            suggested_response = await self._generate_natural_response(
                questions,
                phase=1,
                profile=profile
            )
        else:
            suggested_response = None

        # Track which field this question will mark as asked
        next_tracks_field = questions[0].get("tracks_field") if questions else None

        return {
            "questions": questions,
            "phase": 1,
            "phase_explanation": "Phase 1: Collecting baseline information to begin searching for relevant trials.",
            "suggested_response": suggested_response,
            "all_baseline_collected": len(missing) == 0,
            "next_tracks_field": next_tracks_field
        }

    async def _generate_phase2_questions(
        self,
        profile: PatientProfile,
        gaps: List[dict],
        trial_context: Optional[str]
    ) -> Dict[str, Any]:
        """Generate trial-driven questions (Phase 2)."""

        if not gaps:
            return {
                "questions": [],
                "phase": 2,
                "phase_explanation": "No additional information needed from trials.",
                "suggested_response": None
            }

        # Convert gaps to questions using LLM
        questions = await self._gaps_to_questions(gaps, profile, trial_context)

        # Generate natural response
        if questions:
            suggested_response = await self._generate_natural_response(
                questions[:2],  # Limit to 2 questions at a time
                phase=2,
                profile=profile,
                trial_context=trial_context
            )
        else:
            suggested_response = None

        return {
            "questions": questions[:3],  # Return up to 3 for display
            "phase": 2,
            "phase_explanation": "Phase 2: Asking trial-specific questions based on eligibility criteria from matched trials.",
            "suggested_response": suggested_response
        }

    async def _generate_phase3_questions(
        self,
        profile: PatientProfile,
        gaps: List[dict],
        trial_context: Optional[str]
    ) -> Dict[str, Any]:
        """Generate gap-filling questions (Phase 3)."""

        if not gaps:
            return {
                "questions": [],
                "phase": 3,
                "phase_explanation": "All necessary information collected.",
                "suggested_response": None
            }

        # Focus on remaining unknowns and clarifications
        questions = await self._gaps_to_questions(
            gaps,
            profile,
            trial_context,
            is_clarification=True
        )

        if questions:
            suggested_response = await self._generate_natural_response(
                questions[:2],
                phase=3,
                profile=profile,
                trial_context=trial_context
            )
        else:
            suggested_response = None

        return {
            "questions": questions[:2],
            "phase": 3,
            "phase_explanation": "Phase 3: Final clarifications to confirm eligibility and finalize your personalized trial recommendations.",
            "suggested_response": suggested_response
        }

    async def _gaps_to_questions(
        self,
        gaps: List[dict],
        profile: PatientProfile,
        trial_context: Optional[str],
        is_clarification: bool = False
    ) -> List[dict]:
        """Convert information gaps to natural questions using LLM."""

        if not gaps:
            return []

        # Check for sensitive topics
        sensitive_gaps = []
        regular_gaps = []

        for gap in gaps:
            attr = gap.get("attribute", "").lower()
            is_sensitive = any(s in attr for s in self.sensitive_topics.keys())
            if is_sensitive:
                sensitive_gaps.append(gap)
            else:
                regular_gaps.append(gap)

        # Prioritize regular gaps, then sensitive
        ordered_gaps = regular_gaps + sensitive_gaps

        prompt = f"""Convert these information gaps into natural, empathetic patient questions.

INFORMATION GAPS:
{json.dumps(ordered_gaps[:5], indent=2)}

CURRENT PATIENT INFO:
- Condition: {profile.primary_condition or 'Not yet provided'}
- Age: {profile.age or 'Not yet provided'}

{"TRIAL CONTEXT: " + trial_context if trial_context else ""}

{"These are CLARIFICATION questions - be extra gentle and explain why we're asking again." if is_clarification else ""}

For each gap, generate a question object with:
1. attribute: The attribute being asked about
2. question: A natural, conversational question
3. context: Brief explanation of why this matters (1 sentence)
4. is_sensitive: true if this is a sensitive topic
5. optional_note: If sensitive, a note that answering is optional

IMPORTANT:
- Don't be robotic or clinical
- Show empathy
- Keep questions short and clear
- For sensitive topics, add reassurance

Respond with a JSON array:
[
  {{
    "attribute": "string",
    "question": "string",
    "context": "string",
    "is_sensitive": boolean,
    "optional_note": "string or null"
  }}
]"""

        try:
            response = await self.llm.generate_json(prompt, self.get_system_prompt())
            questions = json.loads(response)
            if isinstance(questions, list):
                # Add phase info
                for q in questions:
                    q["phase"] = 3 if is_clarification else 2
                return questions
        except Exception as e:
            print(f"Question generation error: {e}")

        # Fallback: use question hints from gaps
        fallback_questions = []
        for gap in ordered_gaps[:3]:
            hint = gap.get("question_hint", gap.get("reason", ""))
            if hint:
                fallback_questions.append({
                    "attribute": gap.get("attribute", "unknown"),
                    "question": hint if "?" in hint else f"Could you tell me about your {gap.get('attribute', 'medical history')}?",
                    "context": gap.get("reason", "This helps us evaluate trial eligibility."),
                    "is_sensitive": False,
                    "phase": 3 if is_clarification else 2
                })

        return fallback_questions

    async def _generate_natural_response(
        self,
        questions: List[dict],
        phase: int,
        profile: PatientProfile,
        trial_context: Optional[str] = None
    ) -> str:
        """Generate a natural conversational response incorporating questions."""
        import random

        if not questions:
            return ""

        # Varied acknowledgments
        acks = ["Got it!", "Thanks!", "Perfect!", "Great!", "Noted!", "Appreciate that!", "Wonderful!"]
        transitions = ["Now,", "Next,", "Moving on,", "Also,", "I'd also like to know -", "One more thing -"]

        # Phase-specific natural responses
        if phase == 1:
            if not profile.primary_condition:
                return questions[0]["question"]

            # Acknowledge and ask next question with variety
            ack = random.choice(acks)
            q_text = questions[0]["question"]
            return f"{ack} {q_text}"

        elif phase == 2:
            # Trial-driven phase - explain we found matches and need specifics
            intros = [
                "To help determine which trials you might qualify for, ",
                "Some trials have specific requirements - ",
                "To narrow down the best matches, ",
            ]
            intro = random.choice(intros)
            q1 = questions[0]["question"]
            return intro + q1.lower() if q1[0].isupper() else intro + q1

        else:  # Phase 3
            # Final clarifications - be encouraging
            intros = [
                "We're almost there! ",
                "Just a bit more - ",
                "Nearly done! ",
                "One last thing - "
            ]
            intro = random.choice(intros)
            return intro + questions[0]["question"]


# Singleton instance
question_generation_agent = QuestionGenerationAgent()
