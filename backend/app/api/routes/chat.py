from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional
import uuid
import random
from datetime import datetime

from ...schemas.chat import ChatRequest, ChatResponse, ChatMessage
from ...schemas.patient import PatientProfile, ConversationState
from ...schemas.trial import TrialMatch, EligibilityStatus
from ...agents.patient_profiling_agent import patient_profiling_agent
from ...agents.trial_discovery_agent import trial_discovery_agent
from ...agents.criteria_extraction_agent import criteria_extraction_agent
from ...agents.eligibility_matching_agent import eligibility_matching_agent
from ...agents.gap_analysis_agent import gap_analysis_agent
from ...agents.question_generation_agent import question_generation_agent
from ...services.llm_service import llm_service

router = APIRouter()

# In-memory session storage (use Redis in production)
sessions: Dict[str, ConversationState] = {}

# Cache for trial matches to avoid re-querying
trial_cache: Dict[str, List[TrialMatch]] = {}


def _is_first_message(state: ConversationState) -> bool:
    """Check if this is the first message in the conversation."""
    # Only the current user message exists (added before this check)
    return len(state.messages) <= 1


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return response with potential trial matches.

    Implements three-phase questioning strategy:
    - Phase 1: Baseline screening (age, condition, location)
    - Phase 2: Trial-driven questioning (based on specific trial criteria)
    - Phase 3: Gap-filling and clarification (resolve uncertain eligibility)
    """
    # Get or create session
    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in sessions:
        sessions[session_id] = ConversationState(
            session_id=session_id,
            patient_profile=PatientProfile(),
            phase=1,
            messages=[]
        )

    state = sessions[session_id]

    # Add user message to history
    state.messages.append({
        "role": "user",
        "content": request.message,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Step 1: Extract patient information from message
    profile_result = await patient_profiling_agent.process({
        "message": request.message,
        "current_profile": state.patient_profile
    })

    state.patient_profile = profile_result["updated_profile"]
    profile_updated = bool(profile_result["extracted_attributes"])
    validation_errors = profile_result.get("validation_errors", [])

    # If there are validation errors, we need to re-prompt the user
    # Don't proceed to trial search until input is valid

    # Step 2: Determine current phase and get trial matches if applicable
    trial_matches = []
    gaps = []
    phase_changed = False

    if state.phase == 1:
        # Phase 1: Check if we have minimum info to search for trials
        if _has_minimum_profile(state.patient_profile, state):
            trial_matches = await _find_matching_trials(state.patient_profile)

            if trial_matches:
                # Cache trial matches for this session
                trial_cache[session_id] = trial_matches

                # Transition to Phase 2
                state.phase = 2
                phase_changed = True

                # Analyze gaps for Phase 2 questioning
                gaps = await _analyze_gaps(trial_matches, state.patient_profile)

    elif state.phase == 2:
        # Phase 2: Trial-driven questioning
        # Re-evaluate trials with updated profile
        if profile_updated:
            trial_matches = await _find_matching_trials(state.patient_profile)
            trial_cache[session_id] = trial_matches
        else:
            trial_matches = trial_cache.get(session_id, [])

        # Analyze remaining gaps
        gaps = await _analyze_gaps(trial_matches, state.patient_profile)

        # Check if we should move to Phase 3
        if _should_transition_to_phase3(trial_matches, gaps):
            state.phase = 3
            phase_changed = True

    else:  # Phase 3
        # Phase 3: Gap-filling and final clarification
        if profile_updated:
            trial_matches = await _find_matching_trials(state.patient_profile)
            trial_cache[session_id] = trial_matches
        else:
            trial_matches = trial_cache.get(session_id, [])

        gaps = await _analyze_gaps(trial_matches, state.patient_profile)

    # Step 3: Generate follow-up questions based on phase
    question_result = await question_generation_agent.process({
        "patient_profile": state.patient_profile,
        "phase": state.phase,
        "gaps": gaps,
        "trial_context": _get_trial_context(trial_matches) if trial_matches else None,
        "asked_medications": state.asked_medications,
        "asked_prior_treatments": state.asked_prior_treatments
    })

    # Update tracking flags if this question asks about medications or treatments
    next_tracks_field = question_result.get("next_tracks_field")
    if next_tracks_field == "asked_medications":
        state.asked_medications = True
    elif next_tracks_field == "asked_prior_treatments":
        state.asked_prior_treatments = True

    follow_up_questions = [q["question"] for q in question_result.get("questions", [])]

    # Step 4: Generate response
    response_content = await _generate_response(
        state=state,
        user_message=request.message,
        trial_matches=trial_matches,
        phase_changed=phase_changed,
        question_result=question_result,
        gaps=gaps,
        validation_errors=validation_errors
    )

    # Create response message
    assistant_message = ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=response_content,
        timestamp=datetime.utcnow()
    )

    # Add to history
    state.messages.append({
        "role": "assistant",
        "content": response_content,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Only show trial matches to the user after Phase 3 (or when Phase 3 is complete)
    # During Phase 2, trials are found but hidden while we ask follow-up questions
    show_trials = state.phase == 3 and len(gaps) == 0  # Phase 3 complete, no more gaps

    return ChatResponse(
        session_id=session_id,
        message=assistant_message,
        trial_matches=trial_matches if show_trials else [],
        patient_profile_updated=profile_updated,
        current_phase=state.phase,
        follow_up_questions=follow_up_questions
    )


async def _generate_response(
    state: ConversationState,
    user_message: str,
    trial_matches: List[TrialMatch],
    phase_changed: bool,
    question_result: dict,
    gaps: List[dict],
    validation_errors: List[str] = None
) -> str:
    """Generate an appropriate response based on conversation state and phase."""

    profile = state.patient_profile
    phase = state.phase
    suggested_response = question_result.get("suggested_response")
    is_first = _is_first_message(state)
    validation_errors = validation_errors or []

    # Handle validation errors - re-prompt the user with friendly correction
    if validation_errors:
        # Pick varied ways to gently correct
        corrections = [
            "Hmm, I think there might be a small mix-up. ",
            "Oops! I may have misunderstood. ",
            "Let me clarify - ",
            "Just to make sure I have this right - ",
        ]
        prefix = random.choice(corrections)
        error_msg = validation_errors[0]  # Show first error
        return f"{prefix}{error_msg}"

    # Handle first message with warm welcome
    if is_first and phase == 1:
        # Check if they already provided medical info in first message
        if profile.primary_condition:
            return f"Hello! I'm your Clinical Trial Assistant. Thank you for sharing that you're dealing with {profile.primary_condition}. I'll help you find clinical trials that might be a good fit. First, I need to gather some basic information about you. Could you tell me your age?"
        else:
            return "Hello! I'm your Clinical Trial Assistant. I'll ask you a few questions to help find clinical trials that might be a good fit for your situation. Let's start - could you tell me about the medical condition you're seeking treatment for?"

    # Build phase-specific system prompt
    if phase == 1:
        phase_instruction = """You are in Phase 1 (Baseline Screening).
Your goal is to gather basic information: medical condition, age, biological sex, location, diagnosis date, travel preference, medications, and prior treatments.
Be warm and conversational. Ask one question at a time.
Acknowledge what they've shared before asking for more information."""

    elif phase == 2:
        phase_instruction = f"""You are in Phase 2 (Trial-Driven Questioning).
You have found {len(trial_matches)} potential trial matches based on their condition.
Now you need to ask MORE SPECIFIC questions related to the eligibility criteria of these trials.
Explain that certain trials have specific requirements and that's why you're asking.
Be encouraging - let them know you're making progress in finding matches."""

    else:
        phase_instruction = """You are in Phase 3 (Gap-Filling & Final Clarification).
You're almost done! Focus on resolving the last few uncertainties.
Be encouraging and let them know you're close to finalizing their recommendations.
These are the final details needed to confirm eligibility for the best-matched trials."""

    system_prompt = f"""You are a friendly and empathetic clinical trial assistant.

{phase_instruction}

IMPORTANT - Response Variety Guidelines:
- NEVER start responses the same way twice in a row
- Vary your acknowledgments: "Thanks!", "Got it!", "Perfect!", "Great!", "Appreciate that!", "Wonderful!", "That helps!", "Noted!"
- Vary your transitions: "Now...", "Next up...", "Moving on...", "I'd also like to know...", "One more thing...", "Could you also tell me..."
- Keep a natural, conversational flow - like talking to a helpful friend
- Don't be robotic or use the same phrasing repeatedly

Other Guidelines:
- Be warm, empathetic and conversational
- Avoid medical jargon when possible
- Never provide medical advice
- Keep responses concise (2-3 sentences)
- Briefly acknowledge their answer, then ask the next question

Current patient profile:
{profile.model_dump_json(indent=2)}

{"Suggested follow-up question: " + suggested_response if suggested_response else ""}"""

    # Build context from recent messages
    recent_messages = state.messages[-6:]
    context = "\n".join([f"{m['role']}: {m['content']}" for m in recent_messages])

    # Phase transition: entering Phase 2 (trials found!)
    if phase_changed and phase == 2:
        prompt = f"""You just finished collecting baseline information and searched for clinical trials.
You found {len(trial_matches)} potential clinical trials that match their condition and location.

Now you need to review the eligibility criteria for these trials and ask follow-up questions to determine if they're a good fit.

Recent conversation:
{context}

Patient's message: {user_message}

Your response MUST follow this structure:
1. Thank them for providing all the baseline information
2. Say "Searching for potential trial matches..." or similar
3. Tell them you found {len(trial_matches)} potential trials that match their profile
4. Explain that each trial has specific eligibility requirements (inclusion/exclusion criteria)
5. Tell them you need to ask a few follow-up questions to determine which trials they qualify for
6. Ask the first eligibility question: {suggested_response or 'Ask about their medical history or other health conditions'}

Example tone: "Thank you for all that information! Let me search for clinical trials that match your profile... Great news - I found {len(trial_matches)} potential trials! Each trial has specific eligibility requirements, so I need to ask you a few more questions to see which ones you'd be a good fit for. [first question]"

Be encouraging - they're making great progress!"""

    # Phase transition: entering Phase 3 (almost done!)
    elif phase_changed and phase == 3:
        prompt = f"""You're entering the final phase of eligibility checking. Most information has been collected.

Recent conversation:
{context}

Patient's message: {user_message}

Respond by:
1. Acknowledge their response warmly
2. Let them know "We're almost done! Just a few final clarifications."
3. Explain these last details will help confirm which trials are the best fit
4. Ask the clarification question: {suggested_response or 'Ask for any final clarifying details'}

Be encouraging - they're very close to seeing their matched trials!"""

    # Phase 3 complete - no more gaps, ready to show trials!
    elif phase == 3 and len(gaps) == 0:
        eligible_trials = [m for m in trial_matches if m.eligibility_status == EligibilityStatus.ELIGIBLE]
        uncertain_trials = [m for m in trial_matches if m.eligibility_status == EligibilityStatus.UNCERTAIN]

        prompt = f"""The eligibility assessment is COMPLETE! All questions have been answered.

Results:
- {len(eligible_trials)} trials where the patient appears ELIGIBLE
- {len(uncertain_trials)} trials with UNCERTAIN eligibility (may still qualify)
- {len(trial_matches)} total trials reviewed

Recent conversation:
{context}

Patient's message: {user_message}

Respond by:
1. Thank them for answering all the questions
2. Announce that the eligibility assessment is complete
3. Summarize: "Based on your answers, I found X trials you appear to qualify for" (and mention uncertain ones if any)
4. Tell them the matched trials are now displayed below
5. Offer to answer any questions about the trials or help them understand the next steps

Be celebratory and helpful - this is the payoff for all their effort!"""

    # Phase 1: Still collecting baseline info
    elif phase == 1:
        # Determine what's missing in priority order
        next_question = None
        next_reason = None

        if not profile.primary_condition:
            next_question = "what medical condition they are seeking treatment for"
            next_reason = "This is the starting point for finding relevant trials."
        elif profile.age is None:
            next_question = "their age"
            next_reason = "Many trials have specific age requirements."
        elif not profile.biological_sex:
            next_question = "their biological sex (male/female)"
            next_reason = "Some trials are designed for specific groups."
        elif not profile.country:
            next_question = "what country they are located in"
            next_reason = "Clinical trials are conducted at specific locations."
        elif not profile.state_province:
            next_question = "which state or province they are in"
            next_reason = "This helps find trials closer to them."
        elif profile.willing_to_travel is None:
            next_question = "whether they would be willing to travel for a trial"
            next_reason = "Some excellent trials may require travel."
        elif not profile.diagnosis_date:
            next_question = "when they were first diagnosed"
            next_reason = "Some trials are for newly diagnosed patients, others for longer-term."
        elif not state.asked_medications:
            next_question = "what medications they are currently taking (or none)"
            next_reason = "Some trials have requirements about current medications."
        elif not state.asked_prior_treatments:
            next_question = "what treatments they have already tried (or none)"
            next_reason = "Many trials look for patients who have or haven't tried specific treatments."

        # Count remaining questions
        remaining = []
        if not profile.primary_condition: remaining.append("condition")
        if profile.age is None: remaining.append("age")
        if not profile.biological_sex: remaining.append("sex")
        if not profile.country: remaining.append("country")
        if not profile.state_province: remaining.append("state")
        if profile.willing_to_travel is None: remaining.append("travel preference")
        if not profile.diagnosis_date: remaining.append("diagnosis date")
        if not state.asked_medications: remaining.append("medications")
        if not state.asked_prior_treatments: remaining.append("prior treatments")

        prompt = f"""You are in Phase 1: Baseline Screening. Collecting essential information BEFORE searching for trials.

Questions remaining: {len(remaining)} ({', '.join(remaining) if remaining else 'none'})
Next question should ask about: {next_question or 'nothing - all baseline collected!'}
Why this matters: {next_reason or 'N/A'}

Recent conversation:
{context}

Patient's latest message: "{user_message}"

CRITICAL INSTRUCTIONS:
1. Do NOT mention finding trials yet - you're still gathering info
2. Do NOT repeat the same opening phrase as your previous messages
3. Vary your acknowledgment - use different words each time (Thanks/Got it/Perfect/Great/Appreciate that/etc.)

Respond by:
1. Briefly acknowledge their answer in a FRESH way (not "Thank you for sharing that" every time)
2. Ask about: {next_question}
3. Optionally mention why briefly

Keep it to 2-3 sentences. Be natural and conversational, like a friendly assistant.

Example variety (DO NOT copy these exactly, make your own):
- "Got it! Now, [question]"
- "Perfect, thanks! [question]"
- "Great! [question]"
- "Appreciate that. [question]"

Suggested question to ask: {suggested_response or f'Ask about {next_question}'}"""

    # Phase 2: Continuing trial-driven questions
    elif phase == 2:
        high_priority_gaps = [g for g in gaps if g.get("priority") == "high"]

        prompt = f"""Continue asking trial-specific questions. {len(high_priority_gaps)} high-priority questions remaining.

Recent conversation:
{context}

Patient's message: {user_message}

Respond by:
1. Acknowledging their response
2. Asking the next trial-specific question naturally
3. Briefly mention why this matters for the trials (e.g., "Some trials require..." or "This helps determine...")

Incorporate this question: {suggested_response or 'Ask about their medical history details'}"""

    # Phase 3: Final clarifications
    else:
        prompt = f"""Final phase - resolving last uncertainties.

Recent conversation:
{context}

Patient's message: {user_message}

Respond by:
1. Acknowledging their response warmly
2. If there are more questions, ask gently and explain it's the last few details
3. If all questions are answered, let them know you're ready to show their matched trials

{"Ask this final clarification: " + suggested_response if suggested_response else "Thank them and indicate you have enough information."}"""

    try:
        response = await llm_service.generate(prompt, system_prompt, temperature=0.7)
        return response
    except Exception as e:
        print(f"Response generation error: {e}")

        # Varied acknowledgments for fallback responses
        acks = ["Got it!", "Thanks!", "Perfect!", "Great!", "Noted!", "Appreciate that!"]
        ack = random.choice(acks)

        # Fallback responses by phase - ask baseline questions in order
        if is_first:
            if profile.primary_condition:
                return f"Hello! I'm your Clinical Trial Assistant. Thanks for sharing that you're dealing with {profile.primary_condition}. First, I need to gather some basic information. Could you tell me your age?"
            return "Hello! I'm your Clinical Trial Assistant. I'll ask you a few questions to help find clinical trials that might be a good fit for your situation. Let's start - what medical condition are you hoping to find a trial for?"
        elif phase == 1:
            # Ask baseline questions in order with varied acknowledgments
            if not profile.primary_condition:
                return "Thanks for reaching out! What medical condition are you hoping to find a trial for?"
            elif profile.age is None:
                return f"{ack} Now, could you tell me your age? Many trials have specific age requirements."
            elif not profile.biological_sex:
                return f"{ack} What is your biological sex? Some trials are designed for specific groups."
            elif not profile.country:
                return f"{ack} What country are you located in? This helps me find trials you can access."
            elif not profile.state_province:
                return f"{ack} Which state or province are you in? This helps narrow down nearby trials."
            elif profile.willing_to_travel is None:
                return f"{ack} Would you be open to traveling for a trial, or do you prefer something close to home?"
            elif not profile.diagnosis_date:
                return f"{ack} When were you first diagnosed? A rough timeframe like 'last year' or 'a few months ago' works fine."
            elif not state.asked_medications:
                return f"{ack} Are you currently taking any medications? Just list them, or say 'none' if not."
            elif not state.asked_prior_treatments:
                return f"{ack} What treatments have you already tried for this condition? Or 'none' if you haven't tried any yet."
            else:
                return f"{ack} I have all the baseline info I need. Searching for potential trial matches... Great news - I found some trials that match your profile! Each trial has specific eligibility requirements, so I need to ask a few follow-up questions to see which ones you'd be a good fit for. Do you have any other medical conditions I should know about?"
        elif phase == 2:
            if phase_changed:
                return f"Thank you for all that information! Searching for potential trial matches... I found {len(trial_matches)} potential clinical trials! Each trial has specific eligibility criteria, so I need to ask some follow-up questions to determine which ones you qualify for. Could you tell me about any other medical conditions you have?"
            else:
                return f"{ack} To continue checking your eligibility for these trials, could you tell me about any other medical conditions or health issues you have?"
        elif phase == 3:
            if len(gaps) == 0:
                # Assessment complete - show trials
                eligible_count = sum(1 for m in trial_matches if m.eligibility_status == EligibilityStatus.ELIGIBLE)
                return f"Thank you for answering all my questions! Your eligibility assessment is complete. Based on your answers, I found {eligible_count} trial(s) you appear to qualify for. You can see your matched trials displayed below. Feel free to ask me any questions about these trials!"
            else:
                return f"{ack} We're almost done! Just a couple more details to finalize which trials you qualify for."


def _has_minimum_profile(profile: PatientProfile, state: ConversationState) -> bool:
    """
    Check if we have minimum baseline information to search for trials.

    Phase 1 must collect ALL of these before moving to Phase 2:
    - Primary condition (what they're seeking treatment for)
    - Age (many trials have age requirements)
    - Biological sex (some trials are sex-specific)
    - Country (trials are location-specific)
    - State/province (more specific location)
    - Willingness to travel
    - Diagnosis date (when they were diagnosed)
    - Current medications (asked, even if answer is "none")
    - Prior treatments (asked, even if answer is "none")
    """
    has_condition = bool(profile.primary_condition)
    has_age = profile.age is not None
    has_sex = bool(profile.biological_sex)
    has_country = bool(profile.country)
    has_state = bool(profile.state_province)
    has_travel_preference = profile.willing_to_travel is not None
    has_diagnosis_date = bool(profile.diagnosis_date)

    # For medications and treatments, check if we've asked (tracked in state)
    asked_meds = state.asked_medications
    asked_treatments = state.asked_prior_treatments

    return (has_condition and has_age and has_sex and has_country and
            has_state and has_travel_preference and has_diagnosis_date and
            asked_meds and asked_treatments)


def _should_transition_to_phase3(trial_matches: List[TrialMatch], gaps: List[dict]) -> bool:
    """Determine if we should move from Phase 2 to Phase 3."""
    if not trial_matches:
        return False

    # Move to Phase 3 if:
    # 1. We have at least one eligible trial, OR
    # 2. Most gaps have been addressed (few high-priority gaps remain)
    has_eligible = any(m.eligibility_status == EligibilityStatus.ELIGIBLE for m in trial_matches)

    high_priority_gaps = [g for g in gaps if g.get("priority") == "high"]

    return has_eligible or len(high_priority_gaps) <= 1


async def _analyze_gaps(trial_matches: List[TrialMatch], profile: PatientProfile) -> List[dict]:
    """Use Gap Analysis Agent to identify missing information."""
    if not trial_matches:
        return []

    gap_result = await gap_analysis_agent.process({
        "trial_matches": trial_matches,
        "patient_profile": profile
    })

    return gap_result.get("gaps", [])


def _get_trial_context(trial_matches: List[TrialMatch]) -> str:
    """Generate context about matched trials for question generation."""
    if not trial_matches:
        return ""

    contexts = []
    for match in trial_matches[:3]:  # Top 3 trials
        status = match.eligibility_status.value
        unknown_count = len(match.criteria_unknown)
        contexts.append(
            f"Trial {match.trial.nct_id}: {status}, {unknown_count} criteria need clarification"
        )

    return "; ".join(contexts)


async def _find_matching_trials(profile: PatientProfile) -> List[TrialMatch]:
    """Find and evaluate trials matching the patient profile."""

    # Discover trials
    discovery_result = await trial_discovery_agent.process({
        "patient_profile": profile,
        "max_results": 10
    })

    trials = discovery_result.get("trials", [])
    matches = []

    for trial in trials[:5]:  # Limit to top 5 for performance
        # Extract criteria
        criteria_result = await criteria_extraction_agent.process({"trial": trial})

        trial.inclusion_criteria = criteria_result.get("inclusion_criteria", [])
        trial.exclusion_criteria = criteria_result.get("exclusion_criteria", [])

        # Match against patient
        match_result = await eligibility_matching_agent.process({
            "patient_profile": profile,
            "trial": trial
        })

        if match_result.get("trial_match"):
            matches.append(match_result["trial_match"])

    # Sort by eligibility (eligible first, then uncertain, then ineligible)
    eligibility_order = {"eligible": 0, "uncertain": 1, "ineligible": 2}
    matches.sort(key=lambda m: eligibility_order.get(m.eligibility_status.value, 3))

    return matches


@router.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session state for debugging/monitoring."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    state = sessions[session_id]
    return {
        "session_id": session_id,
        "patient_profile": state.patient_profile.model_dump(),
        "phase": state.phase,
        "message_count": len(state.messages),
        "cached_trials": len(trial_cache.get(session_id, []))
    }


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id in sessions:
        del sessions[session_id]
    if session_id in trial_cache:
        del trial_cache[session_id]
    return {"status": "deleted"}
