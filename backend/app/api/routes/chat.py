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


def _generate_phase2_transition_messages(
    trial_matches: List[TrialMatch],
    first_question: str = None
) -> List[ChatMessage]:
    """Generate multiple messages for Phase 2 transition.

    Splits the transition into separate chat bubbles for better UX:
    1. Acknowledgment + searching message
    2. Results found + explanation of eligibility
    3. Intro to follow-up questions
    """
    from datetime import timedelta

    base_time = datetime.utcnow()
    messages = []

    # Count only eligible + uncertain trials (these are the ones we'll show)
    eligible_count = sum(1 for m in trial_matches if m.eligibility_status == EligibilityStatus.ELIGIBLE)
    uncertain_count = sum(1 for m in trial_matches if m.eligibility_status == EligibilityStatus.UNCERTAIN)
    displayable_count = eligible_count + uncertain_count

    # Message 1: Acknowledgment and searching
    messages.append(ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content="Thanks for all that information! Let me search for clinical trials that match your profile...",
        timestamp=base_time
    ))

    # Message 2: Results found - be specific about what we found
    if displayable_count > 0:
        result_msg = f"Great news! I found {displayable_count} potential trial(s) that could be a good fit."
    else:
        result_msg = "I searched for trials matching your profile. Let me ask a few more questions to refine the results."

    messages.append(ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=result_msg,
        timestamp=base_time + timedelta(milliseconds=100)
    ))

    # Message 3: Intro to follow-up + first question
    if first_question:
        messages.append(ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=f"To confirm your eligibility, I have a few more questions. {first_question}",
            timestamp=base_time + timedelta(milliseconds=200)
        ))
    else:
        messages.append(ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content="To determine which trials you qualify for, I need to ask a few more questions.",
            timestamp=base_time + timedelta(milliseconds=200)
        ))

    return messages


def _generate_phase3_transition_messages(trial_matches: List[TrialMatch]) -> List[ChatMessage]:
    """Generate messages for Phase 3 transition (showing trials).

    Split into separate chat bubbles:
    1. Thank you + assessment complete
    2. Results summary (matches exactly what will be displayed)
    """
    from datetime import timedelta

    base_time = datetime.utcnow()
    messages = []

    # Count by eligibility - only eligible and uncertain will be shown
    eligible_count = sum(1 for m in trial_matches if m.eligibility_status == EligibilityStatus.ELIGIBLE)
    uncertain_count = sum(1 for m in trial_matches if m.eligibility_status == EligibilityStatus.UNCERTAIN)
    total_shown = eligible_count + uncertain_count

    # Message 1: Thank you
    messages.append(ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content="Thanks for answering all my questions! I've completed your eligibility assessment.",
        timestamp=base_time
    ))

    # Message 2: Results summary - be specific about exactly what's shown
    if total_shown == 0:
        result_msg = "Unfortunately, based on your profile, you don't appear to qualify for the trials I found. You may want to discuss other options with your healthcare provider."
    elif eligible_count > 0 and uncertain_count > 0:
        result_msg = f"I'm showing you {total_shown} trial(s) below: {eligible_count} that you appear to qualify for, and {uncertain_count} where you might qualify."
    elif eligible_count > 0:
        result_msg = f"Great news! I'm showing you {eligible_count} trial(s) below that you appear to qualify for."
    else:
        result_msg = f"I'm showing you {uncertain_count} trial(s) below where you might qualify. These are worth reviewing with your healthcare provider."

    messages.append(ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content=result_msg,
        timestamp=base_time + timedelta(milliseconds=100)
    ))

    return messages


def _get_brief_correction(error_code: str) -> str:
    """Convert error code to brief user-friendly correction message.

    Error codes are in format 'field:expected_type'.
    Returns a short, direct correction - NO question repetition.
    """
    corrections = {
        "age:number": "I need your age as a number (like 45 or 62).",
        "age:valid_range": "That age doesn't seem right. Just the number please.",
        "sex:male_or_female": "Just need male or female for that.",
        "country:name": "I need a country name (like USA, Canada, etc.).",
        "state:name": "I need a state or province name.",
        "travel:yes_or_no": "Just yes or no for the travel question.",
        "diagnosis:date_or_timeframe": "I need when you were diagnosed (like 'last year' or '2023').",
    }

    return corrections.get(error_code, "Could you rephrase that?")


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

                # Analyze gaps for Phase 2 questioning (skip LLM, use fast rule-based)
                gaps = await _analyze_gaps(trial_matches, state.patient_profile, skip_llm=True)

    elif state.phase == 2:
        # Phase 2: Trial-driven questioning
        # IMPORTANT: Never re-query API in Phase 2 - always use cached trials
        trial_matches = trial_cache.get(session_id, [])

        # Analyze remaining gaps (skip LLM for speed, filter already-asked)
        gaps = await _analyze_gaps(trial_matches, state.patient_profile, skip_llm=True)

        # Check if we should show trials:
        # 1. Asked 5 Phase 2 questions already, OR
        # 2. No more gaps to ask about
        if state.phase2_questions_count >= 5 or len(gaps) == 0:
            state.phase = 3
            phase_changed = True

    else:  # Phase 3
        # Phase 3: Show trials - no more questions
        # IMPORTANT: Never re-query API - always use cached trials
        trial_matches = trial_cache.get(session_id, [])
        gaps = []  # No more questions in Phase 3 - just show trials

    # Filter gaps to remove already-answered topics
    filtered_gaps = [
        g for g in gaps
        if g.get("attribute", "").lower() not in [t.lower() for t in state.answered_topics]
    ]

    # Step 3: Generate follow-up questions based on phase (NO LLM - instant)
    question_result = await question_generation_agent.process({
        "patient_profile": state.patient_profile,
        "phase": state.phase,
        "gaps": filtered_gaps,
        "trial_context": _get_trial_context(trial_matches) if trial_matches else None,
        "asked_medications": state.asked_medications,
        "asked_prior_treatments": state.asked_prior_treatments,
        "session_id": session_id,  # Track asked questions per session
        "answered_topics": state.answered_topics,  # Pass already answered topics
        "phase1_asked": set(state.phase1_asked),  # Phase 1 questions already asked - NEVER repeat
        "phase2_asked": set(state.phase2_asked),  # Phase 2 questions already asked - NEVER repeat
        "phase2_questions_count": state.phase2_questions_count  # Cap at 5
    })

    # Track the topic being asked so we don't ask again
    if question_result.get("questions"):
        asked_attr = question_result["questions"][0].get("attribute", "")
        if asked_attr:
            # Track in answered_topics
            if asked_attr.lower() not in [t.lower() for t in state.answered_topics]:
                state.answered_topics.append(asked_attr.lower())
            # Track Phase 1 questions separately - these are NEVER repeated
            if state.phase == 1 and asked_attr not in state.phase1_asked:
                state.phase1_asked.append(asked_attr)
            # Track Phase 2 questions - NEVER repeated, increment counter
            elif state.phase == 2 and asked_attr not in state.phase2_asked:
                state.phase2_asked.append(asked_attr)
                state.phase2_questions_count += 1

    # Update tracking flags if this question asks about medications or treatments
    next_tracks_field = question_result.get("next_tracks_field")
    if next_tracks_field == "asked_medications":
        state.asked_medications = True
    elif next_tracks_field == "asked_prior_treatments":
        state.asked_prior_treatments = True

    follow_up_questions = [q["question"] for q in question_result.get("questions", [])]

    # Step 4: Generate response(s)
    # For phase transitions, we return multiple messages for a better UX
    messages_list = []

    if phase_changed and state.phase == 2:
        # Phase 2 transition - split into multiple messages
        messages_list = _generate_phase2_transition_messages(
            trial_matches=trial_matches,
            first_question=question_result.get("suggested_response")
        )
    elif phase_changed and state.phase == 3:
        # Phase 3 transition - show trials!
        messages_list = _generate_phase3_transition_messages(trial_matches)
    else:
        # Single message for normal flow
        response_content = await _generate_response(
            state=state,
            user_message=request.message,
            trial_matches=trial_matches,
            phase_changed=phase_changed,
            question_result=question_result,
            gaps=gaps,
            validation_errors=validation_errors
        )
        messages_list = [ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response_content,
            timestamp=datetime.utcnow()
        )]

    # Primary message is the last one (for backwards compatibility)
    assistant_message = messages_list[-1] if messages_list else ChatMessage(
        id=str(uuid.uuid4()),
        role="assistant",
        content="I'm here to help you find clinical trials.",
        timestamp=datetime.utcnow()
    )

    # Add all messages to history
    for msg in messages_list:
        state.messages.append({
            "role": "assistant",
            "content": msg.content,
            "timestamp": msg.timestamp.isoformat()
        })

    # Show trial matches when we reach Phase 3 (after 5 Phase 2 questions or no more gaps)
    # During Phase 2, trials are found but hidden while we ask follow-up questions
    show_trials = state.phase == 3

    # Only show eligible and uncertain trials (not ineligible ones)
    displayable_trials = [
        m for m in trial_matches
        if m.eligibility_status in (EligibilityStatus.ELIGIBLE, EligibilityStatus.UNCERTAIN)
    ] if show_trials else []

    return ChatResponse(
        session_id=session_id,
        message=assistant_message,
        messages=messages_list,  # All messages for frontend to display
        trial_matches=displayable_trials,
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
    """Generate an appropriate response based on conversation state and phase.

    OPTIMIZED: Uses pre-built responses for speed. Only uses LLM for complex transitions.
    """
    profile = state.patient_profile
    phase = state.phase
    suggested_response = question_result.get("suggested_response")
    is_first = _is_first_message(state)
    validation_errors = validation_errors or []

    # Varied acknowledgments for natural conversation
    acks = ["Got it!", "Thanks!", "Perfect!", "Great!", "Noted!", "Appreciate that!", "Understood!"]
    ack = random.choice(acks)

    # Handle validation errors - brief correction, NO question repeat
    if validation_errors:
        # Just state what's needed, don't repeat the full question
        return _get_brief_correction(validation_errors[0])

    # Handle first message - user already got greeting from frontend
    if is_first and phase == 1:
        if profile.primary_condition:
            return f"Thanks for sharing that you're dealing with {profile.primary_condition}. I'll help you find trials that might be a good fit. Could you tell me your age?"
        else:
            return "Thanks for reaching out! What medical condition are you looking for a trial for?"

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
    # IMPORTANT: Only use suggested_response from question_generation_agent
    # That agent tracks phase1_asked to NEVER repeat questions
    elif phase == 1:
        # If no suggested_response, all questions have been asked - just acknowledge
        if not suggested_response:
            return f"{ack} I have all the information I need. Let me search for trials..."

        prompt = f"""You are in Phase 1: Baseline Screening.

Recent conversation:
{context}

Patient's latest message: "{user_message}"

CRITICAL INSTRUCTIONS:
1. Briefly acknowledge their answer (vary your acknowledgment each time)
2. Ask EXACTLY this question: {suggested_response}
3. Keep it to 2 sentences max

Do NOT ask any other question. Only ask the one provided above."""

    # Phase 2: Trial-driven questions
    # IMPORTANT: Only use suggested_response from question_generation_agent
    # That agent tracks phase2_asked to NEVER repeat questions
    elif phase == 2:
        # If no suggested_response, all Phase 2 questions done - move to show trials
        if not suggested_response:
            return f"{ack} I have all the information I need to show you your matched trials!"

        prompt = f"""You are in Phase 2: Asking trial eligibility questions.

Recent conversation:
{context}

Patient's message: {user_message}

CRITICAL INSTRUCTIONS:
1. Briefly acknowledge their answer (vary your acknowledgment each time)
2. Ask EXACTLY this question: {suggested_response}
3. Keep it to 2 sentences max

Do NOT ask any other question. Only ask the one provided above."""

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

        # Fallback: Use suggested_response directly (respects phase1_asked/phase2_asked tracking)
        if suggested_response:
            return f"{ack} {suggested_response}"
        elif phase == 3:
            eligible_count = sum(1 for m in trial_matches if m.eligibility_status == EligibilityStatus.ELIGIBLE)
            return f"Your eligibility assessment is complete. I found {eligible_count} trial(s) you may qualify for. Check them out below!"
        else:
            return f"{ack} Let me process that."


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


async def _analyze_gaps(trial_matches: List[TrialMatch], profile: PatientProfile, skip_llm: bool = False) -> List[dict]:
    """Use Gap Analysis Agent to identify missing information.

    Args:
        skip_llm: If True, only use rule-based gap detection (faster, no API call)
    """
    if not trial_matches:
        return []

    # Quick rule-based detection without LLM call
    if skip_llm:
        gaps = []
        for match in trial_matches:
            for criterion in match.criteria_unknown:
                gaps.append({
                    "attribute": criterion.attribute or "unknown",
                    "reason": criterion.original_text[:100],
                    "priority": "medium"
                })
        # Deduplicate by attribute
        seen = set()
        unique_gaps = []
        for g in gaps:
            if g["attribute"] not in seen:
                seen.add(g["attribute"])
                unique_gaps.append(g)
        return unique_gaps[:5]  # Limit to 5 gaps

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

    for trial in trials[:3]:  # Limit to top 3 for performance and API savings
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
    # Clear asked questions tracking
    question_generation_agent.clear_session(session_id)
    return {"status": "deleted"}
