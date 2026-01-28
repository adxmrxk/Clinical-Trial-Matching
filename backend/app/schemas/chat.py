from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from .trial import TrialMatch


class ChatMessage(BaseModel):
    """A single message in the conversation."""
    id: str
    role: str = Field(..., description="'user' or 'assistant'")
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ChatRequest(BaseModel):
    """Request from frontend to send a message."""
    session_id: Optional[str] = None
    message: str


class ChatResponse(BaseModel):
    """Response from backend with assistant message(s) and optional trial matches."""
    session_id: str
    message: ChatMessage  # Primary message (for backwards compatibility)
    messages: List[ChatMessage] = Field(default_factory=list)  # Multiple messages for phase transitions
    trial_matches: List[TrialMatch] = Field(default_factory=list)
    patient_profile_updated: bool = False
    current_phase: int = Field(1, description="Current questioning phase")
    follow_up_questions: List[str] = Field(default_factory=list)
