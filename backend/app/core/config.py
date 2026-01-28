from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path

# Find the backend directory (where .env is located)
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Clinical Trial Matcher"

    # CORS
    FRONTEND_URL: str = "http://localhost:3000"

    # ClinicalTrials.gov API
    CLINICAL_TRIALS_API_BASE: str = "https://clinicaltrials.gov/api/v2"

    # LLM Settings - supports multiple keys for rate limit fallback
    # Fallback order: Groq 1 -> Gemini 1 -> Groq 2 (last resort)
    GROQ_API_KEY: Optional[str] = None
    GROQ_API_KEY_2: Optional[str] = None  # Last resort backup
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # Google Gemini
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_API_KEY_2: Optional[str] = None  # Gemini backup
    GEMINI_MODEL: str = "gemini-2.0-flash"


settings = Settings()
