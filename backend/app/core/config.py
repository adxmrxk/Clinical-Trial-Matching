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

    # LLM Settings (Groq) - supports multiple keys for rate limit fallback
    GROQ_API_KEY: Optional[str] = None
    GROQ_API_KEY_2: Optional[str] = None  # Backup key when first one hits rate limit
    GROQ_API_KEY_3: Optional[str] = None  # Third backup key
    LLM_MODEL: str = "llama-3.3-70b-versatile"

    # OpenAI (alternative)
    OPENAI_API_KEY: Optional[str] = None


settings = Settings()
