from typing import Optional, List
from groq import AsyncGroq
from openai import AsyncOpenAI
from ..core.config import settings


class LLMService:
    """
    Service for interacting with LLMs (Groq/LLaMA or OpenAI).
    Supports multiple Groq API keys with automatic failover on rate limits.
    """

    def __init__(self):
        self.groq_clients: List[AsyncGroq] = []
        self.groq_key_names: List[str] = []  # For logging which key is being used
        self.current_groq_index: int = 0
        self.openai_client: Optional[AsyncOpenAI] = None

        # Initialize Groq clients (support multiple keys)
        if settings.GROQ_API_KEY:
            self.groq_clients.append(AsyncGroq(api_key=settings.GROQ_API_KEY))
            self.groq_key_names.append("GROQ_API_KEY (primary)")

        if settings.GROQ_API_KEY_2:
            self.groq_clients.append(AsyncGroq(api_key=settings.GROQ_API_KEY_2))
            self.groq_key_names.append("GROQ_API_KEY_2 (backup)")

        if settings.GROQ_API_KEY_3:
            self.groq_clients.append(AsyncGroq(api_key=settings.GROQ_API_KEY_3))
            self.groq_key_names.append("GROQ_API_KEY_3 (backup 2)")

        # Initialize OpenAI client
        if settings.OPENAI_API_KEY:
            self.openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

        # Log available services
        print(f"LLM Service initialized:")
        print(f"  - Groq keys available: {len(self.groq_clients)}")
        print(f"  - OpenAI available: {self.openai_client is not None}")

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the error is a rate limit error (429)."""
        error_str = str(error).lower()
        return "429" in error_str or "rate limit" in error_str or "rate_limit" in error_str

    async def _try_groq(
            self,
            messages: List[dict],
            temperature: float,
            max_tokens: int
    ) -> Optional[str]:
        """Try all available Groq clients, switching on rate limit errors."""

        if not self.groq_clients:
            return None

        # Try each Groq client starting from current index
        attempts = len(self.groq_clients)

        for _ in range(attempts):
            client = self.groq_clients[self.current_groq_index]
            key_name = self.groq_key_names[self.current_groq_index]

            try:
                response = await client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            except Exception as e:
                print(f"Groq error ({key_name}): {e}")

                # If rate limited, try the next key
                if self._is_rate_limit_error(e):
                    old_index = self.current_groq_index
                    self.current_groq_index = (self.current_groq_index + 1) % len(self.groq_clients)

                    if self.current_groq_index != old_index:
                        new_key_name = self.groq_key_names[self.current_groq_index]
                        print(f"Rate limited! Switching from {key_name} to {new_key_name}")
                    else:
                        print(f"All Groq keys are rate limited!")
                        return None  # All keys exhausted
                else:
                    # Non-rate-limit error, don't switch keys
                    return None

        return None

    async def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 1024
    ) -> str:
        """
        Generate a response from the LLM.
        Tries multiple Groq keys if available, then falls back to OpenAI.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # Try Groq (with automatic key switching on rate limit)
        result = await self._try_groq(messages, temperature, max_tokens)
        if result:
            return result

        # Fall back to OpenAI
        if self.openai_client:
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"OpenAI error: {e}")

        raise RuntimeError("No LLM service available. Please configure GROQ_API_KEY or OPENAI_API_KEY.")

    async def generate_json(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.3
    ) -> str:
        """
        Generate a JSON response from the LLM.
        Uses lower temperature for more deterministic output.
        """
        json_system = (system_prompt or "") + "\n\nRespond ONLY with valid JSON. No explanations or markdown."
        return await self.generate(prompt, json_system, temperature)


# Singleton instance
llm_service = LLMService()
