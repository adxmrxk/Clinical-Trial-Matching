from typing import Optional, List, Tuple, Any
from groq import AsyncGroq
from google import genai
from ..core.config import settings


class LLMService:
    """
    Service for interacting with LLMs (Groq and Google Gemini).
    Fallback order: Groq 1 -> Gemini 1 -> Gemini 2 -> Groq 2 (last resort)
    """

    def __init__(self):
        # List of (client, name, provider) tuples in fallback order
        self.clients: List[Tuple[Any, str, str]] = []
        self.current_index: int = 0

        # Build fallback chain: Groq 1 -> Gemini 1 -> Gemini 2 -> Groq 2

        # 1. Groq primary
        if settings.GROQ_API_KEY:
            client = AsyncGroq(api_key=settings.GROQ_API_KEY)
            self.clients.append((client, "GROQ_API_KEY (primary)", "groq"))

        # 2. Gemini primary
        if settings.GEMINI_API_KEY:
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            self.clients.append((client, "GEMINI_API_KEY (primary)", "gemini"))

        # 3. Gemini backup
        if settings.GEMINI_API_KEY_2:
            client = genai.Client(api_key=settings.GEMINI_API_KEY_2)
            self.clients.append((client, "GEMINI_API_KEY_2 (backup)", "gemini"))

        # 4. Groq last resort
        if settings.GROQ_API_KEY_2:
            client = AsyncGroq(api_key=settings.GROQ_API_KEY_2)
            self.clients.append((client, "GROQ_API_KEY_2 (last resort)", "groq"))

        # Log available services
        print(f"LLM Service initialized with {len(self.clients)} providers:")
        for _, name, provider in self.clients:
            print(f"  - {name} ({provider})")

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if the error is a rate limit error."""
        error_str = str(error).lower()
        return (
            "429" in error_str or
            "rate limit" in error_str or
            "rate_limit" in error_str or
            "quota" in error_str or
            "resource exhausted" in error_str
        )

    async def _try_groq(
            self,
            client: AsyncGroq,
            messages: List[dict],
            temperature: float,
            max_tokens: int
    ) -> str:
        """Try a Groq client."""
        response = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content

    async def _try_gemini(
            self,
            client: genai.Client,
            messages: List[dict],
            temperature: float,
            max_tokens: int
    ) -> str:
        """Try a Gemini client using the new google-genai SDK."""
        # Convert messages to Gemini format
        prompt_parts = []
        for msg in messages:
            if msg["role"] == "system":
                prompt_parts.append(f"Instructions: {msg['content']}\n\n")
            elif msg["role"] == "user":
                prompt_parts.append(msg["content"])

        full_prompt = "".join(prompt_parts)

        response = await client.aio.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=full_prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )
        return response.text

    async def generate(
            self,
            prompt: str,
            system_prompt: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 1024
    ) -> str:
        """
        Generate a response from the LLM.
        Fallback order: Groq 1 -> Gemini 1 -> Gemini 2 -> Groq 2
        """
        if not self.clients:
            raise RuntimeError("No LLM service available. Please configure GROQ_API_KEY or GEMINI_API_KEY.")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try each client in fallback order
        last_error = None
        for i in range(len(self.clients)):
            idx = (self.current_index + i) % len(self.clients)
            client, name, provider = self.clients[idx]

            try:
                if provider == "groq":
                    result = await self._try_groq(client, messages, temperature, max_tokens)
                else:  # gemini
                    result = await self._try_gemini(client, messages, temperature, max_tokens)

                # Success - update current index to this working client
                self.current_index = idx
                return result

            except Exception as e:
                print(f"LLM error ({name}): {e}")
                last_error = e

                if self._is_rate_limit_error(e):
                    print(f"Rate limited on {name}, trying next provider...")
                    continue
                else:
                    # Non-rate-limit error, still try next provider
                    continue

        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

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
