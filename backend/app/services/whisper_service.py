import tempfile
import os

# Make faster-whisper optional for deployment
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    WhisperModel = None


class WhisperService:
    """Service for speech-to-text using local Whisper model (faster-whisper)."""

    def __init__(self):
        self.model = None
        self.model_size = "base"  # Options: tiny, base, small, medium, large-v3
        if WHISPER_AVAILABLE:
            print(f"Whisper service initialized (model will load on first use: {self.model_size})")
        else:
            print("Whisper not available - speech-to-text disabled")

    def _load_model(self):
        """Lazy load the Whisper model on first use."""
        if not WHISPER_AVAILABLE:
            raise RuntimeError("Speech-to-text is not available on this server")

        if self.model is None:
            print(f"Loading Whisper model '{self.model_size}'... (this may take a moment)")
            # Use CPU with int8 quantization for efficiency
            # Change to "cuda" if you have an NVIDIA GPU
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8"
            )
            print("Whisper model loaded successfully!")

    async def transcribe(self, audio_data: bytes, filename: str = "audio.webm") -> str:
        """
        Transcribe audio data using local Whisper model.

        Args:
            audio_data: Raw audio bytes
            filename: Original filename (used to determine format)

        Returns:
            Transcribed text
        """
        # Load model if not already loaded
        self._load_model()

        # Write audio to temp file (faster-whisper needs a file path)
        suffix = os.path.splitext(filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        try:
            # Transcribe
            segments, info = self.model.transcribe(
                tmp_path,
                language="en",
                beam_size=5,
                vad_filter=True  # Filter out silence
            )

            # Combine all segments into one string
            text = " ".join(segment.text.strip() for segment in segments)
            return text

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# Singleton instance
whisper_service = WhisperService()
