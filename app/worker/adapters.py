import asyncio
import io
import wave
import base64
import logging
from typing import Optional, Any

from app.config import settings
from livekit.agents import tts, stt, utils, DEFAULT_API_CONNECT_OPTIONS

logger = logging.getLogger("voice_adapters")

# Try importing Google GenAI SDK and LiveKit Google plugin
try:
    from google import genai
    _GENAI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    _GENAI_AVAILABLE = False

try:
    from livekit.plugins.google.beta import GeminiSTT
    _GEMINI_STT_AVAILABLE = True
except ImportError:
    GeminiSTT = None  # type: ignore
    _GEMINI_STT_AVAILABLE = False

from livekit.plugins import openai


class GeminiTTSAdapter(tts.TTS):
    """Provider-independent LiveKit TTS adapter for Google Gemini 3.8 Flash-Lite TTS

    Uses the Google GenAI Interactions API (POST /v1beta/interactions) to synthesize
    natural, high-fidelity conversational audio with dynamic multilingual voice mapping.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        voice: Optional[str] = None,
        api_key: Optional[str] = None,
        sample_rate: int = 24000,
        num_channels: int = 1,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self._model = model or settings.GEMINI_TTS_MODEL or "gemini-3.8-flash-lite-tts"
        self._voice = voice or settings.get_tts_voice("en") or "Kore"
        self._api_key = api_key or settings.GEMINI_API_KEY

        if not _GENAI_AVAILABLE:
            raise RuntimeError("google-genai package is not installed.")

        self._client = genai.Client(
            api_key=self._api_key,
            http_options={"api_version": "v1beta"}
        )

    @property
    def model(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "google-gemini"

    @property
    def voice(self) -> str:
        return self._voice

    def update_voice(self, voice_name: str) -> None:
        """Dynamically switch voice persona between turns (e.g., English -> Kore, Hindi -> Puck, Arabic -> Leda)."""
        if voice_name and voice_name != self._voice:
            logger.info(f"GeminiTTSAdapter voice updated: '{self._voice}' -> '{voice_name}'")
            self._voice = voice_name

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Any = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _GeminiTTSChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options
        )


class _GeminiTTSChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        adapter: GeminiTTSAdapter = self._tts  # type: ignore
        loop = asyncio.get_running_loop()

        def _call_interactions_api():
            return adapter._client.interactions.create(
                model=adapter._model,
                input=[{
                    "type": "user_input",
                    "content": [{
                        "type": "text",
                        "text": self._input_text,
                        "annotations": [{
                            "type": "speech_metadata",
                            "style": "natural and conversational",
                        }],
                    }],
                }],
                response_format={"type": "audio"},
                generation_config={
                    "speech_config": [
                        {"voice": adapter._voice},
                    ]
                },
            )

        interaction = await loop.run_in_executor(None, _call_interactions_api)
        audio_b64 = getattr(interaction.output_audio, "data", None)
        if not audio_b64:
            logger.warning("Gemini TTS response contained no audio data")
            return

        wav_bytes = base64.b64decode(audio_b64)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            pcm_bytes = wf.readframes(wf.getnframes())
            rate = wf.getframerate()
            channels = wf.getnchannels()

        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=rate,
            num_channels=channels,
            mime_type="audio/pcm",
        )
        output_emitter.push(pcm_bytes)
        output_emitter.flush()


def create_stt_adapter(vad=None, prompt: Optional[str] = None) -> stt.STT:
    """Creates a replaceable STT adapter conforming to LiveKit's stt.STT interface.

    Returns Google Gemini 3.5 Transcribe Live when configured, otherwise falls back
    to OpenAI-compatible Whisper with StreamAdapter.
    """
    provider = settings.effective_stt_provider

    if provider == "gemini" and _GEMINI_STT_AVAILABLE and settings.GEMINI_API_KEY:
        logger.info(
            f"Instantiating Gemini 3.5 Transcribe Live STT (model={settings.GEMINI_STT_MODEL}, automatic language detection)"
        )
        return GeminiSTT(
            model=settings.GEMINI_STT_MODEL or "gemini-3.5-transcribe-live",
            language=None,
            language_codes=[],  # Automatic multilingual detection across EN, HI, AR
            api_key=settings.GEMINI_API_KEY,
            http_options={"api_version": "v1alpha"},
        )

    logger.info(
        f"Falling back to OpenAI-compatible STT (model={settings.effective_stt_model}, base_url={settings.effective_stt_base_url})"
    )
    base_stt = openai.STT(
        model=settings.effective_stt_model,
        base_url=settings.effective_stt_base_url,
        api_key=settings.effective_stt_api_key,
        prompt=prompt or "Haanji, boliye. Main aapki kya madad kar sakta hoon?",
        use_realtime=False,
        language="",
        detect_language=True,
    )
    if vad is not None:
        return stt.StreamAdapter(stt=base_stt, vad=vad)
    return base_stt


def create_tts_adapter(voice: Optional[str] = None) -> tts.TTS:
    """Creates a replaceable TTS adapter conforming to LiveKit's tts.TTS interface.

    Returns Google Gemini 3.8 Flash-Lite TTS when configured, otherwise falls back
    to OpenAI/ElevenLabs compatible TTS.
    """
    provider = settings.effective_tts_provider

    if provider == "gemini" and _GENAI_AVAILABLE and settings.GEMINI_API_KEY:
        chosen_voice = voice or settings.get_tts_voice("en")
        logger.info(
            f"Instantiating Gemini 3.8 Flash-Lite TTS (model={settings.GEMINI_TTS_MODEL}, voice={chosen_voice})"
        )
        return GeminiTTSAdapter(
            model=settings.GEMINI_TTS_MODEL or "gemini-3.8-flash-lite-tts",
            voice=chosen_voice,
            api_key=settings.GEMINI_API_KEY,
        )

    chosen_voice = voice or settings.effective_tts_voice
    logger.info(
        f"Falling back to OpenAI-compatible TTS (model={settings.effective_tts_model}, voice={chosen_voice})"
    )
    return openai.TTS(
        model=settings.effective_tts_model,
        base_url=settings.effective_tts_base_url,
        api_key=settings.effective_tts_api_key,
        voice=chosen_voice,
    )

