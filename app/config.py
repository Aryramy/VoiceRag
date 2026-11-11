import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # OpenRouter LLM / Embedding configuration
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_CHAT_MODEL: str = "openai/gpt-4o-mini"
    OPENROUTER_FAST_MODEL: str = "openai/gpt-4o-mini"
    OPENROUTER_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # Provider Selection: "gemini" | "openai" | "openrouter" | "cheaper_inference"
    STT_PROVIDER: str = "gemini"
    TTS_PROVIDER: str = "gemini"

    # Google Gemini API configuration
    GEMINI_API_KEY: str = ""
    GEMINI_STT_MODEL: str = "gemini-3.5-transcribe-live"
    GEMINI_TTS_MODEL: str = "gemini-3.8-flash-lite-tts"

    # Gemini Configurable Voice Mapping System (Verified prebuilt voices)
    GEMINI_TTS_VOICE_EN: str = "Kore"
    GEMINI_TTS_VOICE_HI: str = "Puck"
    GEMINI_TTS_VOICE_AR: str = "Leda"
    GEMINI_TTS_DEFAULT_VOICE: str = "Kore"

    # CheaperInference & Audio Cascade configuration
    CHEAPER_INFERENCE_API_KEY: str = ""
    CHEAPER_INFERENCE_BASE_URL: str = "https://api.cheaperinference.com/v1"
    CHEAPER_INFERENCE_MODEL: str = "deepseek-v4-flash-0731"

    # STT / TTS configuration (Defaults to CheaperInference for working Whisper/TTS endpoints)
    STT_API_KEY: str = ""
    STT_BASE_URL: str = ""
    STT_MODEL: str = "whisper-1"

    TTS_API_KEY: str = ""
    TTS_BASE_URL: str = ""
    TTS_MODEL: str = "elevenlabs/eleven-multilingual-v2"
    TTS_VOICE: str = "zT03pEAEi0VHKciJODfn"
    TTS_HINGLISH_MODEL: str = "elevenlabs/eleven-multilingual-v2"
    TTS_HINGLISH_VOICE: str = "zT03pEAEi0VHKciJODfn"

    # LLM / Embedding provider overrides (Defaults to high-quality multilingual model)
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_CHAT_MODEL: str = "gpt-4o-mini"
    LLM_FAST_MODEL: str = "gpt-4o-mini"

    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    OPENROUTER_STT_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_STT_MODEL: str = "openai/whisper-large-v3-turbo"
    OPENROUTER_TTS_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_TTS_MODEL: str = "elevenlabs/eleven-multilingual-v2"
    OPENROUTER_TTS_VOICE: str = "zT03pEAEi0VHKciJODfn"

    @property
    def effective_stt_api_key(self) -> str:
        return self.OPENROUTER_API_KEY or self.STT_API_KEY or self.CHEAPER_INFERENCE_API_KEY

    @property
    def effective_stt_base_url(self) -> str:
        # If effective STT API key is an OpenRouter key, base URL must be OpenRouter
        if self.effective_stt_api_key and self.effective_stt_api_key.startswith("sk-or-"):
            return self.OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1"
        return self.OPENROUTER_STT_BASE_URL or self.STT_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL

    @property
    def effective_stt_model(self) -> str:
        if "openrouter" in str(self.effective_stt_base_url).lower():
            model = self.OPENROUTER_STT_MODEL or "openai/whisper-large-v3-turbo"
            # whisper-1 on OpenRouter does not support verbose_json (required by LiveKit)
            return model if model != "whisper-1" else "openai/whisper-large-v3-turbo"
        return self.STT_MODEL or "whisper-1"

    @property
    def effective_tts_api_key(self) -> str:
        return self.OPENROUTER_API_KEY or self.TTS_API_KEY or self.CHEAPER_INFERENCE_API_KEY

    @property
    def effective_tts_base_url(self) -> str:
        # If effective TTS API key is an OpenRouter key, base URL must be OpenRouter
        if self.effective_tts_api_key and self.effective_tts_api_key.startswith("sk-or-"):
            return self.OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1"
        return self.OPENROUTER_TTS_BASE_URL or self.TTS_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL

    @property
    def effective_tts_model(self) -> str:
        if "openrouter" in str(self.effective_tts_base_url).lower():
            model = self.OPENROUTER_TTS_MODEL or "elevenlabs/eleven-multilingual-v2"
            return model if model != "tts-1" else "elevenlabs/eleven-multilingual-v2"
        return self.TTS_MODEL or "elevenlabs/eleven-multilingual-v2"

    @property
    def effective_tts_voice(self) -> str:
        raw_voice = self.OPENROUTER_TTS_VOICE or self.TTS_VOICE or "zT03pEAEi0VHKciJODfn"
        v_lower = str(raw_voice).strip().lower()
        # Default to Raju (ElevenLabs native Indian conversationalist - natural, warm, fluent)
        if v_lower in ("raju", "zt03peaei0vhkcijodfn", "alloy", "george", "echo", "fable", "onyx", "nova", "shimmer", "default"):
            return "zT03pEAEi0VHKciJODfn"
        if v_lower in ("monika", "2zrm7pkgwbpiau2jvvxc"):
            return "2zRM7PkgwBPiau2jvVXc"
        if v_lower in ("aakash", "n2al4jd45e882svx17su"):
            return "N2al4jd45e882svx17SU"
        if v_lower in ("vikram", "q6lbiuyq6l1h58xiplan"):
            return "q6lbiuyq6L1H58xIpLAN"
        if v_lower in ("karan", "a6d176ebfa69", "jjlrvsvj1m0zmrngzulo"):
            return "jjlrvsvJ1M0ZMRNgZuLO"
        if v_lower in ("aditi", "zcaohnbs3c14rbihafp1"):
            return "zcAOhNBS3c14rBihAFp1"
        if v_lower in ("priya", "tht5kcbeypx3keuqqhph"):
            return "ThT5KcBeYPX3keUQqHPh"
        return str(raw_voice).strip()

    @property
    def effective_stt_provider(self) -> str:
        prov = (self.STT_PROVIDER or "gemini").lower().strip()
        if prov == "gemini" and self.GEMINI_API_KEY:
            return "gemini"
        return "openai"

    @property
    def effective_tts_provider(self) -> str:
        prov = (self.TTS_PROVIDER or "gemini").lower().strip()
        if prov == "gemini" and self.GEMINI_API_KEY:
            return "gemini"
        return "openai"

    def get_tts_voice(self, lang: str = "en") -> str:
        """Returns the appropriate voice for the detected language and active TTS provider."""
        lang_code = (lang or "en").lower().strip()
        provider = self.effective_tts_provider

        # Detect target language category
        if any(h in lang_code for h in ("hi", "hin", "hinglish", "urdu")):
            target_lang = "hi"
        elif any(a in lang_code for a in ("ar", "ara", "arabic", "gulf", "qatari")):
            target_lang = "ar"
        else:
            target_lang = "en"

        if provider == "gemini":
            if target_lang == "hi":
                return self.GEMINI_TTS_VOICE_HI or "Puck"
            elif target_lang == "ar":
                return self.GEMINI_TTS_VOICE_AR or "Leda"
            return self.GEMINI_TTS_VOICE_EN or self.GEMINI_TTS_DEFAULT_VOICE or "Kore"
        else:
            # Fallback for OpenAI / ElevenLabs provider
            if target_lang == "hi":
                return self.TTS_HINGLISH_VOICE or "zT03pEAEi0VHKciJODfn"
            return self.effective_tts_voice

    @property
    def effective_llm_api_key(self) -> str:
        return self.OPENROUTER_API_KEY or self.LLM_API_KEY or self.CHEAPER_INFERENCE_API_KEY

    @property
    def effective_llm_base_url(self) -> str:
        return self.OPENROUTER_BASE_URL or self.LLM_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL

    @property
    def effective_chat_model(self) -> str:
        if "openrouter" in self.effective_llm_base_url.lower():
            return self.OPENROUTER_CHAT_MODEL or "openai/gpt-4o-mini"
        return self.LLM_CHAT_MODEL or "gpt-4o-mini"

    @property
    def effective_fast_model(self) -> str:
        if "openrouter" in self.effective_llm_base_url.lower():
            return self.OPENROUTER_FAST_MODEL or "openai/gpt-4o-mini"
        return self.LLM_FAST_MODEL or "gpt-4o-mini"

    @property
    def effective_embedding_api_key(self) -> str:
        return self.OPENROUTER_API_KEY or self.EMBEDDING_API_KEY or self.CHEAPER_INFERENCE_API_KEY

    @property
    def effective_embedding_base_url(self) -> str:
        return self.OPENROUTER_BASE_URL or self.EMBEDDING_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL

    @property
    def effective_embedding_model(self) -> str:
        return self.OPENROUTER_EMBEDDING_MODEL or self.EMBEDDING_MODEL or "BAAI/bge-small-en-v1.5"

    # LiveKit configuration
    LIVEKIT_URL: str = "ws://127.0.0.1:7880"
    LIVEKIT_API_KEY: str = "devkey"
    LIVEKIT_API_SECRET: str = "secret"

    # Embedded Qdrant local storage configuration
    QDRANT_STORAGE_PATH: str = "./qdrant_storage"
    QDRANT_COLLECTION_NAME: str = "enterprise_kb"
    FALLBACK_SCORE_THRESHOLD: float = 0.20
    FALLBACK_STRICT_RESPONSE: str = (
        "I'm sorry, but I couldn't find specific information regarding that in my knowledge base."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
