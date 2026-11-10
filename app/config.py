import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # OpenRouter LLM / Embedding configuration (Primary upstream routing)
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_CHAT_MODEL: str = "meta-llama/llama-3.1-8b-instruct"
    OPENROUTER_FAST_MODEL: str = "meta-llama/llama-3.1-8b-instruct"
    OPENROUTER_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # CheaperInference & Audio Cascade configuration (Secondary fallback when OpenRouter key is absent)
    CHEAPER_INFERENCE_API_KEY: str = ""
    CHEAPER_INFERENCE_BASE_URL: str = "https://api.cheaperinference.com/v1"
    CHEAPER_INFERENCE_MODEL: str = "deepseek-v4-flash-0731"

    # STT / TTS configuration (Used as secondary fallback or custom override)
    STT_API_KEY: str = ""
    STT_BASE_URL: str = ""
    STT_MODEL: str = "whisper-1"

    TTS_API_KEY: str = ""
    TTS_BASE_URL: str = ""
    TTS_MODEL: str = "tts-1"
    TTS_VOICE: str = "alloy"
    TTS_HINGLISH_MODEL: str = "eleven-multilingual-v2"
    TTS_HINGLISH_VOICE: str = "q6lbiuyq6L1H58xIpLAN"

    # LLM / Embedding provider overrides (Secondary fallback when OpenRouter key is absent)
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_CHAT_MODEL: str = "meta-llama/llama-3.1-8b-instruct"
    LLM_FAST_MODEL: str = "meta-llama/llama-3.1-8b-instruct"

    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    OPENROUTER_STT_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_STT_MODEL: str = "openai/gpt-4o-mini-transcribe"
    OPENROUTER_TTS_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_TTS_MODEL: str = "elevenlabs/eleven-v4-turbo"
    OPENROUTER_TTS_VOICE: str = "george"

    @property
    def effective_stt_api_key(self) -> str:
        return self.OPENROUTER_API_KEY or self.STT_API_KEY or self.CHEAPER_INFERENCE_API_KEY

    @property
    def effective_stt_base_url(self) -> str:
        if self.OPENROUTER_API_KEY:
            if self.OPENROUTER_STT_BASE_URL and "cheaperinference" not in self.OPENROUTER_STT_BASE_URL.lower():
                return self.OPENROUTER_STT_BASE_URL
            return self.OPENROUTER_BASE_URL
        return self.STT_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL

    @property
    def is_openrouter_stt(self) -> bool:
        return "openrouter.ai" in self.effective_stt_base_url.lower()

    @property
    def effective_stt_model(self) -> str:
        if self.is_openrouter_stt:
            # livekit.plugins.openai forces verbose_json for exact "whisper-1", which OpenRouter rejects with HTTP 400
            if self.OPENROUTER_STT_MODEL and self.OPENROUTER_STT_MODEL != "whisper-1":
                return self.OPENROUTER_STT_MODEL
            return "openai/gpt-4o-mini-transcribe"
        return self.STT_MODEL or "whisper-1"

    @property
    def effective_tts_api_key(self) -> str:
        return self.OPENROUTER_API_KEY or self.TTS_API_KEY or self.CHEAPER_INFERENCE_API_KEY

    @property
    def effective_tts_base_url(self) -> str:
        if self.OPENROUTER_API_KEY:
            if self.OPENROUTER_TTS_BASE_URL and "cheaperinference" not in self.OPENROUTER_TTS_BASE_URL.lower():
                return self.OPENROUTER_TTS_BASE_URL
            return self.OPENROUTER_BASE_URL
        return self.TTS_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL

    @property
    def is_openrouter_tts(self) -> bool:
        return "openrouter.ai" in self.effective_tts_base_url.lower()

    @property
    def effective_tts_model(self) -> str:
        if self.is_openrouter_tts:
            # Guard against OpenAI-only "tts-1" model ID when routed to OpenRouter
            if self.OPENROUTER_TTS_MODEL and self.OPENROUTER_TTS_MODEL not in ("tts-1", "openai/tts-1"):
                return self.OPENROUTER_TTS_MODEL
            return "elevenlabs/eleven-v4-turbo"
        return self.TTS_MODEL or "tts-1"

    @property
    def effective_tts_voice(self) -> str:
        if self.is_openrouter_tts:
            if self.OPENROUTER_TTS_VOICE and self.OPENROUTER_TTS_VOICE != "alloy":
                return self.OPENROUTER_TTS_VOICE
            return "george"
        return self.TTS_VOICE or "alloy"

    @property
    def effective_tts_hinglish_model(self) -> str:
        return self.TTS_HINGLISH_MODEL or "eleven-multilingual-v2"

    @property
    def effective_tts_hinglish_voice(self) -> str:
        return self.TTS_HINGLISH_VOICE or "q6lbiuyq6L1H58xIpLAN"

    @property
    def effective_llm_api_key(self) -> str:
        return self.OPENROUTER_API_KEY or self.LLM_API_KEY or self.CHEAPER_INFERENCE_API_KEY

    @property
    def effective_llm_base_url(self) -> str:
        if self.OPENROUTER_API_KEY:
            return self.OPENROUTER_BASE_URL
        return self.LLM_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL

    @property
    def effective_chat_model(self) -> str:
        if "cheaperinference" in self.effective_llm_base_url.lower():
            return self.LLM_CHAT_MODEL or "gpt-4o-mini"
        return self.OPENROUTER_CHAT_MODEL or "meta-llama/llama-3.1-8b-instruct"

    @property
    def effective_fast_model(self) -> str:
        if "cheaperinference" in self.effective_llm_base_url.lower():
            return self.LLM_FAST_MODEL or "gpt-4o-mini"
        return self.OPENROUTER_FAST_MODEL or "meta-llama/llama-3.1-8b-instruct"

    @property
    def effective_embedding_api_key(self) -> str:
        return self.OPENROUTER_API_KEY or self.EMBEDDING_API_KEY or self.CHEAPER_INFERENCE_API_KEY

    @property
    def effective_embedding_base_url(self) -> str:
        if self.OPENROUTER_API_KEY:
            return self.OPENROUTER_BASE_URL
        return self.EMBEDDING_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL

    @property
    def effective_embedding_model(self) -> str:
        # Local offline FastEmbed BGE is active in qdrant_client; guard against legacy remote OpenAI embedding names in .env
        if self.EMBEDDING_MODEL and "text-embedding-" not in self.EMBEDDING_MODEL.lower():
            return self.EMBEDDING_MODEL
        return "BAAI/bge-small-en-v1.5"

    # LiveKit configuration
    LIVEKIT_URL: str = "ws://127.0.0.1:7880"
    LIVEKIT_API_KEY: str = "devkey"
    LIVEKIT_API_SECRET: str = "secret"

    # Embedded Qdrant local storage configuration
    QDRANT_STORAGE_PATH: str = "./qdrant_storage"
    QDRANT_COLLECTION_NAME: str = "enterprise_kb"
    FALLBACK_SCORE_THRESHOLD: float = 0.55
    FALLBACK_STRICT_RESPONSE: str = (
        "I'm sorry, but I couldn't find specific information regarding that in my knowledge base."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
