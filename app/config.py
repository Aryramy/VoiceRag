import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # OpenRouter LLM / Embedding configuration
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_CHAT_MODEL: str = "meta-llama/llama-3.1-70b-instruct"
    OPENROUTER_FAST_MODEL: str = "meta-llama/llama-3.1-8b-instruct"
    OPENROUTER_EMBEDDING_MODEL: str = "text-embedding-3-small"

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
    TTS_MODEL: str = "tts-1"
    TTS_VOICE: str = "alloy"
    TTS_HINGLISH_MODEL: str = "eleven-multilingual-v2"
    TTS_HINGLISH_VOICE: str = "q6lbiuyq6L1H58xIpLAN"

    # LLM / Embedding provider overrides (Defaults to CheaperInference when OpenRouter is unreachable)
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_CHAT_MODEL: str = "gpt-4o-mini"
    LLM_FAST_MODEL: str = "gpt-4o-mini"

    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    OPENROUTER_STT_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_STT_MODEL: str = "openai/whisper-large-v3-turbo"
    OPENROUTER_TTS_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_TTS_MODEL: str = "elevenlabs/eleven-v4-turbo"
    OPENROUTER_TTS_VOICE: str = "george"

    @property
    def effective_stt_api_key(self) -> str:
        return self.STT_API_KEY or self.CHEAPER_INFERENCE_API_KEY or self.OPENROUTER_API_KEY

    @property
    def effective_stt_base_url(self) -> str:
        return self.STT_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL or self.OPENROUTER_STT_BASE_URL

    @property
    def effective_tts_api_key(self) -> str:
        return self.TTS_API_KEY or self.CHEAPER_INFERENCE_API_KEY or self.OPENROUTER_API_KEY

    @property
    def effective_tts_base_url(self) -> str:
        return self.TTS_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL or self.OPENROUTER_TTS_BASE_URL

    @property
    def effective_llm_api_key(self) -> str:
        return self.LLM_API_KEY or self.CHEAPER_INFERENCE_API_KEY or self.OPENROUTER_API_KEY

    @property
    def effective_llm_base_url(self) -> str:
        return self.LLM_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL or self.OPENROUTER_BASE_URL

    @property
    def effective_chat_model(self) -> str:
        if "cheaperinference" in self.effective_llm_base_url.lower():
            return self.LLM_CHAT_MODEL or "gpt-4o-mini"
        return self.OPENROUTER_CHAT_MODEL

    @property
    def effective_fast_model(self) -> str:
        if "cheaperinference" in self.effective_llm_base_url.lower():
            return self.LLM_FAST_MODEL or "gpt-4o-mini"
        return self.OPENROUTER_FAST_MODEL

    @property
    def effective_embedding_api_key(self) -> str:
        return self.EMBEDDING_API_KEY or self.CHEAPER_INFERENCE_API_KEY or self.OPENROUTER_API_KEY

    @property
    def effective_embedding_base_url(self) -> str:
        return self.EMBEDDING_BASE_URL or self.CHEAPER_INFERENCE_BASE_URL or self.OPENROUTER_BASE_URL

    @property
    def effective_embedding_model(self) -> str:
        return self.EMBEDDING_MODEL or self.OPENROUTER_EMBEDDING_MODEL or "text-embedding-3-small"

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
