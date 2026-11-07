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

    OPENROUTER_STT_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_STT_MODEL: str = "openai/whisper-large-v3-turbo"
    OPENROUTER_TTS_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_TTS_MODEL: str = "elevenlabs/eleven-v4-turbo"
    OPENROUTER_TTS_VOICE: str = "george"

    # LiveKit configuration
    LIVEKIT_URL: str = "ws://127.0.0.1:7880"
    LIVEKIT_API_KEY: str = "devkey"
    LIVEKIT_API_SECRET: str = "secret"

    # Embedded Qdrant local storage configuration
    QDRANT_STORAGE_PATH: str = "./qdrant_storage"
    QDRANT_COLLECTION_NAME: str = "enterprise_kb"
    FALLBACK_SCORE_THRESHOLD: float = 0.45
    FALLBACK_STRICT_RESPONSE: str = (
        "I'm sorry, but I couldn't find specific information regarding that in my knowledge base."
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

