import pytest
import os
from dotenv import load_dotenv

load_dotenv()

from app.config import settings
from app.worker.adapters import create_stt_adapter, create_tts_adapter, GeminiTTSAdapter
from app.graph.nodes import detect_user_language
from app.worker.audio_pipeline import strip_citations_for_tts

def test_language_detection_multilingual():
    # English
    assert detect_user_language("What is the return policy?") == "en"
    assert detect_user_language("Can I get a refund on my purchase?") == "en"

    # Hindi / Hinglish
    assert detect_user_language("agar product damage nikle to kya refund hoga?") == "hi"
    assert detect_user_language("mujhe return policy ke baare mein bataiye") == "hi"
    assert detect_user_language("کیا واپسی ممکن ہے؟") == "hi"  # Urdu Nastaliq

    # Arabic (Standard & Gulf/Qatari phrasing)
    assert detect_user_language("مرحباً، شنو سياسة الترجيع في أبيكس جلوبال؟") == "ar"
    assert detect_user_language("أبي أرجع منتج خربان شريته قبل أسبوعين") == "ar"
    assert detect_user_language("كيف أقدر استبدل الجهاز تحت الضمان؟") == "ar"
    assert detect_user_language("هلا والله، شخبارك؟") == "ar"

def test_voice_mapping_system():
    # Ensure central voice mapping picks the right voice per language
    assert settings.get_tts_voice("en") == settings.GEMINI_TTS_VOICE_EN
    assert settings.get_tts_voice("hi") == settings.GEMINI_TTS_VOICE_HI
    assert settings.get_tts_voice("ar") == settings.GEMINI_TTS_VOICE_AR
    assert settings.get_tts_voice("gulf-arabic") == settings.GEMINI_TTS_VOICE_AR

def test_audio_pipeline_multilingual():
    # Hindi phonetic smoothing
    hindi_raw = "Product toota hua hai aur 5 se 7 working days lagenge."
    hindi_cleaned = strip_citations_for_tts(hindi_raw, lang="hi")
    assert "tuta" in hindi_cleaned
    assert "say" in hindi_cleaned

    # Arabic preservation (no Roman substitutions or garbled characters)
    arabic_raw = "أهلاً بك! تقدر ترجع المنتج خلال 30 يوم. [chunk_1234]"
    arabic_cleaned = strip_citations_for_tts(arabic_raw, lang="ar")
    assert "[chunk_1234]" not in arabic_cleaned
    assert "أهلاً بك! تقدر ترجع المنتج خلال 30 يوم." in arabic_cleaned

def test_adapters_instantiation():
    stt = create_stt_adapter()
    assert stt is not None
    assert hasattr(stt, "capabilities")

    tts = create_tts_adapter()
    assert tts is not None
    assert hasattr(tts, "synthesize")
    assert hasattr(tts, "update_voice")
