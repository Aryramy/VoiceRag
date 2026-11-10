import re

CITATION_REGEX = re.compile(r"\[chunk_[a-zA-Z0-9_\-]+\]|\[chunk_\d+\]")

def strip_citations_for_tts(text: str) -> str:
    """Strips citation markers like [chunk_1] or [chunk_uuid] before sending text to Speech Synthesis.
    
    This ensures that raw citation bracket markers are retained for logging and evaluation tracing
    in the LangGraph state, but never spoken aloud to the user by the TTS engine.
    """
    cleaned = CITATION_REGEX.sub("", text)
    # Fix whitespace before punctuation (e.g. 'word .' -> 'word.')
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    # Phonetic smoothing for Indian English / Roman Hinglish TTS (removes stammering on awkward transliterations)
    cleaned = re.sub(r"\bse\b", "say", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btoota\b", "tuta", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btut+a+\b", "tuta", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\btoot+a+\b", "tuta", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bsampark\b", "sumpark", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bAapex\b", "Apex", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpeese\b", "paise", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpeeson\b", "paison", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bvapas\b", "wapas", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\byah condition hai ki\b", "shart yeh hai ki", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bpata karegi\b", "update karegi", cleaned, flags=re.IGNORECASE)
    # Collapse multiple spaces or whitespace artifacts
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
