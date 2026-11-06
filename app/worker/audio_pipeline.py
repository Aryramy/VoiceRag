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
    # Collapse multiple spaces or whitespace artifacts
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
