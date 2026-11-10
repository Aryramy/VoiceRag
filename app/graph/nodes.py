import asyncio
import logging
import re
import time
import weakref
from typing import Dict, Any, AsyncIterator, List, Tuple
import httpx
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.core.state import AgenticVoiceState
from app.core.qdrant_client import search_qdrant
from app.config import settings

logger = logging.getLogger("graph_nodes")
FALLBACK_STRICT_RESPONSE = settings.FALLBACK_STRICT_RESPONSE
HINGLISH_FALLBACK_RESPONSE = "Arey, is baare mein hamare documents mein koi information nahin hai."

FEW_SHOT_DEMONSTRATIONS = """FEW-SHOT EXAMPLES FOR ENGLISH-TO-ROMAN-HINGLISH TRANSLATION:
EXAMPLE 1:
User: "Return policy kya hai?"
Context: "[chunk_1] Customers can return eligible items within 30 days of delivery for a full refund if accompanied by the original receipt."
Assistant Output: "Haan, aap delivery ke 30 din ke andar item return kar sakte hain, bas original receipt saath honi chahiye. [chunk_1]"

EXAMPLE 2:
User: "Meri delivery kab tak aayegi?"
Context: "[chunk_2] Standard shipping takes between 3 to 5 business days across all metropolitan areas."
Assistant Output: "Arey, standard delivery mein lagbhag 3 se 5 business days lagte hain aapke area tak. [chunk_2]"

EXAMPLE 3:
User: "Pricing details batao."
Context: "[chunk_3] The enterprise subscription starts at $49 per month per seat with annual billing."
Assistant Output: "Dekhiye, enterprise subscription 49 dollars per month per seat se shuru hota hai, jo annually bill kiya jaata hai. [chunk_3]"
"""

_UNICODE_PUNCT_TABLE = str.maketrans({
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2026": "...",
    "\u00a0": " ",
})


def sanitize_roman_ascii(text: str, collapse_spaces: bool = False) -> str:
    """Removes any accidental Devanagari, Urdu/Arabic, or other non-ASCII characters for clean ElevenLabs TTS."""
    if not text:
        return ""
    cleaned = text.translate(_UNICODE_PUNCT_TABLE)
    cleaned = re.sub(r"[^\x00-\x7F]+", "", cleaned)
    if collapse_spaces:
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned).strip()
    return cleaned

# Distinct Hindi/Urdu keywords in Roman script that do not collide with English
HINGLISH_KEYWORDS = {
    'agar', 'kya', 'paani', 'pani', 'mein', 'hai', 'hain', 'achcha', 'acha', 
    'kese', 'kaise', 'kab', 'kahan', 'kaha', 'thik', 'theek', 'aap', 'aapka', 'aapki', 'aapke',
    'tum', 'tumhara', 'hum', 'hamara', 'hamari', 'hamare', 'batlao', 'batao', 'bataiye', 'boliye', 'kisse', 'kis', 
    'kun', 'kyun', 'kyu', 'nahi', 'nahin', 'hoga', 'hogi', 'honge', 'karte', 
    'karta', 'karti', 'sakta', 'sakte', 'sakti', 'chahiye', 'chahta', 'chahti', 'chahte', 
    'milega', 'milegi', 'milenge', 'kitne', 'kitna', 'kitni', 'din', 'lagenge', 'lagta', 
    'lagti', 'shuru', 'hota', 'hoti', 'karna', 'karein', 'karo', 'toh', 'aur', 
    'yeh', 'ye', 'woh', 'wo', 'hona', 'gaya', 'gayi', 'gaye', 'hoon', 'hun', 'bhi', 'wala', 'wali',
    'namaste', 'shukriya', 'dhanyawad', 'mujhe', 'mera', 'meri', 'mere', 'baare', 'jankari', 'maaf', 'kijiye',
    'haan', 'nhi', 'bhee', 'acha', 'zaroor', 'samajh', 'lagbhag', 'matlab'
}

# Context-dependent pronouns that require prior conversation history resolution
HISTORY_PRONOUNS = {
    "it", "its", "it's", "that", "those", "these", "they", "them", "their", "theirs",
    "he", "him", "his", "she", "her", "hers", "former", "latter", "same",
    "iska", "iski", "iske", "uska", "uski", "uske", "woh", "wo", "unka", "unki", "inhe", "unhe",
}

HISTORY_FOLLOWUP_PHRASES = (
    "what about", "how about", "tell me more", "explain that", "translate",
    "in english", "in hindi", "in hinglish", "same thing", "previous", "above", "bataya",
)

# Hinglish-to-English domain supplement for deterministic semantic query expansion
HINGLISH_DOMAIN_EXPANSIONS = {
    "paani": "water liquid moisture",
    "pani": "water liquid moisture",
    "gir": "drop accidental damage",
    "girna": "drop accidental damage",
    "gira": "drop accidental damage",
    "gayi": "dropped",
    "warranty": "warranty hardware replacement coverage",
    "garanti": "warranty coverage",
    "guarantee": "warranty coverage",
    "saal": "year years limited warranty",
    "return": "return policy eligibility",
    "wapas": "return policy",
    "wapis": "return policy",
    "waapas": "return policy",
    "wapsi": "return policy",
    "policy": "policy rules",
    "product": "product item",
    "refund": "refund processing days payment",
    "paise": "refund payment fee",
    "paisa": "refund payment fee",
    "din": "days window timeline",
    "window": "return window days",
    "milta": "eligible period",
    "milte": "eligible period",
    "milti": "coverage included",
    "cover": "cover covered coverage",
    "repair": "repair third-party",
    "accidental": "accidental drops damage",
    "drops": "accidental drops",
    "contact": "contact support hours",
    "hours": "working hours schedule Eastern Time",
    "time": "hours timeline business days",
    "baje": "hours schedule AM PM",
    "support": "customer technical support Tier-2",
    "engineers": "Tier-2 support engineers",
    "shipping": "shipping delivery expedited",
    "delivery": "delivery shipping days",
    "receipt": "original receipt proof of purchase",
    "packaging": "original packaging unused",
    "defective": "defective manufacturing defects",
    "defect": "manufacturing defects",
    "kharab": "defective manufacturing defects",
    "khrab": "defective manufacturing defects",
    "toot": "broken accidental damage",
    "tut": "broken accidental damage",
    "kab": "when hours timeline",
    "kitne": "how many how long",
    "kitna": "how much how long",
    "kitni": "how many",
    "hardware": "hardware items",
    # Devanagari & Urdu/Arabic domain terms (in case upstream STT emits native script on heavy Hindi/Urdu)
    "रिटर्न": "return policy eligibility",
    "वापस": "return policy eligibility",
    "वापसी": "return policy eligibility",
    "पॉलिसी": "policy rules eligibility",
    "वाप्स": "return policy eligibility",
    "واپس": "return policy eligibility",
    "واپسی": "return policy eligibility",
    "پالیسی": "policy rules eligibility",
    "रिफंड": "refund processing days payment",
    "पैसा": "refund payment fee",
    "पैसे": "refund payment fee",
    "رقم": "refund payment fee",
    "پیسے": "refund payment fee",
    "वारंटी": "warranty hardware replacement coverage",
    "लिमिटेड": "limited warranty",
    "गारंटी": "warranty coverage",
    "وارنٹی": "warranty hardware replacement coverage",
    "گارنٹی": "warranty coverage",
    "हार्डवेयर": "hardware items",
    "ہارڈویئر": "hardware items",
    "ہارڈ": "hardware",
    "ویئر": "items",
    "आइटम्स": "hardware items",
    "آئٹمز": "hardware items",
    "प्रोडक्ट": "product item",
    "सामान": "product item",
    "پروڈکٹ": "product item",
    "پراڈکٹ": "product item",
    "खरीद": "purchase order",
    "खरीदा": "purchase order",
    "खरीदने": "purchase order",
    "खराब": "defective manufacturing defects",
    "टूटा": "broken accidental damage",
    "टूट": "broken accidental damage",
    "خراب": "defective manufacturing defects",
    "ٹوٹ": "broken accidental damage",
    "ٹوٹی": "broken accidental damage",
    "विंडो": "days return window timeline",
    "दिन": "days window timeline",
    "دن": "days window timeline",
    "साल": "year years limited warranty",
    "سال": "year years limited warranty",
    "एक्सीडेंटल": "accidental drops damage",
    "ड्रॉप्स": "accidental drops damage",
    "اکسیڈنٹل": "accidental drops damage",
    "ایکسیڈنٹل": "accidental drops damage",
    "ڈیمیج": "damage drops",
    "नقصان": "damage",
    "نقصان": "damage",
    "पानी": "water liquid moisture",
    "सपोर्ट": "customer technical support Tier-2",
    "मदद": "support help",
    "सहायता": "support help",
    "سپورٹ": "customer technical support Tier-2",
    "مدد": "support help",
    "इंजीनियर": "Tier-2 support engineers",
    "वर्किंग": "working hours schedule Eastern Time",
    "घंटे": "hours schedule",
    "शिपिंग": "shipping delivery days",
    "डिलीवरी": "delivery shipping days",
    "شپنگ": "shipping delivery days",
    "ڈلیوری": "delivery shipping days",
    "ऑर्डर": "order shipping",
    "آرڈر": "order shipping",
    "रसीद": "original receipt proof of purchase",
    "पैकेजिंग": "original packaging unused",
    "फीस": "fee charge",
    "फ़ीस": "fee charge",
    "चार्ज": "fee charge",
    "فیس": "fee charge",
    "چارج": "fee charge",
    "ईमेल": "email contact support",
    "फोन": "phone contact support",
    "फ़ोन": "phone contact support",
    "ای میل": "email contact support",
    "فون": "phone contact support",
    "एपेक्स": "Apex Global",
    "इपेक्स": "Apex Global",
    "ईपेक्स": "Apex Global",
    "ग्लोबल": "Global",
}

HINGLISH_STOP_PARTICLES = {
    "se", "ke", "ki", "ka", "ko", "par", "mein", "hai", "hain", "kya", "aur",
    "yeh", "ye", "karne", "liye", "karti", "karta", "karte", "hote", "hoti",
    "hota", "hoga", "hogi", "sakta", "sakte", "sakti", "lagta", "lagenge",
    "aane", "chahiye", "milega", "milegi", "ho", "toh", "mein", "bhi",
}

# Persistent loop-safe HTTP keep-alive & LLM client singletons
_llm_http_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
    weakref.WeakKeyDictionary()
)
_fast_llm_cache: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[Tuple[str, str, str], ChatOpenAI]]" = (
    weakref.WeakKeyDictionary()
)
_chat_llm_cache: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Dict[Tuple[str, str, str], ChatOpenAI]]" = (
    weakref.WeakKeyDictionary()
)
_fallback_clients: Dict[Tuple[str, str, str, float, bool], ChatOpenAI] = {}


def get_llm_http_client() -> httpx.AsyncClient:
    """Returns a persistent keep-alive httpx.AsyncClient for the active event loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        cached = _llm_http_clients.get(loop)
        if cached is not None and not cached.is_closed:
            return cached

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=5.0),
        limits=httpx.Limits(max_keepalive_connections=10, keepalive_expiry=60.0),
    )
    if loop is not None:
        _llm_http_clients[loop] = client
    return client


def _resolve_api_key(config: RunnableConfig | None) -> str:
    if config and "configurable" in config:
        cfg = config.get("configurable", {})
        if cfg.get("api_key"):
            return cfg["api_key"]
        or_key = cfg.get("openrouter_key")
        if or_key:
            # Avoid sending an OpenRouter key to CheaperInference when legacy callers only set openrouter_key
            if (
                "cheaperinference" in settings.effective_llm_base_url.lower()
                and or_key == settings.OPENROUTER_API_KEY
                and settings.effective_llm_api_key
                and settings.effective_llm_api_key != or_key
            ):
                return settings.effective_llm_api_key
            return or_key
    return settings.effective_llm_api_key


def get_fast_llm(api_key: str | None = None) -> ChatOpenAI:
    """Returns a persistent ChatOpenAI singleton for fast query rewriting."""
    resolved_key = api_key or settings.effective_llm_api_key
    model = settings.effective_fast_model
    base_url = settings.effective_llm_base_url
    key_tuple = (model, base_url, resolved_key)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        loop_map = _fast_llm_cache.setdefault(loop, {})
        if key_tuple in loop_map:
            return loop_map[key_tuple]
        t0 = time.perf_counter()
        llm = ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=resolved_key,
            temperature=0.0,
            http_async_client=get_llm_http_client(),
        )
        loop_map[key_tuple] = llm
        logger.info(f"[PERF][llm_init] Fast ChatOpenAI singleton created in {(time.perf_counter() - t0)*1000:.2f}ms")
        return llm

    fb_key = (*key_tuple, 0.0, False)
    if fb_key not in _fallback_clients:
        _fallback_clients[fb_key] = ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=resolved_key,
            temperature=0.0,
        )
    return _fallback_clients[fb_key]


def get_chat_llm(api_key: str | None = None) -> ChatOpenAI:
    """Returns a persistent streaming-capable ChatOpenAI singleton for answer generation."""
    resolved_key = api_key or settings.effective_llm_api_key
    model = settings.effective_chat_model
    base_url = settings.effective_llm_base_url
    key_tuple = (model, base_url, resolved_key)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None:
        loop_map = _chat_llm_cache.setdefault(loop, {})
        if key_tuple in loop_map:
            return loop_map[key_tuple]
        t0 = time.perf_counter()
        llm = ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=resolved_key,
            temperature=0.3,
            streaming=True,
            http_async_client=get_llm_http_client(),
        )
        loop_map[key_tuple] = llm
        logger.info(f"[PERF][llm_init] Chat ChatOpenAI singleton created in {(time.perf_counter() - t0)*1000:.2f}ms")
        return llm

    fb_key = (*key_tuple, 0.3, True)
    if fb_key not in _fallback_clients:
        _fallback_clients[fb_key] = ChatOpenAI(
            model=model,
            openai_api_base=base_url,
            openai_api_key=resolved_key,
            temperature=0.3,
            streaming=True,
        )
    return _fallback_clients[fb_key]


def warm_llm_clients(api_key: str | None = None) -> None:
    """Pre-initializes fast and chat LLM singletons + keep-alive HTTP pool on current loop."""
    if not (api_key or settings.effective_llm_api_key):
        return
    get_llm_http_client()
    get_fast_llm(api_key)
    get_chat_llm(api_key)


def detect_is_hinglish(text: str) -> bool:
    """Detects whether text contains Hindi/Hinglish words or Devanagari/Hinglish scripts."""
    if not text:
        return False
    if re.search(r'[\u0900-\u097F\u0600-\u06FF]', text):
        return True
    words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    return bool(words & HINGLISH_KEYWORDS)


def expand_hinglish_query(raw: str) -> Tuple[str, int]:
    """Maps common Roman Hinglish and native-script Hindi/Urdu domain terms to English search phrases deterministically."""
    tokens = re.findall(r'[a-zA-Z0-9\-]+|[\u0900-\u097F]+|[\u0600-\u06FF]+', raw)
    terms: List[str] = []
    seen = set()
    matched_domain_terms = 0

    for tok in tokens:
        low = tok.lower()
        is_ascii = tok.isascii()
        if low in HINGLISH_DOMAIN_EXPANSIONS:
            matched_domain_terms += 1
            for exp in HINGLISH_DOMAIN_EXPANSIONS[low].split():
                if exp.lower() not in seen:
                    seen.add(exp.lower())
                    terms.append(exp)
        elif tok in HINGLISH_DOMAIN_EXPANSIONS:
            matched_domain_terms += 1
            for exp in HINGLISH_DOMAIN_EXPANSIONS[tok].split():
                if exp.lower() not in seen:
                    seen.add(exp.lower())
                    terms.append(exp)
        if is_ascii and low not in HINGLISH_STOP_PARTICLES and low not in HINGLISH_KEYWORDS:
            if low not in seen:
                seen.add(low)
                terms.append(tok)

    expanded = " ".join(terms).strip()
    return (expanded if expanded else raw), matched_domain_terms


async def rewrite_query_node(state: AgenticVoiceState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    t_node_start = time.perf_counter()
    raw = state.get("raw_query", "").strip()
    if not raw:
        return {"rewritten_query": raw}

    history = state.get("messages", [])[-4:]
    user_is_hinglish = detect_is_hinglish(raw)
    word_tokens = re.findall(r'\b[a-zA-Z0-9\'\-]+\b', raw)
    word_tokens_lower = {w.lower() for w in word_tokens}
    raw_lower = raw.lower()

    needs_history_resolution = bool(history) and (
        bool(word_tokens_lower & HISTORY_PRONOUNS)
        or any(phrase in raw_lower for phrase in HISTORY_FOLLOWUP_PHRASES)
    )

    # Deterministic fast-path bypass: short self-contained English OR dictionary-mapped Hinglish
    if not user_is_hinglish and len(word_tokens) < 15 and not needs_history_resolution:
        logger.info(
            f"[PERF][rewrite_query] fast_path=english_bypass words={len(word_tokens)} "
            f"total={(time.perf_counter() - t_node_start)*1000:.2f}ms"
        )
        return {"rewritten_query": raw}

    has_native_script = bool(re.search(r'[\u0900-\u097F\u0600-\u06FF]', raw))
    if user_is_hinglish and not needs_history_resolution:
        expanded_query, matched_count = expand_hinglish_query(raw)
        native_tokens = re.findall(r'[\u0900-\u097F]+|[\u0600-\u06FF]+', raw) if has_native_script else []
        dict_bypass_ok = (
            (matched_count >= 2 or (matched_count >= 1 and len(native_tokens) <= 2))
            if has_native_script
            else matched_count >= 1
        )
        if dict_bypass_ok:
            logger.info(
                f"[PERF][rewrite_query] fast_path=hinglish_dict matched={matched_count} "
                f"total={(time.perf_counter() - t_node_start)*1000:.2f}ms"
            )
            return {"rewritten_query": expanded_query}

    api_key = _resolve_api_key(config)
    fallback_query = expand_hinglish_query(raw)[0] if user_is_hinglish else raw

    try:
        t_init0 = time.perf_counter()
        llm = get_fast_llm(api_key)
        t_init_ms = (time.perf_counter() - t_init0) * 1000

        prompt = (
            f"You are a query preprocessor for a semantic vector database.\n"
            f"Rewrite the following spoken user query into a concise, natural semantic search phrase optimized for text embedding retrieval against an English enterprise knowledge base.\n"
            f"Strict rules:\n"
            f"1. DO NOT write SQL, Cypher, code, operators, or boolean logic.\n"
            f"2. DO NOT add conversational filler, quotes, or punctuation.\n"
            f"3. If the user query is in Hindi, Urdu, or Hinglish, map the core intent into clear English search terms (e.g. 'phone paani mein girna' -> 'phone water liquid damage warranty coverage', 'refund aane mein kitne din lagenge' -> 'refund processing days timeline') so it matches English documents in the vector store.\n"
            f"4. Return ONLY the plain text search phrase.\n\n"
            f"Conversation History: {history}\n"
            f"User Query: {raw}"
        )
        t_llm0 = time.perf_counter()
        rw_timeout = 2.5 if has_native_script else 1.5
        res = await asyncio.wait_for(llm.ainvoke([HumanMessage(content=prompt)]), timeout=rw_timeout)
        t_llm_ms = (time.perf_counter() - t_llm0) * 1000
        rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
        logger.info(
            f"[PERF][rewrite_query] fast_path=llm_rewrite model={settings.effective_fast_model} "
            f"client_init={t_init_ms:.2f}ms llm_ainvoke={t_llm_ms:.1f}ms "
            f"total={(time.perf_counter() - t_node_start)*1000:.1f}ms"
        )
        return {"rewritten_query": rewritten if rewritten else fallback_query}
    except asyncio.TimeoutError:
        logger.warning(
            f"[PERF][rewrite_query] timeout_fallback={rw_timeout}s total={(time.perf_counter() - t_node_start)*1000:.1f}ms"
        )
        return {"rewritten_query": fallback_query}
    except Exception as e:
        logger.warning(
            f"[PERF][rewrite_query] error_fallback ({e}) total={(time.perf_counter() - t_node_start)*1000:.1f}ms"
        )
        return {"rewritten_query": fallback_query}


async def retrieve_node(state: AgenticVoiceState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("rewritten_query") or state.get("raw_query", "")
    try:
        results, max_score = await search_qdrant(
            query=query,
            top_k=3,
            min_score_threshold=settings.FALLBACK_SCORE_THRESHOLD,
        )
    except Exception as e:
        logger.warning(f"Qdrant retrieval failed ({e}), falling back to empty context")
        results, max_score = [], 0.0

    logger.info(
        f"[PERF][retrieve] qdrant_total={(time.perf_counter() - t0)*1000:.1f}ms "
        f"hits={len(results)} max_score={max_score:.3f}"
    )
    if not results or max_score < settings.FALLBACK_SCORE_THRESHOLD:
        return {"retrieved_context": [], "is_context_relevant": False}

    return {"retrieved_context": results, "is_context_relevant": True}


def detect_turn_hinglish(query: str, messages: Any = None) -> bool:
    """Analyzes current query and recent user turns while preserving strict Hinglish mirroring rules."""
    user_is_hinglish = detect_is_hinglish(query)
    if not user_is_hinglish and messages:
        for m in reversed(list(messages)[-4:]):
            if isinstance(m, HumanMessage) and detect_is_hinglish(str(m.content)):
                user_is_hinglish = True
                break
    return user_is_hinglish


def prepare_answer_prompt(state: AgenticVoiceState) -> Tuple[List[Any], List[Dict[str, Any]], bool, List[Dict[str, str]]]:
    """Builds exact grounding prompt messages, language mode, and citation metadata."""
    query = state.get("raw_query", "")
    retrieved_docs = state.get("retrieved_context", [])

    if retrieved_docs and state.get("is_context_relevant", False):
        context_str = "\n".join([f"[{doc['id']}] {doc['text']}" for doc in retrieved_docs])
        knowledge_block = f"RETRIEVED KNOWLEDGE BASE CONTEXT (IN ENGLISH):\n{context_str}"
        citation_rule = (
            "CITATION TRACKING:\n"
            "- For factual claims drawn from retrieved context, append internal tracking tags like [chunk_id] at the end of the sentence. (These are stripped before voice synthesis)."
        )
    else:
        knowledge_block = "NO NEW KNOWLEDGE BASE DOCUMENTS RETURNED FOR THIS TURN."
        citation_rule = (
            "NO-CONTEXT GUARDRAIL:\n"
            "- Because no knowledge base documents were returned, NEVER output bracketed [chunk_id] tags and NEVER answer using general or outside world knowledge.\n"
            "- Only answer if Conversation History directly contains the information needed to fulfill a follow-up, clarification, or translation request; otherwise reply ONLY with the exact strict fallback phrase."
        )

    user_is_hinglish = detect_turn_hinglish(query, state.get("messages"))

    if user_is_hinglish:
        sys_prompt = f"""You are a friendly, warm, calm, and jolly AI voice assistant speaking on a real-time voice call.

ABSOLUTE SCRIPT CONSTRAINT:
- OUTPUT 100% IN ROMAN HINGLISH (Latin Alphabet / ASCII English letters only).
- STRICT PROHIBITION: NEVER emit Devanagari (हिन्दी) or Urdu/Arabic (اردو) characters under ANY circumstances. Even if the user input or transcript uses native Devanagari or Urdu script, ALWAYS reply strictly in Roman Hinglish script!

PHONETIC TTS OPTIMIZATION (FOR ELEVENLABS ENGLISH VOICES):
- Write words phonetically so an English speech synthesizer pronounces them naturally like a native Indian/Pakistani speaker.
- Standard Hinglish Vocabulary Guidelines:
  * Use 'haan' or 'jee haan' (not 'ha'), 'nahin' (not 'nhi'), 'bhee' (not 'bhi'), 'achha' (not 'acha'), 'kyun' (not 'q').
  * Use 'aapka' / 'tumhara', 'kaise', 'bataiye', 'samajh', 'zaroor', 'shukriya'.
  * Keep common technical and business nouns in English: 'return policy', 'warranty', 'account', 'status', 'order', 'database', 'system', 'process', 'support'.
- Use natural conversational breathing pauses and filler words organically at the start of sentences (e.g., 'Arey...', 'Haan toh...', 'Dekhiye...', 'Waise...').

SPOKEN VOICE CONSTRAINTS:
- Keep responses strictly to 1 or 2 short conversational sentences.
- NEVER use lists, bullet points, numbers, bolding (**), asterisks, or markdown formatting. Write continuous conversational prose.

RAG TRANSLATION & GROUNDING PROTOCOL:
- You will receive knowledge base excerpts written in ENGLISH.
- DO NOT read English sentences verbatim since the user is conversing in Hindi/Urdu/Hinglish.
- Seamlessly translate the core factual meaning from the English context directly into conversational Roman Hinglish.
- Your factual knowledge is strictly limited to the provided RETRIEVED KNOWLEDGE BASE CONTEXT and relevant CONVERSATION HISTORY follow-ups. NEVER invent facts or use outside world knowledge.
- If the answer is not present in context or history, say cheerfully in Roman Hinglish:
  '{HINGLISH_FALLBACK_RESPONSE}'

{citation_rule}

{FEW_SHOT_DEMONSTRATIONS}

{knowledge_block}"""
    else:
        sys_prompt = f"""You are a warm, cool, calm, happy, and jolly AI voice assistant. You speak naturally like a helpful colleague on a real-time phone call.

LANGUAGE MODE:
- DETECTED USER LANGUAGE: PURE ENGLISH.
- MANDATORY: Reply strictly in natural, fluent English. Do not use Hindi/Hinglish.

SPOKEN VOICE CONSTRAINTS:
- Keep responses strictly to 1 or 2 short conversational sentences.
- NEVER use bullet points, numbered lists, bolding (**), asterisks, or Markdown formatting of any kind.
- Speak directly as a human speaks using smooth conversational prose.

STRICT GROUNDING & HALLUCINATION PREVENTION:
- Your factual knowledge is strictly limited to the provided RETRIEVED KNOWLEDGE BASE CONTEXT.
- NEVER invent facts, extrapolate unstated details, or use parametric outside knowledge.
- When "NO NEW KNOWLEDGE BASE DOCUMENTS RETURNED FOR THIS TURN" is indicated:
  a) First check CONVERSATION HISTORY to fulfill clarification/follow-up requests directly answered there.
  b) Otherwise reply strictly with: '{FALLBACK_STRICT_RESPONSE}'

{citation_rule}

{knowledge_block}"""

    prompt_messages: List[Any] = [SystemMessage(content=sys_prompt)]
    history_msgs = state.get("messages", [])
    for m in history_msgs[-6:]:
        prompt_messages.append(m)
    if not history_msgs or history_msgs[-1].content != query:
        prompt_messages.append(HumanMessage(content=query))

    citations = [
        {
            "chunk_id": str(doc.get("id", "chunk")),
            "source_file": str(doc.get("source_file", "Knowledge Base")),
            "text": str(doc.get("text", "")),
        }
        for doc in retrieved_docs
    ]
    return prompt_messages, retrieved_docs, user_is_hinglish, citations


async def stream_answer_chunks(
    state: AgenticVoiceState,
    api_key: str | None = None,
    result_holder: Dict[str, Any] | None = None,
) -> AsyncIterator[str]:
    """Streams LLM answer tokens in real time while recording TTFT, full text, and citations."""
    t_gen_start = time.perf_counter()
    prompt_messages, _, user_is_hinglish, citations = prepare_answer_prompt(state)
    fallback_msg = (
        HINGLISH_FALLBACK_RESPONSE
        if user_is_hinglish
        else FALLBACK_STRICT_RESPONSE
    )

    # Deterministic 0ms strict abstention when no relevant docs were found and query is not a conversation-history follow-up
    raw_query = (state.get("raw_query") or "").strip()
    raw_lower = raw_query.lower()
    raw_words = set(re.findall(r"\b[a-zA-Z0-9'\-]+\b", raw_lower))
    has_prior_agent_turn = any(isinstance(m, AIMessage) for m in (state.get("messages") or []))
    followup_tokens = HISTORY_PRONOUNS | {
        "this", "these", "those", "issey", "iske", "iskeliye", "isme", "usme",
        "mean", "meaning", "clarify", "repeat", "again", "short", "brief",
        "summarize", "summary", "explain", "elaborate", "details", "more",
        "hindi", "hinglish", "english", "urdu", "batao", "samjhao", "matlab",
    }
    is_history_followup = has_prior_agent_turn and (
        bool(raw_words & followup_tokens)
        or any(phrase in raw_lower for phrase in HISTORY_FOLLOWUP_PHRASES)
    )
    if not state.get("is_context_relevant", False) and not is_history_followup:
        logger.info(
            f"[PERF][generate_answer] fast_path=strict_abstain total={(time.perf_counter() - t_gen_start)*1000:.2f}ms"
        )
        yield fallback_msg
        if result_holder is not None:
            result_holder["final_answer"] = fallback_msg
            result_holder["citations"] = citations
            result_holder["user_is_hinglish"] = user_is_hinglish
        return

    chunks: List[str] = []
    first_token_ms: float | None = None

    try:
        t_init0 = time.perf_counter()
        llm = get_chat_llm(api_key)
        t_init_ms = (time.perf_counter() - t_init0) * 1000

        t_llm0 = time.perf_counter()
        async for chunk in llm.astream(prompt_messages):
            raw_piece = chunk.content if hasattr(chunk, "content") else str(chunk)
            if not raw_piece:
                continue
            piece = sanitize_roman_ascii(raw_piece, collapse_spaces=False)
            if not piece:
                continue
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - t_llm0) * 1000
            chunks.append(piece)
            yield piece

        t_llm_full_ms = (time.perf_counter() - t_llm0) * 1000
        raw_answer = "".join(chunks)
        answer_text = sanitize_roman_ascii(raw_answer, collapse_spaces=True)
        if not answer_text:
            answer_text = fallback_msg
            yield fallback_msg

        logger.info(
            f"[PERF][generate_answer] model={settings.effective_chat_model} "
            f"client_init={t_init_ms:.2f}ms ttft={first_token_ms or -1:.1f}ms "
            f"full_gen={t_llm_full_ms:.1f}ms chars={len(answer_text)} "
            f"total={(time.perf_counter() - t_gen_start)*1000:.1f}ms"
        )
    except Exception as e:
        logger.warning(f"Answer generation LLM stream failed ({e}), using fallback response")
        answer_text = fallback_msg
        if not chunks:
            yield fallback_msg

    if result_holder is not None:
        result_holder["final_answer"] = answer_text
        result_holder["citations"] = citations
        result_holder["user_is_hinglish"] = user_is_hinglish


async def generate_answer_node(state: AgenticVoiceState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    api_key = _resolve_api_key(config)
    result_holder: Dict[str, Any] = {}
    collected: List[str] = []
    async for piece in stream_answer_chunks(state, api_key=api_key, result_holder=result_holder):
        collected.append(piece)

    raw_answer_text = result_holder.get("final_answer") or "".join(collected)
    answer_text = sanitize_roman_ascii(raw_answer_text, collapse_spaces=True) or raw_answer_text.strip()
    citations = result_holder.get("citations", [])

    return {
        "final_answer": answer_text,
        "messages": [AIMessage(content=answer_text)],
        "citations": citations,
    }
