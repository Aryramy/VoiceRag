import re
import logging
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.core.state import AgenticVoiceState
from app.core.qdrant_client import search_qdrant
from app.config import settings

logger = logging.getLogger("graph_nodes")
FALLBACK_STRICT_RESPONSE = settings.FALLBACK_STRICT_RESPONSE

def warm_llm_clients(api_key: str | None = None) -> None:
    pass


# Regex for Arabic script and Urdu-specific characters
ARABIC_SCRIPT_REGEX = re.compile(r'[\u0600-\u06FF]')
URDU_SPECIFIC_CHARS = set('ٹڈڑںےپچگھ')

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
    'namaste', 'shukriya', 'dhanyawad', 'mujhe', 'mera', 'meri', 'mere', 'baare', 'jankari', 'maaf', 'kijiye'
}

# Distinct Arabic / Gulf phrasing keywords in Latin transliteration
ARABIC_KEYWORDS = {
    'marhaba', 'shukran', 'ahlan', 'hala', 'kayf', 'keif', 'keifik', 'shlonak', 
    'shlonik', 'yaatik', 'afia', 'mishkoor', 'ayesh', 'sheeno', 'abi', 'ibghi', 'waddi'
}

def detect_user_language(text: str, history: List[Any] | None = None) -> str:
    """Accurately detects whether text is English ('en'), Hindi/Hinglish ('hi'), or Arabic ('ar')."""
    if not text:
        return "en"
    
    # Check for Arabic script
    if ARABIC_SCRIPT_REGEX.search(text):
        if any(c in text for c in URDU_SPECIFIC_CHARS):
            return "hi"
        return "ar"

    words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    
    # Check Latin Arabic keywords (transliterated Arabic / Gulf phrases)
    if bool(words & ARABIC_KEYWORDS):
        return "ar"

    # Check Hindi/Hinglish (Devanagari or Hinglish Roman vocabulary)
    if re.search(r'[\u0900-\u097F]', text) or bool(words & HINGLISH_KEYWORDS):
        return "hi"

    # Fallback to history if current query is neutral/ambiguous (e.g. "yes", "tell me more")
    if history:
        for m in reversed(history[-4:]):
            content = str(getattr(m, "content", "") or "")
            if content:
                lang = detect_user_language(content, history=None)
                if lang != "en":
                    return lang

    return "en"

def detect_is_hinglish(text: str) -> bool:
    """Backward compatibility helper for Hinglish detection."""
    return detect_user_language(text) == "hi"

async def romanize_script(text: str, api_key: str | None = None) -> str:
    """Translates/transliterates any Urdu Nastaliq or Devanagari script into natural, colloquial Roman Hinglish.
    Does NOT modify authentic Arabic script.
    """
    if not text or not re.search(r'[\u0600-\u06FF\u0900-\u097F]', text):
        return text
    
    # Do NOT romanize Arabic text
    if detect_user_language(text) == "ar":
        return text

    try:
        llm = ChatOpenAI(
            model=settings.effective_fast_model,
            openai_api_base=settings.effective_llm_base_url,
            openai_api_key=api_key or settings.effective_llm_api_key,
            temperature=0.0
        )
        sys_msg = (
            "You are a Roman Urdu / Hinglish transliterator. Your job is to convert Urdu or Hindi text into natural Roman English characters (Latin script). "
            "Output ONLY the transliterated text in standard English alphabet without quotes, explanations, or Arabic/Devanagari letters."
        )
        res = await llm.ainvoke([SystemMessage(content=sys_msg), HumanMessage(content=f"Convert to Roman script: {text}")])
        romanized = res.content.strip() if hasattr(res, "content") else str(res).strip()
        romanized = re.sub(r'^["\']|["\']$', '', romanized).strip()
        if romanized and not re.search(r'[\u0600-\u06FF\u0900-\u097F]', romanized):
            return romanized
    except Exception as e:
        logger.warning(f"Script romanization failed: {e}")
    return text

async def rewrite_query_node(state: AgenticVoiceState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    raw = state.get("raw_query", "")
    api_key = (
        config.get("configurable", {}).get("api_key")
        or config.get("configurable", {}).get("openrouter_key")
        if config and "configurable" in config
        else settings.effective_llm_api_key
    ) or settings.effective_llm_api_key

    try:
        llm = ChatOpenAI(
            model=settings.effective_fast_model,
            openai_api_base=settings.effective_llm_base_url,
            openai_api_key=api_key,
            temperature=0.0
        )
        history = state.get("messages", [])[-4:]
        
        prompt = (
            f"You are a query preprocessor for a semantic vector database.\n"
            f"Rewrite the following spoken user query into a concise, natural semantic search phrase optimized for text embedding retrieval against an English enterprise knowledge base.\n"
            f"Strict rules:\n"
            f"1. DO NOT write SQL, Cypher, code, operators, or boolean logic.\n"
            f"2. DO NOT add conversational filler, quotes, or punctuation.\n"
            f"3. If the user query is in Hindi, Urdu, Hinglish, or Arabic (including Gulf/Qatari Arabic), map the core intent into clear English search terms "
            f"(e.g. 'phone paani mein girna' -> 'phone water liquid damage warranty coverage', 'refund aane mein kitne din lagenge' -> 'refund processing days timeline', "
            f"'أبي أرجع منتج خربان' -> 'return defective product warranty refund policy') so it matches English documents in the vector store.\n"
            f"4. Return ONLY the plain text search phrase.\n\n"
            f"Conversation History: {history}\n"
            f"User Query: {raw}"
        )
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
        return {"rewritten_query": rewritten if rewritten else raw}
    except Exception as e:
        if settings.CHEAPER_INFERENCE_API_KEY and "openrouter" in str(settings.effective_llm_base_url).lower():
            try:
                fallback_llm = ChatOpenAI(
                    model=settings.CHEAPER_INFERENCE_MODEL or "deepseek-v4-flash-0731",
                    openai_api_base=settings.CHEAPER_INFERENCE_BASE_URL,
                    openai_api_key=settings.CHEAPER_INFERENCE_API_KEY,
                    temperature=0.0
                )
                res = await fallback_llm.ainvoke([HumanMessage(content=prompt)])
                rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
                if rewritten:
                    return {"rewritten_query": rewritten}
            except Exception:
                pass
        logger.warning(f"Query rewrite LLM call failed ({e}), falling back to raw query: {raw}")
        return {"rewritten_query": raw}

async def retrieve_node(state: AgenticVoiceState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    query = state.get("rewritten_query") or state.get("raw_query", "")
    try:
        results, max_score = await search_qdrant(
            query=query,
            top_k=3,
            min_score_threshold=settings.FALLBACK_SCORE_THRESHOLD
        )
    except Exception as e:
        logger.warning(f"Qdrant retrieval failed ({e}), falling back to empty context")
        results, max_score = [], 0.0
    
    if not results or max_score < settings.FALLBACK_SCORE_THRESHOLD:
        return {"retrieved_context": [], "is_context_relevant": False}
    
    return {"retrieved_context": results, "is_context_relevant": True}

async def generate_answer_node(state: AgenticVoiceState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    api_key = (
        config.get("configurable", {}).get("api_key")
        or config.get("configurable", {}).get("openrouter_key")
        if config and "configurable" in config
        else settings.effective_llm_api_key
    ) or settings.effective_llm_api_key

    query = state.get("raw_query", "")
    retrieved_docs = state.get("retrieved_context", [])
    
    if retrieved_docs and state.get("is_context_relevant", False):
        context_str = "\n".join([f"[{doc['id']}] {doc['text']}" for doc in retrieved_docs])
    else:
        context_str = "NO NEW KNOWLEDGE BASE DOCUMENTS RETURNED"

    # Analyze language of current query and recent conversation turns
    detected_lang = detect_user_language(query, history=state.get("messages", []))

    if detected_lang == "ar":
        language_directive = (
            "CRITICAL LANGUAGE MIRRORING ENFORCEMENT:\n"
            "DETECTED USER LANGUAGE: ARABIC (Modern Standard / Gulf / Qatari Arabic).\n"
            "MANDATORY: The user spoke in Arabic. You MUST output your final answer strictly in natural, professional, and friendly Arabic script (العربية).\n"
            "DIALECT AWARENESS: Maintain natural conversational Gulf/Qatari warmth (e.g. 'أهلاً بك', 'تقدر ترجع المنتج خلال 30 يوم', 'المبلغ يرجع لحسابك خلال 5 إلى 7 أيام عمل').\n"
            "SCRIPT: Write exclusively in clear Arabic letters. NEVER write in English alphabet or Latin transliteration.\n"
            "CRITICAL LENGTH LIMIT: Strictly maximum 1 or 2 short, conversational sentences (total under 30 words). Never output 3 or more sentences.\n"
            "NO MARKDOWN: Never use bullet points, bolding, numbered lists, or asterisks."
        )
    elif detected_lang == "hi":
        language_directive = (
            "CRITICAL LANGUAGE MIRRORING ENFORCEMENT:\n"
            "DETECTED USER LANGUAGE: HINGLISH (Hindi/Urdu/Hinglish in Roman Latin script).\n"
            "MANDATORY: The user spoke in Hindi/Urdu/Hinglish. You MUST output your final answer strictly in Colloquial Roman Hinglish (Latin English alphabet A-Z only).\n"
            "ABSOLUTE SCRIPT PROHIBITION: NEVER use Arabic, Urdu, or Devanagari characters (like ابپ or अआइ). Every single word MUST be written in standard English Latin letters.\n"
            "PHONETIC PRONUNCIATION ACCURACY: Spell Hindi/Urdu words using standard smooth Roman phonetics so TTS does not stutter or stammer (e.g., write 'toota' or 'damaged' - NEVER write 'tutaa'; write 'Apex' - NEVER write 'Aapex'; write 'theek' - NEVER write 'thik'; write 'yeh' - NEVER write 'yah'; write 'paise' - NEVER write 'peese'; write 'wapas' - NEVER write 'vapas').\n"
            "HINGLISH STYLE AND GRAMMAR GUIDE:\n"
            "- CRITICAL LENGTH LIMIT: Never speak more than 2 short sentences. Do not list all steps. Just give the primary answer directly.\n"
            "- Never use formal words like 'sampark' or 'ensure'. Use everyday casual words like 'contact', 'baat karein', or 'dhyaan rakhein'.\n"
            "- Never say '5 se 7 working days mein mil jayega jab product unke inspection warehouse mein receive aur approve ho jayega.' Just say: 'Warehouse inspection ke baad 5 se 7 din mein refund mil jayega.'\n"
            "- Never use awkward literal phrasing like 'lekin yah condition hai ki' or 'product ki use kiya hai' or 'apne peese' or 'bahut achi hai' or 'yadi'."
        )
    else:
        language_directive = (
            "CRITICAL LANGUAGE MIRRORING ENFORCEMENT:\n"
            "DETECTED USER LANGUAGE: PURE ENGLISH.\n"
            "MANDATORY: Reply strictly in natural, fluent English. Do not use Hindi or Arabic."
        )

    # For Hindi/Urdu, pre-clean query if in Arabic/Urdu/Devanagari script so LLM sees Roman Latin script
    clean_query = await romanize_script(query, api_key=api_key) if detected_lang == "hi" else query

    try:
        llm = ChatOpenAI(
            model=settings.effective_chat_model,
            openai_api_base=settings.effective_llm_base_url,
            openai_api_key=api_key,
            temperature=0.1
        )
        
        sys_prompt = f"""You are a warm, cool, calm, happy, and jolly AI voice assistant. You speak naturally like a helpful colleague on a phone call.

CRITICAL OPERATING RULES:

1. DYNAMIC LANGUAGE MIRRORING (STRICT SINGLE-MODE):
   - Analyze the user's input language and tone.
   - ENGLISH MODE: If the user speaks in English, reply ONLY in natural, fluent English.
   - HINGLISH MODE: If the user speaks Hindi/Urdu/Hinglish, reply ONLY in natural, everyday Colloquial Roman Hinglish (Latin alphabet).
   - ARABIC MODE: If the user speaks Arabic, reply ONLY in natural, polite Arabic script with warm Gulf/Qatari phrasing.
   - You must translate the retrieved English context into the user's language before speaking. Never reply in English to a Hindi or Arabic question.
   - NEVER mix English, Hinglish, and Arabic in a single response.

2. TEXT-TO-SPEECH (TTS) PROSE & PHONETICS:
   - CRITICAL LENGTH LIMIT: Strictly maximum 1 or 2 short, crisp, conversational sentences (total under 30 words). Never output 3 or more sentences.
   - For Hinglish, spell words with smooth phonetics: 'toota' or 'damaged' (never 'tutaa'), 'Apex' (never 'Aapex'), 'theek' (never 'thik'), 'yeh' (never 'yah'), 'paise' (never 'peese'), 'wapas' (never 'vapas'), 'agar' (never 'yadi').
   - ABSOLUTE PROHIBITION: Never use bullet points, numbered lists, bolding, italics, or Markdown formatting of any kind.
   - Speak directly as a human speaks, using smooth conversational transitions.

3. STRICT GROUNDING & HALLUCINATION PREVENTION:
   - Your factual knowledge is strictly limited to the provided RETRIEVED KNOWLEDGE BASE CONTEXT.
   - NEVER invent facts, extrapolate unstated details, or use parametric outside knowledge.
   - If the retrieved context does not contain the answer AND conversation history cannot solve it, cheerfully state that you do not have that specific information in your documents.
   - Abstain politely using the user's detected language (English, Hinglish, or Arabic).

4. CONVERSATIONAL MEMORY & FALLBACK REASONING:
   - When "NO NEW KNOWLEDGE BASE DOCUMENTS RETURNED" is indicated:
     a) First check CONVERSATION HISTORY. If the user is asking to translate, clarify, or follow up on a previous turn, fulfill the request using history.
     b) If the user is asking a brand-new factual question not present in context or history, execute Rule 3 (polite abstention).

5. CITATION TRACKING:
   - For factual claims drawn from retrieved context, append internal tracking tags like [chunk_id] at the end of the sentence. (These are stripped before voice synthesis).

{language_directive}

Retrieved Knowledge Base Context:
{context_str}"""

        prompt_messages = [SystemMessage(content=sys_prompt)]
        history_msgs = state.get("messages", [])
        for m in history_msgs[-6:]:
            prompt_messages.append(m)
        if not history_msgs or history_msgs[-1].content != clean_query:
            prompt_messages.append(HumanMessage(content=clean_query))

        response = await llm.ainvoke(prompt_messages)
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        # Fail-safe romanization check strictly for Hindi/Hinglish mode
        if detected_lang == "hi" and answer_text and re.search(r'[\u0600-\u06FF\u0900-\u097F]', answer_text):
            answer_text = await romanize_script(answer_text, api_key=api_key)
    except Exception as e:
        logger.warning(f"Answer generation primary LLM call failed ({e}), checking fallback provider...")
        answer_text = None
        if settings.CHEAPER_INFERENCE_API_KEY and "openrouter" in str(settings.effective_llm_base_url).lower():
            try:
                fallback_llm = ChatOpenAI(
                    model=settings.CHEAPER_INFERENCE_MODEL or "deepseek-v4-flash-0731",
                    openai_api_base=settings.CHEAPER_INFERENCE_BASE_URL,
                    openai_api_key=settings.CHEAPER_INFERENCE_API_KEY,
                    temperature=0.0
                )
                fb_res = await fallback_llm.ainvoke(prompt_messages)
                answer_text = fb_res.content if hasattr(fb_res, "content") else str(fb_res)
                logger.info("Fallback LLM call to CheaperInference succeeded.")
            except Exception as e2:
                logger.warning(f"Fallback LLM call also failed ({e2})")

        if not answer_text:
            if detected_lang == "ar":
                answer_text = "عذراً، لم أجد معلومات محددة بخصوص ذلك في قاعدة بياناتنا."
            elif detected_lang == "hi":
                answer_text = "Maaf kijiye, mujhe is baare mein hamari knowledge base mein koi specific jankari nahi mili."
            else:
                answer_text = FALLBACK_STRICT_RESPONSE
    
    citations = [
        {
            "chunk_id": str(doc.get("id", "chunk")),
            "source_file": str(doc.get("source_file", "Knowledge Base")),
            "text": str(doc.get("text", ""))
        }
        for doc in retrieved_docs
    ]
    
    return {
        "final_answer": answer_text,
        "messages": [AIMessage(content=answer_text)],
        "citations": citations,
        "detected_language": detected_lang
    }
