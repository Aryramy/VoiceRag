import re
import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.core.state import AgenticVoiceState
from app.core.qdrant_client import search_qdrant
from app.config import settings

logger = logging.getLogger("graph_nodes")
FALLBACK_STRICT_RESPONSE = settings.FALLBACK_STRICT_RESPONSE

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

def detect_is_hinglish(text: str) -> bool:
    """Detects whether text contains Hindi/Urdu words or Devanagari/Urdu scripts."""
    if not text:
        return False
    if re.search(r'[\u0900-\u097F\u0600-\u06FF]', text):
        return True
    words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    return bool(words & HINGLISH_KEYWORDS)

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
            f"3. If the user query is in Hindi, Urdu, or Hinglish, map the core intent into clear English search terms (e.g. 'phone paani mein girna' -> 'phone water liquid damage warranty coverage', 'refund aane mein kitne din lagenge' -> 'refund processing days timeline') so it matches English documents in the vector store.\n"
            f"4. Return ONLY the plain text search phrase.\n\n"
            f"Conversation History: {history}\n"
            f"User Query: {raw}"
        )
        res = await llm.ainvoke([HumanMessage(content=prompt)])
        rewritten = res.content.strip() if hasattr(res, "content") else str(res).strip()
        return {"rewritten_query": rewritten if rewritten else raw}
    except Exception as e:
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
    user_is_hinglish = detect_is_hinglish(query)
    if not user_is_hinglish and state.get("messages"):
        for m in reversed(state.get("messages")[-4:]):
            if isinstance(m, HumanMessage) and detect_is_hinglish(str(m.content)):
                user_is_hinglish = True
                break

    if user_is_hinglish:
        language_directive = (
            "CRITICAL LANGUAGE MIRRORING ENFORCEMENT:\n"
            "DETECTED USER LANGUAGE: HINGLISH (Hindi/Urdu in Roman script).\n"
            "MANDATORY: The user spoke in Hindi/Urdu/Hinglish. You MUST output your final answer strictly in Colloquial Roman Hinglish (natural Hindi/Urdu written in English script).\n"
            "ABSOLUTE PROHIBITION: NEVER reply in English to this question! You must translate the retrieved English knowledge base facts into natural Roman Hinglish before speaking."
        )
    else:
        language_directive = (
            "CRITICAL LANGUAGE MIRRORING ENFORCEMENT:\n"
            "DETECTED USER LANGUAGE: PURE ENGLISH.\n"
            "MANDATORY: Reply strictly in natural, fluent English. Do not use Hindi/Hinglish."
        )

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
   - HINGLISH MODE: If the user speaks or requests Hindi/Urdu (using words like 'agar', 'kya', 'paani', 'mein', 'hai', 'achcha', 'kese', 'kab', 'kahan', 'thik he', 'aap', 'tum', 'hum', 'batlao', 'kisse', 'kun') or Devanagari/Urdu script, reply ONLY in natural, everyday Colloquial Roman Hinglish.
   - You must translate the retrieved English context into the user's language before speaking. Never reply in English to a Hindi/Hinglish question.
   - NEVER mix English and Hinglish modes in a single response.

2. TEXT-TO-SPEECH (TTS) PROSE FORMATTING:
   - Keep answers strictly to 1 or 2 short, conversational sentences.
   - ABSOLUTE PROHIBITION: Never use bullet points, numbered lists, bolding, italics, or Markdown formatting of any kind.
   - Speak directly as a human speaks, using smooth conversational transitions.

3. STRICT GROUNDING & HALLUCINATION PREVENTION:
   - Your factual knowledge is strictly limited to the provided RETRIEVED KNOWLEDGE BASE CONTEXT.
   - NEVER invent facts, extrapolate unstated details, or use parametric outside knowledge.
   - If the retrieved context does not contain the answer AND conversation history cannot solve it, cheerfully state that you do not have that specific information in your documents.
   - Abstain politely using the user's detected language (English or Hinglish).

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
        if not history_msgs or history_msgs[-1].content != query:
            prompt_messages.append(HumanMessage(content=query))

        response = await llm.ainvoke(prompt_messages)
        answer_text = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        logger.warning(f"Answer generation LLM call failed ({e}), using fallback response")
        if user_is_hinglish:
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
        "citations": citations
    }
