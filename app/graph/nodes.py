import re
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.core.state import AgenticVoiceState
from app.core.qdrant_client import search_qdrant
from app.config import settings

FALLBACK_STRICT_RESPONSE = settings.FALLBACK_STRICT_RESPONSE

async def rewrite_query_node(state: AgenticVoiceState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    api_key = (
        config.get("configurable", {}).get("openrouter_key")
        if config and "configurable" in config
        else settings.OPENROUTER_API_KEY
    ) or settings.OPENROUTER_API_KEY

    llm = ChatOpenAI(
        model=settings.OPENROUTER_FAST_MODEL,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        openai_api_key=api_key,
        temperature=0.0
    )
    raw = state.get("raw_query", "")
    history = state.get("messages", [])[-4:]
    
    prompt = (
        f"You are a query preprocessor for a semantic vector database.\n"
        f"Rewrite the following spoken user query into a concise, natural semantic search phrase optimized for text embedding retrieval.\n"
        f"Strict rules:\n"
        f"1. DO NOT write SQL, Cypher, code, operators, or boolean logic.\n"
        f"2. DO NOT add conversational filler, quotes, or punctuation.\n"
        f"3. Return ONLY the plain text search phrase.\n\n"
        f"Conversation History: {history}\n"
        f"User Query: {raw}"
    )
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    rewritten = res.content.strip()
    return {"rewritten_query": rewritten if rewritten else raw}

async def retrieve_node(state: AgenticVoiceState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    query = state.get("rewritten_query") or state.get("raw_query", "")
    results, max_score = await search_qdrant(
        query=query,
        top_k=3,
        min_score_threshold=settings.FALLBACK_SCORE_THRESHOLD
    )
    
    if not results or max_score < settings.FALLBACK_SCORE_THRESHOLD:
        return {"retrieved_context": [], "is_context_relevant": False}
    
    return {"retrieved_context": results, "is_context_relevant": True}

async def generate_answer_node(state: AgenticVoiceState, config: RunnableConfig | None = None) -> Dict[str, Any]:
    if not state.get("is_context_relevant", False):
        return {
            "final_answer": FALLBACK_STRICT_RESPONSE,
            "messages": [AIMessage(content=FALLBACK_STRICT_RESPONSE)]
        }

    api_key = (
        config.get("configurable", {}).get("openrouter_key")
        if config and "configurable" in config
        else settings.OPENROUTER_API_KEY
    ) or settings.OPENROUTER_API_KEY

    llm = ChatOpenAI(
        model=settings.OPENROUTER_CHAT_MODEL,
        openai_api_base=settings.OPENROUTER_BASE_URL,
        openai_api_key=api_key,
        temperature=0.2
    )
    
    context_str = "\n".join([f"[{doc['id']}] {doc['text']}" for doc in state.get("retrieved_context", [])])
    query = state.get("raw_query", "")
    
    sys_prompt = f"""You are a concise, grounded voice AI assistant. 
Answer the query using ONLY the context below. 
Rules:
1. Max 1-2 short sentences for natural spoken delivery.
2. Append citation tags [chunk_id] for every claim (e.g. [chunk_1]).
3. If context is missing or does not answer the question, say you don't know.

Retrieved Context:
{context_str}"""

    response = await llm.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=query)])
    return {"final_answer": response.content, "messages": [AIMessage(content=response.content)]}
