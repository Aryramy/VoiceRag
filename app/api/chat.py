import json
import os
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from app.graph.workflow import graph
from app.core.database import save_chat_message, get_chat_history, list_sessions
from app.config import settings

logger = logging.getLogger("chat_api")
router = APIRouter(prefix="/api", tags=["chat"])

class ChatRequest(BaseModel):
    session_id: str
    message: str

class CitationItem(BaseModel):
    chunk_id: str
    source_file: str
    text: str

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[CitationItem]

@router.post("/chat", response_model=ChatResponse)
async def handle_chat(req: ChatRequest):
    try:
        # 1. Save incoming user message to SQLite
        await save_chat_message(req.session_id, "user", req.message)

        # 2. Invoke LangGraph pipeline
        api_key = settings.effective_llm_api_key
        config = {
            "configurable": {
                "thread_id": req.session_id,
                "api_key": api_key,
                "openrouter_key": api_key
            }
        }
        initial_state = {"raw_query": req.message, "messages": []}

        result = await graph.ainvoke(initial_state, config=config)

        final_answer = result.get("final_answer", "")
        retrieved_docs = result.get("retrieved_context", [])
        raw_citations = result.get("citations", [])

        # 3. Format citations for UI rendering
        citations = []
        if raw_citations:
            for c in raw_citations:
                citations.append(
                    CitationItem(
                        chunk_id=str(c.get("chunk_id", "chunk")),
                        source_file=str(c.get("source_file", "Knowledge Base")),
                        text=str(c.get("text", ""))
                    )
                )
        else:
            for doc in retrieved_docs:
                citations.append(
                    CitationItem(
                        chunk_id=str(doc.get("id", "chunk")),
                        source_file=str(doc.get("source_file", "Knowledge Base")),
                        text=str(doc.get("text", ""))
                    )
                )

        citations_dict = [c.model_dump() if hasattr(c, "model_dump") else c.dict() for c in citations]
        citations_json = json.dumps(citations_dict)
        
        # 4. Save agent response to SQLite
        await save_chat_message(req.session_id, "agent", final_answer, citations_json)

        return ChatResponse(
            session_id=req.session_id,
            answer=final_answer,
            citations=citations
        )

    except Exception as e:
        logger.error(f"Chat processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@router.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    history = await get_chat_history(session_id)
    return {"session_id": session_id, "history": history}

@router.get("/sessions")
async def get_sessions():
    sessions = await list_sessions()
    return {"sessions": sessions}
