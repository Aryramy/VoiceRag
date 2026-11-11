import logging
import os
import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants
from app.config import settings
from app.core.database import init_db, close_db_connection
from app.core.qdrant_client import search_qdrant, init_qdrant, get_embeddings
from app.graph.nodes import warm_llm_clients
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router

logger = logging.getLogger("fastapi_main")

app = FastAPI(
    title="Agentic RAG Voice AI Backend",
    description="Low-latency WebRTC Voice AI with LangGraph and embedded Qdrant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup():
    t0 = time.perf_counter()
    await init_db()
    await init_qdrant()
    get_embeddings()
    warm_llm_clients()
    logger.info(f"[PERF][fastapi_startup] Prewarmed SQLite WAL, Qdrant, Embeddings & LLMs in {(time.perf_counter() - t0)*1000:.1f}ms")


@app.on_event("shutdown")
async def on_shutdown():
    await close_db_connection()


# Register Phase 5 upload and multimodal chat routers
app.include_router(upload_router)
app.include_router(chat_router)


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "environment": "user-space-venv",
        "qdrant_storage": settings.QDRANT_STORAGE_PATH,
        "collection": settings.QDRANT_COLLECTION_NAME
    }


@app.post("/api/internal-search")
async def internal_search_endpoint(payload: dict):
    t0 = time.perf_counter()
    query = payload.get("query", "")
    top_k = payload.get("top_k", 3)
    min_score = payload.get("min_score_threshold", settings.FALLBACK_SCORE_THRESHOLD)
    docs, max_score = await search_qdrant(query=query, top_k=top_k, min_score_threshold=min_score)
    logger.info(
        f"[PERF][internal_search] total={(time.perf_counter() - t0)*1000:.2f}ms "
        f"hits={len(docs)} max_score={max_score:.3f}"
    )
    return {"docs": docs, "max_score": max_score}

@app.get("/api/get-join-token")
async def get_join_token(
    room_name: str = Query("voice-room-1", description="LiveKit Room Name"),
    participant_name: str = Query("user-web", description="Participant Identity")
):
    api_key = settings.LIVEKIT_API_KEY or "devkey"
    api_secret = settings.LIVEKIT_API_SECRET or "secret"
    ws_url = settings.LIVEKIT_URL or "ws://127.0.0.1:7880"
    
    try:
        grant = VideoGrants(room_join=True, room=room_name)
        token = (
            AccessToken(api_key, api_secret)
            .with_identity(participant_name)
            .with_grants(grant)
        )
        return {
            "token": token.to_jwt(),
            "url": ws_url,
            "room": room_name,
            "participant": participant_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate LiveKit token: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
