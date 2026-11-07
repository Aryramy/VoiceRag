import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants
from app.config import settings
from app.core.database import init_db
from app.api.upload import router as upload_router
from app.api.chat import router as chat_router

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
    await init_db()

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
    from app.core.qdrant_client import search_qdrant
    query = payload.get("query", "")
    top_k = payload.get("top_k", 3)
    min_score = payload.get("min_score_threshold", 0.45)
    docs, max_score = await search_qdrant(query=query, top_k=top_k, min_score_threshold=min_score)
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
