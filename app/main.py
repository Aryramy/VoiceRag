import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants
from app.config import settings

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

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "environment": "user-space-venv",
        "qdrant_storage": settings.QDRANT_STORAGE_PATH,
        "collection": settings.QDRANT_COLLECTION_NAME
    }

@app.get("/api/get-join-token")
async def get_join_token(
    room_name: str = Query("voice-room-1", description="LiveKit Room Name"),
    participant_name: str = Query("user-web", description="Participant Identity")
):
    api_key = settings.LIVEKIT_API_KEY or "devkey"
    api_secret = settings.LIVEKIT_API_SECRET or "secret"
    ws_url = settings.LIVEKIT_URL or "ws://localhost:7880"
    
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

