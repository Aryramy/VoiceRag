# Master Project Blueprint: Production Agentic RAG Voice Assistant

This document serves as the single source of truth for the repository structure, state machine graph topology, API cascade, runtime execution rules, and frontend WebRTC interface.

---

# Part 1: Backend Architecture (Python / LangGraph / LiveKit / Qdrant)

**System Persona & Architectural Mandate**: An expert AI engineer building a low-latency, WebRTC-enabled, Agentic RAG Voice AI system.

---

## 1. System Constraints & Environmental Rules

1. **User-Space Execution Only**: Absolute prohibition of Docker, `sudo`, `systemctl`, or system-level binary dependencies. Everything must execute inside a Python virtual environment (`python -m venv .venv`) and Node.js (`pnpm` or `npm`).
2. **Qdrant Vector Database**: Use `qdrant-client` in local disk-persisted embedded mode (`path="./qdrant_storage"`) or in-memory mode to eliminate Docker background container dependencies.
3. **Cascaded Cloud Inference Protocol**:
   * **STT**: CheaperInference or OpenRouter OpenAI-compatible REST/WebSocket endpoint using `Whisper Large v3 Turbo` or `Voxtral`.
   * **LLM Reasoning & Orchestration**: OpenRouter API (`Llama 3.1 70B / 8B` or `Gemini 2.0 Flash`).
   * **TTS**: OpenRouter or CheaperInference streaming audio endpoints (`ElevenLabs v4 Turbo` or `Gemini Flash TTS`).
4. **Latency Target**: Strict end-to-end first-audio budget of **600–800 ms** (hard ceiling 1,200 ms).

---

## 2. Project Directory Structure

```
rag-voice-assistant/
├── .env.example
├── requirements.txt
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI server entry point & HTTP/WS routes
│   ├── config.py                # Pydantic Settings for API keys & parameters
│   ├── core/
│   │   ├── __init__.py
│   │   ├── qdrant_client.py     # Embedded Qdrant local initialization & ingestion
│   │   └── state.py             # LangGraph TypedDict state definitions & reducers
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── nodes.py             # Query rewriter, vector retrieval, and answer nodes
│   │   ├── edges.py             # Conditional router & fallback edge logic
│   │   └── workflow.py          # StateGraph compilation with SQLite checkpointer
│   └── worker/
│       ├── __init__.py
│       ├── livekit_worker.py    # LiveKit AgentServer entrypoint & RTC session logic
│       └── audio_pipeline.py    # STT/TTS cascade wrappers & VAD event handlers
├── data/
│   └── knowledge_base/          # Source documents (PDF, Markdown, TXT) for ingestion
└── frontend/                    # Next.js 14+ App Router WebRTC Interface
    ├── package.json
    ├── src/
    │   ├── app/
    │   │   └── page.tsx         # Voice UI with LiveKit Room components
    │   └── hooks/
    │       └── useLiveKitVoice.ts
```

---

## 3. Dependency Specification (`requirements.txt`)

```
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
pydantic>=2.6.0
pydantic-settings>=2.2.0
langgraph>=0.0.30
langchain-core>=0.1.30
langchain-openai>=0.0.8
qdrant-client>=1.8.0
livekit-agents>=0.8.0
livekit-api>=0.6.0
numpy>=1.26.0
requests>=2.31.0
python-dotenv>=1.0.1
```

---

## 4. LangGraph State Machine Specification

### 4.1 State Schema Definition (`app/core/state.py`)

The state uses `Annotated` with `add_messages` reducer to accumulate conversation history across turns.

```python
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgenticVoiceState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    raw_query: str
    rewritten_query: str
    retrieved_context: list[str]
    is_context_relevant: bool
    final_answer: str
```

### 4.2 Node Logic & Fallback Guard (`app/graph/nodes.py`)

1. **rewrite_query**: Converts noisy conversational speech transcripts (e.g., "uh, what about the return policy thing?") into concise vector search queries using conversation context.
2. **retrieve**: Searches local Qdrant index (`path="./qdrant_storage"`). If max similarity score is below `0.70` or results are empty, flags `is_context_relevant = False`.
3. **generate_answer**: Formulates concise 1–2 sentence spoken responses with claim citation markers (`[chunk_id]`) for evaluation tracing, which are stripped before TTS output.

```python
import re
from typing import Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.core.state import AgenticVoiceState
from app.core.qdrant_client import search_qdrant

FALLBACK_STRICT_RESPONSE = (
    "I'm sorry, but I couldn't find specific information regarding that in my knowledge base."
)

async def rewrite_query_node(state: AgenticVoiceState, config: Dict[str, Any]) -> Dict[str, Any]:
    llm = ChatOpenAI(
        model="meta-llama/llama-3.1-8b-instruct",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=config["configurable"]["openrouter_key"],
        temperature=0.0
    )
    raw = state["raw_query"]
    history = state.get("messages", [])[-4:]
    
    prompt = f"Rewrite the following spoken user query into a clean, precise vector database search query. Do not add conversational fluff.\nContext: {history}\nUser query: '{raw}'"
    res = await llm.ainvoke([HumanMessage(content=prompt)])
    return {"rewritten_query": res.content.strip()}

async def retrieve_node(state: AgenticVoiceState, config: Dict[str, Any]) -> Dict[str, Any]:
    query = state["rewritten_query"]
    results, max_score = await search_qdrant(query=query, top_k=3, min_score_threshold=0.70)
    
    if not results or max_score < 0.70:
        return {"retrieved_context": [], "is_context_relevant": False}
    
    return {"retrieved_context": results, "is_context_relevant": True}

async def generate_answer_node(state: AgenticVoiceState, config: Dict[str, Any]) -> Dict[str, Any]:
    if not state.get("is_context_relevant", False):
        return {"final_answer": FALLBACK_STRICT_RESPONSE, "messages": [AIMessage(content=FALLBACK_STRICT_RESPONSE)]}

    llm = ChatOpenAI(
        model="meta-llama/llama-3.1-70b-instruct",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=config["configurable"]["openrouter_key"],
        temperature=0.2
    )
    
    context_str = "\n".join([f"[{doc['id']}] {doc['text']}" for doc in state["retrieved_context"]])
    query = state["raw_query"]
    
    sys_prompt = f"""You are a concise, grounded voice AI assistant. 
Answer the query using ONLY the context below. 
Rules:
1. Max 1-2 short sentences for natural spoken delivery.
2. Append citation tags [chunk_id] for every claim.
3. If context is missing, say you don't know.
4. If the user asks for Hindi or Urdu, you MUST translate your entire answer (including retrieved context) into colloquial Hinglish (a natural conversational mix of simple Hindi/Urdu and common English words) before speaking. Do not use overly formal or pure Hindi/Urdu words.

Retrieved Context:
{context_str}"""

    response = await llm.ainvoke([SystemMessage(content=sys_prompt), HumanMessage(content=query)])
    return {"final_answer": response.content, "messages": [AIMessage(content=response.content)]}
```

### 4.3 StateGraph Construction (`app/graph/workflow.py`)

```python
from langgraph.graph import StateGraph, END
from app.core.state import AgenticVoiceState
from app.graph.nodes import rewrite_query_node, retrieve_node, generate_answer_node

def route_after_retrieval(state: AgenticVoiceState) -> str:
    # If no context found, skip generation and jump directly to fallback completion
    if not state.get("is_context_relevant", False):
        return "generate_answer"
    return "generate_answer"

builder = StateGraph(AgenticVoiceState)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate_answer", generate_answer_node)

builder.set_entry_point("rewrite_query")
builder.add_edge("rewrite_query", "retrieve")
builder.add_conditional_edges("retrieve", route_after_retrieval)
builder.add_edge("generate_answer", END)

graph = builder.compile()
```

---

## 5. LiveKit Audio Routing & Worker Logic (`app/worker/livekit_worker.py`)

Manages WebRTC streams via `livekit-agents`, applies VAD barge-in handling, routes speech to ASR, triggers parallel LangGraph retrieval, and streams synthesized audio back.

```python
import asyncio
import re
from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli, inference
from app.graph.workflow import graph

CITATION_REGEX = re.compile(r"\[chunk_\d+\]")

def strip_citations_for_tts(text: str) -> str:
    """Strips citation markers like [chunk_1] before sending text to Speech Synthesis."""
    return CITATION_REGEX.sub("", text).strip()

server = AgentServer()

@server.rtc_session(agent_name="agentic-voice-rag")
async def entrypoint(ctx: JobContext):
    # Configure low-latency pipeline endpoints
    session = AgentSession(
        vad=inference.VAD(), # Real-time Silero VAD for barge-in detection
        stt=inference.STT(
            model="whisper-large-v3-turbo",
            base_url="https://openrouter.ai/api/v1"
        ),
        llm=inference.LLM(
            model="meta-llama/llama-3.1-70b-instruct",
            base_url="https://openrouter.ai/api/v1"
        ),
        tts=inference.TTS(
            model="eleven-turbo-v4",
            base_url="https://openrouter.ai/api/v1"
        )
    )

    agent = Agent(instructions="You are an enterprise voice assistant.")
    await session.start(room=ctx.room, agent=agent)

    @session.on("user_speech_committed")
    async def on_user_speech(ev):
        user_transcript = ev.transcript
        
        # Parallel Execution: Execute LangGraph state machine while warming up LLM stream
        config = {"configurable": {"thread_id": ctx.room.name, "openrouter_key": "YOUR_OPENROUTER_KEY"}}
        initial_state = {"raw_query": user_transcript, "messages": []}
        
        result = await graph.ainvoke(initial_state, config=config)
        raw_answer = result.get("final_answer", "")
        
        # Clean response for TTS engine
        spoken_text = strip_citations_for_tts(raw_answer)
        await session.say(spoken_text)

if __name__ == "__main__":
    cli.run_app(server)
```

---

## 6. User-Space Qdrant Vector Engine (`app/core/qdrant_client.py`)

No Docker daemon is required. Persistent state is maintained via a local user-space folder.

```python
import os
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_openai import OpenAIEmbeddings

STORAGE_PATH = os.path.abspath("./qdrant_storage")
COLLECTION_NAME = "enterprise_kb"

client = AsyncQdrantClient(path=STORAGE_PATH)
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_base="https://openrouter.ai/api/v1"
)

async def init_qdrant():
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

async def search_qdrant(query: str, top_k: int = 3, min_score_threshold: float = 0.70):
    await init_qdrant()
    vector = await embeddings.aembed_query(query)
    
    search_result = await client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=top_k
    )
    
    if not search_result:
        return [], 0.0
        
    max_score = max([hit.score for hit in search_result])
    docs = [{"id": f"chunk_{hit.id}", "text": hit.payload["text"]} for hit in search_result if hit.score >= min_score_threshold]
    
    return docs, max_score
```

---

## 7. FastAPI API Server (`app/main.py`)

Provides session token generation for LiveKit WebRTC authentication alongside REST management endpoints.

```python
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from livekit.api import AccessToken, VideoGrants
import os

app = FastAPI(title="Agentic RAG Voice AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "ok", "environment": "user-space-venv"}

@app.get("/api/get-join-token")
async def get_join_token(room_name: str, participant_name: str):
    api_key = os.getenv("LIVEKIT_API_KEY", "devkey")
    api_secret = os.getenv("LIVEKIT_API_SECRET", "secret")
    
    grant = VideoGrants(room_join=True, room=room_name)
    token = AccessToken(api_key, api_secret, grants=grant, identity=participant_name)
    
    return {"token": token.to_jwt(), "url": os.getenv("LIVEKIT_URL", "ws://localhost:7880")}
```

---

## 8. Critical Antigravity Validation Checklist

* [x] **No Docker references**: Verify `qdrant-client` uses `path="..."` local storage.
* [x] **Barge-in / Interrupt Handling**: Ensure Silero VAD is bound to the `AgentSession` so that active user audio cancels ongoing TTS streaming.
* [x] **Citation Stripping**: Confirm citation regex (`\[chunk_\d+\]`) strips markers before passing text to the TTS engine so the agent does not speak raw bracket markers.
* [x] **Strict Fallbacks**: Ensure queries without relevant context trigger the explicit fallback string rather than generating hallucinations.

---

# Part 2: Next.js 14 WebRTC Voice UI Frontend

**System Persona & Architectural Mandate**: Expert Full-Stack UI/UX Engineer specializing in real-time WebRTC audio interfaces, Next.js 14 (App Router), and Tailwind CSS.

---

## 1. Environment Variables & Package Specification

### 1.1 Environment Configuration (`.env.local`)

```env
# LiveKit Server Connection (Public for client WebRTC transport)
NEXT_PUBLIC_LIVEKIT_URL=ws://localhost:7880

# LiveKit Admin Credentials (Server-only for API token signing)
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

### 1.2 Package Dependencies (`package.json`)

```json
{
  "name": "agentic-voice-ui",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "@livekit/components-react": "^2.6.0",
    "@livekit/components-styles": "^1.1.0",
    "clsx": "^2.1.0",
    "livekit-client": "^2.5.0",
    "livekit-server-sdk": "^2.6.0",
    "lucide-react": "^0.350.0",
    "next": "14.2.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "tailwind-merge": "^2.2.0"
  },
  "devDependencies": {
    "@types/node": "^20.0.0",
    "@types/react": "^18.2.0",
    "@types/react-dom": "^18.2.0",
    "autoprefixer": "^10.4.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.4.0"
  }
}
```

---

## 2. Directory Layout (Next.js 14 App Router)

```
frontend/
├── .env.local
├── package.json
├── tailwind.config.ts
├── src/
│   ├── app/
│   │   ├── layout.tsx              # Root layout importing LiveKit CSS
│   │   ├── page.tsx                # Main Voice Assistant UI
│   │   └── api/
│   │       └── token/
│   │           └── route.ts        # Server-side JWT token generation
│   ├── components/
│   │   ├── VoiceVisualizer.tsx     # Audio spectrum bar visualizer & state indicator
│   │   ├── StatusBadge.tsx         # Listening / Thinking / Speaking badge
│   │   └── Controls.tsx            # Custom Mute/Disconnect button panel
│   └── hooks/
│       └── useLiveKitSession.ts    # Custom hook for fetching tokens & managing state
```

---

## 3. Server-Side Token Route (`src/app/api/token/route.ts`)

```typescript
import { NextRequest, NextResponse } from "next/server";
import { AccessToken } from "livekit-server-sdk";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const room = searchParams.get("room") || "voice-room-1";
    const username = searchParams.get("username") || `user-${Math.floor(Math.random() * 10000)}`;

    const apiKey = process.env.LIVEKIT_API_KEY;
    const apiSecret = process.env.LIVEKIT_API_SECRET;
    const wsUrl = process.env.NEXT_PUBLIC_LIVEKIT_URL;

    if (!apiKey || !apiSecret || !wsUrl) {
      return NextResponse.json(
        { error: "Server missing LiveKit environment credentials" },
        { status: 500 }
      );
    }

    const at = new AccessToken(apiKey, apiSecret, {
      identity: username,
      ttl: "1h",
    });

    at.addGrant({
      room,
      roomJoin: true,
      canPublish: true,
      canSubscribe: true,
    });

    const token = await at.toJwt();

    return NextResponse.json({
      token,
      url: wsUrl,
    });
  } catch (error) {
    console.error("Error generating LiveKit token:", error);
    return NextResponse.json(
      { error: "Failed to create connection token" },
      { status: 500 }
    );
  }
}
```

---

## 4. Audio Visualizer & State Indicator (`src/components/VoiceVisualizer.tsx`)

```tsx
"use client";

import React from "react";
import {
  useVoiceAssistant,
  BarVisualizer,
} from "@livekit/components-react";
import { Mic, Cpu, Volume2, Radio } from "lucide-react";

export function VoiceVisualizer() {
  const { state, audioTrack, agent } = useVoiceAssistant();

  // Map LiveKit state to human-readable status & styling
  const getStatusConfig = () => {
    switch (state) {
      case "listening":
        return {
          label: "Listening...",
          color: "bg-emerald-500/10 text-emerald-400 border-emerald-500/30",
          icon: Mic,
          pulse: true,
        };
      case "thinking":
        return {
          label: "Thinking & Searching RAG...",
          color: "bg-amber-500/10 text-amber-400 border-amber-500/30",
          icon: Cpu,
          pulse: true,
        };
      case "speaking":
        return {
          label: "Speaking",
          color: "bg-indigo-500/10 text-indigo-400 border-indigo-500/30",
          icon: Volume2,
          pulse: false,
        };
      default:
        return {
          label: "Connecting...",
          color: "bg-slate-500/10 text-slate-400 border-slate-500/30",
          icon: Radio,
          pulse: false,
        };
    }
  };

  const status = getStatusConfig();
  const StatusIcon = status.icon;

  return (
    <div className="flex flex-col items-center justify-center p-8 space-y-6">
      {/* State Badge */}
      <div className={`flex items-center gap-2 px-4 py-1.5 rounded-full border text-sm font-medium ${status.color} ${status.pulse ? "animate-pulse" : ""}`}>
        <StatusIcon className="w-4 h-4" />
        <span>{status.label}</span>
      </div>

      {/* Dynamic Audio Spectrum Visualizer */}
      <div className="w-full max-w-md h-32 flex items-center justify-center bg-slate-900/50 rounded-2xl border border-slate-800 p-4">
        {audioTrack ? (
          <BarVisualizer
            state={state}
            barCount={24}
            trackRef={audioTrack}
            className="w-full h-full flex items-center justify-center gap-1.5"
          />
        ) : (
          <div className="text-slate-500 text-sm flex items-center gap-2">
            <Radio className="w-4 h-4 animate-spin" />
            <span>Waiting for audio stream...</span>
          </div>
        )}
      </div>

      {/* Agent Info */}
      <div className="flex items-center gap-2 text-xs text-slate-400">
        <span className="w-2 h-2 rounded-full bg-emerald-500 animate-ping" />
        <span>Connected to LiveKit WebRTC Agent</span>
      </div>
    </div>
  );
}
```

---

## 5. Main UI Page (`src/app/page.tsx`) & Components

The main page serves as the LiveKit `LiveKitRoom` container wrapping audio controls, `RoomAudioRenderer`, and `VoiceVisualizer`:
- Uses `@livekit/components-react` components (`LiveKitRoom`, `RoomAudioRenderer`, `VoiceAssistantControlBar` / custom `Controls`).
- Manages connection lifecycle (Connect, Disconnect, Error recovery, Mic permissions).
- Renders conversational feedback, status indicators, and low-latency audio stream visualizer.

