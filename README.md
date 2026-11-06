# Production Agentic RAG Voice Assistant

A low-latency, WebRTC-enabled, Agentic RAG Voice Assistant powered by **LangGraph**, **LiveKit**, **Qdrant (Embedded Local Storage)**, and **Next.js 14**.

---

## Architecture Overview

```
[ Next.js 14 Voice UI (WebRTC) ] 
               ▲ 
               │ WebRTC Audio Stream (Opus)
               ▼
[ LiveKit Server / Cloud Gateway ]
               ▲
               │ RTC Session & Audio Hooks
               ▼
[ LiveKit Agent Worker (Python) ]
  ├── Silero VAD (Barge-in / Interrupt detection)
  ├── STT: Whisper Large v3 Turbo (OpenAI-compatible)
  ├── LangGraph State Machine:
  │     ├── 1. Rewrite Query (Llama 3.1 8B Instruct)
  │     ├── 2. Embedded Qdrant Retrieval (Cosine >= 0.45 threshold)
  │     └── 3. Grounded Answer Generation (Llama 3.1 70B Instruct + [chunk_id] tags)
  └── TTS: ElevenLabs v4 Turbo (with bracketed citation stripping)
```

---

## Environmental Constraints & User-Space Mandate

- **Zero Docker**: Qdrant runs in embedded user-space local file storage (`./qdrant_storage`).
- **Python Virtual Environment**: Runs entirely inside `.venv`.
- **Node.js**: Runs natively via `npm` or `pnpm`.
- **Strict Grounding Guard**: Any query with maximum vector similarity below `0.45` immediately returns a strict fallback string instead of hallucinating.
- **Citation Stripping**: Spoken TTS responses never vocalize markdown or bracketed markers (e.g., `[chunk_xxx]`).

---

## Getting Started

### 1. Backend Setup

1. **Activate virtual environment**:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```

2. **Ingest Knowledge Base**:
   ```powershell
   .venv\Scripts\python.exe -m app.core.ingest
   ```

3. **Start FastAPI Gateway Server** (for JWT token generation & health checks):
   ```powershell
   .venv\Scripts\uvicorn.exe app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. **Start LiveKit Agent Worker**:
   ```powershell
   .venv\Scripts\python.exe -m app.worker.livekit_worker start
   ```
   *(Or for local development: `.venv\Scripts\python.exe -m app.worker.livekit_worker dev`)*

---

### 2. Frontend Setup

1. **Navigate to `frontend/`**:
   ```powershell
   cd frontend
   npm run dev
   ```

2. Open [http://localhost:3000](http://localhost:3000) in your browser.
3. Click **"Start Voice Call"** and grant microphone permissions to speak directly to the Agentic RAG assistant.

---

## Running Verification Tests

Run the LangGraph state machine test suite:
```powershell
.venv\Scripts\python.exe -m tests.test_pipeline
```

Run the FastAPI token and health gateway tests:
```powershell
.venv\Scripts\python.exe -m tests.test_api
```

