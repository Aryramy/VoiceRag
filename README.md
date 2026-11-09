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

## Dependencies Installed After Cloning (In Simple Words)

Because `bin/`, `.venv/`, `frontend/node_modules/`, and `.env` files are excluded from Git (`.gitignore`), you need to install these 4 pieces once after cloning the repository on Windows:

### 1. LiveKit Local Server (`.\bin\livekit-server.exe`)
- **What it is**: A single standalone executable (`v1.13.11`) that acts as the real-time WebRTC audio server on port `7880`, connecting your browser microphone/speaker to the AI agent.
- **Extra dependencies**: **None.** In `--dev` mode it runs standalone with no Docker or Redis required.
- **How to install**:
  ```powershell
  # Download livekit_1.13.11_windows_amd64.zip from https://github.com/livekit/livekit/releases
  # Extract livekit-server.exe into .\bin\livekit-server.exe
  New-Item -ItemType Directory -Path ".\bin" -Force | Out-Null
  Invoke-WebRequest -Uri "https://github.com/livekit/livekit/releases/download/v1.13.11/livekit_1.13.11_windows_amd64.zip" -OutFile "$env:TEMP\livekit.zip"
  Expand-Archive -Path "$env:TEMP\livekit.zip" -DestinationPath "$env:TEMP\livekit_extract" -Force
  Move-Item "$env:TEMP\livekit_extract\livekit-server.exe" ".\bin\livekit-server.exe" -Force
  ```

### 2. Python Virtual Environment (`.venv`) & Backend Packages (`requirements.txt`)
- **What it is**: An isolated Python `3.12` environment in `.venv\` holding all backend AI, database, and voice packages so they do not conflict with the rest of your PC.
- **What gets installed inside `.venv`**:
  - **Web Server (`fastapi`, `uvicorn[standard]`, `python-multipart`)**: Runs the local backend API on port `8000` for health checks, file uploads, text chat, and LiveKit room tokens.
  - **AI Brain & Memory (`langgraph`, `langgraph-checkpoint-sqlite`, `langchain-core`, `langchain-openai`)**: Runs the 3-step RAG workflow (rewrite question -> search documents -> generate grounded answer) and keeps conversation memory.
  - **Local Vector Database (`qdrant-client`)**: Stores and searches your document embeddings locally on disk in `./qdrant_storage` (no Docker needed).
  - **Voice Agent (`livekit-agents`, `livekit-api`, `livekit-plugins-openai`, `livekit-plugins-silero`, `numpy`)**: Connects the Python worker to LiveKit, detects speech/interruptions locally with Silero VAD, transcribes speech (STT), and converts answers to audio (TTS).
  - **Storage, PDFs, Config & Tests (`aiosqlite`, `pypdf`, `pydantic`, `pydantic-settings`, `python-dotenv`, `requests`, `httpx`, `pytest`)**: Saves chat history in `app.db`, reads uploaded `.pdf` files, loads `.env` keys, and runs `start_dev.py`'s pre-flight tests.
- **How to install** (run line by line in PowerShell from the project root):
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  ```

### 3. Frontend Web Packages (`frontend\node_modules`)
- **What it is**: All JavaScript/React libraries needed to run the Next.js 14 voice + chat web UI on `http://127.0.0.1:3000/`.
- **What gets installed**:
  - **UI Framework (`next`, `react`, `react-dom`)**: Serves the browser interface on port `3000`.
  - **LiveKit Browser SDK (`livekit-client`, `@livekit/components-react`, `@livekit/components-styles`, `livekit-server-sdk`)**: Connects your browser mic/audio to LiveKit, renders the audio visualizer bars, and creates join tokens.
  - **UI Helpers (`lucide-react`, `react-dropzone`, `tailwindcss`, `clsx`, `tailwind-merge`, `typescript`)**: Icons, drag-and-drop document upload UI, and styling.
- **How to install** *(use `npm.cmd` in PowerShell so `npm.ps1` is not blocked by Windows script signing)*:
  ```powershell
  npm.cmd --prefix .\frontend install
  ```

### 4. Environment Files (`.env` and `frontend\.env.local`)
- **What they are**: Local config files holding your AI API keys and local server URLs.
- **How to create them**:
  ```powershell
  Copy-Item .env.example .env
  Copy-Item frontend\.env.example frontend\.env.local
  ```
  Then open `.env` and fill in your `OPENROUTER_API_KEY` and/or `CHEAPER_INFERENCE_API_KEY`. (`frontend\.env.local` already works out of the box with `ws://127.0.0.1:7880`, `devkey`/`secret`, and `http://127.0.0.1:8000`.)

---

## Line-by-Line Startup Guide (`http://127.0.0.1:3000/`)

Once the dependencies above are installed, follow these exact lines whenever you want to start the project:

### Step 1 — Terminal 1: Start LiveKit Server
```powershell
.\bin\livekit-server.exe --dev
```
Wait until it says it is running in development mode on port `7880`. Leave this terminal open.

### Step 2 — Terminal 2: Activate Python Environment
```powershell
.\.venv\Scripts\activate
```

### Step 3 — (First Time / New Files Only) Index Knowledge Base
If `./qdrant_storage` is empty or you added new files to `data/knowledge_base/`, run this **before** starting `start_dev.py` (embedded Qdrant allows only one process to lock `./qdrant_storage` at a time):
```powershell
python -m app.core.ingest
```
*(Once the app is running, you can also upload PDF/TXT/MD files directly from the web UI instead.)*

### Step 4 — Terminal 2: Start Backend, Voice Worker, and Frontend
```powershell
python start_dev.py
```
This script automatically checks `.venv`, verifies LiveKit is listening on port `7880`, runs the pre-flight tests (`pytest tests/test_phase5.py -v`), and launches all 3 services:
- **FastAPI Backend**: `http://localhost:8000`
- **LiveKit Voice Agent Worker**: connected to `ws://127.0.0.1:7880`
- **Next.js Frontend UI**: `http://127.0.0.1:3000`

### Step 5 — Open the App
Open [http://127.0.0.1:3000/](http://127.0.0.1:3000/) in your browser, click **"Start Voice Call"** (or use the text chat pane), and allow microphone access.

---

### Alternative: Manual Startup (Without `start_dev.py`)
If you prefer starting each service in its own terminal instead of `python start_dev.py`:
1. Terminal 1: `.\bin\livekit-server.exe --dev`
2. Terminal 2: `.\.venv\Scripts\uvicorn.exe app.main:app --host 127.0.0.1 --port 8000 --reload`
3. Terminal 3: `.\.venv\Scripts\python.exe -m app.worker.livekit_worker dev`
4. Terminal 4: `npm.cmd --prefix .\frontend run dev`

---

## Running Verification Tests

Run the pre-flight API, upload, chat, and citation tests:
```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase5.py -v
```

Run the LangGraph state machine test suite:
```powershell
.\.venv\Scripts\python.exe -m tests.test_pipeline
```

Run the FastAPI token and health gateway tests:
```powershell
.\.venv\Scripts\python.exe -m tests.test_api
```

