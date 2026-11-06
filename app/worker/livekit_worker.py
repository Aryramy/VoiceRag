import asyncio
import os
import logging
from dotenv import load_dotenv

load_dotenv()

from app.config import settings

# Populate environment variables
os.environ.setdefault("LIVEKIT_URL", settings.LIVEKIT_URL or "ws://localhost:7880")
os.environ.setdefault("LIVEKIT_API_KEY", settings.LIVEKIT_API_KEY or "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", settings.LIVEKIT_API_SECRET or "secret")

from livekit.agents import (
    AgentServer,
    JobContext,
    Agent,
    AgentSession,
    cli,
    UserInputTranscribedEvent
)
from livekit.plugins import openai, silero
from app.graph.workflow import graph
from app.worker.audio_pipeline import strip_citations_for_tts

logger = logging.getLogger("livekit_worker")
logging.basicConfig(level=logging.INFO)

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx: JobContext):
    logger.info(f"Incoming room job for room: {ctx.room.name}")
    
    # 1. Connect agent worker WebRTC transport to the room
    await ctx.connect()
    logger.info(f"Agent successfully connected to WebRTC room: {ctx.room.name}")

    # 2. Wait for user participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined room: {participant.identity}")

    # 3. Configure plugins: local Silero VAD + OpenRouter/CheaperInference STT & TTS
    vad = silero.VAD.load()
    stt = openai.STT(
        model=settings.OPENROUTER_STT_MODEL,
        base_url=settings.OPENROUTER_STT_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY
    )
    tts = openai.TTS(
        model=settings.OPENROUTER_TTS_MODEL,
        base_url=settings.OPENROUTER_TTS_BASE_URL,
        api_key=settings.OPENROUTER_API_KEY,
        voice=settings.OPENROUTER_TTS_VOICE
    )

    # Use local VAD turn detection to prevent 401 cloud barge-in timeouts
    session = AgentSession(
        vad=vad,
        stt=stt,
        tts=tts,
        turn_detection="vad"
    )

    agent = Agent(instructions="You are an enterprise voice assistant.")
    await session.start(room=ctx.room, agent=agent)

    # 4. Greet participant to publish initial audio track & activate visualizer
    try:
        await session.say("Hello! How can I help you today?")
    except Exception as e:
        logger.warning(f"Initial greeting notice: {e}")

    # 5. Async processing worker for user speech
    async def process_user_speech(ev: UserInputTranscribedEvent):
        if not getattr(ev, "is_final", True):
            return

        user_transcript = getattr(ev, "transcript", "").strip()
        if not user_transcript:
            return

        logger.info(f"User Speech Committed: '{user_transcript}'")
        
        # Parallel Execution: Execute LangGraph state machine
        config = {
            "configurable": {
                "thread_id": ctx.room.name,
                "openrouter_key": settings.OPENROUTER_API_KEY
            }
        }
        initial_state = {
            "raw_query": user_transcript,
            "messages": []
        }
        
        try:
            result = await graph.ainvoke(initial_state, config=config)
            raw_answer = result.get("final_answer", settings.FALLBACK_STRICT_RESPONSE)
            
            # Clean response for TTS engine (strip bracketed citations)
            spoken_text = strip_citations_for_tts(raw_answer)
            logger.info(f"TTS Spoken Output: {spoken_text}")
            await session.say(spoken_text)
        except Exception as e:
            logger.error(f"Error executing LangGraph pipeline: {e}")
            await session.say(settings.FALLBACK_STRICT_RESPONSE)

    # 6. Synchronous event listener to comply with LiveKit EventEmitter rules
    @session.on("user_input_transcribed")
    def on_user_speech(ev: UserInputTranscribedEvent):
        asyncio.create_task(process_user_speech(ev))

if __name__ == "__main__":
    cli.run_app(server)
