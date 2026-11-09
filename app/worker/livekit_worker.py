import asyncio
import os
import sys
import json
from pathlib import Path
import logging
from dotenv import load_dotenv

# Ensure project root is in sys.path for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from app.config import settings

# Populate environment variables
os.environ.setdefault("LIVEKIT_URL", settings.LIVEKIT_URL or "ws://127.0.0.1:7880")
os.environ.setdefault("LIVEKIT_API_KEY", settings.LIVEKIT_API_KEY or "devkey")
os.environ.setdefault("LIVEKIT_API_SECRET", settings.LIVEKIT_API_SECRET or "secret")

from livekit.agents import (
    AgentServer,
    JobContext,
    Agent,
    AgentSession,
    TurnHandlingOptions,
    EndpointingOptions,
    cli,
    stt,
    llm,
    StopResponse,
    UserInputTranscribedEvent,
)
from livekit.plugins import openai, silero
import livekit.plugins.openai.tts as oai_tts
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.workflow import graph
from app.core.database import get_chat_history, save_chat_message
from app.worker.audio_pipeline import strip_citations_for_tts

# Guard against OpenAI-compatible endpoints (like CheaperInference) that reject stream_format="sse"
class _AudioStreamModelsSet(set):
    def __contains__(self, item):
        return True

oai_tts.AUDIO_STREAM_MODELS = _AudioStreamModelsSet()

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

    # 3. Configure plugins: local Silero VAD + CheaperInference STT & TTS
    vad = silero.VAD.load()
    
    # Use StreamAdapter to wrap standard REST Whisper STT with local VAD buffering
    base_stt = openai.STT(
        model=settings.STT_MODEL,
        base_url=settings.effective_stt_base_url,
        api_key=settings.effective_stt_api_key,
        use_realtime=False,
        # stop STT auto-translate:
        language="",
        detect_language=True
    )
    adapted_stt = stt.StreamAdapter(stt=base_stt, vad=vad)

    tts_english = openai.TTS(
        model=settings.TTS_MODEL,
        base_url=settings.effective_tts_base_url,
        api_key=settings.effective_tts_api_key,
        voice=settings.TTS_VOICE
    )

    tts_hinglish = openai.TTS(
        model=settings.TTS_HINGLISH_MODEL,
        base_url=settings.effective_tts_base_url,
        api_key=settings.effective_tts_api_key,
        voice=settings.TTS_HINGLISH_VOICE
    )

    # 4. TurnHandlingOptions with 2.5-second silence threshold (min_delay = 2.5s)
    turn_handling: TurnHandlingOptions = {
        "turn_detection": "vad",
        "endpointing": {
            "mode": "fixed",
            "min_delay": 2.5,
            "max_delay": 4.0,
        },
        "interruption": {"enabled": True, "mode": "vad"}
    }

    session = AgentSession(
        vad=vad,
        stt=adapted_stt,
        tts=tts_english,
        turn_handling=turn_handling
    )

    # 5. Async processing worker for user speech
    recent_transcripts = set()

    async def handle_user_query(user_transcript: str):
        user_transcript = user_transcript.strip()
        if not user_transcript or user_transcript in recent_transcripts:
            return

        recent_transcripts.add(user_transcript)
        if len(recent_transcripts) > 50:
            recent_transcripts.clear()

        logger.info(f"User Speech Committed: '{user_transcript}'")
        
        session_id = ctx.room.name
        api_key = settings.effective_llm_api_key
        config = {
            "configurable": {
                "thread_id": session_id,
                "api_key": api_key,
                "openrouter_key": api_key
            }
        }

        # Inspect existing messages in LangGraph checkpointer for this session
        current_state = await graph.aget_state(config)
        existing_msgs = current_state.values.get("messages", []) if current_state else []

        # If checkpointer has no history yet, seed it with past SQLite history if available
        messages_to_pass = []
        if not existing_msgs:
            try:
                db_history = await get_chat_history(session_id)
                for entry in db_history:
                    sender = entry.get("sender")
                    content = entry.get("content", "")
                    if sender == "user":
                        messages_to_pass.append(HumanMessage(content=content))
                    elif sender == "agent":
                        messages_to_pass.append(AIMessage(content=content))
            except Exception as e:
                logger.warning(f"Failed to load chat history from DB: {e}")

        # Pass current turn's user speech as HumanMessage into LangGraph
        messages_to_pass.append(HumanMessage(content=user_transcript))

        initial_state = {
            "raw_query": user_transcript,
            "messages": messages_to_pass
        }

        # Persist user speech to SQLite
        try:
            await save_chat_message(session_id, "user", user_transcript)
        except Exception as e:
            logger.warning(f"Failed to save user speech to DB: {e}")
        
        from app.graph.nodes import detect_is_hinglish
        is_user_hinglish = detect_is_hinglish(user_transcript)
        fallback_msg = (
            "Maaf kijiye, mujhe is baare mein hamari knowledge base mein koi specific jankari nahi mili."
            if is_user_hinglish
            else settings.FALLBACK_STRICT_RESPONSE
        )

        try:
            result = await graph.ainvoke(initial_state, config=config)
            raw_answer = result.get("final_answer") or fallback_msg
            citations = result.get("citations", [])

            # Persist agent response and citations to SQLite
            try:
                citations_json = json.dumps(citations)
                await save_chat_message(session_id, "agent", raw_answer, citations_json)
            except Exception as e:
                logger.warning(f"Failed to save agent response to DB: {e}")

            # Clean response for TTS engine (strip bracketed citations)
            spoken_text = strip_citations_for_tts(raw_answer)
            logger.info(f"TTS Spoken Output: {spoken_text}")

            # Dynamic TTS Routing: Indian voice for Hinglish/Hindi turns, alloy for English turns
            is_turn_hinglish = is_user_hinglish or detect_is_hinglish(spoken_text)
            chosen_tts = tts_hinglish if is_turn_hinglish else tts_english
            session._tts = chosen_tts
            if hasattr(agent, "_tts"):
                agent._tts = chosen_tts
            logger.info(
                f"TTS Voice Selected: {'Indian Voice (' + settings.TTS_HINGLISH_VOICE + ')' if is_turn_hinglish else 'English Alloy (' + settings.TTS_VOICE + ')'}"
            )

            await session.say(spoken_text)
        except Exception as e:
            logger.error(f"Error executing LangGraph pipeline: {e}")
            chosen_tts = tts_hinglish if is_user_hinglish else tts_english
            session._tts = chosen_tts
            if hasattr(agent, "_tts"):
                agent._tts = chosen_tts
            await session.say(fallback_msg)

    class VoiceRAGAgent(Agent):
        async def on_user_turn_completed(
            self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
        ) -> None:
            user_text = "".join(new_message.content) if isinstance(new_message.content, list) else str(new_message.content or "")
            logger.info(f"[Turn Completed] User message: '{user_text}'")
            if user_text:
                await handle_user_query(user_text)
            raise StopResponse()

    agent = VoiceRAGAgent(
        instructions="You are an enterprise voice assistant.",
        turn_handling=turn_handling
    )
    await session.start(room=ctx.room, agent=agent)

    # 6. Greet participant to publish initial audio track & activate visualizer
    try:
        await session.say("Hello! How can I help you today?")
    except Exception as e:
        logger.warning(f"Initial greeting notice: {e}")

    # 7. Event listeners
    @session.on("user_input_transcribed")
    def on_user_speech(ev: UserInputTranscribedEvent):
        logger.info(f"[LiveKit Speech Event] is_final={ev.is_final}, transcript='{ev.transcript}'")
        if getattr(ev, "is_final", True) and getattr(ev, "transcript", ""):
            asyncio.create_task(handle_user_query(ev.transcript))

if __name__ == "__main__":

    cli.run_app(server)
