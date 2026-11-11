import asyncio
import os
import sys
import json
import re
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
    cli,
    stt,
    llm,
    StopResponse,
    UserInputTranscribedEvent,
)
from livekit.plugins import silero
from langchain_core.messages import HumanMessage, AIMessage
from app.graph.workflow import graph
from app.graph.nodes import detect_user_language, detect_is_hinglish, romanize_script
from app.core.database import get_chat_history, save_chat_message
from app.worker.audio_pipeline import strip_citations_for_tts
from app.worker.adapters import create_stt_adapter, create_tts_adapter

logger = logging.getLogger("livekit_worker")
logging.basicConfig(level=logging.INFO)

class RomanizingSTTWrapper(stt.STT):
    """Intercepts STT transcription frames to ensure Urdu/Hindi script is transliterated
    to Roman Hinglish, while strictly preserving authentic Arabic script and English text.
    """
    def __init__(self, wrapped: stt.STT):
        super().__init__(capabilities=wrapped.capabilities)
        self._wrapped = wrapped

    async def _recognize_impl(
        self,
        buffer,
        *,
        language=None,
        conn_options=None,
    ):
        event = await self._wrapped._recognize_impl(
            buffer, language=language, conn_options=conn_options
        )
        if event and event.alternatives:
            for alt in event.alternatives:
                if alt.text and detect_user_language(alt.text) == "hi" and re.search(r'[\u0600-\u06FF\u0900-\u097F]', alt.text):
                    alt.text = await romanize_script(alt.text)
        return event

    def stream(self, *args, **kwargs):
        return self._wrapped.stream(*args, **kwargs)

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

    # 3. Configure plugins: Silero VAD + Replaceable STT & TTS Adapters
    vad = silero.VAD.load()
    
    raw_stt = create_stt_adapter(vad=vad)
    adapted_stt = RomanizingSTTWrapper(raw_stt)
    tts = create_tts_adapter()

    logger.info(
        f"Active Providers: STT='{settings.effective_stt_provider}', TTS='{settings.effective_tts_provider}'"
    )

    # 4. TurnHandlingOptions with responsive VAD endpointing
    turn_handling: TurnHandlingOptions = {
        "turn_detection": "vad",
        "endpointing": {
            "mode": "fixed",
            "min_delay": 2.0,
            "max_delay": 3.5,
        },
        "interruption": {"enabled": True, "mode": "vad"}
    }

    session = AgentSession(
        vad=vad,
        stt=adapted_stt,
        tts=tts,
        turn_handling=turn_handling
    )

    # 5. Async processing worker for user speech
    recent_transcripts = set()

    async def handle_user_query(user_transcript: str):
        user_transcript = user_transcript.strip()
        if not user_transcript:
            return

        user_lang = detect_user_language(user_transcript)

        # For Hindi/Hinglish with Urdu/Devanagari script, spawn background romanization
        # so RAG graph begins immediately without waiting ~1.0s
        if user_lang == "hi" and re.search(r'[\u0600-\u06FF\u0900-\u097F]', user_transcript):
            async def _bg_log_roman(raw_t: str):
                try:
                    r_text = await romanize_script(raw_t)
                    logger.info(f"User Speech Romanized [HI]: '{r_text}'")
                except Exception:
                    pass
            asyncio.create_task(_bg_log_roman(user_transcript))

        if user_transcript in recent_transcripts:
            return

        recent_transcripts.add(user_transcript)
        if len(recent_transcripts) > 50:
            recent_transcripts.clear()

        logger.info(f"User Speech Committed [{user_lang.upper()}]: '{user_transcript}'")
        
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
        
        if user_lang == "ar":
            fallback_msg = "عذراً، لم أجد معلومات محددة بخصوص ذلك في قاعدة بياناتنا."
        elif user_lang == "hi":
            fallback_msg = "Maaf kijiye, mujhe is baare mein hamari knowledge base mein koi specific jankari nahi mili."
        else:
            fallback_msg = settings.FALLBACK_STRICT_RESPONSE

        try:
            result = await graph.ainvoke(initial_state, config=config)
            raw_answer = result.get("final_answer") or fallback_msg
            detected_lang = result.get("detected_language") or user_lang
            citations = result.get("citations", [])

            # Persist agent response and citations to SQLite
            try:
                citations_json = json.dumps(citations)
                await save_chat_message(session_id, "agent", raw_answer, citations_json)
            except Exception as e:
                logger.warning(f"Failed to save agent response to DB: {e}")

            # Dynamic Voice Mapping: Select language-appropriate voice
            chosen_voice = settings.get_tts_voice(detected_lang)
            if hasattr(tts, "update_voice"):
                tts.update_voice(chosen_voice)

            # Clean response for TTS engine
            spoken_text = strip_citations_for_tts(raw_answer, lang=detected_lang)
            if detected_lang == "hi" and re.search(r'[\u0600-\u06FF\u0900-\u097F]', spoken_text):
                spoken_text = await romanize_script(spoken_text)

            logger.info(f"TTS Spoken Output [{detected_lang.upper()}]: {spoken_text}")
            logger.info(
                f"TTS Voice Active: {chosen_voice} (Provider: {settings.effective_tts_provider})"
            )
            session.say(spoken_text)
        except Exception as e:
            logger.error(f"Error executing LangGraph pipeline: {e}")
            session.say(fallback_msg)

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
