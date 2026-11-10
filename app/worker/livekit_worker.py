import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
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
    JobProcess,
    Agent,
    AgentSession,
    TurnHandlingOptions,
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
from app.graph.nodes import (
    detect_is_hinglish,
    detect_turn_hinglish,
    rewrite_query_node,
    retrieve_node,
    stream_answer_chunks,
    warm_llm_clients,
)
from app.core.database import get_chat_history, save_chat_message, get_db_connection
from app.core.qdrant_client import get_http_client
from app.worker.audio_pipeline import CITATION_REGEX, strip_citations_for_tts

# Guard against OpenAI-compatible endpoints (like CheaperInference) that reject stream_format="sse"
class _AudioStreamModelsSet(set):
    def __contains__(self, item):
        return True

oai_tts.AUDIO_STREAM_MODELS = _AudioStreamModelsSet()

logger = logging.getLogger("livekit_worker")
logging.basicConfig(level=logging.INFO)


def prewarm(proc: JobProcess):
    """Preloads Silero VAD ONNX model and core HTTP modules during worker startup."""
    t0 = time.perf_counter()
    proc.userdata["vad"] = silero.VAD.load(min_silence_duration=0.55)
    logger.info(
        f"[PERF][PREWARM] silero.VAD.load(min_silence_duration=0.55) completed in {(time.perf_counter() - t0)*1000:.2f}ms"
    )


server = AgentServer(setup_fnc=prewarm)


class _StreamingCitationStripper:
    """Strips [chunk_xxx] markers across arbitrary LLM stream token boundaries."""

    def __init__(self) -> None:
        self._buf = ""
        self._pending_space = False

    def _emit(self, text: str) -> str:
        if not text:
            return ""
        out = ""
        if self._pending_space:
            if text[0] not in ".,!?;:":
                out += " "
            self._pending_space = False

        if text.endswith(" "):
            text = text[:-1]
            self._pending_space = True
        return out + text

    def feed(self, chunk: str) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        emitted: list[str] = []

        while "[" in self._buf:
            idx = self._buf.index("[")
            if idx > 0:
                emitted.append(self._emit(self._buf[:idx]))
                self._buf = self._buf[idx:]

            if "]" in self._buf:
                end = self._buf.index("]")
                tag = self._buf[: end + 1]
                self._buf = self._buf[end + 1 :]
                if CITATION_REGEX.fullmatch(tag):
                    continue
                emitted.append(self._emit(tag))
            else:
                # Keep buffering while partial token can still match [chunk_...]
                tail = self._buf[1:]
                can_be_citation = (
                    len(self._buf) <= 64
                    and ("chunk_".startswith(tail) or tail.startswith("chunk_"))
                )
                if can_be_citation:
                    break
                emitted.append(self._emit(self._buf))
                self._buf = ""
                break

        if "[" not in self._buf and self._buf:
            emitted.append(self._emit(self._buf))
            self._buf = ""

        return "".join(emitted)

    def flush(self) -> str:
        out = self._emit(self._buf) if self._buf else ""
        self._buf = ""
        if self._pending_space:
            self._pending_space = False
        return out


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    logger.info(f"Incoming room job for room: {ctx.room.name}")

    # 1. Reuse prewarmed Silero VAD from worker setup_fnc (zero entrypoint load stall)
    vad = ctx.proc.userdata.get("vad")
    if vad is None:
        t_vad0 = time.perf_counter()
        vad = silero.VAD.load(min_silence_duration=0.55)
        logger.info(f"[PERF][VAD_FALLBACK_LOAD] {(time.perf_counter() - t_vad0)*1000:.2f}ms")

    # Warm loop-local singletons (LLM/HTTP/SQLite WAL) concurrently with WebRTC connect
    warm_llm_clients(settings.effective_llm_api_key)
    get_http_client()
    bg_tasks: set[asyncio.Task] = set()

    def _spawn_bg_task(coro, label: str) -> asyncio.Task:
        task = asyncio.create_task(coro)
        bg_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            bg_tasks.discard(t)
            if not t.cancelled():
                exc = t.exception()
                if exc is not None:
                    logger.warning(f"Background task failed ({label}): {exc}")

        task.add_done_callback(_done)
        return task

    _spawn_bg_task(get_db_connection(), "warm_db")

    # 2. Connect agent worker WebRTC transport & configure low-latency plugins
    await ctx.connect()
    logger.info(f"Agent successfully connected to WebRTC room: {ctx.room.name}")

    # Use StreamAdapter to wrap standard REST Whisper STT with local VAD buffering
    base_stt = openai.STT(
        model=settings.effective_stt_model,
        base_url=settings.effective_stt_base_url,
        api_key=settings.effective_stt_api_key,
        use_realtime=False,
        prompt=(
            "Transcribe spoken Hindi, Urdu, or English strictly into the Roman alphabet (Hinglish/Latin script). "
            "Never use Devanagari (हिन्दी) or Arabic/Urdu (اردو) scripts. "
            "Example: 'Haan main sun raha hoon, bataiye aapko kya help chahiye?'"
        ),
        # Forcing 'en' mode steers Latin-alphabet acoustic decoding
        language="en",
        detect_language=False,
    )
    adapted_stt = stt.StreamAdapter(stt=base_stt, vad=vad)

    # Low-latency TTS: OpenRouter expects response_format="pcm" @ 24kHz for ElevenLabs models;
    # secondary CheaperInference supports opus @ 24kHz for tts-1 and pcm @ 44.1kHz for eleven-multilingual-v2.
    en_tts_format = "pcm" if settings.is_openrouter_tts else "opus"
    hi_tts_sample_rate = 24000 if settings.is_openrouter_tts else 44100

    tts_english = openai.TTS(
        model=settings.effective_tts_model,
        base_url=settings.effective_tts_base_url,
        api_key=settings.effective_tts_api_key,
        voice=settings.effective_tts_voice,
        response_format=en_tts_format,
        sample_rate=24000,
    )

    tts_hinglish = openai.TTS(
        model=settings.effective_tts_hinglish_model,
        base_url=settings.effective_tts_base_url,
        api_key=settings.effective_tts_api_key,
        voice=settings.effective_tts_hinglish_voice,
        response_format="pcm",
        sample_rate=hi_tts_sample_rate,
    )

    # Prewarm TLS/HTTP pools for both English and Hinglish TTS clients at session start
    tts_english.prewarm()
    tts_hinglish.prewarm()

    # Wait for user participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined room: {participant.identity}")

    # 3. TurnHandlingOptions calibrated right above Silero's 0.55s silence threshold
    turn_handling: TurnHandlingOptions = {
        "turn_detection": "vad",
        "endpointing": {
            "mode": "fixed",
            "min_delay": 0.6,
            "max_delay": 2.5,
        },
        "interruption": {"enabled": True, "mode": "vad"}
    }

    session = AgentSession(
        vad=vad,
        stt=adapted_stt,
        tts=tts_english,
        turn_handling=turn_handling
    )

    # 4. End-to-end millisecond telemetry tracking
    turn_marks: dict[str, float | None] = {
        "speech_end_perf": None,
        "vad_eos_perf": None,
        "stt_final_perf": None,
        "say_called_perf": None,
        "first_tts_token_perf": None,
    }

    @session.on("user_state_changed")
    def _perf_on_user_state(ev):
        if getattr(ev, "old_state", None) == "speaking" and getattr(ev, "new_state", None) == "listening":
            now = time.perf_counter()
            turn_marks["vad_eos_perf"] = now
            turn_marks["speech_end_perf"] = now - 0.55

    @session.on("agent_state_changed")
    def _perf_on_agent_state(ev):
        if getattr(ev, "new_state", None) == "speaking" and turn_marks["say_called_perf"] is not None:
            now = time.perf_counter()
            tts_from_say_ms = (now - turn_marks["say_called_perf"]) * 1000
            tts_from_tok_ms = (
                (now - turn_marks["first_tts_token_perf"]) * 1000
                if turn_marks["first_tts_token_perf"] is not None
                else -1.0
            )
            e2e_from_eos_ms = (
                (now - turn_marks["vad_eos_perf"]) * 1000
                if turn_marks["vad_eos_perf"] is not None
                else -1.0
            )
            e2e_from_speech_ms = (
                (now - turn_marks["speech_end_perf"]) * 1000
                if turn_marks["speech_end_perf"] is not None
                else -1.0
            )
            logger.info(
                f"[PERF][AUDIO_OUT] tts_ttfb_from_say={tts_from_say_ms:.1f}ms | "
                f"tts_from_first_token={tts_from_tok_ms:.1f}ms | "
                f"TTFA_from_VAD_EOS={e2e_from_eos_ms:.1f}ms | "
                f"TTFA_from_user_stop_speaking={e2e_from_speech_ms:.1f}ms"
            )
            turn_marks["say_called_perf"] = None
            turn_marks["first_tts_token_perf"] = None

    @session.on("metrics_collected")
    def _perf_on_metrics(ev):
        m = getattr(ev, "metrics", ev)
        logger.info(f"[PERF][LK_METRICS] {type(m).__name__}: {m}")

    # 5. Authoritative turn-completion processor (no duplicate STT trigger)
    async def handle_user_query(user_transcript: str):
        t_turn_start = time.perf_counter()
        user_transcript = user_transcript.strip()
        if not user_transcript:
            return

        # Deduplicate spurious rapid identical commits while still permitting repeated user questions across turns
        stt_lag_ms = (
            (t_turn_start - turn_marks["vad_eos_perf"]) * 1000
            if turn_marks["vad_eos_perf"] is not None
            else -1.0
        )
        logger.info(
            f"[PERF][STT] stt_to_turn_commit={stt_lag_ms:.1f}ms | transcript_len={len(user_transcript)}"
        )
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
        t_state0 = time.perf_counter()
        current_state = await graph.aget_state(config)
        t_aget_ms = (time.perf_counter() - t_state0) * 1000
        existing_msgs = list(current_state.values.get("messages", [])) if current_state else []

        # Seed checkpointer with SQLite history only on turn 1 when memory is empty
        messages_to_pass = []
        t_hist_ms = 0.0
        if not existing_msgs:
            t_h0 = time.perf_counter()
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
            t_hist_ms = (time.perf_counter() - t_h0) * 1000

        user_human_msg = HumanMessage(content=user_transcript)
        messages_for_prompt = list(existing_msgs) + messages_to_pass + [user_human_msg]

        # Non-blocking background persistence for user turn (0ms stall on voice critical path)
        _spawn_bg_task(save_chat_message(session_id, "user", user_transcript), "save_user_msg")

        is_user_hinglish = detect_turn_hinglish(user_transcript, existing_msgs or messages_to_pass)
        fallback_msg = (
            "Arey, is baare mein hamare documents mein koi information nahin hai."
            if is_user_hinglish
            else settings.FALLBACK_STRICT_RESPONSE
        )

        try:
            # Stage A & B-1: Fast-path query rewrite + optimized singleton Qdrant retrieval
            t_rw0 = time.perf_counter()
            rw_state = await rewrite_query_node(
                {"raw_query": user_transcript, "messages": messages_for_prompt},
                config=config,
            )
            t_rw_ms = (time.perf_counter() - t_rw0) * 1000
            rewritten_query = rw_state.get("rewritten_query", user_transcript)

            t_ret0 = time.perf_counter()
            ret_state = await retrieve_node({"rewritten_query": rewritten_query, "raw_query": user_transcript}, config=config)
            t_ret_ms = (time.perf_counter() - t_ret0) * 1000

            retrieved_context = ret_state.get("retrieved_context", [])
            is_context_relevant = ret_state.get("is_context_relevant", False)

            # Select TTS voice cleanly BEFORE starting streaming TTS session
            chosen_tts = tts_hinglish if is_user_hinglish else tts_english
            session._tts = chosen_tts
            if hasattr(agent, "_tts"):
                agent._tts = chosen_tts
            logger.info(
                f"TTS Voice Selected: {'Indian Voice (' + settings.effective_tts_hinglish_voice + ')' if is_user_hinglish else 'English Voice (' + settings.effective_tts_voice + ')'}"
            )
            logger.info(
                f"[PERF][PRE_STREAM] aget_state={t_aget_ms:.2f}ms | db_select={t_hist_ms:.2f}ms | "
                f"rewrite={t_rw_ms:.2f}ms | retrieve={t_ret_ms:.1f}ms | "
                f"pre_tts_elapsed={(time.perf_counter() - t_turn_start)*1000:.1f}ms"
            )

            gen_state = {
                "raw_query": user_transcript,
                "rewritten_query": rewritten_query,
                "retrieved_context": retrieved_context,
                "is_context_relevant": is_context_relevant,
                "messages": messages_for_prompt,
            }
            result_holder: dict = {}

            async def _spoken_stream():
                stripper = _StreamingCitationStripper()
                spoken_collected: list[str] = []
                first_emitted = False
                try:
                    async for raw_chunk in stream_answer_chunks(
                        gen_state, api_key=api_key, result_holder=result_holder
                    ):
                        clean_chunk = stripper.feed(raw_chunk)
                        if clean_chunk:
                            if not first_emitted:
                                first_emitted = True
                                turn_marks["first_tts_token_perf"] = time.perf_counter()
                            spoken_collected.append(clean_chunk)
                            yield clean_chunk
                    tail = stripper.flush()
                    if tail:
                        if not first_emitted:
                            first_emitted = True
                            turn_marks["first_tts_token_perf"] = time.perf_counter()
                        spoken_collected.append(tail)
                        yield tail
                finally:
                    raw_answer = result_holder.get("final_answer", "")
                    citations = result_holder.get("citations", [])
                    if raw_answer:
                        logger.info(f"TTS Spoken Output: {strip_citations_for_tts(raw_answer)}")
                        new_state_msgs = messages_to_pass + [user_human_msg, AIMessage(content=raw_answer)]
                        _spawn_bg_task(
                            graph.aupdate_state(
                                config,
                                {
                                    "raw_query": user_transcript,
                                    "rewritten_query": rewritten_query,
                                    "retrieved_context": retrieved_context,
                                    "is_context_relevant": is_context_relevant,
                                    "final_answer": raw_answer,
                                    "messages": new_state_msgs,
                                    "citations": citations,
                                },
                            ),
                            "update_graph_state",
                        )
                        citations_json = json.dumps(citations)
                        _spawn_bg_task(
                            save_chat_message(session_id, "agent", raw_answer, citations_json),
                            "save_agent_msg",
                        )

            say_t0 = time.perf_counter()
            turn_marks["say_called_perf"] = say_t0
            speech_handle = session.say(_spoken_stream())

            def _on_say_done(_):
                now = time.perf_counter()
                logger.info(
                    f"[PERF][SAY_PLAYOUT_COMPLETE] say_to_done={(now - say_t0)*1000:.1f}ms | "
                    f"turn_total={(now - t_turn_start)*1000:.1f}ms"
                )

            speech_handle.add_done_callback(_on_say_done)
        except Exception as e:
            logger.error(f"Error executing streaming LangGraph pipeline: {e}")
            chosen_tts = tts_hinglish if is_user_hinglish else tts_english
            session._tts = chosen_tts
            if hasattr(agent, "_tts"):
                agent._tts = chosen_tts
            turn_marks["say_called_perf"] = time.perf_counter()
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

    # 7. Telemetry-only transcription listener (single authoritative turn trigger is on_user_turn_completed)
    @session.on("user_input_transcribed")
    def on_user_speech(ev: UserInputTranscribedEvent):
        if getattr(ev, "is_final", True) and getattr(ev, "transcript", ""):
            turn_marks["stt_final_perf"] = time.perf_counter()
            logger.info(f"[PERF][STT_EVENT] is_final=True transcript='{ev.transcript}'")
        else:
            logger.debug(f"[LiveKit Speech Event] is_final={ev.is_final}, transcript='{ev.transcript}'")


if __name__ == "__main__":
    cli.run_app(server)
