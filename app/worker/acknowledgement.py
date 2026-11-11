import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from livekit import rtc
from livekit.agents.utils.audio import audio_frames_from_file

logger = logging.getLogger("acknowledgement_manager")

DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "acknowledgements"

class AcknowledgementManager:
    """Manages immediate, non-blocking pre-recorded acknowledgement audio clips.

    Eliminates perceived latency when the user stops speaking by playing a short,
    language-appropriate conversational cue (e.g. 'One moment', 'Ek second',
    'لحظة واحدة' or a gentle chime) while RAG retrieval and LLM reasoning run in the background.

    Crucially:
    - Streams directly to session.output.audio (bypasses AgentSession.say so LiveKit's
      turn-commit state machine never cuts or interrupts the audio prematurely).
    - Starts in < 1ms upon confirmed speech end.
    - Smoothly stopped and cleared the instant real TTS audio is ready.
    - Zero TTS API cost at runtime (pre-recorded static assets in RAM).
    - Zero audio collision with real assistant speech.
    """

    def __init__(self, assets_dir: Path | str = DEFAULT_ASSETS_DIR):
        self.assets_dir = Path(assets_dir)
        self._clips: Dict[str, List[Tuple[str, List[rtc.AudioFrame]]]] = {
            "en": [],
            "hi": [],
            "ar": [],
            "neutral": [],
        }
        self._last_played_idx: Dict[str, int] = {}
        self._play_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._current_turn_id: Optional[str] = None
        self._is_playing: bool = False
        self._turn_completed: bool = False
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Preloads all audio assets into RAM on startup."""
        if self._initialized:
            return

        t0 = time.perf_counter()
        loaded_count = 0

        for lang in ("en", "hi", "ar", "neutral"):
            lang_dir = self.assets_dir / lang
            if not lang_dir.exists():
                logger.warning(f"Acknowledgement directory missing: {lang_dir}")
                continue

            for wav_file in sorted(lang_dir.glob("*.wav")):
                try:
                    frames: List[rtc.AudioFrame] = []
                    async for f in audio_frames_from_file(str(wav_file), sample_rate=24000, num_channels=1):
                        frames.append(f)
                    if frames:
                        self._clips[lang].append((wav_file.name, frames))
                        loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to preload acknowledgement {wav_file}: {e}")

        self._initialized = True
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            f"AcknowledgementManager initialized: {loaded_count} clips cached in RAM in {elapsed_ms:.1f}ms."
        )

    def select_clip(self, lang: Optional[str] = None) -> Tuple[str, List[rtc.AudioFrame]]:
        """Selects a language-appropriate clip, rotating to avoid sounding repetitive."""
        target_lang = (lang or "neutral").lower().strip()
        if target_lang not in self._clips or not self._clips[target_lang]:
            target_lang = "neutral"
            if not self._clips["neutral"] and self._clips["en"]:
                target_lang = "en"

        available = self._clips.get(target_lang, [])
        if not available:
            return ("empty", [])

        # Avoid playing the exact same variant consecutive times
        last_idx = self._last_played_idx.get(target_lang, -1)
        if len(available) > 1:
            candidates = [i for i in range(len(available)) if i != last_idx]
            chosen_idx = random.choice(candidates)
        else:
            chosen_idx = 0

        self._last_played_idx[target_lang] = chosen_idx
        return available[chosen_idx]

    async def _stream_frames_to_output(
        self,
        audio_output: Any,
        frames: List[rtc.AudioFrame],
        clip_name: str,
        lang: str
    ) -> None:
        """Paces and writes audio frames directly to LiveKit's audio output sink."""
        try:
            logger.info(f"[PERF][Ack] Direct audio stream started: {clip_name} ({len(frames)} frames)")
            for frame in frames:
                if self._stop_event.is_set():
                    logger.info(f"[PERF][Ack] Direct audio stream stopped early: {clip_name}")
                    break
                await audio_output.capture_frame(frame)
                frame_dur = getattr(frame, "duration", 0.02) or 0.02
                await asyncio.sleep(frame_dur * 0.92)

            if not self._stop_event.is_set() and hasattr(audio_output, "flush"):
                audio_output.flush()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"Error streaming acknowledgement audio frames: {e}")
        finally:
            self._is_playing = False

    async def trigger_acknowledgement(
        self,
        session: Any,
        lang: str,
        turn_id: str,
        t_user_speech_end: float,
    ) -> Optional[asyncio.Task]:
        """Plays a pre-recorded acknowledgement immediately after confirmed user speech end.

        Streams directly to session.output.audio to bypass LiveKit's turn-commit interruption logic.
        """
        async with self._lock:
            if not self._initialized:
                await self.initialize()

            # Prevent duplicate triggers for the same turn
            if self._current_turn_id == turn_id and self._is_playing:
                return None

            # If real response has somehow already finished before ack started, skip it
            if self._turn_completed:
                return None

            # Stop any previous playing task
            if self._play_task and not self._play_task.done():
                self._stop_event.set()
                self._play_task.cancel()
                self._play_task = None

            audio_output = getattr(getattr(session, "output", None), "audio", None)
            if not audio_output:
                logger.warning("session.output.audio not available for acknowledgement playback")
                return None

            self._current_turn_id = turn_id
            self._turn_completed = False
            self._is_playing = True
            self._stop_event.clear()

            clip_name, frames = self.select_clip(lang)
            if not frames:
                self._is_playing = False
                return None

            t_ack_start = time.perf_counter()
            ack_trigger_delay_ms = (t_ack_start - t_user_speech_end) * 1000

            logger.info(
                f"[PERF][Ack] User speech committed -> Starting pre-recorded acknowledgement "
                f"('{clip_name}', lang={lang}) in {ack_trigger_delay_ms:.1f}ms"
            )

            # Spawn direct playback task
            self._play_task = asyncio.create_task(
                self._stream_frames_to_output(audio_output, frames, clip_name, lang)
            )
            return self._play_task

    async def stop_acknowledgement(self, reason: str = "real_tts_ready", session: Any = None) -> None:
        """Stops and clears any active acknowledgement audio immediately without overlap."""
        async with self._lock:
            self._turn_completed = True
            self._stop_event.set()

            if self._play_task and not self._play_task.done():
                logger.info(f"[PERF][Ack] Stopping acknowledgement ({reason}) for seamless transition to real speech")
                self._play_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(self._play_task), timeout=0.1)
                except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                    pass
                self._play_task = None

            self._is_playing = False

            if session:
                audio_output = getattr(getattr(session, "output", None), "audio", None)
                if audio_output and hasattr(audio_output, "clear_buffer"):
                    audio_output.clear_buffer()

    def reset_turn(self, session: Any = None) -> None:
        """Resets turn tracking state and cancels audio."""
        self._turn_completed = False
        self._current_turn_id = None
        self._is_playing = False
        self._stop_event.set()
        if self._play_task and not self._play_task.done():
            self._play_task.cancel()
            self._play_task = None
        if session:
            audio_output = getattr(getattr(session, "output", None), "audio", None)
            if audio_output and hasattr(audio_output, "clear_buffer"):
                audio_output.clear_buffer()
