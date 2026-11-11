import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator, Any
from livekit import rtc
from livekit.agents.voice.speech_handle import SpeechHandle
from livekit.agents.utils.audio import audio_frames_from_file

logger = logging.getLogger("acknowledgement_manager")

DEFAULT_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets" / "acknowledgements"

class AcknowledgementManager:
    """Manages immediate, non-blocking pre-recorded acknowledgement audio clips.

    Eliminates perceived latency when the user stops speaking by playing a short,
    language-appropriate conversational cue (e.g. 'One moment', 'Ek second',
    'لحظة واحدة' or a gentle chime) while RAG retrieval and LLM reasoning run in the background.

    Crucially:
    - Zero TTS API cost at runtime (pre-recorded static assets in RAM).
    - Starts in < 1ms upon confirmed end-of-turn detection.
    - Automatically interrupted and purged the instant real TTS audio is ready.
    - Zero audio overlap or collision.
    """

    def __init__(self, assets_dir: Path | str = DEFAULT_ASSETS_DIR):
        self.assets_dir = Path(assets_dir)
        # In-memory storage: lang -> list of (clip_name, list[rtc.AudioFrame])
        self._clips: Dict[str, List[Tuple[str, List[rtc.AudioFrame]]]] = {
            "en": [],
            "hi": [],
            "ar": [],
            "neutral": [],
        }
        self._last_played_idx: Dict[str, int] = {}
        self._current_handle: Optional[SpeechHandle] = None
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

    async def trigger_acknowledgement(
        self,
        session: Any,
        lang: str,
        turn_id: str,
        t_user_speech_end: float,
    ) -> Optional[SpeechHandle]:
        """Plays a pre-recorded acknowledgement immediately after confirmed user speech end.

        Runs concurrently with background RAG retrieval + LLM synthesis.
        """
        async with self._lock:
            if not self._initialized:
                await self.initialize()

            # Prevent duplicate triggers for the same turn
            if self._current_turn_id == turn_id and (self._is_playing or self._current_handle):
                return None

            # If real response has somehow already finished before ack started, skip it
            if self._turn_completed:
                return None

            # Interrupt any lingering previous acknowledgement
            if self._current_handle and not self._current_handle.done():
                try:
                    self._current_handle.interrupt(force=True)
                except Exception:
                    pass

            self._current_turn_id = turn_id
            self._turn_completed = False
            self._is_playing = True

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

            async def _frame_generator() -> AsyncGenerator[rtc.AudioFrame, None]:
                for frame in frames:
                    yield frame

            try:
                handle = session.say(
                    text="",
                    audio=_frame_generator(),
                    allow_interruptions=True,
                    add_to_chat_ctx=False,
                )
                self._current_handle = handle

                def _on_done(h: Any) -> None:
                    self._is_playing = False

                handle.add_done_callback(_on_done)
                return handle
            except Exception as e:
                logger.warning(f"Failed to play acknowledgement clip: {e}")
                self._is_playing = False
                return None

    async def stop_acknowledgement(self, reason: str = "real_tts_ready") -> None:
        """Stops and clears any active acknowledgement audio immediately without overlap."""
        async with self._lock:
            self._turn_completed = True
            handle = self._current_handle
            self._current_handle = None
            self._is_playing = False

            if handle and not handle.done():
                try:
                    logger.info(f"[PERF][Ack] Stopping acknowledgement ({reason}) for seamless transition to real speech")
                    handle.interrupt(force=True)
                except Exception as e:
                    logger.debug(f"Acknowledgement interrupt notice: {e}")

    def reset_turn(self) -> None:
        """Resets turn tracking state."""
        self._turn_completed = False
        self._current_turn_id = None
        self._is_playing = False
        if self._current_handle and not self._current_handle.done():
            try:
                self._current_handle.interrupt(force=True)
            except Exception:
                pass
        self._current_handle = None
