import pytest
import asyncio
from pathlib import Path
from app.worker.acknowledgement import AcknowledgementManager

def test_acknowledgement_manager_initialization():
    async def _test():
        mgr = AcknowledgementManager()
        await mgr.initialize()
        
        assert mgr._initialized is True
        assert len(mgr._clips["en"]) >= 3
        assert len(mgr._clips["hi"]) >= 3
        assert len(mgr._clips["ar"]) >= 3
        assert len(mgr._clips["neutral"]) >= 2
    asyncio.run(_test())

def test_acknowledgement_selection_and_rotation():
    async def _test():
        mgr = AcknowledgementManager()
        await mgr.initialize()

        # Verify language selection
        name_en, frames_en = mgr.select_clip("en")
        assert len(frames_en) > 0
        assert name_en.endswith(".wav")

        name_hi, frames_hi = mgr.select_clip("hi")
        assert len(frames_hi) > 0

        name_ar, frames_ar = mgr.select_clip("ar")
        assert len(frames_ar) > 0

        # Fallback to neutral
        name_neutral, frames_neutral = mgr.select_clip("unknown_lang")
        assert len(frames_neutral) > 0

        # Rotation: consecutive selections should rotate between variants
        selections = [mgr.select_clip("en")[0] for _ in range(5)]
        assert len(set(selections)) > 1
    asyncio.run(_test())

def test_acknowledgement_stop_and_reset():
    async def _test():
        mgr = AcknowledgementManager()
        await mgr.initialize()

        class MockAudioOutput:
            def __init__(self):
                self.frames = []
                self.cleared = False
            async def capture_frame(self, frame):
                self.frames.append(frame)
            def flush(self):
                pass
            def clear_buffer(self):
                self.cleared = True

        class MockSession:
            def __init__(self):
                class Out:
                    audio = MockAudioOutput()
                self.output = Out()

        session = MockSession()
        task = await mgr.trigger_acknowledgement(session, "en", "turn_1", 0.0)
        assert task is not None
        assert mgr._is_playing is True

        await asyncio.sleep(0.05)
        await mgr.stop_acknowledgement(reason="test_stop", session=session)
        assert mgr._is_playing is False
        assert session.output.audio.cleared is True
    asyncio.run(_test())
