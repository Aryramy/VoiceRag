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
        # Ensure not all 5 are identical
        assert len(set(selections)) > 1
    asyncio.run(_test())

def test_acknowledgement_stop_and_reset():
    async def _test():
        mgr = AcknowledgementManager()
        await mgr.initialize()

        class MockHandle:
            def __init__(self):
                self.interrupted = False
                self._done = False
            def done(self):
                return self._done
            def interrupt(self, force=True):
                self.interrupted = True
                self._done = True

        mock_h = MockHandle()
        mgr._current_handle = mock_h
        mgr._is_playing = True

        await mgr.stop_acknowledgement(reason="test_stop")
        assert mock_h.interrupted is True
        assert mgr._is_playing is False
        assert mgr._current_handle is None
    asyncio.run(_test())
