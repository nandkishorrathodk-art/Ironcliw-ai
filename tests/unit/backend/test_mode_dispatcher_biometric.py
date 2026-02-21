from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import numpy as np

from backend.audio.mode_dispatcher import ModeDispatcher
from backend.audio.mode_dispatcher import VoiceMode


class _UnlockService:
    def __init__(self):
        self.initialized = True
        self.calls = []

    async def _perform_unlock(self, speaker_name, context_analysis, scenario_analysis):
        self.calls.append(
            {
                "speaker_name": speaker_name,
                "context_analysis": context_analysis,
                "scenario_analysis": scenario_analysis,
            }
        )
        return {"success": True, "message": "Unlocked", "reason": "ok"}


class _VBIAAdapter:
    async def verify_speaker(self, _threshold):
        return True, 0.93


async def test_authenticate_voice_verifies_then_unlocks(monkeypatch):
    mod = types.ModuleType("backend.voice_unlock.voice_biometric_intelligence")

    class _VBI:
        async def verify_and_announce(self, audio_data, context, speak=False):
            assert audio_data is not None
            assert context["command_type"] == "verify_only"
            assert speak is False
            return SimpleNamespace(
                verified=True,
                confidence=0.92,
                speaker_name="Derek",
            )

    async def _get_vbi():
        return _VBI()

    mod.get_voice_biometric_intelligence = _get_vbi
    monkeypatch.setitem(
        sys.modules,
        "backend.voice_unlock.voice_biometric_intelligence",
        mod,
    )

    dispatcher = ModeDispatcher()
    service = _UnlockService()
    dispatcher.set_voice_unlock_service(service)

    result = await dispatcher._authenticate_voice(np.ones(16000, dtype=np.float32))

    assert result["success"] is True
    assert result["verified"] is True
    assert result["unlocked"] is True
    assert result["speaker"] == "Derek"
    assert service.calls


async def test_authenticate_voice_fallback_verifies_but_fails_without_unlock_service():
    dispatcher = ModeDispatcher()
    dispatcher.set_vbia_adapter(_VBIAAdapter())

    result = await dispatcher._authenticate_voice(np.ones(16000, dtype=np.float32))

    assert result["success"] is False
    assert result["verified"] is True
    assert result["unlocked"] is False
    assert result["reason"] == "no_unlock_service"


async def test_on_biometric_done_cancelled_task_restores_previous_mode():
    dispatcher = ModeDispatcher()
    dispatcher._current_mode = VoiceMode.BIOMETRIC
    dispatcher._previous_mode = VoiceMode.COMMAND
    called = {"count": 0}

    async def _fake_return():
        called["count"] += 1
        dispatcher._current_mode = VoiceMode.COMMAND

    dispatcher.return_from_biometric = _fake_return

    biometric_task = asyncio.create_task(asyncio.sleep(10))
    biometric_task.cancel()
    try:
        await biometric_task
    except asyncio.CancelledError:
        pass

    dispatcher._on_biometric_done(biometric_task)
    await asyncio.sleep(0)

    assert called["count"] == 1
