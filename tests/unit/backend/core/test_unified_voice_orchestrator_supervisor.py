"""Supervisor-focused tests for UnifiedVoiceOrchestrator speech routing."""

import sys
import types

import pytest

from backend.core.supervisor.unified_voice_orchestrator import (
    UnifiedVoiceOrchestrator,
    VoiceConfig,
)


class _DummyProc:
    """Minimal async subprocess stub for create_subprocess_exec monkeypatching."""

    def __init__(self):
        self.returncode = None

    async def wait(self):
        self.returncode = 0
        return 0


@pytest.mark.asyncio
async def test_execute_say_device_free_prefers_direct_say_with_canonical_voice(monkeypatch):
    """Device-free startup speech should use direct say with canonical voice."""
    monkeypatch.setenv("JARVIS_ENFORCE_CANONICAL_VOICE", "true")
    monkeypatch.setenv("JARVIS_CANONICAL_VOICE_NAME", "Daniel")
    monkeypatch.setenv("JARVIS_VOICE_NAME", "Samantha")
    monkeypatch.setattr(
        "backend.core.supervisor.unified_voice_orchestrator.platform.system",
        lambda: "Darwin",
    )

    orchestrator = UnifiedVoiceOrchestrator(config=VoiceConfig(voice="Samantha"))
    assert orchestrator.config.voice == "Daniel"

    # AudioBus probe reports "device not held".
    fake_audio_bus_mod = types.ModuleType("backend.audio.audio_bus")

    class _AudioBusFree:
        @staticmethod
        def get_instance_safe():
            return None

    fake_audio_bus_mod.AudioBus = _AudioBusFree
    monkeypatch.setitem(sys.modules, "backend.audio.audio_bus", fake_audio_bus_mod)

    # If this path is used in device-free mode, it's a regression.
    fake_tts_mod = types.ModuleType("backend.voice.engines.unified_tts_engine")

    async def _unexpected_get_tts_engine():
        raise AssertionError("UnifiedTTSEngine should not be used in device-free direct-say path")

    fake_tts_mod.get_tts_engine = _unexpected_get_tts_engine
    monkeypatch.setitem(sys.modules, "backend.voice.engines.unified_tts_engine", fake_tts_mod)

    calls = []

    async def _fake_create_subprocess_exec(*cmd, **kwargs):
        calls.append(cmd)
        return _DummyProc()

    monkeypatch.setattr(
        "backend.core.supervisor.unified_voice_orchestrator.asyncio.create_subprocess_exec",
        _fake_create_subprocess_exec,
    )

    await orchestrator._execute_say("Hello from startup")

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[0] == "say"
    assert cmd[1] == "-v"
    assert cmd[2] == "Daniel"


@pytest.mark.asyncio
async def test_execute_say_device_held_does_not_fall_back_to_legacy_paths(monkeypatch):
    """When device is held and AudioBus TTS fails, orchestrator should skip speech safely."""
    monkeypatch.setenv("JARVIS_ENFORCE_CANONICAL_VOICE", "true")
    monkeypatch.setenv("JARVIS_CANONICAL_VOICE_NAME", "Daniel")
    monkeypatch.setattr(
        "backend.core.supervisor.unified_voice_orchestrator.platform.system",
        lambda: "Darwin",
    )

    orchestrator = UnifiedVoiceOrchestrator(config=VoiceConfig())

    # AudioBus probe reports "device held".
    fake_audio_bus_mod = types.ModuleType("backend.audio.audio_bus")

    class _RunningBus:
        is_running = True

    class _AudioBusHeld:
        @staticmethod
        def get_instance_safe():
            return _RunningBus()

    fake_audio_bus_mod.AudioBus = _AudioBusHeld
    monkeypatch.setitem(sys.modules, "backend.audio.audio_bus", fake_audio_bus_mod)

    # Simulate TTS failure while device is held.
    fake_tts_mod = types.ModuleType("backend.voice.engines.unified_tts_engine")

    class _FailingTTS:
        async def speak(self, *_args, **_kwargs):
            raise RuntimeError("simulated TTS failure")

    async def _get_tts_engine():
        return _FailingTTS()

    fake_tts_mod.get_tts_engine = _get_tts_engine
    monkeypatch.setitem(sys.modules, "backend.voice.engines.unified_tts_engine", fake_tts_mod)

    legacy_calls = []

    async def _unexpected_create_subprocess_exec(*cmd, **kwargs):
        legacy_calls.append(cmd)
        return _DummyProc()

    monkeypatch.setattr(
        "backend.core.supervisor.unified_voice_orchestrator.asyncio.create_subprocess_exec",
        _unexpected_create_subprocess_exec,
    )

    # Must not raise and must not call legacy say path.
    await orchestrator._execute_say("Held-device test")
    assert legacy_calls == []
