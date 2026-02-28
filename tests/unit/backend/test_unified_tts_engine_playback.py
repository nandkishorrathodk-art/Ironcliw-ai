"""Playback routing tests for UnifiedTTSEngine."""

import io
import sys
import types

import numpy as np
import pytest
import soundfile as sf

from backend.voice.engines.unified_tts_engine import UnifiedTTSEngine
from backend.voice.engines.base_tts_engine import TTSEngine


def _wav_bytes(sample_rate: int = 22050) -> bytes:
    """Create short in-memory WAV bytes for playback tests."""
    samples = np.zeros(sample_rate // 20, dtype=np.float32)  # ~50ms silence
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()


@pytest.mark.asyncio
async def test_play_audio_prefers_afplay_on_macos_when_bus_not_running(monkeypatch):
    """macOS path should use afplay directly when AudioBus is not running."""
    engine = UnifiedTTSEngine(enable_cache=False)
    audio_bytes = _wav_bytes()
    sample_rate = 22050

    called = {"afplay": 0}

    # Force Darwin path.
    monkeypatch.setattr(
        "backend.voice.engines.unified_tts_engine.platform.system",
        lambda: "Darwin",
    )

    # Ensure AudioBus probe resolves but reports "not running".
    fake_audio_bus_mod = types.ModuleType("backend.audio.audio_bus")

    class _FakeAudioBus:
        @staticmethod
        def get_instance_safe():
            return None

    fake_audio_bus_mod.AudioBus = _FakeAudioBus
    monkeypatch.setitem(sys.modules, "backend.audio.audio_bus", fake_audio_bus_mod)

    # afplay should be selected.
    monkeypatch.setattr(
        engine,
        "_play_with_afplay",
        lambda _audio: called.__setitem__("afplay", called["afplay"] + 1),
    )

    # If sounddevice path is reached, fail the test.
    monkeypatch.setattr(
        "backend.voice.engines.unified_tts_engine.sd.play",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("sd.play should not be called on Darwin fallback path")
        ),
    )
    monkeypatch.setattr("backend.voice.engines.unified_tts_engine.sd.wait", lambda: None)

    await engine._play_audio(audio_bytes, sample_rate)

    assert called["afplay"] == 1


def test_pyttsx3_disabled_by_default_on_macos(monkeypatch):
    """Darwin fallback order should not include pyttsx3 unless explicitly enabled."""
    monkeypatch.setattr(
        "backend.voice.engines.unified_tts_engine.platform.system",
        lambda: "Darwin",
    )
    monkeypatch.delenv("Ironcliw_TTS_ALLOW_PYTTSX3_DARWIN", raising=False)
    monkeypatch.setenv("Ironcliw_CANONICAL_VOICE_NAME", "Daniel")
    monkeypatch.setenv("Ironcliw_VOICE_NAME", "Samantha")
    monkeypatch.setenv("Ironcliw_ENFORCE_CANONICAL_VOICE", "true")

    engine = UnifiedTTSEngine(
        preferred_engine=TTSEngine.PYTTSX3,
        enable_cache=False,
    )

    assert engine.preferred_engine == TTSEngine.MACOS
    assert engine.config.voice == "Daniel"
    assert TTSEngine.PYTTSX3 not in engine.fallback_order


def test_pyttsx3_can_be_explicitly_enabled_on_macos(monkeypatch):
    """Darwin path should allow pyttsx3 when opt-in env var is set."""
    monkeypatch.setattr(
        "backend.voice.engines.unified_tts_engine.platform.system",
        lambda: "Darwin",
    )
    monkeypatch.setenv("Ironcliw_TTS_ALLOW_PYTTSX3_DARWIN", "true")
    monkeypatch.setenv("Ironcliw_ENFORCE_CANONICAL_VOICE", "false")

    engine = UnifiedTTSEngine(enable_cache=False)

    assert TTSEngine.PYTTSX3 in engine.fallback_order
