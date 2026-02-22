from unittest.mock import AsyncMock

import numpy as np
import pytest

import backend.voice.engines.unified_tts_engine as tts_module
from backend.voice.engines.base_tts_engine import TTSEngine
from backend.voice.engines.unified_tts_engine import UnifiedTTSEngine


@pytest.mark.asyncio
async def test_play_audio_skips_when_screen_locked(monkeypatch):
    engine = UnifiedTTSEngine(preferred_engine=TTSEngine.MACOS, enable_cache=False)
    monkeypatch.setattr(
        engine,
        "_is_screen_locked_for_playback",
        AsyncMock(return_value=True),
    )

    sf_read_called = {"value": False}

    def fake_sf_read(*_args, **_kwargs):
        sf_read_called["value"] = True
        return np.zeros(8, dtype=np.float32), 16000

    monkeypatch.setattr(tts_module.sf, "read", fake_sf_read)

    await engine._play_audio(b"not-a-real-wave", 16000)

    # Lock-state short-circuit should occur before any decode/playback work.
    assert sf_read_called["value"] is False


@pytest.mark.asyncio
async def test_play_audio_respects_output_cooldown(monkeypatch):
    engine = UnifiedTTSEngine(preferred_engine=TTSEngine.MACOS, enable_cache=False)
    engine._enter_audio_output_cooldown("pre-existing failure")
    monkeypatch.setattr(
        engine,
        "_is_screen_locked_for_playback",
        AsyncMock(return_value=False),
    )

    sf_read_called = {"value": False}

    def fake_sf_read(*_args, **_kwargs):
        sf_read_called["value"] = True
        return np.zeros(8, dtype=np.float32), 16000

    monkeypatch.setattr(tts_module.sf, "read", fake_sf_read)

    await engine._play_audio(b"not-a-real-wave", 16000)

    assert sf_read_called["value"] is False


@pytest.mark.asyncio
async def test_play_audio_enters_cooldown_on_expected_output_error(monkeypatch):
    engine = UnifiedTTSEngine(preferred_engine=TTSEngine.MACOS, enable_cache=False)
    monkeypatch.setattr(
        engine,
        "_is_screen_locked_for_playback",
        AsyncMock(return_value=False),
    )

    def fake_sf_read(*_args, **_kwargs):
        return np.zeros(8, dtype=np.float32), 16000

    def raise_afplay(_audio_bytes):
        raise RuntimeError("returned non-zero exit status 1")

    def raise_sd(_data, _sample_rate):
        raise RuntimeError(
            "Error opening OutputStream: Internal PortAudio error [PaErrorCode -9986]"
        )

    monkeypatch.setattr(tts_module.sf, "read", fake_sf_read)
    monkeypatch.setattr(engine, "_play_with_afplay", raise_afplay)
    monkeypatch.setattr(engine, "_has_sounddevice_output", lambda: True)
    monkeypatch.setattr(engine, "_play_with_sounddevice", raise_sd)

    await engine._play_audio(b"not-a-real-wave", 16000)

    assert engine._audio_output_failure_streak >= 1
    assert engine._audio_output_cooldown_remaining() > 0

