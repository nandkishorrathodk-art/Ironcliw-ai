"""
Barge-In Controller (Layer 4)
==============================

When the user speaks over Ironcliw, cancel TTS and switch to listening.
No cooldown needed in conversation mode — AEC handles echo suppression.

Architecture:
    VAD (AEC-cleaned) ──▶ BargeInController ──▶ cancel_event.set()
                                                     │
                                              AudioBus.flush_playback()
                                              SpeechState.stop_speaking()

The controller monitors the AEC-cleaned VAD output. If speech is detected
while Ironcliw is playing TTS, it triggers a barge-in:
    1. Sets the cancel event (stops TTS streaming)
    2. Flushes the playback buffer (silence within one frame)
    3. Notifies the speech state manager
"""

import asyncio
import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum consecutive speech frames before triggering barge-in
# Prevents single-frame noise from interrupting
_MIN_SPEECH_FRAMES = int(os.getenv("Ironcliw_BARGEIN_MIN_FRAMES", "3"))

# Cooldown after barge-in before allowing another (ms)
_BARGEIN_COOLDOWN_MS = int(os.getenv("Ironcliw_BARGEIN_COOLDOWN_MS", "500"))


class BargeInController:
    """
    Monitors AEC-cleaned VAD output and interrupts TTS when the user speaks.

    In conversation mode (with AEC), no post-speech cooldown is needed.
    The AEC removes the speaker output from the mic, so any speech detected
    on the cleaned signal is genuinely the user.
    """

    def __init__(self):
        self._cancel_event = asyncio.Event()
        self._speech_frame_count = 0
        self._last_barge_in_ms: float = 0.0
        self._enabled = True

        # Stats
        self._total_barge_ins = 0
        self._suppressed_barge_ins = 0

        # References set during wiring
        self._audio_bus = None
        self._speech_state = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_audio_bus(self, audio_bus) -> None:
        """Set the AudioBus reference for flush_playback."""
        self._audio_bus = audio_bus

    def set_speech_state(self, speech_state) -> None:
        """Set the UnifiedSpeechStateManager reference."""
        self._speech_state = speech_state

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the event loop for scheduling async operations."""
        self._loop = loop

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def on_vad_speech_detected(self, is_speech: bool) -> None:
        """
        Called from the audio thread with VAD results on AEC-cleaned signal.

        If speech is detected while Ironcliw is speaking, trigger barge-in.
        """
        if not self._enabled:
            return

        if is_speech:
            self._speech_frame_count += 1

            if self._speech_frame_count >= _MIN_SPEECH_FRAMES:
                # Check if Ironcliw is currently speaking
                if self._is_jarvis_speaking():
                    self._trigger_barge_in()
        else:
            self._speech_frame_count = 0

    def _is_jarvis_speaking(self) -> bool:
        """Check if Ironcliw is currently outputting audio."""
        # Check AudioBus playback buffer
        if self._audio_bus is not None:
            try:
                buf = self._audio_bus.device
                if buf is not None and buf.playback_buffer.available > 0:
                    return True
            except Exception:
                pass

        # Check speech state manager
        if self._speech_state is not None:
            try:
                return self._speech_state.is_speaking
            except Exception:
                pass

        return False

    def _trigger_barge_in(self) -> None:
        """Execute barge-in: cancel TTS, flush audio, notify state."""
        now_ms = time.time() * 1000

        # Cooldown check
        if now_ms - self._last_barge_in_ms < _BARGEIN_COOLDOWN_MS:
            self._suppressed_barge_ins += 1
            return

        self._last_barge_in_ms = now_ms
        self._total_barge_ins += 1
        self._speech_frame_count = 0

        logger.info(
            f"[BargeIn] User interrupted Ironcliw "
            f"(total: {self._total_barge_ins})"
        )

        # 1. Set cancel event (stops streaming TTS)
        self._cancel_event.set()

        # 2. Flush playback buffer
        if self._audio_bus is not None:
            try:
                flushed = self._audio_bus.flush_playback()
                logger.debug(f"[BargeIn] Flushed {flushed} frames")
            except Exception as e:
                logger.debug(f"[BargeIn] Flush error: {e}")

        # 3. Notify speech state (schedule async on event loop)
        if self._speech_state is not None and self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(
                    lambda: self._loop.create_task(
                        self._speech_state.stop_speaking()
                    )
                )
            except Exception:
                pass

    def get_cancel_event(self) -> asyncio.Event:
        """
        Get the cancellation event. TTS streaming should check this
        and stop if set.
        """
        return self._cancel_event

    def reset(self) -> None:
        """Reset after barge-in has been handled."""
        self._cancel_event.clear()
        self._speech_frame_count = 0

    def get_status(self) -> dict:
        """Get controller status."""
        return {
            "enabled": self._enabled,
            "total_barge_ins": self._total_barge_ins,
            "suppressed_barge_ins": self._suppressed_barge_ins,
            "cancel_event_set": self._cancel_event.is_set(),
            "speech_frame_count": self._speech_frame_count,
        }
