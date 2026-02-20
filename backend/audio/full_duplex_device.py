"""
Full-Duplex Audio Device Abstraction (Layer -1)
=================================================

Single sounddevice.Stream with synchronized capture and playback at the same
sample clock. This is the foundation for acoustic echo cancellation — both
indata (mic) and outdata (speaker) arrive in the same callback at the same
frame boundary, guaranteeing time-alignment.

CONSTRAINT: This is the ONLY place in the codebase that opens a sounddevice
stream. All audio I/O flows through this handle.

Architecture:
    Microphone ──▶ indata ──▶ capture_callback(np.ndarray)
                               (registered by AudioBus)
    outdata ◀── PlaybackRingBuffer ◀── write_playback(np.ndarray)
                                        (called by AudioBus)
"""

import asyncio
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

from backend.audio.playback_ring_buffer import PlaybackRingBuffer

logger = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    """Audio device configuration. All values can be overridden via env vars."""

    input_device: Optional[int] = None    # None = system default mic
    output_device: Optional[int] = None   # None = system default speaker
    sample_rate: int = 48000              # Native macOS rate
    internal_rate: int = 16000            # VAD/STT processing rate
    frame_duration_ms: int = 20           # 960 samples at 48kHz
    channels: int = 1
    dtype: str = "float32"
    playback_buffer_seconds: float = 2.0  # Ring buffer capacity

    def __post_init__(self):
        # Allow env var overrides
        self.sample_rate = int(os.getenv(
            "JARVIS_AUDIO_SAMPLE_RATE", str(self.sample_rate)
        ))
        self.internal_rate = int(os.getenv(
            "JARVIS_AUDIO_INTERNAL_RATE", str(self.internal_rate)
        ))
        self.frame_duration_ms = int(os.getenv(
            "JARVIS_AUDIO_FRAME_MS", str(self.frame_duration_ms)
        ))
        self.playback_buffer_seconds = float(os.getenv(
            "JARVIS_AUDIO_BUFFER_SECONDS", str(self.playback_buffer_seconds)
        ))

        dev_in = os.getenv("JARVIS_AUDIO_INPUT_DEVICE")
        if dev_in is not None:
            self.input_device = int(dev_in)

        dev_out = os.getenv("JARVIS_AUDIO_OUTPUT_DEVICE")
        if dev_out is not None:
            self.output_device = int(dev_out)

    @property
    def frame_size(self) -> int:
        """Samples per frame at device sample rate."""
        return int(self.sample_rate * self.frame_duration_ms / 1000)

    @property
    def internal_frame_size(self) -> int:
        """Samples per frame at internal (processing) rate."""
        return int(self.internal_rate * self.frame_duration_ms / 1000)

    @property
    def playback_buffer_frames(self) -> int:
        """Total ring buffer capacity in samples."""
        return int(self.sample_rate * self.playback_buffer_seconds)


class FullDuplexDevice:
    """
    Single sounddevice.Stream — ALL audio I/O through this handle.

    The callback processes both input (mic) and output (speaker) in the same
    invocation, ensuring frame-level synchronization required for AEC.
    """

    def __init__(self, config: Optional[DeviceConfig] = None):
        self.config = config or DeviceConfig()
        self._stream: Optional["sd.Stream"] = None
        self._playback_buffer = PlaybackRingBuffer(
            capacity_frames=self.config.playback_buffer_frames
        )

        # Capture callback — set by AudioBus
        self._capture_callbacks: List[Callable[[np.ndarray], None]] = []
        self._capture_lock = threading.Lock()

        # Last output frame for AEC reference
        self._last_output_frame = np.zeros(
            self.config.frame_size, dtype=np.float32
        )
        self._output_frame_lock = threading.Lock()

        self._running = False
        self._started_event = asyncio.Event()

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    @property
    def frame_size(self) -> int:
        return self.config.frame_size

    @property
    def playback_buffer(self) -> PlaybackRingBuffer:
        return self._playback_buffer

    async def start(self) -> None:
        """Open the full-duplex stream."""
        if self._running:
            logger.warning("[FullDuplexDevice] Already running")
            return

        if sd is None:
            raise ImportError("sounddevice is not installed")

        try:
            if os.getenv("JARVIS_AUDIO_VALIDATE_DEVICES", "true").lower() in (
                "1",
                "true",
                "yes",
                "on",
            ):
                self._validate_device_selection()

            self._stream = sd.Stream(
                samplerate=self.config.sample_rate,
                blocksize=self.config.frame_size,
                device=(self.config.input_device, self.config.output_device),
                channels=self.config.channels,
                dtype=self.config.dtype,
                callback=self._audio_callback,
                finished_callback=self._stream_finished,
            )
            self._stream.start()
            self._running = True
            self._started_event.set()

            in_device_label = (
                self.config.input_device
                if self.config.input_device is not None
                else "default"
            )
            out_device_label = (
                self.config.output_device
                if self.config.output_device is not None
                else "default"
            )

            logger.info(
                f"[FullDuplexDevice] Started: sr={self.config.sample_rate}, "
                f"frame={self.config.frame_size} samples "
                f"({self.config.frame_duration_ms}ms), "
                f"in={in_device_label}, "
                f"out={out_device_label}"
            )
        except Exception as e:
            logger.error(f"[FullDuplexDevice] Failed to start: {e}")
            raise

    def _validate_device_selection(self) -> None:
        """
        Validate and resolve duplex device selection before opening the stream.

        This prevents PortAudio device=-1 startup failures and avoids partially
        initialized audio state that can manifest as startup noise/static.
        """
        assert sd is not None  # guarded by caller

        try:
            devices = sd.query_devices()
        except Exception as e:
            raise RuntimeError(f"Unable to query audio devices: {e}") from e

        if not devices:
            raise RuntimeError("No audio devices available")

        default_input = None
        default_output = None
        try:
            defaults = getattr(sd.default, "device", None)
            if isinstance(defaults, (tuple, list)) and len(defaults) >= 2:
                default_input = int(defaults[0]) if defaults[0] is not None else None
                default_output = int(defaults[1]) if defaults[1] is not None else None
        except Exception:
            default_input = None
            default_output = None

        input_device = (
            int(self.config.input_device)
            if self.config.input_device is not None
            else default_input
        )
        output_device = (
            int(self.config.output_device)
            if self.config.output_device is not None
            else default_output
        )

        if input_device is None or input_device < 0:
            raise RuntimeError("No valid default input device available")
        if output_device is None or output_device < 0:
            raise RuntimeError("No valid default output device available")

        try:
            sd.check_input_settings(
                device=input_device,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                dtype=self.config.dtype,
            )
            sd.check_output_settings(
                device=output_device,
                channels=self.config.channels,
                samplerate=self.config.sample_rate,
                dtype=self.config.dtype,
            )
        except Exception as e:
            raise RuntimeError(
                f"Audio device validation failed (in={input_device}, out={output_device}): {e}"
            ) from e

        self.config.input_device = input_device
        self.config.output_device = output_device

    async def stop(self) -> None:
        """Close the stream and release resources."""
        if not self._running:
            return

        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                logger.warning(f"[FullDuplexDevice] Error stopping stream: {e}")
            self._stream = None

        self._started_event.clear()
        logger.info("[FullDuplexDevice] Stopped")

    def add_capture_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        """Register a callback to receive mic frames (float32, device rate)."""
        with self._capture_lock:
            if cb not in self._capture_callbacks:
                self._capture_callbacks.append(cb)

    def remove_capture_callback(self, cb: Callable[[np.ndarray], None]) -> None:
        """Unregister a capture callback."""
        with self._capture_lock:
            if cb in self._capture_callbacks:
                self._capture_callbacks.remove(cb)

    def write_playback(self, audio: np.ndarray) -> int:
        """
        Queue audio for playback. Returns number of frames written.
        Audio must be float32 at the device sample rate.
        """
        if not self._running:
            return 0
        return self._playback_buffer.write(audio)

    def flush_playback(self) -> int:
        """
        Immediately discard all queued playback audio.
        Used for barge-in interruption.

        Returns:
            Number of frames discarded.
        """
        return self._playback_buffer.flush()

    def get_last_output_frame(self) -> np.ndarray:
        """
        Get the last frame sent to the speaker. Used by AEC as reference signal.
        Thread-safe — can be called from any thread.
        """
        with self._output_frame_lock:
            return self._last_output_frame.copy()

    def _audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: "sd.CallbackTimeInfo",
        status: "sd.CallbackFlags",
    ) -> None:
        """
        sounddevice stream callback — runs in audio thread.

        CRITICAL: This must be fast (<1ms). No allocations, no locks that
        could block, no I/O. The ring buffer read and callback dispatch
        are the only operations.
        """
        if status:
            logger.debug(f"[FullDuplexDevice] Stream status: {status}")

        # --- OUTPUT: Fill outdata from ring buffer ---
        out_flat = outdata[:, 0] if outdata.ndim == 2 else outdata
        frames_read = self._playback_buffer.read(out_flat)

        # Save output for AEC reference
        with self._output_frame_lock:
            self._last_output_frame = out_flat.copy()

        # --- INPUT: Dispatch mic frames to consumers ---
        in_flat = indata[:, 0].copy() if indata.ndim == 2 else indata.copy()

        with self._capture_lock:
            for cb in self._capture_callbacks:
                try:
                    cb(in_flat)
                except Exception:
                    # Never let a consumer crash the audio thread
                    pass

    def _stream_finished(self) -> None:
        """Called when the stream finishes (e.g., device disconnected)."""
        logger.warning("[FullDuplexDevice] Stream finished unexpectedly")
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def get_device_info(self) -> dict:
        """Get info about the currently active audio devices."""
        if sd is None:
            return {"error": "sounddevice not available"}

        try:
            info = {
                "input": sd.query_devices(
                    self.config.input_device or sd.default.device[0]
                ),
                "output": sd.query_devices(
                    self.config.output_device or sd.default.device[1]
                ),
                "sample_rate": self.config.sample_rate,
                "frame_size": self.config.frame_size,
                "frame_duration_ms": self.config.frame_duration_ms,
                "running": self._running,
            }
            return info
        except Exception as e:
            return {"error": str(e)}

    async def wait_until_started(self, timeout: float = 5.0) -> bool:
        """Wait until the device stream is running."""
        try:
            await asyncio.wait_for(self._started_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
