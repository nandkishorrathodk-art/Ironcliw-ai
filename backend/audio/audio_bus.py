"""
AudioBus + Acoustic Echo Cancellation (Layer 0)
=================================================

Central audio routing singleton. Enforces single-speaker by construction.
Provides AEC-cleaned mic input to all consumers.

CONSTRAINT — not convention:
    The FullDuplexDevice callback is private to AudioBus. Nothing can bypass
    it. This is the ONLY way audio reaches speakers or leaves the microphone.

Architecture:
    Mic ──▶ FullDuplexDevice ──▶ AEC(mic, ref) ──▶ Resample 48→16k ──▶ consumers
                  ▲                                                        │
                  │                                                   (VAD/STT)
    Speaker ◀── PlaybackRingBuffer ◀── Resample 16→48k ◀── TTS output
"""

import asyncio
import logging
import os
import threading
from abc import ABC, abstractmethod
from typing import AsyncIterator, Callable, ClassVar, Dict, List, Optional

import numpy as np

from backend.audio.full_duplex_device import DeviceConfig, FullDuplexDevice

logger = logging.getLogger(__name__)


# ============================================================================
# Resampler (libsamplerate wrapper or fallback)
# ============================================================================

class Resampler:
    """
    High-quality audio resampler. Uses libsamplerate (samplerate package)
    when available, falls back to numpy linear interpolation.
    """

    def __init__(self, from_rate: int, to_rate: int, channels: int = 1):
        self.from_rate = from_rate
        self.to_rate = to_rate
        self.channels = channels
        self._ratio = to_rate / from_rate
        self._use_libsamplerate = False
        self._resampler = None

        if from_rate == to_rate:
            return  # No-op

        try:
            import samplerate
            self._resampler = samplerate.Resampler("sinc_fastest", channels=channels)
            self._use_libsamplerate = True
            logger.debug(
                f"[Resampler] Using libsamplerate: {from_rate} -> {to_rate}"
            )
        except ImportError:
            logger.info(
                f"[Resampler] libsamplerate not available, using linear "
                f"interpolation: {from_rate} -> {to_rate}"
            )

    def process(self, data: np.ndarray) -> np.ndarray:
        """Resample audio data."""
        if self.from_rate == self.to_rate:
            return data

        if self._use_libsamplerate and self._resampler is not None:
            return self._resampler.process(data, self._ratio)

        # Fallback: linear interpolation
        n_out = int(len(data) * self._ratio)
        if n_out == 0:
            return np.zeros(0, dtype=np.float32)
        indices = np.linspace(0, len(data) - 1, n_out)
        return np.interp(indices, np.arange(len(data)), data).astype(np.float32)


# ============================================================================
# Acoustic Echo Canceller
# ============================================================================

class AcousticEchoCanceller:
    """
    Wraps speexdsp for acoustic echo cancellation. Falls back to spectral
    subtraction if speexdsp is not available.

    The AEC operates at the internal processing rate (16kHz by default).
    """

    def __init__(self, frame_size: int, sample_rate: int = 16000, tail_ms: int = 200):
        self._frame_size = frame_size
        self._sample_rate = sample_rate
        self._tail_length = int(sample_rate * tail_ms / 1000)
        self._aec = None
        self._use_speexdsp = False

        try:
            import speexdsp
            self._aec = speexdsp.EchoCanceller(
                frame_size=frame_size,
                filter_length=self._tail_length,
                sample_rate=sample_rate,
            )
            self._use_speexdsp = True
            logger.info(
                f"[AEC] Using speexdsp: frame={frame_size}, "
                f"tail={tail_ms}ms, sr={sample_rate}"
            )
        except (ImportError, Exception) as e:
            logger.info(
                f"[AEC] speexdsp not available ({e}), using spectral "
                f"subtraction fallback"
            )

    def cancel_echo(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """
        Remove echo of the reference signal (speaker output) from the mic signal.

        Args:
            mic: Microphone capture (float32, internal rate)
            ref: Speaker output reference (float32, internal rate, same length)

        Returns:
            Echo-cancelled mic signal (float32).
        """
        if len(mic) == 0:
            return mic

        if self._use_speexdsp and self._aec is not None:
            try:
                # speexdsp expects int16
                mic_i16 = (mic * 32767).astype(np.int16)
                ref_i16 = (ref * 32767).astype(np.int16)

                # Pad or truncate ref to match mic length
                if len(ref_i16) < len(mic_i16):
                    ref_i16 = np.pad(
                        ref_i16, (0, len(mic_i16) - len(ref_i16))
                    )
                elif len(ref_i16) > len(mic_i16):
                    ref_i16 = ref_i16[:len(mic_i16)]

                out_i16 = self._aec.process(
                    mic_i16.tobytes(),
                    ref_i16.tobytes(),
                )
                return np.frombuffer(out_i16, dtype=np.int16).astype(np.float32) / 32767.0
            except Exception as e:
                logger.debug(f"[AEC] speexdsp error, falling back: {e}")

        # Fallback: spectral subtraction
        return self._spectral_subtraction(mic, ref)

    def _spectral_subtraction(
        self, mic: np.ndarray, ref: np.ndarray
    ) -> np.ndarray:
        """Simple spectral subtraction for echo reduction."""
        if len(ref) == 0 or np.max(np.abs(ref)) < 1e-6:
            return mic  # No reference signal, nothing to subtract

        # Pad ref to match mic
        if len(ref) < len(mic):
            ref = np.pad(ref, (0, len(mic) - len(ref)))
        elif len(ref) > len(mic):
            ref = ref[:len(mic)]

        n_fft = len(mic)
        mic_fft = np.fft.rfft(mic, n=n_fft)
        ref_fft = np.fft.rfft(ref, n=n_fft)

        # Estimate echo magnitude and subtract
        alpha = float(os.getenv("JARVIS_AEC_ALPHA", "1.0"))
        mic_mag = np.abs(mic_fft)
        ref_mag = np.abs(ref_fft) * alpha

        # Spectral floor to avoid musical noise
        floor = 0.01 * mic_mag
        cleaned_mag = np.maximum(mic_mag - ref_mag, floor)

        # Reconstruct with original phase
        phase = np.angle(mic_fft)
        cleaned_fft = cleaned_mag * np.exp(1j * phase)
        cleaned = np.fft.irfft(cleaned_fft, n=n_fft)

        return cleaned[:len(mic)].astype(np.float32)


# ============================================================================
# Audio Sink ABC + Implementations
# ============================================================================

class AudioSink(ABC):
    """Pluggable output target for audio."""

    @abstractmethod
    async def write(self, audio: np.ndarray, sample_rate: int) -> None:
        """Write audio to the sink."""


class LocalSpeakerSink(AudioSink):
    """Routes audio through the FullDuplexDevice playback path."""

    def __init__(self, device: FullDuplexDevice, resampler: Resampler):
        self._device = device
        self._resampler = resampler

    async def write(self, audio: np.ndarray, sample_rate: int) -> None:
        """Resample to device rate and queue for playback."""
        if sample_rate != self._device.sample_rate:
            # Create a one-shot resampler for non-standard rates
            temp_resampler = Resampler(sample_rate, self._device.sample_rate)
            audio = temp_resampler.process(audio)
        elif self._resampler.from_rate != self._device.sample_rate:
            audio = self._resampler.process(audio)

        self._device.write_playback(audio)


class WebSocketSink(AudioSink):
    """
    Streams audio to a browser client via WebSocket.
    AEC is CLIENT-side (browser WebRTC handles it).
    """

    def __init__(self, send_func: Callable):
        self._send = send_func

    async def write(self, audio: np.ndarray, sample_rate: int) -> None:
        """Send audio as int16 PCM bytes over WebSocket."""
        pcm_i16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        try:
            await self._send(pcm_i16.tobytes())
        except Exception as e:
            logger.debug(f"[WebSocketSink] Send error: {e}")


# ============================================================================
# AudioBus (Singleton)
# ============================================================================

class AudioBus:
    """
    Central audio routing bus. Singleton.

    ALL audio I/O in the system flows through this class.
    Provides AEC-cleaned mic input to registered consumers.
    Routes TTS output through the local speaker or WebSocket sinks.
    """

    _instance: ClassVar[Optional["AudioBus"]] = None
    _creation_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self):
        # These are set during start()
        self._device: Optional[FullDuplexDevice] = None
        self._aec: Optional[AcousticEchoCanceller] = None
        self._resampler_down: Optional[Resampler] = None  # 48k -> 16k
        self._resampler_up: Optional[Resampler] = None    # 16k -> 48k
        self._config: Optional[DeviceConfig] = None

        # Mic consumers receive AEC-cleaned, 16kHz float32 frames
        self._mic_consumers: List[Callable[[np.ndarray], None]] = []
        self._consumer_lock = threading.Lock()

        # Output sinks
        self._sinks: Dict[str, AudioSink] = {}
        self._local_sink: Optional[LocalSpeakerSink] = None

        # State
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # v265.0: Mic gate — when active, no mic frames are dispatched.
        # Used by speech state manager to suppress self-voice during TTS.
        self._mic_gate_active: bool = False

    @classmethod
    def get_instance(cls) -> "AudioBus":
        """Get or create the singleton AudioBus."""
        if cls._instance is None:
            with cls._creation_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def get_instance_safe(cls) -> Optional["AudioBus"]:
        """Get the singleton if it exists, otherwise None."""
        return cls._instance

    async def start(self, config: Optional[DeviceConfig] = None) -> None:
        """
        Initialize and start the audio device, AEC, and resamplers.
        """
        if self._running:
            logger.warning("[AudioBus] Already running")
            return

        self._config = config or DeviceConfig()
        self._loop = asyncio.get_running_loop()

        # Initialize resamplers
        self._resampler_down = Resampler(
            self._config.sample_rate, self._config.internal_rate
        )
        self._resampler_up = Resampler(
            self._config.internal_rate, self._config.sample_rate
        )

        # Initialize AEC at internal rate
        self._aec = AcousticEchoCanceller(
            frame_size=self._config.internal_frame_size,
            sample_rate=self._config.internal_rate,
        )

        # Initialize full-duplex device
        self._device = FullDuplexDevice(self._config)
        await self._device.start()

        # Register our callback to receive raw mic frames
        self._device.add_capture_callback(self._on_mic_frame)

        # Create local speaker sink
        self._local_sink = LocalSpeakerSink(self._device, self._resampler_up)
        self._sinks["local"] = self._local_sink

        self._running = True
        logger.info("[AudioBus] Started — all audio routing through bus")

    async def stop(self) -> None:
        """Stop the audio bus and release all resources."""
        if not self._running:
            return

        self._running = False

        if self._device is not None:
            self._device.remove_capture_callback(self._on_mic_frame)
            await self._device.stop()

        self._sinks.clear()
        self._local_sink = None

        with self._consumer_lock:
            self._mic_consumers.clear()

        logger.info("[AudioBus] Stopped")

    # ---- Output (TTS → Speaker) ----

    async def play_audio(
        self, audio: np.ndarray, sample_rate: int, sink_id: str = "local"
    ) -> None:
        """
        Play a complete audio buffer through the specified sink.

        Args:
            audio: float32 audio data
            sample_rate: Sample rate of the audio
            sink_id: Which sink to route to (default "local" speaker)
        """
        sink = self._sinks.get(sink_id)
        if sink is None:
            if self._local_sink is not None:
                sink = self._local_sink
            else:
                logger.warning(f"[AudioBus] No sink '{sink_id}' and no local sink")
                return

        await sink.write(audio, sample_rate)

    async def play_stream(
        self,
        chunks: AsyncIterator[np.ndarray],
        sample_rate: int,
        cancel: Optional[asyncio.Event] = None,
        sink_id: str = "local",
    ) -> None:
        """
        Stream audio chunks to the speaker. Stops if cancel event is set
        (barge-in).

        Args:
            chunks: Async iterator yielding float32 audio arrays
            sample_rate: Sample rate of the chunks
            cancel: Optional event to signal cancellation
            sink_id: Which sink to route to
        """
        sink = self._sinks.get(sink_id, self._local_sink)
        if sink is None:
            logger.warning("[AudioBus] No sink available for streaming")
            return

        async for chunk in chunks:
            if cancel is not None and cancel.is_set():
                logger.debug("[AudioBus] Stream cancelled (barge-in)")
                break
            await sink.write(chunk, sample_rate)

    def flush_playback(self) -> int:
        """
        Immediately discard all queued playback audio.
        Used for barge-in interruption. Thread-safe.

        Returns:
            Number of frames flushed.
        """
        if self._device is not None:
            return self._device.flush_playback()
        return 0

    # ---- Input (Mic → Consumers) ----

    def register_mic_consumer(self, cb: Callable[[np.ndarray], None]) -> None:
        """
        Register a callback to receive AEC-cleaned, 16kHz mic frames.

        The callback is invoked from the audio thread — it must be fast
        and non-blocking. Use a queue if processing is needed.
        """
        with self._consumer_lock:
            if cb not in self._mic_consumers:
                self._mic_consumers.append(cb)
                logger.debug(
                    f"[AudioBus] Registered mic consumer "
                    f"(total: {len(self._mic_consumers)})"
                )

    def unregister_mic_consumer(self, cb: Callable[[np.ndarray], None]) -> None:
        """Unregister a mic consumer."""
        with self._consumer_lock:
            if cb in self._mic_consumers:
                self._mic_consumers.remove(cb)

    # ---- Sink Management ----

    def register_sink(self, sink_id: str, sink: AudioSink) -> None:
        """Register a named audio output sink."""
        self._sinks[sink_id] = sink
        logger.debug(f"[AudioBus] Registered sink: {sink_id}")

    def unregister_sink(self, sink_id: str) -> None:
        """Unregister a named sink."""
        self._sinks.pop(sink_id, None)

    # ---- Mic Gating (v265.0) ----

    def set_mic_gate(self, active: bool) -> None:
        """Gate mic consumers — when active, no mic frames are dispatched.

        v265.0: Used by speech state manager to suppress self-voice during TTS.
        """
        self._mic_gate_active = active
        logger.debug(f"[AudioBus] Mic gate {'ACTIVE' if active else 'INACTIVE'}")

    @property
    def mic_gate_active(self) -> bool:
        """Check if mic gate is active."""
        return self._mic_gate_active

    # ---- Internal: Mic frame processing ----

    def _on_mic_frame(self, raw_frame: np.ndarray) -> None:
        """
        Called from the audio thread with raw mic data at device rate.

        Pipeline: raw mic → downsample → AEC → dispatch to consumers
        """
        if not self._running:
            return

        # v265.0: Mic gate — discard frames when TTS is active
        if self._mic_gate_active:
            return

        try:
            # 1. Downsample from device rate to internal rate
            if self._resampler_down is not None:
                internal_frame = self._resampler_down.process(raw_frame)
            else:
                internal_frame = raw_frame

            # 2. Get AEC reference (last output at device rate, downsampled)
            if self._device is not None and self._aec is not None:
                ref_device = self._device.get_last_output_frame()
                if self._resampler_down is not None:
                    ref_internal = self._resampler_down.process(ref_device)
                else:
                    ref_internal = ref_device

                # 3. Apply AEC
                cleaned = self._aec.cancel_echo(internal_frame, ref_internal)
            else:
                cleaned = internal_frame

            # 4. Dispatch to all consumers
            with self._consumer_lock:
                for consumer in self._mic_consumers:
                    try:
                        consumer(cleaned)
                    except Exception:
                        pass  # Never crash the audio thread

        except Exception as e:
            logger.debug(f"[AudioBus] Mic frame processing error: {e}")

    # ---- Properties ----

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def device(self) -> Optional[FullDuplexDevice]:
        return self._device

    @property
    def config(self) -> Optional[DeviceConfig]:
        return self._config

    def get_status(self) -> dict:
        """Get current audio bus status."""
        return {
            "running": self._running,
            "device_running": self._device.is_running if self._device else False,
            "mic_consumers": len(self._mic_consumers),
            "sinks": list(self._sinks.keys()),
            "playback_buffered": (
                self._device.playback_buffer.available
                if self._device else 0
            ),
            "aec_type": (
                "speexdsp" if self._aec and self._aec._use_speexdsp
                else "spectral_subtraction" if self._aec else "none"
            ),
            "mic_gate_active": self._mic_gate_active,  # v265.0
        }


# ============================================================================
# Module-level access
# ============================================================================

def get_audio_bus() -> AudioBus:
    """Get the AudioBus singleton."""
    return AudioBus.get_instance()


def get_audio_bus_safe() -> Optional[AudioBus]:
    """Get the AudioBus singleton if it exists and is running."""
    bus = AudioBus.get_instance_safe()
    if bus is not None and bus.is_running:
        return bus
    return bus
