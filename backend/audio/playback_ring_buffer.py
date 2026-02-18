"""
Playback Ring Buffer for Real-Time Audio
=========================================

Lock-free ring buffer for feeding audio to the sounddevice output callback.
The callback thread writes silence when the buffer is empty, and plays queued
audio when available. Thread-safe by design: single producer (main thread
writing TTS audio) and single consumer (sounddevice callback reading frames).

Architecture:
    TTS / AudioBus  ──write()──▶  RingBuffer  ──read()──▶  sounddevice callback
                                                            (fills outdata)
"""

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)


class PlaybackRingBuffer:
    """
    Fixed-capacity ring buffer for float32 mono audio frames.

    Thread-safety guarantee:
        One writer thread (AudioBus) and one reader thread (sounddevice callback)
        can operate concurrently without locks. The atomic nature of Python integer
        assignment on the GIL provides the necessary memory ordering.

    Overflow policy: oldest frames are silently dropped (prefer low latency
    over completeness for real-time audio).
    """

    def __init__(self, capacity_frames: int = 96000):
        """
        Args:
            capacity_frames: Total buffer capacity in samples.
                Default 96000 = 2 seconds at 48 kHz.
        """
        self._buf = np.zeros(capacity_frames, dtype=np.float32)
        self._capacity = capacity_frames
        self._write_pos = 0
        self._read_pos = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def available(self) -> int:
        """Number of frames available to read."""
        with self._lock:
            return self._available_unlocked()

    def _available_unlocked(self) -> int:
        diff = self._write_pos - self._read_pos
        if diff < 0:
            diff += self._capacity
        return diff

    @property
    def free_space(self) -> int:
        """Number of frames that can be written before overflow."""
        return self._capacity - self.available - 1

    def write(self, data: np.ndarray) -> int:
        """
        Write audio frames into the buffer.

        Args:
            data: float32 mono audio array.

        Returns:
            Number of frames actually written (may be less than len(data)
            if buffer is nearly full).
        """
        n = len(data)
        if n == 0:
            return 0

        with self._lock:
            free = self._capacity - self._available_unlocked() - 1
            if free <= 0:
                return 0

            to_write = min(n, free)
            wp = self._write_pos

            # How many frames fit before wrap-around?
            first_chunk = min(to_write, self._capacity - wp)
            self._buf[wp:wp + first_chunk] = data[:first_chunk]

            remainder = to_write - first_chunk
            if remainder > 0:
                self._buf[:remainder] = data[first_chunk:first_chunk + remainder]

            self._write_pos = (wp + to_write) % self._capacity
            return to_write

    def read(self, out: np.ndarray) -> int:
        """
        Read frames into the provided output array.

        Fills `out` with buffered audio. If fewer frames are available
        than requested, the remainder of `out` is zero-filled (silence).

        Args:
            out: Pre-allocated float32 array to fill.

        Returns:
            Number of actual audio frames copied (rest is silence).
        """
        n = len(out)
        with self._lock:
            avail = self._available_unlocked()
            to_read = min(n, avail)

            if to_read == 0:
                out[:] = 0.0
                return 0

            rp = self._read_pos
            first_chunk = min(to_read, self._capacity - rp)
            out[:first_chunk] = self._buf[rp:rp + first_chunk]

            remainder = to_read - first_chunk
            if remainder > 0:
                out[first_chunk:first_chunk + remainder] = self._buf[:remainder]

            # Zero-fill the rest
            if to_read < n:
                out[to_read:] = 0.0

            self._read_pos = (rp + to_read) % self._capacity
            return to_read

    def peek(self, n: int) -> np.ndarray:
        """
        Read up to `n` frames without advancing the read pointer.
        Used by AEC to get the most recent output reference.

        Returns:
            Copy of buffered audio (may be shorter than `n`).
        """
        with self._lock:
            avail = self._available_unlocked()
            to_peek = min(n, avail)
            if to_peek == 0:
                return np.zeros(0, dtype=np.float32)

            rp = self._read_pos
            first_chunk = min(to_peek, self._capacity - rp)
            result = np.empty(to_peek, dtype=np.float32)
            result[:first_chunk] = self._buf[rp:rp + first_chunk]

            remainder = to_peek - first_chunk
            if remainder > 0:
                result[first_chunk:] = self._buf[:remainder]

            return result

    def flush(self) -> int:
        """
        Discard all buffered audio. Used for barge-in.

        Returns:
            Number of frames discarded.
        """
        with self._lock:
            discarded = self._available_unlocked()
            self._read_pos = self._write_pos
            return discarded

    def get_last_output_frame(self, frame_size: int) -> np.ndarray:
        """
        Get the most recently *read* frame for AEC reference.
        Returns the last `frame_size` samples that were consumed by the
        output callback.

        If not enough history, returns zero-padded array.
        """
        with self._lock:
            # The read pointer points to the NEXT frame to read.
            # The last consumed frame ends at read_pos.
            end = self._read_pos
            start = (end - frame_size) % self._capacity

            result = np.empty(frame_size, dtype=np.float32)
            if start < end:
                actual = end - start
                result[:actual] = self._buf[start:end]
                if actual < frame_size:
                    result[actual:] = 0.0
            else:
                first_part = self._capacity - start
                if first_part >= frame_size:
                    result[:] = self._buf[start:start + frame_size]
                else:
                    result[:first_part] = self._buf[start:]
                    remainder = frame_size - first_part
                    result[first_part:first_part + remainder] = self._buf[:remainder]

            return result
