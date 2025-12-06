#!/usr/bin/env python3
"""
Audio Format Converter
Ensures audio data is in the correct format for STT processing.

CRITICAL FIX: Handles WebM, MP3, OGG, and other compressed formats
that browsers send. Uses pydub + FFmpeg for proper transcoding.
"""

import base64
import numpy as np
import logging
import json
import struct
import wave
import io
import tempfile
import os
import subprocess
import shutil
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Check FFmpeg availability at module load
FFMPEG_AVAILABLE = shutil.which('ffmpeg') is not None
PYDUB_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    logger.warning("pydub not installed - falling back to basic audio handling")

# Audio format magic bytes for detection
# Note: Order matters - more specific signatures should come first
AUDIO_SIGNATURES = {
    b'\x1a\x45\xdf\xa3': 'webm',      # WebM/Matroska
    b'OggS': 'ogg',                    # OGG Vorbis/Opus
    b'\xff\xfb': 'mp3',                # MP3 (MPEG Audio Layer 3)
    b'\xff\xfa': 'mp3',                # MP3 variant
    b'\xff\xf3': 'mp3',                # MP3 variant
    b'\xff\xf2': 'mp3',                # MP3 variant
    b'ID3': 'mp3',                     # MP3 with ID3 tag
    b'RIFF': 'wav',                    # WAV
    b'fLaC': 'flac',                   # FLAC
    # Note: MP4/M4A detection is handled separately in detect_audio_format()
    # because the 'ftyp' box appears at byte 4, not at the start
}

class AudioFormatConverter:
    """Convert various audio formats to standard format for STT"""

    @staticmethod
    def convert_to_bytes(audio_data) -> bytes:
        """
        Convert any audio format to proper bytes for processing.

        Args:
            audio_data: Audio in any format (string, base64, bytes, list, etc.)

        Returns:
            bytes: PCM audio data as bytes
        """

        if audio_data is None:
            logger.warning("Audio data is None, returning empty bytes")
            return b''

        # Already bytes - validate it's proper audio
        if isinstance(audio_data, bytes):
            logger.debug(f"Audio is already bytes: {len(audio_data)} bytes")
            return audio_data

        # Base64 encoded string
        if isinstance(audio_data, str):
            # Check if it's JSON
            if audio_data.startswith('{') or audio_data.startswith('['):
                try:
                    data = json.loads(audio_data)
                    if isinstance(data, list):
                        # Array of samples
                        return AudioFormatConverter._array_to_bytes(data)
                    elif isinstance(data, dict) and 'data' in data:
                        # Nested data object
                        return AudioFormatConverter.convert_to_bytes(data['data'])
                except:
                    pass

            # Try base64 decoding
            try:
                # Standard base64
                decoded = base64.b64decode(audio_data)
                logger.debug(f"Decoded base64: {len(decoded)} bytes")
                return decoded
            except:
                pass

            # Try base64 with padding fix
            try:
                # Add padding if needed
                padding = 4 - len(audio_data) % 4
                if padding != 4:
                    audio_data += '=' * padding
                decoded = base64.b64decode(audio_data)
                logger.debug(f"Decoded base64 with padding: {len(decoded)} bytes")
                return decoded
            except:
                pass

            # URL-safe base64
            try:
                decoded = base64.urlsafe_b64decode(audio_data)
                logger.debug(f"Decoded URL-safe base64: {len(decoded)} bytes")
                return decoded
            except:
                pass

            # Hex string
            try:
                decoded = bytes.fromhex(audio_data)
                logger.debug(f"Decoded hex: {len(decoded)} bytes")
                return decoded
            except:
                pass

            # Comma-separated values
            if ',' in audio_data:
                try:
                    values = [int(x.strip()) for x in audio_data.split(',')]
                    return AudioFormatConverter._array_to_bytes(values)
                except:
                    pass

            logger.warning(f"Could not decode string of length {len(audio_data)}, returning as string for now")
            # BUGFIX: We were returning the string here which caused type errors!
            # Now we return it so prepare_audio_for_stt can handle it
            return audio_data

        # List or array of samples
        if isinstance(audio_data, (list, tuple)):
            return AudioFormatConverter._array_to_bytes(audio_data)

        # NumPy array
        if hasattr(audio_data, 'dtype'):
            return AudioFormatConverter._numpy_to_bytes(audio_data)

        # Dictionary with audio data
        if isinstance(audio_data, dict):
            # Check common keys
            for key in ['data', 'audio', 'audio_data', 'samples', 'buffer']:
                if key in audio_data:
                    return AudioFormatConverter.convert_to_bytes(audio_data[key])

        # Try to convert to bytes directly
        try:
            return bytes(audio_data)
        except:
            logger.error(f"Cannot convert audio data of type {type(audio_data)}")
            return b''

    @staticmethod
    def _array_to_bytes(array) -> bytes:
        """Convert array of samples to bytes"""
        try:
            # Determine data type
            if all(isinstance(x, float) for x in array[:100]):
                # Float samples (-1.0 to 1.0)
                samples = np.array(array, dtype=np.float32)
                # Convert to int16
                samples = (samples * 32767).astype(np.int16)
            else:
                # Integer samples
                samples = np.array(array, dtype=np.int16)

            logger.debug(f"Converted array of {len(samples)} samples to bytes")
            return samples.tobytes()
        except Exception as e:
            logger.error(f"Failed to convert array to bytes: {e}")
            return b''

    @staticmethod
    def _numpy_to_bytes(arr) -> bytes:
        """Convert numpy array to bytes"""
        try:
            if arr.dtype in [np.float32, np.float64]:
                # Convert float to int16
                arr = (arr * 32767).astype(np.int16)
            elif arr.dtype != np.int16:
                # Convert to int16
                arr = arr.astype(np.int16)

            logger.debug(f"Converted numpy array {arr.shape} to bytes")
            return arr.tobytes()
        except Exception as e:
            logger.error(f"Failed to convert numpy array to bytes: {e}")
            return b''

    @staticmethod
    def detect_audio_format(audio_bytes: bytes) -> str:
        """
        Detect the audio format from magic bytes.

        Returns:
            str: Format name ('webm', 'mp3', 'ogg', 'wav', 'flac', 'mp4', 'raw_pcm')
        """
        if not audio_bytes or len(audio_bytes) < 4:
            return 'raw_pcm'

        # Check magic bytes
        for signature, format_name in AUDIO_SIGNATURES.items():
            if audio_bytes.startswith(signature):
                return format_name

        # Special check for MP4/M4A (ftyp usually at byte 4)
        if len(audio_bytes) > 8 and audio_bytes[4:8] == b'ftyp':
            return 'mp4'

        # If no match, assume raw PCM
        return 'raw_pcm'

    @staticmethod
    def transcode_with_pydub(audio_bytes: bytes, detected_format: str,
                             sample_rate: int = 16000) -> Optional[bytes]:
        """
        Transcode audio to PCM using pydub + FFmpeg.

        Args:
            audio_bytes: Compressed audio bytes
            detected_format: Format name from detect_audio_format()
            sample_rate: Target sample rate

        Returns:
            bytes: Raw PCM bytes (16-bit, mono, target sample rate) or None on failure
        """
        if not PYDUB_AVAILABLE:
            logger.warning("pydub not available for transcoding")
            return None

        if not FFMPEG_AVAILABLE:
            logger.warning("FFmpeg not available for transcoding")
            return None

        try:
            # Map format names to pydub format strings
            format_map = {
                'webm': 'webm',
                'ogg': 'ogg',
                'mp3': 'mp3',
                'flac': 'flac',
                'mp4': 'mp4',
                'm4a': 'm4a',
            }

            pydub_format = format_map.get(detected_format, detected_format)

            # Create temp file for input
            with tempfile.NamedTemporaryFile(suffix=f'.{pydub_format}', delete=False) as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_in_path = tmp_in.name

            try:
                # Load audio with pydub
                audio = AudioSegment.from_file(tmp_in_path, format=pydub_format)

                # Convert to mono, target sample rate, 16-bit
                audio = audio.set_channels(1)
                audio = audio.set_frame_rate(sample_rate)
                audio = audio.set_sample_width(2)  # 16-bit = 2 bytes

                # Get raw PCM data
                pcm_bytes = audio.raw_data

                logger.info(f"Transcoded {detected_format} to PCM: {len(audio_bytes)} -> {len(pcm_bytes)} bytes, "
                           f"{audio.frame_rate}Hz, {audio.channels}ch, {audio.sample_width*8}bit")

                return pcm_bytes

            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_in_path)
                except:
                    pass

        except Exception as e:
            logger.error(f"pydub transcode failed for {detected_format}: {e}")
            return None

    @staticmethod
    def transcode_with_ffmpeg_direct(audio_bytes: bytes, detected_format: str,
                                     sample_rate: int = 16000) -> Optional[bytes]:
        """
        Fallback: Transcode using FFmpeg directly via subprocess.

        Args:
            audio_bytes: Compressed audio bytes
            detected_format: Format name
            sample_rate: Target sample rate

        Returns:
            bytes: Raw PCM bytes or None on failure
        """
        if not FFMPEG_AVAILABLE:
            logger.warning("FFmpeg not available")
            return None

        try:
            # Create temp files
            suffix = f'.{detected_format}' if detected_format != 'raw_pcm' else '.bin'
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
                tmp_in.write(audio_bytes)
                tmp_in_path = tmp_in.name

            tmp_out_path = tmp_in_path + '.wav'

            try:
                # Run FFmpeg to convert to WAV
                cmd = [
                    'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                    '-i', tmp_in_path,
                    '-ar', str(sample_rate),  # Sample rate
                    '-ac', '1',                # Mono
                    '-acodec', 'pcm_s16le',   # 16-bit PCM
                    '-f', 'wav',
                    tmp_out_path
                ]

                result = subprocess.run(cmd, capture_output=True, timeout=30)

                if result.returncode != 0:
                    logger.error(f"FFmpeg failed: {result.stderr.decode()}")
                    return None

                # Read the output WAV and extract PCM
                with open(tmp_out_path, 'rb') as f:
                    wav_data = f.read()

                # Extract PCM from WAV (skip 44-byte header)
                if wav_data.startswith(b'RIFF'):
                    with io.BytesIO(wav_data) as wav_io:
                        with wave.open(wav_io, 'rb') as wav:
                            pcm_bytes = wav.readframes(wav.getnframes())
                            logger.info(f"FFmpeg transcoded {detected_format}: {len(audio_bytes)} -> {len(pcm_bytes)} bytes")
                            return pcm_bytes

                return None

            finally:
                # Clean up temp files
                for path in [tmp_in_path, tmp_out_path]:
                    try:
                        os.unlink(path)
                    except:
                        pass

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg transcode timed out")
            return None
        except Exception as e:
            logger.error(f"FFmpeg direct transcode failed: {e}")
            return None

    @staticmethod
    def ensure_pcm_format(audio_bytes: bytes, sample_rate: int = 16000) -> bytes:
        """
        Ensure audio bytes are in PCM format suitable for STT and biometric processing.

        CRITICAL: This method now properly handles compressed formats (WebM, MP3, OGG)
        that browsers send. Previously, compressed audio was passed through as-is,
        causing voice biometric failures (0% confidence).

        Args:
            audio_bytes: Raw or compressed audio bytes
            sample_rate: Target sample rate (default 16000 Hz)

        Returns:
            bytes: PCM audio data at 16kHz, 16-bit, mono
        """
        if not audio_bytes:
            return b''

        # Detect the audio format
        detected_format = AudioFormatConverter.detect_audio_format(audio_bytes)
        logger.debug(f"Detected audio format: {detected_format} ({len(audio_bytes)} bytes)")

        # If it's already a WAV file, extract the PCM data
        if detected_format == 'wav':
            try:
                with io.BytesIO(audio_bytes) as wav_io:
                    with wave.open(wav_io, 'rb') as wav:
                        # Check if resampling/conversion is needed
                        if wav.getnchannels() == 1 and wav.getframerate() == sample_rate and wav.getsampwidth() == 2:
                            # Already in the right format, just extract frames
                            return wav.readframes(wav.getnframes())
                        else:
                            # Need to resample/convert - use pydub
                            if PYDUB_AVAILABLE:
                                audio = AudioSegment.from_wav(wav_io)
                                audio = audio.set_channels(1).set_frame_rate(sample_rate).set_sample_width(2)
                                return audio.raw_data
                            else:
                                # Fallback: just extract frames as-is
                                return wav.readframes(wav.getnframes())
            except Exception as e:
                logger.warning(f"WAV parsing failed, will try transcoding: {e}")

        # If it's a compressed format, transcode to PCM
        if detected_format in ['webm', 'ogg', 'mp3', 'flac', 'mp4', 'm4a']:
            logger.info(f"Transcoding {detected_format} audio to PCM ({len(audio_bytes)} bytes)")

            # Try pydub first (faster, more reliable)
            pcm_data = AudioFormatConverter.transcode_with_pydub(audio_bytes, detected_format, sample_rate)
            if pcm_data:
                return pcm_data

            # Fallback to direct FFmpeg
            pcm_data = AudioFormatConverter.transcode_with_ffmpeg_direct(audio_bytes, detected_format, sample_rate)
            if pcm_data:
                return pcm_data

            logger.error(f"Failed to transcode {detected_format} audio - returning empty bytes")
            return b''

        # If it's raw PCM, validate and return
        if detected_format == 'raw_pcm':
            # PCM should have even number of bytes (16-bit samples)
            if len(audio_bytes) % 2 != 0:
                audio_bytes = audio_bytes + b'\x00'
            return audio_bytes

        # Unknown format - try transcoding anyway
        logger.warning(f"Unknown audio format, attempting transcode: {detected_format}")
        pcm_data = AudioFormatConverter.transcode_with_pydub(audio_bytes, detected_format, sample_rate)
        if pcm_data:
            return pcm_data

        # Last resort - return as-is (may cause issues)
        logger.warning("Could not transcode audio - returning as-is")
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes + b'\x00'
        return audio_bytes

    @staticmethod
    def create_wav_header(audio_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
        """
        Add WAV header to raw PCM data.

        Args:
            audio_bytes: Raw PCM audio data
            sample_rate: Sample rate in Hz
            channels: Number of channels

        Returns:
            bytes: Complete WAV file with header
        """

        # WAV header parameters
        bits_per_sample = 16
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        data_size = len(audio_bytes)
        file_size = data_size + 44 - 8  # Total file size minus RIFF header

        # Create WAV header
        header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',           # ChunkID
            file_size,         # ChunkSize
            b'WAVE',           # Format
            b'fmt ',           # Subchunk1ID
            16,                # Subchunk1Size (16 for PCM)
            1,                 # AudioFormat (1 for PCM)
            channels,          # NumChannels
            sample_rate,       # SampleRate
            byte_rate,         # ByteRate
            block_align,       # BlockAlign
            bits_per_sample,   # BitsPerSample
            b'data',           # Subchunk2ID
            data_size          # Subchunk2Size
        )

        return header + audio_bytes


# Global converter instance
audio_converter = AudioFormatConverter()

def prepare_audio_for_stt(audio_data) -> bytes:
    """
    Prepare any audio format for STT processing.

    Args:
        audio_data: Audio in any format

    Returns:
        bytes: PCM audio ready for STT
    """

    # Convert to bytes
    audio_bytes = audio_converter.convert_to_bytes(audio_data)

    # CRITICAL FIX: Ensure we ALWAYS return bytes, never a string
    if isinstance(audio_bytes, str):
        logger.error(f"convert_to_bytes returned a string instead of bytes! Attempting base64 decode...")
        try:
            # Try standard base64
            audio_bytes = base64.b64decode(audio_bytes)
            logger.info(f"Emergency base64 decode successful: {len(audio_bytes)} bytes")
        except Exception as e:
            logger.error(f"Emergency base64 decode failed: {e}")
            # Return empty bytes to avoid type error
            return b''

    # If we got empty bytes, return them (will fail gracefully later)
    if not audio_bytes:
        logger.warning("No audio data after conversion - returning empty bytes")
        return b''

    # Ensure PCM format
    pcm_bytes = audio_converter.ensure_pcm_format(audio_bytes)

    logger.info(f"✅ Prepared audio: {len(pcm_bytes)} bytes of PCM data")
    return pcm_bytes