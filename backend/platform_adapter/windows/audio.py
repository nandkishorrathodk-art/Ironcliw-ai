"""
Ironcliw Windows Audio Engine Implementation
═══════════════════════════════════════════════════════════════════════════════

Windows implementation of audio processing using C# AudioEngine DLL with WASAPI.

Features:
    - Audio device enumeration (input/output)
    - Audio recording (WASAPI capture)
    - Audio playback (WASAPI render)
    - Default device management
    - Sample rate conversion

C# DLL Methods Used:
    - AudioEngine.GetInputDevices()
    - AudioEngine.GetOutputDevices()
    - AudioEngine.GetDefaultInputDevice()
    - AudioEngine.GetDefaultOutputDevice()
    - AudioEngine.StartRecording(callback, sampleRate, bitDepth)
    - AudioEngine.StopRecording()
    - AudioEngine.PlayAudio(audioData, sampleRate, bitDepth)

Author: Ironcliw System
Version: 1.0.0 (Windows Port)
"""
from __future__ import annotations

import os
import io
from typing import List, Optional, Callable
from pathlib import Path

try:
    import clr
except ImportError:
    raise ImportError(
        "pythonnet (clr) is not installed. Install with: pip install pythonnet"
    )

from ..base import (
    BaseAudioEngine,
    AudioDeviceInfo,
)


class WindowsAudioEngine(BaseAudioEngine):
    """Windows implementation of audio engine using C# WASAPI DLL"""
    
    def __init__(self):
        """Initialize Windows audio engine with C# DLL"""
        self._engine = None
        self._recording_buffer = bytearray()
        self._is_recording = False
        self._recording_callback = None
        self._load_native_dll()
    
    def _load_native_dll(self):
        """Load C# AudioEngine DLL"""
        dll_path = os.environ.get(
            'WINDOWS_NATIVE_DLL_PATH',
            str(Path(__file__).parent.parent.parent / 'windows_native' / 'bin' / 'Release')
        )
        
        dll_file = Path(dll_path) / 'AudioEngine.dll'
        
        if not dll_file.exists():
            raise FileNotFoundError(
                f"AudioEngine.dll not found at: {dll_file}\n"
                f"Please build the C# project first:\n"
                f"  cd backend/windows_native\n"
                f"  .\\build.ps1"
            )
        
        try:
            clr.AddReference(str(dll_file.resolve()))
            from JarvisWindowsNative.AudioEngine import AudioEngine
            self._engine = AudioEngine()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load AudioEngine.dll: {e}\n"
                f"Make sure .NET Runtime and NAudio package are installed."
            ) from e
    
    def list_devices(self, input_only: bool = False) -> List[AudioDeviceInfo]:
        """List available audio devices"""
        try:
            devices = []
            
            if input_only:
                input_devices = self._engine.GetInputDevices()
                for dev in input_devices:
                    devices.append(AudioDeviceInfo(
                        device_id=dev.Id,
                        name=dev.Name,
                        is_input=True,
                        is_default=dev.IsDefault,
                        sample_rate=16000,
                        channels=1,
                    ))
            else:
                input_devices = self._engine.GetInputDevices()
                for dev in input_devices:
                    devices.append(AudioDeviceInfo(
                        device_id=dev.Id,
                        name=dev.Name,
                        is_input=True,
                        is_default=dev.IsDefault,
                        sample_rate=16000,
                        channels=1,
                    ))
                
                output_devices = self._engine.GetOutputDevices()
                for dev in output_devices:
                    devices.append(AudioDeviceInfo(
                        device_id=dev.Id,
                        name=dev.Name,
                        is_input=False,
                        is_default=dev.IsDefault,
                        sample_rate=16000,
                        channels=2,
                    ))
            
            return devices
        except Exception as e:
            print(f"Warning: Failed to list audio devices: {e}")
            return []
    
    def get_default_input_device(self) -> Optional[AudioDeviceInfo]:
        """Get default microphone device"""
        try:
            dev = self._engine.GetDefaultInputDevice()
            if dev is None:
                return None
            
            return AudioDeviceInfo(
                device_id=dev.Id,
                name=dev.Name,
                is_input=True,
                is_default=True,
                sample_rate=16000,
                channels=1,
            )
        except Exception as e:
            print(f"Warning: Failed to get default input device: {e}")
            return None
    
    def get_default_output_device(self) -> Optional[AudioDeviceInfo]:
        """Get default speaker device"""
        try:
            dev = self._engine.GetDefaultOutputDevice()
            if dev is None:
                return None
            
            return AudioDeviceInfo(
                device_id=dev.Id,
                name=dev.Name,
                is_input=False,
                is_default=True,
                sample_rate=16000,
                channels=2,
            )
        except Exception as e:
            print(f"Warning: Failed to get default output device: {e}")
            return None
    
    def start_recording(self, device_id: Optional[str] = None, 
                       sample_rate: int = 16000, 
                       channels: int = 1,
                       callback: Optional[Callable] = None) -> bool:
        """Start audio recording"""
        try:
            if self._is_recording:
                self.stop_recording()
            
            self._recording_buffer.clear()
            self._recording_callback = callback
            self._is_recording = True
            
            def data_callback(data: bytes):
                """Callback for audio data from C#"""
                if self._is_recording:
                    self._recording_buffer.extend(data)
                    if self._recording_callback:
                        try:
                            self._recording_callback(data)
                        except Exception as e:
                            print(f"Warning: Recording callback error: {e}")
            
            import System
            callback_delegate = System.Action[System.Array[System.Byte]](data_callback)
            
            success = self._engine.StartRecording(callback_delegate, sample_rate, 16)
            
            if not success:
                self._is_recording = False
            
            return success
        except Exception as e:
            print(f"Warning: Failed to start recording: {e}")
            self._is_recording = False
            return False
    
    def stop_recording(self) -> Optional[bytes]:
        """Stop recording and return audio data"""
        try:
            if not self._is_recording:
                return None
            
            self._engine.StopRecording()
            self._is_recording = False
            
            data = bytes(self._recording_buffer)
            self._recording_buffer.clear()
            
            return data if len(data) > 0 else None
        except Exception as e:
            print(f"Warning: Failed to stop recording: {e}")
            self._is_recording = False
            return None
    
    def play_audio(self, audio_data: bytes, sample_rate: int = 16000) -> bool:
        """Play audio data"""
        try:
            import System
            
            byte_array = System.Array[System.Byte](list(audio_data))
            
            success = self._engine.PlayAudio(byte_array, sample_rate, 16)
            
            return success
        except Exception as e:
            print(f"Warning: Failed to play audio: {e}")
            return False
    
    def is_recording(self) -> bool:
        """Check if currently recording"""
        return self._is_recording
