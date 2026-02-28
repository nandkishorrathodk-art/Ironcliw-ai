"""
System Capabilities Detection
==============================

Runtime detection of available system features and hardware capabilities.
Used to determine which features can be enabled on the current platform.
"""

import platform
import sys
from typing import Dict, Optional, List


class SystemCapabilities:
    """
    Runtime detection of available system features.
    
    Detects:
    - Operating system and architecture
    - GPU availability and type (Metal, CUDA, DirectML, etc.)
    - Audio backend capabilities
    - Native extension availability (Swift, Rust)
    - Development vs. production environment
    """
    
    def __init__(self):
        self.os = platform.system()
        self.arch = platform.machine()
        self.python_version = sys.version_info
        
        # Core capabilities
        self.has_gpu = self._detect_gpu()
        self.gpu_type = self._get_gpu_type()
        self.has_neural_engine = self._detect_neural_engine()
        self.has_swift = self._check_swift_runtime()
        self.has_rust_extensions = self._check_rust_extensions()
        self.audio_backend = self._detect_audio_backend()
        
        # Environment detection
        self.is_dev_mode = self._detect_dev_mode()
        self.is_docker = self._detect_docker()
        self.is_wsl = self._detect_wsl()
    
    def _detect_gpu(self) -> bool:
        """Detect if any GPU is available."""
        if self.os == "Darwin":
            # macOS always has GPU (integrated or discrete)
            return True
        
        # Try to detect CUDA
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass
        
        # Try to detect DirectML on Windows
        if self.os == "Windows":
            try:
                import onnxruntime as ort
                return "DmlExecutionProvider" in ort.get_available_providers()
            except ImportError:
                pass
        
        return False
    
    def _get_gpu_type(self) -> str:
        """
        Determine GPU type.
        
        Returns:
            str: "metal", "cuda", "directml", "rocm", or "cpu"
        """
        if self.os == "Darwin":
            # macOS uses Metal
            return "metal"
        
        if self.os == "Windows":
            # Check for CUDA first (NVIDIA)
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            
            # Fall back to DirectML (Windows ML)
            try:
                import onnxruntime as ort
                if "DmlExecutionProvider" in ort.get_available_providers():
                    return "directml"
            except ImportError:
                pass
        
        if self.os == "Linux":
            # Check for CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
            
            # Check for ROCm (AMD)
            try:
                import torch
                if hasattr(torch, "hip") and torch.hip.is_available():
                    return "rocm"
            except (ImportError, AttributeError):
                pass
        
        return "cpu"
    
    def _detect_neural_engine(self) -> bool:
        """Detect Apple Neural Engine (macOS only)."""
        if self.os != "Darwin":
            return False
        
        # Neural Engine available on Apple Silicon (arm64)
        if "arm64" in self.arch.lower() or "aarch64" in self.arch.lower():
            # Check if CoreML is available
            try:
                import coremltools
                return True
            except ImportError:
                pass
        
        return False
    
    def _check_swift_runtime(self) -> bool:
        """Check if Swift runtime/extensions are available."""
        if self.os != "Darwin":
            return False
        
        try:
            import importlib.util
            spec = importlib.util.find_spec("backend.swift_bridge")
            return spec is not None
        except (ImportError, AttributeError):
            return False
    
    def _check_rust_extensions(self) -> bool:
        """Check if Rust performance extensions are available."""
        try:
            import importlib.util
            spec = importlib.util.find_spec("backend.rust_performance")
            if spec is not None:
                return True
            
            spec = importlib.util.find_spec("jarvis_rust_performance")
            return spec is not None
        except (ImportError, AttributeError):
            return False
    
    def _detect_audio_backend(self) -> str:
        """
        Detect available audio backend.
        
        Returns:
            str: "coreaudio" (macOS), "wasapi" (Windows), 
                 "alsa"/"pulseaudio"/"pipewire" (Linux), or "none"
        """
        if self.os == "Darwin":
            return "coreaudio"
        elif self.os == "Windows":
            return "wasapi"
        elif self.os == "Linux":
            # Try to detect Linux audio system
            try:
                import subprocess
                result = subprocess.run(
                    ["pactl", "info"],
                    capture_output=True,
                    timeout=1,
                )
                if result.returncode == 0:
                    if b"PipeWire" in result.stdout:
                        return "pipewire"
                    return "pulseaudio"
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
            
            # Fall back to ALSA
            return "alsa"
        
        return "none"
    
    def _detect_dev_mode(self) -> bool:
        """Detect if running in development mode."""
        import os
        
        # Check environment variables
        if os.getenv("Ironcliw_DEV_MODE", "").lower() == "true":
            return True
        if os.getenv("DEBUG", "").lower() == "true":
            return True
        if os.getenv("ENVIRONMENT", "").lower() in ("dev", "development"):
            return True
        
        # Check if running from source (not installed package)
        try:
            import __main__
            if hasattr(__main__, "__file__"):
                main_file = __main__.__file__
                if main_file and ("unified_supervisor.py" in main_file or "main.py" in main_file):
                    return True
        except:
            pass
        
        return False
    
    def _detect_docker(self) -> bool:
        """Detect if running inside Docker container."""
        import os
        from pathlib import Path
        
        # Check for .dockerenv file
        if Path("/.dockerenv").exists():
            return True
        
        # Check cgroup
        try:
            with open("/proc/self/cgroup", "r") as f:
                return "docker" in f.read()
        except:
            pass
        
        return False
    
    def _detect_wsl(self) -> bool:
        """Detect if running in Windows Subsystem for Linux."""
        if self.os != "Linux":
            return False
        
        try:
            with open("/proc/version", "r") as f:
                version = f.read().lower()
                return "microsoft" in version or "wsl" in version
        except:
            pass
        
        return False
    
    def get_ml_config(self) -> Dict[str, any]:
        """
        Get recommended ML configuration for this platform.
        
        Returns:
            dict: Configuration for ML models with keys:
                - gpu_layers: Number of layers to offload to GPU (-1 = all)
                - context_size: Recommended context window
                - batch_size: Recommended batch size
                - threads: Recommended CPU threads
        """
        if self.gpu_type == "metal":
            return {
                "gpu_layers": -1,  # All layers on GPU
                "context_size": 4096,
                "batch_size": 512,
                "threads": 8,
            }
        elif self.gpu_type == "cuda":
            return {
                "gpu_layers": -1,
                "context_size": 4096,
                "batch_size": 512,
                "threads": 8,
            }
        elif self.gpu_type == "directml":
            return {
                "gpu_layers": -1,
                "context_size": 2048,  # DirectML may have more limitations
                "batch_size": 256,
                "threads": 4,
            }
        else:
            # CPU fallback
            return {
                "gpu_layers": 0,
                "context_size": 2048,
                "batch_size": 128,
                "threads": 4,
            }
    
    def get_execution_providers(self) -> List[str]:
        """
        Get ONNX Runtime execution providers in priority order.
        
        Returns:
            list: Ordered list of execution provider names
        """
        providers = []
        
        if self.os == "Darwin":
            if self.has_neural_engine:
                providers.append("CoreMLExecutionProvider")
        
        if self.gpu_type == "cuda":
            providers.append("CUDAExecutionProvider")
        elif self.gpu_type == "directml":
            providers.append("DmlExecutionProvider")
        elif self.gpu_type == "rocm":
            providers.append("ROCMExecutionProvider")
        
        # Always include CPU as fallback
        providers.append("CPUExecutionProvider")
        
        return providers
    
    def to_dict(self) -> Dict[str, any]:
        """
        Export capabilities as dictionary.
        
        Returns:
            dict: All capability flags and detected features
        """
        return {
            "os": self.os,
            "arch": self.arch,
            "python_version": f"{self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            "has_gpu": self.has_gpu,
            "gpu_type": self.gpu_type,
            "has_neural_engine": self.has_neural_engine,
            "has_swift": self.has_swift,
            "has_rust_extensions": self.has_rust_extensions,
            "audio_backend": self.audio_backend,
            "is_dev_mode": self.is_dev_mode,
            "is_docker": self.is_docker,
            "is_wsl": self.is_wsl,
            "ml_config": self.get_ml_config(),
            "execution_providers": self.get_execution_providers(),
        }
    
    def __repr__(self) -> str:
        """String representation of capabilities."""
        return (
            f"SystemCapabilities(\n"
            f"  OS: {self.os} {self.arch}\n"
            f"  GPU: {self.gpu_type.upper()}\n"
            f"  Audio: {self.audio_backend}\n"
            f"  Swift: {'✓' if self.has_swift else '✗'}\n"
            f"  Rust: {'✓' if self.has_rust_extensions else '✗'}\n"
            f"  Dev Mode: {'✓' if self.is_dev_mode else '✗'}\n"
            f")"
        )


# Global singleton instance
_capabilities: Optional[SystemCapabilities] = None


def get_capabilities() -> SystemCapabilities:
    """
    Get global capabilities instance (singleton).
    
    Returns:
        SystemCapabilities: Cached capabilities instance
    """
    global _capabilities
    if _capabilities is None:
        _capabilities = SystemCapabilities()
    return _capabilities
