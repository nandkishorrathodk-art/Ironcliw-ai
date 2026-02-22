"""
JARVIS Platform Detection System
═══════════════════════════════════════════════════════════════════════════════

Runtime platform detection with comprehensive fallback and caching.

Supports:
    - macOS (Darwin)
    - Windows (Windows 10/11, Windows Server)
    - Linux (Ubuntu, Debian, Fedora, Arch, etc.)

Detection Strategy:
    1. Check platform.system() for OS family
    2. Verify OS version and capabilities
    3. Detect hardware features (GPU, NPU, etc.)
    4. Cache result for performance

Author: JARVIS System
Version: 1.0.0 (Windows Port)
"""
from __future__ import annotations

import os
import sys
import platform
import subprocess
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# PLATFORM INFO DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlatformInfo:
    """Comprehensive platform information"""
    os_family: str  # 'macos', 'windows', 'linux'
    os_name: str  # 'Darwin', 'Windows', 'Linux'
    os_version: str  # '14.2.1', '10.0.22631', '6.5.0-14'
    os_release: str  # 'macOS 14.2.1', 'Windows 11', 'Ubuntu 22.04'
    architecture: str  # 'arm64', 'x86_64', 'amd64'
    machine: str  # 'Apple M1', 'AMD Ryzen 7', 'Intel Core i7'
    python_version: str  # '3.11.7'
    home_dir: Path
    user_name: str
    
    # Hardware capabilities
    has_gpu: bool = False
    has_npu: bool = False
    has_metal: bool = False  # macOS Metal
    has_directml: bool = False  # Windows DirectML
    has_cuda: bool = False  # NVIDIA CUDA
    
    # Platform features
    supports_voice: bool = False
    supports_vision: bool = False
    supports_automation: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# PLATFORM DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class PlatformDetector:
    """Runtime platform detection with caching"""
    
    _cached_platform: Optional[str] = None
    _cached_info: Optional[PlatformInfo] = None
    
    @staticmethod
    def detect_platform() -> str:
        """
        Detect current platform
        
        Returns:
            'macos', 'windows', or 'linux'
        """
        if PlatformDetector._cached_platform:
            return PlatformDetector._cached_platform
        
        system = platform.system().lower()
        
        if system == 'darwin':
            PlatformDetector._cached_platform = 'macos'
        elif system == 'windows':
            PlatformDetector._cached_platform = 'windows'
        elif system == 'linux':
            PlatformDetector._cached_platform = 'linux'
        else:
            raise RuntimeError(
                f"Unsupported platform: {system}. "
                f"JARVIS supports macOS, Windows, and Linux only."
            )
        
        return PlatformDetector._cached_platform
    
    @staticmethod
    def get_platform_info() -> PlatformInfo:
        """
        Get comprehensive platform information
        
        Returns:
            PlatformInfo object with full system details
        """
        if PlatformDetector._cached_info:
            return PlatformDetector._cached_info
        
        os_family = PlatformDetector.detect_platform()
        
        # Basic info
        info = PlatformInfo(
            os_family=os_family,
            os_name=platform.system(),
            os_version=platform.version(),
            os_release=platform.platform(),
            architecture=platform.machine(),
            machine=platform.processor() or platform.machine(),
            python_version=platform.python_version(),
            home_dir=Path.home(),
            user_name=os.getenv('USER') or os.getenv('USERNAME') or 'unknown',
        )
        
        # Detect hardware capabilities
        if os_family == 'macos':
            info.has_metal = PlatformDetector._check_metal()
            info.has_gpu = info.has_metal  # macOS GPU = Metal
            info.has_npu = PlatformDetector._check_apple_neural_engine()
        elif os_family == 'windows':
            info.has_directml = PlatformDetector._check_directml()
            info.has_gpu = PlatformDetector._check_windows_gpu()
            info.has_npu = PlatformDetector._check_windows_npu()
            info.has_cuda = PlatformDetector._check_cuda()
        elif os_family == 'linux':
            info.has_gpu = PlatformDetector._check_linux_gpu()
            info.has_cuda = PlatformDetector._check_cuda()
        
        # Detect platform features
        info.supports_voice = True  # All platforms support voice now
        info.supports_vision = True  # All platforms support vision
        info.supports_automation = True  # All platforms support automation
        
        PlatformDetector._cached_info = info
        return info
    
    @staticmethod
    def is_macos() -> bool:
        """Check if running on macOS"""
        return PlatformDetector.detect_platform() == 'macos'
    
    @staticmethod
    def is_windows() -> bool:
        """Check if running on Windows"""
        return PlatformDetector.detect_platform() == 'windows'
    
    @staticmethod
    def is_linux() -> bool:
        """Check if running on Linux"""
        return PlatformDetector.detect_platform() == 'linux'
    
    @staticmethod
    def get_config_dir() -> Path:
        """Get platform-specific config directory"""
        home = Path.home()
        
        if PlatformDetector.is_macos():
            return home / '.jarvis'
        elif PlatformDetector.is_windows():
            appdata = os.getenv('APPDATA')
            if appdata:
                return Path(appdata) / 'JARVIS'
            return home / '.jarvis'
        else:  # Linux
            xdg_config = os.getenv('XDG_CONFIG_HOME')
            if xdg_config:
                return Path(xdg_config) / 'jarvis'
            return home / '.config' / 'jarvis'
    
    @staticmethod
    def get_data_dir() -> Path:
        """Get platform-specific data directory"""
        home = Path.home()
        
        if PlatformDetector.is_macos():
            return home / 'Library' / 'Application Support' / 'JARVIS'
        elif PlatformDetector.is_windows():
            appdata = os.getenv('LOCALAPPDATA')
            if appdata:
                return Path(appdata) / 'JARVIS'
            return home / '.jarvis' / 'data'
        else:  # Linux
            xdg_data = os.getenv('XDG_DATA_HOME')
            if xdg_data:
                return Path(xdg_data) / 'jarvis'
            return home / '.local' / 'share' / 'jarvis'
    
    @staticmethod
    def get_cache_dir() -> Path:
        """Get platform-specific cache directory"""
        home = Path.home()
        
        if PlatformDetector.is_macos():
            return home / 'Library' / 'Caches' / 'JARVIS'
        elif PlatformDetector.is_windows():
            temp = os.getenv('TEMP') or os.getenv('TMP')
            if temp:
                return Path(temp) / 'JARVIS'
            return home / '.jarvis' / 'cache'
        else:  # Linux
            xdg_cache = os.getenv('XDG_CACHE_HOME')
            if xdg_cache:
                return Path(xdg_cache) / 'jarvis'
            return home / '.cache' / 'jarvis'
    
    # ═══════════════════════════════════════════════════════════════════════════
    # HARDWARE DETECTION HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    @staticmethod
    def _check_metal() -> bool:
        """Check if macOS Metal is available"""
        try:
            result = subprocess.run(
                ['system_profiler', 'SPDisplaysDataType'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'Metal' in result.stdout
        except Exception:
            return False
    
    @staticmethod
    def _check_apple_neural_engine() -> bool:
        """Check if Apple Neural Engine is available"""
        try:
            # Check for Apple Silicon
            result = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'Apple' in result.stdout
        except Exception:
            return False
    
    @staticmethod
    def _check_directml() -> bool:
        """Check if Windows DirectML is available"""
        try:
            # Try importing DirectML via ONNX Runtime
            import onnxruntime as ort
            return 'DmlExecutionProvider' in ort.get_available_providers()
        except ImportError:
            return False
    
    @staticmethod
    def _check_windows_gpu() -> bool:
        """Check if Windows has GPU"""
        try:
            # Use wmic to check for GPU
            result = subprocess.run(
                ['wmic', 'path', 'win32_VideoController', 'get', 'name'],
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout.lower()
            # Check for known GPU vendors
            return any(vendor in output for vendor in ['nvidia', 'amd', 'intel', 'radeon', 'geforce'])
        except Exception:
            return False
    
    @staticmethod
    def _check_windows_npu() -> bool:
        """Check if Windows has NPU (Neural Processing Unit)"""
        try:
            # Check for Intel AI Boost or AMD Ryzen AI
            result = subprocess.run(
                ['wmic', 'cpu', 'get', 'name'],
                capture_output=True,
                text=True,
                timeout=5
            )
            cpu_name = result.stdout.lower()
            
            # Intel Core Ultra (Meteor Lake+) has NPU
            if 'ultra' in cpu_name and 'intel' in cpu_name:
                return True
            
            # AMD Ryzen 7040/8040 series has NPU
            if 'ryzen' in cpu_name and ('7040' in cpu_name or '8040' in cpu_name):
                return True
            
            return False
        except Exception:
            return False
    
    @staticmethod
    def _check_cuda() -> bool:
        """Check if NVIDIA CUDA is available"""
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @staticmethod
    def _check_linux_gpu() -> bool:
        """Check if Linux has GPU"""
        try:
            # Check for GPU using lspci
            result = subprocess.run(
                ['lspci'],
                capture_output=True,
                text=True,
                timeout=5
            )
            output = result.stdout.lower()
            return any(keyword in output for keyword in ['vga', 'nvidia', 'amd', 'radeon'])
        except Exception:
            return False
    
    @staticmethod
    def reset_cache() -> None:
        """Reset cached platform info (for testing)"""
        PlatformDetector._cached_platform = None
        PlatformDetector._cached_info = None


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_platform() -> str:
    """Get current platform: 'macos', 'windows', or 'linux'"""
    return PlatformDetector.detect_platform()


def get_platform_info() -> PlatformInfo:
    """Get comprehensive platform information"""
    return PlatformDetector.get_platform_info()


def is_macos() -> bool:
    """Check if running on macOS"""
    return PlatformDetector.is_macos()


def is_windows() -> bool:
    """Check if running on Windows"""
    return PlatformDetector.is_windows()


def is_linux() -> bool:
    """Check if running on Linux"""
    return PlatformDetector.is_linux()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (FOR TESTING)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("JARVIS Platform Detection")
    print("=" * 80)
    
    info = get_platform_info()
    
    print(f"Platform: {info.os_family}")
    print(f"OS: {info.os_release}")
    print(f"Architecture: {info.architecture}")
    print(f"Machine: {info.machine}")
    print(f"Python: {info.python_version}")
    print(f"User: {info.user_name}")
    print(f"Home: {info.home_dir}")
    print()
    print("Hardware Capabilities:")
    print(f"  GPU: {info.has_gpu}")
    print(f"  NPU: {info.has_npu}")
    print(f"  Metal: {info.has_metal}")
    print(f"  DirectML: {info.has_directml}")
    print(f"  CUDA: {info.has_cuda}")
    print()
    print("Platform Features:")
    print(f"  Voice: {info.supports_voice}")
    print(f"  Vision: {info.supports_vision}")
    print(f"  Automation: {info.supports_automation}")
    print()
    print("Directories:")
    print(f"  Config: {PlatformDetector.get_config_dir()}")
    print(f"  Data: {PlatformDetector.get_data_dir()}")
    print(f"  Cache: {PlatformDetector.get_cache_dir()}")
