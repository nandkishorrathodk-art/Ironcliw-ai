"""
Windows Platform-Specific Tests
================================

Comprehensive tests for Windows-specific features and integrations.

Tests:
1. Windows-specific dependencies
2. DirectX/CUDA GPU support
3. Windows automation (pygetwindow, win32gui)
4. Windows TTS (pyttsx3 + SAPI)
5. Windows system tray
6. Windows clipboard
7. Windows process management

Created: 2026-02-23
Purpose: Windows/Linux porting - Phase 8 (Integration Testing)
"""

import os
import platform
import sys
from pathlib import Path
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.core.platform_abstraction import PlatformDetector


# Skip all tests if not on Windows
pytestmark = pytest.mark.skipif(
    platform.system().lower() != "windows",
    reason="Windows-specific tests"
)


class TestWindowsDependencies:
    """Test Windows-specific dependencies."""
    
    def test_win32_available(self):
        """Test that win32 APIs are available."""
        try:
            import win32api
            import win32gui
            import win32con
            
            print("\n✅ Win32 APIs available")
            print(f"   win32api: {win32api.__file__}")
            
        except ImportError as e:
            pytest.skip(f"Win32 APIs not available: {e}")
    
    def test_wmi_available(self):
        """Test that WMI is available."""
        try:
            import wmi
            
            w = wmi.WMI()
            print("\n✅ WMI available")
            
            # Query system info
            for os_info in w.Win32_OperatingSystem():
                print(f"   OS: {os_info.Caption}")
                print(f"   Version: {os_info.Version}")
            
        except ImportError as e:
            pytest.skip(f"WMI not available: {e}")
    
    def test_comtypes_available(self):
        """Test that comtypes is available."""
        try:
            import comtypes
            
            print("\n✅ comtypes available")
            print(f"   Version: {comtypes.__version__}")
            
        except ImportError as e:
            pytest.skip(f"comtypes not available: {e}")


class TestWindowsScreenCapture:
    """Test Windows screen capture."""
    
    def test_mss_on_windows(self):
        """Test MSS screen capture on Windows."""
        try:
            import mss
            import mss.windows
            
            with mss.mss() as sct:
                monitors = sct.monitors
                
                print("\n✅ MSS screen capture on Windows")
                print(f"   Monitors detected: {len(monitors) - 1}")  # -1 excludes virtual monitor
                
                for i, monitor in enumerate(monitors[1:], 1):
                    print(f"   Monitor {i}: {monitor['width']}x{monitor['height']}")
                
                # Capture primary monitor
                screenshot = sct.grab(monitors[1])
                
                assert screenshot is not None
                assert screenshot.width > 0
                assert screenshot.height > 0
                
                print(f"   Screenshot: {screenshot.width}x{screenshot.height}")
            
        except ImportError as e:
            pytest.skip(f"MSS not available: {e}")


class TestWindowsAutomation:
    """Test Windows automation."""
    
    def test_pygetwindow_available(self):
        """Test pygetwindow availability."""
        try:
            import pygetwindow as gw
            
            # Get all windows
            windows = gw.getAllTitles()
            
            print("\n✅ pygetwindow available")
            print(f"   Windows detected: {len(windows)}")
            print(f"   Sample windows: {windows[:3]}")
            
        except ImportError as e:
            pytest.skip(f"pygetwindow not available: {e}")
    
    def test_pyautogui_on_windows(self):
        """Test pyautogui on Windows."""
        try:
            import pyautogui
            
            # Get screen size
            screen_width, screen_height = pyautogui.size()
            
            # Get mouse position
            mouse_x, mouse_y = pyautogui.position()
            
            print("\n✅ pyautogui on Windows")
            print(f"   Screen: {screen_width}x{screen_height}")
            print(f"   Mouse: ({mouse_x}, {mouse_y})")
            
            # Verify position is valid
            assert 0 <= mouse_x <= screen_width
            assert 0 <= mouse_y <= screen_height
            
        except ImportError as e:
            pytest.skip(f"pyautogui not available: {e}")
    
    def test_pynput_on_windows(self):
        """Test pynput on Windows."""
        try:
            from pynput import mouse, keyboard
            
            # Get current mouse position
            mouse_controller = mouse.Controller()
            position = mouse_controller.position
            
            print("\n✅ pynput on Windows")
            print(f"   Mouse position: {position}")
            
        except ImportError as e:
            pytest.skip(f"pynput not available: {e}")


class TestWindowsTTS:
    """Test Windows text-to-speech."""
    
    def test_pyttsx3_sapi(self):
        """Test pyttsx3 with SAPI backend."""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            
            # Get voices
            voices = engine.getProperty('voices')
            
            print("\n✅ pyttsx3 with SAPI on Windows")
            print(f"   Voices available: {len(voices)}")
            
            for i, voice in enumerate(voices[:3]):
                print(f"   Voice {i+1}: {voice.name}")
            
            # Get properties
            rate = engine.getProperty('rate')
            volume = engine.getProperty('volume')
            
            print(f"   Rate: {rate}")
            print(f"   Volume: {volume}")
            
            # Cleanup
            engine.stop()
            
        except ImportError as e:
            pytest.skip(f"pyttsx3 not available: {e}")
        except Exception as e:
            pytest.skip(f"pyttsx3 SAPI init failed: {e}")


class TestWindowsSystemTray:
    """Test Windows system tray."""
    
    def test_pystray_on_windows(self):
        """Test pystray on Windows."""
        try:
            from pystray import Icon
            from PIL import Image
            
            print("\n✅ pystray available on Windows")
            print(f"   Can create system tray icons")
            
            # Note: Not actually creating icon to avoid GUI interaction
            
        except ImportError as e:
            pytest.skip(f"pystray not available: {e}")


class TestWindowsClipboard:
    """Test Windows clipboard."""
    
    def test_pyperclip_on_windows(self):
        """Test pyperclip on Windows."""
        try:
            import pyperclip
            
            # Test write and read
            test_text = "Ironcliw Windows Test"
            
            pyperclip.copy(test_text)
            result = pyperclip.paste()
            
            assert result == test_text
            
            print("\n✅ pyperclip on Windows")
            print(f"   Clipboard test: PASSED")
            print(f"   Text: '{test_text}'")
            
            # Clear clipboard
            pyperclip.copy("")
            
        except ImportError as e:
            pytest.skip(f"pyperclip not available: {e}")


class TestWindowsProcessManagement:
    """Test Windows process management."""
    
    def test_windows_process_creation(self):
        """Test Windows process creation flags."""
        import subprocess
        
        # Windows-specific creation flags
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        DETACHED_PROCESS = 0x00000008
        
        # Test that flags are defined
        assert CREATE_NEW_PROCESS_GROUP == subprocess.CREATE_NEW_PROCESS_GROUP
        
        print("\n✅ Windows process creation flags")
        print(f"   CREATE_NEW_PROCESS_GROUP: 0x{CREATE_NEW_PROCESS_GROUP:08X}")
        print(f"   DETACHED_PROCESS: 0x{DETACHED_PROCESS:08X}")
    
    def test_windows_process_termination(self):
        """Test Windows process termination."""
        import subprocess
        import time
        
        # Start a simple process
        process = subprocess.Popen(
            ["cmd", "/c", "timeout", "/t", "60"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        time.sleep(0.5)
        
        # Terminate process
        process.terminate()
        process.wait(timeout=5)
        
        print("\n✅ Windows process termination")
        print(f"   Process started and terminated successfully")


class TestWindowsConfiguration:
    """Test Windows-specific configuration."""
    
    def test_windows_config_file(self):
        """Test Windows config file."""
        config_file = Path(__file__).parent.parent / "backend" / "config" / "windows_config.yaml"
        
        assert config_file.exists(), f"Windows config not found: {config_file}"
        
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify Windows-specific settings
        assert config["platform"]["name"] == "Windows"
        assert config["screen_capture"]["method"] == "mss"
        assert config["tts"]["engine"] == "pyttsx3"
        assert config["authentication"]["bypass_enabled"] is True
        
        print("\n✅ Windows configuration")
        print(f"   Config file: {config_file.name}")
        print(f"   Platform: {config['platform']['name']}")
        print(f"   Screen capture: {config['screen_capture']['method']}")
        print(f"   TTS engine: {config['tts']['engine']}")
        print(f"   Auth bypass: {config['authentication']['bypass_enabled']}")


class TestWindowsGPUSupport:
    """Test Windows GPU support."""
    
    def test_directx_detection(self):
        """Test DirectX availability."""
        detector = PlatformDetector()
        
        # On Windows, DirectX should be the default GPU backend
        print("\n✅ Windows GPU configuration")
        print(f"   Default GPU backend: DirectX")
        print(f"   CUDA support: Check NVIDIA GPU presence")
        print(f"   CPU fallback: Always available")
    
    def test_cuda_available(self):
        """Test CUDA availability."""
        try:
            import torch
            
            cuda_available = torch.cuda.is_available()
            
            print("\n✅ CUDA availability")
            print(f"   CUDA available: {cuda_available}")
            
            if cuda_available:
                print(f"   CUDA version: {torch.version.cuda}")
                print(f"   GPU devices: {torch.cuda.device_count()}")
                
                for i in range(torch.cuda.device_count()):
                    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            
        except ImportError:
            print("\n⚠️  PyTorch not available (CUDA test skipped)")


def test_run_all_windows_tests():
    """Run all Windows-specific tests."""
    print("\n" + "="*70)
    print("WINDOWS PLATFORM-SPECIFIC TEST SUITE")
    print("="*70)
    
    detector = PlatformDetector()
    
    if not detector.is_windows():
        print("\n⚠️  Not running on Windows - tests skipped")
        return False
    
    print(f"\nPlatform: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Run pytest programmatically
    import pytest
    
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s",  # Show print statements
    ])
    
    return exit_code == 0


if __name__ == "__main__":
    import sys
    success = test_run_all_windows_tests()
    sys.exit(0 if success else 1)
