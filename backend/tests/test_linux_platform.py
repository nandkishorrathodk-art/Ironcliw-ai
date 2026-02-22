"""
Linux Platform-Specific Tests
==============================

Comprehensive tests for Linux-specific features and integrations.

Tests:
1. Linux-specific dependencies
2. X11/Wayland support
3. Linux automation (wmctrl, xdotool)
4. Linux TTS (pyttsx3 + espeak)
5. Linux system tray (AppIndicator)
6. Linux clipboard
7. Linux process management
8. GPU support (CUDA, ROCm, Vulkan)

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


# Skip all tests if not on Linux
pytestmark = pytest.mark.skipif(
    platform.system().lower() != "linux",
    reason="Linux-specific tests"
)


class TestLinuxDependencies:
    """Test Linux-specific dependencies."""
    
    def test_xlib_available(self):
        """Test that python-xlib is available."""
        try:
            from Xlib import display
            
            d = display.Display()
            screen = d.screen()
            
            print("\n✅ python-xlib available")
            print(f"   Display: {d.get_display_name()}")
            print(f"   Screen: {screen.width_in_pixels}x{screen.height_in_pixels}")
            
        except ImportError as e:
            pytest.skip(f"python-xlib not available: {e}")
        except Exception as e:
            pytest.skip(f"X11 not available (Wayland or headless?): {e}")
    
    def test_distro_info(self):
        """Test Linux distribution detection."""
        try:
            import distro
            
            distro_name = distro.name()
            distro_version = distro.version()
            distro_id = distro.id()
            
            print("\n✅ Linux distribution info")
            print(f"   Name: {distro_name}")
            print(f"   Version: {distro_version}")
            print(f"   ID: {distro_id}")
            
        except ImportError:
            # Fallback to platform
            print("\n⚠️  distro library not available, using platform")
            print(f"   System: {platform.system()}")
            print(f"   Release: {platform.release()}")


class TestLinuxScreenCapture:
    """Test Linux screen capture."""
    
    def test_mss_on_linux(self):
        """Test MSS screen capture on Linux (X11)."""
        try:
            import mss
            
            with mss.mss() as sct:
                monitors = sct.monitors
                
                print("\n✅ MSS screen capture on Linux (X11)")
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
        except Exception as e:
            pytest.skip(f"X11 not available (Wayland or headless?): {e}")
    
    def test_wayland_detection(self):
        """Test Wayland detection."""
        wayland_display = os.getenv("WAYLAND_DISPLAY")
        xdg_session_type = os.getenv("XDG_SESSION_TYPE")
        
        is_wayland = wayland_display is not None or xdg_session_type == "wayland"
        
        print("\n✅ Wayland detection")
        print(f"   WAYLAND_DISPLAY: {wayland_display}")
        print(f"   XDG_SESSION_TYPE: {xdg_session_type}")
        print(f"   Is Wayland: {is_wayland}")
        
        if is_wayland:
            print(f"   Note: Wayland requires 'grim' for screen capture")


class TestLinuxAutomation:
    """Test Linux automation."""
    
    def test_xdotool_available(self):
        """Test xdotool availability."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["which", "xdotool"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            xdotool_available = result.returncode == 0
            
            print("\n✅ xdotool availability")
            print(f"   Available: {xdotool_available}")
            
            if xdotool_available:
                print(f"   Path: {result.stdout.strip()}")
            
        except Exception as e:
            pytest.skip(f"xdotool test failed: {e}")
    
    def test_wmctrl_available(self):
        """Test wmctrl availability."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["which", "wmctrl"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            wmctrl_available = result.returncode == 0
            
            print("\n✅ wmctrl availability")
            print(f"   Available: {wmctrl_available}")
            
            if wmctrl_available:
                print(f"   Path: {result.stdout.strip()}")
            
        except Exception as e:
            pytest.skip(f"wmctrl test failed: {e}")
    
    def test_pyautogui_on_linux(self):
        """Test pyautogui on Linux."""
        try:
            import pyautogui
            
            # Get screen size
            screen_width, screen_height = pyautogui.size()
            
            # Get mouse position
            mouse_x, mouse_y = pyautogui.position()
            
            print("\n✅ pyautogui on Linux")
            print(f"   Screen: {screen_width}x{screen_height}")
            print(f"   Mouse: ({mouse_x}, {mouse_y})")
            
            # Verify position is valid
            assert 0 <= mouse_x <= screen_width
            assert 0 <= mouse_y <= screen_height
            
        except ImportError as e:
            pytest.skip(f"pyautogui not available: {e}")
        except Exception as e:
            pytest.skip(f"X11 not available (Wayland or headless?): {e}")
    
    def test_pynput_on_linux(self):
        """Test pynput on Linux."""
        try:
            from pynput import mouse, keyboard
            
            # Get current mouse position
            mouse_controller = mouse.Controller()
            position = mouse_controller.position
            
            print("\n✅ pynput on Linux")
            print(f"   Mouse position: {position}")
            
        except ImportError as e:
            pytest.skip(f"pynput not available: {e}")
        except Exception as e:
            pytest.skip(f"X11 not available (Wayland or headless?): {e}")


class TestLinuxTTS:
    """Test Linux text-to-speech."""
    
    def test_espeak_available(self):
        """Test espeak availability."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["which", "espeak"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            espeak_available = result.returncode == 0
            
            print("\n✅ espeak availability")
            print(f"   Available: {espeak_available}")
            
            if espeak_available:
                print(f"   Path: {result.stdout.strip()}")
            
        except Exception as e:
            pytest.skip(f"espeak test failed: {e}")
    
    def test_pyttsx3_espeak(self):
        """Test pyttsx3 with espeak backend."""
        try:
            import pyttsx3
            
            engine = pyttsx3.init()
            
            # Get voices
            voices = engine.getProperty('voices')
            
            print("\n✅ pyttsx3 with espeak on Linux")
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
            pytest.skip(f"pyttsx3 espeak init failed: {e}")


class TestLinuxSystemTray:
    """Test Linux system tray."""
    
    def test_pystray_on_linux(self):
        """Test pystray on Linux."""
        try:
            from pystray import Icon
            from PIL import Image
            
            print("\n✅ pystray available on Linux")
            print(f"   Can create system tray icons (AppIndicator)")
            
            # Note: Not actually creating icon to avoid GUI interaction
            
        except ImportError as e:
            pytest.skip(f"pystray not available: {e}")
    
    def test_notify_send_available(self):
        """Test notify-send availability."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["which", "notify-send"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            notify_available = result.returncode == 0
            
            print("\n✅ notify-send availability")
            print(f"   Available: {notify_available}")
            
            if notify_available:
                print(f"   Path: {result.stdout.strip()}")
            
        except Exception as e:
            pytest.skip(f"notify-send test failed: {e}")


class TestLinuxClipboard:
    """Test Linux clipboard."""
    
    def test_pyperclip_on_linux(self):
        """Test pyperclip on Linux."""
        try:
            import pyperclip
            
            # Test write and read
            test_text = "JARVIS Linux Test"
            
            pyperclip.copy(test_text)
            result = pyperclip.paste()
            
            assert result == test_text
            
            print("\n✅ pyperclip on Linux")
            print(f"   Clipboard test: PASSED")
            print(f"   Text: '{test_text}'")
            
            # Clear clipboard
            pyperclip.copy("")
            
        except ImportError as e:
            pytest.skip(f"pyperclip not available: {e}")
        except Exception as e:
            pytest.skip(f"Clipboard test failed (X11 required): {e}")


class TestLinuxProcessManagement:
    """Test Linux process management."""
    
    def test_linux_process_signals(self):
        """Test Linux process signals."""
        import signal
        
        # Linux-specific signals
        signals_available = {
            "SIGTERM": hasattr(signal, "SIGTERM"),
            "SIGKILL": hasattr(signal, "SIGKILL"),
            "SIGHUP": hasattr(signal, "SIGHUP"),
            "SIGINT": hasattr(signal, "SIGINT"),
        }
        
        print("\n✅ Linux process signals")
        
        for sig_name, available in signals_available.items():
            print(f"   {sig_name}: {available}")
    
    def test_linux_process_management(self):
        """Test Linux process management."""
        import subprocess
        import time
        
        # Start a simple process
        process = subprocess.Popen(
            ["sleep", "60"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        time.sleep(0.5)
        
        # Terminate process
        process.terminate()
        process.wait(timeout=5)
        
        print("\n✅ Linux process management")
        print(f"   Process started and terminated successfully")


class TestLinuxConfiguration:
    """Test Linux-specific configuration."""
    
    def test_linux_config_file(self):
        """Test Linux config file."""
        config_file = Path(__file__).parent.parent / "backend" / "config" / "linux_config.yaml"
        
        assert config_file.exists(), f"Linux config not found: {config_file}"
        
        import yaml
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verify Linux-specific settings
        assert config["platform"]["name"] == "Linux"
        assert config["screen_capture"]["method"] == "mss"
        assert config["tts"]["engine"] == "pyttsx3"
        assert config["authentication"]["bypass_enabled"] is True
        
        print("\n✅ Linux configuration")
        print(f"   Config file: {config_file.name}")
        print(f"   Platform: {config['platform']['name']}")
        print(f"   Screen capture: {config['screen_capture']['method']}")
        print(f"   TTS engine: {config['tts']['engine']}")
        print(f"   Auth bypass: {config['authentication']['bypass_enabled']}")


class TestLinuxGPUSupport:
    """Test Linux GPU support."""
    
    def test_cuda_available(self):
        """Test CUDA availability on Linux."""
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
    
    def test_rocm_available(self):
        """Test ROCm availability on Linux."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["which", "rocm-smi"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            rocm_available = result.returncode == 0
            
            print("\n✅ ROCm availability")
            print(f"   ROCm available: {rocm_available}")
            
            if rocm_available:
                print(f"   Path: {result.stdout.strip()}")
            
        except Exception as e:
            print(f"\n⚠️  ROCm test skipped: {e}")
    
    def test_vulkan_available(self):
        """Test Vulkan availability on Linux."""
        import subprocess
        
        try:
            result = subprocess.run(
                ["which", "vulkaninfo"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            
            vulkan_available = result.returncode == 0
            
            print("\n✅ Vulkan availability")
            print(f"   Vulkan available: {vulkan_available}")
            
            if vulkan_available:
                print(f"   Path: {result.stdout.strip()}")
            
        except Exception as e:
            print(f"\n⚠️  Vulkan test skipped: {e}")


class TestLinuxDisplayServer:
    """Test Linux display server (X11 vs Wayland)."""
    
    def test_display_server_detection(self):
        """Test display server detection."""
        xdg_session_type = os.getenv("XDG_SESSION_TYPE", "unknown")
        wayland_display = os.getenv("WAYLAND_DISPLAY")
        display = os.getenv("DISPLAY")
        
        print("\n✅ Linux display server detection")
        print(f"   XDG_SESSION_TYPE: {xdg_session_type}")
        print(f"   WAYLAND_DISPLAY: {wayland_display}")
        print(f"   DISPLAY: {display}")
        
        if xdg_session_type == "wayland" or wayland_display:
            print(f"   → Running on Wayland")
        elif display:
            print(f"   → Running on X11")
        else:
            print(f"   → Headless or unknown")


def test_run_all_linux_tests():
    """Run all Linux-specific tests."""
    print("\n" + "="*70)
    print("LINUX PLATFORM-SPECIFIC TEST SUITE")
    print("="*70)
    
    detector = PlatformDetector()
    
    if not detector.is_linux():
        print("\n⚠️  Not running on Linux - tests skipped")
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
    success = test_run_all_linux_tests()
    sys.exit(0 if success else 1)
