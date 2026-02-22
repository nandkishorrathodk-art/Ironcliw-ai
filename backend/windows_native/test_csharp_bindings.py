"""
Python bindings test for C# Windows Native Layer components.

Requirements:
    pip install pythonnet

Usage:
    python test_csharp_bindings.py
"""

import sys
import os
from pathlib import Path

def test_system_control():
    """Test SystemControl C# component."""
    print("\n=== Testing SystemControl ===")
    
    try:
        import clr
        
        # Add reference to the compiled DLL
        dll_path = Path(__file__).parent / "SystemControl" / "bin" / "Release" / "net8.0-windows" / "SystemControl.dll"
        if not dll_path.exists():
            print(f"❌ DLL not found: {dll_path}")
            print("   Please build the C# projects first: dotnet build -c Release")
            return False
        
        clr.AddReference(str(dll_path))
        from JarvisWindowsNative.SystemControl import SystemController
        
        controller = SystemController()
        
        # Test window enumeration
        print("📊 Getting all windows...")
        windows = controller.GetAllWindows()
        print(f"   Found {len(windows)} windows")
        
        if len(windows) > 0:
            print(f"   First window: {windows[0].Title} (PID: {windows[0].ProcessId})")
        
        # Test focused window
        print("🔍 Getting focused window...")
        focused = controller.GetFocusedWindow()
        if focused:
            print(f"   Focused: {focused.Title}")
        
        # Test volume control
        print("🔊 Testing volume control...")
        volume = controller.GetVolume()
        print(f"   Current volume: {volume}%")
        
        print("✅ SystemControl tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ SystemControl test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_screen_capture():
    """Test ScreenCapture C# component."""
    print("\n=== Testing ScreenCapture ===")
    
    try:
        import clr
        
        # Add reference to the compiled DLL
        dll_path = Path(__file__).parent / "ScreenCapture" / "bin" / "Release" / "net8.0-windows10.0.19041.0" / "ScreenCapture.dll"
        if not dll_path.exists():
            print(f"❌ DLL not found: {dll_path}")
            print("   Please build the C# projects first: dotnet build -c Release")
            return False
        
        clr.AddReference(str(dll_path))
        from JarvisWindowsNative.ScreenCapture import ScreenCaptureEngine, MultiMonitorCapture
        
        engine = ScreenCaptureEngine()
        
        # Test screen size detection
        print("📐 Getting screen size...")
        size = engine.GetScreenSize()
        print(f"   Screen size: {size.Item1}x{size.Item2}")
        
        # Test screen capture
        print("📸 Capturing screen...")
        screenshot = engine.CaptureScreen()
        print(f"   Captured {len(screenshot)} bytes")
        
        # Test saving to file
        test_file = Path(__file__).parent / "test_screenshot.png"
        print(f"💾 Saving to {test_file}...")
        success = engine.SaveScreenToFile(str(test_file))
        
        if success and test_file.exists():
            print(f"   ✅ Screenshot saved ({test_file.stat().st_size} bytes)")
            test_file.unlink()  # Clean up
        else:
            print("   ⚠️ Failed to save screenshot")
        
        # Test multi-monitor
        print("🖥️ Testing multi-monitor support...")
        multi = MultiMonitorCapture()
        monitors = multi.GetAllMonitors()
        print(f"   Found {len(monitors)} monitor(s)")
        
        for i, mon in enumerate(monitors):
            primary = " (PRIMARY)" if mon.IsPrimary else ""
            print(f"   Monitor {i}: {mon.Width}x{mon.Height} at ({mon.X}, {mon.Y}){primary}")
        
        print("✅ ScreenCapture tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ ScreenCapture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_audio_engine():
    """Test AudioEngine C# component."""
    print("\n=== Testing AudioEngine ===")
    
    try:
        import clr
        
        # Add reference to the compiled DLL
        dll_path = Path(__file__).parent / "AudioEngine" / "bin" / "Release" / "net8.0-windows" / "AudioEngine.dll"
        if not dll_path.exists():
            print(f"❌ DLL not found: {dll_path}")
            print("   Please build the C# projects first: dotnet build -c Release")
            return False
        
        clr.AddReference(str(dll_path))
        from JarvisWindowsNative.AudioEngine import AudioEngine
        
        engine = AudioEngine()
        
        # Test device enumeration
        print("🎤 Getting input devices...")
        input_devices = engine.GetInputDevices()
        print(f"   Found {len(input_devices)} input device(s)")
        
        for dev in input_devices:
            default = " (DEFAULT)" if dev.IsDefault else ""
            print(f"   - {dev.Name}{default}")
        
        print("🔊 Getting output devices...")
        output_devices = engine.GetOutputDevices()
        print(f"   Found {len(output_devices)} output device(s)")
        
        for dev in output_devices:
            default = " (DEFAULT)" if dev.IsDefault else ""
            print(f"   - {dev.Name}{default}")
        
        # Test volume control
        print("🔊 Testing volume control...")
        volume = engine.GetVolume()
        print(f"   Current volume: {volume * 100:.0f}%")
        
        is_muted = engine.IsMuted()
        print(f"   Muted: {is_muted}")
        
        # Cleanup
        engine.Dispose()
        
        print("✅ AudioEngine tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ AudioEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("C# Windows Native Layer - Python Bindings Test")
    print("=" * 60)
    
    # Check pythonnet
    try:
        import clr
        print("✅ pythonnet is installed")
    except ImportError:
        print("❌ pythonnet not found. Install with: pip install pythonnet")
        return 1
    
    # Run tests
    results = {
        "SystemControl": test_system_control(),
        "ScreenCapture": test_screen_capture(),
        "AudioEngine": test_audio_engine()
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for component, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{component:20s} {status}")
    
    all_passed = all(results.values())
    print("\n" + ("🎉 All tests passed!" if all_passed else "⚠️ Some tests failed"))
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
