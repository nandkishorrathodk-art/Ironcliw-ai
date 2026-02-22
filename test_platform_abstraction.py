"""
Test Platform Abstraction Layer
================================

Quick test script to verify the platform abstraction layer works correctly.
"""

import asyncio
import sys
import os
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    try:
        import codecs
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
        sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")
    except:
        pass  # Fallback to ASCII-only output

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.platform import get_platform
from backend.core.capabilities import get_capabilities
from backend.core.path_manager import get_path_manager


async def test_platform_detection():
    """Test that platform is correctly detected."""
    print("=" * 70)
    print("JARVIS Platform Abstraction Layer Test")
    print("=" * 70)
    print()
    
    # Test platform detection
    print("[*] Testing Platform Detection...")
    try:
        platform = get_platform()
        print(f"[OK] Platform detected: {platform.__class__.__name__}")
        print(f"   OS: {platform.os_name}")
        print(f"   Version: {platform.os_version}")
        print(f"   Architecture: {platform.architecture}")
        print()
    except Exception as e:
        print(f"[ERROR] Platform detection failed: {e}")
        return False
    
    # Test capabilities
    print("[*] Testing Capabilities Detection...")
    try:
        caps = get_capabilities()
        print(f"[OK] Capabilities detected:")
        print(f"   OS: {caps.os} {caps.arch}")
        print(f"   GPU: {caps.gpu_type.upper()}")
        print(f"   Audio: {caps.audio_backend}")
        print(f"   Swift Extensions: {'Yes' if caps.has_swift else 'No'}")
        print(f"   Rust Extensions: {'Yes' if caps.has_rust_extensions else 'No'}")
        print(f"   Dev Mode: {'Yes' if caps.is_dev_mode else 'No'}")
        print()
        
        # Show ML config
        ml_config = caps.get_ml_config()
        print(f"   ML Configuration:")
        print(f"     GPU Layers: {ml_config['gpu_layers']}")
        print(f"     Context Size: {ml_config['context_size']}")
        print(f"     Batch Size: {ml_config['batch_size']}")
        print()
    except Exception as e:
        print(f"[ERROR] Capabilities detection failed: {e}")
        print()
    
    # Test path manager
    print("[*] Testing Path Manager...")
    try:
        paths = get_path_manager()
        print(f"[OK] Path manager initialized:")
        print(f"   JARVIS Home: {paths.get_jarvis_home()}")
        print(f"   Cache: {paths.get_cache_dir()}")
        print(f"   Models: {paths.get_models_dir()}")
        print(f"   Logs: {paths.get_logs_dir()}")
        print(f"   Config: {paths.get_config_dir()}")
        print()
    except Exception as e:
        print(f"[ERROR] Path manager failed: {e}")
        print()
    
    # Test platform capabilities
    print("[*] Testing Platform Capabilities...")
    try:
        caps_dict = platform.get_capabilities()
        print("[OK] Platform capabilities:")
        for cap, enabled in caps_dict.items():
            status = "Yes" if enabled else "No"
            print(f"   {status:3s} {cap}")
        print()
    except Exception as e:
        print(f"[ERROR] Platform capabilities failed: {e}")
        print()
    
    # Test system information
    print("[*] Testing System Information...")
    try:
        sys_info = await platform.get_system_info()
        print("[OK] System information:")
        print(f"   Hostname: {sys_info['hostname']}")
        print(f"   CPU Cores: {sys_info['cpu_count']}")
        print(f"   Memory: {sys_info['memory_total'] / (1024**3):.1f} GB total, "
              f"{sys_info['memory_available'] / (1024**3):.1f} GB available")
        print()
    except Exception as e:
        print(f"[ERROR] System info failed: {e}")
        print()
    
    # Test idle time
    print("[*] Testing Idle Time Detection...")
    try:
        idle = await platform.get_idle_time()
        print(f"[OK] Idle time: {idle:.1f} seconds")
        print()
    except Exception as e:
        print(f"[ERROR] Idle time detection failed: {e}")
        print()
    
    # Test monitors
    print("[*] Testing Monitor Detection...")
    try:
        monitors = await platform.get_monitors()
        print(f"[OK] Found {len(monitors)} monitor(s):")
        for mon in monitors:
            primary = " (PRIMARY)" if mon['is_primary'] else ""
            print(f"   Monitor {mon['id']}: {mon['width']}x{mon['height']} at ({mon['x']}, {mon['y']}){primary}")
        print()
    except Exception as e:
        print(f"[ERROR] Monitor detection failed: {e}")
        print()
    
    # Test audio devices
    print("[*] Testing Audio Device Detection...")
    try:
        audio = await platform.get_audio_devices()
        print(f"[OK] Audio devices:")
        print(f"   Input devices: {len(audio['inputs'])}")
        for dev in audio['inputs'][:3]:  # Show first 3
            print(f"     - {dev['name']}")
        print(f"   Output devices: {len(audio['outputs'])}")
        for dev in audio['outputs'][:3]:  # Show first 3
            print(f"     - {dev['name']}")
        print()
    except Exception as e:
        print(f"[ERROR] Audio device detection failed: {e}")
        print()
    
    # Test mouse position
    print("[*] Testing Mouse Position...")
    try:
        x, y = await platform.get_mouse_position()
        print(f"[OK] Mouse position: ({x}, {y})")
        print()
    except Exception as e:
        print(f"[ERROR] Mouse position failed: {e}")
        print()
    
    # Test battery (may not be available on desktops)
    print("[*] Testing Battery Status...")
    try:
        battery = await platform.get_battery_status()
        if battery:
            print(f"[OK] Battery:")
            print(f"   Level: {battery['percent']}%")
            print(f"   Plugged: {'Yes' if battery['plugged'] else 'No'}")
            if battery['time_left']:
                print(f"   Time Left: {battery['time_left'] / 60:.0f} minutes")
        else:
            print("[INFO] No battery detected (desktop or battery info unavailable)")
        print()
    except Exception as e:
        print(f"[ERROR] Battery status failed: {e}")
        print()
    
    print("=" * 70)
    print("[OK] Platform Abstraction Layer Test Complete!")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_platform_detection())
    sys.exit(0 if success else 1)
