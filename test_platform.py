"""
Test platform abstraction layer on Windows
"""
import sys
import asyncio

# Test platform detection
print("=" * 60)
print("JARVIS Platform Abstraction Layer - Windows Test")
print("=" * 60)

try:
    from backend.platform_adapter import get_platform
    
    platform = get_platform()
    print(f"\n[OK] Platform loaded: {platform.__class__.__name__}")
    print(f"  OS: {platform.os_name}")
    print(f"  Version: {platform.os_version}")
    print(f"  Architecture: {platform.architecture}")
    
    # Check capabilities
    print("\nCapabilities:")
    caps = platform.get_capabilities()
    for cap, available in caps.items():
        status = "[OK]" if available else "[--]"
        print(f"  {status} {cap}: {available}")
    
    # Test basic functions
    async def test_platform():
        print("\nTesting Platform Functions:")
        
        # Test system info
        try:
            info = await platform.get_system_info()
            print(f"  [OK] System Info: {info.get('hostname', 'N/A')}")
        except Exception as e:
            print(f"  [FAIL] System Info: {e}")
        
        # Test idle time
        try:
            idle = await platform.get_idle_time()
            print(f"  [OK] Idle Time: {idle:.1f}s")
        except Exception as e:
            print(f"  [FAIL] Idle Time: {e}")
        
        # Test monitor detection
        try:
            monitors = await platform.get_monitors()
            print(f"  [OK] Monitors: {len(monitors)} detected")
            for i, mon in enumerate(monitors):
                print(f"     Monitor {i+1}: {mon['width']}x{mon['height']}")
        except Exception as e:
            print(f"  [FAIL] Monitors: {e}")
        
        # Test window enumeration
        try:
            windows = await platform.get_window_info()
            print(f"  [OK] Windows: {len(windows)} open windows")
        except Exception as e:
            print(f"  [FAIL] Windows: {e}")
        
        # Test audio devices
        try:
            devices = await platform.get_audio_devices()
            inputs = len(devices.get('inputs', []))
            outputs = len(devices.get('outputs', []))
            print(f"  [OK] Audio: {inputs} inputs, {outputs} outputs")
        except Exception as e:
            print(f"  [FAIL] Audio: {e}")
        
        # Test battery
        try:
            battery = await platform.get_battery_status()
            if battery:
                print(f"  [OK] Battery: {battery['percent']}% (plugged: {battery['plugged']})")
            else:
                print(f"  [INFO] Battery: No battery detected (desktop)")
        except Exception as e:
            print(f"  [FAIL] Battery: {e}")
    
    asyncio.run(test_platform())
    
    print("\n" + "=" * 60)
    print("Platform Abstraction Test Complete!")
    print("=" * 60)
    
except ImportError as e:
    print(f"\n[FAIL] Failed to import platform module: {e}")
    print("\nMissing dependencies. Install with:")
    print("  pip install pywin32 pyautogui mss sounddevice win10toast")
    sys.exit(1)
except Exception as e:
    print(f"\n[FAIL] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

