#!/usr/bin/env python3
"""
Test script to verify backend/main.py imports successfully on Windows.
"""
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 70)
print("JARVIS Backend Import Test")
print("=" * 70)
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Working directory: {os.getcwd()}")
print("=" * 70)

# Test 1: Platform detection
print("\n[Test 1] Platform detection...")
try:
    from backend.platform import get_platform, is_windows, get_platform_info
    platform = get_platform()
    is_win = is_windows()
    info = get_platform_info()
    print(f"[OK] Platform detected: {platform}")
    print(f"[OK] Is Windows: {is_win}")
    print(f"[OK] OS Release: {info.os_release}")
    print(f"[OK] Architecture: {info.architecture}")
except Exception as e:
    print(f"[FAIL] Platform detection failed: {e}")
    sys.exit(1)

# Test 2: Import backend.main (this will test all module-level code)
print("\n[Test 2] Importing backend.main...")
try:
    # This will execute all the module-level initialization
    import backend.main as main_module
    print(f"[OK] backend.main imported successfully")
    
    # Check if platform constants are set
    if hasattr(main_module, 'JARVIS_PLATFORM'):
        print(f"[OK] JARVIS_PLATFORM = {main_module.JARVIS_PLATFORM}")
    if hasattr(main_module, 'JARVIS_IS_WINDOWS'):
        print(f"[OK] JARVIS_IS_WINDOWS = {main_module.JARVIS_IS_WINDOWS}")
    
    # Check if FastAPI app was created
    if hasattr(main_module, 'app'):
        print(f"[OK] FastAPI app created")
        print(f"     Routes: {len(main_module.app.routes)}")
    
except ImportError as e:
    print(f"[FAIL] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"[FAIL] Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] ALL TESTS PASSED")
print("=" * 70)
