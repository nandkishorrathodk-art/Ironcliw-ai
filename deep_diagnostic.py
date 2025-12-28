#!/usr/bin/env python3
"""
Deep diagnostic to find the REAL screen recording permission issue.
"""

import sys
import subprocess
import os

print("=" * 70)
print("🔍 DEEP DIAGNOSTIC - Screen Recording Permission")
print("=" * 70)
print()

# Check 1: macOS version
print("1️⃣  macOS Version:")
try:
    result = subprocess.run(['sw_vers'], capture_output=True, text=True)
    print(result.stdout)
except Exception as e:
    print(f"   Error: {e}")
print()

# Check 2: Python version and path
print("2️⃣  Python Information:")
print(f"   Python version: {sys.version}")
print(f"   Python executable: {sys.executable}")
print(f"   Real path: {os.path.realpath(sys.executable)}")
print()

# Check 3: TCC Database check
print("3️⃣  TCC Database Check (Privacy Database):")
try:
    tcc_db = os.path.expanduser("~/Library/Application Support/com.apple.TCC/TCC.db")
    if os.path.exists(tcc_db):
        print(f"   TCC database exists: {tcc_db}")
        result = subprocess.run(
            ['sqlite3', tcc_db,
             "SELECT client, auth_value, auth_reason FROM access WHERE service='kTCCServiceScreenCapture';"],
            capture_output=True, text=True
        )
        if result.stdout:
            print(f"   Screen Capture permissions in TCC:")
            print(f"   {result.stdout}")
        else:
            print(f"   No Screen Capture permissions found in user TCC database")
    else:
        print(f"   User TCC database not found")

    # Check system TCC database
    system_tcc = "/Library/Application Support/com.apple.TCC/TCC.db"
    if os.path.exists(system_tcc):
        print(f"\n   Checking system TCC database...")
        result = subprocess.run(
            ['sudo', 'sqlite3', system_tcc,
             "SELECT client, auth_value FROM access WHERE service='kTCCServiceScreenCapture';"],
            capture_output=True, text=True
        )
        if result.stdout:
            print(f"   System TCC entries: {result.stdout}")
except Exception as e:
    print(f"   Error checking TCC: {e}")
print()

# Check 4: PyObjC availability
print("4️⃣  PyObjC Framework Check:")
try:
    from Quartz import (
        CGWindowListCreateImage,
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        kCGWindowImageDefault,
        CGImageGetWidth,
        CGImageGetHeight,
    )
    print("   ✅ PyObjC Quartz framework available")

    # Try to actually capture
    print("\n   Testing CGWindowListCreateImage...")
    test_image = CGWindowListCreateImage(
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        1,  # Window ID 1
        kCGWindowImageDefault
    )

    print(f"   CGWindowListCreateImage returned: {test_image}")

    if test_image:
        width = CGImageGetWidth(test_image)
        height = CGImageGetHeight(test_image)
        print(f"   Image dimensions: {width} x {height}")

        if width > 0 and height > 0:
            print("   ✅ Successfully captured image!")
            print("   🎉 SCREEN RECORDING PERMISSION IS ACTUALLY GRANTED!")
        else:
            print("   ❌ Image has zero dimensions")
    else:
        print("   ❌ CGWindowListCreateImage returned None")
        print("   This means permission is NOT granted")

except ImportError as e:
    print(f"   ❌ PyObjC not available: {e}")
except Exception as e:
    print(f"   ❌ Error during capture test: {e}")
    import traceback
    traceback.print_exc()
print()

# Check 5: Alternative screen capture method
print("5️⃣  Alternative Capture Method (screencapture command):")
try:
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name

    # Try to capture screen
    result = subprocess.run(
        ['screencapture', '-x', '-t', 'png', tmp_path],
        capture_output=True,
        text=True,
        timeout=5
    )

    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
        print(f"   ✅ screencapture command works!")
        print(f"   Captured to: {tmp_path} ({os.path.getsize(tmp_path)} bytes)")
        os.unlink(tmp_path)
    else:
        print(f"   ❌ screencapture failed or produced empty file")
        print(f"   stderr: {result.stderr}")
except Exception as e:
    print(f"   Error: {e}")
print()

# Check 6: Check for Screen Recording in System Preferences
print("6️⃣  Checking Terminal in Screen Recording list:")
try:
    result = subprocess.run(
        ['tccutil', 'status', 'ScreenCapture'],
        capture_output=True,
        text=True
    )
    print(f"   tccutil status: {result.stdout}")
    print(f"   stderr: {result.stderr}")
except Exception as e:
    print(f"   Error: {e}")
print()

# Check 7: Process information
print("7️⃣  Current Process Information:")
print(f"   Process ID: {os.getpid()}")
print(f"   Parent Process ID: {os.getppid()}")
print(f"   Effective UID: {os.geteuid()}")
print(f"   Real UID: {os.getuid()}")

# Find parent process
try:
    result = subprocess.run(['ps', '-p', str(os.getppid()), '-o', 'comm='],
                          capture_output=True, text=True)
    print(f"   Parent process: {result.stdout.strip()}")
except:
    pass
print()

print("=" * 70)
print("📋 SUMMARY & RECOMMENDATIONS")
print("=" * 70)
print()
print("Based on the diagnostics above:")
print()
print("If CGWindowListCreateImage returned None:")
print("   → Screen Recording permission is definitely NOT granted")
print("   → Try: System Preferences → Security & Privacy → Screen Recording")
print("   → Make sure 'Terminal' is checked")
print("   → Then RESTART YOUR MAC (full reboot)")
print()
print("If CGWindowListCreateImage worked:")
print("   → Permission IS granted, but check_screen_recording_permission.py")
print("     might have a bug")
print()
print("If tccutil shows an error:")
print("   → Your macOS version might not support this command")
print("   → You may need to grant permission differently")
print()
print("=" * 70)
