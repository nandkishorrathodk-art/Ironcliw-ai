#!/usr/bin/env python3
"""
Quick check for macOS Screen Recording permission.
This must be granted for JARVIS to watch your screen.
"""

import sys

try:
    from Quartz import (
        CGWindowListCreateImage,
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        kCGWindowImageDefault,
        CGImageGetWidth,
        CGImageGetHeight,
    )
    pyobjc_available = True
except ImportError:
    pyobjc_available = False

def check_screen_recording_permission():
    """
    Check if we can capture screen content.
    Returns True if permission granted, False otherwise.
    """
    if not pyobjc_available:
        print("❌ PyObjC not available - cannot check permissions")
        return False

    try:
        # Try to capture a test window (window ID 1 usually exists)
        # If Screen Recording permission is not granted, this will return None
        test_image = CGWindowListCreateImage(
            CGRectNull,
            kCGWindowListOptionIncludingWindow,
            1,  # Test window ID
            kCGWindowImageDefault
        )

        if test_image:
            width = CGImageGetWidth(test_image)
            height = CGImageGetHeight(test_image)
            if width > 0 and height > 0:
                print("✅ Screen Recording permission is GRANTED")
                print(f"   Successfully captured test window: {width}x{height}")
                return True

        print("❌ Screen Recording permission is NOT granted")
        print("\n📋 To grant permission:")
        print("   1. Open System Preferences → Security & Privacy")
        print("   2. Click 'Privacy' tab")
        print("   3. Select 'Screen Recording' from the left panel")
        print("   4. Check the box next to 'Terminal' (or your Python app)")
        print("   5. You may need to restart Terminal for changes to take effect")
        print("\n🔄 After granting permission, run this test again.")
        return False

    except Exception as e:
        print(f"❌ Error checking Screen Recording permission: {e}")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("macOS Screen Recording Permission Check")
    print("=" * 70)
    print()

    has_permission = check_screen_recording_permission()

    print()
    print("=" * 70)

    sys.exit(0 if has_permission else 1)
