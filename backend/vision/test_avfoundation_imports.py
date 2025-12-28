#!/usr/bin/env python3
"""
Diagnostic script to test AVFoundation imports and identify issues
"""
import sys

print("=" * 80)
print("AVFoundation Import Diagnostic Tool")
print("=" * 80)
print()

# Test 1: PyObjC core
print("[1/10] Testing PyObjC core...")
try:
    import objc
    print(f"✅ objc imported successfully (version: {objc.__version__})")
except ImportError as e:
    print(f"❌ Failed to import objc: {e}")
    sys.exit(1)

# Test 2: Foundation framework
print("\n[2/10] Testing Foundation framework...")
try:
    from Foundation import (
        NSObject,
        NSRunLoop,
        NSDefaultRunLoopMode,
        NSDate,
    )
    print("✅ Foundation imports successful")
    print(f"   - NSObject: {NSObject}")
    print(f"   - NSRunLoop: {NSRunLoop}")
    print(f"   - NSDefaultRunLoopMode: {NSDefaultRunLoopMode}")
    print(f"   - NSDate: {NSDate}")
except ImportError as e:
    print(f"❌ Failed to import Foundation: {e}")
    sys.exit(1)

# Test 3: AVFoundation framework
print("\n[3/10] Testing AVFoundation framework...")
try:
    import AVFoundation
    print(f"✅ AVFoundation imported (version: {AVFoundation.__version__ if hasattr(AVFoundation, '__version__') else 'unknown'})")

    # Test specific classes
    print("   Testing AVFoundation classes...")
    from AVFoundation import (
        AVCaptureSession,
        AVCaptureScreenInput,
        AVCaptureVideoDataOutput,
        AVCaptureSessionPreset1920x1080,
        AVCaptureSessionPreset1280x720,
        AVCaptureSessionPreset640x480,
    )
    print("   ✅ AVCaptureSession:", AVCaptureSession)
    print("   ✅ AVCaptureScreenInput:", AVCaptureScreenInput)
    print("   ✅ AVCaptureVideoDataOutput:", AVCaptureVideoDataOutput)
    print("   ✅ Session presets imported successfully")
except ImportError as e:
    print(f"❌ Failed to import AVFoundation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: CoreMedia framework
print("\n[4/10] Testing CoreMedia framework...")
try:
    import CoreMedia
    print(f"✅ CoreMedia imported")

    from CoreMedia import (
        CMSampleBufferGetImageBuffer,
        CMTimeMake,
    )
    print("   ✅ CMSampleBufferGetImageBuffer:", CMSampleBufferGetImageBuffer)
    print("   ✅ CMTimeMake:", CMTimeMake)
except ImportError as e:
    print(f"❌ Failed to import CoreMedia: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Quartz framework (CoreVideo functions)
print("\n[5/10] Testing Quartz/CoreVideo framework...")
try:
    import Quartz
    print(f"✅ Quartz imported")

    # CRITICAL: CoreVideo functions are in Quartz.CoreVideo, not directly in Quartz
    print("   Testing CoreVideo functions...")

    # Try method 1: Direct import from Quartz
    try:
        from Quartz import (
            CVPixelBufferLockBaseAddress,
            CVPixelBufferUnlockBaseAddress,
            CVPixelBufferGetBaseAddress,
            CVPixelBufferGetBytesPerRow,
            CVPixelBufferGetHeight,
            CVPixelBufferGetWidth,
            kCVPixelBufferPixelFormatTypeKey,
            kCVPixelFormatType_32BGRA,
        )
        print("   ✅ Method 1: Direct import from Quartz works")
        CV_IMPORT_METHOD = "direct"
    except ImportError as e1:
        print(f"   ⚠️  Method 1 failed: {e1}")

        # Try method 2: Import CoreVideo submodule
        try:
            from Quartz import CoreVideo
            print("   ✅ Method 2: Import CoreVideo submodule works")
            print(f"      - CVPixelBufferLockBaseAddress: {CoreVideo.CVPixelBufferLockBaseAddress}")
            print(f"      - kCVPixelFormatType_32BGRA: {CoreVideo.kCVPixelFormatType_32BGRA}")
            CV_IMPORT_METHOD = "submodule"
        except ImportError as e2:
            print(f"   ❌ Method 2 also failed: {e2}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print(f"   ℹ️  Recommended import method: {CV_IMPORT_METHOD}")

except ImportError as e:
    print(f"❌ Failed to import Quartz: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: libdispatch
print("\n[6/10] Testing libdispatch...")
try:
    # libdispatch can be imported multiple ways
    import_methods = []

    # Method 1: Direct import
    try:
        import libdispatch as ld1
        print("   ✅ Method 1: Direct import 'libdispatch' works")
        print(f"      dispatch_queue_create: {ld1.dispatch_queue_create}")
        import_methods.append("direct")
    except ImportError:
        print("   ⚠️  Method 1: Direct import failed")

    # Method 2: From dispatch
    try:
        import dispatch
        print("   ✅ Method 2: Import 'dispatch' works")
        print(f"      dispatch_queue_create: {dispatch.dispatch_queue_create}")
        import_methods.append("dispatch")
    except ImportError:
        print("   ⚠️  Method 2: Import 'dispatch' failed")

    if not import_methods:
        print("   ❌ All libdispatch import methods failed")
        print("   ℹ️  This may cause issues with dispatch queues")
    else:
        print(f"   ℹ️  Available methods: {import_methods}")

except Exception as e:
    print(f"⚠️  libdispatch test encountered error: {e}")

# Test 7: Create AVCaptureSession
print("\n[7/10] Testing AVCaptureSession creation...")
try:
    session = AVFoundation.AVCaptureSession.alloc().init()
    print(f"✅ Successfully created AVCaptureSession: {session}")
    print(f"   Session running: {session.isRunning()}")
except Exception as e:
    print(f"❌ Failed to create AVCaptureSession: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Create screen input
print("\n[8/10] Testing AVCaptureScreenInput creation...")
try:
    display_id = 0  # Main display
    screen_input = AVFoundation.AVCaptureScreenInput.alloc().initWithDisplayID_(display_id)
    if screen_input:
        print(f"✅ Successfully created AVCaptureScreenInput for display {display_id}")
        print(f"   Input: {screen_input}")
    else:
        print(f"❌ AVCaptureScreenInput.initWithDisplayID_({display_id}) returned None")
except Exception as e:
    print(f"❌ Failed to create AVCaptureScreenInput: {e}")
    import traceback
    traceback.print_exc()

# Test 9: Check screen recording permission
print("\n[9/10] Checking screen recording permission...")
try:
    # Try to check permission using CGPreflightScreenCaptureAccess (macOS 10.15+)
    try:
        from Quartz import CGPreflightScreenCaptureAccess, CGRequestScreenCaptureAccess
        has_permission = CGPreflightScreenCaptureAccess()
        print(f"   Permission status: {'✅ GRANTED' if has_permission else '❌ DENIED'}")

        if not has_permission:
            print("   ℹ️  To grant permission:")
            print("      1. Open System Settings → Privacy & Security → Screen Recording")
            print("      2. Enable permission for Terminal or your app")
            print("      3. Restart this script")
    except (ImportError, AttributeError):
        print("   ⚠️  Permission check API not available (macOS < 10.15)")
        print("   ℹ️  You may need to manually check screen recording permission")
except Exception as e:
    print(f"⚠️  Permission check failed: {e}")

# Test 10: Summary
print("\n[10/10] Summary")
print("=" * 80)

summary = {
    'pyobjc_core': True,  # If we got here, it's installed
    'foundation': True,
    'avfoundation': 'AVFoundation' in sys.modules,
    'coremedia': 'CoreMedia' in sys.modules,
    'quartz': 'Quartz' in sys.modules,
    'libdispatch': 'libdispatch' in sys.modules or 'dispatch' in sys.modules,
}

all_ok = all(summary.values())

if all_ok:
    print("✅ ALL CHECKS PASSED!")
    print()
    print("AVFoundation is ready to use.")
    print()
    print("Next steps:")
    print("1. Ensure screen recording permission is granted")
    print("2. Test capture with: python3 backend/vision/test_avfoundation_capture.py")
else:
    print("❌ SOME CHECKS FAILED")
    print()
    print("Failed components:")
    for name, status in summary.items():
        if not status:
            print(f"  - {name}")
    print()
    print("Please install missing frameworks:")
    print("  pip install pyobjc-framework-AVFoundation pyobjc-framework-Quartz \\")
    print("              pyobjc-framework-CoreMedia pyobjc-framework-libdispatch")

print("=" * 80)

# Return exit code
sys.exit(0 if all_ok else 1)
