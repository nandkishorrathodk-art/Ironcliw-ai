# ✅ Fixed: "Video streaming using fallback mode" Warning

## Problem

After installing PyObjC frameworks for AVFoundation support, Ironcliw was still showing:
```
⚠ Video streaming using fallback mode
```

Despite AVFoundation being properly installed and working.

---

## Root Cause Analysis

The issue was caused by **import conflicts and missing dependencies** in the module loading chain:

### **Issue #1: Missing PIL/Pillow Dependency**
```python
# In video_stream_capture.py line 21
from PIL import Image  # ← ModuleNotFoundError if Pillow not installed
```

**Impact**: Module import failed silently, falling back before even checking AVFoundation availability.

### **Issue #2: NSObject Import Conflict**
```python
# Advanced import succeeded, but legacy code path tried to use NSObject
class VideoFrameDelegate(NSObject):  # ← NameError: NSObject not defined
```

**Impact**: Even when advanced capture was available, the legacy compatibility code caused import failures.

### **Issue #3: Duplicate Objective-C Class Registration**
```python
# Both modules defined VideoFrameDelegate
# macos_video_capture_advanced.py:
class VideoFrameDelegate(NSObject):  # ← First definition

# video_stream_capture.py:
class VideoFrameDelegate(NSObject):  # ← objc.error: class already exists
```

**Impact**: PyObjC doesn't allow the same Objective-C class to be registered twice, causing import failures.

---

## Solution Implemented

### **Fix #1: Install Missing Dependencies** ✅

```bash
pip install Pillow  # For PIL support
```

**Result**: Module can now import successfully.

### **Fix #2: Robust Import Chain with Fallbacks** ✅

**Updated:** `backend/vision/video_stream_capture.py` (lines 23-81)

```python
# Try advanced capture first
try:
    from .macos_video_capture_advanced import (
        create_video_capture,
        AdvancedCaptureConfig,
        ...
    )
    MACOS_CAPTURE_ADVANCED_AVAILABLE = True
    MACOS_CAPTURE_AVAILABLE = AVFOUNDATION_AVAILABLE

    # If advanced import succeeded, also import legacy PyObjC classes
    # for backward compatibility with old MacOSVideoCapture class
    if AVFOUNDATION_AVAILABLE:
        try:
            import AVFoundation
            import CoreMedia
            from Cocoa import NSObject
            import objc
            ...
        except ImportError:
            # Define dummy classes for legacy code
            NSObject = object
            objc = None
    else:
        # Advanced module imported but AVFoundation not available
        NSObject = object
        objc = None

except ImportError as e:
    # Advanced capture not available - try legacy import
    MACOS_CAPTURE_ADVANCED_AVAILABLE = False
    try:
        import AVFoundation
        import CoreMedia
        from Cocoa import NSObject
        import objc
        ...
        MACOS_CAPTURE_AVAILABLE = True
    except ImportError:
        MACOS_CAPTURE_AVAILABLE = False
        # Define dummy classes so module can still load
        NSObject = object
        objc = None
```

**Key Features:**
- ✅ Graceful degradation at each level
- ✅ Dummy class definitions prevent NameError
- ✅ Backward compatibility maintained
- ✅ Module always loads (even without AVFoundation)

### **Fix #3: Conditional VideoFrameDelegate Definition** ✅

**Updated:** `backend/vision/video_stream_capture.py` (lines 324-348)

```python
# Only define LegacyVideoFrameDelegate if macOS frameworks are available
# The advanced capture (v10.6) has its own VideoFrameDelegate
if MACOS_CAPTURE_AVAILABLE and not MACOS_CAPTURE_ADVANCED_AVAILABLE:
    # Only define if advanced capture is NOT available (avoid duplicate registration)
    class VideoFrameDelegate(NSObject):
        """Legacy delegate for handling video frames (backward compatibility)"""
        ...

elif not MACOS_CAPTURE_AVAILABLE:
    # Placeholder class when macOS frameworks aren't available
    class VideoFrameDelegate:
        """Placeholder delegate for non-macOS systems"""
        def __init__(self):
            raise NotImplementedError("VideoFrameDelegate requires macOS frameworks")

# else: Advanced capture is available, use its VideoFrameDelegate
```

**Logic:**
- If advanced capture available → Use advanced VideoFrameDelegate
- If only legacy available → Define legacy VideoFrameDelegate
- If nothing available → Define placeholder VideoFrameDelegate

### **Fix #4: Enhanced Diagnostics in start_system.py** ✅

**Updated:** `start_system.py` (lines 8214-8246)

```python
# Check for native video capture (v10.6 - enhanced diagnostics)
try:
    from backend.vision.video_stream_capture import (
        MACOS_CAPTURE_AVAILABLE,
        MACOS_CAPTURE_ADVANCED_AVAILABLE,
    )

    if MACOS_CAPTURE_ADVANCED_AVAILABLE:
        print(f"{Colors.GREEN}✓ Advanced macOS video capture available (v10.6){Colors.ENDC}")
        print(f"{Colors.GREEN}  • Native AVFoundation with async support{Colors.ENDC}")
        print(f"{Colors.GREEN}  • 🟣 Purple indicator enabled{Colors.ENDC}")
        print(f"{Colors.GREEN}  • Real-time metrics and adaptive quality{Colors.ENDC}")
    elif MACOS_CAPTURE_AVAILABLE:
        print(f"{Colors.YELLOW}⚠ Legacy macOS video capture (basic mode){Colors.ENDC}")
    else:
        print(f"{Colors.YELLOW}⚠ Video streaming using fallback mode{Colors.ENDC}")
        print(f"{Colors.YELLOW}  • Install PyObjC for native capture{Colors.ENDC}")

except ImportError as import_err:
    print(f"{Colors.YELLOW}⚠ Video capture module import failed: {import_err}{Colors.ENDC}")
```

**Now shows:**
- ✅ Advanced capture status (v10.6)
- ✅ Native AVFoundation availability
- ✅ Purple indicator status
- ✅ Feature availability
- ✅ Clear error messages with actionable advice

---

## Verification

### **Test #1: Import Chain**
```bash
PYTHONPATH="$PWD:$PWD/backend" python3 -c "
from backend.vision import video_stream_capture
print(f'Advanced: {video_stream_capture.MACOS_CAPTURE_ADVANCED_AVAILABLE}')
print(f'Available: {video_stream_capture.MACOS_CAPTURE_AVAILABLE}')
"
```

**Output:**
```
✅ Advanced macOS capture available with legacy compatibility
Advanced: True
Available: True
```

### **Test #2: AVFoundation Availability**
```bash
PYTHONPATH="$PWD:$PWD/backend" python3 -c "
from backend.vision.macos_video_capture_advanced import check_capture_availability
import json
print(json.dumps(check_capture_availability(), indent=2))
"
```

**Output:**
```json
{
  "pyobjc_installed": true,
  "avfoundation_available": true,
  "screencapturekit_available": false,
  "macos_version": "14.1",
  "python_version": "3.9.6",
  "recommended_method": "AVFoundation",
  "memory_available_mb": 4037.42,
  "cpu_count": 8
}
```

### **Test #3: Start System Detection**
```bash
python3 start_system.py
```

**Output:**
```
✓ Advanced macOS video capture available (v10.6)
  • Native AVFoundation with async support
  • 🟣 Purple indicator enabled
  • Real-time metrics and adaptive quality
```

---

## Before vs After

### **Before (Broken)**
```
⚠ macOS capture frameworks not available - will use fallback: No module named 'AVFoundation'
⚠ Video streaming using fallback mode

Issues:
❌ Missing PIL dependency caused import failure
❌ NSObject import conflicts
❌ Duplicate Objective-C class registration
❌ Silent failures with no diagnostic info
❌ Incorrect fallback mode despite AVFoundation being installed
```

### **After (Fixed)**
```
✅ Advanced macOS capture available with legacy compatibility
✓ Advanced macOS video capture available (v10.6)
  • Native AVFoundation with async support
  • 🟣 Purple indicator enabled
  • Real-time metrics and adaptive quality

Features:
✅ All dependencies installed (PyObjC, Pillow)
✅ No import conflicts
✅ No duplicate class registration
✅ Comprehensive diagnostics
✅ Graceful fallback chain
✅ Advanced capture working correctly
```

---

## Files Modified

1. **`backend/vision/video_stream_capture.py`**
   - Fixed import chain (lines 23-81)
   - Fixed VideoFrameDelegate conditional (lines 324-348)
   - Added backward compatibility

2. **`start_system.py`**
   - Enhanced video capture diagnostics (lines 8214-8246)
   - Shows advanced vs legacy vs fallback status
   - Clear actionable error messages

3. **Dependencies Added**
   - `Pillow` (PIL support)
   - Already had: `pyobjc-framework-AVFoundation` and related

---

## Configuration

No configuration needed - works out of the box!

**Optional environment variables:**
```bash
# Capture settings
export Ironcliw_CAPTURE_DISPLAY_ID=0
export Ironcliw_CAPTURE_RESOLUTION=1920x1080
export Ironcliw_CAPTURE_FPS=30

# Performance
export Ironcliw_CAPTURE_ADAPTIVE=true
export Ironcliw_CAPTURE_MAX_MEMORY_MB=500

# Diagnostics
export Ironcliw_CAPTURE_DIAGNOSTICS=true
export Ironcliw_CAPTURE_LOG_METRICS=false
```

---

## Technical Details

### **Import Priority Chain**

1. **First Priority:** Advanced macOS Capture (v10.6)
   - Native AVFoundation via PyObjC
   - Async/await support
   - Real-time metrics
   - Purple indicator
   - Highest quality

2. **Second Priority:** Legacy macOS Capture
   - Basic AVFoundation
   - Synchronous
   - Works but limited features

3. **Third Priority:** Swift Video Bridge
   - Swift-based capture
   - Good compatibility

4. **Final Fallback:** Screenshot Loop
   - PIL/Pillow screenshots
   - Always works
   - Reduced quality

### **Graceful Degradation Strategy**

```python
if MACOS_CAPTURE_ADVANCED_AVAILABLE:
    # Use advanced capture (best)
    use_advanced_avfoundation()
elif MACOS_CAPTURE_AVAILABLE:
    # Use legacy capture (good)
    use_legacy_avfoundation()
elif SWIFT_BRIDGE_AVAILABLE:
    # Use Swift bridge (okay)
    use_swift_bridge()
else:
    # Use screenshot loop (works)
    use_screenshot_loop()
```

Each level gracefully falls back to the next if unavailable.

---

## Diagnostic Tools

### **test_avfoundation_imports.py**
```bash
python3 backend/vision/test_avfoundation_imports.py
```

**Tests:**
1. PyObjC core
2. Foundation framework
3. AVFoundation framework
4. CoreMedia framework
5. Quartz/CoreVideo
6. libdispatch
7. AVCaptureSession creation
8. AVCaptureScreenInput creation
9. Screen recording permission
10. Summary and recommendations

### **check_capture_availability()**
```python
from backend.vision.macos_video_capture_advanced import check_capture_availability
print(check_capture_availability())
```

---

## Status

**✅ COMPLETELY FIXED**

- ✅ All dependencies installed
- ✅ All imports working
- ✅ No conflicts
- ✅ Advanced capture available
- ✅ Purple indicator enabled
- ✅ Diagnostics comprehensive
- ✅ Fallback chain robust
- ✅ Backward compatible

**Version:** v10.6 (Advanced macOS Capture)
**Date:** December 27, 2025
**Status:** Production Ready

---

## Next Steps

The fallback warning is now fixed. When you run Ironcliw, you should see:

```
✓ Advanced macOS video capture available (v10.6)
  • Native AVFoundation with async support
  • 🟣 Purple indicator enabled
  • Real-time metrics and adaptive quality
```

Instead of:

```
⚠ Video streaming using fallback mode
```

If you still see the warning, run diagnostics:

```bash
# Test imports
python3 backend/vision/test_avfoundation_imports.py

# Check availability
PYTHONPATH="$PWD:$PWD/backend" python3 -c "
from backend.vision.video_stream_capture import MACOS_CAPTURE_ADVANCED_AVAILABLE
print(f'Advanced capture: {MACOS_CAPTURE_ADVANCED_AVAILABLE}')
"
```
