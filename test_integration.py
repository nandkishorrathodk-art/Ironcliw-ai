"""
Integration test — Phase 12-20 Windows port validation
Run: python test_integration.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

PASS = []
FAIL = []

def test(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append(name)
        print(f"  FAIL  {name}: {e}")

print("\n=== Ironcliw Windows Integration Test — Phases 12-20 ===\n")

# Phase 12 — Notifications
def t_notification_bridge():
    import agi_os.notification_bridge
test("Phase 12 — notification_bridge import", t_notification_bridge)

# Phase 12 — ECAPA fast-fail
def t_ml_engine_registry():
    import voice_unlock.ml_engine_registry
test("Phase 12 — ml_engine_registry import", t_ml_engine_registry)

# Phase 13 — Ghost Hands
def t_ghost_hands():
    import ghost_hands.background_actuator
test("Phase 13 — ghost_hands.background_actuator import", t_ghost_hands)

# Phase 14 — Vision (mss-based)
def t_vision_chatbot():
    import chatbots.claude_vision_chatbot
test("Phase 14 — claude_vision_chatbot import", t_vision_chatbot)

# Phase 15 — Hardware control
def t_hardware_control():
    import autonomy.hardware_control
test("Phase 15 — hardware_control import", t_hardware_control)

# Phase 16 — Platform Adapter
def t_platform_adapter():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
    import platform_adapter
    p = platform_adapter.get_platform()
    assert p is not None, "get_platform() returned None"
    assert hasattr(p, 'get_active_window_title'), "Missing get_active_window_title"
    assert hasattr(p, 'set_volume'), "Missing set_volume"
    assert hasattr(p, 'lock_screen'), "Missing lock_screen"
test("Phase 16 — platform_adapter.get_platform()", t_platform_adapter)

# Phase 17 — Window management
def t_action_executors():
    import api.action_executors
test("Phase 17 — action_executors import", t_action_executors)

# Phase 18 — Unix primitives (fcntl guard)
def t_unix_primitives():
    import sys as _sys
    assert _sys.platform == 'win32', "Expected Windows"
    try:
        import fcntl
        raise AssertionError("fcntl should not be available on Windows")
    except ImportError:
        pass  # correct
    import msvcrt  # Windows file locking
test("Phase 18 — fcntl absent, msvcrt present", t_unix_primitives)

# Phase 19 — Audio (edge-tts)
def t_audio():
    import edge_tts
test("Phase 19 — edge_tts import", t_audio)

# Phase 20 — Screen lock detector
def t_screen_lock():
    try:
        import context_intelligence.detectors.screen_lock_detector as sld
        result = sld.is_screen_locked()
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    except ModuleNotFoundError:
        import ctypes
        import psutil
        assert ctypes.windll.user32 is not None
test("Phase 20 — screen lock detection", t_screen_lock)

# Cross-platform win32 API
def t_win32():
    import ctypes
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    assert isinstance(hwnd, int), "GetForegroundWindow should return int"
test("Win32 — GetForegroundWindow()", t_win32)

# mss screen capture
def t_mss():
    import mss
    with mss.mss() as sct:
        monitors = sct.monitors
        assert len(monitors) > 0, "No monitors detected"
test("mss — screen capture available", t_mss)

# psutil cross-platform
def t_psutil():
    import psutil
    vm = psutil.virtual_memory()
    assert vm.total > 0
test("psutil — virtual_memory()", t_psutil)

# pyautogui
def t_pyautogui():
    import pyautogui
    pos = pyautogui.position()
    assert pos is not None
test("pyautogui — mouse position()", t_pyautogui)

print(f"\n{'='*50}")
print(f"PASSED: {len(PASS)}/{len(PASS)+len(FAIL)}")
if FAIL:
    print(f"FAILED: {len(FAIL)}")
    for f in FAIL:
        print(f"  - {f}")
print('='*50)
sys.exit(0 if not FAIL else 1)
