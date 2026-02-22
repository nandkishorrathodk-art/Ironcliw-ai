"""
Test if unified_supervisor.py can be imported on Windows.
"""
import sys
import os

# Fix Windows console encoding for Unicode
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("Python version:", sys.version)
print("Platform:", sys.platform)

try:
    import unified_supervisor
    print("[PASS] unified_supervisor.py imports successfully")
    print(f"Platform flags: Windows={unified_supervisor._is_windows}, Linux={unified_supervisor._is_linux}, macOS={unified_supervisor._is_macos}")
except Exception as e:
    print(f"[FAIL] Failed to import unified_supervisor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[PASS] All tests passed!")
