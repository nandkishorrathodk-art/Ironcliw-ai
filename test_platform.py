#!/usr/bin/env python3
"""Quick test for platform detection"""

from backend.platform import get_platform, get_platform_info, is_windows, is_macos, is_linux

print("JARVIS Platform Detection Test")
print("=" * 60)

platform = get_platform()
print(f"Platform: {platform}")
print(f"Is Windows: {is_windows()}")
print(f"Is macOS: {is_macos()}")
print(f"Is Linux: {is_linux()}")

info = get_platform_info()
print(f"\nPlatform Info:")
print(f"  OS: {info.os_release}")
print(f"  Architecture: {info.architecture}")
print(f"  Python: {info.python_version}")
print(f"  GPU: {info.has_gpu}")
print(f"  NPU: {info.has_npu}")

assert platform == 'windows', f"Expected 'windows', got '{platform}'"
assert is_windows(), "is_windows() should return True"
assert not is_macos(), "is_macos() should return False"
assert not is_linux(), "is_linux() should return False"

print("\n[PASS] All tests passed!")
