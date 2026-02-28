#!/usr/bin/env python3
"""
Quick health check test - verifies the backend can import and return platform info.
This doesn't require starting the actual server.
"""
import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 70)
print("Quick Platform & Health Check Test")
print("=" * 70)

# Import backend.main
try:
    print("\n[1/4] Importing backend.main...")
    import backend.main as main
    print("[OK] backend.main imported")
except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    sys.exit(1)

# Check platform detection
try:
    print("\n[2/4] Checking platform detection...")
    print(f"[OK] Platform: {main.Ironcliw_PLATFORM}")
    print(f"[OK] Is Windows: {main.Ironcliw_IS_WINDOWS}")
    print(f"[OK] Is macOS: {main.Ironcliw_IS_MACOS}")
    print(f"[OK] Is Linux: {main.Ironcliw_IS_LINUX}")
    
    if main.Ironcliw_PLATFORM_INFO:
        print(f"[OK] OS Release: {main.Ironcliw_PLATFORM_INFO.os_release}")
        print(f"[OK] Architecture: {main.Ironcliw_PLATFORM_INFO.architecture}")
except Exception as e:
    print(f"[FAIL] Platform check failed: {e}")
    sys.exit(1)

# Check component loading
try:
    print("\n[3/4] Checking loaded components...")
    if hasattr(main, 'components'):
        loaded = [name for name, comp in main.components.items() if comp]
        print(f"[OK] Loaded components: {len(loaded)}")
        for comp_name in sorted(loaded)[:5]:
            print(f"     - {comp_name}")
        if len(loaded) > 5:
            print(f"     ... and {len(loaded) - 5} more")
    else:
        print("[WARN] Components not yet loaded (server not started)")
except Exception as e:
    print(f"[WARN] Component check failed: {e}")

# Check FastAPI app
try:
    print("\n[4/4] Checking FastAPI app...")
    if hasattr(main, 'app'):
        print(f"[OK] FastAPI app created")
        print(f"[OK] Total routes: {len(main.app.routes)}")
        
        # Check for key routes
        route_paths = [getattr(r, 'path', '') for r in main.app.routes if hasattr(r, 'path')]
        key_routes = ['/health', '/lock-now', '/api/command', '/ws']
        found_routes = [r for r in key_routes if any(r in path for path in route_paths)]
        print(f"[OK] Key routes found: {len(found_routes)}/{len(key_routes)}")
        for route in found_routes:
            print(f"     - {route}")
    else:
        print("[FAIL] FastAPI app not created")
        sys.exit(1)
except Exception as e:
    print(f"[FAIL] FastAPI check failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("[SUCCESS] Backend is ready for Windows!")
print("=" * 70)
print("\nNext steps:")
print("  1. Start server: python -m uvicorn backend.main:app --port 8010")
print("  2. Test health: curl http://localhost:8010/health")
print("  3. Check platform: Look for 'platform' key in health response")
