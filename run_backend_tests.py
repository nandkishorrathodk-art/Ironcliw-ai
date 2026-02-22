#!/usr/bin/env python3
"""
Complete test runner for backend/main.py on Windows.
Starts the server, runs tests, then stops it.
"""
import sys
import os
import time
import subprocess
import signal
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 70)
print("JARVIS Backend Complete Test Suite")
print("=" * 70)

# Test 1: Import test
print("\n[Phase 1] Import Test...")
result = subprocess.run([sys.executable, "test_backend_import.py"], capture_output=False)
if result.returncode != 0:
    print("[FAIL] Import test failed")
    sys.exit(1)
print("[OK] Import test passed")

# Test 2: Server startup and endpoint tests
print("\n[Phase 2] Server Startup Test...")
print("NOTE: This will start the FastAPI server on port 8010")
print("      Press Ctrl+C to stop the server and exit tests")
print("-" * 70)

server_process = None
try:
    # Check if uvicorn is available
    try:
        import uvicorn
        print("[OK] uvicorn is installed")
    except ImportError:
        print("[FAIL] uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)
    
    # Start the backend server
    print("\nStarting backend server...")
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8010"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    print("[OK] Server process started (PID: {})".format(server_process.pid))
    print("      Waiting for server to initialize...")
    
    # Monitor server output for a few seconds
    import select
    import time
    start_time = time.time()
    server_ready = False
    
    while time.time() - start_time < 30:  # Wait max 30 seconds
        # Check if server is still running
        if server_process.poll() is not None:
            print("[FAIL] Server process exited unexpectedly")
            print("      Reading output:")
            for line in server_process.stdout:
                print(f"      {line.rstrip()}")
            sys.exit(1)
        
        # Try to read output (non-blocking on Windows is tricky, so we'll just wait)
        time.sleep(1)
        
        # Check if server is responding
        try:
            import urllib.request
            req = urllib.request.Request('http://localhost:8010/health')
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    server_ready = True
                    break
        except:
            pass
    
    if not server_ready:
        print("[FAIL] Server did not become ready within 30 seconds")
        if server_process:
            server_process.terminate()
        sys.exit(1)
    
    print("[OK] Server is ready and responding")
    
    # Run endpoint tests
    print("\n[Phase 3] Endpoint Tests...")
    result = subprocess.run([sys.executable, "test_backend_server.py"], capture_output=False)
    test_passed = result.returncode == 0
    
    print("\n" + "=" * 70)
    if test_passed:
        print("[SUCCESS] ALL TESTS PASSED")
        print("=" * 70)
        print("\nBackend/main.py is working correctly on Windows!")
        print("- Platform detection: Working")
        print("- Server startup: Working")
        print("- Health endpoint: Working")
        print("- API endpoints: Working")
        print("- WebSocket: Working")
    else:
        print("[PARTIAL] Some tests failed - see output above")
        print("=" * 70)
    
except KeyboardInterrupt:
    print("\n\n[INFO] Test interrupted by user")
except Exception as e:
    print(f"\n[ERROR] Test runner error: {e}")
    import traceback
    traceback.print_exc()
finally:
    # Stop the server
    if server_process:
        print("\n[Cleanup] Stopping backend server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            print("[OK] Server stopped gracefully")
        except subprocess.TimeoutExpired:
            print("[WARN] Server did not stop gracefully, killing...")
            server_process.kill()
        print("[OK] Cleanup complete")

print("\n" + "=" * 70)
print("Test suite complete")
print("=" * 70)
