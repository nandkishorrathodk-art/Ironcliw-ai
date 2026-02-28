#!/usr/bin/env python3
"""
Test the updated startup components
"""

import subprocess
import time
import requests
import sys

def test_websocket_build():
    """Test if WebSocket router builds successfully"""
    print("1️⃣ Testing WebSocket build...")
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd="backend/websocket",
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ WebSocket build successful")
        return True
    else:
        print(f"❌ WebSocket build failed: {result.stderr}")
        return False

def test_websocket_health():
    """Test if WebSocket router starts and responds to health check"""
    print("\n2️⃣ Testing WebSocket health endpoint...")
    
    # Start the router
    process = subprocess.Popen(
        ["npm", "start"],
        cwd="backend/websocket",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # Give it time to start
    time.sleep(5)
    
    try:
        # Check health endpoint
        response = requests.get("http://localhost:8001/health", timeout=5)
        if response.status_code == 200:
            print("✅ WebSocket health check passed")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Failed to connect to health endpoint: {e}")
        return False
    finally:
        # Stop the process
        process.terminate()
        process.wait()

def test_backend_import():
    """Test if backend main.py can be imported"""
    print("\n3️⃣ Testing backend imports...")
    
    sys.path.insert(0, 'backend')
    try:
        import main
        print("✅ Backend main.py imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import main.py: {e}")
        return False

def test_vision_system():
    """Test if vision system is functional"""
    print("\n4️⃣ Testing vision system...")
    
    try:
        from vision.screen_capture_fallback import capture_screen_fallback
        screenshot = capture_screen_fallback()
        if screenshot:
            print("✅ Vision capture working")
            return True
        else:
            print("❌ Vision capture failed")
            return False
    except Exception as e:
        print(f"❌ Vision system error: {e}")
        return False

def main():
    print("🔍 Ironcliw Startup Test Suite")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("WebSocket Build", test_websocket_build()))
    results.append(("WebSocket Health", test_websocket_health()))
    results.append(("Backend Import", test_backend_import()))
    results.append(("Vision System", test_vision_system()))
    
    # Summary
    print("\n📊 Test Summary:")
    print("=" * 50)
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    if all_passed:
        print("\n✨ All tests passed! System is ready to start.")
        print("\nYou can now run:")
        print("  • python start_system.py")
        print("  • python start_minimal.py")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")

if __name__ == "__main__":
    main()