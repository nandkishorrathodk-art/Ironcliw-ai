#!/usr/bin/env python3
"""
Test Ironcliw Activation
Verifies that all Ironcliw services are working properly
"""

import requests
import time
import websocket
import json
import threading

def test_backend_health():
    """Test if backend is healthy"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Backend health check passed")
            return True
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Backend not responding: {e}")
        return False

def test_voice_activation():
    """Test Ironcliw voice activation"""
    try:
        response = requests.post("http://localhost:8000/voice/jarvis/activate", timeout=5)
        if response.status_code == 200:
            print("✅ Ironcliw voice activation successful")
            return True
        else:
            print(f"❌ Ironcliw voice activation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Voice activation error: {e}")
        return False

def test_websocket_connection():
    """Test WebSocket connections"""
    success = True
    
    # Test TypeScript router
    try:
        ws = websocket.create_connection("ws://localhost:8001/ws/vision", timeout=5)
        ws.send(json.dumps({"type": "ping"}))
        ws.close()
        print("✅ TypeScript WebSocket router working")
    except Exception as e:
        print(f"❌ TypeScript router error: {e}")
        success = False
    
    # Test ML audio WebSocket
    try:
        ws = websocket.create_connection("ws://localhost:8000/audio/ml/stream", timeout=5)
        ws.close()
        print("✅ ML Audio WebSocket working")
    except Exception as e:
        print(f"❌ ML Audio WebSocket error: {e}")
        success = False
    
    return success

def test_vision_status():
    """Test vision system status"""
    try:
        response = requests.get("http://localhost:8000/vision/status", timeout=5)
        if response.status_code == 200:
            print("✅ Vision system status check passed")
            return True
        else:
            print(f"❌ Vision system status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Vision system error: {e}")
        return False

def test_ml_audio_config():
    """Test ML audio configuration"""
    try:
        response = requests.get("http://localhost:8000/audio/ml/config", timeout=5)
        if response.status_code == 200:
            print("✅ ML Audio configuration accessible")
            return True
        else:
            print(f"❌ ML Audio config failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ ML Audio config error: {e}")
        return False

def main():
    print("🤖 Ironcliw Activation Test Suite")
    print("=" * 50)
    
    # Wait a moment for services to stabilize
    print("\nWaiting for services to stabilize...")
    time.sleep(3)
    
    # Run all tests
    tests = [
        ("Backend Health", test_backend_health),
        ("Voice Activation", test_voice_activation),
        ("WebSocket Connections", test_websocket_connection),
        ("Vision System", test_vision_status),
        ("ML Audio Config", test_ml_audio_config)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Ironcliw is fully activated and operational!")
    else:
        print("\n⚠️  Some services are not working properly")
        print("Run 'python start_system.py' to start all services")

if __name__ == "__main__":
    main()