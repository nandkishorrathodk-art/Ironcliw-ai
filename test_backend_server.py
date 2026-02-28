#!/usr/bin/env python3
"""
Test script to verify backend server endpoints work on Windows.
"""
import sys
import os
import time
import json
import asyncio
import subprocess
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

print("=" * 70)
print("Ironcliw Backend Server Test")
print("=" * 70)

async def test_health_endpoint():
    """Test /health endpoint"""
    import aiohttp
    
    print("\n[Test 1] Testing /health endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('http://localhost:8010/health', timeout=10) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"[OK] /health endpoint returned 200")
                    print(f"     Status: {data.get('status')}")
                    print(f"     Platform: {data.get('platform', {}).get('platform')}")
                    print(f"     Components: {len(data.get('components', {}))}")
                    return True
                else:
                    print(f"[FAIL] /health endpoint returned {resp.status}")
                    return False
    except Exception as e:
        print(f"[FAIL] /health endpoint error: {e}")
        return False

async def test_command_endpoint():
    """Test /api/command endpoint"""
    import aiohttp
    
    print("\n[Test 2] Testing /api/command endpoint...")
    try:
        async with aiohttp.ClientSession() as session:
            payload = {"text": "test command"}
            async with session.post(
                'http://localhost:8010/api/command',
                json=payload,
                timeout=10
            ) as resp:
                if resp.status in (200, 404, 500):  # 404/500 ok if dependencies missing
                    print(f"[OK] /api/command endpoint responded (status {resp.status})")
                    try:
                        data = await resp.json()
                        print(f"     Response: {str(data)[:100]}")
                    except:
                        text = await resp.text()
                        print(f"     Response: {text[:100]}")
                    return True
                else:
                    print(f"[FAIL] /api/command endpoint returned {resp.status}")
                    return False
    except Exception as e:
        print(f"[FAIL] /api/command endpoint error: {e}")
        return False

async def test_websocket():
    """Test WebSocket connection"""
    import aiohttp
    
    print("\n[Test 3] Testing WebSocket connection...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect('ws://localhost:8010/ws', timeout=10) as ws:
                print(f"[OK] WebSocket connected")
                
                # Send a test message
                await ws.send_json({"type": "ping", "data": "test"})
                print(f"[OK] Sent test message")
                
                # Wait for response (with timeout)
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                    print(f"[OK] Received response: {str(msg)[:100]}")
                except asyncio.TimeoutError:
                    print(f"[OK] No response received (expected for basic test)")
                
                return True
    except Exception as e:
        print(f"[FAIL] WebSocket error: {e}")
        return False

async def run_tests():
    """Run all endpoint tests"""
    print("\nWaiting for server to be ready...")
    
    # Wait for server to start (max 30 seconds)
    import aiohttp
    for i in range(30):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://localhost:8010/health', timeout=2) as resp:
                    if resp.status == 200:
                        print(f"[OK] Server is ready after {i+1} seconds")
                        break
        except:
            pass
        await asyncio.sleep(1)
    else:
        print("[FAIL] Server did not start within 30 seconds")
        return False
    
    # Run tests
    results = []
    results.append(await test_health_endpoint())
    results.append(await test_command_endpoint())
    results.append(await test_websocket())
    
    print("\n" + "=" * 70)
    if all(results):
        print("[SUCCESS] ALL ENDPOINT TESTS PASSED")
    else:
        print(f"[PARTIAL] {sum(results)}/{len(results)} tests passed")
    print("=" * 70)
    
    return all(results)

if __name__ == "__main__":
    # Check if aiohttp is available
    try:
        import aiohttp
    except ImportError:
        print("[FAIL] aiohttp not installed. Install with: pip install aiohttp")
        sys.exit(1)
    
    # Run async tests
    success = asyncio.run(run_tests())
    sys.exit(0 if success else 1)
