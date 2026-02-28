#!/usr/bin/env python3
"""
Test Voice Unlock Direct
========================

Test Voice Unlock daemon directly to debug unlock issues.
"""

import asyncio
import websockets
import json

async def test_unlock():
    """Test unlock command directly"""
    print("🔓 TESTING VOICE UNLOCK DAEMON DIRECTLY")
    print("=" * 50)
    
    uri = "ws://localhost:8765/voice-unlock"
    
    try:
        async with websockets.connect(uri) as ws:
            print("✅ Connected to Voice Unlock daemon")
            
            # First get status
            print("\n1️⃣ Getting status...")
            status_cmd = json.dumps({
                "type": "command",
                "command": "get_status"
            })
            await ws.send(status_cmd)
            response = await ws.recv()
            result = json.loads(response)
            print(f"Status: {json.dumps(result, indent=2)}")
            
            # Now try to unlock
            print("\n2️⃣ Sending unlock_screen command...")
            print("⚠️  This will attempt to unlock your screen!")
            await asyncio.sleep(2)
            
            unlock_cmd = json.dumps({
                "type": "command",
                "command": "unlock_screen"
            })
            
            print(f"Sending: {unlock_cmd}")
            await ws.send(unlock_cmd)
            
            # Wait for response with longer timeout
            print("Waiting for response (15s timeout)...")
            try:
                response = await asyncio.wait_for(ws.recv(), timeout=15.0)
                result = json.loads(response)
                
                print(f"\n3️⃣ Response received:")
                print(json.dumps(result, indent=2))
                
                if result.get('success'):
                    print("\n✅ Unlock command succeeded!")
                else:
                    print(f"\n❌ Unlock failed: {result.get('message', 'Unknown error')}")
                    
            except asyncio.TimeoutError:
                print("\n❌ TIMEOUT: No response after 15 seconds")
                print("Voice Unlock daemon may be stuck or not properly configured")
                
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")

async def check_daemon_log():
    """Check Voice Unlock daemon log"""
    print("\n\n📋 CHECKING DAEMON LOG")
    print("=" * 50)
    
    import subprocess
    import os
    
    log_path = os.path.expanduser("~/Documents/repos/Ironcliw-AI-Agent/backend/voice_unlock/objc/server/websocket_server.log")
    
    if os.path.exists(log_path):
        # Get last 20 lines
        result = subprocess.run(['tail', '-20', log_path], capture_output=True, text=True)
        print("Last 20 lines of daemon log:")
        print(result.stdout)
    else:
        print("No log file found")
        
    # Check if daemon is running
    result = subprocess.run(['pgrep', '-f', 'websocket_server.py'], capture_output=True, text=True)
    if result.stdout:
        print(f"\nDaemon PID: {result.stdout.strip()}")
    else:
        print("\nDaemon not running!")

async def main():
    """Run tests"""
    await test_unlock()
    await check_daemon_log()

if __name__ == "__main__":
    asyncio.run(main())