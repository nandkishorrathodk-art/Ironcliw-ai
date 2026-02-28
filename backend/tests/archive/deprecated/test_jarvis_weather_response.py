#!/usr/bin/env python3
"""Test Ironcliw weather response in limited mode"""

import asyncio
import json
import websockets
import time

async def test_weather_via_websocket():
    """Test weather command through WebSocket"""
    print("🌤️ Testing Ironcliw Weather Response via WebSocket\n")
    
    uri = "ws://localhost:8000/voice/jarvis/stream"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Ironcliw WebSocket")
            
            # Wait for greeting
            greeting = await websocket.recv()
            print(f"📥 Received: {json.loads(greeting)['type']}")
            
            # Send weather command
            command = {
                "type": "command",
                "text": "What's the weather today?"
            }
            
            print(f"\n📤 Sending: {command['text']}")
            await websocket.send(json.dumps(command))
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            
            print(f"\n📥 Response type: {data.get('type')}")
            print(f"📝 Text: {data.get('text', 'No text')}")
            
            # Check if Weather app was opened
            time.sleep(1)
            import subprocess
            result = subprocess.run(
                ['osascript', '-e', 'tell application "System Events" to get name of (processes where frontmost is true)'],
                capture_output=True,
                text=True
            )
            
            if "Weather" in result.stdout:
                print("\n✅ Weather app is now active!")
            
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
        print("\nMake sure the Ironcliw backend is running with:")
        print("  cd backend && python main.py")

if __name__ == "__main__":
    asyncio.run(test_weather_via_websocket())