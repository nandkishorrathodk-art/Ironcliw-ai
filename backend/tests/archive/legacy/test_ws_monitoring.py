#!/usr/bin/env python3
"""Test Ironcliw monitoring through WebSocket"""

import asyncio
import websockets
import json

async def test_websocket_monitoring():
    """Test monitoring command through WebSocket"""
    uri = "ws://localhost:8000/voice/jarvis/stream"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Ironcliw WebSocket")
            
            # Wait for connected message
            message = await websocket.recv()
            data = json.loads(message)
            print(f"📨 Initial message: {data}")
            
            # Send monitoring command
            command = {
                "type": "command",
                "text": "start monitoring my screen"
            }
            print(f"\n🎙️ Sending command: {command}")
            await websocket.send(json.dumps(command))
            
            # Wait for response
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                print(f"\n📨 Response: {json.dumps(data, indent=2)}")
                
                if data.get('type') == 'response':
                    response_text = data.get('text', '')
                    if 'Failed to start' in response_text:
                        print("\n❌ Monitoring command failed!")
                    else:
                        print("\n✅ Monitoring command successful!")
                    break
                    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket_monitoring())