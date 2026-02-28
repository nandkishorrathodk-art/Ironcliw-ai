#!/usr/bin/env python3
"""
Test Ironcliw voice WebSocket with vision command
"""

import asyncio
import websockets
import json

async def test_jarvis_voice_vision():
    """Test vision command through Ironcliw voice WebSocket"""
    uri = "ws://localhost:8010/voice/jarvis/stream"
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected to Ironcliw Voice WebSocket")
            
            # Wait for connection message
            msg = await websocket.recv()
            print(f"📥 Connection message: {json.loads(msg)}")
            
            # Send vision command
            command = {
                "text": "can you see my screen"
            }
            
            print(f"\n📤 Sending: {json.dumps(command, indent=2)}")
            await websocket.send(json.dumps(command))
            
            # Wait for response
            response = await asyncio.wait_for(websocket.recv(), timeout=15.0)
            response_data = json.loads(response)
            
            print(f"\n📥 Received: {json.dumps(response_data, indent=2)}")
            
            if response_data.get("text"):
                print(f"\n✅ Ironcliw Response: {response_data['text'][:200]}...")
            else:
                print(f"\n❌ Unexpected response format")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

async def main():
    print("🔍 Testing Ironcliw Voice Vision Command")
    print("=" * 50)
    await test_jarvis_voice_vision()

if __name__ == "__main__":
    asyncio.run(main())