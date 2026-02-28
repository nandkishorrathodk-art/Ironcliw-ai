#!/usr/bin/env python3
"""
Test Final Unlock Fix
=====================

Tests that "unlock my screen" is properly routed to voice unlock handler
"""

import asyncio
import json
from datetime import datetime
import websockets


async def test_unlock_command():
    """Test unlock my screen command via WebSocket"""
    print("\n🧪 Testing Manual Unlock Command Fix")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%I:%M %p')}")
    
    try:
        # Wait a bit more for Ironcliw to be ready
        print("\n⏳ Waiting for Ironcliw to be fully ready...")
        await asyncio.sleep(5)
        
        print("\n📡 Connecting to Ironcliw WebSocket...")
        
        # Connect to Ironcliw
        async with websockets.connect('ws://localhost:8000/voice/jarvis/stream') as ws:
            print("✅ Connected to Ironcliw")
            
            # Send the unlock command as text (like the frontend does)
            command_data = {
                "audio_chunk": "",  # Empty audio
                "text": "unlock my screen",  # Text command
                "source": "test"
            }
            
            print(f"\n🗣️  Sending: 'unlock my screen'")
            await ws.send(json.dumps(command_data))
            
            # Collect responses
            responses = []
            print("\n📨 Responses from Ironcliw:")
            print("-"*40)
            
            # Wait for responses
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    data = json.loads(response)
                    responses.append(data)
                    
                    if data.get('text'):
                        print(f"   Ironcliw: {data['text']}")
                        if data.get('speak'):
                            print(f"           (spoken aloud)")
                    
                    # Check command type
                    if data.get('command_type'):
                        print(f"   [Command type: {data['command_type']}]")
                    
                    # Check if this is the final response
                    if data.get('type') == 'response' and not data.get('intermediate'):
                        break
                        
            except asyncio.TimeoutError:
                # Normal - no more messages
                pass
            
            print("-"*40)
            
            # Analyze results
            print("\n📊 Analysis:")
            
            # Check if we got a proper response
            success = False
            for resp in responses:
                text = resp.get('text', '').lower()
                if 'unlock' in text and ('successfully' in text or "i'll unlock" in text):
                    success = True
                    print("   ✅ Unlock command processed successfully!")
                    break
                elif "don't have a handler" in text:
                    print("   ❌ Still getting handler error")
                    break
                elif "couldn't unlock" in text:
                    print("   ⚠️  Unlock failed (check daemon)")
                    success = True  # Command was processed correctly
                    break
            
            if not success:
                print("   ❌ Unexpected response")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure Ironcliw is running on port 8000")
        return
    
    print("\n🎉 Test complete!")


if __name__ == "__main__":
    print("🔧 Testing Manual Unlock Command Fix")
    print("This verifies that 'unlock my screen' is properly handled")
    asyncio.run(test_unlock_command())