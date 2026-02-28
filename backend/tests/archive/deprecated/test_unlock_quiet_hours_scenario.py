#!/usr/bin/env python3
"""
Test Unlock During Quiet Hours Scenario
======================================

Simulates the scenario where user tries to unlock screen during quiet hours
"""

import asyncio
import websockets
import json
from datetime import datetime


async def test_unlock_scenario():
    """Test unlock my screen command via WebSocket"""
    print("\n🌙 Testing 'unlock my screen' During Quiet Hours")
    print("="*60)
    print(f"Current time: {datetime.now().strftime('%I:%M %p')}")
    
    current_hour = datetime.now().hour
    if 22 <= current_hour or current_hour < 7:
        print("✅ Currently in quiet hours (10 PM - 7 AM)")
    else:
        print("ℹ️  Not in quiet hours, but testing anyway")
    
    print("\n📡 Connecting to Ironcliw WebSocket...")
    
    try:
        # Connect to Ironcliw
        async with websockets.connect('ws://localhost:8000/voice/jarvis/stream') as ws:
            print("✅ Connected to Ironcliw")
            
            # Send the unlock command
            command = {
                "type": "command",
                "command": "unlock my screen",
                "source": "test"
            }
            
            print(f"\n🗣️  Sending: 'unlock my screen'")
            await ws.send(json.dumps(command))
            
            # Collect all responses
            responses = []
            print("\n📨 Responses from Ironcliw:")
            print("-"*40)
            
            # Wait for responses (with timeout)
            try:
                while True:
                    response = await asyncio.wait_for(ws.recv(), timeout=3.0)
                    data = json.loads(response)
                    responses.append(data)
                    
                    if data.get('text'):
                        print(f"   Ironcliw: {data['text']}")
                        if data.get('speak'):
                            print(f"           (spoken aloud)")
                    
                    # Check if this is the final response
                    if data.get('type') == 'response' and not data.get('intermediate'):
                        break
                        
            except asyncio.TimeoutError:
                # Normal - no more messages
                pass
            
            print("-"*40)
            
            # Analyze results
            print("\n📊 Analysis:")
            
            # Check if any response mentions policy/quiet hours
            policy_blocked = False
            for resp in responses:
                text = resp.get('text', '').lower()
                if 'policy' in text or 'quiet hours' in text:
                    policy_blocked = True
                    print("   ❌ Blocked by quiet hours policy")
                    break
            
            if not policy_blocked:
                # Check if unlock was mentioned
                unlock_mentioned = any('unlock' in r.get('text', '').lower() for r in responses)
                if unlock_mentioned:
                    print("   ✅ Manual unlock command processed!")
                    print("   ✅ No policy restrictions applied!")
                else:
                    print("   ⚠️  Unexpected response")
            
            # Show command type
            for resp in responses:
                if resp.get('command_type'):
                    print(f"   Command type: {resp['command_type']}")
                    break
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure Ironcliw is running on port 8000")
        return
    
    print("\n🎉 Test complete!")


if __name__ == "__main__":
    print("🔧 Manual Unlock During Quiet Hours Test")
    print("This tests that 'unlock my screen' works even during quiet hours")
    asyncio.run(test_unlock_scenario())