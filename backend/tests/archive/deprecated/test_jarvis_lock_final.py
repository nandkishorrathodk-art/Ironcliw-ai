#!/usr/bin/env python3
"""
Test Ironcliw Lock Command - Final
================================

Test if "lock my screen" works through Ironcliw now.
"""

import asyncio
import aiohttp
import json

async def test_jarvis_lock():
    """Test lock command through Ironcliw"""
    print("🔐 Testing Ironcliw Lock Command")
    print("=" * 50)
    
    url = "http://localhost:8000/api/voice-command"
    headers = {'Content-Type': 'application/json'}
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test command
            data = {
                "text": "lock my screen",
                "audio_data": None
            }
            
            print(f"\nSending to Ironcliw: '{data['text']}'")
            print("⚠️  This will lock your screen if it works!")
            
            async with session.post(url, json=data, headers=headers) as response:
                result = await response.json()
                
                print(f"\nIroncliw Response:")
                print(f"Success: {result.get('success')}")
                print(f"Response: {result.get('response')}")
                
                if result.get('success'):
                    print("\n✅ Command processed successfully!")
                else:
                    print(f"\n❌ Command failed")
                    
    except aiohttp.ClientConnectorError:
        print("\n❌ Could not connect to Ironcliw on port 8000")
        print("Make sure Ironcliw is running: python main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    print("This test will send 'lock my screen' to Ironcliw")
    print("Your screen will lock if the command works!")
    
    response = input("\nContinue? (y/N): ")
    
    if response.lower() == 'y':
        asyncio.run(test_jarvis_lock())
    else:
        print("Test cancelled.")