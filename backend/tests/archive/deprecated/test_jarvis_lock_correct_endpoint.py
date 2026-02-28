#!/usr/bin/env python3
"""
Test Ironcliw Lock Command - Correct Endpoint
===========================================

Test if "lock my screen" works through Ironcliw using the correct endpoint.
"""

import asyncio
import aiohttp
import json

async def test_jarvis_lock():
    """Test lock command through Ironcliw"""
    print("🔐 Testing Ironcliw Lock Command")
    print("=" * 50)
    
    url = "http://localhost:8000/api/command"  # Correct endpoint
    headers = {'Content-Type': 'application/json'}
    
    try:
        async with aiohttp.ClientSession() as session:
            # Test command
            data = {
                "text": "lock my screen",
            }
            
            print(f"\nSending to Ironcliw: '{data['text']}'")
            print("⚠️  This will lock your screen if it works!")
            
            countdown = 3
            while countdown > 0:
                print(f"{countdown}...")
                await asyncio.sleep(1)
                countdown -= 1
            
            async with session.post(url, json=data, headers=headers) as response:
                print(f"\nHTTP Status: {response.status}")
                
                # Get raw response first
                text = await response.text()
                print(f"\nRaw Response: {text}")
                
                try:
                    result = json.loads(text)
                    print(f"\nIroncliw Response:")
                    print(json.dumps(result, indent=2))
                    
                    if result.get('success'):
                        print("\n✅ Command processed successfully!")
                        print(f"Response: {result.get('response')}")
                    else:
                        print(f"\n❌ Command failed")
                        if result.get('error'):
                            print(f"Error: {result.get('error')}")
                except:
                    print("Could not parse JSON response")
                    
    except aiohttp.ClientConnectorError:
        print("\n❌ Could not connect to Ironcliw on port 8000")
        print("Make sure Ironcliw is running: python main.py")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")

async def test_simple_command():
    """Test a simple command first"""
    print("\n📝 Testing Simple Command First")
    print("=" * 50)
    
    url = "http://localhost:8000/api/command"
    
    try:
        async with aiohttp.ClientSession() as session:
            data = {"text": "what time is it"}
            
            print(f"\nSending: '{data['text']}'")
            
            async with session.post(url, json=data) as response:
                result = await response.json()
                print(f"\nResponse: {result.get('response', 'No response')}")
                
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    print("This test will send 'lock my screen' to Ironcliw")
    print("Your screen WILL LOCK if the command works!")
    
    response = input("\nContinue? (y/N): ")
    
    if response.lower() == 'y':
        asyncio.run(test_simple_command())
        
        print("\n" + "="*50)
        response2 = input("\nNow test lock command? (y/N): ")
        
        if response2.lower() == 'y':
            asyncio.run(test_jarvis_lock())
        else:
            print("Lock test cancelled.")
    else:
        print("Test cancelled.")