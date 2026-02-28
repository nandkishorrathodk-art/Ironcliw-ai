#!/usr/bin/env python3
"""
Test script to verify:
1. Purple indicator appears when monitoring starts
2. Purple indicator disappears when "stop monitoring" command is issued
3. Microphone toggle functionality
"""

import asyncio
import aiohttp
import json
import time

async def test_monitoring_commands():
    """Test start and stop monitoring commands"""
    base_url = "http://localhost:8000"
    
    print("\n🟣 Testing Purple Indicator & Stop Monitoring")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        # Test 1: Start monitoring
        print("\n1️⃣ Testing 'start monitoring' command...")
        command = "start monitoring my screen"
        
        response = await session.post(
            f"{base_url}/voice/jarvis/command",
            json={"text": command}
        )
        
        result = await response.json()
        print(f"Command: {command}")
        print(f"Response: {result.get('response', 'No response')}")
        print("\n✅ CHECK: Purple indicator should be visible in menu bar!")
        
        # Wait a bit for user to see the indicator
        print("\n⏳ Waiting 5 seconds for you to verify purple indicator...")
        await asyncio.sleep(5)
        
        # Test 2: Stop monitoring
        print("\n2️⃣ Testing 'stop monitoring' command...")
        command = "stop monitoring my screen"
        
        response = await session.post(
            f"{base_url}/voice/jarvis/command",
            json={"text": command}
        )
        
        result = await response.json()
        print(f"Command: {command}")
        print(f"Response: {result.get('response', 'No response')}")
        print("\n✅ CHECK: Purple indicator should disappear!")
        
        # Test 3: Verify it's really stopped
        await asyncio.sleep(2)
        print("\n3️⃣ Verifying monitoring is stopped...")
        
        # Try alternative stop commands
        alt_commands = [
            "stop watching my screen",
            "disable monitoring",
            "turn off screen capture"
        ]
        
        print("\n4️⃣ Testing alternative stop commands...")
        for cmd in alt_commands:
            print(f"\nTrying: {cmd}")
            response = await session.post(
                f"{base_url}/voice/jarvis/command",
                json={"text": cmd}
            )
            result = await response.json()
            print(f"Response: {result.get('response', 'No response')}")
            await asyncio.sleep(1)

async def main():
    print("\n🎯 Ironcliw Purple Indicator & Monitoring Test")
    print("=" * 60)
    print("\n📋 This test will:")
    print("1. Start screen monitoring (purple indicator should appear)")
    print("2. Stop screen monitoring (purple indicator should disappear)")
    print("3. Test various stop commands")
    
    print("\n⚠️  Prerequisites:")
    print("- Ironcliw backend must be running (python start_system.py)")
    print("- Screen recording permission must be granted")
    
    input("\nPress Enter to begin test...")
    
    try:
        await test_monitoring_commands()
        print("\n✅ Test completed!")
        print("\n📝 Frontend Microphone Toggle:")
        print("1. Open http://localhost:3000")
        print("2. Click 'Activate Ironcliw'")
        print("3. Click '🎤 Start Listening' - mic should stay on")
        print("4. Say 'Hey Ironcliw' multiple times - should always respond")
        print("5. Click '🔴 Stop Listening' - mic should turn off")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("Make sure Ironcliw backend is running!")

if __name__ == "__main__":
    asyncio.run(main())