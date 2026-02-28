#!/usr/bin/env python3
"""
Runtime fix for vision streaming issue
This can be run while the backend is running to fix the issue
"""

import asyncio
import aiohttp
import json

async def send_jarvis_command(command: str):
    """Send a command to Ironcliw via API"""
    url = "http://localhost:8000/voice/jarvis/command"
    
    async with aiohttp.ClientSession() as session:
        try:
            response = await session.post(
                url,
                json={"text": command},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status == 200:
                result = await response.json()
                return result.get('response', 'No response')
            else:
                return f"Error: HTTP {response.status}"
                
        except Exception as e:
            return f"Error: {e}"

async def check_vision_status():
    """Check if vision system is properly initialized"""
    # Try a simple vision command
    response = await send_jarvis_command("test vision system")
    print(f"Vision test response: {response}")
    
    # Check if the error is about import
    if "vision analyzer not available" in response.lower():
        return False
    return True

async def restart_services():
    """Send commands to reinitialize services"""
    print("\n🔧 Attempting runtime fix...")
    
    # Try to trigger a reload
    commands = [
        "reload vision system",
        "reinitialize vision analyzer",
        "restart monitoring services"
    ]
    
    for cmd in commands:
        print(f"\n> {cmd}")
        response = await send_jarvis_command(cmd)
        print(f"< {response[:200]}...")

async def test_monitoring():
    """Test if monitoring works after fix"""
    print("\n🧪 Testing monitoring...")
    
    # Try to start monitoring
    print("\n> start monitoring my screen")
    response = await send_jarvis_command("start monitoring my screen")
    print(f"< {response[:200]}...")
    
    if "failed" in response.lower() or "error" in response.lower():
        print("\n❌ Still not working")
        return False
    else:
        print("\n✅ Monitoring started!")
        
        # Wait a bit
        await asyncio.sleep(5)
        
        # Stop monitoring
        print("\n> stop monitoring")
        response = await send_jarvis_command("stop monitoring")
        print(f"< {response[:200]}...")
        
        return True

async def main():
    print("🔧 Ironcliw Vision Runtime Fix")
    print("=" * 60)
    
    print("\n1️⃣ Checking current vision status...")
    vision_ok = await check_vision_status()
    
    if not vision_ok:
        print("⚠️  Vision system not properly initialized")
        print("\n2️⃣ Attempting runtime fix...")
        await restart_services()
    
    print("\n3️⃣ Testing monitoring command...")
    success = await test_monitoring()
    
    if success:
        print("\n✅ Fix successful! Vision monitoring is working.")
    else:
        print("\n❌ Runtime fix failed. Backend restart required.")
        print("\n🔄 To fix:")
        print("1. Stop the backend (Ctrl+C)")
        print("2. Start it again: python start_system.py")
        print("\nThe import fix has been applied and will work after restart.")

if __name__ == "__main__":
    asyncio.run(main())