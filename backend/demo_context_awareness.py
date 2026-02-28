#!/usr/bin/env python3
"""
Demo Context Awareness for Ironcliw
=================================

Demonstrates the context-aware screen lock/unlock functionality
"""

import asyncio
import aiohttp
import json
import time
import subprocess


def run_voice_command(command: str):
    """Simulate voice command through React app"""
    print(f"\n🎤 Voice command: '{command}'")
    print("(This would normally come through the React app's voice interface)")


async def send_command(command: str):
    """Send command to Ironcliw backend"""
    api_url = "http://localhost:8000/api/command"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(api_url, json={"command": command}) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result
                else:
                    return {"error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"error": str(e)}


def check_screen_locked():
    """Check if screen is locked"""
    try:
        result = subprocess.run(['python', '-c', '''
import Quartz
session_dict = Quartz.CGSessionCopyCurrentDictionary()
if session_dict:
    locked = session_dict.get("CGSSessionScreenIsLocked", False)
    print("true" if locked else "false")
else:
    print("false")
'''], capture_output=True, text=True)
        
        return result.stdout.strip().lower() == "true"
    except Exception as e:
        print(f"Error checking screen lock: {e}")
        return False


async def demo_context_awareness():
    """Demonstrate context-aware screen handling"""
    
    print("🎯 Ironcliw Context Awareness Demo")
    print("=" * 60)
    print("\nThis demo shows how Ironcliw handles commands when your screen is locked.")
    print("The context awareness feature will:")
    print("1. Detect when the screen is locked")
    print("2. Inform you it needs to unlock")
    print("3. Type in your password to unlock") 
    print("4. Execute your command")
    print("5. Confirm what was done")
    print("\n" + "=" * 60)
    
    # Step 1: Normal command test
    print("\n✅ Step 1: Testing normal command (screen unlocked)")
    run_voice_command("What time is it?")
    
    result = await send_command("What time is it?")
    print(f"Ironcliw response: {result.get('response', result)}")
    
    await asyncio.sleep(2)
    
    # Step 2: Lock the screen
    print("\n✅ Step 2: Locking the screen")
    run_voice_command("Lock my screen")
    
    result = await send_command("lock my screen")
    print(f"Ironcliw response: {result.get('response', result)}")
    
    await asyncio.sleep(3)
    is_locked = check_screen_locked()
    print(f"Screen locked: {is_locked}")
    
    if not is_locked:
        print("\n⚠️  Screen lock didn't work. You may need to:")
        print("1. Enable screen lock in System Preferences")
        print("2. Have voice unlock properly set up")
        print("3. Manually lock your screen (Cmd+Ctrl+Q)")
        print("\nPress Enter after locking your screen manually...")
        input()
    
    # Step 3: Try a command that requires screen access
    print("\n✅ Step 3: Command that requires screen access (while locked)")
    print("\n🔐 Your screen is currently locked.")
    print("Now let's see how Ironcliw handles this...")
    
    await asyncio.sleep(2)
    
    command = "Open Safari and search for artificial intelligence"
    run_voice_command(command)
    print("\n⏳ Sending command to Ironcliw...")
    
    result = await send_command(command)
    
    print("\n📝 Ironcliw Response:")
    print("-" * 40)
    print(f"Response: {result.get('response', 'No response')}")
    print(f"Success: {result.get('success', False)}")
    print(f"Context handled: {result.get('context_handled', False)}")
    print(f"Screen unlocked: {result.get('screen_unlocked', False)}")
    print("-" * 40)
    
    await asyncio.sleep(3)
    
    # Check final screen state
    final_locked = check_screen_locked()
    print(f"\nScreen locked after command: {final_locked}")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("\nWhat you should have seen:")
    print("1. ✅ Ironcliw locked your screen")
    print("2. ✅ Ironcliw detected the locked screen")
    print("3. ✅ Ironcliw said it would unlock by typing the password")
    print("4. ✅ Ironcliw unlocked your screen")
    print("5. ✅ Ironcliw opened Safari and searched")
    print("6. ✅ Ironcliw confirmed all actions taken")
    
    print("\n💡 This demonstrates how Ironcliw is context-aware and handles")
    print("   situations intelligently without breaking the user experience!")


if __name__ == "__main__":
    print("\n🚀 Starting Ironcliw Context Awareness Demo\n")
    print("Make sure:")
    print("1. Ironcliw backend is running on port 8000")
    print("2. Voice Unlock is set up and working")
    print("3. You have Safari installed")
    print("\nPress Enter to continue...")
    input()
    
    asyncio.run(demo_context_awareness())