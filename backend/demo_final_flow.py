#!/usr/bin/env python3
"""
FINAL DEMO: Context Intelligence Automated Flow
===============================================

Demonstrates the complete automated flow as described in the PRD:
1. Lock screen
2. Issue command requiring screen access
3. Ironcliw detects lock, queues command, unlocks screen, executes command
4. All fully automated with proper feedback
"""

import asyncio
import aiohttp
import json
import time
import sys

async def wait_for_jarvis():
    """Wait for Ironcliw to be ready"""
    print("⏳ Waiting for Ironcliw to start...")
    for i in range(30):  # Wait up to 30 seconds
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/") as response:
                    if response.status == 200:
                        print("✅ Ironcliw is ready!")
                        return True
        except:
            pass
        await asyncio.sleep(1)
    return False

async def demo_automated_flow():
    """Demonstrate the complete automated flow"""
    print("\n" + "="*70)
    print("🤖 CONTEXT INTELLIGENCE AUTOMATED FLOW DEMONSTRATION")
    print("="*70)
    print("\nThis demonstration shows the complete flow from the PRD:")
    print("1. Lock the Mac screen")
    print("2. Say 'Ironcliw, open Safari and search for dogs'")
    print("3. Ironcliw will:")
    print("   - Detect screen is locked ✓")
    print("   - Queue the request ✓")
    print("   - Provide feedback: 'Your screen is locked, unlocking now' ✓")
    print("   - Automatically unlock by typing password ✓")
    print("   - Execute the queued command (open Safari, search) ✓")
    print("   - Report success: 'I unlocked your screen and opened Safari...' ✓")
    print("\n⚠️  YOUR SCREEN WILL LOCK AND AUTOMATICALLY UNLOCK!")
    print("="*70)
    
    await asyncio.sleep(5)
    
    url = "http://localhost:8000/api/command"
    
    try:
        async with aiohttp.ClientSession() as session:
            # Step 1: Lock the screen
            print(f"\n[{time.strftime('%H:%M:%S')}] 🔒 STEP 1: Locking screen...")
            data = {"command": "lock my screen"}
            
            async with session.post(url, json=data) as response:
                result = await response.json()
                print(f"[{time.strftime('%H:%M:%S')}] Ironcliw: {result.get('response', 'No response')}")
                
            # Wait for lock to take effect
            print(f"\n[{time.strftime('%H:%M:%S')}] ⏳ Waiting 5 seconds for screen to lock...")
            await asyncio.sleep(5)
            
            # Step 2: Issue command that requires screen
            print(f"\n[{time.strftime('%H:%M:%S')}] 🗣️  STEP 2: User says: 'Ironcliw, open Safari and search for dogs'")
            print(f"[{time.strftime('%H:%M:%S')}] 📡 Sending command to Ironcliw...")
            
            start_time = time.time()
            data = {"command": "open safari and search for dogs"}
            
            # Make request with longer timeout
            timeout = aiohttp.ClientTimeout(total=30)
            async with session.post(url, json=data, timeout=timeout) as response:
                if response.status == 200:
                    result = await response.json()
                    elapsed = time.time() - start_time
                    
                    print(f"\n[{time.strftime('%H:%M:%S')}] ✅ Ironcliw RESPONSE:")
                    print(f"[{time.strftime('%H:%M:%S')}] '{result.get('response', 'No response')}'")
                    print(f"[{time.strftime('%H:%M:%S')}] Time taken: {elapsed:.1f} seconds")
                    
                    # Check success
                    if result.get('success'):
                        print(f"\n[{time.strftime('%H:%M:%S')}] ✅ SUCCESS! Command executed after auto-unlock")
                    
                    # Verify the flow worked correctly
                    response_text = result.get('response', '').lower()
                    if "unlocked your screen" in response_text and "safari" in response_text:
                        print(f"\n{'='*70}")
                        print("🎉 DEMONSTRATION SUCCESSFUL!")
                        print("="*70)
                        print("The Context Intelligence System successfully:")
                        print("✅ Detected the screen was locked")
                        print("✅ Queued the command")
                        print("✅ Automatically unlocked the screen") 
                        print("✅ Executed the queued command")
                        print("✅ Provided proper feedback throughout")
                        print("\nThis matches the PRD example scenario perfectly!")
                    else:
                        print(f"\n⚠️  Unexpected response - check if unlock worked")
                else:
                    print(f"\n❌ HTTP error: {response.status}")
                    body = await response.text()
                    print(f"Response: {body}")
                    
    except asyncio.TimeoutError:
        print(f"\n❌ Request timed out - unlock might be taking longer")
    except Exception as e:
        print(f"\n❌ Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Run the demonstration"""
    # Wait for Ironcliw
    if not await wait_for_jarvis():
        print("❌ Ironcliw failed to start")
        sys.exit(1)
    
    # Run demo
    await demo_automated_flow()
    
    print("\n" + "="*70)
    print("Demo complete. Your Mac should have:")
    print("1. Locked")
    print("2. Automatically unlocked")
    print("3. Opened Safari and searched for 'dogs'")
    print("All without any manual intervention!")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())