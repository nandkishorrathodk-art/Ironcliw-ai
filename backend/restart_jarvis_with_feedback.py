#!/usr/bin/env python3
"""
Restart Ironcliw with Voice Feedback Fix
======================================
"""

import subprocess
import time
import os
import signal
import sys

def restart_jarvis():
    """Restart Ironcliw to pick up the voice feedback changes"""
    
    print("🔄 Restarting Ironcliw with Voice Feedback Fix")
    print("="*50)
    
    # Find Ironcliw process
    print("\n1️⃣ Finding Ironcliw process...")
    result = subprocess.run(
        ["pgrep", "-f", "python main.py"],
        capture_output=True, text=True
    )
    
    if result.stdout.strip():
        pid = int(result.stdout.strip())
        print(f"   Found Ironcliw running with PID: {pid}")
        
        # Kill the process
        print("\n2️⃣ Stopping Ironcliw...")
        try:
            os.kill(pid, signal.SIGTERM)
            print("   Sent termination signal")
            time.sleep(2)
            
            # Check if still running
            try:
                os.kill(pid, 0)
                print("   Process still running, sending SIGKILL...")
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                print("   ✅ Ironcliw stopped")
                
        except Exception as e:
            print(f"   ❌ Error stopping Ironcliw: {e}")
    else:
        print("   ℹ️  Ironcliw not currently running")
    
    # Wait a moment
    time.sleep(2)
    
    # Start Ironcliw
    print("\n3️⃣ Starting Ironcliw with updated code...")
    
    # Change to backend directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Start Ironcliw in background
    process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True
    )
    
    print(f"   Started Ironcliw with PID: {process.pid}")
    
    # Wait for it to initialize
    print("\n4️⃣ Waiting for Ironcliw to initialize...")
    for i in range(10):
        print(f"   {10-i}...", end='\r')
        time.sleep(1)
    
    print("\n\n✅ Ironcliw restarted with voice feedback fix!")
    print("\n📢 What's fixed:")
    print("   - Ironcliw will now speak: 'I see your screen is locked...'")
    print("   - Before unlocking, not after")
    print("   - Clear feedback throughout the process")
    
    print("\n🎤 To test:")
    print("   1. Lock your screen")
    print("   2. Say: 'Ironcliw, open Safari and search for dogs'")
    print("   3. Listen for the lock detection announcement")

if __name__ == "__main__":
    restart_jarvis()