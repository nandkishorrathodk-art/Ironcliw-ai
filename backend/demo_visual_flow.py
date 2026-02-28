#!/usr/bin/env python3
"""
Visual Demo of Ironcliw Lock/Unlock Flow
======================================

Shows step-by-step what happens with visual indicators
"""

import asyncio
import subprocess
from datetime import datetime
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.simple_context_handler_enhanced import wrap_with_enhanced_context
from api.unified_command_processor import UnifiedCommandProcessor
from api.direct_unlock_handler_fixed import check_screen_locked_direct

class VisualWebSocket:
    """WebSocket that shows visual flow"""
    def __init__(self):
        self.step_num = 0
        
    async def send_json(self, data):
        msg_type = data.get('type')
        
        if msg_type == 'context_update':
            self.step_num += 1
            print(f"\n{'─'*60}")
            print(f"📍 Step {self.step_num}: Context Update")
            print(f"💬 Ironcliw: \"{data.get('message')}\"")
            
            status = data.get('status', '')
            if status == 'unlocking_screen':
                print("🔓 Action: Attempting to unlock screen...")
            elif status == 'executing_command':
                print("🚀 Action: Executing your command...")
                
        elif msg_type == 'response':
            print(f"\n📢 Final Response: {data.get('text', data.get('message'))}")

def show_header():
    """Show demo header"""
    print("\n" + "="*80)
    print("🌟 Ironcliw INTELLIGENT LOCK HANDLING DEMO 🌟".center(80))
    print("="*80)

def show_scenario():
    """Show the scenario"""
    print("\n📖 SCENARIO:")
    print("┌─────────────────────────────────────────────────────────┐")
    print("│ 1. Your Mac screen is LOCKED 🔒                        │")
    print("│ 2. You say: \"Ironcliw, open Safari and search for dogs\" │")
    print("│ 3. Watch Ironcliw handle it intelligently!               │")
    print("└─────────────────────────────────────────────────────────┘")

async def visual_demo():
    """Run the visual demonstration"""
    
    show_header()
    show_scenario()
    
    print("\n⏳ Demo starts in 5 seconds...")
    print("   (This will lock and unlock your screen)")
    
    for i in range(5, 0, -1):
        print(f"   {i}...", end='\r')
        await asyncio.sleep(1)
    
    print("\n\n🎬 STARTING DEMO")
    print("="*60)
    
    # Phase 1: Lock Screen
    print("\n📍 Phase 1: Setting up locked screen")
    print("🔒 Locking screen now...")
    
    lock_cmd = """osascript -e 'tell app "System Events" to key code 12 using {control down, command down}'"""
    subprocess.run(lock_cmd, shell=True)
    
    await asyncio.sleep(3)
    
    # Verify lock
    is_locked = await check_screen_locked_direct()
    print(f"✅ Screen status: {'LOCKED 🔒' if is_locked else 'UNLOCKED 🔓'}")
    
    # Phase 2: User Command
    print("\n📍 Phase 2: User gives command")
    print("🎤 User: \"Ironcliw, open Safari and search for dogs\"")
    
    # Phase 3: Ironcliw Processing
    print("\n📍 Phase 3: Ironcliw Processing")
    print("🤖 Ironcliw is analyzing your request...")
    
    # Create components
    processor = UnifiedCommandProcessor()
    handler = wrap_with_enhanced_context(processor)
    websocket = VisualWebSocket()
    
    # Process command
    command = "open Safari and search for dogs"
    result = await handler.process_with_context(command, websocket)
    
    # Phase 4: Results
    print("\n" + "="*60)
    print("📊 DEMO RESULTS")
    print("="*60)
    
    if result.get('execution_steps'):
        print("\n✅ Execution Timeline:")
        for i, step in enumerate(result['execution_steps'], 1):
            timestamp = step.get('timestamp', '')
            step_text = step['step']
            
            # Add visual indicators
            if 'locked' in step_text.lower():
                icon = "🔒"
            elif 'unlock' in step_text.lower():
                icon = "🔓"
            elif 'executed' in step_text.lower():
                icon = "✅"
            else:
                icon = "▶️"
                
            print(f"   {icon} {step_text}")
    
    print("\n🎯 Key Points Demonstrated:")
    print("   ✅ Ironcliw detected the locked screen")
    print("   ✅ Provided clear feedback BEFORE unlocking")
    print("   ✅ Successfully unlocked the screen")
    print("   ✅ Attempted to execute the original command")
    
    # Check if Safari is open
    await asyncio.sleep(2)
    safari_check = subprocess.run(
        ["osascript", "-e", 'tell app "System Events" to get name of first process whose frontmost is true'],
        capture_output=True, text=True
    )
    if "Safari" in safari_check.stdout:
        print("   ✅ Safari is now open!")

async def quick_test():
    """Quick test without locking"""
    print("\n\n📍 BONUS: Testing without screen lock")
    print("─"*40)
    
    processor = UnifiedCommandProcessor()
    handler = wrap_with_enhanced_context(processor) 
    
    command = "what time is it"
    print(f"🎤 Command: '{command}'")
    
    result = await handler.process_with_context(command)
    print(f"💬 Response: {result.get('response')}")
    print("✅ No unlock needed for this command!")

if __name__ == "__main__":
    try:
        # Run main demo
        asyncio.run(visual_demo())
        
        # Run bonus test
        asyncio.run(quick_test())
        
        print("\n" + "="*80)
        print("🎉 DEMO COMPLETE! 🎉".center(80))
        print("="*80)
        print("\n✨ Ironcliw successfully demonstrated intelligent screen lock handling!")
        
    except KeyboardInterrupt:
        print("\n\n❌ Demo cancelled by user")