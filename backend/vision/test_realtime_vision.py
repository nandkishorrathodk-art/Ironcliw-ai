#!/usr/bin/env python3
"""
Test Ironcliw real-time vision capabilities
Demonstrates how Ironcliw can see and respond to the screen in real-time
"""

import asyncio
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_realtime_vision():
    """Test real-time vision features"""
    from claude_vision_analyzer import ClaudeVisionAnalyzer
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("Please set ANTHROPIC_API_KEY")
        return
    
    # Initialize Ironcliw vision
    jarvis = ClaudeVisionAnalyzer(api_key, enable_realtime=True)
    logger.info("🤖 Ironcliw Vision System Initialized")
    
    print("\n" + "="*60)
    print("🎯 Ironcliw Real-Time Vision Test")
    print("="*60)
    
    # Test 1: Basic screen understanding
    print("\n📊 Test 1: What can Ironcliw see right now?")
    context = await jarvis.get_screen_context()
    
    if 'error' not in context:
        print(f"\n👁️ Ironcliw sees: {context.get('description', 'Nothing')[:200]}...")
        
        if context.get('behavior_insights'):
            print(f"\n🧠 Detected patterns: {context['behavior_insights']['detected_patterns']}")
            print(f"💡 Suggested actions: {[a['description'] for a in context['behavior_insights']['suggested_actions']]}")
    else:
        print(f"❌ Error: {context['error']}")
    
    # Test 2: Command with visual context
    print("\n\n📊 Test 2: Responding to commands with visual awareness")
    test_commands = [
        "What application is open?",
        "Are there any notifications?",
        "What can I click on?",
        "Is there any text I should read?"
    ]
    
    for cmd in test_commands[:2]:  # Test first 2 commands
        print(f"\n👤 User: {cmd}")
        response = await jarvis.see_and_respond(cmd)
        
        if response['success']:
            print(f"🤖 Ironcliw: {response['response'][:200]}...")
        else:
            print(f"❌ Error: {response.get('error', 'Unknown error')}")
    
    # Test 3: Real-time monitoring
    print("\n\n📊 Test 3: Real-time monitoring (10 seconds)")
    print("Try opening/closing windows, receiving notifications, etc.")
    
    # Event counter
    event_count = 0
    
    async def vision_event_handler(event):
        nonlocal event_count
        event_count += 1
        print(f"\n🔔 Event {event_count}: Change detected at {event.get('timestamp', 'unknown')}")
        print(f"   Description: {event.get('description', 'No description')[:100]}...")
    
    # Start real-time vision
    result = await jarvis.start_jarvis_vision(vision_event_handler)
    
    if result['success']:
        print(f"✅ Real-time vision started in {result['mode']} mode")
        print("👀 Watching for changes...")
        
        # Monitor for 10 seconds
        await asyncio.sleep(10)
        
        # Stop monitoring
        await jarvis.stop_jarvis_vision()
        print(f"\n✅ Monitoring stopped. Detected {event_count} events.")
    else:
        print(f"❌ Failed to start real-time vision: {result.get('error')}")
    
    # Test 4: Notification monitoring
    print("\n\n📊 Test 4: Checking for notifications (5 seconds)")
    notifications = await jarvis.monitor_for_notifications(duration=5.0)
    
    if notifications:
        print(f"📬 Found {len(notifications)} notifications:")
        for notif in notifications:
            print(f"  - {notif['description'][:100]}...")
    else:
        print("📭 No notifications detected")
    
    # Cleanup
    await jarvis.cleanup_all_components()
    print("\n✅ Test complete!")

async def demo_autonomous_behaviors():
    """Demonstrate autonomous behavior detection"""
    from claude_vision_analyzer import ClaudeVisionAnalyzer
    
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        return
    
    jarvis = ClaudeVisionAnalyzer(api_key)
    
    print("\n" + "="*60)
    print("🤖 Ironcliw Autonomous Behavior Demo")
    print("="*60)
    
    # Get current context
    context = await jarvis.get_real_time_context()
    
    if context.get('behavior_insights'):
        insights = context['behavior_insights']
        
        print(f"\n🔍 Detected patterns: {insights['detected_patterns']}")
        
        # Handle each suggested action
        for action in insights['suggested_actions']:
            action_type = action['type']
            print(f"\n🎯 Handling: {action['description']}")
            
            # Handle the behavior
            result = await jarvis.handle_autonomous_behavior(action_type, context)
            
            if result['success']:
                print(f"✅ Action completed: {action_type}")
                if 'content' in result:
                    print(f"   Content: {result['content'].get('description', '')[:100]}...")
            else:
                print(f"❌ Action failed: {result.get('error')}")
    else:
        print("\n😴 No behaviors detected on current screen")
    
    await jarvis.cleanup_all_components()

async def main():
    """Run all tests"""
    print("🚀 Ironcliw Vision System Test Suite")
    print("==================================")
    
    # Run basic tests
    await test_realtime_vision()
    
    # Run autonomous behavior demo
    await demo_autonomous_behaviors()
    
    print("\n✨ All tests complete!")
    print("\n💡 Ironcliw can now:")
    print("  - See your screen in real-time")
    print("  - Understand visual context")
    print("  - Respond to commands based on what's visible")
    print("  - Detect and handle notifications")
    print("  - Suggest autonomous actions")

if __name__ == "__main__":
    asyncio.run(main())