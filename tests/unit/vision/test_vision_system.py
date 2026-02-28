#!/usr/bin/env python3
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


Test script for Ironcliw Vision System
"""

import asyncio
import logging
import os
from backend.vision.screen_vision import ScreenVisionSystem, IroncliwVisionIntegration
from backend.vision.claude_vision_analyzer import ClaudeVisionAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_vision():
    """Test basic vision capabilities"""
    print("\n🔍 Testing Basic Vision System...")
    
    vision_system = ScreenVisionSystem()
    
    # Test screen capture
    print("📸 Capturing screen...")
    screenshot = await vision_system.capture_screen()
    print(f"✅ Screen captured: {screenshot.shape}")
    
    # Test text detection
    print("\n📝 Detecting text on screen...")
    text_elements = await vision_system.detect_text_regions(screenshot)
    print(f"✅ Found {len(text_elements)} text regions")
    for i, elem in enumerate(text_elements[:5]):  # Show first 5
        print(f"  {i+1}. '{elem.text}' at {elem.location}")
    
    # Test update detection
    print("\n🔄 Checking for software updates...")
    updates = await vision_system.analyze_for_updates(screenshot)
    if updates:
        print(f"✅ Found {len(updates)} potential updates:")
        for update in updates:
            print(f"  - {update.application}: {update.description} ({update.urgency})")
    else:
        print("ℹ️  No updates detected")
    
    # Test screen context
    print("\n🖥️  Getting screen context...")
    context = await vision_system.get_screen_context()
    print(f"✅ Screen context:")
    print(f"  - Text elements: {len(context['text_elements'])}")
    print(f"  - UI elements: {len(context['ui_elements'])}")
    print(f"  - Detected apps: {', '.join(context['detected_apps']) if context['detected_apps'] else 'None'}")


async def test_jarvis_integration():
    """Test Ironcliw voice integration"""
    print("\n🤖 Testing Ironcliw Integration...")
    
    vision_system = ScreenVisionSystem()
    jarvis_vision = IroncliwVisionIntegration(vision_system)
    
    # Test various voice commands
    commands = [
        "What's on my screen?",
        "Check for software updates",
        "What applications are open?",
        "Analyze my screen"
    ]
    
    for command in commands:
        print(f"\n💬 Command: '{command}'")
        response = await jarvis_vision.handle_vision_command(command)
        print(f"🗣️  Ironcliw: {response}")


async def test_claude_vision():
    """Test Claude vision analysis (requires API key)"""
    print("\n🧠 Testing Claude Vision Analysis...")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  Skipping Claude vision test - no API key found")
        return
    
    claude_analyzer = ClaudeVisionAnalyzer(api_key)
    vision_system = ScreenVisionSystem()
    
    # Capture screen
    screenshot = await vision_system.capture_screen()
    
    # Test update detection
    print("\n🔄 Using Claude to check for updates...")
    result = await claude_analyzer.check_for_software_updates(screenshot)
    print(f"✅ Claude analysis: {result}")
    
    # Test activity understanding
    print("\n👀 Understanding user activity...")
    activity = await claude_analyzer.understand_user_activity(screenshot)
    print(f"✅ Activity analysis: {activity.get('description', 'No description')}")


async def test_update_monitoring():
    """Test continuous update monitoring"""
    print("\n📊 Testing Update Monitoring...")
    
    vision_system = ScreenVisionSystem()
    
    print("Starting 30-second monitoring test...")
    print("(This will check for updates every 10 seconds)")
    
    check_count = 0
    
    async def update_callback(updates):
        nonlocal check_count
        check_count += 1
        print(f"\n⏰ Check #{check_count}: Found {len(updates)} updates")
        for update in updates:
            print(f"  - {update.application}: {update.description}")
    
    # Run for 30 seconds with 10-second intervals
    monitoring_task = asyncio.create_task(
        vision_system.monitor_screen_continuously(update_callback, interval=10)
    )
    
    await asyncio.sleep(30)
    monitoring_task.cancel()
    
    print(f"\n✅ Monitoring complete. Performed {check_count} checks.")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🤖 Ironcliw Vision System Test Suite")
    print("=" * 60)
    
    try:
        # Test basic vision
        await test_basic_vision()
        
        # Test Ironcliw integration
        await test_jarvis_integration()
        
        # Test Claude vision (if API key available)
        await test_claude_vision()
        
        # Test monitoring (optional - takes 30 seconds)
        print("\n❓ Run 30-second monitoring test? (y/n): ", end="")
        if input().lower() == 'y':
            await test_update_monitoring()
        
        print("\n✅ All tests completed!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())