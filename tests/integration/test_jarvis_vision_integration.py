#!/usr/bin/env python3
"""
Test Ironcliw vision integration
"""

import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from backend.voice.jarvis_agent_voice import IroncliwAgentVoice


async def test_vision_commands():
    """Test vision commands through Ironcliw"""
    print("🤖 Testing Ironcliw Vision Integration\n")
    
    # Initialize Ironcliw
    jarvis = IroncliwAgentVoice(user_name="Sir")
    
    # Check if vision is enabled
    print(f"Vision enabled: {jarvis.vision_enabled}")
    
    if not jarvis.vision_enabled:
        print("❌ Vision system is not enabled!")
        return
    
    # Test commands
    test_commands = [
        "can you see my screen",
        "analyze what's on my screen",
        "what's on my screen",
        "check for updates",
        "look at my screen"
    ]
    
    for command in test_commands:
        print(f"\n🎤 Command: '{command}'")
        try:
            response = await jarvis._handle_vision_command(command)
            print(f"🤖 Ironcliw: {response}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    asyncio.run(test_vision_commands())