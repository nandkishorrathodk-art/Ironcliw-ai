#!/usr/bin/env python3
"""Test Ironcliw commands directly"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.voice.jarvis_agent_voice import IroncliwAgentVoice

async def test_commands():
    """Test various Ironcliw commands"""
    print("Initializing Ironcliw Agent...")
    
    jarvis = IroncliwAgentVoice()
    jarvis.running = True  # Activate Ironcliw
    
    # Test commands
    test_cases = [
        "open chrome",
        "set volume to 50%",
        "take a screenshot",
        "close safari",
        "list open applications",
    ]
    
    for command in test_cases:
        print(f"\n{'='*50}")
        print(f"Command: {command}")
        try:
            response = await jarvis.process_voice_input(command)
            print(f"Response: {response}")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_commands())