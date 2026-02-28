#!/usr/bin/env python3
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


Test Ironcliw closing apps functionality
"""

import asyncio
from backend.voice.jarvis_agent_voice import IroncliwAgentVoice


async def test_close_apps():
    """Test closing apps through Ironcliw"""
    print("🤖 Testing Ironcliw App Closing\n")
    
    # Initialize Ironcliw
    jarvis = IroncliwAgentVoice(user_name="Sir")
    
    if not jarvis.system_control_enabled:
        print("❌ System control is not enabled. Please set ANTHROPIC_API_KEY.")
        return
    
    # Test commands
    test_commands = [
        "close whatsapp",
        "close preview", 
        "close whatsapp and preview",
        "close safari and notes",
        "list open applications"
    ]
    
    for command in test_commands:
        print(f"\n🎤 Command: '{command}'")
        
        try:
            # Process as system command
            response = await jarvis._handle_system_command(command)
            print(f"🤖 Ironcliw: {response}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Wait a bit between commands
        await asyncio.sleep(2)


async def test_direct_controller():
    """Test the controller directly"""
    print("\n\n🔧 Testing Direct Controller\n")
    
    from backend.system_control.macos_controller import MacOSController
    controller = MacOSController()
    
    # Test closing WhatsApp specifically
    print("Testing WhatsApp close...")
    success, message = controller.close_application("WhatsApp")
    print(f"Result: {success} - {message}")
    
    # Test closing Preview
    print("\nTesting Preview close...")
    success, message = controller.close_application("Preview")
    print(f"Result: {success} - {message}")
    
    # List open apps
    print("\nOpen applications:")
    apps = controller.list_open_applications()
    for app in apps:
        print(f"  - {app}")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Ironcliw App Closing Functionality")
    print("=" * 60)
    
    # Test through Ironcliw
    await test_close_apps()
    
    # Test controller directly
    await test_direct_controller()
    
    print("\n✅ Testing complete!")


if __name__ == "__main__":
    asyncio.run(main())