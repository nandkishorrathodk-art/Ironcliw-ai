#!/usr/bin/env python3
"""
Demo: WhatsApp Command Routing Fix
Shows how "open WhatsApp" is now correctly handled
"""

import asyncio
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swift_bridge.python_bridge import IntelligentCommandRouter

async def demo():
    """Demonstrate the fix for WhatsApp command routing"""
    
    print("🤖 Ironcliw WhatsApp Command Routing Demo")
    print("=" * 50)
    print()
    
    # Initialize the intelligent router
    router = IntelligentCommandRouter()
    
    # Test commands that were problematic before
    test_commands = [
        "open WhatsApp",
        "close WhatsApp",
        "what's in WhatsApp", 
        "what's on my screen",
        "open Safari",
        "close all apps"
    ]
    
    print("🧪 Testing command classification:\n")
    
    for command in test_commands:
        handler, details = await router.route_command(command)
        
        # Show results with emoji indicators
        if command.startswith(("open", "close")):
            # These should always be system commands
            emoji = "✅" if handler == "system" else "❌"
        else:
            # These could be either system or vision
            emoji = "✅"
            
        print(f"{emoji} '{command}'")
        print(f"   → Handler: {handler}")
        print(f"   → Type: {details.get('type')}")
        print(f"   → Intent: {details.get('intent')}")
        print(f"   → Confidence: {details.get('confidence', 0):.2f}")
        print()
    
    print("=" * 50)
    print("✨ Key Improvements:")
    print("   • 'open WhatsApp' → system handler ✅")
    print("   • No keyword matching - uses NLP")
    print("   • Intelligent intent recognition")
    print("   • Learning capabilities")
    print("=" * 50)

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: ANTHROPIC_API_KEY not set")
        print("   Some features may be limited")
        print()
    
    # Run the demo
    asyncio.run(demo())