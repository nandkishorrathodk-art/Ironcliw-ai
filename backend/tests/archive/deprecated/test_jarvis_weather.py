#!/usr/bin/env python3
"""Test Ironcliw weather functionality end-to-end"""

import asyncio
import logging

# Set up minimal logging
logging.basicConfig(level=logging.WARNING)

async def test_jarvis_weather():
    """Test Ironcliw weather responses"""
    print("=== Testing Ironcliw Weather Response ===\n")
    
    # Import Ironcliw voice system
    from voice.jarvis_agent_voice import IroncliwAgentVoice
    
    # Create Ironcliw instance
    jarvis = IroncliwAgentVoice()
    
    # Test weather queries
    test_queries = [
        "what's the weather for today",
        "what's the temperature",
        "is it raining",
        "weather in Toronto",
        "how's the weather"
    ]
    
    print("Testing Ironcliw weather responses:")
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        try:
            # Simulate processing the query
            response = await jarvis._handle_weather_command(query.lower())
            print(f"   Response: {response}")
            
            # Check if response contains problematic values
            if "32°F" in response and "unknown conditions" in response:
                print("   ❌ ERROR: Got hardcoded fallback response!")
            elif "32°F" in response:
                print("   ⚠️  WARNING: Response contains 32°F - might be fallback")
            elif "unknown conditions" in response:
                print("   ⚠️  WARNING: Response contains 'unknown conditions'")
            else:
                print("   ✅ Response looks good!")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    asyncio.run(test_jarvis_weather())