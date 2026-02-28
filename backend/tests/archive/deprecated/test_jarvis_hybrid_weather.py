#!/usr/bin/env python3
"""Test Ironcliw with Hybrid Weather Integration"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add backend to path
sys.path.insert(0, '/Users/derekjrussell/Documents/repos/Ironcliw-AI-Agent/backend')

async def test_jarvis_weather():
    """Test Ironcliw weather with hybrid approach"""
    print("🤖 Testing Ironcliw with Hybrid Weather")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    
    # Import Ironcliw components
    from api.jarvis_voice_api import IroncliwVoiceAPI, IroncliwCommand
    from api.jarvis_factory import set_app_state
    from types import SimpleNamespace
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    
    # Initialize app state with vision
    vision_analyzer = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    app_state = SimpleNamespace(
        vision_analyzer=vision_analyzer
    )
    set_app_state(app_state)
    
    # Initialize Ironcliw
    jarvis_api = IroncliwVoiceAPI()
    
    # Test weather queries
    test_queries = [
        "What's the weather for today?",
        "What's the temperature?",
        "Is it going to rain?",
        "What's the weather in New York?",
        "What's the weather in Toronto?"
    ]
    
    for query in test_queries:
        print(f"\n🎙️  You: {query}")
        
        try:
            # Process command
            command = IroncliwCommand(text=query)
            result = await jarvis_api.process_command(command)
            
            response = result.get('response', '')
            print(f"🤖 Ironcliw: {response}")
            
            # Check source
            if "vision" in response.lower():
                print("   📍 Source: Vision-based extraction")
            elif "api" in response.lower():
                print("   📍 Source: OpenWeatherMap API")
            elif "core location" in response.lower():
                print("   📍 Source: Core Location")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/Ironcliw-AI-Agent/backend')
    asyncio.run(test_jarvis_weather())