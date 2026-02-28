#!/usr/bin/env python3
"""Test optimized weather response"""

import asyncio
import os
import time

async def test_weather_optimized():
    """Test Ironcliw weather with optimizations"""
    print("🚀 Testing Optimized Weather Response")
    print("="*60)
    
    from api.jarvis_voice_api import IroncliwVoiceAPI, IroncliwCommand
    from api.jarvis_factory import set_app_state
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    from system_control.weather_system_config import initialize_weather_system
    from types import SimpleNamespace
    
    # Setup
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    weather_system = initialize_weather_system(vision, controller)
    
    app_state = SimpleNamespace(
        vision_analyzer=vision,
        weather_system=weather_system
    )
    set_app_state(app_state)
    
    jarvis_api = IroncliwVoiceAPI()
    
    # Test
    print("\nAsking Ironcliw: 'What's the weather for today?'")
    start_time = time.time()
    
    command = IroncliwCommand(text="What's the weather for today?")
    result = await jarvis_api.process_command(command)
    
    elapsed = time.time() - start_time
    
    response = result.get('response', '')
    print(f"\nIroncliw Response ({elapsed:.1f}s):")
    print(f"{response}")
    
    # Analysis
    print("\n" + "-"*60)
    print("Analysis:")
    
    if elapsed < 15:
        print(f"✅ Fast response time: {elapsed:.1f} seconds")
    elif elapsed < 25:
        print(f"⚠️  Moderate response time: {elapsed:.1f} seconds")
    else:
        print(f"❌ Slow response time: {elapsed:.1f} seconds")
    
    if 'toronto' in response.lower():
        print("✅ Successfully reading Toronto weather")
    elif 'new york' in response.lower():
        print("⚠️  Reading New York weather instead of Toronto")
    elif 'difficulty' in response.lower():
        print("❌ Failed to read weather")
    else:
        print("✅ Weather information provided")
    
    if 'currently' in response.lower() and '°' in response:
        print("✅ Proper formatting with temperature")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/Ironcliw-AI-Agent/backend')
    asyncio.run(test_weather_optimized())