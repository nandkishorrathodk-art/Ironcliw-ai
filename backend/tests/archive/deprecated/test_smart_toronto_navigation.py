#!/usr/bin/env python3
"""Test smart Toronto navigation"""

import asyncio
import os

async def test_smart_navigation():
    """Test the smart navigation approach"""
    print("🧭 Testing Smart Toronto Navigation")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    from system_control.weather_location_forcer import WeatherSmartNavigator
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    
    navigator = WeatherSmartNavigator(controller, vision)
    
    print("\nAttempting smart navigation to Toronto...")
    success = await navigator.navigate_to_toronto()
    
    if success:
        print("\n✅ Successfully navigated to Toronto!")
        
        # Now test with Ironcliw
        print("\nTesting with Ironcliw...")
        from api.jarvis_voice_api import IroncliwVoiceAPI, IroncliwCommand
        from api.jarvis_factory import set_app_state
        from system_control.weather_system_config import initialize_weather_system
        from types import SimpleNamespace
        
        weather_system = initialize_weather_system(vision, controller)
        app_state = SimpleNamespace(
            vision_analyzer=vision,
            weather_system=weather_system
        )
        set_app_state(app_state)
        
        jarvis_api = IroncliwVoiceAPI()
        
        command = IroncliwCommand(text="What's the weather for today?")
        result = await jarvis_api.process_command(command)
        
        response = result.get('response', '')
        print(f"\nIroncliw says: {response}")
        
        if 'toronto' in response.lower():
            print("\n🎉 SUCCESS! Ironcliw is reading Toronto weather!")
        else:
            print("\n⚠️ Ironcliw is not reading Toronto weather yet")
    else:
        print("\n❌ Failed to navigate to Toronto")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/Ironcliw-AI-Agent/backend')
    asyncio.run(test_smart_navigation())